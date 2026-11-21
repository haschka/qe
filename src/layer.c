/* The elementwise half of a step: norms, gates and stream mixers.

   A per-token byte budget says what the memory controller carries; it says
   nothing about the work the CPU does on the vectors the GEMVs hand back, and on
   this checkpoint that work is not proportional to weight bytes at all.
   Hyper-connections keep four streams of 2560 resident, normalise them twice per
   layer, gate the concatenation with a 10240-wide sigmoid, average the four
   streams into the block input, and add the block output back with four more
   sigmoids. Over 48 layers that is 96 grouped normalisations and 96 gates per
   token, plus 96 broadcast residual adds - and none of it appears in the 5568 MiB
   of docs/DESIGN.md §3, because none of it touches a weight matrix.

   Three facts decided the shape of this file.

   The sigmoid is the expensive one. A libm call per element is a correctly
   rounded exponential per element: 10240 of them per gate, 96 gates per token, so
   roughly a million transcendental calls sitting on the critical path of every
   layer. The SIMD path here is a degree-5 minimax exp2 plus one divide, which
   bench_kern holds against 1/(1+exp(-x)) and reports the worst error of; it is
   about a thousand times below the noise a Q8_0 dot product already introduces,
   and the test is what makes that sentence checkable rather than conventional.

   Norms are grouped, not flat. `hc_*_norm.weight` is 10240 wide because the four
   streams are concatenated, and the reduction is per stream of 2560: one flat
   mean-square over 10240 is a different model that runs perfectly and shows up
   only as perplexity. `ssm_norm` and `attn_q_norm` are the opposite shape - one
   128- or 256-entry gamma shared by 48 or 24 heads - so the gamma stride is an
   argument here, not an assumption. 0 means shared, which is what those two need.

   The mixers keep the vectors in registers. `qe_stream_gate_mean` reads the gate
   and the normalised streams once, computing sigmoid inline, and writes one
   2560-vector; the obvious version (sigmoid into scratch, then multiply and mean)
   touches 2.5x the memory for the same answer. That is the whole reason these
   three functions exist as fused primitives instead of being written inline at
   each of the 96 call sites.

   Nothing here is threaded: each call moves tens of KiB, and the pool's dispatch
   is a few µs, so a thread handoff per mixer would be the dominant cost. They
   belong inside whatever loop drives a layer, next to the GEMVs.
*/
#include "qe.h"

#include <math.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define LOG2E 1.4426950408889634f

/* exp2 for eight floats, ~1e-7 relative: round to an integer exponent, degree-5
 * minimax for the remaining [-0.5, 0.5], and put the integer into the exponent
 * field by addition. Clamped first because the exponent trick has no meaning
 * outside the float range and the only caller saturates anyway. */
static inline __m256 qe_exp2_avx2(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-100.0f)), _mm256_set1_ps(100.0f));
    __m256i n = _mm256_cvtps_epi32(x);
    __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(n));
    __m256 p = _mm256_set1_ps(1.3333558146428443e-3f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291076284778e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504108664815974e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022650695910071e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718055994531e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    /* p is in [0.707, 1.414], so adding to the exponent cannot carry into the
     * sign bit or denormalise; that is why f is reduced to [-0.5, 0.5] first. */
    return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p),
                                                _mm256_slli_epi32(n, 23)));
}

static inline __m256 qe_sigmoid_avx2(__m256 x) {
    __m256 e = qe_exp2_avx2(_mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), x),
                                          _mm256_set1_ps(LOG2E)));
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

static inline float qe_sigmoid1(float x) {
    /* expf saturates on its own: expf(1e30) is +inf and 1/(1+inf) is 0, so the
     * extremes need no special case, only a division that cannot trap. */
    return 1.0f / (1.0f + expf(-x));
}

void qe_sigmoid(const float *x, float *out, long n) {
    long i = 0;
#if defined(__x86_64__)
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, qe_sigmoid_avx2(_mm256_loadu_ps(x + i)));
#endif
    for (; i < n; i++)
        out[i] = qe_sigmoid1(x[i]);
}

void qe_rms_norm_grouped(const float *x, const float *restrict gamma, long gamma_stride,
                         float *out, long width, int n_groups, float eps) {
    const float inv_w = 1.0f / (float)width;
    for (int g = 0; g < n_groups; g++) {
        const float *xi = x + (size_t)g * (size_t)width;
        float *oi = out + (size_t)g * (size_t)width;
        const float *gi = gamma_stride > 0 ? gamma + (size_t)g * (size_t)gamma_stride : gamma;
        float ss = 0.0f;
        long j = 0;
#if defined(__x86_64__)
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (; j + 32 <= width; j += 32) {
            __m256 v0 = _mm256_loadu_ps(xi + j);
            __m256 v1 = _mm256_loadu_ps(xi + j + 8);
            __m256 v2 = _mm256_loadu_ps(xi + j + 16);
            __m256 v3 = _mm256_loadu_ps(xi + j + 24);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
            a1 = _mm256_fmadd_ps(v1, v1, a1);
            a2 = _mm256_fmadd_ps(v2, v2, a2);
            a3 = _mm256_fmadd_ps(v3, v3, a3);
        }
        for (; j + 8 <= width; j += 8) {
            __m256 v = _mm256_loadu_ps(xi + j);
            a0 = _mm256_fmadd_ps(v, v, a0);
        }
        __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        float t[8];
        _mm256_storeu_ps(t, s);
        ss = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#endif
        for (; j < width; j++)
            ss += xi[j] * xi[j];
        const float inv = 1.0f / sqrtf(ss * inv_w + eps);
        j = 0;
#if defined(__x86_64__)
        __m256 vi = _mm256_set1_ps(inv);
        for (; j + 8 <= width; j += 8) {
            __m256 v = _mm256_loadu_ps(xi + j);
            __m256 gm = _mm256_loadu_ps(gi + j);
            _mm256_storeu_ps(oi + j, _mm256_mul_ps(_mm256_mul_ps(v, vi), gm));
        }
#endif
        for (; j < width; j++)
            oi[j] = xi[j] * inv * gi[j];
    }
}

void qe_stream_gate_mean(const float *restrict xn, const float *restrict gate,
                         float *restrict mixed, long width, int n_streams) {
    long j = 0;
#if defined(__x86_64__)
    for (; j + 8 <= width; j += 8) {
        __m256 acc = _mm256_mul_ps(
            _mm256_loadu_ps(xn + j),
            qe_sigmoid_avx2(_mm256_loadu_ps(gate + j)));
        for (int s = 1; s < n_streams; s++) {
            long o = (long)s * width + j;
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(xn + o),
                                  qe_sigmoid_avx2(_mm256_loadu_ps(gate + o)), acc);
        }
        _mm256_storeu_ps(mixed + j,
                         _mm256_mul_ps(acc, _mm256_set1_ps(1.0f / (float)n_streams)));
    }
#endif
    for (; j < width; j++) {
        float acc = 0.0f;
        for (int s = 0; s < n_streams; s++) {
            long o = (long)s * width + j;
            acc += xn[o] * qe_sigmoid1(gate[o]);
        }
        mixed[j] = acc / (float)n_streams;
    }
}

void qe_stream_inject_add(float *restrict res, const float *restrict y,
                          const float *restrict inject, long width, int n_streams) {
    for (int s = 0; s < n_streams; s++) {
        /* 2*sigmoid centred on 1: a zero injection has to be an ordinary residual
         * add, which is the whole reason the reference scales it this way. */
        const float w = 2.0f * qe_sigmoid1(inject[s]);
        float *rs = res + (size_t)s * (size_t)width;
        long j = 0;
#if defined(__x86_64__)
        __m256 vw = _mm256_set1_ps(w);
        for (; j + 8 <= width; j += 8)
            _mm256_storeu_ps(rs + j, _mm256_fmadd_ps(vw, _mm256_loadu_ps(y + j),
                                                     _mm256_loadu_ps(rs + j)));
#endif
        for (; j < width; j++)
            rs[j] += w * y[j];
    }
}

/* ------------------------------------------------- block-driver primitives
 *
 * The pieces the layer drivers are made of. Each is a transcription of the
 * reference op it has to agree with, down to where epsilon goes and which axis
 * reduces, because those are the details that produce a working model rather
 * than a running one:
 *
 *   qe_softplus         log1p(exp(-|x|)) + max(x,0), the reference's softplus
 *                       verbatim. The algebraically prettier log1p(exp(x))
 *                       overflows above x=88 and silently returns inf at
 *                       exactly the input the GDN decay is sensitive to.
 *   qe_l2_norm_grouped  x / max(||x||_2, eps): the norm is NOT divided by the
 *                       width, so this is not a variant of RMSNorm and sharing
 *                       a code path with it would be a different model. The
 *                       epsilon is a floor here; qe_l2_norm_grouped_under puts
 *                       it under the root instead, which is where the delta-net
 *                       reference appears to have it. Both live because the two
 *                       readings differ only at ||x|| ~ 0 and only the loss on
 *                       real text can say which one this checkpoint wants.
 *   qe_rope_init/apply  partial rotary - 64 of the 256 head dims - with
 *                       split-half pairs (p, p + n_dims/2) and
 *                       theta_p = pos * base^(-2p/n_dims) accumulated by
 *                       repeated multiplication. This file has
 *                       rope.dimension_sections and no rope.type, which is the
 *                       m-rope mode, and its pair layout is split-half rather
 *                       than adjacent. Rotating adjacent pairs instead is
 *                       perfectly stable, changes every rotated dimension, and
 *                       is invisible until something measures perplexity.
 */

void qe_silu(const float *x, float *out, long n) {
    long i = 0;
#if defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, qe_sigmoid_avx2(v)));
    }
#endif
    for (; i < n; i++)
        out[i] = x[i] * qe_sigmoid1(x[i]);
}

void qe_silu_mul(float *h, const float *g, const float *u, long n) {
    long i = 0;
#if defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 gv = _mm256_loadu_ps(g + i);
        __m256 prod = _mm256_mul_ps(gv, qe_sigmoid_avx2(gv));
        _mm256_storeu_ps(h + i, _mm256_mul_ps(prod, _mm256_loadu_ps(u + i)));
    }
#endif
    for (; i < n; i++)
        h[i] = g[i] * qe_sigmoid1(g[i]) * u[i];
}

void qe_softplus(const float *x, float *out, long n) {
    for (long i = 0; i < n; i++)
        out[i] = fmaxf(x[i], 0.0f) + log1pf(expf(-fabsf(x[i])));
}

static void l2_norm_grouped(const float *x, float *out, long width, int n_groups, float eps,
                            int under_root) {
    for (int g = 0; g < n_groups; g++) {
        const float *xi = x + (size_t)g * (size_t)width;
        float *oi = out + (size_t)g * (size_t)width;
        /* Accumulated in double: this norm divides by the norm itself, so a few
         * ulps of drift here is a relative error on every element of the head,
         * and the reference accumulates in double too. */
        double ss = 0.0;
        for (long j = 0; j < width; j++)
            ss += (double)xi[j] * (double)xi[j];
        const float inv =
            under_root ? 1.0f / sqrtf((float)(ss + (double)eps))
                       : 1.0f / fmaxf((float)sqrt(ss), eps);
        long j = 0;
#if defined(__x86_64__)
        __m256 vi = _mm256_set1_ps(inv);
        for (; j + 8 <= width; j += 8)
            _mm256_storeu_ps(oi + j, _mm256_mul_ps(_mm256_loadu_ps(xi + j), vi));
#endif
        for (; j < width; j++)
            oi[j] = xi[j] * inv;
    }
}

void qe_l2_norm_grouped(const float *x, float *out, long width, int n_groups, float eps) {
    l2_norm_grouped(x, out, width, n_groups, eps, 0);
}

void qe_l2_norm_grouped_under(const float *x, float *out, long width, int n_groups, float eps) {
    l2_norm_grouped(x, out, width, n_groups, eps, 1);
}

void qe_rope_init(float *cs, long n_dims, long pos, float theta_base) {
    const float step = powf(theta_base, -2.0f / (float)n_dims);
    float theta = (float)pos;
    for (long p = 0; p < n_dims / 2; p++) {
        cs[2 * p + 1] = sinf(theta);
        cs[2 * p] = cosf(theta);
        theta *= step;
    }
}

void qe_rope_apply(float *restrict v, long n_heads, long stride, long n_dims,
                   const float *restrict cs) {
    const long half = n_dims / 2;
    for (long h = 0; h < n_heads; h++) {
        float *x = v + (size_t)h * (size_t)stride;
        for (long p = 0; p < half; p++) {
            const float c = cs[2 * p], s = cs[2 * p + 1];
            const float x0 = x[p], x1 = x[p + half];
            x[p] = x0 * c - x1 * s;
            x[p + half] = x0 * s + x1 * c;
        }
    }
}

float qe_dot(const float *restrict a, const float *restrict b, long n) {
    long i = 0;
    float r;
#if defined(__x86_64__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    float t[8];
    _mm256_storeu_ps(t, s);
    r = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#else
    double d = 0.0;
    for (long j = 0; j < n; j++)
        d += (double)a[j] * (double)b[j];
    r = (float)d;
    i = n;
#endif
    for (; i < n; i++)
        r += a[i] * b[i];
    return r;
}

void qe_axpy(float *restrict y, const float *restrict x, float a, long n) {
    long i = 0;
#if defined(__x86_64__)
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),
                                                _mm256_loadu_ps(y + i)));
#endif
    for (; i < n; i++)
        y[i] += a * x[i];
}

/* One absolute maximum for the row and one byte per value: 8.25 bits per value
 * with the scale, against 32. The scale is per row and a row is one head of one
 * token, which is what keeps a key's direction: the head is already RMS
 * normalised before it is cached, so every row has roughly the same widest value
 * and there is no outlier to fight over the scale. */
void qe_quant_q8_row(const float *src, int8_t *dst, float *scale, long n) {
    float amax = 0.0f;
    for (long i = 0; i < n; i++) {
        float v = src[i];
        float a = v < 0.0f ? -v : v;
        if (a > amax)
            amax = a; /* a NaN fails this, and a row of NaN becomes zeros */
    }
    if (!(amax > 0.0f)) {
        *scale = 0.0f;
        memset(dst, 0, (size_t)n);
        return;
    }
    const float s = amax / 127.0f;
    const float inv = 1.0f / s;
    *scale = s;
    for (long i = 0; i < n; i++) {
        float v = src[i] * inv;
        if (v > 127.0f)
            v = 127.0f;
        else if (v < -127.0f)
            v = -127.0f;
        dst[i] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    }
}

/* Eight quantised values of a cached row, widened to the eight floats one FMA
 * wants. One sign-extending move and one convert: the values are small integers,
 * so the conversion is exact and the only approximation already happened when
 * the row was written. */
#if defined(__x86_64__)
static inline __m256 q_i8x8(const int8_t *q) {
    __m128i b = _mm_loadl_epi64((const __m128i *)q);
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
}
#endif

float qe_dot_q8(const float *restrict a, const int8_t *restrict q, long n) {
    long i = 0;
    float r;
#if defined(__x86_64__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        __m256 f0 = q_i8x8(q + i);
        __m256 f1 = q_i8x8(q + i + 8);
        __m256 f2 = q_i8x8(q + i + 16);
        __m256 f3 = q_i8x8(q + i + 24);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), f0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), f1, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), f2, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), f3, a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), q_i8x8(q + i), a0);
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    float t[8];
    _mm256_storeu_ps(t, s);
    r = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#else
    double d = 0.0;
    for (long j = 0; j < n; j++)
        d += (double)a[j] * (double)q[j];
    r = (float)d;
    i = n;
#endif
    for (; i < n; i++)
        r += a[i] * (float)q[i];
    return r;
}

void qe_axpy_q8(float *restrict y, const int8_t *restrict q, float a, long n) {
    long i = 0;
#if defined(__x86_64__)
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, q_i8x8(q + i), _mm256_loadu_ps(y + i)));
#endif
    for (; i < n; i++)
        y[i] += a * (float)q[i];
}

/* ------------------------------------------------------------------ attention
 *
 * Two passes over a selected cell - score it, then accumulate its value - are the
 * same reads whether the softmax lives in a score array or in a register, so what
 * this buys is not fewer bytes per head. It is fewer walks: one pass over the
 * selection serving `per` query heads instead of `per` passes serving one each,
 * which is what a KV group shared twelve ways asks for. The key row and the value
 * row of a cell are read once and stay in L1 across the twelve heads that want
 * them, and nothing is written to memory between them.
 *
 * One exponent base is carried per head and the accumulator behind it is rescaled
 * whenever a score rises more than QE_ATTN_RESCALE_NATS above it. That is the only
 * reason the exponentials cannot overflow: a weight is exp of something at most the
 * threshold above the base, so a partial sums to nsel * exp(threshold) rather than
 * to whatever the context happens to contain, and the base is carried out to the
 * combine, which is what makes two chunks of one selection addable at all.
 *
 * Rescaling on every new maximum - the textbook form - is the wrong trigger here.
 * The selection arrives in address order, which is position order, and a context's
 * attention weights are biased towards its end, so a run of records is the expected
 * case rather than a pathological one, and each record would cost a pass over the
 * accumulator. The threshold bounds that pass by the score range over it, which is
 * a few. */

/* How far a score may run ahead of its head's exponent base before the accumulator
 * behind it is pulled up. Sixteen nats puts a partial's sum at about 7e10 for the
 * longest selection this indexer can ask for, nowhere near a float's range, and it
 * puts a rescale at roughly one per chunk. */
#define QE_ATTN_RESCALE_NATS 16.0f

static inline void qe_scale_row(float *restrict y, float a, long n) {
    long i = 0;
#if defined(__x86_64__)
    const __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, _mm256_loadu_ps(y + i)));
#endif
    for (; i < n; i++)
        y[i] *= a;
}

void qe_attn_walk(const qe_attn_unit *u) {
    const long hd = u->hd;
    for (int h = 0; h < u->per; h++) {
        u->m[h] = -1e30f; /* the floor the score loop has always carried */
        u->l[h] = 0.0f;
        memset(u->y + (size_t)h * hd, 0, (size_t)hd * sizeof(float));
    }
    for (int i = 0; i < u->nsel; i++) {
        const long cell = u->sel[i];
        const int8_t *kq = NULL, *vq = NULL;
        const float *kf = NULL, *vf = NULL;
        float ks = 1.0f, vs = 1.0f;
        if (u->q8) {
            kq = u->kd + (size_t)cell * u->krow + (size_t)u->kv_head * hd;
            vq = u->vd + (size_t)cell * u->krow + (size_t)u->kv_head * hd;
            ks = u->ks[(size_t)cell * u->sstride + u->kv_head];
            vs = u->vs[(size_t)cell * u->sstride + u->kv_head];
        } else {
            kf = u->kc + (size_t)cell * (long)u->krow + (size_t)u->kv_head * hd;
            vf = u->vc + (size_t)cell * (long)u->krow + (size_t)u->kv_head * hd;
        }
        for (int h = 0; h < u->per; h++) {
            const float *q = u->q + (size_t)h * hd;
            float *y = u->y + (size_t)h * hd;
            float s = (u->q8 ? qe_dot_q8(q, kq, hd) : qe_dot(q, kf, hd)) * ks;
            s *= u->scale;
            float m = u->m[h];
            if (s - m > QE_ATTN_RESCALE_NATS) {
                const float alpha = expf(m - s);
                u->m[h] = m = s;
                qe_scale_row(y, alpha, hd);
                u->l[h] *= alpha;
            }
            const float w = expf(s - m);
            u->l[h] += w;
            if (u->q8)
                qe_axpy_q8(y, vq, w * vs, hd);
            else
                qe_axpy(y, vf, w * vs, hd);
        }
    }
}

void qe_attn_combine(const float *py, long ystride, const float *pm, const float *pl,
                     long pstride, int nparts, int hd, float *out) {
    float m = -1e30f;
    for (int i = 0; i < nparts; i++)
        if (pm[(long)i * pstride] > m)
            m = pm[(long)i * pstride];
    memset(out, 0, (size_t)hd * sizeof(float));
    if (!(m > -1e30f))
        return; /* every partial saw an empty chunk */
    double sum = 0.0;
    for (int i = 0; i < nparts; i++) {
        const float a = expf(pm[(long)i * pstride] - m);
        sum += (double)pl[(long)i * pstride] * a;
        if (a != 0.0f)
            qe_axpy(out, py + (long)i * ystride, a, hd);
    }
    if (sum > 0.0)
        qe_scale_row(out, (float)(1.0 / sum), hd);
}
