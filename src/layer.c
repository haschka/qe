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

#include "kern.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

static inline __m256 qe_sigmoid_avx2(__m256 x) {
    __m256 e = qe_exp2_avx2(_mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), x),
                                          _mm256_set1_ps(QE_LOG2E)));
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

/* The indexer's block scores, four heads against two blocks at a time: every dot is
 * qe_dot's (four accumulators over runs of 32, (a0 + a1) + (a2 + a3), then the eight lanes
 * summed left to right), so every score is the same float; the eight lane sums are done as
 * one transpose and seven vector adds in that same order instead of eight stores and 56
 * scalar adds, and a block's row is loaded once per two heads instead of once per head.
 * Shapes other than four heads of a multiple of 32 go through qe_dot. */
static int idx_fast_cached = -1;
void qe_idx_set_fast(int on) { idx_fast_cached = on ? 1 : 0; }
static int idx_fast(void) {
    if (idx_fast_cached < 0) {
        const char *e = getenv("QE_IDX_FAST");
        idx_fast_cached = e && *e ? atoi(e) != 0 : 1;
    }
    return idx_fast_cached;
}

#if defined(__x86_64__)
static inline void tr8_sum(const __m256 v[8], float out[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(v[0], v[1]), t1 = _mm256_unpackhi_ps(v[0], v[1]);
    const __m256 t2 = _mm256_unpacklo_ps(v[2], v[3]), t3 = _mm256_unpackhi_ps(v[2], v[3]);
    const __m256 t4 = _mm256_unpacklo_ps(v[4], v[5]), t5 = _mm256_unpackhi_ps(v[4], v[5]);
    const __m256 t6 = _mm256_unpacklo_ps(v[6], v[7]), t7 = _mm256_unpackhi_ps(v[6], v[7]);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    /* lane j of every input: L0..L7 */
    const __m256 L0 = _mm256_permute2f128_ps(u0, u4, 0x20), L4 = _mm256_permute2f128_ps(u0, u4, 0x31);
    const __m256 L1 = _mm256_permute2f128_ps(u1, u5, 0x20), L5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    const __m256 L2 = _mm256_permute2f128_ps(u2, u6, 0x20), L6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    const __m256 L3 = _mm256_permute2f128_ps(u3, u7, 0x20), L7 = _mm256_permute2f128_ps(u3, u7, 0x31);
    __m256 r = _mm256_add_ps(L0, L1);
    r = _mm256_add_ps(r, L2);
    r = _mm256_add_ps(r, L3);
    r = _mm256_add_ps(r, L4);
    r = _mm256_add_ps(r, L5);
    r = _mm256_add_ps(r, L6);
    r = _mm256_add_ps(r, L7);
    _mm256_storeu_ps(out, r);
}

/* Two heads against one row: qe_dot's accumulators for each. */
static inline void dot2_rows(const float *q0, const float *q1, const float *row, long n,
                             __m256 *s0, __m256 *s1) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (long i = 0; i < n; i += 32) {
        const __m256 r0 = _mm256_loadu_ps(row + i), r1 = _mm256_loadu_ps(row + i + 8);
        const __m256 r2 = _mm256_loadu_ps(row + i + 16), r3 = _mm256_loadu_ps(row + i + 24);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i), r0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 8), r1, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 16), r2, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 24), r3, a3);
        b0 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i), r0, b0);
        b1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 8), r1, b1);
        b2 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 16), r2, b2);
        b3 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 24), r3, b3);
    }
    *s0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    *s1 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
}
#endif

void qe_idx_scores(const float *iq, const float *pool, int nih, int id, float scale, long lo,
                   long hi, float *out) {
    long b = lo;
#if defined(__x86_64__)
    if (nih == 4 && id % 32 == 0 && idx_fast()) {
        const float *q0 = iq, *q1 = iq + id, *q2 = iq + 2 * (size_t)id, *q3 = iq + 3 * (size_t)id;
        for (; b + 2 <= hi; b += 2) {
            __m256 v[8];
            float d[8];
            const float *r0 = pool + (size_t)b * (size_t)id, *r1 = r0 + id;
            dot2_rows(q0, q1, r0, id, &v[0], &v[1]);
            dot2_rows(q2, q3, r0, id, &v[2], &v[3]);
            dot2_rows(q0, q1, r1, id, &v[4], &v[5]);
            dot2_rows(q2, q3, r1, id, &v[6], &v[7]);
            tr8_sum(v, d);
            for (int k = 0; k < 2; k++) {
                float acc = 0.0f;
                for (int h = 0; h < 4; h++)
                    if (d[4 * k + h] > 0.0f)
                        acc += d[4 * k + h];
                out[b + k] = acc * scale;
            }
        }
    }
#endif
    for (; b < hi; b++) {
        float acc = 0.0f;
        for (int h = 0; h < nih; h++) {
            float d = qe_dot(iq + (size_t)h * (size_t)id, pool + (size_t)b * (size_t)id, id);
            if (d > 0.0f)
                acc += d;
        }
        out[b] = acc * scale;
    }
}

/* The same scores over pooled keys stored as halves (QE_IDX_F16): each row widened with
 * F16C and walked exactly as qe_idx_scores walks a float row. */
#if defined(__x86_64__)
static inline void dot2_rows_h(const float *q0, const float *q1, const uint16_t *row, long n,
                               __m256 *s0, __m256 *s1) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (long i = 0; i < n; i += 32) {
        const __m256 r0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + i)));
        const __m256 r1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + i + 8)));
        const __m256 r2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + i + 16)));
        const __m256 r3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(row + i + 24)));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i), r0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 8), r1, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 16), r2, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i + 24), r3, a3);
        b0 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i), r0, b0);
        b1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 8), r1, b1);
        b2 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 16), r2, b2);
        b3 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i + 24), r3, b3);
    }
    *s0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    *s1 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
}
#endif

void qe_idx_scores_h(const float *iq, const void *poolv, int nih, int id, float scale, long lo,
                     long hi, float *out) {
    const uint16_t *pool = poolv;
    long b = lo;
#if defined(__x86_64__)
    if (nih == 4 && id % 32 == 0) {
        const float *q0 = iq, *q1 = iq + id, *q2 = iq + 2 * (size_t)id, *q3 = iq + 3 * (size_t)id;
        for (; b + 2 <= hi; b += 2) {
            __m256 v[8];
            float d[8];
            const uint16_t *r0 = pool + (size_t)b * (size_t)id, *r1 = r0 + id;
            dot2_rows_h(q0, q1, r0, id, &v[0], &v[1]);
            dot2_rows_h(q2, q3, r0, id, &v[2], &v[3]);
            dot2_rows_h(q0, q1, r1, id, &v[4], &v[5]);
            dot2_rows_h(q2, q3, r1, id, &v[6], &v[7]);
            tr8_sum(v, d);
            for (int k = 0; k < 2; k++) {
                float acc = 0.0f;
                for (int h = 0; h < 4; h++)
                    if (d[4 * k + h] > 0.0f)
                        acc += d[4 * k + h];
                out[b + k] = acc * scale;
            }
        }
    }
#endif
    for (; b < hi; b++) {
        float acc = 0.0f;
        for (int h = 0; h < nih; h++) {
            const float *q = iq + (size_t)h * (size_t)id;
            const uint16_t *r = pool + (size_t)b * (size_t)id;
            float d = 0.0f;
            for (int c = 0; c < id; c++)
                d += q[c] * _cvtsh_ss(r[c]);
            if (d > 0.0f)
                acc += d;
        }
        out[b] = acc * scale;
    }
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

/* The heads of one cell batched so that their weights are one exponential call.
 * Twelve is `per` at this checkpoint, so a cell is one eight-wide exp2 and four
 * scalar polynomials; the bound exists to keep the two scratch rows on the stack
 * and degrades to a head at a time for a checkpoint whose group is wider. */
#define QE_ATTN_SIMD_MAX 16

/* The same walk, blocked (QE_ATTN_BLK, on): the floating-point operations every output
 * element sees are the ones qe_attn_walk performs, in the same order, so the result is
 * the same bytes; what changes is how often the operands are fetched and widened.
 *
 *  - a key row is widened once per three heads instead of once per head, the three
 *    heads' four accumulators each living in registers across the row (qe_dot_q8's own
 *    accumulator layout, so the same sums);
 *  - the softmax bookkeeping runs per cell and per head exactly as before, scalar, and
 *    records each cell's weight and any rescale;
 *  - eight cells' value rows are widened once into L1, and each head's accumulator is
 *    loaded once per eight-float slice, receives the eight cells' rescales and FMAs in
 *    cell order in a register, and is stored once, where qe_axpy_q8 loaded and stored
 *    it per cell per head. */
#define ATTN_BLK 8
#if defined(__x86_64__)
static inline float hsum8_seq(__m256 s) {
    float t[8];
    _mm256_storeu_ps(t, s);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

/* One cell's key row against `nh` (1..3) heads: qe_dot_q8 per head, bit for bit. */
static inline void dot3_q8(const float *q, long hd, int nh, const int8_t *k, float *out) {
    __m256 a[3][4];
    for (int h = 0; h < 3; h++)
        for (int r = 0; r < 4; r++)
            a[h][r] = _mm256_setzero_ps();
    for (long i = 0; i < hd; i += 32) {
        const __m256 f0 = q_i8x8(k + i), f1 = q_i8x8(k + i + 8);
        const __m256 f2 = q_i8x8(k + i + 16), f3 = q_i8x8(k + i + 24);
        for (int h = 0; h < 3; h++) {
            if (h >= nh)
                break;
            const float *qh = q + (size_t)h * hd + i;
            a[h][0] = _mm256_fmadd_ps(_mm256_loadu_ps(qh), f0, a[h][0]);
            a[h][1] = _mm256_fmadd_ps(_mm256_loadu_ps(qh + 8), f1, a[h][1]);
            a[h][2] = _mm256_fmadd_ps(_mm256_loadu_ps(qh + 16), f2, a[h][2]);
            a[h][3] = _mm256_fmadd_ps(_mm256_loadu_ps(qh + 24), f3, a[h][3]);
        }
    }
    for (int h = 0; h < nh; h++)
        out[h] = hsum8_seq(_mm256_add_ps(_mm256_add_ps(a[h][0], a[h][1]),
                                         _mm256_add_ps(a[h][2], a[h][3])));
}

static void qe_attn_walk_blk(const qe_attn_unit *u) {
    const long hd = u->hd;
    const int per = u->per;
    for (int h = 0; h < per; h++) {
        u->m[h] = -1e30f;
        u->l[h] = 0.0f;
        memset(u->y + (size_t)h * hd, 0, (size_t)hd * sizeof(float));
    }
    float vf[ATTN_BLK][256] __attribute__((aligned(32)));
    float wv[ATTN_BLK][QE_ATTN_SIMD_MAX]; /* weight times the cell's value scale */
    float al[ATTN_BLK][QE_ATTN_SIMD_MAX]; /* rescale factor, 1 when none */
    unsigned char rs[ATTN_BLK][QE_ATTN_SIMD_MAX];
    for (int c0 = 0; c0 < u->nsel; c0 += ATTN_BLK) {
        const int nc = u->nsel - c0 < ATTN_BLK ? u->nsel - c0 : ATTN_BLK;
        int any_rs = 0;
        for (int c = 0; c < nc; c++) {
            const long cell = u->sel[c0 + c];
            const int8_t *kq = u->kd + (size_t)cell * u->krow + (size_t)u->kv_head * hd;
            const int8_t *vq = u->vd + (size_t)cell * u->krow + (size_t)u->kv_head * hd;
            const float ks = u->ks[(size_t)cell * u->sstride + u->kv_head];
            const float vs = u->vs[(size_t)cell * u->sstride + u->kv_head];
            float sc[QE_ATTN_SIMD_MAX], d[QE_ATTN_SIMD_MAX], w[QE_ATTN_SIMD_MAX];
            for (int h0 = 0; h0 < per; h0 += 3)
                dot3_q8(u->q + (size_t)h0 * hd, hd, per - h0 < 3 ? per - h0 : 3, kq, sc + h0);
            for (int h = 0; h < per; h++) {
                float s = sc[h] * ks;
                s *= u->scale;
                float m = u->m[h];
                if (s - m > QE_ATTN_RESCALE_NATS) {
                    const float alpha = expf(m - s);
                    u->m[h] = s;
                    u->l[h] *= alpha;
                    al[c][h] = alpha;
                    rs[c][h] = 1;
                    any_rs = 1;
                    d[h] = 0.0f;
                } else {
                    rs[c][h] = 0;
                    d[h] = s - m;
                }
            }
            if (u->exp2)
                qe_exp2v(d, w, per);
            else
                for (int h = 0; h < per; h++)
                    w[h] = expf(d[h]);
            for (int h = 0; h < per; h++) {
                u->l[h] += w[h];
                wv[c][h] = w[h] * vs;
            }
            for (long i = 0; i < hd; i += 8)
                _mm256_store_ps(&vf[c][i], q_i8x8(vq + i));
        }
        for (int h = 0; h < per; h++) {
            float *y = u->y + (size_t)h * hd;
            __m256 wb[ATTN_BLK], ab[ATTN_BLK];
            int hr = 0;
            for (int c = 0; c < nc; c++) {
                wb[c] = _mm256_set1_ps(wv[c][h]);
                ab[c] = _mm256_set1_ps(rs[c][h] ? al[c][h] : 1.0f);
                hr |= any_rs && rs[c][h];
            }
            /* Four slices in flight: one slice's eight FMAs are a dependent chain, and
             * a chain at a time is the FMA latency, not its throughput. */
            for (long i = 0; i < hd; i += 32) {
                __m256 a0 = _mm256_loadu_ps(y + i), a1 = _mm256_loadu_ps(y + i + 8);
                __m256 a2 = _mm256_loadu_ps(y + i + 16), a3 = _mm256_loadu_ps(y + i + 24);
                if (hr) {
                    for (int c = 0; c < nc; c++) {
                        if (rs[c][h]) {
                            a0 = _mm256_mul_ps(ab[c], a0);
                            a1 = _mm256_mul_ps(ab[c], a1);
                            a2 = _mm256_mul_ps(ab[c], a2);
                            a3 = _mm256_mul_ps(ab[c], a3);
                        }
                        a0 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i]), a0);
                        a1 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i + 8]), a1);
                        a2 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i + 16]), a2);
                        a3 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i + 24]), a3);
                    }
                } else {
                    for (int c = 0; c < nc; c++) {
                        a0 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i]), a0);
                        a1 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i + 8]), a1);
                        a2 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i + 16]), a2);
                        a3 = _mm256_fmadd_ps(wb[c], _mm256_load_ps(&vf[c][i + 24]), a3);
                    }
                }
                _mm256_storeu_ps(y + i, a0);
                _mm256_storeu_ps(y + i + 8, a1);
                _mm256_storeu_ps(y + i + 16, a2);
                _mm256_storeu_ps(y + i + 24, a3);
            }
        }
    }
}
#endif

static int attn_blk_cached = -1;
static int attn_blk_env(void) {
    if (attn_blk_cached < 0) {
        const char *e = getenv("QE_ATTN_BLK");
        attn_blk_cached = e && *e ? atoi(e) != 0 : 1;
    }
    return attn_blk_cached;
}
void qe_attn_set_blk(int on) { attn_blk_cached = on ? 1 : 0; }
int qe_attn_get_blk(void) { return attn_blk_env(); }

void qe_attn_walk(const qe_attn_unit *u) {
    const long hd = u->hd;
#if defined(__x86_64__)
    if (u->q8 && hd % 32 == 0 && hd <= 256 && u->per <= QE_ATTN_SIMD_MAX && attn_blk_env()) {
        qe_attn_walk_blk(u);
        return;
    }
#endif
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
        for (int h0 = 0; h0 < u->per; h0 += QE_ATTN_SIMD_MAX) {
            const int nb =
                u->per - h0 < QE_ATTN_SIMD_MAX ? u->per - h0 : QE_ATTN_SIMD_MAX;
            float d[QE_ATTN_SIMD_MAX], w[QE_ATTN_SIMD_MAX];
            /* Scores and bases first, weights after, because the wide form wants a
             * cell's exponents at once. A head whose score runs past its base still
             * rescales here, in place, one head at a time: the accumulator it pulls
             * up cannot be shared with a neighbour's. Its own exponent is then zero,
             * and both forms return exactly one at zero, which is what keeps a
             * rescale identical under QE_EXP2_SIMD rather than approximately so. */
            for (int h = 0; h < nb; h++) {
                const int hh = h0 + h;
                const float *q = u->q + (size_t)hh * hd;
                float s = (u->q8 ? qe_dot_q8(q, kq, hd) : qe_dot(q, kf, hd)) * ks;
                s *= u->scale;
                float m = u->m[hh];
                if (s - m > QE_ATTN_RESCALE_NATS) {
                    const float alpha = expf(m - s);
                    u->m[hh] = m = s;
                    qe_scale_row(u->y + (size_t)hh * hd, alpha, hd);
                    u->l[hh] *= alpha;
                    d[h] = 0.0f;
                } else {
                    d[h] = s - m;
                }
            }
            if (u->exp2)
                qe_exp2v(d, w, nb);
            else
                for (int h = 0; h < nb; h++)
                    w[h] = expf(d[h]);
            for (int h = 0; h < nb; h++) {
                const int hh = h0 + h;
                float *y = u->y + (size_t)hh * hd;
                u->l[hh] += w[h];
                if (u->q8)
                    qe_axpy_q8(y, vq, w[h] * vs, hd);
                else
                    qe_axpy(y, vf, w[h] * vs, hd);
            }
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
