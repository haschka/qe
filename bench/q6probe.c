/* Scratch probe: what a q6_K walk is limited by. Five forms of the same dot, the
 * double reference beside them, and a linear read of the same bytes as the floor.
 * Not a make target - built and run by hand, and deleted once it has decided. */
#include "kern.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double qe_dot_q6_k_ref(const qe_block_q6_k *w, const float *x, long nb);

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int qe_q6_hshift(int slot) {
    return (slot >> 1) * 4 + (slot & 1) * 2;
}

/* A panel: one activation row as int8, with one scale per 16 values and the integer
 * sum of each window times 32, which is the -32 offset returned in the integer
 * domain. One row of K = 2560 costs 2560 + 640 + 640 bytes, all L1-resident. */
typedef struct {
    int8_t *x8;
    float *sc;
    int32_t *sum;
} panel;

static void panel_fill(const float *x, long K, panel *p) {
    const long nw = K / 16;
    for (long j = 0; j < nw; j++) {
        float amax = 0;
        for (int i = 0; i < 16; i++) {
            const float a = fabsf(x[j * 16 + i]);
            if (a > amax) amax = a;
        }
        if (!(amax > 0)) {
            memset(p->x8 + j * 16, 0, 16);
            p->sc[j] = 0;
            p->sum[j] = 0;
            continue;
        }
        const float s = amax / 127.0f;
        long sum = 0;
        for (int i = 0; i < 16; i++) {
            const float q = x[j * 16 + i] / s;
            int qi = (int)(q >= 0 ? q + 0.5f : q - 0.5f);
            if (qi > 127) qi = 127;
            if (qi < -128) qi = -128;
            p->x8[j * 16 + i] = (int8_t)qi;
            sum += qi;
        }
        p->sc[j] = s;
        p->sum[j] = 32 * (int32_t)sum;
    }
}

/* The sixteen sub-block scales of a block as floats, one vector pass instead of
 * sixteen scalar int->float conversions. */
static inline void scales16(const int8_t *sc, float *sf) {
    _mm256_store_ps(sf, _mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)sc))));
    _mm256_store_ps(sf + 8, _mm256_cvtepi32_ps(
                                _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(sc + 8)))));
}

/* ------------------------------------------------------- v0: the tree's walk */

static inline void r0(__m256 *acc, __m256i ql, __m256i qh, int slot, const int8_t *sc,
                      const float *x) {
    const int s = qe_q6_hshift(slot);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qh, s), _mm256_set1_epi8(3));
    __m256i v = _mm256_or_si256(slot >= 2 ? qe_nib_hi(ql) : qe_nib_lo(ql),
                                _mm256_slli_epi16(hi, 4));
    v = _mm256_sub_epi8(v, _mm256_set1_epi8(32));
    const __m256 f0 = _mm256_set1_ps((float)sc[0]);
    const __m256 f1 = _mm256_set1_ps((float)sc[1]);
    __m128i a = _mm256_castsi256_si128(v), b = _mm256_extracti128_si256(v, 1);
    acc[0] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(a), f0), _mm256_loadu_ps(x + 0), acc[0]);
    acc[1] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(a, 8)), f0),
                             _mm256_loadu_ps(x + 8), acc[1]);
    acc[2] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(b), f1), _mm256_loadu_ps(x + 16), acc[2]);
    acc[3] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(b, 8)), f1),
                             _mm256_loadu_ps(x + 24), acc[3]);
}

static float d0(const qe_block_q6_k *w, const float *x, long nb) {
    float t = 0;
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d);
        const uint8_t *ql = w[b].ql, *qh = w[b].qh;
        const int8_t *sc = w[b].scales;
        QE_ACC_DECL(acc);
        for (int g = 0; g < 2; g++) {
            const __m256i qh0 = _mm256_loadu_si256((const __m256i *)(qh + g * 32));
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(ql + g * 64));
            const __m256i up = _mm256_loadu_si256((const __m256i *)(ql + g * 64 + 32));
            const float *xr = x + b * 256 + g * 128;
            r0(acc, lo, qh0, 0, sc + g * 8 + 0, xr + 0);
            r0(acc, up, qh0, 1, sc + g * 8 + 2, xr + 32);
            r0(acc, lo, qh0, 2, sc + g * 8 + 4, xr + 64);
            r0(acc, up, qh0, 3, sc + g * 8 + 6, xr + 96);
        }
        t += d * QE_ACC_HSUM(acc);
    }
    return t;
}

/* ------------------------------------------------ v1: v0 with hoisted scales */

static inline void r1(__m256 *acc, __m256i ql, __m256i qh, int slot, const float *sf, int j,
                      const float *x) {
    const int s = qe_q6_hshift(slot);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qh, s), _mm256_set1_epi8(3));
    __m256i v = _mm256_or_si256(slot >= 2 ? qe_nib_hi(ql) : qe_nib_lo(ql),
                                _mm256_slli_epi16(hi, 4));
    v = _mm256_sub_epi8(v, _mm256_set1_epi8(32));
    const __m256 f0 = _mm256_set1_ps(sf[j]);
    const __m256 f1 = _mm256_set1_ps(sf[j + 1]);
    __m128i a = _mm256_castsi256_si128(v), b = _mm256_extracti128_si256(v, 1);
    acc[0] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(a), f0), _mm256_loadu_ps(x + 0), acc[0]);
    acc[1] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(a, 8)), f0),
                             _mm256_loadu_ps(x + 8), acc[1]);
    acc[2] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(b), f1), _mm256_loadu_ps(x + 16), acc[2]);
    acc[3] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(b, 8)), f1),
                             _mm256_loadu_ps(x + 24), acc[3]);
}

static float d1(const qe_block_q6_k *w, const float *x, long nb) {
    float t = 0;
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d);
        const uint8_t *ql = w[b].ql, *qh = w[b].qh;
        float sf[16] __attribute__((aligned(32)));
        scales16(w[b].scales, sf);
        QE_ACC_DECL(acc);
        for (int g = 0; g < 2; g++) {
            const __m256i qh0 = _mm256_loadu_si256((const __m256i *)(qh + g * 32));
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(ql + g * 64));
            const __m256i up = _mm256_loadu_si256((const __m256i *)(ql + g * 64 + 32));
            const float *xr = x + b * 256 + g * 128;
            r1(acc, lo, qh0, 0, sf, 0, xr + 0);
            r1(acc, up, qh0, 1, sf, 2, xr + 32);
            r1(acc, lo, qh0, 2, sf, 4, xr + 64);
            r1(acc, up, qh0, 3, sf, 6, xr + 96);
        }
        t += d * QE_ACC_HSUM(acc);
    }
    return t;
}

/* --------------------- v2: v1 plus the two-op placement of the 2-bit field */

static inline void r2(__m256 *acc, __m256i nib, __m256i h4, const float *sf, int j,
                      const float *x) {
    __m256i v = _mm256_or_si256(nib, h4);
    v = _mm256_sub_epi8(v, _mm256_set1_epi8(32));
    const __m256 f0 = _mm256_set1_ps(sf[j]);
    const __m256 f1 = _mm256_set1_ps(sf[j + 1]);
    __m128i a = _mm256_castsi256_si128(v), b = _mm256_extracti128_si256(v, 1);
    acc[0] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(a), f0), _mm256_loadu_ps(x + 0), acc[0]);
    acc[1] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(a, 8)), f0),
                             _mm256_loadu_ps(x + 8), acc[1]);
    acc[2] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(b), f1), _mm256_loadu_ps(x + 16), acc[2]);
    acc[3] = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(b, 8)), f1),
                             _mm256_loadu_ps(x + 24), acc[3]);
}

/* The field of a byte that a run reads, placed at bits 4-5 with one shift: for s <= 4
 * it shifts left and every other field's bits land below the 0x30 mask, and s = 6 is
 * the one run that can only shift right. */
#define Q6_H4(SHIFT) _mm256_and_si256(SHIFT, _mm256_set1_epi8(0x30))

static float d2(const qe_block_q6_k *w, const float *x, long nb) {
    float t = 0;
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d);
        const uint8_t *ql = w[b].ql, *qh = w[b].qh;
        float sf[16] __attribute__((aligned(32)));
        scales16(w[b].scales, sf);
        QE_ACC_DECL(acc);
        for (int g = 0; g < 2; g++) {
            const __m256i qh0 = _mm256_loadu_si256((const __m256i *)(qh + g * 32));
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(ql + g * 64));
            const __m256i up = _mm256_loadu_si256((const __m256i *)(ql + g * 64 + 32));
            const __m256i nlo_l = qe_nib_lo(lo), nlo_u = qe_nib_lo(up);
            const __m256i nhi_l = qe_nib_hi(lo), nhi_u = qe_nib_hi(up);
            const float *xr = x + b * 256 + g * 128;
            r2(acc, nlo_l, Q6_H4(_mm256_slli_epi16(qh0, 4)), sf, 0, xr + 0);
            r2(acc, nlo_u, Q6_H4(_mm256_slli_epi16(qh0, 2)), sf, 2, xr + 32);
            r2(acc, nhi_l, Q6_H4(qh0), sf, 4, xr + 64);
            r2(acc, nhi_u, Q6_H4(_mm256_srli_epi16(qh0, 2)), sf, 6, xr + 96);
        }
        t += d * QE_ACC_HSUM(acc);
    }
    return t;
}

/* v3: v2 with the -32 returned by the widening instead of by a subtract, through the
 * 64 the int16 widen and wrap leaves in every weight. One vector of x sums per block
 * pays it back at the fold. */
static float d3(const qe_block_q6_k *w, const float *x, long nb) {
    const __m256i m30 = _mm256_set1_epi8(0x30), m0f = _mm256_set1_epi8(0x0f);
    const __m256i c64 = _mm256_set1_epi8(64);
    const __m256i z = _mm256_setzero_si256();
    float t = 0;
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d);
        const uint8_t *ql = w[b].ql, *qh = w[b].qh;
        float sf[16] __attribute__((aligned(32)));
        scales16(w[b].scales, sf);
        QE_ACC_DECL(acc);
        __m256 sx0 = z, sx1 = z, sx2 = z, sx3 = z;
        for (int g = 0; g < 2; g++) {
            const __m256i qh0 = _mm256_loadu_si256((const __m256i *)(qh + g * 32));
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(ql + g * 64));
            const __m256i up = _mm256_loadu_si256((const __m256i *)(ql + g * 64 + 32));
            const __m256i nib[4] = { _mm256_and_si256(lo, m0f), _mm256_and_si256(up, m0f),
                                     _mm256_and_si256(_mm256_srli_epi16(lo, 4), m0f),
                                     _mm256_and_si256(_mm256_srli_epi16(up, 4), m0f) };
            const __m256i h4[4] = { _mm256_and_si256(_mm256_slli_epi16(qh0, 4), m30),
                                    _mm256_and_si256(_mm256_slli_epi16(qh0, 2), m30),
                                    _mm256_and_si256(qh0, m30),
                                    _mm256_and_si256(_mm256_srli_epi16(qh0, 2), m30) };
            const float *xr = x + b * 256 + g * 128;
            for (int sl = 0; sl < 4; sl++) {
                const int j = 2 * sl;
                const __m256i v = _mm256_sub_epi8(_mm256_or_si256(nib[sl], h4[sl]), c64);
                const __m256 f0 = _mm256_set1_ps(sf[j]);
                const __m256 f1 = _mm256_set1_ps(sf[j + 1]);
                const __m256i e0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
                const __m256i e1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
                const __m256 p0 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(e0, z));
                const __m256 p1 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(e0, z));
                const __m256 p2 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(e1, z));
                const __m256 p3 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(e1, z));
                acc[0] = _mm256_fmadd_ps(_mm256_mul_ps(p0, f0), _mm256_loadu_ps(xr + 0), acc[0]);
                acc[1] = _mm256_fmadd_ps(_mm256_mul_ps(p1, f0), _mm256_loadu_ps(xr + 8), acc[1]);
                acc[2] = _mm256_fmadd_ps(_mm256_mul_ps(p2, f1), _mm256_loadu_ps(xr + 16), acc[2]);
                acc[3] = _mm256_fmadd_ps(_mm256_mul_ps(p3, f1), _mm256_loadu_ps(xr + 24), acc[3]);
                sx0 = _mm256_add_ps(sx0, _mm256_loadu_ps(xr + 0));
                sx1 = _mm256_add_ps(sx1, _mm256_loadu_ps(xr + 8));
                sx2 = _mm256_add_ps(sx2, _mm256_loadu_ps(xr + 16));
                sx3 = _mm256_add_ps(sx3, _mm256_loadu_ps(xr + 24));
            }
        }
        const __m256 sx = _mm256_add_ps(_mm256_add_ps(sx0, sx1), _mm256_add_ps(sx2, sx3));
        t += d * (QE_ACC_HSUM(acc) + 32.0f * qe_hsum(sx));
    }
    return t;
}

/* v4: the same walk on an int8 panel. One vpdpbusd per 32 weights, the -32 offset
 * returned inside the integer accumulator from the window sums the panel carries,
 * and the exit folding two windows' sub-scales with one blend. Not exact against a
 * float activation: the error it costs is measured here, not assumed. */
static float d4(const qe_block_q6_k *w, const panel *p, long nb) {
    const __m256i m30 = _mm256_set1_epi8(0x30), m0f = _mm256_set1_epi8(0x0f);
    const __m256i z = _mm256_setzero_si256();
    float tot = 0;
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d);
        const uint8_t *ql = w[b].ql, *qh = w[b].qh;
        const int8_t *sc = w[b].scales;
        float sf[16] __attribute__((aligned(32)));
        scales16(sc, sf);
        __m256 sumv = _mm256_setzero_ps();
        for (int g = 0; g < 2; g++) {
            const __m256i qh0 = _mm256_loadu_si256((const __m256i *)(qh + g * 32));
            const __m256i lo = _mm256_loadu_si256((const __m256i *)(ql + g * 64));
            const __m256i up = _mm256_loadu_si256((const __m256i *)(ql + g * 64 + 32));
            const __m256i nib[4] = { _mm256_and_si256(lo, m0f), _mm256_and_si256(up, m0f),
                                     _mm256_and_si256(_mm256_srli_epi16(lo, 4), m0f),
                                     _mm256_and_si256(_mm256_srli_epi16(up, 4), m0f) };
            const __m256i h4[4] = { _mm256_and_si256(_mm256_slli_epi16(qh0, 4), m30),
                                    _mm256_and_si256(_mm256_slli_epi16(qh0, 2), m30),
                                    _mm256_and_si256(qh0, m30),
                                    _mm256_and_si256(_mm256_srli_epi16(qh0, 2), m30) };
            const int8_t *x8 = p->x8 + b * 256 + g * 128;
            const int32_t *sm = p->sum + 8 * g;
            for (int sl = 0; sl < 4; sl++) {
                const int j = 2 * sl;
                const __m256i v = _mm256_or_si256(nib[sl], h4[sl]);
                const __m256i xv = _mm256_loadu_si256((const __m256i *)(x8 + sl * 32));
                __m256i acc = _mm256_dpbusd_epi32(z, v, xv);
                const __m128i seedl = _mm_set1_epi32(sm[j]);
                const __m128i seedh = _mm_set1_epi32(sm[j + 1]);
                acc = _mm256_sub_epi32(
                    acc, _mm256_inserti128_si256(_mm256_castsi128_si256(seedl), seedh, 1));
                const __m256 pf = _mm256_cvtepi32_ps(acc);
                const __m256 sa = _mm256_set1_ps(sf[j]);
                const __m256 sb = _mm256_set1_ps(sf[j + 1]);
                sumv = _mm256_fmadd_ps(pf, _mm256_blend_ps(sa, sb, 0xF0), sumv);
            }
        }
        tot += d * qe_hsum(sumv);
    }
    return tot;
}

/* ------------------------------------------------------------ driver */

typedef struct {
    const void *W;
    const float *x;
    const panel *p;
    float *y;
    long rows, nb, row_bytes;
    int which;
    void (*gemv)(const void *, const float *, float *, long, long, long);
} job;

static void row_task(void *v, int tid, int nt) {
    job *j = (job *)v;
    long per = (j->rows + nt - 1) / nt;
    long lo = (long)tid * per, hi = lo + per;
    if (hi > j->rows) hi = j->rows;
    if (j->which == 5) { /* the engine's own entry point, one call per slice */
        if (lo < hi)
            j->gemv((const uint8_t *)j->W + (size_t)lo * j->row_bytes, j->x, j->y + lo, hi - lo,
                    j->nb * 256, j->nb);
        return;
    }
    for (long r = lo; r < hi; r++) {
        const qe_block_q6_k *row = (const qe_block_q6_k *)((const uint8_t *)j->W +
                                                           (size_t)r * j->row_bytes);
        switch (j->which) {
        case 0: j->y[r] = d0(row, j->x, j->nb); break;
        case 1: j->y[r] = d1(row, j->x, j->nb); break;
        case 2: j->y[r] = d2(row, j->x, j->nb); break;
        case 3: j->y[r] = d3(row, j->x, j->nb); break;
        case 4: j->y[r] = d4(row, j->p, j->nb); break;
        }
    }
}

static size_t rd_n;
static const void *rd_buf;
static void rd_task(void *v, int tid, int nt) {
    (void)v;
    const uint64_t *p = (const uint64_t *)rd_buf;
    size_t per = (rd_n / 8 + nt - 1) / nt;
    size_t lo = (size_t)tid * per, hi = lo + per;
    if (hi > rd_n / 8) hi = rd_n / 8;
    uint64_t s = 0;
    for (size_t i = lo; i < hi; i += 7)
        s += p[i];
    if (s == 12345) rd_buf = 0;
}

static const char *NAMES[6] = { "v0 tree", "v1 scales", "v2 shift", "v3 widen", "v4 int8",
                                "v5 gemv" };

int main(int argc, char **argv) {
    qe_fuse(argc > 1 ? atoi(argv[1]) : 120);
    const int nthr = argc > 2 ? atoi(argv[2]) : 16;
    const int xmode = argc > 3 ? atoi(argv[3]) : 0; /* 0 random, 1 = bench_kern's sine */
    const long K = 2560, nb = K / 256;
    const size_t MB = 384;
    const long row_bytes = nb * (long)sizeof(qe_block_q6_k);
    const long rows = (long)((MB << 20) / (size_t)row_bytes);

    uint8_t *W = qe_alloc_huge(MB << 20);
    float *x = aligned_alloc(64, (size_t)K * 4);
    float *ya = aligned_alloc(64, (size_t)rows * 4), *yb = aligned_alloc(64, (size_t)rows * 4);
    if (!W || !x || !ya || !yb) { fprintf(stderr, "alloc\n"); return 1; }
    uint64_t s = 0x243f6a8885a308d3ull;
    for (size_t i = 0; i < (MB << 20) / 8; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        ((uint64_t *)W)[i] = s * 0x2545f4914f6cdd1dull;
    }
    for (long i = 0; i < K; i++) {
        if (xmode) {
            x[i] = (float)sin(i * 0.0017) * 0.5f;
            continue;
        }
        uint64_t t = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull;
        t ^= t >> 29;
        x[i] = (float)(((t >> 11) * 0x1p-53) - 0.5) * 2.0f;
    }
    for (long off = 0; off + row_bytes <= (long)(MB << 20); off += row_bytes)
        for (long b = 0; b < nb; b++) {
            uint16_t h = 0x3800u | (uint16_t)((off / 34 + b * 7) & 0x3ff);
            memcpy(W + off + b * (long)sizeof(qe_block_q6_k) + 208, &h, 2);
        }

    panel p = {0};
    p.x8 = aligned_alloc(64, (size_t)K);
    p.sc = aligned_alloc(64, (size_t)(K / 16) * 4);
    p.sum = aligned_alloc(64, (size_t)(K / 16) * 4);
    panel_fill(x, K, &p);

    qe_pool_init(nthr, NULL, 0);

    /* Accuracy: against the double reference, and for the exact forms the count of
     * rows whose float differs from the tree's walk at all. */
    double worst[5] = { 0, 0, 0, 0, 0 }, abs4 = 0;
    for (int r = 0; r < 256; r++) {
        const qe_block_q6_k *row = (const qe_block_q6_k *)((const uint8_t *)W +
                                                           (size_t)r * row_bytes);
        const double ref = qe_dot_q6_k_ref(row, x, nb);
        const double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
        const float g[5] = { d0(row, x, nb), d1(row, x, nb), d2(row, x, nb), d3(row, x, nb),
                             d4(row, &p, nb) };
        for (int i = 0; i < 5; i++)
            worst[i] = fmax(worst[i], fabs(ref - g[i]) / den);
        abs4 = fmax(abs4, fabs(ref - g[4]));
    }
    printf("K=%ld, %zu MiB, %d threads, %ld rows of %ld bytes, xmode %d\n", K, MB, nthr, rows,
           row_bytes, xmode);
    printf("worst relative error against the double reference: %.0e %.0e %.0e %.0e %.0e (abs4 %.3g)\n",
           worst[0], worst[1], worst[2], worst[3], worst[4], abs4);
    fflush(stdout);

    /* Measured in a rotatable order and printed in the canonical one, because a form
     * that looks 24 % slow only when it is timed last is an ordering artifact and not
     * a kernel, and the only way to tell the two apart is to be able to swap them. */
    const int rot = argc > 4 ? ((atoi(argv[4]) % 6) + 6) % 6 : 0;
    double gb_all[6][3];
    int diff_all[6];
    for (int i = 0; i < 6; i++) {
        const int w = (i + rot) % 6;
        job j = { W, x, &p, ya, rows, nb, row_bytes, w, qe_gemv_q6_k };
        double *gb = gb_all[w];
        const int tc[3] = { 1, 4, nthr };
        for (int c = 0; c < 3; c++) {
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                double a = now();
                qe_pool_run(row_task, &j, tc[c]);
                double d = now() - a;
                if (d < best) best = d;
            }
            gb[c] = (double)rows * row_bytes / best / 1e9;
        }
        job jc = j;
        jc.y = yb;
        jc.which = 0;
        qe_pool_run(row_task, &jc, nthr);
        int diff = 0;
        for (long r = 0; r < rows; r++)
            if (ya[r] != yb[r])
                diff++;
        diff_all[w] = diff;
    }
    const double base = gb_all[0][2];
    for (int w = 0; w < 6; w++) {
        const double *gb = gb_all[w];
        printf("%-10s %7.1f %7.1f %7.1f GB/s   vs v0 %.3f   lm_head %.1f ms   rows differing: %d\n",
               NAMES[w], gb[0], gb[1], gb[2], base > 0 ? gb[2] / base : 1.0,
               497.3 * 1048576.0 / (gb[2] * 1e9) * 1e3, diff_all[w]);
        fflush(stdout);
    }

    rd_buf = W;
    rd_n = MB << 20;
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double a = now();
        qe_pool_run(rd_task, &rd_n, nthr);
        double d = now() - a;
        if (d < best) best = d;
    }
    printf("%-10s  stride-one read of the same bytes: %.1f GB/s\n", "floor",
           (double)rd_n / best / 1e9);
    return 0;
}
