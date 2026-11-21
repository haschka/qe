/* Kernels for the checkpoint's remaining on-disk types.

   Which type earns engineering effort is decided by the per-token byte budget,
   not by how interesting it is to write:

     Q8_0   3.69 GiB  attention/GDN projections, hyper-connections, shared expert
     IQ3_S  0.63 GiB  routed expert gate/up
     Q6_K   0.49 GiB  lm_head (248320-entry vocab)
     IQ4_NL 0.41 GiB  routed expert down + PLE n-gram rows
     IQ4_XS 8 MiB     two tensors
     BF16   0.04 GiB  indexer projections

   Q4_K and Q5_0 are not in that table because they are not in the trunk's
   per-token budget at all: 2.6 GiB belongs to the draft head, which a
   checkpoint does not carry and a decode step does not read unless
   speculation is drafting. They were unwritten until the head had to be
   walked; both go through the same reference-then-SIMD discipline as the rest.

   The scalar reference for every type is compiled unconditionally. It is the
   oracle bench_kern checks the SIMD path against, and a wrong codebook entry
   or bit-field offset yields a plausible-looking tensor and a model that
   speaks fluent nonsense, so "it runs" is not evidence of anything.

   Format facts, read off tables.h and the block layouts rather than recalled:
     IQ4_NL  w = d * kvalues_iq4nl[nibble]; one fp16 scale per 32 weights
     IQ4_XS  same codebook, but a 6-bit scale (ls - 32) per 32 weights
     IQ3_S   9-bit index into a 512-entry codebook whose entries are four
             3-bit magnitudes (1,3,..,15) packed little-endian into a uint32;
             signs live in their own array, and each 32-weight group has its
             own scale d * (1 + 2*nibble)
     Q6_K    6-bit weights split across a 4-bit array and a 2-bit array, 16
             int8 sub-scales, one fp16 super-scale, bias -32
     Q5_0    int4 nibbles plus a 32-bit array of 5th bits, one fp16 scale per
             32 weights, bias -16; bit e of that array is the 5th bit of
             weight e, both halves alike
     Q4_K    int4 weights, one fp16 super-scale and one fp16 min, and eight
             6-bit scale/min pairs packed into 12 bytes; a 64-weight group's
             low nibbles take pair 2j and its high nibbles pair 2j+1, and the
             weight is d*sc*nib - min*m, so a dot needs the plain sum of x as
             well as the nibble dot
*/
#include "kern.h"
#include "tables.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define ROWPTR(type, w, r, ldd) \
    ((const type *)((const uint8_t *)(w) + (size_t)(r) * (size_t)(ldd) * sizeof(type)))

/* ---------------------------------------------------------------- references */

/* slot 0..3 are the four interleaved runs of a 128-weight group: they read
 * ql[l], ql[l+32], ql[l], ql[l+32] and land at offsets 0, 32, 64, 96. */
static int qe_q6_qbyte(int slot, int l) {
    return (slot & 1) ? 32 + l : l;
}
static int qe_q6_hshift(int slot) {
    return (slot >> 1) * 4 + (slot & 1) * 2;
}

double qe_dot_q6_k_ref(const qe_block_q6_k *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        const double d = qe_half_to_f32(w[b].d);
        const uint8_t *ql = w[b].ql, *qh = w[b].qh;
        const int8_t *sc = w[b].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                for (int slot = 0; slot < 4; slot++) {
                    const int qb = qe_q6_qbyte(slot, l);
                    const int nib = (slot >= 2) ? (ql[qb] >> 4) : (ql[qb] & 0xf);
                    const int q = nib | (((qh[l] >> qe_q6_hshift(slot)) & 3) << 4);
                    acc += d * sc[is + 2 * slot] * (double)(q - 32) *
                           x[b * 256 + n + slot * 32 + l];
                }
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

double qe_dot_iq4_nl_ref(const qe_block_iq4_nl *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        const double d = qe_half_to_f32(w[b].d);
        for (int j = 0; j < 16; j++) {
            acc += d * (double)(int8_t)qe_kvalues_iq4nl[w[b].qs[j] & 0xf] * x[b * 32 + j];
            acc += d * (double)(int8_t)qe_kvalues_iq4nl[w[b].qs[j] >> 4] * x[b * 32 + 16 + j];
        }
    }
    return acc;
}

double qe_dot_iq4_xs_ref(const qe_block_iq4_xs *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        const double d = qe_half_to_f32(w[b].d);
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((w[b].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                           (((w[b].scales_h >> (2 * ib)) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                const uint8_t q = w[b].qs[ib * 16 + j];
                acc += dl * (double)(int8_t)qe_kvalues_iq4nl[q & 0xf] * x[b * 256 + ib * 32 + j];
                acc += dl * (double)(int8_t)qe_kvalues_iq4nl[q >> 4] * x[b * 256 + ib * 32 + 16 + j];
            }
        }
    }
    return acc;
}

double qe_dot_iq3_s_ref(const qe_block_iq3_s *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        const double d = qe_half_to_f32(w[b].d);
        for (int p = 0; p < 4; p++) {
            for (int half = 0; half < 2; half++) {
                /* g0 indexes the 32-weight groups; qs, qh and signs all step
                 * by one entry per group, so one index covers all three. */
                const int g0 = p * 2 + half;
                const double db = d * (1 + 2.0 * ((w[b].scales[p] >> (4 * half)) & 0xf));
                const int hb = w[b].qh[g0];
                for (int k = 0; k < 8; k++) {
                    const uint32_t g =
                        qe_iq3s_grid[w[b].qs[g0 * 8 + k] | ((uint32_t)((hb >> k) & 1) << 8)];
                    for (int j = 0; j < 4; j++) {
                        const int e = g0 * 32 + k * 4 + j;
                        const int neg = (w[b].signs[e / 8] >> (e % 8)) & 1;
                        acc += db * (double)((g >> (8 * j)) & 0xff) *
                               (neg ? -x[b * 256 + e] : x[b * 256 + e]);
                    }
                }
            }
        }
    }
    return acc;
}

/* Scale and min of sub-block j, unpacked from the 12-byte scale array. The two
 * 6-bit fields live in separate bytes for j < 4 and, for j >= 4, share the
 * high two bits of the j-4 and j bytes, which is why the else arm reads four
 * bytes to produce two values. */
static void qe_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

double qe_dot_q4_k_ref(const qe_block_q4_k *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        const double d = qe_half_to_f32(w[b].d), min = qe_half_to_f32(w[b].dmin);
        const uint8_t *q = w[b].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mm;
            qe_scale_min_k4(is, w[b].scales, &sc, &mm);
            const double d1 = d * sc, m1 = min * mm;
            qe_scale_min_k4(is + 1, w[b].scales, &sc, &mm);
            const double d2 = d * sc, m2 = min * mm;
            for (int l = 0; l < 32; l++)
                acc += (d1 * (double)(q[l] & 0xF) - m1) * x[b * 256 + j + l];
            for (int l = 0; l < 32; l++)
                acc += (d2 * (double)(q[l] >> 4) - m2) * x[b * 256 + j + 32 + l];
            q += 32;
            is += 2;
        }
    }
    return acc;
}

double qe_dot_q5_0_ref(const qe_block_q5_0 *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        const double d = qe_half_to_f32(w[b].d);
        uint32_t qh;
        memcpy(&qh, w[b].qh, sizeof(qh));
        for (int j = 0; j < 16; j++) {
            const int h0 = ((qh >> j) << 4) & 0x10;
            const int h1 = (qh >> (j + 12)) & 0x10;
            acc += d * (double)(((w[b].qs[j] & 0xF) | h0) - 16) * x[b * 32 + j];
            acc += d * (double)(((w[b].qs[j] >> 4) | h1) - 16) * x[b * 32 + 16 + j];
        }
    }
    return acc;
}

double qe_dot_bf16_ref(const uint16_t *w, const float *x, long n) {
    double acc = 0;
    for (long i = 0; i < n; i++) {
        union {
            float f;
            uint32_t u;
        } v;
        v.u = (uint32_t)w[i] << 16;
        acc += (double)v.f * x[i];
    }
    return acc;
}

/* ---------------------------------------------------------------------- AVX2 */

#if defined(__x86_64__)

/* The IQ4 codebook as a pshufb operand. It is exactly 16 bytes, so one copy
 * per 128-bit lane plus a 0..15 index is a byte-wise lookup with no cross-lane
 * traffic. */
static inline __m256i qe_iq4_lut(void) {
    return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)qe_kvalues_iq4nl));
}

/* 16 nibble-pair bytes -> 32 codebook values in weight order, i.e.
 * [lut[lo(qs[0..15])], lut[hi(qs[0..15])]]. */
static inline __m256i qe_iq4_expand(__m256i v, __m256i lut) {
    __m256i lo = _mm256_shuffle_epi8(lut, qe_nib_lo(v));
    __m256i hi = _mm256_shuffle_epi8(lut, qe_nib_hi(v));
    return _mm256_permute2x128_si256(lo, hi, 0x20);
}

static float qe_dot_iq4_nl_avx2(const qe_block_iq4_nl *w, const float *x, long nb) {
    const __m256i lut = qe_iq4_lut();
    __m256 tot = _mm256_setzero_ps();
    for (long b = 0; b < nb; b++) {
        __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)w[b].qs));
        __m256i q = qe_iq4_expand(v, lut);
        QE_ACC_DECL(acc);
        QE_FMA_I8_X32(acc, q, x + b * 32);
        QE_ACC_FOLD(acc, tot, qe_half_to_f32(w[b].d));
    }
    return qe_hsum(tot);
}

static float qe_dot_iq4_xs_avx2(const qe_block_iq4_xs *w, const float *x, long nb) {
    const __m256i lut = qe_iq4_lut();
    __m256 tot = _mm256_setzero_ps();
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d);
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((w[b].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                           (((w[b].scales_h >> (2 * ib)) & 3) << 4);
            const uint8_t *qs = w[b].qs + ib * 16;
            __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)qs));
            __m256i q = qe_iq4_expand(v, lut);
            QE_ACC_DECL(acc);
            QE_FMA_I8_X32(acc, q, x + b * 256 + ib * 32);
            QE_ACC_FOLD(acc, tot, d * (float)(ls - 32));
        }
    }
    return qe_hsum(tot);
}

/* The codebook lookup is the whole of iq3_s: eight lookups per 32 weights,
 * and nothing else in the loop comes close to that cost. The eight indices are
 * eight bytes of one 8-byte run with one bit of the group's qh byte at bit 8,
 * and the tempting vector form - vpmovzxbd, vpsrlvd, vpor, then eight extracts
 * back to GPRs because the codebook is a scalar load - is 25 to 50 % slower
 * here (it halves the instruction count and loses anyway: bw_old against
 * bw_new in the §2 table of `make walk`): the round trip from a vector
 * register to an index register is port-limited and costs several times what a
 * movzbl costs, and the eight lookups are L1 hits either way. Same reason
 * vpgatherdd lost (39.8 vs 44.1 GB/s at 16 threads, bench_kern): the gather is
 * one instruction and the machine is not waiting on the arithmetic. */
static inline __m256i qe_iq3_group_loads(const uint8_t *qs, unsigned hb) {
    uint64_t qb;
    memcpy(&qb, qs, 8);
    return _mm256_setr_epi32(
        (int)qe_iq3s_grid[(uint32_t)(qb & 0xff) | ((uint32_t)(hb & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 8) & 0xff) | ((uint32_t)((hb >> 1) & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 16) & 0xff) | ((uint32_t)((hb >> 2) & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 24) & 0xff) | ((uint32_t)((hb >> 3) & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 32) & 0xff) | ((uint32_t)((hb >> 4) & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 40) & 0xff) | ((uint32_t)((hb >> 5) & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 48) & 0xff) | ((uint32_t)((hb >> 6) & 1u) << 8)],
        (int)qe_iq3s_grid[(uint32_t)((qb >> 56) & 0xff) | ((uint32_t)((hb >> 7) & 1u) << 8)]);
}

/* The eight group scales of one block, in one vector pass.
 *
 * Nibble g of the word the four scale bytes make is group g's scale, so one
 * variable shift across eight lanes produces all eight where the scalar form ran a
 * load, a shift, a mask, a convert and a three-op scale chain per group. Exact the
 * same way both ways: a nibble converts exactly, 1 + 2n is exact below 32, and the
 * only rounding is the multiply by the block's d - so a group dequantised through
 * this table gets weight bits identical to a group dequantised by the scalar form. */
static inline void qe_iq3_block_scales(const qe_block_iq3_s *w, long b, float *dbv) {
    uint32_t sc;
    memcpy(&sc, w[b].scales, sizeof(sc));
    const __m256i n = _mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32((int)sc),
                          _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)),
        _mm256_set1_epi32(0xf));
    const __m256 d = _mm256_set1_ps(qe_half_to_f32(w[b].d));
    _mm256_store_ps(dbv, _mm256_mul_ps(_mm256_fmadd_ps(_mm256_cvtepi32_ps(n),
                                                       _mm256_set1_ps(2.0f), _mm256_set1_ps(1.0f)),
                                       d));
}

/* Parameterised by the lookup so a competing lookup implementation can be
 * raced against this one without the sign handling and accumulation drifting
 * apart between the two copies. */
#define QE_IQ3_DOT(LOOKUP)                                                                 \
    static float qe_dot_iq3_s_avx2(const qe_block_iq3_s *w, const float *x, long nb) {      \
        __m256 tot = _mm256_setzero_ps();                                                   \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                     \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));         \
        /* Element e takes bit e%8 of sign byte e/8, so spreading the four sign              \
         * bytes to eight copies each lines the bits up with the magnitudes. */             \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,   \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,   \
                                                3, 3, 3, 3);                                \
        const __m256i one8 = _mm256_set1_epi8(1);                                          \
        for (long b = 0; b < nb; b++) {                                                    \
            const float d = qe_half_to_f32(w[b].d);                                        \
            const uint8_t *qs = w[b].qs, *qh = w[b].qh, *sg = w[b].signs;                   \
            for (int g = 0; g < 8; g++) {                                                   \
                /* One lookup of eight codebook entries is the whole 32-weight               \
                 * group: an entry is four magnitudes packed into a uint32, and               \
                 * consecutive entries are consecutive weights, so the loaded                 \
                 * dwords need no unpacking at all. */                                        \
                const int p = g >> 1, half = g & 1;                                          \
                const __m256i mag = LOOKUP(qs, qh[g]);                                       \
                const __m256i sb = _mm256_shuffle_epi8(                                      \
                    _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *)sg)),        \
                    spread);                                                                 \
                const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(sb, bitv), bitv);      \
                const __m256i q = _mm256_sign_epi8(mag, _mm256_or_si256(neg, one8));           \
                QE_ACC_DECL(acc);                                                             \
                QE_FMA_I8_X32(acc, q, x + b * 256 + g * 32);                                  \
                const float db = d * (1.0f + 2.0f * ((w[b].scales[p] >> (4 * half)) & 0xf));  \
                QE_ACC_FOLD(acc, tot, db);                                                    \
                qs += 8;                                                                      \
                sg += 4;                                                                      \
            }                                                                                \
        }                                                                                    \
        return qe_hsum(tot);                                                               \
    }

QE_IQ3_DOT(qe_iq3_group_loads)


/* One 32-weight run of a q6_K group. All eight runs of a row share the four
 * accumulators: the split exists for instruction-level parallelism only, so
 * which run lands in which register is irrelevant as long as each 8-lane chunk
 * keeps the same x offset.
 *
 * `ql` carries the 4-bit halves, `qh` the 2-bit halves, and the run straddles
 * two of the 16-weight sub-scales. */
static inline void qe_q6_run(__m256 *acc, __m256i ql, __m256i qh, int slot,
                             const int8_t *sc, const float *x) {
    const int s = qe_q6_hshift(slot);
    /* A 16-bit right shift bleeds between the two bytes of each halfword, but
     * only into bit positions above the low two, so the mask is unaffected. */
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

static float qe_dot_q6_k_avx2(const qe_block_q6_k *w, const float *x, long nb) {
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
            qe_q6_run(acc, lo, qh0, 0, sc + g * 8 + 0, xr + 0);
            qe_q6_run(acc, up, qh0, 1, sc + g * 8 + 2, xr + 32);
            qe_q6_run(acc, lo, qh0, 2, sc + g * 8 + 4, xr + 64);
            qe_q6_run(acc, up, qh0, 3, sc + g * 8 + 6, xr + 96);
        }
        t += d * QE_ACC_HSUM(acc);
    }
    return t;
}

/* ---------------------------------------------------------- Q4_K and Q5_0
 *
 * Both formats are the draft head's, and neither is on the trunk's critical
 * path, so they are written to be exact and cheap rather than clever. What
 * they share with the formats above is the shape of the oracle: a double
 * reference, compiled unconditionally, that bench_kern races against the SIMD
 * path.
 *
 * Q4_K is the only format in this file whose dot product needs two sums. The
 * weight is d*sc*nib - min*m, so a 32-weight group contributes
 *
 *      d*sc * sum(nib*x)  -  min*m * sum(x)
 *
 * and the second term does not depend on the weights at all. Both sums are free
 * to compute from the same four x loads, which is what QE_DOT_Q4K_X32 does: it
 * keeps the loaded activation vectors rather than a second accumulator, so the
 * group's sum of x is three adds at the fold instead of a chain of them, and a
 * sub-block that is summed once needs no zeroed start value either. Folding the
 * min into the weight instead would need a float weight per lane, and at 4.5
 * bpw the whole point of the format is that the weights are not floats.
 *
 * Q5_0 is the opposite: the 5th bit is part of the weight, so once the 32
 * int8s are assembled the group is a plain q8_0 group and one chain is enough.
 */

/* 32 int8 weights in w against 32 fp32 at x: acc[0..3] the four products,
 * xv[0..3] the four activation vectors, both in lane order. */
#define QE_DOT_Q4K_X32(acc, xv, w, x)                                                      \
    do {                                                                                   \
        __m128i _a = _mm256_castsi256_si128(w);                                            \
        __m128i _b = _mm256_extracti128_si256(w, 1);                                       \
        (xv)[0] = _mm256_loadu_ps((x) + 0);                                                \
        (xv)[1] = _mm256_loadu_ps((x) + 8);                                                \
        (xv)[2] = _mm256_loadu_ps((x) + 16);                                               \
        (xv)[3] = _mm256_loadu_ps((x) + 24);                                               \
        (acc)[0] = _mm256_mul_ps(qe_cvt8to8_f32(_a), (xv)[0]);                             \
        (acc)[1] = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(_a, 8)), (xv)[1]);          \
        (acc)[2] = _mm256_mul_ps(qe_cvt8to8_f32(_b), (xv)[2]);                             \
        (acc)[3] = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(_b, 8)), (xv)[3]);          \
    } while (0)

#define QE_SUM4(v)                                                                          \
    _mm256_add_ps(_mm256_add_ps((v)[0], (v)[1]), _mm256_add_ps((v)[2], (v)[3]))

/* One 32-weight sub-block's two terms into tot. */
#define QE_Q4K_FOLD(acc, xv, tot, s, m)                                                     \
    do {                                                                                   \
        (tot) = _mm256_fmadd_ps(QE_SUM4(acc), _mm256_set1_ps(s), (tot));                   \
        (tot) = _mm256_fnmadd_ps(QE_SUM4(xv), _mm256_set1_ps(m), (tot));                   \
    } while (0)

static float qe_dot_q4_k_avx2(const qe_block_q4_k *w, const float *x, long nb) {
    __m256 tot = _mm256_setzero_ps();
    for (long b = 0; b < nb; b++) {
        const float d = qe_half_to_f32(w[b].d), min = qe_half_to_f32(w[b].dmin);
        const uint8_t *q = w[b].qs;
        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mm;
            qe_scale_min_k4(is, w[b].scales, &sc, &mm);
            const float d1 = d * (float)sc, m1 = min * (float)mm;
            qe_scale_min_k4(is + 1, w[b].scales, &sc, &mm);
            const float d2 = d * (float)sc, m2 = min * (float)mm;
            /* One 32-byte load is 64 weights: the low nibbles of byte l are
             * weights 64j+l and the high nibbles 64j+32+l, so the order the
             * format writes them in is already the lane order here. */
            const __m256i v = _mm256_loadu_si256((const __m256i *)q);
            {
                __m256 acc[4], xv[4];
                QE_DOT_Q4K_X32(acc, xv, qe_nib_lo(v), x + b * 256 + j * 64);
                QE_Q4K_FOLD(acc, xv, tot, d1, m1);
            }
            {
                __m256 acc[4], xv[4];
                QE_DOT_Q4K_X32(acc, xv, qe_nib_hi(v), x + b * 256 + j * 64 + 32);
                QE_Q4K_FOLD(acc, xv, tot, d2, m2);
            }
            q += 32;
            is += 2;
        }
    }
    return qe_hsum(tot);
}

/* The 32 int8 weights of a q5_0 group, in weight order: lanes 0..15 are the low
 * nibbles of qs[0..15] and lanes 16..31 their high nibbles.
 *
 * Broadcasting the 16 scale bytes to both halves and taking the nibbles where
 * they belong leaves the two halves in the two 128-bit lanes of one register
 * each, so one lane-crossing permute assembles them. The 5th bits then come in
 * the same way iq3_s brings its signs in: a shuffle of the four qh bytes, one
 * bit mask per lane, a compare, and an and that lands the bit at 0x10.
 * Subtracting 16 in int8 makes the range -16..15, which is what the q8_0 fold
 * expects. */
static float qe_dot_q5_0_avx2(const qe_block_q5_0 *w, const float *x, long nb) {
    __m256 tot = _mm256_setzero_ps();
    /* Lane e shows byte e/8 of qh and bit e%8 of it, so all four bytes are
     * reached with indices 0..3 inside each 128-bit half, which is what
     * vpshufb's within-lane indexing needs. */
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));
    for (long b = 0; b < nb; b++) {
        uint32_t qh;
        memcpy(&qh, w[b].qh, sizeof(qh));
        const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)w[b].qs));
        const __m256i nib = _mm256_permute2x128_si256(qe_nib_lo(v), qe_nib_hi(v), 0x30);
        const __m256i sh =
            _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_cvtsi32_si128((int)qh)), spread);
        const __m256i hi5 = _mm256_and_si256(
            _mm256_cmpeq_epi8(_mm256_and_si256(sh, bitv), bitv), _mm256_set1_epi8(0x10));
        const __m256i q = _mm256_sub_epi8(_mm256_or_si256(nib, hi5), _mm256_set1_epi8(16));
        QE_ACC_DECL(acc);
        QE_FMA_I8_X32(acc, q, x + b * 32);
        QE_ACC_FOLD(acc, tot, qe_half_to_f32(w[b].d));
    }
    return qe_hsum(tot);
}

/* ------------------------------------------------- shared-dequant chunk forms
 *
 * iq4_nl and iq3_s are the two formats this checkpoint reaches a projection
 * with, and unlike the Q8_0 projections they are not memory-bound: 60 and
 * 44 GB/s against a 76-79 GB/s ceiling (`make alu`), because the arithmetic of
 * turning a block into floats costs more than reading it. iq3_s is the extreme
 * case - eight 256-entry grid lookups, a sign spread and a compare per 32
 * weights - and that is arithmetic a chunk of T tokens does not need to repeat,
 * because every token multiplies the *same* weights.
 *
 * So: dequantise once into four scaled float vectors and pay one FMA group per
 * token, which is the same shape as qe_gemv_q8_0_multi and for the same reason
 * (src/kern.c explains the register budget). Two accumulator chains per token,
 * no more: 4T+1 ymm is 17 at T=4 and does not fit in 16, and the dequantised
 * vector is the one thing that must not be spilled.
 *
 * The scale moves from the accumulator to the weight. The single-row kernels
 * accumulate raw codebook values and fold with the per-block scale, because
 * folding is then one FMA per block; sharing one block between T tokens means
 * the scale has to be on the weight before the first multiply. That is the same
 * sum in another order, not bit-identical to the T calls it replaces, and it is
 * checked against the double reference for exactly that reason (bench_kern). */

/* One 32-weight iq4_nl block, dequantised once into four scaled float vectors.
 * Declares q0..q3 in the enclosing scope, so every use needs its own { }. */
#define QE_QUAD_IQ4_NL(b, lut)                                                                     \
    __m256 q0, q1, q2, q3;                                                                         \
    do {                                                                                           \
        __m256 vd = _mm256_set1_ps(qe_half_to_f32((b)->d));                                        \
        __m256i v =                                                                                  \
            _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(b)->qs));                  \
        __m256i wv = qe_iq4_expand(v, (lut));                                                        \
        __m128i h = _mm256_castsi256_si128(wv);                                                      \
        __m128i l = _mm256_extracti128_si256(wv, 1);                                                 \
        q0 = _mm256_mul_ps(qe_cvt8to8_f32(h), vd);                                                   \
        q1 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(h, 8)), vd);                                \
        q2 = _mm256_mul_ps(qe_cvt8to8_f32(l), vd);                                                   \
        q3 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(l, 8)), vd);                                \
    } while (0)

/* One 32-weight group of an iq3_s block: the grid lookup and the sign pass that
 * the single-row kernel does, and the group's scale folded into the weight. */
#define QE_QUAD_IQ3_S(qs, hb, sg, db, bitv, spread, one8)                                          \
    __m256 q0, q1, q2, q3;                                                                         \
    do {                                                                                           \
        __m256 vd = _mm256_set1_ps(db);                                                            \
        __m256i mag = qe_iq3_group_loads((qs), (hb));                                              \
        __m256i sb = _mm256_shuffle_epi8(                                                          \
            _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *)(sg))), (spread));         \
        __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(sb, (bitv)), (bitv));                      \
        __m256i wv = _mm256_sign_epi8(mag, _mm256_or_si256(neg, (one8)));                            \
        __m128i h = _mm256_castsi256_si128(wv);                                                      \
        __m128i l = _mm256_extracti128_si256(wv, 1);                                                 \
        q0 = _mm256_mul_ps(qe_cvt8to8_f32(h), vd);                                                   \
        q1 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(h, 8)), vd);                                \
        q2 = _mm256_mul_ps(qe_cvt8to8_f32(l), vd);                                                   \
        q3 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(l, 8)), vd);                                \
    } while (0)

/* The 32 weights above against 32 activations at x. Same lane assignment as
 * QE_FMA_I8_X32, so a chunk and a decode step agree lane for lane. */
#define QE_FMA4_SCALED(acc, x)                                                                     \
    do {                                                                                           \
        const float *xa_ = (x);                                                                    \
        (acc) = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xa_ + 0), (acc));                              \
        (acc) = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xa_ + 8), (acc));                              \
        (acc) = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xa_ + 16), (acc));                             \
        (acc) = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xa_ + 24), (acc));                             \
    } while (0)

/* The block the two QUAD macros above just dequantised, parked in a 32-float aligned
 * line. Eight tokens want eight accumulators and the block will not sit next to them,
 * so the block goes to L1 and pays for its four stores the second time it is read;
 * QE_Q8_0_MULTI8_FORM in src/kern.c is the same argument at the same length. */
#define QE_STORE_QB                                                                                \
    do {                                                                                           \
        _mm256_store_ps(qb + 0, q0);                                                               \
        _mm256_store_ps(qb + 8, q1);                                                               \
        _mm256_store_ps(qb + 16, q2);                                                              \
        _mm256_store_ps(qb + 24, q3);                                                              \
    } while (0)

/* The 32 weights at qb against 32 activations at x. Lane for lane the same products as
 * QE_FMA4_SCALED, so the eight-token form differs from the four-token one only in
 * where the weight vector came from. */
#define QE_FMA4_QB(acc, x)                                                                         \
    do {                                                                                           \
        const float *xa_ = (x);                                                                    \
        (acc) = _mm256_fmadd_ps(_mm256_load_ps(qb + 0), _mm256_loadu_ps(xa_ + 0), (acc));          \
        (acc) = _mm256_fmadd_ps(_mm256_load_ps(qb + 8), _mm256_loadu_ps(xa_ + 8), (acc));          \
        (acc) = _mm256_fmadd_ps(_mm256_load_ps(qb + 16), _mm256_loadu_ps(xa_ + 16), (acc));        \
        (acc) = _mm256_fmadd_ps(_mm256_load_ps(qb + 24), _mm256_loadu_ps(xa_ + 24), (acc));        \
    } while (0)

/* T has to be a literal so the token loop unrolls into registers, as with
 * QE_Q8_0_MULTI_FORM: a runtime T puts the accumulators on the stack, and the
 * kernel is then slower than the T calls it replaced.
 *
 * The token rows arrive as pointers and not as a stride because a chunk of
 * routed experts is a *set* of tokens, not a run of them: the T rows that chose
 * one expert are wherever the gather left them, and copying them next to each
 * other costs a pass over T x K floats per call to save nothing. */
#define QE_IQ4_NL_MULTI_FORM(name, T)                                                              \
    static void name(const qe_block_iq4_nl *restrict w, const float *const *restrict xs,           \
                     float *const *restrict ys, long row, long nb) {                                \
        const __m256i lut = qe_iq4_lut();                                                          \
        __m256 a[T][2];                                                                            \
        for (int t = 0; t < T; t++) {                                                              \
            a[t][0] = _mm256_setzero_ps();                                                         \
            a[t][1] = a[t][0];                                                                     \
        }                                                                                          \
        long b = 0;                                                                                \
        for (; b + 2 <= nb; b += 2) {                                                              \
            {                                                                                      \
                QE_QUAD_IQ4_NL(w + b, lut);                                                        \
                for (int t = 0; t < T; t++)                                                        \
                    QE_FMA4_SCALED(a[t][0], xs[t] + b * 32);                                       \
            }                                                                                      \
            {                                                                                      \
                QE_QUAD_IQ4_NL(w + b + 1, lut);                                                    \
                for (int t = 0; t < T; t++)                                                        \
                    QE_FMA4_SCALED(a[t][1], xs[t] + (b + 1) * 32);                                 \
            }                                                                                      \
        }                                                                                          \
        for (; b < nb; b++) {                                                                      \
            QE_QUAD_IQ4_NL(w + b, lut);                                                            \
            for (int t = 0; t < T; t++)                                                            \
                QE_FMA4_SCALED(a[t][0], xs[t] + b * 32);                                           \
        }                                                                                          \
        for (int t = 0; t < T; t++)                                                                \
            ys[t][row] = qe_hsum(_mm256_add_ps(a[t][0], a[t][1]));                                  \
    }

QE_IQ4_NL_MULTI_FORM(qe_row_iq4_nl_multi2, 2)
QE_IQ4_NL_MULTI_FORM(qe_row_iq4_nl_multi3, 3)
QE_IQ4_NL_MULTI_FORM(qe_row_iq4_nl_multi4, 4)

/* Eight tokens, one dequantisation, and the block in L1 instead of in registers - see
 * QE_STORE_QB above and QE_Q8_0_MULTI8_FORM for why eight is where the register file
 * stops this instruction set rather than where anyone chose to stop. */
#define QE_IQ4_NL_MULTI8_FORM(name, T)                                                             \
    static void name(const qe_block_iq4_nl *restrict w, const float *const *restrict xs,           \
                     float *const *restrict ys, long row, long nb) {                                \
        const __m256i lut = qe_iq4_lut();                                                          \
        __attribute__((aligned(32))) float qb[32];                                                 \
        __m256 a[T];                                                                               \
        for (int t = 0; t < T; t++)                                                                \
            a[t] = _mm256_setzero_ps();                                                            \
        for (long b = 0; b < nb; b++) {                                                            \
            {                                                                                      \
                QE_QUAD_IQ4_NL(w + b, lut);                                                        \
                QE_STORE_QB;                                                                       \
            }                                                                                      \
            const long at = b * 32;                                                                \
            for (int t = 0; t < T; t++)                                                            \
                QE_FMA4_QB(a[t], xs[t] + at);                                                      \
        }                                                                                          \
        for (int t = 0; t < T; t++)                                                                \
            ys[t][row] = qe_hsum(a[t]);                                                            \
    }

QE_IQ4_NL_MULTI8_FORM(qe_row_iq4_nl_multi8, 8)


/* A bucket wider than eight, one dequantisation for all of it. What will not seat in
 * the register file is the accumulators, not the block, so this form swaps the two:
 * the dequantised group stays in q0..q3 and each token's running sum lives in an
 * aligned line of L1, read and written once per 32 weights. That is two loads and a
 * store per token per group, paid against a whole walk of the quantiser for the tail
 * tokens that a dispatch by eights forces a bucket of ten to walk twice. Measured
 * against the eight-plus-tail dispatch it replaces: 1.32x at twelve tokens, 1.16x at
 * sixteen, 1.26x at thirty-two (bench_moe, /tmp/moe_park1.log).
 *
 * The literal T is load-bearing. The same body with the count taken at run time is
 * 1.3x SLOWER than eight - the sums leave the schedule and take the block with them -
 * so the ceiling below is instantiated rather than looped over.
 *
 * One chain per token, exactly as the eight-token form has, so a parked bucket and a
 * full group of eight agree lane for lane. */
#define QE_IQ4_NL_PARK_FORM(name, T)                                                               \
    static void name(const qe_block_iq4_nl *restrict w, const float *const *restrict xs,           \
                     float *const *restrict ys, long row, long nb) {                                \
        const __m256i lut = qe_iq4_lut();                                                          \
        __attribute__((aligned(32))) float ab[(T) * 8];                                            \
        for (int t = 0; t < (T); t++)                                                              \
            _mm256_store_ps(ab + t * 8, _mm256_setzero_ps());                                      \
        for (long b = 0; b < nb; b++) {                                                            \
            const long at = b * 32;                                                                \
            {                                                                                      \
                QE_QUAD_IQ4_NL(w + b, lut);                                                        \
                for (int t = 0; t < (T); t++) {                                                    \
                    __m256 s = _mm256_load_ps(ab + t * 8);                                         \
                    const float *xa_ = xs[t] + at;                                                 \
                    s = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xa_ + 0), s);                          \
                    s = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xa_ + 8), s);                          \
                    s = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xa_ + 16), s);                         \
                    s = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xa_ + 24), s);                         \
                    _mm256_store_ps(ab + t * 8, s);                                                \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        for (int t = 0; t < (T); t++)                                                              \
            ys[t][row] = qe_hsum(_mm256_load_ps(ab + t * 8));                                      \
    }

QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park9, 9)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park10, 10)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park11, 11)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park12, 12)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park13, 13)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park14, 14)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park15, 15)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park16, 16)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park20, 20)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park24, 24)
QE_IQ4_NL_PARK_FORM(qe_row_iq4_nl_park32, 32)

/* Two output rows of one expert, one load of the activations per token.
 *
 * The forms above spend four activation loads and four FMAs per token per 32 weights, and
 * half of that is a vector every row of the same expert multiplies: row r and row r+1 want
 * the same activations, and the walk meets them in two passes and reads them twice. Holding
 * both dequantised blocks beside the eight sums spends the register file the same way `mix8`
 * does and takes the loads out of the middle of the walk, which at this shape is most of a
 * row's cost: `make walk` reads a tensor of 0.5 GiB at 1.05x for a group of eight, 1.14x at
 * sixteen and 1.14x at thirty-two (iq3_s 1.11x, 1.15x, 1.10x), and `make moe` prices it on the
 * grouped expert pass the engine runs.
 *
 * This is not a change of number. The chain per token is the eight-token form's - one
 * accumulator, the same four FMAs per 32 weights in block order, the same hsum - so a pair
 * walked together leaves the bits two walks left, at every row count and every group width,
 * which is what section 3 of `make walk` holds it to. That is also why there is only one
 * instantiation: the two-, three- and four-token forms accumulate in two chains by block
 * parity, so pairing *those* groups would be a different number as well as a different
 * schedule, and the dispatchers hand their leftover tokens to the single-row forms instead.
 * The row pair is a pair of *rows*, not a stride: a bucket's rows are wherever the gather
 * left them, so the pairing is over the output rows of one expert matrix, which are
 * contiguous, and never over tokens. */
#define QE_IQ4_NL_MULTI2R_FORM(name, T)                                                            \
    static void name(const qe_block_iq4_nl *restrict w0, const qe_block_iq4_nl *restrict w1,       \
                     const float *const *restrict xs, float *const *restrict ys,                   \
                     long row0, long row1, long nb) {                                               \
        const __m256i lut = qe_iq4_lut();                                                           \
        __m256 a0[T], a1[T];                                                                        \
        for (int t = 0; t < (T); t++) {                                                             \
            a0[t] = _mm256_setzero_ps();                                                            \
            a1[t] = _mm256_setzero_ps();                                                            \
        }                                                                                           \
        for (long b = 0; b < nb; b++) {                                                             \
            __m256 q00, q01, q02, q03, q10, q11, q12, q13;                                          \
            {                                                                                       \
                QE_QUAD_IQ4_NL(w0 + b, lut);                                                        \
                q00 = q0;                                                                           \
                q01 = q1;                                                                           \
                q02 = q2;                                                                           \
                q03 = q3;                                                                           \
            }                                                                                       \
            {                                                                                       \
                QE_QUAD_IQ4_NL(w1 + b, lut);                                                        \
                q10 = q0;                                                                           \
                q11 = q1;                                                                           \
                q12 = q2;                                                                           \
                q13 = q3;                                                                           \
            }                                                                                       \
            for (int t = 0; t < (T); t++) {                                                         \
                const float *xa_ = xs[t] + b * 32;                                                   \
                const __m256 x0 = _mm256_loadu_ps(xa_ + 0);                                          \
                const __m256 x1 = _mm256_loadu_ps(xa_ + 8);                                          \
                const __m256 x2 = _mm256_loadu_ps(xa_ + 16);                                         \
                const __m256 x3 = _mm256_loadu_ps(xa_ + 24);                                         \
                a0[t] = _mm256_fmadd_ps(q00, x0, a0[t]);                                             \
                a0[t] = _mm256_fmadd_ps(q01, x1, a0[t]);                                             \
                a0[t] = _mm256_fmadd_ps(q02, x2, a0[t]);                                             \
                a0[t] = _mm256_fmadd_ps(q03, x3, a0[t]);                                             \
                a1[t] = _mm256_fmadd_ps(q10, x0, a1[t]);                                             \
                a1[t] = _mm256_fmadd_ps(q11, x1, a1[t]);                                             \
                a1[t] = _mm256_fmadd_ps(q12, x2, a1[t]);                                             \
                a1[t] = _mm256_fmadd_ps(q13, x3, a1[t]);                                             \
            }                                                                                       \
        }                                                                                           \
        for (int t = 0; t < (T); t++) {                                                             \
            ys[t][row0] = qe_hsum(a0[t]);                                                           \
            ys[t][row1] = qe_hsum(a1[t]);                                                           \
        }                                                                                           \
    }

QE_IQ4_NL_MULTI2R_FORM(qe_row_iq4_nl_multi2r8, 8)

/* Chains are group parity rather than block parity here: a block is eight
 * groups, and a group is what has one scale, so this is the finest split the
 * format allows. No T=4 form: three constants plus four live dequantised
 * vectors plus 2T accumulators is 13 at T=3 and 15 at T=4, and the four that
 * spill are the ones holding the block. */
#define QE_IQ3_S_MULTI_FORM(name, T)                                                               \
    static void name(const qe_block_iq3_s *restrict w, const float *const *restrict xs,            \
                     float *const *restrict ys, long row, long nb) {                                \
        __attribute__((aligned(32))) float dbv[8];                                                 \
        __m256 a[T][2];                                                                            \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                            \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,          \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,          \
                                                3, 3, 3, 3);                                       \
        const __m256i one8 = _mm256_set1_epi8(1);                                                  \
        for (int t = 0; t < T; t++) {                                                              \
            a[t][0] = _mm256_setzero_ps();                                                         \
            a[t][1] = a[t][0];                                                                     \
        }                                                                                          \
        for (long b = 0; b < nb; b++) {                                                            \
            qe_iq3_block_scales(w, b, dbv);                                                        \
            const uint8_t *qs = w[b].qs, *sg = w[b].signs;                                          \
            for (int g = 0; g < 8; g++) {                                                           \
                {                                                                                  \
                    QE_QUAD_IQ3_S(qs, w[b].qh[g], sg, dbv[g], bitv, spread, one8);                  \
                    for (int t = 0; t < T; t++)                                                    \
                        QE_FMA4_SCALED(a[t][g & 1], xs[t] + b * 256 + g * 32);                      \
                }                                                                                  \
                qs += 8;                                                                           \
                sg += 4;                                                                           \
            }                                                                                      \
        }                                                                                          \
        for (int t = 0; t < T; t++)                                                                \
            ys[t][row] = qe_hsum(_mm256_add_ps(a[t][0], a[t][1]));                                  \
    }

QE_IQ3_S_MULTI_FORM(qe_row_iq3_s_multi2, 2)
QE_IQ3_S_MULTI_FORM(qe_row_iq3_s_multi3, 3)

/* The same eight tokens for the format that costs the most to dequantise, which is
 * the reason to want them: iq3_s is eight grid lookups, a sign spread and a compare
 * per 32 weights, and a group of eight tokens pays that once instead of three times.
 * The chains are one per token here rather than two, so the sum a chunk produces is
 * the same products in a third order - checked against the double reference, as the
 * four-token forms are. */
#define QE_IQ3_S_MULTI8_FORM(name, T)                                                              \
    static void name(const qe_block_iq3_s *restrict w, const float *const *restrict xs,            \
                     float *const *restrict ys, long row, long nb) {                                \
        __attribute__((aligned(32))) float dbv[8];                                                 \
        __attribute__((aligned(32))) float qb[32];                                                 \
        __m256 a[T];                                                                               \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                            \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,          \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,          \
                                                3, 3, 3, 3);                                       \
        const __m256i one8 = _mm256_set1_epi8(1);                                                   \
        for (int t = 0; t < T; t++)                                                                \
            a[t] = _mm256_setzero_ps();                                                            \
        for (long b = 0; b < nb; b++) {                                                            \
            qe_iq3_block_scales(w, b, dbv);                                                        \
            const uint8_t *qs = w[b].qs, *sg = w[b].signs;                                          \
            for (int g = 0; g < 8; g++) {                                                          \
                {                                                                                  \
                    QE_QUAD_IQ3_S(qs, w[b].qh[g], sg, dbv[g], bitv, spread, one8);                  \
                    QE_STORE_QB;                                                                   \
                }                                                                                  \
                const long at = b * 256 + g * 32;                                                  \
                for (int t = 0; t < T; t++)                                                        \
                    QE_FMA4_QB(a[t], xs[t] + at);                                                  \
                qs += 8;                                                                           \
                sg += 4;                                                                           \
            }                                                                                      \
        }                                                                                          \
        for (int t = 0; t < T; t++)                                                                \
            ys[t][row] = qe_hsum(a[t]);                                                            \
    }

QE_IQ3_S_MULTI8_FORM(qe_row_iq3_s_multi8, 8)


/* Nine to thirty-two tokens of one bucket in a single walk of the grid lookups, which
 * is the format's deardest half: eight grid lookups, a sign spread and a compare per
 * 32 weights. A bucket of ten through the dispatch above is an eight and a two, which
 * is two of those walks for ten tokens; this is one. Same argument, same shape and the
 * same measured reason as QE_IQ4_NL_PARK_FORM - read that comment for why the count is
 * a literal and what the parked sums cost. */
#define QE_IQ3_S_PARK_FORM(name, T)                                                                \
    static void name(const qe_block_iq3_s *restrict w, const float *const *restrict xs,            \
                     float *const *restrict ys, long row, long nb) {                                \
        __attribute__((aligned(32))) float dbv[8];                                                 \
        __attribute__((aligned(32))) float ab[(T) * 8];                                            \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                            \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,          \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,          \
                                                3, 3, 3, 3);                                       \
        const __m256i one8 = _mm256_set1_epi8(1);                                                   \
        for (int t = 0; t < (T); t++)                                                              \
            _mm256_store_ps(ab + t * 8, _mm256_setzero_ps());                                      \
        for (long b = 0; b < nb; b++) {                                                            \
            qe_iq3_block_scales(w, b, dbv);                                                        \
            const uint8_t *qs = w[b].qs, *sg = w[b].signs;                                          \
            for (int g = 0; g < 8; g++) {                                                          \
                const long at = b * 256 + g * 32;                                                  \
                {                                                                                  \
                    QE_QUAD_IQ3_S(qs, w[b].qh[g], sg, dbv[g], bitv, spread, one8);                  \
                    for (int t = 0; t < (T); t++) {                                                 \
                        __m256 s = _mm256_load_ps(ab + t * 8);                                      \
                        const float *xa_ = xs[t] + at;                                              \
                        s = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xa_ + 0), s);                        \
                        s = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xa_ + 8), s);                        \
                        s = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xa_ + 16), s);                       \
                        s = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xa_ + 24), s);                       \
                        _mm256_store_ps(ab + t * 8, s);                                              \
                    }                                                                               \
                }                                                                                   \
                qs += 8;                                                                            \
                sg += 4;                                                                            \
            }                                                                                       \
        }                                                                                           \
        for (int t = 0; t < (T); t++)                                                              \
            ys[t][row] = qe_hsum(_mm256_load_ps(ab + t * 8));                                      \
    }

QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park9, 9)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park10, 10)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park11, 11)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park12, 12)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park13, 13)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park14, 14)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park15, 15)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park16, 16)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park20, 20)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park24, 24)
QE_IQ3_S_PARK_FORM(qe_row_iq3_s_park32, 32)

/* Eight to eleven tokens of one bucket in one walk, with the first eight sums in
 * registers and only the residue parked.
 *
 * The form above pays a load and a store per token per 32 weights for every token it is
 * handed, including the eight whose sums would have seated in registers. At ten tokens
 * that is twenty memory ops a group bought to save one walk of the quantiser, which is
 * the difference between the probe that liked parking (+32 % at twelve tokens in
 * isolation) and the engine that did not (ideas.md §7.2b: a fifth of the stage lost by
 * sixteen). A bucket of ten wants eight of its rows in registers and two of them in L1.
 *
 * What that changes per group of 32 weights: a register row costs four activation loads
 * and four FMAs, where the parked walk costs four loads, one accumulator load, one store
 * and four FMAs, and the eight register rows are not paying for an L1 line to hold the
 * block either, because the block is four registers here rather than a line loaded twice
 * per token. So the eight rows this form shares with `multi8` are cheaper as well as the
 * tail it adds: eight loads a group per token there, four here.
 *
 * R=8 with P=2 is fourteen live vectors in a sixteen-register machine, which is the
 * reason P stops at eight: the three dequant constants are dead across the FMA loops and
 * get rematerialised, and past fourteen the accumulator of a register row is what the
 * allocator moves instead, and that is the failure the run-time-T form below has. The
 * sizes are instantiated rather than looped over for the same reason as above, and
 * `make fast`'s multi check reaches every group size the dispatch can ask for. */
#define QE_IQ3_S_MIX_FORM(name, R, P)                                                              \
    static void name(const qe_block_iq3_s *restrict w, const float *const *restrict xs,            \
                     float *const *restrict ys, long row, long nb) {                                \
        __attribute__((aligned(32))) float dbv[8];                                                 \
        __attribute__((aligned(32))) float ab[((P) ? (P) : 1) * 8];                                \
        __m256 a[R];                                                                               \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                            \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,          \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,          \
                                                3, 3, 3, 3);                                       \
        const __m256i one8 = _mm256_set1_epi8(1);                                                   \
        for (int t = 0; t < (R); t++)                                                              \
            a[t] = _mm256_setzero_ps();                                                            \
        for (int t = 0; t < (P); t++)                                                              \
            _mm256_store_ps(ab + t * 8, _mm256_setzero_ps());                                      \
        for (long b = 0; b < nb; b++) {                                                            \
            qe_iq3_block_scales(w, b, dbv);                                                        \
            const uint8_t *qs = w[b].qs, *sg = w[b].signs;                                           \
            for (int g = 0; g < 8; g++) {                                                          \
                const long at = b * 256 + g * 32;                                                  \
                {                                                                                  \
                    QE_QUAD_IQ3_S(qs, w[b].qh[g], sg, dbv[g], bitv, spread, one8);                  \
                    for (int t = 0; t < (R); t++)                                                   \
                        QE_FMA4_SCALED(a[t], xs[t] + at);                                           \
                    for (int t = 0; t < (P); t++) {                                                 \
                        __m256 s = _mm256_load_ps(ab + t * 8);                                      \
                        const float *xa_ = xs[(R) + t] + at;                                        \
                        s = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xa_ + 0), s);                        \
                        s = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xa_ + 8), s);                        \
                        s = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xa_ + 16), s);                       \
                        s = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xa_ + 24), s);                       \
                        _mm256_store_ps(ab + t * 8, s);                                              \
                    }                                                                               \
                }                                                                                   \
                qs += 8;                                                                            \
                sg += 4;                                                                            \
            }                                                                                       \
        }                                                                                           \
        for (int t = 0; t < (R); t++)                                                               \
            ys[t][row] = qe_hsum(a[t]);                                                             \
        for (int t = 0; t < (P); t++)                                                               \
            ys[(R) + t][row] = qe_hsum(_mm256_load_ps(ab + t * 8));                                 \
    }

QE_IQ3_S_MIX_FORM(qe_row_iq3_s_mix8, 8, 0)
QE_IQ3_S_MIX_FORM(qe_row_iq3_s_mix9, 8, 1)
QE_IQ3_S_MIX_FORM(qe_row_iq3_s_mix10, 8, 2)
QE_IQ3_S_MIX_FORM(qe_row_iq3_s_mix11, 8, 3)

/* The same pairing for iq3_s, where it is worth more than it is worth above: this format
 * derives eight 256-entry grid values per 32 weights, and a pair of rows halves that per
 * (row, token) product as well as halving the activation loads. Register pressure is the
 * same wall as everywhere else in this file - a group of eight is sixteen sums beside eight
 * weight vectors, and the allocator takes the overflow out of the sums - and it still wins
 * by enough to be the default (`make walk`, `make moe`). */
#define QE_IQ3_S_MULTI2R_FORM(name, T)                                                             \
    static void name(const qe_block_iq3_s *restrict w0, const qe_block_iq3_s *restrict w1,          \
                     const float *const *restrict xs, float *const *restrict ys,                    \
                     long row0, long row1, long nb) {                                                \
        __attribute__((aligned(32))) float dbv0[8], dbv1[8];                                          \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                              \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                  \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,            \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,            \
                                                3, 3, 3, 3);                                         \
        const __m256i one8 = _mm256_set1_epi8(1);                                                     \
        __m256 a0[T], a1[T];                                                                          \
        for (int t = 0; t < (T); t++) {                                                              \
            a0[t] = _mm256_setzero_ps();                                                             \
            a1[t] = _mm256_setzero_ps();                                                             \
        }                                                                                            \
        for (long b = 0; b < nb; b++) {                                                              \
            qe_iq3_block_scales(w0, b, dbv0);                                                        \
            qe_iq3_block_scales(w1, b, dbv1);                                                        \
            const uint8_t *qs0 = w0[b].qs, *sg0 = w0[b].signs;                                         \
            const uint8_t *qs1 = w1[b].qs, *sg1 = w1[b].signs;                                         \
            for (int g = 0; g < 8; g++) {                                                            \
                const long at = b * 256 + g * 32;                                                    \
                __m256 q00, q01, q02, q03, q10, q11, q12, q13;                                        \
                {                                                                                    \
                    QE_QUAD_IQ3_S(qs0, w0[b].qh[g], sg0, dbv0[g], bitv, spread, one8);                 \
                    q00 = q0;                                                                        \
                    q01 = q1;                                                                        \
                    q02 = q2;                                                                        \
                    q03 = q3;                                                                        \
                }                                                                                    \
                {                                                                                    \
                    QE_QUAD_IQ3_S(qs1, w1[b].qh[g], sg1, dbv1[g], bitv, spread, one8);                 \
                    q10 = q0;                                                                        \
                    q11 = q1;                                                                        \
                    q12 = q2;                                                                        \
                    q13 = q3;                                                                        \
                }                                                                                    \
                for (int t = 0; t < (T); t++) {                                                      \
                    const float *xa_ = xs[t] + at;                                                   \
                    const __m256 x0 = _mm256_loadu_ps(xa_ + 0);                                       \
                    const __m256 x1 = _mm256_loadu_ps(xa_ + 8);                                       \
                    const __m256 x2 = _mm256_loadu_ps(xa_ + 16);                                      \
                    const __m256 x3 = _mm256_loadu_ps(xa_ + 24);                                      \
                    a0[t] = _mm256_fmadd_ps(q00, x0, a0[t]);                                          \
                    a0[t] = _mm256_fmadd_ps(q01, x1, a0[t]);                                          \
                    a0[t] = _mm256_fmadd_ps(q02, x2, a0[t]);                                          \
                    a0[t] = _mm256_fmadd_ps(q03, x3, a0[t]);                                          \
                    a1[t] = _mm256_fmadd_ps(q10, x0, a1[t]);                                          \
                    a1[t] = _mm256_fmadd_ps(q11, x1, a1[t]);                                          \
                    a1[t] = _mm256_fmadd_ps(q12, x2, a1[t]);                                          \
                    a1[t] = _mm256_fmadd_ps(q13, x3, a1[t]);                                          \
                }                                                                                    \
                qs0 += 8;                                                                            \
                sg0 += 4;                                                                            \
                qs1 += 8;                                                                            \
                sg1 += 4;                                                                            \
            }                                                                                        \
        }                                                                                            \
        for (int t = 0; t < (T); t++) {                                                              \
            ys[t][row0] = qe_hsum(a0[t]);                                                            \
            ys[t][row1] = qe_hsum(a1[t]);                                                            \
        }                                                                                            \
    }

QE_IQ3_S_MULTI2R_FORM(qe_row_iq3_s_multi2r8, 8)

/* Four parked rows is where this stops: at R=8,P=4 the allocator has taken the four
 * accumulator registers it wants out of the eight, and the assembly it emits for a group
 * of twelve is the assembly the parked form emits for a group of twelve, register names
 * aside. Measured on `-O3 -march=native` by counting the ymm stores to the frame: nine
 * and ten rows cost two more than the eight-row walk, twelve costs ten more, and past
 * that the two forms are the same function. So the sizes above are the family, and a
 * wider bucket parks the way it always did. */

/* ------------------------------------------------------------- integer forms
 *
 * The two shared-dequant walks above with the chunk's activations recoded to int8: four
 * converts and four multiplies per 32 weights stop existing for every token row a block
 * serves, because vpdpbusd spends the codebook value against the activation's int8 value
 * and the group's scale is applied once, in the exit, for the whole block. The decode is
 * paid once per block either way, so what leaves the walk is the floating-point half of
 * what a token row pays for a block - ideas.md 7.10's instrument says that half is worth
 * 1.36x of the walk on the shape the routed experts have, and it is the half the grid
 * lookups were never the wall for.
 *
 * Both codebooks fit vpdpbusd without help: iq3_s magnitudes are odd and under sixteen,
 * iq4_nl's table is int8, and an xor of the sign bit is the exact +128 the unsigned
 * operand carries. The correction that owes is then the panel's - one number per token
 * per 32 activations, shared by every output row of the walk - which is why the bias goes
 * on the weights and not on the activations (include/qe.h, qe_i8panel).
 *
 * T is a literal for the usual reason: a run-time T puts the accumulators on the stack,
 * and the walk is then slower than the one it replaced. */
#define QE_I8_EXIT(acc, d, sa, pv, dot)                                                            \
    (acc) = _mm256_fmadd_ps(_mm256_mul_ps((d), (sa)),                                              \
                            _mm256_sub_ps(_mm256_cvtepi32_ps(dot), (pv)), (acc))

/* The panel cursor, scale cursor and bias seed cursor of the walk's t-th token row. The
 * panel is contiguous - token row t is panel row t, `s8` bytes and `sc` values down - so
 * that the walk needs three registers for the whole panel rather than three arrays of
 * eight, which is the difference between a group that runs in registers and one whose
 * cursors come back from the frame every 32 weights (measured: 472 GMAC/s against 497 for
 * the walk this replaces). */
#define QE_I8_PANEL(T)                                                                             \
    const int8_t *const x8b = a->x8 + (size_t)p0 * (size_t)a->stride8;                               \
    const float *const spb = a->s + (size_t)p0 * (size_t)a->nb;                                        \
    const int32_t *const ppb = a->p + (size_t)p0 * (size_t)a->nb;                                       \
    const long s8 = a->stride8, sc = a->nb;

#define QE_IQ4_NL_I8_FORM(name, T, SSD, RS)                                                           \
    static void name(const qe_block_iq4_nl *restrict w, const qe_i8panel *restrict a, int p0,       \
                     float *const *restrict ys, long row, long nb) {                                \
        __m256 acc[T];                                                                             \
        QE_I8_PANEL(T);                                                                            \
        const __m256i m80 = _mm256_set1_epi8((char)0x80);                                          \
        const __m256i lut = qe_iq4_lut();                                                          \
        for (int t = 0; t < T; t++)                                                                \
            acc[t] = _mm256_setzero_ps();                                                          \
        for (long b = 0; b < nb; b++) {                                                            \
            const __m256i v = _mm256_broadcastsi128_si256(                                          \
                _mm_loadu_si128((const __m128i *)w[b].qs));                                          \
            const __m256i u = QE_I8_BIAS((SSD), qe_iq4_expand(v, lut), m80);                            \
            const __m256 d = _mm256_set1_ps(qe_half_to_f32(w[b].d));                                   \
            for (int t = 0; t < T; t++) {                                                             \
                const long so = (size_t)t * (size_t)sc;                                                \
                const __m256i pv = _mm256_set1_epi32(ppb[so + b]);                                      \
                const __m256i dot = QE_I8_DOT(                                                           \
                    (SSD), u,                                                                            \
                    _mm256_loadu_si256((const __m256i *)(x8b + (size_t)t * (size_t)s8 + 32 * b)), pv);    \
                /* With one activation scale per row (`RS`, `qe_quant_q8_0_row_rs`) the product    */  \
                /* of the two scales is no longer a function of the block, so the accumulator      */  \
                /* runs in "weight scale" space and the row's scale multiplies it once at the      */  \
                /* end. Two of the four operands an output row owes per 32 weights go with it.     */  \
                acc[t] = (RS) ? _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(dot), acc[t])                    \
                              : _mm256_fmadd_ps(_mm256_mul_ps(d, _mm256_set1_ps(spb[so + b])),            \
                                                _mm256_cvtepi32_ps(dot), acc[t]);                        \
            }                                                                                         \
        }                                                                                             \
        if (RS)                                                                                         \
            for (int t = 0; t < T; t++)                                                                   \
                acc[t] = _mm256_mul_ps(acc[t], _mm256_set1_ps(spb[(size_t)t * (size_t)sc]));              \
        for (int t = 0; t < T; t++)                                                                    \
            ys[t][row] = qe_hsum(acc[t]);                                                               \
    }

QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_1, 1, 0, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_2, 2, 0, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_4, 4, 0, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_8, 8, 0, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_10, 10, 0, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_12, 12, 0, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_16, 16, 0, 0)
#if defined(__AVXVNNIINT8__)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_1_ssd, 1, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_2_ssd, 2, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_4_ssd, 4, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_8_ssd, 8, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_10_ssd, 10, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_12_ssd, 12, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_16_ssd, 16, 1, 0)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_1_rs, 1, 1, 1)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_2_rs, 2, 1, 1)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_4_rs, 4, 1, 1)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_8_rs, 8, 1, 1)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_10_rs, 10, 1, 1)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_12_rs, 12, 1, 1)
QE_IQ4_NL_I8_FORM(qe_row_iq4_nl_i8_16_rs, 16, 1, 1)
#endif

/* Two output rows of one expert matrix against one load of a token's activation block.
 *
 * The single-row walk above reads the panel once per output row: at a group of eight tokens
 * and K=2560 that is 20 KiB of panel per 256-byte block of weights walked, and the exit the
 * row owes - a convert, a multiply by the two scales, an FMA - is paid per row and per token.
 * Holding two dequantised weight blocks beside one set of activation vectors takes half the
 * panel loads and half the `pv`/`sa` broadcasts per dot, which is the load-per-multiply item
 * ideas.md 7.10 ends on and 7.11 leaves as the one thing the integer side still owns.
 *
 * It is not a change of number: each row's accumulator keeps the single-row form's chain -
 * the same seed, the same block order, the same exit - so a pair walked together leaves the
 * bits two walks left, which is what the fourth section of `make walk` holds it to instead of
 * a tolerance. Two rows is the ceiling: against the eight-token group that is sixteen
 * accumulators plus the two weight blocks, the two scales and the shared exit operands, and
 * the register file is what it is (`QE_I8_R=1` is the knob that says so on a shape). Only the
 * eights of a group take this form, for the same reason the shared-dequant pairing takes only
 * the eights: a group's leftover tokens go through the forms they have today, once per row. */
#define QE_IQ4_NL_I8_R2_FORM(name, T, SSD)                                                               \
    static void name(const qe_block_iq4_nl *restrict w0, const qe_block_iq4_nl *restrict w1,             \
                     const qe_i8panel *restrict a, int p0, float *const *restrict ys,                    \
                     long row0, long row1, long nb) {                                                    \
        __m256 acc0[T], acc1[T];                                                                       \
        QE_I8_PANEL(T);                                                                                \
        const __m256i m80 = _mm256_set1_epi8((char)0x80);                                              \
        const __m256i lut = qe_iq4_lut();                                                              \
        for (int t = 0; t < T; t++) {                                                                  \
            acc0[t] = _mm256_setzero_ps();                                                             \
            acc1[t] = _mm256_setzero_ps();                                                             \
        }                                                                                              \
        for (long b = 0; b < nb; b++) {                                                                \
            const __m256i v0 =                                                                         \
                _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)w0[b].qs));                 \
            const __m256i v1 =                                                                         \
                _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)w1[b].qs));                 \
            const __m256i u0 = QE_I8_BIAS((SSD), qe_iq4_expand(v0, lut), m80);                            \
            const __m256i u1 = QE_I8_BIAS((SSD), qe_iq4_expand(v1, lut), m80);                            \
            const __m256 d0 = _mm256_set1_ps(qe_half_to_f32(w0[b].d));                                   \
            const __m256 d1 = _mm256_set1_ps(qe_half_to_f32(w1[b].d));                                   \
            for (int t = 0; t < T; t++) {                                                              \
                const long so = (size_t)t * (size_t)sc;                                                \
                const __m256i pv = _mm256_set1_epi32(ppb[so + b]);                                      \
                const __m256 sa = _mm256_set1_ps(spb[so + b]);                                          \
                const __m256i xv = _mm256_loadu_si256(                                                     \
                    (const __m256i *)(x8b + (size_t)t * (size_t)s8 + 32 * b));                            \
                const __m256i dot0 = QE_I8_DOT((SSD), u0, xv, pv);                                         \
                acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(d0, sa), _mm256_cvtepi32_ps(dot0), acc0[t]);       \
                const __m256i dot1 = QE_I8_DOT((SSD), u1, xv, pv);                                         \
                acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(d1, sa), _mm256_cvtepi32_ps(dot1), acc1[t]);       \
            }                                                                                          \
        }                                                                                              \
        for (int t = 0; t < T; t++) {                                                                  \
            ys[t][row0] = qe_hsum(acc0[t]);                                                            \
            ys[t][row1] = qe_hsum(acc1[t]);                                                            \
        }                                                                                              \
    }

QE_IQ4_NL_I8_R2_FORM(qe_row_iq4_nl_i8_r2t8, 8, 0)
#if defined(__AVXVNNIINT8__)
QE_IQ4_NL_I8_R2_FORM(qe_row_iq4_nl_i8_r2t8_ssd, 8, 1)
#endif

/* One 32-weight group of an iq3_s block as the int8 the integer walks spend: the grid
 * lookup and the sign pass `QE_QUAD_IQ3_S` does, and then the exact xor of the sign bit that
 * leaves vpdpbusd's unsigned operand carrying a +128 the quantiser owes back. Assigns to
 * `out`, which the caller declares, so that a walk holding more than one group at a time -
 * a tile of two output rows, one per row - can name them. Both integer walks below take
 * their codebook from here, which is what keeps a single-row walk and a tile the same
 * arithmetic rather than two decodes that happen to agree today. */
#define QE_I8_U_IQ3_S(out, qs, hb, sg, m80, bitv, spread, one8, SSD)                                     \
    do {                                                                                              \
        const __m256i mag_ = qe_iq3_group_loads((qs), (hb));                                          \
        const __m256i sb_ = _mm256_shuffle_epi8(                                                      \
            _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i *)(sg))), (spread));           \
        const __m256i neg_ = _mm256_cmpeq_epi8(_mm256_and_si256(sb_, (bitv)), (bitv));                \
        (out) = QE_I8_BIAS((SSD),                                                                     \
                           _mm256_sign_epi8(mag_, _mm256_or_si256(neg_, (one8))), (m80));            \
    } while (0)

/* iq3_s through the same exit: the grid lookups and the sign pass are the walk of the
 * codebook the f32 form already does, and what this stops doing is the four converts and
 * four multiplies that turned its result into floats. The group scale is dbv[g], which
 * `qe_iq3_block_scales` already produces for the whole block in one pass. */
#define QE_IQ3_S_I8_FORM(name, T, SSD, RS)                                                           \
    static void name(const qe_block_iq3_s *restrict w, const qe_i8panel *restrict a, int p0,         \
                     float *const *restrict ys, long row, long nb) {                                 \
        __attribute__((aligned(32))) float dbv[8];                                                 \
        __m256 acc[T];                                                                             \
        QE_I8_PANEL(T);                                                                            \
        const __m256i m80 = _mm256_set1_epi8((char)0x80);                                          \
        const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                             \
            1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                  \
        const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,            \
                                                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,            \
                                                3, 3, 3, 3);                                          \
        const __m256i one8 = _mm256_set1_epi8(1);                                                     \
        for (int t = 0; t < T; t++)                                                                  \
            acc[t] = _mm256_setzero_ps();                                                             \
        for (long b = 0; b < nb; b++) {                                                              \
            qe_iq3_block_scales(w, b, dbv);                                                           \
            const uint8_t *qs = w[b].qs, *sg = w[b].signs;                                             \
            for (int g = 0; g < 8; g++) {                                                              \
                const long at = b * 256 + g * 32;                                                     \
                __m256i u;                                                                            \
                QE_I8_U_IQ3_S(u, qs, w[b].qh[g], sg, m80, bitv, spread, one8, (SSD));                    \
                const __m256 d = _mm256_set1_ps(dbv[g]);                                                 \
                for (int t = 0; t < T; t++) {                                                             \
                    const long o = (size_t)t * (size_t)sc + b * 8 + g;                                     \
                    const __m256i pv = _mm256_set1_epi32(ppb[o]);                                           \
                    const __m256i dot = QE_I8_DOT(                                                         \
                        (SSD), u,                                                                            \
                        _mm256_loadu_si256(                                                                    \
                            (const __m256i *)(x8b + (size_t)t * (size_t)s8 + at)), pv);                      \
                    /* `RS`: one activation scale for the row, so the accumulator runs in           */\
                    /* "weight scale" space and the row's scale multiplies it once below the         */\
                    /* block loop, and a broadcast and a multiply per token per 32 weights stop       */\
                    /* existing. Same argument and same shape as the iq4_nl form above.              */\
                    acc[t] = (RS)                                                                       \
                        ? _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(dot), acc[t])                           \
                        : _mm256_fmadd_ps(_mm256_mul_ps(d, _mm256_set1_ps(spb[o])),                     \
                                          _mm256_cvtepi32_ps(dot), acc[t]);                             \
                }                                                                                            \
                qs += 8;                                                                                \
                sg += 4;                                                                                \
            }                                                                                           \
        }                                                                                               \
        if (RS)                                                                                         \
            for (int t = 0; t < T; t++)                                                                 \
                acc[t] = _mm256_mul_ps(acc[t], _mm256_set1_ps(spb[(size_t)t * (size_t)sc]));           \
        for (int t = 0; t < T; t++)                                                                     \
            ys[t][row] = qe_hsum(acc[t]);                                                                \
    }

QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_1, 1, 0, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_2, 2, 0, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_4, 4, 0, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_8, 8, 0, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_10, 10, 0, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_12, 12, 0, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_16, 16, 0, 0)
#if defined(__AVXVNNIINT8__)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_1_ssd, 1, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_2_ssd, 2, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_4_ssd, 4, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_8_ssd, 8, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_10_ssd, 10, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_12_ssd, 12, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_16_ssd, 16, 1, 0)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_1_rs, 1, 1, 1)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_2_rs, 2, 1, 1)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_4_rs, 4, 1, 1)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_8_rs, 8, 1, 1)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_10_rs, 10, 1, 1)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_12_rs, 12, 1, 1)
QE_IQ3_S_I8_FORM(qe_row_iq3_s_i8_16_rs, 16, 1, 1)
#endif

/* The iq3_s tile: two rows' blocks decoded beside one another and spent against one load of
 * a token's activation group, so the decode is paid once for two output rows as well as the
 * panel load being. This is the format where the sharing reaches furthest - an iq3_s group
 * costs a grid lookup, a sign pass and a broadcast to produce, and the walk spent that per
 * row per token group before - and it is also the format where it costs most in registers,
 * because `mag`/`sb`/`neg` of two rows are live at the same time. Sixteen accumulators is
 * the ceiling this family has room for, so it is the eight-token group that takes it and the
 * narrower groups that stay on the single-row forms above, exactly as the shared-dequant
 * pairing leaves them (same reasons, same bits: `make walk`'s fourth section). */
#define QE_IQ3_S_I8_R2_FORM(name, T, SSD)                                                               \
 static void name(const qe_block_iq3_s *restrict w0, const qe_block_iq3_s *restrict w1,                \
     const qe_i8panel *restrict a, int p0, float *const *restrict ys,                    \
     long row0, long row1, long nb) {                                                     \
  __attribute__((aligned(32))) float dbv0[8], dbv1[8];                                            \
  __m256 acc0[T], acc1[T];                                                                       \
  QE_I8_PANEL(T);                                                                                \
  const __m256i m80 = _mm256_set1_epi8((char)0x80);                                              \
  const __m256i bitv = _mm256_broadcastsi128_si256(_mm_setr_epi8(                                 \
   1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128));                    \
  const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,              \
            1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,              \
            3, 3, 3, 3);                                            \
  const __m256i one8 = _mm256_set1_epi8(1);                                                      \
  for (int t = 0; t < T; t++) {                                                                  \
   acc0[t] = _mm256_setzero_ps();                                                             \
   acc1[t] = _mm256_setzero_ps();                                                             \
  }                                                                                              \
  for (long b = 0; b < nb; b++) {                                                                \
   qe_iq3_block_scales(w0, b, dbv0);                                                          \
   qe_iq3_block_scales(w1, b, dbv1);                                                          \
   const uint8_t *qs0 = w0[b].qs, *sg0 = w0[b].signs;                                         \
   const uint8_t *qs1 = w1[b].qs, *sg1 = w1[b].signs;                                         \
   for (int g = 0; g < 8; g++) {                                                              \
    const long at = b * 256 + g * 32;                                                      \
    __m256i u0, u1;                                                                        \
    QE_I8_U_IQ3_S(u0, qs0, w0[b].qh[g], sg0, m80, bitv, spread, one8, (SSD));              \
    QE_I8_U_IQ3_S(u1, qs1, w1[b].qh[g], sg1, m80, bitv, spread, one8, (SSD));              \
    const __m256 d0 = _mm256_set1_ps(dbv0[g]);                                             \
    const __m256 d1 = _mm256_set1_ps(dbv1[g]);                                             \
    for (int t = 0; t < T; t++) {                                                          \
     const long o = (size_t)t * (size_t)sc + b * 8 + g;                                 \
     const __m256i pv = _mm256_set1_epi32(ppb[o]);                                      \
     const __m256 sa = _mm256_set1_ps(spb[o]);                                          \
     const __m256i xv =                                                                 \
      _mm256_loadu_si256((const __m256i *)(x8b + (size_t)t * (size_t)s8 + at));      \
     const __m256i dot0 = QE_I8_DOT((SSD), u0, xv, pv);                                 \
     acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(d0, sa), _mm256_cvtepi32_ps(dot0),          \
           acc0[t]);                                                \
     const __m256i dot1 = QE_I8_DOT((SSD), u1, xv, pv);                                 \
     acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(d1, sa), _mm256_cvtepi32_ps(dot1),          \
           acc1[t]);                                                \
    }                                                                                      \
    qs0 += 8;                                                                              \
    sg0 += 4;                                                                              \
    qs1 += 8;                                                                              \
    sg1 += 4;                                                                              \
   }                                                                                          \
  }                                                                                              \
  for (int t = 0; t < T; t++) {                                                                  \
   ys[t][row0] = qe_hsum(acc0[t]);                                                            \
   ys[t][row1] = qe_hsum(acc1[t]);                                                            \
  }                                                                                              \
 }

QE_IQ3_S_I8_R2_FORM(qe_row_iq3_s_i8_r2t8, 8, 0)
#if defined(__AVXVNNIINT8__)
QE_IQ3_S_I8_R2_FORM(qe_row_iq3_s_i8_r2t8_ssd, 8, 1)
#endif

/* Which of the two dots a process walks with, indexed rather than branched on: the forms
* above are the same walk twice, once against `vpdpbusd` and its bias, once against
* `vpdpbssd` and neither, and the table is what keeps the dispatchers below readable while
* they pick one. Where the compiler has no signed instruction the second row repeats the
* first and `qe_i8_ssd()` cannot answer 1. */
typedef void (*qe_iq4_i8_fn)(const qe_block_iq4_nl *, const qe_i8panel *, int, float *const *, long,
       long);
typedef void (*qe_iq3_i8_fn)(const qe_block_iq3_s *, const qe_i8panel *, int, float *const *, long,
       long);
typedef void (*qe_iq4_i8_r2_fn)(const qe_block_iq4_nl *, const qe_block_iq4_nl *,
        const qe_i8panel *, int, float *const *, long, long, long);
typedef void (*qe_iq3_i8_r2_fn)(const qe_block_iq3_s *, const qe_block_iq3_s *,
        const qe_i8panel *, int, float *const *, long, long, long);

/* The token widths this family has a form for, widest first.
 *
 * A pass over one output row pays the codebook decode of every block of that row and one dot
 * per token, so the tokens are the cheap part of a pass and the passes are not. The family
 * stopped at eight tokens while the buckets it is handed are ten wide on average - a
 * 512-token chunk routes 5120 slots over 512 experts - which left every bucket between nine
 * and sixteen paying for its expert's weights twice: `make walk`'s group section read 12.8 ms
 * for eight token rows against 32.1 for eleven. Ten, twelve and sixteen are the widths that
 * closes, and `qe_i8_tmax()` says how far up the table a walk may reach. */
enum { QE_I8_C16 = 0, QE_I8_C12, QE_I8_C10, QE_I8_C8, QE_I8_C4, QE_I8_C2, QE_I8_NCLASS };
static const signed char qe_i8_class_tokens[QE_I8_NCLASS] = { 16, 12, 10, 8, 4, 2 };

/* The one-token walk is not a class: no division of a group of two or more wants it, because a
 * leftover token is cheaper inside the two-token form with its neighbour (six operands of decode
 * against nine) and the plan table below says so. It exists for the walk a decode step hands
 * these functions - one token row, every expert of one layer - which has no neighbour to borrow
 * and is the only group the family could not walk until it had it. It sits past the classes so the
 * plan, which is indexed by class, cannot reach it. */
#define QE_I8_ONE (QE_I8_NCLASS)

/* What a pass costs, in the only model the plan needs: the six operands a 32-weight block's
 * decode is and the three a token's dot, convert and FMA are. A group is then divided by the
 * division this says is cheapest, which is not the greedy one - a group of twenty-two is two
 * passes of twelve and ten, and widest-first is three. */
#define QE_I8_PASS_COST 6
#define QE_I8_TOKEN_COST 3
#define QE_I8_PLAN_MAX 1024

/* First class of the cheapest division of a group of n tokens, one row of the table per widest
 * class a walk may open. Every row is built at once, because these walks run on nineteen threads
 * and an instrument that moved the ceiling between two runs would otherwise have a walk reading a
 * plan another thread is rebuilding - which is not a wrong number but a class wider than the
 * group it is dividing. A row is therefore prefilled with the two-token class, which is a valid
 * division of anything, before the cheapest one is written over it: a reader that arrives mid-build
 * walks badly and stays inside the panel.
 *
 * Rows are indexed by the widest class, so the row a walk reads is the class index of its
 * ceiling; classes descend, so that is the first one no wider than the ceiling. */
static signed char qe_i8_plan_first[QE_I8_NCLASS][QE_I8_PLAN_MAX + 1];
static int qe_i8_plan_built = 0;

static void qe_i8_plan_build(void) {
    static int cost[QE_I8_PLAN_MAX + 1];
    for (int c = 0; c < QE_I8_NCLASS; c++)
        for (int n = 0; n <= QE_I8_PLAN_MAX; n++)
            qe_i8_plan_first[c][n] = QE_I8_C2;
    for (int c = 0; c < QE_I8_NCLASS; c++) {
        qe_i8_plan_first[c][0] = -1;
        cost[0] = 0;
        for (int n = 1; n <= QE_I8_PLAN_MAX; n++) {
            int best = 1 << 30;
            signed char which = QE_I8_C2;
            for (int d = c; d < QE_I8_NCLASS; d++) {
                const int w = qe_i8_class_tokens[d];
                if (w > n)
                    continue;
                const int rest = n - w;
                /* A leftover single token is not a pass of its own: it takes the two-token form
                 * with the token before it, which is one token recomputed and the same number
                 * written twice. */
                const int tail =
                    rest == 1 ? QE_I8_PASS_COST + 2 * QE_I8_TOKEN_COST : cost[rest];
                const int tot = QE_I8_PASS_COST + w * QE_I8_TOKEN_COST + tail;
                if (tot < best) {
                    best = tot;
                    which = (signed char)d;
                }
            }
            cost[n] = best;
            qe_i8_plan_first[c][n] = which;
        }
    }
    qe_i8_plan_built = 1;
}

/* The row of the plan table a ceiling reads, and the class a walk of `maxc` tokens or fewer
 * opens a pass on. */
static int qe_i8_class_of(int maxc) {
    for (int c = 0; c < QE_I8_NCLASS; c++)
        if (qe_i8_class_tokens[c] <= maxc)
            return c;
    return QE_I8_C2;
}

static const qe_iq4_i8_fn qe_iq4_i8_forms[2][QE_I8_NCLASS + 1] = {
 { qe_row_iq4_nl_i8_16, qe_row_iq4_nl_i8_12, qe_row_iq4_nl_i8_10, qe_row_iq4_nl_i8_8,
   qe_row_iq4_nl_i8_4, qe_row_iq4_nl_i8_2, qe_row_iq4_nl_i8_1 },
#if defined(__AVXVNNIINT8__)
 { qe_row_iq4_nl_i8_16_ssd, qe_row_iq4_nl_i8_12_ssd, qe_row_iq4_nl_i8_10_ssd,
   qe_row_iq4_nl_i8_8_ssd, qe_row_iq4_nl_i8_4_ssd, qe_row_iq4_nl_i8_2_ssd,
   qe_row_iq4_nl_i8_1_ssd },
#else
 { qe_row_iq4_nl_i8_16, qe_row_iq4_nl_i8_12, qe_row_iq4_nl_i8_10, qe_row_iq4_nl_i8_8,
   qe_row_iq4_nl_i8_4, qe_row_iq4_nl_i8_2, qe_row_iq4_nl_i8_1 },
#endif
};

/* The same walks against a panel with one activation scale per row. Only the signed dot
 * has them - the unsigned one still owes the seed per block, so the row scale would save
 * its multiply and keep its broadcast - and `qe_i8_rowscale()` cannot answer 1 where the
 * signed instruction does not exist, which is why the fallback row is never read. */
static const qe_iq4_i8_fn qe_iq4_i8_rs_forms[QE_I8_NCLASS + 1] = {
#if defined(__AVXVNNIINT8__)
 qe_row_iq4_nl_i8_16_rs, qe_row_iq4_nl_i8_12_rs, qe_row_iq4_nl_i8_10_rs, qe_row_iq4_nl_i8_8_rs,
 qe_row_iq4_nl_i8_4_rs, qe_row_iq4_nl_i8_2_rs, qe_row_iq4_nl_i8_1_rs
#else
 qe_row_iq4_nl_i8_16, qe_row_iq4_nl_i8_12, qe_row_iq4_nl_i8_10, qe_row_iq4_nl_i8_8,
 qe_row_iq4_nl_i8_4, qe_row_iq4_nl_i8_2, qe_row_iq4_nl_i8_1
#endif
};

static const qe_iq3_i8_fn qe_iq3_i8_forms[2][QE_I8_NCLASS + 1] = {
 { qe_row_iq3_s_i8_16, qe_row_iq3_s_i8_12, qe_row_iq3_s_i8_10, qe_row_iq3_s_i8_8,
   qe_row_iq3_s_i8_4, qe_row_iq3_s_i8_2, qe_row_iq3_s_i8_1 },
#if defined(__AVXVNNIINT8__)
 { qe_row_iq3_s_i8_16_ssd, qe_row_iq3_s_i8_12_ssd, qe_row_iq3_s_i8_10_ssd, qe_row_iq3_s_i8_8_ssd,
   qe_row_iq3_s_i8_4_ssd, qe_row_iq3_s_i8_2_ssd, qe_row_iq3_s_i8_1_ssd },
#else
 { qe_row_iq3_s_i8_16, qe_row_iq3_s_i8_12, qe_row_iq3_s_i8_10, qe_row_iq3_s_i8_8,
   qe_row_iq3_s_i8_4, qe_row_iq3_s_i8_2, qe_row_iq3_s_i8_1 },
#endif
};

static const qe_iq3_i8_fn qe_iq3_i8_rs_forms[QE_I8_NCLASS + 1] = {
#if defined(__AVXVNNIINT8__)
 qe_row_iq3_s_i8_16_rs, qe_row_iq3_s_i8_12_rs, qe_row_iq3_s_i8_10_rs, qe_row_iq3_s_i8_8_rs,
 qe_row_iq3_s_i8_4_rs, qe_row_iq3_s_i8_2_rs, qe_row_iq3_s_i8_1_rs
#else
 qe_row_iq3_s_i8_16, qe_row_iq3_s_i8_12, qe_row_iq3_s_i8_10, qe_row_iq3_s_i8_8,
 qe_row_iq3_s_i8_4, qe_row_iq3_s_i8_2, qe_row_iq3_s_i8_1
#endif
};

static const qe_iq4_i8_r2_fn qe_iq4_i8_r2_forms[2] = {
 qe_row_iq4_nl_i8_r2t8,
#if defined(__AVXVNNIINT8__)
 qe_row_iq4_nl_i8_r2t8_ssd,
#else
 qe_row_iq4_nl_i8_r2t8,
#endif
};

static const qe_iq3_i8_r2_fn qe_iq3_i8_r2_forms[2] = {
 qe_row_iq3_s_i8_r2t8,
#if defined(__AVXVNNIINT8__)
 qe_row_iq3_s_i8_r2t8_ssd,
#else
 qe_row_iq3_s_i8_r2t8,
#endif
};

/* Walk tokens [p0, p0+n) of one output row, divided by the plan for the widest class allowed.
 * A leftover token takes the two-token form with the token before it, which recomputes a token
 * this row already has and writes the same number over it: the alternatives are a one-token
 * form, which shares nothing, or a call back to the f32 walk, whose activation rows this
 * walk's caller does not have. A group of one therefore arrives only as the whole of a walk -
 * the one-token walk at the bottom of this file, where there is no row before it to borrow -
 * and a tile's tail always has the tile's own rows before it. */
static void qe_iq4_i8_row(const qe_iq4_i8_fn *const forms, const qe_block_iq4_nl *restrict w,
                          const qe_i8panel *restrict a, int p0, int n,
                          float *const *restrict ys, long row, long nb, int maxc) {
    if (!qe_i8_plan_built)
        qe_i8_plan_build();
    const int cmax = qe_i8_class_of(maxc);
    for (int t = p0; t < p0 + n;) {
        const int left = p0 + n - t > QE_I8_PLAN_MAX ? QE_I8_PLAN_MAX : p0 + n - t;
        if (left == 1) {
            /* A token with a neighbour takes the two-token form with it, which recomputes one
             * token and writes the same number over it. A token with no neighbour is the whole
             * of a decode step's walk and takes the one-token form, because borrowing would
             * read a token row the caller never passed. */
            if (t)
                forms[QE_I8_C2](w, a, t - 1, ys + t - 1, row, nb);
            else
                forms[QE_I8_ONE](w, a, 0, ys, row, nb);
            break;
        }
        const int c = qe_i8_plan_first[cmax][left];
        if (c < 0)
            return;
        forms[c](w, a, t, ys + t, row, nb);
        t += qe_i8_class_tokens[c];
    }
}

void qe_gemv_iq4_nl_i8(const void *restrict W, const qe_i8panel *restrict a, int T,
                       float *const *restrict ys, long n_rows, long K, long ldd_blocks) {
    const long nb = K / 32;
    const int m = qe_i8_ssd();
    /* Which exit this row walks is the panel's business, and both sides of it read
     * `qe_i8_rowscale()`: a walk and the panel it is handed cannot disagree about whether
     * the scales in it are per block or per row. A per-block walk on a row-scaled panel is
     * also correct - the scales are equal - which is why the tile form below needs no row
     * scale of its own to keep working. */
    const qe_iq4_i8_fn *const forms = qe_i8_rowscale() ? qe_iq4_i8_rs_forms : qe_iq4_i8_forms[m];
    const int maxc = qe_i8_tmax();
    long r = 0;
    /* A decode step is one token row against every row of the expert it routed to, which is the
     * group this family could not divide: a token with no neighbour has nothing to borrow, so
     * until the one-token form existed the caller took the dequantising float walk instead. The
     * panel this costs is one row of K bytes per expert, recoded once per expert. */
    if (T == 1) {
        for (long r0 = 0; r0 < n_rows; r0++)
            forms[QE_I8_ONE](ROWPTR(qe_block_iq4_nl, W, r0, ldd_blocks), a, 0, ys, r0, nb);
        return;
    }
    /* Rows two at a time, for the groups of eight the tile form replaces: one load of a
     * token's activation block feeding two output rows, with the panel cursors the tile
     * shares being the whole point of it (`QE_IQ4_NL_I8_R2_FORM`). A group's leftover tokens
     * go through the single-row forms once per row, as in the shared-dequant walk, so the
     * tile is a change of loads and not of numbers - `make fast` holds the two settings to
     * bit-identical outputs at every width and row count, on the panel the tile was built for:
     * it has no row-scaled form, and reading a row-scaled panel with the per-block exit is
     * correct on equal scales but not the same rounding.
     *
     * It is built against eight token rows and no more, so a wider class ceiling leaves it off
     * rather than growing the register file of a form that is already sixteen accumulators and
     * already a measured loss (`QE_I8_ROWS_DEFAULT`). */
    if (qe_i8_rows(QE_I8_ROWS_DEFAULT) >= 2 && T >= 8 && maxc <= 8 && n_rows >= 2) {
        for (; r + 1 < n_rows; r += 2) {
            const qe_block_iq4_nl *w0 = ROWPTR(qe_block_iq4_nl, W, r, ldd_blocks);
            const qe_block_iq4_nl *w1 = ROWPTR(qe_block_iq4_nl, W, r + 1, ldd_blocks);
            int t = 0;
            for (; T - t >= 8; t += 8)
                qe_iq4_i8_r2_forms[m](w0, w1, a, t, ys + t, r, r + 1, nb);
            if (T - t) {
                qe_iq4_i8_row(forms, w0, a, t, T - t, ys, r, nb, 4);
                qe_iq4_i8_row(forms, w1, a, t, T - t, ys, r + 1, nb, 4);
            }
        }
    }
    for (; r < n_rows; r++)
        qe_iq4_i8_row(forms, ROWPTR(qe_block_iq4_nl, W, r, ldd_blocks), a, 0, T, ys, r, nb, maxc);
}

/* The same division of one output row's token group on the iq3_s forms. */
static void qe_iq3_s_i8_row(const qe_iq3_i8_fn *const forms, const qe_block_iq3_s *restrict w,
                            const qe_i8panel *restrict a, int p0, int n,
                            float *const *restrict ys, long row, long nb, int maxc) {
    if (!qe_i8_plan_built)
        qe_i8_plan_build();
    const int cmax = qe_i8_class_of(maxc);
    for (int t = p0; t < p0 + n;) {
        const int left = p0 + n - t > QE_I8_PLAN_MAX ? QE_I8_PLAN_MAX : p0 + n - t;
        if (left == 1) {
            /* A token with a neighbour takes the two-token form with it, which recomputes one
             * token and writes the same number over it. A token with no neighbour is the whole
             * of a decode step's walk and takes the one-token form, because borrowing would
             * read a token row the caller never passed. */
            if (t)
                forms[QE_I8_C2](w, a, t - 1, ys + t - 1, row, nb);
            else
                forms[QE_I8_ONE](w, a, 0, ys, row, nb);
            break;
        }
        const int c = qe_i8_plan_first[cmax][left];
        if (c < 0)
            return;
        forms[c](w, a, t, ys + t, row, nb);
        t += qe_i8_class_tokens[c];
    }
}

void qe_gemv_iq3_s_i8(const void *restrict W, const qe_i8panel *restrict a, int T,
                      float *const *restrict ys, long n_rows, long K, long ldd_blocks) {
    const long nb = K / 256;
    const int m = qe_i8_ssd();
    const qe_iq3_i8_fn *const forms = qe_i8_rowscale() ? qe_iq3_i8_rs_forms : qe_iq3_i8_forms[m];
    const int maxc = qe_i8_tmax();
    long r = 0;
    /* The one-token walk of the iq4_nl form above, on the same terms. */
    if (T == 1) {
        for (long r0 = 0; r0 < n_rows; r0++)
            forms[QE_I8_ONE](ROWPTR(qe_block_iq3_s, W, r0, ldd_blocks), a, 0, ys, r0, nb);
        return;
    }
    /* The same tile on the same terms as iq4_nl above, and for the same reason only while the
     * class ceiling is the eight the tile was built against. */
    if (qe_i8_rows(QE_I8_ROWS_DEFAULT) >= 2 && T >= 8 && maxc <= 8 && n_rows >= 2) {
        for (; r + 1 < n_rows; r += 2) {
            const qe_block_iq3_s *w0 = ROWPTR(qe_block_iq3_s, W, r, ldd_blocks);
            const qe_block_iq3_s *w1 = ROWPTR(qe_block_iq3_s, W, r + 1, ldd_blocks);
            int t = 0;
            for (; T - t >= 8; t += 8)
                qe_iq3_i8_r2_forms[m](w0, w1, a, t, ys + t, r, r + 1, nb);
            if (T - t) {
                qe_iq3_s_i8_row(forms, w0, a, t, T - t, ys, r, nb, 4);
                qe_iq3_s_i8_row(forms, w1, a, t, T - t, ys, r + 1, nb, 4);
            }
        }
    }
    for (; r < n_rows; r++)
        qe_iq3_s_i8_row(forms, ROWPTR(qe_block_iq3_s, W, r, ldd_blocks), a, 0, T, ys, r, nb, maxc);
}
#endif /* x86 */

/* ---------------------------------------------------------------- dispatch */

/* The widest token group a shared-dequant walk takes (src/kern.h), read once and
 * overridable with QE_PARK because the number is under measurement: see the comment on
 * iq_park_take for what the two ends of its range are measured on. One process setting
 * it per point is enough for a server; `make forward` uses the setter below, because a
 * curve that needs one load of this checkpoint per sample is not a curve. */
static int iq_park_cap = -1;

int qe_iq_park_cap(void) {
    if (iq_park_cap < 0) {
        const char *e = getenv("QE_PARK");
        long v = e ? strtol(e, NULL, 10) : QE_IQ_PARK_DEFAULT;
        if (v < 0)
            v = 0;
        if (v > QE_IQ_PARK_MAX)
            v = QE_IQ_PARK_MAX;
        iq_park_cap = (int)v;
    }
    return iq_park_cap;
}

void qe_iq_set_park_cap(int v) {
    if (v < 0)
        v = 0;
    if (v > QE_IQ_PARK_MAX)
        v = QE_IQ_PARK_MAX;
    iq_park_cap = v;
}

/* Which walk a full group of eight takes: the dequantised block parked in an L1 line and
 * loaded again for every row of the group (`multi8`, the form that shipped), or the block
 * held in four vector registers beside the eight sums (`mix8`). The second is four fewer
 * loads a row per 32 weights, which is the whole of a row's cost at this shape, and it is
 * not obviously free, because fourteen live vectors is the register file and a spilled
 * accumulator is worse than the load it saved. `make moe`'s shared-dequant table times the
 * two against each other inside one process, alternating with the group, because two runs
 * of this bench on this host disagree by a quarter on the byte tables. */
static int iq3s_reg_block = -1;

int qe_iq3s_reg_block(void) {
    if (iq3s_reg_block < 0) {
        const char *e = getenv("QE_IQ3S_REGBLOCK");
        iq3s_reg_block = e ? (atoi(e) != 0) : QE_IQ3S_REGBLOCK_DEFAULT;
    }
    return iq3s_reg_block;
}

void qe_iq3s_set_reg_block(int on) { iq3s_reg_block = on ? 1 : 0; }

/* Whether a shared-dequant walk takes an expert's output rows two at a time, so that one
 * load of a token's activations feeds two rows instead of one (`QE_IQ4_NL_MULTI2R_FORM`,
 * `QE_IQ3_S_MULTI2R_FORM`). It is the load-per-multiply item that ideas.md 7.8 and 7.11 end
 * every section on, done on the arithmetic the engine already ships: the products per output
 * element and their order are untouched, so this is not a quality decision and does not need
 * the ingestion quality table - `make fast`'s multi check and `make moe`'s grouping check
 * hold it to the same bound as the walk it replaces. `make moe`'s shared-dequant table and
 * `make walk`'s GMAC/s columns A/B it in one process, alternating, which is the only way two
 * numbers for one shape agree on this host. */
static int iq_two_row = -1;

int qe_iq_two_row(void) {
    if (iq_two_row < 0) {
        const char *e = getenv("QE_IQ_2ROW");
        iq_two_row = e ? (atoi(e) != 0) : QE_IQ_2ROW_DEFAULT;
    }
    return iq_two_row;
}

void qe_iq_set_two_row(int on) { iq_two_row = on ? 1 : 0; }


long qe_type_block_ne(qe_type t) {
    const qe_type_info *i = qe_type_info_of(t);
    return i ? (long)i->block_ne : -1;
}

void qe_gemv_iq4_nl(const void *restrict W, const float *restrict x, float *restrict y,
                    long n_rows, long K, long ldd_blocks) {
    const long nb = K / 32;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_iq4_nl *w = ROWPTR(qe_block_iq4_nl, W, r, ldd_blocks);
#if defined(__x86_64__)
        y[r] = qe_dot_iq4_nl_avx2(w, x, nb);
#else
        y[r] = (float)qe_dot_iq4_nl_ref(w, x, nb);
#endif
    }
}

void qe_gemv_iq4_xs(const void *restrict W, const float *restrict x, float *restrict y,
                    long n_rows, long K, long ldd_blocks) {
    const long nb = K / 256;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_iq4_xs *w = ROWPTR(qe_block_iq4_xs, W, r, ldd_blocks);
#if defined(__x86_64__)
        y[r] = qe_dot_iq4_xs_avx2(w, x, nb);
#else
        y[r] = (float)qe_dot_iq4_xs_ref(w, x, nb);
#endif
    }
}

void qe_gemv_iq3_s(const void *restrict W, const float *restrict x, float *restrict y,
                   long n_rows, long K, long ldd_blocks) {
    const long nb = K / 256;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_iq3_s *w = ROWPTR(qe_block_iq3_s, W, r, ldd_blocks);
#if defined(__x86_64__)
        y[r] = qe_dot_iq3_s_avx2(w, x, nb);
#else
        y[r] = (float)qe_dot_iq3_s_ref(w, x, nb);
#endif
    }
}

#if defined(__x86_64__)
/* One walk per group for a bucket wider than eight. The sizes are the ones the parked
 * forms are instantiated for (QE_IQ_PARK_MAX in src/kern.h), so 17..19 and 21..23 and
 * 25..31 step down to the nearest one and the residue is left as a group in its own
 * right - what is being minimised is walks of the quantiser, and a bucket of eleven
 * cut as 8+3 pays two of them for the price of one. */
#define QE_PARK_SWITCH(pre)                                                                        \
    switch (T) {                                                                                   \
    case 9:                                                                                        \
        pre##9(w, xs, ys, row, nb);                                                                \
        return 1;                                                                                  \
    case 10:                                                                                       \
        pre##10(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 11:                                                                                       \
        pre##11(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 12:                                                                                       \
        pre##12(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 13:                                                                                       \
        pre##13(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 14:                                                                                       \
        pre##14(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 15:                                                                                       \
        pre##15(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 16:                                                                                       \
        pre##16(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 20:                                                                                       \
        pre##20(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 24:                                                                                       \
        pre##24(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    case 32:                                                                                       \
        pre##32(w, xs, ys, row, nb);                                                               \
        return 1;                                                                                  \
    default:                                                                                       \
        return 0;                                                                                  \
    }

static int iq4_nl_park(const qe_block_iq4_nl *restrict w, const float *const *restrict xs,
                       float *const *restrict ys, long row, long nb, int T) {
    QE_PARK_SWITCH(qe_row_iq4_nl_park)
}

static int iq3_s_park(const qe_block_iq3_s *restrict w, const float *const *restrict xs,
                      float *const *restrict ys, long row, long nb, int T) {
    QE_PARK_SWITCH(qe_row_iq3_s_park)
}

/* The mixed walk for a group of eight to eleven (QE_IQ3_S_MIX_FORM above). Asked first,
 * because every row it can seat in a register is a load and a store per group that the
 * parked walk pays and this one does not; wider groups have too many residue rows for
 * the split to still be worth eight registers. */
static int iq3_s_mix(const qe_block_iq3_s *restrict w, const float *const *restrict xs,
                     float *const *restrict ys, long row, long nb, int T) {
    switch (T) {
    case 8:
        qe_row_iq3_s_mix8(w, xs, ys, row, nb);
        return 1;
    case 9:
        qe_row_iq3_s_mix9(w, xs, ys, row, nb);
        return 1;
    case 10:
        qe_row_iq3_s_mix10(w, xs, ys, row, nb);
        return 1;
    case 11:
        qe_row_iq3_s_mix11(w, xs, ys, row, nb);
        return 1;
    default:
        return 0;
    }
}

/* How much of the rest of a bucket the parked form takes in one walk.
 *
 * The ceiling is a measurement, not a shape of the instruction set, which is why it is
 * a run-time number (`QE_PARK`, 0 turns parking off and leaves the eight-plus-tail
 * dispatch this replaced). In isolation the parked walk wins at every count `make moe`
 * can ask for and the probe it came out of is monotone to thirty-two tokens; in the
 * engine, over forty-eight layers of real routing and nineteen threads, the same switch
 * cost the expert stage 16 % at the shipping width (14.34 → 16.64 ms/token of `moe_ffn`,
 * `/tmp/f6_wide.log` against `/tmp/f7_wide.log`, where every other stage of the two runs
 * agrees to 2 %). What differs between the two readings is not the arithmetic, it is
 * what the cores are sitting on: a parked group walks an activation panel of one
 * K-row per token, so thirty-two of them is 320 KiB per thread and 6 MiB of L2 across a
 * nineteen-thread pool, against 1.6 MiB for groups of eight. `make forward` prints the
 * curve, and the default is the point on it that the engine, and not the kernel, likes.
 *
 * Sizes between the instantiated ones step down rather than up: a bucket of twenty-one
 * is a walk of twenty and a walk of one, and one walk too many is cheaper than a panel
 * too wide. */
static int iq_park_take(int rem) {
    int cap = qe_iq_park_cap();
    if (cap > QE_IQ_PARK_MAX)
        cap = QE_IQ_PARK_MAX;
    if (rem > cap)
        rem = cap;
    if (rem < 9)
        return 0;
    if (rem >= 32)
        return 32;
    if (rem >= 25)
        return 24;
    if (rem >= 20)
        return 20;
    if (rem >= 17)
        return 16;
    return rem;
}
#endif /* x86 */

/* T token rows through the same iq4_nl rows with the dequantisation shared. xs[t]
 * is token t's K activations and ys[t] its n_rows outputs; nothing about the
 * layout has to be regular, which is the whole reason this exists next to
 * qe_gemv_multi_t - the tokens that chose one expert of a chunk are a set of
 * rows, not a run of them. */
void qe_gemv_iq4_nl_multi(const void *restrict W, const float *const *restrict xs,
                          float *const *restrict ys, long n_rows, long K, long ldd_blocks,
                          int T) {
#if defined(__x86_64__)
    const long nb = K / 32;
    long r = 0;
    /* Rows two at a time, for the groups of eight the pairing form replaces, while
     * nothing wants parking: a parked group is a different trade (ideas.md 7.2b), so a
     * walk that parks takes the row-at-a-time path below unchanged.
     *
     * Only the eights. A group's leftover tokens are walked by the same single-row form
     * the walk below would have used for them, once per row of the pair rather than
     * twice with the activations loaded twice, because the pairing buys its win on the
     * loads and not on the arithmetic - and the two-token, three-token and four-token
     * forms accumulate in two chains by block parity where the eight-token form uses one
     * chain per token, so a paired form for those groups would be a different number as
     * well as a different schedule. With the pairing on the eight-groups the two walks
     * leave bit-identical outputs at every width and every row count, which is what
     * section 3 of `make walk` holds it to. */
    if (qe_iq_two_row() && T >= 8 && n_rows >= 2 && iq_park_take(T) == 0) {
        for (; r + 1 < n_rows; r += 2) {
            const qe_block_iq4_nl *w0 = ROWPTR(qe_block_iq4_nl, W, r, ldd_blocks);
            const qe_block_iq4_nl *w1 = ROWPTR(qe_block_iq4_nl, W, r + 1, ldd_blocks);
            int t = 0;
            for (; T - t >= 8; t += 8)
                qe_row_iq4_nl_multi2r8(w0, w1, xs + t, ys + t, r, r + 1, nb);
            for (; T - t >= 4; t += 4) {
                qe_row_iq4_nl_multi4(w0, xs + t, ys + t, r, nb);
                qe_row_iq4_nl_multi4(w1, xs + t, ys + t, r + 1, nb);
            }
            if (T - t == 3) {
                qe_row_iq4_nl_multi3(w0, xs + t, ys + t, r, nb);
                qe_row_iq4_nl_multi3(w1, xs + t, ys + t, r + 1, nb);
            } else if (T - t == 2) {
                qe_row_iq4_nl_multi2(w0, xs + t, ys + t, r, nb);
                qe_row_iq4_nl_multi2(w1, xs + t, ys + t, r + 1, nb);
            } else if (T - t == 1) {
                ys[t][r] = qe_dot_iq4_nl_avx2(w0, xs[t], nb);
                ys[t][r + 1] = qe_dot_iq4_nl_avx2(w1, xs[t], nb);
            }
        }
    }
    for (; r < n_rows; r++) {
        const qe_block_iq4_nl *w = ROWPTR(qe_block_iq4_nl, W, r, ldd_blocks);
        int t = 0;
        for (;;) {
            int g = iq_park_take(T - t);
            if (g <= 0)
                break;
            if (!iq4_nl_park(w, xs + t, ys + t, r, nb, g)) {
                g = 8;
                qe_row_iq4_nl_multi8(w, xs + t, ys + t, r, nb);
            }
            t += g;
        }
        for (; T - t >= 8; t += 8)
            qe_row_iq4_nl_multi8(w, xs + t, ys + t, r, nb);
        for (; T - t >= 4; t += 4)
            qe_row_iq4_nl_multi4(w, xs + t, ys + t, r, nb);
        if (T - t == 3)
            qe_row_iq4_nl_multi3(w, xs + t, ys + t, r, nb);
        else if (T - t == 2)
            qe_row_iq4_nl_multi2(w, xs + t, ys + t, r, nb);
        else if (T - t == 1)
            ys[t][r] = qe_dot_iq4_nl_avx2(w, xs[t], nb);
    }
#else
    for (int t = 0; t < T; t++)
        qe_gemv_iq4_nl(W, xs[t], ys[t], n_rows, K, ldd_blocks);
#endif
}

void qe_gemv_iq3_s_multi(const void *restrict W, const float *const *restrict xs,
                         float *const *restrict ys, long n_rows, long K, long ldd_blocks, int T) {
#if defined(__x86_64__)
    const long nb = K / 256;
    long r = 0;
    /* Rows two at a time, on the same terms and for the same reason as iq4_nl above:
     * the eight-groups go through the pairing form, and a group's leftover tokens go
     * through the forms the walk below uses, once per row, so the pairing is a change of
     * loads and not of numbers. */
    if (qe_iq_two_row() && T >= 8 && n_rows >= 2 && iq_park_take(T) == 0) {
        for (; r + 1 < n_rows; r += 2) {
            const qe_block_iq3_s *w0 = ROWPTR(qe_block_iq3_s, W, r, ldd_blocks);
            const qe_block_iq3_s *w1 = ROWPTR(qe_block_iq3_s, W, r + 1, ldd_blocks);
            int t = 0;
            for (; T - t >= 8; t += 8)
                qe_row_iq3_s_multi2r8(w0, w1, xs + t, ys + t, r, r + 1, nb);
            for (; T - t >= 3; t += 3) {
                qe_row_iq3_s_multi3(w0, xs + t, ys + t, r, nb);
                qe_row_iq3_s_multi3(w1, xs + t, ys + t, r + 1, nb);
            }
            if (T - t == 2) {
                qe_row_iq3_s_multi2(w0, xs + t, ys + t, r, nb);
                qe_row_iq3_s_multi2(w1, xs + t, ys + t, r + 1, nb);
            } else if (T - t == 1) {
                ys[t][r] = qe_dot_iq3_s_avx2(w0, xs[t], nb);
                ys[t][r + 1] = qe_dot_iq3_s_avx2(w1, xs[t], nb);
            }
        }
    }
    for (; r < n_rows; r++) {
        const qe_block_iq3_s *w = ROWPTR(qe_block_iq3_s, W, r, ldd_blocks);
        int t = 0;
        /* One walk where the parked form seats it (see QE_IQ_PARK_MAX), eight where the
         * register budget holds the sums in registers, then three at a time for what is
         * left: the eight-token form holds the dequantised block in L1, the three-token
         * form keeps it in registers, which is cheaper per token but cannot reach eight.
         * See the register note above. */
        for (;;) {
            int g = iq_park_take(T - t);
            if (g <= 0)
                break;
            if (!iq3_s_mix(w, xs + t, ys + t, r, nb, g) &&
                !iq3_s_park(w, xs + t, ys + t, r, nb, g)) {
                g = 8;
                qe_row_iq3_s_multi8(w, xs + t, ys + t, r, nb);
            }
            t += g;
        }
        for (; T - t >= 8; t += 8) {
            if (qe_iq3s_reg_block())
                qe_row_iq3_s_mix8(w, xs + t, ys + t, r, nb);
            else
                qe_row_iq3_s_multi8(w, xs + t, ys + t, r, nb);
        }
        for (; T - t >= 3; t += 3)
            qe_row_iq3_s_multi3(w, xs + t, ys + t, r, nb);
        if (T - t == 2)
            qe_row_iq3_s_multi2(w, xs + t, ys + t, r, nb);
        else if (T - t == 1)
            ys[t][r] = qe_dot_iq3_s_avx2(w, xs[t], nb);
    }
#else
    for (int t = 0; t < T; t++)
        qe_gemv_iq3_s(W, xs[t], ys[t], n_rows, K, ldd_blocks);
#endif
}

/* 1 when the dispatcher below has a shared-dequant form for this type. A caller that
 * has to decide before it starts writing into accumulators cannot ask by calling and
 * seeing whether the answer was -1, because by then half the answer exists; this is
 * the same list, kept adjacent so the two are one edit. */
int qe_gemv_multi_ptrs_has(qe_type t) {
    switch (t) {
    case QE_IQ4_NL:
    case QE_IQ3_S:
        return 1;
    default:
        return 0;
    }
}

int qe_gemv_multi_ptrs_t(qe_type t, const void *restrict W, const float *const *xs,
                         float *const *ys, int T, long n_rows, long K, long ldd) {
    if (T < 2 || n_rows <= 0 || !K)
        return -1;
    if (qe_gemv_t(t, W, xs[0], ys[0], 0, K, ldd))
        return -1;
    switch (t) {
    case QE_IQ4_NL:
        qe_gemv_iq4_nl_multi(W, xs, ys, n_rows, K, ldd, T);
        return 0;
    case QE_IQ3_S:
        qe_gemv_iq3_s_multi(W, xs, ys, n_rows, K, ldd, T);
        return 0;
    default:
        return -1;
    }
}

/* The integer walks are a different list from the shared-dequant ones above, because what
 * decides is not only the weight format: a caller has to be able to recode the rows it is
 * multiplying with, and a recode is a quality decision the stage makes, not the kernel.
 * Q8_0 is there because the dense projections are Q8_0 and a chunk's panel is a quarter
 * of the bytes of the f32 rows they read today; IQ3_S and IQ4_NL are there because their
 * codebooks come out of the decode as int8 and the four converts and four multiplies that
 * turn them into floats are the walk's most expensive repeated act. */
int qe_gemv_i8_has(qe_type t) {
    switch (t) {
    case QE_Q8_0:
    case QE_IQ4_NL:
    case QE_IQ3_S:
        return 1;
    default:
        return 0;
    }
}

/* Whether the list above answers for one token row, which is a narrower question than whether
 * it answers at all: a decode step's expert walk has one activation row, and only the codebook
 * walks have a form for it. A caller that decides before it starts writing outputs has to ask,
 * because a walk it expected to be refused is a wrong number and not a slower one. */
int qe_gemv_i8_has_1(qe_type t) {
    switch (t) {
    case QE_IQ4_NL:
    case QE_IQ3_S:
        return 1;
    default:
        return 0;
    }
}

int qe_gemv_i8_t(qe_type t, const void *restrict W, const qe_i8panel *restrict a, int T,
                 float *const *restrict ys, long n_rows, long K, long ldd) {
    if (T < 1 || n_rows <= 0 || !K)
        return -1;
#if defined(__x86_64__)
    switch (t) {
    case QE_Q8_0:
        /* A group of one is not a form here either: the narrowest Q8_0 tile is a pair, and its
         * remainder takes the tile's own last row with it, which a caller holding one token does
         * not have. The codebook walks have a one-token form; this one does not, and says so. */
        if (T == 1)
            return -1;
        qe_gemv_q8_0_i8(W, a, T, ys, n_rows, K, ldd);
        return 0;
    case QE_IQ4_NL:
        qe_gemv_iq4_nl_i8(W, a, T, ys, n_rows, K, ldd);
        return 0;
    case QE_IQ3_S:
        qe_gemv_iq3_s_i8(W, a, T, ys, n_rows, K, ldd);
        return 0;
    default:
        return -1;
    }
#else
    (void)W;
    (void)a;
    (void)T;
    (void)ys;
    (void)n_rows;
    (void)K;
    (void)ldd;
    return -1;
#endif
}

void qe_gemv_q6_k(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks) {
    const long nb = K / 256;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_q6_k *w = ROWPTR(qe_block_q6_k, W, r, ldd_blocks);
#if defined(__x86_64__)
        y[r] = qe_dot_q6_k_avx2(w, x, nb);
#else
        y[r] = (float)qe_dot_q6_k_ref(w, x, nb);
#endif
    }
}

void qe_gemv_q4_k(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks) {
    const long nb = K / 256;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_q4_k *w = ROWPTR(qe_block_q4_k, W, r, ldd_blocks);
#if defined(__x86_64__)
        y[r] = qe_dot_q4_k_avx2(w, x, nb);
#else
        y[r] = (float)qe_dot_q4_k_ref(w, x, nb);
#endif
    }
}

void qe_gemv_q5_0(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks) {
    const long nb = K / 32;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_q5_0 *w = ROWPTR(qe_block_q5_0, W, r, ldd_blocks);
#if defined(__x86_64__)
        y[r] = qe_dot_q5_0_avx2(w, x, nb);
#else
        y[r] = (float)qe_dot_q5_0_ref(w, x, nb);
#endif
    }
}

/* f32 -> bf16 with round-to-nearest-even, which is what a bf16 weight matrix
 * made by the quantiser contains. Rounding rather than truncating matters here:
 * the shadow is compared against the f32 router expert by expert, and a
 * truncated copy disagrees about the winners far more often than one rounded
 * the way the checkpoint's own bf16 tensors were produced. */
void qe_bf16_from_f32(const float *src, void *dst, size_t n) {
    uint16_t *d = (uint16_t *)dst;
    for (size_t i = 0; i < n; i++) {
        union {
            float f;
            uint32_t u;
        } v;
        v.f = src[i];
        if (((v.u >> 23) & 0xffu) == 0xffu && (v.u & 0x7fffffu)) {
            d[i] = (uint16_t)((v.u >> 16) | 0x0040u); /* NaN stays NaN */
            continue;
        }
        v.u += 0x7fffu + ((v.u >> 16) & 1u);
        d[i] = (uint16_t)(v.u >> 16);
    }
}

void qe_gemv_bf16(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd) {
    for (long r = 0; r < n_rows; r++) {
        const uint16_t *p = (const uint16_t *)W + r * ldd;
        float s = 0;
        long k = 0;
#if defined(__x86_64__)
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (; k + 16 <= K; k += 16) {
            for (int h = 0; h < 2; h++) {
                /* bf16 is the top half of a float, so widening is a shift. The
                 * load has to be the 128-bit one: an 8 u16 group is 16 bytes,
                 * and a 64-bit load quietly zeroes the upper four lanes, which
                 * is a dot product that returns half its terms and passes any
                 * test that only looks at magnitudes. */
                __m256 w = _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p + k + h * 8))), 16));
                if (h)
                    a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + k + 8), a1);
                else
                    a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + k), a0);
            }
        }
        s = qe_hsum(_mm256_add_ps(a0, a1));
#endif
        for (; k < K; k++) {
            union {
                float f;
                uint32_t u;
            } v;
            v.u = (uint32_t)p[k] << 16;
            s += v.f * x[k];
        }
        y[r] = s;
    }
}

int qe_gemv_t(qe_type t, const void *restrict W, const float *restrict x, float *restrict y,
              long n_rows, long K, long ldd) {
    switch (t) {
    case QE_F32: qe_gemv_f32(W, x, y, n_rows, K, ldd); return 0;
    case QE_Q8_0: qe_gemv_q8_0(W, x, y, n_rows, K, ldd); return 0;
    case QE_Q6_K: qe_gemv_q6_k(W, x, y, n_rows, K, ldd); return 0;
    case QE_Q4_K: qe_gemv_q4_k(W, x, y, n_rows, K, ldd); return 0;
    case QE_Q5_0: qe_gemv_q5_0(W, x, y, n_rows, K, ldd); return 0;
    case QE_IQ4_NL: qe_gemv_iq4_nl(W, x, y, n_rows, K, ldd); return 0;
    case QE_IQ4_XS: qe_gemv_iq4_xs(W, x, y, n_rows, K, ldd); return 0;
    case QE_IQ3_S: qe_gemv_iq3_s(W, x, y, n_rows, K, ldd); return 0;
    case QE_BF16: qe_gemv_bf16(W, x, y, n_rows, K, ldd); return 0;
    default: return -1;
    }
}

int qe_gemv_multi_t(qe_type t, const void *restrict W, const float *restrict x,
                    float *restrict y, long n_rows, long K,
                    long ldd, int T, long x_stride, long y_stride, size_t row_bytes,
                    size_t block_bytes) {
    if (T < 1 || n_rows <= 0 || !row_bytes)
        return -1;
    /* Zero rows is a no-op in every kernel and still reports an unsupported
     * type, so this rejects a bad dtype without having written anything. */
    if (qe_gemv_t(t, W, x, y, 0, K, ldd))
        return -1;
    if (!x_stride) x_stride = K;
    if (!y_stride) y_stride = n_rows;
    if (!block_bytes) block_bytes = 1u << 20;
    long rows = (long)(block_bytes / row_bytes);
    if (rows < 1)
        rows = 1;
    /* Rows outside, tokens inside: a block is read once and then used T times
     * from wherever it landed, which is the entire point. Reversing the loops
     * would walk the whole matrix once per token.
     *
     * Q8_0 with more than one token goes one step further and shares the
     * dequantisation as well, so the arithmetic stops growing with T; see
     * qe_gemv_q8_0_multi for what that costs the summation order. */
    for (long r0 = 0; r0 < n_rows; r0 += rows) {
        long n = n_rows - r0 < rows ? n_rows - r0 : rows;
        const uint8_t *w = (const uint8_t *)W + (size_t)r0 * row_bytes;
        if (t == QE_Q8_0 && T >= 2) {
            qe_gemv_q8_0_multi(w, x, y + r0, n, K, ldd, T, x_stride, y_stride);
            continue;
        }
        for (int i = 0; i < T; i++)
            qe_gemv_t(t, w, x + (size_t)i * x_stride, y + (size_t)i * y_stride + r0, n, K, ldd);
    }
    return 0;
}

/* ------------------------------------------------------------ quantising */

/* Nearest codebook index for a value already divided by the block's scale.
   kvalues_iq4nl is monotone, so the answer is a count of the fifteen midpoints it
   does not exceed, and sixteen boundaries are four descending steps. A value
   landing exactly on a midpoint goes to the larger magnitude, which is where the
   smaller relative error lives. */
static inline int qe_iq4_nl_code(float s) {
    static const float mid[16] = {-1e+30f, -115.5f, -93.5f,  -74.0f, -57.0f, -42.0f,
                                  -28.5f,  -16.0f,  -4.5f,  7.0f,   19.0f,  31.5f,
                                  45.5f,   61.0f,   79.0f,  101.0f};
    int c = s >= mid[8] ? 8 : 0;
    if (s >= mid[c + 4])
        c += 4;
    if (s >= mid[c + 2])
        c += 2;
    if (s >= mid[c + 1])
        c += 1;
    return c;
}

/* Candidate block scales, as multiples of the scale that puts the block's largest
   weight on the codebook's largest value. The reason a block usually does not want
   that scale is one outsized weight forcing everyone else onto coarser steps, and
   the trade is a smaller scale plus a clipped peak: this grid walks it. It is a
   grid, not the reference quantiser's search, and what settles whether that is
   good enough is the loss on real text (docs/DESIGN.md 12), not a number here.
   Two anchors because the codebook is not symmetric: +113 is its largest positive
   value, -127 its largest magnitude, and a block whose extreme weight is negative
   cannot name its own scale from the first one. Omit the second and a block built
   from codebook values at an exactly representable scale does not come back
   unchanged, which is bench_kern's weight-at-a-time check. The wider search is free
   where it is not needed: over the 3429 blocks bench_kern scores, the worst rms is
   0.1830 with the second anchor and 0.1830 without it, and no block is worse for the
   grid having been doubled. */
#define QE_IQ4_NL_TRIES 10
static const float qe_iq4_nl_try[QE_IQ4_NL_TRIES] = {1.00f, 0.90f, 0.81f, 0.73f, 0.66f,
                                                     0.59f, 0.53f, 0.48f, 0.43f, 0.39f};

static void qe_quant_block_iq4_nl(const float *v, qe_block_iq4_nl *out) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        const float a = fabsf(v[i]);
        if (a > amax)
            amax = a;
    }
    if (!(amax > 0.0f)) { /* all zeros, or a block with nothing but NaNs */
        out->d.d = 0;
        memset(out->qs, 0, sizeof out->qs);
        return;
    }
    qe_half best = {0};
    double best_sse = 0.0;
    int have = 0;
    const float anchor[2] = {amax * (1.0f / 127.0f), amax * (1.0f / 113.0f)};
    for (int a = 0; a < 2; a++)
    for (int c = 0; c < QE_IQ4_NL_TRIES; c++) {
        const qe_half h = qe_f32_to_half(anchor[a] * qe_iq4_nl_try[c]);
        const float d = qe_half_to_f32(h);
        if (!(d > 0.0f))
            continue; /* flushed to zero: smaller than a half can name */
        const float inv = 1.0f / d;
        double sse = 0.0;
        for (int i = 0; i < 32; i++) {
            const int q = qe_iq4_nl_code(v[i] * inv);
            const double e = (double)v[i] - (double)d * (double)(int8_t)qe_kvalues_iq4nl[q];
            sse += e * e;
        }
        if (!have || sse < best_sse) {
            have = 1;
            best_sse = sse;
            best = h;
        }
    }
    if (!have)
        best = qe_f32_to_half(amax * (1.0f / 113.0f));
    const float d = qe_half_to_f32(best);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    /* Low nibble is the first half of the block, high nibble the second, which is
       how qe_dot_iq4_nl_ref reads it back; getting this backwards is invisible to
       an error bound and obvious to bench_kern's weight-at-a-time check. */
    for (int j = 0; j < 16; j++)
        out->qs[j] = (uint8_t)(qe_iq4_nl_code(v[j] * inv) |
                               (qe_iq4_nl_code(v[j + 16] * inv) << 4));
    out->d.d = best.d;
}

size_t qe_quant_iq4_nl(const float *x, qe_block_iq4_nl *blocks, size_t n) {
    size_t nb = n / 32;
    for (size_t b = 0; b < nb; b++)
        qe_quant_block_iq4_nl(x + b * 32, blocks + b);
    return nb;
}

/* The same codebook search with a Q8_0 block on the input side.
   A block's 32 values are `d * q`, which is the product the walk that reads them
   reconstructs (qe_dequant_q8_0), so rebuilding them here and handing them to the
   block quantiser is the identical search on the identical numbers - a requantised
   copy made this way is what `qe_quant_iq4_nl` would have written for the file's own
   floats, and bench_kern holds the two to bit equality over a Q8_0 row for exactly
   that claim. The reconstruction stays in a 32-value stack frame rather than a row
   buffer because a block is where the scale lives: nothing about this pass needs the
   row it belongs to, and a row buffer would be 4 GiB of f32 written and read to carry
   values that are each used 20 times and then dead. */
size_t qe_quant_iq4_nl_q8_0(const qe_block_q8_0 *blocks, qe_block_iq4_nl *out, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float d = qe_half_to_f32(blocks[b].d);
        float v[32];
        for (int i = 0; i < 32; i++)
            v[i] = d * (float)blocks[b].qs[i];
        qe_quant_block_iq4_nl(v, out + b);
    }
    return nblocks;
}

/* One Q6_K block's 256 values, rebuilt in the order they are worth: a block is the
 * widest unit whose scale is constant enough to quantise, and 1 KiB of stack is nothing
 * next to the row it is a tenth of. The expression is `qe_dot_q6_k_ref`'s, so what the
 * encoders below see is what a walk would have multiplied. */
static void qe_q6_k_values(const qe_block_q6_k *w, float *v) {
    const float d = qe_half_to_f32(w->d);
    const uint8_t *ql = w->ql, *qh = w->qh;
    const int8_t *sc = w->scales;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            for (int slot = 0; slot < 4; slot++) {
                const int qb = qe_q6_qbyte(slot, l);
                const int nib = (slot >= 2) ? (ql[qb] >> 4) : (ql[qb] & 0xf);
                const int q = nib | (((qh[l] >> qe_q6_hshift(slot)) & 3) << 4);
                v[n + slot * 32 + l] = d * (float)sc[is + 2 * slot] * (float)(q - 32);
            }
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

void qe_dequant_q6_k(const void *w, float *dst, size_t n) {
    const qe_block_q6_k *b = (const qe_block_q6_k *)w;
    for (size_t i = 0; i + 256 <= n; i += 256, b++)
        qe_q6_k_values(b, dst + i);
}

size_t qe_quant_q8_0(const float *x, qe_block_q8_0 *blocks, size_t n) {
    size_t nb = n / 32;
    for (size_t b = 0; b < nb; b++) {
        const float *v = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float a = fabsf(v[i]);
            if (a > amax)
                amax = a;
        }
        /* No scale, no block: the same rule qe_quant_iq4_nl writes, and for the same
         * reason - a block of nothing decodes to nothing whichever scale names it. */
        if (!(amax > 0.0f)) {
            blocks[b].d.d = 0;
            memset(blocks[b].qs, 0, sizeof blocks[b].qs);
            continue;
        }
        /* Both anchors, because an fp16 scale is the only thing that can name a level:
         * the codebook's own largest magnitude is 127 here, so the two ends of the grid
         * are one scale and one step below it. A block whose extreme value is negative
         * wants the second one, and a block that lands exactly on a representable scale
         * has to come back unchanged. */
        qe_half best = {0};
        double best_sse = 0.0;
        int have = 0;
        const float anchor[2] = { amax * (1.0f / 127.0f), amax * (1.0f / 128.0f) };
        for (int a = 0; a < 2; a++) {
            const qe_half h = qe_f32_to_half(anchor[a]);
            const float d = qe_half_to_f32(h);
            if (!(d > 0.0f))
                continue;
            const float inv = 1.0f / d;
            double sse = 0.0;
            for (int i = 0; i < 32; i++) {
                float q = v[i] * inv;
                if (q > 127.0f)
                    q = 127.0f;
                if (q < -127.0f)
                    q = -127.0f;
                const double e = (double)v[i] - (double)d * (double)lrintf(q);
                sse += e * e;
            }
            if (!have || sse < best_sse) {
                have = 1;
                best_sse = sse;
                best = h;
            }
        }
        if (!have)
            continue; /* smaller than a half can name: nothing above it will survive either */
        blocks[b].d = best;
        const float d = qe_half_to_f32(best);
        const float inv = 1.0f / d;
        for (int i = 0; i < 32; i++) {
            float q = v[i] * inv;
            if (q > 127.0f)
                q = 127.0f;
            if (q < -127.0f)
                q = -127.0f;
            blocks[b].qs[i] = (int8_t)lrintf(q);
        }
    }
    return nb;
}

size_t qe_quant_iq4_nl_q6_k(const qe_block_q6_k *blocks, qe_block_iq4_nl *out, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        float v[256];
        qe_q6_k_values(blocks + b, v);
        qe_quant_iq4_nl(v, out + b * 8, 256);
    }
    return nblocks;
}

size_t qe_quant_q8_0_q6_k(const qe_block_q6_k *blocks, qe_block_q8_0 *out, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        float v[256];
        qe_q6_k_values(blocks + b, v);
        qe_quant_q8_0(v, out + b * 8, 256);
    }
    return nblocks;
}
