/* SIMD kernels.
 *
 * Everything in this engine is sized by one measurement: a decode step reads
 * 5.44 GiB of weights (measured from the tensor table by
 * bin/bench_gemv). On this host (Core Ultra 7 265, dual-channel DDR5) the
 * achievable read bandwidth is ~85-114 GB/s, which puts the ceiling at
 * 14.6-19.5 tok/s no matter how good the code is. So the kernels are written to
 * saturate the memory controller, not to win a FLOP count:
 *
 *   - each 64 B cache line is touched exactly once, no re-reads
 *   - rows are streamed, one accumulator per output, no scratch traffic
 *   - 4 blocks in flight per thread so the load queue stays full
 *   - fp32 accumulate (no fp16 rounding drift over 2560-long dots)
 *
 * Q8_0 matters most: it carries ~68% of the bytes in a decode step
 * (attention/GDN projections, hyper-connections, shared expert).
 */
#include "kern.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- references */

double qe_dot_q8_0_ref(const qe_block_q8_0 *w, const float *x, long nb) {
    double acc = 0;
    for (long b = 0; b < nb; b++) {
        float d = qe_half_to_f32(w[b].d);
        for (int i = 0; i < 32; i++)
            acc += (double)w[b].qs[i] * d * x[b * 32 + i];
    }
    return acc;
}

/* Row `r` of a Q8_0 matrix whose rows are padded to `ldd` blocks. Both the
 * vector walk and the fallback below index rows this way. */
#define Q8_ROW(w, r, ldd)                                                                          \
    ((const qe_block_q8_0 *)((const uint8_t *)(w) + (size_t)(r) * (size_t)(ldd) * 34u))

/* ------------------------------------------------------------------ AVX2 */

#if defined(__x86_64__)

/* One 32-weight block into `acc`, scaled by the block's fp16 scale.
 *
 * The scale is per 32-weight block, so it has to be applied inside the block,
 * not to the row sum. Folding it as mul-then-fma costs one extra multiply per
 * 8 weights, which is free here: at 8.5 bpw we serve ~32 bytes per 8 weights
 * and the memory controller, not the FMA units, is the limit.
 *
 * The lane stride is 8 bytes. Loading through a __m128i* would advance 16 and
 * silently skip half the block while reading into the next one - which is the
 * bug the correctness check in bench_kern exists to catch.
 */
static inline __m256 qe_dot_block_q8_0(__m256 acc, const qe_block_q8_0 *restrict b,
                                       const float *restrict x) {
    const int8_t *qs = b->qs;
    __m256 vd = _mm256_set1_ps(qe_half_to_f32(b->d));
    __m128i q8 = _mm_loadl_epi64((const __m128i *)(qs + 0));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(q8), vd), _mm256_loadu_ps(x + 0), acc);
    q8 = _mm_loadl_epi64((const __m128i *)(qs + 8));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(q8), vd), _mm256_loadu_ps(x + 8), acc);
    q8 = _mm_loadl_epi64((const __m128i *)(qs + 16));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(q8), vd), _mm256_loadu_ps(x + 16), acc);
    q8 = _mm_loadl_epi64((const __m128i *)(qs + 24));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(q8), vd), _mm256_loadu_ps(x + 24), acc);
    return acc;
}

/* Single row, 4 blocks in flight. This is the inner loop that decides whether
 * we reach the bandwidth ceiling. */
static float qe_dot_q8_0_avx2(const qe_block_q8_0 *restrict w, const float *restrict x, long nb) {
    long b = 0;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; b + 4 <= nb; b += 4) {
        a0 = qe_dot_block_q8_0(a0, w + b + 0, x + (b + 0) * 32);
        a1 = qe_dot_block_q8_0(a1, w + b + 1, x + (b + 1) * 32);
        a2 = qe_dot_block_q8_0(a2, w + b + 2, x + (b + 2) * 32);
        a3 = qe_dot_block_q8_0(a3, w + b + 3, x + (b + 3) * 32);
    }
    float t = qe_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; b < nb; b++) {
        float d = qe_half_to_f32(w[b].d);
        for (int i = 0; i < 32; i++)
            t += (float)w[b].qs[i] * d * x[b * 32 + i];
    }
    return t;
}

void qe_gemv_q8_0(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks) {
    long nb = K / 32;
    for (long r = 0; r < n_rows; r++)
        y[r] = qe_dot_q8_0_avx2(Q8_ROW(W, r, ldd_blocks), x, nb);
}

/* ---- T activations through one dequantised weight row ------------------------
 *
 * The kernel above is written for one activation, and it is right to be: four
 * blocks in flight, the dequantisation of a block inside the loop, because one
 * activation needs it once. A chunk needs the same block T times. Chunking the
 * walk already shares the bytes - that is how a chunk costs 65 ms a token against
 * a 104 ms decode - but it shares no arithmetic: calling the kernel above once per
 * token pays the widen, the convert, the multiply by the block scale and the fma
 * T times over. Per 32-weight block the cost goes from T*(4 converts + 4 muls + 4
 * fmas) to (4 converts + 4 muls) + T*4 fmas, which is the difference between an
 * ingestion limited by its arithmetic and one limited by the bytes under it.
 *
 * Two accumulator chains per token, and no more. The single-row kernel's four
 * chains would want 4T+1 ymm registers next to the dequantised block, and 17 does
 * not fit in 16; spilling the block costs more than the extra chain buys. Two
 * chains per token is what fits at every T below, and it is the shape that was
 * measured before it was written here (bench/bench_alu.c: 1.98x for T=2 and 3.78x
 * for T=4 over 19 threads, 1.96x and 3.72x over eight performance cores).
 *
 * Summing two chains instead of the shipped four changes the order of the sum, so
 * a chunked row is *not* bit-identical to the T calls it replaces. It is the same
 * dot product in another order, and bench_kern checks it against the double
 * reference rather than against the shipped kernel for exactly that reason. */

/* One 32-weight block, dequantised once into four scaled float vectors. Declares
 * q0..q3 in the enclosing scope, so every use has to sit inside its own { }. */
#define QE_DEQUAD_Q8_0(b)                                                                          \
    __m256 q0, q1, q2, q3;                                                                         \
    do {                                                                                           \
        __m256 vd = _mm256_set1_ps(qe_half_to_f32((b)->d));                                        \
        __m256i wv = _mm256_loadu_si256((const __m256i *)(b)->qs);                                 \
        __m128i h = _mm256_castsi256_si128(wv);                                                    \
        __m128i l = _mm256_extracti128_si256(wv, 1);                                               \
        q0 = _mm256_mul_ps(qe_cvt8to8_f32(h), vd);                                                 \
        q1 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(h, 8)), vd);                              \
        q2 = _mm256_mul_ps(qe_cvt8to8_f32(l), vd);                                                 \
        q3 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(l, 8)), vd);                              \
    } while (0)

/* acc += the 32 dequantised weights above, against 32 activations at x. Same lane
 * assignment as qe_dot_block_q8_0, so a token group and a decode step agree lane
 * for lane and differ only in the order their four vectors are added. */
#define QE_FMA4_Q8_0(acc, x)                                                                       \
    do {                                                                                           \
        const float *xa_ = (x);                                                                    \
        (acc) = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xa_ + 0), (acc));                              \
        (acc) = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xa_ + 8), (acc));                              \
        (acc) = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xa_ + 16), (acc));                             \
        (acc) = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xa_ + 24), (acc));                             \
    } while (0)

/* One weight row, T tokens. T has to be a literal for the token loop below to be
 * unrolled into registers, which is the whole point of writing it this way: a
 * runtime T puts the accumulators on the stack and the kernel is then slower than
 * the four chains it replaces. */
#define QE_Q8_0_MULTI_FORM(name, T)                                                                \
    static void name(const qe_block_q8_0 *restrict w, const float *restrict x,                     \
                     float *restrict y, long nb, long xstride, long ystride) {                     \
        __m256 a[T][2];                                                                            \
        const float *xp[T];                                                                        \
        for (int t = 0; t < T; t++) {                                                              \
            a[t][0] = _mm256_setzero_ps();                                                         \
            a[t][1] = a[t][0];                                                                     \
            xp[t] = x + (size_t)t * (size_t)xstride;                                               \
        }                                                                                          \
        long b = 0;                                                                                \
        for (; b + 2 <= nb; b += 2) {                                                              \
            {                                                                                      \
                QE_DEQUAD_Q8_0(w + b);                                                             \
                for (int t = 0; t < T; t++)                                                        \
                    QE_FMA4_Q8_0(a[t][0], xp[t] + b * 32);                                         \
            }                                                                                      \
            {                                                                                      \
                QE_DEQUAD_Q8_0(w + b + 1);                                                         \
                for (int t = 0; t < T; t++)                                                        \
                    QE_FMA4_Q8_0(a[t][1], xp[t] + (b + 1) * 32);                                   \
            }                                                                                      \
        }                                                                                          \
        for (; b < nb; b++) {                                                                      \
            QE_DEQUAD_Q8_0(w + b);                                                                 \
            for (int t = 0; t < T; t++)                                                            \
                QE_FMA4_Q8_0(a[t][0], xp[t] + b * 32);                                             \
        }                                                                                          \
        for (int t = 0; t < T; t++)                                                                \
            y[(size_t)t * (size_t)ystride] =                                                       \
                qe_hsum(_mm256_add_ps(a[t][0], a[t][1]));                                          \
    }

QE_Q8_0_MULTI_FORM(qe_row_q8_0_multi2, 2)
QE_Q8_0_MULTI_FORM(qe_row_q8_0_multi3, 3)
QE_Q8_0_MULTI_FORM(qe_row_q8_0_multi4, 4)

/* Eight tokens against one dequantised block.
 *
 * Sixteen registers is the whole argument: eight accumulators and the block do not
 * both fit, so the block goes to a 32-float aligned line on the stack - four stores,
 * L1-hot, read once per token - and each token owns one accumulator instead of the
 * two the four-token forms use. That is five loads and four FMAs per token-block
 * against four and four, and it is worth it because the kernel was nowhere near its
 * FMA limit: a chunk of sixteen now walks a weight row twice instead of four times.
 *
 * That stack line is what the source says, not what the machine does. SROA promotes
 * qb to registers here, so what assembles is eight groups of four FMAs and no ymm to
 * memory move anywhere in the loop: the fifth load per token-block is a count of the
 * C, not of the walk. Checked by compiling a form out of line and counting the memory
 * moves, which is the only way the two readings can be told apart - a sibling form
 * that holds the block in registers by construction assembles to the same
 * instructions, so the eviction the C describes is not a cost this kernel pays.
 *
 * Eight is a ceiling this instruction set has, not a taste. AVX-512's 32 registers
 * would take it to sixteen or twenty-four, and this host has AVX-512 fused off. */
#define QE_Q8_0_MULTI8_FORM(name, T)                                                               \
    static void name(const qe_block_q8_0 *restrict w, const float *restrict x,                     \
                     float *restrict y, long nb, long xstride, long ystride) {                     \
        __attribute__((aligned(32))) float qb[32];                                                 \
        __m256 a[T];                                                                               \
        const float *xp[T];                                                                        \
        for (int t = 0; t < T; t++) {                                                              \
            a[t] = _mm256_setzero_ps();                                                            \
            xp[t] = x + (size_t)t * (size_t)xstride;                                               \
        }                                                                                          \
        for (long b = 0; b < nb; b++) {                                                            \
            {                                                                                      \
                QE_DEQUAD_Q8_0(w + b);                                                             \
                _mm256_store_ps(qb + 0, q0);                                                       \
                _mm256_store_ps(qb + 8, q1);                                                       \
                _mm256_store_ps(qb + 16, q2);                                                      \
                _mm256_store_ps(qb + 24, q3);                                                      \
            }                                                                                      \
            for (int t = 0; t < T; t++) {                                                          \
                const float *xa_ = xp[t] + b * 32;                                                 \
                a[t] = _mm256_fmadd_ps(_mm256_load_ps(qb + 0), _mm256_loadu_ps(xa_ + 0), a[t]);    \
                a[t] = _mm256_fmadd_ps(_mm256_load_ps(qb + 8), _mm256_loadu_ps(xa_ + 8), a[t]);    \
                a[t] = _mm256_fmadd_ps(_mm256_load_ps(qb + 16), _mm256_loadu_ps(xa_ + 16), a[t]);  \
                a[t] = _mm256_fmadd_ps(_mm256_load_ps(qb + 24), _mm256_loadu_ps(xa_ + 24), a[t]);  \
            }                                                                                      \
        }                                                                                          \
        for (int t = 0; t < T; t++)                                                                \
            y[(size_t)t * (size_t)ystride] = qe_hsum(a[t]);                                        \
    }

QE_Q8_0_MULTI8_FORM(qe_row_q8_0_multi8, 8)

/* Two output rows of one matrix, one load of the activations per token.
 *
 * The forms above spend four activation loads and four FMAs per token per 32 weights, and
 * every weight row of the same matrix multiplies the same vector: row r and row r+1 are
 * walked in two passes and read the token's activations twice. Holding both dequantised
 * blocks beside the sixteen sums takes four of those eight loads out of the inner loop, and
 * it is the same item the IQ4_NL and IQ3_S walks take through `qe_iq_two_row`.
 *
 * Sixteen accumulators is the AVX2 register file spent in full, which is why there is one
 * instantiation and why it is the eight-token group's: the two- to four-token forms split a
 * token's sum over two chains by block parity, so pairing *those* would change the number as
 * well as the schedule, and the dispatcher leaves a group's leftover tokens to them. The
 * chain this keeps is `qe_row_q8_0_multi8`'s - one accumulator per token, the same four FMAs
 * per 32 weights in block order, the same hsum - so a pair walked together leaves the bits
 * two walks left, which is what the third section of `make walk` holds it to. */
#define QE_Q8_0_MULTI2R_FORM(name, T)                                                              \
    static void name(const qe_block_q8_0 *restrict w0, const qe_block_q8_0 *restrict w1,           \
                     const float *restrict x, float *restrict y0, float *restrict y1, long nb,     \
                     long xstride, long ystride) {                                                 \
        __m256 a0[T], a1[T];                                                                       \
        const float *xp[T];                                                                        \
        for (int t = 0; t < (T); t++) {                                                            \
            a0[t] = _mm256_setzero_ps();                                                           \
            a1[t] = a0[t];                                                                         \
            xp[t] = x + (size_t)t * (size_t)xstride;                                               \
        }                                                                                          \
        for (long b = 0; b < nb; b++) {                                                            \
            __m256 q00, q01, q02, q03, q10, q11, q12, q13;                                         \
            {                                                                                      \
                QE_DEQUAD_Q8_0(w0 + b);                                                            \
                q00 = q0;                                                                          \
                q01 = q1;                                                                          \
                q02 = q2;                                                                          \
                q03 = q3;                                                                          \
            }                                                                                      \
            {                                                                                      \
                QE_DEQUAD_Q8_0(w1 + b);                                                            \
                q10 = q0;                                                                          \
                q11 = q1;                                                                          \
                q12 = q2;                                                                          \
                q13 = q3;                                                                          \
            }                                                                                      \
            for (int t = 0; t < (T); t++) {                                                        \
                const float *xa_ = xp[t] + b * 32;                                                 \
                const __m256 x0 = _mm256_loadu_ps(xa_ + 0);                                        \
                const __m256 x1 = _mm256_loadu_ps(xa_ + 8);                                        \
                const __m256 x2 = _mm256_loadu_ps(xa_ + 16);                                       \
                const __m256 x3 = _mm256_loadu_ps(xa_ + 24);                                       \
                a0[t] = _mm256_fmadd_ps(q00, x0, a0[t]);                                           \
                a0[t] = _mm256_fmadd_ps(q01, x1, a0[t]);                                           \
                a0[t] = _mm256_fmadd_ps(q02, x2, a0[t]);                                           \
                a0[t] = _mm256_fmadd_ps(q03, x3, a0[t]);                                           \
                a1[t] = _mm256_fmadd_ps(q10, x0, a1[t]);                                           \
                a1[t] = _mm256_fmadd_ps(q11, x1, a1[t]);                                           \
                a1[t] = _mm256_fmadd_ps(q12, x2, a1[t]);                                           \
                a1[t] = _mm256_fmadd_ps(q13, x3, a1[t]);                                           \
            }                                                                                      \
        }                                                                                          \
        for (int t = 0; t < (T); t++) {                                                            \
            y0[(size_t)t * (size_t)ystride] = qe_hsum(a0[t]);                                      \
            y1[(size_t)t * (size_t)ystride] = qe_hsum(a1[t]);                                      \
        }                                                                                          \
    }

QE_Q8_0_MULTI2R_FORM(qe_row_q8_0_multi2r8, 8)

/* Token group `g` of weight rows [r0, r1). x and y are already offset to the
 * group's first token, so a row is y[r] and the activations are x.
 *
 * Eight at a time is the largest group that seats one accumulator per token in a
 * register next to the block (see QE_Q8_0_MULTI8_FORM). A chunk of sixteen is
 * therefore two walks of the row instead of four, and the dequantisation under each
 * walk is shared by eight tokens rather than four. g is the same for a whole call, so
 * the chain below is one predicted branch per row against a row that costs hundreds of
 * cycles. */
static void qe_rows_q8_0_group(const void *restrict W, const float *restrict x,
                               float *restrict y, long r0, long r1, long ldd_blocks, long nb,
                               int g, long x_stride, long y_stride) {
    for (long r = r0; r < r1; r++) {
        const qe_block_q8_0 *w = Q8_ROW(W, r, ldd_blocks);
        float *yr = y + r;
        if (g == 8)
            qe_row_q8_0_multi8(w, x, yr, nb, x_stride, y_stride);
        else if (g == 4)
            qe_row_q8_0_multi4(w, x, yr, nb, x_stride, y_stride);
        else if (g == 3)
            qe_row_q8_0_multi3(w, x, yr, nb, x_stride, y_stride);
        else if (g == 2)
            qe_row_q8_0_multi2(w, x, yr, nb, x_stride, y_stride);
        else
            *yr = qe_dot_q8_0_avx2(w, x, nb);
    }
}

/* The same token group over rows [r0, r1), two rows at a time.
 *
 * Only the eights are paired: an odd last row of the tile is handed to the row loop above,
 * and a group narrower than eight never reaches the pair form at all, because the only
 * one-chain form this instruction set has room for is the eight-token one. Both of those
 * are why flipping `qe_iq_two_row` moves zero outputs rather than a bounded number. */
static void qe_rows_q8_0_group2r(const void *restrict W, const float *restrict x,
                                 float *restrict y, long r0, long r1, long ldd_blocks, long nb,
                                 long x_stride, long y_stride) {
    long r = r0;
    for (; r + 1 < r1; r += 2) {
        const qe_block_q8_0 *w0 = Q8_ROW(W, r, ldd_blocks);
        const qe_block_q8_0 *w1 = Q8_ROW(W, r + 1, ldd_blocks);
        qe_row_q8_0_multi2r8(w0, w1, x, y + r, y + r + 1, nb, x_stride, y_stride);
    }
    if (r < r1)
        qe_rows_q8_0_group(W, x, y, r, r + 1, ldd_blocks, nb, 8, x_stride, y_stride);
}

/* Weight rows to hold in L2 while the token groups walk them.
 *
 * The nest is row tile outside, token group inside, and the tile is what makes it
 * worth doing. One group of eight reads 8*K*4 bytes of activations per row, and
 * those bytes are the same for every row in the tile, so a tile of rb rows turns
 * T*K*4 bytes of activation traffic *per row* into T*K*4/rb. With no tile every
 * row of the matrix re-reads the whole T-token panel: at T=512 and K=2560 that is
 * 5.24 MB per row, which is why a wide chunk used to be slower than a narrow one
 * rather than cheaper (bench/bench_walk.c: attn_qkv 476 GMAC/s at T=64 and 195 at
 * T=512, and flat across every row block, because the row block is chosen outside
 * this function and cannot reach it).
 *
 * The budget is for the innermost walk alone - the tile's weights plus one
 * group's activations, the two buffers the inner loop reads over and over - and
 * 256 KiB is a quarter of a performance core's L2, which leaves the rest to the
 * thread's own stack and the neighbouring tensor. That is 66 rows at K=2560, 10
 * at K=6144 and 740 at K=320. Below four rows the tile stops paying for the loop
 * around it, so four is the floor even for a K whose panel alone blows the budget.
 * `QE_MULTI_KIB` overrides it for a sweep. Past the pairing the budget is no longer the
 * lever it was: 256/512/1024 KiB read 205.6/205.8/203.5 ms for 100000 rows of K=2560 at
 * 512 tokens (`/tmp/walk_pairs3.log`, `/tmp/kib_512.log`, `/tmp/kib_1024.log`), 1 % apart,
 * which is what a walk that reads half the activation vectors looks like - the tile was
 * sized against panel traffic the pair form has since halved. */
static long multi_tile_bytes(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_MULTI_KIB");
        long v = e ? strtol(e, NULL, 10) : 256;
        cached = v < 8 ? 8 : v;
    }
    return cached * 1024;
}

/* The token group the walk builds from, capped at `cap`.
 *
 * Eight is where one accumulator per token stops fitting next to the dequantised
 * block, and at four or below the block and two accumulators per token do. What that
 * is worth is not a load count: the two forms differ in how many chains a token's sum
 * is split over, and the load-count story this comment used to tell (two loads per FMA
 * at eight against one at four, because the block "moves to an L1 line") is not in the
 * machine - see the disassembly note above QE_Q8_0_MULTI8_FORM. What is real is the
 * opposite trade: a narrower group walks the row more times to dequantise the same
 * bytes. QE_MULTI_G caps the group so one run of `make walk` can say which side wins,
 * because that is a measurement and not an instruction-count. Zero, the default, is
 * "eight whenever eight fit", which is what the kernel did before the cap existed. */
static int multi_group_cached = -1;

static int multi_group_cap(void) {
    if (multi_group_cached < 0) {
        const char *e = getenv("QE_MULTI_G");
        int v = e ? (int)strtol(e, NULL, 10) : 0;
        /* Only the group sizes there is a form for; anything else leaves the walk
         * to the scalar fallback in qe_rows_q8_0_group, which is a slow answer
         * rather than a wrong one but would be read as a kernel result. */
        multi_group_cached = (v == 2 || v == 3 || v == 4 || v == 8) ? v : 8;
    }
    return multi_group_cached;
}

void qe_gemv_q8_0_multi(const void *restrict W, const float *restrict x, float *restrict y,
                        long n_rows, long K, long ldd_blocks, int T, long x_stride,
                        long y_stride) {
    if (T < 2) {
        for (int t = 0; t < T; t++)
            qe_gemv_q8_0(W, x + (size_t)t * (size_t)x_stride, y + (size_t)t * (size_t)y_stride,
                         n_rows, K, ldd_blocks);
        return;
    }
    long nb = K / 32;
    long row_bytes = nb * (long)sizeof(qe_block_q8_0);
    long panel = 8 * K * (long)sizeof(float);
    long budget = multi_tile_bytes();
    long rb = budget > panel + row_bytes ? (budget - panel) / row_bytes : 4;
    if (rb < 4)
        rb = 4;
    const int gmax = multi_group_cap();
    const int pair = qe_iq_two_row() && gmax == 8;
    for (long r0 = 0; r0 < n_rows; r0 += rb) {
        long r1 = r0 + rb < n_rows ? r0 + rb : n_rows;
        for (int t = 0; t < T;) {
            int g = T - t >= gmax ? gmax
                                   : (T - t >= 4 ? 4 : (T - t == 3 ? 3 : (T - t == 2 ? 2 : 1)));
            if (pair && g == 8 && r1 - r0 >= 2)
                qe_rows_q8_0_group2r(W, x + (size_t)t * (size_t)x_stride,
                                     y + (size_t)t * (size_t)y_stride, r0, r1, ldd_blocks, nb,
                                     x_stride, y_stride);
            else
                qe_rows_q8_0_group(W, x + (size_t)t * (size_t)x_stride,
                                   y + (size_t)t * (size_t)y_stride, r0, r1, ldd_blocks, nb, g,
                                   x_stride, y_stride);
            t += g;
        }
    }
}

/* ============================ int8 activation panels and the integer walk ======
 *
 * Recoding a chunk's activations to int8 turns the 32-weights-against-32-values step
 * of a walk from four converts, four multiplies and four FMAs against four loads of
 * an activation row into one vpdpbusd, one convert, one multiply and one FMA against
 * one load - and the load is a quarter of the bytes, which is the part that decides
 * whether a chunk's activation panel rides in L1 or pushes the weights it is about to
 * multiply out of L2 (ideas.md 7.2b is what that costs when it is done backwards).
 *
 * The two things the recode owes are the scale, which is one float per 32 activations
 * and lands in the same exit the weight's per-32 scale already pays for, and the +128
 * that vpdpbusd's unsigned operand carries, which goes onto the weights so that what
 * comes out of it is 128 times the sum of the *activation* block - one number per
 * token per 32 values that every output row of the walk shares - rather than the sum
 * of the weights, which each output row would pay again.
 *
 * The quantiser is exact where being exact is free: `p` is computed from the values
 * that were actually stored, and 128 * sum of 32 values each under 128 fits an integer
 * exactly in f32, so the subtraction in the walk is between two integers rather than
 * between two large numbers that happen to differ. What `p` holds is one eighth of that
 * number - 16 * sum - because vpdpbusd owes the correction to each of the eight lanes of
 * its accumulator, and the row's horizontal sum adds the eight shares back into the one
 * block total it is.
 */

/* One row of activations: block scale, the int8 values, and the bias term.
 *
 * The scale is amax/127 and the value is round(x * 127 / amax), which cannot reach 128
 * for a finite amax because |x| <= amax bounds the product at 127 plus one ulp - and it
 * is clamped anyway, because the pack below saturates only against the int8 range and a
 * row that arrives with an infinite value should cost a wrong product rather than a
 * wrapped one. A block whose amax is zero, infinite or not a number is stored as zeros
 * with a zero scale: the same answer the f32 walks give, and the non-finite counters are
 * upstream of here, where the value still exists to be counted. */
#if defined(__x86_64__)
void qe_quant_q8_0_row(const float *restrict x, int8_t *restrict x8, float *restrict s,
                       int32_t *restrict p, long K) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 lim = _mm256_set1_ps(127.0f);
    for (long b = 0; b + 32 <= K; b += 32) {
        __m256 v[4], am = _mm256_setzero_ps();
        for (int i = 0; i < 4; i++) {
            v[i] = _mm256_loadu_ps(x + b + 8 * i);
            am = _mm256_max_ps(am, _mm256_andnot_ps(sign, v[i]));
        }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        const float amax = _mm_cvtss_f32(m);
        if (!(amax > 0.0f) || amax > 1e30f) {
            _mm256_storeu_si256((__m256i *)(x8 + b), _mm256_setzero_si256());
            s[b / 32] = 0.0f;
            p[b / 32] = 0;
            continue;
        }
        const float d = amax * (1.0f / 127.0f);
        const __m256 r = _mm256_set1_ps(1.0f / d);
        __m256 sum = _mm256_setzero_ps();
        __m256i q[4];
        for (int i = 0; i < 4; i++) {
            __m256 t = _mm256_round_ps(_mm256_mul_ps(v[i], r),
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            t = _mm256_min_ps(_mm256_max_ps(t, _mm256_sub_ps(_mm256_setzero_ps(), lim)), lim);
            sum = _mm256_add_ps(sum, t);
            q[i] = _mm256_cvtps_epi32(t);
        }
        /* The two packs are lane-local, so the 32 bytes come out interleaved by
         * 4-value groups; one vpermd puts them back in the order the walk reads. */
        _mm256_storeu_si256(
            (__m256i *)(x8 + b),
            _mm256_permutevar8x32_epi32(
                _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]), _mm256_packs_epi32(q[2], q[3])),
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
        s[b / 32] = d;
        p[b / 32] = -16 * (int32_t)qe_hsum(sum);
    }
}
#else
void qe_quant_q8_0_row(const float *restrict x, int8_t *restrict x8, float *restrict s,
                       int32_t *restrict p, long K) {
    for (long b = 0; b + 32 <= K; b += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float a = x[b + i] < 0.0f ? -x[b + i] : x[b + i];
            if (a > amax) amax = a;
        }
        if (!(amax > 0.0f) || amax > 1e30f) {
            for (int i = 0; i < 32; i++) x8[b + i] = 0;
            s[b / 32] = 0.0f;
            p[b / 32] = 0;
            continue;
        }
        const float d = amax * (1.0f / 127.0f);
        const float r = 1.0f / d;
        long sum = 0;
        for (int i = 0; i < 32; i++) {
            float t = x[b + i] * r;
            t = t < -127.0f ? -127.0f : (t > 127.0f ? 127.0f : t);
            t = t < 0.0f ? -(float)(int64_t)(-t + 0.5f) : (float)(int64_t)(t + 0.5f);
            x8[b + i] = (int8_t)t;
            sum += (long)t;
        }
        s[b / 32] = d;
        p[b / 32] = -16 * (int32_t)sum;
    }
}
#endif

void qe_quant_q8_0_rows(const float *restrict x, long x_stride, int8_t *restrict x8,
                        float *restrict s, int32_t *restrict p, long T, long K) {
    if (x_stride <= 0) x_stride = K;
    const long nb = K / 32;
    for (long t = 0; t < T; t++)
        qe_quant_q8_0_row(x + (size_t)t * (size_t)x_stride, x8 + (size_t)t * (size_t)K,
                          s + (size_t)t * (size_t)nb, p + (size_t)t * (size_t)nb, K);
}

/* The same panel with one scale for a whole row instead of one per 32 values.
 *
 * What that buys is not the storage - `s` is still one float per 32 activations, all equal,
 * which is what makes a row-scaled panel a valid panel for a walk that reads `s[b]` and
 * never heard of this function. What it buys is the walk: with one scale per row the product
 * of the weight block's scale and the activation row's scale no longer depends on the token,
 * so it leaves the token loop and the exit an output row owes per 32 weights drops from
 * {broadcast, multiply, convert, FMA} to {convert, FMA}. ideas.md 7.10's instrument priced
 * that half at 1.67x against the 1.36x of the per-block panel.
 *
 * What it costs is the range: every value in the row is now quantised against the widest one,
 * so one outlier spends itself on the whole row's resolution. A vector about to be multiplied
 * by an expert is post-norm, which is the reason this is worth measuring at all, and the
 * reason it is a knob and not a rewrite: `make score --ingest` is what says whether the
 * resolution was worth taking. */
#if defined(__x86_64__)
void qe_quant_q8_0_row_rs(const float *restrict x, int8_t *restrict x8, float *restrict s,
                          int32_t *restrict p, long K) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 lim = _mm256_set1_ps(127.0f);
    const __m256 zero = _mm256_setzero_ps();
    __m256 am = zero;
    long b = 0;
    for (; b + 32 <= K; b += 32)
        for (int i = 0; i < 4; i++) {
            const __m256 v = _mm256_loadu_ps(x + b + 8 * i);
            am = _mm256_max_ps(am, _mm256_andnot_ps(sign, v));
        }
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    const float amax = _mm_cvtss_f32(m);
    const long nb = K / 32;
    if (!(amax > 0.0f) || amax > 1e30f) {
        for (long q = 0; q + 32 <= K; q += 32)
            _mm256_storeu_si256((__m256i *)(x8 + q), _mm256_setzero_si256());
        for (long q = 0; q < nb; q++) {
            s[q] = 0.0f;
            p[q] = 0;
        }
        return;
    }
    const float d = amax * (1.0f / 127.0f);
    const __m256 r = _mm256_set1_ps(1.0f / d);
    for (long q = 0; q < nb; q++) {
        __m256 sum = zero;
        __m256i v[4];
        for (int i = 0; i < 4; i++) {
            __m256 t = _mm256_round_ps(
                _mm256_mul_ps(_mm256_loadu_ps(x + 32 * q + 8 * i), r),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            t = _mm256_min_ps(_mm256_max_ps(t, _mm256_sub_ps(zero, lim)), lim);
            sum = _mm256_add_ps(sum, t);
            v[i] = _mm256_cvtps_epi32(t);
        }
        _mm256_storeu_si256(
            (__m256i *)(x8 + 32 * q),
            _mm256_permutevar8x32_epi32(
                _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3])),
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
        s[q] = d;
        p[q] = -16 * (int32_t)qe_hsum(sum);
    }
}
#else
void qe_quant_q8_0_row_rs(const float *restrict x, int8_t *restrict x8, float *restrict s,
                          int32_t *restrict p, long K) {
    float amax = 0.0f;
    for (long i = 0; i + 32 <= K; i += 32)
        for (int j = 0; j < 32; j++) {
            const float a = x[i + j] < 0.0f ? -x[i + j] : x[i + j];
            if (a > amax) amax = a;
        }
    const long nb = K / 32;
    if (!(amax > 0.0f) || amax > 1e30f) {
        for (long q = 0; q < 32 * nb; q++) x8[q] = 0;
        for (long q = 0; q < nb; q++) {
            s[q] = 0.0f;
            p[q] = 0;
        }
        return;
    }
    const float d = amax * (1.0f / 127.0f);
    const float r = 1.0f / d;
    for (long q = 0; q < nb; q++) {
        long sum = 0;
        for (int j = 0; j < 32; j++) {
            float t = x[32 * q + j] * r;
            t = t < -127.0f ? -127.0f : (t > 127.0f ? 127.0f : t);
            t = t < 0.0f ? -(float)(int64_t)(-t + 0.5f) : (float)(int64_t)(t + 0.5f);
            x8[32 * q + j] = (int8_t)t;
            sum += (long)t;
        }
        s[q] = d;
        p[q] = -16 * (int32_t)sum;
    }
}
#endif

/* The knob for `kern.h`: how many output rows of one matrix share one read of the panel.
 * Read once, like the other two walk knobs, and settable per process so that `make walk`
 * and `make moe` can race the two settings against each other inside one run - two numbers
 * for one shape do not agree on this host any other way. */
static int i8_rows_cached = -1;

int qe_i8_rows(int dflt) {
    if (i8_rows_cached < 0) {
        const char *e = getenv("QE_I8_R");
        if (e)
            i8_rows_cached = strtol(e, NULL, 10) == 1 ? 1 : 2;
    }
    return i8_rows_cached < 0 ? dflt : i8_rows_cached;
}

void qe_i8_set_rows(int n) { i8_rows_cached = (n >= 2) ? 2 : 1; }

/* The widest token group a codebook walk is allowed to open a pass on. A pass costs the
 * codebook decode of every 32-weight block of the row it walks plus one dot per token, so
 * the tokens are the cheap part of it and the passes are not: `make walk`'s group section
 * reads 12.8 ms for a walk of eight token rows against 32.1 for eleven, and eleven tokens is
 * two passes of the same row rather than one and a half. A 512-token chunk routes 5120 slots
 * over 512 experts, which is a bucket of ten - so the family stopped at eight while the
 * buckets it is handed are ten, and every bucket between nine and sixteen pays for its
 * expert's weights twice. */
static int i8_tmax_cached = -1;

int qe_i8_tmax(void) {
    if (i8_tmax_cached < 0) {
        const char *e = getenv("QE_I8_T");
        i8_tmax_cached = e ? (int)strtol(e, NULL, 10) : QE_I8_TMAX_DEFAULT;
    }
    if (i8_tmax_cached < 8)
        return 8;
    return i8_tmax_cached > 16 ? 16 : i8_tmax_cached;
}

void qe_i8_set_tmax(int n) {
    i8_tmax_cached = n < 0 ? -1 : n >= 16 ? 16 : (n >= 12 ? 12 : (n >= 10 ? 10 : 8));
}

/* One activation scale per panel row rather than one per 32 values, which is what takes the
 * scale out of a walk's token loop (`qe_quant_q8_0_row_rs`). On, at the price both instruments
 * were asked for: `make walk` returns 1.13x at eight token rows and 1.27x at the 512 a chunk
 * ingests at on IQ4_NL (1.21x on IQ3_S), with the Q8_0 column - which takes this panel without
 * changing its kernel - reading 1.00x as the control it is (`/tmp/walk_rs1.log`), and
 * `make moe`'s grouped pass goes 0.783 to 0.686 of the f32 walk at 512 rows (`/tmp/moe_rs1.log`).
 * `make score --ingest` prices the coarser range, and `make forward` prints the head against the
 * panel it ships with. Both sides read this knob, so a panel and the walk that consumes it
 * cannot disagree about which panel it is - and a row-scaled panel is also a valid per-block
 * panel with equal scales, so a walk this knob does not cover still reads it correctly.
 *
 * The Q8_0 column is no longer the control it was printed as. A walk only takes this panel's
 * gift if it multiplies the row's scale once after the block loop, and the Q8_0 walk did not:
 * it read the equal scales as the per-block ones they are and paid the exit it had always
 * paid. With that hoist in (`qe_i8_hoist`) the same column is 1.31x at sixteen token rows,
 * 1.39x at thirty-two and 1.42x at the 512 a chunk ingests at - 538 GMAC/s against 762 - which
 * is the largest thing the dense half of a token has been offered since this walk landed
 * (`/tmp/walk_hoist.log`).
 *
 * It rides best on the signed dot, which is why both walks now default to it: the unsigned
 * walk's exit also owes the -16x-the-sum seed, one broadcast per token per 32 weights however
 * the scale is chosen, and the signed form has no seed to load once the row's scale is out of
 * the loop (1.92x of the unsigned per-block exit at 512 rows against 1.42x; see `qe_i8_ssd`
 * below). */
static int i8_rs_cached = -1;

int qe_i8_rowscale(void) {
    if (i8_rs_cached < 0) {
#if defined(__AVXVNNIINT8__)
        const char *e = getenv("QE_I8_RS");
        i8_rs_cached = e ? (strtol(e, NULL, 10) != 0) : 1;
#else
        i8_rs_cached = 0;
#endif
    }
    return i8_rs_cached;
}

void qe_i8_set_rowscale(int v) { i8_rs_cached = v < 0 ? -1 : (v ? 1 : 0); }

/* Whether the Q8_0 walk takes a row-scaled panel's row scale out of its block loop.
 *
 * The row scale was put on the signed walks only, on the argument that the unsigned walk's
 * seed is per block whatever the scale is, so the hoist saves one multiply of two operands
 * and keeps the broadcast - and that the Q8_0 walk, the unsigned one, reads its tensor at the
 * machine's byte ceiling anyway (`qe_i8_ssd` below). Both halves of that were measured at
 * eight token rows. At the 512 a chunk ingests at the same walk moves 1.3 GB/s of tensor and
 * returns 537 GMAC/s, which is a walk spending cycles on its exit rather than on bytes, and
 * the seed it owes is free in either case because `vpdpbusd` takes it as the accumulator's
 * seed operand. So the hoist is a whole multiply on the one walk that was said not to have
 * one to save, and this knob is how `make walk` says so at the widths the engine runs.
 *
 * On by default. It is a different place to multiply, not a different number: the products a
 * row sums and their order are the ones the per-block exit already used. */
static int i8_hoist_cached = -1;

int qe_i8_hoist(void) {
    if (i8_hoist_cached < 0) {
        const char *e = getenv("QE_I8_HOIST");
        i8_hoist_cached = e ? (strtol(e, NULL, 10) != 0) : 1;
    }
    return i8_hoist_cached;
}

void qe_i8_set_hoist(int v) { i8_hoist_cached = v < 0 ? -1 : (v ? 1 : 0); }

/* The signed×signed dot, where the compiler has it. One knob for both files: the walk in
 * `kern_iq.c` spends the same two operands this one's quantiser computes once, so the two
 * have to agree about which instruction a process is running.
 *
 * The default was per format, because what the two saved operands are worth is not the same
 * on both walks. A codebook row is decoded to signed int8 by the walk that consumes it, so
 * the signed dot costs nothing there and is worth 1.10x on an IQ4_NL row and 1.13x on an
 * IQ3_S row at the eight-to-sixteen-token groups an expert bucket hands (`make walk`). A
 * Q8_0 row used to be the exception on the argument that its walk reads its tensor at this
 * host's byte ceiling, and a leg already waiting on bytes has no exit left to save; measured
 * that way the signed form was 0.95x at eight token rows. Both halves of that were eight-row
 * numbers: the same walk at the 512 a chunk ingests at moves 2.1 GB/s of tensor for 538
 * GMAC/s, which is a walk spending its cycles on the exit, and the exit it spends them on is
 * the one the row scale removes (`qe_i8_hoist`). With that taken out the signed dot is ahead
 * of the unsigned on a row-scaled panel at every width this file prices - 1.15x at eight
 * token rows, 1.33x at the 512 - because the unsigned form still loads the bias seed per
 * token per 32 weights while the signed one has no seed to load. So both walks default to
 * the signed dot, `QE_I8_SSD` set either way overrides both, and `make walk`'s hoist table
 * holds the pair this runs on to being the fastest of the four it races.
 *
 * It is the better number as well as the faster one: the unsigned form pays the +128 its
 * operand carries by summing a lane that carries as much bias as signal and cancelling it in
 * the horizontal sum at the end of a row, and the signed form never owes that. */
static int i8_ssd_cached = -1;
static int i8_ssd_q8_cached = -1;

static int i8_ssd_env(void) {
    const char *e = getenv("QE_I8_SSD");
    if (!e)
        return -1;
    return strtol(e, NULL, 10) != 0;
}

int qe_i8_ssd(void) {
    if (i8_ssd_cached < 0) {
#if defined(__AVXVNNIINT8__)
        const int e = i8_ssd_env();
        i8_ssd_cached = e < 0 ? 1 : e;
#else
        i8_ssd_cached = 0;
#endif
    }
    return i8_ssd_cached;
}

int qe_i8_ssd_q8(void) {
    if (i8_ssd_q8_cached < 0) {
#if defined(__AVXVNNIINT8__)
        const int e = i8_ssd_env();
        i8_ssd_q8_cached = e < 0 ? 1 : e;
#else
        i8_ssd_q8_cached = 0;
#endif
    }
    return i8_ssd_q8_cached;
}

void qe_i8_set_ssd(int on) {
#if defined(__AVXVNNIINT8__)
    i8_ssd_cached = on ? 1 : 0;
    i8_ssd_q8_cached = on ? 1 : 0;
#else
    (void)on;
    i8_ssd_cached = 0;
    i8_ssd_q8_cached = 0;
#endif
}

/* The Q8_0 dot on its own, which is the pin an A/B of the Q8_0 walk needs: `qe_i8_set_ssd`
 * moves the codebook walks' choice with it, and those walks are the other half of an expert
 * stage, so a leg that asked this question with that setter would be measuring something
 * else. A negative argument hands the choice back to the configuration. */
void qe_i8_set_ssd_q8(int on) {
#if defined(__AVXVNNIINT8__)
    i8_ssd_q8_cached = on < 0 ? -1 : (on ? 1 : 0);
#else
    (void)on;
    i8_ssd_q8_cached = 0;
#endif
}

/* Put both walks back on what the process was configured with, which is the only way an
 * A/B that pinned them with `qe_i8_set_ssd` hands the choice back: the two answers are no
 * longer the same, so restoring one of them is not restoring the default. */
void qe_i8_unset_ssd(void) {
    i8_ssd_cached = -1;
    i8_ssd_q8_cached = -1;
}

#if defined(__x86_64__)

/* R output rows and T panel rows against one 32-activation group of the weights.
 *
 * `u` is the weight block with its sign bit xored, which is the +128 vpdpbusd's
 * unsigned operand carries; `d` is the block's scale, `sa` the panel block's, and `pv`
 * the share of the 128-times-the-sum the xor owes that each accumulator lane pays, so
 * that the eight lanes of the horizontal sum at the end of the row pay it once. R is 1
 * or 2 and T at most 8, which is what the
 * register file holds without spilling: R*T accumulators, R weight blocks, R scales,
 * and one activation block that every output row of the tile reads from a register
 * instead of from the panel - the thing the f32 walk cannot do, because its activation
 * group is four vectors wide and it needs four FMAs and four loads per token to spend
 * one 32-weights block on it.
 *
 * `RS=1` is the walk of a row-scaled panel (`qe_quant_q8_0_row_rs`, the configuration this
 * engine ships with): every block of a token row carries the same scale, so the product of
 * the two scales is no longer a function of the block, the accumulator runs in weight-scale
 * space, and the row's scale multiplies it once after the block loop. That is one multiply
 * fewer per token per 32 weights on the walk that has the seed anyway - and the seed is free,
 * because `vpdpbusd` takes it as the accumulator it was given.
 *
 * Both literals are load-bearing, as everywhere else in this file: a run-time count
 * puts the accumulators on the stack, and the walk is then slower than the one it
 * replaced. */
#define QE_I8_TILE_FORM_Q8_0(name, R, T, SSD, RS)                                                     \
    static void name(const qe_block_q8_0 *const *wp, long nb, const int8_t *x8, long s8,              \
                     const float *sp, const int32_t *pp, long sc, float *const *yp, long r0) {        \
        __m256 a[R][T];                                                                              \
        __m256i u[R];                                                                                \
        __m256 d[R];                                                                                 \
        const __m256i m80 = _mm256_set1_epi8((char)0x80);                                            \
        for (int rr = 0; rr < (R); rr++) {                                                           \
            for (int t = 0; t < (T); t++)                                                            \
                a[rr][t] = _mm256_setzero_ps();                                                      \
        }                                                                                            \
        for (long g = 0; g < nb; g++) {                                                              \
            for (int rr = 0; rr < (R); rr++) {                                                       \
                u[rr] = QE_I8_W((SSD), wp[rr][g].qs, m80);                                            \
                d[rr] = _mm256_set1_ps(qe_half_to_f32(wp[rr][g].d));                                 \
            }                                                                                        \
            for (int t = 0; t < (T); t++) {                                                           \
                const long so = (size_t)t * (size_t)sc;                                                \
                const __m256i pv = _mm256_set1_epi32(pp[so + g]);                                      \
                const int8_t *const xt = x8 + (size_t)t * (size_t)s8;                                  \
                for (int rr = 0; rr < (R); rr++) {                                                     \
                    const __m256i dot = QE_I8_DOT(                                                       \
                        (SSD), u[rr],                                                                    \
                        _mm256_loadu_si256((const __m256i *)(xt + (long)32 * g)), pv);                    \
                    a[rr][t] = (RS)                                                                      \
                        ? _mm256_fmadd_ps(d[rr], _mm256_cvtepi32_ps(dot), a[rr][t])                      \
                        : _mm256_fmadd_ps(                                                                 \
                            _mm256_mul_ps(d[rr], _mm256_set1_ps(sp[so + g])),                              \
                            _mm256_cvtepi32_ps(dot), a[rr][t]);                                           \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
        for (int rr = 0; rr < (R); rr++)                                                               \
            for (int t = 0; t < (T); t++) {                                                            \
                const float sr = (RS) ? sp[(size_t)t * (size_t)sc] : 1.0f;                              \
                yp[t][r0 + rr] = qe_hsum(a[rr][t]) * sr;                                                \
            }                                                                                          \
    }

QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t2, 1, 2, 0, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t4, 1, 4, 0, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t8, 1, 8, 0, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t2, 2, 2, 0, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t4, 2, 4, 0, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t8, 2, 8, 0, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t2_rs, 1, 2, 0, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t4_rs, 1, 4, 0, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t8_rs, 1, 8, 0, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t2_rs, 2, 2, 0, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t4_rs, 2, 4, 0, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t8_rs, 2, 8, 0, 1)
#if defined(__AVXVNNIINT8__)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t2_ssd, 1, 2, 1, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t4_ssd, 1, 4, 1, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t8_ssd, 1, 8, 1, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t2_ssd, 2, 2, 1, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t4_ssd, 2, 4, 1, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t8_ssd, 2, 8, 1, 0)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t2_ssd_rs, 1, 2, 1, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t4_ssd_rs, 1, 4, 1, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r1t8_ssd_rs, 1, 8, 1, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t2_ssd_rs, 2, 2, 1, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t4_ssd_rs, 2, 4, 1, 1)
QE_I8_TILE_FORM_Q8_0(qe_i8_q8_0_r2t8_ssd_rs, 2, 8, 1, 1)
#endif

/* How many output rows of one matrix share one read of the panel. Two is eight accumulators,
 * two weight blocks and two scales, which is the register file; beyond that the walk reaches
 * for the panel with a register already holding something else. One is what it runs, on every
 * format and every width: the fourth section of `make walk` reads the pair at 0.95-1.04x of
 * two single-row walks on the expert formats and 0.89-0.97x on Q8_0, at eight tokens and at
 * five hundred and twelve, in DRAM and with one expert's bytes resident (ideas.md 7.12 - the
 * 1.41x this file and `ideas.md` used to quote for Q8_0 at eight tokens was one leg of one
 * run, and three runs of the same table since put it with the other single-row outlier).
 * Two remains reachable with QE_I8_R=2 for a shape and a kernel that wants it, and the
 * section re-measures both columns every time it runs. */
static int i8_rows_cap(void) { return qe_i8_rows(QE_I8_ROWS_DEFAULT); }

typedef void (*qe_i8_q8_0_fn)(const qe_block_q8_0 *const *, long, const int8_t *, long,
                              const float *, const int32_t *, long, float *const *, long);

static const qe_i8_q8_0_fn qe_i8_q8_0_forms[2][2][2][3] = {
    { { { qe_i8_q8_0_r1t2, qe_i8_q8_0_r1t4, qe_i8_q8_0_r1t8 },
        { qe_i8_q8_0_r2t2, qe_i8_q8_0_r2t4, qe_i8_q8_0_r2t8 } },
      { { qe_i8_q8_0_r1t2_rs, qe_i8_q8_0_r1t4_rs, qe_i8_q8_0_r1t8_rs },
        { qe_i8_q8_0_r2t2_rs, qe_i8_q8_0_r2t4_rs, qe_i8_q8_0_r2t8_rs } } },
#if defined(__AVXVNNIINT8__)
    { { { qe_i8_q8_0_r1t2_ssd, qe_i8_q8_0_r1t4_ssd, qe_i8_q8_0_r1t8_ssd },
        { qe_i8_q8_0_r2t2_ssd, qe_i8_q8_0_r2t4_ssd, qe_i8_q8_0_r2t8_ssd } },
      { { qe_i8_q8_0_r1t2_ssd_rs, qe_i8_q8_0_r1t4_ssd_rs, qe_i8_q8_0_r1t8_ssd_rs },
        { qe_i8_q8_0_r2t2_ssd_rs, qe_i8_q8_0_r2t4_ssd_rs, qe_i8_q8_0_r2t8_ssd_rs } } },
#else
    { { { qe_i8_q8_0_r1t2, qe_i8_q8_0_r1t4, qe_i8_q8_0_r1t8 },
        { qe_i8_q8_0_r2t2, qe_i8_q8_0_r2t4, qe_i8_q8_0_r2t8 } },
      { { qe_i8_q8_0_r1t2_rs, qe_i8_q8_0_r1t4_rs, qe_i8_q8_0_r1t8_rs },
        { qe_i8_q8_0_r2t2_rs, qe_i8_q8_0_r2t4_rs, qe_i8_q8_0_r2t8_rs } } },
#endif
};

void qe_gemv_q8_0_i8(const void *restrict W, const qe_i8panel *restrict a, int T,
                     float *const *restrict ys, long n_rows, long K, long ldd_blocks) {
    const long nb = K / 32;
    const long row_bytes = nb * (long)sizeof(qe_block_q8_0);
    const long panel = 8 * K;
    const long budget = multi_tile_bytes();
    long rb = budget > panel + row_bytes ? (budget - panel) / row_bytes : 4;
    if (rb < 4) rb = 4;
    const int rmax = i8_rows_cap();
    const int mode = qe_i8_ssd_q8();
    /* A row-scaled panel and a walk that knows it: the second index. Without the row scale
     * the block's scale is the block's own and there is nothing to hoist. */
    const int hoisted = (qe_i8_rowscale() && qe_i8_hoist()) ? 1 : 0;

    for (long r0 = 0; r0 < n_rows; r0 += rb) {
        const long r1 = r0 + rb < n_rows ? r0 + rb : n_rows;
        for (int t = 0; t < T;) {
            const int left = T - t;
            const int ti = left >= 8 ? 2 : (left >= 4 ? 1 : 0);
            const int tt = (int[]){ 2, 4, 8 }[ti];
            /* A group of one is not a form: the narrowest tile is a pair, so a
             * remainder of one row takes the last row of the previous tile with it
             * and writes it twice. Stepping down instead would read ys[t+1], a token
             * pointer the caller never passed, for any T the tiles do not divide -
             * T = 3, 5, 7, 9, which is exactly the group width a partly-filled
             * routed-expert bucket hands this walk. */
            const int tb = (left == 1 && t) ? t - 1 : t;
            float *yp[8];
            for (int i = 0; i < tt; i++) yp[i] = ys[tb + i];
            const int8_t *const x8 = a->x8 + (size_t)tb * (size_t)a->stride8;
            const float *const sp = a->s + (size_t)tb * (size_t)a->nb;
            const int32_t *const pp = a->p + (size_t)tb * (size_t)a->nb;
            for (long r = r0; r < r1;) {
                const int ri = r1 - r >= rmax ? rmax - 1 : 0;
                const int rr = ri + 1;
                const qe_block_q8_0 *wp[2];
                for (int i = 0; i < rr; i++)
                    wp[i] = Q8_ROW(W, r + i, ldd_blocks);
                qe_i8_q8_0_forms[mode][hoisted][ri][ti](wp, nb, x8, a->stride8, sp, pp, a->nb, yp,
                                                        r);
                r += rr;
            }
            t += (tb == t) ? tt : 1;
        }
    }
}

#if defined(__AVXVNNIINT8__)
/* A copy of the same matrix with one scale for a whole output row, walked with nothing left
 * in its block loop.
 *
 * The row-scaled panel and the hoisted exit (`qe_i8_rowscale`, `qe_i8_hoist`) leave the Q8_0
 * walk four operations per token per 32 weights: load the activation block, spend the dot,
 * convert the int32 lane, fold the block's scale into the f32 sum. Two of those four exist
 * only because a block of weights carries a scale of its own. Give the matrix one scale per
 * row - which costs a copy of the weights requantised to it - and what is left per token is
 * a load and a `vpdpbssd`, with the row's scale and the token's paid once after the loop.
 * Freeing the register the block scale lived in is the second half of the prize: an
 * accumulator can then be the dot's own seed operand, which is what makes it one instruction
 * instead of a dot and an add, and the tile can afford more of them. `R*T + R + 1` vectors
 * is the budget (T accumulators per output row, R weight blocks, one activation block that
 * every row of the tile reads from a register), which is 16 exactly at R=3, T=4.
 *
 * Only the signed instruction qualifies. The unsigned one owes a bias seed per block, and a
 * seed that depends on the block is precisely the thing being removed.
 *
 * The accumulators are int32 and run the whole row: 32 * 127 * 127 per block, so 131072
 * values is the first row this can overflow and no matrix in this checkpoint is a tenth of
 * that. The panel has to be a row-scaled one - the walk reads `sp[t * sc]` as the whole of
 * token t's scale, which is what `qe_quant_q8_0_row_rs` writes and not what a per-block panel
 * means. */
#define QE_I8_FLAT_FORM_Q8_0(name, R, T)                                                             \
    static void name(const int8_t *const *wp, long nb, const int8_t *x8, long s8,                    \
                     const float *sp, long sc, const float *sw, float *const *yp, long r0) {         \
        __m256i acc[R][T];                                                                           \
        for (int rr = 0; rr < (R); rr++) {                                                           \
            for (int t = 0; t < (T); t++)                                                            \
                acc[rr][t] = _mm256_setzero_si256();                                                 \
        }                                                                                            \
        for (long g = 0; g < nb; g++) {                                                              \
            __m256i u[R];                                                                            \
            for (int rr = 0; rr < (R); rr++)                                                          \
                u[rr] = _mm256_loadu_si256((const __m256i *)(wp[rr] + (long)32 * g));                \
            for (int t = 0; t < (T); t++) {                                                           \
                const __m256i xv = _mm256_loadu_si256((const __m256i *)(                             \
                    x8 + (size_t)t * (size_t)s8 + (long)32 * g));                                     \
                for (int rr = 0; rr < (R); rr++)                                                       \
                    acc[rr][t] = _mm256_dpbssd_epi32(acc[rr][t], u[rr], xv);                           \
            }                                                                                          \
        }                                                                                              \
        for (int rr = 0; rr < (R); rr++)                                                                 \
            for (int t = 0; t < (T); t++) {                                                               \
                const float v = qe_hsum(_mm256_cvtepi32_ps(acc[rr][t]));                                   \
                yp[t][r0 + rr] = v * (sw[r0 + rr] * sp[(size_t)t * (size_t)sc]);                          \
            }                                                                                          \
    }

QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r1t2, 1, 2)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r1t4, 1, 4)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r1t8, 1, 8)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r2t2, 2, 2)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r2t4, 2, 4)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r3t2, 3, 2)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r3t4, 3, 4)
QE_I8_FLAT_FORM_Q8_0(qe_i8_flat_r4t2, 4, 2)

typedef void (*qe_i8_flat_fn)(const int8_t *const *, long, const int8_t *, long, const float *,
                              long, const float *, float *const *, long);

/* Indexed by output rows per tile - 1 and [T index] - then by the token tile 2, 4, 8. A cell
 * that would spill the register file is 0 and the entry point never asks for it. */
static const qe_i8_flat_fn qe_i8_flat_forms[4][3] = {
    { qe_i8_flat_r1t2, qe_i8_flat_r1t4, qe_i8_flat_r1t8 },
    { qe_i8_flat_r2t2, qe_i8_flat_r2t4, 0 },
    { qe_i8_flat_r3t2, qe_i8_flat_r3t4, 0 },
    { qe_i8_flat_r4t2, 0, 0 },
};

/* Rows of the flat copy `T` tokens at a time. `ws` is one scale per output row of the whole
 * matrix, `ldd` is the byte stride between its rows, `tile_r` is how many output rows share
 * one activation load. Returns nonzero for a shape there is no form for, which is a caller
 * error rather than a fallback: a token group the tiles do not divide would be walked twice
 * at two different scales. */
int qe_gemv_q8_0_flat_i8(const void *restrict W, const float *restrict ws,
                         const qe_i8panel *restrict a, int T, float *const *restrict ys,
                         long n_rows, long K, long ldd, int tile_r) {
    const long nb = K / 32;
    if (tile_r < 1 || tile_r > 4)
        return 1;
    const int ri = tile_r - 1;
    /* Widest token tile the row tile has a form for. */
    int ti = 2;
    while (ti >= 0 && !qe_i8_flat_forms[ri][ti])
        ti--;
    if (ti < 0)
        return 1;
    const int tstep = (int[]){ 2, 4, 8 }[ti];
    if (!nb || T <= 0 || T % tstep)
        return 1;
    const int8_t *const w = (const int8_t *)W;
    const long stride = ldd > 0 ? ldd : nb * 32;
    for (long r0 = 0; r0 < n_rows;) {
        /* A tail of fewer rows than the tile takes is walked one row at a time: the forms for
         * one row exist at every token width, and a row is never written twice, because the
         * scale this walk applies is the row's own. */
        const int rr_i = (n_rows - r0 >= tile_r) ? ri : 0;
        const int rr = rr_i + 1;
        const int8_t *wp[4];
        for (int i = 0; i < rr; i++)
            wp[i] = w + (size_t)(r0 + i) * (size_t)stride;
        for (int t = 0; t < T; t += tstep) {
            float *yp[8];
            for (int i = 0; i < tstep; i++)
                yp[i] = ys[t + i];
            qe_i8_flat_forms[rr_i][ti](wp, nb, a->x8 + (size_t)t * (size_t)a->stride8, a->stride8,
                                       a->s + (size_t)t * (size_t)a->nb, a->nb, ws, yp, r0);
        }
        r0 += rr;
    }
    return 0;
}

#endif /* AVXVNNIINT8 */

#endif /* x86 */

void qe_gemv_f32(const void *restrict W, const float *restrict x, float *restrict y,
                 long n_rows, long K, long ldd) {
    const float *w = (const float *)W;
    for (long r = 0; r < n_rows; r++) {
        const float *p = w + r * ldd;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        long k = 0;
        for (; k + 16 <= K; k += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + k + 0), _mm256_loadu_ps(x + k + 0), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + k + 8), _mm256_loadu_ps(x + k + 8), a1);
        }
        float t = qe_hsum(_mm256_add_ps(a0, a1));
        for (; k < K; k++)
            t += p[k] * x[k];
        y[r] = t;
    }
}

#else /* portable fallback */

void qe_gemv_q8_0(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks) {
    long nb = K / 32;
    for (long r = 0; r < n_rows; r++) {
        const qe_block_q8_0 *w = Q8_ROW(W, r, ldd_blocks);
        double a = 0;
        for (long b = 0; b < nb; b++) {
            float d = qe_half_to_f32(w[b].d);
            for (int i = 0; i < 32; i++)
                a += (double)w[b].qs[i] * d * x[b * 32 + i];
        }
        y[r] = (float)a;
    }
}

/* Nothing here can share a dequantisation across tokens without a scratch buffer,
 * and a fallback that cannot be timed on this host is not owed one: this is what
 * the blocked walk did before there was a multi-token kernel at all, and the
 * results are the same as T single-token calls down to the summation order. */
void qe_gemv_q8_0_multi(const void *restrict W, const float *restrict x, float *restrict y,
                        long n_rows, long K, long ldd_blocks, int T, long x_stride,
                        long y_stride) {
    for (int t = 0; t < T; t++)
        qe_gemv_q8_0(W, x + (size_t)t * (size_t)x_stride, y + (size_t)t * (size_t)y_stride,
                     n_rows, K, ldd_blocks);
}

void qe_gemv_f32(const void *restrict W, const float *restrict x, float *restrict y,
                 long n_rows, long K, long ldd) {
    const float *w = (const float *)W;
    for (long r = 0; r < n_rows; r++) {
        double a = 0;
        for (long k = 0; k < K; k++)
            a += (double)w[r * ldd + k] * x[k];
        y[r] = (float)a;
    }
}
#endif

/* Q8_0 -> f32, for re-coding a tensor into another format. A row at a time is the
 * unit that matters, not the whole tensor: a requantising pass walks a row, quantises
 * it, and never needs the f32 form of the row again, so this is a building block and
 * not a stage. */
void qe_dequant_q8_0(const void *w, float *dst, size_t n) {
    const qe_block_q8_0 *b = (const qe_block_q8_0 *)w;
    for (size_t i = 0; i + 32 <= n; i += 32, b++) {
        const float d = qe_half_to_f32(b->d);
        for (int j = 0; j < 32; j++)
            dst[i + j] = d * (float)b->qs[j];
    }
}
