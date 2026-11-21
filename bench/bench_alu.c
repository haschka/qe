/* Is a decode step ALU-bound or memory-bound on this host, and what is each core
   class worth?

   Every rate in this repository used to be quoted in GB/s against a 113.8 GB/s DRAM
   ceiling, which quietly assumes the ceiling is the wall. It is not, and this bench
   is what settled it: the highest linear read measured here is 85.5 GB/s (eight
   performance cores), and the shipped kernel runs at 96-99 % of an identical-bytes
   read at the shapes a decode step has. See docs/DESIGN.md 1.2 for the tables and
   for the kernel rewrites this instrument killed.

   The question it was written for: the
   shipped Q8_0 kernel spends about twenty four instructions per 32-weight block
   turning int8 into float, scaling it, and multiplying it into an accumulator,
   while a kernel that only *reads* the same bytes spends three. If the arithmetic
   is the limit, the next order of magnitude comes from an integer dot product and
   no amount of cache or placement work will find it. If the memory controller is
   the limit, the integer kernel is a distraction and the only real moves are
   traffic ones.

   So this prints both numbers over the same bytes, at the same shapes, in the same
   process, plus the two things that decide whether a dot product is even reachable
   on all twenty cores: what one P core and one E core do alone, and what the pool
   does when it is pinned to a named set of cpus instead of being left to the
   scheduler. The Core Ultra has eight cores at 5300 MHz and twelve at 4600 MHz
   with smaller caches behind them, and `qe_pool_init` with no cpu list pins
   nothing at all - so "thread 19 does not exist, and the core left out is an
   efficient one" is a statement about the scheduler's mood rather than about this
   program.

   One process per thread set, because a pool's cpu list is fixed when it is built:

     ./bin/bench_alu --cpus 0-7  --kernels all     # eight performance cores
     ./bin/bench_alu --cpus 8-19 --kernels all     # twelve efficient cores
     ./bin/bench_alu                               # the engine's default pool
     ./bin/bench_alu --span 128 --kernels read,state,statent   # the state's own shape
     ./bin/bench_alu --kernels gemv,multi1,multi4,shipped4  # the chunk question

   The last one is about the other half of a step. A token reads 5568 MiB of weights
   and about 216 MiB of recurrent state, and writes that state back; a store that is
   not a streaming store fetches a line before it overwrites it, so the state can
   cost the bus three passes where a streaming store would spend two. `state` and
   `statent` are that shape - read an f32 span, decay it, write it back - differing
   only in the store, and both are charged for the read and the write. That is the
   instrument for the traffic arithmetic in docs/DESIGN.md 14.

   `multi1`, `multi2`, `multi4` and `shipped2`, `shipped4` are the third question,
   the one about ingestion. Chunking a prefill removes DRAM bytes but removes no
   arithmetic, so the walk that dequantises a weight run once and multiplies it into
   T token accumulators is the kernel worth writing; the rows differ only in T, and
   the number they exist to produce is the ratio this prints against the T=1 row.
   T.00x means T tokens came for the price of one walk and the bus was the wall;
   1.00x means the arithmetic was the wall and sharing a dequant buys nothing. Those
   rows print per token and as that ratio, because per pass they look slower than a
   decode and are not. `shipped2` and `shipped4` are qe_gemv_q8_0_multi - the kernel
   that came out of the two candidates - timed the same way, so the pair says whether
   the shipped form gives away anything to the walk that predicted it.

   GB/s is the honest column against the DRAM ceiling; Gw/s is weights per second,
   which is the column that compares kernels reading different numbers of bytes per
   weight. Nothing here touches the checkpoint: the span is synthetic, laid out
   exactly as a Q8_0 tensor of the model's own width, and the scales are finite. */
#define _GNU_SOURCE
#include "qe.h"
#include "kern.h"

#include <immintrin.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 64

/* The double precision reference the kernels are checked against, same one
 * bench_kern uses. */
extern double qe_dot_q8_0_ref(const qe_block_q8_0 *w, const float *x, long nb);

#define K_DIM 2560 /* the model's K: 80 blocks of 32 weights */
#define SPAN_MIB 512
#define ROW_BYTES ((K_DIM / 32) * 34)
#define N_BLOCKS (K_DIM / 32)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ------------------------------------------------------------ the four walks */

typedef struct {
    const uint8_t *W;
    const float *x;
    const int8_t *xi;   /* x quantised to s8, block by block */
    const int32_t *xc;  /* per block, per dpbusd lane: 128 * sum of that lane's x */
    const float *xs;    /* per block: the quantisation scale of x */
    float *y;
    long rows;
    long bf16_rows;
    uint8_t *Ww; /* the same weights, 32 B runs only, no scales interleaved */
    uint8_t *Ws; /* the same scales, fp16, one dense array of 2 B per block */
    uint8_t *S;  /* a span of f32 to read and write back: the recurrent state's shape */
    int iters; /* walks per timed pass: a span smaller than L2 takes microseconds,
                * and a dispatch barrier costs more than the work it measures */
    const float *xt[16]; /* T activation rows, one per token in a chunk */
    float *yt;          /* rows * 16 outputs: where a multi-token walk puts them */
} job;

/* 1. What the engine runs today: widen, convert, scale, fma. */
static void walk_gemv(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->rows, tid, nthreads, &lo, &hi);
    /* The row offset is the whole measurement: with every thread starting at row 0
     * the same 20 MB is walked twenty times over and the number that comes out is
     * an L2 rate, not a DRAM one. */
    for (int it = 0; it < j->iters; it++)
        qe_gemv_q8_0(j->W + (size_t)lo * ROW_BYTES, j->x, j->y + lo, hi - lo, K_DIM, K_DIM / 32);
}

/* 2. The same bytes with the arithmetic taken out: one vpsadbw per block, and the
 * block's scale read because a 34-byte block layout reads it whether it is used
 * or not. This is the access-pattern ceiling, and the gap between it and row 1 is
 * exactly the arithmetic. */
static void walk_read(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->rows, tid, nthreads, &lo, &hi);
    for (int it = 0; it < j->iters; it++)
    for (long r = lo; r < hi; r++) {
        const uint8_t *p = j->W + (size_t)r * ROW_BYTES;
        __m256i acc = _mm256_setzero_si256();
        float dsum = 0.0f;
        for (long b = 0; b < N_BLOCKS; b++) {
            const uint8_t *w = p + b * 34;
            acc = _mm256_add_epi64(
                acc, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)w),
                                     _mm256_setzero_si256()));
            dsum += qe_half_to_f32(((const qe_block_q8_0 *)p)[b].d);
        }
        j->y[r] = (float)(_mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 2)) * dsum;
    }
}

#if defined(__AVXVNNI__)
/* 3. The integer walk: one vpdpbusd per 32 weights. The weights are signed and the
 * instruction wants an unsigned first operand, so they are shifted by 128 and the
 * 128 * (sum of x in that lane) that the shift adds is subtracted from the same
 * lane - a vector per block that depends only on x, which is why it is table
 * lookup and not work. The block's two fp16 scales then meet once per block
 * instead of once per eight weights. */
static void walk_vnni(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->rows, tid, nthreads, &lo, &hi);
    const __m256i k128 = _mm256_set1_epi8((char)128);
    for (int it = 0; it < j->iters; it++)
    for (long r = lo; r < hi; r++) {
        const uint8_t *p = j->W + (size_t)r * ROW_BYTES;
        float acc = 0.0f;
        for (long b = 0; b < N_BLOCKS; b++) {
            const uint8_t *w = p + b * 34;
            __m256i s = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                            _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)w),
                                                            k128),
                                            _mm256_loadu_si256((const __m256i *)(j->xi + b * 32)));
            s = _mm256_sub_epi32(s, _mm256_loadu_si256((const __m256i *)(j->xc + b * 8)));
            long long t = _mm256_extract_epi32(s, 0) + _mm256_extract_epi32(s, 1) +
                          _mm256_extract_epi32(s, 2) + _mm256_extract_epi32(s, 3) +
                          _mm256_extract_epi32(s, 4) + _mm256_extract_epi32(s, 5) +
                          _mm256_extract_epi32(s, 6) + _mm256_extract_epi32(s, 7);
            acc += (float)t * (qe_half_to_f32(((const qe_block_q8_0 *)p)[b].d) * j->xs[b]);
        }
        j->y[r] = acc;
    }
}
#endif

/* 4. Two rows in flight, and one 32 B load per block instead of four 8 B loads.
 *
 * The number that motivated this walk is per thread: a performance core streams the
 * same bytes at 31.2 GB/s when it is only reading them and at 14.1 GB/s through the
 * shipped kernel, and a thread has at most one row's blocks in flight. Doubling the
 * rows in flight doubles the misses a thread can have outstanding, which is the
 * only lever that moves a per-thread rate. The arithmetic per row is the shipped
 * kernel's, accumulator for accumulator and in the same order, so the numbers have
 * to come out bit for bit - checked for exactly that, not to a tolerance. */
/* Four widening multiplies and four fmas per 32-weight block, over a block that is
 * already in a register. Eight extracts/moves is what a 32 B int8 run costs, and
 * the trap this shape has is advancing 16 bytes at a time and silently skipping half
 * the block - which is why both candidates here go through this one macro and both
 * are compared against the shipped kernel before they are timed. */
#define ALU_DOT_W(acc, w, vd, x)                                                                 \
    do {                                                                                         \
        __m128i h = _mm256_castsi256_si128(w);                                                   \
        __m128i l = _mm256_extracti128_si256(w, 1);                                              \
        (acc) = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(h), vd), _mm256_loadu_ps(x), (acc)); \
        (acc) = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(h, 8)), vd),         \
                                _mm256_loadu_ps((x) + 8), (acc));                                \
        (acc) = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(l), vd), _mm256_loadu_ps((x) + 16),  \
                                (acc));                                                          \
        (acc) = _mm256_fmadd_ps(_mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(l, 8)), vd),         \
                                _mm256_loadu_ps((x) + 24), (acc));                               \
    } while (0)

#define ALU_DOT_BLOCK(acc, b, x)                                                                 \
    ALU_DOT_W((acc), _mm256_loadu_si256((const __m256i *)(b)->qs),                               \
              _mm256_set1_ps(qe_half_to_f32((b)->d)), (x))

/* One 34 B block, dequantised once into four scaled float vectors.
 *
 * This is the whole point of the multi-token walks: ALU_DEQUAD runs once per block
 * and ALU_FMA4 consumes it once per token, so the widen, convert and scale that a
 * decode step pays per token are paid once per chunk. It declares q0..q3 in the
 * enclosing scope, which means each use has to sit inside its own { }, and the
 * reason the two are split at all is so that the split is visible: eight instructions
 * of dequantisation against four fmas per token is the trade being measured. */
#define ALU_DEQUAD(b)                                                                            \
    __m256 q0, q1, q2, q3;                                                                       \
    do {                                                                                         \
        __m256 vd = _mm256_set1_ps(qe_half_to_f32((b)->d));                                      \
        __m256i w = _mm256_loadu_si256((const __m256i *)(b)->qs);                                \
        __m128i h = _mm256_castsi256_si128(w);                                                   \
        __m128i l = _mm256_extracti128_si256(w, 1);                                              \
        q0 = _mm256_mul_ps(qe_cvt8to8_f32(h), vd);                                                \
        q1 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(h, 8)), vd);                              \
        q2 = _mm256_mul_ps(qe_cvt8to8_f32(l), vd);                                                \
        q3 = _mm256_mul_ps(qe_cvt8to8_f32(_mm_srli_si128(l, 8)), vd);                              \
    } while (0)

/* acc += (the 32 dequantised weights in q0..q3) * (32 activations at x), one chain,
 * the same lane assignment ALU_DOT_W uses. */
#define ALU_FMA4(acc, x)                                                                         \
    do {                                                                                         \
        const float *xa_ = (x);                                                                  \
        (acc) = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xa_ + 0), (acc));                              \
        (acc) = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xa_ + 8), (acc));                              \
        (acc) = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xa_ + 16), (acc));                             \
        (acc) = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xa_ + 24), (acc));                             \
    } while (0)

static void walk_gemv2(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->rows, tid, nthreads, &lo, &hi);
    long r = lo;
    for (int it = 0; it < j->iters; it++) {
        for (r = lo; r + 2 <= hi; r += 2) {
            const qe_block_q8_0 *u = (const qe_block_q8_0 *)(j->W + (size_t)r * ROW_BYTES);
            const qe_block_q8_0 *v = u + N_BLOCKS;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
            for (long b = 0; b + 4 <= N_BLOCKS; b += 4) {
                const float *x = j->x + b * 32;
                ALU_DOT_BLOCK(a0, u + b + 0, x + 0);
                ALU_DOT_BLOCK(b0, v + b + 0, x + 0);
                ALU_DOT_BLOCK(a1, u + b + 1, x + 32);
                ALU_DOT_BLOCK(b1, v + b + 1, x + 32);
                ALU_DOT_BLOCK(a2, u + b + 2, x + 64);
                ALU_DOT_BLOCK(b2, v + b + 2, x + 64);
                ALU_DOT_BLOCK(a3, u + b + 3, x + 96);
                ALU_DOT_BLOCK(b3, v + b + 3, x + 96);
            }
            j->y[r] = qe_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
            j->y[r + 1] = qe_hsum(_mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3)));
        }
        if (it + 1 == j->iters && r < hi) /* shipped math on the odd last row */
            j->y[r] = qe_dot_q8_0_ref((const qe_block_q8_0 *)(j->W + (size_t)r * ROW_BYTES),
                                      j->x, N_BLOCKS);
    }
}

/* 5. T tokens through one dequantised weight run: the shape a chunked prefill wants,
 * which this instrument found and the engine then built.
 *
 * Chunking (qe_forward_prefill) removes DRAM bytes - one read of a weight matrix
 * feeds T tokens - and until these walks existed it removed no arithmetic:
 * qe_gemv_multi_t blocked a weight walk and called the single-row kernel once per
 * token over the block, so a chunk of eight did eight times the widen, convert,
 * scale and fma of a decode step and shared only where the block happened to land.
 * That is why ingestion was 65 ms per token against a 104 ms decode and not 8x
 * faster: the bytes were shared and the work was not. These walks are the kernel
 * that does share, generated at three chunk sizes from one shape so the only
 * variable is how many tokens one dequantised block feeds; `shipped2` and `shipped4`
 * below are the same charge paid by the kernel that came out of them.
 *
 * Read the rows two ways. GB/s is charged for the weight bytes one pass reads, so it
 * says what the bus is doing; Gw/s is charged for the weights actually multiplied,
 * so it says what the arithmetic is doing. The number that answers "is a chunked
 * kernel worth writing" is the ratio this prints per token: T.00x means T tokens
 * arrived for the price of one and the machine was waiting on memory, 1.00x means each
 * token cost what a token costs on its own and the fma units were the wall, so
 * sharing the dequant buys nothing.
 *
 * Two accumulator chains per token, the same for every T, which is not the shipped
 * kernel's four - a T-token form of those needs 4T accumulators and 4T+1 of them does
 * not fit in 16 ymm registers alongside the dequantised block. Two chains is what
 * fits, so these numbers are also the honest ceiling for a real kernel, and they are
 * checked against the double reference to a tolerance rather than against the shipped
 * result bit for bit, because a different number of chains is a different summation
 * order. */
#define ALU_MULTI(name, T)                                                                        \
    static void name(void *vp, int tid, int nthreads) {                                           \
        job *j = vp;                                                                              \
        long lo, hi;                                                                              \
        qe_slice(j->rows, tid, nthreads, &lo, &hi);                                                \
        for (int it = 0; it < j->iters; it++)                                                      \
            for (long r = lo; r < hi; r++) {                                                       \
                const qe_block_q8_0 *u =                                                           \
                    (const qe_block_q8_0 *)(j->W + (size_t)r * ROW_BYTES);                         \
                __m256 a[T][2];                                                                    \
                for (int t = 0; t < T; t++) {                                                      \
                    a[t][0] = _mm256_setzero_ps();                                                 \
                    a[t][1] = a[t][0];                                                             \
                }                                                                                  \
                long b = 0;                                                                        \
                for (; b + 2 <= N_BLOCKS; b += 2) {                                                \
                    {                                                                              \
                        ALU_DEQUAD(u + b);                                                         \
                        for (int t = 0; t < T; t++)                                                \
                            ALU_FMA4(a[t][0], j->xt[t] + b * 32);                                  \
                    }                                                                              \
                    {                                                                              \
                        ALU_DEQUAD(u + b + 1);                                                     \
                        for (int t = 0; t < T; t++)                                                \
                            ALU_FMA4(a[t][1], j->xt[t] + (b + 1) * 32);                            \
                    }                                                                              \
                }                                                                                  \
                for (; b < N_BLOCKS; b++) {                                                        \
                    ALU_DEQUAD(u + b);                                                             \
                    for (int t = 0; t < T; t++)                                                    \
                        ALU_FMA4(a[t][0], j->xt[t] + b * 32);                                      \
                }                                                                                  \
                for (int t = 0; t < T; t++)                                                        \
                    j->yt[(size_t)t * j->rows + r] =                                                \
                        qe_hsum(_mm256_add_ps(a[t][0], a[t][1]));                                  \
            }                                                                                      \
    }

ALU_MULTI(walk_multi1, 1)
ALU_MULTI(walk_multi2, 2)
ALU_MULTI(walk_multi4, 4)

/* The kernel that now ships, through the same row slice every other walk takes.
 *
 * `multi1`, `multi2` and `multi4` above are the candidate and the ceiling; these
 * are the thing that runs, so that the number in the table is the number the
 * engine gets. Same span, same slice, same activation rows, and the same charge:
 * one pass of weight bytes for T tokens of arithmetic. The kernel takes x and y
 * with strides rather than pointer arrays, which is why the walk below reads
 * like a call site rather than like the walks above. */
#define SHIPPED_MULTI(name, T)                                                                     \
    static void name(void *vp, int tid, int nthreads) {                                            \
        job *j = vp;                                                                               \
        long lo, hi;                                                                               \
        qe_slice(j->rows, tid, nthreads, &lo, &hi);                                                \
        const uint8_t *w = j->W + (size_t)lo * ROW_BYTES;                                          \
        for (int it = 0; it < j->iters; it++)                                                      \
            qe_gemv_q8_0_multi(w, j->xt[0], j->yt + lo, hi - lo, K_DIM,                            \
                               K_DIM / 32, T, K_DIM, j->rows);                                     \
    }

SHIPPED_MULTI(walk_shipped2, 2)
SHIPPED_MULTI(walk_shipped4, 4)
SHIPPED_MULTI(walk_shipped8, 8)
SHIPPED_MULTI(walk_shipped16, 16)

/* 6. Two bytes per weight and no dequantisation at all, which is only worth
 * anything if the arithmetic really is the wall: it doubles the bytes to buy back
 * the conversion. Read as bf16, the span holds half as many weights, so the
 * comparison is made in Gw/s. */
static void walk_bf16(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->bf16_rows, tid, nthreads, &lo, &hi);
    for (int it = 0; it < j->iters; it++)
        qe_gemv_bf16(j->W + (size_t)lo * K_DIM * 2, j->x, j->y + lo, hi - lo, K_DIM, K_DIM);
}

/* ------------------------------------------------------------------ driver */

/* 7. The machine's own ceiling, with the row shape taken out: byte ranges divided
 * between the threads so every load is 64 B aligned, one add per vector, no
 * arithmetic worth naming. This is the walk that says what the memory controller
 * can do for this process; if the row walks land well below it, the row shape or
 * the block layout costs the difference, not the DRAM. */
static void walk_flat(void *vp, int tid, int nthreads) {
    job *j = vp;
    size_t n = ((size_t)j->rows * ROW_BYTES) & ~(size_t)63u;
    size_t chunk = (n / (size_t)nthreads) & ~(size_t)63u;
    const uint8_t *base = j->W;
    size_t lo = chunk * (size_t)tid, hi = tid + 1 == nthreads ? n : lo + chunk;
    __m256i acc = _mm256_setzero_si256();
    for (int it = 0; it < j->iters; it++)
        for (size_t o = lo; o + 256 <= hi; o += 256) {
            for (int u = 0; u < 8; u++)
                acc = _mm256_add_epi64(
                    acc, _mm256_sad_epu8(_mm256_load_si256((const __m256i *)(base + o + u * 32)),
                                         _mm256_setzero_si256()));
        }
    j->y[tid] = (float)(_mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 2));
}

/* 8, 9. The recurrent state's own shape: read a span of f32, write it back.
 *
 * docs/DESIGN.md §14 does this in arithmetic and does not pretend to have measured
 * it: the block stage reads 216 MiB of recurrent state per token and writes 216
 * MiB, and a store that is not a streaming store fetches the line before it
 * overwrites it, so the same bytes can occupy the bus three times. These two rows
 * are that shape with the delta rule taken out - one decay multiply per float,
 * which is all the recurrence is at the level of bytes - and `state` and `statent`
 * are the ordinary and the non-temporal store. Their gap is the most a streaming
 * store could buy, which is the number §14 asked for. In place, because that is
 * what the engine does: the state is read where it is and updated where it is. */
static void state_span(const job *j, int tid, int nthreads, size_t *lop, size_t *hip) {
    size_t n = ((size_t)j->rows * ROW_BYTES) & ~(size_t)255u;
    size_t chunk = (n / (size_t)nthreads) & ~(size_t)63u;
    *lop = chunk * (size_t)tid;
    *hip = tid + 1 == nthreads ? n : *lop + chunk;
}

static void walk_state_store(void *vp, int tid, int nthreads) {
    job *j = vp;
    size_t lo, hi;
    state_span(j, tid, nthreads, &lo, &hi);
    float *S = (float *)j->S;
    const __m256 decay = _mm256_set1_ps(0.999f);
    for (int it = 0; it < j->iters; it++)
        for (size_t o = lo; o + 32 <= hi; o += 32) {
            __m256 v = _mm256_load_ps(S + o / 4);
            _mm256_store_ps(S + o / 4, _mm256_mul_ps(v, decay));
        }
    j->y[tid] = S[lo / 4];
}

static void walk_state_nt(void *vp, int tid, int nthreads) {
    job *j = vp;
    size_t lo, hi;
    state_span(j, tid, nthreads, &lo, &hi);
    float *S = (float *)j->S;
    const __m256 decay = _mm256_set1_ps(0.999f);
    for (int it = 0; it < j->iters; it++)
        for (size_t o = lo; o + 32 <= hi; o += 32) {
            __m256 v = _mm256_load_ps(S + o / 4);
            _mm256_stream_ps(S + o / 4, _mm256_mul_ps(v, decay));
        }
    _mm_sfence();
    j->y[tid] = S[lo / 4];
}

/* 10, 11. The split layout: the 32 B weight runs and the 2 B scales live in separate
 * arrays, so a Q8_0 row is 2560 B of weights and 160 B of scales instead of 80
 * interleaved 34 B blocks. Same bytes per row - 2720 B either way, so the DRAM
 * traffic is identical - but the fetch is not. In this engine's q8_0 the fp16 scale
 * comes first in the block, so a row is d,32,d,32,... and every one of those weight
 * runs starts two bytes off a 32 B boundary at a 34 B stride: the shipped kernel
 * reads them as four 8 B loads, and three of every four straddle a 64 B line. The
 * split layout puts each run on its own 32 B alignment, which is one aligned load
 * per block and no line splits at all. */
static void walk_read_split(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->rows, tid, nthreads, &lo, &hi);
    for (int it = 0; it < j->iters; it++)
        for (long r = lo; r < hi; r++) {
            const uint8_t *q = j->Ww + (size_t)r * K_DIM;
            const qe_half *s = (const qe_half *)(j->Ws + (size_t)r * N_BLOCKS * 2);
            __m256i acc = _mm256_setzero_si256();
            float dsum = 0.0f;
            for (long b = 0; b < N_BLOCKS; b++) {
                acc = _mm256_add_epi64(acc,
                                       _mm256_sad_epu8(_mm256_load_si256((const __m256i *)(q + b * 32)),
                                                       _mm256_setzero_si256()));
                dsum += qe_half_to_f32(s[b]);
            }
            j->y[r] = (float)(_mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 2)) * dsum;
        }
}

static void walk_gemv_split(void *vp, int tid, int nthreads) {
    job *j = vp;
    long lo, hi;
    qe_slice(j->rows, tid, nthreads, &lo, &hi);
    for (int it = 0; it < j->iters; it++)
        for (long r = lo; r < hi; r++) {
            const uint8_t *q = j->Ww + (size_t)r * K_DIM;
            const qe_half *s = (const qe_half *)(j->Ws + (size_t)r * N_BLOCKS * 2);
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (long b = 0; b + 4 <= N_BLOCKS; b += 4) {
                const uint8_t *w = q + b * 32;
                const float *xx = j->x + b * 32;
                ALU_DOT_W(a0, _mm256_load_si256((const __m256i *)(w + 0)),
                          _mm256_set1_ps(qe_half_to_f32(s[b + 0])), xx + 0);
                ALU_DOT_W(a1, _mm256_load_si256((const __m256i *)(w + 32)),
                          _mm256_set1_ps(qe_half_to_f32(s[b + 1])), xx + 32);
                ALU_DOT_W(a2, _mm256_load_si256((const __m256i *)(w + 64)),
                          _mm256_set1_ps(qe_half_to_f32(s[b + 2])), xx + 64);
                ALU_DOT_W(a3, _mm256_load_si256((const __m256i *)(w + 96)),
                          _mm256_set1_ps(qe_half_to_f32(s[b + 3])), xx + 96);
            }
            j->y[r] = qe_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        }
}

typedef struct {
    const char *name;
    const char *key; /* what --kernels takes */
    void (*fn)(void *, int, int);
    int rowsel;            /* 0 = rows, 1 = the bf16 row count, 2 = the whole span,
                            * 3 = the whole span twice (a read and a write) */
    double bytes_per_row;  /* bytes touched per row this walk is charged */
    double bytes_per_weight;
    int tokens; /* tokens one pass over the rows produces: 1 for a decode, T for a
                 * chunk. Only used to say so in the table. */
} kind;

static int parse_cpus(const char *s, int *out, int max) {
    int n = 0;
    while (*s && n < max) {
        int a = (int)strtol(s, (char **)&s, 10);
        int b = a;
        if (*s == '-')
            b = (int)strtol(s + 1, (char **)&s, 10);
        for (int c = a; c <= b && n < max; c++)
            out[n++] = c;
        if (*s == ',')
            s++;
    }
    return n;
}

int main(int argc, char **argv) {
    const char *cpus_s = NULL;
    const char *kernels_s = "all";
    int want = 0, reps = 3, inner = 1;
    long span_mib = SPAN_MIB;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpus") && i + 1 < argc)
            cpus_s = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            want = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--span") && i + 1 < argc)
            span_mib = atol(argv[++i]);
        else if (!strcmp(argv[i], "--kernels") && i + 1 < argc)
            kernels_s = argv[++i];
        else if (!strcmp(argv[i], "--inner") && i + 1 < argc)
            inner = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: bench_alu [--cpus 0-7] [--threads N] "
                            "[--reps 3] [--span 512] [--inner 1] [--kernels LIST]\n"
                            "  LIST = all | gemv,gemv2,read,gemvsplit,readsplit,vnni,bf16,\n"
                            "         multi1,multi2,multi4,shipped2,shipped4,shipped8,shipped16,"
                            "flat,state,\n"
                            "         statent   (state walks are charged for the read\n"
                            "         and the write, and have no Gweights/s; multi* are\n"
                            "         one walk for T tokens and shipped* the kernel that\n"
                            "         came out of them, so their ratio is the row that\n"
                            "         says whether sharing a dequant was worth it)\n");
            return 2;
        }
    }

    int cpus[MAX_THREADS], ncpus = 0;
    if (cpus_s) {
        ncpus = parse_cpus(cpus_s, cpus, MAX_THREADS);
    } else {
        /* The engine's own default: the whole affinity mask, every core but the
         * last, and no cpu list, which is to say no pinning. */
        cpu_set_t cs;
        CPU_ZERO(&cs);
        sched_getaffinity(0, sizeof cs, &cs);
        for (int c = 0; c < CPU_SETSIZE && ncpus < MAX_THREADS; c++)
            if (CPU_ISSET(c, &cs))
                cpus[ncpus++] = c;
        if (ncpus > 1)
            ncpus--;
    }
    if (!want)
        want = ncpus;

    /* With one thread there is no pool to pin, and an unpinned process floats onto
     * whichever core the scheduler feels like - so when the caller named cpus, the
     * process itself goes to them. That is what makes the single-core rows of the
     * table mean what they say. */
    if (cpus_s && ncpus > 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        for (int i = 0; i < ncpus; i++)
            CPU_SET(cpus[i], &cs);
        sched_setaffinity(0, sizeof cs, &cs);
    }

    size_t n = (size_t)span_mib * 1048576;
    long rows = (long)(n / ROW_BYTES);
    uint8_t *W = qe_alloc_huge(n);
    uint8_t *Ww = qe_alloc_huge((size_t)rows * K_DIM + 4096);
    uint8_t *Ws = qe_alloc_huge((size_t)rows * N_BLOCKS * 2 + 4096);
    uint8_t *S = qe_alloc_huge(n);
    float *x = qe_alloc_huge((size_t)K_DIM * sizeof(float) + 4096);
    /* Four activation rows and four output rows per weight row: the multi-token walks
     * need a distinct token per accumulator, or the compiler could in principle fold
     * them and the measurement would be worth nothing. */
    float *xrows = qe_alloc_huge((size_t)16 * K_DIM * sizeof(float) + 4096);
    float *y = qe_alloc_huge((size_t)rows * sizeof(float) + 4096);
    float *y4 = qe_alloc_huge((size_t)16 * rows * sizeof(float) + 4096);
    int8_t *xi = qe_alloc_huge(K_DIM + 4096);
    int32_t *xc = qe_alloc_huge(N_BLOCKS * 8 * sizeof(int32_t) + 4096);
    float *xs = qe_alloc_huge(N_BLOCKS * sizeof(float) + 4096);
    if (!W || !Ww || !Ws || !S || !x || !xrows || !y || !y4 || !xi || !xc || !xs)
        return fprintf(stderr, "out of memory for a %ld MiB span\n", span_mib), 1;

    /* Weights: any finite pattern. Scales: finite and away from zero, or the
     * fp16 path produces the infinities the checker in bench_kern hunts. The split
     * arrays are the source of truth and the interleaved span is composed from
     * them, so every walk reads the same numbers out of the same number of bytes. */
    uint64_t s = 0x2545f4914f6cdd1dull;
    size_t nww = (size_t)rows * K_DIM;
    for (size_t i = 0; i < nww / 8; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        ((uint64_t *)Ww)[i] = s * 0x9e3779b97f4a7c15ull;
    }
    for (long r = 0; r < rows; r++)
        for (long b = 0; b < N_BLOCKS; b++) {
            uint16_t h = (uint16_t)(0x3800u | ((r + b * 7u) & 0x1fu));
            memcpy(Ws + (size_t)r * N_BLOCKS * 2 + b * 2, &h, 2);
        }
    /* Composed through the block struct, not through offsets, because which of the
     * scale and the weight run comes first in a 34 B block is the layout's business
     * and not this file's - and in this engine's q8_0 the scale comes first, which
     * is what makes every weight run in the interleaved span misaligned. */
    for (long r = 0; r < rows; r++) {
        qe_block_q8_0 *blk = (qe_block_q8_0 *)(W + (size_t)r * ROW_BYTES);
        const uint8_t *src = Ww + (size_t)r * K_DIM;
        const qe_half *sc = (const qe_half *)(Ws + (size_t)r * N_BLOCKS * 2);
        for (long b = 0; b < N_BLOCKS; b++) {
            blk[b].d = sc[b];
            memcpy(blk[b].qs, src + b * 32, 32);
        }
    }
    for (long i = 0; i < K_DIM; i++)
        x[i] = (float)(((i * 37) % 2000) - 1000) / 1000.0f;
    /* Token 0 of the chunk is the row every other walk uses, byte for byte, so the
     * T=1 multi row and the shipped row are comparable; the rest are different tokens,
     * which is what a chunk is. */
    memcpy(xrows, x, (size_t)K_DIM * sizeof(float));
    for (int t = 1; t < 16; t++)
        for (long i = 0; i < K_DIM; i++)
            xrows[(size_t)t * K_DIM + i] =
                (float)(((i * (37 + 11 * t)) % 1990) - 995) / 1000.0f;
    /* The state span is finite and not all zero: a buffer of zeros stays zero after
     * any number of decay multiplies, and a denormal one costs microcode assists
     * that would be read as bandwidth. */
    for (size_t i = 0; i < n / 8; i++)
        ((uint64_t *)S)[i] = 0x3f7a1a1a3f7a1a1aull + i;
    /* x as q8_0, block by block: amax over 32 weights, then round. */
    for (long b = 0; b < N_BLOCKS; b++) {
        float amax = 1e-20f;
        for (int i = 0; i < 32; i++) {
            float v = fabsf(x[b * 32 + i]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        xs[b] = d;
        int32_t lanes[8] = { 0 };
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(x[b * 32 + i] / d);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            xi[b * 32 + i] = (int8_t)q;
            lanes[i / 4] += q;
        }
        for (int l = 0; l < 8; l++)
            xc[b * 8 + l] = 128 * lanes[l];
    }

    job j = { W,  x,   xi,  xc,  xs, y, rows,
              (long)((n / 2) / (K_DIM * 2)), Ww, Ws, S, inner < 1 ? 1 : inner,
              { xrows, xrows + K_DIM, xrows + 2 * (size_t)K_DIM, xrows + 3 * (size_t)K_DIM,
                xrows + 4 * (size_t)K_DIM, xrows + 5 * (size_t)K_DIM,
                xrows + 6 * (size_t)K_DIM, xrows + 7 * (size_t)K_DIM,
                xrows + 8 * (size_t)K_DIM, xrows + 9 * (size_t)K_DIM,
                xrows + 10 * (size_t)K_DIM, xrows + 11 * (size_t)K_DIM,
                xrows + 12 * (size_t)K_DIM, xrows + 13 * (size_t)K_DIM,
                xrows + 14 * (size_t)K_DIM, xrows + 15 * (size_t)K_DIM },
              y4 };
    /* Only a named set pins: the default row has to be the pool the engine builds,
     * and that pool passes no cpu list to qe_pool_init at all. */
    qe_pool_init(want, cpus_s ? cpus : NULL, cpus_s ? ncpus : 0);

    kind kinds[] = {
        { "Q8_0 fma (shipped)", "gemv",      walk_gemv,       0, ROW_BYTES, 34.0 / 32.0, 1 },
        { "Q8_0 2 rows",        "gemv2",     walk_gemv2,      0, ROW_BYTES, 34.0 / 32.0, 1 },
        { "Q8_0 read only",     "read",      walk_read,       0, ROW_BYTES, 34.0 / 32.0, 1 },
        { "split fma",          "gemvsplit", walk_gemv_split, 0, ROW_BYTES, 34.0 / 32.0, 1 },
        { "split read only",    "readsplit", walk_read_split, 0, ROW_BYTES, 34.0 / 32.0, 1 },
        /* One dequantised weight run feeding T token accumulators. Charged for the
         * weight bytes one pass reads (that is what the bus is asked for) and for T
         * times the weights multiplied (that is what the core is asked for). */
        { "Q8_0 dequant T=1",   "multi1",    walk_multi1,     0, ROW_BYTES, 34.0 / 32.0, 1 },
        { "Q8_0 dequant T=2",   "multi2",    walk_multi2,     0, ROW_BYTES, 34.0 / 64.0, 2 },
        { "Q8_0 dequant T=4",   "multi4",    walk_multi4,     0, ROW_BYTES, 34.0 / 128.0, 4 },
        /* The same three, but the kernel that ships rather than a walk written to
         * measure the idea. Held to the same charge, so a gap between the two rows
         * of a pair is the kernel being worse than the shape it implements. */
        { "shipped multi T=2",  "shipped2",  walk_shipped2,   0, ROW_BYTES, 34.0 / 64.0, 2 },
        { "shipped multi T=4",  "shipped4",  walk_shipped4,   0, ROW_BYTES, 34.0 / 128.0, 4 },
        /* Past four tokens the shipped kernel changes shape: the dequantised block
         * moves from registers to an L1 line so that eight accumulators fit, and
         * sixteen tokens are therefore two of those rather than four of the old
         * ones. Whether the shape pays is exactly what these two rows say. */
        { "shipped multi T=8",  "shipped8",  walk_shipped8,   0, ROW_BYTES, 34.0 / 256.0, 8 },
        { "shipped multi T=16", "shipped16", walk_shipped16,  0, ROW_BYTES, 34.0 / 512.0, 16 },
#if defined(__AVXVNNI__)
        { "Q8_0 vpdpbusd",      "vnni",      walk_vnni,       0, ROW_BYTES, 34.0 / 32.0, 1 },
#endif
        { "BF16 fma",           "bf16",      walk_bf16,       1, 2.0 * K_DIM, 2.0, 1 },
        { "flat aligned read",  "flat",      walk_flat,       2, ROW_BYTES, 34.0 / 32.0, 1 },
        /* Charged for both directions: the walk reads the span and writes it back,
         * and the rate that comes out is bytes moved, which is what a store's cost
         * has to be compared in. */
        { "state store",        "state",     walk_state_store, 3, ROW_BYTES, 0.0, 1 },
        { "state nt store",     "statent",   walk_state_nt,    3, ROW_BYTES, 0.0, 1 },
    };
    /* --kernels picks the rows. Default is all of them, because the whole point of
     * the instrument is the ratio between a walk that does arithmetic and one that
     * only reads. */
    char want_k[16][16];
    int nwant_k = 0;
    if (strcmp(kernels_s, "all")) {
        const char *p = kernels_s;
        while (*p && nwant_k < 16) {
            int l = 0;
            while (p[l] && p[l] != ',' && l < 15)
                l++;
            memcpy(want_k[nwant_k], p, l);
            want_k[nwant_k][l] = 0;
            nwant_k++;
            p += l;
            if (*p == ',')
                p++;
        }
    }
    size_t nkind = 0;
    for (size_t ki = 0; ki < sizeof(kinds) / sizeof(kinds[0]); ki++) {
        int keep = !nwant_k;
        for (int w = 0; w < nwant_k; w++)
            if (!strcmp(want_k[w], kinds[ki].key))
                keep = 1;
        if (keep)
            kinds[nkind++] = kinds[ki];
    }
    if (!nkind)
        return fprintf(stderr, "no such kernel in --kernels: %s\n", kernels_s), 2;

    /* Both candidates are checked against the shipped kernel before anything is
     * timed, single threaded, over enough rows to cover every alignment a 34 B
     * stride produces. The split layout may differ in rounding (different load
     * granularity, same math); two rows in flight may not differ at all, because
     * each row's accumulators are filled in the shipped order. */
    /* Survives the checking block because the results table prints it: the
     * shared-dequant walks are reassociated by design, and "by how much" is the
     * thing a reader needs in order to trust the Gw/s column. */
    double worst_multi = 0.0;
    {
        long save = j.rows, check = rows < 4096 ? rows : 4096;
        int saved_iters = j.iters;
        float *ref = qe_alloc_huge((size_t)check * sizeof(float) + 4096);
        if (!ref)
            return fprintf(stderr, "out of memory checking\n"), 1;
        j.rows = check;
        j.iters = 1;
        walk_gemv(&j, 0, 1);
        memcpy(ref, y, (size_t)check * sizeof(float));
        double worst_split = 0.0;
        walk_gemv_split(&j, 0, 1);
        for (long r = 0; r < check; r++) {
            double err = fabs((double)y[r] - (double)ref[r]) / (fabs((double)ref[r]) + 1.0);
            if (err > worst_split)
                worst_split = err;
        }
        walk_gemv2(&j, 0, 1);
        for (long r = 0; r < check; r++)
            if (y[r] != ref[r])
                return fprintf(stderr, "two rows in flight is not the shipped arithmetic: "
                                       "row %ld gave %.9g, shipped gave %.9g\n",
                               r, y[r], ref[r]), 1;
        /* The multi-token walks are a different summation from the shipped kernel by
         * construction - two chains per token instead of four per row - so they are
         * held to the double reference and not to the shipped bits. A real bug (a
         * block skipped, a token reading another token's activations) is order 1, and
         * a reassociation is order 1e-4 at this depth (two chains over eighty blocks
         * against a double sum), so the bound below separates the two. */
        {
            worst_multi = 0.0;
            for (int T = 1; T <= 4; T *= 2) {
                if (T == 1)
                    walk_multi1(&j, 0, 1);
                else if (T == 2)
                    walk_multi2(&j, 0, 1);
                else
                    walk_multi4(&j, 0, 1);
                for (long r = 0; r < check; r += 7)
                    for (int t = 0; t < T; t++) {
                        double want_v = qe_dot_q8_0_ref(
                            (const qe_block_q8_0 *)(W + (size_t)r * ROW_BYTES), j.xt[t], N_BLOCKS);
                        double got = j.yt[(size_t)t * check + r];
                        double err = fabs(want_v - got) / (fabs(want_v) + 1.0);
                        if (err > worst_multi)
                            worst_multi = err;
                    }
            }
            if (worst_multi > 1e-3) /* an order-1 bug, not a reassociation */
                return fprintf(stderr,
                               "a shared-dequant walk disagrees with the double reference: "
                               "max relative error %.3g\n", worst_multi), 1;
        }
        /* The double reference keeps the shipped kernel itself on the hook. */
        double worst_ref = 0.0;
        for (long r = 0; r < check; r += 61) {
            double want_v = qe_dot_q8_0_ref((const qe_block_q8_0 *)(W + (size_t)r * ROW_BYTES),
                                            x, N_BLOCKS);
            double err = fabs(want_v - (double)ref[r]) / (fabs(want_v) + 1.0);
            if (err > worst_ref)
                worst_ref = err;
        }
        if (worst_ref > 1e-3) /* a real kernel bug is order 1, not order 1e-4 */
            return fprintf(stderr, "shipped kernel disagrees with the double reference: "
                                   "max relative error %.3g\n", worst_ref), 1;
        qe_free_huge(ref, (size_t)check * sizeof(float) + 4096);
        j.rows = save;
        j.iters = saved_iters;
        if (worst_split > 1e-5)
            return fprintf(stderr, "split layout disagrees with the shipped kernel: "
                                   "max relative error %.3g\n", worst_split), 1;
    }

    char pin[128];
    if (cpus_s)
        snprintf(pin, sizeof pin, "%s", cpus_s);
    else
        snprintf(pin, sizeof pin, "default pool, unpinned");
    printf("== %ld MiB span, K=%d, %ld rows, %d threads on cpus %s, %dx per pass ==\n",
           span_mib, K_DIM, rows, want, pin, inner < 1 ? 1 : inner);
    printf("  %-18s %8s %8s %10s %10s\n", "walk", "GB/s", "Gw/s", "best ms", "vs read");
    double ceiling = 0.0;
    double gb_of[16], gw_of[16], ms_of[16];
    for (size_t ki = 0; ki < nkind; ki++) {
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double a = now();
            qe_pool_run(kinds[ki].fn, &j, want);
            double d = now() - a;
            if (d < best) best = d;
        }
        /* Rows are the unit the walks are charged in: bf16 covers half the span
         * because a weight is twice the bytes, and the flat read is charged the
         * whole span because it is sliced by bytes and not by rows. */
        double bytes = kinds[ki].rowsel == 2
                           ? (double)rows * ROW_BYTES
                           : kinds[ki].rowsel == 3 ? 2.0 * (double)rows * ROW_BYTES
                           : (double)(kinds[ki].rowsel == 1 ? j.bf16_rows : rows) *
                                 kinds[ki].bytes_per_row;
        bytes *= (double)(inner < 1 ? 1 : inner);
        gb_of[ki] = bytes / best / 1e9;
        gw_of[ki] = kinds[ki].bytes_per_weight > 0.0
                        ? bytes / kinds[ki].bytes_per_weight / best / 1e9
                        : 0.0;
        ms_of[ki] = best * 1e3 / (double)(inner < 1 ? 1 : inner);
        if (!strcmp(kinds[ki].key, "read"))
            ceiling = gb_of[ki];
        fflush(stdout);
    }
    for (size_t ki = 0; ki < nkind; ki++) {
        char gws[16];
        if (kinds[ki].bytes_per_weight > 0.0)
            snprintf(gws, sizeof gws, "%.1f", gw_of[ki]);
        else
            snprintf(gws, sizeof gws, "-"); /* no weights in this walk */
        printf("  %-18s %8.1f %8s %10.1f %9.0f%%\n", kinds[ki].name, gb_of[ki], gws,
               ms_of[ki], ceiling > 0 ? 100.0 * gb_of[ki] / ceiling : 100.0);
    }
    printf("  5568 MiB at the read walk's rate: %.1f ms/token of weights\n", 5568.0 * 1048576.0 /
           (1e9 * (ceiling > 0 ? ceiling : 1)) * 1e3);
    /* The chunk walks, restated in the unit the decision is made in. A multiT row
     * walks the weight bytes once and produces T tokens, so the "best ms" in the
     * table above is the price of a chunk of T and the price of a token is that over
     * T; read straight, that column says a chunk of four is slower than a decode,
     * which is true of the pass and false of the token.
     *
     * Then the ratio against the T=1 row, which is the only reason these walks exist.
     * Tx means T tokens came for the price of one: the machine was waiting on memory,
     * and a kernel that shares its dequant across a chunk is worth about T. 1.00x
     * means the second token cost exactly what the first cost: the fma units were the
     * wall, and sharing the dequant buys nothing. Anything between is the fraction of
     * a chunk the arithmetic can still hide - and it is this number, not a guess, that
     * says whether qe_gemv_multi_t should be rewritten. */
    {
        long one = -1;
        for (size_t ki = 0; ki < nkind; ki++)
            if (!strcmp(kinds[ki].key, "multi1"))
                one = (long)ki;
        if (one >= 0) {
            int any = 0;
            printf("  shared dequant, per token:");
            for (size_t ki = 0; ki < nkind; ki++)
                if (kinds[ki].tokens > 1) {
                    printf(" %s %.2f ms", kinds[ki].key, ms_of[ki] / kinds[ki].tokens);
                    any = 1;
                }
            if (any)
                printf("\n");
            else
                printf(" (only the T=1 row ran)\n");
            for (size_t ki = 0; ki < nkind; ki++)
                if (kinds[ki].tokens > 1 && ms_of[ki] > 0.0)
                    printf("  tokens/s vs one token, %s: %.2fx "
                           "(1.00x = the fma units were the wall, %.2fx = memory was)\n",
                           kinds[ki].key, (double)kinds[ki].tokens * ms_of[one] / ms_of[ki],
                           (double)kinds[ki].tokens);
            printf("  shared dequant vs the double reference: %.1e max relative error. Every\n"
                   "  walk here reassociates by construction - two chains per token, not the\n"
                   "  shipped kernel's four - so they are held to a tolerance and were never\n"
                   "  expected to agree bit for bit with a decode step.\n", worst_multi);
        }
    }

    printf("  note: the read walk is a bound on the ceiling, not the ceiling. One load and\n"
           "  one vpsadbw per block, and the per-row scale reads and horizontal sum that a\n"
           "  GEMV pays too. Anything that beats it is reading bytes the walk did not.\n");
    qe_pool_shutdown();
    return 0;
}
