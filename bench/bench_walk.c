/* What the dense half of an ingested token costs, and what would make it cheaper.
 *
 * Six questions, all answered on synthetic tensors rather than the checkpoint,
 * because all three are properties of a kernel and not of a model:
 *
 *   1. How fast is the chunked walk on the shapes the model actually has, and is
 *      the row block the forward pass uses (`QE_ROW_BLOCK`) the right one for
 *      them? `bench_kern` walks one big square-ish matrix; the forward pass walks
 *      `attn_qkv` at 10240x2560 and a mixer down at 320x10240, which have almost
 *      nothing in common - one has 32x the rows of the other and a K a quarter as
 *      long, so the activations a thread holds while it walks a row block differ
 *      by the same factor. Where a shape is far from the machine's rate, the row
 *      block is the first knob to try, so this prints both.
 *
 *   2. Would reading fewer bytes make the walk faster? Q8_0 is what every dense
 *      projection in this checkpoint is. IQ4_NL is 2.25x smaller per weight and
 *      IQ3_S 2.48x, and both have a kernel here already - the shared-dequant one
 *      the expert pass uses, and the ordinary one. If time fell with bytes,
 *      requantising the dense half would be the biggest item in the repository;
 *      the row below is what it is worth instead.
 *
 *   3. Would one load of the activations feeding more than one output row make the
 *      walk faster? A row of weights and a row of activations are read in the same
 *      pass, so a walk of two output rows reads the activations twice and the
 *      weights once each. The pairing forms - `kern_iq.c`'s for the expert walk,
 *      `kern.c`'s Q8_0 form for this file's own dense walk - hold two dequantised
 *      blocks and one set of activation vectors; section 3 prices both against the
 *      walks they replace and holds each to it at a difference of exactly zero, at
 *      every token width from 2 to 512 and an odd row count so that a row with no
 *      partner is checked too.
 *
 *   4. Would the same sharing be worth more on the recoded panel, where the walk also
 *      owes a scale exit every 32 weights and reads an activation row that is a quarter
 *      of the bytes of an f32 one? Section 4 walks the same two tensors with the
 *      activations Q8_0-recoded - the panel the grouped expert pass hands the integer
 *      kernels - and prices `QE_I8_R=2` against `QE_I8_R=1` for the two expert formats,
 *      with the Q8_0 column as the control: same tile, no codebook decode in the loop
 *      to share. Same exactness bound, same widths.
 *
 *   5. Would one activation scale for a whole row instead of one per 32 values make the
 *      integer walk faster? The exit an output row pays every 32 weights multiplies the
 *      accumulator by the product of the two scales; with one scale per row that product is
 *      no longer a function of the block, so the accumulator runs in weight-scale space and
 *      the row's scale multiplies it once after the block loop. Section 4 ends by pricing
 *      that against the walk it ships with, on both expert formats, with Q8_0 - whose walk
 *      reads the row-scaled panel as the per-block panel it is - as the control. What it
 *      costs is range, so the leg prints the difference it leaves as well as the rate.
 *
 *   6. Would a pass over more than eight token rows make a bucket cheaper? A pass costs the
 *      codebook decode of every block of an output row plus one dot per token, so the tokens a
 *      pass holds are the cheap part of it and the passes a group needs are not. Eight was the
 *      widest class the codebook walks had, and the routing hands them ten: 512 experts, ten
 *      picks a token, so a 512-token chunk's modal bucket walked an expert's weights twice.
 *      Section 6 instantiates ten-, twelve- and sixteen-token passes, divides a group by the
 *      fewest passes a cost model says they need, holds every one of them to the eight-token
 *      walk's bits at the widths a bucket arrives at, and then prints the four ceilings against
 *      each other at nine to thirty-two token rows.
 *
 * Bytes are what the kernel walks, not what arithmetic says it should walk: every
 * buffer here is many times the 30 MiB L3, so a pass is a DRAM walk at T=1 and a
 * reuse problem at T=chunk width, which is the situation the engine is in. */
#define _GNU_SOURCE
#include "qe.h"
#include "kern.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXT 516
#define K_ 2560
#define ROWS_ 400000 /* 1.0 GiB of Q8_0: nothing of it survives in the 30 MiB L3 */

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* The 1-minute load average, printed next to a row that reads as a loss. These are timing
 * gates over a 19-thread pool, and a row that measures 0.91x while somebody else is
 * compiling on the box says nothing about the kernel - the same row reads 1.19x on a quiet
 * one, three loads running. The pool is in this number too, so read it as "busy" or "not",
 * not as a count of strangers. -1 when it cannot be read. */
static double load_average_1m(void) {
    FILE *f = fopen("/proc/loadavg", "r");
    if (!f)
        return -1.0;
    double v = -1.0;
    if (fscanf(f, "%lf", &v) != 1)
        v = -1.0;
    fclose(f);
    return v;
}

/* A walk of `rows` rows of one type, T activations against each of them. */
typedef struct {
    qe_type t;
    const char *w;
    const float *x;
    float *y;
    long rows, K;
    size_t row_bytes;
    long ldd_blocks;
    int T;
    size_t block;
    int shared; /* one dequantisation per weight block, T accumulators */
} walk_job;

static void walk_task(void *v, int tid, int nt) {
    walk_job *j = v;
    long lo, hi;
    qe_slice(j->rows, tid, nt, &lo, &hi);
    if (lo >= hi)
        return;
    if (j->shared) {
        const float *xs[MAXT];
        float *ys[MAXT];
        for (int t = 0; t < j->T; t++) {
            xs[t] = j->x + (size_t)t * (size_t)j->K;
            ys[t] = j->y + (size_t)t * (size_t)j->rows + lo;
        }
        if (qe_gemv_multi_ptrs_t(j->t, j->w + (size_t)lo * j->row_bytes, xs, ys, j->T, hi - lo,
                                 j->K, j->ldd_blocks) != 0) {
            fprintf(stderr, "no shared-dequant kernel for this type\n");
            exit(1);
        }
        return;
    }
    if (qe_gemv_multi_t(j->t, j->w + (size_t)lo * j->row_bytes, j->x, j->y + lo, hi - lo, j->K,
                        j->ldd_blocks, j->T, j->K, j->rows, (long)j->row_bytes, j->block) != 0) {
        fprintf(stderr, "no kernel for this type\n");
        exit(1);
    }
}

/* Every fp16 scale field of a synthetic tensor set to 1.0. Random bytes put a
 * denormal scale in most blocks, and a denormal multiply is a microcode assist,
 * which measures the assist and not the kernel. */
static void fill(char *w, size_t n, size_t block) {
    for (size_t b = 0; (b + 1) * block <= n; b++) {
        char *p = w + b * block;
        p[0] = 0x00;
        p[1] = 0x3c;
        for (size_t i = 2; i < block; i++)
            p[i] = (char)((int)((b * 37 + i * 11) % 200) - 100);
    }
}

/* The activations: values in [-1,1). The subtraction is done in a signed type on
 * purpose. Written over a `size_t` index the same expression wraps whenever the
 * residue is under 1000, and the kernel is then walked over inputs of about 1e19 -
 * finite, so nothing complains, with products at the top of the float range. The
 * rate would be believable and the outputs would not, which is the failure worth
 * designing out rather than discovering once a run. */
static void fill_x(float *x, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const long v = (long)((i * 29 + 3) % 2000) - 1000;
        x[i] = (float)v * 1e-3f;
        if (fabs((double)x[i]) > worst)
            worst = fabs((double)x[i]);
    }
    if (!(worst < 1.1)) {
        fprintf(stderr, "activations are not in range: largest |x| is %g\n", worst);
        exit(1);
    }
}

/* The A/B tables below walk the same tensor twice, once per leg, and take the minimum of
 * a few reps because this is a host with other users on it: a load average that belongs
 * to somebody else has stolen a leg before now, and a min over three reps does not tell
 * that apart from a kernel that lost. A row that reads as a loss therefore gets up to
 * three passes over the same reps, which costs seconds and keeps the gate one a run can
 * be trusted on. Three passes of losing is a loss. */
#define MEASURE_PASSES 3

/* --- section 1: the model's own dense shapes, at every row block --- */

static const struct { const char *name; long n, K; } shapes[] = {
    { "attn_qkv   10240x2560", 10240, 2560 },
    { "attn_q     12288x2560", 12288, 2560 },
    { "ssm_out     2560x6144",  2560, 6144 },
    { "attn_out    2560x6144",  2560, 6144 },
    { "hc_down      320x10240",   320, 10240 },
    { "hc_up      10240x320",   10240, 320 },
    { "sh_gate      640x2560",   640, 2560 },
    { "sh_down     2560x640",   2560, 640 },
};

/* A shape this small would be walked from the L3, and the point is the rate, so
 * several copies are walked per pass instead. */
#define COPIES 4

static void section_shapes(int nthr) {
    long max_n = 0, max_K = 0;
    for (unsigned s = 0; s < sizeof shapes / sizeof *shapes; s++) {
        if (shapes[s].n > max_n)
            max_n = shapes[s].n;
        if (shapes[s].K > max_K)
            max_K = shapes[s].K;
    }
    const size_t rb_max = (size_t)(max_K / 32) * sizeof(qe_block_q8_0);
    char *w = qe_alloc_huge((size_t)max_n * rb_max * COPIES);
    float *x = qe_alloc_huge((size_t)MAXT * (size_t)max_K * sizeof(float));
    float *y = qe_alloc_huge((size_t)max_n * COPIES * sizeof(float) * MAXT);
    if (!w || !x || !y) {
        fprintf(stderr, "alloc\n");
        exit(1);
    }
    fill(w, (size_t)max_n * rb_max * COPIES, sizeof(qe_block_q8_0));
    fill_x(x, (size_t)MAXT * (size_t)max_K);
    memset(y, 0, (size_t)max_n * COPIES * sizeof(float) * MAXT);

    static const int TW[] = { 1, 16, 64, 128, 256, 512 };
    static const size_t BW[] = { 64u << 10, 256u << 10, 1u << 20, 8u << 20 };
    printf("== the walk on the model's own dense shapes, Q8_0, %d copies per pass ==\n",
           COPIES);
    for (int wi = 0; wi < (int)(sizeof TW / sizeof *TW); wi++) {
        const int T = TW[wi];
        printf("--- T=%d: a chunk of %d tokens through one walk of the weights ---\n", T, T);
        printf("  %-22s", "shape");
        for (unsigned bi = 0; bi < sizeof BW / sizeof *BW; bi++)
            printf("%11zu%s", BW[bi] >> 10, BW[bi] == (1u << 20) ? " KiB*" : " KiB ");
        printf("   best GB/s  GB/s as shipped\n");
        for (unsigned s = 0; s < sizeof shapes / sizeof *shapes; s++) {
            const long n = shapes[s].n, K = shapes[s].K;
            const size_t rb = (size_t)(K / 32) * sizeof(qe_block_q8_0);
            const double bytes = (double)n * COPIES * (double)rb;
            double best_gbs = 0.0, shipped_gbs = 0.0;
            printf("  %-22s", shapes[s].name);
            for (unsigned bi = 0; bi < sizeof BW / sizeof *BW; bi++) {
                walk_job j = { QE_Q8_0, w, x, y, n * COPIES, K, rb, K / 32, T, BW[bi], 0 };
                double best = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    double t0 = now();
                    qe_pool_run(walk_task, &j, nthr);
                    double dt = now() - t0;
                    if (dt < best)
                        best = dt;
                }
                const double macs = (double)n * COPIES * (double)K * (double)T;
                const double gbs = bytes / best / 1e9;
                if (gbs > best_gbs)
                    best_gbs = gbs;
                if (BW[bi] == (1u << 20))
                    shipped_gbs = gbs;
                printf("%12.0f", macs / best / 1e9);
                fflush(stdout);
            }
            printf("   %8.1f   %8.1f%s\n", best_gbs, shipped_gbs,
                   shipped_gbs < 0.95 * best_gbs ? "  <- 5% off best" : "");
            fflush(stdout);
        }
    }
    printf("* the row block the forward pass uses (QE_ROW_BLOCK)\n");
    printf("GMAC/s columns are 10^9 multiply-accumulates per second, whole machine\n\n");
    /* qe_alloc_huge is an mmap, so it is released with its own call: freeing it
     * with the allocator aborts on the first one. */
    qe_free_huge(w, (size_t)max_n * rb_max * COPIES);
    qe_free_huge(x, (size_t)MAXT * (size_t)max_K * sizeof(float));
    qe_free_huge(y, (size_t)max_n * COPIES * sizeof(float) * MAXT);
}

/* --- section 2: the same walk with smaller weights --- */

/* Whether the shared-dequant kernel and the per-row kernel agree on the same
 * bytes. `make fast` holds each of them to a double reference; this holds them to
 * each other, which is the property the grouped expert pass relies on when it
 * swaps one for the other. */
static void check_shared(qe_type t, const char *w, size_t row_bytes, long ldd_blocks,
                         const float *x, float *yp, float *ys, long rows, int T) {
    walk_job a = { t, w, x, yp, rows, K_, row_bytes, ldd_blocks, T, 1u << 20, 0 };
    walk_job b = { t, w, x, ys, rows, K_, row_bytes, ldd_blocks, T, 1u << 20, 1 };
    qe_pool_run(walk_task, &a, qe_pool_size());
    qe_pool_run(walk_task, &b, qe_pool_size());
    double worst = 0.0, scale = 1.0;
    for (long r = 0; r < rows; r++)
        for (int t2 = 0; t2 < T; t2++) {
            double d = fabs((double)ys[(size_t)t2 * rows + r] -
                            (double)yp[(size_t)t2 * rows + r]);
            double m = fabs((double)yp[(size_t)t2 * rows + r]);
            if (m > scale)
                scale = m;
            if (d > worst)
                worst = d;
        }
    const int ok = worst < 2e-3 * scale;
    printf("  %-8s shared vs per-row over %ld rows x %d tokens: worst diff %.3g of a "
           "largest |y| of %.3g  %s\n",
           t == QE_IQ4_NL ? "IQ4_NL" : "IQ3_S", rows, T, worst, scale, ok ? "ok" : "FAIL");
    if (!ok)
        exit(1);
}

static void section_bytes(int nthr) {
    const size_t q8_rb = (size_t)(K_ / 32) * sizeof(qe_block_q8_0);
    const size_t i4_rb = (size_t)(K_ / 32) * sizeof(qe_block_iq4_nl);
    const size_t i3_rb = (size_t)(K_ / 256) * sizeof(qe_block_iq3_s);
    const size_t n8 = (size_t)ROWS_ * q8_rb, n4 = (size_t)ROWS_ * i4_rb,
                 n3 = (size_t)ROWS_ * i3_rb;
    char *w8 = qe_alloc_huge(n8);
    char *w4 = qe_alloc_huge(n4);
    char *w3 = qe_alloc_huge(n3);
    float *x = qe_alloc_huge((size_t)MAXT * (size_t)K_ * sizeof(float));
    float *y = qe_alloc_huge((size_t)ROWS_ * sizeof(float) * MAXT);
    if (!w8 || !w4 || !w3 || !x || !y) {
        fprintf(stderr, "alloc\n");
        exit(1);
    }
    fill(w8, n8, sizeof(qe_block_q8_0));
    fill(w4, n4, sizeof(qe_block_iq4_nl));
    fill(w3, n3, sizeof(qe_block_iq3_s));
    fill_x(x, (size_t)MAXT * (size_t)K_);
    memset(y, 0, (size_t)ROWS_ * sizeof(float) * MAXT);

    printf("== the same walk with smaller weights: K=%d, %d rows (%.1f GiB Q8_0, %.1f "
           "IQ4_NL, %.1f IQ3_S) ==\n",
           K_, ROWS_, (double)n8 / 1073741824.0, (double)n4 / 1073741824.0,
           (double)n3 / 1073741824.0);

    /* The two kernels first, on a slice small enough to hold twice. */
    {
        const long crows = 4096;
        float *yp = qe_alloc_huge((size_t)crows * 16 * sizeof(float));
        float *ys = qe_alloc_huge((size_t)crows * 16 * sizeof(float));
        if (!yp || !ys) {
            fprintf(stderr, "alloc\n");
            exit(1);
        }
        check_shared(QE_IQ4_NL, w4, i4_rb, K_ / 32, x, yp, ys, crows, 16);
        check_shared(QE_IQ3_S, w3, i3_rb, K_ / 256, x, yp, ys, crows, 16);
        qe_free_huge(yp, (size_t)crows * 16 * sizeof(float));
        qe_free_huge(ys, (size_t)crows * 16 * sizeof(float));
    }

    static const int TW[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    printf("%-5s %10s %9s %10s %9s %10s %9s   %-9s %s\n", "T", "Q8_0 ms", "Q8 GB/s",
           "IQ4 sh ms", "IQ4 GB/s", "IQ3 sh ms", "IQ3 GB/s", "Q8 vs IQ4", "Q8 vs IQ3");
    for (int wi = 0; wi < (int)(sizeof TW / sizeof *TW); wi++) {
        const int T = TW[wi];
        double best[3] = { 1e30, 1e30, 1e30 };
        for (int rep = 0; rep < 3; rep++)
            for (int which = 0; which < 3; which++) {
                walk_job j = { which == 0 ? QE_Q8_0 : (which == 1 ? QE_IQ4_NL : QE_IQ3_S),
                               which == 0 ? w8 : (which == 1 ? w4 : w3),
                               x,
                               y, ROWS_, K_,
                               which == 0 ? q8_rb : (which == 1 ? i4_rb : i3_rb),
                               which == 2 ? K_ / 256 : K_ / 32,
                               T, 1u << 20, which != 0 && T >= 2 };
                double t0 = now();
                qe_pool_run(walk_task, &j, nthr);
                double dt = now() - t0;
                if (dt < best[which])
                    best[which] = dt;
                /* A walk that wrote nothing would time beautifully. */
                if (!(y[7] == y[7]) || y[7] == 0.0f) {
                    fprintf(stderr, "the walk wrote nothing useful at T=%d\n", T);
                    exit(1);
                }
            }
        printf("%-5d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f   %8.2fx %8.2fx\n", T, best[0] * 1e3,
               (double)n8 / best[0] / 1e9, best[1] * 1e3, (double)n4 / best[1] / 1e9,
               best[2] * 1e3, (double)n3 / best[2] / 1e9, best[0] / best[1], best[0] / best[2]);
        fflush(stdout);
    }
    printf("GB/s is that column's own bytes: IQ4_NL is %.2fx fewer bytes than Q8_0, so a\n"
           "ratio near 1.00 means the smaller tensor bought nothing.\n",
           (double)q8_rb / (double)i4_rb);
    qe_free_huge(w8, n8);
    qe_free_huge(w4, n4);
    qe_free_huge(w3, n3);
    qe_free_huge(x, (size_t)MAXT * (size_t)K_ * sizeof(float));
    qe_free_huge(y, (size_t)ROWS_ * sizeof(float) * MAXT);
}

/* --- section 3: one load of the activations feeding two output rows --- */

/* What the pairing is owed: the same bits.
 *
 * The pairing holds two dequantised weight blocks beside one set of activation vectors
 * where the walk above holds one of each, and it is a change of loads and not of
 * arithmetic - so the bound here is that it leaves the row-at-a-time walk's answer, bit
 * for bit, at every group width the dispatcher knows about. Taking that instead of a
 * tolerance is what keeps a loss table out of this: the engine's next token either does
 * or does not move when the switch flips, and `make forward`'s ingestion legs read the
 * same tokens either way. A tolerance would let a reordered chain in, and a reordered
 * chain in this family is a number nobody has measured (`check_iq_multi` in `make fast`
 * holds both walks to the double reference at every group size, padded row and
 * per-token-different row it knows about, which is the accuracy question; this is the
 * identity question, and it has a sharper instrument).
 *
 * The row count is odd on purpose: the pair path stops one row short of an odd tail and
 * hands that row back to the row-at-a-time walk, and a row touched twice or not at all
 * shows up here rather than in a loss table. */
static void check_pair(qe_type t, const char *label, const char *w, size_t row_bytes,
                       long ldd_blocks, const float *x, float *buf, long rows, int T, int shared) {
    float *ylone = buf, *ypair = buf + (size_t)MAXT * (size_t)rows;
    walk_job a = { t, w, x, ylone, rows, K_, row_bytes, ldd_blocks, T, 1u << 20, shared };
    walk_job b = { t, w, x, ypair, rows, K_, row_bytes, ldd_blocks, T, 1u << 20, shared };
    /* The pairing is gated per format now, and this host ships one of the two formats off
     * (src/kern.h has the numbers), so toggling the master switch alone would compare a walk
     * with itself and print the result as an identity. Force this format's gate open for the
     * duration, so what is being compared is the form and not the default; a format whose
     * dispatcher has no gate returns 0 from the getter and is left exactly as it was. */
    const int gate = qe_iq_pair_min_t(t);
    if (gate == 0)
        qe_iq_set_pair_min_t(t, 1);
    qe_iq_set_two_row(0);
    qe_pool_run(walk_task, &a, qe_pool_size());
    qe_iq_set_two_row(1);
    qe_pool_run(walk_task, &b, qe_pool_size());
    if (gate == 0)
        qe_iq_set_pair_min_t(t, gate);
    long moved = 0;
    for (long i = 0; i < (long)((size_t)T * rows); i++)
        if (ypair[i] != ylone[i])
            moved++;
    printf("  %-8s paired over %ld rows x %d tokens: %ld of %ld outputs moved  %s\n", label, rows,
           T, moved, (long)((size_t)T * rows), moved ? "FAIL" : "ok");
    fflush(stdout);
    if (moved)
        exit(1);
}

static void section_pairs(int nthr) {
    const size_t i4_rb = (size_t)(K_ / 32) * sizeof(qe_block_iq4_nl);
    const size_t i3_rb = (size_t)(K_ / 256) * sizeof(qe_block_iq3_s);
    const size_t q8_rb = (size_t)(K_ / 32) * sizeof(qe_block_q8_0);
    const long DROWS_ = 100000; /* 260 MiB of Q8_0: the dense walk's own tensor */
    const size_t n4 = (size_t)ROWS_ * i4_rb, n3 = (size_t)ROWS_ * i3_rb;
    const size_t nq = (size_t)DROWS_ * q8_rb;
    char *w4 = qe_alloc_huge(n4);
    char *w3 = qe_alloc_huge(n3);
    char *wq = qe_alloc_huge(nq);
    float *x = qe_alloc_huge((size_t)MAXT * (size_t)K_ * sizeof(float));
    float *y = qe_alloc_huge((size_t)ROWS_ * sizeof(float) * MAXT);
    if (!w4 || !w3 || !wq || !x || !y) {
        fprintf(stderr, "alloc\n");
        exit(1);
    }
    fill(w4, n4, sizeof(qe_block_iq4_nl));
    fill(w3, n3, sizeof(qe_block_iq3_s));
    fill(wq, nq, sizeof(qe_block_q8_0));
    fill_x(x, (size_t)MAXT * (size_t)K_);
    memset(y, 0, (size_t)ROWS_ * sizeof(float) * MAXT);

    printf("== one load of the activations feeding two output rows: K=%d, %d rows (%.1f "
           "IQ4_NL, %.1f IQ3_S), %ld rows of Q8_0 (%.2f) ==\n",
           K_, ROWS_, (double)n4 / 1073741824.0, (double)n3 / 1073741824.0, DROWS_,
           (double)nq / 1073741824.0);
    {
        const long crows = 4103; /* odd: the last row has no partner */
        float *buf = qe_alloc_huge((size_t)2 * MAXT * (size_t)crows * sizeof(float));
        if (!buf) {
            fprintf(stderr, "alloc\n");
            exit(1);
        }
        /* Every width a dispatcher has a form for, then the widths a chunk and a bucket
         * actually arrive at: the gate that keeps a group's leftover tokens on the old
         * forms is per call, so an identity checked at eight and asserted at five hundred
         * and twelve is an identity argued, not measured. */
        static const int TT[] = { 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                  13, 14, 15, 16, 17, 24, 32, 40, 64, 512 };
        for (int ti = 0; ti < (int)(sizeof TT / sizeof *TT); ti++) {
            const int T = TT[ti];
            check_pair(QE_IQ4_NL, "IQ4_NL", w4, i4_rb, K_ / 32, x, buf, crows, T, 1);
            check_pair(QE_IQ3_S, "IQ3_S", w3, i3_rb, K_ / 256, x, buf, crows, T, 1);
            check_pair(QE_Q8_0, "Q8_0", wq, q8_rb, K_ / 32, x, buf, crows, T, 0);
        }
        qe_free_huge(buf, (size_t)2 * MAXT * (size_t)crows * sizeof(float));
    }

    /* Four is the width the pairing is switched off at, eight the width it replaces, and
     * eleven and thirteen what a bucket does when its width is not a multiple of eight: a
     * group's leftover tokens are walked one row at a time beside the paired eights, so
     * those two rows are the price of keeping the arithmetic identical.
     *
     * A row is gated only where the format's own dispatcher pairs, and since this file raced
     * the two codebook formats against each other that is no longer the same width for both:
     * the two constants are QE_IQ4_NL_PAIR_MIN_T and QE_IQ3_S_PAIR_MIN_T, with the numbers
     * that chose them next to them. */
    static const int TW[] = { 4, 8, 11, 13, 16, 32 };
    /* The two formats do not agree about whether the second accumulator pays, so the gate is
     * per format (QE_IQ4_NL_PAIR_MIN_T and QE_IQ3_S_PAIR_MIN_T in src/kern.h, with the rows of
     * numbers that chose them beside them). Both columns still race the form - the gate is
     * forced open for this table and put back after - but only a format that actually ships it
     * is gated on its number, because asserting on the other one would make this leg fail for
     * the measurement that is the reason for the gate. What is printed unshipped is printed as
     * such, and if it ever climbs back above 1.05x the row says so. */
    const int ship4 = qe_iq_pair_min_t(QE_IQ4_NL), ship3 = qe_iq_pair_min_t(QE_IQ3_S);
    qe_iq_set_pair_min_t(QE_IQ4_NL, 1);
    qe_iq_set_pair_min_t(QE_IQ3_S, 1);
    printf("%-5s %11s %11s %9s %11s %11s %9s\n", "T", "IQ4 1-row", "IQ4 2-row", "IQ4 gain",
           "IQ3 1-row", "IQ3 2-row", "IQ3 gain");
    for (int wi = 0; wi < (int)(sizeof TW / sizeof *TW); wi++) {
        const int T = TW[wi];
        double best[2][2] = { { 1e30, 1e30 }, { 1e30, 1e30 } };
        /* Below eight the dispatchers do not pair at all - the pair form is the eight-token
         * one, and a narrower group never reaches it - so the two legs are the same code and
         * the row is this run's noise floor rather than a result. 13 % has been measured
         * between two identical walks of 4 ms, which is a reason to print it and not a reason
         * to gate on it. Same for a format this host ships the pairing off for entirely. */
        const int on4 = ship4 > 0 && T >= ship4, on3 = ship3 > 0 && T >= ship3;
        const double floor4 = on4 ? 1.0 : 0.0, floor3 = on3 ? 1.0 : 0.0;
        int passes = 0;
        for (;;) {
            for (int rep = 0; rep < 3; rep++)
                for (int leg = 0; leg < 2; leg++)
                    for (int which = 0; which < 2; which++) {
                        qe_iq_set_two_row(leg);
                        walk_job j = { which == 0 ? QE_IQ4_NL : QE_IQ3_S,
                                       which == 0 ? w4 : w3, x, y, ROWS_, K_,
                                       which == 0 ? i4_rb : i3_rb,
                                       which == 0 ? K_ / 32 : K_ / 256, T, 1u << 20, 1 };
                        double t0 = now();
                        qe_pool_run(walk_task, &j, nthr);
                        double dt = now() - t0;
                        if (dt < best[leg][which])
                            best[leg][which] = dt;
                        /* A walk that wrote nothing would time beautifully. */
                        if (!(y[7] == y[7]) || y[7] == 0.0f) {
                            fprintf(stderr, "the walk wrote nothing useful at T=%d\n", T);
                            exit(1);
                        }
                    }
            qe_iq_set_two_row(1);
            if (++passes >= MEASURE_PASSES || (best[0][0] / best[1][0] >= floor4 &&
                                               best[0][1] / best[1][1] >= floor3))
                break;
        }
        const double g4 = best[0][0] / best[1][0], g3 = best[0][1] / best[1][1];
        char note[112] = "";
        if (g4 < floor4 || g3 < floor3)
            snprintf(note, sizeof note, "  <- the pairing is slower (load average %.2f)",
                     load_average_1m());
        else if (T >= 8 && ((!on4 && g4 >= 1.05) || (!on3 && g3 >= 1.05)))
            snprintf(note, sizeof note, "  <- a pairing this host ships off measured %.2fx: %s",
                     !on4 && g4 >= 1.05 ? g4 : g3,
                     !on4 && g4 >= 1.05 ? "iq4_nl pairing is above 1" : "iq3_s pairing is above 1");
        else if (T < 8 && (g4 < 0.95 || g4 > 1.05 || g3 < 0.95 || g3 > 1.05))
            snprintf(note, sizeof note, "  <- the same walk on both legs, so this is the noise");
        else if (passes > 1)
            snprintf(note, sizeof note, "  <- re-measured over %d passes", passes);
        printf("%-5d %11.1f %11.1f %8.2fx %11.1f %11.1f %8.2fx%s\n", T, best[0][0] * 1e3,
               best[1][0] * 1e3, g4, best[0][1] * 1e3, best[1][1] * 1e3, g3, note);
        fflush(stdout);
        if (g4 < floor4 || g3 < floor3)
            exit(1);
    }
    qe_iq_set_pair_min_t(QE_IQ4_NL, ship4);
    qe_iq_set_pair_min_t(QE_IQ3_S, ship3);
    printf("ms is a whole walk of the tensor; gain is one-row over two-row, so 1.00x means the\n"
           "second accumulator bought nothing. The gate is %d for iq4_nl and %d for iq3_s\n"
           "(0 = never); both columns above raced the form regardless, and the expert pass takes\n"
           "the gate through the same switch, which `make moe` prices on the mixed bucket a\n"
           "bucket actually walks.\n",
           ship4, ship3);

    /* The dense walk, on the switch the expert walk already uses. The engine's chunk width
     * is the last row because that is the width a chunk actually ingests at, and a tile of
     * 66 weight rows against a 512-token panel is a different machine from a bucket of
     * thirty-two tokens against eight rows. */
    static const int DT[] = { 4, 8, 16, 32, 512 };
    printf("\n%-5s %11s %11s %9s  (%ld rows of Q8_0, the dense walk the engine's projections use)\n",
           "T", "Q8_0 1-row", "Q8_0 2-row", "Q8_0 gain", DROWS_);
    for (int wi = 0; wi < (int)(sizeof DT / sizeof *DT); wi++) {
        const int T = DT[wi];
        double best[2] = { 1e30, 1e30 };
        /* Four is below the group the pair form exists for, so both legs are the same walk
         * and the row prints as this run's noise floor; see the expert table above. */
        const double floor_ = T >= 8 ? 1.0 : 0.0;
        int passes = 0;
        const int reps = T >= 256 ? 5 : 3; /* the 512-token row is 200 ms of nineteen
                                                       * threads, and a min of three is a thin
                                                       * estimate of that */
        for (;;) {
            for (int rep = 0; rep < reps; rep++)
                for (int leg = 0; leg < 2; leg++) {
                    qe_iq_set_two_row(leg);
                    walk_job j = { QE_Q8_0, wq, x, y, DROWS_, K_, q8_rb, K_ / 32, T, 1u << 20, 0 };
                    double t0 = now();
                    qe_pool_run(walk_task, &j, nthr);
                    double dt = now() - t0;
                    if (dt < best[leg])
                        best[leg] = dt;
                    if (!(y[7] == y[7]) || y[7] == 0.0f) {
                        fprintf(stderr, "the dense walk wrote nothing useful at T=%d\n", T);
                        exit(1);
                    }
                }
            qe_iq_set_two_row(1);
            if (++passes >= MEASURE_PASSES || best[0] / best[1] >= floor_)
                break;
        }
        const double g = best[0] / best[1];
        char note[112] = "";
        if (g < floor_)
            snprintf(note, sizeof note, "  <- the pairing is slower (load average %.2f)",
                     load_average_1m());
        else if (T < 8 && (g < 0.95 || g > 1.05))
            snprintf(note, sizeof note, "  <- the same walk on both legs, so this is the noise");
        else if (passes > 1)
            snprintf(note, sizeof note, "  <- re-measured over %d passes", passes);
        printf("%-5d %11.1f %11.1f %8.2fx%s\n", T, best[0] * 1e3, best[1] * 1e3, g, note);
        fflush(stdout);
        if (g < floor_)
            exit(1);
    }
    qe_free_huge(wq, nq);
    qe_free_huge(w4, n4);
    qe_free_huge(w3, n3);
    qe_free_huge(x, (size_t)MAXT * (size_t)K_ * sizeof(float));
    qe_free_huge(y, (size_t)ROWS_ * sizeof(float) * MAXT);
}

/* --- section 4: the integer walk, one panel load feeding two output rows --- */

/* Section 3's question on the arithmetic the expert stage would rather run. An int8 dot
 * leaves i32 for f32 every 32 weights into a multiply by the two scales and an FMA, and the
 * panel it reads is a quarter of the bytes of the f32 rows above - so the walk has two things
 * to share between output rows instead of one, the exit and the load, which is why the tile
 * is built for this family and priced here rather than argued from section 3's ratios
 * (ideas.md 7.10 ends on it, 7.11 leaves it as the one thing the integer side still owns).
 *
 * The bound is the same exactness: each output row keeps the single-row form's seed, block
 * order and exit, so the two settings leave the same bits at every width and row count, and
 * what is left to print is the rate. `make fast` holds the same pair to each other over the
 * K, row count and group width a bucket arrives at; this holds the pair to each other on the
 * shape a stage does, and then times it. */
typedef struct {
    qe_type t;
    const char *w;
    const int8_t *x8;
    const float *sp;
    const int32_t *pp;
    float *y;
    long rows, K;
    size_t row_bytes;
    long ldd_blocks;
    int T;
} i8_job;

static void i8_walk_task(void *v, int tid, int nt) {
    i8_job *j = v;
    long lo, hi;
    qe_slice(j->rows, tid, nt, &lo, &hi);
    if (lo >= hi)
        return;
    float *ys[MAXT];
    for (int t = 0; t < j->T; t++)
        ys[t] = j->y + (size_t)t * (size_t)j->rows + lo;
    /* The panel is the whole token group for every thread: what a walk of a slice of the
     * output rows shares is the activations, not a private copy of them. */
    qe_i8panel a = { (int8_t *)j->x8, (float *)j->sp, (int32_t *)j->pp, j->K, j->K / 32 };
    if (qe_gemv_i8_t(j->t, j->w + (size_t)lo * j->row_bytes, &a, j->T, ys, hi - lo, j->K,
                     j->ldd_blocks) != 0) {
        fprintf(stderr, "no integer kernel for this type\n");
        exit(1);
    }
}

/* The same slice of output rows walked out of a copy of the matrix that carries one scale per
 * row instead of one per 32 weights (`qe_gemv_q8_0_flat_i8`). The copy is the thing being
 * priced, so this file builds it: the bytes are the Q8_0 blocks' `qs` de-interleaved into a
 * flat row, which is the same int8 matrix the walk above walks, and what the flat model changes
 * is only where the scales are applied. `ws` is slice-relative, as the kernel wants it. */
typedef struct {
    const char *w;
    const float *ws;
    const int8_t *x8;
    const float *sp;
    float *y;
    long rows, K;
    int T, tile_r;
} flat_job;

static void flat_walk_task(void *v, int tid, int nt) {
    flat_job *j = v;
    long lo, hi;
    qe_slice(j->rows, tid, nt, &lo, &hi);
    if (lo >= hi)
        return;
    float *ys[MAXT];
    for (int t = 0; t < j->T; t++)
        ys[t] = j->y + (size_t)t * (size_t)j->rows + lo;
    const qe_i8panel a = { (int8_t *)j->x8, (float *)j->sp, NULL, j->K, j->K / 32 };
    if (qe_gemv_q8_0_flat_i8(j->w + (size_t)lo * (size_t)j->K, j->ws + lo, &a, j->T, ys, hi - lo,
                             j->K, j->K, j->tile_r) != 0) {
        fprintf(stderr, "no flat form for %d tokens at %d rows per tile\n", j->T, j->tile_r);
        exit(1);
    }
}

static void check_tile(qe_type t, const char *label, const i8_job *base, float *buf, long rows,
                       int T) {
    i8_job a = *base, b = *base;
    a.rows = b.rows = rows;
    a.T = b.T = T;
    a.y = buf;
    b.y = buf + (size_t)MAXT * (size_t)rows;
    qe_i8_set_rows(1);
    qe_pool_run(i8_walk_task, &a, qe_pool_size());
    qe_i8_set_rows(2);
    qe_pool_run(i8_walk_task, &b, qe_pool_size());
    long moved = 0;
    for (long i = 0; i < (long)((size_t)T * rows); i++)
        if (b.y[i] != a.y[i])
            moved++;
    printf("  %-8s tile over %ld rows x %d tokens: %ld of %ld outputs moved  %s\n", label, rows,
           T, moved, (long)((size_t)T * rows), moved ? "FAIL" : "ok");
    fflush(stdout);
    if (moved)
        exit(1);
}

static void check_ssd(qe_type t, const char *label, const i8_job *base, float *buf, long rows,
                      int T) {
    /* The same walk twice, once with `vpdpbusd` and the bias its unsigned operand carries,
     * once with `vpdpbssd`, which needs neither that xor nor the seed that pays the bias back.
     *
     * The two are the same value in exact arithmetic, but not bit-for-bit: the seed is inside
     * the float accumulator, so the unsigned walk sums a lane that carries as much bias as
     * signal and cancels it only in the horizontal sum at the end of a row, while the signed
     * walk sums the small numbers. Same products, same block order, different rounding - so
     * this leg is a bound rather than an equality, and the bound is the difference the f32
     * chain can account for, not the zero the row-tile gate above holds. */
    i8_job a = *base, b = *base;
    a.rows = b.rows = rows;
    a.T = b.T = T;
    a.y = buf;
    b.y = buf + (size_t)MAXT * (size_t)rows;
    qe_i8_set_rows(1);
    qe_i8_set_ssd(0);
    qe_pool_run(i8_walk_task, &a, qe_pool_size());
    qe_i8_set_ssd(1);
    qe_pool_run(i8_walk_task, &b, qe_pool_size());
    qe_i8_unset_ssd();
    double worst = 0, big = 0;
    for (long i = 0; i < (long)((size_t)T * rows); i++) {
        const double d = fabs((double)b.y[i] - (double)a.y[i]);
        if (d > worst) worst = d;
        if (fabs((double)a.y[i]) > big) big = fabs((double)a.y[i]);
    }
    const int bad = !(big > 0 && worst <= 1e-5 * big);
    printf("  %-8s signed dot over %ld rows x %d tokens: worst diff %.3g of a largest |y| of"
           " %.3g  %s\n",
           label, rows, T, worst, big, bad ? "FAIL" : "ok");
    fflush(stdout);
    if (bad)
        exit(1);
}

static void check_rowscale(const char *label, const i8_job *pb, const i8_job *rs, float *buf,
                           long rows, int T) {
    /* One walk on the per-block panel and one on a panel with a single scale for the whole
     * row, which is the panel `qe_quant_q8_0_row_rs` writes and the reason the codebook walk
     * gets to drop two of the four operands its exit pays per 32 weights.
     *
     * Not bit-identical, and no such thing is claimed: the row's scale is the row's largest
     * value, so every block quieter than the loudest one is quantised coarser than it was.
     * What this leg holds is therefore a relative bound - the difference has to stay the size
     * of a coarser quantisation and not of a wrong number - together with every output being
     * finite. The quality price itself is what `make score --ingest` is for. */
    i8_job a = *pb, b = *rs;
    a.rows = b.rows = rows;
    a.T = b.T = T;
    a.y = buf;
    b.y = buf + (size_t)MAXT * (size_t)rows;
    qe_i8_set_rowscale(0);
    qe_pool_run(i8_walk_task, &a, qe_pool_size());
    qe_i8_set_rowscale(1);
    qe_pool_run(i8_walk_task, &b, qe_pool_size());
    qe_i8_set_rowscale(-1);
    double worst = 0, big = 0;
    long nonfinite = 0;
    for (long i = 0; i < (long)((size_t)T * rows); i++) {
        if (!(b.y[i] == b.y[i]) || !(a.y[i] == a.y[i]))
            nonfinite++;
        const double d = fabs((double)b.y[i] - (double)a.y[i]);
        if (d > worst) worst = d;
        if (fabs((double)a.y[i]) > big) big = fabs((double)a.y[i]);
    }
    const int bad = !(big > 0 && worst <= 5e-2 * big) || nonfinite;
    printf("  %-8s row scale over %ld rows x %d tokens: worst diff %.3g of a largest |y| of"
           " %.3g (%.2f%%), %ld non-finite  %s\n",
           label, rows, T, worst, big, big > 0 ? 100.0 * worst / big : 0.0, nonfinite,
           bad ? "FAIL" : "ok");
    fflush(stdout);
    if (bad)
        exit(1);
}

static void check_hoist(const char *label, const i8_job *base, float *buf, long rows, int T) {
    /* One row-scaled panel and the two exits a walk can read it with: the row's scale
     * multiplied per 32 weights inside the block loop, or once after it. The products a row
     * sums and the order they come in are the same, so what separates the two is rounding,
     * and this holds them to the bound `check_ssd` holds a change of instruction to.
     *
     * The codebook walks print zero: their row-scaled form has taken the row's scale outside
     * the loop since the row scale landed, and the knob below is not what selects it. The
     * Q8_0 walk is the one that had not, because a row-scaled panel is also a valid per-block
     * panel and it read it that way. */
    i8_job a = *base, b = *base;
    a.rows = b.rows = rows;
    a.T = b.T = T;
    a.y = buf;
    b.y = buf + (size_t)MAXT * (size_t)rows;
    qe_i8_set_rowscale(1);
    qe_i8_set_hoist(0);
    qe_pool_run(i8_walk_task, &a, qe_pool_size());
    qe_i8_set_hoist(1);
    qe_pool_run(i8_walk_task, &b, qe_pool_size());
    qe_i8_set_hoist(-1);
    qe_i8_set_rowscale(-1);
    double worst = 0, big = 0;
    for (long i = 0; i < (long)((size_t)T * rows); i++) {
        const double d = fabs((double)b.y[i] - (double)a.y[i]);
        if (d > worst) worst = d;
        if (fabs((double)a.y[i]) > big) big = fabs((double)a.y[i]);
    }
    const int bad = !(big > 0 && worst <= 1e-5 * big);
    printf("  %-8s hoisted row scale over %ld rows x %d tokens: worst diff %.3g of a largest"
           " |y| of %.3g  %s\n",
           label, rows, T, worst, big, bad ? "FAIL" : "ok");
    fflush(stdout);
    if (bad)
        exit(1);
}

static void section_tile(int nthr) {
    const size_t i4_rb = (size_t)(K_ / 32) * sizeof(qe_block_iq4_nl);
    const size_t i3_rb = (size_t)(K_ / 256) * sizeof(qe_block_iq3_s);
    const size_t q8_rb = (size_t)(K_ / 32) * sizeof(qe_block_q8_0);
    const long DROWS_ = 100000; /* 260 MiB of Q8_0: the dense walk's own tensor */
    const size_t n4 = (size_t)ROWS_ * i4_rb, n3 = (size_t)ROWS_ * i3_rb;
    const size_t nq = (size_t)DROWS_ * q8_rb;
    const long pn = (size_t)MAXT * (size_t)K_;
    char *w4 = qe_alloc_huge(n4);
    char *w3 = qe_alloc_huge(n3);
    char *wq = qe_alloc_huge(nq);
    float *x = qe_alloc_huge((size_t)MAXT * (size_t)K_ * sizeof(float));
    int8_t *x8 = qe_alloc_huge(pn);
    float *sp = qe_alloc_huge((size_t)MAXT * (size_t)(K_ / 32) * sizeof(float));
    int32_t *pp = qe_alloc_huge((size_t)MAXT * (size_t)(K_ / 32) * sizeof(int32_t));
    float *y = qe_alloc_huge((size_t)ROWS_ * sizeof(float) * MAXT);
    if (!w4 || !w3 || !wq || !x || !x8 || !sp || !pp || !y) {
        fprintf(stderr, "alloc\n");
        exit(1);
    }
    fill(w4, n4, sizeof(qe_block_iq4_nl));
    fill(w3, n3, sizeof(qe_block_iq3_s));
    fill(wq, nq, sizeof(qe_block_q8_0));
    fill_x(x, (size_t)MAXT * (size_t)K_);
    /* The same activations section 3 walks as floats, recoded once - which is the panel the
     * grouped expert pass hands these kernels (src/moe.c), built by the same call. */
    qe_quant_q8_0_rows(x, K_, x8, sp, pp, MAXT, K_);
    memset(y, 0, (size_t)ROWS_ * sizeof(float) * MAXT);

    /* Read before any leg flips the knob: the legs set it, and a gate that reads it later is
     * reading what the last leg asked for. */
    const int ships = qe_i8_rows(QE_I8_ROWS_DEFAULT) >= 2;

    printf("== the same walk on a recoded panel, two output rows to one load: K=%d, %d rows "
           "(%.1f IQ4_NL, %.1f IQ3_S), %ld rows of Q8_0 (%.2f) ==\n", K_, ROWS_,
           (double)n4 / 1073741824.0, (double)n3 / 1073741824.0, DROWS_,
           (double)nq / 1073741824.0);
    {
        const long crows = 4103; /* odd: the last row has no partner */
        float *buf = qe_alloc_huge((size_t)2 * MAXT * (size_t)crows * sizeof(float));
        if (!buf) {
            fprintf(stderr, "alloc\n");
            exit(1);
        }
        const i8_job j4 = { QE_IQ4_NL, w4, x8, sp, pp, y, ROWS_, K_, i4_rb, K_ / 32, 8 };
        const i8_job j3 = { QE_IQ3_S, w3, x8, sp, pp, y, ROWS_, K_, i3_rb, K_ / 256, 8 };
        const i8_job jq = { QE_Q8_0, wq, x8, sp, pp, y, DROWS_, K_, q8_rb, K_ / 32, 8 };
        /* Every width a form exists for plus the widths a bucket and a chunk arrive at: the
         * tile takes the eights of a group and hands the rest to the single-row forms, so
         * 11 and 13 are where a borrowed partner would show up. */
        static const int TT[] = { 8, 9, 11, 13, 16, 24, 32, 64, 512 };
        for (int ti = 0; ti < (int)(sizeof TT / sizeof *TT); ti++) {
            const int T = TT[ti];
            check_tile(QE_IQ4_NL, "IQ4_NL", &j4, buf, crows, T);
            check_tile(QE_IQ3_S, "IQ3_S", &j3, buf, crows, T);
            check_tile(QE_Q8_0, "Q8_0", &jq, buf, crows, T);
#if defined(__AVXVNNIINT8__)
            check_ssd(QE_IQ4_NL, "IQ4_NL", &j4, buf, crows, T);
            check_ssd(QE_IQ3_S, "IQ3_S", &j3, buf, crows, T);
            check_ssd(QE_Q8_0, "Q8_0", &jq, buf, crows, T);
#endif
        }
        qe_free_huge(buf, (size_t)2 * MAXT * (size_t)crows * sizeof(float));
    }

    static const int TW[] = { 4, 8, 11, 13, 16, 32 };
    printf("%-5s %11s %11s %9s %11s %11s %9s\n", "T", "IQ4 1-row", "IQ4 2-row", "IQ4 gain",
           "IQ3 1-row", "IQ3 2-row", "IQ3 gain");
    for (int wi = 0; wi < (int)(sizeof TW / sizeof *TW); wi++) {
        const int T = TW[wi];
        double best[2][2] = { { 1e30, 1e30 }, { 1e30, 1e30 } };
        /* Below eight there is no tile at all, so those legs are the same walk and the row
         * is this run's noise floor rather than a result (section 3 says the same). Above it
         * the row is a gate only where the engine runs the tile: an instrument that prices a
         * form the tree does not use should print the number, not fail a build over it. */
        const double floor_ = (T >= 8 && ships) ? 1.0 : 0.0;
        int passes = 0;
        for (;;) {
            for (int rep = 0; rep < 3; rep++)
                for (int leg = 0; leg < 2; leg++)
                    for (int which = 0; which < 2; which++) {
                        qe_i8_set_rows(leg ? 2 : 1);
                        i8_job j = { which == 0 ? QE_IQ4_NL : QE_IQ3_S,
                                     which == 0 ? w4 : w3, x8, sp, pp, y, ROWS_, K_,
                                     which == 0 ? i4_rb : i3_rb,
                                     which == 0 ? K_ / 32 : K_ / 256, T };
                        double t0 = now();
                        qe_pool_run(i8_walk_task, &j, nthr);
                        double dt = now() - t0;
                        if (dt < best[leg][which])
                            best[leg][which] = dt;
                        if (!(y[7] == y[7]) || y[7] == 0.0f) {
                            fprintf(stderr, "the integer walk wrote nothing useful at T=%d\n", T);
                            exit(1);
                        }
                    }
            qe_i8_set_rows(2);
            if (++passes >= MEASURE_PASSES || (best[0][0] / best[1][0] >= floor_ &&
                                               best[0][1] / best[1][1] >= floor_))
                break;
        }
        const double g4 = best[0][0] / best[1][0], g3 = best[0][1] / best[1][1];
        char note[112] = "";
        if (g4 < floor_ || g3 < floor_)
            snprintf(note, sizeof note, "  <- the tile is slower (load average %.2f)",
                     load_average_1m());
        else if (T < 8 && (g4 < 0.95 || g4 > 1.05 || g3 < 0.95 || g3 > 1.05))
            snprintf(note, sizeof note, "  <- the same walk on both legs, so this is the noise");
        else if (passes > 1)
            snprintf(note, sizeof note, "  <- re-measured over %d passes", passes);
        printf("%-5d %11.1f %11.1f %8.2fx %11.1f %11.1f %8.2fx%s\n", T, best[0][0] * 1e3,
               best[1][0] * 1e3, g4, best[0][1] * 1e3, best[1][1] * 1e3, g3, note);
        fflush(stdout);
        if (g4 < floor_ || g3 < floor_)
            exit(1);
    }

    /* The same pair where the weights survive the L3, because the two columns of ideas.md
     * 7.10 disagree about everything else in this family: a 640-row expert matrix is 0.9 MiB,
     * sixteen copies of it are 14 MiB, and a walk of those is the rate this shape has when
     * nothing is streaming. It is not the rate a 512-token chunk routes at - the routed
     * experts of a layer are 21 MiB and do not survive - but a tile that lost on the streaming
     * shape and wins here lost to bytes and not to the register file, and the two answers send
     * a reader to different places. */
    {
        const long ER = 640, EC = 16;
        const size_t en4 = (size_t)ER * EC * i4_rb, en3 = (size_t)ER * EC * i3_rb;
        char *e4 = qe_alloc_huge(en4), *e3 = qe_alloc_huge(en3);
        if (!e4 || !e3) {
            fprintf(stderr, "alloc\n");
            exit(1);
        }
        fill(e4, en4, sizeof(qe_block_iq4_nl));
        fill(e3, en3, sizeof(qe_block_iq3_s));
        printf("\n%-5s %11s %11s %9s %11s %11s %9s  (%ld rows x %ld copies resident:"
               " %.1f and %.1f MiB)\n",
               "T", "IQ4 1-row", "IQ4 2-row", "IQ4 gain", "IQ3 1-row", "IQ3 2-row", "IQ3 gain",
               ER, EC, (double)en4 / 1048576.0, (double)en3 / 1048576.0);
        static const int RT[] = { 8, 16, 32 };
        for (int wi = 0; wi < (int)(sizeof RT / sizeof *RT); wi++) {
            const int T = RT[wi];
            double best[2][2] = { { 1e30, 1e30 }, { 1e30, 1e30 } };
            for (int rep = 0; rep < 7; rep++)
                for (int leg = 0; leg < 2; leg++)
                    for (int which = 0; which < 2; which++) {
                        qe_i8_set_rows(leg ? 2 : 1);
                        i8_job j = { which == 0 ? QE_IQ4_NL : QE_IQ3_S,
                                     which == 0 ? e4 : e3, x8, sp, pp, y, ER * EC, K_,
                                     which == 0 ? i4_rb : i3_rb,
                                     which == 0 ? K_ / 32 : K_ / 256, T };
                        double t0 = now();
                        qe_pool_run(i8_walk_task, &j, nthr);
                        double dt = now() - t0;
                        if (dt < best[leg][which])
                            best[leg][which] = dt;
                    }
            qe_i8_set_rows(2);
            printf("%-5d %11.1f %11.1f %8.2fx %11.1f %11.1f %8.2fx\n", T, best[0][0] * 1e3,
                   best[1][0] * 1e3, best[0][0] / best[1][0], best[0][1] * 1e3,
                   best[1][1] * 1e3, best[0][1] / best[1][1]);
            fflush(stdout);
        }
        qe_free_huge(e4, en4);
        qe_free_huge(e3, en3);
    }

    /* Q8_0 has had this tile since the integer walk was written, and its pairing is already
     * on for the dense projections (`QE_Q8_0_MULTI2R_FORM` in `kern.c`), so this table is the
     * control: the format with no codebook decode in the loop is where a shared exit and a
     * halved panel load show up without a decode paying for them as well. */
    static const int DT[] = { 8, 16, 32, 512 };
    printf("\n%-5s %11s %11s %9s  (%ld rows of Q8_0, the same tile with no codebook decode)\n",
           "T", "Q8_0 1-row", "Q8_0 2-row", "Q8_0 gain", DROWS_);
    for (int wi = 0; wi < (int)(sizeof DT / sizeof *DT); wi++) {
        const int T = DT[wi];
        double best[2] = { 1e30, 1e30 };
        int passes = 0;
        const int reps = T >= 256 ? 5 : 3;
        for (;;) {
            for (int rep = 0; rep < reps; rep++)
                for (int leg = 0; leg < 2; leg++) {
                    qe_i8_set_rows(leg ? 2 : 1);
                    i8_job j = { QE_Q8_0, wq, x8, sp, pp, y, DROWS_, K_, q8_rb, K_ / 32, T };
                    double t0 = now();
                    qe_pool_run(i8_walk_task, &j, nthr);
                    double dt = now() - t0;
                    if (dt < best[leg])
                        best[leg] = dt;
                    if (!(y[7] == y[7]) || y[7] == 0.0f) {
                        fprintf(stderr, "the dense integer walk wrote nothing at T=%d\n", T);
                        exit(1);
                    }
                }
            qe_i8_set_rows(2);
            if (++passes >= MEASURE_PASSES || best[0] / best[1] >= (ships ? 1.0 : 0.0))
                break;
        }
        const double g = best[0] / best[1];
        printf("%-5d %11.1f %11.1f %8.2fx%s\n", T, best[0] * 1e3, best[1] * 1e3, g,
               passes > 1 ? "  <- re-measured over the minimum of three reps" : "");
        fflush(stdout);
    }
    printf("ms is a whole walk of the tensor at one Q8_0 recode of the activations, so the two\n"
           "columns of a row differ by how much of a walk a second output row can share. The\n"
           "gain is one-row over two-row, and 1.00x means the tile bought nothing.\n");

#if defined(__AVXVNNIINT8__)
    /* The two dots against each other on the same shapes: same walk, same forms, same panel,
     * and the signed one carries neither the xor that biases the unsigned operand nor the seed
     * that pays the bias back into the accumulator. That is two of the about seven operands an
     * output row's exit costs, paid per 32 weights per token, so the gain should show up at
     * every width and on every format - and on the codebook formats most, because there the
     * decode is shorter than the exit. Legs alternate inside one pass and the number printed
     * is a ratio of two legs of one pass, which is the only comparison this host survives. */
    {
        printf("\n%-5s %10s %10s %8s %10s %10s %8s %10s %10s %8s  (%s)\n", "T", "IQ4 unsigned",
               "IQ4 signed", "gain", "IQ3 unsigned", "IQ3 signed", "gain", "Q8_0 unsigned",
               "Q8_0 signed", "gain",
               "vpdpbusd against vpdpbssd, one row per walk, same panel");
        static const int SDT[] = { 8, 11, 16, 32, 64, 512 };
        for (int wi = 0; wi < (int)(sizeof SDT / sizeof *SDT); wi++) {
            const int T = SDT[wi];
            double best[2][3] = { { 1e30, 1e30, 1e30 }, { 1e30, 1e30, 1e30 } };
            qe_i8_set_rows(1);
            for (int rep = 0; rep < 3; rep++)
                for (int leg = 0; leg < 2; leg++)
                    for (int which = 0; which < 3; which++) {
                        qe_i8_set_ssd(leg);
                        i8_job j = { which == 0 ? QE_IQ4_NL : which == 1 ? QE_IQ3_S : QE_Q8_0,
                                     which == 0 ? w4 : which == 1 ? w3 : wq, x8, sp, pp, y,
                                     which == 2 ? DROWS_ : ROWS_, K_,
                                     which == 0 ? i4_rb : which == 1 ? i3_rb : q8_rb,
                                     which == 0 ? K_ / 32 : which == 1 ? K_ / 256 : K_ / 32, T };
                        double t0 = now();
                        qe_pool_run(i8_walk_task, &j, nthr);
                        double dt = now() - t0;
                        if (dt < best[leg][which])
                            best[leg][which] = dt;
                    }
            qe_i8_unset_ssd();
            /* The claim is about the exit an output row pays, so it is testable on the two
             * formats whose walk is ALU-bound at these widths, and their default is the signed
             * dot. Q8_0 is the control here, on a per-block panel - `main` pins the row scale
             * off for this file's fills - and on that panel the unsigned dot is ahead at every
             * width, so that column gates only the direction that would be a surprise. What a
             * Q8_0 walk actually runs is a row-scaled panel, where the signed dot is the faster
             * one at every width, and that pair is gated by the hoist table at the end of this
             * section. */
            int bad = 0;
            /* The signed codebook walk executes a strict subset of the operands the unsigned one
             * does - no xor to bias the operand, no seed to pay that bias back per 32 weights -
             * so this row can only ever gate that the two are the same walk, and a loss is
             * scatter rather than a mechanism. The run that made it a failure read IQ3_S 919.4 ms
             * unsigned against 925.7 ms signed (0.99x) at T=512 over 400000 rows, in a row where
             * IQ4_NL measured the other way at 784.0 against 713.6 (1.10x). Held to 1.00x while
             * the effect costs several percent of a leg that is short enough for the ratio to
             * mean it, and to 0.98x - about three times the scatter of the row just quoted, and
             * two seconds of leg - at the width where neither claim is resolvable. */
            if (best[0][0] / best[1][0] < (T >= 512 ? 0.98 : 1.0) ||
                best[0][1] / best[1][1] < (T >= 512 ? 0.98 : 1.0))
                bad = 1;
            /* The other direction, and the measurements are in the printed row: on a per-block
             * panel the unsigned dot is ahead of the signed one by 3-7 % at eight to thirty-two
             * token rows, where the walk reads its tensor at the machine's byte ceiling and
             * cannot show an instruction it stopped executing, and by 40 % at five hundred and
             * twelve (169.2 ms against 237.0 ms), where it is not byte-bound and paying the
             * unsigned bias back per 32 weights costs what the signed dot's saved operand is
             * worth. So this bound fires only if the unsigned leg becomes the slower one - held
             * to 10 % above sixty-four rows and to 5 % below it, because the byte-bound claim it
             * was gating was never a claim about the wide end. That the signed dot loses here by
             * four in ten while the engine's default is signed is not a contradiction: the
             * default reads a row-scaled panel, and the hoist legs at the end of this section
             * are what price that pair. */
            if (best[0][2] / best[1][2] > (T >= 64 ? 1.10 : 1.05)) bad = 2;
            printf("%-5d %10.1f %10.1f %7.2fx %10.1f %10.1f %7.2fx %10.1f %10.1f %7.2fx%s\n", T,
                   best[0][0] * 1e3, best[1][0] * 1e3, best[0][0] / best[1][0], best[0][1] * 1e3,
                   best[1][1] * 1e3, best[0][1] / best[1][1], best[0][2] * 1e3, best[1][2] * 1e3,
                   best[0][2] / best[1][2],
                   bad == 1 ? "  <- a signed codebook leg is slower"
                   : bad == 2 ? "  <- the unsigned Q8_0 leg, the exit a Q8_0 walk uses, is lost"
                              : "");
            fflush(stdout);
            if (bad) {
                fprintf(stderr, "the signed dot lost on the %s legs at T=%d\n",
                        bad == 1 ? "codebook" : "unsigned Q8_0", T);
                exit(1);
            }
        }
    }

    /* The signed dot also changes the answer to the older question, which is whether a
     * second output row sharing one int8 activation load pays at all. It is the same
     * question because the part a pair shares is the exit, and the exit is what just got
     * two operands shorter: §7.12 measured 0.95-1.04x for the pair while the dot still
     * carried the xor and the seed, and a pair of rows against one load is the one
     * structure that takes the loads per multiply below one. Same alternation, same
     * panel, one row of each; the column is one-row over two-row, so above 1.00x is the
     * pair winning. Reported, not gated: which way this goes is a measurement, and the
     * register file is the same 16 accumulators either way. */
    {
        printf("\n%-5s %11s %11s %8s %11s %11s %8s %11s %11s %8s  (%s)\n", "T", "IQ4 one row",
               "IQ4 two rows", "gain", "IQ3 one row", "IQ3 two rows", "gain", "Q8_0 one row",
               "Q8_0 two rows", "gain",
               "the exit each format uses, one output row against two per activation load");
        static const int RT[] = { 8, 16, 32, 512 };
        for (int wi = 0; wi < (int)(sizeof RT / sizeof *RT); wi++) {
            const int T = RT[wi];
            double best[2][3] = { { 1e30, 1e30, 1e30 }, { 1e30, 1e30, 1e30 } };
            qe_i8_unset_ssd();
            for (int rep = 0; rep < 3; rep++)
                for (int leg = 0; leg < 2; leg++)
                    for (int which = 0; which < 3; which++) {
                        qe_i8_set_rows(leg + 1);
                        i8_job j = { which == 0 ? QE_IQ4_NL : which == 1 ? QE_IQ3_S : QE_Q8_0,
                                     which == 0 ? w4 : which == 1 ? w3 : wq, x8, sp, pp, y,
                                     which == 2 ? DROWS_ : ROWS_, K_,
                                     which == 0 ? i4_rb : which == 1 ? i3_rb : q8_rb,
                                     which == 0 ? K_ / 32 : which == 1 ? K_ / 256 : K_ / 32, T };
                        double t0 = now();
                        qe_pool_run(i8_walk_task, &j, nthr);
                        double dt = now() - t0;
                        if (dt < best[leg][which])
                            best[leg][which] = dt;
                    }
            qe_i8_set_rows(QE_I8_ROWS_DEFAULT);
            printf("%-5d %11.1f %11.1f %7.2fx %11.1f %11.1f %7.2fx %11.1f %11.1f %7.2fx\n", T,
                   best[0][0] * 1e3, best[1][0] * 1e3, best[0][0] / best[1][0], best[0][1] * 1e3,
                   best[1][1] * 1e3, best[0][1] / best[1][1], best[0][2] * 1e3, best[1][2] * 1e3,
                   best[0][2] / best[1][2]);
            fflush(stdout);
        }
    }
#endif

#if defined(__AVXVNNIINT8__)
    /* One activation scale for a whole row instead of one per 32 values. The prize is the
     * exit: with equal scales the product of the two scales is no longer a function of the
     * block, so the accumulator runs in weight-scale space and the row's scale multiplies it
     * once after the block loop. Two of the four operands an output row owes per 32 weights
     * go with it, which is the difference between ideas.md 7.10's 1.36x and its 1.67x - and
     * 7.13 says the routed walk is the one that has anything left to save, so this is the last
     * arithmetic the integer side owns. What it costs is range, priced above by the difference
     * it leaves and by `make score --ingest`.
     *
     * Q8_0 is the control: its walk takes the row-scaled panel as the per-block panel it
     * already is, so both of its legs run the same kernel and that column is this run's noise
     * floor. Reported and not gated - the knob ships off, and a rate that does not pay for the
     * range is recorded as a rate that does not pay for the range. */
    {
        const long pnr = (size_t)MAXT * (size_t)K_;
        int8_t *x8r = qe_alloc_huge(pnr);
        float *spr = qe_alloc_huge((size_t)MAXT * (size_t)(K_ / 32) * sizeof(float));
        int32_t *ppr = qe_alloc_huge((size_t)MAXT * (size_t)(K_ / 32) * sizeof(int32_t));
        if (!x8r || !spr || !ppr) {
            fprintf(stderr, "alloc\n");
            exit(1);
        }
        for (int t = 0; t < MAXT; t++)
            qe_quant_q8_0_row_rs(x + (size_t)t * (size_t)K_, x8r + (size_t)t * (size_t)K_,
                                 spr + (size_t)t * (size_t)(K_ / 32),
                                 ppr + (size_t)t * (size_t)(K_ / 32), K_);
        const i8_job jb[3] = { { QE_IQ4_NL, w4, x8, sp, pp, y, ROWS_, K_, i4_rb, K_ / 32, 8 },
                               { QE_IQ3_S, w3, x8, sp, pp, y, ROWS_, K_, i3_rb, K_ / 256, 8 },
                               { QE_Q8_0, wq, x8, sp, pp, y, DROWS_, K_, q8_rb, K_ / 32, 8 } };
        const i8_job jr[3] = { { QE_IQ4_NL, w4, x8r, spr, ppr, y, ROWS_, K_, i4_rb, K_ / 32, 8 },
                               { QE_IQ3_S, w3, x8r, spr, ppr, y, ROWS_, K_, i3_rb, K_ / 256, 8 },
                               { QE_Q8_0, wq, x8r, spr, ppr, y, DROWS_, K_, q8_rb, K_ / 32, 8 } };
        {
            const long crows = 4103;
            float *buf = qe_alloc_huge((size_t)2 * MAXT * (size_t)crows * sizeof(float));
            if (!buf) {
                fprintf(stderr, "alloc\n");
                exit(1);
            }
            static const int CT[] = { 8, 11, 16, 32, 512 };
            printf("\n== one activation scale per row: one walk, two panels, %ld rows ==\n",
                   crows);
            for (int ti = 0; ti < (int)(sizeof CT / sizeof *CT); ti++) {
                const int T = CT[ti];
                check_rowscale("IQ4_NL", &jb[0], &jr[0], buf, crows, T);
                check_rowscale("IQ3_S", &jb[1], &jr[1], buf, crows, T);
                check_rowscale("Q8_0", &jb[2], &jr[2], buf, crows, T);
            }
            printf("\n== the same row-scaled panel, two exits: %ld rows ==\n", crows);
            for (int ti = 0; ti < (int)(sizeof CT / sizeof *CT); ti++) {
                const int T = CT[ti];
                check_hoist("IQ4_NL", &jr[0], buf, crows, T);
                check_hoist("IQ3_S", &jr[1], buf, crows, T);
                check_hoist("Q8_0", &jr[2], buf, crows, T);
            }
            qe_free_huge(buf, (size_t)2 * MAXT * (size_t)crows * sizeof(float));
        }
        printf("\n%-5s %13s %11s %8s %13s %11s %8s %14s %11s %8s  (%s)\n", "T", "IQ4 per block",
               "IQ4 row", "gain", "IQ3 per block", "IQ3 row", "gain", "Q8_0 per block", "Q8_0 row",
               "gain", "one scale per row against one per 32, one output row per walk");
        static const int BST[] = { 8, 16, 32, 64, 512 };
        for (int wi = 0; wi < (int)(sizeof BST / sizeof *BST); wi++) {
            const int T = BST[wi];
            double best[2][3] = { { 1e30, 1e30, 1e30 }, { 1e30, 1e30, 1e30 } };
            qe_i8_set_rows(1);
            for (int rep = 0; rep < 3; rep++)
                for (int leg = 0; leg < 2; leg++)
                    for (int which = 0; which < 3; which++) {
                        qe_i8_set_rowscale(leg);
                        i8_job jj = leg ? jr[which] : jb[which];
                        jj.T = T;
                        double t0 = now();
                        qe_pool_run(i8_walk_task, &jj, nthr);
                        double dt = now() - t0;
                        if (dt < best[leg][which])
                            best[leg][which] = dt;
                        if (!(y[7] == y[7]) || y[7] == 0.0f) {
                            fprintf(stderr, "the row-scaled walk wrote nothing useful at T=%d\n",
                                    T);
                            exit(1);
                        }
                    }
            qe_i8_set_rowscale(-1);
            printf("%-5d %13.1f %11.1f %7.2fx %13.1f %11.1f %7.2fx %14.1f %11.1f %7.2fx"
                   "  (load average %.2f)\n",
                   T, best[0][0] * 1e3, best[1][0] * 1e3, best[0][0] / best[1][0],
                   best[0][1] * 1e3, best[1][1] * 1e3, best[0][1] / best[1][1], best[0][2] * 1e3,
                   best[1][2] * 1e3, best[0][2] / best[1][2], load_average_1m());
            fflush(stdout);
        }

        /* The exit the row scale was meant to buy, priced on the walk that owes the seed
         * anyway.
         *
         * The Q8_0 walk takes the unsigned dot, and `vpdpbusd` takes that bias as the
         * accumulator it was handed, so a row-scaled panel leaves this walk exactly one
         * operand to remove. Removing it was declined for this format on two measurements
         * made at eight token rows, where the walk waits on bytes and has no exit left to
         * save; at the 512 a chunk ingests at the same walk moves 1.3 GB/s of tensor for 537
         * GMAC/s, which is a walk spending its cycles on the exit and not on the DRAM. Both
         * dots are therefore raced on one panel, and the gate is that the hoist is not the
         * slower exit at any width. */
        printf("\n%-5s %11s %11s %8s %11s %11s %8s %11s %11s  (%s)\n", "T", "i8 block",
               "i8 hoist", "gain", "ssd block", "ssd hoist", "gain", "GMAC/s block",
               "GMAC/s hoist", "one row-scaled panel, two exits, 100000 rows of Q8_0");
        static const int HT[] = { 8, 16, 64, 512 };
        for (int hi = 0; hi < (int)(sizeof HT / sizeof *HT); hi++) {
            const int T = HT[hi];
            double rate[2][2] = { { 1e30, 1e30 }, { 1e30, 1e30 } };
            const double floor_ = 0.98; /* a wash is not a win, but it is not a FAIL either */
            /* Read before any leg pins it: which of the four cells below this process
             * actually runs is what the last gate is about. */
            const int dot_ship = qe_i8_ssd_q8();
            qe_i8_set_rows(1);
            qe_i8_set_rowscale(1);
            int passes = 0;
            for (;;) {
                for (int rep = 0; rep < 3; rep++)
                    for (int dot = 0; dot < 2; dot++)
                        for (int leg = 0; leg < 2; leg++) {
                            qe_i8_set_ssd(dot);
                            qe_i8_set_hoist(leg);
                            i8_job jh = jr[2];
                            jh.T = T;
                            double t0 = now();
                            qe_pool_run(i8_walk_task, &jh, nthr);
                            double dt = now() - t0;
                            if (dt < rate[dot][leg]) rate[dot][leg] = dt;
                            if (!(y[7] == y[7]) || y[7] == 0.0f) {
                                fprintf(stderr, "the hoisted exit wrote nothing useful at T=%d\n",
                                        T);
                                exit(1);
                            }
                        }
                qe_i8_unset_ssd();
                qe_i8_set_hoist(-1);
                if (rate[0][0] / rate[0][1] >= floor_ && rate[1][0] / rate[1][1] >= floor_) {
                    double b4 = rate[0][0];
                    if (rate[0][1] < b4)
                        b4 = rate[0][1];
                    if (rate[1][0] < b4)
                        b4 = rate[1][0];
                    if (rate[1][1] < b4)
                        b4 = rate[1][1];
                    if (T != 512 || rate[dot_ship][1] <= 1.02 * b4)
                        break;
                }
                if (++passes >= 3 * MEASURE_PASSES) break;
            }
            const double gu = rate[0][0] / rate[0][1], gs = rate[1][0] / rate[1][1];
            const double macs = (double)DROWS_ * (double)K_ * (double)T;
            /* The pair this engine runs is the signed dot on a hoisted exit; at the width a
             * chunk is ingested at that pair has to be the fastest of the four, or the
             * default is the wrong default. */
            double best4 = rate[0][0];
            if (rate[0][1] < best4)
                best4 = rate[0][1];
            if (rate[1][0] < best4)
                best4 = rate[1][0];
            if (rate[1][1] < best4)
                best4 = rate[1][1];
            const double off = rate[dot_ship][1] / best4;
            const int behind = (T == 512) && off > 1.02;
            char note[112] = "";
            if (gu < floor_ || gs < floor_)
                snprintf(note, sizeof note, "  <- the hoist is the slower exit (load average %.2f)",
                         load_average_1m());
            else if (behind)
                snprintf(note, sizeof note, "  <- what this runs is %.2fx off the fastest cell",
                         off);
            else if (passes)
                snprintf(note, sizeof note, "  <- re-measured over %d passes", passes + 1);
            printf("%-5d %11.1f %11.1f %7.2fx %11.1f %11.1f %7.2fx %11.1f %11.1f%s\n", T,
                   rate[0][0] * 1e3, rate[0][1] * 1e3, gu, rate[1][0] * 1e3, rate[1][1] * 1e3, gs,
                   macs / rate[0][0] / 1e9, macs / rate[0][1] / 1e9, note);
            fflush(stdout);
            if (gu < floor_ || gs < floor_ || behind) {
                fprintf(stderr, "the hoisted row-scale exit lost at T=%d\n", T);
                exit(1);
            }
        }

        /* One scale for a whole output row instead of one for every 32 weights, which is the
         * exit removed rather than hoisted.
         *
         * The hoist above leaves four operations per token per 32 weights in the Q8_0 walk: a
         * load, the dot, the int32 convert, and the fold of the block's scale into the f32 sum.
         * Two of the four are there because the scale is the block's. Take a copy of the matrix
         * with one scale per row and the pair disappears - the accumulator runs in int32 the
         * whole row, becomes a float once, and pays the row's scale and the token's - and the
         * register the block scale held pays for a wider tile: three output rows sharing one
         * activation load is twelve accumulators, which is the register file exactly.
         *
         * The copy this leg walks is the same int8 matrix, de-interleaved out of the Q8_0
         * blocks, and the scales this file gives it are the mean of each row's block scales,
         * which for this tensor is every one of them: the fill makes them equal, so the flat
         * walk and the hoisted walk are computing the same number, and that is the bound the
         * first check below holds them to. What it costs in the engine - a requantised copy of
         * the dense matrices, and the resolution a row-wide scale gives up - is a checkpoint
         * question, and `make forward`'s quality legs are what answer it. */
        {
            const long frows = 4103; /* odd: the row tile's tail has no partner */
            const long RROWS_ = 2048; /* 5 MiB of flat rows: resident, unlike the 250 MiB above */
            char *wf = qe_alloc_huge((size_t)DROWS_ * (size_t)K_);
            float *sw = qe_alloc_huge((size_t)DROWS_ * sizeof(float));
            float *bf = qe_alloc_huge((size_t)2 * MAXT * (size_t)frows * sizeof(float));
            if (!wf || !sw || !bf) {
                fprintf(stderr, "alloc\n");
                exit(1);
            }
            for (long r = 0; r < DROWS_; r++) {
                const qe_block_q8_0 *b = (const qe_block_q8_0 *)(wq + (size_t)r * q8_rb);
                int8_t *dst = (int8_t *)(wf + (size_t)r * (size_t)K_);
                float mean = 0.0f;
                for (long g = 0; g < K_ / 32; g++) {
                    memcpy(dst + 32 * g, b[g].qs, 32);
                    mean += qe_half_to_f32(b[g].d);
                }
                sw[r] = mean / (float)(K_ / 32);
            }

            /* Exactness: the flat walk against the walk this engine ships, on one row-scaled
             * panel, at the widest tile and at a one-row tile, and at a row count the tiles do
             * not divide. */
            qe_i8_set_rowscale(1);
            qe_i8_set_hoist(1);
            qe_i8_set_ssd(1);
            qe_i8_set_rows(1);
            for (int which = 0; which < 3; which++) {
                const int tr = (int[]){ 3, 1, 5 }[which];
                i8_job a = jr[2];
                a.rows = frows;
                a.T = 8;
                a.y = bf;
                qe_pool_run(i8_walk_task, &a, nthr);
                flat_job b = { wf, sw, x8r, spr, bf + (size_t)MAXT * (size_t)frows, frows, K_, 8,
                               tr };
                qe_pool_run(flat_walk_task, &b, nthr);
                double worst = 0.0;
                float maxy = 0.0f;
                for (int t = 0; t < 8; t++)
                    for (long r = 0; r < frows; r++) {
                        const float u = bf[(size_t)t * (size_t)frows + r];
                        const float v = bf[(size_t)MAXT * (size_t)frows + (size_t)t * (size_t)frows +
                                           r];
                        const double d = fabs((double)u - (double)v);
                        if (d > worst) worst = d;
                        if (fabsf(u) > maxy) maxy = fabsf(u);
                    }
                printf("  flat r%d over %ld rows x 8 tokens: worst diff %.6g of %.4g  %s\n", tr,
                       frows, worst, (double)maxy,
                       worst <= 1e-5 * (double)maxy ? "ok" : "FAIL");
                fflush(stdout);
                if (worst > 1e-5 * (double)maxy) exit(1);
            }

            printf("\n%-5s %11s %11s %7s %11s %7s %11s %7s %11s %7s %11s %7s  (%s)\n", "T",
                   "hoist+ssd", "flat r1", "gain", "flat r2", "gain", "flat r3", "gain", "flat r4",
                   "gain", "flat r5", "gain",
                   "one row-scaled panel, one scale per row, 100000 rows of Q8_0");
            static const int FT[] = { 8, 16, 64, 512 };
            for (int fi = 0; fi < (int)(sizeof FT / sizeof *FT); fi++) {
                const int T = FT[fi];
                double rate[6]; /* [0] is the pair this engine runs, [1..5] the row tiles */
                for (int i = 0; i < 6; i++)
                    rate[i] = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    i8_job a = jr[2];
                    a.T = T;
                    double t0 = now();
                    qe_pool_run(i8_walk_task, &a, nthr);
                    double dt = now() - t0;
                    if (dt < rate[0]) rate[0] = dt;
                    for (int tr = 1; tr <= 5; tr++) {
                        flat_job b = { wf, sw, x8r, spr, y, DROWS_, K_, T, tr };
                        t0 = now();
                        qe_pool_run(flat_walk_task, &b, nthr);
                        dt = now() - t0;
                        if (dt < rate[tr]) rate[tr] = dt;
                        if (!(y[7] == y[7]) || y[7] == 0.0f) {
                            fprintf(stderr, "the flat walk wrote nothing useful at T=%d\n", T);
                            exit(1);
                        }
                    }
                }
                double best = rate[0];
                int best_r = 0;
                for (int tr = 1; tr <= 5; tr++)
                    if (rate[tr] < best) {
                        best = rate[tr];
                        best_r = tr;
                    }
                const double macs = (double)DROWS_ * (double)K_ * (double)T;
                char note[112] = "";
                if (!best_r)
                    snprintf(note, sizeof note, "  <- the exit is not the wall (load average %.2f)",
                             load_average_1m());
                else
                    snprintf(note, sizeof note, "  <- fastest at %d rows per tile, %.0f GMAC/s",
                             best_r, macs / best / 1e9);
                printf("%-5d %11.1f %11.1f %6.2fx %11.1f %6.2fx %11.1f %6.2fx %11.1f %6.2fx "
                       "%11.1f %6.2fx%s\n",
                       T, rate[0] * 1e3, rate[1] * 1e3, rate[0] / rate[1], rate[2] * 1e3,
                       rate[0] / rate[2], rate[3] * 1e3, rate[0] / rate[3], rate[4] * 1e3,
                       rate[0] / rate[4], rate[5] * 1e3, rate[0] / rate[5], note);
                fflush(stdout);

                /* The same five tiles with the bytes resident: 2048 rows of flat weights
                 * and the whole panel are 6 MiB, which is this machine's L3, against the
                 * 250 MiB the row above walks out of DRAM.
                 *
                 * The two lines together are what ideas.md 7.22 leaves "unpiced". A wider
                 * macro-tile changes how many accumulators a core holds; it does not change
                 * how many weight bytes the bus is asked for. So the gap between the two
                 * lines is the part no tile can recover, and the most the resident line ever
                 * says is the ceiling the question is about. Times are normalised to the
                 * 100000 rows of the row above, so a column reads against its own cell. */
                double rrate[6];
                for (int tr = 1; tr <= 5; tr++) {
                    rrate[tr] = 1e30;
                    for (int rep = 0; rep < 3; rep++) {
                        flat_job b = { wf, sw, x8r, spr, y, RROWS_, K_, T, tr };
                        double t0 = now();
                        qe_pool_run(flat_walk_task, &b, nthr);
                        double dt = now() - t0;
                        if (dt < rrate[tr]) rrate[tr] = dt;
                    }
                    rrate[tr] *= (double)DROWS_ / (double)RROWS_;
                }
                double rbest = rrate[1];
                for (int tr = 2; tr <= 5; tr++)
                    if (rrate[tr] < rbest)
                        rbest = rrate[tr];
                printf("%-5s", "resid");
                for (int tr = 1; tr <= 5; tr++)
                    printf(" %11.1f %6.2fx", rrate[tr] * 1e3, rate[0] / rrate[tr]);
                printf("        <- resident: %.0f GMAC/s at the fastest tile\n",
                       (double)RROWS_ * (double)K_ * (double)T /
                           (rbest * (double)RROWS_ / (double)DROWS_) / 1e9);
                fflush(stdout);
            }
            qe_i8_unset_ssd();
            qe_i8_set_hoist(-1);
            qe_i8_set_rows(-1);
            qe_free_huge(bf, (size_t)2 * MAXT * (size_t)frows * sizeof(float));
            qe_free_huge(sw, (size_t)DROWS_ * sizeof(float));
            qe_free_huge(wf, (size_t)DROWS_ * (size_t)K_);
        }

        qe_free_huge(x8r, pnr);
        qe_free_huge(spr, (size_t)MAXT * (size_t)(K_ / 32) * sizeof(float));
        qe_free_huge(ppr, (size_t)MAXT * (size_t)(K_ / 32) * sizeof(int32_t));
    }
#endif

    qe_free_huge(wq, nq);
    qe_free_huge(w4, n4);
    qe_free_huge(w3, n3);
    qe_free_huge(x, (size_t)MAXT * (size_t)K_ * sizeof(float));
    qe_free_huge(x8, pn);
    qe_free_huge(sp, (size_t)MAXT * (size_t)(K_ / 32) * sizeof(float));
    qe_free_huge(pp, (size_t)MAXT * (size_t)(K_ / 32) * sizeof(int32_t));
    qe_free_huge(y, (size_t)ROWS_ * sizeof(float) * MAXT);
}

/* --- section 6: how many passes a bucket of tokens costs --- */

/* A pass over one output row pays the codebook decode of every block of that row plus one dot
 * per token, so the tokens are the cheap part of a pass and the passes are not. The family had
 * forms for eight tokens; a 512-token chunk routes 5120 slots over 512 experts, which is a
 * bucket of ten, so the modal bucket walked its expert's weights twice. This prices the widths
 * a bucket arrives at against the eight the family stopped at, on the panel a bucket is handed
 * (row-scaled, filled by the same call the grouped expert pass uses) and on a tensor that does
 * not survive the L3, which is what an expert matrix is at a 512 window.
 *
 * The widths are a change of which tokens share a pass and nothing else: a token's block order,
 * seed and exit are the same in every form, so the gate is the zero the tile gate above holds,
 * not a tolerance. */
static void check_group(qe_type t, const char *label, const i8_job *base, float *buf, long rows,
                        int T, int wide) {
    i8_job a = *base, b = *base;
    a.rows = b.rows = rows;
    a.T = b.T = T;
    a.y = buf;
    b.y = buf + (size_t)MAXT * (size_t)rows;
    qe_i8_set_tmax(8);
    qe_pool_run(i8_walk_task, &a, qe_pool_size());
    qe_i8_set_tmax(wide);
    qe_pool_run(i8_walk_task, &b, qe_pool_size());
    long moved = 0;
    for (long i = 0; i < (long)((size_t)T * rows); i++)
        if (b.y[i] != a.y[i])
            moved++;
    printf("  %-8s %d-token passes over %ld rows x %d tokens: %ld of %ld outputs moved  %s\n", label,
           wide, rows, T, moved, (long)((size_t)T * rows), moved ? "FAIL" : "ok");
    fflush(stdout);
    if (moved)
        exit(1);
}

static void section_groups(int nthr) {
    const size_t i4_rb = (size_t)(K_ / 32) * sizeof(qe_block_iq4_nl);
    const size_t i3_rb = (size_t)(K_ / 256) * sizeof(qe_block_iq3_s);
    const size_t n4 = (size_t)ROWS_ * i4_rb, n3 = (size_t)ROWS_ * i3_rb;
    const long pn = (size_t)MAXT * (size_t)K_;
    char *w4 = qe_alloc_huge(n4);
    char *w3 = qe_alloc_huge(n3);
    float *x = qe_alloc_huge((size_t)MAXT * (size_t)K_ * sizeof(float));
    int8_t *x8 = qe_alloc_huge(pn);
    float *sp = qe_alloc_huge((size_t)MAXT * (size_t)(K_ / 32) * sizeof(float));
    int32_t *pp = qe_alloc_huge((size_t)MAXT * (size_t)(K_ / 32) * sizeof(int32_t));
    float *y = qe_alloc_huge((size_t)ROWS_ * sizeof(float) * MAXT);
    if (!w4 || !w3 || !x || !x8 || !sp || !pp || !y) {
        fprintf(stderr, "alloc\n");
        exit(1);
    }
    fill(w4, n4, sizeof(qe_block_iq4_nl));
    fill(w3, n3, sizeof(qe_block_iq3_s));
    fill_x(x, (size_t)MAXT * (size_t)K_);
    /* One scale per row, which is the panel the engine's expert pass fills and the panel the
     * walks that have a row-scaled form were selected for. */
    for (int t = 0; t < MAXT; t++)
        qe_quant_q8_0_row_rs(x + (size_t)t * (size_t)K_, x8 + (size_t)t * (size_t)K_,
                             sp + (size_t)t * (size_t)(K_ / 32),
                             pp + (size_t)t * (size_t)(K_ / 32), K_);
    memset(y, 0, (size_t)ROWS_ * sizeof(float) * MAXT);

    qe_i8_set_rowscale(1);
    qe_i8_set_rows(1);

    static const int TT[] = { 8, 9, 10, 11, 12, 13, 14, 16, 20, 24, 32 };
    static const int WW[] = { 10, 12, 16 };
    {
        const long crows = 4103;
        float *buf = qe_alloc_huge((size_t)2 * MAXT * (size_t)crows * sizeof(float));
        if (!buf) {
            fprintf(stderr, "alloc\n");
            exit(1);
        }
        const i8_job j4 = { QE_IQ4_NL, w4, x8, sp, pp, y, ROWS_, K_, i4_rb, K_ / 32, 8 };
        const i8_job j3 = { QE_IQ3_S, w3, x8, sp, pp, y, ROWS_, K_, i3_rb, K_ / 256, 8 };
        for (int ti = 0; ti < (int)(sizeof TT / sizeof *TT); ti++)
            for (int wi = 0; wi < (int)(sizeof WW / sizeof *WW); wi++) {
                check_group(QE_IQ4_NL, "IQ4_NL", &j4, buf, crows, TT[ti], WW[wi]);
                check_group(QE_IQ3_S, "IQ3_S", &j3, buf, crows, TT[ti], WW[wi]);
            }
        qe_free_huge(buf, (size_t)2 * MAXT * (size_t)crows * sizeof(float));
    }

    printf("\n%-5s %10s %10s %8s %10s %8s %10s %8s   %s\n", "T", "8 tokens", "10 tokens", "gain",
           "12 tokens", "gain", "16 tokens", "gain", "IQ4_NL: ms for one walk of the tensor");
    for (int ti = 0; ti < (int)(sizeof TT / sizeof *TT); ti++) {
        const int T = TT[ti];
        double best[4] = { 1e30, 1e30, 1e30, 1e30 };
        const int tw[4] = { 8, 10, 12, 16 };
        for (int rep = 0; rep < 3; rep++)
            for (int leg = 0; leg < 4; leg++) {
                qe_i8_set_tmax(tw[leg]);
                i8_job j = { QE_IQ4_NL, w4, x8, sp, pp, y, ROWS_, K_, i4_rb, K_ / 32, T };
                double t0 = now();
                qe_pool_run(i8_walk_task, &j, nthr);
                double dt = now() - t0;
                if (dt < best[leg])
                    best[leg] = dt;
                if (!(y[7] == y[7]) || y[7] == 0.0f) {
                    fprintf(stderr, "the integer walk wrote nothing useful at T=%d\n", T);
                    exit(1);
                }
            }
        printf("%-5d %10.1f %10.1f %7.2fx %10.1f %7.2fx %10.1f %7.2fx\n", T, best[0] * 1e3,
               best[1] * 1e3, best[0] / best[1], best[2] * 1e3, best[0] / best[2], best[3] * 1e3,
               best[0] / best[3]);
        fflush(stdout);
    }

    printf("\n%-5s %10s %10s %8s %10s %8s %10s %8s   %s\n", "T", "8 tokens", "10 tokens", "gain",
           "12 tokens", "gain", "16 tokens", "gain", "IQ3_S: the same four ceilings on the other expert format");
    for (int ti = 0; ti < (int)(sizeof TT / sizeof *TT); ti++) {
        const int T = TT[ti];
        double best[4] = { 1e30, 1e30, 1e30, 1e30 };
        const int tw[4] = { 8, 10, 12, 16 };
        for (int rep = 0; rep < 3; rep++)
            for (int leg = 0; leg < 4; leg++) {
                qe_i8_set_tmax(tw[leg]);
                i8_job j = { QE_IQ3_S, w3, x8, sp, pp, y, ROWS_, K_, i3_rb, K_ / 256, T };
                double t0 = now();
                qe_pool_run(i8_walk_task, &j, nthr);
                double dt = now() - t0;
                if (dt < best[leg])
                    best[leg] = dt;
            }
        printf("%-5d %10.1f %10.1f %7.2fx %10.1f %7.2fx %10.1f %7.2fx\n", T, best[0] * 1e3,
               best[1] * 1e3, best[0] / best[1], best[2] * 1e3, best[0] / best[2], best[3] * 1e3,
               best[0] / best[3]);
        fflush(stdout);
    }
    printf("ms is a whole walk of the tensor at one row-scaled recode of the activations, and a\n"
           "column is the widest token pass that walk was allowed to open. A bucket of ten walked\n"
           "in one pass is the whole item: the classes only change which tokens share a pass.\n");
    fflush(stdout);

    qe_i8_set_tmax(-1);
    qe_i8_set_rows(-1);
    qe_i8_set_rowscale(-1);
    qe_free_huge(w4, n4);
    qe_free_huge(w3, n3);
    qe_free_huge(x, (size_t)MAXT * (size_t)K_ * sizeof(float));
    qe_free_huge(x8, pn);
    qe_free_huge(sp, (size_t)MAXT * (size_t)(K_ / 32) * sizeof(float));
    qe_free_huge(pp, (size_t)MAXT * (size_t)(K_ / 32) * sizeof(int32_t));
    qe_free_huge(y, (size_t)ROWS_ * sizeof(float) * MAXT);
}

int main(int argc, char **argv) {
    int nthr = QE_THREADS_AUTO;
    if (argc > 1 && atoi(argv[1]) > 0)
        nthr = atoi(argv[1]);
    qe_pool_init(nthr, NULL, 0);
    const int n = qe_pool_size();
    printf("%d threads\n\n", n);
    fflush(stdout);
    /* Every panel below is filled by qe_quant_q8_0_row[s], one scale per 32 activations, so
     * the knob that tells a walk which kind of panel it was handed is pinned off here and set
     * by hand by the legs that measure the row scale. Left at the configuration default the
     * fills and the walks in this file would disagree about the panel. */
    qe_i8_set_rowscale(0);
    /* The pairing section on its own, for when the question is only that one. */
    if (argc > 2 && strcmp(argv[2], "pairs") == 0) {
        section_pairs(n);
        return 0;
    }
    /* And the integer tile on its own, for the same reason. */
    if (argc > 2 && strcmp(argv[2], "tile") == 0) {
        section_tile(n);
        return 0;
    }
    /* And the width of a token pass on its own. */
    if (argc > 2 && strcmp(argv[2], "groups") == 0) {
        section_groups(n);
        return 0;
    }
    section_shapes(n);
    section_pairs(n);
    section_tile(n);
    section_bytes(n);
    section_groups(n);
    return 0;
}
