/* Do the two big walks of an ingested token want the same part of the machine?
 *
 * ideas.md §7.8 prices one structural idea: run a layer's expert walk while a
 * dense walk is still in flight, so the token pays the max of the two stages
 * instead of their sum. At the width that ships (`/tmp/f9_wide.log`) that is
 * 14.56 ms of experts and 7.60 ms of dense projections against a 22.16 ms sum,
 * and the idea is only worth anything if the two walks are limited by different
 * things. Nobody had measured that. The section quoted 1.52x on the assumption
 * that they were, and an assumption of that size is exactly what this file
 * exists to replace with a number.
 *
 * So: one pool, one barrier, k threads walking a dense Q8_0 tensor at the
 * shipping width and n-k threads walking IQ3_S expert rows at the width the
 * expert kernels seat in registers. Each half is timed alone on the same thread
 * group, then together, and the mixed wall is judged against the thing a pipeline
 * would actually displace: both walks at the full width, one after the other.
 * That is the only honest baseline here, because a split takes the machine apart
 * rather than adding to it. If the two walks want different parts of the machine
 * the mixed wall lands under the full-width sequential pass; if the machine is
 * merely additive across them it lands on it, which buys nothing however the split
 * is drawn. Judging the same run against the sum of the two halves alone on their
 * own share of the pool instead - which is what the first version of this file
 * did - reports a near-additive machine as a 1.5x win, because each half is doing
 * the full amount of work on half the machine, so the sum it is compared with is
 * about twice the work the sequential pass is doing. ideas.md §7.8 carried that
 * reading as a priced prize until this run was taken back to `base`.
 *
 * No checkpoint here on purpose: the shapes are the model's own (attn_qkv
 * 10240x2560 Q8_0 at T=512, the row block the forward pass uses; expert rows of
 * 640x2560 IQ3_S at T=8) but they are synthetic bytes, so the answer is
 * reproducible in a minute instead of in the time a load of 87 GiB costs.
 *
 * The expert half walks every row of its pool against the same eight
 * activations, which is the hot-expert shape the shared-dequant kernels exist
 * for and the shape `make moe`'s shared-dequant table measures; uniform routing
 * would spread the same arithmetic over more distinct bytes without changing
 * what each walk is limited by.
 */
#include <stdatomic.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "qe.h"

#define DENSE_T 512 /* QE_CHUNK_DEFAULT: rows through one walk of a projection */
#define EXPR_T 8    /* rows through one walk of an expert row at QE_PARK=0 */
#define MAXT 8

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t rstate = 0x243f6a8885a308d3ull;
static uint64_t rnd(void) {
    rstate ^= rstate << 13;
    rstate ^= rstate >> 7;
    rstate ^= rstate << 17;
    return rstate;
}

static size_t rb_of(qe_type t, long cols) {
    const qe_type_info *ti = qe_type_info_of(t);
    return (size_t)(cols / (long)ti->block_ne * (long)ti->block_sz);
}

/* --- the two workloads ---------------------------------------------------------------- */

/* One dense projection walked once, T activations against every row. */
typedef struct {
    const char *w;
    const float *x;
    float *y;
    long rows, K, ldd;
    size_t rb;
    int T, reps;
} dense_job;

/* One expert matrix walked once, T activations against every row. */
typedef struct {
    const char *w;
    const float *x;
    float *y;
    long rows, K, ldd;
    size_t rb;
    int T, reps;
} expert_job;

/* Rows [lo,hi) of the dense tensor: the unit of work, whether a thread count or a
 * queue hands it out. */
static void dense_rows(const dense_job *j, long lo, long hi) {
    for (int r = 0; r < j->reps; r++)
        if (qe_gemv_multi_t(QE_Q8_0, j->w + (size_t)lo * j->rb, j->x, j->y + lo, hi - lo, j->K,
                            j->ldd, j->T, j->K, j->rows, (long)j->rb, (size_t)1 << 20) != 0) {
            fprintf(stderr, "no multi-row kernel for Q8_0\n");
            exit(1);
        }
}

static void expert_rows(const expert_job *j, long lo, long hi) {
    const float *xs[MAXT];
    float *ys[MAXT];
    for (int t = 0; t < j->T; t++) {
        xs[t] = j->x + (size_t)t * (size_t)j->K;
        ys[t] = j->y + (size_t)t * (size_t)j->rows + lo;
    }
    for (int r = 0; r < j->reps; r++)
        if (qe_gemv_multi_ptrs_t(QE_IQ3_S, j->w + (size_t)lo * j->rb, xs, ys, j->T, hi - lo, j->K,
                                j->ldd) != 0) {
            fprintf(stderr, "no shared-dequant kernel for IQ3_S\n");
            exit(1);
        }
}

static void dense_part(const dense_job *j, int tid, int nt) {
    long lo, hi;
    qe_slice(j->rows, tid, nt, &lo, &hi);
    if (lo < hi)
        dense_rows(j, lo, hi);
}

static void expert_part(const expert_job *j, int tid, int nt) {
    long lo, hi;
    qe_slice(j->rows, tid, nt, &lo, &hi);
    if (lo < hi)
        expert_rows(j, lo, hi);
}

static void dense_task(void *v, int tid, int nt) { dense_part((const dense_job *)v, tid, nt); }
static void expert_task(void *v, int tid, int nt) { expert_part((const expert_job *)v, tid, nt); }

typedef struct {
    dense_job *d;
    expert_job *e;
    int split; /* threads [0,split) walk the dense tensor, the rest the experts */
} mix_job;

static void mix_task(void *v, int tid, int nt) {
    mix_job *m = v;
    if (tid < m->split)
        dense_part(m->d, tid, m->split);
    else
        expert_part(m->e, tid - m->split, nt - m->split);
}

/* --- the schedule an engine would actually run ---------------------------------------
 *
 * A fixed k + (n-k) split cannot show the one thing a pipeline is for: a thread that
 * finishes its half early has nowhere to go while the other half still has rows, and
 * a split therefore reads as a wash on a machine that is merely additive even when
 * the tail it cannot reach is worth something. The forward pass has no such wall, it
 * has a task queue, so the schedule worth pricing is one queue over both walks where
 * any thread takes the next range of whichever walk still has work. Two cursors and
 * an alternating preference is that queue; the tail it removes is the prize. */
typedef struct {
    dense_job *d;
    expert_job *e;
    long chunk;
    long d_lim, e_lim;
    _Alignas(64) _Atomic long cd;
    _Alignas(64) _Atomic long ce;
} dyn_job;

static void dyn_reset(dyn_job *m) {
    atomic_store_explicit(&m->cd, 0, memory_order_relaxed);
    atomic_store_explicit(&m->ce, 0, memory_order_relaxed);
}

static void dyn_task(void *v, int tid, int nt) {
    dyn_job *m = v;
    (void)nt;
    for (int which = tid & 1;; which ^= 1) {
        _Atomic long *cur = which ? &m->ce : &m->cd;
        const long lim = which ? m->e_lim : m->d_lim;
        const long lo = atomic_fetch_add_explicit(cur, m->chunk, memory_order_relaxed);
        if (lo < lim) {
            const long hi = lo + m->chunk > lim ? lim : lo + m->chunk;
            if (which)
                expert_rows(m->e, lo, hi);
            else
                dense_rows(m->d, lo, hi);
        } else if (atomic_load_explicit(&m->cd, memory_order_relaxed) >= m->d_lim &&
                   atomic_load_explicit(&m->ce, memory_order_relaxed) >= m->e_lim) {
            return;
        }
    }
}

/* --- measurement --------------------------------------------------------------------- */

/* Best of three: the same rule every timing leg here uses, because the first pass
 * over a hundred megabytes of fresh pages is not a measurement of the kernel. */
static double time_pool(void (*fn)(void *, int, int), void *arg, int nt) {
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        qe_pool_run(fn, arg, nt);
        double dt = now() - t0;
        if (dt < best)
            best = dt;
    }
    return best;
}

/* Both walks at full width, one after the other, inside one timed region: the
 * schedule the forward pass runs today, barrier and all. */
static double time_seq(const dense_job *d, const expert_job *e, int nt) {
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        qe_pool_run(dense_task, (void *)d, nt);
        qe_pool_run(expert_task, (void *)e, nt);
        double dt = now() - t0;
        if (dt < best)
            best = dt;
    }
    return best;
}

static double time_dyn(dyn_job *m, int nt) {
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        dyn_reset(m);
        double t0 = now();
        qe_pool_run(dyn_task, m, nt);
        double dt = now() - t0;
        if (dt < best)
            best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    qe_fuse(argc > 1 ? atoi(argv[1]) : 240);
    int nthr = argc > 2 ? (!strcmp(argv[2], "auto") ? QE_THREADS_AUTO : atoi(argv[2]))
                        : QE_THREADS_AUTO;
    if (qe_pool_init(nthr, NULL, 0) == 0) {
        fprintf(stderr, "pool\n");
        return 1;
    }
    const int n = qe_pool_size();

    /* Dense: attn_qkv, four copies so the walk cannot live in the L3, at the
     * row block the forward pass uses. Expert: 128 experts x 640 rows of IQ3_S,
     * which is the gate half of 128 experts and 92 MiB. */
    const long drows = 10240L * 4, dK = 2560;
    const long erows = 128L * 640, eK = 2560;
    dense_job d = { 0, 0, 0, drows, dK, dK / 32, rb_of(QE_Q8_0, dK), DENSE_T, 1 };
    expert_job e = { 0, 0, 0, erows, eK, eK / 256, rb_of(QE_IQ3_S, eK), EXPR_T, 1 };
    const size_t dwbytes = (size_t)drows * d.rb, ewbytes = (size_t)erows * e.rb;
    char *dw = qe_alloc_huge(dwbytes);
    char *ew = qe_alloc_huge(ewbytes);
    float *dx = qe_alloc_huge((size_t)DENSE_T * (size_t)dK * sizeof(float));
    float *dy = qe_alloc_huge((size_t)drows * (size_t)DENSE_T * sizeof(float));
    float *ex = qe_alloc_huge((size_t)EXPR_T * (size_t)eK * sizeof(float));
    float *ey = qe_alloc_huge((size_t)erows * (size_t)EXPR_T * sizeof(float));
    if (!dw || !ew || !dx || !dy || !ex || !ey) {
        fprintf(stderr, "alloc\n");
        return 1;
    }
    /* Scales finite (a denormal multiply is a microcode assist and would be
     * measured instead of the kernel), payloads arbitrary but in range. */
    for (size_t i = 0; i < dwbytes; i++)
        dw[i] = (char)((int)((i * 37) % 200) - 100);
    for (size_t b = 0; (b + 2) <= dwbytes; b += sizeof(qe_block_q8_0))
        dw[b] = 0x00, dw[b + 1] = 0x3c;
    for (size_t i = 0; i < ewbytes; i++)
        ew[i] = (char)(uint8_t)(rnd() >> 32);
    for (size_t b = 0; (b + 2) <= ewbytes; b += sizeof(qe_block_iq3_s)) {
        uint16_t h = 0x2c00u | (uint16_t)(rnd() & 0x1ff);
        memcpy(ew + b, &h, 2);
    }
    for (long i = 0; i < DENSE_T * dK; i++)
        dx[i] = (float)((i * 29 + 3) % 2000 - 1000) * 1e-3f;
    for (long i = 0; i < EXPR_T * eK; i++)
        ex[i] = (float)sin(i * 0.0017) * 0.5f;
    memset(dy, 0, (size_t)drows * (size_t)DENSE_T * sizeof(float));
    memset(ey, 0, (size_t)erows * (size_t)EXPR_T * sizeof(float));
    d.w = dw;
    d.x = dx;
    d.y = dy;
    e.w = ew;
    e.x = ex;
    e.y = ey;

    printf("== can an expert walk and a dense walk share the machine, %d threads ==\n", n);
    printf("  dense  attn_qkv %ldx%ld Q8_0, T=%d, %zu copies, 1 MiB row block\n", drows / 4, dK,
           DENSE_T, (size_t)4);
    printf("  expert IQ3_S %lder x %ld, T=%d, %zu experts\n", (long)640, eK, EXPR_T,
           (size_t)(erows / 640));
    fflush(stdout);

    /* One pass of each at full width, to size the repeat counts so the two halves
     * take about as long as each other: the sum-versus-max question is only
     * legible when the two are comparable. */
    d.reps = e.reps = 1;
    double d19 = time_pool(dense_task, &d, n), e19 = time_pool(expert_task, &e, n);
    const double target = 0.25;
    d.reps = (int)(target / d19 + 0.5);
    e.reps = (int)(target / e19 + 0.5);
    if (d.reps < 1)
        d.reps = 1;
    if (e.reps < 1)
        e.reps = 1;
    const double dmacs = (double)drows * (double)dK * DENSE_T * d.reps;
    const double emacs = (double)erows * (double)eK * EXPR_T * e.reps;
    printf("  one pass of the dense walk on %d threads is %.3f s, of the expert walk %.3f s;"
           " repeats %d and %d\n",
           n, d19, e19, d.reps, e.reps);
    /* Alternating legs, every schedule once per pass.
     *
     * The first verdict from this file compared a reference timed once, at the top of
     * the run, against splits timed after two minutes of 19-thread AVX2. That host
     * detail matters more than the comparison did: the absolute rate of this machine
     * moves by tens of percent between one process and the next and drifts in one
     * direction inside a process (the QE_MULTI_G sweep read a 30 % slowdown that was
     * only a warm clock), so the reference sat on boost clocks while every schedule
     * it judged sat under soak, and a schedule could not have drawn 1.00.
     *
     * So nothing here is compared against a number from earlier in the run. Each pass
     * times the sequential pass, one thread split and the shared queue, in an order
     * that alternates between passes, and the best-of-3 inside each leg does the rest:
     * whichever schedule is genuinely faster wins it in the clock state both saw.
     *
     * What the schedules are. `sequential` is the forward pass today: both walks at
     * the full width, one after the other, barrier between. `split` pins k threads to
     * one walk and n-k to the other, which is the only shape a two-group pipeline
     * could take and which reads as a wash on an additive machine by construction -
     * it cannot reach the tail either group leaves. `queue` is one cursor over both
     * walks, so a thread that runs out of expert rows takes dense rows: the thing the
     * pool would actually do and the only leg that can win the tail. */
    const int splits[3] = { n / 4, n / 2, (3 * n) / 4 };
    const long chunk = argc > 3 ? atol(argv[3]) : 0;
    dyn_job dm = { &d, &e, chunk > 0 ? chunk : (drows + erows) / 128, drows, erows, 0, 0 };
    enum { LEG_SEQ, LEG_S0, LEG_S1, LEG_S2, LEG_QUEUE, NLEG };
    double best[NLEG], td_alone = 1e30, te_alone = 1e30;
    for (int i = 0; i < NLEG; i++)
        best[i] = 1e30;
    const int passes = 4;
    printf("  queue chunk %ld rows; %d passes, every schedule once per pass\n", dm.chunk, passes);
    for (int pass = 0; pass < passes; pass++) {
        double leg[NLEG] = { 0 };
        for (int s = 0; s < NLEG; s++) {
            const int which = (pass & 1) ? NLEG - 1 - s : s;
            switch (which) {
            case LEG_SEQ:
                leg[which] = time_seq(&d, &e, n);
                break;
            case LEG_QUEUE:
                leg[which] = time_dyn(&dm, n);
                break;
            default: {
                const int si = which - LEG_S0;
                leg[which] = (splits[si] < 1 || splits[si] >= n)
                                 ? 1e30
                                 : time_pool(mix_task, &(mix_job){ &d, &e, splits[si] }, n);
                break;
            }
            }
            if (leg[which] < best[which])
                best[which] = leg[which];
        }
        double a = time_pool(dense_task, &d, n), b = time_pool(expert_task, &e, n);
        if (a < td_alone) td_alone = a;
        if (b < te_alone) te_alone = b;
        printf("  pass %d: seq %.1f  splits %.1f %.1f %.1f  queue %.1f  queue/seq %.3f\n", pass,
               leg[LEG_SEQ] * 1e3, leg[LEG_S0] * 1e3, leg[LEG_S1] * 1e3, leg[LEG_S2] * 1e3,
               leg[LEG_QUEUE] * 1e3, leg[LEG_QUEUE] / leg[LEG_SEQ]);
        fflush(stdout);
    }
    printf("  %-10s %10s %9s %9s  %s\n", "schedule", "wall ms", "vs seq", "GMAC/s",
           "(dense, expert)");
    printf("  %-10s %10.1f %9s %9.0f  %.0f, %.0f\n", "sequential", best[LEG_SEQ] * 1e3, "1.000",
           (dmacs + emacs) / best[LEG_SEQ] / 1e9, dmacs / td_alone / 1e9, emacs / te_alone / 1e9);
    for (int si = 0; si < 3; si++) {
        if (splits[si] < 1 || splits[si] >= n)
            continue;
        printf("  %2d + %-5d %10.1f %9.3f\n", splits[si], n - splits[si], best[LEG_S0 + si] * 1e3,
               best[LEG_SEQ] / best[LEG_S0 + si]);
    }
    printf("  %-10s %10.1f %9.3f  %.0f, %.0f (mixed)\n", "queue", best[LEG_QUEUE] * 1e3,
           best[LEG_SEQ] / best[LEG_QUEUE], dmacs / best[LEG_QUEUE] / 1e9,
           emacs / best[LEG_QUEUE] / 1e9);
    printf("  vs seq is the only column here a pipeline could bank, and it is now measured\n"
           "  in the same clock state as the thing it has to beat. A queue lands on 1.00\n"
           "  when the machine is additive across the two walks and above it when either\n"
           "  walk leaves a tail the other can eat.\n");
    const double gain = best[LEG_SEQ] / best[LEG_QUEUE];
    const double shortest =
        best[LEG_S0] < best[LEG_S1]
            ? (best[LEG_S0] < best[LEG_S2] ? best[LEG_S0] : best[LEG_S2])
            : (best[LEG_S1] < best[LEG_S2] ? best[LEG_S1] : best[LEG_S2]);
    printf("  verdict: queue %.3f of sequential, best split %.3f; %s\n", gain,
           best[LEG_SEQ] / shortest,
           gain > 1.05
               ? "one queue over both walks shortens the wait the engine pays between the\n"
                 "  two stages, which is the prize bench_forward's expert stage graph takes"
               : "one queue over both walks does not shorten the wait, so the tail a barrier\n"
                 "  leaves is not worth eating here and the graph of ideas.md 7.8 earns nothing");
    qe_pool_shutdown();
    qe_free_huge(dw, dwbytes);
    qe_free_huge(ew, ewbytes);
    qe_free_huge(dx, (size_t)DENSE_T * (size_t)dK * sizeof(float));
    qe_free_huge(dy, (size_t)drows * (size_t)DENSE_T * sizeof(float));
    qe_free_huge(ex, (size_t)EXPR_T * (size_t)eK * sizeof(float));
    qe_free_huge(ey, (size_t)erows * (size_t)EXPR_T * sizeof(float));
    return 0;
}
