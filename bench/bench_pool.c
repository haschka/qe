/* The pool's park/wake handshake, tested without a checkpoint.

   A worker gives up spinning and parks on a condvar once a dispatch has kept it
   waiting long enough. That handover is the only place in the engine where a
   job can be published and never run, and when it goes wrong nothing fails:
   the master spins on the completion counter for ever and a decode step hangs
   with 87 GiB of weights resident and no way to tell it is dead. The test is
   therefore a hang detector as much as a correctness test, and it is the reason
   `make pool` has a fuse of its own.

   The sequence that loses a wake-up is a worker announcing itself after the
   master has looked for sleepers, so the only way to exercise it is to make the
   workers actually park between dispatches - hence the idle gaps, which are
   longer than the spin budget rather than shorter. The second loop is the same
   dispatch with no gap at all, and what it reports is the thing the spin budget
   exists to protect: the cost of waking a parked pool.

     make pool        # or: ./bin/bench_pool [THREADS] [GAP_MS]
*/
#include "qe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXT 64
#define NDISPATCH 48

typedef struct {
    int nthreads;
    int mark;
    long *hits;
    int *ran;
} job;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void sleep_ms(long ms) {
    struct timespec t = { .tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L };
    nanosleep(&t, NULL);
}

static void bump(void *vp, int tid, int nthreads) {
    job *j = vp;
    (void)nthreads;
    if (tid < j->nthreads) {
        __atomic_fetch_add(&j->hits[tid], 1, __ATOMIC_RELAXED);
        j->ran[tid] = j->mark;
    }
}

/* The stage graph: several items run as one dispatch, each waiting on the items its
 * `deps` mask names and on nothing else. Three things have to be true about it and
 * none of them shows up in a forward pass, where a broken one is a wrong number
 * rather than a crash. A slot has to run exactly once however the threads race for
 * it. A dependent item must not start on a predecessor that is *claimed* rather
 * than finished, which is the difference between the two counters here and the one
 * counter that would be tempting to write. And two independent items have to
 * actually share the machine, because an item set run one item after another would
 * be correct, deterministic, and the whole reason the primitive exists.
 *
 * So the tasks spin. Without work in a task a race that is there is over in
 * nanoseconds, and co-running is indistinguishable from queued. */
typedef struct {
    int nitems;
    int nthreads[QE_POOL_MAX_ITEMS];
    uint32_t deps[QE_POOL_MAX_ITEMS];
    volatile long runs[QE_POOL_MAX_ITEMS][MAXT];
    volatile long done[QE_POOL_MAX_ITEMS];
    volatile int live[QE_POOL_MAX_ITEMS];
    volatile int active; /* items with a slot inside a task right now */
    int widest;          /* the widest overlap any rep of the run reached */
    int bad;
} graph;

typedef struct {
    graph *g;
    int i;
    long spin; /* work in the task, so a race has time to be seen */
} gslot;

static void gtask(void *vp, int tid, int nthreads) {
    (void)nthreads;
    const gslot *s = vp;
    graph *g = s->g;
    const int i = s->i;
    for (int d = 0; d < g->nitems; d++)
        if (((g->deps[i] >> (unsigned)d) & 1u) &&
            __atomic_load_n(&g->done[d], __ATOMIC_ACQUIRE) != g->nthreads[d])
            g->bad = 1;
    __atomic_fetch_add(&g->runs[i][tid], 1, __ATOMIC_RELAXED);
    if (__atomic_fetch_add(&g->live[i], 1, __ATOMIC_ACQ_REL) == 0) {
        const int a = __atomic_add_fetch(&g->active, 1, __ATOMIC_ACQ_REL);
        int w = __atomic_load_n(&g->widest, __ATOMIC_RELAXED);
        while (a > w && !__atomic_compare_exchange_n(&g->widest, &w, a, 0, __ATOMIC_RELAXED,
                                                     __ATOMIC_RELAXED))
            ;
    }
    for (volatile long k = 0; k < s->spin; k++)
        ;
    if (__atomic_sub_fetch(&g->live[i], 1, __ATOMIC_ACQ_REL) == 0)
        __atomic_sub_fetch(&g->active, 1, __ATOMIC_ACQ_REL);
    /* Last, after anything this slot produced: a dependent item is allowed to start
     * on the sight of this count, so a predecessor that published after it would be
     * still running while its own output was already being read. */
    __atomic_add_fetch(&g->done[i], 1, __ATOMIC_ACQ_REL);
}

/* The stack traffic a dispatch has to survive being published into.
 *
 * A stage graph names its work in an array the dispatch built. In this file's
 * first leg nothing runs between one dispatch and the next, so the frame that held
 * it is still holding the same numbers when a worker looks at it late; a forward
 * pass does not behave like that, and a graph whose work array is only as durable
 * as the frame that built it was found in an engine seven minutes into a prompt and
 * not here. So the test churns the stack between a graph and the dispatch after it,
 * the way the next function of a layer would. */
__attribute__((noinline)) static void scribble_stack(void) {
    volatile char pad[16384];
    for (size_t i = 0; i < sizeof pad; i++)
        pad[i] = (char)(i * 7u);
}

int main(int argc, char **argv) {
    /* "all" is every cpu in the affinity mask, "auto" is what the engine takes by
     * default (QE_THREADS_AUTO). Running the two back to back is the whole
     * measurement: same binary, same handshake, one core of difference. */
    int threads = 20;
    if (argc > 1)
        threads = !strcmp(argv[1], "auto") ? QE_THREADS_AUTO
                  : !strcmp(argv[1], "all") ? qe_cpus()
                                            : atoi(argv[1]);
    long gap = argc > 2 ? atol(argv[2]) : 120;

    /* Half again the spin budget below, so a worker is certainly asleep before
     * the next dispatch is published. */
    if (gap < 20)
        gap = 20;
    qe_fuse(120);
    int n = qe_pool_init(threads, NULL, 0);
    if (n < 2) {
        fprintf(stderr, "pool: need at least two threads, got %d\n", n);
        return 1;
    }
    long *hits = calloc((size_t)n, sizeof *hits);
    long *expect = calloc((size_t)n, sizeof *expect);
    int *ran = calloc((size_t)n, sizeof *ran);
    if (!hits || !expect || !ran)
        return 1;
    printf("== pool ==\n  threads %d, %d cold dispatches with %ld ms gaps, then %d hot ones\n",
           n, NDISPATCH, gap, NDISPATCH);
    fflush(stdout);

    /* How a range is divided is as much a contract as the handshake is. A stage
     * whose per-element answer must not depend on the width of the pool - the
     * indexer's block scores are the case that depends on it - rests on the
     * slices being ordered, disjoint and exhaustive at every width, including the
     * widths where a thread is handed nothing. A task function cannot check that
     * for itself, so it is checked where the pool is. */
    int bad = 0;
    for (int w = 1; w <= n; w++) {
        for (long items = 0; items <= 2048; items++) {
            long prev = 0;
            for (int t = 0; t < w; t++) {
                long lo, hi;
                qe_slice(items, t, w, &lo, &hi);
                if (lo < prev || hi < lo || hi > items)
                    bad++;
                prev = hi;
            }
            if (prev != items)
                bad++;
        }
    }
    printf("  slice ranges     %s (%d widths, ranges of 0..2048 items)\n",
           bad ? "FAIL" : "pass", n);
    fflush(stdout);

    /* Per dispatch, not summed. This host is shared, and one multi-second
     * scheduling stall out of forty-eight moves an average by more than the
     * thing being measured, which is exactly how a 0.6 us handshake came to be
     * reported as 3300 us. Median is the number; worst is the warning. */
    double cold_s[NDISPATCH], hot_s[NDISPATCH];
    int ncold = 0;
    int nhot = 0;
    long slowest = 0;
    double cold_total = 0.0;
    for (int d = 0; d < NDISPATCH; d++) {
        sleep_ms(gap);
        /* Varying the width covers the workers that sit out a dispatch, which is
         * the other way a completion count can end up waiting for somebody who
         * is not going to run. */
        int width = 2 + (d % (n - 1));
        job j = { .nthreads = width, .mark = d + 1, .hits = hits, .ran = ran };
        double t0 = now();
        qe_pool_run(bump, &j, width);
        double dt = now() - t0;
        cold_total += dt;
        cold_s[ncold < NDISPATCH ? ncold++ : 0] = dt;
        long us = (long)(dt * 1e6);
        if (us > slowest)
            slowest = us;
        for (int t = 0; t < n; t++) {
            int ran_this = ran[t] == d + 1;
            if (t < width) {
                expect[t]++;
                if (!ran_this) {
                    printf("  FAIL: width %d, slice %d never ran (dispatch %d)\n", width, t, d);
                    bad++;
                }
            } else if (ran_this) {
                /* A worker outside the width running anyway is the bug that made
                 * the completion count wrong for a whole generation of this pool. */
                printf("  FAIL: width %d, slice %d ran anyway (dispatch %d)\n", width, t, d);
                bad++;
            }
            if (hits[t] != expect[t]) {
                printf("  FAIL: slice %d ran %ld times, expected %ld\n", t, hits[t], expect[t]);
                bad++;
            }
        }
    }

    double h0 = now();
    for (int d = 0; d < NDISPATCH; d++) {
        job j = { .nthreads = n, .mark = NDISPATCH + d + 1, .hits = hits, .ran = ran };
        double t1 = now();
        qe_pool_run(bump, &j, n);
        hot_s[nhot < NDISPATCH ? nhot++ : 0] = now() - t1;
        for (int t = 0; t < n; t++)
            if (ran[t] != NDISPATCH + d + 1) {
                printf("  FAIL: hot dispatch %d, slice %d did not run\n", d, t);
                bad++;
            }
    }
    double hot_us = (now() - h0) * 1e6 / NDISPATCH;
    (void)hot_us;
    for (int t = 0; t < n; t++)
        expect[t] += NDISPATCH;
    for (int t = 0; t < n; t++)
        if (hits[t] != expect[t]) {
            printf("  FAIL: slice %d ran %ld times, expected %ld\n", t, hits[t], expect[t]);
            bad++;
        }

    double cold_med = median(cold_s, ncold), hot_med = median(hot_s, nhot);
    double hot_worst = hot_s[(size_t)(nhot - 1)];
    printf("  parked wake-up   %.1f ms median, %.1f ms mean, slowest %ld ms\n", cold_med * 1e3,
           cold_total / NDISPATCH * 1e3, slowest);
    printf("  hot dispatch     %.1f us median, %.1f us mean, slowest %.1f us\n", hot_med * 1e6,
           hot_us, hot_worst * 1e6);

    /* The stage graph, rep after rep: the same four items again and again, because
     * the races it is exposed to are the taking of one slot by two threads and the
     * publishing of an answer one step too early, and neither announces itself
     * except by a count that is off by one eventually. */
    const int greps = n > MAXT ? 0 : 60;
    long gruns = 0;
    int widest = 0;
    for (int rep = 0; rep < greps && !bad; rep++) {
        graph g;
        gslot slot[4];
        qe_pool_item items[4];
        memset(&g, 0, sizeof g);
        g.nitems = 4;
        /* Two halves that between them fill the pool, a stage that needs both of
         * them, and a tail that needs that. The halves are the only pair in the
         * graph that may co-run, so any overlap counted below came from them. */
        const int half = n / 2;
        g.nthreads[0] = half;
        g.nthreads[1] = n - half;
        g.nthreads[2] = n;
        g.nthreads[3] = n > 3 ? 3 : n;
        g.deps[2] = (1u << 0) | (1u << 1);
        g.deps[3] = 1u << 2;
        for (int i = 0; i < 4; i++) {
            slot[i] = (gslot){ &g, i, 100000L };
            items[i] = (qe_pool_item){ gtask, &slot[i], g.nthreads[i], g.deps[i] };
        }
        qe_pool_run_items(items, 4);
        for (int i = 0; i < 4; i++)
            for (int t = 0; t < n; t++) {
                const long want = t < g.nthreads[i] ? 1 : 0;
                if ((long)g.runs[i][t] != want) {
                    printf("  FAIL: item %d slot %d ran %ld times, expected %ld\n", i, t,
                           g.runs[i][t], want);
                    bad++;
                }
                gruns += (long)g.runs[i][t];
            }
        if (g.bad) {
            printf("  FAIL: an item started while a predecessor was still running\n");
            bad++;
        }
        if (g.widest > widest)
            widest = g.widest;
    }
    if (greps) {
        printf("  stage graph      %d dispatches of 4 items, %ld slots, widest overlap %d"
               " items at once\n",
               greps, gruns, widest);
        if (widest < 2) {
            printf("  FAIL: two independent items were never resident at once\n");
            bad++;
        }
    }

    /* A graph dispatch and an ordinary one, back to back, over and over.
     *
     * The counts above are all about one graph being correct. This leg is about what
     * a graph leaves behind: it hands the pool back to the ordinary path, and an
     * ordinary dispatch completes only if every worker that walked the graph got out
     * of it. A worker still reading a finished graph never reports a slice of the
     * next dispatch, and the master of that dispatch then waits on a completion count
     * it can never reach - so this leg is mostly a shape for the target's deadline to
     * catch, and the wall it prints is the proof that no round stalled. The short spin
     * is deliberate: it is the graph ending while a worker is still in it that this
     * tests, not the arithmetic of the task. */
    const int ireps = n > MAXT ? 0 : 240;
    long bhits[MAXT] = { 0 };
    int bran[MAXT] = { 0 };
    double worst = 0;
    for (int rep = 0; rep < ireps && !bad; rep++) {
        graph g;
        gslot slot[4];
        qe_pool_item items[4];
        memset(&g, 0, sizeof g);
        g.nitems = 4;
        const int half = n / 2;
        g.nthreads[0] = half;
        g.nthreads[1] = n - half;
        g.nthreads[2] = n;
        g.nthreads[3] = n > 3 ? 3 : n;
        g.deps[2] = (1u << 0) | (1u << 1);
        g.deps[3] = 1u << 2;
        for (int i = 0; i < 4; i++) {
            slot[i] = (gslot){ &g, i, 300 };
            items[i] = (qe_pool_item){ gtask, &slot[i], g.nthreads[i], g.deps[i] };
        }
        const double t0 = now();
        qe_pool_run_items(items, 4);
        scribble_stack();
        job j = { n, rep + 1, bhits, bran };
        qe_pool_run(bump, &j, n);
        const double dt = now() - t0;
        if (dt > worst)
            worst = dt;
        for (int i = 0; i < 4 && !bad; i++)
            if (g.done[i] != g.nthreads[i]) {
                printf("  FAIL: item %d of the interleaved graph returned %ld of %d slots\n", i,
                       g.done[i], g.nthreads[i]);
                bad++;
            }
    }
    if (ireps) {
        int lost = 0;
        for (int t = 0; t < n; t++)
            if (bhits[t] != ireps || bran[t] != ireps)
                lost++;
        printf("  graph then queue %d pairs, worst round %.2f ms, every worker in every"
               " queue dispatch: %s\n",
               ireps, worst * 1e3, lost ? "no" : "yes");
        if (lost) {
            printf("  FAIL: %d worker(s) missed a dispatch that followed a graph\n", lost);
            bad++;
        }
        if (worst > 0.05) {
            printf("  FAIL: a round took %.2f ms, which is a stall and not a dispatch\n",
                   worst * 1e3);
            bad++;
        }
    }

    printf("%s\n", bad ? "FAIL" : "PASS");
    fflush(stdout);
    qe_pool_shutdown();
    free(hits);
    free(expect);
    free(ran);
    return bad ? 1 : 0;
}
