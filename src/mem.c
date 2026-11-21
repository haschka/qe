/* Execution pool + page/NUMA primitives.
 *
 * Decode is bandwidth-bound, so the pool's job is not "more parallelism" but
 * (a) enough concurrent streams to saturate the memory controller, (b) no
 * scheduler migration (a migrated worker loses its share of L2/L3 and re-dirties
 * cache lines), and (c) no wake-up latency: at ~60us/token budget, an
 * uncontended futex syscall per layer would be measurable, so workers spin.
 */
#define _GNU_SOURCE
#include "qe.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define QE_PAUSE() _mm_pause()
#else
#define QE_PAUSE() ((void)0)
#endif

long qe_page_size(void) {
    static long ps;
    if (!ps)
        ps = sysconf(_SC_PAGESIZE);
    return ps;
}

/* ---------------------------------------------------------------- fuse */

#include <signal.h>

static void qe_fuse_handler(int sig) {
    (void)sig;
    /* _exit skips the exit handlers, so anything still in stdout is otherwise
     * lost - and a fuse that fires on a block-buffered run takes the numbers
     * with it, which makes the deadline look like a silent failure. Not
     * async-signal-safe, and the process is three instructions from gone. */
    fflush(stdout);
    static const char msg[] = "\n[qe fuse] wall-clock deadline hit, aborting\n";
    ssize_t r = write(2, msg, sizeof(msg) - 1);
    (void)r;
    _exit(124);
}

void qe_fuse(int seconds) {
    if (seconds <= 0) {
        alarm(0);
        return;
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = qe_fuse_handler;
    sigaction(SIGALRM, &sa, NULL);
    alarm((unsigned)seconds);
}

/* ------------------------------------------------------------------- pool */

#define QE_MAX_THREADS 64
#define QE_SPIN_BEFORE_YIELD 4000
/* Roughly twenty milliseconds of _mm_pause before a worker parks.
 *
 * A decode step dispatches every hundred microseconds or so, so parking between
 * two dispatches costs more than the dispatch it is waiting for: a futex wake
 * for nineteen threads is tens of microseconds, and it lands on the critical
 * path of a stage whose whole budget is tens of microseconds. Spinning to
 * roughly a request's worth of idle instead means the pool is awake for every
 * dispatch and only gives up its cores when the engine has really stopped. */
#define QE_SPIN_BEFORE_SLEEP 2000000

typedef struct {
    pthread_t th;
    int id;
    int cpu;
} qe_worker;

static void qe_slice_table(int n, const int *cpus, int ncpus);
static void qe_cpu_weight_table(int n, const int *cpus, int ncpus);
static int qe_pool_pin_list(int *list, int max);

/* Switches the environment reads go one way: unset is the default, and only a
 * non-zero value turns the thing off. A bare "0" once meant "on" for QE_KV_F32,
 * which is the mistake this exists to stop happening twice. */
static int qe_env_wanted(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

/* The cpus this process may run on, not the cpus the machine has: on a host with
 * a cpuset, or under a container runtime, the difference is the whole point of
 * the question "how many threads should the pool have". */
int qe_cpus(void) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof cs, &cs) == 0) {
        int n = CPU_COUNT(&cs);
        if (n > 0)
            return n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

/* One core short of everything. The workers spin (a futex round trip per layer
 * would be measurable), so a pool as wide as the cpu count means the first
 * competing runnable task on the host preempts a spinning worker and the barrier
 * it was standing in waits for a scheduler timeslice: 1.6 us per dispatch at 19
 * threads against 69 us at 20 on this host, worst case 2.5 ms. See QE_THREADS_AUTO
 * in qe.h for the measurement and `make pool` to reproduce it.
 *
 * The core that is left out is only an efficient one with QE_PIN=1, which is what
 * makes the ordering of the cpu list mean anything: without it the pool passes no
 * list to pthread_setaffinity, the scheduler places the workers, and "thread 19
 * does not exist" leaves a core free in whichever way the scheduler feels like.
 * `bin/bench_alu` measures the two classes at the same bytes (512 MiB Q8_0 span,
 * pinned to the named cpus): 8 performance cores stream 59.3 GB/s through the
 * GEMV and 12 efficient cores 60.8, so neither class is spare capacity, and
 * pinning changes bandwidth by less than the run-to-run spread. */
int qe_pool_recommended(void) {
    int c = qe_cpus();
    return c > 2 ? c - 1 : c;
}

/* The dispatch a generation names, and the stamp that says it is still there.
 *
 * A worker copies these fields and keeps the copy only if `dstamp` still names the
 * generation it woke up for. Reading them live - which is what this did - lets a worker
 * that was not in the dispatch it woke for run the one after it: it belongs to neither,
 * yet it adds to the completion count of the one it ran, so the master of that one
 * returns while a real participant is still inside its task. Tasks are usually a
 * caller's stack frame, so what a late worker then runs is a job three dispatches dead:
 * `bench_pool`'s varying-width loop faulted inside its own `bump` on a `hits` pointer
 * read out of a frame the master had already moved on from (twelve runs of
 * `./bin/bench_pool auto 120` on a loaded host, twelve segfaults, backtrace in
 * `qe_worker_main`). The stamp is cleared before the fields are written and set after
 * them, and checked once a copy is complete, so a copy is either whole or thrown away -
 * and throwing one away can only ever be a worker nobody is waiting for, because a
 * participant's count is exactly what let the master move on. */
typedef struct {
    qe_task_fn fn;
    void *arg;
    int nthreads;
    const qe_pool_item *items;
    int nitems;
} qe_dispatch;

static struct {
    qe_dispatch d;
    unsigned dstamp; /* 0 = retired, otherwise the generation that owns `d` */
    unsigned generation;
    int parked;
    int done;
    /* The stage graph of the dispatch in flight, with one cursor and one
     * completion count per item. `ready` is the mask of items whose slots have all
     * run, which is what a dependent item waits on: a slot that was taken and not
     * yet returned from has not run, so the two counters are not one counter. Read
     * by every worker as soon as it wakes, and it stays empty for the ordinary
     * dispatch, which is all a decode step ever issues.
     *
     * `items` points into `store`, which belongs to the pool and not to the caller:
     * a worker reads this array after the dispatch that published it has already
     * decided it was finished, so an array on the caller's stack is a frame that
     * stops existing while it is still being read. `indrain` is what makes the
     * store outlive the reading - every worker that looks at it announces itself
     * first, and the master clears `items` and then waits for the count to fall
     * before it returns, so a graph's slots cannot be rewritten over the top of a
     * worker that is still walking them. */
    const qe_pool_item *items;
    qe_pool_item store[QE_POOL_MAX_ITEMS];
    int indrain;
    int nitems;
    int cursor[QE_POOL_MAX_ITEMS];
    int icensus[QE_POOL_MAX_ITEMS];
    unsigned ready;
    pthread_mutex_t mx;
    pthread_cond_t cv;
    int n;
    int cpus[QE_MAX_THREADS];
    qe_worker w[QE_MAX_THREADS];
    volatile int shutdown;
} P;

/* One dispatch of a stage graph, run by whoever woke up to it.
 *
 * A worker takes the lowest-numbered ready slot it can see and re-reads the
 * completion mask after every one, which is what keeps the threads on the long
 * stage: an item with no slots left is not looked at, and a short item that
 * finished early is simply no longer a place to work. A worker leaves when every
 * slot of every item has been taken, which is the moment there is nothing left for
 * it here; it does not wait for the threads still inside a task, exactly as
 * qe_pool_run does not, because the master is the one that has to know when the
 * answers exist.
 *
 * A slot is taken by a compare-exchange and not a fetch-add because a worker that
 * loses the race has to be able to look at the next item instead of running a
 * slice of work it does not have.
 *
 * Joining announces the read, and the identity check at the top of each turn is
 * what ends it: the master clears `items` when the answers exist and waits for
 * the join count to fall, so a worker that is still in this loop when that
 * happens leaves at the next turn instead of walking a store the next graph has
 * already rewritten, and gets back to the worker loop in time for the dispatch
 * after it. A worker that never leaves is a worker that never reports a slice of
 * the next ordinary dispatch either, and the master of that one waits on a count
 * it is now never going to see. */
static void qe_drain_items(const qe_dispatch *d) {
    const qe_pool_item *items = d->items;
    if (!items)
        return;
    __atomic_add_fetch(&P.indrain, 1, __ATOMIC_ACQ_REL);
    const int n = d->nitems;
    for (;;) {
        /* Every way out of this loop is a break, never a return: the count that
         * keeps the store alive is dropped at the bottom, and a worker that left
         * it out leaves the master of the next graph waiting on a drainer that
         * stopped existing. */
        if (__atomic_load_n(&P.items, __ATOMIC_ACQUIRE) != items)
            break;
        const unsigned ready = __atomic_load_n(&P.ready, __ATOMIC_ACQUIRE);
        int took = 0;
        for (int i = 0; i < n; i++) {
            if (items[i].deps & ~ready)
                continue;
            int c = __atomic_load_n(&P.cursor[i], __ATOMIC_ACQUIRE);
            if (c >= items[i].nthreads)
                continue;
            if (!__atomic_compare_exchange_n(&P.cursor[i], &c, c + 1, 0, __ATOMIC_ACQ_REL,
                                             __ATOMIC_ACQUIRE))
                continue;
            items[i].fn(items[i].arg, c, items[i].nthreads);
            if (__atomic_add_fetch(&P.icensus[i], 1, __ATOMIC_ACQ_REL) == items[i].nthreads)
                __atomic_or_fetch(&P.ready, 1u << i, __ATOMIC_ACQ_REL);
            took = 1;
            break;
        }
        if (took)
            continue;
        int left = 0;
        for (int i = 0; i < n; i++)
            if (P.cursor[i] < items[i].nthreads)
                left = 1;
        if (!left || P.shutdown)
            break;
        /* Nothing this worker may run is unclaimed, which means a thread is still
         * inside a stage somebody else made this one wait for. Short, bounded, and
         * never a sleep: the same reason the pool spins between dispatches. */
        QE_PAUSE();
    }
    __atomic_sub_fetch(&P.indrain, 1, __ATOMIC_ACQ_REL);
}

static void *qe_worker_main(void *v) {
    qe_worker *w = (qe_worker *)v;
    if (w->cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(w->cpu, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    unsigned seen = 0;
    for (;;) {
        long spins = 0;
        while (__atomic_load_n(&P.generation, __ATOMIC_ACQUIRE) == seen && !P.shutdown) {
            QE_PAUSE();
            if (++spins == QE_SPIN_BEFORE_YIELD)
                sched_yield();
            if (spins > QE_SPIN_BEFORE_SLEEP) {
                /* Counted in before the last look at the generation, never
                 * after it, and both sides use a full barrier. The other order
                 * - check, then announce, then wait - has a window the width of
                 * an instruction: the master publishes a job, finds nobody
                 * parked, and the worker that was between its check and its
                 * announcement sleeps through it. Every worker then ends up on
                 * the condvar and the master spins on the completion counter
                 * for ever, which is how a decode step hung with the model
                 * loaded and nothing running. */
                pthread_mutex_lock(&P.mx);
                __atomic_fetch_add(&P.parked, 1, __ATOMIC_SEQ_CST);
                while (__atomic_load_n(&P.generation, __ATOMIC_ACQUIRE) == seen && !P.shutdown)
                    pthread_cond_wait(&P.cv, &P.mx);
                __atomic_fetch_sub(&P.parked, 1, __ATOMIC_SEQ_CST);
                pthread_mutex_unlock(&P.mx);
                spins = 0;
            }
        }
        if (P.shutdown)
            return NULL; /* must NOT touch P.done: the master is not waiting */
        seen = __atomic_load_n(&P.generation, __ATOMIC_ACQUIRE);
        qe_dispatch d;
        d.fn = __atomic_load_n(&P.d.fn, __ATOMIC_RELAXED);
        d.arg = __atomic_load_n(&P.d.arg, __ATOMIC_RELAXED);
        d.nthreads = __atomic_load_n(&P.d.nthreads, __ATOMIC_RELAXED);
        d.items = __atomic_load_n(&P.d.items, __ATOMIC_RELAXED);
        d.nitems = __atomic_load_n(&P.d.nitems, __ATOMIC_RELAXED);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (__atomic_load_n(&P.dstamp, __ATOMIC_SEQ_CST) != seen)
            continue; /* the master retired this dispatch mid-copy; nobody owes it a slice */
        if (d.items)
            qe_drain_items(&d);
        else if (w->id < d.nthreads) {
            d.fn(d.arg, w->id, d.nthreads);
            /* Only workers that actually ran contribute to the count. Counting
             * idle ones too lets the master return out of qe_pool_run while a
             * working thread is still inside fn, which hands it a freed
             * stack-allocated job - the nondeterministic crash this fixes. */
            __atomic_fetch_add(&P.done, 1, __ATOMIC_RELEASE);
        }
    }
}

int qe_pool_init(int n, const int *cpus, int ncpus) {
    if (n == QE_THREADS_AUTO)
        n = qe_pool_recommended();
    if (n <= 1)
        return 0;
    if (n > QE_MAX_THREADS)
        n = QE_MAX_THREADS;
    /* QE_PIN=1 asks for the pool's own cpu list, ordered so that the low thread ids
     * get the high-capacity cores. Anything the caller passed wins: an explicit
     * list is an instruction, and QE_PIN is the default-behaviour switch. */
    int default_list[QE_MAX_THREADS];
    if ((!cpus || ncpus <= 0) && qe_env_wanted("QE_PIN")) {
        int nlist = qe_pool_pin_list(default_list, QE_MAX_THREADS);
        if (nlist >= n) {
            cpus = default_list;
            ncpus = nlist;
        }
    }
    P.n = n;
    qe_slice_table(n, cpus, ncpus);
    qe_cpu_weight_table(n, cpus, ncpus);
    pthread_mutex_init(&P.mx, NULL);
    pthread_cond_init(&P.cv, NULL);
    P.generation = 0;
    P.done = 0;
    P.shutdown = 0;
    for (int i = 1; i < n; i++) {
        /* Slot 0 belongs to the calling thread, which runs its own slice in
         * qe_pool_run. Creating a worker for it as well duplicated slice 0's
         * work and inflated the completion count by one. */
        P.w[i].id = i;
        P.w[i].cpu = (cpus && i < ncpus) ? cpus[i] : -1;
        pthread_create(&P.w[i].th, NULL, qe_worker_main, &P.w[i]);
    }
    /* Pin the calling thread to the first worker's CPU so the master does not
     * float onto a core a worker is already spinning on. */
    if (cpus && ncpus > 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpus[0], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    return n;
}

int qe_pool_size(void) {
    return P.n ? P.n : 1;
}

/* ----------------------------------------------------------- work sharing */

/*
 * Shares are equal. The host this engine is built for is a Core Ultra 7 265: eight
 * P-cores at 5300 MHz and twelve E-cores at 4600 MHz, no SMT, and the kernel's own
 * per-CPU throughput figure (cpu_capacity, 1011 against 783) says the classes are
 * 1.30x apart. Charging thread ids by that number is the obviously right thing for
 * a loop whose cost is arithmetic, and it is wrong for the loops this engine runs,
 * because a dispatch that saturates the memory fabric does not care which core
 * asked for the bytes. `bin/bench_moe` measures the same bytes through the same
 * kernels with both halves of one process, interleaved round by round (GB/s, 19
 * threads; "weights" is qe_slice with cpu_capacity, "even" is an even division of
 * the same rows):
 *
 *     IQ3_S  gate+up      13 MiB    37.5 weights   41.5 even
 *                        54 MiB    44.3            50.4
 *                       215 MiB    44.0            50.6   (ceiling 51.1)
 *     IQ4_NL down          9 MiB    59.2            61.8
 *                        35 MiB    60.3            64.1
 *                       141 MiB    60.3            68.9   (ceiling 70.0)
 *
 * The ceiling column is bench_kern's, measured on one contiguous matrix. An even
 * split reaches it and the weighted one does not, at any width, for either type:
 * weighting hands a P-core 1.30x the rows of an E-core and the dispatch then ends
 * when that P-core does, so the surplus is makespan and nothing else. The whole
 * engine agrees - equal shares against cpu_capacity weights, same binary, same
 * corpus, identical nll at every one of 255 positions: 103.4 ms a token against
 * 107.5, with the expert stage going 35.5 -> 32.3 ms and 0.3-0.6 ms off every other
 * GEMV stage. cpu_capacity is an ALU measurement, and an ALU measurement is not
 * what a bandwidth-bound dispatch should be divided by.
 *
 * `QE_SLICE=capacity` builds the weights anyway. It is the escape hatch for a host
 * or a stage where the arithmetic really is the limit - a pool narrower than the
 * machine's memory ceiling, where the fast cores can convert their clock into
 * bytes - and for measuring the assumption on a machine this file cannot test.
 */
static struct {
    int n;                                              /* threads with weights */
    int weighted;                                       /* 0 = equal shares */
    double cum[QE_MAX_THREADS + 1][QE_MAX_THREADS + 2]; /* cum[k][i]: start of thread i as a fraction, k running */
    char desc[96];
} S;

static double qe_cpu_capacity(int cpu) {
    char path[128];
    if (snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu) >=
        (int)sizeof path)
        return 0.0;
    FILE *f = fopen(path, "r");
    if (!f)
        return 0.0;
    double v = 0.0;
    if (fscanf(f, "%lf", &v) != 1 || !(v > 0.0))
        v = 0.0;
    fclose(f);
    return v;
}

/* The pool's own cpu list, best core first: the affinity mask this process is
 * allowed, sorted by the kernel's per-cpu throughput score with ties broken by cpu
 * number. Thread ids then mean something - 0..7 are the performance cores on a Core
 * Ultra and the thread the pool does not create is an efficient one - and the slice
 * table, which charges thread i to list[i], stops being a guess. Returns 0 when the
 * kernel publishes no scores, because an arbitrary order is not worth pinning to. */
static int qe_pool_pin_list(int *list, int max) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof cs, &cs) != 0)
        return 0;
    int n = 0;
    for (int c = 0; c < CPU_SETSIZE && n < max; c++)
        if (CPU_ISSET(c, &cs))
            list[n++] = c;
    if (n < 2)
        return 0;
    double cap[QE_MAX_THREADS];
    for (int i = 0; i < n; i++) {
        cap[i] = qe_cpu_capacity(list[i]);
        if (!(cap[i] > 0.0))
            return 0;
    }
    for (int i = 1; i < n; i++) { /* twenty entries: insertion sort */
        int cpu = list[i];
        double c = cap[i];
        int k = i - 1;
        while (k >= 0 && (cap[k] < c || (cap[k] == c && list[k] > cpu))) {
            list[k + 1] = list[k];
            cap[k + 1] = cap[k];
            k--;
        }
        list[k + 1] = cpu;
        cap[k + 1] = c;
    }
    return n;
}

/* Called from qe_pool_init with the pool's size and pin list. Runs once, before
 * any worker exists, so nothing here needs to be race-free. */
static void qe_slice_table(int n, const int *cpus, int ncpus) {
    const char *mode = getenv("QE_SLICE");
    S.n = 0;
    S.weighted = 0;
    snprintf(S.desc, sizeof S.desc, "equal shares");
    if (!mode || strcmp(mode, "capacity")) {
        if (mode)
            snprintf(S.desc, sizeof S.desc, "equal shares (QE_SLICE=%s)", mode);
        return;
    }

    /* Thread i is charged to list[i]. With an explicit pin list that is exact;
     * otherwise take the affinity mask in ascending CPU order, which is the
     * order a worker would be placed in anyway. */
    int list[QE_MAX_THREADS], nlist = 0;
    if (cpus && ncpus > 0) {
        for (int i = 0; i < n && i < ncpus; i++)
            list[nlist++] = cpus[i];
    } else {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        if (sched_getaffinity(0, sizeof cs, &cs) == 0)
            for (int c = 0; c < CPU_SETSIZE && nlist < QE_MAX_THREADS; c++)
                if (CPU_ISSET(c, &cs))
                    list[nlist++] = c;
    }
    if (nlist < 2)
        return;
    int ncpu = n < nlist ? n : nlist;

    double w[QE_MAX_THREADS], run[QE_MAX_THREADS + 1], cap[QE_MAX_THREADS];
    double total = 0.0;
    for (int i = 0; i < ncpu; i++) {
        cap[i] = w[i] = qe_cpu_capacity(list[i]);
        if (!(w[i] > 0.0)) {
            snprintf(S.desc, sizeof S.desc, "equal shares (no cpu_capacity)");
            return; /* partial table is worse than none */
        }
        total += w[i];
    }
    int same = 1;
    for (int i = 1; i < ncpu; i++)
        if (cap[i] != cap[0])
            same = 0;
    if (same)
        return; /* homogeneous part: the split is already right */

    /* Mean weight 1, so a weighted range is comparable to the equal one and the
     * summary reads as a multiplier rather than an absolute score. */
    run[0] = 0.0;
    for (int i = 0; i < ncpu; i++) {
        w[i] *= (double)ncpu / total;
        run[i + 1] = run[i] + w[i];
    }
    for (int k = 1; k <= ncpu; k++)
        for (int i = 0; i <= k; i++)
            S.cum[k][i] = run[i] / run[k];
    S.n = ncpu;
    S.weighted = 1;

    /* "8x1.29 + 12x1.00 (cpu_capacity)" - the groups a reader can check. */
    S.desc[0] = 0;
    for (int i = 0; i < ncpu;) {
        int j = i + 1;
        while (j < ncpu && cap[j] == cap[i])
            j++;
        char grp[48];
        snprintf(grp, sizeof grp, "%s%dx%.2f", S.desc[0] ? " + " : "", j - i, w[i]);
        strncat(S.desc, grp, sizeof S.desc - strlen(S.desc) - 1);
        i = j;
    }
    strncat(S.desc, " (cpu_capacity)", sizeof S.desc - strlen(S.desc) - 1);
}

/* Per-thread relative throughput, for the one stage on this host whose cost is
 * arithmetic rather than bytes: the routed expert walk, which reaches ~40 GB/s of the
 * 76-79 this machine reads and therefore ends when the arithmetic is done. The scores
 * are the ones qe_slice_table above charges its own weights to - the kernel's
 * cpu_capacity, 1011 against 783 here - read whatever QE_SLICE says, because that
 * switch governs the *byte-bound* divisions and this is not one of them. The engine-
 * wide experiment in the comment above that table is the reason this is per-stage and
 * not the same switch: charging every dispatch by cpu_capacity took the expert stage
 * from 35.5 to 32.3 ms and put 0.3-0.6 ms on each of the others.
 *
 * Thread i is charged to the cpu the pool would place it on, which is exact with a pin
 * list and an assumption without one (the same assumption qe_slice makes). Returns 0
 * when the kernel publishes no scores, when the thread is off the end of the list, or
 * when the caller asks about more threads than this pool was built with - and a caller
 * that gets 0 divides the work the way it would have without this function. */
static struct {
    int n;
    double cap[QE_MAX_THREADS];
} W;

double qe_cpu_weight(int tid, int nthreads) {
    if (!W.n || tid < 0 || nthreads <= 0 || nthreads > W.n || tid >= nthreads)
        return 0.0;
    double mean = 0;
    for (int i = 0; i < nthreads; i++)
        mean += W.cap[i];
    if (!(mean > 0.0))
        return 0.0;
    return W.cap[tid] * (double)nthreads / mean;
}

/* Called with the pool's size and pin list, before any worker exists. */
static void qe_cpu_weight_table(int n, const int *cpus, int ncpus) {
    W.n = 0;
    int list[QE_MAX_THREADS];
    int m = 0;
    if (cpus && ncpus > 0) {
        for (int i = 0; i < n && i < ncpus && m < QE_MAX_THREADS; i++)
            list[m++] = cpus[i];
    } else {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        if (sched_getaffinity(0, sizeof cs, &cs) != 0)
            return;
        for (int c = 0; c < CPU_SETSIZE && m < QE_MAX_THREADS; c++)
            if (CPU_ISSET(c, &cs))
                list[m++] = c;
    }
    if (m < 2)
        return;
    double sum = 0;
    for (int i = 0; i < m; i++) {
        W.cap[i] = qe_cpu_capacity(list[i]);
        if (!(W.cap[i] > 0.0))
            return; /* no scores here: every share stays equal */
        sum += W.cap[i];
    }
    for (int i = 0; i < m; i++)
        W.cap[i] /= sum / (double)m;
    W.n = m;
}

void qe_slice(long n, int tid, int nthreads, long *lo, long *hi) {
    if (nthreads <= 1) {
        *lo = 0;
        *hi = n;
        return;
    }
    long a, b;
    if (S.weighted && nthreads <= S.n) {
        const double *c = S.cum[nthreads];
        a = (long)(n * c[tid]);
        b = (long)(n * c[tid + 1]);
    } else {
        a = n * tid / nthreads;
        b = n * (tid + 1) / nthreads;
    }
    /* Rounding has to stay monotone and exhaustive; pinning the ends is cheaper
     * than arguing about it, and a one-row difference at the boundary is well
     * inside the error of the weights themselves. */
    if (tid == 0)
        a = 0;
    if (tid == nthreads - 1)
        b = n;
    if (b > n)
        b = n;
    if (b < a)
        b = a;
    *lo = a;
    *hi = b;
}

const char *qe_slice_summary(char *buf, size_t buflen) {
    snprintf(buf, buflen, "%s", S.weighted ? S.desc : "equal shares");
    return buf;
}

void qe_pool_run(qe_task_fn fn, void *arg, int nthreads) {
    if (P.n <= 1 || nthreads <= 1) {
        fn(arg, 0, 1);
        return;
    }
    if (nthreads > P.n)
        nthreads = P.n;
    /* Retire the dispatch before writing over it, so a worker copying it fails its
     * stamp check rather than reading half of one dispatch and half of the next. */
    __atomic_store_n(&P.dstamp, 0u, __ATOMIC_SEQ_CST);
    P.d.fn = fn;
    P.d.arg = arg;
    P.d.nthreads = nthreads;
    P.d.items = NULL;
    P.d.nitems = 0;
    P.done = 0;
    __atomic_store_n(&P.dstamp, P.generation + 1, __ATOMIC_SEQ_CST);
    __atomic_add_fetch(&P.generation, 1, __ATOMIC_RELEASE);
    /* SEQ_CST against the workers' SEQ_CST registration: a worker that counted
     * itself in after this load is a worker whose next look at the generation
     * has to see the job published above, and a worker that counted itself in
     * before it is one this load cannot miss. Anything weaker and the store to
     * the count and the load of the generation can be seen in either order. */
    if (__atomic_load_n(&P.parked, __ATOMIC_SEQ_CST)) {
        pthread_mutex_lock(&P.mx);
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.mx);
    }
    /* The master runs its own slice, then waits on the completion counter
     * rather than a barrier: a barrier deadlocks against a worker that exited
     * early during shutdown, which is exactly how the first version of this
     * pool hung on exit. */
    fn(arg, 0, nthreads);
    while (__atomic_load_n(&P.done, __ATOMIC_ACQUIRE) < nthreads - 1) {
        if (P.shutdown)
            return;
        QE_PAUSE();
    }
}

void qe_pool_run_items(const qe_pool_item *items, int nitems) {
    if (!items || nitems <= 0)
        return;
    if (nitems > QE_POOL_MAX_ITEMS) {
        /* Dropping the tail of a stage graph would drop its arithmetic, which is
         * a wrong answer forty-eight layers later and not a smaller pass. */
        fprintf(stderr, "qe: a stage graph of %d items does not fit in %d\n", nitems,
                QE_POOL_MAX_ITEMS);
        abort();
    }
    if (P.n <= 1) {
        /* One thread, so the graph is only an ordering: run whatever is ready
         * until everything has been. The same nthreads a wide dispatch would have
         * handed each task is what the tasks key their scratch by, so 1 here means
         * a task that sums partials sums exactly the one partial there is. */
        unsigned done_mask = 0;
        for (int spun = 0; spun < nitems; spun++)
            for (int i = 0; i < nitems; i++) {
                if ((done_mask >> i) & 1u)
                    continue;
                if (items[i].deps & ~done_mask)
                    continue;
                items[i].fn(items[i].arg, 0, 1);
                done_mask |= 1u << i;
            }
        return;
    }
    /* Clamped into storage the pool owns: an item asking for more threads than the
     * pool has would leave slots nobody could ever take, and the wait below has no
     * timeout for a stage that never finished. Workers read these entries after
     * this frame has stopped meaning anything, which is the whole reason they are
     * copied rather than pointed at. */
    for (int i = 0; i < nitems; i++) {
        P.store[i] = items[i];
        if (P.store[i].nthreads < 1)
            P.store[i].nthreads = 1;
        if (P.store[i].nthreads > P.n)
            P.store[i].nthreads = P.n;
    }
    for (int i = 0; i < nitems; i++)
        P.cursor[i] = P.icensus[i] = 0;
    P.ready = 0;
    __atomic_store_n(&P.dstamp, 0u, __ATOMIC_SEQ_CST);
    P.d.fn = NULL;
    P.d.arg = NULL;
    P.d.nthreads = 0;
    P.d.items = P.store;
    P.d.nitems = nitems;
    __atomic_store_n(&P.items, P.store, __ATOMIC_RELEASE);
    __atomic_store_n(&P.dstamp, P.generation + 1, __ATOMIC_SEQ_CST);
    __atomic_add_fetch(&P.generation, 1, __ATOMIC_RELEASE);
    if (__atomic_load_n(&P.parked, __ATOMIC_SEQ_CST)) {
        pthread_mutex_lock(&P.mx);
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.mx);
    }
    /* The master is a worker of this dispatch too, and the first: an item nobody
     * else took gets run before it starts waiting for anybody. It reads its own
     * snapshot, which it is the only writer of until the next publish. */
    qe_drain_items(&P.d);
    const unsigned want = nitems == 32 ? ~0u : (1u << nitems) - 1u;
    while ((__atomic_load_n(&P.ready, __ATOMIC_ACQUIRE) & want) != want) {
        if (P.shutdown)
            return;
        QE_PAUSE();
    }
    /* After the last answer, never before it: a worker still inside a task reads
     * this array, and a worker that wakes after this finds the ordinary path. */
    __atomic_store_n(&P.items, NULL, __ATOMIC_RELEASE);
    /* And wait for the ones that had already started reading it. They are spinning,
     * not working - every slot returned is what ended the wait above - so this is
     * a few turns of the pause loop, and without it the next graph would be built
     * on top of the array one of them is still holding. */
    while (__atomic_load_n(&P.indrain, __ATOMIC_ACQUIRE)) {
        if (P.shutdown)
            return;
        QE_PAUSE();
    }
}

void qe_pool_shutdown(void) {
    if (!P.n)
        return;
    P.shutdown = 1;
    __atomic_add_fetch(&P.generation, 1, __ATOMIC_RELEASE);
    pthread_mutex_lock(&P.mx);
    pthread_cond_broadcast(&P.cv);
    pthread_mutex_unlock(&P.mx);
    for (int i = 1; i < P.n; i++)
        pthread_join(P.w[i].th, NULL);
    P.n = 0;
}

/* ------------------------------------------------------------------- memory */

/* hugetlb (2 MiB, and 1 GiB where available) via MAP_HUGETLB. This is not a
 * micro-optimisation: the expert pools are tens of GiB, so at 4 KiB pages the
 * working set needs ~10M page-table entries and every random expert access
 * costs multiple page walks. With 2 MiB pages the same range needs ~20K.
 * Falls back to THP-by-madvise, then to plain pages. */
void *qe_alloc_huge(size_t n) {
    n = qe_align_up(n, 2u << 20);
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED)
        return p;
    p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        return NULL;
#ifdef MADV_HUGEPAGE
    madvise(p, n, MADV_HUGEPAGE);
#endif
    return p;
}

void qe_free_huge(void *p, size_t n) {
    if (p && p != MAP_FAILED)
        munmap(p, qe_align_up(n, 2u << 20));
}

/* --------------------------------------------------- staging arena backings
 *
 * Why this exists at all: the checkpoint is on ZFS and this pool runs a 1 GiB
 * ARC against an 87 GiB file, so the page cache holds about 1% of it. A
 * demand-paged mapping of the weights is therefore not "slow until warm" - it
 * is a raidz2 stripe seek per expert row, forever. The arena below is the
 * opposite bet: read every byte once, sequentially, and never go to the disk
 * again while serving.
 *
 * Two backings, and the difference between them is 22 million page faults plus
 * a page walk per random access versus 23 thousand of each:
 *   - hugetlb 2 MiB pages, which need a reserved pool (root, once per boot),
 *     and can be a *file* on hugetlbfs so the 3 minute read survives a restart;
 *   - plain anonymous pages, which need nothing but pay the per-4K cost.
 */

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

#define QE_HUGE_2M (2u << 20)

size_t qe_huge_free_mib(size_t page_mib) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/kernel/mm/hugepages/hugepages-%zukB/free_hugepages",
             page_mib * 1024);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    unsigned long long n = 0;
    if (fscanf(f, "%llu", &n) != 1)
        n = 0;
    fclose(f);
    return (size_t)(n * page_mib);
}

size_t qe_mem_available_mib(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return 0;
    char key[64];
    unsigned long long kb;
    size_t avail = 0;
    /* The leading space in the format is not cosmetic: without it the key
     * captures the previous line's newline and every strcmp fails, which reads
     * as "0 MiB available". */
    while (fscanf(f, " %63[^:]: %llu kB", key, &kb) == 2) {
        if (!strcmp(key, "MemAvailable")) {
            avail = (size_t)(kb >> 10);
            break;
        }
    }
    fclose(f);
    return avail;
}

void *qe_alloc_backed(size_t n, unsigned flags, const char **how) {
    n = qe_align_up(n, QE_HUGE_2M);
    if (flags & QE_STAGE_HUGE) {
        /* Needs the pool reserved and either CAP_IPC_LOCK or membership of
         * vm.hugetlb_shm_group; both are one-time root settings. */
        void *p = mmap(NULL, n, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            if (how)
                *how = "hugetlb";
            return p;
        }
        /* Refused: no pool, or no permission to use it. Report the backing that
         * was actually got, never the one that was asked for. */
    }
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        return NULL;
    if (flags & QE_STAGE_THP) {
        /* Asks for 2 MiB pages without reserving anything: the kernel takes the
         * huge ones where it can get them and falls back to 4 KiB where memory
         * is fragmented. Nothing here needs root and nothing is taken away from
         * the page cache for the rest of the machine's life. */
#ifdef MADV_HUGEPAGE
        madvise(p, n, MADV_HUGEPAGE);
#endif
        if (how)
            *how = "ram+thp";
        return p;
    }
    if (how)
        *how = "ram";
    return p;
}

void *qe_alloc_arena_file(const char *dir, const char *name, size_t n, int *fresh) {
    if (!dir || !*dir || !name || !*name)
        return NULL;
    n = qe_align_up(n, QE_HUGE_2M);
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);

    struct stat st;
    int existed = (stat(path, &st) == 0);
    if (existed && (size_t)st.st_size != n) {
        /* A leftover arena from a different checkpoint or a different volume
         * size cannot be validated, so it cannot be reused. */
        if (unlink(path) != 0)
            return NULL;
        existed = 0;
    }
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0)
        return NULL;
    if (!existed && ftruncate(fd, (off_t)n) != 0) {
        int e = errno;
        close(fd);
        unlink(path);
        errno = e;
        return NULL;
    }
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED)
        return NULL;
    if (fresh)
        *fresh = !existed;
    return p;
}

int qe_lock_resident(void *p, size_t n) {
    if (mlock(p, n) == 0)
        return 0;
    return -errno;
}

void qe_madvise_seq(const void *p, size_t n) {
#ifdef MADV_SEQUENTIAL
    madvise((void *)p, n, MADV_SEQUENTIAL);
#else
    (void)p;
    (void)n;
#endif
}

void qe_prefault(void *p, size_t n, int nthreads) {
    (void)nthreads;
    volatile char *c = (volatile char *)p;
    long ps = qe_page_size();
    for (size_t o = 0; o < n; o += (size_t)ps)
        (void)c[o];
}

/* ------------------------------------------------------------- bandwidth */

typedef struct {
    const volatile char *p;
    size_t n;
    long total;
    long counts[QE_MAX_THREADS];
} bw_job;

static void bw_task(void *v, int tid, int nt) {
    bw_job *j = (bw_job *)v;
    size_t chunk = (j->n / nt) & ~(size_t)4095;
    size_t lo = (size_t)tid * chunk;
    size_t hi = (tid == nt - 1) ? j->n : lo + chunk;
    if (lo >= hi)
        return;
    /* Read as 64B cache lines with 8 streams so the hardware prefetchers have
     * something to chew on and the loop cannot be optimised away. */
    const volatile char *b = j->p + lo;
    size_t len = hi - lo;
    long acc = 0;
    size_t i = 0;
    for (; i + 512 <= len; i += 512) {
        long a0 = b[i + 0], a1 = b[i + 64], a2 = b[i + 128], a3 = b[i + 192];
        long a4 = b[i + 256], a5 = b[i + 320], a6 = b[i + 384], a7 = b[i + 448];
        acc += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    for (; i < len; i += 64)
        acc += b[i];
    j->counts[tid] = acc;
}

double qe_measure_read_bw(const void *p, size_t n, int nthreads) {
    bw_job j;
    memset(&j, 0, sizeof(j));
    j.p = (const volatile char *)p;
    j.n = n;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    qe_pool_run(bw_task, &j, nthreads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (s <= 0)
        return 0;
    return (double)n / s / 1e9;
}
