/* Chunked prefill: what the CPU actually touches per ingested token.

   Decoding one token at a time means every projection weight is read once per
   token, and there is nothing left to say about that. Ingestion is a different
   problem: a step carrying T tokens needs each weight row T times, and a weight
   byte can only be paid for once if all T uses of it happen while it is still
   close. qe_gemv_multi_t is that loop order - weight rows blocked to fit in L2,
   and inside a block one pass per token.

   Three things make a number here worth reading.

   The round is shaped like a step. One measurement visits every dense tensor of
   a layer once, in order, because a matrix measured on its own in a tight loop
   simply lives in the 30 MiB L3 and reports zero DRAM bytes for both paths. A
   round moves 566 MiB, so nothing survives from one round to the next, which is
   the situation a real step is in.

   Bytes come from the hardware, not from the loop: the last-level-cache load
   miss count, armed on every thread that will do work, tested against a known
   256 MiB walk before any of it is trusted, and then tested again against the
   dense round, whose byte count is known from the tensor table. A counter that
   reads zero reports a marvellous optimisation; a counter that counts only some
   of the threads reports a plausible one, and only the second check sees it.
   Counting is therefore done with the pool pinned to the performance cores:
   this package's efficient cores have a different monitoring unit, and a
   per-thread event is simply not counted while its thread runs there.

   The numerics are checked before any byte figure means anything: the blocked
   order must produce bit-identical results to T separate single-vector calls,
   except where the dispatcher deliberately shares one dequantisation across the
   chunk (Q8_0 at T >= 2), which is a different summation by construction and is
   therefore held to the error it makes against the exact dot product instead of
   against the other rounded path: 1e-3 of sum|x*w|, and no more than an order of
   magnitude worse than the single-token path. A chunking change that quietly
   moved a rounding boundary is the worst kind of optimisation, and it would not
   fail any test that only looks at loss; the answer to that is to name which
   boundary is allowed to move and to measure how far it moved - against the size
   of the results, since a dot product that cancels to almost nothing will show a
   few percent for a difference that is a single rounding unit.
*/
#define _GNU_SOURCE
#include "qe.h"

#include <errno.h>
#include <linux/perf_event.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAXT 17
#define MAX_TH 32
#define MAX_E 16
#define ROW_BLOCK (1u << 20) /* half a 2 MiB L2: a block has to share it with the
                              * T activation rows and their outputs */

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* fp16 by the IEEE-754 definition rather than by this tree's converter. A
 * reference that inherits the kernel's idea of what a block scale is cannot
 * catch a kernel whose idea is wrong: bench_kern has that rule for a reason,
 * and every scale in every quantised kernel passes through it. */
static long double h2d(uint16_t h) {
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    long double v = e == 0u ? ldexpl((long double)m, -24)
                            : ldexpl(1.0L + (long double)m / 1024.0L, (int)e - 15);
    return h & 0x8000u ? -v : v;
}

/* Row `row` of a Q8_0 matrix, exactly: the dot product accumulated in long
 * double, and with it sum|x*w|, the quantity a float32 accumulation of that dot
 * is entitled to be wrong by. Reporting the two together is the only way to tell
 * a moved rounding boundary from a wrong row: the error is meaningless until it
 * is divided by the cancellation it survived. */
static void q8_row_exact(const void *W, long row, long ldd, const float *x, long K,
                         long double *dot, long double *cond) {
    const qe_block_q8_0 *w =
        (const qe_block_q8_0 *)((const uint8_t *)W + (size_t)row * (size_t)ldd * 34u);
    long double s = 0, c = 0;
    for (long b = 0; b < K / 32; b++) {
        long double d = h2d(w[b].d.d);
        for (int i = 0; i < 32; i++) {
            long double t = (long double)w[b].qs[i] * d * (long double)x[b * 32 + i];
            s += t;
            c += fabsl(t);
        }
    }
    *dot = s;
    *cond = c;
}

/* The checkpoint's dense projections: what a step walks for every token. */
static const struct {
    const char *name;
    const char *why;
} TENSORS[] = {
    { "blk.0.attn_qkv.weight", "widest projection of a layer" },
    { "blk.0.attn_gate.weight", "gated projection" },
    { "blk.0.ssm_out.weight", "output projection" },
    { "blk.0.ffn_gate_shexp.weight", "shared expert, read by every token anyway" },
    { "blk.0.hc_attn_up.weight", "hyper-connection" },
    { "blk.0.ffn_gate_inp.weight", "router: 512 rows, whole, every token" },
    { "output.weight", "lm_head: 497 MiB per token on its own" },
};

/* ---- one tensor of the round ------------------------------------------- */
typedef struct {
    const char *name;
    qe_type type;
    const void *W;
    float *y; /* MAXT rows x n_rows, per tensor */
    long rows, K, ldd;
    size_t rb;
    double mib;
} entry;

static entry LIST[MAX_E];
static int NL;
static float *X; /* MAXT activations, used with each tensor's own K as stride */

/* ---- last level cache misses, counted per thread ---------------------- */

static struct ctr_spec {
    int type;
    unsigned long long config;
    const char *what;
} CTR, CTR2;
static int CTR_OK;
static int CTR2_OK;
static int CTR_ARM_FAIL;
static int CTR_FD[MAX_TH], CTR_FD2[MAX_TH];
static unsigned long long CTR_N[MAX_TH];
static int CTR_SILENT; /* threads whose counter came back empty */

static int ctr_open(const struct ctr_spec *s) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = (unsigned)s->type;
    a.config = s->config;
    a.disabled = 1;
    /* User-space only is what perf_event_paranoid=2 will hand out, and it is
     * also the honest scope: kernel entries are not this loop's traffic. */
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    return (int)syscall(__NR_perf_event_open, &a, 0 /* this thread */, -1, -1, 0);
}

/* "event=0x2e,umask=0x41" as the kernel exports it per PMU. */
static int ctr_from_sysfs(const char *pmu, const char *ev, struct ctr_spec *out) {
    char path[512];
    unsigned long long type = 0, umask = 0, config = 0;
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/%s/type", pmu);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    if (fscanf(f, "%llu", &type) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/%s/events/%s", pmu, ev);
    f = fopen(path, "r");
    if (!f)
        return 0;
    char buf[256];
    if (!fgets(buf, sizeof buf, f)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    const char *p = strstr(buf, "umask=0x");
    if (p)
        umask = strtoull(p + 6, NULL, 16);
    p = strstr(buf, "event=0x");
    if (p)
        config = strtoull(p + 6, NULL, 16);
    if (!config)
        return 0;
    out->type = (int)type;
    out->config = config | (umask << 8);
    out->what = "vendor LLC-miss event";
    return 1;
}

/* A hybrid package has two monitoring units, cpu_core (type 4) and cpu_atom
 * (type 8), and a task-bound event on one of them is not scheduled while its
 * thread runs on the other. Arming both on every thread and adding the two reads
 * counts every thread whichever core it happens to be on, and the two are
 * disjoint by construction - same event number on different PMUs - so nothing is
 * counted twice. Without this the byte columns only exist where they can be
 * counted (the performance cores) and a sixteen-thread run, which is the width
 * the engine actually serves at, silently loses nine threads. */
static int ctr_pair(struct ctr_spec *a, struct ctr_spec *b) {
    if (!ctr_from_sysfs("cpu_core", "cache-misses", a) ||
        !ctr_from_sysfs("cpu_atom", "cache-misses", b))
        return 0;
    int fa = ctr_open(a), fb = ctr_open(b);
    if (fa < 0 || fb < 0) {
        if (fa >= 0)
            close(fa);
        if (fb >= 0)
            close(fb);
        return 0;
    }
    close(fa);
    close(fb);
    a->what = "cpu_core LLC miss";
    b->what = "cpu_atom LLC miss";
    return 1;
}

static void ctr_pick(char *why, size_t n) {
    struct ctr_spec core, atom;
    if (ctr_pair(&core, &atom)) {
        CTR = core;
        CTR2 = atom;
        CTR_OK = CTR2_OK = 1;
        snprintf(why, n, "the LLC-miss event on both PMUs (cpu_core type %d + cpu_atom type %d)",
                 core.type, atom.type);
        return;
    }
    struct ctr_spec cand[6];
    int nc = 0;
    cand[nc].type = PERF_TYPE_HW_CACHE;
    cand[nc].config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    cand[nc].what = "generic last-level-cache read miss";
    nc++;
    const char *pmus[] = { "cpu_core", "cpu" };
    const char *evs[] = { "ll_cache_miss_rd", "cache_misses_llc", "cache-misses" };
    for (unsigned p = 0; p < sizeof pmus / sizeof pmus[0]; p++)
        for (unsigned e = 0; e < sizeof evs / sizeof evs[0] && nc < 6; e++)
            if (ctr_from_sysfs(pmus[p], evs[e], &cand[nc])) {
                static char names[3][64];
                snprintf(names[nc % 3], sizeof names[0], "%s %s", pmus[p], evs[e]);
                cand[nc].what = names[nc % 3];
                nc++;
            }
    snprintf(why, n, "none");
    for (int i = 0; i < nc; i++) {
        int fd = ctr_open(&cand[i]);
        if (fd < 0)
            continue;
        CTR = cand[i];
        close(fd);
        snprintf(why, n, "%s", cand[i].what);
        CTR_OK = 1;
        return;
    }
    /* Opened but never validated is not a measurement, so the loop below keeps
     * the distinction: without a counter it prints computed figures and says so. */
    CTR_OK = 0;
}

/* ---- the pool tasks: arm, walk a round, harvest ------------------------ */

typedef struct {
    int T, chunked;
} round_job;

static void task_arm(void *arg, int tid, int nthreads) {
    (void)arg;
    (void)nthreads;
    CTR_FD[tid] = CTR_OK ? ctr_open(&CTR) : -1;
    CTR_FD2[tid] = CTR2_OK ? ctr_open(&CTR2) : -1;
    if ((CTR_OK && CTR_FD[tid] < 0) || (CTR2_OK && CTR_FD2[tid] < 0))
        if (!CTR_ARM_FAIL)
            CTR_ARM_FAIL = errno;
    for (int f = 0; f < 2; f++) {
        int fd = f == 0 ? CTR_FD[tid] : CTR_FD2[tid];
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    CTR_N[tid] = 0;
}

/* One pass over every tensor of the round, this thread's slice of each of their
 * rows. Threads stay together on a tensor, exactly as the engine's row split
 * does, so the sharing of the L3 between them is the sharing a step sees. */
static void task_round(void *arg, int tid, int nthreads) {
    round_job *j = (round_job *)arg;
    for (int i = 0; i < NL; i++) {
        const entry *e = &LIST[i];
        long r0 = e->rows * (long)tid / nthreads;
        long rn = e->rows * (long)(tid + 1) / nthreads - r0;
        if (rn <= 0)
            continue;
        const uint8_t *W = (const uint8_t *)e->W + (size_t)r0 * e->rb;
        float *y = e->y + r0;
        if (j->chunked) {
            qe_gemv_multi_t(e->type, W, X, y, rn, e->K, e->ldd, j->T, e->K, e->rows, e->rb,
                            ROW_BLOCK);
        } else {
            /* One full walk per token: what a step carrying one token does. */
            for (int t = 0; t < j->T; t++)
                qe_gemv_t(e->type, W, X + (size_t)t * e->K, y + (size_t)t * e->rows, rn, e->K,
                          e->ldd);
        }
    }
}

static void task_harvest(void *arg, int tid, int nthreads) {
    (void)arg;
    (void)nthreads;
    /* A counter with no read_format hands back exactly one u64. Asking for a
     * group here reads zero and reports an optimisation that never happened. */
    unsigned long long total = 0;
    for (int f = 0; f < 2; f++) {
        int fd = f == 0 ? CTR_FD[tid] : CTR_FD2[tid];
        if (fd < 0)
            continue;
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        unsigned long long v = 0;
        if (read(fd, &v, sizeof v) == (ssize_t)sizeof v)
            total += v;
        close(fd);
        if (f == 0)
            CTR_FD[tid] = -1;
        else
            CTR_FD2[tid] = -1;
    }
    CTR_N[tid] = total;
}

/* Fastest of `attempts` rounds, together with the misses of that same attempt:
 * a slow round's time next to a fast round's bytes would be a fabrication. */
static double timed_round(int T, int chunked, int nthr, int attempts, unsigned long long *misses) {
    round_job j = { T, chunked };
    double best = 1e30;
    unsigned long long best_miss = 0;
    for (int a = 0; a < attempts; a++) {
        if (CTR_OK)
            qe_pool_run(task_arm, NULL, nthr);
        double t0 = now();
        qe_pool_run(task_round, &j, nthr);
        double dt = now() - t0;
        unsigned long long sum = 0;
        if (CTR_OK) {
            qe_pool_run(task_harvest, NULL, nthr);
            CTR_SILENT = 0;
            for (int i = 0; i < nthr && i < MAX_TH; i++) {
                sum += CTR_N[i];
                if (!CTR_N[i])
                    CTR_SILENT++;
            }
        }
        if (dt < best) {
            best = dt;
            best_miss = sum;
        }
    }
    *misses = best_miss;
    return best;
}

/* The pool is pinned to the cores the kernel reports at the package's maximum
 * frequency, which on this Arrow Lake part are the eight performance cores
 * (cpu0-7 at 5300000 kHz; cpu8-19 are the twelve efficient cores at 4600000).
 * Two reasons, both about honesty rather than speed: the per-thread LLC event is
 * not counted at all while its thread runs on an efficient core - unpinned, a
 * first run of this bench counted 10% of a round it independently knows the size
 * of - and a rate that mixes 5.3 GHz and 4.6 GHz cores is not a rate.
 *
 * Both sysfs names are tried. The file lives under cpuN/cpufreq/; reading it
 * from cpuN/ instead returns nothing, which this function used to report as a
 * silent zero, and an unpinned run is indistinguishable from a working one
 * everywhere except in the numbers it prints. */
#define MAX_CPUS 256
static int pin_performance_cores(int *cpus, int max) {
    static const char *FMT[] = { "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq",
                                 "/sys/devices/system/cpu/cpu%d/cpuinfo_max_freq" };
    static int freq[MAX_CPUS];
    int n = 0, best = 0;
    for (int c = 0; c < MAX_CPUS; c++) {
        freq[c] = 0;
        for (unsigned f = 0; f < sizeof FMT / sizeof FMT[0] && !freq[c]; f++) {
            char path[160];
            snprintf(path, sizeof path, FMT[f], c);
            FILE *h = fopen(path, "r");
            if (!h)
                continue;
            if (fscanf(h, "%d", &freq[c]) != 1)
                freq[c] = 0;
            fclose(h);
        }
        if (freq[c] > best)
            best = freq[c];
    }
    for (int c = 0; c < MAX_CPUS && n < max; c++)
        if (freq[c] && freq[c] == best)
            cpus[n++] = c;
    return n;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : NULL;
    int max_T = argc > 2 ? atoi(argv[2]) : 16;
    int nthr = argc > 3 ? atoi(argv[3]) : 0;
    if (!path) {
        fprintf(stderr, "usage: %s model.gguf [max_T] [threads]\n", argv[0]);
        return 2;
    }
    if (max_T > MAXT - 1)
        max_T = MAXT - 1;

    char ctr_what[128];
    ctr_pick(ctr_what, sizeof ctr_what);

    printf("== open ==\n");
    fflush(stdout);
    qe_model *m = qe_open(path, QE_M_NONE);
    if (!m) {
        fprintf(stderr, "qe_open: %s\n", qe_strerror());
        return 1;
    }
    printf("  weights %s (%.1f GiB), %s\n", qe_model_stage_method(m),
           (double)qe_model_stage_bytes(m) / 1073741824.0,
           qe_model_stage_pinned(m) ? "pinned" : "reclaimable");

    /* The guard against a counter that exists and lies: a 256 MiB walk cannot
     * miss a 30 MiB cache zero times. */
    if (CTR_OK) {
        char *b = (char *)aligned_alloc(64, 256u << 20);
        if (b) {
            memset(b, 1, 256u << 20);
            int fd[2] = { ctr_open(&CTR), CTR2_OK ? ctr_open(&CTR2) : -1 };
            unsigned long long seen = 0;
            for (int f = 0; f < 2; f++)
                if (fd[f] >= 0) {
                    ioctl(fd[f], PERF_EVENT_IOC_RESET, 0);
                    ioctl(fd[f], PERF_EVENT_IOC_ENABLE, 0);
                }
            volatile char s = 0;
            for (size_t i = 0; i < (256u << 20); i += 64)
                s += b[i];
            (void)s;
            for (int f = 0; f < 2; f++)
                if (fd[f] >= 0) {
                    ioctl(fd[f], PERF_EVENT_IOC_DISABLE, 0);
                    unsigned long long v = 0;
                    if (read(fd[f], &v, sizeof v) == (ssize_t)sizeof v)
                        seen += v;
                    close(fd[f]);
                }
            if (seen < (4u << 20) / 64) {
                printf("  %s counted %llu misses for a 256 MiB walk: unusable, byte figures"
                       " below are computed, not counted\n",
                       ctr_what, seen);
                CTR_OK = 0;
            } else {
                printf("  bytes counted with %s (%llu misses for a 256 MiB calibration walk)\n",
                       ctr_what, seen);
            }
            free(b);
        }
    } else {
        printf("  no LLC-miss counter available: byte figures below are computed, not counted\n");
    }

    int cpus[MAX_CPUS / 8];
    int ncpus = pin_performance_cores(cpus, (int)(sizeof cpus / sizeof cpus[0]));
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (nthr <= 0)
        nthr = online > 0 ? (int)online : 8;
    if (nthr > 20)
        nthr = 20;
    int pinned = ncpus > 0 && nthr <= ncpus;
    qe_pool_init(nthr, pinned ? cpus : NULL, pinned ? nthr : 0);
    if (nthr > qe_pool_size())
        nthr = qe_pool_size();
    printf("  %d threads%s, chunk = %u KiB of weight rows\n", nthr,
           pinned ? " on the performance cores" : " (unpinned)", (unsigned)(ROW_BLOCK >> 10));
    if (ncpus > 0) {
        printf("  fastest cores the kernel reports: %d of them at maximum frequency:", ncpus);
        for (int i = 0; i < ncpus; i++)
            printf(" cpu%d", cpus[i]);
        printf("%s\n", pinned ? "" : ", more of them than threads, so unpinned");
    } else {
        printf("  no per-core frequency in sysfs: the pool is unpinned, and on a hybrid package"
               " that means the counters below see only the threads that happen to run on a"
               " performance core\n");
    }
    fflush(stdout);

    const int TS[] = { 1, 2, 4, 8, 16 };
    const int NTS = (int)(sizeof TS / sizeof TS[0]);
    long K_max = 32, rows_max = 1;
    for (unsigned i = 0; i < sizeof TENSORS / sizeof TENSORS[0]; i++) {
        const qe_tensor *t = qe_tensor_by_name(m, TENSORS[i].name);
        if (!t || t->ndim != 2 || !t->ne[1])
            continue;
        if ((long)t->ne[0] > K_max)
            K_max = (long)t->ne[0];
        if ((long)t->ne[1] > rows_max)
            rows_max = (long)t->ne[1];
    }
    X = aligned_alloc(64, (size_t)MAXT * (size_t)K_max * sizeof *X);
    float *ya = aligned_alloc(64, (size_t)MAXT * (size_t)rows_max * sizeof *ya);
    float *yb = aligned_alloc(64, (size_t)MAXT * (size_t)rows_max * sizeof *yb);
    if (!X || !ya || !yb) {
        fprintf(stderr, "alloc\n");
        return 1;
    }
    /* Deterministic activations in [-1,1). The values do not matter, only that
     * both paths are handed the same ones. */
    for (long i = 0; i < (long)MAXT * K_max; i++)
        X[i] = (float)(((i * 1103515245L + 12345L) >> 7) % 1000) / 500.0f - 1.0f;

    /* ---- collect the round, checking the blocked order ------------------ */
    double round_mib = 0;
    double worst_shared = 0;
    const char *worst_shared_name = NULL;
    long shared_moved = 0, shared_elems = 0;
    double worst_exact_single = 0, worst_exact_chunked = 0;
    printf("\n== the round ==\n");
    for (unsigned i = 0; i < sizeof TENSORS / sizeof TENSORS[0] && NL < MAX_E; i++) {
        const qe_tensor *t = qe_tensor_by_name(m, TENSORS[i].name);
        if (!t || t->ndim != 2 || !t->ne[1]) {
            printf("  %s: not a 2-D tensor in this file\n", TENSORS[i].name);
            continue;
        }
        const void *W = qe_tensor_data(m, t);
        long K = (long)t->ne[0], rows = (long)t->ne[1];
        size_t rb = (size_t)(t->bytes / (uint64_t)rows);
        long bne = qe_type_block_ne(t->type);
        long ldd = bne > 0 ? K / bne : K;
        const qe_type_info *info = qe_type_info_of(t->type);
        if (!W || rb < 8) {
            printf("  %s: skipped (%s)\n", TENSORS[i].name, !W ? "not resident" : "row size");
            continue;
        }
        if (qe_gemv_t(t->type, W, X, ya, rows, K, ldd)) {
            printf("  %s: no kernel for %s\n", TENSORS[i].name, info ? info->name : "?");
            continue;
        }
        /* Every element of every row, T of them. For a type whose blocked walk is
         * still T calls of the single-token kernel the order may not move a single
         * bit, and that is asserted byte for byte. Q8_0 at T >= 2 is the exception
         * the dispatcher makes: it goes to qe_gemv_q8_0_multi, which dequantises a
         * block once for the whole chunk and therefore sums two chains per token
         * where the single-row kernel runs four per row (`src/kern.c`, and
         * include/qe.h for what it costs).
         *
         * How far those two paths may disagree is not a question the disagreement
         * between them can answer. A dot product whose terms cancel to 5e-06 is
         * owed a rounding unit like any other, so one moved result reads as 4%
         * against the number it landed on and as 4e-09 against the size of the
         * round's results, and only the second of those is a fact about the kernel.
         * The gate is therefore each path's error against the exact dot product -
         * long double, from the format's own definition - over sum|x*w|, the
         * quantity a float32 accumulation of that dot is allowed to be wrong by.
         * Two things follow. The chunked error stays under 1e-3 of that, and this
         * is where the margin lives: observed at a few times 1e-8, five orders of
         * magnitude below the bound, and no amount of reassociation reaches it,
         * while a row read from the wrong offset is off by the row. Second, its RMS
         * stays within an order of magnitude of the single-token path's. That bound
         * is deliberately loose, because a tighter one would be measuring some-
         * thing's absence rather than a bug: the shared sweep keeps two accumulator
         * chains where the single-row kernel keeps four, so it is entitled to sum a
         * little worse, and synthetic runs land at 2x with the per-element maxima
         * moving further than the RMS by luck. What the ratio does catch is a path
         * that is wrong rather than merely louder. Both errors are printed, so a
         * change here is read rather than merely passed. */
        for (int t2 = 0; t2 < MAXT - 1; t2++)
            qe_gemv_t(t->type, W, X + (size_t)t2 * K, ya + (size_t)t2 * rows, rows, K, ldd);
        qe_gemv_multi_t(t->type, W, X, yb, rows, K, ldd, MAXT - 1, K, rows, rb, ROW_BLOCK);
        long n = (long)(MAXT - 1) * rows;
        int shared = t->type == QE_Q8_0 && MAXT - 1 >= 2;
        if (!shared) {
            if (memcmp(ya, yb, (size_t)n * sizeof *ya) != 0) {
                long at = 0;
                while (at < n && ya[at] == yb[at])
                    at++;
                printf("  BLOCKED ORDER CHANGED A RESULT in %s at element %ld: %.9g vs %.9g\n",
                       TENSORS[i].name, at, ya[at], yb[at]);
                return 5;
            }
        } else {
            enum { CAND = 64 };
            long ndiff = 0, at_abs = 0, at_rel = 0;
            double worst_abs = 0, worst_rel = 0, scale = 0;
            for (long k = 0; k < n; k++) {
                double a = ya[k], ad = fabs((double)yb[k] - (double)a);
                double rel = fabs(a) > 1e-30 ? ad / fabs(a) : ad;
                if ((double)yb[k] != a)
                    ndiff++;
                if (fabs(a) > scale)
                    scale = fabs(a);
                if (ad > worst_abs)
                    worst_abs = ad, at_abs = k;
                if (rel > worst_rel)
                    worst_rel = rel, at_rel = k;
            }
            /* The movers, plus the one element the raw relative reading points at. */
            long cand[CAND];
            int ncand = 0;
            cand[ncand++] = at_rel;
            if (at_rel != at_abs)
                cand[ncand++] = at_abs;
            for (long k = 0; k < n && ncand < CAND; k++)
                if (fabs((double)yb[k] - (double)ya[k]) >= 0.5 * worst_abs)
                    cand[ncand++] = k;
            double ea_max = 0, eb_max = 0, ea_sq = 0, eb_sq = 0, rms_a = 0, rms_b = 0;
            int nref = 0;
            if (K % 32 == 0) {
                for (int c = 0; c < ncand; c++) {
                    long k = cand[c];
                    long double dot = 0, cond = 0;
                    q8_row_exact(W, k % rows, ldd, X + (k / rows) * K, K, &dot, &cond);
                    if (!(cond > 0))
                        continue;
                    double ea = fabs((double)ya[k] - (double)dot) / (double)cond;
                    double eb = fabs((double)yb[k] - (double)dot) / (double)cond;
                    if (ea > ea_max)
                        ea_max = ea;
                    if (eb > eb_max)
                        eb_max = eb;
                    ea_sq += ea * ea;
                    eb_sq += eb * eb;
                    nref++;
                }
            }
            if (nref > 0) {
                rms_a = sqrt(ea_sq / (double)nref);
                rms_b = sqrt(eb_sq / (double)nref);
            }
            double rel_scale = scale > 0 ? worst_abs / scale : worst_abs;
            int bad = !(nref > 0) || !(rel_scale <= 1e-3) || !(eb_max <= 1e-3) ||
                      !(rms_b <= 10.0 * rms_a);
            printf("  %s: shared dequant moved %ld of %ld results by at most %.3g against a\n"
                   "      scale of %.3g (%.1e of it; %.3g read against the %.9g it landed on).\n"
                   "      Exact dot product for %d of them: %.1e from it single-token, %.1e\n"
                   "      chunked (RMS %.1e vs %.1e). Gate 1e-3 of sum|x*w|, 10x the other way:"
                   " %s\n",
                   TENSORS[i].name, ndiff, n, worst_abs, scale, rel_scale, worst_rel, ya[at_rel],
                   nref, ea_max, eb_max, rms_a, rms_b, bad ? "FAIL" : "ok");
            if (bad) {
                printf("  SHARED DEQUANT CHANGED A RESULT in %s: worst move %.3g against a scale of"
                       "\n"
                       "  %.3g, exact reference says %.1e single-token and %.1e chunked of"
                       " sum|x*w|\n",
                       TENSORS[i].name, worst_abs, scale, ea_max, eb_max);
                return 5;
            }
            if (rel_scale > worst_shared) {
                worst_shared = rel_scale;
                worst_shared_name = TENSORS[i].name;
            }
            shared_moved += ndiff;
            shared_elems += n;
            if (ea_max > worst_exact_single)
                worst_exact_single = ea_max;
            if (eb_max > worst_exact_chunked)
                worst_exact_chunked = eb_max;
        }
        entry *e = &LIST[NL++];
        e->name = TENSORS[i].name;
        e->type = t->type;
        e->W = W;
        e->rows = rows;
        e->K = K;
        e->ldd = ldd;
        e->rb = rb;
        e->mib = (double)rb * (double)rows / 1048576.0;
        e->y = aligned_alloc(64, (size_t)MAXT * (size_t)rows * sizeof *e->y);
        if (!e->y) {
            fprintf(stderr, "alloc\n");
            return 1;
        }
        round_mib += e->mib;
        printf("  %-30s %-7s %6ldx%-7ld %8.1f MiB   %s\n", e->name, info ? info->name : "?", K,
               rows, e->mib, TENSORS[i].why);
    }
    printf("  %d tensors, %.0f MiB per round: at T=1 that is %.0f MiB per token with no reuse\n",
           NL, round_mib, round_mib);
    if (worst_shared_name)
        printf("  shared dequant: the Q8_0 round moved %ld of %ld results, the furthest by %.1e"
               " of their\n  own scale (%s). Read against one result that cancelled to almost"
               " nothing that\n  same move is a few percent, which is why the gate here is each"
               " path's error against the\n  exact dot product rather than the distance between"
               " two rounded ones: worst %.1e\n  single-token and %.1e chunked of sum|x*w|, a"
               " rounding boundary rather than a bug.\n  Not bit-identical, and not meant to be:"
               " one dequantised block for T tokens cannot sum\n  in the same order as T"
               " dequantisations. Every other type above was compared byte for byte\n",
               shared_moved, shared_elems, worst_shared, worst_shared_name, worst_exact_single,
               worst_exact_chunked);

    /* The counter's scale is taken from the one quantity here that is known with
     * no counter at all: a dense round at T=1 walks every byte of the round once,
     * and the tensor table says how many bytes that is. Every byte figure below is
     * therefore a ratio against a measured round of known size, which is the
     * strongest claim a hardware counter supports. The implied bytes per miss is
     * printed rather than assumed: if the event is not 64-byte granular, or if
     * threads went uncounted, the reader sees it as a number instead of as an
     * unexamined column. */
    double bpm = 64.0;
    if (CTR_OK && NL > 0) {
        unsigned long long m1 = 0;
        timed_round(1, 0, nthr, 4, &m1);
        if (m1)
            bpm = round_mib * 1048576.0 / (double)m1;
        printf("\n  counter scale: a dense T=1 round is known to touch %.0f MiB and the counters"
               " saw %llu misses,\n"
               "  so %.1f bytes of the round per counted miss%s\n",
               round_mib, m1, bpm, CTR_SILENT ? ", and some threads counted nothing" : "");
        if (CTR_SILENT)
            printf("  %d of %d threads returned an empty counter: their traffic is missing,"
                   " and the scale above is correspondingly too small\n", CTR_SILENT, nthr);
    }

    printf("\n     T   one at a time              chunked        bytes/token        time/token\n");
    printf("          %9s %8s   %9s %8s   %6s   %9s %9s\n", "MiB", "GB/s", "MiB", "GB/s", "saved",
           "one at a", "chunked");
    fflush(stdout);
    double dsum = 0, csum = 0, dtot = 0, ctot = 0, worst_dev = 0;
    double t1d = 0, t1c = 0, max_dense = 0;
    int have = 0;
    for (int i = 0; i < NTS; i++) {
        int T = TS[i];
        if (T > max_T)
            continue;
        unsigned long long md = 0, mc = 0;
        double dense = timed_round(T, 0, nthr, 4, &md);
        double chunked = timed_round(T, 1, nthr, 4, &mc);
        double dense_mib = CTR_OK ? (double)md * bpm / 1048576.0 / (double)T : round_mib;
        double chunked_mib = CTR_OK ? (double)mc * bpm / 1048576.0 / (double)T : round_mib / T;
        /* A dense round walks the same bytes whatever T is, since the round is
         * twenty times the L3, so this is a check on the counters that needs
         * nothing but the table itself. */
        double dev = dense_mib - round_mib;
        if (dev < 0)
            dev = -dev;
        if (dev > worst_dev)
            worst_dev = dev;
        double dpt = dense / T, cpt = chunked / T;
        printf("   %4d   %9.1f %8.1f   %9.1f %8.1f   %5.1f%%   %8.2f ms %8.2f ms%s\n", T, dense_mib,
               dpt > 0 ? dense_mib * 1048576.0 / dpt / 1e9 : 0.0, chunked_mib,
               cpt > 0 ? chunked_mib * 1048576.0 / cpt / 1e9 : 0.0,
               100.0 * (1.0 - chunked_mib / (dense_mib > 0 ? dense_mib : 1.0)), dpt * 1e3,
               cpt * 1e3, CTR_OK ? "" : "  (computed)");
        fflush(stdout);
        /* The second guard, and the one that matters: a dense round at T=1 is
         * known to touch every byte of the round exactly once, so a faithful set
         * of counters returns round_mib. Much less means some threads were not
         * being counted, and a saving measured with half the counters is not a
         * saving - it is an artefact of the ones that went quiet. */
        if (dense_mib > max_dense)
            max_dense = dense_mib;
        if (T == 1) {
            t1d = dense_mib;
            t1c = chunked_mib;
        }
        if (T == max_T) {
            dsum = dense_mib;
            csum = chunked_mib;
            dtot = dpt;
            ctot = cpt;
            have = 1;
        }
    }
    if (CTR_ARM_FAIL)
        printf("  note: a thread's counter failed to open (%s); its traffic is missing\n",
               strerror(CTR_ARM_FAIL));
    if (CTR_OK) {
        /* Three things a faithful set of counters cannot contradict, none of them
         * a calibration: every dense row walked the same round_mib of weights at
         * every T, no dense row can walk more than the round it walks, and at
         * T=1 the two paths being compared walk it identically. A run whose
         * threads are half-counted fails the third of these loudly - it is what
         * the sixteen-thread configuration does, printing chunked 640 MiB against
         * a 566 MiB round at T=1 - and an earlier version of this gate looked only
         * at the first, printed a 17% straying as acceptable, and let that run
         * through with no warning at all. */
        printf("\n  cross-check: the dense column is the same %.0f MiB of weights at every T;"
               " the furthest it strayed was %.0f MiB (%.0f%%)\n",
               round_mib, worst_dev, 100.0 * worst_dev / (round_mib > 0 ? round_mib : 1.0));
        if (t1c > 0)
            printf("  cross-check at T=1, where the two paths are one walk apiece: dense %.0f MiB"
                   " vs chunked %.0f MiB\n",
                   t1d, t1c);
        double t1_gap = t1d > 0 ? (t1c > t1d ? t1c - t1d : t1d - t1c) / t1d : 0;
        int bad = CTR_SILENT || worst_dev > 0.25 * round_mib || t1_gap > 0.10 ||
                  max_dense > 1.1 * round_mib;
        printf("  %s\n",
               !bad ? "counters consistent: the byte columns are the round's own bytes, counted"
                    : "INCOMPLETE: the counters do not cover the two paths alike, so the byte"
                      " columns are\n"
                      "  relative, not absolute, and must not be quoted as bytes per ingested"
                      " token.\n"
                      "  The times still stand: a stopwatch is not a hybrid PMU.");
    }
    if (have) {
        printf("\nthe dense half of a step at T=%d, %d threads: %.0f MiB and %.1f ms per token"
               " walked one at a time,\n  %.0f MiB and %.1f ms per token chunked%s\n",
               max_T, nthr, dsum, dtot * 1e3, csum, ctot * 1e3, CTR_OK ? "" : " (computed)");
        printf("chunked is %.1f%% of the dense bytes per ingested token and %.1f%% of the time%s\n",
               dsum > 0 ? 100.0 * csum / dsum : 0.0, dtot > 0 ? 100.0 * ctot / dtot : 0.0,
               CTR_OK ? ", counted as last-level-cache misses" : ", computed");
        printf("the MoE half already does this (qe_moe_forward_multi); this is the dense half,\n"
               "and it asks nothing of the machine that a normal step does not already have.\n");
    }
    qe_pool_shutdown();
    qe_close(m);
    return 0;
}
