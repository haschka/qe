/* Where the per-token bytes should live.
 *
 * A decode step of this checkpoint reads 5.44 GiB, and 66.6% of it is dense
 * Q8_0 projections read as one GEMV per weight row. The published GEMV number
 * for this box (73.6-78.7 GB/s, against the 76-79 GB/s that `make alu` measures
 * as the ceiling for this read shape) was measured on a span in malloc'd 4 KiB
 * page memory, because that is what the harness happened
 * to use. On this machine that is a choice with consequences:
 *
 *   - HugePages_Total is 0, so MAP_HUGETLB cannot be granted and qe_alloc_huge
 *     silently falls back.
 *   - Transparent huge pages are in [madvise] mode, so anonymous memory gets
 *     2 MiB pages only if someone asks with madvise.
 *   - The weights themselves arrive through a file mapping, which THP does not
 *     cover at all: 5.44 GiB/token over 4 KiB pages is 1.4M page-table walks
 *     per token, on top of the TLB pressure of a 30 MiB L3.
 *
 * So measure the same real bytes in every placement this box can actually give
 * us, and separate "the memory is slow" from "the kernel is slow" by also
 * timing a pure read of the same span.
 *
 *   cc ... -o bin/bench_stage bench/bench_stage.c src/mem.o src/gguf.o \
 *          src/kern.o src/kern_iq.c && ./bin/bench_stage
 */
#include "qe.h"

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define SPAN_MIB 512.0

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static volatile sig_atomic_t wd_fired;
static void wd_handler(int sig) {
    (void)sig;
    wd_fired = 1;
    const char msg[] = "\n[watchdog] deadline exceeded - aborting, partial output above\n";
    ssize_t r = write(2, msg, sizeof(msg) - 1);
    (void)r;
    _exit(124);
}
static void watchdog(int seconds) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = wd_handler;
    sigaction(SIGALRM, &sa, NULL);
    alarm((unsigned)seconds);
}

/* AnonHugePages from /proc/self/smaps_rollup: the only honest way to say "this
 * placement got 2 MiB pages" on a box where the request can be refused. */
static long anon_huge_kib(void) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (!f)
        return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "AnonHugePages:", 14)) {
            kb = strtol(line + 14, NULL, 10);
            break;
        }
    fclose(f);
    return kb;
}

typedef struct {
    const void *W;
    const float *x;
    float *y;
    long rows, K, ldd;
} gemv_job;

static void gemv_task(void *v, int tid, int nt) {
    gemv_job *j = (gemv_job *)v;
    long per = (j->rows + nt - 1) / nt;
    long lo = (long)tid * per, hi = lo + per;
    if (hi > j->rows) hi = j->rows;
    if (lo >= hi) return;
    qe_gemv_q8_0((const uint8_t *)j->W + (size_t)lo * j->ldd * 34, j->x, j->y + lo,
                 hi - lo, j->K, j->ldd);
}

typedef struct {
    const uint8_t *p;
    size_t n;
    double *out;
} read_job;

/* Same bytes, no arithmetic: the memory-controller question on its own. */
static void read_task(void *v, int tid, int nt) {
    read_job *j = (read_job *)v;
    size_t per = ((j->n / 4096) + (size_t)nt - 1) / (size_t)nt * 4096;
    size_t lo = (size_t)tid * per, hi = lo + per;
    if (lo >= j->n) return;
    if (hi > j->n) hi = j->n;
    const volatile uint8_t *p = j->p + lo;
    unsigned long acc = 0;
    for (size_t i = lo; i < hi; i += 64)
        acc += p[i - lo];
    if (acc == 0xdeadbeefUL)
        j->out[0] = (double)acc;
}

typedef enum { M_MALLOC, M_THP, M_HUGE, M_POPULATED, M_N } mode;
static const char *MODE_NAME[M_N] = {
    "malloc (4 KiB pages)",
    "mmap + MADV_HUGEPAGE",
    "qe_alloc_huge (as the engine uses it)",
    "mmap + MADV_HUGEPAGE + POPULATE",
};

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                                 : "/mnt/data/data/ai-models/llama-cpp/llm/"
                                   "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int deadline = argc > 2 ? atoi(argv[2]) : 300;
    watchdog(deadline);

    /* Placements probe, so it wants the plain file mapping: the point is to
     * compare backings on identical bytes, and staging 87 GiB first would only
     * compare the arena against itself. The engine's own choice is settled
     * (whole checkpoint in RAM at open); see docs/ARCH.md. */
    qe_model *m = qe_open(path, QE_STAGE_MMAP);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    const qe_tensor *te = qe_tensor_by_name(m, "token_embd.weight");
    const void *src = te ? qe_tensor_data(m, te) : NULL;
    if (!te || te->type != QE_Q8_0 || !src) {
        fprintf(stderr, "this bench needs token_embd.weight as Q8_0\n");
        return 1;
    }
    long K = (long)(te->ne[0] & ~31LL), ldd = K / 32;
    size_t row_bytes = (size_t)ldd * 34;
    long rows = (long)((SPAN_MIB * 1048576.0) / (double)row_bytes);
    if (rows > (long)te->ne[1])
        rows = (long)te->ne[1];
    size_t span = (size_t)rows * row_bytes;
    printf("span: %ld rows x %zu B = %.0f MiB of real Q8_0 (K=%ld)\n",
           rows, row_bytes, span / 1048576.0, K);

    float *x = aligned_alloc(64, (size_t)K * sizeof *x);
    float *y = aligned_alloc(64, (size_t)rows * sizeof *y);
    if (!x || !y)
        return 1;
    for (long i = 0; i < K; i++)
        x[i] = (float)sin(i * 0.0017) * 0.5f;

    /* One pass to get the bytes; no bandwidth is reported for it. Storage speed
     * is not a quantity this project optimises: the engine reads the checkpoint
     * once, into RAM, at startup, and every number below is memory. */
    uint8_t *buf[M_N] = { 0 };
    buf[M_MALLOC] = aligned_alloc(64, qe_align_up(span, 4096u));
    if (!buf[M_MALLOC])
        return 1;
    memcpy(buf[M_MALLOC], src, span);
    printf("span filled once from the checkpoint\n\n");

    printf("%-40s %10s %10s %10s %12s\n", "placement", "GEMV GB/s", "read GB/s",
           "GEMV/read", "AnonHugePages");
    for (int t = 1; t <= 2; t++) {
        int nthr = t == 1 ? 20 : 12;
        printf("-- %d threads\n", nthr);
        qe_pool_init(nthr, NULL, 0);
        for (mode md = 0; md < M_N; md++) {
            fprintf(stderr, "  [%s] setting up\n", MODE_NAME[md]);
            if (md != M_MALLOC) {
                size_t n = qe_align_up(span, (size_t)2 << 20);
                int flags = MAP_PRIVATE | MAP_ANONYMOUS;
                if (md == M_POPULATED)
                    flags |= MAP_POPULATE;
                void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, flags, -1, 0);
                if (p == MAP_FAILED) {
                    printf("%-40s   mmap failed: %s\n", MODE_NAME[md], strerror(errno));
                    continue;
                }
#ifdef MADV_HUGEPAGE
                if (md == M_THP || md == M_POPULATED)
                    madvise(p, n, MADV_HUGEPAGE);
#endif
                buf[md] = p;
                fprintf(stderr, "  [%s] mapping at %p, copying\n", MODE_NAME[md], p);
                memcpy(buf[md], buf[M_MALLOC], span);
                fprintf(stderr, "  [%s] AnonHugePages=%ld MiB, measuring\n", MODE_NAME[md],
                        anon_huge_kib() / 1024);
            }
            gemv_job j = { buf[md], x, y, rows, K, ldd };
            double sink = 0;
            read_job rj = { buf[md], span, &sink };
            double best = 1e30, best_r = 1e30;
            for (int rep = 0; rep < 4; rep++) {
                double a = now();
                qe_pool_run(gemv_task, &j, nthr);
                double d = now() - a;
                if (d < best) best = d;
            }
            for (int rep = 0; rep < 4; rep++) {
                double a = now();
                qe_pool_run(read_task, &rj, nthr);
                double d = now() - a;
                if (d < best_r) best_r = d;
            }
            double gb = span / best / 1e9, gr = span / best_r / 1e9;
            long ah = anon_huge_kib();
            printf("%-40s %10.1f %10.1f %9.0f%% %10ld MiB\n", MODE_NAME[md], gb, gr,
                   100.0 * gb / gr, ah > 0 ? ah / 1024 : -1);
            if (md != M_MALLOC)
                munmap(buf[md], qe_align_up(span, (size_t)2 << 20));
            buf[md] = NULL;
        }
    }
    printf("\nread GB/s is the same span touched with no arithmetic: it is the\n"
           "memory-controller answer. GEMV/read below ~90%% means the kernel, not\n"
           "the placement, is what is holding the projections down.\n");
    free(buf[M_MALLOC]);
    free(x);
    free(y);
    qe_close(m);
    return wd_fired ? 124 : 0;
}
