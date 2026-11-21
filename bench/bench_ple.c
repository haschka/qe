/* The n-gram row gather measured against the real table.

   Whether this matters is not an opinion: a decode step is ~73 ms of memory at
   the measured GEMV rate, and the PLE gather is sixteen unrelated lines in a
   26.8 GiB table on the critical path of every token. This measures what that
   costs, how much the row cache and the sorted issue order take off it, what
   fraction of a token it therefore is, and - because a gather that reads the
   wrong row produces plausible floats and a broken model - whether the rows it
   returns are the ones the kernel would have read.

   Everything is read from the staged arena: qe_open() with no flags places the
   whole checkpoint in RAM first, so a page fault here is a TLB event, not a
   disk event.

     make ple                        # or:
     ./bin/bench_ple <volume1.gguf> [deadline_s] [max_tokens] [files...]
*/
#include "qe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int fails;
static void check(int ok, const char *what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    fflush(stdout);
    if (!ok)
        fails++;
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        fclose(f);
        return NULL;
    }
    b[n] = 0;
    fclose(f);
    *len = (size_t)n;
    return b;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* One row's sum, through the checkpoint's own validated IQ4_NL GEMV, as an
 * independent check on the gather's dequantisation: dot(row, ones) is the sum of
 * the very floats qe_ple_gather() wrote. */
static double kernel_row_sum(qe_model *m, long long row, size_t row_bytes, int dim) {
    const qe_tensor *t = qe_tensor_by_name(m, "per_layer_token_embd.weight");
    if (!t)
        return NAN;
    const uint8_t *base = (const uint8_t *)qe_tensor_data(m, t);
    if (!base)
        return NAN;
    float x[4096], y = 0;
    if (dim > (int)(sizeof x / sizeof *x))
        return NAN;
    for (int i = 0; i < dim; i++)
        x[i] = 1.0f;
    qe_gemv_iq4_nl(base + (size_t)row * row_bytes, x, &y, 1, dim, dim / 32);
    return y;
}

typedef struct {
    qe_ple **p;
    const uint32_t *seq;
    size_t n;
    float *emb;
} job;

static void gather_task(void *v, int tid, int nt) {
    job *j = (job *)v;
    size_t lo = j->n * (size_t)tid / (size_t)nt;
    size_t hi = j->n * (size_t)(tid + 1) / (size_t)nt;
    float *emb = j->emb + (size_t)tid * qe_ple_width(j->p[0]);
    for (size_t i = lo; i < hi; i++) {
        long long rows[16];
        qe_ple_rows_at(j->p[tid], j->seq, i, rows);
        qe_ple_gather(j->p[tid], rows, emb);
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                                : "/mnt/data/data/ai-models/llama-cpp/llm/"
                                  "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int deadline = argc > 2 ? atoi(argv[2]) : 600;
    long max_tokens = argc > 3 ? atol(argv[3]) : 20000;
    const char **files = (const char **)(argc > 4 ? argv + 4 : NULL);
    int n_files = argc > 4 ? argc - 4 : 0;
    qe_fuse(deadline);

    printf("== open ==\n");
    fflush(stdout);
    double t0 = now();
    qe_model *m = qe_open(path, QE_M_NONE);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    printf("  weights %s (%.1f GiB), %s\n", qe_model_stage_method(m),
           (double)qe_model_stage_bytes(m) / 1073741824.0,
           qe_model_stage_pinned(m) ? "pinned" : "reclaimable");
    fflush(stdout);

    char terr[256] = { 0 };
    qe_tok *tok = qe_tok_open(m, terr, sizeof terr);
    if (!tok) {
        fprintf(stderr, "tokenizer: %s\n", terr);
        qe_close(m);
        return 1;
    }

    /* ------------------------------------------------------ token stream */
    static const char *DEF[] = { "docs/DESIGN.md", "MANUAL.md", "src/tok.c",
                                 "tools/smoke_http.py" };
    if (!n_files) {
        files = DEF;
        n_files = (int)(sizeof DEF / sizeof DEF[0]);
    }
    int *ids = malloc((size_t)max_tokens * sizeof *ids);
    if (!ids)
        return 1;
    long n = 0;
    for (int f = 0; f < n_files && n < max_tokens; f++) {
        size_t len = 0;
        char *text = read_file(files[f], &len);
        if (!text) {
            printf("  (missing %s)\n", files[f]);
            continue;
        }
        long got = qe_tok_encode(tok, text, ids + n, max_tokens - n);
        if (got > 0)
            n += got;
        free(text);
    }
    if (n < 64) {
        fprintf(stderr, "only %ld tokens; need a real corpus\n", n);
        return 1;
    }
    uint32_t *seq = malloc((size_t)n * sizeof *seq);
    for (long i = 0; i < n; i++)
        seq[i] = (uint32_t)ids[i];
    printf("  %ld real tokens from %d file(s)\n", n, n_files);
    fflush(stdout);

    /* ------------------------------------------------------------- setup */
    char err[256] = { 0 };
    qe_ple *ple = qe_ple_create(m, NULL, err, sizeof err);
    if (!ple) {
        fprintf(stderr, "ple: %s\n", err);
        return 1;
    }
    int H = qe_ple_heads(ple), D = qe_ple_head_dim(ple), W = qe_ple_width(ple);
    size_t rb = qe_ple_row_bytes(ple);
    printf("\n== geometry ==\n  %d heads x %d elements = %d wide, %zu B/row, %lld table rows\n", H,
           D, W, rb, qe_ple_table_rows(ple));
    check(H == 16 && D == 160 && W == 2560 && rb == 90, "geometry matches docs/ARCH.md 6.2");

    printf("\n== selection ==\n");
    long long rows[16], rows2[16];
    qe_ple_rows_at(ple, seq, 3, rows);
    qe_ple_rows_at(ple, seq, 3, rows2);
    check(!memcmp(rows, rows2, sizeof rows), "row ids are a pure function of the window");
    int in_range = 1;
    for (int h = 0; h < H; h++)
        in_range &= rows[h] >= 0 && rows[h] < qe_ple_table_rows(ple);
    check(in_range, "every head selects a row the table has");

    /* The window rule, in both directions. Position 3 is the token, position 2
     * holds an EOS: a bigram reads the token and its predecessor, a trigram
     * reads two back, and the EOS in between erases itself and everything older
     * than it. So replacing those older positions with EOS changes nothing,
     * while changing what sits two back moves the trigram heads only. */
    uint32_t eos = qe_ple_eos_token(ple), teos = (uint32_t)qe_tok_eos(tok);
    printf("  the id that erases the window is %u, the tokenizer's end-of-sequence is %u\n", eos,
           teos);
    check(qe_tok_is_special(tok, (int) eos),
          "the id that erases the window is one of the vocabulary's control tokens");
    uint32_t cut[4] = { seq[3], seq[2], eos, seq[1] };
    uint32_t erased[4] = { eos, eos, eos, seq[1] };
    /* A window with no EOS in it, for the reach test: behind an EOS there is
     * nothing to reach, which is the other check, not this one. */
    uint32_t plain[4] = { seq[3], seq[2], seq[1], seq[0] };
    uint32_t older[4] = { seq[3], seq[2] + 1u, seq[1], seq[0] };
    long long ra[16], rb2[16], rc[16], rp[16];
    qe_ple_rows_at(ple, cut, 3, ra);
    qe_ple_rows_at(ple, erased, 3, rb2);
    qe_ple_rows_at(ple, plain, 3, rp);
    qe_ple_rows_at(ple, older, 3, rc);
    int erased_equal = 1, bigram_same = 1, trigram_moved = 0;
    for (int h = 0; h < H; h++) {
        erased_equal &= ra[h] == rb2[h];
        if (h < 8)
            bigram_same &= rp[h] == rc[h];
        else
            trigram_moved |= rp[h] != rc[h];
    }
    check(erased_equal, "an EOS erases itself and everything older from every head");
    check(bigram_same, "bigram heads ignore what the window does not read");
    check(trigram_moved, "trigram heads read exactly as far back as they claim");

    /* -------------------------------------------------------- numerics */
    printf("\n== numerics ==\n");
    float *emb = aligned_alloc(64, (size_t)W * sizeof *emb);
    float *emb2 = aligned_alloc(64, (size_t)W * sizeof *emb);
    if (!emb || !emb2)
        return 1;
    qe_ple_gather(ple, rows, emb);
    int finite = 1;
    double maxabs = 0;
    for (int i = 0; i < W; i++) {
        finite &= isfinite(emb[i]);
        if (fabs(emb[i]) > maxabs)
            maxabs = fabs(emb[i]);
    }
    check(finite, "gathered embedding is all finite");
    check(maxabs < 100.0, "magnitudes are inside a quantised 4-bit range");
    double ks = kernel_row_sum(m, rows[0], rb, D);
    double mine = 0;
    for (int i = 0; i < D; i++)
        mine += emb[i];
    check(isfinite(ks) && fabs(mine - ks) <= 1e-3 * (fabs(ks) + 1.0),
          "gathered row is what the IQ4_NL GEMV kernel reads");

    /* Same row, cached and uncached: a hit must not be allowed to differ. */
    qe_ple_gather(ple, rows, emb2);
    check(!memcmp(emb, emb2, (size_t)W * sizeof *emb), "a cache hit returns identical floats");

    /* ------------------------------------------------------- throughput */
    printf("\n== throughput (%ld tokens, single thread) ==\n", n);
    fflush(stdout);
    qe_ple_stats_clear(ple);
    double a = now();
    for (long i = 0; i < n; i++) {
        qe_ple_rows_at(ple, seq, (size_t)i, rows);
        qe_ple_gather(ple, rows, emb);
    }
    double cold = now() - a;
    qe_ple_stats c1;
    qe_ple_get_stats(ple, &c1);
    qe_ple_stats_clear(ple);
    a = now();
    for (long i = 0; i < n; i++) {
        qe_ple_rows_at(ple, seq, (size_t)i, rows);
        qe_ple_gather(ple, rows, emb);
    }
    qe_ple_stats_clear(ple);
    a = now();
    for (long i = 0; i < n; i++) {
        qe_ple_rows_at(ple, seq, (size_t)i, rows);
        qe_ple_gather(ple, rows, emb);
    }
    double warm = now() - a;
    qe_ple_stats c2;
    qe_ple_get_stats(ple, &c2);
    printf("  first pass over the corpus : %6.1f ns/token  %7.1f Mrows/s\n",
           cold * 1e9 / n, (double)c1.rows / cold / 1e6);
    printf("  second pass (cache warm)   : %6.1f ns/token  %7.1f Mrows/s\n",
           warm * 1e9 / n, (double)c2.rows / warm / 1e6);
    printf("  cold pass: %llu rows, %llu after dedup, %.1f%% served from the row cache\n",
           c1.rows, c1.distinct,
           100.0 * (double)c1.cache_hits / (double)(c1.rows ? c1.rows : 1));
    printf("  warm pass: %llu rows, %llu after dedup, %.1f%% served from the row cache\n",
           c2.rows, c2.distinct,
           100.0 * (double)c2.cache_hits / (double)(c2.rows ? c2.rows : 1));
    printf("  share of a 73 ms token     : %.3f%% cold, %.3f%% warm\n",
           cold * 1e9 / n / 73.0e6 * 100.0, warm * 1e9 / n / 73.0e6 * 100.0);
    check(cold * 1e9 / n < 20000.0, "the gather stays under 20 us/token cold");
    check(warm <= cold * 1.1, "a warm cache is not slower than a cold one");

    /* Issue/collect overlap: the loads are started at token-id time and the
     * decode happens where the engine needs the embedding. */
    a = now();
    for (long i = 0; i < n; i++) {
        qe_ple_issue(ple, seq, (size_t)i);
        qe_ple_collect(ple, emb);
    }
    double piped = now() - a;
    printf("  issue/collect pipeline     : %6.1f ns/token\n", piped * 1e9 / n);
    check(qe_ple_issue(ple, seq, 0) == 0 && qe_ple_issue(ple, seq, 1) == 0,
          "issue accepts pending tokens");
    int full = 0;
    for (int k = 0; k < 16; k++)
        full |= qe_ple_issue(ple, seq, (size_t)(k % n)) < 0;
    check(full, "issue reports the pipeline is full instead of dropping a token");
    for (int k = 0; k < 16 && qe_ple_collect(ple, emb) == 0; k++)
        ;

    /* Two different amplifications, because they are answered by two different
     * parts of the machine. A row is 90 bytes, so DRAM answers it in two 64-byte
     * lines: that is the byte cost, and it is what a wider page cannot reduce.
     * The page count is the translation cost - the number of distinct 4 KiB
     * pages whose entries a token's 16 rows touch, which is what a 2 MiB page
     * and a row cache both attack. */
    uint64_t *pages = malloc((size_t)n * 16 * sizeof *pages);
    if (pages) {
        size_t np = 0;
        for (long i = 0; i < n; i++) {
            qe_ple_rows_at(ple, seq, (size_t)i, rows);
            for (int h = 0; h < H; h++)
                pages[np++] = (uint64_t)((size_t)rows[h] * rb) >> 12;
        }
        qsort(pages, np, sizeof *pages, cmp_u64);
        size_t distinct = 1;
        for (size_t i = 1; i < np; i++)
            distinct += pages[i] != pages[i - 1];
        printf("  %.1f cache lines per token for %.0f B wanted: %.2fx at the memory bus\n", 2.0 * H,
               (double)H * rb, 2.0 * 64.0 / (double)rb);
        printf("  %.1f distinct 4 KiB pages per token to translate (2 MiB pages: %.2f each)\n",
               (double)distinct / n, (double)distinct / n / 512.0);
        printf("  page-table entries touched per token: %.1f at 4 KiB, %.2f at 2 MiB\n",
               (double)distinct / n, 1.0 * H * rb / (2.0 * 1024.0 * 1024.0) + 1.0);
        free(pages);
    }

    /* ------------------------------------------------ thread scaling */
    printf("\n== thread scaling (private caches, second pass) ==\n");
    fflush(stdout);
    static const int TC[] = { 1, 4, 8, 16, 20 };
    qe_pool_init(20, NULL, 0);
    qe_ple_desc small = { NULL, 1 << 16 };
    qe_ple *all[20] = { 0 };
    for (int t = 0; t < 20; t++) {
        qe_ple *q = qe_ple_create(m, &small, err, sizeof err);
        if (!q) {
            fprintf(stderr, "ple: %s\n", err);
            return 1;
        }
        all[t] = q;
    }
    float *embank = aligned_alloc(64, 20 * (size_t)W * sizeof *embank);
    if (!embank)
        return 1;
    for (int ti = 0; ti < (int)(sizeof TC / sizeof TC[0]); ti++) {
        int nt = TC[ti];
        job j = { all, seq, (size_t)n, embank };
        /* one warm-up so the cache is holding rows before it is measured */
        qe_pool_run(gather_task, &j, nt);
        double b = now();
        qe_pool_run(gather_task, &j, nt);
        double d = now() - b;
        printf("  %2d threads: %7.1f Mrows/s  (%.1f ns/token of a 73 ms step)\n", nt,
               (double)n * H / d / 1e6, d * 1e9 / n);
    }
    free(embank);
    for (int t = 0; t < 20; t++)
        qe_ple_destroy(all[t]);
    qe_pool_shutdown();

    qe_ple_destroy(ple);
    free(emb);
    free(emb2);
    free(seq);
    free(ids);
    qe_tok_close(tok);
    qe_close(m);
    printf("\n%s (%d failures) in %.1fs\n", fails ? "FAIL" : "PASS", fails, now() - t0);
    return fails ? 1 : 0;
}
