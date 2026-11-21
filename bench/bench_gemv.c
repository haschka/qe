/* Load-time measurement harness.
 *
 * Establishes, with the real weights and before any graph code exists:
 *   1. that the loader agrees with the file byte-for-byte (offset-chain proof)
 *   2. that the SIMD Q8_0 kernel agrees with a scalar reference
 *   3. how many GB/s a decode-shaped GEMV gets on this box, and hence the
 *      tok/s ceiling the rest of the engine has to live under
 *
 * Everything is budgeted by wall clock, not by iterations: a phase that stalls
 * is worse than no benchmark, because it reports nothing and looks like a hang.
 * Hence the watchdog. The deadline has to cover the one-time startup read too -
 * qe_open places the whole 87 GiB checkpoint in RAM before any of this runs -
 * so it is counted in minutes, and a short one aborts in the middle of staging.
 */
#include "qe.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

extern double qe_dot_q8_0_ref(const qe_block_q8_0 *w, const float *x, long nb);

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

/* One tensor per architecture role that every token touches. */
static const char *HOT[] = {
    "blk.0.attn_qkv.weight",      /* 36 gated-delta-net input projections */
    "blk.0.ssm_out.weight",       /* GDN output projection                */
    "blk.0.attn_gate.weight",     /* gated output gate                    */
    "blk.3.attn_q.weight",        /* full-attention projections           */
    "blk.3.attn_output.weight",
    "blk.0.hc_attn_up.weight",    /* hyper-connections                    */
    "blk.0.hc_ffn_down.weight",
    "blk.0.ffn_gate_shexp.weight",/* shared expert                        */
    "blk.0.ffn_gate_inp.weight",  /* router (f32, dense)                  */
};
#define NHOT ((int)(sizeof(HOT) / sizeof(HOT[0])))

/* Rows the gemv tasks are allowed to write into y. The DRAM-bound phase asks
 * for however many rows fill a 512 MiB span, which at K=2560 is ~197k rows, so
 * this has to cover that. It used to be a fixed 32768 and the DRAM phase
 * walked straight past the end of y into a SIGSEGV. The row-length control reads
 * the same span as K=320 rows, which is eight times the rows for the same bytes,
 * hence 2M. */
#define Y_ROWS (2L << 20)

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
    qe_gemv_q8_0((const uint8_t *)j->W + (size_t)lo * j->ldd * 34, j->x,
                 j->y + lo, hi - lo, j->K, j->ldd);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                                 : "/mnt/data/data/ai-models/llama-cpp/llm/"
                                   "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int nthr = argc > 2 ? atoi(argv[2]) : 16;
    if (nthr < 1) nthr = 1;
    int deadline = argc > 3 ? atoi(argv[3]) : 600;
    watchdog(deadline);

    /* ---- open + structural proof ----------------------------------- */
    double t0 = now();
    qe_model *m = qe_open(path, QE_M_NONE);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    /* Says where the bytes are, because everything measured below is only a
     * memory figure if this line says ram. How long the load took is not
     * reported: it is a one-time read of a device nothing here optimises. */
    printf("== weights %s (%.1f GiB placed, %.1f GiB reused), %s ==\n",
           qe_model_stage_method(m), (double)qe_model_stage_bytes(m) / 1073741824.0,
           (double)qe_model_stage_cached(m) / 1073741824.0,
           qe_model_stage_pinned(m) ? "pinned" : "reclaimable");
    fflush(stdout);

    long long L = 0, E = 0, EU = 0, H = 0, FAI = 0;
    qe_model_get_int(m, "qwen4exp.block_count", &L);
    qe_model_get_int(m, "qwen4exp.expert_count", &E);
    qe_model_get_int(m, "qwen4exp.expert_used_count", &EU);
    qe_model_get_int(m, "qwen4exp.embedding_length", &H);
    qe_model_get_int(m, "qwen4exp.full_attention_interval", &FAI);
    const char *arch = "?", *nm = "?";
    qe_model_get_str(m, "general.architecture", &arch);
    qe_model_get_str(m, "general.name", &nm);

    uint64_t total = 0, dtbytes[QE_TYPE_COUNT] = {0};
    unsigned dtcount[QE_TYPE_COUNT] = {0};
    for (size_t i = 0; i < qe_tensor_count(m); i++) {
        const qe_tensor *t = qe_tensor_by_index(m, i);
        total += t->bytes;
        dtcount[t->type]++;
        dtbytes[t->type] += t->bytes;
    }
    printf("== %s (%s) ==\n", nm, arch);
    printf("layers=%lld hidden=%lld experts=%lld active=%lld full_attn_every=%lld\n",
           L, H, E, EU, FAI);
    printf("tensors=%zu weights=%.2f GiB  [offset chain verified across %d volumes]\n",
           qe_tensor_count(m), total / 1073741824.0, 3);
    printf("\n%-9s %8s %12s %8s\n", "dtype", "count", "bytes", "bpw");
    for (unsigned d = 0; d < QE_TYPE_COUNT; d++)
        if (dtcount[d]) {
            const qe_type_info *ti = qe_type_info_of((qe_type)d);
            printf("%-9s %8u %9.2f GiB %8.4f\n", ti->name, dtcount[d],
                   dtbytes[d] / 1073741824.0, qe_type_bpw_x1000((qe_type)d) / 1000.0);
        }
    fflush(stdout);

    /* ---- bytes one decode step reads, from the tensor table -------- */
    /* Every tok/s number in this project is a bandwidth divided by this
     * number, so it is computed from the file rather than carried as a
     * constant someone typed into a comment. For the record: this came out
     * 5.2% above the figure this project budgeted with for weeks, and the
     * difference is exactly the F32 router logits (240 MiB) and the BF16
     * indexer (38 MiB), both of which had been written off as tiny.
     *
     * The weight each tensor contributes per token:
     *   ffn_*_exps            expert_used_count of expert_count run
     *   token_embd            embedding lookup: one row, not the table
     *   per_layer_token_embd  PLE n-gram table: ~16 rows
     *   everything else       read in full by every token that decodes */
    enum { C_ATTN, C_EXPS, C_HC, C_LMH, C_ROUT, C_SHEXP, C_PLE, C_PLEP, C_EMBD, C_OTHER, NCOMP };
    static const char *CNAME[NCOMP] = {
        "attn + GDN + indexer", "routed experts", "hyper-connections",
        "lm_head", "router logits (F32)", "shared expert", "PLE n-gram rows",
        "PLE projections", "token_embd (1 row)", "other",
    };
    double cmib[NCOMP] = { 0 }, step_mib = 0;
    {
        const qe_tensor *te1 = qe_tensor_by_name(m, "token_embd.weight");
        const qe_tensor *tp = qe_tensor_by_name(m, "per_layer_token_embd.weight");
        double w_embd = (te1 && te1->ne[1]) ? 1.0 / (double)te1->ne[1] : 0.0;
        double w_ple = (tp && tp->ne[1]) ? 16.0 / (double)tp->ne[1] : 0.0;
        double w_exps = (E && EU) ? (double)EU / (double)E : 1.0;
        for (size_t i = 0; i < qe_tensor_count(m); i++) {
            const qe_tensor *t = qe_tensor_by_index(m, i);
            const char *s = t->name;
            if (!strncmp(s, "blk.", 4)) {
                const char *d = strchr(s + 4, '.');
                if (d) s = d + 1;
            }
            int c = C_OTHER;
            double wgt = 1.0;
            if (strstr(s, "_exps.weight")) { c = C_EXPS; wgt = w_exps; }
            else if (!strcmp(s, "output.weight")) c = C_LMH;
            else if (!strcmp(s, "ffn_gate_inp.weight")) c = C_ROUT;
            else if (strstr(s, "_shexp.weight")) c = C_SHEXP;
            else if (!strncmp(s, "hc_", 3) || !strncmp(s, "output_hc", 9)) c = C_HC;
            else if (!strncmp(s, "per_layer_token_embd", 20)) { c = C_PLE; wgt = w_ple; }
            else if (!strncmp(s, "ple_", 4)) c = C_PLEP;
            else if (!strcmp(s, "token_embd.weight")) { c = C_EMBD; wgt = w_embd; }
            else if (!strncmp(s, "attn_", 5) || !strncmp(s, "ssm_", 4) ||
                     !strncmp(s, "indexer.", 8)) c = C_ATTN;
            cmib[c] += (double)t->bytes * wgt / 1048576.0;
        }
        for (int c = 0; c < NCOMP; c++) step_mib += cmib[c];
        printf("\n== bytes one decode step reads (from the tensor table) ==\n");
        printf("%-24s %11s %7s\n", "component", "MiB/token", "share");
        for (int c = 0; c < NCOMP; c++) {
            if (step_mib > 0 && cmib[c] < step_mib * 0.001) continue;
            printf("%-24s %11.1f %6.1f%%\n", CNAME[c], cmib[c],
                   100.0 * cmib[c] / step_mib);
        }
        printf("%-24s %11.1f %6.1f%%  = %.3f GiB\n", "total", step_mib, 100.0,
               step_mib * 1048576.0 / 1073741824.0);
        printf("(rows below 0.1%% of the step are omitted; %d experts of %d run, "
               "PLE table counts 16 rows)\n", (int)EU, (int)E);
        fflush(stdout);
    }

    /* Draft head assumed by docs/DESIGN.md 4: unsloth's shared-Q4_K_M MTP file
     * is 1.91 GB = 1821 MiB, reported mean draft length 2.76 tokens at greedy.
     * Cost per emitted token is (target + draft x tok_per_step) / tok_per_step. */
    const double DRAFT_MIB = 1821.0, DRAFT_TOK = 2.76;
    const double mtp_mib = (step_mib + DRAFT_MIB * DRAFT_TOK) / DRAFT_TOK;

    long long h = H ? H : 2560;
    /* Wide enough for the widest row this bench walks. Most projections are
     * hidden-width, but a hyper-connection down projection contracts over all four
     * residual streams at once - 10240 columns - so an activation of hidden width
     * would force the bench to measure those rows truncated, which is a fifth of
     * the bytes and a different kernel argument. */
    const long xcap = h > 10240 ? h : 10240;
    float *x = aligned_alloc(64, (size_t)xcap * 4);
    float *y = aligned_alloc(64, (size_t)Y_ROWS * 4);
    for (long i = 0; i < xcap; i++)
        x[i] = (float)sin(i * 0.0017) * 0.5f;

    /* Nothing to prefault - the arena is already resident - so this only decides
     * which tensors are warm in cache when the per-tensor table starts. */
    qe_model_warm(m, HOT, NHOT, 0);

    /* Sized for the widest sweep below, because qe_pool_run caps the request at
     * the pool size and a capped run reports two thread counts doing the same
     * work. */
    qe_pool_init(nthr > 20 ? nthr : 20, NULL, 0);

    /* ---- Q8_0 GEMV: correctness then throughput -------------------- */
    printf("\n== Q8_0 GEMV on real tensors (K=%lld) ==\n", h);
    printf("%-26s %7s %9s %9s %10s %9s\n",
           "tensor", "rows", "MiB", "GB/s 1T", "GB/s xT", "maxrel");
    fflush(stdout);
    double sum1 = 0, sumn = 0, bwsum = 0, bwbest = 0;
    int nbw = 0;
    double worst = 0;
    for (int i = 0; i < NHOT; i++) {
        const qe_tensor *t = qe_tensor_by_name(m, HOT[i]);
        if (!t || t->type != QE_Q8_0) continue;
        const void *W = qe_tensor_data(m, t);
        if (!W) continue;
        long K = (long)(t->ne[0] & ~31LL);
        if (K > h) K = h & ~31LL;
        long rows = (long)t->ne[1];
        if (rows > Y_ROWS) rows = Y_ROWS;
        long ldd = K / 32;

        /* correctness against an independent scalar reference */
        float probe[8];
        qe_gemv_q8_0(W, x, probe, 8, K, ldd);
        double maxerr = 0;
        for (int r = 0; r < 8; r++) {
            const qe_block_q8_0 *row =
                (const qe_block_q8_0 *)((const uint8_t *)W + (size_t)r * ldd * 34);
            double ref = qe_dot_q8_0_ref(row, x, K / 32);
            double e = fabs(ref - probe[r]);
            double rel = e / (fabs(ref) > 1e-6 ? fabs(ref) : 1e-6);
            if (rel > maxerr) maxerr = rel;
        }
        if (maxerr > worst) worst = maxerr;

        size_t bytes = (size_t)rows * ldd * 34;
        gemv_job j = {W, x, y, rows, K, ldd};
        double t1 = 1e30, tm = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            double a = now();
            gemv_task(&j, 0, 1);
            double d = now() - a;
            if (d < t1) t1 = d;
        }
        for (int rep = 0; rep < 6; rep++) {
            double a = now();
            qe_pool_run(gemv_task, &j, nthr);
            double d = now() - a;
            if (d < tm) tm = d;
        }
        double g1 = bytes / t1 / 1e9, gm = bytes / tm / 1e9;
        printf("%-26s %7ld %9.1f %9.1f %10.1f %9.1e%s\n", HOT[i], rows,
               bytes / 1048576.0, g1, gm, maxerr,
               bytes > (28u << 20) ? "" : "  <- fits in 30 MiB L3, not a DRAM number");
        fflush(stdout);
        sum1 += g1; sumn += 1;
        bwsum += gm; nbw += 1;
        if (gm > bwbest) bwbest = gm;
    }
    if (nbw) {
        printf("%-26s %7s %9s %9.1f %10.1f\n", "MEAN", "", "",
               sum1 / (sumn ? sumn : 1), bwsum / nbw);
        printf("kernel correctness: worst relative error over 8 rows x %d tensors = %.2e\n",
               nbw, worst);
    }

    /* ---- DRAM-bound GEMV ------------------------------------------- */
    /* The per-tensor numbers above are cache-hot: anything under the 30 MiB L3
     * gets re-read from cache and reports bandwidth the memory controller is
     * not providing. A decode step reads 5.44 GiB, so the number that matters
     * is GEMV throughput over a span far larger than L3. token_embd.weight is
     * 644 MiB of Q8_0 with exactly the K=2560 row shape the real projections
     * use, so it stands in for the whole stream.
     *
     * Read straight out of the staging arena, because that is where a decode
     * step reads from: qe_open placed the volume in private pinned memory at
     * startup, so this pointer is RAM and the only thing between it and the
     * kernel is the memory controller. Nothing in this section touches the
     * file. */
    printf("\n== DRAM-bound Q8_0 GEMV: 512 MiB span out of the staged weights ==\n");
    fflush(stdout);
    {
        const qe_tensor *te = qe_tensor_by_name(m, "token_embd.weight");
        const void *W = te ? qe_tensor_data(m, te) : NULL;
        if (!te)
            printf("skipped: no token_embd.weight in this checkpoint\n");
        else if (te->type != QE_Q8_0 || !W)
            printf("skipped: token_embd.weight is %s and %s\n",
                   te->type == QE_Q8_0 ? "Q8_0" : "not Q8_0",
                   W ? "mapped" : "not mapped");
        else {
            long K = (long)(te->ne[0] & ~31LL);
            long ldd = K / 32;
            long rows = (long)te->ne[1];
            if (K > h) K = h & ~31LL, ldd = K / 32;
            size_t row_bytes = (size_t)ldd * 34;
            long span_rows = (long)((512.0 * 1048576.0) / (double)row_bytes);
            if (span_rows > rows) span_rows = rows;
            if (span_rows > Y_ROWS) span_rows = Y_ROWS;
            size_t span_bytes = (size_t)span_rows * row_bytes;
            const char *how = qe_model_stage_method(m);
            int resident = strcmp(how, "mmap") != 0;

            gemv_job j = {(uint8_t *)W, x, y, span_rows, K, ldd};
            double step = step_mib * 1048576.0;
            /* Thread counts swept here, not by restarting the bench: a restart
             * means reading another 87 GiB, and the bytes are already in RAM
             * where the engine will read them. Repeated cold loads of the same
             * checkpoint are a mistake, not an experiment. */
            int tc[4] = {nthr, 8, 16, 20};
            double best_gb = 0;
            int best_n = nthr;
            for (int ti = 0; ti < 4; ti++) {
                int nt = tc[ti];
                if (nt < 1 || nt > 64)
                    continue;
                int dup = 0;
                for (int k = 0; k < ti; k++)
                    dup |= (tc[k] == nt);
                if (dup)
                    continue;
                double bm = 1e30, bworst = 0;
                for (int rep = 0; rep < 3; rep++) {
                    double a = now();
                    qe_pool_run(gemv_task, &j, nt);
                    double d = now() - a;
                    if (d < bm) bm = d;
                    if (d > bworst) bworst = d;
                }
                double gb = span_bytes / bm / 1e9;
                printf("%ld rows x %zu B = %.0f MiB from the %s weights through kernel "
                       "x%2d threads: %6.1f GB/s (worst rep %.1f)%s\n", span_rows, row_bytes,
                       span_bytes / 1048576.0, how, nt, gb, span_bytes / bworst / 1e9,
                       nt == nthr ? "  <- requested" : "");
                if (gb > best_gb)
                    best_gb = gb, best_n = nt;
            }
            if (!resident)
                printf("  (stage mmap: those pages come from the file, so this is "
                       "a partly-cached figure and not a DRAM ceiling)\n");
            printf("best of the sweep: %d threads\n", best_n);
            printf("--> engine ceiling at measured GEMV rate (%.0f MiB/step): "
                   "%5.1f tok/s\n", step_mib, best_gb * 1e9 / step);
            printf("--> with MTP shared-Q4_K_M @%.2f tok/step (%.0f MiB/token): "
                   "%5.1f tok/s\n", DRAFT_TOK, mtp_mib,
                   best_gb * 1e9 / (mtp_mib * 1048576.0));

            /* Row length, controlled. The hyper-connection mixers are the stage
             * whose engine-side rate is a third below the projections' - 656 MiB
             * over the 12.5 ms the profile charges them, against 81-84 GB/s for
             * the 2560-wide GEMV stages - and the one unusual thing about them is
             * a 340-byte row where a projection row is 2720. No tensor in the
             * model is large enough to ask that of directly: every one of them
             * fits in the 30 MiB L3 and reports cache, which is what the
             * per-tensor table above prints for them. So the question gets the
             * instrument bench_moe used to settle the same question about the down
             * projection - one span, one kernel, one thread count, rows re-tiled.
             * If row length is the cause the rate falls with the row; if it does
             * not, the mixers are slow for a reason this bench cannot see, and the
             * checkpoint-side folding idea in docs/DESIGN.md §14 has no support. */
            printf("\n== the same span with the rows re-tiled: is it the row length? ==\n");
            fflush(stdout);
            {
                const long kss[4] = { 2560, 1280, 640, 320 };
                for (int ki = 0; ki < 4; ki++) {
                    long kk = kss[ki] & ~31LL;
                    if (kk > K)
                        kk = K & ~31LL;
                    long ld = kk / 32;
                    size_t rb = (size_t)ld * 34;
                    long nr = (long)(span_bytes / rb);
                    if (nr > Y_ROWS)
                        nr = Y_ROWS;
                    size_t nb = (size_t)nr * rb;
                    gemv_job jk = { (uint8_t *)W, x, y, nr, kk, ld };
                    double bm = 1e30;
                    for (int rep = 0; rep < 3; rep++) {
                        double a = now();
                        qe_pool_run(gemv_task, &jk, nthr);
                        double d = now() - a;
                        if (d < bm)
                            bm = d;
                    }
                    printf("%4zu B/row K=%4ld x %8ld rows = %5.0f MiB through kernel x%2d:"
                           " %6.1f GB/s\n",
                           rb, kk, nr, nb / 1048576.0, nthr, nb / bm / 1e9);
                    fflush(stdout);
                }
            }
        }
        fflush(stdout);
    }

    /* ---- the real hyper-connection walk ---------------------------- */
    /* The row-length control prices a 340-byte row at 18% of a GEMV, and the
     * engine charges its two mixer stages 12.5 ms for 656 MiB, which is 52 GB/s
     * against the 63.9 that row length just reached. Either the mixers are at the
     * limit of the shapes the checkpoint wrote, or 2.3 ms of every token is on the
     * table, and only the tensors themselves can say which. Two shapes live under
     * one name here, which is the other reason to measure rather than infer:
     * `hc_*_up` is 320 columns by 10240 rows (340 B a row, a wide parallel walk)
     * and `hc_*_down` is 10240 by 320 (10880 B a row, 320 rows to hand out to 19
     * threads). A per-tensor loop cannot see either: 3.3 MiB is a tenth of the
     * L3, so this walks every hyper-connection tensor of every layer once, in
     * tensor order, which is the order a step reads them. */
    printf("\n== every hyper-connection tensor, in the order a step reads them ==\n");
    fflush(stdout);
    {
        static gemv_job jobs[512];
        static int jfam[512];
        static double jbytes[512];
        int nj = 0;
        for (size_t i = 0; i < qe_tensor_count(m) && nj < 512; i++) {
            const qe_tensor *t = qe_tensor_by_index(m, i);
            const char *s = t->name;
            if (!strstr(s, "hc") || !strstr(s, ".weight") || t->type != QE_Q8_0)
                continue;
            const void *W = qe_tensor_data(m, t);
            if (!W)
                continue;
            long K = (long)(t->ne[0] & ~31LL);
            if (K > xcap)
                K = xcap & ~31LL;
            long rows = (long)t->ne[1];
            if (rows > Y_ROWS)
                rows = Y_ROWS;
            if (K < 32 || rows < 1)
                continue;
            long ldd = K / 32;
            jobs[nj] = (gemv_job){ W, x, y, rows, K, ldd };
            jbytes[nj] = (double)rows * ldd * 34;
            jfam[nj] = strstr(s, "_up.weight") ? 0 : 1;
            nj++;
        }
        double fbest[2] = { 1e30, 1e30 }, fbytes[2] = { 0, 0 };
        long fn[2] = { 0, 0 };
        for (int k = 0; k < nj; k++) {
            fbytes[jfam[k]] += jbytes[k];
            fn[jfam[k]]++;
        }
        for (int rep = 0; rep < 4; rep++) {
            double tsum[2] = { 0, 0 };
            for (int k = 0; k < nj; k++) {
                double a = now();
                qe_pool_run(gemv_task, &jobs[k], nthr);
                tsum[jfam[k]] += now() - a;
            }
            for (int f = 0; f < 2; f++)
                if (fn[f] && tsum[f] < fbest[f])
                    fbest[f] = tsum[f];
        }
        static const char *fname[2] = { "hc_*_up  K=320,  10240 rows",
                                        "hc_*_down K=10240, 320 rows" };
        double allb = 0, alls = 0;
        for (int f = 0; f < 2; f++) {
            if (!fn[f])
                continue;
            printf("%-33s %3ld tensors %6.0f MiB x%2d threads: %6.1f GB/s (%4.1f ms)\n",
                   fname[f], fn[f], fbytes[f] / 1048576.0, nthr,
                   fbytes[f] / fbest[f] / 1e9, fbest[f] * 1e3);
            allb += fbytes[f];
            alls += fbest[f];
        }
        if (alls > 0)
            printf("%-33s %3ld tensors %6.0f MiB in one walk  : %6.1f GB/s (%4.1f ms)\n",
                   "all of them", (long)nj, allb / 1048576.0, allb / alls / 1e9, alls * 1e3);
        fflush(stdout);
    }

    /* ---- ceiling check over the staged weights --------------------- */
    printf("\n== ceiling check ==\n");
    fflush(stdout);
    const void *p = NULL;
    {
        const qe_tensor *t = qe_tensor_by_name(m, "blk.0.attn_qkv.weight");
        if (t) p = qe_tensor_data(m, t);
    }
    size_t n = 64u << 20;
    double best = 0;
    if (!p) {
        printf("skipped: blk.0.attn_qkv.weight unavailable\n");
    } else {
        for (int r = 0; r < 4; r++) {
            double g = qe_measure_read_bw(p, n, nthr);
            if (g > best) best = g;
        }
    }
    double step = step_mib * 1048576.0;
    printf("stream read 64 MiB x%d threads : %6.1f GB/s\n", nthr, best);
    printf("Q8_0 GEMV, hottest tensor     : %6.1f GB/s\n", bwbest);
    printf("(that GEMV figure is L3-resident - the per-tensor table says which - "
           "so it divides the step into nothing)\n");
    if (best > 0) {
        printf("--> decode ceiling, stream rate: %5.1f tok/s (%.0f MiB/step)\n",
               best * 1e9 / step, step_mib);
        printf("--> with MTP shared-Q4_K_M @%.2f tok/step: %5.1f tok/s\n", DRAFT_TOK,
               best * 1e9 / (mtp_mib * 1048576.0));
    }
    fflush(stdout);

    qe_pool_shutdown();
    qe_close(m);
    free(x);
    free(y);
    printf("== done in %.1fs ==\n", now() - t0);
    return 0;
}
