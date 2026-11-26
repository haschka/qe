/* bench_decode: where a decode step's time goes, and what each decode-side switch
 * buys, in one load of the checkpoint.
 *
 * A step is ~800 dispatches of the pool over ~4 GiB of weights, and it runs at well
 * under the bus (88 ms against the ~55 ms the bytes cost at 76 GB/s). This instrument
 * splits that gap three ways per stage, using the pool's own counters
 * (qe_pool_stats): the time inside dispatches, the part of that the master spent
 * waiting for the slowest worker (the barrier tail), and the time outside any
 * dispatch, which is serial work while nineteen threads spin.
 *
 * Then it races arms: every arm is the same prefix, the same teacher-forced ids, the
 * same number of steps, interleaved across repetitions so a drifting clock moves every
 * arm, best of each. The last step's logits of each arm are compared with arm 0's, so
 * an arm that is faster for the wrong reason says so.
 *
 *   ./bin/bench_decode [--model PATH] [--mtp PATH] [--text FILE] [--prefix N]
 *                      [--steps N] [--reps N] [--arms a,b,c] [--fuse S]
 */
#define _GNU_SOURCE
#include "qe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static const char *stage_name(int s) {
    static const char *n[QE_ST_N] = { "embed", "block_proj", "block_core", "hc_attn", "hc_ffn",
                                      "moe_ffn", "ple", "head_mixer", "lm_head" };
    return s >= 0 && s < QE_ST_N ? n[s] : "?";
}

static long argmax(const float *l, long n) {
    long b = 0;
    for (long i = 1; i < n; i++)
        if (l[i] > l[b])
            b = i;
    return b;
}

static char *slurp(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, fp) != (size_t)n) {
        free(b);
        b = NULL;
    }
    if (b)
        b[n] = 0;
    fclose(fp);
    return b;
}

/* An arm is a name and the switches it sets. Every arm starts from the state the
 * open chose (`arm_reset`), so an arm only names what it changes. */
typedef struct {
    int decode_i4, head_copy, tier;
    int multi_i4; /* QE_I4_MULTI: rows of a pass that may read the dense copy */
    int stage_prefetch;
    int step_fuse;
    long dyn_kb;
    int step_i8;
    int chunk_dyn;
    int attn_batch;
    int rows_par;
    int delta_simd;
    int moe_dyn;
    int width;
} pass_state;

static pass_state open_state;

static void state_get(qe_forward *f, pass_state *s) {
    s->decode_i4 = qe_forward_get_decode_i4(f);
    s->head_copy = qe_forward_get_head_copy(f);
    s->tier = qe_forward_get_decode_tier(f);
    s->multi_i4 = qe_forward_get_multi_i4(f);
    s->stage_prefetch = qe_forward_get_prefetch(f);
    s->step_fuse = qe_forward_get_step_fuse(f);
    s->dyn_kb = qe_forward_get_dyn_kb(f);
    s->step_i8 = qe_forward_get_step_i8(f);
    s->chunk_dyn = qe_forward_get_chunk_dyn(f);
    s->attn_batch = qe_forward_get_attn_batch(f);
    s->rows_par = qe_forward_get_rows_par(f);
    s->delta_simd = qe_forward_get_delta_simd(f);
    s->moe_dyn = qe_forward_get_moe_dyn(f);
    s->width = qe_forward_get_chunk(f);
}

static void state_set(qe_forward *f, const pass_state *s) {
    qe_forward_set_decode_i4(f, s->decode_i4);
    qe_forward_set_head_copy(f, s->head_copy);
    qe_forward_set_decode_tier(f, s->tier);
    qe_forward_set_multi_i4(f, s->multi_i4);
    qe_forward_set_prefetch(f, s->stage_prefetch);
    qe_forward_set_step_fuse(f, s->step_fuse);
    qe_forward_set_dyn_kb(f, s->dyn_kb);
    qe_forward_set_step_i8(f, s->step_i8);
    qe_forward_set_chunk_dyn(f, s->chunk_dyn);
    qe_forward_set_attn_batch(f, s->attn_batch);
    qe_forward_set_rows_par(f, s->rows_par);
    qe_forward_set_delta_simd(f, s->delta_simd);
    qe_forward_set_moe_dyn(f, s->moe_dyn);
    qe_forward_set_chunk(f, s->width);
}

static int arm_apply(qe_forward *f, const char *name) {
    pass_state s = open_state;
    if (!strcmp(name, "ship")) {
    } else if (!strcmp(name, "stored")) {
        s.decode_i4 = 0;
        s.head_copy = 0;
    } else if (!strcmp(name, "nofuse")) {
        s.step_fuse = 0;
    } else if (!strcmp(name, "fuse")) {
        s.step_fuse = 1;
    } else if (!strcmp(name, "old")) { /* the engine before ideas.md 7.30 */
        s.step_fuse = 0;
        s.dyn_kb = 0;
        s.step_i8 = 0;
        s.chunk_dyn = 0;
    } else if (!strcmp(name, "i8")) {
        s.step_i8 = 1;
    } else if (!strcmp(name, "noi8")) {
        s.step_i8 = 0;
    } else if (!strncmp(name, "m", 1) && name[1] >= '0' && name[1] <= '9') {
        s.multi_i4 = atoi(name + 1); /* m2: windows of up to two rows read the IQ4_NL copy */
    } else if (!strncmp(name, "w", 1) && name[1] >= '0' && name[1] <= '9') {
        s.width = atoi(name + 1); /* w1024: the chunk width */
    } else if (!strcmp(name, "nomoedyn")) {
        s.moe_dyn = 0;
    } else if (!strcmp(name, "nodsimd")) {
        s.delta_simd = 0;
    } else if (!strcmp(name, "norows")) {
        s.rows_par = 0;
    } else if (!strcmp(name, "noab")) {
        s.attn_batch = 0;
    } else if (!strcmp(name, "nocdyn")) {
        s.chunk_dyn = 0;
    } else if (!strcmp(name, "nodyn")) {
        s.dyn_kb = 0;
    } else if (!strncmp(name, "dyn", 3)) {
        s.dyn_kb = atol(name + 3);
    } else if (!strcmp(name, "nopf")) {
        s.stage_prefetch = 0;
    } else if (!strcmp(name, "pf")) {
        s.stage_prefetch = 1;
    } else {
        return -1;
    }
    state_set(f, &s);
    return 0;
}

/* --timeline: one line per landed chunk of the profiled ingestion - where in the prompt,
 * and that chunk's own rate - so a long prompt's rate can be read as a function of depth. */
static double tl_last;
static long tl_prev;
static int tl_cb(void *user, long done, long total) {
    (void)user;
    const double t = now();
    printf("  chunk to %6ld of %ld: %4ld tokens %7.3f s %6.1f tok/s\n", done, total, done - tl_prev,
           t - tl_last, (double)(done - tl_prev) / (t - tl_last));
    fflush(stdout);
    tl_last = t;
    tl_prev = done;
    return 0;
}

int main(int argc, char **argv) {
    const char *path = "/mnt/data/data/ai-models/llama-cpp/llm/"
                       "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    const char *mtp_path = NULL, *text_path = "README.md";
    const char *arms_arg = "ship";
    int threads = QE_THREADS_AUTO, fuse = 0, prefix = 512, steps = 16, reps = 3, nll_n = 0, pp_n = 0, win = 0;
    int timeline = 0;
    const char *depths_arg = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "--mtp") && i + 1 < argc)
            mtp_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc)
            text_path = argv[++i];
        else if (!strcmp(argv[i], "--prefix") && i + 1 < argc)
            prefix = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--win") && i + 1 < argc)
            win = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pp") && i + 1 < argc)
            pp_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depths") && i + 1 < argc)
            depths_arg = argv[++i];
        else if (!strcmp(argv[i], "--timeline"))
            timeline = 1;
        else if (!strcmp(argv[i], "--nll") && i + 1 < argc)
            nll_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--arms") && i + 1 < argc)
            arms_arg = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fuse") && i + 1 < argc)
            fuse = atoi(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s [--model P] [--mtp P] [--text F] [--prefix N] [--steps N]"
                    " [--reps N] [--arms a,b] [--threads N] [--fuse S] [--pp N [--timeline]] [--depths a,b,...]\n",
                    argv[0]);
            return 2;
        }
    }
    if (fuse > 0)
        qe_fuse(fuse);
    qe_pool_init(threads, NULL, 0);

    char *text = slurp(text_path);
    if (!text) {
        fprintf(stderr, "cannot read %s\n", text_path);
        return 1;
    }
    printf("== open ==\n");
    fflush(stdout);
    qe_model *m = qe_open(path, QE_STAGE_RAM | QE_STAGE_STRICT);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    char terr[256] = { 0 };
    qe_tok *tok = qe_tok_open(m, terr, sizeof terr);
    if (!tok) {
        fprintf(stderr, "tokenizer: %s\n", terr);
        return 1;
    }
    qe_model *mh = NULL;
    if (mtp_path && !(mh = qe_open(mtp_path, QE_STAGE_RAM | QE_STAGE_STRICT))) {
        fprintf(stderr, "draft head %s: %s\n", mtp_path, qe_strerror());
        return 1;
    }
    long need = (long)prefix + (long)(steps > nll_n ? steps : nll_n) + 8 + 2L * 16 * (win > 0);
    if (pp_n + 8 > need)
        need = pp_n + 8;
    long depths[16];
    int nd = 0;
    if (depths_arg) {
        char *dbuf = strdup(depths_arg);
        for (char *t = strtok(dbuf, ","); t && nd < 16; t = strtok(NULL, ","))
            depths[nd++] = atol(t);
        free(dbuf);
        if (nd > 0 && depths[nd - 1] + steps + 8 > need)
            need = depths[nd - 1] + steps + 8;
    }
    qe_forward_opts fo = { .n_ctx = (int)(need + 64 > 4096 ? need + 64 : 4096),
                           .n_threads = threads,
                           .mtp_model = mh };
    char ferr[512] = { 0 };
    qe_forward *f = qe_forward_open(m, &fo, ferr, sizeof ferr);
    if (!f) {
        fprintf(stderr, "qe_forward_open: %s\n", ferr);
        return 1;
    }
    state_get(f, &open_state);
    const unsigned variant = qe_forward_variant(f);
    const long vocab = qe_forward_vocab(f);
    long cap = qe_tok_count(tok, text) + 8;
    int *ids = malloc((size_t)cap * sizeof *ids);
    long n = qe_tok_encode(tok, text, ids, cap);
    if (n < need) {
        fprintf(stderr, "%s is %ld tokens, need %ld\n", text_path, n, need);
        return 1;
    }
    printf("  %d threads, %ld tokens of %s, prefix %d, %d steps, best of %d\n", qe_pool_size(),
           n, text_path, prefix, steps, reps);
    fflush(stdout);

    char *arms_buf = strdup(arms_arg);
    const char *arms[16];
    int na = 0;
    for (char *t = strtok(arms_buf, ","); t && na < 16; t = strtok(NULL, ","))
        arms[na++] = t;
    double best[16];
    float *lg = malloc((size_t)vocab * sizeof *lg);
    float *lref = malloc((size_t)vocab * sizeof *lref);
    float *lfin = malloc((size_t)vocab * 16 * sizeof *lfin);
    for (int a = 0; a < na; a++)
        best[a] = 1e30;
    for (int a = 0; a < na; a++)
        if (arm_apply(f, arms[a])) {
            fprintf(stderr, "unknown arm %s\n", arms[a]);
            return 2;
        }

    /* --depths: one growing sequence, read a segment at a time up to each depth and then
     * `steps` decode steps taken there (the text's own next tokens, so the sequence stays
     * one text), each half profiled per stage with the attention core split three ways.
     * What it answers is what a long conversation pays per token as it grows. */
    if (nd > 0) {
        arm_apply(f, arms[0]);
        qe_forward_reset(f);
        qe_pool_stats_enable(1);
        qe_forward_reconfigure(f, variant, 1);
        long pos = 0;
        for (int k = 0; k < nd; k++) {
            const long d = depths[k];
            if (d <= pos)
                continue;
            qe_forward_diag d0, d1;
            qe_forward_diag_of(f, &d0);
            double t1 = now();
            if (qe_forward_prefill(f, ids + pos, d - pos, NULL)) {
                printf("== depth %ld: prefill refused ==\n", d);
                return 1;
            }
            double sec = now() - t1;
            qe_forward_diag_of(f, &d1);
            const long nt = d - pos;
            printf("== depth %ld: read %ld..%ld at %.1f tok/s (%.2f ms a token) ==\n", d, pos, d,
                   nt / sec, sec * 1e3 / nt);
            printf("  ingest ms/tok:");
            for (int st = 0; st < QE_ST_N; st++) {
                const double v = (d1.stage_seconds[st] - d0.stage_seconds[st]) * 1e3 / nt;
                if (v >= 0.005)
                    printf(" %s %.2f", stage_name(st), v);
            }
            printf("\n  ingest attention ms/tok: pre %.3f idx %.3f walk %.3f\n",
                   (d1.attn_pre_seconds - d0.attn_pre_seconds) * 1e3 / nt,
                   (d1.attn_idx_seconds - d0.attn_idx_seconds) * 1e3 / nt,
                   (d1.attn_walk_seconds - d0.attn_walk_seconds) * 1e3 / nt);
            pos = d;
            qe_forward_diag_of(f, &d0);
            t1 = now();
            for (int s2 = 0; s2 < steps; s2++)
                qe_forward_step(f, ids[pos + s2], lg);
            sec = (now() - t1) / steps;
            qe_forward_diag_of(f, &d1);
            pos += steps;
            printf("  step at %ld: %.2f ms (%.2f tok/s); ms/step:", d, sec * 1e3, 1.0 / sec);
            for (int st = 0; st < QE_ST_N; st++) {
                const double v = (d1.stage_seconds[st] - d0.stage_seconds[st]) * 1e3 / steps;
                if (v >= 0.005)
                    printf(" %s %.2f", stage_name(st), v);
            }
            uint64_t hsh = 1469598103934665603ull;
            for (size_t b = 0; b < (size_t)vocab * sizeof *lg; b++)
                hsh = (hsh ^ ((const unsigned char *)lg)[b]) * 1099511628211ull;
            printf("\n  logits after the steps: fnv %016llx, argmax %ld", (unsigned long long)hsh,
                   argmax(lg, vocab));
            printf("\n  step attention ms: idx %.3f walk %.3f\n",
                   (d1.attn_idx_seconds - d0.attn_idx_seconds) * 1e3 / steps,
                   (d1.attn_walk_seconds - d0.attn_walk_seconds) * 1e3 / steps);
            fflush(stdout);
        }
        qe_forward_reconfigure(f, variant, 0);
        qe_pool_stats_enable(0);
        return 0;
    }

    /* The profile pass of arm 0 first: per stage, ms a step, dispatches a step, pool
     * ms, barrier tail ms, and serial (outside the pool) ms. */
    for (int a = 0; a < na; a++) {
        arm_apply(f, arms[a]);
        qe_forward_reset(f);
        {
            const int prc = qe_forward_prefill(f, ids, prefix, NULL);
            if (prc) {
                qe_forward_diag dd;
                qe_forward_diag_of(f, &dd);
                printf("== %s: the prefix was refused, rc %d, %ld non-finite values ==\n", arms[a],
                       prc, dd.nonfinite);
                fflush(stdout);
                return 1;
            }
        }
        qe_forward_diag d0, d1;
        qe_pool_stats_enable(1);
        qe_forward_reconfigure(f, variant, 1);
        qe_forward_diag_of(f, &d0);
        qe_pool_stats p0, p1;
        qe_pool_stats_get(&p0);
        const double t1 = now();
        for (int s = 0; s < steps; s++)
            qe_forward_step(f, ids[prefix + s], lg);
        const double wall = (now() - t1) / steps;
        qe_pool_stats_get(&p1);
        qe_forward_diag_of(f, &d1);
        qe_forward_reconfigure(f, variant, 0);
        qe_pool_stats_enable(0);
        printf("== profile: %s, %.1f ms a step (profiled) ==\n", arms[a], wall * 1e3);
        printf("  %-11s %8s %6s %8s %8s %8s\n", "stage", "ms", "disp", "pool", "tail", "serial");
        double sum = 0.0;
        for (int st = 0; st < QE_ST_N; st++) {
            const double sec = (d1.stage_seconds[st] - d0.stage_seconds[st]) / steps;
            const double pool = (d1.stage_pool[st] - d0.stage_pool[st]) / steps;
            const double tail = (d1.stage_tail[st] - d0.stage_tail[st]) / steps;
            const double disp = (double)(d1.stage_disp[st] - d0.stage_disp[st]) / steps;
            sum += sec;
            if (sec < 1e-5)
                continue;
            printf("  %-11s %8.2f %6.0f %8.2f %8.2f %8.2f\n", stage_name(st), sec * 1e3, disp,
                   pool * 1e3, tail * 1e3, (sec - pool) * 1e3);
        }
        printf("  %-11s %8.2f %6.0f %8.2f %8.2f\n", "all", wall * 1e3,
               (double)(p1.dispatches - p0.dispatches) / steps,
               (p1.seconds - p0.seconds) / steps * 1e3, (p1.tail - p0.tail) / steps * 1e3);
        printf("  outside every stage %.2f ms, outside the pool %.2f ms a step\n",
               (wall - sum) * 1e3, (wall - (p1.seconds - p0.seconds) / steps) * 1e3);
        fflush(stdout);
    }

    for (int r = 0; r < reps; r++)
        for (int a = 0; a < na; a++) {
            arm_apply(f, arms[a]);
            qe_forward_reset(f);
            if (qe_forward_prefill(f, ids, prefix, NULL))
                return 1;
            const double t1 = now();
            for (int s = 0; s < steps; s++)
                if (qe_forward_step(f, ids[prefix + s], lg))
                    return 1;
            const double ms = (now() - t1) * 1e3 / steps;
            if (ms < best[a])
                best[a] = ms;
            if (r == 0)
                memcpy(lfin + (size_t)a * (size_t)vocab, lg, (size_t)vocab * sizeof *lg);
        }
    memcpy(lref, lfin, (size_t)vocab * sizeof *lref);
    printf("== arms: ms a step, best of %d ==\n", reps);
    for (int a = 0; a < na; a++) {
        const float *l = lfin + (size_t)a * (size_t)vocab;
        double md = 0.0, lo = 1e30, hi = -1e30;
        for (long i = 0; i < vocab; i++) {
            double dd = fabs((double)l[i] - (double)lref[i]);
            if (dd > md)
                md = dd;
            if (lref[i] < lo)
                lo = lref[i];
            if (lref[i] > hi)
                hi = lref[i];
        }
        printf("  %-10s %8.2f ms  %6.2f tok/s  %5.3fx  argmax %ld%s  max|dlogit| %.4f (%.4f of range)\n",
               arms[a], best[a], 1e3 / best[a], best[0] / best[a], argmax(l, vocab),
               argmax(l, vocab) == argmax(lref, vocab) ? " =" : " DIFFERS", md, md / (hi - lo));
    }
    fflush(stdout);

    /* Verify windows of `win` rows: the pass a k = win - 1 draft costs, both rows' logits
     * included, against a step of the same arm - the ratio speculation has to beat. */
    if (win > 1) {
        qe_ckpt *ck = qe_ckpt_new(f, win);
        float *wl = malloc((size_t)vocab * (size_t)win * sizeof *wl);
        if (!ck || !wl)
            return 1;
        const int nw = 12;
        double wbest[16];
        for (int a = 0; a < na; a++)
            wbest[a] = 1e30;
        for (int r = 0; r < reps; r++)
            for (int a = 0; a < na; a++) {
                arm_apply(f, arms[a]);
                qe_forward_reset(f);
                if (qe_forward_prefill(f, ids, prefix, NULL))
                    return 1;
                const double t1 = now();
                for (int w = 0; w < nw; w++) {
                    if (qe_forward_step_window(f, ids + prefix + (long)w * win, win, ck) ||
                        qe_forward_window_logits(f, win, wl))
                        return 1;
                }
                const double ms = (now() - t1) * 1e3 / nw;
                if (ms < wbest[a])
                    wbest[a] = ms;
            }
        /* Where a window's time goes, for arm 0: the same per-stage table as a step's. */
        {
            arm_apply(f, arms[0]);
            qe_forward_reset(f);
            if (qe_forward_prefill(f, ids, prefix, NULL))
                return 1;
            qe_forward_diag d0, d1;
            qe_pool_stats_enable(1);
            qe_forward_reconfigure(f, variant, 1);
            qe_forward_diag_of(f, &d0);
            const double t1 = now();
            for (int w = 0; w < nw; w++)
                if (qe_forward_step_window(f, ids + prefix + (long)w * win, win, ck) ||
                    qe_forward_window_logits(f, win, wl))
                    return 1;
            const double wall = (now() - t1) / nw;
            qe_forward_diag_of(f, &d1);
            qe_forward_reconfigure(f, variant, 0);
            qe_pool_stats_enable(0);
            printf("== window profile: %s, %d rows, %.1f ms a window ==\n", arms[0], win,
                   wall * 1e3);
            printf("  %-11s %8s %6s %8s %8s %8s\n", "stage", "ms", "disp", "pool", "tail",
                   "serial");
            for (int st = 0; st < QE_ST_N; st++) {
                const double sec = (d1.stage_seconds[st] - d0.stage_seconds[st]) / nw;
                const double pool = (d1.stage_pool[st] - d0.stage_pool[st]) / nw;
                const double tail = (d1.stage_tail[st] - d0.stage_tail[st]) / nw;
                const double disp = (double)(d1.stage_disp[st] - d0.stage_disp[st]) / nw;
                if (sec < 1e-5)
                    continue;
                printf("  %-11s %8.2f %6.0f %8.2f %8.2f %8.2f\n", stage_name(st), sec * 1e3,
                       disp, pool * 1e3, tail * 1e3, (sec - pool) * 1e3);
            }
            fflush(stdout);
        }
        printf("== arms: a verify window of %d rows with every row's logits, best of %d ==\n",
               win, reps);
        for (int a = 0; a < na; a++)
            printf("  %-10s %8.2f ms  = %.2f steps of this arm\n", arms[a], wbest[a],
                   wbest[a] / best[a]);
        fflush(stdout);
        free(wl);
        qe_ckpt_free(ck);
    }

    /* Ingestion: the profile of one prefill of `pp_n` tokens per arm, then the arms raced. */
    if (pp_n > 0) {
        for (int a = 0; a < na; a++) {
            arm_apply(f, arms[a]);
            qe_forward_reset(f);
            qe_forward_diag d0, d1;
            qe_pool_stats_enable(1);
            qe_forward_reconfigure(f, variant, 1);
            qe_forward_diag_of(f, &d0);
            qe_pool_stats pp0, pp1;
            qe_pool_stats_get(&pp0);
            const double t1 = now();
            if (timeline) {
                tl_last = t1;
                tl_prev = 0;
                qe_forward_set_progress(f, tl_cb, NULL);
            }
            const int prc = qe_forward_prefill(f, ids, pp_n, lg);
            qe_forward_set_progress(f, NULL, NULL);
            qe_pool_stats_get(&pp1);
            if (prc) {
                qe_forward_diag dd;
                qe_forward_diag_of(f, &dd);
                printf("== ingestion: %s REFUSED, rc %d, %ld non-finite values ==\n", arms[a], prc,
                       dd.nonfinite);
                qe_forward_reconfigure(f, variant, 0);
                qe_pool_stats_enable(0);
                continue;
            }
            const double wall = (now() - t1) / pp_n;
            qe_forward_diag_of(f, &d1);
            qe_forward_reconfigure(f, variant, 0);
            qe_pool_stats_enable(0);
            printf("== ingestion profile: %s, %d tokens, %.2f ms a token ==\n", arms[a], pp_n,
                   wall * 1e3);
            printf("  %-11s %8s %6s %8s %8s %8s\n", "stage", "ms/tok", "disp", "pool", "tail",
                   "serial");
            double psum = 0.0;
            for (int st = 0; st < QE_ST_N; st++)
                psum += (d1.stage_seconds[st] - d0.stage_seconds[st]) / pp_n;
            for (int st = 0; st < QE_ST_N; st++) {
                const double sec = (d1.stage_seconds[st] - d0.stage_seconds[st]) / pp_n;
                const double pool = (d1.stage_pool[st] - d0.stage_pool[st]) / pp_n;
                const double tail = (d1.stage_tail[st] - d0.stage_tail[st]) / pp_n;
                const double disp = (double)(d1.stage_disp[st] - d0.stage_disp[st]) / pp_n;
                if (sec < 1e-6)
                    continue;
                printf("  %-11s %8.3f %6.2f %8.3f %8.3f %8.3f\n", stage_name(st), sec * 1e3,
                       disp, pool * 1e3, tail * 1e3, (sec - pool) * 1e3);
            }
            printf("  %-11s %8.3f  (the per-token loops between stages, all serial)\n", "outside",
                   (wall - psum) * 1e3);
            if (pp1.item_capacity > pp0.item_capacity)
                printf("  stage-graph dispatches: the pool was busy %.0f %% of their thread-time"
                       " (%.2f of %.2f thread-ms a token)\n",
                       100.0 * (pp1.item_busy - pp0.item_busy) /
                           (pp1.item_capacity - pp0.item_capacity),
                       (pp1.item_busy - pp0.item_busy) / pp_n * 1e3,
                       (pp1.item_capacity - pp0.item_capacity) / pp_n * 1e3);
            fflush(stdout);
        }
        double pbest[16];
        for (int a = 0; a < na; a++)
            pbest[a] = 1e30;
        for (int r = 0; r < reps; r++)
            for (int a = 0; a < na; a++) {
                arm_apply(f, arms[a]);
                qe_forward_reset(f);
                const double t1 = now();
                if (qe_forward_prefill(f, ids, pp_n, lg)) {
                    pbest[a] = 1e30;
                    continue;
                }
                const double ms = (now() - t1) * 1e3 / pp_n;
                {
                    /* Each repetition with its wall-clock span, so a run beside another
                     * process (an iGPU walk, ideas.md 7.32.4) can be lined up with it. */
                    struct timespec rt;
                    clock_gettime(CLOCK_REALTIME, &rt);
                    printf("  rep %d %-10s %7.2f ms/tok, ended at %.1f, took %.1f s\n", r, arms[a],
                           ms, rt.tv_sec + 1e-9 * rt.tv_nsec, ms * pp_n / 1e3);
                    fflush(stdout);
                }
                if (ms < pbest[a])
                    pbest[a] = ms;
                if (r == 0)
                    memcpy(lfin + (size_t)a * (size_t)vocab, lg, (size_t)vocab * sizeof *lg);
            }
        printf("== arms: ingestion of %d tokens, ms a token, best of %d ==\n", pp_n, reps);
        for (int a = 0; a < na; a++) {
            const float *l = lfin + (size_t)a * (size_t)vocab;
            const int same = !memcmp(l, lfin, (size_t)vocab * sizeof *l);
            printf("  %-10s %7.2f ms/tok  %6.1f tok/s  %5.3fx  argmax %ld  %s\n", arms[a], pbest[a],
                   1e3 / pbest[a], pbest[0] / pbest[a], argmax(l, vocab),
                   same ? "logits byte-identical to arm 0" : "logits differ from arm 0");
        }
        fflush(stdout);
    }

    /* What each arm costs the answer: teacher-forced next-token loss over `nll_n` positions
     * after the same prefix, and how many of arm 0's greedy choices it keeps. */
    if (nll_n > 0) {
        long *top0 = malloc((size_t)nll_n * sizeof *top0);
        double nll0 = 0.0;
        printf("== arms: next-token loss over %d teacher-forced positions ==\n", nll_n);
        for (int a = 0; a < na; a++) {
            arm_apply(f, arms[a]);
            qe_forward_reset(f);
            if (qe_forward_prefill(f, ids, prefix, NULL))
                return 1;
            double nll = 0.0, worst = 0.0;
            long same = 0;
            const double t1 = now();
            for (int s = 0; s < nll_n; s++) {
                if (qe_forward_step(f, ids[prefix + s], lg))
                    return 1;
                double mx = lg[0];
                for (long i = 1; i < vocab; i++)
                    if (lg[i] > mx)
                        mx = lg[i];
                double se = 0.0;
                for (long i = 0; i < vocab; i++)
                    se += exp((double)lg[i] - mx);
                const double l = mx + log(se) - (double)lg[ids[prefix + s + 1]];
                nll += l;
                if (l > worst)
                    worst = l;
                const long am = argmax(lg, vocab);
                if (!a)
                    top0[s] = am;
                else if (am == top0[s])
                    same++;
            }
            const double ms = (now() - t1) * 1e3 / nll_n;
            if (!a)
                nll0 = nll / nll_n;
            printf("  %-10s nll %.4f  (%+.4f)  worst %.2f  top-1 same as arm 0: %ld of %d  %.1f ms\n",
                   arms[a], nll / nll_n, nll / nll_n - nll0, worst, a ? same : nll_n,
                   nll_n, ms);
            fflush(stdout);
        }
        free(top0);
    }
    qe_forward_close(f);
    return 0;
}
