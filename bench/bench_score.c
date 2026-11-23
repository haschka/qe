/* Teacher-forced loss as the oracle for "is the forward pass right".
   `make forward` asks whether a token is finite and roughly the right size, and
   that is where cheap checks stop: every way this pass can be wrong while still
   printing numbers - a group-broadcast read the other way round, a norm with
   epsilon in the wrong place, a conv reading pre-normalised activations - is
   finite, bounded, fast and nonsense. Loss is the only cheap thing that
   separates them, and perplexity of held-out text is the accepted form of it.

   One load, many measurements, because a load is four minutes and the model
   will not coexist with a second copy of itself. In one process this scores
   every variant the forward pass exposes, prints the depth-probe loss curve
   (what the head would have said at layer N, which localises the damage to a
   layer class), the greedy text per variant, and the per-stage wall clock of a
   step.

     make score                              # or:
     ./bin/bench_score --model <vol1.gguf> --threads auto --tokens 128

   The default corpus is text from this repository, which is not held out for a
   model trained on the internet, so the absolute number is meaningless. Its only
   job is to be much lower for the reading that is right than for the ones that
   are wrong, and a depth curve that falls with depth rather than wandering. For
   a number with meaning, --corpus a file of text the model family did train on
   (a reference engine's own output is the best available here): a working pass
   puts that in the single digits of nats per token, and nothing else in this
   file is worth as much.
*/
#include "qe.h"

#include "kern.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* A decoded control token printed to a log is one grep away from being pasted
 * into a prompt, and the vertical bars that spell one are the same ones that
 * break a tool call. Nothing about scoring needs them: the ids are printed
 * instead, and the bars come out of ordinary text. */
static void safe_puts(const char *s) {
    for (; *s; s++)
        fputc(*s == '|' ? '!' : *s, stdout);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static char *slurp(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (n <= 0) {
        fclose(fp);
        return NULL;
    }
    char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, fp) != (size_t)n) {
        free(b);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    b[n] = 0;
    *len = (size_t)n;
    return b;
}

/* Cross-entropy of a logits vector against one id, in one pass and without
 * touching the 248320 floats twice. */
static double xent(const float *lg, long n, int target) {
    float m = lg[0];
    for (long i = 1; i < n; i++)
        if (lg[i] > m)
            m = lg[i];
    double sum = 0.0;
    for (long i = 0; i < n; i++)
        sum += exp((double)(lg[i] - m));
    /* -log p(target) = m + log sum - lg[target]. The minus sign in front of the
     * log is not a typo waiting to happen, it is the whole measurement: with it
     * a uniform distribution scores minus the log of the vocabulary, the depth
     * curve comes out below zero, and every variant ranking is quietly
     * inverted. */
    double e = (double)m + log(sum) - (double)lg[target];
    /* A saturated logit makes m inf and the difference inf-inf, which is not a
     * number and is not a loss; a caller summing losses over a document has to
     * be able to tell those apart from a real one. */
    return isfinite(e) ? e : NAN;
}

static void top5(const qe_tok *tok, const float *lg, long n) {
    int idx[5] = { -1, -1, -1, -1, -1 };
    float val[5] = { -1e30f, -1e30f, -1e30f, -1e30f, -1e30f };
    for (long i = 0; i < n; i++)
        for (int k = 0; k < 5; k++)
            if (lg[i] > val[k]) {
                for (int j = 4; j > k; j--) {
                    val[j] = val[j - 1];
                    idx[j] = idx[j - 1];
                }
                val[k] = lg[i];
                idx[k] = (int)i;
                break;
            }
    printf("     top5:");
    char buf[64];
    for (int k = 0; k < 5; k++) {
        if (qe_tok_is_special(tok, idx[k])) {
            printf(" [%d] #%d", (int)val[k], idx[k]);
            continue;
        }
        long m = qe_tok_decode(tok, &idx[k], 1, buf, sizeof buf);
        printf(" [%d] \"", (int)val[k]);
        if (m > 0)
            safe_puts(buf);
        printf("\"");
    }
    printf("\n");
}

/* The next-token distribution, as ids and nats. An aggregate loss says a pass is
 * wrong; a reference engine's top-k for the same prefix says which one, because
 * it names the candidates this pass failed to rank. Written to a file rather than
 * only to stdout so a comparison against another engine's logprobs is a join and
 * not a reading task. */
static void logit_dump(FILE *fp, const qe_tok *tok, const float *lg, long n, long step, int k) {
    float m = lg[0];
    for (long i = 1; i < n; i++)
        if (lg[i] > m)
            m = lg[i];
    double sum = 0.0;
    for (long i = 0; i < n; i++)
        sum += exp((double)(lg[i] - m));
    const double lse = (double)m + log(sum);

    int idx[8];
    for (int i = 0; i < k; i++)
        idx[i] = -1;
    for (long i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            if (lg[i] > (idx[j] < 0 ? -1e30f : lg[idx[j]])) {
                for (int q = k - 1; q > j; q--)
                    idx[q] = idx[q - 1];
                idx[j] = (int)i;
                break;
            }
    printf("    logits step %ld: lse %.4f max %.4f\n", step, lse, (double)m);
    for (int j = 0; j < k; j++) {
        if (idx[j] < 0)
            continue;
        char buf[64] = { 0 };
        printf("      #%-7d %8.4f", idx[j], lse - (double)lg[idx[j]]);
        if (!qe_tok_is_special(tok, idx[j]) &&
            qe_tok_decode(tok, &idx[j], 1, buf, sizeof buf) > 0) {
            printf("  \"");
            safe_puts(buf);
            printf("\"");
        }
        printf("\n");
        if (fp)
            fprintf(fp, "%ld\t%d\t%.6f\n", step, idx[j], lse - (double)lg[idx[j]]);
    }
}

struct doc {
    int *ids;
    long n;
};

int main(int argc, char **argv) {
    const char *path = "/mnt/data/data/ai-models/llama-cpp/llm/"
                       "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int threads = QE_THREADS_AUTO, fuse = 1500, n_ctx = 4096, tokens = 128;
    int want_probe = 0, want_profile = 0;
    /* The width a lone token's weight walk asks for. `full` and 0 are the pass width, so
     * a run that names one and a run that does not are the same arithmetic twice, and the
     * nll column of the sweep below is what says so. */
    int walk_threads = 0;
    /* Prompts prefilled twice per arithmetic, to price the ingestion path the
     * token-at-a-time sweep below cannot reach: the integer expert walk, and the
     * grouped pass it sits on, which is the yardstick. */
    int ingest = 0;
    int gen = 24, nlogits = 0, nhidden = 0;
    const char *logits_out = NULL, *hidden_out = NULL;
    const char *prompt = "Explain, in one short paragraph, why a CPU inference "
                         "engine has to keep the weights resident in RAM.\n";
    unsigned sweep[32];
    int nsweep = 0;
    /* Text to score. The repository's own prose is convenient and is not held out
     * for anything, which is fine for ranking variants against each other. The
     * corpus that decides things is text the reference engine produced itself
     * (`tools/oracle_capture.py`), because its per-position logprobs go next to
     * the per-token nll printed below, and that is the difference between a loss
     * that ranks hypotheses and one that merely looks large. */
    const char *corpus[8];
    int ncorpus = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            /* "auto" = the engine's own choice (QE_THREADS_AUTO); a number is
             * honoured exactly, so an A/B of thread counts still works. */
            const char *t = argv[++i];
            threads = (*t == 'a' || *t == 'A') ? QE_THREADS_AUTO : atoi(t);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)
            n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--walk-threads") && i + 1 < argc) {
            const char *t = argv[++i];
            walk_threads = (*t == 'f' || *t == 'F' || *t == 'a' || *t == 'A') ? 0 : atoi(t);
        } else if (!strcmp(argv[i], "--fuse") && i + 1 < argc)
            fuse = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc)
            tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe"))
            want_probe = 1;
        else if (!strcmp(argv[i], "--ingest") && i + 1 < argc)
            ingest = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc)
            nlogits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits-out") && i + 1 < argc)
            logits_out = argv[++i];
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc)
            nhidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden-out") && i + 1 < argc)
            hidden_out = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc)
            prompt = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc)
            gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--profile"))
            want_profile = 1;
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) {
            if (ncorpus < 8)
                corpus[ncorpus++] = argv[++i];
            else
                i++;
        } else if (!strcmp(argv[i], "--variants") && i + 1 < argc) {
            char *list = strdup(argv[++i]), *s = list;
            while (*s && nsweep < 32) {
                sweep[nsweep++] = (unsigned)strtoul(s, &s, 0);
                if (*s == ',')
                    s++;
            }
            free(list);
        } else {
            fprintf(stderr,
                    "usage: %s [--model PATH] [--threads N] [--ctx N] [--fuse S]"
                    " [--walk-threads N]"
                    " [--tokens N] [--corpus FILE]... [--variants 0x0,0x1,...]"
                    " [--prompt TEXT] [--gen N] [--probe] [--profile] [--ingest N]"
                    " [--logits N] [--logits-out FILE] [--hidden N] [--hidden-out FILE]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!ncorpus) {
        corpus[ncorpus++] = "docs/DESIGN.md";
        corpus[ncorpus++] = "MANUAL.md";
        corpus[ncorpus++] = "src/tok.c";
    }
    if (!nsweep) {
        /* The committed reading first, because the depth probes and the stage
         * profile run on the first entry: those describe the engine as shipped,
         * not the alternative. The rest are the committed reading plus one
         * hypothesis at a time, so a difference names exactly one bit. */
        static const unsigned d[] = { QE_VARIANT_DEFAULT,
                                      0x00,
                                      QE_VARIANT_DEFAULT | QE_VARIANT_L2NORM_ROOT,
                                      QE_VARIANT_DEFAULT | QE_VARIANT_PLE_PRENORM,
                                      QE_VARIANT_DEFAULT | QE_VARIANT_NO_PLE };
        nsweep = 5;
        for (int i = 0; i < nsweep; i++)
            sweep[i] = d[i];
    }
    /* Line buffered: the interesting part of this output is written after a
     * five minute load and a ten minute sweep, and a block-buffered stdout that
     * is only flushed by normal exit means a fuse, a kill, or a hang takes the
     * numbers with it. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (fuse > 0)
        qe_fuse(fuse);
    /* Without this the pool answers every dispatch by running the slice on the
     * caller, and the whole sweep measures one core. */
    qe_pool_init(threads, NULL, 0);
    printf("  pool %d threads on %d cpus\n", qe_pool_size(), qe_cpus());

    printf("== open ==\n");
    fflush(stdout);
    qe_model *m = qe_open(path, QE_STAGE_RAM | QE_STAGE_STRICT);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    printf("  weights %s (%.1f GiB), %s\n", qe_model_stage_method(m),
           (double)qe_model_stage_bytes(m) / 1073741824.0,
           qe_model_stage_pinned(m) ? "pinned" : "reclaimable");
    /* Every number below is a wall clock, and on a part with two kinds of core
     * a wall clock is a statement about how work was divided. */
    {
        char sbuf[128];
        printf("  slicing: %s\n", qe_slice_summary(sbuf, sizeof sbuf));
    }
    FILE *lfp = logits_out ? fopen(logits_out, "w") : NULL;
    if (logits_out && !lfp)
        fprintf(stderr, "cannot write %s\n", logits_out);
    /* The trunk as bytes. A second engine's per-token hidden states are the only
     * thing finer than a logit that says where a pass stops agreeing with it, so
     * they are written in step order, one row of the trunk width, no header. */
    FILE *hfp = hidden_out ? fopen(hidden_out, "w") : NULL;
    if (hidden_out && !hfp)
        fprintf(stderr, "cannot write %s\n", hidden_out);
    float *hv = hfp ? malloc(16384 * sizeof *hv) : NULL;
    long hwidth = 0;

    char terr[256] = { 0 };
    qe_tok *tok = qe_tok_open(m, terr, sizeof terr);
    if (!tok) {
        fprintf(stderr, "tokenizer: %s\n", terr);
        return 1;
    }

    /* ---- corpus: the configured text, tokenised once ---- */
    struct doc docs[8];
    int ndocs = 0;
    long total = 0;
    for (int d = 0; d < ncorpus && ndocs < 8; d++) {
        size_t len = 0;
        char *text = slurp(corpus[d], &len);
        if (!text) {
            printf("  note: no %s, not scored\n", corpus[d]);
            continue;
        }
        long cap = qe_tok_count(tok, text) + 8;
        int *ids = malloc((size_t)cap * sizeof *ids);
        long n = ids ? qe_tok_encode(tok, text, ids, cap) : 0;
        free(text);
        if (n < 8) {
            free(ids);
            continue;
        }
        if (total + n > tokens)
            n = tokens - total;
        docs[ndocs].ids = ids;
        docs[ndocs].n = n;
        ndocs++;
        total += n;
        if (total >= tokens)
            break;
    }
    if (!ndocs) {
        fprintf(stderr, "no corpus: run this from the repository root\n");
        return 1;
    }
    printf("  corpus: %d documents, %ld tokens\n", ndocs, total);
    fflush(stdout);

    qe_forward_opts fo = { .n_ctx = n_ctx, .n_threads = threads, .walk_threads = walk_threads };
    char ferr[512] = { 0 };
    qe_forward *f = qe_forward_open(m, &fo, ferr, sizeof ferr);
    if (!f) {
        fprintf(stderr, "qe_forward_open: %s\n", ferr);
        return 1;
    }
    const long vocab = qe_forward_vocab(f);
    /* The same two numbers the server banner prints, because "one sequence at 8k
     * costs 270 MiB" is a claim about this graph and this context width, and it
     * should be checkable with the command that already loads the model. It is how
     * docs/ARCH.md 5's table got reconciled with the engine: the attention cache is
     * 12.2 KiB of the 19.7 KiB per token, and the indexer's keys are the rest. */
    {
        size_t st_total = 0, st_token = 0;
        qe_forward_state_bytes(f, &st_total, &st_token);
        printf("  state:   %d tokens of context = %.0f MiB, %.1f KiB/token, kv %s\n", n_ctx,
               (double)st_total / 1048576.0, (double)st_token / 1024.0,
               qe_forward_kv_q8(f) ? "q8_0" : "f32");
    }
    float *logits = malloc((size_t)vocab * sizeof *logits);
    if (!logits)
        return 1;

    /* ------------------------------------------------ variant sweep */
    printf("\n== teacher-forced loss ==\n");
    printf("  %-6s %9s %9s %7s %6s  %s\n", "variant", "nll", "ppl", "nonfin", "ms/tok",
           "accuracy");
    fflush(stdout);
    for (int v = 0; v < nsweep; v++) {
        printf("  variant 0x%02x ...\n", sweep[v]);
        fflush(stdout);
        qe_forward_reconfigure(f, sweep[v], 0);
        qe_forward_reset(f);
        double loss = 0.0;
        long scored = 0, hits = 0, steps = 0;
        int shown = 0;
        double t0 = now();
        for (int d = 0; d < ndocs; d++) {
            if (d)
                qe_forward_reset(f); /* one sequence per document */
            for (long i = 0; i + 1 < docs[d].n; i++) {
                if (qe_forward_step(f, docs[d].ids[(int)i], logits) != 0)
                    continue;
                steps++;
                double e = xent(logits, vocab, docs[d].ids[(int)i + 1]);
                if (!isfinite(e))
                    continue;
                loss += e;
                scored++;
                int best = 0;
                for (long j = 1; j < vocab; j++)
                    if (logits[j] > logits[best])
                        best = (int)j;
                hits += best == docs[d].ids[(int)i + 1];
                /* Per token, for the first document: the reference engine's own
                 * logprobs for the same text are the only absolute comparison
                 * available, and they are per token, so an aggregate cannot be
                 * put next to them. */
                if (v == 0 && nlogits > 0 && steps <= nlogits)
                    logit_dump(lfp, tok, logits, vocab, steps, 8);
                if (v == 0 && hfp && steps <= nhidden) {
                    long got = qe_forward_hidden(f, hv, 16384);
                    if (got > 0) {
                        fwrite(hv, sizeof *hv, (size_t)got, hfp);
                        hwidth = got;
                    }
                }
                if (steps <= 512)
                    printf("    %3ld  nll %8.4f  top1 #%-8d want #%-8d%s%s\n", steps, e, best,
                           docs[d].ids[(int)i + 1], best == docs[d].ids[(int)i + 1] ? "" : "  miss",
                           v ? "  (ablated)" : "");
                if (!shown++) {
                    printf("  variant 0x%02x, first step:\n", sweep[v]);
                    top5(tok, logits, vocab);
                }
            }
        }
        double secs = now() - t0;
        if (lfp)
            fflush(lfp);
        if (hfp) {
            fclose(hfp);
            hfp = NULL;
            printf("  wrote %d hidden vectors of %ld floats\n", nhidden, hwidth);
        }
        if (!scored) {
            printf("  0x%02x     %9s %9s\n", sweep[v], "nan", "nan");
            continue;
        }
        qe_forward_diag dg;
        qe_forward_diag_of(f, &dg);
        printf("  0x%02x  %9.4f %9.4f %7ld %6.1f  %ld/%ld\n", sweep[v], loss / scored,
               exp(loss / scored), dg.nonfinite,
               steps ? 1000.0 * secs / (double)steps : 0.0, hits, scored);
        fflush(stdout);
    }

    /* ------------------------------------------------ ingestion arithmetic
     * Teacher forcing is one token at a time, and the grouped expert pass that
     * feeds a prompt does not exist at T=1, so nothing above sees the arithmetic a
     * prompt is ingested with - which is exactly what the integer expert walk
     * changes (src/moe.c, QE_I8_MOE). Each window is therefore prefilled twice,
     * once per arithmetic, and what the two leave at the end of the prompt is
     * printed: the loss on the token that really follows, whether the greedy choice
     * moved, and the worst move anywhere in the head priced against the head\'s own
     * range - the bound `make forward` uses for the same comparison on one prompt.
     * A window is a prompt, not a corpus position: ingestion is a prompt, and the
     * nll column says what the recode costs the next token\'s distribution rather
     * than what it costs a single logit. The grouped pass gets the same table in
     * the same load, because "21 % of the head" means nothing until it is read
     * against the change this engine already ships. */
    if (ingest > 0) {
        const long W = 512;
        /* The sweep above ended on the last variant it walked, and an ablation is
         * not the reading this section's two legs are about. */
        qe_forward_reconfigure(f, QE_VARIANT_DEFAULT, 0);
        float *la = malloc((size_t)vocab * sizeof *la);
        float *lb = malloc((size_t)vocab * sizeof *lb);
        if (!la || !lb)
            return 1;
        /* Four A/Bs over the same windows in the same load: the arithmetries under
         * review, and the schedule change that already ships. A recode is only cheap
         * relative to something, and that something is on this machine already, so the
         * yardstick table is what the others are read against rather than a second number
         * to be impressed by. The last arm is a panel against a panel - one scale per
         * activation row against one per 32 values - because that change replaces the panel
         * the first arm ships with, and not the float walk. */
        static const char *ab_title[4] = {
            "the f32 expert walk against the integer walk",
            "serial expert passes against the grouped pass, which is the yardstick",
            "the f32 projection walk against the integer walk",
            "the per-block activation panel against one scale per row"
        };
        static const char *ab_off[4] = { "nll f32", "nll serial", "nll f32", "nll panel" };
        static const char *ab_on[4] = { "nll i8", "nll grouped", "nll i8", "nll row" };
        static const char *ab_who[4] = { "the integer walk", "the grouped pass",
                                         "the integer projection walk", "the row scale" };
        static const char *ab_nm[4] = { "the integer expert walk", 0,
                                        "the integer projection walk", "the row-scaled panel" };
        double a_sum[4] = { 0, 0, 0, 0 }, a_worst[4] = { 0, 0, 0, 0 };
        double a_gap[4] = { 0, 0, 0, 0 }, a_scale[4] = { 0, 0, 0, 0 };
        long a_done[4] = { 0, 0, 0, 0 }, a_moved[4] = { 0, 0, 0, 0 };
        for (int ab = 0; ab < 4; ab++) {
            printf("\n== ingestion A/B: %s ==\n", ab_title[ab]);
            printf("  %6s %7s %9s %9s %9s %11s  %s\n", "window", "prompt", ab_off[ab],
                   ab_on[ab], "delta", "head gap", "greedy choice");
            fflush(stdout);
            for (int d = 0; d < ndocs && a_done[ab] < ingest; d++) {
                for (long at = 0; at + W + 1 <= docs[d].n && a_done[ab] < ingest; at += W) {
                    const int want = docs[d].ids[(int)(at + W)];
                    long top[2] = { -1, -1 };
                    double nl[2] = { 0, 0 }, gap = 0, scale = 0;
                    const int was_mg = qe_forward_get_moe_grouped(f);
                    const int was_pj = qe_forward_get_proj_i8(f);
                    const int was_rs = qe_i8_rowscale();
                    /* Leg 0 is this engine without the change under test, leg 1
                     * with it. For the yardstick table the integer walk is held
                     * off in both legs, so what is left is the schedule alone. */
                    for (int leg = 0; leg < 2; leg++) {
                        if (ab == 0) {
                            qe_moe_set_int8(leg);
                        } else if (ab == 1) {
                            qe_moe_set_int8(0);
                            qe_forward_set_moe_grouped(f, leg);
                        } else if (ab == 3) {
                            qe_moe_set_int8(1);
                            qe_i8_set_rowscale(leg);
                        } else {
                            qe_moe_set_int8(0);
                            qe_forward_set_proj_i8(f, leg);
                        }
                        qe_forward_reset(f);
                        if (qe_forward_prefill(f, docs[d].ids + at, W, leg ? lb : la)) {
                            fprintf(stderr, "prefilling window %ld failed\n", a_done[ab]);
                            return 1;
                        }
                    }
                    qe_moe_set_int8(-1);
                    qe_i8_set_rowscale(was_rs);
                    qe_forward_set_moe_grouped(f, was_mg);
                    qe_forward_set_proj_i8(f, was_pj);
                    qe_forward_reset(f);
                    for (int leg = 0; leg < 2; leg++) {
                        const float *lg = leg ? lb : la;
                        long best = 0;
                        for (long j = 1; j < vocab; j++)
                            if (lg[j] > lg[best])
                                best = j;
                        top[leg] = best;
                        nl[leg] = xent(lg, vocab, want);
                    }
                    for (long j = 0; j < vocab; j++) {
                        const double g = fabs((double)la[j] - (double)lb[j]);
                        if (g > gap)
                            gap = g;
                        const double mag = fabs((double)la[j]) > fabs((double)lb[j])
                                               ? fabs((double)la[j])
                                               : fabs((double)lb[j]);
                        if (mag > scale)
                            scale = mag;
                    }
                    const double d_nll = nl[1] - nl[0];
                    a_moved[ab] += top[0] != top[1];
                    if (fabs(d_nll) > a_worst[ab])
                        a_worst[ab] = fabs(d_nll);
                    a_sum[ab] += fabs(d_nll);
                    /* Kept as a fraction, because the bound is a fraction: one
                     * prompt's head is worth 20 logits and another's 60, and only
                     * the ratio says whether the two arithmetics still agree about
                     * the head. */
                    if (scale > 0 && (a_scale[ab] == 0 || gap / scale > a_gap[ab] / a_scale[ab])) {
                        a_gap[ab] = gap;
                        a_scale[ab] = scale;
                    }
                    a_done[ab]++;
                    printf("  %6ld %7ld %9.4f %9.4f %+9.4f %6.2f/%-4.0f  #%ld%s\n",
                           a_done[ab] - 1, at, nl[0], nl[1], d_nll, gap, scale, top[0],
                           top[0] == top[1] ? "" : ", moved");
                    fflush(stdout);
                    if (top[0] != top[1])
                        printf("         %s takes #%ld where the other leg took #%ld\n",
                               ab_who[ab], top[1], top[0]);
                }
            }
            if (!a_done[ab]) {
                printf("  note: no window of %ld tokens fits in the corpus, nothing asserted\n",
                       W);
                continue;
            }
            printf("  %ld windows of %ld tokens: next-token loss moves by %.4f nats at worst"
                   " (%.4f mean), the greedy choice moved in %ld of %ld, worst head gap"
                   " %.2f against a head of %.0f (%.0f%% of its range, %s half)\n",
                   a_done[ab], W, a_worst[ab], a_sum[ab] / (double)a_done[ab], a_moved[ab],
                   a_done[ab], a_gap[ab], a_scale[ab],
                   a_scale[ab] > 0 ? 100.0 * a_gap[ab] / a_scale[ab] : 0.0,
                   (a_scale[ab] > 0 && a_gap[ab] < a_scale[ab] / 2.0) ? "inside" : "outside");
            fflush(stdout);
        }
        for (int ab = 0; ab < 4; ab++) {
            if (!a_done[ab])
                continue;
            printf("  verdict: %s half the head's range, the bound the two ingestion legs of"
                   " `make forward` hold a change of schedule to. The sweep above cannot price"
                   " this change either way: it steps the corpus one token at a time, and the"
                   " grouped expert pass has no one-token case.\n",
                   (a_scale[ab] > 0 && a_gap[ab] < a_scale[ab] / 2.0) ? "inside" : "outside");
            if (!a_done[1] || ab == 1)
                continue;
            const double m0 = a_sum[ab] / (double)a_done[ab];
            const double m1 = a_sum[1] / (double)a_done[1];
            printf("  yardstick: over these windows the grouped pass, which ships and was"
                   " accepted on exactly this kind of evidence, moves the next token's loss"
                   " by %.4f nats mean and %.4f worst and the greedy choice %ld of %ld times;"
                   " %s moves it %.4f and %.4f and the choice %ld of %ld times."
                   " Both are single-sample columns: the number that decides is the choice"
                   " count and the head gap, not the mean.\n",
                   m1, a_worst[1], a_moved[1], a_done[1], ab_nm[ab], m0, a_worst[ab],
                   a_moved[ab], a_done[ab]);
        }
        free(la);
        free(lb);
        fflush(stdout);
    }

    /* ------------------------------------------------- the two attention walks
     * One walk of a KV group's rows serving twelve query heads, against one walk
     * per query head: the same rows, the same products, the sums formed in another
     * order and a softmax carried in registers instead of a score array. Scored
     * both ways over the same documents in the load that is already open, because
     * the claim on the table is that the loss did not move, and the number that
     * says so has to come from the same weights, the same tokens and the same
     * minute. Per-position nll and top-1 from the first pass are kept so the second
     * can be reported as a difference rather than as a second aggregate that might
     * coincide by accident. */
    {
        double *base_nll = malloc((size_t)(total + 1) * sizeof *base_nll);
        int *base_top = malloc((size_t)(total + 1) * sizeof *base_top);
        long nbase = 0;
        printf("\n== attention walk: one pass per KV group against one per query head ==\n");
        printf("  %-22s %9s %9s %7s %6s  %s\n", "walk", "nll", "ppl", "nonfin",
               "ms/tok", "accuracy");
        for (int w = 0; w < 2; w++) {
            const int fused = w == 0;
            qe_forward_set_attn_fused(f, fused);
            qe_forward_reset(f);
            double loss = 0.0;
            long scored = 0, hits = 0, steps = 0;
            double worst_d = 0.0;
            long moved = 0, k = 0;
            const double t0 = now();
            for (int d = 0; d < ndocs; d++) {
                if (d)
                    qe_forward_reset(f);
                for (long i = 0; i + 1 < docs[d].n; i++) {
                    if (qe_forward_step(f, docs[d].ids[(int)i], logits) != 0)
                        continue;
                    steps++;
                    const double e = xent(logits, vocab, docs[d].ids[(int)i + 1]);
                    if (!isfinite(e))
                        continue;
                    int best = 0;
                    for (long j = 1; j < vocab; j++)
                        if (logits[j] > logits[best])
                            best = (int)j;
                    /* Both walks see the same positions in the same order, so the
                     * index is the alignment; `steps` is not, because a step can
                     * fail to produce logits in one walk and not the other. */
                    if (w == 0) {
                        if (base_nll && base_top && nbase < total) {
                            base_nll[nbase] = e;
                            base_top[nbase] = best;
                            nbase++;
                        }
                    } else if (k < nbase) {
                        const double d2 = fabs(e - base_nll[k]);
                        if (d2 > worst_d)
                            worst_d = d2;
                        moved += best != base_top[k];
                    }
                    loss += e;
                    scored++;
                    k++;
                    hits += best == docs[d].ids[(int)i + 1];
                }
            }
            const double secs = now() - t0;
            qe_forward_diag dg;
            qe_forward_diag_of(f, &dg);
            printf("  %-22s %9.4f %9.4f %7ld %6.1f  %ld/%ld\n",
                   fused ? "one pass per group" : "one pass per head",
                   scored ? loss / scored : 0.0, scored ? exp(loss / scored) : 0.0,
                   dg.nonfinite, steps ? 1000.0 * secs / (double)steps : 0.0, hits, scored);
            if (w == 1)
                printf("  %-22s %9s %9s %7s %6s  worst %.1e nats, %ld of %ld top-1s moved\n",
                       "difference", "", "", "", "", worst_d, moved, scored);
            fflush(stdout);
        }
        qe_forward_set_attn_fused(f, 1);
        free(base_nll);
        free(base_top);
    }

    /* ------------------------------------- the walks a decode step alone reads
     * A step walks a dense matrix once, so its time is that matrix's bytes over the bus
     * and a smaller format is nearly paid for by its byte ratio (ideas.md 4.1). It is also
     * the one arithmetic decision in this engine that changes the numbers a *generated*
     * token is made of while leaving an ingested prompt alone, because every copy below is
     * selected by a width test - so the ingestion tables above cannot see it and the loss of
     * a teacher-forced walk is the only thing that can. Scored arm by arm over the same
     * documents in the same load, per position, so every arm after the first is a
     * difference and not a second aggregate: bytes per token is the claim on the left, the
     * top-1 count is the claim that says whether it is still the same model.
     *
     * An arm that asks for a copy this open did not stage says so beside its own name: the
     * getters report what was built, which is how a log tells "the switch made no
     * difference" from "there was nothing for it to change". */
    struct decode_arm {
        const char *name;
        int dq;
        int tier;
        int head;
    };
    /* The dense leg is one speed-up bought with loss, so its two named refinements are
     * arms of the same table rather than a second experiment: the mixers and the trunk's
     * ends are where a 4.5-bit row is most likely to be the answer's fault, and holding
     * them at the file's own Q8_0 is what it costs to find out. */
    const struct decode_arm arms[] = {
        { "as stored", 0, 0, QE_HEAD_STORED },
        { "IQ4 dense: all", 1, QE_DCLASS_HC | QE_DCLASS_EDGE, QE_HEAD_STORED },
        { "IQ4 dense: no mixers", 1, QE_DCLASS_EDGE, QE_HEAD_STORED },
        { "IQ4 dense: lean", 1, QE_DCLASS_PLAIN, QE_HEAD_STORED },
        { "Q8_0 head", 0, 0, QE_HEAD_Q8 },
        { "all + Q8 head", 1, QE_DCLASS_HC | QE_DCLASS_EDGE, QE_HEAD_Q8 },
        { "lean + Q8 head", 1, QE_DCLASS_PLAIN, QE_HEAD_Q8 },
    };
    const int n_arms = (int)(sizeof arms / sizeof *arms);
    const int dq0 = qe_forward_get_decode_i4(f);
    const int tier0 = qe_forward_get_decode_tier(f);
    const int head0 = qe_forward_get_head_copy(f);
    double *arm_ms = calloc((size_t)n_arms, sizeof *arm_ms);
    double *arm_nll = calloc((size_t)n_arms, sizeof *arm_nll);
    {
        double *base_nll = malloc((size_t)(total + 1) * sizeof *base_nll);
        int *base_top = malloc((size_t)(total + 1) * sizeof *base_top);
        long nbase = 0;
        printf("\n== decode walk: a step's dense rows and head against their copies ==\n");
        /* Whether the head's copies exist is not what the setting says a step should
         * read, so ask by asking for one: the setter answers with the reading this open
         * actually has, which is QE_HEAD_STORED when a copy was never staged. */
        qe_forward_set_head_copy(f, QE_HEAD_I4);
        const int head_staged = qe_forward_get_head_copy(f) == QE_HEAD_I4;
        qe_forward_set_head_copy(f, head0);
        printf("  note: dense copies %s, reading %s, head copies %s\n",
               dq0 ? "staged" : "not staged",
               tier0 == (QE_DCLASS_HC | QE_DCLASS_EDGE)
                   ? "all of them by default"
                   : tier0 == QE_DCLASS_EDGE ? "all but the mixers by default"
                             : tier0 == QE_DCLASS_HC ? "all but the trunk's ends by default"
                                                     : "neither class by default",
               head_staged ? "staged" : "not staged");
        printf("  %-22s %9s %9s %7s %6s  %s\n", "what a step reads", "nll", "ppl", "nonfin",
               "ms/tok", "accuracy");
        for (int w = 0; w < n_arms; w++) {
            qe_forward_set_decode_i4(f, arms[w].dq);
            qe_forward_set_decode_tier(f, arms[w].tier);
            qe_forward_set_head_copy(f, arms[w].head);
            char label[64];
            snprintf(label, sizeof label, "%s%s", arms[w].name,
                     (arms[w].dq && !qe_forward_get_decode_i4(f)) ||
                             (arms[w].head != QE_HEAD_STORED &&
                              qe_forward_get_head_copy(f) != arms[w].head)
                         ? " (unstaged)"
                         : "");
            qe_forward_reset(f);
            double loss = 0.0;
            long scored = 0, hits = 0, steps = 0;
            double worst_d = 0.0;
            long moved = 0, k = 0;
            const double t0 = now();
            for (int d = 0; d < ndocs; d++) {
                if (d)
                    qe_forward_reset(f);
                for (long i = 0; i + 1 < docs[d].n; i++) {
                    if (qe_forward_step(f, docs[d].ids[(int)i], logits) != 0)
                        continue;
                    steps++;
                    const double e = xent(logits, vocab, docs[d].ids[(int)i + 1]);
                    if (!isfinite(e))
                        continue;
                    int best = 0;
                    for (long j = 1; j < vocab; j++)
                        if (logits[j] > logits[best])
                            best = (int)j;
                    if (w == 0) {
                        if (base_nll && base_top && nbase < total) {
                            base_nll[nbase] = e;
                            base_top[nbase] = best;
                            nbase++;
                        }
                    } else if (k < nbase) {
                        const double d2 = fabs(e - base_nll[k]);
                        if (d2 > worst_d)
                            worst_d = d2;
                        moved += best != base_top[k];
                    }
                    loss += e;
                    scored++;
                    k++;
                    hits += best == docs[d].ids[(int)i + 1];
                }
            }
            const double secs = now() - t0;
            qe_forward_diag dg;
            qe_forward_diag_of(f, &dg);
            const double ms = steps ? 1000.0 * secs / (double)steps : 0.0;
            if (arm_ms)
                arm_ms[w] = ms;
            if (arm_nll)
                arm_nll[w] = scored ? loss / scored : 0.0;
            printf("  %-22s %9.4f %9.4f %7ld %6.1f  %ld/%ld\n", label,
                   scored ? loss / scored : 0.0, scored ? exp(loss / scored) : 0.0,
                   dg.nonfinite, ms, hits, scored);
            if (w)
                printf("  %-22s %+9.4f %9s %7s %5.2fx  worst %.1e nats, %ld of %ld top-1s"
                       " moved\n",
                       "difference", scored ? loss / scored - arm_nll[0] : 0.0, "", "",
                       (arm_ms && arm_ms[0] > 0.0 && ms > 0.0) ? arm_ms[0] / ms : 0.0,
                       worst_d, moved, scored);
            fflush(stdout);
        }
        qe_forward_set_decode_i4(f, dq0);
        qe_forward_set_decode_tier(f, tier0);
        qe_forward_set_head_copy(f, head0);
        free(base_nll);
        free(base_top);
    }
    free(arm_ms);
    free(arm_nll);

    /* --------------------------------------------------------- greedy text */
    /* Loss ranks variants; only the text says whether any of them is a model.
     * Both are cheap once the weights are resident and expensive before that,
     * so they live in the same run. */
    long pcap = gen > 0 ? qe_tok_count(tok, prompt) + 8 : 0;
    int *pids = pcap > 0 ? malloc((size_t)pcap * sizeof *pids) : NULL;
    long nprompt = pids ? qe_tok_encode(tok, prompt, pids, (long)pcap) : 0;
    if (nprompt > 0 && nprompt + gen + 2 < n_ctx) {
        const int eos = qe_tok_eos(tok);
        printf("\n== greedy text: %d tokens after a %ld-token prompt ==\n", gen, nprompt);
        fflush(stdout);
        for (int v = 0; v < nsweep; v++) {
            qe_forward_reconfigure(f, sweep[v], 0);
            qe_forward_reset(f);
            printf("  0x%02x | ", sweep[v]);
            fflush(stdout);
            int failed = 0;
            /* The prompt's logits have one reader, the token generated after it,
             * so the head is run once for the whole prompt rather than once per
             * token of it. bench_forward asserts that this lands where feeding the
             * same tokens one at a time lands. */
            if (qe_forward_prefill(f, pids, nprompt, logits) != 0)
                failed = 1;
            for (int g = 0; g < gen && !failed; g++) {
                int best = 0;
                for (long j = 1; j < vocab; j++)
                    if (logits[j] > logits[best])
                        best = (int)j;
                if (best == eos)
                    break;
                char buf[64];
                long dn = qe_tok_decode(tok, &best, 1, buf, sizeof buf);
                if (dn > 0)
                    safe_puts(buf);
                if (qe_forward_step(f, best, logits) != 0)
                    failed = 1;
            }
            printf("%s\n", failed ? " [step failed]" : "");
            fflush(stdout);
        }
    } else if (gen > 0) {
        printf("\n  note: no greedy text, prompt did not encode\n");
        fflush(stdout);
    }
    free(pids);

    /* ------------------------------------------------ depth probes */
    if (want_probe) {
        static const int layers[] = { 0, 1, 2, 3, 5, 7, 11, 17, 23, 35, 47 };
        const int np = (int)(sizeof layers / sizeof *layers);
        float *pbuf = malloc((size_t)np * (size_t)vocab * sizeof *pbuf);
        if (pbuf) {
            qe_forward_reconfigure(f, sweep[0], 0);
            qe_forward_probe_set(f, layers, np, pbuf);
            qe_forward_reset(f);
            double *acc = calloc((size_t)np + 1, sizeof *acc);
            long n = 0;
            for (int d = 0; d < ndocs; d++) {
                if (d)
                    qe_forward_reset(f);
                for (long i = 0; i + 1 < docs[d].n; i++) {
                    if (qe_forward_step(f, docs[d].ids[(int)i], logits) != 0)
                        continue;
                    double last = xent(logits, vocab, docs[d].ids[(int)i + 1]);
                    if (!isfinite(last))
                        continue;
                    for (int p = 0; p < np; p++) {
                        double e = xent(pbuf + (size_t)p * (size_t)vocab, vocab,
                                        docs[d].ids[(int)i + 1]);
                        acc[p] += isfinite(e) ? e : 0.0;
                    }
                    acc[np] += last;
                    n++;
                }
            }
            printf("\n== depth: nll of the head run after layer N (variant 0x%02x, %ld steps) ==\n",
                   sweep[0], n);
            for (int p = 0; p < np; p++)
                printf("  after %2d  %8.4f\n", layers[p], n ? acc[p] / (double)n : 0.0);
            printf("  final     %8.4f\n", n ? acc[np] / (double)n : 0.0);
            fflush(stdout);
            /* The forward pass keeps the layer list and the logits buffer, and
             * both belong to the caller: clearing them here is the difference
             * between an optional phase and a use-after-free in whichever phase
             * runs next. */
            qe_forward_probe_set(f, NULL, 0, NULL);
            free(pbuf);
            free(acc);
        }
    }

    /* ------------------------------------------------ stage profile */
    if (want_profile) {
        static const char *names[] = { "embed",      "block_proj", "block_core", "hc_attn",
                                       "hc_ffn",     "moe_ffn",    "ple",        "head_mixer",
                                       "lm_head" };
        qe_forward_reconfigure(f, sweep[0], 1);
        qe_forward_reset(f);
        int steps = 0;
        for (int d = 0; d < ndocs && steps < 4; d++)
            for (long i = 0; i < docs[d].n && steps < 4; i++, steps++)
                if (qe_forward_step(f, docs[d].ids[(int)i], logits) != 0)
                    break;
        qe_forward_diag dg;
        qe_forward_diag_of(f, &dg);
        printf("\n== where a step goes (variant 0x%02x, %d steps) ==\n", sweep[0], steps);
        double tot = 0.0;
        for (int s = 0; s < QE_ST_N; s++)
            tot += dg.stage_seconds[s];
        for (int s = 0; s < QE_ST_N; s++)
            printf("  %-11s %8.1f ms %5.1f%%\n", names[s],
                   1000.0 * dg.stage_seconds[s] / (double)(steps ? steps : 1),
                   tot > 0.0 ? 100.0 * dg.stage_seconds[s] / tot : 0.0);
        printf("  %-11s %8.1f ms\n", "sum", 1000.0 * tot / (double)(steps ? steps : 1));
        fflush(stdout);
    }

    int bad = 0;
    printf("\n");
    printf("%s\n", bad ? "FAIL" : "done");
    return bad;
}
