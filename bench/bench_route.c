/* How many distinct experts do T consecutive tokens actually touch?

   The grouped multi-row MoE pass (qe_moe_forward_multi) exists to make the
   answer to that question matter: it reads the distinct experts of a window
   once instead of top_k per row. bench_moe measured that pass against synthetic
   routing, and the honest result was break-even at uniform routing - grouping
   is a byte-saving mechanism, and its payoff is exactly the routing overlap,
   which synthetic draws can only bracket. Real overlap needs three things this
   machine now has: a tokenizer that produces the checkpoint's own ids, the
   checkpoint's own embedding rows, and the checkpoint's own router matrices.

   What is faithful here and what is not, stated once and printed again:

     - Token ids are real (qe_tok_encode over the file's BPE).
     - Embedding rows are real (token_embd.weight, dequantised here).
     - Layer 0's MoE router input is norm(embedding) to within the first
       hyper-connection mix, so layer 0's row is the real thing.
     - Every other layer is measured against the same normalised embedding,
       which has the right scale and statistics but is not that layer's actual
       activations. Read those rows as "a realistic hidden state", not as the
       trace. What the spread across layers does show is whether overlap is a
       property of depth or of content.

   Nothing in here reads an expert's weights: 5 MiB of router per layer against
   a few thousand tokens is a few seconds of disk, while the expert pool is
   53 GiB. */
#include "qe.h"
#include "kern.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + t.tv_nsec * 1e-9;
}

/* Text is ASCII on purpose and only its token count is ever printed: the
 * vocabulary contains chat delimiters whose byte sequences must not be echoed
 * by a benchmark. The mix is deliberate - prose, source, markup, repetition and
 * a question - because the whole question is whether overlap depends on the
 * genre. */
static const char *CORPUS[] = {
    "The decode step is memory bound, which means the only optimizations that "
    "matter are the ones that move fewer bytes. A kernel that computes the same "
    "dot product while reading half the weights is worth more than a kernel that "
    "reads all of them twice as fast, and a policy that keeps the hot experts "
    "resident is worth more than either. This is why the budget is written in "
    "gigabytes per token and not in floating point operations per second. The "
    "corollary is uncomfortable: most of a sparse model's weights are read by "
    "almost nobody, and the cost of that arrangement is paid in page faults "
    "rather than in flops. When the pool of experts is fifty three gigabytes and "
    "the machine has one hundred twenty eight, the kernel's notion of what "
    "deserves to stay resident is a load balancing decision whether anyone "
    "designed it that way or not.",
    "static int qe_moe_select(const qe_moe *m, const float *logits, qe_route *out) {\n"
    "    const long n = m->d.n_experts;\n"
    "    for (long i = 0; i < n; i++) {\n"
    "        const float v = logits[i];\n"
    "        long slot = topk_slot(out, v);\n"
    "        if (slot < 0) continue;\n"
    "        for (long j = TOPK - 1; j > slot; j--) out[j] = out[j - 1];\n"
    "        out[slot].logit = v;\n"
    "        out[slot].expert = (int32_t) i;\n"
    "    }\n"
    "    return topk_renorm(m, out);\n"
    "}\n",
    "{\"model\": \"Qwen3.8-Flash-Next\", \"quant\": \"IQ4_XS\", \"volumes\": 3, "
    "\"tensors\": 1224, \"context\": 262144, \"experts\": 512, \"active\": 10, "
    "\"ple\": {\"ngram_size\": 3, \"heads\": 16, \"table_rows\": 320001536}, "
    "\"hc\": {\"count\": 4, \"low_rank\": 320}, \"indexer\": {\"top_k\": 2048, "
    "\"compress_ratio\": 4}}\n",
    "The decode step is memory bound. The decode step is memory bound. "
    "The decode step is memory bound. The decode step is memory bound. "
    "Read fewer bytes. Read fewer bytes. Read fewer bytes. Read fewer bytes. "
    "same expert again. same expert again. same expert again. same expert again.\n",
    "Is the n-gram local context embedding and expert routing fully optimized "
    "for this machine, and if the model fits in RAM why is throughput still "
    "limited by the disk? Please be specific about page faults and about which "
    "bytes each token actually reads.\n",
};
#define N_CORPUS ((int) (sizeof(CORPUS) / sizeof(CORPUS[0])))

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *) a, y = *(const uint32_t *) b;
    return x < y ? -1 : x > y ? 1 : 0;
}
static int cmp_u64_desc(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* Distinct experts expected when routing has no overlap at all: a token takes
 * top_k of n_experts uniformly, so one expert escapes that token with
 * probability 1 - top_k/n and escapes T independent tokens with (1-top_k/n)^T. */
static double uniform_distinct(long n_experts, int top_k, int T) {
    return (double) n_experts *
           (1.0 - pow(1.0 - (double) top_k / (double) n_experts, (double) T));
}

/* Exact top-K by ascending kept logit: the same selection qe_moe_select makes,
 * and softmax would not change the winner set. */
static void topk_exact(const float *logits, long n_experts, int topk, int16_t *dst, float *kept) {
    for (int s = 0; s < topk; s++) {
        dst[s] = -1;
        kept[s] = -INFINITY;
    }
    for (long e = 0; e < n_experts; e++) {
        float v = logits[e];
        long slot = -1;
        for (int s = 0; s < topk; s++)
            if (v > kept[s]) {
                slot = s;
                break;
            }
        if (slot < 0)
            continue;
        for (int s = topk - 1; s > slot; s--) {
            kept[s] = kept[s - 1];
            dst[s] = dst[s - 1];
        }
        kept[slot] = v;
        dst[slot] = (int16_t) e;
    }
}

static int cmp_i16(const void *a, const void *b) {
    return (int) *(const int16_t *) a - (int) *(const int16_t *) b;
}

/* Sets, not order: two routers that picked the same ten experts in a different
 * order run the same ten matrices. */
static int same_set(const int16_t *a, const int16_t *b, int n) {
    int16_t s[16], t[16];
    if (n > 16)
        return 0;
    for (int i = 0; i < n; i++) {
        s[i] = a[i];
        t[i] = b[i];
    }
    qsort(s, (size_t) n, sizeof *s, cmp_i16);
    qsort(t, (size_t) n, sizeof *t, cmp_i16);
    return memcmp(s, t, (size_t) n * sizeof *s) == 0;
}
static double uniform_jaccard(int top_k, long n_experts) {
    double inter = (double) top_k * (double) top_k / (double) n_experts;
    return inter / (2.0 * (double) top_k - inter);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                                : "/mnt/data/data/ai-models/llama-cpp/llm/"
                                  "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int deadline = argc > 2 ? atoi(argv[2]) : 300;
    int max_tokens = argc > 3 ? atoi(argv[3]) : 1200;
    const char **files = (const char **) (argc > 4 ? argv + 4 : NULL);
    int n_files = argc > 4 ? argc - 4 : 0;
    long max_layers = argc > 5 ? atol(argv[5]) : 0;
    qe_fuse(deadline);

    qe_model *m = qe_open(path, QE_M_NONE);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    char terr[256] = { 0 };
    qe_tok *tok = qe_tok_open(m, terr, sizeof terr);
    if (!tok) {
        fprintf(stderr, "tokenizer: %s\n", terr);
        qe_close(m);
        return 1;
    }
    char tdesc[224];
    qe_tok_describe(tok, tdesc, sizeof tdesc);

    const char *arch = "qwen4exp";
    qe_model_get_str(m, "general.architecture", &arch);
    long long L = 0, E = 0, TOPK = 0, H = 0;
    char key[128];
    snprintf(key, sizeof key, "%s.block_count", arch);
    qe_model_get_int(m, key, &L);
    snprintf(key, sizeof key, "%s.expert_count", arch);
    qe_model_get_int(m, key, &E);
    snprintf(key, sizeof key, "%s.expert_used_count", arch);
    qe_model_get_int(m, key, &TOPK);
    snprintf(key, sizeof key, "%s.embedding_length", arch);
    qe_model_get_int(m, key, &H);
    double eps = 1e-6;
    snprintf(key, sizeof key, "%s.rms_epsilon", arch);
    qe_model_get_float(m, key, &eps);
    if (L < 1 || E < 2 || TOPK < 1 || H < 1 || TOPK > 64) {
        fprintf(stderr, "unreadable shape: layers=%lld experts=%lld top_k=%lld width=%lld\n", L, E,
                TOPK, H);
        return 1;
    }

    printf("== routing overlap on real weights (%s) ==\n", arch);
    printf("tokenizer   %s\n", tdesc);
    printf("shape       %lld layers, %lld experts, top-%lld, width %lld, rms eps %g\n\n", L, E,
           TOPK, H, eps);
    fflush(stdout);

    /* ---- tokenise ---------------------------------------------------- */
    char **fb = calloc((size_t) (n_files + 1), sizeof *fb);
    int n_texts = N_CORPUS;
    const char **texts = malloc((size_t) (N_CORPUS + n_files + 1) * sizeof *texts);
    for (int c = 0; c < N_CORPUS; c++)
        texts[c] = CORPUS[c];
    for (int i = 0; i < n_files; i++) {
        FILE *f = fopen(files[i], "rb");
        if (!f) {
            fprintf(stderr, "cannot read %s: %s\n", files[i], strerror(errno));
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long flen = ftell(f);
        fseek(f, 0, SEEK_SET);
        fb[i] = malloc((size_t) flen + 1);
        if (!fb[i] || fread(fb[i], 1, (size_t) flen, f) != (size_t) flen) {
            fprintf(stderr, "short read on %s\n", files[i]);
            return 1;
        }
        fclose(f);
        fb[i][flen] = 0;
        texts[n_texts++] = fb[i];
    }
    int nn = 0, cap = 8192;
    int *ids = malloc((size_t) cap * sizeof *ids);
    long encoded = 0;
    for (int c = 0; c < n_texts && nn < max_tokens; c++) {
        long need = qe_tok_count(tok, texts[c]);
        if (need <= 0)
            continue;
        encoded += need;
        if (nn + need > cap) {
            cap = (int) (nn + need) * 3 / 2 + 256;
            ids = realloc(ids, (size_t) cap * sizeof *ids);
        }
        long got = qe_tok_encode(tok, texts[c], ids + nn, cap - nn);
        if (got < 0) {
            fprintf(stderr, "encode failed for corpus %d\n", c);
            return 1;
        }
        printf("corpus %d    %6ld tokens  %7zu chars  %.2f bytes/token\n", c, got,
               strlen(texts[c]), (double) strlen(texts[c]) / (double) got);
        fflush(stdout);
        nn += (int) got;
        if (nn > max_tokens)
            nn = max_tokens;
    }
    if (nn < 64) {
        fprintf(stderr, "only %d tokens; too few to measure windows\n", nn);
        return 1;
    }
    printf("stream      %d tokens (%ld encoded%s)\n", nn, encoded,
           nn < encoded ? ", truncated to the cap" : "");

    /* ---- gather the embedding rows, once per distinct id -------------- */
    uint32_t *uniq = malloc((size_t) nn * sizeof *uniq);
    long n_uniq = 0;
    for (int i = 0; i < nn; i++)
        uniq[n_uniq++] = (uint32_t) ids[i];
    qsort(uniq, (size_t) n_uniq, sizeof *uniq, cmp_u32);
    n_uniq = 1;
    for (long i = 1; i < nn; i++)
        if (uniq[i] != uniq[n_uniq - 1])
            uniq[n_uniq++] = uniq[i];

    long n_vocab = qe_tok_n_vocab(tok);
    int32_t *slot_of = malloc((size_t) n_vocab * sizeof *slot_of);
    for (long i = 0; i < n_vocab; i++)
        slot_of[i] = -1;
    for (long u = 0; u < n_uniq; u++)
        slot_of[uniq[u]] = (int32_t) u;

    const qe_tensor *te = qe_tensor_by_name(m, "token_embd.weight");
    if (!te || te->type != QE_Q8_0 || (long) te->ne[0] != H) {
        fprintf(stderr, "token_embd.weight: %s\n", te ? "unexpected shape or dtype" : "missing");
        return 1;
    }
    const qe_block_q8_0 *emb_base = (const qe_block_q8_0 *) qe_tensor_data(m, te);
    if (!emb_base) {
        fprintf(stderr, "token_embd.weight is not mapped\n");
        return 1;
    }
    /* Ascending id order is not a detail: it turns the gather into one walk of
     * the region instead of a random seek per token, which on spinning media is
     * the difference between seconds and minutes. */
    const long nb = H / 32;
    float *pool = malloc((size_t) n_uniq * (size_t) H * sizeof *pool);
    float *tmp = aligned_alloc(64, (size_t) H * sizeof *tmp);
    if (!pool || !tmp) {
        fprintf(stderr, "alloc failed (%ld ids x %lld wide)\n", n_uniq, H);
        return 1;
    }
    double t0 = now();
    double abs_sum = 0, abs_max = 0;
    long non_finite = 0;
    for (long u = 0; u < n_uniq; u++) {
        const qe_block_q8_0 *b = emb_base + (size_t) uniq[u] * (size_t) nb;
        double ss = 0;
        for (long k = 0; k < nb; k++) {
            float d = qe_half_to_f32(b[k].d);
            for (int j = 0; j < 32; j++) {
                float v = d * (float) b[k].qs[j];
                tmp[k * 32 + j] = v;
                ss += (double) v * v;
            }
        }
        float sc = (float) (1.0 / sqrt(ss / (double) H + eps));
        if (!isfinite(sc))
            non_finite++;
        float *dst = pool + (size_t) u * (size_t) H;
        for (long k = 0; k < H; k++) {
            dst[k] = tmp[k] * sc;
            double a = fabs((double) dst[k]);
            abs_sum += a;
            if (a > abs_max)
                abs_max = a;
        }
    }
    double gather_s = now() - t0;
    const float **row = malloc((size_t) nn * sizeof *row);
    for (int i = 0; i < nn; i++) {
        int id = ids[i];
        if (id < 0 || (long) id >= n_vocab || slot_of[id] < 0) {
            fprintf(stderr, "token id %d has no embedding row\n", id);
            return 1;
        }
        row[i] = pool + (size_t) slot_of[id] * (size_t) H;
    }
    printf("embedding   token_embd.weight Q8_0: %ld distinct ids of %d tokens, "
           "%.2f MiB gathered and RMS-normalised in %.1f s\n",
           n_uniq, nn, (double) n_uniq * H * 4 / 1048576.0, gather_s);
    printf("            normalised rows: mean |x| %.4f, max |x| %.4f, %ld rows with a "
           "non-finite scale\n", abs_sum / ((double) n_uniq * (double) H), abs_max, non_finite);
    fflush(stdout);

    /* ---- route -------------------------------------------------------- */
    int16_t *sets = malloc((size_t) L * (size_t) nn * (size_t) TOPK * sizeof *sets);
    float *logits = malloc((size_t) E * sizeof *logits);
    /* The bf16 router control: the same matrix rounded the way a bf16 checkpoint
     * holds it, and the answer to the only question that matters before halving
     * 240 MiB of the step - does any token get a different set of experts. */
    void *router_bf16 = aligned_alloc(64, ((size_t) E * (size_t) H * 2 + 63) & ~(size_t) 63);
    float *logits_b = malloc((size_t) E * sizeof *logits_b);
    int16_t set_b[16];
    float kept_b[16];
    long bf16_tokens = 0, bf16_layers = 0, bf16_diff_tokens = 0, bf16_diff_experts = 0;
    uint64_t *hist_all = calloc((size_t) E, sizeof *hist_all);
    uint8_t *seen = malloc((size_t) E);
    float *kept = malloc((size_t) TOPK * sizeof *kept);
    long short_rows = 0;
    if (!sets || !logits || !hist_all || !seen || !kept || !router_bf16 || !logits_b) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    const int TS[] = { 2, 4, 8, 16 };
    const int NT = (int) (sizeof(TS) / sizeof(TS[0]));
    double sum_dist[8] = { 0 }, sum_jacc = 0, sum_same = 0;
    int n_rows = 0;

    printf("\nRouter inputs are RMS-normalised embedding rows: layer 0 is faithful, later layers "
           "are realistic-but-not-the-true-activations.\n\n");
    printf("%-6s %10s %8s %8s %8s %8s %9s %10s\n", "layer", "router", "T=2", "T=4", "T=8", "T=16",
           "jaccard", "same tok");
    fflush(stdout);
    for (long lay = 0; lay < L && (!max_layers || lay < max_layers); lay++) {
        char name[QE_MAX_NAME];
        snprintf(name, sizeof name, "blk.%ld.ffn_gate_inp.weight", lay);
        const qe_tensor *rt = qe_tensor_by_name(m, name);
        if (!rt || rt->type != QE_F32 || (long) rt->ne[0] != H || (long) rt->ne[1] != E) {
            printf("%-6ld   skipped (%s)\n", lay, rt ? "unexpected shape or dtype" : "missing");
            continue;
        }
        const float *W = (const float *) qe_tensor_data(m, rt);
        if (!W) {
            printf("%-6ld   skipped (not mapped)\n", lay);
            continue;
        }
        qe_bf16_from_f32(W, router_bf16, (size_t) E * (size_t) H);
        bf16_layers++;
        for (int i = 0; i < nn; i++) {
            qe_gemv_f32(W, row[i], logits, E, H, H);
            if (lay == 0 && i == 0) { /* the numbers behind the table, once */
                float lo = logits[0], hi = logits[0];
                double mean = 0;
                long nan = 0;
                for (long e = 0; e < E; e++) {
                    if (!(logits[e] == logits[e]))
                        nan++;
                    if (logits[e] < lo)
                        lo = logits[e];
                    if (logits[e] > hi)
                        hi = logits[e];
                    mean += (double) logits[e];
                }
                printf("\nlayer-0 logits over %lld experts: min %.4f max %.4f mean %.4f, "
                       "NaN %ld\n", (long long) E, (double) lo, (double) hi,
                       mean / (double) E, nan);
                fflush(stdout);
            }
            int16_t *dst = sets + ((size_t) lay * (size_t) nn + (size_t) i) * (size_t) TOPK;
            topk_exact(logits, E, (int) TOPK, dst, kept);
            qe_gemv_bf16(router_bf16, row[i], logits_b, E, H, H);
            topk_exact(logits_b, E, (int) TOPK, set_b, kept_b);
            bf16_tokens++;
            if (!same_set(dst, set_b, TOPK)) {
                bf16_diff_tokens++;
                for (long s = 0; s < TOPK; s++) {
                    int in = 0;
                    for (long t = 0; t < TOPK; t++)
                        in |= dst[s] == set_b[t];
                    bf16_diff_experts += !in;
                }
            }
            for (long s = 0; s < TOPK; s++)
                if (dst[s] >= 0)
                    hist_all[dst[s]]++;
                else
                    short_rows++;
        }

        double dist[8];
        for (int ti = 0; ti < NT; ti++) {
            int T = TS[ti];
            long total = 0, windows = 0;
            for (int start = 0; start + T <= nn; start += T) {
                memset(seen, 0, (size_t) E);
                long d = 0;
                for (int t = start; t < start + T; t++) {
                    const int16_t *s =
                        sets + ((size_t) lay * (size_t) nn + (size_t) t) * (size_t) TOPK;
                    for (long j = 0; j < TOPK; j++)
                        if (s[j] >= 0 && !seen[s[j]]) {
                            seen[s[j]] = 1;
                            d++;
                        }
                }
                total += d;
                windows++;
            }
            dist[ti] = windows ? (double) total / (double) windows : 0.0;
            sum_dist[ti] += dist[ti];
        }

        /* Adjacent tokens: Jaccard over the two top-k sets. Independent routing
         * of two tokens holding 10 of 512 expects an intersection of 100/512. */
        double jac = 0;
        long nji = 0;
        for (int i = 0; i + 1 < nn; i++) {
            const int16_t *a = sets + ((size_t) lay * (size_t) nn + (size_t) i) * (size_t) TOPK;
            const int16_t *b = a + (size_t) TOPK;
            memset(seen, 0, (size_t) E);
            for (long j = 0; j < TOPK; j++)
                if (a[j] >= 0)
                    seen[a[j]] = 1;
            long inter = 0;
            for (long j = 0; j < TOPK; j++)
                inter += b[j] >= 0 && seen[b[j]] != 0;
            jac += (double) inter / (double) (2 * TOPK - inter);
            nji++;
        }
        jac = nji ? jac / (double) nji : 0.0;
        sum_jacc += jac;

        double same = 0;
        if (lay > 0) {
            for (int i = 0; i < nn; i++) {
                const int16_t *a =
                    sets + ((size_t) (lay - 1) * (size_t) nn + (size_t) i) * (size_t) TOPK;
                const int16_t *b =
                    sets + ((size_t) lay * (size_t) nn + (size_t) i) * (size_t) TOPK;
                memset(seen, 0, (size_t) E);
                for (long j = 0; j < TOPK; j++)
                    if (a[j] >= 0)
                        seen[a[j]] = 1;
                for (long j = 0; j < TOPK; j++)
                    same += b[j] >= 0 && seen[b[j]] != 0;
            }
            same /= (double) nn;
            sum_same += same;
        }
        n_rows++;
        printf("%-6ld %7.1f MiB %8.1f %8.1f %8.1f %8.1f %9.3f %10.2f\n", lay,
               (double) rt->bytes / 1048576.0, dist[0], dist[1], dist[2], dist[3], jac, same);
        fflush(stdout);
    }
    if (!n_rows) {
        fprintf(stderr, "no router tensor had the expected shape\n");
        return 1;
    }
    if (max_layers && max_layers < L) {
        printf("\npartial run (%ld of %lld layers): window and popularity numbers are for those "
               "layers only\n", max_layers, L);
        return 0;
    }

    double mdist[8];
    if (short_rows)
        printf("\n!! %ld of %ld routed rows had fewer than %lld experts (a NaN logit); "
               "these numbers are not trustworthy\n", short_rows, (long) n_rows * nn, TOPK);
    /* Repeated content lowers the distinct-expert count all by itself: two
     * occurrences of one id route to one expert set, and that is a real saving
     * for the grouped pass, but it is not the same claim as "the router
     * reuses experts across different tokens". Both baselines are printed so
     * the reader can tell them apart. */
    double uid[8];
    {
        int32_t *idmark = malloc((size_t) n_vocab * sizeof *idmark);
        for (long i = 0; i < n_vocab; i++)
            idmark[i] = -1;
        if (!idmark) {
            fprintf(stderr, "alloc failed for the id-mark table\n");
            return 1;
        }
        for (int ti = 0; ti < NT; ti++) {
            int T = TS[ti];
            long total = 0, windows = 0;
            int32_t mark = 0;
            for (int start = 0; start + T <= nn; start += T, mark++) {
                long u = 0;
                for (int t = start; t < start + T; t++)
                    if (idmark[ids[t]] != mark) {
                        idmark[ids[t]] = mark;
                        u++;
                    }
                total += u;
                windows++;
            }
            uid[ti] = windows ? (double) total / (double) windows : 0.0;
        }
        free(idmark);
    }
    printf("\ndistinct token ids per window of T: %.2f  %.2f  %.2f  %.2f (of 2 4 8 16 rows)\n",
           uid[0], uid[1], uid[2], uid[3]);
    printf("independent per distinct id: %6.1f %6.1f %6.1f %6.1f\n",
           uniform_distinct(E, (int) TOPK, (int) (uid[0] + 0.5)),
           uniform_distinct(E, (int) TOPK, (int) (uid[1] + 0.5)),
           uniform_distinct(E, (int) TOPK, (int) (uid[2] + 0.5)),
           uniform_distinct(E, (int) TOPK, (int) (uid[3] + 0.5)));
    for (int ti = 0; ti < NT; ti++)
        mdist[ti] = sum_dist[ti] / n_rows;
    printf("\nmean                     %8.1f %8.1f %8.1f %8.1f %9.3f %10.2f\n", mdist[0], mdist[1],
           mdist[2], mdist[3], sum_jacc / n_rows, n_rows > 1 ? sum_same / (n_rows - 1) : 0.0);
    printf("no overlap at all would    %8.1f %8.1f %8.1f %8.1f %9.3f %10.2f\n",
           uniform_distinct(E, (int) TOPK, 2), uniform_distinct(E, (int) TOPK, 4),
           uniform_distinct(E, (int) TOPK, 8), uniform_distinct(E, (int) TOPK, 16),
           uniform_jaccard((int) TOPK, E), (double) TOPK * TOPK / (double) E);
    printf("overlap                    %7.1f%% %6.1f%% %6.1f%% %6.1f%%\n",
           100.0 * (1.0 - mdist[0] / (2.0 * TOPK)), 100.0 * (1.0 - mdist[1] / (4.0 * TOPK)),
           100.0 * (1.0 - mdist[2] / (8.0 * TOPK)), 100.0 * (1.0 - mdist[3] / (16.0 * TOPK)));

    /* ---- bytes, which is the only currency here ----------------------- */
    long expert_bytes = 0;
    {
        char n1[QE_MAX_NAME], n2[QE_MAX_NAME], n3[QE_MAX_NAME];
        snprintf(n1, sizeof n1, "blk.0.ffn_gate_exps.weight");
        snprintf(n2, sizeof n2, "blk.0.ffn_up_exps.weight");
        snprintf(n3, sizeof n3, "blk.0.ffn_down_exps.weight");
        const qe_tensor *t1 = qe_tensor_by_name(m, n1), *t2 = qe_tensor_by_name(m, n2),
                        *t3 = qe_tensor_by_name(m, n3);
        if (t1 && t2 && t3 && t1->expert_stride && t2->expert_stride && t3->expert_stride)
            expert_bytes = (long) t1->expert_stride + (long) t2->expert_stride +
                           (long) t3->expert_stride;
    }
    if (expert_bytes) {
        printf("\nRouted-expert bytes per layer, mean routing (gate+up+down of one expert = "
               "%.1f MiB):\n",
               (double) expert_bytes / 1048576.0);
        for (int ti = 0; ti < NT; ti++) {
            int T = TS[ti];
            double serial = (double) T * TOPK * expert_bytes / 1048576.0;
            double grouped = mdist[ti] * expert_bytes / 1048576.0;
            printf("  T=%-2d  one row at a time %7.1f MiB   grouped %7.1f MiB   %.1f%% fewer\n", T,
                   serial, grouped, 100.0 * (1.0 - grouped / serial));
        }
    } else {
        printf("\n(expert tensor strides unavailable; byte translation skipped)\n");
    }

    /* ---- the router itself is 4.3% of the step, so: can it be half as wide? */
    if (bf16_layers && bf16_tokens) {
        double mib4 = (double) bf16_layers * (double) E * (double) H * 4 / 1048576.0;
        long slots = bf16_tokens * TOPK;
        printf("\nRouter width over %ld layer-token pairs (%ld layers, %d experts x %lld wide):\n",
               bf16_tokens, bf16_layers, (int) E, H);
        printf("  f32 rows    %6.1f MiB per token across those layers\n", mib4);
        printf("  bf16 shadow %6.1f MiB per token, and %.1f MiB of arena for all %lld of them\n",
               mib4 / 2.0, (double) L * (double) E * (double) H * 2 / 1048576.0, L);
        printf("  control     %ld of %ld pairs picked a different set of %d experts (%.4f%%)\n",
               bf16_diff_tokens, bf16_tokens, (int) TOPK,
               100.0 * (double) bf16_diff_tokens / (double) bf16_tokens);
        printf("              %ld of %ld expert slots differed (%.4f%% of selections)\n",
               bf16_diff_experts, slots,
               100.0 * (double) bf16_diff_experts / (double) slots);
        printf("  %s\n", bf16_diff_tokens == 0
                              ? "routing is identical on every measured token, so halving the "
                                "router is available"
                              : "routing moved: the f32 router stays the default and the "
                                "halving is not free");
    }

    /* ---- how concentrated is the traffic? ---------------------------- */
    uint64_t total_sel = 0;
    for (long e = 0; e < E; e++)
        total_sel += hist_all[e];
    uint64_t *hs = malloc((size_t) E * sizeof *hs);
    for (long e = 0; e < E; e++)
        hs[e] = hist_all[e];
    qsort(hs, (size_t) E, sizeof *hs, cmp_u64_desc);
    printf("\nexpert popularity over %llu selections (%d layers x %d tokens):\n",
           (unsigned long long) total_sel, n_rows, nn);
    const int NS[] = { 16, 32, 64, 128, 256 };
    for (int i = 0; i < (int) (sizeof(NS) / sizeof(NS[0])); i++) {
        if (NS[i] > (int) E)
            break;
        uint64_t cum = 0;
        for (int j = 0; j < NS[i]; j++)
            cum += hs[j];
        printf("  hottest %3d of %3lld experts carry %5.1f%% of selections (%.1f GiB if pinned "
               "at %.1f MiB each)\n",
               NS[i], (long long) E, 100.0 * (double) cum / (double) total_sel,
               (double) NS[i] * expert_bytes / 1073741824.0, (double) expert_bytes / 1048576.0);
    }
    long never = 0;
    for (long e = 0; e < E; e++)
        never += hist_all[e] == 0;
    printf("  never selected by this stream: %ld of %lld experts\n", never, (long long) E);
    printf("  (a single genre of %d tokens underestimates the spread; treat this as a floor "
           "on how hot the hot set is, not a profile)\n", nn);

    free(kept);
    free(hs);
    free(logits);
    free(router_bf16);
    free(logits_b);
    free(sets);
    free(seen);
    free(hist_all);
    free(row);
    free(pool);
    free(tmp);
    free(uniq);
    free(slot_of);
    free(ids);
    for (int i = 0; i < n_files; i++)
        free(fb[i]);
    free(fb);
    free(texts);
    qe_tok_close(tok);
    qe_close(m);
    return 0;
}
