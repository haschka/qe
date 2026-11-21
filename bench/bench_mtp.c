/* Speculation harness.

   There is no transformer in the tree yet, so this file tests the one thing
   that is actually subtle about speculative decoding - the control loop -
   against a toy target whose next-token distribution is a pure function of
   position. That makes the correct output known in advance, which turns the
   two properties that matter into assertions:

     1. At temp == 0 the emitted sequence must be *byte-identical* to plain
        greedy decoding for every drafter, including a drafter that is wrong
        every single time. Speculation is an optimisation; if a control-loop
        bug could change a token, this is where it shows. The failure mode is
        real and silent: emitting a drafted token before it has been verified,
        or re-verifying from a window that is off by one, both produce fluent
        garbage.
     2. A drafter that reports the target's own probabilities must be accepted
        100% of the time at any temperature. That is exactly the accept test
        min(1, p/q) with p == q, so it fails if p_target is computed wrongly -
        which is precisely how the temperature path was broken once already.

   Everything else measured here is the number the whole exercise is for:
   tokens per target pass. A decode step moves ~5.44 GiB whether it emits one
   token or five, so tok_per_step *is* the speedup, and the last table converts
   it to tok/s at the 88.0 ms/step that `make fast` measures.
*/
#include "qe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 16 /* toy vocabulary */
#define MAXTOK 4096

/* What the toy target wants at absolute position q. Coprime step, so it
 * visits every id and never settles into a period the drafter could exploit. */
static int want(long q) {
    return (int)(((q * 7 + 3) % V));
}

/* Logit row for predicting position q: a unique argmax at want(q) plus a
 * graded tail, so greedy is unambiguous and sampling has real spread. */
static void row_for(float *row, long q) {
    const int w = want(q);
    for (int t = 0; t < V; t++)
        row[t] = -0.3f * (float)((t + 5 * w + 1) % V);
    row[w] = 3.0f;
}

enum {
    D_PERFECT,  /* proposes the target's own tokens, no probabilities */
    D_TRUEP,    /* as above, and reports p_target for each draft */
    D_WRONG,    /* wrong at the very first position, every step */
    D_CERTAIN,  /* right token, but claims probability 1.0 (overconfident) */
    D_LAZY,     /* correct for k-1, then wrong */
    D_NONE      /* declines to draft */
};

static const char *dname(int kind) {
    switch (kind) {
    case D_PERFECT: return "perfect";
    case D_TRUEP: return "perfect+p";
    case D_WRONG: return "always-wrong";
    case D_CERTAIN: return "overconfident";
    case D_LAZY: return "k-1 correct";
    default: return "no-draft";
    }
}

typedef struct {
    qe_spec_model m;
    int kind;
    float temp; /* the temperature the caller is sampling at */
    long verifies, verify_toks, drafts, draft_toks;
    int strict;     /* greedy: every window token must be the target's own */
    int fail_at;    /* fail the Nth verify() call, -1 for never */
    int stop_after; /* ask the driver to stop after N tokens, -1 for never */
    int *out;
    long n_out, cap_out;
    uint64_t rng;
} toy;

static int toy_verify(void *ctx, long long pos, const int *toks, int n, float *logits) {
    toy *t = ctx;
    if (t->fail_at >= 0 && t->verifies == (long)t->fail_at)
        return 1;
    t->verifies++;
    t->verify_toks += n;
    /* A target pass that ignored its input would still be correct for this toy,
     * so require that the anchor of the window - the last accepted token, at
     * absolute position pos - is one the target itself would have produced. The
     * rest of the window is the draft, which is deliberately *not* the target's
     * own choice; that is the entire point of speculation. Checking the anchor
     * is what catches a driver that appended an unverified draft token. */
    if (t->strict && toks[0] != want(pos))
        return 2;
    for (int j = 0; j < n; j++)
        row_for(logits + (size_t)j * V, pos + j + 1); /* toks[j] is at pos+j */
    return 0;
}

static int toy_draft(void *ctx, long long pos, const int *toks, int n, int *out, float *probs,
                     int k) {
    (void)toks;
    (void)n;
    toy *t = ctx;
    if (t->kind == D_NONE)
        return 0;
    t->drafts++;
    for (int i = 0; i < k; i++) {
        long q = pos + i;
        int right = want(q);
        int id = right;
        if (t->kind == D_WRONG)
            id = (right + 1) % V;
        else if (t->kind == D_LAZY && i == k - 1)
            id = (right + 1) % V;
        out[i] = id;
        if (probs) {
            if (t->kind == D_CERTAIN) {
                probs[i] = 1.0f;
            } else if (t->kind == D_TRUEP) {
                float row[V];
                row_for(row, q);
                double mx = row[0], s = 0;
                for (int u = 0; u < V; u++)
                    if (row[u] > mx)
                        mx = row[u];
                for (int u = 0; u < V; u++)
                    s += exp((row[u] - mx) / (double)t->temp);
                probs[i] = (float)(exp((row[id] - mx) / (double)t->temp) / s);
            } else {
                probs[i] = 0.0f; /* "unknown": the driver treats it as 1 */
            }
        }
    }
    t->draft_toks += k;
    return k;
}

static int toy_emit(void *ctx, int tok, int is_eos) {
    toy *t = ctx;
    if (t->n_out >= t->cap_out) {
        long cap = t->cap_out ? t->cap_out * 2 : 256;
        t->out = realloc(t->out, (size_t)cap * sizeof *t->out);
        t->cap_out = cap;
    }
    t->out[t->n_out++] = tok;
    if (t->stop_after >= 0 && t->n_out >= (long)t->stop_after)
        return 1;
    (void)is_eos;
    return 0;
}

static void toy_init(toy *t, int kind, float temp) {
    memset(t, 0, sizeof *t);
    t->m.ctx = t;
    t->m.vocab = V;
    t->m.verify = toy_verify;
    t->m.draft = toy_draft;
    t->m.emit = toy_emit;
    t->kind = kind;
    t->temp = temp;
    t->strict = (temp <= 0.0f);
    t->fail_at = -1;
    t->stop_after = -1;
    t->rng = 0x9e3779b97f4a7c15ull;
}

static long run(toy *t, int k, float temp, long max_tokens, int eos, unsigned seed,
                qe_spec_stats *st) {
    qe_spec_opts o = { k, temp, seed, max_tokens, eos };
    int prompt[4] = { want(0), want(1), want(2), want(3) };
    return qe_spec_generate(&t->m, &o, prompt, 4, st);
}

static int fails;

static void check(int ok, const char *what) {
    if (!ok) {
        fails++;
        printf("  FAIL: %s\n", what);
    }
}

/* Cost of one target pass at the rates `make fast` measures against the
 * 5568 MiB step that bin/bench_gemv derives from the file: per-type kernel
 * times summed (83.4 ms) plus the F32/BF16 remainder at stream rate (4.5 ms).
 * It is the target pass only - the draft head's own bytes are extra, which is
 * why docs/DESIGN.md 4 divides them in rather than reusing this column. */
#define STEP_MS 88.0

/* ---------------------------------------------- the corpus-as-truth target
 *
 * The n-gram drafter (src/draft.c) has no useful toy to be measured against: a
 * target whose next token is a function of position is something a hash of the
 * tail has no chance with, so its accept rate measures the toy and not the
 * drafter. What does measure it is a target that agrees with a real sequence -
 * the corpus is the truth, one byte per id - because then the only question is
 * how often the text ahead of us also appeared behind us, which is exactly the
 * question the drafter is built to answer. Output identity moves here too, and
 * it is worth more than the toy version: greedy speculation against a target
 * that is not a formula, with a drafter that is wrong often. */
#define VC 256

static const char *const CORPUS_FALLBACK[] = {
    "run the tests and tell me what failed\n",
    "$ make test\n  ok   bin/bench_chat\n  ok   bin/bench_mtp\nPASS (0 failures)\n",
    "the tests pass. what should we do next?\n",
    "$ make test\n  ok   bin/bench_chat\n  ok   bin/bench_mtp\nPASS (0 failures)\n",
    "{\"role\":\"tool\",\"tool_call_id\":\"call-0\",\"content\":\"PASS (0 failures)\"}\n",
    "good. now run the same thing with the checkpoint loaded\n",
    "$ make test\n  ok   bin/bench_chat\n  ok   bin/bench_mtp\nPASS (0 failures)\n",
    NULL
};

/* A file if one was given and it is long enough to be a sequence rather than a
 * sentence; the built-in transcript otherwise, repeated so that the drafter has
 * something to find. */
static unsigned char *load_corpus(const char *path, long *n_out) {
    if (path) {
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            unsigned char *buf = sz > 0 ? malloc((size_t)sz) : NULL;
            size_t got = buf ? fread(buf, 1, (size_t)sz, f) : 0;
            fclose(f);
            if ((long)got > 4096) {
                *n_out = (long)got;
                return buf;
            }
            free(buf);
        }
    }
    long cap = 16384, len = 0;
    unsigned char *buf = malloc((size_t)cap);
    if (!buf)
        return NULL;
    for (int round = 0; round < 12; round++)
        for (int i = 0; CORPUS_FALLBACK[i]; i++) {
            const long l = (long)strlen(CORPUS_FALLBACK[i]);
            if (len + l >= cap)
                continue;
            memcpy(buf + len, CORPUS_FALLBACK[i], (size_t)l);
            len += l;
        }
    *n_out = len;
    return buf;
}

static void row_byte(float *row, int w) {
    for (int t = 0; t < VC; t++)
        row[t] = -0.3f * (float)((t + 5 * w + 1) % VC);
    row[w] = 3.0f;
}

typedef struct {
    qe_spec_model m;
    const unsigned char *text;
    long n;
    qe_draft *d;
    int *out;
    long n_out, cap_out;
    long verifies, verify_toks;
} ctarget;

static int c_verify(void *ctx, long long pos, const int *toks, int n, float *logits) {
    ctarget *c = ctx;
    (void)toks;
    c->verifies++;
    c->verify_toks += n;
    for (int j = 0; j < n; j++) {
        long q = pos + j + 1; /* toks[j] sits at pos+j, so its successor is next */
        if (q >= c->n)
            q = c->n - 1;
        row_byte(logits + (size_t)j * VC, c->text[q]);
    }
    return 0;
}

static int c_draft(void *ctx, long long pos, const int *toks, int n, int *out, float *probs,
                   int k) {
    ctarget *c = ctx;
    (void)pos;
    (void)probs; /* greedy: the drafter has no distribution to report */
    return qe_draft_suggest(c->d, toks, n, out, k);
}

static int c_emit(void *ctx, int tok, int is_eos) {
    ctarget *c = ctx;
    if (c->n_out >= c->cap_out) {
        long cap = c->cap_out ? c->cap_out * 2 : 4096;
        c->out = realloc(c->out, (size_t)cap * sizeof *c->out);
        c->cap_out = cap;
    }
    c->out[c->n_out++] = tok;
    (void)is_eos;
    return 0;
}

static int c_init(ctarget *c, const unsigned char *text, long n) {
    memset(c, 0, sizeof *c);
    c->m.ctx = c;
    c->m.vocab = VC;
    c->m.verify = c_verify;
    c->m.draft = c_draft;
    c->m.emit = c_emit;
    c->text = text;
    c->n = n;
    c->d = qe_draft_open(0, 0);
    return c->d ? 0 : -1;
}

/* What one verification step costs on this host at a given window width, in
 * milliseconds per row. This is the width sweep from `make forward` (512-token
 * chunk, 1 MiB grouping threshold, every shipping default on) written down rather
 * than guessed: a step that verifies T rows takes T * rows_ms(T) ms, so the whole
 * speculation trade is decided here and not by a token count. The curve is not
 * monotone - a 5-row window is dearer than a 6-row one - which is why the table is
 * consulted by nearest measured width instead of interpolated. */
static double rows_ms(int rows) {
    static const int W[] = { 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32 };
    static const double MS[] = { 112.1, 61.6, 55.6, 44.5, 45.2, 40.8,
                                 42.9, 34.6, 31.0, 29.2, 26.7, 25.5 };
    int best = 0;
    double bestd = 1e30;
    for (unsigned i = 0; i < sizeof W / sizeof *W; i++) {
        double d = (double)(W[i] - rows);
        if (d < 0)
            d = -d;
        if (d < bestd)
            bestd = d, best = (int)i;
    }
    return MS[best];
}

/* The floor: one row, greedy, no speculation. */
#define PLAIN_MS rows_ms(1)

int main(int argc, char **argv) {
    const char *corpus_path = NULL;
    double window_ms = STEP_MS;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--corpus") && i + 1 < argc)
            corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--window-ms") && i + 1 < argc)
            window_ms = atof(argv[++i]);
        else
            printf("unknown argument %s (known: --corpus FILE, --window-ms MS)\n", argv[i]);
    }
    printf("toy target: vocab %d, deterministic by position, %d tokens max\n", V, MAXTOK);

    /* ---- 1. greedy output identity -------------------------------------- */
    printf("\n== greedy output vs plain greedy (temp 0, 200 tokens) ==\n");
    printf("%-14s %-3s %8s %9s %9s %8s  %s\n", "drafter", "k", "steps", "tok/step", "acc/draft",
           "tok/s", "output");
    static int ref[MAXTOK];
    long ref_n = 0;
    for (long i = 0; i < 200; i++)
        ref[ref_n++] = want(4 + i); /* prompt is 4 tokens, positions 0..3 */
    {
        toy t;
        toy_init(&t, D_NONE, 0.0f);
        qe_spec_stats st;
        long n = run(&t, 0, 0.0f, 200, -1, 1, &st);
        check(n == 200 && st.tok_per_step == 1.0, "plain greedy: one token per step");
        check(st.accept_rate == 0.0, "plain greedy: nothing drafted");
        check(t.n_out == ref_n && !memcmp(t.out, ref, (size_t)ref_n * sizeof *ref),
              "plain greedy is the greedy sequence, by construction");
        free(t.out);
    }
    for (int i = 0; i < 6; i++) {
        int kind = i;
        int k = (kind == D_LAZY) ? 4 : (kind == D_WRONG ? 5 : 4);
        toy t;
        toy_init(&t, kind, 0.0f);
        qe_spec_stats st;
        long n = run(&t, k, 0.0f, 200, -1, 1, &st);
        int same = (t.n_out == ref_n) && !memcmp(t.out, ref, (size_t)ref_n * sizeof *ref);
        printf("%-14s %-3d %8ld %9.2f %5ld/%-3ld %8.1f  %s\n", dname(kind), k, st.steps,
               st.tok_per_step, st.accepted, st.drafted, 1000.0 / window_ms * st.tok_per_step,
               same ? "identical" : "DIFFERENT");
        check(same, "emitted sequence is byte-identical to plain greedy");
        check(n == 200, "returns the number of tokens emitted");
        if (kind == D_PERFECT || kind == D_TRUEP) {
            char msg[128];
            sprintf(msg, "perfect drafter: tok/step == k+1 == %d", k + 1);
            check(st.tok_per_step == (double)(k + 1), msg);
        }
        if (kind == D_WRONG || kind == D_NONE)
            check(st.tok_per_step == 1.0, "useless drafter: exactly one token per step");
        if (kind == D_LAZY)
            check(st.tok_per_step == (double)k, "k-1 drafter: exactly k tokens per step");
        free(t.out);
    }
    printf("tok/s column assumes %.1f ms per target pass, wide window or not\n", window_ms);

    /* ---- 2. perfect drafter, exact tok/step for k that divides 200 ------- */
    printf("\n== draft length scaling (perfect drafter) ==\n");
    static const int KS[] = { 1, 3, 4, 7, 19 }; /* k+1 divides 200 */
    for (unsigned i = 0; i < sizeof KS / sizeof *KS; i++) {
        int k = KS[i];
        toy t;
        toy_init(&t, D_PERFECT, 0.0f);
        qe_spec_stats st;
        run(&t, k, 0.0f, 200, -1, 1, &st);
        printf("k=%-3d %7.0f%% of the draft verified -> %.1f tok/step -> %.1f tok/s\n", k,
               100.0 * st.accept_rate, st.tok_per_step, 1000.0 / window_ms * st.tok_per_step);
        check(st.tok_per_step == (double)(k + 1), "tok/step == k+1");
        check(t.verify_toks == t.verifies * (k + 1), "verify window is k+1 wide");
        free(t.out);
    }

    /* ---- 3. temperature path -------------------------------------------- */
    printf("\n== sampling (temp 1.0, 400 tokens, seed 7) ==\n");
    {
        toy t;
        toy_init(&t, D_TRUEP, 1.0f);
        qe_spec_stats st;
        long n = run(&t, 4, 1.0f, 400, -1, 7, &st);
        printf("%-14s accept %.4f  tok/step %.2f\n", dname(D_TRUEP), st.accept_rate,
               st.tok_per_step);
        check(n == 400, "sampled run emits what it was asked to");
        check(st.drafted > 0, "sampling path does draft");
        /* Draft probabilities are floats, so p_target/p_draft can fall a few
         * 1e-7 below 1 and reject; anything else means p_target is wrong. */
        check(st.accept_rate >= 0.999, "p_draft == p_target is (almost) always accepted");
        free(t.out);
    }
    {
        toy t;
        toy_init(&t, D_CERTAIN, 1.0f);
        qe_spec_stats st;
        long n = run(&t, 4, 1.0f, 400, -1, 7, &st);
        printf("%-14s accept %.4f  tok/step %.2f  drafted %ld accepted %ld\n", dname(D_CERTAIN),
               st.accept_rate, st.tok_per_step, st.drafted, st.accepted);
        check(n == 400 && st.steps > 0, "overconfident run makes progress");
        check(st.accepted < st.drafted, "an overconfident draft is rejected sometimes");
        check(st.accept_rate > 0.0, "and not always rejected");
        check(st.tok_per_step < 5.0, "rejections cost tokens per step");
        int inv = 0;
        for (long i = 0; i < t.n_out; i++)
            if (t.out[i] < 0 || t.out[i] >= V)
                inv++;
        check(inv == 0, "every sampled token is in the vocabulary");
        free(t.out);
    }
    {
        /* Same seed, same opts: the temperature path must be reproducible, or
         * a retry of a request silently changes the answer. */
        long a[MAXTOK], b[MAXTOK];
        long na = 0, nb = 0;
        for (int pass = 0; pass < 2; pass++) {
            toy t;
            toy_init(&t, D_CERTAIN, 1.0f);
            run(&t, 4, 1.0f, 300, -1, 99, NULL);
            long *dst = pass ? b : a;
            memcpy(dst, t.out, (size_t)t.n_out * sizeof *dst);
            if (pass)
                nb = t.n_out;
            else
                na = t.n_out;
            free(t.out);
        }
        check(na == nb && !memcmp(a, b, (size_t)na * sizeof *a),
              "temp > 0 is reproducible for a fixed seed");
    }

    /* ---- 4. limits, eos, errors ----------------------------------------- */
    printf("\n== stopping conditions ==\n");
    static const long LIMITS[] = { 1, 2, 7, 33, 200 };
    for (unsigned i = 0; i < sizeof LIMITS / sizeof *LIMITS; i++) {
        for (int k = 0; k <= 5; k += 5) {
            toy t;
            toy_init(&t, k ? D_PERFECT : D_NONE, 0.0f);
            qe_spec_stats st;
            long n = run(&t, k, 0.0f, LIMITS[i], -1, 3, &st);
            check(n == LIMITS[i] && t.n_out == LIMITS[i], "max_tokens is exact, never exceeded");
            free(t.out);
        }
    }
    printf("max_tokens {1,2,7,33,200} x k {0,5}: exact\n");
    {
        toy t;
        toy_init(&t, D_PERFECT, 0.0f);
        qe_spec_stats st;
        long n = run(&t, 0, 0.0f, 100, want(4), 1, &st);
        check(n == 1 && st.steps == 1 && t.n_out == 1, "eos stops a greedy run at once");
        free(t.out);

        toy_init(&t, D_PERFECT, 0.0f);
        n = run(&t, 4, 0.0f, 100, want(6), 1, &st);
        check(n == 3, "eos inside a verified draft stops there (3 tokens)");
        free(t.out);

        toy_init(&t, D_PERFECT, 0.0f);
        t.stop_after = 10;
        n = run(&t, 4, 0.0f, 100, -1, 1, &st);
        check(n == 10 && t.n_out == 10, "emit() returning non-zero stops the loop");
        free(t.out);

        toy_init(&t, D_PERFECT, 0.0f);
        t.fail_at = 0;
        n = run(&t, 4, 0.0f, 100, -1, 1, &st);
        check(n == -1, "a failing target pass aborts the request with -1");
        free(t.out);

        toy_init(&t, D_PERFECT, 0.0f);
        t.fail_at = 3;
        n = run(&t, 4, 0.0f, 100, -1, 1, &st);
        check(n == -1, "a target pass failing mid-generation still reports -1");
        free(t.out);
    }

    /* ---- 6. a drafter that reads no weights, on text that repeats -------- */
    printf("\n== the n-gram drafter, corpus as target (temp 0) ==\n");
    long clen = 0;
    unsigned char *corpus = load_corpus(corpus_path, &clen);
    if (!corpus) {
        check(0, "a corpus to speculate over");
    } else {
        printf("%s, %ld bytes, one id per byte\n",
               corpus_path ? corpus_path : "built-in transcript", clen);
        const long NP = 64; /* a prompt long enough for an n-gram to land in */
        const long room = clen - NP - 8;
        const long budget = room < 2000 ? room : 2000;
        static const int CKS[] = { 1, 2, 3, 4, 7, 12 };
        double best_ratio = 0;
        int best_k = 0;
        for (unsigned i = 0; i < sizeof CKS / sizeof *CKS; i++) {
            const int k = CKS[i];
            ctarget c;
            if (c_init(&c, corpus, clen)) {
                check(0, "the corpus target opened");
                break;
            }
            int prompt[64];
            for (long j = 0; j < NP; j++)
                prompt[j] = corpus[j];
            qe_spec_opts o = { k, 0.0f, 7, budget, -1 };
            qe_spec_stats st;
            long n = qe_spec_generate(&c.m, &o, prompt, (int)NP, &st);
            qe_draft_stats ds;
            qe_draft_get_stats(c.d, &ds);
            /* The answer is the corpus. Speculation is allowed to be slower and
             * never allowed to be different, so one byte out of place is a
             * control-loop bug that would read somewhere else as fluent output. */
            int wrong = 0;
            for (long j = 0; j < n && NP + j < clen; j++)
                if (c.out[j] != corpus[NP + j])
                    wrong++;
            const double gain = st.steps ? (double)n / (double)st.steps : 0;
            /* The trade, priced: a step that checks k+1 rows costs (k+1) rows of
             * the measured width curve and emits `gain` tokens, so the ratio to a
             * plain greedy step is what a user would feel. tokens per step on its
             * own is not the question - a 8-row window that doubles the token count
             * loses, and a 2-row window that adds 15 % of a row wins. */
            const double step_ms = rows_ms(k + 1) * (k + 1);
            const double ms_tok = gain > 0 ? step_ms / gain : 1e30;
            const double ratio = PLAIN_MS / ms_tok;
            printf("k=%-3d accept %5.1f%%  tok/step %5.2f  matched %4ld/%4ld  "
                   "match %5.1f/%2d long  %6.1f ms/token against %5.1f plain -> %.2fx\n",
                   k, 100.0 * st.accept_rate, st.tok_per_step, ds.matched, ds.lookups,
                   ds.matched ? ds.match_len / (double)ds.matched : 0.0, ds.longest, ms_tok,
                   PLAIN_MS, ratio);
            check(n == budget, "a corpus run emits the whole budget");
            check(wrong == 0, "speculated output is the corpus, byte for byte");
            check(c.verify_toks <= st.steps * (k + 1), "a window is never wider than k+1");
            if (k == 1)
                check(ds.matched > 0, "the drafter finds the repetition it was built for");
            if (ratio > best_ratio)
                best_ratio = ratio, best_k = k;
            qe_draft_close(c.d);
            free(c.out);
        }
        /* The floor this whole exercise has to clear, and it is now arithmetic
         * rather than a guess: something has to beat one row of the measured width
         * sweep. Which k does it, and by how much, is printed above - at a 2-row
         * window the sweep is already 55 % cheaper per row, so a drafter needs only
         * 1.10 tokens per step to be ahead, and the weights-free n-gram drafter
         * clears that on text with repetition in it. */
        printf("best: k=%d at %.2fx of a plain greedy step (%.1f -> %.1f ms/token)\n", best_k,
               best_ratio, PLAIN_MS, PLAIN_MS / (best_ratio > 0 ? best_ratio : 1.0));
        check(best_ratio > 1.0, "some draft length beats a plain greedy step");

        /* The control: the same target, the same loop, and a sequence with
         * nothing in it to find. A drafter that reports matches it did not find,
         * or a loop that emits a token it never verified, both show up here as
         * tokens per step above one on text where none is possible. */
        {
            long rl = clen < 20000 ? clen : 20000;
            unsigned char *rnd = malloc((size_t)rl);
            uint64_t z = 0x2545f4914f6cdd1dull;
            for (long j = 0; j < rl; j++) {
                z += 0x9e3779b97f4a7c15ull;
                uint64_t y = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                y = (y ^ (y >> 27)) * 0x94d049bb133111ebull;
                rnd[j] = (unsigned char)((y ^ (y >> 31)) >> 40);
            }
            const int k = 4;
            ctarget c;
            if (c_init(&c, rnd, rl) == 0) {
                int prompt[64];
                for (long j = 0; j < NP; j++)
                    prompt[j] = rnd[j];
                qe_spec_opts o = { k, 0.0f, 7, 800, -1 };
                qe_spec_stats st;
                long n = qe_spec_generate(&c.m, &o, prompt, (int)NP, &st);
                int wrong = 0;
                for (long j = 0; j < n && NP + j < rl; j++)
                    if (c.out[j] != rnd[NP + j])
                        wrong++;
                printf("noise control  accept %5.1f%%  tok/step %5.2f (a sequence with "
                       "nothing to find)\n",
                       100.0 * st.accept_rate, st.tok_per_step);
                check(wrong == 0, "a speculative run over noise is still the target's own");
                /* One token a step is what a drafter that finds nothing has to
                 * report, and it is what it reports. This bound is the same
                 * arithmetic as the one above, where a window wider than a row is
                 * only affordable if it is nearly always right: 1.05 says at most
                 * one token in twenty came from a draft that could not have been. */
                check(st.tok_per_step < 1.05, "nothing is speculated where nothing repeats");
                qe_draft_close(c.d);
                free(c.out);
            }
            free(rnd);
        }
        free(corpus);
    }

    printf("\n%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
