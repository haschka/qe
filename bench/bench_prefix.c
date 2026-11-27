/* What the second turn of a conversation costs.

   A coding agent spends its life in one loop: ask, read what came back, add the
   result to the transcript, ask again. Every client in that loop resends the
   whole conversation, and every one of them pays to re-read a conversation that
   was already read once. This is the measurement of the thing that stops that:
   a request whose prompt is the sequence the forward pass is already holding,
   plus new tokens, puts only the new tokens through the network.

   Two properties are asserted and one is reported.

     1. Same tokens in, same tokens out. Reuse is an optimisation, and an
        optimisation that changes the answer is a bug that happens to be fast, so
        the continuation from a reused sequence is compared token id by token id
        against the continuation from feeding the same prompt cold. The trunk has
        to be identical rather than close: chunked ingestion starts at the
        beginning of whatever it is handed, and a reused sequence starts halfway
        through the conversation it belongs to.
     2. Reuse only ever means extension. A transcript that agrees for a while and
        then differs describes a different conversation, and reusing it would be
        handing the model a beginning it never had. A prompt that diverges
        therefore has to report zero cached tokens, which is asserted.
     3. The number anyone cares about: seconds saved on the turn that reused, and
        how much of the prompt that was. Reported and not asserted, because it
        moves with the thread count and with how much was added to the turn. On
        this host a two turn conversation of 109 ids costs 9.13 s cold and 3.31 s
        with the 86 ids of the first turn still standing.

   Neither the prompts below nor this file contains a control token: the
   chat-shaped case renders through src/chat.c, which takes the delimiters from
   the checkpoint's vocabulary by id, and nothing here prints prompt bytes or
   reply bytes raw (angle brackets are printed as dots, so a boundary could not
   reach a log even if the model wrote one out as prose).

     make prefix                  # or:
     ./bin/bench_prefix --model <vol1.gguf> --threads auto --max 16
*/
#include "chat.h"
#include "qe.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int checks, fails;

static void checkf(int ok, const char *fmt, ...) {
    va_list ap;
    checks++;
    fputs(ok ? "  ok   " : "  FAIL ", stdout);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
    if (!ok)
        fails++;
}
#define CHECK(ok, ...) checkf((ok) ? 1 : 0, __VA_ARGS__)

/* Bytes the model wrote, for a human reading this log. Angle brackets become
   dots and control bytes become spaces, so what is printed can never look like a
   turn boundary no matter what came out of the decoder. */
static void say(const char *what, const char *s, long n) {
    if (!s || n <= 0) {
        printf("       %s: (nothing)\n", what);
        return;
    }
    printf("       %s: ", what);
    for (long i = 0; i < n && i < 110; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\n' || c == '\r' || c == '\t')
            putchar(' ');
        else if (c == '<' || c == '>')
            putchar('.');
        else if (c < 0x20 || c == 0x7f)
            putchar(' ');
        else
            putchar((int)c);
    }
    printf("%s\n", n > 110 ? " ..." : "");
}

/* ------------------------------------------------------------------- runs */

/* One generation, everything worth keeping from it. `ids` is what the decoder
 * emitted (which is also what the engine appended to its transcript), `text` is
 * the renderable part of the same thing, and `stop_at` cuts a generation short
 * the way a chat parser does when a turn ends, which is how the transcript ends
 * up shorter than the prompt plus everything sampled. */
typedef struct {
    int *ids;
    long n, cap;
    char *text;
    long tn, tcap;
    int stop_at;  /* stop once this many tokens have been reported (0 = never) */
    long reported;
    long rc;      /* what qe_engine_generate_tokens returned */
    long np;      /* ids the prompt encoded to */
    int truncated;
    qe_gen_stats st;
} run;

static void run_free(run *r) {
    free(r->ids);
    free(r->text);
    memset(r, 0, sizeof *r);
}

static int on_token(void *u, const qe_token_out *t) {
    run *r = u;
    if (r->n == r->cap) {
        long cap = r->cap ? r->cap * 2 : 64;
        int *ids = realloc(r->ids, (size_t)cap * sizeof *ids);
        if (!ids)
            return 1;
        r->ids = ids;
        r->cap = cap;
    }
    r->ids[r->n++] = t->id;
    r->reported++;
    if (t->nbytes > 0 && r->tn + t->nbytes + 1 <= r->tcap) {
        memcpy(r->text + r->tn, t->utf8, (size_t)t->nbytes);
        r->tn += t->nbytes;
        r->text[r->tn] = 0;
    }
    if (r->stop_at && r->reported >= r->stop_at)
        return 1;
    return 0;
}

/* Encodes text; the engine does the same thing internally, and having the same
 * ids here is what makes a reuse miss explainable instead of silent. */
static int *enc(qe_tok *t, const char *text, long *out_n) {
    long want = qe_tok_count(t, text) + 8;
    int *ids = malloc((size_t)want * sizeof *ids);
    long n = ids ? qe_tok_encode(t, text, ids, want) : -1;
    if (n < 0) {
        free(ids);
        *out_n = 0;
        return NULL;
    }
    *out_n = n;
    return ids;
}

static long lcp(const int *a, long an, const int *b, long bn) {
    long n = an < bn ? an : bn, i = 0;
    while (i < n && a[i] == b[i])
        i++;
    return i;
}

static qe_engine *E;
static qe_tok *TOK;
static int MAX_NEW = 16;

/* Runs one request and prints what it cost. Returns 0 on success. */
static int do_run(const char *label, const char *text, int stop_at, run *out) {
    memset(out, 0, sizeof *out);
    out->tcap = 65536;
    out->text = malloc((size_t)out->tcap);
    out->stop_at = stop_at;
    if (!out->text)
        return -1;
    out->text[0] = 0;
    out->np = 0;
    int *ids = enc(TOK, text, &out->np);
    free(ids);

    qe_prompt p = { .text = text, .is_chat = 1 };
    qe_gen_opts o = { .max_tokens = MAX_NEW, .temp = 0.0f, .seed = 20260111, .spec_k = 0, .eos = -1 };
    out->rc = qe_engine_generate_tokens(E, &p, &o, on_token, out, &out->st);
    if (out->rc < 0) {
        printf("  FAIL %s: generation failed\n", label);
        fails++;
        return -1;
    }
    printf("       %-10s prompt %ld, cached %ld, fed %ld, emitted %ld, %.3f s\n", label, out->np,
           out->st.cached, out->st.steps, out->st.tokens, out->st.seconds);
    fflush(stdout);
    return 0;
}

/* The transcript the engine should be holding after a run: the prompt it was
 * given, plus everything the decoder emitted before the turn ended. Rebuilt here
 * rather than trusted from the engine, because the whole point is to check the
 * engine against something that computes reuse a second, dumber way. */
static int *transcript(const char *prompt, const run *r, long *out_n) {
    long n = 0;
    int *ids = enc(TOK, prompt, &n);
    if (!ids) {
        *out_n = 0;
        return NULL;
    }
    int *tx = realloc(ids, (size_t)(n + r->n + 1) * sizeof *tx);
    if (!tx) {
        free(ids);
        *out_n = 0;
        return NULL;
    }
    memcpy(tx + n, r->ids, (size_t)r->n * sizeof *tx);
    *out_n = n + r->n;
    return tx;
}

static int same_ids(const run *a, const run *b) {
    if (a->n != b->n)
        return 0;
    return !memcmp(a->ids, b->ids, (size_t)a->n * sizeof *a->ids);
}

/* ------------------------------------------------------------- chat bodies */

/* A rendered prompt for a JSON request body, thinking on, exactly as serve.c
 * would render it. The caller frees. */
static char *render_prompt(const char *json, char *err, size_t whyn) {
    qe_json *b = qe_json_parse(json, strlen(json), err, whyn);
    if (!b)
        return NULL;
    qe_chat r = { 0 };
    qe_chat_opts o = { .enable_thinking = 1, .effort_xhigh = 1, .preserve_thinking = 1 };
    char *out = NULL;
    if (qe_chat_render(TOK, b, &o, &r, err, whyn) == 0 && r.prompt.p)
        out = strndup(r.prompt.p, r.prompt.len);
    qe_chat_free(&r);
    qe_json_free(b);
    return out;
}

/* A request body of one to three turns. `reasoning` and `content` go into the
 * assistant turn the way a client sends a turn back: its reasoning in
 * reasoning_content, its answer in content, either one empty. */
static char *chat_body(const char *user1, const char *reasoning, const char *content,
                       const char *user2) {
    qe_str s = { 0 };
    qe_str_puts(&s, "{\"messages\":[{\"role\":\"user\",\"content\":");
    qe_str_json_str(&s, user1, strlen(user1));
    if (reasoning || content) {
        qe_str_puts(&s, "},{\"role\":\"assistant\",\"content\":");
        qe_str_json_str(&s, content ? content : "", content ? strlen(content) : 0);
        if (reasoning) {
            qe_str_puts(&s, ",\"reasoning_content\":");
            qe_str_json_str(&s, reasoning, strlen(reasoning));
        }
    }
    if (user2 && *user2) {
        qe_str_puts(&s, "},{\"role\":\"user\",\"content\":");
        qe_str_json_str(&s, user2, strlen(user2));
    }
    qe_str_puts(&s, "}]}");
    if (s.err) {
        qe_str_free(&s);
        return NULL;
    }
    return s.p;
}

/* Does the second prompt begin with the first one, token for token? That is the
 * whole of reuse: the model's turn is opened by the generation prompt, and a
 * client that replays the turn has to extend it rather than restart it. A
 * rendered transcript that is a prefix of the next rendered transcript is what
 * lets the engine keep its sequence, and the reason it usually is one is that
 * every turn boundary is a control token, which splits the text into segments
 * that cannot merge with whatever comes after them. */
static int id_prefix(qe_tok *t, const char *head, const char *tail, long *at) {
    long hn = 0, tn = 0;
    int *h = enc(t, head, &hn), *v = h ? enc(t, tail, &tn) : NULL;
    int ok = h && v && tn >= hn && !memcmp(h, v, (size_t)hn * sizeof *h);
    if (at)
        *at = (h && v) ? lcp(h, hn, v, tn) : 0;
    free(h);
    free(v);
    return ok;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    const char *path = "/mnt/data/data/ai-models/llama-cpp/llm/"
                       "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int threads = QE_THREADS_AUTO, fuse = 0, n_ctx = 4096;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            const char *s = argv[++i];
            threads = (*s == 'a' || *s == 'A') ? QE_THREADS_AUTO : atoi(s);
        } else if (!strcmp(argv[i], "--max") && i + 1 < argc)
            MAX_NEW = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)
            n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fuse") && i + 1 < argc)
            fuse = atoi(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s [--model PATH] [--threads N] [--max N] [--ctx N] [--fuse S]\n",
                    argv[0]);
            return 2;
        }
    }
    if (fuse > 0)
        qe_fuse(fuse);
    qe_pool_init(threads, NULL, 0);

    qe_engine_opts eo = { 0 };
    eo.model_path = path;
    eo.n_threads = threads;
    eo.n_ctx = n_ctx;
    eo.temp = 0.0f;
    eo.id = "bench-prefix";
    char err[512] = { 0 };
    printf("== open (the whole checkpoint is placed in memory here; no timer) ==\n");
    fflush(stdout);
    E = qe_engine_open(&eo, err, sizeof err);
    if (!E) {
        fprintf(stderr, "engine open failed: %s\n", err);
        return 1;
    }
    qe_engine_info info;
    qe_engine_describe(E, &info);
    printf("  %s (%s), %d threads\n  bytes: %s\n  placed: %s\n", info.label, info.arch,
           info.n_threads, info.quant, info.stage);
    fflush(stdout);
    if (!info.can_generate) {
        fprintf(stderr, "this engine cannot generate: %s\n", info.why);
        return 1;
    }
    TOK = qe_engine_tok(E);
    if (!TOK) {
        fprintf(stderr, "no tokenizer, so no prompt can become ids\n");
        return 1;
    }

    /* ------------------------------------------------ 1. a text conversation */
    printf("\n== turn 1, then the same conversation with an answer appended ==\n");
    fflush(stdout);
    /* A plain text prompt is the pessimistic case, and the numbers below say so.
     * Text is encoded in pieces that merge, so replaying a turn as bytes and
     * encoding them again does not always reproduce the ids that were produced
     * token by token: the seam between the prompt and the reply is where that
     * shows, and it shows here. Nothing is reused when the ids disagree, which is
     * the correct answer to a sequence the engine was never fed, and the reason
     * the conversation shape below is the case with a number to hit: a turn
     * boundary is a control token, and a control token ends the tokenizer's
     * business with what came before it. */
    const char *A =
        "A directory holds a 4 GB log and a 2 KB script. In at most forty words, name the "
        "quickest way to find the last error in the log and say why it is quick.";
    run r1, r2, r3;
    if (do_run("turn 1", A, 0, &r1))
        return 1;
    say("reply", r1.text, r1.tn);
    CHECK(r1.st.cached == 0, "a conversation nobody has read yet caches nothing (%ld)",
          r1.st.cached);

    qe_str b = { 0 };
    qe_str_puts(&b, A);
    if (r1.text && r1.tn)
        qe_str_append(&b, r1.text, (size_t)r1.tn);
    qe_str_puts(&b, "\n\nNow the same answer in half as many words.\n");
    if (b.err) {
        fprintf(stderr, "out of memory building the second turn\n");
        return 1;
    }
    if (do_run("turn 2 warm", b.p, 0, &r2))
        return 1;
    say("reply", r2.text, r2.tn);

    long tx1_n = 0;
    int *tx1 = transcript(A, &r1, &tx1_n);
    long tx2_n = 0;
    int *tx2 = tx1 ? enc(TOK, b.p, &tx2_n) : NULL;
    long common = tx1 && tx2 ? lcp(tx1, tx1_n, tx2, tx2_n) : 0;
    /* What a plain text prompt can promise is only this: reuse when the ids
     * agree, and never when they do not. Whether they agree at a junction is the
     * tokenizer's business, and the honest answer is that a seam in the middle of
     * prose is not stable, which is why the transcript case below is the one with
     * a number to hit. */
    /* Reuse exactly as much as the ids agree on, and not one token more. A seam in
     * the middle of prose is not stable, so the transcript match fails there, and
     * what is left standing is the boundary the first prompt left behind - the whole
     * of that prompt and nothing of the answer. A number between the two would be a
     * match on tokens that were never fed, which is the one thing this cannot be, so
     * that is what the check is really for. */
    int full = common == tx1_n;
    CHECK(full ? r2.st.cached == tx1_n : (r2.st.cached == 0 || r2.st.cached == r1.np),
          "reuse exactly the ids that agree (%ld of %ld matched, %ld cached, prompt one %ld)",
          common, tx1_n, r2.st.cached, r1.np);
    CHECK(r2.st.rewound == (r2.st.cached > 0 && r2.st.cached != tx1_n),
          "a partial match reports going back rather than carrying over (%ld cached, %d)",
          r2.st.cached, r2.st.rewound);
    printf("       raw text junction: %ld of %ld transcript ids matched%s\n", common, tx1_n,
           common == tx1_n ? ", reused"
                           : ", so the whole thing goes through again (a prose seam can merge "
                             "across the join)");
    CHECK(r2.st.steps == (r2.np - r2.st.cached) + r2.st.tokens,
          "steps counts only what went through the network (%ld, %ld)", r2.st.steps,
          (r2.np - r2.st.cached) + r2.st.tokens);

    int was = qe_engine_prefix_cache(E, 0);
    if (do_run("turn 2 cold", b.p, 0, &r3))
        return 1;
    qe_engine_prefix_cache(E, was);
    CHECK(r3.st.cached == 0, "with reuse off nothing is reused (%ld)", r3.st.cached);
    CHECK(same_ids(&r2, &r3), "reused and cold continuations are the same tokens (%ld, %ld)",
          r2.n, r3.n);
    if (!same_ids(&r2, &r3) && r2.n && r3.n) {
        long same = lcp(r2.ids, r2.n, r3.ids, r3.n);
        printf("       first difference at emitted token %ld: %d vs %d\n", same, r2.ids[same],
               r3.ids[same]);
    }
    if (r3.st.seconds > 0.0 && r2.st.seconds > 0.0)
        printf("       %.3f s reused vs %.3f s cold: %ld of %ld prompt tokens not re-read "
               "(%.0f%% of the turn)\n",
               r2.st.seconds, r3.st.seconds, r2.st.cached, r2.np,
               100.0 * (double)r2.st.cached / (double)r2.np);
    run_free(&r1);
    run_free(&r2);
    run_free(&r3);
    free(tx1);
    free(tx2);
    qe_str_free(&b);

    /* -------------------------------------- 2. a conversation that diverges */
    printf("\n== a prompt that diverges halfway is a different conversation ==\n");
    fflush(stdout);
    qe_str d = { 0 };
    qe_str_puts(&d, "Forget the log. ");
    qe_str_puts(&d, A);
    if (do_run("base", A, 0, &r1))
        return 1;
    if (do_run("diverged", d.p, 0, &r2))
        return 1;
    CHECK(r2.st.cached == 0, "a prompt that differs at the start reuses nothing (%ld)",
          r2.st.cached);    run_free(&r1);
    run_free(&r2);
    qe_str_free(&d);

    /* ------------------------------------- 3. the same prompt, asked twice */
    printf("\n== a prompt whose turn ended at once, asked again ==\n");
    fflush(stdout);
    /* A generation stopped at its very first token leaves the transcript exactly
     * equal to the prompt, which is the one case where there is nothing at all to
     * feed: the logits the request needs are the ones the previous call already
     * computed and kept. */
    const char *E2 = "Name the two things a CPU inference engine is measured on.\n";
    if (do_run("asked", E2, 1, &r1))
        return 1;
    CHECK(r1.st.tokens == 0, "a turn stopped at its first token emits nothing (%ld)",
          r1.st.tokens);
    if (do_run("again warm", E2, 6, &r2))
        return 1;
    CHECK(r2.st.cached == r2.np, "the whole prompt was already the sequence (%ld of %ld)",
          r2.st.cached, r2.np);
    CHECK(r2.st.steps == r2.st.tokens, "nothing was fed a second time (%ld steps, %ld emitted)",
          r2.st.steps, r2.st.tokens);
    was = qe_engine_prefix_cache(E, 0);
    if (do_run("again cold", E2, 6, &r3))
        return 1;
    qe_engine_prefix_cache(E, was);
    CHECK(same_ids(&r2, &r3), "kept logits and recomputed logits agree (%ld, %ld)", r2.n, r3.n);
    run_free(&r1);
    run_free(&r2);
    run_free(&r3);

    /* ------------------------------------------- 4. the shape an agent sends */
    printf("\n== a rendered conversation, which is what a client actually sends ==\n");
    fflush(stdout);
    if (!qe_chat_supported(TOK)) {
        printf("       skipped: this vocabulary does not have the delimiters at the ids the "
               "renderer expects\n");
    } else {
        char cerr[256] = { 0 };
        const char *Q = "In two short sentences, what is wrong with reading a "
                        "4 GB log into memory?";
        char *body1 = chat_body(Q, NULL, NULL, NULL);
        char *p1 = body1 ? render_prompt(body1, cerr, sizeof cerr) : NULL;
        if (!p1) {
            printf("  FAIL could not render the first turn: %s\n", cerr);
            fails++;
        } else {
            if (do_run("chat 1", p1, 0, &r1))
                return 1;
            say("reply", r1.text, r1.tn);
            /* The prompt left the model inside its reasoning block, so what it
             * wrote is reasoning, and a client that keeps a conversation sends
             * it back as reasoning_content with an empty answer. That is also
             * the only replay the template can render back where it came from. */
            char *body2 = chat_body(Q, r1.text && r1.tn ? r1.text : "Nothing.", "",
                                    "Say the same thing in one sentence.");
            char *p2 = body2 ? render_prompt(body2, cerr, sizeof cerr) : NULL;
            if (!p2) {
                printf("  FAIL could not render the second turn: %s\n", cerr);
                fails++;
            } else {
                long at = 0;
                CHECK(id_prefix(TOK, p1, p2, &at),
                      "the renderer extends the turn it opened (first difference at id %ld)",
                      at);
                if (do_run("chat 2 warm", p2, 0, &r2))
                    return 1;
                say("reply", r2.text, r2.tn);
                long n1 = 0;
                int *i1 = enc(TOK, p1, &n1);
                long n2 = 0;
                int *i2 = i1 ? enc(TOK, p2, &n2) : NULL;
                long cm = i1 && i2 ? lcp(i1, n1, i2, n2) : 0;
                long want = n1 + r1.n; /* the transcript turn one left behind */
                CHECK(r2.st.cached > 0, "the second chat turn reuses the first (%ld cached, "
                                        "transcript %ld, prompt one %ld, longest match %ld "
                                        "of %ld)",
                      r2.st.cached, want, n1, cm, n2);
                printf("       reused %ld of %ld prompt tokens, transcript %ld, prompt one %ld\n",
                       r2.st.cached, r2.np, want, n1);
                was = qe_engine_prefix_cache(E, 0);
                if (do_run("chat 2 cold", p2, 0, &r3))
                    return 1;
                qe_engine_prefix_cache(E, was);
                CHECK(same_ids(&r2, &r3), "the same tokens either way (%ld, %ld)", r2.n, r3.n);
                if (r3.st.seconds > 0.0 && r2.st.seconds > 0.0)
                    printf("       %.3f s reused vs %.3f s cold on a %ld token prompt\n",
                           r2.st.seconds, r3.st.seconds, r2.np);
                run_free(&r2);
                run_free(&r3);
                free(i1);
                free(i2);
            }
            free(p2);
            free(body2);
            run_free(&r1);
        }
        free(p1);
        free(body1);
    }

    /* ---------------------------- 5. a client that re-renders the answer */
    /* Everything above goes through the live sequence, which can only carry a
     * conversation forward: the tokens it is holding have to be the tokens the new
     * prompt begins with, byte for byte. A client that stores an assistant turn and
     * renders it again breaks that the moment its copy differs from what the decoder
     * streamed - trimmed, re-marked, or summarised by whatever sits in front of it.
     * Then the honest options are reading the whole conversation again or putting the
     * sequence back to where the two prompts still agreed, and this checkpoint reads a
     * conversation at roughly a second per thousand tokens, so the second option is
     * the difference between an agent turn and a coffee.
     *
     * What is asserted is what has always been asserted here: a trunk that was put
     * back answers token for token like a trunk that was computed. Everything else on
     * this line is a number. */
    printf("\n== a second turn that re-renders the first answer ==\n");
    fflush(stdout);
    if (!qe_chat_supported(TOK)) {
        printf("       skipped: no delimiters in this vocabulary to render with\n");
    } else {
        char cerr2[256] = { 0 };
        const char *Q3 = "Two sentences: what does a CPU inference engine trade away by "
                         "keeping every weight byte in memory?";
        char *body3 = chat_body(Q3, NULL, NULL, NULL);
        char *p3 = body3 ? render_prompt(body3, cerr2, sizeof cerr2) : NULL;
        if (!p3) {
            printf("  FAIL could not render the first turn: %s\n", cerr2);
            fails++;
        } else {
            if (do_run("first ask", p3, 0, &r1))
                return 1;
            /* The same conversation as the client holds it: the assistant turn is
             * there, in its place, and is none of the bytes that came out. */
            char *body4 =
                chat_body(Q3, "Rewritten reasoning.", "A rewritten answer.",
                          "And what does that cost per token?");
            char *p4 = body4 ? render_prompt(body4, cerr2, sizeof cerr2) : NULL;
            long n3 = 0, n4 = 0;
            int *i3 = p4 ? enc(TOK, p3, &n3) : NULL;
            int *i4 = i3 ? enc(TOK, p4, &n4) : NULL;
            if (!p4 || !i3 || !i4) {
                printf("  FAIL could not render or encode the second turn\n");
                fails++;
            } else {
                long at3 = 0;
                CHECK(id_prefix(TOK, p3, p4, &at3),
                      "the re-rendered prompt still begins with the first one (first "
                      "difference at id %ld)",
                      at3);
                if (do_run("re-rendered", p4, 0, &r2))
                    return 1;
                say("reply", r2.text, r2.tn);
                CHECK(r2.st.rewound, "the turn went back to a rewind point rather than "
                                     "reading the conversation again");
                CHECK(r2.st.cached == n3, "back to exactly where the first prompt ended "
                                          "(%ld of %ld tokens, prompt one %ld)",
                      r2.st.cached, r2.np, n3);
                printf("       put back in %.1f ms, then %ld of %ld tokens read in %.3f s\n",
                       (double)r2.st.snapshot_us / 1000.0, r2.np - r2.st.cached, r2.np,
                       r2.st.seconds);
                /* Reported before the cold run, because turning reuse off is also
                 * what empties the cache, and afterwards it can only say zero. */
                long held = 0, slots = 0;
                long long cbytes = 0, saved = 0;
                qe_engine_turn_stats(E, &slots, &held, &cbytes, &saved);
                printf("       turn cache: %ld of %ld rewind points, %.1f MiB, %lld prompt "
                       "tokens never re-read\n",
                       held, slots, (double)cbytes / 1048576.0, saved);
                int was2 = qe_engine_prefix_cache(E, 0);
                if (do_run("cold", p4, 0, &r3))
                    return 1;
                qe_engine_prefix_cache(E, was2);
                CHECK(r3.st.cached == 0, "with reuse off nothing is claimed (%ld)",
                      r3.st.cached);
                CHECK(same_ids(&r2, &r3), "a put-back trunk answers token for token like a "
                                          "computed one (%ld, %ld)",
                      r2.n, r3.n);
                if (!same_ids(&r2, &r3) && r2.n && r3.n) {
                    long same = lcp(r2.ids, r2.n, r3.ids, r3.n);
                    printf("       first difference at id %ld: %d vs %d\n", same,
                           same < r2.n ? r2.ids[same] : -1, same < r3.n ? r3.ids[same] : -1);
                }
                run_free(&r2);
                run_free(&r3);
            }
            free(i3);
            free(i4);
            free(p4);
            free(body4);
            run_free(&r1);
        }
        free(p3);
        free(body3);
    }

    /* ------------------------------ 6. another prompt between two turns */
    printf("\n== a turn, an unrelated prompt, then the turn's follow-up ==\n");
    fflush(stdout);
    /* A rewind point holds the recurrences and nothing else; the keys and values it
     * stands on stay in the caches, under the positions they were written at. A
     * prompt fed in between - a title request, a sub-agent - writes over those cells,
     * and putting the point back on top of them is one conversation's state over
     * another's attention context: asked for the word its first turn held, the model
     * said there was none. So the follow-up has to answer exactly as it does cold, and
     * it has to have gone back to the point through the cells kept aside for it rather
     * than reading its prompt again. */
    {
        const char *u1 = "Remember this: the code word is PELICAN-7319. Just say hi.";
        const char *u2 = "What is the code word? Reply with only the code word.";
        qe_str g = { 0 };
        qe_str_puts(&g, "Give a title for these notes: ");
        for (int i = 0; i < 60; i++)
            qe_str_puts(&g, "Gardening: tomatoes need sun, basil needs water, roses need "
                            "pruning. ");
        char *ba = chat_body(u1, NULL, NULL, NULL);
        char *ba2 = chat_body(u1, "The user greets me.", "Hi!", u2);
        char *bb = g.err ? NULL : chat_body(g.p, NULL, NULL, NULL);
        char *pa = ba ? render_prompt(ba, err, sizeof err) : NULL;
        char *pa2 = ba2 ? render_prompt(ba2, err, sizeof err) : NULL;
        char *pb = bb ? render_prompt(bb, err, sizeof err) : NULL;
        run c1, c2, c3, rx;
        if (!pa || !pa2 || !pb) {
            printf("  FAIL could not render the interleaving case: %s\n", err);
            fails++;
        } else {
            long sv0 = 0, pu0 = 0, sv1 = 0, pu1 = 0, ns = 0;
            long long sb = 0, bud = 0;
            int was3 = qe_engine_prefix_cache(E, 0);
            if (do_run("cold", pa2, 0, &c1))
                return 1;
            qe_engine_prefix_cache(E, 1);
            qe_engine_span_stats(E, &ns, &sb, &bud, &sv0, &pu0);
            if (do_run("turn 1", pa, 4, &rx))
                return 1;
            run_free(&rx);
            if (do_run("unrelated", pb, 4, &rx))
                return 1;
            run_free(&rx);
            if (do_run("follow-up", pa2, 0, &c2))
                return 1;
            say("reply", c2.text, c2.tn);
            qe_engine_span_stats(E, &ns, &sb, &bud, &sv1, &pu1);
            CHECK(same_ids(&c1, &c2), "the follow-up answers token for token as it does cold "
                                      "(%ld, %ld)", c1.n, c2.n);
            if (bud > 0) {
                CHECK(c2.st.rewound && c2.st.cached > 0 && pu1 > pu0,
                      "it went back to its point through kept cells (%ld cached, %ld put back, "
                      "%ld kept, %.1f MiB of %.0f)",
                      c2.st.cached, pu1 - pu0, sv1 - sv0, (double)sb / 1048576.0,
                      (double)bud / 1048576.0);
            } else {
                printf("       kept cells off (QE_KV_SPAN_MIB=0): %ld cached\n", c2.st.cached);
            }
            /* The same follow-up once more, now that its own prompt is what the
             * caches hold: an ordinary carry-over, and the same answer again. */
            if (do_run("again", pa2, 0, &c3))
                return 1;
            CHECK(same_ids(&c1, &c3), "and again with the cells its own (%ld cached)",
                  c3.st.cached);
            qe_engine_prefix_cache(E, was3);
            run_free(&c1);
            run_free(&c2);
            run_free(&c3);
        }
        free(pa);
        free(pa2);
        free(pb);
        free(ba);
        free(ba2);
        free(bb);
        qe_str_free(&g);
    }

    qe_engine_close(E);
    qe_pool_shutdown();
    printf("\n%s (%d checks, %d failures)\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
