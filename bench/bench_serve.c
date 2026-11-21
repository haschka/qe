/* What a second request costs.

   Every ingestion number in this repository answers the question "how long does one
   prompt take", and one prompt is the easy case: it has the machine to itself, it
   uses all of it, and there is nobody else's turn behind it. A server does not get
   that case. Two clients arrive three seconds apart, the first one holds the engine
   for the whole of its ingestion, and the question that matters is not the sum of the
   two walls but when the second one sees its first character.

   This is that measurement, driven through the engine's own request entry
   (`qe_engine_generate_tokens`, the same call `serve.c` reaches) rather than through
   `qe_forward_*`, because the thing being measured is the queue in front of the
   network as much as the network: one lock is held from the tokeniser to the last
   sampled token (src/engine.c), so a wave of requests is a wave of waits, and the
   waits are the product.

   Three things are reported and one is asserted.

     1. Reported: the solo reference, at each prompt width - time to first token,
        wall, and what the answer cost after the prompt was in. Everything else in
        the file is read as a multiple of these.
     2. Reported: a wave. N requests with a fixed spacing between their arrivals, each
        one timed from arrival to first token and to last, and the percentiles of both.
        A bar is printed at several heights (250, 500, 1000, 2000 ms) rather than one,
        because the share of requests a bar lets through is a function of where the bar
        is, and a single number invites a comparison to a bar nobody wrote down.
     3. Reported: the makespan of the wave against the sum of the same requests run
        one at a time. Those two are equal exactly when the engine has no concurrency
        to find, which is the shape of the answer today and is worth printing rather
        than assuming either way.
     4. Asserted: every request in a wave completes and emits what it was asked for,
        and a wave's makespan is not *below* the sum of its parts by more than a few
        percent. The second of those is a ratchet on a property rather than a wish: if
        it ever fails, the engine started overlapping requests, and every wait figure
        in this file and in docs/DESIGN.md needs rewriting at the same time.

   The prompts are synthetic and say so. They are built a line at a time, each line
   numbered, because repeating one 512-token block until it reaches 2048 tokens routes
   the same ten experts again and again and reads as a wide chunk the real router never
   makes (the §7.3b warning, in the other direction). A synthetic prompt is still not a
   transcript: what it routes is plausible and it is not the traffic, so the ratios
   inside a wave travel and the absolute waits do not.

     make serve                        # or:
     ./bin/bench_serve --model <vol1.gguf> --threads auto --max 24
*/
#include "qe.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static qe_engine *E;
static qe_tok *TOK;
static int MAX_NEW = 24;
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

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void wait_until(double when) {
    for (;;) {
        double d = when - now_s();
        if (d <= 0.0)
            return;
        if (d > 0.002)
            d = 0.002;
        struct timespec ts;
        ts.tv_sec = (time_t)d;
        ts.tv_nsec = (long)((d - (double)ts.tv_sec) * 1e9);
        nanosleep(&ts, NULL);
    }
}

/* ------------------------------------------------------------------ requests */

typedef struct {
    const char *text;   /* the rendered prompt, chat transcript included */
    double arrive;      /* seconds after the start of the wave to submit it */
    double t0;          /* the wave's start */
    /* results, written by the issuing thread only */
    double submitted;
    double first;
    double done;
    long emitted;
    long cached;
    long dropped;
    long fed;
    int ok;
} req;

static int on_token(void *u, const qe_token_out *t) {
    req *r = u;
    (void)t;
    if (!r->first)
        r->first = now_s();
    r->emitted++;
    return 0;
}

static void *issue(void *arg) {
    req *r = arg;
    wait_until(r->t0 + r->arrive);
    r->submitted = now_s();
    qe_prompt p = { .text = r->text, .is_chat = 1 };
    qe_gen_opts o = { .max_tokens = MAX_NEW, .temp = 0.0f, .seed = 20261116, .spec_k = 0, .eos = -1 };
    qe_gen_stats st;
    memset(&st, 0, sizeof st);
    long rc = qe_engine_generate_tokens(E, &p, &o, on_token, r, &st);
    r->done = now_s();
    r->ok = rc >= 0;
    r->emitted = st.tokens;
    r->cached = st.cached;
    r->dropped = st.dropped;
    r->fed = st.steps;
    return NULL;
}

/* One generation, timed the way a client would time it. */
static req solo(const char *text) {
    req r;
    memset(&r, 0, sizeof r);
    r.text = text;
    r.arrive = 0.0;
    r.t0 = now_s();
    issue(&r);
    return r;
}

/* Stand between two measurements: the engine reuses the sequence it is holding, and a
 * bench that measures a second request without clearing that is measuring a client that
 * resent its own transcript. Both are real clients and they are measured apart here. */
static void idle_engine(void) {
    qe_engine_prefix_cache(E, 0);
    qe_engine_prefix_cache(E, 1);
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static double pct(double *v, int n, double p) {
    if (!n)
        return 0.0;
    int i = (int)(p * (double)(n - 1) + 0.5);
    return v[i];
}

/* A wave: n requests, `space` seconds between arrivals, printed as a table of waits
 * and as the share of them a bar of several heights would let through. The requests
 * alternate between two prompts, which is what two clients are: neither one's prompt is
 * the sequence the other left standing, so nothing is reused and every one of them pays
 * for its whole ingestion. `resend` repeats one prompt instead, which is the same client
 * asking twice and is the case the prefix cache exists for. */
static double wave(const char *label, const char *const *texts, int n, double space,
                   int resend) {
    pthread_t th[8];
    req r[8];
    if (n > 8)
        n = 8;
    double t0 = now_s();
    for (int i = 0; i < n; i++) {
        memset(&r[i], 0, sizeof r[i]);
        r[i].text = resend ? texts[0] : texts[i % 2];
        r[i].arrive = (double)i * space;
        r[i].t0 = t0;
        if (pthread_create(&th[i], NULL, issue, &r[i])) {
            fprintf(stderr, "no thread for request %d\n", i);
            exit(1);
        }
    }
    for (int i = 0; i < n; i++)
        pthread_join(th[i], NULL);
    double makespan = 0.0;
    long dropped = 0;
    for (int i = 0; i < n; i++) {
        if (r[i].done - t0 > makespan)
            makespan = r[i].done - t0;
        dropped += r[i].dropped;
    }

    double ttft[8], wall[8];
    int nttft = 0, nwall = 0;
    printf("  %s: %d request(s), %.2f s apart\n", label, n, space);
    printf("  %3s %8s %8s %8s %8s %9s\n", "i", "queue ms", "first ms", "wall ms", "cached",
           "tok/s");
    fflush(stdout);
    for (int i = 0; i < n; i++) {
        double queued = (r[i].submitted - (r[i].t0 + r[i].arrive)) * 1e3;
        double first = (r[i].first - r[i].submitted) * 1e3;
        double wallms = (r[i].done - r[i].submitted) * 1e3;
        double gen = r[i].done - r[i].first;
        printf("  %3d %8.0f %8.0f %8.0f %8ld %9.2f%s\n", i, queued, first, wallms, r[i].cached,
               gen > 0.0 ? (double)r[i].emitted / gen : 0.0, r[i].ok ? "" : "  FAILED");
        if (r[i].first)
            ttft[nttft++] = first;
        if (r[i].ok)
            wall[nwall++] = wallms;
    }
    qsort(ttft, (size_t)nttft, sizeof *ttft, cmp_d);
    qsort(wall, (size_t)nwall, sizeof *wall, cmp_d);
    printf("  p50 first %.0f ms, p99 first %.0f ms; p50 wall %.0f ms, p99 wall %.0f ms\n",
           pct(ttft, nttft, 0.50), pct(ttft, nttft, 0.99), pct(wall, nwall, 0.50),
           pct(wall, nwall, 0.99));
    static const int bars[] = { 250, 500, 1000, 2000 };
    printf("  inside the bar, by time to first token:");
    for (unsigned b = 0; b < sizeof bars / sizeof bars[0]; b++) {
        int in = 0;
        for (int i = 0; i < nttft; i++)
            if (ttft[i] <= (double)bars[b])
                in++;
        printf(" %d ms: %d/%d", bars[b], nttft ? in : 0, nttft);
    }
    printf("\n  makespan of the wave %.2f s\n", makespan);
    {
        char what[192];
        snprintf(what, sizeof what,
                 "no request in this wave lost the front of its transcript to the window"
                 " (%ld tokens dropped)",
                 dropped);
        checkf(dropped == 0, what);
    }
    fflush(stdout);
    return makespan;
}

/* ------------------------------------------------------------------ prompts */

/* About `want` tokens of numbered prose. The numbers are the point twice over: a prompt
 * made by repeating one block routes that block's experts once per repeat (the §7.3b
 * mistake), and two prompts that begin identically are not two clients but one client
 * that resent its transcript, which is the case the prefix cache is built for and has to
 * be measured separately. `variant` moves the numbering, so variant 0 and variant 1 of a
 * width share no prefix at all. */
static char *make_prompt(long want, int variant) {
    static const char *nouns[] = { "panel", "listener", "bucket", "window", "chunk",
                                   "row",   "trace",   "quota", "queue",  "cache" };
    size_t cap = (size_t)want * 8 + 4096;
    char *out = malloc(cap);
    if (!out)
        return NULL;
    size_t len = 0;
    out[0] = 0;
    char line[256];
    const int base = 1 + variant * 100003;
    for (int i = 0;; i++) {
        int k = snprintf(line, sizeof line,
                         "%d. the %s held its place while frame %d was drawn, and the next "
                         "row began where the last one ended.\n",
                         base + i, nouns[(i + variant) % (int)(sizeof nouns / sizeof nouns[0])],
                         (base + i) * 7 + 3);
        if (k <= 0 || len + (size_t)k + 1 > cap)
            break;
        memcpy(out + len, line, (size_t)k + 1);
        len += (size_t)k;
        if (qe_tok_count(TOK, out) >= want)
            break;
    }
    /* Trim to whole lines below the target: an approximate prompt width is honest,
     * a width that cuts a word in the middle is not reproducible. */
    for (;;) {
        long n = qe_tok_count(TOK, out);
        if (n <= want)
            break;
        char *nl = strrchr(out, '\n');
        if (!nl || nl == out)
            break;
        *nl = 0;
    }
    return out;
}

/* --------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    const char *path = "/mnt/data/data/ai-models/llama-cpp/llm/"
                       "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    int threads = QE_THREADS_AUTO, n_ctx = 4096, fuse = 0;
    long wide = 2048, narrow = 512;
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
        else if (!strcmp(argv[i], "--wide") && i + 1 < argc)
            wide = atol(argv[++i]);
        else if (!strcmp(argv[i], "--narrow") && i + 1 < argc)
            narrow = atol(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s [--model PATH] [--threads N] [--max N] [--ctx N] "
                    "[--fuse S] [--wide N] [--narrow N]\n",
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
    eo.id = "bench-serve";
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

    char *pn[2] = { make_prompt(narrow, 0), make_prompt(narrow, 1) };
    char *pw[2] = { make_prompt(wide, 0), make_prompt(wide, 1) };
    if (!pn[0] || !pn[1] || !pw[0] || !pw[1]) {
        fprintf(stderr, "out of memory building prompts\n");
        return 1;
    }

    /* ------------------------------------------- 1. the solo reference points */
    printf("\n== one request on an idle engine ==\n");
    printf("  %9s %10s %10s %10s %8s\n", "prompt", "first ms", "wall ms", "answer",
           "tok/s");
    fflush(stdout);
    double solo_n = 0.0, solo_w = 0.0;
    {
        const long want[2] = { narrow, wide };
        const char *txt[2] = { pn[0], pw[0] };
        double *dst[2] = { &solo_n, &solo_w };
        for (int i = 0; i < 2; i++) {
            idle_engine();
            req r = solo(txt[i]);
            double gen = r.done - r.first;
            printf("  %9ld %10.0f %10.0f %10ld %8.2f\n", (long)qe_tok_count(TOK, txt[i]),
                   (r.first - r.submitted) * 1e3, (r.done - r.submitted) * 1e3, r.emitted,
                   gen > 0.0 ? (double)r.emitted / gen : 0.0);
            checkf(r.ok && r.emitted >= MAX_NEW, "request %ld/%d completed and emitted %d tokens",
                   want[i], MAX_NEW, MAX_NEW);
            /* A solo request on an engine that was just cleared has nothing to reuse.
             * If it found something, every figure below is a reuse number and not a
             * queue number, and the two are different products. */
            checkf(r.cached == 0, "an idle engine reused nothing (cached %ld)", r.cached);
            *dst[i] = r.done - r.submitted;
            fflush(stdout);
        }
    }

    /* -------------------------------------------------------- 2. a wave of two */
    /* Spacings that bracket the solo wall at each width: below it the second request
     * is queued behind a live one, above it the engine is idle when the second one
     * arrives and the wave is two solos with a gap in it. Reading the two apart is
     * the point of the sweep. */
    double solos[2] = { solo_n, solo_w };
    const char *names[2] = { "narrow", "wide" };
    const char *txt[2][2] = { { pn[0], pn[1] }, { pw[0], pw[1] } };
    for (int w = 0; w < 2; w++) {
        printf("\n== %s prompt (%.2f s alone): two clients, their own prompts, spaced ==\n",
               names[w], solos[w]);
        fflush(stdout);
        double spaces[4];
        int ns = 0;
        for (int k = 0; k < 4; k++) {
            double s = solos[w] * (k == 0 ? 0.0 : (k == 1 ? 0.25 : (k == 2 ? 0.5 : 2.0)));
            spaces[ns++] = s;
        }
        for (int k = 0; k < ns; k++) {
            char label[96];
            snprintf(label, sizeof label, "spacing %.2f s", spaces[k]);
            idle_engine();
            double ms = wave(label, txt[w], 2, spaces[k], 0);
            if (k == 0) {
                /* The ratchet: two together against two one at a time. Equal means the
                 * second one waited for the first the whole way, which is what the
                 * engine lock in src/engine.c says it does. */
                double sum = 2.0 * solos[w];
                char what[160];
                snprintf(what, sizeof what,
                         "two together cost the sum of two alone (%.2f against %.2f s): the "
                         "queue is serial, and every wait figure written with it needs"
                         " rewriting if this ever stops holding",
                         ms, sum);
                checkf(ms >= 0.92 * sum, what);
            }
        }
        /* The other client: the same transcript sent twice, which is every chat client
         * that regenerates and every agent that resends a conversation. The engine
         * answers that one by putting only what was added through the network, so it
         * queues for much less than a prompt's worth, and it is the cheapest serving
         * win in the file precisely because it does not need the machine shared. */
        char label[96];
        snprintf(label, sizeof label, "same prompt twice, no spacing");
        idle_engine();
        wave(label, txt[w], 2, 0.0, 1);
    }

    printf("\n== four clients at once, their own prompts, no spacing ==\n");
    fflush(stdout);
    idle_engine();
    wave("four narrow", txt[0], 4, 0.0, 0);

    printf("\n%d checks, %d failed\n", checks, fails);
    qe_engine_close(E);
    qe_pool_shutdown();
    return fails ? 1 : 0;
}
