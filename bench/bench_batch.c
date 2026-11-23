/* B sequences through one walk of the weights.

   A lone decode step is not slow because the engine is careless: it reads 5568 MiB of
   weights once at 76-79 GB/s, which is 97 % of the fastest linear read this host does
   (docs/DESIGN.md 1.2), so the only way to make one token cheaper is to read those bytes
   for more than one token. That is what qe_forward_step_batch does, and this is the
   instrument that says whether it does it honestly and what it is worth.

   Two questions, in that order, because a fast wrong answer is worse than no answer.

   1. IS A ROW THE SAME ARITHMETIC AS THAT SEQUENCE ALONE? Each sequence's baseline is a
      single-sequence pass, greedy, recorded token by token and logit by logit. The same
      token streams then go through a batched pass and the rows are compared to the
      recording - a memcmp of them, not a tolerance. A batch of B rows is a chunk of B
      tokens, and a chunk of B is not a chunk of one: which bytes a walk reads is a
      function of the width. So the comparison runs per arithmetic, and the arithmetic is
      the same on both sides of every row it reports (see the three arms below).

   2. WHAT IS IT WORTH, AND WHAT DOES IT COST THE ANSWER? Aggregate tokens per second at
      B = 1, 2, 4, 8 against a single-stream recording of the same streams taken on the
      same arithmetic. Two numbers come out of that: the multiplier, and the move in the
      logprob of the token the engine would have said, which is the price of whatever the
      wider chunk read differently. The B=1 row of each table is the control: a batch of
      one is the same chunk of one token on the same copies, so it has to be byte-identical
      and 1.00x, and if it is not, nothing in the rows below it means anything either.

   3. IS A ROW STILL ITSELF WHEN THE STEP IS ONLY PARTLY FULL? Leg 2 hands every step the
      sequences 0..B-1 in order, which is what qe_forward_step_batch is and not what a
      serving lane is: a lane has prompts arriving mid-conversation, a row map that
      renumbers itself whenever a sibling finishes, and slots recycled two steps after a
      sequence was reset. That is qe_forward_step_batch_rows, and it is leg 1's question
      asked again over a map that moves.

   The three arithmetics, because one number cannot tell a scheduling change from a
   requantisation:

     exact - every width-dependent walk pinned off on both sides. A batch is then a
             scheduling change and has to report zero, at every width, or it is a bug.
     walk  - the two walks a chunk exists to run: one weight walk over the rows, and the
             expert stage bucketed across them. Neither requantises anything, so what this
             arm costs is reassociation and nothing else. It is the arm a shipped batch
             runs on unless the next one is paid for.
     ship  - everything the engine ships turned on, including the load-time copies that a
             wider chunk reads instead: the int8 panel, the flat and IQ4_NL dense copies,
             and the head copy a window reads. Those are requantisations of the weights
             this pass is holding, so this arm is where a quality cost can hide, and its
             nats column is the bill for it.

   Every run is deadline-budgeted (a stalled phase reports nothing and looks like a hang),
   and nothing here reads the checkpoint twice: the single-sequence pass is closed before
   the batched one opens, because 87 GiB of weights are resident and two passes' worth of
   requantised copies is a host that does not exist. */
#include "qe.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

/* The prompt every sequence is cut from: ordinary English, because the alternative is a
   stream of ids drawn from the vocabulary, and that is not a language. A uniform draw over
   248,320 ids reaches the trunk as a sequence of embeddings no layer has ever seen, the
   residuals run away, and the pass reports a non-finite value at the fortieth token - which
   is the pass being right and the instrument being wrong. Real text keeps the arithmetic in
   the range where a logprob means something, which is the whole point of leg 2.

   It is long on purpose: eight sequences are cut out of it, each starting `stride` tokens
   after the last, and an eight-way overlap of the same paragraph is eight sequences that
   agree about which experts to route to, which is the one condition a bucketed expert pass
   does not have to survive. */
static const char *kCorpus =
    "A decode step is a memory instruction with a model attached. The weights of this "
    "checkpoint are eighty seven gigabytes, and a single token walks five and a half "
    "gigabytes of them: the attention blocks, the delta net, the indexer that decides which "
    "of the cached keys are worth reading, ten routed experts out of five hundred and "
    "twelve, a shared expert, four hyper connections a layer, and the output head at the "
    "end. None of that is recomputed twice for one token, and almost all of it is read once, "
    "which is why the fastest engine on this machine is the one that has already put the "
    "bytes in memory and the one that can point one read at more than one answer. "
    "Scheduling is free where arithmetic is not. A chunk of prompt tokens shares a weight "
    "walk, and the walk is the difference between reading a tensor once and reading it once "
    "per token, so the width of a chunk is a real dial and not a constant someone forgot "
    "to tune. What a chunk cannot share is anything with a history: a recurrent state, a "
    "convolution ring, a cache of keys, and the window of text the n gram table hashes "
    "over. Those belong to one sequence, and every one of them is still exactly right after "
    "the four hundredth token of a conversation. "
    "Measure the thing you are about to optimise, and keep the instrument. Four rewrites of "
    "one walk raced each other in a single process before any of them was believed, because "
    "a run that pays twelve minutes of loading only gets to answer one question, and a "
    "question that turns out to be the wrong one costs the afternoon. "
    "A router is a classifier wearing a softmax, and it is the reason a mixture of experts "
    "is a memory system and not a computation: the ten experts a token asks for are ten "
    "pointers into a pool of five hundred and twelve, and a second token in the same step "
    "asks for ten more, and a third for ten more still. Group the requests and the pool is "
    "read once for both; do not group them and it is read twice, which is the same mistake "
    "a naive batch would make of the whole model. Skew is the failure mode nobody predicts "
    "and everybody meets: a router that loves three experts hands three buckets all the "
    "work and five hundred and nine the afternoon off, and a schedule that slices by rows "
    "instead of by cost watches the bus go idle. "
    "Bandwidth is a ceiling and not a wall, and the way to tell the two apart is to read "
    "something that is not a weight. A linear run over the cache of keys is seventy six to "
    "seventy nine gigabytes a second on nineteen threads, a quantised matrix vector product "
    "reaches seventy six, and a flat read on eight performance cores gets eighty five. A "
    "decode step sits at ninety seven percent of the first number, so a step is not slow "
    "because a kernel is careless; it is slow because it is one token. The fix for one token "
    "is not a better multiply. It is more tokens. "
    "Quality is a number too, and it is the one an optimisation gets to argue with. Perplexity "
    "over a held out panel, the logprob of the token the engine goes on to say, and the count "
    "of greedy choices that move: those three and not a benchmark. A requantisation that buys "
    "six percent of a step for three hundredths of a nat is a trade worth making twice and "
    "measuring three times, and one that buys the same for half a nat is a different product. "
    "Say which of the two it is with both numbers, and let the reader decide once, in writing.";

/* Sequence `s`'s `len` tokens: the corpus from an offset of its own, so that B sequences in
   one batch want different experts, choose different cached blocks, and hash different
   n-grams, which is the condition a shared weight walk has to survive. */
static void fill_stream(int *out, int len, const int *base, long nb, long off) {
    for (int i = 0; i < len; i++)
        out[i] = base[(off + i) % nb];
}

static long argmax(const float *v, long n) {
    long best = 0;
    for (long i = 1; i < n; i++)
        if (v[i] > v[best])
            best = i;
    return best;
}

/* log P(chosen | row), in nats. The coarse check on a batched row is that greedy says the
   same token; this is the fine one, the same quantity `make score` prices, over the rows a
   batch actually ran. */
static double logprob(const float *lg, long vocab, long chosen) {
    double m = lg[0];
    for (long i = 1; i < vocab; i++)
        if (lg[i] > m)
            m = lg[i];
    double sum = 0.0;
    for (long i = 0; i < vocab; i++)
        sum += exp((double)lg[i] - m);
    return (double)lg[chosen] - (m + log(sum));
}

/* One sequence's recorded greedy run. Row j of `lg` are the logits that chose `tok[j]`,
 * so `tok[j]` is what the next step is fed and row j+1 is what it answers. Feeding the
 * recorded tokens instead of the run's own keeps a comparison honest: a batch that went
 * somewhere else at row 9 still gets row 10 to compare, and says so as a difference
 * instead of silently diverging onto its own continuation. */
typedef struct {
    int *tok;
    float *lg;
    double seconds;
    long steps;
} rec;

static rec *rec_new(int n, long vocab) {
    rec *r = calloc(1, sizeof *r);
    if (!r)
        return NULL;
    r->tok = malloc((size_t)(n + 1) * sizeof *r->tok);
    /* n + 1 rows: the row that chose the first fed token plus one per step. */
    r->lg = malloc((size_t)(n + 1) * (size_t)vocab * sizeof *r->lg);
    if (!r->tok || !r->lg) {
        free(r->tok);
        free(r->lg);
        free(r);
        return NULL;
    }
    r->steps = n;
    return r;
}
static void rec_free(rec *r) {
    if (!r)
        return;
    free(r->tok);
    free(r->lg);
    free(r);
}

/* The arithmetic a walk runs on, as the seven switches that say which bytes it reads.
 *
 * `walk` is the chunk's own projection walk (one weight walk over the rows in flight) and
 * `grouped` is the expert stage bucketed across them; neither is a requantisation, so the
 * worst they do to a row is reassociate its adds. `i8`, `flat` and `i4` are the load-time
 * copies a chunk of two rows and up reads instead of the tensor as stored, and `head` is
 * the copy of the output head a window of rows reads instead of the row of one. Those four
 * are requantisations of bytes this pass is already holding, and they are what the nats
 * column of a table is a bill for. */
typedef struct {
    size_t walk;
    int i8, flat, grouped, i4, head, head_win;
} pins;

enum { ARM_EXACT = 0, ARM_WALK, ARM_SHIP, NARM };

static const struct {
    const char *tag;
    const char *what;
} kArm[NARM] = {
    { "exact", "every width-dependent walk off: a batch is a scheduling change or a bug" },
    { "walk",  "the chunk's projection walk and the bucketed experts, no requantised copies" },
    { "ship",  "every shipping switch on, load-time copies of the weights included" },
};

static void pins_capture(pins *p, const qe_forward *f) {
    p->walk = qe_forward_get_proj_chunk(f);
    p->i8 = qe_forward_get_proj_i8(f);
    p->flat = qe_forward_get_flat_i8(f);
    p->grouped = qe_forward_get_moe_grouped(f);
    p->i4 = qe_forward_get_decode_i4(f);
    p->head = qe_forward_get_head_copy(f);
    p->head_win = qe_forward_get_head_window(f);
}
/* The arm's switches, applied to a pass. `base` is what the pass opened with, which is
 * what ARM_SHIP is, and the other two are it with names struck out - so an engine that
 * ships a different default is measured on a different `exact` in exactly the same way
 * and the arm still means the same thing. */
static void pins_apply(qe_forward *f, const pins *base, int arm) {
    pins p = *base;
    if (arm == ARM_EXACT) {
        p.walk = 0;
        p.grouped = 0;
    }
    if (arm <= ARM_WALK) {
        p.i8 = 0;
        p.flat = 0;
        p.i4 = 0;
        p.head = QE_HEAD_STORED;
        p.head_win = 0;
    }
    qe_forward_set_proj_chunk(f, p.walk);
    qe_forward_set_proj_i8(f, p.i8);
    qe_forward_set_flat_i8(f, p.flat);
    qe_forward_set_moe_grouped(f, p.grouped);
    qe_forward_set_decode_i4(f, p.i4);
    qe_forward_set_head_copy(f, p.head);
    qe_forward_set_head_window(f, p.head_win);
}

/* Walk one sequence's prompt and then its own greedy tokens, recording every row. Only
 * ever called on a single-sequence pass, which is what makes it the thing a batch is
 * compared against rather than another batch. The pass is reset first, so `s` is only ever
 * in the messages. */
static int record_run(qe_forward *f, int s, const int *prompt, int plen, rec *out,
                      int warm) {
    const long vocab = qe_forward_vocab(f);
    qe_forward_reset(f);
    if (qe_forward_prefill_seq(f, 0, prompt, plen, out->lg))
        return -1;
    out->tok[0] = (int)argmax(out->lg, vocab);
    for (int k = 0; k < out->steps; k++) {
        const double t0 = now();
        const int rc = qe_forward_step_seq(f, 0, out->tok[k],
                                           out->lg + (size_t)(k + 1) * (size_t)vocab);
        const double dt = now() - t0;
        if (rc) {
            fprintf(stderr, "bench_batch: step failed on sequence %d at %d (%d)\n", s, k, rc);
            return rc;
        }
        /* The first steps of a pass pay for whatever the buffers have not touched yet,
         * which is a page fault and not a measurement. */
        if (k >= warm)
            out->seconds += dt;
        out->tok[k + 1] = (int)argmax(out->lg + (size_t)(k + 1) * (size_t)vocab, vocab);
    }
    return 0;
}

/* One (arm, width) row of the table, as measured. */
typedef struct {
    int bad;
    long pl_rows, pl_same;
    long rows, same, hits;
    int bad_s, bad_k;
    double maxdl, nats_got, nats_ref;
    double seconds;
} result;

int main(int argc, char **argv) {
    const char *path = NULL;
    int threads = 0; /* 0 = the pool's size */
    int n_ctx = 4096;
    int plen = 48, steps = 24; /* the three arithmetics run on these */
    int warm = 2;
    int deadline = 1500;
    const char *seqs_s = "1,2,4,8";
    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)
            n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc)
            plen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc)
            warm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq") && i + 1 < argc)
            seqs_s = argv[++i];
        else if (!strcmp(argv[i], "--deadline") && i + 1 < argc)
            deadline = atoi(argv[++i]);
        else if (!path)
            path = argv[i];
        else {
            fprintf(stderr,
                    "usage: bench_batch MODEL [--threads N] [--ctx N] [--prompt N] [--steps N] "
                    "[--warm N] [--seq 1,2,4,8] [--deadline S]\n");
            return 2;
        }
    }
    if (!path) {
        fprintf(stderr, "bench_batch: a checkpoint is required\n");
        return 2;
    }
    int want[QE_MAX_SEQ], nwant = 0;
    for (const char *p = seqs_s; p && *p;) {
        int v = atoi(p);
        if (v >= 1 && v <= QE_MAX_SEQ && nwant < QE_MAX_SEQ)
            want[nwant++] = v;
        const char *c = strchr(p, ',');
        if (!c)
            break;
        p = c + 1;
    }
    if (!nwant)
        want[nwant++] = 1;
    int bmax = 1;
    for (int w = 0; w < nwant; w++)
        if (want[w] > bmax)
            bmax = want[w];
    /* Every sequence of the widest batch asked for has to hold a prompt and a run of steps
     * in its own slice of the cells, and the batched pass is one open for all of them. */
    if ((long)n_ctx / bmax < plen + steps + 2) {
        fprintf(stderr,
                "bench_batch: %d cells split %d ways is %ld a sequence, below the %d a prompt "
                "of %d and %d steps needs\n",
                n_ctx, bmax, (long)n_ctx / bmax, plen + steps + 2, plen, steps);
        return 2;
    }
    watchdog(deadline);
    if (threads == 0)
        threads = qe_pool_recommended();
    qe_pool_init(threads, NULL, 0);

    qe_model *m = qe_open(path, QE_M_NONE);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", qe_strerror());
        return 1;
    }
    printf("== weights %s (%.1f GiB placed) ==\n", qe_model_stage_method(m),
           (double)qe_model_stage_bytes(m) / 1073741824.0);
    printf("== B sequences in one step: %d threads, %d cells, prompt %d, %d steps ==\n",
           qe_pool_size(), n_ctx, plen, steps);
    fflush(stdout);

    /* The corpus, as ids. A batch of B sequences wants B streams that are each ordinary
     * text and are not each other, so one passage encoded once and entered at B spread
     * offsets is the whole of what this needs. */
    char terr[256] = { 0 };
    qe_tok *tok = qe_tok_open(m, terr, sizeof terr);
    if (!tok) {
        fprintf(stderr, "bench_batch: no tokenizer (%s), and a stream of random ids is not "
                        "a language model's input\n",
                terr);
        qe_close(m);
        return 1;
    }
    long nb = qe_tok_count(tok, kCorpus) + 8;
    int *corpus = malloc((size_t)nb * sizeof *corpus);
    nb = qe_tok_encode(tok, kCorpus, corpus, nb);
    const long stride = (long)plen + steps + 8;
    if (nb < stride) {
        fprintf(stderr, "bench_batch: the corpus encoded to %ld tokens, below the %ld one "
                        "sequence needs to start clear of the next\n",
                nb, stride);
        return 1;
    }
    printf("  corpus: %ld tokens, a sequence every %ld of them%s\n", nb, stride,
           nb / stride >= bmax ? "" : " (fewer than the batch, so they wrap and overlap)");
    fflush(stdout);

    char ferr[512] = { 0 };
    qe_forward_opts fo = { .n_ctx = n_ctx, .n_threads = threads };
    qe_forward *f1 = qe_forward_open(m, &fo, ferr, sizeof ferr);
    if (!f1) {
        fprintf(stderr, "qe_forward_open: %s\n", ferr);
        return 1;
    }
    const long vocab = qe_forward_vocab(f1);
    pins base;
    pins_capture(&base, f1);
    size_t st1 = 0, stb = 0;
    qe_forward_state_bytes(f1, &st1, NULL);

    /* Every sequence is recorded once per arithmetic, and every recording is read back by
     * the batched pass downstream. */
    rec *recs[NARM][QE_MAX_SEQ];
    int *prompt = malloc((size_t)plen * sizeof *prompt);
    if (!prompt) {
        fprintf(stderr, "bench_batch: out of memory\n");
        return 1;
    }
    for (int a = 0; a < NARM; a++)
        for (int s = 0; s < bmax; s++) {
            recs[a][s] = rec_new(steps, vocab);
            if (!recs[a][s]) {
                fprintf(stderr,
                        "bench_batch: out of memory for the logits (%d rows x %ld vocab x %d "
                        "sequences x %d arithmetics)\n",
                        steps, vocab, bmax, NARM);
                return 1;
            }
        }
    printf("  vocab %ld, %.2f GiB of state at n_seq 1, %.0f MiB of logits recorded per "
           "arithmetic\n",
           vocab, (double)st1 / 1073741824.0,
           (double)bmax * (steps + 1) * (double)vocab * sizeof(float) / 1048576.0);
    fflush(stdout);

    /* ---- 1. the recordings: each sequence walked alone, once per arithmetic ----------- */
    printf("\n== recordings: each sequence walked alone ==\n");
    fflush(stdout);
    double single_ms[NARM];
    for (int a = 0; a < NARM; a++) {
        pins_apply(f1, &base, a);
        double seconds = 0.0;
        for (int s = 0; s < bmax; s++) {
            fill_stream(prompt, plen, corpus, nb, (long)s * stride);
            if (record_run(f1, s, prompt, plen, recs[a][s], warm) < 0)
                return 1;
            seconds += recs[a][s]->seconds;
        }
        single_ms[a] = 1000.0 * seconds / ((double)steps * bmax);
        printf("  %-6s %d sequences x %d steps, %7.2f ms/token = %5.2f tok/s single "
               "stream\n",
               kArm[a].tag, bmax, steps, single_ms[a], 1000.0 / single_ms[a]);
        fflush(stdout);
    }
    /* How big a logit this head produces, so that the worst-move column of the tables
     * below has a unit next to it. */
    double head_scale = 0.0;
    for (int s = 0; s < bmax; s++)
        for (long k = 0; k < recs[ARM_SHIP][s]->steps; k++)
            for (long v = 0; v < vocab; v++) {
                const double a = fabs((double)recs[ARM_SHIP][s]->lg[(size_t)k * (size_t)vocab + v]);
                if (a > head_scale)
                    head_scale = a;
            }
    printf("  largest logit in the recording: %.1f\n", head_scale);
    fflush(stdout);

    int fails = 0;

    /* A draft window is the one thing these rows are not: T rows of *one* sequence
     * walked in one chunk, the state after every row but the last kept so that the draft
     * nobody accepted can be walked back out of the conversation. Whether that is worth
     * anything turns on one detail - a rewind moves the sequence's position along with
     * its matrices, because a step taken at the position the abandoned rows reached
     * reads the key-value cells of tokens the sequence never said. It chooses the same
     * token doing that, which is how a step of four rows and the step out of a rewind of
     * it both passed a greedy-only check while the ids they go on to produce were not
     * the ids the sequence produces alone. A window is refused for a batched pass, so
     * this asks the single-sequence one, on the arm where a rewind is exact or wrong. */
    {
        const int win = 4;
        int wtok[QE_MAX_SEQ];
        for (int j = 0; j < win; j++)
            wtok[j] = corpus[((long)plen + j) % nb];
        float *wref = malloc((size_t)(win + 2) * (size_t)vocab * sizeof *wref);
        qe_ckpt *ck = wref ? qe_ckpt_new(f1, win) : NULL;
        if (!ck) {
            fprintf(stderr, "bench_batch: out of memory for the draft-window leg\n");
            return 1;
        }
        float *wgot = wref + (size_t)win * (size_t)vocab;
        pins_apply(f1, &base, ARM_EXACT);
        fill_stream(prompt, plen, corpus, nb, 0);
        int rc = 0, landed = 0, same = 0, read_ok = 0, first = -1;
        /* Two walkings of one prompt have to land in one place, or nothing below is
         * comparing the window to the steps it replaces. */
        qe_forward_reset(f1);
        rc = qe_forward_prefill(f1, prompt, plen, wgot);
        for (int j = 0; !rc && j < win; j++)
            rc = qe_forward_step(f1, wtok[j], wref + (size_t)j * (size_t)vocab);
        qe_forward_reset(f1);
        if (!rc)
            rc = qe_forward_prefill(f1, prompt, plen, wgot + (size_t)vocab);
        if (!rc)
            landed = !memcmp(wgot, wgot + (size_t)vocab, (size_t)vocab * sizeof *wgot);
        if (!rc)
            rc = qe_forward_step_window(f1, wtok, win, ck);
        for (int b = 1; !rc && b < win; b++) {
            if (qe_forward_rewind(f1, ck, b) < 0) {
                rc = -1;
                break;
            }
            if (!qe_forward_row_logits(f1, b - 1, wgot + (size_t)vocab) &&
                !memcmp(wgot + (size_t)vocab, wref + (size_t)(b - 1) * (size_t)vocab,
                        (size_t)vocab * sizeof *wgot))
                read_ok++;
            if (qe_forward_step(f1, wtok[b], wgot + (size_t)vocab)) {
                rc = -1;
                break;
            }
            if (!memcmp(wgot + (size_t)vocab, wref + (size_t)b * (size_t)vocab,
                        (size_t)vocab * sizeof *wgot))
                same++;
            else if (first < 0)
                first = b;
        }
        printf("\n== a draft window rewound to: the position its state was left at ==\n");
        printf("  %d rows in one walk, one step from each of the %d boundaries: %d/%d byte for"
               " byte with the step it replaces, %d/%d of those rows still readable, the"
               " prompt they start from landed %s\n",
               win, win - 1, same, win - 1, read_ok, win - 1,
               landed ? "in one place both times" : "in two places");
        if (rc) {
            printf("    FAIL window: the pass refused the window or a step out of it\n");
            fails++;
        } else if (!landed) {
            printf("    FAIL window: one prompt walked twice is two states, so the %d rows"
                   " below compare to nothing\n",
                   win);
            fails++;
        } else if (same != win - 1) {
            printf("    FAIL window: the step out of boundary %d is not the step it replaces,"
                   " so the sequence went on reading the cells of rows it abandoned\n",
                   first + 1);
            fails++;
        } else if (read_ok != win - 1) {
            printf("    FAIL window: %d of %d rows a rewind leaves readable are not the rows"
                   " the window walked\n",
                   read_ok, win - 1);
            fails++;
        } else {
            printf("    PASS window: a rewind takes the position back with the state\n");
        }
        qe_ckpt_free(ck);
        free(wref);
        fflush(stdout);
    }

    /* The batched pass wants its own copy of every byte of state, including the
     * requantised dense copies the chunk walks read. Closing this pass is what keeps the
     * peak down to one pass's worth of those. */
    qe_forward_close(f1);

    /* ---- 2. the batched pass --------------------------------------------------------- */
    qe_forward_opts bo = { .n_ctx = n_ctx, .n_threads = threads, .n_seq = bmax };
    qe_forward *fb = qe_forward_open(m, &bo, ferr, sizeof ferr);
    if (!fb) {
        fprintf(stderr, "bench_batch: n_seq %d refused: %s\n", bmax, ferr);
        return 1;
    }
    if (qe_forward_n_seq(fb) != bmax) {
        fprintf(stderr, "bench_batch: pass opened with n_seq %d, asked for %d\n",
                qe_forward_n_seq(fb), bmax);
        return 1;
    }
    qe_forward_state_bytes(fb, &stb, NULL);
    printf("\n== batched pass: n_seq %d, %.2f GiB of state (%.0f MiB more than one sequence, "
           "%d cells each) ==\n",
           qe_forward_n_seq(fb), (double)stb / 1073741824.0,
           (double)(stb - st1) / 1048576.0, n_ctx / bmax);
    fflush(stdout);

    int *batch = malloc((size_t)bmax * sizeof *batch);
    float *lg = malloc((size_t)bmax * (size_t)vocab * sizeof *lg);
    float *one = malloc((size_t)vocab * sizeof *one);
    if (!batch || !lg || !one) {
        fprintf(stderr, "bench_batch: out of memory for the batch (%ld MiB)\n",
                (long)bmax * vocab * sizeof *lg / 1048576);
        return 1;
    }

    for (int a = 0; a < NARM; a++) {
        printf("\n== %s: %s ==\n", kArm[a].tag, kArm[a].what);
        printf("  B   prefill  identical   greedy    worst dl    nats/tok   ms/step   "
               "ms/token   tok/s   vs single\n");
        fflush(stdout);
        pins_apply(fb, &base, a);
        for (int w = 0; w < nwant; w++) {
            const int B = want[w];
            result r = { .bad_s = -1, .bad_k = -1 };
            qe_forward_reset(fb);
            for (int s = 0; s < B && !r.bad; s++) {
                fill_stream(prompt, plen, corpus, nb, (long)s * stride);
                if (qe_forward_prefill_seq(fb, s, prompt, plen, one)) {
                    r.bad = 1;
                    break;
                }
                r.pl_rows++;
                if (memcmp(one, recs[a][s]->lg, (size_t)vocab * sizeof *one) == 0)
                    r.pl_same++;
            }
            if (r.bad) {
                fprintf(stderr, "  %s B=%d: prefill_seq failed\n", kArm[a].tag, B);
                fails++;
                continue;
            }
            for (int s = 0; s < B; s++)
                batch[s] = recs[a][s]->tok[0];
            for (int k = 0; k < steps && !r.bad; k++) {
                const double t0 = now();
                const int rc = qe_forward_step_batch(fb, batch, B, lg);
                const double dt = now() - t0;
                if (rc) {
                    fprintf(stderr, "  %s B=%d: step_batch failed at row %d (%d)\n",
                            kArm[a].tag, B, k, rc);
                    r.bad = 1;
                    break;
                }
                if (k >= warm)
                    r.seconds += dt;
                for (int s = 0; s < B; s++) {
                    const float *got = lg + (size_t)s * (size_t)vocab;
                    const float *exp = recs[a][s]->lg + (size_t)(k + 1) * (size_t)vocab;
                    const long chosen = recs[a][s]->tok[k + 1];
                    r.rows++;
                    if (memcmp(got, exp, (size_t)vocab * sizeof *exp) == 0)
                        r.same++;
                    else if (r.bad_s < 0) {
                        r.bad_s = s;
                        r.bad_k = k;
                    }
                    for (long i = 0; i < vocab; i++) {
                        const double d = fabs((double)got[i] - (double)exp[i]);
                        if (d > r.maxdl)
                            r.maxdl = d;
                    }
                    if (argmax(got, vocab) == argmax(exp, vocab))
                        r.hits++;
                    r.nats_got += logprob(got, vocab, chosen);
                    r.nats_ref += logprob(exp, vocab, chosen);
                    batch[s] = (int)chosen;
                }
            }
            if (r.bad) {
                fails++;
                continue;
            }
            const double ms_step = 1000.0 * r.seconds / steps;
            const double ms_tok = ms_step / B;
            const double rate = 1000.0 / ms_tok;
            printf("  %d   %ld/%ld     %ld/%ld   %ld/%ld   %8.4f   %+.6f   %7.2f   "
                   "%8.2f   %6.2f   %5.2fx\n",
                   B, r.pl_same, r.pl_rows, r.same, r.rows, r.hits, r.rows, r.maxdl,
                   (r.nats_got - r.nats_ref) / (double)r.rows, ms_step, ms_tok, rate,
                   single_ms[a] / ms_tok);
            /* A batch of one is a chunk of one token of a sequence whose slice starts at
             * the same cell zero a lone pass starts at, reading the same copies. If that
             * is not the recording bit for bit, the control is broken and nothing below it
             * in the table is a measurement of batching. */
            const int control = (B == 1) && (r.same != r.rows || r.pl_same != r.pl_rows);
            /* On the arm with no width-dependent walk at all, a row of a batch has to be
             * the sequence's own row to the bit, at every width, or a batch is a second
             * model and not a scheduling change. */
            const int strict =
                (a == ARM_EXACT) && (r.same != r.rows || r.pl_same != r.pl_rows);
            if (control || strict) {
                printf("      first differing row: sequence %d, step %d\n", r.bad_s, r.bad_k);
                printf("    FAIL %s B=%d: %s\n", kArm[a].tag, B,
                       B == 1 ? "a batch of one is not the single-stream recording"
                              : "rows are not byte-identical, so a batch is not only a "
                                "scheduling change");
                fails++;
                continue;
            }
            printf("    %s %s B=%d: %.1f%% of rows byte-identical, greedy says the same "
                   "token on %.1f%% of them, worst move %.1f%% of the head's largest logit, "
                   "%.2fx of a single stream per token\n",
                   B == 1 ? "PASS" : "note", kArm[a].tag, B,
                   100.0 * (double)r.same / (double)r.rows,
                   100.0 * (double)r.hits / (double)r.rows,
                   head_scale > 0.0 ? 100.0 * r.maxdl / head_scale : 0.0, single_ms[a] / ms_tok);
            fflush(stdout);
        }
    }
    /* ---- 3. the row map a lane actually drives ---------------------------------------- */
    /* Leg 1 asked whether a row is the sequence's own row; it asked it of a step whose rows
     * were the sequences 0..B-1 in order, which is the only map leg 2 drives and not the
     * map a server drives. A lane has a prompt fed three tokens into somebody else's
     * conversation, a row map that renumbers itself every time a sibling answers its last
     * token, and a sequence handed to the next request two steps after `reset_seq` took
     * the last one's state out of it. The promise all of that rests on is that a row is its
     * sequence's row whatever else happens to be in the step, in whatever order, however
     * long ago the pass was opened - and the only arm that promise is testable on is the
     * one with no width-dependent walk, where "whatever" means to the bit.
     *
     * So: sequences join one at a time, half of them leave and are recycled back into their
     * own slot mid-run, and the map rotates every step so that it is rarely the identity.
     * Every row of every step is memcmp'd against that sequence's own recording, and so is
     * the logits row of a prompt fed while four other sequences were already generating. */
    if (bmax > 1 && steps >= 6) {
        pins_apply(fb, &base, ARM_EXACT);
        qe_forward_reset(fb);
        int live[QE_MAX_SEQ] = { 0 }, ns[QE_MAX_SEQ] = { 0 };
        int join_at[QE_MAX_SEQ], out_at[QE_MAX_SEQ];
        for (int s = 0; s < bmax; s++) {
            join_at[s] = (int)((long)s * steps / (2 * bmax));
            out_at[s] = s < bmax / 2 ? join_at[s] + steps / 2 : steps + 1;
        }
        long rows = 0, same = 0, pfr = 0, pfs = 0;
        int bad_s = -1, bad_k = -1, bad_row = -1, bad = 0;
        int width_seen[QE_MAX_SEQ + 1] = { 0 };
        for (int k = 0; k < steps && !bad; k++) {
            for (int s = 0; s < bmax; s++)
                if (live[s] && k == out_at[s]) {
                    qe_forward_reset_seq(fb, s);
                    live[s] = 0;
                }
            for (int s = 0; s < bmax; s++) {
                if (live[s] || (k != join_at[s] && k != out_at[s] + 2))
                    continue;
                fill_stream(prompt, plen, corpus, nb, (long)s * stride);
                if (qe_forward_prefill_seq(fb, s, prompt, plen, one)) {
                    fprintf(stderr, "  ragged: the prompt of sequence %d failed at step %d\n",
                            s, k);
                    bad = 1;
                    break;
                }
                pfr++;
                if (memcmp(one, recs[ARM_EXACT][s]->lg, (size_t)vocab * sizeof *one) == 0)
                    pfs++;
                else if (bad_s < 0)
                    bad_s = s, bad_k = k, bad_row = -1;
                live[s] = 1;
                ns[s] = 0;
            }
            if (bad)
                break;
            int mp[QE_MAX_SEQ], rmap[QE_MAX_SEQ], feed[QE_MAX_SEQ], nlive = 0;
            for (int s = 0; s < bmax; s++)
                if (live[s])
                    mp[nlive++] = s;
            if (!nlive)
                continue;
            for (int t = 0; t < nlive; t++) {
                const int s = mp[(t + k) % nlive];
                rmap[t] = s;
                feed[t] = recs[ARM_EXACT][s]->tok[ns[s]];
            }
            width_seen[nlive]++;
            if (qe_forward_step_batch_rows(fb, feed, rmap, nlive, lg)) {
                fprintf(stderr, "  ragged: the step of %d rows failed at step %d\n", nlive, k);
                bad = 1;
                break;
            }
            for (int t = 0; t < nlive; t++) {
                const int s = rmap[t];
                const float *got = lg + (size_t)t * (size_t)vocab;
                const float *exp =
                    recs[ARM_EXACT][s]->lg + (size_t)(ns[s] + 1) * (size_t)vocab;
                rows++;
                if (memcmp(got, exp, (size_t)vocab * sizeof *exp) == 0)
                    same++;
                else if (bad_s < 0)
                    bad_s = s, bad_k = k, bad_row = t;
                ns[s]++;
            }
        }
        long widths = 0;
        int wmin = QE_MAX_SEQ, wmax = 0;
        for (int w = 1; w <= QE_MAX_SEQ; w++)
            if (width_seen[w]) {
                widths++;
                if (w < wmin)
                    wmin = w;
                if (w > wmax)
                    wmax = w;
            }
        printf("\n== ragged row map: sequences joining, leaving and being recycled mid-pass ==\n");
        printf("  %ld/%ld prompts fed mid-pass byte-identical, %ld/%ld rows of %d steps "
               "byte-identical, steps of %d to %d rows (%ld distinct widths)\n",
               pfs, pfr, same, rows, steps, wmin, wmax, widths);
        if (bad) {
            printf("    FAIL ragged: the pass refused a step or a prompt\n");
            fails++;
        } else if (same != rows || pfs != pfr) {
            printf("      first differing row: sequence %d, step %d, row %d (%s)\n", bad_s, bad_k,
                   bad_row, bad_row < 0 ? "a prompt fed mid-pass" : "a row of a partial map");
            printf("    FAIL ragged: a row of a partial map is not its own sequence's row, so a "
                   "lane can be answered out of a sibling's logits\n");
            fails++;
        } else if (widths < 2) {
            printf("    FAIL ragged: the schedule never varied the map, so nothing was tested\n");
            fails++;
        } else {
            printf("    PASS ragged: a row is its sequence's row whatever else is in the step\n");
        }
        fflush(stdout);
    } else if (bmax > 1) {
        printf("\n== ragged row map: skipped, %d steps is too few to join and leave in ==\n",
               steps);
    }
    if (fails)
        fprintf(stderr, "bench_batch: %d failure(s)\n", fails);

    free(one);
    free(lg);
    free(batch);
    qe_forward_close(fb);
    for (int a = 0; a < NARM; a++)
        for (int s = 0; s < bmax; s++)
            rec_free(recs[a][s]);
    free(prompt);
    qe_tok_close(tok);
    qe_close(m);
    qe_pool_shutdown();
    printf("\n== done: %d failure(s) ==\n", fails);
    fflush(stdout);
    return fails ? 1 : 0;
}
