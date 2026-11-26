/* The engine hook: what the HTTP layer calls "the model".

   Two implementations live here, and which one is active is not a secret from
   the client - /v1/models reports it, along with a sentence saying what is
   actually running:

     - checkpoint engine: opens the GGUF (the whole model resident at launch; 87
       GiB on a spinning disk is not something to read twice by accident) and
       runs it. src/forward.c executes the qwen4exp layer graph -
       gated-delta-net state, hyper-connections, the sparse-attention indexer,
       routed experts and the n-gram prediction table - one token per step, and
       src/sampler.c turns the logits into a token id. If a tensor the
       architecture names is missing, the engine fails to start and says which
       one, rather than listening behind a half-built model.

     - selftest engine: a real generator over a synthetic target, driven by the
       same qe_spec_generate loop the checkpoint engine uses. It exists so the
       serving path (JSON in, SSE out, cancellation, usage accounting) can be
       exercised without 87 GiB of weights, and it is what answers when no
       checkpoint was given.

   Both hide behind one function, so the transport never learns which one it is
   talking to. */
#define _GNU_SOURCE
#include "qe.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#include "tui.h"
#include "turns.h"

/* A turn boundary the engine can put the sequence back onto. It is not a copy of a
 * conversation: it is the state a prompt left behind, plus the ids that produce it,
 * so matching one against a new prompt is a token comparison and restoring one is
 * a few memcpy calls. See qe_state in qe.h for what the state consists of. */
typedef struct {
    qe_state *st;
    float *tail;   /* the logits after the last of those tokens */
    long n;        /* tokens covered, and the length `tail` follows */
    int tail_valid;
    int answer;    /* covers an answer as well as its prompt, so only an exact
                    * replay of that answer can match it: see `turns_store` */
    uint64_t used; /* eviction order, not a timestamp */
} boundary;

/* A guard on an absurd budget, not a limit anyone reasons with. The budget is what
 * decides how many points exist, and a slot costs nothing until a turn actually
 * fills it, so all this protects is the one eager allocation below, which is an
 * array of descriptors rather than the states. At this checkpoint's 113 MiB a point
 * that is 57 GiB of rewind points, which is a typo and not a configuration. */
#define TURN_SLOTS_MAX 512

/* The speculative decoder: a draft window, the state after each of its rows, and
 * the request whose tokens those rows are deciding. Defined with the loop that
 * runs it, below the forward pass's own section. */
typedef struct spec_target spec_target;

static void spec_drop(spec_target **sp);

/* A request waiting for the batched lane, on its own thread. The driver writes the
 * three fields below and sets `done` last, all of them under the engine's lock, which
 * is the lock the waiter is asleep on - so a waiter that wakes up cannot be looking at
 * a half-written result, and the driver never touches this again after `done`. */
typedef struct {
    int done;
    long rc;
    int err;
} lane_wait;

/* One request on the batched lane: one sequence of the forward pass, one sampler, one
 * client. The sequence is the lane's index in the engine's array, which is why nothing
 * here names it and why a lane can finish without any state moving: the row map handed
 * to qe_forward_step_batch_rows is the lane table read as it stands.
 *
 * Every field belongs to the engine's lock. The prompt ids are the exception and only
 * in the sense that matters - written by the requester before the lane is armed and
 * read only by the driver afterwards, so a request queued behind a full batch is never
 * holding a buffer the driver has to wait for. */
typedef struct {
    int used;        /* a request owns this sequence */
    int admitted;    /* its prompt is through the network; it takes part in steps */
    int have_logits; /* the row at `row` answers its next token and is unemitted */
    int stop;        /* nothing more to generate: eos, or a client that went away */
    int err;         /* errno for the caller, 0 = no failure */
    int row;         /* whose row of the engine's row buffer its logits are in */
    int *ids;
    long n_ids;
    long fed;     /* prompt ids fed so far, up to n_ids */
    long mt;      /* answer budget, already trimmed to what the window leaves */
    long emitted;
    int pending;  /* the token sampled and waiting for its step */
    int eos;
    int dropped;  /* the prompt had to lose its front to fit the window */
    qe_sampler_opts so;
    uint64_t rng;
    qe_token_cb cb;
    void *user;
    qe_gen_stats *st;
    qe_progress_cb pcb;  /* this request's ingestion hook, and nobody else's */
    void *pcb_user;
    lane_wait *w;
    double t0;
} lane;

struct qe_engine {
    qe_engine_info info;
    qe_engine_opts o;
    qe_model *mm;     /* checkpoint metadata, mapped read-only */
    qe_model *mmproj; /* vision projector, metadata only */
    qe_model *mtp;    /* draft head, in its own GGUF, resident like the weights */
    qe_tok *tok;       /* real BPE tokenizer, NULL if the file's is unusable */
    char tok_err[256]; /* why tok is NULL, for --report and /v1/models */
    qe_forward *fw;    /* the qwen4exp forward pass, NULL until it opens */
    qe_sampler_opts samp; /* the file's own sampler defaults */
    /* The transcript the forward pass is holding: every token it was fed for the
     * conversation in hand, in order, and the logits that follow the last of
     * them. A request whose prompt is this transcript plus new tokens continues
     * it instead of starting a transcript, which is the difference between
     * re-reading a conversation and reading what was added to it.
     *
     * Only extension is reused. A prompt that agrees for a while and then
     * diverges is a different conversation as far as these bytes are concerned,
     * and the answer to that is to feed it again, not to decide that a prefix
     * that is wrong from halfway through is probably close enough. */
    int *fed;
    long fed_n;
    int fed_valid; /* the transcript above describes the whole sequence, from 0 */
    float *tail;   /* logits after the last fed token, room for one vocabulary */
    int tail_valid;
    /* Built the first time a greedy request asks for a draft window, and dropped
     * with the engine: one per engine is enough because one per engine is how many
     * sequences the forward pass holds. */
    spec_target *spec;
    int prefix_cache; /* 0 only for the measurement that proves reuse is identical */
    /* Turn boundaries kept against the possibility of being asked to go back to one.
     * A client that re-renders the transcript - which is every client that sends a
     * conversation on every turn - does not extend the sequence the engine is
     * holding, so extension alone means re-reading the whole thing. These are the
     * positions it can come back to instead. Guarded by the same lock as the
     * transcript, because that is the lock that serialises generations. */
    boundary *turns;
    qe_turn_slot *turn_views; /* the same slots as src/turns.c sees them */
    int n_turns;
    size_t turn_slot_bytes;
    int turn_budget_mib; /* what it was given, whoever gave it */
    int turn_budget_auto; /* 1 = the engine picked that from the free RAM */
    uint64_t turn_clock;
    long long turn_saved; /* prompt tokens never read a second time */
    /* The ingestion hook the caller installed, reached through the engine's own, so
     * that the counters below and the caller's keepalive both see every chunk. */
    qe_progress_cb user_progress;
    void *user_progress_user;
    long ingest_base;
    long ingest_total;
    /* The batched lane, which exists exactly when n_batch > 1 and is the only thing
     * that ever runs the forward pass when it does. `lock` guards these and every lane,
     * as it guards the transcript; the thread driving the batch is the only one that
     * touches the pass while a lane is live, and it holds no lock while walking.
     *
     * `rows` is the batch's logits: n_batch + 1 vocabulary-wide rows, one per lane plus
     * one more that a prompt being ingested answers out of. A step writes row t, and a
     * lane emits from the row it was given, so nothing is copied per token; the extra
     * row is safe to share because a lane is emitted in the iteration after it is
     * admitted and before any other prompt is fed. */
    int n_batch;
    long lane_ctx; /* tokens one sequence may use: the cache divided by n_batch */
    lane *lanes;
    float *rows;
    int lane_driving;
    int lane_live;
    pthread_cond_t lane_cv;
    pthread_mutex_t lock;
};

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

/* The quantization mix by bytes, biggest first, for --report and /v1/models. A
 * single "quant" field on a checkpoint that mixes IQ3_S, IQ4_NL, Q8_0 and f32
 * is a lie unless it says which one dominates and how much of the total each
 * one is. The mix is also the whole reason the kernel bench covers five types:
 * 512 experts at 10 active means these bytes are read by every token. */
static void quant_mix(qe_model *m, char *out, size_t n) {
    double bytes[QE_TYPE_COUNT] = { 0 };
    double total = 0;
    out[0] = 0;
    for (size_t i = 0; i < qe_tensor_count(m); i++) {
        const qe_tensor *t = qe_tensor_by_index(m, i);
        if (t->type >= 0 && t->type < QE_TYPE_COUNT)
            bytes[t->type] += (double)t->bytes;
    }
    for (int i = 0; i < QE_TYPE_COUNT; i++)
        total += bytes[i];
    if (total <= 0) {
        snprintf(out, n, "unknown");
        return;
    }
    size_t used = (size_t)snprintf(out, n, "%.1f GiB: ", total / 1073741824.0);
    if (used >= n) return;
    int first = 1;
    for (;;) {
        int best = -1;
        for (int i = 0; i < QE_TYPE_COUNT; i++)
            if (bytes[i] > 0 && (best < 0 || bytes[i] > bytes[best])) best = i;
        if (best < 0) break;
        double pct = 100.0 * bytes[best] / total;
        const qe_type_info *ti = qe_type_info_of((qe_type)best);
        if (pct >= 0.5) {
            int k = snprintf(out + used, n - used, "%s%s %.0f%%", first ? "" : ", ",
                             ti ? ti->name : "?", pct);
            if (k < 0 || (size_t)k >= n - used) break;
            first = 0;
            used += (size_t)k;
        }
        bytes[best] = 0;
    }
    if (!used) snprintf(out, n, "unknown");
}

/* ------------------------------------------------------------- turn boundaries
 *
 * The budget is in MiB and a boundary is a fixed size for a given forward pass, so
 * the number of them is a division and the policy that picks a victim is least
 * recently used. Nothing is allocated until the first boundary is actually worth
 * keeping, which is what keeps a server that is only ever asked one question from
 * holding 450 MiB of copies of a state it will not go back to. */

/* What an undecided budget buys. A boundary is the state of a sequence, so this is
 * a purchase of RAM for a conversation, and the honest number depends on what the
 * weights already took: a fifth of what is still free when the checkpoint is
 * resident, and never more than a cache large enough to be mistaken for a leak.
 *
 * The floor is a number of points rather than a number of MiB, which is the whole
 * reason it is written this way. A point is 112.6 MiB plus a vocabulary of logits
 * at this checkpoint, so a floor expressed in MiB silently buys four points at most
 * hosts - two turns of history - and a fifth turn then re-reads the conversation
 * while the banner says the cache is on. Twelve points is six turns, which is what
 * a client that re-renders its own transcript needs before it starts evicting its
 * own history, and it is a floor only while the host can actually spare it: half of
 * MemAvailable is the hard cap, and a host that cannot afford one point gets no
 * cache and a line saying so rather than an allocation that pushes it into reclaim. */
#define TURN_CACHE_SHARE 0.20
#define TURN_CACHE_POINTS 12 /* six turns: prompt end and answer end of each */
#define TURN_CACHE_SPARE_DIV 2
#define TURN_CACHE_MIB_MAX 8192

/* What the kernel says is left to spend, in MiB, or 0 when it will not say. Asked
 * after the weights are placed, which is the only order in which the answer means
 * anything for a purchase this size. */
static long mem_available_mib(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return 0;
    char line[192];
    long kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1)
            break;
    fclose(f);
    return kb / 1024;
}

static void turns_init(qe_engine *e) {
    if (!e->fw)
        return;
    const size_t slot = qe_state_size(e->fw) + (size_t)e->info.n_vocab * sizeof(float);
    if (!slot)
        return;
    e->info.turn_slot_bytes = (long)slot;
    e->info.turn_cache_points = TURN_CACHE_POINTS;
    /* Rounded up, because a budget that is a hair short of a point buys one fewer
     * than the number printed next to it. */
    const long point = qe_turns_point_mib((long)slot);
    long mib = e->o.turn_cache_mib;
    if (!mib) {
        mib = qe_turns_auto_mib(mem_available_mib(), point, TURN_CACHE_POINTS,
                                TURN_CACHE_SHARE, TURN_CACHE_SPARE_DIV, TURN_CACHE_MIB_MAX);
        e->turn_budget_auto = 1;
        e->info.turn_cache_auto = 1;
    }
    if (mib <= 0) {
        if (mib < 0)
            snprintf(e->info.turn_cache_err, sizeof e->info.turn_cache_err,
                     "off (--turn-cache %ld: every turn reads its whole prompt)", mib);
        return;
    }
    long slots = (long)(((double)mib * 1048576.0) / (double)slot);
    if (slots > TURN_SLOTS_MAX)
        slots = TURN_SLOTS_MAX;
    if (slots < 1) {
        /* Said rather than swallowed: without this line the only evidence that the
         * cache is absent is a banner saying "off", and the reason - the weights
         * took what was free - is the one thing the operator can act on. */
        if (e->turn_budget_auto)
            snprintf(e->info.turn_cache_err, sizeof e->info.turn_cache_err,
                     "afforded nothing: %.1f MiB is a rewind point here, and less than "
                     "one was free after the weights (--turn-cache to insist)",
                     (double)slot / 1048576.0);
        else
            snprintf(e->info.turn_cache_err, sizeof e->info.turn_cache_err,
                     "off: %ld MiB is less than one rewind point at %.1f MiB each", mib,
                     (double)slot / 1048576.0);
        return;
    }
    e->turns = calloc((size_t)slots, sizeof *e->turns);
    e->turn_views = calloc((size_t)slots, sizeof *e->turn_views);
    if (!e->turns || !e->turn_views) {
        free(e->turns);
        free(e->turn_views);
        e->turns = NULL;
        e->turn_views = NULL;
        return;
    }
    e->n_turns = (int)slots;
    e->turn_slot_bytes = slot;
    e->turn_budget_mib = (int)mib;
    e->info.turn_cache_mib = (int)mib;
}

static void turns_discard(qe_engine *e) {
    for (int i = 0; i < e->n_turns; i++) {
        if (e->turns[i].st)
            qe_state_free(e->turns[i].st);
        free(e->turns[i].tail);
    }
    if (e->turns)
        memset(e->turns, 0, (size_t)e->n_turns * sizeof *e->turns);
    if (e->turn_views)
        memset(e->turn_views, 0, (size_t)e->n_turns * sizeof *e->turn_views);
}

static void turns_stats(qe_engine *e, long *held, long long *bytes) {
    long h = 0;
    for (int i = 0; i < e->n_turns; i++)
        if (e->turns[i].st)
            h++;
    if (held)
        *held = h;
    if (bytes)
        *bytes = (long long)h * (long long)e->turn_slot_bytes;
}

/* The held points as src/turns.c sees them. Refilled from the states before every
 * decision rather than maintained beside them: the two representations then agree by
 * construction, and a mirror that is only right for as long as every writer remembers
 * it is a bug waiting for the next writer. The cost is one pass over a few hundred
 * descriptors, against a decision that is taken once a request. */
static void turns_views(qe_engine *e) {
    for (int i = 0; i < e->n_turns; i++) {
        const boundary *b = &e->turns[i];
        e->turn_views[i].pos = b->st ? qe_state_pos(b->st) : 0;
        e->turn_views[i].ids = b->st ? qe_state_ids(b->st) : NULL;
        e->turn_views[i].have_logits = b->tail_valid;
        e->turn_views[i].answer = b->answer;
        e->turn_views[i].used = b->used;
    }
}

/* The longest boundary that is this prompt from its first token, and therefore the
 * most that does not have to go through the network again. A boundary matches only
 * token for token over its whole length: a sequence is defined by its beginning, and
 * "close from halfway through" is a different conversation. `agreed` is the other
 * half of the answer, and the one a client actually needs on a miss: how far the
 * prompt and the best boundary on hand did agree before they disagreed, which is a
 * position in the conversation rather than the fact that one was lost. Callers that
 * match also take the slot out of the eviction queue's future. */
static int turns_match(qe_engine *e, const int *ids, long np, long *at, long *agreed) {
    if (!e->turn_views) {
        if (at)
            *at = 0;
        if (agreed)
            *agreed = 0;
        return -1;
    }
    turns_views(e);
    const int best = qe_turns_match(e->turn_views, e->n_turns, ids, np, at, agreed,
                                    &e->turn_clock);
    if (best >= 0)
        e->turns[best].used = e->turn_clock;
    return best;
}

/* Which held point a store may take a slot from, or -1 when the cache is full of
 * things more worth keeping. The order is the argument.
 *
 * A point whose logits are gone cannot match anything, so it goes first. Then a slot
 * that has never been written. Then the answer-end of a turn, least recently matched
 * first, because only a client replaying that answer token for token can come back to
 * it, whereas the prompt-end of a turn is what every client that re-renders the
 * transcript does come back to. So an answer-end point never displaces a prompt-end
 * one: a full cache then holds one point per turn rather than two, and the point it
 * keeps is the useful one. Within that, least recently matched first, which is what
 * keeps the newest turns rewound to and the oldest ones re-read. */
static int turns_victim(qe_engine *e, int answer) {
    if (!e->turn_views)
        return -1;
    turns_views(e);
    return qe_turns_victim(e->turn_views, e->n_turns, answer);
}

/* Which slot a boundary may take: the one already holding this position, or the one
 * the eviction policy says is worth replacing. A boundary with the same length as
 * one already held replaces it: it is the same position in the conversation, and the
 * newer logits are the ones a repeat of this prompt needs. */
static int turns_slot(qe_engine *e, long n, int answer) {
    for (int i = 0; i < e->n_turns; i++)
        if (e->turns[i].st && qe_state_pos(e->turns[i].st) == n)
            return i;
    return turns_victim(e, answer);
}

/* A slot with room for a boundary of this shape, or NULL. */
static boundary *turns_room(qe_engine *e, long n, long vocab, int answer) {
    const int victim = turns_slot(e, n, answer);
    if (victim < 0)
        return NULL;
    boundary *r = &e->turns[victim];
    if (!r->st)
        r->st = qe_state_new(e->fw);
    if (!r->st)
        return NULL;
    if (!r->tail && vocab > 0)
        r->tail = malloc((size_t)vocab * sizeof *r->tail);
    return r;
}

/* What a slot remembers once its state is in it: the position it is the answer to,
 * the logits a request ending exactly here needs so it never takes a pass, and the
 * eviction queue's stamp. */
static void turns_commit(qe_engine *e, boundary *r, long n, const float *logits, long vocab,
                         int answer) {
    r->n = n;
    r->answer = answer;
    r->used = ++e->turn_clock;
    if (r->tail && logits) {
        memcpy(r->tail, logits, (size_t)vocab * sizeof *r->tail);
        r->tail_valid = 1;
    } else {
        r->tail_valid = 0;
    }
}

static long turns_store(qe_engine *e, long n, const float *logits, long vocab, int answer) {
    if (!e->n_turns || !e->fw || n < 1 || qe_forward_pos(e->fw) != n)
        return 0;
    boundary *r = turns_room(e, n, vocab, answer);
    if (!r)
        return 0;
    double t0 = now();
    if (qe_forward_snapshot(e->fw, r->st))
        return 0;
    const long us = (long)((now() - t0) * 1e6);
    turns_commit(e, r, n, logits, vocab, answer);
    return us;
}

/* The same, for a position the pass has already walked past. A draft window starts
 * by feeding the token the last window earned, so the state after the last token of
 * the prompt is boundary 1 of its first window - kept, and no longer in the pass by
 * the time the request ends and anyone knows whether it was worth keeping. The
 * bytes go straight into the slot, so the one copy a window already made is the
 * only one this costs. */
static long turns_store_boundary(qe_engine *e, long n, const qe_ckpt *c, int boundary_no,
                                 const float *logits, long vocab, int answer) {
    if (!e->n_turns || !e->fw || n < 1 || !c)
        return 0;
    boundary *r = turns_room(e, n, vocab, answer);
    if (!r)
        return 0;
    double t0 = now();
    if (qe_ckpt_state(e->fw, c, boundary_no, r->st) || qe_state_pos(r->st) != n)
        return 0;
    const long us = (long)((now() - t0) * 1e6);
    turns_commit(e, r, n, logits, vocab, answer);
    return us;
}

/* The engine's own ingestion hook, installed once per forward pass and left there:
 * the counters the terminal interface reads and whatever a connection wants to know
 * are both per chunk, and only one of them can be the hook. */
static void fwd_progress_fine(void *vp, long done, long total) {
    qe_engine *e = vp;
    qe_ui_ingest(e->ingest_base + done, e->ingest_total > 0 ? e->ingest_total : total);
}

static int fwd_progress(void *vp, long done, long total) {
    qe_engine *e = vp;
    qe_ui_ingest(e->ingest_base + done, e->ingest_total > 0 ? e->ingest_total : total);
    if (e->user_progress)
        return e->user_progress(e->user_progress_user, done, total);
    return 0;
}

/* Everything an engine does with a draft head that arrived as its own file, which is
 * how these checkpoints are actually delivered: the nextn block is one more decoder
 * layer plus the norms around it, quantised, alongside a copy of the embeddings and
 * the output row it reads and writes.
 *
 * Same rule as every other weight byte here - resident before the listener opens,
 * STRICT, no laziness - with one exception: a metadata-only open stays metadata-only,
 * because `--report` is worth running and a sidecar that turned it into a 2.6 GiB read
 * would stop being run.
 *
 * `validate` says whether there is a target architecture to check the head against. A
 * checkpoint engine refuses a head for another model at startup, because a head from
 * a different model drafts confidently and is rejected on every token, which measures
 * as speculation being slow rather than as the wrong file being named on a command
 * line. The synthetic target has no architecture to refuse one for: it stages the
 * file, reports where it came from, and generates nothing from it.
 *
 * Returns 0 when the head is staged, -1 when the engine must not start. */
static int stage_mtp(qe_engine *e, int validate, char *err, size_t errn) {
    const char *path = e->o.mtp_path;
    unsigned mstage = !strcmp(e->info.stage, "mmap") ? QE_STAGE_MMAP | QE_STAGE_DESCRIBE
                                                     : QE_STAGE_RAM | QE_STAGE_STRICT;
    e->mtp = qe_open(path, mstage);
    if (!e->mtp) {
        if (err && errn)
            snprintf(err, errn, "draft head %s: %s", path, qe_strerror());
        return -1;
    }
    /* Read the head's own metadata under the architecture the head itself names.
     * The synthetic target has no architecture of its own to look keys up under,
     * and a head always carries the one it was written for. */
    char farch[64] = { 0 };
    const char *fv = NULL;
    if (qe_model_get_str(e->mtp, "general.architecture", &fv) == 0 && fv && *fv)
        snprintf(farch, sizeof farch, "%s", fv);
    if (!farch[0])
        snprintf(farch, sizeof farch, "%s", e->info.arch);
    char key[96];
    long long v;
    int memb = 0;
    snprintf(key, sizeof key, "%s.embedding_length", farch);
    if (qe_model_get_int(e->mtp, key, &v) == 0)
        memb = (int)v;
    if (validate && (strcmp(farch, e->info.arch) ||
                     (memb && e->info.n_embd && memb != e->info.n_embd))) {
        if (err && errn)
            snprintf(err, errn,
                     "draft head %s is a %s head %d wide; this engine is %s %d wide. A head"
                     " for another model drafts tokens the target rejects every time, so it"
                     " is refused at startup rather than measured as a low accept rate",
                     path, farch[0] ? farch : "(unreadable architecture)", memb, e->info.arch,
                     e->info.n_embd);
        qe_close(e->mtp);
        e->mtp = NULL;
        return -1;
    }
    e->info.has_mtp_head = 1;
    snprintf(e->info.mtp_path, sizeof e->info.mtp_path, "%s", path);
    snprintf(e->info.mtp_stage, sizeof e->info.mtp_stage, "%s", qe_model_stage_method(e->mtp));
    size_t mnt = qe_tensor_count(e->mtp);
    long long mbytes = 0;
    for (size_t i = 0; i < mnt; i++)
        mbytes += (long long)qe_tensor_by_index(e->mtp, i)->bytes;
    e->info.mtp_tensors = (long long)mnt;
    e->info.mtp_bytes = mbytes;
    quant_mix(e->mtp, e->info.mtp_quant, sizeof e->info.mtp_quant);
    snprintf(key, sizeof key, "%s.nextn_predict_layers", farch);
    if (qe_model_get_int(e->mtp, key, &v) == 0)
        e->info.mtp_layers = (int)v;
    snprintf(key, sizeof key, "%s.block_count", farch);
    /* "Beyond the checkpoint's" needs a checkpoint to be beyond: with no target
     * blocks there is no difference to report, and 49 would read as a head with
     * forty-nine layers rather than a file that says it has forty-nine. */
    if (e->info.n_blocks > 0 && qe_model_get_int(e->mtp, key, &v) == 0 &&
        v > e->info.n_blocks)
        e->info.mtp_head_blocks = (int)(v - e->info.n_blocks);
    return 0;
}

qe_engine *qe_engine_open(const qe_engine_opts *o, char *err, size_t errn) {
    if (err && errn)
        err[0] = 0;
    qe_engine *e = calloc(1, sizeof *e);
    if (!e) {
        if (err && errn)
            snprintf(err, errn, "out of memory");
        return NULL;
    }
    e->o = *o;
    e->prefix_cache = 1;
    snprintf(e->info.kv_cache, sizeof e->info.kv_cache, "none");
    pthread_mutex_init(&e->lock, NULL);
    /* A batched pass needs a slot per sequence, and the pass says how many it can
     * have. Refused rather than clamped: an operator who asked for sixteen and got
     * eight is reading a startup line that contradicts a command line. */
    if (o->n_batch > QE_MAX_SEQ) {
        if (err && errn)
            snprintf(err, errn,
                     "--batch %d is more sequences than a pass drives (%d): the row map of "
                     "one step is that wide, and every sequence is a share of the cache and"
                     " a copy of the recurrent state",
                     o->n_batch, QE_MAX_SEQ);
        pthread_mutex_destroy(&e->lock);
        free(e);
        return NULL;
    }
    e->n_batch = o->n_batch > 0 ? o->n_batch : 1;
    e->info.n_batch = e->n_batch;
    if (e->n_batch > 1 && (!o->model_path || !*o->model_path)) {
        /* The synthetic target has no forward pass and no sequence cache, so there is
         * nothing for eight requests to share, and a server that took the flag here would
         * answer with one queue while reporting a batch. */
        if (err && errn)
            snprintf(err, errn,
                     "--batch %d needs a checkpoint: the synthetic target has no pass and"
                     " no sequence cache to divide, so there is nothing for a second"
                     " request to share",
                     e->n_batch);
        pthread_mutex_destroy(&e->lock);
        free(e);
        return NULL;
    }
    if (e->n_batch > 1 && o->mtp_path && *o->mtp_path) {
        if (err && errn)
            snprintf(err, errn,
                     "--batch %d cannot draft: a draft window is one sequence's state and"
                     " the pass cannot walk nine rows of one conversation, so the head"
                     " would be resident bytes that nothing reads (--mtp without --batch)",
                     e->n_batch);
        pthread_mutex_destroy(&e->lock);
        free(e);
        return NULL;
    }
    if (e->n_batch > 1) {
        /* Speculation and the turn cache are both one sequence's business: a window is
         * verified against the state a row was drafted from, and a rewind point is one
         * sequence's whole state. Neither is a thing eight sequences can each have, so
         * the lane says so at startup instead of running a request without what it asked
         * for. */
        e->o.spec_k = 0;
        e->prefix_cache = 0;
    }
    pthread_cond_init(&e->lane_cv, NULL);
    /* No number here: what a draft is worth depends on which drafter this engine
     * ends up with, and that is not known until a head has been opened or refused.
     * The selftest engine has no head to open, so it is decided now; a checkpoint
     * engine is decided below, next to the drafter it really has. */
    if (e->o.spec_k == QE_SPEC_K_AUTO && (!o->model_path || !*o->model_path))
        e->o.spec_k = QE_SPEC_K_NGRAM;
    e->info.n_threads = e->o.n_threads;
    e->info.spec_k = e->o.spec_k;

    if (!o->model_path || !*o->model_path) {
        snprintf(e->info.id, sizeof e->info.id, "%s",
                 (o->id && *o->id) ? o->id : "qe-selftest");
        snprintf(e->info.arch, sizeof e->info.arch, "selftest");
        snprintf(e->info.quant, sizeof e->info.quant, "none (synthetic target)");
        snprintf(e->info.label, sizeof e->info.label, "qe selftest (not a model)");
        e->info.is_checkpoint = 0;
        e->info.can_generate = 1;
        e->info.has_mtp_head = 1; /* the synthetic drafter is real code */
        /* Nothing was placed anywhere, and an empty string here would make the
         * metadata look like a field that was forgotten rather than a fact. */
        snprintf(e->info.stage, sizeof e->info.stage, "none");
        snprintf(e->info.tokenizer, sizeof e->info.tokenizer,
                 "selftest word ids, not a real vocabulary; prompt token counts are estimated");
        snprintf(e->info.why, sizeof e->info.why,
                 "no checkpoint was loaded. Tokens come from a synthetic target "
                 "driven through the real speculation loop, which exercises the "
                 "HTTP path end to end. They are not model output.");
        /* A head named on the command line is still a fact about how this process was
         * started, and the panel reports it rather than dropping the flag on the
         * floor. There is no architecture here to validate it against and no nextn
         * block to run it through, so it is staged and reported, which is all a
         * sidecar is worth to an engine that is not a model. */
        if (o->mtp_path && *o->mtp_path && stage_mtp(e, 0, err, errn)) {
            pthread_mutex_destroy(&e->lock);
            free(e);
            return NULL;
        }
        if (e->n_batch > 1)
            strncat(e->info.why,
                    " --batch is a checkpoint thing: there is no forward pass here to"
                    " share one weight walk across, so it is reported and not run.",
                    sizeof e->info.why - strlen(e->info.why) - 1);
        return e;
    }

    /* A checkpoint engine holds the checkpoint: no part of serving a token may
     * reach a device. So the whole thing is placed in memory here, at launch,
     * and STRICT removes the quieter option - if the bytes cannot be resident,
     * the engine fails to start instead of listening behind a disk. An explicit
     * QE_STAGE_MMAP is a request to read metadata, not to serve, and is refused:
     * on this host it means one raidz2 seek per expert row. */
    unsigned stage = o->stage;
    if (QE_STAGE_MODE(stage) == QE_STAGE_MMAP && !(stage & QE_STAGE_DESCRIBE)) {
        if (err && errn)
            snprintf(err, errn,
                     "a checkpoint engine cannot serve from a demand-paged mapping: the"
                     " checkpoint has to be resident in RAM at startup (stage auto or ram)."
                     " Metadata-only readers say so with QE_STAGE_DESCRIBE");
        pthread_cond_destroy(&e->lane_cv);
        pthread_mutex_destroy(&e->lock);
        free(e);
        return NULL;
    }
    if (!(stage & QE_STAGE_DESCRIBE))
        stage |= QE_STAGE_STRICT;
    e->mm = qe_open(o->model_path, stage);
    if (!e->mm) {
        if (err && errn)
            snprintf(err, errn, "%s", qe_strerror());
        pthread_cond_destroy(&e->lane_cv);
        pthread_mutex_destroy(&e->lock);
        free(e);
        return NULL;
    }
    e->info.is_checkpoint = 1;
    e->info.can_generate = 0;
    snprintf(e->info.id, sizeof e->info.id, "%s", (o->id && *o->id) ? o->id : o->model_path);

    /* The tokenizer is loaded even though no forward pass exists yet, because
     * the two things that need it today need no weights: usage.prompt_tokens
     * has to be a real count of real tokens, and the routing-overlap question
     * (how many distinct experts T consecutive tokens touch) can only be asked
     * about real token ids. Non-fatal: a file with an unreadable tokenizer
     * still serves, it just reports estimated counts. */
    e->tok = qe_tok_open(e->mm, e->tok_err, sizeof e->tok_err);
    e->info.has_tokenizer = e->tok != NULL;
    if (e->tok)
        qe_tok_describe(e->tok, e->info.tokenizer, sizeof e->info.tokenizer);
    else
        snprintf(e->info.tokenizer, sizeof e->info.tokenizer,
                 "not loaded (%s); prompt token counts are estimated",
                 e->tok_err[0] ? e->tok_err : "unknown reason");

    const char *arch = NULL;
    if (qe_model_get_str(e->mm, "general.architecture", &arch) == 0 && arch && *arch)
        snprintf(e->info.arch, sizeof e->info.arch, "%s", arch);
    else
        snprintf(e->info.arch, sizeof e->info.arch, "unknown");

    const char *label = NULL, *size_label = NULL;
    if (qe_model_get_str(e->mm, "general.name", &label) == 0 && label)
        snprintf(e->info.label, sizeof e->info.label, "%s", label);
    if (qe_model_get_str(e->mm, "general.size_label", &size_label) == 0 && size_label)
        snprintf(e->info.size_label, sizeof e->info.size_label, "%s", size_label);

    /* Read the shape the same way a forward pass would: every architecture key
     * is prefixed with the architecture name. */
    {
        char key[128];
        long long v;
#define QE_SHAPE(field, suffix)                                          \
    snprintf(key, sizeof key, "%s." suffix, e->info.arch);               \
    if (qe_model_get_int(e->mm, key, &v) == 0)                           \
        e->info.field = (int)v;
        QE_SHAPE(n_blocks, "block_count")
        QE_SHAPE(n_embd, "embedding_length")
        QE_SHAPE(n_experts, "expert_count")
        QE_SHAPE(n_experts_active, "expert_used_count")
#undef QE_SHAPE
        snprintf(key, sizeof key, "%s.context_length", e->info.arch);
        if (qe_model_get_int(e->mm, key, &v) == 0)
            e->info.n_ctx_train = (int)v;
        size_t nv = 0;
        if (qe_model_get_arr_len(e->mm, "tokenizer.ggml.tokens", &nv) == 0)
            e->info.n_vocab = (int)nv;
        if (qe_model_get_int(e->mm, "tokenizer.ggml.eos_token_id", &v) == 0)
            e->info.eos_token_id = (int)v;
        snprintf(key, sizeof key, "%s.ple.image_token_id", e->info.arch);
        if (qe_model_get_int(e->mm, key, &v) == 0)
            e->info.image_token_id = (int)v;
        const char *tpl = NULL;
        if (qe_model_get_str(e->mm, "tokenizer.chat_template", &tpl) == 0 && tpl && *tpl)
            e->info.has_chat_template = 1;
    }

    size_t nt = qe_tensor_count(e->mm);
    long long bytes = 0;
    for (size_t i = 0; i < nt; i++)
        bytes += (long long)qe_tensor_by_index(e->mm, i)->bytes;
    e->info.n_tensors = (long long)nt;
    e->info.bytes = bytes;
    e->info.n_volumes = qe_model_volumes(e->mm);
    snprintf(e->info.stage, sizeof e->info.stage, "%s", qe_model_stage_method(e->mm));
    e->info.stage_bytes = (long long)qe_model_stage_bytes(e->mm);
    e->info.stage_cached = (long long)qe_model_stage_cached(e->mm);
    e->info.stage_pinned = qe_model_stage_pinned(e->mm);
    quant_mix(e->mm, e->info.quant, sizeof e->info.quant);

    /* Speculation needs a drafter, and a drafter needs weights. Search for one
     * rather than assume the file has none: an MTP head shows up as a tensor
     * named for it, and this checkpoint has no such tensor, so spec_k stays
     * honest at "nothing to draft with". A head shipped beside the checkpoint is
     * the other way to have one, and is picked up below from --mtp. */
    for (size_t i = 0; i < nt && !e->info.has_mtp_head; i++) {
        const char *nm = qe_tensor_by_index(e->mm, i)->name;
        if (strstr(nm, "mtp") || strstr(nm, "predict") || strstr(nm, "eagle") ||
            strstr(nm, "draft"))
            e->info.has_mtp_head = 1;
    }

    if (o->mmproj_path && *o->mmproj_path) {
        /* Same rule as the weights: the projector is resident before it is used,
         * so an image costs arithmetic and not a read. It is under a GiB, so
         * there is no version of this that is worth doing lazily. */
        e->mmproj = qe_open(o->mmproj_path, QE_STAGE_RAM | QE_STAGE_STRICT);
        if (!e->mmproj && err && errn)
            snprintf(err, errn, "warning: mmproj %s: %s", o->mmproj_path, qe_strerror());
    }

    if (o->mtp_path && *o->mtp_path) {
        if (stage_mtp(e, 1, err, errn)) {
            if (e->mtp)
                qe_close(e->mtp);
            if (e->mmproj)
                qe_close(e->mmproj);
            if (e->tok)
                qe_tok_close(e->tok);
            qe_close(e->mm);
            pthread_mutex_destroy(&e->lock);
            free(e);
            return NULL;
        }
    }

    /* The honest part, in both directions. Opening the forward pass here, at
     * launch, is deliberate: it is a lookup of every tensor the architecture
     * names, so a checkpoint missing one is a startup failure with the name of
     * the tensor in it, and not a 501 that appears on the first request. When it
     * opens, the sentence below says what is running; when it does not, naming
     * the missing pieces is the point, because "not implemented" is worth
     * nothing to a reader who cannot tell what a complete build would do. */
    char place[96];
    if (!strcmp(e->info.stage, "mmap"))
        snprintf(place, sizeof place, "mapped on demand (stage mmap)");
    else
        snprintf(place, sizeof place, "resident in RAM (stage %s)", e->info.stage);
    qe_sampler_opts_from_model(e->mm, &e->samp);
    /* Precedence, top to bottom: the checkpoint's general.sampling.*, then the
     * flags this process was started with, then whatever a request sends. Same
     * "0 means not sent, negative means off" convention at every level. */
    if (o->top_k > 0)
        e->samp.top_k = o->top_k;
    else if (o->top_k < 0)
        e->samp.top_k = 0;
    if (o->top_p > 0.0f)
        e->samp.top_p = o->top_p;
    if (o->min_p > 0.0f)
        e->samp.min_p = o->min_p;
    else if (o->min_p < 0.0f)
        e->samp.min_p = 0.0f;
    int n_gdn = 0, n_qsa = 0;
    for (int il = 0; il < e->info.n_blocks; il++) {
        char nm[QE_MAX_NAME];
        snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", il);
        if (qe_tensor_by_name(e->mm, nm))
            n_gdn++;
        else
            n_qsa++;
    }
    if (!strcmp(e->info.arch, "qwen4exp")) {
        /* Sequence capacity defaults to what the file says the model was trained
         * for. A server started with less quietly discards the beginning of every
         * long conversation, and the cache a token of context costs (19.7 KiB at
         * q8_0, so 5.0 GiB at 262144) is not a reason to shrink a model's own
         * window on a host with this much RAM. `make score` prints both numbers
         * for whatever capacity was chosen. */
        int dflt_ctx = e->info.n_ctx_train > 0 ? e->info.n_ctx_train : 4096;
        /* The lane's own memory, asked for before the pass it belongs to: a slot per
         * sequence and the batch's logits, which is one vocabulary per sequence plus
         * the extra row a prompt being ingested answers out of. Nine vocabularies at
         * this checkpoint's 248320 is 8.5 MiB, and an engine that cannot have it
         * cannot have anything, so it is a startup failure here rather than a server
         * that silently serves one request at a time behind a `--batch 8`. */
        if (e->n_batch > 1) {
            e->lanes = calloc((size_t)e->n_batch, sizeof *e->lanes);
            e->rows = malloc((size_t)(e->n_batch + 1) * (size_t)e->info.n_vocab *
                             sizeof *e->rows);
            if (!e->lanes || !e->rows) {
                if (err && errn)
                    snprintf(err, errn,
                             "out of memory for the batch of %d sequences (%d vocabularies"
                             " of logits): --batch without them is a queue, not a batch",
                             e->n_batch, e->n_batch + 1);
                free(e->lanes);
                free(e->rows);
                e->lanes = NULL;
                e->rows = NULL;
                if (e->mtp)
                    qe_close(e->mtp);
                if (e->mmproj)
                    qe_close(e->mmproj);
                if (e->tok)
                    qe_tok_close(e->tok);
                qe_close(e->mm);
                pthread_cond_destroy(&e->lane_cv);
                pthread_mutex_destroy(&e->lock);
                free(e);
                return NULL;
            }
        }
        qe_forward_opts fo = { .n_ctx = o->n_ctx > 0 ? o->n_ctx : dflt_ctx,
                               .n_seq = e->n_batch,
                               .n_threads = e->info.n_threads,
                               .walk_threads = o->walk_threads,
                               .kv_f32 = o->kv_f32,
                               .attn_head_walk = o->attn_head_walk,
                               .mtp_model = e->mtp };
        char ferr[512] = { 0 };
        e->fw = qe_forward_open(e->mm, &fo, ferr, sizeof ferr);
        /* What one request may use. The caches are shared and not copied, so a batched
         * pass gives each sequence its slice of them, and this is that slice - the
         * number a client budgets an answer against, and the number that makes
         * `--batch 8` a trade and not a free multiplier: eight sequences out of a
         * 262144-token window is 32768 tokens each. Reported as the context, because it
         * is the window a request on this server really has. */
        e->lane_ctx = e->fw ? fo.n_ctx / e->n_batch : 0;
        e->info.n_ctx = e->fw ? (int)e->lane_ctx : 0;
        e->info.n_batch = e->n_batch;
        if (!e->fw && err && errn)
            snprintf(err, errn, "warning: not generating: %s", ferr);
    }
    e->info.can_generate = e->fw != NULL;
    if (e->fw) {
        /* The hook belongs to the forward pass and not to a request: it outlives
         * every connection, and a request installs a pointer to itself in the
         * engine's copy of it. */
        qe_forward_set_progress(e->fw, fwd_progress, e);
    qe_forward_set_progress_fine(e->fw, fwd_progress_fine, e);
        if (e->n_batch > 1) {
            /* Every sequence at zero, once, which is what lets a lane be retired by
             * resetting its own sequence and nothing else: the first request to a slot
             * finds the same emptiness the tenth one does.
             *
             * The turn cache is not built, because a rewind point is one sequence's
             * state and eight of them is a budget nobody has. It is reported in the
             * same field that says "off" for a host that cannot afford a point. */
            qe_forward_reset(e->fw);
            snprintf(e->info.turn_cache_err, sizeof e->info.turn_cache_err,
                     "off (--batch %d: a rewind point is one sequence's state, and a"
                     " batched request reads its whole prompt every turn)",
                     e->n_batch);
        } else {
            turns_init(e);
        }
        /* The walk threshold the operator asked for, which outranks the environment
         * the pass read at open: positive blocks every projection of that size in
         * MiB and above, negative gives the ordered walk, 0 leaves the pass's own
         * default standing. Said with a setter rather than an extra field on the
         * forward opts because the number is one a running pass can change. */
        if (o->proj_chunk_mb > 0)
            qe_forward_set_proj_chunk(e->fw, (size_t)o->proj_chunk_mb * 1024 * 1024);
        else if (o->proj_chunk_mb < 0)
            qe_forward_set_proj_chunk(e->fw, 0);
    }
    snprintf(e->info.kv_cache, sizeof e->info.kv_cache,
             e->fw && qe_forward_kv_q8(e->fw) ? "q8_0" : "f32");
    int used;
    /* What speculation is running off, said in the same breath as the sampler,
     * because the two decide together how fast a decode goes: greedy requests
     * speculate and verify a whole window in one pass, sampled ones take the filters
     * in this loop and run a token at a time. The drafter is the nextn head when the
     * pass opened one and the n-gram hash otherwise - and with a head open the hash
     * is still what answers for a sequence too short to seed a chain. */
    const int head_drafts = e->fw && qe_forward_has_mtp(e->fw);
    e->info.mtp_drafting = head_drafts;
    /* The drafter decides the length, which is the one thing about speculation this
     * file does not guess at: 1.19x of the plain step loop at one draft off a head
     * and 1.05x at two off the hash, against 0.75x and 0.91x at eight (docs/DESIGN.md
     * 4, `/tmp/f40_head.log`). Anything the caller said wins. */
    if (e->o.spec_k == QE_SPEC_K_AUTO)
        e->o.spec_k = head_drafts ? QE_SPEC_K_HEAD : QE_SPEC_K_NGRAM;
    /* ideas.md §7.30 moved the step and not the window: a lone token is 1.3x faster
     * since its walks share dispatches and hand their rows out from a cursor, and a
     * verify window of two rows is still 1.45-1.6 steps of the faster step. A pass that
     * costs that plus a draft row and emits 1.6-1.7 ids is no faster than the plain
     * loop, so nothing speculates unless it is asked to (`--spec-k N`, or
     * QE_SPEC_AUTO=1 for the old resolution). */
    {
        const char *ev = getenv("QE_SPEC_AUTO");
        if (e->o.spec_k > 0 && o->spec_k == QE_SPEC_K_AUTO && !(ev && atoi(ev) > 0))
            e->o.spec_k = 0;
    }
    e->info.spec_k = e->o.spec_k;
    /* A head nothing will draft from is still walked once per prompt chunk and once per
     * step to keep its cache warm: one more block on every token, for drafts that never
     * come. With speculation off, it is resident and idle; --spec-k N brings it back. */
    if (head_drafts && e->o.spec_k <= 0)
        qe_forward_set_mtp_warm(e->fw, 0);
    char specnote[640];
    if (e->o.spec_k > 0)
        snprintf(specnote, sizeof specnote,
                 "Greedy generation speculates %d tokens a window off the %s and "
                 "verifies each window in one pass (--spec-k 0 turns it off, and the "
                 "default is the length that drafter measures worth); sampled "
                 "requests run one token at a time.",
                 e->o.spec_k, head_drafts ? "resident nextn head" : "n-gram drafter");
    else
        snprintf(specnote, sizeof specnote,
                 "Every request runs one token at a time (--spec-k N adds %s to greedy "
                 "ones).",
                 head_drafts ? "nextn-head speculation" : "n-gram speculation");
    if (e->info.has_mtp_head && !head_drafts)
        strncat(specnote,
                " The draft head is resident but the pass could not open it, so the "
                "n-gram drafter answers instead; what was missing was printed at open.",
                sizeof specnote - strlen(specnote) - 1);
    if (e->n_batch > 1)
        strncat(specnote,
                " A batched lane runs neither: a draft window and a rewind point are one"
                " sequence's state, so --spec-k and --turn-cache ask for nothing here and"
                " every request reads its whole prompt.",
                sizeof specnote - strlen(specnote) - 1);
    if (e->fw) {
        size_t st_total = 0, st_token = 0;
        qe_forward_state_bytes(e->fw, &st_total, &st_token);
        /* What sharing a weight walk costs a sequence, said in the same sentence that
         * says it is sharing one: the window each of them gets and the state each of
         * them needs, because those are the two things a batch trades away and an
         * operator reading /v1/models cannot see them from a count of sequences. */
        char lanenote[224];
        if (e->n_batch > 1)
            snprintf(lanenote, sizeof lanenote,
                     "%d sequences through one walk of the weights, %d tokens of window "
                     "each (the caches are divided and the recurrent state copied, so a "
                     "batch is a share of the machine and not a machine per request)",
                     e->n_batch, e->info.n_ctx);
        else
            snprintf(lanenote, sizeof lanenote, "One sequence at a time");
        used = snprintf(e->info.why, sizeof e->info.why,
             "generating. %d layers (%d gated-delta-net, %d sparse attention with the "
             "indexer, hyper-connections and %d-of-%d routed experts) executed from %s "
             "(%s, %lld tensors, %.1f GiB, %d volumes), %s. %s, "
             "%d tokens of state, key and value cache %s (%.0f MiB of state, %.1f KiB "
             "per token of context); sampler starts at the file's own "
             "top_k %d, top_p %.2f, min_p %.3f, "
             "temp %.2f. Images are described in the prompt text: the vision projector is "
             "resident but no encoder drives it. %s",
             e->info.n_blocks, n_gdn, n_qsa, e->info.n_experts_active, e->info.n_experts,
             o->model_path, e->info.arch, e->info.n_tensors, (double)bytes / 1073741824.0,
             e->info.n_volumes, place, lanenote, e->info.n_ctx, e->info.kv_cache,
             (double)st_total / 1048576.0, (double)st_token / 1024.0,
             e->samp.top_k, (double)e->samp.top_p, (double)e->samp.min_p,
             (double)e->samp.temp, specnote);
    } else {
        used = snprintf(e->info.why,
             sizeof e->info.why,
             "checkpoint %s (%s, %lld tensors, %.1f GiB of weights, %d volumes) is %s "
             "but not executed: %s. This build reads its shape (%d layers, %d experts "
             "with %d active, %d wide, %d-token vocabulary), its quant mix, its tokenizer and "
             "its chat template, and then refuses to generate. The tensor-level contract and "
             "the layer-by-layer forward pass are specified in docs/ARCH.md section 6; "
             "src/forward.c implements that for arch qwen4exp and refuses to guess at a "
             "tensor it cannot find rather than reading a zero for it, which is what this "
             "refusal means. No logits are produced, so none are invented.",
             o->model_path, e->info.arch, e->info.n_tensors, (double)bytes / 1073741824.0,
             e->info.n_volumes, place,
             (err && err[0]) ? err : "no forward pass is implemented for this architecture",
             e->info.n_blocks, e->info.n_experts, e->info.n_experts_active, e->info.n_embd,
             e->info.n_vocab);
    }
    /* Silent truncation is the failure this file goes out of its way to avoid:
     * the sentence above is the reason given to every client that asks. */
    if (used >= (int)sizeof e->info.why)
        fprintf(stderr, "qe: the engine message did not fit in %zu bytes\n",
                sizeof e->info.why);
    return e;
}

void qe_engine_close(qe_engine *e) {
    if (!e)
        return;
    /* A batched lane is driven by whichever request arrived to an empty lane table, and
     * that thread owns the forward pass until the last sequence retires. Freeing the pass
     * under it would be a step through freed weights, so close waits for the lane to
     * empty - and it does empty, because every lane either answered, errored, or had its
     * client stop reading, and the driver keeps walking while any of them is standing.
     * A serial engine has no lane and goes straight through. */
    pthread_mutex_lock(&e->lock);
    while (e->lane_live > 0)
        pthread_cond_wait(&e->lane_cv, &e->lock);
    pthread_mutex_unlock(&e->lock);
    turns_discard(e);
    free(e->turns);
    free(e->turn_views);
    spec_drop(&e->spec);
    if (e->fw)
        qe_forward_close(e->fw);
    if (e->tok)
        qe_tok_close(e->tok);
    if (e->mtp)
        qe_close(e->mtp);
    if (e->mmproj)
        qe_close(e->mmproj);
    if (e->mm)
        qe_close(e->mm);
    pthread_mutex_destroy(&e->lock);
    pthread_cond_destroy(&e->lane_cv);
    free(e->fed);
    free(e->tail);
    free(e->lanes);
    free(e->rows);
    free(e);
}

void qe_engine_describe(const qe_engine *e, qe_engine_info *out) {
    if (e)
        *out = e->info;
    else
        memset(out, 0, sizeof *out);
}

qe_tok *qe_engine_tok(qe_engine *e) {
    return e ? e->tok : NULL;
}

/* ---------------------------------------------------------- selftest target */

/* The synthetic target's next token is the corpus word for that position, so
 * the correct greedy output is knowable without a model, and the drafter is a
 * deliberately noisy version of the truth - which is what makes the accept
 * statistics on a captured response worth reading. */
typedef struct {
    const char *const *corpus;
    int n_corpus; /* ids 1..n_corpus-1 are words; 0 is end-of-sequence */
    int words;
    int any;
    qe_token_cb cb;
    void *user;
    int cb_failed;
    uint64_t rng;
} st_ctx;

#define ST_EOS 0

/* After the corpus has been produced once, the target's own choice is eos, so
 * a short max_tokens ends with finish_reason "length" and a long one with
 * "stop". Both paths matter and both are reachable. */
static int st_want(long pos, int n_corpus) {
    int words = n_corpus - 1;
    if (words < 1)
        return ST_EOS;
    if (pos / words > 0)
        return ST_EOS;
    return 1 + (int)(((unsigned long)pos * 7u + 5u) % (unsigned)words);
}

static int st_verify(void *vp, long long pos, const int *toks, int n, float *logits) {
    st_ctx *c = vp;
    (void)toks;
    for (int j = 0; j < n; j++) {
        float *row = logits + (size_t)j * (size_t)c->n_corpus;
        int want = st_want(pos + j + 1, c->n_corpus);
        for (int i = 0; i < c->n_corpus; i++)
            row[i] = -0.2f * (float)((i + 3 * want + 1) % c->n_corpus);
        row[want] = 4.0f;
    }
    return 0;
}

static uint64_t st_rnd(uint64_t *s) {
    uint64_t z = *s += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static int st_draft(void *vp, long long pos, const int *toks, int n, int *out, float *probs,
                    int k) {
    st_ctx *c = vp;
    (void)toks;
    (void)n;
    for (int i = 0; i < k; i++) {
        int want = st_want(pos + (long)i, c->n_corpus);
        /* A perfect drafter reports k+1 tokens per step forever and says
         * nothing about rejections, so one position in seven is wrong. */
        if (((st_rnd(&c->rng) >> 11) % 100u) < 15u)
            want = st_want(pos + (long)i + 3, c->n_corpus);
        out[i] = want == ST_EOS ? 1 : want;
        if (probs)
            probs[i] = 0.5f;
    }
    return k;
}

static int st_emit(void *vp, int tok, int is_eos) {
    st_ctx *c = vp;
    if (is_eos || tok == ST_EOS)
        return 1;
    if (tok < 0 || tok >= c->n_corpus)
        return 1;
    char sep = (c->any && c->words % 12 == 0) ? '\n' : ' ';
    c->words++;
    c->any = 1;
    qe_ui_emit();
    qe_token_out out = { .id = tok, .special = 0 };
    out.utf8 = &sep;
    out.nbytes = 1;
    if (c->cb(c->user, &out)) {
        c->cb_failed = 1;
        return 1;
    }
    out.utf8 = c->corpus[tok];
    out.nbytes = (int)strlen(c->corpus[tok]);
    if (c->cb(c->user, &out))
        c->cb_failed = 1;
    return c->cb_failed;
}

/* ------------------------------------------------------------- real weights */

/* Generation over a checkpoint. One sequence lives in the forward pass at a
 * time, so the whole call is under the engine lock; state is cleared here rather
 * than by a pool of sessions, which means two concurrent requests are served
 * one after the other rather than interleaved. The prompt is fed one token at a
 * time - the kernels are row-at-a-time GEMVs, and batching them is a change to
 * the kernels, not to this loop. */

/* The transcript bookkeeping, which has to say exactly what the forward pass was
 * fed: a token the decoder emitted but never entered (the end of sequence, say)
 * is not part of it, and a stale entry here is a wrong answer rather than a slow
 * one. Both are called under the engine lock. */
static void fed_reset(qe_engine *e) {
    e->fed_n = 0;
    e->fed_valid = 1;
    e->tail_valid = 0;
}

/* Appends to the transcript. Running out of room cannot happen while a sequence
 * fits (a prompt is shortened to leave room for what it might generate, and the
 * transcript is never longer than prompt plus generation) but if it ever did, the
 * honest response is to stop trusting the transcript, not to keep a window of it
 * and compare a conversation against the middle of one. */
static int fed_append(qe_engine *e, const int *ids, long n, long cap) {
    if (!e->fed) {
        e->fed = malloc((size_t)cap * sizeof *e->fed);
        e->fed_n = 0;
        if (!e->fed)
            return -1;
    } else if (e->fed_n + n > cap) {
        e->fed_n = 0;
        e->fed_valid = 0;
        return 0;
    }
    memcpy(e->fed + e->fed_n, ids, (size_t)n * sizeof *e->fed);
    e->fed_n += n;
    return 0;
}

static int tail_ensure(qe_engine *e, long vocab) {
    if (!e->tail)
        e->tail = malloc((size_t)vocab * sizeof *e->tail);
    return e->tail ? 0 : -1;
}

/* --------------------------------------------------- speculative decoding
 *
 * One step of this model reads 5.44 GiB whether it decides one token or four, so
 * the only lever left on decode speed is how many tokens a step is allowed to
 * decide (docs/DESIGN.md 4, ideas.md 7.4). This is that lever wired to the one
 * drafter that needs no weights: an index of the n-grams already in the sequence
 * (src/draft.c), proposing whatever followed the longest tail of it the last time
 * the engine saw it. For a coding agent that is the most repeatable text there
 * is - the same file comes back every turn, and the commands it ran are in the
 * transcript - and where it finds nothing the loop degenerates to one token a
 * step, which is slow and not wrong.
 *
 * What it costs, measured on this host (the `make forward` width sweep, saved in
 * bench_mtp's own table): a window of 2 rows is 123 ms against 112 for one, of 5
 * rows 226 ms, of 8 rows 277 ms. So a draft has to be right often enough to pay
 * for the rows it wasted, and a window that is wider than the drafter's evidence
 * is a loss rather than a smaller win - which is why the rows past the first
 * disagreement are walked back out of the sequence (qe_forward_rewind) instead of
 * being kept, and why each of them is not charged for the output matrix.
 *
 * Greedy only, on purpose and not by omission. The top-k, nucleus and min-p
 * filters are applied by the loop below and not by the speculation loop, so a
 * sampled request routed through here would answer with a different distribution
 * than the one it asked for; and a count over a hash table is not a distribution
 * that a rejection test could keep the target's answer with. A sampled request
 * takes the plain path and gets exactly what it asked for.
 *
 * The invariant the whole thing hangs on, and it is one line: while a speculative
 * generation is running, the forward pass stands at one token short of the
 * transcript, because the next window starts by feeding the token the last window
 * earned. The step taken after the loop is what makes the two equal again, and
 * every check in the code above that compares them runs after it. */
struct spec_target {
    qe_engine *e;
    qe_spec_model m; /* the callbacks below, with this as their ctx */
    qe_draft *ngram;
    /* The draft head, when one is open: two residual rows to chain through - the one
     * a draft row is fed and the one it exports, ping-ponged - and one row of the
     * head's logits. NULL with no head, which is what leaves every draft to the
     * n-gram drafter. */
    float *m_h;
    float *m_logits;
    long drafted;  /* tokens the head proposed, for the same count mtp.c keeps */
    qe_ckpt *ck;
    int rows;      /* boundaries `ck` holds, and the widest window it takes */
    int win;       /* rows in the window the last row was read from */
    long vocab;

    /* The logits after the prompt, which the loop's first window is the first
     * thing to compute and the turn cache wants before anything is generated. */
    float *prompt_logits;
    int prompt_have;

    /* Whether the prompt's own rewind point is being asked for, and what taking it
     * cost. A request that speculates never stands at the end of its prompt: the
     * last prompt token is row 0 of a window, and the state after it is a boundary
     * of that window rather than something the pass still holds. */
    int turn_prompt;
    long boundary_us;

    /* The last row of the last window whose head was taken. A window that ended on
     * a token the transcript did not take - the marker that closes a turn, or a
     * callback that stopped streaming - leaves the target standing exactly on that
     * row, and its head is the frontier's distribution, which is the one thing both
     * the caller and the turn cache still want. */
    int last_read;
    int read_any;

    /* the request in hand */
    qe_token_cb cb;
    void *user;
    long cap;    /* the sequence's ceiling, which is the transcript's bound */
    long emitted;
    int last;    /* the last token the transcript took ... */
    int open;    /* ... and the forward pass has yet to be fed */
    int failed;

    long windows; /* target passes taken */
    long fed_rows;/* rows of those passes */
};

static int spec_row(void *vp, long long pos, const int *toks, int n, int i, float *out);
static void spec_rewind(void *vp, long long pos, int used);
static int spec_draft(void *vp, long long pos, const int *toks, int n, int *out, float *probs,
                      int k);
static int spec_emit(void *vp, int tok, int is_eos);

static void spec_drop(spec_target **sp) {
    if (!sp || !*sp)
        return;
    qe_draft_close((*sp)->ngram);
    qe_ckpt_free((*sp)->ck);
    free((*sp)->m_h);
    free((*sp)->m_logits);
    free((*sp)->prompt_logits);
    free(*sp);
    *sp = NULL;
}

/* Room for a window of `rows` rows, once. Growing one is a rebuild: the draft
 * length is a start-up setting for a server, and a request that asks for a wider
 * window than the last is rare enough to be worth being honest about. */
static spec_target *spec_get(qe_engine *e, int rows, long vocab) {
    if (rows < 2)
        return NULL;
    if (e->spec && e->spec->rows >= rows)
        return e->spec;
    spec_drop(&e->spec);
    spec_target *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    s->e = e;
    s->vocab = vocab;
    s->rows = rows;
    s->ngram = qe_draft_open(0, 0);
    s->ck = qe_ckpt_new(e->fw, rows);
    s->prompt_logits = malloc((size_t)vocab * sizeof *s->prompt_logits);
    if (!s->ngram || !s->ck || !s->prompt_logits) {
        spec_drop(&s);
        return NULL;
    }
    /* The head's front end, when there is a head: two rows of the four-stream
     * residual and one row of its logits. A head that cannot be allocated is a
     * trunk drafting off the n-gram hash and not a failed request, so the two
     * pointers being NULL is a state every path below has to survive. */
    if (qe_forward_has_mtp(e->fw)) {
        const int hc = qe_forward_hc_dim(e->fw);
        s->m_h = malloc(2 * (size_t)hc * sizeof *s->m_h);
        s->m_logits = malloc((size_t)vocab * sizeof *s->m_logits);
        if (!s->m_h || !s->m_logits) {
            free(s->m_h);
            free(s->m_logits);
            s->m_h = NULL;
            s->m_logits = NULL;
        }
    }
    s->m.ctx = s;
    s->m.vocab = vocab;
    s->m.verify_row = spec_row;
    s->m.rewind = spec_rewind;
    s->m.draft = spec_draft;
    s->m.emit = spec_emit;
    e->spec = s;
    return s;
}

static int spec_row(void *vp, long long pos, const int *toks, int n, int i, float *out) {
    spec_target *s = vp;
    if (i < 0 || n > s->rows)
        return -1;
    if (i == 0) {
        s->win = n;
        s->windows++;
        s->fed_rows += n;
        if (qe_forward_step_window(s->e->fw, toks, n, s->ck))
            return -1;
    }
    if (qe_forward_row_logits(s->e->fw, i, out))
        return -1;
    s->last_read = i;
    s->read_any = 1;
    if (i == 0 && s->prompt_logits && !s->prompt_have) {
        s->prompt_have = 1;
        memcpy(s->prompt_logits, out, (size_t)s->vocab * sizeof *out);
        /* Row 0 is the last token of the prompt, so this is the one moment the
         * request has the state that goes with it, and it is the moment the next
         * turn comes back to. With more than one row it is not in the pass - the
         * pass is standing at the window's last row - but it is in the checkpoint,
         * which is where a boundary is copied from instead. */
        if (s->turn_prompt)
            s->boundary_us =
                n > 1 ? turns_store_boundary(s->e, (long)pos + 1, s->ck, 1, out, s->vocab, 0)
                      : turns_store(s->e, (long)pos + 1, out, s->vocab, 0);
    }
    return 0;
}

static void spec_rewind(void *vp, long long pos, int used) {
    spec_target *s = vp;
    (void)pos;
    /* A window whose every row was read left the pass standing exactly where the
     * next window starts, and boundary 0 is not something a window can put back
     * (it is the state before the window, which is the one thing not copied). */
    if (used < 1 || used >= s->win)
        return;
    if (qe_forward_rewind(s->e->fw, s->ck, used) < 0)
        s->failed = 1;
}

/* The probability a draft distribution needs for a rejection test: the head's own,
 * read off its logits. Greedy verification never asks for it (an index has no
 * probability to report), and a sampled request does not come through speculation at
 * all in this engine, so this is here because the callback's contract says a draft
 * reports what it believes and not because a caller in this tree reads it. */
static float draft_prob(const float *logits, long vocab, int id) {
    long best = 0;
    for (long v = 1; v < vocab; v++)
        if (logits[v] > logits[best])
            best = v;
    double sum = 0.0;
    for (long v = 0; v < vocab; v++)
        sum += exp((double)logits[v] - (double)logits[best]);
    if (!(sum > 0.0))
        return 1.0f;
    return (float)(exp((double)logits[id] - (double)logits[best]) / sum);
}

/* The nextn head, chained. Row 0 is fed the token the transcript ends on and the
 * residual the target left at the position the head pairs with it; every row after is
 * fed the token the row before chose and the residual that row exported. That is the
 * chain the block was trained to walk - ref/qwen4exp.cpp:654 exports exactly the
 * residual the next row reads - and it is why k drafted tokens cost k walks of one
 * block and one block's output matrix, against a target step that moves 5.4 GiB.
 *
 * The n-gram drafter is the fallback and not a relic: with no head open it is the
 * drafter, and a frontier with no residual to read - the first token of a session,
 * where there is nothing to seed a chain with - is better served by the hash table's
 * guess than by no draft at all. A row that fails mid-chain stops the chain and keeps
 * what was already proposed, because a short draft is a smaller win and a wrong one
 * is nothing. */
static int spec_draft(void *vp, long long pos, const int *toks, int n, int *out, float *probs,
                     int k) {
    spec_target *s = vp;
    qe_forward *fw = s->e->fw;
    (void)pos;
    if (!s->m_h || !qe_forward_has_mtp(fw))
        return qe_draft_suggest(s->ngram, toks, n, out, k);

    const int hc = qe_forward_hc_dim(fw);
    const long frontier = (long)n - 1; /* the position the transcript ends on */
    float *h_in = s->m_h, *h_out = s->m_h + hc;
    if (qe_forward_mtp_hidden(fw, frontier - qe_forward_get_mtp_back(fw), h_in) < 0)
        return qe_draft_suggest(s->ngram, toks, n, out, k);

    int tok = toks[n - 1], nd = 0;
    for (int i = 0; i < k; i++) {
        if (qe_forward_mtp_row(fw, tok, frontier + i, h_in, h_out, s->m_logits))
            break;
        long best = 0;
        for (long v = 1; v < s->vocab; v++)
            if (s->m_logits[v] > s->m_logits[best])
                best = v;
        out[nd++] = (int)best;
        if (probs)
            probs[nd - 1] = draft_prob(s->m_logits, s->vocab, (int)best);
        float *swap = h_in;
        h_in = h_out;
        h_out = swap;
        tok = (int)best;
    }
    s->drafted += nd;
    if (!nd)
        return qe_draft_suggest(s->ngram, toks, n, out, k);
    return nd;
}

/* One token of the answer, exactly as the plain loop below reports it: decoded,
 * streamed, counted, and only then added to the transcript - and the token that
 * ends the turn is reported and not kept, because the forward pass was never told
 * about it. */
static int spec_emit(void *vp, int tok, int is_eos) {
    spec_target *s = vp;
    char piece[128];
    long n = qe_tok_decode(s->e->tok, &tok, 1, piece, sizeof piece);
    int special = qe_tok_is_special(s->e->tok, tok);
    qe_token_out out = { .id = tok, .special = special };
    out.utf8 = (!special && n > 0) ? piece : "";
    out.nbytes = (!special && n > 0) ? (int)n : 0;
    /* Two ways a token is reported and not kept, and both close the turn: a caller
     * that stopped listening, and the marker that ends it. Neither is fed, so the
     * target is already standing where the transcript ends and the token held back
     * for the step after the loop has to go back to being nothing - which is what
     * the loop's boundary counts on. */
    if (s->cb(s->user, &out)) {
        s->open = 0;
        return 1;
    }
    if (is_eos) {
        s->open = 0;
        return 1;
    }
    s->emitted++;
    qe_ui_emit();
    if (fed_append(s->e, &tok, 1, s->cap))
        s->failed = 1;
    s->last = tok;
    s->open = 1;
    return s->failed;
}


/* What this request samples with: the file's defaults, moved by the engine's flags,
 * moved by what the request itself sent. The same rules on both lanes, because a
 * request must not get a different distribution by arriving at a busy moment.
 *
 * 0 means "not sent", and for top_k and min_p a negative is the only way to ask for
 * off when the checkpoint ships a non-zero default. */
static qe_sampler_opts samp_for(const qe_engine *e, const qe_gen_opts *o) {
    qe_sampler_opts so = e->samp;
    if (o->temp >= 0.0f)
        so.temp = o->temp;
    if (o->top_k > 0)
        so.top_k = o->top_k;
    else if (o->top_k < 0)
        so.top_k = 0;
    if (o->top_p > 0.0f)
        so.top_p = o->top_p;
    else if (o->min_p > 0.0f)
        so.min_p = o->min_p;
    else if (o->min_p < 0.0f)
        so.min_p = 0.0f;
    return so;
}

static long gen_checkpoint(qe_engine *e, const qe_prompt *p, const qe_gen_opts *o,
                           qe_token_cb cb, void *user, qe_gen_stats *st) {
    long rc = -1;
    long emitted = 0;
    int dropped = 0;
    int *ids = NULL;
    float *logits = NULL;
    if (!e->tok) {
        /* Without the file's tokenizer there is no honest way to turn text into
         * ids, and inventing an encoding is worse than refusing. */
        errno = ENOSYS;
        return -1;
    }

    pthread_mutex_lock(&e->lock);
    double t0 = now();
    /* The request's own ingestion hook, for the length of this call: the forward pass
     * has one hook and it is the engine's, so a serial generation is the one case where
     * "the request in hand" and "the engine" are the same thing. Cleared on the way out,
     * which is also the failure path - a pointer to a connection that outlives the
     * request is a pointer to a closed socket. */
    e->user_progress = o->progress;
    e->user_progress_user = o->progress_user;

    long vocab = qe_forward_vocab(e->fw);
    int ctx = qe_forward_ctx(e->fw);
    int eos = o->eos >= 0 ? o->eos : qe_tok_eos(e->tok);

    long need = qe_tok_count(e->tok, p->text);
    ids = malloc(((size_t)need + 8) * sizeof *ids);
    logits = malloc((size_t)vocab * sizeof *logits);
    if (!ids || !logits)
        goto done;
    long np = qe_tok_encode(e->tok, p->text, ids, need + 8);
    if (np <= 0) {
        errno = EINVAL;
        goto done;
    }

    long mt = o->max_tokens > 0 ? o->max_tokens : 64;
    /* The window is the only ceiling on an answer, and a reasoning model needs
     * all of it: what the prompt left is the budget, and an ask larger than that
     * is trimmed to it rather than refused or cut to a round number. No constant
     * belongs here - a thinking block of 12000 tokens is normal for this model,
     * and 74.9 ms a token is what actually bounds how long a reply may run. */
    if (ctx > 0 && mt > (long)ctx - np - 1)
        mt = (long)ctx - np - 1;
    if (mt < 0)
        mt = 0;
    /* The sequence is fixed, so a prompt that does not fit leaves from the
     * front. Dropping the start of a chat transcript is lossy and the only
     * alternative is refusing a request that has a full answer in it. */
    if (np + mt + 1 > ctx) {
        long keep = (long)ctx - mt - 1;
        if (keep <= 0) {
            errno = ENOSPC;
            goto done;
        }
        memmove(ids, ids + (np - keep), (size_t)keep * sizeof *ids);
        np = keep;
        dropped = 1;
    }

    qe_sampler_opts so = samp_for(e, o);
    uint64_t rng =
        o->seed ? (uint64_t)o->seed : (uint64_t)(t0 * 1e9) ^ 0x853c49e6748fea9bull;

    /* Is this the same conversation, extended? The transcript has to match token
     * for token up to what it holds, and the forward pass still has to be
     * standing where the transcript ends - a forward pass that was reset, or told
     * something else in between, has already forgotten it. A prompt that had to
     * have its beginning thrown away to fit is a different sequence of positions
     * than the one that was fed, so it never reuses. */
    long reuse = 0;
    const float *cached = NULL;
    int rewound = 0;
    long snap_us = 0;
    /* How far this prompt agrees with anything the engine could go back to, kept
     * even for the answer that it cannot go back anywhere. A client that rewrote an
     * old tool result wants the token it diverged at rather than a zero, and this is
     * the only place in the engine that ever sees both the prompt and the history. */
    long lcp = 0;
    if (e->prefix_cache && e->fed_valid && !dropped && e->fed_n > 0 && e->fed &&
        qe_forward_pos(e->fw) == e->fed_n) {
        long lim = e->fed_n < np ? e->fed_n : np;
        long same = 0;
        while (same < lim && e->fed[same] == ids[same])
            same++;
        lcp = same;
        if (same == e->fed_n)
            reuse = same;
    }
    /* A prompt that is entirely inside the transcript has nothing to feed, and the
     * logits it needs are the ones after the last token that was fed. Without
     * those there is nothing to continue from, and re-feeding the whole thing is
     * the honest answer. */
    if (reuse == np && (!e->tail_valid || !e->tail))
        reuse = 0;
    if (reuse == np)
        cached = e->tail;

    /* Nothing carried over, so ask the boundaries kept from earlier requests. This
     * is the case a coding agent actually lives in: its next prompt is not the
     * sequence being held plus new tokens, because the turn just answered comes back
     * rendered from the client's own copy of it and rarely encodes to the ids that
     * were produced token by token. It does normally begin with the prompt that was
     * read last time - and a boundary is a state plus the ids that made it, so
     * agreeing with those ids is the whole of what a match has to mean. A prompt
     * whose front had to be thrown away to fit is not exempt either: if its first
     * tokens spell a boundary's whole prefix, that boundary is genuinely the state
     * after those tokens, at those positions. */
    if (!reuse && e->prefix_cache && e->n_turns > 0) {
        long at = 0, far = 0;
        int slot = turns_match(e, ids, np, &at, &far);
        if (far > lcp)
            lcp = far;
        if (slot >= 0 && qe_forward_restore(e->fw, e->turns[slot].st) == 0) {
            reuse = at;
            rewound = 1;
            if (reuse == np)
                cached = e->turns[slot].tail;
        }
    }

    e->ingest_base = reuse;
    e->ingest_total = np;
    qe_ui_gen_begin(np, reuse);
    /* Speculation, when it is asked for: greedy, with a prompt that leaves room for
     * a window, and not the case where the prompt is the transcript the pass is
     * already standing on - a window has to start one token short of the sequence,
     * and that case has no token left to hold back. A window that will not allocate
     * is not a reason to fail a request: the plain loop below is the same network
     * and the same answer, one token at a time.
     *
     * The drafter is left to sort out which conversation it is indexing. It keeps
     * one index of the sequence it was last handed and starts over when what it is
     * handed stops being an extension of it, so a second request on one server
     * reuses the index of a conversation it really is a continuation of and nothing
     * else - and what is left over is a proposal the target rejects, which is the
     * only kind of mistake a draft is allowed to make. */
    spec_target *sp = NULL;
    qe_spec_stats spec_st = { 0, 0, 0, 0, 0, 0 };
    long feed_to = np;
    if (o->spec_k > 0 && so.temp <= 0.0f && mt > 0 && reuse != np && np >= 2 && ctx > 0 &&
        mt <= (long)ctx - np - o->spec_k)
        sp = spec_get(e, o->spec_k + 1, vocab);
    if (sp) {
        feed_to = np - 1;
        sp->cb = cb;
        sp->user = user;
        sp->cap = (long)ctx;
        sp->emitted = 0;
        sp->open = 0;
        sp->prompt_have = 0;
        sp->turn_prompt = e->prefix_cache && !(rewound && reuse == np);
        sp->boundary_us = 0;
        sp->failed = 0;
        sp->windows = 0;
        sp->fed_rows = 0;
        sp->last_read = 0;
        sp->read_any = 0;
    }
    if (reuse == np) {
        memcpy(logits, cached, (size_t)vocab * sizeof *logits);
    } else if (reuse && feed_to > reuse) {
        if (qe_forward_prefill(e->fw, ids + reuse, feed_to - reuse, logits)) {
            errno = EIO;
            goto done;
        }
    } else if (feed_to > reuse) {
        qe_forward_reset(e->fw);
        fed_reset(e);
        if (qe_forward_prefill(e->fw, (const int *)ids, feed_to, logits)) {
            errno = EIO;
            goto done;
        }
    } else {
        qe_forward_reset(e->fw);
        fed_reset(e);
    }
    const double t_read = now();
    if (rewound) {
        /* The transcript is this prompt now. Whatever the engine remembered past the
         * boundary it restored describes a conversation that diverged somewhere, and
         * keeping any of it would be a claim about tokens that were never fed. */
        fed_reset(e);
        if (fed_append(e, ids, np, (long)ctx) || tail_ensure(e, vocab)) {
            errno = ENOMEM;
            goto done;
        }
    } else if (fed_append(e, ids + reuse, np - reuse, (long)ctx) || tail_ensure(e, vocab)) {
        errno = ENOMEM;
        goto done;
    }
    /* The boundary the next turn starts from: the prompt read, the answer not begun.
     * A client that re-renders the turn it was given comes back to this position,
     * and what it then pays for is only what it added. Skipped when the sequence was
     * just restored from this same position, because copying it back over itself
     * buys nothing - and skipped for a draft window, whose first row is what those
     * logits are, taken below where it has been taken. */
    if (e->prefix_cache && !sp && !(rewound && reuse == np))
        snap_us += turns_store(e, np, logits, vocab, 0);

    /* One of the two loops runs a request, never both: this one samples, feeds, and
     * appends a token at a time, and the other (src/mtp.c) drafts, verifies a window
     * of rows, and appends what agreed. They agree on what a token is worth because
     * the rules for keeping a draft are the same rules, and on one arithmetic - the
     * one where a window row is the same computation as a decode step - the sequence
     * that comes out is plain greedy's, which is what `make spec` holds them to.
     * The arithmetic that ships is the other one, and it costs byte equality rather
     * than tokens: see the speculation section of qe.h. */
    for (long i = 0; !sp && i < mt; i++) {
        int id = qe_sample(logits, vocab, &so, &rng);
        if (id < 0)
            break;
        /* Every token is reported, control tokens included, and the caller
         * decides what a turn boundary means. Dropping them silently is what
         * made a reasoning model look like it never reasoned: the block
         * delimiters were the only evidence, and they were being thrown away
         * one token at a time. */
        char piece[128];
        long n = qe_tok_decode(e->tok, &id, 1, piece, sizeof piece);
        int special = qe_tok_is_special(e->tok, id);
        qe_token_out out = { .id = id, .special = special };
        out.utf8 = (!special && n > 0) ? piece : "";
        out.nbytes = (!special && n > 0) ? (int)n : 0;
        if (cb(user, &out))
            break;
        if (eos >= 0 && id == eos)
            break;
        emitted++;
        qe_ui_emit();
        if (qe_forward_step(e->fw, id, logits)) {
            errno = EIO;
            goto done;
        }
        /* What the forward pass was fed, and nothing else: the token that ended
         * generation was chosen but never entered, so it is not part of the
         * transcript a later request would continue. */
        if (fed_append(e, &id, 1, (long)ctx)) {
            errno = ENOMEM;
            goto done;
        }
    }

    if (sp) {
        qe_spec_opts ss = { .k = o->spec_k,
                            .temp = 0.0f,
                            .seed = 0,
                            .max_tokens = mt,
                            .eos = eos };
        if (qe_spec_generate(&sp->m, &ss, ids, (int)np, &spec_st) < 0) {
            errno = sp->failed ? ENOMEM : EIO;
            goto done;
        }
        if (sp->failed) {
            errno = ENOMEM;
            goto done;
        }
        emitted = sp->emitted;
        /* The token a window earned was never fed: the next window was going to
         * feed it, and there is no next window. This one pass is what puts the
         * forward pass and the transcript back in agreement - and it is one pass
         * per request rather than per token, which is why holding the token back
         * costs nothing worth naming.
         *
         * When the turn ended on a token the transcript did not take there is no
         * token to feed, and the frontier's logits are the head of the row it came
         * from, which the window left in the trunk. One head walk buys the same
         * tail the plain loop gets for free on its last step. */
        if (sp->open && qe_forward_step(e->fw, sp->last, logits)) {
            errno = EIO;
            goto done;
        }
        if (!sp->open && sp->read_any &&
            qe_forward_row_logits(e->fw, sp->last_read, logits)) {
            errno = EIO;
            goto done;
        }
        if (!sp->open && !sp->read_any && sp->prompt_have)
            memcpy(logits, sp->prompt_logits, (size_t)vocab * sizeof *logits);
        /* The prompt's boundary was taken while it was still on hand, at the row
         * that earned these logits; all that is left to do is charge for it. */
        snap_us += sp->boundary_us;
    }

    /* The logits after the last token fed are what a request that adds nothing to
     * the transcript needs, and recomputing them costs a full pass. */
    if (e->tail) {
        memcpy(e->tail, logits, (size_t)vocab * sizeof *e->tail);
        e->tail_valid = 1;
    } else
        e->tail_valid = 0;

    /* The other boundary, with the answer included. Worth a second copy because a
     * client that replays the turn's tokens exactly - which the re-rendering ones
     * do not - continues from here instead, and then pays only for what it puts
     * after the answer. Only when there is an answer: with none, this is the
     * boundary that was already taken before the first token. */
    if (e->prefix_cache && e->fed_valid && e->fed_n > np &&
        qe_forward_pos(e->fw) == e->fed_n)
        snap_us += turns_store(e, e->fed_n, logits, vocab, 1);

    if (reuse) {
        e->turn_saved += reuse;
        qe_ui_cached(reuse);
        if (rewound)
            qe_ui_rewind(reuse);
    }
    qe_ui_ingested(np - reuse);
    {
        long held = 0;
        long long bytes = 0;
        turns_stats(e, &held, &bytes);
        qe_ui_cache(e->n_turns, held, bytes, snap_us);
    }
    qe_ui_gen_end();

    rc = emitted;
    if (st) {
        st->tokens = emitted;
        /* Positions this call actually put through the network. The reused part of
         * the prompt was put through one earlier, and saying so here is what keeps
         * a reported tokens-per-second a measurement of this request. A draft
         * window puts a row through for every token it was asked about, including
         * the ones it threw away, so its count is the rows rather than the tokens -
         * which is exactly the number that says what speculation cost. */
        st->steps = sp ? (feed_to - reuse) + sp->fed_rows + (sp->open ? 1 : 0)
                       : (np - reuse) + emitted;
        st->cached = reuse;
        st->drafted = sp ? spec_st.drafted : 0;
        st->accepted = sp ? spec_st.accepted : 0;
        st->tok_per_step =
            sp && st->steps > 0 ? (double)emitted / (double)st->steps : 1.0;
        st->truncated = emitted >= mt;
        st->budget = mt;
        st->dropped = dropped;
        st->rewound = rewound;
        st->rewind_lcp = lcp;
        st->snapshot_us = snap_us;
        st->seconds = now() - t0;
        st->prefill_seconds = t_read - t0;
    }
done:
    /* Also on the way out of a failure: a panel left saying "ingesting" because a
     * request died is a panel that teaches people to ignore it. */
    qe_ui_gen_end();
    e->user_progress = NULL;
    e->user_progress_user = NULL;
    pthread_mutex_unlock(&e->lock);
    free(ids);
    free(logits);
    return rc;
}

/* --------------------------------------------------------------- generation */

/* ------------------------------------------------------------ batched lane */

/* How much of one prompt the lane feeds before it lets the batch have another token.
 *
 * One slice is the whole of what a sibling stalls for: the driver is one thread, and while
 * it is walking a prompt nobody else's row moves. At the full chunk that stall is the
 * ingestion of a whole prompt - `make lane` measured 8.4 s for a 512-token one and 34.3 s
 * for a 2048-token one, which is a client watching one token appear every eight seconds -
 * so the lane feeds less than a chunk and buys the batch a heartbeat.
 *
 * What it pays is nothing measurable, because on this host a prompt row costs the same
 * whatever the chunk is cut into: 16.1 ms a token at 48 rows, 15.7 at 512, 16.8 at 2048
 * (`make lane`, and the bar's 61.3-62.9 tok/s is the 512-row figure again - 15.9 to
 * 16.3 ms). A 128-row slice therefore costs the same per token as a 512-row one and bounds
 * a sibling's wait at about two seconds. Feeding more than a chunk is not possible, a chunk
 * being the widest the pass runs. */
#define QE_LANE_SLICE 128

static long lane_slice(const qe_engine *e) {
    const int w = qe_forward_get_chunk(e->fw);
    const long cap = (w < 1 || w > QE_CHUNK_MAX) ? QE_CHUNK_MAX : (long)w;
    return cap < QE_LANE_SLICE ? cap : QE_LANE_SLICE;
}

/* The arithmetic a multi-row step runs, kept next to the numbers that chose it. A step
 * with more than one row goes through the same chunk walks a wide prompt does, and those
 * read the int8 and IQ4 copies made at load time. `make batch` raced the two at eight
 * rows: the copies are worth 1.11x of the step - 38.9 against 43.34 ms a token - and cost
 * 0.604 nats where the shared weight read alone costs 0.060 (ideas.md 7.25,
 * /tmp/batch3.log). Eleven percent of a step is not worth half a nat, so a multi-row step
 * declines the copies. A step of one row is left exactly as the serial engine runs it,
 * which is what makes a lone request on a batched engine the same bits it gets today, and
 * a prompt being ingested keeps them and the bar they buy.
 *
 * The walk of the projections and the bucketed expert pass are not touched: they are what
 * makes B rows cost one weight read instead of B, and their reassociation across rows is
 * the 0.060 nats that batching costs. */
typedef struct {
    size_t walk;
    int i8, flat, grouped, i4, head, head_win;
} pins;

static void pins_capture(pins *p, const qe_forward *f) {
    p->walk = qe_forward_get_proj_chunk(f);
    p->i8 = qe_forward_get_proj_i8(f);
    p->flat = qe_forward_get_flat_i8(f);
    p->grouped = qe_forward_get_moe_grouped(f);
    p->i4 = qe_forward_get_decode_i4(f);
    p->head = qe_forward_get_head_copy(f);
    p->head_win = qe_forward_get_head_window(f);
}

static void pins_apply(qe_forward *f, const pins *p) {
    qe_forward_set_proj_chunk(f, p->walk);
    qe_forward_set_proj_i8(f, p->i8);
    qe_forward_set_flat_i8(f, p->flat);
    qe_forward_set_moe_grouped(f, p->grouped);
    qe_forward_set_decode_i4(f, p->i4);
    qe_forward_set_head_copy(f, p->head);
    qe_forward_set_head_window(f, p->head_win);
}

static void pins_decline_copies(qe_forward *f) {
    pins p;
    pins_capture(&p, f);
    p.i8 = 0;
    p.flat = 0;
    p.i4 = 0;
    p.head = QE_HEAD_STORED;
    p.head_win = 0;
    pins_apply(f, &p);
}

/* Nothing left for this lane to do: it answered, the client went away, its budget ran
 * out, or the network produced a step that failed. */
static int lane_over(const lane *L) {
    return L->err || L->stop || L->emitted >= L->mt;
}

/* Hand the sequence back. The reset is what makes a batch a set of independent
 * sequences and not one sequence with a wider name: the next conversation handed this
 * slot starts at position 0 of a zeroed slice, and a delta-net state left over from the
 * last one answers its first token. It happens here, on the driver, because that is the
 * thread allowed to touch the pass; arming a lane on a request thread does not. */
static void lane_retire(qe_engine *e, int i) {
    lane *L = &e->lanes[i];
    lane_wait *w = L->w;
    const long rc = L->err ? -1 : L->emitted;
    const int err = L->err;
    if (L->st) {
        qe_gen_stats *st = L->st;
        st->tokens = L->emitted;
        /* Prompt tokens fed plus tokens chosen, which is the serial lane's count with
         * nothing reused: the token that ended a turn was chosen and never fed, and a
         * batched step is still one token of network for the one sequence it advanced.
         * What a batch shares is invisible per request by design - the aggregate is what
         * `make lane` prints, because that is the only place a shared step exists. */
        st->steps = L->fed + L->emitted;
        st->cached = 0;
        st->drafted = 0;
        st->accepted = 0;
        st->tok_per_step = 1.0;
        st->truncated = L->emitted >= L->mt;
        st->budget = L->mt;
        st->dropped = L->dropped;
        st->rewound = 0;
        st->rewind_lcp = 0;
        st->snapshot_us = 0;
        st->seconds = now() - L->t0;
    }
    qe_forward_reset_seq(e->fw, i);
    free(L->ids);
    memset(L, 0, sizeof *L);
    e->lane_live--;
    qe_ui_gen_end();
    if (w) {
        w->rc = rc;
        w->err = err;
        w->done = 1; /* last, and under the lock the waiter sleeps on */
    }
    pthread_cond_broadcast(&e->lane_cv);
}

/* One token of one lane's answer, sampled from the row the last walk left for it. The
 * caller's callback runs on the driver's thread, which is where the serial lane runs it
 * too: a sink writes one connection's bytes and only ever one thread is doing that. */
static void lane_say(qe_engine *e, lane *L, const float *lg) {
    const long vocab = qe_forward_vocab(e->fw);
    int id = qe_sample(lg, vocab, &L->so, &L->rng);
    if (id < 0) {
        L->stop = 1; /* the sampler had nothing to choose from */
        return;
    }
    char piece[128];
    long n = qe_tok_decode(e->tok, &id, 1, piece, sizeof piece);
    int special = qe_tok_is_special(e->tok, id);
    qe_token_out out = { .id = id, .special = special };
    out.utf8 = (!special && n > 0) ? piece : "";
    out.nbytes = (!special && n > 0) ? (int)n : 0;
    if (L->cb(L->user, &out))
        L->stop = 1; /* the client stopped reading; the rest of its answer is nobody's */
    else if (L->eos >= 0 && id == L->eos)
        L->stop = 1;
    else {
        L->emitted++;
        qe_ui_emit();
        L->pending = id;
        L->have_logits = 0;
    }
}

/* Feed one slice of a lane's prompt. Called with the lock held, by the driver only.
 * The ingestion hook is per engine and per request, so it is this lane's for the length
 * of one slice - which is the whole reason a batched request hands its keepalive in
 * qe_gen_opts instead: there is one hook, and there are B prompts. */
static void lane_ingest(qe_engine *e, int i) {
    lane *L = &e->lanes[i];
    const long vocab = qe_forward_vocab(e->fw);
    long n = L->n_ids - L->fed;
    const long cap = lane_slice(e);
    if (n > cap)
        n = cap;
    e->ingest_base = L->fed;
    e->ingest_total = L->n_ids;
    e->user_progress = L->pcb;
    e->user_progress_user = L->pcb_user;
    const int rc = qe_forward_prefill_seq(e->fw, i, L->ids + L->fed, n,
                                         e->rows + (size_t)L->row * (size_t)vocab);
    e->user_progress = NULL;
    e->user_progress_user = NULL;
    if (rc) {
        L->err = EIO;
        return;
    }
    L->fed += n;
    if (L->fed >= L->n_ids) {
        L->admitted = 1;
        L->have_logits = 1;
    }
}

/* Drive the batch until no lane is left. Whichever request found the lane table empty
 * when it arrived is the driver, and the last one out turns the role off; there is never
 * a second one, because the role is taken under the same lock as the table.
 *
 * One iteration is: speak for every lane that has a token to speak, feed one slice of one
 * prompt, then walk one row per sequence that has a token waiting. That order is a
 * correctness fact and not a preference - a lane speaks in the iteration after it is
 * admitted, so the row a prompt answered out of is free by the time the next prompt is
 * fed into it. */
static void lane_drive(qe_engine *e) {
    const long vocab = qe_forward_vocab(e->fw);
    pins base;
    pins_capture(&base, e->fw);
    for (;;) {
        lane *ingest = NULL;
        lane *say[QE_MAX_SEQ];
        int nsay = 0;
        pthread_mutex_lock(&e->lock);
        for (int i = 0; i < e->n_batch; i++) {
            lane *L = &e->lanes[i];
            if (!L->used)
                continue;
            if (!L->admitted) {
                if (!ingest)
                    ingest = L;
                continue;
            }
            if (L->have_logits && nsay < QE_MAX_SEQ)
                say[nsay++] = L;
        }
        pthread_mutex_unlock(&e->lock);

        for (int k = 0; k < nsay; k++) {
            lane *L = say[k];
            lane_say(e, L, e->rows + (size_t)L->row * (size_t)vocab);
        }
        if (ingest) {
            const int i = (int)(ingest - e->lanes);
            pthread_mutex_lock(&e->lock);
            if (ingest->used && !ingest->admitted && !ingest->err)
                lane_ingest(e, i);
            pthread_mutex_unlock(&e->lock);
        }

        /* What the walk owes now: one row for every sequence holding a sampled token. */
        int tok[QE_MAX_SEQ], rseq[QE_MAX_SEQ];
        lane *rows[QE_MAX_SEQ];
        int nrow = 0, alive = 0;
        pthread_mutex_lock(&e->lock);
        for (int i = 0; i < e->n_batch; i++) {
            lane *L = &e->lanes[i];
            if (!L->used)
                continue;
            if (lane_over(L)) {
                lane_retire(e, i);
                continue;
            }
            alive = 1;
            if (L->admitted && !L->have_logits) {
                rseq[nrow] = i;
                tok[nrow] = L->pending;
                rows[nrow] = L;
                L->row = nrow;
                nrow++;
            }
        }
        if (!alive) {
            e->lane_driving = 0;
            pthread_cond_broadcast(&e->lane_cv);
            pthread_mutex_unlock(&e->lock);
            break;
        }
        pthread_mutex_unlock(&e->lock);
        if (!nrow)
            continue;

        if (nrow > 1)
            pins_decline_copies(e->fw);
        const int rc = qe_forward_step_batch_rows(e->fw, tok, rseq, nrow, e->rows);
        if (nrow > 1)
            pins_apply(e->fw, &base);

        pthread_mutex_lock(&e->lock);
        for (int k = 0; k < nrow; k++) {
            if (rc)
                rows[k]->err = EIO;
            else
                rows[k]->have_logits = 1;
        }
        pthread_mutex_unlock(&e->lock);
    }
    pins_apply(e->fw, &base);
}

/* One request on the batched lane: encode, claim a sequence, and either drive the batch
 * or wait for whoever is. Everything the serial lane decides before it touches the
 * network - the answer budget, what has to be thrown away to fit the window, the sampler,
 * the seed - is decided here the same way, because a request must not get a different
 * answer by arriving at a busy moment. */
static long gen_batched(qe_engine *e, const qe_prompt *p, const qe_gen_opts *o,
                        qe_token_cb cb, void *user, qe_gen_stats *st) {
    if (!e->tok) {
        errno = ENOSYS;
        return -1;
    }
    const long need = qe_tok_count(e->tok, p->text);
    int *ids = malloc(((size_t)need + 8) * sizeof *ids);
    if (!ids) {
        errno = ENOMEM;
        return -1;
    }
    long np = qe_tok_encode(e->tok, p->text, ids, need + 8);
    if (np <= 0) {
        free(ids);
        errno = EINVAL;
        return -1;
    }
    const long ctx = e->lane_ctx;
    long mt = o->max_tokens > 0 ? o->max_tokens : 64;
    if (ctx > 0 && mt > ctx - np - 1)
        mt = ctx - np - 1;
    if (mt < 0)
        mt = 0;
    int dropped = 0;
    if (np + mt + 1 > ctx) {
        long keep = (long)ctx - mt - 1;
        if (keep <= 0) {
            free(ids);
            errno = ENOSPC;
            return -1;
        }
        memmove(ids, ids + (np - keep), (size_t)keep * sizeof *ids);
        np = keep;
        dropped = 1;
    }

    lane_wait w = { 0, 0, 0 };
    pthread_mutex_lock(&e->lock);
    lane *L = NULL;
    while (!L) {
        for (int i = 0; i < e->n_batch; i++)
            if (!e->lanes[i].used) {
                L = &e->lanes[i];
                break;
            }
        if (!L)
            pthread_cond_wait(&e->lane_cv, &e->lock);
    }
    L->used = 1;
    L->ids = ids;
    L->n_ids = np;
    L->mt = mt;
    L->eos = o->eos >= 0 ? o->eos : qe_tok_eos(e->tok);
    L->dropped = dropped;
    L->so = samp_for(e, o);
    L->rng = o->seed ? (uint64_t)o->seed
                     : ((uint64_t)(now() * 1e9) ^ 0x853c49e6748fea9bull);
    L->cb = cb;
    L->user = user;
    L->st = st;
    L->pcb = o->progress;
    L->pcb_user = o->progress_user;
    L->w = &w;
    L->t0 = now();
    L->row = e->n_batch; /* the row a prompt answers out of */
    e->lane_live++;
    qe_ui_gen_begin(np, 0);
    if (!e->lane_driving) {
        e->lane_driving = 1;
        pthread_mutex_unlock(&e->lock);
        lane_drive(e); /* returns with this lane retired and `w` written */
    } else {
        while (!w.done)
            pthread_cond_wait(&e->lane_cv, &e->lock);
        pthread_mutex_unlock(&e->lock);
    }
    if (w.err) {
        errno = w.err;
        return -1;
    }
    return w.rc;
}

int qe_engine_batch_active(qe_engine *e) {
    if (!e)
        return 0;
    pthread_mutex_lock(&e->lock);
    int n = e->lane_live;
    pthread_mutex_unlock(&e->lock);
    return n > 0 ? n : 0;
}

void qe_engine_turn_stats(qe_engine *e, long *slots, long *held, long long *bytes,
                          long long *cached_tokens) {
    long h = 0;
    long long b = 0;
    long s = 0;
    long long saved = 0;
    if (e) {
        pthread_mutex_lock(&e->lock);
        turns_stats(e, &h, &b);
        s = e->n_turns;
        saved = e->turn_saved;
        pthread_mutex_unlock(&e->lock);
    }
    if (slots)
        *slots = s;
    if (held)
        *held = h;
    if (bytes)
        *bytes = b;
    if (cached_tokens)
        *cached_tokens = saved;
}

size_t qe_engine_proj_chunk(qe_engine *e) {
    if (!e || !e->fw)
        return 0;
    return qe_forward_get_proj_chunk(e->fw);
}

/* Turn the transcript reuse off, or back on, and report what it was. It exists
 * for the one measurement that matters to a cache like this: the same two turns
 * with reuse on and off have to answer identically, and anything that is only
 * approximately the same answer is a wrong answer that happens to be fast. */
int qe_engine_prefix_cache(qe_engine *e, int on) {
    if (!e)
        return 0;
    pthread_mutex_lock(&e->lock);
    /* A batched lane has no transcript to reuse and no rewind point to reuse it from,
     * so there is nothing here to switch - and "off" means "throw away what the pass is
     * holding", which on a lane is everybody else's answer standing in it. Say what it
     * was and touch nothing. */
    if (e->n_batch > 1) {
        const int was = e->prefix_cache;
        pthread_mutex_unlock(&e->lock);
        return was;
    }
    int was = e->prefix_cache;
    e->prefix_cache = on ? 1 : 0;
    if (!on) {
        /* Off means off from now on, not off for this request only: what the
         * forward pass is holding is no longer something to trust, and neither is
         * anything copied out of it. */
        fed_reset(e);
        turns_discard(e);
        if (e->fw)
            qe_forward_reset(e->fw);
    }
    pthread_mutex_unlock(&e->lock);
    return was;
}

/* Fixed-size arena for the corpus strings: one allocation, one free, and the
 * corpus pointers cannot outlive their storage. */
struct arena {
    char *p, *end;
};

static const char *arena_fmt(struct arena *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(a->p, (size_t)(a->end - a->p), fmt, ap);
    va_end(ap);
    if (n < 0 || a->p + n >= a->end)
        return NULL;
    char *out = a->p;
    a->p += n + 1;
    return out;
}

/* Text only, control tokens dropped: what a completion endpoint wants, expressed
 * on top of the stream that keeps them. */
typedef struct {
    qe_text_cb cb;
    void *user;
} text_only;

static int text_only_forward(void *vp, const qe_token_out *t) {
    text_only *a = vp;
    if (t->special || t->nbytes <= 0)
        return 0;
    return a->cb(a->user, t->utf8, t->nbytes);
}

long qe_engine_generate(qe_engine *e, const qe_prompt *p, const qe_gen_opts *o, qe_text_cb cb,
                        void *user, qe_gen_stats *st) {
    text_only a = { cb, user };
    return qe_engine_generate_tokens(e, p, o, text_only_forward, &a, st);
}

long qe_engine_generate_tokens(qe_engine *e, const qe_prompt *p, const qe_gen_opts *o,
                               qe_token_cb cb, void *user, qe_gen_stats *st) {
    if (!e || !p || !o || !cb)
        return -1;
    if (!e->info.can_generate) {
        /* serve.c checks qe_engine_info first and answers 501 with info.why;
         * if a caller skipped that, this is the same refusal. */
        errno = ENOSYS;
        return -1;
    }
    if (e->fw) {
        if (e->n_batch > 1)
            return gen_batched(e, p, o, cb, user, st);
        return gen_checkpoint(e, p, o, cb, user, st);
    }

    static const char *FILLER[] = { "quantized", "streaming", "verified",  "bandwidth",
                                    "resident",  "speculative", "accepted", "drafted",
                                    "tokens",    "per",         "step",     "gated",
                                    "expert",    "routed",      "kernel",   "vector",
                                    "warming",   "pages",       "linear",   "sparse" };
    const int nfill = (int)(sizeof FILLER / sizeof *FILLER);
    const int nhead = 6;
    const int ncorpus = 1 + nhead + nfill; /* +1: id 0 is eos */

    long rc = -1;
    qe_spec_stats ss = { 0, 0, 0, 0, 0, 0 };
    const char **corpus = calloc((size_t)ncorpus, sizeof *corpus);
    size_t alen = 4096 + strlen(p->text);
    char *arena = malloc(alen);
    struct arena a = { 0 };
    if (!corpus || !arena)
        goto done;
    a.p = arena;
    a.end = arena + alen;

    pthread_mutex_lock(&e->lock);
    double t0 = now();

    corpus[ST_EOS] = "";
    corpus[1] = arena_fmt(&a, "[selftest]");
    corpus[2] = arena_fmt(&a, "model=%s", e->info.id);
    corpus[3] = arena_fmt(&a, "arch=%s", e->info.arch);
    corpus[4] = arena_fmt(&a, "prompt_chars=%zu", strlen(p->text));
    corpus[5] = arena_fmt(&a, "images=%d", p->n_images);
    corpus[6] = arena_fmt(&a, "head=%.100s", p->text);
    for (int i = 0; i < nfill; i++)
        corpus[7 + i] = FILLER[i];
    for (int i = 0; i < ncorpus; i++)
        if (!corpus[i])
            goto unlocked;

    st_ctx c = {
        .corpus = corpus,
        .n_corpus = ncorpus,
        .cb = cb,
        .user = user,
        .rng = (uint64_t)(o->seed ? o->seed : 0x853c49e6748fea9bull) | 1,
    };
    qe_spec_model m = { .ctx = &c, .vocab = ncorpus, .verify = st_verify, .draft = st_draft,
                        .emit = st_emit };
    int prompt_id = 1;
    long mt = o->max_tokens > 0 ? o->max_tokens : 64;
    /* No window to trim to. The selftest engine allocates no sequence, so
     * qe_forward_ctx() answers 0 here and the ceiling is the one the HTTP layer
     * applied - which is exactly what a client should be able to see, and is, in
     * the reported budget. A real checkpoint has a window and takes the rule
     * above. */
    long win = qe_forward_ctx(e->fw);
    if (win > 0 && mt > win - 1)
        mt = win - 1;
    qe_spec_opts so = {
        .k = o->spec_k > 0 ? o->spec_k : 4,
        .temp = o->temp,
        .seed = o->seed,
        .max_tokens = mt,
        .eos = ST_EOS,
    };
    /* This generator has no network to read a prompt through, so the panel is told
     * the same characters/4 the HTTP layer reports as prompt_tokens rather than a
     * count it cannot have. It exists so that a panel drawn against the selftest
     * engine is the panel a checkpoint draws, counters and all. */
    const long est = (long)((strlen(p->text) + 3) / 4);
    qe_ui_gen_begin(est, 0);
    rc = qe_spec_generate(&m, &so, &prompt_id, 1, &ss);
    /* Read in one step: this generator has no chunks to report progress between,
     * and a bar left at zero under a finished count reads as a stall. */
    qe_ui_ingest(est, est);
    qe_ui_ingested(est);
    if (st) {
        st->tokens = rc > 0 ? rc : 0;
        st->steps = ss.steps;
        st->drafted = ss.drafted;
        st->accepted = ss.accepted;
        st->tok_per_step = ss.tok_per_step;
        /* The corpus ends with the target's own eos, so a run that stops short
         * of max_tokens stopped because the model chose to. */
        st->truncated = !c.cb_failed && st->tokens >= mt;
        st->budget = mt;
        st->seconds = now() - t0;
        /* Nothing here was ever fed to anything: the selftest engine has no
         * sequence to hold, so it has no reuse to report and no front of a prompt
         * to throw away either. */
        st->cached = 0;
        st->dropped = 0;
        st->rewound = 0;
        st->rewind_lcp = 0;
        st->snapshot_us = 0;
    }
unlocked:
    qe_ui_gen_end(); /* a frame left mid-request is a lie about a finished one */
    pthread_mutex_unlock(&e->lock);

done:
    free(corpus);
    free(arena);
    return rc;
}
