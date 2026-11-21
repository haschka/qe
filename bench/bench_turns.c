/* Which rewind point survives, tested without a checkpoint.

   The engine keeps a handful of sequence states so that a client which re-renders its
   own transcript can be put back onto a position it has already paid for instead of
   reading the conversation again. Two decisions decide whether that ever happens: how
   many points a budget buys, and which point a new one takes a slot from. Both are
   pure functions of a few integers, and until now both were asserted nowhere, because
   the only target that exercised them - `make prefix` - needs the checkpoint and costs
   minutes a run. That is how a cache worth four points at this checkpoint, two turns
   of history, shipped with a banner saying the cache was on: the third turn re-read
   everything, and nothing local could say so.

   So the policy lives in src/turns.c over an array of views of slots, and this is the
   replay it was written for: thirty turns of a coding-agent-shaped conversation - a
   system block, then an answer and a new block per turn, every prompt rendered from
   the start - run against caches of different sizes, with each turn's reuse checked
   against the position it should have been given. Then eviction on its own, then the
   front rewrite a re-rendered tool result is, then the arithmetic of the budget, which
   is where the bug actually was.

   Microseconds, and needs no checkpoint, no forward pass and no vocabulary. */
#include "turns.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void check(int ok, const char *what) {
    if (!ok) {
        fails++;
        printf("  FAIL: %s\n", what);
    }
}

/* One rewind point at the checkpoint in use: 112.6 MiB of recurrent state plus a
 * vocabulary of logits. The budget tests are stated against it because the bug being
 * regression-tested was a budget that was an exact multiple of it and still bought two
 * turns. */
#define POINT_MIB 116
#define FREE_MIB (121 * 1024) /* what this host has spare after the weights */
#define POINTS 12             /* the floor the engine asks for */

/* How a replay is judged. Every turn is expected to re-read at least the block it just
 * added; a cache with room for an answer-end point as well re-reads no more than that
 * plus the answer it got, and never goes back to token 0. COLLAPSE is what one point
 * does once the transcript is rewritten ahead of it. */
typedef struct {
    double saved;     /* percent of prompt tokens that never went through the network */
    long first_zero;  /* first turn that started again from token 0, -1 for none */
    long first_short; /* first turn that re-read more than answer and new block, -1 */
    long first_answer;/* first turn that re-read the answer it had just got, -1 */
    long refused;     /* stores the policy turned down */
    long shortest;    /* the shortest point still held at the end */
} result;

/* ---- a cache, without the engine ----------------------------------------- */

#define MAX_SLOTS 64

typedef struct {
    qe_turn_slot s[MAX_SLOTS]; /* what the policy is shown */
    uint32_t *ids[MAX_SLOTS];  /* the tokens those views point at, owned here */
    int n;
    uint64_t clock;
    long stored, refused;
} cache;

static void cache_open(cache *c, int n) {
    memset(c, 0, sizeof *c);
    c->n = n > MAX_SLOTS ? MAX_SLOTS : n;
}

static void cache_close(cache *c) {
    for (int i = 0; i < c->n; i++)
        free(c->ids[i]);
    memset(c, 0, sizeof *c);
}

/* A point covering the first `np` tokens of `tok`, stored the way turns_store stores
 * one: a length already held replaces itself, anything else takes the slot the policy
 * gives up. A point stored without its logits can never match anything, which the
 * policy is supposed to notice before it evicts something that can. */
static int cache_store(cache *c, const int *tok, long np, int answer, int logits) {
    int victim = -1;
    for (int i = 0; i < c->n; i++)
        if (c->s[i].pos == np) {
            victim = i;
            break;
        }
    if (victim < 0)
        victim = qe_turns_victim(c->s, c->n, answer);
    if (victim < 0) {
        c->refused++;
        return -1;
    }
    free(c->ids[victim]);
    uint32_t *v = malloc((size_t)np * sizeof *v);
    if (!v) {
        printf("out of memory\n");
        exit(2);
    }
    for (long i = 0; i < np; i++)
        v[i] = (uint32_t)tok[i];
    c->ids[victim] = v;
    c->s[victim].pos = np;
    c->s[victim].ids = v;
    c->s[victim].have_logits = logits;
    c->s[victim].answer = answer;
    c->s[victim].used = ++c->clock;
    c->stored++;
    return victim;
}

/* How many leading tokens of the prompt a held point reproduces, and how far the
 * prompt and the furthest point on hand agreed. Mirrors turns_match in engine.c,
 * including the stamp that takes the winner out of the eviction queue's future. */
static long cache_match(cache *c, const int *ids, long np, long *agreed) {
    long at = 0;
    *agreed = 0;
    const int best = qe_turns_match(c->s, c->n, ids, np, &at, agreed, &c->clock);
    if (best >= 0)
        c->s[best].used = c->clock;
    return best < 0 ? 0 : at;
}

/* ---- a conversation ------------------------------------------------------ */

/* Ids are position based, so two renderings of the same text are equal token for token
 * and a rendering that differs from some position on diverges from exactly that
 * position, which is the only property the matching rules care about. */
#define TOK_BASE 1000u
#define TOK_EDITED 500000u

#define TURNS 30
#define SYSTEM 900 /* the system block, read once and never again */

static long user_len(int t) { return 200 + 35 * (long)t; }
static long answer_len(int t) { return 40 + 7 * (long)t; }

typedef struct {
    int *tok;
    long n, cap;
} seq;

static void seq_open(seq *s, long cap) {
    s->tok = malloc((size_t)cap * sizeof *s->tok);
    if (!s->tok) {
        printf("out of memory\n");
        exit(2);
    }
    s->n = 0;
    s->cap = cap;
}

static void seq_close(seq *s) {
    free(s->tok);
    memset(s, 0, sizeof *s);
}

/* Render `k` tokens of the next block onto the end of the transcript. */
static long seq_append(seq *s, long k, unsigned base) {
    if (s->n + k > s->cap) {
        printf("transcript overflow\n");
        exit(2);
    }
    for (long i = 0; i < k; i++)
        s->tok[s->n + i] = (int)(base + (unsigned long)(s->n + i));
    s->n += k;
    return s->n;
}

/* What one cache of `slots` points does over `TURNS` turns, and how much of each turn
 * it had to read again. `edit` is the turn that re-renders the transcript from
 * `edit_at` on - a tool result the client is holding came back differently - or -1 for
 * a client that never rewrites anything. */
static void replay(int slots, int edit, long edit_at, result *out) {
    cache c;
    cache_open(&c, slots);
    seq s;
    seq_open(&s, 1 << 18);
    seq_append(&s, SYSTEM, TOK_BASE);

    long prompt_tokens = 0, reread = 0, plen_prev = 0;
    memset(out, 0, sizeof *out);
    out->first_zero = -1;
    out->first_short = -1;
    out->first_answer = -1;
    long shortest = 0;
    for (int t = 0; t < TURNS; t++) {
        const long before = s.n; /* where the previous turn's answer ended */
        if (t == edit) {
            s.n = edit_at;
            seq_append(&s, user_len(t), TOK_EDITED);
        } else {
            seq_append(&s, user_len(t), TOK_BASE);
        }
        const long np = s.n;

        long agreed = 0;
        const long at = cache_match(&c, s.tok, np, &agreed);
        prompt_tokens += np;
        reread += np - at;
        if (t) {
            /* The floor a policy this one can hold: the previous turn's prompt ended at
             * `plen_prev`, that point is a prefix of this prompt, and so anything short
             * of it is a turn reading a second time what the cache was for. */
            const long floor = plen_prev;
            if (at == 0 && out->first_zero < 0)
                out->first_zero = t;
            if (at < floor && out->first_short < 0)
                out->first_short = t;
            if (at < before && out->first_answer < 0)
                out->first_answer = t;
        }

        cache_store(&c, s.tok, np, 0, 1); /* the prompt's own end */
        const long ae = seq_append(&s, answer_len(t), TOK_BASE);
        cache_store(&c, s.tok, ae, 1, 1); /* and the answer's */
        plen_prev = np;

        shortest = 0;
        for (int i = 0; i < c.n; i++)
            if (c.s[i].pos && (!shortest || c.s[i].pos < shortest))
                shortest = c.s[i].pos;
    }

    out->saved = 100.0 * (double)(prompt_tokens - reread) / (double)prompt_tokens;
    out->refused = c.refused;
    out->shortest = shortest;
    printf("  %3d points: %5.1f%% of prompt tokens reused, %ld stores, %ld refused,"
           " re-read from 0 at turn %ld, re-read an answer from turn %ld\n",
           slots, out->saved, c.stored, c.refused, out->first_zero, out->first_answer);

    cache_close(&c);
    seq_close(&s);
}

/* ---- eviction, on its own ------------------------------------------------ */

/* Slot `i` holding a point of `pos` tokens, of kind `answer`, with or without its
 * logits, matched `i + 1` passes ago. */
static void held(cache *c, int i, long pos, int answer, int logits) {
    c->s[i].pos = pos;
    c->s[i].have_logits = logits;
    c->s[i].answer = answer;
    c->s[i].used = (uint64_t)(i + 1);
}

static void eviction(void) {
    printf("\neviction order\n");
    cache c;

    /* A point that cannot match anything goes before a slot that has never been
     * written, even though the empty slot is the one that costs nothing to keep. */
    cache_open(&c, 4);
    held(&c, 0, 5000, 0, 0);
    check(qe_turns_victim(c.s, c.n, 0) == 0, "a point without logits is given up first");
    cache_close(&c);

    cache_open(&c, 4);
    held(&c, 0, 5000, 0, 1);
    check(qe_turns_victim(c.s, c.n, 0) == 1, "then an empty slot");
    cache_close(&c);

    /* An answer-end point is only reachable by a client replaying that answer token for
     * token, so it loses to every prompt-end point rather than the other way round. */
    cache_open(&c, 3);
    held(&c, 0, 5000, 0, 1);
    held(&c, 1, 6000, 1, 1);
    held(&c, 2, 7000, 0, 1);
    check(qe_turns_victim(c.s, c.n, 0) == 1,
          "an answer-end point is given up before a prompt-end one");
    check(qe_turns_victim(c.s, c.n, 1) == 1, "and that holds for an answer-end store too");
    cache_close(&c);

    cache_open(&c, 2);
    held(&c, 0, 5000, 0, 1);
    held(&c, 1, 6000, 0, 1);
    check(qe_turns_victim(c.s, c.n, 1) == -1,
          "a cache full of prompt-end points refuses an answer-end store outright");
    cache_close(&c);

    /* The anchor. With nothing left but prompt-end points the shortest is the state
     * after the least conversation, and it is the one a client that rewrote something
     * earlier than all of them still comes back to. Losing it is what turns one edited
     * tool result into re-reading the whole session. */
    cache_open(&c, 3);
    held(&c, 0, 9000, 0, 1);
    held(&c, 1, 1000, 0, 1);
    held(&c, 2, 8000, 0, 1);
    const int v = qe_turns_victim(c.s, c.n, 0);
    check(v != 1, "the shortest point is never the victim");
    check(v == 0, "and among the rest it is the least recently matched that goes");
    cache_close(&c);

    cache_open(&c, 1);
    held(&c, 0, 4000, 0, 1);
    check(qe_turns_victim(c.s, c.n, 0) == 0, "one point held, and it is this one's turn to lose");
    cache_close(&c);

    /* Matching: token for token over the whole point, longest wins, and nothing longer
     * than the prompt is even a candidate. */
    printf("\nmatching\n");
    static uint32_t ids[200];
    for (int i = 0; i < 200; i++)
        ids[i] = TOK_BASE + (unsigned)i;
    int prompt[200];
    for (int i = 0; i < 200; i++)
        prompt[i] = (int)(TOK_BASE + (unsigned)i);
    long at = 0, agreed = 0;

    cache_open(&c, 3);
    c.s[0] = (qe_turn_slot){ 50, ids, 1, 0, 1 };
    c.s[1] = (qe_turn_slot){ 120, ids, 1, 0, 2 };
    c.s[2] = (qe_turn_slot){ 300, ids, 1, 0, 3 };
    const uint64_t stamp = c.clock;
    int best = qe_turns_match(c.s, c.n, prompt, 200, &at, &agreed, &c.clock);
    check(best == 1 && at == 120, "the longest point that fits the prompt is the one returned");
    check(c.s[1].used == stamp + 1 && c.clock == stamp + 1, "and a match stamps its winner");
    check(agreed == 120, "agreed never counts further than the end of a point on hand");
    cache_close(&c);

    cache_open(&c, 2);
    c.s[0] = (qe_turn_slot){ 120, ids, 0, 0, 1 };
    c.s[1] = (qe_turn_slot){ 60, ids, 1, 1, 2 };
    best = qe_turns_match(c.s, c.n, prompt, 200, &at, &agreed, &c.clock);
    check(best == 1 && at == 60, "a point whose logits are gone is never a match");
    cache_close(&c);

    /* A re-rendered tool result: the transcript is the same until token 90 and new from
     * there. The point ending before the edit is genuinely the state after those
     * tokens, and `agreed` is the number whoever is debugging this needs. */
    for (int i = 90; i < 200; i++)
        prompt[i] = (int)(TOK_EDITED + (unsigned)i);
    cache_open(&c, 2);
    c.s[0] = (qe_turn_slot){ 60, ids, 1, 0, 1 };
    c.s[1] = (qe_turn_slot){ 150, ids, 1, 0, 2 };
    best = qe_turns_match(c.s, c.n, prompt, 200, &at, &agreed, &c.clock);
    check(best == 0 && at == 60, "a point that ends before an edit still matches");
    check(agreed == 90, "and agreed says where the prompt and the transcript parted");
    cache_close(&c);

    cache_open(&c, 1);
    c.s[0] = (qe_turn_slot){ 150, ids, 1, 0, 1 };
    for (int i = 0; i < 200; i++)
        prompt[i] = (int)(TOK_EDITED + (unsigned)i);
    best = qe_turns_match(c.s, c.n, prompt, 200, &at, &agreed, &c.clock);
    check(best < 0 && at == 0 && agreed == 0,
          "a transcript rewritten from token 0 matches nothing, and says so");
    cache_close(&c);
}

/* ---- the budget ---------------------------------------------------------- */

static void budget(void) {
    printf("\nbudget\n");
    /* Rounded up, because a budget a hair short of a point buys one fewer than the
     * number printed next to it. */
    check(qe_turns_point_mib(118043648L) == 113, "a 112.6 MiB point is priced at 113 MiB");
    check(qe_turns_point_mib(0) == 0, "no state, no price");
    check(qe_turns_point_mib(-1) == 0, "a negative state is not a price either");

    /* The bug, as the number it was: this is the budget that shipped. A whole multiple
     * of a point, a banner saying the cache was on, two turns of history in it. */
    const long shipped = 512;
    check(shipped / POINT_MIB == 4, "the budget that shipped bought four points");
    check(shipped / POINT_MIB / 2 == 2, "and four points is two turns");

    /* What the default buys on a host like this one: the 8 GiB cap, which is 70 points
     * at this checkpoint rather than 4. */
    const long mib = qe_turns_auto_mib(FREE_MIB, POINT_MIB, POINTS, 0.20, 2, 8192);
    check(mib == 8192, "the default is the cap on a host with this much free");
    check(mib >= POINT_MIB * POINTS, "it buys at least the points it promises");
    check(mib / POINT_MIB >= POINTS, "the points printed are the points held");
    check(mib / POINT_MIB / 2 >= POINTS / 2, "which is six turns, not two");

    /* Half of what is free is a hard cap, and it can bite the floor. That is the case
     * the banner calls short, and it has to be said rather than silently held. */
    const long tight = qe_turns_auto_mib(2000, POINT_MIB, POINTS, 0.20, 2, 8192);
    check(tight == 1000, "a host with 2 GiB free is capped at half of it, below the floor");
    check(tight / POINT_MIB < POINTS, "and then holds fewer points than it asked for");
    check(tight / POINT_MIB >= 1, "but still holds a cache, and says it is short");

    /* A host that cannot afford one point gets no cache at all, which is the engine's
     * "afforded nothing" branch and not a zero-length cache. */
    check(qe_turns_auto_mib(100, POINT_MIB, POINTS, 0.20, 2, 8192) < POINT_MIB,
          "a host with 100 MiB spare is refused rather than pushed into reclaim");

    /* And with a kernel that will not say what is free, the floor is the whole answer. */
    check(qe_turns_auto_mib(0, POINT_MIB, POINTS, 0.20, 2, 8192) == POINT_MIB * POINTS,
          "no answer from /proc/meminfo is answered with the floor");
}

int main(void) {
    printf("turn cache: %d MiB a point, %d points asked\n", POINT_MIB, POINTS);
    result big, small, edited, alone;

    /* The same conversation at the two sizes that matter: the four points that shipped
     * and the twelve asked for now. Both hold their floor - a turn re-reads the block it
     * added and no more than that plus the answer it got - and neither goes back to
     * token 0. What the larger cache buys is the turns that go by before an answer has
     * to be read again, and the total that never goes through the network. */
    printf("\nreplay: a transcript rendered from the start every turn\n");
    replay(POINTS, -1, 0, &big);
    replay(4, -1, 0, &small);
    check(big.first_short < 0 && small.first_short < 0,
          "no turn of either replay re-read more than the answer it got and the block it added");
    check(big.first_zero < 0 && small.first_zero < 0,
          "neither replay ever went back to token 0");
    check(big.first_answer > small.first_answer,
          "twelve points go on re-using a turn's answer for longer than four do");
    check(big.saved > small.saved, "and the larger cache reads less of it a second time");

    /* One turn re-renders a tool result, so the transcript diverges from token 1500 on.
     * Every point past the edit is a different conversation from then on, and the only
     * thing left to rewind to is a point that ended before it. With twelve points the
     * shortest has been kept out of the eviction queue since the first turn and still
     * is; with one, the newest point is the only point, and the whole prompt is read
     * again from the first token. */
    printf("\nreplay: one tool result re-rendered at turn 10\n");
    replay(POINTS, 10, 1500, &edited);
    replay(1, 10, 1500, &alone);
    check(edited.first_zero < 0,
          "twelve points survive a rewrite of the transcript, and never read from token 0");
    check(edited.shortest > 0 && edited.shortest <= 1500,
          "the point the anchor rule protects is the one a rewritten front still comes back to");
    check(alone.first_zero == 10,
          "one point cannot, which is the whole argument for a floor in points");

    eviction();
    budget();

    printf("\n%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
