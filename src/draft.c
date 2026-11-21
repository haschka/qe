/* A drafter that reads no weights.

   The arithmetic that makes speculation worth anything is in docs/DESIGN.md 4: a
   decode step is bound by walking the weights, so a step that emits three tokens
   costs about as much as one that emits one. What it needs in return is a proposal
   the target agrees with often enough, and the cheapest source of one is the
   sequence itself - agents repeat themselves for a living, and a coding client
   spends its turns re-sending the file it is editing. This file is that source: an
   index of the n-grams already seen, and a draft taken from wherever the longest
   tail of the sequence last ended.

   It is a hint and nothing else. Every token it proposes is checked by the target
   before it is kept (src/mtp.c), so a wrong draft costs a verification and changes
   no output; at temp 0 an over-eager drafter is indistinguishable from a slow one.
   That is the property this file is written to: it can be wrong as often as it
   likes, and the only thing it can lose is time.

   Three decisions are worth stating, because each is a way this was wrong in
   draft form before it was measured:

     - The index holds every suffix length from `n_min` to `n_max` for a position,
       not only the longest. A longest-only index can be asked about exactly one
       length, and a sequence of short repeats - a shell prompt, a JSON list - only
       ever matches short ones. The cost is bounded by n_max-n_min+1 inserts per
       token, paid once per token, off the ~90 ms a step takes.
     - A key is remembered at two positions, not one. The tail being matched is
       itself in the index, so an index that keeps only the most recent occurrence
       answers every lookup with the tail and matches nothing; the position that
       matters is the most recent one with a token after it.
     - A key is a hash and therefore a possible lie. Candidate positions are
       compared token by token before anything is drafted from them, so a collision
       costs a missed draft rather than a token that was never in the sequence.
     - Not every match is worth proposing. The shortest tails are the ones a chat
       template supplies twice - a role marker and the newline after it - and what
       they predict is the middle of somebody's prompt, so a two-token match is a
       verification bought in order to be told no. A window of k+1 rows costs about
       twice a single step at k=4, which means a wrong draft has to be wrong less
       than half the time to pay for itself; `QE_DRAFT_MIN` says where that line is. */
#include "qe.h"

#include <stdlib.h>
#include <string.h>

#define DRAFT_N_DEFAULT 16
#define DRAFT_N_MIN 2
#define DRAFT_MIN_DEFAULT 3

/* The shortest tail worth proposing. Read once, like every other switch here, and
 * generous rather than clever: at the default it refuses only the matches that are
 * shorter than a repeated word, and a sequence with nothing but two-token repeats
 * (QE_DRAFT_MIN=2) still drafts them. */
static int draft_min_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_DRAFT_MIN");
        cached = e ? atoi(e) : DRAFT_MIN_DEFAULT;
    }
    return cached < 1 ? 1 : cached;
}

typedef struct {
    /* Two candidate positions per key: the newest occurrence, and the one it
     * displaced. Slots, not pointers, because a lookup is one probe. */
    uint64_t key[2];
    uint32_t pos[2];  /* index of the n-gram's last token */
    uint32_t prev[2]; /* the occurrence this one pushed out, or PREV_NONE */
    uint32_t used[2]; /* 0 is the empty marker, so a slot says so separately */
} draft_bucket;

#define PREV_NONE 0xffffffffu

struct qe_draft {
    int n_max, n_min;
    int draft_min; /* the shortest tail worth proposing, which is not the shortest
                    * one worth indexing */

    int *seq;       /* the sequence, exactly as it was fed */
    uint64_t *pref; /* pref[i] folds seq[0..i) together multiplicatively */
    uint64_t *pow_p;   /* pow_p[n] = M^n, which is what turns a prefix fold back
                        * into a window fold */
    long len, cap;
    long indexed;   /* how far the table has been brought up to date */

    draft_bucket *tab;
    long buckets, inserts;

    qe_draft_stats s;
};

/* The FNV prime. Odd, so multiplication by it is invertible mod 2^64, which is what
 * lets a window be recovered from two prefixes by subtraction. */
#define HASH_MUL 0x100000001b3ull

/* splitmix64: no dependencies, and it spreads token ids, which are mostly small
 * integers, instead of mixing them into themselves. */
static uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* The key of the n tokens ending at index i-1: pref[i] - pref[i-n] * M^n is a fold
 * of those tokens and nothing else, which is the whole property a lookup needs - a
 * repetition has to arrive at the same number as the copy of itself being written.
 * A chained hash (h(i) = mix(h(i-1) ^ token)) does not have it: the same n tokens at
 * two positions hash differently there, and an index built that way answers every
 * lookup with "never seen", which is what this drafter measured at first.
 *
 * The length goes into the key because two windows that differ only in how many
 * tokens they hold must not answer each other's questions. */
static uint64_t window_key(const uint64_t *pref, const uint64_t *pow_p, long i, int n) {
    const uint64_t h = pref[i] - pref[i - n] * pow_p[n];
    return mix64(h ^ (0x9e3779b97f4a7c15ull * (uint64_t)n));
}

qe_draft *qe_draft_open(int n_max, int n_min) {
    qe_draft *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->n_max = n_max > 0 ? n_max : DRAFT_N_DEFAULT;
    if (d->n_max > 64)
        d->n_max = 64;
    d->n_min = n_min > 0 ? n_min : DRAFT_N_MIN;
    if (d->n_min > d->n_max)
        d->n_min = d->n_max;
    d->draft_min = draft_min_env();
    if (d->draft_min > d->n_max)
        d->draft_min = d->n_max;
    d->cap = 4096;
    d->seq = malloc((size_t)d->cap * sizeof *d->seq);
    d->pref = malloc((size_t)d->cap * sizeof *d->pref);
    d->pow_p = malloc(((size_t)d->n_max + 1) * sizeof *d->pow_p);
    d->buckets = 1u << 16;
    d->tab = calloc((size_t)d->buckets, sizeof *d->tab);
    if (!d->seq || !d->pref || !d->pow_p || !d->tab) {
        qe_draft_close(d);
        return NULL;
    }
    d->pow_p[0] = 1;
    for (int n = 1; n <= d->n_max; n++)
        d->pow_p[n] = d->pow_p[n - 1] * HASH_MUL;
    d->pref[0] = 0x243f6a8885a308d3ull;
    return d;
}

void qe_draft_close(qe_draft *d) {
    if (!d)
        return;
    free(d->seq);
    free(d->pref);
    free(d->pow_p);
    free(d->tab);
    free(d);
}

void qe_draft_reset(qe_draft *d) {
    if (!d)
        return;
    d->len = 0;
    d->indexed = 0;
    d->inserts = 0;
    memset(d->tab, 0, (size_t)d->buckets * sizeof *d->tab);
    d->pref[0] = 0x243f6a8885a308d3ull;
}

static void bucket_put(draft_bucket *b, uint64_t key, uint32_t pos) {
    for (int s = 0; s < 2; s++) {
        if (b->used[s] && b->key[s] == key) {
            if (b->pos[s] != pos)
                b->prev[s] = b->pos[s];
            b->pos[s] = pos;
            return;
        }
    }
    for (int s = 0; s < 2; s++) {
        if (!b->used[s]) {
            b->used[s] = 1;
            b->key[s] = key;
            b->pos[s] = pos;
            b->prev[s] = PREV_NONE;
            return;
        }
    }
    /* Both slots hold other keys: replace whichever candidate is oldest, which is
     * the one least likely to be part of the repetition being written now. */
    int old = 0;
    for (int s = 1; s < 2; s++)
        if (b->pos[s] < b->pos[old])
            old = s;
    b->key[old] = key;
    b->pos[old] = pos;
    b->prev[old] = PREV_NONE;
}

/* The newest occurrence of `key` that ends before `before`, so that the caller has
 * a token after it to draft. Without the `before` test the answer to every
 * question is the tail itself. */
static long bucket_get(const draft_bucket *b, uint64_t key, long before) {
    for (int s = 0; s < 2; s++) {
        if (!b->used[s] || b->key[s] != key)
            continue;
        if ((long)b->pos[s] < before)
            return b->pos[s];
        if (b->prev[s] != PREV_NONE && (long)b->prev[s] < before)
            return b->prev[s];
    }
    return -1;
}

/* Rebuilding is cheaper than degrading: the table is a hint, so an evicted key
 * costs one missed draft, while a table at capacity costs every lookup that has to
 * settle for a full bucket - and those are the lookups that were going to hit.
 * `seq` is the reason a rebuild is possible at all. */
static int tab_grow(qe_draft *d) {
    const long want = d->buckets * 2;
    draft_bucket *tab = calloc((size_t)want, sizeof *tab);
    if (!tab)
        return -1;
    for (int n = d->n_min; n <= d->n_max; n++)
        for (long e = n; e <= d->len; e++) {
            const uint64_t k = window_key(d->pref, d->pow_p, e, n);
            bucket_put(&tab[mix64(k) & (uint64_t)(want - 1)], k, (uint32_t)(e - 1));
        }
    free(d->tab);
    d->tab = tab;
    d->buckets = want;
    return 0;
}

/* Everything that ends on token `i`, from the shortest length indexed to the
 * longest. A slot is one probe, so the cost of an index that answers for every
 * tail length is n_max-n_min+1 inserts a token - against the ~17 ms a step is. */
static int index_one(qe_draft *d, long i) {
    for (int n = d->n_min; n <= d->n_max && n <= (int)i + 1; n++) {
        const uint64_t k = window_key(d->pref, d->pow_p, i + 1, n);
        bucket_put(&d->tab[mix64(k) & (uint64_t)(d->buckets - 1)], k, (uint32_t)i);
        d->inserts++;
    }
    return 0;
}

static int seq_grow(qe_draft *d, long need) {
    if (need <= d->cap)
        return 0;
    long cap = d->cap;
    while (cap < need)
        cap *= 2;
    uint64_t *pref = realloc(d->pref, (size_t)cap * sizeof *pref);
    int *seq = realloc(d->seq, (size_t)cap * sizeof *seq);
    if (!pref)
        return -1;
    d->pref = pref;
    if (!seq)
        return -1;
    d->seq = seq;
    d->cap = cap;
    return 0;
}

/* Bring the table up to date with the sequence. Tokens arrive one at a time from
 * a generating loop and in whole runs from a client re-sending a transcript, and
 * both come through here, which is the only reason the index cannot drift from
 * the sequence it claims to index. */
static int sync(qe_draft *d) {
    while (d->indexed < d->len) {
        const long i = d->indexed;
        d->pref[i + 1] = d->pref[i] * HASH_MUL + (uint64_t)(uint32_t)d->seq[i] + 1;
        if (index_one(d, i))
            return -1;
        d->indexed++;
        if (d->inserts * 10 > d->buckets * 7 && tab_grow(d))
            return -1;
    }
    return 0;
}

int qe_draft_push(qe_draft *d, int id) {
    if (!d || seq_grow(d, d->len + 1))
        return -1;
    d->seq[d->len++] = id;
    return 0;
}

long qe_draft_feed(qe_draft *d, const int *ids, long n) {
    if (!d || !ids || n < 0)
        return -1;
    if (seq_grow(d, d->len + n))
        return -1;
    memcpy(d->seq + d->len, ids, (size_t)n * sizeof *ids);
    d->len += n;
    if (sync(d))
        return -1;
    return d->len;
}

/* Where the longest tail of the sequence last ended, verified token by token. A
 * tail that has never been seen leaves `len_out` at 0. */
static long longest_match(qe_draft *d, int *len_out) {
    *len_out = 0;
    const long i = d->len;
    /* The search stops where proposing stops, so a tail shorter than that is not
     * looked up at all: the table still holds those lengths, because a long repeat
     * is found by its long suffix and the short entries cost two inserts a token. */
    const int floor = d->draft_min > d->n_min ? d->draft_min : d->n_min;
    for (int n = d->n_max; n >= floor; n--) {
        if (i < n)
            continue;
        const uint64_t k = window_key(d->pref, d->pow_p, i, n);
        /* `i - n` and not `i - 1`: the tail being matched is in the index, and an
         * answer that overlaps it drafts tokens that are already known. The
         * occurrence worth drafting from ends before the tail begins. */
        const long end = bucket_get(&d->tab[mix64(k) & (uint64_t)(d->buckets - 1)], k, i - n);
        if (end < 0)
            continue;
        /* The key is a hash. Up to n compares is nothing next to the verification
         * pass this draft is about to be given, and it is the difference between
         * missing a match and inventing a token. */
        if (memcmp(d->seq + (end - n + 1), d->seq + (i - n), (size_t)n * sizeof(int)))
            continue;
        *len_out = n;
        return end;
    }
    return -1;
}

int qe_draft_suggest(qe_draft *d, const int *toks, long n, int *out, int k) {
    if (!d || !toks || !out || k <= 0 || n < d->n_min)
        return 0;
    /* Self-synchronising, because the index is a function of the sequence: a
     * caller that hands it the whole sequence every step cannot make it disagree
     * with itself about how much of that sequence it knows. What it *can* do is hand
     * over a different whole sequence of the same length or a longer one, and two
     * requests on one server do exactly that - the index belongs to the target, so
     * it outlives the transcript it was built from, and a table that keeps indexing
     * somebody else's conversation answers lookups from it. So the shared prefix is
     * compared rather than the length: four bytes a token, once a step, against the
     * ~90 ms that step is, and it buys the rule that actually holds - a table is
     * reusable exactly while it is a prefix of what it is being asked about. */
    const long shared = n < d->len ? n : d->len;
    if (n < d->len || memcmp(d->seq, toks, (size_t)shared * sizeof *toks) != 0)
        qe_draft_reset(d);
    while (d->len < n) {
        if (qe_draft_push(d, toks[d->len]))
            return 0;
    }
    if (sync(d))
        return 0;

    d->s.lookups++;
    int m = 0;
    const long end = longest_match(d, &m);
    if (end < 0)
        return 0;
    d->s.matched++;
    d->s.match_len += m;
    if (m > d->s.longest)
        d->s.longest = m;

    /* What followed that occurrence is the draft, for as far as the sequence
     * actually goes. Past the end of what has been seen there is nothing to
     * propose, and making something up there only buys a rejection. */
    int nd = 0;
    for (long p = end + 1; p < d->len && nd < k; p++)
        out[nd++] = d->seq[p];
    d->s.drafted += nd;
    return nd;
}

void qe_draft_get_stats(const qe_draft *d, qe_draft_stats *s) {
    if (!d || !s)
        return;
    *s = d->s;
}

long qe_draft_length(const qe_draft *d) {
    return d ? d->len : 0;
}
