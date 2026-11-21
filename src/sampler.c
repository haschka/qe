/* Choosing the next token from a vector of 248,320 logits.

   Nothing about this file is bandwidth-bound, and that is exactly why it is
   kept separate: the decode step is 5568 MiB of weight reads, and a sampler
   that sorted the vocabulary would add a tenth of a step of pure CPU to every
   token. So the ordering work is bounded by k, not by n.

   The shape that falls out of that is one pass with a bounded min-heap of the
   current best k, then the nucleus inside those k. top-k is the cheaper filter
   to compute and the one the checkpoint ships a default for (top_k 20,
   top_p 0.95, temp 1.0 in general.sampling.*), so it runs first and reduces
   the vocabulary to a list that can be sorted by hand. That heap ranks logits
   rather than probabilities, which is exact because exp is monotone and is what
   keeps the vocabulary's exponentials out of the pass entirely (see qe_sample).
   min-p - drop candidates below a fraction of the best candidate's probability -
   runs between them, so
   the three filters apply in the order top-k, min-p, nucleus. Renormalising
   after the nucleus cut rather than before is what makes top_p mean "the
   smallest set whose total probability is p" instead of "the smallest set of
   the renormalised distribution", which is a different, sharper cut.

   A temperature of 0 is greedy and deterministic. That is not a convenience:
   it is the only way to run the same prompt twice and compare the two answers,
   which is how the forward pass gets checked at all.

   The generator is splitmix64 - 4 instructions, 64 bits of state, passes
   BigCrush at the level of scrutiny a token sampler needs. It is seeded from
   the request so a client that sends the same seed gets the same text.
*/
#include "qe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

void qe_sampler_opts_from_model(qe_model *m, qe_sampler_opts *o) {
    long long k;
    double d;
    o->top_k = 20;
    o->top_p = 0.95f;
    o->temp = 1.0f;
    o->min_p = 0.0f;
    o->seed = 0;
    if (!m)
        return;
    if (qe_model_get_int(m, "general.sampling.top_k", &k) == 0)
        o->top_k = (int)k;
    if (qe_model_get_float(m, "general.sampling.top_p", &d) == 0)
        o->top_p = (float)d;
    if (qe_model_get_float(m, "general.sampling.min_p", &d) == 0)
        o->min_p = (float)d;
    if (qe_model_get_float(m, "general.sampling.temp", &d) == 0)
        o->temp = (float)d;
}

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* Uniform in [0,1): the top 53 bits of a 64-bit draw. */
static double unit(uint64_t *s) {
    return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct {
    float v;
    int id;
} cand;

/* Min-heap of the best k seen so far: the root is the one a new candidate is
 * measured against, so a pass over the vocabulary costs one comparison per
 * entry and a swap per insertion. */
static void heap_push(cand *h, long n, cand c) {
    long i = n++;
    h[i] = c;
    while (i > 0) {
        long p = (i - 1) / 2;
        if (h[p].v <= h[i].v)
            break;
        cand t = h[p];
        h[p] = h[i];
        h[i] = t;
        i = p;
    }
}

static void heap_fix(cand *h, long n) {
    long i = 0;
    for (;;) {
        long l = 2 * i + 1, r = 2 * i + 2, best = i;
        if (l < n && h[l].v < h[best].v)
            best = l;
        if (r < n && h[r].v < h[best].v)
            best = r;
        if (best == i)
            break;
        cand t = h[best];
        h[best] = h[i];
        h[i] = t;
        i = best;
    }
}

static int cmp_desc(const void *a, const void *b) {
    const cand *x = a, *y = b;
    if (x->v < y->v)
        return 1;
    if (x->v > y->v)
        return -1;
    return x->id - y->id;
}

/* The two full-vocabulary scans, which at 248,320 entries are the whole cost of
 * a draw and are both pure data-parallel compares. They are written once as
 * vectors and once as the loop they replace, and the loop is the contract: a
 * vocabulary is a float array, so a scan that skipped an entry would change
 * which token comes back rather than how fast it comes back.
 *
 * The floor at -1e30 is part of that contract. It is what reports an
 * all-non-finite vocabulary as no answer at all, and a NaN must be neither the
 * maximum nor able to become one. */
#if defined(__x86_64__)

/* max_ps returns its second operand when either side is NaN, so with the running
 * maximum second a NaN lane is passed over - which is what `v[i] > m` does. */
static float vocab_max(const float *v, long n) {
    __m256 acc = _mm256_set1_ps(-1e30f);
    long i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_max_ps(_mm256_loadu_ps(v + i), acc);
    float t[8];
    _mm256_storeu_ps(t, acc);
    float m = -1e30f;
    for (int j = 0; j < 8; j++)
        if (t[j] > m)
            m = t[j];
    for (; i < n; i++)
        if (v[i] > m)
            m = v[i];
    return m;
}

/* Where the maximum first appears. Greedy has to answer with the earliest of
 * several equal maxima, because that is what a scan that only ever strict-greater
 * compares answers with. */
static long vocab_first(const float *v, long n, float mx) {
    __m256 w = _mm256_set1_ps(mx);
    long i = 0;
    for (; i + 8 <= n; i += 8) {
        unsigned m = (unsigned)_mm256_movemask_ps(
            _mm256_cmp_ps(_mm256_loadu_ps(v + i), w, _CMP_EQ_OQ));
        if (m)
            return i + __builtin_ctz(m);
    }
    for (; i < n; i++)
        if (v[i] == mx)
            return i;
    return -1;
}

#else

static float vocab_max(const float *v, long n) {
    float m = -1e30f;
    for (long i = 0; i < n; i++)
        if (v[i] > m)
            m = v[i];
    return m;
}

static long vocab_first(const float *v, long n, float mx) {
    for (long i = 0; i < n; i++)
        if (v[i] == mx)
            return i;
    return -1;
}

#endif

/* The scratch the candidate list and the probabilities live in. Generation is
 * serialised by the engine (one decode at a time, because it is bandwidth-bound
 * and the MoE scratch is shared), so there is one of these in the process, it
 * grows once to the vocabulary size, and it is never returned. */
static cand *g_cand;
static long g_cand_cap;

int qe_sample(const float *logits, long n, const qe_sampler_opts *o, uint64_t *rng) {
    if (!logits || n <= 0)
        return -1;
    qe_sampler_opts d;
    if (o)
        d = *o;
    else {
        d.temp = 1.0f;
        d.top_k = 0;
        d.top_p = 1.0f;
        d.min_p = 0.0f;
        d.seed = 0;
    }
    uint64_t state = rng ? *rng : (uint64_t)d.seed;
    if (!state)
        state = 0x243f6a8885a308d3ull;

    /* The largest logit: the greedy answer, and the shift every other entry is
     * measured against. Only the greedy path also needs its position, and for
     * that the cost is a second scan instead of a compare on every entry of
     * every sampled token. */
    float bestv = vocab_max(logits, n);
    if (!(bestv > -1e30f))
        return -1; /* every logit was non-finite, or at or below the floor */
    if (d.temp <= 0.0f) {
        long best = vocab_first(logits, n, bestv);
        if (rng)
            *rng = state;
        return (int)best;
    }

    long k = d.top_k > 0 ? d.top_k : n;
    if (k > n)
        k = n;
    if (g_cand_cap < k) {
        free(g_cand);
        g_cand = (cand *)malloc((size_t)k * sizeof *g_cand);
        g_cand_cap = g_cand ? k : 0;
        if (!g_cand)
            return -1;
    }
    /* Selection ranks logits, not probabilities. exp((l - mx)/T) is
     * non-decreasing in l, so the heap's smallest logit carries the heap's
     * smallest probability and the comparison that decides a replacement reads
     * the same value either way - which means the k that survive here are
     * exactly the k that survived when every entry was exponentiated first. The
     * exponential then belongs to the candidates that actually beat the root, a
     * few dozen of a 248,320-entry vocabulary, and to the k survivors below -
     * not to all 248,320, which was serial libm on the master thread every
     * token. The shift is what keeps the exponent of the largest entry at 0
     * rather than at 40, where it would saturate every other probability in the
     * vocabulary. */
    const float inv_t = 1.0f / d.temp;
    const float mx = bestv;
    long nsel = 0;
    float root_p = 0.0f; /* probability of the root, recomputed whenever the root moves */
    long at = 0;
    /* The heap starts as the first k admissible entries, in order, because that
     * is what a single pass over the vocabulary does. */
    for (; at < n && nsel < k; at++) {
        float l = logits[at];
        if (!(l > -1e30f))
            continue;
        heap_push(g_cand, nsel, (cand){ l, (int)at });
        nsel++;
        if (nsel == k)
            root_p = expf((g_cand[0].v - mx) * inv_t);
    }
    if (!nsel)
        return -1; /* nothing was admissible, so there is no distribution */
#if defined(__x86_64__)
    /* Once the heap is full the root only ever rises, so eight entries are
     * dismissed with one compare against it - and the floor test is subsumed,
     * because the root is above the floor by construction. A vocabulary is then
     * dismissed 31k times instead of 248k, and only a lane that beat the root is
     * looked at, in the order a scalar scan would have reached them. */
    if (nsel == k) {
        for (; at + 8 <= n; at += 8) {
            __m256 v = _mm256_loadu_ps(logits + at);
            unsigned m = (unsigned)_mm256_movemask_ps(
                _mm256_cmp_ps(v, _mm256_set1_ps(g_cand[0].v), _CMP_GT_OQ));
            while (m) {
                long j = at + __builtin_ctz(m);
                m &= (unsigned)((int)m - 1);
                float l = logits[j];
                if (!(l > g_cand[0].v))
                    continue; /* the root moved past this lane */
                if (!(expf((l - mx) * inv_t) > root_p))
                    continue;
                g_cand[0] = (cand){ l, (int)j };
                heap_fix(g_cand, nsel);
                root_p = expf((g_cand[0].v - mx) * inv_t);
            }
        }
    }
#endif
    /* Two compares for the rest of the vocabulary, and only a candidate that
     * already beat the root costs an expf. On a machine without the vectors
     * above, this is the whole scan. */
    for (; at < n; at++) {
        float l = logits[at];
        if (!(l > -1e30f))
            continue;
        if (!(l > g_cand[0].v))
            continue;
        if (!(expf((l - mx) * inv_t) > root_p))
            continue;
        g_cand[0] = (cand){ l, (int)at };
        heap_fix(g_cand, nsel);
        root_p = expf((g_cand[0].v - mx) * inv_t);
    }
    /* The heap is a heap of logits up to here; every filter after it is a filter
     * on probabilities, so this is where the exponentials are paid. */
    for (long i = 0; i < nsel; i++)
        g_cand[i].v = expf((g_cand[i].v - mx) * inv_t);
    if (rng)
        *rng = state;
    if (nsel > 1)
        qsort(g_cand, (size_t)nsel, sizeof *g_cand, cmp_desc);

    /* Minimum probability: drop everything whose chance is a small fraction of
     * the best candidate's. The list is descending and g_cand[0].v is exactly
     * 1 (the shift makes the argmax exp(0)), so this is a prefix cut and, like
     * the nucleus below, it cannot drop the argmax. Order matters: top-k, then
     * min-p, then nucleus, so that top_p still means "the smallest set whose
     * mass reaches p" *within* what survived the relative floor. */
    long keep = nsel;
    if (d.min_p > 0.0f) {
        const float floor_v = d.min_p * g_cand[0].v;
        while (keep > 1 && g_cand[keep - 1].v < floor_v)
            keep--;
    }

    /* Nucleus: keep the smallest prefix whose mass reaches top_p. Because the
     * list is already descending, the prefix is the set, and the cut cannot
     * drop the argmax. */
    if (d.top_p < 1.0f) {
        float sum = 0.0f, total = 0.0f;
        for (long i = 0; i < keep; i++)
            total += g_cand[i].v;
        if (total > 0.0f) {
            for (long i = 0; i < keep; i++) {
                sum += g_cand[i].v;
                if (sum >= d.top_p * total) {
                    keep = i + 1;
                    break;
                }
            }
        }
    }
    float sum = 0.0f;
    for (long i = 0; i < keep; i++)
        sum += g_cand[i].v;
    if (!(sum > 0.0f))
        return g_cand[0].id;
    double u = unit(rng ? rng : &state) * (double)sum;
    float acc = 0.0f;
    for (long i = 0; i < keep; i++) {
        acc += g_cand[i].v;
        if ((double)acc > u)
            return g_cand[i].id;
    }
    return g_cand[keep - 1].id;
}
