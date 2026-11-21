/* Sampler contract, tested.

   The sampler is the one part of the decode path whose output is not a number
   anyone can diff against a reference, so what gets pinned here is the shape of
   the distribution rather than its exact draws: which ids are reachable at all,
   which order the three filters run in, and that the top candidate survives all
   of them. Those are the properties a refactor breaks, and they are the ones a
   client cannot tell apart from correct output by reading the text.

   The vocabulary is six wide with probabilities 0.50, 0.30, 0.12, 0.05, 0.02,
   0.01, entered as log-probabilities at temperature 1 so the exp() inside the
   sampler reproduces them exactly and every expected share below is arithmetic
   rather than luck. Draws are counted over thousands of samples with a fixed
   seed, so a case that fails here fails identically on every host.

   Six candidates are enough for the filters and nowhere near enough for top-k,
   so the second half of the file works at 4096 and at the checkpoint's real
   248,320: what a bounded heap selects, whether it selects the same set the
   straightforward "exponentiate everything and rank" sampler did, and what the
   two cost. ref_top below is that straightforward sampler, kept as the contract
   and as the cost column.

   It needs no checkpoint and no vocabulary: qe_sample takes a float array. */
#include "qe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails;

static void check(int ok, const char *what) {
    if (!ok) {
        fails++;
        printf("  FAIL: %s\n", what);
    }
}

#define NVOC 6
static const float P[NVOC] = { 0.50f, 0.30f, 0.12f, 0.05f, 0.02f, 0.01f };

/* log-probabilities, so that exp(l - max) at temp 1 is the ratio of two
 * probabilities and the kept set is decided by the filters alone. */
static void logits(float *v) {
    for (int i = 0; i < NVOC; i++)
        v[i] = logf(P[i]);
}

/* Draw `n` times from one seeded stream and count what came back. */
static void draw(const float *v, const qe_sampler_opts *o, long n, long *counts) {
    uint64_t rng = 0x243f6a8885a308d3ull;
    memset(counts, 0, (size_t)NVOC * sizeof *counts);
    for (long i = 0; i < n; i++) {
        int id = qe_sample(v, NVOC, o, &rng);
        if (id < 0 || id >= NVOC) {
            counts[0] = -1;
            return;
        }
        counts[id]++;
    }
}

static int reachable(const long *counts, long n) {
    int r = 0;
    for (int i = 0; i < NVOC; i++)
        if (counts[i] > 0 && counts[i] < n)
            r |= 1 << i;
    return r;
}

static qe_sampler_opts opts(float temp, int top_k, float top_p, float min_p) {
    qe_sampler_opts o;
    memset(&o, 0, sizeof o);
    o.temp = temp;
    o.top_k = top_k;
    o.top_p = top_p;
    o.min_p = min_p;
    return o;
}



/* The selection this sampler made before it ranked logits: exponentiate every
 * entry, then keep the k largest probabilities. It stays here as the contract to
 * compare against, because "the k are the same set either way" is the whole
 * argument for ranking logits - and as the cost column of the timing line, since
 * 248,320 exponentials on the master thread is what that change removes. */
typedef struct {
    float v;
    int id;
} rcand;

static void rpush(rcand *h, long n, rcand c) {
    long i = n++;
    h[i] = c;
    while (i > 0) {
        long p = (i - 1) / 2;
        if (h[p].v <= h[i].v)
            break;
        rcand t = h[p];
        h[p] = h[i];
        h[i] = t;
        i = p;
    }
}

static void rfix(rcand *h, long n) {
    long i = 0;
    for (;;) {
        long l = 2 * i + 1, r = 2 * i + 2, best = i;
        if (l < n && h[l].v < h[best].v)
            best = l;
        if (r < n && h[r].v < h[best].v)
            best = r;
        if (best == i)
            break;
        rcand t = h[best];
        h[best] = h[i];
        h[i] = t;
        i = best;
    }
}

static long ref_top(const float *logits, long n, int k, float temp, rcand *out) {
    float mx = -1e30f;
    for (long i = 0; i < n; i++)
        if (logits[i] > mx)
            mx = logits[i];
    long nsel = 0;
    for (long i = 0; i < n; i++) {
        if (!(logits[i] > -1e30f))
            continue;
        float p = expf((logits[i] - mx) / temp);
        if (nsel < k) {
            rpush(out, nsel, (rcand){ p, (int)i });
            nsel++;
        } else if (p > out[0].v) {
            out[0] = (rcand){ p, (int)i };
            rfix(out, nsel);
        }
    }
    return nsel;
}

#define DRAWS 40000

int main(void) {
    float v[NVOC];
    long counts[NVOC];
    logits(v);
    printf("sampling: p = [");
    for (int i = 0; i < NVOC; i++)
        printf("%s%.2f", i ? ", " : "", P[i]);
    printf("]\n");

    /* ---- greedy: temperature 0 ignores everything else ---- */
    {
        qe_sampler_opts o = opts(0.0f, 0, 1.0f, 0.9f);
        long all = 0;
        for (int i = 0; i < 100; i++)
            all += qe_sample(v, NVOC, &o, NULL) == 0;
        check(all == 100, "temp 0 is greedy even with min_p 0.9 set");
    }

    /* ---- top_k 1 leaves one candidate ---- */
    {
        qe_sampler_opts o = opts(1.0f, 1, 1.0f, 0.0f);
        draw(v, &o, DRAWS, counts);
        check(counts[0] == DRAWS, "top_k 1 always returns the top candidate");
    }

    /* ---- no filters: every id is reachable, roughly at its own rate ---- */
    {
        qe_sampler_opts o = opts(1.0f, 0, 1.0f, 0.0f);
        draw(v, &o, DRAWS, counts);
        check(reachable(counts, DRAWS) == (1 << NVOC) - 1,
              "with no filters all six ids are reachable");
        double share = (double)counts[0] / DRAWS;
        check(share > 0.48 && share < 0.52, "and the top id keeps its 0.50 share");
    }

    /* ---- min_p is relative to the best candidate, not to 1 ---- */
    {
        qe_sampler_opts o = opts(1.0f, 0, 1.0f, 0.5f);
        draw(v, &o, DRAWS, counts);
        check(counts[2] == 0 && counts[3] == 0 && counts[4] == 0 && counts[5] == 0,
              "min_p 0.5 drops everything below half the top candidate");
        check(counts[0] > 0 && counts[1] > 0, "and keeps the two above it");
        double share = (double)counts[0] / (double)(counts[0] + counts[1]);
        check(share > 0.60 && share < 0.65, "the kept pair renormalises to 0.625/0.375");
    }

    /* ---- a floor above the second candidate cannot empty the set ---- */
    {
        qe_sampler_opts o = opts(1.0f, 0, 1.0f, 0.9f);
        draw(v, &o, DRAWS, counts);
        check(counts[0] == DRAWS, "min_p 0.9 collapses to the argmax, never to nothing");
    }

    /* ---- min_p 0 and a negative min_p are both off ---- */
    {
        qe_sampler_opts off = opts(1.0f, 0, 1.0f, 0.0f);
        qe_sampler_opts neg = opts(1.0f, 0, 1.0f, -1.0f);
        draw(v, &off, 4000, counts);
        long a = counts[5];
        draw(v, &neg, 4000, counts);
        check(a > 0 && counts[5] == a, "min_p 0 and min_p < 0 draw the same distribution");
    }

    /* ---- the order: top-k, then min-p, then the nucleus ----
     *
     * top_p 0.95 over the whole vocabulary cuts where the cumulative mass first
     * reaches 0.95 of 1.00, which is after 0.05: ids 0..3 are kept, 0.02 and
     * 0.01 are not. top_k 3 narrows the field to 0..2, and min_p 0.5 then removes
     * id 2 before the nucleus renormalises inside {0,1}. A sampler that ran the
     * nucleus before the floor would still show id 2 in the second case. */
    {
        qe_sampler_opts nucleus_only = opts(1.0f, 0, 0.95f, 0.0f);
        draw(v, &nucleus_only, DRAWS, counts);
        check(counts[2] > 0 && counts[3] > 0 && counts[4] == 0,
              "top_p 0.95 alone keeps ids 0 through 3");

        qe_sampler_opts both = opts(1.0f, 3, 0.95f, 0.5f);
        draw(v, &both, DRAWS, counts);
        check(counts[0] > 0 && counts[1] > 0, "top_k 3 with min_p 0.5 keeps 0 and 1");
        check(counts[2] == 0, "min_p runs before the nucleus, so 2 is gone");
    }

    /* ---- A vocabulary wide enough for top-k to be the whole selection.
     *
     * Six candidates never fill a k of 20, so the replacement path - the one a
     * real 248,320-entry vocabulary spends every token in - needs a vocabulary
     * of its own to be tested at all. Values are a bijection of a shallow ramp:
     * the best 20 are scattered through the array rather than sitting at its
     * front, and they differ from each other by 1e-6, which makes the expected
     * answer a set rather than a distribution - those 20 ids are reachable and
     * nothing else is. Multiplication by an odd constant modulo 2^12 is a
     * permutation, so the 4096 values are distinct and argmax is unambiguous.
     * Temperature scales the differences and cannot reorder them, so the same
     * set is the answer at 1, at 0.5 and at 2. */
    {
        const long WIDE = 4096;
        const int K = 20;
        float *w = (float *)malloc((size_t)WIDE * sizeof *w);
        char *expected = (char *)calloc((size_t)WIDE, 1);
        long *hits = (long *)calloc((size_t)WIDE, sizeof *hits);
        for (long i = 0; i < WIDE; i++)
            w[i] = -(float)(((unsigned)i * 2654435761u) & (unsigned)(WIDE - 1)) * 1e-6f;
        for (int r = 0; r < K; r++) {
            long bi = -1;
            for (long i = 0; i < WIDE; i++)
                if (!expected[i] && (bi < 0 || w[i] > w[bi]))
                    bi = i;
            expected[bi] = 1;
        }
        const float temps[3] = { 1.0f, 0.5f, 2.0f };
        for (int t = 0; t < 3; t++) {
            qe_sampler_opts o = opts(temps[t], K, 1.0f, 0.0f);
            memset(hits, 0, (size_t)WIDE * sizeof *hits);
            uint64_t rng = 0x243f6a8885a308d3ull + (uint64_t)t;
            for (long i = 0; i < 20000; i++) {
                int id = qe_sample(w, WIDE, &o, &rng);
                if (id >= 0 && id < WIDE)
                    hits[id]++;
            }
            long foreign = 0, missing = 0;
            for (long i = 0; i < WIDE; i++) {
                if (hits[i] && !expected[i])
                    foreign++;
                if (expected[i] && !hits[i])
                    missing++;
            }
            check(foreign == 0, "top_k over 4096 logits never returns a 21st id");
            check(missing == 0, "and reaches all 20 of them, at temp 1, 0.5 and 2");
        }
        free(w);
        free(expected);
        free(hits);
    }

    /* ---- The set is the one the old ranking produced, on shapes nobody chose.
     *
     * Each trial builds a vocabulary whose top k are known only to the reference
     * above, then asks the sampler the only question it answers: what comes
     * back. Nothing outside the reference's set may ever come back, and with the
     * k at near-equal probabilities everything inside it comes back fast. Those
     * two together say the two rankings selected the same set, which is the
     * whole claim of ranking logits instead of exponentiating first. */
    {
        const long WIDE = 4096;
        const int K = 20;
        float *w = (float *)malloc((size_t)WIDE * sizeof *w);
        rcand *ref = (rcand *)malloc((size_t)K * sizeof *ref);
        char *want = (char *)calloc((size_t)WIDE, 1);
        long foreign = 0, missed = 0, short_set = 0;
        const float scales[3] = { 1e-6f, 1e-4f, 1e-2f };
        for (int trial = 0; trial < 60; trial++) {
            float scale = scales[trial % 3];
            unsigned rot = (unsigned)trial * 631u;
            for (long i = 0; i < WIDE; i++)
                w[i] = -(float)(((unsigned)(i + rot) * 2654435761u) & 4095u) * scale;
            long nsel = ref_top(w, WIDE, K, 1.0f, ref);
            if (nsel != K)
                short_set++;
            memset(want, 0, (size_t)WIDE);
            for (long i = 0; i < nsel; i++)
                want[ref[i].id] = 1;
            uint64_t rng = 0x243f6a8885a308d3ull + (uint64_t)trial * 0x9e3779b97f4a7c15ull;
            qe_sampler_opts o = opts(1.0f, K, 1.0f, 0.0f);
            for (int i = 0; i < 1000; i++) {
                int id = qe_sample(w, WIDE, &o, &rng);
                if (id < 0 || id >= WIDE || !want[id])
                    foreign++;
                else
                    want[id] = 2;
            }
            for (long i = 0; i < WIDE; i++)
                if (want[i] == 1)
                    missed++;
        }
        check(short_set == 0, "the reference keeps k candidates on every shape");
        check(foreign == 0, "ranking logits selects no id the old ranking did not");
        check(missed == 0, "and misses none of the ids it selected");
        free(w);
        free(ref);
        free(want);
    }

    /* ---- What the selection ranks, exactly.
     *
     * Ranking on the logit is exact while a larger logit means a larger
     * probability, and two adjacent floats below the maximum can exponentiate to
     * the same float. Where this libm has such a pair, the candidate already in
     * the heap keeps its slot - which is what ranking the probabilities did, and
     * not what ranking the logit alone would do. The pairs are searched for
     * rather than assumed, because how many exist is a property of the libm. */
    {
        const int W = 4;
        float tie[W];
        int pairs = 0;
        for (int j = 0; j < 4096 && !pairs; j++) {
            float b = -1.0f - (float)j * 1e-4f;
            float a = nextafterf(b, INFINITY);
            if (expf(b) == expf(a)) {
                tie[0] = 0.0f; /* the maximum, so the shift is by zero */
                tie[1] = b;    /* in first, and inside a k of 2 */
                tie[2] = a;    /* a larger logit with the same probability */
                tie[3] = -20.0f;
                pairs = 1;
                long seen_b = 0, seen_a = 0;
                qe_sampler_opts o = opts(1.0f, 2, 1.0f, 0.0f);
                uint64_t rng = 0x243f6a8885a308d3ull;
                for (int i = 0; i < 4000; i++) {
                    int id = qe_sample(tie, W, &o, &rng);
                    seen_b += id == 1;
                    seen_a += id == 2;
                }
                /* A k of 2 is the argmax plus one slot, and that slot is the
                 * boundary between the two equal probabilities. */
                check(seen_a == 0 && seen_b > 0,
                      "an equal probability does not displace the candidate it ties");
            }
        }
        printf("  (%d collapsing logit pair%s on this libm)\n", pairs,
               pairs == 1 ? "" : "s");
    }

    /* ---- And what it costs, since the point of ranking logits is the
     * exponentials it does not compute. One selection and one draw over the
     * checkpoint's real vocabulary size, next to the same question asked the old
     * way, which is the number this change is claiming. */
    {
        const long VOCAB = 248320;
        const int K = 20;
        float *l = (float *)malloc((size_t)VOCAB * sizeof *l);
        rcand *ref = (rcand *)malloc((size_t)K * sizeof *ref);
        uint64_t s = 0x123456789abcdefull;
        for (long i = 0; i < VOCAB; i++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            l[i] = -10.0f + (float)(s % 100000u) * 1e-4f;
        }
        struct timespec t0, t1;
        long sink = 0;
        qe_sampler_opts o = opts(1.0f, K, 0.95f, 0.0f);
        uint64_t rng = 7;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < 200; i++)
            sink += qe_sample(l, VOCAB, &o, &rng);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double now_us = ((double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                         (double)(t1.tv_nsec - t0.tv_nsec)) / 200.0 / 1e3;
        double ref_us = 0.0;
        for (int i = 0; i < 3; i++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int r = 0; r < 50; r++)
                sink += (long)ref_top(l, VOCAB, K, 1.0f, ref);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double us = ((double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                         (double)(t1.tv_nsec - t0.tv_nsec)) / 50.0 / 1e3;
            if (i == 0 || us < ref_us)
                ref_us = us;
        }
        check(now_us < ref_us, "ranking logits is cheaper than exponentiating first");
        printf("  sampler: %.1f us per draw over %ld logits, against %.1f us to "
               "select the same k by exponentiating every one of them\n",
               now_us, VOCAB, ref_us);
        free(l);
        free(ref);
        (void)sink;
    }

    /* ---- the same seed is the same answer ---- */
    {
        qe_sampler_opts o = opts(1.0f, 20, 0.95f, 0.1f);
        uint64_t r1 = 99, r2 = 99;
        int same = 1;
        for (int i = 0; i < 200; i++)
            same &= qe_sample(v, NVOC, &o, &r1) == qe_sample(v, NVOC, &o, &r2);
        check(same, "one seed twice is one answer twice");
        uint64_t r3 = 100;
        int differs = 0;
        for (int i = 0; i < 200; i++)
            differs |= qe_sample(v, NVOC, &o, &r1) != qe_sample(v, NVOC, &o, &r3);
        check(differs, "a different seed is a different answer");
    }

    /* ---- the caller's rng stream advances, so a loop of draws is a
     * sequence and not one token repeated ---- */
    {
        qe_sampler_opts o = opts(1.0f, 0, 1.0f, 0.0f);
        uint64_t r = 7, r0 = 7;
        int a = qe_sample(v, NVOC, &o, &r);
        check(r != r0, "a draw advances the stream it was handed");
        int b = qe_sample(v, NVOC, &o, &r);
        uint64_t r_again = 7;
        check(a == qe_sample(v, NVOC, &o, &r_again), "restarting the stream repeats it");
        check(b == qe_sample(v, NVOC, &o, &r_again), "token for token");
    }

    /* ---- what the checkpoint does not say is off ---- */
    {
        qe_sampler_opts o;
        qe_sampler_opts_from_model(NULL, &o);
        check(o.min_p == 0.0f, "a model that says nothing leaves min_p off");
        check(o.top_k == 20 && o.top_p == 0.95f && o.temp == 1.0f,
              "and the documented defaults for the rest");
    }

    /* ---- degenerate inputs ---- */
    {
        qe_sampler_opts o = opts(1.0f, 0, 1.0f, 0.5f);
        check(qe_sample(NULL, NVOC, &o, NULL) == -1, "NULL logits are refused");
        check(qe_sample(v, 0, &o, NULL) == -1, "an empty vocabulary is refused");
        float flat[NVOC];
        for (int i = 0; i < NVOC; i++)
            flat[i] = -1e31f;
        check(qe_sample(flat, NVOC, &o, NULL) == -1, "all-non-finite logits are refused");
        float one[NVOC] = { 0.0f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f };
        check(qe_sample(one, NVOC, &o, NULL) == 0,
              "a floor above every other candidate still returns the only candidate");
    }

    printf("\n%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
