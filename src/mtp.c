/* Speculative decoding control loop.
 *
 * The whole reason this file exists is the arithmetic in docs/DESIGN.md 4: one
 * decode step moves 5.44 GiB whether it emits one token or 2.76 of them, so
 * raising tokens-per-step is the only lever that changes the tok/s ceiling. At
 * the measured kernel rates, 2.76 tokens per step is the difference between
 * ~11.4 and ~16.5 tok/s.
 *
 * The rule that keeps this honest: at the first position where the draft and
 * the target disagree, the target's own token is emitted and the rest of the
 * draft is thrown away. With temp == 0 that makes the emitted sequence exactly
 * equal to what plain greedy decoding would have produced, drafts being an
 * optimisation and never a change of output. bench_mtp asserts precisely that
 * against a toy target with both a perfect and a useless drafter.
 *
 * With temp > 0 the accept test is the usual rejection test, accept the draft
 * token x with probability min(1, p_target(x) / p_draft(x)). On rejection the
 * continuation is drawn from the *target* rather than from the usual
 * max(0, p_target - p_draft) residual: the residual needs the draft model's
 * full distribution at that position, and the MTP head as shipped exposes only
 * the drafted token. That is a documented deviation from the textbook, not an
 * oversight, and it only affects sampled output.
 */
#include "qe.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define QE_SPEC_MAX_K 64

/* splitmix64, uniform in [0,1). 53 bits, so every value is representable. */
static double rng01(uint64_t *rng) {
    uint64_t z = *rng += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

/* --------------------------------------------------------------- row math */

static int row_argmax(const float *r, long n) {
    long best = 0;
    float bv = -FLT_MAX;
    for (long i = 0; i < n; i++)
        if (r[i] > bv) {
            bv = r[i];
            best = i;
        }
    return (int)best;
}

/* log sum exp of row/temp, so a logits row that peaks at 80 does not overflow
 * expf and a row that peaks at -80 does not underflow to a flat distribution. */
static double row_logsumexp(const float *r, long n, float temp) {
    float mx = r[0];
    for (long i = 1; i < n; i++)
        if (r[i] > mx)
            mx = r[i];
    const double inv = 1.0 / temp;
    double s = 0;
    for (long i = 0; i < n; i++)
        s += exp(((double)r[i] - mx) * inv);
    return (double)mx * inv + log(s);
}

/* Probability of `id` under softmax(row/temp). Same inv and the same lse as
 * row_logsumexp, so the two agree by construction. */
static double row_prob(const float *r, long n, int id, float temp) {
    const double inv = 1.0 / temp;
    return exp((double)r[id] * inv - row_logsumexp(r, n, temp));
}

/* One draw from softmax(row / temp) by Gumbel-max: argmax of
 * (row/temp + log(-log(u))) is distributed exactly as the softmax, with no
 * normalising constant and one pass over the row. */
static int row_sample(const float *r, long n, float temp, uint64_t *rng) {
    const double inv = 1.0 / temp;
    long best = 0;
    double bv = -DBL_MAX;
    for (long i = 0; i < n; i++) {
        double u = rng01(rng);
        double g = log(-log(u));
        double v = (double)r[i] * inv + g;
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return (int)best;
}

/* ------------------------------------------------------------- emit state */

typedef struct {
    int *seq;
    long cap;
    long len;
    long emitted;
    long limit;
    int done;
    int err;
    uint64_t rng;
    const qe_spec_model *m;
    int eos;
} spec_out;

/* Append one token, stream it, and report whether generation should stop.
 * Everything that has to happen per token lives here and nowhere else,
 * because the three call sites (rejected draft, accepted draft, free token)
 * were otherwise free to drift apart. */
static int emit_tok(spec_out *o, int t) {
    if (o->emitted >= o->limit) {
        o->done = 1;
        return 1;
    }
    if (o->len >= o->cap) {
        long cap = o->cap * 2;
        int *seq = realloc(o->seq, (size_t)cap * sizeof *seq);
        if (!seq) {
            o->err = 1;
            return 1;
        }
        o->seq = seq;
        o->cap = cap;
    }
    o->seq[o->len++] = t;
    o->emitted++;
    int last = (t == o->eos);
    if (o->m->emit(o->m->ctx, t, last))
        o->done = 1;
    if (last)
        o->done = 1;
    return o->done;
}

/* ------------------------------------------------------------------ driver */

/* Row `i` of the window, from a target that walks them one at a time. The scratch
 * is the first row of the buffer the other spelling would have written every row
 * into, which is why one allocation serves both. */
static const float *row_at(const qe_spec_model *m, long long pos, const int *win, int n, int i,
                           float *scratch) {
    return m->verify_row(m->ctx, pos, win, n, i, scratch) ? NULL : scratch;
}

long qe_spec_generate(const qe_spec_model *m, const qe_spec_opts *o, const int *prompt, int np,
                      qe_spec_stats *stats) {
    /* Cleared before anything else, because both of the ways this call fails -
     * a model with no callbacks, an allocation that did not come back - return
     * before there is a count to report, and a caller that prints what it was
     * handed would be reading whatever the stack happened to hold. */
    if (stats)
        memset(stats, 0, sizeof *stats);
    if (!m || (!m->verify && !m->verify_row) || !m->emit || !prompt || np <= 0 || m->vocab <= 0)
        return -1;
    qe_spec_opts def = { 0, 0.0f, 0, 0, -1 };
    if (!o)
        o = &def;
    int k = o->k > 0 ? o->k : 0;
    if (k > QE_SPEC_MAX_K)
        k = QE_SPEC_MAX_K;
    const long vocab = (long)m->vocab;
    const float temp = o->temp > 0 ? o->temp : 0.0f;

    long cap = (long)np + (long)k + 64;
    spec_out out = {
        .seq = malloc((size_t)cap * sizeof *out.seq),
        .cap = cap,
        .len = np,
        .emitted = 0,
        .limit = o->max_tokens > 0 ? o->max_tokens : LONG_MAX,
        .done = 0,
        .err = 0,
        .rng = (uint64_t)(o->seed ? o->seed : 0x2545f4914f6cdd1dull) | 1,
        .m = m,
        .eos = o->eos,
    };
    /* A target that walks its rows one at a time needs one row of the window in
     * hand, not the whole window, and says so by not having a `verify`. */
    float *logits = malloc((size_t)(m->verify ? k + 1 : 1) * (size_t)vocab * sizeof *logits);
    int *win = malloc((size_t)(k + 1) * sizeof *win);
    int d_ids[QE_SPEC_MAX_K];
    float d_p[QE_SPEC_MAX_K];
    if (!out.seq || !logits || !win) {
        free(out.seq);
        free(logits);
        free(win);
        return -1;
    }
    memcpy(out.seq, prompt, (size_t)np * sizeof *out.seq);

    qe_spec_stats s = { 0, 0, 0, 0, 0, 0 };

    while (!out.done && out.emitted < out.limit) {
        const long len = out.len;
        int nd = 0;
        if (k > 0 && m->draft) {
            nd = m->draft(m->ctx, len, out.seq, (int)len, d_ids, temp > 0 ? d_p : NULL, k);
            if (nd < 0)
                nd = 0;
            if (nd > k)
                nd = k;
        }
        /* Verification window: the last accepted token followed by the draft.
         * Row i of the returned logits is therefore the distribution for the
         * token that should follow d_ids[i-1], which is compared against
         * d_ids[i] - and row nd is the free token this step earns. */
        win[0] = out.seq[len - 1];
        for (int i = 0; i < nd; i++)
            win[i + 1] = d_ids[i];
        if (m->verify && m->verify(m->ctx, len - 1, win, nd + 1, logits)) {
            free(out.seq);
            free(logits);
            free(win);
            return -1;
        }
        s.steps++;
        s.drafted += nd;

        /* The boundary this window ends at is the number of tokens it put into the
         * sequence, and nothing else: the loop's invariant is that the target stands
         * one token short of the sequence, one is fed by row 0 of the next window,
         * and so the rows that were right are exactly the ones that were paid for.
         * Counting them here rather than tracking a row index is what keeps the two
         * ways a token is not kept - a draft that reached the limit, and the marker
         * that closes a turn - from leaving the target standing one row too far. */
        const long kept0 = out.emitted;
        for (int i = 0; i < nd; i++) {
            const float *row = m->verify_row ? row_at(m, len - 1, win, nd + 1, i, logits)
                                             : logits + (size_t)i * vocab;
            if (!row) {
                out.err = 1;
                goto step_end;
            }
            if (temp <= 0) {
                int best = row_argmax(row, vocab);
                if (best != d_ids[i]) {
                    emit_tok(&out, best);
                    goto step_end;
                }
                s.accepted++;
            } else {
                const double pt = row_prob(row, vocab, d_ids[i], temp);
                const double q = d_p[i] > 0 ? d_p[i] : 1.0;
                if (rng01(&out.rng) * q > pt) {
                    emit_tok(&out, row_sample(row, vocab, temp, &out.rng));
                    goto step_end;
                }
                s.accepted++;
            }
            emit_tok(&out, d_ids[i]);
            if (out.done || out.err)
                goto step_end;
        }
        /* Every draft survived, so the target owes one more token for free. */
        if (!out.done) {
            const float *row =
                m->verify_row ? row_at(m, len - 1, win, nd + 1, nd, logits)
                              : logits + (size_t)nd * vocab;
            if (!row) {
                out.err = 1;
                goto step_end;
            }
            int t = temp > 0 ? row_sample(row, vocab, temp, &out.rng) : row_argmax(row, vocab);
            emit_tok(&out, t);
        }
    step_end:;
        /* The rows nobody read never happened. A target that walks them one at a
         * time has still walked them through its trunk - that is what a window
         * is - and this is where the sequence goes back to the last row whose
         * token it kept. */
        if (m->rewind)
            m->rewind(m->ctx, len - 1, (int)(out.emitted - kept0));
        if (out.err)
            break;
    }

    free(out.seq);
    free(logits);
    free(win);
    if (stats) {
        s.tokens = out.emitted;
        s.tok_per_step = s.steps ? (double)s.tokens / (double)s.steps : 0;
        s.accept_rate = s.drafted ? (double)s.accepted / (double)s.drafted : 0;
        *stats = s;
    }
    if (out.err)
        return -1;
    return out.emitted;
}
