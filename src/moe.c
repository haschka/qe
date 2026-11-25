/* Routed-expert pipeline: router, top-k, and the two-phase expert sweep.
 *
 * The budget says the routed experts are 1.08 GiB of the 5.44 GiB read per
 * token (19.9%), and that 10/512 is already the floor: no cleverness reads
 * fewer bytes. What is still on the table is *how* those bytes arrive.
 *
 *   - Looping over the 10 selected experts serially puts one core in charge of
 *     2.3 MiB of DRAM while 19 idle. Slicing by (expert, row) instead gives
 *     every core a contiguous run of one or two experts, which is both
 *     machine-wide parallel and prefetcher-friendly.
 *   - Gate/up and down cannot be fused per expert, because down needs the
 *     whole intermediate vector. So the pass is two sweeps with one pool
 *     barrier between them rather than ten serial descents: the 10 x 2.5 KiB
 *     intermediate stays in L1 across that barrier.
 *   - Writing the down output directly would need a cross-thread reduction per
 *     element, because every expert contributes to all K outputs. Per-thread
 *     partials plus one cheap transpose-reduce beat atomics outright: 20 x
 *     10 KiB of L1-resident scratch and no cache-line ping-pong.
 *   - Prefetch was measured and lost, both forms of it. Software prefetch at
 *     distances 8, 24 and 64 made the expert sweep 6-67% slower than issuing
 *     none, and warming a chosen expert's matrix head before the pass was
 *     slower in two of three thread counts and noise-level in the third. Rows
 *     are contiguous inside an expert, which is what the hardware prefetcher
 *     is for; nothing here beats it. See bench_moe.
 *
 * What this deliberately does not do: rebalance routing for load, reuse
 * another token's expert output, or otherwise change which weights a token
 * sees. Those change model output, and routing is not ours to renegotiate.
 */
#include "qe.h"
#include "kern.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define QE_MOE_CHUNK 8 /* rows per gate/up call: one AVX2 silu, no allocation */

/* Rows of one int8 panel per thread. The grouped pass never asks a shared-dequant call
 * for more than QE_MOE_SHARED_GROUP rows of a bucket, and that is clamped to
 * QE_IQ_PARK_MAX, so this is the ceiling on what a panel has to hold. Named here
 * because the panel is sized when the object is created, long before a pass runs. */
#define QE_MOE_PANEL_ROWS QE_IQ_PARK_MAX

/* What one pass over the routed experts is about: its rows, its routes, where its
 * output goes, and how many rows of one bucket one shared-dequant call covers. The
 * tasks of a pass take this as their argument, so it has to outlive them, which is
 * what the moe object's own copy (below) is for: the pass in flight. */
struct moe_job {
    const struct qe_moe *m;
    const float *x;
    const qe_route *r;
    int n;
    float *y;
    /* Rows of one bucket per shared-dequant call (moe_shared_group). Read once, on
     * the thread that owns the pass, so that nineteen workers cannot be handed a
     * cap that moved underneath them; unused by the two grouped tasks, which have
     * no group. */
    int gmax;
};
typedef struct moe_job moe_job;

typedef struct {
    const struct qe_moe *m;
    const float *x;
    float *logits;
} router_job;

struct qe_moe {
    qe_moe_desc d;
    router_job rjob; /* a step's router, when it runs as an item of a graph */
    /* A decode step's routed pass (qe_moe_step_begin): the routes it walks, the output,
     * and the two cursors its tasks hand rows out from. */
    struct {
        const float *x;
        const qe_route *r;
        float *y;
        int n;
        int i8;
        long next_gu, next_dn;
    } st;
    long rb_gate, rb_up, rb_down;       /* bytes per weight row */
    long ldd_gate, ldd_up, ldd_down;    /* leading dimension, in blocks */
    int n_threads;
    int shared_multi; /* grouped pass shares the dequantisation where it can */
    float *logits; /* n_experts */
    float *h;      /* top_k * N intermediate */
    float *part;   /* n_threads * K down partials */
    float *tmp;    /* n_threads * K down output before scaling */
    /* int8 recode of the two vectors the grouped pass multiplies with, for the
     * integer expert walks (include/qe.h, qe_i8panel): [0] is the K-wide activation
     * row of gate/up, [1] the N-wide intermediate row of down. Per pool thread,
     * because the thread that gathered a bucket is the thread that walks it. */
    struct {
        int8_t *x8;
        float *s;
        int32_t *p;
        long row; /* bytes between panel rows of one thread */
        long nb;  /* 32-value blocks in a panel row */
    } i8[2];
    void *scratch_base;
    size_t scratch;
    uint64_t *hist;
    long tokens;

    /* Grouped multi-row pass. Allocated on first use, grown on demand.
     *
     * A decode step that carries T rows - an MTP verify window, or T
     * conversations served in one batch - and takes top_k experts per row does
     * not need T*top_k expert matrices: several rows choose the same expert,
     * and one read of that expert applied to every row that asked for it costs
     * the same DRAM as a read for the first row alone. The grouping is a
     * counting sort over expert ids (512 of them, so the histogram is a
     * kilobyte) and the pass then walks buckets instead of items. */
    int mt_cap;               /* rows the multi scratch fits */
    int mt_T;                 /* rows in the grouping built by the last call */
    int n_bkt;                /* buckets in the grouping built by the last call */
    int dyn;                  /* QE_MOE_DYN: the grouped pass on cursors, down deterministic */
    long next_gu, next_dn;    /* the cursors of the pass in flight */
    int *itm_row;             /* item -> which of the T rows it belongs to */
    int *itm_e;               /* item -> expert id */
    float *itm_w;             /* item -> routing weight */
    int *bkt_lo;              /* bucket -> first item, n_bkt+1 entries */
    int *bkt_e;               /* bucket -> expert id */
    int *sort_itm;            /* item index in expert order */
    int *sort_hist;           /* counting-sort histogram, n_experts+1 */
    int *sort_cur;            /* write cursor for the counting sort */
    float *h2;                /* T * top_k * N intermediate */
    float *part2;             /* n_threads * T * K down partials (sliced down only) */
    size_t part2_bytes;
    int part2_T;
    void *mscratch_base;
    size_t mscratch;
    const float *mx;          /* T activations, stride mx_stride */
    float *my;                /* T outputs, stride my_stride */
    long mx_stride, my_stride;
    const qe_route *mr;
    int mr_per_row;
    /* The grouped pass in flight. The three tasks of that pass take a pointer to
     * this, and one pass per layer is what a chunk means, so there is exactly one
     * of them and it is the moe object's to keep alive. */
    moe_job mjob;
};

/* ------------------------------------------------------------- int8 walks */

/* Whether the grouped expert pass recodes the vector it multiplies with to int8 and
 * walks the experts with vpdpbusd instead of dequantising per 32 weights into floats
 * (include/qe.h, qe_i8panel). It does that wherever the weight format has an integer walk
 * and the width divides into whole blocks; QE_I8_MOE=0 is how an operator asks for the
 * dequantising walk back, and `make moe` and `make forward` A/B the two in one load with
 * the setter below.
 *
 * It ships on, and the price is on the record rather than negotiable (`bench_score
 * --ingest 5`: five 512-token windows of the corpus, each prefilled once per arithmetic,
 * /tmp/ingest2.log). The walk is worth 0.824x of the expert stage and 1.100x of a whole
 * step, and what it charges is 1.10 nats mean and 4.69 worst on the loss of the token that
 * follows the window, with the greedy choice moved once in five. The bucket reordering this
 * engine already ships, measured over the same five windows by the same instrument, charges
 * 0.29 mean and 0.49 worst and moves the choice none. Both stay inside the half-of-the-head's
 * range that the ingestion legs of `make forward` hold a change of *schedule* to (30 %
 * against 11 %). What decided it is ideas.md 7.13: the format-matched instrument put the
 * expert stage's entire remaining headroom, walked with nothing expert about it, at 1.2-1.6x,
 * and this walk is 1.21x of that - so the choice was never this against a safer kernel, it
 * was this against no kernel, because the row-pair and codebook rewrites that were supposed
 * to be worth 3x measured dead. A panel that carries less activation error at the same
 * multiply rate is still the item that would make it cheap (ideas.md 7.11); until one exists
 * the speed and the loss ship together. */
static int moe_int8 = -1;

static int moe_int8_on(void) {
    if (moe_int8 < 0) {
        const char *e = getenv("QE_I8_MOE");
        moe_int8 = e ? (strtol(e, NULL, 10) != 0) : 1;
    }
    return moe_int8;
}

/* Negative means "ask the environment again", which is what a bench that switches the
 * walk on for one leg and off for the next needs the way back to be. */
void qe_moe_set_int8(int v) { moe_int8 = v < 0 ? -1 : (v ? 1 : 0); }

int qe_moe_int8(void) { return moe_int8_on(); }

/* ------------------------------------------------------------- balance */

/* How a grouped pass cuts the expert stage (include/qe.h): 0 by count, 1 by cost, 2 by
 * cost and core class. */
static int moe_bal = -1;

static int moe_balanced(void) {
    if (moe_bal < 0) {
        const char *e = getenv("QE_MOE_BAL");
        long v = e ? strtol(e, NULL, 10) : QE_MOE_BAL_DEFAULT;
        moe_bal = v < 0 ? 0 : (v > 2 ? 2 : (int)v);
    }
    return moe_bal;
}

void qe_moe_set_balance(int v) { moe_bal = v < 0 ? -1 : (v > 2 ? 2 : v); }

int qe_moe_balance(void) { return moe_balanced(); }

/* Panel `w` for thread `tid`: its own rows, so two threads walking different buckets
 * never write the same scratch, with the rows of one group in the order they were
 * gathered (qe_gemv_i8_t reads a panel in the panel's own order). */
static qe_i8panel moe_panel(const struct qe_moe *m, int w, int tid) {
    const size_t rows = (size_t)tid * (size_t)QE_MOE_PANEL_ROWS * (size_t)m->i8[w].row;
    const size_t nbv = (size_t)tid * (size_t)QE_MOE_PANEL_ROWS * (size_t)m->i8[w].nb;
    qe_i8panel a = { .x8 = m->i8[w].x8 + rows,
                     .s = m->i8[w].s + nbv,
                     .p = m->i8[w].p + nbv,
                     .stride8 = m->i8[w].row,
                     .nb = m->i8[w].nb };
    return a;
}

/* Recode `np` gathered rows of `x_stride`-spaced source into panel rows 0..np-1. */
static void moe_panel_fill(const qe_i8panel *a, const float *const *xs, int np) {
    /* One scale per row when the walks that will read this panel were selected for it; the
     * row-scaled panel is the same layout with equal scales, so the expert rows whose walk
     * has no row-scale form (the Q8_0 ones, whose unsigned exit owes a seed per block anyway)
     * still read it correctly. */
    const int rs = qe_i8_rowscale();
    for (int q = 0; q < np; q++)
        (rs ? qe_quant_q8_0_row_rs : qe_quant_q8_0_row)(
            xs[q], a->x8 + (size_t)q * a->stride8, a->s + (size_t)q * a->nb,
            a->p + (size_t)q * a->nb, a->stride8);
}

/* ------------------------------------------------------------------ layout */

static long blocks_per_row(qe_type t, long cols) {
    const qe_type_info *ti = qe_type_info_of(t);
    return ti ? cols / (long)ti->block_ne : 0;
}

static long row_bytes_of(qe_type t, long cols) {
    const qe_type_info *ti = qe_type_info_of(t);
    return ti ? cols / (long)ti->block_ne * (long)ti->block_sz : 0;
}

static inline const uint8_t *expert_row(const void *base, long stride, long expert, long row,
                                        long row_bytes) {
    return (const uint8_t *)base + (size_t)expert * (size_t)stride +
           (size_t)row * (size_t)row_bytes;
}

/* --------------------------------------------------------------- activation */

/* h = silu(g) * u comes from qe_silu_mul (include/qe.h, src/layer.c): the same
 * block primitive the rest of the engine calls, so the expert MLP and the
 * dense one cannot drift into computing two different activations. */

/* ------------------------------------------------------------------- create */

qe_moe *qe_moe_create(const qe_moe_desc *d) {
    if (!d || !d->gate || !d->up || !d->down || d->n_experts <= 0 || d->N <= 0 || d->K <= 0 ||
        d->top_k <= 0) {
        errno = EINVAL;
        return NULL;
    }
    int nt = d->n_threads;
    if (nt <= 0) {
        /* Whole pool, from bench_moe's sweep with QE_MOE_REPEATS=5 (median and
         * range of five full sweeps, GB/s, on a synthetic 1.11 GiB pool with
         * routes redrawn every iteration):
         *   4  -> median 19.2      (19.1 .. 19.3, no variance at all)
         *   8  -> median 24.5      (23.9 .. 24.6)
         *  12  -> median 25.8      (24.4 .. 32.1)
         *  20  -> median 28.1      (22.8 .. 47.3)
         * Twenty has the best median (+15% over eight) and by far the best
         * ceiling, but its spread is enormous - twenty threads once measured
         * worse than four. Eight threads is the choice if token *arrival*
         * regularity matters more than tok/s; it is within 3% every run. Both
         * claims that died producing this table are worth keeping: "8 beats 20"
         * (a band of noise read as an ordering from one sample) and "more is
         * monotone" (true of medians, false of any single run). Cap it with
         * --threads. */
        nt = qe_pool_size();
    }
    if (nt <= 0)
        nt = 1;
    struct qe_moe *m = calloc(1, sizeof *m);
    if (!m)
        return NULL;
    m->d = *d;
    if (m->d.top_k > d->n_experts)
        m->d.top_k = (int)d->n_experts;
    m->n_threads = nt;
    {
        const char *ev = getenv("QE_MOE_DYN");
        m->dyn = ev && *ev ? atoi(ev) : 1;
    }
    m->shared_multi = !d->no_shared_multi;
    m->rb_gate = row_bytes_of(d->t_gate, d->K);
    m->rb_up = row_bytes_of(d->t_up, d->K);
    m->rb_down = row_bytes_of(d->t_down, d->N);
    m->ldd_gate = blocks_per_row(d->t_gate, d->K);
    m->ldd_up = blocks_per_row(d->t_up, d->K);
    m->ldd_down = blocks_per_row(d->t_down, d->N);
    if (!m->rb_gate || !m->rb_up || !m->rb_down) {
        free(m);
        errno = EINVAL;
        return NULL;
    }
    /* One mapping, so the whole working set of the scratch is a single
     * huge page for the sizes that matter. */
    const long tk = m->d.top_k;
    const size_t lb = qe_align_up((size_t)d->n_experts * sizeof(float), 64);
    const size_t hb = qe_align_up((size_t)(tk * d->N) * sizeof(float), 64);
    const size_t pb = qe_align_up((size_t)nt * d->K * sizeof(float), 64);
    /* Panels: QE_MOE_PANEL_ROWS rows per thread of the widths the two walks read,
     * which round down to whole 32-value blocks because that is what a block is. */
    const long kq = (d->K / 32) * 32, nq = (d->N / 32) * 32;
    const size_t gsz = (size_t)nt * QE_MOE_PANEL_ROWS;
    const size_t xk = qe_align_up(gsz * (size_t)kq, 64);
    const size_t xn = qe_align_up(gsz * (size_t)nq, 64);
    const size_t sk = 2 * qe_align_up(gsz * (size_t)(kq / 32) * sizeof(float), 64);
    const size_t sn = 2 * qe_align_up(gsz * (size_t)(nq / 32) * sizeof(float), 64);
    m->i8[0].row = kq;
    m->i8[0].nb = kq / 32;
    m->i8[1].row = nq;
    m->i8[1].nb = nq / 32;
    m->scratch = lb + hb + 2 * pb + xk + xn + sk + sn;
    char *s = qe_alloc_huge(m->scratch);
    if (!s) {
        free(m);
        errno = ENOMEM;
        return NULL;
    }
    m->logits = (float *)s;
    m->h = (float *)(s + lb);
    m->part = (float *)(s + lb + hb);
    m->tmp = (float *)(s + lb + hb + pb);
    {
        char *q = s + lb + hb + 2 * pb;
        for (int w = 0; w < 2; w++) {
            const size_t rb = w ? xn : xk, sb = w ? sn : sk;
            m->i8[w].x8 = (int8_t *)q;
            m->i8[w].s = (float *)(q + rb);
            m->i8[w].p = (int32_t *)(q + rb + sb / 2);
            q += rb + sb;
        }
    }
    m->scratch_base = s;
    m->hist = calloc((size_t)d->n_experts, sizeof *m->hist);
    if (!m->hist) {
        qe_free_huge(s, m->scratch);
        free(m);
        errno = ENOMEM;
        return NULL;
    }
    return m;
}

void qe_moe_destroy(qe_moe *m) {
    if (!m)
        return;
    free(m->hist);
    if (m->mscratch)
        qe_free_huge(m->mscratch_base, m->mscratch);
    if (m->part2)
        qe_free_huge(m->part2, m->part2_bytes);
    if (m->scratch)
        qe_free_huge(m->scratch_base, m->scratch);
    free(m);
}

/* ------------------------------------------------------------------- router */

static void router_task(void *v, int tid, int nt) {
    router_job *j = v;
    const struct qe_moe *m = j->m;
    long lo, hi;
    qe_slice(m->d.n_experts, tid, nt, &lo, &hi);
    if (lo >= hi)
        return;
    /* Rows are contiguous either way; only the element size and the leading
     * dimension's unit change, both of which are in elements. */
    if (m->d.router_bf16)
        qe_gemv_bf16((const uint8_t *)m->d.router_bf16 + (size_t)lo * (size_t)m->d.K * 2, j->x,
                     j->logits + lo, hi - lo, m->d.K, m->d.K);
    else
        qe_gemv_f32((const float *)m->d.router + lo * m->d.K, j->x, j->logits + lo, hi - lo, m->d.K,
                    m->d.K);
}

void qe_moe_router(const qe_moe *m, const float *x, float *logits) {
    router_job j = { m, x, logits };
    qe_pool_run(router_task, &j, m->n_threads);
}

typedef struct {
    const struct qe_moe *m;
    const float *x;
    long x_stride;
    float *logits;
    long logits_stride;
    int T;
} router_multi_job;

static void router_multi_task(void *v, int tid, int nt) {
    router_multi_job *j = v;
    const struct qe_moe *m = j->m;
    long lo, hi;
    qe_slice(m->d.n_experts, tid, nt, &lo, &hi);
    if (lo >= hi)
        return;
    /* Rows outside, tokens inside, and no block smaller than the router: at 512
     * rows the whole slice is one block, so a row is read once and used T times
     * rather than T times over. Both widths go through the same dispatcher, and
     * neither has a shared-dequant form, so what each thread does is T copies of
     * router_task's inner call on its own slice. */
    const qe_type t = m->d.router_bf16 ? QE_BF16 : QE_F32;
    const void *w = m->d.router_bf16 ? (const void *)m->d.router_bf16 : (const void *)m->d.router;
    const size_t row_bytes = (size_t)m->d.K * (m->d.router_bf16 ? 2u : 4u);
    if (qe_gemv_multi_t(t, (const char *)w + (size_t)lo * row_bytes, j->x, j->logits + lo, hi - lo,
                        m->d.K, m->d.K, j->T, j->x_stride, j->logits_stride, row_bytes,
                        (size_t)m->d.n_experts * row_bytes) != 0)
        return;
}

void qe_moe_router_multi(const qe_moe *m, const float *x, long x_stride, float *logits,
                         long logits_stride, int T) {
    if (T < 1)
        return;
    if (T == 1) {
        qe_moe_router(m, x, logits);
        return;
    }
    router_multi_job j = { m, x, x_stride, logits, logits_stride, T };
    qe_pool_run(router_multi_task, &j, m->n_threads);
}

static void renorm_weights(qe_route *out, int k);

/* ------------------------------------------------------------------- select */

int qe_moe_select(const qe_moe *m, const float *logits, qe_route *out) {
    const long E = m->d.n_experts;
    const int k = m->d.top_k;
    for (int i = 0; i < k; i++) {
        out[i].logit = -INFINITY;
        out[i].expert = -1;
        out[i].weight = 0.0f;
    }
    /* Kept sorted ascending, so the running cutoff is one compare against
     * out[0] and 502 of the 512 logits cost no more than that. */
    for (long i = 0; i < E; i++) {
        const float l = logits[i];
        if (!(l == l) || l <= out[0].logit)
            continue;
        out[0].logit = l;
        out[0].expert = (int32_t)i;
        /* The replacement belongs at the top of the table, so it walks right
         * past everything smaller than it. Getting this comparison the other
         * way round leaves the table unsorted, the cutoff stops being the
         * cutoff, and the renormalisation below overflows to NaN. */
        for (int p = 0; p + 1 < k && out[p].logit > out[p + 1].logit; p++) {
            qe_route t = out[p];
            out[p] = out[p + 1];
            out[p + 1] = t;
        }
    }
    renorm_weights(out, k);
    return k;
}

/* softmax over the kept logits, renormalised to sum to 1. */
static void renorm_weights(qe_route *out, int k) {
    const float mx = out[k - 1].logit;
    double sum = 0;
    for (int i = 0; i < k; i++) {
        out[i].weight = (float)exp((double)(out[i].logit - mx));
        sum += out[i].weight;
    }
    if (sum > 0)
        for (int i = 0; i < k; i++)
            out[i].weight = (float)(out[i].weight / sum);
}

/* The k winning rows re-read at full width, so that the mixing weights are the
 * f32 router's and not the shadow's: an unrefined weight is off 1.3e-04 on a
 * shared pick, and 8 bits of mantissa on a number that scales an expert's whole
 * output is the one place a rounded read should not leak. k rows of K floats is
 * 100 KiB against the 120 MiB the shadow saves. A no-op when the descriptor has
 * only one width, since then there is nothing to be more exact than. */
void qe_moe_refine(const qe_moe *m, const float *x, qe_route *routes, int k) {
    if (!m->d.router || !m->d.router_bf16)
        return;
    /* A chunk's routes are padded to top_k with expert -1 once a token found fewer
     * winners than that, and the padding belongs to the grouped pass, not to the
     * distribution: renormalising over it would read a -INFINITY logit and make every
     * weight of the row NaN. */
    while (k > 0 && routes[k - 1].expert < 0)
        k--;
    if (k < 1)
        return;
    const float *W = (const float *)m->d.router;
    for (int i = 0; i < k; i++)
        if (routes[i].expert >= 0)
            qe_gemv_f32(W + (size_t)routes[i].expert * (size_t)m->d.K, x, &routes[i].logit, 1,
                        m->d.K, m->d.K);
    renorm_weights(routes, k);
}

/* Router, selection and exact weights, for a descriptor that carries the router
 * in both widths.
 *
 * The bf16 shadow is what gets read: all 512 rows of it cost half the f32 rows,
 * which over 48 layers is 240 MiB of step down to 120 MiB. Selection is the only
 * thing allowed to look at rounded values, and bin/bench_route is what says the
 * ten picks come out the same ten over 96000 real layer-token pairs. The mixing
 * weights come from qe_moe_refine, which leaves the renormalisation running on
 * exactly the numbers the f32 router produced.
 */
int qe_moe_route(const qe_moe *m, const float *x, float *logits, qe_route *out) {
    qe_moe_router(m, x, logits);
    int k = qe_moe_select(m, logits, out);
    qe_moe_refine(m, x, out, k);
    return k;
}

/* The rows of a chunk, refined in one dispatch.
 *
 * Per row this is 10 dots of K floats - a few microseconds - which is what makes
 * it worth a handback at all: 512 rows of them is 1.5 ms of a chunk's routed half,
 * and the top-k is already done by the time anyone asks, so the rows are
 * independent. Below a handful of rows the dispatch costs more than the rows it
 * spreads, and the loop is what ran before this function existed. */
typedef struct {
    const qe_moe *m;
    const float *x;
    long x_stride;
    qe_route *routes;
    int k, T;
} refine_job;

static void refine_task(void *v, int tid, int nt) {
    refine_job *j = v;
    long lo, hi;
    qe_slice((long)j->T, tid, nt, &lo, &hi);
    for (long t = lo; t < hi; t++)
        qe_moe_refine(j->m, j->x + (size_t)t * (size_t)j->x_stride,
                      j->routes + (size_t)t * (size_t)j->k, j->k);
}

void qe_moe_refine_rows(const qe_moe *m, const float *x, long x_stride, qe_route *routes,
                        int k, int T) {
    if (!m->d.router || !m->d.router_bf16 || k < 1 || T < 1)
        return;
    if (T < 4) {
        for (int t = 0; t < T; t++)
            qe_moe_refine(m, x + (size_t)t * (size_t)x_stride,
                          routes + (size_t)t * (size_t)k, k);
        return;
    }
    refine_job j = { m, x, x_stride, routes, k, T };
    qe_pool_run(refine_task, &j, m->n_threads);
}

/* ----------------------------------------------------------------- forward */

static void gate_up_task(void *v, int tid, int nt) {
    moe_job *j = v;
    const struct qe_moe *m = j->m;
    const long N = m->d.N, K = m->d.K;
    long lo, hi;
    qe_slice((long)j->n * N, tid, nt, &lo, &hi);
    if (lo >= hi)
        return;
    float g[QE_MOE_CHUNK], u[QE_MOE_CHUNK];
    for (long it = lo; it < hi;) {
        const long e = it / N, row0 = it % N;
        long run = N - row0;
        if (it + run > hi)
            run = hi - it;
        const qe_route *rt = j->r + e;
        const uint8_t *gp = expert_row(m->d.gate, m->d.gate_stride, rt->expert, row0, m->rb_gate);
        const uint8_t *up = expert_row(m->d.up, m->d.up_stride, rt->expert, row0, m->rb_up);
        long done = 0;
        while (done < run) {
            const long c = run - done >= QE_MOE_CHUNK ? QE_MOE_CHUNK : run - done;
            qe_gemv_t(m->d.t_gate, gp + (size_t)done * (size_t)m->rb_gate, j->x, g, c, K,
                      m->ldd_gate);
            qe_gemv_t(m->d.t_up, up + (size_t)done * (size_t)m->rb_up, j->x, u, c, K, m->ldd_up);
            qe_silu_mul(m->h + it + done, g, u, c);
            done += c;
        }
        it += run;
    }
}

static void down_task(void *v, int tid, int nt) {
    moe_job *j = v;
    const struct qe_moe *m = j->m;
    const long N = m->d.N, K = m->d.K;
    long lo, hi;
    qe_slice((long)j->n * K, tid, nt, &lo, &hi);
    float *part = m->part + (size_t)tid * K;
    float *tmp = m->tmp + (size_t)tid * K;
    /* Zeroing the whole partial costs 10 KiB of L1 traffic and removes any
     * reasoning about which row ranges this slice touches twice. It also has
     * to happen for a slice that ended up empty: reduce_task sums every
     * thread's partial regardless. */
    memset(part, 0, (size_t)K * sizeof(float));
    if (lo >= hi)
        return;
    for (long it = lo; it < hi;) {
        const long e = it / K, row0 = it % K;
        long run = K - row0;
        if (it + run > hi)
            run = hi - it;
        const qe_route *rt = j->r + e;
        const uint8_t *dp = expert_row(m->d.down, m->d.down_stride, rt->expert, row0, m->rb_down);
        /* Rows are consecutive inside one expert here, which is exactly what
         * the hardware prefetcher wants, so no software prefetch. */
        qe_gemv_t(m->d.t_down, dp, m->h + (size_t)e * N, tmp, run, N, m->ldd_down);
        const float w = rt->weight;
        float *dst = part + row0;
        for (long i = 0; i < run; i++)
            dst[i] += w * tmp[i];
        it += run;
    }
}

static void reduce_task(void *v, int tid, int nt) {
    moe_job *j = v;
    const struct qe_moe *m = j->m;
    long lo, hi;
    qe_slice(m->d.K, tid, nt, &lo, &hi);
    for (long k = lo; k < hi; k++) {
        float s = 0;
        for (int t = 0; t < nt; t++)
            s += m->part[(size_t)t * m->d.K + k];
        j->y[k] = s;
    }
}

void qe_moe_forward(qe_moe *m, const float *x, const qe_route *routes, int n_routes, float *y) {
    if (n_routes <= 0) {
        memset(y, 0, (size_t)m->d.K * sizeof(float));
        return;
    }
    moe_job j = { m, x, routes, n_routes, y, 0 };
    if (m->hist)
        for (int i = 0; i < n_routes; i++)
            if (routes[i].expert >= 0)
                m->hist[routes[i].expert]++;
    m->tokens++;
    qe_pool_run(gate_up_task, &j, m->n_threads);
    qe_pool_run(down_task, &j, m->n_threads);
    qe_pool_run(reduce_task, &j, m->n_threads);
}

/* ------------------------------------------------ grouped multi-row pass */

/* One grouped pass over T rows.
 *
 * Reading each selected expert once per row that chose it is the difference
 * between T*top_k expert matrices per step and n_distinct of them. For this
 * checkpoint that is 2.27 MiB per expert per layer, so every pair of rows that
 * agree on an expert is 2.27 MiB of the 88 ms step not read. The grouping costs
 * a counting sort over 512 ids, which is why it is done once per layer on the
 * calling thread rather than distributed.
 *
 * The weight reuse itself is L1 arithmetic: a bucket walks N rows in chunks of
 * QE_MOE_CHUNK rows and, inside a chunk, applies the same gate/up rows to each
 * row that asked for that expert. Eight rows of either matrix is 17 KiB, well
 * inside the 48 KiB L1, so the second and later rows hit cache, not DRAM. */
/* The per-thread partials of the sliced down (QE_MOE_DYN=0 only), grown on demand. */
static int moe_need_part2(qe_moe *m, int T) {
    if (m->part2 && T <= m->part2_T)
        return 1;
    const size_t bytes = (size_t)m->n_threads * (size_t)T * (size_t)m->d.K * sizeof(float);
    float *p = qe_alloc_huge(bytes);
    if (!p)
        return 0;
    if (m->part2)
        qe_free_huge(m->part2, m->part2_bytes);
    m->part2 = p;
    m->part2_bytes = bytes;
    m->part2_T = T;
    return 1;
}

static int moe_need_multi(qe_moe *m, int T) {
    if (T <= m->mt_cap)
        return 1;
    long items = (long)T * m->d.top_k;
    long E = m->d.n_experts;
    size_t bytes = 0;
    bytes += qe_align_up((size_t)items * sizeof(int), 64);       /* itm_row */
    bytes += qe_align_up((size_t)items * sizeof(int), 64);       /* itm_e */
    bytes += qe_align_up((size_t)items * sizeof(float), 64);     /* itm_w */
    bytes += qe_align_up((size_t)items * sizeof(int), 64);       /* sort_itm */
    bytes += qe_align_up((size_t)(items + 1) * sizeof(int), 64); /* bkt_lo */
    bytes += qe_align_up((size_t)items * sizeof(int), 64);       /* bkt_e */
    bytes += qe_align_up((size_t)(E + 1) * sizeof(int), 64);     /* sort_hist */
    bytes += qe_align_up((size_t)(E + 1) * sizeof(int), 64);     /* sort_cur */
    bytes += qe_align_up((size_t)items * m->d.N * sizeof(float), 64);   /* h2 */
    /* part2 lives in its own allocation (moe_need_part2): the deterministic down never reads
     * it, and at a 1024-row chunk it would be 200 MB a layer. */
    char *s = qe_alloc_huge(bytes);
    if (!s)
        return 0;
    qe_free_huge(m->mscratch_base, m->mscratch);
    m->mscratch_base = s;
    m->mscratch = bytes;
    size_t off = 0;
    m->itm_row = (int *)(s + off);
    off += qe_align_up((size_t)items * sizeof(int), 64);
    m->itm_e = (int *)(s + off);
    off += qe_align_up((size_t)items * sizeof(int), 64);
    m->itm_w = (float *)(s + off);
    off += qe_align_up((size_t)items * sizeof(float), 64);
    m->sort_itm = (int *)(s + off);
    off += qe_align_up((size_t)items * sizeof(int), 64);
    m->bkt_lo = (int *)(s + off);
    off += qe_align_up((size_t)(items + 1) * sizeof(int), 64);
    m->bkt_e = (int *)(s + off);
    off += qe_align_up((size_t)items * sizeof(int), 64);
    m->sort_hist = (int *)(s + off);
    off += qe_align_up((size_t)(E + 1) * sizeof(int), 64);
    m->sort_cur = (int *)(s + off);
    off += qe_align_up((size_t)(E + 1) * sizeof(int), 64);
    m->h2 = (float *)(s + off);
    off += qe_align_up((size_t)items * m->d.N * sizeof(float), 64);
    (void)off;
    m->mt_cap = T;
    return 1;
}

/* Build the expert buckets for T rows x routes_per_row routes. Returns the
 * number of distinct experts, or -1 if no route is valid. */
static int moe_group(qe_moe *m, const qe_route *r, int per_row, int T) {
    const long E = m->d.n_experts;
    memset(m->sort_hist, 0, (size_t)(E + 1) * sizeof(int));
    int items = 0;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < per_row; i++) {
            const qe_route *rt = r + (size_t)t * per_row + i;
            if (rt->expert < 0 || rt->expert >= E)
                continue;
            m->itm_row[items] = t;
            m->itm_e[items] = (int)rt->expert;
            m->itm_w[items] = rt->weight;
            m->sort_hist[rt->expert + 1]++;
            items++;
        }
    if (!items)
        return -1;
    for (long e = 0; e < E; e++)
        m->sort_hist[e + 1] += m->sort_hist[e];
    memcpy(m->sort_cur, m->sort_hist, (size_t)(E + 1) * sizeof(int));
    for (int i = 0; i < items; i++)
        m->sort_itm[m->sort_cur[m->itm_e[i]]++] = i;

    int nb = 0;
    for (int p = 0; p < items;) {
        int e = m->itm_e[m->sort_itm[p]];
        int q = p;
        while (q < items && m->itm_e[m->sort_itm[q]] == e)
            q++;
        m->bkt_e[nb] = e;
        m->bkt_lo[nb] = p;
        nb++;
        p = q;
    }
    m->bkt_lo[nb] = items;
    m->n_bkt = nb;
    return nb;
}

/* Rows of one bucket handed to a shared-dequant kernel at a time, which is the widest
 * group the kernels can walk the quantiser for once (QE_IQ_PARK_MAX, src/kern.h). The
 * cost that follows from this number is the number of walks, not the number of rows:
 * a bucket of ten at the eight this used to hold is two walks of an expert row for ten
 * tokens, which is why a real routing - 512 experts, ten picks a token, so an average
 * bucket of ten at a 512-token chunk - is exactly the case that was paying double. A
 * bucket holds at most one row per token, so this is also the largest bucket the pass
 * can serve in one group; wider ones split, into as few pieces as the sizes in
 * src/kern_iq.c allow. */
#define QE_MOE_SHARED_GROUP QE_IQ_PARK_MAX

/* The group limit actually used. It is a slice limit and nothing more: the kernels decide
 * for themselves how wide a walk they want (iq_park_take, src/kern_iq.c) and this only
 * stops this pass from staging a group wider than the kernels would take, which matters
 * in the down pass, where a group's outputs sit in a QE_MOE_DOWN_BLOCK-tall block of L1
 * between the walk that wrote them and the accumulate that read them.
 *
 * A cap under two means parking is off, and the slice then goes to eight rather than to
 * the ceiling above, so that the switch is one thing and not two: at that setting this
 * pass hands the kernels exactly the groups they take, which is what it did before the
 * parked forms existed, and `QE_PARK=0` is a way back to that configuration rather than
 * a third configuration nobody has measured. */
static int moe_shared_group(void) {
    const int c = qe_iq_park_cap();
    if (c < 2)
        return 8;
    if (c > QE_MOE_SHARED_GROUP)
        return QE_MOE_SHARED_GROUP;
    return c;
}

/* The two charge boundaries that belong to thread `tid` out of `total`. Level 2 divides
 * them by core class as well as by charge, which is what src/mem.c's engine-wide record
 * says is right for this dispatch and wrong for the byte-bound ones: an expert walk
 * reaches ~40 GB/s of the ~78 this machine reads, so it ends when the arithmetic is done,
 * and a core the kernel scores at 783 against 1011 should be handed 1/1.30x of the
 * arithmetic rather than 1/19th of the units. */
static void moe_charge_want(int tid, int nt, long total, int by_class, long *want) {
    double w[64], sum = 0; /* 64: the pool's own ceiling on threads (src/mem.c) */
    if (by_class && nt <= (int)(sizeof w / sizeof w[0])) {
        int i = 0;
        for (; i < nt; i++) {
            w[i] = qe_cpu_weight(i, nt);
            if (!(w[i] > 0.0)) {
                sum = 0;
                break;
            }
            sum += w[i];
        }
    } else {
        sum = 0;
    }
    if (sum > 0.0) {
        double lo = 0, hi = 0, acc = 0;
        for (int i = 0; i < nt; i++) {
            if (i == tid)
                lo = acc;
            acc += w[i] / sum;
            if (i == tid)
                hi = acc;
        }
        want[0] = (long)((double)total * lo);
        want[1] = (long)((double)total * hi);
    } else {
        want[0] = total * tid / nt;
        want[1] = total * (tid + 1) / nt;
    }
    /* Monotone and exhaustive by construction. Pinning the ends is what qe_slice does for
     * the same reason: it is cheaper than arguing about the last division. */
    if (tid == 0)
        want[0] = 0;
    if (tid == nt - 1)
        want[1] = total;
}

/* What one output row of one bucket costs, in units of a token row walked once: flat to a
 * chunk of token rows, which all share the same walk of the weight rows, and one more walk's
 * worth per chunk past that (bench_walk's own T table: eight rows cost 1.11x of one, sixteen
 * 2.45x). A bucket of two and a bucket of thirty are therefore not the same job, and at a
 * 512-row chunk with ten picks a token the router does not deal them evenly either. */
static long moe_unit_charge(const struct qe_moe *m, int b) {
    const long r = (long)m->bkt_lo[b + 1] - m->bkt_lo[b];
    return r > QE_MOE_CHUNK ? r : QE_MOE_CHUNK;
}

/* The slice of `per` output rows of every bucket that belongs to thread `tid`, out of a
 * stage whose units are (bucket, output row). Off, that is qe_slice's equal share of the
 * count. On, it is an equal share of the charge above, with a bucket's output rows divided
 * in proportion to it, which is what stops the stage waiting on whichever thread drew the
 * fat buckets. Both are a function of the histogram, so the split is the same on every run
 * of a chunk; nothing about it depends on which thread gets there first. */
static void moe_unit_slice(const struct qe_moe *m, int tid, int nt, long per, long *lo,
                           long *hi) {
    const long units = (long)m->n_bkt * per;
    const int bal = moe_balanced();
    if (bal < 1 || nt <= 1) {
        qe_slice(units, tid, nt, lo, hi);
        return;
    }
    long total = 0;
    for (int b = 0; b < m->n_bkt; b++)
        total += per * moe_unit_charge(m, b);
    long want[2];
    moe_charge_want(tid, nt, total, bal >= 2, want);
    long out[2] = { 0, units };
    long acc = 0;
    for (int b = 0, wi = 0; b < m->n_bkt && wi < 2; b++) {
        const long f = moe_unit_charge(m, b);
        const long c = f * per;
        for (; wi < 2 && want[wi] < acc + c; wi++)
            out[wi] = (long)b * per + (want[wi] - acc) / f;
        acc += c;
    }
    *lo = out[0];
    *hi = out[1];
}

static void gate_up_range(moe_job *j, int tid, long lo, long hi);

/* gate/up of a grouped pass. With the cursor (QE_MOE_DYN, ideas.md §7.35) a thread takes the
 * next (bucket, QE_MOE_GU_ROWS-row) unit instead of a slice fixed by what the buckets were
 * estimated to cost: every output row is still one walk on one thread, so the bytes are the
 * slice's, and a P-core that is done takes more instead of waiting for an E-core. */
#define QE_MOE_GU_ROWS 64
static void gate_up_multi(void *v, int tid, int nt) {
    moe_job *j = v;
    struct qe_moe *m = (struct qe_moe *)j->m;
    const long N = m->d.N;
    if (m->dyn) {
        const long upb = (N + QE_MOE_GU_ROWS - 1) / QE_MOE_GU_ROWS;
        const long units = upb * m->n_bkt;
        for (;;) {
            const long c = __atomic_fetch_add(&m->next_gu, 1, __ATOMIC_RELAXED);
            if (c >= units)
                return;
            const long b = c / upb, r0 = (c % upb) * QE_MOE_GU_ROWS;
            const long run = N - r0 < QE_MOE_GU_ROWS ? N - r0 : QE_MOE_GU_ROWS;
            gate_up_range(j, tid, b * N + r0, b * N + r0 + run);
        }
    }
    long lo, hi;
    moe_unit_slice(m, tid, nt, N, &lo, &hi);
    if (lo < hi)
        gate_up_range(j, tid, lo, hi);
}

static void gate_up_range(moe_job *j, int tid, long lo, long hi) {
    const struct qe_moe *m = j->m;
    const long N = m->d.N, K = m->d.K;
    float g[QE_MOE_CHUNK], u[QE_MOE_CHUNK];
    /* A shared-dequant group cannot share those two buffers: one walk of the rows
     * writes all of its outputs, which is the whole point. */
    const float *xs[QE_MOE_SHARED_GROUP];
    float *gop[QE_MOE_SHARED_GROUP], *uop[QE_MOE_SHARED_GROUP];
    float gm[QE_MOE_SHARED_GROUP][QE_MOE_CHUNK];
    float um[QE_MOE_SHARED_GROUP][QE_MOE_CHUNK];
    const int shg = m->shared_multi && qe_gemv_multi_ptrs_has(m->d.t_gate) &&
                    qe_gemv_multi_ptrs_has(m->d.t_up);
    const int i8g = moe_int8_on() && m->shared_multi && !(K % 32) &&
                    qe_gemv_i8_has(m->d.t_gate) && qe_gemv_i8_has(m->d.t_up);
    /* A decode step routes one token, so every bucket it fills is one row wide and the whole
     * grouped pass degenerates to the per-row walk this branch exists to avoid - except that the
     * row it would walk is an expert's entire gate and up matrices, and the integer walk is the
     * only form that spends a codebook value against an int8 activation instead of converting the
     * codebook to floats first. Where the format has a one-token form a bucket of one therefore
     * takes this branch too (include/qe.h, qe_gemv_i8_has_1). */
    const int i8one = i8g && qe_gemv_i8_has_1(m->d.t_gate) && qe_gemv_i8_has_1(m->d.t_up);
    const qe_i8panel ap = moe_panel(m, 0, tid);
    const int gmax = j->gmax;
    for (long it = lo; it < hi;) {
        const long b = it / N, row0 = it % N;
        long run = N - row0;
        if (it + run > hi)
            run = hi - it;
        const int e = m->bkt_e[b];
        const int first = m->bkt_lo[b], last = m->bkt_lo[b + 1];
        const uint8_t *gp = expert_row(m->d.gate, m->d.gate_stride, e, row0, m->rb_gate);
        const uint8_t *up = expert_row(m->d.up, m->d.up_stride, e, row0, m->rb_up);
        /* The integer walk takes the two loops the other way round. Recoding a token
         * row is cheap once and ruinous at a slice's rate: this loop's other order walks
         * the same bucket's rows once per eight output rows, so a panel filled inside it
         * would be rewritten ceil(run/8) times over for the same numbers. Group outer,
         * columns inner instead, and a row is recoded once per expert it routed to,
         * which is one pass over the chunk's K columns per expert (bench_moe 4d's `int`
         * column is what that costs against the walk it buys). The weight rows of a
         * bucket are then streamed per group rather than held in an eight-row window,
         * which is the trade and is what that column measures. */
        if (i8g && last - first >= (i8one ? 1 : 2)) {
            const float *gx[QE_MOE_SHARED_GROUP];
            float *gop2[QE_MOE_SHARED_GROUP], *uop2[QE_MOE_SHARED_GROUP];
            for (int p = first; p < last;) {
                const int np = last - p > gmax ? gmax : last - p;
                for (int q = 0; q < np; q++) {
                    const int itm = m->sort_itm[p + q];
                    gx[q] = m->mx + (size_t)m->itm_row[itm] * m->mx_stride;
                    gop2[q] = gm[q];
                    uop2[q] = um[q];
                }
                moe_panel_fill(&ap, gx, np);
                for (long d2 = 0; d2 < run;) {
                    const long c = run - d2 >= QE_MOE_CHUNK ? QE_MOE_CHUNK : run - d2;
                    /* The walk is refused only for a group narrower than its narrowest form,
                     * and a one-row group arrives here only when `i8one` says it has one. */
                    qe_gemv_i8_t(m->d.t_gate, gp + (size_t)d2 * (size_t)m->rb_gate, &ap, np,
                                 gop2, c, K, m->ldd_gate);
                    qe_gemv_i8_t(m->d.t_up, up + (size_t)d2 * (size_t)m->rb_up, &ap, np, uop2, c,
                                 K, m->ldd_up);
                    for (int q = 0; q < np; q++)
                        qe_silu_mul(m->h2 + (size_t)m->sort_itm[p + q] * N + row0 + d2, gm[q],
                                    um[q], c);
                    d2 += c;
                }
                p += np;
            }
            it += run;
            continue;
        }
        long done = 0;
        while (done < run) {
            const long c = run - done >= QE_MOE_CHUNK ? QE_MOE_CHUNK : run - done;
            const uint8_t *gw = gp + (size_t)done * (size_t)m->rb_gate;
            const uint8_t *uw = up + (size_t)done * (size_t)m->rb_up;
            for (int p = first; p < last;) {
                const int np = last - p > gmax ? gmax : last - p;
                if (shg && np >= 2) {
                    for (int q = 0; q < np; q++) {
                        const int itm = m->sort_itm[p + q];
                        xs[q] = m->mx + (size_t)m->itm_row[itm] * m->mx_stride;
                        gop[q] = gm[q];
                        uop[q] = um[q];
                    }
                    qe_gemv_multi_ptrs_t(m->d.t_gate, gw, xs, gop, np, c, K, m->ldd_gate);
                    qe_gemv_multi_ptrs_t(m->d.t_up, uw, xs, uop, np, c, K, m->ldd_up);
                    for (int q = 0; q < np; q++)
                        qe_silu_mul(m->h2 + (size_t)m->sort_itm[p + q] * N + row0 + done, gm[q],
                                    um[q], c);
                } else {
                    for (int q = 0; q < np; q++) {
                        const int itm = m->sort_itm[p + q];
                        const float *xr = m->mx + (size_t)m->itm_row[itm] * m->mx_stride;
                        qe_gemv_t(m->d.t_gate, gw, xr, g, c, K, m->ldd_gate);
                        qe_gemv_t(m->d.t_up, uw, xr, u, c, K, m->ldd_up);
                        qe_silu_mul(m->h2 + (size_t)itm * N + row0 + done, g, u, c);
                    }
                }
                p += np;
            }
            done += c;
        }
        it += run;
    }
}

/* Down-output rows held for one shared-dequant group. A bucket's rows arrive as the
 * slice the pool handed out, which can be the whole of K, so a group's outputs are
 * built a block of rows at a time; the block is also what keeps T output rows and the
 * weight rows they were computed from in the core that read them. */
#define QE_MOE_DOWN_BLOCK 128

/* down of a grouped pass, deterministically under a cursor (QE_MOE_DYN, ideas.md §7.35).
 *
 * The sliced form below sums per-thread partials, so which thread walked which bucket is part
 * of what an output sums to, and it zeroes and then reduces nineteen T x K partials a layer -
 * 5 MB each at a 512-row chunk. Here a unit is QE_MOE_DN_ROWS output rows of every bucket:
 * it zeroes those rows of the output for the chunk's tokens and adds each bucket's
 * contribution in bucket order, so a row of the answer is summed the same way whichever thread
 * took the unit, and nothing is reduced afterwards. */
#define QE_MOE_DN_ROWS 32
static void down_det(moe_job *j, int tid, int nt) {
    struct qe_moe *m = (struct qe_moe *)j->m;
    const long N = m->d.N, K = m->d.K;
    const int T = m->mt_T;
    const long units = (K + QE_MOE_DN_ROWS - 1) / QE_MOE_DN_ROWS;
    const float *xs[QE_MOE_SHARED_GROUP];
    float *dop[QE_MOE_SHARED_GROUP];
    float dm[QE_MOE_SHARED_GROUP][QE_MOE_DN_ROWS];
    float tmp[QE_MOE_DN_ROWS];
    const int shd = m->shared_multi && qe_gemv_multi_ptrs_has(m->d.t_down);
    const int i8d = moe_int8_on() && m->shared_multi && !(N % 32) && qe_gemv_i8_has(m->d.t_down);
    const int i8one = i8d && qe_gemv_i8_has_1(m->d.t_down);
    const qe_i8panel ap = moe_panel(m, 1, tid);
    const int gmax = j->gmax;
    for (;;) {
        const long c = __atomic_fetch_add(&m->next_dn, 1, __ATOMIC_RELAXED);
        if (c >= units)
            return;
        const long k0 = c * QE_MOE_DN_ROWS;
        const long run = K - k0 < QE_MOE_DN_ROWS ? K - k0 : QE_MOE_DN_ROWS;
        for (int t = 0; t < T; t++)
            memset(m->my + (size_t)t * m->my_stride + k0, 0, (size_t)run * sizeof(float));
        for (int b = 0; b < m->n_bkt; b++) {
            const int e = m->bkt_e[b];
            const uint8_t *dp = expert_row(m->d.down, m->d.down_stride, e, k0, m->rb_down);
            const int first = m->bkt_lo[b], last = m->bkt_lo[b + 1];
            for (int p = first; p < last;) {
                const int np = last - p > gmax ? gmax : last - p;
                if ((shd && np >= 2) || (i8one && np == 1)) {
                    for (int q = 0; q < np; q++) {
                        xs[q] = m->h2 + (size_t)m->sort_itm[p + q] * N;
                        dop[q] = dm[q];
                    }
                    int walked = 0;
                    if (i8d) {
                        moe_panel_fill(&ap, xs, np);
                        walked = !qe_gemv_i8_t(m->d.t_down, dp, &ap, np, dop, run, N, m->ldd_down);
                    }
                    if (!walked)
                        qe_gemv_multi_ptrs_t(m->d.t_down, dp, xs, dop, np, run, N, m->ldd_down);
                    for (int q = 0; q < np; q++) {
                        const int itm = m->sort_itm[p + q];
                        const float w = m->itm_w[itm];
                        float *dst = m->my + (size_t)m->itm_row[itm] * m->my_stride + k0;
                        for (long i = 0; i < run; i++)
                            dst[i] += w * dm[q][i];
                    }
                } else {
                    for (int q = 0; q < np; q++) {
                        const int itm = m->sort_itm[p + q];
                        qe_gemv_t(m->d.t_down, dp, m->h2 + (size_t)itm * N, tmp, run, N,
                                  m->ldd_down);
                        const float w = m->itm_w[itm];
                        float *dst = m->my + (size_t)m->itm_row[itm] * m->my_stride + k0;
                        for (long i = 0; i < run; i++)
                            dst[i] += w * tmp[i];
                    }
                }
                p += np;
            }
        }
    }
}

static void down_multi(void *v, int tid, int nt) {
    moe_job *j = v;
    if (((const struct qe_moe *)j->m)->dyn) {
        down_det(j, tid, nt);
        return;
    }
    const struct qe_moe *m = j->m;
    const long N = m->d.N, K = m->d.K;
    const int T = m->mt_T;
    float *part = m->part2 + (size_t)tid * (size_t)T * K;
    float *tmp = m->tmp + (size_t)tid * K;
    memset(part, 0, (size_t)T * K * sizeof(float));
    long lo, hi;
    moe_unit_slice(m, tid, nt, K, &lo, &hi);
    if (lo >= hi)
        return;
    const float *xs[QE_MOE_SHARED_GROUP];
    float *dop[QE_MOE_SHARED_GROUP];
    float dm[QE_MOE_SHARED_GROUP][QE_MOE_DOWN_BLOCK];
    const int shd = m->shared_multi && qe_gemv_multi_ptrs_has(m->d.t_down);
    const int i8d =
        moe_int8_on() && m->shared_multi && !(N % 32) && qe_gemv_i8_has(m->d.t_down);
    const int i8one = i8d && qe_gemv_i8_has_1(m->d.t_down);
    const qe_i8panel ap = moe_panel(m, 1, tid);
    const int gmax = j->gmax;
    for (long it = lo; it < hi;) {
        const long b = it / K, row0 = it % K;
        long run = K - row0;
        if (it + run > hi)
            run = hi - it;
        const int e = m->bkt_e[b];
        const uint8_t *dp = expert_row(m->d.down, m->d.down_stride, e, row0, m->rb_down);
        const int first = m->bkt_lo[b], last = m->bkt_lo[b + 1];
        for (int p = first; p < last;) {
            const int np = last - p > gmax ? gmax : last - p;
            if ((shd && np >= 2) || (i8one && np == 1)) {
                for (int q = 0; q < np; q++) {
                    xs[q] = m->h2 + (size_t)m->sort_itm[p + q] * N;
                    dop[q] = dm[q];
                }
                int walked = 0;
                if (i8d)
                    moe_panel_fill(&ap, xs, np);
                for (long r0 = 0; r0 < run;) {
                    const long c = run - r0 > QE_MOE_DOWN_BLOCK ? QE_MOE_DOWN_BLOCK : run - r0;
                    const uint8_t *dw = dp + (size_t)r0 * (size_t)m->rb_down;
                    if (i8d)
                        walked = !qe_gemv_i8_t(m->d.t_down, dw, &ap, np, dop, c, N,
                                               m->ldd_down);
                    if (!walked)
                        qe_gemv_multi_ptrs_t(m->d.t_down, dw, xs, dop, np, c, N, m->ldd_down);
                    for (int q = 0; q < np; q++) {
                        const int itm = m->sort_itm[p + q];
                        const float w = m->itm_w[itm];
                        float *dst = part + (size_t)m->itm_row[itm] * K + row0 + r0;
                        for (long i = 0; i < c; i++)
                            dst[i] += w * dm[q][i];
                    }
                    r0 += c;
                }
            } else {
                for (int q = 0; q < np; q++) {
                    const int itm = m->sort_itm[p + q];
                    qe_gemv_t(m->d.t_down, dp, m->h2 + (size_t)itm * N, tmp, run, N,
                              m->ldd_down);
                    const float w = m->itm_w[itm];
                    float *dst = part + (size_t)m->itm_row[itm] * K + row0;
                    for (long i = 0; i < run; i++)
                        dst[i] += w * tmp[i];
                }
            }
            p += np;
        }
        it += run;
    }
}

static void reduce_multi(void *v, int tid, int nt) {
    moe_job *j = v;
    const struct qe_moe *m = j->m;
    if (m->dyn)
        return; /* down_det wrote the answer in place */
    const long K = m->d.K;
    const int T = m->mt_T;
    long lo, hi;
    qe_slice((long)T * K, tid, nt, &lo, &hi);
    for (long it = lo; it < hi; it++) {
        const int t = (int)(it / K);
        const long k = it % K;
        float s = 0;
        for (int th = 0; th < nt; th++)
            s += m->part2[((size_t)th * T + t) * K + k];
        m->my[(size_t)t * m->my_stride + k] = s;
    }
}

/* The three tasks of a grouped pass, without the three dispatches that usually run
 * them, for a caller that has other work for the pool to run beside them: the
 * shared expert of the same chunk in the engine, which is a dense walk of its own
 * and needs nothing the routed half produces. `job` is where a pass's slice of the
 * world lives, so a caller that split a chunk into windows has a description of
 * each. What cannot be duplicated is the moe object's own bucket state, filled
 * here and read by all three tasks, so a graph carries one grouped pass per moe
 * object at a time - which is what a layer has anyway.
 *
 * Everything a pass decides once - that the grouped form applies at all, how the
 * rows sorted into buckets, what the routing statistics say about it - is decided
 * here, on the thread that asked, exactly as it was when the three tasks were
 * dispatched one after another from here. */
int qe_moe_multi_begin(qe_moe *m, const float *x, long x_stride, const qe_route *routes,
                       int routes_per_row, int T, float *y, long y_stride) {
    if (!m || T <= 0 || routes_per_row <= 0)
        return 0;
    if (!moe_need_multi(m, T))
        return 0;
    if (!m->dyn && !moe_need_part2(m, T))
        return 0;
    int nb = moe_group(m, routes, routes_per_row, T);
    if (nb < 0) {
        for (int t = 0; t < T; t++)
            memset(y + (size_t)t * y_stride, 0, (size_t)m->d.K * sizeof(float));
        return 2;
    }
    if (m->hist)
        for (int i = 0; i < T * routes_per_row; i++)
            if (routes[i].expert >= 0 && routes[i].expert < m->d.n_experts)
                m->hist[routes[i].expert]++;
    m->tokens += T;
    m->mx = x;
    m->my = y;
    m->mx_stride = x_stride;
    m->my_stride = y_stride;
    m->mr = routes;
    m->mr_per_row = routes_per_row;
    m->mt_T = T;
    m->mjob = (moe_job){ m, x, routes, T * m->d.top_k, y, moe_shared_group() };
    m->next_gu = 0;
    m->next_dn = 0;
    return 1;
}

void qe_moe_multi_item(qe_moe *m, int which, qe_pool_item *out) {
    out->fn = which == 0 ? gate_up_multi : which == 1 ? down_multi : reduce_multi;
    out->arg = &m->mjob;
    out->nthreads = m->n_threads;
    out->deps = which ? (1u << (unsigned)(which - 1)) : 0;
}

void qe_moe_forward_multi(qe_moe *m, const float *x, long x_stride, const qe_route *routes,
                          int routes_per_row, int T, float *y, long y_stride) {
    if (!m || T <= 0 || routes_per_row <= 0)
        return;
    const int rc = qe_moe_multi_begin(m, x, x_stride, routes, routes_per_row, T, y, y_stride);
    if (rc == 2)
        return; /* zeroed above: the answer a route list with nothing in it gets */
    if (!rc) {
        if (T == 1) {
            qe_moe_forward(m, x, routes, routes_per_row, y);
            return;
        }
        /* No scratch, so no grouping. The per-token pass needs none and is merely
         * slower; returning here instead would leave y holding whatever the
         * previous step wrote there, which is a wrong answer that looks finished
         * and, forty-eight layers later, a silently corrupt trunk. */
        for (int t = 0; t < T; t++)
            qe_moe_forward(m, x + (size_t)t * x_stride, routes + (size_t)t * routes_per_row,
                           routes_per_row, y + (size_t)t * y_stride);
        return;
    }
    /* One dispatch, three items, each waiting on the one whose output it reads.
     * The two hand-backs this removes are the ones §7.8 charges a layer for, and
     * a task that has to wait for its predecessor is no slower for the wait than
     * the thread that spun waiting for it. */
    qe_pool_item items[3];
    for (int i = 0; i < 3; i++)
        qe_moe_multi_item(m, i, &items[i]);
    qe_pool_run_items(items, 3);
}

void qe_moe_set_dyn(qe_moe *m, int on) {
    if (m)
        m->dyn = on ? 1 : 0;
}

int qe_moe_get_dyn(const qe_moe *m) { return m ? m->dyn : 0; }

void qe_moe_step_router_item(qe_moe *m, const float *x, qe_pool_item *out) {
    m->rjob = (router_job){ m, x, m->logits };
    *out = (qe_pool_item){ router_task, &m->rjob, m->n_threads, 0 };
}

/* A decode step's routed experts, handed out by row chunk instead of by fixed slice.
 *
 * The grouped pass slices its work statically, by what a bucket costs, which is right for
 * a chunk whose buckets are many and uneven and wrong for a step whose ten buckets are one
 * row each and whose nineteen threads run on two kinds of core: the P-cores finish their
 * equal share and wait at the barrier for the E-cores (`make decode` prints that wait as
 * the tail, a third of a step before this). Here every thread takes the next chunk of
 * rows from a cursor, so a fast core takes more.
 *
 * The answer does not depend on which thread took what. gate/up writes each expert's
 * intermediate rows in place; down writes each expert's weighted output into that
 * expert's own partial (the K-wide `part` rows, one per route instead of one per thread);
 * and the reduce sums the ten partials in route order. The grouped pass summed per-thread
 * partials, whose grouping depends on the slicing, so the two forms differ by a
 * reassociation and the step is priced in nats (`make score`), not asserted bitwise. */
#define QE_MOE_STEP_GU 32   /* gate/up rows a chunk */
#define QE_MOE_STEP_DN 128  /* down rows a chunk */

static void gate_up_step(void *v, int tid, int nt) {
    qe_moe *m = v;
    const long N = m->d.N, K = m->d.K;
    const long per = (N + QE_MOE_STEP_GU - 1) / QE_MOE_STEP_GU;
    const long units = per * m->st.n;
    const qe_i8panel ap = moe_panel(m, 0, 0); /* filled once, by the master, in begin */
    float g[QE_MOE_STEP_GU], u[QE_MOE_STEP_GU];
    float *gp1[1] = { g }, *up1[1] = { u };
    for (;;) {
        const long c = __atomic_fetch_add(&m->st.next_gu, 1, __ATOMIC_RELAXED);
        if (c >= units)
            return;
        const int e = (int)(c / per);
        const long r0 = (c % per) * QE_MOE_STEP_GU;
        const long n = r0 + QE_MOE_STEP_GU > N ? N - r0 : QE_MOE_STEP_GU;
        const qe_route *rt = m->st.r + e;
        const uint8_t *gw = expert_row(m->d.gate, m->d.gate_stride, rt->expert, r0, m->rb_gate);
        const uint8_t *uw = expert_row(m->d.up, m->d.up_stride, rt->expert, r0, m->rb_up);
        if (m->st.i8) {
            qe_gemv_i8_t(m->d.t_gate, gw, &ap, 1, gp1, n, K, m->ldd_gate);
            qe_gemv_i8_t(m->d.t_up, uw, &ap, 1, up1, n, K, m->ldd_up);
        } else {
            qe_gemv_t(m->d.t_gate, gw, m->st.x, g, n, K, m->ldd_gate);
            qe_gemv_t(m->d.t_up, uw, m->st.x, u, n, K, m->ldd_up);
        }
        qe_silu_mul(m->h + (size_t)e * (size_t)N + r0, g, u, n);
    }
}

static void down_step(void *v, int tid, int nt) {
    qe_moe *m = v;
    const long N = m->d.N, K = m->d.K;
    const long per = (K + QE_MOE_STEP_DN - 1) / QE_MOE_STEP_DN;
    const long units = per * m->st.n;
    const qe_i8panel ap = moe_panel(m, 1, tid);
    float o[QE_MOE_STEP_DN];
    float *op1[1] = { o };
    int have = -1; /* which expert's intermediate this thread's panel holds */
    for (;;) {
        const long c = __atomic_fetch_add(&m->st.next_dn, 1, __ATOMIC_RELAXED);
        if (c >= units)
            return;
        const int e = (int)(c / per);
        const long r0 = (c % per) * QE_MOE_STEP_DN;
        const long n = r0 + QE_MOE_STEP_DN > K ? K - r0 : QE_MOE_STEP_DN;
        const qe_route *rt = m->st.r + e;
        const float *he = m->h + (size_t)e * (size_t)N;
        const uint8_t *dw = expert_row(m->d.down, m->d.down_stride, rt->expert, r0, m->rb_down);
        if (m->st.i8) {
            if (have != e) {
                const float *xs[1] = { he };
                moe_panel_fill(&ap, xs, 1);
                have = e;
            }
            qe_gemv_i8_t(m->d.t_down, dw, &ap, 1, op1, n, N, m->ldd_down);
        } else {
            qe_gemv_t(m->d.t_down, dw, he, o, n, N, m->ldd_down);
        }
        float *dst = m->part + (size_t)e * (size_t)K + r0;
        const float w = rt->weight;
        for (long i = 0; i < n; i++)
            dst[i] = w * o[i];
    }
}

static void reduce_step(void *v, int tid, int nt) {
    qe_moe *m = v;
    const long K = m->d.K;
    long lo, hi;
    qe_slice(K, tid, nt, &lo, &hi);
    for (long k = lo; k < hi; k++) {
        float s = 0.0f;
        for (int e = 0; e < m->st.n; e++)
            s += m->part[(size_t)e * (size_t)K + k];
        m->st.y[k] = s;
    }
}

int qe_moe_step_begin(qe_moe *m, const float *x, qe_route *routes, float *y) {
    const int k = m->d.top_k;
    int n = qe_moe_select(m, m->logits, routes);
    for (int i = n; i < k; i++)
        routes[i].expert = -1;
    qe_moe_refine(m, x, routes, n);
    /* Valid routes first, in the order select left them: that order is the reduce's. */
    int nv = 0;
    for (int i = 0; i < n; i++)
        if (routes[i].expert >= 0)
            routes[nv++] = routes[i];
    for (int i = nv; i < k; i++)
        routes[i].expert = -1;
    if (nv == 0) {
        memset(y, 0, (size_t)m->d.K * sizeof(float));
        return 0;
    }
    /* The per-route partials need a row each and `part` has one per pool thread. */
    if (nv > m->n_threads) {
        qe_moe_forward(m, x, routes, nv, y);
        return 0;
    }
    if (m->hist)
        for (int i = 0; i < nv; i++)
            m->hist[routes[i].expert]++;
    m->tokens++;
    const long K = m->d.K, N = m->d.N;
    m->st.x = x;
    m->st.r = routes;
    m->st.y = y;
    m->st.n = nv;
    m->st.next_gu = 0;
    m->st.next_dn = 0;
    m->st.i8 = moe_int8_on() && m->i8[0].x8 && m->i8[1].x8 && !(K % 32) && !(N % 32) &&
               qe_gemv_i8_has_1(m->d.t_gate) && qe_gemv_i8_has_1(m->d.t_up) &&
               qe_gemv_i8_has_1(m->d.t_down);
    if (m->st.i8) {
        const qe_i8panel ap = moe_panel(m, 0, 0);
        const float *xs[1] = { x };
        moe_panel_fill(&ap, xs, 1);
    }
    return 3;
}

void qe_moe_step_item(qe_moe *m, int which, qe_pool_item *out) {
    out->fn = which == 0 ? gate_up_step : which == 1 ? down_step : reduce_step;
    out->arg = m;
    out->nthreads = m->n_threads;
    out->deps = which ? (1u << (unsigned)(which - 1)) : 0;
}

void qe_moe_layer(qe_moe *m, const float *x, float *y, qe_route *routes, int *n_routes) {
    qe_moe_router(m, x, m->logits);
    qe_route stack64[64];
    qe_route *r = routes;
    if (!r) {
        if (m->d.top_k > 64) {
            r = malloc((size_t)m->d.top_k * sizeof *r);
            if (!r) {
                memset(y, 0, (size_t)m->d.K * sizeof(float));
                if (n_routes)
                    *n_routes = 0;
                return;
            }
        } else {
            r = stack64;
        }
    }
    int n = qe_moe_select(m, m->logits, r);
    qe_moe_refine(m, x, r, n);
    qe_moe_forward(m, x, r, n, y);
    if (n_routes)
        *n_routes = n;
    if (r != routes && r != stack64)
        free(r);
}

const uint64_t *qe_moe_expert_hist(const qe_moe *m, long *n_experts) {
    if (n_experts)
        *n_experts = m->d.n_experts;
    return m->hist;
}

void qe_moe_hist_reset(qe_moe *m) {
    if (m->hist)
        memset(m->hist, 0, (size_t)m->d.n_experts * sizeof *m->hist);
    m->tokens = 0;
}
