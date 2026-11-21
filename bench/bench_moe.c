/* MoE pipeline harness: correctness of the routing/slicing, then bandwidth.

   The pool is built with the checkpoint's real geometry and real dtypes
   (512 experts, N=640, K=2560, gate/up IQ3_S, down IQ4_NL) so the bytes per
   token measured here are the bytes the engine will actually move: 10 experts
   x 2.33 MiB = 23.3 MiB per layer, 1.08 GiB per token across 48 layers.

   Three things are checked, in this order, because each one is a plausible
   silent-wrongness:
     1. top-k selection against a full sort (the incremental version can drop a
        winner if the cutoff is compared before the table is warm)
     2. forward against a serial single-thread reference (row slicing across
        20 threads, expert strides, and the weighted reduction all have to agree)
     3. prefetch on/off, which must not change the numbers at all - if it does,
        the cursor is reading bytes it does not own

   Then throughput against the wall an expert read actually runs into - 76-79 GB/s
   for the same bytes at 19 threads, 85.5 for a flat read on eight performance cores
   (`make alu`) - with and without software prefetch, which is the only honest way to
   decide whether the prefetch code stays.
*/
#include "qe.h"
#include "kern.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define E_ 512 /* experts */
#define K_ 2560
#define N_ 640
#define TOPK 10
#define LAYERS 48

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t rnd(uint64_t *s) {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    return *s * 0x2545f4914f6cdd1dull;
}

static size_t rb_of(qe_type t, long cols) {
    const qe_type_info *ti = qe_type_info_of(t);
    return (size_t)(cols / (long)ti->block_ne * (long)ti->block_sz);
}

/* Naive reference: one expert at a time, in order, no threads. Uses the same
 * GEMV kernels, so it validates the pipeline and not the kernels. */
static void ref_forward(const qe_moe_desc *d, const qe_route *r, int n, const float *x, float *y) {
    static float h[N_];
    static float g[N_], u[N_], part[K_];
    memset(y, 0, (size_t)K_ * sizeof *y);
    for (int e = 0; e < n; e++) {
        const uint8_t *gp = (const uint8_t *)d->gate + (size_t)r[e].expert * d->gate_stride;
        const uint8_t *up = (const uint8_t *)d->up + (size_t)r[e].expert * d->up_stride;
        const uint8_t *dp = (const uint8_t *)d->down + (size_t)r[e].expert * d->down_stride;
        qe_gemv_t(d->t_gate, gp, x, g, N_, K_, K_ / 256);
        qe_gemv_t(d->t_up, up, x, u, N_, K_, K_ / 256);
        for (long i = 0; i < N_; i++)
            h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        qe_gemv_t(d->t_down, dp, h, part, K_, N_, N_ / 32);
        for (long k = 0; k < K_; k++)
            y[k] += r[e].weight * part[k];
    }
}

/* Kernel-only floor: the same bytes through the same kernels with none of the
 * pipeline, so the expert pass can be attributed instead of assumed away.
 * qe_moe_forward is gate/up + silu + down + per-thread partials + reduce, and what
 * it costs over its own kernels needed four questions asked one at a time. Is it
 * the pipeline? No: 6% of the pass. Is it walking ten scattered 0.7 MiB spans
 * instead of one long run, or a 640-column row instead of a 2560-column one? No:
 * within 2% of a contiguous run of the same experts, and re-tiling the down matrix
 * changes nothing. Is it the length of a dispatch? Only for IQ4_NL, and only 61 ->
 * 69 GB/s from 9 MiB to 141 MiB. Is it how the rows are divided between threads?
 * Yes: 12-15% at every width and both types, which is the table at the end of this
 * section, and the reason the pool shares rows equally. */
typedef struct {
    const uint8_t *W;
    long stride;        /* bytes between experts */
    const int32_t *ex;  /* expert chosen per route */
    const float *x;
    float *y;           /* n * rows scratch */
    long n, rows, cols, ldd, rb;
    qe_type t;
    int flat;  /* the experts are consecutive: one matrix, one call per thread */
    int equal; /* divide rows evenly instead of asking the pool's policy */
} floor_job;

static void floor_task(void *v, int tid, int nt) {
    floor_job *j = v;
    const long total = j->n * j->rows;
    long lo, hi;
    if (j->equal) {
        const long per = (total + nt - 1) / nt;
        lo = (long)tid * per;
        hi = lo + per > total ? total : lo + per;
    } else {
        qe_slice(total, tid, nt, &lo, &hi);
    }
    if (lo >= hi)
        return;
    if (j->flat) {
        qe_gemv_t(j->t, j->W + (size_t)j->ex[0] * (size_t)j->stride + (size_t)lo * (size_t)j->rb,
                  j->x, j->y + lo, hi - lo, j->cols, j->ldd);
        return;
    }
    for (long it = lo; it < hi;) {
        const long e = it / j->rows, row0 = it % j->rows;
        long run = j->rows - row0;
        if (it + run > hi) run = hi - it;
        qe_gemv_t(j->t, j->W + (size_t)j->ex[e] * (size_t)j->stride + (size_t)row0 * (size_t)j->rb,
                  j->x, j->y + it, run, j->cols, j->ldd);
        it += run;
    }
}

/* The expert pool walked with nothing expert about it: every expert visited once, each
 * against a contiguous run of its own token rows. `fill` is the window's own histogram or
 * NULL for `muni` rows each, and `wrap` folds the visit onto the first `wrap` matrices
 * when the caller has only a sample of the format resident. The MAC count is sum(fill) x
 * rows x cols either way, which is the whole point: the arithmetic is the routed pass's
 * arithmetic with the gather, the bucketing and the silu taken out of it. Token offsets
 * are the caller's prefix sums, so a thread starting mid-slice does not have to walk the
 * histogram.
 *
 * What this walk is compared against has to be the same call the engine makes, or the
 * answer is about kernels and not about layout, so the walk is shaped by the same three
 * numbers the grouped pass shapes its calls with: `chunk_r` output rows of one matrix per
 * call (QE_MOE_CHUNK there), up to `maxr` token rows per call (the shared group there),
 * and `nblk` output-row items per matrix, which is the pool's unit of work and exists so
 * that a leg with ten matrices in it still occupies nineteen threads. Setting chunk_r or
 * maxr to zero asks for the whole matrix and the whole visit in one call, which is what a
 * dense projection does and is a different question, asked on its own leg. */
#define REF_MAXG 512 /* token rows handed to one walk: the widest window this file runs */

typedef struct {
    const uint8_t *W[3];
    long stride[3], rb[3], rows[3], cols[3], ldd[3];
    qe_type t[3];
    int nmat;
    const float *x;
    float *y;
    long xstride, ystride, ntok;
    const int32_t *fill;
    const long *tokoff;
    int muni, wrap, nexp;
    int nblk, chunk_r, maxr;
} ref_job;

static void ref_task(void *v, int tid, int nt) {
    const ref_job *j = v;
    const long nblk = j->nblk > 0 ? j->nblk : 1;
    const int maxr = j->maxr > 0 ? (j->maxr > REF_MAXG ? REF_MAXG : j->maxr) : REF_MAXG;
    long lo, hi;
    qe_slice((long)j->nexp * nblk, tid, nt, &lo, &hi);
    const float *xs[REF_MAXG];
    float *ys[REF_MAXG];
    for (long it = lo; it < hi; it++) {
        const long i = it / nblk;
        const int blk = (int)(it % nblk);
        const int e = j->wrap ? (int)(i % j->wrap) : (int)i;
        const int f = j->fill ? j->fill[i] : j->muni;
        if (f <= 0)
            continue;
        const long to = j->tokoff ? j->tokoff[i] : (long)i * j->muni;
        const float *x = j->x + (size_t)(to % j->ntok) * (size_t)j->xstride;
        float *y = j->y + (size_t)to * (size_t)j->ystride;
        for (int w = 0; w < j->nmat; w++) {
            const long rsz = j->rows[w] / nblk;
            const long r0 = (long)blk * rsz;
            const long r1 = blk + 1 == nblk ? j->rows[w] : r0 + rsz;
            const uint8_t *wp =
                j->W[w] + (size_t)e * (size_t)j->stride[w] + (size_t)r0 * (size_t)j->rb[w];
            float *yp = y + r0;
            const long nr = r1 - r0;
            for (long d = 0; d < nr;) {
                const long c = j->chunk_r > 0 && nr - d > j->chunk_r ? j->chunk_r : nr - d;
                for (int q = 0; q < f;) {
                    const int n = f - q > maxr ? maxr : f - q;
                    if (n >= 2 && qe_gemv_multi_ptrs_has(j->t[w])) {
                        for (int q2 = 0; q2 < n; q2++) {
                            xs[q2] = x + (size_t)(q + q2) * (size_t)j->xstride;
                            ys[q2] = yp + (size_t)d +
                                     (size_t)(q + q2) * (size_t)j->ystride;
                        }
                        qe_gemv_multi_ptrs_t(j->t[w], wp + (size_t)d * (size_t)j->rb[w], xs,
                                             ys, n, c, j->cols[w], j->ldd[w]);
                    } else if (n >= 2) {
                        qe_gemv_multi_t(j->t[w], wp + (size_t)d * (size_t)j->rb[w],
                                        x + (size_t)q * (size_t)j->xstride,
                                        yp + (size_t)d, c, j->cols[w], j->ldd[w], n,
                                        j->xstride, j->ystride, (size_t)j->rb[w], 0);
                    } else {
                        qe_gemv_t(j->t[w], wp + (size_t)d * (size_t)j->rb[w], x, yp + (size_t)d,
                                  c, j->cols[w], j->ldd[w]);
                    }
                    q += n;
                }
                d += c;
            }
        }
    }
}

static int cmp_i32(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int cmp_desc(const void *a, const void *b) {
    float x = ((const qe_route *)a)->logit, y = ((const qe_route *)b)->logit;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    qe_fuse(argc > 1 ? atoi(argv[1]) : 120);
    /* The pool this bench measures through. A pool as wide as the cpu count makes
     * even the narrow rows of the sweep below pay the spinning-worker penalty
     * described in QE_THREADS_AUTO, because the workers it is not using are still
     * spinning, so the default is the engine's own width and the sweep is relative
     * to it. Pass a number to measure a specific pool instead. */
    int nthr = argc > 2 ? (!strcmp(argv[2], "auto") ? QE_THREADS_AUTO : atoi(argv[2]))
                        : QE_THREADS_AUTO;

    const size_t gsz = rb_of(QE_IQ3_S, K_) * N_;
    const size_t dsz = rb_of(QE_IQ4_NL, N_) * K_;
    printf("expert: gate %zu KiB + up %zu KiB + down %zu KiB = %.2f MiB; pool %.2f GiB\n",
           gsz >> 10, gsz >> 10, dsz >> 10, (2 * gsz + dsz) / 1048576.0,
           E_ * (2 * gsz + dsz) / 1073741824.0);

    uint8_t *gate = qe_alloc_huge(gsz * E_);
    uint8_t *up = qe_alloc_huge(gsz * E_);
    uint8_t *down = qe_alloc_huge(dsz * E_);
    float *router = qe_alloc_huge(sizeof(float) * E_ * K_);
    if (!gate || !up || !down || !router) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    float *x = aligned_alloc(64, (size_t)K_ * sizeof *x);
    float *y = aligned_alloc(64, (size_t)K_ * sizeof *y);
    float *yr = aligned_alloc(64, (size_t)K_ * sizeof *yr);
    float *logits = aligned_alloc(64, (size_t)E_ * sizeof *logits);
    qe_route *routes = malloc(TOPK * sizeof *routes);
    qe_route *sorted = malloc(E_ * sizeof *sorted);
    uint64_t s = 0x243f6a8885a308d3ull;

    /* Arbitrary bytes: every IQ3_S and IQ4_NL pattern decodes. The fp16
     * super-scale of each block is forced finite so the reference cannot NaN
     * its way into agreement. */
    for (int e = 0; e < E_; e++)
        for (int which = 0; which < 3; which++) {
            uint8_t *p = which == 0 ? gate + (size_t)e * gsz
                     : which == 1 ? up + (size_t)e * gsz
                                  : down + (size_t)e * dsz;
            size_t n = which == 2 ? dsz : gsz;
            for (size_t i = 0; i < n; i++)
                p[i] = (uint8_t)(rnd(&s) >> 32);
            long block = which == 2 ? 18 : 110;
            for (size_t off = 0; off + 2 <= n; off += (size_t)block) {
                uint16_t h = 0x2c00u | (uint16_t)(rnd(&s) & 0x1ff);
                memcpy(p + off, &h, 2);
            }
        }
    for (long i = 0; i < (long)E_ * K_; i++)
        router[i] = (float)((double)(rnd(&s) % 20000) / 10000.0 - 1.0) * 0.02f;
    for (long i = 0; i < K_; i++)
        x[i] = (float)sin(i * 0.0017) * 0.5f;

    qe_pool_init(nthr, NULL, 0);

    qe_moe_desc d = {
        .gate = gate,
        .up = up,
        .down = down,
        .gate_stride = (long)gsz,
        .up_stride = (long)gsz,
        .down_stride = (long)dsz,
        .t_gate = QE_IQ3_S,
        .t_up = QE_IQ3_S,
        .t_down = QE_IQ4_NL,
        .router = router,
        .n_experts = E_,
        .N = N_,
        .K = K_,
        .top_k = TOPK,
        .n_threads = 0,
    };

    /* ---- 1. selection against a full sort, over many random logits ---- */
    qe_moe *m = qe_moe_create(&d);
    if (!m) {
        fprintf(stderr, "qe_moe_create: %s\n", strerror(errno));
        return 1;
    }
    int sel_bad = 0;
    for (int trial = 0; trial < 500; trial++) {
        for (long i = 0; i < E_; i++)
            logits[i] = (float)((double)(rnd(&s) % 200000) / 1000.0 - 100.0);
        int n = qe_moe_select(m, logits, routes);
        for (long i = 0; i < E_; i++) {
            sorted[i].logit = logits[i];
            sorted[i].expert = (int32_t)i;
        }
        qsort(sorted, E_, sizeof *sorted, cmp_desc);
        for (int i = 0; i < n; i++)
            if (sorted[E_ - 1 - i].logit != routes[n - 1 - i].logit)
                sel_bad++;
        double wsum = 0;
        for (int i = 0; i < n; i++)
            wsum += routes[i].weight;
        if (fabs(wsum - 1.0) > 1e-5)
            sel_bad++;
    }
    printf("top-k over 500 random router vectors: %s (%d failures)\n",
           sel_bad ? "FAIL" : "PASS", sel_bad);

    /* ---- 1b. the bf16 router shadow. The picks may come from the shadow,
     * because that is what halves the router read; the mixing weights may not,
     * and qe_moe_route re-reads the k winners at f32 so they are exact. The
     * weights are what is asserted here, matched by expert rather than by slot:
     * two routers that picked the same ten experts can list them in a different
     * order when a tie is decided differently.
     *
     * The engine's own pair - qe_moe_select against the shadow plus
     * qe_moe_refine, which is what qe_moe_layer and a chunk's routed half run -
     * is measured the same way, because that is the composition that decides
     * what an expert's output is scaled by in a real step.
     *
     * The expert sets are reported, not asserted. These logits are synthetic
     * and deliberately full of near-ties, where an 8-bit mantio genuinely does
     * move a pick; what the checkpoint's own routers do is bin/bench_route's
     * measurement, and it says 0 differing sets in 96000 pairs. ---- */
    uint16_t *router_b = malloc(sizeof *router_b * (size_t)E_ * K_);
    float *x_keep = malloc((size_t)K_ * sizeof *x_keep);
    int bcode = router_b && x_keep ? 0 : 1;
    double worst_w_refined = 0, worst_w_shadow = 0, worst_w_engine = 0;
    int bsets = 0, btrials = 300;
    qe_moe *mb = NULL;
    if (router_b && x_keep) {
        memcpy(x_keep, x, (size_t)K_ * sizeof *x);
        qe_bf16_from_f32(router, router_b, (size_t)E_ * K_);
        qe_moe_desc db = d;
        db.router_bf16 = router_b;
        mb = qe_moe_create(&db);
        if (!mb)
            bcode = 1;
    }
    if (mb) {
        /* A private stream: this section must not shift the random draws of the
         * sections below, or their results move for a reason unrelated to them. */
        uint64_t sb = 0x9e3779b97f4a7c15ull;
        for (int trial = 0; trial < btrials; trial++) {
            for (long i = 0; i < K_; i++)
                x[i] = (float)sin((i + trial * 7) * 0.0017) * 0.5f +
                       (float)((double)(rnd(&sb) % 2000) / 1000.0 - 1.0) * 0.01f;
            qe_route r32[TOPK], rsh[TOPK], rpl[TOPK], rlay[TOPK];
            qe_moe_router(m, x, logits); /* f32 router: the reference */
            int n32 = qe_moe_select(m, logits, r32);
            int nsh = qe_moe_route(mb, x, logits, rsh);
            /* logits still holds the shadow's own values: the refinement wrote
             * into the route table, not back into the vector */
            int npl = qe_moe_select(mb, logits, rpl);
            /* What a forward pass does: select from the shadow, then refine. A
             * chunk and a lone token share this pair, so it is the pair under
             * test rather than qe_moe_route, which is the same two calls. */
            int nly = qe_moe_select(mb, logits, rlay);
            qe_moe_refine(mb, x, rlay, nly);
            if (n32 != nsh || n32 != npl || n32 != nly)
                bcode++;
            int a[TOPK], b[TOPK];
            for (int i = 0; i < n32; i++) {
                a[i] = r32[i].expert;
                b[i] = rsh[i].expert;
            }
            qsort(a, n32, sizeof *a, cmp_i32);
            qsort(b, n32, sizeof *b, cmp_i32);
            int same = 1;
            for (int i = 0; i < n32; i++)
                same = same && a[i] == b[i];
            if (!same) {
                bsets++;
                continue;
            }
            for (int i = 0; i < n32; i++) {
                int j = -1;
                for (int p = 0; p < n32; p++)
                    if (rsh[p].expert == r32[i].expert)
                        j = p;
                if (j < 0) {
                    bcode++;
                    continue;
                }
                double dw = fabs(rsh[j].weight - r32[i].weight);
                if (dw > worst_w_refined)
                    worst_w_refined = dw;
                double best = 1e30;
                for (int p = 0; p < n32; p++)
                    if (rpl[p].expert == r32[i].expert) {
                        double dp = fabs(rpl[p].weight - r32[i].weight);
                        if (dp < best)
                            best = dp;
                    }
                if (best < 1e30 && best > worst_w_shadow)
                    worst_w_shadow = best;
                double we = 1e30;
                for (int p = 0; p < n32; p++)
                    if (rlay[p].expert == r32[i].expert) {
                        double dp = fabs(rlay[p].weight - r32[i].weight);
                        if (dp < we)
                            we = dp;
                    }
                if (we < 1e30 && we > worst_w_engine)
                    worst_w_engine = we;
            }
        }
        /* The chunk form has to agree with the row form to the bit, or a width-1
         * chunk and a width-5 one are two models. Stacked activations at a stride,
         * a top-k per row, and then the two refinements compared. */
        {
            enum { RT = 5 };
            float *xr = malloc((size_t)RT * K_ * sizeof *xr);
            qe_route *ra = malloc((size_t)RT * TOPK * sizeof *ra);
            qe_route *rb = malloc((size_t)RT * TOPK * sizeof *rb);
            int okr = xr && ra && rb;
            for (int t = 0; t < RT && okr; t++)
                for (long i = 0; i < K_; i++)
                    xr[(size_t)t * K_ + i] = (float)sin((i + t * 977) * 0.0013) * 0.4f;
            for (int t = 0; t < RT && okr; t++) {
                qe_moe_router(mb, xr + (size_t)t * K_, logits);
                if (qe_moe_select(mb, logits, ra + (size_t)t * TOPK) != TOPK)
                    okr = 0;
            }
            if (okr) {
                memcpy(rb, ra, (size_t)RT * TOPK * sizeof *rb);
                qe_moe_refine_rows(mb, xr, K_, rb, TOPK, RT);
                for (int t = 0; t < RT && okr; t++) {
                    qe_moe_refine(mb, xr + (size_t)t * K_, ra + (size_t)t * TOPK, TOPK);
                    for (int i = 0; i < TOPK; i++)
                        if (rb[(size_t)t * TOPK + i].weight != ra[(size_t)t * TOPK + i].weight ||
                            rb[(size_t)t * TOPK + i].expert != ra[(size_t)t * TOPK + i].expert)
                            okr = 0;
                }
            }
            if (!okr)
                bcode++;
            printf("router chunk form : %s (%d rows at stride %d refined across the pool,"
                   " equal to the row form)\n", okr ? "pass" : "FAIL", (int)RT, (int)K_);
            free(xr);
            free(ra);
            free(rb);
        }
        memcpy(x, x_keep, (size_t)K_ * sizeof *x);
        printf("bf16 router shadow  : %s (%d activations, %d of them picking different experts"
               " on synthetic near-ties; weights for a shared pick off by %.1e, and %.1e without"
               " the f32 re-read; the engine's select+refine pair off by %.1e)\n",
               (bcode || worst_w_refined > 1e-6 || worst_w_engine > 1e-6) ? "FAIL" : "pass",
               btrials, bsets, worst_w_refined, worst_w_shadow, worst_w_engine);
    } else {
        printf("bf16 router shadow  : FAIL (no descriptor)\n");
    }
    if (mb)
        qe_moe_destroy(mb);
    free(router_b);
    free(x_keep);

    /* ---- 2. forward against the serial reference ---- */
    qe_moe_router(m, x, logits);
    int n = qe_moe_select(m, logits, routes);
    qe_moe_forward(m, x, routes, n, y);
    ref_forward(&d, routes, n, x, yr);
    double worst = 0, denom = 0;
    int nonfinite = 0;
    for (long k = 0; k < K_; k++)
        denom += fabs((double)yr[k]);
    denom /= K_;
    for (long k = 0; k < K_; k++) {
        if (!(y[k] == y[k]) || !(yr[k] == yr[k])) {
            nonfinite++;
            continue;
        }
        double rel = fabs((double)yr[k] - y[k]) / (denom > 1e-9 ? denom : 1e-9);
        if (rel > worst) worst = rel;
    }
    if (nonfinite) {
        printf("forward vs serial reference: %d non-finite outputs -> FAIL\n", nonfinite);
        worst = 1e30;
    } else
        printf("forward vs serial reference: worst %.1e (avg |y| %.3g) -> %s\n", worst, denom,
               worst < 1e-4 ? "PASS" : "FAIL");

    /* ---- 3. throughput ----
     * Routes are redrawn every iteration. Keeping them fixed would measure the
     * 22 MiB of one expert set being re-read from L3, which is not a bandwidth
     * number the engine will ever see.
     *
     * The thread sweep is the shape a layer sees: top_k experts, one expert pool
     * the size of the checkpoint's, one dispatch each for gate/up and down, and
     * routes redrawn per iteration. What it does NOT settle is how the rows are
     * divided, because that is the pool's policy and this table can only read it,
     * never race it - `qe_slice_summary` in the header says which one is in force.
     * The dispatch-width table in section 3b is the race, and it is the reason the
     * pool shares rows equally now.
     *
     * Read this table and the engine in the same breath. The expert round is the
     * one shape in a step whose cost per byte is high enough - two quantised
     * decoders, a silu, and a cross-thread reduce per row - that weighting rows by
     * cpu_capacity moves it, and it moves it the wrong way relative to the step:
     * the round prefers weights by ~10% while the step prefers equal shares by 4%,
     * because the step is 5.4 GiB of bandwidth and the round is 22 MiB of ALU. The
     * step is what ships, so a lower number here after the slicing change is not a
     * regression in it. Medians of five sweeps (QE_MOE_REPEATS=5) at 4 / 8 / 12 /
     * 19 threads, best column, equal shares: 12.5 / 24.8 / 34.4 / 39.8 GB/s. */
    const double exp_bytes = TOPK * (2.0 * gsz + dsz);
    const double router_bytes = (double)E_ * K_ * 4;
    printf("\npipeline: %.2f MiB of expert bytes + %.1f MiB router per layer\n",
           exp_bytes / 1048576.0, router_bytes / 1048576.0);
    /* One sample per thread count cannot order two counts whose spread
     * overlaps, which is precisely how "8 beats 20" survived for a session.
     * QE_MOE_REPEATS=n sweeps n times and reports best/median/worst, so the
     * default thread count is chosen on a distribution. */
    int reps = 1;
    {
        const char *r = getenv("QE_MOE_REPEATS");
        if (r) {
            reps = atoi(r);
            if (reps < 1) reps = 1;
            if (reps > 16) reps = 16;
        }
    }
    printf("%-8s %9s %9s %9s  %s\n", "threads", "best GB/s", "median", "worst",
           reps > 1 ? "per repeat" : "(set QE_MOE_REPEATS=5 to see the spread)");
    char charbuf[96];
    printf("slice: %s\n", qe_slice_summary(charbuf, sizeof charbuf));
    const int cands[4] = { qe_pool_size(), 12, 8, 4 };
    int THR[4], nsweep = 0;
    for (int i = 0; i < 4; i++)
        if (cands[i] >= 1 && cands[i] <= qe_pool_size() &&
            (nsweep == 0 || cands[i] != THR[nsweep - 1]))
            THR[nsweep++] = cands[i];
    for (int ti = 0; ti < nsweep; ti++) {
        qe_moe_desc dt = d;
        dt.n_threads = THR[ti];
        qe_moe *mt = qe_moe_create(&dt);
        double gb[16], ms[16], rt[16];
        for (int rep = 0; rep < reps; rep++) {
            double best_e = 1e30, best_r = 1e30;
            const int iters = 40;
            for (int pass = 0; pass < 3; pass++) {
                double t0 = now(), t1;
                for (int i = 0; i < iters; i++) {
                    for (int e = 0; e < TOPK; e++) {
                        routes[e].expert = (int32_t)(rnd(&s) % E_);
                        routes[e].weight = 1.0f / TOPK;
                    }
                    qe_moe_forward(mt, x, routes, TOPK, y);
                }
                t1 = now();
                for (int i = 0; i < iters; i++) {
                    qe_moe_router(mt, x, logits);
                    qe_moe_select(mt, logits, routes);
                }
                double tr = (now() - t1) / iters;
                if ((t1 - t0) / iters < best_e) best_e = (t1 - t0) / iters;
                if (tr < best_r) best_r = tr;
            }
            gb[rep] = exp_bytes / best_e / 1e9;
            ms[rep] = best_e;
            rt[rep] = best_r;
        }
        /* median by insertion sort on a copy (n <= 16) */
        double srt[16];
        for (int i = 0; i < reps; i++) srt[i] = gb[i];
        for (int i = 1; i < reps; i++) {
            double k = srt[i];
            int j = i - 1;
            while (j >= 0 && srt[j] > k) { srt[j + 1] = srt[j]; j--; }
            srt[j + 1] = k;
        }
        double best = srt[reps - 1], lo = srt[0], med = srt[reps / 2];
        printf("%-8d %9.1f %9.1f %9.1f  ", THR[ti], best, med, lo);
        fflush(stdout);
        if (reps > 1)
            for (int i = 0; i < reps; i++) printf("%.0f ", gb[i]);
        else
            printf("ms/experts %.3f  ms/router %.3f  tok/s @48 %.1f", ms[0] * 1e3, rt[0] * 1e3,
                   1.0 / ((ms[0] + rt[0]) * LAYERS));
        putchar('\n');
        qe_moe_destroy(mt);
    }

    /* ---- 3b. where the gap to the kernels actually is ----
     * bench_kern streams one 384 MiB matrix at K=2560 and calls gemv once per
     * thread. The expert pass calls it per chunk of rows, over matrices whose
     * second dimension is 640 rather than 2560, and its ten experts are scattered
     * through a 1.1 GiB pool. Which of those cost the 33% this pass was once
     * thought to be off its own kernels is not guessable, so each is measured with
     * the same pool, the same routes and the same slicing:
     * kernels only, no silu, no per-thread partials, no reduce. The contiguous
     * layout separates the access pattern from everything else, and the last row
     * is a control: the down pool read as 2560-column rows, which is the same
     * bytes through the same kernel with a longer row, so a difference there is
     * the kernel's per-row cost and nothing else. Routes advance by TOPK per
     * iteration so no iteration re-reads what the last one left in L3. */
    {
        float *fy = malloc((size_t)E_ * K_ * sizeof *fy);
        int32_t *fx = malloc((size_t)E_ * sizeof *fx);
        /* Expert order for the two layouts this table races. Adjacent experts are
         * the best case a router could ever hand the engine; a fixed shuffle is a
         * real router's worst case, and it is fixed and covering - 51 iterations
         * read all 512 experts once - because drawing ten experts from a running
         * PRNG inside the timed loop made this table swing 20% between runs of the
         * same binary, which is more than the effect it is here to measure. */
        int32_t *perm = malloc((size_t)E_ * sizeof *perm);
        uint64_t sp = 0x9e3779b97f4a7c15ull;
        for (int e = 0; e < E_; e++)
            perm[e] = e;
        for (int e = E_ - 1; e > 0; e--) {
            sp ^= sp >> 12;
            sp ^= sp << 25;
            sp ^= sp >> 27;
            const int j = (int)(sp % (uint64_t)(e + 1));
            const int32_t t = perm[e];
            perm[e] = perm[j];
            perm[j] = t;
        }
        const long ldd_g = K_ / (long)qe_type_info_of(QE_IQ3_S)->block_ne;
        const long ldd_d = N_ / (long)qe_type_info_of(QE_IQ4_NL)->block_ne;
        /* The down pool re-tiled as 640 rows of K=2560 per expert: identical
         * bytes, identical pool, only the row length differs. */
        const qe_type_info *i4 = qe_type_info_of(QE_IQ4_NL);
        const long dcols = K_;
        const long drb = dcols / (long)i4->block_ne * (long)i4->block_sz;
        const long drows = (long)dsz / drb;
        floor_job fj[4] = {
            { gate, d.gate_stride, fx, x, fy, TOPK, N_, K_, ldd_g, (long)(gsz / N_), QE_IQ3_S, 0,
              0 },
            { up, d.up_stride, fx, x, fy, TOPK, N_, K_, ldd_g, (long)(gsz / N_), QE_IQ3_S, 0, 0 },
            { down, d.down_stride, fx, x, fy, TOPK, K_, N_, ldd_d, (long)(dsz / K_), QE_IQ4_NL, 0,
              0 },
            { down, d.down_stride, fx, x, fy, TOPK, drows, dcols, dcols / (long)i4->block_ne, drb,
              QE_IQ4_NL, 0, 0 },
        };
        struct { const char *name; int mask, blocked; } fl[] = {
            { "gate  IQ3_S 640x2560", 1, 0 },
            { "up    IQ3_S 640x2560", 2, 0 },
            { "down  IQ4_NL 2560x640", 4, 0 },
            { "all three, scattered", 7, 0 },
            { "all three, adjacent", 7, 1 },
            { "down pool as K=2560", 8, 0 },
        };
        printf("\nkernel-only floor, %d threads (compare with the table above)\n", qe_pool_size());
        printf("  %-26s %8s %8s %8s %8s\n", "layout", "ms", "median", "worst", "GB/s");
        /* Interleaved, one round of each layout per round: the question this table
         * asks is whether a whole is more than the sum of its rows, and a block
         * spent measuring one layout drifts enough on this host to answer it.
         *
         * It is not, and best/median/worst are printed because that answer only
         * appears once the cell with two modes is shown with both. The two
         * single-matrix rows read 6.9 MiB per timed unit while touching all 343 MiB
         * of one pool in 51 dispatches, and they sit in one of two modes: 0.136-0.142
         * ms (49.8-51.8 GB/s, bench_kern's ceiling) or 0.200-0.214 ms (33.5-35.3),
         * with a run picking a mode for the whole of its first two rows and the
         * median agreeing with the mode it landed in. The 22.7 MiB rows move 3%.
         * Taking the best of each row therefore compares a lucky 0.434 against an
         * unlucky-but-typical 0.543 and reports a 22% composition penalty that does
         * not exist: median against median is 0.570-0.575 of gate+up+down alone
         * against 0.539-0.556 for all three together, i.e. the whole is 96% of its
         * rows, and the only real gap left is the one the engine's own 0.589 ms
         * shows against the 0.546 ms of the same bytes through the kernels with no
         * silu, no per-thread partials and no reduce - the pipeline, 6-8%. */
        const unsigned nfl = sizeof fl / sizeof fl[0];
        const int frounds = 7;
        double fbest[16], fmed[16], fworst[16], fbytes[16], ft[16][16];
        for (unsigned fi = 0; fi < nfl; fi++) {
            fbest[fi] = 1e30;
            fbytes[fi] = 0;
            for (int mi = 0; mi < 3; mi++)
                if (fl[fi].mask & (1 << mi))
                    fbytes[fi] += (double)(mi < 2 ? gsz : dsz) * TOPK;
            if (fl[fi].mask & 8) fbytes[fi] += (double)dsz * TOPK;
        }
        for (int rep = 0; rep < frounds; rep++)
            for (unsigned fi = 0; fi < nfl; fi++) {
                double t0 = now();
                const int iters = E_ / TOPK;
                for (int i = 0; i < iters; i++) {
                    for (int e = 0; e < TOPK; e++)
                        fx[e] = fl[fi].blocked ? (int32_t)((i * TOPK + e) % E_) : perm[(i * TOPK + e) % E_];
                    for (int mi = 0; mi < 4; mi++)
                        if (fl[fi].mask & (1 << mi))
                            qe_pool_run(floor_task, &fj[mi], qe_pool_size());
                }
                double t = (now() - t0) / iters;
                ft[fi][rep] = t;
                if (t < fbest[fi]) fbest[fi] = t;
            }
        double fsum = 0, fmedsum = 0, fsumw = 0;
        for (unsigned fi = 0; fi < nfl; fi++) {
            for (int a = 1; a < frounds; a++) {      /* insertion sort, n <= 16 */
                double v = ft[fi][a];
                int b = a - 1;
                while (b >= 0 && ft[fi][b] > v) {
                    ft[fi][b + 1] = ft[fi][b];
                    b--;
                }
                ft[fi][b + 1] = v;
            }
            fmed[fi] = ft[fi][frounds / 2];
            fworst[fi] = ft[fi][frounds - 1];
            printf("  %-26s %8.3f %8.3f %8.3f %8.1f\n", fl[fi].name, fbest[fi] * 1e3,
                   fmed[fi] * 1e3, fworst[fi] * 1e3, fbytes[fi] / fbest[fi] / 1e9);
            fflush(stdout);
            if (fi < 3) fsum += fbest[fi], fmedsum += fmed[fi], fsumw += fworst[fi];
        }
        printf("  %-26s %8.3f %8.3f %8.3f\n", "sum of the three alone", fsum * 1e3,
               fmedsum * 1e3, fsumw * 1e3);
        /* Two things are left once the pipeline and the layout are out of the
         * way: how long a dispatch runs for, and how its rows are handed out.
         * "pool split" is what qe_slice gives as configured (equal shares by
         * default, so the first two columns are the same number until this binary
         * runs under QE_SLICE=capacity), "even split" is an explicit even division
         * of the same rows, and the last two are bench_kern's own calling pattern -
         * one contiguous matrix, one gemv call per thread, rows divided evenly - so
         * that cell is the rate bench_kern's per-type table may be compared
         * against and the first is what the expert pass can actually have,
         * because its ten experts are chosen by the router. The four modes are
         * measured interleaved, one round of each, because on a host this shared
         * a block spent measuring one mode drifts by more than the effect being
         * looked for. */
        printf("\n  dispatch width and row split, contiguous experts (GB/s)\n");
        printf("  (\"pool split\" is qe_slice as configured, so with the default this table\n"
               "   reads one number per row; run it under QE_SLICE=capacity to see the two)\n");
        printf("  %-22s %11s %11s %11s %11s\n", "bytes per dispatch", "pool split",
               "even split", "one matrix", "matrix+even");
        const int widths[3] = { 10, 40, 160 };
        for (int which = 0; which < 2; which++) {
            const double per_exp = which ? (double)dsz : 2.0 * (double)gsz;
            const int nm = which ? 1 : 2;
            printf("  %s\n", which ? "down IQ4_NL 2560x640" : "gate+up IQ3_S 2 x 640x2560");
            for (unsigned wi = 0; wi < sizeof widths / sizeof widths[0]; wi++) {
                const int w = widths[wi];
                floor_job base[2] = { fj[which ? 2 : 0], fj[1] };
                const int iters = E_ / w;
                double best[4] = { 1e30, 1e30, 1e30, 1e30 };
                for (int rep = 0; rep < 5; rep++)
                    for (int md = 0; md < 4; md++) {
                        floor_job j2[2] = { base[0], base[1] };
                        for (int q = 0; q < nm; q++) {
                            j2[q].n = w;
                            j2[q].flat = md >> 1;
                            j2[q].equal = md & 1;
                        }
                        double t0 = now();
                        for (int i = 0; i < iters; i++) {
                            for (int e = 0; e < w; e++)
                                fx[e] = (int32_t)(i * w + e);
                            for (int q = 0; q < nm; q++)
                                qe_pool_run(floor_task, &j2[q], qe_pool_size());
                        }
                        double t = (now() - t0) / iters;
                        if (t < best[md]) best[md] = t;
                    }
                char row[48];
                snprintf(row, sizeof row, "%d experts / %.0f MiB", w,
                         per_exp * (double)w / 1048576.0);
                printf("  %-22s", row);
                for (int md = 0; md < 4; md++)
                    printf(" %11.1f", per_exp * (double)w / best[md] / 1e9);
                putchar('\n');
                fflush(stdout);
            }
        }
        free(fy);
        free(fx);
        free(perm);
        fflush(stdout);
    }

    /* ---- 4. grouped multi-row pass -------------------------------------
     * Bytes are the whole game in this pass. A step carrying T rows reads
     * T*top_k expert matrices if each row is routed on its own, or n_distinct
     * of them if the rows are grouped by expert and each distinct expert is
     * applied to every row that asked for it. The saving is exactly the
     * overlap between the rows' top-k sets, so this measures three things
     * separately: that grouping is the same computation, what the overlap is
     * at each routing skew, and what that is worth in ms per token.
     *
     * Skew is a knob here, not a measurement. Real MoE routing concentrates
     * on hot experts - it is why Strata ranks experts by routing frequency and
     * pins the hot ones - but the concentration of *this* checkpoint's router
     * is a property of activations this build cannot yet produce. Zipf over
     * 512 experts, exponent 0 being uniform, is the honest bracket, and the
     * uniform column is the floor: a future measurement under it means the
     * grouping is broken. */
    {
        /* The widths a chunk actually runs, to the ceiling the engine allows. Up
         * to 16 the question is whether grouping helps at all; above it the
         * question is whether it keeps helping as the distinct experts a chunk
         * touches approach the whole pool, which is where the bytes per token
         * stop falling and the pass has either earned its keep or not. Above 512
         * every one of the 512 experts is read once already, so the only thing a
         * wider window can still buy is a bigger bucket to share one
         * dequantisation over - which is what the GMAC/s column is for. */
        static const int TS[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };
        const int NT = (int)(sizeof TS / sizeof TS[0]);
        const int TSWEEP = 512;
        float *X = aligned_alloc(64, (size_t)TSWEEP * K_ * sizeof *X);
        float *Ym = aligned_alloc(64, (size_t)TSWEEP * K_ * sizeof *Ym);
        float *Yr = aligned_alloc(64, (size_t)TSWEEP * K_ * sizeof *Yr);
        qe_route *R = malloc((size_t)TSWEEP * TOPK * sizeof *R);
        double *cdf = malloc((size_t)E_ * sizeof *cdf);
        if (!X || !Ym || !Yr || !R || !cdf) {
            printf("\ngrouped multi-row pass: skipped (no memory)\n");
        } else {
            for (int t = 0; t < TSWEEP; t++)
                for (long i = 0; i < K_; i++)
                    X[t * K_ + i] = (float)sin((i + 37 * t) * 0.0017) * 0.5f;

            /* ---- 4a. same computation? ---- *
             * Grouping is the thing under test here, not the arithmetic of one walk, so
             * the comparison runs with the int8 recode the grouped pass otherwise uses
             * (src/moe.c) switched off: bucketing, routing weights and row order have to
             * reproduce T separate forwards to the last bit of summation order either
             * way, and a 7-bit activation recode would hide a mis-bucketed expert under
             * an error 1000x larger than the one it is meant to detect. What the recode
             * costs is the next line's business, and it has its own limit. */
            double worst_mt = 0;
            int mt_nonfinite = 0;
            qe_moe_set_int8(0);
            for (int ti = 0; ti < NT; ti++) {
                int T = TS[ti];
                for (int t = 0; t < T; t++)
                    for (int e = 0; e < TOPK; e++) {
                        R[(size_t)t * TOPK + e].expert = (int32_t)(rnd(&s) % E_);
                        R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                    }
                qe_moe_forward_multi(m, X, K_, R, TOPK, T, Ym, K_);
                for (int t = 0; t < T; t++)
                    qe_moe_forward(m, X + (size_t)t * K_, R + (size_t)t * TOPK, TOPK,
                                   Yr + (size_t)t * K_);
                for (int t = 0; t < T; t++) {
                    /* Relative to the size of the terms, not to their sum. These
                     * ten expert outputs are each around 1e5 and cancel, so an
                     * element whose total lands near zero is not a place where a
                     * last-bit difference is 3e-3 of anything - divided by the
                     * sum, this check's verdict was decided by which elements
                     * happened to cancel, and it moved when nothing but the
                     * random draws moved. Divided by the scale of what was added,
                     * a wrong expert or a wrong weight is still O(1). */
                    double scale = 0;
                    for (long k = 0; k < K_; k++)
                        scale += fabs((double)Yr[(size_t)t * K_ + k]);
                    scale /= (double)K_;
                    if (scale < 1e-9)
                        scale = 1e-9;
                    for (long k = 0; k < K_; k++) {
                        double a = Ym[(size_t)t * K_ + k], b = Yr[(size_t)t * K_ + k];
                        if (!(a == a) || !(b == b)) {
                            mt_nonfinite++;
                            continue;
                        }
                        double rel = fabs(a - b) / scale;
                        if (rel > worst_mt) worst_mt = rel;
                    }
                }
            }
            printf("\ngrouped multi-row vs %d separate forwards: worst %.1e, %d non-finite -> %s\n",
                   TS[NT - 1], worst_mt, mt_nonfinite,
                   (!mt_nonfinite && worst_mt < 1e-4) ? "PASS" : "FAIL");

            /* The same comparison with the int8 recode in force, which is the walk the
             * engine runs: same buckets, same routes, a different number for the vector
             * they multiply. Normalised the same way, so the two lines differ only by
             * the recode. A wrong expert or a wrong weight is still O(1) against this
             * scale; what the recode costs was measured at 3e-03 rms against a double
             * reference on the shapes it runs on (ideas.md 7.10). */
            double worst_i8 = 0;
            int i8_nonfinite = 0;
            qe_moe_set_int8(1);
            for (int ti = 5; ti <= 6; ti++) {
                int T = TS[ti];
                for (int t = 0; t < T; t++)
                    for (int e = 0; e < TOPK; e++) {
                        R[(size_t)t * TOPK + e].expert = (int32_t)(rnd(&s) % E_);
                        R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                    }
                qe_moe_forward_multi(m, X, K_, R, TOPK, T, Ym, K_);
                for (int t = 0; t < T; t++)
                    qe_moe_forward(m, X + (size_t)t * K_, R + (size_t)t * TOPK, TOPK,
                                   Yr + (size_t)t * K_);
                for (int t = 0; t < T; t++) {
                    double scale = 0;
                    for (long k = 0; k < K_; k++)
                        scale += fabs((double)Yr[(size_t)t * K_ + k]);
                    scale /= (double)K_;
                    if (scale < 1e-9)
                        scale = 1e-9;
                    for (long k = 0; k < K_; k++) {
                        double a = Ym[(size_t)t * K_ + k], b = Yr[(size_t)t * K_ + k];
                        if (!(a == a) || !(b == b)) {
                            i8_nonfinite++;
                            continue;
                        }
                        double rel = fabs(a - b) / scale;
                        if (rel > worst_i8) worst_i8 = rel;
                    }
                }
            }
            qe_moe_set_int8(-1); /* back to whatever the environment asked for */
            printf("the same with the int8 expert walk: worst %.1e, %d non-finite -> %s\n",
                   worst_i8, i8_nonfinite,
                   (!i8_nonfinite && worst_i8 < 2e-1) ? "PASS" : "FAIL");

            /* ---- 4b. overlap by routing skew ----
             * Distinct experts for T rows averaged over 2000 draws. At
             * exponent 0 this is the birthday-problem floor; at 1.0 the hot
             * experts are shared by most rows and the pass reads a fraction of
             * the bytes. */
            printf("\nrouting overlap (distinct experts per step of T rows, 2000 draws)\n");
            printf("%-14s %10s %10s %10s %10s\n", "zipf exponent", "T=2", "T=4", "T=8", "T=16");
            static const double ZZ[] = { 0.0, 0.5, 0.8, 1.0 };
            for (unsigned zi = 0; zi < sizeof ZZ / sizeof ZZ[0]; zi++) {
                double z = ZZ[zi], acc = 0;
                for (long i = 0; i < E_; i++)
                    cdf[i] = (acc += 1.0 / pow((double)i + 1.0, z));
                for (long i = 0; i < E_; i++)
                    cdf[i] /= acc;
                printf("%-14.1f", z);
                for (int ti = 0; ti < NT; ti++) {
                    int T = TS[ti];
                    double tot = 0;
                    static char seen[4096];
                    for (int draw = 0; draw < 2000; draw++) {
                        memset(seen, 0, (size_t)E_);
                        int nd = 0;
                        for (int i = 0; i < T * TOPK; i++) {
                            double u = (double)(rnd(&s) >> 40) / (double)(1ull << 24);
                            int lo = 0, hi = (int)E_ - 1;
                            while (lo < hi) {
                                int mid = (lo + hi) / 2;
                                if (cdf[mid] < u)
                                    lo = mid + 1;
                                else
                                    hi = mid;
                            }
                            if (!seen[lo]) {
                                seen[lo] = 1;
                                nd++;
                            }
                        }
                        tot += nd;
                    }
                    printf("%10.1f", tot / 2000.0);
                }
                printf("\n");
            }
            printf("  (routed alone, T rows would read T*%d = %d, %d, %d, %d expert matrices)\n",
                   TOPK, 2 * TOPK, 4 * TOPK, 8 * TOPK, 16 * TOPK);

            /* ---- 4c. what it is worth ---- */
            const double exp_one = 2.0 * gsz + dsz;
            printf("\ngrouped vs serial, uniform routing, %d threads (48 layers)\n",
                   qe_pool_size());
            printf("%-5s %12s %14s %9s %16s %11s %9s\n", "rows", "serial ms", "grouped ms",
                   "speedup", "distinct/T*topk", "tok/s @48", "GMAC/s");
            for (int ti = 0; ti < NT; ti++) {
                int T = TS[ti];
                qe_moe_desc md = d;
                md.n_threads = nthr;
                qe_moe *mt = qe_moe_create(&md);
                if (!mt) {
                    printf("%-5d create failed\n", T);
                    continue;
                }
                double ts = 1e30, tg = 1e30, ndist = 0;
                /* The serial leg is T rows through the one-row pass, so its cost
                 * is linear in T: 24 of them is already seconds at 128 rows, and
                 * the ratio it is there for does not need that many samples. */
                const int iters = T <= 16 ? 24 : 6;
                for (int pass = 0; pass < 3; pass++) {
                    double a0 = now();
                    for (int i = 0; i < iters; i++) {
                        for (int t = 0; t < T; t++)
                            for (int e = 0; e < TOPK; e++) {
                                R[(size_t)t * TOPK + e].expert = (int32_t)(rnd(&s) % E_);
                                R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                            }
                        for (int t = 0; t < T; t++)
                            qe_moe_forward(mt, X + (size_t)t * K_, R + (size_t)t * TOPK, TOPK,
                                           Yr + (size_t)t * K_);
                    }
                    double a1 = now();
                    for (int i = 0; i < iters; i++) {
                        for (int t = 0; t < T; t++)
                            for (int e = 0; e < TOPK; e++) {
                                R[(size_t)t * TOPK + e].expert = (int32_t)(rnd(&s) % E_);
                                R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                            }
                        qe_moe_forward_multi(mt, X, K_, R, TOPK, T, Ym, K_);
                    }
                    double a2 = now();
                    if ((a1 - a0) / iters < ts) ts = (a1 - a0) / iters;
                    if ((a2 - a1) / iters < tg) tg = (a2 - a1) / iters;
                }
                static char seen[4096];
                for (int draw = 0; draw < 200; draw++) {
                    memset(seen, 0, (size_t)E_);
                    int nd = 0;
                    for (int i = 0; i < T * TOPK; i++) {
                        int e = (int)(rnd(&s) % E_);
                        if (!seen[e]) {
                            seen[e] = 1;
                            nd++;
                        }
                    }
                    ndist += nd;
                }
                ndist /= 200.0;
                /* Routed multiply-accumulates alone: every row multiplies its own
                 * ten experts whether or not the bytes were shared, so this is the
                 * arithmetic the pass is doing and not the bytes it read. The dense
                 * Q8_0 walk holds 500-640 GMAC/s at these widths (`make walk`), and
                 * whatever this column is short of that is dequantisation and
                 * schedule rather than traffic. */
                printf("%-5d %14.3f %16.3f %8.2fx %16.1f %11.1f %9.0f\n", T, ts * 1e3, tg * 1e3,
                       ts / tg, ndist, 1.0 / (tg / T * LAYERS),
                       (double)T * TOPK * 3.0 * (double)N_ * (double)K_ / tg / 1e9);
                printf("      expert bytes per layer: serial %.1f MiB (%d matrices), grouped %.1f "
                       "MiB (%d distinct), %.1f GB/s of distinct bytes\n",
                       T * TOPK * exp_one / 1048576.0, T * TOPK,
                       ndist * exp_one / 1048576.0, (int)(ndist + 0.5),
                       ndist * exp_one / tg / 1e9);
                qe_moe_destroy(mt);
            }

            /* ---- 4d. the dequantisation, isolated ----
             * Uniform routing gives buckets of about one row, which is the honest
             * floor for bytes and a useless instrument for the other thing a bucket
             * can share: one dequantisation of an expert row serving every row that
             * chose it. This leg therefore puts every row of the chunk on the *same*
             * ten experts - what a hot expert under real routing looks like from the
             * kernel's side - and times one pass with the shared-dequant kernels and
             * one without. Both columns read each expert matrix once, so what they
             * differ by is arithmetic and not traffic: iq3_s costs eight 256-entry
             * grid lookups, a sign spread and a compare per 32 weights, and that is
             * what the second column stops paying T times. */
            printf("\nshared dequantisation, every row on the same experts, %d threads\n", nthr);
            printf("%-5s %12s %12s %9s %9s %9s %9s\n", "rows", "per-row ms", "shared ms",
                   "shared", "reg ms", "int ms", "int+rs ms");
            printf("      (reg = the same walk with the dequantised block held in four\n"
                   "       registers instead of parked in an L1 line; int = the same pass\n"
                   "       with the chunk's activations recoded to int8 and the experts\n"
                   "       walked with vpdpbusd; int+rs = that pass with one scale for a\n"
                   "       whole activation row instead of one per 32 values, which is what\n"
                   "       takes two of the four exit operands out of the token loop. All\n"
                   "       three are ratios to shared ms; lower is better)\n");
            for (int ti = 0; ti < NT; ti++) {
                int T = TS[ti];
                qe_moe_desc md = d;
                md.n_threads = nthr;
                qe_moe *mon = qe_moe_create(&md);
                md.no_shared_multi = 1;
                qe_moe *moff = qe_moe_create(&md);
                if (!moff || !mon) {
                    printf("%-5d create failed\n", T);
                    qe_moe_destroy(moff);
                    qe_moe_destroy(mon);
                    continue;
                }
                for (int t = 0; t < T; t++)
                    for (int e = 0; e < TOPK; e++) {
                        R[(size_t)t * TOPK + e].expert = e;
                        R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                    }
                /* Four legs and one pool: the walk that parks the dequantised block in an
                 * L1 line, the walk that keeps it in registers, the pass that walks the
                 * quantiser once per row, and that pass with a row-scaled panel. The order
                 * reverses every other pass, because the two shared legs differ by four loads
                 * a row and a leg that always follows the same neighbour inherits its
                 * warm-up. */
                double tshare = 1e30, tsolo = 1e30, treg = 1e30, ti8 = 1e30, trs = 1e30;
                const int iters = T <= 16 ? 24 : 8;
                for (int pass = 0; pass < 3; pass++) {
                    for (int leg = 0; leg < 5; leg++) {
                        const int which = (pass & 1) ? 4 - leg : leg;
                        qe_iq3s_set_reg_block(which == 1);
                        qe_i8_set_rowscale(which == 4);
                        qe_moe_set_int8(which == 3 || which == 4);
                        const double t0 = now();
                        for (int i = 0; i < iters; i++)
                            qe_moe_forward_multi(which == 2 ? moff : mon, X, K_, R, TOPK, T, Ym,
                                                 K_);
                        const double dt = (now() - t0) / iters;
                        if (which == 0 && dt < tshare)
                            tshare = dt;
                        if (which == 1 && dt < treg)
                            treg = dt;
                        if (which == 2 && dt < tsolo)
                            tsolo = dt;
                        if (which == 3 && dt < ti8)
                            ti8 = dt;
                        if (which == 4 && dt < trs)
                            trs = dt;
                    }
                }
                qe_iq3s_set_reg_block(0);
                qe_i8_set_rowscale(-1);
                qe_moe_set_int8(-1);
                printf("%-5d %14.3f %12.3f %10.2fx %10.3f %10.3f %10.3f\n", T, tsolo * 1e3,
                       tshare * 1e3, tsolo / tshare, treg / tshare, ti8 / tshare, trs / tshare);
                fflush(stdout);
                qe_moe_destroy(moff);
                qe_moe_destroy(mon);
            }

            /* ---- 4e. the same walk with nothing expert about it ----
             * The expert stage has been quoted against the rate the dense Q8_0
             * projections get - 207 GMAC/s against 637, "3.1x" - and that one comparison
             * has three differences in it: the formats differ (an iq3_s row costs eight
             * grid lookups and a sign spread per 32 weights where a q8_0 row costs a
             * load), the number of token rows sharing one weight read differs (a
             * projection walks with every token of the chunk, an expert with the ten that
             * chose it), and the routed pass gathers its rows, applies silu and reduces
             * partials where a projection does none of that. Five legs, each holding one
             * of the three still:
             *   routed      the grouped pass as shipped, with the window's routes;
             *   uniform     the same three matrices and the same MACs walked contiguously,
             *               every expert handed exactly T*topk/512 token rows, in the same
             *               call shape the grouped pass uses (8 output rows, a park-sized
             *               group) so that what is left when this leg is subtracted from
             *               the one above is schedule and not kernel;
             *   real-fills  the same again with the window's own histogram, so the
             *               balance no router produces is taken out as well;
             *   dense-shape the same MACs again with every token row of the window sharing
             *               one weight read, which is the shape a dense projection walks
             *               and therefore the column the "637" belongs in;
             *   q8_0        the uniform leg with the weights in the format the 637 was
             *               measured in, which is the only column here that is allowed to
             *               be read as what the format costs.
             * MACs are matched by construction, so every column is the same arithmetic and
             * the ratios are costs. Routes are drawn before any timing: drawing them
             * inside made this file's tables swing 20 %, which is more than two of the
             * three effects here. */
            {
                const size_t rb8g = rb_of(QE_Q8_0, K_) * (size_t)N_;
                const size_t rb8d = rb_of(QE_Q8_0, N_) * (size_t)K_;
                const int EQ8 = 64; /* a sample is enough to price a format */
                uint8_t *q8g = qe_alloc_huge(rb8g * (size_t)EQ8);
                uint8_t *q8d = qe_alloc_huge(rb8d * (size_t)EQ8);
                float *Y5 = aligned_alloc(64, (size_t)TSWEEP * TOPK * K_ * sizeof *Y5);
                int32_t *fill = malloc((size_t)E_ * sizeof *fill);
                long *tokoff = malloc((size_t)E_ * sizeof *tokoff);
                if (!q8g || !q8d || !Y5 || !fill || !tokoff) {
                    printf("\nformat-matched reference walk: skipped (no memory)\n");
                } else {
                    for (int e = 0; e < EQ8; e++)
                        for (int which = 0; which < 2; which++) {
                            uint8_t *p =
                                which ? q8d + (size_t)e * rb8d : q8g + (size_t)e * rb8g;
                            const size_t nbytes = which ? rb8d : rb8g;
                            for (size_t i = 0; i < nbytes; i++)
                                p[i] = (uint8_t)(rnd(&s) >> 32);
                            for (size_t off = 0; off + 2 <= nbytes; off += 34) {
                                const uint16_t h = 0x2c00u | (uint16_t)(rnd(&s) & 0x1ff);
                                memcpy(p + off, &h, 2);
                            }
                        }
                    const long ldd_g = K_ / (long)qe_type_info_of(QE_IQ3_S)->block_ne;
                    const long ldd_d = N_ / (long)qe_type_info_of(QE_IQ4_NL)->block_ne;
                    printf("\nformat-matched reference walk, %d threads (gate/up IQ3_S"
                           " 640x2560, down IQ4_NL 2560x640)\n", qe_pool_size());
                    printf("%-5s %10s %10s %12s %10s %10s %10s %10s %9s  %s\n", "rows",
                           "routed ms", "uniform ms", "real-fills ms", "dense ms", "q8_0 ms",
                           "unif/rout", "dense/rout", "q8/unif", "GMAC/s");
                    printf("      columns are routed, uniform, dense-shape and q8_0"
                           " (r/u/d/q8)\n");
                    for (int ti = 7; ti < NT; ti++) {
                        const int T = TS[ti];
                        const int mu = T * TOPK / E_;
                        if (!mu || (long)mu * E_ != (long)T * TOPK)
                            continue; /* only widths that match MACs exactly */
                        memset(fill, 0, (size_t)E_ * sizeof *fill);
                        for (int t = 0; t < T; t++)
                            for (int e = 0; e < TOPK; e++) {
                                const int32_t w = (int32_t)(rnd(&s) % E_);
                                R[(size_t)t * TOPK + e].expert = w;
                                R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                                fill[w]++;
                            }
                        long off = 0;
                        for (int e = 0; e < E_; e++) {
                            tokoff[e] = off;
                            off += fill[e];
                        }
                        qe_moe_desc md = d;
                        md.n_threads = nthr;
                        qe_moe *mt = qe_moe_create(&md);
                        if (!mt) {
                            printf("%-5d create failed\n", T);
                            continue;
                        }
                        ref_job jr = { .W = { gate, up, down },
                                       .stride = { (long)gsz, (long)gsz, (long)dsz },
                                       .rb = { (long)(gsz / N_), (long)(gsz / N_),
                                               (long)(dsz / K_) },
                                       .rows = { N_, N_, K_ },
                                       .cols = { K_, K_, N_ },
                                       .ldd = { ldd_g, ldd_g, ldd_d },
                                       .t = { QE_IQ3_S, QE_IQ3_S, QE_IQ4_NL },
                                       .nmat = 3,
                                       .x = X,
                                       .y = Y5,
                                       .xstride = K_,
                                       .ystride = K_,
                                       .ntok = T,
                                       .fill = fill,
                                       .tokoff = tokoff,
                                       .nexp = E_ };
                        /* The call shape the grouped pass uses: QE_MOE_CHUNK output rows of
                         * one matrix per call and a park-sized group of token rows
                         * (QE_IQ_PARK_MAX), so that the only differences left between this
                         * leg and the one above it are the gather, the silu, the reduce and
                         * the order the buckets are visited in. */
                        jr.chunk_r = 8;
                        jr.maxr = 32;
                        ref_job ju = jr;
                        ju.fill = NULL;
                        ju.tokoff = NULL;
                        ju.muni = mu;
                        ref_job jq = ju;
                        jq.W[0] = jq.W[1] = q8g;
                        jq.W[2] = q8d;
                        jq.stride[0] = jq.stride[1] = (long)rb8g;
                        jq.stride[2] = (long)rb8d;
                        jq.rb[0] = jq.rb[1] = (long)rb8g / N_;
                        jq.rb[2] = (long)rb8d / K_;
                        jq.ldd[0] = jq.ldd[1] = K_ / 32;
                        jq.ldd[2] = N_ / 32;
                        jq.t[0] = jq.t[1] = jq.t[2] = QE_Q8_0;
                        jq.wrap = EQ8;
                        /* Same MACs, and the reuse a projection gets: TOPK matrices, each
                         * walked by every token of the window, so one weight load serves T
                         * rows instead of the T*topk/512 a bucket holds. TOPK x 8 items
                         * keeps nineteen threads fed with ten matrices in play. */
                        ref_job jd = ju;
                        jd.nexp = TOPK;
                        jd.muni = T;
                        jd.nblk = 8;
                        jd.chunk_r = 0;
                        jd.maxr = 0;
                        /* Five legs, order reversed on the odd passes, because a leg that
                         * always follows the same neighbour inherits its warm pools. */
                        double tleg[5] = { 1e30, 1e30, 1e30, 1e30, 1e30 };
                        const int iters = 2;
                        for (int pass = 0; pass < 3; pass++)
                            for (int leg = 0; leg < 5; leg++) {
                                const int which = (pass & 1) ? 4 - leg : leg;
                                const double t0 = now();
                                for (int i = 0; i < iters; i++) {
                                    if (which == 0)
                                        qe_moe_forward_multi(mt, X, K_, R, TOPK, T, Ym, K_);
                                    else
                                        qe_pool_run(ref_task,
                                                    which == 1   ? &ju
                                                    : which == 2 ? &jr
                                                    : which == 3 ? &jd
                                                    : &jq,
                                                    qe_pool_size());
                                }
                                const double dt = (now() - t0) / iters;
                                if (dt < tleg[which])
                                    tleg[which] = dt;
                            }
                        const double macs = (double)T * TOPK * 3.0 * (double)N_ * (double)K_;
                        printf("%-5d %10.1f %10.1f %13.1f %10.1f %10.1f %10.2fx %10.2fx %9.2fx",
                               T, tleg[0] * 1e3, tleg[1] * 1e3, tleg[2] * 1e3, tleg[3] * 1e3,
                               tleg[4] * 1e3, tleg[1] / tleg[0], tleg[3] / tleg[0],
                               tleg[4] / tleg[1]);
                        printf(" %5.0f/%.0f/%.0f/%.0f\n", macs / tleg[0] / 1e9,
                               macs / tleg[1] / 1e9, macs / tleg[3] / 1e9, macs / tleg[4] / 1e9);
                        printf("      routed against a contiguous walk of the same formats,"
                               " MACs and call shape: %.2fx (the window's own bucket fills are"
                               " %.2fx of that, the gather, silu and reduce the rest). Same"
                               " MACs with every row sharing one weight read: %.2fx of routed"
                               " (%.0f vs %.0f GMAC/s). Format alone, q8_0 against the"
                               " codebooks at equal MACs and shape: %.2fx\n", tleg[0] ? tleg[1] / tleg[0] : 0.0,
                               tleg[1] ? tleg[2] / tleg[1] : 0.0, tleg[0] ? tleg[3] / tleg[0] : 0.0,
                               macs / tleg[0] / 1e9, macs / tleg[3] / 1e9,
                               tleg[1] ? tleg[4] / tleg[1] : 0.0);
                        fflush(stdout);
                        qe_moe_destroy(mt);
                    }
                    qe_free_huge(q8g, rb8g * (size_t)EQ8);
                    qe_free_huge(q8d, rb8d * (size_t)EQ8);
                }
                free(Y5);
                free(fill);
                free(tokoff);
            }
        }
        /* ---- 4f. an uneven router, and who waits for it ----
         * Every leg above draws its routes uniformly, which is the friendliest histogram a
         * router could deal: at a 512-row window each expert gets about ten rows, so an
         * equal share of the (bucket, output row) count is nearly an equal share of the
         * work and the question never gets asked. A real router is not that fair, and a
         * bucket of forty costs several times what a bucket of two costs while counting as
         * exactly one bucket. Four zipf exponents, the same routes through the three ways
         * of cutting the stage - by count, by cost, and by cost times core class
         * (src/moe.c, moe_unit_slice) - plus the ratio the boundaries themselves predict,
         * which is the column that says whether a win here is balance or noise. */
        {
            printf("\nsliced by cost against sliced by count, %d threads\n", qe_pool_size());
            printf("%-6s %-6s %8s %10s %10s %10s %8s %10s %10s %8s\n", "rows", "zipf",
                   "max bkt", "count ms", "cost ms", "cost+cls", "by cost",
                   "cost+cls", "GMAC/s", "imbal");
            static const double ZS[] = { 0.0, 0.5, 0.8, 1.0 };
            const int TZS[] = { 128, 512 };
            int *fc = malloc((size_t)E_ * sizeof *fc);
            long *bc = malloc((size_t)E_ * sizeof *bc);
            for (unsigned zi = 0; zi < sizeof ZS / sizeof ZS[0] && fc && bc; zi++) {
                const double z = ZS[zi];
                double cdfacc = 0;
                for (long i = 0; i < E_; i++)
                    cdf[i] = (cdfacc += 1.0 / pow((double)i + 1.0, z));
                for (long i = 0; i < E_; i++)
                    cdf[i] /= cdfacc;
                for (unsigned tj = 0; tj < sizeof TZS / sizeof TZS[0]; tj++) {
                    const int T = TZS[tj];
                    memset(fc, 0, (size_t)E_ * sizeof *fc);
                    for (int t = 0; t < T; t++)
                        for (int e = 0; e < TOPK; e++) {
                            const double u = (double)(rnd(&s) >> 40) / (double)(1ull << 24);
                            int lo = 0, hi = (int)E_ - 1;
                            while (lo < hi) {
                                const int mid = (lo + hi) / 2;
                                if (cdf[mid] < u)
                                    lo = mid + 1;
                                else
                                    hi = mid;
                            }
                            R[(size_t)t * TOPK + e].expert = lo;
                            R[(size_t)t * TOPK + e].weight = 1.0f / TOPK;
                            fc[lo]++;
                        }
                    /* Buckets in the order a grouping builds them, charged as the pass
                     * charges them, and what an equal share of the count leaves each
                     * thread with. */
                    int nbkt = 0, maxb = 0;
                    long chg = 0;
                    for (long e = 0; e < E_; e++)
                        if (fc[e]) {
                            bc[nbkt++] = fc[e] > 8 ? fc[e] : 8;
                            if (fc[e] > maxb)
                                maxb = fc[e];
                            chg += bc[nbkt - 1];
                        }
                    double imb = 0;
                    const int nth = qe_pool_size();
                    if (nbkt && chg) {
                        double heaviest = 0;
                        for (int th = 0; th < nth; th++) {
                            const long b0 = (long)nbkt * th / nth;
                            const long b1 = (long)nbkt * (th + 1) / nth;
                            long mine = 0;
                            for (long b = b0; b < b1; b++)
                                mine += bc[b];
                            if ((double)mine > heaviest)
                                heaviest = (double)mine;
                        }
                        imb = heaviest * (double)nth / (double)chg;
                    }
                    qe_moe_desc md = d;
                    md.n_threads = nthr;
                    qe_moe *mt = qe_moe_create(&md);
                    if (!mt) {
                        printf("%-6d %-6.1f create failed\n", T, z);
                        continue;
                    }
                    double tl[3] = { 1e30, 1e30, 1e30 };
                    const int iters = T <= 128 ? 6 : 3;
                    for (int pass = 0; pass < 3; pass++)
                        for (int leg = 0; leg < 3; leg++) {
                            const int which = (pass & 1) ? 2 - leg : leg;
                            qe_moe_set_balance(which);
                            const double t0 = now();
                            for (int i = 0; i < iters; i++)
                                qe_moe_forward_multi(mt, X, K_, R, TOPK, T, Ym, K_);
                            const double dt = (now() - t0) / iters;
                            if (dt < tl[which])
                                tl[which] = dt;
                        }
                    qe_moe_set_balance(-1);
                    const double macs = (double)T * TOPK * 3.0 * (double)N_ * (double)K_;
                    printf("%-6d %-6.1f %8d %10.1f %10.1f %10.1f %7.3fx %7.3fx %10.0f %7.2fx\n",
                           T, z, maxb, tl[0] * 1e3, tl[1] * 1e3, tl[2] * 1e3,
                           tl[1] ? tl[0] / tl[1] : 0.0, tl[2] ? tl[0] / tl[2] : 0.0,
                           macs / tl[1] / 1e9, imb);
                    fflush(stdout);
                    qe_moe_destroy(mt);
                }
            }
            free(bc);
            free(fc);
        }
        free(X);
        free(Ym);
        free(Yr);
        free(R);
        free(cdf);
    }

    /* ---- 5. expert load spread ---- */
    long ne = 0;
    const uint64_t *hist = qe_moe_expert_hist(m, &ne);
    long used = 0, maxc = 0;
    double sum = 0;
    for (long i = 0; i < ne; i++)
        if (hist[i]) {
            used++;
            if ((long)hist[i] > maxc) maxc = (long)hist[i];
            sum += (double)hist[i];
        }
    printf("\nafter %ld routed layers: %ld of %ld experts seen, hottest %.1f%% of selections\n",
           (long)(sum / TOPK), used, ne, 100.0 * maxc / (sum > 0 ? sum : 1));

    if (nonfinite)
        worst = 1e30;
    qe_moe_destroy(m);
    qe_pool_shutdown();
    qe_free_huge(gate, gsz * E_);
    qe_free_huge(up, gsz * E_);
    qe_free_huge(down, dsz * E_);
    qe_free_huge(router, sizeof(float) * E_ * K_);
    free(x);
    free(y);
    free(yr);
    free(logits);
    free(routes);
    free(sorted);
    return sel_bad || worst >= 1e-4 || bcode || worst_w_refined > 1e-6;
}
