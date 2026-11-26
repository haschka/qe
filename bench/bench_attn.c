/* The attention walk on its own: qe_attn_walk's two forms on one synthetic q8 cache, the
 * shape this checkpoint has (twelve query heads to a KV head, head_dim 256, ~2051 selected
 * cells in ascending order), compared byte for byte and timed single-threaded. */
#include "qe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t rng = 88172645463325252ull;
static double urand(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (rng >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv) {
    const int hd = 256, per = 12, nkvh = 2, krow = hd * nkvh;
    const long ncells = argc > 1 ? atol(argv[1]) : 16384;
    const int nsel = argc > 2 ? atoi(argv[2]) : 2051;
    const int reps = argc > 3 ? atoi(argv[3]) : 200;
    int8_t *kd = malloc((size_t)ncells * krow), *vd = malloc((size_t)ncells * krow);
    float *ks = malloc((size_t)ncells * nkvh * sizeof(float)),
          *vs = malloc((size_t)ncells * nkvh * sizeof(float));
    for (long i = 0; i < ncells * krow; i++) {
        kd[i] = (int8_t)(urand() * 254 - 127);
        vd[i] = (int8_t)(urand() * 254 - 127);
    }
    for (long i = 0; i < ncells * nkvh; i++) {
        ks[i] = 0.002f + 0.004f * (float)urand();
        vs[i] = 0.002f + 0.004f * (float)urand();
    }
    float *q = malloc((size_t)per * hd * sizeof(float));
    for (int i = 0; i < per * hd; i++)
        q[i] = (float)(urand() * 2 - 1) * 3.0f;
    int *sel = malloc((size_t)nsel * sizeof(int));
    /* ascending: blocks of four, as the indexer hands them */
    long c = 0;
    for (int i = 0; i < nsel; i++) {
        if (i % 4 == 0)
            c += (long)(urand() * 2 * (ncells / (nsel + 1))) / 4 * 4;
        sel[i] = (int)(c % ncells);
        c++;
    }
    for (int i = 1; i < nsel; i++)
        if (sel[i] < sel[i - 1])
            sel[i] = sel[i - 1];
    float y[2][12 * 256], m[2][12], l[2][12];
    double best[2] = { 1e30, 1e30 };
    for (int ex = 0; ex < 2; ex++) {
        for (int form = 0; form < 2; form++) {
            qe_attn_set_blk(form);
            qe_attn_unit u = { .q = q, .kd = kd, .vd = vd, .ks = ks, .vs = vs, .sel = sel,
                               .y = y[form], .m = m[form], .l = l[form], .nsel = nsel,
                               .per = per, .hd = hd, .krow = krow, .kv_head = 1,
                               .sstride = nkvh, .q8 = 1, .exp2 = ex, .scale = 0.0625f };
            for (int r = 0; r < reps; r++) {
                const double t = now();
                qe_attn_walk(&u);
                const double d = now() - t;
                if (d < best[form])
                    best[form] = d;
            }
        }
        const int same = !memcmp(y[0], y[1], sizeof y[0]) && !memcmp(m[0], m[1], sizeof m[0]) &&
                         !memcmp(l[0], l[1], sizeof l[0]);
        printf("exp2 %d: walk %.1f us, blocked %.1f us, %.2fx, %s\n", ex, best[0] * 1e6,
               best[1] * 1e6, best[0] / best[1], same ? "byte-identical" : "DIFFERENT");
        best[0] = best[1] = 1e30;
    }
    /* The indexer: four heads of 128 against `ncells / 4` pooled blocks. */
    {
        const int id = 128, nih = 4;
        const long nb = ncells / 4;
        float *pool = malloc((size_t)nb * id * sizeof(float)), *iq = malloc(nih * id * sizeof(float));
        float *o[2] = { malloc((size_t)nb * sizeof(float)), malloc((size_t)nb * sizeof(float)) };
        for (long i = 0; i < nb * id; i++)
            pool[i] = (float)(urand() * 2 - 1);
        for (int i = 0; i < nih * id; i++)
            iq[i] = (float)(urand() * 2 - 1);
        double bt[2] = { 1e30, 1e30 };
        for (int form = 0; form < 2; form++) {
            qe_idx_set_fast(form);
            for (int r = 0; r < reps; r++) {
                const double t = now();
                qe_idx_scores(iq, pool, nih, id, 0.088f, 0, nb, o[form]);
                const double d = now() - t;
                if (d < bt[form])
                    bt[form] = d;
            }
        }
        printf("indexer, %ld blocks: loop %.1f us, fast %.1f us, %.2fx, %s\n", nb, bt[0] * 1e6,
               bt[1] * 1e6, bt[0] / bt[1],
               memcmp(o[0], o[1], (size_t)nb * sizeof(float)) ? "DIFFERENT" : "byte-identical");
    }
    return 0;
}
