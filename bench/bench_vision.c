/* The vision tower against the reference, and its time.

     bin/bench_vision <mmproj.gguf> <image> [reference.bin] [out.bin] [max_tokens]

   Loads the mmproj alone (0.9 GiB, never the trunk), encodes one image and prints
   where the time went per stage. With a reference - the raw dump llama-mtmd-cli
   writes when MTMD_DEBUG_EMBEDDINGS names a path, [int32 rows][int32 width][f32...]
   - it compares row for row: the grid has to be the same grid, and the values are
   held to a cosine per row and a relative norm over the whole, because the
   reference rounds activations to BF16 for its products and this tower does not.
   `out.bin` is this tower's output in the same format.

   Exit status is non-zero when the comparison fails, so `make vision` is a test. */
#include "qe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

static void *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = len > 0 ? malloc((size_t)len) : NULL;
    if (p && fread(p, 1, (size_t)len, f) != (size_t)len) {
        free(p);
        p = NULL;
    }
    fclose(f);
    *n = p ? (size_t)len : 0;
    return p;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s mmproj.gguf image [reference.bin] [out.bin] [max_tokens]\n",
                argv[0]);
        return 2;
    }
    const int max_tokens = argc > 5 ? atoi(argv[5]) : 4096;
    qe_pool_init(QE_THREADS_AUTO, NULL, 0); /* returns the width it chose */
    double t0 = now();
    qe_model *m = qe_open(argv[1], QE_STAGE_RAM | QE_STAGE_STRICT);
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], qe_strerror());
        return 1;
    }
    char err[256] = { 0 };
    qe_vision *v = qe_vision_open(m, max_tokens, err, sizeof err);
    if (!v) {
        fprintf(stderr, "vision: %s\n", err);
        return 1;
    }
    (void)t0;
    size_t nb = 0;
    void *bytes = slurp(argv[2], &nb);
    if (!bytes) {
        fprintf(stderr, "%s: cannot read\n", argv[2]);
        return 1;
    }
    unsigned char *rgb = NULL;
    int w = 0, h = 0;
    if (qe_vision_decode(bytes, nb, &rgb, &w, &h, err, sizeof err)) {
        fprintf(stderr, "decode: %s\n", err);
        return 1;
    }
    int nx = 0, ny = 0;
    qe_vision_grid(v, w, h, &nx, &ny);
    const int D = qe_vision_dim(v);
    float *out = malloc((size_t)nx * ny * D * sizeof(float));
    qe_vision_stats st = { 0 };
    double t1 = now();
    long rows = qe_vision_encode(v, rgb, w, h, out, (long)nx * ny, &nx, &ny, NULL, NULL, &st);
    double t2 = now();
    if (rows < 0) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }
    printf("image %dx%d -> %dx%d, %ld patches, %ld tokens of %d\n", w, h, st.width, st.height,
           st.patches, rows, D);
    printf("time: %.3f s (preprocess %.3f, patch %.3f, blocks %.3f of which attention %.3f, "
           "merger %.3f), %d threads\n",
           t2 - t1, st.preprocess, st.patch, st.blocks, st.attention, st.merge, qe_pool_size());
    double ss = 0.0;
    for (long i = 0; i < rows * D; i++)
        ss += (double)out[i] * out[i];
    printf("rms %.5f, row 0 starts %.6f %.6f %.6f %.6f\n", sqrt(ss / (double)(rows * D)), out[0],
           out[1], out[2], out[3]);

    if (argc > 4 && strcmp(argv[4], "-")) {
        FILE *f = fopen(argv[4], "wb");
        int32_t hdr[2] = { (int32_t)rows, D };
        if (f) {
            fwrite(hdr, sizeof hdr, 1, f);
            fwrite(out, sizeof(float), (size_t)rows * D, f);
            fclose(f);
        }
    }
    int fails = 0;
    if (argc > 3 && strcmp(argv[3], "-")) {
        size_t rn = 0;
        int32_t *ref = slurp(argv[3], &rn);
        if (!ref || rn < 8) {
            fprintf(stderr, "%s: cannot read reference\n", argv[3]);
            return 1;
        }
        const long rrows = ref[0], rd = ref[1];
        const float *r = (const float *)(ref + 2);
        printf("reference: %ld rows of %ld\n", rrows, rd);
        if (rrows != rows || rd != D || rn != 8 + (size_t)rrows * rd * 4) {
            printf("FAIL the grid differs from the reference's\n");
            return 1;
        }
        /* Per-row cosines, sorted, and the norm of the difference over the whole. The
         * tail is not the test: a few tokens of a ViT are ill-conditioned, and
         * llama.cpp's own flash and plain attention disagree on them by 0.97 in cosine
         * and 2.3 % overall on the same image. The median and the bulk are. */
        double num = 0.0, den = 0.0;
        double *cs = malloc((size_t)rows * sizeof *cs);
        long worst_at = 0;
        for (long i = 0; i < rows; i++) {
            double ab = 0, aa = 0, bb = 0;
            for (int k = 0; k < D; k++) {
                const double a = out[i * D + k], b = r[i * D + k];
                ab += a * b;
                aa += a * a;
                bb += b * b;
                num += (a - b) * (a - b);
                den += b * b;
            }
            cs[i] = ab / sqrt(aa * bb + 1e-30);
            if (cs[i] < cs[worst_at])
                worst_at = i;
        }
        const double worst = cs[worst_at];
        long good = 0;
        for (long i = 0; i < rows; i++)
            good += cs[i] >= 0.999;
        for (long i = 1; i < rows; i++) /* insertion sort: rows are a few thousand */
            for (long k = i; k > 0 && cs[k] < cs[k - 1]; k--) {
                double t = cs[k];
                cs[k] = cs[k - 1];
                cs[k - 1] = t;
            }
        const double median = cs[rows / 2], frac = (double)good / (double)rows;
        const double rel = sqrt(num / (den + 1e-30));
        printf("against the reference: median row cosine %.6f, %.1f %% of rows >= 0.999, "
               "worst %.4f (row %ld), relative L2 %.4f\n",
               median, 100.0 * frac, worst, worst_at, rel);
        if (median < 0.9995 || frac < 0.90 || rel > 0.06) {
            printf("FAIL beyond the reference's own precision spread (median >= 0.9995, "
                   ">= 90 %% of rows >= 0.999, relative L2 <= 0.06)\n");
            fails++;
        } else {
            printf("PASS\n");
        }
        free(cs);
        free(ref);
    }
    free(out);
    free(rgb);
    free(bytes);
    qe_vision_close(v);
    qe_close(m);
    qe_pool_shutdown();
    return fails ? 1 : 0;
}
