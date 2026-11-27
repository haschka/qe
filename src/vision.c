/* The vision tower: an image's bytes in, one 2560-wide row per merged 2x2 patch out,
 * which is the width the trunk embeds a token at and so the width an image row is fed
 * to it with (src/forward.c, qe_forward_prefill_mm).
 *
 * What the mmproj in this checkpoint's directory is, read out of the file: a 27-block
 * ViT, 1152 wide with 16 heads of 72, 16-pixel patches embedded by two kernels that a
 * still image runs both of (they are a temporal pair, and an image is a clip of one
 * frame shown twice), a 48x48 learned position table, a 2D rotary in every block, and a
 * merger that concatenates each 2x2 group of patches into 4608 values and walks them
 * through two matrices to 2560 (clip.projector_type qwen3vl_merger, no deepstack
 * layers). The reference is llama.cpp's tools/mtmd at b12359 - mtmd_graph_qwen3vl,
 * the qwen3vl preprocessor, ggml's vision rope and its align-corners bilinear - and
 * everything up to the first matrix is reproduced from it line for line, because an
 * image that is resized one pixel differently is a different image:
 *
 *   - decode: stb_image, the decoder the reference uses, so a JPEG decodes to the same
 *     pixels. WebP is libwebp's, opened at run time, because stb has no WebP and the
 *     reference has it only through ffmpeg.
 *   - size: the "smart resize" - each edge rounded half-to-even to a multiple of 32,
 *     then scaled to stay within [8, max_tokens] merged tokens - in the float
 *     arithmetic the reference does it in.
 *   - resize: Pillow's separable bicubic (a = -0.5) in 22-bit fixed point.
 *   - patches: in merge order, the four patches of a 2x2 group consecutive, which is
 *     the order both the position table and the rotary positions are laid out in.
 *
 * After that the arithmetic is f32 with the BF16 weights widened as they are packed,
 * where the reference rounds activations to BF16 for its matrix products and uses a
 * half-precision GELU table. Those are its approximations, not the model's, so the
 * comparison is a tolerance and `bin/bench_vision` reports it against the reference's
 * own dump (MTMD_DEBUG_EMBEDDINGS).
 *
 * Nothing here runs unless a request carries an image. The weights are the mmproj's
 * resident bytes, staged at open with everything else, and the only memory an encode
 * allocates is its own activations, freed when it returns. */
#define _GNU_SOURCE
#include "qe.h"
#include "kern.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_GIF
#define STBI_ONLY_BMP
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include "stb_image.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define V_PATCH 16
#define V_MERGE 2
#define V_ALIGN (V_PATCH * V_MERGE)
#define V_MIN_TOKENS 8
#define V_ROPE_BASE 10000.0f

typedef struct {
    const uint16_t *qkv_w, *o_w, *up_w, *down_w; /* BF16, rows of K */
    const float *qkv_b, *o_b, *up_b, *down_b;
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
} vblock;

struct qe_vision {
    qe_model *m;
    int n_embd, n_head, d_head, n_ff, n_layer, out_dim, max_tokens;
    int pos_side; /* the learned table is pos_side x pos_side */
    float eps;
    float mean[3], std[3];
    float *patch_w; /* the two temporal kernels summed: [1152][3][16][16] */
    const float *patch_b, *pos_embd, *post_w, *post_b;
    const uint16_t *mm0_w, *mm2_w;
    const float *mm0_b, *mm2_b;
    vblock *blk;
    int nthr;
    /* QE_VISION_EMULATE=1: the reference's two approximations, for verifying this
     * tower against it and for nothing else - activations rounded to BF16 before every
     * product with a BF16 weight, and GELU through a half-precision table. */
    int emulate;
    float **pack; /* one packing panel per pool slot */
    size_t pack_cap;
};

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ decoding */

/* stb has no WebP and the reference has it only through ffmpeg; libwebp is on most
 * hosts that have a browser, and asking for it at run time keeps it out of the
 * build. Two symbols are all it takes. */
typedef uint8_t *(*webp_decode_fn)(const uint8_t *, size_t, int *, int *);
typedef void (*webp_free_fn)(void *);

static int is_webp(const unsigned char *p, size_t n) {
    return n >= 12 && !memcmp(p, "RIFF", 4) && !memcmp(p + 8, "WEBP", 4);
}

static unsigned char *decode_webp(const unsigned char *p, size_t n, int *w, int *h, char *err,
                                  size_t errn) {
    static void *lib;
    static webp_decode_fn dec;
    static webp_free_fn fre;
    if (!lib) {
        const char *names[] = { "libwebp.so.7", "libwebp.so", NULL };
        for (int i = 0; names[i] && !lib; i++)
            lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (lib) {
            dec = (webp_decode_fn)dlsym(lib, "WebPDecodeRGB");
            fre = (webp_free_fn)dlsym(lib, "WebPFree");
        }
    }
    if (!lib || !dec) {
        snprintf(err, errn, "WebP image: libwebp is not installed on this host; send PNG or JPEG");
        return NULL;
    }
    uint8_t *rgb = dec(p, n, w, h);
    if (!rgb) {
        snprintf(err, errn, "WebP image could not be decoded");
        return NULL;
    }
    size_t bytes = (size_t)*w * (size_t)*h * 3;
    unsigned char *out = malloc(bytes);
    if (out)
        memcpy(out, rgb, bytes);
    if (fre)
        fre(rgb);
    else
        free(rgb);
    if (!out)
        snprintf(err, errn, "out of memory decoding the image");
    return out;
}

int qe_vision_decode(const void *bytes, size_t n, unsigned char **rgb, int *w, int *h, char *err,
                     size_t errn) {
    const unsigned char *p = bytes;
    *rgb = NULL;
    if (!p || n < 8 || n > (size_t)INT32_MAX) {
        snprintf(err, errn, "image is empty or too large");
        return -1;
    }
    if (is_webp(p, n)) {
        *rgb = decode_webp(p, n, w, h, err, errn);
        return *rgb ? 0 : -1;
    }
    int nc = 0;
    stbi_uc *d = stbi_load_from_memory(p, (int)n, w, h, &nc, 3);
    if (!d) {
        snprintf(err, errn, "image could not be decoded (%s); PNG, JPEG, GIF, BMP and WebP are read",
                 stbi_failure_reason() ? stbi_failure_reason() : "unknown format");
        return -1;
    }
    *rgb = d; /* stbi_image_free is free() with the default allocator */
    return 0;
}

/* --------------------------------------------------------------- preprocess */

/* The reference's calc_size_preserved_ratio with round-half-even, in its float
 * arithmetic: an edge that lands on .5 of a multiple goes to the even one, as
 * Python's round() does in the processor the model was trained behind. */
static void target_size(int width, int height, int max_tokens, int *ow, int *oh) {
    const int f = V_ALIGN;
    const int min_px = V_MIN_TOKENS * f * f, max_px = max_tokens * f * f;
    int w_bar = (int)nearbyintf((float)width / (float)f) * f;
    int h_bar = (int)nearbyintf((float)height / (float)f) * f;
    if (w_bar < f)
        w_bar = f;
    if (h_bar < f)
        h_bar = f;
    if ((long)h_bar * w_bar > max_px) {
        const float beta = sqrtf((float)height * (float)width / (float)max_px);
        h_bar = (int)floorf((float)height / beta / (float)f) * f;
        w_bar = (int)floorf((float)width / beta / (float)f) * f;
        if (h_bar < f)
            h_bar = f;
        if (w_bar < f)
            w_bar = f;
    } else if ((long)h_bar * w_bar < min_px) {
        const float beta = sqrtf((float)min_px / ((float)height * (float)width));
        h_bar = (int)ceilf((float)height * beta / (float)f) * f;
        w_bar = (int)ceilf((float)width * beta / (float)f) * f;
    }
    *ow = w_bar;
    *oh = h_bar;
}

int qe_vision_grid(const qe_vision *v, int w, int h, int *nx, int *ny) {
    if (!v || w <= 0 || h <= 0)
        return -1;
    int tw, th;
    target_size(w, h, v->max_tokens, &tw, &th);
    *nx = tw / V_ALIGN;
    *ny = th / V_ALIGN;
    return 0;
}

/* Pillow's resample, bicubic with a = -0.5: separable, weights normalised per output
 * pixel, then rounded to 22 fractional bits by adding a half and truncating. */
#define PREC 22

static double bicubic(double x) {
    const double a = -0.5;
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

static int weights_for(int in, int out, int **bounds, int32_t **wts) {
    double scale = (double)in / out, fscale = scale < 1.0 ? 1.0 : scale;
    double support = 2.0 * fscale;
    int ksize = (int)ceil(support) * 2 + 1;
    double *pre = malloc((size_t)out * (size_t)ksize * sizeof *pre);
    *bounds = malloc((size_t)out * 2 * sizeof **bounds);
    *wts = malloc((size_t)out * (size_t)ksize * sizeof **wts);
    if (!pre || !*bounds || !*wts) {
        free(pre);
        return -1;
    }
    for (int xx = 0; xx < out; xx++) {
        double center = (xx + 0.5) * scale, ww = 0.0, ss = 1.0 / fscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0)
            xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in)
            xmax = in;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; x++) {
            double wv = bicubic((x + xmin - center + 0.5) * ss);
            pre[(size_t)xx * ksize + x] = wv;
            ww += wv;
        }
        for (x = 0; x < xmax; x++)
            if (ww != 0.0)
                pre[(size_t)xx * ksize + x] /= ww;
        for (; x < ksize; x++)
            pre[(size_t)xx * ksize + x] = 0;
        (*bounds)[xx * 2] = xmin;
        (*bounds)[xx * 2 + 1] = xmax;
    }
    const double fxp = ldexp(1.0, PREC);
    for (size_t i = 0; i < (size_t)out * ksize; i++)
        (*wts)[i] = (int32_t)(pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
    free(pre);
    return ksize;
}

static inline uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }

static unsigned char *resize_h(const unsigned char *src, int inw, int h, int outw) {
    int *b = NULL;
    int32_t *k = NULL;
    int ks = weights_for(inw, outw, &b, &k);
    unsigned char *out = ks > 0 ? malloc((size_t)outw * h * 3) : NULL;
    if (out) {
        for (int y = 0; y < h; y++) {
            const unsigned char *row = src + (size_t)y * inw * 3;
            unsigned char *dst = out + (size_t)y * outw * 3;
            for (int xx = 0; xx < outw; xx++) {
                const int xmin = b[xx * 2], xc = b[xx * 2 + 1];
                const int32_t *kk = k + (size_t)xx * ks;
                const unsigned char *p = row + (size_t)xmin * 3;
                int32_t s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xc; x++, p += 3) {
                    s0 += p[0] * kk[x];
                    s1 += p[1] * kk[x];
                    s2 += p[2] * kk[x];
                }
                dst[xx * 3] = clip8(s0 >> PREC);
                dst[xx * 3 + 1] = clip8(s1 >> PREC);
                dst[xx * 3 + 2] = clip8(s2 >> PREC);
            }
        }
    }
    free(b);
    free(k);
    return out;
}

static unsigned char *resize_v(const unsigned char *src, int w, int inh, int outh) {
    int *b = NULL;
    int32_t *k = NULL;
    int ks = weights_for(inh, outh, &b, &k);
    const size_t re = (size_t)w * 3;
    unsigned char *out = ks > 0 ? malloc(re * outh) : NULL;
    int32_t *acc = out ? malloc(re * sizeof *acc) : NULL;
    if (out && acc) {
        for (int yy = 0; yy < outh; yy++) {
            const int ymin = b[yy * 2], yc = b[yy * 2 + 1];
            const int32_t *kk = k + (size_t)yy * ks;
            for (size_t i = 0; i < re; i++)
                acc[i] = 1 << (PREC - 1);
            for (int y = 0; y < yc; y++) {
                const unsigned char *row = src + (size_t)(ymin + y) * re;
                const int32_t wv = kk[y];
                for (size_t i = 0; i < re; i++)
                    acc[i] += row[i] * wv;
            }
            unsigned char *dst = out + (size_t)yy * re;
            for (size_t i = 0; i < re; i++)
                dst[i] = clip8(acc[i] >> PREC);
        }
    } else {
        free(out);
        out = NULL;
    }
    free(acc);
    free(b);
    free(k);
    return out;
}

/* ------------------------------------------------------------------- GEMM
 *
 * Y[n][m] = bias[m] + sum_k X[n][k] W(m, k), with W either a row-major matrix of
 * output rows (a weight, BF16 or f32) or a column view (V of an attention head, read
 * as K x M). Sixteen output columns are widened into a panel of K x 16 floats, and the
 * panel is swept by six input rows at a time: twelve accumulators, two loads and six
 * broadcasts for every twelve FMAs. */

enum { W_BF16, W_F32, W_COL };

typedef struct {
    const void *w;
    long ldw;
    int kind;
} wsrc;

static inline float bf16f(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } c = { .u = (uint32_t)h << 16 };
    return c.f;
}

static void pack16(float *dst, wsrc w, long m0, int mc, long K) {
    if (w.kind == W_COL) {
        const float *b = (const float *)w.w;
        for (long k = 0; k < K; k++) {
            const float *r = b + (size_t)k * w.ldw + m0;
            float *d = dst + (size_t)k * 16;
            int j = 0;
            for (; j < mc; j++)
                d[j] = r[j];
            for (; j < 16; j++)
                d[j] = 0.0f;
        }
        return;
    }
    for (int j = 0; j < 16; j++) {
        if (j >= mc) {
            for (long k = 0; k < K; k++)
                dst[(size_t)k * 16 + j] = 0.0f;
            continue;
        }
        if (w.kind == W_BF16) {
            const uint16_t *r = (const uint16_t *)w.w + (size_t)(m0 + j) * w.ldw;
            for (long k = 0; k < K; k++)
                dst[(size_t)k * 16 + j] = bf16f(r[k]);
        } else {
            const float *r = (const float *)w.w + (size_t)(m0 + j) * w.ldw;
            for (long k = 0; k < K; k++)
                dst[(size_t)k * 16 + j] = r[k];
        }
    }
}

/* Rows [n0, n1) of X against one packed panel; columns m0..m0+mc of Y. */
static void sweep16(const float *X, long ldx, const float *P, long K, long n0, long n1, float *Y,
                    long ldy, long m0, int mc, const float *bias, int acc_into) {
    long n = n0;
#if defined(__x86_64__)
    for (; n + 6 <= n1; n += 6) {
        __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00, a20 = a00,
               a21 = a00, a30 = a00, a31 = a00, a40 = a00, a41 = a00, a50 = a00, a51 = a00;
        const float *x0 = X + (size_t)n * ldx, *x1 = x0 + ldx, *x2 = x1 + ldx, *x3 = x2 + ldx,
                    *x4 = x3 + ldx, *x5 = x4 + ldx;
        for (long k = 0; k < K; k++) {
            const __m256 b0 = _mm256_loadu_ps(P + (size_t)k * 16);
            const __m256 b1 = _mm256_loadu_ps(P + (size_t)k * 16 + 8);
            __m256 a;
            a = _mm256_broadcast_ss(x0 + k);
            a00 = _mm256_fmadd_ps(a, b0, a00);
            a01 = _mm256_fmadd_ps(a, b1, a01);
            a = _mm256_broadcast_ss(x1 + k);
            a10 = _mm256_fmadd_ps(a, b0, a10);
            a11 = _mm256_fmadd_ps(a, b1, a11);
            a = _mm256_broadcast_ss(x2 + k);
            a20 = _mm256_fmadd_ps(a, b0, a20);
            a21 = _mm256_fmadd_ps(a, b1, a21);
            a = _mm256_broadcast_ss(x3 + k);
            a30 = _mm256_fmadd_ps(a, b0, a30);
            a31 = _mm256_fmadd_ps(a, b1, a31);
            a = _mm256_broadcast_ss(x4 + k);
            a40 = _mm256_fmadd_ps(a, b0, a40);
            a41 = _mm256_fmadd_ps(a, b1, a41);
            a = _mm256_broadcast_ss(x5 + k);
            a50 = _mm256_fmadd_ps(a, b0, a50);
            a51 = _mm256_fmadd_ps(a, b1, a51);
        }
        float t[6][16];
        _mm256_storeu_ps(t[0], a00);
        _mm256_storeu_ps(t[0] + 8, a01);
        _mm256_storeu_ps(t[1], a10);
        _mm256_storeu_ps(t[1] + 8, a11);
        _mm256_storeu_ps(t[2], a20);
        _mm256_storeu_ps(t[2] + 8, a21);
        _mm256_storeu_ps(t[3], a30);
        _mm256_storeu_ps(t[3] + 8, a31);
        _mm256_storeu_ps(t[4], a40);
        _mm256_storeu_ps(t[4] + 8, a41);
        _mm256_storeu_ps(t[5], a50);
        _mm256_storeu_ps(t[5] + 8, a51);
        for (int r = 0; r < 6; r++) {
            float *y = Y + (size_t)(n + r) * ldy + m0;
            if (acc_into)
                for (int j = 0; j < mc; j++)
                    y[j] += t[r][j];
            else
                for (int j = 0; j < mc; j++)
                    y[j] = t[r][j] + (bias ? bias[m0 + j] : 0.0f);
        }
    }
#endif
    for (; n < n1; n++) {
        float acc[16] = { 0 };
        const float *x = X + (size_t)n * ldx;
        for (long k = 0; k < K; k++) {
            const float xv = x[k];
            const float *p = P + (size_t)k * 16;
            for (int j = 0; j < 16; j++)
                acc[j] += xv * p[j];
        }
        float *y = Y + (size_t)n * ldy + m0;
        for (int j = 0; j < mc; j++)
            y[j] = acc_into ? y[j] + acc[j] : acc[j] + (bias ? bias[m0 + j] : 0.0f);
    }
}

/* One thread, the whole product: what an attention head's two products are, inside a
 * task that already owns a thread. */
static void gemm1(const float *X, long ldx, long N, wsrc w, long M, long K, float *Y, long ldy,
                  const float *bias, float *pack) {
    for (long m0 = 0; m0 < M; m0 += 16) {
        const int mc = M - m0 < 16 ? (int)(M - m0) : 16;
        pack16(pack, w, m0, mc, K);
        sweep16(X, ldx, pack, K, 0, N, Y, ldy, m0, mc, bias, 0);
    }
}

/* The same product blocked over K, for a long K against few columns: an attention
 * head's weights against its values, where K is every key. Unblocked, the block of
 * weights (QB rows of every key, megabytes) is streamed once per 16 columns; blocked,
 * a slice of it stays in L2 while all the columns are taken from it. */
#define KB 256

static void gemm1_kblocked(const float *X, long ldx, long N, wsrc w, long M, long K, float *Y,
                           long ldy, float *pack) {
    for (long n = 0; n < N; n++)
        memset(Y + (size_t)n * ldy, 0, (size_t)M * sizeof(float));
    for (long k0 = 0; k0 < K; k0 += KB) {
        const long kc = K - k0 < KB ? K - k0 : KB;
        wsrc ws = w;
        ws.w = (const float *)w.w + (size_t)k0 * w.ldw; /* a column view: rows are keys */
        for (long m0 = 0; m0 < M; m0 += 16) {
            const int mc = M - m0 < 16 ? (int)(M - m0) : 16;
            pack16(pack, ws, m0, mc, kc);
            sweep16(X + k0, ldx, pack, kc, 0, N, Y, ldy, m0, mc, NULL, 1);
        }
    }
}

/* The pool, over tiles of 16 columns x 96 rows taken from a cursor: a P-core takes
 * more of them than an E-core instead of waiting at the end for it. */
#define NT_ROWS 96

typedef struct {
    qe_vision *v;
    const float *X;
    long ldx, N, M, K, ldy;
    wsrc w;
    float *Y;
    const float *bias;
    long tiles_m, tiles;
    long cursor;
} gemm_job;

static void gemm_task(void *arg, int tid, int nthr) {
    gemm_job *j = arg;
    (void)nthr;
    float *pack = j->v->pack[tid];
    long have_m = -1;
    for (;;) {
        long t = __atomic_fetch_add(&j->cursor, 1, __ATOMIC_RELAXED);
        if (t >= j->tiles)
            break;
        /* Column-major over tiles, so consecutive tiles of one thread usually share a
         * panel and the widening is paid once for several row blocks. */
        const long tm = t / ((j->N + NT_ROWS - 1) / NT_ROWS);
        const long tn = t % ((j->N + NT_ROWS - 1) / NT_ROWS);
        const long m0 = tm * 16;
        const int mc = j->M - m0 < 16 ? (int)(j->M - m0) : 16;
        if (have_m != m0) {
            pack16(pack, j->w, m0, mc, j->K);
            have_m = m0;
        }
        const long n0 = tn * NT_ROWS, n1 = n0 + NT_ROWS < j->N ? n0 + NT_ROWS : j->N;
        sweep16(j->X, j->ldx, pack, j->K, n0, n1, j->Y, j->ldy, m0, mc, j->bias, 0);
    }
}

static inline float round_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } c = { .f = f };
    if ((c.u & 0x7fffffffu) > 0x7f800000u)
        return f;
    c.u = ((c.u + (0x7fffu + ((c.u >> 16) & 1u))) >> 16) << 16;
    return c.f;
}

static void gemm(qe_vision *v, const float *X, long ldx, long N, wsrc w, long M, long K, float *Y,
                 long ldy, const float *bias) {
    if (v->emulate && w.kind == W_BF16)
        for (long n = 0; n < N; n++)
            for (long k = 0; k < K; k++) {
                float *xp = (float *)X + (size_t)n * ldx + k;
                *xp = round_bf16(*xp);
            }
    gemm_job j = { .v = v, .X = X, .ldx = ldx, .N = N, .M = M, .K = K, .ldy = ldy, .w = w,
                   .Y = Y, .bias = bias };
    j.tiles_m = (M + 15) / 16;
    j.tiles = j.tiles_m * ((N + NT_ROWS - 1) / NT_ROWS);
    qe_pool_run(gemm_task, &j, v->nthr);
}

/* ------------------------------------------------------------ elementwise */

typedef struct {
    float *x;
    const float *w, *b;
    long n, d;
    float eps;
    int op; /* 0 layernorm, 1 gelu */
    int emulate;
    float *out;
} rows_job;

static inline float gelu1(float x) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static void rows_task(void *arg, int tid, int nthr) {
    rows_job *j = arg;
    long lo, hi;
    qe_slice(j->n, tid, nthr, &lo, &hi);
    for (long r = lo; r < hi; r++) {
        float *x = j->x + (size_t)r * j->d;
        float *o = j->out ? j->out + (size_t)r * j->d : x;
        if (j->op == 1) {
            if (j->emulate) {
                for (long i = 0; i < j->d; i++) {
                    const float xi = x[i];
                    o[i] = xi <= -10.0f ? 0.0f
                         : xi >= 10.0f  ? xi
                                        : _cvtsh_ss(_cvtss_sh(gelu1(_cvtsh_ss(_cvtss_sh(xi, 0))), 0));
                }
                continue;
            }
            for (long i = 0; i < j->d; i++)
                o[i] = gelu1(x[i]);
            continue;
        }
        double s = 0.0;
        for (long i = 0; i < j->d; i++)
            s += x[i];
        const float mean = (float)(s / (double)j->d);
        double ss = 0.0;
        for (long i = 0; i < j->d; i++) {
            const float c = x[i] - mean;
            ss += (double)c * c;
        }
        const float inv = 1.0f / sqrtf((float)(ss / (double)j->d) + j->eps);
        for (long i = 0; i < j->d; i++)
            o[i] = (x[i] - mean) * inv * j->w[i] + j->b[i];
    }
}

static void layernorm(qe_vision *v, const float *x, float *out, long n, const float *w,
                      const float *b) {
    rows_job j = { .x = (float *)x, .w = w, .b = b, .n = n, .d = v->n_embd, .eps = v->eps,
                   .op = 0, .out = out };
    qe_pool_run(rows_task, &j, v->nthr);
}

static void gelu_rows(qe_vision *v, float *x, long n, long d) {
    rows_job j = { .x = x, .n = n, .d = d, .op = 1, .emulate = v->emulate };
    qe_pool_run(rows_task, &j, v->nthr);
}

/* ------------------------------------------------------------ attention */

/* The vision rotary, ggml's GGML_ROPE_TYPE_VISION with sections of d_head/4: the
 * pairs are (j, j + d/2), the first d/4 of them turned by the patch's row and the
 * next d/4 by its column, each section's angles a fresh ladder from its own
 * position, accumulated by repeated multiplication as the reference does. */
static void rope_table(const qe_vision *v, const int *py, const int *px, long n, float *cs) {
    const int half = v->d_head / 2, sec = v->d_head / 4;
    const float step = powf(V_ROPE_BASE, -2.0f / (float)half);
    for (long i = 0; i < n; i++) {
        float *c = cs + (size_t)i * v->d_head;
        float ty = (float)py[i], tx = (float)px[i];
        for (int j = 0; j < half; j++) {
            if (j == sec)
                tx = (float)px[i]; /* the column section restarts its ladder */
            const float th = j < sec ? ty : tx;
            c[2 * j] = cosf(th);
            c[2 * j + 1] = sinf(th);
            ty *= step;
            tx *= step;
        }
    }
}

typedef struct {
    qe_vision *v;
    float *qkv;       /* N x 3E */
    const float *cs;  /* N x d_head */
    long n;
} rope_job;

static void rope_task(void *arg, int tid, int nthr) {
    rope_job *j = arg;
    const qe_vision *v = j->v;
    const int E = v->n_embd, d = v->d_head, half = d / 2;
    long lo, hi;
    qe_slice(j->n, tid, nthr, &lo, &hi);
    for (long i = lo; i < hi; i++) {
        const float *c = j->cs + (size_t)i * d;
        for (int part = 0; part < 2; part++) { /* q, then k */
            float *base = j->qkv + (size_t)i * 3 * E + (size_t)part * E;
            for (int h = 0; h < v->n_head; h++) {
                float *x = base + (size_t)h * d;
                for (int p = 0; p < half; p++) {
                    const float co = c[2 * p], si = c[2 * p + 1];
                    const float x0 = x[p], x1 = x[p + half];
                    x[p] = x0 * co - x1 * si;
                    x[p + half] = x0 * si + x1 * co;
                }
            }
        }
    }
}

#define QB 128 /* query rows a task scores at once */

typedef struct {
    qe_vision *v;
    const float *qkv;
    float *out; /* N x E */
    long n;
    float scale;
    float **scr; /* per slot: QB x N scores */
    long tasks, cursor;
} attn_job;

static void softmax_row(float *s, long n, float scale) {
    float mx = -INFINITY;
    for (long i = 0; i < n; i++)
        if (s[i] > mx)
            mx = s[i];
    const float k = scale * QE_LOG2E, off = mx * k;
    long i = 0;
    double sum = 0.0;
#if defined(__x86_64__)
    __m256 acc = _mm256_setzero_ps();
    const __m256 vk = _mm256_set1_ps(k), vo = _mm256_set1_ps(off);
    for (; i + 8 <= n; i += 8) {
        __m256 e = qe_exp2_avx2(_mm256_fmsub_ps(_mm256_loadu_ps(s + i), vk, vo));
        _mm256_storeu_ps(s + i, e);
        acc = _mm256_add_ps(acc, e);
    }
    float t[8];
    _mm256_storeu_ps(t, acc);
    for (int q = 0; q < 8; q++)
        sum += t[q];
#endif
    for (; i < n; i++) {
        s[i] = qe_exp2_f32(s[i] * k - off);
        sum += s[i];
    }
    const float inv = (float)(1.0 / sum);
    for (i = 0; i < n; i++)
        s[i] *= inv;
}

static void attn_task(void *arg, int tid, int nthr) {
    attn_job *j = arg;
    (void)nthr;
    qe_vision *v = j->v;
    const int E = v->n_embd, d = v->d_head;
    const long nqb = (j->n + QB - 1) / QB;
    float *S = j->scr[tid], *pack = v->pack[tid];
    for (;;) {
        long t = __atomic_fetch_add(&j->cursor, 1, __ATOMIC_RELAXED);
        if (t >= j->tasks)
            break;
        const int h = (int)(t / nqb);
        const long q0 = (t % nqb) * QB, nq = j->n - q0 < QB ? j->n - q0 : QB;
        const float *Q = j->qkv + (size_t)q0 * 3 * E + (size_t)h * d;
        const float *Kh = j->qkv + E + (size_t)h * d;
        const float *Vh = j->qkv + 2 * (size_t)E + (size_t)h * d;
        /* scores: the head's queries against every key, unscaled */
        gemm1(Q, 3 * E, nq, (wsrc){ Kh, 3 * E, W_F32 }, j->n, d, S, j->n, NULL, pack);
        for (long r = 0; r < nq; r++)
            softmax_row(S + (size_t)r * j->n, j->n, j->scale);
        /* and the values, read as a column view of the head's slice */
        gemm1_kblocked(S, j->n, nq, (wsrc){ Vh, 3 * E, W_COL }, d, j->n,
                       j->out + (size_t)q0 * E + (size_t)h * d, E, pack);
    }
}

/* -------------------------------------------------------------- the tower */

static const qe_tensor *need(qe_model *m, const char *name, qe_type type, long n, char *err,
                             size_t errn) {
    const qe_tensor *t = qe_tensor_by_name(m, name);
    if (!t) {
        snprintf(err, errn, "mmproj has no %s", name);
        return NULL;
    }
    long have = 1;
    for (int i = 0; i < t->ndim; i++)
        have *= (long)t->ne[i];
    if (t->type != type || (n > 0 && have != n)) {
        snprintf(err, errn, "mmproj %s is type %d with %ld values, expected type %d with %ld",
                 name, (int)t->type, have, (int)type, n);
        return NULL;
    }
    return t;
}

#define F32P(name, n)                                                                      \
    ({                                                                                     \
        const qe_tensor *t_ = need(m, name, QE_F32, n, err, errn);                         \
        t_ ? (const float *)qe_tensor_data(m, t_) : NULL;                                  \
    })
#define BF16P(name, n)                                                                     \
    ({                                                                                     \
        const qe_tensor *t_ = need(m, name, QE_BF16, n, err, errn);                        \
        t_ ? (const uint16_t *)qe_tensor_data(m, t_) : NULL;                               \
    })

qe_vision *qe_vision_open(qe_model *m, int max_tokens, char *err, size_t errn) {
    long long iv;
    double dv;
    if (!m)
        return NULL;
    {
        /* The projector type is the one fact that says which graph this is. */
        const char *pt = NULL;
        if (qe_model_get_str(m, "clip.projector_type", &pt) || !pt ||
            strcmp(pt, "qwen3vl_merger")) {
            snprintf(err, errn, "mmproj projector is '%s'; this build encodes qwen3vl_merger only",
                     pt ? pt : "(none)");
            return NULL;
        }
    }
    qe_vision *v = calloc(1, sizeof *v);
    if (!v)
        return NULL;
    v->m = m;
    v->max_tokens = max_tokens > 0 ? max_tokens : 4096;
    if (v->max_tokens < V_MIN_TOKENS)
        v->max_tokens = V_MIN_TOKENS;
#define GETI(key, dst)                                                                     \
    do {                                                                                   \
        if (qe_model_get_int(m, key, &iv)) {                                               \
            snprintf(err, errn, "mmproj has no %s", key);                                  \
            goto fail;                                                                     \
        }                                                                                  \
        dst = (int)iv;                                                                     \
    } while (0)
    int patch = 0, merge = 0;
    GETI("clip.vision.embedding_length", v->n_embd);
    GETI("clip.vision.attention.head_count", v->n_head);
    GETI("clip.vision.feed_forward_length", v->n_ff);
    GETI("clip.vision.block_count", v->n_layer);
    GETI("clip.vision.projection_dim", v->out_dim);
    GETI("clip.vision.patch_size", patch);
    GETI("clip.vision.spatial_merge_size", merge);
    if (patch != V_PATCH || merge != V_MERGE || v->n_head <= 0 || v->n_embd % v->n_head ||
        (v->n_embd / v->n_head) % 4) {
        snprintf(err, errn, "mmproj geometry (patch %d, merge %d, %d heads of %d) is not the one "
                 "this encoder is written for", patch, merge, v->n_head, v->n_embd);
        goto fail;
    }
    v->d_head = v->n_embd / v->n_head;
    v->eps = qe_model_get_float(m, "clip.vision.attention.layer_norm_epsilon", &dv) ? 1e-6f
                                                                                   : (float)dv;
    for (int c = 0; c < 3; c++) {
        v->mean[c] = 0.5f;
        v->std[c] = 0.5f;
    }
    {
        const double *arr = NULL;
        size_t na = 0;
        if (!qe_model_get_arr_real(m, "clip.vision.image_mean", &arr, &na) && na == 3)
            for (int c = 0; c < 3; c++)
                v->mean[c] = (float)arr[c];
        if (!qe_model_get_arr_real(m, "clip.vision.image_std", &arr, &na) && na == 3)
            for (int c = 0; c < 3; c++)
                v->std[c] = (float)arr[c];
    }
    {
        /* Deepstack features would be concatenated onto every row, and the trunk's
         * input is one 2560 row; a file that has any is a different graph. */
        const long long *ds = NULL;
        size_t nd = 0;
        if (!qe_model_get_arr_int(m, "clip.vision.is_deepstack_layers", &ds, &nd))
            for (size_t i = 0; i < nd; i++)
                if (ds[i]) {
                    snprintf(err, errn, "mmproj has deepstack layers, which this encoder does "
                             "not compute");
                    goto fail;
                }
    }
    const long E = v->n_embd, F = v->n_ff, PP = (long)V_PATCH * V_PATCH * 3;
    const float *pw0 = F32P("v.patch_embd.weight", PP * E);
    const float *pw1 = F32P("v.patch_embd.weight.1", PP * E);
    v->patch_b = F32P("v.patch_embd.bias", E);
    const qe_tensor *pe = need(m, "v.position_embd.weight", QE_F32, 0, err, errn);
    v->post_w = F32P("v.post_ln.weight", E);
    v->post_b = F32P("v.post_ln.bias", E);
    v->mm0_w = BF16P("mm.0.weight", 16 * E * E);
    v->mm0_b = F32P("mm.0.bias", 4 * E);
    v->mm2_w = BF16P("mm.2.weight", 4 * E * (long)v->out_dim);
    v->mm2_b = F32P("mm.2.bias", v->out_dim);
    if (!pw0 || !pw1 || !v->patch_b || !pe || !v->post_w || !v->post_b || !v->mm0_w ||
        !v->mm0_b || !v->mm2_w || !v->mm2_b)
        goto fail;
    if ((long)pe->ne[0] != E) {
        snprintf(err, errn, "mmproj position table is %ld wide, expected %ld", (long)pe->ne[0], E);
        goto fail;
    }
    v->pos_side = (int)lround(sqrt((double)pe->ne[1]));
    if ((long)v->pos_side * v->pos_side != (long)pe->ne[1]) {
        snprintf(err, errn, "mmproj position table has %ld rows, not a square",
                 (long)pe->ne[1]);
        goto fail;
    }
    v->pos_embd = qe_tensor_data(m, pe);
    /* A still image is both frames of the temporal pair, so its patch embedding is one
     * convolution by the sum of the two kernels. */
    v->patch_w = malloc((size_t)PP * E * sizeof(float));
    if (!v->patch_w) {
        snprintf(err, errn, "out of memory");
        goto fail;
    }
    for (long i = 0; i < PP * E; i++)
        v->patch_w[i] = pw0[i] + pw1[i];

    v->blk = calloc((size_t)v->n_layer, sizeof *v->blk);
    if (!v->blk)
        goto fail;
    for (int il = 0; il < v->n_layer; il++) {
        char nm[96];
        vblock *b = &v->blk[il];
#define BN(field, suffix, kind, n)                                                         \
    snprintf(nm, sizeof nm, "v.blk.%d.%s", il, suffix);                                    \
    b->field = kind(nm, n);                                                                \
    if (!b->field)                                                                         \
        goto fail;
        BN(qkv_w, "attn_qkv.weight", BF16P, 3 * E * E);
        BN(qkv_b, "attn_qkv.bias", F32P, 3 * E);
        BN(o_w, "attn_out.weight", BF16P, E * E);
        BN(o_b, "attn_out.bias", F32P, E);
        BN(up_w, "ffn_up.weight", BF16P, E * F);
        BN(up_b, "ffn_up.bias", F32P, F);
        BN(down_w, "ffn_down.weight", BF16P, E * F);
        BN(down_b, "ffn_down.bias", F32P, E);
        BN(ln1_w, "ln1.weight", F32P, E);
        BN(ln1_b, "ln1.bias", F32P, E);
        BN(ln2_w, "ln2.weight", F32P, E);
        BN(ln2_b, "ln2.bias", F32P, E);
#undef BN
    }
    v->nthr = qe_pool_size() > 0 ? qe_pool_size() : 1;
    v->emulate = getenv("QE_VISION_EMULATE") && atoi(getenv("QE_VISION_EMULATE")) > 0;
    /* A panel is K x 16 floats. The widest K a weight walk has is the merger's 4608
     * and the values walk of an attention head reads as many keys as there are
     * patches, which max_tokens bounds. */
    long kmax = 4 * E > F ? 4 * E : F;
    const long nmax = (long)v->max_tokens * 4;
    if (nmax > kmax)
        kmax = nmax;
    v->pack_cap = (size_t)kmax * 16;
    v->pack = calloc((size_t)v->nthr, sizeof *v->pack);
    if (!v->pack)
        goto fail;
    for (int t = 0; t < v->nthr; t++)
        if (!(v->pack[t] = malloc(v->pack_cap * sizeof(float)))) {
            snprintf(err, errn, "out of memory");
            goto fail;
        }
    return v;
#undef GETI
fail:
    qe_vision_close(v);
    return NULL;
}

void qe_vision_close(qe_vision *v) {
    if (!v)
        return;
    if (v->pack)
        for (int t = 0; t < v->nthr; t++)
            free(v->pack[t]);
    free(v->pack);
    free(v->blk);
    free(v->patch_w);
    free(v);
}

int qe_vision_dim(const qe_vision *v) { return v ? v->out_dim : 0; }
int qe_vision_max_tokens(const qe_vision *v) { return v ? v->max_tokens : 0; }

/* The learned positions at this grid, ggml's bilinear with aligned corners from the
 * square table, in raster order (row y, column x). */
static void positions(const qe_vision *v, int pw, int ph, float *out) {
    const int S = v->pos_side, E = v->n_embd;
    if (pw == S && ph == S) {
        memcpy(out, v->pos_embd, (size_t)S * S * E * sizeof(float));
        return;
    }
    float sf0 = (float)pw / S, sf1 = (float)ph / S;
    if (pw > 1 && S > 1)
        sf0 = (float)(pw - 1) / (S - 1);
    if (ph > 1 && S > 1)
        sf1 = (float)(ph - 1) / (S - 1);
    for (int i1 = 0; i1 < ph; i1++) {
        const float y = (float)i1 / sf1;
        long y0 = (long)floorf(y), y1 = y0 + 1;
        y0 = y0 < 0 ? 0 : y0 > S - 1 ? S - 1 : y0;
        y1 = y1 < 0 ? 0 : y1 > S - 1 ? S - 1 : y1;
        float dy = y - (float)y0;
        dy = dy < 0 ? 0 : dy > 1 ? 1 : dy;
        for (int i0 = 0; i0 < pw; i0++) {
            const float x = (float)i0 / sf0;
            long x0 = (long)floorf(x), x1 = x0 + 1;
            x0 = x0 < 0 ? 0 : x0 > S - 1 ? S - 1 : x0;
            x1 = x1 < 0 ? 0 : x1 > S - 1 ? S - 1 : x1;
            float dx = x - (float)x0;
            dx = dx < 0 ? 0 : dx > 1 ? 1 : dx;
            const float *a = v->pos_embd + ((size_t)y0 * S + x0) * E;
            const float *b = v->pos_embd + ((size_t)y0 * S + x1) * E;
            const float *c = v->pos_embd + ((size_t)y1 * S + x0) * E;
            const float *d = v->pos_embd + ((size_t)y1 * S + x1) * E;
            float *o = out + ((size_t)i1 * pw + i0) * E;
            for (int e = 0; e < E; e++)
                o[e] = a[e] * (1 - dx) * (1 - dy) + b[e] * dx * (1 - dy) + c[e] * (1 - dx) * dy +
                       d[e] * dx * dy;
        }
    }
}

typedef struct {
    const qe_vision *v;
    const float *img; /* planar, normalised */
    int W, H, pw;
    const float *pos; /* raster */
    const int *py, *px;
    float *x; /* merge order */
    long n;
} patch_job;

static void patch_task(void *arg, int tid, int nthr) {
    patch_job *j = arg;
    const qe_vision *v = j->v;
    const int E = v->n_embd;
    long lo, hi;
    qe_slice(j->n, tid, nthr, &lo, &hi);
    float pix[V_PATCH * V_PATCH * 3];
    for (long i = lo; i < hi; i++) {
        const int py = j->py[i], px = j->px[i];
        for (int c = 0; c < 3; c++)
            for (int ky = 0; ky < V_PATCH; ky++)
                memcpy(pix + (c * V_PATCH + ky) * V_PATCH,
                       j->img + ((size_t)c * j->H + (size_t)py * V_PATCH + ky) * j->W +
                           (size_t)px * V_PATCH,
                       V_PATCH * sizeof(float));
        float *o = j->x + (size_t)i * E;
        const float *pe = j->pos + ((size_t)py * j->pw + px) * E;
        for (int e = 0; e < E; e++)
            o[e] = qe_dot(v->patch_w + (size_t)e * V_PATCH * V_PATCH * 3, pix,
                          V_PATCH * V_PATCH * 3) +
                   v->patch_b[e] + pe[e];
    }
}

long qe_vision_encode(qe_vision *v, const unsigned char *rgb, int w, int h, float *out,
                      long cap_rows, int *nx_out, int *ny_out,
                      int (*progress)(void *, long, long), void *pu, qe_vision_stats *st) {
    if (!v || !rgb || w <= 0 || h <= 0 || !out)
        return -1;
    const double t0 = now_s();
    int tw, th;
    target_size(w, h, v->max_tokens, &tw, &th);
    const int nx = tw / V_ALIGN, ny = th / V_ALIGN, pw = tw / V_PATCH, ph = th / V_PATCH;
    const long n = (long)pw * ph, ntok = (long)nx * ny;
    if (ntok > cap_rows)
        return -1;
    const long E = v->n_embd, F = v->n_ff;

    /* resize: horizontal first, then vertical, each skipped when it is the identity */
    const unsigned char *src = rgb;
    unsigned char *t1 = NULL, *t2 = NULL;
    if (tw != w) {
        t1 = resize_h(src, w, h, tw);
        if (!t1)
            return -1;
        src = t1;
    }
    if (th != h) {
        t2 = resize_v(src, tw, h, th);
        if (!t2) {
            free(t1);
            return -1;
        }
        src = t2;
    }
    float *img = malloc((size_t)tw * th * 3 * sizeof(float));
    float *pos = malloc((size_t)n * E * sizeof(float));
    int *py = malloc((size_t)n * sizeof(int)), *px = malloc((size_t)n * sizeof(int));
    float *x = malloc((size_t)n * E * sizeof(float));
    float *hb = malloc((size_t)n * E * sizeof(float));
    float *qkv = malloc((size_t)n * 3 * E * sizeof(float));
    float *ff = malloc((size_t)n * (F > 4 * E ? F : 4 * E) * sizeof(float));
    float *cs = malloc((size_t)n * v->d_head * sizeof(float));
    float **scr = calloc((size_t)v->nthr, sizeof *scr);
    long rc = -1;
    if (!img || !pos || !py || !px || !x || !hb || !qkv || !ff || !cs || !scr)
        goto done;
    for (int t = 0; t < v->nthr; t++)
        if (!(scr[t] = malloc((size_t)QB * n * sizeof(float))))
            goto done;
    for (long p = 0; p < (long)tw * th; p++)
        for (int c = 0; c < 3; c++)
            img[(size_t)c * tw * th + p] =
                ((float)src[p * 3 + c] / 255.0f - v->mean[c]) / v->std[c];
    free(t1);
    free(t2);
    t1 = t2 = NULL;

    /* merge order: the four patches of a 2x2 group consecutive, groups in raster order */
    {
        long i = 0;
        for (int ty = 0; ty < ph / 2; ty++)
            for (int tx = 0; tx < pw / 2; tx++)
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++, i++) {
                        py[i] = ty * 2 + dy;
                        px[i] = tx * 2 + dx;
                    }
    }
    positions(v, pw, ph, pos);
    const double t1s = now_s();
    {
        patch_job pj = { .v = v, .img = img, .W = tw, .H = th, .pw = pw, .pos = pos, .py = py,
                         .px = px, .x = x, .n = n };
        qe_pool_run(patch_task, &pj, v->nthr);
    }
    rope_table(v, py, px, n, cs);
    const double t2s = now_s();

    const float scale = 1.0f / sqrtf((float)v->d_head);
    double t_attn = 0.0;
    for (int il = 0; il < v->n_layer; il++) {
        const vblock *b = &v->blk[il];
        layernorm(v, x, hb, n, b->ln1_w, b->ln1_b);
        gemm(v, hb, E, n, (wsrc){ b->qkv_w, E, W_BF16 }, 3 * E, E, qkv, 3 * E, b->qkv_b);
        {
            rope_job rj = { .v = v, .qkv = qkv, .cs = cs, .n = n };
            qe_pool_run(rope_task, &rj, v->nthr);
        }
        {
            const double ta = now_s();
            attn_job aj = { .v = v, .qkv = qkv, .out = hb, .n = n, .scale = scale, .scr = scr };
            aj.tasks = (long)v->n_head * ((n + QB - 1) / QB);
            qe_pool_run(attn_task, &aj, v->nthr);
            t_attn += now_s() - ta;
        }
        /* the output projection, then the residual, into ff as scratch */
        gemm(v, hb, E, n, (wsrc){ b->o_w, E, W_BF16 }, E, E, ff, E, b->o_b);
        for (long i = 0; i < n * E; i++)
            x[i] += ff[i];
        layernorm(v, x, hb, n, b->ln2_w, b->ln2_b);
        gemm(v, hb, E, n, (wsrc){ b->up_w, E, W_BF16 }, F, E, ff, F, b->up_b);
        gelu_rows(v, ff, n, F);
        gemm(v, ff, F, n, (wsrc){ b->down_w, F, W_BF16 }, E, F, hb, E, b->down_b);
        for (long i = 0; i < n * E; i++)
            x[i] += hb[i];
        if (progress && progress(pu, il + 1, v->n_layer))
            goto done;
    }
    const double t3s = now_s();
    /* the merger: post-norm every patch, four consecutive patches are one row of 4E */
    layernorm(v, x, hb, n, v->post_w, v->post_b);
    gemm(v, hb, 4 * E, ntok, (wsrc){ v->mm0_w, 4 * E, W_BF16 }, 4 * E, 4 * E, ff, 4 * E,
         v->mm0_b);
    gelu_rows(v, ff, ntok, 4 * E);
    gemm(v, ff, 4 * E, ntok, (wsrc){ v->mm2_w, 4 * E, W_BF16 }, v->out_dim, 4 * E, out,
         v->out_dim, v->mm2_b);
    const double t4s = now_s();
    if (nx_out)
        *nx_out = nx;
    if (ny_out)
        *ny_out = ny;
    if (st) {
        st->preprocess = t1s - t0;
        st->patch = t2s - t1s;
        st->blocks = t3s - t2s;
        st->merge = t4s - t3s;
        st->attention = t_attn;
        st->patches = n;
        st->width = tw;
        st->height = th;
    }
    rc = ntok;
done:
    free(t1);
    free(t2);
    if (scr)
        for (int t = 0; t < v->nthr; t++)
            free(scr[t]);
    free(scr);
    free(img);
    free(pos);
    free(py);
    free(px);
    free(x);
    free(hb);
    free(qkv);
    free(ff);
    free(cs);
    return rc;
}
