/* probe_igpu: can the integrated GPU walk a prompt chunk's routed experts faster than the
 * CPU does, and what does it cost the CPU while it does? (ideas.md 7.32.4)
 *
 * Standalone on purpose: nothing in the engine links OpenCL, and this decides whether
 * anything ever should. It builds one layer's routed expert pools in the checkpoint's own
 * block layouts and shapes - IQ3_S gate and up (512 experts x 640 rows x 2560), IQ4_NL down
 * (512 x 2560 x 640) - in page-aligned host memory the GPU reads in place, routes a chunk of
 * T tokens to top-k experts with a skewed router, sorts them into buckets the way the
 * engine's grouped pass does, and runs the bucketed walks with 4x8-bit dot products against
 * a row-scaled int8 panel. It reports GMAC/s and what one ingested token's expert stage
 * would cost at that rate (48 layers), checks a sample of outputs against a CPU reference,
 * and with --seconds S keeps walking for S seconds so a CPU ingestion can run beside it.
 *
 *   cc -O2 -o bin/probe_igpu tools/probe_igpu.c -lOpenCL -lm
 *   ./bin/probe_igpu [--chunk 512] [--topk 10] [--reps 20] [--tile 16] [--seconds 0]
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define K 2560
#define N 640
#define E 512

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (uint32_t)(rs >> 11);
}

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cl_int e_ = (x);                                                                           \
        if (e_ != CL_SUCCESS) {                                                                    \
            fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_);                       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* fp16 <-> f32, enough for a scale */
static uint16_t f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e < 113)
        return (uint16_t)s;
    return (uint16_t)(s | ((e - 112) << 10) | (m >> 13));
}
static float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x = e ? (s | ((e + 112) << 23) | (m << 13)) : s;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static const int8_t kv4[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                1,    13,   25,  38,  53,  69,  89,  113 };

/* IQ3_S's grid is 512 packed rows of four magnitudes from {1,3,...,15}; the walk's speed
 * does not depend on which rows, so the probe builds a table of the right shape. */
static uint32_t grid[512];

/* ---------------------------------------------------------------- kernels */
/* v3. The activations of a bucket are staged once per work-group in shared local memory
 * (SLM), and each 16-lane subgroup walks RPS weight rows against them, its lanes dealing a
 * row's 32-weight groups between them. gate and up are decoded together, so one activation
 * load feeds sixteen 4x8-bit dots; TILE tokens at a time keeps the accumulators inside a
 * SIMD16 lane's share of the register file. The first version read every token's
 * activations from global memory once per weight row - 17 GB of cache traffic a layer - and
 * ran at 2 % of the hardware's dot rate. */
static const char *src =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#pragma OPENCL EXTENSION cl_khr_subgroup_clustered_reduce : enable\n"
"#define K 2560\n#define N 640\n#define TILE %d\n#define RPS %d\n#define SLMT %d\n"
"__constant char kv4[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};\n"
"inline char4 iq3w(__local const uint *grid, uint qi, uint hbit, uint s4) {\n"
"  const uint gv = grid[qi | (hbit << 8)];\n"
"  const uint m4 = ((s4 & 1) * 0xffu) | ((s4 & 2) * 0x7f80u) | ((s4 & 4) * 0x3fc000u) | ((s4 & 8) * 0x1fe00000u);\n"
"  return as_char4(gv ^ m4) - as_char4(m4);\n"
"}\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void gateup_iq3s(__global const uchar *G, __global const uchar *U,\n"
"    __global const uint *grid_g, __global const int *bkt_e, __global const int *bkt_lo,\n"
"    __global const int *bkt_n, __global const int *itok, __global const char *x8,\n"
"    __global const float *xs, __global float *h) {\n"
"  __local uint grid[512];\n"
"  __local int xl[SLMT * (K / 4)];\n"
"  const int lid = get_local_id(0), lsz = get_local_size(0);\n"
"  for (int i = lid; i < 512; i += lsz) grid[i] = grid_g[i];\n"
"  const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();\n"
"  const int b = get_group_id(1);\n"
"  const int e = bkt_e[b], lo = bkt_lo[b], n = bkt_n[b];\n"
"  const long rb = (K / 256) * 110;\n"
"  const int row0 = get_group_id(0) * nsg * RPS;\n"
"  for (int s0 = 0; s0 < n; s0 += SLMT) {\n"
"    const int ns = min(SLMT, n - s0);\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int i = lid; i < ns * (K / 4); i += lsz) {\n"
"      const int t = i / (K / 4), c = i % (K / 4);\n"
"      xl[i] = ((__global const int *)(x8 + (long)itok[lo + s0 + t] * K))[c];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int r = 0; r < RPS; r++) {\n"
"      const int row = row0 + r * nsg + sg;\n"
"      if (row >= N) continue;\n"
"      __global const uchar *wg = G + ((long)e * N + row) * rb;\n"
"      __global const uchar *wu = U + ((long)e * N + row) * rb;\n"
"      for (int t0 = 0; t0 < ns; t0 += TILE) {\n"
"        float ag[TILE], au[TILE];\n"
"        __attribute__((opencl_unroll_hint)) for (int t = 0; t < TILE; t++) { ag[t] = 0; au[t] = 0; }\n"
"        for (int gi = lane; gi < K / 32; gi += 16) {\n"
"          const int sb = gi >> 3, g0 = gi & 7;\n"
"          __global const uchar *bg = wg + sb * 110, *bu = wu + sb * 110;\n"
"          const float dg = vload_half(0, (__global const half *)bg) * (1 + 2 * ((bg[106 + (g0 >> 1)] >> (4 * (g0 & 1))) & 0xf));\n"
"          const float du = vload_half(0, (__global const half *)bu) * (1 + 2 * ((bu[106 + (g0 >> 1)] >> (4 * (g0 & 1))) & 0xf));\n"
"          const uint2 qg = as_uint2(vload8(0, bg + 2 + g0 * 8));\n"
"          const uint2 qu = as_uint2(vload8(0, bu + 2 + g0 * 8));\n"
"          const uint hg = bg[66 + g0], hu = bu[66 + g0];\n"
"          const uint sgg = as_uint(vload4(0, bg + 74 + g0 * 4));\n"
"          const uint sgu = as_uint(vload4(0, bu + 74 + g0 * 4));\n"
"          char4 w[8], v[8];\n"
"          __attribute__((opencl_unroll_hint)) for (int k = 0; k < 8; k++) {\n"
"            const uint qq = k < 4 ? qg.x : qg.y, qv = k < 4 ? qu.x : qu.y;\n"
"            w[k] = iq3w(grid, (qq >> (8 * (k & 3))) & 0xff, (hg >> k) & 1, (sgg >> (4 * k)) & 0xf);\n"
"            v[k] = iq3w(grid, (qv >> (8 * (k & 3))) & 0xff, (hu >> k) & 1, (sgu >> (4 * k)) & 0xf);\n"
"          }\n"
"          __attribute__((opencl_unroll_hint)) for (int t = 0; t < TILE; t++) {\n"
"            __local const int *xr = xl + min(t0 + t, ns - 1) * (K / 4) + gi * 8;\n"
"            int a = 0, c = 0;\n"
"            __attribute__((opencl_unroll_hint)) for (int k = 0; k < 8; k++) { const char4 xv = as_char4(xr[k]); a += dot(w[k], xv); c += dot(v[k], xv); }\n"
"            ag[t] += dg * a; au[t] += du * c;\n"
"          }\n"
"        }\n"
"        __attribute__((opencl_unroll_hint)) for (int t = 0; t < TILE; t++) {\n"
"          const float g = sub_group_reduce_add(ag[t]), u = sub_group_reduce_add(au[t]);\n"
"          if (lane == 0 && t0 + t < ns) {\n"
"            const int it = lo + s0 + t0 + t;\n"
"            const float sc = xs[itok[it]];\n"
"            const float gg = g * sc, uu = u * sc;\n"
"            h[(long)it * N + row] = gg / (1.0f + exp(-gg)) * uu;\n"
"          }\n"
"        }\n"
"      }\n"
"    }\n"
"  }\n"
"}\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void down_iq4nl(__global const uchar *D, __global const int *bkt_e,\n"
"    __global const int *bkt_lo, __global const int *bkt_n, __global const char *h8,\n"
"    __global const float *hs, __global float *y) {\n"
"  __local int hl[SLMT * (N / 4)];\n"
"  const int lid = get_local_id(0), lsz = get_local_size(0);\n"
"  const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();\n"
"  const int sl = lane & 3;\n"
"  const int b = get_group_id(1);\n"
"  const int e = bkt_e[b], lo = bkt_lo[b], n = bkt_n[b];\n"
"  const long rb = (N / 32) * 18;\n"
"  const int row0 = get_group_id(0) * nsg * 4 * RPS;\n"
"  for (int s0 = 0; s0 < n; s0 += SLMT) {\n"
"    const int ns = min(SLMT, n - s0);\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int i = lid; i < ns * (N / 4); i += lsz)\n"
"      hl[i] = ((__global const int *)(h8 + (long)(lo + s0) * N))[i];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int r = 0; r < RPS; r++) {\n"
"      const int row = row0 + (r * nsg + sg) * 4 + (lane >> 2);\n"
"      __global const uchar *wr = D + ((long)e * K + min(row, K - 1)) * rb;\n"
"      for (int t0 = 0; t0 < ns; t0 += TILE) {\n"
"        float a[TILE];\n"
"        __attribute__((opencl_unroll_hint)) for (int t = 0; t < TILE; t++) a[t] = 0;\n"
"        for (int bl = sl; bl < N / 32; bl += 4) {\n"
"          __global const uchar *blk = wr + bl * 18;\n"
"          const float d = vload_half(0, (__global const half *)blk);\n"
"          char4 w[8];\n"
"          __attribute__((opencl_unroll_hint)) for (int k = 0; k < 4; k++) {\n"
"            const uchar4 q = vload4(0, blk + 2 + 4 * k);\n"
"            w[k] = (char4)(kv4[q.x & 15], kv4[q.y & 15], kv4[q.z & 15], kv4[q.w & 15]);\n"
"            w[k + 4] = (char4)(kv4[q.x >> 4], kv4[q.y >> 4], kv4[q.z >> 4], kv4[q.w >> 4]);\n"
"          }\n"
"          __attribute__((opencl_unroll_hint)) for (int t = 0; t < TILE; t++) {\n"
"            __local const int *xr = hl + min(t0 + t, ns - 1) * (N / 4) + bl * 8;\n"
"            int acc = 0;\n"
"            __attribute__((opencl_unroll_hint)) for (int k = 0; k < 8; k++) acc += dot(w[k], as_char4(xr[k]));\n"
"            a[t] += d * acc;\n"
"          }\n"
"        }\n"
"        __attribute__((opencl_unroll_hint)) for (int t = 0; t < TILE; t++) {\n"
"          const float vv = sub_group_clustered_reduce_add(a[t], 4);\n"
"          if (sl == 0 && t0 + t < ns && row < K)\n"
"            y[(long)(lo + s0 + t0 + t) * K + row] = vv * hs[lo + s0 + t0 + t];\n"
"        }\n"
"      }\n"
"    }\n"
"  }\n"
"}\n";

/* CPU references for one output, the same decode in plain C. */
static double ref_gate(const uint8_t *G, int e, int row, const int8_t *x) {
    const long rb = (K / 256) * 110;
    const uint8_t *w = G + ((long)e * N + row) * rb;
    double acc = 0;
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = w + sb * 110;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const double d = h2f(dh);
        const uint8_t *qs = blk + 2, *qh = blk + 66, *sg = blk + 74, *sc = blk + 106;
        for (int g0 = 0; g0 < 8; g0++) {
            const double db = d * (1 + 2 * ((sc[g0 >> 1] >> (4 * (g0 & 1))) & 0xf));
            long acci = 0;
            for (int k = 0; k < 8; k++) {
                const uint32_t gv = grid[qs[g0 * 8 + k] | (((qh[g0] >> k) & 1) << 8)];
                const int sbits = sg[g0 * 4 + (k >> 1)] >> ((k & 1) * 4);
                for (int j = 0; j < 4; j++) {
                    int v = (int)((gv >> (8 * j)) & 0xff);
                    if ((sbits >> j) & 1)
                        v = -v;
                    acci += v * x[sb * 256 + g0 * 32 + k * 4 + j];
                }
            }
            acc += db * (double)acci;
        }
    }
    return acc;
}

static double ref_down(const uint8_t *D, int e, int row, const int8_t *h) {
    const long rb = (N / 32) * 18;
    const uint8_t *w = D + ((long)e * K + row) * rb;
    double acc = 0;
    for (int bl = 0; bl < N / 32; bl++) {
        const uint8_t *blk = w + bl * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        long acci = 0;
        for (int k = 0; k < 4; k++)
            for (int j = 0; j < 4; j++) {
                const uint8_t q = blk[2 + k * 4 + j];
                acci += kv4[q & 15] * h[bl * 32 + k * 4 + j];
                acci += kv4[q >> 4] * h[bl * 32 + 16 + k * 4 + j];
            }
        acc += h2f(dh) * (double)acci;
    }
    return acc;
}

static void *host_alloc(size_t n) {
    void *p = NULL;
    n = (n + 4095) & ~(size_t)4095;
    if (posix_memalign(&p, 4096, n))
        return NULL;
    mlock(p, n);
    return p;
}

int main(int argc, char **argv) {
    int T = 512, topk = 10, reps = 20, tile = 4, rps = 4, slmt = 20, wg = 256;
    double seconds = 0, duty = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--chunk") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rps") && i + 1 < argc) rps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slm") && i + 1 < argc) slmt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wg") && i + 1 < argc) wg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--duty") && i + 1 < argc) duty = atof(argv[++i]);
        else { fprintf(stderr, "usage: %s [--chunk T] [--topk k] [--reps n] [--tile t] [--seconds s]\n", argv[0]); return 2; }
    }
    for (int i = 0; i < 512; i++) {
        uint32_t g = 0;
        for (int j = 0; j < 4; j++)
            g |= (uint32_t)(1 + 2 * (rnd() % 8)) << (8 * j);
        grid[i] = g;
    }

    /* weights: one layer's routed pools */
    const size_t gsz = (size_t)E * N * (K / 256) * 110, dsz = (size_t)E * K * (N / 32) * 18;
    uint8_t *G = host_alloc(gsz), *U = host_alloc(gsz), *D = host_alloc(dsz);
    for (size_t i = 0; i < gsz; i++) { G[i] = (uint8_t)rnd(); U[i] = (uint8_t)rnd(); }
    for (size_t i = 0; i < dsz; i++) D[i] = (uint8_t)rnd();
    /* sane scales: every block's fp16 d a small positive number */
    for (size_t r = 0; r < (size_t)E * N * (K / 256); r++) {
        uint16_t h = f2h(1e-3f * (1 + rnd() % 100));
        memcpy(G + r * 110, &h, 2);
        h = f2h(1e-3f * (1 + rnd() % 100));
        memcpy(U + r * 110, &h, 2);
    }
    for (size_t r = 0; r < (size_t)E * K * (N / 32); r++) {
        uint16_t h = f2h(1e-3f * (1 + rnd() % 100));
        memcpy(D + r * 18, &h, 2);
    }

    /* routing: a skewed router, top-k distinct experts a token, then buckets */
    const int items = T * topk;
    int *tok_e = malloc(sizeof(int) * items);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < topk; k++) {
            int e, dup;
            do {
                const double u = (rnd() + 0.5) / 2097152.0; /* (0,1) */
                e = (int)(E * pow(u, 1.6)) % E;              /* hot experts get more rows */
                dup = 0;
                for (int j = 0; j < k; j++)
                    dup |= tok_e[t * topk + j] == e;
            } while (dup);
            tok_e[t * topk + k] = e;
        }
    int cnt[E] = { 0 };
    for (int i = 0; i < items; i++) cnt[tok_e[i]]++;
    int *bkt_e = host_alloc(sizeof(int) * E), *bkt_lo = host_alloc(sizeof(int) * E),
        *bkt_n = host_alloc(sizeof(int) * E), *itok = host_alloc(sizeof(int) * items);
    int nb = 0, lo = 0, maxn = 0;
    int start[E];
    for (int e = 0; e < E; e++) {
        start[e] = lo;
        if (cnt[e]) {
            bkt_e[nb] = e; bkt_lo[nb] = lo; bkt_n[nb] = cnt[e];
            if (cnt[e] > maxn) maxn = cnt[e];
            nb++;
            lo += cnt[e];
        }
    }
    int fill[E];
    memcpy(fill, start, sizeof fill);
    for (int i = 0; i < items; i++) itok[fill[tok_e[i]]++] = i / topk;

    int8_t *x8 = host_alloc((size_t)T * K);
    float *xs = host_alloc(sizeof(float) * T);
    for (size_t i = 0; i < (size_t)T * K; i++) x8[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int t = 0; t < T; t++) xs[t] = 0.01f;
    int8_t *h8 = host_alloc((size_t)items * N);
    float *hs = host_alloc(sizeof(float) * items);
    for (size_t i = 0; i < (size_t)items * N; i++) h8[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int i = 0; i < items; i++) hs[i] = 0.01f;
    float *h = host_alloc(sizeof(float) * (size_t)items * N);
    float *y = host_alloc(sizeof(float) * (size_t)items * K);
    memset(h, 0, sizeof(float) * (size_t)items * N);
    memset(y, 0, sizeof(float) * (size_t)items * K);

    printf("chunk %d tokens, top-%d of %d experts: %d buckets, largest %d rows, mean %.1f\n", T,
           topk, E, nb, maxn, (double)items / nb);
    printf("pools %.0f + %.0f + %.0f MiB, host memory read in place\n", gsz / 1048576.0,
           gsz / 1048576.0, dsz / 1048576.0);

    /* OpenCL */
    cl_platform_id pl;
    cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &pl, NULL));
    CHECK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    char name[256];
    cl_uint cu;
    CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, NULL));
    CHECK(clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL));
    printf("device %s, %u EUs\n", name, cu);
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    CHECK(err);
    char *srcbuf = malloc(strlen(src) + 128);
    sprintf(srcbuf, src, tile, rps, slmt);
    const char *srcp = srcbuf;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, NULL, &err);
    CHECK(err);
    if (clBuildProgram(prog, 1, &dev, "-cl-std=CL3.0 -cl-mad-enable", NULL, NULL) != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
#define HB(p, n) clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (((n) + 4095) & ~(size_t)4095), (p), &err)
    cl_mem bG = HB(G, gsz); CHECK(err);
    cl_mem bU = HB(U, gsz); CHECK(err);
    cl_mem bD = HB(D, dsz); CHECK(err);
    cl_mem bgrid = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof grid, grid, &err); CHECK(err);
    cl_mem be = HB(bkt_e, sizeof(int) * E); CHECK(err);
    cl_mem blo = HB(bkt_lo, sizeof(int) * E); CHECK(err);
    cl_mem bn = HB(bkt_n, sizeof(int) * E); CHECK(err);
    cl_mem bit = HB(itok, sizeof(int) * items); CHECK(err);
    cl_mem bx = HB(x8, (size_t)T * K); CHECK(err);
    cl_mem bxs = HB(xs, sizeof(float) * T); CHECK(err);
    cl_mem bh = HB(h, sizeof(float) * (size_t)items * N); CHECK(err);
    cl_mem bh8 = HB(h8, (size_t)items * N); CHECK(err);
    cl_mem bhs = HB(hs, sizeof(float) * items); CHECK(err);
    cl_mem by = HB(y, sizeof(float) * (size_t)items * K); CHECK(err);
    cl_kernel kg = clCreateKernel(prog, "gateup_iq3s", &err); CHECK(err);
    cl_kernel kd = clCreateKernel(prog, "down_iq4nl", &err); CHECK(err);
    cl_mem ag[] = { bG, bU, bgrid, be, blo, bn, bit, bx, bxs, bh };
    for (int i = 0; i < 10; i++) CHECK(clSetKernelArg(kg, i, sizeof(cl_mem), &ag[i]));
    cl_mem ad[] = { bD, be, blo, bn, bh8, bhs, by };
    for (int i = 0; i < 7; i++) CHECK(clSetKernelArg(kd, i, sizeof(cl_mem), &ad[i]));
    /* a work-group is wg/16 subgroups; gate/up covers nsg*rps rows of a bucket, down
     * nsg*4*rps rows */
    const size_t nsg = (size_t)wg / 16;
    const size_t ngu = (N + nsg * rps - 1) / (nsg * rps), ndn = (K + nsg * 4 * rps - 1) / (nsg * 4 * rps);
    const size_t lg[2] = { (size_t)wg, 1 }, gg[2] = { ngu * wg, (size_t)nb };
    const size_t gd[2] = { ndn * wg, (size_t)nb };
    printf("kernel: tile %d tokens, %d rows a subgroup, %d tokens staged, %d work items a group\n",
           tile, rps, slmt, wg);

    const double mac_gu = 2.0 * items * N * (double)K, mac_d = (double)items * K * N;
    /* warm */
    CHECK(clEnqueueNDRangeKernel(q, kg, 2, NULL, gg, lg, 0, NULL, NULL));
    CHECK(clEnqueueNDRangeKernel(q, kd, 2, NULL, gd, lg, 0, NULL, NULL));
    CHECK(clFinish(q));

    /* correctness: sampled outputs against the C reference */
    double worst = 0, wgu = 0, wdn = 0;
    for (int s = 0; s < 64; s++) {
        const int b = rnd() % nb, row = rnd() % N, j = rnd() % bkt_n[b];
        const int it = bkt_lo[b] + j, tk = itok[it];
        const double g = ref_gate(G, bkt_e[b], row, x8 + (long)tk * K) * xs[tk];
        const double u = ref_gate(U, bkt_e[b], row, x8 + (long)tk * K) * xs[tk];
        const double want = g / (1 + exp(-g)) * u;
        const double rel = fabs(h[(long)it * N + row] - want) / (fabs(want) + 1e-6);
        if (rel > worst) worst = rel;
        if (rel > wgu) wgu = rel;
        const int r2 = rnd() % K;
        const double wd = ref_down(D, bkt_e[b], r2, h8 + (long)it * N) * hs[it];
        const double rel2 = fabs(y[(long)it * K + r2] - wd) / (fabs(wd) + 1e-6);
        if (rel2 > worst) worst = rel2;
        if (rel2 > wdn) wdn = rel2;
    }
    printf("check: 128 sampled outputs against the C reference, worst relative error %.2e %s"
           " (gate/up %.1e, down %.1e)\n", worst, worst < 1e-3 ? "ok" : "FAIL", wgu, wdn);

    double tg = 1e30, td = 1e30;
    for (int r = 0; r < reps; r++) {
        double t1 = now();
        CHECK(clEnqueueNDRangeKernel(q, kg, 2, NULL, gg, lg, 0, NULL, NULL));
        CHECK(clFinish(q));
        double t2 = now();
        CHECK(clEnqueueNDRangeKernel(q, kd, 2, NULL, gd, lg, 0, NULL, NULL));
        CHECK(clFinish(q));
        double t3 = now();
        if (t2 - t1 < tg) tg = t2 - t1;
        if (t3 - t2 < td) td = t3 - t2;
    }
    const double tl = tg + td;
    printf("gate+up IQ3_S: %.2f ms, %.0f GMAC/s\n", tg * 1e3, mac_gu / tg / 1e9);
    printf("down IQ4_NL:   %.2f ms, %.0f GMAC/s\n", td * 1e3, mac_d / td / 1e9);
    printf("one layer's routed experts for %d tokens: %.2f ms, %.0f GMAC/s\n", T, tl * 1e3,
           (mac_gu + mac_d) / tl / 1e9);
    printf("=> %.2f ms of an ingested token's expert stage over 48 layers (the CPU: ~7.2 ms)\n",
           48.0 * tl * 1e3 / T);
    fflush(stdout);

    if (seconds > 0) {
        /* Keep walking for `seconds`, reporting every ~5 s with the wall clock; with --duty D
         * walk D seconds and rest D seconds, so a CPU run beside it sees both states. */
        double t0 = now(), last = t0;
        long n = 0;
        double acc = 0;
        while (now() - t0 < seconds) {
            if (duty > 0 && fmod(now() - t0, 2 * duty) >= duty) {
                struct timespec nap = { 0, 200000000 };
                nanosleep(&nap, NULL);
                if (now() - last > 5.0) {
                    struct timespec rt;
                    clock_gettime(CLOCK_REALTIME, &rt);
                    printf("  at %.1f  idle\n", rt.tv_sec + 1e-9 * rt.tv_nsec);
                    fflush(stdout);
                    last = now();
                }
                continue;
            }
            double t1 = now();
            CHECK(clEnqueueNDRangeKernel(q, kg, 2, NULL, gg, lg, 0, NULL, NULL));
            CHECK(clEnqueueNDRangeKernel(q, kd, 2, NULL, gd, lg, 0, NULL, NULL));
            CHECK(clFinish(q));
            acc += now() - t1;
            n++;
            if (now() - last > 5.0 && n) {
                struct timespec rt;
                clock_gettime(CLOCK_REALTIME, &rt);
                printf("  at %.1f  walking: %.2f ms a layer, %.0f GMAC/s, %.2f ms a token over 48 layers\n",
                       rt.tv_sec + 1e-9 * rt.tv_nsec, acc / n * 1e3,
                       (mac_gu + mac_d) / (acc / n) / 1e9, 48.0 * acc / n * 1e3 / T);
                fflush(stdout);
                last = now();
                acc = 0;
                n = 0;
            }
        }
    }
    return 0;
}
