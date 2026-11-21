/* qe - qwen4exp inference engine
 *
 * CPU-only, bandwidth-first design. See docs/DESIGN.md for the measured
 * budget this code is tuned against.
 */
#ifndef QE_H
#define QE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ types */

/* ggml type ids. Values are fixed by the GGUF on-disk format. The
 * (block_ne, bytes_per_block) table was recovered from the shard offsets of
 * the checkpoint itself (0 chain errors against file size) rather than
 * assumed, because a wrong entry silently misreads every tensor. */
typedef enum {
    QE_F32 = 0,
    QE_F16 = 1,
    QE_Q4_0 = 2,
    QE_Q4_1 = 3,
    QE_Q5_0 = 6,
    QE_Q5_1 = 7,
    QE_Q8_0 = 8,
    QE_Q2_K = 10,
    QE_Q3_K = 11,
    QE_Q4_K = 12,
    QE_Q5_K = 13,
    QE_Q6_K = 14,
    QE_IQ2_XXS = 16,
    QE_IQ2_XS = 17,
    QE_IQ3_XXS = 18,
    QE_IQ1_S = 19,
    QE_IQ4_NL = 20,
    QE_IQ3_S = 21,
    QE_IQ2_S = 22,
    QE_IQ4_XS = 23,
    QE_IQ1_M = 29,
    QE_BF16 = 30,
    QE_TYPE_COUNT = 31
} qe_type;

typedef struct {
    uint32_t block_ne; /* elements per block */
    uint32_t block_sz; /* bytes per block */
    const char *name;
} qe_type_info;

const qe_type_info *qe_type_info_of(qe_type t);
long qe_type_bpw_x1000(qe_type t); /* bits per weight * 1000 */

/* On-disk block layouts (little endian, packed). */
typedef struct {
    uint16_t d;
} qe_half;

typedef struct {
    qe_half d;
    int8_t qs[32];
} qe_block_q8_0; /* 34 bytes / 32 weights, 8.5 bpw */

typedef struct {
    uint8_t ql[128]; /* low 4 bits */
    uint8_t qh[64];  /* high 2 bits, four weights per byte */
    int8_t scales[16];
    qe_half d;
} qe_block_q6_k; /* 210 bytes / 256 weights, 6.5625 bpw */

typedef struct {
    qe_half d;
    uint8_t qh[4]; /* the fifth bit of each quant: bit j feeds weight j, bit j+12 weight j+16 */
    uint8_t qs[16]; /* low nibbles are weights 0..15, high nibbles weights 16..31 */
} qe_block_q5_0; /* 22 bytes / 32 weights, 5.5 bpw */

typedef struct {
    qe_half d;        /* super-block scale for the eight sub-block scales */
    qe_half dmin;     /* ... and for the eight sub-block minima */
    uint8_t scales[12]; /* eight 6-bit scales and eight 6-bit minima, packed */
    uint8_t qs[128]; /* one byte feeds two weights 32 apart: low nibble first, high second */
} qe_block_q4_k; /* 144 bytes / 256 weights, 4.5 bpw */

typedef struct {
    qe_half d;
    uint8_t qs[16]; /* two 4-bit codebook indices per byte */
} qe_block_iq4_nl; /* 18 bytes / 32 weights, 4.5 bpw */

typedef struct {
    qe_half d;
    uint8_t qs[64];    /* codebook index, low 8 bits */
    uint8_t qh[8];     /* codebook index, 9th bit, one per pair of qs bytes */
    uint8_t signs[32]; /* one sign per weight */
    uint8_t scales[4]; /* two 4-bit scales per byte */
} qe_block_iq3_s; /* 110 bytes / 256 weights, 3.4375 bpw */

typedef struct {
    qe_half d;
    uint16_t scales_h; /* high 2 bits of the six-bit sub-block scales */
    uint8_t scales_l[4];
    uint8_t qs[128];
} qe_block_iq4_xs; /* 136 bytes / 256 weights, 4.25 bpw */

_Static_assert(sizeof(qe_block_q8_0) == 34, "q8_0 block must be 34 bytes");
_Static_assert(sizeof(qe_block_q6_k) == 210, "q6_K block must be 210 bytes");
_Static_assert(sizeof(qe_block_q5_0) == 22, "q5_0 block must be 22 bytes");
_Static_assert(sizeof(qe_block_q4_k) == 144, "q4_K block must be 144 bytes");
_Static_assert(sizeof(qe_block_iq4_nl) == 18, "iq4_nl block must be 18 bytes");
_Static_assert(sizeof(qe_block_iq3_s) == 110, "iq3_s block must be 110 bytes");
_Static_assert(sizeof(qe_block_iq4_xs) == 136, "iq4_xs block must be 136 bytes");

/* One unit of pool work: `nthreads` slices of a task, each run as
 * fn(arg, slot, nthreads). See qe_pool_run for the contract and
 * qe_pool_run_items for running several of them in one dispatch. */
typedef void (*qe_task_fn)(void *arg, int tid, int nthreads);

#define QE_POOL_MAX_ITEMS 32

typedef struct {
    qe_task_fn fn;
    void      *arg;
    int        nthreads;
    uint32_t   deps; /* bit i set => item i's every slot runs before this one's first */
} qe_pool_item;

/* --------------------------------------------------------------- gguf file */

#define QE_MAX_SHARDS 8
#define QE_MAX_NAME 128

typedef struct qe_model qe_model;

typedef struct {
    char name[QE_MAX_NAME];
    uint64_t ne[4];
    int ndim;
    qe_type type;
    int shard;   /* index into the owner's shards[] */
    uint64_t off; /* byte offset within that shard */
    uint64_t bytes;
    /* Flat per-expert view (MoE only): expert i starts at i*expert_stride. */
    uint32_t n_experts;
    uint64_t expert_stride;
    int is_replicated;
    /* Which model this row was read out of. A forward pass holds one model and walks
     * a draft head's extra layer out of another (docs/DESIGN.md §4), and an offset is
     * nothing without the volume it indexes, so a row carries its own home and
     * `qe_tensor_data` answers from that rather than from the handle it was handed. */
    qe_model *owner;
} qe_tensor;

typedef struct {
    const char *path;
    int fd;
    void *map;
    size_t map_len;   /* length of the mapping */
    size_t file_len;  /* st_size */
    size_t alignment; /* general.alignment for this volume */
    size_t data_off;  /* start of the tensor data section */
    uint64_t off_base;/* the smallest tensor offset in this volume */
    uint64_t n_tensors;
    int first;        /* volume 0, holds the authoritative metadata */
    int staged;       /* map is a private RAM arena, not a file mapping */
} qe_shard;

typedef struct qe_model qe_model;

qe_model *qe_open(const char *path, unsigned flags);
void qe_close(qe_model *m);

#define QE_M_NONE 0u /* no open flags: stage however qe_open thinks best */

/* How qe_open gets the volume bytes. The default matters more on this host than
 * any kernel flag here: the checkpoint lives on ZFS, and the pool's ARC is
 * capped at 1 GiB against an 87 GiB file, so a demand-paged MAP_SHARED mapping
 * is not a slow cache, it is no cache - every expert row read is a raidz2
 * stripe seek. Staging the volumes into RAM once, at open, is the only way the
 * routed pools can be read at memory speed at all.
 *
 * "No opinion" is therefore the zero value, so that a caller who passes nothing
 * interesting in cannot accidentally become a process that reads the model from
 * disk between tokens: 0 means QE_STAGE_AUTO, and demand-paged mapping has to be
 * asked for by name. */
#define QE_STAGE_AUTO 0u /* the best backing that keeps every byte resident */
#define QE_STAGE_RAM 1u  /* private anonymous arena, read in, then mlock()ed */
#define QE_STAGE_HUGE 2u /* as RAM but 2 MiB hugetlb pages (needs a pool) */
#define QE_STAGE_MMAP 4u /* demand-paged MAP_SHARED: fast open, disk per miss.
                          * Metadata-only use; never a good reading path. */
/* Ask for a stage mode qe_open cannot honour: without STRICT it falls back down
 * the list (HUGE -> RAM) and qe_model_stage_method() says which it got. With
 * STRICT there is no fallback to demand-paged mapping at all: a mode that cannot
 * be honoured is an error naming the two numbers that decided it, because a
 * process holding a checkpoint has to have the checkpoint in memory. AUTO with
 * STRICT - what an engine asks for - means hugetlbfs arena if a pool is big
 * enough, else private RAM if the checkpoint fits with room to spare, else no
 * engine. */
#define QE_STAGE_STRICT 8u
/* Ask for 2 MiB transparent huge pages inside whichever backing was chosen.
 * Deliberately orthogonal instead of a fourth mode: THP are opportunistic (a
 * pool reservation by root is not), and the point of the split is to be able to
 * measure whether 2 MiB pages are worth anything here at all. Set with
 * QE_THP=1. */
#define QE_STAGE_THP 16u
/* This open is for metadata and nothing else: headers, tensor table, tokenizer.
 * It is the one thing QE_STAGE_MMAP is good for: it answers "what is in this
 * file" without laying a hand on the weight bytes. Paired with MMAP it says "I
 * know these bytes are not resident, and I am not going to read them on behalf
 * of a token"; without it, an engine refuses to open a checkpoint that is not in
 * memory. */
#define QE_STAGE_DESCRIBE 32u
/* Which mode bits are set (0 = QE_STAGE_AUTO, no opinion). */
#define QE_STAGE_MODE(f) ((f) & (QE_STAGE_RAM | QE_STAGE_HUGE | QE_STAGE_MMAP))

const char *qe_strerror(void);

size_t qe_tensor_count(qe_model *m);
int qe_model_volumes(qe_model *m); /* resolved shard volumes (1 for a single file) */
const qe_tensor *qe_tensor_by_index(qe_model *m, size_t i);
const qe_tensor *qe_tensor_by_name(qe_model *m, const char *name);
/* Metadata lookups. Returns 0 and writes *out on success, -1 when the key is
 * absent or holds another type - so test `== 0`, not truthiness. */
int qe_model_get_int(qe_model *m, const char *key, long long *out);
int qe_model_get_float(qe_model *m, const char *key, double *out);
int qe_model_get_str(qe_model *m, const char *key, const char **out);
int qe_model_get_arr_int(qe_model *m, const char *key, const long long **out, size_t *n);
int qe_model_get_arr_str(qe_model *m, const char *key, const char *const **out, size_t *n);
int qe_model_get_arr_len(qe_model *m, const char *key, size_t *n);

/* Host pointer for a tensor's bytes, or NULL if not resident. */
/* Host pointer for a tensor's bytes, or NULL if not mapped. Tensor offsets in
 * the info table count from the start of that volume's data section, so this
 * is map + data_off + (off - off_base): dropping the header term displaces
 * every read by the size of the volume header, which is small enough to look
 * like plausible data and enough to misalign every quantization block. */
const void *qe_tensor_data(qe_model *m, const qe_tensor *t);

/* Page-fault / TLB behaviour diagnostics for one tensor. */
void qe_tensor_describe(qe_model *m, const qe_tensor *t, char *buf, size_t n);

/* Demand-page (and optionally mlock) a chosen set of tensors. Returns MiB
 * touched, or -1. Prefer this over warming whole volumes: the checkpoint is
 * 87 GiB on spinning disk. */
int qe_model_warm(qe_model *m, const char *const *names, int nnames, int mlock_it);

/* What qe_open actually did with the volume bytes: "mmap", "ram" or "hugetlb",
 * which is the difference between reading an expert row in 12 us and in 4 ms on
 * this host. `cached` counts bytes that came back from a previous run's staging
 * arena instead of off disk. Nothing here times the load: where the bytes ended
 * up is the fact worth reporting, how long the device took is not a quantity
 * this engine measures or optimises. */
const char *qe_model_stage_method(qe_model *m);
size_t qe_model_stage_bytes(qe_model *m);
size_t qe_model_stage_cached(qe_model *m);
/* 0 = reclaimable, 1 = mlock()ed, 2 = hugetlb (never pageable in the first
 * place, so there was nothing to pin). */
int qe_model_stage_pinned(qe_model *m);

/* ----------------------------------------------------------------- kernels */

/* y[r] = dot(W row r, x). W rows are contiguous, K elements per row,
 * ldd expressed in blocks (elements for f32).
 *
 * The three buffers are marked restrict because no caller in the engine has
 * ever overlapped them: W is staged checkpoint bytes, x is an activation row,
 * y is a scratch row, and an alias between any two would change the product
 * rather than merely the speed. That is the audit this header is asserting, and
 * the same holds for the primitives below: only buffers a call site provably
 * keeps apart are marked, which is why the norm and activation pairs below are
 * not - in-place is a supported use of those (`qe_silu(f->lo, f->lo, ...)`,
 * `qe_rms_norm_grouped(iq, ..., iq, ...)`), `make fast` pins it, and marking
 * such a pair would be a bug wearing a hint.
 *
 * What the marking bought is counted, not hoped. The quantized and f32 kernels
 * come out of the compiler instruction for instruction the same either way -
 * they are hand-written intrinsics, so there was nothing left to ask - and four
 * of the elementwise primitives lose a duplicate version that only the alias
 * question was causing: qe_axpy 199 instructions to 137,
 * qe_stream_inject_add 293 to 210, qe_rope_apply 414 to 400,
 * qe_rms_norm_grouped 461 to 451. The read walk is where it was before,
 * 75-78 GB/s, which is the honest summary: this is the compiler stopping having
 * to ask, not the memory controller going faster. */
void qe_gemv_q8_0(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks);
void qe_gemv_f32(const void *restrict W, const float *restrict x, float *restrict y,
                 long n_rows, long K, long ldd);
void qe_gemv_q6_k(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks);
void qe_gemv_q4_k(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks);
void qe_gemv_q5_0(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd_blocks);
void qe_gemv_iq4_nl(const void *restrict W, const float *restrict x, float *restrict y,
                    long n_rows, long K, long ldd_blocks);
void qe_gemv_iq3_s(const void *restrict W, const float *restrict x, float *restrict y,
                   long n_rows, long K, long ldd_blocks);
void qe_gemv_iq4_xs(const void *restrict W, const float *restrict x, float *restrict y,
                    long n_rows, long K, long ldd_blocks);
void qe_gemv_bf16(const void *restrict W, const float *restrict x, float *restrict y,
                  long n_rows, long K, long ldd);

/* Q8_0 with T activations through one dequantised weight row: the block is
 * widened, converted and scaled once and then multiplied against all T rows, so
 * the arithmetic a chunk pays per token stops growing with T. Chunking the walk
 * already shares the weight bytes; this shares the work done on them, which is
 * what the ALU-bound part of ingestion is made of.
 *
 * The x/y layout is qe_gemv_multi_t's: consecutive tokens x_stride and y_stride
 * apart, every y[t][r] written exactly once. T must be at least 2 to be worth it;
 * T of 1 falls through to the single-row kernel.
 *
 * Not bit-identical to T calls of qe_gemv_q8_0. Two accumulator chains per token
 * is the most that fits in sixteen ymm registers alongside a dequantised block
 * (the single-row kernel's four would need 4T+1), and a different number of chains
 * is a different order of summation. Measured against a double reference: 2.3e-5
 * worst relative over the shapes bench_kern walks, 9e-5 across the 512 MiB span
 * bench_alu times. */
void qe_gemv_q8_0_multi(const void *restrict W, const float *restrict x, float *restrict y,
                        long n_rows, long K, long ldd_blocks, int T, long x_stride,
                        long y_stride);

/* IQ3_S and IQ4_NL with T activations through one dequantised weight row: the same
 * sharing as qe_gemv_q8_0_multi, for the two formats the routed experts are stored
 * in and for a sharper reason. Those two are not bandwidth-bound: 60 and 44 GB/s
 * against the 76-79 GB/s ceiling this host reaches for a flat read (`make alu`),
 * because a block of iq3_s costs eight 256-entry grid lookups, a sign spread and a
 * compare per 32 weights. A chunk of T tokens multiplies the same bytes T times, so
 * that is T-1 copies of the most expensive arithmetic in the step.
 *
 * The rows arrive as pointers and not as a stride, which is the one difference from
 * the Q8_0 form and the reason it is a separate entry point: the T rows that chose
 * one expert are wherever the routing left them, and gathering them next to each
 * other costs a pass over T x K floats per call to save nothing.
 *
 * Not bit-identical to T single-row calls, for the same reason as the Q8_0 form: the
 * block scale moves onto the weight before the first multiply so one dequantised
 * block can serve all T rows, and there are two accumulator chains per token instead
 * of four. bench_kern holds it to 1e-3 relative against the double reference in
 * src/kern_iq.c, and `make moe` holds the pass that calls it to 1e-4 of the mean
 * magnitude of what it sums.
 *
 * T is the number of rows in xs/ys; every ys[t][r] is written exactly once. */
void qe_gemv_iq4_nl_multi(const void *restrict W, const float *const *restrict xs,
                          float *const *restrict ys, long n_rows, long K, long ldd_blocks,
                          int T);
void qe_gemv_iq3_s_multi(const void *restrict W, const float *const *restrict xs,
                         float *const *restrict ys, long n_rows, long K, long ldd_blocks,
                         int T);

/* The shared-dequant forms behind one dtype argument, for a caller that walks experts
 * and must not care which kernel it is calling: IQ4_NL and IQ3_S today, the two
 * formats this checkpoint reaches a projection with. `ldd` is in blocks. T < 2, a
 * type with no shared form, or an empty matrix returns -1 having written nothing, so
 * the caller can fall back to T single-row calls without a second type table. */
int qe_gemv_multi_ptrs_t(qe_type t, const void *restrict W, const float *const *xs,
                         float *const *ys, int T, long n_rows, long K, long ldd);
/* 1 when the dispatcher above has one, for a caller that has to know before it has
 * written anywhere. Same list, adjacent by construction. */
int qe_gemv_multi_ptrs_has(qe_type t);

/* Dtype-agnostic row of the above. The MoE needs it: gate/up are IQ3_S in
 * some layers and Q8_0 in others, and down is IQ4_NL or Q8_0, so the expert
 * loop must not care which kernel it is calling. `ldd` is in blocks for the
 * quantized types and in elements for f32/bf16. Returns 0, or -1 for a type
 * with no kernel (silently skipping would score the call as a zero logit). */
int qe_gemv_t(qe_type t, const void *restrict W, const float *restrict x, float *restrict y,
              long n_rows, long K, long ldd);

/* The same product for T activations against one weight matrix, with the weight
 * bytes read once for all T instead of T times.
 *
 * Prefill and a batched decode step have exactly this shape: every token needs
 * the whole matrix, and a weight byte can only be paid for once if all T uses
 * of it happen while it is still close. So the loop is blocked over weight rows:
 * take a block of rows that fits in the L2 of a core, run the ordinary
 * single-vector kernel over that block once per token, move on. For most types
 * nothing about the arithmetic changes - each y[t][r] is produced by the same
 * kernel call on the same inputs as a standalone qe_gemv_t, so the results are
 * bit-identical to T separate calls, and the DRAM cost of the matrix becomes
 * row_bytes * n_rows / T per token.
 *
 * Q8_0 with T >= 2 is the exception and the reason the walk is not just a loop:
 * it goes through qe_gemv_q8_0_multi, which shares one dequantised block between
 * the T tokens instead of converting it T times. That reassociates, so a chunked
 * Q8_0 product is not bit-identical to the T decode steps it replaces; bench_kern
 * holds it to 1e-3 relative against a double reference (2.3e-5 observed), and
 * `make prefix` is what checks the chunked trunk against the cold one.
 *
 * x holds T rows of K activations, consecutive tokens x_stride elements apart
 * (0 for K). y holds T rows, consecutive tokens y_stride apart (0 for n_rows);
 * every y[t][r] is written exactly once, so nothing needs pre-clearing.
 * row_bytes is the stride between weight rows in bytes, which is what a caller
 * with a qe_tensor already has: `t->bytes / t->ne[1]`, the arithmetic
 * src/forward.c's tensor_row_bytes and src/moe.c's row_bytes_of do for their own
 * walks (neither is exported, so a new caller writes the division). block_bytes bounds a
 * row block; 0 takes a default of 1 MiB, half of a 2 MiB L2 so the block leaves
 * room for the T activation rows and their output. Returns 0, or -1 for a type
 * with no kernel, in which case nothing is written.
 */
int qe_gemv_multi_t(qe_type t, const void *restrict W, const float *restrict x,
                    float *restrict y, long n_rows, long K,
                    long ldd, int T, long x_stride, long y_stride, size_t row_bytes,
                    size_t block_bytes);
long qe_type_block_ne(qe_type t);

/* ------------------------------------------------ int8 panels and integer walks
 *
 * Every format this checkpoint stores weights in puts a scale in front of every 32
 * weights, and the shared-dequant walks above use that once per 32 weights to turn
 * the weight bytes back into floats: four converts, four multiplies, and then four
 * FMAs per token row against four loads of that token's activations. Recoding the
 * *activations* to int8 instead moves the float work off the token: a 32-weight
 * group against a 32-activation group is one vpdpbusd, whose result leaves i32 for
 * f32 once per 32 weights into a multiply by the two scales and an FMA. Five
 * vector ops per 32 weights per token where the walk above paid eight ops and eight
 * loads, and a token row of the panel is a quarter of the bytes, which at T=8
 * and K=2560 is the difference between a panel that rides in L1 and one that
 * displaces the weights it is multiplied against (ideas.md 7.10, where this was
 * measured one term at a time before it was written).
 *
 * vpdpbusd multiplies unsigned by signed, so one side carries a +128. It goes on the
 * weights (an exact xor of the sign bit) because the correction to the sum is then
 * 128 times the sum of the *activation* block, which every output row of the walk
 * shares and the quantiser computes once, instead of 128 times the sum of the
 * weights, which each output row pays for again.
 *
 * A panel row is therefore three things per 32 activations: the int8 values, their scale
 * `s`, and `p` = -16 times their sum *as an integer*: the 128-times-the-sum correction
 * owed back, in the form vpdpbusd's accumulator wants it - one eighth of it per lane,
 * because the eight lanes' horizontal sum adds the eight shares back into the one block
 * total, and negative because it is a subtraction the seed does for free rather than an
 * op the exit pays. Both arrays are one value per 32 activations. The recode is exact
 * where exactness is cheap: `p` is the sum of the values that were actually stored, and
 * 16 * sum of 32 values each under 128 is under 2^16, so the seed is an exact integer
 * against an exact integer rather than two large f32s that happen to be near each other.
 * A block whose widest value is zero, not a number, or infinite comes back as zeros with
 * a zero scale, which is what the f32 walks do with such a row. */
typedef struct {
    int8_t *x8;    /* token t's values at x8 + (size_t)t * stride8 */
    float *s;      /* token t's block scales at s + (size_t)t * nb */
    int32_t *p;    /* token t's bias seeds at p + (size_t)t * nb */
    long stride8;  /* bytes between token rows of x8, normally K */
    long nb;       /* 32-value blocks in a row, K / 32 */
} qe_i8panel;

/* One row, and T rows `x_stride` apart, which is also how a caller gathers: the source
 * rows need not be the panel's order, only `x_stride` apart. `x8` needs K bytes, `s` and
 * `p` K/32 values each; K rounds down to whole 32-value blocks, as every tensor this is
 * used on does. */
void qe_quant_q8_0_row(const float *restrict x, int8_t *restrict x8, float *restrict s,
                       int32_t *restrict p, long K);
void qe_quant_q8_0_rows(const float *restrict x, long x_stride, int8_t *restrict x8,
                        float *restrict s, int32_t *restrict p, long T, long K);

/* One row again, with one scale for the whole row instead of one per 32 values: `s` is
 * still one float per 32 activations, every one of them equal, which is what makes the
 * panel this writes a valid panel for a walk that reads `s[b]`. What the equal scales buy
 * is that a walk can hoist the product of the two scales out of its token loop, and what
 * they cost is range - one outlier spends itself on the resolution of the row. Only the
 * signed codebook walks read such a panel differently (`qe_i8_rowscale`); every other walk
 * takes it as the per-block panel it already knows how to read. */
void qe_quant_q8_0_row_rs(const float *restrict x, int8_t *restrict x8, float *restrict s,
                          int32_t *restrict p, long K);

/* n_rows output rows of one weight matrix against T rows of a panel, with the panel
 * read once for the token group rather than once per output row and one f32 exit
 * serving every output row of the tile. ys[t] is the i-th token row's output vector, and
 * every ys[t][r] is written exactly once for r in [0, n_rows).
 *
 * The panel is read in its own order - token row t is panel row t - because the walk that
 * is worth having keeps three panel cursors per token row in registers, and indexing them
 * through a `rows` array is what put them on the stack and gave the whole win back. A
 * caller whose tokens are a *set* of rows rather than a run of them quantises them into
 * the panel in the order it wants, which is one pass over T x K floats it is doing
 * anyway, and gets a contiguous panel out of it.
 *
 * Not bit-identical to the walk it replaces, and not trying to be: the activations are
 * a different number, quantised to 7 bits and a scale per 32 values, which is the same
 * recode a Q8_0 key/value cache already makes of a vector this one multiplies exactly.
 * What it costs is measured, and the measurement is what switched it off: `make score
 * --ingest N` prefills corpus windows twice and prices the next token's loss, its greedy
 * choice and the head's gap, because teacher forcing steps one token at a time and a
 * one-token group has no grouped pass to recode - `make score` without `--ingest` cannot
 * see this walk at all. src/moe.c has the numbers; accuracy against a double reference on
 * the shapes bench_kern walks is reported there.
 *
 * Returns 0, or -1 for a type with no integer walk, or for a Q8_0 walk asked for one token row
 * (whose narrowest tile is a pair), having written nothing, so a caller can fall back to the f32
 * walk without a second type table. `qe_gemv_i8_has_1` is how a caller that has to decide before
 * it starts writing asks the one-token question. */
int qe_gemv_i8_t(qe_type t, const void *restrict W, const qe_i8panel *restrict a, int T,
                 float *const *restrict ys, long n_rows, long K, long ldd);
/* 1 when the dispatcher above has an integer walk for this type. */
int qe_gemv_i8_has(qe_type t);
/* 1 when it also has one for a group of a single token row, which is what a decode step's
 * expert walk is: one activation row against every row of the expert it routed to. The codebook
 * walks have a one-token form; the Q8_0 tile does not, because a pair is its narrowest form and
 * a remainder takes the pair's own last row with it. */
int qe_gemv_i8_has_1(qe_type t);

/* Re-coding weights from one on-disk type to another. The pair exists so that a
 * projection stored as Q8_0 can be read as IQ4_NL: the bytes per token are the
 * ceiling on this host (docs/DESIGN.md 1), and the smaller format already has a
 * kernel, so what was missing was the way in.
 *
 * qe_dequant_q8_0 writes floor(n/32) blocks worth of f32 into dst.
 * qe_quant_iq4_nl writes floor(n/32) blocks and returns how many it wrote; the
 * scale of each block is chosen by searching a grid of candidates against the
 * codebook, which is a cheaper rule than the reference quantiser's and is paid for
 * or not by the loss on real text (`make score`), not by a bound written here. Both
 * stop at the last whole block: every tensor this is used on has a row count that
 * is a multiple of 32, and a row is the unit a caller wants.
 */
void qe_dequant_q8_0(const void *w, float *dst, size_t n);
size_t qe_quant_iq4_nl(const float *x, qe_block_iq4_nl *blocks, size_t n);
/* The same search with a Q8_0 block on the input side, so that a copy is made from
 * the bytes a file holds instead of through a float copy of the model. The block's
 * values are rebuilt as `d * q` - the product the walk that reads them reconstructs -
 * which makes the result bit-identical to handing `qe_quant_iq4_nl` the dequantised
 * row, and `make fast` holds the two to that. `nblocks` blocks in, `nblocks` out. */
size_t qe_quant_iq4_nl_q8_0(const qe_block_q8_0 *blocks, qe_block_iq4_nl *out, size_t nblocks);
/* qe_dequant_q6_k writes floor(n/256) blocks worth of f32, rebuilding each value the
 * way the walk that reads it does, which is the same expression `qe_dot_q6_k_ref`
 * accumulates (`make fast` races the two). The two encoders beside it are the same
 * rebuild fed to the codebook and the int8 search, so a copy of a 6-bit matrix made in
 * another format is that matrix's values re-rounded and not a different reading of it. */
void qe_dequant_q6_k(const void *w, float *dst, size_t n);
/* The int8 counterpart of qe_quant_iq4_nl: floor(n/32) blocks, each with the fp16 scale
 * that minimises that block's own squared error over the two scales it could pick. */
size_t qe_quant_q8_0(const float *x, qe_block_q8_0 *blocks, size_t n);
size_t qe_quant_iq4_nl_q6_k(const qe_block_q6_k *blocks, qe_block_iq4_nl *out, size_t nblocks);
size_t qe_quant_q8_0_q6_k(const qe_block_q6_k *blocks, qe_block_q8_0 *out, size_t nblocks);

/* Bandwidth probe over an existing mapping. */
double qe_measure_read_bw(const void *p, size_t n, int nthreads);

/* ------------------------------------------------- layer elementwise primitives
 *
 * The work a step does on activations rather than on weights: the grouped
 * RMSNorms the hyper-connections and the per-head norms need, the sigmoid that
 * gates them, and the two stream mixers. Kept as fused primitives because the
 * widest of them reads two 40 KiB vectors and writes a 10 KiB one, and the
 * unfused version of the same expression costs 2.5x the memory traffic at every
 * one of the 96 call sites per token. Error against libm and against a double
 * reference is what `make fast` reports.
 */

/* out[i] = 1/(1+exp(-x[i])). ~1e-6 worst error against libm over the range the
 * model produces; saturates rather than overflowing at the extremes. */
void qe_sigmoid(const float *x, float *out, long n);

/* n_groups independent RMSNorms of `width` consecutive floats each, scaled
 * element-wise by gamma. gamma_stride is the distance between one group's gamma
 * and the next: 0 means one shared gamma, which is what ssm_norm (128 entries,
 * 48 heads) and attn_q_norm (256 entries, 24 heads) are, while hc_*_norm is
 * 10240 entries for four streams and so passes `width`. The reduction is per
 * group by definition - flattening it is a different model. */
void qe_rms_norm_grouped(const float *x, const float *restrict gamma, long gamma_stride,
                         float *out, long width, int n_groups, float eps);

/* mixed[j] = mean over streams of xn[s][j] * sigmoid(gate[s][j]). */
void qe_stream_gate_mean(const float *restrict xn, const float *restrict gate,
                         float *restrict mixed, long width, int n_streams);

/* res[s][j] += y[j] * 2*sigmoid(inject[s]), the hyper-connection's combine step:
 * one vector broadcast over the streams, each with its own scalar. */
void qe_stream_inject_add(float *restrict res, const float *restrict y,
                          const float *restrict inject, long width, int n_streams);

/* The same block primitives the reference uses, each matching one op it uses:
 * silu and silu(g)*u (the expert MLP), softplus in the form log1p(exp(-|x|)) +
 * max(x,0) that cannot overflow for large positive x, and the GDN q/k norm,
 * which divides by max(||x||_2, eps) - no width in the denominator, epsilon
 * flooring the norm instead of sitting under the root, so it is not RMSNorm and
 * must not share its code path with it. */
void qe_silu(const float *x, float *out, long n);
void qe_silu_mul(float *h, const float *g, const float *u, long n);
void qe_softplus(const float *x, float *out, long n);
void qe_l2_norm_grouped(const float *x, float *out, long width, int n_groups, float eps);
/* The same normalisation with epsilon under the root instead of flooring the
 * norm: x / sqrt(sum(x^2) + eps), which is what the reference's
 * scale(rms_norm(x, eps/n), 1/sqrt(n)) reduces to. The two differ by ~1e-12 at
 * any magnitude this model reaches, so only the shape of the code path
 * distinguishes them. */
void qe_l2_norm_grouped_under(const float *x, float *out, long width, int n_groups, float eps);

/* Partial rotary: n_dims of a head's dimensions rotate, the rest are copied.
 * init() builds the n_dims/2 (cos,sin) pairs for one position, with
 * theta_p = pos * base^(-2p/n_dims) accumulated by repeated multiplication;
 * apply() rotates split-half pairs (p, p + n_dims/2) in place at the start of
 * each of n_heads rows `stride` elements apart. Split-half is what this
 * checkpoint's rope metadata means - see the note in src/layer.c. */
void qe_rope_init(float *cs, long n_dims, long pos, float theta_base);
void qe_rope_apply(float *restrict v, long n_heads, long stride, long n_dims,
                   const float *restrict cs);

float qe_dot(const float *restrict a, const float *restrict b, long n);
void qe_axpy(float *restrict y, const float *restrict x, float a, long n);
/* The same two against a q8_0 row: one absolute-maximum scale for the whole row
 * and one byte per value. A quantized key/value cache is read about two thousand
 * rows per query per full-attention layer and written once per token, so what a
 * row costs in bytes is what the cache is made of, and these two are where those
 * bytes turn back into numbers. `qe_quant_q8_row` is the write side. A row whose
 * widest value is zero, not a number, or infinite comes back as zeros: the
 * non-finite counters are upstream of this, where the value still exists. */
float qe_dot_q8(const float *restrict a, const int8_t *restrict q, long n);
void qe_axpy_q8(float *restrict y, const int8_t *restrict q, float a, long n);
void qe_quant_q8_row(const float *src, int8_t *dst, float *scale, long n);

/* One KV group's walk over the selected cells, softmax carried inside the walk.
 *
 * A query head's pass over the selection is one key row and one value row per
 * cell, and `n_q_heads` query heads share a KV head - twelve ways at this
 * checkpoint - so walking the selection once per query head reads every row of it
 * twelve times to produce twelve answers that were available from one pass. This
 * is the one-pass form: the cells of one chunk, the `per` query heads of one KV
 * group inside the cell loop, and the exponent base and sum of weights updated as
 * each score is produced, so no score ever reaches memory. That is what the
 * flash-attention shape means on a machine with no tensor cores - not different
 * arithmetic, but never letting a row that was just read go cold before every head
 * that wants it has had it - and it removes both the twelvefold gather and the
 * score array the head-at-a-time form needs.
 *
 * One pass over the selection is only `nkvh` units of work and a decode step has
 * twenty cores to fill, so the selection is chunked: call once per chunk with
 * `sel` narrowed to that chunk, then combine the partials with qe_attn_combine.
 * A chunk's accumulator is unnormalised and its exponent base is its own, which is
 * what lets the combine happen in any order and on any thread.
 *
 * `sel` belongs to the caller and ascending is what makes the walk a stride the
 * hardware prefetcher can follow instead of a random touch of 512-byte rows.
 * Accumulation therefore follows the address order rather than the order the
 * indexer's heap produced, which is a reassociation of an exp-weighted sum: it
 * agrees with the head-at-a-time form to ~1e-6 relative, not bit for bit. */
typedef struct {
    const float *q;        /* `per` query heads of one KV group, head_dim apart */
    const int8_t *kd, *vd; /* the q8_0 cache, one byte per value; NULL with the f32 cache */
    const float *kc, *vc;  /* the f32 cache */
    const float *ks, *vs;  /* one scale per KV head of one cell: ks[cell*sstride + kv_head] */
    const int *sel;        /* the cells this chunk walks */
    float *y;              /* out: per*head_dim accumulator, unnormalised */
    float *m, *l;          /* out: per exponent base and sum of weights */
    int nsel;              /* cells in sel */
    int per;               /* query heads sharing this KV head */
    int hd;                /* head_dim */
    int krow;              /* bytes (q8) or elements (f32) from one cell's row to the next */
    int kv_head;           /* which head of the row this group reads */
    int sstride;           /* n_kv_heads: the scale row's stride */
    int q8;
    float scale;           /* 1/sqrt(head_dim) */
} qe_attn_unit;

void qe_attn_walk(const qe_attn_unit *u);

/* Fold a head's chunk partials into its output: out = sum(y_i * exp(m_i - M)) /
 * sum(l_i * exp(m_i - M)) with M the highest of the partials' exponent bases.
 * Partials are `nparts` accumulator rows `ystride` floats apart, with their bases
 * and sums `pstride` floats apart, so one call can take the chunks of one head out
 * of a units-by-heads accumulator. A head whose every partial saw no cells comes
 * back as zeros. */
void qe_attn_combine(const float *py, long ystride, const float *pm, const float *pl,
                     long pstride, int nparts, int hd, float *out);

/* ----------------------------------------------------------------- tokenizer
 *
 * Byte-level BPE built from the checkpoint's own tokenizer.* metadata: the
 * vocabulary, the merges with their ranks, the token_type array that marks
 * control tokens, and tokenizer.ggml.pre for the pre-tokenizer. Encoding is
 * lossless for any input, so decode(encode(s)) == s always holds. */

typedef struct qe_tok qe_tok;

/* Returns NULL with a reason in err when the file's tokenizer is missing or is
 * a scheme this build does not implement. */
qe_tok *qe_tok_open(qe_model *m, char *err, size_t errn);
void qe_tok_close(qe_tok *t);

long qe_tok_n_vocab(const qe_tok *t);
int qe_tok_eos(const qe_tok *t);
int qe_tok_bos(const qe_tok *t);
int qe_tok_is_special(const qe_tok *t, int id);

/* Token ids of text. Returns how many were produced, or -1 if text is NULL, the
 * ids array is too small (nothing partial is left behind) or allocation failed.
 * max_ids may be 0 with ids == NULL to just count. */
long qe_tok_encode(const qe_tok *t, const char *text, int *ids, long max_ids);
/* How many ids encode() would produce, without storing them. */
long qe_tok_count(const qe_tok *t, const char *text);

/* Text for ids. Like snprintf: returns the length that would have been written
 * and always NUL-terminates when cap > 0. Ids outside the vocabulary are
 * skipped rather than guessed at. */
long qe_tok_decode(const qe_tok *t, const int *ids, long n, char *out, size_t cap);

/* One-line summary of what was loaded, for --report and benches. */
void qe_tok_describe(const qe_tok *t, char *buf, size_t n);

/* --------------------------------------------------------------------- MoE
 *
 * 512 routed experts, 10 active per token. The pipeline exists for two
 * reasons, and neither is the byte count: reading 10/512 of the pool is
 * already the minimum, so the win available here is latency and locality.
 *
 *   - two phases instead of ten per-expert serial passes, so all 20 cores
 *     stream 10 experts at once instead of two cores streaming one
 *   - slicing the work by (expert, row) in flattened order, which keeps every
 *     thread inside at most two experts and therefore inside a handful of
 *     DRAM pages rather than ten
 *
 * Software prefetch is not on that list because it lost: bench_moe measured it
 * as a 6-67% regression against issuing none, and warming the selected experts
 * ahead of the pass was noise at best.
 *
 * One qe_moe drives one inference at a time: the scratch is shared, and the
 * decode loop is bandwidth-bound anyway, so a mutex in front of it costs
 * nothing. See qe_serve for how requests are serialised.
 */

typedef struct {
    int32_t expert; /* index in [0, n_experts) */
    float   weight; /* routing weight after renormalisation */
    float   logit;  /* raw router logit, for metrics */
} qe_route;

typedef struct {
    /* Flat (experts, N, K) weights: expert e's matrix starts at
     * base + e * stride. No gather, no repack, no copy. */
    const void *gate, *up, *down;
    long gate_stride, up_stride, down_stride; /* bytes between experts */
    qe_type t_gate, t_up, t_down;
    const float *router; /* f32 (n_experts, K) */
    /* Optional bfloat16 shadow of the same matrix, built by the caller with
     * qe_bf16_from_f32(). The router is read in full for every token, so it
     * costs 240 MiB of the 5.44 GiB step at f32 and 120 MiB here. Measured, not
     * guessed: over 96000 real layer-token pairs of the real routers, the bf16
     * shadow selected the same 10 of 512 in every single one (0 differing sets,
     * 0 differing slots) - see the "Router width" block of `make route`.
     * Selection is therefore allowed to read the shadow; the mixing weights are
     * not, which is what qe_moe_route does with both pointers set. */
    const void *router_bf16;
    long n_experts;      /* 512 */
    long N;              /* expert width (640) */
    long K;              /* hidden size (2560) */
    int  top_k;          /* 10 */
    int  n_threads;      /* 0 = the pool's size, which qe_moe_create's own thread
                          * table argues for; --threads N caps it */
    /* Non-zero keeps the grouped multi-row pass on T single-row kernel calls even for a
     * type that has a shared-dequant kernel (qe_gemv_multi_ptrs_t). Grouping shares an
     * expert's *bytes* either way; this says whether it also shares the dequantisation
     * of them, which is a different order of summation (include/qe.h,
     * qe_gemv_iq3_s_multi) and therefore something a measurement has to have decided.
     * Zero shares it. */
    int  no_shared_multi;
} qe_moe_desc;

typedef struct qe_moe qe_moe;

qe_moe *qe_moe_create(const qe_moe_desc *d); /* NULL + errno on failure */
void qe_moe_destroy(qe_moe *m);

/* Router logits for one token: dense (n_experts, K) . x, row-split across the
 * pool, from the bf16 shadow when the descriptor carries one and from f32
 * otherwise. At 512 x 2560 the row is read whole for every token, so this is
 * 240 MiB per token at f32 and 120 MiB at bf16 - the largest single saving
 * available on the dense side, and the reason the two paths are kept side by
 * side rather than one replacing the other. */
void qe_moe_router(const qe_moe *m, const float *x, float *logits);

/* T tokens' logits in one pass over the router. Token t's activations are
 * x + t*x_stride and its logits are logits + t*logits_stride, and the result is
 * byte for byte qe_moe_router called once per token: the walk is row-split, and
 * every kernel it uses finishes one output row from one row of weights, so
 * sharing the pass moves the arithmetic, it does not change it. What it saves is
 * the repeated read of the router, which is the whole row for every token, and
 * T-1 dispatches per layer. */
void qe_moe_router_multi(const qe_moe *m, const float *x, long x_stride, float *logits,
                         long logits_stride, int T);

/* Round-to-nearest-even f32 -> bf16, for the router shadow. */
void qe_bf16_from_f32(const float *src, void *dst, size_t n);

/* Top-k by logit with softmax renormalisation (weights sum to 1). Returns the
 * number selected, which is desc.top_k unless fewer experts exist. */
int qe_moe_select(const qe_moe *m, const float *logits, qe_route *out);

/* qe_moe_router + qe_moe_select in one call, and when the descriptor carries
 * both router widths, the k winning rows are re-read at f32 so the renormalised
 * weights are exactly the f32 router's while only the picks came from the
 * shadow. `logits` needs room for n_experts floats and belongs to the caller,
 * because a forward pass already owns one per layer in flight. */
int qe_moe_route(const qe_moe *m, const float *x, float *logits, qe_route *out);

/* The refinement qe_moe_route applies, on its own, for a caller that did the
 * router and the top-k itself: re-read the k winning rows at f32 and renormalise
 * `routes` from them, so mixing weights never inherit the shadow's mantissa.
 * Routes keep the order qe_moe_select left them in. Does nothing when the
 * descriptor carries one router width. */
void qe_moe_refine(const qe_moe *m, const float *x, qe_route *routes, int k);

/* Multi-row form of the same thing: T rows of activations with stride x_stride
 * and T x k routes, refined across the pool in one dispatch. Same per-row
 * arithmetic, so a chunk and a token fed alone cannot disagree. */
void qe_moe_refine_rows(const qe_moe *m, const float *x, long x_stride, qe_route *routes,
                        int k, int T);

/* y[K] = sum over selected experts of weight * down(silu(gate.x) * up.x).
 * routes must come from qe_moe_select against the same logits. */
void qe_moe_forward(qe_moe *m, const float *x, const qe_route *routes, int n_routes, float *y);

/* Grouped multi-row form of the same pass: x holds T activations with stride
 * x_stride, y holds T outputs with stride y_stride, and routes holds T x
 * routes_per_row entries (row t, choice i at routes[t*routes_per_row + i]).
 *
 * The point is bytes, not flops. Grouped by expert, one read of an expert
 * serves every row that chose it, so the step reads the *distinct* experts of
 * the window instead of T*top_k of them, and the routed-expert share of the
 * per-token budget falls by the overlap factor. This is the technique Strata
 * documents as "the rows of each expert are read once for all of its tokens",
 * and it is what makes an MTP verify window or several concurrent
 * conversations cheaper per token than a single stream.
 *
 * y is written for every row. Because the down outputs are summed per thread
 * over expert buckets rather than over the serial route list, the order of the
 * floating-point additions differs from qe_moe_forward: results agree to
 * ~1e-6 relative, not bit for bit. Where the expert type has a shared-dequant
 * kernel the pass also takes it, so one dequantisation of a row serves every row
 * of the bucket that chose that expert; that moves the block scale onto the weight
 * before the first multiply, which is a second reassociation of the same size and
 * the reason desc.no_shared_multi exists. T == 1 defers to qe_moe_forward. */
void qe_moe_forward_multi(qe_moe *m, const float *x, long x_stride, const qe_route *routes,
                          int routes_per_row, int T, float *y, long y_stride);

/* The same grouped pass as three items instead of three dispatches, for a caller
 * that would rather run them in a graph it owns (`qe_pool_run_items`) than one
 * thread at a time. `qe_moe_multi_begin` is everything a pass decides once - that
 * the grouped form applies at all, how its rows sorted into buckets, what the
 * routing statistics say about it - and it leaves the pass standing in the moe
 * object, so one grouped pass per moe object may be in a graph at a time, which is
 * what a layer has.
 *
 * Returns 1 when the three items are ready; 0 when the grouped form does not apply,
 * which is the caller's cue to do what qe_moe_forward_multi does in that case (one
 * pass per row); and 2 when the pass is already over, which is how a route list
 * with nothing in it is answered with zeroes - y is written, and no item may run.
 *
 * Item 0 walks the gate and up rows of every bucket, item 1 its down rows, item 2
 * sums the per-thread partials into y. Only item 0 has no predecessor: a caller
 * copying these into a graph sets 1 to depend on 0 and 2 on 1. All three want the
 * same `nthreads` - the partials of item 1 are counted by thread, and item 2 sums
 * them with however many it ran under - so a graph that gave item 1 fewer threads
 * than it declares would be reading partials nobody wrote. */
int qe_moe_multi_begin(qe_moe *m, const float *x, long x_stride, const qe_route *routes,
                       int routes_per_row, int T, float *y, long y_stride);
void qe_moe_multi_item(qe_moe *m, int which, qe_pool_item *out);

/* Whether the grouped expert pass recodes the vector it multiplies with to int8 and walks
 * the experts with the integer forms above. It ships on, and QE_I8_MOE=0 is the way back:
 * the walk is worth 1.10x of a step, and ideas.md 7.13 measured the expert stage's whole
 * remaining headroom - walked with nothing expert about it - at 1.2-1.6x, so what this is
 * traded against is no kernel and not a safer one. What it charges is on the same record
 * rather than negotiable: 1.10 nats mean and 4.69 worst on the loss of the token that
 * follows a window, greedy choice moved once in five, `bench_score --ingest 5`; src/moe.c
 * has the numbers. A setter because the honest A/B of the two walks is one load of a
 * checkpoint with the switch thrown between passes, not two loads. Negative asks the
 * environment again, which is the way back a bench that toggles it between legs wants. */
void qe_moe_set_int8(int v);

/* What the engine will actually do, resolved against the environment. A bench that marks
 * one leg "(default)" has to ask this instead of reading QE_I8_MOE itself: the default and
 * the environment stopped being the same question when the walk shipped on. */
int qe_moe_int8(void);

/* How the two grouped tasks cut the expert stage. Level 0 is an equal share of the
 * (bucket, output row) *count*. Level 1 is an equal share of what the stage *costs*: one
 * walk of an expert's weight rows costs about the same for one token row as for a chunk of
 * them and about one more chunk per chunk past that, so a bucket of forty is several
 * buckets of two while counting as exactly one unit, and a thread that drew the fat ones
 * leaves the rest waiting. Level 2 charges each thread's share by its core class as well
 * (the kernel's cpu_capacity, 1011 against 783 on this host), which is the right
 * denominator for this one dispatch and the wrong one for every byte-bound stage in the
 * engine - src/mem.c has the record of applying it engine-wide.
 *
 * Both level 1 and level 2 are a function of the bucket histogram and the machine, never
 * of which thread got somewhere first, so a chunk splits the same way every time it is
 * run. QE_MOE_BAL_DEFAULT is what ships, QE_MOE_BAL=0 is the way back, and `make moe`
 * A/Bs all three against the same routing draws at four router skews. */
#define QE_MOE_BAL_DEFAULT 1
void qe_moe_set_balance(int v);
int qe_moe_balance(void);

/* A thread's share of a stage whose cost is arithmetic rather than bytes, relative to a
 * mean of 1 for the threads the caller names: 1.15 against 0.89 on this host's eight
 * performance and twelve efficient cores. Zero means the kernel publishes no such score, or
 * the caller named more threads than the pool has, and a caller that gets 0 divides the work
 * as it would have without asking. See src/mem.c for why this is asked by one stage and not
 * applied by qe_slice. */
double qe_cpu_weight(int tid, int nthreads);

/* --------------------------------------------------------------- PLE gather */

/* The n-gram per-layer embedding of docs/ARCH.md 6.2. Sixteen rows of 160
 * IQ4_NL elements - 90 bytes, five blocks - are the layer-1 embedding input:
 * heads 0..7 hash the bigram, heads 8..15 the trigram, each row indexed by
 * mixed % head_vocab_sizes[head] + head_offsets[head].
 *
 * Those 1440 bytes are nothing in the 5.44 GiB a token reads, and they are the
 * only part of a token that is pure latency: sixteen lines in sixteen distant
 * pages of a 26.8 GiB table, so sixteen page walks and sixteen cold fills on the
 * critical path. Hence the three things this does: sort and dedupe the sixteen
 * ids before touching memory, let the caller issue them at token-id time and
 * collect before layer 1, and cache rows across tokens. */
typedef struct qe_ple qe_ple;

typedef struct {
    const char *tensor;    /* default "per_layer_token_embd.weight" */
    long long cache_rows;  /* rows kept between tokens, 0 = 1<<20 */
} qe_ple_desc;

typedef struct {
    unsigned long long rows;       /* row slots produced */
    unsigned long long distinct;   /* rows actually fetched, after dedup */
    unsigned long long cache_hits; /* rows served from the row cache */
} qe_ple_stats;

qe_ple *qe_ple_create(qe_model *m, const qe_ple_desc *d, char *err, size_t errlen);
void qe_ple_destroy(qe_ple *p);

int qe_ple_heads(const qe_ple *p);         /* 16 */
int qe_ple_head_dim(const qe_ple *p);      /* 160 */
int qe_ple_width(const qe_ple *p);         /* 2560 = heads * head_dim */
size_t qe_ple_row_bytes(const qe_ple *p);  /* 90 */
long long qe_ple_table_rows(const qe_ple *p);
/* The id that erases the window, from the file's own ple.eos_token_id. It is
 * not the tokenizer's end-of-sequence id: this checkpoint's metadata says
 * 248044 where the tokenizer's eos is 248046, so a caller that substitutes the
 * one for the other fills the window with a token the PLE hashes as ordinary. */
uint32_t qe_ple_eos_token(const qe_ple *p);

/* Row ids for the n-gram window ending at seq[i], following the rule the file
 * states: a predecessor EOS replaces itself and everything older, a missing
 * predecessor reads as EOS, and the token's own EOS does not cut its window.
 * Image positions must already hold ple.image_token_id in seq. */
void qe_ple_rows_at(const qe_ple *p, const uint32_t *seq, size_t i, long long *rows);

/* Fetch and dequantize rows into emb[heads * head_dim]. */
int qe_ple_gather(qe_ple *p, const long long *rows, float *emb);

/* The same work split so it can straddle the layers between choosing a token id
 * and needing layer 1: issue() resolves the ids and starts the loads, collect()
 * decodes. Pending tokens are a FIFO of depth 8; issue() returns -1 when it is
 * full, which means the caller is further ahead than the pipeline can cover. */
int qe_ple_issue(qe_ple *p, const uint32_t *seq, size_t i);
int qe_ple_collect(qe_ple *p, float *emb);
/* Forget everything still queued. A pending entry is one token's row ids, and a
 * queue that outlives the sequence it was issued for would answer the next
 * collect with the wrong token's rows - a finite wrong answer, which is the
 * expensive kind. Call it whenever the sequence is rewound or started over; an
 * ordinary chunk needs it never, because it issues and collects the same count. */
void qe_ple_drain(qe_ple *p);

void qe_ple_get_stats(const qe_ple *p, qe_ple_stats *out);
void qe_ple_stats_clear(qe_ple *p);

/* Router + select + forward in one call; routes/n_routes may be NULL. */
void qe_moe_layer(qe_moe *m, const float *x, float *y, qe_route *routes, int *n_routes);

/* Selection histogram, for load-balance reporting. Pointer is owned by the
 * qe_moe and valid until it is destroyed. */
const uint64_t *qe_moe_expert_hist(const qe_moe *m, long *n_experts);
void qe_moe_hist_reset(qe_moe *m);

/* ------------------------------------------------------------ speculation */

/* A bandwidth-bound step costs the same for one token as for three, so the
 * MTP head is the only lever that moves the ceiling (docs/DESIGN.md 4). What
 * follows is the control loop around it: draft k tokens, verify all of them in
 * a single target pass, keep the longest agreeing prefix, and always take the
 * target's own token at the first disagreement. That last rule is what makes a
 * draft never change the *rule*: on the model's own arithmetic the emitted sequence
 * is exactly what plain greedy decoding produces, which is what bench_mtp checks
 * against a toy model and what `make spec` checks against a checkpoint started with
 * every ingestion fast path switched off. On the shipping settings it is not byte
 * equality that holds but fidelity: a row of a window is computed by the chunked
 * walk of the projections and a single token by the `T == 1` walk, which is the same
 * trade ingestion already makes and, over a greedy argmax, enough to part the two
 * trajectories at a near-tie.
 *
 * The engine is the caller's problem: qe_spec_model is the smallest interface
 * that a real transformer and a test can both satisfy. (The checkpoint handle
 * is qe_model, from the gguf section above - two different things.)
 */
typedef struct qe_spec_model {
    void *ctx;
    long long vocab;
    /* One target pass over the `n` tokens starting at absolute position `pos`.
     * Writes the next-token logits of every position into `logits`, which is
     * n * vocab floats. Return 0 on success, non-zero to abort the request. */
    int (*verify)(void *ctx, long long pos, const int *toks, int n, float *logits);
    /* The same pass read one row at a time, and the reason `verify` is a function
     * pointer rather than a requirement: a target whose state can be put back
     * (src/forward.c's window) has no reason to walk the rows past the one that
     * disagreed, and every row it walks of a window it has already run costs a
     * walk of the output matrix. When this is set `verify` is not called, `logits`
     * is not written, and the loop calls `rewind` with the rows it read. Row `i`
     * means the same thing in both spellings, and a target that answers both is
     * answered identically - bench_mtp checks that it is. */
    int (*verify_row)(void *ctx, long long pos, const int *toks, int n, int i, float *logits);
    /* Called once a step when `verify_row` is set, with the number of rows the
     * step read: the sequence goes back to standing after `used` of them, which
     * is the position the next step's window starts at. */
    void (*rewind)(void *ctx, long long pos, int used);
    /* Draft up to k tokens following toks[0..n). Writes ids to out and, when
     * probs is not NULL, the probability of each drafted id under the draft
     * distribution (only that scalar is needed for the accept test). Returns
     * how many were drafted; 0 means "no speculation this step". */
    int (*draft)(void *ctx, long long pos, const int *toks, int n, int *out, float *probs, int k);
    /* Called once per emitted token, in order. Return non-zero to stop. */
    int (*emit)(void *ctx, int tok, int is_eos);
} qe_spec_model;

typedef struct {
    int k;          /* draft length; 0 disables speculation entirely */
    /* Only the temperature is honoured here. The loop draws from the target's own
     * softmax (Gumbel-max, so the draft/target rejection test stays exact), and
     * top-k, the nucleus and min-p are properties of qe_sampler_opts on the plain
     * path in src/engine.c, which is the one a checkpoint without MTP weights
     * takes. The two paths therefore agree exactly at temp 0 - bench_mtp asserts
     * that identity - and a checkpoint that grows an MTP head grows this with it. */
    float temp;     /* 0 = greedy */
    unsigned seed;
    long max_tokens;
    int eos;        /* token id that ends generation, or -1 */
} qe_spec_opts;

typedef struct {
    long steps;    /* target passes */
    long tokens;   /* tokens emitted */
    long drafted;
    long accepted;
    double tok_per_step;
    double accept_rate;
} qe_spec_stats;

/* Decodes until eos, max_tokens, or a non-zero return from emit. Returns the
 * number of tokens emitted, or -1 if the model reported failure. `stats`, when
 * given, is zeroed before any of that, so a call that fails still leaves a
 * caller's counters readable rather than uninitialised. */
long qe_spec_generate(const qe_spec_model *m, const qe_spec_opts *o, const int *prompt, int np,
                      qe_spec_stats *stats);

/* ---------------------------------------------------------- n-gram drafter
 *
 * A drafter that reads no weights: an index of the n-grams already in the
 * sequence, and a proposal taken from wherever the longest tail of it last
 * ended. It exists because the target pass is bandwidth-bound, so the cost of
 * a step barely depends on how many tokens it verifies, and because the text a
 * coding agent produces is the most repetitive text this engine has ever been
 * shown - the same file comes back every turn, and the commands it runs repeat
 * within a session.
 *
 * Nothing here changes an answer. Every token it proposes goes through the
 * target's own verification (src/mtp.c), where the first disagreement emits the
 * target's token instead, so at temp 0 the output is the same sequence plain
 * greedy decoding would have produced and the drafter can only be slow. That is
 * also why it is not offered for sampled requests: a proposal has to carry its
 * own probabilities for the rejection test to keep the target's distribution,
 * and a count over a hash table is not a distribution.
 *
 * `qe_draft_suggest` is the shape `qe_spec_model.draft` wants and keeps itself
 * in step with whatever sequence it is handed, so a caller has no index to
 * maintain beside its own tokens. A tail shorter than three tokens is not
 * proposed at all (`QE_DRAFT_MIN` moves the line): the shortest tails are the ones
 * a chat template supplies twice, what they predict is the middle of an earlier
 * turn, and finding that out costs the target a row of a window.
 */
typedef struct qe_draft qe_draft;

typedef struct {
    long lookups;   /* steps a draft was asked for */
    long matched;   /* of those, steps that found the tail somewhere earlier */
    long drafted;   /* tokens proposed */
    double match_len; /* sum of matched tail lengths, for a mean */
    int longest;    /* longest tail ever matched */
} qe_draft_stats;

/* The tail lengths to index. 0 for either is the default: 16 down to 2, which is
 * wide enough for a repeated command line and narrow enough to match a JSON list. */
qe_draft *qe_draft_open(int n_max, int n_min);
void qe_draft_close(qe_draft *d);
/* Forget everything. The index is a function of the sequence, so this is also the
 * only way to say "the next tokens are a different conversation". */
void qe_draft_reset(qe_draft *d);
int qe_draft_push(qe_draft *d, int id);
long qe_draft_feed(qe_draft *d, const int *ids, long n);
/* Propose up to k tokens following toks[0..n). Returns how many, 0 for "no
 * proposal", which is the common answer for a first token of something new. */
int qe_draft_suggest(qe_draft *d, const int *toks, long n, int *out, int k);
void qe_draft_get_stats(const qe_draft *d, qe_draft_stats *s);
long qe_draft_length(const qe_draft *d);

/* ------------------------------------------------------------------ forward
 *
 * docs/ARCH.md section 6 in C: embeddings in, logits out, one sequence, one
 * token per call. A qe_forward holds the tensor lookups, the state a running
 * sequence owns (gated-delta-net state, the two conv histories, the KV cache,
 * the indexer's raw and pooled keys) and every activation vector a step
 * touches; the model itself belongs to the caller and stays resident.
 *
 * The API is deliberately small because the state is not. A step is only
 * meaningful as the next token of the sequence the previous steps were the
 * first tokens of; reset() is what starts a new sequence, and one qe_forward
 * drives one sequence at a time, for the same reason a qe_moe does.
 */

typedef struct qe_forward qe_forward;

/* Places where the reference and the diagram disagree, or where the reference
 * itself is ambiguous and only held-out loss can settle it. Each bit selects the
 * alternative; a caller that leaves the word zero gets the committed reading,
 * which qe_forward_open spells out rather than leaving implicit.
 *
 * They exist because a wrong one is indistinguishable at runtime: no NaN, no
 * crash, full speed, and a model that prints noise. `make score` sweeps them
 * against teacher-forced loss in one load, which is the only cheap way to tell
 * six hypotheses apart when a load costs four minutes.
 *
 * Measured on this checkpoint, teacher-forced over 32 tokens of text the reference
 * engine produced itself, against the reference's own per-token logprobs
 * (`tools/llama_logprobs.py`, `tools/cmp_oracle.py`): the committed reading scores
 * 3.5607 nats/token, matches the reference's argmax on 13 of 16 positions, and on
 * the seven positions the reference itself is confident about it is 0.442 nats
 * where the reference is 0.452, with the same argmax on all seven. It prints
 * fluent English. That is the state that settled GVA_MOD and the delta read-out
 * scale (`QE_VARIANT_GDN_NOSCALE` is worth 4.25 nats and half the top-1s).
 *
 * The other two bits are settled too, but only by the wide corpus: over 32 tokens
 * of prose `QE_VARIANT_PLE_PRENORM` looks 0.17 nats *better* than the committed
 * reading and skipping the PLE 0.04 better, and over 193 positions where the
 * reference is confident both are worse (0.333 and 0.367 against 0.292, with 184
 * and 181 of 193 top-1s against 189). `QE_VARIANT_L2NORM_ROOT` is indistinguishable
 * either way (0.292 both), so the reference's floored form is what ships. Nothing
 * is open now except the four confident positions out of 193 where the two engines
 * disagree, all of them positions the reference is barely sure of either. */
#define QE_VARIANT_GDN_GVA_MOD 1u /* value head h reads key head h % nk (ggml_repeat's
                                   * reading) instead of h / kdiv (einops "(h g)") */
#define QE_VARIANT_L2NORM_ROOT 2u /* x / sqrt(||x||^2 + eps) instead of ggml_l2_norm's
                                   * x / max(||x||, eps) */
#define QE_VARIANT_PLE_PRENORM 4u  /* the PLE conv reads the gated value before its norm */
#define QE_VARIANT_NO_PLE      8u  /* skip the PLE block entirely */

/* Not readings of the architecture: these switch a whole stage off, which is how
 * you find out whether a stage contributes anything at all. A pass that is wrong
 * in one block is not distinguishable at the loss from one that is wrong in all
 * of them - both are nonsense - but they are very distinguishable by the loss
 * being *identical* when one of them is removed. Diagnostic only: nothing
 * production passes these, and `--variants 0x1,0x11,0x21,0x41` is what they are
 * for. */
#define QE_VARIANT_ABL_ATTN 0x10u /* the twelve QSA layers contribute nothing */
#define QE_VARIANT_ABL_GDN  0x20u /* the gated-delta-net layers contribute nothing */
#define QE_VARIANT_ABL_MOE 0x40u /* the routed and shared experts contribute nothing */

/* ggml_compute_forward_gated_delta_net_one_chunk ends the recurrence with
 * attn_data[j] = sum * (1/sqrtf(S_v)) - the delta read-out carries a head-width
 * scale. It cancels through the per-head RMS norm that follows the gated read-out
 * except through the epsilon under that norm's root, so it is not a no-op, and it
 * is small enough (1/11.3 at S_v = 128) that it has to be A/B'd rather than
 * assumed. This bit restores the pre-fix scale of 1. */
#define QE_VARIANT_GDN_NOSCALE 0x100u

#define QE_VARIANT_DEFAULT     QE_VARIANT_GDN_GVA_MOD

typedef struct {
    int n_ctx;     /* cells one sequence may grow to (KV and indexer caches) */
    int n_threads; /* 0 = the pool's size */
    unsigned variant; /* QE_VARIANT_* bits, 0 = the committed reading */
    int profile;      /* nonzero: accumulate per-stage wall time into the diag */
    int kv_f32;       /* nonzero: keep the key/value cache in f32 instead of q8_0 */
    int attn_head_walk; /* nonzero: walk the selection once per query head, which is
                         * the form the one-pass-per-KV-group walk replaced. The
                         * default is the fused walk, so the field asks for the
                         * other one rather than for itself. */
    /* A draft head that arrived as its own file: one more decoder block, the nextn
     * fold in front of it, and the mixer and output rows behind it. When it is set,
     * the pass opens that block as one more layer - resident like every other weight
     * byte, its own place in the key/value and indexer caches, and never run by the
     * trunk's loop - and a caller can walk a row of it with qe_forward_mtp_row.
     * NULL, the default, is a pass with no head: the same code, the same bytes, and
     * nothing extra resident. */
    qe_model *mtp_model;
} qe_forward_opts;

/* The stages a step is made of, timed separately when profile is on. Guessing
 * where a decode step goes costs more than measuring it: the byte budget says
 * the dense projections are half the token, and that is the one part already
 * running at the measured kernel rate.
 *
 * The attention/linear-attention block is timed as two stages and not one,
 * because chunked prefill makes them two different kinds of cost. `block_proj`
 * is the projections and the read-out: one walk of the weights for the whole
 * chunk, so a chunk of eight pays it an eighth as often as a decode does.
 * `block_core` is the recurrence - conv ring, delta rule, cache writes - and it
 * runs once per token no matter how the tokens are grouped. One bucket for both
 * is what let "the block stage is 47.8 ms" stand for a prefill number too: only
 * the part that amortizes got cheaper, and the split is what says by how much. */
enum qe_forward_stage {
    QE_ST_EMBED, QE_ST_BLOCK_PROJ, QE_ST_BLOCK_CORE, QE_ST_HC_ATTN, QE_ST_HC_FFN,
    QE_ST_MOE, QE_ST_PLE, QE_ST_HEAD, QE_ST_LMHEAD, QE_ST_N
};

/* What a step says about itself. This exists because the common failure of a
 * forward pass is not an error: a wrong reduction axis or a rotary pair wired
 * the other way round runs at full speed, never produces a NaN, and is
 * nonsense. Non-finite counts and per-stage magnitude bands are the cheap half
 * of noticing that; the expensive half is perplexity, and these are what
 * `make forward` asserts on. */
typedef struct {
    long steps;
    long nonfinite; /* values that were NaN or +-inf, anywhere in a step */
    long over_ctx;  /* steps refused for want of cache */
    float max_abs_res;    /* widest hyper-connection stream value */
    float max_abs_block;  /* a token-mixer output before it is injected */
    float max_abs_logits; /* a logit */
    double step_seconds;  /* the most recent step, compute only; for a chunk, the
                           * whole chunk's, since the chunk is what was run */
    double stage_seconds[QE_ST_N]; /* accumulated over every step, if profile */
    double stage_macs[QE_ST_N];    /* multiply-accumulates those stages' walks
                                    * actually did, same condition. Divided by the
                                    * seconds above it is the rate a walk ran at,
                                    * counted from inside the stage rather than from
                                    * a tensor table someone hoped matched it. */
} qe_forward_diag;

/* Reads the checkpoint's shape and every tensor the pass needs, and refuses
 * with a sentence naming what is missing or the wrong type. Nothing is lazy:
 * a tensor that is absent is an error here, not a zero later. */
qe_forward *qe_forward_open(qe_model *m, const qe_forward_opts *o, char *err, size_t errn);
void qe_forward_close(qe_forward *f);
/* New sequence: state cleared, position 0. */
void qe_forward_reset(qe_forward *f);
/* Feeds `token` and writes the vocabulary-wide logits predicting the next one
 * into logits_out (room for qe_forward_vocab floats). 0 on success, negative
 * when the sequence is full or the step produced a non-finite value. */
int qe_forward_step(qe_forward *f, int token, float *logits_out);
/* Feeds a prompt: n tokens, with the head and the probe heads run only for the
 * last one, whose logits go to logits_out (NULL to keep the state and the answer
 * to yourself). Everything a step updates - the recurrent state, the caches, the
 * position, the diagnostics - is updated for every token exactly as
 * qe_forward_step would.
 *
 * What that costs in bits is one threshold and nothing else. With the projection
 * walk off (QE_PROJ_MB=0, or --proj-chunk -1 at engine start) the trunk a prompt
 * ends in is bit-identical to feeding it one token at a time, and `make forward`
 * asserts it at every boundary a chunk could end at. With the walk on - the
 * default, because it feeds a prompt 1.20x faster - a Q8_0 projection is summed in
 * a different order, and `make forward` asserts what a client can still check: the
 * same next token, and a worst-case logit move inside a tenth of the head's range.
 *
 * Either way it is not one step each: the tokens go through eight at a time, one
 * stage of a layer at a time, so a stage is entered once per chunk instead of once
 * per token, and the T calls one tensor needs run back to back, which is what keeps
 * its bytes from being read T times. Width on its own is scheduling and not
 * arithmetic - with the walk off, every width gives the same bits, which `make
 * forward` asserts - and QE_CHUNK overrides it, with QE_CHUNK=1 the
 * one-token-at-a-time walk, for A/Bing the two.
 * And what is not computed at all is the 497 MiB lm_head per token that has no
 * reader until the prompt ends. Returns the first non-zero qe_forward_step code,
 * having fed the tokens before it. */
int qe_forward_prefill(qe_forward *f, const int *tokens, long n, float *logits_out);
/* Chunk-progress hook for a prefill that takes minutes.
 *
 * An 11,758-token prompt is 1,470 chunks of eight tokens and about 860 s of
 * network, and a client sitting on the first byte of a stream cannot tell that
 * apart from a server that hung - crush gives up after 2 min of silence by
 * default. The hook runs on the generating thread, which is the thread that owns
 * the client, once a chunk has landed, with the tokens fed so far out of the
 * tokens this call was given; what it does with that (a keepalive, a progress
 * line) is its own affair. Non-zero abandons the rest of the prompt, which is how
 * a reader that went away during ingestion stops the engine instead of waiting
 * for it. Install before the call and clear it afterwards; it says nothing about
 * qe_forward_step, which is called once per token anyway. */
typedef int (*qe_progress_cb)(void *user, long done, long total);
void qe_forward_set_progress(qe_forward *f, qe_progress_cb cb, void *user);

/* How big a projection has to be for a chunked pass to walk it once for the whole
 * chunk rather than once per token: a tensor of at least `bytes` goes through
 * qe_gemv_multi_t, and 0 sends every projection through T ordinary calls.
 *
 * The default is 1 MiB, which in this checkpoint means every projection, because
 * the walk is the fastest setting this engine has for feeding a prompt: 384 tokens
 * of one prompt, both ways, in one load, measured 57.8 ms/token walked against
 * 69.1 ordered, which is 1.20x (`make forward` prints the pair, and §12 of
 * docs/DESIGN.md is the post-mortem of the comparison that used to say the
 * opposite).
 *
 * Also worth being explicit about, because it is not free the way width is: the
 * ordered form is bit-identical to one token at a time, and a Q8_0 tensor taken
 * through the shared walk reassociates (include/qe.h, qe_gemv_q8_0_multi). On this
 * checkpoint the worst move in a logit measured 1.6 against a head whose largest
 * logit was 16.9 over a 951-token prompt, 9 % of that - a different answer sometimes,
 * and not a rounding.
 * The default is the walk because ingestion is what the engine spends most of a
 * coding turn on; 0 is the way to ask for the exact bits, and setting it
 * mid-process is for measurement. `make prefix` is what shows what either setting
 * does to a turn that goes back to a stored boundary. */
void qe_forward_set_proj_chunk(qe_forward *f, size_t bytes);
size_t qe_forward_get_proj_chunk(const qe_forward *f);

/* Whether a chunk's projections recode the chunk's T activation rows to int8 once per
 * projection and walk the weights with a vector integer multiply-accumulate (the same
 * walks `qe_moe_set_int8` puts under the routed experts) instead of dequantising every
 * 32 weights into floats. Unlike the threshold above this changes the number: the
 * activations become 7 bits and a scale per 32 values, so the leg that turns it on is a
 * measurement of speed *and* of loss, never of one alone. On by default; QE_I8_PROJ=0
 * is the operator's switch, and the panel is refused - the float walk taken - for a
 * token group below two, a K with ragged 32-value blocks, or a type with no integer
 * walk, which is every type except q8_0, iq4_nl and iq3_s. */
void qe_forward_set_proj_i8(qe_forward *f, int on);
int qe_forward_get_proj_i8(const qe_forward *f);

/* Whether a chunk's dense projections walk a second copy of their weights: the same
 * matrices requantised to one scale for a whole output row instead of one for every 32
 * values, which is what leaves `qe_gemv_q8_0_flat_i8` with no scale exit inside its block
 * loop and lets three output rows share one load of the activation panel (ideas.md 7.18,
 * 1.84x of the walk `qe_forward_set_proj_i8` selects at T=512 in isolation and 1.65x of it
 * in the engine, block_proj 1.77 against 2.92 ms/token at the width a chunk ingests at).
 *
 * The copy is staged during the open, in full, out of the bytes that are already resident,
 * and only when this was asked for at open with QE_I8_FLAT: 496 dense matrices and 3.3 GiB on
 * this checkpoint, which is the Q8_0 matrices less the routed experts and the embedding table.
 * An open that cannot afford it says so with the two numbers and goes on walking the Q8_0
 * blocks, so this is the one setting that can refuse - setting it on when the copies were never
 * staged leaves the getter answering 0, which is how a caller tells "never built" from "made no
 * difference". On by default: the head's answer to the prompt `make forward` walks moved
 * 6.9e-04 nats on the token the engine goes on to say, 10 % of its own range across its top 32
 * with the same next token; QE_I8_FLAT=0 is the way back.
 * It needs the row-scaled panel, and it is refused for a token group the tile does not
 * divide, which is the ragged tail of a prompt. */
void qe_forward_set_flat_i8(qe_forward *f, int on);
int qe_forward_get_flat_i8(const qe_forward *f);

/* The same dense matrices a second time, this time in IQ4_NL, for the other end of the
 * width range. A chunk of one walks a matrix once, so its time is its bytes over the bus
 * and 4.5 bits a weight against 8.5 is nearly the two to one the byte ratio says
 * (ideas.md 4.1: 1.74x of the Q8_0 walk at T=1, 0.96-1.02x at the width a chunk ingests
 * at), and this copy is therefore read by `chunk_T == 1` and by nothing else - an
 * ingested token keeps walking the bytes it walked before. Staged at open with the flat
 * copies and refused with two numbers on a host that cannot afford them, so a getter
 * answering 0 distinguishes "never built" from "asked for and made no difference".
 * QE_I4_DENSE=0 is the way back; `make score`'s decode A/B is the price. */
void qe_forward_set_decode_i4(qe_forward *f, int on);
int qe_forward_get_decode_i4(const qe_forward *f);

/* Which dense matrices the copy above is refused to, as a mask of the classes below. A
 * copy is staged for every candidate matrix whatever the policy, so this is a reading and
 * not a second load: `make score` races the policies against each other in the load that
 * staged them. QE_I4_DENSE=all|lean|nohc|noedge|0 names the four of them worth naming;
 * "all" is what ships, and the two classes it can hold back are the hyper-connection
 * mixers and the trunk's first and last blocks (ideas.md 4.1, 7.23). */
typedef enum {
    QE_DCLASS_PLAIN = 0, /* an ordinary projection: the copy is always there for it */
    QE_DCLASS_HC = 1,    /* a hyper-connection mixer */
    QE_DCLASS_EDGE = 2   /* the trunk's first block and its last two */
} qe_dense_class;
void qe_forward_set_decode_tier(qe_forward *f, int classes);
int qe_forward_get_decode_tier(const qe_forward *f);

/* The output head a third and fourth time. It is the one matrix a lone token walks whole
 * and only once, so its bytes are the largest single item in a decode step's bill, and on
 * this checkpoint it is a Q6_K matrix - the format whose walk this host is slowest at.
 * Both copies are staged whenever the head is Q6_K at all, whatever the setting below
 * says, because the A/B that decides between them (`make score`'s decode pass) walks all
 * three readings in the one load that staged them. A draft head opened with --mtp gets the
 * same two copies of its own output matrix - unless it is the same bytes as the trunk's,
 * which is the usual answer and then one pair of arenas serves both walks - because a
 * draft row walks it exactly once, which is what made it the next thing in a speculative
 * step after the step itself (ideas.md 7.23).
 *
 * Asking for a copy that was never staged leaves the getter answering QE_HEAD_STORED, so
 * a caller can always tell "never built" from "built and made no difference". */
typedef enum {
    QE_HEAD_STORED = 0, /* walk the head as the file wrote it */
    QE_HEAD_I4 = 1,     /* walk the IQ4_NL copy: 4.5 bits a weight */
    QE_HEAD_Q8 = 2,     /* walk the Q8_0 copy: 8.5, and a simpler walk */
} qe_head_copy;

void qe_forward_set_head_copy(qe_forward *f, int which);
int qe_forward_get_head_copy(const qe_forward *f);

/* Whether a walk of more than one row of a head - a speculative verify window - reads that
 * copy as well as a lone row does. Several rows read the head once and multiply it T times,
 * which is the regime where a Q8_0 copy's 30 % extra bytes might count against the ALU they
 * save; `make head`'s arm says they do not, and a winning speculative pass goes 1.05x faster
 * with it, so this ships on. QE_I4_HEAD_WIN=0 turns it off; it is inert when
 * `qe_forward_get_head_copy` says QE_HEAD_STORED. */
void qe_forward_set_head_window(qe_forward *f, int on);
int qe_forward_get_head_window(const qe_forward *f);

/* The widest chunk of prompt tokens that shares one weight walk, and the width a pass
 * opens with. Two numbers because they answer different questions: the max is what the
 * chunk buffers are sized for and what a width sweep may ask for, the default is what
 * serving runs. Both are settled by the sweep in `make forward` (§12), and a wider
 * default than the max is refused rather than clamped in silence.
 *
 * 128 was the max because that is where the measured sweep stopped paying: 45.5 ms a
 * token at width 8, 41.1 at 16, 38.1 at 32, **34.7** at 64, 34.8 at 96, 35.3 at 128 -
 * one plateau and not three settings. It is 512 now because a plateau at the top of a
 * table is not a ceiling, and the byte model says the table ends too early: what a
 * chunk shares is a tensor's bytes, so the bytes an ingested token owes fall roughly as
 * 1/T, and the one part of this checkpoint that is 57 GiB wide is the routed experts - a
 * window of 16 dedupes 160 picks to 138 distinct of them, a window of 64 to 366 of 640,
 * a window of 512 to essentially all 512. Widening the window is the only lever that
 * moves that bill, and llama.cpp b12359 on this host against this file makes the size
 * of the lever plain: its prompt throughput, measured with its own batch defaults and
 * `-ngl 0 -t 20 -fa on`, is 38.8 tok/s at a 64-token batch, 49.5 at 128, 57.3 at 256
 * and 50.3 at 2048. Its ubatch is 512 - four times the ceiling this pass ran under
 * until now - which is most of its ingestion advantage; at an equal 64-row batch it is
 * also 1.36x ahead of this pass per token, and that part is a kernel and scheduling
 * question rather than a batch one.
 * Raising the ceiling costs buffer sizing and nothing else: the chunk scratch in
 * `src/forward.c` scales with it, `PLE_PENDING` is defined from it, and the grouped
 * expert pass grows its per-thread partials to `n_threads * T * K` per layer - at 512
 * that is 4.8 GB of expert partials over the 48 layers on this checkpoint, which is the
 * number to look at before going further than 512, not the 210 MiB of chunk scratch.
 *
 * The sweep has now measured past 128, and the default is the fastest width it found:
 * 92.2 ms a token at width 1, 45.6 at 8, 41.4 at 16, 37.2 at 32, 35.1 at 64, 33.4 at 96,
 * 33.1 at 128, 34.1 at 192, 32.2 at 256, 31.8 at 384, **31.1** at 512
 * (`/tmp/f9_wide.log`). It is not monotone - 96 read 33.4 and 192 read 34.1 across a
 * plateau, which is a run of a plateau rather than eight
 * settings - and it only pays at the top after one other thing was fixed: the same
 * sweep before the multi-row Q8_0 walk was re-tiled read 42.9 at 512, a 22 % penalty
 * for a wide chunk, because every weight row re-read the whole 512-token activation
 * panel. With the rows tiled to the L2 budget the panel is read once per tile and the
 * curve falls the way the byte model says it should. Both rows are `make forward`;
 * two earlier loads of this configuration read 30.9 and 31.4 in the 512 row, which is the
 * 1.6 % one load of this checkpoint is worth, and the ceiling stays at 512 because that is
 * as far as either has been measured. */
#define QE_CHUNK_MAX 512
#define QE_CHUNK_DEFAULT 512
/* Whether serving groups the expert stage of a chunked prefill. Measured on, and the
 * size of the win is a function of the width it runs at: 46.8 -> 30.5 ms/token at
 * width 512 (1.53x), 46.3 -> 33.3 at 128 (1.39x), 46.1 -> 34.2 at 64 (1.35x), and
 * 46.7 -> 37.2 at 32 (1.26x) - `make forward` A/Bs all four on one load and prints each
 * (`/tmp/f9_wide.log`). Grouping reads the distinct experts of the window,
 * so the more rows in the window the more of them are already warm; at the shipping width
 * the stage it runs is half of an ingested token (14.56 ms of a 29.44 ms profiler sum),
 * which is the opposite of the ~15 % it looked like when the sweep stopped at width 16. */
#define QE_MOE_GROUP_DEFAULT 1

/* How many rows of a chunk one grouped expert pass covers. The chunk stays one chunk
 * for every other stage and for the router; the tile only decides how many rows share
 * a bucket, and therefore how many rows one walk of an expert's quantiser feeds. That
 * is the whole cost of the stage: the walk is once per eight rows of a bucket whether
 * the bucket holds nine of them or five, and at width 512 an expert averages ten routed
 * rows, so the second walk of it is nearly empty. Halving the rows a pass covers barely
 * changes what it reads (509 distinct experts at 256 rows against 512.0 at 512,
 * `/tmp/moe_t19.log`) while halving the number of walks per row, and `make moe` costs
 * that 8.9 % cheaper per token: 57.035 ms for 256 rows against 124.171 for 512, which is
 * 0.2228 ms per token per layer against 0.2425. Below that a pass stops paying for its
 * own barrier - 35.151 ms for 128 rows is 0.2746 per token.
 *
 * 0, the whole chunk, is what ships until the engine says otherwise. That curve is a
 * synthetic one (uniform routing, and a bucket that is therefore as even as the router
 * allows), and §7.2b is this repository's standing warning that the two readings of a
 * bucket question can disagree; `make forward` sweeps {0,128,256,384} against the real
 * router at the width it serves, four passes with the sweep order rotated so the
 * machine's drift cannot be read as a window, and the default moves to whatever wins
 * there. */
#define QE_MOE_TILE_DEFAULT 0

/* Whether the expert stage of a chunked prefill is one dispatch of a stage graph
 * (qe_pool_run_items) instead of seven dispatches run one after another: the three
 * tasks of the grouped expert walk, the shared expert's three projections and the
 * silu between them. Four of those never needed to be ordered against the other
 * three - both halves read the same block output and write disjoint buffers - and
 * six hand-backs a layer is not free at nineteen threads (§7.8 priced one at tens
 * of microseconds against an ~800 dispatches a token). The grouped form runs the
 * two halves at once, which is the co-scheduling the record once dismissed on a
 * fixed thread split, and eats the tail of whichever of them finishes first.
 *
 * On, because `make forward` said so: it A/Bs the two against the same prompt on one
 * load, four passes with the legs alternating inside each, and the graph won all four
 * (22.89/22.50, 22.98/22.26, 22.85/22.50, 22.88/22.59 ms/token in `/tmp/f25_wide.log`,
 * 1.026x at the fastest of each, the expert stage 11.11 ms against 10.67) while leaving
 * the trunk byte for byte where the queued schedule left it. That last part is what took
 * two loads to get: the graph carries the chunk recode as items of its own now, because a
 * graph with no panel had no integer walk and was quietly a different arithmetic than the
 * schedule it was being measured against (`/tmp/f24_wide.log` printed that as a trunk that
 * was not the same bytes, and the 1.009x it replaced was the graph paying for the float
 * walk it fell back to). `QE_MOE_CO=0` is the way back. */
#define QE_MOE_CO_DEFAULT 1

/* How many prompt tokens share one walk of the weights: the width a chunked prefill
 * feeds through a stage at a time, clamped to 1..QE_CHUNK_MAX (above, which is also
 * what sizes the chunk buffers). The default is QE_CHUNK_DEFAULT unless QE_CHUNK says
 * otherwise, and a sweep is what settled it: 92.2 ms/token at width 1 against 45.6 at 8,
 * 41.4 at 16, 37.2 at 32, 35.1 at 64 and **31.1** at 512, with the projection walk on at
 * the shipping threshold (`make forward` prints the whole row). The row is monotone in
 * direction and noisy in detail - width 5 and width 6 measure the same, 96 reads 33.4 while
 * 128 reads 33.1 and 192 reads 34.1 - so what it supports is "wider keeps paying to about
 * 128 and again past 256" and not "384 beats 256". Every stage of a chunk schedules T rows
 * at once, so a wider chunk is
 * also a row that waits longer for the slowest recurrence in it, and that is the shape of
 * the plateau: the dense stages stop getting cheaper at about the point the recurrences
 * start getting dearer, and around width 64 the mixers are visibly slower again
 * (1.54 ms/token of hc_attn at width 16 to 2.13 at 64, back to 1.56 at 512, which no row
 * block fixes -
 * `make walk` is the instrument). Width is scheduling and not arithmetic, but only with
 * the walk off: a width-1 chunk never blocks a projection, so the widest chunk differs
 * from the narrowest by the reassociation above as well as by the scheduling. It is
 * settable for the same reason the walk is - one load, every width - and 1 is the ordinary
 * one token per weight pass walk. */
void qe_forward_set_chunk(qe_forward *f, int width);
int qe_forward_get_chunk(const qe_forward *f);
/* The widest chunk this build can run, which is the largest width
 * qe_forward_set_chunk accepts rather than the width it opens with. A sweep needs to
 * be able to ask up to it without restating it from the engine's private macro. */
int qe_forward_chunk_max(void);
/* Whether the expert stage of a chunked prefill groups the chunk's rows by expert
 * (qe_moe_forward_multi) or runs the serial per-token pass (qe_moe_forward) T times.
 * Grouping reads the *distinct* experts of the window instead of T*top_k of them, which
 * is a byte saving, but it sums a row's expert contributions in bucket order instead of
 * route order and, where the type has one, takes the shared-dequant kernel, which moves
 * the block scale onto the weight. Two reassociations of ~1e-6 relative, so the switch
 * exists before the measurement does and one load decides both. */
void qe_forward_set_moe_grouped(qe_forward *f, int on);
int qe_forward_get_moe_grouped(const qe_forward *f);
/* How many rows of a chunk one grouped expert pass covers, clamped to 0..QE_CHUNK_MAX
 * (above, where is the measured cost curve). 0 and any width at or above the chunk pass
 * the whole chunk in one call, which is what shipped before the curve was measured; a
 * smaller tile hands the expert stage several windows of the same chunk, so a bucket is
 * narrower and one walk of an expert's quantiser is less likely to be nearly empty.
 * Router, per-row top-k selection and the shared expert stay whole-chunk: this is a
 * width for the bucketed kernels only. A row lands in one window, and a window's buckets
 * are sorted by expert id exactly as a chunk's are, so a row's contributions are summed
 * in the same order whichever tile computes it. What a narrower window can change is
 * which rows share a walk of a bucket's rows, and where the type has several multi forms
 * that is a different order of the same products - the same ~1e-6 the grouping switch
 * itself moves. It only costs what it saves when a row still shares an expert with
 * another row of its window, so it is a lever on wide chunks and a no-op below it. */
void qe_forward_set_moe_tile(qe_forward *f, int rows);
int qe_forward_get_moe_tile(const qe_forward *f);
/* Whether a chunk's expert stage is one dispatch of a stage graph, with the shared
 * expert's projections running beside the grouped expert walk instead of after it
 * (QE_MOE_CO_DEFAULT, which is on, and `make forward` is what priced it). The
 * arithmetic of both halves is untouched: same kernels, same rows, same order of
 * sums per output, so the trunk the two produce is the same bytes. What differs is
 * which thread is where, which is why this is a switch and not a rewrite. */
void qe_forward_set_moe_co(qe_forward *f, int on);
int qe_forward_get_moe_co(const qe_forward *f);
/* Whether a window's rows are read out of the head together: one walk of the output
 * matrix with the rows blocked through it, rather than one walk of the same 497 MiB
 * per row (on, QE_HEAD_BATCH=0 to turn off). No arithmetic differs - each row comes
 * from the same kernel on the same inputs - so what this moves is only bytes read
 * twice, which is why `make forward` can A/B it on one load. */
void qe_forward_set_head_batch(qe_forward *f, int on);
int qe_forward_get_head_batch(const qe_forward *f);
/* Whether the sparse attention walks the selected cells once per KV group, serving
 * the twelve query heads that share it from one read of a key and value row (the
 * default), or once per query head. The two read the same rows, multiply the same
 * numbers and normalise the same way; what differs is the order the sums are formed
 * in and, for the first, how many times a cache line is walked. Worth about 1e-6
 * relative on the attention output, so one load can answer both which walk is faster
 * at a given context length and whether the loss moved. */
void qe_forward_set_attn_fused(qe_forward *f, int on);
int qe_forward_get_attn_fused(const qe_forward *f);
long qe_forward_vocab(const qe_forward *f);
int qe_forward_ctx(const qe_forward *f);
long qe_forward_pos(const qe_forward *f); /* tokens fed so far */
/* 1 when the key and value cache holds q8_0 rows, 0 when it holds f32. Reported
 * by every layer that prints what it is serving from, because it is a fact about
 * the answers rather than a detail of the layout. */
int qe_forward_kv_q8(const qe_forward *f);
/* Bytes of sequence state this forward owns, and the bytes each further token of
 * context costs. Either may be NULL. Concurrency is capped by these two and not by
 * bandwidth, and the cache's width is most of the second one: 12 KiB per token at
 * the q8_0 default against 48 with f32 keys and values. */
void qe_forward_state_bytes(const qe_forward *f, size_t *total, size_t *per_token);
/* ---------------------------------------------------------------- rewind points
 *
 * A sequence is two shapes of state, and the difference decides what putting one
 * back costs. Everything that grows with the sequence - keys, values, the
 * indexer's raw and pooled keys, the token ids - is append-only: the position says
 * how much of it is live, and the next token overwrites the rest. So a rewind
 * copies only what is a recurrence in place, which is three buffers: the delta-net
 * matrices and the two short convolution rings, plus the position those rings are
 * indexed by (each ring's head is `pos % rows`, so the position is what makes the
 * copy exact rather than approximately right).
 *
 * That is 118 MB for this checkpoint whatever the prompt is, against the 650 MB a
 * copy of the cache range would cost at 11,000 tokens - and the important half is
 * that it does not grow with the prompt at all. The cost is three copies, about
 * 3 ms at the memory ceiling, plus 4 bytes per token covered.
 *
 * What is deliberately not in one, and why that is safe:
 *   - the PLE row cache, keyed by row id rather than by position, so a stale entry
 *     is a miss and never another sequence's row; the prefetch queue *is* positional
 *     and is therefore emptied by a restore;
 *   - the MoE expert histogram, which counts rather than computes;
 *   - the diagnostics, which describe what this forward has run and not where the
 *     sequence stands;
 *   - the chunk cursors, which every chunk sets before it reads them.
 *
 * Take a snapshot between calls, never between two tokens of one chunk, because a
 * chunk is where the state is mid-update. A prompt that has just been ingested is
 * the natural place: qe_forward_prefill ends on a chunk boundary. */
typedef struct qe_state qe_state;

/* Bytes one rewind point of this forward costs, and therefore how many a memory
 * budget buys. Includes the ids it covers, which is what lets a caller match a
 * prompt against it without going near the sequence. */
size_t qe_state_size(const qe_forward *f);
qe_state *qe_state_new(const qe_forward *f); /* NULL when out of memory */
void qe_state_free(qe_state *s);
/* Records the state as it stands, position included. 0, or -1 if `out` came from a
 * forward of a different shape, in which case nothing was written. */
int qe_forward_snapshot(const qe_forward *f, qe_state *out);
/* Puts it back - the three recurrences, the position, the ids, an empty prefetch
 * queue - and returns 0, or -1 without touching `f` at all. The snapshot stays
 * valid and can be restored repeatedly. */
int qe_forward_restore(qe_forward *f, const qe_state *in);
/* How many tokens a rewind point covers, and the ids of those tokens. Both belong
 * to the snapshot and are good until it is freed. */
long qe_state_pos(const qe_state *s);
const uint32_t *qe_state_ids(const qe_state *s);
/* The vector the output projection reads: the head mixer's output for the token
 * the last step fed. Two implementations of one graph disagree about 248320
 * logits and about nothing else, so comparing this is comparing the whole trunk;
 * and because the projection is a single matrix, a difference here is neither a
 * rounding argument nor a vocabulary argument. Returns the number of floats
 * written, or -1 if dst is too small or probes have overwritten it. */
long qe_forward_hidden(const qe_forward *f, float *dst, long n);
void qe_forward_diag_of(const qe_forward *f, qe_forward_diag *out);

/* ------------------------------------------------------- a rewindable window
 *
 * Speculation feeds a target pass a row per draft token plus the token they all
 * follow, and keeps only the rows that agreed. Everything the target computed
 * past the first disagreement therefore has to be un-done, and most of a
 * sequence's state cannot be: the gated-delta-net state is a dense recurrence with
 * no position in its address, so the state after row 5 is not recoverable from the
 * state after row 8.
 *
 * The rest is positional and so self-correcting - the attention cache and the
 * indexer's raw and pooled keys are written at an address derived from the
 * position, which means re-feeding a position overwrites exactly what the abandoned
 * row left there. The two conv histories are the shape that argument does not
 * cover, and they are copied with the state: a *minimal* ring holds `kernel-1`
 * rows for the whole sequence, so a tap of position `pos` is not the row addressed
 * by `pos` but whatever those slots hold, and an abandoned row at `p > pos` has
 * left its `qkv` in one of them to be read as the retained row's own history.
 *
 * A window's checkpoint is therefore one copy of the delta-net state and of both
 * conv histories per row, taken while that row's layer is still walking - which is
 * the only reason the rows of a checkpointed window run one at a time through the
 * block core instead of one dispatch per layer. One copy is 112 MiB of the 118 MiB
 * state; call it 3 ms at the memory ceiling, and it is spent inside the pass that
 * was already reading those bytes.
 *
 * As with a rewind point, the rows of a window are walked with the block core
 * once per row, so a checkpointed window is for the few rows of a draft and not
 * for a prompt: prefill keeps the chunked core and never takes one. */
typedef struct qe_ckpt qe_ckpt;

/* Room for the state after each of `rows` - 1 rows: the last row leaves the
 * sequence where it stands, so its copy would never be read. NULL when out of
 * memory, which for 8 rows is 788 MiB of the 118 MiB state. */
qe_ckpt *qe_ckpt_new(const qe_forward *f, int rows);
void qe_ckpt_free(qe_ckpt *c);
/* One target pass over `T` rows starting at the position the sequence stands at,
 * with the state after every row but the last kept in `c` (whose `rows` has to be
 * at least `T`). No logits: they are per row, and see the next call. `T` of one is
 * legal and is what a window with nothing drafted is. */
int qe_forward_step_window(qe_forward *f, const int *tokens, int T, qe_ckpt *c);
/* The logits a row of the window that just ran would have produced: that row's
 * head mixer and one walk of the output matrix. Called in row order, once per row,
 * and stopped at the first row whose token was not the one drafted.
 *
 * Stopping early still saves the trunk's work, which is the part a row past a
 * disagreement has no claim on, but it does not save the head: a window of more than
 * one row pays for all of its readable rows the first time one is asked for, in one
 * blocked walk of the output matrix rather than one walk per row. The row a draft
 * abandoned then costs a copy of the vocabulary, and the abandoned rows are what a
 * drafter that reads its own prompt is built to keep few. */
int qe_forward_row_logits(qe_forward *f, int row, float *logits_out);
/* All `rows` rows at once, `rows` * vocab floats out, from row 0 of the window in
 * flight. Same numbers as reading them one at a time - each row comes from the same
 * kernel on the same inputs - for one walk of the output matrix instead of `rows`
 * of them, which is the difference between 497 MiB and `rows` times it. Returns -2
 * if any row came out non-finite. */
int qe_forward_window_logits(qe_forward *f, int rows, float *logits_out);
/* Back to the state after row `boundary` of the window that `c` holds, with the
 * sequence's length set to match. `boundary` is one of `1 .. rows - 1`: the last
 * row is where the window already left the sequence, and the state before its first
 * row is the one thing a window cannot put back. Everything that is not addressed
 * by position is copied back - the delta net and both conv histories - and the
 * rows past it in the caches are left as they are, because every one of them is
 * overwritten before it is read again, which is the whole argument for that part
 * being a position and not a restore. The rows up to and including the boundary
 * stay readable with qe_forward_row_logits, because a rewind moves the state and
 * not the trunk's activations, and the row a window ended on is the one the
 * caller still wants the head's answer for. Returns the position it left, or -1. */
long qe_forward_rewind(qe_forward *f, qe_ckpt *c, int boundary);
/* The same boundary, as a rewind point: the three recurrences, the position, and
 * the ids up to it, into `out` (which must have been made for this forward, and is
 * completely overwritten). What the window itself cannot put back - a state for
 * later, rather than one to go back to - is what this is for: see
 * qe_forward_step_window for why the row a window starts on is a position some
 * later request will want returned. Returns 0, or -1 with `out` untouched. */
int qe_ckpt_state(const qe_forward *f, const qe_ckpt *c, int boundary, qe_state *out);

/* ------------------------------------------------------------- draft head
 *
 * The nextn block of a head opened through qe_forward_opts::mtp_model. It is one
 * more decoder block - full attention with the indexer, routed and shared experts -
 * with a fold in front of it and a mixer behind it, and it is the block these
 * checkpoints were trained to draft with.
 *
 * What it reads, per row, is a pair: the embedding of the token at position p and
 * the four-stream residual of the position *before* it. That is the fold as the
 * head was trained (ref/qwen4exp.cpp:629-636 normalises both halves and projects
 * their concatenation), and it is what makes a chain of draft rows close: a row's
 * exported residual is the h for the token that row predicted, so row 0 seeds from
 * the trunk and every row after it seeds from the row before.
 *
 * The head's own keys and values live in one more slot of the same caches, at the
 * positions of the tokens fed to it, so a chain leaves the head able to attend over
 * the sequence it drafted. Nothing here moves the trunk's position or touches its
 * state: a draft row is a query, and the sequence stays where the target left it. */

/* 1 when a head is open and qe_forward_mtp_row can be called. */
int qe_forward_has_mtp(const qe_forward *f);

/* The width of the residual a draft row reads and writes: HC streams of the model
 * width, which is what a caller sizes `h_in` and `h_out` with. */
int qe_forward_hc_dim(const qe_forward *f);

/* The hyper-connection residual of one row of the chunk that just ran: the
 * hc_dim-wide four-stream residual before the output mixer, which is exactly the
 * half of a draft row's input that comes from the target. `row` is a row of the
 * pass in flight (qe_forward_row_logits takes its rows the same way), or -1 for the
 * last of them - the last token the pass consumed, which is the row a draft of the
 * next token seeds from. The return is hc_dim, or -1 when there is no such row. */
long qe_forward_row_hidden_hc(const qe_forward *f, int row, float *dst);

/* `T` draft rows: tokens `toks` at positions `pos0`..., each with the residual in
 * row t of `h_in` (rows `qe_forward_row_hidden_hc` wide apart). The exported
 * residual of row t goes to row t of `h_out` (may be NULL), and when `logits` is
 * not NULL the head runs for every row, T rows of the vocabulary.
 *
 * `h_out` may not alias `h_in` unless they are the same pointer; `h_in` may be the
 * trunk's own residual, which is why the rows are read before anything is written.
 * 0 on success, negative when there is no head, the row does not fit, or the walk
 * produced a non-finite value. */
int qe_forward_mtp_rows(qe_forward *f, const int *toks, const float *h_in, int T, long pos0,
                        float *h_out, float *logits);

/* One draft row, which is the call a drafter actually makes: T of 1. */
int qe_forward_mtp_row(qe_forward *f, int token, long pos, const float *h_in, float *h_out,
                       float *logits);

/* The target's four-stream residual at the position of a token this pass consumed,
 * which is the half of a draft row's input that comes from the target. A pass keeps
 * the rows it walked and the one row before them, so every position a drafter can be
 * standing at is answerable and older ones are not: -1 when this pass has no such
 * row, rather than the nearest one it does. The return is hc_dim.
 *
 * `p` is a position, not a row: which position pairs with the token a draft row is
 * fed is `qe_forward_get_mtp_back`'s answer, and a caller chains a draft by feeding
 * row i the residual row i-1 exported. */
long qe_forward_mtp_hidden(const qe_forward *f, long p, float *dst);

/* Feed the head the prompt it was not part of: rows for cells 1..n-1 of the
 * sequence already fed, taken from the residuals the prefill left behind. A prompt
 * of n tokens leaves the head able to attend over n-1 of its cells, and it is what
 * a draft row's attention was trained to expect. On by default when a head is open;
 * 0 turns it off to measure what it is worth (it is one more block walked per
 * prompt token, and the prompt is the one place a chunk makes that cheap). */
void qe_forward_set_mtp_warm(qe_forward *f, int on);
int qe_forward_get_mtp_warm(const qe_forward *f);

/* How far the residual a draft row reads lags the token it is fed: 1 (the default,
 * and what the head files' nextn block trains on) pairs the embedding of a position
 * with the residual of the one before it, 0 pairs it with its own. It is a knob and
 * not a constant because the only instrument that can tell the two apart is an
 * accept rate, and both are one load: flipping it re-warms nothing, so compare it on
 * two runs of one prompt from a reset pass, which is what `make forward` does. */
void qe_forward_set_mtp_back(qe_forward *f, int back);
int qe_forward_get_mtp_back(const qe_forward *f);

/* Depth probes: the logits the head would have produced at a given depth.
 *
 * After each listed layer the head mixer and the output projection run on the
 * residual as it stands, and those logits go to buf + i*vocab (the caller
 * allocates n * vocab floats). Teacher-forced loss against the same continuation
 * then says which layer class stopped helping, which is the difference between
 * "the model is wrong" and "layer 1's PLE is wrong" - and without it every
 * hypothesis costs one load per guess. Costs one lm_head per probe per step;
 * probes are off unless this is called, and off again with n = 0. */
void qe_forward_probe_set(qe_forward *f, const int *layers, int n, float *buf);

/* Swap the hypothesis and the instrumentation of an open pass without opening a
 * second one, which matters because each holds a sequence's worth of state
 * caches. Call reset() afterwards: state a different variant produced is not
 * comparable. */
void qe_forward_reconfigure(qe_forward *f, unsigned variant, int profile);

/* ------------------------------------------------------------------ sampler
 *
 * Temperature, top-k, min-p and top-p over a logits vector, with the defaults the
 * checkpoint ships (top_k 20, top_p 0.95, temp 1.0 in general.sampling.*).
 * Greedy when temp is 0, which is what a determinism check wants. */

typedef struct {
    float temp;   /* <= 0 = greedy */
    int top_k;    /* <= 0 = off */
    float top_p;  /* >= 1 = off */
    /* Drop anything whose probability is below this fraction of the best token's.
     * The one filter that scales with how confident the model is, which is why a
     * flat 0.0 is the interesting value: it changes nothing when the distribution
     * is peaked and cuts hard when it is flat. 0 = off. Applied after top-k and
     * before the nucleus, so it can never remove the argmax. */
    float min_p;
    unsigned seed;
} qe_sampler_opts;

/* Filled with the file's own sampler.* keys where it has them and the
 * documented defaults otherwise. */
void qe_sampler_opts_from_model(qe_model *m, qe_sampler_opts *o);

/* One call per logits vector: sorts what it has to, draws, returns the id.
 * Returns -1 when every logit is -inf or the vocabulary is empty. */
int qe_sample(const float *logits, long n, const qe_sampler_opts *o, uint64_t *rng);

/* ------------------------------------------------------------- engine hook
 *
 * The smallest thing a generator has to be for the HTTP layer to use it. It
 * exists so the serving path can be finished, tested, and curl-driven while
 * the forward pass is still missing, and so "the model" is one pointer swap
 * rather than a refactor of six files.
 *
 * Generation is push-based on purpose: the caller cannot batch, but it can
 * cancel, which is the only way a disconnected SSE client stops a 12 GB/s
 * decode from running to completion for nobody.
 */

typedef struct {
    const char *mime; /* "image/png", "image/jpeg", ... or NULL if unknown */
    const void *bytes;
    size_t len;
} qe_image;

typedef struct {
    const char *text; /* rendered prompt, UTF-8, NUL-terminated */
    const qe_image *images;
    int n_images;
    int is_chat; /* 1 = chat transcript, 0 = raw completion prompt */
} qe_prompt;

typedef struct {
    int max_tokens;
    float temp;  /* 0 = greedy */
    /* 0 in each of these three means "what the checkpoint says", which is how a
     * client that sends nothing gets the model's own sampling. Beyond that a
     * request can be explicit about turning a filter off, which is what the
     * negatives are for: top_k < 0 and min_p < 0 mean off, top_p 1 means off.
     * The HTTP layer is what tells "absent" from "sent as 0", and maps a sent 0
     * to the negative, so a client can always ask for the whole distribution. */
    int top_k;
    float top_p;
    float min_p;
    unsigned seed;
    int spec_k;  /* draft length; 0 disables speculation */
    int eos;     /* vocabulary id that ends generation, or -1 */
} qe_gen_opts;

typedef struct {
    long tokens;
    double seconds;
    long steps, drafted, accepted; /* verbatim from qe_spec_stats */
    double tok_per_step;
    int truncated; /* hit max_tokens rather than eos */
    /* The token ceiling the reply actually ran under: what was asked for, or
     * what the context left after the prompt, whichever was smaller. A caller
     * that asked for 64000 needs this to tell a finished answer from one the
     * window ended, which is the same question `truncated` answers when the
     * client set its own limit. */
    long budget;
    /* Tokens of the prompt that did not have to be put through the network
     * because the sequence already ended where this prompt begins. A client that
     * charges for what was computed needs this number, and so does anyone
     * wondering why a second turn was quick. */
    long cached;
    /* Tokens at the beginning of the prompt that had to be thrown away because the
     * prompt plus the answer did not fit the window. The sequence is fixed, so what
     * leaves is the front of the transcript - the system prompt, the earliest tool
     * result - and a reply that ignored them looks like a model that cannot
     * follow instructions rather than like a window that ran out. A caller that
     * could have sent less needs to see this number rather than infer it. */
    long dropped;
    /* How the reuse above was obtained, since the two cost differently: 1 means the
     * sequence was put back onto a rewind point kept from an earlier request, 0
     * that it was still standing where this prompt begins. A turn that reuses by
     * extension has not gone anywhere; one that reuses by rewind copied state back
     * and then read only what the new turn added. */
    int rewound;
    /* How far this prompt agreed with the rewind points on hand, in tokens, whether
     * or not any of them was usable: the longest agreement between this prompt and
     * any stored boundary, full or partial. It is what turns "the cache did not
     * help" into a sentence with a position in it. A prompt that rewrote something
     * in its own history reports the token where the rewrite starts, which is the
     * thing to fix; one that reports its whole length matched a boundary and yet
     * reused nothing has a different problem. */
    long rewind_lcp;
    /* Microseconds spent taking and restoring rewind points for this request. It is
     * the cost side of `cached`: a 118 MB copy is a few milliseconds, and a number
     * that says so is what makes the saving checkable rather than asserted. 0 when
     * no copy happened. */
    long snapshot_us;
} qe_gen_stats;

/* Emits generated text in order; `utf8` is a fragment and need not be a whole
 * token. Non-zero stops generation. */
typedef int (*qe_text_cb)(void *user, const char *utf8, int nbytes);

/* ---- the same stream, with the structure left in ----
 *
 * A reply from this model is not only text. Its reasoning block, the end of its
 * turn, and a request to call a tool are control tokens, and a client that is
 * told "here are three characters" cannot tell any of that apart from prose.
 * The text callback above drops them, which is exactly right for a completion
 * endpoint and useless for a chat one, so this reports every token the decoder
 * emits, with two rules:
 *
 *   - a control token arrives with nbytes 0 and no text, because its bytes are
 *     structure and structure is not something to render
 *   - the end-of-sequence token is reported before generation stops, so a
 *     listener can see a turn close rather than infer it from silence
 */
typedef struct {
    int id;
    int special;    /* 1 = a control token of the checkpoint's vocabulary */
    const char *utf8; /* "" when there is nothing to render */
    int nbytes;
} qe_token_out;

typedef int (*qe_token_cb)(void *user, const qe_token_out *t);

typedef struct {
    char id[128];
    char arch[64];
    char quant[160];    /* byte-weighted mix, e.g. "87.2 GiB: IQ4_NL 52%, ..." */
    char label[64];     /* general.name, e.g. "Qwen3.8 Flash Next" */
    char size_label[32];/* general.size_label, e.g. "512x56B" */
    /* why can_generate is 0, in words for the client. Roomy on purpose: the
     * sentence names every missing component and every implemented one, and a
     * snprintf that truncates it mid-word turns an honest refusal into something
     * that reads like a bug (it happened at 1024). */
    char why[1536];
    int is_checkpoint;  /* 1 = real GGUF metadata was read from disk */
    int can_generate;
    int n_volumes;
    long long n_tensors;
    long long bytes;
    int n_threads;
    int spec_k;
    /* Architecture facts as the checkpoint states them; 0 when it does not
     * say. These are what the forward pass would be written against. */
    int n_blocks;
    int n_embd;
    int n_experts;
    int n_experts_active;
    int n_ctx_train;
    /* Sequence capacity this engine will actually take: the cache, indexer keys
     * and recurrent state allocated at launch. A client planning a conversation
     * needs the window the server has, which is not necessarily the one the file
     * was trained for. 0 = none was allocated. */
    int n_ctx;
    int n_vocab;
    int eos_token_id;
    int image_token_id;
    int has_chat_template; /* tokenizer.chat_template present in the file */
    int has_mtp_head;      /* any tensor that could drive qe_spec_opts.spec_k */
    /* The draft head, when it is a file rather than a tensor in the checkpoint.
     * Reported with its path because a server that speculated from a head nobody
     * can point at is indistinguishable, from outside, from one that guessed. */
    char mtp_path[512];
    char mtp_stage[32];    /* where its bytes are, in the same words as `stage` */
    char mtp_quant[96];    /* byte-weighted mix of its own tensor types */
    long long mtp_tensors;
    long long mtp_bytes;   /* weight bytes the file holds */
    int mtp_layers;        /* <arch>.nextn_predict_layers, 0 when it does not say */
    int mtp_head_blocks;   /* decoder blocks it carries beyond the checkpoint's */
    /* 1 when the pass really opened that head and greedy requests draft from it,
     * 0 when it is resident bytes only and the n-gram drafter answers instead. A
     * head that was staged and never walked is a GiB of RAM an operator is paying
     * for without knowing, so the two states cannot share a field. */
    int mtp_drafting;
    int has_tokenizer;     /* real BPE tokenizer loaded; counts are exact */
    char tokenizer[224];   /* what was loaded, or why it could not be */
    /* Where the weight bytes live. On this host the difference between "mmap"
     * and "ram" is four orders of magnitude per expert read, so every response
     * says which one it is serving from. */
    char stage[32];
    long long stage_bytes;
    long long stage_cached;
    int stage_pinned; /* 0 = reclaimable, 1 = mlock, 2 = hugetlb (never pageable) */
    /* How the key and value cache holds its bytes, since that decides what the
     * answers are worth: "q8_0" is one byte per value with a scale per head, and
     * an accurate model of a quantized cache is a slightly different model. */
    char kv_cache[16];
    /* What the turn cache was given to hold rewind points, and what one point
     * costs. Reported rather than assumed by the reader, because the default sizes
     * itself against the memory still free once the weights are resident. Both are
     * 0 when there is no cache. */
    int turn_cache_mib;
    long turn_slot_bytes;
    /* Whether that budget was inferred at open or handed in, how many points the
     * inferred budget asks for, and - when there is no cache but there is a forward
     * pass to have one for - why not. "off" and "the host cannot afford a point"
     * are different facts and a client cannot tell them apart from a zero. All are
     * 0 or empty when the engine holds no sequence. */
    int turn_cache_auto;
    int turn_cache_points;
    char turn_cache_err[112];
} qe_engine_info;

typedef struct qe_engine qe_engine;

/* What a caller with no opinion about draft length asks for. The measured answer
 * depends on which drafter the engine ended up with, and docs/DESIGN.md 4 is the
 * measurement: one drafted token is what a resident nextn head is worth drafting, two
 * is what the n-gram hash is worth, and every longer draft costs the target more to
 * verify than the tokens it saves. 0 means off, and always did. */
#define QE_SPEC_K_AUTO (-1)
#define QE_SPEC_K_HEAD 1
#define QE_SPEC_K_NGRAM 2

typedef struct {
    const char *model_path; /* GGUF volume 0, or NULL for the selftest engine */
    const char *mmproj_path;
    /* A draft head delivered as a separate GGUF (one nextn block plus the norms
     * around it). Placed in memory at launch exactly like the checkpoint, and
     * refused at launch if it is a head for another model. */
    const char *mtp_path;
    const char *id;   /* model id reported to clients */
    int n_threads;    /* 0 = all cores */
    int spec_k;       /* default draft length, or QE_SPEC_K_AUTO for the drafter's */
    float temp;       /* default temperature */
    /* Defaults the server offers when a request does not say, in the same "0 means
     * the checkpoint's own value" convention as qe_gen_opts. This is what makes
     * `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0` mean the same thing here as
     * it does on the other server on this host. */
    int top_k;
    float top_p, min_p;
    int n_ctx;      /* sequence capacity a request may use: the KV cache, the
                     * indexer keys and the recurrent state are all sized by it.
                     * 0 = 4096. */
    int kv_f32;     /* nonzero: full-precision key and value cache instead of q8_0 */
    int attn_head_walk; /* nonzero: per-query-head attention walk instead of the
                         * one pass per KV group the engine runs by default */
    /* QE_STAGE_MMAP / _RAM / _HUGE (+QE_STAGE_STRICT, +QE_STAGE_DESCRIBE for a
     * metadata-only open). Only meaningful with a model_path; the selftest
     * engine has no bytes to place. */
    unsigned stage;
    /* What the turn cache may hold, in MiB. A rewind point is the recurrent state
     * of a sequence (118 MB for this checkpoint) plus 4 bytes per token it covers,
     * and it does not grow with the prompt, so the number of them a budget buys is
     * the number of turn boundaries the server can go back to. 0 lets the engine
     * decide at open, from the memory still free once the weights are resident, and
     * `qe_engine_info.turn_cache_mib` reports what it chose; negative turns the
     * cache off, which is the way to measure what it is worth. */
    int turn_cache_mib;
    /* The projection walk: the smallest tensor, in MiB, over which a chunked pass may
     * spread its tokens by blocking them through one sweep of the weights instead of
     * sweeping the tensor once per token. 0 takes the default this build ships with,
     * negative never blocks (the ordered walk, one sweep per token), and a positive
     * number is that many MiB. Overrides QE_PROJ_MB, which the forward pass reads at
     * open; docs/DESIGN.md §12 holds the measurement that decided the default. */
    int proj_chunk_mb;
} qe_engine_opts;

/* Opens the engine. `stage` says where the checkpoint's bytes live; the default
 * QE_STAGE_AUTO reads the whole checkpoint in at open and pins it, on STRICT
 * terms: an engine that cannot get every weight byte into memory fails here
 * instead of listening from behind a disk. err is filled on failure (and on a
 * warning worth printing, such as a hugetlb pool that was too short). */
qe_engine *qe_engine_open(const qe_engine_opts *o, char *err, size_t errn);
void qe_engine_close(qe_engine *e);
void qe_engine_describe(const qe_engine *e, qe_engine_info *out);
/* Reuse of the sequence the forward pass is holding: a request whose prompt is
 * that sequence plus new tokens feeds only what was added. On unless turned off,
 * and returns what it was. Turning it off discards what was being held. */
int qe_engine_prefix_cache(qe_engine *e, int on);
/* The checkpoint's tokenizer, or NULL when the file's tokenizer could not be
 * loaded (or this is the selftest engine, which has no vocabulary). Owned by
 * the engine; callers must not close it. Reported token counts use it when it
 * is present so usage.prompt_tokens is exact rather than estimated. */
qe_tok *qe_engine_tok(qe_engine *e);
/* Serialised: one decode at a time, because it is bandwidth-bound and the MoE
 * scratch is shared. Returns tokens emitted, or -1. */
long qe_engine_generate(qe_engine *e, const qe_prompt *p, const qe_gen_opts *o, qe_text_cb cb,
                        void *user, qe_gen_stats *st);
/* The same generation, reported token by token with its control tokens intact.
 * Returns tokens emitted, or -1. */
long qe_engine_generate_tokens(qe_engine *e, const qe_prompt *p, const qe_gen_opts *o,
                               qe_token_cb cb, void *user, qe_gen_stats *st);

/* Route the ingestion progress of the next generation on this engine to `cb`.
 * It is the forward hook above, reached through the engine because a server has
 * the client and not the forward pass; taking it back off with a NULL cb before
 * the call returns is what makes it safe to point at a connection. Only one
 * hook is live per engine, which is the same one-client-per-generation the
 * engine lock already implies. */
void qe_engine_set_progress(qe_engine *e, qe_progress_cb cb, void *user);
/* What the turn cache is doing, for anything that has to explain a fast second
 * turn without being able to see the terminal: how many rewind points the budget
 * buys, how many exist, what they cost in bytes, and how many prompt tokens this
 * engine has not had to read a second time. Any may be NULL; an engine with no
 * sequence to go back to - the selftest engine - reports zero slots. */
void qe_engine_turn_stats(qe_engine *e, long *slots, long *held, long long *bytes,
                          long long *cached_tokens);
/* The projection walk in force, in bytes: a tensor at or above this is blocked
 * through one sweep for a whole chunk rather than swept once per token, and 0 means
 * nothing is ever blocked. Reports what the forward pass is doing rather than what
 * was asked for, because the option and QE_PROJ_MB can disagree; an engine with no
 * forward pass reports 0. */
size_t qe_engine_proj_chunk(qe_engine *e);

/* ------------------------------------------------------------------ serving
 *
 * OpenAI-compatible endpoints: GET /v1/models, GET /health,
 * POST /v1/chat/completions, POST /v1/completions, with stream: true as SSE.
 */

/* Whether qe_serve should draw its live panel. See qe_serve_opts.ui. */
typedef enum {
    QE_UI_AUTO = 0, /* stdout is a terminal */
    QE_UI_ON = 1,
    QE_UI_OFF = -1
} qe_ui_mode;

typedef struct {
    int port;
    const char *bind_addr; /* NULL = every interface */
    int max_conns;
    int verbose;
    /* Milliseconds a streamed reply may go without a byte before an SSE comment is
     * written, so that a reader which went away during a minutes-long ingestion is
     * noticed then rather than after the whole turn was generated for nobody: an
     * 11,758-token prompt holds the engine 862 s before a delta exists, and nothing
     * else on the wire says anything during that time. This is not a way to satisfy
     * a client's stream idle deadline - a client that credits a deadline per parsed
     * part credits nothing here, because a comment parses to no part; see the note
     * at sink_progress. 0 means the default (15 s); negative turns the comments off. */
    int sse_keepalive_ms;
    /* Replay an assistant turn that carried no reasoning text as an empty reasoning
     * block, which is the spelling the chat template uses. Off by default: an empty
     * block is not a neutral placeholder but the shape this model reads as "this
     * assistant thought of nothing", and a client that keeps thinking for itself and
     * never sends it back (crush does) would then replay a transcript in which the
     * assistant never thinks. On puts the template-faithful empty block back, which
     * is what to compare against when measuring whether thinking got shorter. */
    int empty_thinking_blocks;
    /* How long a SIGTERM waits for a request that is still running. The listener
     * closes immediately either way; this is the wait for the generation already
     * holding the engine, which `stopping` cuts short at the next pass, so the
     * number here is mostly a patience allowance for a keep-alive client sitting
     * between two requests. When it runs out the process exits without freeing the
     * engine, because freeing it under a thread that is still generating is not a
     * thing a shutdown path should do, and weights are the kernel's to reclaim
     * anyway. 0 means the default (5 s). */
    int shutdown_grace_secs;
    /* The live panel: tokens going through the network and tokens coming out,
     * redrawn several times a second on stdout. QE_UI_AUTO (the zero value) shows it
     * only when stdout is a terminal, which is also when the per-request log line
     * moves inside the panel rather than going to stderr beside it: two writers on
     * one screen make a scramble and not a log. */
    qe_ui_mode ui;
    qe_engine_opts engine;
} qe_serve_opts;

void qe_serve_opts_default(qe_serve_opts *o);
/* Blocks until the listener fails. A request still running when the listener
 * closed gets shutdown_grace_secs to finish; one that outlives that exits the
 * process from inside this function, because the engine it is holding cannot be
 * freed and the wait for it cannot be allowed to outlast the stop. Returns
 * non-zero when the listener fails. */
int qe_serve(const qe_serve_opts *o);

/* ---------------------------------------------------------------- threads */

/* qe_task_fn is declared with the other core types, at the top of this header. */

/* Thread count meaning "however many this machine should serve with", which is
 * NOT "every cpu I am allowed to run on". A dispatch returns when its last
 * thread returns, and the workers spin rather than sleep, so a pool as wide as
 * the cpu count has no core left for anything else on the host: one competing
 * runnable task preempts a spinning worker and the barrier waits for a scheduler
 * timeslice. Measured on this host (20 cpus) with `make pool`, back-to-back
 * trivial dispatches:
 *
 *     threads 20   median 69 us, worst 2487 us   (one run: median 2868 us)
 *     threads 19   median 1.6 us, worst  2.0 us
 *     threads 18   median 1.5 us, worst  1.9 us
 *
 * Forty to two thousand times the cost of the same handshake, for the last
 * core. With the ~800 dispatches a token makes that is tens of milliseconds a
 * step, and it is: the same binary, corpus and weights score identically at 20 and
 * 19 threads (nll 2.2425, 32/63 argmax hits) and take 125.2 ms and 106.2 ms a
 * token; 18 threads is 106.6 and 16 is 107.6, so the plateau below the cliff is
 * flat and one core is the whole reservation. Pass QE_THREADS_AUTO to keep the
 * engine's choice; an explicit count is honoured exactly, so `--threads 20` still
 * means 20, and the benches take `--threads auto` for the same reason.
 *
 * The pool names no cpus, which is the other half of that measurement: the
 * scheduler places the workers, and confining them is worse. QE_PIN=1 builds the
 * cpu list from `cpu_capacity`, highest first, and pins every worker to one entry
 * of it - the only configuration in which "the core the engine left out is a slow
 * one" describes the engine at all. It is off because it is slower: hot-dispatch
 * median 2.1 and 2.9 us unpinned against 5.7 and 4.8 pinned, and 4.9 against 6.2 at
 * the full mask (two runs each, `QE_PIN=1 ./bin/bench_pool auto 120`);
 * docs/DESIGN.md §12 is the write-up.
 */
#define QE_THREADS_AUTO (-1)

int qe_pool_init(int nthreads, const int *cpus, int ncpus);
void qe_pool_run(qe_task_fn fn, void *arg, int nthreads);
void qe_pool_shutdown(void);
int qe_pool_size(void);
/* CPUs this process may run on (affinity mask, so a cpuset-limited host is
 * measured honestly), or the online count if the mask cannot be read. */
int qe_cpus(void);
/* What qe_pool_init(QE_THREADS_AUTO) would pick: qe_cpus() - 1, never below 1,
 * and no reservation at all on a machine too small for one to matter. */
int qe_pool_recommended(void);

/* A stage graph run as one dispatch.
 *
 * qe_pool_run is one task sliced over the pool, and a layer is made of a good
 * many of them: a chunk's expert stage alone is seven dispatches, each ending in
 * a hand-back that charges the slowest thread of the stage that just finished to
 * every thread of the one about to start. An item is a task that says what it
 * depends on. One dispatch of a set of items runs anything whose predecessors
 * have finished, on whatever threads are free, so the hand-backs between stages
 * that never needed ordering go away, and a stage with less work than threads
 * lends the rest of them to a stage that has more instead of parking them.
 *
 * An item runs each of its slots 0..nthreads-1 exactly once, as fn(arg, slot,
 * nthreads) - the contract of qe_pool_run, so a task that keys its scratch by
 * tid needs no change. `deps` is a mask of items all of whose slots have to have
 * run before any slot of this one may; a stage independent of everything has
 * deps 0 and starts immediately. Item order is a preference and nothing more: a
 * worker takes the lowest-numbered ready slot it can see, which keeps the wide or
 * the slow stage on the most threads without any of them waiting for a turn.
 *
 * Two rules the caller owns. The items and their args live in the caller, so they
 * have to outlive the call, exactly as with qe_pool_run. And a cycle in `deps`
 * never finishes: nothing here detects one, so a caller that builds a graph built
 * an acyclic one. nthreads is clamped to the pool, as before. */
void qe_pool_run_items(const qe_pool_item *items, int nitems);

/* Split [0, n) into one contiguous range of equal size for each thread.
 *
 * Equal, deliberately. A dispatch ends when its last thread ends, so on a part
 * whose cores differ in speed - this host has P-cores whose cpu_capacity is 1.30x
 * the E-cores' - charging rows by that number looks free and costs the same 30%
 * on any dispatch big enough to saturate the memory fabric, which every weight
 * read in a decode step is. `bin/bench_moe`'s dispatch-width table is the
 * measurement, and it ends with the even split at the streaming ceiling of
 * bench_kern and the weighted one 12-15% under it for both expert types. The
 * summary line `qe_slice_summary` prints is what the startup banner reports, so
 * the policy in force is visible in a log.
 *
 * QE_SLICE=capacity restores the cpu_capacity weighting, which is the right ruler
 * only where arithmetic is the limit: a pool narrower than the machine's memory
 * ceiling, or a kernel whose cost is ALU rather than bytes. Ranges are
 * non-overlapping, exhaustive, identical for every thread that asks about a given
 * (n, nthreads), and fall back to equal when no table exists. */
void qe_slice(long n, int tid, int nthreads, long *lo, long *hi);
/* One-line description of the table, for the startup banner. Returns BUF. */
const char *qe_slice_summary(char *buf, size_t buflen);

/* ------------------------------------------------------------------ memory */

void *qe_alloc_huge(size_t n);
void qe_free_huge(void *p, size_t n);
/* Backing for a large private arena that holds model bytes. QE_STAGE_HUGE asks
 * for a hugetlb pool, QE_STAGE_RAM for ordinary anonymous pages, and
 * QE_STAGE_THP layers 2 MiB transparent huge pages on top of whichever backing
 * was used. *how names what was actually obtained, which is not necessarily
 * what was asked for. NULL on failure. */
void *qe_alloc_backed(size_t n, unsigned flags, const char **how);
/* Map a model-sized arena out of a file in DIR: hugetlbfs when DIR is a
 * hugetlbfs mount, tmpfs when it is a tmpfs one. Either way the bytes outlive
 * the process, so a restart inherits the checkpoint instead of reading it
 * again. Creates the file at n bytes when absent; an existing file of a
 * different size is a leftover from another checkpoint and is replaced. The
 * file is deliberately not unlinked, and *fresh says whether the bytes in it
 * are new (and therefore still need to be read) or inherited. */
void *qe_alloc_arena_file(const char *dir, const char *name, size_t n, int *fresh);

/* Pin against reclaim. 0 when the range is really locked (memlock is unlimited
 * here), negative errno otherwise - a caller that prints a residency claim must
 * call this rather than assume. */
int qe_lock_resident(void *p, size_t n);
size_t qe_mem_available_mib(void);
size_t qe_huge_free_mib(size_t page_mib);
void qe_madvise_seq(const void *p, size_t n);
void qe_prefault(void *p, size_t n, int nthreads);
long qe_page_size(void);

/* Hard fuse. Arms a wall-clock deadline; when it fires the process writes one
 * line to stderr and _exit()s with status 124. Any binary that touches the
 * 87 GiB checkpoint must arm this first: an unbounded run on a spinning disk
 * is indistinguishable from a hang, and a hang wastes more time than a
 * deliberately failed measurement. Pass 0 to disarm. */
void qe_fuse(int seconds);

static inline size_t qe_align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

#ifdef __cplusplus
}
#endif
#endif /* QE_H */
