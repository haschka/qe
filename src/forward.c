/* The forward pass: one sequence, a token or a chunk of them at a time, exactly
 * as docs/ARCH.md §6 specifies it and as the upstream model definition computes
 * it.
 *
 * Two things made this file worth writing rather than sketching.
 *
 * The first is that the interesting bugs are not in the matrix products. Every
 * GEMV here is one of the measured kernels from src/kern*.c, so the risk is
 * entirely in what happens *between* them: whether a norm reduces over a stream
 * or over the whole hyper-connection width, whether a rotary pair is adjacent
 * or split-half, whether epsilon is added under a root or floors a norm, whether
 * a value head reads key head h/3 or h % 16, whether the gated delta rule's
 * output carries a 1/sqrt(state_size) that the diagram does not show. Each of
 * those is a model that runs at full speed, produces no NaN, and is wrong. They
 * are answered here from the reference operators rather than from the diagram,
 * and the answers are written down at the call site because the next person to
 * read this will otherwise ask the same question.
 *
 * The second is that "correct" here has a cheap first signal and an expensive
 * final one. Cheap: no non-finite value anywhere, and per-stage magnitudes that
 * stay in the band a trained network keeps its activations in. Expensive: the
 * perplexity of held-out text, which is the only test that catches a component
 * that is self-consistently wrong. Both are wired to `make forward`.
 *
 * Ingestion is chunked: a prompt goes eight tokens at a time, one stage of a
 * layer at a time, so that a stage is entered once for the chunk instead of once
 * per token, and the T calls one tensor needs are issued back to back, so that
 * the second row onward is served by the 30 MiB L3 rather than by the memory
 * controller. A 1005 token prompt ingests at 65 ms/token that way against 95 for
 * one token at a time (`make forward`).
 *
 * None of it is a second arithmetic: chunk_impl with T == 1 *is* the decode step,
 * on row 0 of the same buffers, and a row of a chunked projection is the same
 * kernel on the same inputs as the call it replaces. `make forward` asserts the
 * chunked trunk against the one-token trunk bit for bit, and QE_CHUNK=1 is the
 * switch that A/Bs the two. That identity holds as served, with the blocked walk
 * off; the blocked walk is the one place a chunk can reassociate, which is what
 * the two paragraphs below are about.
 *
 * What is *not* on is the deeper form of the same idea, qe_gemv_multi_t - one
 * weight walk blocked against T activations inside one core's L2
 * (docs/DESIGN.md §12) - which is wired behind QE_PROJ_MB and defaults off. It
 * has been measured here, and it did not pay: the same prompt ingests at
 * 65 ms/token with it off, 67 blocking every projection above 4 MiB, 66 blocking
 * everything down to 1 MiB, same trunk bits in all three. Blocking buys bytes
 * never read twice at the price of the DRAM efficiency that reads them fast, and
 * the back-to-back ordering above has already stopped them being read twice,
 * because the L3 is bigger than any single projection in this model.
 *
 * Those three numbers were taken while the blocked walk called the single-token
 * kernel once per token, and it no longer does that for Q8_0: it runs
 * qe_gemv_q8_0_multi, which dequantises a block once for the whole T and is worth
 * 1.98x at T=2 and 3.75x at T=4 in bench_alu. The default is therefore a
 * measurement of a kernel that no longer exists, and a kernel ratio is not an
 * ingestion ratio: what moves it is the line `make forward` prints, which ingests
 * one slice of the prompt both ways inside the load it already needed. It stays
 * wired for the case where there is nothing else to share a walk with: a matrix far
 * larger than the L3, which is what a batched decode's 497 MiB head is. The experts
 * are per token for the same reason in reverse - a chunk shares only the experts its
 * tokens happen to pick in common (§9: 54% fewer expert bytes at T=16) - and the
 * PLE gather is per token because its rows are.
 */

#include "qe.h"
#include "kern.h"

#include <errno.h>
#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QE_PLE_MAX_HEADS 32

/* How many history rows the GDN conv core holds in registers-to-pointer form. The
 * checkpoint's kernel is 4, so three taps; the array is fixed because the tap walk is
 * the innermost loop of the conv and an indirection through a runtime-length one
 * would put the ring's modulo back. */
#define QE_GDN_TAP_MAX 8

/* QE_CHUNK, clamped to QE_CHUNK_MAX: the width a pass opens with. Defined with the
 * chunk loop it configures; declared here because qe_forward_open reads it once
 * before any chunk runs. */
/* Tokens of a prompt chunk whose attention partials are held at once (qsa_core_chunk). */
#define ATTN_GROUP 32
#define QE_MAX_HEAD_DIM 512

static int prefill_chunk(void);

/* QE_MOE_GROUP: whether a chunked prefill groups the expert stage. Declared here
 * for the same reason as prefill_chunk. */
static int moe_group_env(void);

/* QE_MOE_TILE: how many rows of a chunk one grouped expert pass covers. Declared
 * here for the same reason as the two above. */
static int moe_tile_env(void);

/* QE_MOE_CO: whether the expert stage of a chunk is one stage graph. Declared here
 * for the same reason as the three above. */
static int moe_co_env(void);

/* QE_HEAD_BATCH: whether a window's rows are given their head in one blocked walk.
 * Declared here for the same reason as the four above. */
static int head_batch_env(void);

/* QE_MTP_WARM: whether a prompt also feeds the draft head, so that a draft row's
 * attention sees the sequence it was trained on. Declared here because
 * qe_forward_open reads it once, before any prompt runs. */
static int mtp_warm_env(void);

/* QE_MTP_BACK: how far the residual a draft row reads lags the token it is fed.
 * Declared here for the same reason: it decides what a warm row is, and a warm row
 * is written once per prompt token. */
static int mtp_back_env(void);

/* QE_CHUNK_MAX and QE_CHUNK_DEFAULT, in include/qe.h: the first sizes every chunk
 * buffer below and bounds what qe_forward_set_chunk accepts, the second is the width
 * a pass opens with. */

/* A row block for the chunked projections: half a 2 MiB L2, so that a block of
 * weight rows and the T activation rows it multiplies both stay in the core that
 * read them. Same number `make prefill` tunes with. */
#define QE_ROW_BLOCK (1u << 20)

struct qe_forward;

/* scratch slots, indexed rather than named so that the allocation list is one
 * loop and the free list is one loop, and neither can forget a buffer. */
enum {
    E_cres, E_cxn, E_hn, E_gate, E_lo, E_cinj, E_cy, E_sh,
    E_qkv, E_z, E_conv, E_beta, E_alpha, E_gexp, E_att, E_attnorm,
    E_qg, E_qwork, E_qgate, E_kcur, E_vcur, E_attny, E_scores,
    E_iq, E_ikraw, E_ipool, E_iscore,
    E_pkey, E_pval, E_pgated, E_pgnorm, E_pconv, E_pemb,
    E_e640, E_u640, E_h640, E_mlogits, E_cmlogits, E_ey,
    E_cs, E_sel, E_selbits, E_attnpy, E_attnpm, E_attnpl,
    E_routes, E_seq, E_heap, E_heapb, E_i8x, E_i8s, E_i8p,
    E_cqwork, E_csel, E_tiscore, E_theap, E_theapb, E_tbits, E_gpy, E_gpm, E_gpl, E_rpos,
    E_COUNT
};

typedef struct {
    const qe_tensor *qkv, *zgate; /* GDN: [q|k|v] and the output gate z */
    const qe_tensor *conv, *ssm_out, *ssm_alpha, *ssm_beta, *ssm_a, *ssm_dt, *ssm_norm;
    const qe_tensor *aq, *ak, *av, *aout, *aqn, *akn; /* QSA */
    const qe_tensor *ixq, *ixk, *ixqn, *ixkn;         /* indexer */
    const qe_tensor *hc_norm[2], *hc_down[2], *hc_up[2], *hc_inject[2];
    const qe_tensor *router, *ex_g, *ex_u, *ex_d;
    const qe_tensor *sh_gate, *sh_up, *sh_down, *sh_inp;
    const qe_tensor *ple_key, *ple_value, *ple_nk, *ple_nq, *ple_nc, *ple_conv;
    int qsa;  /* 1 = full attention with the indexer, 0 = gated delta net */
    int ple;
    int gdn_slot, qsa_slot;
} qe_layer;

struct qe_forward {
    qe_pool_stats pst0; /* the pool counters when the stage being timed began */
    qe_model *m;
    qe_forward_opts o;
    int n_threads;
    int walk_threads; /* what `gemv` asks the pool for; see qe_forward_opts.walk_threads */
    int small_proj_batch; /* a projection under the chunk walk's threshold is asked of the
                           * pool once for the chunk instead of once a token; see
                           * QE_SMALL_PROJ_BATCH and `proj_small` */
    int exp2_simd;        /* the gated-delta decay and one cell's attention weights use
                           * the eight-wide exp2 of src/layer.c's gates instead of libm
                           * `expf`; see QE_EXP2_SIMD */
    int n_layer, K, HC, hc_dim;
    int n_layer_all; /* n_layer, plus the draft head's one block when a head is open */
    int n_q_heads, n_kv_heads, head_dim, n_rope;
    int n_idx_heads, idx_dim, kpool_ratio, idx_topk_blocks;
    int n_gdn_heads, n_gdn_kheads, head_state, gdn_channels;
    int n_experts, n_experts_active, n_ff_exp, n_ff_shexp;
    int low_rank;    /* hyper_connection.low_rank: the mixer's bottleneck width */
    int gdn_kernel;  /* ssm.conv_kernel, dilation 1 */
    int ple_kernel;  /* ple.conv_kernel, dilation 3 -> ple_lag history rows */
    int ple_lag, ple_dilation;
    float eps, theta, gdn_scale, attn_scale, idx_scale;
    long long ple_eos;

    qe_layer *L;
    qe_moe **moe;
    void **moe_shadow;
    const qe_tensor *embd, *outw, *head_norm, *head_down, *head_up;
    qe_ple *ple;

    /* The draft head: one more decoder block, in a file of its own, opened as layer
     * `n_layer` of this pass. Everything below belongs to it (docs/DESIGN.md §4).
     *
     * The fold in front of the block reads a pair - the embedding of the token at
     * position p and the four-stream residual of position p-1 - normalises each half,
     * and projects their concatenation, which is what makes a chain of draft rows
     * close: a row's exported residual is the h of the token that row predicted.
     * The head's embedding half therefore comes from `embd`, the same table the trunk
     * reads, and not from the head's own re-quantised copy of it: same rows, fewer
     * bits lost, and no second walk of a 340 MiB table to fetch a row this pass has
     * already fetched. */
    qe_model *mh;
    int mtp_ready, mtp_warm;
    const qe_tensor *mtp_eh, *mtp_enorm, *mtp_hnorm;
    const qe_tensor *mtp_hc_norm, *mtp_hc_down, *mtp_hc_up, *mtp_outw;
    /* [e_norm | h_norm_s] per stream, so that one projection walks the fold for the
     * whole chunk: row t of stream s starts at mcat + (t*HC + s)*2K. `mtp_eh` is one
     * matrix read HC times - the fold broadcasts the projection over the stream axis
     * (ref/qwen4exp.cpp:634-636), so the streams differ in which 2K-wide half of
     * `mcat` they read and not in which rows of the file they walk. */
    float *mcat;
    float *mhin; /* T rows of hc_dim: the residual the head's block runs on */
    float *mout; /* T rows of K: the fold's embedding half, before either half is normed */
    /* The residual of the last row of the chunk before this one. A cell is fed the
     * token that follows its residual, so the last row of a chunk has to wait for the
     * first token of the next one; this is the row that waits, and the position its
     * token belongs to (-1 when nothing is waiting). */
    float *mpend;
    long mpend_pos;
    /* Where the chunk in flight started. A draft row is asked for by the position of
     * the token it continues, and the only handle on a residual is a row of the pass
     * that computed it (qe_forward_mtp_hidden). */
    long pass_pos0;
    /* How far the residual a draft row reads lags the token it is fed. The head files
     * train the fold on the embedding of one position and the residual of the one
     * before it, which is what makes a chain close; `make forward` can say so with 0
     * and read what that pairing costs in accepted drafts. */
    int mtp_back;

    /* sequence state */
    float *gdn;     /* gdn_slots x 48 x 128 x 128, row-major: S[j][i] with the
                     * delta rule contracting over i for both the read of k and
                     * the read of q */
    float *gconv;   /* gdn_slots x 3 x 10240, oldest tap first */
    /* The keys and values of the full-attention layers, and this is the one part
     * of a sequence that grows with the sequence. In f32 it is 4 KiB per cell per
     * layer (2 KV heads x 256 dimensions x K and V x 4 bytes), which at the 262144
     * cells the checkpoint was trained on is 12 GiB of the 87.2 GiB the weights
     * already are not. q8_0 is 4 KiB / 3.94 = 1.04 KiB: one byte per value and one
     * scale per head of one cell, which is the granularity that keeps a key's
     * direction because the head is RMS normalised before it is written.
     *
     * `kv` is the f32 form and is only allocated when kv_q8 is 0; `kvd` and `kvs`
     * are the quantised form, a row of values and the scale of that row. */
    float *kv;      /* qsa_slots x 2 x n_ctx x (n_kv_heads * head_dim) */
    int8_t *kvd;    /* the same shape, one byte per value  */
    float *kvs;     /* qsa_slots x 2 x n_ctx x n_kv_heads, one scale per row */
    int kv_q8;
    float *kraw;    /* qsa_slots x n_ctx x 128, pre-norm indexer keys */
    float *kpool;   /* qsa_slots x ceil(n_ctx/4) x 128, normed and rotated */
    int idx_f16;    /* QE_IDX_F16: the pooled keys stored as halves (ideas.md §7.37) */
    float *ple_hist;
    uint32_t *seq;  /* the token ids fed so far, PLE-visible (image positions
                     * already substituted) */
    /* The rope position of every cell, as the (t, h, w) the checkpoint's interleaved
     * m-rope reads (rope.dimension_sections, ref/qwen4exp.cpp: LLAMA_ROPE_TYPE_IMROPE).
     * For text the three are one number, and until a sequence holds an image that
     * number is the cell, so the table is written and never read: `mrope[s]` is what
     * turns it on, and every angle a text-only sequence takes is the arithmetic it
     * always took. An image row's three are its own (llama.cpp's mtmd decoder
     * positions: t at the image's start, h and w that plus its row and column in the
     * merged grid), and a text row's is one past the largest of the row before it -
     * which puts the text after an image max(nx, ny) on from where the image began. */
    int32_t *rpos;  /* n_ctx x 3 */
    int mrope[QE_MAX_SEQ];
    int msec[4];    /* the sections, in rope pairs */
    /* The rows of the prefill in flight that are not token embeddings, as
     * qe_forward_prefill_mm hands them over: one pointer per row (NULL for a token)
     * and that row's (t, h, w). Set for exactly one chunk at a time and NULL between
     * calls, which is what keeps a step from ever reading them. */
    const float *const *mm_emb;
    const int32_t *mm_pos;
    long pos;    int gdn_slots, qsa_slots;
    size_t kv_row, n_ctx;
    /* The sequences one step may carry (qe_forward_opts.n_seq). One is everything
     * this file was before the field existed, and it stays exactly those bytes:
     * `seq_stride` is then `n_ctx`, sub-slot `slot * 1 + 0` is `slot`, and every
     * position below is `pos`, which is why no layer function has an "or else" in
     * it. See qe_forward_step_batch.
     *
     * `spos` is each sequence's own position, and the only thing a row needs along
     * with it: a rope angle, an indexer block and a conv-ring head are all functions
     * of that, and each sequence is numbered from its own 0. Cells are addressed
     * relative to the sub-slot, which is why the selection, the block scores and the
     * top-k never hear about any of this. */
    int n_seq;
    size_t seq_stride; /* cells one sequence owns: n_ctx / n_seq */
    long spos[QE_MAX_SEQ];

    int chunk_t;      /* row this call is computing            */
    int chunk_T;      /* rows in flight; 1 for an ordinary step */
    /* A window's rows read out of the head in one walk: the logits of rows
     * [0, lg_rows) of the chunk in flight, so that a caller that takes them one at a
     * time - which is what verifying a draft does, because it stops at the first row
     * whose token was not the one it proposed - walks the 497 MiB head once rather
     * than once per row. Owned by the pass, because the rows it holds are the rows
     * that pass made: every chunk clears it. */
    float *lg_cache;
    int lg_cap;   /* rows the cache has room for */
    int lg_rows;  /* rows of it that are logits of the chunk in flight; 0 when none */
    /* Set for the duration of a checkpointed window, and NULL every other
     * microsecond of a pass's life: the block core asks for it once per row, and
     * a window is the only thing that ever wants a row's state kept. */
    qe_ckpt *ck;
    /* Multiply-accumulates counted where a walk is issued, and what the last stage
     * boundary saw of it. A stage's rate is then `d.stage_macs` over its seconds, and
     * that pair is the only way to say whether a dense stage walks at the rate its
     * kernel gets on its own: the alternative is counting rows out of a tensor table by
     * hand and hoping the subset you chose is the subset the stage timed (ideas.md 7.8).
     * Counted only while profiling, and it is one add per projection call either way. */
    double walk_mac, mac_mark;
    /* The size a projection has to reach for the chunk to be worth one weight walk
     * instead of T of them. A field and not only the environment because one load
     * of an 87 GiB file cannot be asked twice: `make forward` measures both forms
     * in the pass it already has open. 0 is never, which is the default. */
    size_t proj_chunk_bytes;
    /* Whether a chunk's projections recode the chunk's activations to int8 once and
     * walk the weights with vpdpbusd/vpdpbssd instead of dequantising per 32 weights
     * into floats. A field for the same reason as the one above, and it is the only
     * way the integer walks reach the dense matrices: `make forward` A/Bs the two
     * arithmetic in one load. Off by default until `make score --ingest` has priced
     * what the recode costs on real text; the expert stage's answer to that question
     * (src/moe.c) is why this is a switch and not a rewrite.
     *
     * The panel is one buffer per pass, not per thread: it depends only on the
     * activation being multiplied, so every projection that reads that activation
     * walks the same bytes, and the recode is a fill the pool does once between the
     * two dispatches rather than K values per thread per row-slice. */
    int proj_i8;
    int8_t *i8x; /* QE_CHUNK_MAX rows of the widest walkable matrix */
    float *i8s;
    int32_t *i8pseed;
    long i8k; /* K the panel above is sized for, in activations */
    /* A second copy of every dense matrix the chunk walk reads, requantised to one
     * scale for a whole output row instead of one for every 32 values. That is what
     * leaves a Q8_0 walk with no scale exit at all: the accumulator stays in int32 for
     * the row, becomes a float once, and pays the row's scale and the token's - and the
     * register the block scale held pays for a third output row sharing one activation
     * load (`qe_gemv_q8_0_flat_i8`, ideas.md 7.18, 1.84x of the walk that ships).
     *
     * The copies are staged at open, in full, out of the bytes that are already
     * resident, and never on first use: a token may not be the first thing to read a
     * weight byte (AGENTS.md 1). They cost 3.3 GiB on this checkpoint, which is the
     * Q8_0 matrices less the routed experts and the embedding table, neither of which
     * any caller of this walk reads. On unless QE_I8_FLAT=0 says otherwise, and an open
     * that cannot afford them says so with the two numbers that decided it rather than
     * quietly walking a different arithmetic than it was asked to. */
    int flat_i8;
    void *flat_w;    /* every copy, one arena, a tensor's rows K bytes apart */
    float *flat_s;   /* one scale per output row of every copy above */
    long *flat_woff; /* byte offset of a tensor's first row, by tensor index, -1 for none */
    long *flat_soff; /* offset of that tensor's first scale, by tensor index */
    long flat_n;     /* entries in the two maps above */
    size_t flat_bytes; /* what the copies hold, for the line that says so */
    /* A second copy of the same dense matrices, this one in IQ4_NL, for the other end
     * of the width range. A lone token walks a matrix once, so its time is its bytes
     * over the bus, and 4.5 bits a weight against 8.5 is close to two to one there
     * (ideas.md 4.1 measures 1.74x for this swap at T=1) where the same swap is a wash
     * at the width a chunk ingests at. It is therefore read by a step and never by a
     * chunk: `chunk_T == 1` is the whole test, which is what leaves an ingested token
     * walking the bytes it walked before, arithmetically unchanged. Staged at open like
     * the flat copies, refused with two numbers on a host that cannot afford them, and
     * 1.8 GiB on this checkpoint. */
    int dq_i4;
    int dq_multi;   /* widest pass (rows) that reads the IQ4_NL dense copy: QE_I4_MULTI */
    int stage_pf;   /* QE_STAGE_PF: idle threads pull the next weights toward the cache */
    int step_fuse;  /* QE_STEP_FUSE: a step's independent walks share one dispatch */
    size_t dyn_bytes; /* QE_DYN_KB: a lone token's walk is handed out in chunks this big */
    int step_i8;      /* QE_I8_STEP: a lone token's walk spends int8 against the codebook */
    int chunk_dyn;    /* QE_CHUNK_DYN: a multi-row walk hands its rows out from a cursor too */
    int delta_simd;   /* QE_DELTA_SIMD: the delta rule eight rows at a time (§7.34) */
    int rows_par;     /* QE_ROWS_PAR: a chunk's per-token loops across the pool (§7.33) */
    int attn_batch;   /* QE_ATTN_BATCH: a prompt chunk's attention as a few dispatches, not 3 a token */
    float *cqwork;    /* QE_CHUNK_MAX rows of normed, rotated queries */
    int *csel;        /* QE_CHUNK_MAX selections of up to sel_cap cells */
    long sel_cap;
    float *tiscore;   /* per pool thread: one row of block scores */
    float *theap;     /* per pool thread: a top-k heap */
    int *theapb;
    uint64_t *tbits;  /* per pool thread: the selection bitmap */
    long tbits_words;
    float *gpy, *gpm, *gpl; /* partials of one group of a chunk's attention */
#define QE_STEP_PANELS 8
    qe_i8panel spanel[QE_STEP_PANELS];
    const float *spanel_x[QE_STEP_PANELS];
    long spanel_k[QE_STEP_PANELS];
    void *dq_w;    /* every copy, one arena, a tensor's rows K/32 blocks apart */
    long *dq_woff; /* byte offset of a tensor's first row, by tensor index, -1 for none */
    /* Which classes of matrix this is, by tensor index: the hyper-connection mixers, the
     * first and last blocks of the trunk, or neither. A copy is staged for every
     * candidate and read only by a tensor whose classes the policy below allows, so the
     * price of a policy is a walk and not another load. */
    unsigned char *dq_wcls;
    int dq_tier; /* which classes may read a copy: QE_DCLASS_* or'd */
    long dq_n;     /* entries in the map above */
    size_t dq_bytes; /* what the copies hold, for the line that says so */
    /* The output head is the one matrix in the file that is neither Q8_0 nor a codebook
     * expert - it is Q6_K, and it reads at 45 GB/s against the 76-79 the bus does, which
     * is the only stage of a step measured slow at a rate no other format in the file
     * shares (ideas.md 21). Two copies of it are staged, because the two ways out point
     * in opposite directions: IQ4_NL buys bytes and gives up resolution on the matrix
     * whose every row is the answer, Q8_0 gives up bytes and buys the ALU the 6-bit
     * assembly costs, and which of those is worth it is a question about this
     * checkpoint's loss and not about two sizes. `head_copy` says which one a step reads.
     * A Q8_0 walk of a Q8_0 copy of a Q6_K matrix is a re-rounding at 8.5 bits of
     * something written at 6.5625, so it is the arm that answers the question without
     * asking one. */
    void *h4_w;  /* output.weight as IQ4_NL */
    void *hq_w;  /* output.weight as Q8_0 */
    /* The draft head has its own `output.weight`, and a draft row walks it the way a
     * step walks the trunk's, so the same two copies are staged for it - unless the two
     * tensors are the same bytes, which is what a nextn head trained against its target
     * usually holds, and then one pair of arenas serves both. */
    void *mh4_w; /* the draft head's output.weight as IQ4_NL */
    void *mhq_w; /* the draft head's output.weight as Q8_0 */
    int head_copy; /* QE_HEAD_STORED / _I4 / _Q8, as staged */
    /* Whether a walk of more than one row of a head - a verify window - may read the copy
     * too. A lone row is where the copy was priced; several rows read the head once and
     * multiply it T times, which is the regime where the 30 % more bytes of a Q8_0 copy of
     * a Q6_K matrix start to count against the ALU it saves. `make head`'s own arms are the
     * instrument, and QE_I4_HEAD_WIN=1 asks for it. */
    int head_win;
    /* How many prompt tokens share one walk of the weights. A field for the same
     * reason as the one above, and read once per prefill rather than once per
     * process, so that `make forward` can sweep the width inside one load.
     * Bounded by QE_CHUNK_MAX because the chunk buffers are sized by it. */
    int chunk_width;

    /* Non-zero runs a chunk's routed experts as one grouped pass instead of T
     * separate ones, and its shared expert through the projection walk. Grouping
     * is what lets one read of an expert matrix - and one dequantisation of its
     * rows - serve every token of the chunk that chose it. It is a field for the
     * same reason as the two above: `make forward` measures it against T calls in
     * the pass it already has open. */
    int moe_grouped;

    /* How many rows of a chunk one grouped expert pass covers; 0 is the whole chunk.
     * A field for the same reason as the one above: the cost of a pass is a function of
     * how full its buckets are, and only one load can say where that curve turns. */
    int moe_tile;

    /* One walk of a KV group's rows serving all twelve of its query heads, or one
     * walk per query head. A field for the same reason as the three above: the two
     * differ only in the order the same sums are formed, and one load of an 87 GiB
     * file is what decides that. */
    int attn_fused;

    /* A chunk's expert stage as one dispatch of a stage graph, with the shared
     * expert's three projections running beside the grouped expert walk instead of
     * in six dispatches after it. A field for the same reason as the four above,
     * and it is the one of them that changes no arithmetic at all: same kernels,
     * same rows, same thread counts per task, so the trunk is the same bytes and
     * only the schedule differs. */
    int moe_co;

    /* Whether a window's rows are read out of the head together (one blocked walk
     * of the output matrix) or one at a time (`rows` walks of the same 497 MiB).
     * Same numbers either way - see qe_forward_window_logits - so this is a switch
     * only because `make forward` prices the two on one load. */
    int head_batch;

    /* scratch */
    void *sp[E_COUNT];
    /* The chunk in flight: QE_CHUNK_MAX rows of the three buffers one stage
     * leaves for the next. res, xn, y and inj are cursors into these, moved by
     * use_row(); everything else is a per-call temporary. */
    float *cres, *cxn, *cy, *cinj;
    float *res, *xn, *hn, *gate, *lo, *inj, *y, *sh_out;
    float *qkv, *z, *conv_out, *beta, *alpha, *gexp, *att, *att_norm;
    float *qg, *qwork, *qgate, *kcur, *vcur, *attn_y, *scores;
    float *iq, *ikraw, *ipool, *iscore;
    float *pkey, *pval, *pgated, *pgnorm, *pconv, *pemb;
    float *e640, *u640, *h640, *moe_logit, *cmoe_logit, *expert_y;
    float *cs;
    int *sel;
    uint64_t *sel_bits; /* the winning blocks, one bit each, walked ascending */
    /* The fused attention's partials: one unnormalised accumulator, maximum and
     * sum per unit, where a unit is one KV group's chunk of the selection. Sized
     * for the widest split the dispatch ever asks for. */
    float *attn_py, *attn_pm, *attn_pl;
    float *heap_s;   /* the indexer's bounded top-k, a min-heap of winners */
    int *heap_b;
    qe_route *routes;
    /* The chunk's routes, T rows of n_experts_active of them: the grouped expert
     * pass reads a rectangle, one row per token in flight. */
    qe_route *croutes;

    /* everything this struct has to munmap, because qe_free_huge needs the
     * length it mapped and a decoded step cannot afford a page fault */
#define QE_MAX_OWNED 128
    void *own_p[QE_MAX_OWNED];
    size_t own_n[QE_MAX_OWNED];
    int n_owned;

    /* diagnostics */
    qe_forward_diag d;

    /* depth probes, see qe_forward_probe_set */
    int n_probe;
    const int *probe_layers;
    float *probe_buf;

    /* chunk-progress hook, see qe_forward_set_progress */
    qe_progress_cb progress;
    /* Called after every layer of a prompt chunk with the tokens that layer has carried,
     * as a fraction of the chunk: what a status panel draws between the chunk hooks, which
     * on a 512-row chunk are eight seconds apart. A count of work done, never an estimate. */
    void (*progress_fine)(void *user, long done, long total);
    void *progress_fine_user;
    long pf_base, pf_total; /* where the prompt in flight is, while one is */
    /* Each layer's share of the last prompt chunk's time, in tenths of a permille,
     * cumulative: a progress report after layer il says the chunk is pf_cum[il] done
     * by time rather than (il + 1) / n_layer by count. Attention layers at depth and
     * gated-delta layers do not cost the same, and a panel reading a three-second
     * window off an even split shows a rate that swings with which layers the window
     * happened to cover (44-80 tok/s on a prompt that ingests at a steady 79). */
#define PF_LAYERS 128
    int pf_cum[PF_LAYERS];
    int pf_have;
    void *progress_user;
};

/* The draft head's block, opened by qe_forward_open and walked by everything after
 * qsa_out: both are declared here because the open needs the tensors resident before
 * it has sized a single cache, and the walk needs the block the open sized. */
static int mtp_layer_open(qe_forward *f, qe_model *mh, int *qsa_slots, char *err, size_t errn);
static int mtp_rows(qe_forward *f, const int *toks, const float *h_in, int T, long pos0,
                    float *h_out, float *logits);
static int mtp_warm_chunk(qe_forward *f, long pos0, int T);

/* ------------------------------------------------------------------ helpers */

static long tensor_row_bytes(const qe_tensor *t) {
    return (long)(t->bytes / (long)t->ne[1]);
}

/* qe_gemv_* wants the row stride in blocks for quantized types and in elements
 * for f32/bf16, which is the only thing the kernels cannot infer. */
static long tensor_ldd(const qe_tensor *t) {
    if (t->type == QE_F32 || t->type == QE_BF16)
        return (long)t->ne[0];
    return tensor_row_bytes(t) / (long)qe_type_info_of(t->type)->block_sz;
}

/* Which slice of a cache array sequence `s` lives in.
 *
 * The caches are laid out `[slot][2][n_ctx][row]`, and a pass carrying B sequences
 * reads them as `[slot * B + s][2][n_ctx / B][row]`: the same bytes cut into B shorter
 * sequences instead of one long one, which is why `n_seq` costs no memory and buys a
 * pointer form that is not a second one. `f->n_ctx` is only ever the width a
 * sequence's cells are numbered against, so a sub-slot and that width are the whole
 * of what a layer needs to know about a sequence - and with `n_seq == 1` the
 * sub-slot is the slot and the width is `n_ctx`, which is why nothing below this has
 * an "or else" clause.
 *
 * The recurrences are the other shape of state and are not split: the delta matrices
 * and the conv rings have no length to be short about, so a sequence gets its own
 * copy of both, which is the ~216 MiB one more sequence actually costs. */
static long qsa_sub(const qe_forward *f, int slot, int s) {
    return (long)slot * (long)f->n_seq + (long)s;
}

static long gdn_sub(const qe_forward *f, int slot, int s) {
    return (long)slot * (long)f->n_seq + (long)s;
}

/* A sequence's own token ids and its own PLE convolution ring. Both are indexed by
 * the sequence's position, which restarts at 0 for each of them, so the PLE window of
 * n-grams a row hashes over never reaches into another sequence's text. */
static uint32_t *seq_ids(const qe_forward *f, int s) {
    return f->seq + (size_t)s * f->seq_stride;
}

static float *seq_ple_hist(const qe_forward *f, int s) {
    return f->ple_hist + (size_t)s * (size_t)f->ple_lag * (size_t)f->hc_dim;
}

/* The sizes of the state arrays, derived once. The allocation, the reset and the
 * snapshot below all clear or copy all of them, and three hand-written versions of
 * one product is how a reset ends up missing the last sequence of a batch. */
static size_t state_gdn_bytes(const qe_forward *f) {
    return (size_t)f->n_seq * (size_t)f->gdn_slots * (size_t)f->n_gdn_heads *
           (size_t)f->head_state * (size_t)f->head_state * sizeof(float);
}

static size_t state_gconv_bytes(const qe_forward *f) {
    return (size_t)f->n_seq * (size_t)f->gdn_slots * (size_t)(f->gdn_kernel - 1) *
           (size_t)f->gdn_channels * sizeof(float);
}

/* The pooled blocks are numbered from each sequence's own 0, so the array is strided
 * by `seq_stride / ratio + 1` of them and the slack is per sequence. At n_seq == 1
 * this is the product it always was. */
static size_t state_kpool_bytes(const qe_forward *f) {
    return (size_t)f->qsa_slots * (size_t)f->n_seq *
           (f->seq_stride / (size_t)f->kpool_ratio + 1) * (size_t)f->idx_dim * sizeof(float);
}

#define QE_IDX_DIM_MAX 512
/* One sequence's pooled keys, at the element width they are stored in: halves under
 * QE_IDX_F16, floats otherwise. The buffer is sized for floats either way. */
static inline void *kpool_seq(const qe_forward *f, long sub) {
    const size_t esz = f->idx_f16 ? 2 : 4;
    return (char *)f->kpool +
           (size_t)sub * (f->seq_stride / (size_t)f->kpool_ratio + 1) * (size_t)f->idx_dim * esz;
}
static inline void kpool_put(const qe_forward *f, void *base, long b, const float *src) {
    const int id = f->idx_dim;
    if (f->idx_f16) {
        uint16_t *d = (uint16_t *)base + (size_t)b * (size_t)id;
        int c = 0;
#if defined(__x86_64__)
        for (; c + 8 <= id; c += 8)
            _mm_storeu_si128((__m128i *)(d + c),
                             _mm256_cvtps_ph(_mm256_loadu_ps(src + c), _MM_FROUND_TO_NEAREST_INT));
#endif
        for (; c < id; c++)
            d[c] = _cvtss_sh(src[c], _MM_FROUND_TO_NEAREST_INT);
    } else
        memcpy((float *)base + (size_t)b * (size_t)id, src, (size_t)id * sizeof(float));
}

static size_t state_kraw_bytes(const qe_forward *f) {
    return (size_t)f->qsa_slots * (size_t)f->n_seq * f->seq_stride * (size_t)f->idx_dim *
           sizeof(float);
}

static size_t state_kv_bytes(const qe_forward *f) {
    const size_t ent = (size_t)f->qsa_slots * 2 * (size_t)f->n_seq * f->seq_stride;
    return ent * f->kv_row * (size_t)(f->kv_q8 ? 1 : (int)sizeof(float));
}

static size_t state_kvs_bytes(const qe_forward *f) {
    return (size_t)f->qsa_slots * 2 * (size_t)f->n_seq * f->seq_stride *
           (size_t)f->n_kv_heads * sizeof(float);
}

static size_t state_ple_bytes(const qe_forward *f) {
    return (size_t)f->n_seq * (size_t)f->ple_lag * (size_t)f->hc_dim * sizeof(float);
}

typedef struct {
    const qe_tensor *t;
    const void *w;
    const float *x;
    float *y;
    long K, row_bytes, ldd;
    /* Set when a lone token should walk this tensor's requantised copy instead of its
     * own bytes: the copy, and its row stride and block stride rather than the
     * tensor's. `w4` being set is the whole decision, which is why the type of the copy
     * travels with it instead of being read back off `t`. */
    const void *w4;
    qe_type type;
    /* More than one token of a chunk reading this tensor's rows, and the distance
     * between their input and output rows: `x` and `y` are then row 0 of the chunk, and
     * `gemv_task` hands the flattened (token, row) space to the pool. Zero strides and
     * a `T` of 1 is what every walk of a step uses, and the task keeps that path as its
     * own branch so nothing about a step changes. */
    int T;
    long xstride, ystride;
    /* Rows handed out by an atomic cursor in chunks of `dyn` rows instead of one fixed
     * slice per thread (0 = the fixed slice). See `dyn_rows`. */
    long dyn;
    long next;
    /* A lone token's activation recoded to int8 once for the dispatch, when the type the
     * walk reads has a one-token integer form (QE_I8_STEP). */
    const qe_i8panel *ap;
} gemv_job;

static long dq_copy_for(const qe_forward *f, const qe_tensor *t);
static int head_copy_for(const qe_forward *f, const qe_tensor *t, int rows, qe_type *type);
static const void *head_copy_at(const qe_forward *f, const qe_tensor *t, qe_type type);

/* ------------------------------------------------- the width of a lone token's walk */

/* The width a walk of one token's weights asks for is the width of the pass. This is the
 * account of why there is a knob here at all instead of a fourth thread count.
 *
 * A step reads each matrix once, so a step is the bus, and one kernel's walk of 512 MiB of
 * these weights raced at four widths in one process is not fastest at the width the pass
 * uses (`make gemv`, `/tmp/gemv_base.log`):
 *
 *   8 threads  54.7 GB/s   16 threads  78.3 GB/s (worst repetition 77.5)
 *  19 threads  73.7 GB/s   20 threads  77.7 GB/s (worst repetition 67.1)
 *
 * which reads as 6.2 % of a decode step left on the table by a pass width one core short
 * of the machine. It is not there. The same walk inside the engine - six steps after a
 * 128-token prefix, arms interleaved across three repetitions, best of each arm
 * (`make forward`'s leg for it, `/tmp/f37_wide.log`) - is monotonically worse the fewer
 * cores walk it:
 *
 *   19 threads  87.0 ms/token   16 threads  93.2 ms/token (1.07x slower)
 *  18 threads  89.2 ms/token   13 threads 101.9 ms/token (1.17x slower)
 *
 * and two readings of the same load agree with the leg: the sweep's width-1 token at 83.3
 * against 77.4 ms and its `block_proj` at 34.38 against 29.72 (`/tmp/f37_wide.log` against
 * `/tmp/f36_wide.log`), and generation at 10.66 against 11.27 tok/s. One kernel racing one
 * buffer at a width of its own is not the engine, which issues some 800 dispatches a token
 * over a bus it shares with a 19-thread expert walk, and the core the pass leaves out is
 * what that handshake buys (QE_THREADS_AUTO: 1.6 us median dispatch at 19 threads against
 * 69 us at 20). The walk wants it spent the way the barrier does.
 *
 * The width is therefore the pass width. The knob stays because the measurement above is
 * of this machine and not of the next one: `--walk-threads` and `QE_WALK_THREADS` price
 * the same question elsewhere, and the leg that fails when the shipped width stops being
 * the fastest of them is the gate on that. */

/* QE_WALK_THREADS names the width the engine starts on: a number is the width, `0` and
 * `full` are the pass width, which is also what an unset variable means. Only read at
 * open. */
static int walk_threads_env(void) {
    static long cached = 1 << 30;
    if (cached == (1 << 30)) {
        const char *e = getenv("QE_WALK_THREADS");
        if (!e)
            cached = 0;
        else if (!strcmp(e, "full") || !strcmp(e, "all"))
            cached = -1;
        else {
            const long v = strtol(e, NULL, 10);
            cached = v > 0 ? v : -1;
        }
    }
    return (int)cached;
}

void qe_forward_set_walk_threads(qe_forward *f, int n) {
    if (!f)
        return;
    if (n <= 0)
        f->walk_threads = f->n_threads; /* the width it was opened with, meaning "as asked" */
    else
        f->walk_threads = n < f->n_threads ? n : f->n_threads;
}

int qe_forward_get_walk_threads(const qe_forward *f) {
    return f ? f->walk_threads : 0;
}

/* ------------------------------------------------- how often a small projection asks */

/* QE_SMALL_PROJ_BATCH: whether a projection under the chunk walk's threshold is asked of
 * the pool once for the whole chunk or once for every token of it. Unset is the batch, and
 * only a "0" puts the per-token loop back, which is what `make forward`'s leg for it races
 * the default against. */
static int small_proj_env(void) {
    static long cached = 1 << 30;
    if (cached == (1 << 30)) {
        const char *e = getenv("QE_SMALL_PROJ_BATCH");
        cached = !e || strcmp(e, "0") != 0;
    }
    return (int)cached;
}

void qe_forward_set_small_proj_batch(qe_forward *f, int on) {
    if (f)
        f->small_proj_batch = on ? 1 : 0;
}

int qe_forward_get_small_proj_batch(const qe_forward *f) {
    return f ? f->small_proj_batch : 0;
}

/* QE_EXP2_SIMD: whether the two transcendental hot spots of ideas.md §6.6, the
 * gated-delta decay and one cell's attention weights, take the eight-wide exp2 or
 * libm. Unset is libm - the approximation is 3.5e-06 relative where `expf` is 1e-7,
 * so it is a change of arithmetic and only `make forward`'s leg can say what that
 * costs - and only a "1" asks for it. */
static int exp2_simd_env(void) {
    static long cached = 1 << 30;
    if (cached == (1 << 30)) {
        const char *e = getenv("QE_EXP2_SIMD");
        cached = e && strcmp(e, "0") != 0;
    }
    return (int)cached;
}

void qe_forward_set_exp2_simd(qe_forward *f, int on) {
    if (f)
        f->exp2_simd = on ? 1 : 0;
}

int qe_forward_get_exp2_simd(const qe_forward *f) {
    return f ? f->exp2_simd : 0;
}

/* Rows `row0`..`row0 + n - 1` of one tensor against one activation row. Every kernel
 * this reaches is a loop over output rows, so which rows were in the call alongside a
 * row is not part of what that row sums to - which is what lets `gemv_task` below cut
 * the same job up by token as well as by row, and what `make prefill` and
 * `make forward`'s small-projection leg hold it to. */
static void gemv_row_call(const gemv_job *j, const float *x, float *y, long row0, long n) {
    const void *W = j->w4 ? j->w4 : j->w;
    if (j->ap) {
        float *yp[1] = { y };
        if (qe_gemv_i8_t(j->type, (const char *)W + (size_t)row0 * (size_t)j->row_bytes, j->ap, 1,
                         yp, n, j->K, j->ldd) == 0)
            return;
    }
    if (qe_gemv_t(j->type, (const char *)W + (size_t)row0 * (size_t)j->row_bytes, x, y, n,
                  j->K, j->ldd) != 0) {
        /* A type with no kernel would otherwise score itself as zeros, which a
         * forward pass turns into a plausible-looking wrong answer. */
        fprintf(stderr, "qe: no GEMV kernel for type %s on %s\n",
                qe_type_info_of(j->t->type)->name, j->t->name);
        abort();
    }
}

static void gemv_task(void *v, int tid, int nthreads) {
    gemv_job *j = v;
    const long n_rows = (long)j->t->ne[1];
    long lo, hi;
    if (j->T == 1 && j->dyn > 0) {
        /* Whoever is free takes the next chunk. A row is still one kernel call on one
         * thread, so the bytes of every output are what the fixed slice makes; only
         * which core walked them moves, and a P-core that finishes early walks more. */
        for (;;) {
            const long r0 = __atomic_fetch_add(&j->next, j->dyn, __ATOMIC_RELAXED);
            if (r0 >= n_rows)
                return;
            const long n = r0 + j->dyn > n_rows ? n_rows - r0 : j->dyn;
            gemv_row_call(j, j->x, j->y + r0, r0, n);
        }
    }
    if (j->T == 1) {
        qe_slice(n_rows, tid, nthreads, &lo, &hi);
        if (lo < hi)
            gemv_row_call(j, j->x, j->y + lo, lo, hi - lo);
        return;
    }
    /* Several tokens of one tensor: slice the flattened (token, row) space, because a
     * tensor with four rows and nineteen threads has nothing to give fifteen of them
     * otherwise, and walk each token's rows in one call. */
    qe_slice(n_rows * (long)j->T, tid, nthreads, &lo, &hi);
    for (long i = lo; i < hi;) {
        const long tk = i / n_rows, r = i % n_rows;
        long n = n_rows - r;
        if (i + n > hi)
            n = hi - i;
        gemv_row_call(j, j->x + (size_t)tk * (size_t)j->xstride,
                  j->y + (size_t)tk * (size_t)j->ystride + r, r, n);
        i += n;
    }
}

/* The job a walk of `t` is, built the one way: the smaller copy when this width should
 * read it, the tensor's own bytes otherwise, and the row bytes and leading dimension
 * that go with whichever was picked. A step reads the smaller copy when one was staged
 * for this tensor; a chunk does not, and reads the tensor's own bytes at the width where
 * fewer of them buys nothing (ideas.md 4.1's table is the whole reason the test is a
 * width). A staged head copy is asked for as the one-row walk it is: the two heads are the
 * only tensors that have one, and a chunk of theirs is walked by `gemv_rows`, not here. */
static void gemv_job_for(qe_forward *f, const qe_tensor *t, const float *x, float *y,
                         long xstride, long ystride, int T, gemv_job *out) {
    qe_type ctype = t->type;
    const long di = dq_copy_for(f, t);
    const void *w4 = di >= 0 ? (const char *)f->dq_w + f->dq_woff[di] : NULL;
    if (w4)
        ctype = QE_IQ4_NL;
    else if (head_copy_for(f, t, 1, &ctype))
        w4 = head_copy_at(f, t, ctype);
    const long nb = (long)t->ne[0] / 32;
    *out = (gemv_job){ t,
                       qe_tensor_data(f->m, t),
                       x,
                       y,
                       (long)t->ne[0],
                       w4 ? nb * (long)qe_type_info_of(ctype)->block_sz : tensor_row_bytes(t),
                       w4 ? nb : tensor_ldd(t),
                       w4,
                       ctype,
                       T,
                       xstride,
                       ystride,
                       0,
                       0,
                       NULL };
    if (T == 1 && f->dyn_bytes > 0) {
        /* Chunks of about `dyn_bytes` of weights, never fewer rows than a kernel's
         * own row group, and small enough that every thread gets several. */
        long rows = (long)(f->dyn_bytes / (size_t)out->row_bytes);
        const long per = (long)t->ne[1] / (4L * f->n_threads);
        if (rows > per)
            rows = per;
        out->dyn = rows < 8 ? 8 : (rows + 7) & ~7L;
    }
    if (out->K <= 0 || t->ne[1] <= 0) {
        fprintf(stderr, "qe: tensor %s has an impossible shape\n", t->name);
        abort();
    }
}

/* The int8 recode of a lone token's input for the walks of one dispatch (QE_I8_STEP).
 *
 * A step's dense walk reads the IQ4_NL copy through the float kernel, which converts every
 * codebook value to a float and multiplies: 60 GB/s on this host where the bus is ~80 and
 * the head's Q8_0 walk reaches it. The one-token integer form of §7.22 spends the codebook
 * against an int8 activation instead - the same panel a chunk recodes, filled once on the
 * master for every matrix of the dispatch that reads that input. Panels live in the chunk
 * panel's rows, which a step never uses. `nslot` distinct inputs at most per dispatch. */
static qe_i8panel *step_panel(qe_forward *f, const float *x, long K, int slot) {
    if (!f->step_i8 || !f->i8x || K % 32 || K > f->i8k || slot >= QE_STEP_PANELS)
        return NULL;
    qe_i8panel *a = &f->spanel[slot];
    *a = (qe_i8panel){ .x8 = f->i8x + (size_t)slot * (size_t)f->i8k,
                       .s = f->i8s + (size_t)slot * (size_t)(f->i8k / 32),
                       .p = f->i8pseed + (size_t)slot * (size_t)(f->i8k / 32),
                       .stride8 = K,
                       .nb = K / 32 };
    (qe_i8_rowscale() ? qe_quant_q8_0_row_rs : qe_quant_q8_0_row)(x, a->x8, a->s, a->p, K);
    f->spanel_x[slot] = x;
    f->spanel_k[slot] = K;
    return a;
}

/* Attach a panel to a job when its walk has a one-token integer form, reusing the panel of
 * an earlier job of the same dispatch that read the same input. */
static void step_attach(qe_forward *f, gemv_job *j, int *nslots) {
    if (!f->step_i8 || j->T != 1 || !qe_gemv_i8_has_1(j->type))
        return;
    for (int i = 0; i < *nslots; i++)
        if (f->spanel_x[i] == j->x && f->spanel_k[i] == j->K) {
            j->ap = &f->spanel[i];
            return;
        }
    qe_i8panel *a = step_panel(f, j->x, j->K, *nslots);
    if (a) {
        j->ap = a;
        (*nslots)++;
    }
}

static void gemv(qe_forward *f, const qe_tensor *t, const float *x, float *y) {
    gemv_job j;
    gemv_job_for(f, t, x, y, 0, 0, 1, &j);
    if (f->chunk_T == 1) {
        int ns = 0;
        step_attach(f, &j, &ns);
    }
    if (f->o.profile)
        f->walk_mac += (double)t->ne[0] * (double)t->ne[1];
    /* A width of its own, on a token of its own and on nothing else: a chunk's rows share a
     * weight walk whether they go through `gemv_chunk_task` or, for a tensor the chunk walk
     * will not take, through `proj_small` or this function once a token, and the width that
     * is right for a bus is wrong for an arithmetic-bound walk. On this host the two widths
     * measure the same (see the account above `walk_threads_env`), so this is the pass width
     * unless a caller asks otherwise. And the reason it is safe to ask at all: one output row
     * goes to one thread and that thread walks it in one call, so which threads did the
     * walking is not part of what any row sums to. `make forward` races widths against each
     * other and asserts the logits of a step byte for byte. */
    qe_pool_run(gemv_task, &j, f->chunk_T == 1 ? f->walk_threads : f->n_threads);
}

/* Several walks of one token that read inputs nobody in the group writes, as one dispatch
 * of the pool's stage graph instead of one dispatch each (ideas.md §7.30).
 *
 * A decode step is ~800 dispatches over ~4 GiB, and each of them ends in a barrier where
 * the bus goes quiet until the slowest of nineteen threads is done and the next matrix's
 * first lines are on their way. Four projections that read the same mixer output have no
 * ordering between them, so a thread that finishes its slice of the first takes a slice of
 * the next one instead of waiting: one barrier where there were four. Every slice is still
 * one output row range of one tensor walked in one kernel call, so what any row sums to is
 * the same bytes as the four dispatches - this is a scheduling change and nothing else. */
#define QE_GROUP_MAX 8
static void gemv_group(qe_forward *f, int n, const qe_tensor *const *t, const float *const *x,
                       float *const *y) {
    if (f->chunk_T != 1 || !f->step_fuse || n < 2) {
        for (int i = 0; i < n; i++)
            gemv(f, t[i], x[i], y[i]);
        return;
    }
    gemv_job j[QE_GROUP_MAX];
    qe_pool_item it[QE_GROUP_MAX];
    int ns = 0;
    for (int i = 0; i < n; i++) {
        gemv_job_for(f, t[i], x[i], y[i], 0, 0, 1, &j[i]);
        step_attach(f, &j[i], &ns);
        if (f->o.profile)
            f->walk_mac += (double)t[i]->ne[0] * (double)t[i]->ne[1];
        it[i] = (qe_pool_item){ gemv_task, &j[i], f->walk_threads, 0 };
    }
    qe_pool_run_items(it, n);
}

/* A projection under the chunk walk's threshold, every token of the chunk in one ask of
 * the pool.
 *
 * Which tensors that is, counted out of this checkpoint's own header: 180 of the projections
 * a token walks are under the 1 MiB threshold - a hyper-connection mixer's injection matrix
 * (4 x 10240 f32, 160 KiB, 96 of them), a gated-delta block's alpha and beta (48 x 2560 f32,
 * 480 KiB, 72), an indexer's key projection (128 x 2560 bf16, 640 KiB, 12) - and every one
 * of them is read by every token of a chunk. What that used to cost: `proj` looped the
 * chunk's tokens through `gemv`, so a width-512 chunk took 92,160 barriers of a nineteen-thread
 * pool where the walkable projections of the same chunk take one dispatch each, and a hot
 * dispatch is 2.6 us (`make pool`) - 0.47 ms of an ingested token of arithmetic. What the leg
 * measures is 0.46 of it in the stages and 0.58 ms/token of the wall (16.44 against 15.86,
 * arms interleaved, best of three): `block_proj` 1.81 -> 1.60 and the two mixers 1.69 -> 1.44
 * ms/token (`/tmp/f39_wide.log`).
 *
 * One ask does the same 512 walks over the flattened (token, row) space, which `gemv_task` is
 * free to cut up however it likes because every kernel it calls is a loop over output rows:
 * which rows were in the call alongside a row is not part of what that row sums to, and the
 * leg holds the chunk's logits to that byte for byte.
 *
 * A step is not in this at all: one token is one dispatch whichever way the question is
 * asked (88.1 against 88.4 ms, which is the same step twice), and the loop in `proj` is what
 * runs when `QE_SMALL_PROJ_BATCH=0` puts the pass back the way it was. */
static void proj_small(qe_forward *f, const qe_tensor *t, const float *x, long xstride,
                       float *y, long ystride) {
    gemv_job j;
    gemv_job_for(f, t, x, y, xstride, ystride, f->chunk_T, &j);
    if (f->o.profile)
        f->walk_mac += (double)t->ne[0] * (double)t->ne[1] * (double)f->chunk_T;
    qe_pool_run(gemv_task, &j, f->n_threads);
}

typedef struct {
    const qe_tensor *t;
    const void *w;
    const float *x;
    float *y;
    long K, row_bytes, ldd, xstride, ystride;
    int T;
    /* Non-NULL when the chunk has just been recoded into it: the walk then reads one
     * int8 row per token instead of a float row, and one f32 dequantisation of a weight
     * block serves all T tokens of the chunk rather than one. */
    const qe_i8panel *ap;
    /* The requantised copy of this tensor, whose rows are `K` bytes apart, and its one
     * scale per output row (both NULL when there is no copy or the copy is not what was
     * asked for). Only ever read with `ap` set: a flat row has to be multiplied by a
     * panel that carries one scale of its own, which is what `qe_i8_rowscale` says. */
    const void *wf;
    const float *ws;
    /* The type the walk reads, which is the tensor's own except for a staged copy of a
     * head: those are another format, and the walk is handed row bytes and a leading
     * dimension that only make sense with them. */
    qe_type type;
    /* Rows handed out from a cursor in chunks of `dyn` (0 = one fixed slice a thread), the
     * same trade as gemv_job's: every output row is still one kernel call's. */
    long dyn;
    long next;
} gemv_chunk_job;

/* The chunk recode, one slice of the T token rows per thread. It is a dispatch of its
 * own and not a clause inside the walk below because the panel depends only on the
 * activation: every projection that reads it wants the same bytes, and a fill repeated
 * per row-slice would be T x K values per thread instead of per projection. */
typedef struct {
    const float *x;
    int8_t *x8;
    float *s;
    int32_t *p;
    long xstride, K;
    int T;
} panel_job;

static void panel_fill_task(void *v, int tid, int nthreads) {
    panel_job *j = v;
    long lo, hi;
    qe_slice(j->T, tid, nthreads, &lo, &hi);
    if (lo >= hi)
        return;
    const long nb = j->K / 32;
    /* Which scale a row carries belongs to the walk as much as to this fill: the codebook
     * walks read `qe_i8_rowscale()` to choose their scale exit, so a panel and the rows it
     * feeds may not be allowed to disagree about it (src/moe.c makes the same branch, and a
     * row-scaled panel is a per-block panel with equal scales for the walks that have no
     * row-scale form). */
    const int rs = qe_i8_rowscale();
    for (long t = lo; t < hi; t++)
        (rs ? qe_quant_q8_0_row_rs : qe_quant_q8_0_row)(
            j->x + (size_t)t * (size_t)j->xstride, j->x8 + (size_t)t * (size_t)j->K,
            j->s + (size_t)t * (size_t)nb, j->p + (size_t)t * (size_t)nb, j->K);
}

/* Output rows that share one activation load in the flat walk. Three is what the register
 * file holds: twelve accumulators for three rows against four tokens, three weight
 * registers and one activation register is the sixteen ymm exactly, and a fourth row would
 * spill them (`make walk` prices 1, 2, 3 and 4 at `/tmp/walk_flat2.log`, three fastest at
 * every width above eight tokens: 1.84x of the walk that ships at T=512). */
#define QE_FLAT_TILE 3

static void gemv_chunk_rows(gemv_chunk_job *j, long lo, long hi);

static void gemv_chunk_task(void *v, int tid, int nthreads) {
    gemv_chunk_job *j = v;
    long n_rows = (long)j->t->ne[1];
    long lo, hi;
    if (j->dyn > 0) {
        for (;;) {
            lo = __atomic_fetch_add(&j->next, j->dyn, __ATOMIC_RELAXED);
            if (lo >= n_rows)
                return;
            hi = lo + j->dyn > n_rows ? n_rows : lo + j->dyn;
            gemv_chunk_rows(j, lo, hi);
        }
    }
    qe_slice(n_rows, tid, nthreads, &lo, &hi);
    if (lo >= hi)
        return;
    gemv_chunk_rows(j, lo, hi);
}

/* Rows [lo, hi) of one chunk walk: the body of gemv_chunk_task for one range. */
static void gemv_chunk_rows(gemv_chunk_job *j, long lo, long hi) {
    if (j->ap) {
        /* One pointer per token row of the chunk, offset to this slice's first row: the
         * integer walk writes ys[u][r] for r in [0, n_rows) of the weight rows it was
         * handed, which is the same rectangle the walk below walks. */
        float *yp[QE_CHUNK_MAX];
        for (int u = 0; u < j->T; u++)
            yp[u] = j->y + (long)u * j->ystride + lo;
        /* Three rows of the matrix per load of a token group, out of the copy that carries
         * one scale for a whole row. It refuses a token group its tile does not divide,
         * which in practice is the ragged tail of a prompt: 598 tokens at width 512 ends in
         * 86 rows, and the widest form this file has walks four tokens at a time. The tail
         * takes the walk with the hoisted exit below rather than being walked twice at two
         * different scales. */
        if (j->wf && qe_gemv_q8_0_flat_i8((const char *)j->wf + (size_t)lo * (size_t)j->K,
                                          j->ws + lo, j->ap, j->T, yp, hi - lo, j->K, j->K,
                                          QE_FLAT_TILE) == 0)
            return;
        if (qe_gemv_i8_t(j->type,
                         (const char *)j->w + (size_t)lo * (size_t)j->row_bytes, j->ap, j->T, yp,
                         hi - lo, j->K, j->ldd) == 0)
            return;
    }
    if (qe_gemv_multi_t(j->type, (const char *)j->w + (size_t)lo * (size_t)j->row_bytes, j->x,
                        j->y + lo, hi - lo, j->K, j->ldd, j->T, j->xstride, j->ystride,
                        (size_t)j->row_bytes, QE_ROW_BLOCK) != 0) {
        fprintf(stderr, "qe: no GEMV kernel for type %s on %s\n",
                qe_type_info_of(j->type)->name, j->t->name);
        abort();
    }
}

/* The chunk size a multi-row walk hands out, in rows: about four per thread so a fast core
 * can take a fifth while a slow one finishes, a multiple of 24 so the three-row flat tile and
 * the eight-row walks keep whole groups, and none at all when `QE_DYN_KB` is 0. */
static long chunk_dyn_rows(const qe_forward *f, long n_rows) {
    if (!f->dyn_bytes || !f->chunk_dyn)
        return 0;
    long r = n_rows / (4L * f->n_threads);
    r = (r / 24) * 24;
    return r < 24 ? 24 : r;
}

/* Row t of a chunk buffer that is `stride` floats per row. */
static float *crow(float *base, long stride, int t) {
    return base + (size_t)t * (size_t)stride;
}

/* One tensor against T token rows of activation at once. `x` is the chunk's own
 * layout (T rows `f->K` apart), `y` is T rows `ystride` apart, and row blocks are
 * swept outside the tokens, so a block of weights is read once and multiplied T
 * times instead of a whole tensor being walked once per token.
 *
 * That ordering is a loss on the dense projections, and `make forward` has the
 * numbers: they are small enough that the T-1 later rows are served from the cache
 * the first row filled anyway, and blocking them trades DRAM efficiency for bytes
 * that were never read twice. A tensor far bigger than the L3 has no such free
 * lunch - the head is 497 MiB against a 30 MiB cache - and this is the walk for it.
 * Each y[r][v] still comes from the same kernel on the same inputs as the separate
 * call, so unlike a Q8_0 walk that shares its dequantisation this one is bit-exact
 * with T calls of gemv(). */
static void gemv_rows(qe_forward *f, const qe_tensor *t, const float *x, float *y, int T,
                      long ystride) {
    /* A window of one row is the same walk a step already does with its own logits, so it
     * reads the same copy of the head; two rows and up read the tensor's own bytes, where
     * the copy of a Q6_K head is the larger of the two and the assembly the smaller one
     * avoids is amortised over the rows sharing it. */
    qe_type ctype = QE_Q8_0;
    const void *wh = head_copy_for(f, t, T, &ctype) ? head_copy_at(f, t, ctype) : NULL;
    const long nb = (long)t->ne[0] / 32;
    gemv_chunk_job j = { t,
                         wh ? wh : qe_tensor_data(f->m, t),
                         x,
                         y,
                         (long)t->ne[0],
                         wh ? nb * (long)qe_type_info_of(ctype)->block_sz : tensor_row_bytes(t),
                         wh ? nb : tensor_ldd(t), (long)f->K, ystride,
                         T,
                         NULL,
                         NULL,
                         NULL,
                         wh ? ctype : t->type,
                         chunk_dyn_rows(f, (long)t->ne[1]),
                         0 };
    if (f->o.profile)
        f->walk_mac += (double)t->ne[0] * (double)t->ne[1] * (double)T;
    qe_pool_run(gemv_chunk_task, &j, f->n_threads);
}

/* Whether proj() would take the chunk walk for this tensor. The other form asks the pool
 * for the chunk once, or once a token with `QE_SMALL_PROJ_BATCH=0`, and either way it is a
 * dispatch - and a dispatch inside an item of a stage graph is the pool running the pool,
 * so a graph carries only walkable projections. */
static int proj_walkable(const qe_forward *f, const qe_tensor *t) {
    return f->chunk_T > 1 && f->proj_chunk_bytes && t->bytes >= f->proj_chunk_bytes;
}

/* Whether proj() would recode the chunk for this tensor's walk, which is a question about
 * the tensor and the width and not about who is calling. It has to be one test, asked the
 * same way by proj() and by moe_stage_graph(): the graph and the schedule it replaces are
 * only the same arithmetic if the two agree about which walk a tensor takes, and a graph
 * that quietly declined the recode while the schedule took it is a different model wearing
 * a scheduling change (`/tmp/f24_wide.log` measured exactly that, as a trunk that was not
 * the same bytes). */
static int proj_i8_wanted(const qe_forward *f, const qe_tensor *t) {
    const long K = (long)t->ne[0];
    return f->proj_i8 && f->chunk_T >= 2 && !(K % 32) && K <= f->i8k &&
           qe_gemv_i8_has(t->type);
}

/* Which entry of the tensor table this is: the walk is handed a tensor and nothing else,
 * and that is how it finds what was staged for it. The table is one contiguous array
 * (src/gguf.c answers an index with &m->t[i]), so the difference is the index. */
static long tensor_index(qe_model *m, const qe_tensor *t) {
    const qe_tensor *first = qe_tensor_by_index(m, 0);
    if (!t || !first)
        return -1;
    const long i = (long)(t - first);
    return (i >= 0 && (size_t)i < qe_tensor_count(m)) ? i : -1;
}

/* Which staged requantised copy this walk should read, or -1 for the walk that reads the
 * Q8_0 blocks. Three things have to agree before a copy is usable: it has to exist (staged
 * at open, never on first use), the panel has to be the row-scaled one (`qe_i8_rowscale` -
 * a flat row multiplied by a panel carrying thirty-two scales a row is the wrong number
 * twice over), and there has to be a chunk, since the whole point is one load of a token
 * group serving more than one output row.
 *
 * Asked the same way by proj() and by proj_item(), which is the same reason proj_i8_wanted
 * is one test: a stage graph and the schedule it replaces are only the same arithmetic if
 * the two agree about which copy a tensor reads. */
static long flat_copy_for(const qe_forward *f, const qe_tensor *t) {
    if (!f->flat_i8 || !f->flat_woff || f->chunk_T < 2 || !qe_i8_rowscale())
        return -1;
    /* The map below is indexed by a tensor's place in *this* model's table, and a
     * draft head is a second model whose rows happen to be the same shape. Asking
     * `tensor_index` about one of those returns -1 today, but the copy it would have
     * found is a plausible-looking offset into a requantised matrix belonging to a
     * layer that never asked for it, so the two models are kept apart by their owner
     * rather than by the arithmetic happening to work. */
    if (t->owner && t->owner != f->m)
        return -1;
    const long i = tensor_index(f->m, t);
    return (i >= 0 && f->flat_woff[i] >= 0) ? i : -1;
}

static long dq_copy_for(const qe_forward *f, const qe_tensor *t) {
    if (!f->dq_i4 || !f->dq_woff || f->chunk_T < 1 || f->chunk_T > f->dq_multi)
        return -1;
    if (t->owner && t->owner != f->m)
        return -1;
    const long i = tensor_index(f->m, t);
    if (i < 0 || f->dq_woff[i] < 0)
        return -1;
    /* A tensor of both classes is refused unless both are allowed, which is what makes
     * "lean" mean what it says about a mixer that also sits at an end of the trunk. */
    if (f->dq_wcls && (f->dq_wcls[i] & ~f->dq_tier))
        return -1;
    return i;
}

/* Whether this walk is an output head reading one of its staged copies, and which
 * format that copy is. The trunk's `output.weight` and the draft head's own, and only
 * them: the head is the one matrix of this checkpoint that a pass walks whole and once per
 * row, and it is the largest single item in a decode step's byte bill. Both heads count
 * because both are walked once per row of what they emit, so a draft row and a verified row
 * would otherwise read the same weights through two different arithmetics. */
static int head_copy_for(const qe_forward *f, const qe_tensor *t, int rows, qe_type *type) {
    /* One row of a pass with one row is the case `make score`'s decode arms priced, and it
     * is the whole of what a step does. Several rows at once is the verify window, and
     * whether the same copy is worth it there is `head_win`'s to say - a window of `rows`
     * is walked with `chunk_T` at `rows`, so the two tests cannot be one. */
    if (f->head_copy == QE_HEAD_STORED)
        return 0;
    if (rows == 1 ? f->chunk_T != 1 : !f->head_win)
        return 0;
    const int draft = t == f->mtp_outw && t != f->outw;
    if (!draft && t != f->outw)
        return 0;
    const void *i4 = draft ? f->mh4_w : f->h4_w;
    const void *q8 = draft ? f->mhq_w : f->hq_w;
    if (f->head_copy == QE_HEAD_I4 && i4) {
        *type = QE_IQ4_NL;
        return 1;
    }
    if (f->head_copy == QE_HEAD_Q8 && q8) {
        *type = QE_Q8_0;
        return 1;
    }
    return 0;
}

static const void *head_copy_at(const qe_forward *f, const qe_tensor *t, qe_type type) {
    const int draft = t == f->mtp_outw && t != f->outw;
    const void *q8 = draft ? f->mhq_w : f->hq_w;
    const void *i4 = draft ? f->mh4_w : f->h4_w;
    return type == QE_Q8_0 ? q8 : i4;
}

/* One projection as an item of a stage graph. `job` is where the walk's arguments
 * live, which means it has to outlive the dispatch the item belongs to, exactly as
 * it has to outlive qe_pool_run. `ap` is the recoded chunk, or NULL for the walk that
 * dequantises the weights instead; a graph that wants it fills it with an item of its
 * own, because an item that dispatched one would be the pool running the pool. */
static void proj_item(qe_forward *f, const qe_tensor *t, const float *x, long xstride, float *y,
                      long ystride, const qe_i8panel *ap, gemv_chunk_job *job,
                      qe_pool_item *out) {
    /* The copy is only read with a recoded panel under it, so a tensor that is not
     * getting one gets the walk it has always got. */
    const long di = dq_copy_for(f, t);
    const long fi = (ap && di < 0) ? flat_copy_for(f, t) : -1;
    const long nb = (long)t->ne[0] / 32;
    if (f->o.profile)
        f->walk_mac += (double)t->ne[0] * (double)t->ne[1] * (double)f->chunk_T;
    *job = (gemv_chunk_job){ t,
                             di >= 0 ? (const void *)((const char *)f->dq_w + f->dq_woff[di])
                                     : qe_tensor_data(f->m, t),
                             x,       y,
                             (long)t->ne[0],
                             di >= 0 ? nb * (long)qe_type_info_of(QE_IQ4_NL)->block_sz
                                     : tensor_row_bytes(t),
                             di >= 0 ? nb : tensor_ldd(t), xstride,
                             ystride,        f->chunk_T,          ap,
                             fi >= 0 ? (const char *)f->flat_w + f->flat_woff[fi] : NULL,
                             fi >= 0 ? f->flat_s + f->flat_soff[fi] : NULL,
                             di >= 0 ? QE_IQ4_NL : t->type,
                             chunk_dyn_rows(f, (long)t->ne[1]),
                             0 };
    out->fn = gemv_chunk_task;
    out->arg = job;
    out->nthreads = f->n_threads;
    out->deps = 0;
}

/* The activation of the shared expert, over the whole chunk in one dispatch.
 *
 * It is its own item and not a loop in the caller because of what it sits between:
 * the two projections that feed it and the one that consumes it, which is the same
 * shape of wait the expert walk is in.
 *
 * Sliced by row and not by output element, which is the difference between a
 * scheduling change and a different model. qe_silu_mul does eight at a time and
 * finishes with a scalar tail, and the two use different sigmoids (the body a vector
 * exp2 approximation, the tail libm expf), so the bytes a row comes out with depend
 * on where the calls covering it start and stop. The loop this task replaces called
 * it once per row with exactly ns elements; slicing the T*ns elements instead hands
 * the same row to calls of arbitrary length and moves every value in it by an ulp,
 * which `make forward` then reports as a trunk that is not the same bytes. One whole
 * row per call is the same arithmetic by construction. */
typedef struct {
    float *g, *u;
    long ns;
    int T;
} silu_job;

static void silu_rows_task(void *v, int tid, int nthreads) {
    silu_job *j = v;
    const long ns = j->ns;
    long lo, hi;
    qe_slice(j->T, tid, nthreads, &lo, &hi);
    for (long row = lo; row < hi; row++) {
        const size_t at = (size_t)row * (size_t)ns;
        qe_silu_mul(j->g + at, j->g + at, j->u + at, ns);
    }
}

/* The size at which a projection is worth one walk for the chunk instead of T,
 * QE_PROJ_MB=<MiB>, default 1. For this checkpoint 1 MiB means every projection:
 * they are 2560-wide Q8_0 matrices of some millions of bytes, the widest being
 * attn_q's 33 MiB, and the row block the walk works in is half a 2 MiB L2.
 *
 * What it buys was measured by `make forward` on 384 tokens of prose at the
 * default width of 8, in one load, one prompt, both ways round: **69.1 ms a token
 * ordered against 57.8 blocked, 1.20x**. That is §12's kernel ratio arriving at the
 * step level with most of itself intact, and it is the whole reason the walk is the
 * default: the ordered form costs a fifth more ingestion to keep bits that nothing
 * downstream reads differently. What it costs is stated in proj(): a Q8_0
 * projection taken through the shared walk sums in two chains per token instead of
 * the single kernel's four, so a chunked prefill does not land on the trunk that
 * feeding the same prompt one token at a time lands on, bit for bit. The measured
 * size of that is 9.0e-01 in a logit against a scale of 18.4, worst over the 248320
 * the head emits - 5 % of the head's own range, which is a change in the answer and
 * not in the last bit of it, and is why `make forward` asserts the two separately
 * instead of comparing them for equality. Over the whole 951-token prompt the same
 * comparison prints 1.6 against 16.9, 9 %.
 *
 * The reason this file has a history paragraph: `proj` used to compare an unsigned
 * `t->bytes` against this threshold, so the documented "0 is never" tested
 * `bytes < 0`, was never true, and the default of 0 blocked everything. Every
 * measurement that concluded that blocking is worth nothing measurable - 65 ms a
 * token ordered, 67 at 4 MiB, 66 at 1 MiB, "the same trunk bits in all three" -
 * had therefore compared the walk against itself, and the identity check that was
 * supposed to notice copied the trunk twice before feeding any tokens and compared
 * the two copies. Both are fixed, and what is left is a measured trade with a name:
 * 1.20x of ingestion for a reassociation, default on, `--proj-chunk -1` to opt out.
 *
 * 1 MiB and not 1 byte, so that a tensor below one row block of the walk is not
 * sent through it at all. */
static size_t proj_chunk_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_PROJ_MB");
        long v = e ? strtol(e, NULL, 10) : 1;
        cached = v < 0 ? 0 : v;
    }
    return (size_t)cached * 1024 * 1024;
}

void qe_forward_set_proj_chunk(qe_forward *f, size_t bytes) {
    f->proj_chunk_bytes = bytes;
}

size_t qe_forward_get_proj_chunk(const qe_forward *f) {
    return f->proj_chunk_bytes;
}

/* Whether a chunk's projections recode the chunk to int8 and walk the weights with a
 * vector integer multiply-accumulate instead of dequantising to floats. On: the dense
 * walk is 0.817x of the one it replaces (6.09 to 4.98 ms of an ingested token) and the
 * step 1.089x (26.71 to 24.53 ms/token, `/tmp/f20_wide.log`), which is the second-largest
 * single item left in a token after the expert stage. It is a change of number and not of
 * instruction, so the same two instruments price it as the expert recode: `make forward`
 * holds it to half the head's range (21 %, same next token) and `make score --ingest`
 * charges it against the token after the window in the same load as the schedule change
 * this engine already ships. QE_I8_PROJ=0 is the way back; the setter below is how one
 * load of an 87 GiB file measures both arithmetics against each other. */
static int proj_i8_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_I8_PROJ");
        cached = e ? strtol(e, NULL, 10) != 0 : 1;
    }
    return (int)cached;
}

void qe_forward_set_proj_i8(qe_forward *f, int on) {
    if (f)
        f->proj_i8 = on ? 1 : 0;
}

int qe_forward_get_proj_i8(const qe_forward *f) {
    return f ? f->proj_i8 : 0;
}

/* On: a load priced it at 1.65x of the walk it replaces (block_proj 2.92 against 1.77
 * ms/token at the width a chunk ingests at) and the head's answer moved 6.9e-04 nats on the
 * token the engine goes on to say, 10 % of the head's own range across its top 32 with the
 * same next token (ideas.md 7.18, /tmp/f31_wide.log). The copies this walk reads are staged
 * at open and only when it was asked for at open, which makes this the one switch here that
 * can refuse: setting it on without the copies leaves the getter answering 0, and that is how
 * a bench tells "never built, this open is a metadata view" from "asked for and made no
 * difference". QE_I8_FLAT=0 is the way back, and it costs 1.08x of a whole ingested token in
 * the same load (15.71 against 14.58 ms/token). */
static int flat_i8_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_I8_FLAT");
        cached = e ? strtol(e, NULL, 10) != 0 : 1;
    }
    return (int)cached;
}

void qe_forward_set_flat_i8(qe_forward *f, int on) {
    if (f)
        f->flat_i8 = (on && f->flat_woff) ? 1 : 0;
}

int qe_forward_get_flat_i8(const qe_forward *f) {
    return f ? f->flat_i8 : 0;
}

/* The other end of the same file's bytes: the copies below are for a chunk of one,
 * where a matrix is walked once and a token's time is therefore its bytes over the
 * bus. ideas.md 4.1 measured that swap - the same projection, three quantisations, one
 * walk of each - at 1.74x of the Q8_0 walk at T=1 and at 0.96-1.02x at the width a
 * chunk ingests at, which is the whole reason this switch is a width test in `gemv` and
 * not a second walk everywhere. What it costs the answer is the question the decode A/B
 * of `make score` answers in the load that is already open, and ideas.md 7.23 records
 * what it said; `QE_I4_DENSE=0` is the way back. Like the flat copies, these are staged
 * at open or not at all, so setting this against an open that could not afford them
 * leaves the getter answering 0. */
static int dq_i4_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_I4_DENSE");
        cached = !e ? 1 : !strcmp(e, "0") || !strcmp(e, "off") || !strcmp(e, "none") ? 0 : 1;
    }
    return (int)cached;
}

void qe_forward_set_decode_i4(qe_forward *f, int on) {
    if (f)
        f->dq_i4 = (on && f->dq_woff) ? 1 : 0;
}

int qe_forward_get_decode_i4(const qe_forward *f) {
    return f ? f->dq_i4 : 0;
}

/* QE_I4_MULTI=N: passes of up to N rows read the IQ4_NL dense copy too (1 = a lone
 * token only, which is how the copy first shipped). */
static int dq_multi_env(void) {
    const char *e = getenv("QE_I4_MULTI");
    int v = e && *e ? atoi(e) : QE_I4_MULTI_DEFAULT;
    return v < 1 ? 1 : v > QE_CHUNK_MAX ? QE_CHUNK_MAX : v;
}

void qe_forward_set_multi_i4(qe_forward *f, int rows) {
    if (f)
        f->dq_multi = rows < 1 ? 1 : rows > QE_CHUNK_MAX ? QE_CHUNK_MAX : rows;
}

int qe_forward_get_multi_i4(const qe_forward *f) { return f ? f->dq_multi : 0; }

static int stage_pf_env(void) {
    const char *e = getenv("QE_STAGE_PF");
    return e && *e ? atoi(e) : QE_STAGE_PF_DEFAULT;
}

void qe_forward_set_prefetch(qe_forward *f, int on) {
    if (f)
        f->stage_pf = on;
}

int qe_forward_get_prefetch(const qe_forward *f) { return f ? f->stage_pf : 0; }

static int step_fuse_env(void) {
    const char *e = getenv("QE_STEP_FUSE");
    return e && *e ? atoi(e) : QE_STEP_FUSE_DEFAULT;
}

void qe_forward_set_step_fuse(qe_forward *f, int on) {
    if (f)
        f->step_fuse = on;
}

int qe_forward_get_step_fuse(const qe_forward *f) { return f ? f->step_fuse : 0; }

static size_t dyn_env(void) {
    const char *e = getenv("QE_DYN_KB");
    long kb = e && *e ? atol(e) : QE_DYN_KB_DEFAULT;
    return kb > 0 ? (size_t)kb * 1024u : 0;
}

void qe_forward_set_dyn_kb(qe_forward *f, long kb) {
    if (f)
        f->dyn_bytes = kb > 0 ? (size_t)kb * 1024u : 0;
}

long qe_forward_get_dyn_kb(const qe_forward *f) { return f ? (long)(f->dyn_bytes / 1024u) : 0; }

void qe_forward_set_step_i8(qe_forward *f, int on) {
    if (f)
        f->step_i8 = on;
}

int qe_forward_get_step_i8(const qe_forward *f) { return f ? f->step_i8 : 0; }

void qe_forward_set_chunk_dyn(qe_forward *f, int on) {
    if (f)
        f->chunk_dyn = on;
}

int qe_forward_get_chunk_dyn(const qe_forward *f) { return f ? f->chunk_dyn : 0; }

void qe_forward_set_attn_batch(qe_forward *f, int on) {
    if (f)
        f->attn_batch = on && f->cqwork ? 1 : 0;
}

int qe_forward_get_attn_batch(const qe_forward *f) { return f ? f->attn_batch : 0; }

void qe_forward_set_rows_par(qe_forward *f, int on) {
    if (f)
        f->rows_par = on;
}

int qe_forward_get_rows_par(const qe_forward *f) { return f ? f->rows_par : 0; }

void qe_forward_set_delta_simd(qe_forward *f, int on) {
    if (f)
        f->delta_simd = on;
}

int qe_forward_get_delta_simd(const qe_forward *f) { return f ? f->delta_simd : 0; }

void qe_forward_set_moe_dyn(qe_forward *f, int on) {
    if (f)
        for (int il = 0; il < f->n_layer; il++)
            if (f->moe[il])
                qe_moe_set_dyn(f->moe[il], on);
}

int qe_forward_get_moe_dyn(const qe_forward *f) {
    return f && f->n_layer > 0 && f->moe[0] ? qe_moe_get_dyn(f->moe[0]) : 0;
}

/* Which classes of dense matrix are refused the smaller copy - the `qe_dense_class` of the
 * header. These are ideas.md 4.1's two refinements, and it is there that they were first
 * written down as a way of buying back a loss instead of taking a speed-up or leaving it:
 * the low-rank hyper-connection mixers, whose every output row is a gate on the residual the
 * next block reads, and the two ends of the trunk, where an error has the fewest layers left
 * to be averaged away by. A class is decided from a tensor's name once, when its copy is
 * staged, and applied per walk, so every policy shares the arenas that were already built. */

/* Which classes a policy lets read a copy. A tensor of two classes needs both allowed. */
static int dq_tier_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_I4_DENSE");
        cached = !e || !*e || !strcmp(e, "all") ? QE_DCLASS_HC | QE_DCLASS_EDGE
                 : !strcmp(e, "lean")          ? QE_DCLASS_PLAIN
                 : !strcmp(e, "nohc")          ? QE_DCLASS_EDGE
                 : !strcmp(e, "noedge")        ? QE_DCLASS_HC
                                               : QE_DCLASS_HC | QE_DCLASS_EDGE;
    }
    return (int)cached;
}

void qe_forward_set_decode_tier(qe_forward *f, int classes) {
    if (f)
        f->dq_tier = classes & (QE_DCLASS_HC | QE_DCLASS_EDGE);
}

int qe_forward_get_decode_tier(const qe_forward *f) {
    return f ? f->dq_tier : 0;
}

/* QE_I4_HEAD=stored|i4|q8. Whichever it says, the copy has to have been staged, so an
 * open that could not afford one reports the reading it actually has. Both copies cost
 * under 1 GiB on this checkpoint, next to the 1.9 GiB of dense copies, and both are
 * staged whenever the head is a Q6_K matrix at all - which one a step reads is a field,
 * and `make score`'s decode A/B walks every reading in the load that staged them.
 * Stored is what ships until that A/B says otherwise: unlike a dense projection, the head
 * is where the answer comes from, so what a smaller codebook costs is a number about this
 * checkpoint's loss and not about two sizes. */
static int head_copy_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_I4_HEAD");
        cached = !e ? QE_HEAD_Q8 : !strcmp(e, "q8") ? QE_HEAD_Q8
                                 : !strcmp(e, "i4") ? QE_HEAD_I4
                                                    : QE_HEAD_STORED;
    }
    return (int)cached;
}

/* QE_I4_HEAD_WIN=0 keeps a verify window off the head's copy. It ships on: `make head` raced
 * the two in the load that staged the copy, and a speculative pass at the draft length that
 * wins goes 90.6 to 86.4 ms per emitted id with it (`/tmp/head_d.log`). What made it worth
 * asking is that the copy of a Q6_K head is 30 % bigger than the format it replaces and wins
 * only on the ALU its walk avoids - and a window is exactly where that ALU is amortised, so
 * the 1.02x a lone row buys said nothing about a window of several. */
static int head_win_env(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_I4_HEAD_WIN");
        cached = e ? strtol(e, NULL, 10) != 0 : 1;
    }
    return (int)cached;
}

void qe_forward_set_head_window(qe_forward *f, int on) {
    if (f)
        f->head_win = on ? 1 : 0;
}

int qe_forward_get_head_window(const qe_forward *f) {
    return f ? f->head_win : 0;
}

void qe_forward_set_head_copy(qe_forward *f, int which) {
    if (!f)
        return;
    if (which == QE_HEAD_Q8 && !f->hq_w)
        which = QE_HEAD_STORED;
    if (which == QE_HEAD_I4 && !f->h4_w)
        which = QE_HEAD_STORED;
    f->head_copy = which;
}

int qe_forward_get_head_copy(const qe_forward *f) {
    return f ? f->head_copy : QE_HEAD_STORED;
}

void qe_forward_set_chunk(qe_forward *f, int width) {
    if (!f)
        return;
    f->chunk_width = width < 1 ? 1 : (width > QE_CHUNK_MAX ? QE_CHUNK_MAX : width);
}

int qe_forward_get_chunk(const qe_forward *f) {
    return f->chunk_width;
}

int qe_forward_chunk_max(void) {
    return QE_CHUNK_MAX;
}

/* Off unless QE_MOE_GROUP says otherwise: see qe_moe_desc.no_shared_multi for what
 * the grouped pass does to the order of the additions. */
void qe_forward_set_moe_grouped(qe_forward *f, int on) {
    if (f)
        f->moe_grouped = on ? 1 : 0;
}

int qe_forward_get_moe_grouped(const qe_forward *f) {
    return f->moe_grouped;
}

void qe_forward_set_moe_tile(qe_forward *f, int rows) {
    if (!f)
        return;
    if (rows <= 0 || rows > QE_CHUNK_MAX)
        f->moe_tile = 0;
    else
        f->moe_tile = rows;
}

int qe_forward_get_moe_tile(const qe_forward *f) {
    return f->moe_tile;
}

void qe_forward_set_attn_fused(qe_forward *f, int on) {
    if (f)
        f->attn_fused = on ? 1 : 0;
}

int qe_forward_get_attn_fused(const qe_forward *f) {
    return f ? f->attn_fused : 0;
}

void qe_forward_set_moe_co(qe_forward *f, int on) {
    if (f)
        f->moe_co = on ? 1 : 0;
}

int qe_forward_get_moe_co(const qe_forward *f) {
    return f ? f->moe_co : 0;
}

void qe_forward_set_head_batch(qe_forward *f, int on) {
    if (!f)
        return;
    f->head_batch = on ? 1 : 0;
    f->lg_rows = 0; /* whatever is cached was made by the other walk */
}

int qe_forward_get_head_batch(const qe_forward *f) {
    return f ? f->head_batch : 0;
}

/* A projection for the chunk in flight, all T rows of it: either T ordinary
 * gemv() calls issued back to back against the same tensor, where the L3 serves
 * rows 1..T-1 out of the walk that pulled row 0, or - above QE_PROJ_MB -
 * qe_gemv_multi_t, one sweep of the tensor with the T activations blocked
 * through a core's L2 alongside it. The threshold is why one of those is the
 * default and the other is a switch; qe_forward_set_proj_chunk() moves it inside
 * one open pass, which is how `make forward` A/Bs the two on one load.
 *
 * The first form is bit-identical to T calls of gemv() on the same rows: each
 * y[t][r] comes from the same kernel on the same inputs, which `make prefill`
 * asserts. The second is bit-identical for every type except Q8_0, which the
 * blocked walk hands to qe_gemv_q8_0_multi and which therefore sums in two
 * chains per token instead of the single kernel's four. That difference is
 * bounded and measured (include/qe.h), and it is one reason the switch is off.
 * With a chunk of one neither form is used: proj() takes the ordinary call, so
 * the 100 ms decode step is untouched by any of this.
 *
 * The caller has to know that all T input rows exist and that nothing reads the
 * output rows before this returns, which is why the projections of a block run in
 * their own stage, over the whole chunk, rather than inside a token's pass. x and
 * y are the *bases* of the chunk buffers and xstride, ystride the distance
 * between their rows. */
static void proj(qe_forward *f, const qe_tensor *t, const float *x, long xstride, float *y,
                 long ystride) {
    /* `proj_chunk_bytes == 0` is "never", and it has to be tested: `t->bytes` is
     * unsigned, so `t->bytes < 0` is not a true statement about a threshold of zero,
     * and reading the clause that way made the default threshold block every
     * projection in the model instead of blocking none. It is worth naming what that
     * cost, because it was not small: the default engine ran the shared walk, whose
     * Q8_0 kernel reassociates, while `make forward` printed a verdict computed from
     * two runs of that same one path and a trunk compared against itself. */
    if (!proj_walkable(f, t)) {
        if (f->chunk_T > 1 && f->small_proj_batch) {
            proj_small(f, t, x, xstride, y, ystride);
            return;
        }
        for (int u = 0; u < f->chunk_T; u++)
            gemv(f, t, x + (long)u * xstride, y + (long)u * ystride);
        return;
    }
    /* The recode is worth its price only against a walk big enough to notice, so it goes
     * to the same projections the chunk walk already takes, and it is skipped for a K the
     * panel is not sized for or a K whose 32-value blocks do not divide it, both of which
     * the integer walk would refuse anyway. */
    qe_i8panel panel = { 0 };
    const long K = (long)t->ne[0];
    if (f->o.profile)
        f->walk_mac += (double)t->ne[0] * (double)t->ne[1] * (double)f->chunk_T;
    const int i8 = proj_i8_wanted(f, t);
    if (i8) {
        panel = (qe_i8panel){ .x8 = f->i8x,
                              .s = f->i8s,
                              .p = f->i8pseed,
                              .stride8 = K,
                              .nb = K / 32 };
        panel_job pj = { x, f->i8x, f->i8s, f->i8pseed, xstride, K, f->chunk_T };
        qe_pool_run(panel_fill_task, &pj, f->n_threads);
    }
    /* A pass of a few rows - a verify window, a batched step - is still a walk the bus
     * pays for, so it reads the IQ4_NL copy a lone token reads when one was staged and
     * `QE_I4_MULTI` says this width is narrow enough; the flat Q8_0 copy below is for
     * the widths where the walk is arithmetic and fewer bytes buy nothing. */
    const long di = dq_copy_for(f, t);
    const long fi = (i8 && di < 0) ? flat_copy_for(f, t) : -1;
    const long nb = K / 32;
    gemv_chunk_job j = { t,
                         di >= 0 ? (const void *)((const char *)f->dq_w + f->dq_woff[di])
                                 : qe_tensor_data(f->m, t),
                         x,       y,
                         (long)t->ne[0],
                         di >= 0 ? nb * (long)qe_type_info_of(QE_IQ4_NL)->block_sz
                                 : tensor_row_bytes(t),
                         di >= 0 ? nb : tensor_ldd(t), xstride,
                         ystride,        f->chunk_T,          i8 ? &panel : NULL,
                         fi >= 0 ? (const char *)f->flat_w + f->flat_woff[fi] : NULL,
                         fi >= 0 ? f->flat_s + f->flat_soff[fi] : NULL,
                         di >= 0 ? QE_IQ4_NL : t->type,
                         chunk_dyn_rows(f, (long)t->ne[1]),
                         0 };
    qe_pool_run(gemv_chunk_task, &j, f->n_threads);
}

/* One activation buffer, zero-filled and registered for the free list. */
static void *hugobuf(qe_forward *f, size_t bytes) {
    if (!bytes)
        bytes = 4096;
    void *p = qe_alloc_huge(bytes);
    if (!p)
        return NULL;
    if (f->n_owned >= QE_MAX_OWNED) {
        qe_free_huge(p, bytes);
        return NULL;
    }
    f->own_p[f->n_owned] = p;
    f->own_n[f->n_owned] = bytes;
    f->n_owned++;
    return p;
}

/* qe_alloc_huge rounds every request up to a 2 MiB boundary, which is what
 * makes the memset below stay inside the mapping. Touching the whole buffer
 * here is deliberate: the alternative is taking the fault in the middle of a
 * decode step. */
static void *scratch(qe_forward *f, int slot, size_t bytes) {
    void *p = hugobuf(f, bytes);
    f->sp[slot] = p;
    if (p)
        memset(p, 0, qe_align_up(bytes, 2u << 20));
    return p;
}

/* ------------------------------------------------- the flat requantised dense copies */

/* One matrix requantised from thirty-two scales a row to one.
 *
 * A row's scale is its largest magnitude over 127 and every value of the row is rescaled
 * to it, so the resolution given up is the outlier's: one large value in a row of 2560
 * spends itself on the other 2559. What that costs the head is a question about this
 * checkpoint, and `make forward` answers it with the head in front of it (ideas.md 7.18) -
 * which is also why the walk that reads these is off by default. This pass only makes the
 * copy exactly, to the rounding of one int8, and is never timed: like the load it is part
 * of, the one-time placement of bytes is not a number this repository reports (AGENTS.md 2). */
typedef struct {
    const char *src;
    int8_t *dst;
    float *scales;
    long row_bytes, n_rows, K;
} flat_job;

static void flat_build_task(void *v, int tid, int nthreads) {
    flat_job *j = v;
    long lo, hi;
    qe_slice(j->n_rows, tid, nthreads, &lo, &hi);
    if (lo >= hi)
        return;
    const long nb = j->K / 32;
    for (long r = lo; r < hi; r++) {
        const qe_block_q8_0 *b =
            (const qe_block_q8_0 *)(j->src + (size_t)r * (size_t)j->row_bytes);
        float mx = 0.0f;
        for (long g = 0; g < nb; g++) {
            int m = 0;
            for (int i = 0; i < 32; i++) {
                const int q = b[g].qs[i];
                if (q < 0 ? -q > m : q > m)
                    m = q < 0 ? -q : q;
            }
            const float a = qe_half_to_f32(b[g].d) * (float)m;
            if (a > mx)
                mx = a;
        }
        /* mx == 0 is a row of zeroes, which has no scale worth naming: it quantises to
         * zeroes at scale 0 and multiplies to the same answer the blocks do. */
        j->scales[r] = mx > 0.0f ? mx / 127.0f : 0.0f;
        const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
        int8_t *out = j->dst + (size_t)r * (size_t)j->K;
        for (long g = 0; g < nb; g++) {
            const float s = qe_half_to_f32(b[g].d) * inv;
            for (int i = 0; i < 32; i++) {
                long q = lrintf((float)b[g].qs[i] * s);
                if (q > 127)
                    q = 127;
                if (q < -127)
                    q = -127;
                out[32 * g + i] = (int8_t)q;
            }
        }
    }
}

/* Which tensors get a copy: the Q8_0 matrices a chunk walk reads. The shape does most of
 * the excluding by itself - the routed experts are three-dimensional, one expert per
 * slice, and their walk is a gather a flat tile has nothing to say about (3.9 GiB of this
 * checkpoint's 8.3) - and the embedding table is the other thing that is Q8_0 and is not a
 * walk, which is another 0.6 GiB left out. What remains is 3.3 GiB. */
static int flat_candidate(const qe_forward *f, const qe_tensor *t) {
    return t->type == QE_Q8_0 && t->ndim == 2 && t->ne[1] > 1 && !(t->ne[0] % 32) &&
           !t->n_experts && (long)t->ne[0] <= f->i8k &&
           t != (const qe_tensor *)f->embd;
}

/* What a host has to have free, beyond the copies themselves, for an open that asked for
 * them to go ahead anyway. The engine's own state and scratch are committed by the time
 * this is asked, so the margin is for what the step goes on to touch: the chunk buffers are
 * already in that total, and the rest is the kernel's. Four GiB is what this host keeps
 * spare in practice. */
#define QE_FLAT_MARGIN_MIB 4096

/* Stage every copy, or say with two numbers why it is not happening. Called from the open,
 * after the engine's own allocations, so that the memory it reads as free is the memory a
 * step will not need. */
static void flat_stage(qe_forward *f) {
    const size_t n = qe_tensor_count(f->m);
    long *woff = (long *)malloc(n * sizeof *woff);
    long *soff = (long *)malloc(n * sizeof *soff);
    if (!woff || !soff) {
        free(woff);
        free(soff);
        fprintf(stderr, "qe: QE_I8_FLAT asked for and not built: no room for the map\n");
        f->flat_i8 = 0;
        return;
    }
    for (size_t i = 0; i < n; i++) {
        woff[i] = -1;
        soff[i] = 0;
    }
    size_t wbytes = 0, scount = 0;
    long nten = 0;
    for (size_t i = 0; i < n; i++) {
        const qe_tensor *t = qe_tensor_by_index(f->m, i);
        if (!flat_candidate(f, t))
            continue;
        woff[i] = (long)wbytes;
        soff[i] = (long)scount;
        wbytes += (size_t)t->ne[0] * (size_t)t->ne[1];
        scount += (size_t)t->ne[1];
        nten++;
    }
    const size_t need = (wbytes + scount * sizeof(float)) / (1u << 20) + 2;
    const size_t avail = qe_mem_available_mib();
    if (!nten || need + QE_FLAT_MARGIN_MIB > avail) {
        fprintf(stderr,
                "qe: QE_I8_FLAT asked for and not built: %zu MiB of copies, %zu MiB free at"
                " open (%d MiB kept); walking the Q8_0 blocks instead\n",
                need, avail, QE_FLAT_MARGIN_MIB);
        free(woff);
        free(soff);
        f->flat_i8 = 0;
        return;
    }
    f->flat_woff = woff;
    f->flat_soff = soff;
    f->flat_n = (long)n;
    /* Two arenas rather than two buffers a tensor: the maps above are the only thing that
     * has to know where a matrix's rows are, and a hugobuf per tensor would spend the
     * registry on 500 entries. Nothing memsets these: the pass below writes every byte of
     * both, which is the same touch with less traffic. */
    if (!(f->flat_w = hugobuf(f, wbytes)) ||
        !(f->flat_s = hugobuf(f, scount * sizeof(float)))) {
        fprintf(stderr, "qe: QE_I8_FLAT asked for and not built: %zu MiB of copies would not"
                        " allocate (%zu MiB free at open)\n", need, avail);
        free(woff);
        free(soff);
        f->flat_woff = f->flat_soff = NULL;
        f->flat_n = 0;
        f->flat_i8 = 0;
        return;
    }
    f->flat_bytes = wbytes + scount * sizeof(float);
    for (size_t i = 0; i < n; i++) {
        if (woff[i] < 0)
            continue;
        const qe_tensor *t = qe_tensor_by_index(f->m, i);
        flat_job j = { (const char *)qe_tensor_data(f->m, t),
                       (int8_t *)f->flat_w + (size_t)woff[i],
                       f->flat_s + (size_t)soff[i],
                       tensor_row_bytes(t), (long)t->ne[1], (long)t->ne[0] };
        qe_pool_run(flat_build_task, &j, f->n_threads);
    }
    fprintf(stderr,
            "qe: flat int8 walk on: %ld dense matrices (%zu MiB) requantised to one scale"
            " per output row, %zu MiB free at open\n",
            nten, need, avail);
}

/* ------------------------------------------ the IQ4_NL copies a decode step reads */

/* Which tensors get the decode copy: the same Q8_0 matrices the flat copies are made
 * of, for the same reasons - the routed experts are three-dimensional and the
 * embedding table is a row gather, and neither is a walk a smaller format helps. The
 * one thing this rule drops is the panel's K bound, which is a constraint on the
 * activation panel the flat walk multiplies against and has nothing to do with a copy
 * read by a float activation. */
static int dq_candidate(const qe_forward *f, const qe_tensor *t) {
    return t->type == QE_Q8_0 && t->ndim == 2 && t->ne[1] > 1 && !(t->ne[0] % 32) &&
           !t->n_experts && t != (const qe_tensor *)f->embd &&
           /* The encoder below counts a row as its own width of blocks and the walk reads
            * the copy the same way, so a tensor whose rows carry padding is not one it can
            * be handed. */
           t->bytes == (size_t)(t->ne[0] / 32) * (size_t)t->ne[1] * sizeof(qe_block_q8_0);
}

/* Which of the two classes named above this matrix is, out of its name and the block it
 * belongs to. A tensor outside a block - the head's mixer - is not an edge by this rule,
 * because there is no block before or after it for an error to travel through. */
static unsigned char dq_class(const qe_forward *f, const qe_tensor *t) {
    unsigned char c = 0;
    long il = -1;
    if (strstr(t->name, "hc_"))
        c |= QE_DCLASS_HC;
    if (sscanf(t->name, "blk.%ld.", &il) == 1 && il >= 0 &&
        (il == 0 || il >= (long)f->n_layer - 2))
        c |= QE_DCLASS_EDGE;
    return c;
}

/* One matrix, from thirty-two values and an fp16 scale to thirty-two values of a
 * codebook and an fp16 scale. The search is `qe_quant_iq4_nl_q8_0`'s and is the same
 * one `qe_quant_iq4_nl` runs on floats, so a copy built here is what the file would
 * have held had it been written in this format - which is the claim `make fast` checks
 * by building both from the same row. Like every other part of the load this is never
 * timed (AGENTS.md 2); it is a pass over bytes that are already resident. */
typedef struct {
    const qe_block_q8_0 *src;
    qe_block_iq4_nl *dst;
    long n_rows, nb;
} dq_job;

static void dq_build_task(void *v, int tid, int nthreads) {
    dq_job *j = v;
    long lo, hi;
    qe_slice(j->n_rows, tid, nthreads, &lo, &hi);
    if (lo >= hi)
        return;
    for (long r = lo; r < hi; r++)
        qe_quant_iq4_nl_q8_0(j->src + (size_t)r * (size_t)j->nb, j->dst + (size_t)r * (size_t)j->nb,
                             j->nb);
}

static void dq_stage(qe_forward *f) {
    const size_t n = qe_tensor_count(f->m);
    long *woff = (long *)malloc(n * sizeof *woff);
    unsigned char *wcls = (unsigned char *)calloc(n, sizeof *wcls);
    if (!woff || !wcls) {
        fprintf(stderr, "qe: QE_I4_DENSE asked for and not built: no room for the map\n");
        free(wcls);
        f->dq_i4 = 0;
        return;
    }
    for (size_t i = 0; i < n; i++)
        woff[i] = -1;
    size_t wbytes = 0, sbytes = 0;
    long nten = 0, nhc = 0, nedge = 0;
    for (size_t i = 0; i < n; i++) {
        const qe_tensor *t = qe_tensor_by_index(f->m, i);
        if (!dq_candidate(f, t))
            continue;
        woff[i] = (long)wbytes;
        wcls[i] = dq_class(f, t);
        nhc += (wcls[i] & QE_DCLASS_HC) != 0;
        nedge += (wcls[i] & QE_DCLASS_EDGE) != 0;
        wbytes += (size_t)(t->ne[0] / 32) * (size_t)t->ne[1] * sizeof(qe_block_iq4_nl);
        sbytes += t->bytes;
        nten++;
    }
    const size_t need = wbytes / (1u << 20) + 2;
    const size_t avail = qe_mem_available_mib();
    if (!nten || need + QE_FLAT_MARGIN_MIB > avail) {
        fprintf(stderr,
                "qe: QE_I4_DENSE asked for and not built: %zu MiB of copies, %zu MiB free at"
                " open (%d MiB kept); a step walks the Q8_0 rows instead\n",
                need, avail, QE_FLAT_MARGIN_MIB);
        free(woff);
        free(wcls);
        f->dq_i4 = 0;
        return;
    }
    f->dq_woff = woff;
    f->dq_wcls = wcls;
    f->dq_n = (long)n;
    /* One arena, and no memset: the pass below writes every byte of it. */
    if (!(f->dq_w = hugobuf(f, wbytes))) {
        fprintf(stderr,
                "qe: QE_I4_DENSE asked for and not built: %zu MiB of copies would not"
                " allocate (%zu MiB free at open)\n",
                need, avail);
        free(woff);
        free(wcls);
        f->dq_woff = NULL;
        f->dq_wcls = NULL;
        f->dq_n = 0;
        f->dq_i4 = 0;
        return;
    }
    f->dq_bytes = wbytes;
    for (size_t i = 0; i < n; i++) {
        if (woff[i] < 0)
            continue;
        const qe_tensor *t = qe_tensor_by_index(f->m, i);
        dq_job j = { (const qe_block_q8_0 *)qe_tensor_data(f->m, t),
                     (qe_block_iq4_nl *)((char *)f->dq_w + (size_t)woff[i]),
                     (long)t->ne[1], (long)t->ne[0] / 32 };
        qe_pool_run(dq_build_task, &j, f->n_threads);
    }
    fprintf(stderr,
            "qe: decode IQ4_NL walk on: %ld dense matrices (%zu MiB of Q8_0) requantised to"
            " %zu MiB of codebook rows, %zu MiB free at open; %ld of them mixers, %ld at the"
            " trunk's ends, a step reads %s\n",
            nten, sbytes / (1u << 20), need, avail, nhc, nedge,
            f->dq_tier == (QE_DCLASS_HC | QE_DCLASS_EDGE)
                ? "all of them"
                : f->dq_tier == QE_DCLASS_EDGE ? "all but the mixers"
                             : f->dq_tier == QE_DCLASS_HC ? "all but the trunk's ends"
                                                          : "neither of those two classes");
}

/* ------------------------------------------ the output head's two smaller copies */

/* One row of the head at a time. A Q6_K block is 256 values wide and both encoders below
 * consume one of them whole, so a row - which the test in `head_stage` makes a whole
 * number of them - is the smallest unit that can be built independently, and one row per
 * unit of the slice means neither encoder ever crosses into the row another thread has.
 * Like every other part of the load this pass is never timed (AGENTS.md 2). */
typedef struct {
    const qe_block_q6_k *src;
    qe_block_iq4_nl *i4;
    qe_block_q8_0 *q8;
    long n_rows, nb256;
} head_job;

static void head_build_task(void *v, int tid, int nthreads) {
    head_job *j = v;
    long lo, hi;
    qe_slice(j->n_rows, tid, nthreads, &lo, &hi);
    if (lo >= hi)
        return;
    const size_t s = (size_t)j->nb256;
    if (j->i4)
        for (long r = lo; r < hi; r++)
            qe_quant_iq4_nl_q6_k(j->src + (size_t)r * s, j->i4 + (size_t)r * s * 8, j->nb256);
    if (j->q8)
        for (long r = lo; r < hi; r++)
            qe_quant_q8_0_q6_k(j->src + (size_t)r * s, j->q8 + (size_t)r * s * 8, j->nb256);
}

/* Stage both copies and record which one the env asked a step to read. Neither is timed,
 * like everything else in the load, and neither is conditional on the reading: the A/B
 * that decides between them walks all three in the load that staged them, and a copy built
 * only for the setting the process opened with could not be raced against its own
 * alternative. */
/* A head a copy can be built of: Q6_K, rows that are a whole number of 256-value blocks,
 * and no padding inside a row, because the pass below walks it by rows from both ends at
 * once and a padded stride would have it encode the wrong weights and print nothing. */
static int head_buildable(const qe_tensor *t) {
    if (!t || t->type != QE_Q6_K || t->ne[0] % 256 || t->ne[1] <= 0)
        return 0;
    return t->bytes == (size_t)(t->ne[0] / 256) * (size_t)t->ne[1] * sizeof(qe_block_q6_k);
}

static size_t head_i4_bytes(const qe_tensor *t) {
    return (size_t)(t->ne[0] / 32) * (size_t)t->ne[1] * sizeof(qe_block_iq4_nl);
}

static size_t head_q8_bytes(const qe_tensor *t) {
    return (size_t)(t->ne[0] / 32) * (size_t)t->ne[1] * sizeof(qe_block_q8_0);
}

/* Both copies of one head, in two fresh arenas, or nothing in either. The data pointer is
 * asked for out of the tensor's own volume, so a draft head in a second file is a caller
 * like any other. */
static void head_build(qe_forward *f, const qe_tensor *t, void **i4w, void **q8w) {
    const long K = (long)t->ne[0];
    void *a = hugobuf(f, head_i4_bytes(t));
    void *b = hugobuf(f, head_q8_bytes(t));
    if (!a || !b) {
        *i4w = NULL;
        *q8w = NULL;
        return;
    }
    head_job j = { (const qe_block_q6_k *)qe_tensor_data(f->m, t), (qe_block_iq4_nl *)a,
                   (qe_block_q8_0 *)b, (long)t->ne[1], K / 256 };
    qe_pool_run(head_build_task, &j, f->n_threads);
    *i4w = a;
    *q8w = b;
}

/* Whether the trunk's head and the draft head are the same weights. A nextn head is
 * trained against its target's own output matrix, so two files carrying the same bytes
 * twice is the usual answer, and when it is, one pair of copies serves both walks. */
static int head_same(const qe_forward *f, const qe_tensor *a, const qe_tensor *b) {
    const void *da = qe_tensor_data(f->m, a), *db = qe_tensor_data(f->m, b);
    return a->bytes == b->bytes && da && db && !memcmp(da, db, a->bytes);
}

/* Stage both copies and record which one the env asked a step to read. Neither is timed,
 * like everything else in the load, and neither is conditional on the reading: the A/B
 * that decides between them walks all three in the load that staged them, and a copy built
 * only for the setting the process opened with could not be raced against its own
 * alternative. */
static void head_stage(qe_forward *f) {
    const qe_tensor *t = f->outw;
    f->head_copy = QE_HEAD_STORED;
    if (!head_buildable(t))
        return;
    const size_t i4 = head_i4_bytes(t), q8 = head_q8_bytes(t);
    /* The draft head is walked once per draft row, which is what makes it worth the same
     * two copies - but only when it is not the trunk's head again, shape and bytes. */
    const qe_tensor *mt = f->mtp_outw;
    const int draft = mt && mt != t && head_buildable(mt) && mt->bytes == t->bytes &&
                      mt->ne[0] == t->ne[0] && mt->ne[1] == t->ne[1] && !head_same(f, t, mt);
    const size_t need = ((i4 + q8) * (draft ? 2 : 1)) / (1u << 20) + 2;
    const size_t avail = qe_mem_available_mib();
    if (need + QE_FLAT_MARGIN_MIB > avail) {
        fprintf(stderr,
                "qe: output head copies not built: %zu MiB of them, %zu MiB free at open"
                " (%d MiB kept); a step walks the head as the file wrote it\n",
                need, avail, QE_FLAT_MARGIN_MIB);
        return;
    }
    head_build(f, t, &f->h4_w, &f->hq_w);
    if (!f->h4_w || !f->hq_w) {
        fprintf(stderr,
                "qe: output head copies not built: %zu MiB would not allocate"
                " (%zu MiB free at open)\n",
                need, avail);
        f->h4_w = NULL;
        f->hq_w = NULL;
        f->head_copy = QE_HEAD_STORED;
        return;
    }
    if (draft) {
        head_build(f, mt, &f->mh4_w, &f->mhq_w);
        if (!f->mh4_w || !f->mhq_w) {
            f->mh4_w = NULL;
            f->mhq_w = NULL;
            fprintf(stderr, "qe: the draft head's own copies would not allocate; a draft row"
                            " walks its head as its file wrote it\n");
        }
    } else if (mt && mt != t && head_buildable(mt) && mt->bytes == t->bytes &&
               mt->ne[0] == t->ne[0] && mt->ne[1] == t->ne[1]) {
        f->mh4_w = f->h4_w;
        f->mhq_w = f->hq_w;
    }
    f->head_copy = head_copy_env();
    fprintf(stderr,
            "qe: output head copies on: %ld rows of %ld as %zu MiB of IQ4_NL and %zu MiB of"
            " Q8_0%s, %zu MiB free at open; a step reads %s%s\n",
            (long)t->ne[1], (long)t->ne[0], i4 / (1u << 20), q8 / (1u << 20),
            mt && mt != t ? (draft ? " (the draft head's are its own)"
                                    : " (the draft head is the same bytes, so one pair serves both)")
                          : "",
            avail,
            f->head_copy == QE_HEAD_Q8 ? "Q8_0"
            : f->head_copy == QE_HEAD_I4 ? "IQ4_NL"
                                         : "Q6_K",
            /* Which walks a verify window is allowed on is the one reading here that has no
             * copy line of its own, and "is the engine wired for it" should be answerable
             * from a log rather than from this file. */
            f->head_copy == QE_HEAD_STORED
                ? ""
                : f->head_win ? ", a window's rows read it too"
                              : ", a verify window does not");
}

/* Point the component functions at token t of the chunk in flight.
 *
 * Every one of them reads the residual streams, the mixer's output and a block's
 * output through f->res, f->xn and f->y rather than through an argument, so
 * moving these three pointers between rows of the chunk buffers is all a
 * multi-token walk needs: the arithmetic of a token is the same code, in the
 * same order, with the same bytes at the same addresses. Row 0 is what a single
 * step uses, which is why a chunk of one is not a special case but the same
 * path. */
static void use_row(qe_forward *f, int t) {
    f->res = f->cres + (size_t)t * (size_t)f->hc_dim;
    f->xn = f->cxn + (size_t)t * (size_t)f->K;
    f->y = f->cy + (size_t)t * (size_t)f->K;
    f->inj = f->cinj + (size_t)t * (size_t)f->HC;
}

static void note_missing(char *err, size_t errn, const char *name) {
    snprintf(err, errn, "checkpoint has no tensor named %s", name);
}

/* -------------------------------------------------------------- construction */

static const qe_tensor *want(qe_model *m, const char *fmt, int layer, char *err, size_t errn,
                             const qe_tensor **out) {
    char name[QE_MAX_NAME];
    snprintf(name, sizeof name, fmt, layer);
    const qe_tensor *t = qe_tensor_by_name(m, name);
    if (!t)
        note_missing(err, errn, name);
    *out = t;
    return t;
}

static int want_f(qe_model *m, const char *fmt, int layer, char *err, size_t errn,
                  const qe_tensor **out, qe_type type) {
    if (!want(m, fmt, layer, err, errn, out) || (*out)->type != type) {
        if (!*out)
            return -1;
        char name[QE_MAX_NAME];
        snprintf(name, sizeof name, fmt, layer);
        snprintf(err, errn, "%s is %s, this driver expects %s", name,
                 qe_type_info_of((*out)->type)->name, qe_type_info_of(type)->name);
        return -1;
    }
    return 0;
}

/* The file's value for a key, or the fallback when it does not carry one.
 * qe_model_get_int returns 0 on success, which is the opposite of the sense a
 * truthiness test gives it, and read the wrong way round it fails quietly: every
 * dimension below keeps its fallback, and on a checkpoint whose shape happens
 * to match the documented one the pass still runs - it is just reading a model
 * nobody asked it to read. Two helpers, one sense, no condition left inline. */
static int dim_int(qe_model *m, const char *key, int fallback) {
    long long v;
    return qe_model_get_int(m, key, &v) == 0 ? (int)v : fallback;
}

static float dim_float(qe_model *m, const char *key, float fallback) {
    double d;
    return qe_model_get_float(m, key, &d) == 0 ? (float)d : fallback;
}

/* Are the keys and values of the full-attention layers 8 bits or 32? Eight by
 * default: on this checkpoint they are 4 KiB per cell per layer either way up,
 * and the whole reason the number matters is that a conversation with a coding
 * agent in it does not stop growing. QE_KV_F32 set to anything but 0 asks for the
 * f32 cache, which is what `make score` and `make cmp` compare against when the
 * question on the table is how much a cache quantization costs rather than what it
 * saves; the A/B that answers that is in docs/DESIGN.md §5. */
static int kv_quant_wanted(const qe_forward_opts *o) {
    if (o->kv_f32)
        return 0;
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QE_KV_F32");
        v = (e && *e && *e != '0') ? 0 : 1;
    }
    return v;
}

/* The fused attention's walk is one unit per chunk of the selection per KV group,
 * so a chunk is the knob that trades parallelism for partials: too short and the
 * combine reads more than the walk wrote, too long and nineteen cores watch two
 * threads work. 32 cells is the point where a unit is a few thousand value rows'
 * worth of arithmetic, which is several times the dispatch handshake it is paid
 * with. */
#define QE_ATTN_CELL_MIN 32

/* Units in the widest split the dispatch asks for: one chunk per worker of each
 * KV group. The partials are sized for exactly this, and the dispatch is written
 * so that it can never ask for more. */
static int attn_units_bound(int n_threads, int nkvh) {
    if (nkvh < 1)
        nkvh = 1;
    const int chunks = (n_threads + nkvh - 1) / nkvh;
    return (chunks < 1 ? 1 : chunks) * nkvh;
}

/* One pass over the selection per KV group, softmax inside the walk, is the
 * default. QE_ATTN_FUSED=0 asks for the head-at-a-time form it replaces, which is
 * how the two are compared from the command line: the same selection, the same
 * rows, one variable between them. */
static int attn_fused_wanted(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QE_ATTN_FUSED");
        v = (e && *e && *e == '0') ? 0 : 1;
    }
    return v;
}

/* ------------------------------------------------------------- draft head */

/* A dimension the head's file has to agree with the trunk's on, and the value it
 * has to agree with. There is no fallback here: a head that disagrees is a head
 * for another model, and the alternative to comparing is running one block of
 * another model's weights through this pass, which produces logits rather than an
 * error. */
static int same_int(qe_model *a, qe_model *b, const char *key, char *err, size_t errn) {
    long long va = 0, vb = 0;
    int ha = qe_model_get_int(a, key, &va), hb = qe_model_get_int(b, key, &vb);
    if (ha && hb)
        return 0; /* neither file carries the key: nothing to disagree about */
    if (ha)
        snprintf(err, errn, "%s is in the trunk but not in the draft head", key);
    else if (hb)
        snprintf(err, errn, "%s is in the draft head but not in the trunk", key);
    else if (va != vb)
        snprintf(err, errn, "%s is %lld in the trunk and %lld in the draft head", key, va, vb);
    else
        return 0;
    return -1;
}

static int same_float(qe_model *a, qe_model *b, const char *key, char *err, size_t errn) {
    double va = 0, vb = 0;
    int ha = qe_model_get_float(a, key, &va), hb = qe_model_get_float(b, key, &vb);
    if (ha && hb)
        return 0;
    if (ha || hb || va != vb) {
        snprintf(err, errn, "%s differs between the trunk and the draft head", key);
        return -1;
    }
    return 0;
}

/* Open the head's nextn block as layer `f->n_layer` and its expert pool as
 * `f->moe[f->n_layer]`, from a file that arrived on its own.
 *
 * Two things make this a function of its own rather than another turn of the
 * trunk's layer loop. It reads a second model, so every tensor fetch is against
 * `mh` and every shape question is a comparison rather than a read; and it must
 * never be fatal. A head that fails to open costs a fast path, not the engine:
 * the trunk is already resident and already correct, so the answer here is to
 * say what was missing and continue with `mtp_ready` 0. */
static int mtp_layer_open(qe_forward *f, qe_model *mh, int *qsa_slots, char *err, size_t errn) {
    const int il = f->n_layer;
    qe_layer *L = &f->L[il];
    char name[QE_MAX_NAME];

    snprintf(err, errn, "the draft head has no nextn block to open");
    long long nnext = 0;
    if (qe_model_get_int(mh, "qwen4exp.nextn_predict_layers", &nnext) || nnext != 1) {
        snprintf(err, errn, "nextn_predict_layers is %lld: this driver drafts with exactly "
                            "one nextn block, which is what the head files ship",
                 nnext);
        return -1;
    }

    static const char *ints[] = {
        "qwen4exp.embedding_length",
        "qwen4exp.hyper_connection.count",
        "qwen4exp.hyper_connection.low_rank",
        "qwen4exp.attention.head_count",
        "qwen4exp.attention.head_count_kv",
        "qwen4exp.attention.key_length",
        "qwen4exp.attention.indexer.head_count",
        "qwen4exp.attention.indexer.key_length",
        "qwen4exp.expert_count",
        "qwen4exp.expert_used_count",
        "qwen4exp.expert_feed_forward_length",
        "qwen4exp.expert_shared_feed_forward_length",
    };
    static const char *floats[] = {
        "qwen4exp.rope.freq_base",
        "qwen4exp.rope.dimension_count",
        "qwen4exp.attention.layer_norm_rms_epsilon",
    };
    for (size_t i = 0; i < sizeof ints / sizeof *ints; i++)
        if (same_int(f->m, mh, ints[i], err, errn))
            return -1;
    for (size_t i = 0; i < sizeof floats / sizeof *floats; i++)
        if (same_float(f->m, mh, floats[i], err, errn))
            return -1;

    /* The head's own embedding table goes unused: the fold reads the trunk's
     * Q8_0 rows, which are the same rows with fewer bits lost. Its output table
     * is the whole head's logits and has to be the trunk's vocabulary. */
    snprintf(name, sizeof name, "blk.%d.ple_key.weight", il);
    if (qe_tensor_by_name(mh, name)) {
        snprintf(err, errn, "the draft head has a PLE block: this pass keeps the head "
                            "recurrence-free, so its cache is one attention slot");
        return -1;
    }

    /* The attention block. Types are not checked: the point of a head that
     * arrives as its own file is that it is quantised however its author
     * quantised it, and every walk below goes through qe_gemv_t, which answers
     * for the formats it can read. Widths are checked, because a head whose
     * matrices are the wrong width is a head for another model and the caches
     * are already sized from the trunk's numbers. */
    if (!want(mh, "blk.%d.attn_q.weight", il, err, errn, &L->aq) ||
        !want(mh, "blk.%d.attn_k.weight", il, err, errn, &L->ak) ||
        !want(mh, "blk.%d.attn_v.weight", il, err, errn, &L->av) ||
        !want(mh, "blk.%d.attn_output.weight", il, err, errn, &L->aout) ||
        !want(mh, "blk.%d.attn_q_norm.weight", il, err, errn, &L->aqn) ||
        !want(mh, "blk.%d.attn_k_norm.weight", il, err, errn, &L->akn) ||
        !want(mh, "blk.%d.indexer.q_proj.weight", il, err, errn, &L->ixq) ||
        !want(mh, "blk.%d.indexer.k_proj.weight", il, err, errn, &L->ixk) ||
        !want(mh, "blk.%d.indexer.q_norm.weight", il, err, errn, &L->ixqn) ||
        !want(mh, "blk.%d.indexer.k_norm.weight", il, err, errn, &L->ixkn))
        return -1;

    /* Shape for shape against the trunk's own attention layer, and not against `K`:
     * half of these are not K wide on their input side at all - `attn_output` walks
     * from the concatenated heads down to K - so the claim being checked is the one
     * that matters, that this is a block that reads these activations, and the way it
     * can fail is by being a block for another model. */
    const qe_layer *ref = NULL;
    for (int i = f->n_layer - 1; i >= 0 && !ref; i--)
        if (f->L[i].qsa)
            ref = &f->L[i];
    if (!ref) {
        snprintf(err, errn,
                 "the trunk has no attention layer, so the draft head has nothing to be "
                 "the same shape as");
        return -1;
    }
    const qe_tensor *mine[6] = { L->aq, L->ak, L->av, L->aout, L->ixq, L->ixk };
    const qe_tensor *theirs[6] = { ref->aq, ref->ak, ref->av, ref->aout, ref->ixq, ref->ixk };
    for (int i = 0; i < 6; i++)
        if (mine[i]->ne[0] != theirs[i]->ne[0] || mine[i]->ne[1] != theirs[i]->ne[1]) {
            snprintf(err, errn,
                     "the draft head's blk.%d attention is %llx x %llx where the trunk's is"
                     " %llx x %llx: this block reads the trunk's activations",
                     il, (unsigned long long)mine[i]->ne[0], (unsigned long long)mine[i]->ne[1],
                     (unsigned long long)theirs[i]->ne[0],
                     (unsigned long long)theirs[i]->ne[1]);
            return -1;
        }
    if ((int)L->ixqn->ne[0] != f->idx_dim || (int)L->ixkn->ne[0] != f->idx_dim ||
        (int)L->aqn->ne[0] != f->head_dim || (int)L->akn->ne[0] != f->head_dim) {
        snprintf(err, errn,
                 "the draft head's norms are %llx and %llx wide against a head of %d and "
                 "an indexer key of %d",
                 (unsigned long long)L->aqn->ne[0], (unsigned long long)L->ixqn->ne[0],
                 f->head_dim, f->idx_dim);
        return -1;
    }

    const char *which[2] = { "attn", "ffn" };
    for (int h = 0; h < 2; h++) {
        char fmt[64];
        snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_norm.weight", which[h]);
        if (!want(mh, fmt, il, err, errn, &L->hc_norm[h]))
            return -1;
        snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_down.weight", which[h]);
        if (!want(mh, fmt, il, err, errn, &L->hc_down[h]))
            return -1;
        snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_up.weight", which[h]);
        if (!want(mh, fmt, il, err, errn, &L->hc_up[h]))
            return -1;
        snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_inject.weight", which[h]);
        if (!want(mh, fmt, il, err, errn, &L->hc_inject[h]))
            return -1;
    }
    if (!want(mh, "blk.%d.ffn_gate_inp.weight", il, err, errn, &L->router) ||
        !want(mh, "blk.%d.ffn_gate_exps.weight", il, err, errn, &L->ex_g) ||
        !want(mh, "blk.%d.ffn_up_exps.weight", il, err, errn, &L->ex_u) ||
        !want(mh, "blk.%d.ffn_down_exps.weight", il, err, errn, &L->ex_d) ||
        !want(mh, "blk.%d.ffn_gate_shexp.weight", il, err, errn, &L->sh_gate) ||
        !want(mh, "blk.%d.ffn_up_shexp.weight", il, err, errn, &L->sh_up) ||
        !want(mh, "blk.%d.ffn_down_shexp.weight", il, err, errn, &L->sh_down) ||
        !want(mh, "blk.%d.ffn_gate_inp_shexp.weight", il, err, errn, &L->sh_inp))
        return -1;
    if ((int)L->router->type != QE_F32) {
        snprintf(err, errn, "the draft head's router is %s; the router shadow is built "
                            "from f32 rows, as the trunk's is",
                 qe_type_info_of(L->router->type)->name);
        return -1;
    }

    /* The fold in front and the mixer behind. */
    if (!want(mh, "blk.%d.nextn.eh_proj.weight", il, err, errn, &f->mtp_eh) ||
        !want(mh, "blk.%d.nextn.enorm.weight", il, err, errn, &f->mtp_enorm) ||
        !want(mh, "blk.%d.nextn.hnorm.weight", il, err, errn, &f->mtp_hnorm) ||
        !want(mh, "blk.%d.nextn.hc_head_norm.weight", il, err, errn, &f->mtp_hc_norm) ||
        !want(mh, "blk.%d.nextn.hc_head_down.weight", il, err, errn, &f->mtp_hc_down) ||
        !want(mh, "blk.%d.nextn.hc_head_up.weight", il, err, errn, &f->mtp_hc_up))
        return -1;
    f->mtp_outw = qe_tensor_by_name(mh, "output.weight");
    if (!f->mtp_outw) {
        snprintf(err, errn, "the draft head has no output.weight to take logits with");
        return -1;
    }
    if ((int)f->mtp_eh->ne[0] != 2 * f->K || (int)f->mtp_eh->ne[1] != f->K) {
        snprintf(err, errn, "nextn.eh_proj is %llx x %llx; the fold concatenates two %d-wide "
                            "halves and projects to %d",
                 (unsigned long long)f->mtp_eh->ne[0], (unsigned long long)f->mtp_eh->ne[1],
                 f->K, f->K);
        return -1;
    }
    if ((int)f->mtp_hnorm->ne[0] != f->hc_dim || (int)f->mtp_enorm->ne[0] != f->K ||
        (int)f->mtp_hc_norm->ne[0] != f->hc_dim || f->mtp_outw->ne[1] != f->embd->ne[1] ||
        (int)f->mtp_outw->ne[0] != f->K) {
        snprintf(err, errn,
                 "the draft head's norms or output (%llx, %llx, %llx for %d values) do not "
                 "match a %d-wide model with a %llx-entry vocabulary",
                 (unsigned long long)f->mtp_hnorm->ne[0], (unsigned long long)f->mtp_enorm->ne[0],
                 (unsigned long long)f->mtp_hc_norm->ne[0], f->K, f->K,
                 (unsigned long long)f->embd->ne[1]);
        return -1;
    }

    /* The fold's concatenation is one strided source, and a projection walk wants
     * rows it can stride rather than rows it has to gather, so the head keeps its
     * own two-buffer front end: `mcat` is [e_norm | h_norm] per stream per row and
     * `mhin` is the residual the block runs on, which is where a draft row's h
     * goes before anything else reads it. */
    if (!f->mcat)
        f->mcat = (float *)hugobuf(f, (size_t)QE_CHUNK_MAX * (size_t)f->HC *
                                          (size_t)(2 * f->K) * sizeof(float));
    if (!f->mhin)
        f->mhin = (float *)hugobuf(f, (size_t)QE_CHUNK_MAX * (size_t)f->hc_dim * sizeof(float));
    if (!f->mout)
        f->mout = (float *)hugobuf(f, (size_t)QE_CHUNK_MAX * (size_t)f->K * sizeof(float));
    if (!f->mpend)
        f->mpend = (float *)hugobuf(f, (size_t)f->hc_dim * sizeof(float));
    if (!f->mcat || !f->mhin || !f->mout || !f->mpend) {
        snprintf(err, errn, "out of memory for the draft head's front end");
        return -1;
    }

    f->mtp_back = mtp_back_env();

    /* One more expert pool, with the same bf16 router shadow the trunk's layers
     * get, for the same reason: the router is read whole per token. */
    {
        qe_moe_desc d = { .gate = qe_tensor_data(mh, L->ex_g),
                          .up = qe_tensor_data(mh, L->ex_u),
                          .down = qe_tensor_data(mh, L->ex_d),
                          .gate_stride = L->ex_g->bytes / L->ex_g->ne[2],
                          .up_stride = L->ex_u->bytes / L->ex_u->ne[2],
                          .down_stride = L->ex_d->bytes / L->ex_d->ne[2],
                          .t_gate = L->ex_g->type,
                          .t_up = L->ex_u->type,
                          .t_down = L->ex_d->type,
                          .router = qe_tensor_data(mh, L->router),
                          .n_experts = f->n_experts,
                          .N = f->n_ff_exp,
                          .K = f->K,
                          .top_k = f->n_experts_active,
                          .n_threads = f->n_threads };
        size_t shadow = (size_t)f->n_experts * (size_t)f->K * 2;
        void *sh = hugobuf(f, shadow);
        if (!sh) {
            snprintf(err, errn, "out of memory for the draft head's router shadow");
            return -1;
        }
        f->moe_shadow[il] = sh;
        qe_bf16_from_f32(qe_tensor_data(mh, L->router), sh,
                         (size_t)f->n_experts * (size_t)f->K);
        d.router_bf16 = sh;
        f->moe[il] = qe_moe_create(&d);
        if (!f->moe[il]) {
            snprintf(err, errn,
                     "the draft head's expert pool (%s gate, %s up, %s down) has no "
                     "dispatch here",
                     qe_type_info_of(d.t_gate)->name, qe_type_info_of(d.t_up)->name,
                     qe_type_info_of(d.t_down)->name);
            return -1;
        }
    }

    L->qsa = 1;
    L->ple = 0;
    L->gdn_slot = -1;
    L->qsa_slot = (*qsa_slots)++; /* last: everything above has to hold before a
                                    * cache is sized for this slot */
    f->mh = mh;
    f->mtp_ready = 1;
    err[0] = 0;
    return 0;
}

qe_forward *qe_forward_open(qe_model *m, const qe_forward_opts *opts, char *err, size_t errn) {

    if (!m) {
        snprintf(err, errn, "no model");
        return NULL;
    }
    qe_forward *f = calloc(1, sizeof *f);
    if (!f) {
        snprintf(err, errn, "out of memory");
        return NULL;
    }
    f->m = m;
    f->proj_chunk_bytes = proj_chunk_env();
    f->proj_i8 = proj_i8_env();
    f->flat_i8 = flat_i8_env();
    f->dq_i4 = dq_i4_env();
    f->dq_multi = dq_multi_env();
    f->stage_pf = stage_pf_env();
    f->step_fuse = step_fuse_env();
    f->dyn_bytes = dyn_env();
    {
        const char *e = getenv("QE_I8_STEP");
        f->step_i8 = e && *e ? atoi(e) : QE_I8_STEP_DEFAULT;
        e = getenv("QE_CHUNK_DYN");
        f->chunk_dyn = e && *e ? atoi(e) : QE_CHUNK_DYN_DEFAULT;
        e = getenv("QE_DELTA_SIMD");
        f->delta_simd = e && *e ? atoi(e) : 1;
        e = getenv("QE_ROWS_PAR");
        f->rows_par = e && *e ? atoi(e) : 1;
    }
    f->dq_tier = dq_tier_env();
    f->head_win = head_win_env();
    f->small_proj_batch = small_proj_env();
    f->exp2_simd = exp2_simd_env();
    f->chunk_width = prefill_chunk();
    f->moe_grouped = moe_group_env();
    f->moe_tile = moe_tile_env();
    f->moe_co = moe_co_env();
    f->head_batch = head_batch_env();
    f->mtp_warm = mtp_warm_env();
    f->attn_fused = attn_fused_wanted();
    if (opts)
        f->o = *opts;
    else {
        memset(&f->o, 0, sizeof f->o);
        f->o.n_ctx = 4096;
    }
    if (f->o.n_ctx <= 0)
        f->o.n_ctx = 4096;
    /* A batch is bounded by QE_MAX_SEQ and by having a cell per sequence to put one
     * in; both are refused with the two numbers rather than quietly serving fewer
     * sequences than the caller asked to. And a draft head is a second sequence - it
     * continues the one the trunk is on, position and all - so a pass driving several
     * sequences does not open one. `--batch` and `--mtp` are therefore alternative
     * answers to "how does this host serve more tokens", and the head's own arms
     * (make head) stay the way they were measured. */
    if (f->o.n_seq > QE_MAX_SEQ) {
        snprintf(err, errn, "n_seq %d above QE_MAX_SEQ %d", f->o.n_seq, QE_MAX_SEQ);
        qe_forward_close(f);
        return NULL;
    }
    if (f->o.n_seq > 1 && f->o.n_ctx / f->o.n_seq < 8) {
        snprintf(err, errn,
                 "%d cells split %d ways is %d a sequence, below the 8 a sequence needs",
                 f->o.n_ctx, f->o.n_seq, f->o.n_ctx / f->o.n_seq);
        qe_forward_close(f);
        return NULL;
    }
    if (f->o.n_seq > 1 && f->o.mtp_model) {
        fprintf(stderr, "qe: draft head not opened: a pass driving %d sequences "
                        "has no single sequence for the head to continue\n",
                f->o.n_seq);
        f->o.mtp_model = NULL;
    }
    /* The command line asks for the other walk; the environment says which one the
     * engine starts on. Both are the same switch, and neither is read again. */
    if (f->o.attn_head_walk)
        f->attn_fused = 0;
    f->n_threads = f->o.n_threads > 0 ? f->o.n_threads : qe_pool_size();
    /* Asking for 20 threads of a pool that was never started is not a slowdown
     * worth having: qe_pool_run() answers a request it cannot fill by running
     * the slice itself, so the pass is correct, deterministic, six times slower,
     * and reports the thread count it was asked for. A step that takes half a
     * second instead of a tenth of one gets noticed eventually; noticing it here
     * is cheaper. */
    if (f->n_threads > qe_pool_size()) {
        snprintf(err, errn,
                 "%d threads requested but the pool has %d: call qe_pool_init(%d, NULL, 0) "
                 "before opening a forward pass",
                 f->n_threads, qe_pool_size(), f->n_threads);
        free(f);
        return NULL;
    }
    /* The width of a lone token's walk: the pass width, unless somebody asks for another
     * one, which is the question the comment above `walk_threads_env` has the table for.
     * It is clamped to the pass width rather than refused above it, because a walk can only
     * spread over threads the pass has, and the pool answers a width it cannot fill by
     * walking on the threads it has. */
    {
        const int env = walk_threads_env();
        int w = f->o.walk_threads > 0 ? f->o.walk_threads : 0;
        if (!w && env > 0)
            w = env;
        f->walk_threads = w > 0 && w < f->n_threads ? w : f->n_threads;
        if (f->walk_threads < f->n_threads)
            fprintf(stderr,
                    "qe: a lone token's weight walk runs on %d of the %d threads this pass"
                    " has; on this host all of them are faster (QE_WALK_THREADS=full)\n",
                    f->walk_threads, f->n_threads);
    }

    /* The committed reading, spelled out so that "0 means default" is a fact
     * about this function rather than about the order of the tests below. The
     * grouped value heads read key head h % n_head_k: teacher-forced over text
     * the reference engine produced, that is 8.09 nats/token and English,
     * against 13.96 and a CJK loop for h / kdiv (ref/AUDIT.md). A caller that
     * passes any bit at all is answered as asked, so the loser stays
     * measurable from the command line. */
    if (!(f->o.variant & (QE_VARIANT_GDN_GVA_MOD | QE_VARIANT_L2NORM_ROOT |
                          QE_VARIANT_PLE_PRENORM | QE_VARIANT_NO_PLE)))
        f->o.variant |= QE_VARIANT_DEFAULT;

    /* Every dimension below is read from the file. The fallbacks are the values
     * docs/ARCH.md §4 says this architecture forces, and they are only used for
     * keys the checkpoint does not carry. */
    f->n_layer = dim_int(m, "qwen4exp.block_count", 48);
    f->K = dim_int(m, "qwen4exp.embedding_length", 2560);
    f->HC = dim_int(m, "qwen4exp.hyper_connection.count", 4);
    if (f->HC != 4) {
        snprintf(err, errn, "hyper-connection count %d: this driver keeps the stream "
                            "count argument-driven where it is cheap and asserts here, "
                            "because the head mixer and the PLE stream norms all assume it",
                 f->HC);
        free(f);
        return NULL;
    }
    f->hc_dim = f->HC * f->K;
    f->low_rank = dim_int(m, "qwen4exp.hyper_connection.low_rank", 320);
    f->gdn_kernel = dim_int(m, "qwen4exp.ssm.conv_kernel", 4);
    f->ple_kernel = dim_int(m, "qwen4exp.ple.conv_kernel", 4);
    if (f->gdn_kernel < 1 || f->ple_kernel < 1) {
        snprintf(err, errn, "conv kernels of %d and %d: a causal convolution needs at "
                            "least one tap, and the history windows are sized from both",
                 f->gdn_kernel, f->ple_kernel);
        free(f);
        return NULL;
    }
    if (f->gdn_kernel - 1 > QE_GDN_TAP_MAX) {
        snprintf(err, errn, "ssm conv kernel %d: the history ring holds %d taps and the "
                            "GDN core seats %d",
                 f->gdn_kernel, f->gdn_kernel - 1, QE_GDN_TAP_MAX);
        free(f);
        return NULL;
    }
    /* The PLE conv is dilated by 3 (docs/ARCH.md 6.2: nine positions of
     * history for a kernel of 4), so its window is (k-1)*d wide. */
    f->ple_dilation = 3;
    f->ple_lag = (f->ple_kernel - 1) * f->ple_dilation;
    f->n_q_heads = dim_int(m, "qwen4exp.attention.head_count", 24);
    f->n_kv_heads = dim_int(m, "qwen4exp.attention.head_count_kv", 2);
    f->head_dim = dim_int(m, "qwen4exp.attention.key_length", 256);
    f->n_rope = dim_int(m, "qwen4exp.rope.dimension_count", 64);
    f->theta = dim_float(m, "qwen4exp.rope.freq_base", 1e7f);
    {
        /* The m-rope sections, in pairs: which share of the rotated pairs reads the
         * temporal, the height and the width position. Absent, every pair reads the
         * first, which is ordinary rope - and that is all a text-only pass ever is. */
        const long long *sec = NULL;
        size_t ns = 0;
        f->msec[0] = f->n_rope / 2;
        f->msec[1] = f->msec[2] = f->msec[3] = 0;
        if (!qe_model_get_arr_int(m, "qwen4exp.rope.dimension_sections", &sec, &ns) && ns >= 3)
            for (size_t i = 0; i < 4; i++)
                f->msec[i] = i < ns ? (int)sec[i] : 0;
    }
    f->eps = dim_float(m, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f);
    f->n_idx_heads = dim_int(m, "qwen4exp.attention.indexer.head_count", 4);
    f->idx_dim = dim_int(m, "qwen4exp.attention.indexer.key_length", 128);
    if (f->idx_dim < 1 || f->idx_dim > QE_IDX_DIM_MAX) {
        snprintf(err, errn, "indexer key_length %d: the pooled-key row is a stack row of at most %d",
                 f->idx_dim, QE_IDX_DIM_MAX);
        free(f);
        return NULL;
    }
    {
        const char *e = getenv("QE_IDX_F16");
        f->idx_f16 = e && *e ? atoi(e) != 0 : 0;
    }
    /* The pooling factor is per layer, and the caches are sized for one number,
     * so the array is read and checked rather than assumed: a checkpoint that
     * mixed ratios would otherwise get a KV cache with the wrong stride and no
     * complaint anywhere. */
    f->kpool_ratio = 4;
    {
        const long long *ratios = NULL;
        size_t nr = 0;
        if (qe_model_get_arr_int(m, "qwen4exp.attention.compress_ratios", &ratios, &nr) == 0) {
            for (size_t i = 0; i < nr && (int)i < f->n_layer; i++)
                if (ratios[i] != 0 && ratios[i] != f->kpool_ratio) {
                    snprintf(err, errn,
                             "attention.compress_ratios[%zu] is %lld: the indexer pools "
                             "every layer by one factor, because the pooled-key cache is a "
                             "single strided arena, and this checkpoint wants two",
                             i, ratios[i]);
                    free(f);
                    return NULL;
                }
        }
    }
    f->idx_topk_blocks = dim_int(m, "qwen4exp.attention.indexer.top_k", 2048) / f->kpool_ratio;
    if (f->idx_topk_blocks < 1) {
        snprintf(err, errn, "indexer top_k of %d blocks: nothing to select with",
                 f->idx_topk_blocks);
        free(f);
        return NULL;
    }
    f->n_gdn_heads = dim_int(m, "qwen4exp.ssm.time_step_rank", 48);
    f->n_gdn_kheads = dim_int(m, "qwen4exp.ssm.group_count", 16);
    f->head_state = dim_int(m, "qwen4exp.ssm.state_size", 128);
    /* Head sharing is the divisor the delta rule's index arithmetic depends on,
     * and it has to be exact: 48 value heads over 16 key groups is three, and a
     * checkpoint where it is not an integer is a checkpoint this loop cannot
     * index without a remainder it would have to hide. */
    if (f->n_gdn_heads < 1 || f->n_gdn_kheads < 1 || f->n_gdn_heads % f->n_gdn_kheads ||
        f->n_q_heads < 1 || f->n_kv_heads < 1 || f->n_q_heads % f->n_kv_heads) {
        snprintf(err, errn,
                 "%d value heads over %d key groups, %d query heads over %d key groups: "
                 "both must divide, because one state is shared by an exact number of heads",
                 f->n_gdn_heads, f->n_gdn_kheads, f->n_q_heads, f->n_kv_heads);
        free(f);
        return NULL;
    }
    f->gdn_channels = 2 * f->head_state * f->n_gdn_kheads + f->head_state * f->n_gdn_heads;
    f->n_experts = dim_int(m, "qwen4exp.expert_count", 512);
    f->n_experts_active = dim_int(m, "qwen4exp.expert_used_count", 10);
    f->n_ff_exp = dim_int(m, "qwen4exp.expert_feed_forward_length", 640);
    f->n_ff_shexp = dim_int(m, "qwen4exp.expert_shared_feed_forward_length", 640);
    if (f->n_experts_active < 1 || f->n_experts_active > f->n_experts) {
        snprintf(err, errn, "%d of %d experts active", f->n_experts_active, f->n_experts);
        free(f);
        return NULL;
    }

    /* Three scales, and the delta rule's is the one the diagrams in docs/ARCH.md
     * §6 leave out: `ggml_compute_forward_gated_delta_net_one_chunk` ends with
     * `attn_data[j] = sum * (1/sqrtf(S_v))`, so the read-out of the state is
     * divided by sqrt of the state size before the per-head norm sees it. It
     * very nearly does not matter - a uniform scale cancels through an RMS norm -
     * but only nearly: what cancels is the scale and what does not is the eps
     * under that norm's root, and the two disagree by whatever the head's mean
     * square is below 1. The attention logits carry 1/sqrt(head_dim) and every
     * indexer head score carries 1/sqrt(indexer_key_length), and dropping either
     * produces a model that is confidently wrong in exactly the layers that are
     * sparse. */
    f->gdn_scale = 1.0f / sqrtf((float)f->head_state);
    f->attn_scale = 1.0f / sqrtf((float)f->head_dim);
    f->idx_scale = 1.0f / sqrtf((float)f->idx_dim);

    /* --- tensors */
    /* One table entry above the trunk's layers for the draft head's block, so that
     * a draft row is a layer of this pass rather than a special case of it: the
     * mixer, the attention and the expert pool the trunk runs are the same three
     * functions asked for layer `n_layer`. Zero when no head was handed over, in
     * which case every loop below is exactly the loop it was. */
    f->n_layer_all = f->n_layer + (f->o.mtp_model ? 1 : 0);
    f->L = calloc((size_t)f->n_layer_all, sizeof *f->L);
    f->moe = calloc((size_t)f->n_layer_all, sizeof *f->moe);
    f->moe_shadow = calloc((size_t)f->n_layer_all, sizeof *f->moe_shadow);
    if (!f->L || !f->moe || !f->moe_shadow) {
        snprintf(err, errn, "out of memory");
        qe_forward_close(f);
        return NULL;
    }
    f->embd = qe_tensor_by_name(m, "token_embd.weight");
    f->outw = qe_tensor_by_name(m, "output.weight");
    f->head_norm = qe_tensor_by_name(m, "output_hc_norm.weight");
    f->head_down = qe_tensor_by_name(m, "output_hc_down.weight");
    f->head_up = qe_tensor_by_name(m, "output_hc_up.weight");
    if (!f->embd || !f->outw || !f->head_norm || !f->head_down || !f->head_up) {
        note_missing(err, errn, "token_embd.weight/output.weight/output_hc_*");
        qe_forward_close(f);
        return NULL;
    }
    if (f->embd->type != QE_Q8_0 || (int)f->embd->ne[0] != f->K || (f->K % 32)) {
        snprintf(err, errn, "token_embd.weight is %s %llx x %llx; this driver can embed "
                            "from Q8_0 rows of width %d and says so rather than guessing",
                 qe_type_info_of(f->embd->type)->name, (unsigned long long)f->embd->ne[1],
                 (unsigned long long)f->embd->ne[0], f->K);
        qe_forward_close(f);
        return NULL;
    }

    int gdn_slots = 0, qsa_slots = 0;
    for (int il = 0; il < f->n_layer; il++) {
        qe_layer *L = &f->L[il];
        char n[QE_MAX_NAME];
        snprintf(n, sizeof n, "blk.%d.attn_qkv.weight", il);
        L->qsa = qe_tensor_by_name(m, n) ? 0 : 1;
        if (!L->qsa) {
            if (want_f(m, "blk.%d.attn_qkv.weight", il, err, errn, &L->qkv, QE_Q8_0) ||
                want_f(m, "blk.%d.attn_gate.weight", il, err, errn, &L->zgate, QE_Q8_0) ||
                want_f(m, "blk.%d.ssm_conv1d.weight", il, err, errn, &L->conv, QE_F32) ||
                want_f(m, "blk.%d.ssm_out.weight", il, err, errn, &L->ssm_out, QE_Q8_0) ||
                want_f(m, "blk.%d.ssm_alpha.weight", il, err, errn, &L->ssm_alpha, QE_F32) ||
                want_f(m, "blk.%d.ssm_beta.weight", il, err, errn, &L->ssm_beta, QE_F32) ||
                want_f(m, "blk.%d.ssm_norm.weight", il, err, errn, &L->ssm_norm, QE_F32)) {
                qe_forward_close(f);
                return NULL;
            }
            if (!want(m, "blk.%d.ssm_a", il, err, errn, &L->ssm_a) ||
                !want(m, "blk.%d.ssm_dt.bias", il, err, errn, &L->ssm_dt)) {
                qe_forward_close(f);
                return NULL;
            }
            L->gdn_slot = gdn_slots++;
        } else {
            if (want_f(m, "blk.%d.attn_q.weight", il, err, errn, &L->aq, QE_Q8_0) ||
                want_f(m, "blk.%d.attn_k.weight", il, err, errn, &L->ak, QE_Q8_0) ||
                want_f(m, "blk.%d.attn_v.weight", il, err, errn, &L->av, QE_Q8_0) ||
                want_f(m, "blk.%d.attn_output.weight", il, err, errn, &L->aout, QE_Q8_0) ||
                want_f(m, "blk.%d.attn_q_norm.weight", il, err, errn, &L->aqn, QE_F32) ||
                want_f(m, "blk.%d.attn_k_norm.weight", il, err, errn, &L->akn, QE_F32) ||
                want_f(m, "blk.%d.indexer.q_proj.weight", il, err, errn, &L->ixq, QE_BF16) ||
                want_f(m, "blk.%d.indexer.k_proj.weight", il, err, errn, &L->ixk, QE_BF16) ||
                want_f(m, "blk.%d.indexer.q_norm.weight", il, err, errn, &L->ixqn, QE_F32) ||
                want_f(m, "blk.%d.indexer.k_norm.weight", il, err, errn, &L->ixkn, QE_F32)) {
                qe_forward_close(f);
                return NULL;
            }
            L->qsa_slot = qsa_slots++;
        }
        const char *which[2] = { "attn", "ffn" };
        for (int h = 0; h < 2; h++) {
            char fmt[64];
            snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_norm.weight", which[h]);
            if (!want(m, fmt, il, err, errn, &L->hc_norm[h])) {
                qe_forward_close(f);
                return NULL;
            }
            snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_down.weight", which[h]);
            if (!want(m, fmt, il, err, errn, &L->hc_down[h])) {
                qe_forward_close(f);
                return NULL;
            }
            snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_up.weight", which[h]);
            if (!want(m, fmt, il, err, errn, &L->hc_up[h])) {
                qe_forward_close(f);
                return NULL;
            }
            snprintf(fmt, sizeof fmt, "blk.%%d.hc_%s_inject.weight", which[h]);
            if (!want(m, fmt, il, err, errn, &L->hc_inject[h])) {
                qe_forward_close(f);
                return NULL;
            }
        }
        if (want_f(m, "blk.%d.ffn_gate_inp.weight", il, err, errn, &L->router, QE_F32) ||
            !want(m, "blk.%d.ffn_gate_exps.weight", il, err, errn, &L->ex_g) ||
            !want(m, "blk.%d.ffn_up_exps.weight", il, err, errn, &L->ex_u) ||
            !want(m, "blk.%d.ffn_down_exps.weight", il, err, errn, &L->ex_d) ||
            !want(m, "blk.%d.ffn_gate_shexp.weight", il, err, errn, &L->sh_gate) ||
            !want(m, "blk.%d.ffn_up_shexp.weight", il, err, errn, &L->sh_up) ||
            !want(m, "blk.%d.ffn_down_shexp.weight", il, err, errn, &L->sh_down) ||
            !want(m, "blk.%d.ffn_gate_inp_shexp.weight", il, err, errn, &L->sh_inp)) {
            qe_forward_close(f);
            return NULL;
        }
        char pn[QE_MAX_NAME];
        snprintf(pn, sizeof pn, "blk.%d.ple_key.weight", il);
        if ((L->ple = qe_tensor_by_name(m, pn) ? 1 : 0)) {
            if (!want(m, "blk.%d.ple_key.weight", il, err, errn, &L->ple_key) ||
                !want(m, "blk.%d.ple_value.weight", il, err, errn, &L->ple_value) ||
                !want(m, "blk.%d.ple_norm_key.weight", il, err, errn, &L->ple_nk) ||
                !want(m, "blk.%d.ple_norm_query.weight", il, err, errn, &L->ple_nq) ||
                !want(m, "blk.%d.ple_norm_conv.weight", il, err, errn, &L->ple_nc) ||
                !want(m, "blk.%d.ple_conv1d.weight", il, err, errn, &L->ple_conv)) {
                qe_forward_close(f);
                return NULL;
            }
        }
        if (L->ple && L->qsa) {
            snprintf(err, errn, "layer %d is both PLE and full attention; the PLE conv "
                                "history needs a linear layer to live in",
                     il);
            qe_forward_close(f);
            return NULL;
        }
    }

    /* --- one qe_moe per layer, with a bf16 shadow of each router: the router is
     * read whole for every token, so the shadow is 240 -> 120 MiB per token, and
     * it was measured to pick the same 10 of 512 in 96000 real layer-token pairs
     * (docs/DESIGN.md §9). The weights themselves still come from the f32 rows. */
    for (int il = 0; il < f->n_layer; il++) {
        qe_layer *L = &f->L[il];
        qe_moe_desc d = { .gate = qe_tensor_data(m, L->ex_g),
                          .up = qe_tensor_data(m, L->ex_u),
                          .down = qe_tensor_data(m, L->ex_d),
                          .gate_stride = L->ex_g->bytes / L->ex_g->ne[2],
                          .up_stride = L->ex_u->bytes / L->ex_u->ne[2],
                          .down_stride = L->ex_d->bytes / L->ex_d->ne[2],
                          .t_gate = L->ex_g->type,
                          .t_up = L->ex_u->type,
                          .t_down = L->ex_d->type,
                          .router = qe_tensor_data(m, L->router),
                          .n_experts = f->n_experts,
                          .N = f->n_ff_exp,
                          .K = f->K,
                          .top_k = f->n_experts_active,
                          .n_threads = f->n_threads };
        size_t shadow = (size_t)f->n_experts * (size_t)f->K * 2;
        void *sh = hugobuf(f, shadow);
        if (!sh) {
            snprintf(err, errn, "out of memory for the router shadow");
            qe_forward_close(f);
            return NULL;
        }
        f->moe_shadow[il] = sh;
        qe_bf16_from_f32(qe_tensor_data(m, L->router), sh, (size_t)f->n_experts * (size_t)f->K);
        d.router_bf16 = sh;
        f->moe[il] = qe_moe_create(&d);
        if (!f->moe[il]) {
            snprintf(err, errn, "layer %d: the expert pool (%s gate, %s up, %s down) has no "
                                "dispatch here",
                     il, qe_type_info_of(d.t_gate)->name, qe_type_info_of(d.t_up)->name,
                     qe_type_info_of(d.t_down)->name);
            qe_forward_close(f);
            return NULL;
        }
    }

    /* --- the draft head, as one more layer. It opens here rather than in
     * engine.c because it wants the layer table sized, the router-shadow idiom the
     * expert pools are built with, and a slot in caches that do not exist yet.
     * A head that cannot open is a slower engine and not a broken one, so every
     * failure below is one line on stderr and a trunk that carries on. */
    if (f->o.mtp_model) {
        char herr[512] = { 0 };
        if (mtp_layer_open(f, f->o.mtp_model, &qsa_slots, herr, sizeof herr) < 0) {
            fprintf(stderr, "qe: draft head not opened: %s\n", herr);
            f->mtp_ready = 0;
        }
    }

    /* --- PLE gather */
    qe_ple_desc pd;
    memset(&pd, 0, sizeof pd);
    f->ple = qe_ple_create(m, &pd, err, errn);
    if (!f->ple) {
        qe_forward_close(f);
        return NULL;
    }
    f->ple_eos = (long long)qe_ple_eos_token(f->ple);

    /* --- state */
    f->gdn_slots = gdn_slots;
    f->qsa_slots = qsa_slots;
    f->kv_row = (size_t)f->n_kv_heads * (size_t)f->head_dim;
    f->n_ctx = (size_t)f->o.n_ctx;
    /* The split (include/qe.h, qe_forward_opts.n_seq): B sequences share the cache
     * bytes by taking n_ctx/B cells each, and replicate only what has no length to
     * share. Both numbers are derived here rather than passed in, because every
     * pointer that divides by them is derived from them the same way, and a pass with
     * one sequence divides by exactly what it divided by before. */
    f->n_seq = f->o.n_seq > 1 ? f->o.n_seq : 1;
    f->seq_stride = f->n_ctx / (size_t)f->n_seq;
    size_t kv_row = f->kv_row;
    size_t n_ctx = f->n_ctx;
    /* The delta matrices and the conv rings, one copy per sequence. A sequence reads
     * and writes its own state every layer of every token, so this is the byte cost
     * of a batched step and not the caches' - and it is small: ~216 MiB against the
     * 12 GiB of cache one sequence already is. */
    f->gdn = (float *)hugobuf(f, state_gdn_bytes(f));
    f->gconv = (float *)hugobuf(f, state_gconv_bytes(f));
    f->kv_q8 = kv_quant_wanted(&f->o);
    if (f->kv_q8) {
        f->kvd = (int8_t *)hugobuf(f, state_kv_bytes(f));
        f->kvs = (float *)hugobuf(f, state_kvs_bytes(f));
    } else {
        f->kv = (float *)hugobuf(f, state_kv_bytes(f));
    }
    f->kraw = (float *)hugobuf(f, state_kraw_bytes(f));
    f->kpool = (float *)hugobuf(f, state_kpool_bytes(f));
    f->ple_hist = (float *)hugobuf(f, state_ple_bytes(f));
    if (!f->gdn || !f->gconv || !f->kraw || !f->kpool || !f->ple_hist ||
        (f->kv_q8 ? (!f->kvd || !f->kvs) : !f->kv)) {
        snprintf(err, errn, "out of memory for %d layers of state", f->n_layer);
        qe_forward_close(f);
        return NULL;
    }

    /* --- the chunk panel the integer projections recode into (proj_i8). Sized for the
     * widest matrix input in the file rather than for the width in force, because the
     * width is a knob `make forward` moves mid-run and the panel must not be
     * reallocated under a walk. One buffer for the pass, not one per thread: the panel
     * is written only between two dispatches and read by all of them. */
    /* The widths a recode is asked for are the widths of the trunk's own matrices:
     * a draft head reads the trunk's activations, so its inputs are K or n_ff_exp
     * wide and this scan of the trunk's table already covers the head. */
    long i8k = 0;
    for (size_t i = 0; i < qe_tensor_count(m); i++) {
        const qe_tensor *t = qe_tensor_by_index(m, i);
        if (t->ne[1] > 1 && qe_gemv_i8_has(t->type) && (long)t->ne[0] > i8k)
            i8k = (long)t->ne[0];
    }
    f->i8k = i8k;
    if (i8k &&
        (!(f->i8x = scratch(f, E_i8x, (size_t)QE_CHUNK_MAX * (size_t)i8k)) ||
         !(f->i8s = scratch(f, E_i8s, (size_t)QE_CHUNK_MAX * (size_t)(i8k / 32) *
                                            sizeof(float))) ||
         !(f->i8pseed = scratch(f, E_i8p, (size_t)QE_CHUNK_MAX * (size_t)(i8k / 32) *
                                            sizeof(int32_t))))) {
        snprintf(err, errn, "out of memory for the chunk panel");
        qe_forward_close(f);
        return NULL;
    }

    /* --- scratch: every activation vector a step touches, allocated once.
     * ROWS() is the deep form of a buffer: one row per token of the chunk in
     * flight, for the outputs a chunked projection fills in a single walk of a
     * weight. Nothing a recurrence writes is deep, because that is where tokens
     * stop being independent. */
    int K = f->K, HCW = f->hc_dim;
#define ROWS(n) ((size_t)QE_CHUNK_MAX * (size_t)(n) * sizeof(float))
    if (!(f->cres = scratch(f, E_cres, (size_t)QE_CHUNK_MAX * HCW * sizeof(float))) ||
        !(f->cxn = scratch(f, E_cxn, (size_t)QE_CHUNK_MAX * HCW * sizeof(float))) ||
        !(f->hn = scratch(f, E_hn, ROWS(HCW))) ||
        !(f->gate = scratch(f, E_gate, ROWS(HCW))) ||
        !(f->lo = scratch(f, E_lo, ROWS(f->low_rank))) ||
        !(f->cinj = scratch(f, E_cinj, (size_t)QE_CHUNK_MAX * f->HC * sizeof(float))) ||
        !(f->cy = scratch(f, E_cy, (size_t)QE_CHUNK_MAX * K * sizeof(float))) ||
        !(f->sh_out = scratch(f, E_sh, ROWS(K))) ||
        !(f->qkv = scratch(f, E_qkv, ROWS(f->gdn_channels))) ||
        !(f->z = scratch(f, E_z, ROWS(f->head_state * f->n_gdn_heads))) ||
        !(f->conv_out = scratch(f, E_conv, ROWS(f->gdn_channels))) ||
        !(f->beta = scratch(f, E_beta, ROWS(f->n_gdn_heads))) ||
        !(f->alpha = scratch(f, E_alpha, ROWS(f->n_gdn_heads))) ||
        !(f->gexp = scratch(f, E_gexp, ROWS(f->n_gdn_heads))) ||
        !(f->att = scratch(f, E_att, ROWS(f->head_state * f->n_gdn_heads))) ||
        !(f->att_norm = scratch(f, E_attnorm, ROWS(f->head_state * f->n_gdn_heads))) ||
        !(f->qg = scratch(f, E_qg, ROWS(2 * f->head_dim * f->n_q_heads))) ||
        !(f->qwork = scratch(f, E_qwork, (size_t)f->head_dim * f->n_q_heads * sizeof(float))) ||
        !(f->qgate = scratch(f, E_qgate, (size_t)f->head_dim * f->n_q_heads * sizeof(float))) ||
        !(f->kcur = scratch(f, E_kcur, ROWS(kv_row))) ||
        !(f->vcur = scratch(f, E_vcur, ROWS(kv_row))) ||
        !(f->attn_y = scratch(f, E_attny, ROWS(f->head_dim * f->n_q_heads))) ||
        !(f->scores = scratch(f, E_scores, (size_t)f->n_q_heads *
                                                (f->idx_topk_blocks * f->kpool_ratio +
                                                 f->kpool_ratio) * sizeof(float))) ||
        !(f->iq = scratch(f, E_iq, ROWS(f->idx_dim * f->n_idx_heads))) ||
        !(f->ikraw = scratch(f, E_ikraw, ROWS(f->idx_dim))) ||
        !(f->ipool = scratch(f, E_ipool, (size_t)f->idx_dim * sizeof(float))) ||
        !(f->iscore = scratch(f, E_iscore, (size_t)f->o.n_ctx * sizeof(float))) ||
        !(f->pkey = scratch(f, E_pkey, (size_t)HCW * sizeof(float))) ||
        !(f->pval = scratch(f, E_pval, (size_t)K * sizeof(float))) ||
        !(f->pgated = scratch(f, E_pgated, (size_t)HCW * sizeof(float))) ||
        !(f->pgnorm = scratch(f, E_pgnorm, (size_t)HCW * sizeof(float))) ||
        !(f->pconv = scratch(f, E_pconv, (size_t)HCW * sizeof(float))) ||
        !(f->pemb = scratch(f, E_pemb, (size_t)qe_ple_width(f->ple) * sizeof(float))) ||
        !(f->e640 = scratch(f, E_e640, ROWS(f->n_ff_shexp))) ||
        !(f->u640 = scratch(f, E_u640, ROWS(f->n_ff_shexp))) ||
        !(f->h640 = scratch(f, E_h640, (size_t)f->n_ff_shexp * sizeof(float))) ||
        !(f->moe_logit = scratch(f, E_mlogits, (size_t)f->n_experts * sizeof(float))) ||
        !(f->cmoe_logit = scratch(f, E_cmlogits, ROWS(f->n_experts))) ||
        !(f->expert_y = scratch(f, E_ey, (size_t)K * sizeof(float))) ||
        !(f->cs = scratch(f, E_cs, (size_t)f->n_rope * sizeof(float))) ||
        !(f->sel = scratch(f, E_sel, ((size_t)f->idx_topk_blocks * (size_t)f->kpool_ratio +
                                      (size_t)f->kpool_ratio) * sizeof(int))) ||
        !(f->sel_bits = scratch(f, E_selbits,
                                ((n_ctx / (size_t)f->kpool_ratio + 64) / 64) * sizeof(uint64_t))) ||
        !(f->attn_py = scratch(f, E_attnpy,
                               (size_t)attn_units_bound(f->n_threads, f->n_kv_heads) *
                                   (size_t)(f->n_q_heads / f->n_kv_heads) *
                                   (size_t)f->head_dim * sizeof(float))) ||
        !(f->attn_pm = scratch(f, E_attnpm,
                               (size_t)attn_units_bound(f->n_threads, f->n_kv_heads) *
                                   (size_t)(f->n_q_heads / f->n_kv_heads) * sizeof(float))) ||
        !(f->attn_pl = scratch(f, E_attnpl,
                               (size_t)attn_units_bound(f->n_threads, f->n_kv_heads) *
                                   (size_t)(f->n_q_heads / f->n_kv_heads) * sizeof(float))) ||
        !(f->heap_s = scratch(f, E_heap, ((size_t)f->idx_topk_blocks + 1) * sizeof(float))) ||
        !(f->heap_b = scratch(f, E_heapb, ((size_t)f->idx_topk_blocks + 1) * sizeof(int))) ||
        !(f->routes = scratch(f, E_routes, (size_t)f->n_experts_active * sizeof(qe_route))) ||
        !(f->croutes = scratch(f, E_routes, ROWS(f->n_experts_active) * (sizeof(qe_route) / sizeof(float)))) ||
        !(f->seq = scratch(f, E_seq, n_ctx * sizeof(uint32_t))) ||
        !(f->rpos = scratch(f, E_rpos, n_ctx * 3 * sizeof(int32_t)))) {
        snprintf(err, errn, "out of memory for the step's scratch");
        qe_forward_close(f);
        return NULL;
    }
    {
        /* The batched attention of a prompt chunk (qsa_core_chunk): optional, so a pass
         * that cannot afford its scratch keeps the per-token form and says nothing. */
        const char *e = getenv("QE_ATTN_BATCH");
        f->attn_batch = e && *e ? atoi(e) : 1;
        /* The gate is applied a head at a time, which is the whole row's sigmoid only when a
         * head is whole vector blocks: the SIMD body and the scalar tail differ. */
        if (f->head_dim > QE_MAX_HEAD_DIM || f->head_dim % 8)
            f->attn_batch = 0;
        const int nthr = f->n_threads > 0 ? f->n_threads : 1;
        const int per = f->n_q_heads / f->n_kv_heads;
        f->sel_cap = (long)f->idx_topk_blocks * f->kpool_ratio + f->kpool_ratio;
        f->tbits_words = (long)((n_ctx / (size_t)f->kpool_ratio + 64) / 64);
        const size_t gunits = (size_t)ATTN_GROUP * (size_t)attn_units_bound(nthr, f->n_kv_heads);
        if (!(f->cqwork = scratch(f, E_cqwork, ROWS(f->head_dim * f->n_q_heads))) ||
            !(f->csel = scratch(f, E_csel, (size_t)QE_CHUNK_MAX * (size_t)f->sel_cap * sizeof(int))) ||
            !(f->tiscore = scratch(f, E_tiscore, (size_t)nthr * (size_t)(f->o.n_ctx / f->kpool_ratio + 1) * sizeof(float))) ||
            !(f->theap = scratch(f, E_theap, (size_t)nthr * ((size_t)f->idx_topk_blocks + 1) * sizeof(float))) ||
            !(f->theapb = scratch(f, E_theapb, (size_t)nthr * ((size_t)f->idx_topk_blocks + 1) * sizeof(int))) ||
            !(f->tbits = scratch(f, E_tbits, (size_t)nthr * (size_t)f->tbits_words * sizeof(uint64_t))) ||
            !(f->gpy = scratch(f, E_gpy, gunits * (size_t)per * (size_t)f->head_dim * sizeof(float))) ||
            !(f->gpm = scratch(f, E_gpm, gunits * (size_t)per * sizeof(float))) ||
            !(f->gpl = scratch(f, E_gpl, gunits * (size_t)per * sizeof(float)))) {
            f->cqwork = NULL;
            f->attn_batch = 0;
        }
    }
    use_row(f, 0);
#undef ROWS
    /* Last, so that the memory it reads as free is memory a step will not need.
     *
     * And only for a model whose bytes are actually placed: asked for against a
     * metadata-only mapping, this pass would read every dense weight byte off the device to
     * requantise it, which is the one thing a metadata-only open exists not to do - and
     * `make report` is that open on purpose (AGENTS.md 2). */
    if (f->flat_i8) {
        if (qe_model_stage_pinned(f->m))
            flat_stage(f);
        else {
            fprintf(stderr, "qe: QE_I8_FLAT asked for and not built: this open is a"
                            " metadata-only view of the checkpoint, not resident bytes\n");
            f->flat_i8 = 0;
        }
    }
    if (f->dq_i4) {
        if (qe_model_stage_pinned(f->m))
            dq_stage(f);
        else {
            fprintf(stderr, "qe: QE_I4_DENSE asked for and not built: this open is a"
                            " metadata-only view of the checkpoint, not resident bytes\n");
            f->dq_i4 = 0;
        }
    }
    /* The head's copies are staged whether or not either above was affordable: they are
     * a fraction of the size of the dense ones, and the head is the one matrix whose
     * bytes a lone token pays for whole. */
    if (qe_model_stage_pinned(f->m))
        head_stage(f);
    qe_forward_reset(f);
    return f;
}

void qe_forward_close(qe_forward *f) {
    if (!f)
        return;
    for (int i = 0; i < f->n_owned; i++)
        qe_free_huge(f->own_p[i], f->own_n[i]);
    if (f->moe) {
        for (int i = 0; i < f->n_layer_all; i++)
            qe_moe_destroy(f->moe[i]);
    }
    qe_ple_destroy(f->ple);
    free(f->moe);
    free(f->moe_shadow);
    free(f->L);
    free(f->flat_woff);
    free(f->flat_soff);
    free(f->dq_woff);
    free(f->dq_wcls);
    free(f->lg_cache);
    free(f);
}

void qe_forward_reset(qe_forward *f) {
    if (!f)
        return;
    if (f->gdn)
        memset(f->gdn, 0, state_gdn_bytes(f));
    if (f->gconv)
        memset(f->gconv, 0, state_gconv_bytes(f));
    if (f->kv)
        memset(f->kv, 0, state_kv_bytes(f));
    if (f->kvd)
        memset(f->kvd, 0, state_kv_bytes(f));
    if (f->kvs)
        memset(f->kvs, 0, state_kvs_bytes(f));
    if (f->kraw)
        memset(f->kraw, 0, state_kraw_bytes(f));
    if (f->kpool)
        memset(f->kpool, 0, state_kpool_bytes(f));
    if (f->ple_hist)
        memset(f->ple_hist, 0, state_ple_bytes(f));
    if (f->sel_bits)
        memset(f->sel_bits, 0, ((f->n_ctx / f->kpool_ratio + 64) / 64) * sizeof(uint64_t));
    /* A pending PLE row set is one token's window, and a new sequence has no use
     * for the old one's; leaving it queued hands the next token another
     * sequence's rows. The chunk loop normally drains itself, so this only ever
     * matters after an abandoned prefill. */
    qe_ple_drain(f->ple);
    f->pos = 0;
    for (int s = 0; s < QE_MAX_SEQ; s++) {
        f->spos[s] = 0;
        f->mrope[s] = 0;
    }
    /* No row is waiting for the token that follows it, and the row a new sequence
     * would be handed belongs to a sequence that no longer exists. The chunk that
     * just finished is history for the same reason: its rows are readable until the
     * next pass replaces them, and after a reset there is no pass whose rows a draft
     * should be seeding from. */
    f->mpend_pos = -1;
    f->pass_pos0 = 0;
    f->chunk_T = 0;
    memset(&f->d, 0, sizeof f->d);
}

long qe_forward_vocab(const qe_forward *f) {
    return f ? (long)f->embd->ne[1] : 0;
}
int qe_forward_ctx(const qe_forward *f) {
    return f ? f->o.n_ctx : 0;
}
long qe_forward_pos(const qe_forward *f) {
    return f ? f->pos : 0;
}
int qe_forward_kv_q8(const qe_forward *f) {
    return f ? f->kv_q8 : 0;
}
/* One sequence's state in bytes, and what one more token of context adds to it.
 * Only the attention cache, the indexer keys and the token ids grow; the rest a
 * running sequence holds whatever its length is, and the recurrent state alone is
 * 113 MiB of that. These two numbers are what caps how many sequences share the
 * host, which is why the width of the cache is a fact about serving and not a
 * detail of the layout. */
void qe_forward_state_bytes(const qe_forward *f, size_t *total, size_t *per_token) {
    if (total)
        *total = 0;
    if (per_token)
        *per_token = 0;
    if (!f)
        return;
    const size_t slots = (size_t)f->qsa_slots, ctx = f->n_ctx;
    /* The three recurrences, which a batched pass holds once per sequence, and the
     * caches, which it holds once however many sequences share them because a
     * sequence's cells are a slice of them and not a copy of them. */
    const size_t fixed = state_gdn_bytes(f) + state_gconv_bytes(f) + state_ple_bytes(f);
    const size_t kv = f->kv_q8
                          ? slots * 2 * ctx * (f->kv_row + (size_t)f->n_kv_heads * sizeof(float))
                          : slots * 2 * ctx * f->kv_row * sizeof(float);
    const size_t idx =
        slots * ctx * (size_t)f->idx_dim * sizeof(float) +
        slots * (ctx / (size_t)f->kpool_ratio + 1) * (size_t)f->idx_dim * sizeof(float);
    const size_t seq = ctx * sizeof(uint32_t);
    if (total)
        *total = fixed + kv + idx + seq;
    if (per_token)
        *per_token = ctx ? (kv + idx + seq) / ctx : 0;
}
/* --------------------------------------------------------- rewind points
 *
 * Sized and laid out from the same shape the pass was opened with, so the three
 * copies below cannot drift from what the layers write: each length is derived
 * here from the same fields qe_forward_reset clears. */
struct qe_state {
    /* Shape identity. A snapshot restored into a forward of another shape would
     * be a silent mis-remembering of a sequence, so it is refused. */
    int gdn_slots, n_gdn_heads, head_state, gdn_kernel, gdn_channels;
    int ple_lag, hc_dim, n_kv_heads, head_dim, idx_dim;
    size_t n_ctx;

    size_t gdn_b, gconv_b, ple_b;
    long pos;
    float *buf;    /* the three recurrences, back to back, one mapping */
    uint32_t *ids; /* n_ctx of them: the prefix this state is the answer to */
};

static void state_layout(const qe_forward *f, struct qe_state *s) {
    s->gdn_b = state_gdn_bytes(f);
    s->gconv_b = state_gconv_bytes(f);
    s->ple_b = state_ple_bytes(f);
}

size_t qe_state_size(const qe_forward *f) {
    if (!f)
        return 0;
    struct qe_state s = { 0 };
    state_layout(f, &s);
    return s.gdn_b + s.gconv_b + s.ple_b + f->n_ctx * sizeof(uint32_t);
}

void qe_state_free(qe_state *s) {
    if (!s)
        return;
    free(s->buf);
    free(s->ids);
    free(s);
}

qe_state *qe_state_new(const qe_forward *f) {
    if (!f)
        return NULL;
    qe_state *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    state_layout(f, s);
    s->gdn_slots = f->gdn_slots;
    s->n_gdn_heads = f->n_gdn_heads;
    s->head_state = f->head_state;
    s->gdn_kernel = f->gdn_kernel;
    s->gdn_channels = f->gdn_channels;
    s->ple_lag = f->ple_lag;
    s->hc_dim = f->hc_dim;
    s->n_kv_heads = f->n_kv_heads;
    s->head_dim = f->head_dim;
    s->idx_dim = f->idx_dim;
    s->n_ctx = f->n_ctx;
    const size_t fixed = s->gdn_b + s->gconv_b + s->ple_b;
    s->buf = malloc(fixed ? fixed : 1);
    s->ids = malloc(f->n_ctx * sizeof *s->ids);
    if (!s->buf || !s->ids) {
        qe_state_free(s);
        return NULL;
    }
    return s;
}

static int state_matches(const qe_forward *f, const qe_state *s) {
    if (!f || !s)
        return 0;
    struct qe_state want = { 0 };
    state_layout(f, &want);
    return s->gdn_slots == f->gdn_slots && s->n_gdn_heads == f->n_gdn_heads &&
           s->head_state == f->head_state && s->gdn_kernel == f->gdn_kernel &&
           s->gdn_channels == f->gdn_channels && s->ple_lag == f->ple_lag &&
           s->hc_dim == f->hc_dim && s->n_kv_heads == f->n_kv_heads &&
           s->head_dim == f->head_dim && s->idx_dim == f->idx_dim &&
           s->n_ctx == f->n_ctx && s->gdn_b == want.gdn_b && s->gconv_b == want.gconv_b &&
           s->ple_b == want.ple_b;
}

/* The copy, over the pool. 118 MiB is not a memcpy: one thread moving that much is
 * off the 76-79 GB/s this host reads at by more than an order of magnitude, and a
 * boundary is worth taking only if putting it back is free next to the ingestion it
 * replaces. One contiguous span per worker, cut on 64-byte lines so no two of them
 * write the same cache line, over the three regions as one buffer - which is why
 * `qe_state` keeps them adjacent instead of in three allocations.
 *
 * Called between calls into the forward pass, never from inside one, so the pool is
 * idle here and cannot be re-entered. */
#define STATE_COPY_PAR_MIN (4u << 20)

typedef struct {
    char *dst;
    const char *src;
    size_t n;
} state_copy_job;

static void state_copy_task(void *vp, int tid, int nthreads) {
    state_copy_job *j = vp;
    size_t per = ((j->n / (size_t)nthreads) + 63) & ~(size_t)63;
    if (!per)
        per = 64;
    size_t lo = per * (size_t)tid;
    if (lo >= j->n)
        return;
    size_t hi = lo + per < j->n ? lo + per : j->n;
    memcpy(j->dst + lo, j->src + lo, hi - lo);
}

static void state_copy(int n_threads, char *dst, const char *src, size_t n) {
    if (!n)
        return;
    int nt = n_threads > 0 ? n_threads : 1;
    if (n < STATE_COPY_PAR_MIN || nt < 2 || qe_pool_size() < 2) {
        memcpy(dst, src, n);
        return;
    }
    state_copy_job j = { dst, src, n };
    qe_pool_run(state_copy_task, &j, nt);
}

int qe_forward_snapshot(const qe_forward *f, qe_state *out) {
    /* A snapshot is one sequence's three recurrences and its position. A pass driving
     * more than one sequence has n_seq of both, and no caller of this has anything to
     * say about which is meant, so it is refused rather than guessed at: the batched
     * lane does not use snapshots, and neither can a prefix cache nor a draft. */
    if (!state_matches(f, out) || f->n_seq > 1)
        return -1;
    state_copy(f->n_threads, (char *)out->buf, (const char *)f->gdn, out->gdn_b);
    state_copy(f->n_threads, (char *)out->buf + out->gdn_b, (const char *)f->gconv,
               out->gconv_b);
    state_copy(f->n_threads, (char *)out->buf + out->gdn_b + out->gconv_b, (const char *)f->ple_hist,
               out->ple_b);
    if (f->pos > 0)
        memcpy(out->ids, f->seq, (size_t)f->pos * sizeof *out->ids);
    out->pos = f->pos;
    return 0;
}

/* A range of cells, copied out of the caches or back into them.
 *
 * A rewind point is the recurrences only, and the cells under it stay in the caches,
 * so a point is good exactly as long as those cells are. A prompt fed from a lower
 * position writes over them, and the conversation the point belongs to has then lost
 * its attention context. These two move cells [from, to) of sequence 0 in and out of a
 * flat buffer, so the engine can keep the cells another prompt is about to overwrite and
 * put them back when their conversation returns.
 *
 * Every array a cell has, for every attention slot (the draft head's included): its key
 * and value rows, in whichever form the cache stores them, with their scales; the
 * indexer's raw key; and the pooled keys of every block the range touches. A block that
 * straddles `from` was pooled from cells on both sides, and the cells below `from` are
 * the shared prefix by construction, so the copy is that block as it was. Then the id
 * and the rope position of every cell. One walk does both directions, so the layout the
 * bytes are written in and the layout they are read back in cannot differ. */
static size_t cells_walk(qe_forward *f, long from, long to, char *buf, int put) {
    const size_t NC = f->seq_stride;
    const size_t krow = f->kv_row, nkvh = (size_t)f->n_kv_heads, id = (size_t)f->idx_dim;
    const size_t n = (size_t)(to - from);
    const long b0 = from / f->kpool_ratio, b1 = (to + f->kpool_ratio - 1) / f->kpool_ratio;
    const size_t esz = f->idx_f16 ? 2 : 4;
    size_t o = 0;
#define MOVE(ptr, bytes)                                                                  \
    do {                                                                                  \
        if (buf) {                                                                        \
            if (put)                                                                      \
                memcpy((ptr), buf + o, (bytes));                                          \
            else                                                                          \
                memcpy(buf + o, (ptr), (bytes));                                          \
        }                                                                                 \
        o += (bytes);                                                                     \
    } while (0)
    for (int slot = 0; slot < f->qsa_slots; slot++) {
        const long sub = qsa_sub(f, slot, 0);
        for (int kv = 0; kv < 2; kv++) { /* keys, then values */
            const size_t row0 = (size_t)kv * NC + (size_t)from;
            if (f->kv)
                MOVE(f->kv + ((size_t)sub * 2 * NC + row0) * krow, n * krow * sizeof(float));
            if (f->kvd)
                MOVE(f->kvd + ((size_t)sub * 2 * NC + row0) * krow, n * krow);
            if (f->kvs)
                MOVE(f->kvs + ((size_t)sub * 2 * NC + row0) * nkvh, n * nkvh * sizeof(float));
        }
        if (f->kraw)
            MOVE(f->kraw + ((size_t)sub * NC + (size_t)from) * id, n * id * sizeof(float));
        if (f->kpool && b1 > b0)
            MOVE((char *)kpool_seq(f, sub) + (size_t)b0 * id * esz, (size_t)(b1 - b0) * id * esz);
    }
    MOVE(f->seq + from, n * sizeof *f->seq);
    MOVE(f->rpos + (size_t)from * 3, n * 3 * sizeof(int32_t));
#undef MOVE
    return o;
}

size_t qe_forward_cells_bytes(qe_forward *f, long from, long to) {
    if (!f || f->n_seq > 1 || from < 0 || to <= from || to > (long)f->seq_stride)
        return 0;
    return cells_walk(f, from, to, NULL, 0);
}

int qe_forward_cells_get(qe_forward *f, long from, long to, void *buf) {
    if (!qe_forward_cells_bytes(f, from, to) || !buf)
        return -1;
    cells_walk(f, from, to, buf, 0);
    return 0;
}

int qe_forward_cells_put(qe_forward *f, long from, long to, const void *buf) {
    if (!qe_forward_cells_bytes(f, from, to) || !buf)
        return -1;
    cells_walk(f, from, to, (char *)buf, 1);
    return 0;
}

int qe_forward_restore(qe_forward *f, const qe_state *in) {
    /* Refused for a batched pass for the reason a snapshot is: the state of a pass
     * driving n_seq sequences holds n_seq copies of every recurrence, and no caller of
     * this has anything to say about which sequence it is putting back. */
    if (!state_matches(f, in) || f->n_seq > 1 || in->pos > (long)f->n_ctx)
        return -1;
    state_copy(f->n_threads, (char *)f->gdn, (const char *)in->buf, in->gdn_b);
    state_copy(f->n_threads, (char *)f->gconv, (const char *)in->buf + in->gdn_b, in->gconv_b);
    state_copy(f->n_threads, (char *)f->ple_hist, (const char *)in->buf + in->gdn_b + in->gconv_b,
               in->ple_b);
    if (in->pos > 0)
        memcpy(f->seq, in->ids, (size_t)in->pos * sizeof *f->seq);
    /* The one cursor there is. Both convolution rings are addressed as
     * `pos % rows`, and the caches and the indexer's pool read their valid extent
     * off the position, so this line is what makes the three copies above mean
     * this position rather than the one they were taken at. Everything past it is
     * stale by construction and written before it is read again. */
    f->pos = in->pos;
    f->spos[0] = in->pos; /* the cursor a chunk reads its positions off */
    /* Whether the cells it stands on take m-rope. The table under them is as
     * append-only as the caches are, so it is the restored position that decides
     * which part of it is this sequence's, and one pass over it says whether any cell
     * there is not plain text at its own index. */
    f->mrope[0] = 0;
    for (long c = 0; c < in->pos && !f->mrope[0]; c++) {
        const int32_t *rp = f->rpos + (size_t)c * 3;
        f->mrope[0] = rp[0] != c || rp[1] != c || rp[2] != c;
    }
    qe_ple_drain(f->ple);
    return 0;
}

long qe_state_pos(const qe_state *s) { return s ? s->pos : 0; }
const uint32_t *qe_state_ids(const qe_state *s) { return s ? s->ids : NULL; }

int qe_state_set_ids(qe_state *s, const int *ids, long n) {
    if (!s || !ids || n != s->pos || n < 0 || (size_t)n > s->n_ctx)
        return -1;
    for (long i = 0; i < n; i++)
        s->ids[i] = (uint32_t)ids[i];
    return 0;
}

long qe_forward_hidden(const qe_forward *f, float *dst, long n) {
    if (!f || !dst || n < f->K)
        return -1;
    memcpy(dst, f->xn, (size_t)f->K * sizeof *dst);
    return f->K;
}
void qe_forward_diag_of(const qe_forward *f, qe_forward_diag *out) {
    if (out)
        *out = f->d;
}

/* --------------------------------------------------------------- one step
 *
 * Everything above is bookkeeping. This is the part docs/ARCH.md §6 is about,
 * written as one pass over the layers with a function per component, in the
 * order the arithmetic happens. The comments below are for the places where two
 * readings of the diagram differ and only one of them is the model:
 *
 *   - the mixer's norm reduces per stream of 2560, not over the 10240 the four
 *     streams are concatenated into (§6.1);
 *   - the PLE gate is a signed square root of its own score, one per stream,
 *     and its conv is dilated by 3, so its history is nine rows and not three
 *     (§6.2);
 *   - a GDN value head h reads key and query head h/3 - 48 value heads share the
 *     16 key heads three ways, and h % 16 is a different model - and the
 *     read-out carries no 1/sqrt(state_size), which the diagram annotates
 *     because it is the thing every reader expects to be there (§6.3);
 *   - the indexer pools raw keys before it normalises and rotates them, and the
 *     angle is the block's first cell's position, not the query's (§6.4);
 *   - attention rotates 64 of the 256 head dims, in split-half pairs, and gates
 *     with a sigmoid of the second half of the same 512-wide row of attn_q
 *     (§6.4);
 *   - the shared expert is gated by one scalar read from a 1-D tensor, not by a
 *     per-channel vector (§6.5).
 */

#define QE_MAX_ROPE 256

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

static const float *f32v(qe_forward *f, const qe_tensor *t) {
    return (const float *)qe_tensor_data(f->m, t);
}

/* A non-finite activation is not an error to be reported later: by the time a
 * logit is wrong the whole step is, so the count is attributed to the stage that
 * produced it and the step fails. A value that is merely absurd (>= 1e30, which
 * fabsf of a NaN also fails) is counted the same way, because that is what a
 * missing scale factor looks like one layer before it becomes a NaN. */
static void sweep(qe_forward *f, const float *v, long n, float *band) {
    float m = band ? *band : 0.0f;
    for (long i = 0; i < n; i++) {
        float a = fabsf(v[i]);
        if (!(a < 1e30f)) {
            f->d.nonfinite++;
            return;
        }
        if (a > m)
            m = a;
    }
    if (band)
        *band = m;
}

/* One row of the embedding table, dequantised to `dst`. Split out of embed()
 * because the draft head wants the same row without the four streams behind it:
 * the fold's embedding half is one K-wide vector, and the head's own copy of the
 * table is a re-quantisation of this one, so reading the trunk's is both fewer
 * bits lost and no second walk of a 340 MiB table. */
static void embed_row(qe_forward *f, long row, float *dst) {
    const qe_block_q8_0 *b = (const qe_block_q8_0 *)qe_tensor_data(f->m, f->embd);
    size_t rstride = (size_t)tensor_row_bytes(f->embd) / sizeof(qe_block_q8_0);
    const qe_block_q8_0 *r = b + (size_t)row * rstride;
    long nb = f->K / 32;
    for (long k = 0; k < nb; k++) {
        float d = qe_half_to_f32(r[k].d);
        for (int i = 0; i < 32; i++)
            dst[k * 32 + i] = d * (float)r[k].qs[i];
    }
}

static void embed(qe_forward *f, long row) {
    embed_row(f, row, f->res);
    /* The four hyper-connection streams start identical; they separate through
     * the injections and never merge again. */
    for (int s = 1; s < f->HC; s++)
        memcpy(f->res + (size_t)s * f->K, f->res, (size_t)f->K * sizeof(float));
}

/* §6.1, both per-layer mixers and the head mixer (which passes L == NULL and
 * gets no injection, because the file ships no output_hc_inject). Leaves the
 * block input in row `t0`..`t0+T-1` of the block-input buffer and the four
 * injection scalings in the injection rows.
 *
 * The three matrices are walks, so they are one walk for the chunk: at the
 * shipping width the hyper-connection pair was the last stage still reading its
 * weights once a token, worth 663 MiB of the 5.4 GiB a decode step moves, and
 * nothing about it is a recurrence. What stays per token is the elementwise
 * between the two walks, which is a few thousand floats. */
/* A prompt chunk's per-token work, a row at a time across the pool (ideas.md §7.33).
 *
 * Between the walks of a chunk sit loops that are a token's business alone - the mixers'
 * grouped norms and gates, the injection into the residual streams and the magnitude bands
 * the diagnostics keep, the gated-delta block's gates and causal convolution, the top-k of a
 * router row. Run on the master they were 3.3 ms of a 15.9 ms ingested token while eighteen
 * threads spun (`make decode --pp`'s serial and outside columns). Each row here is the same
 * function on the same inputs it was in the loop, so every value comes out the same bytes; the
 * one shared quantity, the diagnostics' running maxima and non-finite count, is kept per
 * thread and folded after the dispatch, and a maximum and a count do not care about order. */
#define ROWS_PAR_MIN 4
typedef struct rows_job rows_job;
struct rows_job {
    qe_forward *f;
    void (*fn)(rows_job *j, int t, int tid);
    int T;
    long next;
    const void *a, *b, *c, *d;
    float fa;
    long l0;
    unsigned flags;
    struct {
        float res, block;
        long nonfinite;
        char pad[52];
    } acc[64];
};

static void rows_task(void *v, int tid, int nthreads) {
    rows_job *j = v;
    for (;;) {
        const long t = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (t >= j->T)
            return;
        j->fn(j, (int)t, tid);
    }
}

/* Runs fn over rows [0, T); folds the diagnostics when `diag` says the rows swept anything. */
static void rows_run(qe_forward *f, rows_job *j, int diag) {
    j->f = f;
    j->next = 0;
    const int nt = f->n_threads > 0 ? f->n_threads : 1;
    if (diag)
        for (int i = 0; i < nt && i < 64; i++) {
            j->acc[i].res = f->d.max_abs_res;
            j->acc[i].block = f->d.max_abs_block;
            j->acc[i].nonfinite = 0;
        }
    if (!f->rows_par || j->T < ROWS_PAR_MIN || nt < 2 || nt > 64 || qe_pool_size() < 2)
        for (int t = 0; t < j->T; t++)
            j->fn(j, t, 0);
    else
        qe_pool_run(rows_task, j, nt);
    if (diag)
        for (int i = 0; i < nt && i < 64; i++) {
            if (j->acc[i].res > f->d.max_abs_res)
                f->d.max_abs_res = j->acc[i].res;
            if (j->acc[i].block > f->d.max_abs_block)
                f->d.max_abs_block = j->acc[i].block;
            f->d.nonfinite += j->acc[i].nonfinite;
        }
}

/* sweep(), against one thread's copy of the band. */
static void sweep_row(rows_job *j, int tid, const float *v, long n, int block) {
    float m = block ? j->acc[tid].block : j->acc[tid].res;
    for (long i = 0; i < n; i++) {
        const float a = fabsf(v[i]);
        if (!(a < 1e30f)) {
            j->acc[tid].nonfinite++;
            return;
        }
        if (a > m)
            m = a;
    }
    if (block)
        j->acc[tid].block = m;
    else
        j->acc[tid].res = m;
}

/* The three per-token passes of chunk_impl between its stages. */
static void row_sweep_cxn(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    sweep_row(j, tid, crow(f->cxn, f->K, t), f->K, 0);
}

/* A block's (flags & 1) or the experts' (flags & 2) output into the streams; flags & 4
 * zeroes it first (an ablation), flags & 8 sweeps the residual afterwards. */
static void row_inject(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    float *y = crow(f->cy, f->K, t);
    float *res = f->cres + (size_t)t * (size_t)f->hc_dim;
    const float *inj = f->cinj + (size_t)t * (size_t)f->HC;
    if (j->flags & 4)
        memset(y, 0, (size_t)f->K * sizeof(float));
    sweep_row(j, tid, y, f->K, 1);
    qe_stream_inject_add(res, y, inj, f->K, f->HC);
    if (j->flags & 8)
        sweep_row(j, tid, res, f->hc_dim, 0);
}

/* The mixer's three per-token loops. */
static void row_mix_norm(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    const int HCW = f->hc_dim;
    const int t0 = (int)j->fa;
    qe_rms_norm_grouped(crow(f->cres, HCW, t0 + t), (const float *)j->a, f->K,
                        crow((float *)j->b, HCW, t), f->K, f->HC, f->eps);
}

static void row_mix_silu(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    const int LR = f->low_rank;
    const float q = 1.0f / (float)f->HC;
    float *l = crow((float *)j->a, LR, t);
    for (int i = 0; i < LR; i++)
        l[i] *= q;
    qe_silu(l, l, LR);
}

static void row_mix_gate(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    const int HCW = f->hc_dim;
    const int t0 = (int)j->fa;
    const float q = 1.0f / (float)f->HC;
    qe_stream_gate_mean(crow((float *)j->a, HCW, t), crow((float *)j->b, HCW, t),
                        crow(f->cxn, f->K, t0 + t), f->K, f->HC);
    if (j->flags & 1) {
        float *ir = crow((float *)j->c, f->HC, t);
        for (int s = 0; s < f->HC; s++)
            ir[s] *= q;
    }
}

static void hc_mixer_t(qe_forward *f, const qe_tensor *norm, const qe_tensor *down,
                       const qe_tensor *up, const qe_tensor *inj, int t0, int T,
                       int inject_on) {
    const int HCW = f->hc_dim, LR = f->low_rank;
    const int rows = f->chunk_T;
    f->chunk_T = T;

    /* The three scratch buffers a projection reads and writes are walked from
     * their own base, so `proj` is handed row `t0` of each and not row 0. The
     * residual and the block input are indexed absolutely because they belong to
     * the chunk, not to this mixer: a probe head runs the mixer for one token of
     * a chunk in flight, and reading row 0 of `hn` for it is reading another
     * token's mixer. */
    float *hn = crow(f->hn, HCW, t0), *lo = crow(f->lo, LR, t0), *gate = crow(f->gate, HCW, t0);
    float *cinj = f->cinj + (size_t)t0 * (size_t)f->HC;

    /* Grouped by stream: gamma_stride is the stream width, and one flat
     * reduction over the concatenation is a different network. */
    {
        rows_job rj = { .fn = row_mix_norm, .T = T, .a = f32v(f, norm), .b = hn, .fa = (float)t0 };
        rows_run(f, &rj, 0);
    }
    if (inj && T == 1 && f->step_fuse) {
        const qe_tensor *tt[2] = { down, inj };
        const float *xx[2] = { hn, hn };
        float *yy[2] = { lo, cinj };
        gemv_group(f, 2, tt, xx, yy);
    } else {
        proj(f, down, hn, HCW, lo, LR);
        if (inj)
            proj(f, inj, hn, HCW, cinj, f->HC);
    }
    {
        rows_job rj = { .fn = row_mix_silu, .T = T, .a = lo };
        rows_run(f, &rj, 0);
    }
    proj(f, up, lo, LR, gate, HCW);
    {
        rows_job rj = { .fn = row_mix_gate, .T = T, .a = hn, .b = gate, .c = cinj,
                        .fa = (float)t0, .flags = inject_on ? 1u : 0u };
        rows_run(f, &rj, 0);
    }
    f->chunk_T = rows;
}

/* The same mixer read out of a layer, with the trunk's head mixer (L == NULL, which
 * ships no injection because there is nothing left to add the result to) and the
 * draft head's block as the two special cases it always was. */
static void hc_mixer(qe_forward *f, const qe_layer *L, int h, int t0, int T, int inject_on) {
    const qe_tensor *norm = L ? L->hc_norm[h] : f->head_norm;
    const qe_tensor *down = L ? L->hc_down[h] : f->head_down;
    const qe_tensor *up = L ? L->hc_up[h] : f->head_up;
    const qe_tensor *inj = inject_on ? (L ? L->hc_inject[h] : NULL) : NULL;
    hc_mixer_t(f, norm, down, up, inj, t0, T, inject_on);
}

static void hc_combine(qe_forward *f, const float *y, const float *inject) {
    qe_stream_inject_add(f->res, y, inject, f->K, f->HC);
}

/* §6.3. The state update is row-major S[j][i] with both the read of k and the
 * read of q contracting over i, which is the upstream kernel's form; rows are
 * independent, so the unit of work is (value head, row range) - and the tokens of
 * a chunk belong inside that unit, not outside it.
 *
 * The delta rule is a recurrence and it stays one: a chunk's tokens are walked in
 * position order, each doing the same three passes over the same row it always
 * did, so the arithmetic of row r of head h is the same sequence of floats whether
 * it was fed one token or sixteen. What changes is where the row comes from. The
 * state is 113 MiB read and written again at every token, 226 MiB of traffic a
 * token, and it was the one per-token read a chunk could not share; holding a row
 * while its tokens go by makes that 226/T and keeps the row in L1 for as long as
 * it is needed. */
typedef struct {
    float *S;                  /* nheads * hs * hs, updated in place */
    const float *co;           /* T rows of gdn_channels, each one [q|k|v] */
    const float *beta;   /* T rows of nheads: the write gate */
    const float *gexp;   /* T rows of nheads: exp of the log-decay, taken once a token */
    float *y;                  /* T rows of nheads * hs */
    int T, nheads, hs, kdiv, nk, gva_mod;
    /* Which row of the chunk buffers the T tokens start at. A whole chunk starts
     * at row 0; a window checkpoints a boundary after every row, which means it
     * walks one row at a time, and a row read at row 0 is the first token's
     * projections walked again rather than that row's own. */
    int row0;
    float scale;
    long costep, ystep; /* a token's stride into co, and into y */
    /* Units handed out from a cursor when `dyn` is set: a unit is DELTA_ROWS rows of one
     * head's state, independent of every other, so which thread walks it moves nothing. */
    int dyn;
    long next;
} delta_job;

#define DELTA_ROWS 8

/* The two passes over one row of state look like the place a vector unit should
 * earn its keep, and are not: the row is 512 bytes and hot in L1 between the
 * prediction pass and the update, so the third touch costs nothing. Hand-writing
 * the intrinsics for hs == 128 measured the same 48.2 ms block stage to the
 * tenth of a millisecond, so the scalar form stays - it is the same arithmetic,
 * and the state walk below it is memory-bound at the width the delta rule is
 * defined at.
 *
 * The update loop carries a runtime alias check, and it is not a missing
 * annotation. Marking all six of S, q, k, Sr, v and y restrict - each of which
 * is true on its own terms, the state is its own arena and the two vectors are
 * head slices of one conv_out - leaves delta_task's machine code byte for byte
 * the same, with both loops vectorised before and after. What that check is
 * asking is whether qh and kh overlap, and their offsets are qk * hs and
 * (nk + qk) * hs with qk known only at runtime, so no declaration answers it.
 * Left unmarked: the check is the overlap test, and a marking that changed
 * nothing would only be a claim about this file. */
/* The same unit of work - DELTA_ROWS rows of one head's state while a chunk's tokens go by -
 * with the rows walked together rather than one after another (ideas.md §7.34). The scalar
 * form is two dot products and an update per row per token, each one accumulation chain into a
 * horizontal sum, so a core spends its time waiting on FMA latency: 85 M FMA an ingested
 * token at a few per cent of the FP32 rate. Here the prediction of all eight rows is eight
 * independent accumulators over the same k loads, folded by one transposing reduction; the
 * update and read-out keep eight read-out accumulators in flight the same way. Same
 * operations on the same values, summed in a different order: priced in nats, and the chunk
 * and the step share it, so `make forward`'s identity legs between them hold. */
static inline __m256 hsum8x8(__m256 a0, __m256 a1, __m256 a2, __m256 a3, __m256 a4, __m256 a5,
                             __m256 a6, __m256 a7) {
    const __m256 t0 = _mm256_hadd_ps(a0, a1), t1 = _mm256_hadd_ps(a2, a3);
    const __m256 t2 = _mm256_hadd_ps(a4, a5), t3 = _mm256_hadd_ps(a6, a7);
    const __m256 u0 = _mm256_hadd_ps(t0, t1), u1 = _mm256_hadd_ps(t2, t3);
    return _mm256_add_ps(_mm256_permute2f128_ps(u0, u1, 0x20), _mm256_permute2f128_ps(u0, u1, 0x31));
}

static void delta_task_simd(void *vp, int tid, int nthreads) {
    delta_job *j = vp;
    const int hs = j->hs;
    const long per_head = hs / DELTA_ROWS;
    const long total = (long)j->nheads * per_head;
    long lo, hi;
    qe_slice(total, tid, nthreads, &lo, &hi);
    if (j->dyn) {
        lo = 0;
        hi = total;
    }
    for (long c0 = lo; c0 < hi; c0++) {
        long c = c0;
        if (j->dyn) {
            c = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (c >= total)
                break;
        }
        const int h = (int)(c / per_head);
        const int r0 = (int)((c % per_head) * DELTA_ROWS);
        const int qk = j->gva_mod ? h % j->nk : h / j->kdiv;
        float *S = j->S + (size_t)h * (size_t)hs * (size_t)hs + (size_t)r0 * (size_t)hs;
        for (int t = j->row0; t < j->row0 + j->T; t++) {
            const float *cob = j->co + (long)t * j->costep;
            const float *qh = cob + (size_t)qk * hs;
            const float *kh = cob + (size_t)(j->nk + qk) * hs;
            const float ea = j->gexp[(long)t * j->nheads + h];
            const float be = j->beta[(long)t * j->nheads + h];
            __m256 p0 = _mm256_setzero_ps(), p1 = p0, p2 = p0, p3 = p0, p4 = p0, p5 = p0, p6 = p0,
                   p7 = p0;
            for (int i = 0; i < hs; i += 8) {
                const __m256 k = _mm256_loadu_ps(kh + i);
                p0 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 0 * hs + i), k, p0);
                p1 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 1 * hs + i), k, p1);
                p2 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 2 * hs + i), k, p2);
                p3 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 3 * hs + i), k, p3);
                p4 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 4 * hs + i), k, p4);
                p5 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 5 * hs + i), k, p5);
                p6 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 6 * hs + i), k, p6);
                p7 = _mm256_fmadd_ps(_mm256_loadu_ps(S + 7 * hs + i), k, p7);
            }
            const __m256 pred = hsum8x8(p0, p1, p2, p3, p4, p5, p6, p7);
            const __m256 v = _mm256_loadu_ps(cob + (size_t)(2 * j->nk + h) * hs + r0);
            const __m256 g = _mm256_set1_ps(ea);
            float d[8];
            _mm256_storeu_ps(d, _mm256_mul_ps(_mm256_sub_ps(v, _mm256_mul_ps(pred, g)),
                                              _mm256_set1_ps(be)));
            __m256 o0 = _mm256_setzero_ps(), o1 = o0, o2 = o0, o3 = o0, o4 = o0, o5 = o0, o6 = o0,
                   o7 = o0;
#define DELTA_ROW(r, o)                                                                           \
    do {                                                                                          \
        const __m256 s_ = _mm256_fmadd_ps(k, _mm256_set1_ps(d[r]),                                \
                                          _mm256_mul_ps(_mm256_loadu_ps(S + (r) * hs + i), g));  \
        _mm256_storeu_ps(S + (r) * hs + i, s_);                                                   \
        o = _mm256_fmadd_ps(s_, q, o);                                                            \
    } while (0)
            for (int i = 0; i < hs; i += 8) {
                const __m256 k = _mm256_loadu_ps(kh + i), q = _mm256_loadu_ps(qh + i);
                DELTA_ROW(0, o0);
                DELTA_ROW(1, o1);
                DELTA_ROW(2, o2);
                DELTA_ROW(3, o3);
                DELTA_ROW(4, o4);
                DELTA_ROW(5, o5);
                DELTA_ROW(6, o6);
                DELTA_ROW(7, o7);
            }
#undef DELTA_ROW
            const __m256 out = _mm256_mul_ps(hsum8x8(o0, o1, o2, o3, o4, o5, o6, o7),
                                             _mm256_set1_ps(j->scale));
            _mm256_storeu_ps(j->y + (long)t * j->ystep + (size_t)h * hs + r0, out);
        }
    }
}

static void delta_task(void *vp, int tid, int nthreads) {
    delta_job *j = vp;
    const int hs = j->hs;
    const long per_head = hs / DELTA_ROWS;
    const long total = (long)j->nheads * per_head;
    long lo, hi;
    qe_slice(total, tid, nthreads, &lo, &hi);
    if (j->dyn) {
        lo = 0;
        hi = total;
    }
    for (long c0 = lo; c0 < hi; c0++) {
        long c = c0;
        if (j->dyn) {
            c = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (c >= total)
                break;
        }
        const int h = (int)(c / per_head);
        const int r0 = (int)((c % per_head) * DELTA_ROWS);
        /* Which of the 16 key heads a value head reads. einops writes the group
         * as "(h g)", which is h / kdiv; ggml_repeat tiles, which is h % nk.
         * Both are self-consistent and neither is a bug on its own terms, so
         * the checkpoint decides - see QE_VARIANT_GDN_GVA_MOD. */
        const int qk = j->gva_mod ? h % j->nk : h / j->kdiv;
        float *S = j->S + (size_t)h * (size_t)hs * (size_t)hs;
        /* The decay and the write gate are one number per head per token, and a
         * unit is DELTA_ROWS rows of one head, so taking them once per unit is
         * what keeps expf out of the row loop. The decay is not computed here at
         * all: gdn_core_chunk already made it once for the token, and sixteen
         * value heads of one key head recomputing the same 48 exponentials was the
         * one thing in the stage that read the same number seventeen times. */
        float ea[QE_CHUNK_MAX], be[QE_CHUNK_MAX];
        for (int t = j->row0; t < j->row0 + j->T; t++) {
            ea[t] = j->gexp[(long)t * j->nheads + h];
            be[t] = j->beta[(long)t * j->nheads + h];
        }
        for (int r = r0; r < r0 + DELTA_ROWS; r++) {
            float *Sr = S + (size_t)r * (size_t)hs;
            for (int t = j->row0; t < j->row0 + j->T; t++) {
                const float *cob = j->co + (long)t * j->costep;
                const float *qh = cob + (size_t)qk * hs;
                const float *kh = cob + (size_t)(j->nk + qk) * hs;
                float pred = 0.0f, out = 0.0f;
                for (int i = 0; i < hs; i++)
                    pred += Sr[i] * kh[i];
                const float d =
                    (cob[(size_t)(2 * j->nk + h) * hs + r] - pred * ea[t]) * be[t];
                for (int i = 0; i < hs; i++) {
                    Sr[i] = Sr[i] * ea[t] + kh[i] * d;
                    out += Sr[i] * qh[i];
                }
                j->y[(long)t * j->ystep + (size_t)h * hs + r] = out * j->scale;
            }
        }
    }
}

/* The four projections a GDN block reads, each one walk for the whole chunk. */
static void gdn_proj(qe_forward *f, const qe_layer *L) {
    const long C = f->gdn_channels, Z = (long)f->head_state * f->n_gdn_heads;
    const int nh = f->n_gdn_heads;

    if (f->chunk_T == 1 && f->step_fuse) {
        const qe_tensor *t[4] = { L->qkv, L->zgate, L->ssm_beta, L->ssm_alpha };
        const float *x[4] = { f->cxn, f->cxn, f->cxn, f->cxn };
        float *y[4] = { f->qkv, f->z, f->beta, f->alpha };
        gemv_group(f, 4, t, x, y);
        return;
    }
    proj(f, L->qkv, f->cxn, f->K, f->qkv, C);
    proj(f, L->zgate, f->cxn, f->K, f->z, Z);
    proj(f, L->ssm_beta, f->cxn, f->K, f->beta, nh);
    proj(f, L->ssm_alpha, f->cxn, f->K, f->alpha, nh);
}

/* §6.3, in three parts, because a chunk can share a weight walk and only half a
 * recurrence. gdn_proj reads the mixer's output for the whole chunk in one walk
 * of [q|k|v], the gate and the two scalar projections; gdn_core_chunk is the part
 * where a token's answer depends on the token before it, and gdn_out walks the
 * read-out once for the chunk. With a chunk of one this is the same six dispatches
 * the decode step always made, on the same memory, and proj() gives gemv().
 *
 * Inside the core the two recurrences separate again. The conv ring is a recurrence
 * across channels' history and stays per token, in position order, one row of
 * conv_out per token. The delta rule is a recurrence along rows only, so its token
 * loop moves inside its row loop and a row of state is read once per chunk instead
 * of once per token - 226 MiB of state traffic a token becoming 226/T. Each
 * (row, token) pair still sees the same sequence of floats it saw when the core was
 * called one token at a time, so what this writes is bit-identical to what the
 * per-token form wrote, which is what `make forward` asserts.
 *
 * `row0` is where those T rows start in the chunk buffers, and it is not always 0.
 * A window that keeps a boundary after every row has to walk its rows one at a
 * time, and a row of a window is row `t` of the chunk, not row 0: read at 0 it is
 * the first token's projections walked a second time, and every row past the first
 * never gets written at all. */
/* One row of gdn_core_chunk's prologue: the gates, the causal convolution, the silu and
 * the l2 norm, with the taps read from the chunk's own earlier rows where they are in it. */
static void row_gdn_pre(rows_job *j, int ti, int tid) {
    qe_forward *f = j->f;
    const qe_layer *L = j->a;
    const float *hist = j->b;
    const int row0 = (int)j->fa, hist_rows = (int)j->flags;
    const int t = row0 + ti;
    const long C = f->gdn_channels;
    const int hs = f->head_state, nh = f->n_gdn_heads, nk = f->n_gdn_kheads;
    const int kh = f->gdn_kernel;
    const float *sa = f32v(f, L->ssm_a), *dt = f32v(f, L->ssm_dt);
    const float *w = f32v(f, L->conv);
    const long pos = j->l0 + ti;
    const float *qkv = crow(f->qkv, C, t);
    float *alpha = crow(f->alpha, nh, t), *beta = crow(f->beta, nh, t);
    float *co = crow(f->conv_out, C, t);
    for (int h = 0; h < nh; h++)
        alpha[h] += dt[h];
    qe_sigmoid(beta, beta, nh);
    qe_softplus(alpha, alpha, nh);
    for (int h = 0; h < nh; h++)
        alpha[h] *= sa[h];
    float *gexp = crow(f->gexp, nh, t);
    if (f->exp2_simd)
        qe_exp2v(alpha, gexp, nh);
    else
        for (int h = 0; h < nh; h++)
            gexp[h] = expf(alpha[h]);
    /* tap i is the token hist_rows - i positions back */
    const float *tap[QE_GDN_TAP_MAX];
    for (int i = 0; i < hist_rows; i++) {
        const int back = hist_rows - i;
        tap[i] = ti - back >= 0 ? crow(f->qkv, C, t - back)
                                : hist + (size_t)((pos + i) % hist_rows) * (size_t)C;
    }
    for (long c = 0; c < C; c++) {
        const float *wc = w + (size_t)c * (size_t)kh;
        float acc = wc[kh - 1] * qkv[c];
        for (int i = 0; i < hist_rows; i++)
            acc += wc[i] * tap[i][c];
        co[c] = acc;
    }
    qe_silu(co, co, C);
    if (f->o.variant & QE_VARIANT_L2NORM_ROOT)
        qe_l2_norm_grouped_under(co, co, hs, 2 * nk, f->eps);
    else
        qe_l2_norm_grouped(co, co, hs, 2 * nk, f->eps);
}

static void row_gdn_post(rows_job *j, int ti, int tid) {
    qe_forward *f = j->f;
    const qe_layer *L = j->a;
    const int t = (int)j->fa + ti;
    const int hs = f->head_state, nh = f->n_gdn_heads;
    const long Z = (long)hs * nh;
    float *att_row = crow(f->att, Z, t);
    float *att_norm = crow(f->att_norm, Z, t);
    const float *z = crow(f->z, Z, t);
    qe_rms_norm_grouped(att_row, f32v(f, L->ssm_norm), 0, att_norm, hs, nh, f->eps);
    qe_sigmoid(z, att_row, (long)nh * hs);
    for (long i = 0; i < (long)nh * hs; i++)
        att_norm[i] *= att_row[i];
}

static void gdn_core_chunk(qe_forward *f, const qe_layer *L, int slot, int s, long pos0,
                           int row0, int T) {
    const long C = f->gdn_channels;
    const int hs = f->head_state, nh = f->n_gdn_heads, nk = f->n_gdn_kheads;
    const int kh = f->gdn_kernel, hist_rows = kh - 1;
    const long Z = (long)hs * nh;
    const int tend = row0 + T;
    const long sub = gdn_sub(f, slot, s);

    const float *sa = f32v(f, L->ssm_a), *dt = f32v(f, L->ssm_dt);
    const float *w = f32v(f, L->conv); /* [channel][tap] */
    float *hist = f->gconv + (size_t)sub * (size_t)hist_rows * (size_t)C;

    if (T >= ROWS_PAR_MIN && f->rows_par) {
        /* Every row at once: a row's taps are the rows before it in this chunk, or the ring
         * for the first `hist_rows` of them, and the ring is only written once all rows have
         * read it - the same values in the same order as the loop below (§7.33). */
        rows_job rj = { .fn = row_gdn_pre, .T = T, .a = L, .b = hist, .fa = (float)row0 };
        rj.flags = (unsigned)hist_rows;
        rj.l0 = pos0;
        rows_run(f, &rj, 0);
        for (int t = tend - hist_rows > row0 ? tend - hist_rows : row0; t < tend; t++) {
            const long pos = pos0 + (t - row0);
            memcpy(hist + (size_t)(pos % hist_rows) * (size_t)C, crow(f->qkv, C, t),
                   (size_t)C * sizeof(float));
        }
    } else
    for (int t = row0; t < tend; t++) {
        const long pos = pos0 + (t - row0);
        const float *qkv = crow(f->qkv, C, t);
        float *alpha = crow(f->alpha, nh, t), *beta = crow(f->beta, nh, t);
        float *co = crow(f->conv_out, C, t);

        for (int h = 0; h < nh; h++)
            alpha[h] += dt[h];
        qe_sigmoid(beta, beta, nh);
        /* alpha is the log-decay: ssm_a is -exp(A_log) and the gate it multiplies is
         * the softplus of the projection plus that bias, which is the form that
         * cannot overflow. */
        qe_softplus(alpha, alpha, nh);
        for (int h = 0; h < nh; h++)
            alpha[h] *= sa[h];

        /* ... and the decay itself is its own row, because sixteen value heads of
         * one key head each wanted the same exp of the same number. The table is
         * the whole point of this row (27,648 of a token's exponentials against
         * 1,728 of them); the wide form of the exponential is what makes
         * QE_EXP2_SIMD worth anything, and it needs a contiguous run to widen. */
        float *gexp = crow(f->gexp, nh, t);
        if (f->exp2_simd)
            qe_exp2v(alpha, gexp, nh);
        else
            for (int h = 0; h < nh; h++)
                gexp[h] = expf(alpha[h]);

        /* Depthwise causal conv over [history | qkv], oldest tap first. The history
         * is a ring indexed by position mod (kernel-1), so tap t of position pos is
         * row (pos+t) mod (kernel-1) and the current row overwrites the oldest tap's
         * slot - which is why it is written only after the output is computed.
         *
         * The three taps are three row pointers taken once a token, not a modulo per
         * channel: kernel-1 is 4-1 = 3, so the modulo is a division the compiler
         * cannot fold, and it sat in the innermost loop of a 10240-channel walk. The
         * arithmetic per output element is unchanged, order included. */
        const float *tap[QE_GDN_TAP_MAX];
        for (int i = 0; i < hist_rows; i++)
            tap[i] = hist + (size_t)((pos + i) % hist_rows) * (size_t)C;
        for (long c = 0; c < C; c++) {
            const float *wc = w + (size_t)c * (size_t)kh;
            float acc = wc[kh - 1] * qkv[c];
            for (int i = 0; i < hist_rows; i++)
                acc += wc[i] * tap[i][c];
            co[c] = acc;
        }
        qe_silu(co, co, C);
        memcpy(hist + (size_t)(pos % hist_rows) * (size_t)C, qkv, (size_t)C * sizeof(float));

        /* q and k are l2-normalised per 128-wide head - not RMS-normalised: no width
         * in the denominator, and ggml_l2_norm floors the norm at epsilon rather
         * than adding epsilon under the square root. v, which is everything past the
         * key heads, is left alone. */
        if (f->o.variant & QE_VARIANT_L2NORM_ROOT)
            qe_l2_norm_grouped_under(co, co, hs, 2 * nk, f->eps);
        else
            qe_l2_norm_grouped(co, co, hs, 2 * nk, f->eps);
    }

    delta_job dj = { .S = f->gdn + (size_t)sub * (size_t)nh * (size_t)hs * (size_t)hs,
                     .co = f->conv_out,
                     .gexp = f->gexp,
                     .beta = f->beta,
                     .y = f->att,
                     .T = T,
                     .row0 = row0,
                     .nheads = nh,
                     .hs = hs,
                     .kdiv = nh / nk,
                     .nk = nk,
                     .gva_mod = (f->o.variant & QE_VARIANT_GDN_GVA_MOD) != 0,
                     .scale = (f->o.variant & QE_VARIANT_GDN_NOSCALE) ? 1.0f : f->gdn_scale,
                     .costep = C,
                     .ystep = Z,
                     .dyn = f->dyn_bytes > 0 };
    qe_pool_run(f->delta_simd && dj.hs % 8 == 0 ? delta_task_simd : delta_task, &dj, f->n_threads);

    /* One gamma shared by the 48 heads (gamma_stride 0), and the output gate is a
     * sigmoid, not the silu Qwen3.5's GDN uses - the one numeric difference the
     * file's architecture notes call out. Neither of these touches state, so both
     * run after the walk, one row of the chunk at a time. */
    {
        rows_job rj = { .fn = row_gdn_post, .T = T, .a = L, .fa = (float)row0 };
        rows_run(f, &rj, 0);
    }
}


static void gdn_out(qe_forward *f, const qe_layer *L) {
    const long Z = (long)f->head_state * f->n_gdn_heads;
    proj(f, L->ssm_out, f->att_norm, Z, f->cy, f->K);
}

/* §6.4's selection: the blocks strictly before the query's own block compete on
 * the rectified summed score, and the query's own partial block is always
 * attended. That is where the "2051 cells no matter how long the context" of
 * §4 comes from, and it is why the current cell cannot be left to the ranking:
 * a token that cannot see itself decays into a loop. */
static long top_blocks(const float *sc, long n, long k, int *blocks, float *heap_s,
                       int *heap_b) {
    long m = 0;
    if (k < 1)
        k = 1;
    for (long b = 0; b < n; b++) {
        const float s = sc[b];
        if (m >= k && s <= heap_s[0])
            continue;
        long i = m < k ? m : 0;
        heap_s[i] = s;
        heap_b[i] = (int)b;
        if (m < k) {
            m++;
            while (i > 0) { /* sift up: a new leaf can only violate its parent */
                long p = (i - 1) / 2;
                if (heap_s[p] <= heap_s[i])
                    break;
                float ts = heap_s[p];
                heap_s[p] = heap_s[i];
                heap_s[i] = ts;
                int tb = heap_b[p];
                heap_b[p] = heap_b[i];
                heap_b[i] = tb;
                i = p;
            }
            continue;
        }
        for (;;) { /* sift down: the root was replaced by a larger score */
            long l = 2 * i + 1, r = 2 * i + 2, best = i;
            if (l < m && heap_s[l] < heap_s[best])
                best = l;
            if (r < m && heap_s[r] < heap_s[best])
                best = r;
            if (best == i)
                break;
            float ts = heap_s[best];
            heap_s[best] = heap_s[i];
            heap_s[i] = ts;
            int tb = heap_b[best];
            heap_b[best] = heap_b[i];
            heap_b[i] = tb;
            i = best;
        }
    }
    for (long i = 0; i < m; i++)
        blocks[i] = heap_b[i];
    return m;
}

/* The same set as top_blocks, found by threshold instead of by heap (QE_TOPK_RADIX, on).
 *
 * The heap is serial on one thread per token per layer, and on scores that rise with
 * position - the recent past outscoring the distant one, which is what these scores do -
 * nearly every block displaces the root: ~1 ms a layer at 32k blocks. A score is a sum of
 * rectified dots times a positive scale, so it is never negative and its IEEE bits order
 * as unsigned integers; three histogram passes find the k-th largest bit pattern, and the
 * blocks at or above it are the answer. That is the heap's set exactly whenever the
 * threshold value is not tied across the boundary - every block above it is in both, and
 * so is every block equal to it - so when the copies of the k-th value do not all fit the
 * function returns -1 and the caller asks the heap, whose tie order is its own. The
 * blocks come back ascending; every caller hands them to a bitmap, so order is moot. */
static int topk_radix_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_TOPK_RADIX");
        cached = e && *e ? atoi(e) != 0 : 1;
    }
    return cached;
}

static long topk_radix(const float *sc, long n, long k, int *blocks) {
    static __thread uint32_t *cand;
    static __thread long cand_n;
    if (cand_n < n) {
        free(cand);
        cand = malloc((size_t)n * sizeof *cand);
        cand_n = cand ? n : 0;
        if (!cand)
            return -1;
    }
    const uint32_t *u = (const uint32_t *)sc;
    uint32_t hist[4096];
    /* pass 1: the top twelve bits of every score */
    memset(hist, 0, sizeof hist);
    for (long i = 0; i < n; i++) {
        if (u[i] & 0x80000000u)
            return -1; /* a negative or NaN score is not this function's to order */
        hist[u[i] >> 20]++;
    }
    long need = k, b1 = 4095;
    for (; b1 >= 0; b1--) {
        if (hist[b1] >= (uint32_t)need)
            break;
        need -= hist[b1];
    }
    if (b1 < 0)
        return -1;
    long nc = 0;
    for (long i = 0; i < n; i++)
        if ((u[i] >> 20) == (uint32_t)b1)
            cand[nc++] = u[i];
    /* pass 2: the next twelve, among the boundary bucket */
    memset(hist, 0, sizeof hist);
    for (long i = 0; i < nc; i++)
        hist[(cand[i] >> 8) & 0xfffu]++;
    long b2 = 4095;
    for (; b2 >= 0; b2--) {
        if (hist[b2] >= (uint32_t)need)
            break;
        need -= hist[b2];
    }
    long nc2 = 0;
    for (long i = 0; i < nc; i++)
        if (((cand[i] >> 8) & 0xfffu) == (uint32_t)b2)
            cand[nc2++] = cand[i];
    /* pass 3: the last eight */
    memset(hist, 0, 256 * sizeof hist[0]);
    for (long i = 0; i < nc2; i++)
        hist[cand[i] & 0xffu]++;
    long b3 = 255;
    for (; b3 >= 0; b3--) {
        if (hist[b3] >= (uint32_t)need)
            break;
        need -= hist[b3];
    }
    if (hist[b3] != (uint32_t)need)
        return -1; /* the k-th value is tied across the boundary: the heap decides */
    const uint32_t tau = ((uint32_t)b1 << 20) | ((uint32_t)b2 << 8) | (uint32_t)b3;
    long m = 0;
    for (long i = 0; i < n; i++)
        if (u[i] >= tau)
            blocks[m++] = (int)i;
    return m == k ? m : -1;
}

static long select_blocks(const float *sc, long n, long k, int *blocks, float *heap_s,
                          int *heap_b) {
    if (topk_radix_env()) {
        const long m = topk_radix(sc, n, k, blocks);
        if (m >= 0)
            return m;
    }
    return top_blocks(sc, n, k, blocks, heap_s, heap_b);
}

/* Softmax over the selected cells only, in the order the indexer's heap produced
 * them. Work is split by query head, which is the split that reads a KV group's
 * rows `per` times over: the 24 heads share 2 KV heads twelve ways, so a head's
 * pass over the selection is 2051 rows that its eleven siblings are about to read
 * again, and the sharing is left to the L3. `QE_ATTN_FUSED=0` runs this; the
 * default is the one walk per KV group below. */
typedef struct {
    const float *q; /* n_q_heads * head_dim */
    const float *kc, *vc;   /* the f32 cache */
    const int8_t *kqd, *vqd; /* the q8_0 cache: one byte per value ... */
    const float *kqs, *vqs;  /* ... and one scale per head of one cell */
    const int *sel;
    float *scores; /* n_q_heads * nsel */
    float *y;
    int nqh, nkvh, hd, krow, nsel;
    int q8;
    int exp2; /* QE_EXP2_SIMD, for the head-at-a-time form §6.6 named */
    float scale;
} attn_job;

static void attn_task(void *vp, int tid, int nthreads) {
    attn_job *j = vp;
    const int per = j->nqh / j->nkvh;
    long lo, hi;
    qe_slice(j->nqh, tid, nthreads, &lo, &hi);
    for (int h = (int)lo; h < (int)hi; h++) {
        const int g = h / per;
        const float *q = j->q + (size_t)h * (size_t)j->hd;
        float *s = j->scores + (size_t)h * (size_t)j->nsel;
        float m = -1e30f;
        for (int i = 0; i < j->nsel; i++) {
            const long cell = j->sel[i];
            float d;
            if (j->q8) {
                const int8_t *kp = j->kqd + (size_t)cell * (size_t)j->krow + (size_t)g * j->hd;
                d = qe_dot_q8(q, kp, j->hd) * j->kqs[(size_t)cell * (size_t)j->nkvh + g];
            } else {
                const float *kp = j->kc + (size_t)cell * (size_t)j->krow + (size_t)g * j->hd;
                d = qe_dot(q, kp, j->hd);
            }
            d *= j->scale;
            s[i] = d;
            if (d > m)
                m = d;
        }
        double sum = 0.0;
        if (j->exp2) {
            /* The same shift the libm arm folds into its call, and the same float
             * result: only the transcendental that follows it differs. */
            for (int i = 0; i < j->nsel; i++)
                s[i] -= m;
            qe_exp2v(s, s, j->nsel); /* one run of one head's scores */
        } else
            for (int i = 0; i < j->nsel; i++)
                s[i] = expf(s[i] - m);
        for (int i = 0; i < j->nsel; i++)
            sum += s[i];
        const float inv = (float)(1.0 / sum);
        float *y = j->y + (size_t)h * (size_t)j->hd;
        memset(y, 0, (size_t)j->hd * sizeof(float));
        for (int i = 0; i < j->nsel; i++) {
            const long cell = j->sel[i];
            if (j->q8)
                qe_axpy_q8(y, j->vqd + (size_t)cell * (size_t)j->krow + (size_t)g * j->hd,
                           s[i] * inv * j->vqs[(size_t)cell * (size_t)j->nkvh + g], j->hd);
            else
                qe_axpy(y, j->vc + (size_t)cell * (size_t)j->krow + (size_t)g * j->hd,
                        s[i] * inv, j->hd);
        }
    }
}

/* The winners, ascending.
 *
 * `top_blocks` hands back the blocks that outscored the rest in heap order, which
 * is a random tour of the cache arena: 2051 cells of 512 bytes touched in an order
 * that carries no information, twice per head, with nothing for a prefetcher to
 * follow and a page-table walk of its own making. Marking the winners in a bitmap
 * and walking the bitmap in address order costs one pass over `cur` bits - a
 * kilobyte at the context this checkpoint was trained on, against the two
 * megabytes the gather is about to read - and turns the same touches into a
 * forward stride whose runs are `ratio` cells long.
 *
 * The set of cells is exactly the set the heap won, so the softmax's denominator
 * is the same set; the terms arrive in address order rather than heap order, which
 * is a reassociation of an exp-weighted sum and is therefore answered by the nll
 * gate rather than by a claim of bit-identity. The bitmap is consumed here, so a
 * token's selection leaves nothing behind for the next one. */
static int sel_ascending_to(uint64_t *sel_bits, int *sel, const int *blocks, long nblocks,
                            long cur, int ratio, long pos) {
    for (long i = 0; i < nblocks; i++) {
        const long b = blocks[i];
        sel_bits[b >> 6] |= 1ull << (b & 63);
    }
    int nsel = 0;
    const long words = cur > 0 ? (cur + 63) / 64 : 0;
    for (long w = 0; w < words; w++) {
        uint64_t bits = sel_bits[w];
        sel_bits[w] = 0;
        while (bits) {
            const long b = 64 * w + __builtin_ctzll(bits);
            bits &= bits - 1;
            for (int r = 0; r < ratio; r++)
                sel[nsel++] = (int)(b * ratio + r);
        }
    }
    /* The query's own partial block is always attended, and it is the highest
     * address of the lot, so appending it keeps the walk ascending. */
    for (long c = cur * ratio; c <= pos; c++)
        sel[nsel++] = (int)c;
    return nsel;
}

static int sel_ascending(qe_forward *f, const int *blocks, long nblocks, long cur, int ratio,
                         long pos) {
    return sel_ascending_to(f->sel_bits, f->sel, blocks, nblocks, cur, ratio, pos);
}

/* The fused walk, chunked over the selection so that a KV group's one pass does
 * not mean two threads and nineteen idle cores: a unit is one KV group's chunk of
 * the cells, and its accumulator, maximum and sum are combined afterwards by
 * `qe_attn_combine`. Two dispatches where the head-at-a-time form had one, which
 * at the 1.6 us handshake of §12 is a few tens of microseconds a token against the
 * twelvefold gather it removes; and when the selection is small enough that one
 * chunk per group covers it, the chunks are one and the combine is a pair of
 * divides. */
typedef struct {
    const float *q; /* n_q_heads * head_dim */
    const float *kc, *vc;
    const int8_t *kqd, *vqd;
    const float *kqs, *vqs;
    const int *sel;
    float *py, *pm, *pl;
    int nsel, chunk, nchunks, units;
    int nqh, nkvh, per, hd, krow;
    int q8;
    int exp2; /* QE_EXP2_SIMD, carried to every unit of the walk */
    float scale;
    long next; /* units handed out from a cursor: a unit is one walk, so any thread may take it */
} attnf_job;

static void attnf_task(void *vp, int tid, int nthreads) {
    attnf_job *j = vp;
    for (;;) {
        const int u = (int)__atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (u >= j->units)
            return;
        const int g = u / j->nchunks, c = u % j->nchunks;
        const long first = (long)c * j->chunk;
        const long left = (long)j->nsel - first;
        qe_attn_unit au = { .q = j->q + (size_t)g * j->per * j->hd,
                            .kd = j->kqd, .vd = j->vqd, .kc = j->kc, .vc = j->vc,
                            .ks = j->kqs, .vs = j->vqs,
                            .sel = j->sel + first,
                            .y = j->py + (size_t)u * j->per * j->hd,
                            .m = j->pm + (size_t)u * j->per,
                            .l = j->pl + (size_t)u * j->per,
                            .nsel = (int)(left < j->chunk ? left : j->chunk),
                            .per = j->per, .hd = j->hd, .krow = j->krow,
                            .kv_head = g, .sstride = j->nkvh, .q8 = j->q8,
                            .exp2 = j->exp2,
                            .scale = j->scale };
        qe_attn_walk(&au);
    }
}

typedef struct {
    const float *py, *pm, *pl;
    float *y;
    int nchunks, per, nqh, hd;
} attnm_job;

static void attnm_task(void *vp, int tid, int nthreads) {
    attnm_job *j = vp;
    long lo, hi;
    qe_slice(j->nqh, tid, nthreads, &lo, &hi);
    for (int h = (int)lo; h < (int)hi; h++) {
        const int g = h / j->per, hl = h % j->per;
        const long base = (long)g * j->nchunks * j->per;
        qe_attn_combine(j->py + (base + hl) * j->hd, (long)j->per * j->hd,
                        j->pm + base + hl, j->pl + base + hl, j->per,
                        j->nchunks, j->hd, j->y + (size_t)h * j->hd);
    }
}

/* §6.4, split like the GDN block: five weight walks for the whole chunk, then the
 * indexer and the attention per token (each is a function of the cache as it
 * stands at that position), then the output projection once for the chunk. */
static void qsa_proj(qe_forward *f, const qe_layer *L) {
    const int krow = (int)f->kv_row, id = f->idx_dim, nih = f->n_idx_heads;
    const long qrow = 2 * (long)f->head_dim * f->n_q_heads;

    if (f->chunk_T == 1 && f->step_fuse) {
        const qe_tensor *t[5] = { L->aq, L->ixq, L->ak, L->av, L->ixk };
        const float *x[5] = { f->cxn, f->cxn, f->cxn, f->cxn, f->cxn };
        float *y[5] = { f->qg, f->iq, f->kcur, f->vcur, f->ikraw };
        gemv_group(f, 5, t, x, y);
        return;
    }
    proj(f, L->ixk, f->cxn, f->K, f->ikraw, id);
    proj(f, L->ixq, f->cxn, f->K, f->iq, (long)id * nih);
    proj(f, L->aq, f->cxn, f->K, f->qg, qrow);
    proj(f, L->ak, f->cxn, f->K, f->kcur, krow);
    proj(f, L->av, f->cxn, f->K, f->vcur, krow);
}

/* The indexer's block scores: the one part of a step whose cost grows with the
 * length of the conversation instead of with the shape of the model, because
 * `cur` is the context over the pooling ratio and every pooled block is a
 * rectified dot against each index head. At 8k that is 2048 blocks of four
 * 128-value dots per attention layer, on the master while nineteen workers wait,
 * and it is the term that makes a long session feel superlinear.
 *
 * A block's score depends on no other block's, so splitting the range over the
 * pool leaves every score bit-identical - the same four dots, in the same order,
 * summed the same way. Below a block count where the walk costs less than the
 * dispatch that would spread it, the same rows are computed by the master alone,
 * which is why the arithmetic lives in exactly one function. */
typedef struct {
    const float *iq;   /* n_idx_heads * idx_dim, already normed and rotated */
    const void *pool;  /* cur pooled index keys of idx_dim, floats or halves */
    int f16;
    float *iscore;
    long cur;
    float scale;
    int id, nih;
    long next; /* blocks handed out from a cursor, in runs of `run` */
    long run;
} idx_job;

static void idx_score_rows(const idx_job *j, long lo, long hi) {
    /* rectified and summed unweighted: every head weight is 1 */
    if (j->f16)
        qe_idx_scores_h(j->iq, j->pool, j->nih, j->id, j->scale, lo, hi, j->iscore);
    else
        qe_idx_scores(j->iq, j->pool, j->nih, j->id, j->scale, lo, hi, j->iscore);
}

/* Blocks handed out from a cursor rather than one fixed slice a thread: a score is a
 * function of its block alone, so who computed it moves nothing, and the slowest core of
 * the two kinds stops being the length of the dispatch (ideas.md §7.30). */
static void idx_score_task(void *vp, int tid, int nthreads) {
    idx_job *j = vp;
    long run = j->run;
    if (run <= 0) {
        long lo, hi;
        qe_slice(j->cur, tid, nthreads, &lo, &hi);
        idx_score_rows(j, lo, hi);
        return;
    }
    for (;;) {
        const long lo = __atomic_fetch_add(&j->next, run, __ATOMIC_RELAXED);
        if (lo >= j->cur)
            return;
        idx_score_rows(j, lo, lo + run > j->cur ? j->cur : lo + run);
    }
}

/* Sixty-four blocks is sixty-four sets of four 128-value dots, which is the same
 * order as the 1.6 us hot dispatch of `qe_pool_run` (§12): below it, spreading
 * the work costs more than doing it. A build may pin it above every reachable
 * block count, which is how the split is compared against the master-only path
 * it replaced. */
#ifndef QE_IDX_PAR_MIN
#define QE_IDX_PAR_MIN 64
#endif

static int32_t *seq_rpos(const qe_forward *f, int s) {
    return f->rpos + (size_t)s * f->seq_stride * 3;
}

/* The angles of one cell. A sequence that never held an image takes the path every
 * text pass always took, cell for position, and nothing in the table is read. One that
 * did reads the cell's (t, h, w), and a cell whose three agree is still ordinary rope
 * at that position - only an image's rows take the interleaved form, ggml's
 * GGML_ROPE_TYPE_IMROPE: pair k reads t, h or w by k mod 3 while k is inside that
 * section's share, every ladder accumulated by repeated multiplication as
 * qe_rope_init's is. */
static void rope_at(const qe_forward *f, int s, long cell, int rd, float *cs) {
    if (!f->mrope[s]) {
        qe_rope_init(cs, rd, cell, f->theta);
        return;
    }
    const int32_t *p = seq_rpos(f, s) + (size_t)cell * 3;
    if (p[0] == p[1] && p[1] == p[2]) {
        qe_rope_init(cs, rd, p[0], f->theta);
        return;
    }
    const float step = powf(f->theta, -2.0f / (float)rd);
    float th[3] = { (float)p[0], (float)p[1], (float)p[2] };
    const int s0 = f->msec[0], s1 = f->msec[1], s2 = f->msec[2];
    const int sect = s0 + s1 + s2 + f->msec[3];
    for (int k = 0; k < rd / 2; k++) {
        const int sector = sect > 0 ? k % sect : k;
        int c = 0; /* the fourth section is never non-zero in this file; t is its stand-in */
        if (sector % 3 == 1 && sector < 3 * s1)
            c = 1;
        else if (sector % 3 == 2 && sector < 3 * s2)
            c = 2;
        cs[2 * k] = cosf(th[c]);
        cs[2 * k + 1] = sinf(th[c]);
        th[0] *= step;
        th[1] *= step;
        th[2] *= step;
    }
}

static void qsa_core(qe_forward *f, const qe_layer *L, int slot, int s, long pos) {
    const int hd = f->head_dim, nqh = f->n_q_heads, nkvh = f->n_kv_heads;
    const int krow = (int)f->kv_row, id = f->idx_dim, nih = f->n_idx_heads;
    const int ratio = f->kpool_ratio;
    const int rd = f->n_rope < hd ? f->n_rope : hd;
    const int t = f->chunk_t;
    /* This sequence's cells, in this sequence's slice of the caches. Everything below
     * numbers them from 0, which is what keeps the selection, the block scores and the
     * top-k indifferent to how many sequences share the pass. */
    const long sub = qsa_sub(f, slot, s);
    const size_t NC = f->seq_stride;
    float cs[QE_MAX_ROPE], cs2[QE_MAX_ROPE];
    /* K and V of one layer, in one contiguous run: the second half starts where
     * the first one ends, in either layout. */
    float *kvc = f->kv ? f->kv + (size_t)sub * 2 * NC * (size_t)krow : NULL;
    float *kc = kvc, *vc = kvc ? kvc + NC * (size_t)krow : NULL;
    int8_t *kvd = f->kvd ? f->kvd + (size_t)sub * 2 * NC * (size_t)krow : NULL;
    float *kvs = f->kvs ? f->kvs + (size_t)sub * 2 * NC * (size_t)nkvh : NULL;
    float *krawc = f->kraw + (size_t)sub * NC * (size_t)id;
    void *kpoolc = kpool_seq(f, sub);
    float *ikraw = crow(f->ikraw, id, t), *iq = crow(f->iq, (long)id * nih, t);
    const float *qg = crow(f->qg, 2 * (long)hd * nqh, t);
    float *kcur = crow(f->kcur, krow, t), *vcur = crow(f->vcur, krow, t);
    float *attn_y = crow(f->attn_y, (long)hd * nqh, t);

    const double at0 = f->o.profile ? now() : 0.0;
    /* --- the indexer. Raw keys are cached un-normalised; a block is pooled,
     * then normed, then rotated by its first cell's angle, in that order. */
    memcpy(krawc + (size_t)pos * (size_t)id, ikraw, (size_t)id * sizeof(float));
    const long ncell = pos + 1;
    const long cur = pos / ratio;
    if (ncell % ratio == 0) {
        const long b = ncell / ratio - 1;
        for (int i = 0; i < id; i++) {
            float acc = 0.0f;
            for (int r = 0; r < ratio; r++)
                acc += krawc[((size_t)b * (size_t)ratio + (size_t)r) * (size_t)id + (size_t)i];
            f->ipool[i] = acc / (float)ratio;
        }
        qe_rms_norm_grouped(f->ipool, f32v(f, L->ixkn), 0, f->ipool, id, 1, f->eps);
        rope_at(f, s, b * ratio, rd, cs2);
        qe_rope_apply(f->ipool, 1, id, rd, cs2);
        kpool_put(f, kpoolc, b, f->ipool);
    }
    qe_rms_norm_grouped(iq, f32v(f, L->ixqn), 0, iq, id, nih, f->eps);
    rope_at(f, s, pos, rd, cs);
    qe_rope_apply(iq, nih, id, rd, cs);

    idx_job ij = { .iq = iq,
                   .pool = kpoolc,
                   .f16 = f->idx_f16,
                   .iscore = f->iscore,
                   .cur = cur,
                   .scale = f->idx_scale,
                   .id = id,
                   .nih = nih,
                   .run = f->dyn_bytes ? (cur / (4L * f->n_threads) < 16 ? 16
                                                                       : cur / (4L * f->n_threads))
                                       : 0 };
    if (cur >= QE_IDX_PAR_MIN)
        qe_pool_run(idx_score_task, &ij, f->n_threads);
    else
        idx_score_rows(&ij, 0, cur);
    long nblocks;
    if (cur <= f->idx_topk_blocks) {
        nblocks = cur;
        for (long b = 0; b < cur; b++)
            f->heap_b[b] = (int)b;
    } else {
        nblocks = select_blocks(f->iscore, cur, f->idx_topk_blocks, f->heap_b, f->heap_s, f->heap_b);
    }
    const int nsel = sel_ascending(f, f->heap_b, nblocks, cur, ratio, pos);
    const double at1 = f->o.profile ? now() : 0.0;

    /* --- the attention itself, over attn_q's [q | gate] rows of 512. */
    for (int h = 0; h < nqh; h++) {
        memcpy(f->qwork + (size_t)h * hd, qg + (size_t)h * 2 * hd, (size_t)hd * sizeof(float));
        memcpy(f->qgate + (size_t)h * hd, qg + (size_t)h * 2 * hd + hd,
               (size_t)hd * sizeof(float));
    }
    qe_rms_norm_grouped(f->qwork, f32v(f, L->aqn), 0, f->qwork, hd, nqh, f->eps);
    rope_at(f, s, pos, rd, cs);
    qe_rope_apply(f->qwork, nqh, hd, rd, cs);
    qe_rms_norm_grouped(kcur, f32v(f, L->akn), 0, kcur, hd, nkvh, f->eps);
    qe_rope_apply(kcur, nkvh, hd, rd, cs);
    if (kvd) {
        /* Quantized after the norm and the rotation, which is where the values
         * that are actually compared live: the scale then covers the range the
         * dot product would have read, and nothing before it is kept. */
        float *ksc = kvs + (size_t)pos * (size_t)nkvh;
        float *vsc = kvs + (NC + (size_t)pos) * (size_t)nkvh;
        int8_t *kd = kvd + (size_t)pos * (size_t)krow;
        int8_t *vd = kvd + (NC + (size_t)pos) * (size_t)krow;
        for (int g = 0; g < nkvh; g++) {
            qe_quant_q8_row(kcur + (size_t)g * hd, kd + (size_t)g * hd, ksc + g, hd);
            qe_quant_q8_row(vcur + (size_t)g * hd, vd + (size_t)g * hd, vsc + g, hd);
        }
    } else {
        memcpy(kc + (size_t)pos * (size_t)krow, kcur, (size_t)krow * sizeof(float));
        memcpy(vc + (size_t)pos * (size_t)krow, vcur, (size_t)krow * sizeof(float));
    }

    const int per = nqh / nkvh;
    if (f->attn_fused) {
        attnf_job fj = { .q = f->qwork,
                         .kc = kc,
                         .vc = vc,
                         .kqd = kvd,
                         .vqd = kvd ? kvd + NC * (size_t)krow : NULL,
                         .kqs = kvs,
                         .vqs = kvs ? kvs + NC * (size_t)nkvh : NULL,
                         .sel = f->sel,
                         .py = f->attn_py,
                         .pm = f->attn_pm,
                         .pl = f->attn_pl,
                         .nsel = nsel,
                         .nqh = nqh,
                         .nkvh = nkvh,
                         .per = per,
                         .hd = hd,
                         .exp2 = f->exp2_simd,
                         .krow = krow,
                         .q8 = kvd ? 1 : 0,
                         .scale = f->attn_scale };
        /* One chunk per worker of each KV group, so that no core is handed two
         * units of a stage that could have been split around it: `qe_slice` gives
         * equal shares, and at 20 units over 19 threads the makespan is two units
         * where at 18 it is one. The ceiling on that is the point where a chunk
         * stops being worth the handshake it is dispatched with. */
        const int nthr = f->n_threads > 0 ? f->n_threads : 1;
        int nchunks = nthr / nkvh;
        const int most = (nsel + QE_ATTN_CELL_MIN - 1) / QE_ATTN_CELL_MIN;
        if (nchunks > most)
            nchunks = most;
        if (nchunks < 1)
            nchunks = 1;
        int chunk = (nsel + nchunks - 1) / nchunks;
        if (chunk < 1)
            chunk = 1;
        fj.chunk = chunk;
        fj.nchunks = nchunks = (nsel + chunk - 1) / chunk;
        fj.units = nchunks * nkvh;
        qe_pool_run(attnf_task, &fj, f->n_threads);
        attnm_job mj = { .py = f->attn_py,
                         .pm = f->attn_pm,
                         .pl = f->attn_pl,
                         .y = attn_y,
                         .nchunks = nchunks,
                         .per = per,
                         .nqh = nqh,
                         .hd = hd };
        qe_pool_run(attnm_task, &mj, f->n_threads);
    } else {
        attn_job aj = { .q = f->qwork,
                        .kc = kc,
                        .vc = vc,
                        .kqd = kvd,
                        .vqd = kvd ? kvd + NC * (size_t)krow : NULL,
                        .kqs = kvs,
                        .vqs = kvs ? kvs + NC * (size_t)nkvh : NULL,
                        .sel = f->sel,
                        .scores = f->scores,
                        .y = attn_y,
                        .nqh = nqh,
                        .nkvh = nkvh,
                        .hd = hd,
                        .krow = krow,
                        .nsel = nsel,
                        .q8 = kvd ? 1 : 0,
                        .exp2 = f->exp2_simd,
                        .scale = f->attn_scale };
        qe_pool_run(attn_task, &aj, f->n_threads);
    }
    /* attn_gated in the graph multiplies the attention output by the sigmoid of
     * the gate half of each [q | gate] row, which is what f->qgate holds since
     * the split above. Reading the gate out of the query rows by flat index
     * instead walks the q halves and gate halves alternately, so half the heads
     * are gated by raw query values and the other half by the previous head's
     * gate. */
    qe_sigmoid(f->qgate, f->qgate, (long)nqh * hd);
    for (long i = 0; i < (long)nqh * hd; i++)
        attn_y[i] *= f->qgate[i];
    if (f->o.profile) {
        f->d.attn_idx_seconds += at1 - at0;
        f->d.attn_walk_seconds += now() - at1;
    }
}

/* How a token's selection is cut into chunks for the fused walk: the per-token path's
 * own rule, kept in one place because a chunk that cut it differently would combine its
 * softmax partials in a different order and stop being the same bytes as a step. */
static void attn_chunking(const qe_forward *f, int nsel, int *chunk_out, int *nchunks_out) {
    const int nthr = f->n_threads > 0 ? f->n_threads : 1;
    int nchunks = nthr / f->n_kv_heads;
    const int most = (nsel + QE_ATTN_CELL_MIN - 1) / QE_ATTN_CELL_MIN;
    if (nchunks > most)
        nchunks = most;
    if (nchunks < 1)
        nchunks = 1;
    int chunk = (nsel + nchunks - 1) / nchunks;
    if (chunk < 1)
        chunk = 1;
    *chunk_out = chunk;
    *nchunks_out = (nsel + chunk - 1) / chunk;
}

/* A prompt chunk's attention, all of its tokens at once (ideas.md §7.31).
 *
 * The per-token form asks the pool three times a token - the indexer's scores, the
 * fused walk, the combine - and a token's attention at a few thousand cells of context
 * is too little work to share nineteen ways: at 4096 tokens of context the block stage
 * of an ingested token was 5.7 ms, 2.3 of it nineteen threads waiting for the slowest
 * and 1.0 the master's top-k while they spun. Every token of a chunk reads a cache that
 * the tokens before it in the chunk have already written, so once the cheap per-token
 * prologue has run for all of them - the cache writes, the pooled indexer keys, the
 * normed and rotated queries - the rest is T independent problems, and the pool can be
 * handed all of them at once.
 *
 * Nothing about a token's arithmetic moves. Its indexer scores, its top-k and its
 * selection are the same functions on the same inputs; its walk is cut into chunks by
 * the same rule (`attn_chunking`), each chunk is the same `qe_attn_walk`, and the
 * combine folds the same partials in the same order. Only who ran it and when changes,
 * which is why `make forward`'s chunk-against-step identity legs still hold byte for
 * byte. */
typedef struct {
    qe_forward *f;
    const void *kpoolc;
    long pos0;
    int T, id, nih, ratio;
    int nsel[QE_CHUNK_MAX];
    long next;
} csel_job;

static void csel_task(void *vp, int tid, int nthreads) {
    csel_job *j = vp;
    qe_forward *f = j->f;
    float *isc = f->tiscore + (size_t)tid * (size_t)(f->o.n_ctx / f->kpool_ratio + 1);
    float *hs = f->theap + (size_t)tid * (size_t)(f->idx_topk_blocks + 1);
    int *hb = f->theapb + (size_t)tid * (size_t)(f->idx_topk_blocks + 1);
    int *blocks = hb; /* top_blocks writes its answer over its own heap, as the step does */
    uint64_t *bits = f->tbits + (size_t)tid * (size_t)f->tbits_words;
    for (;;) {
        const long t = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (t >= j->T)
            return;
        const long pos = j->pos0 + t;
        const long cur = pos / j->ratio;
        idx_job ij = { .iq = crow(f->iq, (long)j->id * j->nih, (int)t),
                       .pool = j->kpoolc,
                       .f16 = f->idx_f16,
                       .iscore = isc,
                       .cur = cur,
                       .scale = f->idx_scale,
                       .id = j->id,
                       .nih = j->nih };
        idx_score_rows(&ij, 0, cur);
        long nblocks;
        int *bl = blocks;
        if (cur <= f->idx_topk_blocks) {
            nblocks = cur;
            for (long b = 0; b < cur; b++)
                bl[b] = (int)b;
        } else {
            nblocks = select_blocks(isc, cur, f->idx_topk_blocks, bl, hs, hb);
        }
        j->nsel[t] = sel_ascending_to(bits, f->csel + (size_t)t * (size_t)f->sel_cap, bl,
                                      nblocks, cur, j->ratio, pos);
    }
}

typedef struct {
    qe_forward *f;
    const float *kc, *vc;
    const int8_t *kqd, *vqd;
    const float *kqs, *vqs;
    int t0, nt;            /* the tokens of this group */
    int uoff[ATTN_GROUP + 1]; /* first unit of each token of the group */
    int chunk[ATTN_GROUP], nchunks[ATTN_GROUP];
    const int *nsel;
    int units;
    long next, next2;
} cattn_job;

static void cattn_walk_task(void *vp, int tid, int nthreads) {
    cattn_job *j = vp;
    qe_forward *f = j->f;
    const int per = f->n_q_heads / f->n_kv_heads, hd = f->head_dim;
    for (;;) {
        const int u = (int)__atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (u >= j->units)
            return;
        int i = 0;
        while (j->uoff[i + 1] <= u)
            i++;
        const int t = j->t0 + i, lu = u - j->uoff[i];
        const int g = lu / j->nchunks[i], c = lu % j->nchunks[i];
        const long first = (long)c * j->chunk[i];
        const long left = (long)j->nsel[t] - first;
        qe_attn_unit au = { .q = f->cqwork + (size_t)t * (size_t)hd * f->n_q_heads +
                                 (size_t)g * per * hd,
                            .kd = j->kqd, .vd = j->vqd, .kc = j->kc, .vc = j->vc,
                            .ks = j->kqs, .vs = j->vqs,
                            .sel = f->csel + (size_t)t * (size_t)f->sel_cap + first,
                            .y = f->gpy + (size_t)u * per * hd,
                            .m = f->gpm + (size_t)u * per,
                            .l = f->gpl + (size_t)u * per,
                            .nsel = (int)(left < j->chunk[i] ? left : j->chunk[i]),
                            .per = per, .hd = hd, .krow = (int)f->kv_row,
                            .kv_head = g, .sstride = f->n_kv_heads, .q8 = j->kqd ? 1 : 0,
                            .exp2 = f->exp2_simd,
                            .scale = f->attn_scale };
        qe_attn_walk(&au);
    }
}

static void cattn_comb_task(void *vp, int tid, int nthreads) {
    cattn_job *j = vp;
    qe_forward *f = j->f;
    const int nqh = f->n_q_heads, per = nqh / f->n_kv_heads, hd = f->head_dim;
    const long total = (long)j->nt * nqh;
    for (;;) {
        const long w = __atomic_fetch_add(&j->next2, 1, __ATOMIC_RELAXED);
        if (w >= total)
            return;
        const int i = (int)(w / nqh), h = (int)(w % nqh);
        const int t = j->t0 + i;
        const int g = h / per, hl = h % per;
        const long base = (long)j->uoff[i] * per + (long)g * j->nchunks[i] * per;
        float *y = crow(f->attn_y, (long)hd * nqh, t) + (size_t)h * hd;
        qe_attn_combine(f->gpy + (base + hl) * hd, (long)per * hd, f->gpm + base + hl,
                        f->gpl + base + hl, per, j->nchunks[i], hd, y);
        /* The gate, which the per-token form applies over the whole row afterwards: the
         * same sigmoid of the same values, one head's slice at a time. */
        const float *qg = crow(f->qg, 2 * (long)hd * nqh, t) + (size_t)h * 2 * hd + hd;
        float gt[QE_MAX_HEAD_DIM];
        qe_sigmoid(qg, gt, hd);
        for (int k = 0; k < hd; k++)
            y[k] *= gt[k];
    }
}

typedef struct {
    const qe_layer *L;
    long pos0, b0;
    float *krawc, *kc, *vc, *kvs;
    void *kpoolc;
    int8_t *kvd;
    size_t NC;
} qsa_pre_ctx;

static void row_qsa_pre(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    const qsa_pre_ctx *px = j->a;
    const qe_layer *L = px->L;
    const int hd = f->head_dim, nqh = f->n_q_heads, nkvh = f->n_kv_heads;
    const int krow = (int)f->kv_row, id = f->idx_dim, nih = f->n_idx_heads;
    const int rd = f->n_rope < hd ? f->n_rope : hd;
    const long pos = px->pos0 + t;
    float cs[QE_MAX_ROPE];
    float *ikraw = crow(f->ikraw, id, t), *iq = crow(f->iq, (long)id * nih, t);
    const float *qg = crow(f->qg, 2 * (long)hd * nqh, t);
    float *kcur = crow(f->kcur, krow, t), *vcur = crow(f->vcur, krow, t);
    float *qw = f->cqwork + (size_t)t * (size_t)hd * nqh;
    memcpy(px->krawc + (size_t)pos * (size_t)id, ikraw, (size_t)id * sizeof(float));
    qe_rms_norm_grouped(iq, f32v(f, L->ixqn), 0, iq, id, nih, f->eps);
    rope_at(f, 0, pos, rd, cs);
    qe_rope_apply(iq, nih, id, rd, cs);
    for (int h = 0; h < nqh; h++)
        memcpy(qw + (size_t)h * hd, qg + (size_t)h * 2 * hd, (size_t)hd * sizeof(float));
    qe_rms_norm_grouped(qw, f32v(f, L->aqn), 0, qw, hd, nqh, f->eps);
    qe_rope_apply(qw, nqh, hd, rd, cs);
    qe_rms_norm_grouped(kcur, f32v(f, L->akn), 0, kcur, hd, nkvh, f->eps);
    qe_rope_apply(kcur, nkvh, hd, rd, cs);
    if (px->kvd) {
        const size_t NC = px->NC;
        float *ksc = px->kvs + (size_t)pos * (size_t)nkvh;
        float *vsc = px->kvs + (NC + (size_t)pos) * (size_t)nkvh;
        int8_t *kd = px->kvd + (size_t)pos * (size_t)krow;
        int8_t *vd = px->kvd + (NC + (size_t)pos) * (size_t)krow;
        for (int g = 0; g < nkvh; g++) {
            qe_quant_q8_row(kcur + (size_t)g * hd, kd + (size_t)g * hd, ksc + g, hd);
            qe_quant_q8_row(vcur + (size_t)g * hd, vd + (size_t)g * hd, vsc + g, hd);
        }
    } else {
        memcpy(px->kc + (size_t)pos * (size_t)krow, kcur, (size_t)krow * sizeof(float));
        memcpy(px->vc + (size_t)pos * (size_t)krow, vcur, (size_t)krow * sizeof(float));
    }
}

/* One pooled index key: the mean of a block's raw keys, normed and rotated by its first
 * cell's angle, written where the per-token form wrote it - in place rather than through a
 * shared scratch row, the same operations on the same values. */
static void row_qsa_pool(rows_job *j, int i, int tid) {
    qe_forward *f = j->f;
    const qsa_pre_ctx *px = j->a;
    const int id = f->idx_dim, ratio = f->kpool_ratio, hd = f->head_dim;
    const int rd = f->n_rope < hd ? f->n_rope : hd;
    const long b = px->b0 + i;
    float dst[QE_IDX_DIM_MAX];
    float cs2[QE_MAX_ROPE];
    for (int c = 0; c < id; c++) {
        float acc = 0.0f;
        for (int r = 0; r < ratio; r++)
            acc += px->krawc[((size_t)b * (size_t)ratio + (size_t)r) * (size_t)id + (size_t)c];
        dst[c] = acc / (float)ratio;
    }
    qe_rms_norm_grouped(dst, f32v(f, px->L->ixkn), 0, dst, id, 1, f->eps);
    rope_at(f, 0, b * ratio, rd, cs2);
    qe_rope_apply(dst, 1, id, rd, cs2);
    kpool_put(f, px->kpoolc, b, dst);
}

static void qsa_core_chunk(qe_forward *f, const qe_layer *L, int slot, long pos0, int T) {
    const int nkvh = f->n_kv_heads;
    const int krow = (int)f->kv_row, id = f->idx_dim, nih = f->n_idx_heads;
    const int ratio = f->kpool_ratio;
    const long sub = qsa_sub(f, slot, 0);
    const size_t NC = f->seq_stride;
    float *kvc = f->kv ? f->kv + (size_t)sub * 2 * NC * (size_t)krow : NULL;
    float *kc = kvc, *vc = kvc ? kvc + NC * (size_t)krow : NULL;
    int8_t *kvd = f->kvd ? f->kvd + (size_t)sub * 2 * NC * (size_t)krow : NULL;
    float *kvs = f->kvs ? f->kvs + (size_t)sub * 2 * NC * (size_t)nkvh : NULL;
    float *krawc = f->kraw + (size_t)sub * NC * (size_t)id;
    void *kpoolc = kpool_seq(f, sub);

    /* The prologue: every token's cache writes, norms and rotations, then the pooled index
     * key of every block of `ratio` cells this chunk completes - the per-token form ran both
     * in position order on the master; the second only reads the first's raw keys (§7.33). */
    const double at0 = f->o.profile ? now() : 0.0;
    qsa_pre_ctx px = { .L = L, .pos0 = pos0, .krawc = krawc, .kpoolc = kpoolc, .kc = kc,
                       .vc = vc, .kvd = kvd, .kvs = kvs, .NC = NC };
    {
        rows_job rj = { .fn = row_qsa_pre, .T = T, .a = &px };
        rows_run(f, &rj, 0);
    }
    {
        /* blocks whose last cell is in this chunk */
        const long b_first = (pos0 + ratio) / ratio - 1, b_last = (pos0 + T) / ratio - 1;
        if (b_last >= b_first) {
            px.b0 = b_first;
            rows_job rj = { .fn = row_qsa_pool, .T = (int)(b_last - b_first + 1), .a = &px };
            rows_run(f, &rj, 0);
        }
    }
    const double at1 = f->o.profile ? now() : 0.0;
    /* Every token's selection, one dispatch. */
    static csel_job cj; /* 2 KiB of counts: the pass is one per process, as is this */
    cj = (csel_job){ .f = f, .kpoolc = kpoolc, .pos0 = pos0, .T = T, .id = id, .nih = nih,
                     .ratio = ratio };
    qe_pool_run(csel_task, &cj, f->n_threads);
    const double at2 = f->o.profile ? now() : 0.0;

    /* The walks and the combine, a group of tokens at a time so the partials stay small. */
    for (int t0 = 0; t0 < T; t0 += ATTN_GROUP) {
        cattn_job aj = { .f = f, .kc = kc, .vc = vc, .kqd = kvd,
                         .vqd = kvd ? kvd + NC * (size_t)krow : NULL, .kqs = kvs,
                         .vqs = kvs ? kvs + NC * (size_t)nkvh : NULL, .t0 = t0,
                         .nt = T - t0 < ATTN_GROUP ? T - t0 : ATTN_GROUP, .nsel = cj.nsel };
        int u = 0;
        for (int i = 0; i < aj.nt; i++) {
            aj.uoff[i] = u;
            attn_chunking(f, cj.nsel[t0 + i], &aj.chunk[i], &aj.nchunks[i]);
            u += aj.nchunks[i] * nkvh;
        }
        aj.uoff[aj.nt] = u;
        aj.units = u;
        qe_pool_run(cattn_walk_task, &aj, f->n_threads);
        qe_pool_run(cattn_comb_task, &aj, f->n_threads);
    }
    if (f->o.profile) {
        const double at3 = now();
        f->d.attn_pre_seconds += at1 - at0;
        f->d.attn_idx_seconds += at2 - at1;
        f->d.attn_walk_seconds += at3 - at2;
    }
}

static void qsa_out(qe_forward *f, const qe_layer *L) {
    proj(f, L->aout, f->attn_y, (long)f->head_dim * f->n_q_heads, f->cy, f->K);
}

static int ple_debug(void) {
    static int v = -1;
    if (v < 0)
        v = getenv("QE_PLE_DEBUG") ? 1 : 0;
    return v;
}

static void ple_block(qe_forward *f, const qe_layer *L, int sq, long pos) {
    const long HCW = f->hc_dim;
    const int kh = f->ple_kernel, lag = f->ple_lag, dil = f->ple_dilation;
    if (qe_ple_collect(f->ple, f->pemb) != 0) {
        long long rows[QE_PLE_MAX_HEADS];
        qe_ple_rows_at(f->ple, seq_ids(f, sq), (size_t)pos, rows);
        /* The row ids are the one PLE quantity that is exactly checkable from
         * outside the engine: `tools/ple_hash.py` re-derives them from the
         * checkpoint's own multiplier/offset/vocab-size keys, so a hash that is
         * off by one is a diff rather than a mystery. */
        if (ple_debug() && pos < 4) {
            fprintf(stderr, "ple rows pos %2ld", pos);
            for (int h = 0; h < qe_ple_heads(f->ple); h++)
                fprintf(stderr, " %lld", rows[h]);
            fputc('\n', stderr);
        }
        if (qe_ple_gather(f->ple, rows, f->pemb) != 0) {
            f->d.nonfinite++;
            return;
        }
    }
    gemv(f, L->ple_key, f->pemb, f->pkey);
    qe_rms_norm_grouped(f->pkey, f32v(f, L->ple_nk), f->K, f->pkey, f->K, f->HC, f->eps);
    gemv(f, L->ple_value, f->pemb, f->pval);
    /* The query is the residual itself, normalised - so it reads the streams
     * before this layer's mixer has anything to say about them. */
    qe_rms_norm_grouped(f->res, f32v(f, L->ple_nq), f->K, f->pgnorm, f->K, f->HC, f->eps);

    float gate[8], score[8];
    const float inv_sqrt_k = 1.0f / sqrtf((float)f->K);
    for (int s = 0; s < f->HC; s++) {
        float d = qe_dot(f->pkey + (size_t)s * f->K, f->pgnorm + (size_t)s * f->K, f->K);
        d *= inv_sqrt_k;
        score[s] = d;
        float a = fabsf(d) > 1e-6f ? fabsf(d) : 1e-6f;
        a = d < 0.0f ? -sqrtf(a) : sqrtf(a); /* signed square root, then a sigmoid */
        gate[s] = 1.0f / (1.0f + expf(-a));
    }
    for (int s = 0; s < f->HC; s++)
        for (long j = 0; j < f->K; j++)
            f->pgated[(size_t)s * f->K + j] = f->pval[j] * gate[s];

    /* Whether the norm about to be applied to the gated value normalises anything
     * depends on where its epsilon sits against the mean square: with eps above
     * it the op is linear in its input, and a scale - or a gate - in front of it
     * survives instead of cancelling. The gates here are 0.01-ish, so this is not
     * hypothetical, and it is the same mechanism that made the delta read-out's
     * 1/sqrt(S_v) worth four nats. */
    const int dbg = ple_debug();
    float grms[8];
    if (dbg) {
        for (int s = 0; s < f->HC; s++) {
            float acc = 0.0f;
            for (long j = 0; j < f->K; j++) {
                float g = f->pgated[(size_t)s * f->K + j];
                acc += g * g;
            }
            grms[s] = sqrtf(acc / (float)f->K);
        }
    }

    qe_rms_norm_grouped(f->pgated, f32v(f, L->ple_nc), f->K, f->pgnorm, f->K, f->HC, f->eps);
    /* The conv's input is the *normalised* gated value; only the residual term
     * added afterwards is the raw one. Feeding it the pre-norm value instead is
     * self-consistent, bounded, and turns the one layer that has a PLE into a
     * source of noise that every later layer inherits. */
    const float *cin = (f->o.variant & QE_VARIANT_PLE_PRENORM) ? f->pgated : f->pgnorm;
    float *hist = seq_ple_hist(f, sq);
    const float *w = f32v(f, L->ple_conv);
    for (long c = 0; c < HCW; c++) {
        const float *wc = w + (size_t)c * (size_t)kh;
        float acc = wc[kh - 1] * cin[c];
        for (int t = 0; t < kh - 1; t++)
            acc += wc[t] * hist[(size_t)((pos + (long)t * dil) % lag) * (size_t)HCW + (size_t)c];
        f->pconv[c] = acc;
    }
    /* The conv's pre-activation distribution is the one number that says whether
     * this block is a signal or an offset: silu is flat and negative below -3 and
     * linear above +3, so a block whose conv never goes positive contributes a
     * constant, and only the mean/min/max split shows it. */
    float cv_mean = 0.0f, cv_min = 0.0f, cv_max = 0.0f;
    long cv_pos = 0;
    if (dbg) {
        double sum = 0.0;
        for (long c = 0; c < HCW; c++) {
            float v = f->pconv[c];
            sum += (double)v;
            if (v > cv_max)
                cv_max = v;
            if (v < cv_min)
                cv_min = v;
            cv_pos += v > 0.0f;
        }
        cv_mean = (float)(sum / (double)HCW);
    }
    qe_silu(f->pconv, f->pconv, HCW);
    memcpy(hist + (size_t)(pos % lag) * (size_t)HCW, cin, (size_t)HCW * sizeof(float));
    /* One layer in forty-eight has a PLE, so a wrong one is invisible in a logit
     * dump except through the loss it costs every later layer. QE_PLE_DEBUG=1
     * therefore prints the block's own arithmetic per token - the scores, the
     * gates, and the size of each term against the residual it is about to join
     * - which is the only way to see whether a term is wrong or merely large. */
    if (dbg && pos < 24) {
        float mx_key = 0.0f, mx_val = 0.0f, mx_qry = 0.0f, mx_gat = 0.0f, mx_cnv = 0.0f,
              mx_res = 0.0f;
        for (long i = 0; i < HCW; i++) {
            mx_key = fabsf(f->pkey[i]) > mx_key ? fabsf(f->pkey[i]) : mx_key;
            mx_qry = fabsf(f->pgnorm[i]) > mx_qry ? fabsf(f->pgnorm[i]) : mx_qry;
            mx_gat = fabsf(f->pgated[i]) > mx_gat ? fabsf(f->pgated[i]) : mx_gat;
            mx_cnv = fabsf(f->pconv[i]) > mx_cnv ? fabsf(f->pconv[i]) : mx_cnv;
            mx_res = fabsf(f->res[i]) > mx_res ? fabsf(f->res[i]) : mx_res;
        }
        for (long i = 0; i < f->K; i++)
            mx_val = fabsf(f->pval[i]) > mx_val ? fabsf(f->pval[i]) : mx_val;
        fprintf(stderr,
                "ple pos %2ld s=[%8.3f %8.3f %8.3f %8.3f] gate=[%.4f %.4f %.4f %.4f] "
                "|key| %.3f |qry| %.3f |val| %.3f |gated| %.3f |conv| %.3f |res| %.3f\n"
                "        rms(gated)=[%.6f %.6f %.6f %.6f] vs sqrt(eps) %.6f  "
                "conv pre-silu mean %.4f min %.3f max %.3f pos %ld/%ld\n",
                pos, score[0], score[1], score[2], score[3], gate[0], gate[1], gate[2], gate[3],
                mx_key, mx_qry, mx_val, mx_gat, mx_cnv, mx_res, grms[0], grms[1], grms[2],
                grms[3], sqrtf(f->eps), cv_mean, cv_min, cv_max, cv_pos, HCW);
    }
    for (long i = 0; i < HCW; i++)
        f->res[i] += f->pgated[i] + f->pconv[i];
}

/* A decode step's expert stage in two dispatches instead of seven (ideas.md §7.30).
 *
 * The router and the shared expert's gate and up all read the block's mixer output and
 * nothing else, so they are one dispatch. The top-k and the f32 refine are the master's,
 * as they always were. Then the routed experts go through the grouped pass with one row -
 * which is where the one-token int8 walk of §7.22 lives, and which a step never reached
 * because a step took `qe_moe_layer`'s float walk - beside the shared expert's down
 * projection, which needs nothing the routed half makes. */
static void moe_block_step(qe_forward *f, int il, const qe_layer *L, float *out) {
    qe_moe *m = f->moe[il];
    gemv_job gj, uj, dj;
    qe_pool_item it[6];
    int n = 0;
    qe_moe_step_router_item(m, f->xn, &it[n++]);
    int ns = 0;
    gemv_job_for(f, L->sh_gate, f->xn, f->e640, 0, 0, 1, &gj);
    step_attach(f, &gj, &ns);
    it[n++] = (qe_pool_item){ gemv_task, &gj, f->walk_threads, 0 };
    gemv_job_for(f, L->sh_up, f->xn, f->u640, 0, 0, 1, &uj);
    step_attach(f, &uj, &ns);
    it[n++] = (qe_pool_item){ gemv_task, &uj, f->walk_threads, 0 };
    qe_pool_run_items(it, n);
    const int rc = qe_moe_step_begin(m, f->xn, f->routes, out);
    qe_silu_mul(f->e640, f->e640, f->u640, f->n_ff_shexp);
    gemv_job_for(f, L->sh_down, f->e640, f->sh_out, 0, 0, 1, &dj);
    ns = 0;
    step_attach(f, &dj, &ns);
    n = 0;
    for (int i = 0; i < rc; i++)
        qe_moe_step_item(m, i, &it[n++]);
    it[n++] = (qe_pool_item){ gemv_task, &dj, f->walk_threads, 0 };
    qe_pool_run_items(it, n);
    if (f->o.profile)
        f->walk_mac += (double)L->sh_gate->ne[0] * (double)L->sh_gate->ne[1] +
                       (double)L->sh_up->ne[0] * (double)L->sh_up->ne[1] +
                       (double)L->sh_down->ne[0] * (double)L->sh_down->ne[1];
    float g = qe_dot(f32v(f, L->sh_inp), f->xn, f->K);
    qe_sigmoid(&g, &g, 1);
    qe_axpy(out, f->sh_out, g, f->K);
}

static void moe_block(qe_forward *f, int il, const qe_layer *L, float *out) {
    if (f->chunk_T == 1 && f->step_fuse) {
        moe_block_step(f, il, L, out);
        return;
    }
    qe_moe_layer(f->moe[il], f->xn, out, f->routes, NULL);
    gemv(f, L->sh_gate, f->xn, f->e640);
    gemv(f, L->sh_up, f->xn, f->u640);
    qe_silu_mul(f->e640, f->e640, f->u640, f->n_ff_shexp);
    gemv(f, L->sh_down, f->e640, f->sh_out);
    /* One scalar from a 1-D tensor: the shared expert is switched on or off, not
     * mixed channel by channel. */
    float g = qe_dot(f32v(f, L->sh_inp), f->xn, f->K);
    qe_sigmoid(&g, &g, 1);
    qe_axpy(out, f->sh_out, g, f->K);
}

/* The expert stage's per-row loops (§7.33): a row's top-k, the shared expert's silu, and
 * the shared expert's sigmoid-gated add into the routed output. */
static void row_moe_select(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    qe_moe *m = (qe_moe *)j->a;
    const int topk = f->n_experts_active;
    qe_route *r = f->croutes + (size_t)t * (size_t)topk;
    int n = qe_moe_select(m, f->cmoe_logit + (size_t)t * (size_t)f->n_experts, r);
    for (int i = n; i < topk; i++)
        r[i].expert = -1;
}

static void row_shared_silu(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    const long ns = f->n_ff_shexp;
    float *g = f->e640 + (size_t)t * (size_t)ns;
    qe_silu_mul(g, g, f->u640 + (size_t)t * (size_t)ns, ns);
}

static void row_shared_add(rows_job *j, int t, int tid) {
    qe_forward *f = j->f;
    const qe_layer *L = j->a;
    const float *xn = f->cxn + (size_t)t * (size_t)f->K;
    float g = qe_dot(f32v(f, L->sh_inp), xn, f->K);
    qe_sigmoid(&g, &g, 1);
    qe_axpy(f->cy + (size_t)t * (size_t)f->K, f->sh_out + (size_t)t * (size_t)f->K, g, f->K);
}

/* The expert stage of a chunk as one dispatch of a stage graph: the grouped routed
 * walk, the chunk recode of the activation both halves read, and the shared expert's
 * two projections, their silu and its projection - ten items, where the same stage as
 * queued dispatches costs seven, because the two fills and the silu each queued their
 * own hand-back between them.
 *
 * What the graph buys is nothing exotic. The two halves read the same block output
 * and write disjoint buffers, so the six hand-backs between them were an ordering
 * no data asked for; and they saturate differently, the walk on the quantiser's
 * ports and the projections on DRAM, so a thread that has run out of bucket is
 * worth more on a projection than it is at the back of a queue for one. The tail of
 * whichever half finishes first is the other's to eat, which is the only thing a
 * shared schedule can do that two fixed groups of threads cannot.
 *
 * Two things keep it honest. Every item is a task that already ran this way, the
 * three of the grouped pass and the three of proj(), with the same nthreads, so the
 * same slicing and the same bytes out; and anything that cannot be built means no
 * graph at all, never a smaller one, because a stage whose item was dropped is a
 * stage whose output was never computed. Returns 0 in exactly that case, and the
 * caller runs the schedule that was here first.
 *
 * The rows of one window only: `moe_tile` splits a chunk into several grouped
 * passes, and one moe object carries one pass's buckets, so a tiled chunk keeps the
 * dispatch-per-window shape until the tile sweep says a window is narrower than the
 * chunk. Same for a shared expert below the projection walk's threshold, which is a
 * loop of T dispatches and cannot be an item of a dispatch. */
static int moe_stage_graph(qe_forward *f, int il, const qe_layer *L, int T) {
    qe_moe *m = f->moe[il];
    const int topk = f->n_experts_active;
    const long ns = f->n_ff_shexp;
    const int tile = f->moe_tile;
    if (T <= 1 || (tile > 0 && tile < T))
        return 0;
    if (!proj_walkable(f, L->sh_gate) || !proj_walkable(f, L->sh_up) ||
        !proj_walkable(f, L->sh_down))
        return 0;
    const int rc = qe_moe_multi_begin(m, f->cxn, f->K, f->croutes, topk, T, f->cy, f->K);
    if (rc == 0)
        return 0;
    gemv_chunk_job gjob, ujob, djob;
    panel_job pf1, pf2;
    silu_job sjob = { f->e640, f->u640, ns, T };
    qe_pool_item it[10];
    int n = 0;
    /* rc == 2 is a window whose routes were all padding: the pass answered it with
     * zeroes and has no items to run, and the shared half still has all of its. */
    if (rc == 1)
        for (int i = 0; i < 3; i++)
            qe_moe_multi_item(m, i, &it[n++]);
    /* The chunk recode, as items of this graph and not as the dispatches proj() makes
     * between them. A graph that carried no panel carried no integer walk either, so it
     * ran the shared expert's projections on the dequantising walk while the schedule it
     * was being measured against recoded them - which is both a different model and the
     * reason its number was not worth more (`/tmp/f24_wide.log`: the trunk differed, and
     * the schedule was 1.015x of the graph at best).
     *
     * One fill serves gate and up, because both read the block output, which is a fill the
     * queued path makes twice for the same bytes. Down reads the silu, so its fill is a
     * second one, and the two share f->i8x only because of the ordering below: the down
     * fill depends on the silu, which depends on the two walks that read the first fill,
     * so nothing is rewriting the panel while a walk is in it. */
    const int w_gate = proj_i8_wanted(f, L->sh_gate);
    const int w_up = proj_i8_wanted(f, L->sh_up);
    const int w_down = proj_i8_wanted(f, L->sh_down);
    int i_pf1 = -1, i_pf2 = -1;
    qe_i8panel pgu = { 0 }, pdn = { 0 };
    if (w_gate || w_up) {
        const long K1 = (long)L->sh_gate->ne[0];
        /* One panel for two tensors means the two reduce over the same width. A
         * checkpoint whose shared gate and up differ in K gets the schedule that was
         * here first rather than a walk reading a panel of the other one's shape. */
        if (K1 != (long)L->sh_up->ne[0])
            return 0;
        pgu = (qe_i8panel){ .x8 = f->i8x,
                            .s  = f->i8s,
                            .p  = f->i8pseed,
                            .stride8 = K1,
                            .nb = K1 / 32 };
        pf1 = (panel_job){ f->cxn, f->i8x, f->i8s, f->i8pseed, f->K, K1, f->chunk_T };
        i_pf1 = n;
        it[n++] = (qe_pool_item){ panel_fill_task, &pf1, f->n_threads, 0 };
    }
    const int i_gate = n;
    proj_item(f, L->sh_gate, f->cxn, f->K, f->e640, ns, w_gate ? &pgu : NULL, &gjob, &it[n++]);
    if (i_pf1 >= 0)
        it[i_gate].deps = 1u << (unsigned)i_pf1;
    const int i_up = n;
    proj_item(f, L->sh_up, f->cxn, f->K, f->u640, ns, w_up ? &pgu : NULL, &ujob, &it[n++]);
    if (i_pf1 >= 0)
        it[i_up].deps = 1u << (unsigned)i_pf1;
    const int i_silu = n;
    it[n++] = (qe_pool_item){ silu_rows_task, &sjob, f->n_threads,
                              (1u << (unsigned)i_gate) | (1u << (unsigned)i_up) };
    if (w_down) {
        const long K2 = (long)L->sh_down->ne[0];
        pdn = (qe_i8panel){ .x8 = f->i8x,
                            .s  = f->i8s,
                            .p  = f->i8pseed,
                            .stride8 = K2,
                            .nb = K2 / 32 };
        pf2 = (panel_job){ f->e640, f->i8x, f->i8s, f->i8pseed, ns, K2, f->chunk_T };
        i_pf2 = n;
        it[n++] = (qe_pool_item){ panel_fill_task, &pf2, f->n_threads,
                                  1u << (unsigned)i_silu };
    }
    const int i_down = n;
    proj_item(f, L->sh_down, f->e640, ns, f->sh_out, f->K, w_down ? &pdn : NULL, &djob,
              &it[n++]);
    it[i_down].deps = 1u << (unsigned)(i_pf2 >= 0 ? i_pf2 : i_silu);
    qe_pool_run_items(it, n);
    /* The mixer's scalar for each row, reading the residual as the plain stage does
     * and adding the shared expert's output into the routed one. */
    {
        rows_job rj = { .fn = row_shared_add, .T = T, .a = L };
        rows_run(f, &rj, 0);
    }
    return 1;
}

/* The same block for T tokens at once.
 *
 * The routed half is the reason this exists: grouped by expert, one read of an
 * expert serves every row of the chunk that chose it, so the window reads its
 * distinct experts instead of T*top_k of them. The shared expert and the mixer
 * are ordinary projections and go through proj() like every other dense stage
 * of a chunk, which is what makes a width-1 chunk and a width-1 chunk with the
 * grouping off the same arithmetic: below the walk threshold proj() is a loop of
 * gemv(), so with both switches off this is moe_block() T times over.
 *
 * Routing is qe_moe_router + qe_moe_select + qe_moe_refine_rows rather than
 * qe_moe_route only because the router runs once for the chunk instead of once a
 * row; the mixing weights are the same f32 ones a token fed alone gets, which is
 * the invariant that matters (a chunk that picked different weights than the same
 * token fed alone would not be a scheduling change).
 *
 * Fewer than top_k experts can win for a token, so the row is padded with -1,
 * which the grouping skips rather than routing to expert -1. */
static void moe_block_chunk(qe_forward *f, int il, const qe_layer *L, int T) {
    const int topk = f->n_experts_active;
    /* One pass over the router for the whole chunk. The router is 512 x K and is
     * read in full for every token, so reading it once for T of them is the
     * cheapest saving in the block; the selection stays per row because a top-k
     * is one token's business. */
    qe_moe_router_multi(f->moe[il], f->cxn, f->K, f->cmoe_logit, f->n_experts, T);
    {
        rows_job rj = { .fn = row_moe_select, .T = T, .a = f->moe[il] };
        rows_run(f, &rj, 0);
    }
    /* And the f32 weights behind those picks, one dispatch for the chunk's rows:
     * a row's ten dots are a few microseconds and 512 rows of them are a
     * millisecond and a half of this stage, which is more than the handback costs. */
    qe_moe_refine_rows(f->moe[il], f->cxn, f->K, f->croutes, topk, T);
    /* Both halves of the stage in one dispatch, when that shape applies: it is
     * tried first and declines in favour of the schedule below, which is what ran
     * here before it existed and still runs for a tiled chunk. */
    if (f->moe_co && moe_stage_graph(f, il, L, T))
        return;
    /* Rows of a chunk share an expert whether they are handed to it together or not, so
     * the width of a pass is a question about how full its buckets are and not about
     * what they read: an expert at the shipping width averages ten routed rows, and one
     * walk of its quantiser costs the same whether the eight rows it carries are the
     * bucket or the two rows left over from it. Hence the windows below, which trade the
     * three pool_runs a chunk's expert stage used to cost for as many lots of three as
     * the chunk divides into. 0 is the whole chunk, and see qe_forward_set_moe_tile for
     * what that curve measures. */
    const int tile = f->moe_tile;
    if (tile <= 0 || tile >= T) {
        qe_moe_forward_multi(f->moe[il], f->cxn, f->K, f->croutes, topk, T, f->cy, f->K);
    } else {
        for (int w0 = 0; w0 < T; w0 += tile) {
            const int w = (T - w0 < tile) ? T - w0 : tile;
            qe_moe_forward_multi(f->moe[il], f->cxn + (size_t)w0 * (size_t)f->K, f->K,
                                 f->croutes + (size_t)w0 * (size_t)topk, topk, w,
                                 f->cy + (size_t)w0 * (size_t)f->K, f->K);
        }
    }

    const long ns = f->n_ff_shexp;
    proj(f, L->sh_gate, f->cxn, f->K, f->e640, ns);
    proj(f, L->sh_up, f->cxn, f->K, f->u640, ns);
    {
        rows_job rj = { .fn = row_shared_silu, .T = T };
        rows_run(f, &rj, 0);
    }
    proj(f, L->sh_down, f->e640, ns, f->sh_out, f->K);
    {
        rows_job rj = { .fn = row_shared_add, .T = T, .a = L };
        rows_run(f, &rj, 0);
    }
}

/* Stage timing is opt-in: 240 clock reads a step is nothing next to a decode,
 * but it is not nothing, and the default build should measure the model rather
 * than the measurement. */
#define QE_TICK() (f->o.profile ? (qe_pool_stats_get(&f->pst0), now()) : 0.0)
#define QE_TALLY(stage, ts)                                                                      \
    do {                                                                                         \
        if (f->o.profile) {                                                                      \
            qe_pool_stats ps1_;                                                                  \
            qe_pool_stats_get(&ps1_);                                                            \
            f->d.stage_disp[stage] += ps1_.dispatches - f->pst0.dispatches;                      \
            f->d.stage_pool[stage] += ps1_.seconds - f->pst0.seconds;                            \
            f->d.stage_tail[stage] += ps1_.tail - f->pst0.tail;                                  \
            f->d.stage_seconds[stage] += now() - (ts);                                           \
            f->d.stage_macs[stage] += f->walk_mac - f->mac_mark;                                  \
            f->mac_mark = f->walk_mac;                                                            \
        }                                                                                         \
    } while (0)

/* The head mixer and the output projection run on the residual as it stands.
 * Safe mid-stack: the mixer only reads res, and xn, hn, lo and gate are each
 * rewritten by the next mixer before anything reads them again. */
static void head_probe(qe_forward *f, int row, float *out) {
    hc_mixer(f, NULL, 0, row, 1, 0);
    gemv(f, f->outw, crow(f->cxn, f->K, row), out);
}

/* -------------------------------------------------------------- draft rows
 *
 * The nextn block, the fold in front of it, and the mixer and output matrix behind
 * it, for `T` rows at once.
 *
 * The block is not a second implementation of a decoder layer. It is `hc_mixer`,
 * `qsa_proj`, `qsa_core`, `qsa_out`, `hc_combine` and the expert stage asked for
 * layer `f->n_layer`, which is the layer `mtp_layer_open` filled in from the head's
 * own file - the same six calls the trunk's loop makes per layer, and the reason the
 * head's attention, indexer and routed experts are known to behave like the trunk's
 * is that they are the same code.
 *
 * Two properties hold for every byte this walks:
 *
 *   - The caller's rows are read before anything is written. `h_in` is the trunk's
 *     own residual for every warm row, and the fold is the only thing that touches
 *     it; the block then runs on `mhin`, so the chunk the draft is being made from is
 *     left exactly as the trunk left it.
 *   - Nothing here moves the trunk. Its position, its checkpoint, its recurrent state
 *     and its caches are neither read nor written. The head's keys and values live in
 *     one more slot of the same caches, addressed by the position of the token fed to
 *     it, and are read only up to that position - so an abandoned draft row leaves
 *     nothing behind except bytes that the row taking its place overwrites before
 *     anything reads them, which is the same argument a rewound window rests on.
 */

/* The fold, for the whole chunk: both halves of every row normalised and written
 * beside each other, and `eh_proj` walked over them once per stream.
 *
 * The norm is grouped by stream because `hnorm` is hc_dim wide - one gamma per stream,
 * the same shape every hyper-connection norm in this model has - and the concatenation
 * is what `eh_proj` was trained against (ref/qwen4exp.cpp:636 concatenates the two
 * halves along the feature axis and multiplies one matrix, broadcast over the stream
 * axis, which is what the HC views of it walk). */
static void mtp_fold(qe_forward *f, const int *toks, const float *h_in, int T) {
    const int K = f->K, HC = f->HC;
    const float *en = f32v(f, f->mtp_enorm), *hn = f32v(f, f->mtp_hnorm);

    for (int t = 0; t < T; t++) {
        float *e = f->mout + (size_t)t * (size_t)K;
        embed_row(f, toks[t], e);
        qe_rms_norm_grouped(e, en, 0, e, K, 1, f->eps);
        const float *h = h_in + (size_t)t * (size_t)f->hc_dim;
        float *cat = f->mcat + (size_t)t * (size_t)HC * (size_t)(2 * K);
        for (int s = 0; s < HC; s++) {
            float *dst = cat + (size_t)s * (size_t)(2 * K);
            memcpy(dst, e, (size_t)K * sizeof(float));
            qe_rms_norm_grouped(h + (size_t)s * (size_t)K, hn + (size_t)s * (size_t)K, 0, dst + K,
                                K, 1, f->eps);
        }
    }
    /* One walk per stream over every row in flight: `mcat` puts the chunk's rows
     * HC*2K apart, which is the stride a chunk projection reads, and every stream reads
     * the same matrix - the fold's projection is broadcast over the stream axis, not
     * sliced along it - so the streams differ only in the offsets they start at. */
    for (int s = 0; s < HC; s++)
        proj(f, f->mtp_eh, f->mcat + (size_t)s * (size_t)(2 * K), (long)HC * (2 * K),
             f->mhin + (size_t)s * (size_t)K, f->hc_dim);
}

static int mtp_rows(qe_forward *f, const int *toks, const float *h_in, int T, long pos0,
                    float *h_out, float *logits) {
    if (!f || !f->mtp_ready || !toks || !h_in || T < 1 || T > QE_CHUNK_MAX)
        return -1;
    const long vocab = (long)f->embd->ne[1];
    for (int t = 0; t < T; t++)
        if (toks[t] < 0 || (uint64_t)toks[t] >= (uint64_t)vocab)
            return -1;
    if (pos0 < 0 || pos0 + T > (long)f->n_ctx) {
        f->d.over_ctx++;
        return -1;
    }

    qe_layer *L = &f->L[f->n_layer];
    float *saved_cres = f->cres;
    const int saved_T = f->chunk_T, saved_t = f->chunk_t;
    const long nf0 = f->d.nonfinite;

    /* The one substitution that makes the head's block the trunk's: a layer writes
     * the residual it was handed, so the head is handed its own arena. */
    f->cres = f->mhin;
    f->chunk_T = T;
    f->chunk_t = 0;

    mtp_fold(f, toks, h_in, T);

    hc_mixer(f, L, 0, 0, T, 1);
    qsa_proj(f, L);
    for (int t = 0; t < T; t++) {
        use_row(f, t);
        f->chunk_t = t;
        qsa_core(f, L, L->qsa_slot, 0, pos0 + t);
    }
    qsa_out(f, L);
    for (int t = 0; t < T; t++) {
        use_row(f, t);
        sweep(f, f->y, f->K, &f->d.max_abs_block);
        hc_combine(f, f->y, f->inj);
    }
    hc_mixer(f, L, 1, 0, T, 1);
    if (f->moe_grouped && T > 1)
        moe_block_chunk(f, f->n_layer, L, T);
    else
        for (int t = 0; t < T; t++) {
            use_row(f, t);
            moe_block(f, f->n_layer, L, f->y);
        }
    for (int t = 0; t < T; t++) {
        use_row(f, t);
        sweep(f, f->y, f->K, &f->d.max_abs_block);
        hc_combine(f, f->y, f->inj);
        sweep(f, f->res, f->hc_dim, &f->d.max_abs_res);
    }

    /* The residual the next draft row reads is the block's output *before* the head
     * mixer, which is where the target exports its own (ref/qwen4exp.cpp:553, :655). */
    if (h_out)
        for (int t = 0; t < T; t++)
            memcpy(h_out + (size_t)t * (size_t)f->hc_dim,
                   f->mhin + (size_t)t * (size_t)f->hc_dim, (size_t)f->hc_dim * sizeof(float));

    if (logits) {
        /* The head's mixer is its output norm, with no injection: like the trunk's,
         * there is nothing left to add the result to. */
        hc_mixer_t(f, f->mtp_hc_norm, f->mtp_hc_down, f->mtp_hc_up, NULL, 0, T, 0);
        gemv_rows(f, f->mtp_outw, f->cxn, logits, T, vocab);
        for (int t = 0; t < T; t++)
            sweep(f, logits + (size_t)t * (size_t)vocab, vocab, &f->d.max_abs_logits);
    }

    f->cres = saved_cres;
    f->chunk_T = saved_T;
    f->chunk_t = saved_t;
    use_row(f, saved_t);
    return f->d.nonfinite > nf0 ? -2 : 0;
}

/* Feed the head the chunk the trunk just walked. With the pairing the head trains on,
 * row `t` of the head reads the residual of `pos0+t-1`, so the first row of a chunk
 * reads the last row of the one before it - which is the only thing `mpend` is for -
 * and the rest read rows of the chunk in flight.
 *
 * Nothing is exported and no logits are asked for: what warming buys is a head whose
 * attention has seen the sequence it is being asked to continue, and the head's
 * output matrix is the one part of a draft row a prompt has no use for.
 *
 * Called before the trunk's own output mixer rather than after it, which is what lets
 * the head reuse the trunk's chunk scratch: the mixer rewrites the row the trunk's
 * logits are read from, so the head's last write to that buffer is never what a caller
 * of `qe_forward_row_logits` or `qe_forward_hidden` ends up looking at. */
static int mtp_warm_chunk(qe_forward *f, long pos0, int T) {
    int toks[QE_CHUNK_MAX];
    for (int t = 0; t < T; t++)
        toks[t] = (int)f->seq[pos0 + t];
    if (!f->mtp_back)
        return mtp_rows(f, toks, f->cres, T, pos0, NULL, NULL);
    int rc = 0;
    if (f->mpend_pos == pos0 - 1)
        rc = mtp_rows(f, toks, f->mpend, 1, pos0, NULL, NULL);
    if (rc == 0 && T > 1)
        rc = mtp_rows(f, toks + 1, f->cres, T - 1, pos0 + 1, NULL, NULL);
    return rc;
}

/* --------------------------------------------------- the head, from outside
 *
 * Whether there is a head, how wide a draft row's residual is, what the target's
 * residual was at a position, and one row of the head. Which residual pairs with
 * which token is deliberately the caller's question and not this file's: it is the
 * one decision about these checkpoints that a comment cannot settle, and
 * `make forward` settles it by accepted-draft rate with `qe_forward_set_mtp_back`. */

int qe_forward_has_mtp(const qe_forward *f) { return f && f->mtp_ready; }

int qe_forward_hc_dim(const qe_forward *f) { return f ? f->hc_dim : 0; }

long qe_forward_row_hidden_hc(const qe_forward *f, int row, float *dst) {
    if (!f || !dst)
        return -1;
    const int t = row < 0 ? f->chunk_T - 1 : row;
    if (t < 0 || t >= f->chunk_T)
        return -1;
    memcpy(dst, f->cres + (size_t)t * (size_t)f->hc_dim, (size_t)f->hc_dim * sizeof *dst);
    return f->hc_dim;
}

/* The target's four-stream residual at the position of a token this pass consumed.
 *
 * Two places hold one, and between them they hold every position a drafter can be
 * standing at: the rows of the pass in flight, and the one row of the pass before it
 * that is kept because the token after it had not been seen yet. A position outside
 * both is answered with -1 rather than with the nearest row, because a draft seeded
 * from another position's residual is a confidently wrong draft and no amount of
 * accept-rate measuring tells you so. */
long qe_forward_mtp_hidden(const qe_forward *f, long p, float *dst) {
    if (!f || !dst || !f->mtp_ready)
        return -1;
    float *src = NULL;
    if (p >= f->pass_pos0 && p < f->pass_pos0 + f->chunk_T)
        src = f->cres + (size_t)(p - f->pass_pos0) * (size_t)f->hc_dim;
    else if (f->mpend_pos == p)
        src = f->mpend;
    if (!src)
        return -1;
    memcpy(dst, src, (size_t)f->hc_dim * sizeof *dst);
    return f->hc_dim;
}

int qe_forward_mtp_rows(qe_forward *f, const int *toks, const float *h_in, int T, long pos0,
                        float *h_out, float *logits) {
    return mtp_rows(f, toks, h_in, T, pos0, h_out, logits);
}

int qe_forward_mtp_row(qe_forward *f, int token, long pos, const float *h_in, float *h_out,
                       float *logits) {
    return mtp_rows(f, &token, h_in, 1, pos, h_out, logits);
}

void qe_forward_set_mtp_warm(qe_forward *f, int on) {
    if (f)
        f->mtp_warm = on ? 1 : 0;
}

int qe_forward_get_mtp_warm(const qe_forward *f) { return f ? f->mtp_warm : 0; }

void qe_forward_set_mtp_back(qe_forward *f, int back) {
    if (f)
        f->mtp_back = back ? 1 : 0;
}

int qe_forward_get_mtp_back(const qe_forward *f) { return f ? f->mtp_back : 0; }


/* ------------------------------------------------------- a rewindable window
 *
 * One copy of the recurrent state per row of the window, laid out exactly like the
 * three regions of `qe_state` so that putting boundary `b` back is one copy of each
 * region rather than one per layer. Every GDN layer owns one slot and only writes
 * that slot, so the slices the rows of one window write are disjoint, and a
 * boundary is complete once every layer has checkpointed its own - which is also
 * how the conv history of a layer is captured: same slot, same moment.
 *
 * What is not copied, and why that is enough: the attention cache and the
 * indexer's raw and pooled keys are addressed by position and read only up to it,
 * so an abandoned row's bytes at a position past the frontier are overwritten by
 * the row that takes its place before anything looks at them. The pooled block is
 * the careful case - it is written only when its ratio-th cell completes, and a
 * frontier that has not reached that cell never scores it.
 *
 * The two convolution histories are not like those. Both are *minimal* rings: the
 * gated-delta-net conv keeps `kernel-1` rows and the PLE conv keeps
 * `(kernel-1)*dilation`, so a tap of position `pos` is not "the row addressed by
 * `pos`" but "whatever those slots hold", which is positions `pos-3 .. pos-1` only
 * while positions are fed in order. Feed a row and abandon it and the slots keep
 * the abandoned row's `qkv` at an address a retained position reads as its own
 * history. So a boundary is the delta state *and* both rings: the ring of a layer
 * is copied while that layer's core is walking, next to the state it walks, and
 * the PLE history is copied in the one layer that owns it, row by row. */
struct qe_ckpt {
    int rows;
    int kept;             /* boundaries the window that just ran actually wrote */
    size_t slot_b;        /* one layer's state, in bytes */
    size_t gdn_b;         /* all of them */
    size_t slot_ring_b;   /* one layer's conv history, in bytes */
    size_t conv_b;        /* all of them */
    size_t ple_b;         /* the PLE history, which one layer in the network writes */
    size_t stride;        /* one boundary: the three regions, back to back */
    float *buf;    /* rows - 1 boundaries, oldest first */
    long pos;      /* the position the window started at */
};

qe_ckpt *qe_ckpt_new(const qe_forward *f, int rows) {
    if (!f || rows < 2 || rows > QE_CHUNK_MAX)
        return NULL;
    qe_ckpt *c = calloc(1, sizeof *c);
    if (!c)
        return NULL;
    c->rows = rows;
    c->slot_b = (size_t)f->n_gdn_heads * (size_t)f->head_state * (size_t)f->head_state *
                sizeof(float);
    c->gdn_b = c->slot_b * (size_t)f->gdn_slots;
    c->slot_ring_b = (size_t)(f->gdn_kernel - 1) * (size_t)f->gdn_channels * sizeof(float);
    c->conv_b = c->slot_ring_b * (size_t)f->gdn_slots;
    c->ple_b = (size_t)f->ple_lag * (size_t)f->hc_dim * sizeof(float);
    c->stride = c->gdn_b + c->conv_b + c->ple_b;
    /* The same allocator the state itself is registered with, because this buffer
     * is memcpy'd against that one several times a second: a 108 MiB copy across
     * 4K pages spends more of a window on page walks than on bytes. Touched here
     * for the reason `scratch` gives - the alternative is taking the fault in the
     * middle of a decode step. */
    const size_t want = c->stride * (size_t)(rows - 1);
    c->buf = (float *)qe_alloc_huge(want);
    if (c->buf)
        memset(c->buf, 0, qe_align_up(want, 2u << 20));
    if (!c->buf) {
        free(c);
        return NULL;
    }
    return c;
}

void qe_ckpt_free(qe_ckpt *c) {
    if (!c)
        return;
    qe_free_huge(c->buf, c->stride * (size_t)(c->rows - 1));
    free(c);
}

/* Boundary `b` (1..rows-1) of the row currently being walked, for the layer whose
 * state is in `slot`. Its conv history travels with its state because the ring is
 * minimal: the next row's taps read the slots this row just wrote, so the pair is
 * the same moment and one of them without the other is a position with someone
 * else's past. */
static void state_copy_task(void *vp, int tid, int nthreads);

static void ckpt_slot(qe_forward *f, const qe_ckpt *c, int slot, int b) {
    if (b < 1 || b >= c->rows)
        return;
    char *dst = (char *)c->buf + (size_t)(b - 1) * c->stride;
    /* 3 MiB of one layer's state per boundary, 36 layers a window: on the master alone
     * that was 11 ms of a 126 ms two-row window, the serial half of its block stage
     * (ideas.md §7.30). The pool copies it in slices; the bytes are the bytes. */
    state_copy_job j = { dst + (size_t)slot * c->slot_b,
                         (const char *)f->gdn + (size_t)slot * c->slot_b, c->slot_b };
    if (f->n_threads > 1 && qe_pool_size() > 1)
        qe_pool_run(state_copy_task, &j, f->n_threads);
    else
        memcpy(j.dst, j.src, j.n);
    if (c->conv_b)
        memcpy(dst + c->gdn_b + (size_t)slot * c->slot_ring_b,
               (const char *)f->gconv + (size_t)slot * c->slot_ring_b, c->slot_ring_b);
}

/* The PLE history as of the row just walked. One layer in the network owns it, and
 * it walks every row of a chunk before that chunk's block cores run, so this is
 * the one capture that cannot ride along with `ckpt_slot` - by the time a layer is
 * checkpointing its state the history has already moved on to the window's last
 * row. Captured even when the block was ablated, which leaves the history as it
 * was and is a copy of the right answer rather than of nothing. */
static void ckpt_ple(qe_forward *f, const qe_ckpt *c, int b) {
    if (b < 1 || b >= c->rows || !c->ple_b)
        return;
    memcpy((char *)c->buf + (size_t)(b - 1) * c->stride + c->gdn_b + c->conv_b, f->ple_hist,
           c->ple_b);
}

/* What one token through the network owes beyond the state it always updates.
 * Feeding a prompt needs the head only for its last token - the lm_head is 497
 * MiB of the 5.44 GiB a step reads, 11% of a step for a result a prompt has no
 * use for until the end - and the probe heads are a debugging feature, so
 * prefill asks for neither and gets bit-identical numbers for what it keeps. */
enum {
    STEP_LOGITS = 1u << 0, /* run the lm_head into the caller's buffer  */
    STEP_PROBE  = 1u << 1, /* run the probe heads of the probed layers */
    /* Keep the delta-net state after every row, which is the one part of a
     * sequence's state that a row past the end of a draft cannot be undone
     * without. It costs a copy per row and one dispatch of the block core per
     * row instead of one per layer, so it is asked for by a window of a few rows
     * and never by a prompt. */
    STEP_CKPT = 1u << 2,
    /* `logits_out` is T rows and not one: what every row of a batched step is owed,
     * since each of its rows is a sequence of its own and all of them need a next
     * token. A prompt owes one row however wide its chunk is, so it never asks. */
    STEP_ROWS = 1u << 3,
};


/* How many prompt tokens share one walk of the weights. QE_CHUNK_DEFAULT is measured
 * rather than assumed (`make forward` sweeps 1..QE_CHUNK_MAX on one prompt and prints
 * the row that wins), and QE_CHUNK overrides it for a run. QE_CHUNK=1 is the old
 * one-token-per-weight-pass walk. */
static int prefill_chunk(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_CHUNK");
        long v = e ? strtol(e, NULL, 10) : QE_CHUNK_DEFAULT;
        if (v < 1)
            v = 1;
        if (v > QE_CHUNK_MAX)
            v = QE_CHUNK_MAX;
        cached = (int)v;
    }
    return cached;
}

/* Whether the chunked prefill groups a layer's experts by expert id. Off until a
 * measurement says otherwise (`make forward` A/Bs the two on one load), because
 * grouping is two reassociations of the trunk and not a scheduling change; see
 * qe_forward_set_moe_grouped. */
static int moe_group_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("QE_MOE_GROUP");
        cached = e ? atoi(e) != 0 : QE_MOE_GROUP_DEFAULT;
    }
    return cached;
}

/* How many rows of a chunk one grouped expert pass covers. QE_MOE_TILE_DEFAULT is
 * measured rather than assumed (`make forward` sweeps the tile on one load); 0 is the
 * whole chunk, and so is anything wider than the chunk itself. */
static int moe_tile_env(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("QE_MOE_TILE");
        long v = e ? strtol(e, NULL, 10) : QE_MOE_TILE_DEFAULT;
        cached = (v <= 0 || v > QE_CHUNK_MAX) ? 0 : (int)v;
    }
    return cached;
}

static int moe_co_env(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("QE_MOE_CO");
        cached = e ? atoi(e) != 0 : QE_MOE_CO_DEFAULT;
    }
    return cached;
}

static int head_batch_env(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("QE_HEAD_BATCH");
        cached = e ? atoi(e) != 0 : 1;
    }
    return cached;
}

/* QE_MTP_WARM: does a prompt also feed the draft head?
 *
 * On by default, and it is the cheap half of the head's budget: a prefill runs a
 * chunk of tokens through the trunk anyway, so the head's rows go through the same
 * chunked block at ~0.055 ms/token against the 17 ms a chunked trunk token costs,
 * and what they buy is a head whose attention has seen the sequence it is being
 * asked to continue. Off, the first draft of a long prompt attends over an empty
 * cache, which is not the state the block was trained in. */
static int mtp_warm_env(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("QE_MTP_WARM");
        cached = e ? atoi(e) != 0 : 1;
    }
    return cached;
}

/* QE_MTP_BACK: 1 (the default) feeds a draft row the residual of the position *before*
 * its token, which is the pairing the head files' nextn block trains on
 * (ref/qwen4exp.cpp:636 concatenates the two normalised halves of one row, and the
 * reference drafter chains by handing each row the residual the row before it
 * exported). 0 pairs a row's token with its own residual, which is what a fold that
 * reads the same position would need; it is measurable because the alternative to
 * measuring it is trusting a comment, and `make forward` prints the accept rate both
 * ways in the load it already has open. */
static int mtp_back_env(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("QE_MTP_BACK");
        cached = e ? atoi(e) : 1;
        if (cached != 0 && cached != 1)
            cached = 1;
    }
    return cached;
}


/* T tokens through the network, one stage of a layer at a time.
 *
 * With T == 1 this is the ordinary decode step, on row 0 of the chunk buffers.
 * With T > 1 the loops invert - each of a layer's three stages runs over the
 * whole chunk before the next one starts - and nothing else changes. Tokens are
 * independent of each other inside a layer, and what is ordered by position is
 * the recurrences *within* a stage: the delta-rule state, the conv ring, the KV
 * cache, the indexer's pool, the PLE window. Every one of those loops still
 * walks t ascending, so a token sees exactly the state its predecessor in the
 * chunk left, and the numbers are bit-identical to feeding the same tokens one
 * at a time. `make forward` asserts that.
 *
 * The reason to do it is bytes. A stage's weights stop being read once per token
 * and start being read once per chunk, and for the hyper-connections - 656 MiB of
 * a 5568 MiB step, in tensors of 3.4 MiB, so small enough to live in the 30 MiB
 * L3 for the length of a stage - that is the difference between walking DRAM at
 * 67 GB/s and not walking it at all. The dense projections are 58 MiB a layer and
 * too big for that, so they still leave the cache; qe_gemv_multi_t, which blocks
 * a weight walk against T activations inside one core's L2, is the tool for them
 * and is not wired in here yet.
 *
 * Positions advance by the chunk: every sequence's own position by the rows of it
 * that belong to that sequence, d.steps counts tokens, and d.step_seconds is the
 * wall time of the call, which for T > 1 is T tokens' worth.
 *
 * `rseq`, when set, says which sequence each row belongs to, and that is all a
 * batched step is: T rows, each the next token of a sequence of its own, each with a
 * position counted from its own 0 and a slice of the caches of its own. With it unset
 * every row is the next row of sequence 0, which is the single sequence this function
 * has always walked - the same positions, the same cells, the same bytes. */
static int chunk_impl(qe_forward *f, const int *tokens, int T, float *logits_out,
                      unsigned flags, const int *rseq) {
    if (!f || T < 1 || T > QE_CHUNK_MAX || ((flags & STEP_LOGITS) && !logits_out))
        return -1;
    for (int t = 0; t < T; t++) {
        /* An id outside the vocabulary would read a row that is not in the
         * file; the caller's ids come from the tokenizer, so this is a bug or a
         * corrupt prompt, and either way it is a refusal, not a wild read. */
        if (tokens[t] < 0 || (uint64_t)tokens[t] >= f->embd->ne[1])
            return -1;
    }
    /* One row per sequence's next token, or T tokens of one sequence. Both are the
     * same two arrays: which sequence a row continues, and where in it the row goes.
     * A row of a batched step is the first of its sequence in this chunk and so sits
     * at that sequence's own position; a chunk of one prompt walks that sequence up.
     * Checked row by row because the answer is a per-sequence length and the caller
     * may hand the rows in any order at all. */
    int rs[QE_CHUNK_MAX];
    long ppos[QE_CHUNK_MAX];
    long nrow[QE_MAX_SEQ] = {0};
    for (int t = 0; t < T; t++) {
        const int s = rseq ? rseq[t] : 0;
        if (s < 0 || s >= f->n_seq)
            return -1;
        rs[t] = s;
        ppos[t] = f->spos[s] + nrow[s];
        nrow[s]++;
        if (ppos[t] >= (long)f->seq_stride) {
            f->d.over_ctx++;
            return -1;
        }
    }
    const long pos0 = f->pos;
    const double t0 = now();
    const long nf0 = f->d.nonfinite;
    const unsigned variant = f->o.variant;
    const int last = T - 1;
    double ts = 0.0;
    /* Where the pass in flight starts, and the one row of the last one that has to
     * outlive it. A draft row is fed the token that follows its residual, so the
     * first row of this chunk belongs to a residual that is not in this chunk, and
     * the row a draft stopped at is the row the chunk before the next one reads. Both
     * are the last row of the pass that is ending, which is why the copy happens
     * before anything here overwrites the cursors below. A rewind shortens the chunk
     * in flight to the rows the transcript kept, so this reads - and keeps - exactly
     * the frontier row, and nothing an abandoned row drafted. */
    if (f->mtp_ready) {
        if (f->chunk_T > 0) {
            memcpy(f->mpend, f->cres + (size_t)(f->chunk_T - 1) * (size_t)f->hc_dim,
                   (size_t)f->hc_dim * sizeof(float));
            f->mpend_pos = f->pass_pos0 + f->chunk_T - 1;
        } else {
            f->mpend_pos = -1;
        }
    }
    f->pass_pos0 = pos0;
    f->chunk_T = T;
    f->lg_rows = 0; /* the head's rows of the chunk before this one are history */
    f->mac_mark = f->walk_mac;
    const int moe_grouped = f->moe_grouped && T > 1;
    const int ckpt = (flags & STEP_CKPT) && f->ck && T > 1;

    /* The ids are recorded before anything uses them: the PLE window is the
     * sequence, and its rows are issued here so that the sixteen page walks of a
     * 26.8 GiB table happen while the dense layers before them are running. The
     * queue is a FIFO and both this loop and the collect in ple_block walk t
     * ascending, so a gather is never paired with another token's rows; past
     * QE_CHUNK_MAX pending issues the queue declines and the token falls back to
     * the cold gather, which is a slower answer and not a different one. */
    for (int t = 0; t < T; t++) {
        uint32_t *ids = seq_ids(f, rs[t]);
        ids[ppos[t]] = (uint32_t)tokens[t];
        if (f->ple)
            qe_ple_issue(f->ple, ids, (size_t)ppos[t]);
        /* The cell's rope position, before any layer asks for it. A token continues
         * from the largest component of the cell before it, which for text is the cell
         * itself; an image row brings its own. */
        int32_t *rp = seq_rpos(f, rs[t]) + (size_t)ppos[t] * 3;
        if (f->mm_emb && f->mm_emb[t]) {
            memcpy(rp, f->mm_pos + (size_t)t * 3, 3 * sizeof(int32_t));
        } else {
            int32_t next = 0;
            if (ppos[t] > 0) {
                const int32_t *q = rp - 3;
                next = q[0] > q[1] ? q[0] : q[1];
                next = (next > q[2] ? next : q[2]) + 1;
            }
            rp[0] = rp[1] = rp[2] = next;
        }
        if (rp[0] != ppos[t] || rp[1] != ppos[t] || rp[2] != ppos[t])
            f->mrope[rs[t]] = 1;
    }

    ts = QE_TICK();
    for (int t = 0; t < T; t++) {
        use_row(f, t);
        if (f->mm_emb && f->mm_emb[t]) {
            /* An image row: the vision tower's row is the embedding, in all four
             * streams, exactly where a token's table row would have been put. */
            memcpy(f->res, f->mm_emb[t], (size_t)f->K * sizeof(float));
            for (int hs = 1; hs < f->HC; hs++)
                memcpy(f->res + (size_t)hs * f->K, f->res, (size_t)f->K * sizeof(float));
        } else {
            embed(f, tokens[t]);
        }
    }
    QE_TALLY(QE_ST_EMBED, ts);

    const int pf_on = f->progress_fine && f->pf_total > 0 && T > 1 && f->n_layer <= PF_LAYERS;
    double pf_t[PF_LAYERS + 1];
    if (pf_on)
        pf_t[0] = now();
    for (int il = 0; il < f->n_layer; il++) {
        qe_layer *L = &f->L[il];
        for (int t = 0; t < T; t++) {
            use_row(f, t);
            f->chunk_t = t;
            if (L->ple && !(variant & QE_VARIANT_NO_PLE)) {
                ts = QE_TICK();
                ple_block(f, L, rs[t], ppos[t]);
                QE_TALLY(QE_ST_PLE, ts);
            }
            /* The other minimal ring, and the reason this capture is not next to
             * the layer's state: every row of a chunk is walked before any of that
             * state is checkpointed, so by then the history is the window's last
             * row's and every boundary before it would read the same clobbered
             * slots. */
            if (L->ple && ckpt && t + 1 < T)
                ckpt_ple(f, f->ck, t + 1);
        }
        ts = QE_TICK();
        hc_mixer(f, L, 0, 0, T, 1);
        QE_TALLY(QE_ST_HC_ATTN, ts);
        {
            rows_job rj = { .fn = row_sweep_cxn, .T = T };
            rows_run(f, &rj, 1);
        }
        /* The projections of a block are walks, not recurrences, so each is one
         * call covering the whole chunk: proj computes every row in flight, and
         * only the core - the conv ring, the delta rule, the cache writes - runs
         * per token. */
        ts = QE_TICK();
        if (L->qsa)
            qsa_proj(f, L);
        else
            gdn_proj(f, L);
        QE_TALLY(QE_ST_BLOCK_PROJ, ts);
        if (L->qsa && T > 1 && f->n_seq == 1 && f->attn_batch && f->attn_fused && f->cqwork) {
            ts = QE_TICK();
            qsa_core_chunk(f, L, L->qsa_slot, ppos[0], T);
            QE_TALLY(QE_ST_BLOCK_CORE, ts);
        } else if (L->qsa) {
            for (int t = 0; t < T; t++) {
                use_row(f, t);
                f->chunk_t = t;
                ts = QE_TICK();
                qsa_core(f, L, L->qsa_slot, rs[t], ppos[t]);
                QE_TALLY(QE_ST_BLOCK_CORE, ts);
            }
        } else if (ckpt) {
            /* One row at a time, with the state after each row kept. The chunked
             * form below never materialises a row's state - it walks the rows
             * inside one dispatch and leaves only the last - so the boundary a
             * draft stops at would be nothing to go back to. */
            ts = QE_TICK();
            for (int t = 0; t < T; t++) {
                gdn_core_chunk(f, L, L->gdn_slot, rs[t], ppos[t], t, 1);
                if (t + 1 < T)
                    ckpt_slot(f, f->ck, L->gdn_slot, t + 1);
            }
            QE_TALLY(QE_ST_BLOCK_CORE, ts);
        } else if (f->n_seq > 1) {
            /* One dispatch per sequence: the delta rule shares a row of state across
             * the rows of one sequence, which is the only thing in a GDN block a chunk
             * can share that a projection walk already did not - and a state belongs
             * to one sequence, so nothing crosses between them. */
            ts = QE_TICK();
            for (int t = 0; t < T; t++)
                gdn_core_chunk(f, L, L->gdn_slot, rs[t], ppos[t], t, 1);
            QE_TALLY(QE_ST_BLOCK_CORE, ts);
        } else {
            /* One dispatch for the chunk: the delta rule shares a row of state
             * across the tokens in flight, which is the only thing in a GDN block
             * a chunk can share that a projection walk already did not. */
            ts = QE_TICK();
            gdn_core_chunk(f, L, L->gdn_slot, 0, pos0, 0, T);
            QE_TALLY(QE_ST_BLOCK_CORE, ts);
        }
        /* The read-out is one walk for the chunk and it writes every row of the
         * block-output buffer, which is why the injection into the residual
         * streams is a loop of its own after it. */
        ts = QE_TICK();
        if (L->qsa)
            qsa_out(f, L);
        else
            gdn_out(f, L);
        QE_TALLY(QE_ST_BLOCK_PROJ, ts);
        {
            /* Ablations zero the block's output before it is scattered into the
             * streams, which is exactly the network without that block: the inject
             * weight multiplies a zero, so the residual passes through untouched. */
            const int abl = (L->qsa && (variant & QE_VARIANT_ABL_ATTN)) ||
                            (!L->qsa && (variant & QE_VARIANT_ABL_GDN));
            rows_job rj = { .fn = row_inject, .T = T, .flags = 1u | (abl ? 4u : 0u) };
            rows_run(f, &rj, 1);
        }
        ts = QE_TICK();
        hc_mixer(f, L, 1, 0, T, 1);
        QE_TALLY(QE_ST_HC_FFN, ts);
        ts = QE_TICK();
        if (moe_grouped)
            moe_block_chunk(f, il, L, T);
        else
            for (int t = 0; t < T; t++) {
                use_row(f, t);
                moe_block(f, il, L, f->y);
            }
        QE_TALLY(QE_ST_MOE, ts);
        {
            rows_job rj = { .fn = row_inject, .T = T,
                            .flags = 2u | 8u | ((variant & QE_VARIANT_ABL_MOE) ? 4u : 0u) };
            rows_run(f, &rj, 1);
            use_row(f, last);
            if (flags & STEP_PROBE)
                for (int p = 0; p < f->n_probe; p++)
                    if (f->probe_layers[p] == il)
                        head_probe(f, last, f->probe_buf + (size_t)p * (size_t)f->embd->ne[1]);
        }
        if (f->d.nonfinite > nf0 && getenv("QE_DEBUG_NF")) {
            static int said;
            if (!said++)
                fprintf(stderr, "qe: first non-finite value in layer %d (%s), chunk of %d at %ld\n",
                        il, L->qsa ? "attention" : "gated delta", T, pos0);
        }
        if (pf_on) {
            pf_t[il + 1] = now();
            const long part = f->pf_have ? (long)T * f->pf_cum[il] / 10000
                                         : (long)T * (long)(il + 1) / (long)f->n_layer;
            f->progress_fine(f->progress_fine_user, f->pf_base + part, f->pf_total);
        }
    }
    if (pf_on) {
        const double span = pf_t[f->n_layer] - pf_t[0];
        if (span > 0.0) {
            for (int il = 0; il < f->n_layer; il++)
                f->pf_cum[il] = (int)(10000.0 * (pf_t[il + 1] - pf_t[0]) / span + 0.5);
            f->pf_cum[f->n_layer - 1] = 10000;
            f->pf_have = 1;
        }
    }
    /* The draft head, fed the same chunk. It goes here rather than at the end of the
     * pass because it reuses the trunk's chunk scratch, and the mixer below is what
     * puts the trunk's own row back - so a caller reading a row of this chunk, one
     * logits at a time, never sees the head's write. It is not free (one more block
     * walked per token, against a chunk that already walks the trunk's forty-nine of
     * them) and it is not optional in spirit: without it the first draft of a prompt
     * attends over an empty cache, which is not the state the block was trained in.
     * `QE_MTP_WARM=0` prices it, and `make forward` prints the price.
     *
     * A warm row that goes non-finite fails the pass, the same way the trunk's own
     * arithmetic does. That is deliberate: the only ways it happens are a head
     * quantised into nonsense and a head for another model, and both are worth
     * stopping a request for rather than drafting from. */
    if (f->mtp_ready && f->mtp_warm)
        mtp_warm_chunk(f, pos0, T);
    /* The head mixer is the output norm, and it has no injection: there is
     * nothing left to add the result to. The cursors stay on the chunk's last
     * row, so qe_forward_hidden still reads the hidden state of the token whose
     * logits the caller got. */
    use_row(f, last);
    ts = QE_TICK();
    hc_mixer(f, NULL, 0, last, 1, 0);
    QE_TALLY(QE_ST_HEAD, ts);
    if (flags & STEP_LOGITS) {
        ts = QE_TICK();
        if (flags & STEP_ROWS) {
            /* Every row of a batched step is answered, and the head is the one part of
             * a pass bigger than the cache that would otherwise serve the second row
             * out of the first one's walk: one blocked walk over T rows reads the 497
             * MiB of it once, which at T rows is what makes B sequences cost one
             * token's worth of head. The output norm is re-run per row for the reason
             * qe_forward_row_logits gives: `cxn` is shared by every mixer. */
            for (int t = 0; t < T; t++) {
                use_row(f, t);
                hc_mixer(f, NULL, 0, t, 1, 0);
            }
            const long vocab = (long)f->embd->ne[1];
            gemv_rows(f, f->outw, f->cxn, logits_out, T, vocab);
            for (int t = 0; t < T; t++)
                sweep(f, logits_out + (size_t)t * (size_t)vocab, vocab,
                      &f->d.max_abs_logits);
        } else {
            gemv(f, f->outw, f->xn, logits_out);
            sweep(f, logits_out, (long)f->embd->ne[1], &f->d.max_abs_logits);
        }
        QE_TALLY(QE_ST_LMHEAD, ts);
    }
    /* Every sequence this chunk fed a row to advances by the rows it took, and
     * `f->pos` keeps mirroring sequence 0 because everything that is not a layer
     * (a snapshot, a rewind, the draft head) still speaks of one sequence. */
    for (int s = 0; s < f->n_seq; s++)
        f->spos[s] += nrow[s];
    f->pos = f->spos[0];
    f->d.steps += T;
    f->d.step_seconds = now() - t0;
    return f->d.nonfinite > nf0 ? -2 : 0;
}

int qe_forward_step(qe_forward *f, int token, float *logits_out) {
    if (!f || !logits_out)
        return -1;
    return chunk_impl(f, &token, 1, logits_out, STEP_LOGITS | STEP_PROBE, NULL);
}

int qe_forward_step_seq(qe_forward *f, int s, int token, float *logits_out) {
    if (!f || !logits_out || s < 0 || s >= f->n_seq)
        return -1;
    /* Sequence 0 with no rows to route is the walk a step already was, and saying so
     * here keeps a single-sequence pass off the row map entirely. */
    if (f->n_seq == 1)
        return chunk_impl(f, &token, 1, logits_out, STEP_LOGITS | STEP_PROBE, NULL);
    return chunk_impl(f, &token, 1, logits_out, STEP_LOGITS | STEP_PROBE, &s);
}

/* B rows, one per sequence, in one walk of the weights. The row map is the identity
 * because that is what a batch is: sequence s of the pass is row s of the chunk. */
int qe_forward_step_batch(qe_forward *f, const int *tokens, int B, float *logits_out) {
    if (!f || !tokens || !logits_out || B < 1 || B > f->n_seq || B > QE_CHUNK_MAX)
        return -1;
    int rseq[QE_MAX_SEQ];
    for (int t = 0; t < B; t++)
        rseq[t] = t;
    return chunk_impl(f, tokens, B, logits_out,
                      STEP_LOGITS | STEP_PROBE | STEP_ROWS, rseq);
}

/* The same step with a row map, which is what a server with more sequences than
 * requests needs: a batch does not have to occupy the sequences 0..B-1 in order, because
 * a row only ever needs to know which sequence it continues, and a sequence is a slice
 * of the caches plus its own recurrences plus its own cursor - all of which the map
 * selects. Row t continues sequence `rseq[t]` and is answered at
 * `logits_out + t * qe_forward_vocab()`, so the map is the whole of the difference
 * between this and qe_forward_step_batch, and the identity map is the same call.
 *
 * Two rows of one step may not name the same sequence: a sequence has one cursor, and
 * two rows of one chunk advance it once between them. */
int qe_forward_step_batch_rows(qe_forward *f, const int *tokens, const int *rseq, int B,
                               float *logits_out) {
    if (!f || !tokens || !rseq || !logits_out || B < 1 || B > QE_MAX_SEQ)
        return -1;
    int seen[QE_MAX_SEQ] = { 0 };
    for (int t = 0; t < B; t++) {
        if (rseq[t] < 0 || rseq[t] >= f->n_seq || seen[rseq[t]]++)
            return -1;
    }
    return chunk_impl(f, tokens, B, logits_out,
                      STEP_LOGITS | STEP_PROBE | STEP_ROWS, rseq);
}

/* Forget one sequence: its cells, its delta states, its convolution ring, its PLE
 * window, and its cursor. A server that has answered one conversation and wants that
 * sequence for the next one calls this instead of qe_forward_reset, which would take
 * the other sequences down with it - and which is the reason a batch is a set of
 * sequences and not one sequence with a wider name.
 *
 * The scratch a chunk in flight owns is deliberately not touched: this is called between
 * steps, when there is no chunk. */
int qe_forward_reset_seq(qe_forward *f, int s) {
    if (!f || s < 0 || s >= f->n_seq)
        return -1;
    const int hist_rows = f->gdn_kernel - 1;
    for (int slot = 0; slot < f->gdn_slots; slot++) {
        const long g = gdn_sub(f, slot, s);
        memset(f->gdn + (size_t)g * (size_t)f->n_gdn_heads * (size_t)f->head_state *
                           (size_t)f->head_state,
               0, (size_t)f->n_gdn_heads * (size_t)f->head_state * (size_t)f->head_state *
                      sizeof(float));
        memset(f->gconv + (size_t)g * (size_t)hist_rows * (size_t)f->gdn_channels, 0,
               (size_t)hist_rows * (size_t)f->gdn_channels * sizeof(float));
    }
    const size_t NC = f->seq_stride;
    for (int slot = 0; slot < f->qsa_slots; slot++) {
        const long q = qsa_sub(f, slot, s);
        if (f->kv)
            memset(f->kv + (size_t)q * 2 * NC * f->kv_row, 0,
                   2 * NC * f->kv_row * sizeof(float));
        if (f->kvd)
            memset(f->kvd + (size_t)q * 2 * NC * f->kv_row, 0, 2 * NC * f->kv_row);
        if (f->kvs)
            memset(f->kvs + (size_t)q * 2 * NC * (size_t)f->n_kv_heads, 0,
                   2 * NC * (size_t)f->n_kv_heads * sizeof(float));
        if (f->kraw)
            memset(f->kraw + (size_t)q * NC * (size_t)f->idx_dim, 0,
                   NC * (size_t)f->idx_dim * sizeof(float));
        if (f->kpool)
            memset(kpool_seq(f, q), 0,
                   (NC / (size_t)f->kpool_ratio + 1) * (size_t)f->idx_dim *
                       (f->idx_f16 ? 2 : sizeof(float)));
    }
    if (f->ple_hist)
        memset(seq_ple_hist(f, s), 0, (size_t)f->ple_lag * (size_t)f->hc_dim * sizeof(float));
    /* Whatever was prefetched for the old sequence belongs to the old sequence. */
    qe_ple_drain(f->ple);
    f->spos[s] = 0;
    f->mrope[s] = 0;
    if (s == 0)
        f->pos = 0;
    return 0;
}

long qe_forward_seq_pos(const qe_forward *f, int s) {
    return f && s >= 0 && s < f->n_seq ? f->spos[s] : 0;
}

int qe_forward_n_seq(const qe_forward *f) { return f ? f->n_seq : 0; }

/* A draft window: `T` rows through the trunk in one walk of the weights, the
 * state after every row but the last kept, and no logits at all. The caller reads
 * them a row at a time with qe_forward_row_logits and stops asking at the first
 * row whose token was not the one it drafted, which is what keeps the abandoned
 * rows from paying for the 497 MiB head.
 *
 * `T == 1` is a window the drafter had nothing about: one row, nothing to keep,
 * and the same walk of the trunk a plain step takes. It is accepted because a
 * draft that found nothing is the normal case for new text and not an error, and
 * a caller should not have to know the width at which a window stops being one. */
int qe_forward_step_window(qe_forward *f, const int *tokens, int T, qe_ckpt *c) {
    if (!f || !c || !tokens || T < 1 || T > c->rows || f->n_seq > 1)
        return -1;
    c->pos = f->pos;
    /* The last row of a window has no boundary before it, and a window of one row
     * has no boundary at all; either way the caller must not be handed one that
     * was never written. */
    c->kept = T > 1 ? T : 0;
    f->ck = c;
    const int rc = chunk_impl(f, tokens, T, NULL, STEP_PROBE | STEP_CKPT, NULL);
    f->ck = NULL;
    return rc;
}

/* A window of this many rows or fewer is given its head in one sweep. Sixteen MiB
 * of logits, against a head that costs 12 ms and a third of a gigabyte per row. */
#define QE_HEAD_ROWS 16

/* Every readable row of the window in flight, out of one walk of the head.
 *
 * This is the per-row walk and not an approximation of it: blocking the sweep moves
 * which output rows share a read of the weights, and each row still comes from the
 * same kernel on the same inputs (the type that shares its dequantisation across
 * rows is Q8_0, and the head is Q6_K). What it buys is that T rows read the head
 * once, which is the difference between 497 MiB and T times it.
 *
 * `rows` counts from row 0 - the rows a caller abandons are the trunk it has
 * already walked, not a head it skipped - and `logits_out` holds `rows` rows of the
 * vocabulary. */
int qe_forward_window_logits(qe_forward *f, int rows, float *logits_out) {
    if (!f || !logits_out || rows < 1 || rows > f->chunk_T)
        return -1;
    const long vocab = (long)f->embd->ne[1];
    const long nf0 = f->d.nonfinite;
    /* One output norm per row, into the row of `cxn` that belongs to it: the mixer
     * makes the head's input, and the walk below reads all `rows` of them at once. */
    for (int r = 0; r < rows; r++) {
        use_row(f, r);
        hc_mixer(f, NULL, 0, r, 1, 0);
    }
    gemv_rows(f, f->outw, f->cxn, logits_out, rows, vocab);
    for (int r = 0; r < rows; r++)
        sweep(f, logits_out + (size_t)r * (size_t)vocab, vocab, &f->d.max_abs_logits);
    return f->d.nonfinite > nf0 ? -2 : 0;
}

/* The head for one row of the window that just ran: the output norm for that row
 * and then the walk of the head, which is the only part of a pass a row past the
 * first disagreement is charged for.
 *
 * The norm is re-run for every row, including the last, and that is not
 * redundant: `cxn` is one buffer shared by every mixer in the network, so what a
 * row of a finished window holds in it is the last layer's output and not the
 * head's. Re-running it for a row the pass already did is the same arithmetic
 * twice, and it is what makes this readable after a rewind, where the row count
 * in flight no longer says which rows the pass itself finished.
 *
 * Reading a window a row at a time is the caller's convenience and not a saving:
 * the head is the one part of a pass that is bigger than the cache that would
 * otherwise serve the second row out of the first one's walk, so the first request
 * for a multi-row window pays for all of its readable rows and the rows after it
 * are copies. A rewind does not undo those rows - it moves the state and leaves the
 * activations - and the next chunk clears them, because they describe the walk that
 * made them. */
int qe_forward_row_logits(qe_forward *f, int row, float *logits_out) {
    if (!f || !logits_out || row < 0 || row >= f->chunk_T)
        return -1;
    const long vocab = (long)f->embd->ne[1];
    if (row < f->lg_rows) {
        memcpy(logits_out, f->lg_cache + (size_t)row * (size_t)vocab,
               (size_t)vocab * sizeof *logits_out);
        return 0;
    }
    if (f->head_batch && f->chunk_T > 1 && f->chunk_T <= QE_HEAD_ROWS) {
        if (f->lg_cap < f->chunk_T) {
            free(f->lg_cache);
            f->lg_cache = NULL;
            f->lg_cap = 0;
            f->lg_cache = malloc((size_t)f->chunk_T * (size_t)vocab * sizeof *f->lg_cache);
            if (!f->lg_cache)
                return -1;
            f->lg_cap = f->chunk_T;
        }
        const int rc = qe_forward_window_logits(f, f->chunk_T, f->lg_cache);
        if (rc)
            return rc;
        f->lg_rows = f->chunk_T;
        memcpy(logits_out, f->lg_cache + (size_t)row * (size_t)vocab,
               (size_t)vocab * sizeof *logits_out);
        return 0;
    }
    const long nf0 = f->d.nonfinite;
    use_row(f, row);
    hc_mixer(f, NULL, 0, row, 1, 0);
    gemv(f, f->outw, f->xn, logits_out);
    sweep(f, logits_out, vocab, &f->d.max_abs_logits);
    return f->d.nonfinite > nf0 ? -2 : 0;
}

/* Back to the state after `boundary` rows of the window held in `c`: the delta net
 * and both convolution histories, which is all of a sequence's state that is not
 * addressed by position. The regions go back over the pool for the reason a restore
 * does - 112 MiB is not a memcpy - and between calls into the pass, so the pool is
 * idle here. The prefetch queue is the one part of the rest which is positional and
 * in-flight rather than stored, so it is emptied here as it is by a restore. */
long qe_forward_rewind(qe_forward *f, qe_ckpt *c, int boundary) {
    /* A window is refused for a pass driving more than one sequence, so there is no
     * boundary of one to go back to here either, and refusing is cheaper than a rewind
     * that moved sequence 0 while the rest of the batch carried on from where it was. */
    if (!f || !c || f->n_seq > 1 || boundary < 1 ||
        boundary >= (c->kept ? c->kept : c->rows))
        return -1;
    const char *src = (const char *)c->buf + (size_t)(boundary - 1) * c->stride;
    state_copy(f->n_threads, (char *)f->gdn, src, c->gdn_b);
    state_copy(f->n_threads, (char *)f->gconv, src + c->gdn_b, c->conv_b);
    state_copy(f->n_threads, (char *)f->ple_hist, src + c->gdn_b + c->conv_b, c->ple_b);
    if (f->ple)
        qe_ple_drain(f->ple);
    /* A chunk takes its positions off the sequence's own cursor, so the cursor goes
     * back with the state. Left at what the window's rows reached, the recurrences
     * say the boundary while the layer addresses the cell a draft that was never
     * kept wrote, and the step out of a restored state reads a sibling it never had. */
    f->pos = f->spos[0] = c->pos + boundary;
    /* The rows up to the boundary are still readable, and saying so is what lets
     * the caller take one more look at the last of them: a draft that ended on the
     * token that closes a turn leaves the head's answer for that position as the
     * only thing the next request wants, and it is still in the trunk's per-row
     * activations, which a rewind of the state does not touch. Rows past it are
     * not, because their tokens were never the sequence's. */
    f->chunk_T = boundary;
    return f->pos;
}

/* A boundary of the window that just ran, spelled as a rewind point.
 *
 * What a window keeps is otherwise the target's own business, and there is one
 * moment a caller wants it as something it can put back later: a window starts by
 * feeding the token the last one earned, so boundary 1 of the *first* window is the
 * state after the last token of the prompt - the position the next turn comes back
 * to - and the pass has walked past it before anyone could take a snapshot of it.
 * This is that snapshot without a second walk: the bytes are already in `c`, and
 * the ids are the sequence's own, which past the boundary are drafts that never
 * happened and are not copied. */
int qe_ckpt_state(const qe_forward *f, const qe_ckpt *c, int boundary, qe_state *out) {
    if (!state_matches(f, out) || !c || boundary < 1 || boundary >= c->kept)
        return -1;
    const long pos = c->pos + boundary;
    if (pos < 1 || pos > (long)f->n_ctx)
        return -1;
    const char *src = (const char *)c->buf + (size_t)(boundary - 1) * c->stride;
    state_copy(f->n_threads, (char *)out->buf, src, out->gdn_b);
    state_copy(f->n_threads, (char *)out->buf + out->gdn_b, src + out->gdn_b, out->gconv_b);
    state_copy(f->n_threads, (char *)out->buf + out->gdn_b + out->gconv_b,
               src + out->gdn_b + out->gconv_b, out->ple_b);
    memcpy(out->ids, f->seq, (size_t)pos * sizeof *out->ids);
    out->pos = pos;
    return 0;
}

/* Prompt ingestion is one chunk at a time, the head and the probe heads
 * withheld until the last token of the last one. QE_CHUNK=1 asks for the old
 * walk, one token per weight pass, which is how the two are A/Bed. */
/* `s` < 0 is "sequence 0, the way this function always walked it"; otherwise every
 * row of every chunk belongs to sequence `s` of a pass that is driving more than one
 * of them. The two are the same loop because a prompt is a chunk in which all rows
 * are one sequence's, which is one entry of the row map repeated. */
static int prefill_rows(qe_forward *f, int s, const int *tokens, long n, float *logits_out,
                        const float *const *mm_emb, const int32_t *mm_pos) {
    if (!f || !tokens || n < 1)
        return -1;
    const int width = f->chunk_width;
    int rseq[QE_CHUNK_MAX];
    if (s >= 0)
        for (int t = 0; t < QE_CHUNK_MAX; t++)
            rseq[t] = s;
    for (long done = 0; done < n;) {
        int T = (int)(n - done < width ? n - done : width);
        const int last = (done + T == n);
        unsigned flags = last ? (STEP_PROBE | (logits_out ? STEP_LOGITS : 0)) : 0;
        f->pf_base = done;
        f->pf_total = n;
        f->mm_emb = mm_emb ? mm_emb + done : NULL;
        f->mm_pos = mm_pos ? mm_pos + (size_t)done * 3 : NULL;
        int rc = chunk_impl(f, tokens + done, T, (flags & STEP_LOGITS) ? logits_out : NULL,
                            flags, s >= 0 ? rseq : NULL);
        f->mm_emb = NULL;
        f->mm_pos = NULL;
        f->pf_total = 0;
        if (rc)
            return rc;
        done += T;
        /* After the chunk, so that what a hook reports is tokens that are actually
         * fed, and not at all after the last one, which has nothing left to
         * abandon. A hook that says stop ends the prompt here: the state is left
         * standing at `done` tokens, which is a whole number of chunks and so a
         * legal place for a later reset to start from. */
        if (f->progress && done < n && f->progress(f->progress_user, done, n))
            return -1;
    }
    return 0;
}

int qe_forward_prefill(qe_forward *f, const int *tokens, long n, float *logits_out) {
    return prefill_rows(f, -1, tokens, n, logits_out, NULL, NULL);
}

int qe_forward_prefill_seq(qe_forward *f, int s, const int *tokens, long n, float *logits_out) {
    if (!f || s < 0 || s >= f->n_seq)
        return -1;
    return prefill_rows(f, f->n_seq == 1 ? -1 : s, tokens, n, logits_out, NULL, NULL);
}

int qe_forward_prefill_mm(qe_forward *f, int s, const int *tokens, long n,
                          const float *const *row_emb, const int32_t *row_pos,
                          float *logits_out) {
    if (!f || s < 0 || s >= f->n_seq || (row_emb && !row_pos))
        return -1;
    return prefill_rows(f, f->n_seq == 1 ? -1 : s, tokens, n, logits_out, row_emb, row_pos);
}

int qe_forward_mrope(const qe_forward *f, int s) {
    return f && s >= 0 && s < f->n_seq ? f->mrope[s] : 0;
}

void qe_forward_set_progress_fine(qe_forward *f, void (*cb)(void *, long, long), void *user) {
    if (!f)
        return;
    f->progress_fine = cb;
    f->progress_fine_user = user;
}

void qe_forward_set_progress(qe_forward *f, qe_progress_cb cb, void *user) {
    if (!f)
        return;
    f->progress = cb;
    f->progress_user = user;
}

/* Probes are caller-owned: the layer list and the logits buffer have to outlive
 * the call, which is why this stores two pointers instead of copying. */
void qe_forward_probe_set(qe_forward *f, const int *layers, int n, float *buf) {
    if (!f)
        return;
    f->n_probe = 0;
    f->probe_layers = NULL;
    f->probe_buf = NULL;
    if (!layers || n <= 0 || !buf)
        return;
    for (int i = 0; i < n; i++)
        if (layers[i] < 0 || layers[i] >= f->n_layer)
            return;
    f->probe_layers = layers;
    f->probe_buf = buf;
    f->n_probe = n;
}

unsigned qe_forward_variant(const qe_forward *f) { return f ? f->o.variant : 0; }

void qe_forward_reconfigure(qe_forward *f, unsigned variant, int profile) {
    if (!f)
        return;
    f->o.variant = variant;
    f->o.profile = profile;
}
