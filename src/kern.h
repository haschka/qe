/* Internal helpers shared by the kernel translation units.
 *
 * Kept in a header rather than duplicated because the fp16 and int8->fp32
 * conversions appear in every kernel, and a divergence between copies is the
 * kind of bug that produces a slightly wrong model instead of a crash.
 */
#ifndef QE_KERN_H
#define QE_KERN_H

#include "qe.h"

/* Widest token group the shared-dequant kernels run in a single walk of a quantised
 * block. At eight and below a group's accumulators seat in registers and the
 * dequantised block is parked in an L1 line; above it the two swap places, which costs
 * two loads and a store per token and saves the tail tokens a whole walk of the
 * quantiser - for iq3_s that walk is the dearer half of a row. src/kern_iq.c
 * instantiates the group sizes and splits anything wider, so a caller that can hand a
 * whole bucket of rows to the kernel at once should not cap below this. */
#define QE_IQ_PARK_MAX 32

/* What the walk takes by default, as distinct from the ceiling it can be asked for
 * above. Zero, which is parking off. In isolation a parked group is the faster walk: the
 * probe the forms came out of is monotone to thirty-two tokens, and `make moe`'s
 * uniform-routing table gives 125.7 ms of grouped expert work at 512 rows without them and
 * 109.6 with (`/tmp/moe_tile1.log`, `/tmp/moe_park1.log`). In the engine the same switch
 * costs the expert stage 16 % at the shipping width (14.34 to 16.64 ms/token of
 * `moe_ffn`, `/tmp/f6_wide.log` against `/tmp/f7_wide.log`, where every other stage of the
 * two runs agrees to 2 %), and `make forward`'s cap sweep - one load, every cap, the
 * shipping width and 512 tokens - has put the whole curve twice on the shipping binary.
 * `/tmp/f8_wide.log`: 29.46 ms a token with parking off, 29.17 at eight (the same
 * configuration, so that is the run's own noise: 1 %), 28.93 at twelve, 30.22 at sixteen,
 * 32.51 at twenty-four, 31.62 at thirty-two. `/tmp/f9_wide.log`, a second load: 29.31 off,
 * 29.29 at eight, 29.68 at twelve, 30.09 at sixteen, 33.02 at twenty-four, 31.47 at
 * thirty-two. Twelve is the fastest number on the first table, 1.3 % under the mean of its
 * two identical controls, and 1.3 % over them on the second, which is the same distance the
 * first one was, so the honest reading is that a park
 * buys nothing here and costs a fifth of the stage by sixteen. The engine is what ships, so
 * the default is off and the ceiling above stays at thirty-two, because a kernel that is
 * not reachable from the default still has to be reachable from the measurement that
 * decided it. `QE_PARK=` moves it without a rebuild.
 */
#define QE_IQ_PARK_DEFAULT 0

/* The cap in force (QE_IQ_PARK_DEFAULT unless QE_PARK says otherwise), and the setter
 * that lets one load of a model walk the range instead of one process per point. */
int qe_iq_park_cap(void);
void qe_iq_set_park_cap(int v);

/* Which walk a full group of eight rows takes: 0 is the dequantised block parked in an L1
 * line and reloaded for every row of the group, 1 is the block held in four vector
 * registers with the eight sums in the other eight (src/kern_iq.c).
 *
 * Measured, and off. Against the parked walk, the register form is 0.969 to 1.014 of its
 * rate across 2, 4, 8, 16, 32, 64, 128, 256 and 512 rows - best of three, legs alternating
 * within one process, `make moe`'s shared-dequant table (`/tmp/moe_reg1.log` against the
 * same table in `/tmp/moe_park0_base.log`). Four stores per block per walk and four L1
 * reloads per row is what the form was written to remove, and nothing was waiting on
 * either port, which is the same conclusion the spill count reached for the parked Q8_0
 * form two files over: SROA already promotes the parked line to registers there, so the
 * fifth load per token-block was a count of the C. The forms stay because `QE_PARK=9..11`
 * reaches them and because the third leg of that table is what re-measures this every
 * time `make moe` runs; the default is the walk that ships.
 */
#define QE_IQ3S_REGBLOCK_DEFAULT 0
int qe_iq3s_reg_block(void);
void qe_iq3s_set_reg_block(int on);

/* Whether a shared-dequant walk takes an expert's output rows two at a time, so that one load
 * of a token's activations feeds two rows of the matrix instead of one. Same products, same
 * order, per output element - a kernel question and not a quality one, which is why it has a
 * GMAC/s measurement (`make walk`, `make moe`) and not a loss table. It covers the eights of a
 * token group and leaves a group's leftover tokens to the forms they have today, which is why the
 * difference between its two settings is exactly zero outputs moved at every group width and row
 * count - asserted by the third section of `make walk`, not argued. The same switch carries the
 * dense Q8_0 walk (`QE_Q8_0_MULTI2R_FORM` in `kern.c`), which is the projections' half of an
 * ingested token and worth 1.19x at the width a chunk runs at - 7.36 to 5.77 ms of `block_proj`
 * and 1.5 ms of a 29.7 ms token (`/tmp/f19_wide.log`); the only form either family has room for
 * is the eight-token one, because sixteen accumulators is the register file. */
#define QE_IQ_2ROW_DEFAULT 1
int qe_iq_two_row(void);
void qe_iq_set_two_row(int on);

/* The smallest token group at which each codebook format takes that pair form; 0 means never.
 *
 * They are not the same number, and the reason is the register file. The pair form holds two rows'
 * dequantised blocks beside eight accumulators apiece - twenty-four vectors in a sixteen-vector
 * file - so what it buys on activation loads it pays for in stack traffic, and only a format whose
 * dequantise is dear enough to still come out ahead after that keeps the form. Measured at
 * K=2560 over 400000 rows, one activation scale per row, min of three reps, in the third section
 * of `make walk` (`/tmp/walk_diag.log`):
 *
 *   IQ3_S   1.11x at eight tokens, 1.07x at eleven, 1.02x at thirteen, 1.14x at sixteen,
 *           1.14x at thirty-two
 *   IQ4_NL  0.91x at eight, 0.95x at eleven, 0.96x at thirteen, 0.94x at sixteen,
 *           0.92x at thirty-two
 *
 * An iq3_s block is eight 256-entry grid lookups and a sign spread per 32 weights, so a second row
 * of the same activations is worth more than the spill; an iq4_nl block is a shuffle and a lookup,
 * and its pair form is 4-9 % off the walk it replaced at every width a bucket or a grouped expert
 * pass arrives at. Nothing is deleted and nothing is hidden: the form stays compiled, the two legs
 * of both benches still race it (`make walk` forces it on through this gate so its identity check
 * stays a check of the form and not of the default), and `make moe`'s shared-dequant table now
 * carries the mixed case as its own column, because a bucket walks both formats at once and the
 * two answers are not the same size. */
#define QE_IQ4_NL_PAIR_MIN_T 0
#define QE_IQ3_S_PAIR_MIN_T 8
int qe_iq_pair_min_t(qe_type t);
void qe_iq_set_pair_min_t(qe_type t, int min_T);

/* fp16 -> fp32.
 *
 * This function used to be `(uint32_t)h.d << 16`, which is a bf16 widening, not
 * an fp16 conversion: it drops the five-bit fp16 exponent on top of the eight
 * -bit fp32 exponent field and the mantissa three bits too high, so the result
 * was wrong by a factor that depended on the exponent (measured: 0.0078x for a
 * scale of 1.0, 3.7e-7x for 0.242). Every quantized kernel scales its blocks by
 * an fp16 `d`, so the bug silently rescaled Q8_0, Q6_K, IQ4_NL, IQ3_S and
 * IQ4_XS outputs, and no test caught it because the scalar reference in
 * src/kern.c divides by the same wrong factor - bench_kern measures the SIMD
 * kernel against that reference, so a shared error cancels. bench_kern now
 * checks every one of the 65536 fp16 patterns against an exact bit-field
 * reference; do not "simplify" this back to a shift.
 *
 * The fast arm is the arithmetic form of the field conversion: shifting the
 * mantissa up 13 and adding (127 - 15) << 23 = 0x38000000 to the exponent field
 * is exact for every normal fp16, and the denormal/zero/infinite patterns take
 * the second arm. The scalar form is deliberate: it is called once per 32
 * weights in the inner loops, and four integer ops plus a well-predicted branch
 * is cheaper here than an F16C round trip through memory. */
static inline float qe_half_to_f32(qe_half h) {
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t x = h.d;
    uint32_t e = (x >> 10) & 0x1fu;
    uint32_t m = x & 0x3ffu;
    if (e - 1u < 30u) { /* normal: neither zero, denormal, inf nor NaN */
        v.u = ((x & 0x8000u) << 16) | (((x & 0x7fffu) << 13) + 0x38000000u);
        return v.f;
    }
    if (e == 0u) { /* signed zero, or a denormal: m * 2^-24 is exact */
        float mag = (float)m * 5.9604644775390625e-8f;
        return (x & 0x8000u) ? -mag : mag;
    }
    v.u = ((x & 0x8000u) << 16) | 0x7f800000u | (m << 13); /* inf or NaN */
    return v.f;
}

/* float -> fp16, the direction the quantisers need and the decoders never had.
 *
 * Rounding is round-half-to-even, which is what makes fp16 -> f32 -> fp16 the
 * identity for every normal pattern; bench_kern checks exactly that over all
 * 65536 of them. Below the smallest normal half (2^-14) the value is expressed in
 * units of 2^-24 and rounds half away from zero, because there is no even bit to
 * break the tie with, and that is the only place the two directions are not exact
 * inverses. Values past the largest finite half saturate to infinity rather than
 * becoming one: a weight scale that large is a broken tensor, and a loud wrong
 * answer beats a plausible one. */
static inline qe_half qe_f32_to_half(float f) {
    union {
        float f;
        uint32_t u;
    } v;
    qe_half h;
    v.f = f;
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    union {
        float f;
        uint32_t u;
    } mag;
    mag.u = v.u & 0x7fffffffu; /* |f| by clearing the sign, so NaN stays NaN */
    const float a = mag.f;
    if (!(a >= 0.0f)) { /* NaN: neither comparison above was true */
        h.d = (uint16_t)(sign | 0x7e00u);
        return h;
    }
    if (a >= 65520.0f) { /* past 65504, the largest finite half, or infinite */
        h.d = (uint16_t)(sign | 0x7c00u);
        return h;
    }
    if (a < 6.103515625e-05f) { /* 2^-14: the subnormal range */
        h.d = (uint16_t)(sign | (uint32_t)(a * 16777216.0f + 0.5f));
        return h;
    }
    v.f = a;
    uint32_t e = ((v.u >> 23) & 0xffu) - 127u + 15u;
    uint32_t m = v.u & 0x7fffffu;
    const uint32_t rest = m & 0x1fffu;
    m >>= 13;
    if (rest > 0x1000u || (rest == 0x1000u && (m & 1u))) {
        m++;
        if (m > 0x3ffu) { /* mantissa carried into the exponent */
            m = 0;
            e++;
        }
    }
    if (e >= 31u) {
        h.d = (uint16_t)(sign | 0x7c00u);
        return h;
    }
    h.d = (uint16_t)(sign | (e << 10) | m);
    return h;
}

/* How many output rows of one matrix an integer walk takes at a time, so that one load of a
 * panel row and one copy of the exit operands feed that many rows instead of one
 * (`qe_i8_rows`, `QE_I8_R`). Two is the most any format has room for: two rows against the
 * widest panel group is sixteen accumulators, and the accumulators are the register file.
 *
 * It is one for every integer format, because all three measured the same way. The tile built
 * for IQ3_S and IQ4_NL is worth 0.95-1.04x of the walks it replaces at every width and in both
 * cache regimes, and the Q8_0 tile that used to run two is 0.89-0.97x of the two rows it replaces
 * (ideas.md 7.12, three runs of `make walk`'s fourth section). The reason is arithmetic and not
 * scheduling: the exit an output row owes is about seven operands per dot, a pair shares three of
 * them, and two rows against the widest panel group is sixteen accumulators - which is the
 * register file, so the loads a pair exists to save have nowhere to be kept.
 *
 * Both settings build the same accumulator chain per output row, in the same order, so the two
 * leave bit-identical outputs - checked over the K, row count and group width a bucket arrives
 * at (`make fast`) and over the shape a stage does (`make walk`'s fourth section). This is a
 * kernel question with a GMAC/s measurement and not a loss table. */
#define QE_I8_ROWS_DEFAULT 1
int qe_i8_rows(int dflt);
void qe_i8_set_rows(int n);

/* How wide a token group a codebook walk opens a pass on. A pass over one output row costs
 * the decode of that row's blocks plus one dot per token, so the tokens a group holds are
 * the cheap part of a pass and the passes a group needs are not. The family stopped at eight
 * tokens while the routing hands it ten - a 512-token chunk routes 5120 slots over 512 experts -
 * so the modal bucket walked an expert's weights twice. Ten, twelve and sixteen are built;
 * `QE_I8_T` picks the widest, and a group is then divided by the fewest passes that model allows
 * rather than by the widest class that happens to fit.
 *
 * The ceiling stays at eight because the walk is not where the time is. On a tensor the size of a
 * dense matrix one pass of a bucket of ten is worth 1.05-1.11x on IQ4_NL and 1.32-1.34x on IQ3_S
 * over an eight-and-a-two walk, and past eleven token rows the gain is inside what the instrument
 * can see: its own control rows, eight and nine, are the same walk at every ceiling and reach 1.07x
 * on the weaker of two loads (`make walk`'s sixth section, `/tmp/f52_groups.log`,
 * `/tmp/f56_walk2.log`). At thirty-two rows the sixteen class reads 0.95x and 0.94x there, so it is
 * not "never the loser" either. In the routed pass the same change moves nothing, from `make moe`'s
 * shared-dequant column at every row count from two to 512 through the expert stage of a checkpoint
 * load, where the row-scaled walk's ratio to the
 * panel it replaces reads 0.896x against the 0.891x the eight-token family got on the load before
 * (`/tmp/f54_wide.log`, `/tmp/f35_wide.log`). An expert's gate rows are 688 KiB, so the second
 * pass of a bucket is an L2 re-read and not the DRAM re-read a 0.5 GiB tensor pays - which is the
 * whole of the difference between the two instruments, and ideas.md 7.21 has the table. The wider
 * forms spill for it too (seventy-eight frame stores against twenty-nine at eight tokens), and at
 * eleven to sixteen token rows they cost the signed dot the advantage it has everywhere else - four
 * loads of `make walk` at that ceiling lost a signed codebook leg in every one - which is the
 * assertion that keeps the ceiling where it is.
 *
 * A wider class changes which tokens share a pass and nothing else: every token keeps the
 * same block order, seed and exit, so every setting leaves bit-identical outputs, which is
 * what `make fast` holds it to. */
#define QE_I8_TMAX_DEFAULT 8
int qe_i8_tmax(void);
void qe_i8_set_tmax(int n);

/* Which integer dot a walk spends. `vpdpbusd` multiplies unsigned by signed, so every form
 * built for it puts a +128 on the weights with one xor of the sign bit and owes 128 times the
 * activation block's sum back, which the panel carries as a seed for the dot's accumulator
 * (`qe_quant_q8_0_row`'s `p`). `vpdpbssd` (AVX-VNNI-INT8) multiplies signed by signed and owes
 * nothing: neither the xor nor the seed is in the loop, which is two of the about seven
 * operands a dot's exit costs, on every row and every format, for nothing.
 *
 * The same value in exact arithmetic, and the same i32 per 32-weight group only where the
 * seed happens to land in the same place: the bias rides inside the float accumulator and is
 * cancelled by the horizontal sum at the end of a row, so the two sums round differently.
 * `check_ssd` in bench_walk.c therefore holds the difference to 1e-5 of the largest |y|
 * instead of to zero, and measures 3.4e-6 of it on this host's three formats. This is a
 * cheaper instruction and not a second quantiser.
 *
 * `qe_i8_ssd` is the answer a codebook walk uses and `qe_i8_ssd_q8` the answer a Q8_0 walk
 * uses, and the two defaults differ: the decode of a codebook row hands the signed dot its
 * operand for free, while a Q8_0 row is already bytes and its walk shows the saved operands
 * as a loss at the widths where it is not ALU-bound (see `qe_i8_ssd` in src/kern.c).
 * `QE_I8_SSD=0` or `=1` puts both on one instruction, `qe_i8_set_ssd` pins both for an A/B,
 * and `qe_i8_unset_ssd` hands the choice back to the configuration. */
int qe_i8_ssd(void);
int qe_i8_ssd_q8(void);
void qe_i8_set_ssd(int on);
/* The Q8_0 dot on its own, for a leg that asks what the Q8_0 instruction choice costs and
 * must not move the codebook walks' answer with it. Negative hands it back. */
void qe_i8_set_ssd_q8(int on);
void qe_i8_unset_ssd(void);

/* Whether a panel carries one activation scale per row or one per 32 values, and therefore
 * whether the walks read `s[t * nb]` or `s[t * nb + b]`. The row scale is what lets a walk
 * fold the weight block's scale and the activation row's scale into one value outside the
 * token loop, dropping a broadcast and a multiply per token per 32 weights - ideas.md 7.10's
 * instrument has that at 1.67x of the dot against the 1.36x the per-block panel gets, on the
 * signed walks only, and it is a change of number rather than of instruction: the coarser
 * range is priced by `make score --ingest` and not by a GMAC/s column.
 *
 * Read by both the panel's fill and the walks, so the two cannot disagree; and a row-scaled
 * panel is a per-block panel with equal scales, so a walk with no row-scale form still reads
 * it correctly. `QE_I8_RS` overrides the default (on), `qe_i8_set_rowscale` pins it for one
 * leg of an A/B, and a negative value hands the choice back to the configuration. */
int qe_i8_rowscale(void);
void qe_i8_set_rowscale(int v);

/* Whether a walk of a row-scaled panel takes the row's scale out of its block loop, which is
 * the difference between the row-scaled panel buying an exit and buying nothing.
 * `qe_i8_rowscale` says what the panel is; this says whether the walk knows it. The codebook
 * walks have had the hoisted exit since the row scale landed (`QE_I8_RS`); the Q8_0 walk runs
 * on the same panel and did not, because a row-scaled panel is also a valid per-block panel.
 * On with `QE_I8_HOIST=1`, off with `=0`, pinned per leg by `qe_i8_set_hoist` and handed back
 * by a negative value. It changes no number the model sees beyond the rounding of where the
 * row's scale multiplies: same products, same order, one factor taken outside the sum. */
int qe_i8_hoist(void);
void qe_i8_set_hoist(int v);

/* The engine's one exponential approximation: a degree-5 minimax for exp2 over
 * [-0.5, 0.5] with the integer part rebuilt through the exponent field. src/layer.c
 * has carried the eight-wide form of it since the sigmoid became SIMD, and §13's
 * `make fast` row holds it to 8.5e-08 absolute against a double reference over the
 * whole gate range, checked for monotonicity because a non-monotone gate is legal
 * under every error bound and turns a sigmoid into a generator of ties.
 *
 * Against libm it is 3.5e-06 relative worst over [-14, 14], the range an exponent
 * of either of them is in, where `expf` is 1e-7, so anything that switches to it is
 * a change of arithmetic and not of scheduling: `make fast` prices the error, `make forward` prices the nats.
 *
 * The scalar form exists so a run's tail is approximated the same way its body is,
 * and so a host without AVX2 gets one answer rather than two. */
#define QE_LOG2E 1.4426950408889634f

#include <math.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

static inline float qe_exp2_f32(float x) {
    x = fminf(fmaxf(x, -100.0f), 100.0f);
    /* Round to nearest by the 2^23+2^22 magic rather than `rintf`, whose rounding
     * mode handshake is slower than the five FMAs it rounds. The clamp above is
     * what keeps the shift below inside the exponent field. */
    const float n = (x + 12582912.0f) - 12582912.0f;
    const float f = x - n;
    float p = 1.3333558146428443e-3f;
    p = fmaf(p, f, 9.6181291076284778e-3f);
    p = fmaf(p, f, 5.5504108664815974e-2f);
    p = fmaf(p, f, 2.4022650695910071e-1f);
    p = fmaf(p, f, 6.9314718055994531e-1f);
    p = fmaf(p, f, 1.0f);
    int i;
    memcpy(&i, &p, sizeof i);
    i += ((int)n) << 23;
    memcpy(&p, &i, sizeof p);
    return p;
}

#if defined(__x86_64__)

/* exp2 for eight floats. Clamped first because the exponent trick has no meaning
 * outside the float range, and p lands in [0.707, 1.414] so adding the exponent
 * cannot carry into the sign bit or denormalise - which is why the reduction to
 * [-0.5, 0.5] comes first rather than after. */
static inline __m256 qe_exp2_avx2(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-100.0f)), _mm256_set1_ps(100.0f));
    __m256i n = _mm256_cvtps_epi32(x);
    __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(n));
    __m256 p = _mm256_set1_ps(1.3333558146428443e-3f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291076284778e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504108664815974e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022650695910071e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718055994531e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p),
                                                _mm256_slli_epi32(n, 23)));
}

#endif /* x86 for exp2 */

/* exp() of one contiguous run into another, eight at a time where the eight exist
 * and the same polynomial on the remainder. The runs this is for are one token's
 * heads (48 floats of gated-delta decay) and one cell's query heads (twelve floats
 * of attention weight), so the vectors are few and the call is not: 36 gated-delta
 * layers and 12 attention layers per token. */
static inline void qe_exp2v(const float *x, float *out, long n) {
    long i = 0;
#if defined(__x86_64__)
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i,
                         qe_exp2_avx2(_mm256_mul_ps(_mm256_loadu_ps(x + i),
                                                    _mm256_set1_ps(QE_LOG2E))));
#endif
    for (; i < n; i++)
        out[i] = qe_exp2_f32(x[i] * QE_LOG2E);
}

#if defined(__x86_64__)

#include <immintrin.h>

/* The weight vector of an integer dot, and the dot itself, for a form compiled with `SSD`
 * either 0 or 1.
 *
 * `SSD=0` is the unsigned×signed instruction: the weights arrive with their sign bit xored,
 * which is the +128 that operand carries, and the panel's per-block seed cancels it inside the
 * accumulator. `SSD=1` is the signed×signed instruction, which needs neither - so the xor and
 * the seed are not compiled at all for that form, rather than hoisted or branched on. `SSD` is
 * a literal at every call, which is what lets the seed's broadcast disappear with the xor.
 *
 * The two produce the same i32: the seed exists only to pay back the bias the xor introduced.
 * Where the compiler does not have the signed instruction the second form is never
 * instantiated and `QE_I8_SSD` has nothing to select. */
#if defined(__AVXVNNIINT8__)
#define QE_I8_BIAS(SSD, V, M80) ((SSD) ? (V) : _mm256_xor_si256((V), (M80)))
#define QE_I8_DOT(SSD, U, X, PV)                                                                     \
    ((SSD) ? _mm256_dpbssd_epi32(_mm256_setzero_si256(), (U), (X))                                   \
           : _mm256_dpbusd_epi32((PV), (U), (X)))
#else
#define QE_I8_BIAS(SSD, V, M80) _mm256_xor_si256((V), (M80))
#define QE_I8_DOT(SSD, U, X, PV) _mm256_dpbusd_epi32((PV), (U), (X))
#endif

/* A weight vector straight out of a block: biased for the unsigned dot, itself for the
 * signed one. */
#define QE_I8_W(SSD, PTR, M80)                                                                       \
    QE_I8_BIAS((SSD), _mm256_loadu_si256((const __m256i *)(PTR)), (M80))

/* The integer walk for one type, behind qe_gemv_i8_t. `ldd` is in blocks. Each is the
 * R-output-rows-by-T-panel-rows tile for its format: Q8_0 in src/kern.c, where the
 * quantiser that fills the panel lives, and IQ3_S/IQ4_NL in src/kern_iq.c, where the
 * codebooks do - the two decodes produce int8 weights as the first thing they produce,
 * and it is the walk that stops turning them into floats before spending them. */
void qe_gemv_q8_0_i8(const void *restrict W, const qe_i8panel *restrict a, int T,
                     float *const *restrict ys, long n_rows, long K, long ldd_blocks);
/* The same walk for a copy of the matrix that carries one scale per output row instead of one
 * per 32 weights, which is what leaves it with no exit at all: the block loop is one load and
 * one `vpdpbssd` per token, and the two scales are paid once after the loop. `ws` is one float
 * per row of the whole matrix, `ldd` the byte stride between the copy's rows, and `tile_r` how
 * many output rows share one activation load (1 to 5; 3 rows by 4 tokens and 5 rows by 2 tokens
 * are the two tiles that fill the register file exactly). The panel has to be a row-scaled one, and a token group the tile does not
 * divide returns nonzero rather than walking part of it at the wrong scale. Compiled for the
 * signed instruction only - `ideas.md` 7.17. */
int qe_gemv_q8_0_flat_i8(const void *restrict W, const float *restrict ws,
                         const qe_i8panel *restrict a, int T, float *const *restrict ys,
                         long n_rows, long K, long ldd, int tile_r);
void qe_gemv_iq4_nl_i8(const void *restrict W, const qe_i8panel *restrict a, int T,
                       float *const *restrict ys, long n_rows, long K, long ldd_blocks);
void qe_gemv_iq3_s_i8(const void *restrict W, const qe_i8panel *restrict a, int T,
                      float *const *restrict ys, long n_rows, long K, long ldd_blocks);

static inline __m256 qe_cvt8to8_f32(__m128i q8) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
}

static inline float qe_hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline __m256i qe_nib_lo(__m256i v) {
    return _mm256_and_si256(v, _mm256_set1_epi8(0x0f));
}
static inline __m256i qe_nib_hi(__m256i v) {
    return _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0f));
}

/* acc[0..3] += (32 int8 weights in w) * (32 fp32 values at x).
 *
 * `acc` is a four-element local __m256 array in the caller, indexed only by
 * constants here, so it stays in registers. Eight extracts/moves for the
 * int8->fp32 widening is the price of not carrying a second fp32 copy of every
 * weight: at 3.4-8.5 bpw the weights arrive faster than one vector per cycle
 * of ALU work would suggest, and re-checking this against the bandwidth probe
 * is what decides whether it is worth replacing.
 */
#define QE_FMA_I8_X32(acc, w, x)                                              \
    do {                                                                     \
        __m128i _a = _mm256_castsi256_si128(w);                              \
        __m128i _b = _mm256_extracti128_si256(w, 1);                         \
        (acc)[0] = _mm256_fmadd_ps(qe_cvt8to8_f32(_a),                        \
                                   _mm256_loadu_ps((x) + 0), (acc)[0]);      \
        (acc)[1] = _mm256_fmadd_ps(qe_cvt8to8_f32(_mm_srli_si128(_a, 8)),    \
                                   _mm256_loadu_ps((x) + 8), (acc)[1]);      \
        (acc)[2] = _mm256_fmadd_ps(qe_cvt8to8_f32(_b),                       \
                                   _mm256_loadu_ps((x) + 16), (acc)[2]);     \
        (acc)[3] = _mm256_fmadd_ps(qe_cvt8to8_f32(_mm_srli_si128(_b, 8)),    \
                                   _mm256_loadu_ps((x) + 24), (acc)[3]);     \
    } while (0)

#define QE_ACC_DECL(acc) __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), \
                                           _mm256_setzero_ps(), _mm256_setzero_ps() }
#define QE_ACC_HSUM(acc)                                                                   \
    (qe_hsum(_mm256_add_ps(_mm256_add_ps((acc)[0], (acc)[1]),                              \
                           _mm256_add_ps((acc)[2], (acc)[3]))))

/* Fold one group's four partial sums into `tot`, scaled by s, with no
 * horizontal add. A horizontal sum is ~10 cycles of pure dependency; doing one
 * per 32 weights instead of one per row costs more than the dequant itself on
 * the wide-block formats. */
#define QE_ACC_FOLD(acc, tot, s)                                                           \
    (tot) = _mm256_fmadd_ps(_mm256_add_ps(_mm256_add_ps((acc)[0], (acc)[1]),               \
                                          _mm256_add_ps((acc)[2], (acc)[3])),              \
                            _mm256_set1_ps(s), (tot))

#endif /* x86 */
#endif /* QE_KERN_H */
