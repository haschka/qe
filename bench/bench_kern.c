/* Kernel validation + throughput, on synthetic data.

   Deliberately does not open the checkpoint: the 87 GiB file belongs in the
   slow, run-once harness. This one has to execute in a couple of seconds so it
   stays cheap enough to run after every kernel edit, which is the only way a
   dequant bug gets caught the same minute it is written.

   Every type gets the same treatment: fill a few hundred megabytes with
   arbitrary bytes (all of these formats decode every bit pattern), compare the
   SIMD dot against the scalar reference, then sweep thread counts and report
   the rate as tokens/second against the 5.44 GiB decode budget.

   Every byte pattern is legal, so the fill is pure noise plus one concession:
   the fp16 super-scale is forced into a finite range, because a NaN scale
   makes the reference NaN too and silently deletes the comparison.

   Two families of check, and only the second one catches a wrong reading of
   the format: (a) SIMD dot against the scalar reference, which catches
   vectorisation errors and nothing else, because both sides decode the bytes
   the same way; (b) interpretation checks that share no code with the kernels -
   all 65536 fp16 patterns against an IEEE reference, a hand-computed Q8_0
   answer, and exact scale linearity per type. Family (a) passed while
   qe_half_to_f32 was a bf16 shift; family (b) is why it will not happen again.
*/
#include "qe.h"
#include "kern.h"
#include "tables.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern double qe_dot_q8_0_ref(const qe_block_q8_0 *w, const float *x, long nb);
extern double qe_dot_q6_k_ref(const qe_block_q6_k *w, const float *x, long nb);
extern double qe_dot_iq4_nl_ref(const qe_block_iq4_nl *w, const float *x, long nb);
extern double qe_dot_iq4_xs_ref(const qe_block_iq4_xs *w, const float *x, long nb);
extern double qe_dot_iq3_s_ref(const qe_block_iq3_s *w, const float *x, long nb);
extern double qe_dot_q4_k_ref(const qe_block_q4_k *w, const float *x, long nb);
extern double qe_dot_q5_0_ref(const qe_block_q5_0 *w, const float *x, long nb);

typedef double (*dot_ref_fn)(const void *, const float *, long);
typedef void (*gemv_fn)(const void *, const float *, float *, long, long, long);

static double w8(const void *w, const float *x, long nb) {
    return qe_dot_q8_0_ref((const qe_block_q8_0 *)w, x, nb);
}
static double w6(const void *w, const float *x, long nb) {
    return qe_dot_q6_k_ref((const qe_block_q6_k *)w, x, nb);
}
static double w4n(const void *w, const float *x, long nb) {
    return qe_dot_iq4_nl_ref((const qe_block_iq4_nl *)w, x, nb);
}
static double w4x(const void *w, const float *x, long nb) {
    return qe_dot_iq4_xs_ref((const qe_block_iq4_xs *)w, x, nb);
}
static double w3(const void *w, const float *x, long nb) {
    return qe_dot_iq3_s_ref((const qe_block_iq3_s *)w, x, nb);
}
static double w4k(const void *w, const float *x, long nb) {
    return qe_dot_q4_k_ref((const qe_block_q4_k *)w, x, nb);
}
static double w50(const void *w, const float *x, long nb) {
    return qe_dot_q5_0_ref((const qe_block_q5_0 *)w, x, nb);
}

typedef struct {
    const char *name;
    qe_type type;
    long block_ne, block_sz;
    long scale_off; /* fp16 super-scale inside the block, or -1 */
    long scale_off2; /* a second fp16 that scales with it, or -1 (Q4_K's min) */
    gemv_fn gemv;
    dot_ref_fn dot;
    double budget_mib; /* bytes this type costs per decode step */
} kind;

/* Bytes one decode step reads, per type, for this checkpoint. Derived from the
 * tensor table by `make gemv` (bin/bench_gemv prints the same table per
 * component); STEP_MIB is the total of which these are the kernel-covered
 * part. Every tok/s number this tool prints is a measured rate divided by
 * these shares, so a change here moves the whole budget - which is why the
 * figures are the file's, not estimates: the previous set was 5% light and
 * attributed the difference to "F32 and BF16, both tiny". They are 337 MiB. */
#define STEP_MIB 5568.2

static const kind KINDS[] = {
    { "Q8_0", QE_Q8_0, 32, 34, 0, -1, qe_gemv_q8_0, w8, 3708.8 },
    { "IQ3_S", QE_IQ3_S, 256, 110, 0, -1, qe_gemv_iq3_s, w3, 631.1 },
    { "Q6_K", QE_Q6_K, 256, 210, 208, -1, qe_gemv_q6_k, w6, 497.3 },
    { "IQ4_NL", QE_IQ4_NL, 32, 18, 0, -1, qe_gemv_iq4_nl, w4n, 377.9 },
    { "IQ4_XS", QE_IQ4_XS, 256, 136, 0, -1, qe_gemv_iq4_xs, w4x, 16.6 },
    /* Q4_K and Q5_0 have no budget because no decode step reads them: they are
     * the draft head's 2.6 GiB, walked only while a chain is being drafted.
     * They are in this table for the two checks that catch a misread format,
     * which is exactly what a format nobody has ever walked needs. */
    { "Q4_K", QE_Q4_K, 256, 144, 0, 2, qe_gemv_q4_k, w4k, 0.0 },
    { "Q5_0", QE_Q5_0, 32, 22, 0, -1, qe_gemv_q5_0, w50, 0.0 },
};

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    const void *W;
    const float *x;
    float *y;
    long rows, K, ldd, row_bytes;
    gemv_fn gemv;
} gemv_job;

static void gemv_task(void *v, int tid, int nt) {
    gemv_job *j = (gemv_job *)v;
    long per = (j->rows + nt - 1) / nt;
    long lo = (long)tid * per, hi = lo + per;
    if (hi > j->rows) hi = j->rows;
    if (lo >= hi) return;
    /* The row offset is not decoration: passing the same base to every thread
     * makes the whole sweep read one slice instead of the whole buffer, which
     * fits in L3 and reports memory bandwidth the hardware never had to
     * deliver. */
    j->gemv((const uint8_t *)j->W + (size_t)lo * j->row_bytes, j->x, j->y + lo, hi - lo, j->K,
            j->ldd);
}

/* ---------------------------------------------- interpretation, not vectorisation
 *
 * The rest of this file measures the SIMD dot against the scalar reference in
 * src/kern.c. That is a real test of vectorisation and no test at all of
 * interpretation: when qe_half_to_f32 was a bf16 left shift, both sides divided
 * by the same exponent-dependent wrong factor, and the worst relative error
 * stayed at 1e-16 while every block scale in every quantized kernel was wrong.
 * These checks share no code with the kernels.
 *
 * Reference below is field arithmetic over all 65536 fp16 patterns, written
 * against the IEEE-754 definition rather than against anything in this tree. */
static int check_half_conversion(void) {
    int bad = 0;
    for (long h = 0; h <= 0xffffL; h++) {
        uint16_t x = (uint16_t) h;
        uint32_t e = (x >> 10) & 0x1fu, m = x & 0x3ffu;
        double want;
        if (e == 0u)
            want = ldexp((double) m, -24);
        else if (e == 31u)
            want = m ? NAN : INFINITY;
        else
            want = ldexp(1.0 + m / 1024.0, (int) e - 15);
        if (x & 0x8000u)
            want = -want;
        qe_half hx;
        hx.d = x;
        double got = qe_half_to_f32(hx);
        int ok = (want != want) ? (got != got) : got == want;
        if (!ok) {
            if (bad < 3)
                printf("  FAIL fp16 0x%04X: got %.9g, exact %.9g\n", x, got, want);
            bad++;
        }
    }
    printf("fp16 conversion   : %s (%d of 65536 patterns)\n", bad ? "FAIL" : "pass", bad ? bad : 65536);
    return bad;
}

/* fp32 -> fp16 -> fp32 is the identity for every pattern but a NaN, whose payload
 * an encoder is free to choose: this is what says qe_f32_to_half rounds the way
 * qe_half_to_f32 reads, which is the pair a quantiser depends on. */
static int check_half_roundtrip(void) {
    int bad = 0, checked = 0;
    for (long h = 0; h <= 0xffffL; h++) {
        qe_half h0;
        h0.d = (uint16_t) h;
        const float f = qe_half_to_f32(h0);
        if (h0.d && ((h0.d >> 10) & 0x1fu) == 31u && (h0.d & 0x3ffu))
            continue; /* NaN: payload is not promised */
        checked++;
        const float back = qe_half_to_f32(qe_f32_to_half(f));
        if (back != f) {
            if (bad < 3)
                printf("  FAIL fp16 0x%04X: %.9g encoded back to %.9g\n", h0.d, (double) f,
                       (double) back);
            bad++;
        }
    }
    printf("fp16 round trip   : %s (%d patterns, exact)\n", bad ? "FAIL" : "pass", checked);
    return bad;
}

/* The IQ4_NL encoder, checked against this file's own decoder rather than against
 * itself: every weight of a quantised block is read back on its own, with a unit
 * activation, so a nibble swapped between the two halves of a block, a codebook
 * index off by one, or a scale read as something else cannot hide. The error
 * figures are printed because the point of an encoder this cheap is that the loss
 * is measured on real text and not promised here. */
static int check_iq4_nl_quant(void) {
    int bad = 0, skipped = 0;
    float v[32], one[32];
    double worst_rms = 0.0, worst_abs = 0.0;
    /* The smallest block amplitude the format can name at all: a block scale is one
     * fp16, the smallest positive half is 2^-24, and the largest magnitude the scale
     * multiplies is 127. A block below that has no scale that is not zero, so it
     * decodes to zero whatever the encoder does, and scoring it would be a test of the
     * format rather than of the encoder. */
    const float floor_amax = 5.9604645e-8f * 127.0f;
    uint64_t s = 0x243f6a8885a308d3ull;
    for (int trial = 0; trial < 4000; trial++) {
        const float scale = (trial % 7 == 3) ? 1e-6f : 0.05f;
        for (int i = 0; i < 32; i++) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            const double u = (double)((s * 0x2545f4914f6cdd1dull) >> 11) * 0x1p-53;
            v[i] = (float)((u - 0.5) * 2.0) * scale;
        }
        if (trial % 5 == 0)
            v[11] = 0.0f;
        if (trial % 11 == 0) { /* one outsized weight: the case a fixed scale gets wrong */
            for (int i = 0; i < 32; i++)
                v[i] *= 0.05f;
            v[4] = scale;
        }
        float amax = 0.0f;
        for (int i = 0; i < 32; i++)
            if (fabsf(v[i]) > amax)
                amax = fabsf(v[i]);
        if (!(amax >= floor_amax)) {
            skipped++;
            continue;
        }
        qe_block_iq4_nl b;
        if (qe_quant_iq4_nl(v, &b, 32) != 1) {
            printf("  FAIL iq4_nl: one block in, zero blocks out\n");
            bad++;
            break;
        }
        double sse = 0.0, energy = 0.0, worst = 0.0;
        for (int j = 0; j < 32; j++) {
            memset(one, 0, sizeof one);
            one[j] = 1.0f;
            const double got = qe_dot_iq4_nl_ref(&b, one, 1);
            const double err = fabs(got - (double) v[j]);
            sse += err * err;
            energy += (double) v[j] * v[j];
            if (err > worst)
                worst = err;
        }
        const double rms = energy > 0.0 ? sqrt(sse / 32.0) / sqrt(energy / 32.0) : 0.0;
        if (rms > worst_rms)
            worst_rms = rms;
        if (worst > worst_abs)
            worst_abs = worst;
        /* The bound is the format's, not the search's. The worst block of the 3429 this
         * loop scores reads 0.1830 of its own energy through this encoder and 0.1812
         * through the best fp16 scale any encoder could pick, and no block is more than
         * 0.019 away from that best scale, so what is left is the format: four bits of a
         * codebook skewed toward small magnitudes, fed uniform noise, which is the
         * distribution it likes least. 0.20 sits just above that floor; 0.10 and 0.15
         * were both wishes. */
        if (energy > 0.0 && rms > 0.20) {
            if (bad < 3)
                printf("  FAIL iq4_nl trial %d: rms %.4f of the block's own\n", trial, rms);
            bad++;
        }
    }
    /* A block whose weights are codebook values at an exactly representable scale
     * must come back unchanged, to the bit. */
    qe_half d0;
    d0.d = 0x2d00u; /* 1.0/64 in fp16 */
    const double d = qe_half_to_f32(d0);
    for (int i = 0; i < 32; i++)
        v[i] = (float)(d * (double)(int8_t) qe_kvalues_iq4nl[(i * 7 + 3) & 15]);
    qe_block_iq4_nl b;
    qe_quant_iq4_nl(v, &b, 1 * 32);
    for (int j = 0; j < 32; j++) {
        memset(one, 0, sizeof one);
        one[j] = 1.0f;
        if (qe_dot_iq4_nl_ref(&b, one, 1) != v[j]) {
            printf("  FAIL iq4_nl exact block: weight %d moved\n", j);
            bad++;
            break;
        }
    }
    /* Zero stays zero, and the block says so by carrying no scale at all. */
    memset(v, 0, sizeof v);
    qe_quant_iq4_nl(v, &b, 32);
    if (b.d.d != 0u) {
        printf("  FAIL iq4_nl zero block: scale 0x%04X\n", b.d.d);
        bad++;
    }
    /* The Q8_0-input encoder is the same search on the same numbers, and that is a claim
     * worth holding to: a requantised copy of a file is built with it, and the alternative
     * is that the copy is some other encoder's answer than the one scored above. Rebuild a
     * block's values the way the walk that reads them does - d * q - and the two paths have
     * to write the same 18 bytes, including the scale each of them chose. */
    {
        qe_block_q8_0 qb;
        qe_block_iq4_nl by_float, by_q8;
        float vf[32];
        uint64_t s2 = 0x9e3779b97f4a7c15ull;
        int diff = 0, nsc = 0;
        for (int trial = 0; trial < 600; trial++) {
            qb.d = qe_f32_to_half(ldexpf(1.0f, (int)(trial % 23) - 13));
            for (int i = 0; i < 32; i++) {
                s2 ^= s2 >> 12; s2 ^= s2 << 25; s2 ^= s2 >> 27;
                qb.qs[i] = (int8_t)((s2 >> 24) & 0xff);
            }
            if (trial % 7 == 0)
                qb.qs[9] = 0; /* a zero weight inside an otherwise loud block */
            if (trial % 11 == 0) { /* one outlier: the case the scale grid exists for */
                for (int i = 0; i < 32; i++)
                    qb.qs[i] = (int8_t)(1 + (i * 3) % 17);
                qb.qs[3] = 127;
            }
            if (trial % 17 == 0) { /* no scale at all: every value decodes to zero */
                for (int i = 0; i < 32; i++)
                    qb.qs[i] = 0;
            }
            qe_dequant_q8_0(&qb, vf, 32);
            qe_quant_iq4_nl(vf, &by_float, 32);
            qe_quant_iq4_nl_q8_0(&qb, &by_q8, 1);
            nsc++;
            if (memcmp(&by_float, &by_q8, sizeof by_float))
                diff++;
        }
        if (diff)
            printf("  FAIL iq4_nl from Q8_0: %d of %d blocks differ from the float encoder\n",
                   diff, nsc);
        printf("iq4_nl from Q8_0  : %s (%d of %d blocks differ from the float path)\n",
               diff ? "FAIL" : "pass", diff, nsc);
        bad += diff;
    }
    printf("iq4_nl quantiser  : %s (worst rms %.4f of a block, worst weight %.2e, %d of"
           " 4000 blocks below the format's own scale floor and not scored)\n",
           bad ? "FAIL" : "pass", worst_rms, worst_abs, skipped);
    return bad;
}

/* One IQ4_NL block's 32 values, the way the walk that reads them reconstructs them. */
static void iq4_nl_values(const qe_block_iq4_nl *b, float *v) {
    const float d = qe_half_to_f32(b->d);
    for (int j = 0; j < 16; j++) {
        v[j] = d * (float)(int8_t)qe_kvalues_iq4nl[b->qs[j] & 0xf];
        v[j + 16] = d * (float)(int8_t)qe_kvalues_iq4nl[b->qs[j] >> 4];
    }
}

/* The two encoders that make a smaller copy of a Q6_K output head at load time.
 *
 * A requantisation is only as good as the numbers it was handed, so this holds three
 * things about `qe_quant_iq4_nl_q6_k` and `qe_quant_q8_0_q6_k`, in the order they break:
 *
 *   1. That the floats they see are the numbers the walk would have multiplied: the
 *      rebuilt row, dotted with a random activation, against `qe_dot_q6_k_ref` on the same
 *      blocks. A super-block scale read at the wrong stride or a nibble from the wrong half
 *      of a block moves that dot and is invisible to everything downstream, because the
 *      copy is then a perfectly encoded version of the wrong row.
 *   2. That each encoder is the encoder for its own format and not a near relation: hand
 *      the dequantised row to `qe_quant_iq4_nl` / `qe_quant_q8_0` and the two paths have to
 *      write identical bytes, which is what makes a staged copy the model the quantiser
 *      above is scored on rather than a third one nobody scored.
 *   3. Which copy is the lossy one. 4.5 bits against 6.5625 is a real loss and the number
 *      that decides it is the head's answer over real text (`make score`'s decode arms),
 *      not a synthetic block; what is decided here is only the direction, that the Q8_0
 *      copy is nearer than the codebook copy, as the two byte counts say it must be.
 */
static int check_q6_requant(void) {
    enum { NB = 4, K = NB * 256 };
    static qe_block_q6_k b[NB];
    static float v[K], x[K];
    static qe_block_iq4_nl f4[NB * 8], e4[NB * 8];
    static qe_block_q8_0 f8[NB * 8], e8[NB * 8];
    int bad = 0, i4diff = 0, q8diff = 0, nsc = 0;
    double worst_dot = 0.0, worst_i4 = 0.0, worst_q8 = 0.0;
    double tot_i4 = 0.0, tot_q8 = 0.0, tot_e = 0.0;
    uint64_t s = 0x2545f4914f6cdd1dull;
    for (int trial = 0; trial < 120; trial++) {
        for (int n = 0; n < NB; n++) {
            b[n].d = qe_f32_to_half(ldexpf(1.0f, (int)(trial % 17) - 11));
            for (int i = 0; i < 128; i++) {
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                b[n].ql[i] = (uint8_t)(s >> 24);
            }
            for (int i = 0; i < 64; i++) {
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                b[n].qh[i] = (uint8_t)(s >> 24);
            }
            for (int i = 0; i < 16; i++)
                b[n].scales[i] = (int8_t)((int)((s >> (8 + (i % 5))) % 15)) - 7;
            if (trial % 9 == 0)
                for (int i = 0; i < 16; i++)
                    b[n].scales[i] = 1; /* one scale across the block: the quiet case */
            if (trial % 13 == 0)
                b[n].scales[3] = -8; /* the one sub-scale with no positive twin */
        }
        qe_dequant_q6_k(b, v, K);
        for (int i = 0; i < K; i++) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            x[i] = (float)(int16_t)(s >> 48) / 32768.0f;
        }
        const double ref = qe_dot_q6_k_ref(b, x, NB);
        double mine = 0.0, absum = 0.0;
        for (int i = 0; i < K; i++) {
            mine += (double)v[i] * (double)x[i];
            absum += fabs((double)v[i] * (double)x[i]);
        }
        if (absum > 0.0) {
            const double rel = fabs(mine - ref) / absum;
            if (rel > worst_dot)
                worst_dot = rel;
            if (!(rel < 1e-12))
                bad++;
        }
        qe_quant_iq4_nl(v, f4, K);
        qe_quant_iq4_nl_q6_k(b, e4, NB);
        if (memcmp(f4, e4, sizeof f4))
            i4diff++;
        qe_quant_q8_0(v, f8, K);
        qe_quant_q8_0_q6_k(b, e8, NB);
        if (memcmp(f8, e8, sizeof f8))
            q8diff++;
        nsc++;
        for (int j = 0; j < NB * 8; j++) {
            float c4[32], c8[32];
            iq4_nl_values(&e4[j], c4);
            qe_dequant_q8_0(&e8[j], c8, 32);
            double s4 = 0.0, s8 = 0.0, sq = 0.0;
            for (int l = 0; l < 32; l++) {
                const double w = (double)v[j * 32 + l];
                const double d4 = w - (double)c4[l], d8 = w - (double)c8[l];
                s4 += d4 * d4;
                s8 += d8 * d8;
                sq += w * w;
            }
            if (!(sq > 0.0))
                continue;
            const double r4 = sqrt(s4 / sq), r8 = sqrt(s8 / sq);
            if (r4 > worst_i4)
                worst_i4 = r4;
            if (r8 > worst_q8)
                worst_q8 = r8;
            tot_i4 += s4;
            tot_q8 += s8;
            tot_e += sq;
        }
    }
    bad += i4diff + q8diff;
    if (!(tot_e > 0.0 && tot_q8 < tot_i4)) {
        printf("  FAIL q6_K requantise: the Q8_0 copy is not the nearer one (%.3e against"
               " %.3e of the row's energy)\n", tot_q8, tot_i4);
        bad++;
    }
    printf("q6_K requantise   : %s (%d of %d rows off the float path for IQ4_NL, %d for"
           " Q8_0; worst dot %.1e rel; block rms worst %.4f IQ4_NL, %.4f Q8_0)\n",
           bad ? "FAIL" : "pass", i4diff, nsc, q8diff, worst_dot, worst_i4, worst_q8);
    return bad;
}

/* A Q8_0 row built by hand, where the answer needs no kernel: two blocks of 32,
 * scales 2.0 and 0.25, every quant +16, x all 1.0.
 * 32*16*2 + 32*16*0.25 = 1152. */
static int check_q8_anchor(void) {
    qe_block_q8_0 blk[2];
    float x[64], y = 0;
    memset(blk, 0, sizeof blk);
    memset(x, 0, sizeof x);
    for (int i = 0; i < 64; i++)
        x[i] = 1.0f;
    blk[0].d.d = 0x4000u; /* 2.0 */
    blk[1].d.d = 0x3400u; /* 0.25 */
    memset(blk[0].qs, 16, 32);
    memset(blk[1].qs, 16, 32);
    qe_gemv_q8_0(blk, x, &y, 1, 64, 2);
    int bad = y != 1152.0f;
    printf("q8_0 hand answer  : %s (got %.4f, exact 1152)\n", bad ? "FAIL" : "pass", (double) y);
    return bad;
}

/* T activations through one dequantised weight row, against a reference that shares
 * no code with it.
 *
 * Three separate failures are ruled out here, because this kernel is reached through
 * a layout as well as through arithmetic.
 *
 *   1. The number. Two accumulator chains per token, where the single-row kernel has
 *      four, is a different order of summation - so the bound is a tolerance rather
 *      than equality (1e-3 relative against 9e-5 observed), and a lane left behind is
 *      2x off, which no tolerance hides.
 *   2. The strides. x_stride and y_stride belong to the caller, and a kernel that
 *      quietly assumes K and n_rows writes rows of a chunk buffer that the next layer
 *      reads as if they were right. The gaps between activation rows are therefore
 *      filled with 1e9, which a kernel that ignored the stride multiplies by.
 *   3. The row block. qe_gemv_multi_t sweeps a few rows at a time and hands the
 *      kernel a pointer into the middle of the matrix, so the offset that goes wrong
 *      is the second block of rows and never the first; it is asked for two rows per
 *      block, and its answer is then compared against the kernel's own.
 *
 * Odd block counts come from the K list: the shared sweep takes two blocks at a time,
 * so a K that is not a multiple of 64 weights runs the tail once per row, and a wrong
 * tail is wrong for exactly the last block of every row.
 *
 * MAXT is 16 and not the 8 it was because the walk tiles its weight rows as well as
 * grouping its tokens, and the two interact: at sixteen tokens the group loop runs
 * twice inside every row tile, which is the case a width-512 chunk lives on. */
static int check_q8_multi(void) {
    enum { MAXK = 2560, PAD = 8, ROWS = 5, MAXT = 16 };
    static uint8_t W[MAXK / 32 * 34 * ROWS];
    static float xs[MAXT * (MAXK + PAD)];
    float ym[MAXT * (ROWS + 3)], ys[MAXT * (ROWS + 3)];
    const float UNTOUCHED = 12345.0f;
    int bad = 0, cases = 0;
    double worst = 0.0;
    const long KS[] = { 32, 64, 96, 160, 512, 2560 };
    for (unsigned ki = 0; ki < sizeof KS / sizeof KS[0]; ki++) {
        const long K = KS[ki], nb = K / 32, row_bytes = nb * 34;
        const long xstride = K + PAD, ystride = ROWS + 3;
        uint64_t s = 0x243f6a8885a308d3ull ^ (uint64_t)K;
        for (long r = 0; r < ROWS; r++)
            for (long b = 0; b < nb; b++) {
                const long at = r * row_bytes + b * 34;
                s ^= s >> 12;
                uint16_t h = (uint16_t)(0x3800u | ((s >> 16) & 0x3ff)); /* 0.5 .. 1.0 */
                memcpy(W + at, &h, 2);
                for (int i = 0; i < 32; i++) {
                    s = s * 6364136223846793005ull + 1442695040888963407ull;
                    W[at + 2 + i] = (uint8_t)(s >> 33);
                }
            }
        for (int t = 0; t < MAXT; t++)
            for (long i = 0; i < K + PAD; i++)
                xs[t * xstride + i] =
                    i < K ? (float)sin((double)(i + 1) * 0.0017 + t) * 0.5f : 1.0e9f;

        for (int T = 1; T <= MAXT; T++) {
            for (int i = 0; i < MAXT * (ROWS + 3); i++) {
                ym[i] = UNTOUCHED;
                ys[i] = 0.0f;
            }
            if (qe_gemv_multi_t(QE_Q8_0, W, xs, ym, ROWS, K, nb, T, xstride, ystride,
                                (size_t)row_bytes, (size_t)row_bytes * 2) != 0) {
                printf("  FAIL q8_0 multi K=%ld T=%d: the dispatcher refused Q8_0\n", K, T);
                bad++;
            }
            qe_gemv_q8_0_multi(W, xs, ys, ROWS, K, nb, T, xstride, ystride);
            cases++;
            for (int t = 0; t < T; t++)
                for (long r = 0; r < ROWS; r++) {
                    double ref = qe_dot_q8_0_ref((const qe_block_q8_0 *)(W + r * row_bytes),
                                                 xs + t * xstride, nb);
                    double got = ym[t * ystride + r];
                    double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
                    double rel = fabs(ref - got) / den;
                    if (!(rel < 1e-3)) {
                        if (bad < 6)
                            printf("  FAIL q8_0 multi K=%ld T=%d row %ld tok %d: got %.9g,"
                                   " reference %.9g\n", K, T, r, t, got, ref);
                        bad++;
                    }
                    if (rel > worst)
                        worst = rel;
                    if (got != ys[t * ystride + r]) {
                        if (bad < 6)
                            printf("  FAIL q8_0 multi K=%ld T=%d row %ld tok %d: blocked and"
                                   " unblocked disagree (%.9g vs %.9g)\n", K, T, r, t, got,
                                   (double)ys[t * ystride + r]);
                        bad++;
                    }
                }
            /* Rows outside the chunk belong to somebody else. */
            for (int t = T; t < MAXT; t++)
                for (long r = 0; r < ROWS; r++)
                    if (ym[t * ystride + r] != UNTOUCHED) {
                        printf("  FAIL q8_0 multi T=%d wrote token %d\n", T, t);
                        bad++;
                    }
        }
    }
    printf("%-19s: %s (worst rel %.1e over %d shapes)\n", "q8_0 multi-token",
           bad ? "FAIL" : "pass", worst, cases);
    return bad;
}

/* The shared-dequant forms of the two formats the routed experts are stored in,
 * against the double reference in src/kern_iq.c.
 *
 * The same three classes as check_q8_multi, with one substitution. This entry point
 * takes an array of row pointers rather than a stride, because the tokens that chose
 * one expert are wherever the routing left them - so what can go wrong is not a stride
 * but a row: reading xs[0] for the second token multiplies one token's activations
 * against another token's weights, and the answer is wrong by the size of a term
 * rather than by a rounding. Each row is therefore built from different data, and a
 * row is asked for at a *padded* ldd, which catches the other way this family can be
 * wrong: a kernel that assumes a block-packed matrix and walks the rows end to end.
 *
 * The tolerance is the one the Q8_0 form is held to, and for the same reason: the
 * block scale moves onto the weight so one dequantised block can serve every row of
 * the chunk, which is the same sum in a different order. A lane left behind is 2x,
 * a wrong group scale is 3x, and neither survives 1e-3.
 *
 * MAXT is one past the widest shared-dequant group the dispatchers know about
 * (QE_IQ_PARK_MAX, plus the 17, 21, 25 and 33 that step down from it) because T is
 * swept from 2 to MAXT over every shape, and a group size no test reaches is a group
 * size no test protects. The number used to be 6: the expert pass hands these entry
 * points ten rows an expert at the shipping chunk width, which is two past the eight
 * the forms used to stop at. Where a coordinate is that small it is a residue and not
 * a magnitude, and the bound that catches a lane left behind is meaningless on it, so
 * those few are re-priced against the terms that cancelled; see the note in the loop. */
static int check_iq_multi(void) {
    enum { MAXK = 2560, PAD = 8, ROWS = 5, MAXT = QE_IQ_PARK_MAX + 1,
           MAXNB = MAXK / 32 + 1 };
    static uint8_t W[MAXNB * 110 * ROWS];
    static float xs[MAXT * (MAXK + PAD)];
    float ym[MAXT * (ROWS + 3)];
    const float UNTOUCHED = 12345.0f;
    static const long KS[] = { 32, 64, 96, 160, 256, 512, 2560 };
    static const struct {
        qe_type type;
        const char *name;
        long block_ne, block_sz;
        dot_ref_fn dot;
    } TYPES[] = {
        { QE_IQ4_NL, "iq4_nl multi", 32, 18, w4n },
        { QE_IQ3_S, "iq3_s multi", 256, 110, w3 },
    };

    int bad = 0, cases = 0;
    for (unsigned ti = 0; ti < sizeof TYPES / sizeof TYPES[0]; ti++) {
        const long bsz = TYPES[ti].block_sz, bne = TYPES[ti].block_ne;
        double worst = 0, worst_canc = 0;
        int cancelled = 0;
        for (unsigned ki = 0; ki < sizeof KS / sizeof KS[0]; ki++) {
            if (KS[ki] % bne)
                continue;
            const long K = KS[ki], nb = K / bne, ldd = nb + 1;
            const long xstride = MAXK + PAD, ystride = ROWS + 3;
            uint64_t s = 0x243f6a8885a308d3ull ^ ((uint64_t)K << 8) ^ (uint64_t)bsz;
            for (long r = 0; r < ROWS; r++)
                for (long b = 0; b < ldd; b++) {
                    uint8_t *p = W + (r * ldd + b) * bsz;
                    for (long i = 0; i < bsz; i++) {
                        s = s * 6364136223846793005ull + 1442695040888963407ull;
                        p[i] = (uint8_t)(s >> 33);
                    }
                    /* The fp16 super-scale, finite: a NaN scale makes the double
                     * reference NaN too, and the comparison quietly deletes itself. */
                    uint16_t h = 0x3800u | (uint16_t)((s >> 16) & 0x3ff);
                    memcpy(p, &h, 2);
                }
            for (int t = 0; t < MAXT; t++)
                for (long i = 0; i < xstride; i++)
                    xs[t * xstride + i] =
                        i < K ? (float)sin((double)(i + 1) * 0.0017 + t * 3.0) * 0.5f : 1.0e9f;

            for (int T = 2; T <= MAXT; T++) {
                const float *xsp[MAXT];
                float *ysp[MAXT];
                for (int t = 0; t < MAXT; t++) {
                    xsp[t] = xs + t * xstride;
                    ysp[t] = ym + t * ystride;
                }
                for (int i = 0; i < MAXT * (ROWS + 3); i++)
                    ym[i] = UNTOUCHED;
                if (qe_gemv_multi_ptrs_t(TYPES[ti].type, W, xsp, ysp, T, ROWS, K, ldd)) {
                    printf("  FAIL %s K=%ld T=%d: the dispatcher refused it\n", TYPES[ti].name, K,
                           T);
                    bad++;
                    continue;
                }
                cases++;
                for (int t = 0; t < T; t++)
                    for (long r = 0; r < ROWS; r++) {
                        const double ref = TYPES[ti].dot(W + r * ldd * bsz, xsp[t], nb);
                        const double got = ysp[t][r];
                        const double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
                        const double rel = fabs(ref - got) / den;
                        if (rel > worst)
                            worst = rel;
                        if (!(rel < 1e-3)) {
                            /* Too small a number for a relative bound to mean anything.
                             * A dot of random weights lands on the residue of a
                             * cancellation, and float error is owed to the terms that
                             * cancelled rather than to what they left behind, so retry
                             * against the sum of the block magnitudes: measured, the
                             * worst shape in this table is K=2560 row 4 token 24, where
                             * ten blocks of magnitude 4046 cancel to 0.1448 and the
                             * kernel is out by 2.4e-4 - 5.9e-8 of the terms, which is a
                             * third of a float ULP of them and four orders better than
                             * the 1e-3 the relative bound asks of a residue. 4e-6 of the
                             * magnitudes is thirty-two ULP of them, which is where
                             * accumulating ten blocks in float has to land, and it is
                             * still six orders tighter than the error a wrong scale, a
                             * wrong row or a skipped block makes, all of which are of
                             * order the magnitudes themselves. Printed in full because a
                             * bound nobody can see is a bound nobody can argue with. */
                            double mag = 0;
                            for (long b = 0; b < nb; b++)
                                mag += fabs(TYPES[ti].dot(W + (r * ldd + b) * bsz,
                                                          xsp[t] + b * bne, 1));
                            const double aerr = fabs(ref - got);
                            if (cancelled < 6)
                                printf("  note %s K=%ld T=%d row %ld tok %d: got %.9g,"
                                       " reference %.9g, block magnitudes %.4g, abs err %.3g\n",
                                       TYPES[ti].name, K, T, r, t, got, ref, mag, aerr);
                            cancelled++;
                            if (mag > 0 && aerr / mag > worst_canc)
                                worst_canc = aerr / mag;
                            bad += aerr >= 4e-6 * mag;
                        }
                    }
                /* Past the last row of every token belongs to somebody else. */
                for (int t = 0; t < MAXT; t++)
                    for (long r = ROWS; r < ystride; r++)
                        if (ysp[t][r] != UNTOUCHED) {
                            printf("  FAIL %s T=%d wrote past row %d of token %d\n",
                                   TYPES[ti].name, T, ROWS, t);
                            bad++;
                        }
            }
        }
        /* One row is not a chunk, and a type without a shared form has to say so
         * rather than compute something: both are -1 with nothing written. */
        const float *xsp0 = xs;
        float *ysp0 = ym + (ROWS + 3);
        ym[ROWS + 3] = UNTOUCHED;
        if (!qe_gemv_multi_ptrs_t(TYPES[ti].type, W, &xsp0, &ysp0, 1, 1, 32, 1))
            { printf("  FAIL %s: T=1 accepted\n", TYPES[ti].name); bad++; }
        if (!qe_gemv_multi_ptrs_t(QE_Q6_K, W, &xsp0, &ysp0, 2, 1, 256, 1))
            { printf("  FAIL %s: a type with no shared form was accepted\n", TYPES[ti].name); bad++; }
        if (ym[ROWS + 3] != UNTOUCHED)
            { printf("  FAIL %s: a refused call wrote\n", TYPES[ti].name); bad++; }
        printf("%-19s: %s (worst rel %.1e over %d shapes)\n", TYPES[ti].name, bad ? "FAIL" : "pass",
               worst, cases);
        if (cancelled)
            printf("%-19s  %d coordinate(s) missed the relative bound on a residue and were"
                   " priced at the\n%-19s  terms that cancelled instead (worst %.1e of them,"
                   " bound 4e-06)\n", "", cancelled, "", worst_canc);
    }
    return bad;
}

/* The integer walks (include/qe.h, qe_i8panel): one Q8_0 recode of a chunk's
 * activations serving an expert matrix through vpdpbusd.
 *
 * Two promises are checked, and they are checked against each other rather than
 * against a guess at a tolerance. The quantiser promises that a block of 32
 * activations comes back within half its own scale of what went in, and that the
 * seed it hands the kernel is -16 times the sum of the values it *actually stored*.
 * The kernel promises that the dot product of the recoded activations against the
 * same weight row is the dot product it returns - and that is an exact statement,
 * because the number an integer walk multiplies is a float this test can build for
 * itself (a stored value times its block's scale) and the double reference on that
 * number shares no code with the kernel. Between them they cover the two ways this
 * family is new: a seed with the wrong sign, or a sign-bit xor the exit forgot to
 * owe back, moves the answer by the size of a term, and a scale applied to the
 * wrong block moves it by the size of a block. Neither is a rounding.
 *
 * T is swept from 2 to QE_IQ_PARK_MAX+1 for the reason check_iq_multi gives: the
 * step-downs to the 4- and 2-row forms land on odd group sizes, and a group size no
 * test reaches is a group size no test protects. A group of one has to be refused
 * with nothing written, and so does a type with no integer form.
 */
static int check_i8_walk(int ssd, int rs) {
    /* The same three walks against the same double reference, once on the unsigned dot with
     * its bias and once on the signed dot with neither. The two are not bit-for-bit (see
     * `check_ssd` in bench_walk.c: the bias rides inside the float accumulator and cancels
     * only in the horizontal sum), so what is worth printing for both is the error each one
     * has against the reference rather than the difference they have with each other.
     *
     * The third and fourth runs are a panel with one scale for the whole row
     * (`qe_quant_q8_0_row_rs`), once on each dot, which is the same instrument with a
     * different panel: the reference is still built from what the panel stored, so `sp[b]`
     * being the row's scale instead of the block's changes the number the kernel is held to
     * and nothing else. That is the point - a row scale applied to the wrong row, or an
     * accumulator that multiplied the row scale in at both ends, moves this by a term. The
     * unsigned pair is the run that matters: the Q8_0 walk takes the unsigned dot and the row
     * scale by default, and with them it multiplies the row's scale once after the block loop
     * rather than once per block, which is a reassociation a wrong row scale does not stop
     * being wrong for. */
    const int dot_was = qe_i8_ssd();
    const int rs_was = qe_i8_rowscale();
    qe_i8_set_ssd(ssd);
    qe_i8_set_rowscale(rs ? 1 : 0);
    enum { MAXK = 2560, PAD = 8, ROWS = 5, MAXT = QE_IQ_PARK_MAX + 1,
           MAXNB = MAXK / 32 + 1 };
    static uint8_t W[MAXNB * 110 * ROWS];
    static float xs[MAXT * (MAXK + PAD)];
    static int8_t x8[MAXT * MAXK];
    static float sp[MAXT * MAXNB];
    static int32_t pp[MAXT * MAXNB];
    static float xr[MAXT * MAXK]; /* what the walk actually multiplied */
    float ym[MAXT * (ROWS + 3)];
    const float UNTOUCHED = 12345.0f;
    static const long KS[] = { 32, 64, 96, 160, 256, 512, 2560 };
    static const struct {
        qe_type type;
        const char *name, *rs_name;
        long block_ne, block_sz;
        dot_ref_fn dot;
    } TYPES[] = {
        { QE_Q8_0, "i8 q8_0 walk", "i8 q8_0 row scale", 32, 34, w8 },
        { QE_IQ4_NL, "i8 iq4_nl walk", "i8 iq4_nl row scl", 32, 18, w4n },
        { QE_IQ3_S, "i8 iq3_s walk", "i8 iq3_s row scl", 256, 110, w3 },
    };

    int bad = 0, cases = 0;
    for (unsigned ti = 0; ti < sizeof TYPES / sizeof TYPES[0]; ti++) {
        const long bsz = TYPES[ti].block_sz, bne = TYPES[ti].block_ne;
        const char *name = rs ? TYPES[ti].rs_name : TYPES[ti].name;
        double worst = 0, worst_canc = 0;
        int cancelled = 0;
        for (unsigned ki = 0; ki < sizeof KS / sizeof KS[0]; ki++) {
            if (KS[ki] % bne)
                continue;
            const long K = KS[ki], nb = K / bne, ldd = nb + 1;
            /* A panel is always in 32-value blocks; a weight block is not (256 for
             * iq3_s). Reading the panel at the weight block count evaluates the
             * reference on the first bne activations and nothing else, which is a
             * number no kernel computes. */
            const long pb = K / 32;
            const long xstride = MAXK + PAD, ystride = ROWS + 3;
            uint64_t s = 0x9e3779b97f4a7c15ull ^ ((uint64_t)K << 8) ^ (uint64_t)bsz;
            for (long r = 0; r < ROWS; r++)
                for (long b = 0; b < ldd; b++) {
                    uint8_t *p = W + (r * ldd + b) * bsz;
                    for (long i = 0; i < bsz; i++) {
                        s = s * 6364136223846793005ull + 1442695040888963407ull;
                        p[i] = (uint8_t)(s >> 33);
                    }
                    uint16_t h = 0x3800u | (uint16_t)((s >> 16) & 0x3ff);
                    memcpy(p, &h, 2);
                }
            for (int t = 0; t < MAXT; t++)
                for (long i = 0; i < xstride; i++)
                    xs[t * xstride + i] =
                        i < K ? (float)sin((double)(i + 1) * 0.0017 + t * 3.0) * 0.5f
                              : 1.0e9f;

            /* The panel, and what the panel owes. */
            qe_i8panel a = { x8, sp, pp, K, K / 32 };
            for (int t = 0; t < MAXT; t++) {
                if (rs)
                    qe_quant_q8_0_row_rs(xs + t * xstride, x8 + (size_t)t * K, sp + (size_t)t * pb,
                                         pp + (size_t)t * pb, K);
                else
                    qe_quant_q8_0_row(xs + t * xstride, x8 + (size_t)t * K, sp + (size_t)t * pb,
                                      pp + (size_t)t * pb, K);
                for (long b = 0; b < pb; b++) {
                    const float sc = sp[(size_t)t * pb + b];
                    int32_t sum = 0;
                    for (long i = 0; i < 32; i++) {
                        const long k = b * 32 + i;
                        const int8_t q = x8[(size_t)t * K + k];
                        sum += q;
                        xr[(size_t)t * K + k] = sc * (float)q;
                        const double e = fabs((double)xr[(size_t)t * K + k] -
                                              (double)xs[t * xstride + k]);
                        if (!(e <= 0.5 * (double)sc + 1e-6)) {
                            if (bad < 6)
                                printf("  FAIL %s K=%ld tok %d: recoded %.9g is %.3g from %."
                                       "9g, half a block scale is %.3g\n",
                                       name, K, t, (double)xr[(size_t)t * K + k], e,
                                       (double)xs[t * xstride + k], 0.5 * (double)sc);
                            bad++;
                        }
                    }
                    if (pp[(size_t)t * pb + b] != -16 * sum) {
                        if (bad < 6)
                            printf("  FAIL %s K=%ld tok %d block %ld: seed %d, -16 times the "
                                   "sum stored is %d\n",
                                   name, K, t, b, pp[(size_t)t * nb + b], -16 * sum);
                        bad++;
                    }
                }
            }
            /* Rows the caller did not ask for must survive: a quantiser reading past K
             * would find the 1e9 fill here and carry it into a block scale. */
            for (int t = 0; t < MAXT; t++)
                for (long i = K; i < xstride; i++)
                    if (xs[t * xstride + i] != 1.0e9f) {
                        printf("  FAIL %s K=%ld: the quantiser wrote the gap after row %d\n",
                               name, K, t);
                        bad++;
                    }

            /* T = 1 is a decode step: one activation row against every row of the expert it
             * routed to. The codebook walks have a form for that and the Q8_0 tile does not, so
             * the one-token case gates a refusal as well as an answer. */
            for (int T = 1; T <= MAXT; T++) {
                float *ysp[MAXT];
                for (int t = 0; t < MAXT; t++)
                    ysp[t] = ym + t * ystride;
                for (int i = 0; i < MAXT * (ROWS + 3); i++)
                    ym[i] = UNTOUCHED;
                if (qe_gemv_i8_t(TYPES[ti].type, W, &a, T, ysp, ROWS, K, ldd)) {
                    if (T == 1 && !qe_gemv_i8_has_1(TYPES[ti].type))
                        continue;
                    printf("  FAIL %s K=%ld T=%d: the dispatcher refused it\n", name,
                           K, T);
                    bad++;
                    continue;
                }
                cases++;
                for (int t = 0; t < T; t++)
                    for (long r = 0; r < ROWS; r++) {
                        const double ref = TYPES[ti].dot(W + r * ldd * bsz,
                                                         xr + (size_t)t * K, nb);
                        const double got = ysp[t][r];
                        const double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
                        const double rel = fabs(ref - got) / den;
                        if (rel > worst)
                            worst = rel;
                        if (!(rel < 1e-3)) {
                            /* A residue of a cancellation, priced against the blocks that
                             * cancelled exactly as check_iq_multi does. */
                            double mag = 0;
                            for (long b = 0; b < nb; b++)
                                mag += fabs(TYPES[ti].dot(W + (r * ldd + b) * bsz,
                                                          xr + (size_t)t * K + b * bne, 1));
                            const double aerr = fabs(ref - got);
                            if (cancelled < 6)
                                printf("  note %s K=%ld T=%d row %ld tok %d: got %.9g,"
                                       " reference %.9g, block magnitudes %.4g, abs err %.3g\n",
                                       name, K, T, r, t, got, ref, mag, aerr);
                            cancelled++;
                            if (mag > 0 && aerr / mag > worst_canc)
                                worst_canc = aerr / mag;
                            bad += aerr >= 4e-6 * mag;
                        }
                    }
                for (int t = 0; t < MAXT; t++)
                    for (long r = ROWS; r < ystride; r++)
                        if (ysp[t][r] != UNTOUCHED) {
                            printf("  FAIL %s T=%d wrote past row %d of token %d\n",
                                   name, T, ROWS, t);
                            bad++;
                        }
                /* A wider token pass changes which tokens of a group share a pass and nothing
                 * else - same seed, same block order, same exit - so it leaves the bits the
                 * eight-token walk leaves. Unlike the row tile this is checked on a row-scaled
                 * panel too, because the row-scaled forms are the ones the classes were built
                 * for, and a class that read a panel row it was not handed, or wrote the token
                 * it borrowed as a leftover, moves a bit here and nothing in a rate.
                 *
                 * Both sides walk one output row at a time, which is the only way the two are
                 * the same kernel: a ceiling above eight switches the row tile off, and the
                 * tile has no row-scaled form and reads such a panel with the per-block exit it
                 * has always had - correct on equal scales, a different rounding, which is why
                 * the tile comparison below skips this panel too. */
                {
                    float y3[MAXT * (ROWS + 3)], y4[MAXT * (ROWS + 3)];
                    float *ysp3[MAXT], *ysp4[MAXT];
                    for (int t = 0; t < MAXT; t++) {
                        ysp3[t] = y3 + t * ystride;
                        ysp4[t] = y4 + t * ystride;
                    }
                    for (int i = 0; i < MAXT * (ROWS + 3); i++)
                        y3[i] = y4[i] = UNTOUCHED;
                    const int tmax_was = qe_i8_tmax();
                    const int rows_was = qe_i8_rows(1);
                    const int tmax_now = tmax_was >= 16 ? 8 : 16;
                    qe_i8_set_rows(1);
                    qe_i8_set_tmax(tmax_was);
                    if (qe_gemv_i8_t(TYPES[ti].type, W, &a, T, ysp4, ROWS, K, ldd)) {
                        printf("  FAIL %s K=%ld T=%d: the walk refused at %d tokens a pass\n",
                               name, K, T, tmax_was);
                        bad++;
                    }
                    qe_i8_set_tmax(tmax_now);
                    if (qe_gemv_i8_t(TYPES[ti].type, W, &a, T, ysp3, ROWS, K, ldd)) {
                        printf("  FAIL %s K=%ld T=%d: the walk refused at %d tokens a pass\n",
                               name, K, T, tmax_now);
                        bad++;
                    } else {
                        for (int i = 0; i < MAXT * (ROWS + 3); i++)
                            if (y3[i] != y4[i]) {
                                if (bad < 6)
                                    printf("  FAIL %s K=%ld T=%d: %d tokens a pass differ from"
                                           " %d at %d (%.9g against %.9g)\n",
                                           name, K, T, tmax_now, tmax_was, i, (double)y3[i],
                                           (double)y4[i]);
                                bad++;
                            }
                    }
                    qe_i8_set_tmax(tmax_was);
                    qe_i8_set_rows(rows_was);
                }
                /* The tile leaves the single-row walk's bits. Same seed, same block order,
                 * same exit per output row - so this is an equality and not a tolerance, and
                 * it is taken over the K, row count and group width a bucket arrives at
                 * rather than over one shape: a tile that reads a panel row it was not
                 * handed, or writes the row it borrowed as a partner, moves a bit here and
                 * moves nothing in a rate.
                 *
                 * Not on a row-scaled panel: the tile has no row-scaled form, because a row
                 * it shares already spends the exit once, and it reads such a panel with the
                 * per-block exit it has always had. Equal scales make that correct and do not
                 * make it the same rounding - the multiply happens inside the loop instead of
                 * once after it - so the equality would price a difference that is a reassociation.
                 *
                 * The ceiling goes back to eight too: a walk only takes the tile under that
                 * ceiling, so left at the shipping one this check would be the same walk twice. */
                if (!rs && T >= 2) {
                    float y2[MAXT * (ROWS + 3)];
                    float *ysp2[MAXT];
                    for (int t = 0; t < MAXT; t++)
                        ysp2[t] = y2 + t * ystride;
                    for (int i = 0; i < MAXT * (ROWS + 3); i++)
                        y2[i] = UNTOUCHED;
                    const int rows_was = qe_i8_rows(1);
                    const int tmax_was = qe_i8_tmax();
                    const int rows_now = rows_was == 2 ? 1 : 2;
                    qe_i8_set_tmax(8);
                    qe_i8_set_rows(rows_now);
                    if (qe_gemv_i8_t(TYPES[ti].type, W, &a, T, ysp2, ROWS, K, ldd)) {
                        printf("  FAIL %s K=%ld T=%d: the walk refused at %d rows\n",
                               name, K, T, rows_now);
                        bad++;
                    } else {
                        for (int i = 0; i < MAXT * (ROWS + 3); i++)
                            if (y2[i] != ym[i]) {
                                if (bad < 6)
                                    printf("  FAIL %s K=%ld T=%d: the %d-row walk differs from"
                                           " the %d-row one at %d (%.9g against %.9g)\n",
                                           name, K, T, rows_now, rows_was, i,
                                           (double)y2[i], (double)ym[i]);
                                bad++;
                            }
                    }
                    qe_i8_set_rows(rows_was);
                    qe_i8_set_tmax(tmax_was);
                }
            }
        }
        /* A group of one is the whole of a decode step's walk rather than a group the class
         * family divides: the codebook walks answer it from a one-token form, and the Q8_0 walk,
         * whose narrowest tile is a pair whose partner the caller never passed, has to say so
         * rather than read a token row that does not exist. A type with no integer walk at all
         * says so at every width. */
        float one_y = UNTOUCHED, *ysp1 = &one_y;
        float *ysp0 = ym + (ROWS + 3);
        *ysp0 = UNTOUCHED;
        const int one_ok = qe_gemv_i8_has_1(TYPES[ti].type);
        const int one = qe_gemv_i8_t(TYPES[ti].type, W, &((qe_i8panel){ x8, sp, pp, 32, 1 }), 1,
                                    &ysp1, 1, 32, 1);
        if (one && one_ok) {
            printf("  FAIL %s: T=1 refused with a one-token form built\n", name);
            bad++;
        }
        if (!one && !one_ok) {
            printf("  FAIL %s: T=1 accepted with no one-token form\n", name);
            bad++;
        }
        if (!one && one_ok && one_y == UNTOUCHED) {
            printf("  FAIL %s: T=1 answered and wrote nothing\n", name);
            bad++;
        }
        if (!qe_gemv_i8_t(QE_Q6_K, W, &((qe_i8panel){ x8, sp, pp, 256, 8 }), 2, &ysp0, 1, 256, 8)) {
            printf("  FAIL %s: a type with no integer walk was accepted\n", name);
            bad++;
        }
        if (*ysp0 != UNTOUCHED) {
            printf("  FAIL %s: a refused call wrote\n", name);
            bad++;
        }
        printf("%-19s: %s (worst rel %.1e over %d shapes, %s dot%s)\n", name,
               bad ? "FAIL" : "pass", worst, cases, ssd ? "signed" : "unsigned",
               rs ? ", one scale per row" : "");
        if (cancelled)
            printf("%-19s  %d coordinate(s) missed the relative bound on a residue and were"
                   " priced at the\n%-19s  blocks that cancelled instead (worst %.1e of them,"
                   " bound 4e-06)\n", "", cancelled, "", worst_canc);
    }
    qe_i8_set_ssd(dot_was);
    qe_i8_set_rowscale(rs_was);
    /* And the strided quantiser against the per-row one it is a convenience for. */
    {
        enum { T = 4, K = 256, NBB = T * (K / 32) };
        float x[T * K * 2];
        int8_t a8[2][T * K];
        float as[2][NBB];
        int32_t ap[2][NBB];
        for (int i = 0; i < T * K * 2; i++)
            x[i] = (float)sin(i * 0.011) * 3.0f;
        for (int t = 0; t < T; t++)
            qe_quant_q8_0_row(x + (size_t)t * K * 2, a8[0] + (size_t)t * K,
                              as[0] + (size_t)t * (K / 32), ap[0] + (size_t)t * (K / 32), K);
        qe_quant_q8_0_rows(x, K * 2, a8[1], as[1], ap[1], T, K);
        int diff = memcmp(a8[0], a8[1], sizeof a8[0]) || memcmp(as[0], as[1], sizeof as[0]) ||
                   memcmp(ap[0], ap[1], sizeof ap[0]);
        printf("%-19s: %s\n", "q8_0 rows == row", diff ? "FAIL" : "pass");
        if (diff)
            bad++;
    }
    return bad;
}

/* The key/value cache's whole arithmetic in four assertions: one byte per value
 * and one scale per row, read back by qe_dot_q8 and qe_axpy_q8 and written by
 * qe_quant_q8_row. The read side is tested against bytes made by hand and a scale
 * of 0.25, which keeps scale times value exact in binary, so anything the kernel
 * reports that a double reference does not is the kernel's own error rather than
 * the quantizer's. The write side has one honest bound: half a step.
 *
 * These are the kernels the sparse attention runs about two thousand times per
 * query head per full-attention layer per token, so a scale read from the wrong
 * place is a wrong answer per token rather than a crash. */
static int check_kv_q8(void) {
    enum { N = 256 }; /* head_dim as this checkpoint states it */
    static int8_t qs[N];
    float a[N], deq[N], row[N], y[N], sc = 0.0f;
    int bad = 0;
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < N; i++) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        const long long r = (long long)(s * 0x2545f4914f6cdd1dull);
        row[i] = (float)((double)(long)(r % 20000) / 10000.0 - 1.0);
        a[i] = (float)((double)(long)((r >> 7) % 4000) / 2000.0 - 1.0);
    }
    for (int i = 0; i < N; i++) {
        qs[i] = (int8_t)((i * 37) % 200 - 100);
        deq[i] = (float)qs[i] * 0.25f;
    }
    double want = 0.0;
    for (int i = 0; i < N; i++)
        want += (double)a[i] * (double)deq[i];
    const float got = qe_dot_q8(a, qs, N) * 0.25f;
    if (fabs(got - (float)want) > 1e-4 * (fabs((float)want) + 1.0))
        bad++;

    for (int i = 0; i < N; i++)
        y[i] = 1.0f;
    qe_axpy_q8(y, qs, 0.5f * 0.25f, N);
    double worst_axpy = 0.0;
    for (int i = 0; i < N; i++) {
        const double w = 1.0 + 0.5 * (double)deq[i];
        const double e = fabs((double)y[i] - w);
        if (e > worst_axpy)
            worst_axpy = e;
    }
    if (worst_axpy > 1e-4)
        bad++;

    qe_quant_q8_row(row, qs, &sc, N);
    float worst = 0.0f;
    for (int i = 0; i < N; i++) {
        const float back = (float)qs[i] * sc;
        const float e = fabsf(back - row[i]);
        if (e > worst)
            worst = e;
    }
    if (!(sc > 0.0f) || worst > 0.5f * sc * 1.001f)
        bad++;

    /* A row of nothing is nothing: no scale, and no division by one later. */
    float zeros[N] = { 0 };
    for (int i = 0; i < N; i++)
        qs[i] = 7;
    qe_quant_q8_row(zeros, qs, &sc, N);
    if (sc != 0.0f)
        bad++;
    for (int i = 0; i < N; i++)
        if (qs[i] != 0)
            bad++;

    printf("%-19s: %s (dot %.1e, axpy %.1e, write %.4f of a %.4f half-step)\n", "cached q8 row",
           bad ? "FAIL" : "pass", fabs(got - (float)want), worst_axpy, worst, 0.5f * sc);
    return bad;
}

/* The fused attention walk against the arithmetic it replaced.
 *
 * What the engine now does is read one KV group's rows once for the twelve query
 * heads that share them instead of twelve times, carry the softmax inside the walk
 * instead of in a score array, and cut the selection into chunks so that twenty
 * cores have something to do. None of that is allowed to move an answer: the same
 * cache, the same selection and the same per-head softmax are computed here in
 * double as well, and the fused result has to sit inside a few parts in
 * ten million of that. Scales are powers of two, exactly as in `check_kv_q8`, so
 * the quantizer contributes nothing and what is being measured is the kernel's own
 * error. The head-at-a-time form is computed alongside in f32 - two passes, a
 * score array, the order the indexer's heap produced - because the honest claim
 * about a reassociation is a number, not an adjective.
 *
 * Six shapes, because the chunking has interesting edges: a selection shorter than
 * one chunk, one that divides evenly, one that does not; the f32 cache, which
 * shares the walk but not the scale reads; a chunk of one cell, which maximises
 * how often the combine runs against how much the walk does; and a selection whose
 * scores rise monotonically, the one input that forces the online maximum to
 * rescale the accumulator at every cell instead of about ln(nsel) times. */
static int check_attn_fused(void) {
    enum { HD = 256, PER = 12, NKVH = 2, NQH = PER * NKVH, NCTX = 300, KROW = NKVH * HD,
           MAXSEL = 137, MAXUNITS = 16 };
    static const float kscale = 0.25f, vscale = 0.125f;
    const float scale = 1.0f / sqrtf((float)HD);
    struct shape { int nsel, chunk, f32cache, ramp; };
    static const struct shape shapes[] = {
        { 1, 32, 0, 0 },   { 96, 32, 0, 0 },  { 137, 32, 0, 0 },
        { 137, 32, 1, 0 }, { 137, 1, 0, 0 },  { 64, 8, 0, 1 },
    };
    static float q[NQH * HD], py[MAXUNITS * PER * HD], pm[MAXUNITS * PER], pl[MAXUNITS * PER];
    static float got[NQH * HD], old[NQH * HD];
    static double ref[NQH * HD], sc[MAXSEL], acc[HD];
    int sel[NCTX];
    int bad = 0;
    double worst_fused = 0.0, worst_split = 0.0, worst_pair = 0.0;
    long checks = 0;

    int8_t *kd = malloc((size_t)NCTX * KROW);
    int8_t *vd = malloc((size_t)NCTX * KROW);
    float *kc = malloc((size_t)NCTX * KROW * sizeof(float));
    float *vc = malloc((size_t)NCTX * KROW * sizeof(float));
    float *ks = malloc((size_t)NCTX * NKVH * sizeof(float));
    float *vs = malloc((size_t)NCTX * NKVH * sizeof(float));
    if (!kd || !vd || !kc || !vc || !ks || !vs) {
        printf("%-19s: FAIL (out of memory)\n", "fused attention");
        free(kd); free(vd); free(kc); free(vc); free(ks); free(vs);
        return 1;
    }

    uint64_t s = 0x243f6a8885a308d3ull;
    for (long i = 0; i < (long)NQH * HD; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        q[i] = (float)((double)((s >> 11) & 0xffffffu) / 8388608.0 - 1.0);
    }
    for (long c = 0; c < NCTX; c++)
        for (int g = 0; g < NKVH; g++)
            for (int d = 0; d < HD; d++) {
                const long o = c * KROW + (long)g * HD + d;
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                const int kv = (int)((s >> 17) % 201) - 100;
                const int vv = (int)((s >> 40) % 201) - 100;
                kd[o] = (int8_t)kv;
                vd[o] = (int8_t)vv;
                kc[o] = (float)kv * kscale;
                vc[o] = (float)vv * vscale;
            }
    for (int i = 0; i < NCTX * NKVH; i++) {
        ks[i] = kscale;
        vs[i] = vscale;
    }

    for (unsigned si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const struct shape sh = shapes[si];
        const int8_t *kuse = sh.f32cache ? NULL : kd;
        const int8_t *vuse = sh.f32cache ? NULL : vd;
        const float *kfcache = sh.f32cache ? kc : NULL;
        const float *vfcache = sh.f32cache ? vc : NULL;
        if (sh.ramp) {
            /* Scores that only ever rise: the online maximum rescales on every
             * cell, which is the branch's worst case and its cheapest to force. */
            for (long i = 0; i < (long)NQH * HD; i++)
                q[i] = 0.0625f;
            for (long c = 0; c < NCTX; c++)
                for (int g = 0; g < NKVH; g++)
                    for (int d = 0; d < HD; d++) {
                        const long o = c * KROW + (long)g * HD + d;
                        const int v = 1 + (int)(c % 30);
                        kd[o] = (int8_t)v;
                        kc[o] = (float)v * kscale;
                    }
        }
        for (int i = 0; i < sh.nsel; i++)
            sel[i] = 2 * i;
        const int nchunks = (sh.nsel + sh.chunk - 1) / sh.chunk;
        if (nchunks > MAXUNITS / NKVH)
            continue;

        for (int h = 0; h < NQH; h++) {
            const int g = h / PER;
            const float *qq = q + (long)h * HD;
            double m = -1e300, sum = 0.0;
            /* Reverse order, so the reference is not secretly the same sum. */
            for (int i = 0; i < sh.nsel; i++) {
                const long cell = sel[sh.nsel - 1 - i];
                const long o = cell * KROW + (long)g * HD;
                double d = 0.0;
                for (int j = 0; j < HD; j++)
                    d += sh.f32cache ? (double)qq[j] * (double)kfcache[o + j]
                                     : (double)qq[j] * (double)(int)kuse[o + j];
                if (!sh.f32cache)
                    d *= kscale;
                sc[i] = d * scale;
                if (sc[i] > m)
                    m = sc[i];
            }
            for (int i = 0; i < sh.nsel; i++) {
                sc[i] = exp(sc[i] - m);
                sum += sc[i];
            }
            for (int j = 0; j < HD; j++)
                acc[j] = 0.0;
            for (int i = 0; i < sh.nsel; i++) {
                const long cell = sel[sh.nsel - 1 - i];
                const long o = cell * KROW + (long)g * HD;
                const double w = sc[i] / sum;
                for (int j = 0; j < HD; j++)
                    acc[j] += w * (sh.f32cache ? (double)vfcache[o + j]
                                               : (double)(int)vuse[o + j] * vscale);
            }
            for (int j = 0; j < HD; j++)
                ref[(long)h * HD + j] = acc[j];
        }

        for (int g = 0; g < NKVH; g++)
            for (int c = 0; c < nchunks; c++) {
                const int u = g * nchunks + c;
                const long first = (long)c * sh.chunk;
                const long left = (long)sh.nsel - first;
                qe_attn_unit au = { .q = q + (long)g * PER * HD,
                                    .kd = kuse, .vd = vuse, .kc = kfcache, .vc = vfcache,
                                    .ks = sh.f32cache ? NULL : ks,
                                    .vs = sh.f32cache ? NULL : vs,
                                    .sel = sel + first,
                                    .y = py + (long)u * PER * HD,
                                    .m = pm + (long)u * PER,
                                    .l = pl + (long)u * PER,
                                    .nsel = (int)(left < sh.chunk ? left : sh.chunk),
                                    .per = PER, .hd = HD, .krow = KROW, .kv_head = g,
                                    .sstride = NKVH, .q8 = sh.f32cache ? 0 : 1,
                                    .scale = scale };
                qe_attn_walk(&au);
            }
        for (int h = 0; h < NQH; h++) {
            const int g = h / PER, hl = h % PER;
            const long base = (long)g * nchunks * PER;
            qe_attn_combine(py + (base + hl) * HD, (long)PER * HD, pm + base + hl,
                            pl + base + hl, PER, nchunks, HD, got + (long)h * HD);
        }

        for (int h = 0; h < NQH; h++) {
            const int g = h / PER;
            const float *qq = q + (long)h * HD;
            float *oy = old + (long)h * HD;
            float m2 = -1e30f, sum = 0.0f;
            double peak = 0.0;
            for (int i = 0; i < sh.nsel; i++) {
                const long cell = sel[i];
                const long o = cell * KROW + (long)g * HD;
                float d;
                if (sh.f32cache)
                    d = qe_dot(qq, kfcache + o, HD);
                else
                    d = qe_dot_q8(qq, kuse + o, HD) * ks[(long)cell * NKVH + g];
                d *= scale;
                sc[i] = d;
                if (d > m2)
                    m2 = d;
            }
            for (int i = 0; i < sh.nsel; i++) {
                sc[i] = expf(sc[i] - m2);
                sum += sc[i];
            }
            const float inv = 1.0f / sum;
            for (int j = 0; j < HD; j++)
                oy[j] = 0.0f;
            for (int i = 0; i < sh.nsel; i++) {
                const long cell = sel[i];
                const long o = cell * KROW + (long)g * HD;
                if (sh.f32cache)
                    qe_axpy(oy, vfcache + o, (float)sc[i] * inv, HD);
                else
                    qe_axpy_q8(oy, vuse + o, (float)sc[i] * inv * vs[(long)cell * NKVH + g], HD);
            }
            for (int j = 0; j < HD; j++)
                if (fabs(ref[(long)h * HD + j]) > peak)
                    peak = fabs(ref[(long)h * HD + j]);
            for (int j = 0; j < HD; j++) {
                const double w = peak > 1e-6 ? peak : 1e-6;
                const double ef = fabs((double)got[(long)h * HD + j] - ref[(long)h * HD + j]) / w;
                const double eo = fabs((double)old[(long)h * HD + j] - ref[(long)h * HD + j]) / w;
                const double ep = fabs((double)got[(long)h * HD + j] -
                                       (double)old[(long)h * HD + j]) / w;
                if (!(got[(long)h * HD + j] == got[(long)h * HD + j]))
                    bad++;
                if (ef > worst_fused)
                    worst_fused = ef;
                if (eo > worst_split)
                    worst_split = eo;
                if (ep > worst_pair)
                    worst_pair = ep;
                checks++;
            }
        }
    }

    /* A chunk that saw no cells has to stay out of the answer rather than contribute
     * a maximum of its own, and a head whose every chunk was empty reads as zeros. */
    {
        float m0 = 0.0f, l0 = 0.0f, y0[HD];
        for (int j = 0; j < HD; j++)
            y0[j] = 1.0f;
        qe_attn_unit au = { .q = q, .kd = kd, .vd = vd, .ks = ks, .vs = vs, .sel = sel,
                            .y = y0, .m = &m0, .l = &l0, .nsel = 0, .per = 1, .hd = HD,
                            .krow = KROW, .kv_head = 0, .sstride = NKVH, .q8 = 1,
                            .scale = scale };
        qe_attn_walk(&au);
        float out[HD];
        qe_attn_combine(y0, HD, &m0, &l0, 1, 1, HD, out);
        for (int j = 0; j < HD; j++)
            if (out[j] != 0.0f)
                bad++;
    }

    /* The two walks' cost, one thread, the shipped q8_0 cache, at the sizes a real
     * selection takes. The one-pass column walks ascending; the head-at-a-time
     * column walks in a random order, because a heap returned the winning blocks
     * in heap order and that is the order the form it replaces used. Per QSA layer
     * with all twenty-four query heads in it, so twelve of these is a token. */
    {
        enum { CELLS = 2 * 8192, NSHAPE = 3 };
        static const int shnsel[NSHAPE] = { 512, 2048, 8192 };
        int8_t *kd2 = malloc((size_t)CELLS * KROW);
        int8_t *vd2 = malloc((size_t)CELLS * KROW);
        float *ks2 = malloc((size_t)CELLS * NKVH * sizeof(float));
        float *vs2 = malloc((size_t)CELLS * NKVH * sizeof(float));
        int *sel_a = malloc((size_t)8192 * sizeof(int));
        int *sel_h = malloc((size_t)8192 * sizeof(int));
        float *sco = malloc((size_t)8192 * sizeof(float));
        double tf[NSHAPE], th[NSHAPE];
        if (kd2 && vd2 && ks2 && vs2 && sel_a && sel_h && sco) {
            uint64_t r = 0x9e3779b97f4a7c15ull;
            for (long i = 0; i < (long)CELLS * KROW; i++) {
                r ^= r >> 12; r ^= r << 25; r ^= r >> 27;
                kd2[i] = (int8_t)((int)((r >> 17) % 201) - 100);
                r ^= r >> 12; r ^= r << 25; r ^= r >> 27;
                vd2[i] = (int8_t)((int)((r >> 17) % 201) - 100);
            }
            for (long i = 0; i < (long)CELLS * NKVH; i++) {
                ks2[i] = kscale;
                vs2[i] = vscale;
            }
            printf("\n");
            printf("  %-22s", "walk, one thread");
            for (unsigned ci = 0; ci < NSHAPE; ci++)
                printf("%10d", shnsel[ci]);
            printf("  cells per layer\n");
            for (unsigned ci = 0; ci < NSHAPE; ci++) {
                const int nsel = shnsel[ci];
                for (int i = 0; i < nsel; i++)
                    sel_a[i] = sel_h[i] = 2 * i;
                /* A heap's output is a random tour of the arena, and that is the
                 * baseline being compared against rather than a sorted one. */
                for (int i = nsel - 1; i > 0; i--) {
                    r ^= r >> 12; r ^= r << 25; r ^= r >> 27;
                    const int j = (int)(r % (uint64_t)(i + 1));
                    const int t = sel_h[i];
                    sel_h[i] = sel_h[j];
                    sel_h[j] = t;
                }
                tf[ci] = th[ci] = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    double t0 = now();
                    for (int g = 0; g < NKVH; g++) {
                        qe_attn_unit au = { .q = q + (long)g * PER * HD, .kd = kd2, .vd = vd2,
                                            .ks = ks2, .vs = vs2, .sel = sel_a,
                                            .y = py + (long)g * PER * HD, .m = pm + (long)g * PER,
                                            .l = pl + (long)g * PER, .nsel = nsel, .per = PER,
                                            .hd = HD, .krow = KROW, .kv_head = g, .sstride = NKVH,
                                            .q8 = 1, .scale = scale };
                        qe_attn_walk(&au);
                    }
                    double d = now() - t0;
                    if (d < tf[ci])
                        tf[ci] = d;
                    t0 = now();
                    for (int h = 0; h < NQH; h++) {
                        const int g = h / PER;
                        const float *qq = q + (long)h * HD;
                        float *oy = old + (long)h * HD;
                        float m2 = -1e30f;
                        for (int i = 0; i < nsel; i++) {
                            const long o = (long)sel_h[i] * KROW + (long)g * HD;
                            const float d2 = qe_dot_q8(qq, kd2 + o, HD) * ks2[(long)sel_h[i] * NKVH + g];
                            sco[i] = d2 * scale;
                            if (d2 * scale > m2)
                                m2 = d2 * scale;
                        }
                        for (int i = 0; i < nsel; i++)
                            sco[i] = expf(sco[i] - m2);
                        for (int j = 0; j < HD; j++)
                            oy[j] = 0.0f;
                        for (int i = 0; i < nsel; i++) {
                            const long o = (long)sel_h[i] * KROW + (long)g * HD;
                            qe_axpy_q8(oy, vd2 + o,
                                       sco[i] * vs2[(long)sel_h[i] * NKVH + g], HD);
                        }
                    }
                    d = now() - t0;
                    if (d < th[ci])
                        th[ci] = d;
                }
            }
            printf("  %-22s", "one pass per group");
            for (unsigned ci = 0; ci < NSHAPE; ci++)
                printf("%10.1f", tf[ci] * 1e6);
            printf("  us per layer\n");
            printf("  %-22s", "one pass per head");
            for (unsigned ci = 0; ci < NSHAPE; ci++)
                printf("%10.1f", th[ci] * 1e6);
            printf("  us per layer\n");
            printf("  %-22s", "per token, 12 layers");
            for (unsigned ci = 0; ci < NSHAPE; ci++)
                printf("%9.2f", tf[ci] * 1e3 * 12.0);
            printf("  ms, against");
            for (unsigned ci = 0; ci < NSHAPE; ci++)
                printf(" %6.2f", th[ci] * 1e3 * 12.0);
            printf(" ms\n");
        }
        free(kd2); free(vd2); free(ks2); free(vs2);
        free(sel_a); free(sel_h); free(sco);
    }

    if (worst_fused > 2e-5 || worst_pair > 2e-5 || worst_split > 2e-5)
        bad++;
    printf("%-19s: %s (fused %.1e, head-at-a-time %.1e, between them %.1e, %ld checks)\n",
           "fused attention", bad ? "FAIL" : "pass", worst_fused, worst_split, worst_pair, checks);
    free(kd); free(vd); free(kc); free(vc); free(ks); free(vs);
    return bad;
}

/* Scale linearity, which needs no codebook and is exactly the property the bf16
 * shift broke. Row 0 and row 1 of the synthetic buffer get the same bytes and
 * one clean power-of-two scale in every block; row 1's scales are then doubled,
 * and its dot must be exactly twice row 0's. Power-of-two scaling is exact in
 * binary floating point, so any other factor is a misread scale, not rounding.
 * Negating the sign bit is the same argument. */
static int check_scale_linearity(const uint8_t *W, const float *x, long row_bytes, long nb,
                                 long block_sz, long scale_off, long scale_off2, gemv_fn gemv,
                                 long K, const char *name) {
    if (scale_off < 0)
        return 0;
    uint8_t *r0 = (uint8_t *) W;
    uint8_t *r1 = (uint8_t *) W + row_bytes;
    /* Q4_K's dot is linear in the pair (d, min) together and in neither alone,
     * so a second field that scales with the first has to be written with it
     * for the factor of two to be the whole claim. */
#define QE_SCALE_AT(r, b, v)                                                                 \
    do {                                                                                     \
        memcpy((r) + (b) * block_sz + scale_off, &(v), 2);                                   \
        if (scale_off2 >= 0)                                                                 \
            memcpy((r) + (b) * block_sz + scale_off2, &(v), 2);                              \
    } while (0)
    for (long b = 0; b < nb; b++) {
        uint16_t one = 0x3C00u; /* 1.0 */
        QE_SCALE_AT(r0, b, one);
    }
    memcpy(r1, r0, (size_t) row_bytes);
    for (long b = 0; b < nb; b++) {
        uint16_t two = 0x4000u; /* 2.0 */
        QE_SCALE_AT(r1, b, two);
    }
    float yd[2] = { 0, 0 };
    gemv(W, x, yd, 2, K, nb);

    memcpy(r1, r0, (size_t) row_bytes);
    for (long b = 0; b < nb; b++) {
        uint16_t neg = 0xBC00u; /* -1.0 */
        QE_SCALE_AT(r1, b, neg);
    }
    float yn[2] = { 0, 0 };
    gemv(W, x, yn, 2, K, nb);

    int bad = 0;
    if (yd[0] == 0.0f) {
        printf("  FAIL %s: row summed to zero, the scale check is vacuous\n", name);
        return 1;
    }
    if (yd[1] != 2.0f * yd[0]) {
        printf("  FAIL %s: doubling every block scale scaled the dot by %.9g, not 2\n", name,
               (double) (yd[1] / yd[0]));
        bad++;
    }
    if (yn[0] != yd[0] || yn[1] != -yd[0]) {
        printf("  FAIL %s: negating every block scale did not negate the dot\n", name);
        bad++;
    }
#undef QE_SCALE_AT
    printf("%-19s: %s\n", "scale linearity", bad ? "FAIL" : "pass");
    return bad;
}

/* bf16 owns no block format, so it appears in no KINDS row and none of the
   checks above ever ran on it, while the checkpoint's indexer projections and
   any narrowed router are all bf16. What it needs is the check a dropped lane
   hides: a kernel that widens four of eight lanes returns half a dot product,
   finite, plausible, and 2x off. */
static int check_bf16_gemv(void) {
    extern double qe_dot_bf16_ref(const uint16_t *w, const float *x, long n);
    static float W[512], x[512], y[2];
    static uint16_t Wb[512];
    int bad = 0;
    const long KS[] = { 7, 8, 15, 16, 17, 24, 31, 64, 127, 512 };
    for (unsigned ki = 0; ki < sizeof KS / sizeof KS[0]; ki++) {
        long K = KS[ki];
        for (long i = 0; i < K; i++) {
            W[i] = (float)sin(i * 0.017) * 0.5f + 0.001f * (float)((i % 9) - 4);
            x[i] = (float)cos(i * 0.011);
        }
        qe_bf16_from_f32(W, Wb, (size_t)K);
        y[0] = 0;
        qe_gemv_bf16(Wb, x, y, 1, K, K);
        double ref = qe_dot_bf16_ref(Wb, x, K);
        double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
        if (!(fabs(ref - y[0]) / den < 1e-6)) {
            printf("  FAIL bf16 K=%ld: got %.9g, reference %.9g\n", K, y[0], ref);
            bad++;
        }
        /* Ones in, ones out: an exact count of the terms, so a load that only
         * filled part of a vector cannot pass. */
        for (long i = 0; i < K; i++) {
            W[i] = 1.0f;
            x[i] = 1.0f;
        }
        qe_bf16_from_f32(W, Wb, (size_t)K);
        y[0] = 0;
        qe_gemv_bf16(Wb, x, y, 1, K, K);
        if (y[0] != (float)K) {
            printf("  FAIL bf16 K=%ld: %d terms of 1.0 summed to %.1f\n", K, (int)K, y[0]);
            bad++;
        }
    }
    /* Rounding, not truncation: a half-ulp-low value has to go up, and the
     * worst per-weight error has to stay under half an ulp. */
    float half_ulp = 1.0f + (float)(0.75 * 0.0078125);
    uint16_t r;
    qe_bf16_from_f32(&half_ulp, &r, 1);
    if (r != 0x3F81u) {
        printf("  FAIL bf16 round-to-nearest: %.9g became 0x%04X, not 0x3F81\n",
               (double)half_ulp, r);
        bad++;
    }
    double worst = 0;
    for (int i = 0; i < 512; i++) {
        float v = (float)sin(i * 1.7) * 3.0f;
        qe_bf16_from_f32(&v, &r, 1);
        union {
            uint32_t u;
            float f;
        } g;
        g.u = (uint32_t)r << 16;
        if (fabs(v) > 1e-4f) {
            double e = fabs(g.f - v) / fabs(v);
            if (e > worst)
                worst = e;
        }
    }
    if (!(worst <= 0.0078125 / 2)) {
        printf("  FAIL bf16 conversion: worst relative error %.6f exceeds half an ulp\n", worst);
        bad++;
    }
    printf("%-19s: %s\n", "bf16 dot + rounding", bad ? "FAIL" : "pass");
    return bad;
}

/* The elementwise primitives: a reference in double for each, one hand-computed
 * value whose answer needs no library to state, and the per-token cost of the
 * 96 of them a hyper-connected layer pair costs, because none of that work
 * appears in a byte budget. */
static int check_layer_ops(void) {
    int bad = 0;
    double worst_sig = 0, worst_rms = 0, worst_mix = 0;

    /* 1. sigmoid, over the range a gate can be in and past both ends of it. */
    float prev = -1.0f;
    for (long i = 0; i <= 8000; i++) {
        float x = (float)(i - 4000) * 0.01f;
        float got;
        qe_sigmoid(&x, &got, 1);
        double ref = 1.0 / (1.0 + exp(-(double)x));
        double e = fabs(ref - (double)got);
        if (e > worst_sig)
            worst_sig = e;
        if (!(e < 1e-6)) {
            printf("  FAIL sigmoid(%g): got %.9g, reference %.17g\n", x, got, ref);
            bad++;
            break;
        }
        /* Monotone, on a grid that is actually ordered: an approximation with a
         * non-monotone stretch is legal under every error bound above and turns
         * a sigmoid gate into a source of ties the router then breaks at random. */
        if (i && !(got >= prev)) {
            printf("  FAIL sigmoid is not monotonic at %g: %.9g after %.9g\n", x, got, prev);
            bad++;
            break;
        }
        prev = got;
    }
    {
        float ext[] = { 0.0f, 1.0f, -1.0f, 20.0f, -20.0f, 88.0f, -88.0f, 1e20f, -1e20f };
        float out[9];
        qe_sigmoid(ext, out, 9);
        for (unsigned i = 0; i < 9; i++)
            if (!(out[i] >= 0.0f && out[i] <= 1.0f) || out[i] != out[i]) {
                printf("  FAIL sigmoid(%g) = %g is not a probability\n", ext[i], out[i]);
                bad++;
            }
        if (!(out[0] == 0.5f)) {
            printf("  FAIL sigmoid(0) = %.9g, not 0.5 exactly\n", out[0]);
            bad++;
        }
    }

    /* 2. rms_norm: the answer for [3,4] is 1/sqrt(12.5) * [3,4], stated here in
     * constants rather than recomputed, so a wrong denominator cannot pass by
     * being wrong in both places. */
    {
        const float x[2] = { 3.0f, 4.0f }, g[2] = { 1.0f, 1.0f };
        float o[2];
        qe_rms_norm_grouped(x, g, 2, o, 2, 1, 0.0f);
        if (!(fabs(o[0] - 0.8485281374238570f) < 1e-6f && fabs(o[1] - 1.1313708498984761f) < 1e-6f)) {
            printf("  FAIL rms_norm([3,4]) = [%.9g, %.9g], expected [0.848528, 1.131371]\n", o[0],
                   o[1]);
            bad++;
        }
    }
    /* Grouped means grouped: reducing over the concatenation instead of per
     * stream is the mistake this catches, and it is the one a 10240-wide
     * hc_*_norm.weight invites. */
    {
        enum { W = 2560, S = 4 };
        static float a[W * S], b[W * S], g[W * S];
        for (long i = 0; i < W * S; i++)
            a[i] = (float)sin(i * 0.0007) * (1.0f + 0.5f * (float)((i / W) % 3));
        for (long i = 0; i < W; i++)
            g[i] = 1.0f + 0.25f * (float)sin(i * 0.01);
        memcpy(b, a, sizeof b);
        for (long i = 0; i < W; i++)
            b[i] *= 7.0f; /* only stream 0 */
        qe_rms_norm_grouped(a, g, W, a, W, S, 1e-6f);
        qe_rms_norm_grouped(b, g, W, b, W, S, 1e-6f);
        double dmax = 0;
        for (long i = W; i < W * S; i++)
            if (fabs((double)a[i] - (double)b[i]) > dmax)
                dmax = fabs((double)a[i] - (double)b[i]);
        if (!(dmax == 0.0)) {
            printf("  FAIL rms_norm groups are not independent: other streams moved by %.3g\n",
                   dmax);
            bad++;
        }
    }
    {
        enum { W = 128, S = 48 };
        static float x[W * S], g[W], o[W * S], o2[W * S];
        double ss[S];
        for (int s2 = 0; s2 < S; s2++) {
            ss[s2] = 0;
            for (long i = 0; i < W; i++) {
                x[(long)s2 * W + i] = (float)cos(i * 0.03 + s2);
                ss[s2] += (double)x[(long)s2 * W + i] * (double)x[(long)s2 * W + i];
            }
        }
        for (long i = 0; i < W; i++)
            g[i] = 0.5f + (float)sin(i * 0.1);
        /* gamma_stride 0 is the shared-gamma shape ssm_norm needs. */
        qe_rms_norm_grouped(x, g, 0, o, W, S, 1e-6f);
        qe_rms_norm_grouped(x, g, 0, o2, W, S, 1e-6f);
        for (int s2 = 0; s2 < S; s2++)
            for (long i = 0; i < W; i++) {
                long k = (long)s2 * W + i;
                double ref = (double)x[k] / sqrt(ss[s2] / (double)W + 1e-6) * (double)g[i];
                double e = fabs(ref - (double)o[k]) / (fabs(ref) + 1e-6);
                if (e > worst_rms)
                    worst_rms = e;
                if (!(e < 2e-6)) {
                    printf("  FAIL rms_norm shared gamma [%d,%ld]: got %.9g, reference %.9g\n", s2,
                           i, o[k], ref);
                    bad++;
                    s2 = S;
                    break;
                }
            }
        if (memcmp(o, o2, sizeof o) != 0) {
            printf("  FAIL rms_norm is not reproducible on identical input\n");
            bad++;
        }
    }

    /* 3. the mixer, and the exactness the 2*sigmoid scaling promises: zero
     * injection has to be an ordinary residual add. */
    {
        enum { W = 2560, S = 4 };
        static float xn[W * S], gate[W * S], mixed[W];
        for (long i = 0; i < W * S; i++) {
            xn[i] = (float)sin(i * 0.0011);
            gate[i] = 3.0f * (float)cos(i * 0.0007);
        }
        qe_stream_gate_mean(xn, gate, mixed, W, S);
        for (long j = 0; j < W; j++) {
            double acc = 0, scale = 0;
            for (int s2 = 0; s2 < S; s2++) {
                long k = (long)s2 * W + j;
                double term = (double)xn[k] * (1.0 / (1.0 + exp(-(double)gate[k])));
                acc += term;
                scale += fabs(term);
            }
            double ref = acc / (double)S;
            /* Against the size of the terms, not of the answer: four gated streams
             * average to something much smaller than they are, and dividing the
             * unavoidable rounding by a result that happens to cancel decides the
             * verdict by which elements happened to cancel. Same mistake, same fix
             * as the grouped-expert check in bench_moe. */
            double e = fabs(ref - (double)mixed[j]) / (scale / (double)S + 1e-6);
            if (e > worst_mix)
                worst_mix = e;
            if (!(e < 2e-6)) {
                printf("  FAIL gate mean at %ld: got %.9g, reference %.9g\n", j, mixed[j], ref);
                bad++;
                break;
            }
        }
        float inj[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static float res[W * S], y[W], want[W * S];
        for (long i = 0; i < W * S; i++)
            res[i] = (float)cos(i * 0.0013);
        for (long i = 0; i < W; i++)
            y[i] = (float)sin(i * 0.0017);
        memcpy(want, res, sizeof want);
        for (long i = 0; i < W * S; i++)
            want[i] = want[i] + y[i % W];
        qe_stream_inject_add(res, y, inj, W, S);
        if (memcmp(res, want, sizeof want) != 0) {
            printf("  FAIL a zero injection is not an ordinary residual add\n");
            bad++;
        }
    }

    /* What 96 of each cost, at the shapes the checkpoint uses. These are the
     * numbers a byte budget does not contain: activation traffic and exp work,
     * once per hyper-connected block, 96 of them per token. */
    {
        enum { W = 2560, S = 4, N = 96 };
        static float xn[W * S], gate[W * S], g[W * S], mixed[W], res[W * S], y[W], inj[4];
        for (long i = 0; i < W * S; i++) {
            xn[i] = (float)sin(i * 0.0011);
            gate[i] = 3.0f * (float)cos(i * 0.0007);
            g[i] = 1.0f + 0.1f * (float)sin(i * 0.001);
            res[i] = xn[i];
        }
        for (long i = 0; i < W; i++)
            y[i] = (float)sin(i * 0.0017);
        for (int i = 0; i < 4; i++)
            inj[i] = 0.25f * (float)i;
        double t0 = now();
        for (int r = 0; r < N; r++)
            qe_rms_norm_grouped(xn, g, W, res, W, S, 1e-6f);
        double t_norm = now() - t0;
        t0 = now();
        for (int r = 0; r < N; r++)
            qe_stream_gate_mean(xn, gate, mixed, W, S);
        double t_mix = now() - t0;
        t0 = now();
        for (int r = 0; r < N; r++)
            qe_stream_inject_add(res, y, inj, W, S);
        double t_inj = now() - t0;
        /* The same 96 gates through libm one element at a time, which is what a
         * straightforward implementation of this model does. */
        t0 = now();
        float sink = 0.0f;
        for (int r = 0; r < N; r++)
            for (long i = 0; i < W * S; i++)
                sink += 1.0f / (1.0f + expf(-gate[i]));
        double t_libm = now() - t0;
        /* Accumulated, not stored to a fixed element: a store that the compiler
         * can see is overwritten every iteration measures a loop that only ran
         * once, and it did exactly that here before it was made observable. */
        if (sink == 12345.0f)
            printf("  unreachable\n");
        double per = t_norm + t_mix + t_inj;
        printf("%-19s: %s\n", "layer elementwise", bad ? "FAIL" : "pass");
        printf("  %-19s   sigmoid worst %.1e, rms %.1e, mixer %.1e against a double reference\n",
               "", worst_sig, worst_rms, worst_mix);
        printf("  %-19s   96 of each per token: norms %.2f ms + gate %.2f ms + inject %.2f ms"
               " = %.2f ms\n",
               "", t_norm * 1e3, t_mix * 1e3, t_inj * 1e3, per * 1e3);
        printf("  %-19s   %.1f MiB of activation traffic per token; sigmoid in SIMD is %.1fx the"
               " libm loop (%.1f ms)\n",
               "", 96.0 * (3.0 * W * S + W) * 4.0 / 1048576.0, t_libm > 0 ? t_libm / (t_mix > 0 ? t_mix : 1e-9) : 0.0,
               t_libm * 1e3);
    }
    return bad;
}

int main(int argc, char **argv) {
    qe_fuse(argc > 1 ? atoi(argv[1]) : 60);
    int nthr = argc > 2 ? atoi(argv[2]) : 16;

    const long K = 2560;
    const size_t MB = 384;

    uint8_t *W = qe_alloc_huge(MB << 20);
    if (!W) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    float *x = aligned_alloc(64, (size_t)K * 4);
    float *y = aligned_alloc(64, 8u << 20);
    for (long i = 0; i < K; i++)
        x[i] = (float)sin(i * 0.0017) * 0.5f;

    size_t n = MB << 20;
    uint64_t s = 0x243f6a8885a308d3ull;
    for (size_t i = 0; i < n / 8; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        ((uint64_t *)W)[i] = s * 0x2545f4914f6cdd1dull;
    }

    qe_pool_init(nthr, NULL, 0);

    int interp_bad = check_half_conversion() + check_half_roundtrip() + check_q8_anchor() +
                     check_q8_multi() + check_iq_multi() + check_iq4_nl_quant() +
                     check_q6_requant() +
                     check_i8_walk(0, 0) + check_i8_walk(1, 0) + check_i8_walk(0, 1) +
                     check_i8_walk(1, 1) +
                     check_bf16_gemv() + check_layer_ops() +
                     check_kv_q8() + check_attn_fused();

    printf("K=%ld, %zu MiB per type, THP-backed, %d threads max\n\n", K, MB, nthr);
    printf("%-7s %8s %8s %8s %9s  %-9s %s\n", "type", "1 thr", "4 thr",
           "20 thr", "worst rel", "verdict", "ms/token at peak rate");

    double budget_ms = 0;
    for (unsigned ki = 0; ki < sizeof(KINDS) / sizeof(KINDS[0]); ki++) {
        const kind *k = &KINDS[ki];
        const long nb = K / k->block_ne;
        const long ldd = nb;
        const long rows = (long)((n / (size_t)nb) / (size_t)k->block_sz);
        const long row_bytes = nb * k->block_sz;

        /* Keep every fp16 super-scale finite and away from zero. */
        if (k->scale_off >= 0) {
            for (long off = k->scale_off; off + row_bytes <= (long)n; off += row_bytes)
                for (long b = 0; b < nb; b++) {
                    uint16_t h = 0x3800u | (uint16_t)((off / 34 + b * 7) & 0x3ff);
                    memcpy(W + off + b * k->block_sz, &h, 2);
                    if (k->scale_off2 >= 0)
                        memcpy(W + off + b * k->block_sz + k->scale_off2, &h, 2);
                }
        }

        /* ---- correctness against the scalar reference ---- */
        float probe[8];
        k->gemv(W, x, probe, 8, K, ldd);
        double worst = 0;
        for (int r = 0; r < 8; r++) {
            const void *row = W + (size_t)r * (size_t)row_bytes;
            double ref = k->dot(row, x, nb);
            double den = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
            double rel = fabs(ref - probe[r]) / den;
            if (!(rel == rel)) { /* NaN means the reference itself blew up */
                worst = 1e30;
                break;
            }
            if (rel > worst) worst = rel;
        }

        /* Runs after the reference comparison so the two never race for rows
         * 0 and 1, and rewrites those rows' scale fields. */
        interp_bad += check_scale_linearity(W, x, row_bytes, nb, k->block_sz, k->scale_off,
                                            k->scale_off2, k->gemv, K, k->name);

        /* ---- throughput ---- */
        gemv_job j = { W, x, y, rows, K, ldd, row_bytes, k->gemv };
        double gb[3] = { 0, 0, 0 };
        const int tcand[3] = { 1, 4, nthr };
        for (int c = 0; c < 3; c++) {
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                double a = now();
                qe_pool_run(gemv_task, &j, tcand[c]);
                double d = now() - a;
                if (d < best) best = d;
            }
            gb[c] = (double)rows * row_bytes / best / 1e9;
        }
        double share = k->budget_mib / STEP_MIB;
        double ms = k->budget_mib * 1048576.0 / (gb[2] * 1e9) * 1e3;
        budget_ms += ms;
        printf("%-7s %8.1f %8.1f %8.1f %9.1e  %-9s", k->name, gb[0], gb[1], gb[2], worst,
               worst < 1e-4 ? "PASS" : "FAIL");
        if (k->budget_mib > 0)
            printf(" %.0f%% of step = %.1f ms\n", share * 100.0, ms);
        else
            printf(" (A/B entry, not counted in the total)\n");
        if (gb[2] > 120.0)
            printf("        !! %.0f GB/s is more than the fastest linear read this host has"
                   " done (85.5 GB/s, make alu) by a third: the working set is"
                   " cache-resident, this is not a bandwidth number\n",
                   gb[2]);
        fflush(stdout);
    }

    double counted = 0;
    for (size_t ki2 = 0; ki2 < sizeof(KINDS) / sizeof(KINDS[0]); ki2++)
        counted += KINDS[ki2].budget_mib;
    double other_mib = STEP_MIB - counted;
    /* The rest of the step has no SIMD dot to benchmark: 240 MiB of F32 router
     * logits, 59 MiB of F32 GDN state and 38 MiB of BF16 indexer projections,
     * 6% of the step between them. They move at stream rate, and the stream
     * rate here is 78.1 GB/s - measured by bin/bench_gemv's ceiling check, not
     * re-measured here, so a constant with a source rather than a result. */
    const double STREAM_GBS = 78.1;
    double ms_other = other_mib * 1048576.0 / (STREAM_GBS * 1e9) * 1e3;
    double total_ms = budget_ms + ms_other;

    printf("\nsum of per-type times = %.1f ms/token -> %.1f tok/s single stream\n", budget_ms,
           1e3 / budget_ms);
    printf("+ %.1f MiB of F32 router/state and BF16 indexer at %.1f GB/s = +%.1f ms\n",
           other_mib, STREAM_GBS, ms_other);
    printf("whole step             = %.1f ms/token -> %.1f tok/s single stream\n",
           total_ms, 1e3 / total_ms);
    /* Same draft model as the target budget: 1.91 GB (1821 MiB) shared-Q4_K_M
     * head, 2.76 tokens per step at greedy, so the step cost per emitted token
     * is (step + draft x tok) / tok. */
    const double DRAFT_MIB = 1821.0, DRAFT_TOK = 2.76;
    double mtp_ms = total_ms * ((STEP_MIB + DRAFT_MIB * DRAFT_TOK) / DRAFT_TOK) / STEP_MIB;
    printf("with MTP shared-Q4_K_M @%.2f tok/step: %.1f ms/token -> %.1f tok/s\n",
           DRAFT_TOK, mtp_ms, 1e3 / mtp_ms);

    printf("\ninterpretation checks: %s\n", interp_bad ? "FAIL" : "PASS");
    qe_pool_shutdown();
    qe_free_huge(W, n);
    free(x);
    free(y);
    return interp_bad ? 1 : 0;
}
