/* The PLE n-gram row gather: docs/ARCH.md 6.2 for the arithmetic this
 * implements and docs/DESIGN.md 1.1 for why every byte below is a memory access
 * rather than a read.
 *
 * A token's layer-1 embedding is sixteen 90-byte rows of a 26.8 GiB IQ4_NL
 * table, selected by hashing the token's own id with its one and two
 * predecessors. 1440 bytes is nothing against the 5568 MiB a step reads, so the
 * only question this file has to answer is latency: sixteen unrelated lines in
 * sixteen distant pages, each costing a page-table walk and a fill that misses
 * every prefetcher because the ids are integers, not addresses. The answers, in
 * the order they matter:
 *
 *   - the staging arena is 2 MiB-backed, which is what removes most of the walk;
 *   - the sixteen ids are sorted and deduplicated before any of them is
 *     dereferenced, so the memory system sees an increasing stream and a token
 *     that repeats an id (the same bigram twice in a window) pays once;
 *   - issue()/collect() split the fetch from the decode so the caller can start
 *     it at token-id time and consume it before layer 1;
 *   - a two-way row cache holds rows between tokens, keyed by row id. Rows are
 *     kept encoded - 90 bytes, not the 640 a decoded row costs - because a hit
 *     is worth exactly the page walk it avoids, and dequantising 160 elements is
 *     not what this is limited by. A million cached rows are then 109 MB.
 */
#include "qe.h"
#include "kern.h"
#include "tables.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define PLE_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#define PLE_PREFETCH(p) ((void)0)
#endif

#define PLE_MAX_HEADS 16
#define PLE_MAX_NGRAM 8
/* One entry per token a chunk can hold: the queue is fed one entry per token of the
 * chunk, and an entry the queue declines is a cold gather. Depth is therefore the
 * chunk width and not a tuning knob, because a queue shallower than the chunk makes
 * the tokens at its end pay for their own sixteen page walks, which is a measurement
 * of the queue rather than of the width. */
#define PLE_PENDING QE_CHUNK_MAX
#define PLE_DEFAULT_CACHE (1ll << 20)
#define PLE_SLOT_BYTES 96 /* the row is 90; the padding is a rounded slot */

typedef struct {
    uint64_t key; /* row id + 1; 0 means the slot was never filled */
    uint8_t b[PLE_SLOT_BYTES];
} ple_slot;

struct qe_ple {
    const uint8_t *tab;
    long long n_rows;
    int n_heads, head_dim;
    int ngram_size, heads_per_ngram;
    size_t row_bytes;
    uint64_t mult[PLE_MAX_NGRAM];
    long long off[PLE_MAX_HEADS], vsz[PLE_MAX_HEADS];
    uint32_t eos;

    ple_slot *sets;  /* n_sets ways of two */
    uint8_t *victim; /* one bit per set, in a byte */
    size_t n_sets;
    size_t set_mask; /* n_sets - 1; n_sets is a power of two by construction */

    struct {
        long long rows[PLE_MAX_HEADS];
    } pend[PLE_PENDING];
    int p_head, p_tail, p_len;

    qe_ple_stats st;
};

/* ------------------------------------------------------------------ geometry */

int qe_ple_heads(const qe_ple *p) { return p->n_heads; }
int qe_ple_head_dim(const qe_ple *p) { return p->head_dim; }
int qe_ple_width(const qe_ple *p) { return p->n_heads * p->head_dim; }
size_t qe_ple_row_bytes(const qe_ple *p) { return p->row_bytes; }
long long qe_ple_table_rows(const qe_ple *p) { return p->n_rows; }
uint32_t qe_ple_eos_token(const qe_ple *p) { return p->eos; }

void qe_ple_get_stats(const qe_ple *p, qe_ple_stats *out) { *out = p->st; }
void qe_ple_stats_clear(qe_ple *p) { memset(&p->st, 0, sizeof p->st); }

/* The PLE keys carry the architecture prefix, which for this checkpoint is the
 * architecture, but a file that says otherwise should still load. */
static int get_int(qe_model *m, const char *arch, const char *leaf, long long *out) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, leaf);
    if (qe_model_get_int(m, key, out) == 0)
        return 0;
    snprintf(key, sizeof key, "qwen4exp.%s", leaf);
    return qe_model_get_int(m, key, out);
}

static int get_arr(qe_model *m, const char *arch, const char *leaf,
                   const long long **out, size_t *n) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, leaf);
    if (qe_model_get_arr_int(m, key, out, n) == 0)
        return 0;
    snprintf(key, sizeof key, "qwen4exp.%s", leaf);
    return qe_model_get_arr_int(m, key, out, n);
}

qe_ple *qe_ple_create(qe_model *m, const qe_ple_desc *d, char *err, size_t errlen) {
    if (err && errlen)
        err[0] = 0;
    const char *tname = (d && d->tensor && *d->tensor) ? d->tensor : "per_layer_token_embd.weight";
    long long cache_rows = (d && d->cache_rows > 0) ? d->cache_rows : PLE_DEFAULT_CACHE;

    char arch[64] = "qwen4exp";
    const char *a = NULL;
    if (qe_model_get_str(m, "general.architecture", &a) == 0 && a && *a)
        snprintf(arch, sizeof arch, "%s", a);

    const qe_tensor *t = qe_tensor_by_name(m, tname);
    if (!t) {
        snprintf(err, errlen, "no %s in this checkpoint", tname);
        return NULL;
    }
    if (t->type != QE_IQ4_NL) {
        snprintf(err, errlen, "%s is not IQ4_NL (type %d)", tname, (int)t->type);
        return NULL;
    }
    const void *base = qe_tensor_data(m, t);
    if (!base) {
        snprintf(err, errlen, "%s is not resident", tname);
        return NULL;
    }
    if (t->ne[0] % 32) {
        snprintf(err, errlen, "%s has %llu elements per row, not a whole number of blocks", tname,
                 (unsigned long long)t->ne[0]);
        return NULL;
    }

    qe_ple *p = calloc(1, sizeof *p);
    if (!p) {
        snprintf(err, errlen, "out of memory");
        return NULL;
    }
    p->tab = (const uint8_t *)base;
    p->n_rows = (long long)t->ne[1];
    p->head_dim = (int)t->ne[0];
    p->row_bytes = (size_t)(t->ne[0] / 32) * sizeof(qe_block_iq4_nl);

    long long ngram = 3, hpn = 8, eos = 0;
    get_int(m, arch, "ple.ngram_size", &ngram);
    get_int(m, arch, "ple.heads_per_ngram", &hpn);
    get_int(m, arch, "ple.eos_token_id", &eos);
    if (ngram < 2 || ngram > PLE_MAX_NGRAM || hpn < 1 || hpn > PLE_MAX_HEADS) {
        snprintf(err, errlen, "ple.ngram_size=%lld heads_per_ngram=%lld is not representable",
                 ngram, hpn);
        free(p);
        return NULL;
    }
    p->ngram_size = (int)ngram;
    p->heads_per_ngram = (int)hpn;
    p->eos = (uint32_t)eos;

    const long long *mult = NULL, *off = NULL, *vsz = NULL;
    size_t nm = 0, no = 0, nv = 0;
    if (get_arr(m, arch, "ple.layer_multipliers", &mult, &nm)) {
        snprintf(err, errlen, "no ple.layer_multipliers");
        free(p);
        return NULL;
    }
    if (get_arr(m, arch, "ple.head_offsets", &off, &no)) {
        snprintf(err, errlen, "no ple.head_offsets");
        free(p);
        return NULL;
    }
    if (get_arr(m, arch, "ple.head_vocab_sizes", &vsz, &nv)) {
        snprintf(err, errlen, "no ple.head_vocab_sizes");
        free(p);
        return NULL;
    }
    if (nm < (size_t)ngram) {
        snprintf(err, errlen, "ple.layer_multipliers has %zu of %lld entries", nm, ngram);
        free(p);
        return NULL;
    }
    /* One group of heads_per_ngram heads per n-gram width from 2 to ngram_size. */
    if (no != nv || no != (size_t)(ngram - 1) * (size_t)hpn) {
        snprintf(err, errlen, "ple head tables describe %zu heads, not (%lld-1)*%lld", no, ngram,
                 hpn);
        free(p);
        return NULL;
    }
    p->n_heads = (int)no;
    for (size_t i = 0; i < nm && i < PLE_MAX_NGRAM; i++)
        p->mult[i] = (uint64_t)mult[i];
    for (int h = 0; h < p->n_heads; h++) {
        p->off[h] = off[h];
        p->vsz[h] = vsz[h];
        if (vsz[h] <= 0 || off[h] < 0 || off[h] + vsz[h] > p->n_rows) {
            snprintf(err, errlen, "ple head %d wants rows %lld..%lld of %lld", h, off[h],
                     off[h] + vsz[h] - 1, p->n_rows);
            free(p);
            return NULL;
        }
    }

    size_t want = (size_t)cache_rows;
    /* Rounding up by doubling is what makes n_sets a power of two, which is
       what lets the set index below be a mask: a divide would sit on every
       row the cache misses on and every row it fills. */
    p->n_sets = 1;
    while (p->n_sets * 2 < want)
        p->n_sets <<= 1;
    p->set_mask = p->n_sets - 1;
    p->sets = calloc(p->n_sets * 2, sizeof *p->sets);
    p->victim = calloc(p->n_sets, 1);
    if (!p->sets || !p->victim) {
        snprintf(err, errlen, "out of memory for a %zu-row cache", p->n_sets * 2);
        qe_ple_destroy(p);
        return NULL;
    }
    return p;
}

void qe_ple_destroy(qe_ple *p) {
    if (!p)
        return;
    free(p->sets);
    free(p->victim);
    free(p);
}

/* -------------------------------------------------------------- row selection */

/* The window rule the file states: a predecessor EOS replaces itself and
 * everything older than it, and a position before the start of the sequence
 * reads as EOS. The token's own EOS is not a cut, hence `back` starts at 1. */
static uint32_t ple_ctx(const qe_ple *p, const uint32_t *seq, size_t i, size_t back) {
    if (back > i)
        return p->eos;
    for (size_t k = 1; k <= back; k++)
        if (seq[i - k] == p->eos)
            return p->eos;
    return seq[i - back];
}

void qe_ple_rows_at(const qe_ple *p, const uint32_t *seq, size_t i, long long *rows) {
    uint64_t mixed[PLE_MAX_NGRAM] = { 0 };
    for (int n = 1; n <= p->ngram_size; n++) {
        uint64_t acc = 0;
        for (int j = 0; j < n; j++)
            acc ^= (uint64_t)ple_ctx(p, seq, i, (size_t)j) * p->mult[j];
        mixed[n - 1] = acc;
    }
    for (int h = 0; h < p->n_heads; h++) {
        /* heads 0..hpn-1 hash the bigram, the next group the trigram, and so on
         * up to ngram_size, which is the width the file's own n-gram key gives. */
        int n = 2 + h / p->heads_per_ngram;
        if (n > p->ngram_size)
            n = p->ngram_size;
        rows[h] = (long long)(mixed[n - 1] % (uint64_t)p->vsz[h]) + p->off[h];
    }
}

/* ---------------------------------------------------------------- fetch path */

static void ple_decode(const uint8_t *row, int blocks, float *out) {
    const qe_block_iq4_nl *b = (const qe_block_iq4_nl *)row;
    for (int i = 0; i < blocks; i++) {
        float d = qe_half_to_f32(b[i].d);
        for (int j = 0; j < 16; j++) {
            uint8_t q = b[i].qs[j];
            out[i * 32 + j] = d * (float)(int8_t)qe_kvalues_iq4nl[q & 0xf];
            out[i * 32 + 16 + j] = d * (float)(int8_t)qe_kvalues_iq4nl[q >> 4];
        }
    }
}

static size_t ple_set_of(uint64_t row) {
    uint64_t h = (row + 1) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 32;
    return (size_t)h;
}

static void ple_fetch(qe_ple *p, long long row, float *dst) {
    uint64_t key = (uint64_t)row + 1;
    size_t s = ple_set_of((uint64_t)row) & p->set_mask;
    ple_slot *w = p->sets + 2 * s;
    int hit = w[0].key == key ? 0 : (w[1].key == key ? 1 : -1);
    const uint8_t *bytes;
    if (hit >= 0) {
        p->st.cache_hits++;
        bytes = w[hit].b;
    } else {
        bytes = p->tab + (size_t)row * p->row_bytes;
        int v = p->victim[s];
        memcpy(w[v].b, bytes, p->row_bytes);
        if (p->row_bytes < PLE_SLOT_BYTES)
            memset(w[v].b + p->row_bytes, 0, PLE_SLOT_BYTES - p->row_bytes);
        w[v].key = key;
        p->victim[s] = (uint8_t)(v ^ 1);
    }
    ple_decode(bytes, p->head_dim / 32, dst);
}

int qe_ple_gather(qe_ple *p, const long long *rows, float *emb) {
    int order[PLE_MAX_HEADS];
    for (int h = 0; h < p->n_heads; h++)
        order[h] = h;
    /* Insertion sort: sixteen elements, already partly ordered between
     * neighbouring tokens, and the point is only that memory is touched in
     * increasing order so the duplicates end up adjacent. */
    for (int k = 1; k < p->n_heads; k++) {
        int key = order[k];
        int j = k - 1;
        while (j >= 0 && rows[order[j]] > rows[key]) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }
    for (int k = 0; k < p->n_heads;) {
        long long r = rows[order[k]];
        if (r < 0 || r >= p->n_rows)
            return -1;
        float *dst = emb + (size_t)order[k] * p->head_dim;
        ple_fetch(p, r, dst);
        p->st.distinct++;
        p->st.rows++;
        k++;
        while (k < p->n_heads && rows[order[k]] == r) {
            memcpy(emb + (size_t)order[k] * p->head_dim, dst, (size_t)p->head_dim * sizeof *dst);
            p->st.rows++;
            k++;
        }
    }
    return 0;
}

int qe_ple_issue(qe_ple *p, const uint32_t *seq, size_t i) {
    if (p->p_len >= PLE_PENDING)
        return -1;
    long long *rows = p->pend[p->p_tail].rows;
    qe_ple_rows_at(p, seq, i, rows);
    for (int h = 0; h < p->n_heads; h++) {
        const uint8_t *q = p->tab + (size_t)rows[h] * p->row_bytes;
        /* Two lines: a 90-byte row straddles two cache lines unless it lands on
         * a 32-byte boundary, which it does not by construction. */
        PLE_PREFETCH(q);
        PLE_PREFETCH(q + 64);
    }
    p->p_tail = (p->p_tail + 1) % PLE_PENDING;
    p->p_len++;
    return 0;
}

int qe_ple_collect(qe_ple *p, float *emb) {
    if (!p->p_len)
        return -1;
    int rc = qe_ple_gather(p, p->pend[p->p_head].rows, emb);
    p->p_head = (p->p_head + 1) % PLE_PENDING;
    p->p_len--;
    return rc;
}

/* The rows a pending entry holds were computed from a window of the sequence, so
 * they belong to the token they were issued for and to nothing else. Dropping the
 * queue costs one cold gather; keeping it across a rewind pairs a token with
 * another token's rows. */
void qe_ple_drain(qe_ple *p) {
    if (!p)
        return;
    p->p_head = p->p_tail = p->p_len = 0;
}
