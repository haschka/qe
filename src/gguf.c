/* GGUF v3 reader for split (multi-volume) checkpoints.
 *
 * Two properties of this checkpoint drive the design:
 *
 * 1. It is split. Only the first volume carries the metadata; the remaining
 *    volumes carry their own tensor-info records with offsets that are local
 *    to that volume. Nothing is guessed: after parsing, the loader verifies
 *    that the byte ranges implied by the type table tile each volume's data
 *    section with the declared alignment. If a type entry were wrong the chain
 *    would break and loading fails loudly instead of silently reading garbage.
 *
 * 2. Page-table overhead, not instructions, dominates. The checkpoint is
 *    87 GiB and 31% of it is one per-layer embedding table read as ~16
 *    scattered rows per token. At 4 KiB pages that is millions of PTEs and a
 *    page walk per random access. Huge-page backing removes that.
 */
#define _GNU_SOURCE
#include "qe.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#define QE_ALIGN_DEFAULT 32

enum { KV_NONE, KV_STR, KV_INT, KV_REAL, KV_ARR_STR, KV_ARR_INT, KV_ARR_REAL };

struct kv {
    char key[QE_MAX_NAME];
    int kind;
    long long i;
    double r;
    char *s;
    size_t n_arr;
    void *arr;
};

struct qe_model {
    qe_shard sh[QE_MAX_SHARDS];
    int nshards;

    qe_tensor *t;
    size_t n_tensors, cap_tensors;

    struct kv *kv;
    size_t n_kv, cap_kv;

    char *arena;
    size_t arena_len, arena_used;

    /* How the volume bytes got here. On this host the difference between the
     * two answers is four orders of magnitude per expert read, so it belongs to
     * the model, not to a command line flag. How long the device took to hand
     * the bytes over is recorded nowhere: it is a one-time cost, it is not a
     * quantity this engine optimises, and publishing it turns every startup
     * into a storage benchmark nobody asked for. */
    char stage_method[16];
    size_t stage_bytes, stage_cached;
    int stage_pinned;
};

static char errbuf[512] = "ok";
const char *qe_strerror(void) { return errbuf; }
static void qe_seterr(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(errbuf, sizeof(errbuf), fmt, ap);
    va_end(ap);
}
#define QE_ERR(...)              \
    do {                         \
        qe_seterr(__VA_ARGS__);  \
        goto fail;               \
    } while (0)

/* ------------------------------------------------------------ type table */

/* Recovered from this checkpoint's own offset chain (tools/dtype_derive.py:
 * 0 mismatches, implied size == file size) and cross-checked against each
 * format's documented bits-per-weight. Only ids 0,8,14,20,21,23,30 occur in
 * this file; the rest of the canonical table is filled in so the loader can
 * ingest sibling quantisations. A wrong entry here silently misreads every
 * tensor, which is why it is derived rather than assumed.
 *
 * Q5_0 was wrong until an MTP sidecar file - the first here with a Q5_0 row -
 * tripped the chain check three times, each with the same 204,800 byte gap
 * after a [320x10240] tensor: 102400 blocks of 32, two bytes a block short,
 * which is the fp16 delta of a format that is d + high bits + quants. 22, not
 * 20: 5.5 bpw, and its Q5_1 sibling on the same line was already carrying both
 * of its halves. */
static const qe_type_info TI[QE_TYPE_COUNT] = {
    [QE_F32] = {1, 4, "F32"},          [QE_F16] = {1, 2, "F16"},
    [QE_Q4_0] = {32, 16, "Q4_0"},      [QE_Q4_1] = {32, 20, "Q4_1"},
    [QE_Q5_0] = {32, 22, "Q5_0"},      [QE_Q5_1] = {32, 24, "Q5_1"},
    [QE_Q8_0] = {32, 34, "Q8_0"},      [QE_Q2_K] = {256, 84, "Q2_K"},
    [QE_Q3_K] = {256, 110, "Q3_K"},    [QE_Q4_K] = {256, 144, "Q4_K"},
    [QE_Q5_K] = {256, 176, "Q5_K"},    [QE_Q6_K] = {256, 210, "Q6_K"},
    [QE_IQ2_XXS] = {256, 66, "IQ2_XXS"}, [QE_IQ2_XS] = {256, 82, "IQ2_XS"},
    [QE_IQ3_XXS] = {256, 110, "IQ3_XXS"}, [QE_IQ1_S] = {256, 50, "IQ1_S"},
    [QE_IQ4_NL] = {32, 18, "IQ4_NL"},   [QE_IQ3_S] = {256, 110, "IQ3_S"},
    [QE_IQ2_S] = {256, 66, "IQ2_S"},    [QE_IQ4_XS] = {256, 136, "IQ4_XS"},
    [QE_IQ1_M] = {256, 66, "IQ1_M"},    [QE_BF16] = {1, 2, "BF16"},
};

const qe_type_info *qe_type_info_of(qe_type t) {
    if ((unsigned)t >= QE_TYPE_COUNT || !TI[t].name)
        return NULL;
    return &TI[t];
}

long qe_type_bpw_x1000(qe_type t) {
    const qe_type_info *i = qe_type_info_of(t);
    return i ? (long)i->block_sz * 8000 / (long)i->block_ne : -1;
}

/* ------------------------------------------------------------ header reader */

typedef struct {
    const uint8_t *p;
    const uint8_t *e;
    int bad;
} rd;

static uint64_t r_u64(rd *r) {
    if ((size_t)(r->e - r->p) < 8) { r->bad = 1; return 0; }
    uint64_t v;
    memcpy(&v, r->p, 8);
    r->p += 8;
    return v;
}
static uint32_t r_u32(rd *r) {
    if ((size_t)(r->e - r->p) < 4) { r->bad = 1; return 0; }
    uint32_t v;
    memcpy(&v, r->p, 4);
    r->p += 4;
    return v;
}
static const char *r_str(rd *r, size_t *len) {
    uint64_t n = r_u64(r);
    if ((uint64_t)(r->e - r->p) < n) { r->bad = 1; *len = 0; return NULL; }
    const char *s = (const char *)r->p;
    r->p += n;
    *len = (size_t)n;
    return s;
}
static char *r_strdup(const char *s, size_t n) {
    char *c = malloc(n + 1);
    if (!c) return NULL;
    memcpy(c, s, n);
    c[n] = 0;
    return c;
}

static void kv_put(qe_model *m, const char *key, size_t klen, struct kv *v) {
    if (m->n_kv == m->cap_kv) {
        m->cap_kv = m->cap_kv ? m->cap_kv * 2 : 256;
        m->kv = realloc(m->kv, m->cap_kv * sizeof(*m->kv));
    }
    struct kv *k = &m->kv[m->n_kv++];
    *k = *v;
    size_t n = klen < QE_MAX_NAME - 1 ? klen : QE_MAX_NAME - 1;
    memcpy(k->key, key, n);
    k->key[n] = 0;
}

static int parse_metadata(qe_model *m, rd *r, long *alignment) {
    uint64_t n_kv = r_u64(r);
    for (uint64_t i = 0; i < n_kv && !r->bad; i++) {
        size_t klen;
        const char *key = r_str(r, &klen);
        if (r->bad) return -1;
        uint32_t t = r_u32(r);
        struct kv v;
        memset(&v, 0, sizeof(v));
        switch (t) {
        case 0: v.kind = KV_INT; v.i = (int8_t)r->p[0]; r->p += 1; break;
        case 1: v.kind = KV_INT; v.i = (int8_t)r->p[0]; r->p += 1; break;
        case 2: v.kind = KV_INT; memcpy(&v.i, r->p, 2); r->p += 2; break;
        case 3: { int16_t x; memcpy(&x, r->p, 2); v.kind = KV_INT; v.i = x; r->p += 2; break; }
        case 4: v.kind = KV_INT; memcpy(&v.i, r->p, 4); r->p += 4; break;
        case 5: { int32_t x; memcpy(&x, r->p, 4); v.kind = KV_INT; v.i = x; r->p += 4; break; }
        case 6: { float x; memcpy(&x, r->p, 4); v.kind = KV_REAL; v.r = x; r->p += 4; break; }
        case 7: v.kind = KV_INT; v.i = r->p[0]; r->p += 1; break;
        case 10: v.kind = KV_INT; memcpy(&v.i, r->p, 8); r->p += 8; break;
        case 11: { int64_t x; memcpy(&x, r->p, 8); v.kind = KV_INT; v.i = x; r->p += 8; break; }
        case 12: v.kind = KV_REAL; memcpy(&v.r, r->p, 8); r->p += 8; break;
        case 8: {
            size_t len;
            const char *s = r_str(r, &len);
            if (r->bad) return -1;
            v.kind = KV_STR;
            v.s = r_strdup(s, len);
            break;
        }
        case 9: {
            uint32_t et = r_u32(r);
            uint64_t n = r_u64(r);
            if (et == 8) {
                char **arr = n ? calloc(n, sizeof(char *)) : NULL;
                for (uint64_t j = 0; j < n && !r->bad; j++) {
                    size_t len;
                    const char *s = r_str(r, &len);
                    if (r->bad) { v.kind = KV_ARR_STR; v.arr = arr; v.n_arr = n; return -1; }
                    arr[j] = r->bad ? NULL : r_strdup(s, len);
                }
                v.kind = KV_ARR_STR;
                v.arr = arr;
                v.n_arr = n;
            } else {
                size_t esz;
                int is_real = 0, is64 = 0;
                switch (et) {
                case 0: case 1: case 7: esz = 1; break;
                case 2: case 3: esz = 2; break;
                case 4: case 5: esz = 4; break;
                case 6: esz = 4; is_real = 1; break;
                case 10: case 11: esz = 8; is64 = 1; break;
                case 12: esz = 8; is_real = 1; break;
                default: r->bad = 1; return -1;
                }
                if ((size_t)(r->e - r->p) < n * esz) { r->bad = 1; return -1; }
                const uint8_t *src = r->p;
                r->p += n * esz;
                if (is_real) {
                    double *d = n ? malloc(n * sizeof(double)) : NULL;
                    for (uint64_t j = 0; j < n; j++)
                        d[j] = (esz == 4) ? (double)((const float *)src)[j]
                                          : ((const double *)src)[j];
                    v.kind = KV_ARR_REAL;
                    v.arr = d;
                } else {
                    long long *d = n ? malloc(n * sizeof(long long)) : NULL;
                    for (uint64_t j = 0; j < n; j++) {
                        if (esz == 1) d[j] = (int8_t)src[j];
                        else if (esz == 2) d[j] = (int16_t)((const uint16_t *)src)[j];
                        else if (esz == 4) d[j] = (int32_t)((const uint32_t *)src)[j];
                        else d[j] = (int64_t)((const uint64_t *)src)[j];
                    }
                    (void)is64;
                    v.kind = KV_ARR_INT;
                    v.arr = d;
                }
                v.n_arr = n;
            }
            break;
        }
        default: r->bad = 1; return -1;
        }
        if (r->bad)
            return -1;
        if (strcmp(key, "general.alignment") == 0 && v.kind == KV_INT)
            *alignment = v.i;
        /* Later volumes repeat nothing, but guard against a duplicate key. */
        int dup = 0;
        for (size_t z = 0; z < m->n_kv; z++)
            if (strcmp(m->kv[z].key, key) == 0) { dup = 1; break; }
        if (!dup)
            kv_put(m, key, klen, &v);
        else {
            free(v.s);
            free(v.arr);
        }
    }
    return r->bad ? -1 : 0;
}

static int cmp_off(const void *a, const void *b) {
    const qe_tensor *x = *(const qe_tensor *const *)a;
    const qe_tensor *y = *(const qe_tensor *const *)b;
    return (x->off > y->off) - (x->off < y->off);
}

/* ---------------------------------------------------------------- staging
 *
 * QE_STAGE_RAM and QE_STAGE_HUGE both mean "read the whole volume into private
 * memory at open and never touch the disk again". QE_STAGE_HUGE additionally
 * puts the arena in a hugetlbfs file, which buys two things: 2 MiB pages
 * (23 thousand page tables instead of 22 million) and an arena that outlives the
 * process, so the 87 GiB sequential read costs one load per boot rather than
 * one per restart.
 *
 * Reuse of an existing arena is only allowed when the index next to it says the
 * source file is unchanged (size, mtime, inode) and a fingerprint of every 4 MiB
 * block still matches, so a rebuilt checkpoint can never be served from a stale
 * arena.
 */

#define STAGE_CHUNK (4u << 20)
#define STAGE_PROBE 1024 /* bytes fingerprinted per chunk */

struct stage_hdr {
    char magic[8];    /* "QEARN" + version */
    uint32_t vol;     /* volume number this arena holds */
    uint64_t file_bytes;
    uint64_t mtime_ns;
    uint64_t ino;
    uint64_t chunk;
    uint32_t n_chunk;
    uint32_t reserved;
};

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void volume_path(char *out, size_t n, const char *stem, int v, int total, int width) {
    if (total == 1)
        snprintf(out, n, "%s", stem);
    else if (width > 8)
        snprintf(out, n, "%s-%d-of-%d.gguf", stem, v, total);
    else
        snprintf(out, n, "%s-%0*u-of-%0*u.gguf", stem, width, v, width, total);
}

static const char *stage_dir(void) {
    const char *d = getenv("QE_HUGEPAGE_DIR");
    return (d && *d) ? d : NULL;
}

/* A directory whose files are RAM and outlive the process: tmpfs, or hugetlbfs
 * under --stage huge. With QE_STAGE_DIR set the checkpoint is read once into a
 * file there and later processes map the same pages, which turns one load per
 * process into one load per machine. Those are ordinary page-cache pages: the
 * machine can still reclaim them, which is the property a reserved huge-page
 * pool gives up. */
static const char *arena_dir(void) {
    const char *d = getenv("QE_STAGE_DIR");
    return (d && *d) ? d : NULL;
}

/* What DIR can still give a new file. Writing past the end of a full tmpfs file
 * through a shared mapping is SIGBUS rather than an error return, so the room
 * has to be confirmed before the first byte is written. */
static size_t arena_free_bytes(const char *dir) {
    struct statvfs sv;
    if (statvfs(dir, &sv) != 0)
        return 0;
    return (size_t)sv.f_bavail * (size_t)sv.f_frsize;
}

/* Kept out of the arena for KV cache, activations, the worker stacks and
 * whatever else the process grows. 6 GiB is generous for both: a 16-sequence
 * 8k batch is under 8 GiB of state, and this host has 128 GiB. */
static size_t stage_reserve_mib(void) {
    const char *e = getenv("QE_RESERVE_MIB");
    if (e && *e)
        return (size_t)strtoul(e, NULL, 10);
    return 6144;
}

static int stage_quiet(void) {
    const char *e = getenv("QE_QUIET");
    return e && *e && strcmp(e, "0") != 0;
}

static void stage_say(const char *fmt, ...) {
    if (stage_quiet())
        return;
    va_list ap;
    va_start(ap, fmt);
    fputs("[qe] ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static uint64_t stage_fnv(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Arena name inside the hugetlbfs mount, and the index file next to it. The
 * index has to live outside the arena: the arena is the model's bytes and
 * nothing else, so it can be handed straight to the kernels. */
static void stage_names(const char *dir, const char *stem, int v, char *name, size_t nn,
                        char *idx, size_t ni) {
    const char *base = strrchr(stem, '/');
    base = base ? base + 1 : stem;
    snprintf(name, nn, "qe-%s-v%d.2m", base, v);
    snprintf(idx, ni, "%s/qe-%s-v%d.2m.idx", dir, base, v);
}

/* One fingerprint per STAGE_CHUNK bytes of the volume. Cheap enough to recompute
 * on every start (a few tens of megabytes touched) and tight enough that a
 * rewritten checkpoint cannot be mistaken for the arena it invalidated. */
static void stage_fingerprint(const uint8_t *p, size_t n, uint64_t *out, uint32_t nchunk) {
    for (uint32_t i = 0; i < nchunk; i++) {
        size_t at = (size_t)i * STAGE_CHUNK;
        size_t len = (at + STAGE_PROBE <= n) ? STAGE_PROBE : (n > at ? n - at : 0);
        out[i] = stage_fnv(p + at, len);
    }
}

static int stage_cache_valid(const char *idxpath, const struct stage_hdr *want,
                             const uint8_t *arena, size_t n) {
    FILE *f = fopen(idxpath, "rb");
    if (!f)
        return 0;
    struct stage_hdr h;
    int ok = fread(&h, sizeof(h), 1, f) == 1 && !memcmp(h.magic, "QEARN\1\0\0", 8) &&
             h.vol == want->vol && h.file_bytes == want->file_bytes &&
             h.mtime_ns == want->mtime_ns && h.ino == want->ino && h.chunk == STAGE_CHUNK &&
             (size_t)h.n_chunk * STAGE_CHUNK <= n * 2;
    uint64_t *got = NULL;
    if (ok) {
        got = malloc((size_t)h.n_chunk * sizeof(*got));
        ok = got != NULL && fread(got, sizeof(*got), h.n_chunk, f) == h.n_chunk;
    }
    fclose(f);
    if (ok) {
        uint32_t nchunk = h.n_chunk;
        /* Compare in chunks so a mismatch in the middle costs nothing. */
        ok = 0;
        for (uint32_t i = 0; i < nchunk; i++) {
            size_t at = (size_t)i * STAGE_CHUNK;
            size_t len = (at + STAGE_PROBE <= n) ? STAGE_PROBE : (n > at ? n - at : 0);
            if (stage_fnv(arena + at, len) != got[i])
                goto done;
        }
        ok = 1;
    }
done:
    free(got);
    return ok;
}

static void stage_cache_store(const char *idxpath, const struct stage_hdr *h, const uint8_t *arena,
                              size_t n) {
    if (!h->n_chunk)
        return;
    uint64_t *fp = malloc((size_t)h->n_chunk * sizeof(*fp));
    if (!fp)
        return;
    stage_fingerprint(arena, n, fp, h->n_chunk);
    /* The temp name is the index path with ".tmp<pid>" on the end, so it has to be
     * able to hold that path in full. A fixed buffer truncates silently, and a
     * cache file written to a half-truncated name is one no later run will find,
     * which is a slow startup and nothing louder. */
    size_t need = strlen(idxpath) + 32;
    char *tmp = malloc(need);
    if (!tmp) {
        free(fp);
        return;
    }
    snprintf(tmp, need, "%s.tmp%d", idxpath, (int)getpid());
    FILE *f = fopen(tmp, "wb");
    if (f) {
        if (fwrite(h, sizeof(*h), 1, f) != 1 || fwrite(fp, sizeof(*fp), h->n_chunk, f) != h->n_chunk)
            remove(tmp);
        fclose(f);
        if (rename(tmp, idxpath) != 0)
            remove(tmp);
    }
    free(tmp);
    free(fp);
}

/* Sequential read of a whole volume into an arena. One stream, deliberately: a
 * raidz2 group does less work under several concurrent seek streams than under
 * one reader with readahead in front of it, and this loop is a reader with
 * readahead in front of it. What it does not do is report the device's rate: the
 * load happens once per process, and the only fact worth printing is whether the
 * bytes are resident when it finishes. */
static int stage_read_volume(int fd, uint8_t *dst, size_t n, const char *what,
                             unsigned long long total_bytes, size_t *out_got) {
    const size_t chunk = 8u << 20;
    double t0 = now_s(), last_print = t0;
    size_t got = 0;
#ifdef POSIX_FADV_SEQUENTIAL
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    for (;;) {
        ssize_t r = pread(fd, dst + got, chunk > n - got ? n - got : chunk, (off_t)got);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            qe_seterr("read %s at %zu: %s", what, got, strerror(errno));
            return -1;
        }
        if (r == 0)
            break;
        got += (size_t)r;
        if (got >= n)
            break;
        double tnow = now_s();
        if (tnow - last_print >= 5.0) {
            /* Progress in bytes only. Elapsed time here is a rate in disguise,
             * and a rate for this loop measures a spinning array. */
            stage_say("loading %s into RAM: %.1f / %.1f GiB", what,
                      (double)got / 1073741824.0, (double)n / 1073741824.0);
            last_print = tnow;
        }
    }
    if (got != n) {
        qe_seterr("%s: read %zu of %zu bytes - file truncated?", what, got, n);
        return -1;
    }
    (void)total_bytes;
    *out_got = got;
    return 0;
}

qe_model *qe_open(const char *path, unsigned flags) {
    qe_model *m = calloc(1, sizeof(*m));
    if (!m) {
        qe_seterr("out of memory");
        return NULL;
    }
    for (int i = 0; i < QE_MAX_SHARDS; i++)
        m->sh[i].fd = -1;

    /* Resolve sibling volumes from the first one's name. Split checkpoints are
     * named "<stem>-0000N-of-0000M.gguf" with a 4..8 digit field. */
    char first[4096];
    snprintf(first, sizeof(first), "%s", path);
    char stem[4096];
    int total = 1, width = 4;
    {
        const char *of = NULL, *pos = NULL;
        for (const char *q = first; *q; q++) {
            if (q[0] != '-' || q[1] < '0' || q[1] > '9')
                continue;
            const char *r = q + 1;
            int nd = 0;
            while (*r >= '0' && *r <= '9') {
                r++;
                nd++;
            }
            if (nd >= 1 && strncmp(r, "-of-", 4) == 0) {
                of = r + 4;
                pos = q;
                width = nd;
                break;
            }
        }
        if (of) {
            total = (int)strtol(of, NULL, 10);
            size_t plen = (size_t)(pos - first);
            if (plen >= sizeof(stem)) plen = sizeof(stem) - 1;
            memcpy(stem, first, plen);
            stem[plen] = 0;
            char *dot = strstr((char *)of, ".gguf");
            if (dot) *dot = 0;
        } else {
            snprintf(stem, sizeof(stem), "%s", first);
            total = 1;
        }
    }
    if (total > QE_MAX_SHARDS)
        QE_ERR("checkpoint claims %d volumes, rebuild with QE_MAX_SHARDS>=%d",
               total, total);

    /* Decide the backing before mapping anything: whether the checkpoint fits
     * in RAM is a question about the whole checkpoint, not about one volume,
     * and answering it per volume is how the last volume of a three-volume split
     * fails after 60 GiB have already been read. */
    unsigned stage_want = QE_STAGE_MODE(flags);
    int stage_strict = (flags & QE_STAGE_STRICT) != 0;
    /* Environment form of the same choice. It is only consulted when the caller
     * had no opinion (QE_STAGE_MODE == 0 == QE_STAGE_AUTO), so a caller that
     * asked for a backing cannot be overridden by a stray shell variable. */
    if (!stage_want) {
        const char *e = getenv("QE_STAGE");
        if (e && *e) {
            if (!strcmp(e, "auto"))
                stage_want = QE_STAGE_AUTO;
            else if (!strcmp(e, "ram"))
                stage_want = QE_STAGE_RAM;
            else if (!strcmp(e, "huge"))
                stage_want = QE_STAGE_HUGE;
            else if (!strcmp(e, "mmap"))
                stage_want = QE_STAGE_MMAP;
        }
    }
    /* 2 MiB transparent huge pages, on unless asked not to be. Measured here:
     * staging 87 GiB with a single sequential reader under defrag=madvise put
     * 68.4 GiB of the first ~66 GiB touched into AnonHugePages with no
     * compaction stall, and the accesses that hurt most later - 90-byte rows
     * scattered across a 26.8 GiB table - are page-table walks in full.
     * QE_THP=0 turns it off; nothing here needs root either way. */
    const char *thp_s = getenv("QE_THP");
    int thp = !(thp_s && (!strcmp(thp_s, "0") || !strcmp(thp_s, "never")));
    unsigned stage_flags =
        stage_want | (((flags & QE_STAGE_THP) || thp) ? QE_STAGE_THP : 0);
    size_t total_bytes = 0;
    for (int v = 1; v <= total; v++) {
        char vp[4096];
        volume_path(vp, sizeof(vp), stem, v, total, width);
        struct stat st;
        if (stat(vp, &st) == 0)
            total_bytes += (size_t)st.st_size;
    }
    size_t reserve_mib = stage_reserve_mib();
    size_t need_mib = qe_align_up(total_bytes, 1u << 20) >> 20;
    if (!stage_want) {
        if (stage_dir() && qe_huge_free_mib(2) >= need_mib + need_mib / 50)
            stage_want = QE_STAGE_HUGE;
        else if (qe_mem_available_mib() >= need_mib + reserve_mib)
            stage_want = QE_STAGE_RAM;
        else if (stage_strict)
            QE_ERR("cannot place %zu MiB of checkpoint in memory: MemAvailable is %zu MiB and"
                   " %zu MiB is kept back for state and clients. Free the memory or lower the"
                   " reserve: a demand-paged mapping of this checkpoint is one raidz2 seek per"
                   " expert row, which is not a configuration worth starting a server in",
                   need_mib, qe_mem_available_mib(), reserve_mib);
        else {
            stage_want = QE_STAGE_MMAP;
            stage_say("%zu MiB of checkpoint will be demand-paged from disk: %zu MiB available,"
                      " %zu MiB kept back. Expert reads cost a raidz2 seek each.",
                      need_mib, qe_mem_available_mib(), reserve_mib);
        }
    }
    if (stage_want == QE_STAGE_HUGE) {
        size_t free_mib = qe_huge_free_mib(2);
        if (free_mib < need_mib + need_mib / 50) {
            if (stage_strict)
                QE_ERR("hugetlb staging needs %zu MiB of 2 MiB pages, pool has %zu MiB free"
                       " (reserve %llu of them with vm.nr_hugepages)",
                       need_mib, free_mib,
                       (unsigned long long)(qe_align_up(total_bytes, 2u << 20) >> 21));
            stage_want = QE_STAGE_RAM;
            stage_say("hugetlb pool short (%zu of %zu MiB), staging in plain pages", free_mib,
                      need_mib);
        }
    }
    if (stage_want == QE_STAGE_RAM) {
        size_t avail_mib = qe_mem_available_mib();
        if (avail_mib < need_mib + reserve_mib) {
            if (stage_strict)
                QE_ERR("staging %zu MiB of checkpoint needs %zu MiB free plus %zu MiB reserve,"
                       " MemAvailable is %zu MiB",
                       need_mib, need_mib, reserve_mib, avail_mib);
            stage_want = QE_STAGE_MMAP;
            stage_say("only %zu MiB available for %zu MiB of checkpoint: demand-paging from disk"
                      " (expect milliseconds per expert row)", avail_mib, need_mib);
        }
    }

    for (int v = 1; v <= total; v++) {
        char vp[4096];
        volume_path(vp, sizeof(vp), stem, v, total, width);
        int fd = open(vp, O_RDONLY);
        if (fd < 0)
            QE_ERR("open %s: %s", vp, strerror(errno));
        qe_shard *s = &m->sh[m->nshards];
        s->path = strdup(vp);
        s->fd = fd;
        s->first = (v == 1);

        struct stat st;
        if (fstat(fd, &st) != 0)
            QE_ERR("stat %s: %s", vp, strerror(errno));
        s->file_len = (size_t)st.st_size;
        /* Parse the whole volume out of one mapping. A fixed-size header window
         * is a trap on this checkpoint: volume 1 is 10.4 MiB of which ~10 MiB is
         * metadata (248k-entry vocab plus scores), so an 8 MiB window truncates
         * the KV section and the parse fails at the page boundary. */
        size_t len = qe_align_up(s->file_len, (size_t)qe_page_size());
        void *h = NULL;
        size_t got_bytes = 0, cached_bytes = 0;

        if (stage_want != QE_STAGE_MMAP) {
            /* Fail on a wrong path in four microseconds rather than after three
             * minutes and 87 GiB of staging. */
            uint32_t head[6];
            ssize_t r = pread(fd, head, sizeof head, 0);
            if (r != (ssize_t)sizeof head || memcmp(head, "GGUF", 4) != 0) {
                qe_seterr("%s: not a GGUF file", vp);
                goto fail;
            }
        }

        size_t alen = len;
        char ipath[4352] = "";
        const char *stage_how = NULL;
        struct stage_hdr hdr;
        memset(&hdr, 0, sizeof hdr);
        memcpy(hdr.magic, "QEARN\1\0\0", 8);
        hdr.vol = (uint32_t)v;
        hdr.file_bytes = s->file_len;
        hdr.mtime_ns = (uint64_t)st.st_mtim.tv_sec * 1000000000ULL + (uint64_t)st.st_mtim.tv_nsec;
        hdr.ino = (uint64_t)st.st_ino;
        hdr.chunk = STAGE_CHUNK;
        hdr.n_chunk = (uint32_t)((s->file_len + STAGE_CHUNK - 1) / STAGE_CHUNK);

        if (stage_want == QE_STAGE_HUGE) {
            const char *dir = stage_dir();
            if (!dir) {
                stage_want = QE_STAGE_RAM;
                stage_say("QE_HUGEPAGE_DIR is not set, staging in plain anonymous pages");
            } else {
                char aname[512];
                stage_names(dir, stem, v, aname, sizeof aname, ipath, sizeof ipath);
                alen = qe_align_up(s->file_len, 2u << 20);
                int fresh = 1;
                h = qe_alloc_arena_file(dir, aname, alen, &fresh);
                if (!h) {
                    stage_say("no hugetlbfs arena for volume %d (%s): %s", v, aname,
                              strerror(errno));
                    stage_want = QE_STAGE_RAM;
                    ipath[0] = 0;
                } else if (fresh) {
                    /* hugetlbfs hands back zero pages only for a fresh file, and
                     * this one may be a leftover that lost its index. */
                    memset(h, 0, alen);
                } else if (stage_cache_valid(ipath, &hdr, (const uint8_t *)h, s->file_len)) {
                    cached_bytes = s->file_len;
                    hdr.n_chunk = 0; /* index still describes the bytes */
                    stage_say("volume %d: %.1f GiB already resident in hugetlbfs, no disk read", v,
                              (double)s->file_len / 1073741824.0);
                } else {
                    stage_say("volume %d: arena is stale, re-reading %.1f GiB", v,
                              (double)s->file_len / 1073741824.0);
                    memset(h, 0, alen);
                }
            }
        }

        if (!h && stage_want == QE_STAGE_RAM) {
            const char *dir = arena_dir();
            if (dir) {
                size_t free_b = arena_free_bytes(dir);
                if (free_b < s->file_len + (2u << 20)) {
                    stage_say("%s has %zu MiB free and volume %d needs %zu MiB: private pages, "
                              "so the next start loads it again",
                              dir, free_b >> 20, v, s->file_len >> 20);
                } else {
                    char aname[512];
                    stage_names(dir, stem, v, aname, sizeof aname, ipath, sizeof ipath);
                    alen = qe_align_up(s->file_len, 2u << 20);
                    int fresh = 1;
                    h = qe_alloc_arena_file(dir, aname, alen, &fresh);
                    if (!h) {
                        stage_say("no arena file in %s for volume %d (%s): private pages instead",
                                  dir, v, strerror(errno));
                        ipath[0] = 0;
                        alen = len;
                    } else {
                        stage_how = "ram+file";
                        if (alen > s->file_len)
                            memset((uint8_t *)h + s->file_len, 0, alen - s->file_len);
                        if (fresh) {
                            /* A new file is zero bytes; nothing to validate. */
                        } else if (stage_cache_valid(ipath, &hdr, (const uint8_t *)h,
                                                     s->file_len)) {
                            cached_bytes = s->file_len;
                            hdr.n_chunk = 0; /* the index still describes the bytes */
                            stage_say("volume %d: %.1f GiB already resident in %s, nothing to load",
                                      v, (double)s->file_len / 1073741824.0, dir);
                        } else {
                            stage_say("volume %d: the arena in %s is not this checkpoint, "
                                      "filling it with %.1f GiB",
                                      v, dir, (double)s->file_len / 1073741824.0);
                        }
                    }
                }
            }
        }

        if (!h && stage_want == QE_STAGE_RAM) {
            const char *how = "ram";
            h = qe_alloc_backed(len, stage_flags, &how);
            if (h)
                stage_how = how;
            if (!h) {
                if (stage_strict)
                    QE_ERR("cannot allocate a %zu byte staging arena for %s: %s", len, vp,
                           strerror(errno));
                stage_want = QE_STAGE_MMAP;
                stage_say("no memory for a staging arena (%s): demand-paging from disk",
                          strerror(errno));
            }
        }

        if (h && !cached_bytes) {
            char what[64];
            snprintf(what, sizeof what, "volume %d of %d", v, total);
            if (stage_read_volume(fd, (uint8_t *)h, s->file_len, what, total_bytes,
                                  &got_bytes) != 0) {
                munmap(h, alen);
                goto fail;
            }
            stage_say("volume %d of %d is in RAM: %.1f GiB (one pass; "
                      "nothing reads the file again)",
                      v, total, (double)got_bytes / 1073741824.0);
            /* Only now does the index describe what is in the arena. */
            if (ipath[0])
                stage_cache_store(ipath, &hdr, (const uint8_t *)h, s->file_len);
        }
        if (h && cached_bytes)
            got_bytes = cached_bytes;

        if (h && stage_want == QE_STAGE_HUGE) {
            /* hugetlb pages are not swappable and not reclaimable, so there is
             * nothing to pin; say so instead of claiming an mlock we did not do. */
            m->stage_pinned = 2;
        } else if (h) {
            int e = qe_lock_resident(h, alen);
            m->stage_pinned = e == 0 ? 1 : 0;
            if (e)
                stage_say("mlock(%zu MiB) failed: %s (raising RLIMIT_MEMLOCK fixes it)",
                          s->file_len >> 20, strerror(-e));
        }

        if (h) {
            /* The volume is in private memory and pinned. Nothing in this
             * process maps the file or reads through this descriptor again, and
             * closing it is what makes "we never touch the disk while serving"
             * a property of the code rather than a claim about it. */
            close(fd);
            s->fd = fd = -1;
        }

        if (!h) {
            h = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
            if (h == MAP_FAILED)
                QE_ERR("mmap %s (%zu bytes): %s", vp, len, strerror(errno));
            /* Deliberately no MADV_WILLNEED here, for the demand-paged path only:
             * asking the kernel to read ahead every volume turns a microsecond
             * benchmark into a ten-minute I/O stall. Pages come in on demand, and
             * qe_model_warm() is the explicit opt-in. */
        }
        s->map = h;
        s->map_len = len;
        s->staged = (stage_want != QE_STAGE_MMAP);
        m->stage_bytes += s->file_len;
        m->stage_cached += cached_bytes;
        if (!m->stage_method[0])
            snprintf(m->stage_method, sizeof(m->stage_method), "%s",
                     stage_want == QE_STAGE_MMAP
                         ? "mmap"
                         : (stage_how ? stage_how
                                      : (stage_want == QE_STAGE_HUGE ? "hugetlb" : "ram")));
        rd r = {(const uint8_t *)s->map, (const uint8_t *)s->map + len, 0};
        /* Claim the slot now so any later error path still unmaps/closes it. */
        const int mine = m->nshards;
        m->nshards++;
        if ((size_t)(r.e - r.p) < 24 || memcmp(r.p, "GGUF", 4) != 0)
            QE_ERR("%s: not a GGUF file", vp);
        r.p += 4;
        uint32_t ver = r_u32(&r);
        if (ver < 2 || ver > 3)
            QE_ERR("%s: unsupported GGUF version %u", vp, ver);
        uint64_t n_tensors = r_u64(&r);
        s->n_tensors = n_tensors;
        long alignment = 0;
        /* Volume metadata is authoritative in the first file; later volumes
         * repeat a couple of keys, which parse_metadata de-duplicates. */
        if (parse_metadata(m, &r, &alignment) != 0)
            QE_ERR("%s: metadata parse error at offset %zu of %zu", vp,
                   (size_t)(r.p - (const uint8_t *)s->map), s->map_len);
        s->alignment = alignment > 0 ? (size_t)alignment : QE_ALIGN_DEFAULT;

        if (m->n_tensors + n_tensors > m->cap_tensors) {
            m->cap_tensors = (m->n_tensors + n_tensors) * 3 / 2 + 64;
            m->t = realloc(m->t, m->cap_tensors * sizeof(*m->t));
        }
        for (uint64_t i = 0; i < n_tensors && !r.bad; i++) {
            qe_tensor *t = &m->t[m->n_tensors++];
            memset(t, 0, sizeof(*t));
            size_t nlen;
            const char *nm = r_str(&r, &nlen);
            if (r.bad) break;
            if (nlen >= QE_MAX_NAME) nlen = QE_MAX_NAME - 1;
            memcpy(t->name, nm, nlen);
            t->name[nlen] = 0;
            uint32_t nd = r_u32(&r);
            if (nd > 4) { r.bad = 1; break; }
            for (uint32_t d = 0; d < nd; d++)
                t->ne[d] = r_u64(&r);
            for (int d = (int)nd; d < 4; d++)
                t->ne[d] = 1;
            t->ndim = (int)nd;
            t->type = (qe_type)r_u32(&r);
            t->off = r_u64(&r);
            t->shard = mine;
            t->owner = m;
            const qe_type_info *ti = qe_type_info_of(t->type);
            if (!ti)
                QE_ERR("%s: tensor %s has unknown type id %d", vp, t->name, (int)t->type);
            uint64_t n_el = 1;
            for (int d = 0; d < 4; d++)
                n_el *= t->ne[d];
            if (n_el % ti->block_ne)
                QE_ERR("%s: tensor %s has %" PRIu64 " elements, not a multiple of block %u",
                       vp, t->name, n_el, ti->block_ne);
            t->bytes = n_el / ti->block_ne * ti->block_sz;
            if (nd == 3) {
                t->n_experts = (uint32_t)t->ne[2];
                uint64_t per = t->ne[0] * t->ne[1];
                t->expert_stride = per / ti->block_ne * ti->block_sz;
            }
        }
        if (r.bad)
            QE_ERR("%s: tensor info table truncated", vp);
        /* Where this volume's tensor bytes begin. The offsets in the info table
         * count from here, not from the start of the file, and the difference
         * is this header: 23,584 bytes on volume 2 of this checkpoint and 54,336
         * on volume 3. A constant displacement of that size is invisible to
         * every bandwidth measurement in this tree while it silently shifts each
         * quantization block by 2,358 samples, which is how a checkpoint reads
         * as plausible bytes and dequantises to NaNs. */
        {
            size_t hdr_end = (size_t)((const uint8_t *) r.p - (const uint8_t *) s->map);
            s->data_off = qe_align_up(hdr_end, s->alignment);
            s->off_base = 0;
            if (s->data_off > s->file_len)
                QE_ERR("%s: header ends at %zu, file is %zu", vp, s->data_off, s->file_len);
        }
    }
    if (m->n_tensors == 0)
        QE_ERR("no tensors found in %s", path);

    /* Prove the type table: tensor ranges must tile each volume gaplessly. */
    for (int si = 0; si < m->nshards; si++) {
        const qe_tensor **v = malloc(m->n_tensors * sizeof(*v));
        size_t nv = 0;
        uint64_t minoff = UINT64_MAX;
        for (size_t i = 0; i < m->n_tensors; i++) {
            if (m->t[i].shard != si)
                continue;
            v[nv++] = &m->t[i];
            if (m->t[i].off < minoff)
                minoff = m->t[i].off;
        }
        qsort(v, nv, sizeof(*v), cmp_off);
        uint64_t cur = minoff, end = 0;
        int bad = 0;
        for (size_t a = 0; a < nv; a++) {
            if (v[a]->off != cur) {
                qe_seterr("volume %s: offset chain broken at %s (expected %" PRIu64
                          ", file says %" PRIu64 ") - type table does not match this file",
                          m->sh[si].path, v[a]->name, cur, v[a]->off);
                bad = 1;
                break;
            }
            cur += v[a]->bytes;
            cur = qe_align_up(cur, m->sh[si].alignment);
            end = cur;
        }
        free(v);
        if (bad) goto fail;
        m->sh[si].off_base = nv ? minoff : 0;
        /* The chain is relative to the data section, so the file has to hold
         * the header plus the whole chain. Checking the chain length against the
         * file length alone under-counts by the header. */
        if (nv && m->sh[si].data_off + (end - m->sh[si].off_base) > m->sh[si].file_len)
            QE_ERR("volume %s: tensors need %zu bytes past a %zu byte header, file is %zu",
                   m->sh[si].path, (size_t) (end - m->sh[si].off_base), m->sh[si].data_off,
                   m->sh[si].file_len);
    }

    /* Volumes are already mapped and stay demand-paged. Pinning is opt-in via
     * qe_model_warm() so that tooling which touches a handful of tensors does
     * not first read 87 GiB off disk. */
    (void)flags;
    return m;
fail:
    qe_close(m);
    return NULL;
}

void qe_close(qe_model *m) {
    if (!m)
        return;
    for (int i = 0; i < m->nshards; i++) {
        if (m->sh[i].map && m->sh[i].map != MAP_FAILED)
            munmap(m->sh[i].map, m->sh[i].map_len);
        if (m->sh[i].fd >= 0)
            close(m->sh[i].fd);
        free((void *)m->sh[i].path);
    }
    if (m->kv) {
        for (size_t i = 0; i < m->n_kv; i++) {
            free(m->kv[i].s);
            if (m->kv[i].kind == KV_ARR_STR && m->kv[i].arr)
                for (size_t j = 0; j < m->kv[i].n_arr; j++)
                    free(((char **)m->kv[i].arr)[j]);
            free(m->kv[i].arr);
        }
        free(m->kv);
    }
    free(m->t);
    if (m->arena)
        qe_free_huge(m->arena, m->arena_len);
    free(m);
}

/* ------------------------------------------------------------------ lookups */

size_t qe_tensor_count(qe_model *m) { return m->n_tensors; }

int qe_model_volumes(qe_model *m) { return m->nshards; }

const char *qe_model_stage_method(qe_model *m) {
    if (!m || !m->stage_method[0])
        return "mmap";
    return m->stage_method;
}
size_t qe_model_stage_bytes(qe_model *m) { return m ? m->stage_bytes : 0; }
size_t qe_model_stage_cached(qe_model *m) { return m ? m->stage_cached : 0; }
int qe_model_stage_pinned(qe_model *m) { return m ? m->stage_pinned : 0; }

const qe_tensor *qe_tensor_by_index(qe_model *m, size_t i) {
    return i < m->n_tensors ? &m->t[i] : NULL;
}

const qe_tensor *qe_tensor_by_name(qe_model *m, const char *name) {
    for (size_t i = 0; i < m->n_tensors; i++)
        if (strcmp(m->t[i].name, name) == 0)
            return &m->t[i];
    return NULL;
}

static const struct kv *kv_find(qe_model *m, const char *key) {
    for (size_t i = 0; i < m->n_kv; i++)
        if (strcmp(m->kv[i].key, key) == 0)
            return &m->kv[i];
    return NULL;
}

int qe_model_get_int(qe_model *m, const char *key, long long *out) {
    const struct kv *k = kv_find(m, key);
    if (!k || k->kind != KV_INT) return -1;
    *out = k->i;
    return 0;
}
int qe_model_get_float(qe_model *m, const char *key, double *out) {
    const struct kv *k = kv_find(m, key);
    if (!k) return -1;
    if (k->kind == KV_REAL) *out = k->r;
    else if (k->kind == KV_INT) *out = (double)k->i;
    else return -1;
    return 0;
}
int qe_model_get_str(qe_model *m, const char *key, const char **out) {
    const struct kv *k = kv_find(m, key);
    if (!k || k->kind != KV_STR) return -1;
    *out = k->s;
    return 0;
}
int qe_model_get_arr_int(qe_model *m, const char *key, const long long **out, size_t *n) {
    const struct kv *k = kv_find(m, key);
    if (!k || k->kind != KV_ARR_INT) return -1;
    *out = k->arr;
    *n = k->n_arr;
    return 0;
}
int qe_model_get_arr_str(qe_model *m, const char *key, const char *const **out, size_t *n) {
    const struct kv *k = kv_find(m, key);
    if (!k || k->kind != KV_ARR_STR) return -1;
    *out = k->arr;
    *n = k->n_arr;
    return 0;
}
int qe_model_get_arr_len(qe_model *m, const char *key, size_t *n) {
    const struct kv *k = kv_find(m, key);
    if (!k || (k->kind != KV_ARR_INT && k->kind != KV_ARR_STR && k->kind != KV_ARR_REAL))
        return -1;
    *n = k->n_arr;
    return 0;
}

const void *qe_tensor_data(qe_model *m, const qe_tensor *t) {
    /* A row of a draft head read by a pass that holds the target answers out of the
     * volume it came from, not the one it was asked about: `shard` and `off` index
     * the owner, and reading them through the wrong map is a plausible-looking
     * answer made of somebody else's bytes. */
    qe_model *mm = t->owner ? t->owner : m;
    if ((unsigned)t->shard >= (unsigned)mm->nshards || !mm->sh[t->shard].map)
        return NULL;
    const qe_shard *s = &mm->sh[t->shard];
    if (t->off < s->off_base)
        return NULL; /* an offset from before this volume's data section */
    size_t at = s->data_off + (size_t)(t->off - s->off_base);
    if (at + t->bytes > s->map_len)
        return NULL; /* would read past the mapping */
    return (const uint8_t *)s->map + at;
}

void qe_tensor_describe(qe_model *m, const qe_tensor *t, char *buf, size_t n) {
    const qe_type_info *ti = qe_type_info_of(t->type);
    snprintf(buf, n, "%-38s ne=(%" PRIu64 ",%" PRIu64 ",%" PRIu64 ") %-8s %9.2f MiB %6.3f bpw",
             t->name, t->ne[0], t->ne[1], t->ne[2], ti ? ti->name : "?",
             (double)t->bytes / 1048576.0, qe_type_bpw_x1000(t->type) / 1000.0);
    (void)m;
}

/* Bring selected tensors into RAM (and optionally lock them). Used instead of
 * blanket prefaulting the file: the dense projections and hyper-connections
 * that every token reads are ~4 GiB, while the PLE table they would compete
 * with is 27 GiB and is touched at ~1.4 KiB per token. */
int qe_model_warm(qe_model *m, const char *const *names, int nnames, int mlock_it) {
    long ps = qe_page_size();
    uint64_t touched = 0;
    for (int i = 0; i < nnames; i++) {
        const qe_tensor *t = qe_tensor_by_name(m, names[i]);
        if (!t)
            continue;
        const uint8_t *p = qe_tensor_data(m, t);
        if (!p)
            continue;
        if (mlock_it) {
#ifdef MLOCK_ONFAULT
            if (mlock(p, qe_align_up(t->bytes, (size_t)ps)) != 0)
                return -1;
#else
            (void)mlock_it;
#endif
        }
        volatile const uint8_t *v = p;
        for (uint64_t o = 0; o < t->bytes; o += (uint64_t)ps)
            (void)v[o];
        touched += t->bytes;
    }
    return (int)(touched >> 20);
}
