/* The real tokenizer: byte-level BPE, driven entirely by what the GGUF says.

   Until this file existed the server *estimated* prompt size as len/4, which is
   fine for a progress bar and useless for anything else: it cannot tell you what
   the model was actually shown, it disagrees with the checkpoint's own
   vocabulary, and it makes the usage accounting in OpenAI responses a
   fabrication. Everything here comes from the file instead - the 248,320 token
   strings, the 247,587 merges and their ranks, the token_type array that says
   which strings are control tokens, and tokenizer.ggml.pre = qwen35, which
   selects the pre-tokenizer.

   Three properties worth stating, because the serving code leans on them:

     - Encoding is lossless. Every one of the 256 byte values has a token in this
       vocabulary (checked at load, not assumed), so any input survives
       encode/decode even when no linguistically sensible split exists. No byte
       fallback is needed, and the file ships none (`<0xNN>` tokens are absent),
       which is only consistent with that.

     - Nothing about this model is hardcoded. Vocabulary size, merge count,
       special tokens and eos/bos/pad all come from KV pairs.

     - The pre-tokenizer is the qwen35 rule, translated from llama.cpp's
       unicode_regex_split_custom_qwen35 rather than approximated, because the
       splitter decides whether "don't" is one token or three and there is no way
       to notice a silent disagreement. The Unicode categories it needs are a
       generated table (tools/gen_uni.py -> src/uni_table.h).

   Merge application is the textbook "merge the lowest-ranked adjacent pair,
   repeatedly". It is quadratic in the length of one pre-token piece, so pieces
   are chunked and a work counter bounds the pathological case; when the budget
   runs out the remaining units are emitted as-is, which costs tokens but never
   content. */
#define _GNU_SOURCE
#include "qe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uni_table.h"

/* A pre-token piece longer than this is cut into chunks before merging. Only
 * reachable with long runs of a single class (letters, digits, spaces). */
#define TOK_PIECE_MAX 1024
/* Adjacent-pair probes per chunk. Bounds encode cost on adversarial input. */
#define TOK_MERGE_WORK (1u << 18)

#define U2B_N 0x15B /* highest byte-level codepoint is 0x100 + 90 = 0x15A */

struct qe_tok {
    long n_vocab;
    uint32_t *off;    /* n_vocab + 1 offsets into bytes */
    char *bytes;      /* every token's bytes, in original (pre byte-level) space */
    uint8_t *special; /* 1 for control/user-defined tokens, matched literally */

    int32_t eos, bos, pad;

    /* token string -> id; values are id + 1, 0 means empty */
    uint32_t *vh;
    uint64_t vh_mask;

    /* adjacent pair -> merge. Key is (a << 32) | b stored as key + 1 so 0 can
     * mean empty; mh_r holds rank + 1, mh_t the resulting token id + 1. */
    uint64_t *mh_k;
    uint32_t *mh_r;
    uint32_t *mh_t;
    uint64_t mh_mask;

    uint32_t byte_tok[256]; /* the token for one byte of input */
    int has_bytes;          /* 256 when the vocabulary covers every byte value */

    /* trie over the special tokens, for longest literal match */
    int32_t *tn_head; /* first edge of a node, -1 if none */
    int32_t *tn_tok;  /* token id when a node ends a special, else -1 */
    int32_t *ed_ch;
    int32_t *ed_child;
    int32_t *ed_next;
    int n_node, n_edge;
    uint8_t first[256]; /* bytes a special token can start with */

    /* load-time accounting, surfaced through qe_tok_describe */
    long n_merges, n_merges_used, n_merges_skip, n_special;
    char pre[32];
};

/* ------------------------------------------------------------- byte level */

/* GPT-2's bytes_to_unicode: printable bytes stand for themselves, the rest are
 * lifted into 0x100.. so that no byte value can appear inside a UTF-8 multi-byte
 * sequence. The GGUF stores token strings in that space, so reading them back
 * through the inverse map recovers the bytes the tokenizer actually manipulates.
 * This is the inverse direction only. */
static void u2b_init(int16_t *u2b) {
    for (int i = 0; i < U2B_N; i++)
        u2b[i] = -1;
    int k = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            u2b[b] = (int16_t) b;
        else
            u2b[0x100 + k++] = (int16_t) b;
    }
}

/* One codepoint of UTF-8. Malformed input yields 0xFFFD having consumed a single
 * byte, which keeps byte offsets honest and therefore keeps decode exact. */
static size_t utf8_dec(const char *s, size_t n, size_t i, uint32_t *cp) {
    unsigned char c = (unsigned char) s[i];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    uint32_t v;
    unsigned want, lo;
    if (c < 0xC2) {
        *cp = 0xFFFD;
        return 1;
    } else if (c < 0xE0) {
        v = c & 0x1Fu;
        want = 1;
        lo = 0x80u;
    } else if (c < 0xF0) {
        v = c & 0x0Fu;
        want = 2;
        lo = 0x800u;
    } else if (c < 0xF5) {
        v = c & 0x07u;
        want = 3;
        lo = 0x10000u;
    } else {
        *cp = 0xFFFD;
        return 1;
    }
    if (i + want >= n) {
        *cp = 0xFFFD;
        return 1;
    }
    for (unsigned k = 1; k <= want; k++) {
        unsigned char d = (unsigned char) s[i + k];
        if ((d & 0xC0) != 0x80) {
            *cp = 0xFFFD;
            return 1;
        }
        v = (v << 6) | (d & 0x3Fu);
    }
    if (v < lo || v > 0x10FFFFu || (v >= 0xD800u && v <= 0xDFFFu)) {
        *cp = 0xFFFD;
        return 1;
    }
    *cp = v;
    return want + 1;
}

/* Map a byte-level string back to its bytes; -1 when it is outside the
 * byte-level alphabet. Output is never longer than the input. */
static int bl2raw(const char *s, size_t n, const int16_t *u2b, char *dst, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        size_t adv = utf8_dec(s, n, i, &cp);
        if (cp >= U2B_N || u2b[cp] < 0 || k >= cap)
            return -1;
        dst[k++] = (char) u2b[cp];
        i += adv;
    }
    return (int) k;
}

/* ----------------------------------------------------------------- hashes */

static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++)
        h = (h ^ (unsigned char) s[i]) * 1099511628211ULL;
    return h;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return x;
}

static uint32_t vh_get(const qe_tok *t, const char *s, size_t n) {
    if (!t->vh)
        return 0;
    uint64_t i = fnv(s, n) & t->vh_mask;
    for (;;) {
        uint32_t v = t->vh[i];
        if (!v)
            return 0;
        uint32_t id = v - 1;
        if (t->off[id + 1] - t->off[id] == n && !memcmp(t->bytes + t->off[id], s, n))
            return v;
        i = (i + 1) & t->vh_mask;
    }
}

static int vh_put(qe_tok *t, uint32_t id) {
    size_t n = t->off[id + 1] - t->off[id];
    uint64_t i = fnv(t->bytes + t->off[id], n) & t->vh_mask;
    for (;;) {
        if (!t->vh[i]) {
            t->vh[i] = id + 1;
            return 0;
        }
        if (t->off[t->vh[i] - 1 + 1] - t->off[t->vh[i] - 1] == n &&
            !memcmp(t->bytes + t->off[t->vh[i] - 1], t->bytes + t->off[id], n))
            return -1; /* the same string twice: the file is malformed */
        i = (i + 1) & t->vh_mask;
    }
}

static int mh_get(const qe_tok *t, uint32_t a, uint32_t b, uint32_t *rank, uint32_t *target) {
    if (!t->mh_k)
        return 0;
    uint64_t k = (((uint64_t) a << 32) | b) + 1;
    uint64_t i = mix64(k) & t->mh_mask;
    for (;;) {
        if (!t->mh_k[i])
            return 0;
        if (t->mh_k[i] == k) {
            if (rank)
                *rank = t->mh_r[i];
            if (target)
                *target = t->mh_t[i];
            return 1;
        }
        i = (i + 1) & t->mh_mask;
    }
}

static void mh_put(qe_tok *t, uint32_t a, uint32_t b, uint32_t rank, uint32_t target) {
    uint64_t k = (((uint64_t) a << 32) | b) + 1;
    uint64_t i = mix64(k) & t->mh_mask;
    while (t->mh_k[i])
        i = (i + 1) & t->mh_mask;
    t->mh_k[i] = k;
    t->mh_r[i] = rank;
    t->mh_t[i] = target;
}

/* --------------------------------------------------------- special tokens */

static int trie_new(qe_tok *t, int cap_nodes, int cap_edges) {
    t->tn_head = malloc((size_t) cap_nodes * sizeof *t->tn_head);
    t->tn_tok = malloc((size_t) cap_nodes * sizeof *t->tn_tok);
    t->ed_ch = malloc((size_t) cap_edges * sizeof *t->ed_ch);
    t->ed_child = malloc((size_t) cap_edges * sizeof *t->ed_child);
    t->ed_next = malloc((size_t) cap_edges * sizeof *t->ed_next);
    if (!t->tn_head || !t->tn_tok || !t->ed_ch || !t->ed_child || !t->ed_next)
        return -1;
    t->n_node = 1; /* root */
    t->tn_head[0] = -1;
    t->tn_tok[0] = -1;
    t->n_edge = 0;
    return 0;
}

static void trie_add(qe_tok *t, const char *s, size_t n, int32_t id) {
    int32_t node = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char) s[i];
        int32_t e = t->tn_head[node], child = -1;
        for (; e >= 0; e = t->ed_next[e])
            if (t->ed_ch[e] == c) {
                child = t->ed_child[e];
                break;
            }
        if (child < 0) {
            child = t->n_node++;
            t->tn_head[child] = -1;
            t->tn_tok[child] = -1;
            t->ed_ch[t->n_edge] = c;
            t->ed_child[t->n_edge] = child;
            t->ed_next[t->n_edge] = t->tn_head[node];
            t->tn_head[node] = t->n_edge++;
        }
        node = child;
    }
    t->tn_tok[node] = id;
    t->first[(unsigned char) s[0]] = 1;
}

/* Longest special token at i, or -1; *adv receives its length. */
static int32_t trie_match(const qe_tok *t, const char *s, size_t n, size_t i, size_t *adv) {
    int32_t best = -1, node = 0;
    size_t best_len = 0, k = i;
    while (k < n) {
        unsigned char c = (unsigned char) s[k];
        int32_t nxt = -1;
        for (int32_t e = t->tn_head[node]; e >= 0; e = t->ed_next[e])
            if (t->ed_ch[e] == c) {
                nxt = t->ed_child[e];
                break;
            }
        if (nxt < 0)
            break;
        node = nxt;
        k++;
        if (t->tn_tok[node] >= 0) {
            best = t->tn_tok[node];
            best_len = k - i;
        }
    }
    *adv = best_len;
    return best;
}

/* ------------------------------------------------------------ qwen35 split
 *
 * A translation of llama.cpp's unicode_regex_split_custom_qwen35, applying in
 * order of precedence:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
 *   \p{N}
 *   <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 * cb() receives one pre-token piece at a time. */

#define OUT_OF_RANGE 0xFFFFFFFFu

struct seg {
    const char *s;    /* segment start, in bytes */
    const uint32_t *cp;
    const uint32_t *boff; /* byte offset of each codepoint, plus one sentinel */
    long ini, end;        /* codepoint range */
};

static uint32_t seg_cp(const struct seg *g, long pos) {
    return (pos >= g->ini && pos < g->end) ? g->cp[pos] : OUT_OF_RANGE;
}

/* category flags; 0 outside the segment, matching llama.cpp's zeroed struct */
static uint8_t seg_fl(const struct seg *g, long pos) {
    return (pos >= g->ini && pos < g->end) ? uni_flags(g->cp[pos]) : 0;
}

static void split_qwen35(const struct seg *g, void (*cb)(void *, const char *, size_t),
                         void *ctx) {
    long prev = g->ini;
    long pos = g->ini;
    while (pos < g->end) {
        uint32_t cpt = seg_cp(g, pos);
        uint8_t f = seg_fl(g, pos);
        long nxt = -1;

        /* contractions, case-insensitive */
        if (cpt == '\'' && pos + 1 < g->end) {
            uint32_t n1 = uni_tolower32(seg_cp(g, pos + 1));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd')
                nxt = pos + 2;
            if (nxt < 0 && pos + 2 < g->end) {
                uint32_t n2 = uni_tolower32(seg_cp(g, pos + 2));
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') ||
                    (n1 == 'l' && n2 == 'l'))
                    nxt = pos + 3;
            }
        }

        /* an optional non-letter, then a run of letters and combining marks */
        if (nxt < 0 && !(cpt == '\r' || cpt == '\n' || (f & UNI_NUMBER))) {
            if ((f & (UNI_LETTER | UNI_MARK)) ||
                (seg_fl(g, pos + 1) & (UNI_LETTER | UNI_MARK))) {
                long p = pos + 1;
                while (seg_fl(g, p) & (UNI_LETTER | UNI_MARK))
                    p++;
                nxt = p;
            }
        }

        /* digits are tokens by themselves */
        if (nxt < 0 && (f & UNI_NUMBER))
            nxt = pos + 1;

        /* punctuation runs, optionally started by a space, then newlines */
        if (nxt < 0) {
            uint8_t f2 = (cpt == ' ') ? seg_fl(g, pos + 1) : f;
            if (!(f2 & (UNI_SPACE | UNI_LETTER | UNI_MARK | UNI_NUMBER)) && f) {
                long p = pos + (cpt == ' ');
                uint8_t q = f2;
                while (!(q & (UNI_SPACE | UNI_LETTER | UNI_MARK | UNI_NUMBER)) && q)
                    q = seg_fl(g, ++p);
                uint32_t c2 = seg_cp(g, p);
                while (c2 == '\r' || c2 == '\n')
                    c2 = seg_cp(g, ++p);
                nxt = p;
            }
        }

        /* whitespace: newlines first, then a run that a word follows, then a run */
        if (nxt < 0 && (f & UNI_SPACE)) {
            long nw = 0, last_nl = 0;
            while (seg_fl(g, pos + nw) & UNI_SPACE) {
                uint32_t c2 = seg_cp(g, pos + nw);
                if (c2 == '\r' || c2 == '\n')
                    last_nl = pos + nw + 1;
                nw++;
            }
            if (last_nl > 0)
                nxt = last_nl;
            else if (nw > 1 && seg_cp(g, pos + nw) != OUT_OF_RANGE)
                nxt = pos + nw - 1;
            else
                nxt = pos + nw;
        }

        if (nxt < 0)
            nxt = pos + 1; /* no rule matched: one codepoint */
        if (nxt > g->end)
            nxt = g->end;
        if (nxt > prev) {
            size_t a = g->boff[prev], b = g->boff[nxt];
            cb(ctx, g->s + a, b - a);
            prev = nxt;
        }
        pos = nxt;
    }
}

/* ------------------------------------------------------------------ encode
 *
 * One emit context per call. ids == NULL means "count only". */

struct enc {
    const qe_tok *t;
    int *ids;
    long n, cap;
    int overflow;
    uint32_t u[]; /* merge units, one chunk at a time */
};

static void enc_id(struct enc *e, uint32_t id) {
    if (e->ids) {
        if (e->n >= e->cap) {
            e->overflow = 1;
            return;
        }
        e->ids[e->n] = (int) id;
    }
    e->n++;
}

/* Byte-level encode one pre-token piece, then merge it into tokens. */
static void enc_piece(void *vctx, const char *p, size_t n) {
    struct enc *e = vctx;
    const qe_tok *t = e->t;
    size_t at = 0;
    while (at < n) {
        size_t len = n - at;
        if (len > TOK_PIECE_MAX)
            len = TOK_PIECE_MAX;
        long nu = 0;
        for (size_t i = 0; i < len; i++)
            e->u[nu++] = t->byte_tok[(unsigned char) p[at + i]];
        unsigned work = 0;
        while (nu > 1 && work < TOK_MERGE_WORK) {
            long best = -1;
            uint32_t br = 0, bt = 0;
            for (long i = 0; i + 1 < nu; i++) {
                uint32_t r, tg;
                work++;
                if (mh_get(t, e->u[i], e->u[i + 1], &r, &tg) && (!br || r < br)) {
                    br = r;
                    bt = tg;
                    best = i;
                }
                if (work >= TOK_MERGE_WORK)
                    break;
            }
            if (best < 0)
                break;
            e->u[best] = bt - 1;
            memmove(&e->u[best + 1], &e->u[best + 2], (size_t) (nu - best - 2) * sizeof *e->u);
            nu--;
        }
        for (long i = 0; i < nu; i++)
            enc_id(e, e->u[i]);
        at += len;
    }
}

static void enc_range(struct enc *e, const char *s, size_t a, size_t b, uint32_t *cp,
                      uint32_t *boff) {
    long ncp = 0;
    boff[0] = 0;
    for (size_t i = a; i < b;) {
        uint32_t c;
        size_t adv = utf8_dec(s, b, i, &c);
        cp[ncp++] = c;
        i += adv;
        boff[ncp] = (uint32_t) (i - a);
    }
    struct seg g = { s + a, cp, boff, 0, ncp };
    split_qwen35(&g, enc_piece, e);
}

static long encode(const qe_tok *t, const char *text, int *ids, long max_ids) {
    if (!t || !text)
        return -1;
    size_t n = strlen(text);
    if (!n)
        return 0;

    struct enc *e = malloc(sizeof *e + (TOK_PIECE_MAX + 1) * sizeof *e->u);
    uint32_t *cp = malloc((n + 1) * sizeof *cp);
    uint32_t *boff = malloc((n + 2) * sizeof *boff);
    if (!e || !cp || !boff) {
        free(e);
        free(cp);
        free(boff);
        return -1;
    }
    e->t = t;
    e->ids = ids;
    e->n = 0;
    e->cap = max_ids;
    e->overflow = 0;

    /* One pass over the text. Special tokens are matched literally and cut the
     * text into pieces; everything between them is pre-tokenized and merged. */
    size_t i = 0, seg_start = 0;
    while (i < n) {
        if (!t->first[(unsigned char) text[i]]) {
            i++;
            continue;
        }
        size_t adv = 0;
        int32_t sid = trie_match(t, text, n, i, &adv);
        if (sid < 0) {
            i++;
            continue;
        }
        if (i > seg_start)
            enc_range(e, text, seg_start, i, cp, boff);
        enc_id(e, (uint32_t) sid);
        i += adv;
        seg_start = i;
    }
    if (n > seg_start)
        enc_range(e, text, seg_start, n, cp, boff);

    long out = e->overflow ? -1 : e->n;
    free(e);
    free(cp);
    free(boff);
    return out;
}

/* ------------------------------------------------------------ public API */

qe_tok *qe_tok_open(qe_model *m, char *err, size_t errn) {
    const char *why = NULL;
    qe_tok *t = NULL;
    if (!m) {
        why = "no model";
        goto fail;
    }
    const char *kind = NULL;
    if (qe_model_get_str(m, "tokenizer.ggml.model", &kind) || !kind) {
        why = "tokenizer.ggml.model is missing";
        goto fail;
    }
    if (strcmp(kind, "gpt2")) {
        why = "only tokenizer.ggml.model = gpt2 (byte-level BPE) is supported";
        goto fail;
    }
    const char *pre = NULL;
    qe_model_get_str(m, "tokenizer.ggml.pre", &pre);
    if (pre && strcmp(pre, "qwen35") && strcmp(pre, "qwen2") && strcmp(pre, "qwen3")) {
        why = "this tokenizer's pre-tokenizer is not implemented here";
        goto fail;
    }

    const char *const *toks = NULL;
    size_t ntok = 0;
    if (qe_model_get_arr_str(m, "tokenizer.ggml.tokens", &toks, &ntok) || !ntok) {
        why = "tokenizer.ggml.tokens is missing";
        goto fail;
    }

    t = calloc(1, sizeof *t);
    if (!t) {
        why = "out of memory";
        goto fail;
    }
    snprintf(t->pre, sizeof t->pre, "%s", pre ? pre : "none");
    t->n_vocab = (long) ntok;
    t->eos = t->bos = t->pad = -1;
    long long v;
    if (qe_model_get_int(m, "tokenizer.ggml.eos_token_id", &v) == 0)
        t->eos = (int32_t) v;
    if (qe_model_get_int(m, "tokenizer.ggml.bos_token_id", &v) == 0)
        t->bos = (int32_t) v;
    if (qe_model_get_int(m, "tokenizer.ggml.padding_token_id", &v) == 0)
        t->pad = (int32_t) v;

    /* token strings, mapped back out of the byte-level alphabet */
    size_t total = 1;
    for (size_t i = 0; i < ntok; i++)
        total += strlen(toks[i]) + 1;
    t->off = malloc((ntok + 1) * sizeof *t->off);
    t->bytes = malloc(total);
    t->special = calloc(ntok, 1);
    if (!t->off || !t->bytes || !t->special) {
        why = "out of memory";
        goto fail_t;
    }
    int16_t u2b[U2B_N];
    u2b_init(u2b);
    size_t at = 0;
    for (size_t i = 0; i < ntok; i++) {
        t->off[i] = (uint32_t) at;
        size_t l = strlen(toks[i]);
        int r = bl2raw(toks[i], l, u2b, t->bytes + at, total - at - 1);
        if (r < 0) {
            /* Not a byte-level string (control tokens are, so this is a
             * curiosity). Keep the bytes so decode still works; the token then
             * can never be produced by encode, which is what the hash below
             * enforces by never seeing it. */
            memcpy(t->bytes + at, toks[i], l);
            at += l;
        } else {
            at += (size_t) r;
        }
        t->off[i + 1] = (uint32_t) at;
    }
    t->bytes[at] = 0;

    t->vh_mask = 1;
    while (t->vh_mask < (uint64_t) ntok * 2)
        t->vh_mask = t->vh_mask * 2 + 1;
    t->vh = calloc((size_t) t->vh_mask + 1, sizeof *t->vh);
    if (!t->vh) {
        why = "out of memory";
        goto fail_t;
    }
    for (size_t i = 0; i < ntok; i++) {
        if (t->off[i + 1] == t->off[i])
            continue;
        if (vh_put(t, (uint32_t) i) < 0) {
            why = "duplicate string in tokenizer.ggml.tokens";
            goto fail_t;
        }
    }

    for (int b = 0; b < 256; b++) {
        char one = (char) b;
        uint32_t id1 = vh_get(t, &one, 1);
        if (!id1)
            break;
        t->byte_tok[b] = id1 - 1;
        t->has_bytes++;
    }
    if (t->has_bytes != 256) {
        why = "the vocabulary has no token for every byte value, so lossless "
              "encoding would need byte fallback and this file defines none";
        goto fail_t;
    }

    /* specials: everything the file does not call a normal token */
    const long long *tt = NULL;
    size_t ntt = 0;
    if (qe_model_get_arr_int(m, "tokenizer.ggml.token_type", &tt, &ntt) == 0 &&
        ntt == ntok) {
        size_t nodes = 1, edges = 0;
        for (size_t i = 0; i < ntok; i++)
            if (tt[i] != 1) {
                size_t l = strlen(toks[i]);
                nodes += l + 1;
                edges += l;
                t->n_special++;
            }
        if (t->n_special) {
            if (trie_new(t, (int) nodes, (int) edges) < 0) {
                why = "out of memory";
                goto fail_t;
            }
            for (size_t i = 0; i < ntok; i++)
                if (tt[i] != 1 && strlen(toks[i])) {
                    trie_add(t, toks[i], strlen(toks[i]), (int32_t) i);
                    t->special[i] = 1;
                }
        }
    }

    /* merges */
    const char *const *mg = NULL;
    size_t nmg = 0;
    if (qe_model_get_arr_str(m, "tokenizer.ggml.merges", &mg, &nmg) == 0 && nmg) {
        t->mh_mask = 1;
        while (t->mh_mask < (uint64_t) nmg * 2)
            t->mh_mask = t->mh_mask * 2 + 1;
        t->mh_k = calloc((size_t) t->mh_mask + 1, sizeof *t->mh_k);
        t->mh_r = calloc((size_t) t->mh_mask + 1, sizeof *t->mh_r);
        t->mh_t = calloc((size_t) t->mh_mask + 1, sizeof *t->mh_t);
        char *tmp = malloc(1024);
        if (!t->mh_k || !t->mh_r || !t->mh_t || !tmp) {
            free(tmp);
            why = "out of memory";
            goto fail_t;
        }
        t->n_merges = (long) nmg;
        for (size_t i = 0; i < nmg; i++) {
            const char *s = mg[i];
            const char *sp = strchr(s, ' ');
            if (!sp) {
                t->n_merges_skip++;
                continue;
            }
            size_t la = (size_t) (sp - s), lb = strlen(sp + 1);
            /* A piece can never contain a space: in the byte-level alphabet the
             * space character is itself a multi-byte codepoint. */
            if (strchr(sp + 1, ' ') || la > 500 || lb > 500) {
                t->n_merges_skip++;
                continue;
            }
            int ra = bl2raw(s, la, u2b, tmp, 512);
            int rb = bl2raw(sp + 1, lb, u2b, tmp + 512, 512);
            if (ra < 0 || rb < 0) {
                t->n_merges_skip++;
                continue;
            }
            uint32_t a = vh_get(t, tmp, (size_t) ra);
            uint32_t b = vh_get(t, tmp + 512, (size_t) rb);
            if (!a || !b) {
                t->n_merges_skip++;
                continue;
            }
            memmove(tmp + ra, tmp + 512, (size_t) rb);
            uint32_t c = vh_get(t, tmp, (size_t) ra + (size_t) rb);
            if (!c) {
                t->n_merges_skip++;
                continue;
            }
            mh_put(t, a - 1, b - 1, (uint32_t) i + 1, c);
            t->n_merges_used++;
        }
        free(tmp);
    }
    return t;

fail_t:
    qe_tok_close(t);
    t = NULL;
fail:
    if (err && errn)
        snprintf(err, errn, "%s", why ? why : "tokenizer load failed");
    return t;
}

void qe_tok_close(qe_tok *t) {
    if (!t)
        return;
    free(t->off);
    free(t->bytes);
    free(t->special);
    free(t->vh);
    free(t->mh_k);
    free(t->mh_r);
    free(t->mh_t);
    free(t->tn_head);
    free(t->tn_tok);
    free(t->ed_ch);
    free(t->ed_child);
    free(t->ed_next);
    free(t);
}

long qe_tok_n_vocab(const qe_tok *t) {
    return t ? t->n_vocab : 0;
}

int qe_tok_eos(const qe_tok *t) {
    return t ? t->eos : -1;
}

int qe_tok_bos(const qe_tok *t) {
    return t ? t->bos : -1;
}

int qe_tok_is_special(const qe_tok *t, int id) {
    if (!t || id < 0 || id >= t->n_vocab)
        return 0;
    return t->special[id];
}

long qe_tok_encode(const qe_tok *t, const char *text, int *ids, long max_ids) {
    return encode(t, text, ids, max_ids);
}

long qe_tok_count(const qe_tok *t, const char *text) {
    return encode(t, text, NULL, 0);
}

long qe_tok_decode(const qe_tok *t, const int *ids, long n, char *out, size_t cap) {
    if (!t || (!ids && n))
        return -1;
    size_t k = 0;
    for (long i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->n_vocab)
            continue; /* an id the model invented must not corrupt the rest */
        size_t a = t->off[id], b = t->off[id + 1];
        if (out && b > a) {
            size_t room = cap ? cap - 1 : 0;
            size_t take = b - a;
            if (k + take > room)
                take = k < room ? room - k : 0;
            memcpy(out + k, t->bytes + a, take);
        }
        k += b - a;
    }
    if (out && cap)
        out[k < cap ? k : cap - 1] = 0;
    return (long) k;
}

void qe_tok_describe(const qe_tok *t, char *buf, size_t n) {
    if (!buf || !n)
        return;
    if (!t) {
        snprintf(buf, n, "no tokenizer");
        return;
    }
    snprintf(buf, n,
             "vocab %ld, pre %s, byte tokens %d/256, merges %ld used / %ld skipped, "
             "special %ld, eos %d bos %d pad %d",
             t->n_vocab, t->pre, t->has_bytes, t->n_merges_used, t->n_merges_skip,
             t->n_special, t->eos, t->bos, t->pad);
}
