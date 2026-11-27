#define _GNU_SOURCE
#include "json.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QE_JSON_MAX_DEPTH 64

struct pstate {
    const char *s;
    size_t n, i;
    int depth;
    char *err;
    size_t errn;
};

static void perr(struct pstate *p, const char *msg) {
    if (p->err && p->errn)
        snprintf(p->err, p->errn, "json: %s at offset %zu", msg, p->i);
}

static void skip_ws(struct pstate *p) {
    while (p->i < p->n) {
        char c = p->s[p->i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            p->i++;
        else
            break;
    }
}

static qe_json *node(struct pstate *p, qe_json_kind k) {
    qe_json *v = calloc(1, sizeof *v);
    if (!v) {
        perr(p, "out of memory");
        return NULL;
    }
    v->kind = k;
    return v;
}

void qe_json_free(qe_json *v) {
    while (v) {
        qe_json *next = v->next;
        qe_json_free(v->child);
        free(v->key);
        free(v->str);
        free(v);
        v = next;
    }
}

/* ---- growable byte buffer for unescaped strings ---- */

typedef struct {
    char *p;
    size_t len, cap;
    int err; /* set by the first failed put; every later one no-ops */
} buf;

static int buf_reserve(buf *b, size_t need) {
    if (b->len + need <= b->cap)
        return 0;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + need)
        cap *= 2;
    char *np = realloc(b->p, cap);
    if (!np)
        return 1;
    b->p = np;
    b->cap = cap;
    return 0;
}

static int buf_put(buf *b, const char *p, size_t n) {
    if (buf_reserve(b, n + 1)) {
        b->err = 1;
        return 1;
    }
    memcpy(b->p + b->len, p, n);
    b->len += n;
    b->p[b->len] = 0;
    return 0;
}

static int buf_putc(buf *b, char c) { return buf_put(b, &c, 1); }

static void buf_puts(buf *b, const char *s, size_t n) { buf_put(b, s, n); }

static void buf_printf(buf *b, const char *fmt, ...) {
    char t[64];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(t, sizeof t, fmt, ap);
    va_end(ap);
    if (n > 0)
        buf_put(b, t, (size_t)n);
}

static int utf8_put(buf *b, unsigned cp) {
    char t[4];
    if (cp < 0x80) {
        t[0] = (char)cp;
        return buf_put(b, t, 1);
    }
    if (cp < 0x800) {
        t[0] = (char)(0xC0 | (cp >> 6));
        t[1] = (char)(0x80 | (cp & 0x3F));
        return buf_put(b, t, 2);
    }
    if (cp < 0x10000) {
        t[0] = (char)(0xE0 | (cp >> 12));
        t[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        t[2] = (char)(0x80 | (cp & 0x3F));
        return buf_put(b, t, 3);
    }
    t[0] = (char)(0xF0 | (cp >> 18));
    t[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    t[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    t[3] = (char)(0x80 | (cp & 0x3F));
    return buf_put(b, t, 4);
}

static int hex4(struct pstate *p, unsigned *out) {
    if (p->i + 4 > p->n)
        return 1;
    unsigned v = 0;
    for (int k = 0; k < 4; k++) {
        char c = p->s[p->i + k];
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (unsigned)(c - 'A' + 10);
        else
            return 1;
    }
    p->i += 4;
    *out = v;
    return 0;
}

/* Parses a string value; p->i is at the opening quote. */
static char *parse_string(struct pstate *p, size_t *out_len) {
    if (p->s[p->i] != '"') {
        perr(p, "expected a string");
        return NULL;
    }
    p->i++;
    buf b = { 0 };
    while (p->i < p->n) {
        unsigned char c = (unsigned char)p->s[p->i];
        if (c == '"') {
            p->i++;
            if (!b.p) /* the empty string */
                b.p = strdup("");
            if (!b.p) {
                perr(p, "out of memory");
                return NULL;
            }
            if (out_len)
                *out_len = b.len;
            return b.p;
        }
        if (c < 0x20) {
            /* Control characters are illegal raw in JSON; accept them rather
             * than failing a request over a stray tab in a prompt. */
            if (buf_putc(&b, (char)c))
                break;
            p->i++;
            continue;
        }
        if (c != '\\') {
            if (buf_putc(&b, (char)c))
                break;
            p->i++;
            continue;
        }
        p->i++;
        if (p->i >= p->n)
            break;
        char e = p->s[p->i++];
        int rc = 0;
        switch (e) {
        case '"': rc = buf_putc(&b, '"'); break;
        case '\\': rc = buf_putc(&b, '\\'); break;
        case '/': rc = buf_putc(&b, '/'); break;
        case 'b': rc = buf_putc(&b, '\b'); break;
        case 'f': rc = buf_putc(&b, '\f'); break;
        case 'n': rc = buf_putc(&b, '\n'); break;
        case 'r': rc = buf_putc(&b, '\r'); break;
        case 't': rc = buf_putc(&b, '\t'); break;
        case 'u': {
            unsigned cp = 0;
            if (hex4(p, &cp)) {
                perr(p, "bad \\u escape");
                free(b.p);
                return NULL;
            }
            if (cp >= 0xD800 && cp <= 0xDBFF && p->i + 1 < p->n && p->s[p->i] == '\\' &&
                p->s[p->i + 1] == 'u') {
                size_t save = p->i;
                p->i += 2;
                unsigned lo = 0;
                if (hex4(p, &lo) || lo < 0xDC00 || lo > 0xDFFF) {
                    p->i = save;
                    lo = 0;
                    cp = 0xFFFD;
                } else {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
            } else if (cp >= 0xD800 && cp <= 0xDFFF) {
                cp = 0xFFFD; /* unpaired surrogate */
            }
            rc = utf8_put(&b, cp);
            break;
        }
        default:
            perr(p, "bad escape");
            free(b.p);
            return NULL;
        }
        if (rc)
            break;
    }
    perr(p, "unterminated string");
    free(b.p);
    return NULL;
}

static qe_json *parse_value(struct pstate *p);

/* Parses the members of an object or array. 0 on success (with *out possibly
 * NULL for an empty container, which is not an error), -1 on failure. */
static int parse_members(struct pstate *p, int object, qe_json **out) {
    qe_json *first = NULL, *last = NULL;
    const char close = object ? '}' : ']';
    for (;;) {
        skip_ws(p);
        if (p->i >= p->n) {
            perr(p, "truncated");
            goto fail;
        }
        if (p->s[p->i] == close) {
            p->i++;
            *out = first;
            return 0;
        }
        char *key = NULL;
        if (object) {
            if (p->i >= p->n || p->s[p->i] != '"') {
                perr(p, "expected an object key");
                goto fail;
            }
            key = parse_string(p, NULL);
            if (!key)
                goto fail;
            skip_ws(p);
            if (p->i >= p->n || p->s[p->i] != ':') {
                free(key);
                perr(p, "expected ':'");
                goto fail;
            }
            p->i++;
        }
        qe_json *v = parse_value(p);
        if (!v) {
            free(key);
            goto fail;
        }
        v->key = key;
        if (last)
            last->next = v;
        else
            first = v;
        last = v;
        skip_ws(p);
        if (p->i < p->n && p->s[p->i] == ',') {
            /* A trailing comma before the close is tolerated. Strictness here
             * buys nothing: the members that were named are the members that
             * are used, and senders do emit the comma. */
            p->i++;
            continue;
        }
        if (p->i < p->n && p->s[p->i] == close) {
            p->i++;
            *out = first;
            return 0;
        }
        perr(p, object ? "expected ',' or '}' in object" : "expected ',' or ']' in array");
        goto fail;
    }
fail:
    qe_json_free(first);
    *out = NULL;
    return -1;
}

static qe_json *parse_value(struct pstate *p) {
    if (p->depth > QE_JSON_MAX_DEPTH) {
        perr(p, "nesting too deep");
        return NULL;
    }
    skip_ws(p);
    if (p->i >= p->n) {
        perr(p, "truncated");
        return NULL;
    }
    char c = p->s[p->i];
    if (c == '{' || c == '[') {
        int object = c == '{';
        p->i++;
        p->depth++;
        qe_json *v = node(p, object ? QE_JSON_OBJ : QE_JSON_ARR);
        if (!v) {
            p->depth--;
            return NULL;
        }
        qe_json *kids = NULL;
        if (parse_members(p, object, &kids)) {
            qe_json_free(v);
            return NULL;
        }
        v->child = kids;
        size_t n = 0;
        for (qe_json *k = kids; k; k = k->next)
            n++;
        v->n_child = n;
        p->depth--;
        return v;
    }
    if (c == '"') {
        size_t len = 0;
        char *s = parse_string(p, &len);
        if (!s)
            return NULL;
        qe_json *v = node(p, QE_JSON_STR);
        if (!v) {
            free(s);
            return NULL;
        }
        v->str = s;
        v->str_len = len;
        return v;
    }
    if (c == 't' || c == 'f' || c == 'n') {
        size_t want = c == 't' ? 4 : (c == 'f' ? 5 : 4);
        if (p->i + want > p->n || strncmp(p->s + p->i, c == 't' ? "true" : (c == 'f' ? "false" : "null"), want)) {
            perr(p, "bad literal");
            return NULL;
        }
        p->i += want;
        qe_json *v = node(p, c == 'n' ? QE_JSON_NULL : QE_JSON_BOOL);
        if (!v)
            return NULL;
        v->boolean = c == 't';
        return v;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        size_t start = p->i;
        while (p->i < p->n && (p->s[p->i] == '-' || p->s[p->i] == '+' || p->s[p->i] == '.' ||
                               p->s[p->i] == 'e' || p->s[p->i] == 'E' ||
                               (p->s[p->i] >= '0' && p->s[p->i] <= '9')))
            p->i++;
        char tmp[64];
        size_t len = p->i - start;
        if (len >= sizeof tmp) {
            perr(p, "number too long");
            return NULL;
        }
        memcpy(tmp, p->s + start, len);
        tmp[len] = 0;
        char *end = NULL;
        double d = strtod(tmp, &end);
        if (end != tmp + len || !isfinite(d)) {
            perr(p, "bad number");
            return NULL;
        }
        qe_json *v = node(p, QE_JSON_NUM);
        if (!v)
            return NULL;
        v->num = d;
        /* The literal as written, because 1.0 and 1 are one double and two different
         * things to a serialiser that has to reproduce Python's (src/chat.c). */
        v->str = malloc(len + 1);
        if (v->str) {
            memcpy(v->str, tmp, len + 1);
            v->str_len = len;
        }
        return v;
    }
    perr(p, "unexpected character");
    return NULL;
}

qe_json *qe_json_parse(const char *s, size_t n, char *err, size_t errn) {
    if (err && errn)
        err[0] = 0;
    if (!s || !n) {
        if (err && errn)
            snprintf(err, errn, "json: empty body");
        return NULL;
    }
    struct pstate p = { .s = s, .n = n, .i = 0, .depth = 0, .err = err, .errn = errn };
    qe_json *v = parse_value(&p);
    if (!v)
        return NULL;
    skip_ws(&p);
    if (p.i != p.n) {
        perr(&p, "trailing data");
        qe_json_free(v);
        return NULL;
    }
    if (v->kind != QE_JSON_OBJ && v->kind != QE_JSON_ARR) {
        if (err && errn)
            snprintf(err, errn, "json: body must be an object");
        qe_json_free(v);
        return NULL;
    }
    return v;
}

const qe_json *qe_json_at(const qe_json *obj, const char *key) {
    if (!obj || obj->kind != QE_JSON_OBJ)
        return NULL;
    for (const qe_json *k = obj->child; k; k = k->next)
        if (k->key && !strcmp(k->key, key))
            return k;
    return NULL;
}

const char *qe_json_str_at(const qe_json *obj, const char *key, const char *dflt) {
    const qe_json *v = qe_json_at(obj, key);
    return (v && v->kind == QE_JSON_STR) ? v->str : dflt;
}

double qe_json_num_at(const qe_json *obj, const char *key, double dflt) {
    const qe_json *v = qe_json_at(obj, key);
    return (v && v->kind == QE_JSON_NUM) ? v->num : dflt;
}

int qe_json_bool_at(const qe_json *obj, const char *key, int dflt) {
    const qe_json *v = qe_json_at(obj, key);
    if (!v)
        return dflt;
    if (v->kind == QE_JSON_BOOL)
        return v->boolean;
    if (v->kind == QE_JSON_NUM)
        return v->num != 0;
    return dflt;
}

size_t qe_json_len(const qe_json *arr) { return arr && arr->kind == QE_JSON_ARR ? arr->n_child : 0; }

/* ----------------------------------------------------------------- writing
 *
 * The model's own chat template renders the tool list with a compact-JSON
 * filter, so the schema block the checkpoint was served with is one JSON object
 * per line, keys in the order the client sent them. Re-emitting the parsed
 * request is therefore the only faithful way to build that block, and it is why
 * a serializer exists next to the parser: anything else (pretty-printing, key
 * sorting, a hand-built subset) changes the bytes the model sees at the front
 * of the context and quietly changes what it emits.
 *
 * The buffer grows to fit and the result is copied out, so a caller can size
 * with cap 0 and one retry, exactly like snprintf. */

static void esc_str(buf *b, const char *s, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  buf_puts(b, "\\\"", 2); break;
        case '\\': buf_puts(b, "\\\\", 2); break;
        case '\n': buf_puts(b, "\\n", 2); break;
        case '\r': buf_puts(b, "\\r", 2); break;
        case '\t': buf_puts(b, "\\t", 2); break;
        default:
            if (c < 0x20 || c == 0x7f)
                buf_printf(b, "\\u%04x", c);
            else
                buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

/* Shortest decimal that reads back as the same double, because a schema that
 * said 0.1 and came back 0.100000001490116 is a schema the client did not send.
 * Whole numbers lose the fraction entirely: an integer-valued bound in a tool
 * schema is an integer. */
static void put_num(buf *b, double d) {
    char t[40];
    if (!(d == d) || d > 1e308 || d < -1e308) {
        buf_puts(b, "null", 4);
        return;
    }
    if (d == floor(d) && d >= -1e15 && d <= 1e15) {
        snprintf(t, sizeof t, "%.0f", d);
        buf_puts(b, t, strlen(t));
        return;
    }
    for (int p = 1; p <= 17; p++) {
        snprintf(t, sizeof t, "%.*g", p, d);
        if (strtod(t, NULL) == d)
            break;
    }
    buf_puts(b, t, strlen(t));
}

static void write_value(buf *b, const qe_json *v) {
    if (!v) {
        buf_puts(b, "null", 4);
        return;
    }
    switch (v->kind) {
    case QE_JSON_NULL: buf_puts(b, "null", 4); break;
    case QE_JSON_BOOL: buf_puts(b, v->boolean ? "true" : "false", v->boolean ? 4 : 5); break;
    case QE_JSON_NUM:  put_num(b, v->num); break;
    case QE_JSON_STR:  esc_str(b, v->str ? v->str : "", v->str_len); break;
    case QE_JSON_ARR: {
        buf_putc(b, '[');
        for (const qe_json *k = v->child; k; k = k->next) {
            if (k != v->child)
                buf_putc(b, ',');
            write_value(b, k);
        }
        buf_putc(b, ']');
        break;
    }
    case QE_JSON_OBJ: {
        buf_putc(b, '{');
        for (const qe_json *k = v->child; k; k = k->next) {
            if (k != v->child)
                buf_putc(b, ',');
            esc_str(b, k->key ? k->key : "", k->key ? strlen(k->key) : 0);
            buf_putc(b, ':');
            write_value(b, k);
        }
        buf_putc(b, '}');
        break;
    }
    }
}

long qe_json_write(const qe_json *v, char *out, size_t cap) {
    buf b = { 0 };
    write_value(&b, v);
    if (b.err) {
        free(b.p);
        return -1;
    }
    /* One terminator, and the length excludes it. There used to be two, so the
     * length counted the first and a caller copying `n` bytes copied a NUL. */
    if (buf_putc(&b, 0)) {
        free(b.p);
        return -1;
    }
    size_t n = b.len - 1;
    if (out && cap) {
        size_t take = n < cap - 1 ? n : cap - 1;
        memcpy(out, b.p, take);
        out[take] = 0;
    }
    free(b.p);
    return (long)n;
}

/* ------------------------------------------------------------------ base64 */

long qe_b64_decode(const char *in, size_t n, unsigned char *out, size_t outcap) {
    static signed char T[256];
    static int once;
    if (!once) {
        const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < 256; i++)
            T[i] = -1;
        for (int i = 0; i < 64; i++)
            T[(unsigned char)A[i]] = (signed char)i;
        /* URL-safe alphabet, and the characters that show up when something
         * wrapped a data URL at 76 columns. */
        T[(unsigned)'-'] = 62;
        T[(unsigned)'_'] = 63;
        once = 1;
    }
    size_t o = 0;
    unsigned acc = 0;
    int bits = 0;
    int pad = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            if (c == '=')
                pad++;
            continue;
        }
        /* Percent-encoding inside a data URL is not decoded here: serve.c
         * rejects those bodies with a message that says so, rather than
         * handing the engine a payload that is 20 percent shorter than the
         * sender claimed. */
        signed char v = T[c];
        if (v < 0)
            return -1;
        if (pad)
            return -1; /* data after padding */
        acc = (acc << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (o >= outcap)
                return -1;
            out[o++] = (unsigned char)((acc >> bits) & 0xFF);
        }
    }
    /* A partial group is legal only with 2 or 3 symbols, and only when the
     * bits that do not form a byte are zero. One symbol ("Y") and a non-zero
     * tail ("YR==") are both corrupt input, not short input. */
    if (bits == 6 || (bits && (acc & ((1u << bits) - 1u))))
        return -1;
    if (pad > 2)
        return -1; /* "====" is not empty input, it is a mistake */
    return (long)o;
}
