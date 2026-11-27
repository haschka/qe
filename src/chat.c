/* See chat.h for what this reproduces and why it is written out rather than
 * interpreted. Each block below cites the template construct it came from.
 *
 * Two style rules with real consequences:
 *
 *   - Control tokens are fetched from the checkpoint's vocabulary by id, and the
 *     plain-text tool tags are assembled from character codes. Neither this file
 *     nor a prompt rendered by it needs to contain a marker sequence, so a
 *     rendered prompt can be logged, printed and pasted without breaking
 *     whatever reads such transcripts.
 *   - Messages are collected into an array first, so the forward lookups the
 *     template makes (previous item, next item, the last real query) are
 *     straight indexing rather than three more nested walks. */
#define _GNU_SOURCE
#include "chat.h"

#include "http.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MARK_MAX 48

typedef struct {
    char turn_open[MARK_MAX];
    char turn_close[MARK_MAX];
    char tool_open[MARK_MAX];
    char tool_close[MARK_MAX];
    char out_open[MARK_MAX];
    char out_close[MARK_MAX];
    char reason_open[MARK_MAX];
    char reason_close[MARK_MAX];
} marks;

static void putc1(qe_str *o, char c) { qe_str_append(o, &c, 1); }

/* The text of one control token, straight from the checkpoint's vocabulary. */
static int mark_text(const qe_tok *t, int id, char *out, size_t n) {
    if (!t || !qe_tok_is_special(t, id))
        return -1;
    long got = qe_tok_decode(t, &id, 1, out, n);
    if (got <= 0)
        return -1;
    return 0;
}

static int marks_load(const qe_tok *t, marks *m) {
    memset(m, 0, sizeof *m);
    return mark_text(t, QE_MARK_TURN_OPEN, m->turn_open, sizeof m->turn_open) ||
           mark_text(t, QE_MARK_TURN_CLOSE, m->turn_close, sizeof m->turn_close) ||
           mark_text(t, QE_MARK_TOOL_OPEN, m->tool_open, sizeof m->tool_open) ||
           mark_text(t, QE_MARK_TOOL_CLOSE, m->tool_close, sizeof m->tool_close) ||
           mark_text(t, QE_MARK_TOOL_OUT_OPEN, m->out_open, sizeof m->out_open) ||
           mark_text(t, QE_MARK_TOOL_OUT_CLOSE, m->out_close, sizeof m->out_close) ||
           mark_text(t, QE_MARK_REASON_OPEN, m->reason_open, sizeof m->reason_open) ||
           mark_text(t, QE_MARK_REASON_CLOSE, m->reason_close, sizeof m->reason_close);
}

int qe_chat_supported(const qe_tok *t) {
    marks m;
    if (!t || qe_tok_n_vocab(t) <= QE_MARK_REASON_CLOSE)
        return 0;
    /* The turn close has to be the file's own end-of-sequence: the decode loop
     * stops on eos, and if that id were not the assistant turn terminator a
     * reply would run on past the end of its own turn. */
    if (qe_tok_eos(t) != QE_MARK_TURN_CLOSE)
        return 0;
    return marks_load(t, &m) == 0;
}

/* --------------------------------------------------------------- small bits */

typedef struct {
    const char *p;
    size_t n;
} slice;

/* Jinja's trim filter: whitespace off both ends, the bytes in between kept. */
static slice trim_slice(const char *p, size_t n) {
    slice s = { p ? p : "", n };
    while (s.n && isspace((unsigned char)s.p[0])) {
        s.p++;
        s.n--;
    }
    while (s.n && isspace((unsigned char)s.p[s.n - 1]))
        s.n--;
    return s;
}

static void put_slice(qe_str *o, slice s) { qe_str_append(o, s.p, s.n); }

/* Plain-text tags, from character codes. The model sees the tag bytes; whether
 * one of them is also a control token is the tokenizer's decision, and folding
 * it to a single id is what the training renderer's output did as well. */
static void put_tag(qe_str *o, const char *name, int closing) {
    char t[96];
    snprintf(t, sizeof t, "%c%s%s%c", (char)60, closing ? "/" : "", name, (char)62);
    qe_str_puts(o, t);
}

static void put_tag_value(qe_str *o, const char *name, const char *val) {
    char t[512];
    snprintf(t, sizeof t, "%c%s=%s%c", (char)60, name, val ? val : "", (char)62);
    qe_str_puts(o, t);
}

/* ------------------------------------------------- JSON a client is handed
 *
 * The argument object of a tool request, as it leaves for the client: ASCII only, with
 * < > & ' escaped as well, which is valid JSON any parser takes and safe in any
 * transport. It is not what the prompt uses - the model was trained on Python's
 * json.dumps, which py_jvalue below reproduces - and the two were once one function,
 * which put a schema block in every prompt that the model had never seen. */

static void put_jstr(qe_str *o, const char *p, size_t n) {
    putc1(o, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)p[i];
        const char *e = NULL;
        switch (c) {
        case '"': e = "\\\""; break;
        case '\\': e = "\\\\"; break;
        case '\n': e = "\\n"; break;
        case '\r': e = "\\r"; break;
        case '\t': e = "\\t"; break;
        case '\b': e = "\\b"; break;
        case '\f': e = "\\f"; break;
        default: break;
        }
        if (e) {
            qe_str_puts(o, e);
            continue;
        }
        if (c == '<') { qe_str_puts(o, "\\u003c"); continue; }
        if (c == '>') { qe_str_puts(o, "\\u003e"); continue; }
        if (c == '&') { qe_str_puts(o, "\\u0026"); continue; }
        if (c == '\'') { qe_str_puts(o, "\\u0027"); continue; }
        if (c < 0x20 || c == 0x7f) {
            qe_str_appendf(o, "\\u%04x", c);
            continue;
        }
        if (c < 0x80) {
            putc1(o, (char)c);
            continue;
        }
        /* ASCII-only output: a codepoint arrives as its escape sequence, and
         * above the plane boundary as the surrogate pair the filter emits. */
        unsigned cp = 0;
        int extra = 0;
        if (c < 0xe0) { cp = c & 0x1f; extra = 1; }
        else if (c < 0xf0) { cp = c & 0x0f; extra = 2; }
        else { cp = c & 0x07; extra = 3; }
        int ok = 1;
        for (int k = 1; k <= extra; k++) {
            if (i + (size_t)k >= n || ((unsigned char)p[i + k] & 0xc0) != 0x80) {
                ok = 0;
                break;
            }
            cp = (cp << 6) | ((unsigned char)p[i + k] & 0x3f);
        }
        if (!ok) {
            qe_str_puts(o, "\\ufffd");
            continue;
        }
        i += (size_t)extra;
        if (cp < 0x10000) {
            qe_str_appendf(o, "\\u%04x", cp);
        } else {
            unsigned hi = 0xd800 + ((cp - 0x10000) >> 10);
            unsigned lo = 0xdc00 + ((cp - 0x10000) & 0x3ff);
            qe_str_appendf(o, "\\u%04x\\u%04x", hi, lo);
        }
    }
    putc1(o, '"');
}

static void put_num(qe_str *o, double d) {
    char t[40];
    if (d != d || d > 1e308 || d < -1e308) {
        qe_str_puts(o, "null");
        return;
    }
    if (d == (double)(long long)d && d >= -1e15 && d <= 1e15) {
        snprintf(t, sizeof t, "%.0f", d);
    } else {
        for (int p = 1; p <= 17; p++) {
            snprintf(t, sizeof t, "%.*g", p, d);
            if (strtod(t, NULL) == d)
                break;
        }
    }
    qe_str_puts(o, t);
}

/* ------------------------------------------- the tojson the model was trained with
 *
 * The template writes every tool, and every non-string argument of a replayed call, with
 * `tojson`. Which `tojson` decides the bytes, and the one that matters is the one the
 * training data was rendered with: transformers replaces Jinja's own filter with
 * json.dumps(x, ensure_ascii=False) - keys in the order they came, ", " and ": " between
 * them, non-ASCII and < > & ' as themselves - and llama.cpp's renderer does the same.
 * Jinja's stock filter (sorted keys, everything escaped, no spaces) is what put_jvalue
 * above reproduces, and it is a schema block the model never saw: every key moved,
 * every space gone, on the one part of every prompt that teaches it its tools. */

static void py_jstr(qe_str *o, const char *p, size_t n) {
    putc1(o, '"');
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)p[i];
        switch (c) {
        case '"': qe_str_puts(o, "\\\""); continue;
        case '\\': qe_str_puts(o, "\\\\"); continue;
        case '\n': qe_str_puts(o, "\\n"); continue;
        case '\r': qe_str_puts(o, "\\r"); continue;
        case '\t': qe_str_puts(o, "\\t"); continue;
        case '\b': qe_str_puts(o, "\\b"); continue;
        case '\f': qe_str_puts(o, "\\f"); continue;
        default: break;
        }
        if (c < 0x20)
            qe_str_appendf(o, "\\u%04x", c);
        else
            putc1(o, (char)c);
    }
    putc1(o, '"');
}

/* Python's repr of a float, which is what json.dumps writes: the shortest digits that
 * read back as the same double, fixed between 1e-4 and 1e16 with a ".0" when the value
 * is whole, and an exponent of at least two digits outside that range. */
static void py_float(qe_str *o, double d) {
    char t[40], m[40];
    if (d != d || d > 1.7976931348623157e308 || d < -1.7976931348623157e308) {
        qe_str_puts(o, d != d ? "NaN" : d > 0 ? "Infinity" : "-Infinity");
        return;
    }
    int p = 1;
    for (; p <= 17; p++) {
        snprintf(t, sizeof t, "%.*e", p - 1, d);
        if (strtod(t, NULL) == d)
            break;
    }
    /* t is d.ddde±XX with p significant digits */
    const char *e = strchr(t, 'e');
    const int exp10 = atoi(e + 1);
    size_t nd = 0;
    const int neg = t[0] == '-';
    for (const char *q = t + neg; q < e; q++)
        if (*q != '.')
            m[nd++] = *q;
    m[nd] = 0;
    if (exp10 < -4 || exp10 >= 16) {
        qe_str_appendf(o, "%s%c", neg ? "-" : "", m[0]);
        if (nd > 1)
            qe_str_appendf(o, ".%s", m + 1);
        qe_str_appendf(o, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
        return;
    }
    if (neg)
        putc1(o, '-');
    if (exp10 < 0) {
        qe_str_puts(o, "0.");
        for (int k = 0; k < -exp10 - 1; k++)
            putc1(o, '0');
        qe_str_puts(o, m);
        return;
    }
    for (int k = 0; k <= exp10; k++)
        putc1(o, k < (int)nd ? m[k] : '0');
    putc1(o, '.');
    if ((int)nd > exp10 + 1)
        qe_str_puts(o, m + exp10 + 1);
    else
        putc1(o, '0');
}

/* A JSON number the way Python reads and writes it back: a literal without a fraction or
 * an exponent is an int and stays its digits; anything else is a float. */
static void py_num(qe_str *o, const qe_json *v) {
    const char *lit = v->str;
    if (lit && !strpbrk(lit, ".eE")) {
        const char *q = lit;
        int neg = 0;
        if (*q == '-') {
            neg = 1;
            q++;
        }
        while (q[0] == '0' && q[1])
            q++;
        if (neg && !(q[0] == '0' && !q[1]))
            putc1(o, '-');
        qe_str_puts(o, q);
        return;
    }
    py_float(o, v->num);
}

static void py_jvalue(qe_str *o, const qe_json *v) {
    if (!v) {
        qe_str_puts(o, "null");
        return;
    }
    switch (v->kind) {
    case QE_JSON_NULL: qe_str_puts(o, "null"); break;
    case QE_JSON_BOOL: qe_str_puts(o, v->boolean ? "true" : "false"); break;
    case QE_JSON_NUM: py_num(o, v); break;
    case QE_JSON_STR: py_jstr(o, v->str ? v->str : "", v->str_len); break;
    case QE_JSON_ARR:
        putc1(o, '[');
        for (const qe_json *k = v->child; k; k = k->next) {
            if (k != v->child)
                qe_str_puts(o, ", ");
            py_jvalue(o, k);
        }
        putc1(o, ']');
        break;
    case QE_JSON_OBJ:
        putc1(o, '{');
        for (const qe_json *k = v->child; k; k = k->next) {
            if (k != v->child)
                qe_str_puts(o, ", ");
            py_jstr(o, k->key ? k->key : "", k->key ? strlen(k->key) : 0);
            qe_str_puts(o, ": ");
            py_jvalue(o, k);
        }
        putc1(o, '}');
        break;
    }
}

/* ---------------------------------------------------------------- the texts
 *
 * Verbatim from the template: the sentence prepended to the system turn for a
 * reasoning effort. xhigh is the default, high means xhigh, medium sends no
 * sentence, low sends its own. */
static const char *EFFORT_XHIGH =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
static const char *EFFORT_LOW =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

#define TOOLS_HEAD "# Tools\n\nYou have access to the following functions:\n\n"

/* The worked example of the call format, which sits between the schemas and the
 * reminder. The delimiters around it are control tokens; the tags inside it are
 * text. */
static void put_example_call(qe_str *o, const marks *m) {
    qe_str_puts(o, m->tool_open);
    qe_str_puts(o, "\n");
    put_tag_value(o, "function", "example_function_name");
    qe_str_puts(o, "\n");
    put_tag_value(o, "parameter", "example_parameter_1");
    qe_str_puts(o, "\nvalue_1\n");
    put_tag(o, "parameter", 1);
    qe_str_puts(o, "\n");
    put_tag_value(o, "parameter", "example_parameter_2");
    qe_str_puts(o, "\nThis is the value for the second parameter\nthat can span\nmultiple lines\n");
    put_tag(o, "parameter", 1);
    qe_str_puts(o, "\n");
    put_tag(o, "function", 1);
    qe_str_puts(o, "\n");
    qe_str_puts(o, m->tool_close);
    qe_str_puts(o, "\n\n");
}

static void put_reminder(qe_str *o, const marks *m) {
    put_tag(o, "IMPORTANT", 0);
    qe_str_puts(o, "\nReminder:\n- Function calls MUST follow the specified format: an inner ");
    put_tag_value(o, "function", "...");
    put_tag(o, "function", 1);
    qe_str_puts(o, " block must be nested within ");
    qe_str_puts(o, m->tool_open);
    qe_str_puts(o, m->tool_close);
    qe_str_puts(o, " XML tags\n- Required parameters MUST be specified\n- You may provide optional "
                   "reasoning for your function call in natural language BEFORE the function call, "
                   "but NOT after\n- If there is no function call available, answer the question "
                   "like normal with your current knowledge and do not tell the user about function "
                   "calls\n");
    put_tag(o, "IMPORTANT", 1);
}

/* ----------------------------------------------------------------- options */

const char *const QE_CHAT_KWBOXES[] = { "chat_template_kwargs", "template_kwargs", "extra_body",
                                        NULL };

static const qe_json *lookup_any(const qe_json *body, const char *key) {
    const qe_json *v = qe_json_at(body, key);
    if (v)
        return v;
    /* Chat-template keyword arguments arrive nested, and clients disagree about
     * what to call the container; the ones in the wild are all flat objects. */
    for (int i = 0; QE_CHAT_KWBOXES[i]; i++) {
        const qe_json *b = qe_json_at(body, QE_CHAT_KWBOXES[i]);
        if (b && b->kind == QE_JSON_OBJ) {
            const qe_json *w = qe_json_at(b, key);
            if (w)
                return w;
        }
    }
    return NULL;
}

static int truthy(const qe_json *v, int dflt) {
    if (!v)
        return dflt;
    if (v->kind == QE_JSON_BOOL)
        return v->boolean;
    if (v->kind == QE_JSON_NUM)
        return v->num != 0;
    if (v->kind == QE_JSON_STR && v->str)
        return strcmp(v->str, "false") && strcmp(v->str, "0") && strcmp(v->str, "off");
    return dflt;
}

/* The text at the very end of the last system turn: a string content, or the last
 * text part of a content array. Nothing else about the message is looked at. */
static const char *system_turn_tail(const qe_json *m) {
    const qe_json *c = qe_json_at(m, "content");
    if (!c)
        return NULL;
    if (c->kind == QE_JSON_STR)
        return c->str;
    if (c->kind == QE_JSON_ARR) {
        const char *last = NULL;
        for (const qe_json *p = c->child; p; p = p->next) {
            const qe_json *t = qe_json_at(p, "text");
            if (t && t->kind == QE_JSON_STR && t->str)
                last = t->str;
        }
        return last;
    }
    return NULL;
}

static int ends_ignore_space(const char *s, const char *tail) {
    if (!s)
        return 0;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r'))
        n--;
    size_t m = strlen(tail);
    return n >= m && !memcmp(s + n - m, tail, m);
}

/* A client that wants no reasoning and has no field to say it with puts this
 * model family's own off-marker at the end of the system prompt. charmbracelet's
 * client does exactly that on every title request, so on that request it is the
 * only signal there is, and taking it means the difference between a title and
 * forty tokens of reasoning. Only the end of a system turn counts, because that is
 * where the convention puts it; the sentence about it in the middle of a prompt is
 * not a request. */
static int prompt_turns_thinking_off(const qe_json *body) {
    const qe_json *msgs = qe_json_at(body, "messages");
    if (!msgs || msgs->kind != QE_JSON_ARR)
        return 0;
    const qe_json *last = NULL;
    for (const qe_json *m = msgs->child; m; m = m->next) {
        const char *role = qe_json_str_at(m, "role", NULL);
        if (role && (!strcmp(role, "system") || !strcmp(role, "developer")))
            last = m;
    }
    if (!last)
        return 0;
    const char *tail = system_turn_tail(last);
    return ends_ignore_space(tail, "/no_think") || ends_ignore_space(tail, "/nothink");
}

int qe_chat_opts_from_body(const qe_json *body, qe_chat_opts *o, char *why, size_t whyn) {
    if (!body || !o)
        return -1;
    o->effort_xhigh = 1;
    o->effort_low = 0;
    o->preserve_thinking = 1;
    o->empty_thinking_blocks = 0; /* the server's switch, never the body's */
    o->vision = 0;                /* and so are these two */
    o->vision_batched = 0;
    o->enable_thinking = 1;
    int spoken = 0; /* the request said something about reasoning in a field */

    const qe_json *th = lookup_any(body, "enable_thinking");
    if (th) {
        spoken = 1;
        o->enable_thinking = truthy(th, 1);
    } else {
        /* The other spelling in the wild: an object with an enabled flag or a
         * type, which is how a client that never heard of this model asks. */
        const qe_json *t = qe_json_at(body, "thinking");
        if (t && t->kind == QE_JSON_OBJ) {
            if (qe_json_at(t, "enabled")) {
                spoken = 1;
                o->enable_thinking = truthy(qe_json_at(t, "enabled"), 1);
            }
            const char *type = qe_json_str_at(t, "type", NULL);
            if (type && !strcmp(type, "disabled")) {
                spoken = 1;
                o->enable_thinking = 0;
            }
        }
    }

    const qe_json *eff = lookup_any(body, "reasoning_effort");
    if (eff && eff->kind == QE_JSON_STR && eff->str && *eff->str) {
        spoken = 1;
        /* "max" is the newest client spelling of the top of the scale (fantasy has it,
         * so crush can send it); this model's top is xhigh, so that is what it gets
         * rather than a refusal of the whole turn. */
        if (!strcmp(eff->str, "high") || !strcmp(eff->str, "xhigh") ||
            !strcmp(eff->str, "max")) {
            o->effort_xhigh = 1;
        } else if (!strcmp(eff->str, "medium")) {
            o->effort_xhigh = 0;
        } else if (!strcmp(eff->str, "low") || !strcmp(eff->str, "minimal")) {
            /* "minimal" is another client's word for the shortest thing on offer,
             * and the shortest thing on offer is the low-effort sentence. */
            o->effort_xhigh = 0;
            o->effort_low = 1;
        } else if (!strcmp(eff->str, "none")) {
            o->effort_xhigh = 0;
            o->enable_thinking = 0;
        } else {
            snprintf(why, whyn,
                     "reasoning_effort '%s' is not supported; this model takes xhigh (default; "
                     "high and max mean the same), medium, low, minimal, and none",
                     eff->str);
            return -1;
        }
    }

    /* Nothing in the request spoke about reasoning as such, so the prompt is the
     * only place it could have been asked for. */
    if (!spoken && prompt_turns_thinking_off(body))
        o->enable_thinking = 0;

    const qe_json *pt = lookup_any(body, "preserve_thinking");
    if (pt)
        o->preserve_thinking = truthy(pt, 1);
    return 0;
}

/* --------------------------------------------------------------- rendering */

static int is_system_role(const char *role) {
    return role && (!strcmp(role, "system") || !strcmp(role, "developer"));
}

const char *qe_chat_role(const qe_json *m) {
    const char *role = qe_json_str_at(m, "role", NULL);
    return role ? role : "user";
}

static const qe_json **collect(const qe_json *arr, size_t *n) {
    size_t count = qe_json_len(arr);
    const qe_json **v = malloc((count + 1) * sizeof *v); /* +1: NULL at the end */
    if (!v)
        return NULL;
    size_t i = 0;
    for (const qe_json *k = arr->child; k; k = k->next)
        v[i++] = k;
    v[i] = NULL;
    *n = count;
    return v;
}

int qe_chat_record_image(qe_chat *r, const char *url, char *why, size_t whyn) {
    if (!r || !url)
        return -1;
    if (strncmp(url, "data:", 5)) {
        snprintf(why, whyn,
                 "image_url must be a base64 data: URL; this build does not fetch http(s) "
                 "image URLs");
        return -1;
    }
    const char *comma = strchr(url, ',');
    if (!comma) {
        snprintf(why, whyn, "malformed data URL (no comma)");
        return -1;
    }
    if (!strcasestr(url, ";base64")) {
        snprintf(why, whyn, "only ;base64 data URLs are supported");
        return -1;
    }
    const char *m = url + 5, *me = comma;
    const char *semi = memchr(url + 5, ';', (size_t)(comma - (url + 5)));
    if (semi)
        me = semi;
    snprintf(r->image_mime[r->n_images], QE_CHAT_MIME_MAX, "%.*s", (int)(me - m), m);
    if (!r->image_mime[r->n_images][0])
        snprintf(r->image_mime[r->n_images], QE_CHAT_MIME_MAX, "application/octet-stream");

    const char *data = comma + 1;
    size_t dlen = strlen(data);
    size_t cap = dlen / 4 * 3 + 4;
    if (cap > QE_CHAT_MAX_IMAGE_BYTES) {
        snprintf(why, whyn, "image over %d MiB", QE_CHAT_MAX_IMAGE_BYTES >> 20);
        return -1;
    }
    unsigned char *bytes = malloc(cap);
    if (!bytes) {
        snprintf(why, whyn, "out of memory decoding the image");
        return -1;
    }
    long n = qe_b64_decode(data, dlen, bytes, cap);
    if (n < 0) {
        free(bytes);
        snprintf(why, whyn, "image is not valid base64");
        return -1;
    }
    r->image_buf[r->n_images] = (char *)bytes;
    r->images[r->n_images].bytes = bytes;
    r->images[r->n_images].len = (size_t)n;
    r->images[r->n_images].mime = r->image_mime[r->n_images];
    r->image_bytes += (size_t)n;
    r->n_images++;
    return 0;
}

/* One message's content, string or parts. The placeholder is what a text-only
 * engine can honestly say about a base64 PNG: there is no vision encoder here,
 * so emitting the vision pad ids would claim context that carries nothing. */
static int put_content(qe_str *o, const qe_json *m, qe_chat *r, char *why, size_t whyn) {
    const qe_json *content = qe_json_at(m, "content");
    if (!content || content->kind == QE_JSON_NULL)
        return 0;
    if (content->kind == QE_JSON_STR) {
        put_slice(o, (slice){ content->str, content->str_len });
        return 0;
    }
    if (content->kind != QE_JSON_ARR) {
        snprintf(why, whyn, "message content must be a string or an array of parts");
        return -1;
    }
    for (const qe_json *part = content->child; part; part = part->next) {
        const char *type = qe_json_str_at(part, "type", "");
        const char *text = qe_json_str_at(part, "text", NULL);
        if (text && !*type)
            type = "text";
        if (!strcmp(type, "text")) {
            if (text)
                put_slice(o, (slice){ text, strlen(text) });
            continue;
        }
        if (!strcmp(type, "image_url") || !strcmp(type, "image") ||
            !strcmp(type, "input_image")) {
            if (r->vision && (!strcmp(qe_chat_role(m), "system") ||
                              !strcmp(qe_chat_role(m), "developer"))) {
                snprintf(why, whyn, "System message cannot contain images.");
                return -1;
            }
            const qe_json *iu = qe_json_at(part, "image_url");
            const char *url = iu ? qe_json_str_at(iu, "url", "") : NULL;
            if (iu && iu->kind == QE_JSON_STR)
                url = iu->str; /* some clients send a bare string */
            if (!url || !*url) {
                snprintf(why, whyn, "image part has no url");
                return -1;
            }
            if (r->image_seen++ < r->image_skip || r->n_images >= QE_CHAT_MAX_IMAGES) {
                r->truncated_images = 1;
                qe_str_puts(o, "[an earlier image, left out: this request carries more "
                               "images than the server takes]");
                continue;
            }
            if (qe_chat_record_image(r, url, why, whyn))
                return -1;
            if (r->vision)
                qe_str_puts(o, r->image_mark);
            else
                qe_str_appendf(o, "[image %d: %s, %zu bytes. %s]", r->n_images,
                               r->image_mime[r->n_images - 1], r->images[r->n_images - 1].len,
                               r->image_note ? r->image_note : QE_CHAT_IMAGE_NOTE);
            continue;
        }
        if (!strcmp(type, "video") || !strcmp(type, "video_url")) {
            snprintf(why, whyn, "this build is text-only: video parts are not supported");
            return -1;
        }
        snprintf(why, whyn, "unsupported content part type '%s'", type);
        return -1;
    }
    return 0;
}

/* One parameter of a tool request: its name tag, the value on its own line, the
 * closing tag. Strings go through as text (that is what the model was trained
 * to write); anything else was serialised, so it is serialised the same way. */
static void put_parameter(qe_str *o, const char *name, const qe_json *value) {
    put_tag_value(o, "parameter", name);
    qe_str_puts(o, "\n");
    if (value && value->kind == QE_JSON_STR)
        put_slice(o, (slice){ value->str, value->str_len });
    else
        py_jvalue(o, value);
    qe_str_puts(o, "\n");
    put_tag(o, "parameter", 1);
    qe_str_puts(o, "\n");
}

/* The tool requests inside an assistant turn. The template accepts the flat and
 * the nested spelling of a request, and - against its own exception - a JSON
 * string of arguments, because that string is the only spelling an OpenAI
 * client has; parsing it back into an object is what makes a coding agent's
 * second turn render at all. */
static int put_tool_calls(qe_str *o, const qe_json *m, const marks *mk, size_t content_len,
                          char *why, size_t whyn) {
    const qe_json *calls = qe_json_at(m, "tool_calls");
    if (!calls || calls->kind != QE_JSON_ARR)
        return 0;
    for (const qe_json *c = calls->child; c; c = c->next) {
        const qe_json *fn = qe_json_at(c, "function");
        const qe_json *src = fn && fn->kind == QE_JSON_OBJ ? fn : c;
        const char *name = qe_json_str_at(src, "name", NULL);
        if (!name || !*name) {
            snprintf(why, whyn, "tool call is missing a function name");
            return -1;
        }
        /* The first request is separated from the answer text by a blank line and
         * follows the reasoning block directly when there is no answer text, later ones
         * by a single newline - the template's `if content|trim`. It used to put a
         * newline there in the empty case too, which is every replayed tool turn of an
         * agent that writes no text before its calls: "\n\n\n" after the reasoning,
         * dozens of times a prompt, where the model was trained on "\n\n". */
        qe_str_puts(o, c == calls->child ? (content_len ? "\n\n" : "") : "\n");
        qe_str_puts(o, mk->tool_open);
        qe_str_puts(o, "\n");
        put_tag_value(o, "function", name);
        qe_str_puts(o, "\n");

        const qe_json *args = qe_json_at(src, "arguments");
        qe_json *parsed = NULL;
        if (args && args->kind == QE_JSON_STR) {
            slice s = trim_slice(args->str, args->str_len);
            if (s.n) {
                char jerr[128] = { 0 };
                parsed = qe_json_parse(s.p, s.n, jerr, sizeof jerr);
                if (!parsed || parsed->kind != QE_JSON_OBJ) {
                    qe_json_free(parsed);
                    snprintf(why, whyn, "tool call arguments for '%s' are not a JSON object: %s",
                             name, jerr[0] ? jerr : "expected an object");
                    return -1;
                }
                args = parsed;
            } else {
                args = NULL;
            }
        } else if (args && args->kind != QE_JSON_OBJ && args->kind != QE_JSON_NULL) {
            snprintf(why, whyn, "tool call arguments for '%s' must be an object", name);
            return -1;
        }
        for (const qe_json *kv = args ? args->child : NULL; kv; kv = kv->next)
            put_parameter(o, kv->key, kv);
        qe_json_free(parsed);

        put_tag(o, "function", 1);
        qe_str_puts(o, "\n");
        qe_str_puts(o, mk->tool_close);
    }
    return 0;
}

/* True when a message's text already is a tool-result block, which is how the
 * template tells a real follow-up question apart from a tool round trip. */
static int is_tool_response_text(const qe_json *m, const marks *mk) {
    const qe_json *c = qe_json_at(m, "content");
    if (!c || c->kind != QE_JSON_STR)
        return 0;
    slice s = trim_slice(c->str, c->str_len);
    size_t ol = strlen(mk->out_open), cl = strlen(mk->out_close);
    return s.n > ol + cl && !memcmp(s.p, mk->out_open, ol) &&
           !memcmp(s.p + s.n - cl, mk->out_close, cl);
}

int qe_chat_render(qe_tok *t, const qe_json *body, const qe_chat_opts *o, qe_chat *r, char *why,
                   size_t whyn) {
    marks mk;
    if (!t || !body || !o || !r)
        return -1;
    if (marks_load(t, &mk)) {
        snprintf(why, whyn,
                 "the checkpoint's vocabulary does not have this model's control tokens where the "
                 "model expects them");
        return -1;
    }
    const qe_json *messages = qe_json_at(body, "messages");
    if (!messages || messages->kind != QE_JSON_ARR || !messages->n_child) {
        snprintf(why, whyn, "messages must be a non-empty array");
        return -1;
    }
    size_t nm = 0;
    const qe_json **msg = collect(messages, &nm);
    if (!msg) {
        snprintf(why, whyn, "out of memory rendering the transcript");
        return -1;
    }
    r->chat = 1;
    r->n_messages = (long)nm;
    r->image_note = o->vision_batched ? QE_CHAT_IMAGE_NOTE_BATCH : QE_CHAT_IMAGE_NOTE;
    if (o->vision) {
        char vo[MARK_MAX], vc[MARK_MAX], vp[MARK_MAX];
        if (!mark_text(t, QE_MARK_VISION_OPEN, vo, sizeof vo) &&
            !mark_text(t, QE_MARK_VISION_CLOSE, vc, sizeof vc) &&
            !mark_text(t, QE_MARK_IMAGE_PAD, vp, sizeof vp)) {
            snprintf(r->image_mark, sizeof r->image_mark, "%s%s%s", vo, vp, vc);
            r->vision = 1;
        }
    }

    /* The oldest images over the limit are the ones left out, so count them first, and
     * leave them out in whole steps (QE_CHAT_IMAGE_DROP_STEP): the count left out moves
     * once a step rather than at every new image, so the prompts in between agree with
     * each other from the front and the turn cache can go back into them. */
    {
        int total = 0;
        for (size_t i = 0; i < nm; i++) {
            const qe_json *c = qe_json_at(msg[i], "content");
            if (!c || c->kind != QE_JSON_ARR)
                continue;
            for (const qe_json *part = c->child; part; part = part->next) {
                const char *ty = qe_json_str_at(part, "type", "");
                total += !strcmp(ty, "image_url") || !strcmp(ty, "image") ||
                         !strcmp(ty, "input_image");
            }
        }
        const int over = total - QE_CHAT_MAX_IMAGES, step = QE_CHAT_IMAGE_DROP_STEP;
        r->image_skip = over > 0 ? (over + step - 1) / step * step : 0;
    }

    /* ---- the leading system run, trimmed and joined (template: sysns walk) */
    qe_str sys = { 0 };
    size_t num_sys = 0;
    while (num_sys < nm && is_system_role(qe_chat_role(msg[num_sys]))) {
        qe_str one = { 0 };
        int rc = put_content(&one, msg[num_sys], r, why, whyn);
        if (!rc) {
            slice s = trim_slice(one.p, one.len);
            if (s.n) {
                if (sys.len)
                    putc1(&sys, '\n');
                qe_str_append(&sys, s.p, s.n);
            }
        }
        qe_str_free(&one);
        if (rc) {
            qe_str_free(&sys);
            free(msg);
            return -1;
        }
        num_sys++;
    }

    const char *instr = "";
    size_t instr_len = 0;
    if (o->enable_thinking && o->effort_xhigh) {
        instr = EFFORT_XHIGH;
        instr_len = strlen(instr);
    } else if (o->enable_thinking && o->effort_low) {
        instr = EFFORT_LOW;
        instr_len = strlen(instr);
    }

    /* ---- the schema preamble, or the system turn on its own ---- */
    const qe_json *tools = qe_json_at(body, "tools");
    int have_tools = tools && tools->kind == QE_JSON_ARR && tools->n_child;
    if (have_tools) {
        r->tools_rendered = 1;
        qe_str_puts(&r->prompt, mk.turn_open);
        qe_str_puts(&r->prompt, "system\n");
        if (instr_len) {
            qe_str_append(&r->prompt, instr, instr_len);
            qe_str_puts(&r->prompt, "\n\n");
        }
        qe_str_puts(&r->prompt, TOOLS_HEAD);
        put_tag(&r->prompt, "tools", 0);
        for (const qe_json *tool = tools->child; tool; tool = tool->next) {
            qe_str_puts(&r->prompt, "\n");
            py_jvalue(&r->prompt, tool);
        }
        qe_str_puts(&r->prompt, "\n");
        put_tag(&r->prompt, "tools", 1);
        qe_str_puts(&r->prompt,
                    "\n\nIf you choose to call a function ONLY reply in the following format with "
                    "NO suffix:\n\n");
        put_example_call(&r->prompt, &mk);
        put_reminder(&r->prompt, &mk);
        if (sys.len) {
            qe_str_puts(&r->prompt, "\n\n");
            qe_str_append(&r->prompt, sys.p, sys.len);
        }
        qe_str_puts(&r->prompt, mk.turn_close);
        qe_str_puts(&r->prompt, "\n");
    } else if (sys.len) {
        qe_str_puts(&r->prompt, mk.turn_open);
        qe_str_puts(&r->prompt, "system\n");
        if (instr_len) {
            qe_str_append(&r->prompt, instr, instr_len);
            qe_str_puts(&r->prompt, "\n\n");
        }
        qe_str_append(&r->prompt, sys.p, sys.len);
        qe_str_puts(&r->prompt, mk.turn_close);
        qe_str_puts(&r->prompt, "\n");
    } else if (instr_len) {
        qe_str_puts(&r->prompt, mk.turn_open);
        qe_str_puts(&r->prompt, "system\n");
        qe_str_append(&r->prompt, instr, instr_len);
        qe_str_puts(&r->prompt, mk.turn_close);
        qe_str_puts(&r->prompt, "\n");
    }
    qe_str_free(&sys);

    /* ---- the last real query, and how far reasoning is replayed ----
     * Assistant turns after it always keep their reasoning block; earlier ones
     * keep it unless the caller turned preserve_thinking off. The default keeps
     * everything, which is what a coding agent wants: the model sees its own
     * earlier reasoning instead of redoing it. What is kept is the block a turn
     * actually has - a turn with no reasoning text in it renders no block, which
     * is the other half of this rule and is explained where the block is written. */
    size_t last_query = nm - 1;
    for (size_t i = nm; i-- > 0;) {
        if (strcmp(qe_chat_role(msg[i]), "user"))
            continue;
        if (is_tool_response_text(msg[i], &mk))
            continue;
        last_query = i;
        break;
    }

    /* ---- turns ---- */
    for (size_t i = 0; i < nm; i++) {
        if (i < num_sys)
            continue;
        const char *role = qe_chat_role(msg[i]);

        if (is_system_role(role)) {
            snprintf(why, whyn, "system message must be at the beginning");
            free(msg);
            return -1;
        }

        if (!strcmp(role, "user")) {
            qe_str one = { 0 };
            if (put_content(&one, msg[i], r, why, whyn)) {
                qe_str_free(&one);
                free(msg);
                return -1;
            }
            slice s = trim_slice(one.p, one.len);
            qe_str_puts(&r->prompt, mk.turn_open);
            qe_str_puts(&r->prompt, "user\n");
            qe_str_append(&r->prompt, s.p, s.n);
            qe_str_puts(&r->prompt, mk.turn_close);
            qe_str_puts(&r->prompt, "\n");
            qe_str_free(&one);
            continue;
        }

        if (!strcmp(role, "assistant")) {
            qe_str one = { 0 };
            if (put_content(&one, msg[i], r, why, whyn)) {
                qe_str_free(&one);
                free(msg);
                return -1;
            }
            slice content = trim_slice(one.p, one.len);
            const qe_json *rc = qe_json_at(msg[i], "reasoning_content");
            slice reason = { "", 0 };
            if (rc && rc->kind == QE_JSON_STR)
                reason = trim_slice(rc->str, rc->str_len);
            if (reason.n)
                r->has_reasoning = 1;

            qe_str_puts(&r->prompt, mk.turn_open);
            qe_str_puts(&r->prompt, "assistant\n");
            /* An empty block is a claim, not a spacer. It is the spelling the
             * model was trained on for "this assistant thought of nothing", so a
             * turn whose reasoning never came back - which is every turn of a
             * client that stores thinking for its own UI and does not resend it -
             * would otherwise be a demonstration of not thinking, forty times over
             * on a long coding session. Render the block when there is thinking in
             * it, and when the operator asked for the template-faithful empty one.
             * And when this request does not think: then not thinking is what the
             * turns were, and the empty block is what the generation prompt of the
             * request before held - without it the next prompt parts from that one
             * four tokens before its end, and no turn ever goes back to its own
             * rewind point, so every one of them reads the whole conversation. */
            int keep_reasoning = o->preserve_thinking || i > last_query;
            if (keep_reasoning &&
                (reason.n || o->empty_thinking_blocks || !o->enable_thinking)) {
                qe_str_puts(&r->prompt, mk.reason_open);
                qe_str_puts(&r->prompt, "\n");
                qe_str_append(&r->prompt, reason.p, reason.n);
                qe_str_puts(&r->prompt, "\n");
                qe_str_puts(&r->prompt, mk.reason_close);
                qe_str_puts(&r->prompt, "\n\n");
            }
            qe_str_append(&r->prompt, content.p, content.n);
            if (put_tool_calls(&r->prompt, msg[i], &mk, content.n, why, whyn)) {
                qe_str_free(&one);
                free(msg);
                return -1;
            }
            qe_str_puts(&r->prompt, mk.turn_close);
            qe_str_puts(&r->prompt, "\n");
            qe_str_free(&one);
            continue;
        }

        if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            /* A run of results shares one user turn: it opens before the first
             * and closes after the last. */
            qe_str one = { 0 };
            if (put_content(&one, msg[i], r, why, whyn)) {
                qe_str_free(&one);
                free(msg);
                return -1;
            }
            slice s = trim_slice(one.p, one.len);
            if (i == 0 || strcmp(qe_chat_role(msg[i - 1]), "tool")) {
                qe_str_puts(&r->prompt, mk.turn_open);
                qe_str_puts(&r->prompt, "user");
            }
            qe_str_puts(&r->prompt, "\n");
            qe_str_puts(&r->prompt, mk.out_open);
            qe_str_puts(&r->prompt, "\n");
            qe_str_append(&r->prompt, s.p, s.n);
            qe_str_puts(&r->prompt, "\n");
            qe_str_puts(&r->prompt, mk.out_close);
            if (i + 1 == nm || strcmp(qe_chat_role(msg[i + 1]), "tool")) {
                qe_str_puts(&r->prompt, mk.turn_close);
                qe_str_puts(&r->prompt, "\n");
            }
            qe_str_free(&one);
            continue;
        }

        snprintf(why, whyn, "unexpected message role '%s'", role);
        free(msg);
        return -1;
    }

    /* ---- the generation prompt: it is the model's turn, and it starts inside
     * the reasoning block. Asked not to think, the block is opened and closed
     * empty, which is how the model was trained to suppress reasoning rather
     * than simply not being offered one. */
    qe_str_puts(&r->prompt, mk.turn_open);
    qe_str_puts(&r->prompt, "assistant\n");
    if (!o->enable_thinking) {
        qe_str_puts(&r->prompt, mk.reason_open);
        qe_str_puts(&r->prompt, "\n\n");
        qe_str_puts(&r->prompt, mk.reason_close);
        qe_str_puts(&r->prompt, "\n\n");
    } else {
        qe_str_puts(&r->prompt, mk.reason_open);
        qe_str_puts(&r->prompt, "\n");
        r->has_reasoning = 1;
        r->opens_reasoning = 1;
    }

    free(msg);
    if (r->prompt.err) {
        snprintf(why, whyn, "out of memory rendering the transcript");
        return -1;
    }
    return 0;
}

void qe_chat_free(qe_chat *r) {
    if (!r)
        return;
    qe_str_free(&r->prompt);
    for (int i = 0; i < r->n_images; i++)
        free(r->image_buf[i]);
    memset(r, 0, sizeof *r);
}

/* --------------------------------------------------------------- turn parser
 *
 * Only control tokens change state, so nothing a model writes inside its answer
 * can be mistaken for the start of a reasoning block or a tool request: those
 * are ids, and an id has no bytes to collide with. The inner tags of a request
 * are text, so they are parsed as text - inside a request block, which is
 * itself delimited by ids. See chat.h for why the block is reported whole. */

enum { ST_REASON, ST_ANSWER, ST_BLOCK, ST_ECHO };

struct qe_turn {
    qe_turn_cb cb;
    void *user;
    int state;
    int calls;    /* requests reported, which is the index the next one takes */
    int finished; /* the assistant turn closed on its own */
    long reason_bytes;
    char **ids;   /* one synthesised call id per request, stable for the reply */
    char id_prefix[48]; /* the stem those ids are built from */
    qe_str block; /* text of the request being buffered */
    qe_str blank; /* answer text that is only spaces, held until it has a side */
    const qe_json *tools; /* the request's tools[], borrowed; types the arguments */
    qe_str reasoning;     /* the reasoning text, kept for the rescue at the close */
    long content_bytes;   /* answer text reported so far */
    int announced; /* the request in the block has been opened to the client */
    int aborted;   /* one was opened and its block never closed */
};

qe_turn *qe_turn_new(qe_turn_cb cb, void *user, int starts_in_reasoning) {
    qe_turn *t = calloc(1, sizeof *t);
    if (!t)
        return NULL;
    t->cb = cb;
    t->user = user;
    t->state = starts_in_reasoning ? ST_REASON : ST_ANSWER;
    snprintf(t->id_prefix, sizeof t->id_prefix, "%s", "call");
    return t;
}

void qe_turn_set_id_prefix(qe_turn *t, const char *prefix) {
    if (!t || !prefix || !prefix[0])
        return;
    snprintf(t->id_prefix, sizeof t->id_prefix, "%s", prefix);
}

void qe_turn_set_tools(qe_turn *t, const qe_json *tools) {
    if (t)
        t->tools = tools && tools->kind == QE_JSON_ARR ? tools : NULL;
}

void qe_turn_free(qe_turn *t) {
    if (!t)
        return;
    for (int i = 0; i < t->calls; i++)
        free(t->ids[i]);
    free(t->ids);
    qe_str_free(&t->block);
    qe_str_free(&t->blank);
    qe_str_free(&t->reasoning);
    free(t);
}

static int emit(qe_turn *t, const qe_turn_ev *ev) {
    return t->cb(t->user, ev);
}

static int emit_text(qe_turn *t, int kind, const char *utf8, int nbytes) {
    if (nbytes <= 0)
        return 0;
    qe_turn_ev ev = { .kind = kind, .utf8 = utf8, .nbytes = nbytes };
    if (kind == QE_TURN_REASONING) {
        t->reason_bytes += (long)nbytes;
        qe_str_append(&t->reasoning, utf8, (size_t)nbytes);
    } else if (kind == QE_TURN_CONTENT) {
        t->content_bytes += (long)nbytes;
    }
    return emit(t, &ev);
}

static int is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/* Answer text, with the blank run between an answer and a request held back.
 * The template separates the two with an empty line, and a client that is
 * showing an answer while a request is being made should not be shown that
 * line: it is punctuation between the two, not the beginning of either. So a
 * run of spaces is buffered until it is followed by text, and released then;
 * a turn that goes on to ask for something never sees it at all. */
static int emit_answer(qe_turn *t, const char *utf8, int nbytes) {
    if (!t->blank.len && !is_space((unsigned char)utf8[0]))
        return emit_text(t, QE_TURN_CONTENT, utf8, nbytes);
    int rc = qe_str_append(&t->blank, utf8, (size_t)nbytes);
    if (rc)
        return rc;
    size_t used = 0;
    while (used < t->blank.len && is_space((unsigned char)t->blank.p[used]))
        used++;
    if (used == t->blank.len)
        return 0;
    rc = emit_text(t, QE_TURN_CONTENT, t->blank.p, (int)t->blank.len);
    t->blank.len = 0;
    return rc;
}

/* The held-back blank run, released only if the turn turned out to be an answer
 * rather than a request. */
static int flush_blank(qe_turn *t, int keep) {
    if (!t->blank.len)
        return 0;
    int rc = 0;
    if (keep)
        rc = emit_text(t, QE_TURN_CONTENT, t->blank.p, (int)t->blank.len);
    t->blank.len = 0;
    return rc;
}

/* ------------------------------------------------------------ tag scanning
 *
 * The tags of a request block, rebuilt for scanning rather than for output, so
 * that this file still contains no tag sequence. find_open returns whatever
 * stands between an equals sign and the tag's closing character, which is a
 * function's name in one case and a parameter's name in the other. */

static int find_open(const char *p, size_t n, size_t from, const char *name, slice *val,
                     size_t *at, size_t *after) {
    size_t nl = strlen(name);
    for (size_t i = from; i + nl + 1 < n; i++) {
        if (p[i] != (char)60 || memcmp(p + i + 1, name, nl))
            continue;
        size_t j = i + 1 + nl;
        if (j >= n)
            return 1;
        if (p[j] == (char)62) { /* a bare tag carries no value */
            val->p = "";
            val->n = 0;
            if (at)
                *at = i;
            *after = j + 1;
            return 0;
        }
        if (p[j] != (char)61)
            continue; /* a closing tag, or something else with this prefix */
        size_t k = j + 1;
        while (k < n && p[k] != (char)62)
            k++;
        if (k >= n)
            return 1;
        *val = trim_slice(p + j + 1, k - (j + 1));
        if (at)
            *at = i;
        *after = k + 1;
        return 0;
    }
    return 1;
}

static int find_close(const char *p, size_t n, size_t from, const char *name, size_t *at,
                      size_t *after) {
    char needle[192];
    int len = snprintf(needle, sizeof needle, "%c/%s%c", (char)60, name, (char)62);
    size_t nl = (size_t)len;
    if (len <= 0 || (size_t)len >= sizeof needle || from + nl > n)
        return 1;
    for (size_t i = from; i + nl <= n; i++)
        if (!memcmp(p + i, needle, nl)) {
            *at = i;
            *after = i + nl;
            return 0;
        }
    return 1;
}

/* One value of a request, put back as JSON. What the model wrote between the
 * tags is text; the argument list a client receives is JSON. A value that
 * already reads as a JSON object, array, number, or literal keeps the bytes the
 * model wrote - which is what the template's own round trip would have produced
 * - and anything else becomes a string, so a command that happens to start with
 * a digit stays the command it is. */
/* A value that already reads as a JSON number: an optional sign, digits, and an
 * optional fraction and exponent, and nothing else. "3 files" is a command that
 * happens to start with a digit, which is why this is a grammar and not a
 * strspn of digits. */
static int is_number(const char *p, size_t n) {
    size_t i = 0;
    if (i < n && p[i] == '-')
        i++;
    size_t start = i;
    while (i < n && p[i] >= '0' && p[i] <= '9')
        i++;
    if (i == start)
        return 0;
    /* JSON has no leading zeros: 001 is a shell id, and written as a number it is
     * a document no strict parser takes. */
    if (i - start > 1 && p[start] == '0')
        return 0;
    if (i < n && p[i] == '.') {
        i++;
        start = i;
        while (i < n && p[i] >= '0' && p[i] <= '9')
            i++;
        if (i == start)
            return 0;
    }
    if (i < n && (p[i] == 'e' || p[i] == 'E')) {
        i++;
        if (i < n && (p[i] == '+' || p[i] == '-'))
            i++;
        start = i;
        while (i < n && p[i] >= '0' && p[i] <= '9')
            i++;
        if (i == start)
            return 0;
    }
    return i == n;
}

static int is_word(const char *p, size_t n, const char *word) {
    return n == strlen(word) && !memcmp(p, word, n);
}

/* What the request's schema says a parameter is. Only "a string and nothing else"
 * changes how a value is read; every other answer, including none, reads it as
 * JSON first. */
enum { ARG_UNKNOWN, ARG_STRING, ARG_JSON };

static int type_is(const qe_json *ty, const char *name) {
    return ty && ty->kind == QE_JSON_STR && ty->str && !strcmp(ty->str, name);
}

static int schema_kind(const qe_json *prop) {
    if (!prop || prop->kind != QE_JSON_OBJ)
        return ARG_UNKNOWN;
    const qe_json *ty = qe_json_at(prop, "type");
    if (ty && ty->kind == QE_JSON_STR)
        return type_is(ty, "string") ? ARG_STRING : ARG_JSON;
    if (ty && ty->kind == QE_JSON_ARR) {
        /* ["string", "null"] is a string that may be left out; a union with any
         * other type is a value the text has to decide. */
        int str = 0, other = 0;
        for (const qe_json *k = ty->child; k; k = k->next) {
            if (type_is(k, "string"))
                str = 1;
            else if (!type_is(k, "null"))
                other = 1;
        }
        return str && !other ? ARG_STRING : ARG_UNKNOWN;
    }
    const qe_json *en = qe_json_at(prop, "enum");
    if (en && en->kind == QE_JSON_ARR && en->child) {
        for (const qe_json *k = en->child; k; k = k->next)
            if (k->kind != QE_JSON_STR)
                return ARG_UNKNOWN;
        return ARG_STRING;
    }
    return ARG_UNKNOWN;
}

/* A tool's parameter schema, found by the function name the model wrote. Both the
 * nested {type, function:{name, parameters}} spelling and the flat one are read. */
static int param_kind(const qe_json *tools, slice fname, slice pname) {
    if (!tools)
        return ARG_UNKNOWN;
    for (const qe_json *tool = tools->child; tool; tool = tool->next) {
        const qe_json *fn = qe_json_at(tool, "function");
        const qe_json *src = fn && fn->kind == QE_JSON_OBJ ? fn : tool;
        const char *name = qe_json_str_at(src, "name", NULL);
        if (!name || strlen(name) != fname.n || memcmp(name, fname.p, fname.n))
            continue;
        const qe_json *props = qe_json_at(qe_json_at(src, "parameters"), "properties");
        if (!props || props->kind != QE_JSON_OBJ)
            return ARG_UNKNOWN;
        for (const qe_json *pr = props->child; pr; pr = pr->next)
            if (pr->key && strlen(pr->key) == pname.n && !memcmp(pr->key, pname.p, pname.n))
                return schema_kind(pr);
        return ARG_UNKNOWN;
    }
    return ARG_UNKNOWN;
}

/* One value, as JSON. A string parameter is the text as written: the newline that
 * frames it in the block is already gone, and nothing else is trimmed, because the
 * leading spaces of a line of code are the line. Anything else is read as JSON
 * after trimming, and what leaves is either a token that is already strict JSON or
 * a re-serialisation of what parsed - the parser here is lenient about numbers, and
 * a lenient echo is how 001 reached a client. Text that is not JSON is a string. */
static void put_arg_value(qe_str *o, const char *raw, size_t rawn, int kind) {
    char err[64] = { 0 };
    if (kind == ARG_STRING) {
        put_jstr(o, raw, rawn);
        return;
    }
    const char *p = raw;
    size_t n = rawn;
    while (n && is_space((unsigned char)*p)) {
        p++;
        n--;
    }
    while (n && is_space((unsigned char)p[n - 1]))
        n--;
    if (!n) {
        put_jstr(o, "", 0);
        return;
    }
    if (is_number(p, n) || is_word(p, n, "true") || is_word(p, n, "false") ||
        is_word(p, n, "null")) {
        qe_str_append(o, p, n);
        return;
    }
    size_t numeric = 0;
    while (numeric < n && strchr("+-.0123456789eE", p[numeric]) && p[numeric])
        numeric++;
    if (kind == ARG_JSON && n < 64 && numeric == n) {
        /* The schema asked for something other than a string and the model wrote a
         * number loosely (007, 1.): the value is clear, only its spelling is not
         * JSON. */
        char tmp[64], *end = NULL;
        memcpy(tmp, p, n);
        tmp[n] = 0;
        double d = strtod(tmp, &end);
        if (end == tmp + n && d == d && d < 1e308 && d > -1e308) {
            put_num(o, d);
            return;
        }
    }
    qe_json *v = qe_json_parse(p, n, err, sizeof err);
    if (v && v->kind == QE_JSON_STR) {
        put_jstr(o, v->str ? v->str : "", v->str_len);
        qe_json_free(v);
        return;
    }
    if (v && (v->kind == QE_JSON_OBJ || v->kind == QE_JSON_ARR)) {
        long need = qe_json_write(v, NULL, 0);
        char *buf = need >= 0 ? malloc((size_t)need + 1) : NULL;
        if (buf && qe_json_write(v, buf, (size_t)need + 1) == need)
            qe_str_append(o, buf, (size_t)need);
        else
            o->err = 1;
        free(buf);
        qe_json_free(v);
        return;
    }
    qe_json_free(v);
    put_jstr(o, raw, rawn);
}

/* Reports one buffered request as an id, a name and one argument object.
 * Malformed structure is dropped rather than forwarded: a call the model did
 * not ask for, executed against a real filesystem, is the one failure mode here
 * that cannot be undone. */
/* The request in the block, opened to the client as soon as its function's name is
 * whole: a client draws the call when it opens, and the arguments of a file write are
 * a minute of decoding that would otherwise show nothing at all. Only the name is
 * sent early. The arguments still go once the block has closed, because only then is
 * it known what they are - and a request that never closes is reported as cut off
 * (finish_turn), which is what keeps a client from running a call nobody finished
 * writing: fantasy, for one, fills in {} and dispatches any call it was opened on
 * unless the reply says it was truncated. */
static const char *new_call_id(qe_turn *t);

static int announce_call(qe_turn *t) {
    slice fname;
    size_t after = 0;
    if (t->announced ||
        find_open(t->block.p, t->block.len, 0, "function", &fname, NULL, &after) || !fname.n)
        return 0;
    const char *id = new_call_id(t);
    if (!id)
        return 0;
    t->announced = 1;
    qe_turn_ev ev = { .kind = QE_TURN_CALL_OPEN, .index = t->calls, .name = fname.p,
                      .name_len = (int)fname.n, .call_id = id };
    return emit(t, &ev);
}

/* A call id for the request about to be opened: index t->calls, built on the stem. */
static const char *new_call_id(qe_turn *t) {
    char *id = malloc(64);
    char **grown = id ? realloc(t->ids, (size_t)(t->calls + 1) * sizeof *t->ids) : NULL;
    if (!id || !grown) {
        free(id);
        return NULL;
    }
    t->ids = grown;
    snprintf(id, 64, "%s-%d", t->id_prefix, t->calls);
    t->ids[t->calls] = id;
    return id;
}

/* The other spelling of a call. This template teaches the XML form, but the family is
 * also trained on a JSON object inside the same block - {"name": ..., "arguments":
 * {...}}, the form of its other templates - and a model deep in a long session writes it
 * now and then. Read the way llama.cpp's and vLLM's parsers for this family read it,
 * instead of being dropped: a dropped request is a turn that, to the client, simply
 * stopped. The arguments go out re-serialised, so they are strict JSON whatever the
 * model's spelling of them was. */
static int emit_json_call(qe_turn *t) {
    slice b = trim_slice(t->block.p, t->block.len);
    if (!b.n || b.p[0] != '{')
        return -1;
    qe_json *v = qe_json_parse(b.p, b.n, NULL, 0);
    const qe_json *src = v && v->kind == QE_JSON_OBJ ? v : NULL;
    const qe_json *fn = src ? qe_json_at(src, "function") : NULL;
    if (fn && fn->kind == QE_JSON_OBJ)
        src = fn;
    const char *name = src ? qe_json_str_at(src, "name", NULL) : NULL;
    if (!name || !*name) {
        qe_json_free(v);
        return -1;
    }
    const qe_json *a = qe_json_at(src, "arguments");
    if (!a)
        a = qe_json_at(src, "parameters");
    qe_json *parsed = NULL;
    if (a && a->kind == QE_JSON_STR) {
        parsed = qe_json_parse(a->str, a->str_len, NULL, 0);
        a = parsed;
    }
    if (a && a->kind != QE_JSON_OBJ) {
        qe_json_free(parsed);
        qe_json_free(v);
        return -1;
    }
    char *args = NULL;
    long need = a ? qe_json_write(a, NULL, 0) : 2;
    args = need >= 0 ? malloc((size_t)need + 1) : NULL;
    if (args && a)
        qe_json_write(a, args, (size_t)need + 1);
    else if (args)
        memcpy(args, "{}", 3);
    const char *id = args ? new_call_id(t) : NULL;
    int rc = -1;
    if (id) {
        qe_turn_ev ev = { .kind = QE_TURN_CALL_OPEN, .index = t->calls, .name = name,
                          .name_len = (int)strlen(name), .call_id = id };
        rc = emit(t, &ev);
        if (!rc) {
            ev.kind = QE_TURN_CALL_ARGS;
            ev.name = NULL;
            ev.name_len = 0;
            ev.utf8 = args;
            ev.nbytes = (int)need;
            rc = emit(t, &ev);
        }
        if (!rc) {
            ev.kind = QE_TURN_CALL_CLOSE;
            ev.utf8 = NULL;
            ev.nbytes = 0;
            rc = emit(t, &ev);
        }
        t->calls++;
        rc = rc ? 1 : 0;
    }
    free(args);
    qe_json_free(parsed);
    qe_json_free(v);
    return rc;
}

static int emit_call(qe_turn *t) {
    int rc = announce_call(t);
    if (rc)
        return rc;
    if (!t->announced) {
        /* Nothing that names a function the template's way. The JSON way, then; and
         * failing that it is dropped - and said to have been, because to the client it
         * is a turn that just stopped. */
        rc = emit_json_call(t);
        if (rc >= 0)
            return rc;
        qe_turn_ev ev = { .kind = QE_TURN_CALL_DROPPED, .utf8 = t->block.p ? t->block.p : "",
                          .nbytes = (int)t->block.len };
        return emit(t, &ev);
    }
    const char *p = t->block.p;
    size_t n = t->block.len;
    size_t after = 0;
    slice fname;
    find_open(p, n, 0, "function", &fname, NULL, &after);

    qe_str args = { 0 };
    putc1(&args, '{');
    size_t pos = after;
    int first = 1;
    for (;;) {
        size_t fat = 0, fafter = 0;
        int closed = !find_close(p, n, pos, "function", &fat, &fafter);
        slice pname;
        size_t tag_at = 0, tag_after = 0;
        if (find_open(p, n, pos, "parameter", &pname, &tag_at, &tag_after))
            break;
        if (closed && tag_at > fat)
            break;
        if (!pname.n)
            break;
        size_t cat = 0, cafter = 0;
        if (find_close(p, n, tag_after, "parameter", &cat, &cafter))
            break;
        size_t vs = tag_after, ve = cat;
        if (vs < ve && p[vs] == '\n')
            vs++;
        if (ve > vs && p[ve - 1] == '\n')
            ve--;
        if (!first)
            putc1(&args, ',');
        first = 0;
        put_jstr(&args, pname.p, pname.n);
        putc1(&args, ':');
        put_arg_value(&args, p + vs, ve - vs, param_kind(t->tools, fname, pname));
        pos = cafter;
    }
    putc1(&args, '}');
    /* The call is open on the client's side, so it gets an argument object whatever
     * happens here; an allocation that failed half way is an empty one. */
    const int bad = args.err;

    qe_turn_ev ev = { .kind = QE_TURN_CALL_ARGS, .index = t->calls, .call_id = t->ids[t->calls],
                      .utf8 = bad || !args.p ? "{}" : args.p,
                      .nbytes = bad || !args.p ? 2 : (int)args.len };
    rc = emit(t, &ev);
    if (!rc) {
        ev.kind = QE_TURN_CALL_CLOSE;
        ev.utf8 = NULL;
        ev.nbytes = 0;
        rc = emit(t, &ev);
    }
    t->calls++;
    t->announced = 0;
    qe_str_free(&args);
    return rc;
}

/* The turn closed with its reasoning block still open and nothing else said, so the
 * answer it has is the one it wrote there. See qe_turn_token in chat.h. */
static int rescue_answer(qe_turn *t) {
    if (t->state != ST_REASON || t->calls || t->content_bytes)
        return 0;
    slice s = trim_slice(t->reasoning.p, t->reasoning.len);
    if (!s.n)
        return 0;
    qe_turn_ev ev = { .kind = QE_TURN_CONTENT, .utf8 = s.p, .nbytes = (int)s.n };
    t->content_bytes += (long)s.n;
    return emit(t, &ev);
}

/* The close of an assistant turn. A reply that asked for something ended with a
 * tool call; anything else is a stop. */
static int finish_turn(qe_turn *t) {
    qe_turn_ev ev = { .kind = QE_TURN_STOP,
                      .finish = t->aborted ? "length" : t->calls ? "tool_calls" : "stop" };
    t->finished = 1;
    t->state = ST_ANSWER;
    return emit(t, &ev);
}

int qe_turn_token(qe_turn *t, int id, const char *utf8, int nbytes) {
    if (!t)
        return 0;
    switch (id) {
    case QE_MARK_REASON_OPEN:
        /* The model opening a second reasoning block is a formatting slip, not a
         * reason to leak structure: keep collecting reasoning text. */
        t->state = ST_REASON;
        return flush_blank(t, 1);
    case QE_MARK_REASON_CLOSE:
        t->state = ST_ANSWER;
        return flush_blank(t, 1);
    case QE_MARK_TOOL_OPEN:
        if (t->announced) /* a second opening inside an open request: the first is cut off */
            t->aborted = 1;
        t->announced = 0;
        t->state = ST_BLOCK;
        t->block.len = 0; /* keep the capacity; blocks repeat within a turn */
        t->blank.len = 0; /* the empty line before a request belongs to neither side */
        return 0;
    case QE_MARK_TOOL_CLOSE:
        if (t->state != ST_BLOCK)
            return 0;
        t->state = ST_ANSWER;
        return emit_call(t);
    case QE_MARK_TOOL_OUT_OPEN:
        /* A model echoing a result block is repeating input, not answering, and
         * what stands between the two delimiters is that input. Inside a request
         * block the text is buffered with the rest of it, malformed along with
         * whatever else arrived. */
        if (t->state != ST_BLOCK)
            t->state = ST_ECHO;
        return 0;
    case QE_MARK_TOOL_OUT_CLOSE:
        if (t->state == ST_ECHO)
            t->state = ST_ANSWER;
        return 0;
    case QE_MARK_TURN_CLOSE:
        if (t->state == ST_BLOCK) {
            t->block.len = 0;
            if (t->announced)
                t->aborted = 1;
            t->announced = 0;
        }
        return flush_blank(t, t->calls == 0) || rescue_answer(t) || finish_turn(t);
    default:
        break;
    }
    if (nbytes <= 0)
        return 0; /* some other control token: there is no text for it */
    switch (t->state) {
    case ST_REASON:
        return emit_text(t, QE_TURN_REASONING, utf8, nbytes);
    case ST_BLOCK:
        if (qe_str_append(&t->block, utf8, (size_t)nbytes))
            return 1;
        return t->announced ? 0 : announce_call(t);
    case ST_ECHO:
        return 0;
    default:
        return emit_answer(t, utf8, nbytes);
    }
}

int qe_turn_end(qe_turn *t) {
    if (!t)
        return 0;
    /* A request whose block never closed is dropped: the client suppresses a
     * call that arrived with finish_reason length anyway, and a partial one is
     * the least safe thing to hand it. */
    if (t->state == ST_BLOCK) {
        t->block.len = 0;
        if (t->announced)
            t->aborted = 1;
        t->announced = 0;
    }
    return flush_blank(t, t->calls == 0);
}

int qe_turn_aborted(const qe_turn *t) { return t ? t->aborted : 0; }

long qe_turn_reasoning_bytes(const qe_turn *t) {
    return t ? t->reason_bytes : 0;
}
