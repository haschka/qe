/* The conversation contract, tested.

   Two halves, because they have different dependencies. The stream parser needs
   no vocabulary at all: it is driven by ids, so its cases are synthetic token
   streams, fed one token at a time, including tags that arrive cut in half and
   blocks that never close. The renderer needs the checkpoint's control tokens,
   so it runs only when a path is given, and it opens the file for metadata
   alone (QE_STAGE_MMAP with no serving), which is the one thing this repository
   reads from a device.

   Marker sequences never appear in this file's output either: they are printed
   as B[name] so that a failing case can be pasted into a chat transcript without
   breaking whatever reads it. */
#include "chat.h"
#include "http.h"
#include "json.h"
#include "qe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;

#define CHECK(cond, ...)                                                                        \
    do {                                                                                        \
        checks++;                                                                               \
        if (!(cond)) {                                                                          \
            fails++;                                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                         \
            printf(__VA_ARGS__);                                                                 \
            printf("\n");                                                                       \
        }                                                                                       \
    } while (0)

#define CHECK_STR(got, want, what)                                                              \
    do {                                                                                        \
        const char *g_ = (got), *w_ = (want);                                                    \
        checks++;                                                                               \
        if (!g_ || strcmp(g_, w_)) {                                                             \
            fails++;                                                                            \
            printf("FAIL %s: got [%s] want [%s]\n", (what), g_ ? g_ : "(null)", w_);             \
        }                                                                                       \
    } while (0)

static void redact(const char *s, char *out, size_t n) {
    size_t o = 0;
    while (*s && o + 8 < n) {
        const char *p = strchr(s, '<');
        if (!p)
            break;
        size_t pre = (size_t)(p - s);
        memcpy(out + o, s, pre);
        o += pre;
        s = p;
        const char *end = strchr(s, '>');
        if (!end)
            break;
        size_t len = (size_t)(end - s) + 1;
        char buf[80];
        if (len - 2 >= 2 && s[1] == '|' && end[-1] == '|')
            snprintf(buf, sizeof buf, "B[%.*s]B", (int)(len - 4), s + 2);
        else if (len >= 3 && s[1] == '/')
            snprintf(buf, sizeof buf, "C[%.*s]", (int)(len - 3), s + 2);
        else
            snprintf(buf, sizeof buf, "O[%.*s]", (int)(len - 2), s + 1);
        size_t bl = strlen(buf);
        memcpy(out + o, buf, bl);
        o += bl;
        s = end + 1;
    }
    size_t rest = strlen(s);
    if (rest > n - o - 1)
        rest = n - o - 1;
    memcpy(out + o, s, rest);
    out[o + rest] = 0;
}

static void show(const char *tag, const char *s) {
    char red[4096];
    redact(s, red, sizeof red);
    printf("  %s: %s\n", tag, red);
}

/* ------------------------------------------------------------ the parser */

#include <stdarg.h>

typedef struct {
    qe_str content, reasoning, args;
    qe_str names, ids;
    int n_calls, n_stops;
    const char *finish;
    int order[64];
    int n_order;
} captured;

static int cap_cb(void *vp, const qe_turn_ev *ev) {
    captured *c = vp;
    if (c->n_order < (int)(sizeof c->order / sizeof *c->order))
        c->order[c->n_order++] = ev->kind;
    switch (ev->kind) {
    case QE_TURN_CONTENT:
        qe_str_append(&c->content, ev->utf8, (size_t)ev->nbytes);
        break;
    case QE_TURN_REASONING:
        qe_str_append(&c->reasoning, ev->utf8, (size_t)ev->nbytes);
        break;
    case QE_TURN_CALL_OPEN:
        qe_str_append(&c->names, ev->name, (size_t)ev->name_len);
        qe_str_puts(&c->names, "|");
        qe_str_puts(&c->ids, ev->call_id);
        qe_str_puts(&c->ids, "|");
        c->n_calls++;
        break;
    case QE_TURN_CALL_ARGS:
        qe_str_append(&c->args, ev->utf8, (size_t)ev->nbytes);
        qe_str_puts(&c->args, "|");
        break;
    case QE_TURN_STOP:
        c->n_stops++;
        c->finish = ev->finish;
        break;
    default: break;
    }
    return 0;
}

static void cap_free(captured *c) {
    qe_str_free(&c->content);
    qe_str_free(&c->reasoning);
    qe_str_free(&c->args);
    qe_str_free(&c->names);
    qe_str_free(&c->ids);
    memset(c, 0, sizeof *c);
}

/* Tag text, assembled from character codes (60, 61, 62) instead of written out,
   so that neither this file nor the output of a failing case can contain a
   control sequence. The parser sees these only as bytes: every piece of
   structure it reports comes from an id. */
static void *arena[512];
static size_t arena_n;

static void *keep(void *p) {
    if (arena_n < sizeof arena / sizeof *arena)
        arena[arena_n++] = p;
    return p;
}

static void cleared(void) {
    for (size_t i = 0; i < arena_n; i++)
        free(arena[i]);
    arena_n = 0;
}

static char *dupf(const char *fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return keep(strdup(buf));
}

static char *tag_named(const char *name, const char *val) {
    return dupf("%c%s%c%s%c", (char)60, name, (char)61, val, (char)62);
}

static char *tag_close(const char *name) { return dupf("%c/%s%c", (char)60, name, (char)62); }

/* The inside of one request, exactly as the template writes it: a function tag,
   a parameter tag per argument, the closing function tag. */
static char *call_text(const char *name, const char *const *keys, const char *const *vals,
                       int nval) {
    qe_str s = { 0 };
    qe_str_appendf(&s, "\n%s\n", tag_named("function", name));
    for (int i = 0; i < nval; i++)
        qe_str_appendf(&s, "%s\n%s\n%s\n", tag_named("parameter", keys[i]),
                       vals[i] ? vals[i] : "", tag_close("parameter"));
    qe_str_appendf(&s, "%s\n", tag_close("function"));
    char *out = keep(strdup(s.p ? s.p : ""));
    qe_str_free(&s);
    return out;
}

typedef struct {
    int id;
    const char *text;
} piece;

typedef struct {
    piece *v;
    size_t n, cap;
} feed;

static void f_add(feed *f, int id, const char *text) {
    if (f->n == f->cap) {
        f->cap = f->cap ? f->cap * 2 : 32;
        f->v = realloc(f->v, f->cap * sizeof *f->v);
    }
    f->v[f->n].id = id;
    f->v[f->n].text = text;
    f->n++;
}

static void f_text(feed *f, const char *s) { f_add(f, -1, s); }
static void f_tok(feed *f, int id) { f_add(f, id, ""); }

/* Text cut into chunks of n bytes. A tag that arrives in pieces has to be
   recognised anyway, and sweeping the chunk size covers every way a tag can be
   split across a token boundary. */
static void f_text_split(feed *f, const char *s, size_t n) {
    size_t len = strlen(s);
    for (size_t o = 0; o < len; o += n) {
        size_t k = len - o < n ? len - o : n;
        char *part = keep(malloc(k + 1));
        memcpy(part, s + o, k);
        part[k] = 0;
        f_add(f, -1, part);
    }
}

static void f_call(feed *f, const char *name, const char *const *keys, const char *const *vals,
                   int nval) {
    f_tok(f, QE_MARK_TOOL_OPEN);
    f_text(f, call_text(name, keys, vals, nval));
    f_tok(f, QE_MARK_TOOL_CLOSE);
}

/* Feeds every piece, in order, and reports how many calls the parser refused to
 * continue on: a nonzero return from the callback contract is how a server stops
 * generation, so it must never happen for ordinary text. */
/* The tools[] the next feed is parsed against, as a request would hand it over. */
static const qe_json *feed_tools;

static int f_feed_stem(feed *f, captured *c, int starts_in_reasoning, const char *stem) {
    qe_turn *t = qe_turn_new(cap_cb, c, starts_in_reasoning);
    int refused = 0;
    qe_turn_set_id_prefix(t, stem);
    qe_turn_set_tools(t, feed_tools);
    for (size_t i = 0; i < f->n; i++) {
        const char *text = f->v[i].text ? f->v[i].text : "";
        int nbytes = f->v[i].text ? (int)strlen(text) : 0;
        if (qe_turn_token(t, f->v[i].id, text, nbytes))
            refused++;
    }
    if (qe_turn_end(t))
        refused++;
    qe_turn_free(t);
    free(f->v);
    f->v = NULL;
    f->n = f->cap = 0;
    return refused;
}

static int f_feed(feed *f, captured *c, int starts_in_reasoning) {
    return f_feed_stem(f, c, starts_in_reasoning, NULL);
}

static void test_parser(void) {
    captured c = { 0 };
    feed f = { 0 };
    int before = checks;

    /* An answer, and nothing else. */
    f_text(&f, "Hel");
    f_text(&f, "lo there");
    f_tok(&f, QE_MARK_TURN_CLOSE);
    CHECK(f_feed(&f, &c, 0) == 0, "a plain answer stopped the callback chain");
    CHECK_STR(c.content.p, "Hello there", "plain answer");
    CHECK(c.reasoning.len == 0, "plain answer leaked reasoning");
    CHECK_STR(c.finish, "stop", "plain answer finish");
    cap_free(&c);
    cleared();

    /* A reasoning block, then the answer that follows it. The blank line after
     * the block is part of the answer: there is no request in front of it. */
    f_text(&f, "counting");
    f_text(&f, " the files");
    f_tok(&f, QE_MARK_REASON_CLOSE);
    f_text(&f, "\n\n");
    f_text(&f, "four files");
    f_tok(&f, QE_MARK_TURN_CLOSE);
    f_feed(&f, &c, 1);
    CHECK_STR(c.reasoning.p, "counting the files", "reasoning text");
    CHECK_STR(c.content.p, "\n\nfour files", "answer after reasoning");
    CHECK_STR(c.finish, "stop", "reasoning finish");
    cap_free(&c);
    cleared();

    /* A request, fed every way it can arrive. The empty line in front of it is
     * punctuation between the answer and the request, so neither of them gets
     * it, and the trailing newline is the block's own. */
    for (size_t chunk = 1; chunk <= 7; chunk++) {
        const char *keys[] = { "command" };
        const char *vals[] = { "ls -la" };
        f_text(&f, "working");
        f_tok(&f, QE_MARK_REASON_CLOSE);
        f_text(&f, "\n\n");
        f_tok(&f, QE_MARK_TOOL_OPEN);
        f_text_split(&f, call_text("bash", keys, vals, 1), chunk);
        f_tok(&f, QE_MARK_TOOL_CLOSE);
        f_text(&f, "\n");
        f_tok(&f, QE_MARK_TURN_CLOSE);
        CHECK(f_feed(&f, &c, 1) == 0, "chunk %zu: stopped the callback chain", chunk);
        CHECK(c.n_calls == 1, "chunk %zu: one call expected, got %d", chunk, c.n_calls);
        CHECK_STR(c.names.p, "bash|", "call name");
        CHECK_STR(c.args.p, "{\"command\":\"ls -la\"}|", "call arguments");
        CHECK(c.ids.len > 0, "call id missing");
        CHECK_STR(c.finish, "tool_calls", "call finish");
        CHECK(c.content.len == 0, "request text leaked into the answer: [%s]",
              c.content.p ? c.content.p : "");
        CHECK_STR(c.reasoning.p, "working", "reasoning before a request");
        cap_free(&c);
        cleared();
    }

    /* Values that are not text keep the type the model wrote, and a value that
     * only looks like a number stays a string. */
    {
        const char *keys[] = { "times", "paths", "shell", "empty" };
        const char *vals[] = { "3", "[\"a\",\"b\"]", "3 files", "" };
        f_call(&f, "run", keys, vals, 4);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed(&f, &c, 0);
        CHECK_STR(c.args.p,
                  "{\"times\":3,\"paths\":[\"a\",\"b\"],\"shell\":\"3 files\",\"empty\":\"\"}|",
                  "argument types");
        cap_free(&c);
        cleared();
    }

    /* The schema decides. A shell id is "001": a number to a guess, which wrote it
     * as 001 - not JSON, so the client threw the arguments away - and a string to
     * the schema. A string parameter keeps the text as written, indentation and
     * quotes included, because an edit's old_string has to match the file. A number
     * the schema asked for is written as JSON even when the model spelled it 007. */
    {
        const char *schema =
            "[{\"type\":\"function\",\"function\":{\"name\":\"job_output\",\"parameters\":"
            "{\"type\":\"object\",\"properties\":{\"shell_id\":{\"type\":\"string\"},"
            "\"wait\":{\"type\":\"boolean\"},\"lines\":{\"type\":\"integer\"},"
            "\"mode\":{\"enum\":[\"a\",\"b\"]},\"note\":{\"type\":[\"string\",\"null\"]},"
            "\"opts\":{\"type\":\"object\"}}}}}]";
        qe_json *tools = qe_json_parse(schema, strlen(schema), NULL, 0);
        CHECK(tools != NULL, "test schema did not parse");
        feed_tools = tools;
        const char *keys[] = { "shell_id", "wait", "lines", "mode", "note", "opts", "free" };
        const char *vals[] = { "001", "true", "007", "true", "  \"quoted\"\n\tindented",
                               "{\"a\": 01, \"b\": [1, 2]}", "00A" };
        f_call(&f, "job_output", keys, vals, 7);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed(&f, &c, 0);
        CHECK_STR(c.args.p,
                  "{\"shell_id\":\"001\",\"wait\":true,\"lines\":7,\"mode\":\"true\","
                  "\"note\":\"  \\\"quoted\\\"\\n\\tindented\",\"opts\":{\"a\":1,\"b\":[1,2]},"
                  "\"free\":\"00A\"}|",
                  "arguments typed by the schema");
        cap_free(&c);
        cleared();
        feed_tools = NULL;

        /* With no schema at all a leading zero still cannot become a number. */
        const char *k2[] = { "shell_id", "n" };
        const char *v2[] = { "001", "0" };
        f_call(&f, "job_output", k2, v2, 2);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed(&f, &c, 0);
        CHECK_STR(c.args.p, "{\"shell_id\":\"001\",\"n\":0}|", "no schema, leading zero");
        cap_free(&c);
        cleared();
        qe_json_free(tools);
    }

    /* A turn that closes inside its reasoning, having said nothing else, wrote its
     * answer there: it comes out as the answer, once, before the stop. A sub-agent
     * whose last turn had this shape used to hand its parent no text at all. */
    f_text(&f, "The report: ");
    f_text(&f, "three bugs.\n");
    f_tok(&f, QE_MARK_TURN_CLOSE);
    CHECK(f_feed(&f, &c, 1) == 0, "rescue stopped the callback chain");
    CHECK_STR(c.reasoning.p, "The report: three bugs.\n", "rescued reasoning still reasoning");
    CHECK_STR(c.content.p, "The report: three bugs.", "unclosed reasoning becomes the answer");
    CHECK_STR(c.finish, "stop", "rescued finish");
    CHECK(c.order[c.n_order - 1] == QE_TURN_STOP && c.order[c.n_order - 2] == QE_TURN_CONTENT,
          "the rescued answer comes just before the stop");
    cap_free(&c);
    cleared();

    /* ...and nothing is rescued when the block closed, when there was an answer or a
     * request, or when the turn never closed (the budget ran out). */
    f_text(&f, "thought");
    f_tok(&f, QE_MARK_REASON_CLOSE);
    f_tok(&f, QE_MARK_TURN_CLOSE);
    f_feed(&f, &c, 1);
    CHECK(c.content.len == 0, "a closed reasoning block was rescued");
    cap_free(&c);
    cleared();
    f_text(&f, "thought");
    f_call(&f, "f", NULL, NULL, 0);
    f_tok(&f, QE_MARK_TURN_CLOSE);
    f_feed(&f, &c, 1);
    CHECK(c.content.len == 0 && c.n_calls == 1, "a turn with a request was rescued");
    cap_free(&c);
    cleared();
    f_text(&f, "cut off mid-thou");
    f_feed(&f, &c, 1);
    CHECK(c.content.len == 0, "a turn the budget cut was rescued");
    cap_free(&c);
    cleared();

    /* Two requests in one turn: distinct indexes and distinct ids, because a
     * client accumulates deltas per index and matches results by id. */
    {
        const char *k1[] = { "x" };
        const char *v1[] = { "1" };
        const char *k2[] = { "y" };
        const char *v2[] = { "2" };
        f_call(&f, "a", k1, v1, 1);
        f_call(&f, "b", k2, v2, 1);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed(&f, &c, 0);
        CHECK(c.n_calls == 2, "two calls expected, got %d", c.n_calls);
        CHECK_STR(c.names.p, "a|b|", "two names");
        CHECK_STR(c.args.p, "{\"x\":1}|{\"y\":2}|", "two argument lists");
        CHECK_STR(c.ids.p, "call-0|call-1|", "two ids");
        CHECK_STR(c.finish, "tool_calls", "two calls finish");
        cap_free(&c);
        cleared();
    }

    /* The stem those ids are built from is what a whole conversation's calls are
     * told apart by: a client keeps tool-call state under the id alone, so the same
     * id in a later turn is read as the earlier call, and its result too. */
    {
        const char *k1[] = { "x" };
        const char *v1[] = { "1" };
        f_call(&f, "a", k1, v1, 1);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed_stem(&f, &c, 0, "chatcmpl-0000beef0001");
        CHECK_STR(c.ids.p, "chatcmpl-0000beef0001-0|", "the stem is the front of the id");
        cap_free(&c);
        cleared();
        f_call(&f, "a", k1, v1, 1);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed_stem(&f, &c, 0, "chatcmpl-0000beef0002");
        CHECK_STR(c.ids.p, "chatcmpl-0000beef0002-0|", "a later request repeats nothing");
        cap_free(&c);
        cleared();
    }

    /* Structure that never closes is dropped: a request the model did not finish
     * writing is the one thing that must not be executed. */
    f_tok(&f, QE_MARK_TOOL_OPEN);
    f_text(&f, tag_named("function", "half"));
    f_text(&f, "\n");
    f_text(&f, tag_named("parameter", "x"));
    f_text(&f, "\n1\n");
    f_feed(&f, &c, 0);
    CHECK(c.n_calls == 0, "truncated call reported as a call");
    CHECK(c.n_stops == 0, "truncated call invented a finish");
    cap_free(&c);
    cleared();

    /* Text that merely looks like a tag is an answer: inside the answer there is
     * no structure to parse, only inside a request. */
    {
        char *prose = dupf("write %s literally", tag_named("function", "notacall"));
        f_text(&f, prose);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed(&f, &c, 0);
        CHECK(c.n_calls == 0, "prose tag became a call");
        CHECK_STR(c.content.p, prose, "prose tag content");
        cap_free(&c);
        cleared();
    }

    /* A request block with nothing parsable in it is dropped, and the turn goes
     * on answering afterwards. */
    f_tok(&f, QE_MARK_TOOL_OPEN);
    f_text(&f, "no tags at all");
    f_tok(&f, QE_MARK_TOOL_CLOSE);
    f_text(&f, "recovered");
    f_tok(&f, QE_MARK_TURN_CLOSE);
    f_feed(&f, &c, 0);
    CHECK(c.n_calls == 0, "malformed call reported");
    CHECK_STR(c.content.p, "recovered", "recovered after malformed request");
    CHECK_STR(c.finish, "stop", "malformed finish");
    cap_free(&c);
    cleared();

    /* A blank line in the middle of an answer is the answer's own: held back
     * while its side is unknown, it comes out in front of what follows it and
     * never after. */
    f_text(&f, "one");
    f_text(&f, "\n\n");
    f_text(&f, "two");
    f_tok(&f, QE_MARK_TURN_CLOSE);
    f_feed(&f, &c, 0);
    CHECK_STR(c.content.p, "one\n\ntwo", "blank line inside an answer");
    cap_free(&c);
    cleared();

    /* A model repeating a result block is repeating input, not answering, and
     * what stands between those delimiters is that input. */
    f_tok(&f, QE_MARK_TOOL_OUT_OPEN);
    f_text(&f, "output echoed");
    f_tok(&f, QE_MARK_TOOL_OUT_CLOSE);
    f_text(&f, "after");
    f_tok(&f, QE_MARK_TURN_CLOSE);
    f_feed(&f, &c, 0);
    CHECK_STR(c.content.p, "after", "echoed result block dropped");
    cap_free(&c);
    cleared();

    /* Generation that stops without closing the turn still reports a finish: a
     * client treats a missing one as a broken stream and retries forever. */
    f_text(&f, "half an answer");
    f_feed(&f, &c, 0);
    CHECK_STR(c.content.p, "half an answer", "unclosed turn content");
    CHECK(c.n_stops == 0, "unclosed turn reported a stop");
    cap_free(&c);
    cleared();

    /* Events arrive in order, and answer text never follows a request. */
    {
        f_text(&f, "thinking");
        f_tok(&f, QE_MARK_REASON_CLOSE);
        f_text(&f, "answer");
        f_call(&f, "f", NULL, NULL, 0);
        f_tok(&f, QE_MARK_TURN_CLOSE);
        f_feed(&f, &c, 1);
        CHECK(c.n_order == 6, "events expected 6, got %d", c.n_order);
        int bad = 0;
        for (int i = 1; i < c.n_order; i++)
            if (c.order[i] == QE_TURN_CONTENT && c.order[i - 1] == QE_TURN_CALL_ARGS)
                bad = 1;
        CHECK(!bad, "answer text arrived after a request");
        CHECK(c.order[c.n_order - 1] == QE_TURN_STOP, "last event is not the stop");
        CHECK_STR(c.args.p, "{}|", "a request with no arguments");
        cap_free(&c);
        cleared();
    }

    printf("  parser: %d checks\n", checks - before);
}

/* ------------------------------------------------------ the text boundary */

/* A byte-level vocabulary splits a character wherever its merge table says to,
 * so one token can stop in the middle of a character and the next one finishes
 * it. Nothing that leaves the process may show that: JSON is only JSON if the
 * string inside it decodes. These are the cases the filter has to get right,
 * including the one it was written for - an answer that ended two bytes into a
 * three-byte character made `make probe` reject a whole response body. */

#define FFFD "\xef\xbf\xbd" /* U+FFFD, what undecodable bytes become */

static void feed_case(qe_utf8_out *o, const char *in, size_t inlen, const char *want,
                      const char *what) {
    qe_str out = { 0 };
    checks++;
    if (qe_utf8_feed(o, &out, in, inlen)) {
        fails++;
        printf("FAIL qe_utf8_feed could not grow its output\n");
    }
    if (out.len != strlen(want) || (out.len && memcmp(out.p, want, out.len))) {
        fails++;
        printf("FAIL %s: got [%.*s] want [%s]\n", what, (int)out.len, out.p ? out.p : "", want);
    }
    qe_str_free(&out);
}

static void test_utf8(void) {
    int before = checks;
    qe_utf8_out o = { 0 };

    const char *u3 = "\xe4\xb8\x8d";     /* three bytes: U+4E0D */
    const char *u4 = "\xf0\x9f\x98\x80"; /* four bytes: U+1F600 */

    feed_case(&o, "hello", 5, "hello", "a plain run passes straight through");
    feed_case(&o, u3, 3, u3, "a complete three-byte character is unchanged");
    feed_case(&o, u4, 4, u4, "a complete four-byte character is unchanged");

    /* Split anywhere, in either direction, the character may not be visible. */
    feed_case(&o, u3, 1, "", "the first byte of a three-byte character is not a character");
    feed_case(&o, u3 + 1, 1, "", "nor is the second");
    feed_case(&o, u3 + 2, 1, u3, "the third closes it, and the whole character arrives at once");

    feed_case(&o, u4, 1, "", "one byte of a four-byte character stays held");
    feed_case(&o, u4 + 1, 2, "", "two more still leave it unfinished");
    feed_case(&o, u4 + 3, 1, u4, "and the whole character comes out when the last byte lands");

    /* A reply that ends mid-character loses those bytes, the same way a reply
     * that ends mid-token emits nothing for the rest of the token. */
    feed_case(&o, "a\xe4\xb8", 3, "a", "an answer cut mid-character emits the text before it");
    qe_utf8_finish(&o);
    feed_case(&o, "b", 1, "b", "finish drops the bytes of a character that never completed");
    CHECK(o.hold.len == 0, "finish leaves nothing behind");

    /* Undecodable bytes are replaced, one U+FFFD each, and never passed through.
     * Every case below is a sequence a strict decoder rejects outright. */
    feed_case(&o, "\xa9", 1, FFFD, "a lone continuation byte becomes one replacement character");
    feed_case(&o, "\xc0\x80", 2, FFFD FFFD, "the overlong two-byte form is refused");
    feed_case(&o, "\xe0\x80\x80", 3, FFFD FFFD FFFD, "the overlong three-byte form is refused");
    feed_case(&o, "\xed\xa0\x80", 3, FFFD FFFD FFFD, "a surrogate half is refused");
    feed_case(&o, "\xf5\x80\x80\x80", 4, FFFD FFFD FFFD FFFD, "above U+10FFFF is refused");
    feed_case(&o, "\xf4\x90\x80\x80", 4, FFFD FFFD FFFD FFFD, "and so is one code point past it");
    feed_case(&o, "\xf0\x80\x80\x80", 4, FFFD FFFD FFFD FFFD, "the overlong four-byte form too");

    /* A held lead byte followed by something that cannot continue it resolves
     * as undecodable rather than blocking the stream forever. The bytes that
     * arrive later are still their own characters. */
    feed_case(&o, "\xe4" "a", 2, FFFD "a", "a held lead byte does not swallow later text");
    CHECK(o.hold.len == 0, "nothing is left held after a resolved stream");

    /* What a token callback actually produces: text, a character split down the
     * middle, more text, in that order. */
    feed_case(&o, "ok" "\xe4", 3, "ok", "text emits while the split character waits");
    feed_case(&o, "\xb8\x8d" "!", 3, "\xe4\xb8\x8d" "!",
              "the held character precedes what follows it");

    printf("  text boundary: %d checks\n", checks - before);
}

/* ------------------------------------------------------------ the renderer */

static int count_str(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    int n = 0;
    if (!nl)
        return 0;
    for (const char *p = hay; (p = strstr(p, needle)); p += nl)
        n++;
    return n;
}

static int ends_with(const char *hay, const char *tail) {
    size_t hl = strlen(hay), tl = strlen(tail);
    return hl >= tl && !memcmp(hay + hl - tl, tail, tl);
}

static const char *XHIGH = "Reasoning effort is set to xhigh.";
static const char *LOWTEXT = "Reasoning effort is set to low.";

static qe_json *body(const char *json) {
    char err[128] = { 0 };
    qe_json *v = qe_json_parse(json, strlen(json), err, sizeof err);
    if (!v)
        printf("FAIL test body is not JSON: %s\n", err);
    return v;
}

/* The eight markers of this checkpoint, in the order the counts below are
 * written. Their texts come from the vocabulary: this file contains neither the
 * sequences nor the words they are built from. */
static const int mark_ids[8] = { QE_MARK_TURN_OPEN,  QE_MARK_TURN_CLOSE, QE_MARK_TOOL_OPEN,
                                 QE_MARK_TOOL_CLOSE, QE_MARK_TOOL_OUT_OPEN,
                                 QE_MARK_TOOL_OUT_CLOSE, QE_MARK_REASON_OPEN,
                                 QE_MARK_REASON_CLOSE };
static const char *mark_label[8] = { "turn open",  "turn close", "request open",  "request close",
                                     "result open", "result close", "reasoning open",
                                     "reasoning close" };
static char mark_text[8][64];

static int marks_load(qe_tok *t) {
    for (int i = 0; i < 8; i++)
        if (qe_tok_decode(t, &mark_ids[i], 1, mark_text[i], sizeof mark_text[i]) <= 0)
            return 0;
    return 1;
}

static int mark_count(const char *p, const char *needle) {
    size_t nl = strlen(needle), n = 0;
    if (!nl)
        return 0;
    for (const char *q = p; (q = strstr(q, needle)); q += nl)
        n++;
    return (int)n;
}

/* What a rendered prompt is made of: how many of each marker it holds, and
 * whether each one encodes back to the single id it was written from. A marker
 * that encodes to letters is a turn boundary the model will not see, which is
 * the difference between a transcript and a wall of text. */
static void check_markers(qe_tok *t, const qe_chat *r, const char *what, int to, int tc, int co,
                          int cc, int oo, int oc, int ro, int rc) {
    const int want[8] = { to, tc, co, cc, oo, oc, ro, rc };
    if (!r->prompt.p) {
        CHECK(0, "%s: nothing was rendered", what);
        return;
    }
    long n = qe_tok_count(t, r->prompt.p);
    int *ids = malloc((size_t)(n + 8) * sizeof *ids);
    long got = ids ? qe_tok_encode(t, r->prompt.p, ids, n + 8) : -1;
    CHECK(got == n && n > 0, "%s: prompt should encode (%ld counted, %ld encoded)", what, n, got);
    for (int i = 0; i < 8; i++) {
        int text = mark_count(r->prompt.p, mark_text[i]);
        int asid = 0;
        for (long k = 0; ids && k < got; k++)
            if (ids[k] == mark_ids[i])
                asid++;
        CHECK(text == want[i], "%s: %s expected %d, prompt holds %d", what, mark_label[i], want[i],
              text);
        CHECK(text == asid, "%s: %s wrote %d times as %d ids", what, mark_label[i], text, asid);
    }
    free(ids);
}

static void test_render(qe_tok *t) {
    int before = checks;
    char err[256] = { 0 };
    if (!marks_load(t)) {
        printf("FAIL a control token has no text\n");
        fails++;
        return;
    }
    const char *m_ro = mark_text[6], *m_rc = mark_text[7];
    printf("  markers: ids %d %d %d %d %d %d %d %d\n", mark_ids[0], mark_ids[1], mark_ids[2],
           mark_ids[3], mark_ids[4], mark_ids[5], mark_ids[6], mark_ids[7]);

    /* A single user turn: the prompt is that turn plus the generation prompt,
     * which leaves the model standing inside its reasoning block. */
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"user\",\"content\":\"hi there\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .effort_xhigh = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render: %s", err);
        /* The effort sentence is a system turn of its own when the caller did not
         * supply one, the user turn is the second, the generation prompt the
         * third, and it leaves the model standing inside its reasoning block. */
        check_markers(t, &r, "user turn", 3, 2, 0, 0, 0, 0, 1, 0);
        char tail[128];
        snprintf(tail, sizeof tail, "%s\n", m_ro);
        CHECK(ends_with(r.prompt.p, tail), "prompt does not open a reasoning block");
        CHECK(strstr(r.prompt.p, "hi there") != NULL, "user text missing");
        CHECK(strstr(r.prompt.p, XHIGH) != NULL, "xhigh instruction missing");
        CHECK(strstr(r.prompt.p, "system\n") != NULL, "effort sentence not in a system turn");
        if (getenv("CHAT_DUMP"))
            show("single user turn", r.prompt.p);
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* An image, the way charmbracelet's client sends a viewed file: a user message
     * of one image part after the tool result. With a tower it is the template's three
     * markers - exactly one image pad, which the engine turns into the image's rows -
     * and the pad is one id; without one it is a sentence that says it cannot be seen. */
    {
        const char *js =
            "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":"
            "\"what is this\"},{\"type\":\"image_url\",\"image_url\":{\"url\":"
            "\"data:image/png;base64,iVBORw0KGgo=\"}}]}]}";
        qe_json *b = body(js);
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .effort_xhigh = 1, .preserve_thinking = 1,
                           .vision = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render image: %s", err);
        CHECK(r.vision && r.n_images == 1, "vision render: vision %d, images %d", r.vision,
              r.n_images);
        long cap = (long)r.prompt.len + 8, pads = 0, opens = 0;
        int *ids = malloc((size_t)cap * sizeof *ids);
        long got = ids ? qe_tok_encode(t, r.prompt.p, ids, cap) : -1;
        for (long k = 0; k < got; k++) {
            pads += ids[k] == QE_MARK_IMAGE_PAD;
            opens += ids[k] == QE_MARK_VISION_OPEN;
        }
        CHECK(pads == 1 && opens == 1, "one image is one pad (%ld) inside one open (%ld)", pads,
              opens);
        CHECK(strstr(r.prompt.p, "cannot") == NULL, "a seen image carries the unseen note");
        free(ids);
        qe_chat_free(&r);
        o.vision = 0;
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render image off: %s", err);
        CHECK(!r.vision && strstr(r.prompt.p, "not visible to you") != NULL,
              "an unseen image does not say so");
        qe_chat_free(&r);
        qe_json_free(b);

        /* The template refuses an image in a system message; so does this. */
        b = body("{\"messages\":[{\"role\":\"system\",\"content\":[{\"type\":\"image_url\","
                 "\"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KGgo=\"}}]},"
                 "{\"role\":\"user\",\"content\":\"x\"}]}");
        o.vision = 1;
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) != 0, "a system image was rendered");
        qe_chat_free(&r);
        qe_json_free(b);

        /* Past the limit the oldest go: the last image a request carries is the one a
         * client is asking about. */
        qe_str big = { 0 };
        qe_str_puts(&big, "{\"messages\":[");
        for (int i = 0; i < QE_CHAT_MAX_IMAGES + 3; i++)
            qe_str_appendf(&big, "%s{\"role\":\"user\",\"content\":[{\"type\":\"image_url\","
                                 "\"image_url\":{\"url\":\"data:image/png;base64,%s\"}}]}",
                           i ? "," : "", i == QE_CHAT_MAX_IMAGES + 2 ? "TEFTVA==" : "iVBORw0KGgo=");
        qe_str_puts(&big, "]}");
        b = body(big.p);
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render many images: %s", err);
        CHECK(r.n_images == QE_CHAT_MAX_IMAGES && r.truncated_images,
              "many images: %d kept, truncated %d", r.n_images, r.truncated_images);
        CHECK(r.n_images > 0 && r.images[r.n_images - 1].len == 4 &&
                  !memcmp(r.images[r.n_images - 1].bytes, "LAST", 4),
              "the newest image was the one left out");
        qe_chat_free(&r);
        qe_json_free(b);
        qe_str_free(&big);
    }

    /* Thinking off: no effort sentence, and an empty reasoning block instead of
     * an opened one. */
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 0, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render off: %s", err);
        CHECK(strstr(r.prompt.p, XHIGH) == NULL, "effort sentence sent with thinking off");
        check_markers(t, &r, "thinking off", 2, 1, 0, 0, 0, 0, 1, 1);
        char tail[128];
        snprintf(tail, sizeof tail, "%s\n\n%s\n\n", m_ro, m_rc);
        CHECK(ends_with(r.prompt.p, tail), "thinking off does not close an empty block");
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* Effort levels: medium sends no sentence, low sends its own. */
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                          "\"reasoning_effort\":\"low\"}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .effort_low = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render low: %s", err);
        CHECK(strstr(r.prompt.p, LOWTEXT) != NULL, "low instruction missing");
        qe_chat_free(&r);
        qe_json_free(b);
    }
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                          "\"reasoning_effort\":\"nonsense\"}");
        qe_chat_opts o;
        CHECK(qe_chat_opts_from_body(b, &o, err, sizeof err) == -1,
              "an unknown effort should be refused");
        qe_json_free(b);
    }
    {
        qe_json *b = body("{\"messages\":[],\"reasoning_effort\":\"high\"}");
        qe_chat_opts o;
        char e2[256] = { 0 };
        CHECK(qe_chat_opts_from_body(b, &o, e2, sizeof e2) == 0, "high should be accepted");
        CHECK(o.effort_xhigh == 1, "high should mean xhigh");
        qe_json_free(b);
    }

    /* Tools: the schema block, then the format instructions, then the caller's
     * own system text. Order is the contract. */
    {
        qe_json *b = body("{\"messages\":["
                          "{\"role\":\"system\",\"content\":\"be terse\"},"
                          "{\"role\":\"user\",\"content\":\"list files\"}],"
                          "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
                          "\"description\":\"Run a command\",\"parameters\":{\"type\":\"object\","
                          "\"properties\":{\"command\":{\"type\":\"string\"}},"
                          "\"required\":[\"command\"]}}}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .effort_xhigh = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "render tools: %s", err);
        const char *head = strstr(r.prompt.p, "# Tools");
        const char *reminder = strstr(r.prompt.p, "Reminder:");
        const char *sys = strstr(r.prompt.p, "be terse");
        const char *schema = strstr(r.prompt.p, "bash");
        CHECK(head && schema && reminder && sys, "tools preamble parts missing");
        CHECK(head && schema && head < schema, "schemas should follow the tools heading");
        CHECK(schema && reminder && schema < reminder, "the reminder should follow the schemas");
        CHECK(reminder && sys && reminder < sys, "the caller's system text should come last");
        CHECK(r.tools_rendered == 1, "tools_rendered not reported");
        /* The format instructions the preamble carries hold two example requests
         * and no example result; nothing else in this prompt is a request. */
        check_markers(t, &r, "tools", 3, 2, 2, 2, 0, 0, 1, 0);
        /* Keys sorted, as the template's own filter sorts them, and the tool
         * object compact on one line. */
        CHECK(strstr(r.prompt.p, "{\"function\":{\"description\":") != NULL,
              "tool schema keys should be sorted and compact");
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* Leading system messages merge into one turn; one out of place is refused. */
    {
        qe_json *b = body("{\"messages\":["
                          "{\"role\":\"system\",\"content\":\"one\"},"
                          "{\"role\":\"developer\",\"content\":\"two\"},"
                          "{\"role\":\"user\",\"content\":\"go\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "merge systems: %s", err);
        CHECK(count_str(r.prompt.p, "system\n") == 1, "systems should merge into one turn");
        CHECK(strstr(r.prompt.p, "one\ntwo") != NULL, "merged system text should be joined");
        qe_chat_free(&r);
        qe_json_free(b);
    }
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},"
                          "{\"role\":\"system\",\"content\":\"late\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == -1,
              "a late system message should be refused");
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* Reasoning replay, both ways round the preserve switch. */
    {
        const char *hist = "{\"messages\":["
                           "{\"role\":\"user\",\"content\":\"first\"},"
                           "{\"role\":\"assistant\",\"content\":\"an answer\","
                           "\"reasoning_content\":\"why I said that\"},"
                           "{\"role\":\"user\",\"content\":\"second\"}]}";
        qe_json *b = body(hist);
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "replay: %s", err);
        CHECK(strstr(r.prompt.p, "why I said that") != NULL, "reasoning was not replayed");
        check_markers(t, &r, "reasoning replay", 4, 3, 0, 0, 0, 0, 2, 1);
        qe_chat_free(&r);
        qe_json_free(b);

        b = body(hist);
        qe_chat r2 = { 0 };
        qe_chat_opts o2 = { .enable_thinking = 1, .preserve_thinking = 0 };
        CHECK(qe_chat_render(t, b, &o2, &r2, err, sizeof err) == 0, "replay off: %s", err);
        CHECK(strstr(r2.prompt.p, "why I said that") == NULL,
              "reasoning before the last query should be dropped");
        check_markers(t, &r2, "replay dropped", 4, 3, 0, 0, 0, 0, 1, 0);
        qe_chat_free(&r2);
        qe_json_free(b);
    }

    /* A replayed assistant turn that carried no reasoning text at all, which is
     * every turn of a client that shows thinking but never sends it back. An empty
     * block is not a spacer: it is the shape this model reads as "thought of
     * nothing", so replaying forty of them is a session spent demonstrating an
     * assistant that never thinks. Default renders the turn with no block; the
     * server switch puts the template's own empty block back. */
    {
        const char *quiet = "{\"messages\":["
                            "{\"role\":\"user\",\"content\":\"first\"},"
                            "{\"role\":\"assistant\",\"content\":\"an answer\"},"
                            "{\"role\":\"user\",\"content\":\"second\"}]}";
        qe_json *b = body(quiet);
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "quiet replay: %s", err);
        /* System turn, two user turns, the assistant turn, the generation prompt -
         * and one reasoning marker only, the block the generation prompt opens. */
        check_markers(t, &r, "quiet replay", 5, 4, 0, 0, 0, 0, 1, 0);
        CHECK(strstr(r.prompt.p, "an answer") != NULL, "quiet replay lost the answer text");
        qe_chat_free(&r);
        qe_json_free(b);

        b = body(quiet);
        qe_chat r2 = { 0 };
        qe_chat_opts o2 = { .enable_thinking = 1, .preserve_thinking = 1,
                            .empty_thinking_blocks = 1 };
        CHECK(qe_chat_render(t, b, &o2, &r2, err, sizeof err) == 0, "quiet replay forced: %s", err);
        check_markers(t, &r2, "quiet replay forced", 5, 4, 0, 0, 0, 0, 2, 1);
        qe_chat_free(&r2);
        qe_json_free(b);

        /* The same with the assistant turn last, where the last-real-query rule
         * says its reasoning would be worth keeping if it had any. */
        b = body("{\"messages\":[{\"role\":\"user\",\"content\":\"a question\"},"
                 "{\"role\":\"assistant\",\"content\":\"a partial answer\"}]}");
        qe_chat r3 = { 0 };
        CHECK(qe_chat_render(t, b, &o, &r3, err, sizeof err) == 0, "quiet tail: %s", err);
        check_markers(t, &r3, "quiet tail", 4, 3, 0, 0, 0, 0, 1, 0);
        qe_chat_free(&r3);
        qe_json_free(b);
    }

    /* A tool round trip: the request goes back inside the assistant turn, and a
     * run of results shares one user turn. Arguments arriving as a JSON string -
     * the only spelling an OpenAI client has - become parameters. */
    {
        qe_json *b = body("{\"messages\":["
                          "{\"role\":\"user\",\"content\":\"list\"},"
                          "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\","
                          "\"type\":\"function\",\"function\":{\"name\":\"bash\","
                          "\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]},"
                          "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"a\\nb\"},"
                          "{\"role\":\"tool\",\"tool_call_id\":\"c2\",\"content\":\"c\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "tool round trip: %s", err);
        CHECK(strstr(r.prompt.p, "ls") != NULL, "arguments should become parameters");
        /* No tools were offered with this request, so the only example request in
         * the prompt is the one being replayed, and there is one result pair per
         * result. The user turn, the assistant turn, the one turn the two results
         * share, and the generation prompt: four openings, three closings. */
        check_markers(t, &r, "tool round trip", 4, 3, 1, 1, 2, 2, 2, 1);
        CHECK(strstr(r.prompt.p, "a\nb") != NULL, "result text missing");
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* The same round trip the way a tools client sends it: the schemas come back
     * on every turn, which re-adds the preamble with its example requests, and
     * the reasoning the assistant showed is offered again alongside the call. */
    {
        qe_json *b = body("{\"messages\":["
                          "{\"role\":\"user\",\"content\":\"list\"},"
                          "{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"first "
                          "I look\","
                          "\"tool_calls\":[{\"id\":\"c1\","
                          "\"type\":\"function\",\"function\":{\"name\":\"bash\","
                          "\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]},"
                          "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"a\\nb\"},"
                          "{\"role\":\"tool\",\"tool_call_id\":\"c2\",\"content\":\"c\"}],"
                          "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
                          "\"description\":\"Run a command\",\"parameters\":{\"type\":\"object\","
                          "\"properties\":{\"command\":{\"type\":\"string\"}},"
                          "\"required\":[\"command\"]}}}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "tools round trip: %s", err);
        if (r.prompt.p) {
            check_markers(t, &r, "tools round trip", 5, 4, 3, 3, 2, 2, 2, 1);
            CHECK(strstr(r.prompt.p, "first I look") != NULL,
                  "reasoning alongside a call dropped");
        }
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* A role the template has no rule for is an error, not a silently dropped
     * message. */
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"wizard\",\"content\":\"x\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == -1,
              "an unknown role should be refused");
        qe_chat_free(&r);
        qe_json_free(b);
    }

    /* What goes in must come back out: the prompt has to encode as the ids it
     * was built from, or the model is being fed letters where it expects
     * structure. */
    {
        qe_json *b = body("{\"messages\":[{\"role\":\"system\",\"content\":\"s\"},"
                          "{\"role\":\"user\",\"content\":\"hello world\"},"
                          "{\"role\":\"assistant\",\"content\":\"a\",\"reasoning_content\":\"r\"},"
                          "{\"role\":\"user\",\"content\":\"again\"}]}");
        qe_chat r = { 0 };
        qe_chat_opts o = { .enable_thinking = 1, .effort_xhigh = 1, .preserve_thinking = 1 };
        CHECK(qe_chat_render(t, b, &o, &r, err, sizeof err) == 0, "round trip: %s", err);
        check_markers(t, &r, "round trip", 5, 4, 0, 0, 0, 0, 2, 1);
        qe_chat_free(&r);
        qe_json_free(b);
    }
    printf("  renderer: %d checks against the checkpoint's vocabulary\n", checks - before);
}

int main(int argc, char **argv) {
    printf("bench_chat: the conversation contract\n");
    test_parser();
    test_utf8();

    if (argc > 1 && argv[1][0]) {
        char err[256] = { 0 };
        qe_model *m = qe_open(argv[1], QE_STAGE_MMAP);
        if (!m) {
            printf("FAIL open: %s\n", qe_strerror());
            return 2;
        }
        qe_tok *t = qe_tok_open(m, err, sizeof err);
        if (!t) {
            printf("FAIL tokenizer: %s\n", err);
            qe_close(m);
            return 2;
        }
        if (!qe_chat_supported(t)) {
            printf("  this checkpoint does not have the control tokens at the expected ids;"
                   " renderer checks skipped\n");
        } else {
            test_render(t);
        }
        qe_tok_close(t);
        qe_close(m);
    } else {
        printf("  renderer: skipped (no checkpoint given, metadata only)\n");
    }

    printf("%s (%d failures)\n", fails ? "FAILED" : "PASS", fails);
    return fails ? 1 : 0;
}
