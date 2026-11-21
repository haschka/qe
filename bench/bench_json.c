/* Request-parsing harness.

   The two inputs a chat request can carry that break hand-written parsers are
   a multi-megabyte base64 image and user text containing quotes, newlines and
   non-ASCII. Both are exercised here, plus the rejection cases - a parser that
   accepts malformed JSON answers requests it has not actually read. */
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void check(int ok, const char *what) {
    if (!ok) {
        fails++;
        printf("  FAIL: %s\n", what);
    }
}

static const char *BODY = "{"
                         "\"model\":\"qwen4exp-iq4\","
                         "\"messages\":["
                         "{\"role\":\"system\",\"content\":\"be brief\"},"
                         "{\"role\":\"user\",\"content\":["
                         "{\"type\":\"text\",\"text\":\"what is in \\\"this\\\" image?\\nplease answer\"},"
                         "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,"
                         "iVBORw0KGgo=\",\"detail\":\"auto\"}}"
                         "]}],"
                         "\"max_tokens\":37,"
                         "\"temperature\":0.8,"
                         "\"stream\":true,"
                         "\"n\":1,"
                         "\"stop\":[\"EOF\",\"\\n\\n\"],"
                         "\"stream_options\":{\"include_usage\":true}"
                         "}";

int main(void) {
    char err[128];

    /* ---- 1. a realistic chat body ---- */
    qe_json *root = qe_json_parse(BODY, strlen(BODY), err, sizeof err);
    check(root != NULL, "chat body parses");
    if (!root)
        printf("  reason: %s\n", err);
    if (root) {
        check(!strcmp(qe_json_str_at(root, "model", ""), "qwen4exp-iq4"), "model");
        check(qe_json_num_at(root, "max_tokens", -1) == 37, "max_tokens");
        check(qe_json_bool_at(root, "stream", 0) == 1, "stream");
        check(qe_json_bool_at(root, "absent", 7) == 7, "absent key falls back");
        check(qe_json_at(root, "absent") == NULL, "absent key is NULL");
        const qe_json *msgs = qe_json_at(root, "messages");
        check(msgs && msgs->kind == QE_JSON_ARR && qe_json_len(msgs) == 2, "two messages");
        const qe_json *m1 = msgs ? msgs->child->next : NULL;
        check(m1 && !strcmp(qe_json_str_at(m1, "role", ""), "user"), "second message is user");
        const qe_json *content = m1 ? qe_json_at(m1, "content") : NULL;
        check(content && content->kind == QE_JSON_ARR && qe_json_len(content) == 2,
              "multimodal content is two parts");
        const qe_json *text = content ? content->child : NULL;
        const qe_json *img = content ? content->child->next : NULL;
        check(text && !strcmp(qe_json_str_at(text, "type", ""), "text"), "part 0 is text");
        check(text && !strcmp(qe_json_str_at(text, "text", ""),
                              "what is in \"this\" image?\nplease answer"),
              "text part keeps its escapes decoded");
        const qe_json *iu = img ? qe_json_at(img, "image_url") : NULL;
        const char *url = iu ? qe_json_str_at(iu, "url", "") : "";
        check(!strcmp(url, "data:image/png;base64,iVBORw0KGgo="), "data URL survives intact");
        const qe_json *stop = qe_json_at(root, "stop");
        check(stop && qe_json_len(stop) == 2 && !strcmp(stop->child->next->str, "\n\n"),
              "stop array with escaped entries");
        check(qe_json_bool_at(qe_json_at(root, "stream_options"), "include_usage", 0) == 1,
              "nested stream_options");
        qe_json_free(root);
    }

    /* ---- 2. escapes and code points ---- */
    {
        const char *s = "{\"k\":\"a\\\"b\\\\c\\nd\\te\\u00e9f\\ud83d\\ude00g\\u0000z\"}";
        qe_json *r = qe_json_parse(s, strlen(s), err, sizeof err);
        const char *v = r ? qe_json_str_at(r, "k", "") : "";
        check(r != NULL, "escape body parses");
        check(!strcmp(v, "a\"b\\c\nd\te\xC3\xA9""f\xF0\x9F\x98\x80""g"),
              "escapes decode to the right bytes");
        check(strlen(v) == 17, "decoded length counts UTF-8 bytes, not code points");
        if (fails == 0)
            printf("  decoded: %s\n", v);
        qe_json_free(r);
        /* An unpaired surrogate must not emit an invalid UTF-8 sequence. */
        const char *bad = "{\"k\":\"\\ud83d!\"}";
        r = qe_json_parse(bad, strlen(bad), err, sizeof err);
        v = r ? qe_json_str_at(r, "k", "") : "";
        check(r && !strcmp(v, "\xEF\xBF\xBD" "!"), "unpaired surrogate becomes U+FFFD");
        qe_json_free(r);
    }

    /* ---- 3. numbers ---- */
    {
        const char *s = "{\"a\":-1.5e3,\"b\":0,\"c\":1e-9,\"d\":2,\"e\":-0.0}";
        qe_json *r = qe_json_parse(s, strlen(s), err, sizeof err);
        check(r != NULL, "numbers parse");
        check(qe_json_num_at(r, "a", 1) == -1500.0, "exponent form");
        check(qe_json_num_at(r, "b", 1) == 0.0, "zero is not treated as absent");
        check(qe_json_num_at(r, "c", 1) == 1e-9, "subnormal exponent");
        check(qe_json_bool_at(r, "b", 9) == 0, "0 is falsy for a flag");
        qe_json_free(r);
    }

    /* A top-level array is accepted by the parser: it is valid JSON, and it is
     * the endpoints' job to insist on an object. */
    {
        const char *arr = "[1,\"two\",false]";
        qe_json *r = qe_json_parse(arr, strlen(arr), err, sizeof err);
        check(r && r->kind == QE_JSON_ARR && qe_json_len(r) == 3, "array at the top level parses");
        qe_json_free(r);
    }

    /* ---- 4. rejections ---- */
    static const struct {
        const char *body;
        const char *why;
    } BAD[] = {
        { "{", "truncated object" },
        { "{\"a\":", "truncated value" },
        { "{\"a\":1}}", "trailing data" },

        { "123", "scalar at the top level" },
        { "", "empty body" },
        { "{\"a\" 1}", "missing colon" },
        { "{\"a\":}", "missing value" },
        { "{\"a\":tru}", "misspelt literal" },
        { "{\"a\":1e999}", "infinite number" },
        { "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]"
          "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]",
          "nesting over the depth limit" },
    };
    for (size_t i = 0; i < sizeof BAD / sizeof *BAD; i++) {
        err[0] = 0;
        qe_json *r = qe_json_parse(BAD[i].body, strlen(BAD[i].body), err, sizeof err);
        char msg[128];
        snprintf(msg, sizeof msg, "rejects %s (%s)", BAD[i].why, err);
        check(r == NULL, msg);
        if (r)
            qe_json_free(r);
    }
    {
        /* A trailing comma before a close is tolerated on purpose: it costs
         * nothing and senders do produce it. */
        const char *lenient = "{\"a\":[1,2,],\"b\":{}}";
        qe_json *r = qe_json_parse(lenient, strlen(lenient), err, sizeof err);
        check(r != NULL, "tolerates a trailing comma inside an array");
        check(r && qe_json_at(r, "b") && qe_json_at(r, "b")->kind == QE_JSON_OBJ &&
                  qe_json_len(qe_json_at(r, "b")) == 0,
              "empty object is not an error");
        qe_json_free(r);
    }

    /* ---- 5. base64 ---- */
    static const struct {
        const char *in;
        const char *out;
        int n;
    } B64[] = {
        { "", "", 0 },
        { "YQ==", "a", 1 },
        { "YWI=", "ab", 2 },
        { "YWJj", "abc", 3 },
        { "YWJjZGVmZw==", "abcdefg", 7 },
        { "YQ\r\n", "a", 1 },
        { "Y Q", "a", 1 },
        { "-_8=", "\xFB\xFF", 2 }, /* URL-safe alphabet */
    };
    for (size_t i = 0; i < sizeof B64 / sizeof *B64; i++) {
        unsigned char out[64];
        long n = qe_b64_decode(B64[i].in, strlen(B64[i].in), out, sizeof out);
        char msg[128];
        snprintf(msg, sizeof msg, "b64 \\\"%s\\\" -> %d bytes", B64[i].in, B64[i].n);
        check(n == B64[i].n && (n <= 0 || !memcmp(out, B64[i].out, (size_t)n)), msg);
    }
    static const struct {
        const char *in;
        const char *why;
    } BADB[] = {
        { "YQ=A", "data after padding" },
        { "Y*", "character outside the alphabet" },
        { "Y", "single symbol" },
        { "YR==", "non-zero bits outside the last byte" },
        { "====", "padding only" },
    };
    for (size_t i = 0; i < sizeof BADB / sizeof *BADB; i++) {
        unsigned char out[64];
        long n = qe_b64_decode(BADB[i].in, strlen(BADB[i].in), out, sizeof out);
        char msg[128];
        snprintf(msg, sizeof msg, "b64 rejects %s", BADB[i].why);
        check(n == -1, msg);
    }
    {
        unsigned char out[4];
        check(qe_b64_decode("YWJjZGVmZw==", 12, out, sizeof out) == -1,
              "b64 refuses to overflow the output buffer");
        check(qe_b64_decode("YQ", 2, out, sizeof out) == 1,
              "unpadded two-symbol group is valid base64");
    }

    /* ---- 6. a data URL the size of a real screenshot ---- */
    {
        size_t n = (size_t)1 << 20;
        char *big = malloc(n + 64);
        memcpy(big, "{\"d\":\"", 6);
        memset(big + 6, 'A', n);
        memcpy(big + 6 + n, "\"}", 3);
        qe_json *r = qe_json_parse(big, n + 8, err, sizeof err);
        const qe_json *d = r ? qe_json_at(r, "d") : NULL;
        check(d && d->str_len == n, "1 MiB string parses with the right length");
        check(d && d->str[n - 1] == 'A', "and is not truncated");
        unsigned char *raw = malloc(n);
        long got = d ? qe_b64_decode(d->str, d->str_len, raw, n) : -1;
        check(got == (long)(n / 4) * 3, "1 MiB of base64 decodes to 786432 bytes");
        qe_json_free(r);
        free(big);
        free(raw);
    }

    printf("\n%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
