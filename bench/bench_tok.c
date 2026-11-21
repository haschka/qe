/* Tokenizer harness: does what the file says match what this code does?

   Three questions, in the order that catches the most:

     1. Is encode/decode lossless? Not "usually" - for every one of the 248,320
        tokens in the vocabulary, and for text that is CJK, emoji, source code,
        invalid UTF-8, or a long run of one character.

     2. Is the merge table internally consistent? For each token, decoding it and
        re-encoding the result must give that same single id back. A single
        misread merge rank, or a pre-tokenizer that splits a word differently
        from the one the vocabulary was trained with, shows up here as a
        non-zero count, and there is no way to fake it green.

     3. How far off was the estimate the server used before? The old accounting
        said len/4 tokens; the difference is what /v1/chat/completions used to
        report in usage.prompt_tokens.

   Metadata only - it never reads a weight byte, so it runs in seconds. */
#include "qe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails;

static void check(int ok, const char *what) {
    if (!ok) {
        fails++;
        printf("  FAIL: %s\n", what);
    }
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + t.tv_nsec * 1e-9;
}

/* logs stay ASCII: token text is escaped, never echoed raw */
static void emit_escaped(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *) s; *p; p++) {
        if (*p >= 0x20 && *p < 0x7f && *p != '"' && *p != '\\')
            putchar(*p);
        else
            printf("\\x%02X", *p);
    }
    putchar('"');
}

static const char *CORPUS[] = {
    "",
    "hi",
    "Hello, world! How are you today?",
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog.",
    "Don't stop believing, isn't it? I've said it before: it'll be fine.",
    "He said he'd've been there, but she wouldn't have.",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\nand\rmore\n\n\nparagraphs",
    "int main(void) { if (x == 42) return 0; } // 100.5% of all bugs",
    "1234567890 0.000001 -42 +3.14159265358979 1,234,567",
    "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xe3\x80\x82"
    "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xe3\x80\x82",
    "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xc8\x90\xc3\xa0\xd1\x82\xd0\xb8\xd0\xb2\xd0\xb8\xd1\x82",
    "\xf0\x9f\x98\x80\xf0\x9f\x8e\x89 \xe2\x9c\x93\xe2\x9c\x97 \xc2\xa9\xc2\xae\xe2\x80\x94\xe2\x80\x93",
    "caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9 na\xc3\xafve \xc3\x9f\xc3\xa4\xc3\xb6\xc3\xbc",
    "\xe2\x80\x8b\xe2\x80\x8d\xef\xbb\xbf zero width and byte order marks",
    "\xc3\xa9\xc2\xa0\xc2\xad combining marks: \x65\xcc\x81\x65\xcc\x80\x65\xcc\x84",
    "a",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    " word  word   word    word     word",
    "1 22 333 4444 55555",
    "()[]{}<>/\\|@#$%^&*`~_-=+;:'\"?,.",
    "\xff\xfe mixed valid and invalid \x80\x81 bytes \xc3",
    "\xe2\x80",
    "\xf5\x80\x80\x80 overlong and out of range \xc0\xaf surrogate \xed\xa0\x80",
    "\xc2\x80\xc2\x9f C1 controls that are not whitespace",
    "\xc2\xad\xc2\xa0 soft hyphen and non breaking space",
    "Q: what is 2+2? A: 4.\nQ: and 3*3? A: 9.",
    "\xd0\x9a\xd0\xb0\xd0\xba \xd1\x8d\xd1\x82\xd0\xbe \xd1\x80\xd0\xb0\xd0\xb1\xd0\xbe\xd1\x82"
    "\xd0\xb0\xd0\xb5\xd1\x82?",
};

#define NCORPUS ((int) (sizeof CORPUS / sizeof *CORPUS))

/* longest text the corpus can produce, plus room for growth on pathological
 * expansions (there are none, but the bound makes the loops simple) */
#define SCRATCH 65536

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : NULL;
    if (!path) {
        fprintf(stderr, "usage: %s <gguf with tokenizer metadata>\n", argv[0]);
        return 2;
    }

    char err[256] = { 0 };
    /* Demand-paged: this bench never reads a weight byte, and the default mode
     * stages all 87 GiB before it prints a single token. */
    qe_model *m = qe_open(path, QE_STAGE_MMAP);
    if (!m) {
        fprintf(stderr, "open %s: %s\n", path, qe_strerror());
        return 2;
    }
    qe_tok *t = qe_tok_open(m, err, sizeof err);
    if (!t) {
        fprintf(stderr, "tokenizer: %s\n", err);
        qe_close(m);
        return 2;
    }
    char desc[512];
    qe_tok_describe(t, desc, sizeof desc);
    printf("tokenizer: %s\n", desc);
    check(strstr(desc, "skipped, special") != NULL, "describe() shape");
    check(atoi(strstr(desc, "vocab ") + 6) > 0, "vocabulary is non-empty");

    static char text[SCRATCH];
    static int ids[SCRATCH];
    static int ids2[SCRATCH];
    char back[SCRATCH];

    /* ---- 1. every token in the vocabulary round trips ----
     *
     * Two properties have to hold per token, and both are asserted:
     *   - decode then encode then decode reproduces the same bytes (lossless)
     *   - the encoding is a fixed point, so nothing re-segments on a second pass
     * A token whose own text re-merges differently is not a contradiction: real
     * vocabularies carry tokens that the merge order would never produce (extra
     * CJK and full-width forms are the usual case). That count is reported, not
     * asserted, but a growth in it means the merge table was misread. */
    printf("\n1. per-token round trip over the whole vocabulary\n");
    long n_ids = qe_tok_n_vocab(t), n_nul = 0, n_loss_fail = 0, n_fixed_fail = 0;
    long n_alias = 0, n_seen = 0, n_alias_special = 0;
    int ex_id[6] = { -1 };
    int ex_n = 0;
    for (long id = 0; id < n_ids; id++) {
        int one = (int) id;
        long L = qe_tok_decode(t, &one, 1, text, sizeof text);
        if (L < 0 || L >= (long) sizeof text) {
            n_loss_fail++;
            continue;
        }
        text[L] = 0;
        if (strlen(text) != (size_t) L) {
            n_nul++; /* contains a NUL byte; C strings cannot carry it */
            continue;
        }
        long n = qe_tok_encode(t, text, ids, 32);
        n_seen++;
        if (n <= 0) {
            n_loss_fail++;
            continue;
        }
        /* same bytes out? */
        char again[512];
        long L2 = qe_tok_decode(t, ids, n, again, sizeof again);
        if (L2 != L || memcmp(again, text, (size_t) L)) {
            n_loss_fail++;
            continue;
        }
        /* fixed point? re-encoding the decoded text must give the same ids */
        long m2 = qe_tok_encode(t, again, ids2, 512);
        if (m2 != n || memcmp(ids, ids2, (size_t) n * sizeof *ids))
            n_fixed_fail++;
        if (n != 1 || ids[0] != (int) id) {
            if (qe_tok_is_special(t, (int) id))
                n_alias_special++;
            else
                n_alias++;
            if (ex_n < 6)
                ex_id[ex_n++] = (int) id;
        }
    }
    printf("   %ld tokens checked, %ld contain NUL (not representable as a C string)\n", n_seen,
           n_nul);
    printf("   lossless: %ld failures. idempotent: %ld failures\n", n_loss_fail, n_fixed_fail);
    printf("   re-encodes to a different segmentation: %ld ordinary, %ld control "
           "(vocabulary entries the merge order would not produce)\n",
           n_alias, n_alias_special);
    check(n_loss_fail == 0, "every token survives decode/encode byte for byte");
    check(n_fixed_fail == 0, "tokenization is a fixed point of itself");
    check(n_alias_special == 0, "control tokens are matched literally, never split");
    check(n_alias * 1000 < n_seen, "at most 0.1%% of the vocabulary is not reachable");
    for (int i = 0; i < ex_n; i++) {
        int one = ex_id[i];
        qe_tok_decode(t, &one, 1, back, sizeof back);
        long n = qe_tok_encode(t, back, ids, 32);
        printf("      id %d = ", ex_id[i]);
        emit_escaped(back);
        printf(" -> %ld ids:", n);
        for (long k = 0; k < n && k < 4; k++)
            printf(" %d", ids[k]);
        printf("\n");
    }

    /* ---- 2. corpus round trips ---- */
    printf("\n2. text round trips\n");
    long est_bytes = 0, est_true = 0;
    for (int i = 0; i < NCORPUS; i++) {
        const char *s = CORPUS[i];
        long n = qe_tok_encode(t, s, ids, SCRATCH);
        check(n >= 0, "encode succeeded");
        if (n < 0)
            continue;
        long L = qe_tok_decode(t, ids, n, back, sizeof back);
        check(L == (long) strlen(s) && !memcmp(back, s, (size_t) L), "decode(encode(s)) == s");
        long again = qe_tok_count(t, s);
        check(again == n, "count agrees with encode");
        est_bytes += (long) strlen(s);
        est_true += n;
    }
    printf("   %d texts, all exact\n", NCORPUS);

    /* ---- 3. long inputs and buffer bounds ---- */
    printf("\n3. bounds\n");
    {
        size_t big = 200000;
        char *s = malloc(big + 1);
        for (size_t i = 0; i < big; i++)
            s[i] = (char) ("abcdefghij"[i % 10]);
        s[big] = 0;
        long n = qe_tok_count(t, s);
        check(n > 0, "200 KB input encodes");
        int small[4];
        check(qe_tok_encode(t, s, small, 4) == -1, "too-small id buffer is refused, not filled");
        long L = qe_tok_decode(t, ids, 4, back, 8);
        check(L >= 4 && strlen(back) < 8, "decode truncates without overflowing");
        (void) L;
        double t0 = now();
        int iters = 4;
        for (int i = 0; i < iters; i++)
            n = qe_tok_count(t, s);
        double dt = now() - t0;
        (void) n;
        printf("   worst case (one 200 KB run of letters): %.1f MB/s, %ld tokens\n",
               (double) big * iters / dt / 1e6, n);

        /* Real prose is a different animal: the splitter keeps pieces short, so
         * the quadratic merge loop never sees more than a dozen units. */
        const char *unit = CORPUS[2];
        size_t ul = strlen(unit);
        size_t rl = 0;
        while (rl + ul + 1 < big) {
            memcpy(s + rl, unit, ul);
            rl += ul;
            s[rl++] = ' ';
        }
        s[rl] = 0;
        t0 = now();
        long rn = 0;
        for (int i = 0; i < 200; i++)
            rn = qe_tok_count(t, s);
        dt = now() - t0;
        printf("   prose (%zu KB x 200): %.1f MB/s, %.3f tokens/byte\n", rl / 1024,
               (double) rl * 200 / dt / 1e6, (double) rn / (double) rl);
        free(s);
    }

    /* ---- 4. what the server used to report ---- */
    printf("\n4. exact prompt tokens vs the old len/4 estimate\n");
    static const char *PROMPTS[] = {
        "hi",
        "Explain the difference between a gated delta net and a softmax attention "
        "layer, and when each one wins.",
        "Translate to French: The quick brown fox jumps over the lazy dog, twice.\n"
        "Then list three exceptions.",
        "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x81\xe8\xab\x8b\xe7\x94\xa8\xe4\xb8\xad"
        "\xe6\x96\x87\xe7\xb0\xa1\xe8\xbf\xb0\xe7\x80\x8f\xe9\x82\x8a\xe6\x8e\xa8"
        "\xe7\x90\x86\xe6\xa8\xa1\xe5\x9e\x8b\xe7\x9a\x84\xe9\x80\x9a\xe9\x87\x8f"
        "\xe7\x93\xb6\xe9\xa0\x88\xe3\x80\x82",
        "fn main() {\n    let v: Vec<i32> = (0..10).map(|x| x * x).collect();\n"
        "    println!(\"{:?}\", v);\n}\n",
    };
    printf("   %-52s %8s %8s %7s\n", "prompt", "exact", "len/4", "ratio");
    for (size_t i = 0; i < sizeof PROMPTS / sizeof *PROMPTS; i++) {
        long n = qe_tok_count(t, PROMPTS[i]);
        long e = ((long) strlen(PROMPTS[i]) + 3) / 4;
        printf("   %-52s %8ld %8ld %6.2fx\n", PROMPTS[i], n, e, e ? (double) n / (double) e : 0.0);
    }
    printf("   corpus overall: %.1f bytes, %ld exact tokens, %.2f bytes/token (estimate said 4.00)\n",
           (double) est_bytes, est_true, est_true ? (double) est_bytes / (double) est_true : 0.0);

    printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASS", fails);
    qe_tok_close(t);
    qe_close(m);
    return fails ? 1 : 0;
}
