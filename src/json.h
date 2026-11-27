/* JSON for request bodies, and base64 for data URLs. Internal to qe.

   Written by hand because the alternative is a dependency, and the parsing
   surface an OpenAI request actually exercises is small: objects, arrays,
   strings with escapes, numbers, booleans, null. What is not small is getting
   the string cases right, because the two biggest inputs a request can carry
   are a multi-megabyte base64 image and user text with quotes, newlines, and
   non-ASCII in it. bench_json covers those.

   Depth is bounded so a hostile body cannot recurse the parser off the stack,
   and the parser never reads past the length it was handed. */
#ifndef QE_JSON_H
#define QE_JSON_H

#include <stddef.h>

typedef enum {
    QE_JSON_NULL,
    QE_JSON_BOOL,
    QE_JSON_NUM,
    QE_JSON_STR,
    QE_JSON_ARR,
    QE_JSON_OBJ
} qe_json_kind;

typedef struct qe_json {
    qe_json_kind kind;
    char *key; /* set when this value is a member of an object */
    double num;
    int boolean;
    char *str; /* unescaped, NUL-terminated; str_len excludes the NUL. For a number,
                * the literal as it was written. */
    size_t str_len;
    struct qe_json *child, *next;
    size_t n_child;
} qe_json;

/* err is filled with a one-line reason on failure; both may be NULL. */
qe_json *qe_json_parse(const char *s, size_t n, char *err, size_t errn);
void qe_json_free(qe_json *v);

/* Object accessors. NULL / dflt when absent or the wrong type. */
const qe_json *qe_json_at(const qe_json *obj, const char *key);
const char *qe_json_str_at(const qe_json *obj, const char *key, const char *dflt);
double qe_json_num_at(const qe_json *obj, const char *key, double dflt);
int qe_json_bool_at(const qe_json *obj, const char *key, int dflt);
/* An array's element count, or 0 if it is not an array. */
size_t qe_json_len(const qe_json *arr);

/* Compact re-serialisation, key order and all: this is how the tool schema
 * block gets built, and the model's template compacts each tool object to one
 * line, so anything that reorders or prettifies changes the prompt. snprintf
 * semantics - returns the length written excluding the NUL, or the length that
 * would be needed when cap is too small (call with cap 0 to size), -1 on
 * allocation failure. Numbers print with the fewest digits that read back as
 * the same double, so a schema bound survives the round trip. */
long qe_json_write(const qe_json *v, char *out, size_t cap);

/* Standard alphabet with '=' padding. Tolerates the whitespace and the URL
 * alphabet, because clients that send image data are not consistent about it.
 * Returns bytes written, or -1 if the input is not valid base64. */
long qe_b64_decode(const char *in, size_t n, unsigned char *out, size_t outcap);

#endif /* QE_JSON_H */
