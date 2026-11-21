/* Minimal HTTP/1.1 server, internal to qe.

   Thread-per-connection with blocking I/O and read/write timeouts, not an
   epoll state machine. That is a deliberate trade: generation is serialised by
   the model (one decode at a time, 5.3 GiB of DRAM per step), so request
   concurrency here is about not head-of-line blocking a *parse* behind another
   request's parse, and about keeping a stalled client from stalling everyone.
   A state machine would add a second place where request framing is decided,
   which is the interesting bug class for a hand-rolled server.

   What is supported is what an OpenAI client sends: request line, headers,
   Content-Length bodies, keep-alive, and streaming responses delimited by
   connection close. Chunked request bodies are refused with 501 rather than
   silently mis-handled, because "silently" is how a server ends up answering a
   request it never fully read. */
#ifndef QE_HTTP_H
#define QE_HTTP_H

#include <signal.h>
#include <stdarg.h>
#include <stddef.h>

#define QE_HTTP_MAX_HDR 24
#define QE_HTTP_PATH_MAX 2048

typedef struct qe_http_conn qe_http_conn;

typedef struct {
    char method[8];
    char path[QE_HTTP_PATH_MAX]; /* path only; query string dropped */
    int keep_alive;
    /* Body is NUL-terminated (body_len excludes the NUL) so the JSON parser can
     * run over it without threading a length everywhere. */
    const char *body;
    size_t body_len;
    struct {
        char name[48];  /* lower-cased */
        char val[256];
    } hdr[QE_HTTP_MAX_HDR];
    int n_hdr;
} qe_http_req;

/* Return 0 to close the connection cleanly. Non-zero is reported to the log
 * only; the request has already been answered or not, and there is nobody
 * above this layer to negotiate with. */
typedef int (*qe_http_handler)(qe_http_conn *c, const qe_http_req *req, void *ctx);

typedef struct {
    int port;
    int max_conns;   /* handler threads; over the cap answers 503 */
    int backlog;
    size_t max_head; /* request line + headers */
    size_t max_body; /* data URLs with images make this large */
    int idle_secs;   /* recv timeout per socket operation: a keep-alive client
                      * may think for minutes between two requests */
    int write_secs;  /* send timeout: a stream nobody is reading is stopped, not
                      * waited for, because the engine generates behind it */
    int sndbuf;      /* bytes of stream the kernel may hold for a reader that has
                      * stopped reading. This is what turns the timeout into a
                      * bound: a socket that can queue 4 MiB of deltas never
                      * reaches the timeout within a turn, because a whole turn
                      * fits in the queue. */
    int max_requests;/* keep-alive depth per connection */
    const char *id;  /* Server: header */
} qe_http_opts;

/* Sensible defaults for a local inference server; fields left at 0 are filled
 * in, port must be set by the caller. */
void qe_http_opts_default(qe_http_opts *o);

int qe_http_listen(const qe_http_opts *o); /* fd or -1, errno set */
/* Accepts until the listening socket is closed, qe_http_stop is set, or the
 * listener is broken. Blocks. Returns 0 when it stopped on request. */
int qe_http_serve(int lfd, qe_http_handler h, void *ctx, const qe_http_opts *o);
/* Set from a signal handler to stop between accepts: the blocked accept4 comes
 * back with EINTR, the loop tests this, and serve() returns so exit handlers
 * (and LeakSanitizer) actually run. */
extern volatile sig_atomic_t qe_http_stop;

/* Waiting for the threads qe_http_serve left running, for at most grace_secs.
 * Returns 1 when none are left and 0 when the grace ran out, with *live (optional)
 * saying how many are still inside a handler.
 *
 * This exists because serve() returns the moment the listener closes while a
 * request thread may be halfway through generation holding the engine, and a
 * process that frees the engine under that thread has a use-after-free on the
 * shutdown path - the one path every test harness takes. Waiting without a
 * deadline is no better: a request against a real checkpoint is minutes, so the
 * server would outlive its own SIGTERM and keep the port and the weights with it.
 * Both bounds are the caller's to set; note that a keep-alive client sitting
 * between requests counts as live, which is why the grace is short. */
int qe_http_drain(int grace_secs, int *live);

/* ---- handler side ---- */

const char *qe_http_hdr(const qe_http_req *req, const char *name); /* NULL if absent */

/* Whole response, buffered and sent with an accurate Content-Length. */
int qe_http_json(qe_http_conn *c, int status, const char *json_body);
int qe_http_error(qe_http_conn *c, int status, const char *type, const char *msg);

/* Streaming: head() sends 200 + text/event-stream with Connection: close,
 * data() writes one "data: ...\n\n" event, done() writes the terminating
 * "data: [DONE]\n\n". A non-zero return means the client went away, which is
 * the generator's cue to stop decoding. The head carries no-cache and
 * X-Accel-Buffering: no so that no intermediary between the engine and the
 * client is permitted to hold the stream and hand it over at the end. */
int qe_http_sse_head(qe_http_conn *c);
int qe_http_sse_data(qe_http_conn *c, const char *data);
int qe_http_sse_done(qe_http_conn *c);
/* One SSE comment (": text\n\n"): bytes the client's parser sees and its
 * message assembler does not. Only the first line of `text` is written, since
 * a comment that ran on would swallow the event after it. */
int qe_http_sse_comment(qe_http_conn *c, const char *text);
/* Raw bytes on a stream that already had its head sent. */
int qe_http_write(qe_http_conn *c, const void *p, size_t n);
int qe_http_vprintf_stream(qe_http_conn *c, const char *fmt, va_list ap);
int qe_http_printf_stream(qe_http_conn *c, const char *fmt, ...);
int qe_http_peer(const qe_http_conn *c, char *buf, size_t n);

/* ---- growable strings, for building JSON without a library ---- */

typedef struct {
    char *p;
    size_t len, cap;
    int err; /* set once by the first failing operation; every later one no-ops */
} qe_str;

void qe_str_free(qe_str *s);
int qe_str_append(qe_str *s, const char *p, size_t n);
int qe_str_puts(qe_str *s, const char *p);
int qe_str_appendf(qe_str *s, const char *fmt, ...);
/* Valid JSON string literal including the quotes; escapes ", \, control
 * characters, and leaves UTF-8 bytes alone. */
int qe_str_json_str(qe_str *s, const char *p, size_t n);
/* There is deliberately no macro that "just writes a literal as JSON": passing
 * structure through qe_str_json_str escapes the quotes and produces a string
 * literal instead of an object, which is valid JSON and a broken response.
 * Raw structure goes through qe_str_puts; qe_str_json_str is for values. */

/* One reply's worth of UTF-8 discipline. A byte-level vocabulary cuts a character
 * wherever the merge table says to, so a token can end three bytes into a
 * four-byte character and the next token can be anything at all. Bytes are
 * therefore not forwarded as they arrive: a sequence that is complete and
 * decodable goes out, one that is merely unfinished is held back until the rest
 * of it shows up, and a byte that cannot begin or continue a sequence becomes
 * U+FFFD. A body written through this is valid UTF-8, which is what a client
 * that decodes the response rather than eyeballs it requires - and JSON is only
 * JSON if the string inside it decodes. */
typedef struct {
    qe_str hold; /* the unfinished sequence, at most three bytes */
} qe_utf8_out;

/* Appends the whole sequences in p[0..n) to out and keeps the unfinished tail.
 * Returns non-zero only when out could not be grown. */
int qe_utf8_feed(qe_utf8_out *o, qe_str *out, const char *p, size_t n);
/* A reply that ends inside a character loses those bytes, exactly as a reply
 * that ends at a token boundary does. Cheap enough to call on every way out. */
void qe_utf8_finish(qe_utf8_out *o);
void qe_utf8_free(qe_utf8_out *o);

#endif /* QE_HTTP_H */
