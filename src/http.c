/* HTTP/1.1 server. See http.h for why this is thread-per-connection. */
#define _GNU_SOURCE /* accept4, memmem */
#include "http.h"
#include "qe.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define HEAD_DEFAULT (64u << 10)
#define BODY_DEFAULT (48u << 20) /* one 4 MiB image is ~5.3 MiB base64 */
#define CONN_DEFAULT 32
/* A quarter of a megabyte of stream per client. A token delta is ~200 bytes and
 * this host emits ~13 a second, so the queue is never the limit for a reader that
 * reads; it is only ever full when nobody is. Kept that small on purpose: with the
 * kernel's 4 MiB default a whole turn fits in the queue, `send` never waits, and
 * the send timeout below could not fire inside a turn however long it was. */
#define SNDBUF_DEFAULT (64 << 10)

struct qe_http_conn {
    int fd;
    const qe_http_opts *o;
    char peer[128];
    int responded; /* a whole-response helper already ran */
    int streamed;  /* head() ran; only stream writes are legal now */
    int dead;      /* a write failed: the client is gone */
};

/* ------------------------------------------------------------------ strings */

void qe_str_free(qe_str *s) {
    free(s->p);
    s->p = NULL;
    s->len = s->cap = 0;
}

static int str_grow(qe_str *s, size_t need) {
    if (s->err)
        return 1;
    if (s->len + need <= s->cap)
        return 0;
    size_t cap = s->cap ? s->cap : 256;
    while (cap < s->len + need) {
        if (cap > (size_t)1 << 31) {
            s->err = 1;
            return 1;
        }
        cap *= 2;
    }
    char *p = realloc(s->p, cap);
    if (!p) {
        s->err = 1;
        return 1;
    }
    s->p = p;
    s->cap = cap;
    return 0;
}

int qe_str_append(qe_str *s, const char *p, size_t n) {
    if (str_grow(s, n + 1))
        return 1;
    memcpy(s->p + s->len, p, n);
    s->len += n;
    s->p[s->len] = 0;
    return 0;
}

int qe_str_puts(qe_str *s, const char *p) { return qe_str_append(s, p, strlen(p)); }

int qe_str_appendf(qe_str *s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[512];
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) {
        s->err = 1;
        return 1;
    }
    if ((size_t)n < sizeof tmp)
        return qe_str_append(s, tmp, (size_t)n);
    char *big = malloc((size_t)n + 1);
    if (!big) {
        s->err = 1;
        return 1;
    }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    int rc = qe_str_append(s, big, (size_t)n);
    free(big);
    return rc;
}

int qe_str_json_str(qe_str *s, const char *p, size_t n) {
    if (str_grow(s, n + 2))
        return 1;
    qe_str_append(s, "\"", 1);
    for (size_t i = 0; i < n && !s->err; i++) {
        unsigned char c = (unsigned char)p[i];
        switch (c) {
        case '"': qe_str_append(s, "\\\"", 2); break;
        case '\\': qe_str_append(s, "\\\\", 2); break;
        case '\n': qe_str_append(s, "\\n", 2); break;
        case '\r': qe_str_append(s, "\\r", 2); break;
        case '\t': qe_str_append(s, "\\t", 2); break;
        case '\b': qe_str_append(s, "\\b", 2); break;
        case '\f': qe_str_append(s, "\\f", 2); break;
        default:
            if (c < 0x20)
                qe_str_appendf(s, "\\u%04x", c);
            else
                qe_str_append(s, (const char *)&c, 1);
        }
    }
    qe_str_append(s, "\"", 1);
    return s->err;
}

/* ------------------------------------------------------------------ utf-8 */

/* U+FFFD, the character that means "the bytes there were not text". */
static const char REPLACEMENT[3] = { (char)0xEF, (char)0xBF, (char)0xBD };

/* Length of the sequence starting at p, 0 when what is there is a legal prefix
 * that has not received all its bytes yet, -1 when no sequence starts here. The
 * bounds on the second byte are the ones a strict decoder enforces, so a
 * sequence this accepts is one Go, Python and Rust will decode: 0xC0 and 0xC1
 * never occur at all, 0xE0 may not be followed by a continuation below 0xA0
 * (that is an overlong encoding), 0xED may not reach a surrogate, and nothing
 * above U+10FFFF is a character. */
static int seq_scan(const unsigned char *p, size_t n) {
    unsigned char c = p[0];
    int len;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c < 0x80)
        return 1;
    if (c >= 0xC2 && c <= 0xDF) {
        len = 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
        len = 3;
        if (c == 0xE0)
            lo = 0xA0;
        if (c == 0xED)
            hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        len = 4;
        if (c == 0xF0)
            lo = 0x90;
        if (c == 0xF4)
            hi = 0x8F;
    } else {
        return -1;
    }
    for (int i = 1; i < len; i++) {
        if ((size_t)i >= n)
            return 0;
        unsigned char l = i == 1 ? lo : 0x80, h = i == 1 ? hi : 0xBF;
        if (p[i] < l || p[i] > h)
            return -1;
    }
    return len;
}

int qe_utf8_feed(qe_utf8_out *o, qe_str *out, const char *p, size_t n) {
    if (n && qe_str_append(&o->hold, p, n))
        return 1;
    const unsigned char *b = (const unsigned char *)o->hold.p;
    size_t i = 0, m = o->hold.len;
    while (i < m) {
        int len = seq_scan(b + i, m - i);
        if (len == 0)
            break; /* the start of a character: wait for the rest of it */
        if (len > 0) {
            if (qe_str_append(out, (const char *)b + i, (size_t)len))
                return 1;
            i += (size_t)len;
            continue;
        }
        if (qe_str_append(out, REPLACEMENT, sizeof REPLACEMENT))
            return 1;
        i++; /* undecodable: one replacement character, then carry on */
    }
    if (i) {
        memmove(o->hold.p, o->hold.p + i, m - i);
        o->hold.len = m - i;
        o->hold.p[o->hold.len] = 0;
    }
    return out->err;
}

void qe_utf8_finish(qe_utf8_out *o) { o->hold.len = 0; }

void qe_utf8_free(qe_utf8_out *o) { qe_str_free(&o->hold); }

/* ------------------------------------------------------------------ sockets */

static const char *status_text(int st) {
    switch (st) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 406: return "Not Acceptable";
    case 409: return "Conflict";
    case 413: return "Content Too Large";
    case 414: return "URI Too Long";
    case 422: return "Unprocessable Content";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

static int send_all(int fd, const void *p, size_t n) {
    const char *c = p;
    while (n) {
        ssize_t k = send(fd, c, n, MSG_NOSIGNAL);
        if (k > 0) {
            c += (size_t)k;
            n -= (size_t)k;
            continue;
        }
        if (k < 0 && errno == EINTR)
            continue;
        return -1;
    }
    return 0;
}

/* Receive until the buffer holds the terminating empty line. Returns the offset
 * of the CRLFCRLF, 0 for a clean close, -1 for a limit or error. */
static long read_head(qe_http_conn *c, char **buf, size_t *cap, size_t *len) {
    size_t head_max = c->o->max_head;
    for (;;) {
        if (*len >= 4) {
            void *at = memmem(*buf, *len, "\r\n\r\n", 4);
            if (at)
                return (long)((char *)at - *buf) + 2; /* keep one CRLF for split */
        }
        if (*len + 4096 > *cap) {
            size_t ncap = *cap ? *cap : 8192;
            while (ncap < *len + 4096)
                ncap *= 2;
            if (ncap > head_max) {
                errno = E2BIG;
                return -1;
            }
            char *nb = realloc(*buf, ncap);
            if (!nb)
                return -1;
            *buf = nb;
            *cap = ncap;
        }
        ssize_t k = recv(c->fd, *buf + *len, *cap - *len, 0);
        if (k > 0) {
            *len += (size_t)k;
            continue;
        }
        if (k < 0 && errno == EINTR)
            continue;
        return 0; /* close or timeout: nothing more to parse */
    }
}

static void lower(char *p) {
    for (; *p; p++)
        if (*p >= 'A' && *p <= 'Z')
            *p = (char)(*p + 32);
}

static void trim(char *p) {
    size_t n = strlen(p);
    while (n && (p[n - 1] == ' ' || p[n - 1] == '\t' || p[n - 1] == '\r'))
        p[--n] = 0;
    while (*p == ' ' || *p == '\t')
        memmove(p, p + 1, strlen(p));
}

const char *qe_http_hdr(const qe_http_req *req, const char *name) {
    for (int i = 0; i < req->n_hdr; i++)
        if (!strcmp(req->hdr[i].name, name))
            return req->hdr[i].val;
    return NULL;
}

/* Parses one request head. Sets *err_status on a rejection and returns -1;
 * returns 0 on success. */
static int parse_head(qe_http_conn *c, char *buf, size_t eom, qe_http_req *req, int *err_status,
                      const char **err_why) {
    *err_status = 0;
    *err_why = "";
    buf[eom] = 0; /* the CRLF after the request line is still in the buffer */

    char *line = buf;
    char *eol = strstr(line, "\r\n");
    if (!eol) {
        *err_status = 400;
        *err_why = "malformed request line";
        return -1;
    }
    *eol = 0;
    char *sp1 = strchr(line, ' ');
    if (!sp1) {
        *err_status = 400;
        *err_why = "malformed request line";
        return -1;
    }
    *sp1 = 0;
    char *target = sp1 + 1;
    char *sp2 = strchr(target, ' ');
    if (!sp2) {
        *err_status = 400;
        *err_why = "malformed request line";
        return -1;
    }
    *sp2 = 0;
    snprintf(req->method, sizeof req->method, "%s", line);
    if (strlen(line) >= sizeof req->method) {
        *err_status = 405;
        *err_why = "method too long";
        return -1;
    }
    /* Absolute-URI form (proxies) and query strings both reduce to the path. */
    if (!strncasecmp(target, "http://", 7) || !strncasecmp(target, "https://", 8)) {
        const char *slash = strchr(target + 7, '/');
        target = slash ? (char *)slash : (char *)"/";
    }
    char *q = strchr(target, '?');
    if (q)
        *q = 0;
    snprintf(req->path, sizeof req->path, "%s", target);
    if (strlen(target) >= sizeof req->path) {
        *err_status = 414;
        *err_why = "path too long";
        return -1;
    }
    req->keep_alive = 1;
    const char *ver = sp2 + 1;
    if (strstr(ver, "1.0"))
        req->keep_alive = 0;

    char *hdr = eol + 2;
    while (*hdr && req->n_hdr < QE_HTTP_MAX_HDR) {
        char *ne = strstr(hdr, "\r\n");
        if (ne)
            *ne = 0;
        if (!*hdr)
            break;
        char *colon = strchr(hdr, ':');
        if (colon) {
            *colon = 0;
            char *val = colon + 1;
            trim(hdr);
            lower(hdr);
            trim(val);
            snprintf(req->hdr[req->n_hdr].name, sizeof req->hdr[0].name, "%s", hdr);
            snprintf(req->hdr[req->n_hdr].val, sizeof req->hdr[0].val, "%s", val);
            req->n_hdr++;
        }
        if (!ne)
            break;
        hdr = ne + 2;
    }
    if (req->n_hdr >= QE_HTTP_MAX_HDR && hdr && *hdr) {
        *err_status = 431;
        *err_why = "too many header fields";
        return -1;
    }
    const char *conn = qe_http_hdr(req, "connection");
    if (conn) {
        if (strstr(conn, "close"))
            req->keep_alive = 0;
        else if (strstr(conn, "keep-alive"))
            req->keep_alive = 1;
    }
    return 0;
}

/* ---------------------------------------------------------------- responses */

static int write_head(qe_http_conn *c, int status, const char *type, long long len, int keep,
                      int stream) {
    qe_str h = { 0 };
    qe_str_appendf(&h, "HTTP/1.1 %d %s\r\n", status, status_text(status));
    qe_str_appendf(&h, "Content-Type: %s\r\n", type);
    qe_str_appendf(&h, "Server: %s\r\n", c->o->id ? c->o->id : "qe");
    if (len >= 0)
        qe_str_appendf(&h, "Content-Length: %lld\r\n", len);
    /* A stream is the only response whose whole value is in being delivered as it
     * is produced, so it is the only one that has to defend that: a proxy, a
     * sidecar, or a client library that honours default cache hints will otherwise
     * buffer it and hand the user the answer in one piece at the end, which is
     * indistinguishable from an engine that never answered at all. */
    if (stream) {
        qe_str_puts(&h, "Cache-Control: no-cache\r\n");
        qe_str_puts(&h, "X-Accel-Buffering: no\r\n");
    }
    qe_str_appendf(&h, "Connection: %s\r\n\r\n", keep ? "keep-alive" : "close");
    int rc = send_all(c->fd, h.p, h.len);
    qe_str_free(&h);
    return rc;
}

int qe_http_write(qe_http_conn *c, const void *p, size_t n) {
    if (c->dead)
        return -1;
    if (send_all(c->fd, p, n)) {
        c->dead = 1;
        return -1;
    }
    return 0;
}

int qe_http_json(qe_http_conn *c, int status, const char *json_body) {
    if (c->responded || c->streamed)
        return -1;
    c->responded = 1;
    size_t n = strlen(json_body);
    if (write_head(c, status, "application/json", (long long)n, !c->dead, 0)) {
        c->dead = 1;
        return -1;
    }
    return qe_http_write(c, json_body, n);
}

int qe_http_error(qe_http_conn *c, int status, const char *type, const char *msg) {
    qe_str m = { 0 };
    qe_str_json_str(&m, msg, strlen(msg));
    const char *mjson = m.p ? m.p : "\"\"";
    if (c->streamed) {
        /* Headers are already out, so the only honest carrier left is the
         * stream itself: an SSE error event, then stop. */
        qe_str s = { 0 };
        qe_str_appendf(&s, "{\"error\":{\"message\":%s,\"type\":\"%s\"}}", mjson,
                       type ? type : "server_error");
        int rc = qe_http_sse_data(c, s.p ? s.p : "{}");
        qe_str_free(&s);
        qe_str_free(&m);
        return rc;
    }
    if (c->responded) {
        qe_str_free(&m);
        return -1;
    }
    qe_str s = { 0 };
    qe_str_appendf(&s, "{\"error\":{\"message\":%s,\"type\":\"%s\",\"param\":null,\"code\":%d}}",
                   mjson, type ? type : "invalid_request_error", status);
    int rc = qe_http_json(c, status, s.p ? s.p : "{}");
    qe_str_free(&s);
    qe_str_free(&m);
    return rc;
}

int qe_http_sse_head(qe_http_conn *c) {
    if (c->responded || c->streamed)
        return -1;
    c->streamed = 1;
    return write_head(c, 200, "text/event-stream", -1, 0, 1);
}

int qe_http_sse_data(qe_http_conn *c, const char *data) {
    if (!c->streamed)
        return -1;
    qe_str s = { 0 };
    for (const char *p = data; p && *p;) {
        const char *nl = strchr(p, '\n');
        size_t n = nl ? (size_t)(nl - p) : strlen(p);
        qe_str_appendf(&s, "data: %.*s\n", (int)n, p);
        if (!nl)
            break;
        p = nl + 1;
    }
    if (!data || !*data)
        qe_str_appendf(&s, "data: ");
    qe_str_append(&s, "\n", 1);
    int rc = qe_http_write(c, s.p, s.len);
    qe_str_free(&s);
    return rc;
}

int qe_http_sse_done(qe_http_conn *c) {
    if (!c->streamed)
        return -1;
    return qe_http_write(c, "data: [DONE]\n\n", 14);
}

int qe_http_sse_comment(qe_http_conn *c, const char *text) {
    if (!c->streamed)
        return -1;
    /* Only the caller's first line survives: a comment body containing a newline
     * would end the comment mid-sentence and swallow the event after it. */
    if (!text)
        text = "";
    const char *nl = strchr(text, '\n');
    int n = nl ? (int)(nl - text) : (int)strlen(text);
    qe_str s = { 0 };
    qe_str_appendf(&s, ": %.*s\n\n", n, text);
    int rc = qe_http_write(c, s.p, s.len);
    qe_str_free(&s);
    return rc;
}

int qe_http_vprintf_stream(qe_http_conn *c, const char *fmt, va_list ap) {
    char tmp[1024];
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    if (n < 0)
        return -1;
    if ((size_t)n < sizeof tmp)
        return qe_http_write(c, tmp, (size_t)n);
    char *big = malloc((size_t)n + 1);
    if (!big)
        return -1;
    va_list ap2;
    va_copy(ap2, ap);
    vsnprintf(big, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    int rc = qe_http_write(c, big, (size_t)n);
    free(big);
    return rc;
}

int qe_http_printf_stream(qe_http_conn *c, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = qe_http_vprintf_stream(c, fmt, ap);
    va_end(ap);
    return rc;
}

int qe_http_peer(const qe_http_conn *c, char *buf, size_t n) {
    snprintf(buf, n, "%s", c->peer);
    return 0;
}

/* ------------------------------------------------------------ per-connection */

struct conn_job {
    int fd;
    qe_http_handler h;
    void *ctx;
    const qe_http_opts *o;
    char peer[128];
};

static void serve_conn(struct conn_job *j) {
    qe_http_conn c = { .fd = j->fd, .o = j->o };
    snprintf(c.peer, sizeof c.peer, "%s", j->peer);
    char *buf = NULL;
    size_t cap = 0, len = 0;
    int keep = 1;

    for (int rq = 0; keep && rq < j->o->max_requests; rq++) {
        long eom = read_head(&c, &buf, &cap, &len);
        if (eom < 0) {
            qe_http_error(&c, 431, "invalid_request_error", "request head over the limit");
            break;
        }
        if (eom == 0)
            break; /* client closed, or stalled long enough */
        qe_http_req req;
        memset(&req, 0, sizeof req);
        req.keep_alive = 1;
        int err_status = 0;
        const char *why = "";
        c.responded = 0;
        c.streamed = 0;
        c.dead = 0;
        if (parse_head(&c, buf, (size_t)eom, &req, &err_status, &why)) {
            qe_http_error(&c, err_status ? err_status : 400, "invalid_request_error", why);
            break;
        }
        /* Content-Length is the only framing this server reads. Headers already
         * consumed: the remainder of buf, then the socket. */
        size_t have = len - (size_t)eom - 2;
        const char *cl = qe_http_hdr(&req, "content-length");
        const char *te = qe_http_hdr(&req, "transfer-encoding");
        if (te) {
            qe_http_error(&c, 501, "invalid_request_error",
                          "chunked request bodies are not supported; send Content-Length");
            break;
        }
        size_t need = cl ? (size_t)strtoull(cl, NULL, 10) : 0;
        if (need > j->o->max_body) {
            qe_http_error(&c, 413, "invalid_request_error", "request body over the limit");
            break;
        }
        if (need) {
            char *body = malloc(need + 1);
            if (!body) {
                qe_http_error(&c, 503, "server_error", "out of memory");
                break;
            }
            size_t got = have < need ? have : need;
            memcpy(body, buf + eom + 2, got);
            while (got < need) {
                ssize_t k = recv(c.fd, body + got, need - got, 0);
                if (k > 0) {
                    got += (size_t)k;
                    continue;
                }
                if (k < 0 && errno == EINTR)
                    continue;
                break;
            }
            if (got < need) {
                free(body);
                qe_http_error(&c, 400, "invalid_request_error", "body shorter than Content-Length");
                break;
            }
            body[need] = 0;
            req.body = body;
            req.body_len = need;
            len = 0; /* the whole body was consumed */
            /* buf still holds a later pipelined request, if any: not
             * supported, and not something an OpenAI client sends. */
            j->h(&c, &req, j->ctx);
            free((void *)req.body);
        } else {
            len = 0;
            j->h(&c, &req, j->ctx);
        }
        if (c.dead || c.streamed || !req.keep_alive)
            keep = 0;
        if (!c.responded && !c.streamed)
            qe_http_error(&c, 500, "server_error", "handler produced no response");
    }
    close(c.fd);
    free(buf);
}

static pthread_mutex_t live_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t live_idle = PTHREAD_COND_INITIALIZER;
static int live_conns;

static void *conn_thread(void *vp) {
    struct conn_job *j = vp;
    serve_conn(j);
    pthread_mutex_lock(&live_lock);
    if (!--live_conns)
        pthread_cond_broadcast(&live_idle);
    pthread_mutex_unlock(&live_lock);
    free(j);
    return NULL;
}

int qe_http_drain(int grace_secs, int *live) {
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += grace_secs > 0 ? (time_t)grace_secs : 0;
    pthread_mutex_lock(&live_lock);
    while (live_conns && pthread_cond_timedwait(&live_idle, &live_lock, &until) != ETIMEDOUT)
        ;
    int left = live_conns;
    pthread_mutex_unlock(&live_lock);
    if (live)
        *live = left;
    return !left;
}

/* --------------------------------------------------------------------- listen */

void qe_http_opts_default(qe_http_opts *o) {
    if (!o->max_head)
        o->max_head = HEAD_DEFAULT;
    if (!o->max_body)
        o->max_body = BODY_DEFAULT;
    if (!o->max_conns)
        o->max_conns = CONN_DEFAULT;
    if (!o->backlog)
        o->backlog = 128;
    if (!o->idle_secs)
        o->idle_secs = 300;
    if (!o->write_secs)
        o->write_secs = 30;
    if (!o->sndbuf)
        o->sndbuf = SNDBUF_DEFAULT;
    if (!o->max_requests)
        o->max_requests = 64;
    if (!o->id)
        o->id = "qe/0.1";
}

int qe_http_listen(const qe_http_opts *o) {
    signal(SIGPIPE, SIG_IGN);
    int fd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC, 0);
    int family = AF_INET6;
    if (fd < 0) {
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        family = AF_INET;
    }
    if (fd < 0)
        return -1;
    int one = 1, zero = 0;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    /* Dual-stack on purpose. Binding AF_INET6 with V6ONLY gives every client
     * that resolves "localhost" to 127.0.0.1 a connection refused, which reads
     * as a crashed server. */
    if (family == AF_INET6)
        setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);
    struct sockaddr_in6 a6 = { 0 };
    struct sockaddr_in a4 = { 0 };
    socklen_t alen;
    void *ap;
    if (family == AF_INET6) {
        a6.sin6_family = AF_INET6;
        a6.sin6_port = htons((uint16_t)o->port);
        a6.sin6_addr = in6addr_any;
        ap = &a6;
        alen = sizeof a6;
    } else {
        a4.sin_family = AF_INET;
        a4.sin_port = htons((uint16_t)o->port);
        a4.sin_addr.s_addr = htonl(INADDR_ANY);
        ap = &a4;
        alen = sizeof a4;
    }
    if (bind(fd, ap, alen) || listen(fd, o->backlog)) {
        int e = errno;
        close(fd);
        errno = e;
        return -1;
    }
    return fd;
}

volatile sig_atomic_t qe_http_stop;

int qe_http_serve(int lfd, qe_http_handler h, void *ctx, const qe_http_opts *o) {
    /* Lives as long as the accept loop: connection threads borrow it. serve()
     * only returns when the listener is broken, at which point the process is
     * about to exit anyway. */
    qe_http_opts d = *o;
    qe_http_opts_default(&d);
    for (;;) {
        if (qe_http_stop) {
            errno = ESHUTDOWN;
            return 0;
        }
        struct sockaddr_storage ss;
        socklen_t sl = sizeof ss;
        int fd = accept4(lfd, (struct sockaddr *)&ss, &sl, SOCK_CLOEXEC | SOCK_NONBLOCK);
        if (fd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == ECONNABORTED)
                continue;
            if (errno == EMFILE || errno == ENFILE) {
                /* Nothing to do but tell somebody and hope it passes. */
                usleep(50000);
                continue;
            }
            return -1;
        }
        struct conn_job *j = calloc(1, sizeof *j);
        if (!j) {
            close(fd);
            continue;
        }
        j->fd = fd;
        j->h = h;
        j->ctx = ctx;
        j->o = &d;
        char host[NI_MAXHOST] = "?";
        if (getnameinfo((struct sockaddr *)&ss, sl, host, sizeof host, NULL, 0,
                        NI_NUMERICHOST) == 0) {
            /* keep going */
        }
        snprintf(j->peer, sizeof j->peer, "%s", host);
        pthread_mutex_lock(&live_lock);
        int over = live_conns >= d.max_conns;
        if (!over)
            live_conns++;
        pthread_mutex_unlock(&live_lock);
        if (over) {
            const char *busy = "HTTP/1.1 503 Service Unavailable\r\n"
                               "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(fd, busy, strlen(busy));
            close(fd);
            free(j);
            continue;
        }
        /* The accepted socket inherits the listener's NONBLOCK flag, and the
         * handler is written against blocking reads. */
        int fl = fcntl(fd, F_GETFL, 0);
        if (fl >= 0)
            fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
        struct timeval tv = { .tv_sec = d.idle_secs, .tv_usec = 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        /* A stalled reader and a slow one are different problems, and the same
         * number cannot serve both. Reading is where a keep-alive client thinks:
         * an agent runs a tool for three minutes and reuses the connection, so a
         * read is allowed to go quiet for a long time. Writing is where a stream
         * goes, and a stream that cannot be written means nobody is watching -
         * and behind a blocked write the engine keeps generating, at one core
         * per pool thread, for the rest of the turn. Measured: a client killed
         * mid-stream held the engine 600 s (the old single timeout) and 19 cores
         * at 1900% producing bytes nobody read. */
        struct timeval wtv = { .tv_sec = d.write_secs, .tv_usec = 0 };
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &wtv, sizeof wtv);
        /* Two knobs, one policy. `SO_SNDBUF` says how much of a stream the kernel
         * will hold for a reader that has stopped reading, and `SO_SNDTIMEO` says
         * how long a write may wait once that queue is full; either one on its own
         * is no policy at all, because a timeout can only fire on a queue that
         * actually fills. Measured against this host's 4 MiB default with a client
         * that read 512 bytes and then nothing: the queue reached 448 KB with the
         * server still generating happily, because a turn of 4096 deltas is ~700 KB
         * and it all fit. Asking for 256 KB is no better: Linux doubles the value,
         * and a 2631-delta answer measured 435 KB of stream, which also fit, so the
         * whole turn was produced for nobody and the timeout never fired. The queue
         * has to be smaller than a turn to be a bound at all, so the default is the
         * 64 KB below - ~128 KB in the kernel, about a minute of deltas at 13 tok/s.
         * A reader keeping up never sees the difference: it is the engine, not the
         * client, that sets the pace at 2 KB/s. */
        int sndbuf = d.sndbuf;
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &(int){ 1 }, sizeof(int));
        pthread_t th;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        pthread_attr_setstacksize(&at, 256u << 10);
        if (pthread_create(&th, &at, conn_thread, j)) {
            close(fd);
            free(j);
            pthread_mutex_lock(&live_lock);
            live_conns--;
            pthread_mutex_unlock(&live_lock);
        }
        pthread_attr_destroy(&at);
    }
}
