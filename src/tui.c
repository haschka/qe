/* The terminal interface. See src/tui.h for what it is for and the three rules it
   is built under; this file is the drawing.

   One frame, top to bottom:

     header  what engine this is, where it listens, how long it has been up
     INGEST  how far through the prompt the network is, right now
     EMIT    how many tokens the answer has, and how fast they are coming
     TOTAL   what this process has done since it started
     CACHE   the turn cache: rewind points held, and what they saved
     log     the finished requests, newest last

   INGEST and EMIT are always both drawn, the idle one showing its last value,
   because "which of the two is it doing" is the question the panel exists to
   answer, and a blank line where a number used to be does not answer it. The two
   labels are the same width and in the same columns for the same reason: a label
   that moves is a label that has to be read again.

   Everything is drawn to the width the terminal reports, on every frame. Nothing
   here wraps, nothing here scrolls, and a frame that has not changed writes no
   bytes at all - see the renderer for what each of those three is worth.

   Nothing a model wrote is ever drawn. The counters are integers, the request ids
   are made in serve.c, and the history lines come from the request log, which has
   never carried reply text; every byte is additionally scrubbed, because a screen
   is a log like any other and a control token rendered once is a control token the
   next reader copies. */
#define _GNU_SOURCE
#include "tui.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define LOG_LINES 5   /* finished requests the panel keeps room for */
/* Storage for one composed line, escapes included. One line is at most WIDTH_MAX
 * columns, a column is at most four bytes, and every segment can carry a style on
 * and off again: 400*4 + 400*8 would be the honest number and 1800 is the one that
 * still lets a frame be two screens of static. Overrun is not possible either way -
 * the composer stops rather than truncating an escape in half - this is about having
 * room to draw a rule across a wide terminal at all. */
#define LINE_CHARS 1800

enum { IDLE = 0, INGESTING = 1, EMITTING = 2 };

/* Counters, not state: every one is a number a token path increments and the render
 * thread reads. Nothing here is worth a lock, and the render thread takes none of
 * the engine's, which is the whole reason a status surface cannot slow a decode. */
static struct {
    _Atomic long phase;
    _Atomic long req_done;  /* prompt tokens through the network, this request */
    _Atomic long req_total; /* what its prompt encoded to */
    _Atomic long req_emitted;
    _Atomic double req_started;
    _Atomic long long ingested; /* everything since the process started */
    _Atomic long long emitted;
    _Atomic long long cached;   /* prompt tokens that did not have to be re-read */
    _Atomic long requests;
    _Atomic long slots;      /* rewind points the budget buys */
    _Atomic long held;       /* ... and how many exist */
    _Atomic long long cache_bytes;
    _Atomic long hits;
    _Atomic long snap_us; /* the last snapshot's copy, microseconds */
    /* When each half of the request in hand began and last moved, so that each rate is
     * its own tokens over its own seconds: the prompt's from the first token that had to
     * be read, the answer's from the first token it emitted. One clock for both is how an
     * answer used to be quoted at a third of its speed after a long prompt. */
    _Atomic long ing_base;  /* prompt tokens already standing when the request came */
    _Atomic double ing_t1;  /* the prompt was read: first emit, or the end */
    _Atomic long ing_last;  /* prompt tokens read when it was */
    _Atomic double emit_t0; /* the first emitted token */
    _Atomic double emit_t1; /* the latest one */
    _Atomic double req_end; /* the request finished: its clock stops here */
} c;

static double started;

/* The header lines and the log ring are strings, so they get a mutex. Written once
 * a request, read on every frame that is drawn: cheap either way, and a torn header
 * line would be an embarrassment rather than a bug. */
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static char hdr[2][LINE_CHARS];
static char cur_req[LINE_CHARS];
static char logs[LOG_LINES][LINE_CHARS];
static int nlog, log_next;

static int running;
static pthread_t th;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ feeding */

void qe_ui_head(const char *head, const char *sub) {
    pthread_mutex_lock(&lk);
    if (head)
        snprintf(hdr[0], sizeof hdr[0], "%s", head);
    if (sub)
        snprintf(hdr[1], sizeof hdr[1], "%s", sub);
    pthread_mutex_unlock(&lk);
}

void qe_ui_request(const char *kind, const char *id) {
    pthread_mutex_lock(&lk);
    snprintf(cur_req, sizeof cur_req, "%s %s", kind ? kind : "request", id ? id : "-");
    pthread_mutex_unlock(&lk);
}

/* One finished request, stored oldest first in a ring. Scrubbed for the reason in
 * the header: this is a screen, and a marker that reaches one is a marker that
 * leaves it. */
void qe_ui_log(const char *line) {
    if (!line)
        return;
    pthread_mutex_lock(&lk);
    char *dst = logs[log_next];
    size_t j = 0;
    for (const char *s = line; *s && j + 1 < sizeof logs[0]; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == '\n' || ch == '\r')
            break;
        if (ch == '\t')
            ch = ' ';
        else if (ch < 0x20 || ch == 0x7f || ch == '<' || ch == '>')
            ch = '.';
        dst[j++] = (char)ch;
    }
    dst[j] = 0;
    log_next = (log_next + 1) % LOG_LINES;
    if (nlog < LOG_LINES)
        nlog++;
    pthread_mutex_unlock(&lk);
}

void qe_ui_gen_begin(long prompt_tokens, long already) {
    atomic_store_explicit(&c.req_total, prompt_tokens, memory_order_relaxed);
    atomic_store_explicit(&c.req_done, already, memory_order_relaxed);
    atomic_store_explicit(&c.req_emitted, 0, memory_order_relaxed);
    atomic_store_explicit(&c.ing_base, already, memory_order_relaxed);
    atomic_store_explicit(&c.ing_t1, 0.0, memory_order_relaxed);
    atomic_store_explicit(&c.ing_last, already, memory_order_relaxed);
    atomic_store_explicit(&c.emit_t0, 0.0, memory_order_relaxed);
    atomic_store_explicit(&c.emit_t1, 0.0, memory_order_relaxed);
    atomic_store_explicit(&c.req_end, 0.0, memory_order_relaxed);
    atomic_store_explicit(&c.req_started, now_s(), memory_order_relaxed);
    atomic_store_explicit(&c.phase, INGESTING, memory_order_relaxed);
    atomic_fetch_add_explicit(&c.requests, 1, memory_order_relaxed);
}

void qe_ui_ingest(long done, long total) {
    atomic_store_explicit(&c.req_done, done, memory_order_relaxed);
    if (total > 0)
        atomic_store_explicit(&c.req_total, total, memory_order_relaxed);
    atomic_store_explicit(&c.phase, INGESTING, memory_order_relaxed);
}

void qe_ui_ingested(long tokens) {
    if (tokens > 0)
        atomic_fetch_add_explicit(&c.ingested, (long long)tokens, memory_order_relaxed);
}

static void ingest_closed(double t) {
    if (atomic_load_explicit(&c.ing_t1, memory_order_relaxed) <= 0.0) {
        atomic_store_explicit(&c.ing_last, atomic_load_explicit(&c.req_done, memory_order_relaxed),
                              memory_order_relaxed);
        atomic_store_explicit(&c.ing_t1, t, memory_order_relaxed);
    }
}

void qe_ui_emit(void) {
    const double t = now_s();
    ingest_closed(t);
    if (atomic_load_explicit(&c.emit_t0, memory_order_relaxed) <= 0.0)
        atomic_store_explicit(&c.emit_t0, t, memory_order_relaxed);
    atomic_store_explicit(&c.emit_t1, t, memory_order_relaxed);
    atomic_fetch_add_explicit(&c.req_emitted, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&c.emitted, 1, memory_order_relaxed);
    atomic_store_explicit(&c.phase, EMITTING, memory_order_relaxed);
}

void qe_ui_gen_end(void) {
    ingest_closed(now_s());
    atomic_store_explicit(&c.req_end, now_s(), memory_order_relaxed);
    atomic_store_explicit(&c.phase, IDLE, memory_order_relaxed);
}

void qe_ui_cached(long tokens) {
    if (tokens > 0)
        atomic_fetch_add_explicit(&c.cached, tokens, memory_order_relaxed);
}

void qe_ui_rewind(long tokens) {
    if (tokens > 0)
        atomic_fetch_add_explicit(&c.hits, 1, memory_order_relaxed);
}

void qe_ui_cache(long slots, long held, long long bytes, long snapshot_us) {
    atomic_store_explicit(&c.slots, slots, memory_order_relaxed);
    atomic_store_explicit(&c.held, held, memory_order_relaxed);
    atomic_store_explicit(&c.cache_bytes, bytes, memory_order_relaxed);
    if (snapshot_us >= 0)
        atomic_store_explicit(&c.snap_us, snapshot_us, memory_order_relaxed);
}

/* ------------------------------------------------------------------ drawing */

static void comma(char *dst, size_t n, long long v) {
    char buf[24];
    int len = snprintf(buf, sizeof buf, "%lld", v);
    size_t o = 0;
    for (int i = 0; i < len; i++) {
        /* A separator before any digit that starts a group, counting from the
         * right, which is the only way to know where the groups are. */
        if (i && (len - i) % 3 == 0 && o + 1 < n)
            dst[o++] = ',';
        if (o + 1 < n)
            dst[o++] = buf[i];
    }
    dst[o] = 0;
}

/* A running total in at most six columns: 987,654 stays exact, past that it is
 * 1.23M - a process that has read a hundred million tokens needs its count, not its
 * last digit, and the row has eighty columns to say five things in. */
static void short_count(char *dst, size_t n, long long v) {
    if (v < 1000000)
        comma(dst, n, v);
    else if (v < 1000000000LL)
        snprintf(dst, n, "%.2fM", (double)v / 1e6);
    else
        snprintf(dst, n, "%.2fG", (double)v / 1e9);
}

/* The renderer.
 *
 * Three rules, all of them learned the hard way:
 *
 * 1. Nothing is written unless it changed. The panel is a diff of two frames, and a
 *    frame that matches the last one costs zero bytes. A terminal that is told the
 *    same thing ten times a second is a terminal whose contents visibly swim.
 * 2. Nothing is ever written that could wrap. Every line is composed to the width the
 *    terminal reported, and the width is asked again on every frame, because a
 *    wrapping line is a row the redraw does not know about and one unknown row makes
 *    every later frame land one row off - which is precisely the crawl this file used
 *    to ship with, along with a cursor-up that was one row too many.
 * 3. One frame is one update. The lines between the two synchronise markers arrive as
 *    a single repaint on terminals that implement the mode, and as an ordinary burst
 *    on the ones that do not, so there is nothing to negotiate.
 *
 * Not ncurses. This surface draws numbers into a fixed frame, and the same bytes have
 * to be legible in a log file when stdout is not a terminal - `--ui on` into a file is
 * what `make ui` asserts, and an alternate screen and a terminfo database would make
 * that file a screen instead of text. What was wrong was the redraw, not the absence
 * of a library. */

#define MAX_ROWS 24    /* the tallest panel this file draws */
#define WIDTH_DEFAULT 80
#define WIDTH_MIN 48   /* narrower than this the counters do not fit anyway */
#define WIDTH_MAX 400  /* a rule is one byte per column times four; see LINE_CHARS */
#define TICK_MS 100    /* how often the frame is recomputed; see rule 1 */

/* The two phase labels keep their letters in the same columns, because the question
 * the panel exists to answer is "which of these two is it doing right now", and a
 * label that moves is a label that has to be re-read. */
enum { ST_PLAIN = 0, ST_HEAD, ST_DIM, ST_LABEL, ST_BAR, ST_HOT, ST_N };
static const char *sgr[ST_N] = {
    "", "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m",
};

static int tty;  /* stdout is a terminal: it is measured, styled and synchronised */
static int color;
static int cols = WIDTH_DEFAULT, rows = 24;
static _Atomic int stop_requested;

/* A line is composed once into a string, and that string is what the diff compares,
 * which is why the styles are baked in rather than re-derived at write time. */
typedef struct {
    char *dst;
    size_t n;
    size_t len; /* bytes, escapes included */
    int used;   /* columns the terminal will spend */
    int cols;
} ln;

/* Columns, not bytes and not codepoints: every glyph this file draws is one column
 * wide whether it arrives in one byte or three, so counting sequences is counting
 * columns, and cutting a sequence in half is drawing garbage. */
static int vwidth(const char *s) {
    int w = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
        if (*p < 0x80 || *p >= 0xc0)
            w++;
    return w;
}

static void ln_open(ln *l, char *dst, size_t n, int width) {
    l->dst = dst;
    l->n = n;
    l->len = 0;
    l->used = 0;
    l->cols = width;
    dst[0] = 0;
}

/* Appends n bytes and keeps the terminator inside the buffer, always. A style escape
 * cut in half is not a shorter line, it is every character after it in the wrong
 * colour, so the composer gives up on a segment instead. */
static void ln_cat(ln *l, const char *s, size_t n) {
    if (n > l->n - l->len - 1)
        n = l->n - l->len - 1;
    memcpy(l->dst + l->len, s, n);
    l->len += n;
    l->dst[l->len] = 0;
}

static void ln_put(ln *l, const char *s, int st) {
    if (l->used >= l->cols || l->len + 9 >= l->n)
        return;
    /* The room for one more four-byte cell plus a reset is what the loop asks for
     * below, which is why a segment can never end mid-glyph or without its reset. */
    int styled = st != ST_PLAIN && color;
    if (styled)
        ln_cat(l, sgr[st], strlen(sgr[st]));
    for (const unsigned char *p = (const unsigned char *)s;
         *p && l->used < l->cols && l->len + 9 < l->n;) {
        size_t step = (size_t)(*p >= 0xf0 ? 4 : *p >= 0xe0 ? 3 : *p >= 0xc0 ? 2 : 1);
        /* strnlen and not strlen*step: a segment whose last glyph was cut in half by
         * whatever produced it has to end the loop, not walk off its end copying stack
         * into the middle of the line, which is how a truncated line arrives looking
         * like a complete one that simply stopped early. */
        size_t have = strnlen((const char *)p, step);
        ln_cat(l, (const char *)p, have);
        l->used++;
        if (have < step)
            break;
        p += step;
    }
    if (styled)
        ln_cat(l, "\033[0m", 4);
}

static void ln_addf(ln *l, int st, const char *fmt, ...) {
    char buf[LINE_CHARS];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    ln_put(l, buf, st);
}

/* Numbers are right-aligned into columns wide enough for the largest they will ever
 * be, so that a counter gaining a digit does not move its neighbours. */
static void ln_raddf(ln *l, int width, int st, const char *fmt, ...) {
    char buf[LINE_CHARS];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    for (int i = vwidth(buf); i < width; i++)
        ln_put(l, " ", ST_PLAIN);
    ln_put(l, buf, st);
}

/* Blank up to an absolute column: how the right-hand edge of a line gets pinned
 * without the line knowing what is to its left. Never pads to less than one space,
 * because on a narrow terminal a tail that starts where the numbers ended reads as
 * "…11,758 tok0.0 s", and the tail running off the right edge is the better failure. */
static void ln_to(ln *l, int col) {
    if (col < l->used + 1)
        col = l->used + 1;
    while (l->used < col && l->used < l->cols)
        ln_put(l, " ", ST_PLAIN);
}

/* A rule is the one run the panel draws a lot of, and it is drawn as one style, one
 * run, one reset: a reset per glyph costs three times the bytes of the glyph and,
 * more to the point, makes every rule line in the diff look different when it is not. */
static void ln_rule(ln *l, const char *glyph) {
    if (l->used >= l->cols || l->len + 16 >= l->n)
        return;
    size_t gl = strlen(glyph);
    int styled = color && gl;
    if (styled)
        ln_cat(l, sgr[ST_DIM], strlen(sgr[ST_DIM]));
    while (l->used < l->cols && l->len + gl + 5 < l->n) {
        ln_cat(l, glyph, gl);
        l->used++;
    }
    if (styled)
        ln_cat(l, "\033[0m", 4);
}

/* The glyphs, chosen once. A C or POSIX locale cannot decode a box-drawing character,
 * and every line of the panel then arrives as three bytes of noise; the ASCII frame is
 * uglier on purpose and correct on every terminal. */
static struct {
    const char *rule, *full, *empty, *arrow, *sep;
} g = { "-", "#", ".", "|", " - " };

static void glyphs_init(void) {
    static const char *env[] = { "LC_ALL", "LC_CTYPE", "LANG", NULL };
    for (int i = 0; env[i]; i++) {
        const char *v = getenv(env[i]);
        if (!v)
            continue;
        for (const char *s = v; *s; s++)
            if (!strncasecmp(s, "UTF-8", 5) || !strncasecmp(s, "utf8", 4)) {
                g.rule = "─";
                g.full = "█";
                g.empty = "░";
                g.arrow = "▸";
                g.sep = " · ";
                return;
            }
    }
}

/* The width the terminal has, asked again every frame. An ioctl four times a second
 * is nothing, and a cached width is a panel that stops fitting the moment somebody
 * drags a window. */
static int width_of(void) {
    struct winsize ws;
    if (tty && ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col >= WIDTH_MIN)
        return ws.ws_col > WIDTH_MAX ? WIDTH_MAX : (int)ws.ws_col;
    const char *e = getenv("COLUMNS");
    if (e) {
        long v = strtol(e, NULL, 10);
        if (v >= WIDTH_MIN && v <= WIDTH_MAX)
            return (int)v;
    }
    return WIDTH_DEFAULT;
}

static int height_of(void) {
    struct winsize ws;
    if (tty && ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row >= 8)
        return (int)ws.ws_row;
    const char *e = getenv("LINES");
    if (e) {
        long v = strtol(e, NULL, 10);
        if (v >= 8 && v <= 4000)
            return (int)v;
    }
    return 24;
}

static void bar(char *dst, size_t n, int width, long done, long total) {
    int fill = 0;
    if (total > 0) {
        double f = (double)done / (double)total;
        if (f < 0)
            f = 0;
        if (f > 1)
            f = 1;
        fill = (int)(f * width + 0.5);
    }
    if (fill > width)
        fill = width;
    size_t o = 0;
    for (int i = 0; i < width && o + 8 < n; i++) {
        const char *cell = i < fill ? g.full : g.empty;
        size_t l = strlen(cell);
        memcpy(dst + o, cell, l);
        o += l;
    }
    dst[o] = 0;
}

/* Seconds, at the granularity that line is worth changing for. The uptime used to be
 * drawn to a tenth of a second, which is a line that differs from the last one every
 * frame, forever, on a process with nothing to do. */
static void hms(char *dst, size_t n, double s) {
    if (s < 0)
        s = 0;
    if (s < 60)
        snprintf(dst, n, "%lds", (long)s);
    else if (s < 3600)
        snprintf(dst, n, "%ldm%02lds", (long)(s / 60), (long)((long)s % 60));
    else
        snprintf(dst, n, "%ldh%02ldm", (long)(s / 3600), (long)(((long)s % 3600) / 60));
}

/* How long this request has been held, to a tenth: the one place where the tenth is
 * the answer, and only while a request is in hand. It is one row's last four columns,
 * repainted alone inside a synchronised update, which is a ticking clock and not a
 * flicker. */
static void secs10(char *dst, size_t n, double s) {
    if (s < 0)
        s = 0;
    if (s < 3600)
        snprintf(dst, n, "%.1f s", s);
    else
        hms(dst, n, s);
}

/* Copies at most `cols` columns of `s` into `dst`, stopping on a glyph boundary, and
 * says whether anything was left behind. ln_put clips a line at the width already;
 * what it cannot do is admit to it, and a row that silently loses its end is a row
 * that reports a truncated fact as a complete one. */
static int clip_cols(char *dst, size_t n, const char *s, int want) {
    size_t len = 0;
    int w = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p && w < want && len + 4 < n) {
        size_t step = (size_t)(*p >= 0xf0 ? 4 : *p >= 0xe0 ? 3 : *p >= 0xc0 ? 2 : 1);
        memcpy(dst + len, p, step);
        len += step;
        w++;
        p += step;
    }
    dst[len] = 0;
    return *p != 0;
}

/* Builds one frame into `out` and returns how many lines it filled. `budget` is the
 * tallest panel the screen will take without scrolling the banner off the top. */
static int frame(char out[][LINE_CHARS], int width, int budget) {
    long phase = atomic_load_explicit(&c.phase, memory_order_relaxed);
    long done = atomic_load_explicit(&c.req_done, memory_order_relaxed);
    long total = atomic_load_explicit(&c.req_total, memory_order_relaxed);
    long emit = atomic_load_explicit(&c.req_emitted, memory_order_relaxed);
    long long ing = atomic_load_explicit(&c.ingested, memory_order_relaxed);
    long long emo = atomic_load_explicit(&c.emitted, memory_order_relaxed);
    long long saved = atomic_load_explicit(&c.cached, memory_order_relaxed);
    long reqs = atomic_load_explicit(&c.requests, memory_order_relaxed);
    long slots = atomic_load_explicit(&c.slots, memory_order_relaxed);
    long held = atomic_load_explicit(&c.held, memory_order_relaxed);
    long long cbytes = atomic_load_explicit(&c.cache_bytes, memory_order_relaxed);
    long hits = atomic_load_explicit(&c.hits, memory_order_relaxed);
    long sus = atomic_load_explicit(&c.snap_us, memory_order_relaxed);
    double t0 = atomic_load_explicit(&c.req_started, memory_order_relaxed);

    char who[LINE_CHARS], head[LINE_CHARS], sub[LINE_CHARS], hist[LOG_LINES][LINE_CHARS];
    pthread_mutex_lock(&lk);
    snprintf(who, sizeof who, "%s", cur_req[0] ? cur_req : "idle");
    snprintf(head, sizeof head, "%s", hdr[0]);
    snprintf(sub, sizeof sub, "%s", hdr[1]);
    int nl = nlog;
    for (int i = 0; i < nl; i++) {
        int at = (log_next - nl + i + LOG_LINES * 2) % LOG_LINES;
        snprintf(hist[i], sizeof hist[i], "%s", logs[at]);
    }
    pthread_mutex_unlock(&lk);

    const double tnow = now_s();
    /* The request's clock runs while it is in hand and stops when it is answered: an idle
     * panel still counting is a panel saying a finished request is taking longer. */
    const double rend = atomic_load_explicit(&c.req_end, memory_order_relaxed);
    double el = t0 > 0 ? (phase == IDLE && rend > t0 ? rend : tnow) - t0 : 0.0;
    const long base = atomic_load_explicit(&c.ing_base, memory_order_relaxed);
    const double it1 = atomic_load_explicit(&c.ing_t1, memory_order_relaxed);
    const long ilast = atomic_load_explicit(&c.ing_last, memory_order_relaxed);
    const double et0 = atomic_load_explicit(&c.emit_t0, memory_order_relaxed);
    const double et1 = atomic_load_explicit(&c.emit_t1, memory_order_relaxed);

    /* Rates. Each half is its own tokens over its own seconds, and the live figure is
     * a window over the last three seconds of samples rather than an average since the
     * start, so a prompt that slows as its context grows says so while it does. The
     * samples are this thread's own (it is the only reader), restarted with every
     * request. The strings are recomputed twice a second and held in between: a rate
     * repainted ten times a second is a number nobody can read, and a row that differs
     * every tick is a row the terminal repaints every tick. */
#define RATE_WIN 3.0
#define RATE_HOLD 0.5
    static struct {
        double t;
        long d, e;
    } smp[64];
    static int sn, sh;
    static double s_req = -1.0, s_shown = 0.0;
    static char s_ing[48], s_emit[48];
    if (t0 != s_req) {
        s_req = t0;
        sn = sh = 0;
        s_shown = 0.0;
    }
    {
        const int at = (sh + sn) % 64;
        smp[at].t = tnow;
        smp[at].d = done;
        smp[at].e = emit;
        if (sn < 64)
            sn++;
        else
            sh = (sh + 1) % 64;
        while (sn > 2 && tnow - smp[sh].t > RATE_WIN) {
            sh = (sh + 1) % 64;
            sn--;
        }
    }
    const double span = sn > 1 ? tnow - smp[sh].t : 0.0;
    const double ing_win = span >= 0.4 ? (double)(done - smp[sh].d) / span : 0.0;
    const double emit_win = span >= 0.4 ? (double)(emit - smp[sh].e) / span : 0.0;
    /* Averages over each half, which is what an idle row shows: the prompt from its
     * start to its first answer token, the answer from its first token to its last. */
    const double ing_secs = (it1 > 0.0 ? it1 : tnow) - t0;
    const long ing_read = (it1 > 0.0 ? ilast : done) - base;
    const double ing_avg = t0 > 0 && ing_secs > 0.05 && ing_read > 0 ? ing_read / ing_secs : 0.0;
    const double emit_avg = emit > 1 && et1 > et0 ? (double)(emit - 1) / (et1 - et0) : 0.0;
    if (tnow - s_shown >= RATE_HOLD || s_shown == 0.0) {
        s_shown = tnow;
        const double ir = phase == INGESTING && ing_win > 0.0 ? ing_win : ing_avg;
        if (total > 0 && ir > 0.0) {
            if (phase == INGESTING && total > done) {
                char eta[16];
                hms(eta, sizeof eta, (double)(total - done) / ir);
                snprintf(s_ing, sizeof s_ing, "%5.1f tok/s %6s left", ir, eta);
            } else {
                char d[16];
                hms(d, sizeof d, ing_secs);
                snprintf(s_ing, sizeof s_ing, "%5.1f tok/s %6s", ir, d);
            }
        } else if (total > 0 && base >= total) {
            snprintf(s_ing, sizeof s_ing, "all cached");
        } else {
            snprintf(s_ing, sizeof s_ing, "  -.- tok/s");
        }
        const double er = phase == EMITTING && emit_win > 0.0 ? emit_win : emit_avg;
        snprintf(s_emit, sizeof s_emit, "%5.1f tok/s", er);
    }

    char up[32], elh[32], a[24], b[24], h2[24], u[24], e2[24], s2[24], cb[24];
    hms(up, sizeof up, now_s() - started);
    secs10(elh, sizeof elh, el);
    comma(a, sizeof a, done);
    comma(b, sizeof b, total);
    comma(h2, sizeof h2, emit);
    short_count(u, sizeof u, ing);
    short_count(e2, sizeof e2, emo);
    short_count(s2, sizeof s2, saved);
    comma(cb, sizeof cb, cbytes / 1048576);

    /* The frame is header, sub-header when there is one, rule, ingest, emit, rule,
     * total, cache, and then whatever history the screen has room for. The rules and
     * the last two lines go first when the terminal is short, because the two
     * counters are the answer to the question the panel was written for. */
    int show_sub = sub[0] != '\0';
    int show_rules = budget >= 7;
    int show_tail = budget >= 5;
    int max_log = budget - (show_tail ? (show_rules ? 7 : 5) : 3) - (show_sub ? 1 : 0);
    if (max_log < 0)
        max_log = 0;
    if (max_log > LOG_LINES)
        max_log = LOG_LINES;

    int n = 0;
    ln l;

    ln_open(&l, out[n], LINE_CHARS, width);
    ln_put(&l, " ", ST_PLAIN);
    {
        /* The header gives way to the uptime, never the other way round. */
        char hc[LINE_CHARS];
        int room = width - vwidth(up) - 3;
        if (room < 8)
            room = 8;
        int cut = vwidth(head) > room ? clip_cols(hc, sizeof hc, head, room - 2) : 0;
        ln_addf(&l, ST_HEAD, "%s", cut ? hc : head);
        if (cut)
            ln_addf(&l, ST_LABEL, "..");
    }
    ln_to(&l, width - vwidth(up) < l.used ? l.used : width - vwidth(up));
    ln_addf(&l, ST_DIM, "%s", up);
    n++;

    /* The second row: the file the engine is reading that the header has no room
     * for, in half a column of text. Cut to the terminal and marked as cut, because
     * a row that stops in the middle of a name is read as a name that ended there. */
    if (show_sub) {
        char cut[LINE_CHARS];
        int room = width - 3; /* the leading space and the two columns of the marker */
        int truncated = vwidth(sub) > room ? clip_cols(cut, sizeof cut, sub, room) : 0;
        ln_open(&l, out[n], LINE_CHARS, width);
        ln_put(&l, " ", ST_PLAIN);
        ln_addf(&l, ST_DIM, "%s", truncated ? cut : sub);
        if (truncated)
            ln_addf(&l, ST_LABEL, "..");
        n++;
    }

    if (show_rules) {
        ln_open(&l, out[n++], LINE_CHARS, width);
        ln_rule(&l, g.rule);
    }

    /* INGEST: the bar, how far through the prompt, and how fast it is being read.
     *
     *   INGEST ▸ ██████████████░░░░░░░░░░  12,288/20,480   61.2 tok/s  2m14s left
     *
     * The tail is measured and right-aligned on the last column, so the bar takes what
     * the terminal has left and the numbers never slide sideways as they change. */
    ln_open(&l, out[n], LINE_CHARS, width);
    ln_put(&l, " ", ST_PLAIN);
    ln_addf(&l, phase == INGESTING ? ST_LABEL : ST_DIM, "%-7s%s", "INGEST", g.arrow);
    ln_put(&l, " ", ST_PLAIN);
    if (total > 0) {
        char cnt[64];
        char pad[24];
        snprintf(pad, sizeof pad, "%*s", vwidth(b) - vwidth(a) > 0 ? vwidth(b) - vwidth(a) : 0, "");
        snprintf(cnt, sizeof cnt, "%s%s/%s", pad, a, b);
        const int tail = 2 + vwidth(cnt) + 2 + vwidth(s_ing);
        int bw = width - l.used - tail;
        if (bw > 48)
            bw = 48;
        if (bw >= 6) {
            char cells[48 * 4 + 8];
            bar(cells, sizeof cells, bw, done, total);
            /* Not "%.*s" with bw: the precision of printf is bytes, and a byte precision
             * that lands inside a three-byte block cell both shortens the bar and hands
             * ln_put half a character. bw cells is bw columns; the count lives in bar(). */
            ln_addf(&l, ST_BAR, "%s", cells);
        }
        ln_to(&l, width - tail + 2);
        ln_addf(&l, ST_PLAIN, "%s", pad);
        ln_addf(&l, ST_PLAIN, "%s", a);
        ln_addf(&l, ST_DIM, "/%s", b);
        ln_to(&l, width - vwidth(s_ing));
        ln_addf(&l, phase == INGESTING ? ST_PLAIN : ST_DIM, "%s", s_ing);
    } else {
        ln_addf(&l, ST_DIM, "no prompt in hand");
    }
    n++;

    /* EMIT: the answer so far, how fast it is coming, for how long, and for whom.
     *
     *   EMIT   ▸ 1,234 tok   15.6 tok/s   12s                         chat 3f9a1c */
    ln_open(&l, out[n], LINE_CHARS, width);
    ln_put(&l, " ", ST_PLAIN);
    ln_addf(&l, phase == EMITTING ? ST_LABEL : ST_DIM, "%-7s%s", "EMIT", g.arrow);
    ln_put(&l, " ", ST_PLAIN);
    ln_raddf(&l, 6, emit > 0 ? ST_HOT : ST_PLAIN, "%s", h2);
    ln_addf(&l, ST_PLAIN, " tok");
    ln_addf(&l, ST_DIM, "%s", g.sep);
    ln_addf(&l, phase == EMITTING ? ST_PLAIN : ST_DIM, "%s", s_emit);
    ln_addf(&l, ST_DIM, "%s", g.sep);
    ln_addf(&l, ST_PLAIN, "%s", elh);
    {
        /* The request id is the right-hand end of the row and the first thing to go. */
        char wc[LINE_CHARS];
        int room = width - l.used - 2;
        if (room >= 6) {
            int cut = vwidth(who) > room ? clip_cols(wc, sizeof wc, who, room) : 0;
            ln_to(&l, width - (cut ? room : vwidth(who)));
            ln_addf(&l, ST_DIM, "%s", cut ? wc : who);
        }
    }
    n++;

    if (show_rules) {
        ln_open(&l, out[n++], LINE_CHARS, width);
        ln_rule(&l, g.rule);
    }

    if (show_tail) {
        ln_open(&l, out[n], LINE_CHARS, width);
        ln_put(&l, " ", ST_PLAIN);
        ln_addf(&l, ST_LABEL, "%-9s", "total");
        ln_addf(&l, ST_PLAIN, "%s ingested", u);
        ln_addf(&l, ST_DIM, "%s", g.sep);
        ln_addf(&l, ST_PLAIN, "%s emitted", e2);
        ln_addf(&l, ST_DIM, "%s", g.sep);
        ln_addf(&l, ST_PLAIN, "%ld request%s", reqs, reqs == 1 ? "" : "s");
        ln_addf(&l, ST_DIM, "%s", g.sep);
        ln_addf(&l, ST_PLAIN, "%s reused", s2);
        n++;

        ln_open(&l, out[n], LINE_CHARS, width);
        ln_put(&l, " ", ST_PLAIN);
        ln_addf(&l, ST_LABEL, "%-9s", "cache");
        if (slots > 0) {
            ln_addf(&l, held > 0 ? ST_PLAIN : ST_DIM, "%ld of %ld rewind point%s", held, slots,
                    held == 1 ? "" : "s");
            ln_addf(&l, ST_DIM, "%s", g.sep);
            ln_addf(&l, ST_PLAIN, "%s MiB", cb);
            ln_addf(&l, ST_DIM, "%s", g.sep);
            ln_addf(&l, hits > 0 ? ST_HOT : ST_PLAIN, "%ld hit%s", hits, hits == 1 ? "" : "s");
            ln_addf(&l, ST_DIM, "%s", g.sep);
            ln_addf(&l, ST_DIM, "snapshot %.1f ms", (double)sus / 1000.0);
        } else {
            ln_addf(&l, ST_DIM, "no rewind points (--turn-cache)");
        }
        n++;
    }

    /* Clamped at zero: with fewer finished requests than the panel has room for, a
     * negative start repeats the oldest line once per missing one, and a panel that
     * draws the same request five times is read as five requests that went badly. */
    for (int i = nl - max_log > 0 ? nl - max_log : 0; i < nl; i++) {
        if (n >= MAX_ROWS)
            break;
        ln_open(&l, out[n], LINE_CHARS, width);
        ln_put(&l, " ", ST_PLAIN);
        ln_addf(&l, ST_DIM, "%s", hist[i]);
        n++;
    }
    return n;
}

/* The row of the panel the cursor is believed to be on. Every frame ends with the
 * cursor on the panel's last row, which is what makes "move to row i" a subtraction
 * from a number this file knows instead of a question to the terminal. */
static int cursor_row;
static char prev[MAX_ROWS][LINE_CHARS];
static int prev_n;

static void move_to(int row) {
    int d = cursor_row - row;
    if (d == 0) {
        fputs("\r", stdout);
    } else if (d == -1) {
        /* One row down is a newline, not a cursor move: at the bottom of the screen a
         * newline scrolls and stays coherent, while a cursor-down is clamped and
         * writes the new row over the one beneath the panel. */
        fputc('\n', stdout);
        cursor_row = row;
        return;
    } else {
        fputs("\r", stdout);
        fprintf(stdout, "\033[%d%c", d > 0 ? d : -d, d > 0 ? 'A' : 'B');
    }
    cursor_row = row;
}

static void draw(void) {
    static char out[MAX_ROWS][LINE_CHARS];
    int w = width_of(), h = height_of();
    if (w != cols || h != rows) {
        /* A resize moves the panel's own rows under a reflowing terminal, so where the
         * cursor is stops being knowable. Redraw everything from where it is believed
         * to be and clear downwards, which is self-correcting in every case that a
         * cursor-position report would also fix, without reading from stdin. */
        cols = w;
        rows = h;
        for (int i = 0; i < MAX_ROWS; i++)
            prev[i][0] = 0;
        prev_n = 0;
    }
    int budget = (tty ? rows : MAX_ROWS) - 1;
    if (budget > MAX_ROWS)
        budget = MAX_ROWS;
    /* One column less than the terminal has. A line that ends in the last column puts
     * the terminal in its wrap-pending state, and the erase-to-end-of-line this file
     * writes after every row clears the cell the cursor is on, which is the glyph it
     * was just given - a rule across a wide screen would arrive with its last
     * character missing. The spare column is what the erase is for. */
    int n = frame(out, cols - 1, budget);

    /* Rule 1: a frame that says the same thing writes nothing at all. */
    int changed = n != prev_n;
    for (int i = 0; i < n && !changed; i++)
        if (i >= prev_n || strcmp(prev[i], out[i]))
            changed = 1;
    if (!changed) {
        fflush(stdout);
        return;
    }

    if (tty)
        fputs("\033[?2026h", stdout); /* one frame, never two half-frames */
    /* No seeking to row 0 first: every branch of move_to starts at column zero, so
     * the first row this function draws positions itself, and doing it twice put a
     * spare carriage return in front of every frame. */
    for (int i = 0; i < n; i++) {
        if (i < prev_n && !strcmp(prev[i], out[i]))
            continue;
        move_to(i);
        fputs(out[i], stdout);
        fputs("\033[K", stdout);
    }
    for (int i = n; i < prev_n; i++) {
        /* The frame got shorter. Whatever is under those rows is an old number, and an
         * old number in a status panel is a wrong number. */
        move_to(i);
        fputs("\033[K", stdout);
    }
    move_to(n > 0 ? n - 1 : 0);
    if (tty)
        fputs("\033[?2026l", stdout);
    fflush(stdout);

    for (int i = 0; i < n; i++)
        snprintf(prev[i], LINE_CHARS, "%s", out[i]);
    prev_n = n;
}

static void *render(void *vp) {
    (void)vp;
    struct timespec nap = { TICK_MS / 1000, (TICK_MS % 1000) * 1000000L };
    while (!atomic_load_explicit(&stop_requested, memory_order_relaxed)) {
        draw();
        nanosleep(&nap, NULL);
    }
    return NULL;
}

/* ------------------------------------------------------------------ control */

void qe_ui_start(int want, const char *head) {
    if (want == QE_UI_OFF || running)
        return;
    tty = isatty(STDOUT_FILENO);
    if (want == QE_UI_AUTO && !tty)
        return;
    glyphs_init();
    /* Colour into a pipe is noise for whoever greps it, and `make ui` reads this
     * panel's bytes as text: styles are for terminals that were asked for. */
    const char *term = getenv("TERM");
    color = tty && term && strcmp(term, "dumb") && !getenv("NO_COLOR");
    cols = width_of();
    rows = height_of();
    started = now_s();
    if (head)
        qe_ui_head(head, NULL);
    atomic_store_explicit(&stop_requested, 0, memory_order_relaxed);
    if (pthread_create(&th, NULL, render, NULL))
        return;
    running = 1;
    fputs("\033[?25l", stdout); /* the panel owns the screen while it is up */
    fflush(stdout);
}

void qe_ui_stop(void) {
    if (!running)
        return;
    atomic_store_explicit(&stop_requested, 1, memory_order_relaxed);
    pthread_join(th, NULL);
    running = 0;
    /* Leave the frame on the screen and put the cursor below it, so that whatever
     * the shell prints next starts on a clean line. The synchronise markers are off
     * by then because the last frame flushed them; nothing else is owed. */
    fputs("\033[?25h\n", stdout);
    fflush(stdout);
}

int qe_ui_on(void) { return running; }
