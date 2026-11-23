/* OpenAI-compatible surface: /v1/chat/completions, /v1/completions,
 * /v1/models, /health.
 *
 * Three things in here are deliberate and worth knowing before changing them:
 *
 *   - The engine is refused, not faked. A request against a checkpoint engine
 *     that cannot compute logits gets a 501 carrying the engine's own
 *     explanation. A completion endpoint that returns fluent nonsense is worse
 *     than one that says no, because it looks like it works.
 *
 *   - Prompt token counts are the checkpoint's own tokenizer's, when the file
 *     has one this build can read: prompt_tokens is then the exact BPE count of
 *     the rendered prompt, and /v1/models says so. With no usable tokenizer it
 *     falls back to characters/4 and the `qe` block flags that instead, so an
 *     estimated count is never presented as a measured one. completion_tokens
 *     is always exact (it is what the decoder emitted).
 *
 *   - Image content parts are decoded and counted, not understood. There is no
 *     vision encoder, so a base64 PNG is base64-decoded (validating it really
 *     is base64), sized, handed to the engine hook, and reported back. An
 *     http(s) image URL is refused rather than fetched.
 *
 * Generation is serialised by the engine's own mutex; five concurrent requests
 * queue rather than each issuing their own 12 GB/s of DRAM traffic. */
#define _GNU_SOURCE
#include "chat.h"
#include "http.h"
#include "json.h"
#include "qe.h"
#include "tui.h"

#include <errno.h>
#include <inttypes.h>
/* The counters are `_Atomic uint64_t`, and GCC's `__atomic_add_fetch` builtin takes that type while
 * clang's refuses it; the C11 spelling is the one both accept, and a server that
 * does not build with the compiler the Makefile names is a trap for the next
 * reader. */
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <pthread.h>

/* What a reply gets when the request asked for no length at all. OpenAI reads an
 * absent max_tokens as "until the model stops", and an agent turn that ends in a
 * tool request does not fit in a small number. It is a default and nothing more:
 * the ceiling is the sequence capacity, applied in asked_budget() below, because
 * that is the only number an answer cannot outgrow. A reasoning model thinks in
 * tens of thousands of tokens, so a server-side constant here would silently
 * amputate the thinking and report it as a finished answer. */
#define ANSWER_CAP 4096

/* Set by the signal handler, while the listener is still closing. Two readers,
 * asking the same question - is there any point generating this: the per-token
 * callbacks stop the decode, and the prefill progress hook abandons the prompt.
 * Without them a SIGTERM against a real checkpoint means waiting out an ingestion
 * measured in minutes, and a process that cannot get out of a request cannot be
 * stopped, which is a poor way to hand back 87 GiB and a port. A request that
 * finishes before the flag is set is answered normally; nothing else about the
 * reply changes. */
static volatile sig_atomic_t stopping;

/* What a request asked to be allowed to generate. A double off the wire can say
 * 1e300, and the engine takes an int, so a ceiling is applied here rather than
 * trusted to arrive intact: the sequence capacity the server opened with, which
 * is also what the engine will trim to and what the reply reports back as
 * `qe.measured.budget`. */
static long asked_budget(const qe_json *body, long n_ctx, int chat) {
    double v = qe_json_num_at(body, "max_tokens", 0.0);
    if (chat && v <= 0)
        v = qe_json_num_at(body, "max_completion_tokens", 0.0);
    if (!(v > 0))
        v = ANSWER_CAP;
    /* An engine that advertises no context allocated no sequence at all - the
     * selftest engine - so the default is the only ceiling it has. */
    double cap = n_ctx > 0 ? (double)n_ctx : (double)ANSWER_CAP;
    if (v > cap)
        v = cap;
    return (long)v;
}

typedef struct {
    qe_serve_opts o;
    qe_engine *eng;
    qe_engine_info info;
    int64_t started;
    _Atomic uint64_t requests;
    _Atomic uint64_t generated;
    _Atomic uint64_t seq;
} server;

static int64_t epoch_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (int64_t)t.tv_sec;
}

/* Monotonic milliseconds, for measuring a gap between two writes rather than the
 * wall-clock time they happened at. */
static long now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

/* --------------------------------------------------------------- utilities */

/* One line per reply. This is the instrument for the two questions a client cannot
 * answer from the outside: how long a request actually held the engine, and whether
 * it stopped because the reader went away. A queued request waits for the mutex, so
 * wall time is not the answer - these numbers are taken inside the engine's own
 * clock.
 *
 * `agreed` is the third one, and it is the difference between two failures that look
 * identical from the outside. `cached=0` says nothing was reused; `agreed=0` says the
 * prompt shared not one token with any position the engine could go back to, which is
 * a client that rewrote the front of the transcript, while `agreed=9820` on the same
 * conversation says the rewind points were there and the divergence was one edited
 * tool result a third of the way in. Without it a cache that is too small and a
 * client that keeps re-rendering are the same line.
 *
 * No model text appears here, ever. A log line that carried a vocabulary marker
 * would be a control token in a file that the next reader pastes somewhere.
 *
 * The three numbers at the end of the line - `content`, `thinking`, `calls` - are
 * what a turn was actually made of, and they are here because of the one failure a
 * client cannot report: an assistant turn that produced neither answer text nor a
 * tool request is discarded by charmbracelet's client without a word to the user, so
 * the engine has to be the one that noticed. A turn that thought for two thousand
 * bytes and never closed its thinking block is exactly that case, and without these
 * numbers it is indistinguishable from a turn that produced nothing at all.
 *
 * Where it goes depends on who owns the screen: with the panel up, a line written
 * beside it would be redrawn over or would redraw over, so the panel takes the line
 * and shows it in its own history. With no panel this is the same stderr line it has
 * always been. */
static void request_log(const char *kind, const char *id, long prompt_tokens,
                        const qe_gen_stats *gs, const char *finish, int streamed,
                        int client_left, long content_bytes, long thinking_bytes, int n_calls) {
    char line[512];
    snprintf(line, sizeof line,
             "qe: %s %s prompt=%ld cached=%ld agreed=%ld dropped=%ld answer=%ld passes=%ld %.2f s"
             " %.2f tok/s %.1f passes/s content=%ld thinking=%ld calls=%d finish=%s%s%s",
             kind, id, prompt_tokens, gs->cached, gs->rewind_lcp, gs->dropped, gs->tokens,
             gs->steps, gs->seconds,
             gs->seconds > 0 ? (double)gs->tokens / gs->seconds : 0.0,
             gs->seconds > 0 ? (double)gs->steps / gs->seconds : 0.0,
             content_bytes, thinking_bytes, n_calls, finish ? finish : "-",
             streamed ? " stream" : "", client_left ? " client-left" : "");
    if (qe_ui_on())
        qe_ui_log(line);
    else
        fprintf(stderr, "%s\n", line);
    /* The turn nobody saw. Cut short by the budget, or by a reader that stopped
     * listening, it is expected; a turn that finished on its own holding nothing a
     * client can show is the bug, and it is silent on the client's side. */
    if (content_bytes || n_calls || gs->truncated || client_left)
        return;
    snprintf(line, sizeof line,
             "qe: WARN %s %s finished with no answer and no tool call: %ld thinking byte(s), "
             "%ld token(s), finish=%s. A client that discards an empty assistant message "
             "shows this turn as nothing happening",
             kind, id, thinking_bytes, gs->tokens, finish ? finish : "-");
    if (qe_ui_on())
        qe_ui_log(line);
    else
        fprintf(stderr, "%s\n", line);
}

/* Everything one reply needs: the answer text and the reasoning text (only when
 * the client asked for one response rather than a stream), the tool requests,
 * and the reason generation ended.
 *
 * The stream path deliberately accumulates nothing it has already sent. A
 * 200-token answer is 200 SSE events on its way out and keeping a copy of it
 * here would only matter to a client that reads the same response twice. */
typedef struct {
    qe_str text;      /* the answer, for a non-streaming body */
    qe_str reasoning; /* the reasoning block, same reason */
    qe_str calls;     /* tool requests, as JSON array members */
    /* One per text stream, because a byte-level vocabulary ends a token wherever
     * the merge table says to, and a three-byte character split across two of them
     * is not two strings. See qe_utf8_feed. */
    qe_utf8_out u_text, u_reason, u_args;
    int n_calls;
    char call_name[256];
    char call_id[64];
    const char *finish; /* set when the assistant turn closed on its own */
    qe_turn *turn;      /* the stream parser, for the token callback */

    server *s;
    qe_http_conn *c;
    int streaming;
    const char *id;
    const char *model;
    int64_t created;
    int chunk_failed;
    int first;
    int chat; /* a chat reply has a delta, a completion has text */
    /* When the last byte of this reply went out, and how long the stream may stay
     * quiet before a comment says the engine is busy and not stuck. Only a stream
     * uses either; see qe_serve_opts.sse_keepalive_ms. */
    long written_ms;
    int ka_ms;
    /* Bytes of answer text this reply carried, counted on the stream as well as
     * off it: the stream path keeps no copy of what it already sent, and the
     * request line still has to say whether the turn held anything to show. */
    long content_bytes;
} sink;

/* What a reply held on to while it was being written. The stream path sends each
 * piece and keeps only the reasoning and the requests, because a finished
 * response has no more use for either, and every path out of a handler has to
 * say so. */
static void sink_free(sink *k) {
    qe_str_free(&k->text);
    qe_str_free(&k->reasoning);
    qe_str_free(&k->calls);
    qe_utf8_free(&k->u_text);
    qe_utf8_free(&k->u_reason);
    qe_utf8_free(&k->u_args);
}

/* One completion chunk: text only, no structure, because a completion endpoint
 * has no reasoning block and no tools to report. */
static int sink_emit(void *vp, const char *utf8, int n) {
    sink *k = vp;
    if (stopping)
        return 1;
    qe_str piece = { 0 };
    if (qe_utf8_feed(&k->u_text, &piece, utf8, (size_t)n)) {
        qe_str_free(&piece);
        k->chunk_failed = 1;
        return k->chunk_failed;
    }
    k->content_bytes += (long)piece.len;
    if (k->streaming) {
        qe_str d = { 0 };
        qe_str_puts(&d, "{\"id\":");
        qe_str_json_str(&d, k->id, strlen(k->id));
        qe_str_appendf(&d, ",\"object\":\"text_completion\",\"created\":%" PRId64, k->created);
        qe_str_puts(&d, ",\"model\":");
        qe_str_json_str(&d, k->model, strlen(k->model));
        qe_str_puts(&d, ",\"choices\":[{\"index\":0,\"text\":");
        qe_str_json_str(&d, piece.p ? piece.p : "", piece.len);
        qe_str_puts(&d, ",\"logprobs\":null,\"finish_reason\":null}]}");
        if (qe_http_sse_data(k->c, d.p ? d.p : "{}"))
            k->chunk_failed = 1;
        qe_str_free(&d);
        qe_str_free(&piece);
        k->written_ms = now_ms();
        return k->chunk_failed;
    }
    int rc = piece.len ? qe_str_append(&k->text, piece.p, piece.len) : 0;
    qe_str_free(&piece);
    return rc;
}

static void sink_chunk(sink *k, const char *delta_json, const char *finish) {
    if (!k->streaming)
        return;
    qe_str d = { 0 };
    qe_str_puts(&d, "{\"id\":");
    qe_str_json_str(&d, k->id, strlen(k->id));
    qe_str_appendf(&d, ",\"object\":\"%s\",\"created\":%" PRId64,
                   k->chat ? "chat.completion.chunk" : "text_completion", k->created);
    qe_str_puts(&d, ",\"model\":");
    qe_str_json_str(&d, k->model, strlen(k->model));
    if (k->chat) {
        qe_str_appendf(&d, ",\"choices\":[{\"index\":0,\"delta\":%s,\"logprobs\":null,"
                           "\"finish_reason\":",
                       delta_json);
    } else {
        qe_str_appendf(&d, ",\"choices\":[{\"index\":0,\"text\":\"\",\"logprobs\":null,"
                           "\"finish_reason\":");
    }
    if (finish)
        qe_str_json_str(&d, finish, strlen(finish));
    else
        qe_str_append(&d, "null", 4);
    qe_str_puts(&d, "}]}");
    if (qe_http_sse_data(k->c, d.p ? d.p : "{}"))
        k->chunk_failed = 1;
    qe_str_free(&d);
    k->written_ms = now_ms();
}

/* Nothing is written between the request being read and the first token, and on an
 * agent prompt that gap is 862 s: an 11,758-token prompt goes through the network
 * before a delta exists. A reader that went away inside that gap is, without this,
 * noticed only when the first delta is written - which is to say after the whole
 * turn was generated for nobody, at 74.9 ms a pass. So bytes go out on a timer, and
 * bytes that cannot be written stop the engine, which puts the ingestion half of the
 * dead-reader policy under the same rule the decode half has always been under.
 *
 * Those bytes are an SSE comment, which is what the format defines for "the
 * connection is alive" and what no client's message assembler can mistake for a
 * message. They used to be an empty delta, and that was the worse choice for a
 * reason that is obvious once said out loud: a chunk carrying `"delta":{}` in front
 * of a reader that finalizes a message on anything it cannot classify is one way of
 * manufacturing an empty assistant turn, and an empty assistant turn is exactly what
 * crush throws away without showing the user a thing. A comment cannot be discarded
 * as empty because it was never a part.
 *
 * What this is explicitly not for: keeping a slow client's idle deadline open.
 * Measured, not assumed - crush credits its deadline per parsed stream part, and
 * fantasy yields a part only for non-empty content, a tool call or a finish reason
 * (providers/openai/language_model.go), so an empty delta and an SSE comment are
 * both worth exactly nothing to it: with a 3 s deadline and an empty chunk every
 * 500 ms, the turn never completed, exactly as it never completed on silence. A
 * client with an idle deadline and an engine this slow has to be configured not to
 * have one; `options.request_timeout` is that setting, and MANUAL.md says so. */
static int sink_progress(void *vp, long done, long total) {
    sink *k = vp;
    (void)done;
    (void)total;
    if (stopping)
        return 1;
    if (!k->streaming)
        return 0; /* no head was sent, so there is nothing to put a comment on */
    if (now_ms() - k->written_ms < k->ka_ms)
        return 0;
    if (qe_http_sse_comment(k->c, "keepalive"))
        k->chunk_failed = 1;
    k->written_ms = now_ms();
    return k->chunk_failed;
}

/* Answer text and reasoning text are the same shape with a different key, and a
 * client reads one or the other into a different part of the screen. The key is
 * omitted rather than sent as null when the chunk is not of that kind: a null
 * reasoning value is "nothing here", while an empty string is "the reasoning
 * block is open and empty", and the two are not the same thing to a reader that
 * shows a thinking block as soon as the key appears. */
static void sink_text(sink *k, int reasoning, const char *utf8, int n) {
    if (n <= 0)
        return;
    /* Reasoning is accumulated even on a stream: usage has to say how many
     * tokens of the reply were reasoning, and bytes are not tokens. The answer is
     * not accumulated on a stream, because the client already has it. This is the
     * only place either buffer grows, and it is worth saying so out loud: for as
     * long as this function has existed a second append ran after the streaming
     * branch below and put every token of a non-streaming reply in twice.
     * `make probe` is what found it - the answer came back as "wordword wordword",
     * and the selftest surface never noticed because nothing there compared the
     * assembled text with the same answer streamed. */
    qe_str piece = { 0 };
    if (qe_utf8_feed(reasoning ? &k->u_reason : &k->u_text, &piece, utf8, (size_t)n)) {
        qe_str_free(&piece);
        k->chunk_failed = 1;
        return;
    }
    if (!piece.len)
        return; /* the whole token was the start of a character */
    if (reasoning) {
        qe_str_append(&k->reasoning, piece.p, piece.len);
    } else {
        k->content_bytes += (long)piece.len;
        if (!k->streaming)
            qe_str_append(&k->text, piece.p, piece.len);
    }
    if (k->streaming) {
        qe_str d = { 0 };
        qe_str_puts(&d, reasoning ? "{\"reasoning_content\":" : "{\"content\":");
        qe_str_json_str(&d, piece.p, piece.len);
        qe_str_puts(&d, "}");
        sink_chunk(k, d.p ? d.p : "{}", NULL);
        qe_str_free(&d);
    }
    qe_str_free(&piece);
}

/* Nothing here trims an unfinished character at the end of a reply any more,
 * because nothing here writes one: qe_utf8_feed holds back the bytes of a
 * sequence it has not seen the whole of, and an answer that ends inside a
 * character simply never emits those bytes. A trailing partial sequence is what
 * made `make probe` reject a whole body over 2 bytes of a 3-byte character -
 * strict parsers reject a JSON document whose string does not decode - and the
 * trim could only ever reach the end of the buffer. A byte cut in the *middle*
 * of a reply, which a byte-level vocabulary produces as a matter of course,
 * needed the same discipline applied to every chunk in both directions. */

/* A tool request the client can act on: the id it will be quoted back by, the
 * name, and an argument list that starts empty so a reader that shows progress
 * can display the call before the arguments have arrived. */
static void sink_call_open(sink *k, const qe_turn_ev *ev) {
    size_t nlen = (size_t)(ev->name_len > 0 ? ev->name_len : 0);
    if (nlen >= sizeof k->call_name)
        nlen = sizeof k->call_name - 1;
    memcpy(k->call_name, ev->name, nlen);
    k->call_name[nlen] = 0;
    snprintf(k->call_id, sizeof k->call_id, "%s", ev->call_id ? ev->call_id : "call-0");

    if (k->streaming) {
        qe_str d = { 0 };
        qe_str_puts(&d, "{\"tool_calls\":[{\"index\":");
        qe_str_appendf(&d, "%d,\"id\":", ev->index);
        qe_str_json_str(&d, k->call_id, strlen(k->call_id));
        qe_str_puts(&d, ",\"type\":\"function\",\"function\":{\"name\":");
        qe_str_json_str(&d, k->call_name, strlen(k->call_name));
        qe_str_puts(&d, ",\"arguments\":\"\"}}]}");
        sink_chunk(k, d.p ? d.p : "{}", NULL);
        qe_str_free(&d);
    }
}

/* The arguments arrive whole. On the wire "arguments" is a JSON object carried
 * inside a string, which is why it is escaped rather than nested. */
static void sink_call_args(sink *k, const qe_turn_ev *ev) {
    size_t alen = (size_t)(ev->nbytes > 0 ? ev->nbytes : 0);
    /* One call's arguments arrive whole, so the state is emptied here rather than
     * carried to the next: an argument list that stops inside a character has no
     * next token to complete it. */
    qe_str args = { 0 };
    if (qe_utf8_feed(&k->u_args, &args, ev->utf8 ? ev->utf8 : "", alen))
        args.err = 1;
    qe_utf8_finish(&k->u_args);
    if (k->streaming) {
        qe_str d = { 0 };
        qe_str_puts(&d, "{\"tool_calls\":[{\"index\":");
        qe_str_appendf(&d, "%d,\"function\":{\"arguments\":", ev->index);
        qe_str_json_str(&d, args.p ? args.p : "", args.len);
        qe_str_puts(&d, "}}]}");
        sink_chunk(k, d.p ? d.p : "{}", NULL);
        qe_str_free(&d);
    }
    if (k->n_calls)
        qe_str_puts(&k->calls, ",");
    qe_str_puts(&k->calls, "{\"id\":");
    qe_str_json_str(&k->calls, k->call_id, strlen(k->call_id));
    qe_str_puts(&k->calls, ",\"type\":\"function\",\"function\":{\"name\":");
    qe_str_json_str(&k->calls, k->call_name, strlen(k->call_name));
    qe_str_puts(&k->calls, ",\"arguments\":");
    qe_str_json_str(&k->calls, args.p ? args.p : "", args.len);
    qe_str_puts(&k->calls, "}}");
    qe_str_free(&args);
    k->n_calls++;
}

/* One event from the turn parser. Returning non-zero stops the decode, which is
 * how a client that hung up stops 5 GiB of DRAM traffic per step. */
static int chat_event(void *vp, const qe_turn_ev *ev) {
    sink *k = vp;
    switch (ev->kind) {
    case QE_TURN_CONTENT:   sink_text(k, 0, ev->utf8, ev->nbytes); break;
    case QE_TURN_REASONING: sink_text(k, 1, ev->utf8, ev->nbytes); break;
    case QE_TURN_CALL_OPEN: sink_call_open(k, ev); break;
    case QE_TURN_CALL_ARGS: sink_call_args(k, ev); break;
    case QE_TURN_CALL_CLOSE: break;
    case QE_TURN_STOP:      k->finish = ev->finish; break;
    }
    return k->chunk_failed;
}

/* Tokens are the parser's input, not the client's output: nothing reaches the
 * wire except through chat_event. */
static int chat_token(void *vp, const qe_token_out *t) {
    sink *k = vp;
    if (stopping)
        return 1;
    return qe_turn_token(k->turn, t->id, t->utf8, t->nbytes);
}

static char *new_id(server *s, const char *kind) {
    char *b = malloc(64);
    uint64_t n = atomic_fetch_add_explicit(&s->seq, 1, memory_order_relaxed);
    snprintf(b, 64, "%s-%08x%04" PRIx64, kind, (unsigned)(epoch_s() & 0xffffffffu), n & 0xffff);
    return b;
}

/* Rough token count, used only when the engine has no usable tokenizer; the
 * fact that it is an estimate is stated wherever it is reported. */
static long estimate_tokens(const char *s) {
    size_t n = s ? strlen(s) : 0;
    return (long)((n + 3) / 4);
}

/* usage.prompt_tokens is taken from the checkpoint's real BPE whenever the
 * engine loaded one, so the count a client sees is the count the model would
 * actually be fed. The estimate above is off by 1.75x on source code and 0.81x
 * on English prose (bench_tok measured both), which is more than a rounding
 * error. It stays as the fallback: a checkpoint whose tokenizer this build
 * cannot read is still a checkpoint that serves. */
static long count_prompt_tokens(qe_engine *eng, const char *s) {
    qe_tok *t = eng ? qe_engine_tok(eng) : NULL;
    if (t && s) {
        long n = qe_tok_count(t, s);
        if (n >= 0)
            return n;
    }
    return estimate_tokens(s);
}

/* Images are decoded and counted in chat.c, which owns the transcript they are
 * part of; the placeholder text they leave behind is what a text-only engine can
 * honestly say about a base64 PNG. */

/* -------------------------------------------------------------- engine meta */

static void qe_block(qe_str *s, server *st, const char *extra) {
    char tok_note[400];
    if (st->info.has_tokenizer)
        snprintf(tok_note, sizeof tok_note,
                 "prompt_tokens is exact (%s); completion_tokens is exact",
                 st->info.tokenizer);
    else
        snprintf(tok_note, sizeof tok_note,
                 "prompt_tokens is an estimate of characters/4; completion_tokens is exact"
                 " (tokenizer: %s)",
                 st->info.tokenizer[0] ? st->info.tokenizer : "none");
    qe_str_puts(s, "\"qe\":{");
    qe_str_puts(s, "\"engine\":");
    qe_str_json_str(s, st->info.is_checkpoint ? "checkpoint" : "selftest",
                    strlen(st->info.is_checkpoint ? "checkpoint" : "selftest"));
    qe_str_puts(s, ",\"arch\":");
    qe_str_json_str(s, st->info.arch, strlen(st->info.arch));
    qe_str_puts(s, ",\"quant\":");
    qe_str_json_str(s, st->info.quant, strlen(st->info.quant));
    if (st->info.label[0]) {
        qe_str_puts(s, ",\"name\":");
        qe_str_json_str(s, st->info.label, strlen(st->info.label));
    }
    if (st->info.size_label[0]) {
        qe_str_puts(s, ",\"size_label\":");
        qe_str_json_str(s, st->info.size_label, strlen(st->info.size_label));
    }
    /* What the checkpoint says the model is, even where this build cannot run it. */
    qe_str_appendf(s, ",\"layers\":%d", st->info.n_blocks);
    qe_str_appendf(s, ",\"width\":%d", st->info.n_embd);
    qe_str_appendf(s, ",\"experts\":%d", st->info.n_experts);
    qe_str_appendf(s, ",\"experts_active\":%d", st->info.n_experts_active);
    qe_str_appendf(s, ",\"vocab\":%d", st->info.n_vocab);
    qe_str_appendf(s, ",\"context_train\":%d", st->info.n_ctx_train);
    /* The window this server will actually take, which is the number a client
     * budgets a conversation against: `context_train` is what the file claims and
     * `context` is what was allocated. They differ whenever --ctx said otherwise,
     * and 0 means no cache was allocated at all (the selftest engine). */
    qe_str_appendf(s, ",\"context\":%d", st->info.n_ctx);
    qe_str_appendf(s, ",\"eos_token_id\":%d", st->info.eos_token_id);
    qe_str_appendf(s, ",\"chat_template\":%s", st->info.has_chat_template ? "true" : "false");
    qe_str_appendf(s, ",\"mtp_head\":%s", st->info.has_mtp_head ? "true" : "false");
    /* Which file that head is in. "true" alone makes a server with a head and a
     * server looking at the wrong file answer the same question identically. */
    if (st->info.mtp_path[0]) {
        qe_str_puts(s, ",\"mtp_file\":");
        qe_str_json_str(s, st->info.mtp_path, strlen(st->info.mtp_path));
        qe_str_puts(s, ",\"mtp_stage\":");
        qe_str_json_str(s, st->info.mtp_stage, strlen(st->info.mtp_stage));
        qe_str_appendf(s, ",\"mtp_tensors\":%lld", (long long)st->info.mtp_tensors);
        qe_str_appendf(s, ",\"mtp_weight_bytes\":%lld", (long long)st->info.mtp_bytes);
        qe_str_appendf(s, ",\"mtp_layers\":%d", st->info.mtp_layers);
        qe_str_appendf(s, ",\"mtp_head_blocks\":%d", st->info.mtp_head_blocks);
        /* Staged and walked are different facts, and only one of them is a
         * server that drafts. Say both, because the difference is the RAM. */
        qe_str_appendf(s, ",\"mtp_drafting\":%s",
                       st->info.mtp_drafting ? "true" : "false");
    }
    qe_str_appendf(s, ",\"can_generate\":%s", st->info.can_generate ? "true" : "false");
    qe_str_appendf(s, ",\"volumes\":%d", st->info.n_volumes);
    qe_str_appendf(s, ",\"tensors\":%lld", (long long)st->info.n_tensors);
    qe_str_appendf(s, ",\"weight_bytes\":%lld", (long long)st->info.bytes);
    qe_str_appendf(s, ",\"threads\":%d", st->info.n_threads);
    qe_str_appendf(s, ",\"spec_k\":%d", st->info.spec_k);
    /* Sequences this engine drives through one walk of the weights, and how many are in
     * it at the moment the status was read. A server configured for eight and serving an
     * idle afternoon is both things at once, and the difference is the whole point of
     * asking. */
    qe_str_appendf(s, ",\"batch\":%d", st->info.n_batch);
    qe_str_appendf(s, ",\"batch_active\":%d", qe_engine_batch_active(st->eng));
    qe_str_appendf(s, ",\"tokenizer\":%s", st->info.has_tokenizer ? "true" : "false");
    /* What the attention cache holds per token of context. It is the number that
     * decides how many sequences fit next to each other, so a client planning
     * concurrency reads it rather than asking. */
    qe_str_puts(s, ",\"kv_cache\":");
    qe_str_json_str(s, st->info.kv_cache, strlen(st->info.kv_cache));
    /* Where the weight bytes actually are. "mmap" on this host means a disk seek
     * per expert row, so a client choosing between engines has to be able to see
     * it; it is not a detail this build should keep to itself. */
    qe_str_appendf(s, ",\"weights_resident\":%s",
                   (strcmp(st->info.stage, "mmap") && strcmp(st->info.stage, "none"))
                       ? "true"
                       : "false");
    qe_str_puts(s, ",\"weight_stage\":");
    qe_str_json_str(s, st->info.stage, strlen(st->info.stage));
    qe_str_appendf(s, ",\"weights_pinned\":%s", st->info.stage_pinned ? "true" : "false");
    qe_str_appendf(s, ",\"weight_stage_gib\":%.2f", (double)st->info.stage_bytes / 1073741824.0);
    qe_str_appendf(s, ",\"weight_stage_reused_gib\":%.2f",
                   (double)st->info.stage_cached / 1073741824.0);
    /* What a fast second turn costs and how much of it there is. A client that
     * sees cached_tokens rise wants to know the ceiling it will saturate against,
     * and an operator debugging a cache that never hits needs to see that it holds
     * two boundaries rather than none. */
    {
        long slots = 0, held = 0;
        long long bytes = 0, saved = 0;
        qe_engine_turn_stats(st->eng, &slots, &held, &bytes, &saved);
        qe_str_appendf(s, ",\"turn_cache_slots\":%ld", slots);
        qe_str_appendf(s, ",\"turn_cache_held\":%ld", held);
        qe_str_appendf(s, ",\"turn_cache_bytes\":%lld", bytes);
        qe_str_appendf(s, ",\"turn_cache_saved\":%lld", saved);
        qe_str_appendf(s, ",\"turn_cache_budget_mib\":%d", st->info.turn_cache_mib);
        /* The difference between "the cache is broken" and "this conversation is
         * longer than the cache", which is otherwise a guess a client has to make
         * from a cached_tokens number that stopped rising. */
        qe_str_appendf(s, ",\"turn_cache_short\":%s",
                       (st->info.turn_cache_auto && slots > 0 &&
                        slots < st->info.turn_cache_points)
                           ? "true"
                           : "false");
    }
    qe_str_puts(s, ",\"tokenizer_note\":");
    qe_str_json_str(s, tok_note, strlen(tok_note));
    qe_str_puts(s, ",\"note\":");
    qe_str_json_str(s, st->info.why, strlen(st->info.why));
    if (extra && *extra) {
        qe_str_puts(s, ",");
        qe_str_append(s, extra, strlen(extra));
    }
    qe_str_append(s, "}", 1);
}

/* ---------------------------------------------------------------- handlers */

static int handle_models(qe_http_conn *c, const qe_http_req *req, server *st) {
    if (strcmp(req->method, "GET"))
        return qe_http_error(c, 405, "invalid_request_error", "GET only");
    char *id = new_id(st, "list");
    qe_str s = { 0 };
    qe_str_puts(&s, "{\"object\":\"list\",\"data\":[{\"object\":\"model\",\"id\":");
    qe_str_json_str(&s, st->info.id, strlen(st->info.id));
    qe_str_appendf(&s, ",\"created\":%" PRId64, st->started);
    qe_str_puts(&s, ",\"owned_by\":\"qe\",\"caption\":");
    const char *caption = st->info.label[0] ? st->info.label : st->info.arch;
    qe_str_json_str(&s, caption, strlen(caption));
    /* The same facts again in the shape llama-server's /v1/models uses, because
     * that is the shape clients look for when they want a context window without
     * being told one: crush reads data[].meta.n_ctx, falling back to n_ctx_train,
     * and a server that reports only its own namespace leaves it to guess (or to
     * a config file). Both spellings are the same two numbers. No n_params: this
     * server counts tensors and bytes, not parameters, and an invented one is
     * worse than an absent one. */
    qe_str_puts(&s, ",\"meta\":{\"n_vocab\":");
    qe_str_appendf(&s, "%d,", st->info.n_vocab);
    qe_str_appendf(&s, "\"n_ctx\":%d,", st->info.n_ctx);
    qe_str_appendf(&s, "\"n_ctx_train\":%d,", st->info.n_ctx_train);
    qe_str_appendf(&s, "\"n_embd\":%d,", st->info.n_embd);
    qe_str_appendf(&s, "\"n_layer\":%d,", st->info.n_blocks);
    qe_str_appendf(&s, "\"size\":%lld}", (long long)st->info.bytes);
    qe_str_append(&s, ",", 1);
    qe_block(&s, st, NULL);
    qe_str_puts(&s, "}],\"request_id\":");
    qe_str_json_str(&s, id, strlen(id));
    qe_str_append(&s, "}", 1);
    int rc = qe_http_json(c, 200, s.p ? s.p : "{}");
    qe_str_free(&s);
    free(id);
    return rc;
}

static int handle_health(qe_http_conn *c, const qe_http_req *req, server *st) {
    (void)req;
    qe_str s = { 0 };
    qe_str_puts(&s, "{\"status\":\"ok\"");
    qe_str_appendf(&s, ",\"uptime_s\":%" PRId64, (int64_t)epoch_s() - st->started);
    qe_str_appendf(&s, ",\"requests\":%" PRIu64, (uint64_t)st->requests);
    qe_str_appendf(&s, ",\"generated_tokens\":%" PRIu64, (uint64_t)st->generated);
    qe_str_append(&s, ",", 1);
    qe_block(&s, st, NULL);
    qe_str_append(&s, "}", 1);
    int rc = qe_http_json(c, 200, s.p ? s.p : "{}");
    qe_str_free(&s);
    return rc;
}

/* A checkpoint this engine cannot speak to still gets its messages flattened,
 * because the only engine in that position is the synthetic one, whose prompt is
 * read by a test rather than by a model. No control tokens go in: without the
 * vocabulary there is no honest way to know which ones this file should have
 * used, and guessing them puts ids in a prompt that no model was trained on. */
static int render_plain(const qe_json *messages, qe_chat *r, char *why, size_t whyn) {
    r->chat = 1;
    r->n_messages = (long)qe_json_len(messages);
    if (!r->n_messages) {
        snprintf(why, whyn, "messages must be a non-empty array");
        return -1;
    }
    for (const qe_json *m = messages->child; m; m = m->next) {
        qe_str_appendf(&r->prompt, "%s\n", qe_chat_role(m));
        const qe_json *content = qe_json_at(m, "content");
        if (content && content->kind == QE_JSON_STR) {
            qe_str_append(&r->prompt, content->str, content->str_len);
            qe_str_puts(&r->prompt, "\n");
            continue;
        }
        if (content && content->kind == QE_JSON_ARR) {
            for (const qe_json *part = content->child; part; part = part->next) {
                const char *text = qe_json_str_at(part, "text", NULL);
                const char *type = qe_json_str_at(part, "type", text ? "text" : "");
                if (text && !strcmp(type, "text")) {
                    qe_str_append(&r->prompt, text, strlen(text));
                    continue;
                }
                if (strcmp(type, "image_url") && strcmp(type, "image") &&
                    strcmp(type, "input_image"))
                    continue;
                const qe_json *iu = qe_json_at(part, "image_url");
                const char *url = iu ? qe_json_str_at(iu, "url", "") : NULL;
                if (iu && iu->kind == QE_JSON_STR)
                    url = iu->str;
                if (!url || !*url) {
                    snprintf(why, whyn, "image part has no url");
                    return -1;
                }
                if (r->n_images >= QE_CHAT_MAX_IMAGES) {
                    r->truncated_images = 1;
                    continue;
                }
                if (qe_chat_record_image(r, url, why, whyn))
                    return -1;
                qe_str_appendf(&r->prompt, "[image %d: %s, %zu bytes]\n", r->n_images,
                               r->image_mime[r->n_images - 1], r->images[r->n_images - 1].len);
            }
            qe_str_puts(&r->prompt, "\n");
            continue;
        }
        qe_str_puts(&r->prompt, "\n");
    }
    if (r->prompt.err) {
        snprintf(why, whyn, "out of memory rendering the transcript");
        return -1;
    }
    return 0;
}

/* usage, in the one shape both endpoints and both transports use. The two
 * detail objects are the ones a coding agent reads: input is charged as
 * prompt_tokens minus cached_tokens, and reasoning tokens are what a thinking
 * model charged for on the way to the answer. */
static void put_usage(qe_str *s, long prompt_tokens, long completion, long reasoning,
                      long cached) {
    qe_str_appendf(s,
                   "\"usage\":{\"prompt_tokens\":%ld,\"completion_tokens\":%ld,"
                   "\"total_tokens\":%ld,\"prompt_tokens_details\":{\"cached_tokens\":%ld},"
                   "\"completion_tokens_details\":{\"reasoning_tokens\":%ld}}",
                   prompt_tokens, completion, prompt_tokens + completion, cached, reasoning);
}

/* The sampling knobs, read and checked in one place so both endpoints agree and
 * a bad value is a 400 rather than a silently different distribution. Absent
 * means "whatever the engine was started with", which is what lets a client that
 * knows nothing about min-p still work; present means the client said something,
 * including when what it said is off. The off spellings are the ones the other
 * server on this host uses: top_k -1 (or 0) turns top-k off, top_p 1 turns the
 * nucleus off, min_p 0 keeps everything - and because 0 is also the spelling of
 * "absent" inside qe_gen_opts, an explicit 0 is forwarded as the negative the
 * engine reads as a deliberate off. Without that translation a server started
 * with --min-p 0.3 would give a client no way to ask for the whole distribution. */
static int sampling_from_body(const qe_json *body, int n_vocab, qe_gen_opts *go,
                              char *why, size_t whyn) {
    const qe_json *v;

    if (go->temp < 0.0f) {
        snprintf(why, whyn, "'temperature' must not be negative");
        return -1;
    }
    if ((v = qe_json_at(body, "top_k"))) {
        if (v->kind != QE_JSON_NUM) {
            snprintf(why, whyn, "'top_k' must be a number");
            return -1;
        }
        if (v->num != (double)(int)v->num) {
            snprintf(why, whyn, "'top_k' must be an integer");
            return -1;
        }
        go->top_k = (int)v->num == 0 ? -1 : (int)v->num;
        /* Below -1 is a typo, not a request for off, and the difference only
         * shows up as odd text two hundred tokens later. */
        if (go->top_k < -1) {
            snprintf(why, whyn, "'top_k' must be -1 (off) or a positive count");
            return -1;
        }
        if (n_vocab > 0 && go->top_k > n_vocab) {
            snprintf(why, whyn, "'top_k' is larger than the vocabulary");
            return -1;
        }
    }
    if ((v = qe_json_at(body, "top_p"))) {
        if (v->kind != QE_JSON_NUM) {
            snprintf(why, whyn, "'top_p' must be a number");
            return -1;
        }
        go->top_p = (float)v->num;
        if (!(go->top_p > 0.0f && go->top_p <= 1.0f)) {
            snprintf(why, whyn, "'top_p' must be in (0, 1]");
            return -1;
        }
    }
    if ((v = qe_json_at(body, "min_p"))) {
        if (v->kind != QE_JSON_NUM) {
            snprintf(why, whyn, "'min_p' must be a number");
            return -1;
        }
        float m = (float)v->num;
        if (m < 0.0f || m >= 1.0f) {
            snprintf(why, whyn, "'min_p' must be in [0, 1)");
            return -1;
        }
        go->min_p = m == 0.0f ? -1.0f : m;
    }
    return 0;
}

static int handle_chat(qe_http_conn *c, const qe_http_req *req, server *st) {
    char err[256] = { 0 };
    if (strcmp(req->method, "POST"))
        return qe_http_error(c, 405, "invalid_request_error", "POST only");
    if (!req->body || !req->body_len)
        return qe_http_error(c, 400, "invalid_request_error", "a JSON body is required");
    qe_json *body = qe_json_parse(req->body, req->body_len, err, sizeof err);
    if (!body)
        return qe_http_error(c, 400, "invalid_request_error", err[0] ? err : "body is not JSON");
    if (body->kind != QE_JSON_OBJ) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", "body must be a JSON object");
    }

    const char *model = qe_json_str_at(body, "model", NULL);
    if (model && strcmp(model, st->info.id)) {
        char msg[320];
        snprintf(msg, sizeof msg,
                 "the model '%s' does not exist on this server; this build serves '%s' "
                 "(see GET /v1/models)",
                 model, st->info.id);
        qe_json_free(body);
        return qe_http_error(c, 404, "model_not_found", msg);
    }
    const qe_json *msgs = qe_json_at(body, "messages");
    if (!msgs || msgs->kind != QE_JSON_ARR) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", "'messages' is required");
    }
    if (qe_json_num_at(body, "n", 1) != 1) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", "n > 1 is not supported");
    }

    /* The request's own decisions about reasoning, then the transcript. A
     * checkpoint that has this model's control tokens gets this model's template
     * down to the sentence that states the reasoning effort; anything else gets
     * the flattening above. */
    qe_chat_opts co;
    if (qe_chat_opts_from_body(body, &co, err, sizeof err)) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", err);
    }
    /* Not something a client may ask for: whether a turn that arrived with no
     * reasoning text is replayed as an empty block is one rule for the whole
     * server, because a transcript that mixes the two teaches the model both. */
    co.empty_thinking_blocks = st->o.empty_thinking_blocks;
    qe_chat r = { 0 };
    qe_tok *tk = qe_engine_tok(st->eng);
    int rc = tk && qe_chat_supported(tk) ? qe_chat_render(tk, body, &co, &r, err, sizeof err)
                                        : render_plain(msgs, &r, err, sizeof err);
    if (rc) {
        qe_chat_free(&r);
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", err);
    }

    int stream = qe_json_bool_at(body, "stream", 0);
    int include_usage = 0;
    const qe_json *so = qe_json_at(body, "stream_options");
    if (so)
        include_usage = qe_json_bool_at(so, "include_usage", 0);
    long mt = asked_budget(body, st->info.n_ctx, 1);
    qe_gen_opts go = {
        .max_tokens = (int)mt,
        .temp = (float)qe_json_num_at(body, "temperature", 0.0),
        .seed = (unsigned)qe_json_num_at(body, "seed", 0),
        .spec_k = (int)qe_json_num_at(body, "qe_spec_k", st->info.spec_k),
        .eos = -1,
    };
    if (go.spec_k < 0)
        go.spec_k = 0;
    if (sampling_from_body(body, st->info.n_vocab, &go, err, sizeof err)) {
        qe_chat_free(&r);
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", err);
    }

    qe_prompt p = { .text = r.prompt.p ? r.prompt.p : "", .images = r.images,
                    .n_images = r.n_images, .is_chat = 1 };

    char *id = new_id(st, "chatcmpl");
    qe_ui_request("chat", id);
    sink k = { .s = st, .c = c, .streaming = stream, .id = id, .model = st->info.id,
               .created = epoch_s(), .chat = 1 };
    atomic_fetch_add_explicit(&st->requests, 1, memory_order_relaxed);

    if (!st->info.can_generate) {
        free(id);
        qe_chat_free(&r);
        qe_json_free(body);
        return qe_http_error(c, 501, "engine_unavailable", st->info.why);
    }
    /* The parser starts inside a reasoning block when the prompt actually left
     * the model there. Thinking on is not the test: a transcript rendered without
     * this model's control tokens opens no block at all, and reading the whole
     * reply as reasoning is what makes an engine look like it never answers. */
    k.turn = qe_turn_new(chat_event, &k, r.opens_reasoning);
    if (!k.turn) {
        free(id);
        qe_chat_free(&r);
        qe_json_free(body);
        return qe_http_error(c, 500, "server_error", "out of memory");
    }
    /* Call ids are built from this request's id: a client keeps its tool-call state
     * under the id for the length of the conversation, so an id repeated in a later
     * turn is a call that never appears and a result filed against the wrong call. */
    qe_turn_set_id_prefix(k.turn, id);

    if (stream) {
        if (qe_http_sse_head(c)) {
            qe_turn_free(k.turn);
            free(id);
            qe_chat_free(&r);
            qe_json_free(body);
            return -1;
        }
        sink_chunk(&k, "{\"role\":\"assistant\",\"content\":\"\"}", NULL);
    }
    k.first = 0;
    if (stream && st->o.sse_keepalive_ms > 0) {
        k.ka_ms = st->o.sse_keepalive_ms;
        k.written_ms = now_ms();
        /* The keepalive rides this request's ingestion hook and not an engine-wide
         * one: on a batched engine several prompts are being fed at once, and a comment
         * about one connection's prompt arriving on another's stream is a corrupt
         * stream. `go` is this request's copy of the options. */
        go.progress = sink_progress;
        go.progress_user = &k;
    }

    qe_gen_stats gs = { 0 };
    long n = qe_engine_generate_tokens(st->eng, &p, &go, chat_token, &k, &gs);
    go.progress = NULL;
    go.progress_user = NULL;
    qe_turn_end(k.turn);
    long prompt_tokens = count_prompt_tokens(st->eng, p.text);
    long reasoning_tokens = k.reasoning.len ? count_prompt_tokens(st->eng, k.reasoning.p) : 0;
    /* Three ways a reply ends, and a client has to be able to tell them apart:
     * the model closed its own turn, it ran out of the budget it was given, and
     * it stopped because it asked for something. A missing finish_reason is a
     * truncated stream to a careful client, not a successful one. */
    const char *finish = k.finish ? k.finish
                   : gs.truncated ? "length"
                                    : (k.n_calls ? "tool_calls" : "stop");
    if (n < 0 && !k.chunk_failed) {
        qe_http_error(c, 500, "server_error", "generation failed");
        sink_free(&k);
        qe_turn_free(k.turn);
        free(id);
        qe_chat_free(&r);
        qe_json_free(body);
        return -1;
    }
    atomic_fetch_add_explicit(&st->generated, (uint64_t)(n > 0 ? n : 0), memory_order_relaxed);

    if (stream) {
        sink_chunk(&k, "{}", finish);
        if (include_usage) {
            qe_str u = { 0 };
            qe_str_puts(&u, "{\"id\":");
            qe_str_json_str(&u, id, strlen(id));
            qe_str_appendf(&u, ",\"object\":\"chat.completion.chunk\",\"created\":%" PRId64,
                           k.created);
            qe_str_puts(&u, ",\"model\":");
            qe_str_json_str(&u, st->info.id, strlen(st->info.id));
            qe_str_puts(&u, ",\"choices\":[],");
            /* cached_tokens is what the engine did not have to compute because the
             * sequence already ended where this prompt begins. A client charging
             * prompt_tokens minus cached_tokens is the reason it is reported even
             * when it is zero. */
            put_usage(&u, prompt_tokens, gs.tokens, reasoning_tokens, gs.cached);
            qe_str_append(&u, "}", 1);
            qe_http_sse_data(c, u.p ? u.p : "{}");
            qe_str_free(&u);
        }
        qe_http_sse_done(c);
        request_log("chat", id, prompt_tokens, &gs, finish, 1, k.chunk_failed, k.content_bytes,
                    (long)k.reasoning.len, k.n_calls);
        sink_free(&k);
        qe_turn_free(k.turn);
        free(id);
        qe_chat_free(&r);
        qe_json_free(body);
        return 0;
    }

    char extra[640];
    snprintf(extra, sizeof extra,
             "\"measured\":{\"tokens\":%ld,\"target_passes\":%ld,\"drafted\":%ld,"
             "\"accepted\":%ld,\"tok_per_step\":%.3f,\"seconds\":%.6f,"
             "\"tok_per_s\":%.2f,\"passes_per_s\":%.1f,\"spec_k\":%d,\"messages\":%ld,"
             "\"images\":%d,\"image_bytes\":%zu,\"images_truncated\":%d,"
             "\"reasoning_bytes\":%zu,\"tool_calls\":%d,\"thinking\":%d,\"effort\":\"%s\","
             /* The budget the reply actually ran under: what was asked for, or
              * what the window left after the prompt. A client that sent no
              * max_tokens, or sent one larger than the sequence, cannot otherwise
              * tell a short answer from an ended one, and finish_reason "stop"
              * does not distinguish them either. */
             "\"budget\":%ld,"
             /* Prompt tokens thrown away because the prompt plus the answer did not
              * fit the window. Non-zero means the reply is answering the tail of a
              * transcript and never saw its beginning, which is a fact about the
              * answer and not a detail of the allocator. */
             "\"schemas\":%d,\"cached_tokens\":%ld,\"dropped\":%ld,"
             /* Whether those cached tokens were carried over from the previous
              * request or put back from a rewind point. The distinction is the
              * whole point of the turn cache: a client that re-renders the
              * assistant turn gets a hit here and a full ingestion if it is 0. */
             "\"rewound\":%s,\"rewind_lcp\":%ld,\"snapshot_ms\":%.1f}",
             gs.tokens, gs.steps, gs.drafted, gs.accepted, gs.tok_per_step, gs.seconds,
             gs.seconds > 0 ? (double)gs.tokens / gs.seconds : 0.0,
             gs.seconds > 0 ? (double)gs.steps / gs.seconds : 0.0,
             go.spec_k, r.n_messages,
             r.n_images, r.image_bytes, r.truncated_images, k.reasoning.len, k.n_calls,
             co.enable_thinking, co.effort_low ? "low" : (co.effort_xhigh ? "xhigh" : "medium"),
             gs.budget, r.tools_rendered, gs.cached, gs.dropped,
             gs.rewound ? "true" : "false", gs.rewind_lcp, (double)gs.snapshot_us / 1000.0);
    qe_str s = { 0 };
    qe_str_puts(&s, "{\"id\":");
    qe_str_json_str(&s, id, strlen(id));
    qe_str_appendf(&s, ",\"object\":\"chat.completion\",\"created\":%" PRId64, k.created);
    qe_str_puts(&s, ",\"model\":");
    qe_str_json_str(&s, st->info.id, strlen(st->info.id));
    /* Assembled once: neither buffer can hold an unfinished sequence, because the
     * sink took that possibility away token by token. */
    qe_utf8_finish(&k.u_text);
    qe_utf8_finish(&k.u_reason);
    qe_str_puts(&s, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    qe_str_json_str(&s, k.text.p ? k.text.p : "", k.text.len);
    if (k.reasoning.len) {
        /* A client replays this as the message's reasoning on the next turn,
         * which is how the model's own reasoning survives across turns. */
        qe_str_puts(&s, ",\"reasoning_content\":");
        qe_str_json_str(&s, k.reasoning.p, k.reasoning.len);
    }
    if (k.n_calls) {
        qe_str_puts(&s, ",\"tool_calls\":[");
        qe_str_append(&s, k.calls.p ? k.calls.p : "", k.calls.len);
        qe_str_append(&s, "]", 1);
    }
    qe_str_puts(&s, "},\"logprobs\":null,\"finish_reason\":");
    qe_str_json_str(&s, finish, strlen(finish));
    qe_str_append(&s, "}]}", 3);
    /* insert usage + qe before the closing brace */
    s.len -= 1;
    qe_str_append(&s, ",", 1);
    put_usage(&s, prompt_tokens, gs.tokens, reasoning_tokens, gs.cached);
    qe_str_append(&s, ",", 1);
    qe_block(&s, st, extra);
    qe_str_append(&s, "}", 1);
    int rc2 = qe_http_json(c, 200, s.p ? s.p : "{}");
    request_log("chat", id, prompt_tokens, &gs, finish, 0, k.chunk_failed, k.content_bytes,
                (long)k.reasoning.len, k.n_calls);
    qe_str_free(&s);
    sink_free(&k);
    qe_turn_free(k.turn);
    free(id);
    qe_chat_free(&r);
    qe_json_free(body);
    return rc2;
}

static int handle_completions(qe_http_conn *c, const qe_http_req *req, server *st) {
    char err[256] = { 0 };
    if (strcmp(req->method, "POST"))
        return qe_http_error(c, 405, "invalid_request_error", "POST only");
    if (!req->body || !req->body_len)
        return qe_http_error(c, 400, "invalid_request_error", "a JSON body is required");
    qe_json *body = qe_json_parse(req->body, req->body_len, err, sizeof err);
    if (!body)
        return qe_http_error(c, 400, "invalid_request_error", err[0] ? err : "body is not JSON");
    const char *model = qe_json_str_at(body, "model", NULL);
    if (model && strcmp(model, st->info.id)) {
        qe_json_free(body);
        return qe_http_error(c, 404, "model_not_found", "unknown model; see GET /v1/models");
    }
    /* An array prompt is a batch, so this runs the engine once per element
     * rather than concatenating the prompts into a sentence nobody wrote, and
     * returns one choice per element. The cap exists because generation is
     * serialised: an unbounded array would let one client hold a worker for as
     * long as it liked. */
#define QE_MAX_PROMPTS 16
    const qe_json *pr = qe_json_at(body, "prompt");
    if (!pr) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", "'prompt' is required");
    }
    const qe_json *first = NULL;
    int nprompt = 0;
    if (pr->kind == QE_JSON_STR) {
        first = pr;
        nprompt = 1;
    } else if (pr->kind == QE_JSON_ARR) {
        for (const qe_json *k = pr->child; k; k = k->next) {
            if (k->kind != QE_JSON_STR) {
                qe_json_free(body);
                return qe_http_error(c, 400, "invalid_request_error",
                                     "'prompt' array must contain only strings");
            }
            if (++nprompt > QE_MAX_PROMPTS) {
                qe_json_free(body);
                return qe_http_error(c, 400, "invalid_request_error",
                                     "'prompt' takes at most 16 entries per request");
            }
        }
        first = pr->child;
    } else {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error",
                             "'prompt' must be a string or an array of strings");
    }
    if (!nprompt) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", "'prompt' must not be empty");
    }

    int stream = qe_json_bool_at(body, "stream", 0);
    if (stream && nprompt > 1) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error",
                             "stream: true takes one prompt; post the batch non-stream");
    }
    long mt = asked_budget(body, st->info.n_ctx, 0);
    qe_gen_opts go = {
        .max_tokens = (int)mt,
        .temp = (float)qe_json_num_at(body, "temperature", 0.0),
        .seed = (unsigned)qe_json_num_at(body, "seed", 0),
        .spec_k = (int)qe_json_num_at(body, "qe_spec_k", st->info.spec_k),
        .eos = -1,
    };
    if (sampling_from_body(body, st->info.n_vocab, &go, err, sizeof err)) {
        qe_json_free(body);
        return qe_http_error(c, 400, "invalid_request_error", err);
    }
    char *id = new_id(st, "cmpl");
    qe_ui_request("completion", id);
    sink k = { .s = st, .c = c, .streaming = stream, .id = id, .model = st->info.id,
               .created = epoch_s(), .chat = 0 };
    atomic_fetch_add_explicit(&st->requests, 1, memory_order_relaxed);

    if (!st->info.can_generate) {
        free(id);
        qe_json_free(body);
        return qe_http_error(c, 501, "engine_unavailable", st->info.why);
    }
    if (stream && qe_http_sse_head(c)) {
        free(id);
        qe_json_free(body);
        return -1;
    }

    qe_str s = { 0 };
    if (!stream) {
        qe_str_puts(&s, "{\"id\":");
        qe_str_json_str(&s, id, strlen(id));
        qe_str_appendf(&s, ",\"object\":\"text_completion\",\"created\":%" PRId64, k.created);
        qe_str_puts(&s, ",\"model\":");
        qe_str_json_str(&s, st->info.id, strlen(st->info.id));
        qe_str_puts(&s, ",\"choices\":[");
    }

    long sum_prompt = 0, sum_tokens = 0;
    qe_gen_stats agg = { 0 };
    const char *last_finish = "stop";
    const qe_json *el = first;
    if (stream && st->o.sse_keepalive_ms > 0) {
        k.ka_ms = st->o.sse_keepalive_ms;
        k.written_ms = now_ms();
        go.progress = sink_progress;
        go.progress_user = &k;
    }
    for (int i = 0; i < nprompt; i++, el = el->next) {
        qe_str prompt = { 0 };
        qe_str_append(&prompt, el->str, el->str_len);
        qe_prompt p = { .text = prompt.p ? prompt.p : "", .images = NULL, .n_images = 0,
                        .is_chat = 0 };
        k.first = 0;
        k.chunk_failed = 0;
        qe_gen_stats gs = { 0 };
        long n = qe_engine_generate(st->eng, &p, &go, sink_emit, &k, &gs);
        last_finish = gs.truncated ? "length" : "stop";
        if (n < 0 && !k.chunk_failed) {
            if (!stream)
                qe_http_error(c, 500, "server_error", "generation failed");
            free(id);
            qe_str_free(&prompt);
            sink_free(&k);
            qe_str_free(&s);
            qe_json_free(body);
            return -1;
        }
        atomic_fetch_add_explicit(&st->generated, (uint64_t)(n > 0 ? n : 0), memory_order_relaxed);
        sum_prompt += count_prompt_tokens(st->eng, p.text);
        sum_tokens += gs.tokens;
        agg.tokens += gs.tokens;
        agg.steps += gs.steps;
        agg.drafted += gs.drafted;
        agg.accepted += gs.accepted;
        agg.seconds += gs.seconds;
        agg.dropped += gs.dropped;
        agg.cached += gs.cached;
        agg.rewound |= gs.rewound;
        if (gs.rewind_lcp > agg.rewind_lcp)
            agg.rewind_lcp = gs.rewind_lcp;
        agg.snapshot_us += gs.snapshot_us;
        if (!stream) {
            if (i)
                qe_str_append(&s, ",", 1);
            qe_str_appendf(&s, "{\"index\":%d,\"text\":", i);
            qe_utf8_finish(&k.u_text);
            qe_str_json_str(&s, k.text.p ? k.text.p : "", k.text.len);
            qe_str_puts(&s, ",\"logprobs\":null,\"finish_reason\":");
            qe_str_json_str(&s, last_finish, strlen(last_finish));
            qe_str_append(&s, "}", 1);
        }
        qe_str_free(&k.text);
        qe_str_free(&prompt);
    }
    go.progress = NULL;
    go.progress_user = NULL;

    if (stream) {
        sink_chunk(&k, "{}", last_finish);
        qe_http_sse_done(c);
        request_log("completion", id, sum_prompt, &agg, last_finish, 1, k.chunk_failed,
                    k.content_bytes, 0, k.n_calls);
        free(id);
        sink_free(&k);
        qe_str_free(&s);
        qe_json_free(body);
        return 0;
    }

    qe_str_append(&s, "]", 1);
    qe_str_appendf(&s, ",\"usage\":{\"prompt_tokens\":%ld,\"completion_tokens\":%ld,"
                       "\"total_tokens\":%ld},",
                   sum_prompt, sum_tokens, sum_prompt + sum_tokens);
    char extra[512];
    snprintf(extra, sizeof extra,
             "\"measured\":{\"tokens\":%ld,\"target_passes\":%ld,\"drafted\":%ld,"
             "\"accepted\":%ld,\"tok_per_step\":%.3f,"
             "\"tok_per_s\":%.2f,\"passes_per_s\":%.1f,\"spec_k\":%d,\"prompts\":%d,"
             "\"dropped\":%ld,\"cached_tokens\":%ld,\"rewound\":%s,\"rewind_lcp\":%ld,"
             "\"snapshot_ms\":%.1f}",
             agg.tokens, agg.steps, agg.drafted, agg.accepted,
             agg.steps ? (double)agg.tokens / (double)agg.steps : 0.0,
             agg.seconds > 0 ? (double)agg.tokens / agg.seconds : 0.0,
             agg.seconds > 0 ? (double)agg.steps / agg.seconds : 0.0,
             go.spec_k, nprompt, agg.dropped, agg.cached,
             agg.rewound ? "true" : "false", agg.rewind_lcp,
             (double)agg.snapshot_us / 1000.0);
    qe_block(&s, st, extra);
    qe_str_append(&s, "}", 1);
    int rc = qe_http_json(c, 200, s.p ? s.p : "{}");
    request_log("completion", id, sum_prompt, &agg, last_finish, 0, k.chunk_failed, k.content_bytes,
                0, k.n_calls);
    qe_str_free(&s);
    sink_free(&k);
    free(id);
    qe_json_free(body);
    return rc;
}

static int dispatch(qe_http_conn *connp, const qe_http_req *req, void *vp) {
    server *st = vp;
    const char *path = req->path;
    if (!strcmp(path, "/health") || !strcmp(path, "/healthz"))
        return handle_health(connp, req, st);
    if (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))
        return handle_models(connp, req, st);
    if (!strcmp(path, "/v1/chat/completions"))
        return handle_chat(connp, req, st);
    if (!strcmp(path, "/v1/completions"))
        return handle_completions(connp, req, st);
    if (!strcmp(path, "/v1") || !strcmp(path, "/v1/"))
        return qe_http_error(connp, 404, "invalid_request_error",
                             "endpoints: GET /v1/models, POST /v1/chat/completions, "
                             "POST /v1/completions, GET /health");
    char msg[256];
    snprintf(msg, sizeof msg, "no route for %s %s; try GET /v1/models", req->method, path);
    return qe_http_error(connp, 404, "invalid_request_error", msg);
}

/* -------------------------------------------------------------------- main */

static void stop_handler(int sig) {
    (void)sig;
    stopping = 1;
    qe_http_stop = 1;
}

void qe_serve_opts_default(qe_serve_opts *o) {
    if (!o->port)
        o->port = 8080;
    if (!o->max_conns)
        o->max_conns = 32;
    /* Zero here means "this opts struct was never filled", which is why it takes the
     * drafter's measured length. A caller that means off says so after this runs:
     * --spec-k 0 lands on the parsed opts and is never rewritten. */
    if (!o->engine.spec_k)
        o->engine.spec_k = QE_SPEC_K_AUTO;
    /* Under two minutes because two minutes is crush's default stream idle
     * deadline, and the point of the comment is to be earlier than that. */
    if (!o->sse_keepalive_ms)
        o->sse_keepalive_ms = 15000;
    /* Short on purpose. A keep-alive client that is merely thinking between two
     * requests counts as an unfinished request, so this is also how long a
     * shutdown waits for a client that is doing nothing wrong; it is not the wait
     * for a generation, because `stopping` ends that at the next pass. */
    if (!o->shutdown_grace_secs)
        o->shutdown_grace_secs = 5;
    /* n_ctx is deliberately not defaulted here. 0 means "whatever the checkpoint
     * says it was trained for", and a server that invented 4096 of its own would
     * be shrinking the model it was handed - which is exactly what this line used
     * to do to a 262144-token model. */
    if (!o->engine.id)
        o->engine.id = o->engine.model_path ? "qe" : "qe-selftest";
}

int qe_serve(const qe_serve_opts *opts) {
    qe_serve_opts o = *opts;
    qe_serve_opts_default(&o);
    char err[512] = { 0 };
    qe_engine *eng = qe_engine_open(&o.engine, err, sizeof err);
    if (!eng) {
        fprintf(stderr, "engine: %s\n", err);
        return 1;
    }
    static server st;
    st.o = o;
    st.eng = eng;
    st.started = epoch_s();
    qe_engine_describe(eng, &st.info);

    if (err[0])
        fprintf(stderr, "engine: %s\n", err);
    fprintf(stderr, "qe: engine=%s id=%s arch=%s can_generate=%d\n",
            st.info.is_checkpoint ? "checkpoint" : "selftest", st.info.id, st.info.arch,
            st.info.can_generate);
    /* The window this process will actually take, printed instead of left for a
     * client to curl. The default is the checkpoint's own trained length, which is
     * not visible from the command line at all when no --ctx was given, and a
     * server that opened less than the model was trained for is a server that
     * throws away the front of long transcripts. */
    if (st.info.n_ctx > 0)
        fprintf(stderr, "qe: context: %d tokens of %d trained%s\n", st.info.n_ctx,
                st.info.n_ctx_train,
                st.info.n_ctx != st.info.n_ctx_train ? " (--ctx)" : "");
    else
        fprintf(stderr, "qe: context: none allocated (no sequence cache)\n");
    fprintf(stderr, "qe: stream keepalive: an SSE comment every %d ms of silence%s\n",
            o.sse_keepalive_ms > 0 ? o.sse_keepalive_ms : 0,
            o.sse_keepalive_ms > 0 ? "" : " (off)");
    /* Said out loud because it changes what the model is shown on every turn, and a
     * client measuring a shorter answer needs to know which prompt shape it got. */
    fprintf(stderr, "qe: thinking replay: %s\n",
            o.empty_thinking_blocks
                ? "an empty block for an assistant turn that carried none"
                : "no block for an assistant turn that carried none (--empty-thinking-blocks)");
    /* The ingestion speed switch, said out loud because the two settings ingest the
     * same prompt at different speeds and, for Q8_0 projections, do the same sum in a
     * different number of chains: a server that answers slightly differently from the
     * one a measurement was made against is only explainable if this line is in the
     * log. */
    if (st.info.can_generate) {
        size_t pcb = qe_engine_proj_chunk(eng);
        if (pcb)
            fprintf(stderr,
                    "qe: projection walk: one sweep per chunk for projections of %zu MiB "
                    "and above (--proj-chunk)\n",
                    pcb >> 20);
        else
            fprintf(stderr, "qe: projection walk: off, every projection swept once per "
                            "token (--proj-chunk)\n");
    }
    /* Whether a second turn has anything to go back to. Said out loud because the
     * alternative is a client measuring a full ingestion and having to guess
     * whether the cache was off, full, or never filled. */
    {
        long slots = 0, held = 0;
        long long bytes = 0, saved = 0;
        qe_engine_turn_stats(eng, &slots, &held, &bytes, &saved);
        if (slots > 0) {
            /* Points and their price rather than a bare budget: 112.6 MiB a point is
             * the fact that decides whether raising --turn-cache is cheap or rude,
             * and "chosen from the memory free at open" is the difference between a
             * number the operator set and one the engine inferred. */
            fprintf(stderr, "qe: turn cache: %ld rewind point%s at %.1f MiB each, %d MiB%s\n",
                    slots, slots == 1 ? "" : "s",
                    (double)st.info.turn_slot_bytes / 1048576.0,
                    st.info.turn_cache_mib,
                    st.info.turn_cache_auto
                        ? " chosen from the memory free at open (--turn-cache to set it)"
                        : "");
            /* A cache that is on and too small looks, to the client, exactly like a
             * cache that is broken: the conversation outgrows it and every turn
             * after that reads itself again. Said here, once, at the width the
             * operator still has room to change. */
            if (st.info.turn_cache_auto && slots < st.info.turn_cache_points)
                fprintf(stderr,
                        "qe: turn cache: short of the %d points it asks for, so a "
                        "conversation longer than %ld turns re-reads its own history\n",
                        st.info.turn_cache_points, slots / 2);
        } else if (st.info.can_generate)
            fprintf(stderr, "qe: turn cache: %s\n",
                    st.info.turn_cache_err[0]
                        ? st.info.turn_cache_err
                        : "off (every turn reads its whole prompt)");
        else
            fprintf(stderr, "qe: turn cache: none (this engine holds no sequence)\n");
    }
    if (st.info.is_checkpoint) {
        /* Belt and braces: qe_engine_open refuses a checkpoint it could not make
         * resident, and a listener is not the place to discover that a future
         * caller found a way past it. */
        if (!strcmp(st.info.stage, "mmap")) {
            fprintf(stderr, "qe: refusing to serve: weights are demand-paged, not resident\n");
            qe_engine_close(eng);
            return 1;
        }
        fprintf(stderr, "qe: weights: stage %s, %.1f GiB (%.1f GiB reused), %s\n",
                st.info.stage, (double)st.info.stage_bytes / 1073741824.0,
                (double)st.info.stage_cached / 1073741824.0,
                st.info.stage_pinned == 2 ? "hugetlb (not pageable)"
                                          : (st.info.stage_pinned ? "mlock pinned" : "reclaimable"));
        /* Which rows each pool thread owns. Every latency this server reports is
         * a latency of that division, so it is printed next to the stage line
         * rather than left for the bench to say. */
        {
            char sbuf[128];
            fprintf(stderr, "qe: slicing: %s\n", qe_slice_summary(sbuf, sizeof sbuf));
        }
    }
    /* A draft head is only worth what the operator paid for it - a GiB of RAM and
     * a flag on a command line - so the log says which file was taken, what it
     * holds, and that it is resident rather than implied. */
    if (st.info.mtp_path[0]) {
        char nextn[64];
        if (st.info.mtp_layers)
            snprintf(nextn, sizeof nextn, "nextn %d layer%s", st.info.mtp_layers,
                     st.info.mtp_layers == 1 ? "" : "s");
        else
            snprintf(nextn, sizeof nextn, "%s", "no nextn layer count in the file");
        fprintf(stderr, "qe: draft head: %s\n", st.info.mtp_path);
        fprintf(stderr, "qe: draft head: %lld tensors, %.2f GiB, stage %s, %s\n",
                (long long)st.info.mtp_tensors, (double)st.info.mtp_bytes / 1073741824.0,
                st.info.mtp_stage, nextn);
        /* The part an operator would otherwise find out from an accept rate: whether
         * the nextn block in that file is what is drafting, or whether it is resident
         * bytes the pass declined to open. `make http` is the second case on purpose -
         * it points --mtp at a 160 KiB fixture whose tensors are not a head - so the
         * answer is read from the engine rather than assumed from the flag. */
        if (st.info.mtp_drafting)
            fprintf(stderr,
                    "qe: draft head: walked - greedy requests draft %d token%s a window off"
                    " it and verify the window in one pass (--spec-k N changes the length,"
                    " 0 turns speculation off)\n",
                    st.info.spec_k, st.info.spec_k == 1 ? "" : "s");
        else
            fprintf(stderr, "qe: draft head: staged, not walked - drafts come from the n-gram"
                            " drafter, so the nextn layer in this file is not what is drafting\n");
    }
    if (!st.info.can_generate)
        fprintf(stderr, "qe: %s\n", st.info.why);

    qe_http_opts ho = { .port = o.port,
                        .max_conns = o.max_conns,
                        .max_body = 64u << 20,
                        .idle_secs = 600 };
    /* Ctrl-C or SIGTERM should end the process rather than leave it needing
     * SIGKILL: a shutdown path is what lets the exit-time leak check run.
     * Deliberately no SA_RESTART, so the blocked accept is interrupted. */
    struct sigaction sa = {.sa_handler = stop_handler};
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    int lfd = qe_http_listen(&ho);
    if (lfd < 0) {
        fprintf(stderr, "listen :%d: %s\n", o.port, strerror(errno));
        qe_engine_close(eng);
        return 1;
    }
    fprintf(stderr, "qe: listening on :%d\n", o.port);
    /* The banner above scrolls away as soon as a panel redraws, so the panel gets
     * the sentence it repeats instead: which model, how big a window, where the
     * bytes are. A status line that cannot say what it is watching is guesswork. */
    {
        char head[256];
        snprintf(head, sizeof head, "%s %s%s%s ctx=%d stage=%s :%d", st.info.id, st.info.arch,
                 st.info.label[0] ? " " : "", st.info.label, st.info.n_ctx, st.info.stage,
                 o.port);
        qe_ui_start(o.ui, head);
        /* The second row of the panel, and deliberately the file's own name rather
         * than its path: a long path on a status row is a path that gets cut in half
         * in the middle of a directory name, which is worse than not showing it. The
         * banner above and /v1/models both carry it whole. */
        if (st.info.mtp_path[0]) {
            const char *bn = strrchr(st.info.mtp_path, '/');
            char nextn[24], sub[192];
            if (st.info.mtp_layers > 0)
                snprintf(nextn, sizeof nextn, " nextn=%d", st.info.mtp_layers);
            else
                nextn[0] = 0;
            snprintf(sub, sizeof sub, "draft head %s %.2f GiB %s%s",
                     bn ? bn + 1 : st.info.mtp_path,
                     (double)st.info.mtp_bytes / 1073741824.0, st.info.mtp_stage, nextn);
            qe_ui_head(NULL, sub);
        }
    }
    int rc = qe_http_serve(lfd, dispatch, &st, &ho);
    qe_ui_stop();
    if (qe_http_stop)
        fprintf(stderr, "qe: signalled, closing listener\n");
    else
        fprintf(stderr, "qe: accept loop ended (%s)\n", strerror(errno));
    /* Closing the listener stops new work, not the work in progress, and a request
     * thread is holding the engine this function is about to free. So wait for it -
     * but only as long as the grace. Giving up means leaving the engine alone and
     * leaving by _exit rather than return: the kernel gives back the weights either
     * way, and it gives them back faster than an exit-time free of 87 GiB would,
     * while running atexit handlers underneath a thread that is still generating is
     * a use-after-free with a leak report stapled to it. */
    int live = 0;
    if (!qe_http_drain(o.shutdown_grace_secs, &live)) {
        fprintf(stderr, "qe: %d request(s) still running after %d s, exiting without freeing\n",
                live, o.shutdown_grace_secs);
        fflush(stderr);
        _exit(rc ? 1 : 0);
    }
    qe_engine_close(eng);
    return rc;
}
