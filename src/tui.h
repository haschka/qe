/* The terminal interface qe-serve draws while it is serving.

   One question this server could not previously answer from the outside while it
   was answering: is it working. An agent prompt holds the engine for minutes
   before a byte goes out, and the only evidence during that time was that the
   process still existed. This draws, several times a second, how many tokens are
   going through the network right now and how many are coming out, which is the
   difference between "ingesting a 11,758-token prompt" and "hung".

   Three rules hold it together:

     - It renders to stdout, and only when stdout is a terminal. Every harness in
       this repository redirects it to a file, which is exactly when the panel has
       to disappear and the log lines have to come back.
     - One writer. While the panel is on, the per-request log line lives inside it
       rather than on stderr beside it, because two writers on one screen produce
       a scramble rather than a log.
     - It reads atomics and takes no lock the engine takes. A status surface that
       can make a decode wait 250 ms is not a status surface, so nothing here is
       on a token's path except an integer store. */
#ifndef QE_TUI_H
#define QE_TUI_H

#include "qe.h"

/* Starts the renderer. QE_UI_AUTO means "only if stdout is a terminal". `head` is
 * the sentence the panel repeats above the numbers - engine, window, where the
 * weights are - because the startup banner scrolls away as soon as anything
 * redraws, and a panel that forgets which model it is watching is a panel people
 * stop trusting. NULL is allowed. */
void qe_ui_start(int want, const char *head);
/* Stops the thread and leaves the last frame on the screen. Safe to call when the
 * panel was never started, which is every case but qe_serve. */
void qe_ui_stop(void);
int qe_ui_on(void);

/* What the panel repeats instead of a banner, refreshed when a fact about the
 * engine changes. Either may be NULL. */
void qe_ui_head(const char *head, const char *sub);

/* The request the numbers belong to, and one finished request's line. */
void qe_ui_request(const char *kind, const char *id);
void qe_ui_log(const char *line);

/* Fed by the engine, which is where the truth about tokens is. gen_begin opens a
 * request whose prompt encoded to `prompt_tokens` and of which `already` are
 * already in the sequence; ingest is the running count through that prompt, and
 * ingested/cached are the deltas to add to the process totals. */
void qe_ui_gen_begin(long prompt_tokens, long already);
void qe_ui_ingest(long done, long total);
/* The vision tower is encoding image `image` of `images`, `done` of its `total` blocks
 * through. The prompt rows show the image until ingestion reports again. */
void qe_ui_vision(int image, int images, long done, long total);
void qe_ui_ingested(long tokens);
void qe_ui_cached(long tokens);
void qe_ui_emit(void);
void qe_ui_gen_end(void);
/* A request that began from a rewind point rather than from nothing, by this many
 * tokens; and what the cache is holding now. `snapshot_us` is what the last copy
 * cost, or -1 when nothing was copied. */
void qe_ui_rewind(long tokens);
void qe_ui_cache(long slots, long held, long long bytes, long snapshot_us);

#endif /* QE_TUI_H */
