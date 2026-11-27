/* This model's own conversation contract: how a request becomes prompt bytes,
 * and how a decoded token stream becomes a reply. Internal to qe.

   Why a hand-written renderer instead of evaluating the Jinja template that
   ships in tokenizer.chat_template: a Jinja interpreter is a project on its
   own (docs/DESIGN.md counts it as one), and this checkpoint has exactly one
   template, read here character by character and implemented directly. That
   trades generality for two things the engine cares about - the prompt is
   rendered in microseconds rather than in a template pass, and the rules are
   visible C with the behaviour they reproduce written next to them.

   What is reproduced, in full:

     - leading system/developer messages are trimmed and joined with newlines
       into one system turn; a system message anywhere later is an error
     - the reasoning-effort instruction that is prepended to that turn when
       thinking is on (xhigh is the default, high means xhigh, medium adds no
       sentence, low adds its own)
     - the tools preamble: a schema block of one compact-JSON tool object per
       line followed by the fixed instruction block that spells out the call
       format, with the schema block going first and the caller's own system
       text after it
     - user, assistant and tool turns, including tool requests inside an
       assistant turn and runs of tool results collapsed into one user turn
     - reasoning replay: an assistant turn keeps its reasoning block, with the
       preserve_thinking switch and the last-real-query rule deciding when an
       older turn may drop it. A turn that carried no reasoning text renders no
       block at all, so that a client which stores thinking but never sends it
       back does not spend the transcript demonstrating an assistant that never
       thinks; --empty-thinking-blocks is what puts the empty one back
     - the generation prompt, which opens the reasoning block when thinking is
       on and closes an empty one when it is explicitly off

   The control tokens themselves are never spelled in this file. They are taken
   from the checkpoint's vocabulary by id, so a checkpoint that does not have
   them at those ids gets a refusal and the plain flattening fallback rather
   than a prompt built from a guess. Two of the ids are pinned by what they must
   be, and qe_chat_supported checks exactly that: the turn close must be the
   file's own end-of-sequence, and the turn open must be a distinct control
   token whose text is as long as it is.

   Deviation, deliberate and only in one place: the template refuses tool
   arguments that arrive as a JSON string. Every OpenAI client sends them that
   way, including the one this engine is built to serve, so a string is parsed
   as JSON and used as the object; a blank string means no arguments; anything
   that does not parse is still an error. */
#ifndef QE_CHAT_H
#define QE_CHAT_H

#include "http.h"
#include "json.h"
#include "qe.h"

#include <stddef.h>

/* Images one request may carry. A coding agent sends its whole history every turn,
 * screenshots included, so past this count the oldest ones are left out - the one the
 * user is asking about is the newest. */
#define QE_CHAT_MAX_IMAGES 32
#define QE_CHAT_MIME_MAX 64
#define QE_CHAT_MAX_IMAGE_BYTES (32u << 20)
/* What the transcript says where an image was. There is no vision encoder in this
 * build, and "[image 1: image/png, 253154 bytes]" alone was read by the model as an
 * image it had been shown: it answered a view of a PNG with "Now I can see the
 * pattern much better!" and described what it imagined. So the placeholder says what
 * the model can and cannot know, and an agent can go on to inspect the file with the
 * tools it does have. */
#define QE_CHAT_IMAGE_NOTE                                                                   \
    "The image content is not visible to you: this server has no vision encoder. Do not "  \
    "describe or guess what it shows; say you cannot see it, or inspect the file with "     \
    "other tools"

/* Vocabulary ids of the control tokens this model was trained with, as the
 * checkpoint's token_type array marks them. Kept as numbers so that no literal
 * marker sequence exists anywhere in this repository, including in its source. */
enum {
    QE_MARK_TURN_OPEN = 248045,
    QE_MARK_TURN_CLOSE = 248046, /* also tokenizer.ggml.eos_token_id */
    QE_MARK_TOOL_OPEN = 248058,
    QE_MARK_TOOL_CLOSE = 248059,
    QE_MARK_TOOL_OUT_OPEN = 248066,
    QE_MARK_TOOL_OUT_CLOSE = 248067,
    QE_MARK_REASON_OPEN = 248068,
    QE_MARK_REASON_CLOSE = 248069,
    /* An image in a message, as the template spells it: the start and end markers
     * around one image pad, which is also the file's ple.image_token_id and the
     * position the engine replaces with the image's rows. */
    QE_MARK_VISION_OPEN = 248053,
    QE_MARK_VISION_CLOSE = 248054,
    QE_MARK_IMAGE_PAD = 248056
};

/* One rendered request: the prompt bytes, plus what could not be represented as
 * text. `prompt` is already the prompt - the control tokens are inline in it,
 * and the tokenizer folds each one back to a single id. */
typedef struct {
    qe_str prompt;
    qe_image images[QE_CHAT_MAX_IMAGES];
    char *image_buf[QE_CHAT_MAX_IMAGES];
    char image_mime[QE_CHAT_MAX_IMAGES][QE_CHAT_MIME_MAX];
    int n_images;
    size_t image_bytes;
    int truncated_images; /* more images than QE_CHAT_MAX_IMAGES */
    int image_skip;       /* how many of the oldest images are left out */
    int image_seen;       /* image parts met so far, kept or not */
    int chat;             /* messages were rendered, as opposed to a raw prompt */
    int has_reasoning;   /* the transcript carries the model's own reasoning */
    int opens_reasoning; /* the prompt leaves the model inside an open reasoning block,
                            which is the only thing that decides where its next
                            tokens belong */
    int tools_rendered;   /* the schema preamble went in */
    long n_messages;
    /* Images go in as the template's three markers rather than as a sentence about
     * them: set when the caller said the engine can see (qe_chat_opts.vision) and the
     * vocabulary has the markers. `image_mark` is their text, from the vocabulary. */
    int vision;
    char image_mark[160];
} qe_chat;

/* What the caller asked for, after the request's own spellings are reconciled.
 * A client can ask for thinking through enable_thinking, through a thinking
 * object, or by leaving reasoning_effort alone, so serve.c resolves all of them
 * into this and the renderer sees one decision. */
typedef struct {
    int enable_thinking;  /* 0 = explicitly off, anything else = on */
    int effort_xhigh;     /* 1 = the xhigh sentence, 0 = the low one */
    int effort_low;
    int preserve_thinking; /* 1 unless the caller set it false */
    /* Not a client knob. Set by the server from --empty-thinking-blocks: a replayed
     * assistant turn that carried no reasoning text renders an empty reasoning
     * block only when this is on. An empty block is not neutral, it is the shape
     * the model reads as "thought of nothing", and a client that never sends
     * thinking back would otherwise replay forty turns of an assistant that never
     * thinks. */
    int empty_thinking_blocks;
    /* Not a client knob either: the server sets it when the engine has a vision tower
     * that will turn each image's position into the image's rows. Off, an image is a
     * sentence that says it cannot be seen. */
    int vision;
} qe_chat_opts;

/* Fills o from a request body: enable_thinking (also accepted inside
 * chat_template_kwargs and any other top-level object a client invents),
 * reasoning_effort, preserve_thinking, and the thinking-object shape. Unknown
 * efforts are reported as an error, which is what the template does. */
int qe_chat_opts_from_body(const qe_json *body, qe_chat_opts *o, char *why, size_t whyn);

/* Does this vocabulary have this model's control tokens where the model expects
 * them? 0 means the renderer must not be used; serve.c then flattens messages
 * the plain way, as it always did for a checkpoint it cannot speak to. */
int qe_chat_supported(const qe_tok *t);

/* Renders messages[] (and tools[]) into r. why is filled on failure. */
int qe_chat_render(qe_tok *t, const qe_json *body, const qe_chat_opts *o, qe_chat *r, char *why,
                   size_t whyn);
void qe_chat_free(qe_chat *r);

/* A message's role, with the OpenAI default of user when it is missing. */
const char *qe_chat_role(const qe_json *m);

/* Decodes one base64 data URL into the next slot of r, recording its mime type
 * and size so a caller can report what it was handed. Fills why on failure,
 * including the refusal to fetch an http(s) URL. */
int qe_chat_record_image(qe_chat *r, const char *url, char *why, size_t whyn);

/* --------------------------------------------------------------- turn parser
 *
 * Splits the decoded stream into the three things a reply can contain. Only the
 * control tokens decide state, which is why the parser is driven token by token
 * with its id: the reasoning block, the answer, and a request to call a tool
 * are all delimited by ids, and nothing a model writes inside the answer can be
 * mistaken for one of them.
 *
 * Tool calls are reported when their block closes, not as they stream. The
 * block is delimited by control tokens and its contents are structure rather
 * than text, so buffering it costs a little latency on a path where the client
 * cannot act on a partial argument list and buys an exactly-parsed name and
 * argument object instead. */
enum {
    QE_TURN_CONTENT,   /* answer text */
    QE_TURN_REASONING, /* reasoning text */
    QE_TURN_CALL_OPEN, /* a tool request started; index, name, id are set */
    QE_TURN_CALL_ARGS, /* the request's arguments as one JSON object string */
    QE_TURN_CALL_CLOSE,
    QE_TURN_STOP /* the assistant turn ended */
};

typedef struct {
    int kind;
    const char *utf8;
    int nbytes;
    int index;          /* tool request number, from 0 */
    const char *name;   /* tool name, QE_TURN_CALL_OPEN */
    int name_len;
    const char *call_id; /* synthesised id, stable for one request */
    const char *finish;  /* QE_TURN_STOP: stop or tool_calls */
} qe_turn_ev;

typedef int (*qe_turn_cb)(void *user, const qe_turn_ev *ev);

typedef struct qe_turn qe_turn;

/* starts_in_reasoning: 1 when the prompt left the model inside a reasoning
 * block, which is the normal case, 0 when thinking was turned off. */
qe_turn *qe_turn_new(qe_turn_cb cb, void *user, int starts_in_reasoning);
void qe_turn_free(qe_turn *t);

/* The stem that a reply's tool-call ids are built from, normally the id of the
 * request being answered. A client keys its own tool-call state by id: the block
 * of interface a call is drawn into, and the table a result is folded back into
 * the call it belongs to, are both found by that id. So an id handed out twice in
 * one conversation costs two things at once - the second call never gets a block
 * of its own, because the client finds the first one and writes over it, and the
 * results of both collapse into whichever one the id maps to last. Numbering calls
 * from 0, which is what happens with no stem set, is unique inside a reply and
 * nothing wider, and a reply is one turn of a conversation that keeps going. */
void qe_turn_set_id_prefix(qe_turn *t, const char *prefix);

/* The request's tools[] array, which decides what a parameter's value is. The model
 * writes every value as bare text between two tags, and the argument object a client
 * receives is JSON, so something has to say whether `001` is a number or a string.
 * The schema says so: a parameter declared "string" is the text the model wrote, byte
 * for byte inside the newline that frames it - indentation and quotes included, which
 * is what an edit tool's old_string has to be to match anything. Anything else is
 * read as JSON and re-serialised, so what leaves is always valid JSON. A parameter the
 * schema does not mention falls back to reading the text. charmbracelet's client names
 * background shells "001", "00A", and a guess that wrote 001 as a number made every
 * job_output call arrive as invalid JSON, which the client then replaced with {}.
 * The pointer is borrowed and has to outlive the parser; NULL means no schema. */
void qe_turn_set_tools(qe_turn *t, const qe_json *tools);

/* One decoded token. `utf8` is its text and must be empty for a control token,
 * which is what the engine reports. Non-zero stops generation (a client that
 * went away). Malformed structure is skipped, never forwarded as answer text.
 *
 * One rescue happens at the turn close. A turn that closes while its reasoning block
 * is still open, having written no answer and asked for nothing, wrote its answer
 * inside the reasoning: the reasoning text is then reported once more as answer text,
 * just before the stop. Without it the reply holds nothing a client shows, and a
 * sub-agent whose last turn is that shape hands its parent "produced no text output"
 * after minutes of work. A turn cut off by the budget is not rescued - it did not
 * choose to end, and finish_reason length already says so. */
int qe_turn_token(qe_turn *t, int id, const char *utf8, int nbytes);
/* Flushes what is pending at the end of generation: text held back inside an
 * unterminated block is dropped, because a reply cut off mid-request is
 * suppressed by the client anyway. */
int qe_turn_end(qe_turn *t);
long qe_turn_reasoning_bytes(const qe_turn *t);

#endif /* QE_CHAT_H */
