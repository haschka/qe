"""Smoke test for the OpenAI-compatible surface (default :8099, QE_PORT to move).

Not a unit test: it drives the real socket with a real client library and checks
the shapes a client actually depends on - content-type, SSE framing, the
delta/content vs text difference between the two completion endpoints, and that
refusals are refusals rather than empty 200s.
"""
import base64
import http.client
import json
import re
import os
import sys

PORT = int(os.environ.get("QE_PORT", "8099"))

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def req(method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    status = r.status
    ctype = r.getheader("Content-Type")
    c.close()
    return status, ctype, data


fails = []


def check(ok, what, detail=None):
    print(("  ok   " if ok else "  FAIL ") + what + ("" if detail is None else ": " + str(detail)))
    if not ok:
        fails.append(what)

# Model text is never echoed with its control tokens in it. A vocabulary marker
# that reaches a terminal or a log is a control token one copy-paste away from
# being live, and this repository keeps none of them in text - so an echo of a
# reply names its markers instead of spelling them. Raw string: an escaped pipe is
# a literal, an unescaped one is alternation.
MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")


def say(*xs):
    return " ".join(
        MARKER.sub(lambda m: "{" + m.group(0)[2:-2].strip() + "}", str(x)) for x in xs
    )


print("== GET /health ==")
st, ct, body = req("GET", "/health")
print("   ", say(st, ct, body[:200]))
check(st == 200 and ct.startswith("application/json"), "200 json")
check(json.loads(body)["status"] == "ok", "status ok")

print("== GET /v1/models ==")
st, ct, body = req("GET", "/v1/models")
m = json.loads(body)
print("   ", say(json.dumps(m, indent=2)[:900]))
check(st == 200 and m["object"] == "list", "list shape")
check(m["data"][0]["object"] == "model", "model object")
check(m["data"][0]["id"] == "qe-selftest", "id")
check(m["data"][0]["qe"]["can_generate"] is True, "reports it can generate")
check("tokenizer_note" in m["data"][0]["qe"], "token estimate is labelled")
# The window is what a client budgets a conversation against, and it is also the
# ceiling on one reply, so the number has to be readable before asking.
CTX = m["data"][0]["qe"]["context"]
print("   ", say("advertised context:", CTX))
# Where the weights live decides whether a generation would be memory-speed or
# disk-speed, so it has to be readable without asking the operator.
q = m["data"][0]["qe"]
check(isinstance(q["weights_resident"], bool), "says whether the weights are resident")
# What a second turn has to go back to. The selftest engine allocates no sequence at
# all, so the honest answer about rewind points is zero - and a checkpoint server is
# expected to answer with the points its budget buys, which is why the keys are here
# whether or not there is anything behind them.
check(q["turn_cache_slots"] == 0, "no rewind points for an engine with no sequence")
check(q["turn_cache_held"] == 0, "and none held")
check(q["turn_cache_bytes"] == 0, "costing nothing")
check(q["turn_cache_saved"] == 0, "and having saved nothing yet")
# The two that say whether the cache is why a turn re-read its prompt: short is the
# diagnosis, and the budget is the number to quote when it is.
check(q["turn_cache_budget_mib"] == 0, "and no budget set aside for them")
check(q["turn_cache_short"] is False, "and nothing short when nothing was asked for")
check(isinstance(q["weight_stage"], str) and q["weight_stage"], "names the weight backing")
check(isinstance(q["weights_pinned"], bool), "says whether they are pinned")
check(isinstance(q["weight_stage_gib"], float), "says how much was placed")
# A draft head is a file somebody named on a command line, and `"mtp_head": true`
# alone cannot tell the head this server opened from the head it was never given. So
# the card says which file, how big it is, and where its bytes were placed - the same
# three facts it gives for the checkpoint. `make http` starts this server with
# --mtp pointing at the fixture from tools/make_mtp_fixture.py.
check(q["mtp_head"] is True, "says it has a draft head")
check(isinstance(q.get("mtp_file"), str) and q["mtp_file"].endswith(".gguf"),
      "names the draft head file", say(q.get("mtp_file")))
check(q.get("mtp_tensors") == 4, "counts its tensors", say(q.get("mtp_tensors")))
check(int(q.get("mtp_weight_bytes", 0)) > 0, "says how many weight bytes it holds",
      say(q.get("mtp_weight_bytes")))
check(q.get("mtp_stage") in ("ram", "ram+thp", "hugetlb"),
      "says where the head's bytes were placed", say(q.get("mtp_stage")))
# Resident and drafting are two facts, and only one of them is a server that
# speculates: a head this server opened but declined to walk is a pile of bytes and
# an n-gram drafter. The card has to separate them, since this server is handed a
# fixture that is a head by name and not by tensors.
check(isinstance(q.get("mtp_drafting"), bool),
      "says whether the head is what is drafting", say(q.get("mtp_drafting")))
# The attention cache is the state that grows with the conversation, and whether
# it holds 8 or 32 bits per value decides how many sequences fit on the box.
check(q["kv_cache"] in ("q8_0", "f32", "none"), "names the key/value cache")
# A client that has never heard of this server still wants the window before it
# sends a conversation, and the spelling it looks for is llama-server's: crush
# reads data[].meta.n_ctx, falls back to n_ctx_train, and reads nothing else -
# which is how a server that advertises a window only in its own namespace ends up
# with a client that guessed. Same numbers as the qe block, both spellings.
meta = m["data"][0].get("meta") or {}
print("   ", say("meta block:", json.dumps(meta)))
check(meta.get("n_ctx") == CTX and meta.get("n_ctx_train") == q["context_train"],
      "the window is also advertised where a llama.cpp client looks")

MODEL = m["data"][0]["id"]

print("== POST /v1/chat/completions (string content, greedy) ==")
st, ct, body = req(
    "POST",
    "/v1/chat/completions",
    {"model": MODEL, "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 24},
)
r = json.loads(body)
print("   ", say(json.dumps(r)[:600]))
check(st == 200, "200")
check(r["object"] == "chat.completion", "object")
check(r["choices"][0]["message"]["role"] == "assistant", "role")
check(len(r["choices"][0]["message"]["content"]) > 10, "non-empty content")
check(r["choices"][0]["finish_reason"] in ("stop", "length"), "finish_reason")
check(r["usage"]["completion_tokens"] > 0, "usage.completion_tokens")
check(r["qe"]["engine"] == "selftest", "qe block reports the engine")

print("== determinism: same request, same tokens ==")
st, _, body2 = req(
    "POST",
    "/v1/chat/completions",
    {"model": MODEL, "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 24},
)
check(json.loads(body2)["choices"][0]["message"]["content"] == r["choices"][0]["message"]["content"],
      "greedy is reproducible")
check(json.loads(body2)["id"] != r["id"], "but each response has its own id")

print("== POST /v1/chat/completions with an image content part ==")
url = "data:image/png;base64," + base64.b64encode(PNG).decode()
st, ct, body = req(
    "POST",
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this picture?"},
                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
                ],
            }
        ],
        "max_tokens": 30,
    },
)
r = json.loads(body)
print("   ", say(json.dumps(r)[:700]))
check(st == 200, "200 with an image")
meas = r["qe"]["measured"]
check(meas["images"] == 1, "one image recorded")
check(meas["image_bytes"] == len(PNG), f"decoded to {len(PNG)} bytes")
check("[image 1: image/png" in r["choices"][0]["message"]["content"] or
      "images=1" in r["choices"][0]["message"]["content"], "the image reached the prompt")

print("== POST /v1/chat/completions stream:true ==")
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request(
    "POST",
    "/v1/chat/completions",
    body=json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 20,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ),
    headers={"Content-Type": "application/json"},
)
resp = c.getresponse()
raw = resp.read().decode()
c.close()
print("   ", say(raw[:300].replace("\n", "\\n")))
check(resp.getheader("Content-Type", "").startswith("text/event-stream"), "SSE content type")
# Two headers whose absence is indistinguishable, to the user, from a dead engine:
# anything between here and the client that is allowed to cache or buffers by
# default hands the stream over in one piece when the turn ends, which reads as an
# engine that never answered until it answered all at once.
check(resp.getheader("Cache-Control") == "no-cache", "a stream is marked no-cache")
check(resp.getheader("X-Accel-Buffering") == "no", "a stream asks intermediaries not to buffer it")
lines = [ln for ln in raw.split("\n") if ln.startswith("data: ")]
check(lines[-1] == "data: [DONE]", "terminates with [DONE]")
chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
check(not [x for x in chunks if x["choices"] and not x["choices"][0].get("delta")
           and x["choices"][0].get("finish_reason") is None],
      "no chunk on the wire says nothing")
check(all(x["object"] == "chat.completion.chunk" for x in chunks), "chunk objects")
check(chunks[0]["choices"][0]["delta"].get("role") == "assistant", "first chunk carries the role")
text = "".join(x["choices"][0]["delta"].get("content", "") for x in chunks if x["choices"])
check(len(text) > 10, "streamed text accumulates")
final = [x for x in chunks if x["choices"]]
check(final and final[-1]["choices"][0]["finish_reason"] in ("stop", "length"),
      "final content chunk carries finish_reason")
usage_at = [i for i, x in enumerate(chunks) if "usage" in x]
check(len(usage_at) == 1, "include_usage produced exactly one usage chunk")
# Last, not merely present. charmbracelet's client re-reads the usage from every
# chunk and a chunk without usage resets it to zero, so a usage chunk followed by
# anything - including a chunk that only carries the finish reason - costs the
# client its token counts. Same reason the ids must agree: the accumulator keys a
# reply by the id of its first chunk.
check(usage_at and usage_at[0] == len(chunks) - 1,
      "the usage chunk is the last data event before [DONE]")
check(len(set(x["id"] for x in chunks)) == 1, "every chunk carries the same id")
# The same request without `stream`. Assembling a reply out of deltas and
# assembling it once are two code paths through one buffer, and a token appended
# by both of them arrives twice - which is a thing that happened, and which
# "shapes agree" would never have caught.
st, ct, body = req("POST", "/v1/chat/completions",
                   {"model": MODEL, "messages": [{"role": "user", "content": "stream please"}],
                    "max_tokens": 20})
buffered = json.loads(body)["choices"][0]["message"]["content"]
check(st == 200 and buffered == text,
      "the buffered content is exactly the streamed deltas, each once")

# A request shaped the way charmbracelet/crush sends one: the tool schemas on
# every turn, the assistant's own tool request replayed, one role:"tool" message
# per result, the reasoning the model showed on the previous turn, and
# max_completion_tokens rather than max_tokens. It reads reasoning out of
# delta.reasoning_content, accumulates tool-call deltas per index, and treats a
# stream that never sent a terminal finish_reason as broken. It also sends
# Accept: application/json even for a stream, so the response must not be
# content-negotiated away from event-stream.
CRUSH = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call a tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a\nb"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
                # Always present and always false: charmbracelet's client fills
                # the field in for every tool on every turn, so a server that
                # tripped over it would fail on the first request of every
                # conversation.
                "strict": False,
            },
        }
    ],
    "tool_choice": "auto",
    "reasoning_effort": "high",
    "max_completion_tokens": 24,
}

print("== a crush-shaped request, streamed ==")
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request(
    "POST",
    "/v1/chat/completions",
    body=json.dumps(dict(CRUSH, stream=True, stream_options={"include_usage": True})),
    headers={"Content-Type": "application/json", "Accept": "application/json"},
)
resp = c.getresponse()
raw = resp.read().decode()
c.close()
print("   ", say(resp.status, resp.getheader("Content-Type"), raw[:200].replace("\n", "\\n")))
check(resp.status == 200, "200 for a request carrying tools and results")
check(resp.getheader("Content-Type", "").startswith("text/event-stream"),
      "Accept: application/json does not talk it out of SSE")
# A keepalive is an SSE comment, which is a line a message assembler never sees.
# Asserted as an invariant over the whole stream rather than as one observed
# comment: a comment exists only while the engine is reading a prompt, and the
# selftest surface has no prompt to read, so the lines that prove it exist are a
# checkpoint's to produce.
check(all(ln.startswith("data: ") or ln.startswith(": ") for ln in raw.split("\n") if ln),
      "every line of a stream is an event or a comment, never bare text")
clines = [ln for ln in raw.split("\n") if ln.startswith("data: ")]
check(clines[-1] == "data: [DONE]", "crush-shaped stream terminates")
cchunks = [json.loads(ln[6:]) for ln in clines[:-1]]
# What a client that shows a thinking block as it arrives reassembles, kept so the
# buffered answer can be compared against it below: a reasoning model's reply goes
# down two channels, and both of them are assembled by hand here.
stext = "".join((x["choices"][0]["delta"].get("content") or "") for x in cchunks if x["choices"])
sreason = "".join((x["choices"][0]["delta"].get("reasoning_content") or "")
                  for x in cchunks if x["choices"])
finished = [x for x in cchunks if x["choices"] and x["choices"][0]["finish_reason"]]
check(len(finished) == 1, "exactly one chunk carries a finish_reason")
check(finished and finished[0]["choices"][0]["finish_reason"] in
      ("stop", "length", "tool_calls", "content_filter"), "finish_reason is terminal")
check(all("reasoning_content" not in x["choices"][0]["delta"] or
          x["choices"][0]["delta"]["reasoning_content"] is not None for x in cchunks if x["choices"]),
      "no chunk carries a null reasoning_content")
usage = [x.get("usage") for x in cchunks if x.get("usage")]
check(len(usage) == 1, "one usage chunk")
u = usage[0] if usage else {}
check(isinstance(u.get("prompt_tokens_details", {}).get("cached_tokens"), int),
      "usage reports cached_tokens")
check(isinstance(u.get("completion_tokens_details", {}).get("reasoning_tokens"), int),
      "usage reports reasoning_tokens")
check(u.get("total_tokens") == u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
      "usage adds up")

print("== the same request, buffered ==")
st, ct, body = req("POST", "/v1/chat/completions", CRUSH)
r = json.loads(body)
print("   ", say(json.dumps(r)[:600]))
check(st == 200, "200 buffered")
msg = r["choices"][0]["message"]
check(len(stext) + len(sreason) > 10, "the streamed reply has bytes to compare")
check(stext == (msg.get("content") or ""),
      "buffered content is the streamed content deltas, each once")
check(sreason == (msg.get("reasoning_content") or ""),
      "buffered reasoning is the streamed reasoning deltas, each once")
check("reasoning_content" not in msg or msg["reasoning_content"],
      "an empty reasoning block is omitted, not sent as the empty string")
check(r["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls"), "buffered finish_reason")
check("cached_tokens" in r["usage"]["prompt_tokens_details"], "buffered usage has cached_tokens")
m = r["qe"]["measured"]
# Whether any of this prompt was already in the sequence, said even when the answer
# is no: a client working out what it is being charged for cannot tell an unused
# cache from a broken one without the zero. `rewound` distinguishes the two ways a
# turn can start short - carrying the previous one forward, or putting a saved one
# back - and `snapshot_ms` is what the putting back cost.
check(m["cached_tokens"] == 0, "an engine with no sequence reuses nothing")
check(m["rewound"] is False, "and never rewound")
check(isinstance(m["snapshot_ms"], float), "with the copy time reported anyway")
# The selftest engine has no tokenizer, so this request was flattened rather than
# templated, and the honest report is that no schema preamble went in. With a real
# checkpoint the schemas are rendered and sorted; bin/bench_chat checks that.
check(m["schemas"] == 0, "flattening says so rather than claiming schemas")
check(m["messages"] == 4, "every message was counted, tools and results included")
check(m["effort"] == "xhigh", "reasoning_effort high is answered at xhigh")
check(m["thinking"] == 1, "thinking is on")
# The two rates answer different questions, and mixing them up is a factor of
# however long the prompt was: tok_per_s counts emitted tokens over a duration
# that includes the prompt, passes_per_s counts passes (uncached prompt tokens
# plus emitted tokens) over that same duration. Same denominator, so their ratio
# is passes over tokens whatever the clock rounded to.
want = m["target_passes"] / m["tokens"]
ratio = m["passes_per_s"] / m["tok_per_s"]
check(abs(ratio - want) <= want * 0.01,
      "passes_per_s and tok_per_s divide by the same seconds")
# The budget a reply used is echoed back, because a client that sent no length at
# all cannot otherwise tell a short answer from an amputated one: finish_reason
# "stop" is the same either way. Two request spellings, one number out.
check(m["budget"] == 24, "max_completion_tokens is the budget the reply used")
unbudgeted = {k: v for k, v in CRUSH.items() if k != "max_completion_tokens"}
st, ct, body = req("POST", "/v1/chat/completions", unbudgeted)
m = json.loads(body)["qe"]["measured"]
print("   ", say("budget with no length in the request:", m["budget"]))
check(st == 200 and m["budget"] >= 4096,
      "no length at all means until the model stops, not a few dozen tokens")
# The ceiling on a reply is the window the server opened with and not a constant
# buried in the engine: an ask larger than anything this model can hold comes back
# trimmed to it, and says so. This engine allocated no sequence, so its window is
# the default; a checkpoint reports its whole 262144, which is what keeps a
# 12000-token thinking block a thinking block rather than a finished answer.
st, ct, body = req("POST", "/v1/chat/completions", dict(CRUSH, max_tokens=10 ** 9))
big = json.loads(body)["qe"]["measured"]
print("   ", say("budget for an ask of 1e9:", big["budget"], "in a window of",
                 CTX if CTX > 0 else "none"))
check(st == 200 and big["budget"] == (CTX if CTX > 0 else 4096),
      "an ask past the window is answered at the window, and reported as such")
# Prompt tokens that had to be thrown away because the prompt plus the answer did
# not fit are reported, because a reply that never saw the start of its own
# transcript otherwise reads as a model that cannot follow one. This engine has no
# sequence at all, so the honest answer here is zero - what is being asserted is
# that the number is on the wire rather than that it is small.
check(big.get("dropped") == 0, "says how much of the prompt was dropped")
# The effort scale, at the two ends a client can name that this model has no
# sentence for: "minimal" is the shortest thing on offer, "none" is no reasoning
# block at all. Both are answered rather than refused, because the client that
# sends them has already decided.
for effort, (want_effort, want_thinking) in (("minimal", ("low", 1)), ("none", ("medium", 0))):
    st, ct, body = req("POST", "/v1/chat/completions",
                       dict(CRUSH, reasoning_effort=effort, max_tokens=8))
    m = json.loads(body)["qe"]["measured"]
    print("   ", say(f"{effort} -> {m['effort']} thinking={m['thinking']}"))
    check(st == 200 and m["effort"] == want_effort and m["thinking"] == want_thinking,
          f"reasoning_effort {effort} is answered, not refused")
# The title request: a client that wants no reasoning and has no field for it says
# so with this model family's off-marker at the end of the system prompt, and gives
# forty tokens to answer in. Reasoning on means the answer is nothing at all.
TITLE = {"model": MODEL,
         "messages": [{"role": "system", "content": "Generate a title.\n /no_think"},
                      {"role": "user", "content": "notes about memory bandwidth"}],
         "max_tokens": 40}
st, ct, body = req("POST", "/v1/chat/completions", TITLE)
m = json.loads(body)["qe"]["measured"]
print("   ", say(f"thinking={m['thinking']} reasoning_bytes={m['reasoning_bytes']}"))
check(st == 200 and m["thinking"] == 0, "the no-think marker in a system turn is believed")
check(m["reasoning_bytes"] == 0, "and no reasoning block comes back with the answer")
st, ct, body = req("POST", "/v1/chat/completions", dict(TITLE, reasoning_effort="high"))
check(json.loads(body)["qe"]["measured"]["thinking"] == 1,
      "an explicit effort field outranks the marker in the prompt")

print("== what a reasoning request is refused for ==")
st, ct, body = req("POST", "/v1/chat/completions",
                   dict(CRUSH, reasoning_effort="nonsense", max_tokens=8))
print("   ", say(st, body[:160]))
check(st == 400 and "reasoning_effort" in json.loads(body)["error"]["message"],
      "an unknown reasoning effort is refused rather than guessed")
st, ct, body = req("POST", "/v1/chat/completions",
                   dict(CRUSH, tool_choice={"type": "function", "function": {"name": "bash"}},
                        max_tokens=8))
check(st == 200, "a forced tool_choice is accepted")

print("== POST /v1/completions ==")
st, ct, body = req(
    "POST", "/v1/completions", {"model": MODEL, "prompt": "Once upon a time", "max_tokens": 16}
)
r = json.loads(body)
print("   ", say(json.dumps(r)[:400]))
check(st == 200 and r["object"] == "text_completion", "text_completion object")
check("text" in r["choices"][0] and len(r["choices"][0]["text"]) > 5, "choices[0].text")
# Streamed, the same answer has to reassemble to the buffered one byte for byte.
# Completions assembles in a different function from chat, so this is a second
# copy of the invariant rather than a restatement of the first.
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request(
    "POST",
    "/v1/completions",
    body=json.dumps({"model": MODEL, "prompt": "Once upon a time", "max_tokens": 16,
                     "stream": True}),
    headers={"Content-Type": "application/json"},
)
sraw = c.getresponse().read().decode()
c.close()
stxt = "".join(
    json.loads(ln[6:])["choices"][0].get("text") or ""
    for ln in sraw.split("\n")
    if ln.startswith("data: ") and ln != "data: [DONE]"
)
check(len(stxt) > 5 and stxt == r["choices"][0]["text"],
      "the buffered text is exactly the streamed deltas, each once")
arr, act, abody = req(
    "POST", "/v1/completions", {"model": MODEL, "prompt": ["a", "b"], "max_tokens": 8}
)
r2 = json.loads(abody)
# An array prompt is a batch: one engine run per element, one choice per element,
# in the order they were posted. Anything else silently answers a different
# question than the client asked.
check(arr == 200 and len(r2["choices"]) == 2, "array prompt returns one choice per prompt")
check([c["index"] for c in r2["choices"]] == [0, 1], "choices keep the posted order")
check(all("text" in c and "message" not in c for c in r2["choices"]), "completions shape")

for bad, why in (
    ([], "empty prompt array"),
    (["ok", 5], "non-string element"),
    (["x"] * 17, "more than 16 prompts"),
):
    stx, ctx, bx = req(
        "POST", "/v1/completions", {"model": MODEL, "prompt": bad, "max_tokens": 4}
    )
    msg = json.loads(bx)["error"]["message"]
    print(f"   {why} -> {stx} {msg}")
    check(stx == 400, f"400 on {why}")

stx, ctx, bx = req(
    "POST",
    "/v1/completions",
    {"model": MODEL, "prompt": ["a", "b"], "stream": True, "max_tokens": 4},
)
msg = json.loads(bx)["error"]["message"]
print("   array + stream ->", stx, msg)
check(stx == 400 and "one prompt" in msg, "array prompt with stream:true is refused")

print("== the sampling knobs ==")
# The other server on this host is launched with --temp 1.0 --top-p 0.95
# --top-k 20 --min-p 0.0, so the same four numbers have to mean the same thing
# here, on both endpoints, or moving a client over changes its answers.
for path, payload in (
    ("/v1/chat/completions",
     {"model": MODEL, "messages": [{"role": "user", "content": "sampling knobs"}],
      "max_tokens": 12}),
    ("/v1/completions", {"model": MODEL, "prompt": "sampling knobs", "max_tokens": 12}),
):
    st, ct, body = req("POST", path, dict(payload, temperature=1.0, top_p=0.95,
                                          top_k=20, min_p=0.0))
    print("   ", path, st, body[:120])
    check(st == 200, f"{path} takes temperature/top_p/top_k/min_p")

# What these four knobs do to a distribution cannot be asserted here, for two
# reasons that are both worth knowing. The selftest engine generates through the
# speculation loop, which draws from the target's raw softmax and applies no
# filters at all (qe_spec_opts), so a request's top_k changes nothing: at
# temperature 6 a top_k of 1 answers with the same text as no top_k at all. And
# the synthetic logits tie at the top, so even a filter that did run would be
# hard to see. `make sample` is where the three filters are pinned, against a
# vocabulary whose probabilities are known exactly. What the endpoints are checked
# for is accepting the spellings, refusing the impossible ones, and agreeing about
# presence - all of which is this layer's own business.
for mp in (0.0, 0.5, 0.999):
    stx, ctx, bx = req("POST", "/v1/chat/completions",
                       dict(CRUSH, max_tokens=12, temperature=6.0, seed=3, min_p=mp))
    ok = stx == 200 and len(json.loads(bx)["choices"][0]["message"]["content"]) > 0
    print("   ", "min_p", mp, "->", stx)
    check(ok, f"a floor of {mp} answers at temperature 6")

for bad, why in (
    ({"top_p": 1.5}, "top_p above 1"),
    ({"top_p": -0.1}, "negative top_p"),
    ({"min_p": -0.1}, "negative min_p"),
    ({"min_p": 1.0}, "min_p 1"),
    ({"top_k": -2}, "top_k below -1"),
    ({"top_k": 1.5}, "non-integer top_k"),
    ({"top_k": "auto"}, "string top_k"),
    ({"temperature": -1.0}, "negative temperature"),
):
    stx, ctx, bx = req("POST", "/v1/chat/completions",
                       dict(CRUSH, max_tokens=8, **bad))
    msg = json.loads(bx)["error"]["message"]
    print(f"   {why} -> {stx} {msg}")
    check(stx == 400, f"400 on {why}")
    check("top_" in msg or "min_p" in msg or "temperature" in msg,
          f"the {why} message names the knob")

# The off spellings have to be accepted, not clamped: -1 for top_k and 1 for
# top_p are what a client sends when it wants the whole distribution.
st, ct, body = req("POST", "/v1/completions", {"model": MODEL, "prompt": "off spellings",
                                               "max_tokens": 8, "top_k": -1, "top_p": 1.0,
                                               "min_p": 0.0})
print("   ", say(st, body[:120]))
check(st == 200, "top_k -1 and top_p 1 are accepted as off")


def floored(**kw):
    return dict(CRUSH, max_tokens=12, temperature=1.0, seed=3, **kw)


# "Absent" and "sent as 0" have to land in the same place on a server that was
# not started with a floor, and the two off spellings have to agree with each
# other. Both are the presence-handling a proxy adds without asking, so a server
# that reads only the value would answer differently for reasons nobody set.
def content(body):
    return json.loads(body)["choices"][0]["message"]["content"]


st, _, a = req("POST", "/v1/chat/completions", floored())
st2, _, b = req("POST", "/v1/chat/completions", floored(min_p=0.0))
st3, _, c = req("POST", "/v1/chat/completions", floored(top_k=0))
st4, _, d = req("POST", "/v1/chat/completions", floored(top_k=-1))
check(st == st2 == st3 == st4 == 200, "all four knob spellings answer")
check(content(a) == content(b), "min_p 0 draws what an absent min_p draws")
check(content(c) == content(d), "top_k 0 draws what top_k -1 draws")

print("== refusals ==")
st, ct, body = req("POST", "/v1/chat/completions", {"model": "gpt-99", "messages": []})
check(st == 404 and json.loads(body)["error"]["type"] == "model_not_found", "unknown model 404")
st, ct, body = req("POST", "/v1/chat/completions", {"model": MODEL})
check(st == 400, "missing messages 400")
st, ct, body = req("GET", "/v1/chat/completions")
check(st == 405, "wrong method 405")
st, ct, body = req("GET", "/nope")
check(st == 404, "unknown route 404")
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
c.request("POST", "/v1/chat/completions", body="{nope", headers={"Content-Type": "application/json"})
r = c.getresponse()
print("   malformed body:", r.status, r.read().decode()[:120])
check(r.status == 400, "malformed JSON 400")
c.close()
st, ct, body = req(
    "POST",
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}
        ],
    },
)
check(st == 400 and "data:" in json.loads(body)["error"]["message"], "http image URL refused")
st, ct, body = req(
    "POST",
    "/v1/chat/completions",
    {"model": MODEL, "messages": [{"role": "user", "content": "x"}], "n": 3},
)
check(st == 400, "n>1 refused with 400")

print("== keep-alive: two requests on one connection ==")
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
for i in range(3):
    c.request(
        "GET",
        "/health",
        headers={"Content-Type": "application/json", "Connection": "keep-alive"},
    )
    rr = c.getresponse()
    rr.read()
    check(rr.status == 200, f"keep-alive request {i + 1}")
c.close()

print()
print("FAIL" if fails else "PASS", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
