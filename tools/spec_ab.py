"""A/B of the speculative decode path against plain greedy decoding.

    QE_PORT=8123 python3 tools/spec_ab.py                 # shipping arithmetic
    QE_PORT=8123 QE_SPEC_MODE=exact python3 tools/spec_ab.py

Needs a loaded checkpoint; `make spec` starts one and stops it again.

**Two modes, because two claims are being checked and only one of them can be
exactness.** A draft window is a chunk, and a chunk walks the dense projections
with the shared int8 walk (`qe_forward_set_proj_i8`, `qe_forward_set_flat_i8`)
while a plain decode step is a `T == 1` pass over the exact dequantised rows. That
is the same change of number, not of instruction, that ingestion already ships, and
it means a windowed token and a single token are computed at two fidelities of one
model. Greedy decoding is chaotic, so the two trajectories part within a couple of
tokens even when every row of every window is verified and every state put back
right. `make spec` therefore runs this file twice against two servers:

  `exact` - a server started with the ingestion fast paths switched off (`QE_CHUNK=1`
      and the grouped, chunked and int8 walks set to a real zero, which `make spec`
      does on the server's own environment), where a window's rows are walked with one
      weight pass per row and the arithmetic is the `T == 1` arithmetic, token for
      token. Here a draft is an optimisation and never a change of answer, so the two
      runs are compared byte for byte, and this is the test that fails when a rewind
      puts back the wrong state. `make mtp` proves the rule against a toy target; this
      proves it against the real network, where the thing that can be wrong is the
      state a window leaves behind - the delta-net state checkpointed per row, the two
      conv histories that are minimal rings rather than position-addressed caches, the
      head re-taken for the row a window ended on.

  `fast` - a server with the defaults, where fidelity differs and output identity
      therefore cannot be the assertion. Quality is asserted instead, three ways: the
      copying task has a ground truth that does not depend on the other run (the lines
      it was told to copy), an answer must not be degenerate, and acceptance must
      actually be happening - `accepted > 0` and more than one token per verification,
      which is the whole reason the path exists.

**Everything is asked through the chat endpoint with reasoning turned off.** A raw
completion prompt on a checkpoint tuned for turns sends the model into a thinking
block, and inside a thinking block the next token is new prose: an index of the
n-grams already in the sequence has nothing correct to propose there, acceptance sits
at zero, and what that measures is the prompt and not the decoder. With reasoning off
the two task shapes are the whole story - text the model was told to reproduce, where
a window pays, and new prose, where it does not. A loss on prose is the expected
answer and is printed as one.

**Speed** is reported from the fast server as the ratio of tokens per verification
and of tok/s, from runs that each paid for their own prompt: a request whose prompt
the turn cache recognises is answered from a restored boundary, and a draft window
cannot start from one of those (it has to hold one token back), so a repeated
prompt quietly runs the plain loop. Every prompt in the speed table is therefore
seeded differently, and `cached_tokens == 0` is asserted rather than assumed.

**Counting.** The engine's `target_passes` is the number of positions it put through
the network, and a window puts one through for every token it was asked about, kept or
not, so its tokens-per-pass is bounded by one. Tokens per *verification* is the number
that says something, and it needs no new counter: every window emits the one token the
target itself chose plus the drafts it kept, so the windows a request ran are its
tokens minus its accepted drafts.
"""
import http.client
import json
import os
import re
import sys
import time

PORT = int(os.environ.get("QE_PORT", "8123"))
MODE = os.environ.get("QE_SPEC_MODE", "fast")
QUICK = "--quick" in sys.argv
CACHE = True  # whether this server was opened with a turn cache; main() reads it

fails = []


def check(ok, what, detail=None):
    print(("  ok   " if ok else "  FAIL ") + what + ("" if detail is None else ": " + str(detail)))
    if not ok:
        fails.append(what)


MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")


def say(*xs):
    """Model text never reaches a log with its control tokens spelled out."""
    return " ".join(
        MARKER.sub(lambda m: "{" + m.group(0)[2:-2].strip() + "}", str(x)) for x in xs
    )


def post(path, body=None, method="GET"):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
    hdr = {} if method == "GET" else {"Content-Type": "application/json"}
    c.request(
        method,
        path,
        body=None if method == "GET" else json.dumps(body),
        headers=hdr,
    )
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    status = r.status
    c.close()
    return status, data


def run(messages, spec_k, max_tokens, temperature=0):
    """One greedy chat request, with the measured block it reported."""
    body = {
        "model": "qe",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "enable_thinking": False,
        "qe_spec_k": spec_k,
        "messages": messages,
    }
    t0 = time.time()
    status, data = post("/v1/chat/completions", body, method="POST")
    wall = time.time() - t0
    if status != 200:
        print("   ", say(status, data[:400]))
        fails.append("chat request refused")
        return None
    j = json.loads(data)
    m = (j.get("qe") or {}).get("measured") or {}
    text = j["choices"][0]["message"]["content"]
    tokens = m.get("tokens", 0)
    accepted = m.get("accepted", 0)
    windows = tokens - accepted
    return {
        "text": text,
        "tokens": tokens,
        "passes": m.get("target_passes", 0),
        "drafted": m.get("drafted", 0),
        "accepted": accepted,
        "tps": m.get("tok_per_s", 0.0),
        "tok_per_window": (tokens / float(windows)) if windows > 0 else 0.0,
        "windows": windows,
        "cached": m.get("cached_tokens", 0),
        "spec_k": m.get("spec_k", -1),
        "wall": wall,
    }


def line(tag, r):
    if not r:
        print("    %-20s refused" % tag)
        return
    rate = 100.0 * r["accepted"] / r["drafted"] if r["drafted"] else 0.0
    print(
        "    %-20s %3d tok in %5.1f s = %5.2f tok/s | %5.2f tok/verify %4d rows | "
        "drafted %4d accepted %4d (%.0f%%) | cached %d"
        % (tag, r["tokens"], r["wall"], r["tps"], r["tok_per_window"], r["passes"],
           r["drafted"], r["accepted"], rate, r["cached"])
    )


def geo(vals):
    if not vals:
        return 0.0
    p = 1.0
    for v in vals:
        p *= v
    return p ** (1.0 / len(vals))


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def degenerate(text):
    """An answer that repeats itself is not an answer.

    A rewind that put back the wrong state was seen to produce a newline and then
    the same three words for the rest of the turn, which is what this is for. It is
    deliberately about *distinct 4-grams of an answer*, so it belongs on the answers
    that were not asked to repeat anything - the copying task fails it by design."""
    ws = text.split()
    if len(ws) < 16:
        return False
    grams = set()
    for i in range(len(ws) - 3):
        grams.add(tuple(ws[i : i + 3]))
    return len(grams) < 0.5 * (len(ws) - 3)


# --------------------------------------------------------------- the prompts
#
# One shape the drafter can win on and one it cannot. Both are chat turns, because
# that is the distribution this checkpoint was tuned for, and both are seeded so that
# a speed run can be asked twice over prompts of the same shape that share no suffix,
# which is what keeps the turn cache out of the measurement.

LINES = [
    "cache_dir = /var/lib/{seed}/cache",
    "listen = 10.0.{a}.{b}:{port}",
    "workers = {w}",
    "timeout_ms = {to}",
    "upstream = registry.{seed}.internal:5443",
    "log_level = {lvl}",
]

COPY_SYSTEM = (
    "Copy the block the user gives you, one line at a time, in the same order, "
    "exactly as written. Add nothing else."
)

PROSE_SYSTEM = "Answer in two or three concrete sentences. Do not repeat the question."

COPY = "Copy these lines:\n\n{block}\n\nNow copy them again."
COPY_ONCE = "Copy these lines:\n\n{block}"

PROSE = [
    "Why does a warehouse that has just installed its first conveyor belt still "
    "lose most of its time at the packing stations? ({seed})",
]


def block_for(seed):
    """The block to copy. Every line is seed-specific so that two runs over
    different seeds have nothing to reuse, and the fields vary in length so the
    answer is not one fixed string repeated. Deterministic on purpose: a run that
    could not reproduce its own prompt could not be compared against anything."""
    n = sum(ord(c) for c in seed) % 97
    return "\n".join(
        f.format(seed=seed, a=10 + n % 7, b=1 + n % 29, port=1024 + n, w=4 + n % 13,
                 to=250 + 25 * (n % 11), lvl=["warn", "info", "debug"][n % 3])
        for f in LINES
    )


def copy_messages(seed, twice=True):
    return [
        {"role": "system", "content": COPY_SYSTEM},
        {"role": "user", "content": (COPY if twice else COPY_ONCE).format(
            block=block_for(seed))},
    ]


def prose_messages(seed):
    return [
        {"role": "system", "content": PROSE_SYSTEM},
        {"role": "user", "content": PROSE[0].format(seed=seed)},
    ]


def copied(answer, seed):
    """How much of the answer is the block it was told to copy.

    The quality check that does not need the other run: the prompt says what the
    right answer is, so an answer that lost its state can be shown to be wrong
    rather than merely different. Fractions of the block's lines that come back
    verbatim, in order."""
    want = block_for(seed).split("\n")
    got = answer.split("\n")
    if not got:
        return 0.0
    hits = 0
    at = 0
    for w in want:
        for i in range(at, len(got)):
            if got[i].strip() == w:
                hits += 1
                at = i + 1
                break
    return hits / float(len(want))


def rates(tag, a, b):
    """The rate line, printed only when the two legs did the same amount of work.

    A leg answered from the turn cache walks a fraction of the prompt, and a draft
    window cannot start from a full-prompt boundary at all - the engine falls back to
    the plain loop when the whole prompt was reused - so on a server with the cache on
    the second leg of every pair is the warm one. Printing "7.7 against 9.5 tok/s"
    about that measures the cache and calls it speculation, so this names the warm leg
    instead of ranking the two."""
    if not a or not b:
        return
    warm = [(t, r["cached"]) for t, r in (("spec", a), ("plain", b)) if r["cached"]]
    if warm:
        print("    %s: %.2f against %.2f tok/s, with %s answered from the turn cache"
              " (%d of its tokens) - not the same work, so no ratio"
              % (tag, a["tps"], b["tps"], warm[0][0], warm[0][1]))
        return
    print("    %s: %.2f tok/s speculatively against %.2f plain (%.2fx)"
          % (tag, a["tps"], b["tps"], a["tps"] / max(b["tps"], 1e-9)))


def agree(tag, a, b):
    """Byte-for-byte in `exact` mode.

    In fast mode the two sides are the same model at two fidelities and greedy
    parts at a near-tie, so where they part is reported and not asserted - what is
    asserted about the fast pair is that each answer is a good answer, which is
    what the caller's own checks are for. A shared opening would be a fine
    assertion right up until the first position where two float32 sums rank a tie
    differently, and that position is not knowable from here."""
    if not a or not b:
        check(False, tag + ": a run failed")
        return
    n = common_prefix(a["text"], b["text"])
    if MODE == "exact":
        check(a["text"] == b["text"], tag + ": the answer is byte for byte the same",
              say("%r != %r" % (a["text"][:60], b["text"][:60]))
              if a["text"] != b["text"] else None)
        return
    print("    %s: the two fidelities part after %d of %d chars" % (tag, n, len(a["text"])))


def main():
    st, _ = post("/health", method="GET")
    if st != 200:
        print("no server on :%d (%s) - this needs a loaded checkpoint" % (PORT, st))
        return 2
    # The engine facts a client plans with are on /v1/models; /health only says the
    # listener is up. Whether this server has a turn cache decides which of the
    # checks below can run at all, so it is read from where it is actually reported.
    st2, body = post("/v1/models", method="GET")
    blob = json.loads(body) if st2 == 200 else {}
    # `/v1/models` is an OpenAI list, and the engine facts are the `qe` object on the
    # one model entry - not the envelope, and not the entry itself. Reading them off
    # either of those answers every question with "absent", and "absent" made this
    # file believe a server with a turn cache had none, which turned a warm leg into
    # an assertion that it had started cold.
    data = blob.get("data") or []
    entry = data[0] if data and isinstance(data[0], dict) else blob
    info = entry.get("qe") if isinstance(entry.get("qe"), dict) else entry
    global CACHE
    CACHE = (info.get("turn_cache_budget_mib") or 0) > 0
    print("== %s on :%d (%s arithmetic, turn cache %s, spec_k %s) =="
          % (os.path.basename(__file__), PORT, MODE,
             "on" if CACHE else "off", info.get("spec_k")))

    n_id = 0

    # --- 1. the state a window leaves behind, on text to reproduce ---------
    #
    # The pair is run spec-first. On a server with the turn cache on the second leg
    # then recognises the prompt and is answered from a restored boundary, which is
    # a second invariant for the price of one but not the one that matters: both
    # legs would then start from the checkpoint, and a checkpoint that does not hold
    # the true state agrees with itself perfectly. `make spec` therefore runs this
    # file against a server started with `--turn-cache -1` as well, where both legs
    # start at position 0 and a rewind that restores the wrong thing shows up as two
    # answers that part.
    print("== copying a block (the shape a draft wins on) ==")
    a = run(copy_messages("crates"), 4, 96)
    line("spec_k=4", a)
    b = run(copy_messages("crates"), 0, 96)
    line("spec_k=0 (%s)" % ("warm" if CACHE else "cold"), b)
    if not CACHE:
        check(a and b and not a["cached"] and not b["cached"],
              "both legs started at position 0", [a and a["cached"], b and b["cached"]])
    else:
        # The other invariant, and the one only a cache-on server can show: the plain
        # leg of the pair is a second request over the same conversation, so it walks
        # the boundary the speculative leg left at row 0 of its first window - the
        # state after the last token of the prompt, taken from a pass that then walked
        # past it. A boundary a row short shows up here as the second turn continuing
        # from the wrong place, and it does not.
        check(a and not a["cached"] and b and b["cached"] > 0,
              "the second request continued from the boundary a draft window stored",
              [a and a["cached"], b and b["cached"]])
    check(a and a["drafted"] > 0, "the drafter had something to propose",
          a and a["drafted"])
    check(a and a["accepted"] > 0, "the target kept part of what it was shown",
          a and "%d of %d drafted" % (a["accepted"], a["drafted"]))
    if MODE == "fast" and a:
        check(a["tok_per_window"] > 1.0, "a window decides more than one token a pass",
              round(a["tok_per_window"], 3))
    for r, tag in ((a, "spec"), (b, "plain")):
        s = copied(r["text"], "crates") if r else 0.0
        check(r and s >= 0.8, "%s copied the block it was given" % tag,
              "%.2f of its lines came back: %s" % (s, say(r["text"][:90])) if r else None)
    agree("copying", a, b)
    if a and b:
        n_id += 1

    # --- 2. speed, on prompts nobody has fed before ------------------------
    #
    # Same shape, different seed word, alternating which side runs first so neither
    # owns the warm core of the machine.
    print("== speed on a copy task (each prompt used once) ==")
    plain, spec, seen = [], [], []
    seeds = ["tokio", "serde", "axum"] if not QUICK else ["tokio", "serde"]
    for i, seed in enumerate(seeds):
        side = i % 2
        r = run(copy_messages(seed), 0 if side else 4, 128)
        line("%s spec_k=%d" % (seed, 0 if side else 4), r)
        if not r:
            continue
        (plain if side else spec).append(r)
        seen.append((r, seed))
    cold = [r for r, _ in seen if r["cached"] == 0]
    check(len(cold) == len(seen), "no run was answered from the turn cache",
          [r["cached"] for r, _ in seen])
    off = [(s, round(copied(r["text"], s), 2)) for r, s in seen if copied(r["text"], s) < 0.8]
    check(not off, "and every one of them copied the block it was given", off or None)
    if plain and spec:
        tps = geo([r["tps"] for r in spec]) / max(geo([r["tps"] for r in plain]), 1e-9)
        tps_step = geo([r["tok_per_window"] for r in spec]) / max(
            geo([r["tok_per_window"] for r in plain]), 1e-9)
        print("    ratio on copying: %.2fx tok/s, %.2fx tokens per verification"
              % (tps, tps_step))
        check(tps_step > 1.0, "a window decides more than one token a pass",
              round(tps_step, 3))
        if MODE == "fast":
            check(tps > 1.0, "and is faster for it", round(tps, 3))

    # --- 2b. how far ahead to draft ---------------------------------------
    #
    # A window of T rows costs less than T steps in proportion to how wide it is -
    # the engine reads 48 ms a row at five against 105 for a step, because a row of a
    # window shares the walk of the weights - so the question the server's default
    # answers is not whether to draft but how far ahead to draft it. Each width gets
    # its own seed, because a run that answered from the boundary the width before it
    # stored would be measuring the cache.
    print("== how far ahead to draft (fresh prompt per width) ==")
    if MODE == "fast" and not QUICK:
        base = None
        for k in (0, 2, 4, 6, 8, 12):
            r = run(copy_messages("width%d" % k), k, 128)
            line("spec_k=%-2d" % k, r)
            if not r:
                continue
            if not k:
                base = r["tps"]
            elif base:
                print("      %.2fx plain, %d of %d drafted kept"
                      % (r["tps"] / base, r["accepted"], r["drafted"]))
    else:
        print("    skipped: %s" % ("--quick" if QUICK
                                   else "the exact leg is the identity test, not the rate test"))

    # --- 3. new prose, where a draft has nothing to say --------------------
    #
    # The honest number. An index of what has been seen has nothing to say about a
    # question that has not been asked, so this is where a window is paid for and
    # not paid back, and the ratio is expected to be at or under 1.
    print("== new prose ==")
    m = prose_messages("Topic 11.")
    a = run(m, 4, 64)
    line("spec_k=4", a)
    b = run(m, 0, 64)
    line("spec_k=0 (warm)", b)
    for r, tag in ((a, "spec"), (b, "plain")):
        check(r and not degenerate(r["text"]), "%s wrote something that goes somewhere" % tag,
              say(r["text"][:90]) if r else None)
    agree("prose", a, b)
    if a and b:
        rates("prose", a, b)
        n_id += 1

    # --- 4. a request that ends on the token that closes the turn ----------
    #
    # The only turn shape that reliably reaches the marker that ends a turn, and
    # that token is reported and then not kept, which is the one place the loop's
    # boundary could leave the pass standing a row too far and hand the next request
    # a frontier distribution from the wrong position.
    print("== identity when the answer ends a turn ==")
    short = [
        {"role": "system", "content": "Answer with exactly two short sentences."},
        {"role": "user", "content": "What is the difference between a cache and a buffer? (warm)"},
    ]
    a = run(short, 4, 48)
    line("chat spec_k=4", a)
    b = run(short, 0, 48)
    line("chat spec_k=0", b)
    agree("turn end", a, b)
    if a and b:
        rates("turn end", a, b)
        n_id += 1

    # --- 5. one token, which is a window of one row ------------------------
    #
    # max_tokens 1 asks for a single distribution and no continuation, so the
    # request ends before a window can pay for itself and the tail the turn cache
    # wants has to come from the row that was never fed.
    print("== a one-token request ==")
    m = copy_messages("one")
    a = run(m, 4, 1)
    line("spec_k=4", a)
    b = run(m, 0, 1)
    line("spec_k=0", b)
    if a and b:
        check(a["text"] == b["text"], "the first token is the one a step decides",
              say(repr(a["text"])) if a["text"] == b["text"]
              else "%r against %r" % (say(a["text"]), say(b["text"])))
        rates("one token", a, b)
        n_id += 1

    # --- 6. the turn cache cannot tell the two apart -----------------------
    #
    # A second request over the same conversation is the one that reads the
    # boundary the first one stored. A speculative request stores that boundary at
    # the row that earned the prompt's last logits rather than at the end of the
    # request, because that is the only moment the state after the prompt exists,
    # so this is also the check that the boundary was taken at all.
    print("== the boundary a speculative request leaves behind ==")
    conv = copy_messages("twice") + [
        {"role": "assistant", "content": block_for("twice")},
        {"role": "user", "content": "Copy them once more."},
    ]
    a = run(conv, 4, 32)
    line("turn 1 spec_k=4", a)
    if not CACHE:
        print("    skipped: this server has no turn cache to check (--turn-cache -1)")
    else:
        again = run(conv, 0, 8)
        line("turn 2 spec_k=0", again)
        check(again and again["cached"] > 0, "the second request reused the first",
              again and again["cached"])
        third = run(conv, 4, 8)
        line("turn 3 spec_k=4", third)
        check(third and third["cached"] > 0, "and a speculative turn reuses it as well",
              third and third["cached"])
        n_id += 1

    print()
    print("%d quality checks, mode %s" % (n_id, MODE))
    if fails:
        print("FAIL (%d): %s" % (len(fails), ", ".join(fails)))
        return 1
    print("PASS (0 failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
