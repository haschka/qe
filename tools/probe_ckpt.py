"""Serving the real checkpoint: what it answers, and whether it means it.

    CHECKPOINT=/path/to/volume1.gguf make probe            # one load, ~12 min
    make probe-selftest                                     # same checks, no weights

`make http` covers the HTTP surface against the selftest engine, which generates
through the speculation loop and therefore applies no sampling filters at all.
That leaves the two claims this file exists to check unasserted anywhere: that a
checkpoint engine generates instead of refusing, and that on that path the four
sampler knobs actually change the draw. Both need real logits, so both need the
87 GiB, so this is a tool and not part of `make test`.

Nothing is timed. Greedy text is compared to itself, and a filter is only called
effective when the unfloored draw differs from it, so a server that answered
errors would fail here instead of passing six vacuous equality checks - which is
the bug this file had once already.

Two ways to run it without paying for the load:

  QE_SELFTEST=1  serves the selftest engine instead of a checkpoint. The checks
                 that need real logits are skipped (speculation applies no
                 filters); the ones that are about the HTTP layer run for real.
  QE_EXISTING=1  assumes a server is already listening on $QE_PORT and connects
                 to it, starting nothing.

`QE_SELFTEST=1` is how this file tests itself: it is a client too, and a client
bug costs a twelve minute load to discover. The wait loop therefore retries only
on "nothing to connect to yet" and lets any other exception out immediately.

Model text is never written out as it came in. A vocabulary marker that reaches a
terminal or a log is a control token one copy-paste away from being live, and this
repository keeps none of them in text anywhere, so every byte of model output goes
through `say()` first and comes back with its markers named instead of spelled.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("QE_PORT", "8099"))
BASE = "http://127.0.0.1:%d" % PORT
# Unset means "ask the checkpoint what it was trained for", which is the default
# worth exercising; CTX=4096 pins it for a memory-constrained run.
CTX = os.environ.get("CTX", "")
SELFTEST = os.environ.get("QE_SELFTEST", "") == "1"
EXISTING = os.environ.get("QE_EXISTING", "") == "1"
# A checkpoint load is minutes; a selftest engine is milliseconds.
WAIT_S = int(os.environ.get("QE_WAIT_S", "120" if SELFTEST else "1500"))
PROMPT = "Reply with one short sentence about memory bandwidth."

failures = []
served = {"id": "qe"}
measured = {}

# A control token, written without writing one. Raw string on purpose: an escaped
# pipe is a literal, an unescaped one is alternation, and the first version of this
# line matched every 48-character run of ordinary text instead.
MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")


def say(s):
    """Model text, safe to print: a marker comes back as {its-name} rather than as
    the sequence itself, so nothing this tool prints can be pasted forward."""
    return MARKER.sub(lambda m: "{" + m.group(0)[2:-2].strip() + "}", s)


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, say(str(detail))), flush=True)
    if not ok:
        failures.append(name)


def parse_strict(raw):
    """A JSON body is UTF-8 or the client rejects the whole thing, so a reply that
    stopped inside a three-byte character is a protocol bug and not a nuisance."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError("body is not valid UTF-8: %s (%d bytes)" % (e, len(raw)))
    return json.loads(text)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=600) as r:
        return parse_strict(r.read())


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, parse_strict(r.read())
    except urllib.error.HTTPError as e:
        return e.code, parse_strict(e.read())


def channels(content, reasoning):
    """A reply as two channels, not one string. This is a thinking model: the
    first words land in `reasoning_content`, and `content` can legitimately be
    empty, or be the whitespace between the block and the answer. Keeping the two
    apart is what makes "the stream is the answer" an assertion rather than a
    comparison of two different concatenations of the same parts."""
    return (content or "", reasoning or "")


def chat(**kw):
    """One chat turn. Raises if the server did not answer with a message, so a
    failure can never be counted as an empty-but-passing answer."""
    body = {"model": served["id"], "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 48}
    body.update(kw)
    st, j = post("/v1/chat/completions", body)
    if st != 200:
        raise RuntimeError("HTTP %d %s" % (st, json.dumps(j)[:160]))
    m = j["choices"][0]["message"]
    ext = j.get("qe")
    if isinstance(ext, dict) and isinstance(ext.get("measured"), dict):
        measured.clear()
        measured.update(ext["measured"])
    return channels(m.get("content"), m.get("reasoning_content"))


def chat_stream(**kw):
    """The same turn as SSE, reassembled the way a client does it: two channels,
    each in the order its deltas arrived."""
    body = {"model": served["id"], "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 48, "stream": True}
    body.update(kw)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read().decode("utf-8", "replace")
    out = {"content": "", "reasoning_content": ""}
    for line in raw.split("\n"):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])["choices"][0]["delta"]
        for k in out:
            out[k] += d.get(k) or ""
    return channels(out["content"], out["reasoning_content"])


def words(c):
    return len((c[0] + " " + c[1]).split())


def clip(c):
    return say("%r + %r" % (c[0][:34], c[1][:34]))


def wait_for_listener(srv):
    """Poll /health until the listener answers. Only 'nobody is home yet' is a
    reason to retry: this load takes minutes, and a mistake in this file is not
    a slow load, so anything else raises and prints while there is still time."""
    started = time.time()
    next_note = 0.0
    while time.time() - started < WAIT_S:
        if srv is not None and srv.poll() is not None:
            print("server exited early rc=%d, see /tmp/qe-probe-server.log" % srv.returncode,
                  flush=True)
            return False
        try:
            get("/health")
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            why = "%s: %s" % (type(e).__name__, getattr(e, "reason", e))
            if time.time() >= next_note:
                print("     waiting for the listener (%.0f s), %s" % (time.time() - started, why),
                      flush=True)
                next_note = time.time() + 60
            time.sleep(2)
    return False


def main():
    if not (SELFTEST or EXISTING) and len(sys.argv) < 2:
        print("usage: CHECKPOINT=... tools/probe_ckpt.py PATH/TO/volume1.gguf", file=sys.stderr)
        return 2

    srv = None
    if EXISTING:
        print("connecting to an already running server on :%d" % PORT, flush=True)
    else:
        argv = ["./bin/qe-serve", "--port", str(PORT)]
        if SELFTEST:
            argv += ["--threads", "8", "--id", "qe-smoke"]
        else:
            argv += ["--model", sys.argv[1]]
            if CTX:
                argv += ["--ctx", CTX]
        srv = subprocess.Popen(argv, stdout=open("/tmp/qe-probe-server.log", "w"),
                               stderr=subprocess.STDOUT)
    try:
        check("the listener answers /health" if (SELFTEST or EXISTING)
              else "the listener opens only after the weights are resident",
              wait_for_listener(srv), "waited %d s" % WAIT_S)
        if failures:
            return 1

        models = get("/v1/models")
        blk = models["data"][0]
        served["id"] = blk.get("id", "qe")
        qe = blk.get("qe", {})
        if SELFTEST:
            check("the served model is the selftest target, which is not a model",
                  qe.get("engine") == "selftest" and qe.get("can_generate") is True,
                  "id=%s engine=%s spec_k=%s" % (served["id"], qe.get("engine"),
                                                  qe.get("spec_k")))
        else:
            check("the served model is a checkpoint that says it generates",
                  bool(qe.get("weights_resident")) and qe.get("can_generate") is True,
                  "id=%s stage=%s kv=%s ctx=%s/%s trained"
                  % (served["id"], qe.get("weight_stage"), qe.get("kv_cache"),
                     qe.get("context"), qe.get("context_train")))

        greedy = chat(temperature=0)
        again = chat(temperature=0)
        check("greedy answers with a sentence", words(greedy) >= 3, clip(greedy))
        check("greedy is reproducible", greedy == again, clip(again))
        streamed = chat_stream(temperature=0)
        check("the buffered answer is exactly the streamed deltas, each once",
              streamed == greedy, "%s vs %s" % (clip(streamed), clip(greedy)))
        if measured:
            print("     %d tokens over %s target passes (spec_k %s) in %.6f s"
                  " = %.2f tok/s, %.1f passes/s"
                  % (int(measured.get("tokens", 0)), measured.get("target_passes", "?"),
                     measured.get("spec_k", "?"), float(measured.get("seconds", 0.0)),
                     float(measured.get("tok_per_s", 0.0)),
                     float(measured.get("passes_per_s", 0.0))), flush=True)

        for bad, knob in [({"top_p": 1.5}, "top_p"), ({"top_p": -0.1}, "top_p"),
                          ({"min_p": 1.0}, "min_p"), ({"min_p": -0.1}, "min_p"),
                          ({"top_k": -2}, "top_k"), ({"top_k": 1.5}, "top_k"),
                          ({"top_k": "auto"}, "top_k"), ({"temperature": -1.0}, "temperature")]:
            body = {"model": served["id"],
                    "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
            body.update(bad)
            st, j = post("/v1/chat/completions", body)
            msg = json.dumps(j)
            check("%s %s is a 400 naming the knob" % (knob, json.dumps(bad)[1:-1]),
                  st == 400 and knob in msg, "HTTP %d %s" % (st, say(msg[:80])))

        if SELFTEST:
            print("\n  selftest engine: the four sampler knobs are applied on the "
                  "checkpoint path only, so their semantics are not asserted here",
                  flush=True)
            print("  greedy %s" % clip(greedy), flush=True)
        else:
            topk1 = chat(temperature=2.0, top_k=1, seed=11)
            minp = chat(temperature=2.0, min_p=0.999, seed=11)
            hot = chat(temperature=2.0, seed=11)
            hot2 = chat(temperature=2.0, seed=12)
            check("top_k 1 at temperature 2 is greedy", topk1 == greedy,
                  "%s vs %s" % (clip(topk1), clip(greedy)))
            check("min_p at the argmax is greedy", minp == greedy,
                  "%s vs %s" % (clip(minp), clip(greedy)))
            check("temperature 2 is not greedy", hot != greedy, clip(hot))
            check("a different seed draws different text", hot != hot2,
                  "%s / %s" % (clip(hot), clip(hot2)))

            base = chat(temperature=2.0, seed=5)
            # Absent is not "off". A knob the client did not send takes the
            # value the checkpoint ships - top_k 20, top_p 0.95, min_p 0.000,
            # which `make report` prints - so the invariant worth testing is
            # that naming the file's own number changes nothing. top_p 1.0
            # means off, which is a different request from an unnamed one, and
            # asserting the two were equal is what made this line fail on a
            # healthy engine.
            check("min_p absent is the file's 0.000",
                  base == chat(temperature=2.0, seed=5, min_p=0.0), clip(base))
            check("top_k absent is the file's 20",
                  base == chat(temperature=2.0, seed=5, top_k=20), clip(base))
            check("top_p absent is the file's 0.95",
                  base == chat(temperature=2.0, seed=5, top_p=0.95), clip(base))
            check("top_k 0 and top_k -1 both mean off",
                  chat(temperature=2.0, seed=5, top_k=0)
                  == chat(temperature=2.0, seed=5, top_k=-1), clip(base))

            print("\n  greedy %s" % clip(greedy), flush=True)
            print("  hot    %s" % clip(hot), flush=True)
    except Exception as e:
        check("probe ran to completion", False, "%r" % (e,))
    finally:
        if srv is not None:
            srv.terminate()
            try:
                srv.wait(timeout=120)
            except Exception:
                srv.kill()

    print("\n== %s (%d failures) ==" % ("PASS" if not failures else "FAIL", len(failures)),
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
