"""The client this server exists to serve, run for real, end to end.

    make probe-crush                                # selftest engine, no weights
    CHECKPOINT=/path/volume1.gguf make probe-crush  # the checkpoint: minutes

`make http` asserts the request surface by constructing it by hand. What it cannot
assert is the other half of the contract: that a real agent loop accepts the
stream, asks for a tool, runs the tool, and comes back with the result. That
needs the client, and the client is charmbracelet crush.

So this starts qe-serve, puts a recording proxy in front of it, points a throwaway
crush at the proxy, asks crush to write a file with its write tool, and then reads
the wire back. Nine things have to be true for it to pass:

  1. crush sent `tools` and got a stream it accepted,
  2. every streamed chat reply ended with a `finish_reason` and `data: [DONE]`
     (its SDK calls a stream without a reason incomplete and retries four times),
  3. a `delta.tool_calls` arrived with the opening fields the SDK needs,
  4. the next request carried the result back as `role: "tool"`,
  5. the file the model asked for is on disk with the bytes it asked for,
  6. every `tool_calls` delta carried the `index` its arguments accumulate under,
  7. no two calls were handed the same id, which is what a client looks a call up
     by for the length of a conversation: an id handed out twice draws the second
     call over the first one's interface block and files both results as one,
  8. the transcript a later request carried back named each call once,
  9. crush exited successfully.

The proxy is not a convenience: crush gives no account of what it sent, and every
one of those claims is about bytes on the wire.

`QE_SELFTEST=1` runs the same loop against the selftest engine, which streams and
terminates for real but cannot emit a tool request, because its generator does not
produce this model's control tokens. Checks 3, 4 and 5 are reported as skipped
there, never as passed. `QE_EXISTING=1` points at a server that is already up,
which is how this runs against a checkpoint that took twelve minutes to load.

Nothing the model wrote is printed as it arrived: a vocabulary marker in a terminal
is a control token one copy-paste from being live, so all of it goes through
`say()`.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

PORT = int(os.environ.get("QE_PORT", "8097"))
PROXY_PORT = int(os.environ.get("QE_PROXY_PORT", str(PORT + 1)))
SELFTEST = os.environ.get("QE_SELFTEST", "") == "1"
EXISTING = os.environ.get("QE_EXISTING", "") == "1"
MODEL_ID = "qe"
CTX = int(os.environ.get("CRUSH_CTX", "200000"))
MAX_TOKENS = int(os.environ.get("CRUSH_MAX_TOKENS", "4096"))
WAIT_S = int(os.environ.get("QE_WAIT_S", "120" if SELFTEST else "1500"))
RUN_S = int(os.environ.get("CRUSH_RUN_S", "300" if SELFTEST else "3600"))
# How long the recorder will wait for the next bytes of a reply it is copying.
READ_S = float(os.environ.get("QE_PROXY_READ_S", "3600"))
# crush's own stream idle deadline for the session under test. Its default is 2
# minutes, and it is a deadline of total silence: an eleven-thousand-token agent
# prompt ingests for 862 s before its first delta and aborts at 120 s. That is why
# this harness sets it high, and why an operator whose config does not has a server
# that looks broken while it is working.
REQUEST_TIMEOUT = int(os.environ.get("QE_PROBE_REQUEST_TIMEOUT", "3600"))
# Learn the model from /v1/models instead of naming it: the provider is declared
# `llamacpp`, which is the one crush asks for a context window, and no
# context_window is written anywhere. This is the config the operator does not
# have to hand-tune, and it only works if /v1/models says the window in that
# shape.
DISCOVER = os.environ.get("QE_PROBE_DISCOVER", "") == "1"
PROMPT = ("Use your write tool to create a file named note.txt in the current "
          "directory containing exactly the word HELLO. Then tell me the word.")
CAP = 4_000_000

failures = []
skipped = []
counts = {"checks": 0}

# Raw string on purpose: an escaped pipe is a literal, an unescaped one is
# alternation, and the first version of this line matched ordinary text.
MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")


def say(s):
    """Model text, safe to print: a marker comes back as {its-name} instead of as
    the sequence itself, so nothing printed here can be pasted forward."""
    return MARKER.sub(lambda m: "{" + m.group(0)[2:-2].strip() + "}", str(s))


def check(name, ok, detail=""):
    counts["checks"] += 1
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, say(detail)), flush=True)
    if not ok:
        failures.append(name)


def skip(name, why):
    print("%-4s %s %s" % ("SKIP", name, say(why)), flush=True)
    skipped.append(name)


def head_and_body(raw):
    head, sep, rest = raw.partition(b"\r\n\r\n")
    return (head, rest) if sep else (b"", raw)


def sse_stream(raw):
    """The payloads of one recorded response, or None when that response was not
    a stream. The distinction matters: a 4xx is a real answer with no events, and
    counting it as an empty stream would make "the stream terminated" true by
    accident."""
    head, rest = head_and_body(raw)
    if b"text/event-stream" not in head:
        return None
    return [ln[6:] for ln in rest.split(b"\n") if ln.startswith(b"data:")]


def payloads(responses):
    for raw in responses:
        s = sse_stream(raw)
        if s is not None:
            yield s


def parse(payload):
    if payload.strip() == b"[DONE]":
        return "[DONE]"
    try:
        return json.loads(payload.decode("utf-8", "replace"))
    except ValueError:
        return {"_bad": payload[:120]}


def verdicts(responses):
    """Per streamed reply: the reason it ended, whether it said [DONE], and its
    events in order. Everything the checks ask about is per reply, so keeping the
    replies apart is the whole job."""
    for payloads_ in payloads(responses):
        events = [parse(p) for p in payloads_]
        reasons = [e.get("choices", [{}])[0].get("finish_reason")
                   for e in events if isinstance(e, dict) and e.get("choices")]
        yield {
            "events": events,
            "reason": next((r for r in reversed(reasons) if r), None),
            "done": events and events[-1] == "[DONE]",
        }


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []
        self.responses = []

    def request(self, body):
        with self.lock:
            self.requests.append(body)

    def response(self, body):
        with self.lock:
            self.responses.append(body[:CAP])

    def snapshot(self):
        with self.lock:
            return list(self.requests), list(self.responses)


def request_of(stream):
    """Read one request far enough to know what it asked for. Headers are parsed
    and the body is read to Content-Length; nothing else about HTTP is honoured -
    this is a recorder in front of a loopback server, not a proxy that has to win
    an argument with a browser."""
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = stream.recv(4096)
        if not chunk:
            return None, b""
        head += chunk
    pre, rest = head.split(b"\r\n\r\n", 1)
    length = 0
    for line in pre.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
    while len(rest) < length:
        chunk = stream.recv(65536)
        if not chunk:
            break
        rest += chunk
    return pre, rest


def forward(client, up, rec):
    with client, up:
        head, body = request_of(client)
        if head is None:
            return
        try:
            rec.request(json.loads(body.decode("utf-8", "replace")))
        except ValueError:
            rec.request({"_unparsed": len(body)})
        up.sendall(head + b"\r\n\r\n" + body)
        out = []
        try:
            while True:
                data = up.recv(65536)
                if not data:
                    break
                client.sendall(data)
                out.append(data)
        except OSError:
            pass
        rec.response(b"".join(out))


def serve_proxy(proxy, upstream, rec, done):
    """One request, one connection, both directions. qe-serve answers a stream
    with `Connection: close` and crush sends no `Expect: 100-continue`, so there
    is no keep-alive state to model here."""
    while not done.is_set():
        try:
            client, _ = proxy.accept()
        except OSError:
            break
        try:
            # One timeout for the connect, another for the body, and the second has
            # to be a number the engine can actually beat. A 60 s read timeout was
            # the first version: a 6k-token prompt ingests for 7.5 minutes before
            # its first delta, the recorder saw "no bytes" and hung up on a live
            # stream, and crush reported `stream transport error: unexpected EOF`
            # four times - a harness bug reported as a server bug, which is exactly
            # the kind of green-but-wrong this file exists to avoid.
            up = socket.create_connection(("127.0.0.1", upstream), timeout=10)
            up.settimeout(READ_S)
        except OSError:
            client.close()
            continue
        threading.Thread(target=forward, args=(client, up, rec), daemon=True).start()


def write_config(where, rec_port):
    """The config, in the directory `CRUSH_GLOBAL_CONFIG` wants: that variable is a
    directory, and crush appends `crush.json` to it itself. Handing it the file is
    an error naming a path with the name in it twice."""
    cfg = {
        "$schema": "https://charm.land/crush.json",
        # Crush retries a stream that ends without a reason, and a checkpoint turn
        # can be minutes long; the SDK's own default would cut it off. Set it from
        # the environment to watch the deadline do its work.
        "options": {"request_timeout": REQUEST_TIMEOUT},
        # Not `disable_all_providers`: that key does not exist, and a typo here
        # means every default provider stays a candidate.
        "disable_default_providers": True,
        # A non-interactive run cannot answer a permission prompt, so the tools the
        # test needs are allowed by name. This is the only way to get a tool round
        # trip out of `crush run`: the help text advertises a top-level `--yolo`, and
        # the installed build refuses the shorthand it advertises it under.
        "permissions": {"allowed_tools": ["write", "view", "ls", "grep", "glob", "bash",
                                          "download", "todos"]},
        "providers": {
            "qe": {
                "name": "qe",
                "base_url": "http://127.0.0.1:%d/v1" % rec_port,
                "type": "llamacpp" if DISCOVER else "openai-compat",
                # Any non-empty string: the point is that crush sends the header and
                # the server does not require it.
                "api_key": "qe",
            }
        },
    }
    if DISCOVER:
        # No model named and no window named: crush lists /v1/models, and the
        # llamacpp enricher reads data[].meta.n_ctx out of the answer. A server that
        # keeps its window in a private namespace gets a client that has to be
        # configured by hand, which is the thing this mode exists to notice.
        cfg["providers"]["qe"]["auto_discover_models"] = True
    else:
        cfg["providers"]["qe"]["models"] = [{
            "name": MODEL_ID, "id": MODEL_ID,
            # The truth for this engine is its trained length; a window
            # crush thinks is smaller makes it trim history, which would
            # test a different thing.
            "context_window": CTX,
            "default_max_tokens": MAX_TOKENS,
            "supports_attachments": False,
        }]
    path = os.path.join(where, "crush.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return where


def wait_for_listener(port, srv):
    started = time.time()
    while time.time() - started < WAIT_S:
        if srv is not None and srv.poll() is not None:
            print("server exited early rc=%d" % srv.returncode, flush=True)
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    if shutil.which("crush") is None:
        print("SKIP probe-crush: crush is not on PATH")
        return 0

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else ""
    srv = None
    if not EXISTING:
        cmd = ["./bin/qe-serve", "--port", str(PORT), "--id", MODEL_ID,
               "--threads", os.environ.get("QE_THREADS", "19")]
        if checkpoint:
            cmd += ["--model", checkpoint]
        else:
            cmd[1:1] = ["--spec-k", "4"]
        log = open("/tmp/qe-probe-crush-server.log", "w")
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        print("qe-serve pid %d on :%d (%s)" %
              (srv.pid, PORT, "checkpoint" if checkpoint else "selftest"), flush=True)
        if not wait_for_listener(PORT, srv):
            check("the server this test needs came up", False, "port %d" % PORT)
            return 1

    rec = Recorder()
    done = threading.Event()
    proxy = socket.socket()
    proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy.bind(("127.0.0.1", PROXY_PORT))
    proxy.listen(16)
    threading.Thread(target=serve_proxy, args=(proxy, PORT, rec, done), daemon=True).start()

    work = tempfile.mkdtemp(prefix="qe-probe-crush-")
    # Separate from the working directory on purpose: crush merges every config it
    # finds, and a file that is both the global config and the project config
    # registers the same provider twice, which crush then refuses as ambiguous.
    cfg_dir = tempfile.mkdtemp(prefix="qe-probe-crush-cfg-")
    data = tempfile.mkdtemp(prefix="qe-probe-crush-data-")
    try:
        write_config(cfg_dir, PROXY_PORT)
        # An ambient OPENAI_API_KEY would be sent along with this provider's own,
        # and one server that requires no credentials should not be tested with a
        # client that is confused about which it is answering.
        env = dict(os.environ)
        env.update(CRUSH_GLOBAL_CONFIG=cfg_dir)
        env.pop("OPENAI_API_KEY", None)
        cmd = ["crush", "run", "-q", "-m", MODEL_ID, "--small-model", MODEL_ID,
               "-D", data, "-c", work, PROMPT]
        print("%s ... (cwd %s, up to %d s)" % (" ".join(cmd[:4]), work, RUN_S), flush=True)
        started = time.time()
        # crush retries a stream it considers unfinished, so a control run can
        # retry its way past any deadline the harness sets. That is the outcome, not
        # a harness fault: the turn never completes, which is what the operator sees
        # as "it did not seem to work".
        timed_out = False
        try:
            out = subprocess.run(cmd, cwd=work, env=env, timeout=RUN_S,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired as e:
            timed_out = True
            out = e
            out.returncode = 124
            out.stdout = e.stdout if isinstance(e.stdout, bytes) else b""
        elapsed = time.time() - started
        text = out.stdout.decode("utf-8", "replace")
        with open("/tmp/qe-probe-crush-client.log", "w") as f:
            f.write(say(text))
        print("crush exited rc=%d after %.1f s" % (out.returncode, elapsed), flush=True)
        print("   " + say(" | ".join(text.strip().splitlines()[-6:])))

        requests, responses = rec.snapshot()
        check("crush talked to this server", bool(requests), "%d request(s)" % len(requests))
        if not requests:
            return 1
        check("crush sent its tools", any(r.get("tools") for r in requests),
              "%d of %d request(s) carried tools" %
              (sum(1 for r in requests if r.get("tools")), len(requests)))

        every = list(verdicts(responses))
        # The one event an ingestion keepalive writes: an SSE comment, which is a
        # line the client's parser reads and its message assembler never sees. It
        # is counted by line prefix rather than by shape because it carries no
        # shape: the reason it is a comment and not an empty delta is that an empty
        # delta is a data frame, and a data frame can be finalized as a message.
        kept = sum(1 for r in responses for ln in r.split(b"\n") if ln.startswith(b": "))
        print("   ", say("mode: crush deadline %d s, discovery %s, %d keepalive "
                         "comment(s) seen, crush %s after %.1f s" %
                         (REQUEST_TIMEOUT, DISCOVER, kept,
                          "timed out" if timed_out else "rc=%d" % out.returncode,
                          elapsed)))
        if DISCOVER:
            # Nothing named this model but its id, and nothing anywhere said how
            # large its window is; crush asked /v1/models and ran anyway, which is
            # the assertion that the window is where a llama.cpp client reads it.
            check("the model was found by asking, with no context_window set",
                  not timed_out and out.returncode == 0,
                  "%d request(s), %d stream(s)" % (len(requests), len(every)))
        check("at least one reply streamed", bool(every), "%d stream(s)" % len(every))
        check("every stream ended with a reason",
              bool(every) and all(v["reason"] for v in every),
              "reasons: %s" % [v["reason"] for v in every])
        check("every stream said [DONE]",
              bool(every) and all(v["done"] for v in every),
              "done: %s" % [v["done"] for v in every])
        if SELFTEST:
            skip("a stream slower than the keepalive interval sent an SSE comment",
                 "the selftest engine ingests in milliseconds, so no stream it "
                 "produces is ever quiet long enough for the interval to elapse")
        else:
            check("a stream slower than the keepalive interval sent an SSE comment",
                  kept > 0, "%d event(s)" % kept)
        text_in = "".join(
            ((e.get("choices", [{}])[0].get("delta") or {}).get(k) or "")
            for v in every for e in v["events"]
            if isinstance(e, dict) and e.get("choices") for k in ("content", "reasoning_content"))
        check("tokens arrived while the turn ran", len(text_in) > 0,
              "%d chars" % len(text_in))

        openings = [
            tc for v in every for e in v["events"]
            if isinstance(e, dict) for c in (e.get("choices") or [])
            for tc in ((c.get("delta") or {}).get("tool_calls") or [])
            if tc.get("id") and (tc.get("function") or {}).get("name")
        ]
        call_deltas = [
            tc for v in every for e in v["events"]
            if isinstance(e, dict) for c in (e.get("choices") or [])
            for tc in ((c.get("delta") or {}).get("tool_calls") or [])
        ]
        if SELFTEST:
            why = ("the selftest generator emits no control tokens, so no turn it "
                   "produces can contain a tool request")
            skip("a tool request opened with the fields the SDK needs", why)
            skip("the tool result came back as a role tool message", why)
            skip("the file the model asked for is on disk", why)
            skip("every tool_calls delta carried the index it accumulates under", why)
            skip("no two calls in the conversation share an id", why)
            skip("the transcript that came back named each call once", why)
        else:
            check("a tool request opened with the fields the SDK needs", bool(openings),
                  json.dumps([{"id": t.get("id"), "name": (t.get("function") or {}).get("name")}
                              for t in openings])[:200])
            # A reader accumulates a call's arguments under its index, so a delta
            # without one has nowhere to go; and it keys everything else about the
            # call by id - the block of interface it is drawn into, and the result
            # that is folded back into it - for the length of the conversation,
            # which is why an id may not be handed out twice in one.
            check("every tool_calls delta carried the index it accumulates under",
                  bool(call_deltas) and all("index" in t for t in call_deltas),
                  "%d delta(s) at index/indices %s"
                  % (len(call_deltas), sorted({str(t.get("index")) for t in call_deltas})))
            handed = [t.get("id") for t in openings]
            twice = sorted({i for i in handed if handed.count(i) > 1})
            check("no two calls in the conversation share an id",
                  bool(handed) and not twice,
                  "%d id(s)%s" % (len(handed),
                                  "" if not twice else ", handed twice: %s" % twice[:4]))
            echoes = [[tc.get("id") for m in (r.get("messages") or [])
                       for tc in (m.get("tool_calls") or [])] for r in requests]
            echoes = [e for e in echoes if e]
            mangled = [e for e in echoes if len(e) != len(set(e))]
            check("the transcript that came back named each call once",
                  bool(echoes) and not mangled,
                  "%d transcript(s) carrying %s call(s)"
                  % (len(echoes), max((len(e) for e in echoes), default=0)))
            followups = [r for r in requests
                         if any(m.get("role") == "tool" for m in (r.get("messages") or []))]
            check("the tool result came back as a role tool message", bool(followups),
                  "%d follow-up(s) of %d request(s)" % (len(followups), len(requests)))
            note = os.path.join(work, "note.txt")
            got = open(note).read() if os.path.isfile(note) else None
            check("the file the model asked for is on disk with its bytes",
                  got is not None and got.strip() == "HELLO",
                  "no note.txt" if got is None else "note.txt is %r" % got[:40])
        check("crush finished without an error of its own", out.returncode == 0,
              "rc=%d" % out.returncode)
        return 0
    finally:
        done.set()
        proxy.close()
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(cfg_dir, ignore_errors=True)
        shutil.rmtree(data, ignore_errors=True)
        if srv is not None:
            srv.terminate()


if __name__ == "__main__":
    code = main()
    print("\n%d check(s), %d failure(s), %d skipped" %
          (counts["checks"], len(failures), len(skipped)), flush=True)
    if failures:
        print("failed: " + ", ".join(failures), flush=True)
    sys.exit(1 if failures else code)
