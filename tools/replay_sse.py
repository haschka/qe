#!/usr/bin/env python3
"""Record one real chat reply and replay it as many times as needed.

Why this exists: the difference between a client that parsed a tool request and a
client that drew one is a few bytes of framing, and nothing that loads 87 GiB of
weights is a comfortable way to iterate on a few bytes of framing. So record one
reply from a server that produced it, then answer requests with those exact bytes:

    # once, against a server that is already loaded
    python3 tools/replay_sse.py --port 8150 --upstream 8098 --record-to /tmp/fx &
    # then, as often as wanted, with no weights and no inference
    python3 tools/replay_sse.py --port 8150 --reply /tmp/fx/reply0.sse \
                                --reply /tmp/fx/reply1.sse &

Two ways this harness was wrong before either of them were written down:

  * The first request of a fresh conversation carries no `tools` at all - it is the
    small-model request for the session title. Handing that one the recorded tool
    request leaves the agent's own turn with nothing to say, and every conclusion
    drawn afterwards is about a reply the client never saw. Reply files therefore go
    to the requests that carry `tools`, in order, and to no other request; anything
    else gets a one-word answer that ends the turn.
  * A replay is only evidence while it is byte-identical to what was recorded, so
    the frames are copied verbatim (never re-serialised) and every reply written or
    sent is reported with the hash of exactly those bytes. `--selftest` prints what a
    file contains and what would go back out for it, which is how a fixture is
    checked after the loader has been edited.

Nothing recorded here is printed as it arrived: a vocabulary marker on a terminal is
a control token one copy-paste from being live, so all of it goes through `say()`.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time

DATA = re.compile(rb"^data:\s?(.*)$")
MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")
ENDPOINT = re.compile(rb"^(POST|GET) (\S+)")
# A plain reply that ends a turn, for the requests a recorded turn is not for.
PLAIN = [
    b'data: {"id":"replay","object":"chat.completion.chunk","created":1,"model":"qe",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    b'"logprobs":null,"finish_reason":null}]}',
    b'data: {"id":"replay","object":"chat.completion.chunk","created":1,"model":"qe",'
    b'"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}',
    b"data: [DONE]",
]


def say(s):
    """Model text, safe to print: a marker comes back as {its-name} rather than as
    the sequence itself, so nothing printed here can be pasted forward."""
    return MARKER.sub(lambda m: "{" + m.group(0)[2:-2].strip() + "}", str(s))


def read_request(conn, cap=8 << 20):
    """One HTTP request: returns (method, path, headers_blob, body) or None."""
    head = b""
    while b"\r\n\r\n" not in head:
        d = conn.recv(4096)
        if not d:
            return None
        head += d
        if len(head) > 64 << 10:
            return None
    pre, rest = head.split(b"\r\n\r\n", 1)
    m = ENDPOINT.match(pre)
    length = 0
    for line in pre.split(b"\r\n")[1:]:
        n, _, v = line.partition(b":")
        if n.strip().lower() == b"content-length":
            try:
                length = int(v.strip())
            except ValueError:
                length = 0
    while len(rest) < min(length, cap):
        d = conn.recv(65536)
        if not d:
            break
        rest += d
    return (m.group(1) if m else b"POST", m.group(2) if m else b"/", pre, rest[:length])


def load(path):
    """The frames of one recorded reply, verbatim, with their recorded offsets.

    Accepts a raw capture (an HTTP head, then `data:` lines) or the same with the
    seconds since the head arrived in front of each line and a tab, which is what
    `--record-to` writes so that pacing survives. A frame is the whole wire line
    without its newline, kept byte for byte: a replay that re-serialises a frame is a
    replay of something else.
    """
    frames, delays, prev = [], [], 0.0
    with open(path, "rb") as f:
        if not f.readline().startswith(b"HTTP/"):
            raise SystemExit("%s is not a recorded reply (no HTTP head)" % path)
        for line in f:
            line = line.rstrip(b"\r\n")
            when, sep, payload = line.partition(b"\t")
            if sep and re.match(rb"^\d+(\.\d+)?$", when):
                t = float(when)
            else:
                t, payload = prev, line
            # The head, a blank line between head and body, and anything else that is
            # neither a data frame nor a comment is dropped rather than replayed.
            if not (DATA.match(payload) or payload.startswith(b":")):
                continue
            frames.append(payload)
            delays.append(max(0.0, t - prev))
            prev = t
    return frames, delays


def kinds(frames):
    """What each frame is, named the way the notes about a reply read."""
    out = []
    for f in frames:
        m = DATA.match(f)
        if not m:
            out.append("comment" if f.startswith(b":") else "other")
            continue
        if m.group(1).strip() == b"[DONE]":
            out.append("done")
            continue
        try:
            obj = json.loads(m.group(1).decode("utf-8"))
        except ValueError:
            out.append("unparsed")
            continue
        seen = "other"
        for c in obj.get("choices") or []:
            d = c.get("delta")
            if not isinstance(d, dict):
                continue
            if c.get("finish_reason"):
                seen = "finish"
            elif d.get("tool_calls"):
                seen = "call-open" if d["tool_calls"][0].get("id") else "call-args"
            elif "reasoning_content" in d:
                seen = "reasoning"
            elif "content" in d:
                seen = "opener" if d.get("role") else "content"
        out.append(seen)
    return out


def payload_bytes(frames):
    """The exact bytes one reply puts on the wire."""
    return b"".join(f + b"\n\n" for f in frames)


def digest(frames):
    return hashlib.sha256(payload_bytes(frames)).hexdigest()[:12]


def selftest(path):
    frames, _ = load(path)
    ks = kinds(frames)
    print("%s: %d frame(s), %s, sha256 %s" % (path, len(frames), " ".join(ks[:24]), digest(frames)))
    again, _ = load(path)
    if payload_bytes(again) != payload_bytes(frames):
        print("FAIL %s does not load twice to the same bytes" % path)
        return 1
    if "done" not in ks:
        print("FAIL %s has no data: [DONE], so no client would end the turn" % path)
        return 1
    if "data: " not in str(frames[0]) and not frames[0].startswith(b":"):
        print("FAIL %s's first frame is not a data frame" % path)
        return 1
    print("PASS %s replays as the %d bytes it recorded (%d frame(s))"
          % (path, len(payload_bytes(frames)), len(frames)))
    return 0


def record(port, upstream, outdir):
    """Copy every reply of every request to a file, and pass it through untouched.

    The file is the wire bytes with the seconds since the head arrived written in
    front of each line, which is what lets `--delay` replay a reply at the pace it
    was produced at: the intervals are the evidence that a client had time to render
    something, and a capture that loses them cannot be replayed that way again.
    """
    os.makedirs(outdir, exist_ok=True)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)
    sys.stderr.write("recorder :%d -> upstream :%d, replies to %s/\n" % (port, upstream, outdir))
    sys.stderr.flush()
    state = {"n": 0}
    lock = threading.Lock()

    def handle(conn):
        """Serve one client connection, one request at a time.

        Every reply from this engine says `Connection: close`, so a client that
        keeps the socket open is a client that got the whole turn; the loop is for
        that case rather than against it, and each request gets its own upstream
        connection so that a reply is recorded against the request that earned it.
        """
        while True:
            req = read_request(conn)
            if req is None:
                conn.close()
                return
            try:
                up = socket.create_connection(("127.0.0.1", upstream), timeout=30)
            except OSError as e:
                sys.stderr.write("upstream :%d refused: %s\n" % (upstream, say(e)))
                conn.close()
                return
            up.sendall(req[0] + b" " + req[1] + b" HTTP/1.1\r\n"
                       + req[2].split(b"\r\n", 1)[1] + b"\r\n\r\n" + req[3])
            started = time.monotonic()
            chunks = []
            try:
                while True:
                    d = up.recv(65536)
                    if not d:
                        break
                    conn.sendall(d)
                    chunks.append((d, time.monotonic() - started))
            except OSError:
                pass
            up.close()
            if not chunks:
                continue
            with lock:
                state["n"] += 1
                path = os.path.join(outdir, "reply%d.sse" % state["n"])
            write_timed(path, chunks)
            frames, _ = load(path)
            with open(path + ".kinds", "w") as f:
                f.write(" ".join(kinds(frames)) + "\n")
            sys.stderr.write("recorded %s (%d frame(s), %s)\n" % (path, len(frames), digest(frames)))
            sys.stderr.flush()
            # The reply said `Connection: close`, and a client waiting for that
            # promise is waiting for this socket to end.
            if b"connection: close" in chunks[0][0].lower():
                conn.close()
                return

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def write_timed(path, chunks):
    """Rebuild one reply as the HTTP head and then `<seconds>\t<wire line>`.

    The head keeps its own text with no prefix so that the file starts the way a
    recorded reply has to start, and a line is stamped with the arrival time of the
    chunk that ended it.
    """
    blob, marks = b"", []
    for data, when in chunks:
        blob += data
        marks.append((len(blob), when))
    head, _, body = blob.partition(b"\r\n\r\n")
    out = [head]
    pos = len(head) + 4
    for line in body.split(b"\n"):
        while marks and marks[0][0] < pos + len(line):
            marks.pop(0)
        when = marks[0][1] if marks else 0.0
        out.append(b"%.6f\t%s" % (when, line))
        pos += len(line) + 1
    with open(path, "wb") as f:
        f.write(b"\n".join(out) + b"\n")


def replay(port, replies, delay, log):
    loaded = [load(p) for p in replies]
    for p, (frames, _) in zip(replies, loaded):
        print("reply %s: %d frame(s), %d byte(s), sha256 %s"
              % (p, len(frames), len(payload_bytes(frames)), digest(frames)), flush=True)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)
    sys.stderr.write("replayer :%d with %d recorded replier(s)\n" % (port, len(loaded)))
    sys.stderr.flush()
    state = {"agent": 0}
    lock = threading.Lock()

    def handle(conn):
        req = read_request(conn)
        if req is None:
            conn.close()
            return
        _, _, _, body = req
        try:
            parsed = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            parsed = {}
        agent = bool(parsed.get("tools"))
        with lock:
            if agent:
                state["agent"] += 1
            turn = state["agent"] - 1 if agent else -1
        frames = []
        chosen = "plain"
        if 0 <= turn < len(loaded):
            frames, delays = loaded[turn]
            chosen = replies[turn]
        else:
            frames, delays = PLAIN, [0.0] * len(PLAIN)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     b"Server: qe\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
                     b"Connection: close\r\n\r\n")
        for i, frame in enumerate(frames):
            if delay:
                time.sleep(delays[i])
            try:
                conn.sendall(frame + b"\n\n")
            except OSError:
                break
        try:
            conn.close()
        except OSError:
            pass
        line = json.dumps({"agent_turn": turn, "reply": chosen,
                           "frames": len(frames), "sha256": digest(frames),
                           "tools": agent,
                           "roles": [m.get("role") for m in
                                     (parsed.get("messages") or [])]})
        if log:
            with lock:
                with open(log, "a") as f:
                    f.write(line + "\n")
        print(say(line), flush=True)

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--reply", action="append", default=[],
                    help="a recorded reply to hand to an agent request, in order")
    ap.add_argument("--delay", action="store_true",
                    help="pace the frames as they were recorded (default: instantly)")
    ap.add_argument("--log", default="", help="append one JSON line per request served")
    ap.add_argument("--upstream", type=int, default=0, help="record mode: server to copy")
    ap.add_argument("--record-to", default="", help="record mode: directory for replies")
    ap.add_argument("--selftest", default="", help="check one recorded reply and exit")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.selftest)
    if a.upstream and a.record_to:
        record(a.port, a.upstream, a.record_to)
        return 0
    if not a.reply:
        ap.error("--reply FILE (repeatable), or --upstream/--record-to, or --selftest FILE")
    replay(a.port, a.reply, a.delay, a.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
