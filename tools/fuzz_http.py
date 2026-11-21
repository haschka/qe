"""Hostile-input pass for the hand-rolled HTTP/JSON stack (QE_PORT, default 8099).

The point of this file is not "does it answer correctly" (smoke_http.py does
that). It is "can a client make this process die, hang, or answer a request it
never fully read". Every case is something a real client or a scanner on a
public port will send, sent raw rather than through http.client so that
malformed framing is even possible.

Run it under the sanitiser build, then stop the server with SIGTERM and read
the leak report:

    cc -std=gnu17 -g -O1 -fsanitize=address,undefined -Iinclude -Isrc \\
       -pthread -o /tmp/qe-serve-asan src/*.c -pthread -lm
    /tmp/qe-serve-asan --port 8098 &
    QE_PORT=8098 python3 tools/fuzz_http.py
    kill -TERM %1
"""
import base64
import json
import os
import socket
import sys

PORT = int(os.environ.get("QE_PORT", "8099"))
HOST = "127.0.0.1"

fails = []
cases = 0


def check(ok, what, detail=""):
    global cases
    cases += 1
    if ok:
        print("  ok   %s" % what)
    else:
        print("  FAIL %s %s" % (what, detail))
        fails.append(what)


def raw(payload, read=True, timeout=10.0, pre=None):
    """Send bytes exactly as given; return the response (b"" on close)."""
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    try:
        for p in pre or ():
            s.sendall(p)
        if payload:
            s.sendall(payload)
        if not read:
            return b"<closed early>"
        s.settimeout(timeout)
        out = b""
        try:
            while len(out) < 65536:
                b = s.recv(65536)
                if not b:
                    break
                out += b
        except (socket.timeout, ConnectionResetError, BrokenPipeError):
            pass
        return out
    except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
        return b"<reset: %s>" % str(e).encode()
    finally:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        s.close()


def status(r):
    if not r or not r.startswith(b"HTTP/"):
        return 0
    try:
        return int(r.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return 0


def post(obj, extra=b""):
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\nContent-Length: %d\r\n"
            b"Connection: close\r\n" % len(body)) + extra + b"\r\n"
    return raw(head + body)


def alive():
    """The server must still be serving after all of this."""
    r = raw(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    return status(r) == 200


print("== framing ==")
check(status(raw(b"")) != 200, "empty connection is not a request")
check(status(raw(b"\r\n\r\n")) in (0, 400, 405), "bare CRLF handled", status(raw(b"\r\n\r\n")))
check(status(raw(b"NOTAMETHOD\r\n\r\n")) in (0, 400, 405), "request line without path")
check(status(raw(b"GET HTTP/1.1\r\n\r\n")) in (0, 400, 405), "method-only request line")
check(status(raw(b"GET /health HTTP/1.\r\n\r\n")) in (0, 200, 400), "truncated version")
check(status(raw(b"GET /health HTTP/9.9 HTTP/1.1\r\n\r\n")) in (0, 200, 400, 414),
      "duplicated version token")
check(status(raw(b"GET /health HTTP/1.1\r\nnot-a-header-line\r\n\r\n")) in (0, 200, 400),
      "header with no colon")
check(status(raw(b"GET /health HTTP/1.1\r\nX-a: 1\r\nX-a\r\n\r\n")) in (0, 200, 400),
      "second malformed header")
big_hdr = b"GET /health HTTP/1.1\r\nX-big: " + b"a" * 40000 + b"\r\n\r\n"
check(status(raw(big_hdr)) in (0, 200, 414, 431), "40 KB header line refused",
      status(raw(big_hdr)))
check(status(raw(b"GET /" + b"x" * 4000 + b" HTTP/1.1\r\n\r\n")) in (0, 200, 404, 414),
      "4 KB path refused", status(raw(b"GET /" + b"x" * 4000 + b" HTTP/1.1\r\n\r\n")))

print("== body framing ==")
check(status(raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                b"4\r\ntest\r\n0\r\n\r\n")) == 501, "chunked request refused with 501")
check(status(raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Length: 5\r\nConnection: close\r\n\r\nab")) != 200,
      "body shorter than Content-Length")
check(status(raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Length: 1000000000000\r\nConnection: close\r\n\r\n{}")) != 200,
      "absurd Content-Length")
check(status(raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Length: 2\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")) in
      (0, 400, 501), "duplicate Content-Length")
check(status(raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Length: -5\r\nConnection: close\r\n\r\n{}")) != 200,
      "negative Content-Length")
check(status(raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Length: abc\r\nConnection: close\r\n\r\n{}")) != 200,
      "non-numeric Content-Length")
# Head-of-line: declare a body, send half of it, and go away.
raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 4000\r\n"
    b"Connection: close\r\n\r\n" + b"{" * 40, read=False)
check(alive(), "server survives a client that stops mid-body")

print("== JSON shapes ==")
for body, what in [
    (b"", "empty body"),
    (b" ", "whitespace body"),
    (b"null", "null body"),
    (b"[1,2]", "array body"),
    (b"\"a string\"", "string body"),
    (b"{", "unterminated object"),
    (b"{\"model\":", "unterminated value"),
    (b"{\"model\":\"x\",}", "trailing comma"),
    (b"{}\x00\x01binary", "trailing binary"),
    (b"\xef\xbb\xbf{}", "UTF-8 BOM"),
    (b"{'model': 'x'}", "single quotes"),
    (b'{"model": "x", "messages": undefined}', "bare undefined"),
    (b'{"model":"x","messages":[],"messages":[]}', "duplicate key"),
    (b'{"model":"' + b"a" * 300000 + b'","messages":[]}', "300 KB model id"),
    (b'{"model":"x","messages":[' + b"[" * 5000 + b"]" * 5000 + b"]}]",
     "5000-deep nesting"),
    (b'{"model":"x","n":' + b"9" * 400 + b"}", "400-digit number"),
    (b'{"model":"x","max_tokens":1e400}', "exponent overflow"),
    (b'{"model":"x","messages":[{"role":"user","content":[]},]}',
     "empty content array"),
]:
    st = status(post(body))
    check(st != 200, "%s refused (got %d)" % (what, st))

print("== typed fields ==")
# An absurd max_tokens is a value to survive rather than one to honour at face
# value: it is trimmed to the engine's window (4096 here, 262144 for a checkpoint)
# and reported as `qe.measured.budget`. Before pointing this fuzzer at a
# checkpoint, note that the ceiling is then a window, and a window at 74.9 ms a
# token is hours of engine. The `1e400` above is refused outright, so only the
# accepted-and-trimmed cases generate anything at all.
for k, v in [
    ("max_tokens", -5), ("max_tokens", 0), ("max_tokens", 10 ** 9),
    ("max_tokens", "lots"), ("max_tokens", 1.5),
    ("temperature", -1), ("temperature", 1e18), ("temperature", "hot"),
    ("n", 0), ("n", -3), ("n", 5), ("n", "many"),
    ("seed", -1), ("seed", 2 ** 63), ("seed", "abc"),
    ("stream", "yes"), ("stream", 1),
]:
    st = status(post({"model": "qe-selftest", "messages": [{"role": "user",
                                                            "content": "hi"}], k: v}))
    check(st in (200, 400, 413), "%s=%r handled (got %d)" % (k, v, st))

print("== messages ==")
for msgs, what in [
    ("null", "messages null"),
    ({}, "messages an object"),
    ([], "messages empty"),
    ([{}], "message with no role"),
    ([{"role": "user"}], "message with no content"),
    ([{"role": 1, "content": "x"}], "role a number"),
    ([{"role": "user", "content": {"text": "x"}}], "content an object"),
    ([{"role": "user", "content": "x"}] * 5000, "5000 messages"),
    ([{"role": "user", "content": "\x00\x01\x02"}], "control characters"),
    ([{"role": "user", "content": "\ud800"}], "lone surrogate in client JSON"),
]:
    st = status(post({"model": "qe-selftest", "messages": msgs}))
    check(st in (200, 400, 413), "%s handled (got %d)" % (what, st))

print("== image parts ==")
MIME = "data:image/png;base64,"
for b64, what in [
    ("", "empty data"),
    ("!!!!", "invalid base64 alphabet"),
    ("aGVsbG8", "base64 without padding"),
    ("aGVsbG8=", "base64 with one pad"),
    ("aGVsbG8====", "base64 with too much padding"),
    ("a" * 200000, "150 KB of decoded bytes"),
    ("!!!\x00\x01", "control characters in base64"),
]:
    part = [{"role": "user", "content": [
        {"type": "text", "text": "x"},
        {"type": "image_url", "image_url": {"url": MIME + str(b64)}}]}]
    st = status(post({"model": "qe-selftest", "messages": part}))
    check(st in (200, 400, 413), "image %s handled (got %d)" % (what, st))

st = status(post({"model": "qe-selftest", "messages": [
    {"role": "user", "content": [{"type": "image_url",
                                  "image_url": {"url": "data:application/x-msdownload"
                                                  ";base64," +
                                                  base64.b64encode(b"\x7fELF").decode()}
                                  }]}]}))
check(st in (200, 400, 415), "non-image mime refused or ignored (got %d)" % st)
st = status(post({"model": "qe-selftest", "messages": [
    {"role": "user", "content": [{"type": "image_url",
                                  "image_url": "http://127.0.0.1/x.png"}]}]}))
check(st in (400, 415), "remote image url refused (got %d)" % st)

print("== streams ==")
r = raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\n"
        b"Content-Length: %d\r\nContent-Type: application/json\r\n\r\n%s" %
        (len(b'{"model":"qe-selftest","stream":true,"max_tokens":8,"messages":'
              b'[{"role":"user","content":"hi"}]}'),
         b'{"model":"qe-selftest","stream":true,"max_tokens":8,"messages":'
         b'[{"role":"user","content":"hi"}]}'), timeout=20)
check(r.startswith(b"HTTP/1.1 200") and b"[DONE]" in r, "stream completes")
raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\n"
    b"Content-Length: 100\r\n\r\n"
    b'{"model":"qe-selftest","stream":true,"max_tokens":2000,"messages":'
    b'[{"role":"user","content":"hello"}]}', read=False)
check(alive(), "server survives a client that leaves mid-stream")

print("== keep-alive abuse ==")
pre = [b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n",
       b"GET /health HTTP/1.1\r\n",  # half a request, then abandon
       ]
raw(b"", pre=pre, timeout=5)
check(alive(), "server survives an abandoned pipelined request")

if not alive():
    check(False, "server still alive at the end of the fuzz pass")

print("\n%d checks, %d failures" % (cases, len(fails)))
sys.exit(1 if fails else 0)
