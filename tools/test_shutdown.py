"""SIGTERM is a request-path test, not a formality (default :8098, QE_PORT to move).

Why this is a separate program: closing the listener is the easy half of shutting
down, and it is the only half that used to be tested. A request thread can still be
inside the engine when the accept loop returns - on a real checkpoint, minutes of
ingestion deep - and what happens next is the interesting part. Free the engine
under that thread and shutdown is a use-after-free on the one path every harness
takes; wait for it without a deadline and the server outlives its own SIGTERM,
keeping the port and 87 GiB with it, which reads to whoever runs the next test as a
hung machine.

So both halves are asserted here, against the selftest engine, with no checkpoint:
that a server with work in flight exits by itself within the grace, saying that it
gave up rather than freed, and that a server with nothing in flight still takes the
clean path, because that is the path LeakSanitizer's exit-time check depends on.

The client that gets the server into that state sends request headers for a 48 MiB
body and then three bytes of it: the handler is genuinely blocked reading a request
that is never coming, which is the same position a generation is in as far as the
shutdown path cares - a thread inside a request, holding the engine.
"""
import os
import re
import signal
import socket
import subprocess
import sys
import time

PORT = int(os.environ.get("QE_PORT", "8098"))
GRACE = int(os.environ.get("QE_GRACE", "3"))
BIN = os.environ.get("QE_SERVE", "./bin/qe-serve")
LOG = f"/tmp/qe-shutdown-{PORT}.log"

# Only the lines this program asks for are ever printed, and they are the server's
# own, but a log line is not a place a vocabulary marker belongs either way.
MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")


def say(text):
    return MARKER.sub(lambda m: "{" + m.group(0)[1:-1].strip() + "}", text)


fails = []
skips = []


def check(ok, what, note=""):
    print(("  ok   " if ok else "  FAIL ") + what + ((" " + note) if note else ""))
    if not ok:
        fails.append(what)


def start():
    log = open(LOG, "wb")
    p = subprocess.Popen(
        [BIN, "--port", str(PORT), "--threads", "4", "--id", "qe-selftest",
         "--shutdown-grace", str(GRACE)],
        stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if p.poll() is not None:
            raise SystemExit(f"server exited early rc={p.returncode}: " + open(LOG).read())
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=0.2).close()
            return p
        except OSError:
            time.sleep(0.1)
    p.kill()
    raise SystemExit("server never listened")


def stop_and_wait(p, budget):
    """SIGTERM, then wait for the process to be gone. Never a second signal: the
    whole point is that it does not need one. Returns (rc, seconds)."""
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        return None, time.time() - t0
    return rc, time.time() - t0


def lines(*keys):
    with open(LOG, errors="replace") as f:
        out = [ln.strip() for ln in f if any(k in ln for k in keys)]
    return [say(ln) for ln in out]


def port_free():
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


def half_open_request():
    """Headers for a body that never finishes, sent and left open. Returns the
    socket, which the caller keeps holding so the handler stays inside the
    request; closing it would let the handler finish on a short body."""
    s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    s.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 50000000\r\n"
        b"\r\n"
        b"{__"
    )
    s.settimeout(1)
    return s


def health():
    s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    s.settimeout(5)
    try:
        body = s.recv(4096)
    except OSError:
        body = b""
    s.close()
    return body


print(f"== {BIN} :{PORT} selftest engine, --shutdown-grace {GRACE} ==")

print("== nothing in flight ==")
p = start()
check(health().startswith(b"HTTP/1.1 200"), "serves before the signal")
rc, dt = stop_and_wait(p, GRACE + 10)
check(rc == 0, "exits on SIGTERM without a second signal", f"rc={rc} {dt:.2f} s")
check(dt < 1.0, "and does not wait for a grace it does not need", f"{dt:.2f} s")
check(not lines("still running after"), "says nothing about unfinished work")
check(port_free(), "the port is bindable again")

print("== a request in flight when the signal lands ==")
p = start()
victim = half_open_request()
time.sleep(0.5)  # let the handler get as far as the body
rc, dt = stop_and_wait(p, GRACE + 10)
gave_up = lines("still running after")
check(rc == 0, "exits on SIGTERM without a second signal", f"rc={rc} {dt:.2f} s")
check(gave_up and GRACE - 0.5 < dt < GRACE + 5,
      "outlives the grace by less than the grace buys", f"{dt:.2f} s")
check(bool(gave_up), "says it stopped with work in flight rather than freed under it",
      gave_up[0] if gave_up else "")
try:
    victim.close()
except OSError:
    pass
check(port_free(), "the port is bindable again")

print("== a keep-alive client that is only thinking ==")
p = start()
check(health().startswith(b"HTTP/1.1 200"), "serves before the signal")
idle = socket.create_connection(("127.0.0.1", PORT), timeout=5)
time.sleep(0.3)
rc, dt = stop_and_wait(p, GRACE + 10)
check(rc == 0, "exits on SIGTERM without a second signal", f"rc={rc} {dt:.2f} s")
check(GRACE - 0.5 < dt < GRACE + 5,
      "a client that is idle still costs only the grace", f"{dt:.2f} s")
idle.close()

print("== the graceful path LeakSanitizer depends on ==")
p = start()
check(health().startswith(b"HTTP/1.1 200"), "serves before the signal")
time.sleep(0.3)  # the connection is closed; the handler is not
rc, dt = stop_and_wait(p, GRACE + 10)
check(rc == 0, "exits on SIGTERM without a second signal", f"rc={rc} {dt:.2f} s")
check(dt < 1.0, "immediately, having returned normally", f"{dt:.2f} s")
check(lines("signalled, closing listener"), "reports the normal way out",
      (lines("signalled, closing listener") or [""])[0])
check(not lines("still running after"), "and never mentions the grace")

print()
print("FAIL" if fails else "PASS",
      f"({len(fails)} failures, {len(skips)} skipped)")
sys.exit(1 if fails else 0)
