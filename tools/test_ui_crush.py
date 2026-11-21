#!/usr/bin/env python3
"""Drive the real client on a real terminal and assert what it drew.

`make http` asserts requests, and `make probe-crush` asserts a wire contract with a
recording proxy. Neither can see the thing the operator sees, which is whether the
interface shows the tool the model asked for: that question is answered by the bytes
a terminal receives, and a tool request that is parsed, executed and logged but never
drawn looks, from the outside, exactly like a model that acted on its own.

So: start a server that replays recorded replies (no weights, no inference, the same
bytes a real turn produced), point a throwaway client at it on a pty, type one prompt,
and read three independent accounts afterwards - what the terminal was sent, what the
client stored, and what the client logged about itself. They disagree in interesting
ways: a session whose stored tool calls all carry the same id is a session where the
client found one tool call and wrote the next one over it, which is invisible in the
stored data and obvious on screen.

    python3 tools/test_ui_crush.py --reply tools/testdata/replies/turn0-tool-first.sse \\
        --reply tools/testdata/replies/turn1-tool-second.sse \\
        --reply tools/testdata/replies/turn2-answer.sse \\
        --expect-calls 2 --must-draw-final FIRSTCALLMARK \\
        --must-draw-final SECONDCALLMARK

`--server PORT` points at a server that is already up instead, which is how the same
assertions run against a checkpoint that took minutes to load - and against a real
model the number of calls is the model's business, so `--expect-calls` there reads as a
floor, and `--max-tokens` has to be large enough for a reasoning block to reach a tool
request at all. Nothing drawn here is printed as it arrived - a vocabulary marker on a
terminal is a control token one copy-paste from being live - so all of it goes through
`say()`.
"""

import argparse
import fcntl
import glob
import json
import os
import pty
import re
import select
import shutil
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import termios
import time

MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[=>()][0-9A-Za-z]?|\x1b[OM]")
counts = {"checks": 0}
failures = []

HERE = os.path.dirname(os.path.abspath(__file__))


def say(s):
    """Model text, safe to print: a marker comes back as {its-name} rather than as
    the sequence itself, so nothing printed here can be pasted forward."""
    return MARKER.sub(lambda m: "{" + m.group(0)[2:-2].strip() + "}", str(s))


def check(name, ok, detail=""):
    counts["checks"] += 1
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, say(detail)), flush=True)
    if not ok:
        failures.append(name)


def skip(name, why):
    print("%-4s %s %s" % ("SKIP", name, say(why)), flush=True)


def strip_ansi(raw):
    return ANSI.sub("", raw.decode("utf-8", "replace"))


def write_config(where, port, max_tokens=4096):
    """A throwaway config: one provider, no key to leak, every tool allowed without
    asking, and a request timeout that outlives an ingestion."""
    conf = {
        "$schema": "https://charm.land/crush.json",
        "options": {"request_timeout": 3600, "debug": True},
        "disable_default_providers": True,
        "permissions": {"allowed_tools": ["bash", "write", "view", "ls", "grep", "glob"]},
        "providers": {"qe": {
            "name": "qe",
            "base_url": "http://127.0.0.1:%d/v1" % port,
            "type": "openai-compat",
            "api_key": "qe",
            "models": [{"name": "qe", "id": "qe", "context_window": 200000,
                        "default_max_tokens": max_tokens}]}},
    }
    with open(os.path.join(where, "crush.json"), "w") as f:
        json.dump(conf, f, indent=2)


def drive(port, work, cfg, data, prompts, secs, send_at, rows, cols, raw_path,
          idle=20.0, turn_max=600.0, min_between=20.0):
    """Run the client on a pty until `secs` pass, typing one prompt per turn.

    The first prompt goes in at `send_at`; each later one waits for the previous
    turn to end, which on a live model is minutes of streaming and looks like this:
    bytes keep arriving while a turn runs, and a turn is over when they stop for
    `idle` seconds. A client that animates a spinner never goes quiet, so
    `turn_max` is the other ceiling, and both are needed - send on cadence and the
    prompt lands mid-turn, where it is queued as text and never becomes a turn."""
    env = dict(os.environ)
    env["CRUSH_GLOBAL_CONFIG"] = cfg
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    env.pop("OPENAI_API_KEY", None)
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(work)
        os.execve(shutil.which("crush"),
                  ["crush", "-y", "-c", work, "-D", data, "-d"], env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    out = bytearray()
    started = time.time()
    last_byte, last_sent, sent = started, 0.0, 0
    while time.time() - started < secs:
        now = time.time()
        due = sent < len(prompts) and (
            now - started >= send_at if sent == 0
            else (now - last_byte >= idle or now - last_sent >= turn_max)
            and now - last_sent >= min_between)
        if due:
            os.write(fd, prompts[sent].encode())
            time.sleep(0.4)
            os.write(fd, b"\r")
            sent += 1
            last_sent = last_byte = time.time()
        r, _, _ = select.select([fd], [], [], 0.25)
        if fd in r:
            try:
                d = os.read(fd, 65536)
            except OSError:
                break
            if not d:
                break
            out.extend(d)
            last_byte = time.time()
    with open(raw_path, "wb") as f:
        f.write(bytes(out))
    try:
        os.write(fd, b"\x03")
        time.sleep(0.4)
        os.close(fd)
    except OSError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return bytes(out)


def stored(data_dir):
    """What the client wrote down, one line per message part, and the ids of its
    tool calls, which is what the client looks a tool call up by."""
    db = os.path.join(data_dir, "crush.db")
    if not os.path.isfile(db):
        return [], []
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        rows = con.execute("select role,parts from messages order by created_at").fetchall()
    except sqlite3.Error:
        return [], []
    lines, ids = [], []
    for role, parts in rows:
        try:
            parsed = json.loads(parts)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        bits = []
        for part in parsed:
            d = part.get("data", {})
            kind = part.get("type")
            if kind == "tool_call":
                ids.append(d.get("id"))
                bits.append("tool_call(%s)" % d.get("name"))
            elif kind == "tool_result":
                bits.append("tool_result")
            elif kind in ("text", "reasoning"):
                bits.append("%s(%d)" % (kind, len(str(d.get("text") or d.get("thinking") or ""))))
            else:
                bits.append(str(kind))
        if bits:
            lines.append("%-9s %s" % (role, " | ".join(bits)))
    return lines, ids


def logged(data_dir):
    """Every WARN and ERROR the client wrote about itself."""
    out = []
    logs = sorted(glob.glob(os.path.join(data_dir, "logs", "*.log")), key=os.path.getmtime)
    for path in logs[-1:]:
        with open(path, errors="replace") as f:
            for line in f:
                if not line.startswith("{"):
                    continue
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("level") in ("WARN", "ERROR"):
                    out.append("%s %s %s" % (j.get("level"), j.get("msg"),
                                             json.dumps(j.get("error") or j.get("message") or "")[:160]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reply", action="append", default=[],
                    help="recorded reply to replay, in order (repeatable)")
    ap.add_argument("--server", type=int, default=0, help="use a server already on this port")
    ap.add_argument("--port", type=int, default=8157)
    ap.add_argument("--prompt", default="Use the bash tool to run exactly: echo QETOOLMARK, then tell me it ran.")
    ap.add_argument("--then", action="append", default=[],
                    help="a further prompt, typed when the turn before it goes quiet "
                         "(repeatable) - the shape the reported bug needed")
    ap.add_argument("--idle", type=float, default=20.0,
                    help="seconds of a quiet terminal that mean a turn ended")
    ap.add_argument("--turn-max", type=float, default=600.0,
                    help="seconds one turn is allowed before the next prompt goes in anyway")
    ap.add_argument("--secs", type=float, default=150)
    ap.add_argument("--send-at", type=float, default=12)
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--cols", type=int, default=200)
    ap.add_argument("--tail-bytes", type=int, default=80000,
                    help="how much of the end of the terminal stream to search for --must-draw-final")
    ap.add_argument("--expect-calls", type=int, default=1,
                    help="how many tool calls the run has to produce (a floor against a "
                         "real model, an exact count against a replay)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="what the client is told one turn may be; a reasoning model "
                         "needs tens of thousands to reach a tool request")
    ap.add_argument("--must-draw", action="append", default=[],
                    help="text that has to appear somewhere on the terminal")
    ap.add_argument("--must-draw-final", action="append", default=[],
                    help="text that has to appear in the final frames, so after every turn ran")
    ap.add_argument("--raw", default="/tmp/qe-ui-crush.raw")
    a = ap.parse_args()

    if shutil.which("crush") is None:
        skip("the client drew a block per tool call", "crush is not installed")
        skip("no two tool calls share an id", "crush is not installed")
        print("\n%d check(s), 0 failure(s)" % counts["checks"])
        return 0

    base = "/tmp/qe-ui-crush"
    shutil.rmtree(base, ignore_errors=True)
    work, cfg, data = (os.path.join(base, x) for x in ("work", "cfg", "data"))
    for d in (work, cfg, data):
        os.makedirs(d)
    write_config(cfg, a.server or a.port, a.max_tokens)

    replay = None
    try:
        if a.server:
            port = a.server
        else:
            if not a.reply:
                ap.error("--reply FILE (repeatable) or --server PORT")
            cmd = [sys.executable, os.path.join(HERE, "replay_sse.py"), "--port", str(a.port)]
            for r in a.reply:
                cmd += ["--reply", r]
            replay = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            for _ in range(200):
                s = socket.socket()
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", a.port)) == 0:
                    s.close()
                    break
                s.close()
                time.sleep(0.25)
            else:
                print("FAIL the replayer never listened on :%d" % a.port)
                return 1
            port = a.port

        raw = drive(port, work, cfg, data, [a.prompt] + list(a.then), a.secs, a.send_at,
                    a.rows, a.cols, a.raw, idle=a.idle, turn_max=a.turn_max)
        text = strip_ansi(raw)
        lines, ids = stored(data)
        for line in lines:
            print("  %s" % say(line))
        tail = strip_ansi(raw[-a.tail_bytes:])

        check("the client ran a tool call of its own", any("tool_call(" in l for l in lines),
              "%d call(s) stored: %s" % (len(ids), json.dumps(ids)))
        check("the client drew something for the tool it ran",
              any(m in text for m in ("Bash", "bash")) and len(raw) > 0,
              "%d byte(s) of terminal" % len(raw))
        check("no two tool calls were handed the same id",
              len(ids) >= a.expect_calls and len(set(ids)) == len(ids),
              "%d id(s), %d distinct, expected at least %d"
              % (len(ids), len(set(ids)), a.expect_calls))
        for m in a.must_draw:
            check("the terminal drew %r" % m, m in text, "%d byte(s) streamed" % len(raw))
        for m in a.must_draw_final:
            check("the final frames still draw %r" % m, m in tail,
                  "last %d byte(s) of the terminal stream" % min(len(raw), a.tail_bytes))
        noisy = [l for l in logged(data) if l.startswith("ERROR")]
        check("the client logged no error of its own", not noisy, "; ".join(noisy[:2]))
        print("\n%d check(s), %d failure(s)" % (counts["checks"], len(failures)))
        if failures:
            print("drawn tail:\n%s" % say("\n".join(text.splitlines()[-40:])))
        return 1 if failures else 0
    finally:
        if replay is not None:
            replay.send_signal(signal.SIGTERM)
            try:
                replay.wait(timeout=10)
            except subprocess.TimeoutExpired:
                replay.kill()


if __name__ == "__main__":
    sys.exit(main())
