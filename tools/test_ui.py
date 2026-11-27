"""The live panel: on a terminal, off in a log (default :8092, QE_PORT to move).

Why a pty is the only way to test this: the panel decides whether to exist by
asking whether stdout is a terminal, and every harness in this repository redirects
stdout to a file. That is the same decision that puts the per-request log line on
stderr instead of inside the panel, so one boolean quietly chooses two things - and
both of them are worth asserting. A panel that drew over a redirected log would
corrupt `make http`, and one that never drew on a real terminal would be a feature
nobody ever sees.

Six runs of the same selftest server:

  pty + auto   the panel appears, shows the counters, and owns the log line
  file + auto  nothing is drawn, stdout stays empty, the log line is on stderr
  pty + off    the switch wins over the terminal
  file + on    and the switch wins over the absence of one
  pty + head   a draft head named with --mtp is in the banner whole and in the panel
               by its own name, and no head means no row about one
  pty + narrow a head whose name does not fit sixty columns is cut and marked, not
               wrapped into the row below it

No model text is asserted anywhere, because none is ever drawn: the panel shows
counters, request ids, and scrubbed log lines.
"""
import fcntl
import json
import os
import pty
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
import urllib.request

from make_mtp_fixture import write as write_head

PORT = int(os.environ.get("QE_PORT", "8092"))
BIN = os.environ.get("QE_SERVE", "./bin/qe-serve")

# Draft heads for the two runs that need one. Both are fixtures (tools/
# make_mtp_fixture.py): the real sidecar is 2.6 GiB, and a panel test that took
# minutes to place weights would not be run. The second name is deliberately long,
# because the only way to tell "cut on purpose" from "wrapped and lost" is a name
# that cannot fit the terminal it is drawn in.
HEAD = "/tmp/qe-ui-draft-head.gguf"
LONG_HEAD = ("/tmp/qe-ui-draft-head-fixture-used-to-prove-that-a-panel-row-gets"
             "-cut-and-not-wrapped.gguf")

MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")

# What the panel draws and nothing else does. The labels are matched in both cases
# because the active one is upper case and the idle one is lower, and which one a
# given read catches depends on when the request finished.
PANEL = ["ingest", "emit", "cache", "total", "rewind point", "tok/s"]

# The panel's own header, and the four lines that exist once each. Both are for the
# geometry checks below, which are only meaningful for text that is supposed to be
# single: a substring of a log line is allowed to say "476445.74 tok/s", a second row
# starting "EMIT" is not, because that is what a frame redrawn over the row beneath it
# looks like.
HEADER = "qe-selftest"
LABELS = ["ingest", "emit", "total", "cache"]

fails = []


def check(ok, what, note=""):
    print(("  ok   " if ok else "  FAIL ") + what + ((" " + note) if note else ""))
    if not ok:
        fails.append(what)


def say(text):
    return MARKER.sub(lambda m: "{" + m.group(0)[1:-1].strip() + "}", text)


def wait_port(secs=30.0):
    deadline = time.time() + secs
    while time.time() < deadline:
        s = socket.socket()
        s.settimeout(0.25)
        try:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                return True
        finally:
            s.close()
        time.sleep(0.15)
    return False


class Screen:
    """A terminal just literal enough to say where each byte of a frame landed.

    Every property worth having about a redraw is a property of the screen and not of
    the bytes: the panel repaints the same rows, never writes below itself, never
    scrolls the rows above it, and writes nothing at all when it has nothing new to
    say. Read the stream without modelling the cursor and all four are invisible -
    which is exactly how a cursor-up that was one row too many shipped for as long as
    it did, redrawing the frame one row higher every quarter second."

    HEIGHT is what the panel believes it has when the pty cannot say (this harness
    opens a pty with no window size, so the panel falls back to its default width and
    24 rows).
    """

    HEIGHT = 24

    def __init__(self):
        self.g = {}        # row -> list of characters
        self.r = 0
        self.c = 0
        self.scrolls = 0
        self.wrote = set()  # rows this frame put a character or an erase on
        self.highest = 0

    def text(self, row):
        return "".join(self.g.get(row, []))

    def snapshot(self):
        return {r: self.text(r) for r in range(self.HEIGHT)}

    def put(self, s):
        row = self.g.setdefault(self.r, [])
        while len(row) < self.c:
            row.append(" ")
        for ch in s:
            if self.c < len(row):
                row[self.c] = ch
            else:
                row.append(ch)
            self.c += 1
        self.wrote.add(self.r)
        self.highest = max(self.highest, self.r)

    def lf(self):
        self.c = 0
        if self.r >= self.HEIGHT - 1:
            self.scrolls += 1  # the panel pushed the banner off the screen
            return
        self.r += 1
        self.wrote.add(self.r)
        self.highest = max(self.highest, self.r)

    def csi(self, params, cmd):
        digits = params.split(";")[0]
        n = int(digits) if digits.isdigit() else 1
        if cmd == "G":  # the cursor to a column: where the panel parks it between frames
            self.c = n - 1
            return
        if cmd == "A":
            self.r = max(0, self.r - n)
        elif cmd == "B":
            self.r = min(self.HEIGHT - 1, self.r + n)
        elif cmd == "C":
            self.c += n
        elif cmd == "D":
            self.c = max(0, self.c - n)
        elif cmd == "K":
            row = self.g.setdefault(self.r, [])
            del row[self.c :]
            self.wrote.add(self.r)
            self.highest = max(self.highest, self.r)
        elif cmd == "J":
            for r in range(self.r, self.HEIGHT):
                self.g[r] = []
                self.wrote.add(r)

    def feed(self, data):
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                # Intermediate bytes too: the cursor-shape request (CSI 1 SP q) has one,
                # and a parser that stops at the space prints "[1 q" onto the screen.
                m = re.match(r"\x1b\[([0-9;?]*)([ -/]*)([A-Za-z@])", data[i:])
                if not m:
                    i += 1
                    continue
                if not m.group(2) and (m.group(3) not in "mhlh" or not m.group(1).startswith("?")):
                    self.csi(m.group(1), m.group(3))
                i += m.end()
                continue
            if ch == "\r":
                self.c = 0
            elif ch == "\n":
                self.lf()
            elif ch >= " ":
                self.put(ch)
            i += 1

    def begin_frame(self):
        self.wrote = set()

    def rows_written(self):
        return set(self.wrote)


# A finished request as the panel draws it: the id's tail and the time, then the prompt
# half. The log line's "qe: chat" prefix is dropped there for the columns.
HIST = re.compile(r"^[0-9a-f]{6} \d\d:\d\d:\d\d  in ")
HIST_ANY = re.compile(r"[0-9a-f]{6} \d\d:\d\d:\d\d  in \d")


SGR = re.compile(r"\x1b\[[0-9;]*m")


def inner(row):
    """A row with the panel's box taken off: its side, in either glyph set, and the
    spaces against it."""
    return row.strip().strip("\u2502|").strip()


def frames(data):
    """The stream split into whole updates.

    The synchronisation markers are what make this possible at all: they bracket the
    lines of one frame, so a redraw can be attributed to a frame instead of to a
    position in a byte stream that also contains the startup cursor hide.
    """
    out = []
    for part in data.split("\x1b[?2026h")[1:]:
        out.append(part.split("\x1b[?2026l")[0])
    return out


class Pump(threading.Thread):
    """Reads a master fd the whole time a server is up.

    Slurping the pty from the same thread that talks to the server would deadlock
    the moment the master's buffer filled, and 250 ms of redraws fill it fast."""

    def __init__(self, fd):
        super().__init__(daemon=True)
        self.fd = fd
        self.buf = bytearray()
        self.stop = False
        self.start()

    def run(self):
        while not self.stop:
            try:
                r, _, _ = select.select([self.fd], [], [], 0.1)
            except OSError:
                return
            if self.fd not in r:
                continue
            try:
                b = os.read(self.fd, 65536)
            except OSError:
                return
            if not b:
                return
            self.buf += b

    def text(self):
        return self.buf.decode("utf-8", "replace")


def chat(max_tokens=24):
    body = json.dumps(
        {
            "model": "qe-selftest",
            "messages": [{"role": "user", "content": "count to twenty out loud"}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run(label, args, tty, requests=1, settle=0.7, resize=None):
    """Starts the server one way, drives it, and hands back what each stream got.

    `resize` is a (columns, rows) to change the pty to halfway through: the panel asks
    the terminal for its size on every frame rather than listening for SIGWINCH, so a
    winsize change alone is what a dragged window actually is."""
    err = f"/tmp/qe-ui-{label}.err"
    efd = open(err, "wb")
    master = None
    if tty:
        master, slave = pty.openpty()
        outfd = slave
        outfile = None
    else:
        outpath = f"/tmp/qe-ui-{label}.out"
        outfile = open(outpath, "wb")
        outfd = outfile
    p = subprocess.Popen(
        [BIN, "--port", str(PORT), "--threads", "4", "--id", "qe-selftest",
         "--shutdown-grace", "2"] + args,
        stdin=subprocess.DEVNULL,
        stdout=outfd,
        stderr=efd,
    )
    # The parent does not need the slave side open; leaving it open would also stop
    # the master from ever reporting end-of-file.
    if tty:
        os.close(slave)
    pump = Pump(master) if master is not None else None
    try:
        assert wait_port(), f"{label}: server never listened on :{PORT}"
        got = []
        for _ in range(requests):
            got.append(chat())
        if resize and master is not None:
            fcntl.ioctl(master, termios.TIOCSWINSZ,
                        struct.pack("HHHH", resize[1], resize[0], 0, 0))
        time.sleep(settle)  # at least two more redraws
        body = pump.text() if pump else ""
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        if pump:
            pump.stop = True
            pump.join(timeout=2)
            os.close(master)
        if outfile:
            outfile.close()
        efd.close()
    with open(err, "rb") as f:
        errs = f.read().decode("utf-8", "replace")
    outs = ""
    if not tty:
        with open(f"/tmp/qe-ui-{label}.out", "rb") as f:
            outs = f.read().decode("utf-8", "replace")
    return body, outs, errs, got


def geometry(label, stream):
    """Rebuild every frame the panel drew and report what a screen would have seen.

    Flicker is not a feeling, it is bytes, and every way the old redraw got it wrong is
    a byte pattern this function can look for. Reading the stream is not enough: all
    five properties below are properties of the screen, which is why the emulator
    exists.

      - every row is written only when its text changed, and at most one frame in a run
        (the one after a resize) is allowed to repaint everything
      - the panel's first row is the row it started on, so the frame below it is never
        drawn over the frame above it, which is what a cursor-up one row too many looks
        like
      - no line of the panel exists twice on the screen at once, the other half of that
        bug: the stale copy of a frame left behind below the row it was drawn over, and
        the same finished request repeated down the history rows
      - nothing ever scrolls, and the panel leaves the last row of the screen alone
      - the rows it uses are one contiguous block

    A resize is the interesting case, because the panel asks the terminal how wide it is
    instead of being told, and a harness that never changes the window size never finds
    out what it does when somebody drags one.
    """
    fs = frames(stream)
    # Two is enough for every property below, which are all about one frame against the
    # last. It used to be three, and the third came from the request clock still ticking
    # after the selftest's instant request had finished - a frame the panel no longer
    # draws, because a finished request's clock stops.
    check(len(fs) >= 2, f"{label}: the panel drew several frames", f"{len(fs)}")
    s = Screen()
    stale = []      # rows rewritten to the same text they already had
    fulls = []      # frames that rewrote every row of the panel
    off = []        # frames that put the header somewhere other than row 0
    dup = []        # frames where a panel row exists twice on the screen
    reps = []       # frames that drew the same finished request more than once
    allrows = set()
    for k, f in enumerate(fs):
        before = s.snapshot()
        s.begin_frame()
        s.feed(f)
        after = s.snapshot()
        wrote = s.rows_written()
        allrows |= wrote
        # A frame that rewrote the whole panel is not a flicker, by definition: it is a
        # repaint, and there is exactly one reason to repaint - the terminal changed
        # size under the panel, so what it believes is on screen stops being knowable.
        # Anything short of that has to earn every row it writes. Counting the full
        # repaints instead of allowing them is what keeps this from becoming a loophole.
        band = [r for r in range(Screen.HEIGHT) if after[r].strip()]
        full = bool(band) and set(band) <= wrote
        if full:
            fulls.append(k)
        else:
            for r in wrote:
                if before[r] == after[r]:
                    stale.append((k, r))
        # The header is the one line that exists once, always, and is never rewritten
        # once drawn. Where it landed is therefore where the panel thinks its first row
        # is, and a panel that believes that row moves is a panel crawling up the
        # screen one row a frame - the old redraw's signature, left as a copy of the
        # frame above the row it was drawn over.
        at = [r for r in range(Screen.HEIGHT) if HEADER in after[r]]
        if at != [0]:
            off.append((k, at))
        for want in LABELS:
            hits = [r for r in range(Screen.HEIGHT) if inner(after[r]).lower().startswith(want)]
            if len(hits) > 1:
                dup.append((k, want, hits))
        # The lines under the counters are the last few finished requests, and a
        # finished request is one line. Where the panel has room for more of them than
        # have happened, the ones that have not happened must simply be missing:
        # repeating the oldest to fill the space is a screen that reports one request
        # five times, and five requests is a story about the server that is not true.
        hist = [inner(after[r]) for r in range(Screen.HEIGHT) if HIST.match(inner(after[r]))]
        if len(hist) != len(set(hist)):
            reps.append((k, len(hist)))
    check(not off, f"{label}: every frame began on the row the last one began on",
          f"first few {off[:3]}" if off else "")
    check(not dup, f"{label}: no line of the panel appears twice on the screen",
          f"first few {dup[:3]}" if dup else "")
    check(not reps, f"{label}: no finished request is drawn twice",
          f"first few {reps[:3]}" if reps else "")
    check(not stale, f"{label}: no frame rewrote a row that had not changed",
          f"{len(stale)} needless rows, first few {stale[:4]}")
    # Frame 0 paints a screen that was empty, which is not a repaint. Every full frame
    # after it has to be the resize, and there is only one of those per run.
    again = [k for k in fulls if k]
    check(len(again) <= 1, f"{label}: at most one frame repainted the whole panel",
          f"{len(again)} repaints, frames {again[:6]}")
    check(s.scrolls == 0, f"{label}: the panel never scrolled the screen",
          f"{s.scrolls}")
    check(s.highest < Screen.HEIGHT - 1,
          f"{label}: the panel left the last row of the screen alone",
          f"highest row {s.highest} of {Screen.HEIGHT}")
    rows = sorted(allrows)
    check(rows == list(range(len(rows))),
          f"{label}: the panel occupies one contiguous block of rows",
          f"rows {rows[:8]}..." if rows else "no rows")
    return s


def widest(stream):
    """The longest row on the screen once every frame has been applied.

    Only meaningful after a resize, because a resize forces a full redraw: every row on
    the screen then came from a frame composed for the width that is current, so its
    length is a direct measurement of what the panel thought the terminal was. A row
    one column too long is the row that wraps, and a wrapped row is one the redraw does
    not know about - which is how the old fixed-118 panel counted rows wrong on any
    terminal narrower than that and crawled."""
    s = Screen()
    for f in frames(stream):
        s.begin_frame()
        s.feed(f)
    return max((len(t) for t in s.g.values()), default=0)


def screen_rows(stream):
    """The text of the final screen, row by row, once every frame has been applied."""
    s = Screen()
    for f in frames(stream):
        s.begin_frame()
        s.feed(f)
    return {r: s.text(r) for r in range(s.HEIGHT)}


def head_row(stream):
    """The panel's draft-head row, or None when it drew no such row."""
    for _, text in sorted(screen_rows(stream).items()):
        if "draft head" in text.lower():
            return text
    return None


def main():
    print(f"ui: {BIN} on :{PORT}, selftest engine")

    body, outs, errs, got = run("auto-pty", [], True, requests=2)
    # Colour is between a number and its word on a terminal that has it ("2" in one
    # style, " requests" in another), so the text is read with the styles taken out;
    # the counts below never matched a coloured panel until they were.
    plain = SGR.sub("", body)
    low = plain.lower()
    check(bool(got) and all("choices" in g for g in got),
          "requests answered while the panel was drawing")
    check("\x1b[?25l" in body, "cursor hidden: the panel took the screen")
    for want in PANEL:
        check(want in low, f"panel drew {want!r}")
    # The last frame, not the first: the panel redrew the same line from zero, and a
    # search over the whole stream would read a frame from before the work happened.
    def count(pattern):
        m = re.findall(pattern, low)
        return int(m[-1].replace(",", "")) if m else None

    got_requests = count(r"(\d[\d,]*) request")
    check(got_requests == 2, "panel counted the requests it served", str(got_requests))
    got_emitted = count(r"([\d,]+) emitted")
    check(bool(got_emitted) and got_emitted > 0, "panel counted emitted tokens",
          str(got_emitted))
    got_ingested = count(r"([\d,]+) ingested")
    check(bool(got_ingested) and got_ingested > 0, "panel counted ingested tokens",
          str(got_ingested))
    check("qe-selftest" in body, "panel says which engine it is watching")
    # One writer per screen: with the panel up, the finished request's line belongs
    # to the panel, and a second writer beside it would scramble both.
    check(HIST_ANY.search(plain) is not None, "the request log line moved inside the panel")
    check("qe: chat " not in errs, "the request log line is not on stderr too",
          say(errs[-200:] if errs else ""))
    check("\x1c" not in body or MARKER.search(say(body)) is None,
          "no control token drawn", "see scrubbing")
    geometry("auto-pty", body)
    # A row the panel is allowed to draw is not a row it should invent: with no head
    # named there is nothing for the row to say, and a panel that reported a draft
    # head this server was never handed is the same kind of wrong as one that
    # reported weights it had not placed in RAM.
    check("draft head" not in low, "no draft-head row when no head was named")

    # Nothing happening is something the panel has to be told about, too: a redraw on a
    # timer repaints identical rows whether or not anything changed, and on a terminal
    # across a network that is the flicker, in the strict sense of the word.
    idle_body, _, _, _ = run("idle-pty", [], True, requests=0, settle=2.0)
    idle = frames(idle_body)
    check(1 <= len(idle) <= 6,
          "an idle panel redraws about once a second, not once a tick",
          f"{len(idle)} frames in ~2 s")

    body2, outs, errs, got2 = run("auto-file", [], False)
    check(outs == "", "redirected stdout stays empty", f"{len(outs)} bytes")
    check(not any(w in body2.lower() for w in PANEL), "no panel bytes in a file")
    check("\x1b" not in outs, "no escape sequences written to a file")
    check(bool(got2) and "choices" in got2[0], "requests answered with the panel off")
    # The line every harness here greps for, byte for byte as it has always been -
    # except for the three fields at the end, which are what a turn was made of.
    line = [l for l in errs.splitlines() if l.startswith("qe: chat ")]
    check(len(line) == 1, "the request log line is on stderr", say(errs[-300:]))
    if line:
        check(re.match(r"^qe: chat chatcmpl-[0-9a-f]{12} prompt=\d+ cached=\d+ "
                       r"agreed=\d+ dropped=\d+ answer=\d+ passes=\d+ \d+\.\d\d s "
                       r"\d+\.\d\d tok/s \d+\.\d passes/s content=\d+ thinking=\d+ "
                       r"calls=\d+ finish=\w+$", line[0]) is not None,
              "log line unchanged in shape", say(line[0]))
        check(int(re.search(r"content=(\d+)", line[0]).group(1)) > 0,
              "the selftest reply had something to show", say(line[0]))
    check(not any(l.startswith("qe: WARN ") for l in errs.splitlines()),
          "a turn that answered is not called out as silent")
    check("turn cache:" in errs, "startup says what the turn cache will do")

    body3, _, errs3, _ = run("off-pty", ["--ui", "off"], True)
    check("\x1b[?25l" not in body3, "--ui off wins over a terminal")
    check("qe: chat " not in body3 and "qe: chat " in errs3,
          "--ui off puts the log line back on stderr")

    body4, outs4, errs4, _ = run("on-file", ["--ui", "on"], False)
    check("\x1b[?25l" in outs4 or "ingest" in outs4.lower(),
          "--ui on draws even with stdout in a file", f"{len(outs4)} bytes")
    check("ingest" in outs4.lower(), "and the same bytes land in the file")

    # Halfway through this run the terminal becomes narrower, which is what dragging a
    # window is: a winsize change, no signal the panel has to be told about, and no
    # restart. Every row of the final screen was composed for 72 columns, so the
    # longest one is the panel's own answer to how wide it thinks it is.
    body5, _, _, _ = run("resize-pty", [], True, requests=1, resize=(72, 24))
    geometry("resize-pty", body5)
    w = widest(body5)
    check(0 < w <= 72, "the panel re-laid itself out for a narrower terminal",
          f"widest row {w} of 72")

    # A draft head is 2.6 GiB in real life and named on a command line, which is two
    # facts the panel cannot be vague about: which file, and that it is the one this
    # server opened. The banner carries the path whole; the row carries the name that
    # fits, and says when it had to cut.
    write_head(HEAD)
    write_head(LONG_HEAD)
    head_body, _, head_errs, head_got = run("head-pty", ["--mtp", HEAD], True, requests=1)
    check(bool(head_got) and "choices" in head_got[0],
          "requests answered with a draft head staged")
    check(f"qe: draft head: {HEAD}" in head_errs.splitlines(),
          "banner names the draft head file whole", say(head_errs[-300:]))
    check("nextn 1 layer" in head_errs, "banner says what that head holds",
          say(head_errs[-300:]))
    hr = head_row(head_body)
    check(hr is not None, "the panel drew a draft-head row")
    check(hr is not None and os.path.basename(HEAD) in hr,
          "the panel row says which head it is reading", say(hr or ""))
    geometry("head-pty", head_body)

    narrow_body, _, narrow_errs, _ = run("head-narrow-pty", ["--mtp", LONG_HEAD], True,
                                        requests=1, resize=(60, 24))
    geometry("head-narrow-pty", narrow_body)
    wn = widest(narrow_body)
    check(0 < wn <= 60, "the draft-head row stayed inside a 60 column terminal",
          f"widest row {wn} of 60")
    lr = head_row(narrow_body)
    check(lr is not None, "the narrow panel still drew the draft head")
    check(lr is not None and os.path.basename(LONG_HEAD) not in lr,
          "a name too long for the row is not drawn whole", say(lr or ""))
    check(lr is not None and inner(lr).endswith(".."),
          "and the row admits to being cut", say(lr or ""))
    check(f"qe: draft head: {LONG_HEAD}" in narrow_errs.splitlines(),
          "the banner keeps the whole path the panel had to cut")

    print(f"\n{len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
