#!/usr/bin/env python3
"""Replace vocabulary markers in text read from stdin with their name.

Every log this project produces contains model text, and model text contains
control tokens, so no log is safe to print. Point this at one:

    python3 tools/redact.py /tmp/qe-ckpt.log
    make http 2>&1 | python3 tools/redact.py

The pattern is a raw string on purpose. Written as an ordinary string the
escapes decode to real angle brackets and pipes, which the regex engine then
reads as alternation, and it matches ordinary text instead of markers.
"""

import re
import sys

MARKER = re.compile(r"\x3c\x7c[^\x3c\x3e\x7c\n]{1,48}\x7c\x3e")


def say(text: str) -> str:
    """Return text with every marker-shaped run replaced by its own name."""
    return MARKER.sub(lambda m: "{" + m.group(0)[1:-1].strip() + "}", text)


def main() -> int:
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            with open(path, errors="replace") as f:
                sys.stdout.write(say(f.read()))
    else:
        sys.stdout.write(say(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
