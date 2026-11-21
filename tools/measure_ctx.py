#!/usr/bin/env python3
"""How a request's cost splits between prompt length and answer length.

The one number this engine reports for a whole request - tokens per second -
divides the emitted tokens by a duration that includes the prompt, so a long
prompt looks like a slow model. This tool separates them: it sends the same
short answer against prompts of different sizes and prints the pass count next
to the seconds, which is the only pair that says whether the cost is in the
prompt, in the answer, or in the context the answer now has to read.

    ./bin/qe-serve --model "$CHECKPOINT" --port 8099 &
    python3 tools/measure_ctx.py                       # 50 .. 16000 tokens
    PROMPTS=50,4000,12000 ANSWER=64 python3 tools/measure_ctx.py

Every number it prints is a `measured` block from the server it is talking to,
so a report can name the command that produced it.
"""

import json
import os
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("PORT", os.environ.get("QE_PORT", 8099)))
BASE = "http://127.0.0.1:%d" % PORT
LENGTHS = [int(x) for x in os.environ.get("PROMPTS", "50,2000,8000,16000").split(",")]
ANSWER = int(os.environ.get("ANSWER", "32"))
MODEL = os.environ.get("MODEL", "qe")

# Words, not filler: a repeated single token compresses the tokenizer's work and
# the prompt cache in a way real text does not.
WORDS = ("memory bandwidth controls how fast a decode step finishes on this host "
         "and every optimization that does not move fewer bytes either wastes "
         "a core or leaves the DRAM bus idle while it waits").split()


def prompt(approx_tokens):
    out, n = [], 0
    while n < approx_tokens:
        line = " ".join(WORDS)
        out.append(line)
        n += len(line) // 4
    return "\n".join(out)


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main():
    print("prompt tokens   answer   target passes   seconds      passes/s   tok/s   "
          "prompt_tokens  cached")
    for approx in LENGTHS:
        text = prompt(approx)
        body = {"model": MODEL, "messages": [{"role": "user", "content": text}],
                "max_tokens": ANSWER, "temperature": 0.0}
        t0 = time.time()
        j = post(body)
        wall = time.time() - t0
        m = j.get("qe", {}).get("measured", {})
        u = j.get("usage", {})
        passes = m.get("target_passes", 0)
        secs = m.get("seconds", 0.0)
        print("%13d %8d %15d %10.3f %11.2f %7.2f %13d %7d   (wall %.1f s)" % (
            approx, m.get("tokens", 0), passes, secs,
            passes / secs if secs else 0.0,
            m.get("tokens", 0) / secs if secs else 0.0,
            u.get("prompt_tokens", 0),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            wall))


if __name__ == "__main__":
    main()
