"""Greedy reference text from llama.cpp, for cross-checking the forward pass.

Teacher-forced loss on this repository's prose says whether two readings of the
reference graph differ; it cannot say whether either of them is a working model.
A second engine can: `ref/qwen4exp.cpp` is the same architecture llama.cpp's
qwen4exp support builds, so a greedy run of the reference on the same prompt is
the closest thing to ground truth available offline, and the text it produces is
also the best held-out corpus this repository can produce for `--corpus`.

Two things make the comparison mean something. The request has to be a bare
completion, not a chat turn, because the forward pass under test is given raw
token ids; and the server has to be started with everything resident,

    llama-server -m <vol1.gguf> -lm mmap+mlock --lazy-mode off -ngl 0 -t 20 \
      -c 2048 --port 8095 --host 127.0.0.1

because the default loading mode reads the 26.8 GiB per-layer embedding table
from disk at token time, which measures the filesystem rather than the model.

This never runs at the same time as qe: one checkpoint at a time, 87 GiB each.

  ./tools/llama_reference.py --port 8095 --out /tmp/llama-ref.txt
"""

import argparse
import http.client
import json
import sys
import time

PROMPT = (
    "Explain, in one short paragraph, why a CPU inference engine has to keep the "
    "weights resident in RAM.\n"
)


def completion(port, prompt, n, seed, timeout):
    body = {
        "prompt": prompt,
        "n_predict": n,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": False,
    }
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(
        "POST",
        "/completion",
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    if r.status != 200:
        raise SystemExit(f"completion failed: {r.status} {data[:400]}")
    c.close()
    return json.loads(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    j = completion(args.port, args.prompt, args.n, args.seed, args.timeout)
    wall = time.time() - t0

    text = j.get("content", "")
    timings = j.get("timings", {})
    tps = timings.get("predicted_per_second") or 0.0
    print(f"== llama.cpp reference ({wall:.1f} s wall, {tps:.2f} tok/s generated) ==")
    print("---8<--- reference text starts here")
    sys.stdout.write(text.replace("|", "!") + "\n")
    print("--->8--- reference text ends here")
    if timings:
        print(f"  timings: {json.dumps(timings)}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(text)
        print(f"  wrote {args.out} ({len(text)} chars)")
    return 0 if text.strip() else 1


if __name__ == "__main__":
    sys.exit(main())
