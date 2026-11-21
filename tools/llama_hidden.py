"""Per-token hidden states from llama.cpp, for comparing trunks instead of logit vectors.

Two implementations of one graph disagree about 248320 logits and about nothing
else. The vector the output projection reads is 2560 numbers, it is everything
the trunk decided, and llama.cpp will hand it over: the qwen4exp graph sets
`t_embd` to the output-norm mixer's result, and `--pooling none` makes the
embedding endpoint return that vector for every token rather than one pooled
summary. Cosine similarity per position then says exactly where this engine
stopped agreeing with that one, which no aggregate loss can do.

The server has to be started for embeddings and told not to normalise, or the
vectors are unit-length and the comparison says nothing about scale:

  llama-server -m <vol1.gguf> -lm mmap+mlock --lazy-mode off -ngl 0 -t 20 \\
      -c 2048 --embedding --pooling none --embd-normalize -1 \\
      --port 8096 --host 127.0.0.1

Row `i` of the output is the hidden state after tokens `0..i`, i.e. what
`bench_score --hidden` records as step `i+1`. Never runs at the same time as qe:
one checkpoint at a time, 87 GiB each.

  python3 tools/llama_hidden.py --port 8096 --in /tmp/llama-ref.txt \\
      --out /tmp/llama-hidden.f32
"""

import argparse
import http.client
import json
import struct
import sys


def post(port, path, body, timeout):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, body=json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    c.close()
    if r.status != 200:
        raise SystemExit(f"{path} failed: {r.status} {data[:400]}")
    return json.loads(data)


def rows(j):
    """The per-token vectors, whichever shape this build answered with."""
    if isinstance(j, dict):
        if "embedding" in j and isinstance(j["embedding"], list):
            e = j["embedding"]
            return e if e and isinstance(e[0], list) else [e]
        if "data" in j and isinstance(j["data"], list):
            return [d["embedding"] for d in j["data"]]
    raise SystemExit("unrecognised embedding response: " + json.dumps(j)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scrub", type=int, default=1,
                    help="replace the pipe character with a bang so neither engine "
                         "parses a control-token string out of the text")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    with open(args.infile, encoding="utf-8") as fp:
        text = fp.read()
    if args.scrub:
        text = text.replace("|", "!")

    j = post(args.port, "/embedding", {"content": text, "pooling": "none"}, args.timeout)
    vecs = rows(j)
    widths = {len(v) for v in vecs}
    print(f"== {len(vecs)} vectors, widths {sorted(widths)} ==")
    if len(widths) != 1:
        raise SystemExit("ragged response")
    with open(args.out, "wb") as fp:
        for v in vecs:
            fp.write(struct.pack(f"<{len(v)}f", *v))
    print(f"  wrote {args.out} ({len(vecs)} x {len(vecs[0])} f32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
