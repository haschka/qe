"""Per-token logprobs from llama.cpp, to turn teacher-forced loss into a bisecting instrument.

Teacher-forced nll on text llama.cpp produced itself says only "the engine is
wrong". Per-token logprobs say *where*: a model scored on its own greedy
continuation sits near zero nats at almost every position, so the first position
whose nll is far above the reference is the first position the engine got wrong,
and every position before it is evidence the trunk is fine.

`/completion` with echo returns logprobs for generated tokens only in this
server build, so the same thing costs one request per position: post the growing
prefix as token ids (the tokeniser is asked once, up front, so the prefixes are
exact and never cut mid-token), take the logprob of the token it then produces,
and match it against the corpus. Where the corpus token is not the server's own
argmax the position has no usable reference and is marked `-`.

Never runs at the same time as qe: one checkpoint at a time, 87 GiB each.

  llama-server -m <vol1.gguf> -lm mmap+mlock --lazy-mode off -ngl 0 -t 20 \\
      -tb 20 -c 2048 --port 8095 --host 127.0.0.1
  python3 tools/llama_logprobs.py --port 8095 --in /tmp/llama-ref.txt \\
      --out /tmp/llama-logprobs.tsv
"""

import argparse
import http.client
import json
import math
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


def find_logprobs(node, depth=0):
    """First logprobs-shaped list anywhere in a response, since the server has
    moved it between releases (choices[0].logprobs.content, completion.logprobs,
    a flat content array)."""
    if depth > 8:
        return None
    if isinstance(node, list):
        if node and all(isinstance(x, dict) and "logprob" in x for x in node):
            return node
        for x in node:
            got = find_logprobs(x, depth + 1)
            if got:
                return got
        return None
    if isinstance(node, dict):
        for v in node.values():
            got = find_logprobs(v, depth + 1)
            if got:
                return got
    return None


def gen_one(port, ids, timeout, top):
    """Score one prefix: returns the entry describing the token the server chose."""
    body = {
        "prompt": ids,
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": top,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    j = post(port, "/completion", body, timeout)
    entries = find_logprobs(j)
    if not entries:
        raise SystemExit("no logprobs in response: " + json.dumps(j)[:400])
    return entries[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="score at most this many positions")
    ap.add_argument("--scrub", type=int, default=1,
                    help="replace the pipe character with a bang before sending, so "
                         "neither engine parses a control-token string out of the text")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    with open(args.infile, encoding="utf-8") as fp:
        text = fp.read()
    if args.scrub:
        text = text.replace("|", "!")

    toks = post(args.port, "/tokenize", {"content": text, "special": False}, args.timeout)
    ids = toks.get("tokens") or toks.get("ids")
    if not ids:
        raise SystemExit("could not tokenise: " + json.dumps(toks)[:400])
    n = len(ids) - 1
    if args.limit:
        n = min(n, args.limit)
    print(f"== reference logprobs: {len(ids)} tokens, scoring positions 1..{n} ==")

    rows = []
    for k in range(n):
        e = gen_one(args.port, ids[:k + 1], args.timeout, args.top)
        chosen = e.get("id")
        if chosen is None:
            txt = e.get("token", "")
            chosen = post(args.port, "/tokenize",
                          {"content": txt, "special": False},
                          args.timeout)["tokens"][0]
        want = ids[k + 1]
        cand = {chosen: -e["logprob"]}
        for t in e.get("top_logprobs") or []:
            tid = t.get("id")
            if tid is not None:
                cand[tid] = -t["logprob"]
        rows.append((k + 1, want, chosen, cand.get(want), cand.get(chosen, -e["logprob"])))
        sys.stdout.write(f"\r  {k + 1:3d}/{n}      ")
        sys.stdout.flush()
    print()

    known = [r[2] for r in rows if r[3] is not None]
    if known:
        print(f"  {len(known)}/{len(rows)} positions have a reference; "
              f"mean nll {-sum(known) / len(known):.4f} "
              f"(ppl {math.exp(min(-sum(known) / len(known), 30)):.4f})")
    print("  step   want    top1   nll(want)  nll(top1)")
    for step, want, chosen, nll, top1 in rows:
        w = "   n/a" if nll is None else f"{nll:7.4f}"
        print(f"  {step:4d}  {want:6d}  {chosen:6d}  {w}   {top1:7.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write("step\twant\ttop1\tnll_want\tnll_top1\n")
            for step, want, chosen, nll, top1 in rows:
                fp.write(f"{step}\t{want}\t{chosen}\t"
                         f"{'nan' if nll is None else '%.6f' % nll}\t{top1:.6f}\n")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
