"""Capture a wider oracle: reference text plus per-position logprobs, in one run.

`tools/llama_reference.py` and `tools/llama_logprobs.py` are the two halves, but
each one expects a server that somebody else started and stopped, and the rule
that matters - never qe and llama.cpp with the checkpoint resident at the same
time, 87 GiB each on a 125 GiB box - is the thing a human driving two shells
forgets. This owns the server for the length of the capture and kills it on every
exit path, so the two engines are sequential by construction rather than by care.

What it produces is one corpus of reference prose, concatenated from several
prompts so the text is not one long monologue, and the per-position logprobs of
that same corpus. `tools/cmp_oracle.py` then puts qe's per-token nll next to it.

    python3 tools/oracle_capture.py --model <vol1.gguf> --n 384 \
        --corpus /tmp/ref-wide.txt --oracle /tmp/llama-logprobs-wide.tsv

Nothing here touches qe, and nothing here reads weight bytes at token time:
`--lazy-mode off` is what keeps the 26.8 GiB per-layer embedding table out of the
I/O path, and it is load-once, so the capture costs one model load.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# Prompts chosen to be different in kind, not just in words: arithmetic, prose
# explanation, a list, and code-ish material. A single topic corpus lets one
# mis-scaled subsystem hide inside one register.
PROMPTS = [
    "Explain, in one short paragraph, why a CPU inference engine has to keep the "
    "weights resident in RAM.\n",
    "Write a numbered list of the five things that dominate the latency of a "
    "single token in a large sparse mixture-of-experts model.\n",
    "Here is a C function that computes a dot product of quantized vectors:\n"
    "static float dot(const uint8_t *a, const uint8_t *b, int n) {\n",
    "Q: A server has 20 threads, 125 GiB of RAM and one memory channel per NUMA "
    "node. It runs a 100-billion-parameter model quantized to 4 bits.\n"
    "A: The first thing to work out is the memory traffic per token, which is\n",
]


def wait_ready(port, timeout):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n", type=int, default=384,
                    help="tokens to generate per prompt")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS),
                    help="how many of the built-in prompts to use")
    ap.add_argument("--corpus", default="/tmp/ref-wide.txt")
    ap.add_argument("--oracle", default="/tmp/llama-logprobs-wide.tsv")
    ap.add_argument("--load-timeout", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0,
                    help="score at most this many positions per document")
    args = ap.parse_args()

    cmd = [
        "llama-server", "-m", args.model,
        "-lm", "mmap+mlock", "--lazy-mode", "off", "-ngl", "0",
        "-t", str(args.threads), "-tb", str(args.threads),
        "-c", str(args.ctx),
        "--port", str(args.port), "--host", "127.0.0.1",
    ]
    log = open(f"/tmp/oracle-server-{args.port}.log", "w", encoding="utf-8")
    print(f"== starting {' '.join(cmd)} (log {log.name}) ==", flush=True)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    rc = 1
    try:
        if not wait_ready(args.port, args.load_timeout):
            print("server never became healthy; see the log above", file=sys.stderr)
            return 1
        print("== server healthy ==", flush=True)

        chunks = []
        for i in range(min(args.prompts, len(PROMPTS))):
            out = f"/tmp/oracle-part{i}.txt"
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, "llama_reference.py"),
                 "--port", str(args.port), "--prompt", PROMPTS[i],
                 "--n", str(args.n), "--out", out],
                check=False, capture_output=True, text=True)
            sys.stdout.write(r.stdout[-400:])
            sys.stderr.write(r.stderr[-400:])
            if os.path.exists(out):
                with open(out, encoding="utf-8") as fp:
                    text = fp.read()
                # Neither engine may see a control-token string, and a bang is
                # the character both tokenise literally.
                chunks.append(text.replace("|", "!"))
                print(f"  part {i}: {len(text)} chars", flush=True)

        corpus = "\n\n".join(chunks)
        with open(args.corpus, "w", encoding="utf-8") as fp:
            fp.write(corpus)
        print(f"== corpus {args.corpus}: {len(corpus)} chars, "
              f"{corpus.count('!')} bangs ==", flush=True)
        if not chunks:
            return 1

        rc = subprocess.run(
            [sys.executable, os.path.join(HERE, "llama_logprobs.py"),
             "--port", str(args.port), "--in", args.corpus,
             "--out", args.oracle] +
            (["--limit", str(args.limit)] if args.limit else []),
            check=False).returncode
        return rc
    finally:
        print("== stopping server ==", flush=True)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        for _ in range(60):
            if p.poll() is not None:
                break
            time.sleep(1.0)
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        p.wait()
        log.close()
        mem = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        print(mem.splitlines()[1], flush=True)
        with open("/tmp/oracle-capture.json", "w", encoding="utf-8") as fp:
            json.dump({"corpus": args.corpus, "oracle": args.oracle, "rc": rc}, fp)


if __name__ == "__main__":
    sys.exit(main())
