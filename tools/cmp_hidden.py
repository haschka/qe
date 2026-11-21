"""Where two engines' trunks stop agreeing, position by position.

Both files are raw little-endian f32 rows of the same width: row i is the hidden
state after tokens `0..i`. `tools/llama_hidden.py` writes the reference's, and
`bench_score --hidden N --hidden-out FILE` writes this engine's for steps 1..N,
so the rows already line up. Cosine says whether the direction is right, relative
L2 says by how much, and `scale` is the giveaway for the two failures that hide
behind a good cosine: a residual stream that is the right shape and the wrong
size, which the output projection turns into overconfident logits.

  python3 tools/cmp_hidden.py --a /tmp/llama-hidden.f32 --width 2560 \\
      --b /tmp/qe-hidden.f32
"""

import argparse
import math
import struct


def load(path, width, offset=0):
    with open(path, "rb") as fp:
        blob = fp.read()
    n = len(blob) // 4 // width
    out = []
    for i in range(offset, n):
        out.append(struct.unpack_from(f"<{width}f", blob, i * width * 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference hidden states")
    ap.add_argument("--b", required=True, help="qe hidden states")
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--offset-a", type=int, default=0)
    ap.add_argument("--offset-b", type=int, default=0)
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    A = load(args.a, args.width, args.offset_a)
    B = load(args.b, args.width, args.offset_b)
    n = min(len(A), len(B), args.limit)
    print(f"== trunk agreement: {n} positions of {min(len(A), len(B))} "
          f"({args.width} floats each) ==")
    print("  pos   cosine    rel-L2  |a|     |b|    scale")
    for i in range(n):
        a, b = A[i], B[i]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
        rel = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / na if na > 0 else float("nan")
        print(f"  {i + 1:3d}  {cos:7.4f}  {rel:8.4f}  {na:7.3f} {nb:7.3f}  {nb / na if na else 0:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
