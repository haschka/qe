"""Put qe's per-position teacher-forced numbers next to the reference engine's.

The aggregate nll over a corpus is contaminated by a handful of positions the
reference engine itself is uncertain about; the per-position table is the
instrument. This script reads the reference capture (tools/llama_logprobs.py,
columns step, want, top1, nll_want, nll_top1) and one or more qe logs
(bench_score's per-step lines, "  nll X top1 #Y want #Z"), and prints them side
by side, with the mean restricted to the positions the reference is confident
about.

  python3 tools/cmp_oracle.py --oracle /tmp/llama-logprobs.tsv \
      --qe qe=/tmp/score10.log
"""

import argparse
import math
import re
import sys

STEP = re.compile(r"^\s+(\d+)\s+nll\s+(\S+)\s+top1\s+#(\d+)\s+want\s+#(\d+)")
VARIANT = re.compile(r"^\s+variant (0x[0-9a-f]+) \.\.\.")


def read_oracle(path):
    rows = {}
    with open(path) as fp:
        head = fp.readline().split("\t")
        if head[:2] != ["step", "want"]:
            raise SystemExit("%s: unexpected header %r" % (path, head))
        for line in fp:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            rows[int(f[0])] = {
                "want": int(f[1]),
                "top1": int(f[2]),
                "nll_want": float(f[3]) if f[3] not in ("nan", "") else math.nan,
                "nll_top1": float(f[4]) if f[4] not in ("nan", "") else math.nan,
            }
    return rows


def read_qe(path):
    """Per-step lines, keyed by (variant, step). bench_score prints a
    'variant 0x..' header before each sweep entry's steps."""
    rows = {}
    variant = "0x00"
    with open(path, errors="replace") as fp:
        for line in fp:
            m = VARIANT.match(line)
            if m:
                variant = m.group(1)
                continue
            m = STEP.match(line)
            if not m:
                continue
            rows[(variant, int(m.group(1)))] = {
                "nll": float(m.group(2)),
                "top1": int(m.group(3)),
                "want": int(m.group(4)),
                "miss": " (ablated)" in line,
            }
    return rows


def mean(vals):
    v = [x for x in vals if x == x]
    return sum(v) / len(v) if v else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="/tmp/llama-logprobs.tsv")
    ap.add_argument("--qe", action="append", default=[],
                    help="LABEL=LOG, repeatable (default qe=/tmp/score9.log)")
    ap.add_argument("--confident", type=float, default=1.2,
                    help="reference nll_want below which a position is called confident")
    ap.add_argument("--variant", default=None,
                    help="only compare this qe variant (hex, e.g. 0x001)")
    args = ap.parse_args()

    ref = read_oracle(args.oracle)
    labels = args.qe or ["qe=/tmp/score9.log"]
    qes = []
    for spec in labels:
        label, _, path = spec.partition("=")
        if not path:
            label, path = "qe", spec
        qes.append((label, read_qe(path)))

    # Which variants exist across the logs, so a multi-variant sweep prints one
    # section per variant rather than interleaving them.
    variants = sorted({v for _, rows in qes for (v, _) in rows})
    if args.variant:
        want = args.variant.lower()
        variants = [v for v in variants if v.lower().endswith(want)]

    cols = ["%-4s" % "pos", "%7s" % "ref_w", "%7s" % "ref_t1",
            "%7s" % "ref_nw"]
    for label, _ in qes:
        cols.append("%18s" % (label[:18]))
    print("reference nll_want < %.2f counts as confident\n" % args.confident)

    for variant in variants:
        print("== qe variant %s ==" % variant)
        print(" ".join(cols))
        delta = []
        ours = []
        refs = []
        for pos in sorted(ref):
            r = ref[pos]
            confident = r["nll_want"] == r["nll_want"] and r["nll_want"] < args.confident
            cells = ["%-4d%s" % (pos, "*" if confident else " "),
                     "%-7d" % r["want"],
                     "%-7d" % r["top1"],
                     "%7.3f" % r["nll_want"]]
            for label, rows in qes:
                q = rows.get((variant, pos))
                if not q:
                    cells.append("%18s" % "-")
                    continue
                tag = "HIT" if q["top1"] == r["want"] else "   "
                cells.append("%7.3f %s#%-6d" % (q["nll"], tag, q["top1"]))
                if q["want"] != r["want"]:
                    cells[-1] += " !want"
                if confident:
                    delta.append(q["nll"] - r["nll_want"])
                    ours.append(q["nll"])
                    refs.append(r["nll_want"])
            print(" ".join(cells))
        print("\n  positions the reference is confident about: %d" % len(refs))
        if refs:
            print("  mean nll(want) there: qe %.3f   reference %.3f   delta %+.3f"
                  % (mean(ours), mean(refs), mean(delta)))
            hits = sum(1 for pos in sorted(ref)
                       if ref[pos]["nll_want"] < args.confident
                       and any(rows.get((variant, pos), {}).get("top1") == ref[pos]["want"]
                               for _, rows in qes))
            print("  qe top-1 == reference want on %d of them" % hits)
        print(file=sys.stderr if False else sys.stdout)


if __name__ == "__main__":
    main()
