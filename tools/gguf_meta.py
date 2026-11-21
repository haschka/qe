#!/usr/bin/env python3
"""Dump everything a GGUF file says about itself.

Reads the KV metadata block and the tensor table of a GGUF v2+ file and prints
them, so the questions "is this hyperparameter in the file or are we guessing it"
and "what tensor-name shapes does this architecture actually use" get answered by
reading the file instead of by memory.

  python3 tools/gguf_meta.py /path/to-00001-of-00003.gguf            # KVs + shapes
  python3 tools/gguf_meta.py /path.gguf --tensors                    # name patterns
  python3 tools/gguf_meta.py /path.gguf --keys 'hyper_connection,ple'  # grep KVs
"""
import argparse
import collections
import os
import re
import struct
import sys

TYPES = {
    0: ("B", 1), 1: ("B", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
    6: ("f", 4), 7: ("?", None), 8: ("Q", 1), 9: ("d", 8), 10: ("i", 4), 11: ("I", 4),
    12: ("i", 4),
}
ARR = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 8: 9, 9: 10, 10: 11, 11: 12}
# ids as in include/qe.h, which was derived from this file's own offset chain
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16",
}
# bytes per block for the sizes in this file, same table as src/gguf.c
BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 14: (256, 210), 20: (32, 18),
         21: (256, 110), 23: (256, 136), 30: (1, 2)}


def nbytes(shape, dt):
    n = 1
    for x in shape:
        n *= x
    ne, sz = BLOCK.get(dt, (1, 0))
    if not sz:
        return 0
    return (n + ne - 1) // ne * sz


class R:
    def __init__(self, f):
        self.f = f

    def fixed(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return struct.unpack(fmt, b)[0]

    def s(self):
        n = self.fixed("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def val(self, t):
        if t in (0, 1, 7, 8):          # uint8, int8, bool, string-ish handled below
            if t == 8:
                return self.s()
            return self.fixed("<B")
        if t == 2:
            return self.fixed("<H")
        if t == 3:
            return self.fixed("<h")
        if t == 4:
            return self.fixed("<I")
        if t == 5:
            return self.fixed("<i")
        if t == 6:
            return self.fixed("<f")
        if t == 9:                     # array: element type then count
            et = self.fixed("<I")
            n = self.fixed("<Q")
            if et == 8:
                return [self.s() for _ in range(n)]
            return [self.val(et) for _ in range(n)]
        if t == 10:
            return self.fixed("<Q")
        raise ValueError("type %d" % t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true", help="tensor table and name patterns")
    ap.add_argument("--keys", default="", help="comma-separated regexes; print only these KVs")
    ap.add_argument("--all-arr", action="store_true", help="do not truncate long arrays")
    a = ap.parse_args()

    f = open(a.path, "rb")
    r = R(f)
    if r.f.read(4) != b"GGUF":
        sys.exit("not a GGUF file")
    ver = r.fixed("<I")
    n_tensors = r.fixed("<Q")
    n_kv = r.fixed("<Q")
    print("GGUF v%d  tensors=%d  kv_pairs=%d  file=%s (%.1f GiB)"
          % (ver, n_tensors, n_kv, os.path.basename(a.path),
             os.path.getsize(a.path) / 2**30))
    pats = [re.compile(p) for p in a.keys.split(",") if p.strip()]
    print("\n== metadata ==")
    for _ in range(n_kv):
        k = r.s()
        t = r.fixed("<I")
        v = r.val(t)
        if pats and not any(p.search(k) for p in pats):
            continue
        if isinstance(v, list):
            if not a.all_arr and len(v) > 24:
                print("  %-46s [%d] %s ..." % (k, len(v), v[:24]))
            else:
                print("  %-46s %s" % (k, v))
        elif isinstance(v, str) and len(v) > 220 and not a.all_arr:
            print("  %-46s %s... (%d chars)" % (k, v[:220].replace("\n", "\\n"), len(v)))
        elif isinstance(v, float):
            print("  %-46s %.6f" % (k, v))
        else:
            print("  %-46s %s" % (k, v))

    print("\n== tensors (%d) ==" % n_tensors)
    names = []
    by_shape = collections.Counter()
    by_type = collections.Counter()
    sizes = collections.Counter()
    GGML_BLKS = {}
    for i in range(n_tensors):
        name = r.s()
        nd = r.fixed("<I")
        shape = tuple(r.fixed("<Q") for _ in range(nd))
        dt = r.fixed("<I")
        off = r.fixed("<Q")
        names.append((name, shape, dt, off))
        by_shape[shape] += 1
        tn = GGML_TYPE.get(dt, str(dt))
        by_type[tn] += 1
        sizes[tn] += nbytes(shape, dt)
    print("  %d tensors, %d distinct shapes" % (len(names), len(by_shape)))
    print("\n  dtypes: " + ", ".join("%s x%d" % (t, c) for t, c in by_type.most_common()))
    print("\n  most common shapes:")
    for sh, c in by_shape.most_common(28):
        print("    %-34s x%d" % ("x".join(str(x) for x in sh), c))
    if a.tensors:
        print("\n  name patterns (digits and layer indices removed):")
        pats2 = collections.Counter()
        pat_bytes = collections.Counter()
        ex = {}
        for name, shape, dt, off in names:
            p = re.sub(r"\.\d+\.", ".*.", name)
            p = re.sub(r"\b\d+\b", "*", p)
            pats2[p] += 1
            ex.setdefault(p, (name, shape, dt))
        for p, c in sorted(pats2.most_common(200), key=lambda kv: -pat_bytes[kv[0]]):
            n, sh, dt = ex[p]
            tn = GGML_TYPE.get(dt, str(dt))
            pat_bytes[p] += nbytes(sh, dt)
            print("    %-46s x%-4d %-8s %-20s %6.1f MiB each"
                  % (p, c, tn, "x".join(str(x) for x in sh),
                     nbytes(sh, dt) / 2**20))
        print("\n  first 40 and last 40 names:")
        for n, sh, dt, off in names[:40]:
            print("    %-52s %-22s %s" % (n, "x".join(map(str, sh)), GGML_TYPE.get(dt, dt)))
        print("    ...")
        for n, sh, dt, off in names[-40:]:
            print("    %-52s %-22s %s" % (n, "x".join(map(str, sh)), GGML_TYPE.get(dt, dt)))


if __name__ == "__main__":
    main()
