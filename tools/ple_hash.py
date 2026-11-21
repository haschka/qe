"""The PLE n-gram row ids, computed from the checkpoint's own metadata.

One layer of this model reads sixteen rows of a 26.8 GiB n-gram table per token,
selected by hashing the token with its predecessors. A hash that is wrong still
returns plausible floats, so nothing in the engine notices: only the loss does,
four layers later. This is the reference arithmetic - the window rule, the
multipliers, the modulo and the per-head offset - written out independently of
the engine, from the keys in the file, so that the row ids the engine gathers can
be put next to the row ids the file asks for.

  python3 tools/ple_hash.py --model <volume1.gguf> --tokens 11,279,264,2018
  python3 tools/ple_hash.py --model <volume1.gguf> --token-file ids.txt --positions 4
  QE_PLE_DEBUG=1 ./bin/bench_score ... | grep '^ple '   # the engine's own rows

Missing predecessors (before the start of the sequence) and anything at or before
an EOS in the window read as the PLE EOS id; the token's own EOS does not cut its
own context.
"""

import argparse
import struct
import sys

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32 = range(7)
T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 7, 8, 9, 10, 11, 12

SCALAR = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I",
    T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q",
    T_FLOAT64: "<d",
}
MASK = (1 << 64) - 1


class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        if len(b) != n:
            raise EOFError("short read")
        self.p += n
        return b

    def val(self, t):
        if t == T_STRING:
            n = struct.unpack("<Q", self.take(8))[0]
            return self.take(n).decode("utf-8", "replace")
        if t == T_ARRAY:
            et = struct.unpack("<I", self.take(4))[0]
            n = struct.unpack("<Q", self.take(8))[0]
            return [self.val(et) for _ in range(n)]
        fmt = SCALAR[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]


def metadata(path):
    with open(path, "rb") as fp:
        data = fp.read(1 << 24)  # the KV block of a checkpoint of any size fits
    if data[:4] != b"GGUF":
        raise SystemExit("%s: not a GGUF file" % path)
    r = Reader(data)
    magic = r.take(4)
    version = struct.unpack("<I", r.take(4))[0]
    n_tensors = struct.unpack("<Q", r.take(8))[0]
    n_kv = struct.unpack("<Q", r.take(8))[0]
    kv = {}
    for _ in range(n_kv):
        key = r.val(T_STRING)
        typ = struct.unpack("<I", r.take(4))[0]
        kv[key] = r.val(typ)
    return kv, n_tensors


def row_ids(tokens, ngram, hpn, mult, off, vsz, eos):
    """One list of (ngram-1)*hpn row ids per position, oldest position first."""
    heads = (ngram - 1) * hpn
    out = []
    for i, tok in enumerate(tokens):
        ctx = [tok]
        cut = False
        for s in range(1, ngram):
            t = None if cut else (tokens[i - s] if i - s >= 0 else None)
            cut = cut or t is None or t == eos
            ctx.append(eos if cut else t)
        mixed = []
        for n in range(1, ngram + 1):
            acc = (ctx[0] * mult[0]) & MASK
            for j in range(1, n):
                acc ^= (ctx[j] * mult[j]) & MASK
            mixed.append(acc)
        rows = []
        for h in range(heads):
            n = 2 + h // hpn
            if n > ngram:
                n = ngram
            rows.append(mixed[n - 1] % vsz[h] + off[h])
        out.append(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="first volume of the checkpoint")
    ap.add_argument("--tokens", help="comma-separated token ids, oldest first")
    ap.add_argument("--token-file", help="file of token ids, one per line or comma separated")
    ap.add_argument("--positions", type=int, default=0, help="only this many positions")
    ap.add_argument("--arch", default=None, help="architecture prefix (from the file)")
    args = ap.parse_args()

    if args.tokens:
        ids = [int(x) for x in args.tokens.replace("\n", ",").split(",") if x.strip()]
    elif args.token_file:
        with open(args.token_file) as fp:
            ids = [int(x) for x in fp.read().replace("\n", ",").split(",") if x.strip()]
    else:
        raise SystemExit("pass --tokens or --token-file")
    if args.positions:
        ids = ids[:args.positions]

    kv, _ = metadata(args.model)
    arch = args.arch or kv.get("general.architecture", "qwen4exp")
    p = lambda k: "%s.ple.%s" % (arch, k)
    for k in ("ngram_size", "heads_per_ngram", "layer_multipliers", "head_offsets",
              "head_vocab_sizes", "eos_token_id"):
        if p(k) not in kv:
            raise SystemExit("%s: no %s key in this file" % (args.model, k))
    ngram = int(kv[p("ngram_size")])
    hpn = int(kv[p("heads_per_ngram")])
    mult = [int(x) for x in kv[p("layer_multipliers")]]
    off = [int(x) for x in kv[p("head_offsets")]]
    vsz = [int(x) for x in kv[p("head_vocab_sizes")]]
    eos = int(kv[p("eos_token_id")])

    print("# %s: ngram %d, %d heads per n-gram, eos %d" % (arch, ngram, hpn, eos),
          file=sys.stderr)
    for i, rows in enumerate(row_ids(ids, ngram, hpn, mult, off, vsz, eos)):
        print("%3d  tok %-8d  %s" % (i, ids[i], " ".join(str(r) for r in rows)))


if __name__ == "__main__":
    main()
