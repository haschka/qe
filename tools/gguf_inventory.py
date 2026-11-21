#!/usr/bin/env python3
"""Tensor inventory for a split GGUF checkpoint, read from every volume.

`tools/gguf_meta.py` parses one file, which is right for metadata (only volume 1
carries it) and wrong for the tensor table (only volume 1 carries *no* tensors:
in a split checkpoint each volume lists its own tensors, with offsets relative
to that volume's own data section). This walks all volumes, merges the tables,
and prints what the forward pass needs to know: which layer is which kind, where
a layer's dtypes differ from the pattern, and how many bytes each name pattern
carries.

  python3 tools/gguf_inventory.py /path/-00001-of-00003.gguf
  python3 tools/gguf_inventory.py /path/-00001-of-00003.gguf --tsv /tmp/tensors.tsv
  python3 tools/gguf_inventory.py /path/-00001-of-00003.gguf --pattern 'ssm|attn_q'
"""
import argparse
import collections
import os
import re
import struct
import sys

# id -> (name, elements per block, bytes per block), the whole table src/gguf.c's
# TI[] carries. Both halves of the report used to know only the types the trunk
# happened to be quantised with, so the first file with a type outside that set -
# an MTP sidecar, Q4_K and Q5_0 - died in a KeyError inside a byte arithmetic the
# tool has no way to check. One table, two views, so the names and the sizes
# cannot disagree the way two literals can.
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 16), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 82), 18: ("IQ3_XXS", 256, 110), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 66),
    23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 66), 30: ("BF16", 1, 2),
}
GGML_TYPE = {k: v[0] for k, v in TYPES.items()}
BLOCK = {k: (v[1], v[2]) for k, v in TYPES.items()}


def tbytes(shape, dt):
    n = 1
    for x in shape:
        n *= x
    try:
        ne, sz = BLOCK[dt]
    except KeyError:
        sys.exit("type id %d is not in this tool's table: add it from TI[] in "
                 "src/gguf.c" % dt)
    return (n + ne - 1) // ne * sz


def vol_names(path):
    m = re.search(r"(\d+)-of-(\d+)\.gguf$", path)
    if not m:
        return [path]
    total = int(m.group(2))
    head = path[: m.start(1) - 1]
    width = len(m.group(1))
    return ["%s-%0*d-of-%0*d.gguf" % (head, width, i + 1, len(m.group(2)), total)
            for i in range(total)]


class R:
    def __init__(self, buf):
        self.buf = buf
        self.p = 0

    def fixed(self, fmt):
        n = struct.calcsize(fmt)
        if self.p + n > len(self.buf):
            raise EOFError("short read")
        v = struct.unpack_from(fmt, self.buf, self.p)[0]
        self.p += n
        return v

    def s(self):
        n = self.fixed("<Q")
        v = self.buf[self.p: self.p + n].decode("utf-8", "replace")
        self.p += n
        return v

    def skip_val(self, t):
        if t == 8:
            self.p += 8 + self.fixed("<Q")
        elif t in (0, 1, 7):
            self.p += 1
        elif t in (2, 3):
            self.p += 2
        elif t in (4, 5, 6):
            self.p += 4
        elif t in (10, 11, 12):
            self.p += 8
        elif t == 9:
            et = self.fixed("<I")
            for _ in range(self.fixed("<Q")):
                self.skip_val(et)
        else:
            raise ValueError("kv type %d" % t)


def parse(path):
    """Return (version, n_tensors, tensor table, header end) for one volume.

    The header size is not in the file, so parse optimistically and re-read a
    bigger prefix when the KV block (10.9 MB of vocabulary in volume 1) runs
    past the buffer.
    """
    size = os.path.getsize(path)
    want = min(size, 1 << 20)
    while True:
        with open(path, "rb") as f:
            buf = f.read(want)
        try:
            return _parse(buf)
        except (EOFError, ValueError):
            if want >= size:
                raise
            want = min(size, want * 8)


def _parse(buf):
    if buf[:4] != b"GGUF":
        sys.exit("not GGUF")
    r = R(buf)
    r.p = 4
    ver = r.fixed("<I")
    n_tensors = r.fixed("<Q")
    n_kv = r.fixed("<Q")
    align = 32
    for _ in range(n_kv):
        k = r.s()
        t = r.fixed("<I")
        if k == "general.alignment" and t == 4:
            align = r.fixed("<I")
        else:
            r.skip_val(t)
    out = []
    for _ in range(n_tensors):
        name = r.s()
        nd = r.fixed("<I")
        shape = tuple(r.fixed("<Q") for _ in range(nd))
        dt = r.fixed("<I")
        off = r.fixed("<Q")
        out.append((name, shape, dt, off))
    return ver, n_tensors, out, r.p, align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tsv", default="", help="write the merged tensor table here")
    ap.add_argument("--pattern", default="", help="only print patterns matching this regex")
    a = ap.parse_args()

    vols = vol_names(a.path)
    rows = []
    print("== volumes ==")
    aligns = []
    for i, v in enumerate(vols):
        ver, n_tensors, tbl, hdr, align = parse(v)
        aligns.append(align)
        data = (hdr + align - 1) // align * align
        bad = 0
        prev = 0
        for name, shape, dt, off in sorted(tbl, key=lambda t: t[3]):
            if off != -(-prev // align) * align:
                bad += 1
            prev = off + tbytes(shape, dt)
        span = prev
        print("  %s\n      v%d tensors=%-5d header=%d B alignment=%d data starts %d B "
              "file=%.1f MiB span=%d B headroom=%.1f MiB chain mismatches=%d"
              % (os.path.basename(v), ver, n_tensors, hdr, align, data,
                 os.path.getsize(v) / 2.0**20, span, data + span - os.path.getsize(v), bad))
        for name, shape, dt, off in tbl:
            rows.append((name, shape, dt, off, i))
    rows.sort(key=lambda r: (r[4], r[3]))
    total = sum(tbytes(r[1], r[2]) for r in rows)
    print("\n== merged table: %d tensors, %.1f GiB ==" % (len(rows), total / 2.0**30))

    by_type = collections.Counter()
    bytes_type = collections.Counter()
    for name, shape, dt, off, v in rows:
        by_type[GGML_TYPE[dt]] += 1
        bytes_type[GGML_TYPE[dt]] += tbytes(shape, dt)
    print("\ndtype      count        bytes   share")
    for t, b in bytes_type.most_common():
        print("  %-8s %6d  %10.1f MiB  %5.1f%%" % (t, by_type[t], b / 2.0**20,
                                                   100.0 * b / total))

    pat = re.compile(a.pattern) if a.pattern else None
    groups = collections.defaultdict(lambda: [0, 0, collections.Counter(), None, []])
    for name, shape, dt, off, v in rows:
        key = re.sub(r"\.\d+\.", ".*.", name)
        key = re.sub(r"\b\d+\b", "*", key)
        g = groups[key]
        g[0] += 1
        g[1] += tbytes(shape, dt)
        g[2][GGML_TYPE[dt]] += 1
        g[4].append((name, GGML_TYPE[dt]))
        if g[3] is None or name < g[3][0]:
            g[3] = (name, shape)
    print("\n== name patterns ==")
    print("  %-44s %5s %-18s %11s %10s  dtypes"
          % ("pattern", "count", "shape", "MiB total", "MiB ea."))
    for key, (c, b, dts, ex, members) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        if pat and not pat.search(key):
            continue
        dtxt = "/".join("%s x%d" % (t, n) for t, n in dts.most_common())
        print("  %-44s %5d %-18s %11.1f %10.2f  %s"
              % (key, c, "x".join(map(str, ex[1])), b / 2.0**20, b / 2.0**20 / c, dtxt))

    # Layer kinds and dtype exceptions: for each (suffix, layer) look at what exists.
    layers = collections.defaultdict(dict)
    layer_vol = {}
    for name, shape, dt, off, v in rows:
        m = re.match(r"blk\.(\d+)\.(.*)$", name)
        if m:
            layers[int(m.group(1))][m.group(2)] = (GGML_TYPE[dt], shape)
            layer_vol[int(m.group(1))] = v
    kinds = {}
    for L, d in sorted(layers.items()):
        k = []
        if any(s.startswith("ssm_") for s in d):
            k.append("GDN")
        if "attn_q.weight" in d:
            k.append("QSA")
        if any(s.startswith("ple_") for s in d):
            k.append("PLE")
        if any(s.startswith("hc_") for s in d):
            k.append("HC")
        if any(s.startswith("ffn_gate_inp.") for s in d):
            k.append("MoE")
        kinds[L] = "+".join(k) or "?"
    print("\n== layer kinds (%d layers, volume = which file holds the tensor data) ==" % len(layers))
    for L in sorted(layers):
        print("  blk.%-3d vol%d  %s" % (L, layer_vol[L] + 1, kinds[L]))

    print("\n== dtype exceptions within a pattern ==")
    seen = 0
    for key, (c, b, dts, ex, members) in sorted(groups.items()):
        if len(dts) > 1:
            seen += 1
            major = dts.most_common(1)[0][0]
            odd = [(n, t) for n, t in members if t != major]
            print("  %-40s majority %s, %d of %d differ:" % (key, major, len(odd), c))
            for n, t in sorted(odd)[:12]:
                print("      %-44s %s" % (n, t))
            if len(odd) > 12:
                print("      ... %d more" % (len(odd) - 12))
    if not seen:
        print("  none")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("name\ttype\tshape\tbytes\toff\tvolume\n")
            for name, shape, dt, off, v in rows:
                f.write("%s\t%s\t%s\t%d\t%d\t%d\n"
                        % (name, GGML_TYPE[dt], "x".join(map(str, shape)),
                           tbytes(shape, dt), off, v))
        print("\nwrote %s" % a.tsv)


if __name__ == "__main__":
    main()
