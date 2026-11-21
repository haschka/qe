#!/usr/bin/env python3
"""Writes a draft-head GGUF the size of a screenshot.

The real sidecar this repository targets is 2.6 GiB and takes minutes to place in
RAM, which is the right cost for measuring speculation and the wrong cost for
asserting that a server can be told which head file to read. This writes a file with
the same shape of metadata - the architecture keys a head carries, one extra decoder
block's worth of names, and the nextn tensors a head is made of - in about 160 KiB,
so `make ui` and `make http` can point --mtp at something real without touching a
checkpoint.

The tensor table is written from the loader's own (elements per block, bytes per
block) table, so opening this file proves the loader accepts it and proves nothing
about whether those numbers are right. They are not assumed here: Q5_0 at 22 bytes a
block was derived from the offset chain of the real sidecar, whose three Q5_0 rows
each left a 204,800 byte hole while the table said 20.

Usage: make_mtp_fixture.py PATH [--arch qwen4exp] [--embd 2560] [--blocks 49]
                               [--nextn 1]
"""
import argparse
import struct

T_F32, T_Q5_0 = 0, 6
BLOCK = {T_F32: (1, 4), T_Q5_0: (32, 22)}  # elements per block, bytes per block
ALIGN = 32

META = [
    ("general.architecture", 8, "qwen4exp"),
    ("general.type", 8, "model"),
    ("general.name", 8, "qe fixture draft head"),
    ("general.size_label", 8, "1x0.001B"),
    ("qwen4exp.block_count", 4, 49),
    ("qwen4exp.embedding_length", 4, 2560),
    ("qwen4exp.nextn_predict_layers", 4, 1),
]


def s(t: str) -> bytes:
    b = t.encode()
    return struct.pack("<Q", len(b)) + b


def kv(k: str, ty: int, v) -> bytes:
    if ty == 8:
        return s(k) + struct.pack("<I", ty) + s(v)
    if ty == 4:
        return s(k) + struct.pack("<II", ty, v)
    raise SystemExit(f"type {ty} not wired")


def tensor_bytes(ne, ty):
    n = 1
    for d in ne:
        n *= d
    bper, bsz = BLOCK[ty]
    return -(-n // bper) * bsz


def write(path, arch="qwen4exp", embd=2560, blocks=49, nextn=1):
    # The names are the real head's, so a later lookup can be tested against this
    # file; the shapes are the fixture's, except the norms, which stay one f32 per
    # activation so their widths still mean what the checkpoint says they mean.
    tensors = [
        (f"blk.{blocks - 1}.nextn.enorm.weight", [embd], T_F32),
        (f"blk.{blocks - 1}.nextn.hnorm.weight", [embd * 4], T_F32),
        (f"blk.{blocks - 1}.nextn.hc_head_norm.weight", [embd * 4], T_F32),
        (f"blk.{blocks - 1}.nextn.hc_head_up.weight", [320, 320], T_Q5_0),
    ]
    off = 0
    info = b""
    for name, ne, ty in tensors:
        info += s(name) + struct.pack("<I", len(ne))
        info += b"".join(struct.pack("<Q", d) for d in ne)
        info += struct.pack("<IQ", ty, off)
        off += tensor_bytes(ne, ty)
        off = -(-off // ALIGN) * ALIGN

    meta = b""
    for k, ty, v in META:
        if k.startswith("qwen4exp."):
            k = arch + "." + k[len("qwen4exp."):]
        if ty == 4 and k.endswith("embedding_length"):
            v = embd
        if ty == 4 and k.endswith("block_count"):
            v = blocks
        if ty == 4 and k.endswith("nextn_predict_layers"):
            v = nextn
        meta += kv(k, ty, v)

    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(META))
    body += meta + info
    body += b"\0" * (-(-len(body) // ALIGN) * ALIGN - len(body))
    with open(path, "wb") as f:
        f.write(body)
        f.write(b"\0" * off)
    return len(body) + off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--arch", default="qwen4exp")
    ap.add_argument("--embd", type=int, default=2560)
    ap.add_argument("--blocks", type=int, default=49)
    ap.add_argument("--nextn", type=int, default=1)
    a = ap.parse_args()
    print(f"{a.path}: {write(a.path, a.arch, a.embd, a.blocks, a.nextn)} bytes")


if __name__ == "__main__":
    main()
