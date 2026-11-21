#!/usr/bin/env python3
"""Tensor-name contract check for src/forward.c.

Reads the GGUF tensor tables of all three split volumes (metadata only, no
weight bytes) and checks that every tensor name the forward pass looks up
exists, with the type it asked for. The engine only finds out at open(), after
minutes of loading, so a name or type that drifted is cheaper to catch here.
"""
import re
import struct
import sys

T = {0: "F32", 1: "F16", 2: "BF16", 3: "Q4_0", 6: "Q4_K", 7: "Q5_K", 8: "Q6_K",
     9: "IQ2_XXS", 10: "IQ2_XS", 11: "Q2_K", 12: "Q3_S", 13: "Q2_S", 14: "Q2_K_XS",
     15: "IQ1_S", 16: "IQ1_M", 17: "BF16x4", 18: "Q4_1", 19: "IQ4_NL", 20: "IQ3_S",
     21: "IQ2_S", 22: "IQ4_XS", 23: "IQ3", 24: "IQ2", 25: "IQ1", 26: "TENSOR"}

TYPES = {"F32": 0, "BF16": 2, "Q8_0": 8, "IQ4_NL": 19, "IQ3_S": 20, "IQ4_XS": 22,
         "Q6_K": 7}


def s(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def skip(f, kind):
    if kind in (0, 2, 4, 5, 6, 7, 10, 11):
        f.read({0: 1, 2: 1, 4: 2, 5: 4, 6: 4, 7: 8, 10: 8, 11: 8}[kind])
    elif kind == 8:
        s(f)
    elif kind == 9:
        t = struct.unpack("<B", f.read(1))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip(f, t)


def table(path):
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        f.read(4)
        f.read(8)
        kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(kv):
            s(f)
            skip(f, struct.unpack("<I", f.read(4))[0])
        nt = struct.unpack("<Q", f.read(8))[0]
        for _ in range(nt):
            name = s(f)
            n = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n)]
            typ = struct.unpack("<I", f.read(4))[0]
            f.read(8)
            out[name] = (T.get(typ, str(typ)), dims)
    return out


# What src/forward.c asks for: (name pattern, required type or None).
WANT = [
    ("token_embd.weight", "Q8_0"), ("output.weight", None),
    ("output_hc_norm.weight", None), ("output_hc_down.weight", None),
    ("output_hc_up.weight", None),
    ("blk.%d.attn_qkv.weight", "Q8_0"), ("blk.%d.attn_gate.weight", "Q8_0"),
    ("blk.%d.ssm_conv1d.weight", "F32"), ("blk.%d.ssm_out.weight", "Q8_0"),
    ("blk.%d.ssm_alpha.weight", "F32"), ("blk.%d.ssm_beta.weight", "F32"),
    ("blk.%d.ssm_norm.weight", "F32"), ("blk.%d.ssm_a", None),
    ("blk.%d.ssm_dt.bias", None),
    ("blk.%d.attn_q.weight", "Q8_0"), ("blk.%d.attn_k.weight", "Q8_0"),
    ("blk.%d.attn_v.weight", "Q8_0"), ("blk.%d.attn_output.weight", "Q8_0"),
    ("blk.%d.attn_q_norm.weight", "F32"), ("blk.%d.attn_k_norm.weight", "F32"),
    ("blk.%d.indexer.q_proj.weight", "BF16"), ("blk.%d.indexer.k_proj.weight", "BF16"),
    ("blk.%d.indexer.q_norm.weight", "F32"), ("blk.%d.indexer.k_norm.weight", "F32"),
    ("blk.%d.hc_%s_norm.weight", None), ("blk.%d.hc_%s_down.weight", None),
    ("blk.%d.hc_%s_up.weight", None), ("blk.%d.hc_%s_inject.weight", None),
    ("blk.%d.ffn_gate_inp.weight", "F32"), ("blk.%d.ffn_gate_exps.weight", None),
    ("blk.%d.ffn_up_exps.weight", None), ("blk.%d.ffn_down_exps.weight", None),
    ("blk.%d.ffn_gate_shexp.weight", None), ("blk.%d.ffn_up_shexp.weight", None),
    ("blk.%d.ffn_down_shexp.weight", None), ("blk.%d.ffn_gate_inp_shexp.weight", None),
    ("blk.%d.ple_key.weight", None), ("blk.%d.ple_value.weight", None),
    ("blk.%d.ple_norm_key.weight", None), ("blk.%d.ple_norm_query.weight", None),
    ("blk.%d.ple_norm_conv.weight", None), ("blk.%d.ple_conv1d.weight", None),
    ("per_layer_token_embd.weight", None),
]

NL = 48
if __name__ == "__main__":
    vols = sys.argv[1:] or [
        "/mnt/data/data/ai-models/llama-cpp/llm/Qwen3.8-Flash-Next-UD-IQ4_XS-0000%d-of-00003.gguf" % i
        for i in (1, 2, 3)]
    ts = {}
    for v in vols:
        ts.update(table(v))
    print("%d tensors across %d volumes" % (len(ts), len(vols)))
    missing = []
    for pat, want in WANT:
        for il in range(NL):
            for sub in ("attn", "ffn"):
                name = pat % ((il, sub) if pat.count("%d") + pat.count("%s") > 1 else il)
                if pat.count("%d") and not pat.count("%s"):
                    name = pat % il
                if "%s" in pat:
                    continue
                if name not in ts:
                    missing.append(name)
                elif want and ts[name][0] != want:
                    missing.append("%s is %s, wanted %s" % (name, ts[name][0], want))
        if pat.count("%s"):
            for il in range(NL):
                for sub in ("attn", "ffn"):
                    name = pat % (il, sub)
                    if name not in ts:
                        missing.append(name)
                    elif want and ts[name][0] != want:
                        missing.append("%s is %s, wanted %s" % (name, ts[name][0], want))
    ple = [n for n in ts if ".ple_" in n]
    print("PLE tensors: %d, layers %s" % (len(ple), sorted({int(n.split(".")[1]) for n in ple})))
    kinds = {}
    for n in ts:
        m = re.match(r"blk\.\d+\.([a-z_A-Z0-9.]+)", n)
        if m:
            kinds.setdefault(m.group(1), []).append(n)
    if missing:
        print("MISSING %d:" % len(missing))
        for m in missing[:40]:
            print("  " + m)
        sys.exit(1)
    print("every tensor the forward pass asks for is in the file")
