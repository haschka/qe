# ARCH: what this checkpoint actually is, read out of the file

This is the specification the forward pass has to satisfy. Everything in §1-§5
was read from the three GGUF volumes with `tools/gguf_inventory.py` (written for
this document) and cross-checked against a second independent parser; §6 is the
per-stage arithmetic, transcribed from the upstream reference implementation and
annotated with which tensor in *this* file supplies each operand. Nothing here is
inferred from a model card.

```sh
# regenerate every table in §1-§4 (headers only, ~0.2 s, reads <1 MiB per volume)
python3 tools/gguf_inventory.py \
  /mnt/data/data/ai-models/llama-cpp/llm/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --tsv /tmp/tensors.tsv
python3 tools/gguf_meta.py <volume1>                 # the 67 metadata keys
./bin/qe-serve --report --model <volume1>            # what the engine itself sees
```

## 1. File layout

Three volumes, 87.2 GiB. Metadata and the tokenizer are in volume 1; **volume 1
contains no tensors at all** - each volume carries its own tensor-info table with
offsets relative to *its own* data section.

| volume | tensors | header bytes | `general.alignment` | data section starts | file size |
|---|---|---|---|---|---|
| 1 | 0 | 10,946,618 | 32 | 10,946,624 (= EOF) | 10,946,624 |
| 2 | 373 | 23,584 | 32 | 23,584 | 49,835,229,856 |
| 3 | 851 | 54,317 | 32 | 54,336 | 43,836,407,744 |

Verified, not assumed: in each volume the offsets tile the data section exactly -
`off[0] == 0` and `off[i+1] == align_up(off[i] + bytes[i], 32)` for every tensor
(**0 chain mismatches**), and `data_start + last_end == st_size` for both data
volumes. So the host address of a tensor is

```
map + align_up(header_end(volume), 32) + (off - off_base(volume))
```

which is what `qe_tensor_data()` computes. Before that formula was implemented
every tensor in volumes 2 and 3 was read displaced by exactly its header size -
see `docs/DESIGN.md` §9.

| volume | holds |
|---|---|
| 2 | `token_embd`, `output`, `per_layer_token_embd`, `output_hc_*`, blocks 0-13 |
| 3 | blocks 14-47 |

## 2. Hyperparameters, all 67 keys grouped

`general.architecture = qwen4exp`, `general.name = "Qwen3.8 Flash Next"`,
`general.description = "A Preview of the Qwen4 Architecture"`,
`general.size_label = 512x56B`, `general.file_type = 30` (IQ4_XS),
`general.quantized_by = Unsloth`, imatrix keys present (926 entries, 45 chunks).

**Trunk**: `block_count 48`, `embedding_length 2560`, `context_length 262144`,
`attention.head_count 24`, `attention.head_count_kv 2`,
`attention.key_length = value_length = 256`,
`attention.layer_norm_rms_epsilon 1e-6`, `full_attention_interval 4`.

**RoPE**: `rope.freq_base 1e7`, `rope.dimension_count 64` (partial RoPE: 64 of
the 256 head dims rotate), `rope.dimension_sections [11, 11, 10, 0]`.

**MoE**: `expert_count 512`, `expert_used_count 10`,
`expert_feed_forward_length 640`, `expert_shared_feed_forward_length 640`,
softmax router with renormalised weights, shared expert gated by a sigmoid.

**GDN** (gated delta net, the 36 linear-attention layers):
`ssm.conv_kernel 4`, `ssm.state_size 128`, `ssm.group_count 16`,
`ssm.time_step_rank 48`, `ssm.inner_size 6144`.

**Indexer / QSA**: `attention.indexer.head_count 4`,
`attention.indexer.key_length 128`, `attention.indexer.top_k 2048`,
`attention.compress_ratios` = 48 entries, `4` at layers 3, 7, ..., 47 and `0`
everywhere else (so `indexer_kpool = 4` for all 12 sparse layers, and one ratio
for the whole model).

**Hyper-connections**: `hyper_connection.count 4`,
`hyper_connection.low_rank 320`.

**PLE** (n-gram per-layer embedding): `ple.layers [1]`, `ple.ngram_size 3`,
`ple.heads_per_ngram 8`, `ple.conv_kernel 4`, `ple.eos_token_id 248044`,
`ple.image_token_id 248056`, `embedding_length_per_layer_input 160`,
`ple.layer_multipliers [23703573157769, 20109073645365, 8052911324071]`,
`ple.head_offsets` and `ple.head_vocab_sizes`, 16 entries each (see §6.3).

**Sampler defaults in the file**: `top_k 20`, `top_p 0.95`, `temp 1.0`.

**Tokenizer**: `tokenizer.ggml.model gpt2`, `tokenizer.ggml.pre qwen35`,
248,320 tokens, 247,587 merges, all 256 byte tokens present, `eos 248046`,
`bos = pad = 248044`, `add_bos_token 0`, 276 control-type ids in the
248044-248070 range, plus a 9,993-character Jinja chat template with
image/video handling. `src/tok.c` implements it; the counts it reports are
exact (`docs/DESIGN.md` §7).

## 3. Tensor inventory

1224 tensors, 87.2 GiB. 23 distinct shapes across all three volumes.

| dtype | tensors | bytes | share |
|---|---|---|---|
| IQ4_NL | 44 | 46,816.0 MiB | 52.4% |
| IQ3_S | 94 | 32,312.5 MiB | 36.2% |
| Q8_0 | 502 | 8,519.9 MiB | 9.5% |
| IQ4_XS | 2 | 850.0 MiB | 1.0% |
| Q6_K | 1 | 497.3 MiB | 0.6% |
| F32 | 557 | 299.0 MiB | 0.3% |
| BF16 | 24 | 37.5 MiB | 0.0% |

By role: PLE table 26.8 GiB, routed expert pools 55.4 GiB, everything else
5.0 GiB. A third of the file is one lookup table and 63% is the expert pools;
what a token actually costs is in `docs/DESIGN.md` §3.

Per-tensor sizes and shapes (from `tools/gguf_inventory.py`, "MiB ea." is per
tensor):

| pattern | count | shape | MiB total | dtype |
|---|---|---|---|---|
| `per_layer_token_embd.weight` | 1 | 160x320001536 | 27466.0 | IQ4_NL |
| `blk.*.ffn_down_exps.weight` | 48 | 640x2560x512 | 23600.0 | IQ4_NL x43, Q8_0 x5 |
| `blk.*.ffn_gate_exps.weight` | 48 | 2560x640x512 | 16581.2 | IQ3_S x47, IQ4_XS x1 |
| `blk.*.ffn_up_exps.weight` | 48 | 2560x640x512 | 16581.2 | IQ3_S x47, IQ4_XS x1 |
| `blk.*.attn_qkv.weight` | 36 | 2560x10240 | 956.2 | Q8_0 |
| `token_embd.weight` | 1 | 2560x248320 | 644.1 | Q8_0 |
| `blk.*.attn_gate.weight` | 36 | 2560x6144 | 573.8 | Q8_0 |
| `blk.*.ssm_out.weight` | 36 | 6144x2560 | 573.8 | Q8_0 |
| `output.weight` | 1 | 2560x248320 | 497.3 | Q6_K |
| `blk.*.attn_q.weight` | 12 | 2560x12288 | 382.5 | Q8_0 |
| `blk.*.ffn_gate_inp.weight` | 48 | 2560x512 | 240.0 | F32 |
| `blk.*.attn_output.weight` | 12 | 6144x2560 | 191.2 | Q8_0 |
| `blk.*.hc_{attn,ffn}_{up,down}.weight` | 4x48 | 320x10240 / 10240x320 | 159.4 ea. | Q8_0 |
| `blk.*.ffn_{gate,up,down}_shexp.weight` | 3x48 | 2560x640 / 640x2560 | 79.7 ea. | Q8_0 |
| `blk.*.indexer.q_proj.weight` | 12 | 2560x512 | 30.0 | BF16 |
| `blk.*.ple_key.weight` | 1 | 2560x10240 | 26.6 | Q8_0 |
| `blk.*.ssm_{alpha,beta}.weight` | 2x36 | 2560x48 | 16.9 ea. | F32 |
| `blk.*.attn_{k,v}.weight` | 2x12 | 2560x512 | 15.9 ea. | Q8_0 |
| `blk.*.hc_{attn,ffn}_inject.weight` | 2x48 | 10240x4 | 7.5 ea. | F32 |
| `blk.*.indexer.k_proj.weight` | 12 | 2560x128 | 7.5 | BF16 |
| `blk.*.ple_value.weight` | 1 | 2560x2560 | 6.6 | Q8_0 |
| `blk.*.ssm_conv1d.weight` | 36 | 4x10240 | 5.6 | F32 |
| `output_hc_{down,up}.weight` | 2 | 10240x320 / 320x10240 | 3.3 ea. | Q8_0 |
| `blk.*.hc_{attn,ffn}_norm.weight` | 2x48 | 10240 (as [2560,4]) | 1.9 ea. | F32 |
| `blk.*.ffn_gate_inp_shexp.weight` | 48 | 2560 | 0.5 | F32 |
| `blk.*.ple_{conv1d}.weight` | 1 | 4x10240 | 0.2 | F32 |
| `blk.*.ple_norm_{key,query,conv}.weight`, `output_hc_norm.weight` | 4 | 10240 (as [2560,4]) | ~0 each | F32 |
| `blk.*.ssm_{norm}.weight` | 36 | 128 | ~0 | F32 |
| `blk.*.attn_{q,k}_norm.weight` | 24 | 256 | ~0 | F32 |
| `blk.*.indexer.{q,k}_norm.weight` | 24 | 128 | ~0 | F32 |
| `blk.*.ssm_a`, `blk.*.ssm_dt.bias` | 2x36 | 48 | ~0 | F32 |

**Names carry no architecture prefix.** They are exactly `token_embd.weight`,
`output.weight`, `blk.N.<suffix>` - a lookup of
`qwen4exp.blk.0.ffn_gate_inp.weight` returns nothing (one bench in this repo
silently skipped all 48 layers that way before it started asserting on tensor
presence).

**Two dtype irregularities**, both of which the MoE dispatch has to handle
per layer, not per model:

| tensors | majority | exception |
|---|---|---|
| `ffn_down_exps` | IQ4_NL x43 | **Q8_0 at layers 2, 4, 30, 46, 47** (891 MiB each, 4.4 GiB total) |
| `ffn_gate_exps`, `ffn_up_exps` | IQ3_S x47 | **IQ4_XS at layer 2** |

`general.file_type = 30` (IQ4_XS) therefore describes almost nothing in the
file; the quant mix table above is the truth.

### Layer kinds

`GDN` = gated delta net (linear attention, no KV cache), `QSA` = full attention
with the lightning indexer, `PLE` = n-gram embedding injection, `HC` =
hyper-connections, `MoE` = routed + shared expert. Every layer has HC and MoE.

| kind | layers |
|---|---|
| GDN (linear attention) | every layer with `(i+1) % 4 != 0`: 0,1,2,4,5,6,8,...,45,46 - **36 layers** |
| QSA (indexer + full attention) | `(i+1) % 4 == 0`: 3,7,11,15,19,23,27,31,35,39,43,47 - **12 layers** |
| PLE injection | layer 1 only |
| hyper-connections + MoE | all 48 |

Layer 1 is a GDN layer and carries PLE - the reference requires that, because
the PLE conv history lives in the recurrent cache.

## 4. Derived dimensions (nothing here is in the file; all of it is forced)

- `hc_dim = 4 x 2560 = 10240` - the width of the hyper-connection residual, and
  of `hc_*_{up,down}`, `ssm_conv1d`, `ple_conv1d`, `ple_key`.
- GDN: `head_k_dim = head_v_dim = ssm.state_size = 128`,
  `num_k_heads = group_count = 16`, `num_v_heads = time_step_rank = 48`,
  `key_dim = 2048`, `value_dim = 6144 = inner_size`,
  `conv_dim = 2*key_dim + value_dim = 10240`. So `attn_qkv` (2560 -> 10240) is
  `[q 2048 | k 2048 | v 6144]` and `attn_gate` (2560 -> 6144) is the output gate
  `z`. `ssm_a` is `-exp(A_log)` per v-head, `ssm_dt.bias` the softplus bias.
  3 v-heads share each k-head (`48 / 16 = 3`).
- QSA: `attn_q` (2560 -> 12288) is **`[q | gate]` interleaved per head**: head h
  occupies `q = [256h, 256h+256)` and `gate = [256h+256, 256h+512)` of a
  `2*256`-wide row. `attn_k`/`attn_v` are `2 heads x 256 = 512`.
  `attn_output` is `6144 -> 2560`.
- Indexer: `q_proj` 4 heads x 128 = 512, `k_proj` 128, norms over 128, RoPE over
  the first 64 dims only. Pooled blocks are `compress_ratio = 4` cells; the
  selection budget is `top_k / ratio = 512` blocks plus the `ratio - 1 = 3`
  tail cells, i.e. **2051 attended cells** per query token no matter how long
  the context is.
- PLE: `ple_n_heads = (ngram_size - 1) * heads_per_ngram = 2 * 8 = 16`, which is
  why `head_offsets` has 16 entries. Row width `160` elements = 5 IQ4_NL blocks
  = **90 bytes per row**. Gathered rows are concatenated to `16 x 160 = 2560`.
  Last head ends at `300001275 + 20000171 = 320001446`, the table has
  320,001,536 rows, so **90 rows are padding**.
- No `output_norm`: the final mixer is `output_hc_{norm,up,down}`. There is no
  `hc_*_inject` for the head mixer - its combine step is absent.
- No MTP/draft tensors in the three trunk volumes (`mtp|predict|eagle|draft` matches
  nothing among the 1224 names). The head is a fourth volume: `mtp-...-Q4_K_M.gguf`,
  34 tensors named `blk.48.*` - one above the trunk's last layer - reaching the engine
  through `--mtp`, which stages it resident and runs it as `f->L[f->n_layer]`. So
  `spec_k` does have something to drive it, and what `make head` found is that a
  verified window is dearer than the drafts it saves - 4.45x a step at nine rows against
  1.85 ids a pass - which leaves one token as the only draft length that pays, and the
  engine default follows the drafter that is resident (docs/DESIGN.md §4).

## 5. Per-token cost, recomputed from the inventory

`bin/bench_gemv <checkpoint>` attributes every tensor by role (10/512 of an
expert pool, one row of `token_embd`, 16 rows of the PLE table, everything else
in full). Computed independently from `/tmp/tensors.tsv` for this document:

| component | MiB/token | share |
|---|---|---|
| attention/GDN/indexer projections | 2786.3 | 50.0% |
| routed experts (10 of 512) | 1108.6 | 19.9% |
| hyper-connections | 656.3 | 11.8% |
| lm_head (Q6_K over 248,320) | 497.3 | 8.9% |
| router logits (F32, dense) | 240.0 | 4.3% |
| shared expert | 239.5 | 4.3% |
| PLE projections | 33.5 | 0.6% |
| final mixer | 6.7 | 0.1% |
| PLE n-gram rows (16 x 90 B) | 0.0014 | ~0% |
| `token_embd` (1 row) | 0.0026 | ~0% |
| **total** | **5568.2** | |

One GDN layer costs 83.2 MiB of dense weights, one QSA layer 77.3 MiB, plus
19.9 MiB of shared expert + router and ~1100 MiB of routed experts per layer
across all 48 layers.

### State that a running sequence has to keep

This is what caps concurrency, not bandwidth:

| state | shape | per sequence |
|---|---|---|
| GDN recurrent state | 36 layers x 48 heads x 128x128 f32 | **113.3 MiB** |
| GDN conv history | 36 layers x 3 x 10240 f32 | 4.2 MiB |
| PLE conv history (layer 1) | 9 x 10240 f32 (dilation 3, kernel 4 -> `(4-1)*3 = 9`) | 0.4 MiB |
| QSA KV cache | 12 layers x 2 x 2 heads x 256 x n_ctx | **12.2 KiB/token q8_0** (default: one byte per value, one scale per head of one cell); 48 KiB/token with `--kv-f32` |
| indexer keys | 12 layers x (raw 128 + pooled 128/4) | 7.5 KiB/token |
| **per token, total** | attention cache + indexer keys + the sequence id | **19.7 KiB/token** q8_0, 55.5 with `--kv-f32` |

At 8k context one sequence is **270 MiB** of state on the default q8_0 cache -
557 MiB if the keys and values are kept in f32 (`make score` prints both for any
context width) - and the GDN state is the largest single piece of it. It is per-session and cannot be paged out between tokens.
The two numbers are what `qe_forward_state_bytes()` reports, per sequence and per
token of context, and the engine banner prints both.

### Which of those bytes a turn boundary has to copy

Putting a running sequence back to where it was is only exact if every byte the next
token reads is put back too, so the table above is also the inventory of what a
snapshot could mean. It is not what one costs. Six of those arrays are written only at
`pos` and read only at positions below it, which makes them append-only: truncating
the one cursor called `pos` is the whole rewind, and nothing is copied. What has to be
copied is the state that is *updated in place* at every token, because its old value is
gone by the time anyone asks for it:

| copied | bytes | why it cannot be truncated instead |
|---|---|---|
| GDN recurrent state | 113,246,208 | 36x48 matrices of 128x128 f32, accumulated in place |
| GDN conv history | 4,423,680 | 3-row ring, its head is `pos % 3`, so the bytes are the state |
| PLE conv history (layer 1) | 368,640 | 9-row ring, head `pos % 9` |
| **one boundary** | **118,038,528** | **112.6 MiB, plus 4 bytes per token of the ids that produced it** |

112.6 MiB is the entire cost of a turn boundary: independent of how long the prompt
was, because the append-only half of the table is already in place and is truncated
rather than copied. What that costs is measured rather than extrapolated at the
76-79 GB/s of `docs/DESIGN.md` §1.2 - one boundary moved by one thread is 48 ms, which
is the width of a core and not the bandwidth of this host - so the copy runs over
`qe_pool_run` in 64-byte-aligned spans, and the two boundaries a turn takes together
cost 8.9 ms (`make prefix` prints it). `qe_state_size()` reports the figure for a given
forward pass, and the copy is `qe_forward_snapshot()`/`qe_forward_restore()`.

Deliberately outside it: the PLE n-gram row cache (row-id keyed, so prefix
independent), the PLE pending-row FIFO (one token's window, drained rather than
restored), and the MoE expert histogram (instrumentation, not state).

## 6. The forward pass

Transcribed from the upstream model definition
(`llama.cpp:src/models/qwen4exp.cpp`, reachable from this host with the `fetch`
tool at
`raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen4exp.cpp`),
with each operand named as it appears in this file. Matrices are `[K, N]`
(input width, output width) as GGUF stores them; `x` is a single f32 activation
vector unless stated.

```
h = token_embd[token]                                   (Q8_0 row, 2560)
res = [h h h h]                                         (4 identical HC streams, 10240)
for layer 0..47:
    if layer == 1: res = PLE(res)                       # 6.2
    xn, inject = HC_mix(res, hc_attn_*)                 # 6.1
    y = GDN(xn)  if layer is GDN else QSA(xn)           # 6.3 / 6.4
    res = HC_combine(res, y, inject)
    xn, inject = HC_mix(res, hc_ffn_*)
    y = MoE(xn)                                          # 6.5
    res = HC_combine(res, y, inject)
xn = HC_mix(res, output_hc_norm/up/down)                 # the output norm
logits = output . xn                                     # Q6_K, 248320
```

### 6.1 Hyper-connection mixer (`hc_*`)

Two per layer (before the token mixer, before the MoE) plus one at the head.
`hc = 4`, `hc_dim = 10240`, low rank 320.

```
xn     = rms_norm(res, eps=1e-6) * hc_norm        # grouped: reduce per stream,
                                                  # gamma is [2560,4], stored +1
lo_inj = (1/4) * hc_down_inject . xn              # 321 rows: 320 + 4 in one matmul
lo     = silu(lo_inj[0:320])
gate   = hc_up . lo                                 # 10240
mixed  = mean over streams of (xn * sigmoid(gate))  # 2560 -> the block input
inject = (1/4) * lo_inj[320:324]                    # 4 values
combine: res += y * (2 * sigmoid(inject))           # broadcast y over streams;
                                                    # 2*sigmoid centres on 1
```

`hc_attn_down`/`hc_attn_inject` are separate tensors in this file (the reference
prefers a merged `hc_attn_down_inject` when present), so the two low matmuls
stay separate here. The `2*sigmoid` is what makes a zero injection an ordinary
residual add.

**As implemented:** the three matmuls are ordinary `qe_gemv_t` calls over
`hc_*_down` (10240→320), `hc_*_up` (320→10240) and `hc_*_inject` (10240→4); the
elementwise half is `src/layer.c` - `qe_rms_norm_grouped` (4 groups of 2560, gamma
stride 2560), `qe_stream_gate_mean`, `qe_stream_inject_add` - and its measured cost
per token is docs/DESIGN.md §13. The one detail that silently changes the model is
the grouped reduction: `hc_*_norm.weight` is 10240 wide because the streams are
concatenated, and the mean-square is per stream of 2560.

### 6.2 PLE: n-gram hashed embedding (layer 1 only)

Row selection (host-side integer arithmetic; 16 rows per token):

```
ctx[0] = token; ctx[1..2] = the two predecessor tokens in this sequence
an EOS (248044) at or before a position replaces it and everything older;
a missing predecessor reads as EOS; the token's own EOS does not cut its window;
embedding (image) positions substitute ple.image_token_id 248056

for n in 2..3:                     # n = n-gram width
    mixed = XOR over j<n of (ctx[j] * ple.layer_multipliers[j])   # uint64
    for g in 0..7:
        head = (n-2)*8 + g
        row[head] = mixed % ple.head_vocab_sizes[head] + ple.head_offsets[head]
emb = concat(per_layer_token_embd[row[0..15]])       # 16 x 160 -> 2560
```

So heads 0-7 hash the **bigram** and heads 8-15 the **trigram**; heads 0-7 read
the first eight head ranges, 8-15 the last eight. 2560-wide `emb` then:

```
key   = rms_grouped(ple_key   . emb, ple_norm_key)     # 10240 = 4 streams
value =                   ple_value . emb              # 2560
query = rms_grouped(res, ple_norm_query)               # 10240
s     = sum over the 2560 dims of key*query, / sqrt(2560)     # one value/stream
gate  = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))         # signed sqrt, per stream
gated = value broadcast to 4 streams, * gate           # 10240
conv  = silu(depthwise causal conv(rms_grouped(gated, ple_norm_conv),
                                  ple_conv1d, dilation=3, kernel=4))
res   = res + gated + conv
```

`ple_norm_{key,query,conv}` are `[2560,4]` gammas; the conv needs the previous 9
positions of its normalised input per sequence (state, see §5).

**Measured cost** (`make ple`, the real table, 14930 real tokens, one thread):
5.1 µs/token with a cold row cache and 2.2 µs with a warm one, which is **0.007 %
of a 73 ms decode step cold and 0.003 % warm**. This is 16 random 90-byte rows
out of 26.8 GiB, so the work that paid was latency and translation, not
bandwidth: they want 1440 B and move 32 cache lines, **1.42x** the wanted bytes
at the bus - a wider page cannot reduce that, 64 B is the line - while they sit
in 10 distinct 4 KiB pages, which is 10 page-table entries per token against
**1.00 at 2 MiB pages**, and that is what the staging arena's transparent huge
pages buy (`docs/DESIGN.md` §3). Sorted, deduplicated reads issued at token-id
time and collected before layer 1, plus a bounded row cache that holds rows
*encoded* rather than decoded, keep the rest off the critical path. Deduplication
finds nothing on natural text (all 16 heads hash to distinct rows), and the row
cache serves 36.8 % of a first pass and 97.8 % of a second, so what it pays for
is revisiting a context, not shared heads. Thread scaling, each thread with a
private cache: 6.7 / 24.4 / 39.8 / 68.3 Mrows/s at 1 / 4 / 8 / 16 threads,
falling to 58.1 at 20 as the efficient cores join and the 30 MiB L3 is shared
with the rest of the step.

### 6.3 GDN block (36 layers)

```
qkv = attn_qkv . xn                     # 10240 = q 2048 | k 2048 | v 6144
z   = attn_gate . xn                    # 6144, the output gate
beta  = sigmoid(ssm_beta . xn)                                  # 48
alpha = ssm_a * softplus(ssm_alpha . xn + ssm_dt)               # 48, log-decay
x   = concat(conv_state[3 x 10240], qkv)          # 7 x 10240
conv_state <- last 3 rows of x
qkv = silu(depthwise causal conv(x, ssm_conv1d))   # kernel 4, per channel
q,k = l2_norm each 128-wide head of the first 4096 channels   # eps = 1e-6
v   = last 6144 channels
for head in 0..47:                       # heads 3g,3g+1,3g+2 share k-head g
    S <- S * exp(alpha[head])                                   # S: 128x128
    d  = (v[head] - S . k[head]) * beta[head]
    S <- S + outer(k[head], d)
    y[head] = S . q[head]                                        # scale = 1
y = rms_norm(y, ssm_norm) * sigmoid(z)    # the one numeric difference from
                                          # Qwen3.5's GDN: sigmoid, not silu
out = ssm_out . y                          # 6144 -> 2560
```

`y[head][j] = sum_i S[j][i] * q[i]`, `d[j] = (v[j] - sum_i S[j][i] * k[i]) *
beta`, which is the upstream `gated_delta_net` kernel's row-major form
(`ggml/src/ggml-cpu/ops.cpp:11230`). The state is f32 and per sequence.

### 6.4 QSA block (12 layers)

Indexer first (this is what makes the layer sparse):

```
k_raw = index_k_proj . xn                                # 128, cached raw
pooled[b] = rms_norm(mean of the ratio=4 raw keys in block b)  # then RoPE at the
                                                # block's first member's position
q = rope(rms_norm(index_q_proj . xn, index_q_norm), pos)   # 4 heads x 128
score[b] = (1/sqrt(128)) * sum over the 4 heads of relu(q . pooled[b])
sel = top_k(score, 2048/4 = 512 blocks) * 4 cells  ++  the 3 tail cells
```

Scores are **rectified and summed unweighted**, i.e. the lightning indexer with
all head weights fixed at `1/sqrt(128)`; pooled keys are pooled *before* norm
and RoPE, and the cache row stores `raw | pooled`.

Then ordinary GQA attention restricted to `sel`:

```
qg  = attn_q . xn                         # 12288, [q|gate] interleaved per head
q   = rope(rms_norm(qg.q_head, attn_q_norm), pos)   # 64 of 256 dims rotate
k   = rope(rms_norm(attn_k . xn, attn_k_norm), pos) # 2 kv heads, cached
v   = attn_v . xn                                    # cached
y   = softmax((q . k^T)/sqrt(256) restricted to sel) . V    # 24 heads, 2 kv heads
y   = y * sigmoid(gate)
out = attn_output . y                                  # 6144 -> 2560
```

Upstream also Hadamard-rotates q/k/v around a quantized KV cache, and this build
quantizes one: it writes the cache after the norm and the rotation, which is where
the values that are actually compared live, but it does not rotate them first. The
decorrelation a rotation buys is worth 0.002 nats here - 0.294 against the 0.292 of
the same build keeping f32 keys and values, over the 193 positions of the wide
corpus where the reference engine is confident (docs/DESIGN.md §5) - so the rotation
is the cheaper thing to leave out.

Two things about how that restricted softmax is walked. The cells of `sel` are
emitted in ascending order rather than in the heap order the indexer produced them
in, which is the same set of terms summed in another order; and the walk is over a
KV group rather than a query head, so one pass over `sel` serves the twelve query
heads that share it, carrying each head's exponent base and sum of weights in
registers so that no score array exists. A pass per group is two units of work
against nineteen workers, so `sel` is chunked and the chunk partials are folded
afterwards, which is the flash-decoding split rather than a tile.
`QE_ATTN_FUSED=0` or `--attn-head-walk` selects the per-head walk the engine used
before this, and docs/DESIGN.md §5 has what each side costs.

### 6.5 MoE (all 48 layers)

```
logits = ffn_gate_inp . xn                 # F32, 512
w      = softmax(logits); keep top 10; renormalise the 10 weights to sum 1
y      = sum_i w_i * down_i(silu(gate_i . xn) * (up_i . xn))
y     += sigmoid(ffn_gate_inp_shexp . xn) * down_shexp(silu(gate_shexp . xn)
                                            * (up_shexp . xn))
```

Routed experts are 10 of 512 with per-layer dtypes from §3; `qe_moe_layer()` /
`qe_moe_forward_multi()` (grouped over a window of T rows) implement exactly
this, and the grouped form is a measured byte saving on real routing:
`docs/DESIGN.md` §routing.

## 7. Where the bytes live

Every figure in §5 and §6 assumes a weight read costs memory time. On this host
that is only true if the bytes are in memory, so the loader puts all of them
there at startup:

| | |
|---|---|
| what | 87.2 GiB across 3 volumes, read once sequentially into a private anonymous arena, then `mlock`'d |
| cost | ~88 GiB of pinned memory, and one sequential pass over the file, paid once per process. The pass is deliberately not timed: the only reported facts are which backing holds the bytes and how much of them a previous run left behind |
| pages | THP by `madvise`, on by default: 68.4 GiB of `AnonHugePages` observed during an 87 GiB load, no compaction stall, no root, nothing reserved away from the machine |
| residency | `ulimit -l` is unlimited, so `mlock` takes it; `Unevictable` measured 91.7 GiB with `MemAvailable` still 35 GiB |
| after startup | zero reads of the checkpoint file. Not "fewer": zero |
| GEMV from the arena | 80.0 GB/s at 16 threads (79.6 at 20, 56.5 at 8) over a 512 MiB span of `token_embd.weight`, i.e. 13.7 tok/s and 19.9 tok/s at 2.76 accepted tokens per step |

Why not the alternative: `/mnt/data` is ZFS with `zfs_arc_max` fixed at 1 GiB
against this 87 GiB file, so a demand-paged `MAP_SHARED` mapping is not a
cache-sized-for-the-model arrangement, it is a seek per expert row forever.
Demand-paged mapping therefore survives only for metadata-only opens
(`--report`), and `--stage huge` (a root-reserved 2 MiB pool whose arena files
outlive the process) stays available for operators who want restart warmth
rather than a second read. Staging is STRICT on every serving path: `auto` and
`ram` either get the checkpoint resident at launch or the engine does not start,
because a server that starts half-loaded answers its first token in minutes.

## 8. Strata techniques: status on this host

Reference: https://github.com/Niko1221/Strata (documented techniques for an
MoE on spinning disks). Status against this repository:

| technique | status |
|---|---|
| grouped multi-row expert pass (`run_split_multi`) | **implemented** as `qe_moe_forward_multi()`, validated against real routing (`make route`, `docs/DESIGN.md`) |
| profile-ranked hot-expert pinning | **measured, deprioritised**: the hottest 128 of 512 experts carry only 45.6% of selections and cross-layer persistence measured 0.19 vs a 0.20 independence baseline, so the ~4.4 GiB fixed hot set (projections, `hc_*`, routers, shared experts, embeddings) is the better pin target than the pool |
| zero-copy when `mincore` says an expert is resident, stage otherwise | moot: `--stage ram` is the whole-model version of this, and it is the default |
| `MADV_RANDOM` on the PLE table, `issue()`/`collect()` row reads, sorted+deduped, bounded row cache | **implemented, except the one part that cannot apply here**: the sixteen ids are sorted and deduplicated, `qe_ple_issue()` starts the row reads ahead of the decode and `qe_ple_collect()` lands them in the embedding (`src/ple.c`, an 8-deep FIFO), and a bounded direct-mapped cache holds rows across tokens. `MADV_RANDOM` is the technique that is moot rather than missing - it suppresses page-cache readahead on a file-backed mapping, and the table is served from a private anonymous arena, where there is no readahead to suppress. 5.1 µs/token cold, 2.2 µs warm over the real 26.8 GiB table = 0.007% of a step (`make ple`) |
| `posix_fadvise(WILLNEED)` in 128 KiB steps ahead of the router look-ahead, gated on page-cache occupancy | not applicable: no page cache is involved in serving, the arena is private and pinned |
| huge-page arena guarded by `/proc/buddyinfo` | `qe_alloc_backed()`: THP-by-madvise by default (measured above), `MAP_HUGETLB`/hugetlbfs under `--stage huge` when root reserves a pool; `HugePages_Total: 0` here, so the default path is the one that runs |
| one worker per physical core, host thread joins the drain and is pinned, 20 ms spin then condvar, atomics on separate cache lines | implemented in `src/mem.c` / `src/moe.c` |
| verify windows up to 8 rows | supported by `qe_moe_forward_multi()` and driven by the head at `--mtp`: the expert stage of a nine-row window costs 172.6 ms against a step's 34.5, because the rows route apart and grouping only takes back part of it, which is most of why a window costs 4.14x a step (docs/DESIGN.md §4) |

## 9. Where each number came from

- The checkpoint itself, via `tools/gguf_inventory.py`, `tools/gguf_meta.py`,
  `bin/qe-serve --report`, `bin/bench_gemv`, `bin/bench_route` (all re-runnable).
- `ref/` holds the upstream quantisation references this repo's kernels were
  written against (`ggml-common.h`, `ggml-quants.c`, `ggml-x86-quants.c`) and
  `gguf-quants.py`, which is the independent dequantisation oracle used to
  confirm the block layouts byte for byte.
- Upstream architecture references consulted for §6: llama.cpp
  `src/models/qwen4exp.cpp` plus the `GGML_OP_GATED_DELTA_NET` and
  `GGML_OP_LIGHTNING_INDEXER` CPU kernels in `ggml/src/ggml-cpu/ops.cpp`,
  `transformers:configuration_qwen4_exp.py`, vLLM and SGLang's `qwen4_exp`
  configs, and the GGUF converter's own docstrings. The model file is the
  authority wherever they disagree, and §1-§5 are read from the file rather than
  from any of them.
- Network access from this host works through the `fetch` tool (`curl` and `wget` are
  installed but are not the tools an agent here is allowed to run); the upstream paths
  above were re-fetched while writing this, so §6 is a transcription of current
  upstream code, not of memory.
