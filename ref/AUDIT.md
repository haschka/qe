# Forward-pass audit vs upstream (qwen4exp.cpp + ggml-cpu ops.cpp)

Reference files (fetched 2026-11-11, upstream master):
- `ref/qwen4exp.cpp` - the graph definition (1512 lines)
- `ref/ggml-cpu-ops.cpp` - ggml CPU ops (rope 6164-6460, ssm_conv 9923,
  gated_delta_net 11117, lightning_indexer 12434)
- `ref/delta-net-base.cpp` - build_delta_net_{autoregressive,chunking,fused}
- `ref/ops.cpp`, `ref/vec.h` - the ggml op bodies that graph lowers to (l2_norm,
  sum_rows, ssm_conv, gated_delta_net) and the simd primitives they are written
  against. This is where "does the epsilon go under the root" is settled, and it
  has been the answer to that question twice.

## Confirmed matches (qe already right, do not "fix")

- **rope**: `theta_scale = powf(freq_base, -2/n_dims)` with n_dims = 64, and
  `rotate_pairs(n_dims, n_dims/2, ...)` for MROPE/IMROPE/NEOX => split-half pairs
  (p, p+32); channels 64..255 copied. `qe_rope_init/apply` matches exactly.
  Sections [11,11,10,0] do not change theta for text (all three axes share one
  position id), so they only confirm m-rope pairing.
- **ssm_conv tap order**: `sumf += s[i0 + i1*ncs] * c[i0 + i1*nc]` with the window
  starting at the current token => `c[0]*x[t-3] + c[1]*x[t-2] + c[2]*x[t-1] + c[3]*x[t]`,
  kernel stored [channel][tap] tap-fastest. `gdn_block` ring `(pos+t)%3` matches.
- **PLE conv**: `out[c,t] = sum_k w[k,c]*x[c,t-(K-1-k)*dil]`, dil = ngram_size = 3,
  history 9; `w[k]` is element (k,c) of a [4,10240] tensor = tap-fastest.
  `ple_block` ring `(pos + t*3) % 9` matches.
- **delta rule order**: `S *= exp(g)` (scalar g per v-head, `g = ssm_a * softplus(mm+ssm_dt)`,
  ssm_a = -exp(A_log)); then `pred = S*k`, `d = (v-pred)*beta`, `S += k (x) d`, `y = S*q`.
  `delta_task` computes pred against the *decayed* state (`pred*ea`) => matches.
- **GDN output gate**: rms_norm per 128-wide head with the one shared 128-entry
  `ssm_norm` gamma, times **sigmoid**(z) (qwen3.5 uses silu). Matches.
- **HC mix/combine**: grouped rms over one stream of 2560 with [10240] gamma;
  `lo = silu(mm(down,xn)/hc)`; `gate = mm(up,lo)`; `mixed = mean_s(xn*sigmoid(gate))`;
  `inject = mm(inject_w,xn)/hc`; combine `res += block*2*sigmoid(inject)`; head mixer
  passes no inject. Matches.
- **MoE**: softmax over all 512, top-10, renormalise the 10 to sum 1, weighted sum of
  `down_i(silu(gate_i x)*up_i x)`, plus `sigmoid(mm(ffn_gate_inp_shexp,x)) * down_shexp(...)`.
  No `expert_weights_scale` in the file => 1.0. `src/moe.c` matches.
- **QSA**: `attn_q` is 24 rows of 512 interleaved [q|gate], q normed by attn_q_norm then
  rope, k by attn_k_norm then rope, v raw, kq_scale 1/sqrt(256), `y *= sigmoid(gate)`,
  then wo. Hadamard rotations only when the KV cache is quantized (f32 cache here => skip).
  Matches.
- **indexer**: raw k cached un-normed/un-rotated; pooled key = rope(rms_norm(mean(raw)),
  first-member pos); score = sum_h relu(q_h . k_pooled) / sqrt(128); below top_k blocks
  everything is selected => cannot explain short-prompt gibberish.

## Fixed after the audit

- **The delta read-out carries a scale, and it is not a no-op.**
  `ggml_compute_forward_gated_delta_net_one_chunk` ends a head's recurrence with
  `attn_data[j] = sum * (1/sqrtf(S_v))` (`ref/delta-net-base.cpp`), i.e. the read-out
  of a 128-wide state is divided by 11.31. The reasoning that had removed it was
  that the output is rms-normalised per head immediately afterwards
  (`forward.c:948`) so a constant cancels - true only of a normalisation with no
  epsilon, and `sqrt(mean + 1e-6)` is not one. The read-out is a contraction of the
  state against an *l2-normalised* q, so its mean square sits at the same order as
  that epsilon, which puts the norm in its linear regime rather than its
  scale-invariant one, and the scale comes straight through. With
  `f->gdn_scale = 1.0f/sqrtf(head_state)` (`forward.c:413`): 3.5607 nats/token and
  12 top-1s in 31, fluent English. With `1.0f` (`QE_VARIANT_GDN_NOSCALE`): 7.8071,
  4/31, noise. This was the bug - two months of gibberish and one character.
- **PLE conv read the wrong tensor.** `build_ple` is
  `hidden + gated + silu(conv(grouped_norm(gated, ple_norm_conv)))`: the conv's
  input is the *normalised* gated value, and only the residual term added beside
  it is the raw one. `ple_block` computed `pgnorm` and then convolved `pgated`
  instead - and stored `pgated` in the nine-row history, so one wrong tap
  propagated for three tokens. Tap order and ring indexing were already right;
  only the operand was. This is the strongest gibberish candidate in the file:
  PLE writes into all four residual streams at layer 1, and those streams never
  merge again, so every later layer inherits it. `QE_VARIANT_PLE_PRENORM` keeps
  the old reading reachable for the A/B.
- **L2-norm epsilon placement.** Added `qe_l2_norm_grouped_under`
  (`x / sqrt(sum(x^2) + eps)`, what a naive reading of `build_gdn_l2_norm` suggests)
  as `QE_VARIANT_L2NORM_ROOT`; the committed default is `qe_l2_norm_grouped`,
  `x / max(||x||, eps)`, which is what `ggml_l2_norm` in `ref/ops.cpp` computes -
  epsilon as a floor on the norm, never under the root. With the two bugs above
  fixed the two readings are indistinguishable (3.5607 against 3.5606), so the
  choice rests on the op, not the loss.
- **Tokenizer is not the problem.** `make tok` (cheap now that `bench_tok` opens
  with `QE_STAGE_MMAP`): 248319 tokens round-tripped losslessly and idempotently,
  0 failures, 28/28 text round trips exact, 276 special ids recognised, eos
  248046 / bos 248044. The 203 "re-encodes to a different segmentation" cases are
  ordinary vocabulary entries the merge order would not produce on their own, not
  control tokens. Encoding is therefore not what made the output noise.

## Settled empirically

Teacher-forced loss over text the reference engine produced itself, one load, one
variant per row, plus rows that switch a whole stage off. `/tmp/score10.log`, over
32 tokens of `/tmp/llama-ref.txt`.

| variant | reading | nll | ppl | top-1 | reading of the output |
| --- | --- | --- | --- | --- | --- |
| `0x01` | committed | **3.5607** | 35.2 | 12/31 | fluent English |
| `0x101` | delta read-out scale 1 instead of `1/sqrt(128)` | 7.8071 | 2458 | 4/31 | noise |
| `0x11` | QSA layers off | 5.6228 | 277 | 7/31 | noise |
| `0x21` | delta-net layers off | 14.8612 | 284545 | 0/31 | noise |
| `0x41` | experts off | 8.0307 | 3074 | 3/31 | noise |
| `0x09` | PLE off | 3.5206 | 33.8 | 11/31 | fluent English |

Ablations now rank correctly, which is the check that the trunk is real: every
stage the reference graph says this model has is worth 2 to 11 nats, and removing
anything is much worse than keeping it. Only the PLE does not pass that test yet
(see below).

Aggregate nll over a 64-token corpus is contaminated by positions the reference
itself cannot predict, so the numbers that decide things are per position against
the reference's own logprobs. Two corpora are worth naming because they disagree.

Over the 32 tokens above, on the seven positions where llama.cpp is confident:

    qe mean nll(want) 0.442   reference 0.452   delta -0.010
    qe argmax == the reference's wanted token on 7 of 7

Over a wide corpus - four documents, 1026 positions, 845 of them positions the
reference is confident about, of which the first 255 were scored
(`/tmp/ref-wide.txt`, `/tmp/llama-logprobs-wide.tsv`, `/tmp/score-wide.log`):

| variant | reading | nll | ppl | top-1 | nll(want) on the 193 confident positions |
| --- | --- | --- | --- | --- | --- |
| `0x01` | committed | 0.8828 | 2.42 | 199/255 | **0.292**, 189/193 same argmax |
| `0x03` | `QE_VARIANT_L2NORM_ROOT` | 0.8827 | 2.42 | 199/255 | 0.292, 189/193 |
| `0x05` | PLE conv on the pre-norm value | 0.9110 | 2.49 | 195/255 | 0.333, 184/193 |
| `0x09` | PLE off | 0.9444 | 2.57 | 194/255 | 0.367, 181/193 |

and this is `tools/cmp_oracle.py`; the aggregate row is only a summary.

1. **GDN grouped-head mapping: modulo.**
   `ggml_compute_forward_gated_delta_net_one_chunk` uses `iq1 = iv1 % neq1` and
   `build_delta_net_autoregressive` uses `ggml_repeat(k, s)`, which is modulo =>
   llama.cpp maps v-head h to k/q head `h % 16`. The einops GVA convention
   (`repeat_interleave`, `repeat(x,'b h l d -> b (h g) l d')`) is `h / 3`, and
   that is what `forward.c` used to do. Modulo is right by 5.9 nats and the whole
   difference between noise and a model. Committed as `QE_VARIANT_DEFAULT`;
   `QE_VARIANT_GDN_GVA_MOD` off selects the other reading and stays available for
   the A/B. The head counts behind it are not an assumption: `ssm.group_count` is
   16, `ssm.time_step_rank` is 48, and `attn_qkv` is 2560x10240 = `128*16*2 +
   128*48`, i.e. q and k really do have 16 heads while v has 48, and both q and k
   are indexed `h % 16` in `delta_task`.
2. **The PLE does pay for itself; the 64-token corpus said otherwise.** On the
   wide corpus (below) removing it costs +0.074 nats and 8 top-1s of 193, and
   feeding the conv the pre-norm value - which `build_ple` rules out on the face of
   the graph - costs +0.039 nats and 5 top-1s. On 32 tokens of prose both looked
   *better* than the committed reading (3.5206 and 3.3942 against 3.5607). That is
   the whole argument for scoring per position against the reference instead of in
   aggregate: an aggregate over a short corpus prefers deleting parts of the model.
   `QE_PLE_DEBUG=1` now prints the block's own arithmetic, and what it prints is
   consistent: per-stream scores of -20 to 0, gates 0.01-0.6, conv pre-activations
   centred on zero with about half the 10240 channels positive. One thing it does
   expose: `rms(gated)` is 1e-4 to 6e-3, straddling `sqrt(eps)` = 1e-3, so that
   `grouped_norm` is only half a normalisation here and the gate does not cancel
   through it - the epsilon mechanism the delta read-out above is the cautionary
   tale for.
3. **Variant bits are settled.** `QE_VARIANT_L2NORM_ROOT` is indistinguishable
   (0.292 both ways, 189/193 both ways), so the reference's floored form ships. The
   other two are worse where it can be measured and are off. What is left of this
   list is the four positions, out of 193, where the reference is confident and we
   disagree: 94, 118, 153, 189, every one of them with the reference's own nll(want)
   at 0.5-1.0, i.e. positions where it is barely sure either.
4. **The q8_0 attention cache is worth the 0.002 it costs.** Keys and values are
   stored by default as one byte per value with one scale per head of one cell, and
   the paired A/B is the same build run with `QE_KV_F32=1` over
   `/tmp/ref-wide.txt`, 256 positions, one load each way
   (`/tmp/score-kvq8.log`, `/tmp/score-kvf32.log`). The f32 side reproduces item 3
   exactly - 0.8828 aggregate, 0.292 on the 193 confident positions, 189/193, and
   the same 0.8827/0.9110/0.9444 for the three ablations - so the comparison is a
   control and not a coincidence. The q8_0 side reads 0.8875 aggregate, 0.294 on
   those positions, and its top-1 is the reference's wanted token on 190 of the 193:
   one more than fewer. Timing is equal inside the run (107.9 against 108.0 ms/token
   median, five variants each; four-step stage profile 104.3 against 105.3), because
   at 256 tokens of context the cache is not yet the read the token makes. What the
   default buys is state, not speed: 12.2 KiB per token against 48, and 275 MiB for
   a sequence at 8k context against 562. One implication for the reading of this
   number: upstream Hadamard-rotates q/k/v before quantising and this build does
   not, so the rotation is inside the 0.002 rather than being an addition to it.

## Cross-checking against the other engine

`llama.cpp 0.9.0-dev b12110` on this host **does** implement this architecture:
`/usr/lib64/libllama.so.0` carries `llama_model_qwen4exp` with the same method
names as `ref/qwen4exp.cpp`. Greping the `llama-cli` binary finds nothing
because it dlopens the impl libraries - that false negative cost a session.
Reference runs are sequential only, never with both engines resident (see
`tools/llama_reference.py` for the invocation, including the `--lazy-mode off`
that keeps the 26.8 GiB per-layer embedding table out of the file's I/O path).

The recipe, end to end. `tools/oracle_capture.py` owns the server for the whole
capture and kills it on every exit path, so "sequential" is a property of the
script rather than of somebody's attention; the two halves can be driven by hand
if the server is already up.

```sh
# 1. reference prose + per-position logprobs, one model load (tools/llama_logprobs.py
#    scores one prefix per position, so the corpus length is the time cost)
python3 tools/oracle_capture.py --model <vol1.gguf> --n 256 \
    --corpus /tmp/llama-ref.txt --oracle /tmp/llama-logprobs.tsv

# 2. qe on the same ids, per variant, one load; --gen writes greedy text too
./bin/bench_score --model <vol1.gguf> --threads 20 --fuse 1500 \
    --corpus /tmp/llama-ref.txt --tokens 32 --gen 32 --variants 0x001,0x003,0x005 \
    --logits 16 --logits-out /tmp/qe-logits.tsv

# 3. the two side by side. --variant is spelled the way bench_score prints it (%02x).
python3 tools/cmp_oracle.py --oracle /tmp/llama-logprobs.tsv \
    --qe qe=/tmp/score10.log --variant 0x01
```

What the server build will and will not do, so nobody rediscovers it:

- `echo` plus `logprobs` returns logprobs for *generated* tokens only, so a
  teacher-forced reference costs one request per position; that is why
  `llama_logprobs.py` posts growing token-id prefixes instead of asking for the
  sequence once.
- `top_logprobs` is honoured on `/v1/completions` only (not `/completion`) and is
  capped at 4 by the server, which is enough to tell "we disagree" from "we are a
  different distance from the same distribution".
- `--embedding --pooling none --embd-normalize -1` **crashes** this architecture at
  `ggml-backend.cpp:477`, "tensor read out of bounds", so there is no hidden-state
  oracle from llama.cpp and no per-tensor differencing: the only channel out of the
  reference engine is its logprobs and its text. Do not spend a session on it.
- Greedy text must be scrubbed on the way to anywhere: `tr '|!' '!!'`. The tools
  scrub by default (`--scrub`), and the engines tokenise a bang identically, so the
  two sides keep seeing the same string.

PLE instruments, in case it becomes the suspect again: `QE_PLE_DEBUG=1` prints the
block's own scores, gates and term magnitudes per token (`src/forward.c`,
`ple_block`), and `tools/ple_hash.py` re-derives the n-gram row ids from the
checkpoint's own `ple.layer_multipliers` / `head_offsets` / `head_vocab_sizes`
keys, independently of `src/ple.c`, for `--tokens 11,279,264,2018`.

## Tensor facts from the checkpoint (tools/gguf_inventory.py -> /tmp/tensors.tsv)

- GDN layer: attn_qkv 2560x10240, attn_gate 2560x6144, ssm_conv1d 4x10240,
  ssm_alpha/ssm_beta 2560x48, ssm_a/ssm_dt 48, ssm_norm 128 (shared gamma),
  ssm_out 6144x2560.
- QSA layer: attn_q 2560x12288, attn_k/attn_v 2560x512, attn_output 6144x2560,
  attn_{q,k}_norm 256, indexer.q_proj BF16 2560x512, indexer.k_proj BF16 2560x128,
  indexer.{q,k}_norm 128. No indexer output projection and no per-head score
  weight, so every head's rectified score is summed unweighted - and no Hadamard
  rotation key either, so `self_{k,v}_rot` stay null and the dense path is plain
  rope into an f32 cache.
- HC: hc_*_down 10240x320, hc_*_up 320x10240, hc_*_inject F32 10240x4, hc_*_norm 10240;
  output_hc_{down,up,norm} same shapes, no output_hc_inject.
- MoE: ffn_gate_inp F32 2560x512, ffn_gate_inp_shexp F32 2560 (1-D),
  ffn_{gate,up}_exps 2560x640x512, ffn_down_exps 640x2560x512, shexp 2560x640/640x2560.
  Per-layer dtype exceptions: ffn_down_exps Q8_0 at layers 2,4,30,46,47 (IQ4_NL else);
  ffn_{gate,up}_exps IQ4_XS at layer 2 (IQ3_S else).
- PLE (layer 1 only): ple_key 2560x10240, ple_value 2560x2560, ple_conv1d 4x10240,
  ple_norm_{key,query,conv} 10240; per_layer_token_embd IQ4_NL 160x320001536.
- output.weight Q6_K 2560x248320, token_embd Q8_0 2560x248320.
