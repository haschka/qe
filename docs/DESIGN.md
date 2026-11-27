# qe: measured design budget for Qwen3.8-Flash-Next on this host

Everything below was measured or derived from this machine and this checkpoint,
not taken from a spec sheet. It exists because the design space here is
constrained by two hard numbers, and most plausible optimisations do nothing
to either. For what the checkpoint *is* - tensor by tensor, and the forward pass
the numbers here are a budget for - see [ARCH.md](ARCH.md).

## 1. Host

| | |
|---|---|
| CPU | Intel Core Ultra 7 265 (Arrow Lake-S), 20 cores, no SMT |
| Board | ASRock Z890 Pro RS WiFi White, BIOS 3.24 - a desktop platform, so the DRAM data rate is a firmware negotiation, and it reads `Speed: 5600 MT/s` / `Configured Memory Speed: 5600 MT/s` on both populated slots (`dmidecode -t 17`, root): the modules are running at their rated speed and there is nothing to enable. It is also the only platform quantity a token can see: one NUMA node, no device in the request path, no AVX-512 to gain (ideas.md §3.5) |
| ISA | AVX2 + FMA + F16C + **AVX-VNNI** + BMI2. **No AVX-512** - fused off in silicon (`-march=native` resolves to `arrowlake-s`, `-mavx512f` disabled) |
| RAM | 128 GiB DDR5-5600 (2 slots, rated and configured both at 5600 MT/s), single NUMA node. Two 64-bit channels at 5600 MT/s is **89.6 GB/s** of theoretical read, and the measured rates below are 85-95 % of it |
| L3 | 30 MiB |
| **DRAM read bandwidth** | **76-79 GB/s** for the shape a decode step has (Q8_0 rows, 19 threads, 512 MiB span, `make alu`); **85.5 GB/s** for a flat 64 B-aligned read on the eight performance cores. Those are 85-88 % and 95 % of the 89.6 GB/s this host's two channels at 5600 MT/s can carry. An older figure of **113.8 GB/s** is quoted below and in `--report`: it is 27 % above that physical peak, so it was never a rate this platform could produce at any access pattern, and §1.2 is where that is settled |
| **Model placement** | **all 87.2 GiB resident in RAM from the moment the engine starts** (§1.1). The storage device takes no part in serving a token, so its bandwidth is not a quantity this design measures or optimises |
| **CPU governor** | **`powersave`** (`min_perf_pct=15`) - see §6 |

Kernel `5.x`/Gentoo, THP=`madvise`, `nr_hugepages=0`, `swappiness=60`.

### 1.1 The placement decision, stated once

`qe_open()` reads every byte of the checkpoint exactly once, sequentially, into a
private anonymous arena, asks for 2 MiB transparent huge pages, and `mlock`s it.
Serving a checkpoint costs ~88 GiB of pinned memory and one pass over the file;
after that the file is never read again, and there is no page-cache strategy, no
`posix_fadvise`, no fault-around, no `mincore` check and no re-read anywhere in
the request path. That pass is not timed and no rate is reported for it: it is a
one-time cost on a device this engine has already decided not to depend on, and
publishing a number for it turns a startup into a storage benchmark. `--stage
mmap` is refused for serving and exists only for metadata-only opens
(`--report`).

That makes the storage path a solved, one-time cost, which is the point: every
remaining optimisation is in token ingestion and generation, and each of them is
measured against memory, not against a device.

| measured, from the resident arena | |
|---|---|
| Q8_0 GEMV, 512 MiB span, real row shape | **80.0 GB/s** at 16 threads (79.6 at 20, 56.5 at 8) |
| decode ceiling at 5568 MiB/step | **13.7 tok/s** |
| with a drafter, at the acceptance the resident head reaches | the byte model said **19.9 tok/s**; the measured answer is **9.5 tok/s**, the plain step loop's, because a verified window costs 4.14x a step and pays 1.6-1.9 ids for it (§4, `/tmp/f38_mtp_short.log`) |
| memory left for state and clients after staging | `MemAvailable` 35 GiB, `Unevictable` 91.7 GiB |
| n-gram PLE row gather, 16 rows per token | **5.1 µs/token cold**, 2.2 µs warm over the real 26.8 GiB table = **0.007 % of a 73 ms step** (`make ple`) |
| dense projections, a step carrying 16 tokens | **489 → 33 MiB** and **12.1 → 5.2 ms per ingested token**, counted as last-level-cache misses at 16 threads (§12) |
| prompt ingestion, in the engine | **92.4 ms per ingested token one at a time, 25.2 with every projection of 1 MiB and above swept once for a 512-row chunk, 16.3 with the integer walks and the flat dense copy on top of that** (598-token prompt, default width 512; the swept form is asserted by picking the same next token, and the integer forms by picking it and the eight after it (§12, `make forward`, `/tmp/f35_wide.log`)) |
| router bytes if selection reads a bf16 shadow | **240 → 120 MiB/token**; the ten picks were identical over 96000 real layer-token pairs (§9) |
| routed experts, a step carrying 16 tokens | **355.5 → 163.4 MiB/token** under real routing, 54.0% fewer (§9) |

What is therefore worth optimising, in order of what it buys: fewer bytes per
token (grouped multi-row expert execution, §9; speculation, §4, which is measured, wired, and on
at the one draft length that pays - **1.19x** at k = 1 with a head resident against the 0.75x at
the k = 8 it used to default to), fewer bytes per
*ingested* token (chunked dense walks, §12 - the largest measured win here at
kernel level, and at engine level an ordering result rather than a blocking one,
which is what §12's last section is about, because a step that carries T tokens
has no reason to pay for its weights T times), the router's width (§9), and nothing
else at the kernels: §1.2 measures the shipped GEMV at 96-99 % of a walk over
identical bytes with the arithmetic deleted, so there is no rate left to recover
there. What is **not** worth optimising any more: the PLE row
gather, which this table settles at 0.007 % of a step, and whose remaining cost is
10 distinct 4 KiB pages per token (1.00 at 2 MiB) rather than bytes.

### 1.2 Is the step ALU-bound? No, and here is the instrument that says so

The question is the right one to ask of a CPU engine: a decode step reads 5568 MiB,
and if the memory fabric were the limit then no amount of smarter dequantisation
helps, while if the arithmetic were the limit then the Q8_0 inner loop - four
convert-and-FMA pairs per 32 weights, over 8-byte loads - is where the token goes.
`bin/bench_alu` answers it by walking the *same bytes* several ways over the same
pool and the same span, at a `--cpus` set chosen to isolate a core class:

* the shipped Q8_0 GEMV;
* the identical walk with the arithmetic deleted (a read of exactly the same bytes);
* a 64 B-aligned, byte-sliced flat read - the "what is this machine actually able to
  do" probe, with 8 loads per 256 B and no quantisation format in sight;
* and the candidates (§1.2's last table);
* and the same bytes walked once for T tokens (`multi1`, `multi2`, `multi4`), which is
  the shape a chunked prefill wants: one dequantised weight run feeding T accumulators,
  so the pass costs the same and the token costs a Tth as much. These are the only rows
  in the instrument whose whole purpose is a ratio - what sharing a dequantisation is
  worth - and §12 has the answer and its price.
* and `shipped2`, `shipped4`, which are not candidates: they call the kernel that came
  out of them (`qe_gemv_q8_0_multi`) through the same slice, so the ratio printed for the
  sketch can be checked against the ratio the engine would get.

Spans are chosen so the regime is knowable: 512 MiB is DRAM for every thread count,
8-32 MiB fits the 30 MiB L3, and `--inner N` repeats the span N times inside one timed
pass, because below ~32 MiB the span costs less to walk than the dispatch barrier it
is measured across. Every walk is checked before it is timed (the shipped kernel
against a double-precision reference, the candidates against the shipped kernel,
bit-identically where they claim to be), which is what caught a synthetic-data bug
that had deflated every number in this section by ~15%: composing a Q8_0 block as
`{qs, d}` instead of `{d, qs}` puts random weight bytes where the fp16 scale lives,
and denormal and NaN scales cost microcode assists. Rates are best of three, at the
governor §6 complains about.

**At the shape a decode step has, the kernel is 96-99% of an identical-bytes read.**
512 MiB span, 19 threads (`./bin/bench_alu --span 512 --reps 3 --cpus 0-18`):

| walk, same bytes | GB/s | as a share of the read |
|---|---|---|
| same bytes, arithmetic deleted | 78.9 | 100% |
| 64 B-aligned flat read | 76.2 | 97% |
| **shipped Q8_0 GEMV** | **75.9** | **96%** |
| two rows in flight (MLP) | 74.6 | 95% |
| vpdpbusd / AVX-VNNI dot | 77.1 | 98% |
| bf16 (85.2 GB/s, 42.6 Gweights/s) | 85.2 | - |

Per core the picture is different, and that is what makes the first table mean what
it says (512 MiB, same instrument):

| cpus | GEMV | read | flat read |
|---|---|---|---|
| `0` (one P core) | 17.1 | 23.4 | 31.1 |
| `0-7` (8 P) | 59.3 | 63.4 | **85.5** |
| `8-19` (12 E) | 60.8 | 63.4 | 70.2 |
| `0-18` (19, the engine's width) | 75.9 | 78.9 | 76.2 |

One core alone has ~1.8x of ALU headroom (17.1 against a 31.1 flat read) and the
eight performance cores together stream 85.5 GB/s - 12% more than nineteen threads
manage - but a GEMV on eight P cores gets 59.3, so the surplus is unreachable: it is
queue slots and memory-side limits that bind, not the cores. Twelve efficient cores
reach the same 63.4 GB/s read as eight P cores for 12 threads against 8, which is the
same conclusion one layer down. The E cores do cost streaming bandwidth the P cores
would not (76.2 against 85.5), and a decode step still runs faster with them, because
one core cannot fill the fabric.

**ALU does bind, but only for data that is already close, and nothing in the decode
path is close** (19 threads, `tools/sweep_alu.sh 0-18`):

| span | GEMV | read | GEMV as a share of the read |
|---|---|---|---|
| 512 MiB (DRAM) | 75.9 | 78.9 | 96% |
| 32 MiB (L3) | 240.6 | 355.9 | 68% |
| 8 MiB (L3) | 239.8 | 1035.3 | 23% |

A token reads 5568 MiB into a 30 MiB L3, so the 23% row describes a regime this
engine never enters. That is the whole argument for why the kernel ideas below are
dead and for why the remaining levers are bytes per token (§4, §9) and per-stage
efficiency (§14), in that order.

**Measured and rejected, so that nobody spends an afternoon on them again:**

| idea | result | instrument |
|---|---|---|
| AVX-VNNI `vpdpbusd` integer dot | 10.6 against 17.1 GB/s on one core; 42-98% of the read walk across spans - never better | `./bin/bench_alu --kernels gemv,vnni` |
| Split layout: weights-only rows plus a dense fp16 scale array | 70.0 against 75.9 at 512 MiB, although +43% at 64 MiB - a cache-regime idea, and the decode path has no cache regime | `./bin/bench_alu --kernels gemv,gemvsplit,readsplit` |
| Two rows in flight for more memory-level parallelism | 74.6 against 75.9, and worse everywhere else | `./bin/bench_alu --kernels gemv,gemv2` |
| Pinning workers to cores by `cpu_capacity` | 2-3x the dispatch handshake (§12) | `QE_PIN=1 ./bin/bench_pool auto 120` |
| Non-temporal (streaming) store for the recurrent state | 72.2 GB/s against the ordinary store's 74.0 at 512 MiB, 75.2 against 77.8 at 128 MiB - and there is no fetch to skip, because the same task reads the row first (§14) | `./bin/bench_alu --span 512 --kernels read,state,statent` |
| Software prefetch | 6-67% regression; the code was deleted | §12 |
| A narrower weight walk for a decode step: 16 of the 19 threads, where one kernel's walk of the same staged bytes reads 78.3 GB/s against 73.7 at 19 | 1.07x **slower** a step at 16 threads, 1.10x at 15, 1.17x at 13, logits byte for byte identical at every one of them, and 10.66 against 11.27 tok/s of generation in the load that shipped the knob. The kernel's number is real and is the kernel's; a step pays ~800 dispatches of it (§12 is the other half of why) | `make forward`'s width leg, `./bin/bench_gemv <model> 16 360`, ideas.md 7.26 |

One number in §1 does not survive this section: the **113.8 GB/s** DRAM peak is older
than `bench_alu` and no instrument in this repository reproduces it - and it cannot, because
the modules read `Configured Memory Speed: 5600 MT/s` on both slots, two 64-bit channels at
5600 MT/s is 89.6 GB/s, and the retired figure is 27 % above the hardware's ceiling. The highest
linear read rate measured here is 85.5 GB/s (eight P cores, 512 MiB, flat 64 B-aligned
walk), 95 % of that ceiling, and the rate a decode step actually runs into is 76-79 GB/s at the
engine's width, 85-88 % of it - which is the same wall `make gemv` finds from the other direction
at 80.0 GB/s on this checkpoint's real rows at 16 threads. Every ceiling in this file is now
stated as "which pattern, how many threads", because the pattern is worth 10 GB/s and
the thread count 9, and neither of those two is a firmware setting (ideas.md §3.5).

`general.architecture = qwen4exp`, `general.size_label = 512x56B`.

48 layers, hidden 2560, 24 Q heads / 2 KV heads @ head_dim 256, 262144 ctx,
512 routed experts with 10 active, 1 shared expert (ffn 640), full attention
every 4th layer (so 12 full-attention + 36 gated-delta-net layers),
hyper-connections (4 streams, low rank 320), a sparse-attention indexer
(4 heads, k=128, top_k 2048), and a PLE n-gram per-layer embedding table.

### Type table recovered from the file, not assumed

The bytes-per-block values were derived from the tensor offset chain across all
three volumes: for consecutive tensors, `off[i+1] == align_up(off[i] + bytes)`.
That reconstruction yields **0 chain mismatches** and an implied total that
matches `stat()` to within 1 MiB of an 87 GiB file.

| id | block_ne | bytes/block | bpw | type | used for |
|----|----------|-------------|-----|------|----------|
| 0  | 1   | 4   | 32.0   | F32     | router, norms, conv, dt |
| 8  | 32  | 34  | 8.5    | Q8_0    | attention/GDN, hyper-conn, shared expert, token_embd |
| 14 | 256 | 210 | 6.5625 | Q6_K    | lm_head |
| 20 | 32  | 18  | 4.5    | IQ4_NL  | PLE table, some expert down-proj |
| 21 | 256 | 110 | 3.4375 | IQ3_S   | expert gate/up |
| 23 | 256 | 136 | 4.25   | IQ4_XS  | expert gate/up/down (some layers) |
| 30 | 1   | 2   | 16.0   | BF16    | indexer projections |

An earlier draft of this analysis used a guessed table and got 66.75 GiB total
and the wrong kernel priorities. The chain proof is what caught it.

### What the file says the model is

Read with `./bin/qe-serve --report --model <volume 1>`; every number below is
what that prints, cross-checked against a second independent parser:

| fact | value |
|---|---|
| architecture | `qwen4exp`, name "Qwen3.8 Flash Next", size label 512x56B |
| volumes | 3, metadata only in volume 1 (67 keys), `split.*` in 2 and 3 |
| tensors / weight bytes | 1224 / 87.2 GiB |
| quant mix by bytes | IQ4_NL 52%, IQ3_S 36%, Q8_0 10%, IQ4_XS 1%, Q6_K 1%, F32 0.3% |
| biggest single tensor | `per_layer_token_embd.weight`, 26.8 GiB (30.7% of the file), IQ4_NL `[160, 320001536]` |
| expert pools | `ffn_{gate,up}_exps` 15.8 GiB each (IQ3_S), `ffn_down_exps` 18.9 GiB (IQ4_NL) |
| vocabulary | 248320 tokens, BPE (`tokenizer.ggml.model=gpt2`, `pre=qwen35`, 247587 merges) |
| token ids | eos 248046, bos/pad 248044, image 248056 |
| chat template | present, 9993 chars of Jinja with image/video handling |
| MTP / draft head | **a separate 2.66 GiB volume**, `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`: 34 tensors named `blk.48.*` (the trunk's `n_layer`), `nextn_predict_layers 1`, `eh_proj [5120, 2560]` Q4_K, `output.weight [2560, 248320]` Q6_K = 497.3 MiB of its 2728. Staged only when `--mtp` names it; §4 is what it measures, which is not what §4 used to predict |

The mix matters for one reason beyond curiosity: the PLE n-gram table is 30.7%
of the file but ~0% of the bytes per token (§3), so "how big is the model" and
"how much does a token cost" are two different questions here, and the honest
answer to the first one is that a third of the file is a lookup table.

## 3. The budget that matters: 5.44 GiB per decode token

DRAM bytes read to produce one token. This table is not typed in: `bin/bench_gemv
<checkpoint>` computes it from the tensor table on every run, attributing each
tensor by role - 10/512 of an expert pool, one row of `token_embd`, 16 rows of
the PLE n-gram table, everything else in full.

| component | MiB/token | share | dtype |
|---|---|---|---|
| Attention + GDN + indexer projections (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q`, `attn_output`, `indexer.*`) | 2786 | **50.0%** | Q8_0 (+ BF16 indexer) |
| Routed experts (10 of 512) | 1109 | 19.9% | IQ3_S / IQ4_NL / IQ4_XS / Q8_0 |
| Hyper-connections | 663 | 11.9% | Q8_0 |
| lm_head over 248320 vocab | 497 | 8.9% | Q6_K |
| Router logits (dense 512x2560 f32) | 240 | 4.3% | F32 |
| Shared expert | 240 | 4.3% | Q8_0 |
| PLE projections (`ple_key`, `ple_value`, `ple_norm_*`) | 34 | 0.6% | Q8_0 / F32 |
| PLE n-gram rows (~16 random rows) | 0.0014 | ~0% | IQ4_NL |
| `token_embd` (1 row) | 0.0026 | ~0% | Q8_0 |
| **total** | **5568** | **100%** | |

**This replaced a 5295 MiB figure, and it is worth naming both errors.** The old
number was 273 MiB light and was described as "5.29 GiB", which is a units
mistake on top of an arithmetic one (5295 MiB is 5.19 GiB). The missing bytes
were written off as "F32 router logits and BF16, both tiny": the F32 router
alone is 240 MiB per token, 4.3% of the step. Anything that divides by this
number moved by 5.2%.

The same `bin/bench_gemv` under ASan+UBSan against the checkpoint is clean,
which it was not before: the DRAM-bound phase asked for the rows of a 512 MiB
span (197379 at K=2560) and wrote them into a 32768-row output buffer, so it
died of SIGSEGV *only when pointed at the real file* - the synthetic benchmarks
use spans they were sized for, so no amount of `make fast` would have shown it.
Two lessons, both cheap: a fixed-size scratch buffer needs an assert next to
the row count that fills it, and a benchmark that has never been run against
the thing it measures is not evidence of anything.

**Consequences, stated plainly:**

1. **Ceilings, from the softest number to the hardest.** 19.5 tok/s at a 113.8 GB/s
   DRAM peak and 14.6 tok/s at a "typical" 85 GB/s are arithmetic with zero overhead,
   so neither is reachable - and §1.2 now says the first of them is not even a rate
   this host has ever been measured doing, the highest linear read measured here being
   85.5 GB/s on eight cores. What `bin/bench_gemv`
   actually measures on this file - six runs, because one run of this host is
   noise - is a Q8_0 GEMV over a 512 MiB warm span with the real K=2560 row
   shape at **73.6-78.7 GB/s, median 76.4**, i.e. **12.6-13.5 tok/s**, and a
   64 MiB stream read at **78.1-96.9 GB/s, median 89.0**, i.e. 13.4-16.6 tok/s.
   The spread is the `powersave` governor (§6), not the kernel. The figure this
   build budgets with is the measured per-type kernel mix summed over this
   table: **11.4 tok/s single stream** (`make fast`, 88.0 ms/token). A lone token
   does not read that mix: it reads a load-time IQ4_NL copy of the dense
   projections and a load-time Q8_0 copy of the head, which `make score`'s decode
   arms price at 1.16x and 1.02x of a step in the load that staged them and at
   **1.19x** together - 107.0 → 90.2 ms/token over 599 teacher-forced positions
   (`/tmp/i4g_ship.log`), i.e. **13.3-13.5 tok/s**, with `QE_I4_DENSE=lean` able
   to hold the mixers and the trunk's end blocks at Q8_0 and buy back two thirds
   of the loss for 8 ms of a step. The draft head's `output.weight` proved to be
   these same bytes at open, so one pair of arenas serves a draft row as well,
   and ideas.md §7.23 is the account of what each of them costs the loss.
2. **A token touches 6.7B weights, so "6B active" is roughly true in weight
   count** - 5.44 GiB at the 3.4-8.5 bpw this file actually uses. (These notes
   previously claimed ~10.5B, which was the byte count divided by an assumed
   4 bits per weight.) What the premise hides is not the count but the
   *shape*: 66.6% of the bytes are dense Q8_0 that no routing decision skips.
3. **The MoE is not the bottleneck for a single stream.** Routed experts are
   19.9% of traffic and already read only 10/512 of the pool, so no routing
   decision removes bytes for *one* token, and the ceiling stands. What changed
   since this section was first written is that the claim is now measured rather
   than argued, and it is narrower than it looked: over a **window** of T
   candidate rows, grouped execution reads 11-54% fewer routed bytes than T
   independent passes (§9), because real routing is over-lapped across adjacent
   tokens. Speculation and batched verify windows therefore do move the
   ceiling, through this term.
4. **The dense Q8_0 projections are the bottleneck** at 66.6% of bytes
   (projections alone 50.0%), and they are read in full by every single token
   regardless of routing. That is where
   bytes can actually be removed (quantisation choice), and where kernel
   quality decides whether we sit at 60% or 95% of the wall.
5. Nothing else - threading, SIMD width, KV layout - changes the ceiling. They
   only decide what fraction of it is realised.

## 4. Speculation: built, wired, measured, and on at one token

A bandwidth-bound step costs the same whether it emits one token or several, and that
sentence is the whole case for speculation. It is also the sentence that was wrong here, and
the measurement that replaced it is this section.

The head exists and is resident: `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, GGUF v3, 34 tensors
named `blk.48.*` (the trunk's `n_layer`), `nextn_predict_layers 1`, `eh_proj [5120, 2560]`
Q4_K and its own `output.weight [2560, 248320]` Q6_K - 497.3 MiB of its 2728. `--mtp` stages
it under the same rule as the trunk (`QE_STAGE_RAM | STRICT`, refusing a head whose width or
architecture does not match), and the pass runs it as `f->L[f->n_layer]` through the trunk's
own six stage functions: a draft row is one more layer of this model, not a second engine.
Drafting 32 ids through it moved none of the ids greedy says (`/tmp/f38_mtp_short.log`), so
correctness is not the open question. Cost is.

**A verified row is not free.** This is what §3's 5568 MiB buys when nine rows share it,
measured on the checkpoint at 23-token context:

| pass | 23-token prompt | 598-token prompt |
|---|---|---|
| one target step | 111.2 ms | 108.1 ms |
| one draft row | 14.5 (0.130 of a step) | 14.6 (0.136) |
| nine-row window, logits computed for every row | 460.3 (**4.14x** a step) | 481.2 (**4.45x**) |
| the same nine rows with no checkpoint kept | 324.3 (2.92x) | 338.4 (3.13x) |

The dense projection stage is the one part that behaves: 44.96 ms of a step against 56.47
of a window at the short shape, 42.11 against 55.69 at the shipping one, because nine rows
really do share those bytes. Everything else pays per row.
`block_core` is 106.03 ms of the window and ~80 ms of that is copying GDN state into the
slots that make a window rewindable (drop the boundaries and 1.32 steps of the window go,
and with them the rewind speculation is made of). `moe_ffn` is 172.64 against a step's 34.46,
because nine
rows route to nine different sets of ten experts and grouping takes back less than half of
that at nine rows. And ~97 ms is not billed to a stage at all: every row's logits get
computed, at ~7.2 ms a row, so rejected rows are paid for in full.

The byte table this section used to carry - `(5568 + draft_MiB x tok_per_step) /
tok_per_step`, predicting 19.0 tok/s against 13.1 with no draft - is withdrawn. Its one
non-bandwidth input was `tok_per_step`, and the measured value of that is 1.6-1.9, not the
2.76 the download's notes report:

| acceptance, measured | 23-token prompt | 598-token prompt |
|---|---|---|
| teacher-forced top-1 agreement with the target, 16 positions | 8 | 15 |
| chained acceptance, best arm | 52 % (k = 1) | 92 % (k = 1) |
| ids per pass, best arm | 1.52 | 1.92 exact, 1.85 as shipped |
| passes keeping 1, 2, 3, 4, 5+ tokens at k = 8, exact | 41 %, 24 %, 18 %, 6 %, 0 | 83 %, 58 %, 50 %, 33 %, 17-25 % |

Longer drafts really do reach further at the shipping width (4.00 ids a pass at k = 8) -
the window they pay for it with is 4.45 steps there, 481.2 ms against a 108.1 ms step,
with the 9-row checkpointed state 1.32 steps of that. The cost model is therefore
`cost = (window(k+1) + k x 14.6 ms) / ids(k)`, and it crosses the plain loop exactly once,
at the shortest draft there is:

| arm | 23-token prompt | 598-token prompt, exact | 598-token prompt, as shipped |
|---|---|---|---|
| plain step loop | 105.7 | 108.5 | 107.1 |
| head, k = 1 | 102.8 (**1.03x**) | 111.1 (0.98x) | **90.0 (1.19x)** |
| head, k = 2 | 178.3 (0.59x) | 119.8 | - |
| head, k = 3 | 204.8 | 122.1 | - |
| head, k = 4 | 252.1 (0.42x) | 130.4 | - |
| head, k = 8 | 459.8 (0.23x) | 220.0 | 143.1 (0.75x) |
| n-gram drafter, k = 2 | - | - | 102.4 (**1.05x**) |
| n-gram drafter, k = 8 | - | - | 117.4 (0.91x) |

Read the shape of that prompt before quoting the right column: `make head` feeds the same
sentence twenty-six times, which is the most repeatable 598 tokens that can be built, and
the head's 92 % first-draft acceptance on it is an upper bound that prose will not match.
What does not depend on the prompt is the two curves against each other: ids a pass rise
with k roughly three times faster than the window does at k = 1 and roughly a third as fast
past k = 4, so the ranking of the arms is the shape of a window and not the shape of the
text. Note also that an engine with no head is speculating anyway at the draft length it was
given: the n-gram hash drafts on 31 % of passes at k = 8 and costs 0.91x, and drafts on 10 %
at k = 2 and saves 1.05x.

Three readings from the same two loads settle how a draft row is made. The residual a draft
row reads is the one *before* the frontier: teacher-forced, back = 1 agrees on 8 of 16
positions and 17 of 32 shared ids against back = 0's 6 and 18 at the short shape, and 15 of
16 and 16 of 32 against 15 and 15 at the shipping one - where chained drafts seeded from
the frontier drafted **zero** tokens at k = 4 and at k = 8 alike, because a draft is made
before the position whose residual it would read. Warming the head's rows inside a prompt's
chunks is worth nothing at short context (56 % of passes keeping a single token either way)
and a great deal at the shipping one: the same k = 8 pass accepts 22 % of its drafts cold
against 39 % warm, and the 1.19x arm at k = 1 is an as-shipped number that exact arithmetic
never reaches (0.98x). What warming costs is visible in the same log's prompt leg, 19 ms per
input token with a head resident against 15.9-16.3 with none (`/tmp/f35_wide.log`), and it
buys back more than that in generation whenever the context is long enough to repeat itself.

Of the download's advice, one half survived contact and one half did not: the 1.91 GB Q4_K
head really is the right pick over the 2.79 GB Q8_0 one (bytes are everything here), but the
tensor that dominates it is a Q6_K `output.weight` walked at this host's 44-46 GB/s rather
than Q8_0's 74-83, which is 11 of its 14.6 ms per draft row; and the reported acceptance is
not reproducible at short context, which is the entire difference between the win at k = 1
and the losses at every k above it.

**Speculation's draft length is therefore decided by the drafter, at open, and only as long
as the caller did not say otherwise** (`QE_SPEC_K_AUTO`, which is what `qe-serve` starts
with): one token when a nextn head is resident, two when the n-gram hash is drafting, and
`--spec-k 0` for neither. A caller who asks for a length gets that length; `/v1/models`
reports the resolved one and says which drafter it belongs to.

What would change the answer is priced and unwired: logits computed lazily for only the rows
a window keeps (~50 ms of a 460 ms pass) and boundary copies made cheaper (~80 ms of it) are
together ~1.1x on a path that is 4.14x away from breaking even. The honest reading is that a
verify window pays when a batch is sitting in it - except that the batch which arrived is a
batch of *sequences*, and a window is nine rows of one of them, which a row map cannot express,
so `--batch N` refuses `--mtp` at startup rather than promising it (ideas.md §7.25).

## 5. Design decisions this forces

| decision | why |
|---|---|
| **All 87.2 GiB resident at startup** | §1.1: one sequential read, `mlock`, 2 MiB pages. Everything below assumes a weight read is a memory access |
| Huge-page-backed resident weights | PLE table is 26.8 GiB read as ~16 scattered rows/token; at 4K pages that is millions of PTEs and a page walk per access. 2 MiB pages cut page-table entries ~500x |
| Q8_0 GEMV is the kernel that matters | carries 66.6% of bytes/token |
| Flat `(experts, N, K)` expert layout, strided view, no gather | 3 concurrent 1.5 MiB expert streams/token/layer exceed what the prefetchers track unaided |
| Software prefetch ahead in the expert pass | **Rejected by measurement.** bench_moe: distances 8/24/64 rows cost 6-67% against issuing none, and expert warm-up was noise. Rows are contiguous inside an expert, which is what the hardware prefetcher is for |
| Grouped multi-row expert pass (`qe_moe_forward_multi`) | real routing is over-lapped, so a window of T rows reads the *distinct* experts of the window instead of `T x 10`: 11-54% fewer routed bytes at T=2-16 (§9). Break-even under uniform routing, which is the control `bench_moe` reports |
| Hot-expert pinning | **Not the win the literature implies here.** Hottest 128 of 512 experts carry 45.6% of selections and cross-layer persistence measured 0.19 vs a 0.20 baseline (§9); pin the 4.4 GiB of always-read weights instead |
| Speculation is per-request | §4 caveats |
| Thread count is a cliff, so the pool stops one core short of it | `make pool`: hot-dispatch median 1.6 us at 19 threads against 69 us at 20, worst 2.0 ms against 2.5 ms, and a token is ~800 dispatches - which is why the same binary takes 125.2 ms a token at 20 and 106.2 at 19 on that 64-position corpus (§12). Long passes barely notice, because a stall amortised over a 40-iteration round is nothing, and with rows shared equally 19 is now the faster width there too: five repeats of the expert round give 39.4-39.8 GB/s at 19 against 34.2-37.9 at 20, where under `cpu_capacity` weights the same two widths read 43.3 and 45.5. What does change at 20 is that the numbers stop lying - the old 20-thread sweep ranged 22.8-47.3 GB/s across five repeats, and equal shares narrow both widths to within 4 GB/s. Earlier readings of "8 beats 20", and of an ordering between 4/8/12/20 at all, were this |
| Sparse attention (indexer top_k=2048) behind a long-context threshold | below ~32k ctx the gather costs more than dense |
| **Attention cache in q8_0, keys and values alike** | the one part of a sequence that grows: 2 kv heads x 256 dims, twice, is 4 KiB per cell per layer in f32 against 1.03 KiB quantised - one byte per value and one scale per head of one cell, which is the granularity that survives because a head is RMS-normed before it is written. Over the 12 sparse-attention layers that is 12.2 KiB per token against 48; the indexer's keys are per token as well, so the whole footprint `qe_forward_state_bytes` reports is **19.7 KiB per token against 55.5**, and a sequence at 8k context is **270 MiB** of state instead of **557** (`make score` prints both), which is the number that caps how many sequences share the host. Written after the norm and the rotation, so the scale covers exactly the range the dot product reads. What it costs, on the 1026-position wide corpus against the reference's own logprobs: **0.294 nats against 0.292** for the same build run with `QE_KV_F32=1`, on the 193 positions the reference itself is confident about (`./bin/bench_score --corpus /tmp/ref-wide.txt --tokens 256`, with and without the variable), and no measurable time at the 256 positions that pass reaches - 107.9 ms/token median against 108.0 over five variants each, with the four-step stage profile 104.3 against 105.3. The bytes only become the speed when the context is long enough for the cache to be the read. Upstream Hadamard-rotates q/k/v before quantising and qe does not, which is inside that 0.002 rather than an addition to it |
| Demand-paged mapping is not a serving mode | 87 GiB against a 1 GiB ARC means first-token latency is minutes and per-4K-fault cost dominates, so an engine that cannot make the checkpoint resident at launch exits instead of listening; `--stage mmap` is kept only for header-only opens |
| **Attention walked once per KV group, selection ascending** | The cache above is the storage story; this is the read of it. Twenty-four query heads over two KV heads means a walk per head touches the same 2 MiB-per-layer gather twelve times and writes a 197 KiB score array per layer while it does, and the 2051 cells arrived in the indexer's heap order, which is a random tour of the arena with nothing for a prefetcher to follow. `qe_attn_walk` (cell outside, the twelve heads of one group inside, exponentials and the accumulator's rescale in registers) and `sel_ascending` (the winning blocks marked in a bitmap, the bitmap walked in address order) turn that into one ascending pass with twelve dots per row and no score in memory at all. One pass per group is two units of work against nineteen workers, so the selection is chunked and the partials `(y, base, sum)` are folded by `qe_attn_combine`, one dispatch and one fold per layer. Measured against the form it replaced at one thread and one layer: **1025 us against 1429 at the 2051 cells this indexer's ceiling produces**, 1.39x, and 2.04x at 8192 cells, which the `top_k` of 2048 never reaches. Over a token's twelve QSA layers and nineteen threads that is **0.65 ms against 0.90** of a 103 ms step, so neither the rate nor the loss moves (1.8358 against 1.8356 nats and 1696 against 1702 top-1s over 2999 positions, both walks in one load because `make score` prints the pair), and the reason to have done it is that the twelvefold read was the term that would not have stayed small. Not bit-identical - an ascending chunked sum reassociates an `exp`-weighted sum, 1.1e-06 between the forms, 1.6e-06 from a double reference - so it entered through the nll gate, and `make forward`'s identity legs are byte-identical either way. The exponent base moves on a 16-nat threshold rather than on every new maximum, because the selection arrives in position order and a context's attention weights favour its end, so records are the expected case and each one would cost a pass over the accumulator. `QE_ATTN_FUSED=0`, `qe_forward_set_attn_fused` and `--attn-head-walk` is the way back |

## 6. Free win available right now

The CPU is idle in the `powersave` governor with `min_perf_pct=15`. On a
latency-bound decode loop this adds frequency-ramp delay to every token and
costs clock headroom. Needs root:

```sh
cpupower frequency-set -g performance
echo 35 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo   # optional: stop E/P thrash
```

## 7. Serving

One binary, `bin/qe-serve`, one process, thread-per-connection HTTP/1.1
(`src/http.c`) with an OpenAI-shaped surface on top (`src/serve.c`).

| route | behaviour |
|---|---|
| `GET /health` | status, uptime, request/token counters, `qe` block |
| `GET /v1/models` | the one served id, plus the full `qe` block, plus a llama.cpp-shaped `meta` (`n_ctx`, `n_ctx_train`, `n_vocab`, `n_embd`, `n_layer`, `size`) so a client that only understands that shape learns the window it is about to overflow |
| `POST /v1/chat/completions` | `messages[]` with string or multipart content, `stream` as SSE |
| `POST /v1/completions` | `prompt` as a string, or as an **array of up to 16 strings** = a batch (one engine pass per element, `choices[i].index == i`); `stream` as SSE for a single prompt, and a 400 for `stream: true` together with an array |

Deliberate choices, each of which is a trade and not an oversight:

- **Blocking I/O, one thread per connection, no epoll.** Generation is
  serialised by memory bandwidth (§3), so request concurrency is only about not
  head-of-line blocking a *parse*. A state machine would add a second place
  where request framing is decided, which is the interesting bug class for a
  hand-rolled server.
- **Chunked request bodies get 501.** "Silently buffered" is how a server ends
  up answering a request it never fully read.
- **Generation is serialised by a mutex.** Two 5.4 GiB/token decodes do not
  share bandwidth; they halve each other.
- **`stream: true` is SSE**, terminated by `data: [DONE]`;
  `stream_options.include_usage` appends the usage chunk.
- **Everything additive lives under `"qe"`.** Responses are valid OpenAI
  objects; the extra block carries engine identity, measured
  `tok_per_step`, and the shape facts from §2 so a client can tell what it is
  talking to without reading this file. Of the two rates in it, `tok_per_s`
  divides *emitted* tokens by seconds that include the prompt, so a 12-token
  answer to a 61-token prompt reads 2.02 there on a host doing 12.3 passes a
  second; `passes_per_s` divides `target_passes` (uncached prompt tokens plus
  emitted tokens) by the same seconds, and that is the figure comparable with
  `make score`. Quoting the first as decode speed is off by however long the
  prompt happened to be.
- **Sampling is a three-layer default and the flags say so.** The checkpoint's
  own `general.sampling.*` keys, then `--temp/--top-p/--top-k/--min-p`, then
  `temperature/top_p/top_k/min_p` on the request, each layer overriding the one
  before it (`src/engine.c`, `sampling_from_body` in `src/serve.c`). The HTTP
  layer distinguishes a key that was absent from one that was sent as `0`, which
  is what lets a client turn off a floor the server was started with; out of
  range is a 400 naming the knob. Order is top-k, then min-p, then the nucleus
  (`src/sampler.c`), and `make sample` pins what each one does to the
  distribution, because it is the only place the probabilities are exact enough
  to assert against. The selection ranks logits and exponentiates only the k
  that survive - `exp` is monotone, so the top-k of the logits is the top-k of
  the probabilities, and the survivor's probability is computed with the same
  shift and the same `expf` it always was. That is 38 us a draw against 578 us
  to exponentiate all 248,320 of them, printed by `make sample`, which keeps the
  all-exponentiating selection in the tree as the oracle the new one is compared
  against. 0.54 ms of master-thread time per sampled token is worth having when
  that thread is also the one dispatching the other nineteen.
- **Four ways to ask for less thinking, in the order they are believed.**
  `reasoning_effort` first (xhigh is what this model is trained to, so it is the
  default and `high` means it; `medium` drops the sentence, `low` and `minimal`
  send the low one, `none` closes the block); then `enable_thinking` or
  `thinking: {"type": "disabled"}`; then, when neither field spoke, this model
  family's own off-marker at the end of a system turn, which is what
  charmbracelet's client writes on the request it gives forty tokens for (`make
  http` asserts the marker is believed and that an effort field outranks it).
  Unknown levels are a 400 listing the ones that exist rather than a guess.
  Whatever budget a reply actually used comes back as `qe.measured.budget`,
  because a client that sent no length cannot otherwise tell a short answer from
  an amputated one: an absent `max_tokens` is the 4096-token default, and an ask
  larger than that is trimmed only by the sequence, never by a constant. The
  engine used to cap every reply at 4096 silently, which on a thinking model is the
  worst of both: a 12000-token reasoning block became a 4096-token one reported as
  finished. What replaces the constant is arithmetic the operator can see - 4096
  tokens is 5.1 minutes of a serialised engine at 74.9 ms a token, 16384 is 20 -
  and `qe.measured.budget` is what makes a trimmed reply visible to the client that
  gets it.
- **A replayed turn gets back the thinking it sent, and no more.** An assistant
  turn that arrives with `reasoning_content` is rendered with that block; one that
  arrives without it is rendered with no block at all, because an empty block is
  what the model reads as *thought of nothing*, and a client that never sends
  reasoning back - charmbracelet's does not - would then be replayed a transcript
  in which the assistant never thinks, once per turn, for the length of the
  conversation. Whether that shape is worse than the template-faithful one is a
  claim about the model and not about this server, so it is a switch:
  `--empty-thinking-blocks` renders the empty block, and the banner prints which
  rule is on so that anyone measuring an answer knows what the prompt said. A
  request with thinking off renders the empty block regardless: its own generation
  prompt ends in one, so leaving it out of the replayed turn made every prompt part
  from the one before four tokens short of that prompt's end, and no turn could go
  back to its rewind point.
  `make chat` pins all three renders: quiet turn with no block, quiet turn with the
  flag on, and an assistant turn last.
- **A turn that showed nothing is the engine's to report.** The client discards an
  assistant message that holds neither answer text nor a tool request and says
  nothing to the user (`internal/agent/agent.go`), which leaves the engine as the
  only party that was ever in a position to notice. So the request line carries
  `content=`, `thinking=` and `calls=`, and a turn that finished with none of the
  first, none of the last, not cut short by a budget and its reader still
  connected prints a `qe: WARN` line after it. The mechanism that produces it is a
  prompt that leaves the model inside its thinking block: what is written before the
  block closes is reasoning, and reasoning already sent as reasoning cannot be
  recalled into the answer.
- **What a real agent needs is asserted rather than assumed.** `make probe-crush`
  points the actual client at a recording proxy on the selftest engine, and
  `make probe-crush-ckpt` does the same against the checkpoint, where the three
  checks that only the checkpoint can pass are in play: a tool request whose first
  delta carries an `id` and a `function.name`, the result coming back as a
  `role: "tool"` message, and the file the model asked for on disk holding the bytes
  it asked for. The proxy is part of the design, not scaffolding: the client reports
  nothing about what it sent, so every claim here is made from bytes on the wire.

### What a request costs, measured on the checkpoint

`tools/measure_ctx.py` posts the same short answer against prompts of three sizes
to one loaded engine and prints the server's own pass count beside the server's
own seconds, which is the only pair that says where the time went. Three rows
completed at 19 threads with nothing else on the host:

| prompt tokens | passes | seconds | passes/s |
|---|---|---|---|
| 123 | 155 | 11.79 | 13.15 |
| 1599 | 1631 | 121.18 | 13.46 |
| 6207 | 6239 | 466.18 | 13.38 |

Straight line through the origin: `(466.18 − 121.18) / (6239 − 1631)` is 74.9 ms per
pass. An ingested token is one pass and an emitted token is one pass - the server
says so itself, `prompt=186 answer=6 passes=192` - so this is the price of a
context in the same units the budget is quoted in: 5568 MiB (§3) over 74.9 ms is
74.3 GB/s, which is the ceiling §1.2 measures, and §12 is why reading less does not
make ingestion faster once the bytes were never the limit. What the three rows
cannot settle is the fixed cost: every row emitted the same 32 tokens, so the
intercept (1.5-2.6 s, depending which pair is used) is either request overhead or
an emitted token costing 45-81 ms, and splitting 32 tokens three ways will not say
which. One short request brackets the low end on its own - 54 prompt tokens and 8
emitted in 3.85 s, 16.1 passes/s, i.e. ~62 ms a pass - so a prompt small enough to
ingest in one chunk is a fifth cheaper than the 74.9 ms the long ones pay, and that
is the entire batching win there is.

Two things this rules out, both of which were believed at some point. The chunking
is already in the served turn, so the 74.9 ms is the number *after* it: on the dense
half of one layer (`make prefill`) the grouped path is worth 11.40 → 5.05 ms per
token at T=16, and that saving is already inside what the server pays. And a wider
read beats a narrower one even though the bytes come from RAM - the same bench with
8 threads, a count that fits inside the performance cores and their larger caches,
is slower at every T (15.77 against 11.40 ms at T=1, 5.85 against 5.05 chunked at
T=16), so the extra L2 does not pay for the missing efficiency cores and
`--threads auto` is the right width for ingestion just as it is for decode.

The consequence for an agent loop is the reason this is written down instead of
left in a log file. charmbracelet's client opens a conversation with ~22 KB of
system prompt, its tool schemas and the project's instructions, and one real first
request measured 11,758 prompt tokens: 862.82 s from the request to its first delta,
73.3 ms a pass at 13.8 passes/s, which is the arithmetic above with nothing added.
Fourteen and a half minutes to the first word of a conversation - and that turn
ended in a tool call, so the next one is what an agent actually lives in:
`prompt=11944 cached=0 859.27 s`, another fourteen and a half minutes, because the
transcript is reused only when a request extends what was already fed
(`src/engine.c:51`) and a tool round trip is not that. The client sends the whole
conversation again with the assistant turn rendered from its own copy, which is a
different byte sequence from the one this engine fed, and the state being reused is
recurrent (there is nothing to roll back to), so the honest reading is that a client
which re-renders history pays for all of it every turn. Reuse in the other case is
real - ask for a continuation of the same transcript and `cached_tokens` reports it
- but nothing in an agent loop hits it, and the number to design against is the full
transcript per turn.

### A stream nobody is reading is stopped, not waited for

Two socket options, and the mistake is treating either one as the policy.
`SO_RCVTIMEO` has to stay long, because reading is where a keep-alive agent thinks:
it runs a tool for three minutes on the same connection it opened an hour ago.
`SO_SNDTIMEO` has to stay short, because a stream that cannot be written means
nobody is watching. Neither is a bound without the other, and neither is a bound
at all unless the send queue can actually fill - which is the part that took three
settings on a live checkpoint to learn.

Same client each time: connect, ask for a streamed answer, read 512 bytes, then
never read again and hold the socket open. The rate from the section above is what
makes this a slow test: 75 ms of engine per delta is ~2 KB/s, so a queue is measured
in minutes rather than round trips.

| `SO_SNDBUF` asked for | what the kernel queued | result |
|---|---|---|
| nothing (host `wmem_max` 4 MiB) | 448 KB at 4.5 min and still growing | engine generating, nothing detected |
| 256 KiB (doubled to 512 KiB) | 435 KB by the end of the turn | turn finished on its own, `finish=stop`, for nobody |
| 64 KiB (doubled to 128 KiB) | 131,275 bytes, then stopped | abandoned: `answer=688 passes=748 137.96 s ... finish=stop stream client-left` |

The first row is why the timeout used to be unreachable: a turn of 4096 deltas is
roughly 700 KB, and anything that holds a megabyte will hold a whole turn and hand
it back with a happy ending. The second row is the trap inside the fix - asking for
a quarter megabyte is a reasonable-sounding number that still swallows the largest
answer this engine can produce, and Linux doubles what you ask for, so "small" has
to be stated in bytes and compared against a turn. At 64 KiB the queue holds about
a minute of deltas; the writer parks, and the pool parks with it, which is worth
noting because it was not assumed: process CPU measured 1890% while deltas flowed
and 0.0% once the queue was full, so a dead reader now costs one minute of
generation rather than the rest of the turn.

What the timeout adds is permanence. `SO_SNDTIMEO` is 30 s, and because it restarts
on every partial success the measured wait between "queue full" and "connection
closed" was about 80 s, not 30 - a reader that accepts a few bytes at a time is
given those chances and then dropped. The request that was abandoned above was
followed by a normal one served in 3.85 s, which is the whole point: generation is
serialised, so every second the engine spends writing to a reader that stopped
reading is a second every other request queues behind.

What it does not cover is a reader that stops reading *during ingestion*, and the
cost of that gap is exactly proportional to the section above. Nothing is written
between the request and the first delta, so there is nothing to fail, and the engine
finds out only when it tries to deliver: four retries of one 6k-token prompt, each
abandoned by its client during a 60 s proxy timeout, were still being served in turn
when the server was stopped - four full ingestions for a reader that had ceased to
exist.

That gap is closed by a comment on a timer (`--sse-keepalive-ms`, default
15000 ms), written from the prefill progress hook (`qe_forward_set_progress` →
`sink_progress`), for the simple reason that a write is the only thing on this
server that can fail. What it buys is bounded wasted engine minutes. What it
explicitly does not buy is worth more than the sentence that buys it, because the
obvious hope is wrong: **a stream that says nothing does not hold open a client's
idle deadline.** charmbracelet's client credits its deadline per *parsed stream
part*, and fantasy yields a part only for non-empty `Delta.Content`, a tool call, or
a finish reason (`providers/openai/language_model.go`), while the SDK's `ssestream`
discards `: comment` lines without inspecting them. Both forms were then measured
against a 3 s deadline with a frame every 500 ms, an empty delta and then a comment,
and neither completed the turn; both score exactly zero against that deadline, which
is the only thing they have in common. The comment is what ships, because they are
unequal everywhere else: a comment line is parsed and never reaches the message
assembler, while `"delta": {}` is a data frame, and a client is entitled to treat
one as the beginning of a message it then finalises empty - which is what a
discarded turn looks like from the outside. Every `text/event-stream` head therefore
also carries `Cache-Control: no-cache` and `X-Accel-Buffering: no`, because one
intermediary that buffers event-stream delivers the whole turn at the end and a
reader cannot tell that from an engine that is not answering. Against a client with
an idle deadline and an engine this slow, the fix is on the client, and it is named
below.

1. **`prompt_tokens` is exact when the checkpoint's tokenizer loads.**
   `usage.prompt_tokens` comes from the real BPE over the file's own vocabulary
   and merges (§10); `completion_tokens` is exact either way. If the file's
   tokenizer cannot be loaded the count falls back to characters / 4 and the
   response says so - `qe.tokenizer` is `false` and `qe.tokenizer_note` gives
   the reason - rather than letting an estimate pass as a measurement.
2. **An engine that cannot compute logits answers 501 `engine_unavailable`**, with
   `info.why` in the error body, rather than tokens. That is the exception now, and
   the honest wording: a `qwen4exp` checkpoint this build can execute opens a
   forward pass and generates (`src/engine.c` sets `can_generate` from the forward
   pass, not from the file type), so what a real checkpoint gets is real tokens and
   the refusal is what an unsupported architecture gets. The selftest engine (no
   `--model`) labels itself in every response corpus and in `qe.engine`, because
   "fluent output" and "correct output" are different claims.

### The 120 seconds that end the turn belong to the client

Two `--ctx 262144` failures that were not `--ctx` failures, both measured, both fixed
in configuration rather than in C, and both worth writing down because the symptom
("it does not work with crush") points at this server and the cause does not.

The first is crush's per-request timeout. `DefaultRequestTimeout` is
`2 * time.Minute` (`internal/config/config.go`), the field is
`options.request_timeout` in seconds, and unset is not unlimited: a nil field is the
default, only an explicit `0` disables it. For a stream it is not a deadline on the
request but an **idle** one - the timer is rearmed inside the wrapper that yields
each parsed `fantasy.StreamPart` (`internal/agent/request_timeout.go`), so a stream
that is silent for 120 s is aborted however much work is in flight. The first request
of an agent conversation ingests 11,758 tokens here, which is 862 s and zero bytes,
so the turn ends at exactly 120 s with

```
LLM stream received no data for 2m0s
The model stopped sending data for 2m0s. Increase the limit with
"option request-timeout SECONDS" or set it to 0 to disable the timeout.
```

...and the engine spends the remaining 742 s of that ingestion generating for a
socket that closed. No wire format satisfies it, per the section above, so the server
prints nothing about it and the config says `{"options": {"request_timeout": 3600}}`,
which is what `tools/probe_crush.py` has always had to set for the tool round trip to
be observable at all.

The second is that the window never reached the client at all. crush discovers a
context window from llama.cpp's shape and nothing else: the `llamacpp` provider
enricher asks `GET /v1/models` and reads `data[].meta.n_ctx`, falling back to
`n_ctx_train`, keyed by `data[].id`, and only when the configured window is still 0
(`internal/discover/llamacpp.go`). The enricher is chosen by the provider's `type`
(`internal/config/load.go`), and omitting `type` means `openai-compat`, which has no
enricher. This server used to advertise its window only under its own `qe` key, so
nothing was learned from it, and the visible result was a client that behaved as if
it had never been told. `/v1/models` now answers both shapes:

```
"data":[{ "id":"qe", "object":"model", ...,
          "meta":{"n_vocab":…, "n_ctx":262144, "n_ctx_train":262144,
                  "n_embd":…, "n_layer":…, "size":…} }]
```

`meta.n_params` is deliberately absent, because this repository counts tensors and
bytes and would be inventing a parameter count to put one there. `make
probe-discover` runs the real client with a config that names neither a model nor a
window, which is the only way to assert the number arrived: with a window of 0 crush
skips auto-summarisation and the UI's context meter reads nothing, so the failure is
quiet, and the check is that the answer came back at all with nothing hand-written.
A hand-written `context_window` is still honoured, and must be ≤ the advertised one:
a client that believes a smaller window trims history the engine would have taken.

### A prompt that did not fit says how much of it was dropped

When prompt and answer cannot both fit the window, the front of the transcript is
dropped (`src/engine.c`, a `memmove` of the retained tail) - which is the only
survivable choice, and was until now silent, so a turn whose system prompt had been
thrown away looked like a model that had forgotten its instructions. It is now a
number: `qe_gen_stats.dropped`, printed as `dropped=` in the request log and reported
as `qe.measured.dropped` on both endpoints. `make http` asserts it reads 0 when the
prompt fits, which is the assertion that catches the field lying.

### A turn that re-renders its own history goes back instead of starting over

Two live crush turns measured `cached=0`: 11,758 prompt tokens in 862.82 s, then
11,944 in 859.27 s. Nothing was reused, and the reason is structural. The engine can
carry a sequence forward - that is what `cached` has always reported - but carrying
it forward requires the new prompt to *extend* what was already fed, and a client that
stores an assistant turn and renders it again sends the same conversation with
different bytes where the answer used to be. Its copy is trimmed, re-marked, or
summarised by whatever sits in front of it. The tokens after that point are somebody
else's, so the honest choices are reading the whole conversation again or putting the
sequence back to where the two prompts still agreed.

Putting it back is now the cheaper one. A turn stores two boundaries: where the
sequence stood when the prompt finished being read, and where it stood when the answer
did. A boundary is `qe_state` - the three in-place recurrences and nothing else,
112.6 MiB and one copy each way at the rate `make alu` measures, which is single-digit
milliseconds whatever the prompt was, plus 4 bytes a token of the ids that produced it
(`docs/ARCH.md` §5) - and matching one against a new prompt is a token comparison, so
a hit is decided before any byte is read. `make prefix` prints the copy it actually
took, because arithmetic at a measured ceiling is an estimate and this repository
reports the difference. What makes a copy that
small exact is that six of the seven per-position arrays are append-only: truncating
the sequence's own cursor `spos[s]` *is* the rewind for them, and the pool's valid count is
`pos / kpool_ratio` by construction (`src/forward.c`, `cur = pos / ratio`). Only the
state that is accumulated in place has to be copied, and the PLE n-gram FIFO is
drained rather than restored, because a pending row belongs to a token that is no
longer the last one. That cursor is part of the state and not of the bookkeeping: a rewind
which put the recurrences back and left `spos` where the abandoned rows had reached kept
passing a greedy check, because every row it then produced still reads perfectly - which is why
`make batch` asserts the pair at byte level on the arm where identity is the contract
(ideas.md §7.25).

Matched by content, not by position: the boundary that wins is the longest one whose
own ids are a token-for-token prefix of the new prompt. That is what lets a snapshot
survive the front-drop above - if the ids match the trimmed prompt, it is the state
after those tokens whatever the transcript used to be. The budget is `--turn-cache`
MiB over an LRU of slots allocated on first use, and the default is asked for in points
rather than in mebibytes, because a point is the only unit in which a budget means
anything: 112.6 MiB of state plus a vocabulary of logits is 115 MiB rounded up, so 512
MiB was four points - two turns - and a five-turn conversation spent its cache in the
second turn and read its whole history again from the third on, quietly. Left unset,
`--turn-cache` takes a fifth of the memory that is free once the weights are resident,
with a floor of twelve points (six turns, both boundaries of each) and a ceiling of half
of what is free and of 8 GiB, and the banner prints the points it bought next to what
one costs on this checkpoint. A budget that lands short of twelve says so at startup,
with the number of turns it does cover, because a cache that is on and too small is
indistinguishable to a client from a cache that is broken.

What a turn did with it is in the request line beside the counters: `cached=` is what
the prompt did not have to read, and `agreed=` is how far the prompt reached into the
furthest boundary the engine was holding even where the two then diverged. `cached=0
agreed=0` is a client that re-renders the front of its transcript, which nothing anchored
at token 0 can serve; `cached=0 agreed=9820` is a cache that ran out, which is a budget
and not a bug. `/v1/models` reports the second case for the process as
`turn_cache_short`, beside `turn_cache_budget_mib`.

The cost side is reported with the benefit, because a cache that is fast for the
wrong reason is not a result: `qe.measured.rewound` says the turn started from a
stored boundary rather than a carried-over one, and `qe.measured.snapshot_ms` is what
the copying cost. `/v1/models` carries `turn_cache_slots`/`_held`/`_bytes`/`_saved`.
The assertion that makes any of it worth trusting is in `make prefix`: a re-rendered
second turn must land exactly at the end of the first prompt, and its continuation
must be the same token ids the same prompt produces with reuse turned off. A restored
trunk that answers differently is a wrong answer that happens to be fast.

That gate costs the checkpoint and minutes per run, which is the reason the two
decisions behind it now live in `src/turns.c` rather than inline in the engine. Given
views over the slots - position, ids, whether the logits survived, whether an answer is
in there, when it was last matched - matching and eviction are functions of integers and
nothing else, so they are asserted in microseconds by `make turns`: thirty turns of a
coding agent's conversation replayed against caches of twelve, four and one point, with
the budget arithmetic next to them. That the default buys at least the points it prints;
that half of what is free can bite that floor and then has to say so out loud; that 512
MiB is four points and four points is two turns. Four mutants of the policy - the anchor
unprotected, the logits-less point not first, `agreed` never accumulated, the floor set
back to four points - each fail a named check, which is the only evidence that a test
this cheap is a test.

Measured against the checkpoint by `make prefix`, on a rendered conversation whose
second turn re-creates the assistant turn rather than replaying it (prompt one 71
tokens, prompt two 102, sixteen tokens asked for):

| | tokens through the network | wall clock |
|---|---|---|
| second turn, cold | 102 + 16 | 8.801 s |
| second turn, put back at 71 | 31 + 16 | **3.884 s** |
| the rewind points it cost | 4 held, 454.1 MiB | **8.9 ms** of copying |

`same_ids` over the two continuations - the same sixteen token ids either way - is the
check that makes the row above a speedup rather than a different model, and it is the
assertion that gates the feature. The same run also reuses 86 of 109 tokens on the
turn that *does* extend what the engine is holding (3.365 s against 9.249 s cold), and
40 of 66 on a raw-text junction whose seam did not survive the tokenizer (1.515 s
against 3.716 s): three shapes of reuse, each compared token for token against reading
the whole thing again.

The copy is the number that was worth chasing. Serially, one thread moving 118 MiB
cost 48 ms a boundary, which is not the bandwidth ceiling of this host but the width of
one core; over `qe_pool_run`, cut into 64-byte-aligned spans, the two boundaries a turn
takes cost 8.9 ms together, which is why `qe_state` keeps the three regions adjacent
and why the copy is not a `memcpy`. At 74.9 ms a token of prompt, that is the
difference between a rewind that is free and one you have to think about.

Extrapolated to the turn this was built for - the 11,944-token follow-up of §7 that
shares its first 11.7 thousand tokens with the prompt before it - the arithmetic is
11.7k tokens not read at 74.9 ms, which is 875 s of the 859 s it measured. The
arithmetic is arithmetic; the two-turn table above is the measurement.

What it does not do: nothing is stored partway through a prompt, so a client that
*shortens* history - trimming old turns rather than appending them - still reads what
is left from the beginning. Closing that means boundaries inside a prompt, which means
either a copy per chunk or a delta rule that can be run backwards, and neither is a
few milliseconds.

### A rewind point stands on cells another prompt can overwrite

The 8.9 ms above is cheap because a point copies the recurrences and nothing else. The
keys and values, the indexer's keys and pool, and the rope positions stay in the shared
caches at the cells they were written to. A point is therefore the state of its prefix
only while those cells still hold that prefix. Every prompt fed from a lower position
writes over them: a client's title request between two turns, a sub-agent, another
session. Until this was found, the engine restored such a point anyway, putting one
conversation's recurrent state on top of another's attention context. On the
checkpoint the test is unambiguous: a system prompt holds a code word, a short first
turn, an unrelated 1009-token prompt, then the follow-up "what is the code word". Put
back over the overwritten cells, the model answered that there was none.

Two changes, in `src/engine.c`:

- **Honest matching.** `cells_held` is the agreement of the prompt with the transcript
  that was last fed, from cell 0. A point beyond it is not a match. That alone is
  correct, but it makes interleaving expensive. Replayed offline over a real crush
  session with its two sub-agents (46 requests, 1.04 M prompt rows), the share of rows
  read goes from 12.0 % (fast and wrong) to 78.1 % (correct).
- **Kept cells.** Before a prompt overwrites cells from position `h` on,
  `spans_save` copies `[h, end)` aside, where `end` is the longest held prompt-end point
  that still stands on those cells. It does this only if such a point exists, and only
  for prompt-end points: an answer-end point is the generated answer, which a client
  that re-renders its history never sends again. A later prompt that matches a point
  beyond `h` through a kept range gets the range put back (`qe_forward_cells_put`)
  before the point is restored. A range whose every served point a newer range of the
  same conversation also serves is dropped. Without that, two sub-agents taking turns
  keep each other's history again at every switch. Otherwise the least recently used
  range goes first. The copy is about 20 KiB a cell (K and V of the 12 attention
  layers, the indexer's keys, the pool blocks the range touches, ids and positions).
  The budget is `QE_KV_SPAN_MIB`, default 4096.

The same offline replay with kept cells reads 14.7 % of the rows at a 2 GiB budget
and 12.0 % at 3 or 4 GiB, which is the fast path's figure with the wrong answers taken
out. At 2 GiB the main session's 551 MiB range is the least recently used when the
second sub-agent's ~800 MiB range arrives, and the main session then reads one 36k
prompt again. Hence the 4 GiB default. `make prefix` asserts the interleaved case on
the checkpoint: the follow-up rewinds through kept cells and its continuation is the
cold one token for token.

Images had a cache problem of their own, and it was in the renderer. A request keeps
its newest 32 images, and the 33rd used to push out the oldest one, which stood near
the front of the prompt. That changed the prompt at that point, and every new image
after it did the same with the next oldest, so each image turn of a long session read
the whole session again. Replayed offline, a real 129-request crush session with 39
screenshots read 986k of its 8.28 M prompt rows, nearly all of them from request 106
on, each time about 100k rows. Images are now left out 16 at a time
(`QE_CHAT_IMAGE_DROP_STEP`), so the front stays the same for the next 16 images: the
same replay reads 217k rows (2.7 %), with one full re-read where the step is taken.

### Shutdown is a request path

Closing the listener is the part that already worked, and it is the part that was
never the problem. `qe_http_serve` returns the moment `qe_http_stop` is set, while a
connection thread can be anywhere inside a request, and one of those places is inside
`qe_engine_generate_tokens` holding the engine; `qe_engine_close` took no lock and
waited for nothing, so SIGTERM during a turn was a use-after-free on the one path
every harness, every test target, and every operator takes. The other bad option is
the one a deadline exists to prevent: waiting for the request unconditionally means
the process outlives its own termination by up to the length of a turn, holding the
port and 87 GiB, which is indistinguishable from a hung machine.

So shutdown does both halves, in this order: the signal handler sets the flag the
per-token callbacks and the prefill hook consult, which ends the generation at its
next pass rather than at the end of the turn; `qe_http_drain` then waits for the
handler threads for `--shutdown-grace` seconds (default 5); and if any outlive that,
the process `_exit`s without freeing the engine, because the kernel reclaims the
weights faster than an exit-time free would and running `atexit` - LeakSanitizer
included - underneath a thread that is still generating is a data race with a report
stapled to it. `make shutdown` asserts all of it against the selftest engine without
a checkpoint, by holding a request open on a body that never finishes, which is the
same position a generation is in as far as this path can tell: exit without a second
signal, within the grace, saying which way it left, and the port bindable again.

### Images

With `--mmproj`, an image is seen. `src/vision.c` is the projector's tower - 27 ViT
blocks of 1152 with 16 heads of 72, 16-pixel patches embedded by the summed temporal
pair of kernels, a 48x48 learned position table resampled to the grid, the 2D vision
rotary, and the 2x2 merger to the trunk's 2560 - and `src/engine.c` turns each image
of a request into its merged grid of prompt rows:

- **The prompt.** The renderer writes the template's own spelling of an image, the
  vision start and end markers around one image pad, by id. The engine replaces each
  pad with the image's `nx * ny` rows. The pass reads `ple.image_token_id` at those
  rows, which is what the PLE window hashes there and what llama.cpp's cache stores for
  an embedding cell (`ple_embd_token`); the transcript, the carry-over and the rewind
  points read a tag made of the image's bytes and the row, so the same screenshot sent
  again next turn is reused and a different one of the same size is not.
- **Positions.** The checkpoint's rope is interleaved m-rope over sections
  [11, 11, 10, 0]. An image row is (t, h, w) = (base, base + row, base + column) and a
  text row is one past the largest component of the row before it, which is
  llama.cpp's mtmd decoder position for this projector and puts the text after an
  image max(nx, ny) on from its start. The pass keeps a (t, h, w) per cell and reads it
  only once a sequence has held an image: a text-only sequence takes `qe_rope_init` at
  its cell, the arithmetic it always took.
- **Cost.** The tower runs only for an image whose rows are not in the reused prefix.
  At the default cap of 4096 rows a 5760x2160 screenshot is 4056 rows and 69 s of
  tower (llama.cpp b12359: 77 s); a 640x480 photo is 300 rows and 1.0 s (2.1 s).
  `--image-max-tokens N` lowers the cap, and the time falls with its square.
- **Agreement.** `bin/bench_vision mmproj image ref.bin` compares against llama.cpp's own
  dump (`MTMD_DEBUG_EMBEDDINGS=ref.bin llama-mtmd-cli ...`). The median row cosine is
  0.99998 and 0.999994 on the two images measured; a few rows of a ViT are
  ill-conditioned, and on them llama.cpp's flash and plain attention disagree with each
  other by 0.97 in cosine and 2.3 % overall, which is the size of the gap between this
  f32 tower and either of them. `QE_VISION_EMULATE=1` reproduces the reference's
  BF16-rounded activations and half-precision GELU, for that comparison only.

Decoding is stb_image for PNG, JPEG, GIF and BMP - the decoder the reference uses, so
the pixels agree - and libwebp, opened at run time, for WebP. Remote `http(s)` image
URLs are refused with 400 rather than fetched. Without a tower (no `--mmproj`, or
`--batch N`), an image is recorded in the transcript as `[image N: image/png, 1234
bytes. ...]` with a sentence telling the model it cannot see it.

### How that was verified

- `tools/smoke_http.py` - 127 checks over all four routes, SSE framing, `[DONE]`,
  a 70-byte decoded image part, determinism, keep-alive, and the 404/400/405/415
  refusals. Four of them are about what a stream is allowed to look like when it has
  nothing to say: an event-stream head that says `no-cache` and `no-buffering`, no
  frame anywhere whose delta is empty and whose finish reason is null, and every line
  of a crush-shaped body either `data: ` or a comment. A keepalive comment cannot be
  asserted by observing one, because the selftest generator never calls the progress
  hook, so what is pinned instead is the framing invariant that makes a comment safe:
  if nothing on the wire is an empty data frame, the timer cannot be what confused the
  client. 0 failures.
- `tools/fuzz_http.py` - 74 checks of hostile framing on raw sockets: headerless
  requests, 40 KB header lines, 4 KB paths, chunked bodies, negative and
  non-numeric and 1 TB `Content-Length`, clients that stop mid-body or leave
  mid-stream, 5000-deep JSON, lone surrogates, broken base64. 0 failures, and
  the server answers `/health` afterwards.
- 6 concurrent 120-token requests: all 200, serialised by design, 1.02 s wall,
  26 tokens each (the selftest corpus ends and emits eos), `tok_per_step` 3.714
  at `spec_k` 4. A client that disconnects mid-stream does not take the worker
  down with it.
- Both scripts re-run against an ASan+UBSan build, stopped with `SIGTERM` so
  LeakSanitizer reports at exit: clean after one leak (a `strdup`'d mime string
  in `render_messages`) was found and fixed, and clean again after a second one
  reported by the same gate - the completions handler builds its reply through the
  same `sink` the chat handler frees and then freed only the answer text, so the
  bytes `qe_utf8_feed` holds back between two tokens were never returned: 256 per
  request, forever, on a server that is meant to run for weeks. Graceful shutdown
  exists for that reason and is worth keeping: a server that only dies to `SIGKILL`
  never runs its own leak check - which is precisely why the shutdown that gives up
  does it with `_exit` and says so in the log rather than pretending to have been
  tidy.
- `tools/test_shutdown.py` (`make shutdown`) - 15 checks that the shutdown above is a
  bound and not a hope, using the one trick that needs no checkpoint: a request whose
  `Content-Length` is 48 MiB and whose body is three bytes leaves a handler genuinely
  inside the request, which is the same position a generation is in. Asserted in four
  states (idle, mid-request, keep-alive client between requests, cleanly finished
  request) that the process exits on the first SIGTERM without a second signal, inside
  the grace, naming which way it left, and leaves the port bindable.
- `make probe-discover` - the real client with a config naming neither a model nor a
  context window, which is the only assertion that the window in `/v1/models` was
  read by something rather than merely written by us.
- `tools/probe_ckpt.py` (`make probe`, one load of the real checkpoint) is the
  only thing that watches the checkpoint's own logits over HTTP: that a served
  checkpoint says it generates and answers greedy with the same text twice, that
  `top_k: 1` and `min_p` at the argmax reproduce greedy while `temperature: 2`
  does not, that two seeds differ, that the off spellings draw what an absent key
  draws, and that eight out-of-range values are 400s naming the knob. Everything
  in the two lists above runs against the selftest engine, which speculates and
  therefore filters nothing - a sampling assertion made there passes for a reason
  that has nothing to do with sampling.

```sh
make test                       # kernels, MTP loop, JSON, pool, MoE
./bin/qe-serve --port 8099 --spec-k 4 --id qe-selftest &
python3 tools/smoke_http.py && QE_PORT=8099 python3 tools/fuzz_http.py
```
(or `make http`, which does all three and shuts the server down with SIGTERM,
and `make probe`, which does the same against the real checkpoint instead of a
synthetic one)

Every target in the Makefile runs under `timeout -k`: a fuse that only sends
`SIGTERM` is a fuse against a process that is still able to hear it, and the whole
point of a deadline is that it holds for the case where nothing is listening.

### The token that went in twice, and what it took to notice

`sink_text` (`src/serve.c`) accumulated a reply in two places: once in the branch
that decides reasoning-versus-answer, and once more at the end of the function with
nothing guarding it. On a stream that is invisible - the answer is not accumulated at
all, and the reasoning append returns before the second one runs. On a buffered
request every token landed in the body twice, and the checkpoint's answer came back
as `wordword wordword wordword`, which reads as a broken model and is a broken
string. `make probe` found it, because the probe reads the answer as a person would.

Two things follow, both of which are now tests. The streamed-and-buffered check in
`tools/smoke_http.py` used to compare shapes (`text in content or len(text) > 10`,
which the second half made vacuous); it compares bytes now, on both channels, on both
request shapes. And a buffered body is valid UTF-8 by construction: a reply that
stops inside a three-byte character is trimmed back to the last whole character,
rather than putting a partial sequence inside a JSON string, which is the thing the
same probe run tripped over second. Streamed chunks are not trimmed, because a client
reassembles them and the sequence finishes in the next one.

## 8. What this build does not do

Stated as a list of gaps, each with what closes it, so nobody has to read the
source to find out what is missing.

| gap | what it would take |
|---|---|
| Batching exists for decode and stops there | `--batch N` drives N sequences through one walk of the weights (`qe_forward_step_batch_rows`, asserted by `make batch` and measured through the request entry by `make lane`): **2.15-2.20x** the step eight rows share, **1.79x** a wave of eight short requests, and **1.00-1.02x** a wave of two 512-token ones, because a prompt is still read one at a time. What is left is named by that gap and by `QE_MAX_SEQ`: eight rows is where the row map stops, a row costs its share of the vocabulary-wide logits and one replicated copy of the 113 MiB GDN state (`docs/ARCH.md` §5), and packing several prompts into one chunked row map is measured rather than believed not to be the answer (ideas.md §8.1) |
| Chat templates are rendered by hard-coded ChatML, not evaluated | the file carries 9993 characters of Jinja with image/video count handling. An evaluator is a project of its own; today `render_messages` reproduces the ChatML shape the template produces for text-only transcripts, and that is the honest description of it |
| Vision is serial-lane only | the mmproj's tower is built (`src/vision.c`, §Images) and an image a request carries becomes its rows of the prompt; with `--batch N` images are described instead, because an encode needs the whole pool and the lane driver walks the weights on it outside the engine lock |
| Speculation pays only at the shortest draft, and only just | the drafter is not the gap: `mtp-...-Q4_K_M.gguf` is resident, drafts, and never changes what greedy says. A verified window is the gap - **4.45x** a step at nine rows, ~97 ms of it head and lm_head logits paid for rejected rows as well as kept ones - so acceptance of 1.85 ids a pass closes the arithmetic at k = 1 and nowhere above it: 1.19x at k = 1 with a head and 0.75x at k = 8, 1.05x at k = 2 with the n-gram hash and 0.91x at k = 8 (`/tmp/f40_head.log`). The length is therefore resolved from the drafter at open instead of being a constant (§4); what would widen it - logits for only the rows a window keeps, boundary copies made cheaper - is priced at together ~1.1x |
| A turn that trims its history is still ingested from the beginning | `--turn-cache` stores boundaries at prompt end and answer end, so a client that re-renders or replays goes back instead of re-reading (§7). One that *shortens* the conversation matches nothing, because nothing was stored partway through a prompt. Closing that needs intra-prompt boundaries: a copy per chunk of prefill, or a delta rule that can be run backwards. Which of the two causes a turn that re-read everything it is worth knowing is `agreed=` on the request line: `agreed=0` is this row, `agreed=9820` is a cache that ran out (§7) |
| Ingestion still pays per token for the recurrences, and for the experts wherever routing does not repeat itself | the dense half is chunked: `qe_forward_prefill` runs sixteen tokens of a prompt through one stage of a layer at a time and sweeps each projection once for the chunk, worth 94 → 51 ms per ingested token at the shipping width (§12). What a chunk cannot share is a recurrence - the conv ring and the delta rule are position-ordered by definition. The other two things it could not share are now shared: Q8_0's kernel that turns one dequantised weight run into T accumulators is reached through `qe_gemv_multi_t` behind the projection threshold, which `make forward` prices at **1.15-1.21x** of ingestion (§12), and the experts have the same kernel for both of their types - **1.80x** on a deep expert bucket, flat past four rows because that is the register budget - wired through `moe_block_chunk` for **1.05-1.06x** of ingestion (§12). What is left in this row is what real routing cannot reach: 512 experts and ten picks a token make buckets of about one row, so a chunk shares an expert's bytes far more often than it shares its dequantisation, and a decode step is one row and shares nothing at all |
| Generation is 18% off the §3 budget | 103-104 ms/token against 88.0 predicted. `--profile` splits it (§14): the block stage is 46% of a step, the routed experts 31%, the hyper-connection mixers 12% and the lm_head 11%, and each of the first three is 4-12 ms above what its own bytes cost at the rate `make fast` measures for that type |
| Sampling filters do not apply while speculating | `src/mtp.c` draws from the target's softmax with temperature and nothing else (`qe_spec_opts`), because a filter has to be applied to the *same* distribution the draft was drawn against for the rejection step to stay unbiased - top-k and a nucleus change with the token, so the accepted-token distribution stops being the target's. What it means in practice: the checkpoint path (`gen_checkpoint`) honours the knobs, the selftest engine generates through the speculation loop and does not, so `make http` can only assert that a bad value is refused. Closing it means filtering the target and the draft with the same cut in `qe_spec_generate`, which is a change to a correctness-critical loop, and the host now has the real head to validate it against: `mtp-...-Q4_K_M.gguf` drafted 32 ids without moving one of the ids greedy says, so the acceptance side of that change is measurable rather than hypothetical (§4, `/tmp/f38_mtp_short.log`) |

## 9. Routing measured against real router weights

`make route` (`bin/bench_route <volume1> <deadline> <tokens> <text files...>`)
runs the real `blk.N.ffn_gate_inp.weight` routers over real embeddings and
answers the one question §3 could not: **how much do the expert sets of adjacent
tokens in a decode window actually overlap?**

Router inputs are RMS-normalised rows of `token_embd.weight` (Q8_0 dequantised,
normalised with the file's own `rms_epsilon`), so **layer 0 is faithful** and
layers 1-47 are realistic-but-not-the-true-activations: a real layer-N router
sees a residual stream, not an embedding. The harness prints that warning itself.

2000 tokens of four real files (this document, `MANUAL.md`, `src/tok.c`,
`tools/smoke_http.py`), 960,000 expert selections, 512 experts, top-10:

| window T | measured distinct experts | hard bound `T x 10` | independent routing | independent per **distinct token id** | routed bytes/layer, one row at a time | grouped | fewer bytes |
|---|---|---|---|---|---|---|---|
| 2 | 17.8 | 20 | 19.8 | 19.8 (1.99 ids) | 44.4 MiB | 39.5 MiB | **11.1%** |
| 4 | 30.2 | 40 | 38.8 | 38.8 (3.93 ids) | 88.9 MiB | 67.2 MiB | **24.4%** |
| 8 | 48.2 | 80 | 74.7 | 66.0 (7.48 ids) | 177.7 MiB | 107.1 MiB | **39.7%** |
| 16 | 73.1 | 160 | 138.6 | 123.5 (13.72 ids) | 355.5 MiB | 162.4 MiB | **54.3%** |

**The overlap is real, not an artefact of repeated content.** A window of 8 rows
here contains 7.48 distinct token ids, so some of the saving is the trivial one:
two identical ids obviously choose the same ten experts. Removing that
explanation explicitly (the "independent per distinct id" column is what
independent routing over the *distinct* ids in the window would cost) still
leaves 10.1% / 22.2% / 27.0% / 40.8% fewer experts than chance at T=2/4/8/16.
Adjacent tokens genuinely prefer overlapping expert sets.

Per layer: the most over-lapped are layer 0 (15.2 / 22.1 / 31.7 / 44.7,
adjacent-token Jaccard 0.349), layer 36 (14.3 / 19.7 / 26.6 / 35.9, Jaccard
0.437) and layer 20 (15.3 / 22.4 / 31.9 / 45.0, Jaccard 0.341); the least are
layers 42, 23, 8, 5, 47 at 18.4-18.9 / 32.7-34.4 / 54.1-58.1 / 83.5-90.7 with
Jaccard 0.063-0.095. Layer 0 being the most over-lapped is expected - its input
is the embedding, and semantically similar tokens sit near each other there.

What it buys, and what it does not:

- It validates `qe_moe_forward_multi()` in exactly the way `bin/bench_moe` could
  not. Under that bench's synthetic uniform routing the grouped pass was
  **break-even**, which is the correct control result and was reported as such;
  under real routing it is a 11-54% byte saving that grows with window width, so
  the implementation is right and its payoff is a property of the model, not of
  the scheduler.
- **It is now the thing serving does.** `moe_block_chunk()` runs a chunk's rows
  through it instead of calling `qe_moe_layer()` once per token, with the
  shared-dequant kernels for IQ3_S and IQ4_NL underneath it (§12). End to end
  that is **1.05x** of ingestion at the shipping width, which is this list's own
  arithmetic coming true rather than a surprise: 11-54% of the *routed* bytes,
  routed experts being 19.9% of the bytes and about 15% of a chunked token's
  time, is single-digit percent. The wire did not move the numerics the grouped
  pass already had (1.7e-06 worst relative against the separate forwards, and the
  same next token at 5% of the head's range against the chunk as it ships
  today), and `qe_forward_set_moe_grouped` is the way to ask for the old
  per-token pass.
- Routed experts are 19.9% of per-token traffic (§3), so at T=8 this term is
  worth up to 8% of a window's bytes, and it is the mechanism that makes
  speculative verify windows and small batches cheaper per token.
- **Expert popularity is only mildly concentrated**: the hottest 16/32/64/128/256
  of 512 experts carry 8.3/15.1/27.0/45.6/71.0% of selections, and **no expert is
  never selected**. Pinning a hot expert set is therefore worth far less here
  than the Strata profile suggests; the ~4.4 GiB of always-read weights
  (projections, `hc_*`, routers, shared experts, embeddings) is the better pin
  target, and §5 of `docs/ARCH.md` sizes it.
- **Cross-layer persistence measured ~ 0**: the same token's expert set overlaps
  between adjacent layers by 0.19 of 10, against a 0.20 independence baseline.
  Caveat, because it matters: every layer in this harness got the *same* input
  vector for a given token, so this is weak evidence against layer-to-layer
  affinity, not a refutation of it.
- Corpus stability: the same harness over 1500 tokens of this document alone
  gave 17.8 / 30.3 / 48.3 / 72.8 distinct experts and 10.9 / 24.2 / 39.7 / 54.5%
  savings. Two unrelated corpora agreeing inside half a point is the reason the
  numbers above are quoted at one decimal.

### Half the router: bf16 selection, f32 weights (`qe_moe_route`)

The router is F32 in this checkpoint: 512 rows x 2560 x 4 B = 5.0 MiB per layer,
**240.0 MiB per token** across 48 layers, 4.3% of the 5568 MiB step. A bf16 copy
halves that to **120.0 MiB per token** and costs 120.0 MiB of arena for all 48
shadows. `bin/bench_route` measured the consequence on the checkpoint's own
routers, rounding the weight matrix rather than the logits, over 96000 real
layer-token pairs of real text:

```
Router width over 96000 layer-token pairs (48 layers, 512 experts x 2560 wide):
  f32 rows     240.0 MiB per token across those layers
  bf16 shadow  120.0 MiB per token, and 120.0 MiB of arena for all 48 of them
  control     0 of 96000 pairs picked a different set of 10 experts (0.0000%)
              0 of 960000 expert slots differed (0.0000% of selections)
```

**That control is the reason a kernel bug got found.** Its first run said the
opposite - 99.98% of tokens picked a different set - which is impossible for a
10-of-512 argmax under a 0.4% rounding. The cause was in `qe_gemv_bf16()`: the
inner load was `_mm_loadl_epi64`, a 64-bit load, for an 8-element bf16 group that
is 16 bytes wide, so every bf16 dot product with K >= 8 was summing four lanes out
of eight and returning half its terms. It is fixed (`_mm_loadu_si128`,
`src/kern_iq.c:392`), and what makes the fix stick is the test shape: `make fast`
now runs `check_bf16_gemv()`, which checks the SIMD dot against a scalar double
reference for a dozen K **and** feeds ones to ones, requiring the result to be
exactly K. A dropped lane survives any test that only compares magnitudes; it
cannot survive ones-in-ones-out.

Two consequences for the implementation, both of which the control separates:

- **Selection may read the shadow.** The ten picks are identical, so the k-th
  largest boundary is not being decided at bf16 precision on this checkpoint.
  That is a property of *these* routers (logits measured over the corpus: min
  -4.7357, max 11.9450, mean 1.1080, no NaN), not a theorem, which is why the
  control exists and runs against the checkpoint rather than against synthetic
  weights.
- **The mixing weights may not.** Renormalised top-k weights are read from the
  f32 rows of the k winners after selection - 10 x 2560 floats, 100 KiB per
  token, against the 120 MiB the shadow saves. `qe_moe_refine()` (`src/moe.c`,
  declared in `include/qe.h`) is that re-read and renormalisation, and
  `qe_moe_route()` is bf16 router, top-k, refine. Both of a forward pass's two
  routings call it - `qe_moe_layer()` for a lone token, and a chunk's routed half
  after its one router pass - which is the correction this section owed: the
  sequence existed, was checked, and was described here as what the engine does,
  while `src/forward.c` called the router and the top-k itself and left every
  expert's output scaled by an 8-bit mantissa. A chunk and the same token fed
  alone still agree, because both halves gained it. `bench/bench_moe` section 1b
  checks the resulting routes against the f32 router's **by expert id rather than
  by slot** - the select-plus-refine pair the engine composes included - and
  reports both as exactly equal (0.0e+00) while an unrefined bf16 weight is off
  by 1.3e-04 on a shared pick.

Section 1b plants 300 deliberately near-tied synthetic activations, and 9 of them
do flip an expert set there. Synthetic ties are the pathological case; the
checkpoint's routers, 320x more numerous, produced none. That difference is the
whole argument for measuring the control on the model rather than on a generator,
and the bench reports the flips instead of asserting they are zero, so a future
checkpoint that *is* tie-prone shows up as a number rather than as a green pass.

Net: 120 MiB of the 5568 MiB step, 2.1%, ~1.6 ms of a 73 ms step - smaller than
grouped expert execution, but it is a routing-path saving that applies to every
token including single-token decodes, where the grouped saving is zero.

### 9.1 What the f32 re-read costs

The re-read is 10 rows of 2560 f32 per routed layer - 100 KiB per token per layer,
**4.7 MiB** across the 48, which is 0.08% of the 5568 MiB a decode token reads. A
share that small is no reason to skip the re-read and no reason to call it free
either, so it is priced rather than derived from its bytes. `make forward`'s
chunk-width sweep, one load per form:

| form | width 1 | width 512 |
|---|---|---|
| no refinement (the trunk before this) | 93.4 ms/token | 16.2 ms/token |
| refined per row, inside a chunk's routing loop | 94.4 | 16.5 |
| `qe_moe_refine_rows()`: a chunk's rows in one dispatch | 94.3 | 16.3 |

(`/tmp/f42_forward.log`, `/tmp/f43_forward.log`, `/tmp/f44_forward.log`.) The sweep
is the leg read because it reproduces to ±0.1 ms/token between loads where those same
three loads' `one step` leg moved ±5%; a 0.1 ms/token cost is below what the latter
can see at all.

Engine ingestion is therefore 61.7 → **61.3 tok/s**: +1.0% at width 1, +0.6% at the
width that ships, the bar this repository holds (≥50 tok/s on this host) unaffected,
and `make test` and `make forward` 0-FAIL across the change. Threading the chunk's
rows recovered two thirds of what the per-row form gave up, so the residual is the
reads themselves and not the dispatch - ten dots of 2560 floats is a few microseconds
for one row and a millisecond and a half for a 512-row chunk, which is why the chunk
form exists and why it is a pool job rather than a loop. `qe_moe_refine_rows()` is
bit-identical to `qe_moe_refine()` row for row (`bench/bench_moe`'s router chunk form
check requires it), and that is what keeps a chunk and the same token fed alone on
identical ids now that both halves of a routing read f32.

## 10. Tokenizer: what is exact, and what is not

`src/tok.c` builds a byte-level BPE from the file's own metadata - 248,320
tokens, 247,587 merges (all of them load, 0 skipped), all 256 byte tokens
present, 276 control-type ids, `pre = qwen35` translated from the upstream
pre-tokenizer regex. `make tok` (`bin/bench_tok`) is the check:

| property | result |
|---|---|
| per-token round trip over all 248,320 ids | lossless, idempotent |
| text round trips (28 texts: prose, C source, JSON, UTF-8, control ids) | 28/28 exact |
| encode throughput, prose | 146.7 MB/s (0.273 tokens/byte) |
| encode throughput, pathological (one 200 KB run of letters) | 4.1 MB/s |
| vocabulary entries the merge order cannot reproduce | **203 (0.08%)**, e.g. CJK pairs that re-segment into 2-3 other ids |
| exact count vs the old `len/4` estimate | 0.81x English prose, 0.88x mixed, 1.00x CJK, **1.75x source code**, 2.95 bytes/token over the bench corpus |

The last row is why the estimate was worth replacing: it is 20% high on English
and 43% low on source code, so it was not a noise-level error, and the server
now reports the exact number when the checkpoint tokenizer loads (§7).
The 203 unreachable entries are a property of the merge order in the file, not a
bug in this implementation; they bound the difference between this encoder and
the reference to 0.08% of the vocabulary, and every one of those entries still
*decodes*.

## 11. Two reader bugs that no test caught

Both were found by measuring something that could not be explained, and both are
now covered by tests that fail if they come back. They matter more than any
result in this document, because they invalidate earlier numeric claims.

**1. `qe_half_to_f32()` was not a conversion.** Its body was
`v.u = (uint32_t)h.d << 16`, which is a **bf16** widening applied to fp16 data.
Measured: `0x3C00` (fp16 1.0) became 0.0078125, `0x33C1` (0.24230957) became
8.99e-08, `0x4120` (2.5625) became 10.0 - the error factor depends on the
exponent, so it is not a scale that cancels. Every fp16 block scale in Q8_0,
Q6_K, IQ4_NL, IQ3_S and IQ4_XS went through it, which is to say **every
quantised kernel in this repository returned wrong numbers**, and every logits
vector derived from them. Correct now (IEEE field decomposition, plus denormal,
zero, inf and NaN arms), and `src/kern.h` says "do not simplify this back to a
shift".

**2. `qe_tensor_data()` ignored the volume header.** GGUF tensor offsets count
from the start of each volume's data section, and in a split checkpoint each
volume has its own header (10,946,618 / 23,584 / 54,317 bytes, data at
`align_up(header_end, 32)`). The old code returned `map + t->off`, so **every
tensor in volumes 2 and 3 was read displaced by 23,584 or 54,336 bytes**. Since
that is smaller than a tensor and larger than a quant block, it produced
plausible-looking data with misaligned blocks rather than an obvious fault.

*Why no test saw either.* `bin/bench_kern` compared the AVX2 kernels against the
scalar reference in the same source file, and both called the same broken
helper: the comparison divides by the same wrong factor on both sides, so it
reported "worst relative error 1e-16" for kernels that were wrong in the same
way. The offset bug is invisible to every synthetic benchmark because none of
them open a volume - the file-reading benchmarks only used `token_embd` and the
router, and `token_embd`'s two volume-2 rows read as garbage was explained away
as quantisation noise (row 100 decoded to scale 0.63 with 515 of 532 rows having
a non-finite scale; the same block after the fix reads scale 6.87e-05).

*What the fix was checked against.* Not against itself:

- `check_half_conversion()` in `bin/bench_kern` walks all **65536** fp16 bit
  patterns against an independent IEEE reference built from `ldexp`.
- `check_q8_anchor()` hand-builds a two-block Q8_0 row (scales 2.0 and 0.25, all
  quants +16, `x` all 1.0) whose dot product is exactly 1152.0, so a wrong block
  scale cannot hide inside a comparison against another kernel.
- `check_scale_linearity()` proves per-type that the kernels honour the block
  scale: double row 1's scales and the dot must double bit for bit; negate and
  it must negate. Run for Q8_0, IQ3_S, Q6_K, IQ4_NL and IQ4_XS.
- the offset formula was confirmed by an **independent Python decode** at
  `23584 + 29328614400 + 100*80*34` (volume 2 data start + tensor offset + row),
  which matches `qe_tensor_data()` byte for byte, and `tools/gguf_inventory.py`
  now asserts the offset chain in every volume (0 mismatches, `data_start +
  span == st_size`).

`docs/ARCH.md` §1 documents the layout; `make fast` runs the three new checks on
every invocation. **What did not change: any bandwidth number.** A wrong scale
moves the same bytes, so §3's budget, the GEMV rates, and the MoE thread sweep
all stand. What did change: no numeric output of the quantised kernels produced
before this fix, including any earlier statement that they were validated,
should be trusted.

## 12. Ingestion: chunking the dense half

A decode step carries one token, so every weight byte is read once and there is
nothing to win. Ingestion is not a decode: a step carrying T tokens needs every
row of every projection T times. `qe_gemv_t()` walks a matrix once per activation,
so a T-token prompt walks it T times; `qe_gemv_multi_t()` inverts the nest -
weight rows blocked to 1 MiB so that a block shares the 2 MiB L2 with the T
activation rows - and its payoff is bounded by whether a row-block that just
arrived is still there when the next token asks for it.

`make prefill` measures it on the checkpoint's own layer-0 dense tensors, a 566 MiB
round (`attn_qkv` 26.6, `attn_gate` 15.9, `ssm_out` 15.9, `ffn_gate_shexp` 1.7,
`hc_attn_up` 3.3, `ffn_gate_inp` 5.0, `output.weight` 497.3), at the machine's full
width with bytes counted by the hardware rather than derived from arithmetic:

```
     T   one at a time              chunked        bytes/token        time/token
                MiB     GB/s         MiB     GB/s    saved    one at a   chunked
      1       553.7     45.8       548.9     45.2     0.9%      12.68 ms    12.73 ms
      2       535.0     43.9       301.5     40.4    43.6%      12.77 ms     7.83 ms
      4       517.1     43.3       140.4     23.6    72.8%      12.53 ms     6.24 ms
      8       503.2     42.8        66.2     12.5    86.8%      12.33 ms     5.57 ms
     16       489.0     42.3        32.5      6.5    93.4%      12.12 ms     5.23 ms
```

16 threads over both core types; the same bench confined to the eight performance
cores, where the counters can be checked against a round of known size, says
485.8 → 37.4 MiB (92.3%) and 14.84 → 5.82 ms (39.2%), so the result is a property
of the loop order and not of the configuration it happened to be measured in.

**Bytes fall slightly faster than 1/T** (T=16: 553.7/16 = 34.6 expected, 32.5
measured). Two reasons, both structural: inside a 1 MiB block the T activations are
reused against each other as well as against the weights, and the small tensors of
the round (1.7-5.0 MiB) live in the 30 MiB L3 across the whole round, so at T>1
they are read from there rather than from DRAM. The margin is only 6%, which is
itself the honest reading: this is a reuse win of almost exactly the 1/T the loop
order promises, not a mysterious extra.

**Time does not fall nearly as fast as bytes: 43.1% of the time for 6.6% of the
bytes.** That is the expected shape and not a bug in either number. Once the weights
arrive once instead of T times the kernel is no longer DRAM-bound, and the "GB/s"
column falling from 45.8 to 6.5 is precisely the claim that this path is no longer
moving DRAM: what is left is dequantising each weight and multiplying it into T
accumulators, which is work, and work does not get cheaper by reading less. The
practical figure is the pair: a 16-token prompt costs 83.7 ms of dense projections
chunked against 193.9 ms walked one token at a time - 5.2 ms and 12.1 ms per
ingested token.

For the whole step, the byte side of §3's 5568 MiB budget at T=16 is: 4426 MiB of
chunkable weights (projections, hyper-connections, router, shared expert, head,
final mixer) at the measured 6.6% = 292 MiB, routed experts at the 54% saving §9
measured on real routing = 510 MiB, PLE's 33.5 MiB per token unchanged because
those rows are per token by definition. **~835 MiB per ingested token against 5568
for a decode.** Those are bytes, and the paragraph above is the warning against
turning them into a 7x time claim.

Two notes on what the bench deliberately does *not* exploit. A real ingestion
computes the head for the last position only, so `output.weight`'s 497.3 MiB is a
per-step cost there rather than a per-token one - the engine takes that one now
(`qe_forward_prefill`, and `make forward` asserts that the ordered walk lands
bit-identically where feeding the prompt one token at a time lands; what ships is the
walk, 1.20x sooner, asserted by choice of token instead - see below), while the bench chunks the head
anyway, because the harder case - batched decoding, and multi-token-prediction verify
windows, where every row of the window really does need a head - is the case the
engine has to be fast at. And the round is one layer's: across 48 layers the reuse
pattern is the same but the L3 is being shared by everything else in the step,
which is a reason to expect slightly less than the table, not more.

### What the engine took from this, and what it left

The kernel is wired into the engine's chunked projections and switched **on** by
default. Getting to that sentence cost discovering that the comparison which had said
otherwise was comparing one code path against itself, which is the first thing this
subsection has to say for itself.

What used to be claimed: a 1005-token prompt ingests at **65 ms/token** with
`qe_gemv_multi_t` off, 67 with every projection of 4 MiB and above blocked into one
walk and 66 with everything from 1 MiB up, "the trunk bits identical in all three",
so the chunk is worth a third of ingestion and the blocking on top of it nothing
measurable, and `QE_PROJ_MB` stays off.

What is measured, 384 tokens of prose at the default width of 8, both ways round
inside one load of the checkpoint:

| ingestion of the same 384 tokens                                        | ms/token |
|-------------------------------------------------------------------------|----------|
| ordered: T calls back to back (`--proj-chunk -1`, `QE_PROJ_MB=0`)         | 69.1     |
| blocked: one sweep per projection, what ships                             | **57.8** |

The walk is **1.20x** faster (the run before it printed 68.9 against 57.9, so the pair
reproduces to within a token's worth of noise), so the walk is the default and the
reproduces to within a token's worth of noise), so the walk is the default and the
switch is how you opt out of it. The width sweep under it, taken at the shipping
threshold, is the same measurement of the other knob, and it now runs past 8 because
`QE_CHUNK_MAX` does: the old cap of 8 was half buffer sizing and half the PLE prefetch
queue, which is now defined as the chunk width instead of being an independent 8. Three
runs of 384 tokens of one prompt, `ms/token`, the first two with the experts reached per
token and the third with them grouped - which is the shipping configuration:

| width    |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   | 10   | 11   | 12   | 13   | 14   | 15   | 16   |
|----------|------|------|------|------|------|------|------|------|------|----|------|------|------|------|------|------|
| run A    | 94.4 | 78.6 | 79.7 | 65.6 | 61.4 | 64.3 | 59.5 | 59.2 | 59.5 | 58.8 | 58.0 | 57.2 | 58.0 | 58.2 | 57.7 | **57.1** |
| run B    | 94.6 | 77.8 | 66.6 | 63.3 | 59.4 | 58.5 | 57.3 | 56.7 | 57.2 | 57.0 | 56.8 | 56.1 | 56.7 | 56.5 | 56.6 | 56.1 |
| run C    | 93.5 | 78.5 | 66.6 | 63.8 | 59.9 | 58.7 | 57.2 | 52.6 | 52.9 | 52.7 | 52.1 | 51.6 | 51.9 | 51.6 | 51.1 | **50.7** |

Two facts and one caution. The drop is from width 1 to about width 7 (94 → 57 ms, a
1.65x), and above that **16 beat 8 in all three runs** - by 3.5 %, 1.1 % and 3.6 % - so
"the widest this engine allows is also the fastest" survived being tested past the
number it was written for, and run C is the one that ships and prints `the default is
the fastest setting`. The caution is that the absolute rows are not comparable across
runs: the plateau moved from ~58 (A) to ~56.5 (B) to ~51.5 (C), run B's best width was
12, and run C has a 4.6 ms step between its own widths 7 and 8 that no switch explains.
Run-to-run drift is therefore larger than the difference between the widths, which is
why each row is compared only with itself. What the earlier 1..8 row also said - "one
direction the whole way, so the widest chunk this engine allows is also the fastest" -
was true of a sweep that stopped where the macro stopped, which is the difference between
a measurement and a habit. With the walk off the older sweep ran 94.3 down to 68.9,
which is the other half of the same statement: ordering and blocking are worth about a
fifth each at the old width, and they stack.

The narrowing is itself worth writing down, because it says the two knobs are not
independent. At width 8 the ordered form cost 71.1 ms/token and the walk 58.8 (1.21x);
at width 16 they are 64.9 and 56.3 (**1.15x**), and with the experts grouped the same
pair at 16 is 61.3 against 51.1 (1.20x). A wider chunk gives back part of what
blocking buys, which is what T back-to-back calls reading the same tensor through a
30 MiB L3 were always supposed to do - the blocking still wins, by less, and the
direction is the one to expect if the saving in the ordered form was cache reuse all
along.

**Why the old numbers said the opposite.** `proj()` asked whether a tensor was worth
blocking with `t->bytes < f->proj_chunk_bytes`, and both sides are unsigned. The
documented meaning of a threshold of 0 was "never block"; what it computed was
`bytes < 0`, which is never true, so the default of 0 blocked everything. Every one of
the 65 / 67 / 66 rows above therefore ran the walk, and the pair that was supposed to
disagree was one path run twice - which is also why "the trunk bits are identical in
all three" was both true and meaningless. The check that should have noticed was
`make forward`'s identity leg, and it copied the trunk into two buffers *before
feeding any tokens* and `memcmp`'d the copies: an invariant that could not have
failed. Both are fixed. What is left is a trade with a price on both sides rather than
a non-result:

- What the walk gives up is bit-identity. Two accumulator chains per token are not the
  single kernel's four per row, so a chunked Q8_0 projection reassociates. Over a
  951-token prompt the trunk the prompt ends in is a different set of bits and the
  worst move in a logit measured **1.6 against a head whose largest logit was 16.9** -
  9 % of that, which changes the answer sometimes and is not a rounding (on the
  384-token slice the same comparison prints 0.9 against 18.4, 5 %). `make forward`
  now asserts the two claims apart: the ordered walk against one token at a time byte
  for byte (and at every boundary a chunk could end at, per stage ablation), the
  default against the ordered walk by "the same next token, inside a tenth of the head's
  range", with the delta printed next to the range it has to be read against.
- What it buys is a fifth of ingestion, spent on the part of a coding turn that is
  reading a prompt rather than answering one.
- Every quality number this repository quotes - 3.56 nats/token teacher-forced, top-1
  agreement against the reference engine - was measured with the walk on, because of
  this same bug. Defaulting to the walk keeps those numbers describing the thing that
  ships.
- The tenth of the head's range has since turned out to be a hair trigger rather than a
  bound. The same comparison on the 598-token prompt reads 10 % with the one-pass
  attention walk of §5 and 9 % without it, on two builds that differ only in the order an
  attention sum adds its terms, and 12 % is recorded above for another head's range; the
  fraction a chaotic trunk lands on depends on which prompt was ingested and on no fixable
  thing. That assertion is now a sixth of the range, and the assertion above it, the same
  next token, is unchanged, because what this leg can actually detect is an order of
  magnitude rather than a re-roll.

One request through the server, the same 732-token prompt and 20 tokens of answer, twice
in a row on two loads that differed only in `--proj-chunk`: **42.9 s on the walk, 50.4 s
ordered, and the same continuation both times.** That is the pair at the level a client
feels it, and it is the reason the flag is an engine-start flag rather than a bench
knob: the difference is a sixth of a turn's wall time, and a coding agent on the other
end of a 2-minute silence timeout cares about exactly that.

What the table above is still right about is that a blocked projection stops being
DRAM-bound at T=8 and its GB/s column falls from 42.8 to 12.5, and that is a real
thing to give up. What the earlier reading got wrong is which side of that trade the
engine is on. Back-to-back calls do keep a ≤ 30 MiB tensor in the L3, but the widest
projection here is `attn_q`'s 33.4 MiB, 48 layers of dense projections arrive back to
back with everything else in the step competing for the same 30 MiB, and the blocked
form does not merely reuse bytes - it dequantises them once instead of T times, which
is the part the `multiT` rows below price at 3.8x. Ordering shares a cache. Blocking
shares the dequantisation, and on this host that was the thing still left to win.

What is left after it is the same thing this section has pointed at twice: this is
Q8_0. The routed experts are IQ3_S gate/up and IQ4_NL down, they are the larger and
more ALU-bound share of an ingested token, and they are reached per token rather than
through `proj()`, so the hook is different. Both halves of that are now written, and
the smaller half turned out to be the smaller saving: the arithmetic is in the section
after this one's kernel, and the 1.05x it is worth end to end is in its last paragraph.

### Chunking shares the bytes and not the arithmetic, and that is what 58 ms is

An ingested token at the default width costs 58 ms against a decoded token's 103: 6.7x
fewer bytes per token (§3's budget against this section's) arriving as 1.8x, and the
table above says why: at T=8 an ingested token owes something like 835
MiB against a decode's 5568, which at the same rate would be a 7x. Bytes stopped being
the constraint and arithmetic became it, and this engine had no kernel that shares one
dequantised weight across a chunk's T accumulators: `qe_gemv_multi_t` blocked a weight
walk against T activations and then did the T=1 arithmetic T times - widen, convert,
scale and multiply-accumulate, once per token. What follows is the price of that first,
measured where it could be measured without the checkpoint, and then the kernel that
closes it for Q8_0 and what it does not yet close for the rest.

`bench_alu`'s `multi1`, `multi2` and `multi4` walks are one Q8_0 span at three chunk
depths: one 34 B block dequantised into four float vectors, consumed by two accumulators
per token. They are checked against the double reference to 1e-3 relative and measure
9.0e-05, because two chains per token is a different summation from the shipped kernel's
four per row and was never going to agree bit for bit. The number they exist to print is
tokens/s for a chunk of T against a chunk of one, over identical bytes.

| threads, ms per pass | shipped `gemv` | `multi1` | `multi2` | `multi4` | T=2 | T=4 |
|---|---|---|---|---|---|---|
| default pool, 19 threads | 7.2 | 7.0 | 7.0 | 7.4 | 1.98x | 3.78x |
| eight performance cores | 9.1 | 11.4 | 11.6 | 12.3 | 1.96x | 3.72x |

The `multiT` columns are per pass, and a pass produces T tokens, so a chunk of four costs
1.8 ms per ingested token on the default pool and 3.1 ms on eight cores. The last two
columns are the decision: 1.00x would mean the fourth token cost exactly what the first
cost, so the fma units were the wall and sharing a dequant buys nothing; 4.00x would mean
four tokens arrived for the price of one walk. It comes back 1.98x and 3.78x at both
thread counts, which says the dequantisation is nearly free once it is shared, and that
what sits between 3.78 and 4.00 is the register pressure of carrying T accumulator pairs
alongside the dequantised block - the reason these walks hold to two chains per token and
are therefore the honest ceiling for a real kernel rather than an optimistic one. Against
the kernel the engine actually runs on the same bytes, four tokens through a shared
dequant is 3.9x on the default pool and 3.0x on eight performance cores, and the pinned
row is the one that also says the two-chain form is not free at low thread counts: at
T=1 it is a third slower than the shipped four-chain kernel (11.4 against 9.1 ms), so a
chunked kernel is worth writing as a second shape and not as a replacement for the first.

The kernel that came out of those two rows is `qe_gemv_q8_0_multi`, and `bench_alu`
times it under its own names - `shipped2`, `shipped4` - the real entry point through the
same row slice every other walk takes, so that the number in the table is the number the
engine gets rather than the number a sketch got.

| ms per pass | `multi2` | `shipped2` | `multi4` | `shipped4` | shipped T=2 | shipped T=4 |
|---|---|---|---|---|---|---|
| default pool, 19 threads | 7.0 | 7.0 | 7.4 | 7.3 | 1.98x | 3.79x |
| eight performance cores | 11.6 | 11.7 | 12.3 | 12.2 | 1.93x | 3.70x |

It gives away nothing to the walk that predicted it: 1.98x at two tokens and 3.79x at
four on the default pool, 1.93x and 3.70x pinned, against the candidate's 2.00x and
3.78x. Same shape, for the reason the candidate was that shape - two accumulator chains
per token, blocks handled in groups of four, and past T=4 the accumulators stop fitting
next to the dequantised block, which is the ceiling this instrument predicted and the
kernel then met. What it is not is bit-identical to T decode steps: two chains per token
is not the single kernel's four per row, so a chunked Q8_0 projection reassociates. That
is bounded and checked, by `bench_kern`'s `q8_0 multi-token` case (`make fast`) over 48
shapes at T=1..8 against a double reference, 1e-3 tolerance and 2.3e-05 worst, with the
gaps between activation rows filled so a kernel that ignored the stride is caught and the
output prefilled with a sentinel so a write outside T is caught; `make prefix` is what
checks a chunked trunk against a cold one. Two things follow that are worth stating
rather than implying. The kernel is reached from `qe_gemv_multi_t`, and the threshold
that reaches it is on by default - so the number that ships is the blocked one, and
the number above is the ratio that got it there. And the threshold is a field of the
open pass (`qe_forward_set_proj_chunk`), because a second load of an 87 GiB file is not
a way to ask a question twice: `make forward` ingests one slice of the prompt with the
walk off and then on, inside the load it already needed, and prints
`ordered T calls X ms/token, one blocked walk Y ms/token -> Z×`, the worst difference
the two leave in the logits, and which way that leaves the default.

**A kernel ratio is not an ingestion ratio, and now there are numbers for both.** 3.8x
is what one Q8_0 GEMV does for a token when the weight block is already in a core's L2
and the dequantisation is shared. What the engine pays per ingested token also contains
the DRAM efficiency the blocked walk gives up to be in that position, and the two come
out at **1.20x**: 69.1 ms a token for the T ordered calls, 57.8 for one blocked walk,
384 tokens of one prompt in one load (the table earlier in this section). Most of the
kernel ratio survives into the step, which is the opposite of what this section said
for two sessions running - and what it said was measured by an instrument that
compared the walk with itself, because a `bytes < 0` test under an unsigned `bytes` is
never true and the threshold's documented default of "off" was therefore "on". Read the
two claims as one sentence: the ratio is not the ingestion ratio, but on this host it
is 1.20x of it, which is a sixth of the time a coding agent's prompts take to read -
59 ms per ingested token over a 951-token prompt against 70 for the ordered form.

Two limits on what that buys, both load-bearing. It is Q8_0: this checkpoint's routed
experts are IQ3_S gate/up and IQ4_NL down, whose kernels run at 44 and 60 GB/s against
Q8_0's 74-83 (§1.2), which is kernels spending ALU cycles per weight that Q8_0 does not
spend - so there is more dequantisation there to amortise, on a share of an ingested
token that the Q8_0 projections do not have, and it is the same conversion still
unmade. `bench_alu` has no IQ3_S or IQ4_NL multi row, so the number for it does not
exist yet. And it is
one GEMV shape rather than a step: what a chunked kernel actually saves is the fraction of
an ingested token's 58 ms that is dequantising weights, and `make score` can now attribute
that fraction instead of assuming it, because the block stage is timed twice -
`block_proj`, the projections and the read-out, once per chunk and so divided by T, and
`block_core`, the conv ring, the delta rule and the cache writes, once per token and
divided by nothing. What the two measurements rule out together is the comfortable
reading: ingestion is not waiting on bytes, sharing a dequant is worth 3.8x on the bytes
it covers, and the rest of ingestion is a kernel rather than a cache.

### The experts are the other half of the dequantisation, and they have the kernel too

Q8_0 was the easy type. IQ3_S spends its time on a 256-entry grid lookup, a sign spread
and a compare per 32 weights, and IQ4_NL on an fp16 decode and a scale - which is why
their single-token kernels land at 44 and 60 GB/s against Q8_0's 74-83 (§1.2) on a
machine that reads at 76-79: they are not waiting for bytes, they are dequantising them.
That is the same asymmetry `qe_gemv_q8_0_multi` exploits, on a type where the thing being
shared is more expensive, over a share of an ingested token the dense projections do not
have. So there are two kernels now, `qe_gemv_iq4_nl_multi` and `qe_gemv_iq3_s_multi`,
reached through the same dtype-agnostic door the MoE already uses
(`qe_gemv_multi_ptrs_t`, plus `qe_gemv_multi_ptrs_has` for a caller that has to know
before it starts writing into accumulators).

**The register budget is the whole design, and it is the answer to "how deep should the
chunk be".** One 32-weight group dequantises into four float vectors, and the shape is
those vectors plus two accumulator chains per token: four accumulators pairs, one
dequantised block, three constants, sixteen ymm registers. IQ4_NL therefore goes to T=4;
IQ3_S, which needs three live constants instead of two, stops at T=3 and says so in the
source rather than spilling. A wider T is not a deeper kernel, it is more of these
kernels back to back over the same walk - `QE_MOE_SHARED_GROUP` is that fact, and the
dispatcher splits 8 into 4+4 and 16 into four groups. Which is the kernel-level half of
the user-facing question this section exists to answer: asking the expert kernel for a
chunk of 16 buys nothing over 4, and the measurement below says exactly that, flat from 4
to 16 because there is no sixth accumulator pair to be had.

`make fast` checks both against a double reference recomputed from the format's own
definition, over K = 32..2560 filtered by block divisibility, with the leading dimension
deliberately padded to `blocks + 1`, per-token-different activation rows, sentinels past
every output row, and T = 2..6: **`iq4_nl multi` 5.2e-06 worst relative over 35 shapes,
`iq3_s multi` 1.6e-05 over 50** (the Q8_0 form is 2.3e-05 over 48) at a 1e-3 tolerance,
because two chains per token is not the single kernel's four and was never going to agree
bit for bit. The shape of that test is three failure modes, not one: a wrong number (the
reassociation), a wrong *row* (the pointer-array form lets token A's activations meet
token B's weights, which is wrong by a whole term and invisible to a test whose T rows are
all equal), and an assumed packed layout (the padded leading dimension). A call it cannot
serve - T=1, a type with no form - returns -1 having written nothing, and that is asserted
too, because a half-written accumulator is worse than a refused call.

**What one shared dequantisation is worth on expert bytes** (`make moe`, 19 threads, every
row of the chunk routed to the same ten experts so that each bucket is T rows deep - real
routing makes shallower buckets, and the end of this section is about that):

| rows | per-row ms | shared ms | speedup |
|------|-----------|-----------|---------|
| 2 | 0.811 | 0.496 | **1.63x** |
| 4 | 1.618 | 0.900 | **1.80x** |
| 8 | 3.236 | 1.800 | **1.80x** |
| 16 | 6.490 | 3.614 | **1.80x** |

1.80x and flat, and a third run after the wiring landed agreed to within 0.02 ms
(0.813 → 0.497, 1.619 → 0.905, 3.247 → 1.802, 6.496 → 3.635: 1.63x, 1.79x, 1.80x,
1.79x). It is not Q8_0's 3.8x and it is
not meant to be: what sharing removes is the dequantisation, and on these types that was
a little under half the kernel rather than almost all of it, so what is left is the
fma work, which is per token and was never shared. The flat part is the register
argument above, measured: 4, 8 and 16 rows per bucket all come back 1.80x because they
all run the same four-row kernel more times. On the grouped pass as a whole - uniform
routing, so buckets of about one row and the byte saving doing most of the work - the
same leg improved from the 1.17x at T=8 this document used to quote to **1.34-1.42x at
T=4 and T=8 and 1.51-1.53x at T=16** (T=2 is 0.99-1.06x: two rows are not enough rows
for the grouping to pay for itself), and the grouped-versus-separate numerics gate, which the
shared kernel is now inside, came back at 1.7e-06 worst relative.

**The second wire.** The experts are not reached through `proj()`: `moe_block()` is
called once per row of the chunk, so a chunk was paying for T x top_k expert matrices
while the document two sections up was estimating its bytes as if it grouped them.
`moe_block_chunk()` is the other half: route every row, hand the chunk's rows to
`qe_moe_forward_multi` grouped by expert, and run the shared expert and its gate as
ordinary projections through `proj()` like every other dense stage. Routing is
`qe_moe_router` + `qe_moe_select` and not `qe_moe_route`, because the latter re-reads the
winning rows at f32 to make the mixing weights exactly the f32 router's, decode does not
do that, and a chunk that picked different weights than the same token fed alone would be
a change of arithmetic and not of scheduling; rows with fewer than top_k winners are
padded with expert -1, which the grouping skips. Below the projection threshold, with
the grouping switched off, this is `moe_block()` T times over, which is what makes the
A/B an A/B and not two different programs.

**And it is worth 1.05x of ingestion, which is the number this section used to guess at.**
Same 384 tokens, one load, one switch: at the shipping width of 16, 56.1 ms a token for
sixteen serial expert passes against **53.6** for one grouped pass (**1.05x**), 1.0e+00
apart in a logit against a scale of 22.8 - 5 % of the head's own range, the same next
token. The next run reproduced it at the same width as 54.3 against 51.0 (**1.06x**), and
the same pair at width 12 came back 1.05x, so it is not one reading of a noisy
instrument. It is much less than the 1.5x the expert pass alone reports, and the two
reconcile without residue: grouping makes the expert pass 1.4-1.5x faster, and once the
dense side is chunked that pass is about 15 % of an ingested token, so 0.29 x 15 % is the
4-5 % measured. `QE_MOE_GROUP_DEFAULT` is on for that, and `qe_forward_set_moe_grouped`
is the per-pass opt out; the identity leg turns it off alongside the projection walk,
because the two reassociations are separate claims and the leg that wants "the chunk and
nothing else" cannot have both. What adding it did *not* do is widen the gap between the
chunk and the ordered walk: that measures 2.8e+00 of a 20.3 logit scale with the experts
reached per token and 2.5e+00 of the same scale with them grouped, the same statement
inside the noise of a 598-token prompt.

Summed end to end on one prompt, the two changes together - width 8 with the experts
reached per token against width 16 with them grouped - ingest at 60 against **51**
ms/token: **1.18x** of prompt reading, on a machine whose decode step did not move.

Two things this does not change, both worth saying before someone reads a decode win into
them. A decode step is one row, so `T == 1` and none of this is in its path: 109 ms per
emitted token before and after, and the 19.9 % of a step's bytes that the routed experts
are is still read ten times per token. And the 1.05x is prefill-only, on a workload that
is already 57 ms a token rather than 109: it buys 2.5 ms of prompt reading, which is
real, and it is not the 5x the byte ratio promises, because a chunk this wide has taken
the experts from bandwidth-bound to something else.

### How this number was made trustworthy

Five instruments, and each of the first four was added because it caught a real
lie. The first run of this bench printed `dense 7.6 MiB, chunked 56.7 MiB` at T=1 -
a **-643%** "saving" - and everything below is what replaced that.

1. **A step-shaped round, not a per-tensor loop.** A ≤ 30 MiB matrix measured in a
   tight loop lives in the L3, and both paths correctly report ~0 DRAM bytes; that
   run printed `0.0 MiB` and a negative saving for every small tensor. A 566 MiB
   round is nineteen times the L3, so nothing survives from one round to the next.
2. **`read()` on a perf fd returns 8 bytes.** Asking for 24 without a
   `read_format` gets a short read, which under `if (read(...) == 24)` means every
   counter silently reads zero - the most convincing possible failure, because 0
   bytes missed is a beautiful optimisation. Harvest reads exactly one `u64`.
3. **A hybrid package has two monitoring units, so arm both.** `cpu_core` (type 4)
   and `cpu_atom` (type 8) each export the LLC-miss event `0x2e/0x41`, and a
   task-bound event is not scheduled while its thread runs on the other type.
   Counting only `cpu_core` lost nine of sixteen threads: their traffic simply was
   not in the sum, and nothing said so. Every thread now arms both fds and the two
   reads are added; the events are disjoint by construction, and the calibration
   below is what would notice if they were not.
4. **The counter's scale comes from a round whose size is known without a counter.**
   A dense T=1 round touches 566 MiB because the tensor table says so. Counting it
   with both PMUs gives 8680402 misses, i.e. **68.3 bytes per miss** against a
   64-byte cache line, and an independent 256 MiB walk of known size gives 66.5 -
   two measurements, no inputs, agreeing to 3% on a quantity the bench never tells
   the kernel. Counting only the performance cores instead gave 622 bytes per miss,
   which is how the missing nine threads became visible: the event is line-granular,
   so a scale that is off by ten times is a coverage failure, not a property of the
   chip.
5. **Three gates that use no calibration at all.** At T=1 the two paths being
   compared are one identical walk apiece, and they came back at 553.7 against
   548.9 MiB (0.9%), with the chunked figure never part of the calibration. A dense
   round is the same 566 MiB at every T, so all five dense rows must agree: the
   furthest strayed 14%, in the direction (down) that the L3 holding the small
   tensors at T>1 predicts. And a dense row cannot exceed the round it walks. If any
   gate fails - a silent thread, a dense row 10% over, the two paths disagreeing at
   T=1 - the bench prints `INCOMPLETE` and says the byte columns are relative and
   must not be quoted, while noting that the times still stand, because a stopwatch
   is not a hybrid PMU. An earlier gate checked only the second of these, waved a 17%
   straying through as acceptable, and let a run with nine dead threads print a
   **-15.1%** saving at T=1.
6. **Bit-identity, before any of it means anything.** For every tensor, T separate
   `qe_gemv_t()` calls against one `qe_gemv_multi_t()` over the same T activations,
   compared with `memcmp`: a chunking change that quietly moved a rounding boundary
   is the worst kind of optimisation and would not be caught by anything that only
   looks at loss. Mismatch aborts with the element index and both values. One type is
   exempt, by construction rather than by exception: at T ≥ 2 the dispatcher sends
   Q8_0 through the shared-dequant kernel, which cannot sum in the order T separate
   dequantisations do. What that exemption is allowed to be was first measured as the
   distance between the two rounded paths, held to 1e-3 relative, and it failed the
   first time it saw weights instead of synthetic data: element 72469 of
   `blk.0.attn_qkv.weight`, 5.48362732e-06 against 5.27501106e-06, which is 0.038 of
   itself and 2.4e-07 of that tensor's results. Both statements are about the same two
   numbers, and only one of them is a fact about the kernel - a dot product whose 2560
   terms cancel to 5e-06 is owed a rounding unit like any other, so the distance
   between two rounded answers is not a gate because it has no scale. The check now
   recomputes the dot exactly instead, in long double from the format's own definition
   (fp16 decoded by the IEEE-754 rule rather than by this tree's converter, so a wrong
   block scale cannot be wrong on both sides), and gates each path's error against
   `sum|x*w|`, the quantity a float32 accumulation of that dot is permitted to be wrong
   by. This checkpoint moves 418691 of 477184 Q8_0 results, the furthest by 4.4e-07 of
   their own scale, and against the exact value the two paths sit at 1.7e-07 and
   4.7e-08 of `sum|x*w|` - a bound of 1e-3 with five orders of magnitude left over,
   which is what a moved rounding boundary looks like next to a row read from the wrong
   offset. How the two paths compare to each other is printed rather than gated
   tightly, because the shared sweep keeps two accumulator chains where the single-row
   kernel keeps four and is entitled to sum a little differently: worst 2x on this file
   by RMS, and on the hyper-connection the shared sweep was the more accurate of the
   two. The honest form of a gate has to name the boundary it lets move, say what unit
   it moves in, and measure how far it moved; a gate that merely passes would have been
   deleted along with the kernel.

### Which cores to run on

Pinning is a measuring device here, not a serving policy - and the comparison has
to be made on the same workload, because the round is Q6_K-dominated and its rate
is not the Q8_0 sweep's. Same 566 MiB round, dense T=1: eight threads confined to
the performance cores walk it in 15.53 ms/token (38.2 GB/s), sixteen threads across
both core types in 12.68 ms/token (45.8 GB/s). The twelve efficient cores are worth
22% on this mix, and on pure Q8_0 the same 8→16 step is worth 42% (§1.1: 56.5 →
80.0 GB/s), which is the shape of memory-bound work - the extra cores buy queue
slots, not clock cycles. The engine therefore runs unpinned across every core it is
allowed to use; `qe_pool_init` is handed no cpu list, and only the bench pins, and
only so that the counter can be checked against something.

**How many cores is a bigger deal than which ones, and the answer is not twenty.**
The pool's workers spin rather than sleep, so a pool as wide as the affinity mask
leaves the host nothing, and the first competing runnable task preempts a spinning
worker whose barrier every other thread is standing in. Back-to-back trivial
dispatches, `make pool` (median of 48, because one scheduler stall out of 48 moves
an average by more than the handshake does):

| threads | hot dispatch, median | worst |
|---|---|---|
| 20 (= cpus) | 69 us | 2487 us |
| 19 | 1.6 us | 2.0 us |
| 18 | 1.5 us | 1.9 us |
| 16 | 1.6 us | 1.9 us |
| 12 | 1.2 us | 1.5 us |
| 4 | 0.6 us | - |

Two thousand times the cost of the same handshake for the last core. A token is
~800 dispatches, so on paper that is most of the difference between the byte budget
and the measured step time, and it is: the same binary, corpus and weights score
identically at 20 and 19 threads (nll 2.2425, 32/63 argmax hits) and the step takes
**106.2 ms instead of 125.2**, 15% for one fewer thread. 18 threads is 106.6, i.e.
the plateau is flat below the cliff, so the pool defaults to `qe_cpus() - 1`
(`QE_THREADS_AUTO`) rather than something conservatively narrower. `--threads N`
still means exactly N, and `--threads auto` asks for the engine's choice.

**Which core is left out is not the engine's decision, and pinning to make it one was
measured and costs 2-3x the handshake.** What this paragraph used to claim - that the
excluded core is "whichever one the affinity mask lists last", and therefore an
efficient one - presumes the engine asked for particular cpus. It does not:
`qe_pool_init` is handed no list, so the scheduler places the workers and nothing in
the engine knows which logical cpu a thread last touched. The sentence was a story
about a mechanism that was not there.

The mechanism now exists, so the claim could be tested instead of repeated.
`QE_PIN=1` builds the slice table and the affinity mask from `cpu_capacity`
(performance cores first, ties by cpu number), which is the only configuration where
"the left-out core is a slow one" is a statement about the engine at all. Two runs
each way, `QE_PIN=1 ./bin/bench_pool auto 120` against `QE_PIN=0`, hot-dispatch
median:

| pool | median, two runs | at the full mask (`all`) |
|---|---|---|
| unpinned (default) | 2.1 us, 2.9 us | 4.9 us |
| pinned, `cpu_capacity` order | 5.7 us, 4.8 us | 6.2 us |

Pinning is worse at both widths. A worker confined to one cpu cannot move when
something else wakes up on it, and a spinning pool's barrier waits for precisely that
worker; which class the core belongs to is worth less than the scheduler's freedom to
place it. That is the `cpu_capacity` finding below arrived at from the other side, and
it is why the answer to "can we build threads specific per core and pin them" is no:
the per-core asymmetry is real (next table) and none of it is reachable by placement,
because none of it changes how many bytes a token reads. `QE_PIN` stays opt-in,
`qe_pool_init` still gets no cpu list by default, and what running one thread short of
the mask gives up is a core for the operating system, not a named one.

**The cache asymmetry between the two classes is real and is not exploitable here.**
From `/sys/devices/system/cpu/cpuN/cache/index2`:

| cpus | L2 | shared by |
|---|---|---|
| 0-7 (P, 5300 MHz) | 3072 KiB | one cpu each |
| 8-11, 12-15, 16-19 (E, 4600 MHz) | 4096 KiB | four cpus per cluster |
| 0-19 | 30720 KiB L3 | all twenty |

Four E cores therefore share one 4 MiB L2, which looks like a way to read an expert
row once and hand it to four threads. A decode GEMV cannot use it: every weight row a
token reads is read exactly once by exactly one thread, so there is no reuse to
share - the same fact that makes the grouped MoE pass (`src/moe.c` groups the routed
experts by counting sort) a win for *bytes per token* and not for bytes per read. The
asymmetry that does show up is throughput, and §1.1's rates already price it.

**Rows are divided equally, not by `cpu_capacity`, and that is worth 4% of a token.**
The kernel publishes a per-CPU throughput figure, and on this host it says the two
classes are 1.30x apart (1011 against 783), so charging thread ids by it looked like
the obvious correction for a pool wide enough to straddle both classes. It is the
right ruler for a loop whose cost is arithmetic and the wrong one for the loops this
engine runs: a dispatch that saturates the memory fabric does not care which core
asked for the bytes, and handing a P-core 1.30x the rows of an E-core makes the
surplus pure makespan. `bin/bench_moe` races the two divisions inside one process,
interleaved round by round, over identical bytes through identical kernels (GB/s, 19
threads; "ceiling" is `make fast`'s rate for one contiguous matrix, one call per
thread):

| dispatch (19 threads) | `QE_SLICE=capacity` | equal (default) | ceiling |
|---|---|---|---|
| gate+up IQ3_S, 13 MiB | 37.5 | 41.5 | 51.1 |
| gate+up IQ3_S, 54 MiB | 44.3 | 50.4 | 51.1 |
| gate+up IQ3_S, 215 MiB | 44.0 | 50.6 | 51.1 |
| down IQ4_NL, 9 MiB | 59.2 | 61.8 | 70.0 |
| down IQ4_NL, 35 MiB | 60.3 | 64.1 | 70.0 |
| down IQ4_NL, 141 MiB | 60.3 | 68.9 | 70.0 |

An equal split reaches the ceiling and the weighted one does not, at any width, for
either type. The engine agrees with it: same binary, same corpus, same weights, nll
identical to four decimals at every one of the 255 positions of the reference corpus,
**103.4 and 104.3 ms a token against 107.5 and 110.3** (two runs each way, `make
score`), with the routed expert stage going 35.5-35.6 -> 32.0-32.3 ms and 0.3-0.6 ms
off every other GEMV stage (§14).

The honest wrinkle is that the synthetic 22 MiB expert round of `make moe` still
prefers the weights at 19 threads, 44.4 against 39.8 GB/s, and prefers equal at 12
(33.9 against 34.4). What each workload does per row explains it: the round is two
quantised decoders, a silu and a cross-thread reduce, which is ALU work enough for
clock to buy bytes, while a token is bandwidth. `cpu_capacity` is an ALU measurement,
and the step is not ALU-limited. So equal shares are the default and
`QE_SLICE=capacity` builds the weights on request - the escape hatch for a pool
narrower than the machine's memory ceiling, for an ALU-limited stage, and for
measuring the assumption on a host this file cannot test.

**Two dispatch shapes, because a barrier per matrix is not the only thing a pool is
for.** `qe_pool_run` hands one function to N threads and waits for all of them, which
is what every stage of a decode step uses and always will. The other shape,
`qe_pool_run_items`, is one dispatch carrying a *stage graph*: up to 32 items, each
declaring how many slots it divides into and which other items it waits for, walked by
whoever woke up to it, a worker taking the lowest-numbered ready slot of any item by
compare-exchange and re-reading the completion mask after every one. Independent items
therefore share the machine instead of queueing behind each other, and a short stage
lends its threads to a long one; a dependent item waits on slots that have *returned*,
which is why there are two counters per item and not one. The ordinary dispatch is
untouched - a decode step never builds a graph - and its handshake is unchanged at
**1.4-1.8 us median** at 19 threads (`make pool`, same run as the graph legs).

`make pool` is the gate for it, and it is a gate with a history. A graph rep runs four
items whose slots spin so the races have time to be seen, and asserts that every slot
ran exactly once, that no dependent item started on a predecessor that was merely
claimed, and that two independent items were *resident at the same moment* (an item set
executed one item after another is deterministic, passes the first two, and is the whole
failure mode of the primitive): 60 dispatches, 2460 slots, widest overlap 2 items. Then
comes the leg that is not about one graph at all: a graph, a function that churns the
stack the way the next part of a layer would, and an ordinary full-width dispatch, 240
rounds, checking that every worker turns up in every one of those dispatches and that no
round stalls. That leg is the reason the graph publishes its items into storage the pool
owns rather than into the caller's array: an engine read that array after the dispatch
that built it had returned, mistook a dead frame's contents for a stage that still had
work, and parked one worker on it permanently - which is invisible in a bench where the
frame still holds the same numbers, and is a hang seven minutes into a prompt in a
forward pass, because a worker that never leaves a graph never reports a slice of the
dispatch after it either. The store is the pool's now, every worker that reads it
announces itself, and the master clears the graph and waits for the readers to leave
before it returns.

A third failure of that family was a segfault, and it was in the handshake the graph work
explicitly left alone. A worker woke on a generation and then read `fn`, `arg` and
`nthreads` *live* - not the ones that generation had published. A worker that was not in
the dispatch it woke for therefore ran the one that had followed it, and added its count to
*that* dispatch's completion total, which let its master return while a real participant was
still inside a task that lived on the caller's stack. `make pool` is the tell, because its
cold loop varies the dispatch width every time round and a varying width is what makes a
non-participant out of a lagging worker: on a loaded host twelve runs of
`./bin/bench_pool auto 120` faulted in the bench's own `bump`, backtrace in
`qe_worker_main`, on a `job` pointer read out of a frame the master had moved on from. The
same race is the innocent half of the two soft symptoms this bench has ever printed - a slice
that ran one time too many, and a graph leg reporting zero slots because the dispatch after
the graph had already been spent. The dispatch a generation names is now a stamped snapshot:
the master retires the stamp, writes the fields, sets the stamp, and bumps the generation
last; a worker copies the fields and throws the copy away unless the stamp still names the
generation it woke for. Throwing one away can only ever be a worker nobody is waiting for,
because a participant's own count is what let the master move on, which is the property the
old code was written to keep and did not. Twelve runs before the fix, twelve faults; twelve
after, no fault and no FAIL, on the same loaded host with an engine run for company.
`src/mem.c` has it in the comment above `qe_dispatch`.

### The width went to 64, the experts got grouped, and both walls are now named

Two switches and one instrument later, an ingested token was **35 ms** where this section
started from 58 and then 94.9; the subsection after this one moves it again. None of it is a
new kernel.

**The width.** The sweep `make forward` prints on one load reads 94.9 ms/token at width
1, 45.5 at 8, 41.1 at 16, 38.1 at 32, **34.7 at 64**, 34.8 at 96 and 35.3 at 128, so the
default moved 16 → 64: **1.18x** on ingestion for a constant. The earlier "flat above
width 7" reading behind the old default was taken with the expert pass running serially,
which is the one stage whose cost keeps falling with the width; it was a scheduling
ceiling mistaken for a curve. The row is monotone in direction and noisy in detail (96 is
level with 64, and the 128 row read 40.7 against the profiler's own sum of 33.66 in one
load and 35.3 against 34.03 in the next), so what it supports is "wider keeps paying to
about 64" and not "64 beats 48". Width is scheduling and not arithmetic only with the walk
off: a width-64 chunk differs from the ordered walk by a reassociation, and that is what
the paragraph below this one had to re-price.

**The experts.** `moe_block_chunk` now runs a chunk's rows through the grouped pass, and
`make forward` A/Bs it in the same load at four widths: **46.8 → 30.5 ms/token at the
shipping width 512 (1.53x)**, 1.39x at 128, 1.35x at 64 and 1.26x at 32
(`/tmp/f9_wide.log`; the load that first swept this leg read 47.1 → 34.7 at 64, 1.36x, and
1.16x at 16). The size of the win is a function of the width it runs at, which is the whole
reason the two items above belong in one sentence: grouping reads the *distinct* experts of a
window, and
the window is the chunk. At width 64 the stage was 19.18 ms of a 33.74 ms token, so it
was 57 % of an ingested token and not the ~15 % it looked like when the sweep stopped at
16; at the width that ships it is 15.41 ms of a 28.06 ms profiler sum, still half
(`/tmp/f19_wide.log`), and 13.81 ms of a 26.42 ms sum on the integer walk the paragraph
below the router one is about. What that half is actually waiting on is re-named further
down this section.

A batched router came with it and is worth nothing, which is recorded rather than
buried: `qe_moe_router_multi` scores a chunk's T rows against the 512x2560 router in one
dispatch instead of T of them, it is bit-identical by construction (neither F32 nor BF16
has a shared-dequant form, so it reassociates nothing), and it neither added nor removed
time from the stage it lives in (25.11 → 25.13 ms/token of `moe_ffn` at width 16). It is
kept because a per-chunk router is the shape the MoE needs for other reasons, and it is
claimed for nothing.

**The integer expert walk, and the price it ships at.** The stage then stopped being a dequantisation loop. A chunk's
activations are recoded to int8 once per layer - one panel per thread per layer, `p`
holding `-16 x sum` so the +128 bias cancels across the eight `vpdpbusd` lanes and nothing
is subtracted at the exit - and a bucket's expert columns are walked against that panel for
Q8_0, IQ4_NL and IQ3_S (`QE_I8_MOE`, default on, `qe_moe_set_int8` for a bench). It is
**0.824x of the expert stage** and **1.100x of the whole step** at the shipping width
(13.48 against 16.35 ms of `moe_ffn`, 28.32 against 31.15 ms/token, `/tmp/f16_wide.log`),
and what it pays for that with is 5.2 logits against a head whose largest logit was 25.0:
21 % of the head's own range, the same next token, and *less* than the bucket reordering
that already ships costs at the same width (34 %) or the projection walk (26 %).

That is `make forward`'s one prompt, and it is not enough. `bench_score --ingest 5` runs the
same A/B over five 512-token windows of the corpus and prices the thing a user has: the loss
of the token that follows the window, whether the greedy choice moved, and the worst move in
the head as a fraction of that head's range (`/tmp/ingest2.log`). The recode charges **1.10
nats mean and 4.69 worst** and moves the greedy choice **once in five**; the grouped expert
pass this engine already ships, same instrument and same windows, charges **0.29 mean and
0.49 worst** and moves it none. Both sit inside the half-of-the-head's-range bound the
ingestion legs hold a schedule change to (30 % against 11 %), and that bound is the right
price for a schedule; for a change of *number* the price is the neighbour it ships next to.
That is the arithmetic that first had `moe_int8_on()` default off, and ideas.md §7.13 is what
changed the verdict: the format-matched instrument put the expert stage's entire remaining
headroom, walked with nothing expert about it, at 1.2-1.6x, and this walk is 1.21x of the
stage by itself, so the choice was never this against a safer kernel - the row-pair and
codebook rewrites that were supposed to be worth 3x measured dead - but this against no
kernel. So `moe_int8_on()` now defaults on and `QE_I8_MOE=0` is the way back, the loss bill
above is what it costs rather than what cancels it, and a panel that carries less activation
error at the same multiply rate is still the item that would make it cheap.
Three things about the walk are still worth the paragraphs they cost. It was the fill order, not the integer
arithmetic: building the panel inside the column-outer loop the walks are written in is
12-26 % *slower* than the float walk it replaces, and hoisting it out of that loop is what
made it 1.17-1.29x. It is checked against two different promises, because a lossy kernel
has two: the quantiser's own (the recode reproduces the int8 values exactly) and the
kernel's against its own recoded inputs (4.1e-05 for Q8_0, 6.9e-05 for IQ4_NL, 5.4e-04 for
IQ3_S over 224, 448 and 544 shapes, `make fast`), never relative to a double dot, which is
meaningless once the inputs were deliberately changed. And the check found a live
out-of-bounds read on the first run: the Q8_0 integer walk took a tile of two for a tail of
one and read `ys[t+1]`, a pointer no caller ever passed, for every T the tiles do not divide
- 3, 5, 7, 9, 33, which is exactly the width of a partly-filled expert bucket. The IQ walks
already re-took the previous row; Q8_0 now does too. ideas.md §7.11 has the loop-order
table and the two bugs.

**What the reassociation cost, and the gate that had to be re-priced.** The delta between
the shipping pass and the ordered walk is chaotic by construction, and a wider window
reassociates more of it: the same 598-token prompt read **12 % of the head's logit range
at width 16 and 16 % at width 64**, against an assertion whose own message has always
said "a sixth" while its code said 0.15. The bound is now the sixth the message names,
and the reasoning is in the comment next to it. Two things are worth recording about
pricing it. What does price it: the same next token, and a new fifth ingestion leg that
generates eight greedy steps from the prompt twice, once with the shipping pass and once
one token at a time, and asserts the ids are identical. What does not: **teacher-forced
loss cannot see a chunk at all**, because `bench_score` scores by stepping one token at a
time (`bench_score.c:391`); two loss runs that differ only in width are two identical
computations, and reading 2.5369 nats out of both and calling it an A/B is exactly the
kind of claim this file exists to prevent. The blind spot is now an instrument rather than
a caveat: `bench_score --ingest N` prefills each 512-token window of the corpus twice, once
with the integer expert walk off and once with it on, and prints the next-token NLL of both,
whether the greedy token moved, and the worst head gap as a fraction of the head's own
scale. A 512-token window re-priced it again: the
walk leg reads 9 % of the head's range there where it reads 3 % at the 384-token slice the
suite ships with, and the grouped-expert leg reads 33 % where it reads 4 % - same next
token and same eight greedy step ids at both slices, because the quantity is how far the
two trunks have walked apart and not how many additions were reordered. Two bounds moved
for it, a third on the walk and a half on grouping, each with the recorded fractions next
to the code, which is the only honest place to put a number that is a calibration at a
slice rather than a bound on a bug.

**The instrument that settled two open questions.** `make walk` walks synthetic tensors
in the model's own shapes, so neither answer needs the checkpoint. Row blocking: 64 KiB,
256 KiB, the shipping 1 MiB and 8 MiB land within a few percent of one another on every
shape at every width the chunk runs at (`attn_qkv` at T=64: 476, 474, 473, 470 GMAC/s for
64 KiB, 256 KiB, the shipping 1 MiB and 8 MiB),
so `QE_ROW_BLOCK` is not a knob with anything left on it. Requantising the dense Q8_0
projections: IQ4_NL is **1.74x** at T=1 and 1.05-1.10x at T=16, then 0.98-1.00x at 32 and
0.96-1.02x at 64, IQ3_S never better after that - because at the shipping width the walk
holds 473 GMAC/s while reading 7.9 GB/s of Q8_0, a tenth of the read ceiling, so the
dense half is limited by arithmetic and 53 % fewer bytes buy nothing. It is a decode item
(§4.1), and the ingestion answer is measured rather than assumed. Only the ratios are
quotable: the same instrument on a slower-running machine state reads 326 GMAC/s for the
T=64 row above and 5.8 GB/s under it, and the conclusion - bytes at a tenth of the
ceiling, so arithmetic is the limit - is the same either way.

### Then the width went to 512, and two kernels had to be fixed to get there

`QE_CHUNK_MAX` moved 128 → 512 on a byte argument - what a chunk shares is a tensor's
bytes, and the routed experts are 57 GiB of this checkpoint - and the first sweep that
went there found the opposite of the argument: **42.9 ms a token at width 512 against
34.2 at width 64**, a 22 % penalty for the widest chunk the engine allows, on shapes
`make walk` had never once called slow.

**The dense walk was re-reading its own activations.** `qe_gemv_q8_0_multi` looped weight
rows outside and token groups inside. That order is free while the activation panel fits
somewhere and ruinous when it does not: at T=512 and K=2560 the panel is 5.24 MB, so every
one of the weight rows re-read all of it, and `attn_qkv` fell from 476 GMAC/s at T=64 to
195 at T=512. Row tiles outside, groups inside, tile sized to the L2 budget (`QE_MULTI_KIB`,
256 KiB - 66 rows at K=2560, 10 at K=6144, 740 at K=320), and the panel is read once per
tile instead of once per row: **195 → 518 GMAC/s** at T=512, with `ssm_out` 151 → 467,
`hc_down` 153 → 405, `hc_up` 233 → 596, and the T=1 column unchanged, which is the check
that says the fix is about the panel and not about the arithmetic. Bit-identical, and
asserted that way rather than argued: the same 512-token slice prints the same four logit
deltas to the tenth before and after the change (`/tmp/f5_wide.log`, `/tmp/f6_wide.log`).
The pairing below took the rest of that width dependence out: the same table now reads
`attn_qkv` 630 GMAC/s at T=64 and **631 at T=512**, with `ssm_out` 588 and `hc_down` 443,
because a tile that reads the panel once per two rows is half as sensitive to how many rows
are in the chunk (`make walk`, `/tmp/walk_full_f19d.log`).

**The expert walk was re-dequantising every bucket.** `make moe`'s shared-dequant table
showed cost growing with `ceil(rows/8)` rather than with rows: past eight rows the
accumulators no longer seat in the register file next to the dequantised block, so the
group splits and the tail pays for the quantiser again. A probe over 64 experts x 640 rows
of IQ3_S fits `cost(np) = D + np*F` with D one walk of the weights and F one token, which
makes the ninth token of a bucket cost 8.7x what the tenth costs - and a 512-token window
routes an average of ten rows to an expert, so the average bucket was paying two walks.

The fix was to make the two things that want the same sixteen registers swap places above
eight rows: the block stays in registers and the sums move to an aligned L1 line
(`QE_IQ_PARK_MAX`, `src/kern.h`), so a bucket of up to 32 rows is one walk. It works on the
kernel and it loses on the machine. The probe is monotone to thirty-two tokens and the
uniform-routing half of `make moe` agrees (512 rows 125.7 → 109.6 ms, 256 rows 80.7 → 56.7),
but the shared-dequant half of the same run - one bucket, which is the shape a park is for -
is slower at every row count above eight (256 rows 43.44 → 55.53, 512 rows 89.69 → 115.13),
and in the engine the switch cost the expert stage **16 %** at the shipping width
(`moe_ffn` 14.34 → 16.64 ms/token, `/tmp/f6_wide.log` against `/tmp/f7_wide.log`, where
every other stage of the two runs agrees to 2 %) and 8-10 % at widths 64 and 128, and
nothing where buckets stay under nine. A parked group of thirty-two is also thirty-two
activation rows the cores have to keep: one K-row each, 320 KiB per thread against 80 KiB at
eight, 6 MiB across the pool. So the forms are in the tree and switched off
(`QE_IQ_PARK_DEFAULT 0`, `QE_PARK=` to move it without a rebuild, and `make forward` printing
the cap curve on one load), and this is recorded in full rather than dropped because the
kernel item ideas.md §7.8 keeps wants walks wider than eight and has to answer this
measurement first. ideas.md §7.2b has the two-tables-in-one-run story.

**The sweep, and what the default is now.** 92.2 ms/token at width 1, 45.6 at 8, 41.4 at
16, 37.2 at 32, 35.1 at 64, 33.4 at 96, 33.1 at 128, 34.1 at 192, 32.2 at 256, 31.8 at 384,
**31.1 at 512** (`/tmp/f9_wide.log`; the two loads before it read 30.9 and 31.4 in that row,
which is the 1.6 % a load of this checkpoint is worth), so `QE_CHUNK_DEFAULT` moved 64 → 512
and ingestion was **1.10x** on the old plateau and **1.32x** on the 41.1 ms this section
started from. The same sweep on the shipping engine - the integer expert walk, one
activation scale for a whole row instead of one per 32 values, the same integer walk in
front of the dense projections, and that stage's seven dispatches collapsed into one graph, all
four on - reads 93.6 ms/token at width 1, 36.0 at 8, 31.1 at 16, 27.1 at 32, 24.7 at 64, 23.8 at
128 and **23.2 from 192 through 512**, where the row is flat to the printed decimal: four widths
tie and the harness's own verdict line names 192 the fastest of them, so the default is the
widest of a tie and not the winner (`/tmp/f25_wide.log`). The three recodes were defaults
already when the sweep last moved; what moved it from 23.8 to 23.2 is the schedule, and there is
no second table to compare any more: `QE_I8_MOE=0 QE_I8_RS=0 QE_I8_PROJ=0` is the engine this
section described an hour ago, and it read 27.5 in that row on the load that measured them
(`/tmp/f22_wide.log`), with `QE_MOE_CO=0` the way back off the schedule. Not monotone in
detail - 128 still reads slower than everything from 192 on, which is where the plateau starts
rather than a run of one - and the price
is memory: 4.8 GB
of grouped-expert partials across 48 layers and 210 MiB of chunk scratch, next to 87.2 GiB
of resident checkpoint, which is the number that decides whether the ceiling goes past 512.
Two switches since, both on: the scale exit hoisted out of the Q8_0 walk's token loop and the
signed dot behind it, worth 21.9 → 18.2 → 17.4 → **17.3 ms/token** across four loads, and the
sweep is monotone the whole way to its end - 35.2 at 8, 24.1 at 32, 20.5 at 64, 18.8 at 128, 17.8
at 256 and 384 - stopping at the default, which the leg's own verdict line now calls the fastest
setting (`/tmp/f29_wide.log`, ideas.md §7.17).

**The two walls, after both fixes.** An ingested token is 16.6 ms of profiler
sum at the shipping width: 7.25 ms of experts and 2.97 of dense projections, with the block
core and the two hyper-connection sweeps underneath them (`/tmp/f29_wide.log`; the f25 load read
22.40 with the same four stages, and the two recodes and the hoist are the difference). The old reading of the
first two as one byte bill and one arithmetic bill was an artefact of the width it was taken
at. At 512 rows the expert stage reads 106.6 MiB of *distinct* expert bytes per ingested token,
which is a 2.5 ms floor at the scattered-read rate against the 7.25 ms spent, so the experts
are not waiting on DRAM either. The ratio this section used to quote for them is withdrawn:
207 GMAC/s against a dense Q8_0 row's 637 changes the format, the number of token rows sharing
one weight read and the gather at the same time (ideas.md §7.13). What one process does measure
is the routed pass at 205 GMAC/s against 284 for the same codebook walked uniformly and 330 for
a uniform Q8_0 row, and the walk that ships is 1.18x of that routed leg - so the expert half is
nearly closed, and the dense half is where the unexplained gap is: 343 GMAC/s in the engine
against the 518 the same kernel gives `make walk` alone on the same shape, 1.51x. That isolated
figure has since moved away rather than closer - 773 GMAC/s for the walk with its scale exit
hoisted and its dot signed, and 1737 for the same walk reading one scale per output row of the
matrix instead of one per 32, the latter measured and unwired (ideas.md §7.17, §7.18) - which is
the gap this paragraph is about restated as a bigger number. What is not
left is a wider walk. A ten-row integer form for both expert codebooks is built, and neither
place it could be used pays: handing it a bucket is 8-25 % worse, because a bucket of sixteen
becomes three passes of an expert's weights where eight makes two, and handing it a 512-row
dense walk, where the decode it saves is a real 19 %, is a wash against the eight-row form it
replaces (`/tmp/moe_t10.log`, `/tmp/wide_walk.log`, ideas.md §7.14). Together with the
0.95-1.04x of the two-output-row integer tile (`QE_I8_R`, §7.12) that is the register file
answering: ten and sixteen accumulators on a machine with AVX-512 fused off is no registers
left for the dequantised block, so the remaining structure is a tiled integer gemm and not a
fifth form - and that form is now measured at 1.84x of the walk that ships, three output rows
sharing one activation load, needing a requantised copy of the dense matrices to reach an engine
(ideas.md §7.18). That is also the end of that family rather than its beginning: five row counts
are instantiated now, raced twice, three-by-four ties the one other tile that fits the
`R*T + R + 1 <= 16` accumulator budget exactly (1.4 % ahead of it at the width a chunk ingests at in
one run, dead level with it in the next), the two forms the budget excludes are 48-63 % and 37-78 %
off, and the winning tile with its weights held resident in the L3 is **1.00x** of the same walk
reading DRAM at 512 tokens in both runs - 3.4 GB/s of a 76-79 GB/s bus - which is a walk sitting at
an issue-rate ceiling, where neither a layout nor a faster memory has anything left to sell
(ideas.md §7.28, `/tmp/walk_diag.log`, `/tmp/wide_tile5g.log`). ideas.md §7.11 has the
cap curve; §7.8 has the arithmetic for each gap.

### Two output rows per activation load, on both walks

Every walk in this file multiplies one weight row by T activation vectors, and the schedule
that follows from that reads the activations once per weight row: `QE_Q8_0_MULTI8_FORM` spends
four loads of the token's 32 floats and four FMAs per 32 weights, and the row below it does it
again against the same 32 floats. Holding the dequantised block of *two* rows beside the
sums - `QE_Q8_0_MULTI2R_FORM`, and the same shape in `kern_iq.c` for IQ4_NL and IQ3_S - turns
eight activation loads per token per 32 weights into four, and halves the dequantisation and
the weight decode per FMA, which is the part that was actually issuing.

Two things keep it honest rather than merely faster. It is gated to eight-token groups and to
whole pairs of rows, because sixteen accumulators is the AVX2 register file spent in full and
the two- to four-token forms split a token's sum over two chains, so pairing *those* would
change the number as well as the schedule; an odd last row goes to the row loop and a narrow
group never reaches the pair form. That gate is what makes the second thing possible: the
third section of `make walk` compares both families against their unpaired walks over widths
2..512 at an odd row count, so a partnerless row is checked too, and it prints **0 of
2,100,736 outputs moved** for Q8_0, IQ4_NL and IQ3_S. The engine agrees at the level of the
head: f17 and f19 print byte-identical first-step top-5 logits (`/tmp/f17_wide.log`,
`/tmp/f19_wide.log`).

What the pairs are worth, in the harness that owns them and then in the engine:

| | gain | where |
|---|---|---|
| Q8_0 dense walk, T=8/16/32 | 1.37 / 1.39 / 1.36x | `/tmp/walk_pairs3.log` |
| Q8_0 dense walk, T=512 | 1.19x (536 → 637 GMAC/s) | same |
| IQ3_S expert walk, T=8/16/32 | 1.11 / 1.13 / 1.14x | `/tmp/walk_diag.log` |
| IQ4_NL expert walk, T=8/11/16/32 | **0.92 / 0.95 / 0.97 / 0.98x**, and now gated off | same |
| `block_proj` | 7.36 → 5.77 ms (1.275x) | `/tmp/f19_wide.log` |
| `hc_attn` / `hc_ffn` | 1.62 → 1.37, 1.61 → 1.40 | same |
| `moe_ffn` | 16.48 → 15.28 ms | `/tmp/f18_wide.log` |
| width-512 sweep | 32.1 → 29.7 ms/token | `/tmp/f19_wide.log` |
| the same sweep with the integer walks on | 29.7 → 27.5 → 23.8 → 23.2 → 21.9 → **17.3** ms/token | `/tmp/f20_wide.log`, `/tmp/f22_wide.log`, `/tmp/f25_wide.log`, `/tmp/f26_wide.log`, `/tmp/f29_wide.log` |
| `moe_ffn`, float → int8 panel → int8 panel with one scale per row | 15.13 → 12.49 → 11.28 → **11.07** ms | `/tmp/f22_wide.log`, `/tmp/f25_wide.log` |
| `block_proj` with the integer walk | 5.98 → 4.87 → **4.83** ms (0.830x of the walk it replaces) | `/tmp/f22_wide.log`, `/tmp/f25_wide.log` |
| a chunk's expert stage: seven queued dispatches → one stage graph | 22.85 → **22.26** ms/token (1.026x), trunk the same bytes | `/tmp/f25_wide.log` |
| ten-row walks, as a bucket cap and as a dense-walk group | 8-25 % worse, and a wash | `/tmp/moe_t10.log`, `/tmp/wide_walk.log` |

The dense row is the one to read twice: a walk that reads half the activation vectors is worth
1.28x in the engine, more than the 1.19x the same shape measures alone, because at 512 tokens
the panel is what was being re-walked from L2 once per output row. `QE_IQ_2ROW_DEFAULT` carries
both families and stays on; the integer tile still owns the shared scale exit and still has to
beat this.

The two codebook families no longer share that verdict, and the reason is the register file again.
`QE_IQ4_NL_MULTI2R_FORM` spends the same sixteen accumulators IQ3_S does, and an IQ4_NL block is
only a shuffle and a lookup, so what a second row saves there does not cover the spill; an IQ3_S
block is eight 256-entry grid lookups and a sign spread per 32 weights, and it comes out 1.11-1.14x
ahead after the same spill. Five loads of `make walk` in one process read IQ4_NL at 0.92-0.98x
from eight to thirty-two tokens with the Q8_0 column of the same table still at 1.19-1.40x, and the
HEAD binary reproduces the row, so the gate became per format: `QE_IQ4_NL_PAIR_MIN_T` 0 (never),
`QE_IQ3_S_PAIR_MIN_T` 8, both behind `QE_IQ_2ROW`. Nothing is deleted - the form is still compiled,
is still raced with the gate forced open, and is still 0 of 2,100,736 outputs moved - and
`make moe`'s shared-dequant table carries the mixed bucket (IQ3_S gate/up with IQ4_NL down in one
pass) as its own leg, where forcing the IQ4 pairing reads 1.008x against a 1.1 % control
(ideas.md §7.29, `/tmp/walk_diag.log`, `/tmp/moe_pair.log`).

### Under the threshold, one dispatch per chunk instead of one per token

The chunk walk exists to read a matrix's bytes once for T tokens, so a projection smaller than
`QE_PROJ_MB` (1 MiB) is excluded from it: at that size the bytes are already resident, and the
walk's reassociation is not being paid for. What nobody priced was the other half of that
decision. `proj` sent the excluded tensors to `gemv` **once a token**, and counting the
checkpoint's header finds 180 such projections walked per token: 96 hyper-connection injection
matrices (4 x 10240 f32, 160 KiB), 72 gated-delta alpha and beta (48 x 2560 f32, 480 KiB) and
12 indexer key projections (128 x 2560 bf16, 640 KiB). A width-512 chunk therefore took 92,160
pool barriers where the same chunk's walkable projections take 180, and `make pool` prices a hot
19-thread dispatch at 2.6 us - 0.47 ms of an ingested token, in the two stages the profiler had
been calling slow-for-the-shape for a dozen loads.

Asking once is `proj_small`: the job carries the chunk width and the strides, `gemv_task` slices
the flattened `(token, row)` space instead of the rows alone, and every kernel call is the same
call it always made on the same row. `QE_SMALL_PROJ_BATCH=0` is the loop back.

| | per token | `block_proj` | `hc_attn` | `hc_ffn` | one step |
|---|---|---|---|---|---|
| one ask per token | 16.44 ms | 1.81 | 0.86 | 0.83 | 88.4 ms |
| one ask for the chunk | **15.86 ms** | 1.60 | 0.74 | 0.70 | 88.1 ms |
| | **1.037x** | 1.13x | 1.16x | 1.19x | 1.00x |

`/tmp/f39_wide.log`, arms interleaved best-of-three inside one load; the confirming load reads
16.28 against 15.89 ms/token (1.025x) and a 87.9 ms step against a 88.4 ms one
(`/tmp/f40_wide.log`). The saving is 0.46 ms
attributable in the stages named against 0.47 predicted, the step is 1.00x because a token is one
dispatch either way, and the outputs are not a measurement away from the old ones: a row is
handed to one kernel call, so the token grouping is not part of what it sums to, and
`make forward` asserts a chunk's logits byte for byte at both settings. Ideas.md §7.27 is the
account, including the two theories it killed.

## 13. The elementwise half of a step

Nothing in §3's 5568 MiB is activation work, because none of it reads a weight:
hyper-connections keep four streams of 2560 floats resident, normalise them twice
per layer, gate the 10240-wide concatenation with a sigmoid, average the streams
into the block input, and add the block output back with four broadcast
sigmoids. Per token that is 96 grouped RMSNorms, 96 gates over 10240 elements and
96 broadcast adds - a million transcendental calls if they are written the obvious
way, and a memory traffic bill that no byte budget counts.

`src/layer.c` holds them as three fused primitives plus `qe_sigmoid`, measured in
`make fast` against a double reference on every run:

```
layer elementwise  : pass
                       sigmoid worst 8.5e-08, rms 1.9e-07, mixer 4.3e-07 against a double reference
                       96 of each per token: norms 0.07 ms + gate 0.21 ms + inject 0.02 ms = 0.30 ms
                       12.2 MiB of activation traffic per token; sigmoid in SIMD is 9.1x the libm loop (1.9 ms)
```

**The verdict, like the PLE row's, is that this is not where the time is: 0.30 ms
of a 73 ms step, 0.4%.** The measured cost is small because the work is small -
tens of KiB per mixer, L1-resident in the bench, which flatters it by at most a
fraction of that 0.30 ms. What the section does buy is the one win that is free:
a degree-5 minimax `exp2` with an exponent-field rebuild, held to 8.5e-08 worst
absolute error against libm over the whole gate range and checked for monotonicity
(a non-monotone gate is legal under every error bound and turns a sigmoid into a
generator of ties), is **9.1x faster than the libm loop it replaces - 1.7 ms of
every token** - and it is four orders of magnitude below the noise a Q8_0 dot
product contributes, so nothing else in the model can tell the difference.

Three things in the design were not assumptions after this:

- **The norm is grouped.** `hc_*_norm.weight` is 10240 wide because the streams are
  concatenated; one flat mean-square over 10240 is a different model, runs at the
  same speed, and is visible only as perplexity. `qe_rms_norm_grouped` therefore
  takes the number of streams, and the test perturbs one stream and asserts the
  other three come back bit-identical.
- **Gamma is not always per-element.** `ssm_norm` is 128 entries for 48 heads and
  `attn_q_norm` 256 for 24, so the gamma stride is an argument (0 = shared) rather
  than a convention, and the shared case is the one the test checks against a
  double reference.
- **Fusing is the point of the mixer.** `qe_stream_gate_mean` reads the gate and
  the streams once, applies sigmoid in registers, and writes one 2560-vector; the
  scratch-buffer version of the same expression moves 2.5x the bytes 96 times per
  token, which is 12.2 MiB against 30 MiB of L3. The check for that is not a
  benchmark: it is that zero injection has to be an ordinary residual add
  (`2*sigmoid(0) == 1` exactly), and the test compares the result with `res + y`
  bit for bit.

### The other exponentials, and why 7.1-7.4x of them is not a dial worth turning

The same degree-5 `exp2` reaches the two places that exponentiate for reasons other
than a gate: a gated-delta block's per-head decay, and one cell's `per` attention
weights inside the fused walk `qe_attn_walk`. `make fast` carries both halves of that
claim on every run, the error against a double reference and the rate against libm:

```
exp2               : pass
                       worst 3.5e-06 relative against a double reference over [-14,14], tail 3.1e-06, exp2(0) exact, monotonic
                       expf 1.17 ns/element against 0.16 for the eight-wide form (7.4x); a token asks for 27,648 decays and 86,400 to 295,344 weights
```

The rate is real and the count is not: 114,048 exponentials a token at the selection a
600-cell context reaches, 322,992 at the 2051-cell ceiling, 0.14 to 0.39 ms of CPU - 7
to 20 microseconds of an 88 ms step once nineteen threads share it. Inside the engine
the two forms measured **1.000x** of ingestion and **1.00x** of a step for **0.67
nats** on the token the engine goes on to say (`/tmp/f41_wide.log`), so the
approximation is a dial, `QE_EXP2_SIMD=1`, and not a default - and it is a dial whose
scalar fallback this host no longer has anyway, since glibc's `expf` at these flags is
1.12 to 1.17 ns and the same polynomial written scalar is slower at 1.49
(`/tmp/exp_probe` for the scalar figure).

The item beside it was free all along and ships unconditionally: sixteen value heads of
one key head each computed `exp` of the same float for the same token, 27,648 calls a
token where 1,728 of them are that number, so the decay is now one row of scratch per
token and layer, filled once by `gdn_core_chunk` and read by `delta_task` - same
`expf`, same input, same bits. `qe_softplus`'s 1,728 `expf`-and-`log1pf` pairs a token
stay scalar for the same reason: 4.14 ns an element against 7 us of CPU, on a step that
takes 88 ms to walk 5.5 GiB of weights.

### A norm with an epsilon is not scale-invariant, and that was worth 4.25 nats

`rms_norm(x) = x / sqrt(mean(x²) + eps)`. The scale-invariance people reason with -
"a constant in front of x cancels" - holds only while `mean(x²)` is far above
`eps`, and this checkpoint sets `eps = 1e-6`
(`qwen4exp.attention.layer_norm_rms_epsilon`), so below `rms(x) ≈ 1e-3` the op is
not a normalisation at all: it is `x / sqrt(eps)`, i.e. a multiplication by 1000,
and everything in front of it comes through.

That is not pedantry, it is where the last model-breaking bug lived. The delta
read-out of a 128-wide state ends with `sum * (1/sqrtf(S_v))` upstream, and the
audit note that removed it argued the per-head norm right after it cancels the
constant. The read-out is a contraction against an **l2-normalised** q, so its
mean square is at `eps`'s order, nothing cancelled, and the engine was a model
with one factor missing: 3.5607 nats/token with `1/sqrt(head_state)`, 7.8071 with
`1.0f`, and only the first of those prints English (`ref/AUDIT.md`,
`QE_VARIANT_GDN_NOSCALE`). The same mechanism is the standing question about the
PLE, whose gates are 0.01-ish: through a scale-invariant norm a gate cancels and
the block carries only the sign of the score; through an eps-dominated one it
survives. Whether that block does anything is a question about a magnitude, not
about an operator.

Two rules follow. A scale in front of a norm is a multiplication and has to be
coded, not reasoned about. And the form of the norm is not cosmetic either -
`x/max(||x||, eps)` and `x/sqrt(||x||²+eps)` differ by exactly the same ratio in
the same regime, which is why `qe_l2_norm_grouped` and
`qe_l2_norm_grouped_under` are two functions and the reference's form is the one
that ships.

## 14. What a step is made of, against its own byte budget

The stage profile (`make score`, `--profile`, four steps of the reference corpus at
the engine's default pool width) next to the bytes each stage is defined to read.
The two columns are independent instruments: the left comes from the checkpoint's
tensor table and the per-type rates `make fast` measures, the right is the engine's
own `QE_ST_*` accounting, so a stage that is fast for the wrong reason shows up.

| stage | bytes per token | budget | measured | gap |
|---|---|---|---|---|
| `block`: 36 recurrent + 12 attention blocks | 2786 MiB Q8_0 | 34.6-35.9 ms | 47.6 ms | +12 |
| `moe_ffn`: 10 of 512 experts + shared expert + router | 644 IQ3_S + 422 IQ4_NL + 239 Q8_0 + 240 F32 | 24.0-27.6 ms | 32.3 ms | +5-8 |
| `hc_attn` + `hc_ffn` | 656 MiB Q8_0 | 8.1-8.5 ms | 12.5 ms | +4 |
| `lm_head` | 497 MiB Q6_K | 8.8-11.1 ms | 11.3 ms | +0.2-2.5 |
| `ple`, `head_mixer`, `embed` | rows of a 26.8 GiB table | - | 0.8 ms | - |
| **sum** | **5568 MiB** | **79.5 ms** | **104.4 ms** | **+25** |

**A byte budget is not a lower bound on a stage, and `block` is where the
difference is.** A recurrent block also moves state: 48 heads of 128x128 floats is
3 MiB per layer, read and written, and 36 such layers is 216 MiB a token that no
weight-byte count includes. At a mixed read/write rate that is a few ms of the +12;
the convolution, the two grouped norms per block, and the attention over the
context are the rest. This is also the stage whose arithmetic someone will want to
hand-vectorise next, and the attempt is already on record: writing the delta rule's
two passes over a state row in intrinsics changed the block stage by 0.3 ms out of
48, because gcc already emits AVX2 for them (runtime alias versioning, `Sr` may
alias `kh`) and the 512-byte row is L1-hot between the passes. The register-level
money is in the kernels that decode quantised weights, not between them.

**Hyper-connections are the worst-shaped matrices in the model, and both shapes are now
priced.** `hc_*_up` is 320 columns by 10240 rows, so a row of Q8_0 is 340 bytes;
`hc_*_down` is its mirror, 10240 columns by 320 rows, rows of 10880 bytes. Ninety-seven
pairs of them, 644 MiB of Q8_0 a token, and the engine charges the two mixer stages
12.5 ms: 52 GB/s, where the block stage's projections average 58 and a DRAM-resident
Q8_0 GEMV reaches 77. `make gemv` measures them directly - all 194 tensors, in the order
a step reads them, at their real shapes, with no epilogue in the loop - and gets 65.0-65.2
GB/s for the `up` family, 69.9-70.1 for `down`, **67.4-67.6 GB/s over the whole walk,
10.0 ms**, in two runs.
The mixers are therefore ~2.5 ms above their own bytes at their own shapes, and that
residue has a name: the grouped RMS norm over four streams, the silu, the stream gate
and the F32 inject, all of which the walk leaves out, plus ~290 dispatches at the 1.6 µs
handshake this machine costs.

**Length matters, but only below a kilobyte, and the mixers only just get there.** The
same 512 MiB span, the same kernel, the same threads, the rows re-tiled: 76.6 GB/s at
1360 B rows and 77.3 at 680, and **63.5 and 62.9 at 340** across two runs. The 2720 B
cell of the same sweep reads 77.2 in one run and 68.1 in the next - it is the first cell
measured, so that is the flat part and not that. The byte budget above assumes
~78 GB/s, so the shapes themselves account for 10.0 - 8.3 = 1.7 ms of the stage's +4,
and folding the four mixers into the 2560-wide projections they are read next to moves
the `up` family from 5.2 ms to 4.4 ms, recovering 0.8 of that. The idea is not free and
it is not large, and nothing in the engine can lengthen a row a checkpoint wrote; what
is left of the gap is epilogue, which is code.

**The expert pass is within 6-8% of its own kernels, and four candidate reasons for
the rest have been measured out.** `make moe`'s 39.4-39.6 GB/s at 19 threads is
already experts-only bytes over the forward call (the router's 5 MiB of F32 is timed
separately), so it is directly comparable with the floor below it, and the same
binary now also measures the three expert kernels with nothing around them - no silu,
no per-thread partials, no reduce - over the same pool and the same routes. All three
together, scattered, are 0.539-0.556 ms a layer, 43-44 GB/s, against the 0.589 ms the
same round pays with the pipeline put back, so the pipeline is 6-8% and there is the
gap. Not
the access pattern: scattered against adjacent experts is within 2-3%. Not the row
length: reading the down pool as 2560 columns instead of 640, the same bytes through
the same kernel, is identical at the median. Not the length of a dispatch: once rows
are shared equally the rate is flat from a 13 MiB dispatch to a 215 MiB one (50.3
against 50.4 for IQ3_S, 68.7 against 68.3 for IQ4_NL). Not the composition of the
three passes either - their medians sum to 96% of the whole, so a whole is *not*
more than the sum of its rows, and the 22% penalty this table seemed to show was
best-of-N applied to a cell with two modes, which is why the row prints median and
worst as well as best. What was left is the row split, §12's: dividing the rows
equally rather than by `cpu_capacity` took the stage from 35.5 ms to 32.3 ms (a
repeat pair each way: 35.5 and 35.6 against 32.3 and 32.0). The
block stage's +12 ms is now the largest item on the table, and the state traffic
above is the only part of it that is accounted for.

**The table reproduces, and now with its state bytes printed next to it.** A later run
of the same corpus at 8192 tokens of context (variant 0x01, `make score`) prints
`block 47.8 / moe_ffn 32.1 / lm_head 11.5 / hc_attn+hc_ffn 12.4 / ple+head_mixer 0.7,
sum 104.6 ms` against a teacher-forced wall of 103.8 ms over 255 positions - stage by
stage within 0.3 ms of the table above, which is the reproducibility this file asks
for before it calls a gap a target. The same line now prints the sequence's own
footprint: **270 MiB of state at 8192 tokens, 19.7 KiB per token of context** with the
q8_0 cache, which is the number `qe_forward_state_bytes` backs the "one sequence at
8k" claim with, and 19.7 rather than the 12.2 KiB/token the cache's own size implies
because what it counts per token is the attention cache *and* the indexer's keys.

**That `block` row is one bucket for two stages, and the two do not respond to a chunk
the same way.** It has since been timed twice: `block_proj` is the projections and the
read-out, which a chunk of T walks once and therefore pays an eighth as often at T=8 as
a decode does, and `block_core` is the conv ring, the delta rule and the cache writes,
which runs once per token whatever T is. A decode step prints both and their sum is the
47.8 ms above, so every number in this section still reads; what changes is that a
prefill run can now say how much of its 58 ms per token amortizes over a chunk and how
much does not, which is the question §12 could previously only argue. At the width that
ships now (64) the two rows answer it exactly as the split predicts: `block_proj` falls
43.02 ms of a decode step to **7.81** ms of an ingested token, and `block_core` falls
3.63 to **3.30**, because the delta rule is charged per token no matter how many tokens
shared the walk.

**Counted as traffic rather than as weight bytes, the `block` stage is at the wall,
and that is a measurement now rather than arithmetic.** Weights alone put 2786 MiB
through 47.8 ms, which is 58 GB/s and looks like a 20 % deficit against the 76-79
GB/s of §1.2. The stage's other bytes are what makes that comparison unfair: 216 MiB
of recurrent state read and 216 MiB written back in place. `bin/bench_alu` has exactly
that shape as a row (`--kernels read,state,statent`: an f32 span read, decayed, and
written back, charged for both directions), and at DRAM scale it moves **74.0 GB/s** -
96 % of the same walk read-only, so the writeback costs about one line in twenty-five.
`2786 + 216 + 216 = 3218 MiB` over 47.8 ms is 70.6 GB/s, which is **95 % of what this
host does for that access pattern**, and there is no third bus pass to find: the delta
rule reads a row of state and writes the same row inside one task (`delta_task`,
`src/forward.c`), so the line is in the cache before the store happens. The `statent`
row is the reason not to chase the remaining 5 % either - a non-temporal store, which
would skip the fetch even if there were one, measures *worse* at both scales (72.2
against 74.0 at 512 MiB, 75.2 against 77.8 at 128 MiB). What is left, once again, is
bytes per token (§4, §9).
