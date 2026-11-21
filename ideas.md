# ideas.md: where the remaining performance is on this host

A ranked inventory of refactorings, architectural changes, and code-level work
that could make this engine faster on this machine and this checkpoint. It is a
proposal document, not a measurement record: every number in it is either quoted
from `docs/DESIGN.md` (cited as `§n`) and `docs/ARCH.md` (`ARCH §n`), or it is
explicitly labelled arithmetic. A dotted number (`4.1`, `5.2`) is an item in this
file. Items that have been implemented carry a **Done** paragraph with the measurement
that closed them; the rest are proposals.

Two rules from `AGENTS.md` bind everything below. No idea may be justified by a
storage or load-time number, because a one-time pass over the checkpoint is not a
quantity this repository reports (section 12). And no idea is accepted on a speed
number alone: the gate for a change to the arithmetic path is bit-identity where the
change claims it, and `make score` nll plus `make cmp` against the reference where
it does not. §13's four-nat bug is what that gate exists for.

**Updated gate (owner, 2026-11-16).** Bit-identity is no longer required of a change
whose whole point is to move work: something that comes out *nearly* the same is
acceptable. What replaces it is the tolerance the chunked path already ships under, and
it is not loose - the same next token id, the eight greedy steps after a long prompt
identical id for id, logits inside a third of the head's own range against the ordered
walk, and `make score`'s nll moving no more than the width changes already move it. The
leg that says the ordered walk is the step loop byte for byte stays green regardless,
because that one is a check that two paths agree about the same arithmetic and not a
licence claim on a kernel. What the relaxation buys is reach: §7.2b's parked walk, the
row sub-tiling of §7.3c, and any reassociating accumulation order are now judged on
machine cost alone instead of on an identity they were never going to keep.

## 0. How to read this

There are exactly four levers, and they are named by what they move:

| lever | what it means here | instrument |
|---|---|---|
| bytes/emitted token | the step is a DRAM walk: 5568 MiB at a 76-79 GB/s wall (§3, §1.2) | `make gemv`, `make moe`, `make route` |
| bytes/ingested token | a chunk of T tokens should not pay for a weight T times (§12) | `make prefill`, `make forward` |
| dispatches and barriers | ~800 pool dispatches a token at a 1.6 us handshake (§12) | `make pool`, `make score` |
| serial time | work the master thread does while nineteen others wait | `make score --profile` (§14) |

**A byte ratio is not a time ratio.** §12's central warning, and it applies to
every idea in section 4 below: the per-type kernel ceilings differ, and the tree has
measured three of them - Q8_0 at 78.1 GB/s (the stream rate `make fast` budgets
with), IQ4_NL at 70.0, IQ3_S at 51.1 (§12's ceiling column). Changing a
tensor's format therefore moves time by `bytes_ratio x (78.1 / type_rate)`, not by
`bytes_ratio`. `make fast` prints the table this arithmetic needs, per type and per
thread width, and refuses to print a rate it thinks is cache-resident.

Priority labels: **P0** is cheap and already implied by a measurement someone
else made; **P1** is a real project with a measured payoff; **P2** is worth one
afternoon and a number; **P3** is structural, expensive, and mostly a throughput
story rather than a latency one.

Each idea is written as: mechanism, why it is a lever *on this machine* (Arrow
Lake-S, 20 cores, no AVX-512, 30 MiB L3, 2 MiB THP, single NUMA node, and a step
that streams 5.4 GiB through it), predicted size, the gate, and the risk.

## 1. What a token is made of right now

Quoted from §14 and §3; this is the table every idea below is argued against.

| stage | bytes/token | budget | measured | gap |
|---|---|---|---|---|
| `block`: 36 recurrent + 12 attention blocks | 2786 MiB Q8_0 | 34.6-35.9 ms | 47.8 ms | +12 |
| `moe_ffn`: 10 of 512 + shared + router | 644 IQ3_S + 422 IQ4_NL + 239 Q8_0 + 240 F32 | 24.0-27.6 ms | 32.1 ms | +5-8 |
| `hc_attn` + `hc_ffn` | 656 MiB Q8_0 | 8.1-8.5 ms | 12.4 ms | +4 |
| `lm_head` | 497 MiB Q6_K | 8.8-11.1 ms | 11.5 ms | +0.2-2.5 |
| `ple`, `head_mixer`, `embed` | rows of a 26.8 GiB table | - | 0.7 ms | - |
| **sum** | **5568 MiB** | **79.5 ms** | **104.4 ms** | **+25** |

What §14 then settles, and what therefore is *not* available any more: the kernel
rate (the shipped Q8_0 GEMV is 96% of an identical-bytes read, §1.2), the
block stage's state traffic (3218 MiB of weights plus read-and-written state over
47.8 ms is 95% of what this host does for that access pattern), the elementwise
half (0.30 ms of a 73 ms step, §13), and the PLE row gather (0.007% of a step,
§1.1). What is left is bytes, dispatches, serial work, and the two stages whose
gap is epilogue and shape rather than the bus.

## 2. The shortlist

| # | idea | lever | predicted size | priority |
|---|---|---|---|---|
| 1 | CPU governor to `performance` with a floor (`§6`) | clock ramp | the spread in every rate quoted here | P0, needs root |
| 2 | Top-k the sampler on logits, not probabilities | serial time | ~0.5 ms/token (arithmetic from §13's exp rate) | P0 |
| 3 | Sort the attention gather ascending, and put the KV-group's 12 heads in one walk | cache misses | **done**: one walk per KV group with the softmax inside it (`qe_attn_walk`), ascending selection - 1.39x on the walk at the 2051-cell ceiling the indexer produces, nll unmoved (§5) | P1 |
| 4 | Load-time concatenation of matrices that share an input vector | dispatches + row shape | 1.3 ms handshake floor, plus §14's measured 0.8 ms for the mixers | P1 |
| 5 | Chunked GDN: one state pass per chunk, not per token | bytes/ingested token | **done, and already in the tree**: `delta_task` holds `DELTA_ROWS` state rows while a chunk's tokens go by, and the proof it worked is that `block_core` is flat (3.42 → 3.35 ms/token from width 16 to 128, §12) - there is no state traffic left in an ingested token to remove | P1 |
| 6 | Grouped experts inside the prefill chunk (wire what exists) | bytes/ingested token | **done**: wired (`moe_block_chunk`, shared-dequant kernels for both expert types) - **1.36x** on ingestion at the shipping width, 1.25-1.26x at 32 and 128, and it is the stage whose saving grows with the chunk width (§12) | P1 |
| 7 | Requantise the dense Q8_0 projections | bytes/emitted token | **measured: a decode item, not an ingestion one.** Same walk, three quantisations, per row of one run (`make walk`): IQ4_NL is **1.74x** at T=1 and 1.05-1.10x at T=16, then 0.98-1.00x at 32 and 0.96-1.02x at 64, because at the shipping width the dense walk holds 473 GMAC/s while reading 7.9 GB/s of Q8_0 - a tenth of the read ceiling, so it is ALU-bound and fewer bytes buy nothing. It stays a decode item, where T=1 is the row that counts (§4.1) - and there it is now built and shipping: the same walk over a load-time IQ4_NL copy is **1.16x** of a step for +0.030 nats, with `QE_I4_DENSE` as the quality dial and `make score`'s decode arms pricing every setting in the load that staged the copy (§7.23, `/tmp/i4h_tier.log`) | done for decode, P2 for ingestion |
| 8 | Obtain an MTP drafter and write the Q4_K kernel it needs | bytes/emitted token | **done, and it is the item whose prediction died.** The head is resident (`--mtp`: 2.66 GiB, 34 tensors at `blk.48.*`, Q4_K read without a new kernel) and drafts 32 ids without moving one that greedy says, but the byte model's premise - that verifying a draft row is free because the window shares the model's bytes - is false: a step is 111.2 ms, a draft row 14.5, a nine-row window that computes every row's logits 460.3, which is 4.14 steps, while acceptance is 8 of 16 positions teacher-forced and 1.6-1.9 ids a pass. `cost = (window(k+1) + k x 14.5) / ids(k)` has no `k` that wins except 1, at 1.03x, and the engine default of 8 measured 0.33x. §7.19 is the account, §4.2 is the withdrawn prediction | done, `/tmp/f38_mtp_short.log` |
| 9 | Parallelise the indexer's block scoring | serial time, long context | the term that makes long-context decode superlinear | P1 |
| 10 | Per-op LLC-miss profiling in the engine | instrumentation | makes the +25 ms gap addressable | P1 |
| 11 | Continuous batching over sequences | throughput | weights are read once per step whatever the batch | P3 |
| 12 | LTO, PGO, `__restrict` on the numeric prototypes | codegen | small, cheap, unknown | P2 |
| 13 | Re-tile the multi-row Q8_0 walk so a wide chunk stops paying for its own activation panel | ingestion | **done**: 195 → 518 GMAC/s at T=512, and the width-512 row of the sweep went from a 22 % penalty to the fastest setting (42.9 → 31.1 ms/token, `/tmp/f9_wide.log`); the default width moved 64 → 512 with it (§7.3b) | done |
| 14 | One quantiser walk per expert bucket rather than one per eight rows | ingestion | **built, then switched off**: it wins the kernel probe (1.32x for a walk of twelve) and the uniform-routing rows of `make moe` (125.7 → 109.6 ms at 512 rows, 200 → 230 GMAC/s) and loses everything else - 28 % slower on one wide bucket (43.4 → 55.5 ms at 256 rows) and 16 % slower on the engine's expert stage at the shipping width - so the forms ship behind `QE_PARK=0` (§7.2b). What survives the loss is §7.2's byte-bill account of that stage being wrong | off |
| 15 | One dispatch for a whole stage instead of one per matrix: the pool running a graph of items with dependency edges, so a short stage lends its threads to a long one's tail | dispatches + barrier tails | **built, then switched off**: a chunk's expert stage (the routed walk, the shared expert's two projections, their silu, its projection) is 1.005x of the same seven steps dispatched one after another, at 512 tokens at width 512, on a machine that turned out to be additive about both thread splits and schedules (§7.8). The primitive stays in the tree because the handoff gate that came with it is what found a worker parking on a dead stack frame, and that gate is `make pool` (§6.4) | off |
| 16 | Integer dot products at chunk width: quantise the chunk's activations to Q8_0 once and walk the experts and the dense rows with `vpdpbusd` | measured in four rewrites of the expert walk, in one process with it: **1.36x** for the integer dot with its scale exit where the format puts it (every 32 weights), 2.75x for the same dot pretending the exit is per 256, 1.67x with one activation scale per row, and 0.84x for the codebook taken out at the shipping width. Composite that is ~1.3x of `moe_ffn` and 36 tokens a second: the loads per multiply only go below one when one int8 load feeds several output rows, which is a tiled gemm and not a rewritten dot (§7.10 has the arithmetic and the four legs). The expert half of that composite is now built: 0.824x of the stage, 1.100x of a step, and it ships **on**: the `--ingest` quality leg prices it at four times the next-token loss noise of the grouped pass it sits next to, and the 1.2-1.6x ceiling §7.13 measured for that stage is what decided the price was payable (§7.11). So do the one activation scale per row and the same walk in front of the dense projections: together they are 0.746x of the expert stage, 0.815x of `block_proj` and 42.0 tok/s in one load, and the expert stage's schedule on top of them is 1.026x of seven queued dispatches for **43.1 tok/s**, which is the shipping engine (§6.4, §7.14, `/tmp/f22_wide.log`, `/tmp/f25_wide.log`). The tiled form of the same load - one int8 activation vector feeding **two output rows** - is now built for the expert formats as well and is worth **0.95-1.04x** of the walks it replaces and 0.98-1.01x of the grouped stage with zero outputs moved: measured, `QE_I8_R`, off (§7.12, `/tmp/walk_tile6.log`, `/tmp/moe_r1.log`). Neither is a register problem this engine can talk its way round, because widening a walk's *token* group to ten is built for both codebooks and buys nothing either: as a bucket cap it is 8-25 % worse, a bucket of sixteen becoming three passes of an expert's weights where eight makes two, and on a 512-row dense walk, where the decode it saves is real, it is a wash against the eight-row form (`/tmp/moe_t10.log`, `/tmp/wide_walk.log`) | P0, `/tmp/iq3probe2.log`, `/tmp/f16_wide.log`, `/tmp/ingest2.log`, `/tmp/walk_tile6.log`, `/tmp/f22_wide.log` |
| 17 | One load of a token's activations feeding more than one output row, on the f32 arithmetic the engine already ships | **built, and on**: `QE_IQ4_NL_MULTI2R_FORM`, `QE_IQ3_S_MULTI2R_FORM` and `QE_Q8_0_MULTI2R_FORM` walk two output rows of one matrix against one set of activation vectors, taking the *eights* of a token group and leaving its leftover tokens to the forms they use today - which is why flipping the switch moves **zero outputs**: 0 of 2,100,736 at every width from 2 to 512 for all three formats, counted and not argued by the third section of `make walk` (`/tmp/walk_pairs3.log`). Worth IQ4_NL 1.06x at eight tokens and 1.13x at sixteen and thirty-two, IQ3_S 1.12x and 1.15x, Q8_0 **1.19x at the 512 tokens a chunk ingests at** (1.37-1.39x at eight and sixteen), +2.6-4 % of the 512-row grouped expert pass (200 → 207 GMAC/s), and `moe_ffn` 16.48 → 15.28 ms of a step that went 31.42 → 30.28 ms (`/tmp/f17_wide.log` against `/tmp/f18_wide.log`, 0 FAIL, same next tokens). The dense walk took the same form one width later: `block_proj` 7.36 → 5.77 ms, the two mixers 1.62/1.61 → 1.37/1.40, a 28.06 ms step and a width-512 sweep of **29.7 ms/token** - 31.2 → **33.7 tok/s**, 1.52x behind llama.cpp's 51.33 where this row started at 1.60x - with the first step's top five logits byte for byte the load before it (`/tmp/f19_wide.log`, 0 FAIL). The parked cap that used to beat the default is behind the pairing now, so `QE_IQ_PARK` stays 0, and what is left of the dense walk is not its row tile either (§7.8, §7.11). The same pair on the *integer* side is 1.00x, and §7.12 is the reason the two answers differ: a float row pays four FMAs and four 32 B loads per token, so a shared dequantised block halves the expensive part, while an integer dot pays seven operands and a pair shares three of them | done, `/tmp/walk_pairs3.log`, `/tmp/moe_2row.log`, `/tmp/f18_wide.log`, `/tmp/f19_wide.log` |
| 18 | Share the expert stage by what a bucket costs instead of by how many units it is | idle threads | **built, and on**: 1.001-1.015x on an even router, 1.186-2.500x at 128 rows and 1.565-3.195x at 512 once the router is uneven (`make moe`'s last table, §7.16). What it is worth against the real router is what a checkpoint load knows and `make forward` will say. The same slice weighted by core class as well as by cost (`QE_MOE_BAL=2`) is built and priced as a loser at every skew, pinned or not (§7.16, §11) |
| 19 | A tiled int8 gemm: one dequant-to-int8 tile feeding a 16x16 `vpdpbusd` microkernel, instead of a fifth rewrite of the dot | GMAC/s | unpiced. The survey in §7.15 reports 4.1x on AVX-VNNI and 1.7x on AVX2 for this shape at pp2048 elsewhere; this tree is 1.2-1.6x from its nearby ceiling (§7.13) and several times that from the machine, and §7.12's two-row tile is the same idea at a width where it cannot pay |
| 20 | Two survey items with no code written for them: a *permuted* copy of the ~17 GiB of non-expert matrices (byte-identical values, interleaved for the dot), and a thread count of its own for a chunked prefill | GMAC/s, dispatch | unpiced. Permutation alone is +1.39-1.42x in isolation elsewhere and fits for the dense half, where a second copy of the experts does not fit at all; the prefill thread count is what llama.cpp exposes as `-tb` and this engine has no equivalent (§7.15). The permutation half of it is now measured in this tree rather than in the survey, and larger than the survey said: de-interleaving the Q8_0 rows into flat ones with one scale per row, walked three rows at a time, is 1.84x of the walk that ships (§7.18) |
| 21 | Walk `lm_head`'s 497 MiB of Q6_K at the rate its own bytes cost, by spending fewer ALU ops per byte: the six vectors, two sub-scales and two source arrays a 6-bit weight is assembled from today (`qe_q6_run`) are the obvious candidates, and an int8 activation on the panel a chunk already builds is the larger one | serial time | unpiced, and the ceiling is: 44-46 GB/s measured for this format at this host's rates against Q8_0's 74-83 and a 76-79 ceiling, so the stage's 11.3 ms is worth 6.5 at its own bytes - 1.04-1.07x of a step, and the only part of a decode step that is measured slow at a rate no other format in the file shares. Anything tried has to stay bit-exact against `qe_dot_q6_k_ref` (`make fast` races it) unless it costs accuracy, in which case §4.3's panel numbers are the precedent to price it against. Neither turned out to be necessary: the ALU cost was avoided rather than reduced, by walking a load-time **Q8_0 copy** of the same head instead - 30 % more bytes, 1.02x of a step, and the draft head's `output.weight` is those same bytes - because this host walks a Q6_K row at 44-46 GB/s and a Q8_0 row at 73.6-78.7 (§7.23, `/tmp/i4g_ship.log`). `qe_q6_run` keeps its 44-46 GB/s and is still what an ingested token pays, where the leg does not apply | done, `/tmp/i4g_ship.log` |

Sections 3 onward take these one at a time, plus everything that did not make the
table. Section 11 lists what has already been measured dead, so that nobody spends an
afternoon on it twice, and section 10 is the instrumentation that would let the rest of
these rows be argued with a counter instead of a ceiling.

## 3. Configuration: no code at all

**3.1 P0 Stop running a bandwidth-bound loop on `powersave`.** The host sits at
`min_perf_pct=15` (§6). On a decode step this costs frequency-ramp delay per token
and shows up as the spread in `make gemv`'s six runs (§3). §6 has the three
commands. Everything else in this file is measured against a machine that is
deliberately holding itself down, so the honest order of work is: do this first,
re-run `make alu`, `make gemv`, `make score`, and only then decide what is worth
building.

**3.2 Keep the pool one core short of the mask, unpinned, with equal shares.**
Already the default (`qe_cpus() - 1`, no cpu list, `qe_slice` equal). Recorded here
because the three knobs are tempting and all three are measured: 20 threads costs
15% of a token for one extra thread (§12), `cpu_capacity`-weighted shares cost 4%
(§12), and pinning costs 2-3x the dispatch handshake (§12). `QE_PIN=1` and
`QE_SLICE=capacity` are instruments, not serving modes.

**3.3 P2 Reserve a hugetlb pool so placement stops being a negotiation.** The host
runs `nr_hugepages=0` with `THP=madvise` (§1), so `qe_alloc_huge`'s
`MAP_HUGETLB` attempt always fails and every hot array - weights, the 113 MiB GDN
state, the KV arenas - lands on `madvise`-granted THP, which the kernel can decline
per-range and khugepaged can churn. A reserved pool turns "we asked for 2 MiB
pages" into "we got them", and `qe_alloc_backed` already reports which happened.
This removes jitter and a class of "why is this run 8% slower", not bytes. Gate:
`make score` repeated, plus the `AnonHugePages` figure `bench_stage` already reads.

**3.4 P2 Say something about the two core classes.** Eight P cores stream 85.5 GB/s
and the twelve E cores match eight P cores on read bandwidth for twelve threads
against eight (§1.2), so the E cores are not dead weight - but they are the reason
the flat read at the engine's width (76.2) is below the P-only ceiling (85.5).
Worth measuring once, with the engine rather than the bench: the step at
`--threads 8` confined to P cores versus the default width. Predicted slower
(fewer queue slots, and §1.2 says the surplus is unreachable), but it is the
experiment that would falsify "the extra cores buy queue slots, not clock cycles".

## 4. Bytes per emitted token

The ceiling is bytes over bandwidth and nothing else touches it (§3, consequence
5). This is where a factor of two could still live.

**4.1 P2 Requantise the dense projections: measured, and it is a decode item.** 2786
MiB of Q8_0 projections are read in full by every token regardless of routing (§3), and
the kernels for the smaller formats already exist and are correct against a double
reference: Q6_K, IQ4_NL, IQ4_XS, IQ3_S (`src/kern_iq.c`). The byte ratios come from the
recovered type table (Q8_0 8.5 bpw, Q6_K 6.5625, IQ4_NL 4.5, IQ3_S 3.4375). What was
missing was the *time* ratio, and `make walk` is the instrument for it: 400000 rows of
K=2560 - 1.0 GiB of Q8_0, 0.55 GiB of IQ4_NL, 0.41 GiB of IQ3_S, so none of it survives
the 30 MiB L3 - walked with T activations against it, Q8_0 through `qe_gemv_multi_t` and
the two smaller types through the shared-dequant `qe_gemv_multi_ptrs_t`, best of three,
with the ratio's two columns taken back to back in one run because the absolute rate of
a 1 GiB walk drifts by a third between runs and a within-run ratio does not.

| T (the chunk width) | Q8_0 vs IQ4_NL | Q8_0 vs IQ3_S |
|---|---|---|
| 1, a decode step | **1.74x** | **1.78x** |
| 16 | 1.05-1.10x | 0.94-0.96x |
| 32 | 0.98-1.00x | 0.89-0.93x |
| 64, ingestion as it ships | 0.96-1.02x | 0.88-0.94x |

The first row is the whole of the prize, and the other three are why it is not an
ingestion item. At T=1 the dense projections are read once per token and the walk runs
at 60-73 GB/s, which is where this host's ceiling is, so the byte ratio arrives almost
whole (IQ4_NL is 1.89x fewer bytes, and it buys 1.74x). Against §14's projection budget
of 34.6-35.9 ms that is a saving of about 15 ms in a 104 ms step, i.e. 1.15-1.17x on
generation, and
nothing at all on ingestion: at width 64 the same walk holds 473 GMAC/s while reading
7.9 GB/s (§7.8), a tenth of the read ceiling, so it is limited by arithmetic and
removing 53 % of the bytes moves the time by nothing. That is the measured answer to the
"0.59x time factor" this section used to predict from the byte ratio alone.

It is a flag now, and §7.23 is the account: the copies are staged at open, the
encoders are held to bit equality with the float path, and the price is measured in the
load that uses them - 1.16x of a decode step for 0.0625 nats and one greedy choice in
four, which is real and has to be read rather than assumed away. What §10's missing
oracle still means is that the price is measured as a *change* in this model's own loss
and not against anything coarser. Two refinements worth pricing at the same time: hold layer 0 and the last
few layers at Q8_0, because §9 found layer 0 the most over-lapped layer, so it is the
one whose expert choices the rest of the model depends on; and hold `hc_*` at Q8_0,
because those matrices already read at 67 GB/s and a heavier dequant is the wrong
direction for them. Gate it with `make score` nll plus `make cmp` against the oracle's
logprobs, the way §5 accepted the q8_0 cache on 0.294 against 0.292 nats.

What it would also need, and what `make fast` now prices, is an encoder. A block scale
is one fp16, so the smallest amplitude IQ4_NL can name at all is 2^-24 x 127 =
7.57e-06, and `bench_kern` hands blocks under that to nothing and counts them (571 of
4000, all of them its 1e-6 scale setting). Above the floor the encoder is not the
limit either: the worst block of the 3429 it scores reads 0.1830 of its own energy,
the best fp16 scale any encoder could choose reads 0.1812 on that same block, and no
block sits more than 0.019 above the scale an exhaustive search finds, so the rest is
the codebook's shape on uniform noise. One thing the grid was genuinely missing: the
codebook is asymmetric, +113 largest positive against -127 largest magnitude, so a
block whose extreme weight is negative could not name the scale it was written at.
Anchoring the grid at both costs nothing to measure (worst rms 0.1830 either way, not
one of the 3429 worse for the wider search) and it is what makes a block built from
codebook values at a representable scale come back bit-identical - which is precisely
the case re-quantising a file is made of.

**4.2 Done, and the prediction is withdrawn: the drafter exists, reads, and loses.**
The head is on this host (`mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, 2.66 GiB, 34 tensors
named `blk.48.*`, one above the trunk's last layer), and the second piece of work this
row asked for turned out not to exist: `qe_gemv_t` already reads the head's Q4_K and
Q6_K tensors, so the recommended drafter could always be read. What was missing is the
premise the 19.0 tok/s prediction rested on, that a verified row is free because a
window shares the model's bytes. It is not free: a draft row costs 14.5 ms against a
111.2 ms step, and verifying nine rows whose logits are all computed costs 460.3 ms,
4.14 steps, because a window shares the checkpoint's bytes and none of its arithmetic
(§7.19, `docs/DESIGN.md` §4). The shared-head second-order win is real and is still
not enough, since a pass does read the 497 MiB `lm_head` once for its k+1 rows and
pays for that in the window that then has to verify them. Of §4's two caveats one
survived intact and one did not: acceptance is greedy-reported, and here it is 8 of 16
positions teacher-forced and 1.6-1.9 ids a pass rather than the reported 2.76, which
is the whole difference between a win and a 0.33x loss at the draft length the engine
used to default to.

**4.3 Done: the router's second half is wired, and "costs nothing measurable" was
too generous.** The bf16 shadow is what `router_task` reads, so the 240 to 120
MiB/token §9 measured is banked. What was missing was the other half of that deal.
`qe_moe_route()` carried the f32 re-read of the ten winners, `make moe` section 1b
checked it against an f32 reference by expert id, and docs/DESIGN.md §9 said in the
present tense that the engine renormalises from the f32 rows - but the engine never
called it. Both of a forward pass's routings, `qe_moe_layer()` for a lone token and
a chunk's routed half after its one router pass, did the router and the top-k
themselves, so every expert output of every token was scaled by a weight 1.3e-04 off
the f32 router's on a shared pick, §9's own number for exactly that. It is a
correctness gap that reads as a performance one, and the drift ran the other way too:
the document described the fix as shipped while the code still had it as a helper.

`qe_moe_refine()` is the re-read and renormalisation, pulled out of `qe_moe_route`
and called by both routings; `qe_moe_refine_rows()` is the chunk form, one dispatch
over the chunk's rows, because ten dots of `K` floats is a few microseconds of one
row and a millisecond and a half of a 512-row chunk. Section 1b now measures the pair
the engine composes - select off the shadow, then refine - at 0.0e+00 against the f32
router with the unrefined 1.3e-04 printed beside it, and a second assertion refines
five rows at a stride across the pool and requires every weight to be bit-identical to
the row form, which is the invariant a chunk owes a token (`make moe`).

What it costs is one more read of part of the router per token: 10 rows of 2560 f32
per layer is **4.7 MiB per token**, which at the shipping width is a dispatch's work
and at width 1 is a serial one. `make forward`'s sweep, one load each: 16.2 → **16.3
ms/token** at the width that ships and 93.4 → **94.3 ms/token** at width 1 (`+0.6 %`
and `+1.0 %`; `/tmp/f42_forward.log` against `/tmp/f44_forward.log`, with the
per-row form measured in between at 16.5, so threading the chunk recovered two thirds
of it). The reads are the floor, not the dispatch: nothing in the arithmetic is
redundant once the f32 rows are the ones being multiplied. `make test` and
`make forward` are 0-FAIL across the change, and the engine's decode figure moves
from 61.7 to 61.3 tok/s - 1.19x ahead of llama.cpp rather than 1.20x, which is the
price of not scaling an expert's whole output by 8 bits of mantissa.

**4.4 P1 Blocked `lm_head` for the case where a step carries T rows.** §12's
closing paragraph: `output.weight`'s 497 MiB is sixteen times the L3, so it is the
one matrix where row blocking (`qe_gemv_multi_t`, `QE_PROJ_MB`) has real reuse to
capture, and it is off by default today. It only pays once a verify window or a
batch exists, so it is scheduled behind 4.2 and 8.1, not ahead of them.

**4.5 P2 Fuse the router and the shared expert into the block's weight walk.** Both
read the same input vector and are dispatched separately today: the router is a
512x2560 GEMV and the shared expert three GEMVs with a scalar sigmoid gate
(`moe_block`). Concatenating the router with the shared expert's gate/up (same K,
same input) removes a dispatch and lengthens rows; the shared expert's 240 MiB and
the router's 120 MiB are read adjacent either way.

**4.6 P3 Hot-expert pinning: measured, and not a win here.** The hottest 128 of 512
experts carry 45.6% of selections and cross-layer persistence measured 0.19 against
a 0.20 baseline (§9). The better pin target is the ~4.4 GiB of always-read weights
(`ARCH §5`), which `mlock` already does.

## 5. Locality: where the cache misses are, and which ones to remove first

A token streams 5.4 GiB through a 30 MiB L3, so almost nothing in the weight path
has a cache miss worth avoiding - it is all compulsory. The misses worth chasing are
in the *gathered* reads, where the same bytes are touched more than once, and in
the walks that are serial. Each item below names the access pattern in the source.

**5.1 P1 Sort the selection before the gather.** `top_blocks` returns the winning
blocks in heap order (`forward.c:1425-1426`), and `attn_task` then walks
`sel[]` in that order over a 256 MiB-per-slot arena (`forward.c:1458-1467`), so
2051 cells of 520 bytes are touched in effectively random address order, twice per
head (K then V), and there is no prefetch anywhere in the loop. Sorting ascending
turns that into a strided ascending walk, which is the one pattern the hardware
prefetchers and the L3 both handle. Cost is a sort of 512 integers, once per layer
per token, on a path that today asks for a scattered 256-byte key row 589k times per
token and a value row as often again, against 25 MiB of unique bytes underneath it
(arithmetic from the shape in `forward.c:1430-1434`: 24 heads x 2051 cells x 12
layers requested, 2 KV heads x 2051 x 520 B unique per layer). The same sort also removes
the "reuse distance is one pass over 2 MiB and we are relying on L3 to catch it"
gamble that the code comment at `forward.c:1430` is explicit about. Gate: LLC-miss counters (`make prefill` already harvests them, correctly, on both
monitoring units, for a synthetic round; 10.1 is the version that counts a real step)
and `make forward` bit-identity, since a sorted gather must produce the same
denominator and the same sum - a reordering of an `exp`-weighted sum is *not*
bit-identical, so this one either keeps the summation order (sort the addresses,
keep the accumulation order) or goes through the nll gate.

**Done, and it went through the nll gate rather than a claim.** `sel_ascending` marks
the winning blocks in a bitmap and walks the bitmap in address order, so the gather is
a forward stride with runs of `ratio` cells instead of a random tour of the arena, and
the query's own partial block is appended last because it is the highest address of the
lot. The set of cells is exactly the set the heap won, so the denominator is over the
same terms; the terms arrive in another order, which is a reassociation and is answered
by `make score` rather than by `make forward`: over 2999 positions with the selection at
its ceiling, **1.8358 nats against the 1.8356 of the walk it replaced**, with 105 of
those positions reading a different top-1 and six of them moving the accuracy column.
`make forward`'s identity legs are still byte-identical (the ordered walk against one
token at a time, and the chunk against the token at every boundary a chunk can end at,
per stage ablation), which they have to be: both of their sides run this selection.

The other half of the gate this item asked for, the LLC-miss counters, is 10.1 and is
still unbuilt, so the win is priced as bytes walked rather than as misses: at the 2051
cells the ceiling produces, one layer's gather is 2 KV groups x 2051 cells x 512 B =
2.0 MiB of unique rows, and the head-at-a-time form asked for that twelve times over.

**5.2 P1 Interchange the head loop into the gather, so 12 heads share one walk.**
Twenty-four query heads share two KV heads twelve ways. The split is by query head
across the pool, so up to twelve threads on twelve different cores each make their
own pass over the same 2051 cells, and the sharing is left to the L3. Inverting the
nest - cell outside, the twelve heads of one KV group inside, four 256-byte dots
against one loaded row - converts twelve passes over the gather into one pass with
twelve dots per row, and the row stays in registers. It also fixes the failure mode
the current split has: a thread that gets heads from both KV groups reads every cell
twice. Predicted: the attention cache stops being read twelve times more often than
necessary, which matters most at long context, where the gather is 2 MiB per layer
against weights that no longer dominate. Risk: `scores[]` is indexed by head, and
the softmax must stay per head; keep the summation order per head to stay
bit-identical, and check with `make forward`.

**Done, as `qe_attn_walk` in `layer.c`.** Cell outside, the twelve query heads of one KV
group inside, and the exponential and the accumulator's rescale in the same statement as
the dot, so a score never reaches memory: the 197 KiB score array per layer that the
head-at-a-time form needed is gone, and a cell's key row and value row are read once and
stay in L1 across the twelve heads that want them. One walk per group is two units of
work against twenty cores, so the selection is chunked the way flash-decoding chunks it -
`units = nchunks x nkvh`, each unit carrying its own exponent base and sum of weights,
folded afterwards by a second dispatch split over the query heads (`qe_attn_combine`).
Chunks are bought with dispatches, so the width is one chunk per worker of each group and
never so many that a core is handed two units of a stage that could have been split
around it: at 20 units over 19 threads the makespan is two units where at 18 it is one,
and that difference was worth more at the short end of a corpus than the gather was.

`bin/bench_kern` prices both shapes at one thread, one layer, the shipped q8_0 cache:
**1025 us a layer at the 2051 cells this indexer's ceiling produces, against 1429**
(1.39x), and 2.04x at 8192 cells, which is a column the `top_k` of 2048 in this
checkpoint never reaches; at 512 cells the two are the same height. Over the twelve QSA
layers of a token that is 12.30 ms of one thread's time against 17.15, which the pool
shares nineteen ways: **0.65 ms of a 103 ms step against 0.90**. That is under one per
cent, which is why `make score` prints 108.9 against 109.4 ms/token across the two walks
and cannot resolve the difference, and why this row is about the bytes and the shapes
rather than about a rate: it is the term that stops growing once the context fits in the
L3, and the one that would have grown without limit once it did not.

The exponent base is raised on a threshold (16 nats) rather than on every new maximum,
which is not the textbook form and is deliberate: the selection now arrives in position
order, a context's attention weights are biased towards its end, so a run of records is
the expected case rather than the pathological one and each record would cost a pass over
the accumulator. Checked against a double reference to 1.6e-06 with the two forms 1.1e-06
apart (`make test`'s `fused attention` row), and kept behind `QE_ATTN_FUSED=0` and
`qe_forward_set_attn_fused` so that one load of 87 GiB measures both forms, which is the
same reason 7.2's setter exists.

**5.3 P2 Put a cell's K, V and scales in one place.** A cell is today 512 bytes of
K in `kvd[0, n_ctx)`, 512 of V in `kvd[n_ctx, 2 n_ctx)`, and 8 bytes of scale in a
third array (`forward.c:776-783`): three distant streams per head per cell. Making
the cell self-contained (520 B, or 576 B padded to a line multiple) halves the
number of streams in the gather and the number of page-table entries it walks. It
is a cache-format change, so it needs a version marker and a migration story, and
§5's q8_0-vs-f32 comparison is the accuracy precedent for accepting it. Do it after
5.1 and 5.2, because those two decide how many streams there are in the first place.

**5.4 P2 Try evict-first on the weight walk; it is not the experiment that failed.**
Every weight byte is read exactly once per token, so it earns no cache residency,
while the state, the gather and the activation buffers do. Streaming 5.4 GiB
through a 30 MiB L3 evicts them. `§1.2` and `include/qe.h` record that *software
prefetch* in the expert pass is a 6-67% regression and that a *non-temporal store*
for the state measures worse; neither is an evict-first load hint
(`_MM_HINT_NTA`/`prefetchnta`, or `prefetchw` for the state's write-allocate). The
honest expectation is small, because there is little reuse per token to protect,
but it is a different instruction and it has never been run here. Gate: `bench_alu
--kernels gemv,read` plus `make score`; if it is not a win at 512 MiB it is dead.

**Run, and dead by its own gate.** `_mm_prefetch(..., _MM_HINT_NTA)` in the Q8_0
row walk, two distances, against the same build with no hint, `bench_alu --reps 3
--kernels gemv,read`: 12 blocks (408 B) ahead scored 75.2, 75.1, 75.5 GB/s and
40 blocks (1360 B) ahead scored 73.6, 72.9, 74.4, against 75.2, 75.2, 75.5 for the
shipped walk. What makes the first of those "no win" rather than "unmeasured" is the
control in the same table: the read walk gets no hint, and it drifted 77.4-77.7 to
78.4-78.8 across those two builds, so about 1.5% of any of this is the machine, and
the gemv/read ratio went 97% to 95-96% when the hint reached further. Neutral close
in, 1-3% regression far out - the shape software prefetch has had every time it has
been tried here (§11). Code reverted.

**5.5 P3 The PLE row cache: two cheap cleanups, no payoff expected.** `n_sets` is
already a power of two and the lookup still does a hardware `divq` against it
(`ple.c:280`); and the 109 MiB row cache is 3.6x the L3 while being probed at
sixteen random sites per token. Either mask instead of divide and shrink the cache
to something L3-shaped, or drop it. §1.1 already priced the whole gather at 5.1 us
cold, 0.007% of a step, so this is code hygiene, and the doc quote is the reason it
stays P3.

**Masked, and the cache size is untouched.** `ple_fetch` takes
`ple_set_of(row) & p->set_mask`; `set_mask` is set next to the loop that rounds
`n_sets` up by doubling, which is what makes the mask exact rather than an
approximation of the divide, and `n_sets` is still what the allocation is sized
from. Shrinking the 109 MiB cache toward the L3 is a hit-rate question, and the hit
rate is only measurable with the table resident, so that half stays unimplemented
rather than being changed on an arithmetic guess.

**5.6 P2 340-byte rows are the shape problem, and padding is the wrong fix.**
The mixer `up` matrices are 320 columns by 10240 rows, so a Q8_0 row is 340 bytes
and a row straddles lines; §14 measures 63.5 GB/s at 340 B against 76.6 at 1360 B,
and the same walk at 680 B is 62.9, so length is what binds, not alignment. That is
an argument for concatenation (6.2), not for padding the checkpoint's rows.

**5.7 Not a lever, recorded so it stops being proposed.** The four-E-core shared
4 MiB L2 cannot be used by a decode GEMV: each weight row is read once by one
thread (§12). And the 2 MiB pages already reduce the PLE table's sixteen scattered
rows to one page walk per token (§1.1), so 1 GiB pages for that table are not worth
a boot-config change.

The ingestion half of the same argument is now measured rather than structural. The two
classes are genuinely different caches - 8 P-cores with 3 MiB of private L2 and 48 KiB of
L1d against 12 E-cores with a 4 MiB L2 shared by each cluster of four and 32 KiB of L1d
(`index2/{size,shared_cpu_list}`), so a thread's share of L2 is 3x different by class - and
the engine's one per-thread cache knob, `QE_MULTI_KIB`, does not care: the grouped expert
pass reads 1.168/1.158/1.147 ms at eight rows and 86.4/87.1/85.9 ms at 512 for 64/256/1024
KiB (`/tmp/pe_kib.log`), i.e. inside the run-to-run spread at every bucket width, which is
the same answer the dense walk gave at 19 threads (§7.11). There is nothing for a per-class
budget to be for: a walk that streams a matrix per thread leaves L2 either way, which is
also why capacity-weighted shares lose to equal ones (`mem.c`: 103.4 against 107.5 ms/token)
and why the two classes stream the same bytes at the same rate (59.3 GB/s on eight P-cores,
60.8 on twelve E-cores).

**5.8 P2 Know where the activation buffers sit.** The chunk in flight is three
T x 10240 float buffers plus a T x 2560 and a T x 10240 (`forward.c:799-806`),
about 1 MiB at T=8, and the per-token elementwise touches 12.2 MiB of activation
traffic (§13). They are allocated through one `scratch()` path, so co-locating them
in the order a layer touches them, and confirming with the PMU plumbing
`bench_prefill.c` already has, is an afternoon. This is the one locality item that
could plausibly matter for the mixers' +4 ms, because those stages are the ones
whose epilogue is activation traffic rather than weight bytes.

## 6. Dispatches, barriers, and the serial islands

800 dispatches a token at 1.6 us is 1.3 ms of handshake (arithmetic on §12's two
measured numbers), which is 1.2% of a step - small. The reason this section is
first-class is that the *barriers*, not the handshake, are what keep the master's
serial work from overlapping: `sweep()` four times per layer, all of `layer.c`,
the GDN conv and its alpha/beta pre-pass, `qe_moe_select`'s 512-logit scan, the
whole indexer prologue, and the gather.

**6.1 P1 Cut the dispatch count by concatenating at load time.** Matrices that
share an input vector and a K can be one tensor in the arena: `attn_q` with
`attn_gate` with the indexer query; the mixer pairs with each other where the
dependency allows; the three expert passes already run as two sweeps with one
barrier. Nothing about the arithmetic changes if the row order is preserved, and
row order preservation is exactly what the bit-identity check in `make forward`
can assert. Same lever as 4.5. Predicted: the handshake floor drops with the count,
and the longer rows read faster (§14's row-length table), so the two effects are
additive and the second is the larger one.

**6.2 P1 Do the mixer fold that is already measured.** §14: folding the four mixers
into the 2560-wide projections they are read next to moves the `up` family from
5.2 ms to 4.4 ms, recovering 0.8 of that stage's +4 ms. It is the same mechanism as
6.1 applied to the worst-shaped matrices in the model, and unlike most of this file
it comes with a number.

**6.3 P2 Software-pipeline the epilogue against the next dispatch.** Today the
pattern is dispatch, join, master does elementwise serially, dispatch. Handing the
master a slice of the *next* GEMV before it runs its epilogue, or issuing that
dispatch and letting the master join late, is how the serial list above stops being
serial. It is easy to get wrong (the pool is a process-wide singleton with a single
job slot, `mem.c:133-146`), so it wants either a two-slot job or 6.4.

**6.4 P2 Persistent-task execution as the alternative to 6.3.** One dispatch per
*layer* (or per token) where workers walk the whole stage graph and synchronise on
per-stage counters instead of one barrier per matrix. Deletes the handshake almost
entirely and makes 6.1/6.3 unnecessary. The risk is specific and known: this pool's
failure mode is a preempted spinner costing a scheduler timeslice inside a barrier
(69 us at 20 threads against 1.6 at 19, §12), and a long-lived worker graph
multiplies the blast radius of one straggler. `make pool` extended with a graph
shape is the gate.

**Done at the granularity the risk sentence asked for: one dispatch, not one layer.**
`qe_pool_run_items` (ideas.md §7.8 is where it is used) takes up to 32 items, each
naming how many slots it divides into and which items it waits for, and runs the whole
set as one dispatch: a worker takes the lowest-numbered ready slot of any item by
compare-exchange, re-reads the completion mask after every one, and leaves when no slot
is left unclaimed. The barriers between independent items therefore disappear and a
short stage lends its threads to the tail of a long one, which is the only prize a
shared machine has on an additive host (§7.8). What it deliberately is not is a
long-lived worker graph: the blast radius of a straggler is one dispatch, the ordinary
`qe_pool_run` path is untouched, and a decode step still never builds a graph.

`make pool` grew the two legs, and the second one is the reason the primitive publishes
its items into storage the pool owns. The first runs a four-item graph (two halves that
may co-run, a stage needing both, a tail) 60 times with spinning slots and asserts every
slot ran exactly once, that nothing started on a *claimed* rather than finished
predecessor, and that two independent items were resident at once: `60 dispatches of 4
items, 2460 slots, widest overlap 2 items at once`. The second runs a graph, then a
function that churns the stack the way the next part of a layer would, then an ordinary
full-width dispatch, 240 rounds, and asks whether every worker turns up in every one of
those dispatches: `240 pairs, worst round 0.02 ms, every worker in every queue dispatch:
yes`. That leg is what a graph hands back to the pool, and it is a real failure and not a
hypothetical one: the first version handed the workers the caller's array, an engine read
it after the dispatch that built it had returned, mistook a dead frame for a stage that
still had work, and parked a worker on it for good - invisible in a bench where the frame
still holds the same numbers, and a hang seven minutes into a prompt in a forward pass,
because a worker that cannot leave a graph never reports a slice of the dispatch after it
either. The hot dispatch that the ordinary path pays is unchanged at **1.4-1.8 us median**
at 19 threads in the same runs, which is also the statement that the handshake 6.1 and 6.3
want to delete is still being paid, in full, by every dispatch of a decode step.

The same leg later caught a second bug of that family, in the ordinary path rather than in a
graph: `./bin/bench_pool auto 120` segfaulted twelve runs out of twelve on a loaded host, with
the backtrace in `qe_worker_main` inside `bump`. A worker waking for a generation read `fn`,
`arg` and `nthreads` *live*, so a worker outside the width of the dispatch it woke for found
the fields of the *next* one, ran it, and added itself to that one's completion count - which
lets its master return with a real participant still inside the caller's task. Those tasks are
stack frames, so what the late worker runs is a job several dispatches dead, and the fault
landed on a `hits` pointer read out of a frame the master had moved on from. It is the same
race as three soft failures this bench printed for weeks and nobody could reproduce
(`slice 17 ran 53 times, expected 52`, `two independent items were never resident at once`,
`19 worker(s) missed a dispatch that followed a graph`), and the trigger is `bench_pool`'s cold
loop, which varies the dispatch width every round: most workers are inside a given width, one
is not, and that is enough. The fields a generation names are now a struct stamped with the
generation that owns them, cleared before they are written and stamped after; a worker copies
them and throws the copy away unless the stamp still matches, and a copy thrown away can only
ever belong to a worker nobody is waiting for, which is exactly the worker that used to run
one too many. Twelve runs at twelve PASS under the load that produced twelve segfaults, and
`make test` is green with it.

**6.5 P0 Sample on logits, not on probabilities.** `qe_sampler_draw` computes
`expf((l - max)/T)` for all 248,320 logits to run a bounded top-k heap over them
(`sampler.c:169-181`). `exp` is monotone, so the top-k *is* the top-k of the logits,
and the exponentials that matter are the k that survive - 20 of them. That deletes
248k serial `expf` calls per token from the master thread, about 0.5 ms if §13's
measured element rate for the SIMD-against-libm comparison carries over
(arithmetic). Bit-identical for the survivors if the same `expf` and the same shift
are used there. Greedy (`temp <= 0`) already returns early and is untouched, which
matters because greedy is how every other check in this repo compares two runs.

**Done: 578 us a draw to 38 us.** The heap ranks logits and exponentiates the k that
survive, so the shift and the `expf` are the same ones the survivors always got; the
replacement test is still against the root's *probability*, which is what makes an
`expf` tie come out the same way rather than merely close. The two full-vocabulary
scans (max, then first index with the max) are `vocab_max`/`vocab_first` in
`sampler.c`, AVX2 with scalar fallbacks. `bin/bench_sample` keeps the old
exponentiate-everything selection in the tree as the oracle it is checked against -
60 trials x 1000 draws of the same seeds against it, a 4096-wide vocabulary whose
top-20 set is known by construction, a collapsing logit pair built with
`nextafterf`, and the cost column: `sampler: 38.3 us per draw over 248320 logits,
against 577.8 us`. That is 0.54 ms of master-thread time per sampled token, 0.6% of
an 88 ms step, and the fourth lever in §0's table rather than the first. Greedy now
goes through the same two scans instead of a scalar pass, and returns the same id.

**6.6 P2 One SIMD `exp2` for every softmax in the engine.** `qe_exp2_avx2` is a
static inline in `layer.c:56`; `attn_task` still calls scalar `expf` 2051 times per
head per layer (`forward.c:1475`). Unlike 6.5 this is parallel work, so it is worth
about 0.1 ms (arithmetic), and it is not bit-identical - so it goes through the nll
gate exactly as §13's 8.5e-08 `exp2` did. Promote the helper to `src/kern.h` and use
it in both the attention softmax and the sampler's survivors.

**6.7 P2 Keep `q` in registers across the gather.** The attention inner steps are
cross-TU calls to `qe_dot_q8` and `qe_axpy_q8` (`layer.c:388` and its neighbour),
about 589k calls per token, so the 256-element dot cannot keep the query vector in
registers between cells. Either a specialised static loop inside `attn_task` or a
header-inline variant; 5.2's restructure probably subsumes it.

**Not done, and 5.2 did not subsume it.** `qe_attn_walk` still calls `qe_dot_q8` and
`qe_axpy_q8` once per cell per head, so the call count is the same 589k a token and the
query vector still crosses a translation-unit boundary to meet the row. What 5.2 removed
was the twelve walks those calls were made from, which turns out to have been the whole
cost: the walk is now 0.65 ms of a step, so this item is worth at most that, and the
place to spend it is the header-inline dot inside the twelve-head loop rather than a new
kernel. Keep it after 5.3, which decides how many bytes each call reads.

**6.8 P1 Parallelise the indexer's scoring loop, before the context gets long.**
The per-block scores are computed on the master: one pass over `cur` blocks and
four heads of 128-dim dot products per QSA layer (`forward.c:1552`), then
`top_blocks` runs a 512-entry heap over the same `cur` (`forward.c:1385`). That is
12 layers x 4 heads x 128 MACs per block, serial, and `cur` is context over four -
so this term grows with the conversation while everything else in the table does
not. It is the mechanism that will make a 100k-token crush session feel
superlinear. Split by block range over the pool, and replace the heap with a
selection over the already-computed score array. Note the pairing with 7.5: the
right answer may be "score in parallel, and below a context threshold do not score
at all".

**Done, and measured against the path it replaced.** The loop is `idx_score_rows`
in `forward.c`, called over a `qe_slice` block range by `idx_score_task` through
`qe_pool_run`, and below `QE_IDX_PAR_MIN` (64 blocks) the master runs the same
function over the whole range instead of paying a dispatch for it. Each block's
score is four independent `qe_dot`s accumulated in the same order, so splitting by
block range cannot move a bit. The heap `top_blocks` already selected over the
score array rather than recomputing anything, so the second half of the idea was in
the tree already. `make pool` gained the `qe_slice` invariant the split silently
depends on (ordered, disjoint, exhaustive for 19 widths over ranges of 0..2048
items).

The A/B is two builds of `bin/bench_forward` on one 833-token prompt, the threshold
left at 64 in one and pinned off (`-DQE_IDX_PAR_MIN=1000000000`, which is why the
macro is `#ifndef` guarded) in the other. Both walked 208 blocks of a QSA layer at
the end of the prompt, both reported `prefill vs one token at a time: trunk
identical, logits max diff 0.0e+00`, and every figure they print agrees: the top
five ids with their logits, max |res| 35.3, max |block output| 7.52, max |logit|
23.8, non-finite 0. The split is exact, which is what was claimed of it. What it is
not is a win at this length: 67 ms/token to ingest either way, which is the honest
reading of 208 blocks scored against 74 GB/s of weights. The claim left standing is
about scaling rather than about 833 tokens, and only a much longer session than this
target feeds would settle it.

Getting there cost a one-line repair in `bench/bench_forward.c`: the second hook
prefill started at the position the first one had advanced to, so it stopped outside
the prompt it was meant to abandon and the assertion under it failed for that reason
alone. `make forward` is not in `make test`, which is how a checkpoint target stays
broken quietly.

**6.9 P2 Bound the MoE partial-sum traffic.** `down_task` zeroes a K-float partial
per thread per layer and `reduce_task` sums them (`moe.c:347-385`), which is about
9 MiB/token of zeroing and 9 MiB of reads (arithmetic from 19 threads x 2560 floats
x 48 layers). Fewer partials, a tree reduce, or having the first writer initialise
instead of a `memset` all remove part of it. Small, and §14 says the expert pass is
within 6-8% of its own kernels, so this is one of those 6-8%.

**Measured the ceiling first, and the ceiling is negative.** The way to price a
`memset` is to take it away and see what the pass costs, so `down_task` was changed
to clear 64 bytes instead of `K` floats - the wrong answer, the same instruction
count one nineteen-threads-over, and the pipeline table from `make moe`:
39.7, 39.7 GB/s with it, **38.8 without it**, and 39.6 on the control run after
putting it back (ms/experts 0.587, 0.601, 0.588, so the instrument is worth about
0.3%). Removing the zeroing costs 2%, which is the answer to the whole idea: those
10 KiB are per-thread and already in the fastest cache there is, so what the
`memset` was buying is the line arriving before the accumulate that overwrites it.
The 9 MiB that is real traffic is `reduce_task` reading *other* threads' partials,
and none of the three suggested shapes touches it without reordering the sum, which
is a bit-identity question this is not worth. Code untouched.

## 7. Ingestion: the half a crush turn actually pays for

Ingestion is what the turn cache was built for, and it is still 65-75 ms per token
(§12, §7). A 12k-token prompt is the difference between a usable agent loop and an
unusable one, so this section is ranked by what the follow-up turn pays.

**7.1 Done: the recurrent state is read once per chunk, not once per token.** A
GDN layer's state is 3 MiB read and 3 MiB written per token, 226 MiB of the block
stage's traffic (§14), and a naive chunk would apply the delta rule to it once per
token anyway. `delta_task` is the chunked form: the unit of scheduling is a head's
`DELTA_ROWS` rows of state, held while the chunk's `T` tokens go by, so the row is
written back once per chunk and the state traffic per ingested token is 226/T
(`forward.c:1488`, and the comment above it records that the row is 512 bytes and hot
in L1 between the prediction pass and the update, which is why hand-written
intrinsics for the two passes measured the same 48.2 ms block stage to the tenth of a
millisecond). The conv ring and the per-token l2 norms are in the same chunked core
(`gdn_core_chunk`), position-ordered inside it, because the order is what the
recurrence is defined by while the traffic is not.

The way to see that there is nothing left here is the stage table `make forward`
prints: `block_core` is 3.06 ms/token at width 16, 3.09 at 32, 3.30 at 64 and 3.34 at
128 (3.63 one token at a time). A stage still carrying 226 MiB/token of state would fall
by roughly a factor of four between 16 and 64; it does not move, because at width 16 the
state is already 14 MiB/token and the residual is the arithmetic the delta rule is.

**7.2 Done: the grouped expert pass inside the chunk, and it is a width item.**
`qe_moe_forward_multi` existed, was correct, measured 1.17x at T=8 and sat ~1e-6 off
bit-identity (§8), and `forward.c` called `qe_moe_layer` per row of every chunk, so
prefill never used it. It now runs a chunk's rows through the grouped pass
(`moe_block_chunk`, `qe_forward_set_moe_grouped`), with the shared-dequant kernels for
IQ3_S and IQ4_NL under it. What the first reading of it missed is that its size is a
function of the width it runs at, because what grouping saves is the repeat reads of
an expert a second row in the same window also picked: **1.53x of whole ingestion at
the width that ships** (46.8 ms/token serial against 30.5 grouped), 1.39x at 128, 1.35x at
64, 1.26x at 32 (`make forward` A/Bs four widths in one load, `/tmp/f9_wide.log`; the first
load of this leg swept 16 too and read 1.16x there). The stage table says where it goes: the
expert stage of an ingested token is 25.33 ms at width 16 and 14.56 at 512, out of a 29.44 ms
profiler sum, so it is half of an ingested token and the one stage where a wider chunk keeps
paying.

What it is *not* is a source of bytes-per-read. `make moe` walks the grouped pass
over a synthetic 512-expert pool and it holds 43-51 GB/s of *distinct* bytes from 2
rows to 64 (50.4 at 4, 51.5 at 16, 49.3 at 32, 43.3 at 64), which is this machine's
rate for scattered 2.3 MiB expert tensors and not a number that has improved across
any dispatch width, row count or thread count tried (§12's dispatch table). A 64-token
window dedupes 640 picks to 366 distinct experts, and `make moe` prints the byte bill
that comes to: 812.2 MiB of distinct expert weights per layer per chunk, which is 12.7
MiB per token per layer, **609 MiB of routed expert bytes per ingested token**. At the
43.3 GB/s measured for that exact row count those bytes are a **14.7 ms** floor against
the 19.18 ms the engine spent in the stage at that width, and the 4.5 ms difference has to pay the
silu, the shared expert, the router and the dispatch. So three quarters of the stage is
bytes read at the machine's scattered-read rate, and the wall behind it is bytes: the
only lever left is fewer of them, which is a checkpoint decision. The one measured
anomaly is width 128, where the efficiency per byte drops to 31.9 GB/s and
`make forward`'s sweep agrees.

That last paragraph was wrong, and the way it was wrong is worth keeping next to the
number that retires it. The 43.3 GB/s was never the machine's rate for scattered expert
bytes: it was the rate the pass reached because it was reading the *same expert's rows*
once per eight rows of tokens, and at a 64-token window the two effects - fewer distinct
bytes to read, more tokens to amortise each read over - moved together, so a byte bill
looked like the wall. Widen the window and they separate. At width 512 a chunk makes
5120 picks over 512 experts, which is all of them: 1136 MiB of distinct expert weights
per layer, **106.6 MiB per ingested token**, and 2.5 ms at the same 43.3 GB/s against
the 14.56 ms the stage costs (`/tmp/f9_wide.log`, the stage table's `moe_ffn` column at
w512). The stage is reading a fifth of the bytes it did at width 64 and a quarter of the
time, at 7.7 GB/s of distinct bytes - one tenth of the read ceiling, and the 2.5 ms above
is what the machine's own scattered rate would have made those bytes cost - so neither the
bytes nor the FMA units were the wall. The wall was the quantiser.

`make moe`'s shared-dequant table showed it once the row counts went past 64: cost per
bucket grew with `ceil(rows/8)`, not with rows. A probe over 64 experts x 640 rows x
K=2560 of IQ3_S fits `cost(np) = D + np*F` with D = 1.74 ms - one walk of the 43 MiB of
grid lookups - and F = 0.19 ms a token, so the ninth token of a bucket costs 8.7x what
the tenth costs: past eight rows the accumulators no longer seat in the register file,
the group splits, and the tail pays for the dequantisation again. A 512-token window
averages ten rows an expert, so the average bucket was paying two walks for ten tokens.

**7.2b Measured twice, and the second measurement is the one that ships: one walk per
bucket is off.** The block and the accumulators want the same sixteen registers; below nine
rows the block is the one that moves to an L1 line (`QE_STORE_QB`), and above it the sums
move instead (`QE_IQ_PARK_MAX`, `src/kern.h`), which costs two loads and a store per token
per 32 weights and buys a whole walk of the quantiser. The probe the forms came out of is
monotone to thirty-two tokens (`/tmp/probe_iqg`, 6.80 ms against 5.14 for a walk of twelve,
18.55 against 14.77 for thirty-two), and `make moe`'s grouped table agrees with it: 512
rows over uniform routing 125.7 ms → 109.6, 256 rows 80.7 → 56.7 (`/tmp/moe_tile1.log`
before, `/tmp/moe_park1.log` after).

The same run of the same bench contains the answer, in the table next to it. The
shared-dequant walk - one bucket, every row on the same experts, which is the shape a park
is for - got *slower at every row count above eight*: 32 rows 4.78 → 6.31, 128 rows 21.29 →
27.90, 256 rows 43.44 → 55.53, 512 rows 89.69 → 115.13. And the engine agreed with that
table rather than with the probe: turning parking on cost the expert stage 16 % at the
shipping width (14.34 → 16.64 ms/token of `moe_ffn`, `/tmp/f6_wide.log` against
`/tmp/f7_wide.log`, where every other stage of those two runs agrees to 2 %), 8 and 10 % at
widths 64 and 128, and nothing at 16 and 32, where buckets stay under nine. Which shape an
engine sees is a routing question: the uniform table's buckets are ten rows, where one walk
of ten replaces a walk of eight and a walk of two, and the wide ones are the ones a park
pays for in activations - thirty-two parked tokens is one K-row each, 320 KiB per thread
against 80 KiB at eight, or 6 MiB of L2 across a nineteen-thread pool that also wants room
for the weights it is walking.

What ships is therefore `QE_IQ_PARK_DEFAULT 0` (`src/kern.h`): the eight-token walk, and a
`QE_MOE_SHARED_GROUP` slice that follows it down rather than staying at the ceiling, so
that the switch is one thing and `QE_PARK=0` is the configuration that predates this
section rather than a third one nobody measured. `/tmp/test_cap2.log` is that
configuration re-run end to end and lands where `/tmp/moe_tile1.log` was (123.1 against
125.7 grouped at 512 rows, 42.9 against 43.4 and 89.9 against 89.7 on one bucket), which is
the revert confirmed rather than asserted. `make forward` now prints the curve on one load
rather than one process per point, and it has printed it twice on the shipping
configuration. `/tmp/f8_wide.log`: 29.46 ms a token with parking off, 29.17 at eight - the
same configuration, so that 1 % is what one load's own noise is - 28.93 at twelve, 30.22 at
sixteen, 32.51 at twenty-four, 31.62 at thirty-two. `/tmp/f9_wide.log`, a second load of the
same binary: 29.31 off, 29.29 at eight, 29.68 at twelve, 30.09 at sixteen, 33.02 at
twenty-four, 31.47 at thirty-two. Twelve is the fastest cell on the first table, 1.3 % under
the mean of its two identical controls, and 1.3 % *over* them on the second, which is the
same distance the first one was - so the reading is that a park buys nothing in the engine
and costs a fifth of the stage by sixteen. The forms stay in the tree because the kernel item
§7.8 keeps wants walks wider than eight and has to answer this measurement before it arrives.

Two numbers first written here were wrong and are worth naming, because they were believed
the moment they were printed: the shared-dequant table did not go "43.4 → 37.9" at 256 rows
(it went 43.44 → 55.53, the opposite sign), and the speedup this section first quoted,
"1.23x at 512 rows, 200 → 247 GMAC/s", is a misreading of the grouped table, whose actual
parked row is 109.562 ms at 230 GMAC/s, 1.15x. What is true is narrower than what was
written: a park wins the arithmetic and loses the machine at the shapes this checkpoint
routes.

**7.3 Done: the chunk width swept to 64, and it is 1.18x on ingestion.** The earlier
sweep stopped at 16 because 16 was the ceiling (`PLE_PENDING` was defined off it, and
the chunk buffers with it) and because the dense side is flat above about width 7 -
which was read off a run where the expert pass still ran serially, and the expert pass
is the one stage whose cost keeps falling with width. Run to the ceiling in one load
(`make forward`, 384 tokens, walk on, 19 threads): 94.9 ms/token at width 1, 45.5 at 8,
41.1 at 16, 38.1 at 32, **34.7 at 64**, 34.8 at 96, 35.3 at 128 - a plateau from 48
to 96 with 64 in the middle of it, and honest noise on top: the 128 row read 40.7 in one
load against the profiler's own sum of 33.66 for the same width, and 35.3 against 34.03 in
the next. So the default
moved 16 → 64: **35 ms per ingested token against 41.1**, same next token at the end
of a 598-token prompt, and the §12 identity legs still green (the ordered walk is the
step loop byte for byte, `trunk identical, logits max diff 0.0e+00`, and all five
ablation legs agree at every boundary a width-64 chunk can end at).

Two things came with it. The hyper-connections stop being free at the far end, 1.53
ms/token at width 16 to 2.13 at 128, which is the 320x10240 mixer down projection
walking away from the machine's rate as the row block fills with activation rows, and
no row block fixes it (`make walk`). And the reassociation the shipped walk performs
against the ordered walk grew with the width, 12 % of the head's logit range at width
16 to 16 % at 64, same next token, and the eight greedy steps of what it says next are
identical id for id (`make forward`'s fifth ingestion leg). Teacher-forced loss is not
the ruler here and §12 records why: `bench_score` scores by stepping one token at a time,
so it cannot see a chunk and reads the same 2.5369 nats at every width.

**7.3b Done: the width swept to 512, which needed the dense walk re-tiled first, and
1.10x on ingestion.** Raising `QE_CHUNK_MAX` to 512 made the sweep go there, and the
first thing going there found: the same load that read 34.2 ms a token at width 64 read
**42.9** at 512 - a 22 % penalty for the widest chunk the engine allows - on shapes
`make walk` had never called slow. `qe_gemv_q8_0_multi` looped weight rows outside and
token groups inside, which is the right order while the activation panel is small and a
disaster when it is not: at T=512 and K=2560 the panel is 5.24 MB, so every weight row
re-read all of it, and the walk fell from 476 GMAC/s at T=64 to 195 at T=512. Row tiles
outer, groups inner, with the tile sized to the L2 budget (`QE_MULTI_KIB`, 256 KiB: 66
rows of K=2560, 10 of K=6144, 740 of K=320) and the panel read once per tile instead of
once per row. Bit-identical, and checked as such - the same 512-token slice prints the
same four logit deltas to the tenth before and after (`/tmp/f5_wide.log`,
`/tmp/f6_wide.log`), and `make walk` at the shipped 1 MiB row block reads 195 → 518
GMAC/s on `attn_qkv` at T=512, 151 → 467 on `ssm_out`, 153 → 405 on `hc_down`, 233 → 596
on `hc_up` (`/tmp/walk_base.log` → `/tmp/walk_tile1.log`), with the T=1 column unchanged
as it must be.

The sweep then falls the way the byte model says: 92.2 ms/token at width 1, 45.6 at 8,
41.4 at 16, 37.2 at 32, 35.1 at 64, 33.1 at 128, 32.2 at 256, 31.8 at 384, **31.1** at 512
(`/tmp/f9_wide.log`), so the default moved 64 → 512 and the fastest row on the table is the
one the engine runs on. Two earlier loads of the same configuration read 30.9 and 31.4 at
512 (`/tmp/f6_wide.log`, `/tmp/f8_wide.log`), which is the 1.6 % a load of this checkpoint
is worth, and the plateau under it is not monotone in detail either - 192 read 34.1 where
256 read 32.2 - because a plateau is a run of a plateau. What the width costs is memory and
nothing else: 4.8 GB of grouped-expert partials across the 48 layers (`n_threads * T * K`
floats per layer) and 210 MiB of chunk scratch, on a box that also holds all 87.2 GiB of
checkpoint resident. What it does not cost is the identity legs, though two of their
bounds had to be told what a 512-token window is: see §12 and the two A/B legs in
`bench/bench_forward.c`, both of which now assert a half of the head's range instead of a
tenth and a third, with the recorded fractions at every width they were measured at next to
them and the same-next-token claim untouched.

**7.3c Done: handing the expert stage half a chunk, because a bucket's walk is priced by
how full it is.** A grouped pass walks an expert's quantiser once every eight rows its
bucket holds (`QE_MOE_CHUNK`), which makes the cost of a pass a function of how full its
buckets are rather than of how many of them it has. At the shipping width an expert
averages ten routed rows - 512 rows over 512 experts at top-10 - and ten is a walk of
eight plus a second walk carrying two. Cutting the rows a pass covers in half therefore
buys a walk instead of one and a quarter: `make moe` costs 124.171 ms for a 512-row pass
against 57.035 for a 256-row one and 35.151 for 128, which is 0.2425 ms per token per
layer against **0.2228** and 0.2746 (`/tmp/moe_t19.log`). The curve has a minimum, and the
engine - which since §7.2 has handed the expert stage the whole chunk in one call - is
running on the far side of it as this bench counts.

The 256-row pass is cheaper per token while reading the same experts (509 distinct at 256
rows against 512.0 at 512, 1130.0 MiB against 1137.4 per layer), so per token it moves
*twice* the expert bytes for 8.9 % less time. That is the plainest statement this section
has yet had that the stage is arithmetic-bound at wide widths and not byte-bound: the byte
bill of §7.2's 2.5 ms floor goes up and the stage gets faster anyway. Below 256 it stops
paying - 128 rows costs 23 % more per token than 256 - and what it is paying is the three
`pool_run`s a pass costs, which a pass is now twice as likely to be asked for.

`QE_MOE_TILE` / `qe_forward_set_moe_tile` is the width of one window, 0 being the whole
chunk; the router, the per-row top-k and the shared expert stay whole-chunk, and a row
lands in exactly one window, so a row's expert contributions are still summed in bucket
order. What a narrower window does change is which rows share a walk of a bucket's rows,
which is where the multi forms of a type disagree in the order of the same products -
the same ~1e-6 class §7.2's switch moved, and `make forward` now sweeps
`{0,128,256,384}` on one load and asserts the in-force width costs under 10 % over the
fastest, the same shape as the parked-walk leg above.

Two reasons the default is 0 and not the 256 that paragraph argues for. The first is the
one §7.2b exists for: the curve above is uniform routing, which makes the buckets as even
as a router could ever make them, and the real router's hottest expert is 0.4 % of
selections over 2045 routed layers - a distribution that leg has already read in a
different direction once. The second is that a 512-row chunk has exactly one place to cut
itself in half, while a serving pass at 2048 rows has eight and each of them is three more
`pool_run`s and a bucket population that shrinks with every window. What the leg prints at
the width this engine actually serves is what decides it, and at 512 tokens at width 512 it
decides against the prediction: **30.71** ms/token at tile 0, 32.33 at 128, **31.46** at 256,
31.35 at 384, with the expert stage at 16.50 against 17.91, 17.14 and 16.94
(`/tmp/f12_wide.log`). The whole-chunk window is the fastest of the four and the 256-row
window the kernel curve argues for is 2.4 % slower.

Nothing about the kernel was wrong. The saving it predicts is just smaller than the stage it
lives in: 0.0197 ms a token per layer is 0.95 ms a token over the 48 layers of this model,
3 % of a step, and the measured tile-256 row is 0.75 ms *worse*, so what a window costs (the
three dispatches it adds and the bucket population that thins with every window) is the other
1.7 ms. Uniform routing is what made the prediction look like a gift: it makes every bucket
the same fullness, so halving the rows halves the walks, while the real router's evenness
ends at the average and the second half of a window's rows is not waiting at the end of a
walk of eight. `QE_MOE_TILE` stays 0. The leg stays too, and it is the thing that will notice
if a serving pass wider than the chunk ever makes a window cheaper than the chunk.

**7.3d Done: one vector pass for a block's eight group scales.** The IQ3_S multi form
derived a block's eight group scales one at a time, and each one cost a byte load, a
shift, a mask, an integer-to-float convert and the three-op scale chain, eight times per
block, in the loop that walks a bucket's rows. Nibble `g` of the word the four scale bytes
make *is* group `g`'s scale, so one variable shift against a lane index produces all eight
at once, and `qe_iq3_block_scales()` (`src/kern_iq.c`) now does that once per block with
the eight floats spilling to the stack the group loop already reads them from. Both forms
are exact the same way (a nibble converts exactly, `1 + 2n` is exact below 32, and the only
rounding is the multiply by the block's `d`), so a group dequantised through the vector path
gets weight bits identical to a group dequantised by the scalar one, which is why this
never needed the relaxed gate.

The measurement is the three-type walk bench (`make walk`'s shape: K=2560, 400000 rows,
19 threads, T swept 1 to 128) run as an in-process alternating A/B, three passes of the
two builds, medians taken per T: the IQ3_S column is **+3.5 %** faster, concentrated where
the walk is wide enough for the scale chain to be on the critical path (T=64 +9.5 %, T=32
+4.5 %, T=16 +3.7 %, T=8 +3.3 %, T=2 +4.1 %, T=1 0.0 %). What makes that a win rather than
a warm clock is the other two columns of the same runs, which this change cannot touch:
Q8_0 reads **1.005** of its own old time and IQ4_NL **0.999** (`/tmp/sp/ab2.log`). The
sibling attempt on the same inner loop, building the codebook indices with vectors instead
of the scalar byte chain, is in §11 at 1.49x slower.

**7.4 P1 Intra-prompt rewind points.** `--turn-cache` stores boundaries at prompt
end and answer end, so a client that *shortens* history matches nothing and re-reads
everything (§8's gap row). A boundary per prefill chunk is too many (a snapshot is
112.6 MiB, `ARCH §5`, and 8.9 ms of copying for the two a turn takes), but a
boundary every 256 tokens is 256 x 74.9 ms of re-read avoided for 8.9 ms of copy -
arithmetic on measured rates, and the reason the interval is a policy number rather
than a constant. Sub-case worth pricing first: store the boundary at the end of each
*turn block* rather than every N tokens, which matches how clients trim.

**7.5 P2 Implement the dense-attention threshold that §5 already recommends.**
Selection currently switches from dense to sparse at 2048 tokens and then stays
sparse forever (`forward.c:1559` compares against the indexer's block budget), while
§5 says below ~32k context the gather costs more than dense. Both halves need
measuring against the same 8k/32k/128k prompts: the attention cost as a function of
context with and without selection, and the top_k value that keeps the gather at
break-even at each width. This pairs with 6.8: fewer blocks to score, and the
scoring parallelised.

**7.6 P2 Stop re-tokenising a prompt the engine already read.** A turn-cache hit is
decided on token ids, and the ids are produced from the rendered text on every
request. A small rendered-prompt to token-id cache makes the follow-up turn's
prefix work free and gives the cache a hashable key. Microseconds against 65 ms per
token; the reason to do it is that it makes 7.4 cheaper to reach.

**7.7 P2 Check that the request path does not serialise read, render, tokenise,
generate.** Ingestion is minutes; the HTTP body read and the ChatML render are
milliseconds, but they are milliseconds spent *before* the engine starts, on the
same serial path, and the panel (`--ui`) only starts counting at the first token.
Worth one profile line rather than an assumption.

**7.8 P1 The two walls an ingested token is made of, and the one idea left that could
cross both.** With the expert pass grouped and the width swept to 512 - its quantiser is
still walked once every eight rows, §7.2b being the attempt at wider and what that cost - a
token is 29.44 ms of profiler sum at the shipping width, and it splits into two halves
limited by two different things - though not the two things this section used to say
(`/tmp/f9_wide.log`, and `/tmp/f8_wide.log` is the same configuration at 29.49). The expert
half is 14.56 ms, half of it, and §7.2 records what it is and is not a bill for: at this
width it reads 106.6 MiB of distinct expert bytes per token, 2.5 ms at the machine's
scattered rate, and spends six times that on the walk of the quantiser between the
reads. `make moe` puts it at 204 GMAC/s for the shape a 512-token window routes
(512 rows, 512 distinct experts, `/tmp/test_cap2.log`), against the 518 GMAC/s `make walk`
gets off a dense Q8_0 row of the same checkpoint - so the stage is 2.5x off the machine's
arithmetic rate at the eight rows a walk takes. §7.2b is the attempt at closing that by
widening the walk, and it is a warning rather than a route: a parked walk reaches 230 GMAC/s
on exactly that row of the table and loses 16 % of the stage in the engine, because past
eight rows the activation panel joins the weights in the cache and the win was an L2
question the probe had no room to ask. What is left of the two directions is the other one -
reading two or four expert rows per load of an activation, which keeps a walk at eight rows
and doubles the arithmetic per byte of activation, and is a different kernel shape and the
bigger lift.

The other half is 7.60 ms of projections, 3.89 of block core, 3.09 of the two mixers and
0.30 of PLE, and it is arithmetic, but the honest version of that sentence is a gap
rather than a ceiling: `make walk` measures these very shapes at 518 GMAC/s on
`attn_qkv` at T=512 and 467 on `ssm_out` (`/tmp/walk_tile1.log`), and the 7.60 ms the
engine spends on them is 343 GMAC/s. 1.51x is sitting between the kernel in isolation
and the same kernel called from 48 layers of chunked prefill, and it is not the row
block, which is now a tile of the L2 budget in both paths. Nor was the whole of it the
machine's: the walk itself read the activation panel once per output row where one load
can serve two rows, and pairing them is **1.19x** of that walk at the 512 tokens a chunk
ingests at (§7.8, `/tmp/walk_pairs3.log`), so the gap this section is left with is the part
that survives a walk that reads half as many activation vectors. What is left to find is the
pool: a chunk's dense stage is six pool_runs a layer with a barrier at each, and the
last thread of each barrier is charged to every thread of the next stage. That is
worth one instrument (`§10`) before it is worth a refactor. The cheap half of that
instrument has now run, on the expert stage and with no checkpoint in it: the 512-row
grouped row of `make moe` is 124.2 ms at the shipping 19 threads, and the same table at
8 threads is 141.1 unpinned (`/tmp/moe_t8.log`) and 141.4 with the pool pinned to the
performance cores (`QE_PIN=1`, `/tmp/moe_t8pin.log`) - pinning changes nothing, which is
exactly what the `src/mem.c` comment says it should. The eleven efficient cores are
therefore not what the expert stage is paying for: taking them out costs 12 % of the stage,
and 203 GMAC/s is what the shape and the walk get rather than what the scheduler owes.

So one half is 2.5x off the arithmetic rate on the expert side and the other is 1.51x
off it on the dense side, and both are arithmetic rather than bytes, which is the reason
to expect the two to be able to pay for each other. That expectation is now measured, and
it does not hold. `make cosched` is the instrument: one pool, one barrier, the shipped
`attn_qkv 10240x2560 Q8_0` projection at T=512 on one group of threads and a 128-expert
IQ3_S `640 x 2560` bucket walk at T=8 on the rest, every half measured alone and then
co-running against the same barrier - and, after the first reading of it, measured again
against the only baseline a pipeline could displace, which is both walks at the full width
one after the other (`/tmp/cosched3.log`).

```
  both walks at the full width, one after the other: 455.8 ms
  split        dense ms  expert ms     sum ms   mixed ms vs full-w   vs sum
   4 + 15           604.1      287.0      891.1      638.2    0.714     0.72
   9 + 10           335.6      402.9      738.5      474.3    0.961     0.64
  14 + 5            226.4      611.7      838.1      641.1    0.711     0.76
```

Against the sum of the two halves alone on their own share of the pool the best split is
0.64, which is the number this section was written on and which reads as a 1.5x prize. It
is not one, because that sum is not a schedule anybody can run: each half in it is doing
the *full* amount of work on *half* the machine, so the baseline is about twice the work
the sequential pass is doing, and a split that adds nothing at all scores 0.63 against it.
Against the sequential pass at the full width the same run is **0.961 at the best split and
0.711 at the worst** - the machine is additive across these two walks, so a split finishes
the same work in the same time and the pipeline would move the wait between walks without
shortening it. The per-thread columns say why: dense falls 320 -> 226 GMAC/s and expert
291 -> 248 when they share the machine, and nine threads on the dense walk already return
68 % of what nineteen do, so neither walk is leaving a quarter of the machine idle that
the other could pick up. There is no thread split at which overlapping them wins, because
both halves scale the same way the whole does.

That is the answer for a *split*, and a split is not the strongest form of the idea. A split
hands a fixed group of threads to each walk, so a thread that has run out of bucket stays
home with it; the form that could eat a tail is one schedule over both halves, where the
threads that finish the expert walk first are the dense walk's to use. That form is no longer
hypothetical. `qe_pool_run_items` (§6.4) runs a set of items with dependency edges as one
dispatch, and `moe_stage_graph` is the engine using it: a chunk's expert stage as seven items
(the routed walk's three, the shared expert's two projections, their silu waiting on both of
them, its projection waiting on the silu) in the one dispatch the walk's three were already
in. `make forward` measures the two schedules against each other on one load and one prompt
at the shipping width, four passes, leg order flipped every pass so neither wins by being the
warmer one, the best pass kept per leg, and asserts that the two leave the same trunk behind
them:

```
  pass 1: one after the other  30.97 ms/token, one dispatch  30.69
  pass 2: one after the other  31.01 ms/token, one dispatch  30.71
  pass 3: one after the other  30.93 ms/token, one dispatch  30.74
  pass 4: one after the other  30.78 ms/token, one dispatch  30.62
  fastest: queued 30.78 ms/token (16.54 of it the expert stage), graph 30.62 (16.32): 1.005
```

(`/tmp/f12_wide.log`.) **1.005x**, which is nothing, and the arithmetic says so before the
run does: at the width that ships there are six hand-backs a layer in the expert stage and
six in the dense one, 576 dispatches for a 512-token chunk, which at the **1.6 us** a hot
dispatch costs (§12) is 0.92 ms of handshake for a chunk and **1.8 us of a 30,710 us token**.
Everything a dispatch-per-matrix schedule pays for is five hundredths of one percent of the
step. What was left to win was the tail, and on a machine additive enough that neither walk
is leaving an idle quarter of itself for the other to pick up (§1.2's twelve efficient cores
streaming as much as eight performance cores) the tail of a 16.5 ms stage is worth about what
two loads of the same checkpoint differ by, which in these logs is 4 %.

**It ships now, and for more than the 1.005x above, for a reason worth writing down.** The
projection recode of §7.14 went into `proj()` in between, and `moe_stage_graph` kept building
its projection items with no panel - and no panel means no integer walk. Measured against the
shipping schedule the graph was then a *different arithmetic* and not a different schedule:
`make forward`'s trunk check failed on `/tmp/f24_wide.log`, and the number that leg printed was
the graph paying for the dequantising walk it had quietly fallen back to. The recode is two
items of the graph now, one fill for the shared gate and up (both read the block output, so the
queued path fills that identical panel twice and the graph fills it once) and one for the down
projection after the silu, the two sharing the panel buffer because the second fill depends on
the silu that depends on the two walks that read the first. With that in place, four passes,
legs alternating, one load, at the width that ships:

```
  pass 1: one after the other  22.89 ms/token, one dispatch  22.50
  pass 2: one after the other  22.98 ms/token, one dispatch  22.26
  pass 3: one after the other  22.85 ms/token, one dispatch  22.50
  pass 4: one after the other  22.88 ms/token, one dispatch  22.59
  fastest: queued 22.85 ms/token (11.11 of it the expert stage), graph 22.26 (10.67): 1.026
  the trunk the two leave behind is the same bytes
```

(`/tmp/f25_wide.log`.) **1.026x**, 4 passes of 4, byte-identical trunk, on by default,
`QE_MOE_CO=0` the way back. The hand-back arithmetic above is still correct and still cannot
explain it: the six hand-backs this stage lost are 1.8 us of a 22,260 us token and the graph
took 590 us of it, so what paid is the tail after all - the half that finishes first has its
straggler slots taken by the other, which is the one thing `make cosched`'s fixed split could
not do and measured 0.961x at (§7.8). An additive machine is additive for whole-stage loads;
a stage's last few hundred rows are not a whole-stage load.

So both forms of the structural idea are measured, and it is worth being exact about what was
wrong, because
the mechanism argument behind it was sound and would have justified weeks of work: it is
true that the two walks are limited by different parts of the machine (a dense Q8_0 walk at
T=512 is at its DRAM plateau, the IQ3_S walk is at the quantiser's port pressure), and
differing bottlenecks are what make overlap *possible*; what they do not buy is
throughput, unless each bottleneck is individually unsaturated at full width. Here both
walks are within a third of saturating the box at half its threads, which is a property of
a 8P+12E part where twelve efficient cores stream as much as eight performance cores
(§1.2). The price of entry that was measured for it - the dense walk paying 1.8 % for two
256-row chunks instead of one 512-row one (`/tmp/walk_tile1.log`, 509 GMAC/s against 518)
and the expert walk getting 8.9 % cheaper for the same split (§7.3c) - is now just the
sub-tiling, which is worth having on its own account and does not need a second pool.

The kernel item this section ended on is built, and it turned out to be an item for both
halves and not only the expert one. One load of a token's activations now feeds two output
rows of the same matrix: `QE_IQ4_NL_MULTI2R_FORM` and `QE_IQ3_S_MULTI2R_FORM` for a bucket's
expert walk, `QE_Q8_0_MULTI2R_FORM` for the dense walk the projections use. Both take the
*eights* of a token group and hand a group's leftover tokens to the forms those tokens use
today, which is what makes the switch bit-neutral rather than merely accurate: the third
section of `make walk` walks 4103 rows at every width from 2 to 512 both ways and counts the
outputs that moved, and it is 0 of 2,100,736 for all three formats (`/tmp/walk_pairs3.log`).
That is also the answer to the caveat this section kept attaching to §7.2b: pairing rows
takes activation loads *out* of the walk, so it cannot be the way the panel leaves L1.
What it buys, one whole-tensor walk against another (`/tmp/walk_pairs3.log`, one-row over
two-row, so the number is the second accumulator and nothing else): IQ4_NL 1.06x at eight
tokens and 1.13x at sixteen and thirty-two, IQ3_S 1.12x and 1.15x, and Q8_0 - where the
dense stage's rows and the chunk's panel meet once per output element - 1.37x at eight,
1.39x at sixteen, 1.36x at thirty-two and **1.19x at the 512 tokens a chunk ingests at**.
`make moe` prices the expert half on the grouped pass the engine runs (125.1-126.1 →
121.0-121.8 ms at 512 rows, 200 → 207 GMAC/s), and `make forward`'s profiler is what settled
it: `moe_ffn` 16.48 → 15.28 ms of a 31.42 → 30.28 ms step and a width-512 sweep of 32.1 →
31.0 ms/token, with every ingestion leg printing the same next token and the same top five
logits as the load before it (`/tmp/f17_wide.log` against `/tmp/f18_wide.log`, 0 FAIL). The
dense half went the same way one width later - `block_proj` 7.36 → 5.77 ms, the two mixers
1.62/1.61 → 1.37/1.40, a 28.06 ms step, a width-512 sweep of **29.7 ms/token** - which is
31.2 → **33.7 tok/s** and 1.52x behind llama.cpp's 51.33 rather than 1.60x, and again the
first step's top five logits are the same bytes (`/tmp/f19_wide.log`, 0 FAIL).

**7.9 P0 The bar, measured on this host rather than quoted.** "Model agnostic and
hardware agnostic, this is one checkpoint on one machine" is a real argument, so it is
worth the trouble of measuring the other side instead of asserting it. llama.cpp
b12359/1b1b770d9 - built with its asserts on, which costs it something - run against
this checkpoint on this host at `-ngl 0 -t 20 -fa on`, prompt processing: **38.79 tok/s**
at a 64-token batch, 49.45 at 128, **57.30** at 256, 51.33 at 512, 51.09 at 1024, 50.26
at 2048 (`/tmp/llama_pp.log`). Three numbers to hold next to that, all `make forward`:
this pass was at 34.7 ms/token (28.8 tok/s) before the two items above, is at **31.1 ms
(32.1 tok/s)** at the width the sweep now ships at (`/tmp/f9_wide.log`, 0 FAIL), and reads
one token at a time at 92.2 ms. So the gap is 1.36x at an equal 64-row batch and 1.60x at the
512-row batch it and we both run at, and it is not one thing:

  - **Half of it is the batch.** Its ubatch default is 512, which is what the width sweep
    only just reached here, and its throughput curve is a batch curve: 38.8 at 64 rows,
    57.3 at 256. An engine capped at a 128-row chunk cannot close a batch gap with
    kernels, which is why 7.3b moved the ceiling before it moved anything else.
  - **The rest is per-token kernel work on the shapes this checkpoint happens to use.**
    At an equal 64-row batch it is 1.36x ahead, and §7.8 is where that lives: the expert
    stage runs 2.5x off the arithmetic rate this machine demonstrably achieves on a dense
    row of the same file, and the dense stage in the engine runs 1.51x off the same
    kernel measured in isolation.

What it is *not* is a byte or a load question: the whole model is resident before the
listener opens and nothing in this comparison moves a device. Decode, where the batch
advantage disappears, is already ours - 105 ms/token here against 130.7 for the same
build and file. The kernel item in §7.8 is the priced route to the remaining 1.36x at
equal batch - the structural idea that was listed next to it is measured dead in that
section - and below a 128-row batch the comparison is not close, so that
is a batch fact rather than a kernel one.

**7.10 P0 What the expert walk is bound by, found by taking one part out at a time.**
§7.8 leaves "the expert stage is 2.5x off the dense rate" as the bill. Four rewrites of
the shipped `qe_gemv_iq3_s_multi` walk were built and raced against it inside one process
on the shape the routed experts have (K=2560, 640 rows per expert, eight token rows per
walk, legs alternating by pass and best of four, activations identical in every leg). At
21.5 MiB of IQ3_S the weights do not survive the 30 MiB L3, which is the regime a
512-token chunk routes in; the 3.4 MiB column is the same binary with five experts
(`/tmp/iq3probe2.log`, and `/tmp/iq3probe.log` is the same instrument before the exit legs
were added, at two sizes and two ratios):

| leg, eight token rows per walk                 | streaming | resident |
|------------------------------------------------|-----------|----------|
| shipped: codebook decode, f32 FMA, block in L1 | 495.7     | 320.4    |
| codebook taken out: four aligned f32 loads     | 0.84x     | 1.90x    |
| `vpdpbusd`, scale applied once per 256 weights | 2.75x     | 4.43x    |
| `vpdpbusd` + what a real kernel owes per 32    | **1.36x** | 1.62x    |
| `vpdpbusd`, one activation scale per row       | 1.67x     | 2.57x    |

GMAC/s in the first column of the table's subject row; every other cell is a ratio to the
shipped leg of the same run. Three readings, in the order the instrument gives them.

*The codebook is not the wall at the shipping width.* Taking the quantiser out of the walk
is worth 1.90x when the weights are cache resident and *nothing* when they are not,
because that leg reads 3.9x the weight bytes and lands on the read ceiling (87 GB/s)
instead of on its own rate. The §11 row about building codebook indices with vectors is
therefore not the route: it optimises a term that is already hidden behind the bytes at
512 rows.

*The multiply is not the wall either, but the scales are.* An integer dot with the scale
applied once per 256 weights is 2.75x, and that number is fiction: every format in this
checkpoint puts a scale in front of every 32 weights (Q8_0's `d`, IQ3_S's nibble scale
inside the block's `d`), so a real kernel leaves i32 for f32 every 32 weights, which is
one convert, one multiply against the activation block's scale, and one fused subtract of
the bias term. Paying all three in the same leg is worth **1.36x**, and the `+128` that
`vpdpbusd` forces on its unsigned operand has to be paid by the *weights* (an exact xor of
the sign bit), because the correction is then 128 times the sum of the activation block -
shared by every output row of the walk - rather than the sum of the weights, which each
row pays again. Taking the per-32 activation scale out altogether (one scale for a whole
row, as a Q8_0 KV row has) buys 1.67x, and whether an activation panel survives that is a
quality decision and not a kernel one.

*Which leaves the arithmetic this file has been circling.* Composite, the expert walk with
an integer dot is 1.25-1.35x of the walk it replaces, once the decode is still in the loop
(it gets ~25% cheaper, since four converts and four multiplies per block stop existing),
and 1.36x of the walk *without* the decode is not 1.36x of the stage: 4.8 of the
`moe_ffn` 16.63 ms/token is router, selection, gather and shared expert, so the stage
goes 16.63 to about 13.5 and the step to 27.6 ms/token, 36 tokens a second. Getting the
loads per multiply below one - which is the only remaining direction the table points at -
means the same int8 activation load feeding several *output* rows, with one scale exit
shared between them. That is a tiled integer gemm rather than a rewritten dot product, and
it is what the engine this bar is set against does with a chunk's activations: quantise
them to Q8_0 once per chunk, tile 5-8 output rows against them, and let the exit cost fall
on the tile instead of on the row. P0 with the price on it: the x-side quantiser, one tiled
form per type, the exit arithmetic above already debugged in the instrument, and a leg in
`make moe` holding the result to the walk it replaced. `moe_ffn` at 1.36x is a third of the
distance from 32 to 51; the tile is the rest.

The f32 half of that tile is built and shipped, and it is cheaper to say than it was to
find: two output rows of one matrix against one load of a token's activations, no recode, no
scale exit, no quality decision, and no register pressure the walk was not already spending
(eight accumulators become sixteen, and the spills go to the L1 line the panel was already
in). It is worth **1.19x** of the dense Q8_0 walk at the 512 tokens a chunk ingests at -
1.37x at eight tokens, 1.39x at sixteen - and 1.13-1.15x of the two expert walks, all of it
with zero outputs moved (§7.8, `/tmp/walk_pairs3.log`). In the engine that is `block_proj`
7.36 → 5.77 ms, the two mixers 1.15-1.18x, and 29.7 ms/token at the shipping width (`/tmp/f19_wide.log`). That is the loads-per-multiply item
priced on the arithmetic the engine ships, and it is a reminder that the 1.36x above was a
price on the *dot* and not on the stage: the same direction, taken without the recode, is
worth an eighth of the dense stage and a twentieth of the expert stage, paid for in registers
instead of in nats. What the integer tile still owns is the scale exit a tile shares, and
that is the part §7.11 ends on.

**7.11 P0 Built, measured, priced against its neighbour, and shipped on at that price: the int8 activation panel is 1.10x of a step and 4x the loss noise of the change it sits next to.**
§7.10 priced an integer expert walk at 1.36x of the dot it replaces and forecast the stage at
13.5 ms/token. It is built, wired, and measured: a chunk's gathered rows are recoded to one Q8_0
panel per thread (`src/moe.c`, `moe_panel`/`moe_panel_fill`, three things per 32 activations -
values, scale, and the -16x-the-sum seed `vpdpbusd` wants), and the experts are walked with
`qe_gemv_i8_t` instead of being dequantised to floats every 32 weights. On wherever the weight
format has an integer walk, which is both expert formats this checkpoint stores (IQ3_S gate/up,
IQ4_NL down). It came into the tree behind `QE_I8_MOE=1` and `qe_moe_set_int8()` and off by
default; the quality leg below is why, and the same leg is the reason the sentence "it is
1.10x of a step" is not on its own a reason to ship it. It ships on now, for the reason §7.13
states - there is no safer kernel left to have this argument with - and the leg below is the
price rather than the veto.

Two loads of the checkpoint, both printed by `make forward`'s `== int8 expert walk ==` leg, which is
two profiles of one slice in one process: **the grouped expert pass is 13.48 ms of a token against
the 16.35 the f32 shared-dequant walk takes - 0.824x - and the ingested token goes 31.15 to
28.32 ms, 1.100x** (`/tmp/f16_wide.log`, 0 FAIL; the load before it read 13.96 against 16.62 and
0.840x, `/tmp/f15_wide.log`). At the shipping width 512 the sweep reads **29.1 ms per ingested
token = 34.2 tok/s with the switch on**, from the 31.1 the default takes, with `moe_ffn` at 13.47 of
a profiler sum of 27.65 and the width
curve flat from 192 to 512 (29.0 at 192, so the last decimal is drift). The stage landed where §7.10
forecast, one panel fill short of it. `make moe`'s 4d carries the same switch as a fourth leg of the
whole grouped pass on synthetic buckets: **0.778-0.823x of the f32 walk for every T from 8 to 512**
(0.606 at T=2 and 0.495 at T=4, where a group of eight is two walks and the panel is nearly free).

*The loop order is the finding, not decoration.* Filling the panel where `gate_up_multi` had it -
columns outside, token groups inside - re-quantises a group's rows once per column chunk
(`ceil(run/8)` times at the shipped group of eight) and made the pass **12-26 % slower** than the
walk it was meant to replace. Hoisting the fill above the group loop, with the token group outside
and the weight columns inside, so one fill streams past the whole matrix instead of being remade
every chunk, is what turns it into **1.17-1.29x**. The two paths now want opposite loop orders and
both are kept: the f32 walk shares an eight-row window in L1 and keeps columns outside, the integer
walk streams 1.4 MB of gate+up per group and puts the group outside. Nothing about the f32 path
should be "tidied" into the other shape without re-running the leg.

*What the recode is checked against.* Relative error against a double dot is not a correctness
metric for a lossy kernel - a row that cancels four orders of magnitude turns a rounding into a
headline - so `check_i8_walk` (`make fast`) checks the two promises separately and exactly: the
quantiser's (a block of 32 comes back within half its own scale, and the seed is -16 times the sum
of the values *actually stored*, which is exact against an exact integer), and the kernel's (the dot
it returns is the dot of the *recoded* activations, `xr = s_b q_i`, against the same weight row -
a number this test builds for itself, so the bound is arithmetic and not a tolerance). Worst
relative error over 224/448/544 shapes: 4.1e-05 (Q8_0), 6.9e-05 (IQ4_NL), 5.4e-04 (IQ3_S), with no
coordinate needing the cancellation fallback. Writing that check found two things: the panel is in
32-value blocks while an IQ3_S weight block is 256, and reading the panel at the *weight* block
count evaluates a reference on the first 32 activations of a 256-wide row, which is a number no
kernel computes (worst "error" 222x the magnitude of every term in it); and the Q8_0 tile looped
tokens 8/4/2 at a time with no form for one row, so any T the tiles did not divide - 3, 5, 7, 9,
the width of a partly-filled expert bucket - read `ys[t+1]`, a token pointer the caller never
passed. The IQ walks already handled that by re-taking the previous row; the Q8_0 walk now does the
same, and the check sweeps T from 2 to `QE_IQ_PARK_MAX+1` for exactly that reason.

*Two arithmetics is a fact about the engine, and the gate suite now says so.* The grouped pass is
where the panel lives, so prefill and decode are no longer the same bits: `moe_ffn` for one row
still takes the f32 walk. That is deliberate - the same recode a Q8_0 key/value cache already makes
of a vector this one multiplies - but it made `make forward`'s two schedule legs (`ordered calls
against one blocked walk`, `serial expert passes against one grouped pass`) into comparisons of two
arithmetic rather than of two schedules, and their `same next token` asserts went over on a
48-layer walk through a router whose top-k moves at 1e-3. Both legs now hold the recode off for
their own comparison, with the reason in place of a wider tolerance, and the recode's own cost is
printed where it belongs: the `== int8 expert walk ==` leg reports the worst move in the head
against the head's own range, `make score --ingest N` reports it as next-token loss over windows of
a corpus (teacher forcing steps one token at a time, so the sweep above cannot see a chunk and never
could), and `make moe` 4a carries the accuracy line (3.9e-02 of the row scale against the same
normalisation the grouping check uses, with grouping exactness still asserted with the recode off).

*What it costs.* 2.43 MB of panel per layer object - 19 threads x 32 rows x (2560+640) bytes plus
scales - across 48 layers, and one fill per group per layer, which at T=512 is 0.14 % of the step.
And an arithmetic that is not the one a decode step runs. What that is worth at the head is smaller
than the things next to it, which is the sentence this change needed and did not have until the leg
printed it: **the recode moves the head by 5.2 logits against a head whose largest logit is 25.0 -
21 % of its own range - with the same next token** (`/tmp/f16_wide.log`). The bucket reordering that
shipped before it moves the same head by 8.4 logits (34 %) at the same width, and the projection walk
by 5.3 (26 %) in the served-path leg. The expert stage's own error against a double reference is
5.4e-04 relative (`make fast`), and `make score` cannot see any of it, because teacher forcing steps
one token at a time and a one-token group has no grouped pass to recode - so the corpus-level number
is in `make score --ingest N`, which prefills windows twice and prices the next token's loss and
choice instead.

*What the recode costs, and the verdict that first put it behind a switch.* `make score --ingest 5` prints two A/B
tables over identical 512-token corpus windows: the recode against the f32 walk, and - because a
lossy change is priced against the lossy change it already ships next to and not against an invented
bound - the grouped expert pass as the yardstick. Over five windows (`/tmp/ingest2.log`) the recode
moves the next token's loss by **1.100 nats mean and 4.686 worst** and the greedy choice **moved in 1
of 5**, worst head gap 5.90 against a head of 20, i.e. **30 % of the head's own range**. The grouped
pass, on the same windows, moves loss **0.285 mean and 0.494 worst**, moved the choice **0 of 5**,
worst gap 2.01/19 = 11 %. Four times the noise of its neighbour and a third of the head for 6 %
of an ingested token is not a trade this engine makes while it still believes a better kernel is
around the corner, so the walk came in built, checked, switchable, and **off**. `make forward`'s
one-prompt head gap argued the other way - 21 %, *less* than the grouping's 34 % at the same width
- and it understated the cost because a single prompt prices neither a moved greedy token nor a
loss column; the sweep cannot see it at all. Anything that changes the arithmetic of a chunk owes
the `--ingest` table, not a tolerance. What has since changed is the alternative, not the price:
§7.13 measured the stage's whole remaining headroom at 1.2-1.6x, this walk is 1.21x of the stage,
and the rewrites that were supposed to be worth the other 2x measured dead, so the default is on
and `QE_I8_MOE=0` is the way back.

*What is still open, with the price on it.* The stage is still half an ingested token - 15.41 ms
of a 28.06 ms profiler sum at the shipping width (`/tmp/f19_wide.log`) - and 15.41 ms against the
2.5 ms scattered-read floor is 6.2x of arithmetic. One direction this section used to end on is
now built and shipped: one load of a token's activations feeding **two output rows** of the same
expert, on the f32 arithmetic the engine already runs, is 1.06x of a walk at eight tokens and
1.13x at sixteen and thirty-two, +2.6-4 % of the 512-row grouped pass (125.1-126.1 → 121.0-121.8
ms, 200 → 207 GMAC/s), and 0.73 ms of `moe_ffn` - 16.48 → 15.28 ms of a step that went 31.42 →
30.28 (`/tmp/f17_wide.log` against `/tmp/f18_wide.log`, 0 FAIL, the same next tokens and the same
top five logits on both sides). It moves zero outputs, and `make walk`'s third section is the
instrument that says so (§7.8). The same pair taken to the dense Q8_0 walk is `block_proj`
7.36 → 5.77 ms and 29.7 ms/token at width 512 - **33.7 tok/s**, 1.52x behind llama.cpp - and
it moved zero outputs either (`/tmp/f19_wide.log`). Two directions are left. One *int8* activation load serving
several output rows: the tiled integer gemm §7.10 ends on, where the exit is what a tile shares,
and which is what `i8_rows_cap()` dares only two rows of for Q8_0 because three cursors per token
row are already what the register file is for. That one is now built for the expert formats too,
measured, and switched off: it is worth 0.95-1.04x of the walks it replaces and 0.98-1.01x of a
grouped stage, for sixteen accumulators (§7.12). It had a competitor when it was measured, the f32
pair above having already taken the activation load this tile was going to halve, and it lost to
that competitor as well as to the row (`/tmp/f19_wide.log` reads the integer walk's pairing at
0.887x of a faster walk against 0.836x of a slower one). And a bucket's group past eight rows - the
direction the 4x-cheaper panel was supposed to unlock, now measured twice. With the integer walk on, the
parked-cap sweep read 27.77 ms/token at the shipped cap (a group of eight) against 28.96 at 12,
28.37 at 16, 28.49 at 24 and 28.43 at 32, i.e. **2-4 % worse for every cap that widens a group**;
with the pairing on and the integer walk off, the same sweep reads 28.35 at the shipped cap
against 28.65 at 12, 29.69 at 16, 33.08 at 24 and 32.39 at 32 (`/tmp/f19_wide.log`; the load
that first had the pairing on read the same shape 1.7 % worse at cap 12, `/tmp/f18_wide.log`), and
the cap that used to be 1.6 % *faster* than the default (`/tmp/f17_wide.log`: 30.91 against 31.41)
is now 1.1 % slower, because
the pairing is already collecting the same tail-of-a-group waste that a wider cap collected
differently. The reason widening stays negative is structural: the integer forms exist for two,
four and eight token rows, so a group of sixteen is *two* walks over the same expert bytes and the
byte saving a wider bucket exists for is handed straight back. Widening
a bucket needs `QE_IQ4_NL_I8_FORM`/`QE_IQ3_S_I8_FORM` at T=16 and 32 first, and what those need is
not what this section thought: the panel cursors are three registers, not twenty-four, and the
reason a wider form cannot be written is that the forms hold one accumulator per token row, so
T=16 is sixteen accumulators to a file that has sixteen in it (§7.12, which measured the two-row
version of exactly that trade). A transposed panel, `[block][row][32]`, makes sixteen rows
addressable as one run; it does not make sixteen sums live at once.
`make walk`'s dense column has an f32 pair leg, it is built and on, and it is worth 1.19x at the
shipping width - 7.36 → 5.77 ms of `block_proj` and 29.7 ms/token at width 512 (§7.8);
the Q8_0 *integer* tile is now measured against the walk it replaces rather than assumed from one
width, and it is 0.89-0.97x: worth no recode of the trunk's activations, and worth saying plainly
after a change that moved 1.19x with no bits moved (§7.12). Nor is what is left of the dense walk its row tile: with the pairing on, 256/512/1024
KiB of `QE_MULTI_KIB` read 205.6/205.8/203.5 ms for 100000 rows at 512 tokens, 1 % apart,
which is a tile budget saying it stopped being what that walk waits on once a pair halved the
panel traffic it was sized against (`/tmp/walk_pairs3.log`, `/tmp/kib_512.log`,
`/tmp/kib_1024.log`). That budget is one number for both core classes of this host, which §5.7
already argues it cannot be the lever; the integer tile below has no budget at all, because it has
no rows to block.

**7.12 P0 Built, measured, and switched off: the integer tile shares three operands and spends
the entire register file to get them.**

§7.11 left "one *int8* activation load serving several output rows" as the one thing the integer
side still owned. It is built for both expert formats: `QE_IQ4_NL_I8_R2_FORM` and
`QE_IQ3_S_I8_R2_FORM` (`src/kern_iq.c`) walk two output rows of one expert matrix against one load
of a token's activation block, taking the eights of a group and handing its leftover tokens to the
single-row forms they use today, behind `qe_i8_rows()` (`QE_I8_R`). One row is what every integer
format now takes (`QE_I8_ROWS_DEFAULT`, `QE_IQ_I8_ROWS_DEFAULT`); the measurement that decided it
covered all three, and it is below.

*Nothing moved, so this is a GMAC/s question and not a loss table.* Every output row keeps the
single-row form's seed, block order and exit, which makes the two settings bit-identical rather
than close, and the gate is equality: 27 rows of the new fourth section of `make walk` - 4103 rows,
an odd count so the last row has no partner, T = 8, 9, 11, 13, 16, 24, 32, 64 and 512, three
formats - each read **0 of up to 2,100,736 outputs moved**, and `make fast`'s `check_i8_walk` runs
the other setting over a buffer pre-filled with a sentinel and requires the untouched gaps to still
hold it (`/tmp/walk_tile6.log`, RC=0). The gate paid for itself twice before the first number was
read: a paired path that dropped the four-token leftover block moved 12306 of 53339 outputs at
T=13, and a leg that read the knob inside the timing loop instead of capturing what shipped
compared two runs of the same setting and passed.

*What it is worth is nothing.* Gain is one-row over two-row, min of three reps, 19 threads,
K=2560; the streaming columns are 400000 rows (0.5 GiB of IQ4_NL, 0.4 of IQ3_S) and the resident
columns 640 rows copied sixteen times (14.1 and 10.7 MiB, one expert's worth of reuse):

| T | IQ4_NL streaming | IQ3_S streaming | IQ4_NL resident | IQ3_S resident |
|---|------------------|-----------------|-----------------|----------------|
| 8 | 0.95x            | 0.98x           | 1.00x           | 1.02x          |
| 11 | 0.99x           | 1.01x           | -               | -              |
| 13 | 0.99x           | 1.03x           | -               | -              |
| 16 | 1.01x           | 1.02x           | 1.01x           | 1.03x          |
| 32 | 1.02x           | 1.04x           | 1.01x           | 1.03x          |

and the same switch priced at the level a stage is paid at, `make moe`'s 4d grouped expert pass
run twice with the knob held down each way (`/tmp/moe_r1.log`, `/tmp/moe_r2.log`), is **0.98-1.01x
at every width from 2 rows to 512**: the int8 leg is 0.436 against 0.439 ms at 2 rows, 1.026
against 1.030 at 8, 3.981 against 4.004 at 32, 17.764 against 17.429 at 128, 35.637 against 35.671
at 256. The f32 walk in the same table moved by less than that, which is the honest reading: at a
stage, the integer tile is the same walk.

*The Q8_0 control is the half that mattered, because that tile was already shipped.* Same
section, the format with no codebook decode, 100000 rows, three runs of the table: **0.96x, 0.92x,
0.94x, 0.97x** at eight, sixteen, thirty-two and five hundred and twelve tokens, with the two other
runs reading 0.89-0.95x over the same widths. A pair of Q8_0 rows is therefore *slower* than the
two rows it replaces, at every width, which is the same answer the expert formats gave and one
fewer exception in the file. Two corrections come out of it: the 1.02-1.08x this file used to quote
for that tile was one width of one run, and the 1.41x at eight tokens this section first wrote down
was the single-row leg of one run being 1.5x slower than it is in the other three - the ratio that
survives repetition is 0.96x, and it is the ratio `QE_I8_ROWS_DEFAULT` now agrees with, so all
three integer formats take one output row and `QE_I8_R=2` is the way back. Absolute times in these
runs moved by 40 % with the host's load average (5.9-8.9, `uptime` alongside each log:
`/tmp/walk_tile6.log`, `/tmp/walk_tile8_1.log`, `/tmp/walk_tile8_2.log`, `/tmp/walk_tile9.log`),
which is why the claim here is a ratio between two legs of the same pass, min-of-three, alternating,
and why the outlier is named rather than quoted.

*Why 1.00x, with the arithmetic.* An integer dot spends about seven operands on the way from a
32-activation group to an accumulator: the panel load, the broadcast of the activation block's
scale, the broadcast of the bias seed, the `vpdpbusd`, the `vcvtdq2ps`, one multiply of the two
scales, one FMA. A pair of output rows shares three of them - the panel load, and the two
broadcasts - and doubles four. The f32 pairing has the opposite split, which is why the same idea
is 1.13-1.19x there: its per-token cost is four FMAs and four 32-byte float loads, so one
dequantised block feeding a second row is halving the expensive part instead of trimming a seventh
of it. What a pair does cost is registers: two rows against the eight-token group is sixteen
accumulators, which is the whole architectural register file, before the two weight blocks, the two
scales and the shared operands. `objdump -d` on `qe_gemv_iq4_nl_i8` reads the frame once, so this is
not a spill story - there is simply nowhere to put a seventeenth accumulator, and a tile that has
to recycle a register to hold a shared operand has stopped saving loads. That also retires §7.11's
other open half for the right reason: `QE_IQ4_NL_I8_FORM`/`QE_IQ3_S_I8_FORM` at T=16 and 32 are not
blocked by the panel cursors (three of them, in three registers), they are blocked by T
accumulators, so a sixteen-row group is sixteen accumulators per output row and no transposed panel
arrangement changes that.

*A tile narrower than the token group is worse than no tile.* A temporary `r2t4` instantiation -
two rows against a four-token group, so that a group of eight is two narrow tiles - read **0.82x
(IQ4_NL) and 0.81x (IQ3_S)** of the walk it replaced (`/tmp/walk_tile2.log`), because the weight
matrix is streamed once per tile and a second tile reads it again: the pair exists to save panel
traffic, and at a four-token tile it saves half of what a full-width pair saves while paying for a
second read of 0.5 GiB. The form is deleted; the row is kept because it is the clearest statement
of where the tile's ceiling is, which is one tile per group and no wider than the register file.

*What this leaves, with the price on it.* Sharing the exit does not pay, so the exit is what has to
shrink, and there are two ways to shrink it that this section did not have. One is §7.10's
1.67x/2.57x leg, one activation scale per panel row instead of per 32 values, which deletes the
per-block scale broadcast and its seed for every row of the walk - and is a change of number, so it
owes `make score --ingest N` and the 0.285/1.100-nat yardstick of §7.11. The other costs no
accuracy at all and is the other half of the same arithmetic: this host's `-march=native` resolves
to `-march=arrowlake-s`, which defines `__AVXVNNIINT8__`, and a signed-against-signed `vpdpbusd`
(`_mm256_dpbssd_epi32`, verified to build and run here) needs neither the `^0x80` the walk puts on
the weights nor the -16x-the-sum seed the panel carries - two of the seven operands, deleted for
every row of the walk instead of shared between two of them.


**7.13 P0 The signed exit is on where the decode hands it the operand and off where it does not,
and the expert stage's "3.1x" against a dense Q8_0 row was three comparisons wearing one number.**

*The exit.* §7.12 ended by naming `vpdpbssd` as the one thing that shrinks an output row's exit
without costing accuracy, and it is built: `qe_i8_ssd()` is the codebook walks' answer,
`qe_i8_ssd_q8()` the Q8_0 walk's, `QE_I8_SSD` overrides both, and the fourth section of `make walk`
races them leg against leg in one process on one panel (`/tmp/walk_3.log`). The codebook walks take
the signed dot and are worth 1.03-1.17x on IQ4_NL and 1.09-1.18x on IQ3_S over the widths
8..512 - **1.12x and 1.16x at the 512 tokens a chunk ingests at**, which is the row that matters.
A Q8_0 row loses it at every width the table runs: 0.96x at 8 and at 512, 0.90x at 32, 0.83x at 64.
That is the split the two defaults now encode, and it is the split the decode explains: an IQ3_S or
IQ4_NL row is turned into *signed* int8 by the walk that is about to multiply it, so the signed dot
deletes the `^0x80` and the -16×-sum seed for nothing, while a Q8_0 row is already bytes, never had
those two operands, and pays for the form switch instead. Why switching to the *shorter* exit costs
a Q8_0 row 4-17 % rather than nothing is not something this instrument can see - both forms are
instantiated for every tile, and the loss is not monotone in the token group - so the record keeps
the number, the gate keeps the choice, and `qe_i8_ssd_q8` is where a reader goes to re-measure it.
The gate says what the engine claims rather than what the argument claimed: a signed codebook leg
may not be slower, and the unsigned Q8_0 leg - the one a Q8_0 walk now runs - may not be more than
5 % slower than the signed one it replaced.

*The two are not bit-for-bit, and `src/kern.h` said they were.* The seed that cancels the unsigned
operand's bias rides inside the float accumulator and is paid back by the horizontal sum at the end
of a row, so the two walks sum different numbers and round differently: `check_ssd` holds the
difference to 1e-5 of the largest `|y|` and measures 3.4e-6 on this host's three formats, at every
width from 8 to 512. The header's claim of "the same bits downstream" is corrected with it.

*The row pair, re-asked under the cheaper exit, is still dead.* A pair of output rows against one
int8 activation load is the structure §7.11 and §7.12 left standing as the way to take the loads
per multiply below one, and §7.12 measured it at 0.95-1.04x while the exit still carried the xor and
the seed. The exit is two operands shorter now, which is exactly the part a pair shares, so the
question is re-run rather than inherited: the fifth table of `make walk` walks one output row
against two per activation load on the exit each format uses. It reads **0.91/0.88/0.96/0.96x**
(IQ4_NL), **0.85/0.88/0.94/0.96x** (IQ3_S) and 0.89/0.93/0.87/0.99x (Q8_0) at T = 8/16/32/512.
Deleting the shared operands did not buy the second row, because the thing a pair spends was never
the operands: it is the sixteenth accumulator in a file with sixteen of them. Both forms are now
measured dead, at both exits, and "one int8 load feeding several output rows" leaves the shortlist
by measurement and not by taste.

*The 3.1x.* For several loads of this checkpoint the expert stage has been quoted as **207 GMAC/s
against the 637 GMAC/s** `make walk` gets off a dense Q8_0 row of this same file (`/tmp/walk_pairs3.log`),
"3.1x", in MANUAL.md, in `docs/DESIGN.md` §12 and in §0 of `AGENTS.md` at 518. That one ratio has three
differences in it: the two walks multiply different formats, one of them walks a token row per 512
weight reads and the other per 10, and the routed pass gathers rows, applies silu and reduces
partials where a dense projection does none of that. `bench_moe` 4e is the instrument that holds MACs
equal by construction and takes the three out one at a time - five legs, routes drawn before any
timing, order reversed on the odd passes: the shipped grouped pass; the same three matrices walked
contiguously with every expert handed exactly `T*topk/512` token rows *in the same call shape the
grouped pass uses* (8 output rows, a park-sized token group); the same with the window's own
histogram; the same MACs with every token row of the window sharing one weight read, which is the
shape a dense projection walks; and the uniform leg with the weights in Q8_0 (`/tmp/moe_4e_3.log`,
19 threads, gate/up IQ3_S 640×2560 and down IQ4_NL 2560×640):

| T | routed | uniform | real-fills | dense-shape | q8_0 | GMAC/s r/u/d/q8 |
|---|--------|---------|-----------|-------------|------|-----------------|
| 256 | 58.6 ms | 73.7 | 76.9 | 37.9 | 51.9 | 215/171/332/242 |
| 256 (second load) | 76.1 ms | 79.0 | 73.7 | 43.4 | 51.7 | 165/159/290/243 |
| 512 | 120.2 ms | 97.6 | 114.4 | 98.1 | 74.0 | 209/258/257/340 |
| 512 (second load) | 123.0 ms | 88.7 | 107.7 | 104.0 | 76.3 | 205/284/242/330 |

At the width that ships: the expert stage as routed is **1.23-1.39x** the time of the same
arithmetic walked contiguously in the same call shape, and **1.33-1.63x** the time of the same
arithmetic in Q8_0 at equal MACs and equal shape (q8/unif 0.76x, 0.86x). At 256 rows the contiguous
leg is 1.04-1.26x *slower* than the shipped pass, which is the instrument's own warning about
itself: a reference that writes its token rows at a 10 KiB stride is paying for streams the routed
pass avoids by staging a group through stack buffers, so read that row as "the reference is not a
free lunch", not as "the gather is free". Either way the claim is dead: there is no 3x between the
expert stage and a walk of these same matrices, and the format this engine chose is worth
**1.16-1.54x** of the distance (q8_0 at equal MACs and shape), which is the only column in that
table that is allowed to be read as a format price at all.

*Why 518 and 637 are not the same kind of number as 207.* Those are the rates of a 272 MiB Q8_0
tensor read at 73 GB/s, this host's ceiling, with one weight byte serving 512 token rows: 21 MACs
per byte. A 512-row window routes each of its 512 experts ten rows, which is 21 MACs per byte
divided by nothing but paid against 106.6 MiB of *distinct* expert bytes per ingested token, and the
dense-shape leg is the closest this file gets to asking "what if an expert had the chunk's rows" -
0.82-0.84x of routed at 512, and 0.57-0.65x at 256, where a ten-row group is too narrow for the
shared dequant to seat a walk and the reference's output streams cost it most. So the ceiling for
the expert stage, measured with no experts in it, is
**1.2-1.6x**, and the integer walks already built are worth 1.21x of the stage by themselves
(§7.11). Which moved the bar's remaining distance out of the kernels and onto a decision: the two
integer walks were priced, `QE_I8_MOE` and `QE_I8_PROJ` were the switches, and what stood between
this engine and llama.cpp was mostly the loss those switches charge and not a 3x kernel this
instrument says was never there. Both switches are on (§7.14), which is the whole of what this
instrument said was available and took ingestion from 36.4 to 42.0 tok/s, with the expert stage's
own schedule worth a further 1.026x on top (§6.4) for 43.1 tok/s; the 1.2-1.6x ceiling is
what makes the price payable rather than a reason to reach past it.

**7.14 P0 Built, measured, and on: taking the activation scale out of a walk's token loop is worth 1.13-1.27x of the codebook walks, and the same recode under the dense walk is worth 1.089x of a step.**
§7.10 measured one activation scale per row at 1.67x of the dot against the 1.36x the per-block
panel gets, and left the number as a forecast because nothing in the tree could walk such a
panel. It can now. `qe_quant_q8_0_row_rs` (`src/kern.c`) writes one scale for a whole activation
row into every one of the row's `nb` scale slots, and `QE_IQ4_NL_I8_FORM`/`QE_IQ3_S_I8_FORM` grew
an `RS` argument that runs the accumulator in weight-scale space and multiplies the row's scale
once after the block loop. What leaves the token loop is a broadcast and a multiply per token per
32 weights; what stays is a convert and an FMA. The panel is unchanged in layout, and a row-scaled
panel is a per-block panel with equal scales, which is why the Q8_0 rows - whose unsigned exit
owes a seed per block whatever the scale looks like - read it with their kernel unchanged and are
the control column of the table below rather than a leg. `QE_I8_RS` selects it, both sides read
the same knob, and `make fast` gates the three row-scaled walks against the double reference
(worst relative error 5.6e-05, 1.5e-05 and 5.2e-05).

*What it costs is range, and that is measured as a bound and not argued.* One outlier spends
itself on a whole row: `check_rowscale` (the fifth table of `make walk`) compares the two walks
over 4103 rows of synthetic activations at 8, 11, 16 and 32 tokens and holds the difference to
5e-2 of the largest `|y|`, measuring 0.80 % on IQ4_NL, 0.51 % on IQ3_S and 0.63 % on the Q8_0
control, with 0 non-finite outputs at every width. It is not bit-identical and does not claim to
be - a coarser quantisation is what was bought - so the leg is a relative bound like `check_ssd`,
and the quality price itself is `make score --ingest`'s job.

*The walk, and then the stage that walks it.* One output row against one row-scaled panel is
**1.13x at eight token rows, 1.17 at sixteen, 1.13 at thirty-two, 1.19 at sixty-four and 1.27x at
the 512 a chunk ingests at** on IQ4_NL, and 1.13/1.15/1.19/1.17/1.21 on IQ3_S; the Q8_0 column
reads 1.00-1.01x, which is the control behaving (`/tmp/walk_rs1.log`). The curve is what
"two operands per token per block" is worth, drawn against the number of tokens per block, and it
says the engine cannot have the 1.27x: `QE_MOE_SHARED_GROUP` hands a walk at most eight rows of a
bucket, so what ships walks the 1.13 column. What the rest of the curve is worth has now been
measured rather than forecast, twice, and it is worth nothing - see the last paragraph.
In the grouped pass, where a bucket's panel is filled once and then streamed, `make moe`'s 4d leg
reads the row-scaled panel at **0.686 of the f32 walk at 512 rows against the per-block panel's
0.783**, and 0.734 against 0.846 at eight rows: the row scale is **0.87x of the panel it replaces**,
less than the isolated walk's 1.13-1.27x because the gather and the panel fill are still paid.

*What the two recodes are worth in the engine, and what they cost, both on the shipping default.*
One load of `make forward` at the width the sweep ships, `QE_I8_MOE`, `QE_I8_RS` and `QE_I8_PROJ`
all on: the expert stage is **15.13 ms/token dequantising, 12.49 with the integer panel, 11.28 with
the row scale** - the row scale alone is 0.903x of the panel it replaces and 0.746x of the float
walk - and a whole ingested step in the same leg is 27.14, 24.35, 23.16. One level over, the dense
projections are **5.98 dequantising against 4.87 walked** (0.815x) and the step 24.93 against 22.88.
The sweep reads the same thing from outside: **27.5 ms/token at width 512 before the two switches
went on, 23.8 after**, which is 36.4 against **42.0 tok/s**, and for the first time the sweep's own
verdict is "the default is the fastest setting" (`/tmp/f22_wide.log` against `/tmp/f20_wide.log`).
The next 0.6 ms of it was not an arithmetic: putting the expert stage's seven dispatches into one
stage graph, and then putting the chunk recode inside that graph rather than leaving it behind,
takes the same width to **23.2 ms/token** - 43.1 tok/s - and the sweep is now flat and tied from
192 to 512, so its verdict line names 192 and the default only ties it
(`/tmp/f25_wide.log`).
What it costs is on the same pages: the recode leaves the head 30 % of its own range from the walk
it replaces and the row scale 35 % on top of that, both with the same next token; the schedule
against the ordered walk reads 15 % where the engine as served reads 25 %; the ordered walk is
still the step loop byte for byte (trunk identical, logits diff 0.0e+00) and eight greedy steps
from a 598-token prompt are identical id for id. Three legs of that file were pinning the wrong
thing while the recodes were opt-in and went red the moment they were not - the shared-dequant
schedule leg, the bucket-order leg and the schedule-alone bound all compare two *schedules* of one
arithmetic, and a recode that lives inside one arm of a schedule comparison prices a recode and
calls the result a disagreement. They hold the recodes off and keep their fractions (a third for a
schedule, a half for an arithmetic); the recodes are priced by their own legs above and by nats.

*A ten-row group is built, in both the places it could be, and neither pays.* The obvious reading
of the curve above is that a walk handed ten rows of a bucket instead of eight takes the 1.19
column rather than the 1.13 one, and a ten-row instantiation of both expert forms was built to
test it - not kept, because it pays nothing, so a reader who looks for `qe_row_iq4_nl_i8_10_rs`
in `src/kern_iq.c` will not find it. Ten accumulators beside a dequantised block, a scale and a
codebook is thirteen of the sixteen ymm registers, and the form does spill a couple. Handing
the *bucket* ten rows - `moe_shared_group()` returning ten when the integer walk is running - is
**8-25 % worse** across `make moe`'s 4d table at 16 rows and past: with a cap of ten a bucket of
sixteen is walked as ten, four and two, three passes of an expert's weights where eight makes two,
and the expert walk's ratio to the f32 walk goes 0.718 → 0.843 at sixteen rows and 0.686 → 0.695 at
512 (`/tmp/moe_t10.log` against `/tmp/moe_rs1.log`, same instrument). Handing the *dense* walk ten
rows, where a bucket is not in the way and a 512-row walk really does decode 19 % less per token,
is a **wash**: `make walk`'s row-scaled column reads 124.3 against 119.9 µs at 64 rows and 857.2
against 843.8 at 512, while the per-block column of the same table - code untouched by the change -
moves 141.7 against 143.1 and 1077.4 against 1070.6 in the same runs (`/tmp/wide_walk.log` against
`/tmp/walk_rs1.log`). Eight rows is where this instruction set's register file stops a walk from
going wider, from either end, and §7.12's sixteen-accumulator verdict is now a verdict about ten as
well.


**7.15 P0 A survey, not a measurement: what the engines that ran this width on x86 did about it, and what each of those measures would cost this tree.**

Everything in this section is second-hand on purpose. The bar is 1.16x away and the two walls
left are kernel-shaped: the expert stage at 205 GMAC/s (§7.10's own estimate of the same inner
loop from uops is roughly 10x above it, and the published AVX2 int8 gemm wall is about 3x above
it again), and the dense walk at 343 GMAC/s in the engine against 518 measured in isolation on
the same shape (§7.4, §7.8). So §7.13's "1.2-1.6x of remaining headroom" is a distance to a
nearby kernel and not a distance to this machine, and before writing a fifth rewrite of the
expert dot the cheaper question is what the engines that shipped this shape actually did. Read
that way, and against the record of what this tree has already measured dead, the list is short.

| Measure | Who did it, and what they reported | Here |
|---|---|---|
| Dequantise to int8 *into tiles* on the fly and feed a 16x16 `vpdpbusd` microkernel | llama.cpp PR #27851: pp2048 16.34 → 67.10 tok/s (4.1x) on AVX-VNNI, 1.7x on AVX2, and gated off below 64 rows because a narrow tile pays its own prologue | **Open, and the biggest item left.** The recode half of it ships (`QE_I8_MOE`, `QE_I8_RS`, `QE_I8_PROJ`); what does not exist is the tile, which is what §7.10, §7.11 and §7.12 kept arriving at and never built. §7.12 measured a *two*-row int8 tile at 0.95-1.04x, which is the sixteen-accumulator wall and not the tile's: a 16-token group against a handful of output columns is a different register budget, and the row-scaled panel (§7.14) is what makes its exit cheap enough to attempt. The number this has to beat moved twice since this row was written and is now **1737 GMAC/s**: with the exit hoisted (§7.17) and one scale per output row instead of per block, a three-row-by-four-token tile is **1.84x** of the walk that ships, built, exact to the bit against it, and unwired for want of a load-time copy (§7.18) |
| Repacked weight layouts: same bytes, interleaved so one decode hands the dot whole 32-weight groups (`block_q8_0x4/x8/x16`, `block_q4_Kx8/x16`, `block_iq4_nlx4/x8/x16`, ik_llama's Q8_K_R8) | ik_llama.cpp reports 1.25-3.36x at pp512 with the repack itself ~7 % of the gemm at b512; KTransformers does the same rearrangement at load time, 64-byte aligned, per expert; T-MAC isolates the permutation alone at +1.39x and the interleaving alone at +1.42x | **Closed for the experts, open for the dense matrices.** A permutation changes no value, so numerics are not the objection; the objection is that a permuted copy is a second copy, and 85 GiB of experts do not fit twice in 125 GiB (§11). The *non*-expert matrices are ~17 GiB of the file, so a permuted copy of those does fit, and they are the 4.84 ms/token of `block_proj` that runs 1.51x below its own isolated rate. Nothing in this tree has priced that |
| Fuse the MoE half: one expert-id scan and one activation quantisation for all the expert gemms | ik_llama.cpp PR #229: +3-4 % CPU prefill; b11195 fused the row gather with the repack | Done, and done twice: the bucket is one counting sort (`moe_group`), and the stage graph stopped filling the same panel twice for gate and up (§6.4) |
| The dilution caveat for MoE: effective per-expert batch is batch x active/total, so a tile sees ~10 rows per expert at a 512 window | llama.cpp #531 measured ~1.4x on Qwen3-30B-A3B and close to nothing on DeepSeek below ub 1024; DeepSeek-R1 pp512 60 → 83 tok/s, 117.6 at `-ub 4096` | Read as a warning about the first row above, not as its own item: it says the tile only pays if the window it tiles over is wide, which is the `QE_TILE`/`QE_CHUNK_MAX` question (§7.9 keeps the width sweep, and a 1024 cap models out at ~1 %) |
| Register blocking and cache blocking, measured separately from the dot | BLIS with three-level blocking and explicit prefetch (A at 192/768 bytes, B at 256/1024) reaches 98 % of oneDNN with a 6x16 int8 microkernel; T-MAC isolates tiling alone at +1.45x; ggml's x86 paths use tile blocking but no `__builtin_prefetch` at all | Half in, half measured dead. Blocking is what `QE_MOE_DOWN_BLOCK`, `QE_ROW_BLOCK` and the park caps are. Software prefetch is dead twice over (src/moe.c:19-23, src/main.c:140, 6-67 % regression), and `make moe`'s thread table says why: the routed pass reaches ~40 GB/s of expert bytes at 19 threads against a 76-79 GB/s ceiling and still gains from every thread it is given (40.3 GB/s at 19, 33.3 at 12, 24.8 at 8), so it is compute-bound and not latency-bound, and there is nothing for a prefetcher to hide |
| Hybrid cores: keep a stream on one class of core, and let prefill use a different thread count than decode | OpenVINO runs INT8 on P-cores alone once E/P >= 4 and never mixes classes inside a stream; llama.cpp has `-tb/--threads-batch` so prefill and decode are configured separately | **Open, and it is a configuration item rather than a kernel.** This host is 12 E / 8 P = 1.5, so mixing is right, and `QE_SLICE=capacity` is off and prices within 1 % of equal shares in `make moe`'s two slice columns; what does not exist is a thread count of its own for a chunked prefill. The dead ends near it are the parked caps and P/E-weighted *placement* (bandwidth-neutral, src/mem.c:124-127), not this |
| Balance the expert stage against routing skew: fuse expert gemm tasks and steal dynamically | KTransformers (expert-GEMM task fusion plus dynamic stealing for routing skew) | **Built - §7.16.** A deterministic weighted split was chosen over stealing because `make forward`'s trunk check is byte identity, and stealing makes the down pass's partials depend on which worker got there first |
| AVX-512 and AMX paths | xFasterTransformer and the oneDNN int8 paths; the 6x16 kernel above is ~1/3 of the AVX-512 `vpdpbusd` wall | Out of scope: AVX-512 is fused off on this host and a `-mavx512f` build SIGILLs (AGENTS.md). Noted only so the next reader does not go looking for it |
| Whole other runtimes | OpenVINO plus neural-speed, mlc-llm (reported slower on CPU), vLLM's CPU grouped gemm (no published numbers) | Out of scope because of what this repository is (§12) |

Two honesty notes that belong with a survey rather than with a measurement. First, the AVX2
int8 dot has a trap this tree has already paid for in another form: `_mm256_maddubs_epi16`
saturates at 32767, the usual fix is to xor one operand and subtract a `128 * sum(B)`
correction, and ik_llama measured the correction at -17 % and removed it by keeping both
operands signed. §7.13 already ships per-format signed/unsigned selection for the same
reason. Second, an instrument went soft while this was being written: bench_moe's int8
expert-walk check, which compares the grouped integer pass against the per-row float pass on
*synthetic* uniform activations, reads 7.8e-02 against its 2e-1 threshold on the default draw
and 2.3e+00 - a FAIL - when `QE_MOE_REPEATS=5` changes which routes the draw lands on, with
the row scale both on and off. That is the bench's threshold sitting against an unlucky tail
of a random distribution and not a regression from `QE_I8_RS`: the real-checkpoint yardsticks
are what gate the recode (`make forward`'s legs, 30 % and 35 % of the head's range at f25). It
is on the record here because a FAIL that appears only under a repeat count is exactly the
kind of thing this repository pretends it does not have.

**7.16 P0 Built, measured, and on: an expert stage shared by what a bucket costs instead of by
how many units it is, worth 1.00x on an even router and 1.5-3.2x on an uneven one.**

See the row above for where the idea came from; the thing that made it worth building is a
column of `make moe`'s new last table. Every leg this file had ever run draws its routes
uniformly, which is the friendliest histogram a router could deal: at a 512-row window each
expert gets about ten rows, so an equal share of the (bucket, output row) count is nearly an
equal share of the work, and the question never gets asked. It is not free. One walk of an
expert's weight rows costs about the same for one token row as for a chunk of them and then
about one more chunk per chunk past it (bench_walk's own T table: eight rows 1.11x of one,
sixteen 2.45x), so a bucket of forty is several buckets of two and, sliced by count, all of
them land on one thread.

The fix is `moe_unit_slice` in src/moe.c: charge each bucket `max(QE_MOE_CHUNK, rows)` per
output row, give each thread an equal share of the charge, and divide a bucket's output rows
in proportion. It is a function of the histogram alone, so a chunk splits identically every
run, which is why it is a weighted split and not the stealing the survey's source used.
`make moe` A/Bs the two on one draw with the sweep order rotated:

| zipf | rows | max bucket | by count | by cost | gain |
|---|---|---|---|---|---|
| 0.0 | 128 | 8 | 32.4 ms | 32.4 ms | 1.001x |
| 0.0 | 512 | 22 | 75.4 | 74.3 | 1.015x |
| 0.5 | 128 | 34 | 38.5 | 32.4 | 1.186x |
| 0.5 | 512 | 117 | 112.9 | 72.1 | 1.565x |
| 0.8 | 128 | 91 | 54.3 | 30.3 | 1.791x |
| 0.8 | 512 | 382 | 193.6 | 75.9 | 2.550x |
| 1.0 | 128 | 197 | 67.9 | 27.2 | 2.500x |
| 1.0 | 512 | 733 | 255.2 | 79.9 | 3.195x |

Read the first two rows as the cost of the arithmetic that computes the boundaries (0.1-1.5 %,
and it is the same call the balanced case pays) and the rest as the size of the bet: at every
skew past uniform the by-count column is *idle threads*, and at the skews a real router deals
the pass is 1.5-3.2x its own best case. What this engine's router actually deals is the open
question, and it is the only number in this section that `make moe` cannot answer: the real
histogram is in a checkpoint load, which is what `make forward` will say (`QE_MOE_BAL=0` is
the way back, and `make moe` keeps both legs).

*Level 2, built, priced, and left off.* `qe_cpu_weight` has a second level, reachable with
`QE_MOE_BAL=2`, that charges a unit placed on a performance core 1.15 and one on an efficiency
core 0.89 instead of charging both 1, on the theory that a slice should be equal in elapsed
time rather than in work. It had never been run. Compiled and A/B'd against level 1 on one
draw, unpinned and then pinned (`QE_PIN=1`), `./bin/bench_moe 420 auto`; ms, gains against
by count:

| zipf | rows | by count | by cost | cost+class | by cost | cost+class |
|---|---|---|---|---|---|---|
| 0.0 | 128 | 32.3 | 32.4 | 33.5 | 0.996x | 0.964x |
| 0.0 | 512 | 75.1 | 73.8 | 77.7 | 1.017x | 0.967x |
| 0.5 | 128 | 37.9 | 30.4 | 35.7 | 1.247x | 1.063x |
| 0.5 | 512 | 114.2 | 72.1 | 73.9 | 1.584x | 1.547x |
| 0.8 | 128 | 51.9 | 30.4 | 33.7 | 1.709x | 1.540x |
| 0.8 | 512 | 193.3 | 75.1 | 84.5 | 2.572x | 2.287x |
| 1.0 | 128 | 63.8 | 26.3 | 29.2 | 2.424x | 2.182x |
| 1.0 | 512 | 253.4 | 80.4 | 86.8 | 3.153x | 2.919x |

The class weighting is worse than the cost weighting at every row of the skew range, and it
costs 3.6 % on an even router, where there is no skew for it to correct and the weights are
simply inventing one: 1.15/0.89 says a performance core may take 1.29 of an efficiency core's
slice. Pinning does not rescue it (0.967x and 0.974x at zipf 0.0, 3.089x against level 1's
3.143x at 512 rows and zipf 1.0). The reason is what the two capacity numbers measure: 1011
against 783 is *bandwidth* headroom, and these walks are not bandwidth-bound with nineteen
threads co-running (§7.10's finding), so the capacity ratio is not the ratio of the times a
unit actually takes, and a slice cut to it is cut against the grain. `QE_MOE_BAL` stays 1;
level 2 stays reachable for whoever next wants to price it against a walk that is capacity
bound. One caveat about these two logs: their `imbal` column read -0.05x to -0.41x because it
divided by the auto thread-count sentinel instead of by the thread count, since fixed in
bench_moe; the ms columns are unaffected. **Read since, and it behaves.** Over the eight rows
of `/tmp/test_flat.log` it is 1.01x at 128 rows and zipf 0.0 and 7.79x at 512 rows and zipf
1.0 - heaviest thread's share of the bucket charge over an equal share, which is what
count-slicing leaves on the table - and the by-cost gain over count-slicing in the same rows
rises with it: 1.051x at 1.07x, 1.158x at 1.43x, 2.152x at 3.30x, 3.694x at 6.08x, 3.086x at
7.79x. Nothing in the paragraph above changes: level 1 is still the level that takes it, and
the predictor now says so with a number that is not negative.

**7.17 P0 Built, measured, and on: a Q8_0 walk's row scale belongs outside its token loop and
its dot is signed - 1.88x in a cell, 0.48x of `block_proj`, and an engine at 55-58 tok/s against
llama.cpp's 51.33.**

The row-scaled panel of §7.14 took the activation row's scale out of the *codebook* walks' token
loop and left the Q8_0 walk reading it per 32 weights, because a row-scaled panel is also a valid
per-block panel and the Q8_0 exit had been written against the block. Two operands per token per
32 weights therefore stayed in that loop - a convert of the block's int32 sum and an FMA against
the block's scale - on the one format whose exit is paid every 32 weights rather than every 256,
which is eight times as often. `QE_I8_HOIST` moves that multiply to after the row's block loop,
where one FMA pays the row's scale once; `qe_i8_ssd_q8` (`src/kern.c`) gives the Q8_0 dot its own
read of `QE_I8_SSD`, so an A/B of this walk no longer has to drag the codebook walks with it, and
`vpdpbssd` is what the Q8_0 walk runs by default. Neither is a new idea from this tree - §7.10
priced the exit's operands by moving them - but this is the first form of the Q8_0 walk that is
not paying for a scale it does not need per 32 weights.

*The cell, and it is the largest one this file has printed.* `make walk`'s last two tables
(`/tmp/walk_flat2.log`, 100000 rows × 2560 of Q8_0 against one row-scaled panel): hoisted against
per-block is **1.17x / 1.25x / 1.42x / 1.35x** with the unsigned dot at 8, 16, 64 and 512 token
rows and **1.19x / 1.59x / 1.90x / 1.88x** with the signed one, 572 GMAC/s against 773 at the
shipping width on a load average of 11; and the row scale itself, on the same run, is **1.89x** of
the per-block panel at 512 rows (238.2 ms against 126.1) where the same leg measures 1.28x on
IQ4_NL and 1.20x on IQ3_S. Q8_0 is the format where the exit is cheapest per weight and the most
often paid, and the two now agree.

*One measured exception, and it is a foot-gun.* The signed dot is a win **on a row-scaled panel**.
On a per-block panel it loses, and not by noise: 169.2 ms unsigned against 237.0 ms signed at
T=512, 0.71x, because the unsigned walk's +128 bias arrives as the accumulator `vpdpbusd` was
handed and paying it back per 32 weights is exactly the operand the signed walk saves only when
the scale it folds is the row's. So `QE_I8_RS=0` wants `QE_I8_SSD=0` with it. The codebook column
of the same table is what a 0.7 % row is: IQ3_S read 919.4 ms unsigned against 925.7 signed at
T=512 in a row where IQ4_NL read 1.10x the other way, on a leg where the signed walk executes a
strict subset of the operands, so that bound is now 1.00x below the widest leg and 0.98x at it,
with the row that made it fail in the comment above it.

*The engine, four loads of it and a fifth to confirm.* `make forward` at the width the sweep ships:
**21.9 ms/token at f26, 18.2 at f27, 17.4 at f28, 17.3 at f29, 18.2 at f30** - 45.7, 54.9, 57.5,
57.8, 55.0 tokens a second. The spread of the last four is 5 %, it is the machine and not the
change, and the honest statement of where this leaves the engine is **55 to 58 tok/s, 1.07-1.13x
ahead of llama.cpp b12359's 51.33** where f26 was 1.12x behind it. The projection leg's own A/B is
**4.95, 2.98, 2.94, 2.92, 2.95 ms/token**, and its reading of why is that the integer
projection walk is **0.484x of the dequantising walk** (f30: projections 2.95 against 6.06, whole
step 15.93 against 20.53). The expert stage did not move and should not have: 7.27 ms/token at f30,
0.907x of the panel it replaces and 0.727x of the float walk, because the routed experts in this
checkpoint are IQ3_S gate/up and IQ4_NL down with no Q8_0 in them. Reverts are `QE_I8_HOIST=0` and
`QE_I8_SSD=0`.

*The quality bound this tripped, and the control that settled it.* f27, f28 and f29 each failed
"the int8 expert walk stays inside half the head's range" at **1.1e+01 apart of a 15.9 scale,
66 %**, each with the same next token, while the same metric on the same pair had read 21 %, 17 %,
30 %, 30 %, 30 % and 31 % on the six loads before it. Every failing pair differs only in the expert
arithmetic, and the routed experts in this checkpoint contain no Q8_0 - `make report`: IQ4_NL 52 %,
IQ3_S 36 %, Q8_0 10 %, which is the dense projections, IQ4_XS 1 %, Q6_K 1 % - so what moved was the
trunk the pair sits on, and this repository does not settle that by moving a threshold. It has the
instrument that prices it instead. `== the Q8_0 exits, one arithmetic ==` pins
`qe_i8_set_hoist` × `qe_i8_set_ssd_q8` inside one load - four arms whose expert stage is one
another's bit for bit - and f29 read them **6.6, 8.1 and 9.8 logit units apart**, one of those
three differing from its reference only in where a row's float sum associated, which is rounding
and nothing anyone would call a regression. The expert pairs beside it are in that same band: 8.3
units across the head's top 32 for the recode on a panel the defaults do not run, 8.4 for the
recode as it ships, 9.8 for the row scale's own increment. What made one of them a failure and
none of the others a finding is the denominator - the largest logit of the head this prompt draws
is 15.9 prefilled one way and 24.3 the other, a 53 % swing on the number every one of those
percentages is divided by. `gate_heads` therefore holds what each arm is accountable for: the same
half-the-head's-range bound, measured across the coordinates that can be chosen, on the arms the
defaults run (f29: **34 %** shipping, **41 %** for the row scale's increment), and **one nat** on
the token the engine goes on to say for every pair, gated or not - 0.089, 0.46 and 0.55 nats at
f29, against the 3.56 nats/token the reference engine scores on its own text. The whole-head
maximum, the rank of the entry that sets it, the shared top-32 and the next token are printed for
all three. As for what the old instrument was worth: it passed this pair six loads running at
17-31 % and failed it three at 66 %, on a change none of those loads made.

**7.18 P0 Built and measured at 1.84x of the walk that ships: one scale per output *row* of a
matrix turns the Q8_0 exit into nothing, and what it costs is a copy.**

Four operations per token per 32 weights is what §7.17 leaves in a Q8_0 walk: a load, a dot, a
convert, and the FMA that folds the block's scale. Two of the four are there because the scale is
the *block's*. Take a copy of the matrix with one scale per row and the pair disappears - the
accumulator runs in int32 the whole row, becomes a float once, and pays the row's scale and the
token's - and the register the block scale held pays for a wider tile. The operand count per
output row per token by tile is 2.125 at one row and eight tokens, 1.75 at two and four, **1.583
at three rows and four tokens**, which is twelve accumulators plus three weight registers plus one
activation register: the 16 ymm this file has, exactly. `vpdpbusd` cannot be flat at all, because
its bias seed is per block; the flat walk is signed-only. An int32 accumulator over a row is safe
to 131072 values per row, which at K = 2560 is fifty rows of margin.

`qe_gemv_q8_0_flat_i8` (`src/kern.c`) is built - eight tile forms, a table indexed by rows-per-tile
and token tile, a row tail walked with the one-row forms - and `make walk` prices it against the
shipping walk on 100000 rows × 2560 with one row-scaled panel
(`/tmp/walk_flat2.log`, gains of hoist+ssd over the flat walk, columns are 1, 2, 3 and 4 rows per
tile):

| T | hoist+ssd ms | flat r1 | r2 | **r3** | r4 |
|---|---|---|---|---|---|
| 8 | 3.7 | 1.11x | 1.08x | 1.09x | 1.06x |
| 16 | 5.1 | 1.19x | 1.24x | **1.36x** | 1.24x |
| 64 | 15.6 | 1.20x | 1.44x | **1.68x** | 1.39x |
| 512 | 138.7 | 1.26x | 1.45x | **1.84x** | 1.45x |

Three rows per tile is the fastest at every width above eight tokens, at **1737 GMAC/s** at the
width a chunk is ingested at - against the 773 GMAC/s §7.17 measured for the walk that ships in
the same run, and against the 572 it replaces. The exactness check is bit-identical (worst diff 0
of 1709 at 4103 rows × 8 tokens, for both the three-row and the one-row tile) because this file's
fills give every block the same scale, which is also the reason the check is exact rather than
close: the copy this leg walks is the same int8 matrix, de-interleaved.

*What it costs, and it is built.* A copy of the dense matrices requantised to one scale
per row, made at load time: `flat_stage()` (`src/forward.c`) is called from the open,
after the engine's own allocations so that the memory it reads as free is the memory a
step will not need, and stages two arenas - flat rows in one, a float scale per row in the
other - behind a per-tensor offset map, refusing with two printed numbers when the copies
plus a 4 GiB margin do not fit. On this checkpoint that is 496 dense matrices and 3.3 GiB,
the Q8_0 tensors less the routed experts and the embedding table, and the open says which
answer it came to: "flat int8 walk on: 496 dense matrices (3422 MiB) requantised to one
scale per output row, 28877 MiB free at open" (`/tmp/f35_wide.log`). `gemv_chunk_task`
calls `qe_gemv_q8_0_flat_i8` for the tensors `flat_candidate` selects, at the three-row
tile, and `QE_I8_FLAT=0` is the way back. The `make forward` leg this section asked for is
what carried it:

    the flat dense walk is 0.610x of the hoisted one: projections 1.78 against 2.91,
    whole step 14.92 against 16.00 ms/token          (`/tmp/f35_wide.log`)

0.610x of §7.17's walk and 0.297x of the dequantising walk it first replaced (1.78 against
5.99 ms/token, whole step 14.93 against 20.69), and a width-512 sweep of **16.3 ms/token**
against the 17.3 this file quoted - **61.3 tok/s**, with the four clean loads of the five
ranging 15.9 to 16.3 (`/tmp/f31_wide.log` through `/tmp/f35_wide.log`; `/tmp/f34_wide.log`
read 16.7 and carries the rewind failure §7.19 records). Both unknowns are answered by
measurement rather than by argument: a row-wide weight scale moves the head's answer to the
prompt that leg walks by 2.5 logits, 10 % of that head's own range of 24.3, worst entry
rank 14 of 248320, and 6.9e-04 nats on the token the engine goes on to say, which the leg
holds to one nat; and the copy fits with 28.9 GiB free over an engine holding 87.2 GiB.

What that leaves is the same shape of gap §7.13 named for the experts, in a form this file
has now measured three times over: 1.78 ms/token of projections is not a byte story at
width 512, the tile is no longer the binding constraint (three-row int8 is the third form,
and the two-row and ten-row forms both failed to earn their registers, §7.12, §7.14), and
the copy this walk reads is already the permutation §2's row 20 asked for. So the item
left is a wider tile of the same copy, and §7.15 is where the survey numbers for it are.

**7.19 P0 Built, wired, measured, and switched off: a draft row is 0.130 of a step and a
verified window is 4.14 of one.**

Nothing in this tree could run a real draft head until two bugs were gone, and both were in
the path that opens one. `mtp_layer_open`'s `want()` chain was inverted, so the head
declined to open with "the draft head has no nextn block to open" against a file whose first
tensor is `blk.48.nextn.*`; and the HC "views" that `mtp_ehv` built over the head's
`eh_proj` were bogus tensors, so the fold segfaulted (RC=139) rather than folding. The fold
now calls `proj()` four times over the whole `eh_proj` at a stride of `HC*(2K)`, which is
what the checkpoint's own layout says, and the shape check compares the head's layers
shape-for-shape against the trunk's last `qsa` layer instead of against a table someone
maintained by hand. With those fixed the head opens, drafts 32 ids, and does not move one
of the ids greedy says - no non-finite values, no cache refusals, the trunk left byte for
byte where a run without `--mtp` left it (`/tmp/f38_mtp_short.log`).

The head is not a second engine: it is `f->L[f->n_layer]`, run through the trunk's own six
stage functions, which is why a draft row costs what one layer costs. What it costs, at
23-token context, measured four passes in one load:

| pass | ms | of a step | `block_proj` | `block_core` | `moe_ffn` |
|---|---|---|---|---|---|
| one target step | 111.2 | 1.00x | 44.96 | 3.46 | 34.46 |
| one draft row | 14.5 | 0.130x | - | - | - |
| nine-row window, logits for every row | 460.3 | 4.14x | 56.47 | 106.03 | 172.64 |
| nine tokens, boundaries not kept | 324.3 | 2.92x | - | 26.44 | - |

That table is the whole argument, and it kills the one premise §4.2 and `docs/DESIGN.md` §4
were built on - that a verified row is free because a window shares the model's bytes. The
dense walk does share them (44.96 to 56.47 ms for nine rows, the one stage that behaves).
Everything else pays per row: `block_core` is 106.03 ms of the window and ~80 ms of that is
copying GDN state into checkpoint slots, which is what makes a window rewindable rather than
merely wide, and dropping the boundaries takes 80 ms of it back and takes the speculation
with it; `moe_ffn` is 172.64 against 34.46 because nine rows route to nine different sets of
ten experts and grouping takes back less than a half of that at nine rows; and ~97 ms of the
window is not billed to a stage at all, because computing logits for every row means the
rejected rows are paid for at ~7.2 ms each.

Acceptance is the other half and it does not reach the vendor's number. Teacher-forced over
16 positions the head's top-1 agrees with the trunk's on 8, and the chained acceptance the
engine actually gets is 52-60 % for the first draft and ~18 % by the third; the histogram at
k = 8 is 41 % of passes keeping one token, 24 % two, 18 % three, 6 % four and none five or
more. `cost = (window(k+1) + k x 14.5 ms) / ids(k)` with those two curves has no crossing:

| arm | ms per emitted id | against the plain loop |
|---|---|---|
| plain step loop | 105.7 | 1.00x |
| head, k = 1, as shipped | 102.8 | **1.03x** |
| head, k = 2, exact | 178.3 | 0.59x |
| head, k = 4, exact | 252.1 | 0.42x |
| head, k = 8, exact | 459.8 | 0.23x |
| head, k = 8, as shipped (warm) | 322.1 | 0.33x |

Two secondary readings from the same load. The pairing the drafter seeds from is the
residual one position back, not the frontier's own: teacher-forced, back = 1 agrees on 8 of
16 positions and 17 of 32 shared ids against 6 and 18 for back = 0, and chained drafts seeded
from the frontier drafted **zero** tokens on the first pass, because a draft is made before
the position whose residual it would read. And warming the head's rows inside the chunk buys
nothing at short context (1:56 % of passes keeping one token unwarm'd against 1:56-60 %
warm), so `QE_MTP_WARM` is ingestion cost bought for an acceptance that did not move.

So speculation ships at the shortest draft its drafter can pay for, the length being resolved at
open rather than being a constant (`QE_SPEC_K_AUTO`): one token with a nextn head resident, where
as-shipped it measures **1.19x** the plain loop, two with the n-gram hash, where it measures
**1.05x**, and `--spec-k 0` for neither (`/tmp/f40_head.log`). Both of those numbers were a slower
step's: with §7.23's decode copies shipping, the same two arms are **1.06x** and **1.05x** of a
plain loop that is 91.6 ms an id instead of 110, k=8 is 0.99x for either drafter, a draft row is
0.129 of a step and a nine-row window 4.35x one, and the 1.06x only exists because a verify window
now reads the head's copy as a step does - without it k=1 is 1.02x (§7.23, `/tmp/head_e.log`).
`--mtp --spec-k N` is the way back
in for anyone who wants the measurement repeated on their own bytes - `QE_MTP_WARM=0` and
`QE_MTP_BACK=0` being the two knobs on how a draft row is made. What would change the answer
is measured but not enough to be worth this file's time: computing logits lazily for the rows
a window keeps is worth ~50 ms of a 460 ms pass and cheaper boundary copies ~80 ms of it,
together ~1.1x on a path that is 4.14x away from breaking even, against the routed experts
sitting at 32-39 GB/s in the 105 ms step that ships. One thing this section does leave
unpaid is a leg that has gone green four times in five: "a rewind to any of a 5-row
window's boundaries leaves the state the steps it abandoned left" passes byte for byte
under exact arithmetic and, under the shipping integer walks, where a boundary may differ
only inside the gap that arithmetic itself creates, flipped one boundary's choice on
`/tmp/f34_wide.log` (3 of 4 boundaries chose the abandoned step's token, the differing row
was logits-only at id 11 against 11) and passed on the four loads around it. That reads as
a near-tie decided by reassociation rather than as lost state, since the exact leg is
identical to the bit in the same load, but it is the one assertion in `make forward` that
is not 5/5, and speculation is the consumer that would notice it first.

**7.20 P0 Measured and left open: the q6_K walk the engine calls is 1.31x off the same walk
compiled in the next file, and that walk is the output matrix of a decode step.**

`bench/q6probe.c` is the instrument - 384 MiB of q6_K rows of K = 2560 (191739 rows of 2100
bytes), six walks of it in one process, the engine's own `qe_gemv_q6_k` as the sixth, a stride-one
read of the same bytes as the floor, and a rotation argument (`argv[4]`, `make`-less:
`./bin/q6probe <fuse> <threads> <xmode> <first-form>`) so that a form timed last cannot be a form
timed slow. At 19 threads the floor is 79-81 GB/s, and the tree's dot - `qe_q6_run` and
`qe_dot_q6_k_avx2`, byte for byte the source `src/kern_iq.c` carries, compiled into the probe -
reads **76.5 GB/s**, which is 6.8 ms for the 497.3 MiB output matrix. What the engine pays for the
same bytes through the same arithmetic is **58.5** (`/tmp` runs of that probe, spread 58.0-58.9 over
five loads, against 75.4-76.5 for the in-file copy): **0.767x**, and it is not a bandwidth story at
any width - one thread reads 10.8 against 13.9, four 22.0 against 24.4. The two walks agree to the
bit (`rows differing: 0` of 191739), so this is code generation and nothing else.

Four things were ruled out with measurement rather than with argument. **Order**: rotating the forms
moves the number not at all (58.5 measured first, 58.5 measured last). **Inlining**: `always_inline`
on the run and on the dot together removes the per-row call and the stack realignment, and the
number is 58.0 (reverted, being worth nothing). **Alignment**: building the whole tree with
`-falign-loops=32 -falign-functions=32` leaves `make fast`'s Q6_K line where it was (47.0 → 46.8 GB/s
at the full width) but moves its four-thread column 22.0 → 27.3, which is well outside that run's
spread - so the penalty is alignment-shaped and it moves with code that has nothing to do with the
walk, which is exactly why a fix has to be read off the engine's own legs and not off this probe.
Read off the engine it is worth nothing: over `make forward --fuse 1500` those flags ship the same
16.3 ms/token at the 512-row width the default build ships, and cost 1-2 % at every width between 5
and 128 (17.5 against 17.3 at 128, 34.2 against 33.8 at 8, `/tmp/f49_wide_align.log` against
`/tmp/f35_wide.log`), so they are off again and the 0.767x is still standing.
**The harness**: `make fast` bills the same entry point at 47.0 GB/s and 11.1 ms, 9 % of a step,
against this probe's 58.5, so the two instruments disagree about one kernel by 24 % and the 9 %
above is probably the pessimistic one of the two.

What did not earn its keep, all of it at the same bytes and all of it *wrong* in the same way -
`d1` through `d4` index `sf[j]` where the tree's run is handed `sc + g*8 + j`, so the second half of
every 256-weight block is scaled by the first half's sub-scales - which is why they are speed probes
only and their accuracy columns are meaningless: hoisting the sixteen sub-scales to a float vector
(0.977-0.986x), the two-op placement of the 2-bit field so a run is an `or` and a `sub`
(0.953-0.966x), returning the -32 through the int16 widen and paying it back as an x sum at the fold
(0.921-0.939x), and the int8 `vpdpbusd` panel (0.992-1.006x at 19 threads). That last one is the
interesting failure: **1.54x at one thread and 1.79x at four** against the shipping dot, going flat
only where the bytes run out. It is §7.18's row-scaled tile arriving at q6_K, and the panel it wants
already exists as `qe_i8panel` - `qe_gemv_i8_t` simply does not answer for `QE_Q6_K`, and neither
does `qe_gemv_multi_ptrs_has`, so the format the output matrix is stored in is the one format with
neither an integer walk nor a form that reads a weight row once for several token rows.

Why it is a decode item and not an ingestion one: a chunk spends the output matrix on the prompt's
last token, so 9 ms is 0.1 % of an ingested token, while a decode step spends it on every token and
`make fast` calls it 9 % of the step. Closing the code-generation gap is 8.9 → 6.8 ms, about 2 % of
the 105 ms a token costs. The bigger half is the one nobody has built: `qe_forward_row_logits` walks
that matrix once per row of a verified window - which is what §7.19's ~7.2 ms per rejected row is
made of - so a form that reads a q6_K row once for the two or three rows of a window is worth
another (R-1) × 9 ms per speculative pass, and that is the only way the k = 2 and k = 3 arms of
§7.19 get anywhere near the break-even they miss by 4x. Both halves are gated by the same missing
kernel; neither is gated by the format.

## 7.21 Measured and left at eight: a codebook walk can hold sixteen token rows in one pass, and the routed pass is the reason that is not what ships.

§7.2's retirement note is the claim this went after: past eight rows of a bucket "the accumulators
no longer seat in the register file, the group splits, and the tail pays for the dequantisation
again", and a 512-token window routes 5120 picks over 512 experts, so the modal bucket of ten was
paying two walks of an expert's rows and the ninth token cost 8.7x what the tenth cost. §7.2b
settled the *park* - holding a wide group's activations is what cost L2 - and left the *form*:
nothing in the tree could walk nine to sixteen token rows in one pass. It can now. The macro grew a
width argument and answered with twelve templates (`QE_IQ4_NL_I8_FORM`/`QE_IQ3_S_I8_FORM` at ten,
twelve and sixteen token rows, in both dots and both scale exits), a group is divided by the fewest
passes a six-operands-a-block, three-a-token model calls cheapest rather than greedily
widest-first (a group of twenty-two is 12+10, not 16+4+2), and a leftover single token borrows the
token before it, which recomputes one token and writes the same number over it. `QE_I8_T` picks the
ceiling; eight still ships, and this is the measurement that keeps it there.

*On a tensor the size of a dense matrix the item is real, by less than it first looked.* `make walk`'s
sixth section is 4103 rows of each expert format at K = 2560 against a row-scaled panel with the walk
on the pair the engine runs (signed dot, one scale per row). Every new form holds to the eight-token
walk's bits - 0 of 131296 outputs moved at thirty-two tokens, 0 of 41030 at ten - and the four
ceilings then read against each other. Two loads of that section (`/tmp/f52_groups.log`,
`/tmp/f56_walk2.log`) bound the item, and at the modal bucket one pass replaced two: **1.05-1.11x on
IQ4_NL (14.8 → 13.3 ms, 17.3 → 16.5) and 1.32-1.34x on IQ3_S (16.7 → 12.4, 24.8 → 18.5)**. Eleven
rows read 1.05-1.11x and 1.23-1.35x, twelve and thirteen 1.04-1.08x and 1.19-1.25x, fourteen
1.03-1.04x and 1.15x. Sixteen to twenty-four rows is a rounding: 1.04-1.07x on IQ4_NL and 1.01-1.05x
on IQ3_S. At thirty-two rows the sixteen class loses on the second load, 0.95x and 0.94x, where the
first read 1.01x and 1.03x.

*What that instrument can see, which is the number worth keeping.* Eight and nine rows are the same
walk at every ceiling - there is no nine-row class, so a group of eight or nine is one eight pass and
a borrowed token whatever the ceiling allows - and a reading there that is not 1.00x is the
instrument. On `/tmp/f52_groups.log` those control rows are inside 0.01x; on `/tmp/f56_walk2.log`
they reach 1.07x, and its eight-token column is up to 1.4x the same column of the other load in
absolute ms (its nine-row IQ4 column reads 10.5 against 14.6, its thirty-two-row 42.0 against 59.1).
Three back-to-back ceiling-16 loads of the same binary's ssd table add the shape of the noise: the
eight-row leg reads 13.4, 13.2 and 18.4 ms with a load average of 0.25, so the host has a 1.4x mode on
top of run-to-run spread. The honest reading is therefore one gain and not a curve. At ten and eleven
rows, where a wide pass replaces two, the item clears both loads' control rows. Past sixteen rows it
clears neither, and nothing in the routing has eleven-token buckets. Sixteen is not the loser the
second load made it look like either: the ten class is 0.92-0.96x at twelve, thirteen, twenty-four and
thirty-two rows and the twelve class 0.95x at sixteen, so a ceiling below sixteen is the worse
compromise at the widths between the classes. Spilling was the forecast for not building these and
does not stop them: the row-scaled IQ4 form goes 29 → 39 → 50 → 78 frame stores per 325 → 401 → 487 →
662 instructions from eight rows to sixteen and is the fastest of the four at ten rows on both loads,
because a spill is paid once a pass and the pass is what the codebook decode costs.

*In the pass that walks it, it is nothing.* `make moe`'s shared-dequant column - one bucket, every
row on the same experts, the shape a wide pass is for - reads the same two configurations at every
row count from two to 512 as 0.715 against 0.729 of the f32 walk at eight rows, 0.712 against 0.704
at sixteen, 0.730 against 0.706 at thirty-two, 0.691 against 0.712 at sixty-four, 0.716 against
0.715 at 128, 0.697 against 0.696 at 256, 0.685 against 0.678 at 512 (`/tmp/f50_test.log` at the
eight-token ceiling, `/tmp/f53_test.log` at sixteen). Nothing moves. A checkpoint load says the same
with the biggest instrument in the repository: the row-scaled expert walk is **7.20 ms/token against
the 7.14** the eight-token family got, the whole step 15.06 against 14.93, and the grouped pass 25.3
against 25.2 at the shipping width (`/tmp/f54_wide.log`, `/tmp/f35_wide.log`) - while the two stages
that this change cannot touch move by as much or more in the same pair of loads (block_proj 1.81
against 1.78, block_core 4.04 against 3.98), and the drift-free ratios inside each load agree to the
third decimal: 0.896x and 0.712x against 0.891x and 0.715x.

*Why the two instruments disagree, which is the finding.* An expert's gate rows are 688 KiB, which
seats in an L2, so the second pass of a bucket is an **L2 re-read** - and the 400000-row tensor of
`make walk` is 0.5 GiB, where the second pass is a DRAM re-read at the 40 GB/s that tensor streams
at. The cost the item prices is one cache miss per row of a bucket, and the routed pass does not
have it. That is the same distinction §7.15's dilution row makes about a different lever, and it is
worth writing as a rule: anything priced off "a bucket of ten pays its walk twice" has to be priced
against the slice a bucket walks, and a slice under the L2 pays its second pass in L2 cycles.

*What the wide forms do cost, and what the measurement cost.* The signed dot's edge over the
unsigned one - the whole reason `QE_I8_SSD` is on for the codebook walks - closes at exactly the
widths the new classes are for. Four loads of `make walk`'s ssd table under the sixteen ceiling, the
first on a hammered host (`/tmp/f55_tile_t16.log`) and three run for this paragraph on an idle one
(`/tmp/f57_t16_1.log` to `/tmp/f57_t16_3.log`), lost a signed codebook leg in every one of them: at
eleven rows in three of them, at 0.97x, 0.99x and 0.99x of the unsigned, and at sixteen rows in the
fourth, 0.89x on IQ4_NL and 0.93x on IQ3_S. The same table under the eight-token ceiling reads
1.04-1.06x at eleven rows and 1.09-1.15x at eight across three loads (`/tmp/f51_walk_base.log`,
`/tmp/f55_tile_t8.log`, `/tmp/f56_walk2.log`), so what the wide forms cost the signed dot is the
widths they exist for and not the dot. The engine does not choose a dot at run time, so this is not a
regression; it is the assertion that keeps the ceiling where it is, and it says the register pressure
a sixteen-row pass buys its spill with is not free in the exit either. Two instruments were wrong before any of this could be read. A leftover token fell
through the plan table - a group of one has no first class, by construction - and the walk returned
without writing that token, which `make walk`'s exactness column could not see because the reference
walk dropped the same token, and which `make fast`'s new bitwise ceiling gate caught as 560
coordinates still holding the 12345 sentinel it was filled with. And a ceiling above eight switches
the two-row tile off, the tile being eight rows by construction, so `make fast`'s tile A/B had
silently become the same walk twice and its `qe_i8_rows(2)` read left two rows pinned for the rest
of the process - neither the shipped default nor what the check was comparing. Both gates now pin
the knob they are not testing, and the plan table is one row per ceiling, prefilled with the
narrowest class, so a walk that arrives while a build is in flight divides a group badly and stays
inside the panel it was handed.

## 7.22 A decode step had no form to walk its experts with: the class family stops at two tokens, and a step hands it buckets of one.

The class family of §7.21 has a floor as well as a ceiling. Its floor is two, and the floor is
right: a leftover token takes the two-token form with the token before it, because six operands of
decode against nine is the cheaper way to pay for a pass, and the plan table is a cheapest-division
table that says so at every width. It is right for every group the routing produces and wrong for
the one group a decode step is. A step routes one token over top-k experts, which fills k buckets of
exactly one row; `gate_up_multi`'s bucket test was `last - first >= 2`, so every one of them fell
through to the walk that converts a codebook value to a float before multiplying it. `make fast`
prints that walk's rate as the decode budget: IQ3_S at 44.4 GB/s and IQ4_NL at 59.5 at twenty
threads, against the 79.6 the same table reads for Q8_0 and the 76-79 the shape reads at all
(`make alu`). Those two rows are 18 % of a step and they are 40 % short of a ceiling the
byte-streaming half of the same stage reaches, and what says they are is not the codebook but the
four converts and four multiplies per 32 weights that every grouped pass spends as one `vpdpbssd`.

The family has a one-token form now: one width, two codebooks, both dots and both scale exits, held
at `QE_I8_ONE` outside the class enum so the plan cannot reach it - no division of a group of two or
more wants it, and a division that could reach it would find it cheaper than the borrow only
sometimes. A walk handed `T == 1` takes it for every row of the matrix, and the row helper's
leftover branch takes it in the single case where there is no neighbour to borrow, which until now
was a case the dispatcher refused instead of one that had a form. `qe_gemv_i8_has_1` is the question
for a caller that has to decide before it writes outputs, and the answer for Q8_0 is no: the
narrowest Q8_0 tile is a pair whose partner this caller never passed, and reading `ys[1]` is an
out-of-bounds write and not a slower answer. `gate_up_multi` and the down path therefore take the
integer walk at a bucket of one wherever the format has the form. The arithmetic is not new to the
engine - it is the same panel, the same recode and the same exit the grouped expert pass already
ships with, and src/moe.c carries that walk's recorded price; what changes is that a step reaches
it instead of reaching the float walk.

What prices it, honestly. `make fast` holds the new width to everything the other widths are held
to - the same accuracy reference, the same per-block exit against the row-scaled one, the same
bitwise ceiling A/B - over 455 IQ4_NL and 554 IQ3_S shapes where it held 448 and 544, the
difference being the new width at every K and row count. The rate is a different instrument and it
is not in this section: the change is invisible to `make forward`, whose every number is a chunk of
sixteen tokens or more, and it is a decode-only change by construction, because a chunk of two or
more takes the classes it took before. `make head`'s ms/id is the measurement to read, with
`QE_I8_MOE=0` as the arm; §7.10's 1.36x for an integer dot whose exit sits every 32 weights is the
shape of the claim and not the claim, because it was measured on eight token rows and a step buys
the multiply saving with nothing to share the load with. What makes it worth shipping on the
mechanism rather than on a leg is the cost side: the only thing a one-token walk spends that the
float walk does not is one panel row of K bytes per expert, recoded once, against a matrix of
thousands of rows read at 44 GB/s.

## 7.23 Two thirds of a decode step is one format, and it is not the one the file wrote.

§4.1's table is the whole design brief, and it was written before any of this was buildable: the
same projection walked out of a Q8_0 row and out of an IQ4_NL row is **1.74x** apart at T=1 and
0.96-1.02x apart at the width a chunk ingests at. A smaller copy of a dense matrix is therefore
worth making exactly when the matrix is walked once, and a step is the only time that happens.
What was missing was never the kernel - `make fast` has held `qe_gemv_iq4_nl` correct against a
double reference for its whole life - but the three things around it: an encoder that is
provably the same encoder the format is scored on, a load-time pass that stages the copy or
refuses with two numbers, and a width test in the walk so that an ingested token never reads a
byte it did not read before.

Two copies of the dense projections and two of the head are staged at open, and only at open:
nothing about them is lazy, and an open that cannot afford them says so and goes on walking the
checkpoint's own bytes (AGENTS.md 1). What this checkpoint prints at open, never timed
(AGENTS.md 2): 496 dense matrices, **3625 MiB of Q8_0 requantised to 1921 MiB of codebook rows**,
29049 MiB free; and `output.weight`'s 248320 rows of 2560 - **497 MiB of Q6_K** - twice more, as
**341 MiB of IQ4_NL and 644 MiB of Q8_0**, 27162 MiB free. `QE_I4_DENSE=0` is the way back off the
first, `QE_I4_HEAD=stored|i4|q8` names which of the second a step reads, and both are read at open
into fields, so `qe_forward_get_decode_i4` answering 0 and `qe_forward_get_head_copy` answering
STORED is how a caller tells "the switch changed nothing" from "there was nothing for it to
change".

The encoders are held to bit equality and not to a tolerance, because a copy that is *nearly* the
encoder for its format is a copy of a model nobody scored. `make fast` prints the two lines:
`iq4_nl from Q8_0 : pass (0 of 600 blocks differ from the float path)` and the new
`q6_K requantise : pass (0 of 120 rows off the float path for IQ4_NL, 0 for Q8_0; worst dot
0.0e+00 rel; block rms worst 0.1094 IQ4_NL, 0.0074 Q8_0)`. The dot term of the second is not
about the encoder at all and is the reason the check exists: a Q6_K row has to be *rebuilt* before
anything can be encoded, and a super-block scale read at the wrong stride or a nibble from the
wrong half of a block hands the encoder a plausible row that is not the one the walk would have
multiplied. The check therefore dots the rebuilt row with a random activation and requires
`qe_dot_q6_k_ref`'s answer to 1e-12 relative; it came back at 0.0e+00, which is what an identical
sum in the same order is worth.

What they buy and what they cost, one load, 1499 teacher-forced positions (`/tmp/i4f_score.log`;
the dense leg reproduced at 1.16x and +0.0752 nats on 2499 positions of the same text in a second
load, `/tmp/i4e_score.log`):

| what a step reads | nll | ms/token | vs stored | greedy choices moved |
|---|---|---|---|---|
| as stored | 2.3781 | 108.1 | - | - |
| IQ4_NL dense rows | 2.4405 | 93.1 | **1.16x** | 398 of 1499 |
| IQ4_NL head | 2.3989 | 103.6 | 1.04x | 192 of 1499 |
| **Q8_0 head** | **2.3783** | **106.3** | **1.02x** | **13 of 1499** |
| both IQ4_NL | 2.4610 | 87.6 | 1.23x | 446 of 1499 |

Two things that table cannot say by itself. The corpus is this file's own prose, so its absolute
nll is a statement about the text and not about the model; what the five arms buy is that they and
the control saw the same positions in the same load. And the control is the table immediately
above it in the same log: the fused attention walk, which ships and differs from its alternative
by a reordering, moves **100 of the same 1499** greedy choices for 0.0027 nats. Read against that
floor the Q8_0 head is *inside* the noise - 13 choices, 0.0003 nats - and it is what ships on by
default. The dense leg is a quarter of the choices, and it is not noise.

The Q8_0 head is the counter-intuitive leg, and it is §0's warning with the sign flipped rather
than the ratio wrong: the copy is 30 % **bigger** than the format it replaces, 644 MiB against 497,
and it is 1.02x of a step, because this host walks a Q6_K row at 44-46 GB/s and a Q8_0 row at
73.6-78.7 (§1.2). Bytes up, time down - 2 ms of a step's 108, which is the head's whole share at
those two rates. The IQ4_NL head is faster again, 4.5 ms of the same step (341 MiB at 59 GB/s
against 497 at 45 is where the arithmetic lives), and it costs 0.021 nats and 192 choices, so it
stays behind `QE_I4_HEAD=i4` rather than shipping.

The dense leg ships at 1.16x of a step for 0.0625 nats, which is honest and not cheap: it is
twenty times what this repository paid for the q8_0 KV cache (§5's 0.294 against 0.292) and it
moves one greedy choice in four. `make score`'s decode arms exist so that price is read rather
than argued, and §4.1's two refinements - hold layer 0 and the last few layers at Q8_0, hold
`hc_*` at Q8_0 - are the way to buy it back rather than to take the leg or leave it; `QE_I4_DENSE=0`
is the whole of the way back in the meantime, and it costs 1.16x of a step and nothing else.

The two refinements §4.1 named are a dial now rather than a wish. `QE_I4_DENSE` reads `all`,
`lean`, `nohc`, `noedge` or `0`. A class is decided from a tensor's name once, when its copy is
staged - the stage line says `194 of them mixers, 31 at the trunk's ends` of the 496 - and applied
at *walk* time, so every policy races against every other in the one load that staged the one set
of arenas, and a matrix belonging to two classes is refused by either. `make score`'s decode arms
are that race: seven arms over 1199 teacher-forced positions, each measured against the stored
format rather than against the arm above it (`/tmp/i4h_tier.log`).

| what a step reads | nll | ms/token | of a step | worst nats | top-1s moved |
|---|---|---|---|---|---|
| as stored | 2.3056 | 108.1 | - | - | - |
| IQ4 dense, `all` (ships) | 2.3356 | 93.7 | 1.15x | 7.0 | 295 of 1199 |
| IQ4 dense, `nohc` | 2.3209 | 100.8 | 1.07x | 5.1 | 242 |
| IQ4 dense, `lean` | 2.3160 | 101.3 | 1.07x | 4.9 | 251 |
| Q8_0 head | 2.3054 | 106.1 | 1.02x | 0.077 | 9 |
| `all` + Q8_0 head (ships) | 2.3358 | 91.4 | 1.18x | 7.0 | 294 |
| `lean` + Q8_0 head | 2.3160 | 99.6 | 1.09x | 5.0 | 255 |

Two readings of that table, and they are in tension. The loss is where §4.1 said it would be:
holding the mixers and the trunk's end blocks at Q8_0 recovers two thirds of the dense leg's
+0.0300 nats for 7.6 ms of a step, which says the error that matters is the one a gate on the
residual carries and the one an interior projection's rows carry least. But `all` is what ships,
because the only purchase left on the table is the marginal one - `lean` up to `all` is 8.2 ms of a
100 ms step for +0.0197 nats - and a decode step is what a crush turn pays for. `lean` is therefore
the dial for a reader who wants the loss back, priced rather than argued, and the two halves of it
are not alike: `nohc` - the mixers alone held back - is +0.0153 of that at 100.8 ms, so the 194
mixers are 7.1 ms and half the loss, while the 31 matrices at the trunk's ends are the remaining
0.5 ms and 0.0049 nats. The ends are the cheap half of the two to hold back and the mixers are
where a step is; a reader who wants most of the loss for least of the speed would hold both, which
is `lean`, and would stop there.

The head's leg is free again at -0.0001 nats, nine choices moved of 1199: that copy is a rounding
of a format the file never stored those rows in, and it is the one leg of the three that has no
price.

Two of the things that paragraph named are built now, and the third has a number on it.
`gemv_chunk_job` carries the type its walk reads, so `gemv_rows` - the head walked over several
rows at once - can be handed a copy the way `gemv` always could, and the draft head is a normal
caller of it. Staging proved the MTP file's `output.weight` to be the trunk's bytes, byte for byte
across all 497 MiB, so one pair of arenas serves a step, a window and a draft row and costs no RAM
twice over (`make head` says so at open: `the draft head is the same bytes, so one pair serves
both`). A window's `rows` read the copy too, behind `QE_I4_HEAD_WIN`, which is the regime the copy
wasn't obviously for: it is 30 % bigger than the format it replaces and wins only on the ALU its
walk avoids, and a window is exactly where several rows amortise that ALU. They do. At the draft
length the engine resolves to, a speculative pass goes **89.5 to 86.7 ms per emitted id** with the
window reading the copy, and the plain loop of the same load is 91.6, so k=1 is **1.06x** of no
speculation where the same arm measured 1.02x with the window held at the stored format
(`/tmp/head_e.log`, 0 FAIL, `the head's drafts never change the ids greedy says` and
`no speculative pass was abandoned` both green, and the three ingestion identity legs green with
the copies pinned off for the pass they are made in).

Why that margin moved at all is worth writing down, because it is the kind of number that
gets quoted wrong twice. Giving the head a cheaper copy made a step 1.19x faster, which takes
16 % off the denominator of every ratio a speculative pass is measured by, and added nothing at
all to the window until the window was allowed the same copy: a draft row was 12.5 ms before the
copies and 12.7 after, so all of k=1's margin was on the step's side of the ledger. A win on a
step is therefore not a win for speculation unless every arithmetic in the pass moves together,
and `make head`'s arms are reported against the plain loop of the same load for that reason.

What is left named is one item: the draft head's *own* dense matrices. `dq_copy_for` refuses a
tensor whose owner is not the trunk, so a draft row still walks the Q8_0 it was stored in - 12.7 ms
of a row, 0.129 of a step, on the path that only runs when a draft is being made. The head's two
copies are still staged whether or not the reading asks for either, because the A/B that decides
between them has to race them in the load that staged them, and the dense copies cost 1.9 GiB
resident on top of the 3.3 GiB of §7.18's flat rows, both refused before the listener opens rather
than after.

## 8. Sequences, state and concurrency

**8.1 P3 Continuous batching is the throughput answer, and it is now a scheduling
problem.** Generation is serialised by the engine's own mutex (`engine.c:84`,
`serve.c:23`), so today a second request queues behind a turn. A bandwidth-bound
step reads the weights once per step whatever the batch, so a batch of B tokens
divides the dense half's bytes per token by close to B, and §12's chunked-walk
results are already the kernel-level proof of that. What blocks it is concrete and
mostly already mapped: `qe_forward` owns one sequence's state, so per-sequence state
needs to become an array or a per-sequence allocation; the grouped expert pass (7.2)
and the blocked head (4.4) are the two paths a batch actually needs; and the policy
has to price the per-sequence budget, 270 MiB at 8k context of which 113 MiB is GDN
state that cannot be shared (§8's gap row, `ARCH §5`). Note the tension with
§4: speculation is a net loss at *every* concurrency on this host, not only at eight or
more - §7.19 measured 4.14x of a step to verify a nine-row window against 1.6-1.9 ids a
pass to pay for it - so batching and speculation have to be one decision, not two, and a
batched engine is the one that makes a verify window pay.

**8.2 P2 Allocate the attention cache in blocks, not at full `n_ctx`.** The default
window is the checkpoint's 262144, so a sequence maps 3 GiB of `kvd` plus 1.5 GiB of
`kraw` at open (`ARCH §5`). Blocking it means one short conversation
stops reserving a long conversation's memory, which is the precondition for 8.1 and
for `--ctx` staying out of the operator's way. Cost: an indirection in the gather,
which is exactly the thing 5.1 is making sequential - so do 5.1 first and keep the
indirection out of the inner loop by mapping whole 2 MiB blocks.

**8.3 P2 The 113 MiB GDN state is the concurrency wall.** f32 is what the reference
uses. A bf16 state halves both the residency per sequence and part of the 216
MiB/token of state traffic - the same bytes 7.1 attacks from the other side. The
accuracy risk is the highest in this file: §13's `1/sqrt(head_state)` bug is four
nats, and a recurrence accumulates every error it is given. If it is attempted, it
is attempted with the loss curve, the depth probe, and a per-layer variant flag, not
with a perplexity spot check.

## 9. Build and codegen

**9.1 P2 LTO.** No `-flto` in the flags, and the kernels live in different
translation units from the pass that calls them (6.7 is one symptom). The kernels
are intrinsics, so the win is not in the inner loop; it is in the wrappers, the
epilogues, and cross-TU constant propagation of the shapes.

**Measured, and it is not a win.** `-flto` over the whole tree: Q8_0 fma 74.1, 75.3,
74.6 GB/s against 75.2, 75.2, 75.2 for the shipped flags, with `make fast`'s per-type
sums landing inside their own run-to-run spread (78.8 to 82.2 either way). The
prediction in this paragraph was right about where the win would have to come from
and wrong that it was there: the wrappers are not the cost. Two things came out of
running it. The first is that `-flto` was not warning-free, and one of the two
reasons was real rather than a lint: whole-program view saw that
`qe_spec_generate` has two `return -1` paths that leave the caller's
`qe_spec_stats` unwritten, and `bin/bench_mtp` prints that struct - so a failed run
reported whatever the stack held. It is zeroed on entry now (`mtp.c`, and the header
says so), which is worth having with or without LTO. What remains under `-flto` is
one false positive: a `-Wstringop-overflow` on the prompt copy in
`handle_completions`, where `nprompt` counted those children and each was rejected
unless `kind == QE_JSON_STR` (`serve.c:1008-1030`), so `str_len` is a parsed length
and not the SIZE_MAX the compiler thinks it might be. Saying so in a comment does
not move gcc, and it is not worth restructuring a request path for a flag that is
not on.

**9.2 P2 PGO, then BOLT if PGO pays.** There is a reference corpus and a
deterministic greedy path, which is everything a profile needs. The engine is
dominated by two loops, so expect little; the dispatch and reduce paths and the
indexer's scalar loops are where a compiler could plausibly do better with
statistics.

**9.3 P1 `__restrict` on the numeric prototypes.** No `restrict` appears anywhere in
`src/`, and §14 records the consequence directly: gcc emitted runtime alias
versioning for the delta rule because `Sr` may alias `kh`. Marking the dot, gemv,
norm, axpy and mixer prototypes restrict gives the same freedom to every elementwise
loop for free. Not a claim of speed - a claim that the compiler stops having to ask.

**Done, and it is exactly what the last sentence claims.** The marks are on the dot,
axpy, their q8_0 twins, the seven `qe_gemv_*` plus `qe_gemv_t`/`qe_gemv_multi_t`,
`gamma` in the grouped norm, the two stream mixers, and `qe_rope_apply`; the
elementwise norms and activations are deliberately unmarked because callers *do*
alias them (`qe_silu(f->lo, f->lo, ...)`, `qe_rms_norm_grouped(iq, ..., iq, ...)`) and
`make fast` pins that use. Counted by disassembling both builds: `qe_axpy` 199
instructions to 137, `qe_stream_inject_add` 293 to 210, `qe_rope_apply` 414 to 400,
`qe_rms_norm_grouped` 461 to 451, `layer.o` 4548 to 4379 total. `kern.o` and
`kern_iq.o` come out instruction for instruction the same, which is the expected
answer for hand-written intrinsics. Bandwidth is where it was (75.2 GB/s on the
engine's walk against a 77.4 read control), so the claim holds and no stronger one
does.

The §14 consequence this paragraph cites does not survive contact with the compiler:
marking `S`, `Sr`, `q`, `k`, `v` and `y` restrict in `delta_task` leaves its machine
code byte for byte the same (`cmp` on the two disassemblies). The versioned check on
the update loop is not asking whether the state aliases a vector - it is asking
whether `qh` and `kh` overlap, and their offsets are `qk * hs` and `(nk + qk) * hs`
with `qk` known only at runtime, so no declaration answers it. The comment at the
function now says that instead of the old explanation, which was wrong.

**9.4 P2 Loop and branch alignment.** `-falign-loops`, `-mbranches-within-32B-boundaries`
and friends are free to test on a binary whose runtime is two loops. Expect nothing;
`make fast` answers in under a minute.

**Tested, nothing, as advertised.** `-falign-loops=32 -falign-functions=32` over the
tree: Q8_0 fma 74.7 and 73.3 GB/s against 75.2 and 75.2 for the shipped flags, the
read control 76.2-76.9 against 77.4-77.7, and `make fast`'s sums 83.0/84.6 against
82.4/83.1. If anything slightly worse, and clearly not outside the spread on the
instrument that matters. `-mbranches-within-32B-boundaries` is not an option this
compiler has (`cc -Q --help=target` on gcc 16.2.1 lists `-malign-loops=` and friends
and no branch-boundary flag), so that half of the idea is a Clang experiment.

**9.4b Tested: the other compiler, and the cache-size parameters `-march=native`
guesses on a hybrid part.** A Core Ultra 7 265 has two core classes with different
caches, and `gcc -march=native -Q --help=target` on it reports a cost model built on
`--param=l1-cache-size=32` and `--param=l2-cache-size=30720` - the second of those is not
this chip's L2 at all, it is nearer its L3, which is what a single number for a machine
with two L2 sizes has to become. The question is whether that mis-tuned model is costing
anything, and it has a cheap answer because the flags are not the only thing that decides
codegen: the same tree, the same `-march=native -O3 -funroll-loops -ffp-contract=fast`,
built with gcc 16.2.1 instead of the shipped clang 22.1.8 and run through `make walk` at
the engine's width gives `attn_qkv` at T=512 of 441, 436, 434, 443 GMAC/s against the
shipped 441, 437, 439, 440, `ssm_out` 452, 463, 461, 468 against 454, 454, 450, 461, and
`hc_down` 523, 542, 533, 518 against 528, 552, 529, 530 (`/tmp/walk_gccA.log` against
`/tmp/walk_clangA.log`, one invocation of each, nothing else on the machine). Every pair
inside the spread one run of that bench is worth, so there is no codegen gap to close and
no per-class compilation to argue about. Two things follow that are worth keeping
together: the walks that matter are hand-written intrinsics, so a vectorizer cost model -
whatever number it has been given for L1 - has almost nothing to act on; and the two
thread groups a hybrid machine offers are not reachable by compiling twice anyway,
because one binary runs on both classes and `QE_PIN` was already measured to change
nothing on the expert stage (§7.8). Prefetching, for completeness, is not an untried
idea either: both forms of it lost 6-67 % and were deleted (`src/moe.c`'s header keeps
the account, `--help` says so in one line).


bit-identity rests on, and the whole verification strategy rests on bit-identity);
any `-mavx512*` target (SIGILL on this silicon, `Makefile:3-7`); building the
bandwidth-measuring TUs at `-O3 -march=native` alongside the kernels (the rule
behind `make sanitize` at `-O1`, `Makefile:303-308`).

## 10. Instruments worth building before more guessing

Almost every "predicted size" in this file is arithmetic on a ceiling, which is the
strongest statement this repository allows and still a prediction. The four
instruments below are what turn the rest of it into measurement, and none of them
needs a new idea to justify it: each is judged against a quantity that is known
without the counter, which is `§12`'s instrument discipline.

**10.1 P1 Per-op LLC-miss and bandwidth counters inside the engine.**
`bench/bench_prefill.c:152-219` already has the working harvest, and the parts that
look incidental are the parts that matter. It reads the event out of sysfs instead of
hard-coding it; it arms **both** monitoring units (`cpu_core` type 4 and `cpu_atom`
type 8, both exporting `0x2e/0x41`) because a task-bound event is not scheduled while
its thread runs on the other core class, and counting one unit lost nine of sixteen
threads and printed a beautiful zero (`§12`); it reads exactly one `u64` per fd,
because `read()` returns 8 bytes without a `read_format`, and a short read under a
`== 24` test is the most convincing failure a counter can have; and it calibrates the
result against a walk whose size is known without a counter, which is where 68.3
bytes per miss against a 64-byte line comes from.

Lift it into the engine's `--profile` path (`forward.c:1784`'s stage timing, which
today prints milliseconds and nothing else) so that every stage of a *real* step gets
a byte count next to its time. That is the missing column for the 104.4 against
79.5 ms gap in section 1, and it is the instrument that would rank 4.1, 5.1 through
5.6 and 6.1 against each other by measurement instead of by the ratios in section 0.

**10.2 P2 Dispatch count and barrier wait, per stage.** `qe_pool_run`
(`mem.c:451-483`) is one job slot, a generation counter, and a `QE_PAUSE()` spin on a
completion count. Count dispatches and cycles spent in that spin, summed per stage and
per thread. The items in section 6 disagree about where the handshake floor goes -
6.1 deletes dispatches, 6.3 hides them, 6.4 deletes the barrier - and a counter is the
only thing that distinguishes "1.3 ms of publishing a job" from "12 ms of the master
watching the slowest worker", which today are one invisible number.

**10.3 P2 Serial-time accounting.** A thread-id-0 cycle counter per stage, for the
work only the master does: the elementwise kernels in `layer.c`, the GDN convolution
and its alpha/beta pre-pass, `qe_moe_select`'s 512-logit scan, the indexer prologue
(`forward.c:1549`), the sampler tail, and the `sweep()` passes that run four times a
layer per token (`forward.c:1165`). The +25 ms gap in section 1 is partly this, and
`§14` cannot show it because it times stages, which are twenty threads' work. That
column is what 6.5 through 6.9 are all sized against.

**10.4 P1 One long-context curve.** Every figure in `§14` is an 8k-context number.
One instrumented sweep at 1k, 8k, 32k and 128k, printing the sequence's KV and state
footprint beside it, is the evidence the crush workload (a prompt in the tens of
thousands of tokens) actually needs - and the only thing that can rank 5.1 (gather
misses grow with selected context), 6.8 (an O(pos) indexer) and 7.5 (where dense
stops losing) against each other, since all three grow with context and nothing
measured here says how fast.

**10.5 Method, so the instruments are believed.** Take readings after 3.1, not
before: a `powersave` floor makes every absolute number a function of how long the
process has been hot. Give each counter a check whose answer is known without it - a
round of known bytes, a stage whose traffic the tensor table already implies, a T=1
comparison of two identical walks - and keep that check in the printed output rather
than in a comment, because `§12`'s first attempt printed a `-643%` saving with
complete confidence, and a counter that goes silent on the next kernel release fails
the same way.

## 11. Measured dead: do not resurrect

| idea | result | instrument |
|---|---|---|
| Two thread groups, one chunk: a layer's expert walk co-running with another layer's dense walk | dead as a **thread split**: 0.961x at the best split (9+10) and 0.711x at the worst against both walks at the full width in turn; the machine is additive across the two walks, and the 0.64 that first made it look like 1.5x was the mixed wall against the two halves alone on half the machine each (§7.8). That instrument could only ever have moved work between two fixed groups, so the stronger form (one schedule, where the tail of whichever half finishes first is the other's) was built and measured on its own: a chunk's expert stage as one stage graph instead of seven dispatches is **1.005x** of the schedule that was there first at 512 tokens at width 512 (§6.4, §7.8) - and it is on, and it is **1.026x** now that the graph carries the chunk recode as items of itself instead of falling back to the walk the schedule replaced, 4 passes of 4 with a byte-identical trunk (§6.4, `/tmp/f25_wide.log`) | `make cosched`, `/tmp/cosched3.log`; `make forward`'s expert-stage leg, `/tmp/f12_wide.log` |
| AVX-VNNI `vpdpbusd` dot **for a single-row walk** | 10.6 against 17.1 GB/s on one core, 42-98% of the read walk elsewhere (§1.2). The scope is the row-at-a-time walk a decode step does, measured in bytes: there the dot finishes long before the loads do. Against an eight-row shared-dequant walk the same instruction is 2.75x with the scale applied once per 256 weights and **1.36x** once it is applied where the format puts it, every 32 weights, which is a stage win of about 1.3x and not the 50 tokens a second (§7.10). So the dead thing is the T=1 substitution, and the live thing is a tile | `bench_alu --kernels gemv,vnni`; `/tmp/iq3probe.log` |
| The dequantised IQ3_S block held in four vector registers instead of parked in an L1 line, eight rows per walk | 0.969 to 1.014 of the parked walk across 2 to 512 rows (best of three, legs alternating, one process): a wash, and the four stores it removes were not on a port anything was waiting for, which the `make fast` spill count had already said of the parked form (`src/kern.c`, `QE_Q8_0_MULTI8_FORM`: SROA promotes the parked line to registers there too). Forms stay reachable from `QE_PARK=9..11` and from `make moe`'s third leg; the default stays off | `make moe`, `/tmp/moe_reg1.log` against `/tmp/moe_park0_base.log` |
| Split weights and scale arrays | 70.0 against 75.9 at 512 MiB (+43% at 64 MiB, a regime the step never enters) (§1.2) | `bench_alu --kernels gemv,gemvsplit,readsplit` |
| Two rows in flight for more MLP | 74.6 against 75.9 (§1.2) | `bench_alu --kernels gemv,gemv2` |
| Building IQ3_S codebook indices with vectors instead of the scalar byte chain | 1.49x slower on the walk on average and 2.04x at T=1, 1.78x at T=4: the shorter inner loop is real and the crossings back and forth between vector registers and general registers to feed the gather cost more than the byte loads they replace, and the effect grows as the walk gets narrow. The Q8_0 and IQ4_NL columns of the same interleaved runs read 1.013 and 0.998, so it is the column and not the clock | `/tmp/sp/ab1.log`, the three-type walk A/B |
| Software prefetch in the expert pass | 6-67% regression, code deleted (§1.2, `include/qe.h:401`) | `make moe` |
| Evict-first (`_MM_HINT_NTA`) hint on the weight walk | neutral at 408 B ahead, 1-3% regression at 1360 B, against a control walk that drifted 1.5% by itself (§5.4) | `bench_alu --kernels gemv,read` |
| `-flto`, `-falign-loops=32 -falign-functions=32` | inside the instrument's spread either way; the align build was 1-2% worse on the same control (§9.1, §9.4) | `bench_alu --kernels gemv,read`, `make fast` |
| Clearing less than the whole `down_task` partial | 38.8 GB/s against 39.6 and 39.7 with the `memset` in place (§6.9) | `make moe` |
| Non-temporal store for the recurrent state | 72.2 against 74.0 at 512 MiB (§1.2, §14) | `bench_alu --kernels read,state,statent` |
| Pinning workers, `cpu_capacity` slices | 2-3x the handshake; 4% of a token (§12) | `make pool`, `make score` |
| Weighting the expert slice by core class as well as by cost (`QE_MOE_BAL=2`) | Worse than the cost-only slice at every skew, unpinned and pinned: 0.964x and 0.967x against cost's 0.996x and 1.017x on an even router, 2.919x against 3.153x at 512 rows and zipf 1.0 (3.089x against 3.143x pinned). The P/E capacity ratio (1011 against 783) is bandwidth headroom, and a 19-thread expert walk is not bandwidth-bound, so weighting by it manufactures the skew cost-slicing exists to remove (§7.16) | `make moe`, `/tmp/moe_bal2.log`, `/tmp/moe_bal2_pin.log` |
| Hot-expert pinning | 0.19 against a 0.20 baseline (§9) | `make route` |
| Hand-vectorising the delta rule | 0.3 ms of 48, because the compiler already did it (§14) | `make score --profile` |
| Row-blocked projections in the engine | 65 ms/token against 67 with blocking (`QE_PROJ_MB`, §12) | `make forward` |
| Bigger SSE send queue | a dead reader absorbed a whole answer and the engine generated to the end of the turn for nobody (§7 of DESIGN on the stream timeout) | `make http` |

## 12. Out of scope because of what this repository is

Nothing in this file proposes making the checkpoint resident lazily, demand-paged,
or partially loaded, and nothing prices the load pass: those are the two rules in
`AGENTS.md`, and an engine that reads the device per token is a different product.
That also excludes the tempting-sounding startup ideas (parallelising the load, a
warm arena snapshot to skip it, a compressed on-disk form), because every one of
them is only interesting as a device-dependent number.

Two things are out of scope for a different reason: the checkpoint's own template
evaluator and a vision encoder (`§8`'s gap rows). Both are features, and neither
would move a byte of a decode step.

## 13. Appendix: drift found while reading, worth fixing near the code it lives in

Not performance items, but each one is a place where the next person will be misled
while chasing performance.

- `qe.h` told callers to get `row_bytes` from `qe_tensor_row_bytes`; no such function
  ever existed. It now says what to divide (`t->bytes / t->ne[1]`) and names the two
  file-static helpers that do it.
- `qe.h` said a MoE descriptor's `n_threads` of 0 means `min(pool size, 8)`;
  `qe_moe_create` calls `qe_pool_size()` with no cap and its own thread table argues
  for the full pool. The comment now says what the code does.
- `f->moe_logit` (`forward.c:1971`), `f->h640` (`:1970`) and `f->expert_y` (`:1973`)
  are allocated per sequence and never read on the forward path - declaration and
  allocation are the only mentions either file has - and `qe_prefault` and
  `qe_madvise_seq` (`mem.c:934-943`) still have no callers, with `qe_prefault`
  ignoring its thread count. (`f->sel_mark` is gone.)
- The doc comment that promised the f32 router's mixing weights is gone: the header
  now says selection reads the bf16 shadow and the renormalised weights are re-read
  at f32 by `qe_moe_refine`, which is what the path does (4.3).
- The two formatting and build drift items this list carried are fixed: `Q8_ROW` is
  defined before its first use in either branch, and `moe.c` is formatted. Neither
  could be re-verified as a build here anyway, since `-U__x86_64__` on this host
  selects 32-bit headers before it selects the fallback path.
