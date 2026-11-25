# qe: instructions for agents

Four things that have been slipped before and are not negotiable: the bar below, and the
three rules after it.

## 0. The bar: 50 tokens/second on ingestion

Ingestion is over the bar, and the bar stays. Current reading: **87 to 93 tok/s** (10.7-11.5
ms/token, `make forward` and `make decode --pp` after ideas.md §7.30-7.35, 0 FAIL), and generation **15.6 tok/s** (64.0
ms/token) where it was 10.9 - a third of every decode step had been threads waiting at barriers on
this P+E chip, and rows are now handed out from a cursor (ideas.md §7.30; `make decode` is the
instrument). Any new walk whose output rows are independent should take its rows from a cursor,
not `qe_slice`, and any walk that sums per-thread partials must not, or its answer depends on the
schedule. The figures that follow are the previous cycle's: **61.3 to 63.7 tok/s - 16.3 to 15.7 ms/token at the
512-row batch the sweep ships**, measured eleven times over eleven loads (`/tmp/f31_wide.log`
through `/tmp/f41_wide.log`; nine of them had no failing leg, one carried the rewind failure
ideas.md §7.19 records, one the walk-width default ideas.md §7.26 records and reversed, the
two before it being the load the small-projection batch of ideas.md §7.27 landed on and its
confirmation, green at 15.8 and 15.9 ms/token, and the last the load ideas.md §6.6's exponential
leg landed on, green at 15.7),
against llama.cpp
b12359's 51.33 tok/s at the same width (`/tmp/llama_pp.log`), which is **1.19-1.23x ahead** where
this file opened 1.19x behind. The standing
requirement is **50 tokens/second on this host** with functionality and output quality kept; it is
met, and it was met the same way as before, not by one thing but by switches, one of them this
cycle: the 180 projections this checkpoint walks per token that sit under the chunk walk's
threshold went from one pool dispatch per token to one per chunk (`QE_SMALL_PROJ_BATCH`, on),
worth 16.44 against 15.86 ms/token of a width-512 chunk - **1.037x** of ingestion, 1.13x of
`block_proj`, 1.16-1.19x of the two mixers, and byte for byte the same logits (§7.27). The switch
before it: a dense walk reading one scale per output *row* of a requantised copy made at load time,
three rows against one int8 activation load (`QE_I8_FLAT`, on), worth 1.78 against 2.91 ms/token
of `block_proj` and 14.92 against 16.00 ms/token of a whole ingested token (`/tmp/f35_wide.log`,
ideas.md §7.18). What that leaves is the same two gaps with one of them much the smaller. The
expert stage is 50 % of an ingested token (7.27 ms of a 14.48 ms step at the shipping width, every
shipping default on, including the expert stage as one dispatch), and ideas.md §7.13 puts the whole
of what any better walk of these formats can buy at the shipping width at **1.2-1.6x** - the routed
pass at 205 GMAC/s against 284 for the same codebook walked uniformly and 330 for a uniform Q8_0
row - of which the shipped walk already takes 1.18x, so this gap is nearly closed and is not where
the next 1.1x comes from. The dense stage is 1.60 ms of that step, and what is left in it is no
longer a kernel: it is the fastest of the three walks this file has built, reading a copy that is
already the permutation §2's row 20 asked for, and its wider tile is now built and measured rather
than supposed - of five row counts against one activation load, raced twice, the three that ships
ties the only other tile the sixteen registers leave, and the same walk with its weights resident
is 1.00x of the one reading DRAM at the chunk width, so the wall is an issue-rate ceiling and not a
dot worth rewriting a fourth time (§7.15, §7.18, §7.28).

The instrument that says where the time is is ideas.md §7.10: four rewrites of the expert walk raced against it in one process. The codebook is not the wall (0.84x with it taken out, at the width that ships), the multiply is not the wall either (an integer dot is 1.36x once its scale exit sits where every format in this checkpoint puts it, every 32 weights, and 2.75x only while it pretends otherwise), and the distance to 50 is not in walking these formats better but in one activation load serving several output rows. The fourth of those legs - one activation scale per row instead of per 32 - is built, ships, and measures 1.13x at eight token rows and 1.27x at 512 on the codebook walks, with the Q8_0 column reading 1.00x as the control (`/tmp/walk_rs1.log`); what is left is still the tiled form of one load serving several rows, and §7.12 is the reason that is a gemm and not a second row. That form is measured, wired, and on: with the scale exit hoisted out of the token loop (§7.17)
and one scale per output *row* of the matrix, three output rows sharing one activation load walk
at **1.84x** of the walk that used to ship, 1737 GMAC/s at the width a chunk is ingested at, exact
against it to the bit - and the requantised copy of the dense matrices it needs is staged at load
time, 496 matrices and 3.3 GiB, refused with two printed numbers on a host that cannot afford it
(§7.18, `/tmp/f35_wide.log`). What remains of that family is measured dead rather than open: a
second row is the sixteen-accumulator wall (§7.12), a tenth token of a codebook walk is an
in-engine zero measured twice and left at eight (§7.21), and the wider tile of the flat copy is
built, raced at five row counts in two runs, and ties the one that ships (§7.28). The same wall
runs the other way on the float pairing: the second output row is on for Q8_0 and IQ3_S and **off** for
IQ4_NL, whose paired form measures 0.92-0.98x at every width a bucket arrives at, so the gate is
per format (`QE_IQ4_NL_PAIR_MIN_T`, §7.29).

**Speculation is measured, and it ships at the shortest draft.** The nextn head is on this host,
wired, and correct - `--mtp` stages it resident, it runs as one more layer of the trunk, and
drafting 48 ids through it moved none of the ids greedy says (`make head`, `MTP=`). The wall is
verification, not the head: a draft row is 0.129 of a step while a verified nine-row window is
**4.35x** one, so acceptance has to beat the window, and it only does at the shortest draft -
**1.06x** at k=1 with a head and **1.05x** at k=2 with the n-gram hash, 0.99x at k=8 for the two,
over a plain loop of 91.6 ms an id (`/tmp/head_e.log`). It was 1.19x and 1.05x over a 110 ms loop
(`/tmp/f40_head.log`), and the 0.13x that went is not lost: a step is 1.19x faster now, which is
16 % off that ratio's denominator, and a k=1 pass is only back above 1.05x because a verify window
reads the same head copy a step reads (§7.23). The draft length is accordingly resolved at open
from the drafter that won (`QE_SPEC_K_AUTO`, `--spec-k 0` off), not hardcoded. Do not treat "MTP is not finished" as a
task; ideas.md §7.19 is the account. What it left unproven is the one assertion sitting at 4-of-5
`make forward`'s rewind leg, which flipped one boundary's choice under the shipping arithmetic on
`/tmp/f34_wide.log` while its exact twin passed to the bit in the same load. That assertion has a
byte-level instrument now, in the arm of `make batch` where a window and a step are one
arithmetic, and it has already paid for itself: a rewind that put a sequence's state back and
left its position where the abandoned rows had reached is 0/3 boundaries byte for byte wrong
while every one of those rows still reads perfectly (`/tmp/window_ab.log`, §7.25).

**Throughput is a second axis, and it is measured too.** `--batch N` drives N sequences
through one walk of the weights, and the step eight of them share is **2.15-2.20x** a step of
one while a wave of eight short requests is **1.79x** the same eight one at a time - for
**0.060 nats**, and for a per-request tok/s that gets worse, which is what a throughput
setting is (`make lane`, ideas.md §7.25). The caveat is as large as the claim: prompts are
still read one at a time, so two clients on 512-token prompts together cost **1.00-1.02x** of
what they cost alone and two on 2048-token prompts 1.01x. It is a decode multiplier. And
`--batch 1`, the default, is where byte-exact greedy determinism lives: a step of three rows
and a step of eight are different walks of the same weights, so on a batched engine an answer
is a function of its prompt *and* of who else is in the process. What is *not* left to
chance is which sequence a row belongs to: `make batch` asserts, on the arm where a batch is
only a scheduling change, that a row of a partly full, moving step is byte for byte the row
that sequence walks alone. A row needs a position as much as it needs a row of the cache, so
positions are per sequence, and a draft window - one sequence's whole state, position included -
is refused for a batched pass.

**Width is a per-stage request, and only two walks may make it.** `qe_pool_run` takes the
thread count per call, so `--walk-threads`/`QE_WALK_THREADS` can give a lone token's dense
`gemv` a width of the pass's leaving out: that walk hands one output row to one thread in one
kernel call, so a step's logits are byte for byte one reading at every width, and
`make forward`'s width leg asserts it. The expert stage is the counter-example and it is not
width-invariant: `down_task` and `reduce_task` sum one partial per thread, so moving `nt` moves
the float summation tree, and the expert, attention and delta walks stay at the pass width.
Anything given its own width has to pass the same row-sliced test or be priced in nats. The
width itself is settled: six widths raced inside the engine say the pass width, 1.07x slower at
16 of 19 threads and 1.17x at 13, where one kernel walking the same bytes prefers 16 by 6.2 % -
a kernel's own steady-state read of one buffer does not transfer to a step with ~800 dispatches
of it (§7.26).

**A projection under the chunk walk's threshold is one ask for the chunk.** `proj_walkable`
sends a matrix smaller than `QE_PROJ_MB` (1 MiB) to `gemv` instead of the chunk walk, and `proj`
used to pay for that once a token: this checkpoint has 180 such projections - 96 mixer injection
matrices, 72 gated-delta alpha and beta, 12 indexer key projections - so a width-512 chunk took
92,160 barriers of a nineteen-thread pool where the same chunk's walkable projections take 180.
One ask over the flattened (token, row) space (`proj_small`, `QE_SMALL_PROJ_BATCH`) is
**1.037x** of an ingested token, 15.86 against 16.44 ms/token, 1.13x of `block_proj` and
1.16-1.19x of the two mixers (`/tmp/f39_wide.log`). It reaches a step not at all - one token is
one dispatch either way, and the leg times both arms at an identical 88 ms - and that sameness
is the rule rather than the ratio: a row goes to one kernel call, so neither which token was in
the call beside it nor which thread walked it is part of what the row sums to, and `make forward`
asserts a chunk's logits byte for byte at both settings instead of pricing them in nats. Anything
that regroups rows or tokens has to pass that same test or pay in nats.

Implement rather than adjudicate. A direction counts as measured dead only when the instrument
ran against the form being claimed; if the schedule or kernel being argued about is not in the
tree, build it and measure that. A session that ends with a proof instead of a patch has not
finished.

## 1. The whole model is in memory at engine launch

`qe_engine_open()` stages every weight byte before the listener opens, and it is
STRICT about it: an engine that cannot make the checkpoint resident exits with
the numbers that decided it. Demand-paged mapping (`QE_STAGE_MMAP`) is a metadata
view, refused for serving and accepted only together with `--report`.

Do not add a lazy path, a partial load, a "warm on first request" mode, or an
`mmap` fallback. There is no configuration in which a token reads the device.

## 2. Never measure or report disk speed

The one-time load pass is not timed and no rate is reported for it: no seconds,
no MiB/s, no GiB/s, no "s to place", no `weight_stage_seconds`. Startup prints
where the bytes ended up (`ram+thp`, `hugetlb`, `ram+file`) and how much a
previous run left behind. If a number would change when the model file moves to a
different device, it does not belong in this repository.

Memory bandwidth is a different thing and is measured constantly (the ceiling for
the shape decode reads is 76-79 GB/s at 19 threads, a Q8_0 GEMV reaches 75.9, a
flat read on eight performance cores 85.5: `make alu`, and `docs/DESIGN.md` §1.2 has
the per-shape table). That ceiling is not a knob:
both DIMM slots read `Configured Memory Speed: 5600 MT/s` against the same rated
speed, two 64-bit channels at 5600 peak at 89.6 GB/s, so those measurements are 85-95 %
of the hardware and no firmware setting on this board moves a token (ideas.md §3.5).

## 3. Never write a control token as text

Vocabulary markers are named by id or symbol (`QE_MARK_TURN_CLOSE`, `248046`),
never spelled out, and that applies to prose written *about* them as much as to
`.c`, `.h` and `.py` files. A marker sequence in model-authored text is a real
control token downstream: it terminates the turn, splits a prefix block, or
lands in a log where the next reader pastes it somewhere else, and the damage
happens before any code is run. This repo therefore keeps zero literal marker
sequences anywhere, `src/chat.h` included.

## Commands

```sh
make all       # kernels, benches, bin/qe-serve
make fast      # kernels, ~25 s, never touches the checkpoint
make test      # fast + mtp + json + chat + sample + turns + pool + moe
make http      # selftest server + tools/smoke_http.py + tools/fuzz_http.py
make batch     # B sequences through one forward step, three arithmetic arms, their nats and
               # the byte-level rewind leg (FUSE=2400; it pays the load)
make lane      # the same multiplier through the engine's request entry, with --batch N
               # (BATCH=4 FUSE=2400; it pays the load)
make decode    # where a step's time goes per stage - pool, barrier tail, serial - and a race
               # of decode switches in one load (ARMS=ship,old,...; bench_decode --nll N prices
               # each arm in nats, --win 2 a verify window, --pp N an ingestion; FUSE=2400)
make forward   # the chunk sweep against the checkpoint, one A/B leg per arithmetic switch,
               # plus the width leg that races a step's weight walk against narrower passes
               # (FUSE=2400; minutes, and it pays the load)
make head      # the same pass with the nextn draft head resident (MTP=), the speculation
               # arms and their acceptance histogram (FUSE=2400; it pays the load)
make ui        # the live panel on a pty, and its absence in a log (no checkpoint)
make shutdown  # SIGTERM with a request in flight: bounded, and the port back
MODEL=<volume1.gguf> make report   # headers only, no weight bytes read
```

A checkpoint load takes minutes, so run anything that touches the real file in
the background rather than letting it detach by itself. `-march=native` is
required: this host has AVX-512 fused off and a `-mavx512f` build SIGILLs.

Every target runs under `timeout -k` (`$(TO)` in the Makefile): a deadline that
only sends `SIGTERM` is a deadline against a process still able to hear it, so a
new target gets a number, and a test that cannot hang is worth more than one whose
timing was never asserted. Nothing that talks to the checkpoint may assume it will
be allowed to finish.
