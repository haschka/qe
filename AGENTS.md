# qe: instructions for agents

Four things that have been slipped before and are not negotiable: the bar below, and the
three rules after it.

## 0. The bar: 50 tokens/second on ingestion

Ingestion is over the bar, and the bar stays: **61.3 to 62.9 tok/s - 16.3 to 15.9 ms/token at the
512-row batch the sweep ships**, measured five times over five loads (`/tmp/f31_wide.log` through
`/tmp/f35_wide.log`; four of them had no failing leg, the fifth carried the rewind failure
ideas.md §7.19 records), against llama.cpp
b12359's 51.33 tok/s at the same width (`/tmp/llama_pp.log`), which is **1.19-1.22x ahead** where
this file opened 1.19x behind. The standing
requirement is **50 tokens/second on this host** with functionality and output quality kept; it is
met, and it was met the same way as before, not by one thing but by switches, one of them this
cycle: a dense walk reading one scale per output *row* of a requantised copy made at load time,
three rows against one int8 activation load (`QE_I8_FLAT`, on), worth 1.78 against 2.91 ms/token
of `block_proj` and 14.92 against 16.00 ms/token of a whole ingested token (`/tmp/f35_wide.log`,
ideas.md §7.18). What that leaves is the same two gaps with one of them much the smaller. The
expert stage is 48 % of an ingested token (7.14 ms of a 14.92 ms step at the shipping width, every
shipping default on, including the expert stage as one dispatch), and ideas.md §7.13 puts the whole
of what any better walk of these formats can buy at the shipping width at **1.2-1.6x** - the routed
pass at 205 GMAC/s against 284 for the same codebook walked uniformly and 330 for a uniform Q8_0
row - of which the shipped walk already takes 1.18x, so this gap is nearly closed and is not where
the next 1.1x comes from. The dense stage is 1.78 ms of that step, and what is left in it is no
longer a kernel: it is the fastest of the three walks this file has built, reading a copy that is
already the permutation §2's row 20 asked for, so the item in front of it is a wider tile of that
copy and not a fourth rewrite of the dot (§7.15, §7.18).

The instrument that says where the time is is ideas.md §7.10: four rewrites of the expert walk raced against it in one process. The codebook is not the wall (0.84x with it taken out, at the width that ships), the multiply is not the wall either (an integer dot is 1.36x once its scale exit sits where every format in this checkpoint puts it, every 32 weights, and 2.75x only while it pretends otherwise), and the distance to 50 is not in walking these formats better but in one activation load serving several output rows. The fourth of those legs - one activation scale per row instead of per 32 - is built, ships, and measures 1.13x at eight token rows and 1.27x at 512 on the codebook walks, with the Q8_0 column reading 1.00x as the control (`/tmp/walk_rs1.log`); what is left is still the tiled form of one load serving several rows, and §7.12 is the reason that is a gemm and not a second row. That form is measured, wired, and on: with the scale exit hoisted out of the token loop (§7.17)
and one scale per output *row* of the matrix, three output rows sharing one activation load walk
at **1.84x** of the walk that used to ship, 1737 GMAC/s at the width a chunk is ingested at, exact
against it to the bit - and the requantised copy of the dense matrices it needs is staged at load
time, 496 matrices and 3.3 GiB, refused with two printed numbers on a host that cannot afford it
(§7.18, `/tmp/f35_wide.log`). What remains of that family is measured dead rather than open: a
second row is the sixteen-accumulator wall (§7.12), a tenth token of a codebook walk is an
in-engine zero measured twice and left at eight (§7.21), and a wider tile of the flat copy is the
one thing in the family nobody has built.

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
`/tmp/f34_wide.log` while its exact twin passed to the bit in the same load.

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
flat read on eight performance cores 85.5: `make alu`): see `docs/DESIGN.md` §1.2.

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
make forward   # the chunk sweep against the checkpoint, one A/B leg per arithmetic switch
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
