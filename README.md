# qe

![qe: 1.8x faster prefill, 2.0x faster generation, against llama.cpp on one CPU](marketing.svg)

**A CPU-only inference engine for `Qwen3.8-Flash-Next`, hand-tuned to one machine:
an Intel Core Ultra 7 265 with 128 GB of DDR5 and no GPU in the picture.**

Not a fork of llama.cpp and not a wrapper around one: the GGUF reader, the
quantised kernels, the router, the attention indexer, the draft-head loop and the
HTTP server are all in this tree. It links two libraries - `libpthread` and `libm`.

```
LDFLAGS += -pthread -lm
```

There is no CUDA, Metal, Vulkan, ROCm, OpenCL or SYCL in it, and no `libggml`.
Every number below is a CPU number, taken on the box it was built on, with one
process resident at a time.

## The numbers

Measured on this host, default settings, 19 of 20 cores, one sequence:

| | qe | llama.cpp | |
|---|---|---|---|
| **Prompt ingestion**, 512-token batch | **87 - 93 tok/s** (10.7 - 11.5 ms/token) | 51.33 tok/s | **1.7 - 1.8x** |
| Prompt ingestion at 4,096 tokens of context | 83 tok/s (12.0 ms/token) | | |
| **Generation**, greedy | **15.3 - 15.6 tok/s** (64 - 65 ms/token) | 7.65 tok/s (130.7 ms) | **2.0x** |
| Generation at 4,096 tokens of context | 15.0 tok/s (66.5 ms/token) | | |
| **Throughput**, eight sequences in one step (`--batch 8`) | 18.0 tok/s across a wave of eight (measured before §7.30 below) | none | |

llama.cpp is the same checkpoint and the same host: b12359 for ingestion
(`-ngl 0 -t 20 -fa on`), b12110 for generation on the same corpus. The two engines
are never resident at the same time; nothing here was measured with both of them
holding 90 GB of weights.

The generation row moved from 10.9 to 15.6 tok/s in one cycle, and not by a faster kernel:
a third of every decode step was nineteen threads waiting at a barrier for the slowest of
them, because this chip is eight fast cores and twelve slow ones and every dispatch handed
them equal slices. `make decode` is the instrument that found it (ideas.md §7.30).

Instruments, so that none of the above is a claim: `make decode` (where a step's time goes
per stage - pool, barrier tail, serial - and a race of decode switches priced in nats),
`make forward` (the ingestion
sweep and one A/B leg per arithmetic switch), `make head` (a step, a draft row and
a verified window, with the nextn head resident), `make batch` (B sequences through
one step, three arithmetic arms, and a row of a partly full step asserted byte for
byte against that sequence walked alone), `make lane` (the same multiplier through
the engine's request entry), `make score` (teacher-forced loss and ms/token over
1,200 real positions), `make alu` (what this machine can actually read).
[MANUAL.md](MANUAL.md) is the running log, [docs/DESIGN.md](docs/DESIGN.md)
is the kernel-by-kernel account, and `--report` prints the budget from the file's headers
alone.

## Why it is faster on *this* machine

Because it was written against this machine's measurements instead of a portable
model of a CPU.

- **Rows handed out from a cursor, not sliced equally.** The pool has eight P-cores and
  eleven E-cores in it, and a slice that is equal by construction ends every dispatch on an
  E-core: measured, 32.5 ms of an 89 ms step was the master waiting for the last worker.
  Every walk whose output rows are independent - the dense GEMVs, the expert chunks, the
  indexer's block scores, the attention units, the delta rule's state rows - now takes
  its next chunk of rows from an atomic counter, so a fast core simply walks more; and the
  walks of a step that read the same input share one dispatch of the stage graph (953
  dispatches a step became 461). The tail is 2.1 ms. A row is still one kernel call, so
  this part is byte-identical to what it replaced.
- **A prompt's per-token work on every core, and the recurrence and the experts unstalled.**
  The loops between a chunk's walks (norms, gates, injections, the convolution, the attention
  prologue) were 3.3 ms of every ingested token on one thread; they run across the pool now,
  byte-identical. The gated-delta recurrence walks eight state rows at once instead of waiting
  on one FMA chain at a time, and the expert stage takes its work from cursors with a down
  projection that sums in a fixed order. Ingestion went 63 → 87 tok/s (ideas.md §7.33-7.35).
- **Integer walks on a decode step too.** A step's input is recoded to int8 once per
  dispatch, and both the IQ4_NL dense copy and the routed experts are walked with
  `vpdpbssd` against it - the form a chunk already used, which a lone token never reached.
  The dense projections then read at 80 GB/s, the bus.
- **The whole checkpoint is resident before the listener opens.** 87.25 GiB of
  weights are placed in RAM (or hugetlb) at startup, `mlock`ed and reused; an
  engine that cannot make the file resident prints the two numbers that decided it
  and exits. There is no lazy path, no partial load, no `mmap` fallback: a token
  never reads the device. On a 128 GB board that is not a luxury, it is the whole
  trick - and 7.7 GiB of the box is then spent on *copies* (below).
- **Kernels that run at the machine's own ceiling.** The decode step is one
  bandwidth-bound GEMV, so `make alu` measured the ceiling first: 76-79 GB/s for
  the shape a decode step has at 19 threads, 85.5 GB/s for a flat read on the eight
  performance cores. The shipped Q8_0 GEMV reaches **74-83 GB/s**. Software
  prefetch was measured as a 6-67 % regression and deleted, `vpdpbusd` without a
  matching weight layout was measured as a loss and deleted, and a
  `-mavx512f` build SIGILLs here because AVX-512 is fused off on this die, so the
  build is `-march=native` on AVX2 with AVX-VNNI.
- **Load-time requantised copies, not a faster walk of the same bytes.** A lone
  token is 66.6 % dense `Q8_0` projections and 8.9 % a `Q6_K` output head,
  and this host walks `Q6_K` at 44-46 GB/s. So at open the 496 dense matrices are
  requantised to 1.9 GiB of IQ4_NL rows and the 497 MiB head is re-stored as a
  644 MiB `Q8_0` copy - 30 % *more* bytes, 1.02x faster, because bytes are not what
  was binding. Together a step is **1.19x** faster, and `make score` prices what
  each copy costs the next-token loss so the trade is read rather than argued.
- **One int8 activation load feeding three output rows.** A chunk's activations are
  recoded to int8 once per layer and a whole expert bucket is walked against that
  panel with `vpdpbssd`; the dense projections get the same treatment from a copy
  requantised to one scale per output row, worth **1.84x** of the walk it replaces
  and 1.63x of a chunk's projection stage.
- **10 of 512 experts, walked as one pass.** Serial expert passes are 47.4 ms of an
  ingested token; one grouped pass over the bucketed experts is **27.0 ms**, and the
  seven dispatches around it are one stage graph so a short stage lends its threads
  to a long one's tail.
- **Speculation is built, measured, and off by default.** The nextn head these
  checkpoints ship as a sidecar file stages resident and runs as one more layer of the
  trunk, and at the old step speed it was worth 1.06x at one draft token. The step is now
  1.38x faster and a two-row verify window is not: it costs 1.6 of the new steps, so a pass
  that emits 1.6-1.7 ids no longer beats the plain loop. `--spec-k 1` (with `--mtp`) turns
  it back on; ideas.md row 30 is what would make it pay again.
- **Built for an agent's turn, not a benchmark's.** Prompt reuse across turns
  against a bounded rewind cache, q8_0 KV (21.3 KiB per token of context), an
  indexer that scores blocks so a 262,144-token window stays in line, greedy
  requests that speculate, and a pool one core short of the affinity mask because a
  spinning worker with no core of its own pays a scheduler timeslice at every one
  of the ~800 dispatches a token makes (106.2 ms at 19 threads, 125.2 ms at 20).

## Requirements

- Linux on one NUMA node, x86-64 with AVX2 + AVX-VNNI (this build was tuned on a
  Core Ultra 7 265: 20 logical CPUs - 8 performance, 12 efficient, no SMT - 30 MiB
  L3, max 5.3 GHz, 125.3 GiB addressable of the 128 GB fitted in two DDR5-5600 slots,
  both running at their rated speed).
- A C17 compiler and GNU make. No other dependency: no cmake, no curl, no
  OpenBLAS, no Python at runtime.
- **About 95 GiB of RAM free** for the checkpoint used here (87.25 GiB of weights
  plus 7.7 GiB of the load-time copies), and one GGUF you trust. If the copies do
  not fit, the engine says so at open with the bytes it wanted and the bytes it saw,
  and serves without them.
- Build it on the machine that will run it. `-march=native` is not optional here,
  and the tuning is host-specific: the numbers above are not portable, and on a
  different CPU the honest answer is `make walk` and `make alu` on your own box.

## Install and run

```sh
# 1. Build (a couple of minutes; nothing touches the model file)
make -j"$(nproc)"
make fast            # kernels only, ~25 s, no checkpoint
make test            # kernels, the speculation control loop, request parsing, chat
                     # templates, the sampler, the rewind cache, the pool, the MoE walk

# 2. What this engine thinks a token costs on this host (headers only; no weight byte)
MODEL=/path/to/Qwen3.8-Flash-Next-...-00001-of-00003.gguf ./bin/qe-serve --report

# 3. Serve it. Every weight byte is placed in RAM before the listener opens, so the
#    port stays closed until the engine can actually answer a request. After that
#    the file is never read again.
./bin/qe-serve --port 8097 \
  --model /path/to/Qwen3.8-Flash-Next-...-00001-of-00003.gguf \
  --mtp   /path/to/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf

# 4. Use it
curl -s localhost:8097/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qe","stream":true,"max_tokens":64,"temperature":0,
       "messages":[{"role":"user","content":"Name the three primary colours of light."}]}'
```

Only volume 1 of a multi-volume checkpoint is named; the rest are found by name.
Speculation is off unless asked for: `--spec-k 1` with `--mtp` drafts from the nextn
head, `--spec-k 2` without it from the n-gram hash. At today's step speed neither beats
the plain loop (ideas.md §7.30), which is why neither is the default.
`--stage huge` moves the arena into a reserved hugetlb pool and reuses it across
restarts. `--ctx N` is a memory setting as much as a token limit. `SIGTERM` closes
the listener immediately and gives the port back inside its grace period.

Routes: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, and
SSE streaming. What the engine is doing stays visible while it does it: `/health`
reports the draft head, the draft length and the turn-cache counters, and `--ui`
opens the terminal panel with the per-stage timings of the request in flight
(`make ui` shows that panel on a pty and asserts it never draws into a log).

## Pointing Crush at it

```json
{
  "$schema": "https://charm.land/crush.json",
  "options": { "request_timeout": 3600 },
  "disable_default_providers": true,
  "providers": {
    "qe": {
      "name": "qe",
      "base_url": "http://127.0.0.1:8097/v1",
      "type": "llamacpp",
      "auto_discover_models": true,
      "api_key": "qe"
    }
  }
}
```

`request_timeout` is load-bearing: it is an *idle* deadline on a stream, an unset
one is 120 seconds, and a long ingestion emits no parsed part while it runs.
`type: "llamacpp"` is the other half - its enricher reads `n_ctx` from
`/v1/models`, so the context window this process opened with becomes the window the
client plans with. `make probe-crush` drives exactly this block against a loaded
engine, including a tool call that has to come back as a tool result.

## Dials

Everything that could plausibly trade quality for time is a switch, was measured
against the thing it replaced in one load, and ships with the winning reading. The
ones worth knowing:

| variable | default | what it does |
|---|---|---|
| `QE_DYN_KB` | `64` | a lone token's walk hands its rows out in chunks of about this many KiB of weights from a cursor, so a P-core takes more rows than an E-core instead of waiting for it; `0` is the equal slice. Byte-identical either way |
| `QE_STEP_FUSE` | `1` | a step's walks that read the same input share one dispatch (the four GDN projections, the five attention ones, a mixer's pair, router beside shared expert), and the routed experts take the one-token int8 walk with per-route partials |
| `QE_I8_STEP` | `1` | a lone token's dense walk recodes its input to int8 once per dispatch and walks the IQ4_NL copy with `vpdpbssd`: 23.2 → 20.6 ms of a step, nll -0.03 over 400 positions |
| `QE_I4_MULTI` | `2` | passes of up to this many rows (a verify window) read the IQ4_NL dense copy too |
| `QE_ROWS_PAR` | `1` | a prompt chunk's per-token loops across the pool; byte-identical |
| `QE_DELTA_SIMD` | `1` | the gated delta rule eight state rows at a time; nll unmoved |
| `QE_MOE_DYN` | `1` | the grouped expert pass on cursors, its down summed in bucket order (no per-thread partials, 4.8 GB less RAM) |
| `QE_ATTN_BATCH` | `1` | a prompt chunk's attention as one selection dispatch and two per 32 tokens, instead of three a token; byte-identical |
| `QE_CHUNK_DYN` | `1` | the cursor for multi-row walks as well |
| `QE_I4_DENSE` | `all` | read the load-time IQ4_NL copy of the dense projections; `lean` holds the 194 hyper-connection mixers and the 31 matrices at the trunk's ends at `Q8_0`, buying back two thirds of the loss for 8 ms of a step; `0` is off |
| `QE_I4_HEAD` | `q8` | walk the output head from its `Q8_0` copy (`i4`, or `stored` for the file's own `Q6_K`) |
| `QE_I4_HEAD_WIN` | `1` | let a verify window read that copy too, which is what keeps k=1 speculation ahead |
| `QE_I8_FLAT` | `1` | the flat one-scale-per-row dense copy a chunked prefill walks three rows against |
| `QE_I8_MOE`, `QE_I8_RS`, `QE_I8_PROJ` | on | the int8 activation panel in front of the experts, the row scale, and the dense projections |
| `--spec-k N` | 0 | draft length for greedy requests: `1` with `--mtp` drafts off the head, `2` without it off the n-gram hash; off by default because at today's step speed neither beats the plain loop (`QE_SPEC_AUTO=1` restores the old resolution) |
| `--batch N` | `1` | drive N sequences through one walk of the weights: 2.15-2.20x the step they share and 1.79x a wave of eight short requests, for 0.060 nats and a per-client tok/s that gets worse (`make lane`) |
| `--walk-threads N`, `QE_WALK_THREADS` | every thread `--threads` has | threads a lone token's weight walk asks for. One kernel walking those bytes prefers 16 of 19 by 6.2 %; a step of them is 1.07x slower there, so the default is the whole pass and the knob is for a host whose bus is shaped differently (`make forward`'s width leg) |
| `QE_SMALL_PROJ_BATCH` | `1` | whether a projection under `--proj-chunk`'s 1 MiB threshold is asked of the pool once for the chunk or once for every token of it. 180 of this checkpoint's projections are under it, and the loop barriered a nineteen-thread pool 92,160 times a width-512 chunk for them, so the batch is 1.037x of an ingested token and byte for byte the same logits; `0` is the loop back (`make forward`'s small-projection leg) |
| `QE_EXP2_SIMD` | `0` | whether a gated-delta block's decay and one cell's attention weights use the eight-wide `exp2` the gates have always used rather than libm `expf`. It is 7.1-7.4x the element rate at 3.5e-06 relative, and a token asks for 114,048 to 322,992 exponentials, which at nineteen threads is 7 to 20 microseconds of an 88 ms step: measured 1.000x of ingestion and 1.00x of a step for 0.67 nats, so the dial ships off and the sixteenfold redundancy the table beside it removed ships unconditionally (`make forward`'s exponential leg) |

`make walk`, `make alu`, `make pool` and `make moe` need no checkpoint at all, so
the interesting arguments about this machine can be settled on yours.

## What this is not

- Not a GPU engine, and not a port layer for one.
- Not a throughput server, even with `--batch N`: that flag multiplies decode and
  nothing else, so two clients on 512-token prompts still cost 1.00-1.02x of the same
  two one at a time, and `make serve` still measures the wall a second client waits for.
- Not a general model runner: it runs `qwen4exp` checkpoints (48 layers, 2560 wide,
  512 routed experts with 10 active, the n-gram PLE table, hyper-connections, the
  block indexer), an OpenAI-compatible surface, and a chat template read from the
  file. Anything else it refuses at open rather than approximates.
- Not measured on your machine yet. Run `make alu`, `make walk` and `make report`
  first; the design numbers in this repository are host-specific on purpose, which
  is the only reason any of them are worth quoting.

## Layout

```
src/            20 C files: gguf reader, kernels, layers, MoE, attention,
                indexer, forward pass, sampler, drafter, spec loop, HTTP, server
include/qe.h    the whole interface, with what each function owes whom
bench/          the instruments: gemv, walk, moe, pool, alu, score, forward, head
docs/           DESIGN.md (the budget and the dead ends), ARCH.md (the shapes)
ideas.md        2,500 lines of what was measured, priced, and left off
MANUAL.md       status tables, every make target, every route, the crush notes
```
