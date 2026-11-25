# qe

CPU-only inference engine for `Qwen3.8-Flash-Next` (`general.architecture =
qwen4exp`), built around measured DRAM bandwidth rather than guesses about it.
Read [docs/DESIGN.md](docs/DESIGN.md) first: it says what a token costs on this
host, which kernels were measured and thrown away, and exactly what is not
implemented. Two loader bugs found this session - the special-token table was
read with the wrong element type, and per-volume tensor data offsets were used
shard-absolute instead of shard-relative - mean that no kernel number produced
before those fixes was reading the bytes it claimed to; [DESIGN.md
section 11](docs/DESIGN.md) is the post-mortem.

## Status

| | |
|---|---|
| Decode step, where the time goes (`make decode`) | **built**: the pool counts its own dispatches, their wall and the master's wait for the last worker (`qe_pool_stats`), per stage; `bench/bench_decode.c` prints that table for a step, a prompt (`--pp N`) and a verify window (`--win N`), races any set of switches in one load (`--arms ship,old,nodyn,noab,...`) with every arm's logits compared, and prices each arm in nats over teacher-forced positions (`--nll N`). It is what found that a third of a decode step was barrier tail, and what every row of [ideas.md §7.30](ideas.md) is measured with |
| One dispatch per independent group of a step (`QE_STEP_FUSE`) | **on**: the four projections of a gated-delta block, the five of an attention block, a mixer's down and inject, the router beside the shared expert's gate and up, and the routed experts beside the shared expert's down each share one dispatch of the stage graph; 953 dispatches a step become 461. The routed experts of a step take the one-token int8 walk with per-route partials summed in route order, so the answer does not depend on scheduling (`qe_moe_step_begin`) |
| A step's dense walk on int8 (`QE_I8_STEP`) | **on**: the input recoded once per dispatch, the IQ4_NL copy walked with `vpdpbssd`; `block_proj` 23.2 → 20.6 ms of a step (80 GB/s), nll -0.031 over 400 positions against the step before §7.30 |
| A prompt chunk's attention (`QE_ATTN_BATCH`) | **on**: the per-token prologue on the master, then one dispatch for every token's indexer scores and selection and two per 32 tokens for the fused walk and combine, cut exactly as the per-token form cuts them - `make forward`'s ordered-walk-is-the-step-loop leg holds it byte for byte ([ideas.md §7.31](ideas.md)) |
| Integrated GPU (`make probe-igpu`) | **probed, not used**: the Arrow Lake iGPU does 2.96 TMAC/s of DP4a and reads host memory at 49 GB/s, a correct expert-stage kernel reached 110 GMAC/s (3x slower than the CPU), and while the iGPU is active the cores drop from ~4-4.8 GHz to ~2 GHz, costing a CPU ingestion ~30 % - so offloading the experts nets ~1.1x today and ~1.4-1.5x at best. Parked until a firmware power-balance setting changes that ([ideas.md §7.32.4](ideas.md)) |
| Ingestion on every core ([ideas.md §7.33-7.35](ideas.md)) | **on**: a chunk's per-token loops across the pool (`QE_ROWS_PAR`, byte-identical, 1.16x), the delta rule eight state rows at a time (`QE_DELTA_SIMD`, `block_core` 3.15 → 1.65 ms, nll unmoved), the grouped expert pass on cursors with a deterministic down (`QE_MOE_DYN`, `moe_ffn` 7.72 → 6.60 ms, 4.8 GB of partials no longer allocated). Ingestion **15.85 → 11.50 ms/token, 63 → 87 tok/s** at 2,048 tokens, 93 tok/s on `make forward`'s prompt, 0 FAIL |
| Terminal panel | ingestion tok/s live (3 s window) with an ETA, answer tok/s from the first answer token, passes/s gone, 80 columns, bar moving per layer - see [Watching it work](#watching-it-work) |
| Quantised GEMV kernels | Q8_0 74-83, IQ4_NL 60, IQ4_XS 52-54, Q6_K 44-46, IQ3_S 44 GB/s. Shape is the second axis and it is the cheap one: one span re-tiled through the same kernel is 76.6-77.3 GB/s at 1360 B rows against 62.9-63.5 at 340, so the 194 hyper-connection tensors - the worst-shaped matrices in the model, walked in the order a step reads them - run at **67.4-67.6 GB/s** (`make gemv`) |
| Routed MoE pass (10 of 512 experts) | median 28 GB/s at 20 threads (22.8-47.3 over five sweeps; 8 threads holds 24-25 inside 3%) over a 1.11 GiB pool, 2.22 MiB + 5.0 MiB router per layer |
| n-gram PLE gather (16 hashed rows/token) | 5.1 µs/token cold, 2.2 µs warm over the real 26.8 GiB table = **0.007 % of a decode step**; 68.3 Mrows/s at 16 threads (`make ple`) |
| Layer elementwise kernels (96 norms + gates + adds per token) | **0.30 ms of a 104 ms step (0.3 %)**, 12.2 MiB of activation traffic; SIMD sigmoid 9.1x the libm loop (`make fast`) |
| Chunked ingestion (T tokens per weight walk) | counted LLC misses on the dense projections: **489 → 33 MiB (−93 %)** and 12.1 → 5.2 ms per ingested token at T=16/16 threads, the shared-dequant difference held to 1e-3 of `sum|x*w|` against an exact long double dot rather than against the other rounded path, which has no scale (`make prefill`). In the engine the sharing comes from blocking, and that is a recent correction: the T calls one tensor needs used to run back to back and hope the 30 MiB L3 served rows 1..T-1, and the threshold that was supposed to select between the two (`QE_PROJ_MB`, documented as off by default) compared an unsigned byte count against 0, so it blocked everything and every measurement of "blocking buys nothing" had compared the blocked walk with itself. One prompt ingested both ways in one load says the walk is **1.51x** faster, 46.0 ms a token ordered against 30.5 blocked (`make forward` prints the pair, and [DESIGN.md §12](docs/DESIGN.md) is the post-mortem). The sweep runs to 512 now (eight and then sixteen were the cap only because the PLE queue was sized off it, and the queue is now defined as the width), and getting there found a wall the ceiling had never let anyone hit: the multi-row Q8_0 walk looped weight rows outside and token groups inside, so at T=512 every weight row re-read the whole 5.24 MB activation panel, 195 GMAC/s where the same shape tiled to the L2 budget holds 518 (`make walk`: `attn_qkv` 195 → 518, `ssm_out` 151 → 467, `hc_down` 153 → 405 at T=512). Until that was fixed a 512-token chunk was a 22 % *penalty*; with the rows tiled to a 256 KiB tile the curve falls the way the byte model says it should. The shape it found is not the one the old row claimed: 92.2 ms/token at width 1, 45.6 at 8, 41.4 at 16, 37.2 at 32, 35.1 at 64, 33.1 at 128, 32.2 at 256, 31.8 at 384, **31.1 at 512** - and 512 is the default now, which costs 4.8 GB of grouped-expert partials across the 48 layers and 210 MiB of chunk scratch. That is the load that ships (`/tmp/f9_wide.log`, 0 FAIL); the same sweep on the shipping engine, with the integer expert walk, its row scale, the integer projection walk and the expert stage in one dispatch all on, reads 93.6 ms/token at width 1, 36.0 at 8, 31.1 at 16, 27.1 at 32, 24.7 at 64, 23.8 at 128, and **23.2 from 192 through 512**, where it is flat to the printed decimal: four widths tie, that leg's verdict line names 192 the fastest of them, and the default is the widest of a tie rather than the winner (`/tmp/f25_wide.log`). The dense half is indeed flat above about width 7 - that reading was right - but it was taken with the expert pass running serially, and the expert pass is the half of an ingested token that keeps paying for rows, so the curve has a second, much larger knee and is still falling at the top of the table (§12 has the table, ideas.md §7.3b the tiling). Four switches later the same leg's sweep falls the whole way and stops at the default it ships with: 93.4 ms/token at width 1, 35.2 at 8, 29.2 at 16, 24.1 at 32, 20.5 at 64, 18.8 at 128, 17.8 at 256 and 384, and **17.3 at 512**, where the verdict line reads "the default is the fastest setting" rather than naming a tie (`/tmp/f29_wide.log`) |
| Speculation (MTP) control loop | **built, wired, measured, and off by default since [ideas.md §7.30](ideas.md)** - a two-row verify window is 1.6 of the new 64 ms steps, so k=1 at 1.6-1.7 ids a pass no longer beats the plain loop; `--spec-k 1` (with `--mtp`) turns it on, `QE_SPEC_AUTO=1` restores the old resolution. What follows is the account from when it was on at the shortest draft: `--mtp` stages the nextn head resident and it runs as one more layer of the trunk (`blk.48.*` + Q4_K `eh_proj` + a 497.3 MiB Q6_K `output.weight`), drafting 48 ids through it moved none of the ids greedy says, and drafting is never the thing that changes an id. Cost is the wall, not the drafter: a draft row is **0.129** of a step while a verified nine-row window is **4.35x** one, so acceptance has to beat the window and only does at the shortest draft - as shipped **1.06x** the plain loop at k=1 with a head (1.78 ids/pass, 78 % of drafts accepted), **1.05x** at k=2 with the n-gram hash, and 0.99x at k=8 for the two (`/tmp/head_e.log`) against a plain loop of 91.6 ms an id, the k=1 margin being owed to the verify window reading the head's load-time copy too (`QE_I4_HEAD_WIN=0` puts it back to 1.02x). The draft length is therefore resolved at open from the drafter that won (`QE_SPEC_K_AUTO`: head 1, n-gram 2); `--spec-k 0` is off and any length a caller names is honoured (`make head`, `MTP=`) |
| Throughput: N sequences through one step (`--batch N`) | **built, wired, and a decode multiplier and nothing else**: the KV cells divide per sequence while the recurrent state is replicated (eight sequences 0.96 GiB of state where one costs 0.19), one driver walks a lane table through `qe_forward_step_batch_rows`, and the step eight rows share is **2.15-2.20x** a step of one while the wave of eight 48-token requests they arrived in is **1.79x** the same eight served one at a time - prompts being read one at a time, two clients on 512-token prompts together costing **1.00-1.02x** of what they cost alone. On the arm where a batch is only scheduling, a row of a partly full, moving step is byte for byte the row its sequence walks alone (12 of 12 prompts prefilled mid-pass, 144 of 144 rows of steps of one to eight rows); what a shared walk costs is **0.060 nats** and 11.5 % of greedy at eight rows, the load-time dense copies stop being worth anything once a step is shared, and a draft window is refused for a batched pass because it is one sequence's whole state (`make batch`, `make lane`, [ideas.md §7.25](ideas.md)) |
| GGUF loader | 3 volumes, 1224 tensors, 87.2 GiB, derived type table, metadata-only open for `--report` |
| Per-stage walk width (`--walk-threads`, `QE_WALK_THREADS`) | **built, wired, measured, and settled at the width it replaced**: a lone token's dense `gemv` asks the pool for a width of its own, and the logits of a step are byte for byte one reading at every width (`make forward`'s width leg asserts it), because one output row goes to one thread in one kernel call. Six widths raced inside the engine put the pass width first - 1.07x slower at 16 of 19 threads, 1.17x at 13 - where the same kernel raced on its own prefers 16 by 6.2 %, so the default is every thread `--threads` has ([ideas.md §7.26](ideas.md)) |
| Small-projection batching (`QE_SMALL_PROJ_BATCH`) | **built, wired, measured, and on**: a projection under the chunk walk's 1 MiB threshold was walked once a token, and 180 of this checkpoint's projections are under it - 96 mixer injection matrices at 160 KiB, 72 gated-delta alpha and beta at 480 KiB, 12 indexer key projections at 640 KiB - so a width-512 chunk took 92,160 pool barriers where the same chunk's walkable projections take 180. One dispatch over the flattened (token, row) space is **1.037x** of an ingested token (16.44 → 15.86 ms/token, 60.8 → 63.0 tok/s), 1.13x of `block_proj` and 1.16-1.19x of the two mixers, and 1.00x of a step - which is what lets it ship on a byte-identity assertion instead of a nats measurement, since every GEMV kernel is a loop over output rows and `make forward`'s leg holds a chunk's logits byte for byte at both settings ([ideas.md §7.27](ideas.md)) |
| Exponentials (`QE_EXP2_SIMD`) | **built, wired, measured, and off**: the degree-5 `exp2` the 96 gates have always used is now in `src/kern.h` and reachable from the two places that also exponentiate by the ten thousand - a gated-delta block's per-head decay and one cell's `per` attention weights inside the fused walk - and it is **7.1-7.4x the element rate** (0.16 against 1.12 to 1.17 ns an element, `make fast`) at **3.5e-06 relative**. What it is worth is the element count, and the count is small: 114,048 exponentials a token at the selection a 600-cell context reaches, 322,992 at the 2051-cell ceiling, 0.14 to 0.39 ms of CPU, and a step is nineteen threads, so 7 to 20 microseconds of an 88 ms step is the whole prize - **1.000x** of ingestion (15.56 against 15.55 ms/token) and **1.00x** of a step for **0.67 nats** on the token the engine goes on to say (`/tmp/f41_wide.log`). The dial is therefore off. What it made free instead ships unconditionally: sixteen value heads of one key head each computed `exp` of the same float for the same token, 27,648 calls a token where 1,728 are that number, and one row of scratch per token is the same `expf` on the same input - byte-identical, and the precondition for widening anything ([ideas.md §6.6](ideas.md)) |
| Weight staging | whole checkpoint read once into private pinned RAM **at engine launch**: 87.2 GiB, `mlock`'d, 2 MiB pages, no root. An engine that cannot get it resident does not start; the load is never timed or rate-reported, because storage speed is not a quantity this project optimises |
| OpenAI-compatible server | `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, SSE, images as `data:` URLs |
| Forward pass for `qwen4exp` | **implemented, and it is a working model**: 3.56 nats/token teacher-forced over reference text, 0.442 nats where the reference scores itself 0.452 on the positions it is confident about, same argmax on 7 of those 7, and coherent greedy text (`make score`, `make cmp`) |
| Decode rate on this host | **64.0 ms/token, 15.6 tok/s** at 512 tokens of context and 66.5 ms (15.0 tok/s) at 4,096, against **130.7 ms/token** for llama.cpp b12110 on the same file and host: 2.04x (`make forward`'s generation leg reads 64 ms / 15.54 tok/s, `make decode` races it against the engine before [ideas.md §7.30](ideas.md) at 88.4 ms). The older figures in this row's history - 103-104 ms at the default pool, 88-91 after the load-time copies - are what the table below the line was measured against |
| Pool width | **one core short of the affinity mask**, not the whole mask: the same binary at 20 threads on 20 cpus takes 125.2 ms/token and at 19 takes 106.2 on the 64-position corpus, because a spinning pool with no core left to itself pays a scheduler timeslice at every one of the ~800 dispatches a token makes (`make pool`) |
| Dispatch shapes | one pool, two shapes. `qe_pool_run` is one function over N threads with a barrier at the end, which is what every dispatch of a decode step uses and what still costs the **1.4-1.8 us** handshake above. `qe_pool_run_items` runs a *stage graph* as a single dispatch: up to 32 items, each declaring how many slots it divides into and which items it waits for, with any worker taking the lowest-numbered ready slot of any item, so the barriers between independent stages disappear and a short stage lends its threads to the tail of a long one. It is what a chunk's grouped expert pass runs on (three items, where three dispatches and two barriers used to be), and on that pass alone it changed the cost by nothing measurable: 125.1 ms for the 512-row grouped row against the 124.2 ms the two-barrier form recorded, with the same top-k, forward-vs-serial and grouped-vs-separate checks passing (`make moe`, `/tmp/moe_co.log`). What it is used for in the engine is wider than one pass: a chunk's whole expert stage - the routed walk's three items, the recode of the activation both halves read, the shared expert's two projections, their silu and its projection - is one ten-item graph now instead of seven dispatches, worth **1.026x** of the queued schedule with the trunk left the same bytes, and it is on (`QE_MOE_CO=0`, `/tmp/f25_wide.log`). `make pool` is the gate for the shape: 60 four-item dispatches and 2460 slots asserting every slot ran once, nothing started on a claimed rather than finished predecessor, and two independent items really were resident at once, then 240 handoffs from a graph to an ordinary full-width dispatch with stack churn in between, worst round 0.02 ms (`/tmp/pool_graph.log`) |
| Row split across the pool | **a cursor, not a slice** ([ideas.md §7.30](ideas.md)): every walk whose output rows are independent - a lone token's GEMVs (`QE_DYN_KB`, 64 KiB chunks), a chunk's walks (`QE_CHUNK_DYN`), the routed experts of a step, the indexer's block scores, the attention units, the delta rule's state rows - takes its next chunk of rows from an atomic counter, so a P-core walks more of it than an E-core instead of waiting for one. Equal slices left **32.5 ms of an 89 ms step** as the master waiting at barriers for the slowest worker (`make decode`'s tail column); the cursor leaves 2.1. Byte-identical: a row is still one kernel call on one thread. The capacity-weighted static slice (`QE_SLICE=capacity`) stays an instrument and stays a loss |
| Prompt ingestion | `qe_forward_prefill` runs five hundred and twelve tokens of a prompt through one stage of a layer at a time and the head only for the prompt's last token, and every projection of 1 MiB and above - in this checkpoint, all of them - is swept once for the chunk instead of once per token: **16.3 ms per ingested token** on a 598-token prompt at the width the sweep ships, where the ordered form takes 41.9 blocked into one walk 25.2 and one token at a time takes 92.4 (`make forward` prints the sweep and the pair, `/tmp/f35_wide.log`; `/tmp/f25_wide.log` is the load before the flat dense walk, at 23.2), with 497 MiB of the 5568 a token reads never spent. `--proj-chunk -1` / `QE_PROJ_MB=0` selects the ordered form, and `make forward` asserts that it lands on the trunk and the logits of feeding the same tokens one at a time byte for byte, at every boundary a chunk could end at; the shipped sweep reassociates Q8_0 and, since the integer expert walk shipped, also changes what the experts multiply; `make forward` prints the two apart - the schedule alone worth up to 2.9 in a logit against a head whose largest logit was 20.2, 15 % of that range, and the shipping configuration 4.7 in a logit against 20.1, 23 % of its range (`/tmp/f25_wide.log`) - and neither is settled by that delta but by the next token and by the eight greedy steps of what the engine goes on to say (`bench_forward`'s fifth ingestion leg, identical id for id, both arithmetics passing it), against the 16 % at width 64 and 12 % at width 16 that the first load of this leg printed. Teacher-forced loss is not a ruler for this particular question: `bench_score` scores by stepping one token at a time, so it cannot see a chunk and reads the same 2.5369 nats at width 16 that it reads at 64. 23.2 ms is 0.22x a decode and not the 7x that ingested-token byte budget allows, because a chunk shares a tensor's bytes and its dequantisation but not its arithmetic. The kernel that turns one dequantised weight run into T accumulators exists for Q8_0: `qe_gemv_q8_0_multi` widens, converts and scales a block once and multiplies it against T token accumulators, worth **1.98x at T=2 and 3.75x at T=4** (`make alu`'s `shipped2`/`shipped4`, default pool) - where looping the single-token kernel over the same blocked walk scores 1.00x, because it reads the bytes once and dequantises them T times. It is wired into `qe_gemv_multi_t` and it is what the default now runs. `make alu`'s `multi1`/`multi2`/`multi4` rows are the same measurement on a synthetic walk, one Q8_0 span walked once for one, two and four tokens, **3.8x** and essentially flat in T - the statement that the dequantisation is free once shared and that the missing 4x was a kernel and not a cache. On the experts there was more dequantisation per byte to amortise (IQ3_S gate/up at 44 and IQ4_NL down at 60 GB/s against Q8_0's 74-83, ALU-bound rather than bandwidth-bound), they are 19.9 % of a decode step's bytes, and they were the same conversion unmade. They are not any more: `qe_gemv_iq4_nl_multi` and `qe_gemv_iq3_s_multi` dequantise one weight block and multiply it against a bucket's token accumulators (**1.63x** at two rows, **1.79-1.80x** at four and flat past four because that is what sixteen ymm registers hold rather than a depth anyone chose), and one vector pass for the eight group scales a block carries instead of eight scalar derivations, worth **+3.5 %** on the IQ3_S column of the three-type walk where the Q8_0 and IQ4_NL columns of the same interleaved runs read 1.005 and 0.999 (`/tmp/sp/ab2.log`, ideas.md §7.3d), checked against a double reference to 5.2e-06 and 1.6e-05 with a padded leading dimension and per-token-different rows (`make fast`), and `moe_block_chunk` runs a chunk's rows through the grouped expert pass behind them. End to end that is **1.53x** of ingestion where it ships - 45.4 → 29.6 ms per token at width 512, 1.40x at 64, 1.46x at 128, 1.28x at 32, at 15 % of the head's logit range and the same next token (`make forward` A/Bs four of them in one load) - because what grouping saves is the repeat read of an expert that a second row in the same window also picked, so the wider the window the more there is to save; the stage itself is half of an ingested token at the shipping width (10.85 ms of a 22.40 ms profiler sum at width 512 on the integer walk with one scale per activation row and the expert stage in one dispatch, all of it the default now, against 14.36 for the dequantising walk in the same leg, `/tmp/f25_wide.log`), which is the whole reason the width sweep found anything at all. A decode step is one row, so none of it is in the 105 ms an emitted token costs, and `qe_forward_set_moe_grouped` is the way back (§12) |
| 50 tokens/second on ingestion | **reached, and the bar was measured on this host rather than assumed.** Ingestion is now **87-93 tok/s** (10.7-11.5 ms/token, `make forward` and `make decode --pp 2048`, 0 FAIL, after [ideas.md §7.30-7.35](ideas.md)), 1.7-1.8x llama.cpp; the account below is of the cycle before, when it was **61.3 to 63.7 tok/s** at the width the sweep ships - 16.3 to 15.7 ms/token at width 512, eleven loads (`/tmp/f31_wide.log` through `/tmp/f41_wide.log`, nine with no failing leg) - against llama.cpp b12359's 51.33 at the same 512-row batch (`/tmp/llama_pp.log`), which is **1.19-1.24x ahead** where `/tmp/f30_wide.log` was 1.07-1.13x ahead. The switch that carried it this time is ideas.md §7.27's small-projection batch: 180 of this checkpoint's projections sit under the chunk walk's 1 MiB threshold and were walked once a token, so a width-512 chunk took 92,160 pool barriers where the same chunk's walkable projections take 180, and one ask over the flattened (token, row) space is 16.44 → 15.86 ms/token - **1.037x** of ingestion, 1.00x of a step, and byte for byte the same logits - `QE_SMALL_PROJ_BATCH=0` being the way back (`/tmp/f39_wide.log`). The switch before it is ideas.md §7.18's flat dense walk reaching the engine: 496 dense matrices requantised at load time to one scale per output row, walked three rows against one int8 activation load, worth 1.78 against 2.91 ms/token of projections and 14.92 against 16.00 ms/token of a whole ingested token, `QE_I8_FLAT=0` being the way back (`/tmp/f35_wide.log`). The two switches that carried it are the scale exit hoisted out of the Q8_0 walk's token loop and the signed dot behind it: 21.9 → 18.2 → 17.4 → 17.3 → 18.2 ms/token across five loads (`/tmp/f26_wide.log` through `/tmp/f30_wide.log`), `QE_I8_HOIST=0` and `QE_I8_SSD=0` are the ways back, and [ideas.md §7.17](ideas.md) is the account, including what the change costs the head and what that is worth measuring at all. Before those, this work took ingestion from 41.1 to **43.1 tok/s** (23.2 ms/token at the width the sweep ships, `/tmp/f25_wide.log`) - the last three of those steps being the row-scaled integer walk in front of the experts, the same walk in front of a chunk's dense projections, and that stage's seven dispatches becoming one graph, worth 14.36 → 12.04 → 11.07 ms of the expert stage as its two switches go on, 5.82 → 4.83 ms of `block_proj`, and 22.85 → 22.26 ms of a whole ingested token for the schedule - and left generation at 105 ms/token against 130.7 for llama.cpp on the same file and host. llama.cpp b12359, built with its asserts on, `-ngl 0 -t 20 -fa on`: **38.79** tok/s at a 64-token batch, 49.45 at 128, **57.30** at 256, 51.33 at 512, 50.26 at 2048 (`/tmp/llama_pp.log`, ideas.md §7.9). Half of that lead is a 512-row ubatch this pass only just grew into: it is **1.25x** ahead at an equal 64-row batch (llama.cpp's 25.8 ms/token, from its 38.79 tok/s, against our 20.5-20.7) and **1.07-1.13x** ahead at the 512-row batch it and we both run (19.5 ms/token against our 18.2 to 17.3), and both parts are kernels. What this row used to blame on bytes is not a byte problem at the width that ships: a 512-token window reads 106.6 MiB of *distinct* expert bytes per ingested token - a 2.5 ms floor at the scattered-read rate - and spends 7.27 ms in the stage, down from 10.85, so both halves are arithmetic now. The ratio this row used to quote for the expert half of that is withdrawn, because it is not a measurement of headroom: 207 GMAC/s against a dense Q8_0 row's 637, "3.1x", changes the format, the number of token rows sharing one weight read and the gather at the same time. ideas.md §7.13 raced the routed pass against the same MACs walked with nothing routed about it, in one process, and the whole of what any better walk of these formats can buy at the shipping width is **1.2-1.6x** - 205 GMAC/s routed against 284 for the same codebook walked uniformly and 330 for a uniform Q8_0 row at T=512 - and the walk that now ships is 1.18x of that instrument's routed leg. The dense half is where the unexplained gap is: 343 GMAC/s in-engine against the 518 the same kernel gives `make walk` on the same shape on its own, 1.51x, and §7.8's verdict stands that that is worth one instrument before it is worth a refactor - though the isolated figure has since moved further away rather than closer: 773 with the scale exit hoisted and the dot signed (§7.17), 1737 for the same walk reading one scale per output row (§7.18). ideas.md §7.8 priced the kernel item that has since been built and shipped - two output rows per activation load, so one dequantised block feeds two rows instead of one - on the expert walks (`/tmp/f18_wide.log`, `moe_ffn` 16.48 → 15.28 ms) and on the dense one (`/tmp/f19_wide.log`, `block_proj` 7.36 → 5.77 ms, ingestion 31.2 → 33.7 tok/s), 0 of 2,100,736 outputs moved (`/tmp/walk_pairs3.log`), and the one structural idea that could have crossed both - layer L+1's expert pass in flight while layer L's dense walk still owns the units - is measured dead **as a thread split**: `make cosched` puts the best split at 0.961x of doing both walks at the full width one after the other, and the worst at 0.711x, the machine being additive across these two walks (`/tmp/cosched3.log`; the 0.64 that made it look worth 1.5x was the same run measured against the two halves alone on half the machine each, which is not a schedule). The form a split could not test, one schedule whose threads go where the work is, is now in the tree (ideas.md §6.4), it is on, and it is worth more than the reading that first put it in the tree: the routed walk co-running with the shared expert's three projections and the chunk recode they both need, as one graph, was **1.005x** of the seven-dispatch schedule at width 512 while both walked floats (`/tmp/f12_wide.log`) and is **1.026x** with the recode carried as items of the graph instead of silently dropped by it - 4 passes of 4, byte-identical trunk, 512 tokens at width 512 (`/tmp/f25_wide.log`); §7.2b is the item that tried one walk of the quantiser per expert bucket instead of one per eight rows: 1.15x on uniform routing in isolation, 1.25x *slower* on the one big bucket the real router makes, 16 % of the stage in the engine at every cap that engages, so it ships switched off at `QE_PARK=0`. What the stage is bound by is now measured rather than assumed: four rewrites of the expert walk raced against it in one process say the codebook is worth nothing at the shipping width once the weights stop fitting in cache (0.84x, the leg reads 3.9x the bytes and hits the read ceiling), that an integer dot is 2.75x only while it pretends the scale is per 256 weights and **1.36x** when the exit is put where the format has it every 32, and that one activation scale per row instead of per 32 buys a further 1.23x and a quality decision - built since, and `make walk` returns 1.13x at eight token rows, 1.19x at sixty-four and **1.27x** at the 512 a chunk ingests at on IQ4_NL (1.21x on IQ3_S), with the Q8_0 column, which takes the row-scaled panel without changing its kernel, reading 1.00x as the control it is (`/tmp/walk_rs1.log`). Composite that is ~1.3x of this stage, and with the same walk in front of the dense projections and the stage in one dispatch it is 43.1 tok/s, so the loads per multiply, not the multiply, are the item, and the only structure that takes them below one is one int8 activation load feeding several output rows: a tiled integer gemm, which is what the reference engine runs on a chunk's activations (ideas.md §7.10, `/tmp/iq3probe2.log`). That load is in the tree and **on**. A chunk's activations are recoded to int8 once per layer and a whole expert bucket is walked against that panel with `vpdpbssd`, worth 12.49 against 15.13 ms of the expert stage (0.825x) and 24.35 against 27.14 ms of a whole ingested token in one load; the row scale on top of it is 11.28 and 23.16, and the sweep that measures the two switches together reads 27.5 against 23.8 (`/tmp/f22_wide.log`, `/tmp/f20_wide.log`) and the same two legs on the shipping engine read 14.36 against 12.04 against 11.07 with the stage in one dispatch (`/tmp/f25_wide.log`); the leg that first built it read 0.824x of the stage on `/tmp/f16_wide.log`, and `bench_score --ingest 5` prices what it charges over five 512-token windows of the corpus, each prefilled once per arithmetic: **1.10 nats mean and 4.69 worst** on the loss of the token that follows the window, with the greedy choice moved once in five (`/tmp/ingest2.log`). The bucket reordering this engine already ships, through the same instrument over the same windows, charges 0.29 mean and 0.49 worst and moves the choice none. Both stay inside the half-of-the-head's-range bound a change of *arithmetic* is held to (30 % against 11 %), and on the 598-token prompt of `make forward` the two changes together read 34 % of the head's range where the schedule alone has always read 15 % - which is why that leg prints the schedule and the arithmetic apart now and holds them to a third and to half respectively. So the speed ships, `QE_I8_MOE=0 QE_I8_RS=0 QE_I8_PROJ=0` are the three ways back off the arithmetic and `QE_MOE_CO=0` off the schedule, and what would make the price smaller is a panel that carries less activation error at the same multiply rate. The row scale is not it either: it is the same multiply rate at *more* activation error, 0.5-0.8 % of the largest output of a walk on synthetic activations, and it ships on the strength of the 0.903x of the panel it replaces plus the `make forward` legs that hold it to 35 % of the head's own range with the same next token and the same eight greedy steps. Its own nats price is **not measured**: `bench_score --ingest 5` prints that table, but the reference text it scores is a `/tmp` file that is no longer on this host, and both it and its per-position logprobs come back with `make oracle CORPUS=/tmp/llama-ref.txt`. Until that table exists the row scale is priced in ms/token and in the head's range, which is a weaker gate than the one the expert recode came through (`/tmp/ingest2.log`) - and the 1.1002/4.6859 above were measured on the panel before it, so they are not the row scale's price either. Two further items are priced and unwired: widening the expert bucket measured negative as built (parked caps 12/16/24/32 read 28.65/29.69/33.08/32.39 ms/token against the shipped cap's 28.35, 1-17 % *worse*, because the integer forms stop at eight token rows and the pairing already collects the tail of a group that a wider cap used to collect differently - the cap 12 that was 1.6 % faster than the default before the pairing is 1.1 % slower after it; the same dead end measured from the other side, a ten-row integer form for both codebooks, is built, and handing it a bucket is 8-25 % worse because a bucket of sixteen becomes three passes of an expert's weights where eight makes two, while handing it a 512-row dense walk is a wash (§7.14, `/tmp/moe_t10.log`, `/tmp/wide_walk.log`), and the distance past 50 tok/s is the integer form of that same load, one int8 activation vector feeding several output rows, since the float form of it is now on and worth 1.19x on the dense walk and 1.11-1.14x on the IQ3_S expert one, while the IQ4_NL half of that pair is now gated **off** at 0.92-0.98x and the gate is per format (`QE_IQ4_NL_PAIR_MIN_T`, ideas.md §7.29) (ideas.md §7.11). That integer form is built for both expert formats, measured, and off: two output rows against one int8 load are 0.95-1.04x of the walk they replace and 0.98-1.01x of the grouped stage with zero outputs moved, because an integer dot spends about seven operands and a pair shares three of them while the sixteen accumulators two rows against an eight-token group need are the whole register file (`QE_I8_R`, ideas.md §7.12, `/tmp/walk_tile6.log`). What gets round that file is a three-row tile rather than a second row, and that is built and **on**: **1.84x** of the walk that shipped at the width a chunk ingests at, 1.63x of `block_proj` inside the engine, wired behind a requantised copy of the dense matrices made at load time and refused with two printed numbers when it does not fit (§7.18, `/tmp/f35_wide.log`) - and it is also the ceiling of its own family, because five row counts are now raced against it twice and three output rows against four tokens ties the only other tile the budget leaves at every width from sixteen up (1.4 % ahead of `r5t2` at the chunk width in one run, dead level in the next, while the two forms the budget excludes are 48-63 % off), the budget being `R*T + R + 1` vectors in a sixteen-wide file, with the winning tile reading 1.00x of itself with its weights resident at 512 tokens in both runs, which is a walk at an issue-rate ceiling rather than one the bus is holding (§7.28, `/tmp/walk_diag.log`, `/tmp/wide_tile5g.log`) |
| Attention cache | **q8_0 keys and values**, one byte per value with one scale per head of one cell: the cache itself is **12.2 KiB** per token of context against the 48 KiB f32 costs, and the whole per-token footprint - that cache plus the indexer's keys - is **19.7 KiB** against **55.5**, so one sequence at 8k context is **270 MiB** of state instead of **557** (`qe_forward_state_bytes`, printed by the banner and by `make score`, both numbers measured on this checkpoint). What it costs, on the 193 positions of the wide corpus where the reference engine is confident: **0.294 nats against 0.292** for the same build run with `QE_KV_F32=1` - and one more position where its top-1 is the reference's wanted token, not fewer. At the 256 positions that pass reaches the two are the same speed (median 107.9 against 108.0 ms/token, five variants each), because the gather the cache is read for grows with the context and that corpus barely has one; `--kv-f32` asks for full precision, and `/v1/models` says which one is answering |
| Attention walk | **one pass over the selection per KV group, softmax inside the pass.** Twenty-four query heads share two KV heads twelve ways, so the head-at-a-time walk read every row of a 2 MiB-per-layer gather twelve times to produce twelve answers that were available from one pass, and wrote a 197 KiB score array per layer on the way. `qe_attn_walk` puts the twelve heads inside the cell and the exponentials in registers; the selection is emitted ascending (`sel_ascending`), so the same gather is a forward stride with runs of four cells rather than a heap's random tour; and the pass is split across the pool the way flash-decoding splits one (`units = nchunks x n_kv_heads`, each unit keeping its own exponent base and sum of weights, folded by `qe_attn_combine`). Against the form it replaces, one thread and one layer: **1025 us against 1429 at the 2051 cells this indexer's ceiling produces (1.39x)**, 12.30 against 17.15 ms over the twelve QSA layers of a token, which the pool shares nineteen ways into **0.65 ms of a 103 ms step against 0.90** - under one per cent, which is why the rate does not move and why the row is about bytes rather than ms/token (`make test` prints the shape table). It is not bit-identical: an ascending, chunked sum reassociates an `exp`-weighted sum, measured 1.1e-06 between the two forms and 1.6e-06 from a double reference, so it entered through the nll gate - over 2999 positions **1.8358 against 1.8356 nats**, 1696 against 1702 top-1s, 108.9 against 109.4 ms/token, both walks in one 87 GiB load because `make score` runs the pair (`qe_forward_set_attn_fused`, and `QE_ATTN_FUSED=0` or `--attn-head-walk` for the old form), and `make forward`'s chunked-against-token-at-a-time identity legs are byte-identical either way |
| Is it ALU-bound? | **No, at the shapes a decode step reads.** The shipped Q8_0 GEMV runs at **96-99 %** of a walk over identical bytes with the arithmetic deleted, and at 97 % of a flat 64 B-aligned read: 75.9 against 78.9 against 76.2 GB/s at 512 MiB and 19 threads (`make alu`). The arithmetic does bind once data is cache-resident - 68 % of the read at 32 MiB, 23 % at 8 MiB - and a token reads 5568 MiB into a 30 MiB L3, so that regime is not this one. Every rewrite that looked plausible went and measured itself: AVX-VNNI `vpdpbusd`, a split weights/scales layout, two rows in flight, pinning workers by `cpu_capacity`, building an IQ3_S codebook index with vectors instead of the scalar byte chain (**1.49x slower** on the walk, and 2.04x at T=1, ideas.md §11's `/tmp/sp/ab1.log`), and a streaming store for the recurrent state (72.2 against the ordinary store's 74.0 GB/s - the delta rule reads the row before it writes it, so there is no fetch to skip). All of them are dead ends on the record: [DESIGN.md §1.2](docs/DESIGN.md) is the section, `tools/sweep_alu.sh` the driver |

One decode step reads **5.44 GiB (5568 MiB)** of weights, computed from the
checkpoint's own tensor table by `bin/bench_gemv` rather than estimated, and it
is the divisor behind every rate here: 66.6% dense Q8_0 projections, 19.9%
routed experts (10 of 512), 8.9% Q6_K lm_head, 4.3% F32 router logits. The same
inventory with a draft head open (`--mtp`) adds 2728 MiB, of which 497 is a Q6_K
`output.weight` the head reads once per draft row.

Single-stream decode is predicted at **11.4 tok/s** (88.0 ms/token: measured
per-type kernel rates summed over that table). The pass now runs, so the prediction
has something to be wrong against: a real step is 105 ms (`/tmp/f9_wide.log`), 19 % off
the budget, and `--profile` says where. The hardest measured ceiling on the real bytes is a
Q8_0 GEMV over a 512 MiB span of the staged weights at the real row shape
(`make gemv`, `QE_STAGE` unset so the arena is the one the engine uses): **80.0
GB/s at 16 threads** (79.6 at 20, 56.5 at 8), i.e. **13.7 tok/s**.

What this paragraph used to add on top of that - **16.5 tok/s**, or **19.9 tok/s** at
the ceiling, with an MTP drafter accepted at 2.76 tokens per step - is withdrawn, and the
part that was wrong was not the acceptance number but the assumption under it: that a
verified draft row is free because a window shares the model's bytes. At the width a chunk is
ingested at, one step is 108.1 ms, one draft row 14.6 (**0.136** of a step), and a nine-row
window whose logits are all computed 481.2, which is **4.45 steps** (`/tmp/f40_head.log`).
`cost = (window(k+1) + k x 14.6 ms) / ids(k)` therefore has almost no `k` that beats the plain
loop: as shipped it is **1.06x** at k=1 with the head resident (1.78 ids a pass, 78 % of passes
keeping a drafted token) and **1.05x** at k=2 with the n-gram drafter, against 0.99x at the k=8
this file used to default to (`/tmp/head_e.log`) - where the same three arms read 1.19x, 1.05x and
0.75x over a step 16 % slower than this one (`/tmp/f40_head.log`), the difference being that a
cheaper step shrinks the denominator of every ratio measured against it, and the k=1 win survived
only because a verify window went on to read the same copy of the head a step reads. At short
context it thins further (`/tmp/f38_mtp_short.log`). The draft length is consequently the drafter's, resolved at open (head
1, n-gram 2, `--spec-k 0` for off), so single-stream decode is still the 105 ms figure and
13.7 tok/s is the ceiling that ships; [DESIGN.md §4](docs/DESIGN.md) is the measurement
and ideas.md §7.19 the account.

## Build and test

```sh
make all          # kernels, benches, bin/qe-serve
make test         # fast (kernels), mtp (control loop), json, chat, sample, turns, pool, moe
make fast         # kernels only, ~25 s, never touches the checkpoint
make sample       # the sampler: which ids are reachable, in what order the
                  # three filters cut, and what a draw costs against
                  # exponentiating the whole vocabulary (no checkpoint)
make turns        # the rewind cache's two decisions: thirty turns of a coding agent's
                  # conversation against caches of twelve, four and one point, and the
                  # arithmetic of what a budget buys (no checkpoint)
make alu          # is the step memory-bound or ALU-bound, per core class: the same
                  # bytes walked as the shipped GEMV, as a read, and as a flat
                  # aligned read, over one P core, eight P, twelve E, and the
                  # engine's width (~2 min; tools/sweep_alu.sh does the same across
                  # spans, which is where the cache-resident regime appears)
make moe          # routed-expert bandwidth, ~3 min budgeted
make walk         # the chunked dense walk on synthetic tensors: the model's own
                  # projection shapes at every row block, and Q8_0 against IQ4_NL and
                  # IQ3_S at the widths a chunk runs at (~1 min, no checkpoint)
make cosched      # can the two walls share the machine: a dense Q8_0 projection and an
                  # IQ3_S expert bucket walk on disjoint thread groups of one pool, each
                  # measured alone and then together (~1 min, no checkpoint)
make pool         # the dispatch handshake at every cpu vs the engine's width: the
                  # two medians differ by 40x, which is ~19 ms of a decode step (~1 min)
make gemv         # per-token budget + DRAM GEMV rate off the staged checkpoint (~2.5 min:
                  # it pays the one-time load, then never reads the disk again)
make route        # which experts the real router actually picks, over real text
make ple          # n-gram PLE row gather against the real 26.8 GiB table (~2.5 min)
make prefill      # bytes per ingested token, one token at a time vs a T-token chunk (~2.5 min)
make forward      # what a chunk is worth on the real checkpoint: the width sweep, the ordered
                  # against blocked pair, and one A/B leg per arithmetic switch, each held to
                  # the head's own range and its next token (FUSE=2400, REPEAT=26, ~40 min:
                  # it pays the load)
make head         # the same pass with the nextn draft head resident (`MTP=`): what a step, a
                  # draft row and a verified window each cost, the pairing the drafter seeds
                  # from, the k sweep and its acceptance histogram, and the n-gram arms
                  # (FUSE=2400, ~55 min: it pays the load, plus 2.7 GiB of head)
make batch        # B sequences through one forward step, on three arithmetic arms: a row of a
                  # partly full, moving step against that sequence walked alone, the nats and
                  # greedy agreement a shared walk costs, and the byte-level rewind leg
                  # (FUSE=2400, ~20 min: it pays the load)
make lane         # the same multiplier through the engine's request entry with `--batch N`:
                  # eight requests arriving together, their aggregate against the same eight
                  # one at a time, and the ingestion-shaped mixes where it is worth nothing
                  # (BATCH=4 FUSE=2400, ~35 min: it pays the load)
make prefix       # turn reuse over five conversation shapes, against the checkpoint: that a
                  # turn put back at a rewind point answers with the same token ids a cold
                  # read produces, and what it saved (~15 min: it pays the load)
make serve        # what a second client waits for: one request alone, then two with its own
                  # prompt at spacings around that wall, then the same prompt resent, then
                  # four at once - each timed from arrival to first character and to last,
                  # against bars at four heights (~30 min: it pays the load)
make demo         # transcript of what every route answers (not an assertion)
make http         # starts a selftest server, runs tools/smoke_http.py + tools/fuzz_http.py
make ui           # the live terminal panel, on a pty and off: what it must
                  # draw on a screen and must never draw into a log (~1 min)
make shutdown     # SIGTERM with a request in flight: that it exits by itself, says which
                  # way it left, and gives the port back (SHUTDOWN_GRACE=, ~20 s)
make probe        # the same routes against the real checkpoint: that it generates, and
                  # that the sampler knobs change its draw (~12 min: it pays the load)
make probe-selftest # the HTTP half of those checks against the selftest engine, so a
                  # client bug costs five seconds rather than a load
make probe-crush  # the real charmbracelet client through a recording proxy: that it
                  # streams an answer, and with probe-crush-ckpt that the tool call it
                  # gets came back as a tool result and the file got written
                  # (~5 s selftest; minutes per turn against the checkpoint, which
                  # is what `QE_EXISTING=1 QE_PORT=8097` is for - it points the same
                  # probe at an engine that is already loaded instead of loading one)
make probe-discover # the same client with no model and no context_window in its config,
                  # so the window has to come from /v1/models or the check fails (~10 s)
make sanitize     # the same checks against an ASan/UBSan server, which is the only
                  # build that can see a one-byte overrun in a request parser (~2 min)
make tok          # tokenizer round-trips against the checkpoint's own vocabulary
make score        # teacher-forced loss per variant + greedy text, one load (CORPUS=, SCORELOG=)
make oracle       # reference text and its logprobs from llama.cpp; stops the server again (ORACLE_N=)
make cmp          # qe's per-token nll next to the reference's (ORACLE=, SCORELOG=)
make report       # measured budget; add MODEL=/path/to/volume1.gguf for the checkpoint
```

`-march=native` matters: this host is a Core Ultra 7 265, where AVX-512 is
fused off, so a generic `-mavx512f` build SIGILLs at runtime.

Every target above runs under a wall-clock fuse, and every fuse escalates: `make`
sends `SIGTERM` at the deadline and `SIGKILL` ten seconds later (`KILLGRACE=`),
because a plain `timeout` waits on a process that is blocked in a syscall or caught
the signal, and the difference between the two is whether a stuck run costs its
budget or costs your afternoon. The targets that load the checkpoint take their own
`--fuse` as well (`FUSE=1500`), which is what prints the numbers it had when the
deadline fired instead of nothing.

## Where the weights live

The checkpoint is on ZFS, and that decides everything about how it is read. The
pool's ARC is hard-capped at 1 GiB (`zfs_arc_max=1073741824`) against an 87.25
GiB file on six spinning disks in raidz2, so a demand-paged `MAP_SHARED` mapping
is not a slow cache on this host - it is no cache, and every expert row a token
reads costs a stripe seek. There is exactly one honest reading path: read every
byte once, sequentially, at startup, into memory that is never reclaimed.

```sh
./bin/qe-serve --port 8099 --model .../…-00001-of-00003.gguf   # default: --stage auto
./bin/qe-serve --port 8099 --stage ram --model …               # name the backing explicitly
./bin/qe-serve --report --model …                              # headers only, no weights read
```

| `--stage` | what happens | |
|---|---|---|
| `auto` (default) | hugetlbfs arena if a pool was reserved, else private RAM arena. If neither fits, the engine **exits with the two numbers that decided it** rather than serving off the array | |
| `ram` | 87.2 GiB read in, `mlock`'d (`RLIMIT_MEMLOCK` is unlimited here), 2 MiB transparent huge pages | the route |
| `huge` | as `ram` on top of a root-reserved 2 MiB pool; the arena file survives the process, so a restart reuses the bytes instead of reading them again | opt-in, needs root |
| `QE_STAGE_DIR=/dir` | the same trick without reserving anything: the bytes land in an ordinary tmpfs file, so a restart finds them already resident (`volume 2: 46.4 GiB already resident in /mnt/ram, nothing to load`) and gives 4 KiB pages back in exchange | optional, wants a tmpfs >= 90 GiB |
| `mmap` | refused for serving: it is a disk seek per expert row. Accepted only with `--report`, which reads headers and nothing else | metadata only |

There is no serving configuration that reads the checkpoint between tokens, so
there is nothing to time there either: the loader reports where the bytes ended
up (`ram+thp`, `hugetlb`, `ram+file`) and how much of it was reused, never a
device rate.

Nothing in the default path needs root, reserves memory away from the machine,
or optimises disk access: THP-by-`madvise` covered 68.4 GiB of `AnonHugePages`
during an 87 GiB load with no compaction stall, and `mlock` keeps it there.

A hugetlb pool is not needed for any of this: THP-by-`madvise` gave the 2 MiB
pages, `mlock` pinned them, and all a reservation adds is restart warmth, which
`QE_STAGE_DIR` also provides without taking memory away from the machine. It is
refused, with a reason, if the directory cannot hold the volume, because writing
past the end of a full tmpfs through a shared mapping is `SIGBUS`.

The engine banner and `GET /v1/models` (`weight_stage`, `weights_resident`,
`weights_pinned`, `weight_stage_gib`) both say which of these is serving you, so
a client can tell a pinned arena from a tmpfs one. It cannot report a third
option: every engine answering that request has the whole checkpoint resident,
because the ones that could not are not listening.

## Serve

```sh
# selftest engine: real HTTP surface, synthetic tokens, labelled as such
./bin/qe-serve --port 8099 --spec-k 4 --id qe-selftest

# the real checkpoint: same routes, and it generates
./bin/qe-serve --port 8080 --fuse 600 \
  --model /mnt/data/data/ai-models/llama-cpp/llm/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --mmproj  /mnt/data/data/ai-models/llama-cpp/llm/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

```sh
curl -s localhost:8099/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qe-selftest","stream":true,"max_tokens":32,
       "messages":[{"role":"user","content":"hello"}]}'
```

`--port` is worth setting explicitly: 8080, the default, is already taken by
another service on this host, and `qe-serve` exits with `listen :8080: Address
already in use` rather than joining it. `make demo` uses 8099 and prints a
transcript of every route instead of asserting on it (`tools/demo_http.py`).

`--fuse SECONDS` arms a wall-clock deadline before anything reads the
checkpoint; the deadline has to cover the one-time 87 GiB load, so it is counted
in minutes. `SIGINT`/`SIGTERM` close the listener immediately, and the request in
progress is stopped at its next token rather than waited out; what shutdown does
wait for, up to `--shutdown-grace` (default 5 seconds), is for the handler thread
to be out of the engine, because freeing the engine underneath one is a
use-after-free on the one path every harness takes. Outlive the grace and the
process exits anyway - the weights are the kernel's to reclaim - rather than
standing there holding the port and 87 GiB, which is what `make shutdown` pins.
`--ctx N` is the sequence capacity, and it is a memory setting as much as a token
limit: what one request holds is the per-sequence recurrent state plus 19.7 KiB per
token of context, measured 270 MiB at 8192 and printed for any width by `make score`
(the per-piece breakdown is `docs/ARCH.md` §5, and the server banner prints the
figure the engine opened with). That default is no constant: it is the file's own
`n_ctx_train`, 262144 here, which is why the banner prints the number instead of
leaving it to be inferred from the absence of a flag. Asking for all of it costs
5.0 GiB of cache before a token is generated, and `docs/DESIGN.md` §7 is what a long
context costs in seconds: 74.9 ms per token of prompt, which is the same order as
emitting one.

`--turn-cache MiB` is the budget for going back. A turn leaves two rewind points
behind - where the sequence stood when the prompt had been read, and where it stood
when the answer had - and a later prompt that begins with one of them, token for
token, starts there instead of reading the conversation again. The budget is a number
of points rather than of mebibytes: 112.6 MiB of sequence state plus a vocabulary of
logits makes one point about 115 MiB here, and the default asks for twelve of them -
six turns' worth of both boundaries - which is what the flag leaves unset to get
(`docs/ARCH.md` §5 is the arithmetic, `docs/DESIGN.md` §7 why a copy that small is
exact). Left unset, `--turn-cache` takes a fifth of the memory that is free once the
weights are resident, never less than those twelve points and never more than half of
what is free nor more than 8 GiB, and the banner then says what it bought in the units
that matter:

```
qe: turn cache: 12 rewind points at 115.6 MiB each, 1387 MiB chosen from the memory free at open (--turn-cache to set it)
```

A budget that cannot hold twelve points says so too, with the number of turns it does
cover, because a cache that is on and too small looks to a client exactly like a cache
that is broken. Both decisions behind it - what a budget buys, and which point a new one
is allowed to take a slot from - live in `src/turns.c` and are replayed without a
checkpoint by `make turns`. A negative `--turn-cache` turns reuse off, which is the switch to reach
for when a second turn answers differently. What going back costs is the copy, measured
rather than estimated: 8.9 ms for the two boundaries a turn takes, against the 4.9 s of
reading they replaced on the turn `make prefix` runs (`docs/DESIGN.md` §7 has both
numbers and why a single core costs 48 ms of them). Whether a given turn used it is in
the response: `qe.measured.cached_tokens`, plus `qe.measured.rewound` for the case where
the sequence was put back rather than carried forward, and `qe.measured.snapshot_ms` for
what the copy cost. When a turn does *not* use it, the per-request log line distinguishes
the two reasons: `cached=0 agreed=0` is a prompt that shares not one token with any
position the engine can go back to - a client that re-renders the front of its own
transcript - while `cached=0 agreed=9820` is a cache that ran out, which is what
`turn_cache_short` on `/v1/models` reports for the process as a whole.

`POST /v1/completions` takes an array `prompt` as a batch: at most 16 strings,
run one per engine call, answered with one choice per element and
`choices[i].index == i`. `stream: true` with an array is a 400 (a batch has one
SSE stream and no way to say which prompt a token belongs to), as are an empty
array, a non-string element, and a 17th entry.

Sampling is the same four knobs the other server on this host takes, on both
endpoints and on the command line: `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0`
starts the engine there and a request overrides it with `temperature`, `top_p`,
`top_k` and `min_p`. The filters apply in the order top-k, min-p, nucleus, under
whatever the checkpoint's own `general.sampling.*` keys say, and `make sample` is
what pins each one's effect. Absent means the engine's default; the off spellings
are `top_k: -1` (or `0`), `top_p: 1` and `min_p: 0`; anything out of range is a 400
that names the knob. One limit the source does not hide: the speculation loop draws
from the target's raw softmax and applies no filters (`qe_spec_opts`), and it is the
selftest engine that runs through it, so on a server started with `--id qe-selftest`
the four knobs are parsed and checked but do not change the draw - while on a server
started with `--model` they do, which is what `make probe` asserts against real
logits rather than asserting it nowhere.

Reasoning has one knob and three spellings. `reasoning_effort` takes `xhigh` (the
default, and what `high` means for this model), `medium`, `low`, `minimal`, and
`none`; `enable_thinking: false` and `thinking: {"type": "disabled"}` are the same
door from the other side; and a system turn whose text ends in `/no_think` closes
the block as well, because that is what charmbracelet's client writes on the
request it gives forty tokens for. An effort field in the body outranks the
marker, and an unknown level is a 400 listing the levels that exist. What length a
reply was allowed to use comes back as `qe.measured.budget`, which matters because
`finish_reason: stop` does not distinguish a short answer from an ended one: an
absent `max_tokens` means the 4096-token default ("until the model stops" arrives at
the same place for most turns), and an ask larger than that is honoured up to the
window the server opened with - the checkpoint's 262144 - rather than cut to a
constant the client never chose. The ceiling is therefore the sequence, and the
budget reports what the reply actually ran under.

Replaying a transcript has one rule and one switch. An assistant turn that arrives
with reasoning text is rendered with that reasoning block; one that arrives without
it is rendered with no thinking block, because the empty block a chat template
writes for a turn nobody thought in is the shape the model reads as *thought of
nothing*, and a client that never sends thinking back - charmbracelet's does not -
would then replay an assistant that never thinks. `--empty-thinking-blocks`
restores the template-faithful form, one empty block per replayed turn, and exists
so the two prompt shapes can be measured against each other rather than to be
turned on by default; the banner prints which rule is on. A turn that did carry
reasoning is replayed with it either way, subject to the request's own
`preserve_thinking`, whose default keeps the whole conversation's reasoning.

What that costs is the other half of the same number: at 74.9 ms a token
(`docs/DESIGN.md` §7) a 4096-token thinking block is 5 minutes of a serialised
engine, 16384 is 20, and a client that asks for the whole window can hold every
other request behind it for hours. That trade belongs to the operator, which is why
it is now a client-side choice instead of a constant in the engine.

The agent loop is the expensive shape, and it is worth pricing before pointing a
client at this: ingesting a prompt token costs the same order as emitting one. One
real conversation from the reference client was 11,758 prompt tokens on its first
request and 11,944 on the tool follow-up, and they took 862 s and 859 s with
`cached=0` on both - fourteen and a half minutes a turn, because the follow-up arrives
with the assistant turn rendered by the client rather than as this engine fed it, which
is not something a sequence carried forward can match. That is what `--turn-cache` is
for: the follow-up goes back to where the two prompts still agreed and reads only what
the client added on top. `make prefix` measures that shape on this checkpoint - a
102-token second turn that re-renders the first answer took 8.801 s read from the
beginning and 3.884 s put back at token 71, 8.9 ms of copying, and the same sixteen
token ids either way. Extrapolated to the follow-up above, 11.7 thousand tokens not read
at 74.9 ms is 875 s against the 859 s it measured; that last sentence is arithmetic, the
two-turn numbers are the measurement, and one half of the claim without the other half
is a fast wrong answer (`docs/DESIGN.md` §7).

## Watching it work

Started from a terminal, `qe-serve` draws a panel over its own stdout and updates it as
the numbers move: how far through the prompt the engine is right now, how many tokens
the answer has reached and at what rate, what the process has done since it started, and
what the turn cache is holding.

One frame of it, mid-ingestion, on the 80-column terminal it is laid out for:

```
 qwen38-flash-next Q8_0+IQ4_XS 87.4 GiB stage=ram+thp ctx=262144 :8097    8m25s
 ──────────────────────────────────────────────────────────────────────────────
 INGEST ▸ ███████████████░░░░░░░░░░░░░░   6,024/11,758   62.4 tok/s  1m32s left
 EMIT   ▸      0 tok ·   0.0 tok/s · 96.5 s          chat chatcmpl-6af879a10004
 ──────────────────────────────────────────────────────────────────────────────
 total    184,312 ingested · 1,204 emitted · 6 requests · 23,502 reused
 cache    3 of 12 rewind points · 345 MiB · 2 hits · snapshot 8.9 ms
 qe: chat a10003 14:02:11  in 14+11744 31.2 t/s  out 186 15.4 t/s  tool_calls
```

What each number is, because the panel used to print two that were not what they said:

- **INGEST tok/s** is prompt tokens read over the seconds spent reading them: while a
  prompt is in hand, a window over the last three seconds, so a prompt that slows as its
  context grows says so as it happens; once it is read, the average over the whole
  prompt. Tokens the turn cache supplied are not counted as read. `left` is the rest of
  the prompt at the current rate. The bar moves layer by layer inside a 512-token chunk
  (`qe_forward_set_progress_fine`: tokens carried by the layers done so far, never an
  estimate), so it advances several times a second instead of once every eight.
- **EMIT tok/s** is answer tokens over the seconds since the first of them, not since
  the request arrived: the old figure divided the answer by its prompt's ingestion as
  well, which is how a 15 tok/s answer after a long prompt was drawn as 0.3.
- **passes/s and ms/pass are gone.** They counted prompt tokens and answer tokens on one
  clock, which matches neither half.
- The history row is the request in one line that fits: its id's last six characters,
  the time, `in` as tokens read + tokens reused and the read rate, `out` as tokens and
  their rate, and how it finished. The full stderr line below is unchanged.

Rates are recomputed twice a second and held in between, so the digits do not churn;
the request clock is the only field that ticks every frame. Every row is composed to the
width the terminal reports - the bar takes what the numbers leave, the request id is the
first thing to be cut and is marked when it is - and nothing is drawn in the last column.

The panel exists because an agent prompt holds the engine for minutes before a byte
goes out, and until now the only evidence during that time was that the process still
existed. That is the difference between `ingesting 6,024 of 11,758` and `hung`, and
74.9 ms per token of prompt (§7) means the question gets asked. `--ui off` silences it,
`--ui on` forces it, and the default is to draw only when stdout is a terminal - which
is the same test that decides where the per-request log line goes, so a redirected
server is exactly the server that has always been: no escape sequences, one line per
reply on stderr, and `make http` byte-identical to what it was. `make ui` asserts both
halves, on a pty and into a file, and the half on a pty asserts the rendering
properties too - every frame beginning on the row the last one began on, no row
appearing twice, no row past the reported width, one full repaint per resize - which is
how the crawl this panel used to have got fixed rather than looked over.

Nothing a model wrote is ever drawn. The counters are integers, the request ids are
made in `src/serve.c`, and the history lines come from the request log, which has
never carried reply text; every byte is scrubbed again on the way to the screen, for
the reason in `AGENTS.md`.

One line per reply, and it carries the whole shape of the turn:

```
qe: chat chatcmpl-6af879a10003 prompt=11758 cached=11744 agreed=11744 dropped=0 answer=186 passes=11944 752.10 s 0.25 tok/s 15.9 passes/s content=742 thinking=1810 calls=1 finish=tool_calls stream
```

`content`, `thinking` and `calls` are what the turn was made of - answer bytes,
reasoning bytes, tool requests - and they are there for the one failure a client
cannot report for itself: charmbracelet's client discards an assistant message that
holds neither answer text nor a tool request and says nothing to the user. A turn
that reasoned at length and never closed its thinking block is exactly that case,
because text before the block closes leaves as reasoning and cannot be recalled into
the answer. So a turn that finished holding nothing a client could have shown - no
content, no calls, not cut short by a budget, reader still connected - prints a
second line:

```
qe: WARN chat chatcmpl-6af879a10004 finished with no answer and no tool call: 1912 thinking byte(s), 311 token(s), finish=stop. A client that discards an empty assistant message shows this turn as nothing happening
```

That is the difference between "the work happened and the client did not draw it"
and "the model answered with a thought", and only the second is a model problem.

## Where a stream stops waiting

A stream whose reader stops reading is dropped rather than waited for: 30 seconds
on a full send queue, and the queue is 64 KB - about a minute of deltas - precisely
so that the timeout is reachable within a turn. A bigger queue is not safer: with
the host's 4 MiB default a reader that stopped reading absorbed a whole 2631-token
answer (435 KB of stream) and the engine generated to the end of the turn for
nobody, and a quarter megabyte was only marginally better because Linux doubles
what you ask for. The reason to bound it is not politeness - generation is
serialised, so an undetected dead reader is the engine producing tokens nobody will
read while every other request queues behind it.

## Pointing crush at it

One key in crush's config is load-bearing, and its absence reads as a broken
server. `options.request_timeout` bounds each LLM request; for a stream it is an
*idle* deadline, reset whenever a parsed stream part arrives (`internal/agent/
request_timeout.go`), and unset means `DefaultRequestTimeout` - 120 seconds
(`internal/config/config.go`). Fourteen minutes of ingestion emits no part, so the
turn dies at exactly 120 s with `LLM stream received no data for 2m0s`, in the UI as
*The model stopped sending data for 2m0s*, while this engine is still reading the
request it was given. Nothing this server writes fixes that: a deadline credited per
parsed part is credited nothing by an empty delta or an SSE comment, which was
measured before it was concluded (both against a 3 s deadline, both unfinished, see
`docs/DESIGN.md` §7). Say it in the config instead - `3600` for one hour, `0` to
switch it off:

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

`type` is the second half of the story. The `llamacpp` provider has an enricher that
asks `GET /v1/models` for `data[].meta.n_ctx`, falling back to `n_ctx_train`, so the
window this process opened with becomes the window the client plans with and
`context_window` never has to be written down. `openai-compat` - what you get when
the key is omitted - has no enricher, so the window stays 0, and 0 means no auto
summarisation and a context meter that reads nothing. A `context_window` written by
hand has to be that number or smaller: a client that believes a smaller one trims
history this engine would have taken. `default_max_tokens` is worth 16384-32768 - a
turn that ends in a tool request does not fit in 4096, and on an engine that ingests
at 74.9 ms a token at full context - 59 at a prompt of a thousand - the whole window is
a 5.5 hour turn. `make probe-discover` drives
exactly the block above, with no model named and no window named, and `make
probe-crush` the same client with both written by hand - both against the selftest
engine in seconds. `tools/probe_crush.py` is also the harness that catches a reply
that forgets `finish_reason`, which makes fantasy retry the same minutes-long
request four times.

The third thing it catches is an id handed out twice. A tool request's id is the
only handle a client has for the call: it keys the block of interface the call is
drawn into and the table its result is filed back under, and a client holds both for
the length of a *conversation*, not of a reply. Numbering calls from 0 in every turn
therefore costs two things at once, and both were reproduced against the real client
by replaying one recorded reply as two turns of one conversation: with `call-0` in
both, a single tool block appears whose name changes under the reader and no second
one ever does, and the results collapse to whichever `call-0` was read last; with
distinct ids, two blocks appear, each keeping its own result. So call ids are built
from the id of the request that produced them (`qe_turn_set_id_prefix`, which makes
them unique per request and per process), and every `tool_calls` delta carries the
`index` a reader accumulates a call's arguments under, which is how a second call in
one turn stays a second call.

Two smaller things worth knowing from the same run. The banner prints the window it
actually opened, because the default is the checkpoint's trained length and is
therefore invisible on the command line:

```
qe: context: 262144 tokens of 262144 trained
qe: stream keepalive: an SSE comment every 15000 ms of silence
qe: thinking replay: no block for an assistant turn that carried none (--empty-thinking-blocks)
```

The keepalive line names a comment, written on a timer while the engine is busy
with a prompt, so that the stream is still producing bytes the client can read even
though it has nothing to say. It exists to notice a reader that went away inside a
minutes-long ingestion - otherwise the whole turn is generated for nobody, at
5.44 GiB of DRAM traffic a token - and it is off (`--sse-keepalive-ms -1`) if that is
worth less than the bytes. It is a comment rather than an empty `"delta": {}` chunk
because the two are equal against a client's idle deadline and unequal everywhere
else: a comment is parsed and never reaches the message assembler, while an empty
delta is a data frame that a client is entitled to finalise as an empty message -
which is one way for a turn to arrive as nothing. Every `text/event-stream` head
therefore also says `Cache-Control: no-cache` and `X-Accel-Buffering: no`, because
one intermediary that buffers the stream delivers the whole turn at the end, which
looks to the reader exactly like an engine that is not answering. The keepalive is
not a fix for the deadline above, and it is not what makes a long transcript fit:
when prompt and answer do not both fit the window, the front of the prompt is
dropped, which used to happen silently and now comes back as `qe.measured.dropped`
and `dropped=` in the request log, so a turn that lost the system prompt says so.

When a client shows nothing for a reply the engine knows it wrote, read the client's
own source rather than guessing at it: a Go module proxy will serve the exact
revision a binary was built from, with the `+dirty` suffix dropped.

```sh
go version -m "$(command -v crush)" | grep -m1 'charmbracelet/crush'
# mod  github.com/charmbracelet/crush  v0.97.2-0.20261002085448-73d1a50a8b07+dirty
curl -o /tmp/crush.zip \
  https://proxy.golang.org/github.com/charmbracelet/crush/@v/v0.97.2-0.20261002085448-73d1a50a8b07.zip
```

The `+dirty` suffix is the build's, not the proxy's, and comes off before the URL.

That is how the id rule above was found: three lines of the client's interface code
say that a tool call's block is looked up, stored, and filed by its id, which is the
whole of why a reused id draws nothing. `make ui-crush` is the other half of the
method - `tools/replay_sse.py` records one real reply and hands it back byte for
byte, so a client reproduces in the time it takes to type a prompt and costs no
inference, and `tools/test_ui_crush.py` reads three accounts of the same run: the
terminal's bytes, the rows the client stored, and its own log. The rows are the ones
that look fine. `make ui-crush-live` is the same tool against an engine that is already
serving a checkpoint, typing one prompt per turn and waiting for the terminal to go
quiet between them, which is the shape the bug needed: one turn per call, two turns,
and the assertion that both calls were drawn.

## Layout

```
src/mem.c        staging arenas (anonymous, THP, hugetlb), thread pool, qe_fuse deadline
src/gguf.c       split-GGUF reader: whole checkpoint into RAM at open, then mlock
src/kern*.c      quantised GEMV (Q8_0/Q6_K/Q4_*/IQ4_NL, IQ3_S/IQ4_XS)
src/layer.c      the elementwise half of a layer: norms, gates, rope, silu
src/forward.c    qwen4exp itself: hyper-connections, delta-net, indexer+attention, MoE, PLE
src/ple.c        n-gram PLE: row hashing over the 320 M-row table, gather, async FIFO
src/moe.c        routed expert pipeline over a (experts, N, K) pool
src/tok.c        the checkpoint's own byte-pair encoder, and its special tokens
src/sampler.c    temperature/top-k/min-p/top-p and the greedy path
src/mtp.c        speculative draft/verify loop with rejection sampling
src/json.c       parser + base64 for request bodies
src/http.c       HTTP/1.1, thread-per-connection, SSE
src/engine.c     the one seam between HTTP and a model
src/turns.c      which rewind point a turn goes back to, and which one a new point may
                 take a slot from
src/serve.c      OpenAI-shaped endpoints
bench/           kernels, gemv against the checkpoint, MoE, MTP, JSON, the scorer, the
                 rewind cache
tools/           HTTP smoke test, fuzz pass, table generator, the oracle, the
                 context-cost sweep, a redacting log viewer, and the client that
                 drives a real agent (`probe_crush.py`)
```

## Is it a model?

One checkpoint at a time, because 87 GiB each on a 125 GiB box leaves no room for
the other one, and the reference engine is only ever run while qe is not loaded:

```sh
make oracle CORPUS=/tmp/llama-ref.txt   # reference text + its per-position logprobs
make score  CORPUS=/tmp/llama-ref.txt SCORELOG=/tmp/score.log
make cmp    ORACLE=/tmp/llama-logprobs.tsv SCORELOG=/tmp/score.log
```

`make cmp` prints qe's per-token nll next to the reference's for the same token
ids, and marks the positions where the reference is confident. That per-position
comparison is the only instrument that decides anything here: 64 tokens of prose
contain positions the reference itself cannot predict, and an aggregate over them
ranks readings the wrong way round - it is how "delete the PLE" once looked like
the answer. [ref/AUDIT.md](ref/AUDIT.md) is the running record of what has been
settled, by which measurement, and what is still open.

Every number in this repository is a measurement on this host, and every
measurement names the command that produces it.
