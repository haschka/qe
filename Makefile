# qe - qwen4exp inference engine
#
# Note on -march=native: this host is a Core Ultra 7 265 (Arrow Lake-S), where
# AVX-512 is fused off in silicon, so the ISA actually available is
# AVX2 + FMA + F16C + AVX-VNNI + BMI2. Getting this right is not pedantry:
# building for a generic -mavx512f target produces SIGILL at runtime here, and
# building without -mf16c makes the fp16 dequant path roughly 3x slower.

CC      ?= clang
ARCH    ?= -march=native
OPT     ?= -O3 -fno-math-errno -fno-trapping-math -ffp-contract=fast \
           -funroll-loops -fomit-frame-pointer -fno-plt
CFLAGS  += -std=gnu17 -g $(ARCH) $(OPT) -Iinclude -Isrc -pthread -Wall -Wextra \
           -Wno-unused-parameter -Wshadow -Wno-format-truncation
LDFLAGS += -pthread -lm -ldl

# Every recipe here is a fuse, and this is what makes one a fuse rather than a
# suggestion: plain `timeout` sends SIGTERM and then waits for the child for as
# long as it likes - a process blocked in a syscall, catching the signal, or
# wedged in a thread join never notices, and the run is remembered as "the test
# hung" instead of "the test failed". -k escalates to SIGKILL, so the worst case
# any target here can cost is its own number plus the grace.
KILLGRACE ?= 10
TO := timeout -k $(KILLGRACE)

# CORE is everything the micro-benchmarks need. Keeping it separate means
# `make fast` still builds and runs when the engine layers are mid-edit. The
# tokenizer belongs here too: it reads metadata only, so a bench can hold real
# tokens without pulling in the serving stack. The n-gram drafter belongs here for
# the same reason - it reads no weights and no checkpoint, only tokens.
CORE := src/mem.c src/gguf.c src/kern.c src/kern_iq.c src/layer.c src/moe.c src/tok.c \
        src/ple.c src/forward.c src/sampler.c src/draft.c src/vision.c
ENG := src/json.c src/mtp.c src/http.c src/chat.c src/engine.c src/serve.c src/tui.c \
       src/turns.c
COREOBJ := $(CORE:.c=.o)
ENGOBJ := $(ENG:.c=.o)
BIN := bin/bench_kern bin/bench_gemv bin/bench_moe bin/bench_mtp bin/bench_json bin/bench_chat \
       bin/bench_route bin/bench_stage bin/bench_tok bin/bench_ple bin/bench_prefill \
       bin/bench_forward bin/bench_decode bin/bench_score bin/bench_pool bin/bench_prefix bin/bench_alu \
       bin/bench_sample bin/bench_turns bin/bench_walk bin/bench_cosched bin/bench_serve \
       bin/bench_batch bin/bench_vision

.PHONY: vision all clean test fast moe mtp head json chat sample pool turns tok http spec forward prefix serve lane score oracle cmp report gemv stage route ple prefill walk cosched demo sanitize alu probe probe-selftest probe-crush probe-crush-ckpt probe-discover shutdown ui ui-crush ui-crush-live ui-crush-dup batch decode probe-igpu

all: $(BIN) bin/qe-serve

bin:
	@mkdir -p bin

bin/bench_kern: $(COREOBJ) bench/bench_kern.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_kern.c $(LDFLAGS)

bin/bench_gemv: $(COREOBJ) bench/bench_gemv.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_gemv.c $(LDFLAGS)

# The chunked dense walk on synthetic tensors: the model's own projection shapes at
# every row block, and the same walk at Q8_0 against its smaller quantisations.
bin/bench_walk: $(COREOBJ) bench/bench_walk.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_walk.c $(LDFLAGS)

bin/bench_attn: $(COREOBJ) bench/bench_attn.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_attn.c $(LDFLAGS)

# Whether the expert walk and the dense walk are limited by different parts of the
# machine, which is the premise the pipelined-prefill idea in ideas.md 7.8 rests on.
bin/bench_cosched: $(COREOBJ) bench/bench_cosched.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_cosched.c $(LDFLAGS)

# The n-gram row gather against the real 26.8 GiB table: correctness of the
# hash and of the dequantisation, then what sixteen scattered rows cost.
bin/bench_ple: $(COREOBJ) bench/bench_ple.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_ple.c $(LDFLAGS)

# Chunked prefill: bytes per ingested token when one step carries T tokens.
# Runs at the machine's full width by default and counts the LLC misses of both
# core types; `./bin/bench_prefill <vol> 16 8` confines it to the performance
# cores, which is the only configuration where the counter can be cross-checked
# against a round of known size, so both are worth running.
bin/bench_prefill: $(COREOBJ) bench/bench_prefill.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_prefill.c $(LDFLAGS)

bin/bench_moe: $(COREOBJ) bench/bench_moe.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_moe.c $(LDFLAGS)

# Links src/mtp.c, which is the only engine layer with no dependency on the
# checkpoint, so the control loop stays testable while the model does not exist.
bin/bench_mtp: $(COREOBJ) src/mtp.o bench/bench_mtp.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) src/mtp.o bench/bench_mtp.c $(LDFLAGS)

bin/bench_json: src/json.o bench/bench_json.c | bin
	$(CC) $(CFLAGS) -o $@ src/json.o bench/bench_json.c $(LDFLAGS)

# The sampler takes a float array and a struct, so its contract is testable with
# no checkpoint and no vocabulary: which ids are reachable, the order the three
# filters run in, and that the top candidate survives all of them.
bin/bench_sample: $(COREOBJ) bench/bench_sample.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_sample.c $(LDFLAGS)

# The conversation contract. The stream parser is driven by ids, so its cases run
# with no vocabulary at all; the renderer is checked against the file's real
# control tokens when CHAT_MODEL names a checkpoint (metadata only, never
# weights), which is why `make test` runs it bare and `make chat
# CHAT_MODEL=$(CHECKPOINT)` runs the whole thing.
bin/bench_chat: $(COREOBJ) src/json.o src/http.o src/chat.o bench/bench_chat.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) src/json.o src/http.o src/chat.o bench/bench_chat.c $(LDFLAGS)

# The park/wake handshake of the thread pool, which needs no checkpoint and
# hangs rather than failing, so it runs under a fuse.
bin/bench_pool: $(COREOBJ) bench/bench_pool.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_pool.c $(LDFLAGS)

# Whether a step is limited by the memory controller or by the arithmetic in front
# of it, and what each core class is worth: the same bytes walked four ways - the
# shipped Q8_0 kernel, a byte-only read, an integer vpdpbusd dot, and bf16 - once
# per named set of cpus. `--cpus 0-7` is the performance half of a Core Ultra and
# 8-19 the efficient half; with no --cpus the pool is the engine's default, which
# names no cpus and so pins nothing. Seconds per row, no checkpoint.
ALU_REPS ?= 3
bin/bench_alu: $(COREOBJ) bench/bench_alu.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_alu.c $(LDFLAGS)

# The vision tower alone, against llama.cpp's own dump of the same image: the mmproj
# is 0.9 GiB and the trunk is never opened, so this runs in seconds.
bin/bench_vision: $(COREOBJ) bench/bench_vision.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_vision.c $(LDFLAGS)

bin/bench_tok: $(COREOBJ) bench/bench_tok.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_tok.c $(LDFLAGS)

bin/bench_stage: $(COREOBJ) bench/bench_stage.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_stage.c $(LDFLAGS)

bin/bench_route: $(COREOBJ) bench/bench_route.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_route.c $(LDFLAGS)

# One real token through the real graph: opens the checkpoint, runs the forward
# pass over a prompt, and prints the per-stage magnitudes next to the text it
# produced. Non-finite counts have to stay zero; the magnitudes are what catch a
# pass that is finite and wrong. It links the speculation loop (src/mtp.c) so that
# the draft head leg drives the same accept/rewind rules the greedy HTTP path does,
# against the real checkpoint rather than bench_mtp's synthetic target.
bin/bench_forward: $(COREOBJ) src/mtp.o bench/bench_forward.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) src/mtp.o bench/bench_forward.c $(LDFLAGS)

# Which reading of the reference this checkpoint was trained with is a question
# only loss can answer, so this asks it once per load: teacher-forced nll for
# every variant the forward pass exposes, a depth-probe loss curve that names the
# layer class that stopped helping, and the wall clock of a step by stage.
bin/bench_decode: $(COREOBJ) bench/bench_decode.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_decode.c $(LDFLAGS)

bin/bench_score: $(COREOBJ) bench/bench_score.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_score.c $(LDFLAGS)

# B sequences through one walk of the weights: the byte identity of a batched row against
# that same sequence walked alone, and the aggregate tokens per second the shared weight
# read is worth.
bin/bench_batch: $(COREOBJ) bench/bench_batch.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) bench/bench_batch.c $(LDFLAGS)

bin/qe-serve: $(COREOBJ) $(ENGOBJ) src/main.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) $(ENGOBJ) src/main.c $(LDFLAGS)

# The engine layers plus the serving stack, so this is the one bench that asks
# the engine a question rather than the forward pass: does a second turn reuse the
# sequence the first left standing, and does it answer identically when it does.
bin/bench_prefix: $(COREOBJ) $(ENGOBJ) bench/bench_prefix.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) $(ENGOBJ) bench/bench_prefix.c $(LDFLAGS)

# The same question a server asks and a forward bench never does: not how long one
# prompt takes but when the request behind it sees a first character. Driven through
# the engine's own request entry, one thread per client, on the theory that the
# queue in front of the network is part of the network.
bin/bench_serve: $(COREOBJ) $(ENGOBJ) bench/bench_serve.c | bin
	$(CC) $(CFLAGS) -o $@ $(COREOBJ) $(ENGOBJ) bench/bench_serve.c $(LDFLAGS)

# The eviction policy on its own, and that is the point: no forward pass, no
# checkpoint, no state to copy, so the 30-turn conversation replay below costs
# microseconds instead of the minutes `make prefix` costs.
bin/bench_turns: src/turns.o bench/bench_turns.c | bin
	$(CC) $(CFLAGS) -o $@ src/turns.o bench/bench_turns.c $(LDFLAGS)

%.o: %.c include/qe.h src/kern.h src/http.h src/json.h src/chat.h src/tables.h src/tui.h \
     src/turns.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Fast checks only: kernel correctness and bandwidth on synthetic data. Runs in
# a few seconds and never touches the 87 GiB checkpoint, so it stays cheap
# enough to run after every kernel edit.
fast: bin/bench_kern
	$(TO) 90 ./bin/bench_kern 60 16

# The routed-expert pass on a synthetic pool with the checkpoint's geometry:
# 1.19 GiB of weights, so unlike `fast` this one is a real bandwidth number.
# The thread sweep inside it is noisy above 8 threads; QE_MOE_REPEATS=5 turns it
# into best/median/worst per thread count, which is the only way to read it.
# The pool is the engine's width, and the sweep is relative to it: a full-width
# pool makes even the narrow rows pay for the workers they are not using.
moe: bin/bench_moe
	$(TO) 480 ./bin/bench_moe 420 auto

# Control-loop tests: microseconds, and the only thing standing between a
# speculation bug and fluent wrong output.
mtp: bin/bench_mtp
	$(TO) 60 ./bin/bench_mtp

# Request parsing: escapes, data URLs, and the rejections.
json: bin/bench_json
	$(TO) 60 ./bin/bench_json

# Request rendering and reply parsing. Runs in milliseconds and never reads the
# checkpoint unless CHAT_MODEL says which file's vocabulary to look at.
CHAT_MODEL ?=
chat: bin/bench_chat
	$(TO) 60 ./bin/bench_chat $(CHAT_MODEL)

# Which token comes back when the temperature is not 0. The distribution's shape
# is what is pinned, so this is the test that catches a filter that moved.
sample: bin/bench_sample
	$(TO) 60 ./bin/bench_sample

# Which rewind point a turn goes back to, and which one a new point may take a slot
# from. Thirty turns of a coding agent's conversation against caches of four, twelve
# and one point, plus the budget arithmetic, in microseconds and without a
# checkpoint. This is the gate that was missing when a four-point cache - two turns -
# shipped as though it were a policy: `prefix` asks the same questions of the real
# model and costs minutes and the checkpoint to do it.
turns: bin/bench_turns
	$(TO) 60 ./bin/bench_turns

# The pool with its workers parking between dispatches. A lost wake-up shows up
# as a decode step that never returns, so this is the cheap version of that: same
# handshake, no checkpoint, and a timeout that turns a hang into a failure.
#
# It also owns the thread-count measurement: "all" is every cpu in the affinity
# mask, "auto" is the engine's default (one core short). The hot-dispatch median
# of the two differ by tens of microseconds, which at the ~800 dispatches a token
# makes is tens of milliseconds a step.
POOL_GAP ?= 120
pool: bin/bench_pool
	$(TO) 200 ./bin/bench_pool all $(POOL_GAP)
	$(TO) 200 ./bin/bench_pool auto $(POOL_GAP)

# One core alone tells you the per-core limit; the two halves tell you whether
# twelve slow cores add bandwidth; the last two rows are what the engine does
# today, pinned by hand against the pool it actually builds.
alu: bin/bench_alu
	$(TO) 300 ./bin/bench_alu --cpus 0    --reps $(ALU_REPS)
	$(TO) 300 ./bin/bench_alu --cpus 8    --reps $(ALU_REPS)
	$(TO) 300 ./bin/bench_alu --cpus 0-7  --reps $(ALU_REPS)
	$(TO) 300 ./bin/bench_alu --cpus 8-19 --reps $(ALU_REPS)
	$(TO) 300 ./bin/bench_alu --cpus 0-18 --reps $(ALU_REPS)
	$(TO) 300 ./bin/bench_alu --reps $(ALU_REPS)
	@echo '   the recurrent state at its own size: an in-place read-and-write, ordinary'
	@echo '   store against streaming store. The gap is what a non-temporal store could'
	@echo '   buy the block stage, and docs/DESIGN.md 14 is the arithmetic it answers.'
	$(TO) 300 ./bin/bench_alu --span 128 --inner 4 --reps $(ALU_REPS) \
	  --kernels read,state,statent

# The tokenizer against the checkpoint's real vocabulary: exact round trips,
# cross-checks against the file's own token ids, and token counts for the kinds
# of prompts the endpoint actually receives. Reads metadata only.
tok: bin/bench_tok
	$(TO) 300 ./bin/bench_tok $(CHECKPOINT)

# Where the same bytes go faster: real Q8_0 rows placed on 4 KiB pages, THP and
# hugetlb, measured with the GEMV kernel and with a byte-only read pass.
stage: bin/bench_stage
	$(TO) 420 ./bin/bench_stage $(CHECKPOINT) 120

# How much the grouped MoE pass actually saves, which is exactly the routing
# overlap of real consecutive tokens: real token ids, real embedding rows, real
# router matrices. Reads ~650 MiB of embedding plus 5 MiB of router per layer.
route: bin/bench_route
	$(TO) 620 ./bin/bench_route $(CHECKPOINT) 600 2000 docs/DESIGN.md MANUAL.md src/tok.c tools/smoke_http.py

# End-to-end HTTP: starts a throwaway selftest server, drives both client
# scripts against it, then shuts it down with SIGTERM (the graceful path, so an
# ASan build would still report leaks). Never touches the checkpoint, so the
# refusal policy and the whole request surface stay exercised without 87 GiB of
# I/O - including --mtp, which is handed the 160 KiB head fixture rather than the
# 2.6 GiB sidecar, so the draft-head fields in /v1/models are asserted on every run.
# PORT=nnnn moves it off 8099, MTP_HEAD=path points it somewhere else.
PORT ?= 8099
MTP_HEAD ?= /tmp/qe-http-head.gguf
http: bin/qe-serve
	@python3 tools/make_mtp_fixture.py $(MTP_HEAD) >/dev/null
	@./bin/qe-serve --port $(PORT) --threads 8 --spec-k 4 --id qe-selftest \
	  --mtp $(MTP_HEAD) \
	  >/tmp/qe-http-server.log 2>&1 & \
	server=$$!; \
	trap "kill -TERM $$server 2>/dev/null" EXIT; \
	i=0; \
	while [ $$i -lt 40 ]; do \
	  python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1',$(PORT))))" && break; \
	  i=$$((i+1)); sleep 0.25; \
	done; \
	QE_PORT=$(PORT) $(TO) 300 python3 tools/smoke_http.py && \
	QE_PORT=$(PORT) $(TO) 600 python3 tools/fuzz_http.py

# Speculation measured against the network it runs on. The same greedy request is
# answered twice, once with a drafter proposing a window and once a token at a
# time, before either rate is printed.
#
# It runs that comparison twice, against two servers, because one claim cannot cover
# both questions. A window is a chunk, and a chunk takes the ingestion fast paths -
# the chunked projection walk, the int8 recodes, the grouped expert pass - while a
# plain decode step is a `T == 1` pass with none of them. Same model, two
# arithmetics, and greedy decoding parts at the first near-tie, so on the shipping
# settings the two answers are not the same bytes and never will be: what is asserted
# there is that a speculative answer is a *good* answer (the repeating prompt has a
# ground truth, a free-text answer may not be degenerate) and that windows pay, by
# acceptance and by rate. The first server therefore switches every one of those
# paths off, which leaves a row of a window and a decode step with the same
# arithmetic and makes byte-for-byte identity the assertion again - and that is the
# half that matters for correctness, because what it can fail on is the state a
# window leaves behind: the delta net checkpointed per row, and the two conv
# histories that are minimal rings rather than position-addressed caches.
#
# That first server also runs with the turn cache off (`--turn-cache -1`), and it is
# the part of this target that decides whether the state is right at all. With the
# cache on, the plain leg of a pair reads the boundary the speculative leg stored, so
# both sides take the checkpoint as their starting point and a checkpoint that does
# not hold the true state agrees with itself perfectly. Off, both legs start at
# position 0, and the comparison is a draft-and-rewind walk against a walk that never
# left the sequence - the only comparison that can see a boundary that is merely
# self-consistent. The cache is asserted on the second server, where it is on and a
# boundary wrong in that same way shows up as a turn continuing from the wrong place.
#
# Needs the checkpoint (a toy target already proves the rule in `make mtp`), so
# startup is minutes twice over and the fuse is long. CHECKPOINT=... points it at a
# different file, SPEC_PORT=nnnn moves it off 8123.
#
# No `--ctx`, deliberately: the engine's own default is the file's `n_ctx_train`, and
# the window a draft occupies is charged against that capacity (`mt <= ctx - np -
# spec_k`), so a run that narrowed it would be measuring a context the server never
# serves. A client like Charm Crush plans its turns against the trained window, and
# this is the test that would notice if speculation did not fit inside it.
SPEC_PORT ?= 8123
spec: bin/qe-serve
	@run_spec() { \
	  mode=$$1; shift; \
	  extra=; \
	  if [ "$$mode" = exact ]; then \
	    QE_CHUNK=1 QE_MOE_GROUP=0 QE_MOE_CO=0 QE_PROJ_MB=0 QE_I8_PROJ=0 \
	    QE_I8_FLAT=0 QE_I8_MOE=0 QE_I8_RS=0 QE_I4_DENSE=0 QE_I4_HEAD=stored; \
	    export QE_CHUNK QE_MOE_GROUP QE_MOE_CO QE_PROJ_MB QE_I8_PROJ QE_I8_FLAT \
	      QE_I8_MOE QE_I8_RS QE_I4_DENSE QE_I4_HEAD; \
	    extra="--turn-cache -1"; \
	  else \
	    unset QE_CHUNK QE_MOE_GROUP QE_MOE_CO QE_PROJ_MB QE_I8_PROJ QE_I8_FLAT \
	      QE_I8_MOE QE_I8_RS QE_I4_DENSE QE_I4_HEAD; \
	  fi; \
	  ./bin/qe-serve --model $(CHECKPOINT) --stage ram --port $(SPEC_PORT) \
	    $$extra $$@ >/tmp/qe-spec-$$mode.log 2>&1 & \
	  server=$$!; \
	  i=0; \
	  while [ $$i -lt 600 ]; do \
	    python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(s.connect_ex(('127.0.0.1',$(SPEC_PORT))))" && break; \
	    kill -0 $$server 2>/dev/null || break; \
	    i=$$((i+1)); sleep 1; \
	  done; \
	  QE_PORT=$(SPEC_PORT) QE_SPEC_MODE=$$mode $(TO) 1800 python3 -u tools/spec_ab.py; \
	  rc=$$?; \
	  kill -TERM $$server 2>/dev/null; \
	  while kill -0 $$server 2>/dev/null; do sleep 2; done; \
	  return $$rc; \
	}; \
	rc=0; \
	run_spec exact || rc=1; \
	run_spec fast || rc=1; \
	exit $$rc

# The live panel, asserted by giving it a terminal. It decides whether to exist by
# asking whether stdout is a tty, and every other harness here redirects stdout to a
# file - so both answers need covering: a panel that drew into a log would corrupt
# `make http`, and one that never drew on a real terminal nobody would ever see.
# A pty is the only way to ask, hence python rather than a shell recipe.
UI_PORT ?= 8092
ui: bin/qe-serve
	$(TO) 300 env QE_PORT=$(UI_PORT) python3 -u tools/test_ui.py

# What the client draws, asserted by giving it a terminal. A tool request that is
# parsed, executed and stored but never drawn is indistinguishable from a model that
# acted on its own, and only the bytes a terminal receives can tell them apart - so
# this replays recorded replies (`tools/testdata/replies`, captured from this engine
# by `tools/replay_sse.py --record-to`) at a throwaway client and reads three accounts
# afterwards: the terminal, the client's own session rows, and its log. No weights and
# no inference, because a reply that is already on disk is the same reply: the point is
# that a few bytes of framing are being looked at, and nothing that stages 87 GiB of
# weights is a comfortable way to do that. Skips, without failing, where crush is not
# installed.
UI_CRUSH_PORT ?= 8157
ui-crush:
	$(TO) 400 env PYTHONUNBUFFERED=1 python3 -u tools/test_ui_crush.py \
	  --port $(UI_CRUSH_PORT) \
	  --reply tools/testdata/replies/turn0-tool-first.sse \
	  --reply tools/testdata/replies/turn1-tool-second.sse \
	  --reply tools/testdata/replies/turn2-answer.sse \
	  --expect-calls 2 --must-draw-final FIRSTCALLMARK --must-draw-final SECONDCALLMARK \
	  --prompt "run the two commands we talked about, one at a time, then tell me they finished"

# The same assertions with a real model answering, which is the only way to see what
# was reported: two turns of one conversation, each asking for one command, where the
# second turn's call used to be parsed, executed, stored and never drawn, because the
# id it arrived with was the id the client already had a block under. That needs a
# checkpoint, and a checkpoint takes minutes to stage, so this points at an engine that
# is already up rather than starting one, and a turn is waited for by watching the
# terminal go quiet rather than by a sleep. `--max-tokens` is a reasoning model's
# number here: a turn that ends in a tool request does not fit in 4096.
UI_CRUSH_LIVE_PORT ?= 8098
ui-crush-live:
	$(TO) 1900 env PYTHONUNBUFFERED=1 python3 -u tools/test_ui_crush.py \
	  --server $(UI_CRUSH_LIVE_PORT) --secs 1800 --send-at 25 \
	  --max-tokens 32768 --expect-calls 2 \
	  --must-draw FIRSTCALLMARK --must-draw SECONDCALLMARK \
	  --prompt "Use the bash tool to run exactly this one command: echo FIRSTCALLMARK" \
	  --then "Use the bash tool to run exactly this one other command: echo SECONDCALLMARK"

# The same run against the bytes this engine used to write: the two turns above with
# their call ids numbered from 0 inside each reply, which is what a client that looks a
# tool call up by its id cannot tell apart from a single call that keeps being renamed.
# The point of this target is that it fails - the assertion in `ui-crush` is only
# evidence if something old makes it go red - so the recipe inverts the result and
# wants the id check specifically, not any other reason a run went wrong.
UI_CRUSH_DUP_PORT ?= 8158
ui-crush-dup:
	@$(TO) 400 env PYTHONUNBUFFERED=1 python3 -u tools/test_ui_crush.py \
	  --port $(UI_CRUSH_DUP_PORT) \
	  --reply tools/testdata/replies/legacy-duplicate-ids-turn0.sse \
	  --reply tools/testdata/replies/legacy-duplicate-ids-turn1.sse \
	  --reply tools/testdata/replies/legacy-duplicate-ids-turn2.sse \
	  --expect-calls 2 >/tmp/qe-ui-crush-dup.log 2>&1; \
	rc=$$?; \
	if [ $$rc -eq 0 ]; then \
	  echo 'FAIL the duplicate-id control passed, so the id check is not asserting anything'; exit 1; \
	fi; \
	if grep -q 'FAIL no two tool calls were handed the same id' /tmp/qe-ui-crush-dup.log; then \
	  echo 'PASS the duplicate-id control fails the id check, which is what makes ui-crush evidence'; \
	else \
	  echo 'FAIL the control went wrong for some other reason:'; tail -8 /tmp/qe-ui-crush-dup.log; exit 1; \
	fi

# Shutting down with a request in flight. Closing the listener is the easy half, and
# the two ways to get the rest wrong are a use-after-free of an engine a request
# thread is still inside and a server that never dies and keeps the port with it.
# Asserted without a checkpoint by holding a request open on a body that never
# finishes, which is the same position a generation is in as far as shutdown cares.
# SHUTDOWN_GRACE=nnn moves the deadline the server is started with.
SHUTDOWN_PORT ?= 8098
SHUTDOWN_GRACE ?= 3
shutdown: bin/qe-serve
	$(TO) 300 env QE_PORT=$(SHUTDOWN_PORT) QE_GRACE=$(SHUTDOWN_GRACE) \
	  python3 -u tools/test_shutdown.py

# The same surface against the real checkpoint: that a checkpoint engine generates
# instead of refusing, that the sampler knobs change the draw on that path (the
# selftest engine cannot show this - it speculates, and speculation applies no
# filters), and that a bad value is a 400 naming the knob. This pays the one-time
# load, so it is a tool and not part of `make test`.
probe: bin/qe-serve
	$(TO) 1800 env QE_PORT=$(PORT) python3 tools/probe_ckpt.py $(CHECKPOINT)

# The probe is a client, and a client bug costs a twelve minute load to notice, so
# the same file runs against the selftest engine first: everything about the HTTP
# layer is asserted for real, only the four knob-semantics checks need logits.
probe-selftest: bin/qe-serve
	$(TO) 300 env QE_SELFTEST=1 QE_PORT=$(PORT) python3 tools/probe_ckpt.py

# The client this server exists for, driven for real. A throwaway config, a
# recording proxy, and one ask that only succeeds if crush streamed an answer, got
# a tool request back, ran its tool, and reported the result. Against the selftest
# engine the tool checks skip (that generator has no control tokens to emit), so
# it asserts the stream contract and nothing else. CRUSH_PORT is the server; the
# proxy listens on the next port, so keep both clear of a running engine.
CRUSH_PORT ?= 8093
probe-crush: bin/qe-serve
	$(TO) 600 env QE_SELFTEST=1 QE_PORT=$(CRUSH_PORT) python3 tools/probe_crush.py

# The config an operator does not have to hand-tune: no model list, no
# context_window, provider type `llamacpp` - so crush lists /v1/models and reads
# the window out of the shape it looks for. Fails loudly if the answer keeps that
# number to itself.
probe-discover: bin/qe-serve
	$(TO) 600 env QE_SELFTEST=1 QE_PORT=$(CRUSH_PORT) QE_PROBE_DISCOVER=1 python3 -u tools/probe_crush.py

# The same against the checkpoint, which is the only way a tool request can appear
# at all. One load, then minutes of prompt: this is a tool and not a test. With a
# server already loaded, QE_EXISTING=1 QE_PORT=that-port skips the load.
probe-crush-ckpt: bin/qe-serve
	$(TO) 5400 env QE_PORT=$(CRUSH_PORT) CRUSH_RUN_S=4800 \
	  python3 tools/probe_crush.py $(CHECKPOINT)

# The same request-level checks against an instrumented server. This is the only
# build that can see a one-byte overrun in a request parser or an unaligned load
# in a kernel, and it is a separate binary on purpose: -fsanitize and
# -march=native -O3 do not belong in the same translation unit as a bandwidth
# measurement. SAN_PORT moves it off a port the normal selftest is using.
SAN_PORT ?= 8095
sanitize:
	$(CC) $(CFLAGS) -O1 -fno-omit-frame-pointer -fsanitize=address,undefined \
	  -fno-sanitize-recover=all -static-libasan -o /tmp/qe-serve-asan \
	  $(CORE) $(ENG) src/main.c $(LDFLAGS)
	@ASAN_OPTIONS=detect_leaks=1:abort_on_error=0 UBSAN_OPTIONS=print_stacktrace=1 \
	  /tmp/qe-serve-asan --port $(SAN_PORT) --threads 8 --spec-k 4 --id qe-selftest \
	  >/tmp/qe-asan-server.log 2>&1 & \
	server=$$!; \
	trap "kill -TERM $$server 2>/dev/null" EXIT; \
	i=0; \
	while [ $$i -lt 60 ]; do \
	  python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1',$(SAN_PORT))))" && break; \
	  i=$$((i+1)); sleep 0.5; \
	done; \
	QE_PORT=$(SAN_PORT) $(TO) 300 python3 tools/smoke_http.py && \
	QE_PORT=$(SAN_PORT) $(TO) 600 python3 tools/fuzz_http.py; \
	rc=$$?; \
	echo "instrumented server log:"; tail -20 /tmp/qe-asan-server.log; \
	exit $$rc

# Where the per-token budget comes from: the step table is summed from the
# checkpoint's own tensor table, and the GEMV rate is measured on its real
# K=2560 Q8_0 rows over a span big enough to leave the 30 MiB L3 behind, read
# out of the RAM arena the engine will read from. The deadline covers the
# one-time startup load, not the measurement. CHECKPOINT=... points it at
# another volume.
CHECKPOINT ?= /mnt/data/data/ai-models/llama-cpp/llm/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
# The nextn block these checkpoints ship with, as its own file: 2.6 GiB on top of the
# trunk's 85, opened resident or not opened at all. MTP=... points `make head` and
# `qe-serve --mtp` at another head; a head whose tensors do not match this trunk is
# refused on stderr and the trunk carries on without it.
MTP ?= /mnt/data/data/ai-models/llama-cpp/llm/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
gemv: bin/bench_gemv
	$(TO) 420 ./bin/bench_gemv $(CHECKPOINT) 16 360

# The PLE n-gram row gather: 16 hashed rows of the 26.8 GiB table per token,
# checked against the router's own IQ4_NL kernel and then timed.
ple: bin/bench_ple
	$(TO) 420 ./bin/bench_ple $(CHECKPOINT) 360 20000

# Same idea as the grouped MoE pass, for the dense projections: read a weight
# row once and use it for every token in the step.
# The number has to cover the load and not just the measurement: staging 87.2 GiB
# into RAM is seven minutes on this host, and the sweep over T=1..16 that follows it
# is seconds. A deadline that fires during the load is a deadline against a run that
# never got to measure anything, which is the one thing this target exists to avoid.
prefill: bin/bench_prefill
	$(TO) 1200 ./bin/bench_prefill $(CHECKPOINT) 16

# What the dense half of an ingested token costs and what would make it cheaper,
# measured on synthetic tensors because both questions are about a kernel and not
# about a model: the row block the walk should use, and whether smaller weights
# would be walked faster than Q8_0 at the widths a chunk runs at. No checkpoint.
walk: bin/bench_walk
	$(TO) 900 ./bin/bench_walk $(THREADS)

# Whether an expert walk and a dense walk are limited by different parts of the
# machine: one pool, one barrier, k threads on each, timed alone and together.
# Under a minute, and it is the measurement ideas.md 7.8's overlap prize assumes.
cosched: bin/bench_cosched
	$(TO) 300 ./bin/bench_cosched 240 $(THREADS)

# The forward pass on the checkpoint: greedy tokens, the diagnostics per stage,
# and the step time. FUSE=nnn bounds the whole run, and startup is minutes here,
# so it is set by default. REPEAT is the prompt length, in copies of one sentence:
# 26 of them is about a thousand tokens, which is the length the ingestion numbers
# in docs/DESIGN.md are quoted at and well below which an ingestion ratio is noise.
# The same prompt drives the shared-dequant walk A/B that this prints.
FUSE ?= 2400
REPEAT ?= 26
# Thread count for the targets that load the checkpoint. "auto" is the
# engine's own answer (qe_cpus() - 1); a number is honoured exactly, so
# THREADS=20 reproduces the older measurements.
THREADS ?= auto
forward: bin/bench_forward
	$(TO) $$(( $(FUSE) + 120 )) ./bin/bench_forward --model $(CHECKPOINT) --threads $(THREADS) \
	  --fuse $(FUSE) --max 24 --repeat $(REPEAT)

# The same pass with the nextn block open, which is the only place its two claims are
# made on a checkpoint: that the residual one position back is the one the head reads
# (its logits against the target's own, for one position), and that drafting changes
# how many target passes an answer costs without changing the ids. The id comparison is
# only legitimate with the width-dependent arithmetic off, so the leg runs that first
# and the shipping arm second, and only the first is gated.
head: bin/bench_forward
	$(TO) $$(( $(FUSE) + 900 )) ./bin/bench_forward --model $(CHECKPOINT) --mtp $(MTP) \
	  --threads $(THREADS) --fuse $(FUSE) --max 24 --repeat $(REPEAT)

# B sequences through the trunk in one walk of the weights, against B passes of one
# sequence. The rows have to be byte-identical on the arithmetic a batch shares no copy
# with, and the shipped one then says what the shared weight read is worth per token and
# what a multi-row chunk costs the answer. SEQ are the widths to run; the batched pass is
# opened once, at the widest of them, and a pass at B splits its cells B ways. That last
# pass then walks a row map that moves - sequences joining mid-conversation, siblings
# leaving, slots recycled - because a byte-identical row over the full map and a
# contaminated row of a half-empty one look the same to a client, and only one of them
# looks like a bug to this target.
SEQ ?= 1,2,4,8
batch: bin/bench_batch
	$(TO) $$(( $(FUSE) + 900 )) ./bin/bench_batch $(CHECKPOINT) --threads $(THREADS) \
	  --seq $(SEQ) --deadline $$(( $(FUSE) + 840 ))

# Sequence reuse over the conversation shapes a client actually sends: five of
# them, each with its second turn run once reusing what the first left standing
# and once cold, and the two continuations compared token id by token id -
# including the turn whose second prompt re-renders the first answer, which is
# the shape that has to be put back to a stored boundary rather than carried
# forward. A reused turn that answers differently is a wrong answer that happens
# to be fast, so the comparison is the test and the seconds it saves are the
# report.
prefix: bin/bench_prefix
	$(TO) $$(( $(FUSE) + 600 )) ./bin/bench_prefix --model $(CHECKPOINT) --threads $(THREADS) \
	  --fuse $(FUSE) --max 16

# What two clients cost. Same checkpoint, same engine, but requests instead of
# prompts: one request alone, then two with spacings that bracket the solo wall, then
# four at once, each one timed from arrival to first token and to last. NARROW and
# WIDE are the prompt widths, in tokens, and the answer length is MAX.
serve: bin/bench_serve
	$(TO) $$(( $(FUSE) + 900 )) ./bin/bench_serve --model $(CHECKPOINT) --threads $(THREADS) \
	  --fuse $(FUSE) --max $(or $(MAX),24) --narrow $(or $(NARROW),512) --wide $(or $(WIDE),2048)

# The same harness with N sequences in the engine, which flips its one ratchet: a wave that
# still costs the sum of its parts is a lane that never filled, so what gets asserted instead
# is the step - a step with N rows in it against N steps of one row - plus the two things a
# shared weight read could plausibly get wrong, which are a lane answered out of another
# lane's logits and a sequence recycled from the last request still holding a recurrence that
# changes this one's first token. The makespan is printed and not asserted because it is a
# function of how much of the traffic is ingestion, and a batch does not make an ingestion
# cheaper. BATCH is the engine's sequences; the lane's wave uses all of them and asks 64
# tokens each, and the window widens with BATCH unless --ctx says otherwise.
BATCH ?= 4
lane: bin/bench_serve
	$(TO) $$(( $(FUSE) + 900 )) ./bin/bench_serve --model $(CHECKPOINT) --threads $(THREADS) \
	  --fuse $(FUSE) --batch $(BATCH) --max $(or $(MAX),24) --narrow $(or $(NARROW),512) \
	  --wide $(or $(WIDE),2048)

# The instrument, and the only thing that can tell a finite wrong answer from a
# right one: teacher-forced loss per token per variant, greedy text, the depth-probe
# loss curve and the per-stage profile, all from a single load. CORPUS is text the
# reference engine produced; the per-token nll it prints is what goes next to that
# engine's own logprobs (`cmp` below), because an aggregate over 64 tokens is
# contaminated by positions the reference itself cannot predict and ranks nothing.
CORPUS ?= /tmp/llama-ref.txt
ORACLE ?= /tmp/llama-logprobs.tsv
SCORELOG ?= /tmp/score.log
# Prompt windows prefilled twice per ingestion arithmetic. Four of them is two
# minutes of prefill and the smallest count where "the greedy choice moved in
# N of M" says anything.
INGEST ?= 4
# The integrated GPU probe (ideas.md 7.32.4): one layer's routed experts walked on the iGPU from
# host memory, checked against a C reference, with --duty S for a CPU run beside it. The only
# target that links OpenCL; nothing else in the tree does.
probe-igpu: tools/probe_igpu.c | bin
	$(CC) -O2 -o bin/probe_igpu tools/probe_igpu.c -lOpenCL -lm
	$(TO) 300 ./bin/probe_igpu --tile 8 --rps 8 --reps 6

# Where a decode step's time goes - pool, barrier tail, serial - per stage, and a race of
# the decode-side switches in one load (ARMS=ship,nopf,...). FUSE=2400; it pays the load.
ARMS ?= ship
decode: bin/bench_decode
	$(TO) $$(( $(FUSE) + 120 )) ./bin/bench_decode --model $(CHECKPOINT) --fuse $(FUSE) \
	  --arms $(ARMS) --text ideas.md

# The vision tower alone against llama.cpp's own dump of the same image. Make the dump
# once with MTMD_DEBUG_EMBEDDINGS=$(VISION_REF) llama-mtmd-cli -m ... --mmproj $(MMPROJ)
# --image $(VISION_IMG) -p x -n 1; the trunk is never opened here, so this takes seconds.
MMPROJ ?= /x-data/llm/mmproj-Qwen3.8-Flash-Next-BF16.gguf
VISION_IMG ?= /tmp/vision_test.jpg
VISION_REF ?= -
vision: bin/bench_vision
	$(TO) 600 ./bin/bench_vision $(MMPROJ) $(VISION_IMG) $(VISION_REF)

score: bin/bench_score
	$(TO) $$(( $(FUSE) + 900 )) ./bin/bench_score --model $(CHECKPOINT) --threads $(THREADS) \
	  --fuse $(FUSE) --corpus $(CORPUS) --tokens 96 --gen 32 --probe --profile \
	  --ingest $(INGEST) 2>&1 | tee $(SCORELOG)

# Reference prose and its per-position logprobs, from the other engine, in one
# llama.cpp session that is up and down again inside the run. Owning the server is
# the point: 87 GiB each means the two engines take turns, and a script that
# forgets to stop one has just taken the machine off its owner.
ORACLE_N ?= 256
oracle:
	$(TO) 2400 python3 tools/oracle_capture.py --model $(CHECKPOINT) --n $(ORACLE_N) \
	  --corpus $(CORPUS) --oracle $(ORACLE)

# The two side by side: qe's per-token nll against the reference's, marking the
# positions where the reference is confident. --variant is spelled the way
# bench_score prints it, so 0x01 and not 0x001.
cmp:
	python3 tools/cmp_oracle.py --oracle $(ORACLE) --qe qe=$(SCORELOG) --variant 0x01

# Not a test: a transcript. Starts a throwaway selftest server, prints what
# each route answers, shuts it down. DEMO_PORT moves it off 8099 - note that
# 8080 (qe-serve's default) is already taken by another service on this host.
DEMO_PORT ?= 8099
demo: bin/qe-serve
	@./bin/qe-serve --port $(DEMO_PORT) --threads 8 --id qe-selftest >/dev/null 2>&1 & \
	server=$$!; \
	trap "kill -TERM $$server 2>/dev/null" EXIT; \
	i=0; \
	while [ $$i -lt 40 ]; do \
	  python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1',$(DEMO_PORT))))" && break; \
	  i=$$((i+1)); sleep 0.25; \
	done; \
	QE_PORT=$(DEMO_PORT) $(TO) 300 python3 tools/demo_http.py

# The measured budget plus whatever the checkpoint says it is, in about a
# second: metadata only, no weight bytes read. MODEL=... adds the checkpoint.
report: bin/qe-serve
	./bin/qe-serve --report $(MODEL:%=--model %)

test: fast mtp json chat sample turns pool moe

clean:
	rm -f $(COREOBJ) $(ENGOBJ) bin/*
