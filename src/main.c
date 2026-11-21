/* qe command line.

   Two modes: serve the OpenAI-compatible endpoint, or print the budget this
   build is tuned against and exit. Both arm the hard fuse first: the checkpoint
   is 87 GiB on a spinning disk, and a run that silently turns into a ten-minute
   I/O stall is worth less than one that fails loudly. */
#define _GNU_SOURCE
#include "qe.h"

#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(const char *argv0) {
    printf("usage: %s [options]\n"
           "\n"
           "  --model PATH        GGUF volume 0 of the checkpoint (every weight byte is\n"
           "                      made resident before the listener opens; a checkpoint\n"
           "                      this build cannot execute refuses to serve at startup)\n"
           "  --stage MODE        where the weights live: auto (default: the whole\n"
           "                      checkpoint is resident in RAM before the listener\n"
           "                      opens), ram (private mlock()ed arena), huge (2 MiB\n"
           "                      hugetlb arena, reused across restarts; needs\n"
           "                      QE_HUGEPAGE_DIR and a reserved pool). mmap is refused:\n"
           "                      it reads weights off disk per token, and is accepted\n"
           "                      only together with --report, which wants headers.\n"
           "  --mmproj PATH       vision projector GGUF, also placed in RAM at startup\n"
           "  --mtp PATH          draft-head GGUF, the nextn block these checkpoints are\n"
           "                      shipped with as a sidecar file. Its bytes are placed\n"
           "                      in RAM at startup like every other weight byte, and a\n"
           "                      head for a different model is refused there rather\n"
           "                      than discovered as a low accept rate. The panel and\n"
           "                      /v1/models say which file it is reading and whether the\n"
           "                      nextn block in it is what is drafting: greedy requests\n"
           "                      draft --spec-k tokens a window off it and verify that\n"
           "                      window in one pass\n"
           "  --id NAME           model id reported to clients (default: qe-selftest)\n"
           "  --port N            listen port (default 8080)\n"
           "  --threads N          pool threads (default: every core but one)\n"
           "  --ctx N              sequence capacity: KV cache, indexer keys and\n"
           "                      recurrent state per request (default: the length\n"
           "                      the checkpoint was trained for, 262144 here)\n"
           "  --sse-keepalive-ms N\n"
           "                      how often a stream that has nothing to send writes an\n"
           "                      SSE comment instead, so that a reader which went away\n"
           "                      mid-ingestion is noticed then rather than after the\n"
           "                      turn (default 15000; negative turns it off). A comment\n"
           "                      is bytes the client parses and its message assembler\n"
           "                      never sees. This does not hold open a client's idle\n"
           "                      deadline: MANUAL.md explains what does\n"
           "  --empty-thinking-blocks\n"
           "                      replay an assistant turn that carried no reasoning\n"
           "                      text as an empty thinking block, the way the chat\n"
           "                      template writes one. Off by default: an empty block\n"
           "                      reads to the model as \"thought of nothing\", and a\n"
           "                      client that never sends thinking back would replay a\n"
           "                      transcript of an assistant that never thinks. Turn it\n"
           "                      on to A/B the two prompt shapes\n"
           "  --proj-chunk MiB     how big a projection has to be for a chunked pass to\n"
           "                      block its tokens through one sweep of it instead of\n"
           "                      sweeping it once per token, which is the ingestion\n"
           "                      speed switch: the number is a floor, so a big one\n"
           "                      blocks few tensors. 0 (default) takes what this build\n"
           "                      ships with, which is the sweep and the fastest thing\n"
           "                      here; -1 never blocks, which is the ordered walk that\n"
           "                      a prompt ends in bit-identically to feeding it one\n"
           "                      token at a time. Overrides QE_PROJ_MB, and `make\n"
           "                      forward` prices both on one load of a checkpoint\n"
           "  --shutdown-grace N   seconds a SIGTERM waits for a request that is still\n"
           "                      running before exiting anyway (default 5); the\n"
           "                      listener closes immediately either way\n"
           "  --turn-cache MiB     what a second turn may go back to: the budget for\n"
           "                      rewind points, each one the sequence state at a\n"
           "                      boundary a client's next prompt is likely to byte-\n"
           "                      extend. Left unset (0) it takes a fifth of what is\n"
           "                      still free once the weights are resident, enough for\n"
           "                      six turns of points and never more than half of what\n"
           "                      is free, and the banner says what that bought; a\n"
           "                      negative turns reuse off, so every turn reads its\n"
           "                      own prompt again\n"
           "  --ui MODE            the live terminal panel: auto (default: only when\n"
           "                      stdout is a terminal), on, off. It shows tokens going\n"
           "                      in and tokens coming out several times a second, and\n"
           "                      while it is up the per-request log line is drawn\n"
           "                      inside it rather than beside it\n"
           "  --kv-f32             keep the attention key/value cache in f32: 48 KiB\n"
           "                      per token against the 12 KiB the default q8_0 costs\n"
           "                      (19.7 against 55.5 KiB for everything a token of\n"
           "                      context costs, the indexer's keys included)\n"
           "  --attn-head-walk     walk the selected cells once per query head instead\n"
           "                      of once per KV group: twelve passes over the gather\n"
           "                      where the default makes one, and a score array per\n"
           "                      layer. It is the form the engine used before the\n"
           "                      fused walk, kept to measure against\n"
           "  --spec-k N           draft length for speculative decoding (default: 1\n"
           "                      with a --mtp head and 2 with the n-gram drafter,\n"
           "                      which is what this host measures worth it; 0 off)\n"
           "  --temp X             default temperature (0 = greedy)\n"
           "  --top-k N            default top-k: 0 keeps the file's own (20 here),\n"
           "                      -1 turns it off\n"
           "  --top-p X            default nucleus mass: 0 keeps the file's own (0.95),\n"
           "                      1 turns it off\n"
           "  --min-p X            default floor as a fraction of the best candidate:\n"
           "                      0 keeps the file's own (off), -1 turns it off; applied\n"
           "                      after top-k and before the nucleus\n"
           "  --fuse SECONDS       hard wall-clock deadline, 0 disables (default 0)\n"
           "  --report             print the bandwidth budget and the checkpoint metadata\n"
           "  -v, --verbose\n"
           "  -h, --help\n"
           "\n"
           "endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,\n"
           "           POST /v1/completions (stream: true is SSE)\n",
           argv0);
}

static const char *BUDGET[] = {
    "measured on this host (Core Ultra 7 265, 20 cores, no AVX-512, DDR5):",
    "  DRAM read, same bytes three ways (make alu): the shipped Q8_0 GEMV 75.9, an",
    "    identical walk with the arithmetic deleted 78.9, a flat 64B-aligned read 76.2",
    "    - at 512 MiB and 19 threads. One P core alone: 17.1 / 23.4 / 31.1. Eight P",
    "    cores stream 85.5, which a GEMV does not reach (59.3). The 113.8 GB/s this",
    "    file used to quote as peak is reproduced by nothing here (DESIGN.md 1.2)",
    "  so the step is memory-bound, not ALU-bound: the kernel is 96-99% of the read,",
    "    and the arithmetic only binds once data is cache-resident (68% at 32 MiB,",
    "    23% at 8 MiB), which 5568 MiB into a 30 MiB L3 never is",
    "  64 MiB stream reads 78-97 GB/s, median 89",
    "  read and written back in place, which is what the recurrent state costs every",
    "    token: 74.0 GB/s, 96% of the same walk read only, so the writeback is worth",
    "    about a line in twenty-five (make alu, --kernels read,state,statent)",
    "  a streaming store is worse (72.2): delta_task reads a row of state before it",
    "    writes that row, so the line is already there and there is no fetch to skip",
    "  one decode step of the checkpoint: 5.44 GiB (5568 MiB), whatever it emits",
    "    computed from the tensor table by bin/bench_gemv, not estimated:",
    "    dense Q8_0 projections   66.6%   74-83 GB/s",
    "    routed experts           19.9%   44 GB/s IQ3_S gate/up, 60 IQ4_NL down",
    "    lm_head Q6_K              8.9%   44-46 GB/s",
    "    f32 router logits         4.3%   stream rate, no SIMD dot",
    "  a lone token reads two copies made at load time instead, each priced in the",
    "    load that staged it by `make score`'s decode arms (ideas.md 7.23):",
    "    dense rows as IQ4_NL    1921 MiB for 3625, 1.16x of a step, +0.063 nats",
    "    lm_head as Q8_0          644 MiB for 497,  1.02x of a step, +0.000 nats",
    "    the two as shipped:      1.19x of a step, 107.0 to 90.2 ms/token, +0.021 nats",
    "    QE_I4_DENSE=lean holds the 194 mixers and 31 end matrices at Q8_0: 1.09x",
    "    for +0.0105 nats, which is 8 ms of a step bought back (make score)",
    "    the draft head reads the same two copies: its output.weight is these bytes",
    "    so does a verify window, which is 1.03x of a speculative pass (make head)",
    "  => 11.4 tok/s single stream (88.0 ms/token) as the file stands, 13.3-13.5",
    "     with both copies on, and 1.06x of the plain loop with MTP drafting one token",
    "     the way a window shares that head being most of what that margin is made of",
    "  Q8_0 GEMV on this file's rows, warm 512 MiB span: 73.6-78.7 GB/s,",
    "     which is the hardest ceiling here: 12.6-13.5 tok/s",
    "     (the ~6% run-to-run spread is the powersave governor: see docs/DESIGN.md 6)",
    "  expert reads: best of five at 19/12/8/4 threads is 43.3/31.3/24.3/12.4 GB/s",
    "  the pool is one core short of the affinity mask, because at full width every",
    "  dispatch pays a scheduler timeslice (1.6 us against 69 us, make pool) and a",
    "  token is ~800 dispatches: 106.2 ms a step at 19 threads, 125.2 ms at 20",
    "  pinning workers to cores by cpu_capacity costs 2-3x the handshake and is off",
    "  (QE_PIN=1); software prefetch measured as a 6-67% regression and was deleted;",
    "  so were vpdpbusd, a split weights/scales layout, two rows in flight and a",
    "  streaming store for the state (93-96% of the ordinary store, both spans)",
    "live numbers: make fast (kernels), make moe (experts), make mtp (MTP loop),",
    "make alu (where the limit is, per core class), make gemv (per-token bytes and",
    "GEMV rate, measured off the checkpoint)",
};

int main(int argc, char **argv) {
    qe_serve_opts o;
    memset(&o, 0, sizeof o);
    o.port = 8080;
    o.engine.spec_k = QE_SPEC_K_AUTO;
    int fuse = 0, report = 0;
    float tmp_temp = 0.0f;
    const char *stage_s = NULL;
    const char *ui_s = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char **as_str = NULL;
        int *as_int = NULL;
        float *as_flt = NULL;
        int *as_flag = NULL;
        if (!strcmp(a, "--model"))
            as_str = &o.engine.model_path;
        else if (!strcmp(a, "--stage"))
            as_str = &stage_s;
        else if (!strcmp(a, "--mmproj"))
            as_str = &o.engine.mmproj_path;
        else if (!strcmp(a, "--mtp"))
            as_str = &o.engine.mtp_path;
        else if (!strcmp(a, "--id"))
            as_str = &o.engine.id;
        else if (!strcmp(a, "--port"))
            as_int = &o.port;
        else if (!strcmp(a, "--threads"))
            as_int = &o.engine.n_threads;
        else if (!strcmp(a, "--ctx"))
            as_int = &o.engine.n_ctx;
        else if (!strcmp(a, "--sse-keepalive-ms"))
            as_int = &o.sse_keepalive_ms;
        else if (!strcmp(a, "--empty-thinking-blocks"))
            as_flag = &o.empty_thinking_blocks;
        else if (!strcmp(a, "--proj-chunk"))
            as_int = &o.engine.proj_chunk_mb;
        else if (!strcmp(a, "--shutdown-grace"))
            as_int = &o.shutdown_grace_secs;
        else if (!strcmp(a, "--turn-cache"))
            as_int = &o.engine.turn_cache_mib;
        else if (!strcmp(a, "--ui"))
            as_str = &ui_s;
        else if (!strcmp(a, "--kv-f32"))
            as_flag = &o.engine.kv_f32;
        else if (!strcmp(a, "--attn-head-walk"))
            as_flag = &o.engine.attn_head_walk;
        else if (!strcmp(a, "--spec-k"))
            as_int = &o.engine.spec_k;
        else if (!strcmp(a, "--temp"))
            as_flt = &tmp_temp;
        else if (!strcmp(a, "--top-k"))
            as_int = &o.engine.top_k;
        else if (!strcmp(a, "--top-p"))
            as_flt = &o.engine.top_p;
        else if (!strcmp(a, "--min-p"))
            as_flt = &o.engine.min_p;
        else if (!strcmp(a, "--fuse"))
            as_int = &fuse;
        else if (!strcmp(a, "--report"))
            as_flag = &report;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose"))
            as_flag = &o.verbose;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "%s: unknown argument '%s'\n", argv[0], a);
            usage(argv[0]);
            return 2;
        }
        if (as_flag) {
            *as_flag = 1;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: %s needs a value\n", argv[0], a);
            return 2;
        }
        const char *v = argv[++i];
        char *end = NULL;
        if (as_str) {
            *as_str = v;
            continue;
        }
        long x = strtol(v, &end, 10);
        if (as_int) {
            if (!end || *end) {
                fprintf(stderr, "%s: %s wants an integer\n", argv[0], a);
                return 2;
            }
            *as_int = (int)x;
            continue;
        }
        *as_flt = (float)strtod(v, &end);
        if (!end || *end) {
            fprintf(stderr, "%s: %s wants a number\n", argv[0], a);
            return 2;
        }
    }
    o.engine.temp = tmp_temp;

    /* A server holding a checkpoint is a server that will be asked to generate,
     * and generating from a demand-paged mapping of a 1 GiB ZFS ARC means one
     * raidz2 seek per expert row. So there is one serving configuration: every
     * weight byte is resident before the listener opens, and an engine that
     * cannot get them fails at startup with the two numbers that decided it.
     * --report is the only metadata-only open, because a budget report that
     * costs an 87 GiB read stops being run. */
    if (stage_s) {
        if (!strcmp(stage_s, "auto"))
            o.engine.stage = QE_STAGE_AUTO | QE_STAGE_STRICT;
        else if (!strcmp(stage_s, "ram"))
            o.engine.stage = QE_STAGE_RAM | QE_STAGE_STRICT;
        else if (!strcmp(stage_s, "huge"))
            o.engine.stage = QE_STAGE_HUGE | QE_STAGE_STRICT;
        else if (!strcmp(stage_s, "mmap") && report) {
            o.engine.stage = QE_STAGE_MMAP | QE_STAGE_DESCRIBE;
        } else if (!strcmp(stage_s, "mmap")) {
            fprintf(stderr, "qe: --stage mmap reads weights off disk between tokens,"
                            " which this engine will not do; drop it (auto places the whole"
                            " checkpoint in RAM) or combine it with --report for headers\n");
            return 2;
        } else {
            fprintf(stderr, "qe: --stage wants auto, ram, huge or mmap\n");
            return 2;
        }
    } else if (report) {
        /* Headers only: 0.07 s, no weight bytes read, and nothing is claimed
         * about where they live beyond "not here". */
        o.engine.stage = QE_STAGE_MMAP | QE_STAGE_DESCRIBE;
    } else {
        /* QE_STAGE_AUTO is the zero value: hugetlbfs arena if a pool was
         * reserved, otherwise the whole checkpoint read into private pinned RAM
         * at startup. STRICT means "otherwise" is an error naming the two
         * numbers that made it, never a mapping that reads per expert row. */
        o.engine.stage = QE_STAGE_AUTO | QE_STAGE_STRICT;
    }

    /* The panel's default is AUTO, and AUTO is the zero value, so the only work
     * here is turning a word into one of three constants and refusing a fourth. */
    if (ui_s) {
        if (!strcmp(ui_s, "auto"))
            o.ui = QE_UI_AUTO;
        else if (!strcmp(ui_s, "on"))
            o.ui = QE_UI_ON;
        else if (!strcmp(ui_s, "off"))
            o.ui = QE_UI_OFF;
        else {
            fprintf(stderr, "qe: --ui wants auto, on or off\n");
            return 2;
        }
    }

    /* Before anything can touch the checkpoint. */
    qe_fuse(fuse);

    /* Unspecified is QE_THREADS_AUTO, i.e. one core short of the affinity mask,
     * not the whole mask: see the comment on that constant, because at full width
     * every barrier in a decode step pays a scheduler timeslice. An explicit
     * --threads is honoured exactly. */
    int nthreads = o.engine.n_threads > 0 ? o.engine.n_threads : QE_THREADS_AUTO;
    qe_pool_init(nthreads, NULL, 0);
    if (o.verbose)
        fprintf(stderr, "qe: pool: %d threads on %d cpus (%s)\n", qe_pool_size(), qe_cpus(),
                o.engine.n_threads > 0 ? "as requested" : "one core left for the host");

    if (report) {
        for (size_t i = 0; i < sizeof BUDGET / sizeof *BUDGET; i++)
            printf("%s\n", BUDGET[i]);
        if (o.engine.model_path) {
            char err[512] = { 0 };
            qe_engine *e = qe_engine_open(&o.engine, err, sizeof err);
            if (!e) {
                printf("\ncheckpoint: %s\n", err);
                return 1;
            }
            qe_engine_info info;
            qe_engine_describe(e, &info);
            printf("\ncheckpoint: %s\n  id        %s\n  arch      %s\n  volumes   %d\n"
                   "  tensors   %lld\n  weights   %.1f GiB\n  quant     %s\n",
                   o.engine.model_path, info.id, info.arch, info.n_volumes,
                   (long long)info.n_tensors, (double)info.bytes / 1073741824.0, info.quant);
            if (info.label[0] || info.size_label[0])
                printf("  name      %s %s\n", info.label, info.size_label);
            printf("  shape     %d layers, %d wide, %d experts (%d active), "
                   "context %d, vocab %d\n", info.n_blocks, info.n_embd,
                   info.n_experts, info.n_experts_active, info.n_ctx_train, info.n_vocab);
            printf("  tokens    eos %d, image %d, chat template %s\n", info.eos_token_id,
                   info.image_token_id,
                   info.has_chat_template ? "in file (not evaluated by this build)"
                                          : "not in file");
            printf("  bpe       %s\n", info.tokenizer[0] ? info.tokenizer : "none");
            printf("  resident  stage %s, %.1f GiB (%.1f GiB reused from a previous run), %s\n",
                   info.stage, (double)info.stage_bytes / 1073741824.0,
                   (double)info.stage_cached / 1073741824.0,
                   info.stage_pinned == 2 ? "hugetlb (not pageable)"
                                          : (info.stage_pinned ? "mlock pinned"
                                                               : "reclaimable"));
            if (!strcmp(info.stage, "mmap"))
                printf("            (metadata-only open: pass --stage to place the weights)\n");
            {
                /* How a decode step is divided between the pool threads. Part of
                 * every wall clock below the shape lines. */
                char sbuf[128];
                printf("  slicing   %s\n", qe_slice_summary(sbuf, sizeof sbuf));
            }
            printf("  drafter   %s\n",
                   info.has_mtp_head ? "MTP/draft head present in the checkpoint"
                                     : "no MTP or draft head in this checkpoint");
            if (info.mtp_path[0])
                printf("  head      %s\n            %lld tensors, %.2f GiB, stage %s, "
                       "%s\n",
                       info.mtp_path, (long long)info.mtp_tensors,
                       (double)info.mtp_bytes / 1073741824.0, info.mtp_stage,
                       info.mtp_quant[0] ? info.mtp_quant : "types unknown");
            if (info.mtp_layers || info.mtp_head_blocks)
                printf("            nextn %d layer%s, %d decoder block%s beyond the "
                       "checkpoint's\n",
                       info.mtp_layers, info.mtp_layers == 1 ? "" : "s",
                       info.mtp_head_blocks, info.mtp_head_blocks == 1 ? "" : "s");
            printf("  generate  %s\n  why       %s\n",
                   info.can_generate ? "yes" : "no (see why)", info.why);
            qe_engine_close(e);
        } else {
            printf("\nno --model: the server would answer from the selftest engine%s\n",
                   o.engine.mtp_path
                       ? ", and a --mtp draft head is opened by an engine, not by a report"
                       : "");
        }
        qe_pool_shutdown();
        return 0;
    }

    int rc = qe_serve(&o);
    qe_pool_shutdown();
    return rc;
}
