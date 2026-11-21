#!/bin/sh
# Cache-resident sweep for bin/bench_alu: does the arithmetic bind once the span
# stops coming from DRAM? At 512 MiB every walk lands on the same wall, so the
# only place a cheaper kernel can pay is the regime the mixer and block-stage
# weights actually live in - a few MiB per thread, hot in L2 or L3.
#
#   tools/sweep_alu.sh 0 8 0-7 8-19 0-18
#
# --inner scales the work per timed pass so the small spans cost more to measure
# than the thread barrier does; without it a 8 MiB pass is tens of microseconds
# and the dispatch dominates the number.
set -eu

BIN=${BIN:-./bin/bench_alu}
for span in 8 32 128 512; do
    inner=1
    [ "$span" -le 8 ] && inner=96
    [ "$span" -eq 32 ] && inner=32
    [ "$span" -eq 128 ] && inner=8
    for cpus in "$@"; do
        $BIN --span "$span" --inner "$inner" --reps 3 --cpus "$cpus" |
            grep -vE '^ *note|^ *[0-9]+ MiB at|one vpsadbw|GEMV pays'
        echo
    done
done
