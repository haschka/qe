/* Which rewind point a turn should go back to, and which one a new point may take a
 * slot from.
 *
 * Two decisions, made over an array of fixed-size slots, and nothing else: no
 * allocation, no clock, no file, and no idea what a sequence state is. The engine
 * (`src/engine.c`) owns the states and the logits, and hands these views of them
 * here. That separation is the reason this file exists rather than the two functions
 * staying where they were written: the cache's behaviour is a function of the number
 * of slots and the order they were used in, and the bug it is worth catching - a
 * budget that buys two turns of history and then silently re-reads the conversation
 * from the third turn on - is a bug in that function, not in the copying. Measured
 * against a real checkpoint it costs minutes per run and needs the checkpoint, so
 * almost nothing about eviction is asserted anywhere; against these views it is a
 * 30-turn replay in a few microseconds, and `bench/bench_turns.c` is that replay.
 *
 * The rules themselves are documented at their old home and are reproduced here
 * unchanged, in behaviour and in the order the loops run in. */
#ifndef QE_TURNS_H
#define QE_TURNS_H

#include <stdint.h>

/* One slot as far as these two decisions care about it. A slot with `pos` 0 holds
 * nothing; a slot whose logits are gone can still be a state, but nothing matches it,
 * which is the difference between a point and a candidate. */
typedef struct {
    long pos;             /* tokens the point covers, from the start of the sequence */
    const uint32_t *ids;  /* those tokens, `pos` of them, in order */
    int have_logits;      /* the logits after the last of them, when they are kept */
    int answer;           /* covers an answer as well as its prompt */
    uint64_t used;        /* eviction order, not a timestamp */
} qe_turn_slot;

/* The longest point that is this prompt from its first token, as an index into
 * `slots`, or -1. A point matches only token for token over its whole length: a
 * sequence is defined by its beginning, and "close from halfway through" is a
 * different conversation. `agreed` is the other half of the answer and the one a
 * client needs on a miss - how far the prompt and the furthest point on hand did
 * agree before they disagreed, which is a position in a conversation rather than the
 * fact that one was lost. A match marks its winner used, and the clock it stamps comes
 * from `*clock`, which is advanced. */
int qe_turns_match(qe_turn_slot *slots, int nslots, const int *ids, long np, long *at,
                   long *agreed, uint64_t *clock);

/* Which slot a store may take, or -1 when nothing here is worth giving up. A point
 * whose logits are gone goes first, then an empty slot, then the answer-end point that
 * was matched longest ago, then - for a prompt-end store - the least recently matched
 * point that is not the shortest one held. The order is the argument, and it is at
 * `turns_victim` in `src/engine.c`. */
int qe_turns_victim(const qe_turn_slot *slots, int nslots, int answer);

/* What one point costs, in MiB, rounded up: a budget that is a hair short of a point
 * buys one fewer than the number printed next to it. */
long qe_turns_point_mib(long point_bytes);

/* The budget an undecided `--turn-cache` buys: `share` of what is free, at least
 * enough for `points` points, and never more than `1/spare_div` of what is free nor
 * more than `cap_mib`. A floor in points rather than in MiB is the whole reason this is
 * a function: a point is the state of a sequence, so a floor in MiB silently buys two
 * turns of history on most hosts. The caps are applied in that order and one of them
 * can bite the floor, which is what the engine reports as a cache that is short. */
long qe_turns_auto_mib(long free_mib, long point_mib, int points, double share,
                       int spare_div, long cap_mib);

#endif /* QE_TURNS_H */
