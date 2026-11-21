#include "turns.h"

#include <stddef.h>

int qe_turns_match(qe_turn_slot *slots, int nslots, const int *ids, long np, long *at,
                   long *agreed, uint64_t *clock) {
    int best = -1;
    long bestn = 0;
    long far = 0;
    for (int i = 0; i < nslots; i++) {
        if (!slots[i].pos)
            continue;
        const long n = slots[i].pos;
        if (n < 1 || n > np || !slots[i].have_logits || !slots[i].ids)
            continue;
        const uint32_t *s = slots[i].ids;
        long k = 0;
        while (k < n && (int)s[k] == ids[k])
            k++;
        if (k > far)
            far = k;
        if (k != n)
            continue;
        if (n > bestn) {
            bestn = n;
            best = i;
        }
    }
    if (best >= 0 && clock)
        slots[best].used = ++*clock;
    if (at)
        *at = bestn;
    if (agreed)
        *agreed = far;
    return best;
}

int qe_turns_victim(const qe_turn_slot *slots, int nslots, int answer) {
    for (int i = 0; i < nslots; i++)
        if (slots[i].pos && !slots[i].have_logits)
            return i;
    for (int i = 0; i < nslots; i++)
        if (!slots[i].pos)
            return i;
    uint64_t oldest = 0;
    int victim = -1;
    for (int i = 0; i < nslots; i++) {
        if (!slots[i].pos || !slots[i].answer)
            continue;
        if (oldest == 0 || slots[i].used < oldest) {
            oldest = slots[i].used;
            victim = i;
        }
    }
    if (victim >= 0)
        return victim;
    /* Both kinds of point are in use, and this is an answer-end one. */
    if (answer)
        return -1;
    /* Both kinds of point are in use. The shortest one is kept out of the running
     * here: it is the state after the least amount of conversation, so it is the one
     * a client that rewrites something earlier than all the others still comes back
     * to, and losing it is what turns one edited tool result into re-reading the
     * whole session. Among the rest, least recently matched. */
    long shortest = 0;
    int anchor = -1;
    for (int i = 0; i < nslots; i++) {
        if (!slots[i].pos)
            continue;
        if (anchor < 0 || slots[i].pos < shortest) {
            shortest = slots[i].pos;
            anchor = i;
        }
    }
    oldest = 0;
    for (int i = 0; i < nslots; i++) {
        if (!slots[i].pos || i == anchor)
            continue;
        if (oldest == 0 || slots[i].used < oldest) {
            oldest = slots[i].used;
            victim = i;
        }
    }
    if (victim < 0)
        victim = anchor; /* one point held, and it is this one's turn to lose */
    return victim;
}

long qe_turns_point_mib(long point_bytes) {
    if (point_bytes <= 0)
        return 0;
    return (long)(((double)point_bytes / 1048576.0) + 1.0);
}

long qe_turns_auto_mib(long free_mib, long point_mib, int points, double share,
                       int spare_div, long cap_mib) {
    if (point_mib <= 0)
        return 0;
    const long need = point_mib * (long)(points > 0 ? points : 1);
    double mib = (double)free_mib * share;
    if (mib < (double)need)
        mib = (double)need;
    if (free_mib > 0 && spare_div > 0 && mib > (double)(free_mib / spare_div))
        mib = (double)(free_mib / spare_div);
    if (cap_mib > 0 && mib > (double)cap_mib)
        mib = (double)cap_mib;
    return (long)mib;
}
