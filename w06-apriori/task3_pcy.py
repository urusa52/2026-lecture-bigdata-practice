#!/usr/bin/env python3
"""Week 6 · Task 3 — Make pass two fit.

Textbook §6.3 (PCY), §6.3.2 - §6.3.4.

`PlainApriori` does what Task 1 asked: use pass one to drop infrequent items,
then count every pair of surviving items. That is already much better than
counting all pairs. It is still not enough, because the surviving items are the
common ones, and the common ones appear together constantly.

The harness measures **the peak number of pair counters you held**, because
that is the thing that decides whether the algorithm runs at all. §6.3 is about
spending pass one's spare memory to shrink it.

    python3 bench.py
    python3 bench.py --yours

Correctness first: you must find exactly the same frequent pairs. Finding fewer
is not an optimisation.
"""
from array import array
from collections import Counter
from itertools import combinations


class PlainApriori:
    """Pass one drops infrequent items. Pass two counts every surviving pair."""

    def __init__(self, support):
        self.support = support
        self.peak_counters = 0

    def run(self, baskets):
        counts = Counter()
        for basket in baskets:
            counts.update(basket)
        frequent = {i for i, c in counts.items() if c >= self.support}

        pair_counts = Counter()
        for basket in baskets:
            items = sorted(basket & frequent)
            for pair in combinations(items, 2):
                pair_counts[pair] += 1
            self.peak_counters = max(self.peak_counters, len(pair_counts))

        return {frozenset(p): c for p, c in pair_counts.items()
                if c >= self.support}


class YourAlgorithm:
    """Your frequent-pair finder.

        __init__(support)
        run(baskets) -> {frozenset({a, b}): count}
        .peak_counters -> the most pair counters you ever held at once

    Same pairs as the baseline. Fewer counters.

    Pass one only needs one integer per item, and there are not many items. The
    rest of your memory is sitting idle while you do it. §6.3 spends it: hash
    every pair you see in pass one into a fixed array of buckets, and count the
    buckets rather than the pairs.

    A bucket whose total is below the support threshold cannot contain a
    frequent pair. In pass two you skip every pair landing in such a bucket -
    and the bucket array collapses to a bitmap, one bit each, before you need
    the memory for counters.

    Two things to be careful of:

      * a bucket being frequent does not make its pairs frequent. It is a
        filter, not an answer
      * `peak_counters` is on your honour. Count the pair counters you hold at
        the same time. The bitmap is not a pair counter, but if you keep the
        full bucket counts alive into pass two, that is not free either -
        observation.md asks about it
    """

    # 2^20 buckets. Pass one sees ~3.4 million pair occurrences on this data,
    # so the average bucket holds ~3.3 - far below support 50. A bucket then
    # only reaches 50 if a genuinely frequent pair (or a few heavy ones) lands
    # in it, which is what makes the bitmap a filter rather than a formality.
    # As 4-byte counters this is 4 MB in pass one, against 6.5 MB for
    # A-Priori's triangular matrix. If peak *bytes* rather than peak counters
    # were the score, ~160,000 buckets would be the choice: that is where pass
    # one and pass two cost the same (~0.65 MB). See task3_memory.py.
    N_BUCKETS = 1 << 20

    def __init__(self, support, n_buckets=None):
        self.support = support
        self.n_buckets = n_buckets or self.N_BUCKETS
        self.peak_counters = 0
        # reported for observation.md, not used by the harness
        self.stats = {}

    def _bucket(self, a, b):
        # a < b always (pairs come from a sorted basket). Pair -> one integer,
        # then a multiplicative hash so neighbouring item ids scatter.
        return ((a * 2_000_003 + b) * 0x9E3779B1 >> 7) % self.n_buckets

    def run(self, baskets):
        support, nb = self.support, self.n_buckets
        bucket = self._bucket

        # ---- pass one: item counts AND bucket counts, in the same scan
        item_counts = Counter()
        buckets = array("I", bytes(4 * nb))  # 4-byte counters, pass one only
        for basket in baskets:
            item_counts.update(basket)
            items = sorted(basket)
            for i in range(len(items)):
                a = items[i]
                for b in items[i + 1:]:
                    buckets[bucket(a, b)] += 1
        frequent = {i for i, c in item_counts.items() if c >= support}

        # ---- between passes: collapse the buckets to one bit each, and drop
        # the integer array before a single pair counter exists. The bitmap is
        # nb/8 bytes; the bucket array it replaces was nb Python ints.
        bitmap = bytearray((nb + 7) // 8)
        frequent_buckets = 0
        for k, c in enumerate(buckets):
            if c >= support:
                bitmap[k >> 3] |= 1 << (k & 7)
                frequent_buckets += 1
        del buckets

        # ---- pass two: a pair gets a counter only if both items are frequent
        # AND its bucket is frequent. Both conditions are necessary for a
        # frequent pair, so nothing real is lost; neither is sufficient, so the
        # counts still decide.
        pair_counts = {}
        for basket in baskets:
            items = sorted(basket & frequent)
            for i in range(len(items)):
                a = items[i]
                for b in items[i + 1:]:
                    k = bucket(a, b)
                    if bitmap[k >> 3] >> (k & 7) & 1:
                        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
            if len(pair_counts) > self.peak_counters:
                self.peak_counters = len(pair_counts)

        result = {frozenset(p): c for p, c in pair_counts.items()
                  if c >= support}
        self.stats = {"buckets": nb, "frequent_buckets": frequent_buckets,
                      "frequent_items": len(frequent),
                      "bitmap_bytes": len(bitmap),
                      "candidates": len(pair_counts), "frequent_pairs": len(result)}
        return result
