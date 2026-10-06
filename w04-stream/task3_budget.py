#!/usr/bin/env python3
"""Week 4 · Task 3 — Same memory, fewer mistakes.

Textbook §4.4 (Bloom filters), §4.5 (counting distinct).

`NaiveFilter` is a membership filter in a fixed number of bits. It works. It
also makes far more mistakes than it has to with the memory it was given, and
it does so for a reason you can find by reading §4.4.2 and doing one derivative.

You get **exactly the same number of bits**. Make fewer mistakes.

    python3 bench.py
    python3 bench.py --yours

The rule that makes this interesting: a false negative is not allowed. Ever.
The whole point of this structure is that "no" means no. A filter that gets a
better score by occasionally forgetting something it was given has not improved
anything, it has broken the contract.
"""
import hashlib, math


class NaiveFilter:
    """One hash function, and the bits it was given."""

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed
        self.bits = bytearray(n_bits)

    def _index(self, item):
        d = hashlib.blake2b(str(item).encode(), digest_size=8,
                            key=str(self.seed).encode()).digest()
        return int.from_bytes(d, "big") % self.n_bits

    def add(self, item):
        self.bits[self._index(item)] = 1

    def __contains__(self, item):
        return bool(self.bits[self._index(item)])

    def memory_bits(self):
        return self.n_bits


class YourFilter:
    """Your filter.

        __init__(n_bits, seed=246)
        add(item)
        item in filter  ->  bool
        memory_bits()   ->  how many bits you are using

    `memory_bits()` must not exceed the `n_bits` you were given. The harness
    checks. Counting only some of your memory is not an optimisation.

    §4.4.2 gives the false-positive rate of a filter with m bits, k hashes and
    n items inserted. There is a k that minimises it, and it depends on m/n.
    The harness tells you n before you start, so you have no excuse for guessing.

    Then there is a second question, which is worth more: the harness inserts
    a **known** number of items, but a real stream does not tell you n in
    advance. What would you do then? You do not have to implement it - but
    observation.md asks.
    """

    # The harness inserts 8,000 items and says so up front (bench.py, task3.md).
    EXPECTED_ITEMS = 8_000

    def __init__(self, n_bits, seed=246, expected_items=EXPECTED_ITEMS):
        self.n_bits = n_bits
        self.seed = seed
        # §4.4.2: FP(k) = (1 - e^(-kn/m))^k. Write p = e^(-kn/m) (the chance a
        # bit is still 0), so k = -(m/n) ln p and
        #   ln FP = k ln(1 - p) = -(m/n) * ln(p) * ln(1 - p).
        # ln(p) ln(1-p) is symmetric in p <-> 1-p, so its extreme is at p = 1/2:
        # each bit should end up 1 with probability exactly one half. Then
        #   k* = (m/n) ln 2 = 10 * 0.693 = 6.93  ->  k = 7
        #   FP* = (1/2)^k* = 0.6185^(m/n) = 0.6185^10 ~ 0.82 %
        # The baseline uses k = 1: FP = 1 - e^(-0.1) = 9.52 %.
        self.k = max(1, round(n_bits / expected_items * math.log(2)))
        if self.k > 8:
            raise ValueError("this filter cuts at most 8 hashes from one digest")
        self.key = str(seed).encode()
        self.bits = bytearray((n_bits + 7) // 8)   # packed, 1 bit per position

    def _indexes(self, item):
        # k independent 64-bit slices of one keyed blake2b digest
        d = hashlib.blake2b(str(item).encode(), digest_size=8 * self.k,
                            key=self.key).digest()
        for i in range(self.k):
            yield int.from_bytes(d[8 * i:8 * i + 8], "big") % self.n_bits

    def add(self, item):
        for j in self._indexes(item):
            self.bits[j >> 3] |= 1 << (j & 7)

    def __contains__(self, item):
        return all(self.bits[j >> 3] >> (j & 7) & 1 for j in self._indexes(item))

    def memory_bits(self):
        # The whole filter state is the bit array. k and the key are
        # parameters, the same kind of thing as NaiveFilter's seed.
        return len(self.bits) * 8
