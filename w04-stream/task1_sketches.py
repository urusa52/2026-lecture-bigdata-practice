#!/usr/bin/env python3
"""Week 4 · Task 1 — Answer questions about a stream you cannot store.

Textbook §4.3 (sampling), §4.4 (Bloom filter), §4.5 (Flajolet-Martin).

The premise of the whole chapter: the stream is longer than your memory, it
goes past once, and you still have to answer. Every method here trades an exact
answer for a bounded amount of space, and the job is to know exactly what you
traded.

You build three, and the harness checks each against the truth it is
approximating.

    python3 task1_sketches.py --verify
"""
import argparse, hashlib, math, random, statistics


class BloomFilter:
    """Membership, with one-sided error.

    A Bloom filter never says "no" about something you inserted. It sometimes
    says "yes" about something you did not. That asymmetry is the entire design
    and it is why it is useful for "have I seen this before" and useless for
    "is this definitely in the set".

    `m` bits, `k` hash functions.

    Why there is never a false negative: bits only ever go 0 -> 1. Once `add(x)`
    has set x's k bits, nothing can clear them, so `x in bf` always sees k ones.
    """

    def __init__(self, m, k, seed=246):
        if not 1 <= k <= 8:
            raise ValueError("k must be 1..8 (one 64-byte blake2b digest gives 8 hashes)")
        self.m, self.k = m, k
        self.key = str(seed).encode()
        self.bits = bytearray((m + 7) // 8)       # packed: m bits, not m bytes

    def _indexes(self, item):
        # One keyed blake2b digest of 8k bytes, cut into k independent 64-bit
        # values -> k positions. Independent slices, not h1 + i*h2, so the
        # textbook formula (which assumes independent hashes) applies directly.
        d = hashlib.blake2b(str(item).encode(), digest_size=8 * self.k,
                            key=self.key).digest()
        for i in range(self.k):
            yield int.from_bytes(d[8 * i:8 * i + 8], "big") % self.m

    def add(self, item):
        for j in self._indexes(item):
            self.bits[j >> 3] |= 1 << (j & 7)

    def __contains__(self, item):
        # all() stops at the first 0 bit: a single 0 proves "never inserted"
        return all(self.bits[j >> 3] >> (j & 7) & 1 for j in self._indexes(item))

    def expected_fp_rate(self, n_inserted):
        """The textbook's predicted false-positive rate after n insertions.

        §4.4.2: kn darts thrown at m targets. A given bit is still 0 with
        probability (1 - 1/m)^(kn) ~ e^(-kn/m). An item never inserted is a
        false positive iff all k of its bits are 1:

            FP = (1 - e^(-kn/m))^k
        """
        return (1 - math.exp(-self.k * n_inserted / self.m)) ** self.k


# Flajolet-Martin ------------------------------------------------------------
_HASH_BYTES = 4                     # each hash function is a 32-bit value
# _TZ[b] = trailing zero bits of byte b (b = 1..255); a zero byte is handled
# by moving on to the next byte.
_TZ = [8] + [(b & -b).bit_length() - 1 for b in range(1, 256)]


def _tail_length(digest, start, width=_HASH_BYTES):
    """Trailing zero bits of the little-endian integer digest[start:start+width].

    Done byte by byte with a lookup table instead of int.from_bytes + bit
    tricks: indexing bytes gives a small cached int, so this allocates
    nothing. That matters in task 2, where tracemalloc is watching every
    allocation and big-int arithmetic made FM ~15x slower.
    """
    t = 0
    for j in range(start, start + width):
        byte = digest[j]
        if byte:
            return t + _TZ[byte]
        t += 8
    return t                        # all-zero hash: cap at 8 * width


def fm_max_tails(stream, n_hashes=64, seed=246):
    """The one pass. Returns R_i = max tail length seen under each hash.

    Hash family: one SHAKE-256 digest of n_hashes * 4 bytes per item, cut into
    n_hashes independent 32-bit values. Same item -> same digest, so h_i(x) is
    a fixed function of x, as §4.5 requires.

    Memory: n_hashes small integers, no matter how long the stream is or how
    many distinct items it holds. Duplicates hash to the same value, so they
    cannot raise any R_i - that is what makes this count DISTINCT items.
    """
    prefix = f"fm-{seed}:".encode()
    size = n_hashes * _HASH_BYTES
    R = [0] * n_hashes
    for item in stream:
        d = hashlib.shake_256(prefix + str(item).encode()).digest(size)
        for i in range(n_hashes):
            t = _tail_length(d, i * _HASH_BYTES)
            if t > R[i]:
                R[i] = t
    return R


def combine_mean(R):
    return sum(2 ** r for r in R) / len(R)


def combine_median(R):
    return statistics.median(2 ** r for r in R)


def combine_median_of_means(R, group_size=8):
    """§4.5.3: average inside small groups, then take the median of the averages.

    The mean inside a group lets the estimate fall between powers of two; the
    median across groups is meant to throw away the group that got lucky.
    In practice (see --compare) it overshoots: P(2^R > t) ~ m/t, a tail so
    heavy that every group mean is dragged up by its largest member.
    """
    groups = [R[i:i + group_size] for i in range(0, len(R), group_size)]
    return statistics.median(sum(2 ** r for r in g) / len(g) for g in groups)


# E[R] for the max of m trailing-zero counts is log2(m) + gamma/ln2 - 1/2
# (Flajolet & Martin's analysis; gamma = Euler's constant). ~0.333 bits of bias.
_FM_BIAS_BITS = 0.5772156649 / math.log(2) - 0.5


def combine_log_mean(R):
    """Average the exponents R_i, not the values 2^R_i.  <- the rule used.

    2^mean(R) is the geometric mean of the 2^R_i. Averaging in the exponent
    is what tames the outliers: a lucky hash with R = 22 among R ~ 14 moves
    the mean of 64 R's by 8/64 = 0.125, a factor of 1.09 - whereas in the
    arithmetic mean of 2^R the same hash is 256x everybody else.
    Unlike the median, the result is not stuck on powers of two.

    The mean of R overshoots log2(m) by gamma/ln2 - 1/2 ~ 0.333, so subtract it.
    """
    return 2 ** (statistics.mean(R) - _FM_BIAS_BITS)


def flajolet_martin(stream, n_hashes=64, seed=246):
    """Estimate how many DISTINCT items went past, in almost no memory.

    §4.5. Hash each item, count trailing zeros in the hash, keep the maximum.
    A maximum of R suggests about 2^R distinct items, because seeing R trailing
    zeros is a 1-in-2^R event.

    Combining rule: average the R's, then 2^(mean R), bias-corrected
    (`combine_log_mean`). `--compare` shows what the other rules give.

    The harness accepts anything **within a factor of two** of the truth. That is
    not a generous tolerance, it is an honest one: this method really is that
    crude, and HyperLogLog exists because of it.
    """
    return float(combine_log_mean(fm_max_tails(stream, n_hashes, seed)))


def reservoir_sample(stream, k, seed=246):
    """Keep k items uniformly at random from a stream of unknown length.

    §4.3 (algorithm R). Invariant after i items: each of them is in the
    reservoir with probability k/i. The i-th item is admitted with
    probability k/i and evicts a uniformly chosen slot, which keeps the
    invariant for i+1. The stream length is never needed: the sample is
    correct at every moment, so it is correct whenever the stream stops.
    """
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = rng.randrange(i)          # uniform in 0..i-1
            if j < k:                     # happens with probability k/i
                reservoir[j] = item       # and the slot is uniform among k
    return reservoir


def compare_fm(trials=10):
    """Run the three combining rules on the --verify stream with different seeds."""
    rng = random.Random(246)
    stream = [f"k{rng.randrange(20_000)}" for _ in range(120_000)]
    true = len(set(stream))
    print(f"  true distinct {true:,}   (64 hashes, {trials} seeds)\n")
    rules = {"mean of 2^R": combine_mean,
             "median of 2^R": combine_median,
             "median of means (8x8)": combine_median_of_means,
             "median of means (4x16)": lambda R: combine_median_of_means(R, 16),
             "median of means (2x32)": lambda R: combine_median_of_means(R, 32),
             "median of means (32x2)": lambda R: combine_median_of_means(R, 2),
             "2^mean(R) (used)": combine_log_mean}
    ratios = {name: [] for name in rules}
    for s in range(trials):
        R = fm_max_tails(stream, seed=s)
        for name, f in rules.items():
            ratios[name].append(f(R) / true)
    for name, rs in ratios.items():
        inside = sum(0.5 <= r <= 2 for r in rs)
        print(f"  {name:<24} min {min(rs):.2f}x  max {max(rs):.2f}x  "
              f"within 2x: {inside}/{trials}")


# ------------------------------------------------------------------- harness
def verify():
    fails = 0
    rng = random.Random(246)

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<46} {detail}")
        fails += not ok

    # --- Bloom: no false negatives, ever
    try:
        bf = BloomFilter(m=8192, k=5)
    except NotImplementedError:
        print("  BloomFilter is still a stub"); return 1
    inserted = [f"item-{i}" for i in range(800)]
    for x in inserted:
        bf.add(x)
    check("no false negatives", all(x in bf for x in inserted))

    absent = [f"other-{i}" for i in range(20_000)]
    fp = sum(1 for x in absent if x in bf) / len(absent)
    predicted = bf.expected_fp_rate(len(inserted))
    close = abs(fp - predicted) < max(0.02, predicted * 0.5)
    check("measured false-positive rate matches theory", close,
          f"measured {fp:.3%}, predicted {predicted:.3%}")

    # --- Flajolet-Martin: a factor of two is what this method gives you
    try:
        distinct = 20_000
        stream = [f"k{rng.randrange(distinct)}" for _ in range(120_000)]
        est = flajolet_martin(stream)
    except NotImplementedError:
        print("  flajolet_martin is still a stub"); return 1
    true_distinct = len(set(stream))
    ratio = est / true_distinct
    check("distinct estimate within a factor of 2", 0.5 <= ratio <= 2.0,
          f"estimated {est:,.0f}, true {true_distinct:,} ({ratio:.2f}x)")

    # --- Reservoir: uniform over many trials
    try:
        counts = [0] * 20
        trials = 4000
        for t in range(trials):
            s = reservoir_sample(range(20), 5, seed=t)
            for i in s:
                counts[i] += 1
    except NotImplementedError:
        print("  reservoir_sample is still a stub"); return 1
    expected = trials * 5 / 20
    spread = (max(counts) - min(counts)) / expected
    check("reservoir is uniform across items", spread < 0.15,
          f"spread {spread:.1%} around {expected:.0f}")

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--compare", action="store_true",
                   help="compare Flajolet-Martin combining rules")
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare_fm())
    raise SystemExit(verify() if a.verify else p.print_help())
