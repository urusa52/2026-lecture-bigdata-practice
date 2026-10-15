#!/usr/bin/env python3
"""Week 6 · Task 3, R5 — the honest memory accounting for PCY.

Not part of the harness. Produces the numbers quoted in observation.md.

    python3 task3_memory.py --buckets      # bucket-count sweep on the bench data
    python3 task3_memory.py --crossover    # at what data size does PCY win?

Memory model (textbook §6.2.1 - §6.3.1, 4-byte integers):
    A-Priori pass two   min(triangular 4 * C(k, 2),  triples 12 * distinct pairs)
    PCY pass one        4 * n_buckets                      (the fixed cost)
    PCY pass two        n_buckets / 8  (bitmap)  +  12 * candidate pairs
                        PCY cannot use the triangular matrix - its candidates
                        are scattered, so it is triples or nothing (§6.3.1)
    peak                max(pass one, pass two)
Item tables (one int per item) are the same for both and left out.
"""
import argparse, random, sys, time

import bench
from task3_pcy import PlainApriori, YourAlgorithm


def build(n_baskets, seed=bench.SEED):
    """bench.build() with a different basket count, same distribution."""
    rng = random.Random(seed)
    items = list(range(bench.ITEMS))
    weights = [1 / (i + 1) ** 0.8 for i in range(bench.ITEMS)]
    planted = [(rng.randrange(300), rng.randrange(300)) for _ in range(25)]
    planted = [(a, b) for a, b in planted if a != b]
    baskets = []
    for _ in range(n_baskets):
        k = max(2, int(rng.gauss(bench.AVG, 4)))
        basket = set(rng.choices(items, weights, k=k))
        if rng.random() < 0.08:
            a, b = rng.choice(planted)
            basket.update((a, b))
        baskets.append(basket)
    return baskets


def apriori_bytes(baskets, support):
    algo = PlainApriori(support)
    pairs = algo.run(baskets)
    counts = {}
    for b in baskets:
        for i in b:
            counts[i] = counts.get(i, 0) + 1
    k = sum(c >= support for c in counts.values())
    tri, tpl = 4 * k * (k - 1) // 2, 12 * algo.peak_counters
    return pairs, algo.peak_counters, min(tri, tpl), "tri" if tri <= tpl else "triples"


def pcy_bytes(baskets, support, nb):
    algo = YourAlgorithm(support, nb)
    pairs = algo.run(baskets)
    p1, p2 = 4 * nb, (nb + 7) // 8 + 12 * algo.peak_counters
    return pairs, algo, p1, p2


def kb(x):
    return f"{x / 1024:>9,.0f} KB"


def sweep_buckets():
    baskets = bench.build()
    s = bench.SUPPORT
    truth, base_peak, base_b, how = apriori_bytes(baskets, s)
    print(f"\n  A-Priori: {base_peak:,} counters -> {kb(base_b)} ({how})\n")
    print(f"  {'buckets':>10} {'freq bkts':>10} {'counters':>9} {'cut':>6}"
          f" {'pass one':>12} {'pass two':>12} {'peak':>12}  same")
    for nb in [1_000, 16_384, 65_536, 100_000, 131_072, 160_000, 200_000,
               262_144, 524_288, 1_048_576, 4_194_304, 16_777_216]:
        pairs, algo, p1, p2 = pcy_bytes(baskets, s, nb)
        cut = 1 - algo.peak_counters / base_peak
        print(f"  {nb:>10,} {algo.stats['frequent_buckets']:>10,} "
              f"{algo.peak_counters:>9,} {cut:>6.1%} {kb(p1)} {kb(p2)} "
              f"{kb(max(p1, p2))}  {set(pairs) == set(truth)}")


def crossover():
    print("\n  support = 0.25% of baskets (50 at 20,000), items fixed at 2,000\n")
    print(f"  {'baskets':>8} {'s':>4} {'A-Priori':>12}"
          f" | {'PCY 2^20':>12} {'win':>6} | {'best nb':>9} {'PCY tuned':>12} {'win':>6}")
    grid = [2 ** e for e in range(10, 23)]
    for n in [1_000, 2_500, 5_000, 10_000, 20_000, 40_000, 80_000]:
        baskets = build(n)
        s = max(2, round(n * 0.0025))
        truth, _, base_b, _ = apriori_bytes(baskets, s)
        _, _, p1, p2 = pcy_bytes(baskets, s, 1 << 20)
        fixed = max(p1, p2)
        best = None
        for nb in grid:
            pairs, _, p1, p2 = pcy_bytes(baskets, s, nb)
            assert set(pairs) == set(truth)
            if best is None or max(p1, p2) < best[1]:
                best = (nb, max(p1, p2))
        print(f"  {n:>8,} {s:>4} {kb(base_b)} | {kb(fixed)} {base_b / fixed:>5.2f}x"
              f" | {best[0]:>9,} {kb(best[1])} {base_b / best[1]:>5.2f}x")
        sys.stdout.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--buckets", action="store_true")
    p.add_argument("--crossover", action="store_true")
    a = p.parse_args()
    if a.buckets:
        sweep_buckets()
    if a.crossover:
        crossover()
    if not (a.buckets or a.crossover):
        p.print_help()
