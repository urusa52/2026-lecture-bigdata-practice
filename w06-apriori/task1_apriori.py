#!/usr/bin/env python3
"""Week 6 · Task 1 — A-Priori, and why the second pass is the whole problem.

Textbook §6.1 (market-basket model), §6.2 (A-Priori).

Finding frequent pairs is easy until you count how many pairs there are. With
n items there are n(n-1)/2 of them, and for a supermarket's 10,000 items that
is 50 million counters before you have read a single basket.

A-Priori's observation is one sentence long: **a pair cannot be frequent unless
both of its items are frequent.** One extra pass over the data buys you a much
smaller second pass, and the whole chapter is consequences of that trade.

    python3 task1_apriori.py --verify
"""
import argparse
from itertools import combinations

# Small enough to check by hand. support threshold 3 unless stated.
BASKETS = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "egg"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
    {"bread", "milk", "beer"},
    {"diaper", "beer"},
]


def frequent_singletons(baskets, support):
    """Items appearing in at least `support` baskets. Return {item: count}."""
    # pass one: one counter per item - the only thing pass one needs
    counts = {}
    for basket in baskets:
        for item in basket:
            counts[item] = counts.get(item, 0) + 1
    return {i: c for i, c in counts.items() if c >= support}


def frequent_pairs(baskets, support):
    """Pairs appearing in at least `support` baskets, using A-Priori.

    Return {frozenset({a, b}): count}.

    The requirement that makes this A-Priori rather than brute force: pass two
    must only ever count pairs **both of whose items were frequent in pass one**.
    If you count every pair and filter afterwards you have written the thing
    A-Priori exists to avoid, and it will pass this harness while failing the
    point. Task 3 measures whether you actually did it.
    """
    frequent = frequent_singletons(baskets, support)

    # pass two: a candidate pair exists only if BOTH items survived pass one.
    # Infrequent items are dropped from each basket before pairs are formed,
    # so a counter is never created for a pair that A-Priori rules out.
    counts = {}
    for basket in baskets:
        survivors = sorted(i for i in basket if i in frequent)
        for a, b in combinations(survivors, 2):
            key = frozenset((a, b))
            counts[key] = counts.get(key, 0) + 1
    return {p: c for p, c in counts.items() if c >= support}


def association_rules(baskets, support, min_confidence):
    """Rules I -> j from the frequent pairs.

    For a frequent pair {i, j}, confidence of i -> j is
    support({i, j}) / support({i}).

    Return [(antecedent, consequent, confidence, lift), ...] for rules whose
    confidence is at least `min_confidence`, sorted by confidence descending.

    `lift` is confidence divided by the consequent's own support fraction. A
    rule with high confidence and lift near 1 tells you nothing - the consequent
    was common anyway - and §6.1.3 is about why that matters more than it looks.
    """
    n = len(baskets)
    singles = frequent_singletons(baskets, support)
    pairs = frequent_pairs(baskets, support)

    rules = []
    for pair, both in pairs.items():
        a, b = tuple(pair)
        # each frequent pair gives two rules, and they are not the same rule:
        # the denominator is the antecedent's support, which differs
        for i, j in ((a, b), (b, a)):
            confidence = both / singles[i]
            if confidence >= min_confidence:
                lift = confidence / (singles[j] / n)
                rules.append((i, j, confidence, lift))
    rules.sort(key=lambda r: (-r[2], -r[3], str(r[0]), str(r[1])))
    return rules


def bench_report(top=10):
    """Rules on the task 3 benchmark data, for observation.md."""
    import time
    import bench
    baskets = bench.build()
    n, s = len(baskets), bench.SUPPORT

    t0 = time.perf_counter()
    singles = frequent_singletons(baskets, s)
    pairs = frequent_pairs(baskets, s)
    rules = association_rules(baskets, s, 0.0)
    elapsed = time.perf_counter() - t0

    k = len(singles)
    print(f"\n  {n:,} baskets, {bench.ITEMS:,} items, support {s}   ({elapsed:.2f}s)")
    print(f"  brute-force pair counters  C({bench.ITEMS:,}, 2) = "
          f"{bench.ITEMS * (bench.ITEMS - 1) // 2:,}")
    print(f"  frequent items after pass one = {k:,}  ->  "
          f"candidate pairs C({k}, 2) = {k * (k - 1) // 2:,}")
    print(f"  frequent pairs = {len(pairs):,}\n")

    print(f"  top {top} rules by confidence")
    for i, j, conf, lift in rules[:top]:
        print(f"    {i:>5} -> {j:<5} conf {conf:.3f}  lift {lift:6.2f}   "
              f"sup(i,j) {pairs[frozenset((i, j))]:>4}  sup(i) {singles[i]:>5}  "
              f"sup(j) {singles[j]:>5}")

    # the most lopsided pair: same support({i,j}), very different confidences
    by_rule = {(i, j): conf for i, j, conf, _ in rules}
    gap = max(pairs, key=lambda p: abs(by_rule[tuple(p)] - by_rule[tuple(p)[::-1]]))
    a, b = sorted(gap, key=lambda x: singles[x])
    print(f"\n  most asymmetric pair  {{{a}, {b}}}  support {pairs[gap]}")
    print(f"    {a} -> {b}  conf {by_rule[(a, b)]:.3f}   (sup {a} = {singles[a]})")
    print(f"    {b} -> {a}  conf {by_rule[(b, a)]:.3f}   (sup {b} = {singles[b]})")
    high = [r for r in rules if r[2] >= 0.5]
    print(f"\n  rules with confidence >= 0.5: {len(high)}, of which lift >= 2: "
          f"{sum(r[3] >= 2 for r in high)}")


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<44} {got}"
              + ("" if ok else f"\n{'':>54}want {want}"))
        fails += not ok

    try:
        singles = frequent_singletons(BASKETS, 3)
    except NotImplementedError:
        print("  frequent_singletons is still a stub"); return 1
    # counted by hand: bread 5, milk 5, diaper 5, beer 5, cola 2, egg 1
    check("frequent singletons at support 3",
          dict(sorted(singles.items())),
          {"beer": 5, "bread": 5, "diaper": 5, "milk": 5})

    try:
        pairs = frequent_pairs(BASKETS, 3)
    except NotImplementedError:
        print("  frequent_pairs is still a stub"); return 1
    got = {tuple(sorted(k)): v for k, v in pairs.items()}
    check("frequent pairs at support 3", dict(sorted(got.items())),
          {("beer", "bread"): 3, ("beer", "diaper"): 4, ("beer", "milk"): 3,
           ("bread", "diaper"): 3, ("bread", "milk"): 4, ("diaper", "milk"): 3})

    # cola appears twice, so it must never reach pass two
    check("infrequent items excluded from pairs",
          any("cola" in k or "egg" in k for k in got), False)

    try:
        rules = association_rules(BASKETS, 3, 0.75)
    except NotImplementedError:
        print("  association_rules is still a stub"); return 1
    # diaper -> beer: support 4, diaper 5, confidence 0.8
    hit = [r for r in rules if r[0] == "diaper" and r[1] == "beer"]
    check("diaper -> beer confidence", round(hit[0][2], 3) if hit else None, 0.8)
    check("rules are sorted by confidence",
          all(rules[i][2] >= rules[i + 1][2] for i in range(len(rules) - 1)), True)

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--bench", action="store_true",
                   help="rules and counter counts on the task 3 benchmark data")
    p.add_argument("--rules", action="store_true",
                   help="print every rule on the toy baskets at support 3")
    a = p.parse_args()
    if a.bench:
        bench_report()
        raise SystemExit(0)
    if a.rules:
        for i, j, conf, lift in association_rules(BASKETS, 3, 0.0):
            print(f"  {i:>7} -> {j:<7} confidence {conf:.3f}   lift {lift:.3f}")
        raise SystemExit(0)
    raise SystemExit(verify() if a.verify else p.print_help())
