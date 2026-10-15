#!/usr/bin/env python3
"""Week 6 · Task 4 (optional) — SON: A-Priori that survives being split up.

Textbook §6.4.3 (the SON algorithm) and §6.4.4 (SON and MapReduce).

Tasks 1 and 3 both assume the baskets fit on one machine. SON does not. It is
the textbook's own answer to "what if they do not", it is two passes, and it has
a guarantee that most sampling schemes do not:

    an itemset that is frequent overall is frequent in at least one chunk

which is why pass one can look at chunks independently and still miss nothing.
Proving that to yourself is worth more than the code.

    python3 task4_son.py                 # your SON, single process
    python3 task4_son.py --spark         # the same, as a real MapReduce job

Optional. Not part of the assignment.
"""
import argparse, time
from itertools import combinations


def son_pass_one(chunks, support, total_baskets):
    """Find the CANDIDATES: itemsets frequent in at least one chunk.

    Each chunk is a list of baskets. A chunk of size |chunk| gets a scaled
    threshold - §6.4.3 tells you what to scale it by, and getting that wrong is
    how you lose the guarantee.

    You may reuse your task 1 A-Priori on each chunk.

    Return a set of frozensets: every pair that was frequent somewhere.

    TASK 4a
    """
    from task1_apriori import frequent_pairs

    candidates = set()
    for chunk in chunks:
        if not chunk:
            continue
        # p = this chunk's share of the baskets; the chunk threshold is p * s.
        # Kept as an exact fraction-valued float, never rounded up past p * s:
        # counts are integers, so `count >= p*s` is the same as
        # `count >= ceil(p*s)`, and that is exactly what the proof needs.
        scaled = support * len(chunk) / total_baskets
        candidates.update(frequent_pairs(chunk, scaled))
    return candidates


def son_pass_two(baskets, candidates, support):
    """Count ONLY the candidates over ALL the baskets, and keep the real ones.

    Return {frozenset: count}. This is the pass that makes SON exact - pass one
    can produce false positives and pass two removes them. What it cannot
    produce is false negatives, and that is the whole design.

    TASK 4b

    Instead of testing every candidate against every basket (|baskets| x
    |candidates| subset tests), walk each basket's own pairs - restricted, as
    in task 1, to items that occur in some candidate - and look each one up in
    the candidate set. Work is per basket pair, not per candidate.
    """
    cands = {tuple(sorted(c)) for c in candidates}
    items = {i for pair in cands for i in pair}
    counts = dict.fromkeys(cands, 0)
    for basket in baskets:
        survivors = sorted(basket & items)
        for pair in combinations(survivors, 2):
            if pair in counts:
                counts[pair] += 1
    return {frozenset(p): c for p, c in counts.items() if c >= support}


def son_pass_two_naive(baskets, candidates, support):
    """The obvious pass two, for timing only: every candidate x every basket."""
    counts = dict.fromkeys(candidates, 0)
    for basket in baskets:
        for c in candidates:
            if c <= basket:
                counts[c] += 1
    return {c: n for c, n in counts.items() if n >= support}


# ------------------------------------------------------------------- harness
def chunked(baskets, n_chunks):
    size = max(1, (len(baskets) + n_chunks - 1) // n_chunks)
    return [baskets[i:i + size] for i in range(0, len(baskets), size)]


def spark_son(baskets, support, n_chunks):
    """SON as an actual two-pass MapReduce job. Given, for comparison.

    Read what each pass does. Pass one is a map over chunks with no shuffle
    between them - that independence is the point of §6.4.4.
    """
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[*]").appName("w06")
             .config("spark.ui.enabled", "false").getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    total = len(baskets)
    rdd = sc.parallelize(baskets, n_chunks).map(lambda b: sorted(b)).cache()

    def local_frequent(it):
        chunk = list(it)
        if not chunk:
            return []
        scaled = support * len(chunk) / total
        counts = {}
        for basket in chunk:
            for pair in combinations(basket, 2):
                counts[pair] = counts.get(pair, 0) + 1
        return [p for p, c in counts.items() if c >= scaled]

    candidates = set(rdd.mapPartitions(local_frequent).distinct().collect())
    bc = sc.broadcast(candidates)

    def count_candidates(it):
        counts = {}
        for basket in it:
            s = set(basket)
            for pair in bc.value:
                if s.issuperset(pair):
                    counts[pair] = counts.get(pair, 0) + 1
        return counts.items()

    final = (rdd.mapPartitions(count_candidates)
             .reduceByKey(lambda a, b: a + b)
             .filter(lambda kv: kv[1] >= support)
             .collectAsMap())
    spark.stop()
    return {frozenset(k): v for k, v in final.items()}, len(candidates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spark", action="store_true")
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--naive", action="store_true",
                   help="also time the obvious candidate-x-basket pass two")
    a = p.parse_args()

    import bench
    from task3_pcy import PlainApriori
    baskets = bench.build()
    support = bench.SUPPORT
    print(f"\n  {len(baskets):,} baskets, support {support}, "
          f"{a.chunks} chunks\n")

    t0 = time.perf_counter()
    truth = PlainApriori(support).run(baskets)
    print(f"  task 3 baseline   {time.perf_counter() - t0:7.2f}s   "
          f"pairs {len(truth):>6,}")

    try:
        t0 = time.perf_counter()
        cands = son_pass_one(chunked(baskets, a.chunks), support, len(baskets))
        t1 = time.perf_counter()
        found = son_pass_two(baskets, cands, support)
        t_son = time.perf_counter() - t0
        t_two = time.perf_counter() - t1
        missing = len(set(truth) - set(found))
        extra = len(set(found) - set(truth))
        print(f"  your SON          {t_son:7.2f}s   pairs {len(found):>6,}   "
              f"candidates {len(cands):>7,}   missing {missing}, extra {extra}")
        print(f"                    pass one {t1 - t0:5.2f}s, pass two {t_two:5.2f}s")
        if missing:
            print("    ^ SON must not have false negatives. Re-read §6.4.3 on "
                  "the scaled threshold.")
        if a.naive:
            t0 = time.perf_counter()
            slow = son_pass_two_naive(baskets, cands, support)
            print(f"  naive pass two    {time.perf_counter() - t0:7.2f}s   "
                  f"pairs {len(slow):>6,}   same {set(slow) == set(found)}")
    except NotImplementedError:
        print("  your SON          TASK 4a / 4b not implemented")

    if a.spark:
        try:
            t0 = time.perf_counter()
            sp, n_cand = spark_son(baskets, support, a.chunks)
            t_spark = time.perf_counter() - t0
            missing = len(set(truth) - set(sp))
            print(f"  spark SON         {t_spark:7.2f}s   pairs {len(sp):>6,}   "
                  f"candidates {n_cand:>7,}   missing {missing}")
        except Exception as e:
            print(f"  spark SON         did not run: {type(e).__name__}")
            print("  Record it in observation.md - this task is optional for that reason.")
    print()


if __name__ == "__main__":
    main()
