#!/usr/bin/env python3
"""Week 4 · Task 2 — Find the size where exact stops being possible.

Textbook §4.1 (the stream model), §4.4, §4.5.

Sketches exist because the exact answer does not fit. That sentence is easy to
agree with and hard to feel, so this task makes you watch it happen on your own
machine: hold every distinct item in a set, keep raising the stream size, and
record where your laptop stops coping.

    python3 task2_limits.py --sizes 100000,400000,1600000
    python3 task2_limits.py --sizes 6400000            # keep going

Your numbers will not match anybody else's. That is the point.
"""
import argparse, json, os, platform, time, tracemalloc


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def machine():
    return {"platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version()}


def stream(n, distinct_ratio=0.4, seed=246):
    """A stream of n items with about n*distinct_ratio distinct values."""
    import random
    rng = random.Random(seed)
    span = max(1, int(n * distinct_ratio))
    for _ in range(n):
        yield f"key-{rng.randrange(span)}"


def exact_distinct(n):
    """The honest answer: hold every distinct item."""
    tracemalloc.start()
    t0 = time.perf_counter()
    seen = set()
    for x in stream(n):
        seen.add(x)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return len(seen), elapsed, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="100000,400000,1600000")
    p.add_argument("--exact-only", action="store_true",
                   help="skip Flajolet-Martin: for sizes where only the exact "
                        "set is being pushed to its limit")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    try:
        from task1_sketches import flajolet_martin
    except Exception:
        flajolet_martin = None

    rows = []
    for n in [int(x) for x in a.sizes.split(",")]:
        true, t_exact, m_exact = exact_distinct(n)
        row = {"n": n, "true_distinct": true, "exact_s": t_exact,
               "exact_peak_bytes": m_exact}

        if a.exact_only:
            row["fm_skipped"] = True
        elif flajolet_martin is not None:
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                est = flajolet_martin(stream(n))
                t_fm = time.perf_counter() - t0
                _, m_fm = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                row.update({"fm_estimate": est, "fm_s": t_fm,
                            "fm_peak_bytes": m_fm,
                            "fm_ratio": est / true if true else None})
            except NotImplementedError:
                tracemalloc.stop()

        rows.append(row)
        line = (f"  n={n:>10,}  distinct {true:>9,}   exact {t_exact:>7.2f}s "
                f"{m_exact / 1e6:>8.1f} MB")
        if "fm_s" in row:
            line += (f"   |  fm {row['fm_s']:>7.2f}s {row['fm_peak_bytes'] / 1e6:>6.2f} MB"
                     f"  {row['fm_ratio']:.2f}x")
        print(line)

    path = os.path.join(OUT, "limits.json")
    prior = json.load(open(path)) if os.path.exists(path) else {"runs": []}
    prior["machine"] = machine()
    prior["runs"].extend(rows)
    json.dump(prior, open(path, "w"), indent=2)
    print(f"\n  -> out/limits.json  ({len(prior['runs'])} measurement(s))")
    print("  Keep raising --sizes until the exact version is unbearable. "
          "Record where, and what ran out.")


if __name__ == "__main__":
    main()
