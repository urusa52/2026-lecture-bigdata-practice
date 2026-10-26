#!/usr/bin/env python3
"""Week 7 · Task 1 — k-means, and the thing that decides the answer.

Textbook §7.1 (clustering, the curse of dimensionality), §7.3 (k-means).

k-means is four lines: pick centroids, assign every point to the nearest one,
move each centroid to the mean of its points, repeat. It always converges.

It converges to a **local** optimum, and which one depends entirely on where you
started. That is not a footnote - it is the difference between a clustering that
means something and one that does not, and you can watch it happen on nine points.

    python3 task1_kmeans.py --verify
"""
import argparse, math, random

# Three obvious blobs of three points each. Any sensible clustering finds them.
BLOBS = [(0.0, 0.0), (0.2, 0.1), (0.1, 0.2),
         (8.0, 8.0), (8.2, 8.1), (8.1, 8.2),
         (0.0, 8.0), (0.2, 8.1), (0.1, 8.2)]


def euclidean(a, b):
    """Distance between two points of any dimension."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def assign(points, centroids):
    """Nearest centroid for each point. Return a list of centroid indices.

    Ties go to the lower index, so that everybody's answer is the same.
    """
    labels = []
    for p in points:
        best, bi = None, 0
        for j, c in enumerate(centroids):
            d = euclidean(p, c)
            if best is None or d < best:      # strict '<': a tie keeps the lower j
                best, bi = d, j
        labels.append(bi)
    return labels


def update(points, labels, k, previous=None):
    """Move each centroid to the mean of its points.

    An empty cluster has no mean. Decide what you do - and say so, because the
    choice changes the result and there is more than one defensible answer.

    R3 choice: an empty cluster **keeps its previous centroid**. It costs
    nothing, never invents a point, and the centroid may win points back on a
    later pass. (The alternative - jump to the point furthest from its own
    centroid - repairs bad starts more aggressively but makes the result depend
    on one outlier.) `previous` is the centroid list from the last pass.
    """
    dim = len(points[0])
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k
    for p, l in zip(points, labels):
        counts[l] += 1
        s = sums[l]
        for i, v in enumerate(p):
            s[i] += v
    out = []
    for j in range(k):
        if counts[j]:
            out.append(tuple(v / counts[j] for v in sums[j]))
        elif previous is not None:
            out.append(tuple(previous[j]))
        else:
            raise ValueError(f"cluster {j} is empty and no previous centroid was given")
    return out


def _init_random(points, k, rng):
    """R4: k distinct data points chosen uniformly at random."""
    return [tuple(p) for p in rng.sample(points, k)]


def _init_plusplus(points, k, rng):
    """R5: k-means++ - first centre uniform, each next one with probability
    proportional to D(x)^2, the squared distance to the nearest centre so far."""
    cents = [tuple(rng.choice(points))]
    d2 = [euclidean(p, cents[0]) ** 2 for p in points]
    while len(cents) < k:
        total = sum(d2)
        if total == 0:                        # every point already sits on a centre
            nxt = rng.choice(points)
        else:
            r, acc, nxt = rng.random() * total, 0.0, points[-1]
            for p, w in zip(points, d2):
                acc += w
                if acc >= r and w > 0:
                    nxt = p
                    break
        cents.append(tuple(nxt))
        d2 = [min(old, euclidean(p, cents[-1]) ** 2) for p, old in zip(points, d2)]
    return cents


def kmeans(points, k, init="random", seed=246, max_iter=100):
    """Run to convergence. Return (labels, centroids, sse).

    `sse` is the sum over points of the squared distance to their own centroid.
    It is what k-means minimises, and it is how you compare two runs.

    `init` is either:
        "random"   pick k points at random as the starting centroids
        "++"       k-means++ (§7.3.2): pick the first at random, then pick each
                   next one with probability proportional to the squared distance
                   to the nearest centroid already chosen

    Set `kmeans.iterations` to how many passes you used.
    """
    rng = random.Random(seed)
    if init == "random":
        centroids = _init_random(points, k, rng)
    elif init == "++":
        centroids = _init_plusplus(points, k, rng)
    else:
        raise ValueError(f"unknown init {init!r}")

    labels = None
    kmeans.iterations = 0
    for it in range(1, max_iter + 1):
        new = assign(points, centroids)
        kmeans.iterations = it
        if new == labels:                     # nothing moved: converged
            break
        labels = new
        centroids = update(points, labels, k, previous=centroids)
    return labels, centroids, sse_of(points, labels, centroids)


def sse_of(points, labels, centroids):
    """Sum of squared distances from each point to its assigned centroid."""
    return sum(euclidean(p, centroids[l]) ** 2 for p, l in zip(points, labels))


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<50} {detail}")
        fails += not ok

    try:
        check("distance in 2d", abs(euclidean((0, 0), (3, 4)) - 5) < 1e-9)
        check("distance in 5d",
              abs(euclidean((0,) * 5, (1,) * 5) - math.sqrt(5)) < 1e-9)
    except NotImplementedError:
        print("  euclidean is still a stub"); return 1

    try:
        labels, cents, sse = kmeans(BLOBS, 3, init="++", seed=1)
    except NotImplementedError:
        print("  kmeans is still a stub"); return 1

    groups = {}
    for p, l in zip(BLOBS, labels):
        groups.setdefault(l, []).append(p)
    check("three clusters of three", sorted(len(g) for g in groups.values()) == [3, 3, 3],
          sorted(len(g) for g in groups.values()))
    check("the three blobs were recovered",
          all(max(q[0] for q in g) - min(q[0] for q in g) < 1 and
              max(q[1] for q in g) - min(q[1] for q in g) < 1
              for g in groups.values()))
    check("sse is small when clusters are right", sse < 1.0, f"sse {sse:.4f}")

    # Bad luck with random starts should sometimes be visibly worse.
    randoms = [kmeans(BLOBS, 3, init="random", seed=s)[2] for s in range(40)]
    plusplus = [kmeans(BLOBS, 3, init="++", seed=s)[2] for s in range(40)]
    check("random init sometimes lands badly", max(randoms) > min(randoms) * 5,
          f"sse from {min(randoms):.3f} to {max(randoms):.3f}")
    check("k-means++ is more reliable than random",
          sum(plusplus) / len(plusplus) <= sum(randoms) / len(randoms),
          f"mean sse: ++ {sum(plusplus) / len(plusplus):.3f} vs "
          f"random {sum(randoms) / len(randoms):.3f}")

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
