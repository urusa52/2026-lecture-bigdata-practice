#!/usr/bin/env python3
"""Week 7 · Task 3 — The same clustering, with far fewer distances.

Textbook §7.3, and §7.1.3 for why distance is the expensive part.

`NaiveKMeans` does what the definition says: every iteration, compare every
point with every centroid. That is n x k distance computations per pass, and on
a real dataset the distance computation is almost the entire cost.

Most of that work is wasted. After the first couple of iterations, most points
do not change cluster, and most centroids barely move. The triangle inequality
lets you prove a point cannot have changed cluster **without measuring it**.

    python3 bench.py
    python3 bench.py --yours

Correctness first: your final SSE must match the baseline's. Fewer distances
with a worse clustering is not an improvement, it is a different algorithm.
"""


class NaiveKMeans:
    """Every point against every centroid, every iteration."""

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter

    def run(self, points, initial_centroids, distance):
        centroids = list(initial_centroids)
        labels = [None] * len(points)
        for self.iterations in range(1, self.max_iter + 1):
            changed = False
            for i, p in enumerate(points):
                best, bi = None, 0
                for j, c in enumerate(centroids):
                    d = distance(p, c)
                    if best is None or d < best - 1e-12:
                        best, bi = d, j
                if labels[i] != bi:
                    labels[i] = bi
                    changed = True
            centroids = _means(points, labels, self.k, centroids)
            if not changed:
                break
        return labels, centroids


def _means(points, labels, k, previous):
    out = []
    for j in range(k):
        group = [p for p, l in zip(points, labels) if l == j]
        if group:
            out.append(tuple(sum(v) / len(group) for v in zip(*group)))
        else:
            out.append(previous[j])
    return out


class YourKMeans:
    """Elkan's k-means (2003): Lloyd's algorithm with triangle-inequality bounds.

        __init__(k, max_iter=50)
        run(points, initial_centroids, distance) -> (labels, centroids)

    Same answer as NaiveKMeans - it never approximates, it only skips distances
    it can prove would not change the assignment.

    Per point it stores
        a[i]      its centroid                                   (the label)
        u[i]      an UPPER bound on d(x_i, c_a[i])               1 number
        L[i][j]   a LOWER bound on d(x_i, c_j) for every j       k numbers
    so k + 1 = 13 extra numbers per point here (20,000 x 13 floats).

    Per iteration (k = 12, so tiny next to n = 20,000):
        cc[j][j'] = d(c_j, c_j') / 2       k(k-1)/2 = 66 distances
        s[j]      = min_j' cc[j][j']       half the gap to the nearest other centroid
        mv[j]     = d(c_j old, c_j new)    k = 12 distances

    Three tests, cheapest first. A point x with centroid c = c_a:
      1. u <= s[a]           -> no other centroid can be closer: skip x entirely.
         (d(x,c') >= d(c,c') - d(x,c) >= 2 s - u >= u)
      2. u <= L[j]           -> c_j is provably no closer: skip that j.
      3. u <= cc[a][j]       -> same by the centroid gap: skip that j.
    Only if a j survives all three do we tighten u (1 distance) and then, if it
    still survives, measure d(x, c_j).

    After the centroids move, every bound is repaired without touching points:
        u[i]    += mv[a[i]]        (own centroid moved at most this far away)
        L[i][j] -= mv[j]           (c_j came at most this much closer)
    """

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter
        self.stats = {}

    def run(self, points, initial_centroids, distance):
        k, n = self.k, len(points)
        C = [tuple(c) for c in initial_centroids]
        a = [0] * n
        u = [0.0] * n
        L = [[0.0] * k for _ in range(n)]
        st = self.stats = {"first_iter": 0, "centroid_pairs": 0, "skip_point": 0,
                           "skip_lower": 0, "skip_gap": 0, "tighten": 0,
                           "measured": 0}

        for self.iterations in range(1, self.max_iter + 1):
            changed = False
            if self.iterations == 1:
                # Nothing is known yet: the one pass that measures everything.
                for i, p in enumerate(points):
                    Li, best, bi = L[i], None, 0
                    for j, c in enumerate(C):
                        d = distance(p, c)
                        Li[j] = d
                        if best is None or d < best - 1e-12:
                            best, bi = d, j
                    a[i], u[i] = bi, best
                changed = True
                st["first_iter"] = distance.calls
            else:
                cc = [[0.0] * k for _ in range(k)]
                for j in range(k):
                    for jj in range(j + 1, k):
                        h = distance(C[j], C[jj]) / 2
                        cc[j][jj] = cc[jj][j] = h
                st["centroid_pairs"] += k * (k - 1) // 2
                s = [min(cc[j][jj] for jj in range(k) if jj != j) for j in range(k)]

                for i, p in enumerate(points):
                    ai, ui = a[i], u[i]
                    if ui <= s[ai]:                       # test 1
                        st["skip_point"] += 1
                        continue
                    Li, cca, tight = L[i], cc[ai], False
                    for j in range(k):
                        if j == ai:
                            continue
                        if ui <= Li[j]:                   # test 2
                            st["skip_lower"] += 1
                            continue
                        if ui <= cca[j]:                  # test 3
                            st["skip_gap"] += 1
                            continue
                        if not tight:                     # make u exact, re-test
                            ui = distance(p, C[ai])
                            Li[ai], tight = ui, True
                            st["tighten"] += 1
                            if ui <= Li[j] or ui <= cca[j]:
                                continue
                        d = distance(p, C[j])
                        st["measured"] += 1
                        Li[j] = d
                        # same tie rule as the baseline: lower index wins a tie
                        if d < ui - 1e-12 or (d <= ui + 1e-12 and j < ai):
                            ai, ui, cca = j, d, cc[j]
                    if ai != a[i]:
                        changed = True
                    a[i], u[i] = ai, ui

            new = _means(points, a, k, C)
            if not changed:
                C = new
                break
            mv = [distance(C[j], new[j]) for j in range(k)]
            st["centroid_pairs"] += k
            C = new
            for i in range(n):                            # repair the bounds
                Li = L[i]
                for j in range(k):
                    v = Li[j] - mv[j]
                    Li[j] = v if v > 0 else 0.0
                u[i] += mv[a[i]]
        return list(a), C


class HamerlyKMeans:
    """Hamerly's k-means (2010), kept for comparison - not what bench.py scores.

    Only 2 bounds per point instead of k + 1: u[i] (upper, own centroid) and
    l[i] (lower, the SECOND-nearest centroid). One test per point:
        u[i] <= max(s[a[i]], l[i])  -> skip.
    Less memory, weaker pruning: about 89% cut here against Elkan's 96%.
    """

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter

    def run(self, points, initial_centroids, distance):
        k, n = self.k, len(points)
        C = [tuple(c) for c in initial_centroids]
        a, u, l = [0] * n, [0.0] * n, [0.0] * n

        def full(i, p, known=None):
            best, bi, second = None, 0, float("inf")
            for j, c in enumerate(C):
                d = known if (known is not None and j == a[i]) else distance(p, c)
                if best is None or d < best - 1e-12:
                    if best is not None:
                        second = min(second, best)
                    best, bi = d, j
                else:
                    second = min(second, d)
            return bi, best, second

        for self.iterations in range(1, self.max_iter + 1):
            changed = False
            if self.iterations == 1:
                for i, p in enumerate(points):
                    a[i], u[i], l[i] = full(i, p)
                changed = True
            else:
                s = [float("inf")] * k
                for j in range(k):
                    for jj in range(j + 1, k):
                        d = distance(C[j], C[jj])
                        s[j], s[jj] = min(s[j], d), min(s[jj], d)
                s = [x / 2 for x in s]
                for i, p in enumerate(points):
                    m = max(s[a[i]], l[i])
                    if u[i] <= m:
                        continue
                    u[i] = distance(p, C[a[i]])
                    if u[i] <= m:
                        continue
                    bi, best, second = full(i, p, known=u[i])
                    changed |= bi != a[i]
                    a[i], u[i], l[i] = bi, best, second
            new = _means(points, a, k, C)
            if not changed:
                C = new
                break
            mv = [distance(C[j], new[j]) for j in range(k)]
            C = new
            r1, r2 = sorted(range(k), key=lambda j: -mv[j])[:2]
            for i in range(n):
                u[i] += mv[a[i]]
                l[i] -= mv[r2] if a[i] == r1 else mv[r1]
        return list(a), C
