#!/usr/bin/env python3
"""Week 5 · Task 1 — PageRank, and the two ways it breaks.

Textbook §5.1 (PageRank), §5.1.3 (dead ends), §5.1.4 (spider traps).

PageRank is a random surfer following links forever, and the rank of a page is
how often the surfer is there. As one line of linear algebra it is
`r = M r`, and as written it does not work on the actual web.

Two structures break it, and the fix for both is the same one line. Build the
broken version first so you can see each failure, then fix it.

    python3 task1_pagerank.py --verify
"""
import argparse

# A: -> B, C      B: -> C      C: -> A
SIMPLE = {"A": ["B", "C"], "B": ["C"], "C": ["A"]}

# C has no out-links at all. Rank leaks out of the graph and everything -> 0.
DEAD_END = {"A": ["B", "C"], "B": ["C"], "C": []}

# C and D only point at each other. They absorb everything.
SPIDER_TRAP = {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["C"]}


def pagerank(graph, beta=0.85, iterations=100, tol=1e-10):
    """Rank every node. Return {node: rank}, summing to 1.

    `beta` is the probability the surfer follows a link. With probability
    1 - beta they teleport to a node chosen uniformly.

    You have to handle both of these, and the textbook handles them the same way:

      dead ends    a node with no out-links. Where does its rank go, and where
                   should it go instead?
      spider traps a group of nodes that only link to each other. Without
                   teleporting, they end up with all of it

    Stop early when the ranks stop moving - `tol` is the L1 change below which
    you should call it converged. Return the ranks, and set `pagerank.iterations`
    to how many you actually used, because Task 2 measures that.
    """
    nodes = _all_nodes(graph)
    n = len(nodes)
    r = {v: 1.0 / n for v in nodes}                  # 시작: 균등 분포

    used = 0
    for used in range(1, iterations + 1):
        new = {v: 0.0 for v in nodes}
        dead_mass = 0.0
        for v in nodes:
            outs = graph.get(v, [])
            if outs:                                  # 링크를 따라 균등 분배
                share = r[v] / len(outs)
                for w in outs:
                    new[w] += share
            else:                                     # dead end: 모아뒀다가
                dead_mass += r[v]
        # dead end 의 rank 를 모든 노드에 균등하게 돌려준다 (전체 합 보존)
        # + 확률 1-beta 로 아무 노드로 teleport (spider trap 탈출)
        base = beta * dead_mass / n + (1 - beta) / n
        new = {v: beta * new[v] + base for v in nodes}

        delta = sum(abs(new[v] - r[v]) for v in nodes)   # L1 변화량
        r = new
        if delta < tol:
            break

    pagerank.iterations = used
    return r


def pagerank_no_teleport(graph, iterations=100):
    """The broken version: beta = 1, no teleporting. Build this too.

    It exists so you can watch both failures happen rather than take them on
    trust. The harness checks that it really does fail.
    """
    nodes = _all_nodes(graph)
    n = len(nodes)
    r = {v: 1.0 / n for v in nodes}

    for _ in range(iterations):
        new = {v: 0.0 for v in nodes}
        for v in nodes:
            outs = graph.get(v, [])
            for w in outs:                    # 링크만 따라간다 (r = M r)
                new[w] += r[v] / len(outs)
            # dead end 의 rank 는 아무 데도 안 가고 그냥 사라진다
            # teleport 가 없으니 trap 에 들어간 rank 는 빠져나오지 못한다
        r = new
    return r


def _all_nodes(graph):
    """키뿐 아니라 링크 대상으로만 등장하는 노드도 포함 (순서 유지)."""
    seen = dict.fromkeys(graph)
    for outs in graph.values():
        for w in outs:
            seen.setdefault(w, None)
    return list(seen)


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<48} {detail}")
        fails += not ok

    try:
        r = pagerank(SIMPLE)
    except NotImplementedError:
        print("  pagerank is still a stub"); return 1

    check("ranks sum to 1", abs(sum(r.values()) - 1) < 1e-6, f"{sum(r.values()):.6f}")
    check("every node ranked", set(r) == set(SIMPLE), sorted(r))
    # C is pointed at by both A and B, so it has to come first.
    check("C outranks A and B", r["C"] > r["A"] and r["C"] > r["B"],
          {k: round(v, 4) for k, v in sorted(r.items())})

    try:
        broken = pagerank_no_teleport(DEAD_END)
    except NotImplementedError:
        print("  pagerank_no_teleport is still a stub"); return 1
    check("without the fix, a dead end drains the graph",
          sum(broken.values()) < 0.5, f"total rank {sum(broken.values()):.4f}")

    fixed = pagerank(DEAD_END)
    check("with the fix, rank is conserved",
          abs(sum(fixed.values()) - 1) < 1e-6, f"{sum(fixed.values()):.6f}")

    trapped = pagerank_no_teleport(SPIDER_TRAP)
    check("without the fix, a spider trap takes everything",
          trapped["C"] + trapped["D"] > 0.95,
          f"C+D = {trapped['C'] + trapped['D']:.4f}")

    untrapped = pagerank(SPIDER_TRAP)
    check("with the fix, A and B keep some rank",
          untrapped["A"] > 0.01 and untrapped["B"] > 0.01,
          {k: round(v, 4) for k, v in sorted(untrapped.items())})

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
