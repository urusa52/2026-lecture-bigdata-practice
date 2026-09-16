#!/usr/bin/env python3
"""Week 3 · Task 1 — Minhash and LSH, built from the matrix up.

Textbook §3.2 - §3.4.

Comparing every pair is quadratic, so it stops being possible somewhere around
a hundred thousand documents. The way out is two ideas stacked:

    minhash   replace a set with a short signature, such that the chance two
              signatures agree in a position equals their Jaccard similarity
    LSH       hash bands of those signatures so that similar pairs collide and
              you only ever compare the ones that did

You build both. The textbook's §3.3.5 example is small enough to check by hand,
and the harness checks you against it.

    python3 task1_minhash.py --verify
"""
import argparse

# §3.3.5. Rows are elements 0..4, columns are the sets S1..S4.
BOOK = [[1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 0]]
# The two hash functions the textbook uses on the row numbers.
BOOK_HASHES = [lambda r: (r + 1) % 5, lambda r: (3 * r + 1) % 5]


def jaccard(a, b):
    """|a and b| / |a or b|. Empty union is 0, not an error."""
    union = a | b
    if not union:          # 합집합이 비어 있으면
        return 0
    return len(a & b) / len(union)


def minhash_signatures(columns, hashes, n_rows):
    """(설명 문자열은 그대로 두세요)"""
    # 1) 열 기준 → 행 기준으로 뒤집기: rows[r] = 행 r에 1이 있는 열들
    rows = [[] for _ in range(n_rows)]
    for c, col in enumerate(columns):
        for r in col:
            rows[r].append(c)

    # 2) 서명을 무한대로 초기화: sig[열][해시]
    INF = float("inf")
    sig = [[INF] * len(hashes) for _ in columns]

    # 3) 행을 한 번씩만 순회
    for r in range(n_rows):
        hv = [h(r) for h in hashes]        # 이 행의 해시값들 (행마다 한 번만 계산)
        for c in rows[r]:                  # 이 행에 1이 있는 열만
            for i, v in enumerate(hv):
                if v < sig[c][i]:
                    sig[c][i] = v
    return sig


def lsh_candidates(signatures, bands):
    """(설명 문자열은 그대로 두세요)"""
    from collections import defaultdict
    from itertools import combinations

    if not signatures:
        return set()
    n = len(signatures[0])                 # 서명 길이
    # R5: 나눠떨어지지 않으면 오류. S자 곡선 공식이 모든 밴드 길이가 같다고 가정하기 때문
    if bands <= 0 or n % bands != 0:
        raise ValueError(f"signature length {n} is not divisible by bands={bands}")
    r = n // bands                         # 밴드당 칸 수

    buckets = defaultdict(list)            # (밴드 번호, 구간 값) -> 문서 번호 목록
    for doc, sig in enumerate(signatures):
        for b in range(bands):
            key = (b, tuple(sig[b * r:(b + 1) * r]))
            buckets[key].append(doc)

    cands = set()
    for docs in buckets.values():
        for i, j in combinations(docs, 2):  # 같은 버킷에 든 모든 쌍
            cands.add((i, j))               # enumerate 순서라 항상 i < j
    return cands


# ------------------------------------------------------------------- harness
def columns_from_matrix(matrix):
    n_rows, n_cols = len(matrix), len(matrix[0])
    return [{r for r in range(n_rows) if matrix[r][c]} for c in range(n_cols)]


def verify():
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<44} {got}"
              + ("" if ok else f"\n{'':>54}want {want}"))
        fails += not ok

    cols = columns_from_matrix(BOOK)
    try:
        # S1 = {0,3}, S4 = {0,2,3}: intersection 2, union 3
        check("jaccard(S1, S4)", round(jaccard(cols[0], cols[3]), 4), round(2 / 3, 4))
        check("jaccard(S1, S2)", jaccard(cols[0], cols[1]), 0.0)
        check("jaccard on empty sets", jaccard(set(), set()), 0)
    except NotImplementedError:
        print("  jaccard is still a stub"); return 1

    try:
        sig = minhash_signatures(cols, BOOK_HASHES, len(BOOK))
    except NotImplementedError:
        print("  minhash_signatures is still a stub"); return 1

    # Figure 3.4 in the textbook.
    check("signature of S1", sig[0], [1, 0])
    check("signature of S2", sig[1], [3, 2])
    check("signature of S3", sig[2], [0, 0])
    check("signature of S4", sig[3], [1, 0])

    try:
        cands = lsh_candidates([[1, 0], [3, 2], [0, 0], [1, 0]], bands=2)
    except NotImplementedError:
        print("  lsh_candidates is still a stub"); return 1
    # With one row per band, S1 and S4 are identical, so they must collide.
    check("S1 and S4 are candidates", (0, 3) in cands, True)
    check("S1 and S2 are not", (0, 1) in cands, False)

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    if not fails:
        print("  Note that S1 and S4 agree in both signature positions, which "
              "estimates\n  their similarity as 1.0 when it is actually 2/3. "
              "Two hashes is not many.")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
