#!/usr/bin/env python3
"""Week 3 · Task 3 — Find the same pairs without comparing everything.

Textbook §3.4.

`BruteForce` compares every pair. On 3,000 documents that is 4.5 million
comparisons and it is completely correct. On 3 million documents it is 4.5
trillion and it is completely useless.

Beat it. Find the same near-duplicate pairs while making far fewer comparisons.

    python3 bench.py
    python3 bench.py --yours

The harness counts every call you make to `similarity()`. That is your score.
It also checks **recall** - which of the truly similar pairs you found. Skipping
comparisons is easy; skipping comparisons without losing the pairs is the task.
"""


import random

from task1_minhash import lsh_candidates


class BruteForce:
    """Correct, and quadratic."""

    def __init__(self, threshold):
        self.threshold = threshold

    def find(self, docs, similarity):
        """docs is [set_of_shingles, ...]. Return {(i, j), ...} with i < j."""
        out = set()
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                if similarity(docs[i], docs[j]) >= self.threshold:
                    out.add((i, j))
        return out


class YourFinder:
    """MinHash + LSH 근사 중복 탐지기.

    설계 선택 (observation.md 근거):
        해시 k = 100, 밴드 b = 20, 밴드당 r = 5
        S자 곡선의 급경사 위치 = (1/b)^(1/r) = (1/20)^(1/5) ~= 0.55
        기준 유사도 0.6보다 약간 아래에 두어, 경계 근처의 진짜 쌍을
        놓치지 않는 쪽을 택했다. 놓친 쌍은 되찾을 수 없지만 헛된 후보는
        마지막 정확 비교에서 걸러지기 때문이다.
    """

    def __init__(self, threshold, n_hashes=100, bands=20, seed=42):
        self.threshold = threshold
        self.bands = bands
        self.prime = 5003                      # 원소 값 범위(0~4999)보다 큰 소수
        rng = random.Random(seed)              # 씨앗 고정 -> 실행할 때마다 같은 결과
        self.hashes = [(rng.randrange(1, self.prime), rng.randrange(self.prime))
                       for _ in range(n_hashes)]

    def signature(self, doc):
        """문서 하나를 숫자 n_hashes개로 요약한다 (해시마다 최솟값 하나)."""
        if not doc:
            return [0] * len(self.hashes)
        return [min((a * x + b) % self.prime for x in doc) for a, b in self.hashes]

    def find(self, docs, similarity):
        """서명 -> 밴딩으로 후보 추출 -> 후보만 정확 비교."""
        sigs = [self.signature(d) for d in docs]
        candidates = lsh_candidates(sigs, self.bands)   # 집합이라 같은 쌍은 한 번만
        out = set()
        for i, j in candidates:
            if similarity(docs[i], docs[j]) >= self.threshold:   # 쌍마다 한 번만 호출
                out.add((i, j))
        return out
