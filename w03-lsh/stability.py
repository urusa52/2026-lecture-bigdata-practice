#!/usr/bin/env python3
"""4,000 측정이 왜 느렸는지 원인을 좁히기 위한 보조 실험 (제출물 아님).

같은 작업을 여러 번 반복해 실행 시간의 분포를 본다.
  - 매번 비슷하면: 알고리즘/데이터 쪽 요인
  - 들쭉날쭉하면: 배경 부하나 CPU 스케줄링 등 기계 요인

    python stability.py                 # 8회 반복
    python stability.py --repeat 12 --n 800
"""
import argparse, statistics, time
import bench
from task3_scale import BruteForce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=700)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--label", default="")
    a = p.parse_args()

    bench.N_DOCS = a.n
    docs = bench.build()[:a.n]
    times = []
    for i in range(a.repeat):
        sim = bench.Counter()
        t0 = time.perf_counter()
        BruteForce(0.6).find(docs, sim)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  {i+1:>2}회  {dt:>7.3f}s   {dt / sim.calls * 1e6:>6.2f} us/cmp")

    lo, hi, med = min(times), max(times), statistics.median(times)
    print(f"\n  최소 {lo:.3f}s   중앙 {med:.3f}s   최대 {hi:.3f}s   "
          f"최대/최소 {hi/lo:.2f}배   표준편차 {statistics.pstdev(times):.3f}s")
    if hi / lo > 1.3:
        print("  -> 편차가 크다. 같은 코드·같은 데이터인데 시간이 달라졌다면 기계 요인이다.")
    else:
        print("  -> 편차가 작다. 이 조건에서는 측정이 안정적이다.")
    if a.label:
        print(f"  조건: {a.label}")


if __name__ == "__main__":
    main()
