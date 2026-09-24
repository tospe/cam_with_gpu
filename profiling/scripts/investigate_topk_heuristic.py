#!/usr/bin/env python3
"""Explain the non-monotonic torch.topk cost seen in the DSA sweep.

Selection at N = 16 384 is slower than at N = 32 768. This script shows why:
``torch.topk`` on CUDA picks between two implementations with a hard-coded
heuristic, and at N = 16 384 with a small batch it picks the one that leaves
almost the whole GPU idle.

Run:  source scripts/env.sh && $PYTHON scripts/investigate_topk_heuristic.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile


def uses_multiblock(b: int, n: int, k: int = 64) -> bool:
    """True if this shape dispatches to the multi-block (`mbtopk`) kernels."""
    x = torch.randn(b, n, device="cuda", dtype=torch.float32)
    for _ in range(3):
        torch.topk(x, min(k, n), dim=-1, sorted=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            torch.topk(x, min(k, n), dim=-1, sorted=False)
        torch.cuda.synchronize()
    return any(
        "mbtopk" in e.key
        for e in p.key_averages()
        if (getattr(e, "device_time_total", 0) or 0) > 0
    )


def latency_us(b: int, n: int, k: int, iters: int = 50) -> float:
    x = torch.randn(b, n, device="cuda", dtype=torch.float32)
    for _ in range(10):
        torch.topk(x, min(k, n), dim=-1, sorted=False)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        torch.topk(x, min(k, n), dim=-1, sorted=False)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def find_switch(b: int, lo: int = 500, hi: int = 40000) -> int | None:
    """Bisect the slice_size at which this batch size switches to multi-block."""
    if not uses_multiblock(b, hi):
        return None
    while lo < hi - 50:
        mid = (lo + hi) // 2
        if uses_multiblock(b, mid):
            hi = mid
        else:
            lo = mid
    return hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--out", default=None, help="write a CSV here")
    a = ap.parse_args()

    rows = []
    print(f"torch {torch.__version__}, {torch.cuda.get_device_properties(0).name}, "
          f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs\n")

    print("(1) Latency and kernel family vs N, at B = 1, k = %d" % a.k)
    print(f"    {'N':>8} {'us':>8}   kernel family")
    for n in (4096, 8192, 12288, 16384, 19000, 20000, 20480, 24576, 32768, 65536, 131072):
        t = latency_us(1, n, a.k)
        fam = "mbtopk (multi-block)" if uses_multiblock(1, n) else "single-block gatherTopK"
        print(f"    {n:>8} {t:>8.2f}   {fam}")
        rows.append({"probe": "vs_N", "batch": 1, "n": n, "k": a.k,
                     "latency_us": round(t, 2), "multiblock": fam.startswith("mbtopk")})

    print("\n(2) The switch point depends on the number of rows (num_slices)")
    print(f"    {'B':>6} {'switch N':>10}")
    for b in (1, 20, 21, 40, 41, 80, 81, 200, 1000):
        sw = find_switch(b)
        print(f"    {b:>6} {str(sw):>10}")
        rows.append({"probe": "switch_point", "batch": b, "n": sw, "k": 64,
                     "latency_us": None, "multiblock": None})

    print("\n(3) Single-block path is one block per row: latency is flat in B")
    print(f"    {'B':>4} {'N=16384 (single)':>18} {'N=20480 (multi)':>18}")
    for b in (1, 2, 4, 8, 16, 20):
        t1, t2 = latency_us(b, 16384, a.k), latency_us(b, 20480, a.k)
        print(f"    {b:>4} {t1:>18.1f} {t2:>18.1f}")
        rows.append({"probe": "batch_scaling", "batch": b, "n": 16384, "k": a.k,
                     "latency_us": round(t1, 2), "multiblock": False})
        rows.append({"probe": "batch_scaling", "batch": b, "n": 20480, "k": a.k,
                     "latency_us": round(t2, 2), "multiblock": True})

    print("\n(4) Adding one more row makes the SAME N faster (tier boundary at B=20/21)")
    for b in (20, 21):
        t = latency_us(b, 16384, a.k)
        print(f"    B={b:<3} N=16384: {t:6.1f} us   multiblock={uses_multiblock(b, 16384)}")
        rows.append({"probe": "tier_boundary", "batch": b, "n": 16384, "k": a.k,
                     "latency_us": round(t, 2), "multiblock": uses_multiblock(b, 16384)})

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
