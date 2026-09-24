#!/usr/bin/env python3
"""Why router scoring overtakes selection, and where.

Sweeps E on a fine grid across the transition, recording scoring latency,
selection latency, and the effective read bandwidth of the router table, then
plots the two regimes.

Run:  source scripts/env.sh && $PYTHON scripts/investigate_router_crossover.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.backends.base import RouterConfig
from benchmarks.backends.router_torch import TorchRouterBackend

E_GRID = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
          65536, 131072, 262144, 524288, 1048576]


def bench(fn, iters=100):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-dir", default="results/router-full")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    rows = []
    print(f"L2 = {l2/2**20:.0f} MiB;  D={a.dim}, k={a.topk}, B={a.batch}, bf16")
    print(f"{'E':>9} {'table MB':>9} {'x L2':>6} {'score us':>9} {'select us':>10} {'TB/s':>7}")
    for E in E_GRID:
        be = TorchRouterBackend()
        be.setup(RouterConfig(batch_size=a.batch, num_experts=E, dim=a.dim,
                              topk=a.topk, seed=0, token_pool=4), dev)
        lg = be.score()
        torch.cuda.synchronize()
        sc = bench(be.score)
        se = bench(lambda: be.select(lg))   # topk_sorted=False, as in the sweep
        be.teardown()
        tb = E * a.dim * 2
        rows.append({"num_experts": E, "table_bytes": tb, "table_over_l2": round(tb / l2, 3),
                     "score_us": round(sc, 3), "select_us": round(se, 3),
                     "eff_read_tbps": round(tb / (sc / 1e6) / 1e12, 3)})
        print(f"{E:>9} {tb/2**20:>9.0f} {tb/l2:>6.2f} {sc:>9.2f} {se:>10.2f} "
              f"{tb/(sc/1e6)/1e12:>7.2f}")

    os.makedirs(os.path.join(a.out_dir, "figures"), exist_ok=True)
    csv_path = os.path.join(a.out_dir, "router_crossover_probe.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    Es = [r["num_experts"] for r in rows]
    sc = [r["score_us"] for r in rows]
    se = [r["select_us"] for r in rows]
    bw = [r["eff_read_tbps"] for r in rows]
    e_l2 = l2 / (a.dim * 2)          # E at which the table exactly fills L2

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    ax.axvspan(Es[0], e_l2, color="#2A8A4A", alpha=0.07)
    ax.axvline(20000, color="#C1332C", ls=":", lw=1.3)
    ax.annotate("torch.topk switches to\nmulti-block (E ≥ 20000)", xy=(20000, max(se) * 1.05),
                xytext=(8, 0), textcoords="offset points", fontsize=8.5, color="#C1332C")
    ax.axvline(e_l2, color="#2A8A4A", ls="--", lw=1.3)
    ax.annotate(f"table = L2 ({l2/2**20:.0f} MiB)\nE ≈ {e_l2:,.0f}", xy=(e_l2, min(sc) * 1.4),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color="#2A8A4A")
    ax.plot(Es, sc, "o-", color="#1F77B4", ms=5, label="score  (reads E×D weights)")
    ax.plot(Es, se, "s-", color="#C1332C", ms=5, label="select  (reads E logits, sorted=False)")
    # crossover
    for i in range(1, len(Es)):
        if sc[i] > se[i] and sc[i - 1] <= se[i - 1]:
            ax.plot(Es[i], sc[i], "o", ms=13, mfc="none", mec="#131C25", mew=1.6)
            ax.annotate("crossover", xy=(Es[i], sc[i]), xytext=(-14, 20),
                        textcoords="offset points", fontsize=9,
                        arrowprops=dict(arrowstyle="->", lw=1))
            break
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("candidate experts, E"); ax.set_ylabel("GPU latency (µs)")
    ax.set_title("Three regimes, two different reasons", fontsize=11)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    ax.axvline(e_l2, color="#2A8A4A", ls="--", lw=1.3)
    ax.axhline(2.0, color="#8A6A1F", ls=":", lw=1.4)
    ax.annotate("HBM2e peak ≈ 2.0 TB/s", xy=(Es[-1], 2.0), xytext=(0, 6),
                textcoords="offset points", ha="right", fontsize=9, color="#8A6A1F")
    ax.plot(Es, bw, "o-", color="#1F77B4", ms=5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 3.3)
    ax.set_xlabel("candidate experts, E")
    ax.set_ylabel("effective table read rate (TB/s)")
    ax.set_title("Above L2, every step streams the whole table from HBM", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Router scoring overtakes selection where the table outgrows L2   "
                 f"(D={a.dim}, k={a.topk}, B={a.batch}, bf16, sorted=False, H100 PCIe)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "figures", f"fig6_router_crossover.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("wrote fig6_router_crossover.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
