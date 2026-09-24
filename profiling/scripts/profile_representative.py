#!/usr/bin/env python3
"""Profiler evidence for representative DSA and router configurations.

Profiling runs are kept separate from the timing runs: CUPTI adds per-kernel
overhead, so nothing produced here is used as a latency sample.  Output goes to
``<run_dir>/profiles/`` alongside a machine-readable ``profile_summary.csv``.

What is collected
  * per-kernel GPU durations and launch counts (torch.profiler / CUPTI), which
    show which kernel dominates a stage and how many kernels a stage launches;
  * a Chrome/Perfetto trace per case, in which the gaps between consecutive
    kernels on the execution stream are visible directly;
  * a gap analysis derived from the trace: total GPU kernel time vs the
    wall-clock span of the stage, which separates GPU work from host submission
    gaps without assuming every gap is GPU work.

Tool availability on this machine is recorded in ``profiler_availability.json``:
Nsight Systems (``nsys``) is not installable from the configured package
channels, and the Nsight Compute build available for CUDA 12.9 (2024.1.1)
segfaults against driver 610.43.02.  The CUDA-graph vs eager comparison in the
main sweep provides the launch-gap attribution that an nsys timeline would show.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.backends.base import DSAConfig, RouterConfig
from benchmarks.backends.dsa_torch import TorchDSABackend
from benchmarks.backends.router_torch import TorchRouterBackend

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tool_availability() -> dict:
    info = {}
    for tool in ("nsys", "ncu"):
        path = shutil.which(tool)
        entry = {"on_path": path}
        if path:
            try:
                r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
                entry["version"] = (r.stdout or r.stderr).strip().splitlines()[:2]
                entry["usable"] = r.returncode == 0
            except Exception as exc:
                entry["usable"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["usable"] = False
        info[tool] = entry
    info["torch_profiler"] = {"usable": True, "backend": "CUPTI via torch.profiler/kineto"}
    info["note"] = (
        "nsys is not available from the configured conda channels; the only "
        "Nsight Compute build compatible with CUDA 12.9 (2024.1.1) segfaults on "
        "driver 610.43.02. Kernel-level evidence below comes from CUPTI."
    )
    return info


def analyse_trace(trace_path: str) -> dict:
    """Kernel time vs wall span on the GPU stream, from the exported trace."""
    with open(trace_path) as fh:
        data = json.load(fh)
    events = [
        e for e in data.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not events:
        return {"kernels": 0}
    events.sort(key=lambda e: e["ts"])
    total_kernel_us = sum(e["dur"] for e in events)
    span_us = (events[-1]["ts"] + events[-1]["dur"]) - events[0]["ts"]
    gap_us = max(0.0, span_us - total_kernel_us)
    by_name: dict[str, list[float]] = defaultdict(list)
    for e in events:
        by_name[e["name"]].append(e["dur"])
    top = sorted(((sum(v), len(v), k) for k, v in by_name.items()), reverse=True)[:6]
    return {
        "kernels": len(events),
        "distinct_kernels": len(by_name),
        "gpu_kernel_time_us": round(total_kernel_us, 2),
        "gpu_span_us": round(span_us, 2),
        "inter_kernel_gap_us": round(gap_us, 2),
        "gap_fraction_of_span": round(gap_us / span_us, 4) if span_us > 0 else None,
        "top_kernels": [
            {"name": k[:110], "calls": n, "total_us": round(t, 2)} for t, n, k in top
        ],
    }


def profile_case(name: str, fn, out_dir: str, iters: int = 20) -> dict:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(10):          # warm up outside the profiled region
        fn()
    torch.cuda.synchronize()

    trace_path = os.path.join(out_dir, f"{name}.trace.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)

    rows = []
    for evt in prof.key_averages():
        dev = getattr(evt, "device_time_total", 0) or 0
        if dev <= 0:
            continue
        rows.append({
            "case": name, "kernel": evt.key[:160], "calls": evt.count,
            "gpu_total_us": round(dev, 2), "gpu_avg_us": round(dev / max(evt.count, 1), 3),
        })
    rows.sort(key=lambda r: -r["gpu_total_us"])

    analysis = analyse_trace(trace_path)
    analysis["iters_profiled"] = iters
    analysis["per_iter_gpu_kernel_time_us"] = (
        round(analysis.get("gpu_kernel_time_us", 0) / iters, 3) if analysis.get("kernels") else None
    )
    return {"case": name, "trace": os.path.basename(trace_path),
            "kernel_rows": rows[:12], "trace_analysis": analysis}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="results/<run_id> to write profiles into")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--context-len", type=int, default=32768)
    ap.add_argument("--num-experts", type=int, default=1048576)
    a = ap.parse_args()

    out_dir = os.path.join(a.run_dir, "profiles")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(a.device)

    with open(os.path.join(out_dir, "profiler_availability.json"), "w") as fh:
        json.dump(tool_availability(), fh, indent=2)

    cases = []

    # --- DSA, both backends, score / select / combined -------------------
    backends = [("torch_bf16", TorchDSABackend())]
    try:
        from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

        if DeepGEMMDSABackend.availability()[0]:
            backends.append(("deepgemm", DeepGEMMDSABackend()))
    except Exception as exc:
        print(f"deepgemm backend unavailable: {exc}")

    for tag, be in backends:
        cfg = DSAConfig(batch_size=1, context_len=a.context_len, heads=64, dim=128,
                        topk=2048, seed=0, query_pool=8)
        be.setup(cfg, device)
        scores = be.score()
        torch.cuda.synchronize()
        for stage, fn in (("score", be.score),
                          ("select", lambda: be.select(scores)),
                          ("combined", be.combined)):
            cases.append(profile_case(f"dsa_{tag}_N{a.context_len}_B1_{stage}", fn, out_dir, a.iters))
            print(f"  profiled dsa_{tag}_{stage}")
        be.teardown()

    # --- Router at the largest expert count ------------------------------
    rb = TorchRouterBackend()
    rcfg = RouterConfig(batch_size=1, num_experts=a.num_experts, dim=4096, topk=8,
                        seed=0, token_pool=8)
    rb.setup(rcfg, device)
    logits = rb.score()
    torch.cuda.synchronize()
    for stage, fn in (("score", rb.score),
                      ("select", lambda: rb.select(logits)),
                      ("combined", rb.combined)):
        cases.append(profile_case(f"router_E{a.num_experts}_B1_{stage}", fn, out_dir, a.iters))
        print(f"  profiled router_{stage}")
    rb.teardown()

    with open(os.path.join(out_dir, "profile_summary.json"), "w") as fh:
        json.dump(cases, fh, indent=2)

    csv_path = os.path.join(out_dir, "profile_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "kernels_per_iter", "distinct_kernels", "gpu_kernel_time_us_per_iter",
                    "inter_kernel_gap_us_total", "gap_fraction_of_span", "top_kernel", "top_kernel_us_per_iter"])
        for c in cases:
            an = c["trace_analysis"]
            top = an.get("top_kernels") or [{}]
            w.writerow([
                c["case"],
                round(an.get("kernels", 0) / an.get("iters_profiled", 1), 2),
                an.get("distinct_kernels"),
                an.get("per_iter_gpu_kernel_time_us"),
                an.get("inter_kernel_gap_us"),
                an.get("gap_fraction_of_span"),
                top[0].get("name", ""),
                round(top[0].get("total_us", 0) / an.get("iters_profiled", 1), 3) if top[0] else "",
            ])
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
