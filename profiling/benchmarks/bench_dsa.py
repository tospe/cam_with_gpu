#!/usr/bin/env python3
"""DSA (DeepSeek Sparse Attention) indexer search benchmark on a single GPU.

Modes
  fixed      Fixed-context operator benchmark.  A cache of N candidates is
             prepared once, then one decode query per sequence is scored against
             it and the top-k is selected.  Isolates scaling with N.
  incremental
             Decoding-block benchmark.  Starts from N cached tokens and appends
             one new indexer key per sequence per step for ``--steps`` steps,
             measuring cache update + search together.  Context length grows, so
             the loop is reported as a loop, never as repeated samples at one N.

Scoring semantics, layouts and quantisation follow the selected backend; see
``benchmarks/backends/`` for the per-backend documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import env_info, results
from benchmarks.backends.base import DSAConfig, INPUT_DISTRIBUTION, Unsupported
from benchmarks.backends.dsa_torch import TorchDSABackend
from benchmarks.timing import (
    GraphUnsupported, MemoryProbe, autoscale_group_size, capture_graph, time_gpu,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_backend(name: str, args) -> Any:
    if name == "torch_bf16":
        return TorchDSABackend(score_chunk=args.score_chunk, topk_sorted=args.topk_sorted)
    if name == "deepgemm":
        from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

        return DeepGEMMDSABackend(topk_sorted=args.topk_sorted)
    raise SystemExit(f"unknown backend: {name}")


BACKEND_CHOICES = ["torch_bf16", "deepgemm"]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config supplying defaults for any option below")
    p.add_argument("--backend", action="append", choices=BACKEND_CHOICES,
                   help="repeatable; default: torch_bf16")
    p.add_argument("--mode", action="append", choices=["fixed", "incremental"],
                   help="repeatable; default: fixed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--context-lens", type=int, nargs="+", default=None,
                   help="N, initial cached tokens per sequence")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                   help="B, independent sequences (each with its own cache)")
    p.add_argument("--heads", type=int, default=None, help="indexer heads H")
    p.add_argument("--dim", type=int, default=None, help="indexer head dim D")
    p.add_argument("--topk", type=int, default=None, help="k")
    p.add_argument("--steps", type=int, default=None, help="incremental decode steps")
    p.add_argument("--block-kv", type=int, default=None, help="paged KV block size")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--iters", type=int, default=None, help="measured repetitions")
    p.add_argument("--incremental-trials", type=int, default=None,
                   help="repetitions of the whole incremental loop")
    p.add_argument("--query-pool", type=int, default=None,
                   help="distinct queries cycled across trials")
    p.add_argument("--score-chunk", type=int, default=None,
                   help="torch backend: candidate-axis chunk (0 = no chunking)")
    p.add_argument("--topk-sorted", action=argparse.BooleanOptionalAction, default=None,
                   help="return the k winners in sorted order; adds a separate sort "
                        "kernel on CUDA. Default off, and recorded per row.")
    p.add_argument("--measure-host", action="store_true",
                   help="also record host-observed latency (adds a sync per trial)")
    p.add_argument("--graph-modes", nargs="+", choices=["eager", "cudagraph"], default=None,
                   help="eager launches each kernel from the host; cudagraph replays a "
                        "captured graph, removing per-launch host cost. Both backends "
                        "are measured in the same modes so they stay comparable.")
    p.add_argument("--memory-budget-gib", type=float, default=None,
                   help="skip configurations whose estimate exceeds this")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="validate configurations and estimate memory; allocate nothing")
    p.add_argument("--note", default=None, help="free-text note stored in metadata")

    args = p.parse_args(argv)
    defaults = {
        "backend": ["torch_bf16"], "mode": ["fixed"],
        "context_lens": [8192, 32768], "batch_sizes": [1],
        "heads": 64, "dim": 128, "topk": 2048, "steps": 16, "block_kv": 64,
        "seed": 0, "warmup": 20, "iters": 100, "incremental_trials": 20,
        "query_pool": 256, "score_chunk": 32768, "graph_modes": ["eager"], "topk_sorted": False,
        "memory_budget_gib": 60.0, "out_dir": os.path.join(REPO_ROOT, "results"),
    }
    if args.config:
        import yaml

        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        defaults.update(cfg.get("common", {}))
        defaults.update(cfg.get("dsa", {}))
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, []):
            setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def run_fixed(backend, cfg: DSAConfig, args, base: dict) -> tuple[list, list]:
    """Score / select / combined at a fixed context length."""
    raw, summ = [], []
    device = torch.device(args.device)

    setup_evt = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    with MemoryProbe(device.index or 0) as mem:
        setup_evt[0].record()
        backend.setup(cfg, device)
        setup_evt[1].record()
        torch.cuda.synchronize()
    setup_us = setup_evt[0].elapsed_time(setup_evt[1]) * 1e3

    def do_score():
        backend.score()

    # Score once so the first JIT compile and any autotuning happen before the
    # timed region, and so `select` has a real score tensor to work on.
    first = backend.score()
    torch.cuda.synchronize()

    # The isolated `select` stage needs varying input too: top-k cost can depend
    # on the score distribution, so a pool of score vectors from distinct
    # queries is staged into one fixed buffer (constant pointer, graph-safe)
    # rather than re-selecting over a single vector every trial.
    n_score_pool = min(8, max(2, args.iters))
    score_pool = torch.empty((n_score_pool, *first.shape), device=first.device, dtype=first.dtype)
    for i in range(n_score_pool):
        backend.stage_query(i)
        score_pool[i].copy_(backend.score())
    backend.stage_query(0)
    select_input = torch.empty_like(first)
    select_input.copy_(score_pool[0])
    torch.cuda.synchronize()

    stages = {
        "score": do_score,
        "select": lambda: backend.select(select_input),
        "combined": backend.combined,
    }

    trial_counter = {"i": 0}

    def next_query():
        """Advance to the next trial's inputs -- always outside the timed region."""
        i = trial_counter["i"] = trial_counter["i"] + 1
        backend.stage_query(i)
        select_input.copy_(score_pool[i % n_score_pool])

    for graph_mode in args.graph_modes:
        for stage, fn in stages.items():
            gbase = {**base, "graph_mode": graph_mode}
            try:
                if graph_mode == "cudagraph":
                    # Queries still vary per trial: `stage_query` writes into the
                    # same staging buffers the captured graph reads, so replay
                    # sees the new values without re-capture.
                    graph, _ = capture_graph(fn)
                    run = graph.replay
                else:
                    run = fn
                group = autoscale_group_size(run)
                with MemoryProbe(device.index or 0) as smem:
                    t = time_gpu(
                        run,
                        warmup=args.warmup,
                        iters=args.iters,
                        group_size=group,
                        # A fresh query per trial; staged outside the timed region
                        # so the measured pointers stay constant.
                        reset_fn=next_query,
                        measure_host=args.measure_host,
                    )
            except GraphUnsupported as exc:
                summ.append({**gbase, "stage": stage, "status": "graph_unsupported",
                             "error": str(exc)[:500]})
                continue
            except Exception as exc:  # record, never hide
                summ.append({**gbase, "stage": stage, "status": "error",
                             "error": f"{type(exc).__name__}: {exc}"[:500]})
                continue
            raw.extend(results.raw_rows(t, base=gbase, stage=stage))
            summ.append(
                results.summarize(
                    t, base=gbase, stage=stage, batch_size=cfg.batch_size,
                    peak_allocated=smem.peak_allocated, peak_reserved=smem.peak_reserved,
                    setup_us=setup_us if stage == "combined" else None,
                )
            )
        if graph_mode == "cudagraph":
            torch.cuda.synchronize()
    backend.teardown()
    return raw, summ


def run_incremental(backend, cfg: DSAConfig, args, base: dict) -> tuple[list, list]:
    """A growing-context decode loop, timed as a whole loop."""
    raw, summ = [], []
    device = torch.device(args.device)

    setup_evt = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    setup_evt[0].record()
    backend.setup(cfg, device)
    setup_evt[1].record()
    torch.cuda.synchronize()
    setup_us = setup_evt[0].elapsed_time(setup_evt[1]) * 1e3

    def loop():
        backend.decode_steps(cfg.steps)

    # reset() restores cache contents, valid lengths and replay position so every
    # trial -- including the ones after warmup -- starts from the same state.
    try:
        with MemoryProbe(device.index or 0) as mem:
            t = time_gpu(
                loop,
                warmup=max(2, args.warmup // 10),
                iters=args.incremental_trials,
                group_size=1,
                reset_fn=backend.reset,
                measure_host=args.measure_host,
            )
    except Exception as exc:
        summ.append({**base, "stage": "decode_loop", "status": "error",
                     "error": f"{type(exc).__name__}: {exc}"})
        backend.teardown()
        return raw, summ

    row_base = dict(base)
    row_base["context_len_actual"] = f"{cfg.context_len + 1}..{cfg.context_len + cfg.steps}"
    raw.extend(results.raw_rows(t, base=row_base, stage="decode_loop"))
    s = results.summarize(
        t, base=row_base, stage="decode_loop", batch_size=cfg.batch_size,
        peak_allocated=mem.peak_allocated, peak_reserved=mem.peak_reserved, setup_us=setup_us,
    )
    summ.append(s)

    # Per-step average is derived, and labelled as such.
    per_step = dict(s)
    per_step.update(
        stage="decode_loop_per_step_mean",
        median_us=s["median_us"] / cfg.steps,
        mean_us=s["mean_us"] / cfg.steps,
        min_us=s["min_us"] / cfg.steps,
        p95_us=s["p95_us"] / cfg.steps,
        p99_us=s["p99_us"] / cfg.steps,
        stdev_us=s["stdev_us"] / cfg.steps,
        iqr_us=s["iqr_us"] / cfg.steps,
        host_median_us=(s["host_median_us"] / cfg.steps) if s["host_median_us"] else None,
        throughput_qps=cfg.batch_size * cfg.steps / (s["median_us"] / 1e6),
        timing_method=f"derived_from_{s['timing_method']}_div_{cfg.steps}_steps",
        setup_us=None,
    )
    summ.append(per_step)

    backend.teardown()
    return raw, summ


# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or results.new_run_id("dsa")

    configs = []
    for backend_name in args.backend:
        for mode in args.mode:
            for b in args.batch_sizes:
                for n in args.context_lens:
                    configs.append((backend_name, mode, b, n))

    # --- dry run: validate + estimate, allocate nothing --------------------
    if args.dry_run:
        print(f"dry-run: {len(configs)} configurations, budget {args.memory_budget_gib} GiB")
        ok = True
        for backend_name, mode, b, n in configs:
            cfg = DSAConfig(
                batch_size=b, context_len=n, heads=args.heads, dim=args.dim,
                topk=args.topk, steps=args.steps if mode == "incremental" else 0,
                seed=args.seed, block_kv=args.block_kv, query_pool=args.query_pool,
            )
            backend = build_backend(backend_name, args)
            avail, why = type(backend).availability()
            est = type(backend).estimate_bytes(cfg) / 2**30
            status = "ok"
            if not avail:
                status, ok = f"unavailable: {why}", False
            elif args.topk > n:
                status = f"note: k={args.topk} > N={n}, k clamped to N"
            elif est > args.memory_budget_gib:
                status, ok = f"OVER BUDGET ({est:.2f} GiB)", False
            print(f"  {backend_name:10s} {mode:11s} {cfg.describe():48s} est={est:7.3f} GiB  {status}")
        free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
        print(f"device free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB")
        return 0 if ok else 1

    # --- real run ----------------------------------------------------------
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    writer = results.RunWriter(root=args.out_dir, run_id=run_id).open()
    meta = {
        "run_id": run_id,
        "workload": "dsa",
        "command": " ".join(sys.argv),
        "resolved_config": {k: v for k, v in vars(args).items()},
        "input_distribution": INPUT_DISTRIBUTION,
        "environment": env_info.collect(
            device.index or 0,
            extra_repos={
                "DeepGEMM": os.path.join(REPO_ROOT, "third_party", "DeepGEMM"),
                "DeepSeek-V3.2-Exp": os.path.join(REPO_ROOT, "third_party", "DeepSeek-V3.2-Exp"),
            },
        ),
        "note": args.note,
    }
    writer.write_metadata(meta)
    print(f"run_id={run_id}  dir={writer.dir}")

    for backend_name, mode, b, n in configs:
        cfg = DSAConfig(
            batch_size=b, context_len=n, heads=args.heads, dim=args.dim,
            topk=args.topk, steps=args.steps if mode == "incremental" else 0,
            seed=args.seed, block_kv=args.block_kv, query_pool=args.query_pool,
        )
        backend = build_backend(backend_name, args)
        base = {
            "run_id": run_id, "workload": "dsa", "mode": mode,
            "backend": backend.name, "backend_commit": backend.commit,
            "dtype": backend.dtype_tag, "batch_size": b, "candidate_count": n,
            "dimension": args.dim, "indexer_heads": args.heads, "k": min(args.topk, n),
            "seed": args.seed, "graph_mode": "eager", "steps": cfg.steps or None,
            "topk_sorted": bool(args.topk_sorted),
            "block_kv": args.block_kv if backend_name == "deepgemm" else None,
            "config_id": f"{backend.name}|{mode}|B{b}|N{n}",
            "decode_step": None, "context_len_actual": n if mode == "fixed" else None,
        }

        avail, why = type(backend).availability()
        est_gib = type(backend).estimate_bytes(cfg) / 2**30
        if not avail:
            writer.write_summary({**base, "stage": "all", "status": "unavailable", "error": why})
            print(f"  SKIP {base['config_id']}: {why}")
            continue
        if est_gib > args.memory_budget_gib:
            msg = f"estimated {est_gib:.2f} GiB > budget {args.memory_budget_gib} GiB"
            writer.write_summary({**base, "stage": "all", "status": "over_budget", "error": msg})
            print(f"  SKIP {base['config_id']}: {msg}")
            continue

        print(f"  RUN  {base['config_id']}  est={est_gib:.3f} GiB", flush=True)
        try:
            runner = run_fixed if mode == "fixed" else run_incremental
            raw, summ = runner(backend, cfg, args, base)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            writer.write_summary({**base, "stage": "all", "status": "oom", "error": str(exc)[:500]})
            print(f"  OOM  {base['config_id']}")
            continue
        except Unsupported as exc:
            writer.write_summary({**base, "stage": "all", "status": "unsupported", "error": str(exc)})
            print(f"  SKIP {base['config_id']}: {exc}")
            continue
        except Exception as exc:
            traceback.print_exc()
            writer.write_summary({**base, "stage": "all", "status": "error",
                                  "error": f"{type(exc).__name__}: {exc}"[:500]})
            continue

        writer.write_raw(raw)
        for row in summ:
            writer.write_summary(row)
            if row.get("status") == "ok":
                print(f"       {row['stage']:28s} median={row['median_us']:9.2f} us  "
                      f"p95={row['p95_us']:9.2f} us  cv={row['cv']:.3f}")
            else:
                print(f"       {row['stage']:28s} {row['status']}: {row.get('error')}")

    writer.close()
    print(f"done: {writer.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
