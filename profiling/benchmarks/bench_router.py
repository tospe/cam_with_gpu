#!/usr/bin/env python3
"""Flat MoE-router search benchmark: scaling with the number of candidate experts.

Measures only routing: ``logits = hidden @ W.T`` followed by top-k over E
candidates.  It does not measure dispatch, expert-network execution, cross-GPU
communication, or model accuracy at any expert count.

At E = 2^20 and D = 4096 the BF16 router table alone is 8 GiB, so every
configuration's live allocation is estimated against ``--memory-budget-gib``
before it runs.  Configurations over budget, unsupported, or hitting OOM are
recorded with a non-ok status; E, D, dtype and batch size are never silently
reduced.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import env_info, results
from benchmarks.backends.base import INPUT_DISTRIBUTION, RouterConfig, Unsupported
from benchmarks.backends.router_torch import TorchRouterBackend
from benchmarks.timing import (
    GraphUnsupported, MemoryProbe, autoscale_group_size, capture_graph, time_gpu,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=list(_DTYPES), default=None)
    p.add_argument("--num-experts", type=int, nargs="+", default=None, help="E values to sweep")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                   help="B: tokens routed concurrently in one decode step")
    p.add_argument("--dim", type=int, default=None,
                   help="D; 4096 is a labelled synthetic default, not a model value")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--token-pool", type=int, default=None)
    p.add_argument("--topk-sorted", action=argparse.BooleanOptionalAction, default=None,
                   help="return the k winners in sorted order; adds a separate sort "
                        "kernel on CUDA. Default off, and recorded per row.")
    p.add_argument("--measure-host", action="store_true")
    p.add_argument("--graph-modes", nargs="+", choices=["eager", "cudagraph"], default=None,
                   help="eager launches each kernel from the host; cudagraph replays a "
                        "captured graph, removing per-launch host cost")
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--note", default=None)

    args = p.parse_args(argv)
    defaults = {
        "dtype": "bf16", "num_experts": [16, 256, 4096], "batch_sizes": [1],
        "dim": 4096, "topk": 8, "seed": 0, "warmup": 20, "iters": 100,
        "token_pool": 256, "topk_sorted": False, "memory_budget_gib": 60.0,
        "graph_modes": ["eager"],
        "out_dir": os.path.join(REPO_ROOT, "results"),
    }
    if args.config:
        import yaml

        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        defaults.update(cfg.get("common", {}))
        defaults.update(cfg.get("router", {}))
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, []):
            setattr(args, key, value)
    return args


def run_one(backend, cfg: RouterConfig, args, base: dict) -> tuple[list, list]:
    raw, summ = [], []
    device = torch.device(args.device)

    setup_evt = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    setup_evt[0].record()
    backend.setup(cfg, device)
    setup_evt[1].record()
    torch.cuda.synchronize()
    setup_us = setup_evt[0].elapsed_time(setup_evt[1]) * 1e3

    first = backend.score()
    torch.cuda.synchronize()

    # Vary the isolated select stage's input as well: a pool of logit vectors
    # from distinct tokens, staged into one fixed buffer (graph-safe pointer).
    n_pool = min(8, max(2, args.iters))
    logit_pool = torch.empty((n_pool, *first.shape), device=first.device, dtype=first.dtype)
    for i in range(n_pool):
        backend.stage_query(i)
        logit_pool[i].copy_(backend.score())
    backend.stage_query(0)
    select_input = torch.empty_like(first)
    select_input.copy_(logit_pool[0])
    torch.cuda.synchronize()

    stages = {
        "score": backend.score,
        "select": lambda: backend.select(select_input),
        "combined": backend.combined,
    }
    counter = {"i": 0}

    def next_token():
        i = counter["i"] = counter["i"] + 1
        backend.stage_query(i)
        select_input.copy_(logit_pool[i % n_pool])

    for graph_mode in args.graph_modes:
        for stage, fn in stages.items():
            gbase = {**base, "graph_mode": graph_mode}
            try:
                run = capture_graph(fn)[0].replay if graph_mode == "cudagraph" else fn
                group = autoscale_group_size(run)
                with MemoryProbe(device.index or 0) as mem:
                    t = time_gpu(run, warmup=args.warmup, iters=args.iters, group_size=group,
                                 reset_fn=next_token, measure_host=args.measure_host)
            except GraphUnsupported as exc:
                summ.append({**gbase, "stage": stage, "status": "graph_unsupported",
                             "error": str(exc)[:500]})
                continue
            except Exception as exc:
                summ.append({**gbase, "stage": stage, "status": "error",
                             "error": f"{type(exc).__name__}: {exc}"[:500]})
                continue
            raw.extend(results.raw_rows(t, base=gbase, stage=stage))
            summ.append(results.summarize(
                t, base=gbase, stage=stage, batch_size=cfg.batch_size,
                peak_allocated=mem.peak_allocated, peak_reserved=mem.peak_reserved,
                setup_us=setup_us if stage == "combined" else None,
            ))
        if graph_mode == "cudagraph":
            torch.cuda.synchronize()
    backend.teardown()
    return raw, summ


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or results.new_run_id("router")
    dtype = _DTYPES[args.dtype]

    combos = [(b, e) for b in args.batch_sizes for e in args.num_experts]

    if args.dry_run:
        print(f"dry-run: {len(combos)} configurations, budget {args.memory_budget_gib} GiB")
        ok = True
        backend = TorchRouterBackend(dtype=dtype, topk_sorted=args.topk_sorted)
        for b, e in combos:
            cfg = RouterConfig(batch_size=b, num_experts=e, dim=args.dim,
                               topk=args.topk, seed=args.seed, token_pool=args.token_pool)
            est = backend.estimate_bytes(cfg) / 2**30
            status = "ok"
            if args.topk > e:
                status, ok = f"INVALID: k={args.topk} > E={e}", False
            elif est > args.memory_budget_gib:
                status, ok = f"OVER BUDGET ({est:.2f} GiB)", False
            print(f"  {cfg.describe():44s} est={est:8.3f} GiB  {status}")
        free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
        print(f"device free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB")
        return 0 if ok else 1

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    writer = results.RunWriter(root=args.out_dir, run_id=run_id).open()
    writer.write_metadata({
        "run_id": run_id, "workload": "router",
        "command": " ".join(sys.argv),
        "resolved_config": {k: v for k, v in vars(args).items()},
        "input_distribution": INPUT_DISTRIBUTION,
        "router_note": (
            "Flat router: logits = hidden @ W.T then top-k. Not a model-specific "
            "MoE router (no activation, correction bias, expert-group selection, "
            "or routing-weight normalisation). D is a synthetic default unless "
            "taken from a specific model."
        ),
        "environment": env_info.collect(device.index or 0),
        "note": args.note,
    })
    print(f"run_id={run_id}  dir={writer.dir}")

    for b, e in combos:
        cfg = RouterConfig(batch_size=b, num_experts=e, dim=args.dim, topk=args.topk,
                           seed=args.seed, token_pool=args.token_pool)
        backend = TorchRouterBackend(dtype=dtype, topk_sorted=args.topk_sorted)
        base = {
            "run_id": run_id, "workload": "router", "mode": "flat_router_decode",
            "backend": backend.name, "backend_commit": None, "dtype": backend.dtype_tag,
            "batch_size": b, "candidate_count": e, "dimension": args.dim,
            "indexer_heads": None, "k": args.topk, "seed": args.seed,
            "graph_mode": "eager", "decode_step": None,
            "topk_sorted": bool(args.topk_sorted),
            "config_id": f"{backend.name}|B{b}|E{e}|D{args.dim}",
        }
        est_gib = backend.estimate_bytes(cfg) / 2**30
        if args.topk > e:
            writer.write_summary({**base, "stage": "all", "status": "invalid",
                                  "error": f"k={args.topk} > E={e}"})
            print(f"  SKIP {base['config_id']}: k > E")
            continue
        if est_gib > args.memory_budget_gib:
            msg = f"estimated {est_gib:.2f} GiB > budget {args.memory_budget_gib} GiB"
            writer.write_summary({**base, "stage": "all", "status": "over_budget", "error": msg})
            print(f"  SKIP {base['config_id']}: {msg}")
            continue

        print(f"  RUN  {base['config_id']}  est={est_gib:.3f} GiB", flush=True)
        try:
            raw, summ = run_one(backend, cfg, args, base)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            writer.write_summary({**base, "stage": "all", "status": "oom", "error": str(exc)[:500]})
            print(f"  OOM  {base['config_id']}")
            continue
        except Unsupported as exc:
            writer.write_summary({**base, "stage": "all", "status": "unsupported", "error": str(exc)})
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
                print(f"       {row['stage']:10s} median={row['median_us']:8.2f} us  "
                      f"p95={row['p95_us']:8.2f} us  cv={row['cv']:.3f}")
            else:
                print(f"       {row['stage']:10s} {row['status']}: {row.get('error')}")

    writer.close()
    print(f"done: {writer.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
