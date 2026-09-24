#!/usr/bin/env python3
"""DSA block benchmark: indexer search inside the sparse-attention layer.

Answers "how much of a DSA attention layer is the top-k search?" by running the
real pipeline on an H100:

    1. indexer scoring    DeepGEMM fp8_paged_mqa_logits over N cached tokens
    2. top-k selection    torch.topk, k = 2048
    3. index mapping      logical token ids -> FlashMLA's paged addressing
    4. sparse attention   FlashMLA fp8 sparse MLA decode over the selected 2048

Stages 1-3 are what a CAM would absorb; stage 4 is the attention that remains.
Each stage is timed on its own, and the whole block is timed as one region.
The block total is its own measurement -- stages are not summed to produce it.

**Scope.** Block inputs are prepared projections: the indexer query/key
projections, the q-LoRA path, RoPE and the output projection are not included,
and neither is the MLP. A percentage from this benchmark is a fraction of the
measured DSA attention block, NOT a fraction of full-model inference.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import env_info, results
from benchmarks.backends.base import DSAConfig, INPUT_DISTRIBUTION, Unsupported
from benchmarks.timing import (
    GraphUnsupported, MemoryProbe, autoscale_group_size, capture_graph, time_gpu,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--context-lens", type=int, nargs="+", default=None)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--heads", type=int, default=None, help="indexer heads")
    p.add_argument("--dim", type=int, default=None, help="indexer head dim")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--mla-heads", type=int, default=None,
                   help="MLA query heads (DeepSeek-V3.2: 128)")
    p.add_argument("--block-kv", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--query-pool", type=int, default=None)
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--graph-modes", nargs="+", choices=["eager", "cudagraph"], default=None,
                   help="eager launches each kernel from the host; cudagraph replays a "
                        "captured graph, removing per-launch host cost")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--note", default=None)

    a = p.parse_args(argv)
    defaults = {
        "context_lens": [8192, 16384, 32768, 65536, 131072], "batch_sizes": [1, 8],
        "heads": 64, "dim": 128, "topk": 2048, "mla_heads": 128, "block_kv": 64,
        "seed": 0, "warmup": 20, "iters": 100, "query_pool": 256,
        "memory_budget_gib": 60.0, "graph_modes": ["eager", "cudagraph"],
        "out_dir": os.path.join(REPO_ROOT, "results"),
    }
    if a.config:
        import yaml
        with open(a.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        defaults.update(cfg.get("common", {}))
        defaults.update(cfg.get("dsa_block", {}))
    for k, v in defaults.items():
        if getattr(a, k, None) in (None, []):
            setattr(a, k, v)
    return a


def run_block(idx, mla, cfg: DSAConfig, args, base: dict) -> tuple[list, list]:
    raw, summ = [], []
    dev = torch.device(args.device)

    # ---- setup, timed separately, never folded into steady state ----------
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    idx.setup(cfg, dev)
    mla.setup(cfg.batch_size, cfg.context_len, cfg.topk, cfg.seed, dev)
    e1.record()
    torch.cuda.synchronize()
    setup_us = e0.elapsed_time(e1) * 1e3

    # Prime each stage so JIT/autotune and first-call metadata are done.
    scores = idx.score()
    ids = idx.select(scores)
    mla.set_selected_ids(ids)
    mapped = mla.map_indices()
    mla.attend(mapped)
    torch.cuda.synchronize()

    def stage_score():
        idx.score()

    def stage_select():
        idx.select(scores)

    def stage_map():
        mla.map_indices()

    def stage_attend():
        mla.attend(mapped)

    def whole_block():
        s = idx.score()
        sel = idx.select(s)
        mla.set_selected_ids(sel)
        mla.attend(mla.map_indices())

    stages = {
        "indexer_score": stage_score,
        "indexer_select": stage_select,
        "index_map": stage_map,
        "sparse_mla_attend": stage_attend,
        "dsa_block": whole_block,
    }

    counter = {"i": 0}

    def next_query():
        counter["i"] += 1
        idx.stage_query(counter["i"])

    for graph_mode in args.graph_modes:
        for stage, fn in stages.items():
            gbase = {**base, "graph_mode": graph_mode}
            try:
                run = capture_graph(fn)[0].replay if graph_mode == "cudagraph" else fn
                group = autoscale_group_size(run)
                with MemoryProbe(dev.index or 0) as mem:
                    t = time_gpu(run, warmup=args.warmup, iters=args.iters,
                                 group_size=group, reset_fn=next_query)
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
                setup_us=setup_us if stage == "dsa_block" else None,
            ))
        if graph_mode == "cudagraph":
            torch.cuda.synchronize()

    idx.teardown()
    mla.teardown()
    return raw, summ


def main(argv: Optional[list[str]] = None) -> int:
    from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend
    from benchmarks.backends.mla_flashmla import FlashMLASparseBackend

    args = parse_args(argv)
    run_id = args.run_id or results.new_run_id("dsablock")
    combos = [(b, n) for b in args.batch_sizes for n in args.context_lens]

    if args.dry_run:
        print(f"dry-run: {len(combos)} configurations, budget {args.memory_budget_gib} GiB")
        ok = True
        for b, n in combos:
            cfg = DSAConfig(batch_size=b, context_len=n, heads=args.heads, dim=args.dim,
                            topk=args.topk, seed=args.seed, block_kv=args.block_kv,
                            query_pool=args.query_pool)
            mla = FlashMLASparseBackend(heads_q=args.mla_heads, page_block=args.block_kv)
            est = (DeepGEMMDSABackend.estimate_bytes(cfg)
                   + mla.estimate_bytes(b, n, args.topk)) / 2**30
            status = "ok"
            for be in (DeepGEMMDSABackend, FlashMLASparseBackend):
                avail, why = be.availability()
                if not avail:
                    status, ok = f"unavailable: {why}", False
            if n < args.topk:
                status, ok = f"INVALID: N={n} < k={args.topk}", False
            elif est > args.memory_budget_gib:
                status, ok = f"OVER BUDGET ({est:.2f} GiB)", False
            print(f"  B={b} N={n:>7} MLA_h={args.mla_heads}  est={est:7.3f} GiB  {status}")
        free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
        print(f"device free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB")
        return 0 if ok else 1

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    writer = results.RunWriter(root=args.out_dir, run_id=run_id).open()
    writer.write_metadata({
        "run_id": run_id, "workload": "dsa_block", "command": " ".join(sys.argv),
        "resolved_config": {k: v for k, v in vars(args).items()},
        "input_distribution": INPUT_DISTRIBUTION,
        "scope_note": (
            "Indexer scoring + top-k + index mapping + FlashMLA fp8 sparse MLA decode. "
            "Block inputs are PREPARED PROJECTIONS: q/k projections, q-LoRA, RoPE, the "
            "output projection and the MLP are NOT included. Percentages are fractions of "
            "this measured attention block, not of full-model inference."
        ),
        "environment": env_info.collect(dev.index or 0, extra_repos={
            "DeepGEMM": os.path.join(REPO_ROOT, "third_party", "DeepGEMM"),
            "FlashMLA": os.path.join(REPO_ROOT, "third_party", "FlashMLA"),
            "DeepSeek-V3.2-Exp": os.path.join(REPO_ROOT, "third_party", "DeepSeek-V3.2-Exp"),
        }),
        "note": args.note,
    })
    print(f"run_id={run_id}  dir={writer.dir}")

    for b, n in combos:
        cfg = DSAConfig(batch_size=b, context_len=n, heads=args.heads, dim=args.dim,
                        topk=args.topk, seed=args.seed, block_kv=args.block_kv,
                        query_pool=args.query_pool)
        idx = DeepGEMMDSABackend()
        mla = FlashMLASparseBackend(heads_q=args.mla_heads, page_block=args.block_kv)
        base = {
            "run_id": run_id, "workload": "dsa_block", "mode": "block_fixed_context",
            "backend": f"{idx.name}+{mla.name}", "backend_commit": f"{idx.commit}|{mla.commit}",
            "dtype": "fp8_e4m3", "batch_size": b, "candidate_count": n,
            "dimension": args.dim, "indexer_heads": args.heads, "k": args.topk,
            "seed": args.seed, "graph_mode": "eager", "decode_step": None,
            "topk_sorted": False, "block_kv": args.block_kv,
            "context_len_actual": n,
            "config_id": f"dsa_block|B{b}|N{n}",
        }
        est = (DeepGEMMDSABackend.estimate_bytes(cfg)
               + mla.estimate_bytes(b, n, args.topk)) / 2**30
        bad = [why for be in (DeepGEMMDSABackend, FlashMLASparseBackend)
               for avail, why in [be.availability()] if not avail]
        if bad:
            writer.write_summary({**base, "stage": "all", "status": "unavailable",
                                  "error": "; ".join(bad)})
            print(f"  SKIP {base['config_id']}: {bad}")
            continue
        if est > args.memory_budget_gib:
            writer.write_summary({**base, "stage": "all", "status": "over_budget",
                                  "error": f"{est:.2f} GiB > {args.memory_budget_gib}"})
            print(f"  SKIP {base['config_id']}: over budget {est:.2f} GiB")
            continue

        print(f"  RUN  {base['config_id']}  est={est:.3f} GiB", flush=True)
        try:
            raw, summ = run_block(idx, mla, cfg, args, base)
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
        totals = {r["graph_mode"]: r["median_us"] for r in summ
                  if r.get("stage") == "dsa_block" and r.get("status") == "ok"}
        for row in summ:
            writer.write_summary(row)
            if row.get("status") == "ok":
                tot = totals.get(row["graph_mode"])
                frac = f"{100*row['median_us']/tot:5.1f}% of block" if tot else ""
                print(f"  [{row['graph_mode']:9s}] {row['stage']:20s} median={row['median_us']:9.2f} us  "
                      f"p95={row['p95_us']:9.2f}  {frac}")
            else:
                print(f"       {row['stage']:20s} {row['status']}: {row.get('error')}")

    writer.close()
    print(f"done: {writer.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
