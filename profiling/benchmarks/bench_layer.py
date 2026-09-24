#!/usr/bin/env python3
"""Full DeepSeek-V3.2 decoder layer at decode: where does the top-k search sit?

Runs one complete layer -- attention projections, indexer search, sparse MLA
attention, output projection, router, and the MoE (or dense) feed-forward --
at the model's published shapes, and times each stage plus the whole layer.

    search        = indexer projections + indexer scoring + top-k + index map
    attention     = q/kv projections, absorptions, sparse MLA, output projection
    feed-forward  = router + experts

Scope: ONE layer of 61, decode step, no embedding or LM head, no tensor or
expert parallelism, and the MoE experts run as per-(token, expert) GEMVs with
routing held fixed during timing rather than through a fused grouped-GEMM
kernel (see benchmarks/backends/decoder_layer.py). Percentages are fractions of
this measured layer, not of end-to-end model inference.
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
from benchmarks.backends.decoder_layer import DeepSeekV32Layer, LayerConfig
from benchmarks.timing import (
    GraphUnsupported, MemoryProbe, autoscale_group_size, capture_graph, time_gpu,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--context-lens", type=int, nargs="+", default=[8192, 32768, 131072])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--mlp", nargs="+", choices=["moe", "dense"], default=["moe", "dense"])
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--block-kv", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--graph-modes", nargs="+", choices=["eager", "cudagraph"],
                   default=["eager", "cudagraph"])
    p.add_argument("--memory-budget-gib", type=float, default=70.0)
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--note", default=None)
    return p.parse_args(argv)


def build_stages(layer, idx, mla, cfg):
    """Stage callables. Each re-runs its own dependencies' *outputs* from state."""
    layer.attn_proj_in()
    layer.indexer_proj()
    mla.append_latent(layer.kv_new)
    idx.append_key(layer.idx_k)
    idx.q.copy_(layer.idx_q_fp8)
    idx.w.copy_(layer.idx_w)
    scores = idx.score()
    ids = idx.select(scores)
    mla.set_selected_ids(ids)
    mapped = mla.map_indices()
    mla.q.copy_(layer.q_mla.expand_as(mla.q))
    attn = mla.attend(mapped)
    layer.attn_proj_out(attn.view(layer.batch, -1))
    layer.post_norm()
    is_moe = layer.cfg.mlp == "moe"
    if is_moe:
        layer.router_score()
        layer.router_topk()
    layer.mlp()
    torch.cuda.synchronize()

    def kv_append():
        """Write this step's MLA latent and indexer key into their caches."""
        mla.append_latent(layer.kv_new)
        idx.append_key(layer.idx_k)

    def whole_layer():
        layer.attn_proj_in()
        layer.indexer_proj()
        kv_append()
        idx.q.copy_(layer.idx_q_fp8)
        idx.w.copy_(layer.idx_w)
        sel = idx.select(idx.score())
        mla.set_selected_ids(sel)
        mla.q.copy_(layer.q_mla.expand_as(mla.q))
        a = mla.attend(mla.map_indices())
        layer.attn_proj_out(a.view(layer.batch, -1))
        layer.post_norm()
        if is_moe:
            layer.router_score()
            layer.router_topk()
        layer.mlp()

    stages = {
        "input_norm": layer.input_norm,
        "q_down": layer.q_down,
        "q_up": layer.q_up,
        "q_absorb": layer.q_absorb,
        "kv_down": layer.kv_down,
        "indexer_proj": layer.indexer_proj,
        "kv_append": kv_append,
        "indexer_score": idx.score,
        "indexer_select": lambda: idx.select(scores),
        "index_map": mla.map_indices,
        "sparse_mla": lambda: mla.attend(mapped),
        "o_absorb": lambda: layer.o_absorb(attn.view(layer.batch, -1)),
        "o_proj": layer.o_proj,
        "post_norm": layer.post_norm,
    }
    if is_moe:
        stages["router_score"] = layer.router_score
        stages["router_topk"] = layer.router_topk
    stages["mlp_experts"] = layer.mlp
    stages["layer_total"] = whole_layer
    return stages


def main(argv: Optional[list[str]] = None) -> int:
    from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend
    from benchmarks.backends.mla_flashmla import FlashMLASparseBackend

    args = parse_args(argv)
    run_id = args.run_id or results.new_run_id("layer")
    combos = [(m, b, n) for m in args.mlp for b in args.batch_sizes for n in args.context_lens]

    if args.dry_run:
        print(f"dry-run: {len(combos)} configurations, budget {args.memory_budget_gib} GiB")
        for m, b, n in combos:
            lc = LayerConfig(mlp=m)
            layer = DeepSeekV32Layer(lc)
            cfg = DSAConfig(batch_size=b, context_len=n, heads=lc.index_heads,
                            dim=lc.index_dim, topk=args.topk, seed=args.seed,
                            block_kv=args.block_kv)
            mla = FlashMLASparseBackend(heads_q=lc.n_heads, page_block=args.block_kv)
            est = (layer.estimate_bytes(b) + DeepGEMMDSABackend.estimate_bytes(cfg)
                   + mla.estimate_bytes(b, n, args.topk)) / 2**30
            print(f"  mlp={m:5s} B={b} N={n:>7}  est={est:7.2f} GiB  "
                  f"mlp weight read/step={layer.mlp_weight_bytes(b)/2**20:8.0f} MiB")
        free, total = torch.cuda.mem_get_info()
        print(f"device free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB")
        return 0

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    writer = results.RunWriter(root=args.out_dir, run_id=run_id).open()
    writer.write_metadata({
        "run_id": run_id, "workload": "decoder_layer", "command": " ".join(sys.argv),
        "resolved_config": {k: v for k, v in vars(args).items()},
        "layer_config": vars(LayerConfig()),
        "input_distribution": INPUT_DISTRIBUTION,
        "scope_note": (
            "ONE DeepSeek-V3.2 decoder layer of 61, decode step. No embedding, LM head, "
            "or other layers; no tensor/expert parallelism. MoE experts run as "
            "per-(token,expert) GEMVs with routing held fixed during timing, not a fused "
            "grouped-GEMM MoE kernel: weight traffic is right, launch overhead is "
            "pessimistic. Percentages are fractions of this measured layer, NOT of "
            "end-to-end model inference."
        ),
        "environment": env_info.collect(dev.index or 0, extra_repos={
            "DeepGEMM": os.path.join(REPO_ROOT, "third_party", "DeepGEMM"),
            "FlashMLA": os.path.join(REPO_ROOT, "third_party", "FlashMLA"),
        }),
        "note": args.note,
    })
    print(f"run_id={run_id}  dir={writer.dir}")

    for m, b, n in combos:
        lc = LayerConfig(mlp=m)
        layer = DeepSeekV32Layer(lc)
        cfg = DSAConfig(batch_size=b, context_len=n, heads=lc.index_heads,
                        dim=lc.index_dim, topk=args.topk, seed=args.seed,
                        block_kv=args.block_kv, query_pool=8,
                        steps=1)   # +1 token of cache capacity for this step's append
        idx = DeepGEMMDSABackend()
        mla = FlashMLASparseBackend(heads_q=lc.n_heads, page_block=args.block_kv)
        base = {
            "run_id": run_id, "workload": "decoder_layer", "mode": f"layer_{m}",
            "backend": "deepseek_v32_layer", "backend_commit": f"{idx.commit}|{mla.commit}",
            "dtype": "bf16_weights_fp8_kv", "batch_size": b, "candidate_count": n,
            "dimension": lc.hidden, "indexer_heads": lc.index_heads, "k": args.topk,
            "seed": args.seed, "graph_mode": "eager", "topk_sorted": False,
            "block_kv": args.block_kv, "context_len_actual": n,
            "config_id": f"layer_{m}|B{b}|N{n}",
        }
        est = (layer.estimate_bytes(b) + DeepGEMMDSABackend.estimate_bytes(cfg)
               + mla.estimate_bytes(b, n, args.topk)) / 2**30
        if est > args.memory_budget_gib:
            writer.write_summary({**base, "stage": "all", "status": "over_budget",
                                  "error": f"{est:.2f} GiB"})
            print(f"  SKIP {base['config_id']}: over budget {est:.2f} GiB")
            continue

        print(f"  RUN  {base['config_id']}  est={est:.2f} GiB  "
              f"mlp_read={layer.mlp_weight_bytes(b)/2**20:.0f} MiB", flush=True)
        try:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            layer.setup(b, args.seed, dev)
            idx.setup(cfg, dev)
            mla.setup(b, n, args.topk, args.seed, dev)
            e1.record(); torch.cuda.synchronize()
            setup_us = e0.elapsed_time(e1) * 1e3

            stages = build_stages(layer, idx, mla, cfg)
            rows, summ = [], []
            for gm in args.graph_modes:
                for stage, fn in stages.items():
                    gb = {**base, "graph_mode": gm}
                    try:
                        run = capture_graph(fn)[0].replay if gm == "cudagraph" else fn
                        grp = autoscale_group_size(run)
                        with MemoryProbe(dev.index or 0) as mem:
                            t = time_gpu(run, warmup=args.warmup, iters=args.iters,
                                         group_size=grp)
                    except GraphUnsupported as exc:
                        summ.append({**gb, "stage": stage, "status": "graph_unsupported",
                                     "error": str(exc)[:400]})
                        continue
                    except Exception as exc:
                        summ.append({**gb, "stage": stage, "status": "error",
                                     "error": f"{type(exc).__name__}: {exc}"[:400]})
                        continue
                    rows.extend(results.raw_rows(t, base=gb, stage=stage))
                    summ.append(results.summarize(
                        t, base=gb, stage=stage, batch_size=b,
                        peak_allocated=mem.peak_allocated, peak_reserved=mem.peak_reserved,
                        setup_us=setup_us if stage == "layer_total" else None))
                if gm == "cudagraph":
                    torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            writer.write_summary({**base, "stage": "all", "status": "oom", "error": str(exc)[:400]})
            print(f"  OOM  {base['config_id']}")
            continue
        except Exception as exc:
            traceback.print_exc()
            writer.write_summary({**base, "stage": "all", "status": "error",
                                  "error": f"{type(exc).__name__}: {exc}"[:400]})
            continue

        writer.write_raw(rows)
        tot = {r["graph_mode"]: r["median_us"] for r in summ
               if r.get("stage") == "layer_total" and r.get("status") == "ok"}
        for r in summ:
            writer.write_summary(r)
            if r.get("status") == "ok" and r["graph_mode"] == "cudagraph":
                t0 = tot.get("cudagraph")
                print(f"       {r['stage']:16s} {r['median_us']:9.2f} us"
                      + (f"  {100*r['median_us']/t0:5.1f}%" if t0 else ""))
        idx.teardown(); mla.teardown(); layer.teardown()

    writer.close()
    print(f"done: {writer.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
