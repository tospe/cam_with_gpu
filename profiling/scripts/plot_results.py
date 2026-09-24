#!/usr/bin/env python3
"""Figures for the H100 DSA / router search benchmarks.

Every point plotted is a measured median with a p5-p95 band from the raw
samples.  Lines between points are visual guides only; measured points are
always drawn as markers so interpolation is never mistaken for data.

Usage:
    python scripts/plot_results.py results/<dsa_run_id> [results/<router_run_id>]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STAGE_COLORS = {"score": "#1f77b4", "select": "#d62728", "combined": "#2ca02c",
                "decode_loop": "#9467bd", "decode_loop_per_step_mean": "#8c564b"}
BACKEND_MARKERS = {"torch_bf16": "o", "deepgemm_fp8_paged_mqa": "s", "torch_flat_router": "o"}
MODE_LINESTYLE = {"eager": "-", "cudagraph": "--"}


def save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def load(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ = pd.read_csv(os.path.join(run_dir, "summary.csv"))
    raw = pd.read_csv(os.path.join(run_dir, "measurements.csv"))
    return summ[summ["status"] == "ok"].copy(), raw


def band(raw: pd.DataFrame, keys: dict) -> tuple[float, float]:
    """p5-p95 of the raw samples matching ``keys``; the variability shown."""
    m = pd.Series(True, index=raw.index)
    for k, v in keys.items():
        m &= raw[k] == v
    s = raw.loc[m, "elapsed_us"]
    if s.empty:
        return (float("nan"), float("nan"))
    return float(s.quantile(0.05)), float(s.quantile(0.95))


# ---------------------------------------------------------------------------
# Figure 1: DSA latency vs context length
# ---------------------------------------------------------------------------

def fig_dsa_vs_context(summ, raw, out_dir):
    d = summ[(summ["workload"] == "dsa") & (summ["mode"] == "fixed")]
    if d.empty:
        return
    batches = sorted(d["batch_size"].unique())
    gms = sorted(d["graph_mode"].unique())
    fig, axes = plt.subplots(len(gms), len(batches),
                             figsize=(6.2 * len(batches), 4.6 * len(gms)),
                             squeeze=False, sharey=True)
    for r, gm in enumerate(gms):
        for c, b in enumerate(batches):
            ax = axes[r][c]
            sub = d[(d["batch_size"] == b) & (d["graph_mode"] == gm)]
            for backend in sorted(sub["backend"].unique()):
                for stage in ("score", "select", "combined"):
                    s = sub[(sub["backend"] == backend) & (sub["stage"] == stage)]
                    s = s.sort_values("candidate_count")
                    if s.empty:
                        continue
                    lo, hi = zip(*[
                        band(raw, {"backend": backend, "stage": stage, "batch_size": b,
                                   "candidate_count": n, "graph_mode": gm, "mode": "fixed"})
                        for n in s["candidate_count"]
                    ])
                    ax.fill_between(s["candidate_count"], lo, hi, alpha=0.15,
                                    color=STAGE_COLORS[stage], linewidth=0)
                    ax.plot(s["candidate_count"], s["median_us"],
                            marker=BACKEND_MARKERS.get(backend, "^"),
                            color=STAGE_COLORS[stage],
                            linestyle="-" if backend.startswith("deepgemm") else ":",
                            label=f"{backend} / {stage}", markersize=6)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("cached tokens searched, N")
            if c == 0:
                ax.set_ylabel("GPU latency (us)")
            ax.set_title(f"DSA fixed-context, B={b}, {gm}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle("DSA indexer: scoring, selection and combined latency vs context length\n"
                 "H100 PCIe 80GB - markers are measured medians, bands are p5-p95", y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "fig1_dsa_latency_vs_context")


# ---------------------------------------------------------------------------
# Figure 2: router latency vs expert count
# ---------------------------------------------------------------------------

def fig_router_vs_experts(summ, raw, out_dir):
    d = summ[summ["workload"] == "router"]
    if d.empty:
        return
    batches = sorted(d["batch_size"].unique())
    gms = sorted(d["graph_mode"].unique())
    fig, axes = plt.subplots(len(gms), len(batches),
                             figsize=(6.2 * len(batches), 4.6 * len(gms)),
                             squeeze=False, sharey=True)
    for r, gm in enumerate(gms):
        for c, b in enumerate(batches):
            ax = axes[r][c]
            sub = d[(d["batch_size"] == b) & (d["graph_mode"] == gm)]
            for stage in ("score", "select", "combined"):
                s = sub[sub["stage"] == stage].sort_values("candidate_count")
                if s.empty:
                    continue
                lo, hi = zip(*[
                    band(raw, {"stage": stage, "batch_size": b,
                               "candidate_count": e, "graph_mode": gm})
                    for e in s["candidate_count"]
                ])
                ax.fill_between(s["candidate_count"], lo, hi, alpha=0.15,
                                color=STAGE_COLORS[stage], linewidth=0)
                ax.plot(s["candidate_count"], s["median_us"], marker="o",
                        color=STAGE_COLORS[stage], label=stage, markersize=6)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("candidate experts, E")
            if c == 0:
                ax.set_ylabel("GPU latency (us)")
            ax.set_title(f"Flat router, B={b}, {gm}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("Flat MoE router: scoring, selection and combined latency vs expert count\n"
                 "routing only - no dispatch, expert execution or communication", y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "fig2_router_latency_vs_experts")


# ---------------------------------------------------------------------------
# Figure 3: DSA combined block latency and measured breakdown
# ---------------------------------------------------------------------------

def fig_dsa_block(summ, raw, out_dir):
    d = summ[(summ["workload"] == "dsa") & (summ["mode"] == "fixed")
             & (summ["graph_mode"] == "eager")]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for backend in sorted(d["backend"].unique()):
        for b in sorted(d["batch_size"].unique()):
            s = d[(d["backend"] == backend) & (d["batch_size"] == b)
                  & (d["stage"] == "combined")].sort_values("candidate_count")
            if s.empty:
                continue
            ax.plot(s["candidate_count"], s["median_us"], marker=BACKEND_MARKERS.get(backend, "^"),
                    label=f"{backend}, B={b}", markersize=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("cached tokens searched, N"); ax.set_ylabel("GPU latency (us)")
    ax.set_title("Combined DSA search (scoring + top-k)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    # Stage share of the separately measured stages.  This is NOT stacked onto
    # the combined measurement: `combined` is its own measurement and the stage
    # timings need not sum to it.
    ax = axes[1]
    be = "deepgemm_fp8_paged_mqa"
    sub = d[(d["backend"] == be) & (d["batch_size"] == 1)]
    if sub.empty:
        sub = d[d["batch_size"] == 1]
        be = sub["backend"].iloc[0] if not sub.empty else ""
    ns = sorted(sub["candidate_count"].unique())
    sc = [sub[(sub["candidate_count"] == n) & (sub["stage"] == "score")]["median_us"].median() for n in ns]
    se = [sub[(sub["candidate_count"] == n) & (sub["stage"] == "select")]["median_us"].median() for n in ns]
    co = [sub[(sub["candidate_count"] == n) & (sub["stage"] == "combined")]["median_us"].median() for n in ns]
    x = range(len(ns))
    ax.bar([i - 0.2 for i in x], sc, width=0.2, label="score (measured alone)", color=STAGE_COLORS["score"])
    ax.bar([i for i in x], se, width=0.2, label="select (measured alone)", color=STAGE_COLORS["select"])
    ax.bar([i + 0.2 for i in x], co, width=0.2, label="combined (measured)", color=STAGE_COLORS["combined"])
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{n//1024}K" for n in ns])
    ax.set_xlabel("cached tokens searched, N"); ax.set_ylabel("GPU latency (us)")
    ax.set_title(f"{be}, B=1: stages measured side by side\n(grouped, not stacked - stages need not sum)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "fig3_dsa_block_and_breakdown")


# ---------------------------------------------------------------------------
# Figure 4: throughput vs candidate count
# ---------------------------------------------------------------------------

def fig_throughput(dsa_summ, router_summ, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    d = dsa_summ[(dsa_summ["workload"] == "dsa") & (dsa_summ["mode"] == "fixed")
                 & (dsa_summ["stage"] == "combined")]
    for backend in sorted(d["backend"].unique()):
        for b in sorted(d["batch_size"].unique()):
            for gm in sorted(d["graph_mode"].unique()):
                s = d[(d["backend"] == backend) & (d["batch_size"] == b)
                      & (d["graph_mode"] == gm)].sort_values("candidate_count")
                if s.empty:
                    continue
                ax.plot(s["candidate_count"], s["throughput_qps"],
                        marker=BACKEND_MARKERS.get(backend, "^"),
                        linestyle=MODE_LINESTYLE.get(gm, "-"),
                        label=f"{backend.split('_')[0]}, B={b}, {gm}", markersize=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("cached tokens searched, N")
    ax.set_ylabel("queries / s  (B / batch latency)")
    ax.set_title("DSA search throughput")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    if router_summ is not None and not router_summ.empty:
        d = router_summ[router_summ["stage"] == "combined"]
        for b in sorted(d["batch_size"].unique()):
            for gm in sorted(d["graph_mode"].unique()):
                s = d[(d["batch_size"] == b) & (d["graph_mode"] == gm)].sort_values("candidate_count")
                if s.empty:
                    continue
                ax.plot(s["candidate_count"], s["throughput_qps"], marker="o",
                        linestyle=MODE_LINESTYLE.get(gm, "-"),
                        label=f"B={b}, {gm}", markersize=6)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("candidate experts, E")
        ax.set_ylabel("tokens routed / s  (B / batch latency)")
        ax.set_title("Router throughput")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Throughput = batch size / batch latency (not 1 / per-query latency)", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "fig4_throughput_vs_candidates")


# ---------------------------------------------------------------------------
# Figure 5: incremental decode loop
# ---------------------------------------------------------------------------

def fig_incremental(summ, raw, out_dir):
    d = summ[(summ["workload"] == "dsa") & (summ["mode"] == "incremental")
             & (summ["stage"] == "decode_loop")]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for backend in sorted(d["backend"].unique()):
        for b in sorted(d["batch_size"].unique()):
            s = d[(d["backend"] == backend) & (d["batch_size"] == b)].sort_values("candidate_count")
            if s.empty:
                continue
            lo, hi = zip(*[
                band(raw, {"backend": backend, "stage": "decode_loop",
                           "batch_size": b, "candidate_count": n, "mode": "incremental"})
                for n in s["candidate_count"]
            ])
            # band must be in the same units as the line (ms, not us)
            ax.fill_between(s["candidate_count"],
                            [v / 1e3 for v in lo], [v / 1e3 for v in hi],
                            alpha=0.15, linewidth=0)
            ax.plot(s["candidate_count"], s["median_us"] / 1e3,
                    marker=BACKEND_MARKERS.get(backend, "^"),
                    label=f"{backend}, B={b}", markersize=6)
    steps = int(d["steps"].dropna().iloc[0]) if d["steps"].notna().any() else "?"
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("initial cached tokens, N  (context grows by one token per step)")
    ax.set_ylabel(f"GPU time for the whole {steps}-step loop (ms)")
    ax.set_title(f"DSA incremental decode: append one indexer key + search, {steps} steps\n"
                 "context length grows within each measured loop", fontsize=10)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "fig5_dsa_incremental_loop")


# ---------------------------------------------------------------------------
# Figure 7: the combined search operation, on its own
# ---------------------------------------------------------------------------

def fig_combined_unit(dsa_summ, dsa_raw, router_summ, router_raw, out_dir):
    """Scoring + top-k as a single operation -- the unit a CAM would absorb.

    Plotted from the `combined` stage, which is measured end to end in one
    timed region.  It is not the sum of the separately measured score and
    select stages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    d = dsa_summ[(dsa_summ["workload"] == "dsa") & (dsa_summ["mode"] == "fixed")
                 & (dsa_summ["stage"] == "combined")]
    for backend in sorted(d["backend"].unique()):
        for b in sorted(d["batch_size"].unique()):
            for gm in sorted(d["graph_mode"].unique()):
                s = d[(d["backend"] == backend) & (d["batch_size"] == b)
                      & (d["graph_mode"] == gm)].sort_values("candidate_count")
                if s.empty:
                    continue
                lo, hi = zip(*[
                    band(dsa_raw, {"backend": backend, "stage": "combined", "batch_size": b,
                                   "candidate_count": n, "graph_mode": gm, "mode": "fixed"})
                    for n in s["candidate_count"]
                ])
                ax.fill_between(s["candidate_count"], lo, hi, alpha=0.13, linewidth=0)
                short = "DeepGEMM FP8" if backend.startswith("deepgemm") else "PyTorch BF16"
                ax.plot(s["candidate_count"], s["median_us"],
                        marker=BACKEND_MARKERS.get(backend, "^"),
                        linestyle=MODE_LINESTYLE.get(gm, "-"), ms=6,
                        label=f"{short}, B={b}, {gm}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("cached tokens searched, N")
    ax.set_ylabel("GPU latency of the whole search (µs)")
    ax.set_title("DSA indexer search\nscore + top-k, one timed operation", fontsize=11)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5)

    ax = axes[1]
    if router_summ is not None and not router_summ.empty:
        r = router_summ[router_summ["stage"] == "combined"]
        for b in sorted(r["batch_size"].unique()):
            for gm in sorted(r["graph_mode"].unique()):
                s = r[(r["batch_size"] == b) & (r["graph_mode"] == gm)].sort_values("candidate_count")
                if s.empty:
                    continue
                lo, hi = zip(*[
                    band(router_raw, {"stage": "combined", "batch_size": b,
                                      "candidate_count": e, "graph_mode": gm})
                    for e in s["candidate_count"]
                ])
                ax.fill_between(s["candidate_count"], lo, hi, alpha=0.13, linewidth=0)
                ax.plot(s["candidate_count"], s["median_us"], marker="o",
                        linestyle=MODE_LINESTYLE.get(gm, "-"), ms=6, label=f"B={b}, {gm}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("candidate experts, E")
        ax.set_ylabel("GPU latency of the whole search (µs)")
        ax.set_title("Flat MoE router search\nscore + top-k, one timed operation", fontsize=11)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Search latency vs candidate count — scoring and selection as a single operation\n"
                 "H100 PCIe 80GB · measured end to end, not the sum of separately timed stages", y=1.03)
    fig.tight_layout()
    save(fig, out_dir, "fig7_combined_search_vs_candidates")


# ---------------------------------------------------------------------------
# Figure 8: what fraction of the DSA attention block is the search
# ---------------------------------------------------------------------------

STAGE_ORDER = ["indexer_score", "indexer_select", "index_map", "sparse_mla_attend"]
STAGE_LABEL = {
    "indexer_score": "indexer scoring",
    "indexer_select": "top-k selection",
    "index_map": "index mapping",
    "sparse_mla_attend": "sparse MLA attention",
}
STAGE_FILL = {
    "indexer_score": "#1F77B4",
    "indexer_select": "#C1332C",
    "index_map": "#E0A33E",
    "sparse_mla_attend": "#6C757D",
}


def fig_block_composition(summ, out_dir, graph_mode="cudagraph", show_block_total=False):
    """Stage composition of the DSA attention block, and the search share.

    ``show_block_total`` overlays the independently measured ``dsa_block``
    median on each bar. It is off by default because it reads as clutter, but
    note that without it the stack height implies an end-to-end total that is
    not the measured one -- the four stages agree with the measured block only
    to within -18%/+3%. The right-hand panel's shares are always computed
    against the measured block, not against the stack.
    """
    d = summ[(summ["workload"] == "dsa_block") & (summ["graph_mode"] == graph_mode)]
    if d.empty:
        return
    piv = d.pivot_table(index=["batch_size", "candidate_count"],
                        columns="stage", values="median_us")
    if any(c not in piv.columns for c in STAGE_ORDER + ["dsa_block"]):
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    batches = sorted({b for b, _ in piv.index})

    # -- panel A: absolute stacked composition, with the measured block marked
    ax = axes[0]
    labels, xs, x = [], [], 0
    for b in batches:
        for (bb, n) in [i for i in piv.index if i[0] == b]:
            row = piv.loc[(bb, n)]
            bottom = 0.0
            for st in STAGE_ORDER:
                ax.bar(x, row[st], bottom=bottom, width=0.62, color=STAGE_FILL[st],
                       edgecolor="white", linewidth=0.5,
                       label=STAGE_LABEL[st] if x == 0 else None)
                bottom += row[st]
            if show_block_total:
                ax.plot([x - 0.38, x + 0.38], [row["dsa_block"]] * 2, color="#131C25",
                        lw=2.0, solid_capstyle="butt",
                        label="measured block total" if x == 0 else None)
            labels.append(f"{n // 1024}K")
            xs.append(x)
            x += 1
        x += 0.8
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("GPU latency (µs)" if show_block_total
                  else "GPU latency (µs)  —  bars sum separately measured stages")
    ax.set_xlabel("cached tokens searched, N        (left group B=1, right group B=8)")
    ax.set_title("DSA attention block, stage composition", fontsize=11)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    # -- panel B: search share of the measured block
    ax = axes[1]
    for b in batches:
        idx = sorted([i for i in piv.index if i[0] == b], key=lambda t: t[1])
        ns = [n for _, n in idx]
        search = [100 * (piv.loc[i, "indexer_score"] + piv.loc[i, "indexer_select"]
                         + piv.loc[i, "index_map"]) / piv.loc[i, "dsa_block"] for i in idx]
        attn = [100 * piv.loc[i, "sparse_mla_attend"] / piv.loc[i, "dsa_block"] for i in idx]
        ax.plot(ns, search, "o-", ms=6, label=f"search (score+top-k+map), B={b}")
        ax.plot(ns, attn, "s--", ms=5, label=f"sparse MLA attention, B={b}")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="#8A6A1F", ls=":", lw=1.2)
    ax.set_xlabel("cached tokens searched, N")
    ax.set_ylabel("share of the measured block (%)")
    ax.set_title("Search vs attention, as a share of the block", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5)

    fig.suptitle("How much of a DSA attention block is the top-k search?   "
                 f"H100 PCIe, k=2048, {graph_mode}\n"
                 "block inputs are prepared projections — no q/k projections, RoPE, "
                 "output projection or MLP", y=1.04)
    fig.tight_layout()
    save(fig, out_dir, f"fig8_dsa_block_composition_{graph_mode}")


# ---------------------------------------------------------------------------
# Figure 9: the search inside a full decoder layer
# ---------------------------------------------------------------------------

# A decoder layer contains TWO top-k searches: the DSA indexer over cached
# tokens, and the MoE router over candidate experts. Both are attributed to
# search; only the expert execution counts as feed-forward.
SEARCH_STAGES = ["indexer_proj", "indexer_score", "indexer_select", "index_map"]
ROUTER_STAGES = ["router_score", "router_topk"]
# kv_append is cache maintenance, not search: grouped with attention.
ATTN_STAGES = ["input_norm", "q_down", "q_up", "q_absorb", "kv_down", "kv_append", "sparse_mla", "o_absorb", "o_proj"]
FFN_STAGES = ["post_norm", "mlp_experts"]
GROUP_FILL = {"search": "#C1332C", "router": "#E0A33E",
              "attention": "#1F77B4", "ffn": "#6C757D"}
GROUP_LABEL = {
    "search": "DSA search  (indexer proj + score + top-k + map)",
    "router": "MoE router search  (score + group top-k)",
    "attention": "rest of attention  (q/kv proj, KV append, sparse MLA, o proj)",
    "ffn": "feed-forward  (post-norm + experts)",
}
GROUPS = ("search", "router", "attention", "ffn")


def fig_layer_share(summ, out_dir, graph_mode="cudagraph", only_mode=None):
    """The layer as its two sub-layers, with each top-k search nested inside.

    A decoder layer is two pre-normed residual sub-layers::

        h   = x + MLA(RMSNorm(x))       # the DSA indexer lives inside MLA
        out = h + FFN(RMSNorm(h))       # the MoE router lives inside FFN

    So the segments are ordered MLA-first, FFN-second, with the boundary drawn,
    and each sub-layer's search shown as its own segment. DSA is *not* MLA --
    it is the selection front-end that decides which 2048 cached tokens MLA
    then attends over.
    """
    SUB = {
        "dsa":    (["indexer_proj", "indexer_score", "indexer_select", "index_map"],
                   "#C1332C", "DSA search — indexer proj, score, top-k, id map"),
        "mla":    (["input_norm", "q_down", "q_up", "q_absorb", "kv_down", "sparse_mla", "o_absorb", "o_proj"],
                   "#1F77B4", "MLA attention — q/kv proj, sparse MLA, o proj"),
        "kv":     (["kv_append"],
                   "#7FB3D5", "KV cache append — writes MLA latent + indexer key"),
        "router": (["router_score", "router_topk"],
                   "#E0A33E", "MoE router search — score, group top-k"),
        "ffn":    (["post_norm", "mlp_experts"],
                   "#6C757D", "FFN — post-norm + experts"),
    }
    MLA_KEYS, FFN_KEYS = ("dsa", "mla", "kv"), ("router", "ffn")

    d = summ[(summ["workload"] == "decoder_layer") & (summ["graph_mode"] == graph_mode)]
    if only_mode:
        d = d[d["mode"] == only_mode]
    if d.empty:
        return
    piv = d.pivot_table(index=["mode", "batch_size", "candidate_count"],
                        columns="stage", values="median_us")
    if "layer_total" not in piv.columns:
        return
    piv = piv.fillna(0.0)
    for key, (stages, _, _) in SUB.items():
        have = [c for c in stages if c in piv.columns]
        piv[key] = piv[have].sum(axis=1) if have else 0.0
    piv["MLA"] = piv[list(MLA_KEYS)].sum(axis=1)
    piv["FFN"] = piv[list(FFN_KEYS)].sum(axis=1)
    piv["subtotal"] = piv["MLA"] + piv["FFN"]

    keys = tuple(k for k in ("dsa", "mla", "kv", "router", "ffn")
                 if piv[k].max() > 0)
    modes = [m for m in ("layer_moe", "layer_dense") if m in {i[0] for i in piv.index}]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # -- panel A: composition, MLA below the divider, FFN above
    ax = axes[0]
    xs, labels, x, group_mid = [], [], 0, []
    for m in modes:
        for b in sorted({i[1] for i in piv.index if i[0] == m}):
            first = x
            for n in sorted([i[2] for i in piv.index if i[0] == m and i[1] == b]):
                row = piv.loc[(m, b, n)]
                tot = row["subtotal"]
                bottom = 0.0
                for k in keys:
                    share = 100 * row[k] / tot
                    ax.bar(x, share, bottom=bottom, width=0.72, color=SUB[k][1],
                           edgecolor="white", linewidth=0.5,
                           label=SUB[k][2] if x == 0 else None)
                    bottom += share
                # the MLA / FFN boundary
                mla_pct = 100 * row["MLA"] / tot
                ax.plot([x - 0.42, x + 0.42], [mla_pct] * 2, color="#131C25", lw=1.8,
                        solid_capstyle="butt", zorder=5)
                ax.text(x, 101.5, f"{tot / 1000:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="#44545F", fontfamily="monospace")
                # MLA share goes under the tick, not inside the bar, so it stays
                # legible when the boundary falls on a dark segment.
                labels.append(f"{n // 1024}K\nMLA {mla_pct:.0f}%")
                xs.append(x); x += 1
            group_mid.append(((first + x - 1) / 2,
                              f"{'MoE' if m.endswith('moe') else 'dense'}  B={b}"))
            x += 0.9
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    for xm, lab in group_mid:
        ax.text(xm, -24, lab, ha="center", va="top", fontsize=9.5,
                color="#131C25", fontweight="medium")
    ax.set_ylim(0, 108)
    ax.set_ylabel("share of the layer (%)")
    ax.set_xlabel("")
    ax.set_title("The layer as two sub-layers\n"
                 "black rule = MLA/FFN boundary · top number = layer total, ms",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=2, frameon=False)

    # -- panel B: the two sub-layers in absolute terms
    ax = axes[1]
    style = {("layer_moe", 1): ("o-", 1), ("layer_moe", 8): ("s-", 8),
             ("layer_dense", 1): ("o--", 1), ("layer_dense", 8): ("s--", 8)}
    for m in modes:
        for b in sorted({i[1] for i in piv.index if i[0] == m}):
            idx = sorted([i for i in piv.index if i[0] == m and i[1] == b], key=lambda t: t[2])
            ns = [i[2] for i in idx]
            fmt = style.get((m, b), ("o-", b))[0]
            tag = f"{'MoE' if m.endswith('moe') else 'dense'} B={b}"
            ax.plot(ns, [piv.loc[i, "MLA"] for i in idx], fmt, color="#1F77B4", ms=6,
                    label=f"MLA sub-layer, {tag}")
            ax.plot(ns, [piv.loc[i, "FFN"] for i in idx], fmt, color="#6C757D", ms=6,
                    label=f"FFN sub-layer, {tag}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([400, 600, 1000, 2000, 3000, 5000]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("cached tokens searched, N")
    ax.set_ylabel("GPU latency (µs, log)")
    ax.set_title("MLA grows with context, FFN does not\n"
                 "the only context-dependent work in the layer is the DSA search",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5, ncol=2)

    kind = {"layer_moe": "MoE layer (58 of 61)",
            "layer_dense": "dense layer (3 of 61)"}.get(only_mode, "MoE and dense layers")
    fig.suptitle(f"MLA and FFN, and where each top-k search sits inside them — {kind}   "
                 f"H100 PCIe, k=2048, {graph_mode}\n"
                 "DSA is the selection front-end inside MLA, not MLA itself · "
                 "one layer of 61 · experts as per-(token,expert) GEMVs, routing fixed",
                 y=1.06)
    fig.tight_layout()
    tag = only_mode.replace("layer_", "") + "_" if only_mode else ""
    save(fig, out_dir, f"fig9_layer_search_share_{tag}{graph_mode}")


# ---------------------------------------------------------------------------
# Figure 10: which stages actually use the machine
# ---------------------------------------------------------------------------

HBM_PEAK_BPS = 1.86e12      # measured streaming peak on this H100 PCIe

STAGE_PRETTY = {
    "input_norm": "input RMSNorm",
    "q_down": "q down-projection",
    "q_up": "q up-projection",
    "q_absorb": "W_UK absorb + RoPE",
    "kv_down": "kv down-projection",
    "kv_append": "KV cache append",
    "o_absorb": "W_UV absorb",
    "indexer_proj": "indexer projections",
    "indexer_score": "indexer scoring",
    "indexer_select": "DSA top-k",
    "index_map": "index mapping",
    "sparse_mla": "sparse MLA attention",
    "o_proj": "output projection",
    "post_norm": "post-attention norm",
    "router_score": "router scoring",
    "router_topk": "MoE router top-k",
    "mlp_experts": "expert feed-forward",
}
SELECT_STAGES = {"indexer_select", "router_topk", "index_map"}


def fig_stage_efficiency(summ, out_dir, batch=1, context_len=131072,
                         mode="layer_moe", graph_mode="cudagraph"):
    """Achieved memory bandwidth per stage, against the HBM roofline.

    The point of the layer is not only how time divides but how well each part
    uses the machine. Bytes come from the documented compulsory-traffic model in
    ``DeepSeekV32Layer.stage_bytes`` (an upper bound on efficiency).
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmarks.backends.decoder_layer import DeepSeekV32Layer, LayerConfig

    d = summ[(summ["workload"] == "decoder_layer") & (summ["graph_mode"] == graph_mode)
             & (summ["mode"] == mode) & (summ["batch_size"] == batch)
             & (summ["candidate_count"] == context_len)]
    if d.empty:
        return
    layer = DeepSeekV32Layer(LayerConfig(mlp="moe" if mode.endswith("moe") else "dense"))
    rows = []
    for _, r in d.iterrows():
        by = layer.stage_bytes(r["stage"], batch, context_len)
        if by is None or r["stage"] == "layer_total":
            continue
        bw = by / (r["median_us"] / 1e6)
        rows.append((r["stage"], r["median_us"], by, bw, 100 * bw / HBM_PEAK_BPS))
    if not rows:
        return
    rows.sort(key=lambda t: t[4])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # -- panel A: % of roofline per stage
    ax = axes[0]
    names = [STAGE_PRETTY.get(r[0], r[0]) for r in rows]
    pcts = [max(r[4], 0.008) for r in rows]
    cols = ["#C1332C" if r[0] in SELECT_STAGES else "#1F77B4" for r in rows]
    ax.barh(range(len(rows)), pcts, color=cols, height=0.68)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 300)
    ax.axvline(100, color="#131C25", lw=1.4, ls="--")
    ax.text(100, len(rows) - 0.3, " HBM roofline", fontsize=8.5, color="#131C25", va="top")
    for i, r in enumerate(rows):
        pct = f"{r[4]:.2f}%" if r[4] < 10 else f"{r[4]:.0f}%"
        ax.text(max(r[4], 0.008) * 1.35, i, f"{pct}   ·   {r[1]:.0f} us",
                va="center", fontsize=8.5, color="#44545F")
    ax.set_xlabel("achieved memory bandwidth, % of 1.86 TB/s  (log)")
    ax.set_title(f"How well each stage uses the GPU\nMoE layer, B={batch}, N={context_len//1024}K",
                 fontsize=11)
    ax.grid(True, axis="x", which="both", alpha=0.3)

    # -- panel B: time vs bytes moved, with iso-bandwidth lines
    ax = axes[1]
    for bw, lab in ((HBM_PEAK_BPS, "1.86 TB/s"), (1e11, "100 GB/s"), (1e9, "1 GB/s")):
        xs = [1e3, 1e10]
        ax.plot(xs, [x / bw * 1e6 for x in xs], color="#B4C1CB", lw=1, ls=":", zorder=1)
        ax.text(1e10, 1e10 / bw * 1e6, f" {lab}", fontsize=8, color="#8895A3",
                va="center", ha="right")
    offs = [(8, 5), (8, -12), (-8, 8), (8, 5), (-8, -13), (8, -12),
            (8, 6), (-8, 7), (8, -13), (8, 6), (-8, -13)]
    for i, (st, us, by, bw, pct) in enumerate(rows):
        c = "#C1332C" if st in SELECT_STAGES else "#1F77B4"
        ax.scatter(by, us, s=52, color=c, zorder=3, edgecolor="white", linewidth=0.8)
        dx, dy = offs[i % len(offs)]
        ax.annotate(STAGE_PRETTY.get(st, st), (by, us), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color="#44545F",
                    ha="right" if dx < 0 else "left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bytes the stage must move (log)")
    ax.set_ylabel("GPU latency, µs (log)")
    ax.set_title("Time against work\ntop-left = costs time, moves almost nothing", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Every GEMM in the layer runs near the memory roofline; the selection kernels "
                 "run 200-10000x below it   H100 PCIe, cudagraph\n"
                 "bytes are compulsory traffic, so these are upper bounds on efficiency. Small "
                 "elementwise stages (post-norm) are latency-bound too, but cost little time — "
                 "selection is both inefficient and expensive.", y=1.06)
    fig.tight_layout()
    save(fig, out_dir, f"fig10_stage_efficiency_B{batch}_N{context_len}")


# ---------------------------------------------------------------------------
# Figure 11: the same layer with expert execution excluded
# ---------------------------------------------------------------------------

def fig_layer_share_no_experts(summ, out_dir, graph_mode="cudagraph", only_mode=None):
    """Composition of the decoder layer EXCLUDING the expert feed-forward.

    A deliberately different denominator: everything the layer does except
    executing the selected experts. Two reasons it is a meaningful one --
      * the experts are weight streaming at ~69% of the memory roofline, a
        bandwidth problem with its own solutions (quantisation, offload,
        expert parallelism) and not one a search structure addresses;
      * DeepSeek-V3.2 is not served on one GPU. Under expert parallelism each
        GPU holds a small slice of the 256 experts, so a single-GPU layer
        overstates expert cost relative to attention.
    It is NOT the full layer, and is labelled as such everywhere it appears.
    Tensor parallelism would also shard attention, so this is not by itself a
    per-GPU model either.
    """
    d = summ[(summ["workload"] == "decoder_layer") & (summ["graph_mode"] == graph_mode)]
    if only_mode:
        d = d[d["mode"] == only_mode]
    if d.empty:
        return
    piv = d.pivot_table(index=["mode", "batch_size", "candidate_count"],
                        columns="stage", values="median_us")
    for st in ROUTER_STAGES:
        if st not in piv.columns:
            piv[st] = 0.0
    piv = piv.fillna(0.0)
    piv["search"] = piv[SEARCH_STAGES].sum(axis=1)
    piv["router"] = piv[ROUTER_STAGES].sum(axis=1)
    piv["attention"] = piv[ATTN_STAGES].sum(axis=1)
    piv["ffn"] = piv["post_norm"]                       # experts excluded
    piv["subtotal"] = piv[["search", "router", "attention", "ffn"]].sum(axis=1)
    piv["all_search"] = piv["search"] + piv["router"]
    piv["selection"] = piv["indexer_select"] + piv["router_topk"] + piv["index_map"]

    modes = [m for m in ("layer_moe", "layer_dense") if m in {i[0] for i in piv.index}]
    groups = tuple(g for g in GROUPS if g != "router" or piv["router"].max() > 0)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    labels_ffn = dict(GROUP_LABEL)
    labels_ffn["ffn"] = "post-attention norm"

    ax = axes[0]
    xs, labels, x, group_mid = [], [], 0, []
    for m in modes:
        for b in sorted({i[1] for i in piv.index if i[0] == m}):
            first = x
            for n in sorted([i[2] for i in piv.index if i[0] == m and i[1] == b]):
                row = piv.loc[(m, b, n)]
                bottom = 0.0
                for gkey in groups:
                    share = 100 * row[gkey] / row["subtotal"]
                    ax.bar(x, share, bottom=bottom, width=0.72, color=GROUP_FILL[gkey],
                           edgecolor="white", linewidth=0.5,
                           label=labels_ffn[gkey] if x == 0 else None)
                    bottom += share
                ax.text(x, 101.5, f"{row['subtotal']:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color="#44545F", fontfamily="monospace")
                labels.append(f"{n // 1024}K"); xs.append(x); x += 1
            group_mid.append(((first + x - 1) / 2,
                              f"{'MoE' if m.endswith('moe') else 'dense'}  B={b}"))
            x += 0.9
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5)
    for xm, lab in group_mid:
        ax.text(xm, -13, lab, ha="center", va="top", fontsize=9, color="#131C25")
    ax.set_ylim(0, 108)
    ax.set_ylabel("share of the layer excluding experts (%)")
    ax.set_xlabel("")
    ax.set_title("Decoder layer WITHOUT expert execution\n"
                 "x ticks are cached tokens N · numbers above bars are that subtotal, µs",
                 fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False)

    ax = axes[1]
    style = {("layer_moe", 1): ("o-", "#C1332C"), ("layer_moe", 8): ("s-", "#7B1E19"),
             ("layer_dense", 1): ("o--", "#E0A33E"), ("layer_dense", 8): ("s--", "#8A6A1F")}
    for m in modes:
        for b in sorted({i[1] for i in piv.index if i[0] == m}):
            idx = sorted([i for i in piv.index if i[0] == m and i[1] == b], key=lambda t: t[2])
            ns = [i[2] for i in idx]
            fmt, col = style.get((m, b), ("o-", "#333"))
            lab = f"{'MoE' if m.endswith('moe') else 'dense'}, B={b}"
            ax.plot(ns, [100 * piv.loc[i, "all_search"] / piv.loc[i, "subtotal"] for i in idx],
                    fmt, color=col, ms=6, label=lab)
    ax.set_xscale("log", base=2); ax.set_ylim(0, 100)
    ax.set_xlabel("cached tokens searched, N")
    ax.set_ylabel("both searches, share of layer excluding experts (%)")
    ax.set_title("Search share once expert execution is set aside", fontsize=11)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8.5)
    # Same shares against the FULL layer, so the comparison is mode-specific.
    full = [100 * piv.loc[i, "all_search"] / piv.loc[i, "layer_total"] for i in piv.index]
    ax.annotate(f"with experts included\nthese are {min(full):.1f}–{max(full):.1f}%",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.5, color="#71828F")

    kind = {"layer_moe": "MoE layer (58 of 61)",
            "layer_dense": "dense layer (3 of 61)"}.get(only_mode, "MoE and dense layers")
    excl = [100 * piv.loc[i, "mlp_experts"] / piv.loc[i, "layer_total"] for i in piv.index]
    fig.suptitle(f"The same layer with expert execution excluded — {kind}   "
                 f"H100 PCIe, {graph_mode}\n"
                 f"NOT the full layer: the feed-forward removed here is "
                 f"{min(excl):.0f}–{max(excl):.0f}% of it (see Figure 9 for the complete picture)",
                 y=1.05)
    fig.tight_layout()
    tag = only_mode.replace("layer_", "") + "_" if only_mode else ""
    save(fig, out_dir, f"fig11_layer_excluding_experts_{tag}{graph_mode}")


# ---------------------------------------------------------------------------
# Figure 12: profile of the MLA sub-layer alone
# ---------------------------------------------------------------------------

# Individual blocks of the MLA sub-layer, grouped into families by colour.
# (family key, colour, family label)
MLA_FAMILIES = [
    ("in",     "#1F77B4", "input projections"),
    ("search", "#C1332C", "DSA search"),
    ("cache",  "#7FB3D5", "KV cache append"),
    ("attn",   "#27824A", "sparse MLA attention"),
    ("out",    "#4C6E8A", "output projections"),
]
# (stage, family, pretty name, weight bytes read -- for the annotation)
MLA_BLOCKS = [
    ("input_norm",     "in",     "input RMSNorm",              None),
    ("q_down",         "in",     "q down-proj  7168→1536",     22.0),
    ("q_up",           "in",     "q up-proj  1536→24576",      75.5),
    ("q_absorb",       "in",     "W_UK absorb + q RoPE",       16.8),
    ("kv_down",        "in",     "kv down-proj  7168→576",      8.3),
    ("indexer_proj",   "search", "indexer projections",        26.8),
    ("indexer_score",  "search", "indexer scoring",            None),
    ("indexer_select", "search", "top-k selection",            None),
    ("index_map",      "search", "id mapping",                 None),
    ("kv_append",      "cache",  "KV cache append",            None),
    ("sparse_mla",     "attn",   "sparse MLA attention",       None),
    ("o_absorb",       "out",    "W_UV absorb",                16.8),
    ("o_proj",         "out",    "output proj  16384→7168",   234.9),
]
FAMILY_COLOR = {k: c for k, c, _ in MLA_FAMILIES}


def fig_mla_profile(summ, out_dir, graph_mode="cudagraph", only_mode="layer_moe",
                    detail_n=131072):
    """Profile of the MLA sub-layer, block by block.

    Left: every individual block at one context length, absolute microseconds.
    Right: the same blocks rolled into families, as a share, across contexts.

    Note the finer blocks do not sum to the coarser ``attn_proj_in`` /
    ``attn_proj_out`` stages measured elsewhere -- timed alone in a tight loop a
    small projection keeps its inputs in cache, which it would not when run as
    part of a longer chain. Both are honest measurements of different things.
    """
    d = summ[(summ["workload"] == "decoder_layer") & (summ["graph_mode"] == graph_mode)
             & (summ["mode"] == only_mode)]
    if d.empty:
        return
    piv = d.pivot_table(index=["batch_size", "candidate_count"],
                        columns="stage", values="median_us").fillna(0.0)
    blocks = [b for b in MLA_BLOCKS if b[0] in piv.columns]
    if len(blocks) < 8:
        return
    for fam, _, _ in MLA_FAMILIES:
        cols = [b[0] for b in blocks if b[1] == fam]
        piv[f"fam_{fam}"] = piv[cols].sum(axis=1) if cols else 0.0
    piv["MLA"] = piv[[b[0] for b in blocks]].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))

    # -- panel A: every block, absolute, B=1 vs B=8
    ax = axes[0]
    batches = [b for b in (1, 8) if (b, detail_n) in piv.index]
    # sort by the larger of the two batch sizes, so a block that only becomes
    # expensive at B=8 (indexer scoring) does not sit low in the list
    order = sorted(blocks,
                   key=lambda b: max(piv.loc[(bb, detail_n), b[0]] for bb in batches))
    ys = range(len(order))
    h = 0.38
    for j, b in enumerate(batches):
        row = piv.loc[(b, detail_n)]
        vals = [row[st] for st, *_ in order]
        ax.barh([y + (j - 0.5) * h for y in ys], vals, height=h,
                color=[FAMILY_COLOR[f] for _, f, _, _ in order],
                alpha=1.0 if j == 0 else 0.55,
                edgecolor="white", linewidth=0.4)
        for y, v in zip(ys, vals):
            ax.text(v + 2, y + (j - 0.5) * h, f"{v:.0f}", va="center",
                    fontsize=7, color="#44545F")
    # The per-block weight bytes stay in MLA_BLOCKS (the page table uses them)
    # but are not annotated on the chart.
    ax.set_yticks(list(ys))
    ax.set_yticklabels([name for _, _, name, _ in order], fontsize=8.5)
    ax.set_xlabel("GPU latency (µs)   ·   solid B=1, pale B=8")
    ax.set_title(f"Every block of the MLA sub-layer, N={detail_n // 1024}K", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in MLA_FAMILIES]
    ax.legend(handles, [l for _, _, l in MLA_FAMILIES], fontsize=8,
              loc="lower right", frameon=True, framealpha=0.95)

    # -- panel B: families, as a share, across contexts
    ax = axes[1]
    xs, labels2, x, group_mid = [], [], 0, []
    for b in sorted({i[0] for i in piv.index}):
        first = x
        for n in sorted([i[1] for i in piv.index if i[0] == b]):
            row = piv.loc[(b, n)]
            bottom = 0.0
            for fam, col, lab in MLA_FAMILIES:
                share = 100 * row[f"fam_{fam}"] / row["MLA"]
                ax.bar(x, share, bottom=bottom, width=0.72, color=col,
                       edgecolor="white", linewidth=0.5,
                       label=lab if x == 0 else None)
                bottom += share
            ax.text(x, 101.5, f"{row['MLA']:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#44545F", fontfamily="monospace")
            labels2.append(f"{n // 1024}K"); xs.append(x); x += 1
        group_mid.append(((first + x - 1) / 2, f"B={b}"))
        x += 0.9
    ax.set_xticks(xs); ax.set_xticklabels(labels2, fontsize=8.5)
    for xm, lab in group_mid:
        ax.text(xm, -13, lab, ha="center", va="top", fontsize=9.5, color="#131C25")
    ax.set_ylim(0, 108)
    ax.set_ylabel("share of the MLA sub-layer (%)")
    ax.set_title("Rolled into families, across context lengths\n"
                 "x ticks are cached tokens N · number above bar = MLA total, µs",
                 fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False)

    kind = "MoE layer" if only_mode.endswith("moe") else "dense layer"
    fig.suptitle(f"Profile of the MLA sub-layer, block by block — {kind}, H100 PCIe, "
                 f"k=2048, {graph_mode}\n"
                 "FFN excluded · DSA is inside MLA · the output projection is the single "
                 "largest block, attention the smallest", y=1.03)
    fig.tight_layout()
    tag = only_mode.replace("layer_", "")
    save(fig, out_dir, f"fig12_mla_profile_{tag}_{graph_mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dsa_run", help="results/<dsa run id>")
    ap.add_argument("router_run", nargs="?", help="results/<router run id>")
    ap.add_argument("--block-run", default=None, help="results/<dsa block run id>")
    ap.add_argument("--layer-run", default=None, help="results/<full layer run id>")
    ap.add_argument("--out-dir", default=None,
                    help="default: <dsa_run>/figures")
    a = ap.parse_args()

    dsa_summ, dsa_raw = load(a.dsa_run)
    out_dir = a.out_dir or os.path.join(a.dsa_run, "figures")
    print(f"figures -> {out_dir}")

    fig_dsa_vs_context(dsa_summ, dsa_raw, out_dir)
    fig_dsa_block(dsa_summ, dsa_raw, out_dir)
    fig_incremental(dsa_summ, dsa_raw, out_dir)

    router_summ = router_raw = None
    if a.router_run:
        router_summ, router_raw = load(a.router_run)
        r_out = os.path.join(a.router_run, "figures")
        print(f"router figures -> {r_out}")
        fig_router_vs_experts(router_summ, router_raw, r_out)
        fig_throughput(dsa_summ, router_summ, r_out)
    fig_throughput(dsa_summ, router_summ, out_dir)
    fig_combined_unit(dsa_summ, dsa_raw, router_summ, router_raw, out_dir)
    if a.block_run:
        block_summ, _ = load(a.block_run)
        b_out = os.path.join(a.block_run, "figures")
        print(f"block figures -> {b_out}")
        for gm in ("cudagraph", "eager"):
            fig_block_composition(block_summ, b_out, gm)
    if a.layer_run:
        layer_summ, _ = load(a.layer_run)
        l_out = os.path.join(a.layer_run, "figures")
        print(f"layer figures -> {l_out}")
        for gm in ("cudagraph", "eager"):
            for md in ("layer_moe", "layer_dense"):
                fig_layer_share(layer_summ, l_out, gm, only_mode=md)
        for b in (1, 8):
            fig_stage_efficiency(layer_summ, l_out, batch=b, context_len=131072)
        for gm in ("cudagraph", "eager"):
            for md in ("layer_moe", "layer_dense"):
                fig_layer_share_no_experts(layer_summ, l_out, gm, only_mode=md)
        for md in ("layer_moe", "layer_dense"):
            fig_mla_profile(layer_summ, l_out, "cudagraph", only_mode=md)
    if a.router_run:
        fig_combined_unit(dsa_summ, dsa_raw, router_summ, router_raw,
                          os.path.join(a.router_run, "figures"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
