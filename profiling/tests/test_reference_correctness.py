#!/usr/bin/env python3
"""Correctness checks for the DSA and router workloads.

What these tests establish:
  * the backends compute the documented DSA / router mathematics, checked
    against a float64 reference on small inputs;
  * masking, batching, incremental cache updates, and k near the candidate
    count behave correctly;
  * selection output is structurally valid (shape, dtype, unique ids, in-range
    ids, causal bounds, score/id correspondence);
  * ties are handled explicitly: a different-but-tied id set passes, a set
    containing a strictly lower score than an unselected candidate does not.

What they do NOT establish: model quality.  All inputs are random, so selection
overlap between numerical configurations is reported as a numerical-sensitivity
diagnostic, never as an accuracy result.

Run:  .venv/bin/python -m pytest tests/test_reference_correctness.py -v
  or  .venv/bin/python tests/test_reference_correctness.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.backends.base import (
    DSAConfig,
    RouterConfig,
    dsa_reference_scores,
    router_reference_logits,
)
from benchmarks.backends.dsa_torch import TorchDSABackend
from benchmarks.backends.router_torch import TorchRouterBackend

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Tolerances are dtype-appropriate: bf16 has ~8 mantissa bits, and the score is
# a sum of H=64 weighted dot products of length D, so relative error accumulates.
TOL = {"bf16": 2e-2, "fp8_e4m3": 1.5e-1}


# ---------------------------------------------------------------------------
# Selection validity helpers
# ---------------------------------------------------------------------------

def assert_selection_valid(ids: torch.Tensor, scores: torch.Tensor, k: int, context_lens=None):
    """Structural checks that apply to any top-k selection."""
    b, n = scores.shape
    assert ids.shape == (b, k), f"expected ids {(b, k)}, got {tuple(ids.shape)}"
    assert ids.dtype in (torch.int32, torch.int64), f"unexpected id dtype {ids.dtype}"
    assert int(ids.min()) >= 0 and int(ids.max()) < n, "id out of candidate range"
    for i in range(b):
        row = ids[i]
        assert len(torch.unique(row)) == k, f"duplicate ids in batch row {i}"
        if context_lens is not None:
            limit = int(context_lens[i])
            assert int(row.max()) < limit, f"row {i} selected a masked/causal-future id"


def assert_topk_is_optimal(ids: torch.Tensor, ref_scores: torch.Tensor, k: int, atol: float = 0.0):
    """Tie-aware optimality check.

    The selected set need not match the reference ids -- tied candidates are
    interchangeable -- but every selected score must be >= the largest
    *unselected* score (within ``atol``).  A selection containing a strictly
    lower score than some rejected candidate fails.
    """
    b, n = ref_scores.shape
    for i in range(b):
        sel = ids[i].long()
        mask = torch.ones(n, dtype=torch.bool, device=ref_scores.device)
        mask[sel] = False
        rejected = ref_scores[i][mask]
        if rejected.numel() == 0:
            continue
        worst_selected = ref_scores[i][sel].min()
        best_rejected = rejected.max()
        assert worst_selected >= best_rejected - atol, (
            f"row {i}: selected score {worst_selected.item():.6g} < rejected "
            f"{best_rejected.item():.6g}; selection is not a valid top-{k}"
        )


# ---------------------------------------------------------------------------
# DSA: scoring semantics
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("b,n,h,d", [(1, 256, 8, 128), (3, 1024, 64, 128), (2, 129, 16, 128)])
def test_dsa_scores_match_reference(b, n, h, d):
    """score[b,j] = sum_h w[b,h] * ReLU(dot(q[b,h,:], k[b,j,:]))."""
    cfg = DSAConfig(batch_size=b, context_len=n, heads=h, dim=d, topk=16, seed=7, query_pool=4)
    be = TorchDSABackend(score_chunk=64)  # small chunk exercises the chunk loop
    be.setup(cfg, DEV)
    got = be.score().float()
    ref = dsa_reference_scores(be.q, be.k_cache[:, :n], be.w).float()
    rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-9)
    assert rel < TOL["bf16"], f"relative error {rel:.3e}"
    be.teardown()


@CUDA
def test_dsa_relu_actually_applied():
    """Negative dot products must contribute zero, not a negative value.

    Distinguishes the DSA weighted-ReLU score from a plain weighted dot product.
    """
    b, n, h, d = 1, 64, 4, 128
    cfg = DSAConfig(batch_size=b, context_len=n, heads=h, dim=d, topk=8, seed=1, query_pool=2)
    be = TorchDSABackend(score_chunk=0)
    be.setup(cfg, DEV)
    # Force a key that is the exact negation of head 0's query direction.
    be.k_cache[0, 0] = -be.q[0, 0]
    be.w.fill_(1.0)
    got = be.score().float()
    ref = dsa_reference_scores(be.q, be.k_cache[:, :n], be.w).float()
    no_relu = (torch.bmm(be.q.double(), be.k_cache[:, :n].double().transpose(1, 2))
               * be.w.double().unsqueeze(-1)).sum(1).float()
    assert torch.allclose(got, ref, rtol=TOL["bf16"], atol=1e-3)
    assert not torch.allclose(got[0, 0], no_relu[0, 0], rtol=1e-2), \
        "score for the anti-aligned key matches the non-ReLU form; ReLU is not applied"
    be.teardown()


@CUDA
def test_dsa_masking_and_growth():
    """Positions beyond the valid length are masked; appended keys become visible."""
    b, n, steps = 2, 96, 5
    cfg = DSAConfig(batch_size=b, context_len=n, heads=8, dim=128, topk=8,
                    steps=steps, seed=3, query_pool=8)
    be = TorchDSABackend(score_chunk=0)
    be.setup(cfg, DEV)

    # Capacity is preallocated, but only `n` entries are valid before any append.
    assert be.k_cache.shape[1] == n + steps
    assert be.score().shape[-1] == n

    for t in range(steps):
        be.append(t)
        assert be.context_len == n + t + 1
        scores = be.score().float()
        assert scores.shape[-1] == n + t + 1
        ref = dsa_reference_scores(be.q, be.k_cache[:, : n + t + 1], be.w).float()
        rel = (scores - ref).abs().max() / ref.abs().max().clamp_min(1e-9)
        assert rel < TOL["bf16"], f"step {t}: relative error {rel:.3e}"
        # The token appended at step t must be visible to the query at step t.
        appended = dsa_reference_scores(
            be.q, be.k_cache[:, n + t : n + t + 1], be.w
        ).float()
        assert torch.allclose(scores[:, n + t], appended[:, 0], rtol=TOL["bf16"], atol=1e-3), \
            f"step {t}: appended key is not visible in the scores"
    be.teardown()


@CUDA
def test_dsa_reset_restores_state():
    """reset() must restore cache contents, valid length and replay position."""
    cfg = DSAConfig(batch_size=2, context_len=64, heads=8, dim=128, topk=8,
                    steps=4, seed=5, query_pool=8)
    be = TorchDSABackend(score_chunk=0)
    be.setup(cfg, DEV)
    before = be.score().clone()
    for t in range(4):
        be.append(t)
    assert be.context_len == 68
    be.reset()
    assert be.context_len == 64
    assert torch.equal(be.score(), before), "reset did not restore the pre-append state"
    be.teardown()


@CUDA
@pytest.mark.parametrize("b,n,k", [(1, 64, 1), (2, 64, 63), (2, 64, 64), (3, 128, 2048)])
def test_dsa_selection_valid_including_k_near_n(b, n, k):
    """Selection validity, including k close to and above the candidate count."""
    cfg = DSAConfig(batch_size=b, context_len=n, heads=8, dim=128, topk=k, seed=11, query_pool=4)
    be = TorchDSABackend(score_chunk=0)
    be.setup(cfg, DEV)
    scores = be.score()
    ids = be.select(scores)
    eff_k = min(k, n)
    assert_selection_valid(ids, scores, eff_k)
    ref = dsa_reference_scores(be.q, be.k_cache[:, :n], be.w)
    # bf16 scoring can reorder near-equal candidates; allow a tolerance scaled
    # to the score magnitude when checking optimality against the fp64 reference.
    atol = float(ref.abs().max()) * TOL["bf16"]
    assert_topk_is_optimal(ids, ref, eff_k, atol=atol)
    be.teardown()


@CUDA
def test_dsa_ties_are_handled():
    """With all-equal scores any k ids are valid; a strictly worse pick is not."""
    n, k = 32, 8
    scores = torch.zeros(1, n, device=DEV)
    ids = torch.topk(scores, k, dim=-1)[1]
    assert_selection_valid(ids, scores, k)
    assert_topk_is_optimal(ids, scores, k)          # all tied -> any set is optimal

    scores = torch.arange(n, device=DEV, dtype=torch.float32).unsqueeze(0)
    bad = torch.arange(k, device=DEV).unsqueeze(0)  # the k *smallest*
    with pytest.raises(AssertionError):
        assert_topk_is_optimal(bad, scores, k)


@CUDA
def test_dsa_scores_correspond_to_selected_ids():
    """Gathering the reference scores at the selected ids reproduces top-k values."""
    cfg = DSAConfig(batch_size=2, context_len=512, heads=16, dim=128, topk=32,
                    seed=13, query_pool=4)
    be = TorchDSABackend(score_chunk=0)
    be.setup(cfg, DEV)
    scores = be.score()
    ids = be.select(scores)
    gathered = torch.gather(scores, 1, ids)
    expected = torch.topk(scores, 32, dim=-1)[0]
    assert torch.allclose(gathered.sort(dim=-1)[0], expected.sort(dim=-1)[0])
    be.teardown()


@CUDA
def test_dsa_batch_caches_are_independent():
    """Each batch row must own its cache: writing one row cannot move another."""
    cfg = DSAConfig(batch_size=4, context_len=128, heads=8, dim=128, topk=8,
                    steps=2, seed=17, query_pool=4)
    be = TorchDSABackend(score_chunk=0)
    be.setup(cfg, DEV)
    base = be.score().clone()
    be.k_cache[1, :64] = 0          # perturb only row 1
    after = be.score()
    assert not torch.equal(after[1], base[1])
    for i in (0, 2, 3):
        assert torch.equal(after[i], base[i]), f"row {i} changed when row 1 was written"
    be.teardown()


# ---------------------------------------------------------------------------
# DSA: optimised FP8 backend against the same effective inputs
# ---------------------------------------------------------------------------

def _deepgemm_available() -> bool:
    try:
        from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

        return DeepGEMMDSABackend.availability()[0]
    except Exception:
        return False


DEEPGEMM = pytest.mark.skipif(not _deepgemm_available(), reason="deep_gemm unavailable")


@CUDA
@DEEPGEMM
@pytest.mark.parametrize("b,n", [(1, 512), (4, 1024), (2, 1000)])
def test_deepgemm_scores_match_dequantized_reference(b, n):
    """Implementation correctness: the kernel must match a float64 reference
    evaluated on *the same effective (dequantised) inputs*.

    This isolates kernel correctness from the separate question of how FP8
    quantisation shifts scores relative to bf16, which is checked below.
    """
    from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

    cfg = DSAConfig(batch_size=b, context_len=n, heads=64, dim=128, topk=64,
                    seed=23, query_pool=4)
    be = DeepGEMMDSABackend()
    be.setup(cfg, DEV)
    got = be.score()[:, :n].float()

    q_deq = be.q.squeeze(1).float()                         # [B, H, D]
    k_deq = _dequant_cache(be, n)                           # [B, N, D]
    ref = dsa_reference_scores(q_deq, k_deq, be.w).float()
    rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-9)
    assert rel < 1e-2, f"kernel vs same-input reference: relative error {rel:.3e}"
    be.teardown()


@CUDA
@DEEPGEMM
def test_deepgemm_incremental_updates_visible_and_bounded():
    """Appended keys become visible at the right step, and selection stays
    inside the valid prefix.

    The SM90 kernel cannot mask (``clean_logits`` is rejected), so the padding
    beyond the valid context holds unmasked scores against stale cache blocks.
    This test pins down both facts: the tail really is unmasked garbage, and
    ``select`` nonetheless never returns an id from it.
    """
    from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

    n, steps = 500, 6            # not a multiple of block_kv -> a real padded tail
    cfg = DSAConfig(batch_size=2, context_len=n, heads=64, dim=128, topk=32,
                    steps=steps, seed=29, query_pool=8)
    be = DeepGEMMDSABackend()
    be.setup(cfg, DEV)

    s = be.score()
    assert s.shape[-1] == be.max_model_len >= n
    assert torch.isfinite(s[:, :n]).all(), "valid region has non-finite scores"

    for t in range(steps):
        ctx = n + t + 1
        prev_tail = be.score()[:, ctx - 1].clone()   # not yet valid at step t-1
        be.append(t)
        assert be.context_len == ctx
        s = be.score()
        assert torch.isfinite(s[:, :ctx]).all(), f"step {t}: non-finite score in valid region"
        # The token appended at step t must be visible to the query at step t,
        # and its score must reflect the key just written.
        ref = dsa_reference_scores(
            be.q.squeeze(1).float(), _dequant_cache(be, ctx)[:, ctx - 1 :ctx], be.w
        ).float()
        assert torch.allclose(s[:, ctx - 1], ref[:, 0], rtol=1e-2, atol=1e-2), \
            f"step {t}: appended key not reflected in the scores"
        ids = be.select(s)
        assert_selection_valid(ids, s[:, :ctx], min(cfg.topk, ctx),
                               context_lens=[ctx] * cfg.batch_size)

    be.reset()
    assert be.context_len == n, "reset did not restore the valid length"
    ids = be.select(be.score())
    assert int(ids.max()) < n, "selection escaped the valid prefix after reset"
    be.teardown()


@CUDA
@DEEPGEMM
def test_fp8_vs_bf16_selection_overlap_is_a_diagnostic():
    """FP8 vs BF16 selection overlap on random data.

    Reported as numerical sensitivity only.  Random inputs carry no learned
    structure, so this says nothing about model quality.
    """
    from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

    n, k = 4096, 256
    cfg = DSAConfig(batch_size=2, context_len=n, heads=64, dim=128, topk=k,
                    seed=31, query_pool=4)
    be = DeepGEMMDSABackend()
    be.setup(cfg, DEV)
    fp8_ids = be.select(be.score())

    # Same effective inputs, evaluated in float64 -> the ideal selection.
    ref = dsa_reference_scores(be.q.squeeze(1).float(), _dequant_cache(be, n), be.w)
    ref_ids = torch.topk(ref, k, dim=-1)[1]

    overlap = [
        len(set(fp8_ids[i].tolist()) & set(ref_ids[i].tolist())) / k
        for i in range(cfg.batch_size)
    ]
    print(f"\nFP8 kernel vs float64-on-same-inputs top-{k} overlap: "
          f"{[f'{o:.3f}' for o in overlap]} (numerical diagnostic, not accuracy)")
    assert min(overlap) > 0.9, f"overlap unexpectedly low: {overlap}"
    be.teardown()


def _dequant_cache(be, n: int) -> torch.Tensor:
    """Reconstruct [B, N, D] float keys from the packed FP8 paged cache."""
    cfg = be.cfg
    d, bk = cfg.dim, cfg.block_kv
    flat = be.kv_cache.view(be.kv_cache.shape[0], -1)
    vals = flat[:, : bk * d].view(torch.float8_e4m3fn).view(-1, bk, d).float()
    scales = flat[:, bk * d :].view(torch.float32).view(-1, bk, 1)
    deq = vals * scales                                    # [num_blocks, bk, D]
    out = torch.empty(cfg.batch_size, n, d, device=deq.device, dtype=torch.float32)
    nb = (n + bk - 1) // bk
    for i in range(cfg.batch_size):
        blk = be.block_table[i, :nb].long()
        out[i] = deq[blk].reshape(-1, d)[:n]
    return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("b,e,d,k", [(1, 16, 128, 8), (8, 4096, 512, 8), (2, 1024, 4096, 2)])
def test_router_logits_match_reference(b, e, d, k):
    cfg = RouterConfig(batch_size=b, num_experts=e, dim=d, topk=k, seed=41, token_pool=4)
    be = TorchRouterBackend(dtype=torch.bfloat16)
    be.setup(cfg, DEV)
    got = be.score().float()
    ref = router_reference_logits(be.hidden, be.weights).float()
    rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-9)
    assert rel < TOL["bf16"], f"relative error {rel:.3e}"

    vals, ids = be.select(got)
    assert_selection_valid(ids, got, k)
    atol = float(ref.abs().max()) * TOL["bf16"]
    assert_topk_is_optimal(ids, ref, k, atol=atol)
    assert torch.allclose(vals, torch.gather(got, 1, ids)), "scores do not match returned ids"
    be.teardown()


@CUDA
def test_router_k_equal_to_e():
    """k == E must return every expert exactly once."""
    cfg = RouterConfig(batch_size=2, num_experts=16, dim=128, topk=16, seed=43, token_pool=2)
    be = TorchRouterBackend(dtype=torch.bfloat16)
    be.setup(cfg, DEV)
    _, ids = be.combined()
    for i in range(2):
        assert sorted(ids[i].tolist()) == list(range(16))
    be.teardown()


@CUDA
def test_router_rejects_k_greater_than_e():
    from benchmarks.backends.base import Unsupported

    cfg = RouterConfig(batch_size=1, num_experts=4, dim=128, topk=8, seed=47)
    with pytest.raises(Unsupported):
        TorchRouterBackend().setup(cfg, DEV)


@CUDA
def test_router_fp32_matches_bf16_selection_on_clear_margins():
    """bf16 and fp32 routers agree wherever the top-k margin exceeds bf16 error."""
    cfg = RouterConfig(batch_size=4, num_experts=2048, dim=512, topk=8, seed=53, token_pool=2)
    out = {}
    for tag, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        be = TorchRouterBackend(dtype=dt)
        be.setup(cfg, DEV)
        out[tag] = be.select(be.score().float())[1]
        be.teardown()
    overlap = [
        len(set(out["bf16"][i].tolist()) & set(out["fp32"][i].tolist())) / cfg.topk
        for i in range(cfg.batch_size)
    ]
    assert min(overlap) >= 0.75, f"bf16/fp32 routing overlap too low: {overlap}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
