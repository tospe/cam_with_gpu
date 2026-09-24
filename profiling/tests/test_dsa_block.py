#!/usr/bin/env python3
"""Correctness checks for the DSA attention block (indexer -> top-k -> sparse MLA).

Establishes that the block computes real sparse MLA over the indexer's selected
tokens:
  * FlashMLA's sparse decode matches a float64 reference built by dequantising
    its own KV cache and attending over exactly the selected ids;
  * the on-device index mapping equals FlashMLA's own
    ``abs_indices2indices_in_kvcache`` helper, including for shuffled paging;
  * the attended set really is the selected set -- perturbing a selected token
    changes the output, perturbing an unselected one does not;
  * end to end, the ids the indexer chooses are the ids the attention reads.

Run:  source scripts/env.sh && $PYTHON -m pytest tests/test_dsa_block.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.backends.base import DSAConfig


def _avail() -> bool:
    try:
        from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend
        from benchmarks.backends.mla_flashmla import FlashMLASparseBackend

        return DeepGEMMDSABackend.availability()[0] and FlashMLASparseBackend.availability()[0]
    except Exception:
        return False


BLOCK = pytest.mark.skipif(not _avail(), reason="deep_gemm / flash_mla unavailable")
DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _make_mla(b, n, k, seed=0):
    from benchmarks.backends.mla_flashmla import FlashMLASparseBackend

    be = FlashMLASparseBackend(heads_q=128, page_block=64)
    be.setup(b, n, k, seed, DEV)
    return be


def _random_selection(b, n, k, seed=1):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return torch.stack([torch.randperm(n, generator=g, device=DEV)[:k] for _ in range(b)]).to(torch.int32)


def _fp64_reference(be, mapped):
    """Attend over the selected tokens using the dequantised cache, in float64."""
    from benchmarks.backends.mla_flashmla import D_QK, D_V

    deq = be.quant.dequantize_k_cache(be.k_cache, be.quant.KVCacheLayout.V32_FP8Sparse)
    flat = deq.view(-1, D_QK).double()
    out = torch.empty(be.batch, 1, be.heads_q, D_V, dtype=torch.float64, device=DEV)
    scale = D_QK ** -0.5
    for i in range(be.batch):
        kv = flat[mapped[i, 0].long()]                       # [k, 576]
        p = torch.softmax((be.q[i, 0].double() @ kv.T) * scale, dim=-1)
        out[i, 0] = p @ kv[:, :D_V]
    return out


@BLOCK
@pytest.mark.parametrize("b,n,k", [(1, 4096, 2048), (2, 8192, 2048), (4, 8192, 2048)])
def test_sparse_mla_matches_fp64_reference(b, n, k):
    be = _make_mla(b, n, k)
    be.set_selected_ids(_random_selection(b, n, k))
    mapped = be.map_indices()
    got = be.attend(mapped)
    ref = _fp64_reference(be, mapped)
    rel = (got.double() - ref).abs().max() / ref.abs().max()
    # bf16 output over an fp8 latent cache; 1e-2 is the dtype-appropriate bound.
    assert rel < 1e-2, f"relative error {rel:.3e}"
    be.teardown()


@BLOCK
def test_index_mapping_matches_flashmla_helper():
    """Our fused on-device mapping must equal FlashMLA's own CPU helper."""
    b, n, k = 3, 8192, 2048
    be = _make_mla(b, n, k)
    be.set_selected_ids(_random_selection(b, n, k, seed=5))
    mine = be.map_indices()
    theirs = be.quant.abs_indices2indices_in_kvcache(
        be.abs_indices.cpu(), be.block_table.cpu(), be.page_block
    ).to(mine.device)
    assert torch.equal(mine, theirs)
    # And the mapping must land inside the cache.
    n_slots = be.k_cache.shape[0] * be.k_cache.shape[1]
    assert int(mine.min()) >= 0 and int(mine.max()) < n_slots
    be.teardown()


@BLOCK
def test_attention_reads_exactly_the_selected_tokens():
    """Perturbing a selected token changes the output; an unselected one does not."""
    b, n, k = 1, 8192, 2048
    be = _make_mla(b, n, k)
    sel = _random_selection(b, n, k, seed=7)
    be.set_selected_ids(sel)
    mapped = be.map_indices()
    base = be.attend(mapped).clone()

    chosen = set(sel[0].tolist())
    unselected = next(i for i in range(n) if i not in chosen)

    # The KV cache is deliberately non-contiguous (FlashMLA's own layout), so
    # address it by (block, slot) rather than flattening it.
    bs = be.page_block

    def slot_view(phys: int):
        return be.k_cache[phys // bs, phys % bs, 0]

    # Perturb an UNSELECTED token -> output must be bit-identical.
    phys_un = int(be.block_table[0, unselected // bs]) * bs + unselected % bs
    assert phys_un not in set(mapped[0, 0].tolist())
    saved = slot_view(phys_un).clone()
    slot_view(phys_un).zero_()
    assert torch.equal(be.attend(mapped), base), "unselected token affected the output"
    slot_view(phys_un).copy_(saved)

    # Perturb a SELECTED token -> output must change.
    phys_sel = int(mapped[0, 0, 0])
    saved = slot_view(phys_sel).clone()
    slot_view(phys_sel).zero_()
    assert not torch.equal(be.attend(mapped), base), "selected token did not affect the output"
    slot_view(phys_sel).copy_(saved)
    be.teardown()


@BLOCK
def test_end_to_end_block_uses_indexer_selection():
    """The ids the indexer produces are the ids the attention actually reads."""
    from benchmarks.backends.dsa_deepgemm import DeepGEMMDSABackend

    b, n, k = 2, 8192, 2048
    cfg = DSAConfig(batch_size=b, context_len=n, heads=64, dim=128, topk=k,
                    seed=3, query_pool=4)
    idx = DeepGEMMDSABackend()
    idx.setup(cfg, DEV)
    mla = _make_mla(b, n, k, seed=3)

    scores = idx.score()
    ids = idx.select(scores)
    assert ids.shape == (b, k)
    mla.set_selected_ids(ids)
    mapped = mla.map_indices()

    # Every mapped slot must correspond to the logical id the indexer chose.
    for i in range(b):
        expected = (mla.block_table[i, ids[i].long() // mla.page_block].long() * mla.page_block
                    + ids[i].long() % mla.page_block)
        assert torch.equal(mapped[i, 0].long(), expected)

    out = mla.attend(mapped)
    assert out.shape == (b, 1, 128, 512)
    assert torch.isfinite(out).all()

    # The block output must depend on the indexer's choice: a different query
    # selects a different set and must give a different attention result.
    first = out.clone()
    idx.stage_query(1)
    ids2 = idx.select(idx.score())
    assert not torch.equal(ids, ids2), "different query produced identical selection"
    mla.set_selected_ids(ids2)
    assert not torch.equal(mla.attend(mla.map_indices()), first)

    idx.teardown()
    mla.teardown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
