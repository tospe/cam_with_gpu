"""Selected-attention backend: FlashMLA's FP8 sparse MLA decode kernel (SM90).

This is the attention half of a DeepSeek-V3.2 DSA block. The indexer produces
``topk`` selected token ids; this backend feeds them to
``flash_mla_with_kvcache`` in sparse mode -- the kernel DeepSeek ships for V3.2
sparse decoding on Hopper -- and attends over the latent MLA KV cache.

It is a real MLA kernel, not a gather plus generic attention:
  * the cache is the **latent** KV cache, ``d_qk = 576`` (512 NoPE + 64 RoPE)
    with ``d_v = 512``, one KV head (MQA), paged;
  * quantisation is FlashMLA's own ``V32_FP8Sparse`` layout at 656 B/token
    (512 FP8 e4m3 NoPE values, 4 FP32 tile scales, 64 BF16 RoPE values),
    produced by FlashMLA's own ``tests/quant.py`` so the representation is
    exactly what the kernel requires;
  * selected ids are mapped from logical positions to the kernel's
    ``block_idx * block_size + offset`` addressing with FlashMLA's
    ``abs_indices2indices_in_kvcache``.

**Block inputs are prepared projections.** Query/key projections, the q-LoRA
path, RoPE application and the output projection are *not* included; the block
measured here starts from a prepared query and a populated cache. Anything
reported from it is a DSA-block number, never a full-model number.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import torch

from .base import Unsupported, make_generator, randn

# DeepSeek-V3.2 MLA geometry (absorbed decode form).
D_QK = 576          # 512 latent NoPE + 64 RoPE
D_V = 512
N_HEADS_Q = 128
PAGE_BLOCK = 64


def _flashmla_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "FlashMLA")
    )


def _import_quant():
    """FlashMLA's own quantisation/index helpers, so the KV layout is theirs."""
    tests_dir = os.path.join(_flashmla_root(), "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import quant  # type: ignore

    return quant


def _commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", _flashmla_root(), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


class FlashMLASparseBackend:
    """Sparse MLA decode over a selected set of ``topk`` cached tokens."""

    name = "flashmla_fp8_sparse_decode"
    dtype_tag = "fp8_e4m3_latent_kv_bf16_q"

    def __init__(self, heads_q: int = N_HEADS_Q, page_block: int = PAGE_BLOCK):
        self.heads_q = heads_q
        self.page_block = page_block
        self.commit = _commit()

    @staticmethod
    def availability() -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        try:
            import flash_mla  # noqa: F401
        except Exception as exc:
            return False, f"flash_mla import failed: {type(exc).__name__}: {exc}"
        major = torch.cuda.get_device_properties(0).major
        if major not in (9, 10):
            return False, f"FlashMLA sparse decode requires SM90/SM100, found SM{major}0"
        return True, ""

    def estimate_bytes(self, batch: int, context_len: int, topk: int) -> int:
        blocks_per_seq = (context_len + 1 + self.page_block - 1) // self.page_block
        total_blocks = blocks_per_seq * batch
        kv = total_blocks * self.page_block * 656          # V32_FP8Sparse
        kv_src = total_blocks * self.page_block * D_QK * 2  # bf16 staging (freed after setup)
        q = batch * 1 * self.heads_q * D_QK * 2
        out = batch * 1 * self.heads_q * D_V * 2
        idx = batch * 1 * topk * 4 * 2                      # logical + physical
        return kv + kv_src + q + out + idx

    # -- lifecycle ---------------------------------------------------------
    def setup(self, batch: int, context_len: int, topk: int, seed: int,
              device: torch.device) -> None:
        import flash_mla

        quant = _import_quant()
        self.quant = quant
        self.flash_mla = flash_mla
        self.batch, self.context_len, self.topk = batch, context_len, topk

        if context_len < topk:
            raise Unsupported(f"context_len {context_len} < topk {topk}")

        g = make_generator(seed + 1000, device)
        # +1 token of capacity so a decode step can append its own latent.
        bps = (context_len + 1 + self.page_block - 1) // self.page_block
        self.blocks_per_seq = bps
        total_blocks = bps * batch

        # Distinct, shuffled block ranges per sequence -- separate logical
        # caches, non-contiguous physical layout as in real paged serving.
        pool = torch.randperm(total_blocks, generator=g, device=device).to(torch.int32)
        self.block_table = pool.view(batch, bps).contiguous()

        # Latent KV cache, quantised once (outside timing) into FlashMLA's layout.
        kv_bf16 = randn(total_blocks, self.page_block, 1, D_QK,
                        generator=g, device=device, dtype=torch.bfloat16)
        kv_bf16.clamp_(-1.0, 1.0)
        self.k_cache = quant.quantize_k_cache(
            kv_bf16, quant.KVCacheLayout.V32_FP8Sparse).contiguous()
        del kv_bf16
        torch.cuda.empty_cache()

        # Prepared decode query (one token per sequence).
        self.q = randn(batch, 1, self.heads_q, D_QK, generator=g,
                       device=device, dtype=torch.bfloat16)
        self.q.clamp_(-1.0, 1.0)

        # Fixed buffer the indexer's selected ids are staged into.
        self.abs_indices = torch.zeros(batch, 1, topk, dtype=torch.int32, device=device)
        self.softmax_scale = D_QK ** -0.5

        # Scheduler metadata is created once and reused; FlashMLA asserts the
        # shapes stay consistent, which they do for a fixed configuration.
        self.sched_meta, _ = flash_mla.get_mla_metadata()

        # Fuse the index mapping into one kernel; compiled once, outside timing.
        self._mapper = torch.compile(FlashMLASparseBackend._map_fn, dynamic=False)
        self._mapper(self.abs_indices.view(batch, topk), self.block_table, self.page_block)
        self._bind_cache_views()
        torch.cuda.synchronize()

    def _bind_cache_views(self) -> None:
        """Typed views into the packed V32_FP8Sparse cache, for appending.

        The Sparse layout is TOKEN-major: each token owns 656 contiguous bytes
        -- 512 e4m3 NoPE values, then 4 FP32 tile scales, then 64 BF16 RoPE
        values. (The dense ``V32_FP8`` layout is field-major and different; do
        not reuse one for the other.)
        """
        nb, bs = self.k_cache.shape[0], self.page_block
        d_nope, tiles = 512, 4
        c = self.k_cache.view(nb, bs, -1)
        self._c_nope = c[..., :d_nope]
        self._c_scale = c[..., d_nope : d_nope + tiles * 4].view(torch.float32)
        self._c_rope = c[..., d_nope + tiles * 4 :].view(torch.bfloat16)

    def append_latent(self, latent: torch.Tensor) -> None:
        """Write one new MLA latent per sequence into the paged cache.

        ``latent`` is ``[B, 576]`` BF16 (512 NoPE + 64 RoPE), the output of the
        KV down-projection for the token being decoded. Quantisation matches
        FlashMLA's own ``quantize_k_cache``: per-128 tile amax / 448 rounded up
        to a power of two (ue8m0), RoPE left unquantised in BF16.

        The write targets the same slot every call, which is what a steady-state
        decode step costs; it is not a growing-context replay.
        """
        bs = self.page_block
        pos = self.context_len
        blk = self.block_table[:, pos // bs].long()
        slot = pos % bs
        nope, rope = latent[:, :512], latent[:, 512:]
        amax = nope.view(-1, 4, 128).abs().float().amax(-1) / 448.0
        scale = torch.pow(2.0, amax.clamp_min(1e-4).log2().ceil())      # ue8m0
        q = (nope.view(-1, 4, 128).float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        self._c_nope[blk, slot] = q.view(-1, 512)
        self._c_scale[blk, slot] = scale
        self._c_rope[blk, slot] = rope

    def set_selected_ids(self, ids: torch.Tensor) -> None:
        """Stage indexer output ``[B, topk]`` (logical positions) into the buffer."""
        self.abs_indices.copy_(ids.to(torch.int32).view(self.batch, 1, self.topk))

    # -- stages ------------------------------------------------------------
    @staticmethod
    def _map_fn(abs_flat: torch.Tensor, block_table: torch.Tensor, bs: int) -> torch.Tensor:
        blk = torch.gather(block_table, 1, torch.div(abs_flat, bs, rounding_mode="floor").long())
        return (blk * bs + (abs_flat % bs)).to(torch.int32)

    def map_indices(self) -> torch.Tensor:
        """Logical token positions -> the kernel's block_idx*block_size+offset.

        Same mapping as FlashMLA's ``abs_indices2indices_in_kvcache`` (verified
        equal in ``tests/test_dsa_block.py``), with two differences that matter
        for measurement: it is a pure on-device gather (the upstream helper
        builds its offset index on the CPU, which would put a host round-trip
        inside the measured block), and it is ``torch.compile``-fused into a
        single kernel. Unfused it is six elementwise kernels, which on a decode
        step is ~32 us of almost pure launch overhead -- an artefact of writing
        the mapping in eager PyTorch, not a real cost of the operation.
        """
        return self._mapper(
            self.abs_indices.view(self.batch, self.topk), self.block_table, self.page_block
        ).view(self.batch, 1, self.topk)

    def attend(self, indices_in_kvcache: torch.Tensor) -> torch.Tensor:
        out, _ = self.flash_mla.flash_mla_with_kvcache(
            q=self.q,
            k_cache=self.k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=D_V,
            tile_scheduler_metadata=self.sched_meta,
            softmax_scale=self.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices_in_kvcache,
        )
        return out

    def selected_attention(self) -> torch.Tensor:
        """Index mapping + sparse MLA, from already-selected ids."""
        return self.attend(self.map_indices())

    def teardown(self) -> None:
        for attr in ("k_cache", "q", "abs_indices", "block_table", "sched_meta", "_mapper"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
