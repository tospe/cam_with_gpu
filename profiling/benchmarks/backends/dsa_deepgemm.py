"""Optimised DSA indexer backend built on DeepGEMM's paged MQA-logits kernel.

This is the decode path DeepSeek ships for the V3.2 lightning indexer on SM90:
``fp8_paged_mqa_logits`` fuses the FP8 MQA GEMM, the per-head ReLU, and the
weighted reduction over indexer heads into a single kernel that streams the
paged KV cache once.  Top-k selection is a separate kernel (``torch.topk``),
so scoring and selection are timed separately *and* combined.

Numerical configuration (differs from the bf16 reference -- recorded as an
explicit ``dtype`` field, never conflated):
  * queries are cast to ``float8_e4m3fn`` directly (no per-head scale), exactly
    as in DeepGEMM's ``test_paged_mqa_logits``;
  * indexer keys are stored FP8 e4m3 with one FP32 scale per token
    (``sf = amax/448``), packed into the kernel's
    ``[num_blocks, block_kv, 1, D + 4]`` uint8 cache layout;
  * accumulation and the head reduction are FP32; emitted logits are FP32.

Layout: each sequence owns a distinct set of cache blocks (no cache is shared
across the batch), addressed through a per-sequence ``block_table``.

Masking: ``clean_logits=True`` is not supported by the SM90 kernel (it asserts
``not clean_logits``), so the kernel emits a full ``[B, max_model_len]`` buffer
whose padding beyond the valid context holds unmasked scores against stale cache
blocks -- values of the same magnitude as real ones.  Selection therefore runs
over the valid prefix ``logits[:, :context_len]`` only.  That slice is a strided
view, not a copy or an extra kernel, and it is inside the measured ``select``
stage rather than hidden outside it.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .base import DSAConfig, Unsupported, make_generator, randn

FP8_MAX = 448.0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def quantize_kv_block_layout(x: torch.Tensor) -> torch.Tensor:
    """Pack bf16 keys ``[num_blocks, block_kv, 1, D]`` into the kernel's uint8 cache.

    Mirrors ``kv_cache_cast_to_fp8`` in DeepGEMM's ``tests/test_attention.py``:
    the first ``block_kv * D`` bytes hold e4m3 values, the trailing
    ``block_kv * 4`` bytes hold one FP32 scale per token.
    """
    num_blocks, block_kv, num_heads, head_dim = x.shape
    assert num_heads == 1
    amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = amax / FP8_MAX
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    packed = torch.empty(
        (num_blocks, block_kv * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    packed[:, : block_kv * head_dim] = x_scaled.view(num_blocks, block_kv * head_dim).view(torch.uint8)
    packed[:, block_kv * head_dim :] = sf.view(num_blocks, block_kv).view(torch.uint8)
    return packed.view(num_blocks, block_kv, num_heads, head_dim + 4)


class DeepGEMMDSABackend:
    name = "deepgemm_fp8_paged_mqa"
    dtype_tag = "fp8_e4m3_kv_fp8_q"
    fuses_score_and_select = False   # scoring is fused; top-k is a separate kernel
    supports_incremental = True
    is_reference = False

    def __init__(self, topk_sorted: bool = False):
        self.topk_sorted = topk_sorted
        self.cfg: Optional[DSAConfig] = None
        self.commit = _repo_commit()

    # -- availability -------------------------------------------------------
    @staticmethod
    def availability() -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        try:
            import deep_gemm  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, f"deep_gemm import failed: {type(exc).__name__}: {exc}"
        major = torch.cuda.get_device_properties(0).major
        if major not in (9, 10):
            return False, f"DeepGEMM requires SM90/SM100, found SM{major}0"
        return True, ""

    @staticmethod
    def estimate_bytes(cfg: DSAConfig) -> int:
        b, d, h = cfg.batch_size, cfg.dim, cfg.heads
        blocks_per_seq = _ceil_div(cfg.capacity, cfg.block_kv)
        total_blocks = blocks_per_seq * b
        kv = total_blocks * cfg.block_kv * (d + 4)          # packed uint8 cache
        kv_src = b * cfg.context_len * d * 2                # bf16 staging keys
        q_pool = cfg.query_pool * b * h * d                 # fp8 query pool
        w_pool = cfg.query_pool * b * h * 4
        q_buf = b * h * d
        w_buf = b * h * 4
        max_model_len = blocks_per_seq * cfg.block_kv
        logits = b * max_model_len * 4
        topk_out = b * cfg.topk * (8 + 4)
        block_table = b * blocks_per_seq * 4
        return kv + kv_src + q_pool + w_pool + q_buf + w_buf + logits + topk_out + block_table

    # -- lifecycle ----------------------------------------------------------
    def setup(self, cfg: DSAConfig, device: torch.device) -> None:
        import deep_gemm

        if cfg.dim != 128:
            raise Unsupported(f"SM90 paged MQA logits supports head_dim=128, got {cfg.dim}")
        if cfg.heads not in (32, 64):
            raise Unsupported(f"SM90 paged MQA logits supports 32 or 64 heads, got {cfg.heads}")
        if cfg.block_kv != 64:
            raise Unsupported(f"SM90 paged MQA logits supports block_kv=64, got {cfg.block_kv}")

        self.dg = deep_gemm
        self.cfg = cfg
        self.device = device
        g = make_generator(cfg.seed, device)

        b, h, d = cfg.batch_size, cfg.heads, cfg.dim
        self.blocks_per_seq = _ceil_div(cfg.capacity, cfg.block_kv)
        self.max_model_len = self.blocks_per_seq * cfg.block_kv
        total_blocks = self.blocks_per_seq * b

        # Distinct, shuffled block ranges per sequence: separate logical caches,
        # and a non-contiguous physical layout as in real paged serving.
        self.block_table = torch.zeros((b, self.blocks_per_seq), device=device, dtype=torch.int32)
        pool = torch.randperm(total_blocks, generator=g, device=device).to(torch.int32)
        for i in range(b):
            self.block_table[i] = pool[i * self.blocks_per_seq : (i + 1) * self.blocks_per_seq]

        # Initial cache contents, quantised once outside timing.
        kv_bf16 = torch.zeros((total_blocks, cfg.block_kv, 1, d), device=device, dtype=torch.bfloat16)
        init = randn(b, cfg.context_len, d, generator=g, device=device, dtype=torch.bfloat16)
        for i in range(b):
            blk = self.block_table[i, : _ceil_div(cfg.context_len, cfg.block_kv)].long()
            flat = torch.zeros(len(blk) * cfg.block_kv, d, device=device, dtype=torch.bfloat16)
            flat[: cfg.context_len] = init[i]
            kv_bf16[blk] = flat.view(len(blk), cfg.block_kv, 1, d)
        self.kv_init = quantize_kv_block_layout(kv_bf16)
        self.kv_cache = self.kv_init.clone()
        del kv_bf16, flat

        # Keys appended during incremental replay: pre-quantised outside timing
        # into the per-token (value, scale) form the cache stores.
        if cfg.steps > 0:
            app = randn(b, cfg.steps, d, generator=g, device=device, dtype=torch.bfloat16)
            amax = app.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
            sf = amax / FP8_MAX
            self.app_vals = (app * (1.0 / sf)).to(torch.float8_e4m3fn).view(torch.uint8)  # [B,steps,D]
            self.app_scales = sf.squeeze(-1).contiguous().view(torch.uint8).view(b, cfg.steps, 4)
        else:
            self.app_vals = self.app_scales = None

        # Query / weight pools -> fixed staging buffers (constant pointers).
        pool_n = max(cfg.query_pool, cfg.steps, 1)
        self.pool_size = pool_n
        q_bf16 = randn(pool_n, b, 1, h, d, generator=g, device=device, dtype=torch.bfloat16)
        self.q_pool = q_bf16.to(torch.float8_e4m3fn)
        del q_bf16
        self.w_pool = randn(pool_n, b, h, generator=g, device=device, dtype=torch.float32).abs()

        self.q = torch.empty(b, 1, h, d, device=device, dtype=torch.float8_e4m3fn)
        self.w = torch.empty(b, h, device=device, dtype=torch.float32)

        # 2-D context lengths ([B*next_n, 1]) with next_n = 1: one decode query
        # per sequence, each seeing its own valid prefix.
        self.context_lens = torch.full((b, 1), cfg.context_len, device=device, dtype=torch.int32)
        self.context_len = cfg.context_len
        self.reset()

    def reset(self) -> None:
        """Restore cache contents, valid lengths and replay position (outside timing)."""
        self.kv_cache.copy_(self.kv_init)
        self.context_lens.fill_(self.cfg.context_len)
        self.context_len = self.cfg.context_len
        self.stage_query(0)

    def stage_query(self, trial: int) -> None:
        i = trial % self.pool_size
        self.q.copy_(self.q_pool[i])
        self.w.copy_(self.w_pool[i])

    # -- stages -------------------------------------------------------------
    def _kernel_kwargs(self) -> dict:
        meta = self.dg.get_paged_mqa_logits_metadata(
            context_lens=self.context_lens,
            block_kv=self.cfg.block_kv,
            num_sms=self.dg.get_num_sms(),
        )
        return dict(
            q=self.q,
            kv_cache=self.kv_cache,
            weights=self.w,
            context_lens=self.context_lens,
            block_table=self.block_table,
            schedule_meta=meta,
            max_context_len=self.max_model_len,
            clean_logits=False,   # SM90 kernel asserts `not clean_logits`
        )

    def score(self) -> torch.Tensor:
        """Fused FP8 MQA GEMM + ReLU + weighted head reduction -> [B, max_model_len]."""
        return self.dg.fp8_paged_mqa_logits(**self._kernel_kwargs())

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        """Top-k over the valid prefix only; the padded tail is not masked."""
        k = min(self.cfg.topk, self.context_len)
        return torch.topk(scores[:, : self.context_len], k, dim=-1, sorted=self.topk_sorted)[1]

    def combined(self) -> torch.Tensor:
        return self.select(self.score())

    # -- incremental decoding ----------------------------------------------
    def append_key(self, k_bf16: torch.Tensor) -> None:
        """Quantise and write one freshly computed indexer key per sequence.

        ``k_bf16`` is ``[B, D]``, the output of the indexer key projection for
        the token being decoded. Quantised to FP8 e4m3 with one FP32 per-token
        scale, matching the cache layout, and written at the current tail.
        Used by the full-layer benchmark, where the key is computed live rather
        than pre-generated.
        """
        pos = self.context_len
        blk = self.block_table[:, pos // self.cfg.block_kv].long()
        slot = pos % self.cfg.block_kv
        d = self.cfg.dim
        sf = (k_bf16.abs().float().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-9)
        vals = (k_bf16 / sf).to(torch.float8_e4m3fn).view(torch.uint8)
        flat = self.kv_cache.view(self.kv_cache.shape[0], -1)
        off = slot * d
        flat[blk, off : off + d] = vals
        sc = self.cfg.block_kv * d + slot * 4
        flat[blk, sc : sc + 4] = sf.contiguous().view(torch.uint8).view(-1, 4)

    def append(self, step: int) -> None:
        """Write one pre-quantised indexer key per sequence at the cache tail."""
        pos = self.context_len
        blk = self.block_table[:, pos // self.cfg.block_kv].long()
        slot = pos % self.cfg.block_kv
        d = self.cfg.dim
        # Values occupy the first block_kv*D bytes; scales the trailing 4B/token.
        flat = self.kv_cache.view(self.kv_cache.shape[0], -1)
        val_off = slot * d
        flat[blk, val_off : val_off + d] = self.app_vals[:, step]
        sc_off = self.cfg.block_kv * d + slot * 4
        flat[blk, sc_off : sc_off + 4] = self.app_scales[:, step]
        self.context_lens.add_(1)
        self.context_len = pos + 1

    def decode_steps(self, steps: int) -> torch.Tensor:
        ids = None
        for t in range(steps):
            self.q.copy_(self.q_pool[t % self.pool_size])
            self.w.copy_(self.w_pool[t % self.pool_size])
            self.append(t)
            ids = self.select(self.score())
        return ids

    def teardown(self) -> None:
        for attr in (
            "kv_cache", "kv_init", "app_vals", "app_scales",
            "q_pool", "w_pool", "q", "w", "block_table", "context_lens",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()


def _repo_commit() -> Optional[str]:
    import subprocess

    root = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "DeepGEMM")
    try:
        out = subprocess.run(
            ["git", "-C", os.path.abspath(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None
