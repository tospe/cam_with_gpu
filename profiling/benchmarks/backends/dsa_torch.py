"""Portable PyTorch DSA indexer backend (BF16 keys, FP32 reduction).

Scoring semantics follow the DeepSeek-V3.2 lightning indexer exactly::

    score[b, j] = sum_h  w[b, h] * ReLU( dot(q[b, h, :], k[b, j, :]) )     j < ctx
                = -inf                                                    j >= ctx
    ids[b, :]   = topk(score[b, :], k)

This is an *unfused* implementation: the per-head ReLU forces the ``[N, H]``
intermediate to be materialised before the weighted reduction over heads.  It is
chunked over the candidate axis so that intermediate stays in cache rather than
streaming to HBM, which is what a competent PyTorch implementation would do.  It
is not an attempt to match a fused CUDA kernel, and results from it are labelled
as the reference backend, not as the optimised baseline.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import DSAConfig, Unsupported, make_generator, randn

# Chunk over candidates so the [chunk, H] fp32 intermediate stays resident.
DEFAULT_SCORE_CHUNK = 32768


class TorchDSABackend:
    name = "torch_bf16"
    dtype_tag = "bf16"
    commit = None
    fuses_score_and_select = False
    supports_incremental = True
    is_reference = True

    def __init__(self, score_chunk: int = DEFAULT_SCORE_CHUNK, topk_sorted: bool = False):
        self.score_chunk = score_chunk
        self.topk_sorted = topk_sorted
        self.cfg: Optional[DSAConfig] = None

    # -- availability -------------------------------------------------------
    @staticmethod
    def availability() -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        return True, ""

    @staticmethod
    def estimate_bytes(cfg: DSAConfig) -> int:
        """Live device bytes for this configuration (dry-run memory estimate)."""
        b, d, h = cfg.batch_size, cfg.dim, cfg.heads
        cap = cfg.capacity
        k_cache = b * cap * d * 2                      # bf16 keys
        q_pool = cfg.query_pool * b * h * d * 2        # bf16 query pool
        w_pool = cfg.query_pool * b * h * 4            # fp32 weight pool
        q_buf = b * h * d * 2
        w_buf = b * h * 4
        scores = b * cap * 4                           # fp32 scores
        chunk = min(DEFAULT_SCORE_CHUNK, cap)
        inter = b * chunk * h * 4 * 2                  # fp32 [chunk,H] + bf16 bmm out
        topk_out = b * cfg.topk * (8 + 4)              # int64 ids + fp32 values
        return k_cache + q_pool + w_pool + q_buf + w_buf + scores + inter + topk_out

    # -- lifecycle ----------------------------------------------------------
    def setup(self, cfg: DSAConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        g = make_generator(cfg.seed, device)
        cap = cfg.capacity

        # Persistent cache, preallocated through context_len + steps so no
        # allocation happens inside a timed incremental loop.
        self.k_cache = torch.empty(cfg.batch_size, cap, cfg.dim, device=device, dtype=torch.bfloat16)
        self.k_init = randn(
            cfg.batch_size, cfg.context_len, cfg.dim, generator=g, device=device, dtype=torch.bfloat16
        )
        # Keys appended during incremental replay, generated outside timing.
        self.k_appended = (
            randn(cfg.batch_size, cfg.steps, cfg.dim, generator=g, device=device, dtype=torch.bfloat16)
            if cfg.steps > 0
            else None
        )

        # Query/weight pools: a fresh query per trial rather than one query
        # searched repeatedly.  Pools are staged into fixed buffers outside the
        # timed region so the measured pointers stay constant (graph-safe).
        pool = max(cfg.query_pool, cfg.steps, 1)
        self.pool_size = pool
        self.q_pool = randn(pool, cfg.batch_size, cfg.heads, cfg.dim, generator=g, device=device, dtype=torch.bfloat16)
        self.w_pool = randn(pool, cfg.batch_size, cfg.heads, generator=g, device=device, dtype=torch.float32).abs()

        self.q = torch.empty(cfg.batch_size, cfg.heads, cfg.dim, device=device, dtype=torch.bfloat16)
        self.w = torch.empty(cfg.batch_size, cfg.heads, device=device, dtype=torch.float32)

        self.scores = torch.empty(cfg.batch_size, cap, device=device, dtype=torch.float32)
        self.context_len = cfg.context_len
        self.reset()

    def reset(self) -> None:
        """Restore cache contents and replay position; always called outside timing."""
        cfg = self.cfg
        self.k_cache[:, : cfg.context_len].copy_(self.k_init)
        if cfg.capacity > cfg.context_len:
            self.k_cache[:, cfg.context_len :].zero_()
        self.context_len = cfg.context_len
        self.scores.fill_(float("-inf"))
        self.stage_query(0)

    def stage_query(self, trial: int) -> None:
        """Copy trial ``trial``'s query/weights into the fixed input buffers."""
        i = trial % self.pool_size
        self.q.copy_(self.q_pool[i])
        self.w.copy_(self.w_pool[i])

    # -- stages -------------------------------------------------------------
    def score(self, context_len: Optional[int] = None) -> torch.Tensor:
        """Weighted-ReLU MQA scoring over the first ``context_len`` candidates."""
        n = self.context_len if context_len is None else context_len
        out = self.scores[:, :n]
        chunk = self.score_chunk if self.score_chunk > 0 else n
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # [B, n_chunk, D] @ [B, D, H] -> [B, n_chunk, H]
            dots = torch.bmm(self.k_cache[:, s:e], self.q.transpose(1, 2))
            torch.sum(torch.relu(dots.float()) * self.w.unsqueeze(1), dim=-1, out=out[:, s:e])
        return out

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.cfg.topk, scores.shape[-1])
        return torch.topk(scores, k, dim=-1, sorted=self.topk_sorted)[1]

    def combined(self) -> torch.Tensor:
        return self.select(self.score())

    # -- incremental decoding ----------------------------------------------
    def append(self, step: int) -> None:
        """Write one new indexer key per sequence at the current cache tail."""
        pos = self.context_len
        self.k_cache[:, pos].copy_(self.k_appended[:, step])
        self.context_len = pos + 1

    def decode_steps(self, steps: int) -> torch.Tensor:
        """``steps`` decode iterations: append one key, then score+select.

        Follows the reference convention that the token appended at step ``t``
        is visible to the query issued at step ``t`` (``end_pos`` in
        ``Indexer.forward`` includes the token just written).
        """
        ids = None
        for t in range(steps):
            self.q.copy_(self.q_pool[t % self.pool_size])
            self.w.copy_(self.w_pool[t % self.pool_size])
            self.append(t)
            ids = self.select(self.score())
        return ids

    def teardown(self) -> None:
        for attr in ("k_cache", "k_init", "k_appended", "q_pool", "w_pool", "q", "w", "scores"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
