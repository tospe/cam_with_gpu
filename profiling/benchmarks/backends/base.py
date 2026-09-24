"""Backend interfaces and workload configuration for the DSA and router benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSAConfig:
    """One DSA decoding configuration.

    ``context_len`` is the number of *initial* cached tokens per sequence.  Each
    of the ``batch_size`` sequences owns a distinct logical cache -- no cache is
    shared across the batch.  Storage is preallocated for
    ``context_len + steps`` tokens so incremental runs never allocate inside a
    timed region.
    """

    batch_size: int = 1
    context_len: int = 8192
    heads: int = 64
    dim: int = 128
    topk: int = 2048
    steps: int = 0
    seed: int = 0
    block_kv: int = 64
    query_pool: int = 256

    @property
    def capacity(self) -> int:
        return self.context_len + max(self.steps, 0)

    def describe(self) -> str:
        return (
            f"B={self.batch_size} N={self.context_len} H={self.heads} "
            f"D={self.dim} k={self.topk} steps={self.steps}"
        )


@dataclass(frozen=True)
class RouterConfig:
    """One MoE-router configuration.

    ``num_experts`` counts router candidates, not instantiated expert MLPs.
    ``batch_size`` is the number of tokens routed concurrently in one decode step.
    """

    batch_size: int = 1
    num_experts: int = 16
    dim: int = 4096
    topk: int = 8
    seed: int = 0
    token_pool: int = 256

    def describe(self) -> str:
        return f"B={self.batch_size} E={self.num_experts} D={self.dim} k={self.topk}"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class Unsupported(RuntimeError):
    """Raised when a backend cannot run a configuration. Recorded, never hidden."""


class DSABackend(Protocol):
    name: str
    dtype_tag: str
    commit: Optional[str]
    fuses_score_and_select: bool
    supports_incremental: bool

    def setup(self, cfg: DSAConfig, device: torch.device) -> None: ...
    def reset(self) -> None: ...
    def stage_query(self, trial: int) -> None: ...
    def score(self) -> torch.Tensor: ...
    def select(self, scores: torch.Tensor) -> torch.Tensor: ...
    def combined(self) -> torch.Tensor: ...
    def decode_steps(self, steps: int) -> torch.Tensor: ...
    def teardown(self) -> None: ...


# ---------------------------------------------------------------------------
# Deterministic synthetic inputs
# ---------------------------------------------------------------------------

def make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


INPUT_DISTRIBUTION = "standard_normal(0,1), torch.randn with a per-run seeded CUDA Generator"


def randn(
    *shape: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """All synthetic inputs go through here so the distribution is recorded once."""
    return torch.randn(*shape, generator=generator, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def dsa_reference_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    context_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """High-precision DSA indexer scores; the correctness source of truth.

    Mirrors ``Indexer.forward`` in DeepSeek-V3.2-Exp (``inference/model.py``) and
    ``ref_paged_mqa_logits`` in DeepGEMM (``tests/test_attention.py``)::

        score[b, j] = sum_h  w[b, h] * ReLU( dot(q[b, h, :], k[b, j, :]) )

    computed in float64/float32 with masked positions set to ``-inf``.

    Args:
        q: ``[B, H, D]`` queries.
        k: ``[B, N, D]`` indexer keys (the heads share one key per token -- MQA).
        w: ``[B, H]`` per-head indexer weights.
        context_lens: optional ``[B]`` valid lengths; positions ``>= len`` are
            masked to ``-inf``.
    """
    qf = q.to(torch.float64)
    kf = k.to(torch.float64)
    wf = w.to(torch.float64)
    # [B, H, D] @ [B, D, N] -> [B, H, N]
    dots = torch.bmm(qf, kf.transpose(1, 2))
    scores = (torch.relu(dots) * wf.unsqueeze(-1)).sum(dim=1)  # [B, N]
    if context_lens is not None:
        pos = torch.arange(k.shape[1], device=k.device)
        mask = pos.unsqueeze(0) >= context_lens.unsqueeze(1).to(pos.device)
        scores = scores.masked_fill(mask, float("-inf"))
    return scores


def router_reference_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``logits = hidden @ weight.T`` in float64; the router correctness reference."""
    return hidden.to(torch.float64) @ weight.to(torch.float64).T
