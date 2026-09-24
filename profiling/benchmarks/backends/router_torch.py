"""Flat MoE-router backend: dense scoring against E candidate experts, then top-k.

    logits = hidden_states @ router_weights.T      # [B, D] @ [D, E] -> [B, E]
    scores, ids = topk(logits, k)

``E`` counts router *candidates*; no expert MLP is instantiated and no dispatch,
expert execution, or cross-GPU communication is measured.  This backend is a
generic flat router -- it is deliberately NOT named after any specific model's
router, which would additionally require the activation, correction biases,
expert-group selection and routing-weight normalisation of that model.

Logits are produced in the router's native dtype (bf16 by default, matching how
a bf16 model's gate projection emits scores) and top-k runs on them directly.
``--router-dtype fp32`` selects the higher-precision configuration; the dtype is
recorded as an explicit experimental field, never mixed silently.

``sorted=False`` is the default, matching the DSA backends so the two workloads'
selection stages are comparable. Ordering the k winners is not needed to
dispatch to experts, and on CUDA ``sorted=True`` appends a separate
``bitonicSortKVInPlace`` kernel that roughly doubles selection cost at small E.
The setting is recorded per row as ``topk_sorted``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import RouterConfig, Unsupported, make_generator, randn

_DTYPE_TAGS = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


class TorchRouterBackend:
    name = "torch_flat_router"
    commit = None
    fuses_score_and_select = False

    def __init__(self, dtype: torch.dtype = torch.bfloat16, topk_sorted: bool = False):
        self.dtype = dtype
        self.dtype_tag = _DTYPE_TAGS[dtype]
        self.topk_sorted = topk_sorted
        self.cfg: Optional[RouterConfig] = None

    @staticmethod
    def availability() -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        return True, ""

    def estimate_bytes(self, cfg: RouterConfig) -> int:
        """Live device bytes for this configuration; the router table dominates."""
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        weights = cfg.num_experts * cfg.dim * itemsize
        hidden_pool = cfg.token_pool * cfg.batch_size * cfg.dim * itemsize
        hidden = cfg.batch_size * cfg.dim * itemsize
        logits = cfg.batch_size * cfg.num_experts * itemsize
        topk_out = cfg.batch_size * cfg.topk * (8 + itemsize)   # int64 ids + values
        return weights + hidden_pool + hidden + logits + topk_out

    # -- lifecycle ----------------------------------------------------------
    def setup(self, cfg: RouterConfig, device: torch.device) -> None:
        if cfg.topk > cfg.num_experts:
            raise Unsupported(f"k={cfg.topk} > E={cfg.num_experts}")
        self.cfg = cfg
        self.device = device
        g = make_generator(cfg.seed, device)

        # Router table [E, D], scaled by D^-0.5 so logit magnitude is
        # independent of D and comparisons across dimensions stay meaningful.
        self.weights = randn(cfg.num_experts, cfg.dim, generator=g, device=device, dtype=self.dtype)
        self.weights.mul_(cfg.dim ** -0.5)
        # `weights.T` stays a view: cuBLAS takes the transpose as a layout flag,
        # so no second copy of the table is allocated.  At E = 2^20 and D = 4096
        # a materialised transpose would double an already 8 GiB allocation and
        # make the dry-run estimate wrong.
        self.weights_t = self.weights.T

        # Pool of distinct token vectors so trials do not re-route one vector.
        self.pool_size = max(cfg.token_pool, 1)
        self.hidden_pool = randn(
            self.pool_size, cfg.batch_size, cfg.dim, generator=g, device=device, dtype=self.dtype
        )
        self.hidden = torch.empty(cfg.batch_size, cfg.dim, device=device, dtype=self.dtype)
        self.logits = torch.empty(cfg.batch_size, cfg.num_experts, device=device, dtype=self.dtype)
        self.stage_query(0)

    def reset(self) -> None:
        self.stage_query(0)

    def stage_query(self, trial: int) -> None:
        """Stage trial ``trial``'s token vectors; always called outside timing."""
        self.hidden.copy_(self.hidden_pool[trial % self.pool_size])

    # -- stages -------------------------------------------------------------
    def score(self) -> torch.Tensor:
        torch.mm(self.hidden, self.weights_t, out=self.logits)
        return self.logits

    def select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(self.cfg.topk, logits.shape[-1])
        vals, ids = torch.topk(logits, k, dim=-1, sorted=self.topk_sorted)
        return vals, ids

    def combined(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.select(self.score())

    def teardown(self) -> None:
        for attr in ("weights_t", "weights", "hidden_pool", "hidden", "logits"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
