"""A full DeepSeek-V3.2 decoder layer at decode, so the search can be sized
against the whole layer rather than against the attention block alone.

Geometry is taken verbatim from ``deepseek-ai/DeepSeek-V3.2`` ``config.json``:
hidden 7168, 128 attention heads, q-LoRA rank 1536, KV-LoRA rank 512,
qk_nope 128 / qk_rope 64 / v_head 128, indexer 64 heads x 128 dim with
top-k 2048, MoE 256 routed experts + 1 shared, top-8, moe_intermediate 2048,
dense intermediate 18432 (the first 3 of 61 layers are dense, the other 58 MoE).

What is real here
  * every projection is a real GEMM at the model's true shapes;
  * the indexer query/weight projections feed the DeepGEMM indexer directly --
    the FP8 indexer query really is ``wq_b(q_a_norm)``, the head weights really
    are ``weights_proj(x)``;
  * the MLA query really is ``q_b_proj`` -> W_UK absorption -> RoPE concat;
  * attention is FlashMLA's FP8 sparse MLA decode over the selected 2048;
  * the MoE expert weights are the full 256-expert table (22.5 GiB in BF16), so
    expert reads hit real, scattered HBM addresses.

What is NOT modelled (and so is excluded from every number reported)
  * embedding, the LM head, and the other 60 layers -- this is ONE layer;
  * ``expert dispatch/permutation``: routing indices are held fixed during a
    timed region and each (token, expert) pair is executed as its own GEMV
    rather than through a fused grouped-GEMM MoE kernel. Every expert is the
    same shape, so this does not change the weight traffic, but a production
    MoE kernel would have lower launch overhead. Effective bandwidth is
    reported so the reader can see how close to the roofline the loop lands.
  * training-time paths, attention/MLP bias terms, tensor or expert parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .base import Unsupported, make_generator, randn


@dataclass(frozen=True)
class LayerConfig:
    """DeepSeek-V3.2 layer geometry. Defaults are the published config."""

    hidden: int = 7168
    n_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    index_heads: int = 64
    index_dim: int = 128
    index_topk: int = 2048
    dense_intermediate: int = 18432
    moe_intermediate: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    mlp: str = "moe"            # "moe" (58/61 layers) or "dense" (first 3)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope + self.qk_rope          # 192

    @property
    def mla_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope     # 576


# The 256-expert table is 22.5 GiB; build it once and share it across configs.
_EXPERT_CACHE: dict = {}


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), w, eps)


class DeepSeekV32Layer:
    """One decoder layer, decode step, B tokens."""

    name = "deepseek_v32_decoder_layer"

    def __init__(self, cfg: LayerConfig = LayerConfig()):
        self.cfg = cfg

    @staticmethod
    def availability() -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        return True, ""

    def estimate_bytes(self, batch: int) -> int:
        c = self.cfg
        h, e2 = c.hidden, 2
        w = 0
        w += h * c.q_lora_rank * e2                                   # q_a
        w += c.q_lora_rank * c.n_heads * c.qk_head_dim * e2           # q_b
        w += h * c.mla_dim * e2                                       # kv_a
        w += c.n_heads * c.qk_nope * c.kv_lora_rank * e2              # W_UK
        w += c.n_heads * c.kv_lora_rank * c.v_head * e2               # W_UV
        w += c.n_heads * c.v_head * h * e2                            # o_proj
        w += c.q_lora_rank * c.index_heads * c.index_dim * e2         # indexer wq_b
        w += h * c.index_dim * e2 + h * c.index_heads * 4             # wk, weights_proj
        if c.mlp == "moe":
            per = 3 * c.moe_intermediate * h * e2
            w += (c.n_routed_experts + c.n_shared_experts) * per
            w += h * c.n_routed_experts * e2                          # gate
        else:
            w += 3 * c.dense_intermediate * h * e2
        act = batch * (h + c.q_lora_rank + c.n_heads * c.qk_head_dim
                       + c.n_heads * c.mla_dim + c.moe_intermediate) * e2 * 4
        return w + act

    # -- setup -------------------------------------------------------------
    def setup(self, batch: int, seed: int, device: torch.device) -> None:
        c = self.cfg
        self.batch, self.device = batch, device
        g = make_generator(seed + 7000, device)
        h = c.hidden

        def W(*shape, scale=None, dtype=torch.bfloat16):
            t = torch.empty(*shape, device=device, dtype=dtype)
            t.normal_(0.0, (scale if scale else shape[-1] ** -0.5), generator=g)
            return t

        # --- attention projections ---
        self.w_q_a = W(c.q_lora_rank, h)
        self.n_q_a = torch.ones(c.q_lora_rank, device=device, dtype=torch.bfloat16)
        self.w_q_b = W(c.n_heads * c.qk_head_dim, c.q_lora_rank)
        self.w_kv_a = W(c.mla_dim, h)
        self.n_kv_a = torch.ones(c.kv_lora_rank, device=device, dtype=torch.bfloat16)
        self.w_uk = W(c.n_heads, c.qk_nope, c.kv_lora_rank)
        self.w_uv = W(c.n_heads, c.kv_lora_rank, c.v_head)
        self.w_o = W(h, c.n_heads * c.v_head)
        self.n_in = torch.ones(h, device=device, dtype=torch.bfloat16)
        self.n_post = torch.ones(h, device=device, dtype=torch.bfloat16)

        # --- indexer projections ---
        self.w_idx_q = W(c.index_heads * c.index_dim, c.q_lora_rank)
        self.w_idx_k = W(c.index_dim, h)
        self.n_idx_k = torch.ones(c.index_dim, device=device, dtype=torch.bfloat16)
        self.w_idx_wt = W(c.index_heads, h, dtype=torch.float32)

        # --- MLP ---
        if c.mlp == "moe":
            key = ("moe", c.n_routed_experts, c.moe_intermediate, h)
            if key not in _EXPERT_CACHE:
                n = c.n_routed_experts + c.n_shared_experts
                _EXPERT_CACHE.clear()
                _EXPERT_CACHE[key] = (
                    W(n, c.moe_intermediate, h),   # gate_proj
                    W(n, c.moe_intermediate, h),   # up_proj
                    W(n, h, c.moe_intermediate),   # down_proj
                )
            self.e_gate, self.e_up, self.e_down = _EXPERT_CACHE[key]
            # Router scoring is done in fp32 (as in the reference gate); keep a
            # persistent fp32 copy instead of casting 7.3 MB on every call.
            self.w_router = W(c.n_routed_experts, h, dtype=torch.float32)
            # topk_method 'noaux_tc': a learned per-expert bias is added to the
            # scores for *selection* only; the returned weights use the raw scores.
            self.e_score_bias = torch.zeros(c.n_routed_experts, device=device,
                                            dtype=torch.float32).normal_(0, 0.02, generator=g)
            # Fixed routing during timing: distinct, scattered experts per token
            # so the number of expert reads matches real top-k routing.
            stride = max(1, c.n_routed_experts // (batch * c.experts_per_tok + 1))
            self.routed = [
                [(i * c.experts_per_tok + j) * stride % c.n_routed_experts
                 for j in range(c.experts_per_tok)]
                for i in range(batch)
            ]
            self.shared_idx = c.n_routed_experts   # last slot is the shared expert
        else:
            self.d_gate = W(c.dense_intermediate, h)
            self.d_up = W(c.dense_intermediate, h)
            self.d_down = W(h, c.dense_intermediate)

        # --- activations / rotary tables ---
        self.x = randn(batch, h, generator=g, device=device, dtype=torch.bfloat16)
        half = c.qk_rope // 2
        ang = torch.arange(half, device=device, dtype=torch.float32)
        self.cos = torch.cos(ang).to(torch.bfloat16)
        self.sin = torch.sin(ang).to(torch.bfloat16)

    # -- stages ------------------------------------------------------------
    def _rope(self, t: torch.Tensor) -> torch.Tensor:
        a, b = t[..., ::2], t[..., 1::2]
        return torch.stack((a * self.cos - b * self.sin,
                            a * self.sin + b * self.cos), dim=-1).flatten(-2)

    # -- attention input path, split into its individual projections --------
    def input_norm(self) -> torch.Tensor:
        """Pre-attention RMSNorm over the 7168-dim hidden state."""
        self.xn = _rms_norm(self.x, self.n_in)
        return self.xn

    def q_down(self) -> torch.Tensor:
        """Query down-projection: 7168 -> 1536 (q-LoRA rank), then its RMSNorm.
        22.0 MiB of weights. Its output also feeds the indexer query projection."""
        self.q_a = _rms_norm(F.linear(self.xn, self.w_q_a), self.n_q_a)
        return self.q_a

    def q_up(self) -> torch.Tensor:
        """Query up-projection: 1536 -> 128 heads x 192 (128 nope + 64 rope).
        75.5 MiB of weights, the largest read on the input side."""
        c = self.cfg
        q = F.linear(self.q_a, self.w_q_b).view(self.batch, c.n_heads, c.qk_head_dim)
        self.q_nope, self.q_pe = q.split([c.qk_nope, c.qk_rope], dim=-1)
        return q

    def q_absorb(self) -> torch.Tensor:
        """W_UK absorption + query RoPE: [B,H,128] -> [B,H,512], concat rope -> 576.

        This is what makes MLA decode cheap: instead of up-projecting the cached
        latent to per-head keys, the up-projection is folded into the query, so
        attention runs directly against the 576-dim latent cache."""
        q_lat = torch.bmm(self.q_nope.transpose(0, 1), self.w_uk).transpose(0, 1)
        self.q_mla = torch.cat([q_lat, self._rope(self.q_pe)], dim=-1).unsqueeze(1).contiguous()
        return self.q_mla

    def kv_down(self) -> torch.Tensor:
        """KV down-projection: 7168 -> 512 latent + 64 rope, norm, RoPE.
        8.3 MiB of weights. Produces the token's cache entry."""
        c = self.cfg
        kv = F.linear(self.xn, self.w_kv_a)
        lat, k_pe = kv.split([c.kv_lora_rank, c.qk_rope], dim=-1)
        self.kv_new = torch.cat([_rms_norm(lat, self.n_kv_a), self._rope(k_pe)], dim=-1)
        return self.kv_new

    def attn_proj_in(self) -> torch.Tensor:
        """All of the above in order; kept so callers can time the block as one."""
        self.input_norm()
        self.q_down()
        self.q_up()
        self.q_absorb()
        self.kv_down()
        return self.q_mla

    # -- attention output path ---------------------------------------------
    def o_absorb(self, attn: torch.Tensor) -> torch.Tensor:
        """W_UV absorption: [B,H,512] latent output -> [B,H,128] per-head values.
        16.8 MiB of weights."""
        c = self.cfg
        a = attn.view(self.batch, c.n_heads, c.kv_lora_rank)
        self.v = torch.bmm(a.transpose(0, 1), self.w_uv).transpose(0, 1)
        return self.v

    def o_proj(self) -> torch.Tensor:
        """Output projection: 128 heads x 128 = 16384 -> 7168.
        234.9 MiB of weights, the single largest read in the MLA sub-layer."""
        self.attn_out = F.linear(self.v.reshape(self.batch, -1), self.w_o)
        return self.attn_out

    def attn_proj_out(self, attn: torch.Tensor) -> torch.Tensor:
        self.o_absorb(attn)
        return self.o_proj()

    def indexer_proj(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer query / key / head-weight projections, and FP8 quantisation."""
        c = self.cfg
        qi = F.linear(self.q_a, self.w_idx_q).view(self.batch, 1, c.index_heads, c.index_dim)
        ki = _rms_norm(F.linear(self.xn, self.w_idx_k), self.n_idx_k)
        w = F.linear(self.xn.float(), self.w_idx_wt) * (c.index_heads ** -0.5)
        self.idx_q_fp8 = qi.to(torch.float8_e4m3fn)
        self.idx_w = w.abs()
        self.idx_k = ki
        return self.idx_q_fp8, self.idx_w

    def attn_proj_out(self, attn: torch.Tensor) -> torch.Tensor:
        """W_UV absorption and the output projection."""
        c = self.cfg
        a = attn.view(self.batch, c.n_heads, c.kv_lora_rank)
        v = torch.bmm(a.transpose(0, 1), self.w_uv).transpose(0, 1)   # [B,H,v_head]
        self.attn_out = F.linear(v.reshape(self.batch, -1), self.w_o)
        return self.attn_out

    def post_norm(self) -> torch.Tensor:
        """Residual add + post-attention RMSNorm. Not part of either search."""
        self.hn = _rms_norm(self.x + self.attn_out, self.n_post)
        return self.hn

    def router_score(self) -> torch.Tensor:
        """MoE router scoring: hidden @ W_router.T over all 256 candidate experts,
        then the sigmoid used by DeepSeek-V3.2's gate.

        This is the router half of a *second* top-k search in the same layer --
        the same score-then-select shape as the DSA indexer, over experts
        instead of cached tokens.
        """
        self.route_scores = F.linear(self.hn.float(), self.w_router).sigmoid()
        return self.route_scores

    def router_topk(self) -> torch.Tensor:
        """Group-limited top-k selection over the 256 experts.

        Keeps the best ``topk_group`` of ``n_group`` groups (scored by their top-2
        experts), then takes the top ``experts_per_tok`` and renormalises -- the
        selection DeepSeek-V3.2 actually performs.
        """
        c = self.cfg
        biased = self.route_scores + self.e_score_bias        # selection scores
        grp = biased.view(self.batch, c.n_group, -1)
        top2 = grp.topk(2, dim=-1)[0].sum(-1)
        keep = top2.topk(c.topk_group, dim=-1)[1]
        mask = torch.zeros_like(top2).scatter_(1, keep, 1.0).unsqueeze(-1)
        gated = (grp * mask).view(self.batch, -1)
        ids = gated.topk(c.experts_per_tok, dim=-1)[1]
        # weights come from the RAW scores, renormalised, then scaled
        w = self.route_scores.gather(1, ids)
        w = w / w.sum(-1, keepdim=True) * c.routed_scaling_factor
        self.route_w = w.to(torch.bfloat16)
        return ids

    def mlp(self) -> torch.Tensor:
        """Expert (or dense) feed-forward, including the second residual add.

        Completes the sub-layer: ``out = h + FFN(RMSNorm(h))`` where
        ``h = x + MLA(RMSNorm(x))``.
        """
        c = self.cfg
        resid = self.x + self.attn_out          # h, the attention sub-layer output
        if c.mlp == "dense":
            g = F.linear(self.hn, self.d_gate)
            u = F.linear(self.hn, self.d_up)
            return resid + F.linear(F.silu(g) * u, self.d_down)
        out = torch.zeros_like(self.hn)
        for i in range(self.batch):
            hi = self.hn[i : i + 1]
            for j, e in enumerate(self.routed[i]):
                g = F.linear(hi, self.e_gate[e])
                u = F.linear(hi, self.e_up[e])
                out[i : i + 1] += F.linear(F.silu(g) * u, self.e_down[e]) * self.route_w[i, j]
            s = self.shared_idx
            g = F.linear(hi, self.e_gate[s])
            u = F.linear(hi, self.e_up[s])
            out[i : i + 1] += F.linear(F.silu(g) * u, self.e_down[s])
        return resid + out

    # -- traffic accounting -------------------------------------------------
    def mlp_weight_bytes(self, batch: Optional[int] = None) -> int:
        """Expert weight bytes read for one decode step of `batch` tokens."""
        c = self.cfg
        b = batch if batch is not None else getattr(self, "batch", 1)
        if c.mlp == "dense":
            return 3 * c.dense_intermediate * c.hidden * 2
        per = 3 * c.moe_intermediate * c.hidden * 2
        return (b * c.experts_per_tok + c.n_shared_experts) * per

    def attn_weight_bytes(self) -> int:
        c = self.cfg
        return 2 * (c.hidden * c.q_lora_rank
                    + c.q_lora_rank * c.n_heads * c.qk_head_dim
                    + c.hidden * c.mla_dim
                    + c.n_heads * c.qk_nope * c.kv_lora_rank
                    + c.n_heads * c.kv_lora_rank * c.v_head
                    + c.n_heads * c.v_head * c.hidden)


    # -- per-stage byte model, for roofline / efficiency analysis -----------
    def stage_bytes(self, stage: str, batch: int, context_len: int) -> Optional[int]:
        """Minimum bytes a stage must move, for an achieved-bandwidth estimate.

        Counts the compulsory traffic: weights read once, plus the activations
        or cache entries the stage must touch. It deliberately ignores
        re-reads, spills and intermediates, so the resulting bandwidth is an
        *upper bound on efficiency* -- a stage reported at 3% of roofline is
        doing no better than that, and may be doing worse.
        """
        c = self.cfg
        h, B, N = c.hidden, batch, context_len
        two = 2
        table = {
            "attn_proj_in": two * (h * c.q_lora_rank
                                   + c.q_lora_rank * c.n_heads * c.qk_head_dim
                                   + h * c.mla_dim
                                   + c.n_heads * c.qk_nope * c.kv_lora_rank),
            "input_norm": two * B * h * 2,
            "q_down": two * h * c.q_lora_rank,
            "q_up": two * c.q_lora_rank * c.n_heads * c.qk_head_dim,
            "q_absorb": two * c.n_heads * c.qk_nope * c.kv_lora_rank,
            "kv_down": two * h * c.mla_dim,
            "o_absorb": two * c.n_heads * c.kv_lora_rank * c.v_head,
            "o_proj": two * c.n_heads * c.v_head * h,
            "indexer_proj": two * (c.q_lora_rank * c.index_heads * c.index_dim
                                   + h * c.index_dim) + 4 * h * c.index_heads,
            # FP8 indexer key cache: head_dim bytes + 4 bytes of scale per token
            "indexer_score": B * N * (c.index_dim + 4),
            "indexer_select": B * N * 4,                 # fp32 logits read
            "index_map": B * c.index_topk * 4 * 2,       # ids in, ids out
            # FlashMLA V32_FP8Sparse latent cache: 656 B per selected token
            "sparse_mla": B * c.index_topk * 656,
            "attn_proj_out": two * (c.n_heads * c.kv_lora_rank * c.v_head
                                    + c.n_heads * c.v_head * h),
            "post_norm": two * B * h * 2,
            "router_score": 4 * c.n_routed_experts * h,  # fp32 router table
            "router_topk": B * c.n_routed_experts * 4,
            "mlp_experts": self.mlp_weight_bytes(batch),
        }
        return table.get(stage)

    def teardown(self) -> None:
        for a in ("e_score_bias", "w_q_a", "w_q_b", "w_kv_a", "w_uk", "w_uv", "w_o", "w_idx_q",
                  "w_idx_k", "w_idx_wt", "w_router", "d_gate", "d_up", "d_down",
                  "x", "q_mla", "kv_new", "idx_q_fp8", "idx_w", "idx_k"):
            if hasattr(self, a):
                delattr(self, a)
        torch.cuda.empty_cache()
