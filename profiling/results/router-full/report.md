# H100 baseline: how search latency scales with candidate count

**Runs:** `results/dsa-full/` (DSA), `results/router-full/` (router)
**Hardware:** NVIDIA H100 **PCIe** 80 GB — SM90, 114 SMs, 79.17 GiB usable,
ECC on, persistence on, MIG disabled, power cap 350 W (default 310 W),
PCIe Gen5 ×16. UUID `GPU-ad97a198-f167-e71a-a197-3b0908eb6195`, host
`mns00.cse.nd.edu`. The GPU was **idle and exclusive** for every run
(`compute_processes: []`); no clocks, power limits or other users' processes
were changed.
**Software:** driver 610.43.02, PyTorch 2.10.0+cu129, CUDA 12.9 JIT toolchain,
GCC 13.4, DeepGEMM `78b6900`, DeepSeek-V3.2-Exp `87e509a`.

This is a **baseline** study of existing GPU behaviour. It does not implement a
CAM, does not model one, and makes no claim about CAM speedup or on-chip
placement.

---

## 1. Headline findings

1. **On an H100 at decode, selecting the top-2048 costs several times more than
   computing the DSA scores.** With the optimised FP8 kernel at `B = 1`,
   scoring 8 192 candidates takes **7.7 µs** while `torch.topk` takes
   **54.6 µs** — a factor of **7.1×**. The ratio stays above 5× across the
   whole `N` range at `B = 1`.

   **Read this as a property of the stock selection kernel, not of the
   hardware.** `torch.topk` is the established GPU baseline, but §2 shows it
   picks a single-thread-block implementation for small batches and moderate
   `N`, leaving under 1 % of the GPU busy. A selection kernel tuned for this
   shape would narrow the gap; how far is not measured here.

2. **DSA scoring at `B = 1` barely grows with context length.** From
   `N = 8 192` to `N = 131 072` — a 16× increase in candidates — optimised
   scoring goes from 7.7 µs to 10.8 µs (**1.4×**). At small `N` the kernel is
   latency- and occupancy-bound, not data-bound: effective KV read bandwidth
   rises from 0.14 TB/s at `N = 8 192` to 1.60 TB/s at `N = 131 072`, i.e. it
   only approaches the HBM roofline at the largest context.

3. **Router scoring at large `E` is purely HBM-bound and dominates completely.**
   At `E = 2^20`, `D = 4096`, routing a single token takes **4.71 ms**, of
   which **4.63 ms is the scoring GEMV**. The 8.59 GB BF16 table is streamed at
   **1.86 TB/s** — essentially the H100 PCIe HBM2e roofline. Selection of
   `k = 8` over those 2^20 logits costs only 78 µs.

4. **There is a crossover, at the L2 boundary.** On a fine grid, router scoring
   overtakes selection at **E ≈ 7 000** — precisely where the BF16 table outgrows
   the 50 MiB L2 (E ≈ 6 400). The coarse sweep grid jumps 4 096 → 16 384 and so
   makes it *look* like 2^14. From `E = 65 536` onward scoring is ≥ 86 % of
   combined latency (86.6 %, 95.6 %, 98.3 % at E = 65 536, 262 144, 2^20).
   Below the L2 boundary there are two further regimes, not one — see §4a.

5. **Candidate data is re-read from HBM on every decode step**, and above the
   L2 working-set size that read is the cost. Below it, it is not: at
   `E = 4096` (34 MB table, H100 L2 is 50 MB) apparent bandwidth reaches
   **2.75 TB/s**, above the HBM peak — the table is L2-resident. This is a
   disclosed steady-state effect, not a flushed-cache measurement.

---

## 2. DSA indexer, fixed context

`H = 64`, `D = 128`, `k = 2048`, one decode query per sequence. Median GPU
latency (µs), `cudagraph` mode (GPU work with host launch cost removed):

### B = 1

| N | `deepgemm` score | `deepgemm` select | `deepgemm` combined | `torch_bf16` score | `torch_bf16` combined |
|---:|---:|---:|---:|---:|---:|
| 8 192 | 7.7 | 54.6 | 57.7 | 19.6 | 68.2 |
| 16 384 | 7.7 | 93.9 | 97.0 | 26.9 | 114.1 |
| 32 768 | 8.5 | 49.5 | 58.7 | 40.2 | 92.9 |
| 65 536 | 8.9 | 54.6 | 63.6 | 85.7 | 142.2 |
| 131 072 | 10.8 | 61.0 | 70.9 | 165.8 | 230.2 |

### B = 8

| N | `deepgemm` score | `deepgemm` select | `deepgemm` combined | `torch_bf16` score | `torch_bf16` combined |
|---:|---:|---:|---:|---:|---:|
| 8 192 | 9.7 | 57.9 | 65.8 | 80.5 | 137.0 |
| 16 384 | 11.6 | 102.0 | 110.1 | 182.0 | 279.9 |
| 32 768 | 14.8 | 55.5 | 70.1 | 365.1 | 421.0 |
| 65 536 | 51.8 | 64.1 | 118.7 | 740.9 | 803.9 |
| 131 072 | 92.0 | 78.2 | 170.4 | 1483.0 | 1556.0 |

`score` and `select` are measured independently; `combined` is its own
measurement. They are **not** additive and are never stacked (Figure 3).

**Where time grows.** For the optimised backend, scoring is nearly flat in `N`
at `B = 1` and only becomes the dominant term at `B = 8, N ≥ 65 536`, where
`B × N` finally provides enough work to saturate HBM. Selection is roughly
*independent of N* — it is dominated by `k = 2048`, not by the candidate count.

**Kernel fusion changes the picture, batching changes it more.** The BF16
reference must materialise the `[N, H]` post-ReLU intermediate; DeepGEMM fuses
GEMM + ReLU + weighted head reduction into one kernel and never writes it. At
`B = 8, N = 131 072` this is a **16.1×** difference in scoring latency
(1483 µs vs 92 µs) and a **9.1×** difference in the combined path. Part of this
is fusion and part is FP8 vs BF16 keys (half the bytes) — these are different
numerical configurations and the gap should not be read as fusion alone.

**Effective KV read bandwidth** (`B × N × 132 B` for FP8+scale, `cudagraph`):

| | N=8 192 | 16 384 | 32 768 | 65 536 | 131 072 |
|---|---:|---:|---:|---:|---:|
| B=1 | 0.14 | 0.28 | 0.51 | 0.97 | **1.60** TB/s |
| B=8 | 0.90 | 1.49 | *2.33* | 1.33 | **1.50** TB/s |

The 2.33 TB/s at `B = 8, N = 32 768` exceeds HBM peak: that 34.6 MB working set
fits in the 50 MB L2. At `B = 8, N ≥ 65 536` the working set exceeds L2 and the
rate settles at ~1.5 TB/s of real HBM traffic.

### The top-k non-monotonicity at N = 16 384, explained

Selection is non-monotonic: 54.6 µs at `N = 8 192`, **93.9 µs** at
`N = 16 384`, then back down to 49.5 µs at `N = 32 768`. This reproduces across
backends, batch sizes and graph modes. `scripts/investigate_topk_heuristic.py`
isolates the cause (raw data in `topk_heuristic_probe.csv`).

`torch.topk` on CUDA has **two implementations**, chosen by a hard-coded
heuristic `should_use_multiblock(num_slices, slice_size)` — where `num_slices`
is the number of rows (our batch size) and `slice_size` is the row length
(our `N`):

* **single-block** (`gatherTopK`) — **one thread block per row**, each block
  doing that row's entire radix select;
* **multi-block** (`mbtopk`) — each row split across many blocks with
  cooperative digit-count / cumulative-sum passes (the
  `computeBlockDigitCounts`, `computeDigitCumSum`,
  `computeBlockwiseWithinKCounts` kernels visible in the profile, 21 kernels
  per call).

The heuristic is a **step function**. Bisecting the switch point confirms the
tiers directly:

| rows (B) | switches to multi-block at |
|---:|---:|
| 1 – 20 | N ≥ 20 000 |
| 21 – 40 | N ≥ 10 000 |
| 41 – 80 | N ≥ 8 000 |
| 81 – ~200 | N ≥ 5 000 |
| larger | lower still |

At `B = 1` the threshold is `N ≥ 20 000`, so **`N = 16 384` sits at the very
top of the single-block ramp, just below the switch.** On that path cost grows
linearly with `N` — 23.2, 38.8, 65.2, 84.1 µs at `N` = 4 096, 8 192, 12 288,
16 384 — because a single block is walking the whole row. Past the threshold
the multi-block path takes over and is then almost flat: 64.6 µs at
`N = 20 000` through 71.0 µs at `N = 131 072`.

The decisive demonstration is that **adding work makes it faster**. At a fixed
`N = 16 384`, going from 20 rows to 21 crosses a tier boundary, dropping the
threshold from 20 000 to 10 000 and switching implementations:

| | latency | path |
|---|---:|---|
| B = 20, N = 16 384 | 95.6 µs | single-block |
| **B = 21, N = 16 384** | **71.6 µs** | multi-block |

Likewise, on the single-block path latency is nearly **flat in batch size**
(76 µs at `B = 1`, 93 µs at `B = 20`) — the signature of one block per row, with
rows running concurrently on separate SMs. At `B = 1` that is **one thread block
on a 114-SM GPU**: under 1 % of the machine is doing anything.

**This is a library heuristic mistuned for this shape, not a hardware limit.**

---

## 3. DSA incremental decoding

Start from `N` cached tokens, append one indexer key per sequence per step,
128 steps, whole loop timed (eager). Context **grows within each measured
loop** — these are not repeated samples at a fixed `N`.

| N (initial) | `deepgemm` B=1 | `torch_bf16` B=1 | `deepgemm` B=8 | `torch_bf16` B=8 |
|---:|---:|---:|---:|---:|
| 8 192 | 9.43 ms | 10.14 ms | 11.15 ms | 20.24 ms |
| 32 768 | 14.34 ms | 18.08 ms | 14.96 ms | 61.84 ms |
| 131 072 | 14.44 ms | 37.74 ms | 27.34 ms | 208.22 ms |

Run-to-run variation is very low (CV ≤ 0.02). Setup — allocation plus initial
cache population — is timed separately and is **not** included above: it ranges
from 0.32 ms to 5.60 ms and grows with `B × N`. Any claim about a complete
request would need to include it.

The optimised backend's loop is nearly flat in `N` at `B = 1` (14.2–14.4 ms for
`N` from 16 K to 131 K), confirming that per-step cost at `B = 1` is dominated
by fixed per-step work — selection and launch overhead — rather than by the
growing candidate set.

**A 128-step block replay is not text generation.** These are indexer
scoring + selection + cache-append steps on synthetic data, with no
projections, no attention, no MLP and no sampling.

---

## 4. Flat MoE router

`D = 4096` (synthetic), `k = 8`, BF16, `sorted=False`. Median GPU latency (µs),
`cudagraph`, `B = 1`. Run `router-full-v2`:

| E | score | select | combined | table size | score bandwidth |
|---:|---:|---:|---:|---:|---:|
| 16 | 7.7 | 11.2 | 18.0 | 128 KB | 0.02 TB/s |
| 256 | 8.1 | 12.7 | 20.0 | 2 MB | 0.26 TB/s |
| 1 024 | 8.2 | 17.7 | 25.2 | 8 MB | 1.03 TB/s |
| 4 096 | 12.2 | 28.4 | 39.7 | 34 MB | *2.75 TB/s* (L2-resident) |
| 16 384 | 76.8 | 72.8 | 147.1 | 134 MB | 1.75 TB/s |
| 65 536 | 289.2 | 42.7 | 334.0 | 537 MB | 1.86 TB/s |
| 262 144 | 1 167.9 | 53.5 | 1 221.8 | 2.15 GB | 1.84 TB/s |
| **1 048 576** | **4 627.4** | 77.8 | **4 705.5** | **8.59 GB** | **1.86 TB/s** |

`B = 8` is within a few percent of `B = 1` everywhere — the table read is shared
across the batch, so eight tokens route for almost the price of one
(4 825 µs vs 4 705 µs at `E = 2^20`, i.e. **8× the throughput for 2.6 % more
latency**). Routing is bandwidth-bound in the table, not in the tokens.

All 18 configurations ran; nothing hit OOM (8.59 GB against a 60 GiB budget and
79 GiB device). Peak allocated memory at `E = 2^20, B = 8` was 8.048 GiB against
an 8.016 GiB pre-run estimate — 0.4 % error.

**Scope.** This measures routing only: no dispatch, no expert-network
execution, no cross-GPU communication, no accuracy at any expert count. A
model-specific router would additionally need its activation, correction
biases, expert-group selection and routing-weight normalisation, and is not
what was measured.

---

## 4a. Where router scoring takes over, and why

Three regimes, not one. The fine grid is in `router_crossover_probe.csv`
(`scripts/investigate_router_crossover.py`); figure
`fig6_router_crossover.png`. All values B = 1, `sorted=False`.

| E | table | x L2 | score | select | read rate | larger, and why |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 0.1 MB | 0.00 | 7.8 us | 6.0 us | - | **score** - 2 kernels vs 1 |
| 1 024 | 8 MB | 0.16 | 9.7 us | 12.4 us | 0.86 TB/s | **select** |
| 4 096 | 32 MB | 0.64 | 11.7 us | 22.5 us | 2.87 TB/s | **select** - widest gap |
| 6 144 | 48 MB | 0.96 | 31.3 us | 32.8 us | 1.61 TB/s | cliff |
| 8 192 | 64 MB | 1.28 | 39.3 us | 35.8 us | 1.71 TB/s | **crossover** |
| 65 536 | 512 MB | 10.2 | 285.5 us | 46.6 us | 1.88 TB/s | **score** |
| 1 048 576 | 8 192 MB | 163.8 | 4 617.5 us | 92.8 us | 1.86 TB/s | **score**, 98 % |

**Regime 1, E <= ~256 - scoring is the *larger* of the two.** One trivial CUDA
kernel costs 4.65 us on this machine, and at these sizes both stages are purely
kernel dispatch. Scoring is two kernels because cuBLAS split-Ks the D = 4096
reduction (`nvjet...splitK_TNT` 4.32 us + `splitKreduce_kernel` 1.64 us);
selection is one `sbtopk::gatherTopK` at 4.56 us. Two kernels beat one.

**Regime 2, ~512 to ~6 400 - selection is larger, for two unrelated reasons.**
Scoring is flat (9.7 -> 11.7 us across 4x in E) because the table is still
L2-resident, so the GEMV is latency-bound, not bandwidth-bound; the apparent
read rate of 2.87 TB/s exceeds HBM peak precisely because most of it never
reaches HBM. Selection grows linearly because below E = 20 000 at B = 1
`torch.topk` puts a single thread block on the whole row (12.4, 22.5, 68.5 us at
E = 1 024, 4 096, 16 384). A cached numerator meeting a single-block
denominator: both sides of that gap are soft.

**Regime 3, past ~6 400 - scoring goes bandwidth-bound.** At the cliff a 1.5x
table growth costs 2.7x the time. Thereafter the read rate locks at
1.8-1.88 TB/s, so score = table bytes / 1.86 TB/s, linear over three decades.
Selection switches to the multi-block kernel at E >= 20 000 and *drops* from
68.5 to 45.8 us before creeping to 92.8 us at E = 2^20.

It cannot invert at any batch size: the GEMV does 2*E*D flops on E*D*2 bytes,
**1 FLOP/byte**, against a machine balance of roughly **277 FLOP/byte** here
(516 TFLOP/s measured BF16 / 1.86 TB/s). The router table has no reuse. B = 8
reaches 8 FLOP/byte, still ~35x short - which is why B = 8 costs 2.6 % more than
B = 1 rather than 8x.

### A measurement error this corrected

The first router sweep (`router-full`) ran with `topk_sorted=True` while the DSA
backends used `sorted=False`, so the two workloads' selection stages were not
comparable, and `topk_sorted` was not recorded in the result schema at all. On
CUDA, `sorted=True` appends a separate `bitonicSortKVInPlace` kernel (4.72 us at
E = 16) that roughly doubles selection cost at small E. That single setting was
the whole reason selection appeared to exceed scoring at tiny E.

Fixed: `sorted=False` is now the default for both workloads, a proper
`--topk-sorted / --no-topk-sorted` flag exists, and `topk_sorted` is a column in
every raw and summary row. `router-full-v2` supersedes `router-full`;
`dsa-full-v2` re-runs DSA under the new schema and agrees with `dsa-full` to
0.7 % median (DSA already used `sorted=False`, so it was unaffected). Both v2
runs, and the fine-grid probe, were executed **serially on an otherwise idle
GPU** - an earlier attempt overlapped two jobs and depressed effective bandwidth
from 1.86 to 1.50 TB/s, and was discarded.

---

## 5. Where the time actually goes: profiler evidence

CUPTI (`torch.profiler`), `N = 32 768`, `B = 1`, 20 profiled iterations,
separate from all timing runs:

| case | kernels / iter | GPU kernel time / iter | inter-kernel gap share of span |
|---|---:|---:|---:|
| `deepgemm` score | 2 | 6.5 µs | 77 % |
| `deepgemm` select | 21 | 43.7 µs | 56 % |
| `deepgemm` combined | 23 | 50.7 µs | 57 % |
| `torch_bf16` score | 5 | 37.1 µs | 56 % |
| router score, E=2^20 | 2 | 4 621.3 µs | **0.08 %** |
| router select, E=2^20 | 16 | 69.2 µs | 26 % |

Two things follow:

* **Top-k is genuinely 21 kernels of real GPU work.** `torch.topk` at `k = 2048`
  runs a multi-block radix select (`mbtopk::computeBlockDigitCounts`,
  `computeDigitCumSum`, `computeBlockwiseWithinKCounts`, …). Its ~44 µs is GPU
  work, not launch overhead — which is exactly why CUDA-graph replay barely
  changes it (54.6 µs graph vs 59.2 µs eager at `N = 8 192`).
* **DSA scoring at `B = 1` is dominated by host submission, not GPU work.** Only
  6.5 µs of GPU kernel time per iteration, with 77 % of the span being gaps
  between kernels. CUDA-graph replay removes 7.9–12.2 µs from the eager DSA
  scoring times, which agrees with the trace.

The router GEMV at `E = 2^20` is the opposite extreme: 0.08 % gap, one
`nvjet_tst_512x8_64x3_2x1_v_bz_TNT` cuBLAS kernel doing 4.6 ms of HBM-bound
work. There is nothing to recover from scheduling there.

**This does not say that query-response time is "additional synchronisation
overhead."** The gap fractions above are host-submission gaps in an eager
launch sequence, measured against graph replay. A CAM study would have to
define and measure dependency tracking, data visibility, transport and search
computation separately; none of those are measured here.

---

## 6. What these measurements support, and what they do not

**Supported by the data:**

* Top-k selection, **as implemented by `torch.topk`**, not similarity
  computation, is the dominant cost of DSA search at decode on this GPU for
  `B = 1` at every context length tested, and up to `N = 32 768` at `B = 8`.
  A substantial part of that cost is a mistuned kernel-selection heuristic
  (§2), so this bounds the stock software stack, not the hardware.
* DSA indexer scoring scales far better than linearly with `N` at low batch
  size because it starts far from the memory roofline.
* Flat-router latency grows linearly with `E` once the table exceeds L2, at a
  rate set by HBM bandwidth, reaching 4.7 ms per decode step at `E = 2^20`.
* Candidate data is re-read from HBM every step once the working set exceeds
  L2; below that, L2 residency measurably changes the picture.
* Batching amortises the candidate-table read almost perfectly for the router
  and substantially for DSA.

**Not supported, and not claimed:**

* No statement about **full-model inference latency**. Nothing here runs a
  model; DSA-block numbers are not LLM-latency fractions.
* No **accuracy** result of any kind. All inputs are independent random
  vectors, which do not reproduce learned temporal correlation. The FP8-vs-
  float64 top-256 selection overlap of 1.000 reported by the test suite is a
  numerical-sensitivity diagnostic only.
* No claim that a CAM would be faster, or that on-chip placement helps.
* **No claim that the measured top-k cost is a hardware floor.** It is the cost
  of the stock `torch.topk`; an optimised selection kernel was not built or
  benchmarked, so the remaining headroom is unquantified.
* No transfer of these numbers to H100 SXM (different HBM bandwidth) or to
  Blackwell. In particular, the cited top-k paper (arXiv 2604.22312) evaluates
  Blackwell; its timings and speedups are **not** applicable here and its
  kernels were not run.

---

## 7. Unresolved limitation: selected-attention integration

**Milestone B is incomplete in one respect.** Experiment mode 3 —
connecting selected ids into the actual supported sparse-attention/MLA
implementation and measuring the combined block — is **not implemented**. The
DSA results above cover indexer scoring and top-k selection only.

Reason: this requires FlashMLA's sparse-attention kernels, which were not built
or validated in this environment, and substituting a generic gather or ordinary
attention would produce something that is not a DSA block. Per the brief, no
substitute was used.

Remaining steps to close it:

1. Clone and pin `deepseek-ai/FlashMLA`; confirm its SM90 sparse-MLA kernel
   supports this CUDA 12.9 / PyTorch 2.10 stack and build it with the same
   `scripts/env.sh` toolchain.
2. Add `benchmarks/backends/mla_flashmla.py` holding the paged **latent** KV
   cache (`kv_lora_rank = 512`, `qk_rope_head_dim = 64`) in FlashMLA's required
   representation and quantisation, distinct from the indexer key cache.
3. Feed `selected_ids` from the existing DSA backends straight into that
   kernel, with no intervening host round-trip, and add a `dsa_block` mode to
   `bench_dsa.py` measuring indexer + selection + selected attention as one
   block, plus each stage separately.
4. State explicitly that query/key projections are omitted and that prepared
   projections are the block inputs.
5. Validate selected-attention output against a float64 gather-and-attend
   reference on small inputs before recording any timing.

Also outstanding: `nsys` and a working `ncu` (see README), and replay of real
prepared queries and keys, which needs a checkpoint or captured tensors that
were not available.

---

## 8. Reproducing

```bash
source scripts/env.sh
$PYTHON -m pytest tests/test_reference_correctness.py -q      # 24 passed
$PYTHON benchmarks/bench_dsa.py    --config configs/full.yaml --run-id dsa-full-repro
$PYTHON benchmarks/bench_router.py --config configs/full.yaml --run-id router-full-repro
$PYTHON scripts/plot_results.py results/dsa-full-repro results/router-full-repro
$PYTHON scripts/profile_representative.py results/dsa-full-repro
```

Exact commands and fully resolved configurations for the runs reported here are
in each run's `metadata.json`.

### Figures

| File | Contents |
|---|---|
| `dsa-full/figures/fig1_dsa_latency_vs_context.png` | DSA score/select/combined vs `N`, panels per batch size × graph mode |
| `dsa-full/figures/fig3_dsa_block_and_breakdown.png` | Combined DSA search latency, and stages grouped side by side (not stacked) |
| `dsa-full/figures/fig5_dsa_incremental_loop.png` | 128-step incremental decode loop vs initial `N` |
| `router-full/figures/fig2_router_latency_vs_experts.png` | Router score/select/combined vs `E`, log candidate axis |
| `router-full/figures/fig4_throughput_vs_candidates.png` | Throughput (`B` / batch latency) for `B = 1` and `B = 8` |

All markers are measured medians; bands are p5–p95 from the raw samples; lines
are visual guides only.
