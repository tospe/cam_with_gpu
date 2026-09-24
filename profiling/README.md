# H100 search and top-k benchmarks

Reproducible native-GPU measurements of how **search latency scales with the
number of candidates**, for two decode-time workloads:

1. **DeepSeek Sparse Attention (DSA)** — the lightning indexer scores every
   cached token and selects the top `k = 2048`, as the number of cached tokens
   `N` grows from 8 192 to 131 072.
2. **Mixture-of-experts routing** — a flat router scores `E` candidate experts
   and selects the top `k = 8`, as `E` grows from 16 to 1 048 576 (2^20).

These runs establish the **existing GPU baseline**. They do not implement a CAM,
modify Accel-Sim, or demonstrate any CAM speedup or on-chip-placement advantage.

Measured on one **NVIDIA H100 PCIe 80 GB** (SM90, 114 SMs, HBM2e).
Note this is the **PCIe** form factor, not SXM: peak HBM bandwidth is ≈2.0 TB/s
rather than SXM's ≈3.35 TB/s. Do not transfer these numbers to an SXM H100.

---

## Results at a glance

| | |
|---|---|
| Full DSA sweep | [`results/dsa-full-v2/`](results/dsa-full-v2/) |
| Full router sweep | [`results/router-full-v2/`](results/router-full-v2/) |
| Written analysis | [`results/dsa-full-v2/report.md`](results/dsa-full-v2/report.md) |
| Figures | [`results/dsa-full-v2/figures/`](results/dsa-full-v2/figures/), [`results/router-full-v2/figures/`](results/router-full-v2/figures/) |
| Profiler evidence | [`results/dsa-full/profiles/`](results/dsa-full/profiles/) |
| Superseded | `results/dsa-full/`, `results/router-full/` — the router run there used `topk_sorted=true`; see report §4a |

---

## Install

The machine has no CUDA toolkit and only GCC 8.5, which cannot build DeepGEMM
(it needs C++20 `<format>`, i.e. GCC 13+, and CUDA Toolkit ≥ 12.9). Everything
is provisioned locally, without root, into `third_party/toolchain`.

```bash
cd /localhome/tsousape/cam_with_gpu
bash scripts/setup_env.sh          # venv + pinned deps + GCC 13 / CUDA 12.9 + DeepGEMM
source scripts/env.sh              # must be sourced before every run
```

`scripts/env.sh` puts `nvcc` and GCC 13 on `PATH` and sets
`NVCC_PREPEND_FLAGS=-ccbin <gcc13 g++>`. DeepGEMM JIT-compiles its kernels at
**run** time, so this is required for running, not just for installing.

Pinned versions (see `requirements.txt` and each run's `metadata.json`):

| Component | Version / commit |
|---|---|
| PyTorch | `2.10.0+cu129` |
| CUDA toolkit (JIT) | 12.9 (`nvcc` 12.9.86), driver 610.43.02 |
| Host compiler | GCC 13.4.0 (conda-forge) |
| DeepGEMM | [`78b6900`](https://github.com/deepseek-ai/DeepGEMM) + `third_party/patches/deepgemm-cuda_fp8-include.patch` |
| DeepSeek-V3.2-Exp | [`87e509a`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) (scoring semantics reference) |

The one-line patch adds a missing `#include <cuda_fp8.h>` to a **Blackwell-only**
DeepGEMM header that does not compile under CUDA 12.9. It does not touch any
kernel used here.

## Smoke test

```bash
source scripts/env.sh
$PYTHON -m pytest tests/test_reference_correctness.py -q        # 24 tests, ~2 s
$PYTHON benchmarks/bench_dsa.py    --config configs/quick.yaml --dry-run
$PYTHON benchmarks/bench_router.py --config configs/full.yaml  --dry-run
```

`--dry-run` validates every configuration and estimates live device memory
without allocating the sweep.

## Quick experiment (minutes)

```bash
source scripts/env.sh
$PYTHON benchmarks/bench_dsa.py    --config configs/quick.yaml --run-id dsa-quick
$PYTHON benchmarks/bench_router.py --config configs/quick.yaml --run-id router-quick
$PYTHON scripts/plot_results.py results/dsa-quick results/router-quick
```

## Full sweep (the results in this repository)

```bash
source scripts/env.sh
$PYTHON benchmarks/bench_dsa.py    --config configs/full.yaml --run-id dsa-full \
    --note "Milestone B main DSA sweep: fixed-context + incremental, eager and CUDA-graph modes"
$PYTHON benchmarks/bench_router.py --config configs/full.yaml --run-id router-full \
    --note "Milestone B full router sweep: E = 16 .. 2^20, eager and CUDA-graph"
$PYTHON scripts/plot_results.py results/dsa-full results/router-full
$PYTHON scripts/profile_representative.py results/dsa-full --context-len 32768 --num-experts 1048576
```

Run directories are immutable: re-running with an existing `--run-id` fails
rather than overwriting.

---

## What is measured, and what is not

### DSA

Scoring semantics are taken from the DeepSeek-V3.2 lightning indexer
(`Indexer.forward` in DeepSeek-V3.2-Exp `inference/model.py`, and
`ref_paged_mqa_logits` in DeepGEMM `tests/test_attention.py`):

```
score[b, j] = sum_h  w[b, h] * ReLU( dot(q[b, h, :], k[b, j, :]) )     j < ctx[b]
ids[b, :]   = topk(score[b, :], k)
```

with `H = 64` indexer heads, `D = 128`, `k = 2048`. The indexer heads share one
key per cached token (MQA). This is **not** cosine similarity and **not** a
single-head dot product; `tests/test_reference_correctness.py::test_dsa_relu_actually_applied`
pins the ReLU down explicitly.

Two modes:

* **`fixed`** — a cache of `N` candidates is prepared once, then one decode
  query per sequence is scored and selected. Isolates scaling with `N`.
* **`incremental`** — starts at `N` cached tokens and appends one indexer key
  per sequence per step for 128 steps, measuring cache update **and** search.
  Context grows inside each measured loop, so it is reported as a whole loop,
  never as repeated samples at one fixed `N`.

Batch size `B` means `B` **distinct sequences with distinct logical caches**; no
cache is shared across the batch. Storage is preallocated through `N + steps`,
and cache contents, valid lengths and replay position are reset outside the
timed region before every trial, including after warmup.

### Router

```
logits = hidden_states @ router_weights.T       # [B, D] @ [D, E]
scores, ids = topk(logits, k)
```

`E` counts **router candidates**, not instantiated expert MLPs. `D = 4096` is a
clearly labelled **synthetic default**, not a value taken from a specific model.
This is a generic flat router, deliberately not named after any model's router:
a model-specific router would additionally need its activation, correction
biases, expert-group selection and routing-weight normalisation, and would be
registered as a separate backend and validated against that model.

**The large-`E` runs measure routing scalability only.** They do not measure
dispatch, expert-network execution, cross-GPU communication, or model accuracy
at those expert counts.

### Scope limits that apply throughout

* DSA-block fractions are **not** full-LLM fractions. Nothing here runs a full
  model, so no result is a full-model inference latency.
* A synthetic block replay is not text generation.
* All inputs are random, so **no result here is an accuracy result.** Selection
  overlap between numerical configurations is reported as a numerical
  sensitivity diagnostic only.
* Contexts are within DeepSeek-V3.2's supported range at `N ≤ 131 072`.

---

## Backends

| Backend | dtype | Role |
|---|---|---|
| `torch_bf16` | BF16 keys, FP32 reduction | Portable **reference**. Unfused: the per-head ReLU forces the `[N, H]` intermediate to be materialised. Chunked over candidates (default 32 768) so that intermediate stays cached. |
| `deepgemm_fp8_paged_mqa` | FP8 e4m3 keys (+FP32 per-token scale), FP8 queries, FP32 accumulation | **Optimised baseline**: DeepGEMM's `fp8_paged_mqa_logits`, the SM90 decode kernel DeepSeek ships for the V3.2 indexer. Fuses the MQA GEMM, per-head ReLU and weighted head reduction into one kernel. |
| `torch_flat_router` | BF16 (or `--router-dtype fp32`) | Flat router GEMM + `torch.topk`. |

**BF16 reference and FP8 optimised paths are different numerical
configurations** and are always reported as such (the `dtype` column). They are
not interchangeable and their latencies are not two measurements of one thing.

Two backend-specific facts worth knowing:

* DeepGEMM's SM90 kernel **fuses scoring but not selection**, so `score`,
  `select` and `combined` are all separately measurable. No fusion is disabled
  to produce the stage timings.
* `clean_logits=True` is **rejected** by the SM90 kernel, so its output buffer's
  padding beyond the valid context holds unmasked scores of the same magnitude
  as real ones. Selection therefore runs over `logits[:, :context_len]`. That
  slice is a strided view, not a kernel, and it sits inside the measured
  `select` stage rather than being hidden outside it.

---

## Timing protocol

1. Inputs, outputs and persistent state are allocated before any timed trial.
   DeepGEMM JIT compilation and cuBLAS autotuning are triggered during warmup.
2. 20 warmup iterations, 100 measured iterations for operator runs (20 for the
   128-step incremental loop). Actual counts are recorded per row.
3. GPU elapsed time comes from `torch.cuda.Event` pairs on the execution stream.
   Short kernels are auto-grouped (one event pair around *g* iterations,
   divided by *g*); the method is recorded in `timing_method`, so grouped and
   per-iteration samples are never silently mixed.
4. Host-observed latency (`--measure-host`) is recorded separately from GPU
   elapsed time and never substituted for it.
5. Queries vary across trials: a pool of distinct queries is staged into fixed
   buffers **outside** the timed region, so pointers stay constant (graph-safe)
   while inputs change. The isolated `select` stage likewise cycles a pool of
   score vectors rather than re-selecting over one vector.
6. `score`, `select` and `combined` are each measured independently. **The
   separately measured stages need not sum to `combined`** — cache state and
   overlap differ — so figures group them side by side and never stack them
   into a claimed end-to-end total.
7. Both backends are measured in the same two `graph_mode`s — `eager` and
   `cudagraph`. Graph replay removes per-kernel host launch cost, which is how
   GPU work is separated from host submission gaps rather than assuming every
   gap is GPU work.
8. Setup (allocation + initial cache population) is timed independently and
   reported in its own `setup_us` column, never folded into steady state.
9. Reported statistics: median, p95, p99, stdev, IQR and coefficient of
   variation. For batch `B`, `throughput_qps` is `B / batch_latency`. Batch
   latency divided by `B` would be amortised cost, not the latency a query
   experiences, and is not reported as latency.
10. Profiling runs are separate from timing runs and are never used as latency
    samples.

**Steady-state residency is disclosed, not engineered away.** Tables and caches
stay resident across trials, which is a legitimate serving scenario. Caches are
not flushed artificially. At small `E` this shows up directly as apparent
bandwidth above HBM peak (L2 residency) — see the report.

### Profiler availability on this machine

`nsys` is **not available**: Nsight Systems is not distributed through the
configured conda channels, and NVIDIA's direct download requires an
authenticated session. The only Nsight Compute build compatible with CUDA 12.9
(2024.1.1) **segfaults** against driver 610.43.02; the 2026.2 build requires
CUDA 13 and conflicts with this stack.

Kernel-level evidence therefore comes from **CUPTI via `torch.profiler`**
(`scripts/profile_representative.py`): per-kernel GPU durations, kernel launch
counts, an exported Chrome/Perfetto trace per case, and a derived
kernel-time-vs-span gap analysis. This is recorded in
`results/dsa-full/profiles/profiler_availability.json`. The eager-vs-CUDA-graph
comparison provides the launch-gap attribution an `nsys` timeline would show.

---

## Result schema

`results/<run_id>/` contains:

| File | Contents |
|---|---|
| `metadata.json` | environment, resolved config, exact command, input distribution + seed, repo commits — once per run |
| `measurements.csv` | one row per timing sample |
| `summary.csv` | one row per (configuration, stage), including non-`ok` outcomes |
| `figures/` | PNG + PDF |
| `profiles/` | CUPTI traces, per-kernel tables, tool availability |
| `report.md` | written analysis |

Every row carries `run_id, workload, mode, backend, backend_commit, dtype,
batch_size, candidate_count, dimension, indexer_heads, k, seed, trial,
decode_step, stage, graph_mode, timing_method, elapsed_us, status, error`, plus
`context_len_actual, block_kv, steps, group_size, warmup, config_id`.
Summaries add sample count, median, p95, p99, stdev, IQR, CV, host median,
throughput, peak allocated/reserved bytes and setup time.

Inapplicable fields are written **empty (null), never 0**. Failures,
`over_budget`, `unsupported`, `oom` and `graph_unsupported` cases are preserved
as rows — `E`, `D`, precision and batch size are never silently reduced to make
a configuration fit.

## CLI

Both drivers accept `--device --dtype/--backend --seed --context-lens/--num-experts
--batch-sizes --heads --dim --topk --steps --block-kv --warmup --iters
--query-pool/--token-pool --graph-modes --measure-host --memory-budget-gib
--out-dir --run-id --dry-run --note`, plus `--topk-sorted / --no-topk-sorted`
(default off, recorded per row), and `--config` to supply defaults from YAML.
Command-line flags override the config file.

Selection uses `sorted=False` for **both** workloads so their select stages are
comparable: ordering the k winners is not needed to dispatch to experts or to
gather selected KV, and on CUDA `sorted=True` appends a separate
`bitonicSortKVInPlace` kernel that roughly doubles selection cost at small
candidate counts.

## Layout

```
requirements.txt          pinned dependencies
configs/{quick,full}.yaml sweep definitions
scripts/setup_env.sh      one-shot provisioning
scripts/env.sh            per-shell environment (source before running)
scripts/plot_results.py   figures
scripts/profile_representative.py  CUPTI profiling
scripts/investigate_topk_heuristic.py  why torch.topk is non-monotonic in N
scripts/investigate_router_crossover.py  where router scoring overtakes selection
benchmarks/timing.py      CUDA-event timing, grouping, graph capture, memory probe
benchmarks/results.py     run directories and the result schema
benchmarks/env_info.py    environment capture
benchmarks/backends/      base.py, dsa_torch.py, dsa_deepgemm.py, router_torch.py
benchmarks/bench_dsa.py   DSA driver
benchmarks/bench_router.py router driver
tests/test_reference_correctness.py
third_party/              pinned upstream checkouts, patches, local toolchain
```

## Known gaps

* **Selected-attention (MLA) integration is not implemented.** The DSA
  measurements cover indexer scoring and top-k selection only. See
  "Unresolved limitation" in the report.
* `nsys` / `ncu` unavailable, as described above.
* **No replay of real prepared queries/keys.** No model checkpoint or captured
  tensors were provided, and no import path for them is implemented. All inputs
  are independent random vectors, which do **not** reproduce the learned
  temporal correlation of real decoding. This is labelled on every result.
