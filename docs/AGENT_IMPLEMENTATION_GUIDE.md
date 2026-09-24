# Implementation brief: H100 search and top-k benchmarks

## 1. Objective and boundaries

Implement and run reproducible native-GPU benchmarks on the user's single NVIDIA H100 to measure how search latency scales with the number of candidates:

1. DeepSeek Sparse Attention (DSA): increase the number of cached tokens searched during decoding.
2. Mixture-of-experts routing: increase the number of candidate experts from 16 to 1,048,576 (2^20).

The research motivation is a proposed software-controlled, GPU-wide CAM that stores candidate vectors and performs similarity computation and top-k selection internally. These experiments establish the existing GPU baseline. They do not implement the CAM, modify Accel-Sim, or demonstrate a CAM speedup or an advantage of on-chip placement.

The first priority is a correct, timed DSA scoring-and-selection experiment using the Raschka reference linked by the user. Add the expert-router sweep after this works, then establish an optimized-kernel comparison. Preserve the distinction between operator timing, DSA-block timing, and full-model inference timing throughout the implementation and report.

Do not train a model, download a complete DeepSeek checkpoint, or attempt to instantiate one million expert networks. Do not create an artificially slow GPU baseline to strengthen the motivation.

## 2. Inspect the environment before implementing

- Read repository instructions and inspect any existing implementation. At creation of this brief, the project contains no implementation files.
- Determine where the H100 is accessible. If the current machine lacks it, prepare code and CPU correctness checks locally; request the existing remote connection or execution procedure before GPU runs. Do not invent access details or report local CPU measurements as GPU results.
- Record the GPU model and form factor when available, total/free memory, MIG configuration, driver, CUDA toolkit/runtime, PyTorch, Python, GPU clocks/power configuration, and relevant package versions. Do not assume every H100 has the same memory or bandwidth.
- Record whether the GPU is shared with other jobs. Do not change clocks, power limits, persistence mode, or stop other users' processes without authorization.
- Prefer an isolated environment and pin dependency versions and upstream commits. Reuse working installations when appropriate.

Use the initial inspection to select compatible H100 kernels. Do not replace optimized kernels with an easier-to-trace implementation: this phase measures native hardware and does not require simulator tracing.

## 3. Deliverables

Use a small structure such as the following; adapt it to an existing repository if one appears:

```text
README.md
requirements.txt or pyproject.toml
configs/quick.yaml
configs/full.yaml
benchmarks/bench_dsa.py
benchmarks/bench_router.py
benchmarks/timing.py
benchmarks/backends/
tests/test_reference_correctness.py
scripts/plot_results.py
results/<run_id>/metadata.json
results/<run_id>/measurements.csv
results/<run_id>/summary.csv
results/<run_id>/figures/
results/<run_id>/report.md
```

Provide CLI controls for device, backend, dtype, seed, candidate counts, batch sizes, k, dimensions, warmup, repetitions, and output directory. Provide a dry-run mode that validates configurations and estimates memory without allocating the complete sweep. Avoid adding a general-purpose benchmark framework.

Keep separate immutable run directories. Include exact commands and resolved configurations. Document minimal installation, a smoke test, a quick experiment, and the full sweep.

## 4. DSA workload

### Start from the user's selected implementation

Reuse [Raschka's DSA implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa), linked from [the user's article](https://sebastianraschka.com/blog/2026/deepseek-sparse-attention-from-scratch.html). Pin a commit, retain its license/attribution, and run the upstream tests. Wrap its `LightningIndexer` and existing top-k call rather than writing new CUDA kernels. DeepGEMM installation is not a prerequisite for the first measurements.

Keep three backend labels: `rasbt_reference` for the instrumented original, `rasbt_cached` for a documented cache adaptation, and `optimized` for a later library comparison.

The reference attention computes dense scores and applies a top-k mask; it does not implement MLA or fused sparse attention. Its default indexer dimensions are 4 heads, dimension 64, and k = 64. Use those for an initial smoke test, then explicitly select the larger preset in the sweep below. Label its block timings as GPT-style DSA reference timings, not production DeepSeek performance.

The code reprojects the whole hidden-state context into indexer keys and grows caches with `torch.cat`. Profile those costs separately. For `rasbt_cached`, preallocate caches and append projected indexer keys; validate equivalent scores and selections against the original. Extract the one-query path for long-context sweeps, avoiding full dense prefill. Inspect positional capacity and generation-loop indexing before using the demo as a model benchmark. The demo initializes random weights, so its output is not a trained-model quality result.

### Preserve the scoring semantics

Use the chosen DSA implementation as the source of truth. DeepSeek-V3.2's configuration supplies a useful starting point: 64 indexer heads, indexer dimension 128, and top-k = 2048.

Conceptually, with prepared queries and keys, the indexer computes:

```text
score[b, j] = sum_h weight[b, h] * ReLU(dot(query[b, h, :], key[b, j, :]))
selected_ids[b, :] = topk(score[b, :], k)
```

Preserve scaling, quantization, masking, and layouts from the selected implementation. In the ordinary per-sequence interpretation, query is [B, H, D], indexer keys are [B, N, D], weights are [B, H], and final scores are [B, N]. The indexer heads share the candidate key representation within a sequence. Production paging may use different physical layouts.

Do not substitute cosine similarity or a single-head dot product and call it DSA. If a simplified workload is useful for debugging, label it explicitly and keep it out of the primary DSA results.

Reuse the readable PyTorch reference and supplement its checks on small inputs. After the first reference measurements, investigate DeepGEMM's H100-compatible indexer kernels and an established GPU top-k implementation for the stronger performance baseline. Match score semantics and explicitly account for dtype differences when comparing backends. Use the source links below and record the chosen commits. If an optimized backend cannot be used, report the specific obstacle; the reference measurements remain useful but do not establish an optimized-GPU baseline.

### Experiment modes

1. **Fixed-context operator benchmark, required first:** prepare a cache of N candidates, then time a batch of one query per sequence through scoring and top-k. This isolates scaling with N.
2. **Incremental decoding-block benchmark:** start from N cached tokens and append one new indexer key per sequence per step, running 128 steps. Respect the selected implementation's causal/self-token convention. Measure updates and search together, as well as useful stage timings in separate profiling runs.
3. **Attention integration:** first instrument the reference's GPT-style attention block, labeling its dense masking behavior. For the optimized comparison, connect selected IDs to an actual supported sparse-attention/MLA implementation and preserve its KV representation and quantization. Record each block's boundaries: these architectures are not interchangeable controls for measuring kernel speedup. If query/key projections are omitted, say that prepared projections are the block inputs. Do not label the teaching block, an arbitrary gather, or generic attention as a complete DeepSeek MLA block.

Use independently timed setup for allocation and initial cache population. Report setup separately from steady-state timing and include it in any claim about a complete request. A synthetic block replay is not text generation; full-model latency requires running the actual model.

### Sweep

| Parameter | Quick validation | Main experiment |
|---|---|---|
| Initial cached tokens N | 8192, 32768 | 8192, 16384, 32768, 65536, 131072 |
| Independent sequences B | 1 | 1, 8 |
| Indexer heads H | 64 | 64 |
| Indexer dimension D | 128 | 128 |
| k | 2048 | 2048 |
| Incremental steps | 16 | 128 |

Batch size means distinct sequences with distinct logical caches. Do not silently share a single cache across the batch. Preallocate capacity through N + steps to avoid allocation on each update. Reset cache contents, valid lengths, and replay position outside timing before each incremental trial, including after warmup, so repeated trials start from equivalent states.

Use the native dtype/layout supported by the selected optimized backend; make dtype an explicit experimental field. A BF16 reference and FP8 optimized path are different numerical configurations and must be identified accordingly.

Keep decode separate from prefill. Do not allocate an N-by-N score matrix for a one-query-per-sequence decoding benchmark. Contexts beyond the chosen model's supported range are optional synthetic scaling studies, not evidence of model capability.

## 5. Expert-routing workload

Treat “experts” as MoE router candidates unless the user specifies another meaning. The scalable first benchmark is a flat router:

```text
logits = hidden_states @ router_weights.T
selected_scores, selected_ids = topk(logits, k)
```

Here hidden_states is [B, D] and router_weights is [E, D]. E counts candidate experts, not expert MLPs. For this decode benchmark, B is the number of concurrently routed tokens.

Use D from the user's model when known; otherwise use D = 4096 as a clearly labeled synthetic default. Use k = 8 for the primary sweep and B = 1, 8. Require k <= E. Treat optional k = 2 or additional dimensions as secondary experiments after the primary sweep.

- Quick sweep: E = 16, 256, 4096.
- Full sweep: E = 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576.

The flat router is not automatically equivalent to a model-specific MoE router. If evaluating the user's actual router, preserve activation, correction biases, expert-group selection, normalization, and returned routing weights. Name that backend separately and validate it against the model implementation.

The large-E experiments measure routing scalability only. They do not measure dispatch, expert-network execution, communication across GPUs, or model accuracy at those expert counts.

At E = 2^20 and D = 4096, BF16 router weights alone occupy 8 GiB. Estimate all live allocations before each configuration, including workspaces and outputs. Use a configurable memory budget with headroom; record OOM configurations and continue safely. Never silently lower E, D, precision, or batch size.

## 6. Input data and correctness

- Provide deterministic synthetic inputs and record the seed and generation distribution. Generate inputs outside timed regions.
- Support replay of real prepared queries, weights, and keys when available from the user's model. Validate tensor shapes and metadata on import.
- Use varying queries across trials rather than repeatedly searching one identical query. For incremental replay, document how successive queries and appended keys are obtained. Independent random vectors do not reproduce learned temporal correlation or real token generation.
- Repeated use of resident tables is a legitimate steady-state scenario; disclose it. Do not flush caches arbitrarily. Inspect memory traffic to understand residency effects, especially for small expert tables.
- Compare scores and selection against a high-precision reference on small cases, including masking, multiple batches, updates, and k near the candidate count.
- Define dtype-appropriate numerical tolerances. Handle ties explicitly: valid tied selections need not have identical IDs, but invalid lower-score selections must not pass. Check shapes, unique IDs, valid ranges, causal bounds, and score/ID correspondence.
- For quantized inputs, distinguish implementation correctness against a reference using the same effective inputs from selection changes relative to higher precision. Report selection overlap if relevant; it does not replace model-quality evaluation.

Test mathematical behavior and update visibility, not merely implementation internals. Random-data latency results must not be presented as accuracy results.

## 7. Timing and profiling protocol

1. Allocate inputs, outputs where supported, and persistent state before timed trials. Trigger compilation and backend initialization before measurement.
2. Start with 20 warmup iterations and at least 100 measured iterations for operator runs. Increase repetitions when measurements are unstable, recording the actual counts.
3. Use CUDA events on the execution stream for GPU elapsed time. For very short operations, also time groups of repetitions and divide by the count, reporting that measurement method. Account for other streams if a backend uses them.
4. Synchronize before beginning an independent trial and after recording its end event as needed. Do not place a device-wide synchronization between every pipeline stage. Avoid CPU transfers, tensor printing, allocations for input generation, and Python scalar extraction inside the measured region.
5. Measure scoring, top-k, and the combined path. Separately timed stages are diagnostic and need not sum to the total because fusion, cache state, and overlap can differ.
6. When a backend fuses scoring and top-k, report the combined stage. Mark unavailable separate timings as null, not zero. Do not disable fusion for the primary baseline or assign artificial stage percentages to fused work.
7. For incremental loops, record actual context length per step and total loop duration. Do not summarize a growing-context run as repeated samples at one fixed N. If CUDA Graph capture is supported, report graph mode explicitly and use comparable modes between backends.
8. Report median, p95, and variability across repeated trials. For batch B, report batch latency and B / batch_latency as query throughput. Batch latency divided by B is amortized cost, not the latency experienced by each query.
9. Capture Nsight Systems timelines for representative cases to identify gaps, overlap, and consumer dependencies. Use Nsight Compute separately for selected kernels to inspect memory traffic and execution behavior. Record profiler metrics and availability; profiling runs are not the primary timing samples.
10. Separate GPU elapsed time from host-observed latency. GPU event timing can include host submission gaps between kernels; use timelines or graph replay to identify them rather than assuming every gap is GPU work.

Do not claim that all query-response time is additional synchronization overhead. A later CAM study must separately define dependency tracking, data visibility, transport, and search computation.

## 8. Result schema and figures

Store raw samples and a summary. Every row must identify at least:

```text
run_id, workload, mode, backend, backend_commit, dtype, batch_size,
candidate_count, dimension, indexer_heads, k, seed, trial, decode_step,
stage, graph_mode, timing_method, elapsed_us, status, error
```

Use null for inapplicable fields. Store environment/configuration metadata once per run. Summaries include sample count, median, p95, variability, and peak allocated/reserved memory. Preserve failures and unsupported cases in machine-readable output.

Produce these figures with standard plotting tools, exporting PNG and PDF:

1. DSA scoring, selection, and combined latency versus context length, with separate series/panels for batch size and backend.
2. Router scoring, selection, and combined latency versus expert count, with a logarithmic candidate-count axis.
3. Combined attention-block latency and a supported breakdown, labeling the GPT-style reference and any production sparse-attention integration separately.
4. Throughput versus candidate count for B = 1 and B = 8.

Show variability and distinguish measured points from any interpolation. Use stacked bars only for a breakdown measured in a way that supports an additive interpretation. Never stack independently measured stages and label their sum as observed end-to-end latency.

The report should state where time grows, where kernel fusion or batching changes the picture, whether candidate data is repeatedly read from HBM, and which conclusions the measurements support. DSA-block fractions are not full-LLM fractions. Clearly label synthetic workloads and provisional backends.

## 9. Implementation order and completion criteria

### Milestone A: First useful H100 result

- Environment recorded and dependencies pinned.
- DSA reference passes correctness checks.
- Raschka-based benchmark runs the quick DSA sweep, with scoring, selection, projection, and cache-maintenance coverage documented. Optimized dependencies must not block this milestone.
- Raw samples, a latency plot, and a short interpretation are saved.

### Milestone B: Complete baseline study

- Main DSA fixed-context and incremental sweeps run for the reference and cache adaptation, with the optimized comparison and attention integration documented or explicitly marked incomplete.
- Router quick and full sweeps run; OOM/unsupported cases are recorded without hidden configuration changes.
- Correctness checks cover both workloads.
- Representative profiler evidence is saved without contaminating normal timing runs.
- README contains commands that reproduce the results; report distinguishes every workload scope and numerical configuration.

If hardware access, optimized backend support, or the intended attention integration is unavailable, deliver runnable code and precise remaining steps, marking the affected milestone incomplete. Milestone A can be complete while the optimized comparison in Milestone B is pending. Do not invent measurements or silently substitute a different workload.

The handoff should link the code, resolved configuration, raw data, plots, and report, and identify the exact H100 used. Future work may add Accel-Sim traces and CAM placement comparisons after this baseline has been reviewed.

## 10. Primary references

Consult and pin the implementations actually used; upstream main branches can change.

- [User-selected article: DeepSeek Sparse Attention From Scratch](https://sebastianraschka.com/blog/2026/deepseek-sparse-attention-from-scratch.html)
- [Raschka DSA reference, README, and tests](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa)
- [DeepSeek-V3.2 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)
- [Official DSA reference and kernel guidance](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [DeepGEMM indexer scoring semantics and kernels](https://github.com/deepseek-ai/DeepGEMM)
- [FlashMLA sparse-attention kernels](https://github.com/deepseek-ai/FlashMLA)
- [Top-k motivation and optimized-baseline context](https://arxiv.org/abs/2604.22312)
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)

The cited top-k paper evaluates Blackwell. Do not transfer its timings or speedups to H100 or assume its kernels run unchanged on H100.
