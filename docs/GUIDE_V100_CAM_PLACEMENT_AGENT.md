# Agent implementation guide: V100 CAM placement and overlap microbenchmark

## Objective

Use the user's existing modified V100-based Accel-Sim to evaluate a software-controlled CAM that stores candidate keys and performs similarity scoring and top-k selection internally.

Implement a small DSA-derived workload and answer:

1. How does on-chip versus external placement affect result-use latency?
2. How much of that difference can asynchronous submission and independent GPU work hide?
3. Does completion tracking block only the work that actually depends on the result?
4. How do ordering of updates, query concurrency, and result size change the outcome?

This is a **V100-based architectural extension study**, not an H100 performance prediction. The immediate deliverable is a validated four-way comparison, not a full LLM simulation.

This brief governs the placement study. `AGENT_IMPLEMENTATION_GUIDE.md` describes the earlier H100 software-baseline work; its H100 dependencies and optimized-kernel requirements do not apply to this first V100 milestone.

## 1. Discover and preserve the existing implementation

- Read applicable repository instructions and inspect the modified simulator before changing it. Locate its repository, branch/commit, V100 configuration, tracer, build instructions, and current CAM extensions. These are not present beside this guide at the time it is written; request the actual path or remote execution procedure if unavailable.
- Determine whether a physical V100 is available or only a V100 simulator configuration. Do not assume H100 SASS traces can be replayed as V100 instructions. Identify a supported way to compile, trace, or execute the microbenchmark for the configured frontend.
- Record the compiler/toolkit and packages used; select versions that support the required Volta target. Use FP16/FP32-compatible operations rather than Hopper-only kernels. DeepGEMM is not required.
- Preserve existing CAM functionality. Explain which parts are reused and which parts are added. Avoid unrelated simulator refactoring.
- Inspect existing latency parameters, admission queues, clock domains, completion handling, and statistics. A unit accepting one packet per cycle does not necessarily accept one complete search per cycle.

Do not fabricate missing hardware estimates. Use the user's existing CAM estimates where available. Missing interface parameters may be swept as explicitly hypothetical values, but must not be presented as measured hardware characteristics.

## 2. Minimal workload and functional contract

Use one logically shared CAM and a resident candidate table initially. All placements search the same candidates with the same numerical semantics. Exclude partition fan-out and global top-k merging from this first milestone unless the existing engine requires them; if required, implement the same organization and merge resources in both placements.

### DSA-derived search

Use the scoring semantics from the user's selected [Raschka DSA reference](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa), with a pinned commit and attribution. For a prepared query:

```text
score[j] = sum_h (weight[h] / sqrt(H))
                   * ReLU(dot(query[h, :], key[j, :]) / sqrt(D))
result = top_k(score, k)
```

If scaling is already incorporated into stored inputs, do not apply it twice. Preserve the chosen reference's behavior and define ties deterministically for the initial experiment.

- Candidate table: [N, D]. Query: [H, D]. Query weights: [H].
- Start with H = 64 and D = 128 for the main experiments; allow smaller correctness/smoke cases.
- Use FP16 inputs with a documented accumulation policy, such as FP32, and return int32 IDs plus FP32 scores. Record numerical differences from any hardware-specific CAM representation.
- The CAM performs both scoring and selection. Do not model only top-k on scores already computed by the GPU and claim offload of the complete search.
- Use preprojected resident keys. Reprojection of the entire context and the reference's growing `torch.cat` caches are outside this microbenchmark.
- Generate deterministic, varying queries outside the measured region. Clearly label synthetic data; this does not evaluate a trained model's accuracy.

The GPU consumer reads the returned IDs/scores, computes a small checksum, and stores an observable output. It must not execute with invalid results. This deliberately isolates completion and consumption; selected embedding fetches and attention are later experiments, not part of the initial checksum consumer.

Build a CPU or GPU functional reference for small cases. If timing simulation uses precomputed correct results, explain how result bytes become available only on simulated completion. Preserve result-dependent addresses/control flow when replaying traces. Do not allow placeholder IDs to change downstream work, and do not execute the original GPU search in addition to charging CAM search latency.

## 3. Four required configurations

| ID | Placement | Execution schedule |
|---|---|---|
| A | On GPU die, attached to the shared interconnect near L2 | Submit, wait, independent work, consume |
| B | External CAM near HBM, outside the GPU die | Submit, wait, independent work, consume |
| C | Same on-chip organization as A | Submit, independent work, wait, consume |
| D | Same external organization as B | Submit, independent work, wait, consume |

**All four execute the same useful independent work. Only its order relative to the wait changes.** Do not add work exclusively to the asynchronous cases or compare different computations.

Conceptual code, not an existing CUDA API:

```cpp
request = cam_submit(query, region);
if (schedule == WAIT_FIRST) {
    result = cam_wait(request);
    independent = independent_work(input);
} else {
    independent = independent_work(input);
    result = cam_wait(request);
}
output = consume(result, independent);
```

The independent work must have no data dependency on this search and must contribute to output so it cannot be optimized away. Initially use a controllable arithmetic workload; add memory-intensive independent work only as a separately labeled contention experiment. Inspect generated instructions and verify the instruction mix remains comparable between schedules.

Use device-initiated requests for both placements. Do not insert a host round trip or a CPU launch per external query. The initial integrated-kernel interface is part of the proposed architecture in both cases, not a stock V100 feature.

## 4. Architectural semantics to implement explicitly

### Submission and completion

- Submission returns a request handle after obtaining the required queue/descriptor resources. It does not wait for search completion.
- Track request ownership, IDs, outstanding limits, and result-buffer lifetime. Keep query inputs immutable until safely captured.
- A wait depends on the identified request. An implementation may block the issuing warp, but it must not implicitly block an entire SM, GPC, or GPU. Document the granularity.
- Independent warps remain schedulable. Result storage is visible before the request is signaled complete; consumers cannot see stale or partially written results.
- Backpressure is explicit when queues or outstanding-request slots are full.
- Model completion using scheduler/scoreboard/event mechanisms appropriate to the codebase. Do not assume stock Volta barriers automatically order a new CAM engine.

### Candidate writes

For the later append experiment, a dependent search observes all required committed writes. Define a region/version or another explicit dependency mechanism. Submission order from unrelated producer SMs is not sufficient by itself to publish a complete table.

Start with no concurrent mutation of a searched region. Later compare:

1. A search waiting for all prior accepted CAM writes.
2. A search waiting only for prior writes to its required region.

Implement both policies in both placements. Do not manufacture a synchronization disadvantage exclusively for the external design.

### Frontend and trace integration

Choose the smallest robust extension supported by the existing simulator: explicit modeled instructions, trace annotations, or another documented mechanism. Submission, independent execution, waiting, and completion must remain distinct timed events.

Do not approximate asynchronous execution by adding delay to a whole kernel. Do not rely on a fixed recorded polling-loop iteration count for a completion that changes with placement. Document how the frontend preserves dynamically varying waits and multiple requests.

If the current extension only implements a blocking CAM operation, identify this as a required change for C/D. Do not report overlap results until the engine can actually execute CAM work concurrently with GPU work.

## 5. Fair placement model

Hold these fixed in A/B/C/D:

- Total candidate capacity, contents, precision, scoring semantics, and k.
- CAM search latency L and initiation interval II, supplied by the hardware model and allowed to depend on workload size.
- Search pipelines, internal banks, ports, queues, and outstanding limits.
- Query/result formats, GPU-side submission interface, and completion semantics.
- GPU configuration, workload, work count, and any merge policy.

On-chip path: shared GPU interconnect to the CAM endpoint and back.

External path: shared GPU interconnect, a specified external-interface endpoint/link, external CAM, and return path. Draw the GPU die boundary. Do not assume this link is PCIe, that CAM commands must perform DRAM accesses, or that it shares HBM scheduling unless the proposed architecture explicitly does so.

Model request and response byte counts, packetization, serialization, queueing, and link directionality. For illustration, H = 64, D = 128, FP16 queries and FP32 weights require 16,640 input bytes before headers; k = 2048 with int32 IDs and FP32 scores returns 16,384 bytes. A 64-byte transport unit is not a complete query or response.

Document how inputs reach the endpoint and where results are delivered. If descriptors reference GPU memory, include input fetches and visibility; if data is sent directly, account for its transport. Apply a consistent frontend for the controlled comparison.

Specify external latency in nanoseconds, explicitly stating one-way or round-trip. Document GPU/interconnect/CAM/link clocks and conversions. Avoid mixing DRAM ticks and GPU cycles. Fixed propagation/interface delay must not double-count separately modeled serialization or queueing.

Resident-table experiments require enough configured capacity for the working set, but this is a controlled assumption, not proof that the capacity fits a practical chip area. Measure and report setup separately. Do not use the earlier figure's 64/32/1-unit organizations in the first equal-resource experiment.

## 6. Staged experiment matrix

### Stage 0: correctness and timing smoke tests

Use small N and k, one issuer, and one outstanding request. Validate search results, timing, output visibility, and blocking/overlap behavior before long simulations.

### Stage 1: required four-way placement/overlap result

- Start at N = 8192, H = 64, D = 128, k = 2048; use a smaller documented smoke configuration if necessary.
- Use one issuer warp with one outstanding request to isolate dependency latency. Report this as a controlled issuing pattern, not full-model scheduling.
- Run A/B/C/D with independent-work budgets calibrated to approximately 0, 0.5, 1, and 2 times the unloaded CAM request-to-result latency. Calibrate work alone, then hold its actual work count fixed across placements. Report observed cycles; do not inject fictional compute delay in place of GPU execution.
- Sweep added external round-trip fixed delay over 0, 0.25L, 0.5L, L, 2L, 4L. These are hypothetical sensitivity points; also include the user's real interface estimate when available. Zero added fixed delay may still incur serialization and queueing, so label it accurately.
- Initially use a documented high-bandwidth interface to isolate fixed-delay sensitivity, then rerun at the intended finite bandwidth. Do not describe the high-bandwidth case as a realistic measured link.

Observe A versus B for exposed placement cost; A versus C and B versus D for overlap; C versus D for the remaining placement difference after both support overlap.

### Stage 2: concurrency and backpressure

- Sweep GPU-wide outstanding-request limits of 1, 2, 4, 8, 16 using enough independent issuer warps/query streams to reach them.
- Separate active issuer count, per-issuer outstanding limit, and aggregate outstanding count in configuration and results. A blocking issuer alone cannot generate the full sweep.
- Fix the total query count/work for paired latency comparisons. For throughput runs, use enough queries to distinguish startup/drain from steady state; show convergence before increasing run length further.
- Repeat at N = 32768 and 131072 only where the configured capacity permits.
- Sweep finite interface bandwidth and CAM II independently around their reference settings. Do not change L when sweeping II unless justified by a separate hardware configuration.

### Stage 3: write ordering

Compare resident read-only queries, append-then-search, and writes to region A concurrent with queries to independent region B. Separate genuine data-dependency waiting from unnecessarily broad ordering. Include update traffic and maintain the same ordering guarantees across placements.

### Stage 4: response size

Sweep k = 32, 128, 512, 2048 at a fixed N >= k. Use appropriate CAM L/II values if internal selection cost changes with k. To isolate transport alone, additionally run a clearly labeled controlled variant with internal service parameters fixed.

Stop the initial implementation campaign after the required four-way result and validated concurrency behavior. Implement the later stages incrementally rather than launching the full Cartesian product of every parameter.

## 7. Instrumentation and result interpretation

For each request, record, where applicable:

```text
submit_attempt, submit_accept, endpoint_arrival,
required_writes_ready, service_start, service_end,
result_visible, completion_observed,
consumer_dependency_ready, consumer_issue
```

Record region/version, request ID, issuer, bytes in/out, queue occupancy, backpressure, actual outstanding count, and CAM/link utilization. Separate input capture time if it is not included in these events.

Primary metrics:

- Complete microbenchmark-region time, including the identical independent work and consumer.
- Request-to-result latency distribution and consumer dependency-ready latency.
- Completed-query throughput, with whole-run and steady-state intervals labeled.
- Issuer wait/backpressure cycles, plus unrelated-warp progress during waits.
- Request/response traffic, queueing, and internal service time.

Wait-for-write, queueing, and transport can overlap. Use nonoverlapping categories for additive time breakdowns, or report overlapping counters separately. Aggregate warp-stall cycles are not elapsed application time. Consumer-ready and consumer-issued differ because scheduling may delay a ready instruction.

Expected patterns to investigate, not predetermined outcomes:

- Reduced external penalty with independent work suggests latency hiding.
- Growing queueing and saturated links under concurrency suggest a throughput limit.
- Equal slowdown in both placements may indicate CAM service or GPU-side limits.
- Stalls on region B due to unrelated region A writes expose overly broad ordering.
- A small placement gap is a valid result; do not tune assumptions to force an on-chip win.

## 8. Required validation

- Check functional scores/IDs against the reference, including ties and invalid configurations.
- Verify that A/B/C/D produce equivalent outputs for identical inputs and independent work.
- Check single-request latency against the configured component model, including serialization.
- Verify pipeline admission respects II and queue capacity; latency and initiation interval are not interchangeable.
- Demonstrate independent arithmetic progresses before CAM completion in C/D and not during the explicit wait in A/B for the same issuer.
- Demonstrate another eligible warp can progress while one warp waits.
- Test request ID reuse, out-of-order completions if supported, queue saturation, region isolation, and teardown with outstanding work.
- For append mode, check that the dependent result includes the committed new key when it should rank in top-k.
- Verify trace substitution removes exactly the work offloaded, retains consumers, and does not expose precomputed outputs prematurely.

Run appropriate existing simulator regression checks. Calibrate the unchanged GPU behavior against physical V100 measurements if available; if not, disclose the limitation and avoid claiming hardware validation. H100 measurements are not direct validation of a V100 configuration.

## 9. Warp specialization and DSM: additional controls

The user identified warp specialization, Hopper thread-block clusters, distributed shared memory (DSM), and TMA as relevant alternatives or complements to CAM integration. Address them explicitly rather than assuming all GPU cooperation requires global-memory exchanges.

### V100 follow-up: specialize warps inside one block

Warp specialization predates Hopper and can use producer/consumer synchronization within one block. Add two configurations after A/B/C/D are validated:

| ID | Placement | Schedule |
|---|---|---|
| E | On-chip CAM | Producer warps submit requests; consumer warps consume completed results |
| F | External CAM | The same producer/consumer pipeline and work |

Use a bounded ring of request/result slots. Represent query readiness, submission, result readiness, and consumer release separately. Buffer reuse requires consumer completion, not merely CAM completion. Use per-slot generations or equivalent protection against observing a previous iteration's completion.

- Put communicating warps in the same block for the initial V100 implementation. Do not rely on arbitrary producer and consumer blocks being resident simultaneously.
- Use supported synchronization and memory-ordering mechanisms. Barriers must have correct participation; do not place a whole-block barrier only inside a producer or consumer branch. Include progress/deadlock tests at saturation.
- Explicitly bridge the new CAM completion event to the software-visible ready state. Ordinary barriers do not automatically observe an invented accelerator operation.
- Sweep one, two, and four slots where resource limits allow. Account for shared memory, registers, active warps, and achieved occupancy. Reject configurations exceeding the per-block/SM budget rather than silently changing them.
- Keep query count, work, placement resources, and frontend comparable. Add a nonspecialized control with comparable block size and allocation when attributing gains specifically to warp specialization.
- Use genuinely independent queries, such as different sequences. A next autoregressive query cannot be submitted before its required preceding computation produces it. Preserve this availability constraint and distinguish a saturated synthetic query stream from an application-derived schedule.
- Record producer queue/buffer-full waits, consumer result/buffer-empty waits, handoff cost, outstanding queries, occupancy, single-query latency, and steady-state throughput.

Compare E/F to the strongest applicable A/B/C/D schedule. A reduction of the external-placement penalty is a valid outcome. Throughput improvements do not establish a reduction in the latency of a single dependent query.

### Hopper follow-up: DSM and TMA

DSM is shared-memory access among blocks of a thread-block cluster, not persistent GPU-wide storage. Hopper clusters guarantee co-scheduling within one GPC. Shared-memory lifetime and synchronization still require explicit management; synchronization is not automatic or free. TMA accelerates supported asynchronous data movement, not CAM similarity or top-k itself. Do not assume stock TMA commands can address the proposed CAM.

Do not add DSM/TMA labels to a V100 timing model without implementing their architectural behavior. For a future validated Hopper model, evaluate:

1. A GPU-only tiled scoring/selection implementation that uses cluster cooperation where useful, including local candidate merging in DSM and any remaining global merge.
2. The strongest available within-block warp-specialized control, to separate DSM's contribution from scheduling and fusion.
3. On-chip and external CAMs with equally capable producer/consumer frontends, stating whether and how completion integrates with asynchronous barriers.

Model cluster co-scheduling, remote shared-memory traffic, barriers, memory lifetimes, occupancy, and TMA transactions if used. Account for candidate data still fetched from global memory, cluster capacity, and communication across clusters. Do not equate summed per-cluster shared memory with a single globally accessible CAM capacity.

Report the comparison by scope: within-block cooperation, cluster cooperation, and GPU-wide searchable storage. DSM may reduce intermediate score/merge traffic in the GPU baseline; it does not itself provide a top-k operation or eliminate the need to access the searchable candidates.

E/F are a second milestone; the first four-way experiment remains the starting point. Hopper features are future work unless a compatible model is already available and its use is explicitly selected.

## 10. Fusion follow-up, outside the first milestone

After A/B/C/D work, distinguish two questions:

1. **GPU-only baseline fusion:** does combining scoring and selection reduce intermediate memory traffic? Use a supported implementation and quantify register/shared-memory pressure; do not assume full global top-k fits in one block.
2. **CAM interface integration:** does keeping submission/wait/consumption inside a kernel help compared with separate kernels and materialized buffers?

The first milestone already assumes an integrated device-side CAM interface for both placements. A later separate-kernel comparison must preserve computations and account for buffer traffic and modeled launch costs. If host launch timing is not represented by the simulator, state that and do not infer launch savings from the trace alone.

Do not claim that fusion or device-initiated offload is exclusive to an on-chip CAM. Quantify the actual interface restrictions of each proposed design. Do not treat V100 timing rescaling as a substitute for modeling Hopper's different asynchronous execution features.

## 11. Deliverables and acceptance

Provide:

- Focused simulator changes and a short explanation of the request lifecycle and chosen frontend mechanism.
- The microbenchmark, functional reference, and meaningful correctness/timing tests.
- Reproducible named configurations for A/B/C/D plus a small smoke-run script.
- Machine-readable raw request events, run summaries, and resolved configuration/environment metadata, including simulator commit and seed.
- PNG/PDF plots: total region time versus added external delay; overlap benefit versus independent work; throughput versus outstanding queries. Label all hypothetical hardware parameters.
- A report that states what is measured on hardware, what is simulated, which assumptions are controlled, and which conclusions remain architecture-specific.

For the warp-specialization follow-up, also provide E/F configurations, progress/buffer-reuse tests, resource/occupancy measurements, and a comparison to the nonspecialized controls. State that DSM/TMA are not represented in the V100 results.

The first milestone is complete only when all four configurations run, produce equivalent results, demonstrate actual overlap/backpressure, and generate a reproducible placement plot. A configurable delay alone is insufficient.

If an essential repository, trace source, or hardware parameter is missing, progress on independent code/specification work and request that concrete missing item. Report unrun or blocked experiments explicitly. Do not invent results or migrate the project to a different GPU architecture without instruction.

## References

- [User-selected DSA reference](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa)
- [Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework): inspect the user's actual fork/version rather than assuming current upstream capabilities.
- [NVIDIA Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
- [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUDA thread-block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters)
- [CUDA cluster synchronization and distributed shared-memory example](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html)
- [CudaDMA: warp specialization on pre-Hopper GPUs](https://research.nvidia.com/publication/2011-11_cudadma-optimizing-gpu-memory-bandwidth-warp-specialization)
