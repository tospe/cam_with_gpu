# Agent handoff: build the CAM study on a validated H100 simulator

## Mission

Build the CAM study on an H100-targeted Accel-Sim workflow. Reuse the user's existing CAM implementation if available. The research objective is to compare a GPU-wide on-chip CAM near L2 with an external CAM near HBM, including asynchronous execution, warp specialization, and eventually Hopper cluster cooperation.

**Hardware assumption: no physical V100 is available or required.** H100 is the hardware target for new tracing, measurement, and validation. Existing V100-oriented source code, traces, configurations, or results are optional regression references only. If none exist, start directly from a verified Hopper-capable simulator and implement the CAM from the protocol specification. Do not make a V100 run or cross-generation comparison a prerequisite.

Do not treat an H100 configuration file or faster timing constants as complete Hopper support. Establish which execution mechanisms the selected workloads use, verify their implementation, and validate them before interpreting placement results.

This is an implementation handoff. The workspace that supplied it contains guides, not simulator source. First establish whether a CAM fork is available and audit it if so; otherwise establish a clean Hopper baseline. Do not assume internal filenames or invent a repository path.

Related documents:

- [V100 placement guide](/Users/tomas/Documents/ChatGPT/GPU_CAM/V100_CAM_PLACEMENT_AGENT_GUIDE.md): CAM semantics, A–F schedules, workload, and fairness requirements.
- [Earlier H100 workload guide](/Users/tomas/Documents/ChatGPT/GPU_CAM/AGENT_IMPLEMENTATION_GUIDE.md): native scoring/selection measurements. This migration does not require running the complete million-expert sweep.

## 1. First actions: inventory and baseline preservation

Ask for concrete missing inputs while progressing on work that does not depend on them:

- Modified Accel-Sim repository path/URL or existing SSH alias and directory.
- Working branch/commit, simulator dependencies/submodules, build commands, and an existing simulator test if available. None of these require physical V100 hardware.
- Access to H100 hardware for compilation, tracing, and measurement, or existing trustworthy H100 traces and measurement metadata.
- CAM hardware parameters and external-interface assumptions currently used.

Inspect repository instructions and working-tree changes. Record the simulator, tracer, performance-model, and submodule commits separately. Preserve uncommitted work and any existing executable/configurations/results; work in an isolated development branch/worktree where appropriate. Do not reset or overwrite the user's fork. If no fork exists, create the H100 CAM development branch on the selected verified baseline.

Inventory the current CAM implementation if one exists; otherwise use this list to plan the new implementation:

1. Functional scoring/selection and result generation.
2. Instruction or trace-marker representation.
3. Submit, wait, result visibility, and write-ordering semantics.
4. Queues, service latency, initiation interval, outstanding limits, and backpressure.
5. Interconnect/L2/memory-controller attachment and packet formats.
6. Request ownership, buffer lifetimes, and completion notification.
7. Statistics, plots, and workload scripts.

If an existing simulation can be replayed from saved artifacts without V100 hardware, save that optional regression result and its resolved configuration, outputs, logs, and command. Otherwise begin with an H100 baseline test. Classify any existing plots as native hardware measurements, simulator outputs, analytical calculations, or mixtures. Do not carry unlabeled analytical timings into a new figure as simulated results.

**First deliverable:** `docs/h100_migration_audit.md`, containing the inventory, missing inputs, known-good reproduction status, and proposed migration route.

## 2. Reuse verified Hopper support before implementing it

As checked on September 23, 2026, upstream Accel-Sim documentation advertises Hopper support, including asynchronous operations and cluster behavior. This is a lead to audit, not proof that the user's fork or a particular upstream commit implements every required behavior correctly.

Inspect a pinned upstream version and its matching performance-model/tracer commits. Build and run relevant examples in an isolated checkout before selecting it as a foundation. Record any differences between documented support and observed behavior.

When CAM code exists, prefer porting it onto a verified Hopper-capable base if the patch is sufficiently isolated. If the fork has extensive infrastructure changes, compare that route with a narrow backport of the required Hopper mechanisms. Without existing CAM code, implement the protocol directly on the verified Hopper base. Explain the maintenance cost and feature gaps of the selected approach in the audit. Avoid a wholesale unreviewed merge.

Separate migration patches into baseline/configuration, CAM engine, frontend markers, completion/ordering, routing, statistics, and tests. Verify ordinary GPU behavior with CAM disabled before enabling CAM workloads.

Create `docs/hopper_feature_coverage.md` with:

```text
feature | required workload | implementation location | trace fields
        | functional/progress test | timing evidence | status/limitation
```

Mark each feature as verified, partially modeled, unsupported, or unnecessary for the current milestone. Unsupported instructions must produce a diagnostic, not silently become a NOP or an arbitrary fixed-latency operation.

## 3. Define the actual H100 target

Record the exact SKU, device properties, memory capacity, SM count, MIG state, driver/toolkit, and relevant clocks/power settings. Match the configured target to the available hardware; H100 SXM and PCIe variants are not interchangeable. Do not assume that simulator bank/subpartition counts equal physical HBM channel counts.

Document each parameter as one of:

- Documented hardware property.
- Measured/calibrated quantity.
- Inherited simulator assumption.
- Hypothetical research parameter.

Record source and units. Keep propagation and external-interface delays in physical time, explicitly distinguishing one-way from round-trip. Convert using the appropriate simulated clock. Do not retain a V100 cycle count while claiming it represents the same physical delay on H100.

For controlled on-chip/external H100 comparisons, retain the same CAM technology/service assumptions in nanoseconds where appropriate. Label a faster CAM clock or new CAM design as a separate change. Precision changes and different kernels are also separate experimental factors. A V100/H100 comparison is outside the required scope.

## 4. Hopper features that require explicit checks

Audit the operations used by the actual traced binary. Source-level API availability does not prove the simulator models the generated instructions.

### A. Asynchronous transfers and memory visibility

Verify support for the global/shared-memory transfer path selected by the compiler, including TMA when used. Check descriptors, addressed regions, transfer size, completion, data visibility, outstanding-resource limits, and any required memory-proxy ordering.

A transfer must create memory traffic and complete according to the timing model; recognizing its opcode is insufficient. Verify that a consumer cannot proceed before the required bytes are ready and that producer buffers are not reused early.

### B. Asynchronous barriers and dynamic waits

Check barrier initialization, arrival counts, transaction accounting where applicable, phase/generation reuse, and actual completion conditions. Arrival and transaction completion are distinct events.

Captured polling iterations must not fix the wait duration to the original hardware run. Verify the selected frontend can reevaluate waits when CAM or memory timing changes. Preserve predicate/control-flow semantics and account for any polling overhead represented by the chosen abstraction. Do not truncate a spin loop without replacing its synchronization behavior.

### C. Asynchronous matrix operations, when used

For workloads using WGMMA, verify operand/register dependencies, issue/commit/wait behavior, and result readiness. A synchronous tensor instruction with a changed latency is not an equivalent model. If the initial microbenchmark does not execute these instructions, mark this feature outside that milestone rather than claiming validation.

### D. Clusters and distributed shared memory

Before any DSM experiment, verify cluster launch metadata, co-scheduling/resource reservation within a GPC, cluster barriers, address mapping to the owning block's shared memory, remote read/write/atomic behavior needed by the workload, and buffer/block lifetime.

Do not assume one block always occupies one SM. Do not allow cluster blocks to be independently placed without respecting the cluster constraints. Do not model remote shared-memory access as a zero-cost local access or silently convert it into ordinary global memory.

Start with two-block clusters and expand only within supported resource limits. Account for occupancy, remote traffic, and synchronization. If multicast is used, verify its destinations and completion semantics separately.

DSM is cluster-scoped shared memory, not persistent GPU-wide CAM storage. TMA moves data; it does not perform search or automatically support a custom CAM endpoint.

## 5. Validate the GPU model without CAM first

Use small kernels that isolate relevant mechanisms, followed by held-out combinations:

| Test | Behavior to verify |
|---|---|
| Dependent arithmetic and memory accesses | Baseline instruction/data dependencies and timing |
| Shared-memory producer/consumer | Visibility, correct participation, buffer reuse |
| Asynchronous copy with consumer wait | Actual transfer traffic, completion, overlap |
| Repeated barrier phases | No stale-completion or generation errors |
| Warp-specialized bounded-buffer pipeline | Progress, full/empty waits, resource pressure |
| WGMMA kernel, if used | Asynchronous matrix-operation dependencies |
| Two-block DSM exchange, if used | Co-scheduling, remote access, synchronization, lifetime |

Compile/trace for Hopper using a compatible toolchain and inspect executed instructions. Use new H100 traces for Hopper kernels. A V100 trace replayed with H100 constants is only a labeled architectural sensitivity exercise.

Compare against H100 hardware when available. Record event-timed GPU execution and profiler evidence separately; instrumentation can alter execution. Define measured regions and cache conditions consistently. Use correctness tools where applicable and separate their overhead from timing.

Before parameter fitting, define the selected timing metrics and acceptable error bands with a rationale appropriate to the mechanism. Report errors for every test and held-out case, rather than only an aggregate correlation. Investigate errors that affect the research conclusion; do not tune parameters to produce a desired on-chip advantage.

If hardware or measurements are unavailable, complete structural/progress validation and label timing as uncalibrated. Do not call the result hardware-validated. Allow further implementation to proceed while leaving that validation gate explicitly open.

**Milestone H0:** pinned build, identified target, feature-coverage table, and reproducible baseline tests. Required-but-missing features are resolved or the corresponding experiment is explicitly excluded.

## 6. Port the CAM request lifecycle

Use the smallest known-correct CAM interface initially. Preserve the same protocol and architecture for both placements:

```text
query ready → submit → capture/fetch input → wait for required writes
            → queue/service → result visible → completion → consumer ready
```

Preserve separate search latency and initiation interval, request ownership, queue capacity, and bounded outstanding requests. A wait must block only its defined dependent execution, not implicitly all SMs.

For this first port, retain the existing explicit result-storage and completion mechanism if it is correct. Do not immediately redesign it around TMA or DSM. If completion later targets a Hopper barrier or shared-memory buffer, specify the added architectural mechanism and validate ownership, accounting, generation reuse, visibility, and block lifetime. Never equate it with an already-supported stock TMA transaction without implementing that contract.

On-chip endpoint: outside the GPCs, attached to the shared GPU interconnect near L2. This is not automatically part of L2 caching or DSM.

External endpoint: outside the GPU die near HBM, reached through an explicitly modeled interface. It must not receive an implicit CPU launch, PCIe route, DRAM access, or broad system fence solely because it is external. Apply these only when required by its defined interface.

Hold CAM capacity, engine resources, candidate contents, precision, merge policy, query/result format, and ordering guarantees constant between placements. Model input fetches, serialization, queueing, and response traffic. Do not count an entire query as one 64-byte packet.

If a global search spans multiple CAM partitions, account for fan-out and merge and match that organization between placements. Otherwise keep one logical CAM for the initial experiment.

### Trace and functional consistency

Document exactly which GPU work is replaced by the CAM. Do not execute scoring/selection kernels and then add CAM delay on top. Keep input preparation and consumers that remain on the GPU.

If reference results are precomputed, make them available only after simulated completion, and ensure their values match any traced dependent addresses/control flow. Timing replay alone is not functional validation. Approximate results that change downstream execution need a separate functional workflow.

Use instrumentation-defined measured-region boundaries to avoid tracing markers, setup, or emulation work being counted accidentally. Include launch overhead only if the simulator actually represents it; do not label kernel-only timing as launch-inclusive.

**Milestone H1:** CAM correctness/progress tests pass under H100, ordinary GPU regressions pass with CAM disabled, and a resident-table request exhibits the expected component timing.

## 7. Reproduce the existing A–F experiment before adding DSM

Use the schedules already defined in the V100 guide:

| Pair | On-chip / external | Execution |
|---|---|---|
| A / B | Same CAM resources | Submit, wait, independent work, consume |
| C / D | Same CAM resources | Submit, independent work, wait, consume |
| E / F | Same CAM resources | Within-block producer/consumer warps with bounded buffers |

All cases perform the same useful work. Equalize work counts and explain resource differences; do not give asynchronous cases free future queries. Compare one independent stream with multiple independent streams while holding total query count fixed, preserving real query-availability dependencies.

Begin with a small smoke case, then one representative DSA-derived configuration such as N = 32768, H = 64, D = 128, k = 2048, provided it fits the configured CAM. Keep the chosen precision and score semantics fixed for paired placement comparisons. Use 64 queries initially and report startup/drain separately or include them consistently.

Do not rerun the full original workload grid until one configuration works and its timeline is explained.

The user's earlier plot showed approximately the same external penalty for A/B and C/D, and near-equal E/F. Treat those as observations to investigate, not numerical targets to reproduce. If 353 ns is an additional RTT, 64 times that delay is 22.592 microseconds; check whether any matching gap comes from actual dependencies or a post hoc penalty. Equal rounded E/F bars do not imply identical first-result latency.

Required outputs:

- Grouped A–F completion-time bars for fixed total work, with one-stream and multi-stream panels.
- External-delay sweep, with the meaning of RTT and separately modeled serialization explicit.
- First-result latency, steady-state completion interval, and queueing/occupancy for E/F.
- Per-request/warp timeline proving independent GPU work and CAM service overlap.

**Milestone H2:** a reproducible Hopper A–F result supported by trace, event, and resource evidence. State the hardware-calibration status separately.

## 8. Add Hopper cooperation as controlled follow-ups

### Within-block asynchronous pipeline

Use supported TMA/asynchronous barriers for ordinary GPU data movement where useful. Compare with a valid simpler transfer implementation at equal work. Distinguish reduced instruction overhead from reduced traffic, additional buffering, and changed occupancy. Keep the CAM notification extension independently specified.

### Cluster/DSM implementation

Add a small GPU-only scoring/selection or local-merge implementation using DSM, then compare with an equivalent within-block/global-memory organization. Preserve the candidate set and include any remaining global merge. Quantify both input loading and intermediate/result traffic.

Cluster cooperation can reduce some intermediate exchanges; it cannot create unlimited resident candidate capacity or eliminate candidate accesses. Do not compare different search scopes and attribute the result entirely to DSM.

### Fusion

Compare the strongest supported GPU scoring/selection baseline and CAM-assisted path with clearly defined boundaries. Different numerical formats, candidate partitions, or attention architectures are separate changes. The Raschka teaching block uses dense masking and must not be labeled an optimized sparse-attention baseline.

Perform feature comparisons with valid alternative implementations. Do not disable a required barrier, co-scheduling guarantee, or memory-ordering operation to make a feature-ablation case run faster; that would change correctness.

**Milestone H3:** focused results showing what within-block overlap, TMA, DSM, and fusion contribute beyond H2, with only the features actually implemented and validated claimed.

## 9. Required logs and diagnostics

Record resolved GPU/CAM/link configurations, executable/compiler options, commits, trace format and metadata, seeds, query counts, numerical formats, and measured-region boundaries.

Capture request submission/acceptance, input readiness, required-write readiness, service start/end, result visibility, completion observation, consumer-ready/issued timestamps, bytes transferred, queue occupancy, outstanding requests, buffer-full/empty waits, and occupancy/resource usage.

For Hopper tests, also capture the relevant barrier phases/counts, transfer completions, warpgroup waits, cluster residency, and remote shared-memory accesses. Add counters only where they answer a validation or research question.

Keep additive latency categories disjoint; write-wait and queueing may overlap. Aggregate warp-stall cycles are not wall-clock execution time. Report unsupported cases, deadlocks/timeouts, and uncalibrated parameters explicitly.

## 10. Deliverables and first assignment

Deliver focused patches with runnable tests, not a single broad simulator rewrite. Suggested artifacts:

```text
docs/h100_migration_audit.md
docs/hopper_feature_coverage.md
docs/h100_cam_protocol.md
configs/<resolved-h100-target>/...
microbenchmarks/hopper_validation/...
microbenchmarks/cam_placement/...
scripts/run_h100_smoke.*
results/<run-id>/metadata + raw events + summaries + figures + report
```

Adapt paths to the actual repository. Preserve any existing experiments and commands, but do not require V100 artifacts to complete the H100 implementation.

The first assignment is **audit, baseline build, and a small H100 validation run**. Then port one CAM request and progress through H1/H2. Do not implement a full Hopper model from scratch before checking reusable upstream support, and do not require a full DSA model or checkpoint to validate the primitives.

At each handoff, state which milestone passed, what was executed versus only inspected, measured discrepancies, and the exact remaining dependency. A working configuration name, successful compilation, or rendered plot alone does not establish correct asynchronous execution.

## References and evidence policy

Use exact source commits in the final implementation. Treat current upstream capability statements as claims to verify against code and tests, not as validation of the modified fork.

- [Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework)
- [Accel-Sim release notes](https://github.com/accel-sim/accel-sim-framework/blob/dev/release.notes.md)
- [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA H100 architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUDA cluster programming](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [CUDA asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [PTX instruction and memory-model reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [User-selected DSA reference](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa)
