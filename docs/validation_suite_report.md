# Bounded H100 validation suite for the CAM placement study, 2026-09-25

Scope, as instructed: validate the GPU execution around the CAM (arithmetic, memory, synchronization, and the three scheduling patterns with TMA standing in for CAM requests) on the H100 PCIe. The band is ±15 % per designated metric, with no averaging. Parameters were frozen before reserved runs. **The CAM engine and external-link parameters remain architectural assumptions and are not validated here.**

- Configurations: baseline `SM90_H100_PCIe` (upstream-derived); development `SM90_H100_PCIe_dev` (= baseline + `-dram_latency 283`). Overrides are named per run: frozen arithmetic candidate `-trace_opcode_latency_initiation_sp 3,2`; investigation `…_sp 2,1`.
- Repos: gpgpu-sim_distribution2 `5c273657`, accel-sim-framework2 `62d2be8` (branch `h100-cam`). NVBit 1.8. nvcc 12.9.86. Driver 610.43.02.
- HW protocol: native runs (no instrumentation), 5 warm-ups plus 20–30 repetitions, median and p95. Before every cold kernel, a clean L2 flush (memset of a 256 MiB buffer, then a stream-read of it). The sim equivalent is `-gpgpu_perf_sim_memcpy 0`. The same launch configuration and event boundaries are used on both sides.

## 1. Arithmetic

Compiled chain: 16 dependent `FFMA` per unrolled iteration plus `IADD3/ISETP/BRA`. `FFMA` → `SP_OP` (`hopper_opcode.h:25`), whose latency and initiation come from `-trace_opcode_latency_initiation_sp` (dev: `4,2`).

Sensitivity on the fit case (1 thread, slope 4096 vs 16384): with initiation 2, latency 5/4/3/2/1 → 8/7/6/6/6 cycles per iteration; with initiation 1, latency 1 → 5. The simulator therefore charges **max(latency + 3, initiation + 4)** per dependent FFMA. With execution resources held fixed (as instructed), the floor is 6.0 against HW 4.438: **the constrained fit cannot reach the band**. Frozen candidate: latency 3 (`results/arith-fit/FROZEN.md`, commit `bb0760f`, written before any reserved run).

| Reserved case | HW median (p95) | dev sp 4,2 | **frozen sp 3,2** | investigation sp 2,1 |
|---|---|---|---|---|
| R1 chain slope 1024 vs 8192 (cycles/iter) | 4.438 (4.438) | 7.000 +57.7 % ✗ | 6.000 **+35.2 % ✗** | 5.000 +12.7 % ✓ |
| R1 chain slope 2048 vs 32768 | 4.438 (4.438) | 7.000 +57.7 % ✗ | 6.000 **+35.2 % ✗** | 5.000 +12.7 % ✓ |
| R2 K=2 W=4, latency-bound (cycles/iter) | 5.155 (5.338) | 7.38 +43.1 % ✗ | 6.44 **+24.9 % ✗** | 5.44 +5.5 % ✓ |
| R2 K=2 W=32 (throughput) | 18.391 (18.775) | 32.50 +76.7 % ✗ | 32.50 **+76.7 % ✗** | 18.00 −2.1 % ✓ |
| R2 K=8 W=4 | 12.907 (13.217) | 17.75 +37.5 % ✗ | 17.75 **+37.5 % ✗** | 10.50 −18.6 % ✗ |
| R2 K=8 W=32 (throughput) | 76.123 (76.361) | 130.00 +70.8% ✗ | **130.00 +70.8% ✗** | 72.00 −5.4 % ✓ |
| R3 arithmetic ring (TMA, 2 slots, work 200, cold, µs) | 57.44 | 67.12 +16.8 % ✗ | 59.69 **+3.9 % ✓** | 52.37 −8.8 % ✓ |

Dependency latency and throughput are separated as follows.

- **Dependency latency**: R1 and R2 at K=2 W=4. The frozen candidate fails (+35 %, +25 %) because of a ~3-cycle fixed pipeline overhead per dependent step.
- **Throughput**: R2 at W=32. The simulator caps at **63 FFMA/cycle/SM**, both latencies, against **111.5 on HW (−43.5 %)**. The cause is initiation interval 2 on 4 SP units, i.e. 64 FP32 lanes/SM/cycle. H100 documents 128 FP32 lanes/SM (4 × 32), which corresponds to initiation 1. This is a **resource discrepancy, not a fit parameter**.
- The investigation with initiation 1 (documented value) and latency 2 passes 6 of 7 arithmetic cases. It was run **after** the reserved cases had been seen, so it is not validated. Adopting it requires a new freeze and fresh held-out cases.

## 2. Memory and synchronization (retained)

| Check | HW | Sim (dev) | Error | Source |
|---|---|---|---|---|
| L2-hit latency (cycles/hop) | 272.4 | 255.0 | −6.4 % ✓ | calib-fit |
| DRAM latency, fit case | 658.0 | 641.3 | −2.5 % ✓ | calib-fit |
| DRAM held-out: 1 GiB / 128 B / seed 2 | 665.0 / 658.5 / 660.7 | 644.1 / 648.6 / 641.0 | −3.1 / −1.5 / −3.0 % ✓ | calib-fit |
| DRAM read BW (slope, GB/s) | 1962 | 1848 | −5.8 % ✓ | calib-fit |
| TMA ring 1/2/4/8 slots, cold (µs) | 48.58 / 29.44 / 22.75 / 21.79 | 48.37 / 26.90 / 19.57 / 18.91 | −0.4 / −8.6 / −14.0 / −13.2 % ✓ | validation |
| Barrier phases (up to 64 per barrier), buffer reuse, producer/consumer | 0 mismatches HW + trace | completes, ordering enforced | ✓ | validation |
| **Known failure:** 3648 concurrent random chases | 1108.6 | 757.7 | **−31.7 % ✗** | calib-fit |
| **Known failure:** cp.async work 0 / work 200 (cold) | 43.26 / 48.48 | 34.89 / 61.70 | **−19.3 / +27.3 % ✗** | validation |

The arithmetic candidates have not been adopted into any config file, so no memory check needed rerunning. If one is adopted, rerun the BW benchmark (its reduction uses FP32 adds) and the TMA ring. The latency chases use only INT and load instructions.

## 3. Scheduling patterns with TMA in place of CAM requests (`microbenchmarks/cam_placement/sched.cu`)

Sizes from the first CAM experiment: query 16,640 B (64×128 FP16 plus 64 FP32 weights), built in shared memory and sent with a **TMA bulk store**. Result 16,384 B (2048 × {id, score}), fetched with a **TMA bulk load** with mbarrier completion. Work is a dependent FMA chain of W iterations per thread. Consume: 4 warps checksum a quarter each, then `atomicAdd` to `out[q]`. SASS: `UBLKCP.G.S`, `UBLKCP.S.G`, `UTMACMDFLUSH`, `SYNCS.*`, `REDG.E.ADD`.

- A: prep → submit → wait → work → consume.
- C: prep → submit → work → wait → consume.
- E: 4 producer warps and 4 consumer warps with an S-slot ring.

Useful work per query is identical across schedules (the same 128 threads' prep/work/consume and the same seeds). E has 4 more warps.

- W chosen from the HW transfer latency (submit → result visible, 1036 cycles median; unchanged for the first query of a kernel): 117 / 234 / 468 ≈ 0.5× / 1× / 2×.
- Concurrency: 1 stream and 16 streams (16 SMs). Queries: Q = 1, 32, 64 per stream in three kernels on disjoint data.
- Metrics: first result = T(1); total = T(64); steady interval = (T(64) − T(32)) / 32.
- Acceptance rules for speedups were fixed in `scripts/sched_compare.py` before any simulated result was seen (commit `31c54e7`).

**Correctness**: HW 0 mismatches (results, sent query bytes, work outputs vs host FMA reference), 30 reps × 14 configurations. Trace mode 0 mismatches in all 14. All 28 + 14 simulations completed.

Frozen candidate sp 3,2 (`results/sched/comparison_sp3.md`):

- **Total: 14/14 pass. Steady interval: 14/14 pass.**
- **First result: 14/14 FAIL (−32 % to −51 %).**
- Relative: **8 MATERIAL flags, 0 reversed rankings**.
  - The sim overstates C-vs-A overlap benefit: +31.1 % vs HW +19.1 % at W = 234; +23.7 % vs +15.8 % at W = 468; +30.8 % vs +20.4 % at 16 streams.
  - The sim understates E-vs-C at W = 468: +14.8 % vs +23.1 %.
  - E-vs-A, E-vs-C at W = 117/234, and ring depth 1→2 (HW +24.6 %, sim +30.4 %) and 2→4 (both ~0) are ok.

Investigation sp 2,1 (`comparison_inv21.md`): 3 total/interval values move just outside the band (−15.3 to −17.5 %). First result is still 14/14 fail. The C-vs-A overstatement persists (+28 % vs +19 %), so it is **not only an arithmetic artifact**.

### Specific discrepancies investigated

1. **First-result latency.** Launch-only kernel (Q = 0): HW 3.9–4.4 µs vs sim 2.25–2.42 µs (the configured 3000-cycle launch latency). After subtracting, the first query itself is still HW 4.06 µs vs sim 2.69 µs (A, W = 234). On HW the first query's transfer is **not** slower than later ones (1037 vs 1036 cycles), so the extra time is kernel start-up / first-query setup outside the transfer (candidates: SM/shared-memory setup, cold instruction cache — the sim uses `-gpgpu_perfect_inst_const_cache 1`, first query build). **Unresolved.**
2. **Overlap benefit (C vs A).** Linear fit of A's steady interval over W: HW 4.39 cycles per work iteration + 3245-cycle per-query fixed part (prep + transfer + consume); sim (sp3) 5.97 + 3013 (−7 %). At W = 234, C hides 0.47 of 0.585 µs of work on HW but essentially all of it in sim. With the fixed part only 7 % off overall, the split within it must differ (likely a relatively longer transfer and shorter prep/consume in sim). Per-component timing inside the sim was not instrumented. **Unresolved, specific.**

## 4. Effect on the planned claims

| Planned claim type | Supported? |
|---|---|
| Steady-state throughput / completion interval of A, C, E at fixed useful work | **Yes, with the frozen candidate** (28/28 within ±15 %) |
| Ranking of schedules (E < C < A; ring depth 1 < 2 ≈ 4) | **Yes** (no reversals) |
| Size of the overlap benefit of C over A (independent work hidden behind the request) | **No**: overstated by ~8–12 points (≈ 50–65 % relative) |
| Size of the warp-specialization benefit at long work | **No** at W = 468 (understated) |
| First-result / single-request latency | **No**: 32–51 % low, cause outside the transfer path unresolved |
| Absolute duration of independent arithmetic work | **No**: dependent FFMA +35 % (frozen), FP32 throughput −43.5 % |
| Scattered gathers, memory-tail latency, contention | **No** (standing restriction; detailed DRAM model not calibrated) |
| cp.async-based transfers | **No** (deferred) |

**Consequence for H2**: comparisons of steady-state completion interval and schedule rankings between placements can proceed on the development config, labelled preliminary. Placement claims that depend on how much independent work hides the request latency (the core A-vs-C and E questions) and on first-result latency **are not yet supported**.

## 5. Remaining limitations and next decisions

1. SP initiation interval: the documented H100 value (1) passes 6/7 arithmetic cases in the investigation. It needs a user decision to adopt, then a new freeze with fresh reserved cases, and reruns of BW, the TMA ring and sched.
2. Per-component timing of TMA transfer vs prep/consume in the sim, to resolve the overlap overstatement.
3. First-query start-up cost on HW (e.g. an in-kernel `%globaltimer` breakdown).
4. Deferred per instruction: detailed-DRAM restructuring, cp.async, DSM, WGMMA, random-gather fixes.

## Incidents in this suite

None that invalidated results. The xferlat measurement runs without an L2 flush; it was used only to choose W and to compare the first query with later ones.
