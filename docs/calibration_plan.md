# Calibration plan: SM90_H100_PCIe vs. the physical H100 PCIe

Written 2026-09-25, **before** any hardware/simulator comparison (guide §5: define metrics and error bands before fitting).

## Metrics

| ID | Metric | Hardware measurement | Simulator measurement |
|---|---|---|---|
| M1 | L2-hit load latency (cycles/hop) | single-thread `__ldcg` pointer chase over a 1 MiB list after an all-SM warm-up kernel; slope of in-kernel `clock64` between 256 and 1024 hops | slope of kernel `gpu_sim_cycle` between the same two kernels |
| M2 | DRAM load latency, cold L2 (cycles/hop) | same chase over a 256 MiB random list; L2 flushed (256 MiB memset) before each rep | same; `-gpgpu_perf_sim_memcpy 0` so the list copy does not pre-fill L2 (equivalent cold state) |
| M3 | DRAM read bandwidth (GB/s) | `float4 __ldcg` grid-stride read of 128 and 256 MiB (separate buffers), cold L2; bandwidth from the slope (bytes difference / time difference) using cudaEvent medians | slope of `gpu_sim_cycle` converted at 1755 MHz |
| M4 | Sustained SM clock (MHz) | `clock64` vs `%globaltimer` over ~1 s, light load (1 thread) and full load (all SMs FMA) | n/a (config input) |

Slopes cancel fixed per-kernel costs (launch latency, the final load, clock reads) on both sides. The hardware reports the `%smid` of the chasing thread; the simulator reports its shader ID.

Protocol: hardware mode runs 5 warm-up + 30 (latency) or 20 (bandwidth) measured reps and reports median and p95. Trace mode runs each measured kernel once, with the same cache preparation. The GPU is dedicated (no co-tenant); check with nvidia-smi before and after.

## Error bands

| Metric | Band (sim vs HW median) | Rationale |
|---|---|---|
| M1, M2 | ±15 % | The CAM study sweeps latency parameters in steps of 2x (0, 0.25L … 4L). A 15 % error in memory latency moves a placement break-even point by well under one sweep step. |
| M3 | ±15 % | Same argument for input-fetch/transport terms; streaming inputs dominate CAM query preparation. |
| M4 | informational | If the sustained clock differs from 1755 MHz by more than 5 %, compare in ns as well as cycles and record the clock in the config notes. |

Out-of-band results are reported, and the cause is investigated before any parameter changes. Parameters are not tuned to favor any CAM placement. Any fitted parameter is then checked on a held-out case: the L2 and DRAM chase at a different footprint or stride.

## Known model limits relevant here

- There's no TLB model: hardware DRAM latency over 256 MiB may include TLB misses the simulator lacks. The 256 MiB footprint is 128 × 2 MiB pages.
- There are 2 L2 partitions. A single warp's latency depends on whether each line's home slice is near or far. Both sides average over random addresses, and SM placement is recorded.
- The simple DRAM model uses a fixed `dram_latency` in core cycles plus one 32 B request per DRAM cycle per channel.
