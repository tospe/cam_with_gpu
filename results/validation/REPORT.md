# Validation kernels (guide §5; user step 2), 2026-09-25

Config: `SM90_H100_PCIe_dev`. Code: `microbenchmarks/hopper_validation/pipeline.cu` (`cpasync`, `tma`) and `calib.cu fma_lat`. Traces were made with `scripts/trace_app.sh` (spinlock detection + register values). Hardware GPU dedicated (idle before and after).

## What the kernels exercise (SASS verified)

- `cpasync`: `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, `DEPBAR.LE SB0`, `BAR.SYNC`. Global→shared async copy, consumer wait, cross-thread visibility, buffer reuse guarded by a barrier.
- `tma`: `UBLKCP.S.G` (bulk copy), `SYNCS.ARRIVE.TRANS64[.A1T0]`, `SYNCS.PHASECHK.TRANS64.TRYWAIT`, `SYNCS.EXCH.64`. A warp-specialized ring: 1 producer + 4 consumer warps, S slots, full barriers (arrival + byte count) and empty barriers (4 arrivals). Each slot's barriers go through 64/S phases (up to 64 phases at S = 1).

## Structural / progress results: all PASS

| Check | Result |
|---|---|
| Functional correctness on HW (distinct data per tile, exact host checksum) | 0 mismatches in every configuration, 30 reps each, warm and cold |
| Correctness under tracing | 0 mismatches (all 7 traces) |
| Progress in simulator | all runs exit 0, no deadlock |
| Traffic generated in simulator (cold) | DRAM reads = 933,888 = exactly the number of 32 B data sectors |
| Waits and buffer reuse enforced in simulator | time falls with ring depth (1 slot 48.4 → 8 slots 18.9 µs) and rises with work, as on HW |

## Timing (cold L2 on both sides)

HW cold = before every launch, memset a 256 MiB buffer and then stream-read it (clean L2). Sim = `-gpgpu_perf_sim_memcpy 0`.

| Case | HW µs | Sim µs | Error | Band ±15 % |
|---|---|---|---|---|
| TMA ring, 1 slot | 48.58 | 48.37 | −0.4 % | ✓ |
| TMA ring, 2 slots | 29.44 | 26.90 | −8.6 % | ✓ |
| TMA ring, 4 slots | 22.75 | 19.57 | −14.0 % | ✓ (borderline) |
| TMA ring, 8 slots | 21.79 | 18.91 | −13.2 % | ✓ |
| cp.async, work 0 | 43.26 | 34.89 | −19.3 % | ✗ |
| cp.async, work 200 | 48.48 | 61.70 | +27.3 % | ✗ |
| TMA ring 2 slots, work 200 | 57.44 | 67.12 | +16.9 % | ✗ |

### Cause of the work-case errors: dependent-arithmetic latency

`calib fma_lat` (1 thread, dependent `fmaf` chain, slope of 4096 vs 16384 iterations):
**HW 4.438 cycles/iteration; sim 7.000 (+57.7 %)**. The trace config sets SP latency 4 (`-trace_opcode_latency_initiation_sp 4,2`); the simulator's issue → operand-collect → execute → writeback → scoreboard path adds about 3 cycles to each dependent step. This is guide §5 test row 1 (dependent arithmetic): **FAIL**. The added "work" in the validation kernels is exactly such a chain, which explains why sim is slower than HW by about 12 µs in those cases.

The cp.async work-0 case (−19.3 %) is not explained by this and is still open.

## Incidents (recorded honestly)

- **Dirty-L2 flush.** The first cold procedure used only a memset, which leaves up to 50 MiB of dirty lines that were written back during the measured kernel. That inflated the cold HW times by about 10 µs (`hw_cold.txt`, superseded by `hw_cold_clean.txt`). I checked whether the same flaw affected calibration: M2 661.5 vs 661.2, concurrency 1106.7 vs 1104.5, BW slope 1987 vs 1903 GB/s (overlapping). **The calibration and the 283 fit are unaffected** (`results/calib-fit/hw_flush_comparison.txt`). The warm results (`hw.txt`) were mostly L2 hits on HW but ~50 % DRAM in sim (the memcpy pre-fill covers only the home L2 half), so they are not a like-for-like comparison.

## Status

Required primitives are **functionally and structurally validated** (correctness, progress, traffic, wait and reuse ordering). **Timing is not**: memory-bound TMA pipelines are within band, but dependent-arithmetic latency is +58 % and cp.async work-0 is −19 %.
