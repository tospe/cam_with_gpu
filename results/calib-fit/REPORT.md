# Calibration fit report: `-dram_latency` on SM90_H100_PCIe (2026-09-25)

The procedure followed the user's six steps. Error bands come from `docs/calibration_plan.md` (±15 %), which was written before any comparison. The candidate was frozen in `FROZEN.md` (commit `562a007`) before any held-out case ran.

## 1. Baseline preserved

`results/calib-v1/`: HW measurements and the baseline simulation (upstream-derived `SM90_H100_PCIe`, `dram_latency 194`). The config file is **unchanged**: every fit run used command-line overrides, recorded in each `sim_metadata.txt`.

| Metric | HW median | Sim baseline | Error |
|---|---|---|---|
| M1 L2-hit latency (cycles/hop) | 272.4 | 255.0 | −6.4 % ✓ |
| M2 DRAM latency, cold (cycles/hop) | 658.0 | 552.4 | −16.1 % ✗ |
| M3 DRAM read BW, slope (GB/s) | 1961.8 | 1868 | −4.8 % ✓ |
| M4 sustained SM clock | 1755 MHz | config 1755 | — |

## 2. Parameter verified in our fork

- `-dram_latency` is used at `l2cache.cc:321` (simple model, active) and at `:386` (detailed model, inactive).
- It is in **core cycles**: `ready_cycle` uses `gpu_sim_cycle`, which advances on CORE ticks (`gpu-sim.cc:2362`). Enqueue and dequeue run on DRAM ticks (1593 MHz), so the effective delay rounds up by ≤ ~1.1 core cycles.
- It applies once per L2→DRAM request. The chase does exactly 1 DRAM read per hop (256 / 1280 reads).

## 3. Sensitivity

`dram_latency` 194 → 552.36 cycles/hop; 214 → 572.22, i.e. **+19.86 for +20**. The response is linear. The rerun at 194 on the fixed binary matched the original cycle for cycle.

## 4. Candidate 283 (fit case only, then frozen)

Fit case (256 MiB, 256 B stride, seed 1): **641.29 (−2.53 %)**. It was not tuned further.

Held-out cases (reserved in `FROZEN.md`, simulated only after the freeze):

| Case | HW | Sim 194 | Sim 283 |
|---|---|---|---|
| H1 1 GiB footprint | 665.0 | 555.3 (−16.5 %) | **644.1 (−3.1 %) ✓** |
| H2 128 B stride | 658.5 | 559.6 (−15.0 %) | **648.6 (−1.5 %) ✓** |
| H3 seed 2 | 660.7 | 552.1 (−16.4 %) | **641.0 (−3.0 %) ✓** |

Disclosure: the hardware H1 footprint (1 GiB) had been seen before the freeze, in the TLB footprint sweep (666.9). It was not used for fitting and was re-measured after the freeze (665.0).

## 5. Regressions

| Check | Result |
|---|---|
| M1 L2-hit latency at 283 | 255.00, **identical** to baseline ✓ |
| M3 bandwidth at 283 | 1848 GB/s, **−5.8 %** ✓ (baseline 1868) |
| Concurrency: 114×32 = 3648 random chases, 1 GiB, cold | HW 1108.6; sim 194 → 669.0 (−39.7 %); **sim 283 → 757.7 (−31.7 %) ✗** |

## 6. Failure investigation (the model, not more tuning)

**Hardware shows the loaded slowdown comes mostly from warps waiting on their slowest lane, not DRAM queueing.**

Same chains, different lanes per warp (`hw_lanes_test.txt`):

| Chains | Lanes/warp | HW cycles/hop |
|---|---|---|
| 3648 | 1 | 762 |
| 3648 | 8 | 955 |
| 3648 | 32 | 1107 |
| 912 | 1 | 781 |
| 912 | 8 | 983 |

Concurrency sweep, 32 lanes/warp (`hw_conc_sweep.txt`): 114 → 699, 456 → 893, 912 → 982, 1824 → 1006, 3648 → 1111, 7296 → 1265.

Per-access DRAM latency on HW (single thread, cold, 40,960 hops, `hw_lat_hist.txt`): p1 553, p10 567, **p50 715**, p90 791, p99 1038, max 1524. The distribution looks bimodal (~560 / ~715), consistent with near/far L2 partition or row hit/miss. The expected maximum over N draws is ≈ 775 (N=4), 789 (8), 915 (32), which explains most of the lanes effect.

The simple DRAM model uses a single fixed latency, so the maximum over N lanes equals the mean. Its loaded result at 283 (758) matches the HW **1-lane** case (762), so it captures queueing but not variance.

Detailed DRAM model (`-gpgpu_simple_dram_model 0`, inherited H200 timings at 1593 MHz, untuned):

| Case | Detailed 194 | Detailed 283 |
|---|---|---|
| Fit case (unloaded) | 618.1 (−6.1 %) ✓ | 707.1 (+7.5 %) ✓ |
| Concurrency 3648 | 1141.7 (+3.0 %) ✓ | 1183.1 (+6.7 %) ✓ |
| Bandwidth | **834 GB/s (−57.5 %) ✗** | not run |

The detailed model produces the variance but has about half the bandwidth: its data bus (`dram_buswidth 8`, `BL 4`, ratio 2) delivers 16 B per DRAM cycle per channel, and its bank timings are H200 HBM3e values interpreted at the HBM2e clock.

## Status

| Configuration | M1 | M2 + held-out | M3 | Concurrency |
|---|---|---|---|---|
| simple, 194 (upstream) | ✓ | ✗ | ✓ | ✗ |
| **simple, 283 (frozen candidate)** | ✓ | ✓ | ✓ | ✗ −31.7 % |
| detailed, 194 | not run | ✓ (fit case only) | ✗ −57.5 % | ✓ |

No configuration passes everything. The config file has **not** been changed. That decision is left to the user (see the options in STATUS).

## Methodology incidents

- The first `lat_l2` simulation aborted on a **false deadlock** in the simulator (fixed, gpgpu-sim `b1ec7f13`; timing unaffected).
- The first bandwidth result was **lost**: `sim_run.sh` was edited while running, and bash re-executed part of it. The scripts now parse fully before running.
- The first `lat_hist` measurements were **invalid** (2 cycles): the compiler moved the load out of the timed window. They were fixed with opaque data dependencies and checked in SASS.
- One batch of hardware runs failed to start (missing runtime library path) and was rerun.
