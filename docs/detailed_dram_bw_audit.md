# Bounded audit: detailed DRAM model bandwidth configuration (SM90_H100_PCIe*)

Date: 2026-09-25. Scope (user step 5): establish the implied bandwidth ceiling from the configuration and code **before** any DRAM timing parameter is changed. **No parameters were changed.**

Evidence comes from the detailed-model bandwidth run `results/calib-fit/inv_bw_detailed_dram194/` (resolved config in its `sim.log`) and from gpgpu-sim `src/gpgpu-sim/dram.cc`, `gpu-sim.h` at the pinned fork.

## Configuration as resolved

| Item | Value | Where |
|---|---|---|
| Memory channels (`-gpgpu_n_mem`) | 40 | config |
| L2 sub-partitions per channel | 2 (80 L2 slices) | config |
| Chips per controller | 1 | config |
| Data bus width per channel (`-gpgpu_dram_buswidth`) | 8 B = 64 bit | config |
| Burst length (`-gpgpu_dram_burst_length`) | 4 | config |
| Data:command clock ratio (DDR factor) | 2 | config |
| DRAM command clock | 1593 MHz | config (HBM2e clock, documented) |
| Dual bus interface (row + column command per cycle) | 1 | config |
| `dram_atom_size` = BL × busW × chips | 32 B | `gpu-sim.h:296` |
| Request size reaching DRAM | **32 B** (4,194,304 reads for 128 MiB) | run log |
| Timing | nbk 16, nbkgrp 4, tCCD 1, tCCDL 6, tRCD 19, tRP 19, tRAS 45, tRC 50, CL 19 … (inherited from SM90_H100 = SM90_H200 HBM3e values) | config |
| Bank-group index | lower bank bits (`bank & 3`, policy 1) | `dram.cc:872` |

## How the code limits bandwidth

- Each column command moves `dram_atom_size` = 32 B (`dram.cc:570`).
- A column command may issue only when `CCDc == 0` (tCCD since the last column command in the channel) and the bank group's `CCDLc == 0` (tCCDL since the last one in the same group) (`dram.cc:562`–`572`), besides bank/row readiness.
- The data-bus burst time (BL / ratio = 2 cycles) is **only accumulated into `bwutil`** (`dram.cc:582`). It does **not** block issue. So `bwutil` is normalized to a 16 B/cycle bus that the issue logic does not enforce.

## Implied ceilings (40 channels × 1593 MHz)

| Bound | Column cmds / cycle / channel | B / cycle / channel | GPU total |
|---|---|---|---|
| tCCD = 1 | 1 | 32 | **2.04 TB/s** |
| tCCDL = 6, 4 bank groups perfectly interleaved | 4/6 | 21.3 | **1.36 TB/s** |
| tCCDL = 6, consecutive commands in one bank group | 1/6 | 5.3 | 0.34 TB/s |
| Data-bus normalization used by `bwutil` (not enforced) | 1/2 | 16 | 1.02 TB/s |

Observed (128 MiB read kernel): 898 GB/s kernel-average, 834 GB/s slope. `bwutil` = 0.88 (0.44 column cmds/cycle/channel). `CCDLc_limit` = 710,001 blocked cycles (same-bank-group spacing), `Row_Buffer_Locality` 0.82. **The binding limit in this run is tCCDL with imperfect bank-group interleaving**, below even the 1.36 TB/s perfect-interleave bound.

## Comparison with the physical part (documented)

H100 PCIe: 5 HBM2e stacks, 5120-bit bus, 1593 MHz (DDR ≈ 3.19 Gb/s/pin) → **2.04 TB/s**. In pseudo-channel mode this is 80 × 64-bit pseudo-channels at 25.5 GB/s each.

The detailed model has **40** channels of 64 bit. Counting pins, it models half the pseudo-channels. It reaches 2.04 TB/s only through tCCD = 1 (32 B per command clock per 64-bit channel), which is twice what a 64-bit DDR bus can carry at BL4 (16 B per clock). So the structure (channel count × bus width) and the timing (tCCD, tCCDL, taken from HBM3e at a different clock) are not jointly consistent with HBM2e.

The simple model (development config) reaches 2.04 TB/s by construction: one 32 B request per DRAM cycle per channel × 40 × 1593 MHz.

## Implications before any calibration of the detailed model

1. First fix the structure: channel/pseudo-channel count and bus width so that the pin-level peak equals 2.04 TB/s. For example, 80 × 64-bit, or 40 × 128-bit, which changes the atom size. Also check the L2 slice mapping (80 sub-partitions today).
2. Only then take tCCD/tCCDL/tRCD/tRP/tRAS/tRC/CL from an HBM2e source, converted to 1593 MHz clocks, and record each one's provenance.
3. Only then fit, with a frozen candidate and held-out cases, as for `dram_latency`.

Not done now, per user decision C. Required before claims about scattered gathers, tail latency or contention.
