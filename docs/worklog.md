# Worklog

Newest first. One entry per working session: what was done, what was executed vs only inspected, commits, and open issues.

## 2026-09-25 (calibration)

User's 6-step fit procedure followed; full account in `results/calib-fit/REPORT.md`.
- HW: 1755 MHz; L2 272.4, DRAM 658.0 cycles/hop; BW 1962 GB/s; DRAM latency flat vs footprint (no TLB effect).
- dram_latency 283 (frozen before held-out): fit −2.5 %, held-out −1.5..−3.1 %, L2 unchanged, BW −5.8 %; concurrency −31.7 % (FAIL).
- Cause: HW per-access DRAM latency is widely spread (p10 567, p50 715, p99 1038); warps wait on slowest lane. Simple DRAM model has no variance. Detailed DRAM model reproduces it (+3 %) but has half the bandwidth (−57.5 %).
- Fixed: false deadlock (gpgpu-sim b1ec7f13). Incidents: lost BW run (script edited while running -> scripts now function-wrapped); invalid first lat_hist (compiler hoisting, fixed + SASS-checked).
- Config file unchanged; model choice pending with user.

## 2026-09-25

Decisions: CAM params in GPU core cycles (L = 200); GPU confirmed dedicated (idle in nvidia-smi).

Executed:
- `device_props.cu`: H100 PCIe = 114 SMs, 1755/1593 MHz, 5120-bit HBM2e, 50 MiB L2, 228 KiB smem/SM.
- `SM90_H100_PCIe` config (3 changes vs SM90_H100: SM count, clocks, icnt k), both repos; smoke run `results/smoke-dep_chain-pciecfg/`.
- Read the DRAM model source: simple DRAM model is active upstream; peak BW = n_mem x 32 B x DRAM clock.
- Investigated chase speed-up on PCIe cfg with one-factor swaps: SM-placement x L2-partition x memcpy pre-fill effect (not the config values).

Inspected only: upstream H200 config diff.

Open: calibration (dram_latency, l2_rop_latency, sustained clock) needs hardware runs with warm-up; issuer placement must be controlled.

## 2026-09-24

Executed:
- Guides added by user; renamed `docs/GUIDE_*`, removed duplicate root `guide_h100.md`.
- CAM parameters recovered from old study (`util/moecam/DSA-PLACEMENT.md`); user says they were chosen without a firm basis, so all are hypothetical (STATUS table). Proposal: sweep them and report break-even points.
- Tracer crash diagnosed: NVBit `record_reg_vals` sample works; matrix of `ALLOW_REG_VAL_TRACING` x `SPINLOCK_HANDLING_MODE` showed the crash only with spinlock mode 1/2, i.e. missing spinlock detection data -> null map entry deref at `tracer_tool.cu`. Proper detection-then-trace flow works with reg values.
- Fix on `accel-sim-framework2/h100-cam` `f2d5df6` (clear error instead of segfault; formatted with clang-format 16.0.6).
- `scripts/trace_app.sh` (detection + trace + post-process + metadata); `results/smoke-dep_chain-regval-sxmcfg/`.

Earlier red herring: a gdb run without creating `TRACES_FOLDER` segfaulted in the memcpy callback (unchecked `fopen`), a separate upstream robustness issue, not the reported crash.

Open: NVBit 1.8 officially supports drivers <= 575 (we have 610) — not a problem so far; GPU still had a co-tenant at 100% util.

## 2026-09-23

Executed:
- Created `tospe/accel-sim-framework2` and `tospe/gpgpu-sim_distribution2` from upstream `dev`; `h100-cam` work branches; upstream push disabled.
- Built simulator and NVBit tracer with the local toolchain (added zlib, cuda-profiler-api, libgl-devel, cuobjdump, nvdisasm to `profiling/third_party/toolchain/env`).
- `dep_chain` traced on H100 PCIe and simulated on upstream `SM90_H100` (SXM) config: structural pass, uncalibrated.
- Created `tospe/cam_with_gpu` with submodules; wrote the audit.
- Cloned old `moecam` forks to `reference/`; trial-applied the CAM patch onto 2.0 (16 + 1 conflict hunks, 2 semantic).

Inspected only: Accel-Sim 2.0 release notes and CI (H200-traced/simulated).

Open issues: `ALLOW_REG_VAL_TRACING=1` segfaults; no PCIe config; hardware timings lack warm-up.
