# Worklog

Newest first. One entry per working session: what was done, what was executed vs only inspected, commits, and open issues.

## 2026-09-23

Executed:
- Created `tospe/accel-sim-framework2` and `tospe/gpgpu-sim_distribution2` from upstream `dev`; `h100-cam` work branches; upstream push disabled.
- Built simulator and NVBit tracer with the local toolchain (added zlib, cuda-profiler-api, libgl-devel, cuobjdump, nvdisasm to `profiling/third_party/toolchain/env`).
- `dep_chain` traced on H100 PCIe and simulated on upstream `SM90_H100` (SXM) config: structural pass, uncalibrated.
- Created `tospe/cam_with_gpu` with submodules; wrote the audit.
- Cloned old `moecam` forks to `reference/`; trial-applied the CAM patch onto 2.0 (16 + 1 conflict hunks, 2 semantic).

Inspected only: Accel-Sim 2.0 release notes and CI (H200-traced/simulated).

Open issues: `ALLOW_REG_VAL_TRACING=1` segfaults; no PCIe config; hardware timings lack warm-up.
