# H100 migration audit

Status as of 2026-09-23. First deliverable of `guide_h100.md` §1.

## 1. Decision: route

Build the CAM study on **upstream Accel-Sim 2.0** (advertised full Hopper support). The user's older Accel-Sim (V100-era, with CAM work, already on GitHub under `tospe`) is kept only as an optional reference for porting CAM code; it is not a prerequisite and has not been audited here.

Rationale: 2.0 claims TMA, WGMMA, `mbarrier`, spin-loop and cluster support. Backporting those into the old fork would be far larger than porting a (presumably isolated) CAM patch forward. These claims are unverified until `docs/hopper_feature_coverage.md` is filled from tests.

## 2. Repositories and pinned commits

| Component | Repo (origin) | Upstream | Base commit | Work branch |
|---|---|---|---|---|
| Frontend / tracer | `git@github.com:tospe/accel-sim-framework2.git` | accel-sim/accel-sim-framework `dev` | `d930ad6d02c09bb56867132583735aba0389cff4` (2026-08-26) | `h100-cam` |
| Performance model | `git@github.com:tospe/gpgpu-sim_distribution2.git` | accel-sim/gpgpu-sim_distribution `dev` | `91880c53383d5a6a6742bfb1be2c5f34e39c7871` (2026-08-27) | `h100-cam` |
| NVBit | release tarball | NVlabs/NVBit | v1.8 | — |

- `dev` on each origin is an untouched copy of upstream; all changes go on `h100-cam`.
- `upstream` remotes have push disabled.
- `accel-sim-framework2/gpu-simulator/gpgpu-sim` is a symlink to `../../gpgpu-sim_distribution2` (path is gitignored upstream). No source changes made yet; both working trees clean.

## 3. Hardware target

| Property | Value | Class |
|---|---|---|
| SKU | NVIDIA H100 **PCIe** 80 GB (not SXM) | documented (nvidia-smi) |
| SMs | 114 | documented (profiling/README) |
| Max SM / mem clock | 1755 MHz / 1593 MHz | documented (nvidia-smi) |
| Memory | HBM2e, 81559 MiB, ≈2.0 TB/s peak | documented |
| Power limit | 350 W | documented |
| MIG | disabled | documented |
| Driver / UMD | 610.43.02 / CUDA 13.3 | documented |
| Toolkit used | nvcc 12.9.86, GCC 13.4.0 (local, no root) | — |

The GPU is shared: another user's process held ~42 GB during this audit. Hardware timing runs need an idle GPU or at least recorded co-tenancy.

**Config mismatch:** upstream `SM90_H100` models an SXM-class part (132 SMs, 1980 MHz core, HBM3 3106 MHz, 40 memory channels). A PCIe config (114 SMs, 1755 MHz, HBM2e timing/bandwidth) must be derived and every changed parameter classified before any hardware comparison. Upstream Hopper CI traces on H200 and simulates the **H200** config, so the H100 config's correlation evidence is weaker than the release notes suggest.

## 4. Build and toolchain

Scripts: `scripts/env_sim.sh`, `scripts/build_sim.sh`. Reuses `profiling/third_party/toolchain/env` (micromamba). Added to that env for the simulator, each dry-run checked to install only new packages: `zlib`, `cuda-profiler-api=12.9`, `libgl-devel` (+ X11 deps), `cuda-cuobjdump=12.9`, `cuda-nvdisasm=12.9`.

Build notes:
- The conda GCC has its own sysroot; `CPATH`/`LIBRARY_PATH` must point at the env.
- `util/tracer_nvbit/tracer_tool/traces-processing/Makefile` hardcodes system `g++` (8.5, no `std::filesystem`); `build_sim.sh` builds it with GCC 13 instead.
- The interconnect config is opened relative to the CWD: copy configs into each run directory (also required for recording resolved configs).

## 5. Known-good reproduction status

| Step | Status |
|---|---|
| Simulator build | **pass** |
| NVBit tracer build | **pass** |
| Trace of `dep_chain` on H100 PCIe | **pass without** `ALLOW_REG_VAL_TRACING`; **segfault with** it |
| Simulate on `SM90_H100` | **pass** (`results/smoke-dep_chain-sxmcfg/`) — structural only, uncalibrated |
| Old-fork (V100) replay | not attempted; not required |

## 6. Blocking risks

1. **NVBit vs. driver.** NVBit 1.8 (latest) documents support for drivers ≤ 575.xx; this machine has 610.43. Plain tracing works on a trivial kernel, but register-value tracing (needed for TMA descriptors and `mbarrier` operands) segfaults. This blocks tracing the TMA/async-barrier tests (guide §5 rows 3–5). Options: debug the crash in the tracer (`record_reg_vals` path), test NVBit's own `record_reg_vals` sample to see whether the fault is in NVBit or Accel-Sim's tool, or trace on a machine with a ≤575 driver.
2. **PCIe config** does not exist upstream (§3).
3. **Shared GPU**: user reports it will become dedicated; record co-tenancy in each hardware run until then.

## 7. Old CAM implementation (reuse assessment)

Read-only clones in `reference/` (gitignored):

| Repo | Branch | Base (upstream `dev`) | Ahead | Size of change |
|---|---|---|---|---|
| `tospe/accel-sim-framework` | `moecam` (head `a98246f`, 2026-09-22, "Strategies A-F") | `3016c65` 2026-03-24 | 139 commits | 150 lines in `gpu-simulator/` (`trace_driven.cc` +135, CAM opcodes in `trace_opcode.h`/`volta_opcode.h` only); 37K lines in `util/moecam/` (generators, drivers, jobs, analysis, figures) |
| `tospe/gpgpu-sim_distribution` | `moecam` | `6c3cf4f` 2026-01-22 | 20 commits | 2052 lines in 13 files: `cam.{h,cc}` (functional array), `cam_unit` in `l2cache.{h,cc}`, placement/link/stats in `gpu-sim.{h,cc}`, CAM fields in `warp_inst_t`, issue/writeback hooks in `shader.{h,cc}` |

Both bases are **immediately before** Accel-Sim 2.0: Hopper arrived as one squash commit per repo on 2026-08-25 (`e10018b6` in gpgpu-sim, 61 files +6124/−748; `0db0445` in accel-sim). So the old fork is pre-Hopper, and its CAM patch is small and isolated relative to it.

Trial `git apply --3way` onto the pinned 2.0 bases:
- gpgpu-sim: 16 conflict hunks in 5 files; frontend: 1 hunk (`trace_opcode.h`). Most are additive (both sides added enum values, fields, includes).
- **Semantic conflict 1:** 2.0 changed `mem_fetch` to hold a *pointer* to `warp_inst_t` (was a copy). The CAM code stamps lifecycle timestamps on "the carried copy" via `inst_mut()`; on 2.0 this would mutate the shared instruction. Needs per-request event storage.
- **Semantic conflict 2:** `ldst_unit::writeback()` was refactored around `writeback_complete()`; the CAM completion path must be re-hooked there.
- CAM opcodes are registered in the Volta opcode table only; Hopper's table needs them.
- CAM latencies (`-gpgpu_cam_latency`, `_search_latency_*`, `_write/read/mem_latency`) are **integer cycles**; per guide §3 they must be re-expressed in ns and converted per clock domain.

**Recommendation: reuse, as a reviewed port — not a blind merge.** Port onto `h100-cam` as separate commits (guide §2 split: CAM engine, frontend markers, completion/ordering, routing/placement, statistics, tests), resolving the two semantic conflicts explicitly, and audit each against the protocol in the V100 guide rather than assuming the old behaviour was correct. `util/moecam` tooling comes over afterwards; its existing plots must be classified (simulated vs analytical) before reuse.

## 8. CAM inventory (guide §1 list)

To be filled from the port: functional scoring/selection (`cam.cc`), markers (UCAM* opcodes in `trace_driven.cc`), submit/wait semantics, queues/II/backpressure (`cam_unit`), attachment/link model (`gpu-sim.cc`, `l2cache.cc`), ownership/completion (`shader.cc`), stats and scripts (`util/moecam`).

## 9. Missing inputs

- ~~Old CAM fork~~ found: see §7.
- `V100_CAM_PLACEMENT_AGENT_GUIDE.md` and `AGENT_IMPLEMENTATION_GUIDE.md` (A–F schedules, CAM protocol) — referenced by the guide but only on another machine.
- CAM hardware parameters: search latency, initiation interval, capacity, outstanding limit, and the meaning of the 353 ns external delay (one-way vs RTT; serialization included or not).
- ~~Research repo~~: `git@github.com:tospe/cam_with_gpu.git`.

## 10. Next steps

1. Resolve or characterize the register-value tracing crash.
2. Derive a `SM90_H100_PCIe` config on `h100-cam`, parameters classified per guide §3.
3. Add warm-up + repeated timing to microbenchmarks; write the remaining §5 validation kernels (shared-memory producer/consumer, async copy, barrier phases, warp-specialized pipeline).
4. Start `docs/hopper_feature_coverage.md` from what those traces actually exercise.
