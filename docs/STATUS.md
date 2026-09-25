# Status

Single place to see where the H100 CAM study stands. Update this file in the same commit as the work it describes. Detailed history goes in `worklog.md`.

## Milestones (GUIDE_H100_CAM_EXTENSION_AGENT.md)

| Milestone | Meaning | Status |
|---|---|---|
| Audit | `docs/h100_migration_audit.md` | done (2026-09-23) |
| H0 | pinned build, identified target, feature-coverage table, reproducible baseline tests | in progress |
| H1 | CAM correctness/progress tests pass on H100 config; CAM-disabled regressions pass; one request shows expected component timing | not started |
| H2 | reproducible Hopper A–F result with trace, event and resource evidence | not started |
| H3 | TMA / DSM / fusion contributions beyond H2 | not started |

## H0 checklist

- [x] Simulator + tracer build (`scripts/build_sim.sh`)
- [x] Structural smoke run (`results/smoke-dep_chain-sxmcfg/`), uncalibrated
- [x] Tracer crash: spinlock mode without detection data, fixed `f2d5df6`; use `scripts/trace_app.sh`
- [x] `SM90_H100_PCIe` config (gpgpu-sim `fa2f6f39`, accel-sim `a238d6a`); changed params classified in its header; rest inherited/uncalibrated
- [x] Calibration microbenchmarks + HW timing (`calib.cu`, `results/calib-v1`, `results/calib-fit/REPORT.md`)
- [x] DRAM model decision (2026-09-25, user: option C, restricted scope): development config **`SM90_H100_PCIe_dev`** = simple DRAM model + `dram_latency 283`. Baseline `SM90_H100_PCIe` kept.
- [~] Validation kernels (`results/validation/REPORT.md`): async copy (cp.async), TMA bulk, repeated mbarrier phases, buffer reuse, producer/consumer — **functional + structural PASS** (0 mismatches HW/trace, progress, exact DRAM traffic, ordering enforced); **timing**: TMA ring −0.4..−14 % ✓, cp.async −19 %/+27 % ✗, dependent FMA latency **+57.7 % ✗** (sim 7.0 vs HW 4.44 cycles)
- [ ] Random-access diagnosis: sweep active lanes and resident warps separately (elapsed, loads/s, transactions, outstanding); lanes hypothesis unconfirmed until then
- [x] Bounded audit of detailed DRAM model bandwidth (`docs/detailed_dram_bw_audit.md`): binding limit tCCDL + bank-group interleave (observed 0.90 TB/s; ceilings 2.04 / 1.36 / 1.02 TB/s); structure models 40 of 80 HBM2e pseudo-channels. No parameters changed.
- [ ] Before H2: ordinary-memory workload shaped like the CAM interface (query prep, contiguous transfers, completion sync, consumption), swept over concurrency and buffer depth with competing traffic
- [ ] Control issuer SM / L2 partition and cache state (memcpy pre-fill) in tests — see `results/smoke-dep_chain-pciecfg/metadata.md`
- [ ] Validation kernels (guide §5): shared-mem producer/consumer, async copy, barrier phases, warp-specialized pipeline
- [ ] `docs/hopper_feature_coverage.md`
- [x] Error bands defined before any fitting (`docs/calibration_plan.md`)

## Pinned commits

| Repo | Branch | Commit |
|---|---|---|
| cam_with_gpu | main | see `git log` |
| accel-sim-framework2 | h100-cam | see submodule (upstream `d930ad6` + tracer fix + PCIe trace.configs) |
| gpgpu-sim_distribution2 | h100-cam | see submodule (upstream `91880c5` + PCIe configs + false-deadlock fix) |
| NVBit | release | v1.8 |

## Standing restrictions

- `SM90_H100_PCIe_dev` fails the concurrent random-access check (−31.7 %). No claims about scattered gathers, memory-tail latency or contention that it fails to reproduce until the detailed DRAM model is calibrated (user, 2026-09-25).

## Waiting on user

- [x] Guides in `docs/` (2026-09-24): V100 placement (protocol, A–F), implementation (profiling baseline), H100 extension; files renamed `GUIDE_*`, root `guide_h100.md` duplicate removed
- [x] CAM parameter strategy (2026-09-25): keep the old **cycle** counts (L = 200 etc.), in GPU core cycles; all hypothetical; sweep and report break-even points
- [x] GPU dedicated (2026-09-25: user confirms; nvidia-smi idle, no processes)

## CAM parameters (from old `moecam` study, `util/moecam/DSA-PLACEMENT.md` §3)

Old values were in cycles at the V100-class 1.132 GHz clock. **User (2026-09-24): these values were chosen without a firm basis — treat every row as hypothetical.** Old `TECHNOLOGY.md` cites TCAM macros at 0.2–1.6 ns per subarray search (~10–25 cycles incl. merge), but those are exact/Hamming match, not FP16 H×D dot-product scoring + top-k, so they do not bound L for the DSA workload.

| Parameter | Old value | In ns | Provenance |
|---|---|---|---|
| Search latency L | 200 cycles | 177 | hypothetical (old headline; conservative vs TCAM literature, which does not cover dot-product scoring) |
| Readout | 1 result/cycle | 0.88 per result | hypothetical |
| Initiation interval II | 8 cycles | 7.1 | hypothetical, swept |
| Fill | 21 cycles / 256 B row | 18.6 per row | hypothetical (old UCAMF model) |
| Link bandwidth | 1024 / 150 GB/s per direction | — | hypothetical |
| Added external delay | 0, 88, 177, 353, 707, 1200 ns **round trip** (main 353 = 2L) | — | hypothetical sensitivity points; serialization modeled separately |
| Resident capacity | 8192 x 2048 b = 2 MB | — | controlled assumption |

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-23 | Build on upstream Accel-Sim 2.0, not the old fork | 2.0 adds Hopper; old fork is pre-2.0 |
| 2026-09-23 | Reuse old `moecam` CAM code as a reviewed port | Patch is small and isolated; 2 semantic conflicts to resolve (audit §7) |
| 2026-09-23 | Separate `*2` repos, `dev` = upstream mirror, work on `h100-cam` | Keep old repos intact; one fork per upstream per account |
| 2026-09-23 | No Claude co-author trailers | User preference |
| 2026-09-25 | DRAM model: option C — develop on simple model + dram_latency 283 (`SM90_H100_PCIe_dev`), calibrate detailed model only if needed | Passes latency/held-out/BW; fails concurrency (no per-access variance); detailed model fails BW (−57.5 %) |
| 2026-09-25 | CAM L/II/readout/fill specified in **GPU core cycles**, old values kept (L = 200) | User choice. On H100 PCIe (1755 MHz) L = 200 cycles = 114 ns vs 177 ns in the old V100-class study: this is a faster CAM than before, labeled as such (guide §3), not the same physical CAM. H100 config's L2/NoC clock is 2x core, so the CAM must not silently tick on the L2 clock (200 L2 cycles = 50 ns). External link delay stays in ns. |
