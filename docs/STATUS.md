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
- [ ] `SM90_H100_PCIe` config on `h100-cam`, each parameter classified
- [ ] Microbenchmarks with warm-up + repeats; hardware timing on dedicated GPU
- [ ] Validation kernels (guide §5): shared-mem producer/consumer, async copy, barrier phases, warp-specialized pipeline
- [ ] `docs/hopper_feature_coverage.md`
- [ ] Error bands defined before any fitting

## Pinned commits

| Repo | Branch | Commit |
|---|---|---|
| cam_with_gpu | main | see `git log` |
| accel-sim-framework2 | h100-cam | `f2d5df6` (upstream `d930ad6` + tracer error-check fix) |
| gpgpu-sim_distribution2 | h100-cam | `91880c5` (= upstream dev, unmodified) |
| NVBit | release | v1.8 |

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
| 2026-09-25 | CAM L/II/readout/fill specified in **GPU core cycles**, old values kept (L = 200) | User choice. On H100 PCIe (1755 MHz) L = 200 cycles = 114 ns vs 177 ns in the old V100-class study: this is a faster CAM than before, labeled as such (guide §3), not the same physical CAM. H100 config's L2/NoC clock is 2x core, so the CAM must not silently tick on the L2 clock (200 L2 cycles = 50 ns). External link delay stays in ns. |
