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
- [ ] Register-value tracing crash (NVBit 1.8 vs driver 610): diagnosed / worked around
- [ ] `SM90_H100_PCIe` config on `h100-cam`, each parameter classified
- [ ] Microbenchmarks with warm-up + repeats; hardware timing on dedicated GPU
- [ ] Validation kernels (guide §5): shared-mem producer/consumer, async copy, barrier phases, warp-specialized pipeline
- [ ] `docs/hopper_feature_coverage.md`
- [ ] Error bands defined before any fitting

## Pinned commits

| Repo | Branch | Commit |
|---|---|---|
| cam_with_gpu | main | see `git log` |
| accel-sim-framework2 | h100-cam | `d930ad6` (= upstream dev, unmodified) |
| gpgpu-sim_distribution2 | h100-cam | `91880c5` (= upstream dev, unmodified) |
| NVBit | release | v1.8 |

## Waiting on user

- [x] Guides in `docs/` (2026-09-24): V100 placement (protocol, A–F), implementation (profiling baseline), H100 extension; files renamed `GUIDE_*`, root `guide_h100.md` duplicate removed
- [ ] Confirm the old study's CAM parameters carry over unchanged in ns (table below)
- [ ] GPU dedicated (no co-tenant) before hardware timing runs

## CAM parameters (from old `moecam` study, `util/moecam/DSA-PLACEMENT.md` §3)

Old values were in cycles at the V100-class 1.132 GHz clock; ns is the quantity to hold fixed on H100 (guide §3).

| Parameter | Old value | In ns | Provenance |
|---|---|---|---|
| Search latency L | 200 cycles | 177 | user's headline estimate (8–20x above circuit estimate, old `TECHNOLOGY.md`) |
| Readout | 1 result/cycle | 0.88 per result | hypothetical |
| Initiation interval II | 8 cycles | 7.1 | hypothetical, swept |
| Fill | 21 cycles / 256 B row | 18.6 per row | user's UCAMF model |
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
