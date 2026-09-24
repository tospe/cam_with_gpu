# Status

Single place to see where the H100 CAM study stands. Update this file in the same commit as the work it describes. Detailed history goes in `worklog.md`.

## Milestones (guide_h100.md)

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

- [ ] `V100_CAM_PLACEMENT_AGENT_GUIDE.md` and `AGENT_IMPLEMENTATION_GUIDE.md` copied into the repo (CAM protocol, A–F schedules)
- [ ] CAM parameters in ns; is 353 ns one-way or round-trip, and does it include serialization?
- [ ] GPU dedicated (no co-tenant) before hardware timing runs

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-23 | Build on upstream Accel-Sim 2.0, not the old fork | 2.0 adds Hopper; old fork is pre-2.0 |
| 2026-09-23 | Reuse old `moecam` CAM code as a reviewed port | Patch is small and isolated; 2 semantic conflicts to resolve (audit §7) |
| 2026-09-23 | Separate `*2` repos, `dev` = upstream mirror, work on `h100-cam` | Keep old repos intact; one fork per upstream per account |
| 2026-09-23 | No Claude co-author trailers | User preference |
