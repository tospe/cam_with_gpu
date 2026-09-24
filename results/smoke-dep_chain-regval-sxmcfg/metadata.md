# Run: smoke-dep_chain-regval-sxmcfg (2026-09-23)

- Purpose: confirm register-value tracing (trace version 6) + spinlock mark_region works end to end after the tracer crash was diagnosed. NOT a calibration run.
- Trace: `bash scripts/trace_app.sh traces/dep_chain ./dep_chain` (spinlock detection phases, then SPINLOCK_HANDLING_MODE=2, ALLOW_REG_VAL_TRACING=1). See `trace_metadata.txt`.
  Tracer source at trace time = `f2d5df6` content (metadata says d930ad6 "dirty": the fix was committed right after).
- GPU during tracing: co-tenant at 100% util, ~22 GB used. Tracing-time timings are meaningless anyway (instrumented).
- Simulator: accel-sim-framework2 `f2d5df6` (tracer-only change; simulator code = d930ad6) + gpgpu-sim_distribution2 `91880c5`.
- Config: upstream SM90_H100 (SXM), same files as `smoke-dep_chain-sxmcfg`.

| kernel | gpu_sim_cycle | insn |
|---|---|---|
| vecadd | 4030 | 1,245,184 |
| chase | 130349 | 600 |

Matches the non-reg-val trace (4029 / 130350, identical instruction counts). Status: structural pass, uncalibrated.
