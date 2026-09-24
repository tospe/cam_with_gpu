# Run: smoke-dep_chain-sxmcfg (2026-09-23)

- Purpose: end-to-end pipeline smoke test (trace -> post-process -> simulate). NOT a calibration run.
- Binary: microbenchmarks/hopper_validation/dep_chain.cu, `nvcc -O3 -arch=sm_90a -lineinfo`, nvcc 12.9.86, GCC 13.4.0; default args (n=65536, hops=256)
- Hardware traced on: H100 PCIe 80GB, bus 3D:00.0, driver 610.43.02, MIG off; GPU shared with another process (~42 GB in use)
- Tracer: accel-sim-framework2 @ d930ad6 util/tracer_nvbit, NVBit 1.8 (officially supports driver <= 575)
  env: NVBIT_INSTRUMENTATION_ENABLED=1, SPINLOCK_HANDLING_MODE default, ALLOW_REG_VAL_TRACING unset (=1 segfaults)
- Trace: traces/dep_chain_noreg/traces/kernelslist.g (.tracez)
- Simulator: accel-sim-framework2 @ d930ad6 + gpgpu-sim_distribution2 @ 91880c5, unmodified
- Config: upstream SM90_H100 (SXM: 132 SMs, 1980 MHz core) — copied here unchanged. Does NOT match the PCIe target (114 SMs, 1755 MHz max).
- Command: accel-sim.out -trace ../../traces/dep_chain_noreg/traces/kernelslist.g -config gpgpusim.config -config trace.config
- Wall time 18 s. Output: sim.log

| kernel | gpu_sim_cycle | insn | simulated time @1980 MHz |
|---|---|---|---|
| vecadd | 4029 | 1,245,184 | 2.03 us |
| chase (256 dependent hops) | 130350 | 600 | 65.8 us (~257 ns/hop) |

Hardware cudaEvent times from the untraced run (vecadd 3.08 ms, chase 0.10 ms) include first-launch/module-load overhead and no warm-up; not comparable. Status: structural pass, timing uncalibrated.
