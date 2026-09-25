# Frozen candidate: -trace_opcode_latency_initiation_sp 3,2 (on SM90_H100_PCIe_dev)

Frozen 2026-09-25T17:13:05-04:00, BEFORE any reserved arithmetic case was measured on hardware or simulated.

- FFMA maps to SP_OP (accel-sim ISA_Def/hopper_opcode.h:25) -> latency/initiation from -trace_opcode_latency_initiation_sp
  (dev config: 4,2). Compiled chain (calib.cu fma_chain): 16 dependent FFMA + IADD3/ISETP/BRA per unrolled iteration.
- Fit case (only case used): 1-thread dependent FFMA chain, slope 4096 vs 16384 iterations. HW 4.438 cycles/iter.
- Sensitivity (initiation 2 fixed): latency 5 -> 8.000, 4 -> 7.000, 3 -> 6.000, 2 -> 6.000, 1 -> 6.000 cycles/iter.
  With initiation 1 (NOT allowed during the fit: execution resource): latency 1 -> 5.000.
  => dependent step = max(latency + 3, initiation + 4); floor 6.0 with initiation 2.
- Candidate: latency 3 (smallest departure from the physical 4 that reaches the floor). Fit case 6.000, error +35.2 %: OUTSIDE ±15 %.
  The constrained fit cannot pass; the arithmetic latency check FAILS under the no-resource-change rule.

Reserved cases (defined now):
  R1 dependent FFMA chain slopes at reserved lengths: 1024 vs 8192, 2048 vs 32768 iterations.
  R2 independent FFMAs: accumulators K in {2, 8} x resident warps/SM W in {4, 32} (114 blocks, 1 per SM),
     metric cycles per loop iteration (slope 1024 vs 4096) -> dependency latency (K=2, W=4) and throughput (K=8, W=32).
  R3 arithmetic producer/consumer ring: pipeline tma slots=2 work=200, cold L2 (existing trace + HW clean-cold).
Band ±15 % per metric.
