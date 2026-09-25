# Frozen candidate: -trace_opcode_latency_initiation_sp 3,1 (on SM90_H100_PCIe_dev)

Frozen 2026-09-25T18:09:10-04:00, BEFORE any fresh case (F1-F3) was measured or simulated.

Change vs previous frozen candidate (3,2): FP32 (SP_OP) initiation interval 2 -> 1. SP latency kept at 3.
Execution-unit counts (-gpgpu_num_sp_units 4, sub_core_model 1), pipeline widths and memory settings unchanged.

What the interval controls (our fork):
- Each of the 4 SP units (one per sub-core) has a dispatch register; an instruction entering it gets
  cycles = initiation_interval (gpgpu-sim abstract_hardware_model.cc:64) and stays until dispatch_delay() counts it
  down (abstract_hardware_model.h:1683); can_issue() requires the register empty (shader.h:1539). => one warp
  instruction (32 threads) per unit every  cycles; completion at dispatch + latency -
  initiation_interval (shader.cc:2824).
- Peak FFMA instruction throughput = 4 x 32 / interval per SM per cycle: 64 (interval 2) vs 128 (interval 1).
Supporting documentation: CUDA C++ Programming Guide, arithmetic instruction throughput table, compute capability
9.0: 128 results/clock/SM for 32-bit floating-point add, multiply, multiply-add; NVIDIA H100 architecture
description: 128 FP32 cores per SM (4 processing blocks x 32). HW measured 111.5 FFMA/cycle/SM (R2 K2 W32).
The configured interval is a model parameter consistent with that documented rate, NOT a measurement of a physical
H100 pipeline stage.

Convention: throughput counts FFMA instructions per thread per cycle per SM; 1 FFMA = 2 FLOP.

Expected from the sensitivity rule max(latency + 3, interval + 4): dependent step = max(6, 5) = 6 cycles (HW 4.438),
so dependency-limited cases are predicted to remain outside the band.

Fresh validation set (unused K/W combinations; 114 blocks = 1 per SM; slope 1024 vs 4096 iterations):
  F1 dependency-limited: K=1,  W=2
  F2 intermediate:       K=4,  W=8
  F3 saturated:          K=16, W=16
Regression (already seen): R1 chains (3 lengths), R2 K2W4, K2W32, K8W4, K8W32, R3 arithmetic ring.
No further parameters will be fitted against these cases. Band ±15 %.
