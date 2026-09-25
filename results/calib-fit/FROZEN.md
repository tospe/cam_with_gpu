# Frozen candidate: -dram_latency 283 (SM90_H100_PCIe)

Frozen 2026-09-25T12:15:48-04:00, BEFORE any held-out case was simulated or measured.

- Parameter: `-dram_latency` (gpgpu-sim `l2cache.cc:321`, simple DRAM model). Units: core cycles
  (ready_cycle uses gpu_sim_cycle, which advances on CORE ticks, `gpu-sim.cc:2362`); enqueue/dequeue run on
  DRAM ticks (1593 MHz), so the effective delay rounds up to the next DRAM tick (<= ~1.1 core cycles).
  Applied once per L2->DRAM request; the chase issues exactly one DRAM read per hop (256 / 1024).
- Baseline 194 (inherited from upstream SM90_H100, itself shared with SM90_H200).
- Sensitivity: 194 -> 552.36 cycles/hop, 214 -> 572.22 (+19.86 for +20). Linear.
- Fit case (the only case used): M2 calibration chase, 256 MiB, 256 B stride, seed 1, cold L2,
  `-gpgpu_perf_sim_memcpy 0`. HW median 657.95 cycles/hop.
- Candidate 283 (user-proposed from the DRAM-minus-L2 gap): 641.29 cycles/hop, error -2.53%.
- Not tuned further (e.g. to close the remaining -2.5%); 283 is evaluated as proposed.

Reserved held-out cases (defined now, run only after this commit):
  H1 footprint: 1024 MiB, 256 B stride, seed 1
  H2 stride:    256 MiB, 128 B stride, seed 1
  H3 seed:      256 MiB, 256 B stride, seed 2
Regression checks: M1 L2-hit latency unchanged; M3 bandwidth with 283; a concurrent-memory latency test.
Pass criterion: same ±15 % band as docs/calibration_plan.md.
