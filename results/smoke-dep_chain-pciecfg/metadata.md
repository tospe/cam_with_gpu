# Run: smoke-dep_chain-pciecfg (2026-09-25)

- Purpose: first run on the new `SM90_H100_PCIe` config; structural check. NOT calibrated.
- Trace: `traces/dep_chain` (same as `smoke-dep_chain-regval-sxmcfg`; reg values + spinlock marking).
- Simulator: accel-sim-framework2 `a238d6a`, gpgpu-sim_distribution2 `fa2f6f39`. Config files copied here.
- Device properties of the target: `device_props.txt` (from `microbenchmarks/hopper_validation/device_props.cu`).

| kernel | SXM cfg cycles | PCIe cfg cycles | PCIe ns @1755 |
|---|---|---|---|
| vecadd | 4030 | 4081 | 2.3 us |
| chase (256 dependent hops) | 130349 | 79624 | 45.4 us (177 ns/hop) |

## Finding: single-warp latency depends on SM placement vs. L2 partition

Chase per-kernel deltas: SXM cfg 257 L2 misses / 128 DRAM reads; PCIe cfg 1 L2 miss / 0 DRAM reads.
Cause (verified by one-factor swaps, same trace):
- clocks: no effect (PCIe cfg with SXM clocks = 79624 cycles);
- SM count: flips it (PCIe cfg with 132 SMs = 130397 cycles);
- `-gpgpu_perf_sim_memcpy 0`: both configs ~155.4k cycles (all cold misses, ~607 cycles/hop).

Mechanism: `-gpgpu_perf_sim_memcpy 1` pre-fills the **home** L2 slice on cudaMemcpy; `-gpgpu_n_chiplet_partition 2` splits L2 into two halves. The single chase CTA is placed round-robin after vecadd's 256 CTAs: SM 124 (= 256 mod 132) on SXM cfg, SM 28 (= 256 mod 114) on PCIe cfg, i.e. different halves, so one config hits the pre-filled lines and the other does not.

Consequence for the CAM study: single-issuer latency can change ~2x with issuer SM placement and cache state. Experiments must control/report issuer SM (partition) and define cache state (memcpy pre-fill, warm-up) identically for hardware and simulation.
