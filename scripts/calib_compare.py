#!/usr/bin/env python3
"""Compare hardware vs simulator calibration metrics (docs/calibration_plan.md).

Usage: python3 scripts/calib_compare.py results/calib-v1
Reads <run>/hw/calib_output.csv and <run>/sim_{lat_l2,lat_dram,bw}/sim.log; writes <run>/comparison.md.
"""
import re
import sys
from pathlib import Path

CORE_MHZ = 1755.0
BAND = {"M1": 0.15, "M2": 0.15, "M3": 0.15}
H1, H2 = 256, 1024
B1, B2 = 128 << 20, 256 << 20


def hw_median(text, key):
    m = re.search(rf"^{key},n=\d+,median=([\d.]+),p95=([\d.]+)", text, re.M)
    return float(m.group(1)), float(m.group(2))


def sim_kernels(log):
    """[(kernel_name, gpu_sim_cycle, shader)] in launch order."""
    text = log.read_text()
    names = re.findall(r"^kernel_name = (\S+)", text, re.M)
    cycles = [int(c) for c in re.findall(r"^gpu_sim_cycle = (\d+)", text, re.M)]
    shaders = {}
    for sid, kid in re.findall(r"Shader (\d+) bind to kernel (\d+)", text):
        shaders.setdefault(int(kid), sid)
    return [(n, c, shaders.get(i + 1)) for i, (n, c) in enumerate(zip(names, cycles))]


def main(run):
    run = Path(run)
    hw = (run / "hw" / "calib_output.csv").read_text()
    rows = []

    for mid, tag in (("M1", "lat_l2"), ("M2", "lat_dram")):
        ks = [k for k in sim_kernels(run / f"sim_{tag}" / "sim.log") if "chase" in k[0]]
        sim = (ks[1][1] - ks[0][1]) / (H2 - H1)
        hwv, p95 = hw_median(hw, f"{tag}_cycles_per_hop")
        rows.append((mid, f"{tag} cycles/hop", hwv, p95, sim, f"sim SMs {ks[0][2]},{ks[1][2]}"))

    ks = sim_kernels(run / "sim_bw" / "sim.log")
    dt_s = (ks[1][1] - ks[0][1]) / (CORE_MHZ * 1e6)
    sim_bw = (B2 - B1) / dt_s / 1e9
    hwv, p95 = hw_median(hw, "bw_slope_GBps")
    rows.append(("M3", "DRAM read BW GB/s (slope)", hwv, p95, sim_bw, f"sim cycles {ks[0][1]},{ks[1][1]}"))

    out = ["| ID | metric | HW median | HW p95 | sim | error | band | verdict | note |", "|---|---|---|---|---|---|---|---|---|"]
    for mid, name, hwv, p95, sim, note in rows:
        err = (sim - hwv) / hwv
        ok = "PASS" if abs(err) <= BAND[mid] else "OUT"
        out.append(f"| {mid} | {name} | {hwv:.1f} | {p95:.1f} | {sim:.1f} | {err:+.1%} | ±{BAND[mid]:.0%} | {ok} | {note} |")
    clk = hw_median(hw, "clock_full_load_MHz")[0]
    out.append(f"\nM4 sustained SM clock (full load): {clk:.1f} MHz vs config {CORE_MHZ:.0f} MHz.")
    (run / "comparison.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
