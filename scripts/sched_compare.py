#!/usr/bin/env python3
"""Compare HW vs simulated CAM-scheduling stand-ins (microbenchmarks/cam_placement/sched.cu).

Acceptance rules (fixed 2026-09-25 before any simulated sched result was seen):
- Absolute metrics per configuration: first-result latency T(Q=1), total T(Q=64), steady interval
  (T(64) - T(32)) / 32. Each must be within +-15 % of the HW median. No averaging across configurations.
- Relative metrics per schedule pair at matching (streams, W): improvement = 1 - T_better / T_worse, for
  total and steady interval. Flag MATERIAL if |sim - HW improvement| > max(HW improvement / 3, 0.03).
- Flag REVERSED if the sim ranking of a pair differs from HW, treating |improvement| <= 0.02 as a tie.

Usage: python3 scripts/sched_compare.py results/sched sp3   (sim directories results/sched/sim_<tag>_<cfg>)
"""
import re
import sys
from pathlib import Path

CORE_MHZ = 1755.0
BAND = 0.15


def parse_hw(path):
    res, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"sched=(\w),mode=hw,streams=(\d+),W=(\d+),slots=(\d+)", line)
        if m:
            s, st, w, sl = m.groups()
            cur = f"{s}_{st}_{w}" + (f"_{sl}" if s == "E" else "")
            res[cur] = {}
            continue
        m = re.match(r"hw,total_mismatches=(\d+)", line)
        if m and cur:
            res[cur]["mismatches"] = int(m.group(1))
        m = re.match(r"(\w+),n=\d+,median=([\d.]+),p95=([\d.]+)", line)
        if m and cur:
            res[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return res


def parse_sim(d):
    log = (d / "sim.log").read_text()
    c = [int(x) for x in re.findall(r"^gpu_sim_cycle = (\d+)", log, re.M)]
    if len(c) < 3:
        return None
    us = [x / CORE_MHZ for x in c[:3]]
    return {"first": us[0], "total": us[2], "interval": (us[2] - us[1]) / 32}


def main(root, tag):
    root = Path(root)
    hw = parse_hw(root / "hw.txt")
    rows, sim = [], {}
    out = [f"## Absolute metrics (sim tag {tag}, band ±15 %)", "",
           "| config | metric | HW median | HW p95 | sim | error | verdict |", "|---|---|---|---|---|---|---|"]
    fails = 0
    for cfg, h in hw.items():
        d = root / f"sim_{tag}_{cfg}"
        s = parse_sim(d) if (d / "sim.log").exists() else None
        sim[cfg] = s
        for key, hk in (("first", "first_result_us"), ("total", "total_t64_us"), ("interval", "steady_interval_us")):
            hm, hp = h[hk]
            if s is None:
                out.append(f"| {cfg} | {key} | {hm:.3f} | {hp:.3f} | missing | | |")
                continue
            err = s[key] / hm - 1
            ok = abs(err) <= BAND
            fails += not ok
            out.append(f"| {cfg} | {key} | {hm:.3f} | {hp:.3f} | {s[key]:.3f} | {err:+.1%} | {'PASS' if ok else 'FAIL'} |")

    pairs = [("C_1_117", "A_1_117"), ("C_1_234", "A_1_234"), ("C_1_468", "A_1_468"), ("C_16_234", "A_16_234"),
             ("E_1_234_2", "A_1_234"), ("E_1_234_2", "C_1_234"), ("E_1_117_2", "C_1_117"), ("E_1_468_2", "C_1_468"),
             ("E_16_234_2", "C_16_234"), ("E_1_234_2", "E_1_234_1"), ("E_1_234_4", "E_1_234_2")]
    out += ["", "## Relative metrics (improvement = 1 - T_first / T_second)", "",
            "| pair | metric | HW improvement | sim improvement | difference | flag |", "|---|---|---|---|---|---|"]
    flags = 0
    for a, b in pairs:
        if sim.get(a) is None or sim.get(b) is None:
            continue
        for key, hk in (("total", "total_t64_us"), ("interval", "steady_interval_us")):
            hi = 1 - hw[a][hk][0] / hw[b][hk][0]
            si = 1 - sim[a][key] / sim[b][key]
            flag = []
            if abs(si - hi) > max(abs(hi) / 3, 0.03):
                flag.append("MATERIAL")
            rank = lambda x: 0 if abs(x) <= 0.02 else (1 if x > 0 else -1)
            if rank(hi) != rank(si):
                flag.append("REVERSED")
            flags += bool(flag)
            out.append(f"| {a} vs {b} | {key} | {hi:+.1%} | {si:+.1%} | {si - hi:+.1%} | {' '.join(flag) or 'ok'} |")
    out += ["", f"Absolute failures: {fails}. Flagged relative comparisons: {flags}."]
    text = "\n".join(out) + "\n"
    (root / f"comparison_{tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
