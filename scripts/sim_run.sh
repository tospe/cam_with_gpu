#!/usr/bin/env bash
# Simulate a trace with a named config and record everything needed to reproduce it.
# Usage: bash scripts/sim_run.sh <kernelslist.g> <config: e.g. SM90_H100_PCIe> <out_dir> [extra simulator options...]
# Extra options (e.g. -gpgpu_perf_sim_memcpy 0) override the config and are recorded.
# SIM_BIN overrides the simulator binary (default: the installed bin/release/accel-sim.out).
# The body is a function so bash parses it fully before running: editing this
# file while a job runs cannot change what that job executes.
main() {
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <kernelslist.g> <config> <out_dir> [opts...]"; exit 2; }
KL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; CFG="$2"; OUT="$3"; shift 3
set +u; source "$(dirname "$0")/env_sim.sh" >/dev/null; set -u
G="$ACCELSIM_FRAMEWORK/gpu-simulator/gpgpu-sim/configs/tested-cfgs/$CFG"
A="$ACCELSIM_FRAMEWORK/gpu-simulator/configs/tested-cfgs/$CFG"
mkdir -p "$OUT"; cd "$OUT"
cp "$G"/* . ; cp "$A/trace.config" .
{
  echo "date: $(date -Iseconds)"
  echo "trace: $KL"
  echo "config: $CFG"
  echo "extra options: $*"
  echo "accel-sim-framework2: $(git -C "$ACCELSIM_FRAMEWORK" rev-parse HEAD)$(git -C "$ACCELSIM_FRAMEWORK" diff --quiet || echo ' (dirty)')"
  echo "gpgpu-sim_distribution2: $(git -C "$ACCELSIM_FRAMEWORK/gpu-simulator/gpgpu-sim/" rev-parse HEAD)$(git -C "$ACCELSIM_FRAMEWORK/gpu-simulator/gpgpu-sim/" diff --quiet || echo ' (dirty)')"
  echo "binary: ${SIM_BIN:-installed bin/release/accel-sim.out}"
  echo "command: accel-sim.out -trace $KL -config gpgpusim.config -config trace.config $*"
} > sim_metadata.txt
start=$(date +%s)
rc=0
"${SIM_BIN:-$ACCELSIM_FRAMEWORK/gpu-simulator/bin/release/accel-sim.out}" -trace "$KL" -config gpgpusim.config -config trace.config "$@" > sim.log 2>&1 || rc=$?
echo "exit: $rc wall_s: $(( $(date +%s) - start ))" >> sim_metadata.txt
grep -E '^kernel_name|^gpu_sim_cycle|^gpu_sim_insn|deadlock|CRASH' sim.log > kernels_summary.txt || true
exit $rc
}
main "$@"; exit $?
