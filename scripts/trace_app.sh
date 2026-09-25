#!/usr/bin/env bash
# Trace a CUDA program for Accel-Sim, the way run_hw_trace.py does it:
#   1. spinlock detection, phase 0 and phase 1 (writes spinlock_detection/)
#   2. NVBit trace with spinlock marking and register-value tracing
#   3. post-processing into kernelslist.g + .tracez
# Skipping step 1 while SPINLOCK_HANDLING_MODE>0 used to segfault the tracer.
#
# Usage: bash scripts/trace_app.sh <out_dir> <program> [args...]
# Env:   SPINLOCK_MODE (0 none, 1 fast_forward, 2 mark_region; default 2)
#        REG_VALS (ALLOW_REG_VAL_TRACING, default 1)
#        CUDA_VISIBLE_DEVICES is passed through.
# The body is a function so bash parses it fully before running: editing this
# file while a job runs cannot change what that job executes.
main() {
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <out_dir> <program> [args...]"; exit 2; }
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
set +u; source "$(dirname "$0")/env_sim.sh" >/dev/null; set -u
T="$ACCELSIM_FRAMEWORK/util/tracer_nvbit"
MODE="${SPINLOCK_MODE:-2}"
REG="${REG_VALS:-1}"

rm -rf "$OUT/spinlock_detection" "$OUT/traces"
mkdir -p "$OUT/spinlock_detection"
if [ "$MODE" -gt 0 ]; then
  echo "== spinlock detection"
  TRACES_FOLDER="$OUT" SPINLOCK_PHASE=0 CUDA_INJECTION64_PATH="$T/others/spinlock_tool/spinlock_tool.so" "$@" > "$OUT/spinlock_phase0.log" 2>&1
  TRACES_FOLDER="$OUT" SPINLOCK_PHASE=1 CUDA_INJECTION64_PATH="$T/others/spinlock_tool/spinlock_tool.so" "$@" > "$OUT/spinlock_phase1.log" 2>&1
fi

echo "== trace"
TRACES_FOLDER="$OUT" SPINLOCK_HANDLING_MODE="$MODE" ALLOW_REG_VAL_TRACING="$REG" \
  NVBIT_INSTRUMENTATION_ENABLED=1 CUDA_INJECTION64_PATH="$T/tracer_tool/tracer_tool.so" \
  "$@" > "$OUT/trace.log" 2>&1

echo "== post-process"
"$T/tracer_tool/traces-processing/post-traces-processing" "$OUT/traces" -j 8 > "$OUT/postprocess.log" 2>&1
rm -f "$OUT"/traces/*.trace "$OUT"/traces/*.trace.xz

{
  echo "date: $(date -Iseconds)"
  echo "command: $*"
  echo "cwd: $PWD"
  echo "SPINLOCK_HANDLING_MODE=$MODE ALLOW_REG_VAL_TRACING=$REG"
  echo "accel-sim-framework2: $(git -C "$ACCELSIM_FRAMEWORK" rev-parse HEAD)$(git -C "$ACCELSIM_FRAMEWORK" diff --quiet || echo ' (dirty)')"
  echo "nvbit: $(grep -m1 -oE 'v[0-9.]+' "$T/install_nvbit.sh")"
  echo "nvcc: $(nvcc --version | tail -1)"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/gpu: /'
} > "$OUT/trace_metadata.txt"
echo "== done: $OUT/traces/kernelslist.g"
}
main "$@"; exit $?
