#!/usr/bin/env bash
# Build Accel-Sim + our gpgpu-sim fork. Usage: bash scripts/build_sim.sh
set -e
source "$(dirname "$0")/env_sim.sh"
cd "$ACCELSIM_FRAMEWORK"
ln -sfn ../../gpgpu-sim_distribution2 gpu-simulator/gpgpu-sim
cmake -S gpu-simulator -B gpu-simulator/build -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_PREFIX_PATH="$CUDA_INSTALL_PATH"
cmake --build gpu-simulator/build -j"$(nproc)"
cmake --install gpu-simulator/build

# NVBit tracer. traces-processing/Makefile hardcodes system g++ (8.5, no
# std::filesystem), so build that binary with GCC 13 first.
cd "$ACCELSIM_FRAMEWORK/util/tracer_nvbit"
[ -d nvbit_release ] || ./install_nvbit.sh
(cd tracer_tool/traces-processing && "$CXX" -std=c++17 -O3 -g -pthread -o post-traces-processing post-traces-processing.cpp -lzstd)
make -j"$(nproc)"
