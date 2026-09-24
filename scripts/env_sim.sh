#!/usr/bin/env bash
# Source before building/running Accel-Sim:  source scripts/env_sim.sh
# Reuses the no-root CUDA 12.9 + GCC 13 toolchain from profiling/ (system has
# GCC 8.5 and no CUDA toolkit). zlib, cuda-profiler-api and libgl-devel were
# added to that env for the simulator build.
CAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="${CAM_ROOT}/profiling/third_party/toolchain/env"
export CUDA_INSTALL_PATH="$T" CUDA_HOME="$T"
export PATH="$T/bin:$PATH"
export CC="$T/bin/x86_64-conda-linux-gnu-gcc" CXX="$T/bin/x86_64-conda-linux-gnu-g++"
export CPATH="$T/include:$T/targets/x86_64-linux/include"
export LIBRARY_PATH="$T/lib"
export LD_LIBRARY_PATH="$T/lib:${LD_LIBRARY_PATH}"
export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"
export ACCELSIM_FRAMEWORK="${CAM_ROOT}/accel-sim-framework2"
# gpu-simulator/gpgpu-sim is a symlink to ${CAM_ROOT}/gpgpu-sim_distribution2
source "${ACCELSIM_FRAMEWORK}/gpu-simulator/setup_environment.sh"
