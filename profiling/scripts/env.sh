#!/usr/bin/env bash
# Source this before running any benchmark:  source scripts/env.sh
#
# DeepGEMM compiles its kernels at runtime (DeepJIT), so nvcc and a C++20
# toolchain must be on PATH at *run* time, not just at install time.  The
# system compiler here is GCC 8.5 (no <format>) and there is no system CUDA
# toolkit, so both come from a self-contained micromamba environment under
# third_party/toolchain.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOOLCHAIN="${REPO_ROOT}/third_party/toolchain/env"
export CUDA_HOME="${TOOLCHAIN}"
export PATH="${TOOLCHAIN}/bin:${PATH}"
export LD_LIBRARY_PATH="${TOOLCHAIN}/lib:${LD_LIBRARY_PATH}"
export CC="${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-g++"
export DG_JIT_CACHE_DIR="${REPO_ROOT}/.dg_cache"
export PYTHON="${REPO_ROOT}/.venv/bin/python"

# nvcc's default host compiler is the system g++ (8.5 here), which cannot parse
# CUTLASS/CuTe C++20.  Point every nvcc invocation -- including DeepJIT's
# runtime ones -- at the toolchain's GCC 13.
export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"
export CUDAHOSTCXX="${CXX}"
