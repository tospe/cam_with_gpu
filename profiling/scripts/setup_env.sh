#!/usr/bin/env bash
# One-shot provisioning for the H100 search benchmarks.
#
# This machine has no CUDA toolkit and only GCC 8.5, which cannot build
# DeepGEMM (C++20 <format> needs GCC 13+, and DeepGEMM needs CUDA >= 12.9).
# Everything is installed locally under third_party/, without root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TOOLS="${REPO_ROOT}/third_party/toolchain"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

echo "==> [1/5] Python venv + pinned dependencies"
"$UV" venv --python 3.12 "${REPO_ROOT}/.venv"
VIRTUAL_ENV="${REPO_ROOT}/.venv" "$UV" pip install \
    --index-strategy unsafe-best-match -r requirements.txt

echo "==> [2/5] micromamba"
mkdir -p "$TOOLS"
if [[ ! -x "${TOOLS}/bin/micromamba" ]]; then
    curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$TOOLS" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="${TOOLS}/mamba"

echo "==> [3/5] GCC 13 + CUDA 12.9 toolkit (nvcc, headers, math libs)"
"${TOOLS}/bin/micromamba" create -y -p "${TOOLS}/env" -c nvidia -c conda-forge \
    gxx_linux-64=13 gcc_linux-64=13 elfutils "cuda-version=12.9" \
    cuda-nvcc cuda-cudart-dev cuda-crt cuda-nvrtc-dev cuda-cccl \
    "libcublas-dev=12.9.2.10" libcusparse-dev libcusolver-dev libcufft-dev libcurand-dev

# conda-forge/nvidia put CUDA headers under targets/<arch>/include, but nvcc and
# DeepGEMM's setup.py look in $CUDA_HOME/include. Link them across.
pushd "${TOOLS}/env/targets/x86_64-linux/include" >/dev/null
for f in *; do
    [[ -e "${TOOLS}/env/include/$f" ]] || ln -s "${TOOLS}/env/targets/x86_64-linux/include/$f" "${TOOLS}/env/include/$f"
done
popd >/dev/null
[[ -e "${TOOLS}/env/include/cccl" ]] || ln -s "${TOOLS}/env/targets/x86_64-linux/include" "${TOOLS}/env/include/cccl"
[[ -e "${TOOLS}/env/lib/libcudart.so" ]] || ln -s libcudart.so.12 "${TOOLS}/env/lib/libcudart.so"

echo "==> [4/5] upstream checkouts (pinned)"
mkdir -p third_party
declare -A REPOS=(
    [DeepGEMM]="https://github.com/deepseek-ai/DeepGEMM.git 78b69000794d0937b47ae3387eff7663410264d1"
    [DeepSeek-V3.2-Exp]="https://github.com/deepseek-ai/DeepSeek-V3.2-Exp.git 87e509a2e5a100d221c97df52c6e8be7835f0057"
)
for name in "${!REPOS[@]}"; do
    read -r url commit <<<"${REPOS[$name]}"
    if [[ ! -d "third_party/${name}/.git" ]]; then
        git clone -q "$url" "third_party/${name}"
    fi
    git -C "third_party/${name}" fetch -q --depth 50 origin "$commit" 2>/dev/null || true
    git -C "third_party/${name}" checkout -q "$commit"
done
git -C third_party/DeepGEMM submodule update --init --recursive --depth 1

# Blackwell-only header missing an include; does not touch any kernel used here.
git -C third_party/DeepGEMM apply --check "${REPO_ROOT}/third_party/patches/deepgemm-cuda_fp8-include.patch" 2>/dev/null \
    && git -C third_party/DeepGEMM apply "${REPO_ROOT}/third_party/patches/deepgemm-cuda_fp8-include.patch" \
    && echo "    applied deepgemm-cuda_fp8-include.patch"

echo "==> [5/5] build + install DeepGEMM"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/env.sh"
pushd third_party/DeepGEMM >/dev/null
rm -rf build dist ./*.egg-info
"${REPO_ROOT}/.venv/bin/python" setup.py bdist_wheel
popd >/dev/null
VIRTUAL_ENV="${REPO_ROOT}/.venv" "$UV" pip install --no-deps third_party/DeepGEMM/dist/*.whl
VIRTUAL_ENV="${REPO_ROOT}/.venv" "$UV" pip install pytest

echo
echo "Done. Before every run:  source scripts/env.sh"
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import torch, deep_gemm
p = torch.cuda.get_device_properties(0)
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
print(f"  {p.name}  SM{p.major}{p.minor}  {p.multi_processor_count} SMs  {p.total_memory/2**30:.1f} GiB")
print(f"  deep_gemm OK, num_sms={deep_gemm.get_num_sms()}")
PY
