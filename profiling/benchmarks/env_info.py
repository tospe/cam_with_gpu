"""Environment capture: everything needed to interpret or reproduce a run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any, Optional

import torch


def _sh(cmd: list[str], timeout: int = 30) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


_SMI_FIELDS = [
    "name",
    "uuid",
    "pci.bus_id",
    "memory.total",
    "memory.free",
    "memory.used",
    "driver_version",
    "compute_mode",
    "persistence_mode",
    "power.limit",
    "power.default_limit",
    "enforced.power.limit",
    "clocks.max.sm",
    "clocks.max.memory",
    "clocks.applications.graphics",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "pcie.link.gen.max",
    "pcie.link.width.max",
    "ecc.mode.current",
]


def nvidia_smi_info(index: int = 0) -> dict[str, Any]:
    out = _sh(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_SMI_FIELDS)}",
            "--format=csv,noheader",
            "-i",
            str(index),
        ]
    )
    if out is None:
        return {"available": False}
    values = [v.strip() for v in out.split(",")]
    info: dict[str, Any] = {"available": True}
    info.update(dict(zip(_SMI_FIELDS, values)))

    mig = _sh(["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader", "-i", str(index)])
    info["mig_mode"] = mig

    # Other jobs sharing the GPU materially affect timing; record, never kill.
    procs = _sh(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
            "-i",
            str(index),
        ]
    )
    proc_lines = [ln for ln in (procs or "").splitlines() if ln.strip()]
    info["compute_processes"] = proc_lines
    info["gpu_shared_with_other_processes"] = len(
        [ln for ln in proc_lines if str(os.getpid()) not in ln]
    ) > 0
    return info


def torch_device_info(index: int = 0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    p = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    return {
        "cuda_available": True,
        "device_index": index,
        "device_name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "multi_processor_count": p.multi_processor_count,
        "total_memory_bytes": p.total_memory,
        "mem_get_info_free_bytes": free,
        "mem_get_info_total_bytes": total,
        "l2_cache_size_bytes": getattr(p, "L2_cache_size", None),
        "warp_size": getattr(p, "warp_size", None),
    }


def package_versions() -> dict[str, Any]:
    pkgs: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_git_version": torch.version.git_version,
    }
    for name in ("numpy", "pandas", "matplotlib", "yaml", "triton", "deep_gemm", "tilelang"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = None
    return pkgs


def repo_commit(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None
    return _sh(["git", "-C", path, "rev-parse", "HEAD"])


def collect(device_index: int = 0, extra_repos: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Full environment snapshot, stored once per run in metadata.json."""
    info: dict[str, Any] = {
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "gpu_nvidia_smi": nvidia_smi_info(device_index),
        "gpu_torch": torch_device_info(device_index),
        "packages": package_versions(),
        "nvcc": _sh(["nvcc", "--version"]),
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA_", "TORCH_", "NVIDIA_", "DG_", "PYTORCH_"))
        },
    }
    repos = extra_repos or {}
    info["repos"] = {name: repo_commit(p) for name, p in repos.items()}
    return info


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, default=str))
