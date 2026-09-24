"""Run directories, the raw/summary result schema, and CSV writers.

Each run writes an immutable directory::

    results/<run_id>/metadata.json     environment + resolved config, once per run
    results/<run_id>/measurements.csv  one row per timing sample (raw)
    results/<run_id>/summary.csv       one row per (config, stage)
    results/<run_id>/figures/          plots produced by scripts/plot_results.py
    results/<run_id>/report.md         written by hand after the run

Failures and unsupported configurations are written as rows with ``status`` set
to something other than ``ok`` so they survive in machine-readable form.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Columns required by the brief, in order.  Anything not applicable is written
# as an empty field, which pandas reads back as NaN/None -- never as 0.
RAW_COLUMNS = [
    "run_id",
    "workload",
    "mode",
    "backend",
    "backend_commit",
    "dtype",
    "batch_size",
    "candidate_count",
    "dimension",
    "indexer_heads",
    "k",
    "seed",
    "trial",
    "decode_step",
    "stage",
    "graph_mode",
    "timing_method",
    "elapsed_us",
    "status",
    "error",
    # --- additional context, kept after the required columns ---
    "topk_sorted",
    "context_len_actual",
    "block_kv",
    "steps",
    "group_size",
    "warmup",
    "config_id",
]

SUMMARY_COLUMNS = [
    "run_id",
    "workload",
    "mode",
    "backend",
    "backend_commit",
    "dtype",
    "batch_size",
    "candidate_count",
    "dimension",
    "indexer_heads",
    "k",
    "seed",
    "stage",
    "graph_mode",
    "timing_method",
    "status",
    "error",
    "n_samples",
    "warmup",
    "group_size",
    "median_us",
    "mean_us",
    "min_us",
    "p95_us",
    "p99_us",
    "stdev_us",
    "iqr_us",
    "cv",
    "host_median_us",
    "topk_sorted",
    "throughput_qps",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "setup_us",
    "steps",
    "block_kv",
    "config_id",
]


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


@dataclass
class RunWriter:
    """Owns one immutable run directory."""

    root: str
    run_id: str
    _raw_fh: Any = field(default=None, repr=False)
    _raw_w: Any = field(default=None, repr=False)
    _sum_fh: Any = field(default=None, repr=False)
    _sum_w: Any = field(default=None, repr=False)

    @property
    def dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    def open(self) -> "RunWriter":
        os.makedirs(os.path.join(self.dir, "figures"), exist_ok=True)
        raw_path = os.path.join(self.dir, "measurements.csv")
        sum_path = os.path.join(self.dir, "summary.csv")
        if os.path.exists(raw_path):
            raise FileExistsError(
                f"{raw_path} exists; run directories are immutable, pick a new --run-id"
            )
        self._raw_fh = open(raw_path, "w", newline="")
        self._raw_w = csv.DictWriter(self._raw_fh, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        self._raw_w.writeheader()
        self._sum_fh = open(sum_path, "w", newline="")
        self._sum_w = csv.DictWriter(self._sum_fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        self._sum_w.writeheader()
        return self

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        with open(os.path.join(self.dir, "metadata.json"), "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

    def write_raw(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self._raw_w.writerow({c: _fmt(row.get(c)) for c in RAW_COLUMNS})
        self._raw_fh.flush()

    def write_summary(self, row: dict[str, Any]) -> None:
        self._sum_w.writerow({c: _fmt(row.get(c)) for c in SUMMARY_COLUMNS})
        self._sum_fh.flush()

    def close(self) -> None:
        for fh in (self._raw_fh, self._sum_fh):
            if fh is not None:
                fh.close()

    def __enter__(self) -> "RunWriter":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def _fmt(v: Any) -> Any:
    """None -> empty field (reads back as null, never 0)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def summarize(
    timing,
    *,
    base: dict[str, Any],
    stage: str,
    batch_size: int,
    peak_allocated: Optional[int] = None,
    peak_reserved: Optional[int] = None,
    setup_us: Optional[float] = None,
) -> dict[str, Any]:
    """Build a summary row from a ``TimingResult``.

    ``throughput_qps`` is ``batch_size / batch_latency`` -- the query rate the
    batched call sustains.  It is deliberately not reported as a per-query
    latency, which would understate what each query actually waits.
    """
    row = dict(base)
    med_s = timing.median_us / 1e6
    row.update(
        stage=stage,
        timing_method=timing.timing_method,
        n_samples=timing.n,
        warmup=timing.warmup,
        group_size=timing.group_size,
        median_us=timing.median_us,
        mean_us=timing.mean_us,
        min_us=timing.min_us,
        p95_us=timing.p95_us,
        p99_us=timing.p99_us,
        stdev_us=timing.stdev_us,
        iqr_us=timing.iqr_us,
        cv=timing.cv,
        host_median_us=timing.host_median_us,
        throughput_qps=(batch_size / med_s) if med_s > 0 else None,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        setup_us=setup_us,
        status="ok",
        error=None,
    )
    return row


def raw_rows(timing, *, base: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    rows = []
    for i, us in enumerate(timing.samples_us):
        row = dict(base)
        row.update(
            stage=stage,
            trial=i,
            elapsed_us=us,
            timing_method=timing.timing_method,
            group_size=timing.group_size,
            warmup=timing.warmup,
            status="ok",
            error=None,
        )
        rows.append(row)
    return rows
