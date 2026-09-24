"""CUDA-event timing helpers shared by the DSA and router benchmarks.

Timing protocol (see README section "Timing protocol"):
  * All GPU elapsed times come from ``torch.cuda.Event`` pairs recorded on the
    current (execution) stream.
  * ``group_size > 1`` records one event pair around a group of iterations and
    divides by the group size.  This is used for kernels whose duration is close
    to the CUDA event resolution (~0.5 us).  The chosen method is reported in
    the ``timing_method`` result column so grouped and per-iteration samples are
    never silently mixed.
  * Host-observed latency is measured separately with ``perf_counter`` around a
    synchronised region, so GPU time and host-observed time stay distinguishable.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch


@dataclass
class TimingResult:
    """GPU elapsed-time samples for one (stage, configuration) pair."""

    samples_us: list[float]
    timing_method: str
    group_size: int = 1
    warmup: int = 0
    host_samples_us: list[float] = field(default_factory=list)

    # -- summary statistics -------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.samples_us)

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.samples_us)

    @property
    def min_us(self) -> float:
        return min(self.samples_us)

    @property
    def p95_us(self) -> float:
        return percentile(self.samples_us, 95.0)

    @property
    def p99_us(self) -> float:
        return percentile(self.samples_us, 99.0)

    @property
    def stdev_us(self) -> float:
        return statistics.stdev(self.samples_us) if self.n > 1 else 0.0

    @property
    def iqr_us(self) -> float:
        return percentile(self.samples_us, 75.0) - percentile(self.samples_us, 25.0)

    @property
    def cv(self) -> float:
        """Coefficient of variation; the primary stability signal."""
        m = self.mean_us
        return self.stdev_us / m if m > 0 else 0.0

    @property
    def host_median_us(self) -> Optional[float]:
        return statistics.median(self.host_samples_us) if self.host_samples_us else None


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy-free so this stays importable early)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def time_gpu(
    fn: Callable[[], object],
    *,
    warmup: int = 20,
    iters: int = 100,
    group_size: int = 1,
    reset_fn: Optional[Callable[[], None]] = None,
    measure_host: bool = False,
    stream: Optional[torch.cuda.Stream] = None,
) -> TimingResult:
    """Time ``fn`` on the GPU.

    Args:
        fn: callable launching the work to be measured.  It must not synchronise,
            copy to host, allocate inputs, or extract Python scalars.
        warmup: untimed iterations run first (also triggers JIT/autotune).
        iters: number of recorded samples.
        group_size: iterations per recorded event pair.  ``elapsed/group_size``
            is reported as one sample.
        reset_fn: optional callable run *outside* the timed region before every
            trial (used to restore mutable state such as an incremental cache).
        measure_host: also record host-observed wall time per trial.  This adds a
            synchronisation per trial, so it is reported separately and never
            used as the GPU number.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    stream = stream or torch.cuda.current_stream()

    for _ in range(warmup):
        if reset_fn is not None:
            reset_fn()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    host_us: list[float] = []

    for i in range(iters):
        if reset_fn is not None:
            reset_fn()
            torch.cuda.synchronize()
        if measure_host:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        starts[i].record(stream)
        for _ in range(group_size):
            fn()
        ends[i].record(stream)
        if measure_host:
            torch.cuda.synchronize()
            host_us.append((time.perf_counter() - t0) * 1e6 / group_size)

    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1e3 / group_size for s, e in zip(starts, ends)]

    method = "cuda_event_per_iter" if group_size == 1 else f"cuda_event_grouped_{group_size}"
    return TimingResult(
        samples_us=samples,
        timing_method=method,
        group_size=group_size,
        warmup=warmup,
        host_samples_us=host_us,
    )


def autoscale_group_size(
    fn: Callable[[], object],
    *,
    target_us: float = 50.0,
    max_group: int = 64,
    probe_iters: int = 10,
) -> int:
    """Pick a group size so each recorded interval is comfortably above event noise.

    Returns 1 when the operation is already long enough, which keeps the common
    case on per-iteration timing.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(probe_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    per_iter_us = start.elapsed_time(end) * 1e3 / probe_iters
    if per_iter_us >= target_us:
        return 1
    group = int(target_us / max(per_iter_us, 1e-3)) + 1
    return max(1, min(group, max_group))


class MemoryProbe:
    """Records peak allocated/reserved bytes over a region."""

    def __init__(self, device: torch.device | int = 0):
        self.device = device
        self.peak_allocated = 0
        self.peak_reserved = 0

    def __enter__(self) -> "MemoryProbe":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc) -> None:
        torch.cuda.synchronize()
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
        self.peak_reserved = torch.cuda.max_memory_reserved(self.device)
        return None


# ---------------------------------------------------------------------------
# CUDA Graph capture
# ---------------------------------------------------------------------------

class GraphUnsupported(RuntimeError):
    """A callable could not be captured into a CUDA graph."""


def capture_graph(
    fn: Callable[[], object],
    *,
    warmup_iters: int = 3,
    pool: Optional[object] = None,
) -> tuple["torch.cuda.CUDAGraph", object]:
    """Capture ``fn`` into a CUDA graph and return ``(graph, last_output)``.

    Graph replay removes per-kernel host launch cost, so comparing eager and
    graph timings separates GPU work from host submission gaps.  Capture
    requires that ``fn`` allocate only from the capture pool and perform no
    host synchronisation or CPU-dependent control flow; callables that violate
    this raise :class:`GraphUnsupported`.

    Warmup runs on a side stream, which is what the CUDA graph API requires
    before capture.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup_iters):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, pool=pool):
            out = fn()
    except Exception as exc:
        raise GraphUnsupported(f"{type(exc).__name__}: {exc}") from exc
    torch.cuda.synchronize()
    return graph, out
