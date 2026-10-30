"""Memory budget: what the app costs, as macOS counts it.

`process_footprint()` is the figure Activity Monitor shows as "Memory": on
macOS it is the kernel's physical footprint, which includes the Metal buffers
MLX allocates in unified memory. Elsewhere it falls back to resident set size.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import psutil

GB = 1e9


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


_libproc = None


def process_footprint(pid: int | None = None) -> int:
    global _libproc
    pid = os.getpid() if pid is None else pid
    if sys.platform == "darwin":
        try:
            if _libproc is None:
                _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            info = _RUsageInfoV0()
            if _libproc.proc_pid_rusage(pid, 0, ctypes.byref(info)) == 0:
                return int(info.ri_phys_footprint)
        except Exception:
            pass
    return int(psutil.Process(pid).memory_info().rss)


def swap_used() -> int:
    return int(psutil.swap_memory().used)


def mlx_peak() -> int | None:
    try:
        import mlx.core as mx

        return int(mx.get_peak_memory())
    except Exception:
        return None


@dataclass
class MemorySample:
    t: float
    footprint: int
    swap: int
    mlx_active: int | None = None


@dataclass
class MemoryMonitor:
    """Samples the footprint in the background, to catch peaks and growth."""

    interval: float = 0.5
    samples: list[MemorySample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _sample(self) -> MemorySample:
        active = None
        try:
            import mlx.core as mx

            active = int(mx.get_active_memory())
        except Exception:
            pass
        return MemorySample(time.perf_counter(), process_footprint(), swap_used(), active)

    def start(self) -> "MemoryMonitor":
        self.samples.append(self._sample())

        def run():
            while not self._stop.wait(self.interval):
                self.samples.append(self._sample())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples.append(self._sample())

    @property
    def peak_footprint(self) -> int:
        return max((s.footprint for s in self.samples), default=0)

    @property
    def swap_growth(self) -> int:
        if len(self.samples) < 2:
            return 0
        return self.samples[-1].swap - self.samples[0].swap

    def growth_per_minute(self, skip_fraction: float = 0.2) -> float:
        """Slope of footprint over time (bytes/minute), ignoring the warm-up."""
        s = self.samples[int(len(self.samples) * skip_fraction):]
        if len(s) < 3:
            return 0.0
        import numpy as np

        t = np.array([x.t for x in s])
        f = np.array([x.footprint for x in s], float)
        slope = np.polyfit(t - t[0], f, 1)[0]
        return float(slope * 60)

    def to_dict(self) -> dict:
        return {
            "peak_footprint_gb": self.peak_footprint / GB,
            "swap_growth_gb": self.swap_growth / GB,
            "growth_gb_per_min": self.growth_per_minute() / GB,
            "samples": len(self.samples),
        }
