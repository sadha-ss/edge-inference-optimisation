"""Latency measurement protocol (Section 7.3).

Latency is the time of a single-sample (batch size 1) inference measured with a
monotonic clock, after a warm-up period; median, p95 and p99 are reported.
Thread count is fixed and recorded; cold-start latency is reported separately.
"""
from __future__ import annotations

import copy
import os
import platform
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def measure_latency(model: nn.Module, x: torch.Tensor, warmup: int = 50, iters: int = 300,
                    threads: int = 1, preprocess: Optional[Callable[[], torch.Tensor]] = None) -> Dict[str, float]:
    """Steady-state latency in milliseconds. `preprocess`, if given, is timed with the forward pass."""
    torch.set_num_threads(threads)
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(preprocess() if preprocess else x)
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            model(preprocess() if preprocess else x)
            times.append((time.perf_counter_ns() - t0) / 1e6)
    a = np.asarray(times)
    return dict(median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), mean_ms=float(a.mean()), std_ms=float(a.std()),
                warmup=warmup, iters=iters, threads=threads)


def cold_start_ms(model: nn.Module, x: torch.Tensor, threads: int = 1) -> float:
    """First inference of a freshly copied model (allocation and cache population included)."""
    torch.set_num_threads(threads)
    m = copy.deepcopy(model).eval()
    with torch.inference_mode():
        t0 = time.perf_counter_ns()
        m(x)
        return (time.perf_counter_ns() - t0) / 1e6


def layer_times(model: nn.Module, x: torch.Tensor, iters: int = 50, threads: int = 1) -> Dict[str, float]:
    """Mean time (ms) of each leaf module, for comparing with the roofline estimate of Eq. 6."""
    torch.set_num_threads(threads)
    model.eval()
    acc: Dict[str, float] = {}
    starts: Dict[str, int] = {}
    handles = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_pre_hook(
                lambda mod, inp, n=name: starts.__setitem__(n, time.perf_counter_ns())))
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: acc.__setitem__(
                    n, acc.get(n, 0.0) + (time.perf_counter_ns() - starts[n]) / 1e6)))
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        acc.clear()
        for _ in range(iters):
            model(x)
    for h in handles:
        h.remove()
    return {k: v / iters for k, v in acc.items()}


def environment_info() -> Dict[str, str]:
    """Software and device description to be recorded with every result (Appendix E)."""
    info = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                threads_default=str(torch.get_num_threads()),
                quantized_engine=str(torch.backends.quantized.engine))
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        info["cpu"] = platform.processor()
    gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(gov):
        info["cpu_governor"] = open(gov).read().strip()
    return info
