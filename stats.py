"""Statistics (Section 7.5) and Pareto utilities (Section 3.4). NumPy only."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(y, p, metric: str = "acc", n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0, n_classes: int = None) -> Tuple[float, float, float]:
    """Percentile bootstrap interval (Eq. 17): returns (estimate, lower, upper)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    if metric == "acc":
        correct = (y == p).astype(float)
        vals = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        est = float(correct.mean())
    elif metric == "f1":
        vals = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            vals[b] = macro_f1(y[idx], p[idx], n_classes)
        est = macro_f1(y, p, n_classes)
    else:
        raise ValueError(metric)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def paired_bootstrap_diff(y, p_a, p_b, n_boot: int = 1000, alpha: float = 0.05,
                          seed: int = 0) -> Tuple[float, float, float]:
    """Accuracy difference (A - B) with a paired bootstrap interval on the same test samples."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    d = (p_a == y).astype(float) - (p_b == y).astype(float)
    rng = np.random.default_rng(seed)
    n = len(d)
    vals = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def pareto_mask(acc, cost) -> np.ndarray:
    """True for non-dominated points (maximise accuracy, minimise cost)."""
    acc, cost = np.asarray(acc, float), np.asarray(cost, float)
    mask = np.ones(len(acc), dtype=bool)
    for i in range(len(acc)):
        dom = (acc >= acc[i]) & (cost <= cost[i]) & ((acc > acc[i]) | (cost < cost[i]))
        if dom.any():
            mask[i] = False
    return mask


def pareto_ranks(acc, cost) -> np.ndarray:
    """Non-dominated sorting: rank 0 is the Pareto front, rank 1 the front after removing it, ..."""
    acc, cost = np.asarray(acc, float), np.asarray(cost, float)
    ranks = np.zeros(len(acc), dtype=int)
    remaining = np.arange(len(acc))
    r = 0
    while remaining.size:
        m = pareto_mask(acc[remaining], cost[remaining])
        ranks[remaining[m]] = r
        remaining = remaining[~m]
        r += 1
    return ranks


def select_top_k(acc, cost, k: int) -> np.ndarray:
    """Indices of the k most promising candidates: lowest Pareto rank first, then highest accuracy."""
    acc = np.asarray(acc, float)
    ranks = pareto_ranks(acc, cost)
    return np.lexsort((-acc, ranks))[:k]
