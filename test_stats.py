import numpy as np

from edgeopt.stats import (bootstrap_ci, macro_f1, paired_bootstrap_diff, pareto_mask, pareto_ranks,
                           select_top_k)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 2000)
    p = np.where(rng.random(2000) < 0.8, y, (y + 1) % 10)
    est, lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo <= est <= hi and 0.75 < est < 0.85 and hi - lo < 0.06


def test_macro_f1_perfect_and_wrong():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y, 3) == 1.0
    assert macro_f1(y, (y + 1) % 3, 3) == 0.0


def test_paired_diff_sign():
    y = np.zeros(1000, dtype=int)
    a = np.zeros(1000, dtype=int)
    b = np.where(np.arange(1000) % 10 == 0, 1, 0)       # 90% accurate
    d, lo, hi = paired_bootstrap_diff(y, a, b, n_boot=300)
    assert abs(d - 0.1) < 1e-9 and lo > 0


def test_pareto():
    acc = np.array([0.9, 0.8, 0.7, 0.85, 0.6])
    lat = np.array([10., 5., 4., 9., 6.])
    m = pareto_mask(acc, lat)
    assert m.tolist() == [True, True, True, True, False]      # last point dominated by (0.7, 4)
    assert pareto_ranks(acc, lat).tolist() == [0, 0, 0, 0, 1]


def test_select_top_k():
    acc = np.array([0.9, 0.8, 0.7, 0.85, 0.6])
    lat = np.array([10., 5., 4., 9., 6.])
    idx = select_top_k(acc, lat, 3)
    assert idx.tolist() == [0, 3, 1]                          # front first, ordered by accuracy
