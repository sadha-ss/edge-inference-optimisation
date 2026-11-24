"""Analysis of the campaign results: Pareto plots and hypothesis checks H1, H3, H4, H5 (Table 22).

    python -m edgeopt.analyse results/kws/results.csv --out results/kws/analysis

H2 (precision speed-up vs arithmetic intensity) and H6 (sub-additivity) need
per-layer logs and the full factor grid; see benchmark.layer_times and Section 8.4.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stats import pareto_mask

CFG = ["family", "width", "res"]


def stage1_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.stage.astype(str) == "1") & (df.precision == "fp32")]
    cols = ["acc", "lat_t1_median_ms", "size_bytes", "macs", "params", "peak_act_int8_bytes"]
    return d.groupby(CFG, as_index=False)[cols].mean()


def pareto_plot(g: pd.DataFrame, xcol: str, xlabel: str, path: str):
    g = g.dropna(subset=["acc", xcol])
    mask = pareto_mask(g["acc"].values, g[xcol].values)
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam, sub in g.groupby("family"):
        ax.scatter(sub[xcol], sub["acc"], label=fam, alpha=0.8)
    front = g[mask].sort_values(xcol)
    ax.plot(front[xcol], front["acc"], "k--", lw=1, label="Pareto front")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def h1(g: pd.DataFrame) -> str:
    """H1: counts predict latency less well across families than within one family."""
    out = ["H1: Spearman rank correlation of MACs vs. median latency (stage 1, FP32, 1 thread)"]
    out.append(f"  across all families: {spearmanr(g['macs'], g['lat_t1_median_ms'])[0]:.3f}")
    within = []
    for fam, sub in g.groupby("family"):
        if len(sub) > 2:
            r = spearmanr(sub["macs"], sub["lat_t1_median_ms"])[0]
            within.append(r)
            out.append(f"  within {fam:12s}: {r:.3f}")
    if within:
        out.append(f"  mean within-family: {np.mean(within):.3f}  (H1 supported if this exceeds the across-family value)")
    return "\n".join(out)


def h3(g: pd.DataFrame) -> str:
    """H3: peak activation memory, not weights, binds RAM (INT8 activations vs INT8 weights)."""
    w8 = g["params"]                       # INT8 weights: one byte per parameter
    frac = float((g["peak_act_int8_bytes"] > w8).mean())
    out = ["H3: peak INT8 activation memory vs. INT8 weight memory"]
    for fam, sub in g.groupby("family"):
        out.append(f"  {fam:12s}: activations exceed weights in {100 * float((sub['peak_act_int8_bytes'] > sub['params']).mean()):.0f}% of configs")
    out.append(f"  overall: {100 * frac:.0f}%  (RAM is the first limit when this is high and flash >> RAM)")
    return "\n".join(out)


def h4(df: pd.DataFrame) -> str:
    """H4: structured pruning gives latency gains tracking MACs on dense kernels; unstructured does not."""
    d = df[df.stage.astype(str) == "3"]
    base = d[(d.prune == "none") & (d.kd == "off")].set_index(CFG + ["seed"])
    out = ["H4: latency and MAC ratio vs. unpruned baseline (stage 3, no distillation)"]
    for pr in ("structured", "unstructured"):
        sub = d[(d.prune == pr) & (d.kd == "off")].set_index(CFG + ["seed"])
        j = sub.join(base, rsuffix="_b", how="inner")
        if j.empty:
            continue
        lat = (j["lat_t1_median_ms"] / j["lat_t1_median_ms_b"]).mean()
        macs = (j["macs"] / j["macs_b"]).mean()
        nz = (j["nonzero_params"] / j["nonzero_params_b"]).mean()
        out.append(f"  {pr:13s}: latency ratio {lat:.2f}, MAC ratio {macs:.2f}, non-zero parameter ratio {nz:.2f}")
    return "\n".join(out)


def h5(df: pd.DataFrame) -> str:
    """H5: distillation recovers accuracy (paired by candidate and seed)."""
    d = df[(df.stage.astype(str) == "3") & (df.prune.isin(["none", "structured"]))]
    out = ["H5: accuracy difference, distillation on minus off (stage 3)"]
    for pr in ("none", "structured"):
        on = d[(d.prune == pr) & (d.kd == "on")].set_index(CFG + ["seed"])["acc"]
        off = d[(d.prune == pr) & (d.kd == "off")].set_index(CFG + ["seed"])["acc"]
        diff = (on - off).dropna()
        if len(diff):
            out.append(f"  prune={pr:10s}: mean {100 * diff.mean():+.2f} points over {len(diff)} paired runs "
                       f"(min {100 * diff.min():+.2f}, max {100 * diff.max():+.2f})")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_csv")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(a.results_csv), "analysis")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(a.results_csv)
    g = stage1_table(df)
    pareto_plot(g, "lat_t1_median_ms", "median latency, 1 thread (ms)", os.path.join(out, "pareto_latency.png"))
    pareto_plot(g, "size_bytes", "serialised model size (bytes)", os.path.join(out, "pareto_size.png"))
    parts = [h1(g.dropna(subset=["lat_t1_median_ms"])), h3(g), h4(df), h5(df)]
    text = "\n\n".join(parts)
    with open(os.path.join(out, "hypotheses.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nPlots and summary written to {out}/")


if __name__ == "__main__":
    main()
