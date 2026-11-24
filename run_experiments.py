"""Staged experimental campaign (Section 7.7).

    stage 1  architecture x width x resolution (FP32 baselines)      3 x 4 x 3 = 36 configs
    stage 2  precision (FP32 / FP16 / INT8) on the top-k stage-1 candidates
    stage 3  pruning x distillation on the same top-k candidates

Every configuration is repeated over several seeds. Results are appended to
<out>/<task>/results.csv; finished rows are skipped, so an interrupted run can
be resumed with the same command.

Examples
--------
    # smoke test (minutes on a laptop CPU)
    python -m edgeopt.run_experiments --task cifar10 --quick

    # full campaign for one task
    python -m edgeopt.run_experiments --task kws --stage all --seeds 0 1 2 --epochs 30 --ft-epochs 10
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .benchmark import cold_start_ms, environment_info, measure_latency
from .data import get_data
from .models import TASK_SPECS, ModelConfig, build_model
from .profile import profile_model, serialized_size_bytes
from .prune import MagnitudePruner, prune_structured
from .quantize import quantize_int8, to_fp16
from .stats import bootstrap_ci, macro_f1, select_top_k
from .train import predict, set_seed, train

FAMILIES = ["dscnn", "resnet", "transformer"]
WIDTHS = [0.25, 0.5, 0.75, 1.0]
THREADS = (1, 4)

COLUMNS = ["task", "stage", "family", "width", "res", "precision", "prune", "prune_ratio", "kd", "seed",
           "epochs", "params", "nonzero_params", "macs", "peak_act_int8_bytes", "size_bytes",
           "acc", "acc_lo", "acc_hi", "macro_f1",
           "lat_t1_median_ms", "lat_t1_p95_ms", "lat_t1_p99_ms",
           "lat_t4_median_ms", "lat_t4_p95_ms", "lat_t4_p99_ms", "cold_start_ms", "reused", "note"]


# ----------------------------------------------------------------------------
# results file
# ----------------------------------------------------------------------------


def make_key(stage, family, width, res, precision, prune, kd, seed):
    return (str(stage), family, float(width), int(res), precision, prune, kd, int(seed))


class ResultWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def done_keys(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return set()
        df = pd.read_csv(self.path)
        return {make_key(r.stage, r.family, r.width, r.res, r.precision, r.prune, r.kd, r.seed)
                for r in df.itertuples()}

    def write(self, row: Dict):
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({c: row.get(c, "") for c in COLUMNS})


# ----------------------------------------------------------------------------
# evaluation of one model variant
# ----------------------------------------------------------------------------


def evaluate_variant(model_fp32, cfg: ModelConfig, data: dict, precision: str, args, device) -> Dict:
    """Accuracy (test split, bootstrap CI), profile, size and latency of one model in one precision."""
    model_fp32 = model_fp32.cpu().eval()
    ex = torch.zeros(cfg.input_shape(1))
    prof = profile_model(model_fp32, ex)
    row = dict(params=prof["params"], nonzero_params=prof["nonzero_params"], macs=prof["macs"],
               peak_act_int8_bytes=prof["peak_act_bytes"])
    notes: List[str] = []

    dtype, eval_dev, ex_run = None, torch.device("cpu"), ex
    if precision == "fp32":
        mv, eval_dev = model_fp32, device
    elif precision == "fp16":
        mv, dtype, ex_run = to_fp16(model_fp32), torch.float16, ex.half()
        eval_dev = device if device.type == "cuda" else torch.device("cpu")
    elif precision == "int8":
        mv = quantize_int8(model_fp32, cfg.family, data["calib"], ex)
    else:
        raise ValueError(precision)

    try:
        pred, lab = predict(mv.to(eval_dev), data["test"], eval_dev, dtype)
        acc, lo, hi = bootstrap_ci(lab, pred, "acc")
        row.update(acc=acc, acc_lo=lo, acc_hi=hi, macro_f1=macro_f1(lab, pred, cfg.n_classes))
    except RuntimeError as e:
        row.update(acc=float("nan"), acc_lo=float("nan"), acc_hi=float("nan"), macro_f1=float("nan"))
        notes.append(f"eval failed: {str(e)[:60]}")
    mv = mv.cpu()
    row["size_bytes"] = serialized_size_bytes(mv)

    for t in THREADS:
        try:
            lat = measure_latency(mv, ex_run, warmup=args.lat_warmup, iters=args.lat_iters, threads=t)
            row.update({f"lat_t{t}_median_ms": lat["median_ms"], f"lat_t{t}_p95_ms": lat["p95_ms"],
                        f"lat_t{t}_p99_ms": lat["p99_ms"]})
        except RuntimeError as e:
            row.update({f"lat_t{t}_median_ms": float("nan"), f"lat_t{t}_p95_ms": float("nan"),
                        f"lat_t{t}_p99_ms": float("nan")})
            notes.append(f"latency failed: {str(e)[:60]}")
    try:
        row["cold_start_ms"] = cold_start_ms(mv, ex_run, threads=1)
    except RuntimeError:
        row["cold_start_ms"] = float("nan")
    row["note"] = "; ".join(notes)
    return row


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


class Ctx:
    """Shared state for one campaign (task, args, device, caches)."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.out = os.path.join(args.out, args.task)
        self.ckpt_dir = os.path.join(self.out, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.writer = ResultWriter(os.path.join(self.out, "results.csv"))
        self.done = self.writer.done_keys()
        self._data: Dict[int, dict] = {}
        with open(os.path.join(self.out, "environment.json"), "w") as fh:
            json.dump(dict(environment_info(), args=vars(args)), fh, indent=2)

    def data(self, res: int) -> dict:
        if res not in self._data:
            self._data[res] = get_data(self.args.task, self.args.data_root, res,
                                       self.args.batch_size, self.args.subset)
        return self._data[res]

    def ckpt_path(self, cfg: ModelConfig, seed: int) -> str:
        return os.path.join(self.ckpt_dir, f"{cfg.key()}_s{seed}.pt")

    def load_ckpt(self, cfg: ModelConfig, seed: int):
        p = self.ckpt_path(cfg, seed)
        return torch.load(p, weights_only=False) if os.path.exists(p) else None

    def emit(self, cfg, stage, precision, prune, kd, seed, model, epochs, prune_ratio="", reused=0):
        key = make_key(stage, cfg.family, cfg.width, cfg.res, precision, prune, kd, seed)
        row = evaluate_variant(model, cfg, self.data(cfg.res), precision, self.args, self.device)
        row.update(task=cfg.task, stage=stage, family=cfg.family, width=cfg.width, res=cfg.res,
                   precision=precision, prune=prune, prune_ratio=prune_ratio, kd=kd, seed=seed,
                   epochs=epochs, reused=reused)
        self.writer.write(row)
        self.done.add(key)
        print(f"  -> acc {row['acc']:.4f}  lat(1 thread) {row['lat_t1_median_ms']:.3f} ms  "
              f"size {row['size_bytes'] / 1e3:.0f} KB")


def select_candidates(ctx: Ctx, k: int) -> List[ModelConfig]:
    df = pd.read_csv(ctx.writer.path)
    df = df[(df.stage.astype(str) == "1") & (df.precision == "fp32")].dropna(subset=["acc", "lat_t1_median_ms"])
    g = df.groupby(["family", "width", "res"], as_index=False)[["acc", "lat_t1_median_ms"]].mean()
    if g.empty:
        raise SystemExit("no stage-1 results found; run stage 1 first")
    idx = select_top_k(g["acc"].values, g["lat_t1_median_ms"].values, min(k, len(g)))
    return [ModelConfig(r.family, ctx.args.task, float(r.width), int(r.res)) for r in g.iloc[idx].itertuples()]


# ----------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------


def run_stage1(ctx: Ctx):
    a = ctx.args
    resolutions = a.resolutions or list(TASK_SPECS[a.task]["resolutions"])
    for fam in a.families:
        for w in a.widths:
            for res in resolutions:
                cfg = ModelConfig(fam, a.task, w, res)
                for seed in a.seeds:
                    if make_key(1, fam, w, res, "fp32", "none", "off", seed) in ctx.done:
                        continue
                    print(f"[stage 1] {cfg.key()} seed {seed}")
                    set_seed(seed)
                    model = build_model(cfg)
                    model, _ = train(model, ctx.data(res), a.epochs, fam, ctx.device, seed)
                    torch.save(model.cpu(), ctx.ckpt_path(cfg, seed))
                    ctx.emit(cfg, 1, "fp32", "none", "off", seed, model, a.epochs)


def run_stage2(ctx: Ctx):
    a = ctx.args
    for cfg in select_candidates(ctx, a.top_k):
        for seed in a.seeds:
            base = ctx.load_ckpt(cfg, seed)
            if base is None:
                continue
            for precision in ("fp32", "fp16", "int8"):
                if make_key(2, cfg.family, cfg.width, cfg.res, precision, "none", "off", seed) in ctx.done:
                    continue
                print(f"[stage 2] {cfg.key()} {precision} seed {seed}")
                ctx.emit(cfg, 2, precision, "none", "off", seed, copy.deepcopy(base), a.epochs)


def run_stage3(ctx: Ctx):
    a = ctx.args
    for cfg in select_candidates(ctx, a.top_k):
        teacher_cfg = ModelConfig(cfg.family, cfg.task, 1.0, cfg.res)
        for seed in a.seeds:
            base = ctx.load_ckpt(cfg, seed)
            teacher = ctx.load_ckpt(teacher_cfg, seed)
            if base is None:
                continue
            combos = [("none", "off"), ("none", "on"), ("structured", "off"), ("structured", "on")]
            if a.unstructured_sparsity:
                combos.append(("unstructured", "off"))
            for prune, kd in combos:
                key = make_key(3, cfg.family, cfg.width, cfg.res, "fp32", prune, kd, seed)
                if key in ctx.done:
                    continue
                if kd == "on" and teacher is None:
                    print(f"[stage 3] skipping {cfg.key()} kd: no width-1.0 teacher checkpoint")
                    continue
                print(f"[stage 3] {cfg.key()} prune={prune} kd={kd} seed {seed}")
                if (prune, kd) == ("none", "off"):          # control: the stage-1 baseline, not retrained
                    ctx.emit(cfg, 3, "fp32", prune, kd, seed, copy.deepcopy(base), a.epochs, reused=1)
                    continue
                data = ctx.data(cfg.res)
                tch = copy.deepcopy(teacher) if kd == "on" else None
                on_epoch, post_step, ratio = None, None, ""
                if prune == "none":                         # distillation from scratch, full budget
                    set_seed(seed)
                    model, epochs, lr_scale = build_model(cfg), a.epochs, 1.0
                elif prune == "structured":                 # prune the baseline, then fine-tune
                    model, epochs, lr_scale, ratio = prune_structured(base, a.prune_ratio), a.ft_epochs, 0.2, a.prune_ratio
                else:                                       # gradual unstructured magnitude pruning
                    model, epochs, lr_scale, ratio = copy.deepcopy(base), a.ft_epochs, 0.2, a.unstructured_sparsity
                    pruner = MagnitudePruner(model, a.unstructured_sparsity, 0, max(1, epochs - 1))
                    on_epoch, post_step = pruner.update, pruner.apply
                model, _ = train(model, data, epochs, cfg.family, ctx.device, seed, teacher=tch,
                                 kd_lambda=a.kd_lambda, kd_T=a.kd_temp, lr_scale=lr_scale,
                                 on_epoch_start=on_epoch, post_step=post_step)
                ctx.emit(cfg, 3, "fp32", prune, kd, seed, model, epochs, prune_ratio=ratio)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(TASK_SPECS), required=True)
    p.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30, help="training epochs per configuration (fixed budget)")
    p.add_argument("--ft-epochs", type=int, default=10, help="fine-tuning epochs after pruning")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    p.add_argument("--widths", type=float, nargs="+", default=WIDTHS)
    p.add_argument("--resolutions", type=int, nargs="+", default=None)
    p.add_argument("--top-k", type=int, default=6, help="candidates carried from stage 1 to stages 2-3")
    p.add_argument("--prune-ratio", type=float, default=0.3, help="fraction of channels removed (structured)")
    p.add_argument("--unstructured-sparsity", type=float, default=None,
                   help="if set, also run gradual unstructured pruning to this sparsity (for H4)")
    p.add_argument("--kd-lambda", type=float, default=0.5)
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subset", type=int, default=None, help="use only N training samples (debugging)")
    p.add_argument("--lat-warmup", type=int, default=50)
    p.add_argument("--lat-iters", type=int, default=300)
    p.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    a = p.parse_args(argv)
    if a.quick:
        a.epochs, a.ft_epochs, a.seeds = 1, 1, a.seeds[:1]
        a.subset = a.subset or 2048
        a.widths = [w for w in a.widths if w in (0.5, 1.0)] or [0.5, 1.0]
        a.resolutions = a.resolutions or [TASK_SPECS[a.task]["resolutions"][0]]
        a.lat_warmup, a.lat_iters, a.top_k = 5, 20, min(a.top_k, 3)
    return a


def main(argv=None):
    args = parse_args(argv)
    ctx = Ctx(args)
    stages = ["1", "2", "3"] if args.stage == "all" else [args.stage]
    for s in stages:
        {"1": run_stage1, "2": run_stage2, "3": run_stage3}[s](ctx)
    print(f"\nResults: {ctx.writer.path}")


if __name__ == "__main__":
    main()
