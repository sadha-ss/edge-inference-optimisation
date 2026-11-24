# Optimising AI Models for Edge Inference on Resource-Constrained Devices

## What is implemented

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # unit tests (model tests need PyTorch)
```

## Running the experiments

Smoke test (a few minutes on a laptop CPU, tiny subset, 1 epoch):

```bash
python -m edgeopt.run_experiments --task cifar10 --quick
python -m edgeopt.run_experiments --task kws --quick
```

Full campaign for one task (three seeds, staged design of Section 7.7):

```bash
python -m edgeopt.run_experiments --task cifar10 --stage all --seeds 0 1 2 --epochs 30 --ft-epochs 10
python -m edgeopt.run_experiments --task kws     --stage all --seeds 0 1 2 --epochs 30 --ft-epochs 10
```

- **Stage 1:** 3 families x 4 widths x 3 resolutions = 36 FP32 baselines per seed.
- **Stage 2:** the top-k stage-1 candidates (Pareto-ranked on accuracy vs. latency, `--top-k 6`) in FP32, FP16 and INT8.
- **Stage 3:** the same candidates with pruning (none / structured) x distillation (off / on). Add `--unstructured-sparsity 0.7` to also run unstructured pruning, which is needed for H4.

Results are appended to `results/<task>/results.csv` and finished rows are skipped, so an interrupted run resumes with the same command. The software environment and arguments are saved to `results/<task>/environment.json` (Appendix E checklist).

Analysis:

```bash
python -m edgeopt.analyse results/kws/results.csv
```

This writes Pareto plots and `hypotheses.txt` to `results/kws/analysis/`.

## Design notes

- **Resolution.** For CIFAR-10 it is the image side (32, 24, 16). For keyword spotting it is the number of MFCC frames (49, 40, 32), by interpolation of the 49x10 feature map.
- **Data.** Speech Commands v0.01 (about 65k clips, 30 words) is used, to match the paper. Unknown and silence classes are each about 10% of the keyword count; silence is cut from the background-noise files, which are shared across splits. The CIFAR-10 / training-validation split is fixed (seed 1234), independent of the training seed.
- **Distillation.** The teacher is the width-1.0 model of the same family, resolution and seed from stage 1. For width-1.0 candidates this is self-distillation.
- **Budgets.** Distillation without pruning retrains from scratch for the full budget. Pruned models are fine-tuned for `--ft-epochs`, so their total training budget is larger than the baseline's; this should be stated when reporting. The (no pruning, no distillation) row in stage 3 is the stage-1 baseline, marked `reused=1`.
- **Quantisation.** INT8 CNNs use FX graph-mode post-training quantisation with 10 calibration batches from the training split. FP16 latency on a CPU without half-precision kernels is recorded as NaN with a note.
- **Peak activation memory** follows the sequential rule (Eq. 4) over convolution, linear and pooling layers. Skip connections and runtime arena overhead are not modelled, and attention score tensors are excluded unless `include_scores=True` is passed to `profile_model`.
- **Not implemented:** quantisation-aware training, H2 (needs per-layer logs; `benchmark.layer_times` provides them) and H6 (needs the full factor grid), and energy measurement.

## Reproducibility

Seeds, split definitions and calibration sets are fixed in code. Report the contents of `environment.json` with every result and distinguish proxy (host CPU) measurements from on-target ones.
