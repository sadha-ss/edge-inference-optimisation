"""Pruning (Section 4.2).

Structured pruning physically removes filters/channels, so the pruned network is
smaller and faster on ordinary dense kernels. Unstructured pruning zeroes
individual weights with a gradual schedule (Zhu and Gupta, 2017); on dense
kernels it does not reduce latency (hypothesis H4).

Structured pruning targets, per family:
    dscnn        every convolutional stage (stem, pointwise outputs and the
                 matching depthwise channels, classifier inputs)
    resnet       the hidden channels inside each residual block (conv1 outputs,
                 conv2 inputs); block outputs are kept so skip connections stay valid
    transformer  the hidden units of each feed-forward network
Filters are ranked by the L1 norm of their weights (Li et al., 2017).
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# helpers for slicing layers
# ----------------------------------------------------------------------------


def _n_keep(c: int, ratio: float, minimum: int = 4) -> int:
    return max(minimum, int(round(c * (1.0 - ratio))))


def _l1_keep(weight: torch.Tensor, n_keep: int) -> torch.Tensor:
    scores = weight.detach().abs().reshape(weight.shape[0], -1).sum(1)
    idx = torch.argsort(scores, descending=True)[:n_keep]
    return torch.sort(idx).values


def _slice_conv(conv: nn.Conv2d, out_idx=None, in_idx=None, depthwise: bool = False) -> nn.Conv2d:
    w = conv.weight.data
    if depthwise:
        idx = in_idx
        new = nn.Conv2d(len(idx), len(idx), conv.kernel_size, conv.stride, conv.padding,
                        groups=len(idx), bias=conv.bias is not None)
        new.weight.data = w[idx].clone()
        if conv.bias is not None:
            new.bias.data = conv.bias.data[idx].clone()
        return new
    b = conv.bias.data if conv.bias is not None else None
    if out_idx is not None:
        w = w[out_idx]
        b = b[out_idx] if b is not None else None
    if in_idx is not None:
        w = w[:, in_idx]
    new = nn.Conv2d(w.shape[1], w.shape[0], conv.kernel_size, conv.stride, conv.padding,
                    groups=1, bias=b is not None)
    new.weight.data = w.clone()
    if b is not None:
        new.bias.data = b.clone()
    return new


def _slice_bn(bn: nn.BatchNorm2d, idx) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(len(idx), eps=bn.eps, momentum=bn.momentum)
    new.weight.data = bn.weight.data[idx].clone()
    new.bias.data = bn.bias.data[idx].clone()
    new.running_mean = bn.running_mean[idx].clone()
    new.running_var = bn.running_var[idx].clone()
    return new


def _slice_linear(lin: nn.Linear, out_idx=None, in_idx=None) -> nn.Linear:
    w, b = lin.weight.data, (lin.bias.data if lin.bias is not None else None)
    if out_idx is not None:
        w = w[out_idx]
        b = b[out_idx] if b is not None else None
    if in_idx is not None:
        w = w[:, in_idx]
    new = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    new.weight.data = w.clone()
    if b is not None:
        new.bias.data = b.clone()
    return new


# ----------------------------------------------------------------------------
# structured pruning
# ----------------------------------------------------------------------------


def prune_structured(model: nn.Module, ratio: float) -> nn.Module:
    """Return a smaller copy of `model` with a fraction `ratio` of channels removed."""
    m = copy.deepcopy(model)
    name = type(m).__name__
    if name == "DSCNN":
        _prune_dscnn(m, ratio)
    elif name == "ResNetSmall":
        _prune_resnet(m, ratio)
    elif name == "CompactTransformer":
        _prune_transformer(m, ratio)
    else:
        raise ValueError(f"structured pruning not implemented for {name}")
    return m


def _prune_dscnn(m, ratio):
    keeps: List[torch.Tensor] = [_l1_keep(m.stem_conv.weight, _n_keep(m.stem_conv.out_channels, ratio))]
    for b in m.blocks:
        keeps.append(_l1_keep(b.pw.weight, _n_keep(b.pw.out_channels, ratio)))
    m.stem_conv = _slice_conv(m.stem_conv, out_idx=keeps[0])
    m.stem_bn = _slice_bn(m.stem_bn, keeps[0])
    for i, b in enumerate(m.blocks):
        k_in, k_out = keeps[i], keeps[i + 1]
        b.dw = _slice_conv(b.dw, in_idx=k_in, depthwise=True)
        b.bn1 = _slice_bn(b.bn1, k_in)
        b.pw = _slice_conv(b.pw, out_idx=k_out, in_idx=k_in)
        b.bn2 = _slice_bn(b.bn2, k_out)
    m.fc = _slice_linear(m.fc, in_idx=keeps[-1])


def _prune_resnet(m, ratio):
    for blk in m.blocks:
        keep = _l1_keep(blk.conv1.weight, _n_keep(blk.conv1.out_channels, ratio))
        blk.conv1 = _slice_conv(blk.conv1, out_idx=keep)
        blk.bn1 = _slice_bn(blk.bn1, keep)
        blk.conv2 = _slice_conv(blk.conv2, in_idx=keep)


def _prune_transformer(m, ratio):
    for blk in m.blocks:
        keep = _l1_keep(blk.ff1.weight, _n_keep(blk.ff1.out_features, ratio, minimum=8))
        blk.ff1 = _slice_linear(blk.ff1, out_idx=keep)
        blk.ff2 = _slice_linear(blk.ff2, in_idx=keep)


# ----------------------------------------------------------------------------
# unstructured magnitude pruning with a gradual schedule
# ----------------------------------------------------------------------------


class MagnitudePruner:
    """Per-layer magnitude pruning; sparsity follows s_f * (1 - (1 - t)^3) between start and end epoch."""

    def __init__(self, model: nn.Module, final_sparsity: float, start_epoch: int = 0,
                 end_epoch: int = 1, skip_first_last: bool = True):
        params = [(n, p) for n, p in model.named_parameters() if p.dim() > 1 and not n.endswith("pos")]
        if skip_first_last and len(params) > 2:
            params = params[1:-1]
        self.params: Dict[str, torch.nn.Parameter] = dict(params)
        self.masks = {n: torch.ones_like(p, dtype=torch.bool) for n, p in params}
        self.final, self.start, self.end = final_sparsity, start_epoch, max(end_epoch, start_epoch + 1)

    def sparsity_at(self, epoch: int) -> float:
        if epoch < self.start:
            return 0.0
        if epoch >= self.end:
            return self.final
        t = (epoch - self.start) / (self.end - self.start)
        return self.final * (1.0 - (1.0 - t) ** 3)

    @torch.no_grad()
    def update(self, epoch: int) -> None:
        s = self.sparsity_at(epoch)
        for n, p in self.params.items():
            k = int(s * p.numel())
            if k < 1:
                continue
            thr = torch.kthvalue(p.abs().flatten(), k).values
            self.masks[n] = p.abs() > thr
        self.apply()

    @torch.no_grad()
    def apply(self) -> None:
        for n, p in self.params.items():
            p.mul_(self.masks[n])

    def sparsity(self) -> float:
        zeros = sum(int((p == 0).sum()) for p in self.params.values())
        total = sum(p.numel() for p in self.params.values())
        return zeros / max(1, total)
