"""Model families used in the experiments (Section 8.2, Table 23).

Three families, each with a width multiplier and a task-dependent input resolution:

    dscnn        depthwise-separable CNN (MobileNet / Hello Edge style)
    resnet       small residual CNN (ResNet-8 style, one basic block per stage)
    transformer  compact transformer encoder (patch embedding for images,
                 frame embedding for MFCC sequences)

Resolution means: image side in pixels for CIFAR-10 (32, 24, 16) and number of
MFCC frames for keyword spotting (49, 40, 32; 10 coefficients per frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

N_MFCC = 10

TASK_SPECS = {
    "cifar10": dict(n_classes=10, in_ch=3, resolutions=(32, 24, 16)),
    "kws": dict(n_classes=12, in_ch=1, resolutions=(49, 40, 32)),
}


def scale_ch(c: int, width: float, divisor: int = 1, minimum: int = 4) -> int:
    """Scale a channel count by `width`, keeping it a multiple of `divisor`."""
    v = int(round(c * width / divisor)) * divisor
    return max(minimum, v)


@dataclass(frozen=True)
class ModelConfig:
    family: str
    task: str
    width: float = 1.0
    res: int = 32

    @property
    def n_classes(self) -> int:
        return TASK_SPECS[self.task]["n_classes"]

    @property
    def in_ch(self) -> int:
        return TASK_SPECS[self.task]["in_ch"]

    def input_shape(self, batch: int = 1):
        if self.task == "cifar10":
            return (batch, 3, self.res, self.res)
        return (batch, 1, self.res, N_MFCC)

    def key(self) -> str:
        return f"{self.task}_{self.family}_w{self.width:g}_r{self.res}"


# ----------------------------------------------------------------------------
# Depthwise-separable CNN
# ----------------------------------------------------------------------------


class DSBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each followed by BN and ReLU."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.ReLU()
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        return self.act2(self.bn2(self.pw(x)))


class DSCNN(nn.Module):
    def __init__(self, in_ch, n_classes, width=1.0, channels=(64, 64, 64, 64),
                 strides=(1, 1, 1, 1), stem_kernel=5, stem_stride=2):
        super().__init__()
        chs = [scale_ch(c, width) for c in channels]
        stem_c = chs[0]
        self.stem_conv = nn.Conv2d(in_ch, stem_c, stem_kernel, stem_stride, stem_kernel // 2, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_c)
        self.stem_act = nn.ReLU()
        blocks, cin = [], stem_c
        for c, s in zip(chs, strides):
            blocks.append(DSBlock(cin, c, s))
            cin = c
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for b in self.blocks:
            x = b(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


# ----------------------------------------------------------------------------
# Residual CNN
# ----------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.shortcut = None
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = x if self.shortcut is None else self.shortcut(x)
        return self.act2(out + sc)


class ResNetSmall(nn.Module):
    def __init__(self, in_ch, n_classes, width=1.0, channels=(32, 64, 128), strides=(1, 2, 2)):
        super().__init__()
        chs = [scale_ch(c, width) for c in channels]
        self.stem_conv = nn.Conv2d(in_ch, chs[0], 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(chs[0])
        self.stem_act = nn.ReLU()
        blocks, cin = [], chs[0]
        for c, s in zip(chs, strides):
            blocks.append(BasicBlock(cin, c, s))
            cin = c
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for b in self.blocks:
            x = b(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


# ----------------------------------------------------------------------------
# Compact transformer
# ----------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.scale = (d // heads) ** -0.5
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        # per-sample statistics set during forward, read by profile.py
        self.extra_macs = 0
        self.score_elems = 0

    def forward(self, x):
        B, n, d = x.shape
        qkv = self.qkv(x).reshape(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, n, d)
        self.extra_macs = 2 * n * n * d          # QK^T and attention x V (Eq. 9)
        self.score_elems = self.h * n * n
        return self.proj(out)


class TBlock(nn.Module):
    def __init__(self, d: int, heads: int, ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ratio * d)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(ratio * d, d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff2(self.act(self.ff1(self.norm2(x))))


class CompactTransformer(nn.Module):
    def __init__(self, task, res, n_classes, width=1.0, base_d=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.task = task
        d = scale_ch(base_d, width, divisor=heads, minimum=2 * heads)
        if task == "cifar10":
            assert res % patch == 0, "resolution must be a multiple of the patch size"
            self.embed = nn.Conv2d(3, d, patch, patch)
            n_tokens = (res // patch) ** 2
        else:
            self.embed = nn.Linear(N_MFCC, d)
            n_tokens = res
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        if self.task == "cifar10":
            x = self.embed(x).flatten(2).transpose(1, 2)
        else:
            x = self.embed(x.squeeze(1))
        x = x + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x).mean(1))


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.family == "dscnn":
        if cfg.task == "cifar10":
            return DSCNN(cfg.in_ch, cfg.n_classes, cfg.width, channels=(64, 128, 128, 256),
                         strides=(1, 2, 1, 2), stem_kernel=3, stem_stride=1)
        return DSCNN(cfg.in_ch, cfg.n_classes, cfg.width, channels=(64, 64, 64, 64),
                     strides=(1, 1, 1, 1), stem_kernel=5, stem_stride=2)
    if cfg.family == "resnet":
        return ResNetSmall(cfg.in_ch, cfg.n_classes, cfg.width)
    if cfg.family == "transformer":
        return CompactTransformer(cfg.task, cfg.res, cfg.n_classes, cfg.width)
    raise ValueError(f"unknown family {cfg.family!r}")
