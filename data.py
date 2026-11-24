"""Datasets (Section 7.2): CIFAR-10 and 12-class Speech Commands.

Splits follow the paper: a validation split is held out from the training data
for model selection, calibration sets for quantisation are drawn from the
training split, and the test split is used only for final reporting.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

SPLIT_SEED = 1234     # fixed data split, independent of the training seed

# ----------------------------------------------------------------------------
# CIFAR-10
# ----------------------------------------------------------------------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10(root: str, res: int, batch_size: int = 128, num_workers: int = 2,
                val_size: int = 5000, calib_size: int = 500,
                subset: Optional[int] = None) -> Dict[str, DataLoader]:
    import torchvision
    import torchvision.transforms as T

    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    resize = [T.Resize(res, antialias=True)] if res != 32 else []
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *resize, T.ToTensor(), norm])
    eval_tf = T.Compose([*resize, T.ToTensor(), norm])

    train_full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    eval_full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    calib_idx = train_idx[:calib_size]
    test_idx = list(range(len(test)))
    if subset is not None:
        train_idx, val_idx, test_idx = train_idx[:subset], val_idx[:max(1, subset // 4)], test_idx[:max(1, subset // 2)]
        calib_idx = calib_idx[:min(calib_size, subset)]

    dl = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dict(
        train=dl(Subset(train_full, train_idx), True),
        val=dl(Subset(eval_full, val_idx), False),
        test=dl(Subset(test, test_idx), False),
        calib=dl(Subset(eval_full, calib_idx), False),
    )


# ----------------------------------------------------------------------------
# Speech Commands (12 classes: silence, unknown, ten keywords)
# ----------------------------------------------------------------------------

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = ["silence", "unknown"] + KEYWORDS
SR = 16000
N_MFCC = 10
KWS_VERSION = "speech_commands_v0.01"      # ~65k utterances, 30 words (as in the paper)


def _fix_len(wav: torch.Tensor) -> torch.Tensor:
    wav = wav[0]
    if wav.numel() < SR:
        wav = F.pad(wav, (0, SR - wav.numel()))
    return wav[:SR]


def _mfcc_transform():
    import torchaudio
    # 40 ms window, 20 ms hop, no centring -> 49 frames per one-second clip
    return torchaudio.transforms.MFCC(
        sample_rate=SR, n_mfcc=N_MFCC,
        melkwargs=dict(n_fft=640, win_length=640, hop_length=320, n_mels=40, center=False))


def _kws_split(root: str, subset: str, unknown_pct: float = 0.1, silence_pct: float = 0.1):
    """Return dict(x=(N,49,10) float32, y=(N,) long) for one split, cached on disk."""
    import torchaudio
    cache = os.path.join(root, f"kws_{KWS_VERSION}_{subset}.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    ds = torchaudio.datasets.SPEECHCOMMANDS(root, url=KWS_VERSION, download=True, subset=subset)
    mfcc = _mfcc_transform()
    kw_x, kw_y, unk_x = [], [], []
    for wav, _sr, label, *_ in ds:
        f = mfcc(_fix_len(wav).unsqueeze(0))[0].T          # (time=49, n_mfcc=10)
        if label in KEYWORDS:
            kw_x.append(f)
            kw_y.append(LABELS.index(label))
        else:
            unk_x.append(f)

    g = torch.Generator().manual_seed(SPLIT_SEED + len(subset))
    n_kw = len(kw_x)
    n_unk = min(len(unk_x), int(unknown_pct * n_kw))
    n_sil = int(silence_pct * n_kw)
    unk_sel = torch.randperm(len(unk_x), generator=g)[:n_unk].tolist()

    # silence: random one-second crops of the background-noise recordings, random volume
    noise_files = sorted(glob.glob(os.path.join(ds._path, "_background_noise_", "*.wav")))
    noise = [torchaudio.load(p)[0][0] for p in noise_files]
    sil_x = []
    for _ in range(n_sil):
        w = noise[int(torch.randint(len(noise), (1,), generator=g))]
        off = int(torch.randint(0, w.numel() - SR, (1,), generator=g))
        vol = float(torch.rand(1, generator=g))
        sil_x.append(mfcc((w[off:off + SR] * vol).unsqueeze(0))[0].T)

    x = torch.stack(kw_x + [unk_x[i] for i in unk_sel] + sil_x)
    y = torch.tensor(kw_y + [LABELS.index("unknown")] * n_unk + [LABELS.index("silence")] * n_sil)
    out = dict(x=x.float(), y=y.long())
    torch.save(out, cache)
    return out


def get_kws(root: str, res: int = 49, batch_size: int = 128, num_workers: int = 0,
            calib_size: int = 500, subset: Optional[int] = None) -> Dict[str, DataLoader]:
    """`res` is the number of MFCC frames (49 is the native size)."""
    splits = {s: _kws_split(root, s) for s in ("training", "validation", "testing")}
    mean = splits["training"]["x"].mean(dim=(0, 1), keepdim=True)
    std = splits["training"]["x"].std(dim=(0, 1), keepdim=True) + 1e-5

    def prep(split, n=None):
        x = ((split["x"] - mean) / std).unsqueeze(1)           # (N,1,49,10)
        y = split["y"]
        if n is not None:
            idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(SPLIT_SEED))[:n]
            x, y = x[idx], y[idx]
        if res != x.shape[2]:
            x = F.interpolate(x, size=(res, N_MFCC), mode="bilinear", align_corners=False)
        return x.contiguous(), y

    ntr = subset
    nva = max(1, subset // 4) if subset else None
    nte = max(1, subset // 2) if subset else None
    xtr, ytr = prep(splits["training"], ntr)
    xva, yva = prep(splits["validation"], nva)
    xte, yte = prep(splits["testing"], nte)
    ncal = min(calib_size, len(xtr))

    dl = lambda x, y, shuffle: DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)
    return dict(train=dl(xtr, ytr, True), val=dl(xva, yva, False), test=dl(xte, yte, False),
                calib=dl(xtr[:ncal], ytr[:ncal], False))


def get_data(task: str, root: str, res: int, batch_size: int = 128, subset: Optional[int] = None):
    if task == "cifar10":
        return get_cifar10(os.path.join(root, "cifar10"), res, batch_size, subset=subset)
    if task == "kws":
        return get_kws(os.path.join(root, "kws"), res, batch_size, subset=subset)
    raise ValueError(task)
