"""Training with a fixed budget, knowledge distillation (Eq. 16) and pruning hooks."""
from __future__ import annotations

import copy
import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distillation_loss(logits_s, logits_t, y, lam: float = 0.5, T: float = 4.0):
    """Eq. 16: (1 - lam) * CE(y, p_s) + lam * T^2 * KL(p_t^T || p_s^T)."""
    ce = F.cross_entropy(logits_s, y)
    kl = F.kl_div(F.log_softmax(logits_s / T, dim=-1), F.softmax(logits_t / T, dim=-1),
                  reduction="batchmean")
    return (1 - lam) * ce + lam * (T ** 2) * kl


def make_optimizer(model: nn.Module, family: str, epochs: int, lr_scale: float = 1.0):
    if family == "transformer":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3 * lr_scale, weight_decay=0.05)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.05 * lr_scale, momentum=0.9,
                              nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    return opt, sched


@torch.no_grad()
def predict(model: nn.Module, loader, device, dtype: Optional[torch.dtype] = None):
    """Return (predictions, labels) as numpy arrays."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device=device, dtype=dtype) if dtype is not None else x.to(device)
        preds.append(model(x).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def train(model: nn.Module, data: dict, epochs: int, family: str, device, seed: int = 0,
          teacher: Optional[nn.Module] = None, kd_lambda: float = 0.5, kd_T: float = 4.0,
          lr_scale: float = 1.0, on_epoch_start: Optional[Callable[[int], None]] = None,
          post_step: Optional[Callable[[], None]] = None, verbose: bool = True):
    """Train for a fixed number of epochs; return the model with the best-validation weights.

    teacher        if given, the student is trained with the distillation loss
    on_epoch_start called with the epoch index (used for the pruning schedule)
    post_step      called after every optimiser step (used to re-apply pruning masks)
    """
    set_seed(seed)
    model.to(device)
    opt, sched = make_optimizer(model, family, epochs, lr_scale)
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        model.train()
        total, count = 0.0, 0
        for x, y in data["train"]:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = distillation_loss(out, t_out, y, kd_lambda, kd_T)
            else:
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if family == "transformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if post_step is not None:
                post_step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        sched.step()
        pred, lab = predict(model, data["val"], device)
        acc = float((pred == lab).mean())
        history.append(dict(epoch=epoch, loss=total / max(1, count), val_acc=acc))
        if verbose:
            print(f"    epoch {epoch + 1:3d}/{epochs}  loss {total / max(1, count):.4f}  val acc {acc:.4f}")
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
