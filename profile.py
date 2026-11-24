"""Complexity measures of Section 3.2: parameters, MACs, peak activation memory, serialised size.

Profile the FP32 model (before quantisation). Peak activation memory follows the
sequential rule of Eq. 4 over Conv2d, Linear and pooling layers; BN and
activation functions are assumed fused, and skip connections / runtime arena
overhead are not modelled.
"""
from __future__ import annotations

import io

import torch
import torch.nn as nn

PEAK_TYPES = (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model: nn.Module) -> int:
    return sum(int((p != 0).sum()) for p in model.parameters())


@torch.no_grad()
def profile_model(model: nn.Module, example_input: torch.Tensor, act_bits: int = 8,
                  include_scores: bool = False) -> dict:
    """MACs (per sample) and peak activation memory of one forward pass."""
    model = model.cpu().eval()
    state = {"macs": 0, "peak": 0}

    def hook(mod, inp, out):
        B = out.shape[0]
        in_el, out_el = inp[0].numel() // B, out.numel() // B
        if isinstance(mod, nn.Conv2d):
            state["macs"] += out_el * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            state["macs"] += out_el * mod.in_features
        state["peak"] = max(state["peak"], in_el + out_el)

    def attn_hook(mod, inp, out):
        B = out.shape[0]
        state["macs"] += mod.extra_macs
        if include_scores:
            state["peak"] = max(state["peak"], 2 * (inp[0].numel() // B) + mod.score_elems)

    handles = []
    for m in model.modules():
        if isinstance(m, PEAK_TYPES):
            handles.append(m.register_forward_hook(hook))
        if hasattr(m, "extra_macs"):
            handles.append(m.register_forward_hook(attn_hook))
    model(example_input.cpu())
    for h in handles:
        h.remove()

    return dict(params=count_params(model), nonzero_params=count_nonzero(model), macs=state["macs"],
                peak_act_elems=state["peak"], peak_act_bytes=state["peak"] * act_bits / 8)


def serialized_size_bytes(model: nn.Module) -> int:
    """Size of the serialised state dict; approximates flash use (Section 7.4)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
