"""Quantisation (Section 4.3): FP16 and post-training INT8.

CNNs use static post-training quantisation (FX graph mode, per-channel weights,
calibration on a small training subset). Transformers use dynamic INT8
quantisation of the linear layers, because softmax/LayerNorm blocks are not
supported well by static FX quantisation. Quantised models run on the CPU
quantised kernels (x86 / fbgemm / qnnpack depending on the build).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _engine() -> str:
    supported = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):
        if e in supported:
            return e
    raise RuntimeError(f"no quantised engine available: {supported}")


def to_fp16(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).cpu().half().eval()


def ptq_static_int8(model: nn.Module, calib_loader, example_input: torch.Tensor, n_batches: int = 10):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    engine = _engine()
    torch.backends.quantized.engine = engine
    m = copy.deepcopy(model).cpu().eval()
    prepared = prepare_fx(m, get_default_qconfig_mapping(engine), example_inputs=(example_input.cpu(),))
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            prepared(x.cpu())
            if i + 1 >= n_batches:
                break
    return convert_fx(prepared)


def dynamic_int8_linear(model: nn.Module) -> nn.Module:
    from torch.ao.quantization import quantize_dynamic
    torch.backends.quantized.engine = _engine()
    return quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)


def quantize_int8(model: nn.Module, family: str, calib_loader, example_input: torch.Tensor):
    if family == "transformer":
        return dynamic_int8_linear(model)
    return ptq_static_int8(model, calib_loader, example_input)
