#!/usr/bin/env python3
"""
edge_calculator.py
==================
Analytical calculator for the paper
It reproduces the *analytical* results of the paper (Sections 3-6 and
Appendix C/F): layer-by-layer parameter, MAC and activation counts for the three
reference networks, the roofline latency model (Eq. 6), memory model
(Eqs. 3-5), first-order energy model (Eq. 14), width/resolution/precision
sweeps, microcontroller feasibility maps, the worked scenarios S1-S3,
transformer scaling, batch-size effects, sparse-storage break-even,
Amdahl's law and the quantisation SNR simulation.

IMPORTANT: everything here is computed from layer shapes and ASSUMED device
parameters. No measurement on physical hardware is involved.

Only the Python standard library is required (Python >= 3.8).

Usage
-----
    python edge_calculator.py              # print all reports
    python edge_calculator.py --verify     # check results against the paper
    python edge_calculator.py --csv out/   # also write sweep tables as CSV
    python edge_calculator.py --device-file my_devices.json
        (JSON: {"My MCU": {"pi_gmacs": 0.1, "beta_gbs": 0.4,
                           "ram_kb": 512, "flash_kb": 2048}})
"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class Layer:
    """One layer, with the quantities needed by the cost model."""
    name: str
    kind: str                      # conv | depthwise | pointwise | fc | pool | attn ...
    out_shape: Tuple[int, ...]
    params: int
    macs: int
    in_elems: int
    out_elems: int
    in_peak: bool = True           # include in/out tensors in the peak-activation rule


@dataclass
class Device:
    """Assumed device parameters (Table 11)."""
    name: str
    pi: float        # peak compute rate, MAC/s
    beta: float      # memory bandwidth, bytes/s
    ram: float       # bytes
    flash: float     # bytes


DEVICES: Dict[str, Device] = {
    "MCU-class":  Device("MCU-class", 0.05e9, 0.2e9, 256e3, 1.0e6),
    "Mobile SoC": Device("Mobile SoC", 10e9, 5e9, 2000e6, 16000e6),
    "Edge accel.": Device("Edge accel.", 100e9, 10e9, 512e6, 4000e6),
}

# Approximate energy per operation, 45 nm (Horowitz, 2014), in picojoules.
E_FP32_MAC = 3.7 + 0.9
E_INT8_MAC = 0.2 + 0.03
E_DRAM_32BIT = 640.0


# --------------------------------------------------------------------------
# Network builders
# --------------------------------------------------------------------------


def _out(n: int, stride: int) -> int:
    """Output size with 'same' padding."""
    return -(-n // stride)


def _ch(c: int, alpha: float) -> int:
    return max(1, int(round(c * alpha)))


def build_mobilenet_v1(alpha: float = 1.0, res: int = 224,
                       n_classes: int = 1000) -> List[Layer]:
    """Network A: MobileNetV1 (Howard et al., 2017); BN folded, biases counted."""
    layers: List[Layer] = []
    h = w = res
    c = 3
    c1 = _ch(32, alpha)
    ho = wo = _out(h, 2)
    layers.append(Layer("conv1", "conv", (ho, wo, c1), 9 * c * c1 + c1,
                        ho * wo * c1 * 9 * c, h * w * c, ho * wo * c1))
    h, w, c = ho, wo, c1

    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)] \
        + [(512, 1)] * 5 + [(1024, 2), (1024, 1)]
    for i, (cout_base, stride) in enumerate(cfg, start=1):
        cout = _ch(cout_base, alpha)
        ho, wo = _out(h, stride), _out(w, stride)
        layers.append(Layer(f"dw{i}", "depthwise", (ho, wo, c), 9 * c + c,
                            ho * wo * c * 9, h * w * c, ho * wo * c))
        layers.append(Layer(f"pw{i}", "pointwise", (ho, wo, cout), c * cout + cout,
                            ho * wo * cout * c, ho * wo * c, ho * wo * cout))
        h, w, c = ho, wo, cout
    layers.append(Layer("avgpool", "pool", (1, 1, c), 0, 0, h * w * c, c))
    layers.append(Layer("fc", "fc", (n_classes,), c * n_classes + n_classes,
                        c * n_classes, c, n_classes))
    return layers


def build_kws_dscnn(n_classes: int = 12, channels: int = 64,
                    n_blocks: int = 4) -> List[Layer]:
    """Network B: depthwise-separable keyword-spotting CNN (49x10x1 input)."""
    layers: List[Layer] = []
    h, w, cin = 49, 10, 1
    ho, wo = _out(h, 2), _out(w, 2)
    layers.append(Layer("conv1", "conv", (ho, wo, channels), 25 * cin * channels + channels,
                        ho * wo * channels * 25 * cin, h * w * cin, ho * wo * channels))
    h, w, c = ho, wo, channels
    for i in range(1, n_blocks + 1):
        layers.append(Layer(f"dw{i}", "depthwise", (h, w, c), 9 * c + c,
                            h * w * c * 9, h * w * c, h * w * c))
        layers.append(Layer(f"pw{i}", "pointwise", (h, w, c), c * c + c,
                            h * w * c * c, h * w * c, h * w * c))
    layers.append(Layer("avgpool", "pool", (1, 1, c), 0, 0, h * w * c, c))
    layers.append(Layer("fc", "fc", (n_classes,), c * n_classes + n_classes,
                        c * n_classes, c, n_classes))
    return layers


def build_tiny_transformer(d: int = 128, n_layers: int = 4, n: int = 64,
                           heads: int = 4, n_classes: int = 12,
                           include_scores_in_peak: bool = False) -> List[Layer]:
    """
    Network C: small transformer encoder (Eqs. 8-9).

    Each encoder layer is split into six sub-layers (QKV projection, score
    computation, attention-weighted sum, output projection, two feed-forward
    matrices). Parameters per layer: 12 d^2 + 9 d (weights + biases);
    MACs per layer: 12 n d^2 + 2 n^2 d.

    The paper's Table 17 peak-activation values are reproduced when the n x n
    score tensors are NOT counted in the peak (default). Set
    include_scores_in_peak=True to include them (relevant for long sequences).
    """
    layers: List[Layer] = []
    nd = n * d
    ns = n * n * heads
    for i in range(1, n_layers + 1):
        layers += [
            Layer(f"L{i}.qkv", "attn", (n, 3 * d), 3 * d * d + 3 * d, 3 * n * d * d, nd, 3 * nd),
            Layer(f"L{i}.scores", "attn", (heads, n, n), 0, n * n * d, 2 * nd, ns,
                  in_peak=include_scores_in_peak),
            Layer(f"L{i}.attnV", "attn", (n, d), 0, n * n * d, ns + nd, nd,
                  in_peak=include_scores_in_peak),
            Layer(f"L{i}.out", "attn", (n, d), d * d + d, n * d * d, nd, nd),
            Layer(f"L{i}.ff1", "fc", (n, 4 * d), 4 * d * d + 4 * d, 4 * n * d * d, nd, 4 * nd),
            Layer(f"L{i}.ff2", "fc", (n, d), 4 * d * d + d, 4 * n * d * d, 4 * nd, nd),
        ]
    layers.append(Layer("pool", "pool", (d,), 0, 0, nd, d))
    layers.append(Layer("fc", "fc", (n_classes,), d * n_classes + n_classes,
                        d * n_classes, d, n_classes))
    return layers


# --------------------------------------------------------------------------
# Cost model (Section 3)
# --------------------------------------------------------------------------


def total_params(layers: List[Layer]) -> int:
    return sum(l.params for l in layers)


def total_macs(layers: List[Layer]) -> int:
    return sum(l.macs for l in layers)


def weight_bytes(layers: List[Layer], bw: float) -> float:
    """Eq. 3: M_w = P * b_w / 8."""
    return total_params(layers) * bw / 8


def peak_activation_bytes(layers: List[Layer], ba: float) -> float:
    """Eq. 4: sequential rule, max over layers of (|a_{l-1}| + |a_l|) * b_a / 8."""
    return max((l.in_elems + l.out_elems) for l in layers if l.in_peak) * ba / 8


def layer_bytes(l: Layer, bw: float, ba: float) -> float:
    """Bytes moved by a layer: weights, input and output each counted once."""
    return l.params * bw / 8 + (l.in_elems + l.out_elems) * ba / 8


def arithmetic_intensity(l: Layer, bw: float, ba: float) -> float:
    b = layer_bytes(l, bw, ba)
    return l.macs / b if b else 0.0


def latency(layers: List[Layer], dev: Device, bw: float, ba: float,
            peak_mult: float = 1.0, bw_scale: float = 1.0, pi_scale: float = 1.0,
            aggregated: bool = False, overhead_s: float = 0.0) -> Tuple[float, float]:
    """
    Roofline latency (Eq. 6). Returns (seconds, share of time in memory-bound layers).

    peak_mult  : speed-up of the compute peak (e.g. 4 for INT8 kernels)
    aggregated : if True, use max(sum MACs / Pi, sum bytes / beta) over the whole
                 network (used for the transformer, Table 17); here the traffic
                 is taken as the weights only, which matches the paper's
                 reported values (compute-bound in every row).
    """
    pi = dev.pi * pi_scale * peak_mult
    beta = dev.beta * bw_scale
    if aggregated:
        t_c = total_macs(layers) / pi
        t_m = weight_bytes(layers, bw) / beta
        return max(t_c, t_m) + overhead_s, (1.0 if t_m > t_c else 0.0)
    total = mem_time = 0.0
    for l in layers:
        t_c = l.macs / pi
        t_m = layer_bytes(l, bw, ba) / beta
        t = max(t_c, t_m)
        total += t
        if t_m > t_c:
            mem_time += t
    return total + overhead_s, (mem_time / total if total else 0.0)


def energy_joules(layers: List[Layer], bw: int, e_mac_pj: float) -> Tuple[float, float]:
    """Eq. 14 (weights only): arithmetic and weight-traffic energy, in joules."""
    arith = total_macs(layers) * e_mac_pj * 1e-12
    words = weight_bytes(layers, bw) / 4        # 32-bit DRAM words
    return arith, words * E_DRAM_32BIT * 1e-12


# --------------------------------------------------------------------------
# Further analytical results (Sections 3.5-3.8, 5.10)
# --------------------------------------------------------------------------


def attention_share(d: int, n: int) -> float:
    """Share of per-layer MACs spent in attention (Eq. 9)."""
    a = 2 * n * n * d
    return a / (12 * n * d * d + a)


def sparse_ratio(s: float, bw: int, bi: int) -> float:
    """Eq. 12: sparse storage relative to dense at sparsity s."""
    return (1 - s) * (bw + bi) / bw


def sparse_breakeven(bw: int, bi: int) -> float:
    return bi / (bw + bi)


def amdahl(f: float, k: float) -> float:
    """Eq. 13."""
    return 1.0 / ((1 - f) + f / k)


def batch_fc_table(dev: Device, cin: int = 1024, cout: int = 1000,
                   bits: int = 8, batches=(1, 2, 8, 32, 128)):
    """Section 5.10: fully connected layer vs batch size."""
    rows = []
    bpe = bits / 8
    for B in batches:
        macs = B * cin * cout
        byt = (cin * cout + cout) * bpe + B * (cin + cout) * bpe
        inten = macs / byt
        t = max(macs / dev.pi, byt / dev.beta)
        rows.append((B, inten, min(dev.pi, inten * dev.beta) / 1e9, t / B * 1e6))
    return rows


def simulate_sqnr(bits: int, signal: str = "sine", n: int = 200_000,
                  seed: int = 0) -> float:
    """Eq. 11 check: uniform quantiser SNR in dB on a synthetic signal."""
    rng = random.Random(seed)
    if signal == "sine":
        xs = [math.sin(2 * math.pi * 0.0123 * i + 0.3) for i in range(n)]
        lo, hi = -1.0, 1.0
    else:   # gaussian clipped at 4 sigma
        xs = [max(-4.0, min(4.0, rng.gauss(0, 1))) for _ in range(n)]
        lo, hi = -4.0, 4.0
    delta = (hi - lo) / (2 ** bits)
    sig = noise = 0.0
    for x in xs:
        q = max(lo, min(hi - delta, math.floor((x - lo) / delta) * delta + lo + delta / 2))
        sig += x * x
        noise += (x - q) ** 2
    return 10 * math.log10(sig / noise)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

FMT_MB = lambda b: f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"


def _hdr(title: str):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def report_reference_networks():
    _hdr("Table 7 / 10: reference networks")
    nets = {"A: MobileNetV1 (224x224x3)": build_mobilenet_v1(),
            "B: DS-CNN (49x10x1)": build_kws_dscnn(),
            "C: Tiny transformer (64x128)": build_tiny_transformer()}
    print(f"{'network':32s}{'params':>10s}{'MACs':>10s}{'FP32':>10s}{'FP16':>10s}"
          f"{'INT8':>10s}{'INT4':>10s}{'peak act INT8':>15s}")
    for name, L in nets.items():
        print(f"{name:32s}{total_params(L)/1e6:>9.3f}M{total_macs(L)/1e6:>9.2f}M"
              f"{FMT_MB(weight_bytes(L, 32)):>10s}{FMT_MB(weight_bytes(L, 16)):>10s}"
              f"{FMT_MB(weight_bytes(L, 8)):>10s}{FMT_MB(weight_bytes(L, 4)):>10s}"
              f"{FMT_MB(peak_activation_bytes(L, 8)):>15s}")


def report_validation():
    _hdr("Table 8: MobileNetV1 counts vs. Howard et al. (2017), 224x224")
    reported = {1.0: (569, 4.2), 0.75: (325, 2.6), 0.5: (149, 1.3), 0.25: (41, 0.5)}
    print(f"{'alpha':>6s}{'MACs (comp)':>14s}{'MACs (rep)':>12s}{'Params (comp)':>16s}{'Params (rep)':>14s}")
    for a, (rm, rp) in reported.items():
        L = build_mobilenet_v1(a)
        print(f"{a:>6.2f}{total_macs(L)/1e6:>13.0f}M{rm:>11d}M{total_params(L)/1e6:>15.2f}M{rp:>13.1f}M")


def report_network_a_profile():
    _hdr("Section 5.3: resource profile of network A")
    L = build_mobilenet_v1()
    tm, tp = total_macs(L), total_params(L)
    for kind in ("pointwise", "depthwise", "conv", "fc"):
        m = sum(l.macs for l in L if l.kind == kind)
        p = sum(l.params for l in L if l.kind == kind)
        print(f"{kind:10s} MACs {100*m/tm:5.1f}%   params {100*p/tp:5.1f}%")
    peak = max(L, key=lambda l: l.in_elems + l.out_elems)
    print(f"Peak in+out tensors at layer '{peak.name}': {(peak.in_elems+peak.out_elems)/1e3:.0f} K elements "
          f"= {(peak.in_elems+peak.out_elems)/1e3:.0f} KB at INT8")


def sweep_rows(alphas=(1.0, 0.75, 0.5, 0.25), ress=(224, 192, 160, 128)):
    mcu = DEVICES["MCU-class"]
    rows = []
    for a in alphas:
        for r in ress:
            L = build_mobilenet_v1(a, r)
            flash = weight_bytes(L, 8)
            ram = peak_activation_bytes(L, 8)
            lat, _ = latency(L, mcu, 8, 8)
            over = []
            if flash > mcu.flash:
                over.append("Flash")
            if ram > mcu.ram:
                over.append("RAM")
            rows.append(dict(alpha=a, res=r, params=total_params(L), macs=total_macs(L),
                             flash_bytes=flash, peak_act_bytes=ram, mcu_latency_ms=lat * 1e3,
                             exceeds="+".join(over) if over else "fits"))
    return rows


def report_sweep(rows):
    _hdr("Table 9: network A, MACs (M) / INT8 peak activation (KB)")
    print(f"{'alpha':>6s}{'params':>10s}" + "".join(f"{r:>16d}" for r in (224, 192, 160, 128)))
    for a in (1.0, 0.75, 0.5, 0.25):
        sel = [r for r in rows if r["alpha"] == a]
        print(f"{a:>6.2f}{sel[0]['params']/1e6:>9.2f}M" +
              "".join(f"{r['macs']/1e6:>8.0f} /{r['peak_act_bytes']/1e3:>5.0f}" for r in sel))
    _hdr("Table 13: network A on the assumed MCU-class device (1 MB flash, 256 KB RAM, INT8)")
    print(f"{'alpha':>6s}" + "".join(f"{r:>18d}" for r in (224, 192, 160, 128)))
    for a in (1.0, 0.75, 0.5, 0.25):
        sel = [r for r in rows if r["alpha"] == a]
        cells = [("Fits, %.0f ms" % r["mcu_latency_ms"]) if r["exceeds"] == "fits" else r["exceeds"] for r in sel]
        print(f"{a:>6.2f}" + "".join(f"{c:>18s}" for c in cells))
    print(f"Feasible combinations: {sum(r['exceeds']=='fits' for r in rows)} of {len(rows)}")


def report_latency():
    _hdr("Table 12: estimated single-inference latency (ms), assumed devices")
    nets = {"A": build_mobilenet_v1(), "B": build_kws_dscnn(), "C": build_tiny_transformer()}
    print(f"{'network, case':20s}" + "".join(f"{d:>14s}" for d in DEVICES))
    for n, L in nets.items():
        agg = (n == "C")
        for label, bits, mult in (("FP32", 32, 1), ("INT8", 8, 1), ("INT8, 4x peak", 8, 4)):
            vals = [latency(L, d, bits, bits, mult, aggregated=agg)[0] * 1e3 for d in DEVICES.values()]
            print(f"{n+', '+label:20s}" + "".join(f"{v:>14.2f}" for v in vals))
    _hdr("Section 5.6: share of time in memory-bound layers (network A)")
    L = nets["A"]
    for dname in ("Mobile SoC", "Edge accel."):
        s32 = latency(L, DEVICES[dname], 32, 32)[1]
        s8 = latency(L, DEVICES[dname], 8, 8)[1]
        print(f"{dname:12s} FP32 {100*s32:4.0f}%   INT8 {100*s8:4.0f}%")


def report_sensitivity():
    _hdr("Table 14: sensitivity on the mobile SoC (ms)")
    soc = DEVICES["Mobile SoC"]
    A, B = build_mobilenet_v1(), build_kws_dscnn()
    variants = [("Baseline", dict()), ("Bandwidth halved", dict(bw_scale=0.5)),
                ("Peak halved", dict(pi_scale=0.5)), ("INT8 peak 4x", dict(int8_mult=4)),
                ("INT8 4x and bandwidth halved", dict(int8_mult=4, bw_scale=0.5))]
    print(f"{'variant':32s}{'A FP32':>9s}{'A INT8':>9s}{'B FP32':>9s}{'B INT8':>9s}")
    for name, kw in variants:
        m8 = kw.pop("int8_mult", 1)
        r = [latency(A, soc, 32, 32, **kw)[0], latency(A, soc, 8, 8, m8, **kw)[0],
             latency(B, soc, 32, 32, **kw)[0], latency(B, soc, 8, 8, m8, **kw)[0]]
        print(f"{name:32s}" + "".join(f"{v*1e3:>9.2f}" for v in r))


def report_pipeline():
    _hdr("Table 15: combined optimisation pipeline for network A")
    soc, mcu = DEVICES["Mobile SoC"], DEVICES["MCU-class"]
    steps = [("0: baseline (FP32, 224)", 1.0, 224, 32, 32), ("1: + input 160", 1.0, 160, 32, 32),
             ("2: + width 0.5", 0.5, 160, 32, 32), ("3: + INT8", 0.5, 160, 8, 8),
             ("4: + INT4 weights", 0.5, 160, 4, 8)]
    print(f"{'step':26s}{'MACs':>8s}{'weights':>11s}{'peak act':>11s}{'SoC ms':>10s}{'MCU s':>9s}")
    for name, a, r, bw, ba in steps:
        L = build_mobilenet_v1(a, r)
        print(f"{name:26s}{total_macs(L)/1e6:>7.0f}M{FMT_MB(weight_bytes(L, bw)):>11s}"
              f"{FMT_MB(peak_activation_bytes(L, ba)):>11s}"
              f"{latency(L, soc, bw, ba)[0]*1e3:>10.1f}{latency(L, mcu, bw, ba)[0]:>9.2f}")
    _hdr("Table 16: first-order energy for one inference of network A (weights only)")
    L = build_mobilenet_v1()
    for name, bw, e in (("FP32", 32, E_FP32_MAC), ("INT8", 8, E_INT8_MAC)):
        a, t = energy_joules(L, bw, e)
        print(f"{name:6s} arithmetic {a*1e3:5.2f} mJ   weight traffic {t*1e3:5.2f} mJ   total {(a+t)*1e3:5.2f} mJ")


def report_transformer():
    _hdr("Table 17: scaling of the tiny transformer (INT8)")
    mcu, soc = DEVICES["MCU-class"], DEVICES["Mobile SoC"]
    print(f"{'d':>5s}{'layers':>8s}{'n':>5s}{'params':>9s}{'MACs':>9s}{'attn':>7s}{'peak act':>10s}"
          f"{'MCU ms':>9s}{'SoC ms':>9s}")
    for d, nl, n in [(64, 2, 32), (64, 4, 64), (128, 2, 64), (128, 4, 64), (128, 4, 128),
                     (192, 4, 64), (192, 4, 128)]:
        L = build_tiny_transformer(d, nl, n)
        print(f"{d:>5d}{nl:>8d}{n:>5d}{total_params(L)/1e6:>8.2f}M{total_macs(L)/1e6:>8.1f}M"
              f"{100*attention_share(d, n):>6.0f}%{FMT_MB(peak_activation_bytes(L, 8)):>10s}"
              f"{latency(L, mcu, 8, 8, aggregated=True)[0]*1e3:>9.1f}"
              f"{latency(L, soc, 8, 8, aggregated=True)[0]*1e3:>9.2f}")


def report_batch():
    _hdr("Table 18: fully connected 1024x1000 layer vs batch size (INT8, mobile SoC)")
    print(f"{'batch':>6s}{'intensity':>11s}{'GMAC/s':>9s}{'us/sample':>11s}")
    for B, i, g, t in batch_fc_table(DEVICES["Mobile SoC"]):
        print(f"{B:>6d}{i:>11.1f}{g:>9.1f}{t:>11.0f}")


def scenario_report():
    _hdr("Section 6.3 / Table 20: worked scenarios (assumed limits)")
    mcu, soc = DEVICES["MCU-class"], DEVICES["Mobile SoC"]
    B = build_kws_dscnn()
    f = latency(B, mcu, 32, 32)[0] * 1e3
    q = latency(B, mcu, 8, 8)[0] * 1e3
    q4 = latency(B, mcu, 8, 8, 4)[0] * 1e3
    print(f"S1 (100 ms, 1 MB, 256 KB): network B FP32 {f:.1f} ms "
          f"(weights {weight_bytes(B,32)/1e3:.0f} KB, peak act {peak_activation_bytes(B,32)/1e3:.0f} KB), "
          f"INT8 {q:.1f} ms, INT8 4x peak {q4:.1f} ms")
    print(f"   duty cycle at 100 ms hop: FP32 {100*f/100:.0f}%, INT8 4x {100*q4/100:.0f}%")
    feas = []
    for a in (1.0, 0.75, 0.5, 0.25):
        for r in (224, 192, 160, 128):
            L = build_mobilenet_v1(a, r)
            t = latency(L, soc, 8, 8)[0] * 1e3
            if t <= 33.0:
                feas.append((total_macs(L), a, r, t))
    best = max(feas)
    print(f"S2 (33 ms, mobile SoC): unmodified A INT8 = {latency(build_mobilenet_v1(), soc, 8, 8)[0]*1e3:.1f} ms; "
          f"{len(feas)} of 16 combinations meet it; largest: alpha={best[1]}, {best[2]} px, "
          f"{best[0]/1e6:.0f} M MACs, {best[3]:.1f} ms")
    C = build_tiny_transformer()
    C2 = build_tiny_transformer(64, 2, 32)
    print(f"S3 (250 ms, MCU): C (d=128, 4 layers) INT8 "
          f"{latency(C, mcu, 8, 8, aggregated=True)[0]*1e3:.0f} ms -> infeasible; reduced (d=64, 2 layers, 32 tokens) "
          f"{latency(C2, mcu, 8, 8, aggregated=True)[0]*1e3:.1f} ms, "
          f"{total_macs(C)/1e6:.1f} M -> {total_macs(C2)/1e6:.1f} M MACs, "
          f"{weight_bytes(C2, 8)/1e3:.0f} KB weights")


def report_misc():
    _hdr("Section 3.7 / 3.8 / 3.6: sparse storage, Amdahl's law, quantisation noise")
    for bi in (4, 8, 16):
        print(f"index {bi:>2d} bit: break-even sparsity (8-bit weights) = {100*sparse_breakeven(8, bi):.0f}%")
    print(f"Amdahl: f=0.8, k=inf -> max speed-up {amdahl(0.8, 1e12):.2f}; f=0.8, k=4 -> {amdahl(0.8, 4):.2f}")
    print(f"{'bits':>5s}{'theory dB':>11s}{'sine sim':>10s}{'gauss(4sd)':>12s}")
    for b in (2, 4, 6, 8, 10):
        print(f"{b:>5d}{6.02*b+1.76:>11.1f}{simulate_sqnr(b):>10.1f}{simulate_sqnr(b, 'gauss'):>12.1f}")


def report_layer_table(layers: List[Layer], title: str):
    _hdr(title)
    print(f"{'#':>3s} {'layer':10s}{'output':>16s}{'params':>11s}{'MACs (M)':>10s}{'in+out KB (INT8)':>18s}")
    for i, l in enumerate(layers, 1):
        print(f"{i:>3d} {l.name:10s}{'x'.join(map(str, l.out_shape)):>16s}{l.params:>11d}"
              f"{l.macs/1e6:>10.2f}{(l.in_elems+l.out_elems)/1e3:>18.1f}")


def worked_example():
    _hdr("Appendix F: worked example, layer pw1 of network A")
    soc = DEVICES["Mobile SoC"]
    l = next(x for x in build_mobilenet_v1() if x.name == "pw1")
    for label, bits in (("FP32", 32), ("INT8", 8)):
        b = layer_bytes(l, bits, bits)
        tc, tm = l.macs / soc.pi * 1e3, b / soc.beta * 1e3
        print(f"{label}: moves {b/1e6:.2f} MB, intensity {l.macs/b:.1f} MAC/B, "
              f"compute {tc:.2f} ms, transfer {tm:.2f} ms -> {max(tc, tm):.2f} ms "
              f"({'compute' if tc >= tm else 'memory'}-bound)")


# --------------------------------------------------------------------------
# Verification against the numbers printed in the paper
# --------------------------------------------------------------------------


def verify() -> bool:
    soc, mcu, acc = DEVICES["Mobile SoC"], DEVICES["MCU-class"], DEVICES["Edge accel."]
    A, B, C = build_mobilenet_v1(), build_kws_dscnn(), build_tiny_transformer()
    checks = []  # (name, got, expected, relative tolerance)

    def add(name, got, exp, tol=0.02):
        checks.append((name, got, exp, tol))

    # Table 7
    add("A params (M)", total_params(A) / 1e6, 4.22)
    add("A MACs (M)", total_macs(A) / 1e6, 569, 0.005)
    add("A peak act INT8 (KB)", peak_activation_bytes(A, 8) / 1e3, 1204, 0.002)
    add("B params (K)", total_params(B) / 1e3, 21.6)
    add("B MACs (M)", total_macs(B) / 1e6, 2.54)
    add("B peak act INT8 (KB)", peak_activation_bytes(B, 8) / 1e3, 16.0)
    add("C params (M)", total_params(C) / 1e6, 0.79)
    add("C MACs (M)", total_macs(C) / 1e6, 54.5)
    add("C peak act INT8 (KB)", peak_activation_bytes(C, 8) / 1e3, 41, 0.01)
    # Table 8 / 9
    for a, m, p in ((0.75, 325, 2.58), (0.5, 149, 1.33), (0.25, 41, 0.47)):
        L = build_mobilenet_v1(a)
        add(f"A alpha={a} MACs (M)", total_macs(L) / 1e6, m, 0.01)
        add(f"A alpha={a} params (M)", total_params(L) / 1e6, p, 0.01)
    add("A 0.25@128 peak act (KB)", peak_activation_bytes(build_mobilenet_v1(0.25, 128), 8) / 1e3, 98, 0.01)
    add("A 1.0@128 peak act (KB)", peak_activation_bytes(build_mobilenet_v1(1.0, 128), 8) / 1e3, 393, 0.01)
    # Table 12 (ms)
    add("A FP32 SoC", latency(A, soc, 32, 32)[0] * 1e3, 59.9)
    add("A INT8 SoC", latency(A, soc, 8, 8)[0] * 1e3, 57.0)
    add("A INT8 4x SoC", latency(A, soc, 8, 8, 4)[0] * 1e3, 15.0)
    add("A FP32 MCU", latency(A, mcu, 32, 32)[0] * 1e3, 11376, 0.005)
    add("A FP32 Edge", latency(A, acc, 32, 32)[0] * 1e3, 8.3)
    add("A INT8 Edge", latency(A, acc, 8, 8)[0] * 1e3, 6.1)
    add("B FP32 MCU", latency(B, mcu, 32, 32)[0] * 1e3, 50.9)
    add("B FP32 SoC", latency(B, soc, 32, 32)[0] * 1e3, 0.29, 0.05)
    add("C INT8 MCU", latency(C, mcu, 8, 8, aggregated=True)[0] * 1e3, 1091, 0.005)
    add("C INT8 SoC", latency(C, soc, 8, 8, aggregated=True)[0] * 1e3, 5.5)
    # Table 13
    rows = sweep_rows()
    add("MCU feasible combos", sum(r["exceeds"] == "fits" for r in rows), 3, 0)
    # Table 15
    L = build_mobilenet_v1(0.5, 160)
    add("pipeline step 3 SoC ms", latency(L, soc, 8, 8)[0] * 1e3, 7.7)
    add("pipeline step 4 weights (KB)", weight_bytes(L, 4) / 1e3, 663, 0.01)
    # Table 16
    a32, t32 = energy_joules(A, 32, E_FP32_MAC)
    a8, t8 = energy_joules(A, 8, E_INT8_MAC)
    add("energy FP32 arithmetic (mJ)", a32 * 1e3, 2.62)
    add("energy FP32 traffic (mJ)", t32 * 1e3, 2.70)
    add("energy INT8 arithmetic (mJ)", a8 * 1e3, 0.13)
    add("energy INT8 traffic (mJ)", t8 * 1e3, 0.68)
    # Table 17 spot checks
    for d, nl, n, m, p in ((64, 2, 32, 3.4, 0.10), (128, 4, 128, 117.4, 0.79), (192, 4, 128, 251.7, 1.78)):
        L = build_tiny_transformer(d, nl, n)
        add(f"C d={d},n={n} MACs (M)", total_macs(L) / 1e6, m, 0.01)
        add(f"C d={d},n={n} params (M)", total_params(L) / 1e6, p, 0.02)
    # Table 18
    t = batch_fc_table(soc)
    add("batch=1 us/sample", t[0][3], 205, 0.01)
    add("batch=128 intensity", t[-1][1], 102.1, 0.01)
    # Misc
    add("break-even 8-bit idx (%)", 100 * sparse_breakeven(8, 8), 50, 0)
    add("Amdahl f=0.8 limit", amdahl(0.8, 1e12), 5.0, 0.001)

    ok_all = True
    print("\nVerification against the paper's reported values")
    print("-" * 78)
    for name, got, exp, tol in checks:
        ok = abs(got - exp) <= tol * abs(exp) + 1e-12
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} got {got:12.4f}   paper {exp:10.4f}")
    print("-" * 78)
    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED (see above)")
    return ok_all


# --------------------------------------------------------------------------
# CSV export and main
# --------------------------------------------------------------------------


def write_csv(outdir: str, rows):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "network_a_sweep.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    path2 = os.path.join(outdir, "network_a_layers.csv")
    with open(path2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "kind", "output", "params", "macs", "in_elems", "out_elems"])
        for l in build_mobilenet_v1():
            w.writerow([l.name, l.kind, "x".join(map(str, l.out_shape)), l.params, l.macs,
                        l.in_elems, l.out_elems])
    print(f"\nCSV files written to {outdir}/")


def load_devices(path: str):
    with open(path) as fh:
        data = json.load(fh)
    DEVICES.clear()
    for name, d in data.items():
        DEVICES[name] = Device(name, d["pi_gmacs"] * 1e9, d["beta_gbs"] * 1e9,
                               d["ram_kb"] * 1e3, d["flash_kb"] * 1e3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only run checks against the paper's values")
    ap.add_argument("--csv", metavar="DIR", help="write sweep and layer tables as CSV files")
    ap.add_argument("--device-file", metavar="JSON", help="replace the assumed devices (see docstring)")
    ap.add_argument("--layers", action="store_true", help="also print the layer tables (Appendix C)")
    args = ap.parse_args()

    if args.device_file:
        load_devices(args.device_file)
        print("Custom devices loaded; paper verification values will not match.")
    if args.verify:
        raise SystemExit(0 if verify() else 1)

    print("NOTE: all values are analytical and rely on ASSUMED device parameters; "
          "no hardware measurements are involved.")
    report_reference_networks()
    report_validation()
    report_network_a_profile()
    rows = sweep_rows()
    report_sweep(rows)
    report_latency()
    report_sensitivity()
    report_pipeline()
    report_transformer()
    report_batch()
    scenario_report()
    report_misc()
    worked_example()
    if args.layers:
        report_layer_table(build_mobilenet_v1(), "Table 26: network A layers")
        report_layer_table(build_kws_dscnn(), "Table 27: network B layers")
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
