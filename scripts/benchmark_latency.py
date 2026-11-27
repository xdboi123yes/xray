"""Comprehensive latency, memory, FLOPs, and carbon benchmark for TCR-P.

Measures transparent, hardware-synchronized inference cost across all main
configurations (Reviewer 1 & Reviewer 2 Comment 14):
- Explicit warmup iterations and ``torch.cuda.synchronize()`` timing boundaries.
- Preprocessing (CLAHE + resize + ImageNet normalization) and host-to-device transfer.
- Single-pass latency (batch size 1), vectorized MC-Dropout (T=20), vectorized TTA
  (K=10), and vectorized MC+TTA (T=20, K=10 = 200 passes) for Tier 1 (MobileNetV2)
  and both Tier 2 backbones (EfficientNet-B4 and Ark+ Swin).
- Parameter counts, analytical/hooked MACs/GFLOPs, and peak GPU VRAM.
- Apples-to-apples expected tiered latency:
    (a) Single-pass tiered (A3/A4) vs. single-pass Tier 2 (A2/A11)
    (b) Full-uncertainty tiered (A6/A13/A15) vs. always-Tier-2 uncertainty (A10/A12)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tcrp.data.augmentation import ClassicalAugmentation
from tcrp.models import ModelFactory

try:
    from codecarbon import EmissionsTracker

    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def estimate_flops_and_params(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 224, 224)) -> dict[str, float]:
    """Compute exact parameter count and forward MACs/GFLOPs via lightweight module hooks."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs = 0.0
    handles: list[Any] = []

    def _conv_hook(mod: nn.Conv2d, _inp: Any, out: torch.Tensor) -> None:
        nonlocal macs
        b, c_out, h_out, w_out = out.shape
        k_h, k_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        macs += float(b * c_out * h_out * w_out * (mod.in_channels // mod.groups) * k_h * k_w)

    def _linear_hook(mod: nn.Linear, inp: Any, _out: torch.Tensor) -> None:
        nonlocal macs
        x = inp[0]
        batch_elems = int(np.prod(x.shape[:-1]))
        macs += float(batch_elems * mod.in_features * mod.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(_linear_hook))

    model_cpu = model.eval()
    device = next(model_cpu.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        _ = model_cpu(dummy)
    for h in handles:
        h.remove()

    # Standard convention: 1 MAC = 2 FLOPs
    return {
        "total_params": float(total_params),
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "gmacs": macs / 1e9,
        "gflops": (2.0 * macs) / 1e9,
    }


def benchmark_preprocessing(num_iters: int = 50) -> dict[str, float]:
    """Measure CLAHE + resize(224x224) + ImageNet normalization latency on CPU."""
    pipeline = ClassicalAugmentation(image_size=224, is_training=False).pipeline
    dummy_img = np.random.randint(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
    for _ in range(5):
        _ = pipeline(image=dummy_img)["image"]
    times_ms: list[float] = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _ = pipeline(image=dummy_img)["image"]
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def _time_callable(fn: Any, device: torch.device, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times_ms: list[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "peak_vram_mb": peak_mem_mb,
    }


def benchmark_backbone(
    backbone_key: str,
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
    mc_passes: int = 20,
    tta_passes: int = 10,
    micro_batch: int = 64,
    include_uncertainty: bool = True,
) -> dict[str, Any]:
    """Benchmark single-pass, H2D transfer, MC, TTA, and vectorized MC+TTA latency."""
    print(f"\n--- Benchmarking {backbone_key} on {device} ---")
    model = ModelFactory.create(backbone_key, pretrained=False).to(device)
    model.eval()
    complexity = estimate_flops_and_params(model)

    cpu_tensor = torch.randn(1, 3, 224, 224)
    gpu_tensor = cpu_tensor.to(device)

    h2d = _time_callable(lambda: cpu_tensor.to(device, non_blocking=False), device, warmup=10, iters=50)

    def _single() -> None:
        with torch.no_grad():
            _ = model(gpu_tensor)

    single = _time_callable(_single, device, warmup=warmup, iters=iters)

    res: dict[str, Any] = {
        "backbone": backbone_key,
        "complexity": complexity,
        "h2d_transfer": h2d,
        "single_pass_bs1": single,
    }

    if include_uncertainty:
        unc_iters = max(5, iters // 10)
        unc_warmup = max(2, warmup // 5)

        def _batched_passes(n_passes: int, enable_mc: bool) -> None:
            if enable_mc:
                _enable_dropout_only(model)
            else:
                model.eval()
            done = 0
            with torch.no_grad():
                while done < n_passes:
                    k = min(micro_batch, n_passes - done)
                    chunk = gpu_tensor.expand(k, -1, -1, -1).contiguous()
                    _ = model(chunk)
                    done += k
            model.eval()

        res["mc_only_vectorized"] = _time_callable(
            lambda: _batched_passes(mc_passes, True), device, warmup=unc_warmup, iters=unc_iters
        )
        res["tta_only_vectorized"] = _time_callable(
            lambda: _batched_passes(tta_passes, False), device, warmup=unc_warmup, iters=unc_iters
        )
        res["mc_tta_vectorized"] = _time_callable(
            lambda: _batched_passes(mc_passes * tta_passes, True),
            device,
            warmup=unc_warmup,
            iters=unc_iters,
        )

    print(
        f"  Params: {complexity['total_params_millions']:.2f}M | "
        f"GFLOPs: {complexity['gflops']:.2f} | "
        f"Single-pass (bs=1): {single['mean_ms']:.2f} ± {single['std_ms']:.2f} ms"
    )
    if include_uncertainty:
        print(
            f"  MC({mc_passes}): {res['mc_only_vectorized']['mean_ms']:.2f} ms | "
            f"TTA({tta_passes}): {res['tta_only_vectorized']['mean_ms']:.2f} ms | "
            f"MC+TTA({mc_passes}x{tta_passes}): {res['mc_tta_vectorized']['mean_ms']:.2f} ms"
        )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive TCR-P latency & complexity benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--tta-passes", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=64)
    parser.add_argument("--escalation-rate", type=float, default=0.26)
    parser.add_argument("--output", type=str, default="outputs/results/benchmark_latency.json")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    tracker = None
    if _CODECARBON_AVAILABLE:
        try:
            tracker = EmissionsTracker(measure_power_secs=15, save_to_file=False, log_level="warning")
            tracker.start()
        except Exception:
            tracker = None

    try:
        preproc = benchmark_preprocessing()
        t1 = benchmark_backbone(
            "mobilenet_v2",
            device,
            warmup=args.warmup,
            iters=args.iters,
            include_uncertainty=False,
        )
        t2_eff = benchmark_backbone(
            "efficientnet_b4",
            device,
            warmup=args.warmup,
            iters=args.iters,
            mc_passes=args.mc_passes,
            tta_passes=args.tta_passes,
            micro_batch=args.micro_batch,
            include_uncertainty=True,
        )
        t2_ark = benchmark_backbone(
            "ark_plus",
            device,
            warmup=args.warmup,
            iters=args.iters,
            mc_passes=args.mc_passes,
            tta_passes=args.tta_passes,
            micro_batch=args.micro_batch,
            include_uncertainty=True,
        )
    finally:
        emissions_kg = None
        if tracker is not None:
            try:
                emissions_kg = float(tracker.stop())
            except Exception:
                emissions_kg = None

    esc = args.escalation_rate
    t1_ms = t1["single_pass_bs1"]["mean_ms"]
    eff_single_ms = t2_eff["single_pass_bs1"]["mean_ms"]
    ark_single_ms = t2_ark["single_pass_bs1"]["mean_ms"]
    eff_mctta_ms = t2_eff["mc_tta_vectorized"]["mean_ms"]
    ark_mctta_ms = t2_ark["mc_tta_vectorized"]["mean_ms"]

    # Apples-to-apples expected tiered latency: E[L] = L_T1 + esc * L_T2
    comparisons = {
        "escalation_rate_assumed": esc,
        "single_pass_tiered_eff_ms": t1_ms + esc * eff_single_ms,
        "single_pass_always_eff_ms": eff_single_ms,
        "single_pass_eff_savings_pct": 100.0 * (1.0 - (t1_ms + esc * eff_single_ms) / max(eff_single_ms, 1e-6)),
        "single_pass_tiered_ark_ms": t1_ms + esc * ark_single_ms,
        "single_pass_always_ark_ms": ark_single_ms,
        "single_pass_ark_savings_pct": 100.0 * (1.0 - (t1_ms + esc * ark_single_ms) / max(ark_single_ms, 1e-6)),
        "full_mctta_tiered_eff_ms": t1_ms + esc * eff_mctta_ms,
        "full_mctta_always_eff_ms": eff_mctta_ms,
        "full_mctta_eff_savings_pct": 100.0 * (1.0 - (t1_ms + esc * eff_mctta_ms) / max(eff_mctta_ms, 1e-6)),
        "full_mctta_tiered_ark_ms": t1_ms + esc * ark_mctta_ms,
        "full_mctta_always_ark_ms": ark_mctta_ms,
        "full_mctta_ark_savings_pct": 100.0 * (1.0 - (t1_ms + esc * ark_mctta_ms) / max(ark_mctta_ms, 1e-6)),
    }

    payload = {
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        },
        "protocol": {
            "batch_size": 1,
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
            "mc_passes": args.mc_passes,
            "tta_passes": args.tta_passes,
            "vectorized_micro_batch": args.micro_batch,
        },
        "preprocessing_cpu": preproc,
        "models": {
            "mobilenet_v2": t1,
            "efficientnet_b4": t2_eff,
            "ark_plus": t2_ark,
        },
        "apples_to_apples_tiered_summary": comparisons,
        "carbon_kg_co2eq": emissions_kg,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved structured latency benchmark to {args.output}")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
