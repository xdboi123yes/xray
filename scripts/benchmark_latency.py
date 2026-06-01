"""latency, peak memory, and carbon emissions benchmarks.

Tracks metrics using codecarbon, measuring GPU/CPU latency and peak allocation sizes.
"""

from __future__ import annotations

import os
import sys
import time

import torch

# Support standalone execution (python scripts/benchmark_latency.py): add the project
# root to sys.path so `core` is importable regardless of the current working directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models.factory import ModelFactory

# CodeCarbon is optional: it is frequently unavailable on Colab and may fail to read
# power sensors. Latency / throughput are always reported; carbon only when possible.
try:
    from codecarbon import EmissionsTracker

    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


def benchmark_model(backbone_key: str, device: torch.device, num_iters: int = 100) -> None:
    """Benchmark raw latency and memory footprint of a specific model backbone."""
    print(f"\n--- Benchmarking {backbone_key} on {device} ---")
    model = ModelFactory.create(backbone_key)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Start latency tracking
    start_time = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_iters) * 1000.0
    fps = 1000.0 / avg_latency_ms

    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Throughput (FPS): {fps:.1f} frames/sec")

    # Memory usage
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Peak GPU Memory Allocated: {peak_mem:.2f} MB")
    elif device.type == "mps":
        print("Peak GPU Memory: MPS dynamic memory tracking active.")
    else:
        print("Peak CPU Memory: Standard CPU allocation.")


def main() -> None:
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tracker = None
    if _CODECARBON_AVAILABLE:
        print("Initializing CodeCarbon emissions tracker...")
        try:
            tracker = EmissionsTracker(measure_power_secs=15, save_to_file=False, log_level="warning")
            tracker.start()
        except Exception as exc:
            print(f"CodeCarbon tracker unavailable ({exc}); reporting latency without carbon.")
            tracker = None
    else:
        print("CodeCarbon not installed; reporting latency without carbon.")

    try:
        benchmark_model("mobilenet_v2", device)
        benchmark_model("efficientnet_b4", device)
    finally:
        if tracker is not None:
            try:
                emissions = tracker.stop()
                print(f"\nBenchmark completed. Tracked Carbon Footprint Emissions: {emissions:.8f} kg CO2eq")
            except Exception as exc:
                print(f"\nBenchmark completed. Carbon tracking failed: {exc}")
        else:
            print("\nBenchmark completed. (Carbon footprint not measured.)")


if __name__ == "__main__":
    main()
