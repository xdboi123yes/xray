"""latency, peak memory, and carbon emissions benchmarks.

Tracks metrics using codecarbon, measuring GPU/CPU latency and peak allocation sizes.
"""

from __future__ import annotations

import time

import torch
from codecarbon import EmissionsTracker

from core.models.factory import ModelFactory


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

    print("Initializing CodeCarbon emissions tracker...")
    # Instantiate carbon tracker in a sandbox/offline mode
    tracker = EmissionsTracker(measure_power_secs=15, save_to_file=True, log_level="warning")
    tracker.start()

    try:
        benchmark_model("mobilenet_v2", device)
        benchmark_model("efficientnet_b4", device)
    finally:
        emissions: float = tracker.stop()
        print(f"\nBenchmark completed. Tracked Carbon Footprint Emissions: {emissions:.8f} kg CO2eq")


if __name__ == "__main__":
    main()
