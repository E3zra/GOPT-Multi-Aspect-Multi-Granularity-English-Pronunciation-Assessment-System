from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_api import GOPTInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GOPT inference latency on Windows devices."
    )
    parser.add_argument(
        "--model-path",
        default="pretrained_models/gopt_librispeech/best_audio_model.pth",
        help="Path to the GOPT checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        default="librispeech",
        choices=["librispeech", "paiia", "paiib"],
        help="Dataset to pull features from.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to benchmark with.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0],
        help="Sample indices to benchmark. Multiple indices will be batched together.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up runs to discard.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of measured runs for latency statistics.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration latency.",
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_batch(
    engine: GOPTInference, dataset: str, split: str, indices: Sequence[int]
) -> Dict[str, np.ndarray]:
    features, phn_label, _, _ = engine.load_dataset(dataset, split)

    batch_feat = features[indices]
    batch_phn = phn_label[indices, :, 0]

    return {
        "features": batch_feat,
        "phonemes": batch_phn,
    }


def run_inference_batch(engine: GOPTInference, batch: Dict[str, np.ndarray]) -> None:
    engine.predict(batch["features"], batch["phonemes"], normalize=True)


def percentile(values: Sequence[float], pct: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    engine = GOPTInference(
        model_path=str(args.model_path),
        dataset_name=args.dataset,
        model_type="gopt",
        device=device if device != "auto" else None,
    )

    indices = args.indices
    batch = load_batch(engine, args.dataset, args.split, indices)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss

    device_type = engine.device.type

    def update_peak_rss() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)

    if device_type == "cuda":
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        run_inference_batch(engine, batch)
        update_peak_rss()
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies_ms: List[float] = []
    for idx in range(args.runs):
        start = time.perf_counter()
        run_inference_batch(engine, batch)
        if device_type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        duration_ms = (end - start) * 1_000
        latencies_ms.append(duration_ms)
        update_peak_rss()
        if device_type == "cuda":
            torch.cuda.synchronize()
        if args.verbose:
            print(f"Run {idx + 1:03d}: {duration_ms:.2f} ms")

    final_rss = process.memory_info().rss
    gpu_peak_bytes = (
        torch.cuda.max_memory_allocated()
        if device_type == "cuda"
        else None
    )

    summary = {
        "device": device_type,
        "samples": list(indices),
        "batch_size": len(indices),
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency_ms": {
            "avg": statistics.mean(latencies_ms),
            "stdev": statistics.pstdev(latencies_ms),
            "p50": percentile(latencies_ms, 50),
            "p90": percentile(latencies_ms, 90),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
        "memory": {
            "baseline_rss_mb": baseline_rss / (1024 ** 2),
            "peak_rss_mb": peak_rss / (1024 ** 2),
            "final_rss_mb": final_rss / (1024 ** 2),
            "gpu_peak_mb": None if gpu_peak_bytes is None else gpu_peak_bytes / (1024 ** 2),
        },
    }

    print("\n=== GOPT Inference Benchmark ===")
    print(f"Device: {summary['device']}")
    print(f"Batch size: {summary['batch_size']} (indices={summary['samples']})")
    print(f"Warm-up runs: {summary['warmup_runs']}, Measured runs: {summary['measured_runs']}")
    print("\nLatency (ms):")
    for key in ["min", "p50", "avg", "p90", "p95", "p99", "max", "stdev"]:
        print(f"  {key.upper():>4}: {summary['latency_ms'][key]:8.2f}")

    mem = summary["memory"]
    print("\nMemory (MB):")
    print(f"  Baseline RSS: {mem['baseline_rss_mb']:.2f}")
    print(f"  Peak RSS:     {mem['peak_rss_mb']:.2f}")
    print(f"  Final RSS:    {mem['final_rss_mb']:.2f}")
    if mem["gpu_peak_mb"] is not None:
        print(f"  GPU peak:     {mem['gpu_peak_mb']:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()

