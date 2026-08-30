"""Latency, throughput and VRAM measurement (P4-U5).

Architecture-review §12 fixes the protocol precisely, and this module follows it
literally: "batch 1 and 32, fp16, ``inference_mode``, 100 warmup, median of 1,000
timed iters; VRAM via ``max_memory_allocated``; params + checkpoint size".

Three details decide whether the numbers mean anything:

* **CUDA is asynchronous.** A kernel launch returns immediately, so timing without
  ``torch.cuda.synchronize()`` measures the launch, not the work. Every timed
  iteration synchronises at its boundary.
* **The median, not the mean.** On a laptop GPU, thermal throttling and background
  work produce occasional large outliers. §12 specifies the median for exactly this
  reason; the mean would report the interference rather than the model.
* **Warm-up is not optional.** The first iterations pay for cuDNN autotuning and
  lazy kernel compilation. The 100 warm-up passes are discarded.

Cost numbers are reported for both families under identical conditions, and are how
a **tie** on macro-F1 gets interpreted -- per the pre-registered rule, a tie is read
through the accuracy/latency/VRAM tradeoff, so these measurements are part of the
result, not an appendix to it.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from helmet_rtdetr.models import _Model

from .models import IMAGE_SIZE, ModelSpec, create_model, parameter_count, require_backend

#: The two batch sizes §12 specifies.
BATCH_SIZES: tuple[int, ...] = (1, 32)

#: Discarded iterations before timing starts.
WARMUP_ITERS = 100

#: Timed iterations; the reported latency is their median.
TIMED_ITERS = 1000


class BatchBenchmark(_Model):
    """Timing and memory for one model at one batch size."""

    batch_size: int
    median_latency_ms: float
    p90_latency_ms: float
    throughput_img_per_s: float
    peak_vram_mib: float | None


class ModelBenchmark(_Model):
    """The full §12 cost profile for one model."""

    model: str
    timm_id: str
    family: str
    parameters: int
    checkpoint_bytes: int | None
    device: str
    precision: str
    batches: tuple[BatchBenchmark, ...]


def benchmark_model(
    spec: ModelSpec,
    *,
    checkpoint: Path | None = None,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
    image_size: int = IMAGE_SIZE,
    amp: bool = True,
) -> ModelBenchmark:
    """Measure one model's latency, throughput and peak VRAM.

    Weights are irrelevant to timing, so a checkpoint is optional; when one is
    given it is loaded anyway so the reported ``checkpoint_bytes`` and the timed
    graph come from the same artifact.
    """

    require_backend()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"

    model = create_model(spec, pretrained=False).to(device).eval()
    checkpoint_bytes: int | None = None
    if checkpoint is not None and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        checkpoint_bytes = checkpoint.stat().st_size

    results: list[BatchBenchmark] = []
    for batch_size in batch_sizes:
        images = torch.randn(batch_size, 3, image_size, image_size, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            for _ in range(warmup):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

            samples: list[float] = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if start is not None and end is not None:
                    start.record()
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        model(images)
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end))
                else:  # pragma: no cover - CPU fallback, not the reported path
                    import time

                    began = time.perf_counter()
                    model(images)
                    samples.append((time.perf_counter() - began) * 1000.0)

        samples.sort()
        median = statistics.median(samples)
        peak = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else None
        )
        results.append(
            BatchBenchmark(
                batch_size=batch_size,
                median_latency_ms=median,
                p90_latency_ms=samples[int(0.9 * (len(samples) - 1))],
                throughput_img_per_s=(batch_size * 1000.0 / median) if median > 0 else 0.0,
                peak_vram_mib=peak,
            )
        )

    return ModelBenchmark(
        model=spec.name,
        timm_id=spec.timm_id,
        family=spec.family,
        parameters=parameter_count(model),
        checkpoint_bytes=checkpoint_bytes,
        device=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        precision=("fp16-autocast" if use_amp else "fp32"),
        batches=tuple(results),
    )


def benchmark_all(
    specs: dict[str, ModelSpec], checkpoints: dict[str, Path] | None = None, **kwargs: Any
) -> dict[str, ModelBenchmark]:
    """Benchmark every model under identical conditions, in a stable order."""

    checkpoints = checkpoints or {}
    return {
        name: benchmark_model(spec, checkpoint=checkpoints.get(name), **kwargs)
        for name, spec in sorted(specs.items())
    }
