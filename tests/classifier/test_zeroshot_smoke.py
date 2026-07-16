"""Opt-in REAL zero-shot helmet inference smoke test (P4-U3).

This is the only test that runs the real ``torch`` + ``transformers`` CLIP-family
path. It is **skipped by default** and never touches the network: it runs only when
BOTH

* the optional ``rtdetr`` dependencies are importable (``pip install
  'trafficpulse[rtdetr]'`` -- this backend adds no new dependency), and
* ``TRAFFICPULSE_HELMET_SMOKE_MODEL`` points at a **locally available** CLIP-family
  checkpoint (a HuggingFace cache id or a local directory), whose weight provenance
  has been reviewed per ADR-001 (U4 registry).

It loads with ``local_files_only=True`` (no download), runs one batched forward pass
over synthetic in-memory RGB crops (no dataset, no private footage, nothing
committed), and asserts only **structural** properties: that the backend produces
framework-neutral ``RawHelmetPrediction`` values honouring the P4-U2 batch contract.

It deliberately asserts **nothing about accuracy**. Synthetic pixels carry no helmet,
and zero-shot performance on real CCTV head crops is unvalidated; asserting a label
here would be fabricating a result. Accuracy is the pre-registered CNN-vs-ViT
experiment's job, on licence-cleared data. Optional device override:
``TRAFFICPULSE_HELMET_SMOKE_DEVICE`` (default ``cpu``).
"""

import importlib.util
import os

import numpy as np
import pytest

_MODEL = os.environ.get("TRAFFICPULSE_HELMET_SMOKE_MODEL")
_DEVICE = os.environ.get("TRAFFICPULSE_HELMET_SMOKE_DEVICE", "cpu")
_HAVE_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

pytestmark = pytest.mark.skipif(
    not (_MODEL and _HAVE_DEPS),
    reason=(
        "opt-in real zero-shot helmet smoke test: install trafficpulse[rtdetr] and set "
        "TRAFFICPULSE_HELMET_SMOKE_MODEL to a locally-available CLIP-family checkpoint"
    ),
)


def test_real_zero_shot_inference_on_synthetic_crops() -> None:
    from datetime import UTC, datetime

    from trafficpulse.classifier import (
        Crop,
        RawHelmetPrediction,
        ZeroShotHelmetClassifier,
        ZeroShotHelmetConfig,
    )

    config = ZeroShotHelmetConfig(
        checkpoint=str(_MODEL), device=_DEVICE, local_files_only=True
    )
    classifier = ZeroShotHelmetClassifier(config)

    crops = [
        Crop(
            camera_id="smoke",
            frame_index=index,
            timestamp=datetime.now(UTC),
            track_id=f"r-{index}",
            image=np.zeros((64, 64, 3), dtype=np.uint8),  # synthetic, in-memory only
        )
        for index in range(3)
    ]

    predictions = classifier.classify(crops)

    # Structural only -- no accuracy claim (see module docstring).
    assert len(predictions) == len(crops)
    for prediction in predictions:
        assert isinstance(prediction, RawHelmetPrediction)
        assert type(prediction.label) is str
        assert prediction.label in config.prompts
        assert type(prediction.score) is float and 0.0 <= prediction.score <= 1.0


def test_real_backend_honours_empty_input_without_loading() -> None:
    from trafficpulse.classifier import ZeroShotHelmetClassifier, ZeroShotHelmetConfig

    classifier = ZeroShotHelmetClassifier(
        ZeroShotHelmetConfig(checkpoint=str(_MODEL), device=_DEVICE, local_files_only=True)
    )

    assert tuple(classifier.classify(())) == ()
