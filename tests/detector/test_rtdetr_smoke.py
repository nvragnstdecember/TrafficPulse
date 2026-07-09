"""Opt-in REAL RT-DETR inference smoke test (P1-U7).

This is the only test that runs the real ``torch`` + ``transformers`` RT-DETR path.
It is **skipped by default** and never touches the network: it runs only when BOTH

* the optional ``rtdetr`` dependencies are importable (``pip install
  'trafficpulse[rtdetr]'``), and
* ``TRAFFICPULSE_RTDETR_SMOKE_MODEL`` points at a **locally available** RT-DETR
  checkpoint (a HuggingFace cache id or a local directory), whose weight provenance
  has been reviewed per ADR-001.

It loads with ``local_files_only=True`` (no download), runs one forward pass over a
synthetic in-memory RGB image (no dataset, no private footage, nothing committed),
and asserts only that the backend produces framework-neutral ``RawDetection`` values
in the original-frame pixel convention. Optional device override:
``TRAFFICPULSE_RTDETR_SMOKE_DEVICE`` (default ``cpu``).
"""

import importlib.util
import os
from datetime import UTC, datetime

import numpy as np
import pytest

_MODEL = os.environ.get("TRAFFICPULSE_RTDETR_SMOKE_MODEL")
_DEVICE = os.environ.get("TRAFFICPULSE_RTDETR_SMOKE_DEVICE", "cpu")
_HAVE_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

pytestmark = pytest.mark.skipif(
    not (_MODEL and _HAVE_DEPS),
    reason=(
        "opt-in real RT-DETR smoke test: install trafficpulse[rtdetr] and set "
        "TRAFFICPULSE_RTDETR_SMOKE_MODEL to a locally-available checkpoint"
    ),
)


def test_real_rtdetr_inference_on_synthetic_image() -> None:
    from trafficpulse.detector import Frame, RTDetrConfig, RTDetrDetector

    config = RTDetrConfig(checkpoint=str(_MODEL), device=_DEVICE, local_files_only=True)
    detector = RTDetrDetector(config)

    image = np.zeros((480, 640, 3), dtype=np.uint8)  # synthetic, in-memory only
    frame = Frame(camera_id="smoke", frame_index=0, timestamp=datetime.now(UTC), image=image)

    raws = detector.detect(frame)

    assert isinstance(raws, tuple)
    for raw in raws:
        assert type(raw.label) is str
        assert type(raw.score) is float and 0.0 <= raw.score <= 1.0
        assert len(raw.box) == 4 and all(type(v) is float for v in raw.box)
