"""P4-U10 audit helpers: window arithmetic and the evidence renderer.

The audit's conclusions rest on the rendered frames, so the renderer must not silently drop
a rider or crash on the case that matters most -- a rider whose head crop was gated, which
is precisely the one an inspector needs to see marked. Both are asserted here; the model
calls around them are not, since they are ``score.py``'s and already covered.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.audit import draw_frame, windows_of  # noqa: E402

from trafficpulse.contracts import TrackState  # noqa: E402
from trafficpulse.contracts.enums import ObjectClass, TrackStatus  # noqa: E402
from trafficpulse.contracts.primitives import BoundingBox  # noqa: E402

TS = datetime(2026, 1, 1, tzinfo=UTC)


def state(track_id: str, object_class: ObjectClass, box: tuple[float, ...]) -> TrackState:
    return TrackState(
        track_id=track_id,
        camera_id="audit",
        timestamp=TS,
        frame_index=0,
        object_class=object_class,
        bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        status=TrackStatus.ACTIVE,
    )


def test_windows_are_consecutive_ranges_of_the_requested_length() -> None:
    assert windows_of("0,240,480", 30) == [(0, 30), (240, 270), (480, 510)]


def test_a_single_window_is_accepted() -> None:
    assert windows_of("100", 5) == [(100, 105)]


def test_trailing_separators_do_not_create_an_empty_window() -> None:
    assert windows_of("0,10,", 5) == [(0, 5), (10, 15)]


def test_the_renderer_returns_a_frame_of_the_original_size() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    states = [
        state("m1", ObjectClass.MOTORCYCLE, (100.0, 120.0, 160.0, 200.0)),
        state("r1", ObjectClass.PERSON, (105.0, 60.0, 155.0, 170.0)),
    ]
    annotations = {
        "r1": {
            "zeroshot_label": "helmet",
            "zeroshot_score": 0.8,
            "resnet_label": "helmet",
            "resnet_score": 0.9,
            "head_height_px": 33.0,
        }
    }
    canvas = draw_frame(image, states, {"m1": ["r1"]}, annotations)
    assert canvas.size == (320, 240)
    assert np.asarray(canvas).any(), "boxes were actually drawn onto the frame"


def test_a_gated_rider_is_still_drawn_so_it_can_be_seen() -> None:
    """A rider with no prediction is the case an inspector most needs marked."""

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    states = [
        state("m1", ObjectClass.MOTORCYCLE, (100.0, 120.0, 160.0, 200.0)),
        state("r1", ObjectClass.PERSON, (105.0, 110.0, 155.0, 130.0)),
    ]
    canvas = draw_frame(image, states, {"m1": ["r1"]}, {})
    pixels = np.asarray(canvas)
    assert pixels.any(), "the gated rider is rendered, not silently omitted"
    assert (pixels[..., 0] > 200).any(), "gated riders are drawn in the alert colour"


def test_an_unridden_motorcycle_is_drawn_without_a_rider() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    states = [state("m1", ObjectClass.MOTORCYCLE, (10.0, 10.0, 60.0, 80.0))]
    canvas = draw_frame(image, states, {}, {})
    assert np.asarray(canvas).any()
