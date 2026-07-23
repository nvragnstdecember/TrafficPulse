"""The pure label-collision resolver."""

from __future__ import annotations

from trafficpulse.overlay.layout import LabelRequest, place_labels
from trafficpulse.overlay.metadata import Corner

Rect = tuple[float, float, float, float]


def _overlap(a: Rect, b: Rect) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def test_colliding_labels_are_separated() -> None:
    # two captions wanting the same corner of overlapping boxes
    reqs = [
        LabelRequest(box=(100, 100, 200, 200), width=80, height=30),
        LabelRequest(box=(100, 100, 200, 200), width=80, height=30),
    ]
    a, b = place_labels(reqs, 640.0, 480.0)
    assert _overlap(a, b) == 0.0


def test_labels_stay_inside_the_frame() -> None:
    # a box hugging the top-left: the "above" candidate is off-frame and must clamp
    reqs = [LabelRequest(box=(0, 0, 40, 40), width=120, height=30, prefer=Corner.TOP_LEFT)]
    (x1, y1, x2, y2), = place_labels(reqs, 200.0, 200.0)
    assert x1 >= 0 and y1 >= 0 and x2 <= 200 and y2 <= 200


def test_single_label_lands_near_its_preferred_corner() -> None:
    reqs = [LabelRequest(box=(100, 100, 200, 160), width=60, height=20, prefer=Corner.TOP_LEFT)]
    (x1, y1, _, _), = place_labels(reqs, 640.0, 480.0)
    # top-left preference => left-aligned, above the box
    assert abs(x1 - 100) < 1 and y1 < 100


def test_placement_is_deterministic() -> None:
    reqs = [
        LabelRequest(box=(10, 10, 90, 90), width=50, height=20),
        LabelRequest(box=(20, 20, 100, 100), width=50, height=20),
        LabelRequest(box=(15, 15, 95, 95), width=50, height=20),
    ]
    assert place_labels(reqs, 300.0, 300.0) == place_labels(reqs, 300.0, 300.0)
