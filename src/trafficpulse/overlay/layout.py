"""Deterministic label placement: never let two captions stack unreadably.

Pure geometry. The renderer measures each caption (it owns the font) and hands this
module a :class:`LabelRequest` per caption -- the box it belongs to, the preferred
corner, and the measured ``(width, height)``. :func:`place_labels` returns a
top-left pixel for each, chosen so labels do not overlap **each other** and stay
inside the frame, preferring positions near their box.

Algorithm (greedy, order-stable)
--------------------------------
Captions are placed in request order. For each, a prioritised list of candidate
positions is generated (near the preferred corner first, then a ladder that steps
the label away above/below the box, then beside it); the first candidate that is
in-frame and clear of every already-placed label wins. If every candidate collides,
the one with the least total overlap is used (clamped in-frame) -- a graceful
degradation rather than a hidden label. Placement depends only on the requests and
frame size, so it is fully deterministic and has no Pillow/image dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .metadata import Corner

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class LabelRequest:
    """One caption to place: its owning box, preferred corner, and measured size."""

    box: Rect
    width: float
    height: float
    prefer: Corner = Corner.TOP_LEFT
    pad: float = 6.0


def _overlap(a: Rect, b: Rect) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _clamp(rect: Rect, frame_w: float, frame_h: float) -> Rect:
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    x1 = min(max(0.0, rect[0]), max(0.0, frame_w - w))
    y1 = min(max(0.0, rect[1]), max(0.0, frame_h - h))
    return (x1, y1, x1 + w, y1 + h)


def _candidates(req: LabelRequest) -> list[tuple[float, float]]:
    bx1, by1, bx2, by2 = req.box
    w, h, pad = req.width, req.height, req.pad
    left, right = bx1, bx2 - w
    above, below = by1 - h - pad, by2 + pad
    inside_top, inside_bottom = by1 + pad, by2 - h - pad

    corner_first: dict[Corner, list[tuple[float, float]]] = {
        Corner.TOP_LEFT: [(left, above), (left, inside_top)],
        Corner.TOP_RIGHT: [(right, above), (right, inside_top)],
        Corner.BOTTOM_LEFT: [(left, below), (left, inside_bottom)],
        Corner.BOTTOM_RIGHT: [(right, below), (right, inside_bottom)],
    }
    out: list[tuple[float, float]] = list(corner_first[req.prefer])
    # The other three corners before any long displacement: a label one corner over
    # is still obviously attached to its box, whereas one pushed a ladder-rung away
    # needs a leader line to be readable at all.
    for corner, positions in corner_first.items():
        if corner is not req.prefer:
            out.extend(positions)
    # flush beside the box, still touching it
    out.append((bx2 + pad, by1))  # to the right
    out.append((bx1 - w - pad, by1))  # to the left
    # a ladder stepping away above/below -- the last-resort escape hatches
    step = h + 4.0
    for k in range(1, 4):
        out.append((left, above - k * step))
        out.append((left, below + k * step))
    out.append((left, below))
    return out


#: How much a square pixel of overlap with a merely *undesirable* region counts
#: against a candidate, relative to a hard collision. Low enough that covering
#: scenery always loses to hiding another label, high enough to break ties.
_SOFT_WEIGHT = 0.2


def place_labels(
    requests: list[LabelRequest],
    frame_w: float,
    frame_h: float,
    blocked: Sequence[Rect] = (),
    avoid: Sequence[Rect] = (),
) -> list[Rect]:
    """Return a non-overlapping, in-frame rect per request (index-aligned).

    ``blocked`` are regions already claimed by something the solver does not place
    -- pinned banners, most importantly, which are drawn *over* captions and would
    otherwise hide the very track a confirmed violation is about. They are treated
    exactly like already-placed labels: candidates that collide with them lose.

    ``avoid`` are *soft* regions -- typically the drawn boxes themselves. A caption
    sitting on top of another vehicle's box is legible but misleading, since it
    reads as belonging to whatever it covers. These only break ties: a candidate
    that avoids them wins, but obscuring one is always preferred to hiding another
    label outright.
    """

    placed: list[Rect] = list(blocked)
    reserved = len(placed)
    for req in requests:
        best: Rect | None = None
        best_cost = float("inf")
        for cx, cy in _candidates(req):
            rect = _clamp((cx, cy, cx + req.width, cy + req.height), frame_w, frame_h)
            hard = sum(_overlap(rect, p) for p in placed)
            soft = sum(_overlap(rect, a) for a in avoid) if avoid else 0.0
            cost = hard + soft * _SOFT_WEIGHT
            if cost == 0.0:
                best = rect
                break
            if cost < best_cost:
                best, best_cost = rect, cost
        assert best is not None  # _candidates is always non-empty
        placed.append(best)
    return placed[reserved:]
