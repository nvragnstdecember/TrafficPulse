"""Dominant traffic-flow estimation from observed tracks (H12).

The *observable* half of scene calibration. A roadway's legal travel direction is
a fact you can measure -- it is where the traffic actually goes -- so it can be
derived from a clip rather than typed in by hand. This module owns that
derivation and nothing else.

The algorithm
-------------
The flow is the **vector sum of the net displacement of every substantial vehicle
track**, where "substantial" means alive at least
:data:`MIN_TRACK_LIFETIME_SECONDS` and displaced at least
:data:`MIN_NET_DISPLACEMENT_PX`. These thresholds are not new: they are the ones
validated on real CCTV footage by the upload auto-calibration work that preceded
H12, promoted into the runtime rather than reimplemented, so there is exactly one
implementation.

Calibration only -- no I/O, no model
------------------------------------
Deciding *which* tracks to measure requires decoding a clip and running a detector,
but that is I/O and model orchestration, not calibration, and the application has a
job engine for it (``ProcessingService`` runs the pass and hands the resulting
``TrackState``s here). So this module takes ``TrackState``s and returns a vector:
**pure, no decoding, no detector, no tracker, no ML import**, and unit-testable
without a video.

What cannot be derived, and is therefore not attempted
-------------------------------------------------------
Only observable facts are estimated. A **no-stopping zone is not observable** --
where stopping is illegal is a legal and operational fact about the site, not
something traffic reveals by flowing over it. Inferring one from footage would be
fabrication, so illegal-stopping geometry is authored by the analyst
(:mod:`trafficpulse.scenes.builder`), never guessed here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ..contracts import TrackState
from ..contracts.enums import ObjectClass

#: A track joins the flow estimate only if it is a *substantial mover*: alive at
#: least this long and displaced at least this far. Excludes detector jitter on
#: parked or stationary objects, which would otherwise add near-random short
#: vectors and drag the estimate toward noise.
MIN_TRACK_LIFETIME_SECONDS = 1.0
MIN_NET_DISPLACEMENT_PX = 40.0

#: Only *vehicle* tracks define a roadway's flow. The label map includes
#: ``person`` because helmet reasoning needs riders, but a pedestrian crossing the
#: road -- or walking a footpath in any direction -- does not define the legal
#: direction of traffic, and including them pulls the estimate off the traffic
#: axis. Motorcycles count: they travel with the stream.
FLOW_CLASSES: frozenset[ObjectClass] = frozenset({ObjectClass.CAR, ObjectClass.MOTORCYCLE})

#: Unit-vector components are rounded to this many decimals so the derived scene
#: content -- and therefore its hash, and therefore the content-derived event ids
#: reasoned under it -- are stable against float noise across platforms. Four
#: decimals holds direction to ~0.06 degrees, far finer than any threshold reads.
_VECTOR_PRECISION = 4


@dataclass(frozen=True)
class FlowEstimate:
    """The observed dominant traffic direction of one clip.

    ``dx``/``dy`` are a **unit** vector in image space (x right, y down -- the
    ``FrameSpec`` convention), directly usable as a ``DirectionVector``.
    ``mover_count`` and ``track_count`` are reported so a caller can say how much
    evidence the estimate rests on rather than presenting it as certain.
    """

    dx: float
    dy: float
    heading_degrees: float
    mover_count: int
    track_count: int


def _center(state: TrackState) -> tuple[float, float]:
    box = state.bbox
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def estimate_dominant_flow(
    states: Iterable[TrackState],
    *,
    classes: frozenset[ObjectClass] = FLOW_CLASSES,
    min_lifetime_seconds: float = MIN_TRACK_LIFETIME_SECONDS,
    min_displacement_px: float = MIN_NET_DISPLACEMENT_PX,
) -> FlowEstimate | None:
    """Estimate the dominant flow from tracked vehicles, or ``None`` if unknowable.

    ``states`` may arrive in any order and from any number of tracks; they are
    grouped by ``track_id`` and ordered by timestamp here, so a caller can hand
    over a whole run's states verbatim.

    Returns ``None`` -- never a fabricated default -- when no substantial vehicle
    motion was observed, or when the movers cancel out (traffic flowing equally in
    both directions, which is a two-way road that a single legal direction cannot
    describe). Both are honest "this clip cannot tell you" answers, and the caller
    must fall back to asking the analyst.
    """

    by_track: dict[str, list[TrackState]] = defaultdict(list)
    for state in states:
        if state.object_class in classes:
            by_track[state.track_id].append(state)

    sum_dx = sum_dy = 0.0
    movers = 0
    for track in by_track.values():
        if len(track) < 2:
            continue
        ordered = sorted(track, key=lambda s: (s.timestamp, s.frame_index or 0))
        first, last = ordered[0], ordered[-1]
        lifetime = (last.timestamp - first.timestamp).total_seconds()
        (x0, y0), (x1, y1) = _center(first), _center(last)
        dx, dy = x1 - x0, y1 - y0
        if lifetime >= min_lifetime_seconds and math.hypot(dx, dy) >= min_displacement_px:
            movers += 1
            sum_dx += dx
            sum_dy += dy

    magnitude = math.hypot(sum_dx, sum_dy)
    if movers == 0 or magnitude <= 0.0:
        return None

    dx = round(sum_dx / magnitude, _VECTOR_PRECISION)
    dy = round(sum_dy / magnitude, _VECTOR_PRECISION)
    if dx == 0.0 and dy == 0.0:  # pragma: no cover - defensive: rounding to zero
        return None
    return FlowEstimate(
        dx=dx,
        dy=dy,
        heading_degrees=math.degrees(math.atan2(dy, dx)) % 360.0,
        mover_count=movers,
        track_count=len(by_track),
    )
