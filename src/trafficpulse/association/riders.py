"""Rider <-> motorcycle association derivation (P4-U4).

The first implementation of the frozen U2 ``Association`` contract, and the first
time the ``Association`` box of the architecture-review §14 data flow
(``Detection -> TrackState -> Association -> Observation``) carries data.

Deterministically decides, **within a single frame**, which tracked ``person`` is
riding which tracked ``motorcycle``, using only bounding-box geometry over frozen
``TrackState`` contracts. It performs no detection, tracking, classification,
observation, or reasoning, and reads no pixels.

Why this exists
---------------
Helmet state is only meaningful for a *rider*. Without an explicit link, a helmet
observation could not say whose head it describes, nor could a later unit
attribute anything to the motorcycle. Emitting the link as the frozen
``Association`` contract -- rather than an ad-hoc tuple -- means the rider identity
and the motorcycle identity travel separately and explicitly:
``HelmetStateObservation.track_id`` names the **rider**, while the ``Association``
names the ``(rider, motorcycle)`` pair. The contracts compose exactly as designed,
and no contract change is needed.

The association method (stated in full; no hidden heuristic)
------------------------------------------------------------
For each frame:

1. Candidate pairs are every ``(person, motorcycle)`` in that frame.
2. Overlap is **intersection over the smaller box's area** (``IoMin``), not IoU.
   IoU is the wrong measure here: a rider and a motorcycle are *different-sized,
   nested-ish* boxes, so a genuine rider commonly scores a low IoU purely because
   the union is large. IoMin asks the question actually being asked -- "how much of
   the smaller box lies inside the larger" -- and is scale-robust.
3. A pair is a candidate iff ``IoMin >= min_overlap``.
4. Each person is assigned **at most one** motorcycle: the highest-overlap
   candidate. Ties break on the lowest ``object_track_id`` (lexicographic), so the
   result never depends on input order.
5. A motorcycle may carry several riders (that is the triple-riding case), so no
   uniqueness constraint is imposed on the motorcycle side.

Confidence is the overlap ratio itself. It is a **geometric plausibility**, not a
calibrated probability, and must never be relabelled one (architecture-review
§13): it says how much the boxes overlap, not how likely the link is to be true.

Tainted tracks abstain
----------------------
A ``TrackState`` marked ``tainted`` (the ID-switch guard) never participates in an
association on either side: a tainted identity may abstain but never support a
conclusion (architecture-review §13). The pair is simply not emitted.

Assumptions (explicit; each is a real limitation)
-------------------------------------------------
* **Per-frame only.** Each ``Association`` describes one instant: ``timestamp`` is
  the frame's, and ``interval`` is ``None`` because no sustained window is
  measured here. Sustained/temporal association (an ``interval`` over a stable
  window, with stability-weighted confidence) is deliberately left to a later
  unit; this unit emits the instantaneous fact and lets the temporal layer
  aggregate.
* **Overlap is not riding.** A pedestrian standing in front of a parked
  motorcycle overlaps it. Geometry alone cannot distinguish that from riding;
  ``min_overlap`` only makes the confusion less likely. Persistence over time (a
  later unit) is what separates the two, which is precisely why this unit emits
  facts rather than conclusions.
* **2-D image space.** Boxes are image-space; a rider on a *distant* bike and a
  pedestrian in front of a *near* bike can overlap identically. No depth or
  ground-plane reasoning exists yet (no metric calibration; Phase 5).
"""

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import Association, TrackState
from ..contracts.enums import AssociationType, ObjectClass
from ..contracts.primitives import BoundingBox

# Provisional: no held-out data exists to tune this against, so it is exposed as
# configuration and labelled provisional rather than buried as a constant.
DEFAULT_MIN_OVERLAP = 0.30


class RiderAssociationConfig(BaseModel):
    """Configuration for rider <-> motorcycle association.

    Frozen + strict like the domain contracts. Every parameter that governs a
    decision lives here rather than as an in-line constant, so the policy is
    inspectable and changeable without editing derivation code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_overlap: float = Field(default=DEFAULT_MIN_OVERLAP, ge=0.0, le=1.0)
    """Minimum intersection-over-smaller-area (``IoMin``) for a person to count as
    riding a motorcycle. **Provisional**: not tuned on held-out data. Raising it
    reduces false links (pedestrians near bikes) and increases missed riders."""


def _area(box: BoundingBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def overlap_over_min_area(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection area over the *smaller* box's area (``IoMin``), in ``[0, 1]``.

    Returns ``0.0`` for disjoint boxes. The frozen ``BoundingBox`` contract
    guarantees positive area (``x2 > x1``, ``y2 > y1``), so the denominator is
    never zero.
    """

    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(_area(a), _area(b))
    # Clamp: guards float error only; intersection <= smaller area geometrically.
    return min(1.0, intersection / smaller)


def _association_id(camera_id: str, rider_id: str, motorcycle_id: str, iso_timestamp: str) -> str:
    """Deterministic, content-derived id (no wall-clock, no counter)."""

    preimage = "\x1f".join((camera_id, rider_id, motorcycle_id, iso_timestamp))
    return "asc-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def associate_riders(
    states: Sequence[TrackState],
    *,
    config: RiderAssociationConfig | None = None,
) -> tuple[Association, ...]:
    """Derive ``rider_of_motorcycle`` associations for **one frame**'s track states.

    Args:
        states: the ``TrackState``s of a single frame (as the tracker emits them).
            Non-person / non-motorcycle classes are ignored; tainted states on
            either side abstain.
        config: association policy; defaults to
            :class:`RiderAssociationConfig` defaults.

    Returns:
        One ``Association`` per associated rider, sorted by
        ``(subject_track_id, object_track_id)`` so the result is independent of
        input order. ``interval`` is ``None`` (per-frame fact; see module
        docstring) and ``confidence`` is the overlap ratio.
    """

    policy = config if config is not None else RiderAssociationConfig()
    # Tainted identities abstain on both sides (architecture-review §13).
    motorcycles = [
        s for s in states if s.object_class is ObjectClass.MOTORCYCLE and not s.tainted
    ]
    riders = [s for s in states if s.object_class is ObjectClass.PERSON and not s.tainted]
    if not motorcycles or not riders:
        return ()

    associations: list[Association] = []
    for rider in riders:
        best: tuple[float, str, TrackState] | None = None
        for motorcycle in motorcycles:
            overlap = overlap_over_min_area(rider.bbox, motorcycle.bbox)
            if overlap < policy.min_overlap:
                continue
            # Deterministic selection: highest overlap wins; an exact tie breaks on
            # the lowest track id, so the outcome never depends on input order.
            if (
                best is None
                or overlap > best[0]
                or (overlap == best[0] and motorcycle.track_id < best[1])
            ):
                best = (overlap, motorcycle.track_id, motorcycle)
        if best is None:
            continue
        overlap, _, motorcycle = best
        associations.append(
            Association(
                association_id=_association_id(
                    rider.camera_id,
                    rider.track_id,
                    motorcycle.track_id,
                    rider.timestamp.isoformat(),
                ),
                camera_id=rider.camera_id,
                subject_track_id=rider.track_id,  # the rider
                object_track_id=motorcycle.track_id,  # the motorcycle
                association_type=AssociationType.RIDER_OF_MOTORCYCLE,
                confidence=overlap,  # geometric plausibility, NOT a probability
                timestamp=rider.timestamp,
                interval=None,  # per-frame fact; no sustained window measured here
            )
        )
    return tuple(sorted(associations, key=lambda a: (a.subject_track_id, a.object_track_id)))
