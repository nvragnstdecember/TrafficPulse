"""Deterministic conversion of tracker output into frozen ``TrackState`` contracts.

``TrackAdapter`` is the boundary architecture-review §5 mandates for tracking: it
consumes framework-neutral ``TrackAssignment`` values and produces frozen U2
``TrackState`` contracts, and it is the *only* place tracker output becomes a
contract (the single, centralized ``TrackState`` construction point -- no other
module in the package builds a ``TrackState``). Everything downstream (heading
derivation, rules, events, evidence) consumes ``TrackState`` only and never sees a
``TrackAssignment`` or any tracker framework type.

Conversion rules (deterministic, order-preserving)
--------------------------------------------------
For each ``TrackAssignment`` in tracker-emission order:

* **Identity.** ``track_id`` is stamped verbatim; an empty one is *malformed* and
  rejected (the frozen ``TrackState`` requires a non-empty id).
* **Carry-through.** ``camera_id``, ``frame_index``, ``timestamp``,
  ``object_class``, ``bbox`` and ``confidence`` are read from the assignment's
  frozen source ``Detection`` -- never recomputed -- so a track's per-frame
  attributes always equal the detection it occupies. (``Detection.confidence`` is
  required and becomes the optional ``TrackState.confidence``.)
* **Lifecycle & taint.** ``status`` and ``tainted`` are stamped from the
  assignment. ``tainted`` is carried faithfully: the adapter never clears it, so a
  tracker-declared ID-switch guard reaches the reasoning layer intact.
* **Velocity.** An assignment ``velocity`` (a plain finite ``(vx, vy)`` tuple) is
  wrapped in the contract ``Velocity``; a non-finite or wrong-shaped tuple is
  *malformed*. ``None`` leaves ``TrackState.velocity`` unset.
* **Provenance.** ``tracker`` is stamped from ``config.tracker``.

Malformed assignments raise :class:`MalformedAssignmentError` (never a leaked
pydantic error). Determinism: no wall-clock, no randomness -- every field is a
pure function of the assignment, its source detection, and the config.
"""

import math
from collections.abc import Iterable

from pydantic import ValidationError

from ..contracts import TrackState, Velocity
from .config import TrackerConfig
from .errors import MalformedAssignmentError
from .raw import TrackAssignment


class TrackAdapter:
    """Converts tracker-native ``TrackAssignment`` output into frozen ``TrackState``."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config if config is not None else TrackerConfig()

    @property
    def config(self) -> TrackerConfig:
        return self._config

    def adapt(self, assignments: Iterable[TrackAssignment]) -> tuple[TrackState, ...]:
        """Convert one frame's track assignments into frozen ``TrackState`` contracts.

        The result preserves the assignments' emission order.

        Raises:
            MalformedAssignmentError: if any assignment cannot be stamped into a
                valid ``TrackState`` (empty ``track_id`` or a malformed velocity).
        """

        return tuple(self._adapt_one(ordinal, a) for ordinal, a in enumerate(assignments))

    def _adapt_one(self, ordinal: int, assignment: TrackAssignment) -> TrackState:
        if not assignment.track_id:
            raise self._malformed(ordinal, assignment, "track_id must be a non-empty string")
        detection = assignment.detection
        velocity = self._to_velocity(ordinal, assignment)
        try:
            return TrackState(
                track_id=assignment.track_id,
                camera_id=detection.camera_id,
                timestamp=detection.timestamp,
                frame_index=detection.frame_index,
                object_class=detection.object_class,
                bbox=detection.bbox,
                confidence=detection.confidence,
                status=assignment.status,
                tainted=assignment.tainted,
                velocity=velocity,
                tracker=self._config.tracker,
            )
        except ValidationError as exc:  # pragma: no cover - defensive: fields pre-validated
            raise self._malformed(ordinal, assignment, str(exc)) from exc

    def _to_velocity(self, ordinal: int, assignment: TrackAssignment) -> Velocity | None:
        vector = assignment.velocity
        if vector is None:
            return None
        if len(vector) != 2 or not all(math.isfinite(component) for component in vector):
            raise self._malformed(
                ordinal, assignment, f"velocity {vector!r} is not a finite (vx, vy) tuple"
            )
        return Velocity(vx=vector[0], vy=vector[1])

    @staticmethod
    def _malformed(
        ordinal: int, assignment: TrackAssignment, detail: str
    ) -> MalformedAssignmentError:
        return MalformedAssignmentError(
            f"malformed track assignment at ordinal={ordinal} "
            f"track_id={assignment.track_id!r}: {detail}"
        )
