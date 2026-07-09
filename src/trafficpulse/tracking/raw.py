"""The tracker-native output type at the integration boundary (P1-U8).

``TrackAssignment`` is the *framework-neutral* per-track output a tracker emits
for one frame. A real detection-based tracker (a future P1-U9 unit) converts its
own native track objects / matrices into ``TrackAssignment`` values **inside**
its ``update()`` implementation; those native objects never escape the tracker.
The :class:`~trafficpulse.tracking.adapter.TrackAdapter` then converts
``TrackAssignment`` values into the frozen U2 ``TrackState`` contract. This is the
seam ADR-001 / architecture-review §5 require: no tracker-specific type may leak
past it into observations, rules, events, or evidence.

Design
------
The assignment is deliberately thin, because the tracker's *only* new information
over the detection is identity + lifecycle:

* ``track_id`` -- the tracker-assigned identity (the core tracking output); the
  adapter rejects an empty one;
* ``detection`` -- the frozen source ``Detection`` this track occupies at this
  frame. Every carried-through field (``camera_id``, ``frame_index``,
  ``timestamp``, ``object_class``, ``bbox``, ``confidence``) is read from it, so
  those fields cannot drift from a real, already-validated detection;
* ``status`` -- the ``TrackStatus`` lifecycle marker the tracker owns;
* ``tainted`` -- the ID-switch guard (architecture-review §13); a tracker sets it
  on an identity discontinuity so downstream reasoning may abstain but never
  bridge support across the switch;
* ``velocity`` -- an optional framework-neutral ``(vx, vy)`` pixels/second tuple.
  It is a plain tuple, not a contract ``Velocity`` or any tracker matrix, so no
  native kinematics type crosses the seam; the adapter validates and wraps it.
  Left ``None`` when the tracker reports no velocity (heading derivation does not
  consume it -- P1-U4 uses bbox-center displacement).
"""

from dataclasses import dataclass

from ..contracts import Detection
from ..contracts.enums import TrackStatus


@dataclass(frozen=True)
class TrackAssignment:
    """One tracker-native track assignment for one frame (pre-adaptation)."""

    track_id: str
    detection: Detection
    status: TrackStatus
    tainted: bool = False
    velocity: tuple[float, float] | None = None
