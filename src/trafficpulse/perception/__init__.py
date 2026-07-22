"""Motorcycle perception foundation for TrafficPulse (v1.1 U1).

The common perception layer that future motorcycle violations (no-helmet,
triple-riding) build on. It **reuses** the shipped seams — the ``Detector``
(motorcycle + rider detection), the IoU ``Tracker`` (stable track ids), and the
``association`` derivation (rider ↔ motorcycle) — and adds only the aggregation
that packages them into stable, per-motorcycle perception observations:

* :class:`MotorcycleObservation` — one tracked motorcycle in one frame, with its
  associated riders (``motorcycle_track_id``, ``bbox``, ``confidence``,
  ``rider_track_ids``).
* :class:`RiderObservation` — one rider associated with one motorcycle in one
  frame (geometric association confidence + a stable rider ordinal).
* :class:`MotorcycleTrackObservation` — a temporal summary of one stable
  motorcycle track (lifespan, frame count, peak/union rider set).

It performs no detection, tracking, classification, or rule reasoning, reads no
pixels, and carries no ML dependency — it is a pure aggregation over frozen
``TrackState`` / ``Association`` contracts, so its output serializes and persists
exactly like every other contract.
"""

from .motorcycle import (
    DEFAULT_PERCEPTION_PRODUCER,
    PerceptionFrame,
    derive_perception_frame,
    summarize_motorcycle_track,
    summarize_motorcycle_tracks,
)
from .observations import (
    MotorcycleObservation,
    MotorcycleTrackObservation,
    RiderObservation,
)

__all__ = [
    "DEFAULT_PERCEPTION_PRODUCER",
    "PerceptionFrame",
    "derive_perception_frame",
    "summarize_motorcycle_track",
    "summarize_motorcycle_tracks",
    "MotorcycleObservation",
    "RiderObservation",
    "MotorcycleTrackObservation",
]
