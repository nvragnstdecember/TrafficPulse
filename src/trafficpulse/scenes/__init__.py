"""Scene authoring and calibration (H12).

The layer between an analyst's calibration surface and the frozen
:class:`~trafficpulse.contracts.SceneConfig` contract. Two concerns, deliberately
separate because one is measured and the other is declared:

* :mod:`~trafficpulse.scenes.calibration` derives the facts a clip can *show* --
  currently the dominant traffic-flow direction, estimated from observed vehicle
  tracks. Promoted from ``viewer/calibration.py``, which now consumes it.
* :mod:`~trafficpulse.scenes.builder` expands a :class:`SceneDraft` -- the minimal
  analyst-authorable vocabulary -- into a complete validated ``SceneConfig``,
  deterministically, so the same drawing always yields the same scene hash.

This package computes no geometry of its own (no point-in-polygon, no crossing,
no heading comparison -- those live in ``geometry``/``observations``), runs no
detector or tracker, imports no ML framework, and performs no I/O. It authors
declarative data and estimates one vector.
"""

from __future__ import annotations

from .builder import (
    DirectionDraft,
    RuleTuning,
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)
from .calibration import (
    FLOW_CLASSES,
    MIN_NET_DISPLACEMENT_PX,
    MIN_TRACK_LIFETIME_SECONDS,
    FlowEstimate,
    estimate_dominant_flow,
)

__all__ = [
    # authoring
    "SceneDraft",
    "ZoneDraft",
    "DirectionDraft",
    "StopLineDraft",
    "SignalGroupDraft",
    "RuleTuning",
    "build_scene",
    "full_frame_polygon",
    # calibration
    "FlowEstimate",
    "estimate_dominant_flow",
    "FLOW_CLASSES",
    "MIN_TRACK_LIFETIME_SECONDS",
    "MIN_NET_DISPLACEMENT_PX",
]
