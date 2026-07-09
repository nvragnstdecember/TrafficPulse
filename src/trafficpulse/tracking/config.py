"""Tracker-integration configuration (P1-U8).

``TrackerConfig`` is the *framework-neutral adapter configuration*: the settings
the adapter needs to convert any tracker's ``TrackAssignment`` output into frozen
``TrackState`` contracts. It is intentionally free of tracker-specific tuning
(track buffer / max-age / min-hits / IoU-match / Kalman / re-ID settings) -- those
belong to a future tracker *implementation's* own configuration (P1-U9), not to
this shared seam, so no tracker-specific assumption leaks into callers.

Only one genuinely generic, non-speculative setting is needed at the foundation
seam: the ``tracker`` model reference stamped onto every produced
``TrackState.tracker`` as provenance (the tracking analogue of
``DetectorConfig.source_model``). A ``TrackerConfig`` is therefore not invented
for mere symmetry with the detector config -- it carries exactly the one field
the adapter must stamp, and defaults so a stub needs no configuration at all.

Validation reuses pydantic (already a project runtime dependency); no new
dependency is added. The model is frozen + strict (``extra='forbid'``) like the
domain contracts, but it lives in the ``tracking`` package rather than
``contracts`` because it is component configuration, not part of the typed
perception->reasoning data flow.
"""

from pydantic import BaseModel, ConfigDict

from ..contracts import ModelRef


class TrackerConfig(BaseModel):
    """Framework-neutral configuration for adapting tracker output to ``TrackState``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tracker: ModelRef | None = None
    """Stamped onto every produced ``TrackState.tracker`` as provenance. Named to
    match the ``TrackState`` field. ``None`` (the default) leaves the provenance
    field unset, which is correct for the dependency-free stub."""
