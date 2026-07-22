"""Evidence manifests with real before/trigger/after frame references (H6).

Upgrades the P1-U11 evidence *stub* -- whose trigger reference is a synthetic
locator keyed only by event identity -- to manifests whose frame references name
frames the engine **actually processed**: during a run the engine records one
:class:`FrameStamp` (identity + PTS media time; metadata only, never pixels)
per processed frame, and this builder picks the trigger/before/after frames
from that record deterministically.

Frame picking (media-time; every rule stated)
---------------------------------------------
Event timestamps are datetimes anchored at the pipeline's fixed media-time
epoch; stamps carry PTS seconds -- :func:`media_seconds` maps between them.

* **trigger** -- the latest processed frame at or before the event's
  ``trigger_at``. (One always exists: the event was reasoned from processed
  frames.)
* **before** -- the latest frame at or before ``trigger - before_seconds``;
  when the stream starts too late for the margin, the earliest frame that
  still precedes the trigger frame; ``None`` when no frame precedes it (the
  trigger was the first processed frame) -- an honest absence, never a
  duplicate reference.
* **after** -- the earliest frame at or after ``trigger + after_seconds``;
  when the stream ends inside the margin, the latest frame that still follows
  the trigger frame; ``None`` when no frame follows. A live run that
  finalizes right at the trigger therefore simply has no after-frame yet.

Locators are **relative** (``frames/<camera_id>/<frame_id>``) and carry no
``sha256`` / ``media_type``: nothing is rendered by this unit, and per the
``ArtifactReference`` contract a locator without a hash is exactly "a reference
we have not hashed". The ``frame_id`` is the ingestion/live identity, so a
future renderer can materialise the artifact from the source without guessing.

Everything else -- id linkage, rule trace, provenance carried from the event,
``created_at`` copied (never wall-clock) -- reuses the P1-U11 stub builder, then
adds one trace step recording the picked frames' identities and media times
(the "timestamps + track ids + metadata" the review surface needs; track ids
are already on the event and manifests never duplicate contract fields).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime

from ..contracts import ConfirmedEvent, EvidenceManifest, MeasuredValue
from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference, RuleTraceStep
from ..persistence.evidence_stub import build_evidence_manifest
from ..pipeline.base import _MEDIA_TIME_EPOCH
from .config import EvidenceConfig


@dataclass(frozen=True)
class FrameStamp:
    """Identity + media time of one processed frame (metadata only, no pixels)."""

    camera_id: str
    frame_id: str
    frame_index: int
    timestamp_seconds: float


def media_seconds(at: datetime) -> float:
    """Media-relative seconds of an event datetime (fixed-epoch anchored)."""

    return (at - _MEDIA_TIME_EPOCH).total_seconds()


def _reference(kind: ArtifactKind, stamp: FrameStamp) -> ArtifactReference:
    return ArtifactReference(
        kind=kind, locator=f"frames/{stamp.camera_id}/{stamp.frame_id}"
    )


def pick_evidence_frames(
    stamps: list[FrameStamp], *, trigger_seconds: float, config: EvidenceConfig
) -> tuple[FrameStamp | None, FrameStamp | None, FrameStamp | None]:
    """Pick ``(before, trigger, after)`` stamps per the module-docstring rules.

    ``stamps`` must be in stream order (ascending timestamps) -- the order the
    engine records them in. Returns ``(None, None, None)`` for an empty record
    (an event cannot normally exist without processed frames; the builder then
    degrades to the stub shape rather than fabricate a reference).
    """

    if not stamps:
        return None, None, None
    times = [stamp.timestamp_seconds for stamp in stamps]

    # trigger: latest frame at or before trigger_seconds (clamped to the first
    # frame if the event somehow predates the stream -- deterministic, and the
    # reference still names a genuinely processed frame).
    trigger_pos = max(0, bisect_right(times, trigger_seconds) - 1)
    trigger = stamps[trigger_pos]

    # before/after search only the frames strictly beside the trigger, so a
    # margin of zero (or one smaller than the inter-frame gap) yields the
    # adjacent frame, never a duplicate reference to the trigger itself.
    before: FrameStamp | None = None
    if trigger_pos > 0:
        pos = (
            bisect_right(
                times, trigger.timestamp_seconds - config.before_seconds, hi=trigger_pos
            )
            - 1
        )
        before = stamps[pos] if pos >= 0 else stamps[0]  # clamp: earliest preceding

    after: FrameStamp | None = None
    if trigger_pos < len(stamps) - 1:
        pos = bisect_left(
            times, trigger.timestamp_seconds + config.after_seconds, lo=trigger_pos + 1
        )
        after = stamps[pos] if pos < len(stamps) else stamps[-1]  # clamp: latest following

    return before, trigger, after


def build_engine_manifest(
    event: ConfirmedEvent,
    stamps: list[FrameStamp],
    *,
    config: EvidenceConfig,
) -> EvidenceManifest:
    """Build one event's manifest with real frame references (pure, deterministic).

    A pure function of the event + the processed-frame record + the margins:
    replaying a run rebuilds byte-identical manifests. With an empty record the
    result degrades to exactly the P1-U11 stub manifest (honest: no frames were
    processed, so none can be referenced).
    """

    manifest = build_evidence_manifest(event)
    before, trigger, after = pick_evidence_frames(
        stamps, trigger_seconds=media_seconds(event.trigger_at), config=config
    )
    if trigger is None:
        return manifest

    picked = [("trigger", trigger)]
    if before is not None:
        picked.insert(0, ("before", before))
    if after is not None:
        picked.append(("after", after))
    frames_step = RuleTraceStep(
        index=len(manifest.rule_trace),
        label="evidence-frames",
        note=";".join(f"{name}={stamp.frame_id}" for name, stamp in picked),
        measurements=tuple(
            MeasuredValue(
                name=f"{name}_frame_media_time",
                value=stamp.timestamp_seconds,
                unit="s",
            )
            for name, stamp in picked
        ),
    )
    return manifest.model_copy(
        update={
            "before_frame": None if before is None else _reference(
                ArtifactKind.BEFORE_FRAME, before
            ),
            "trigger_frame": _reference(ArtifactKind.TRIGGER_FRAME, trigger),
            "after_frame": None if after is None else _reference(
                ArtifactKind.AFTER_FRAME, after
            ),
            "rule_trace": (*manifest.rule_trace, frames_step),
        }
    )
