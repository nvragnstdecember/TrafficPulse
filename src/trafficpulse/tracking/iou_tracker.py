"""The first real tracker backend: an in-repo greedy-IoU associator (P1-U9).

This is the concrete tracker behind the frozen P1-U8 ``Tracker`` seam -- the
tracking analogue of the P1-U7 RT-DETR detector backend. It turns real per-frame
``Detection`` streams into identity-bearing ``TrackState`` sequences using
**greedy intersection-over-union (IoU) association** with a track age / min-hits
lifecycle. It uses **only the Python standard library** (plus the already-present
frozen contracts): no external tracker package, no linear-assignment solver
(``lap`` / ``scipy``), no Kalman filter (``filterpy``), no OpenCV, no ML
framework. Its provenance is therefore unambiguous -- it is original TrafficPulse
Apache-2.0 source -- and it carries no AGPL/permissive-posture risk (ADR-001).

Why this backend, not ByteTrack (P1-U9 execution-time audit outcome)
-------------------------------------------------------------------
The Phase 1 plan's *default* direction is ByteTrack (MIT), gated behind a
mandatory execution-time licence/provenance/dependency audit, with this in-repo
greedy-IoU associator as its documented permissive fallback. At execution time
the preferred path was blocked: no ByteTrack package is installed (so no local
licence/provenance evidence exists), web verification is unavailable (so a
concrete fork's licence cannot be confirmed), and the canonical implementation
needs ``lap``/``scipy`` + a Kalman filter that are absent and are excluded by the
P1-U8 boundary invariant. Per the card's stop condition, an unverifiable external
tracker must not be integrated; the plan's permissive fallback is taken instead.
ByteTrack remains a later, separately-audited enhancement behind this same seam.

Association algorithm (deterministic, no motion model)
------------------------------------------------------
For each ``update`` (one frame):

* every alive track holds the ``bbox`` of the detection it last matched (a
  **matched-box** tracker, not a predicted-box one -- there is no Kalman
  prediction);
* IoU is computed between each incoming detection and each alive track **of the
  same ``object_class``** (class-constrained matching preserves TrafficPulse class
  identity and prevents cross-class absorption);
* candidate (detection, track) pairs with ``IoU >= iou_threshold`` are matched
  **greedily** in a fully deterministic order -- highest IoU first, ties broken by
  detection ordinal then ``track_id`` -- each detection and track used at most
  once;
* a matched track advances (``hits += 1``, age reset, box updated); an unmatched
  detection **spawns** a new track with a fresh stream-local id; an unmatched
  alive track ages, and is removed once it has been unmatched for more than
  ``max_age`` populated frames.

Every incoming detection yields exactly one ``TrackState`` (continuing or new), in
input-detection order -- the same positional 1:1 shape the ``StubTracker``
produces, so downstream code is identical.

What crosses the seam (and what does not)
-----------------------------------------
Only frozen ``TrackState`` values escape, built **solely** by the shared
:class:`~trafficpulse.tracking.adapter.TrackAdapter` (the single construction
authority -- this module never constructs a ``TrackState``). The private
:class:`_Track` bookkeeping object never leaves. Per-frame ``bbox`` / ``timestamp``
/ ``frame_index`` / ``camera_id`` / ``object_class`` / ``confidence`` all carry
through from the current matched detection via the adapter, so they cannot drift.

Deliberately-unpopulated fields (documented limitations)
-------------------------------------------------------
* ``velocity`` is always ``None``: a greedy matched-box associator has no
  Kalman/motion state and thus no interpretable, testable pixels/second velocity;
  heading derivation (P1-U4) uses bbox-center displacement and does not need it.
* ``tainted`` is always ``False``: greedy IoU exposes no trustworthy ID-switch
  signal, so fabricating taint would be dishonest. The P1-U8 taint seam is
  preserved intact for a future guard to populate.
* ``TrackStatus.LOST`` / ``TrackStatus.REMOVED`` are **internal** lifecycle states
  governing re-matching and id retirement; they are never emitted, because an
  emitted ``TrackState`` is tied to a current-frame detection and fabricating a
  box for a frame in which the object was not detected would corrupt downstream
  heading derivation. Emitted statuses are ``TENTATIVE`` (before ``min_hits``) and
  ``ACTIVE`` (at/after it).

Identity, reset, and determinism
--------------------------------
Track ids are stream-local, monotonic, prefixed (``"iou-1"``, ``"iou-2"``, ...):
stable within one instance's stream, **not** globally unique across instances or
runs (ADR-004 does not grant cross-run identity). One ``IouTracker`` instance
tracks one stream. :meth:`reset` clears all tracks, rewinds the id counter, and
resets the frame-progress guard, so an identical stream replays to an identical
result (ids included) from one instance. There is no wall-clock and no randomness,
so two fresh instances fed the same frames produce equal output.

Temporal / empty-frame semantics
--------------------------------
Frame identity travels with the detections (P1-U8 seam); the shared
``single_frame_key`` / ``FrameProgress`` helpers enforce single-frame batches and
strictly-ascending ``frame_index`` + ``timestamp``. An **empty** batch is inert:
it returns ``()`` and does not age tracks, because the seam carries no frame
identity/timestamp for an empty frame (matching ``StubTracker``). Aging advancing
only on populated frames is well-defined and sufficient for correctness; bridging
across empty frames (tracks stay alive) is the intended behaviour for a fixed
offline camera. This backend therefore needs **no** P1-U8 interface change.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import Detection, TrackState
from ..contracts.enums import ObjectClass, TrackStatus
from .adapter import TrackAdapter
from .config import TrackerConfig
from .interface import Tracker
from .raw import TrackAssignment
from .sequencing import FrameProgress, single_frame_key

_TRACK_ID_PREFIX = "iou-"


# --- backend configuration ---------------------------------------------------
class IouTrackerConfig(BaseModel):
    """Backend-specific configuration for the greedy-IoU associator.

    Deliberately separate from the framework-neutral ``TrackerConfig`` (which
    configures the *adapter* / provenance stamp): these knobs are meaningful only
    to this backend and must not leak into the shared seam -- exactly mirroring
    ``RTDetrConfig`` vs ``DetectorConfig`` (P1-U7/U6). Frozen + strict
    (``extra='forbid'``) like the domain contracts. Exposes no framework-native
    object. Only parameters this backend actually uses are present -- no
    speculative generic knobs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    """Minimum IoU for a detection to match an existing track. A pair below this
    is never matched (the detection spawns a new track instead)."""

    max_age: int = Field(default=30, ge=0)
    """How many consecutive *populated* frames a track may go unmatched before it
    is removed and its id retired. ``0`` removes a track the first frame it is
    missed (no gap bridging)."""

    min_hits: int = Field(default=1, ge=1)
    """Number of matched frames before a track is promoted ``TENTATIVE ->
    ACTIVE``. The default ``1`` emits ``ACTIVE`` immediately, which is correct for
    this slice because the P1-U3 rule engine already imposes its own
    >=2-observation confirmation, so the tracker needs no warm-up. A higher value
    introduces a ``TENTATIVE`` warm-up window."""


# --- internal track bookkeeping (never escapes the seam) ---------------------
@dataclass
class _Track:
    """Mutable per-track state held between frames. Private -- never returned."""

    track_id: str
    object_class: ObjectClass
    bbox_xyxy: tuple[float, float, float, float]  # last matched detection's box
    hits: int
    misses: int  # consecutive populated frames unmatched
    status: TrackStatus


class IouTracker(Tracker):
    """A real ``Tracker`` that assigns identity by greedy IoU association.

    Construct with an optional :class:`IouTrackerConfig` (algorithm knobs) and an
    optional :class:`TrackerConfig` (the adapter's provenance ``ModelRef`` stamp),
    both defaulting so the tracker is usable with no arguments.
    """

    def __init__(
        self,
        config: IouTrackerConfig | None = None,
        *,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        self._config = config if config is not None else IouTrackerConfig()
        self._adapter = TrackAdapter(tracker_config)
        self._progress = FrameProgress()
        self._tracks: list[_Track] = []
        self._next_id = 1

    @property
    def config(self) -> IouTrackerConfig:
        return self._config

    def update(self, detections: Sequence[Detection]) -> tuple[TrackState, ...]:
        """Advance the tracker one frame; return that frame's ``TrackState``s.

        Every detection produces exactly one ``TrackState`` (continuing an existing
        track or starting a new one), in input-detection order.

        Raises:
            InconsistentDetectionBatchError: if ``detections`` span >1 frame.
            NonMonotonicFrameError: if the frame does not strictly advance.
            MalformedAssignmentError: if the adapter rejects a stamped assignment
                (defensive: carried-through fields come from valid detections).
        """

        key = single_frame_key(detections)
        if key is None:
            return ()  # empty frame: inert, no aging (P1-U8 empty->empty)
        self._progress.advance(key)

        assignments = self._associate(detections)
        return self._adapter.adapt(assignments)

    def reset(self) -> None:
        """Clear all track state, rewind ids, and reset the progress guard.

        Returns the tracker to its initial (pre-stream) state so an identical
        stream replays to an identical result -- ids included -- from one instance.
        """

        self._tracks = []
        self._next_id = 1
        self._progress.reset()

    # --- association ---------------------------------------------------------
    def _associate(self, detections: Sequence[Detection]) -> list[TrackAssignment]:
        """Match this frame's detections to alive tracks and emit assignments.

        Returns one :class:`TrackAssignment` per detection, in input order. Runs
        greedy IoU matching (class-constrained, deterministic tie-break), advances
        matched tracks, spawns tracks for unmatched detections, and ages/removes
        unmatched tracks.
        """

        matched_track_by_det, matched_det_ids = self._match(detections)

        assignments: list[TrackAssignment] = []
        for ordinal, detection in enumerate(detections):
            track = matched_track_by_det.get(ordinal)
            if track is None:
                track = self._spawn(detection)
            else:
                self._advance(track, detection)
            assignments.append(
                TrackAssignment(
                    track_id=track.track_id,
                    detection=detection,
                    status=track.status,
                    tainted=False,  # greedy IoU exposes no trustworthy ID-switch signal
                    velocity=None,  # no Kalman/motion state -> no interpretable velocity
                )
            )

        self._age_unmatched(matched_det_ids)
        return assignments

    def _match(
        self, detections: Sequence[Detection]
    ) -> tuple[dict[int, _Track], set[str]]:
        """Greedily assign detections to alive tracks by descending IoU.

        Returns ``(track_by_detection_ordinal, matched_track_ids)``. Matching is
        class-constrained and deterministic: candidates are sorted by
        ``(-iou, detection_ordinal, track_id)`` and consumed greedily, each
        detection and track used at most once.
        """

        candidates: list[tuple[float, int, str, _Track]] = []
        for ordinal, detection in enumerate(detections):
            det_box = _box_xyxy(detection)
            for track in self._tracks:
                if track.object_class is not detection.object_class:
                    continue  # class-constrained: no cross-class matching
                iou = _iou(det_box, track.bbox_xyxy)
                if iou >= self._config.iou_threshold:
                    candidates.append((iou, ordinal, track.track_id, track))

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        track_by_det: dict[int, _Track] = {}
        used_dets: set[int] = set()
        used_tracks: set[str] = set()
        matched_track_ids: set[str] = set()
        for _iou_value, ordinal, track_id, track in candidates:
            if ordinal in used_dets or track_id in used_tracks:
                continue
            track_by_det[ordinal] = track
            used_dets.add(ordinal)
            used_tracks.add(track_id)
            matched_track_ids.add(track_id)
        return track_by_det, matched_track_ids

    def _spawn(self, detection: Detection) -> _Track:
        """Create and register a new track for an unmatched detection."""

        track_id = f"{_TRACK_ID_PREFIX}{self._next_id}"
        self._next_id += 1
        status = (
            TrackStatus.ACTIVE if self._config.min_hits <= 1 else TrackStatus.TENTATIVE
        )
        track = _Track(
            track_id=track_id,
            object_class=detection.object_class,
            bbox_xyxy=_box_xyxy(detection),
            hits=1,
            misses=0,
            status=status,
        )
        self._tracks.append(track)
        return track

    def _advance(self, track: _Track, detection: Detection) -> None:
        """Update a matched track with its current-frame detection."""

        track.hits += 1
        track.misses = 0
        track.bbox_xyxy = _box_xyxy(detection)
        track.status = (
            TrackStatus.ACTIVE if track.hits >= self._config.min_hits else TrackStatus.TENTATIVE
        )

    def _age_unmatched(self, matched_track_ids: set[str]) -> None:
        """Age tracks not matched this frame; drop those past ``max_age``."""

        survivors: list[_Track] = []
        for track in self._tracks:
            if track.track_id in matched_track_ids:
                survivors.append(track)
                continue
            track.misses += 1
            if track.misses > self._config.max_age:
                continue  # REMOVED: retire the track and its id
            track.status = TrackStatus.LOST  # internal only; not emitted
            survivors.append(track)
        self._tracks = survivors


# --- geometry helpers (plain Python; no numpy/array crosses any boundary) -----
def _box_xyxy(detection: Detection) -> tuple[float, float, float, float]:
    box = detection.bbox
    return (box.x1, box.y1, box.x2, box.y2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes in ``[0, 1]``.

    Both boxes come from the frozen ``BoundingBox`` contract, so each already has
    positive area (``x2 > x1``, ``y2 > y1``); the union is therefore positive and
    the division is safe.
    """

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    intersection = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)
