"""Wrong-way vertical-slice orchestration (P1-U10).

A thin, deterministic **offline** orchestration that runs one recorded stream end
to end through *existing* components -- it wires, it does not compute:

```
FrameRecord (P1-U5 ingestion)
  -> detector Frame (identity + opaque image)
  -> Detector + DetectionAdapter (P1-U6 seam)          -> Detection
  -> Tracker (P1-U8 seam; StubTracker or IouTracker)   -> TrackState
  -> group by (camera_id, track_id) in timestamp order
  -> derive_heading_observations_with_taint (P1-U4)    -> HeadingDerivation
  -> WrongWayReasoner.run_derivation (P1-U4)            -> ConfirmedEvent
```

Design: thin composition
------------------------
Every stage is an injected or existing component consumed **only** across its
frozen contract seam. This module implements no detection, no association, no
heading calculation, and no rule logic; it re-uses
``DetectionAdapter.adapt_from`` and ``WrongWayReasoner.run_derivation`` -- the two
composition points that already exist -- so the wiring provably adds no behaviour
(the acceptance test asserts the pipeline yields the *same* ``ConfirmedEvent`` set
as calling the derivation + reasoner directly on the same ``TrackState``s).

Backend independence
--------------------
The orchestrator depends on the ``Detector`` and ``Tracker`` **abstractions**, the
frozen contracts, and the existing observation/rule APIs -- never on
``RTDetrDetector``, ``StubDetector``, ``IouTracker``, ``StubTracker``, torch,
transformers, or any backend-native object. Imports are taken from the detector
*submodules* (never the ``detector`` package root) so ``RTDetrDetector`` is not
even pulled into this module's namespace. Any implementation of the two seams
drops in through the constructor unchanged.

The one conversion this layer owns: FrameRecord -> detector Frame
-----------------------------------------------------------------
Ingestion emits ``FrameRecord.timestamp_seconds`` as a **media-relative** float
(PTS seconds), while the detector ``Frame`` / frozen ``Detection`` require a
timezone-aware ``datetime``. :func:`frame_record_to_frame` bridges that at a
**fixed UTC epoch anchor** (``1970-01-01T00:00:00Z + timedelta(seconds=pts)``): it
preserves media-time *semantics* exactly -- the inter-frame delta equals the PTS
delta, which is the only temporal quantity the reasoner's ``min_persistence``
consumes -- while using **no wall-clock and no nominal FPS**. The anchor is a
deterministic media-time marker, not a fabricated absolute capture date; mapping
to true capture time remains a later concern requiring external metadata
(ingestion docstring / architecture-review §17, §239). ``camera_id`` resolves to
the frame's own id when present (real ingestion stamps the scene camera onto every
frame) else the scene camera; ``frame_index`` and the opaque ``image`` carry
through unchanged (the image is never copied).

Empty-detection-frame semantics (preserved, not hidden)
-------------------------------------------------------
When the detector returns zero detections for a frame, the ``Tracker`` seam
receives an empty ``Sequence[Detection]``. Per the documented P1-U8/U9 behaviour
that empty batch is **inert**: it returns ``()``, does **not** age tracks, and
does **not** advance the frame-progress guard (the seam carries no frame identity
for an empty frame). The orchestrator honours that faithfully -- it fabricates no
detection and no frame metadata to advance time. The track simply has no state at
that frame; ``IouTracker`` therefore *bridges* identity across the gap when
detections resume, and heading derivation bridges the ordinary gap by displacement
between the states that straddle it. This is a known, tested limitation, not a
redesign of P1-U8/U9.

Determinism
-----------
No wall-clock, no randomness. Track groups are iterated in ``(camera_id,
track_id)`` order, each track's states in ``(timestamp, frame_index)`` order, and
the emitted events sorted by ``(trigger_at, event_id)`` -- so the result is a pure
function of the injected components, the frame stream, and the scene, independent
of transient ordering. ``finalize`` builds a fresh reasoner from the scene each
call, so it is idempotent over the accumulated history; ``reset`` returns the
orchestration to a replayable initial state.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from ..contracts import ConfirmedEvent, Detection, SceneConfig, TrackState, scene_config_hash
from ..contracts.scene import DirectionVector, LegalDirection
from ..detector.adapter import DetectionAdapter
from ..detector.config import DetectorConfig
from ..detector.frame import Frame
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..observations.heading import derive_heading_observations_with_taint
from ..rules.engine import RuleEngine
from ..rules.wrong_way import WrongWayReasoner, wrong_way_parameters
from ..tracking.interface import Tracker
from .errors import SceneConfigurationError

# Fixed media-time anchor: media-relative PTS seconds are added to this epoch to
# form the timezone-aware datetime the detector/contract seams require. It is a
# deterministic marker (no wall-clock), and only inter-frame deltas are load-
# bearing downstream -- see the module docstring.
_MEDIA_TIME_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def frame_record_to_frame(frame_record: FrameRecord, *, camera_id: str) -> Frame:
    """Convert an ingestion ``FrameRecord`` into a detector ``Frame``.

    ``camera_id`` is the resolved non-empty camera id to stamp (the frame's own id
    when present, else the scene camera). The media-relative PTS timestamp is
    anchored at a fixed UTC epoch (see module docstring); ``frame_index`` and the
    opaque ``image`` carry through unchanged (the image is not copied).
    """

    return Frame(
        camera_id=camera_id,
        frame_index=frame_record.frame_index,
        timestamp=_MEDIA_TIME_EPOCH + timedelta(seconds=frame_record.timestamp_seconds),
        image=frame_record.image,
    )


def _resolve_legal_direction(
    scene: SceneConfig, direction_id: str | None
) -> tuple[DirectionVector, str]:
    """Resolve the single governing ``(legal_direction, lane_id)`` for the slice.

    Raises:
        SceneConfigurationError: if the single lane cannot be picked (no legal
            direction; more than one with no ``direction_id``; an unknown
            ``direction_id``; or the chosen direction has no zone/lane id).
    """

    directions = scene.legal_directions
    chosen: LegalDirection
    if direction_id is not None:
        match = next((d for d in directions if d.direction_id == direction_id), None)
        if match is None:
            raise SceneConfigurationError(
                f"scene has no legal direction with direction_id={direction_id!r}"
            )
        chosen = match
    elif not directions:
        raise SceneConfigurationError(
            "scene declares no legal direction; wrong-way orchestration needs one"
        )
    elif len(directions) > 1:
        available = tuple(d.direction_id for d in directions)
        raise SceneConfigurationError(
            "scene declares more than one legal direction; the single-lane slice "
            f"needs an explicit direction_id (available: {available})"
        )
    else:
        chosen = directions[0]

    if not chosen.zone_ids:
        raise SceneConfigurationError(
            f"legal direction {chosen.direction_id!r} carries no zone/lane id"
        )
    return chosen.vector, chosen.zone_ids[0]


class WrongWayPipeline:
    """Deterministic offline orchestration for the first wrong-way vertical slice.

    Composes an injected ``Detector`` and ``Tracker`` with the existing P1-U4
    heading derivation and wrong-way reasoner over one ``SceneConfig``. The
    ``detector_config`` configures the shared ``DetectionAdapter`` seam
    (label map + provenance); ``direction_id`` selects which legal direction
    governs the run when the scene declares more than one (the single-lane slice).
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        direction_id: str | None = None,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._scene = scene
        self._adapter = DetectionAdapter(detector_config)
        self._params = wrong_way_parameters(scene)
        self._legal_direction, self._lane_id = _resolve_legal_direction(scene, direction_id)
        self._scene_hash = scene_config_hash(scene)
        self._history: dict[tuple[str, str], list[TrackState]] = {}

    @property
    def lane_id(self) -> str:
        """The resolved single-lane id this pipeline reasons over."""

        return self._lane_id

    def reset(self) -> None:
        """Return the orchestration to a replayable initial state.

        Resets the injected (stateful) ``Tracker`` and clears the accumulated
        per-track history. The detector needs no reset -- it holds no temporal
        state across frames -- and the reasoner/engine are ephemeral (rebuilt per
        :meth:`finalize`), so the same frame stream replays to an equal result.
        """

        self._tracker.reset()
        self._history = {}

    def process_frame(self, frame_record: FrameRecord) -> tuple[TrackState, ...]:
        """Detect + track one frame, accumulate its states, and return them.

        Runs the detector + adapter (P1-U6 seam) and the tracker (P1-U8 seam) for
        exactly one frame, appends the emitted ``TrackState``s to the per-track
        history grouped by ``(camera_id, track_id)``, and returns that frame's
        states. A zero-detection frame yields ``()`` and changes no track state
        (the empty batch is inert at the tracker seam). Events are computed by
        :meth:`finalize` from the full history, because heading derivation is
        per-track and needs the whole track.
        """

        camera_id = frame_record.camera_id or self._scene.scene.camera_id
        frame = frame_record_to_frame(frame_record, camera_id=camera_id)
        detections: tuple[Detection, ...] = self._adapter.adapt_from(self._detector, frame)
        states = self._tracker.update(detections)
        for state in states:
            self._history.setdefault((state.camera_id, state.track_id), []).append(state)
        return tuple(states)

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        """Derive observations + reason over the accumulated history; return events.

        Groups by ``(camera_id, track_id)``, calls the existing
        ``derive_heading_observations_with_taint`` per track (with the scene's lane
        legal direction and deviation threshold), feeds each derivation -- taint
        restarts included -- to a fresh ``WrongWayReasoner``, and returns the
        confirmed events sorted by ``(trigger_at, event_id)``. Idempotent: it is a
        pure function of the accumulated history (the reasoner is rebuilt here, not
        held across frames).
        """

        reasoner = WrongWayReasoner(
            RuleEngine(), self._params, scene_config_hash=self._scene_hash
        )
        events: list[ConfirmedEvent] = []
        for key in sorted(self._history):
            track = sorted(self._history[key], key=lambda s: (s.timestamp, s.frame_index or 0))
            derivation = derive_heading_observations_with_taint(
                track,
                legal_direction=self._legal_direction,
                lane_id=self._lane_id,
                deviation_max_degrees=self._params.deviation_max_degrees,
            )
            events.extend(reasoner.run_derivation(derivation))
        return tuple(sorted(events, key=lambda e: (e.trigger_at, e.event_id)))

    def process(self, frames: Iterable[FrameRecord]) -> tuple[ConfirmedEvent, ...]:
        """Run one complete offline stream: ``reset`` -> stream frames -> ``finalize``.

        A self-contained run: it resets first, so repeated calls on one instance
        (and fresh instances) replay an identical frame stream to an equal event
        set. Frames must arrive in ascending ``frame_index`` order (the real
        ingestion order); the tracker seam enforces strict frame monotonicity.
        """

        self.reset()
        for frame_record in frames:
            self.process_frame(frame_record)
        return self.finalize()
