"""Offline wrong-way vertical-slice runner + demo command (P1-U12).

The **composition root** for the first real vertical slice: it ties the existing,
independently-tested seams into one deterministic offline run and records a short,
honest report of what was actually executed. It adds **no** new reasoning: it opens
a recorded clip with the P1-U5 ingestion, drives the P1-U10 ``WrongWayPipeline``
(real ``DetectionAdapter`` + injected ``Detector`` + injected ``Tracker`` + the
existing P1-U4 heading derivation / wrong-way reasoner), and persists the resulting
``ConfirmedEvent``s + minimal ``EvidenceManifest``s through the P1-U11 ``EventStore``.

```
recorded clip
  -> open_video (P1-U5)            -> FrameRecord (real PTS)
  -> WrongWayPipeline (P1-U10)     -> ConfirmedEvent
  -> EventStore.persist (P1-U11)   -> events/ + manifests/ JSON under output_dir/run_id
```

Composition root, not a library seam
------------------------------------
Unlike :mod:`trafficpulse.pipeline.wrong_way` (which the boundary test proves
imports **no** backend), this module is the one place that legitimately names
concrete backends -- that is what a composition root does. It is therefore kept out
of ``trafficpulse.pipeline``'s public ``__init__`` exports, so ``import
trafficpulse.pipeline`` stays backend-free and the orchestration core's boundary is
untouched; the runner is reached only through :func:`run_wrong_way_slice` /
``python -m trafficpulse.pipeline``.

Detector honesty (real RT-DETR vs injected detections)
------------------------------------------------------
:func:`run_wrong_way_slice` takes an **injected** ``Detector`` so the caller decides
the perception level and the report records it (``detector_kind``):

* the ``python -m trafficpulse.pipeline`` command builds the **real** RT-DETR
  backend (``detector_kind="rtdetr"``) -- genuine inference behind the P1-U6 seam,
  offline (``local_files_only`` unless ``--allow-download``);
* tests inject a scripted ``StubDetector`` (``detector_kind="StubDetector"``) to
  exercise the real ingestion + real tracker + real rules + real persistence on a
  synthetic clip whose vehicle the COCO RT-DETR does not fire on. Neither path lets
  the runner *fabricate* a detection: the stub replays a caller-authored script; the
  real backend runs real inference.

Offline + deterministic
------------------------
No network (RT-DETR loads ``local_files_only`` by default; ingestion is local), no
wall-clock in the decision path. Event identity is the reasoner's content-derived
hash, so an identical clip + scene + injected components replays to an equal event
set and byte-identical persisted files (see the P1-U11 write-once semantics).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import ValidationError

from ..contracts import ModelRef, ObjectClass, SceneConfig, scene_config_hash
from ..detector.config import DetectorConfig
from ..detector.errors import DetectorError
from ..detector.interface import Detector
from ..ingestion.video import VideoIngestionError, open_video
from ..persistence import EventStore
from ..persistence.errors import PersistenceError
from ..tracking.config import TrackerConfig
from ..tracking.interface import Tracker
from ..tracking.iou_tracker import IouTracker
from .errors import SceneConfigurationError
from .wrong_way import WrongWayPipeline

# Honest run-level provenance the composition root stamps (P2-U1). ``weights_hash``
# stays ``None`` everywhere -- nothing hashes weights in this phase, so a hash would
# be fabricated. The in-repo greedy-IoU associator is our own deterministic
# component, named and versioned truthfully as a provisional reference (it is a
# component/algorithm identity, not a claim of learned weights). The detector ref's
# ``name`` is the real checkpoint the caller passes; its ``version`` is an explicit
# provisional marker because no pinned model version is asserted for this slice.
_IOU_TRACKER_MODEL_REF = ModelRef(name="iou-tracker", version="0.1.0-provisional")


def _rtdetr_model_ref(checkpoint: str) -> ModelRef:
    """Truthful ``ModelRef`` for the real RT-DETR backend built from ``--checkpoint``.

    ``name`` is the checkpoint id/dir actually loaded; ``version`` is a provisional
    marker (no pinned model version is claimed); ``weights_hash`` stays ``None``
    (weights are not hashed in this phase).
    """

    return ModelRef(name=checkpoint, version="provisional")

# Domain error bases the CLI turns into a clean, actionable one-line message + a
# non-zero exit, instead of an unhandled traceback (fail-fast, but legible).
_CLI_ERRORS: tuple[type[Exception], ...] = (
    VideoIngestionError,
    SceneConfigurationError,
    DetectorError,
    PersistenceError,
    ValueError,
)


@dataclass(frozen=True)
class SliceRunReport:
    """An honest, JSON-serialisable summary of one slice run.

    Records exactly what was executed -- the clip and its decoded properties, the
    injected detector/tracker identity (so ``detector_kind`` never lets a stub be
    mistaken for real RT-DETR), the per-stage counts, and where the events were
    persisted. It makes **no** accuracy claim: counts describe this one run only.
    """

    clip_path: str
    scene_config_hash: str
    direction_id: str | None
    lane_id: str
    detector_kind: str
    tracker_kind: str
    checkpoint: str | None
    device: str | None
    width: int
    height: int
    fps: float | None
    codec: str
    frames_processed: int
    track_states_emitted: int
    unique_tracks: int
    event_count: int
    manifest_count: int
    run_id: str
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain ``dict`` (stable key order handled by the JSON dump)."""

        return asdict(self)


def run_wrong_way_slice(
    *,
    clip: Path | str,
    scene: SceneConfig,
    detector: Detector,
    tracker: Tracker,
    detector_config: DetectorConfig,
    output_dir: Path | str,
    run_id: str,
    direction_id: str | None = None,
    camera_id: str | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
) -> SliceRunReport:
    """Run the full offline wrong-way slice on one clip and persist its events.

    Decodes ``clip`` through P1-U5 ingestion, drives the P1-U10 ``WrongWayPipeline``
    with the **injected** ``detector`` + ``tracker`` (the runner fabricates no
    detection and constructs no ``TrackState`` -- both come from the injected seams),
    and persists the confirmed events + minimal manifests via the P1-U11
    ``EventStore`` under ``output_dir/run_id``. Returns a :class:`SliceRunReport`.

    ``checkpoint`` / ``device`` are recorded on the report as provenance only (the
    caller that built a real RT-DETR detector passes them); they are ``None`` for an
    injected stub. A run that yields zero detections/tracks is **not** an error -- it
    returns a report with zero counts (no violation found), distinct from a clip that
    cannot be decoded, which raises a typed ``VideoIngestionError``.

    Raises:
        VideoIngestionError: the clip is missing, not a file, unreadable, or has no
            decodable frames.
        SceneConfigurationError: the scene cannot supply a single governing legal
            direction for the slice.
        DetectorError: the injected detector fails (e.g. real RT-DETR checkpoint or
            dependency missing when constructed by the caller / malformed output).
        PersistenceError: a persisted record conflicts with an earlier differing run.
    """

    pipeline = WrongWayPipeline(
        detector=detector,
        tracker=tracker,
        scene=scene,
        detector_config=detector_config,
        direction_id=direction_id,
    )
    pipeline.reset()

    frames_processed = 0
    track_states_emitted = 0
    tracks: set[tuple[str, str]] = set()
    with open_video(clip, camera_id=camera_id or scene.scene.camera_id) as reader:
        for frame_record in reader:
            states = pipeline.process_frame(frame_record)
            frames_processed += 1
            track_states_emitted += len(states)
            tracks.update((state.camera_id, state.track_id) for state in states)
        metadata = reader.metadata

    events = pipeline.finalize()
    stored = EventStore(output_dir).persist(run_id, events)

    return SliceRunReport(
        clip_path=str(clip),
        scene_config_hash=scene_config_hash(scene),
        direction_id=direction_id,
        lane_id=pipeline.lane_id,
        detector_kind=type(detector).__name__,
        tracker_kind=type(tracker).__name__,
        checkpoint=checkpoint,
        device=device,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        codec=metadata.codec,
        frames_processed=frames_processed,
        track_states_emitted=track_states_emitted,
        unique_tracks=len(tracks),
        event_count=len(events),
        manifest_count=len(stored),
        run_id=run_id,
        output_dir=str(Path(output_dir) / run_id),
    )


# --- CLI (composition root: builds the REAL RT-DETR backend) ------------------
def _load_scene_config(path: Path) -> SceneConfig:
    """Load a ``SceneConfig`` from a JSON or (optionally) YAML file.

    JSON is parsed with pydantic (a base dependency). YAML support is lazy -- it
    imports PyYAML (a dev-only extra, kept out of the runtime dependency set) and
    raises a clear :class:`SceneConfigurationError` if it is absent, so the shipped
    package never hard-depends on YAML.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneConfigurationError(f"cannot read scene config {path}: {exc}") from exc
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise SceneConfigurationError(
                    f"scene file {path} is YAML but PyYAML is not installed; install "
                    "pyyaml (e.g. pip install 'trafficpulse[dev]') or provide a JSON scene"
                ) from exc
            return SceneConfig.model_validate(yaml.safe_load(text))
        return SceneConfig.model_validate_json(text)
    except ValidationError as exc:
        raise SceneConfigurationError(f"invalid scene config {path}: {exc}") from exc


def _parse_label_map(pairs: Sequence[str] | None) -> dict[str, ObjectClass]:
    """Parse ``--label NATIVE=CLASS`` pairs into a detector label map.

    Defaults to ``{"car": ObjectClass.CAR}`` -- the single vehicle class the
    single-lane wrong-way demo needs -- when no pair is given.
    """

    if not pairs:
        return {"car": ObjectClass.CAR}
    label_map: dict[str, ObjectClass] = {}
    for pair in pairs:
        native, sep, cls = pair.partition("=")
        if not sep or not native:
            raise ValueError(f"--label must be NATIVE=CLASS, got {pair!r}")
        try:
            label_map[native] = ObjectClass(cls)
        except ValueError as exc:
            valid = ", ".join(c.value for c in ObjectClass)
            raise ValueError(
                f"--label {pair!r}: {cls!r} is not an ObjectClass (valid: {valid})"
            ) from exc
    return label_map


def _build_rtdetr_detector(
    *, checkpoint: str, device: str, score_threshold: float, local_files_only: bool
) -> Detector:
    """Construct the real RT-DETR backend (fail-fast on missing deps/checkpoint).

    Imported here (not at module top) so the runner module carries no import-time
    coupling to the RT-DETR backend; construction is where real loading happens.
    """

    from ..detector.rtdetr import RTDetrConfig, RTDetrDetector

    return RTDetrDetector(
        RTDetrConfig(
            checkpoint=checkpoint,
            device=device,
            local_files_only=local_files_only,
            # Keep the backend pre-filter at the adapter's authoritative gate so it
            # never hides a detection the adapter would have kept.
            threshold=score_threshold,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trafficpulse.pipeline",
        description=(
            "Offline wrong-way vertical-slice demo: decode one recorded clip, run "
            "real RT-DETR detection + IoU tracking + wrong-way reasoning, and "
            "persist any confirmed events with a minimal evidence manifest. "
            "Fully offline; no network access."
        ),
    )
    parser.add_argument("--clip", required=True, type=Path, help="local video file to process")
    parser.add_argument(
        "--scene", required=True, type=Path, help="SceneConfig file (.json, or .yaml with PyYAML)"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="runtime output root (gitignored; e.g. runs)"
    )
    parser.add_argument("--run-id", required=True, help="identifier for this run's output subdir")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="RT-DETR checkpoint: a locally-cached HuggingFace id or a local directory",
    )
    parser.add_argument(
        "--direction-id",
        default=None,
        help="legal-direction id to reason over when the scene declares more than one",
    )
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, or cuda:N")
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="detector confidence gate (default 0.5)"
    )
    parser.add_argument(
        "--label",
        action="append",
        metavar="NATIVE=CLASS",
        help="map a detector-native label to an ObjectClass (repeatable; default car=car)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit transformers to fetch the checkpoint (default: offline, local only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the offline wrong-way slice demo. Returns an exit code."""

    args = _build_parser().parse_args(argv)
    try:
        scene = _load_scene_config(args.scene)
        detector_config = DetectorConfig(
            label_map=_parse_label_map(args.label),
            score_threshold=args.score_threshold,
            # Truthful detector provenance from the real checkpoint; the adapter
            # stamps it onto every Detection.source_model, and the pipeline collects
            # it into the confirmed events' models (P2-U1).
            source_model=_rtdetr_model_ref(args.checkpoint),
        )
        detector = _build_rtdetr_detector(
            checkpoint=args.checkpoint,
            device=args.device,
            score_threshold=args.score_threshold,
            local_files_only=not args.allow_download,
        )
        report = run_wrong_way_slice(
            clip=args.clip,
            scene=scene,
            detector=detector,
            # Truthful tracker provenance stamped onto every TrackState.tracker.
            tracker=IouTracker(tracker_config=TrackerConfig(tracker=_IOU_TRACKER_MODEL_REF)),
            detector_config=detector_config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            direction_id=args.direction_id,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    except _CLI_ERRORS as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    json.dump(report.to_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0
