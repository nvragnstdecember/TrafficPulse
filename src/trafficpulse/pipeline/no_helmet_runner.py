"""Offline no-helmet vertical-slice runner + demo command (P4-U6).

The **composition root** for the third vertical slice -- the structural twin of the
P1-U12 wrong-way runner (:mod:`trafficpulse.pipeline.runner`) and the P2-U6
illegal-stopping runner. It adds **no** new reasoning: it opens a recorded clip
with the P1-U5 ingestion, drives the P4-U5 ``NoHelmetPipeline`` (real
``DetectionAdapter`` + injected ``Detector`` + injected ``Tracker`` + the P4-U4
association/observation derivation behind the injected ``HelmetClassifier`` seam +
the P4-U5 no-helmet reasoner), and persists the resulting ``ConfirmedEvent``s +
minimal ``EvidenceManifest``s through the **unmodified** P1-U11 ``EventStore``.

```
recorded clip
  -> open_video (P1-U5)              -> FrameRecord (real PTS)
  -> NoHelmetPipeline (P4-U5)        -> ConfirmedEvent
  -> EventStore.persist (P1-U11)     -> events/ + manifests/ JSON under output_dir/run_id
```

Nothing about evidence or persistence is helmet-specific: ``EventStore.persist``
builds each event's manifest through the same violation-agnostic
``build_evidence_manifest`` every other slice uses. This runner does not touch
either -- which is the whole point of the integration.

Thin sibling, not a generic runner
-----------------------------------
Per the Phase 2 decision (E.8/E.9), carried forward: a **thin sibling** rather
than a generalised ``--violation`` selector, leaving the shipped wrong-way and
illegal-stopping composition roots untouched. The stateless CLI helpers that do
not depend on a particular violation's semantics (scene loading, label-map
parsing, the real RT-DETR build, the honest ``ModelRef`` constructors, the CLI
error set) are **reused** from :mod:`trafficpulse.pipeline.runner` so nothing is
re-implemented.

Composition root, not a library seam
------------------------------------
Like its siblings, this module is one of the few places that legitimately names
concrete backends (``IouTracker``, the real RT-DETR build, and -- new here -- the
real ``ZeroShotHelmetClassifier``). It is therefore kept **out** of
``trafficpulse.pipeline``'s public ``__init__`` exports, so ``import
trafficpulse.pipeline`` stays backend-free; the runner is reached only through
:func:`run_no_helmet_slice` or ``python -m trafficpulse.pipeline.no_helmet_runner``.

Perception honesty (real backends vs injected scripts)
------------------------------------------------------
:func:`run_no_helmet_slice` takes an **injected** ``Detector`` *and* an injected
``HelmetClassifier``, so the caller decides the perception level and the report
records **both** (``detector_kind`` / ``classifier_kind``): the CLI builds the real
RT-DETR + real zero-shot backends; tests inject scripted stubs to exercise the real
ingestion + real tracker + real association + real crop extraction + real rules +
real persistence on a synthetic clip that no real model can meaningfully read.
Neither path lets the runner fabricate a detection or a helmet label.

``classifier_kind`` exists for the same reason ``detector_kind`` does: a scripted
stub must never be mistakable for a real model in a run report.

Offline + deterministic
------------------------
No network, no wall-clock in the decision path. Event identity is the reasoner's
content-derived hash, so an identical clip + scene + injected components replays to
an equal event set and byte-identical persisted files (P1-U11 write-once).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..classifier.errors import HelmetClassifierError
from ..classifier.interface import HelmetClassifier
from ..contracts import SceneConfig, scene_config_hash
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import open_video
from ..persistence import EventStore
from ..rules.no_helmet import exempt_riders, no_helmet_parameters
from ..tracking.config import TrackerConfig
from ..tracking.interface import Tracker
from ..tracking.iou_tracker import IouTracker
from .no_helmet import NoHelmetPipeline
from .runner import (
    _CLI_ERRORS,
    _IOU_TRACKER_MODEL_REF,
    _build_rtdetr_detector,
    _load_scene_config,
    _parse_label_map,
    _rtdetr_model_ref,
)

# The label map the no-helmet slice needs by default. Both motorcycle spellings are
# mapped: P4-U1 found ``PekingU/rtdetr_r50vd`` emits the native label "motorbike"
# where COCO says "motorcycle", and the adapter drops unmapped labels silently -- so
# mapping only one spelling would detect zero motorcycles with no error.
DEFAULT_HELMET_LABEL_MAP = ["motorbike=motorcycle", "motorcycle=motorcycle", "person=person"]

# The wrong-way runner's CLI error set plus the classifier seam's base error, so a
# real-backend failure (missing dependency, unavailable checkpoint) surfaces as one
# actionable line rather than a traceback. ``HelmetClassifierError`` is safe to
# import at module top: the classifier error taxonomy carries no ML dependency.
_HELMET_CLI_ERRORS: tuple[type[Exception], ...] = (*_CLI_ERRORS, HelmetClassifierError)


@dataclass(frozen=True)
class NoHelmetSliceRunReport:
    """An honest, JSON-serialisable summary of one no-helmet slice run.

    Records exactly what was executed -- the clip and its decoded properties, the
    injected detector/tracker/classifier identity (so neither a stub detector nor a
    stub classifier can be mistaken for a real model), the applied persistence
    threshold, the per-stage counts, and where the events were persisted.

    The per-stage counts make the pipeline's stages individually legible, so a run
    that confirms nothing can be told apart from a run that observed nothing:
    ``riders_associated`` (association), ``helmet_observations`` /
    ``abstentions`` (perception), ``exempt_riders`` (rule layer), ``event_count``
    (reasoning). It makes **no** accuracy claim: counts describe this one run only.
    """

    clip_path: str
    scene_config_hash: str
    min_persistence_seconds: float
    max_observation_gap_seconds: float | None
    detector_kind: str
    tracker_kind: str
    classifier_kind: str
    checkpoint: str | None
    helmet_checkpoint: str | None
    device: str | None
    width: int
    height: int
    fps: float | None
    codec: str
    frames_processed: int
    track_states_emitted: int
    unique_tracks: int
    riders_associated: int
    helmet_observations: int
    abstentions: int
    exempt_riders: int
    event_count: int
    manifest_count: int
    run_id: str
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain ``dict`` (stable key order handled by the JSON dump)."""

        return asdict(self)


def run_no_helmet_slice(
    *,
    clip: Path | str,
    scene: SceneConfig,
    detector: Detector,
    tracker: Tracker,
    classifier: HelmetClassifier,
    detector_config: DetectorConfig,
    output_dir: Path | str,
    run_id: str,
    camera_id: str | None = None,
    checkpoint: str | None = None,
    helmet_checkpoint: str | None = None,
    device: str | None = None,
) -> NoHelmetSliceRunReport:
    """Run the full offline no-helmet slice on one clip and persist its events.

    Decodes ``clip`` through P1-U5 ingestion, drives the P4-U5 ``NoHelmetPipeline``
    with the **injected** ``detector`` + ``tracker`` + ``classifier`` (the runner
    fabricates no detection, no ``TrackState``, and no helmet label -- all come from
    the injected seams), and persists the confirmed events + minimal manifests via
    the unmodified P1-U11 ``EventStore`` under ``output_dir/run_id``.

    ``checkpoint`` / ``helmet_checkpoint`` / ``device`` are recorded on the report as
    provenance only (the caller that built the real backends passes them); they are
    ``None`` for injected stubs. A run that yields zero detections/riders/events is
    **not** an error -- it returns a report with zero counts, distinct from a clip
    that cannot be decoded, which raises a typed ``VideoIngestionError``.

    Raises:
        VideoIngestionError: the clip is missing, not a file, unreadable, or has no
            decodable frames.
        DetectorError: the injected detector fails (e.g. real RT-DETR checkpoint or
            dependency missing when constructed by the caller / malformed output).
        HelmetClassifierError: the injected classifier fails.
        PersistenceError: a persisted record conflicts with an earlier differing run.
        ValueError: the scene declares no usable ``no_helmet`` parameter block.
    """

    params = no_helmet_parameters(scene)  # fail-fast before any decoding work
    pipeline = NoHelmetPipeline(
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        scene=scene,
        detector_config=detector_config,
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

    # Per-stage perception counts, read back from the observer's own accumulated
    # stream (never recomputed here).
    derivation = pipeline.observer.derivation()
    associations = pipeline.observer.associations()

    return NoHelmetSliceRunReport(
        clip_path=str(clip),
        scene_config_hash=scene_config_hash(scene),
        min_persistence_seconds=params.min_persistence_seconds,
        max_observation_gap_seconds=params.max_observation_gap_seconds,
        detector_kind=type(detector).__name__,
        tracker_kind=type(tracker).__name__,
        classifier_kind=type(classifier).__name__,
        checkpoint=checkpoint,
        helmet_checkpoint=helmet_checkpoint,
        device=device,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        codec=metadata.codec,
        frames_processed=frames_processed,
        track_states_emitted=track_states_emitted,
        unique_tracks=len(tracks),
        riders_associated=len({a.subject_track_id for a in associations}),
        helmet_observations=len(derivation.observations),
        abstentions=len(derivation.abstentions),
        # Reported via the rule layer's own exemption predicate, never re-derived
        # here, so the report can never drift from the decision.
        exempt_riders=len(exempt_riders(derivation.observations)),
        event_count=len(events),
        manifest_count=len(stored),
        run_id=run_id,
        output_dir=str(Path(output_dir) / run_id),
    )


# --- CLI (composition root: builds the REAL RT-DETR + zero-shot backends) -----
def _build_zero_shot_classifier(
    *, checkpoint: str, device: str, local_files_only: bool
) -> HelmetClassifier:
    """Construct the real zero-shot helmet backend (fail-fast on missing artifacts).

    Imported here (not at module top) so the runner module carries no import-time
    coupling to the classifier backend; construction is where real loading happens.
    """

    from ..classifier.zeroshot import ZeroShotHelmetClassifier, ZeroShotHelmetConfig

    return ZeroShotHelmetClassifier(
        ZeroShotHelmetConfig(
            checkpoint=checkpoint, device=device, local_files_only=local_files_only
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trafficpulse.pipeline.no_helmet_runner",
        description=(
            "Offline no-helmet vertical-slice demo: decode one recorded clip, run "
            "real RT-DETR detection + IoU tracking + rider association + real "
            "zero-shot helmet classification + no-helmet reasoning, and persist any "
            "confirmed events with a minimal evidence manifest. Fully offline; no "
            "network access unless --allow-download is given."
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
        "--helmet-checkpoint",
        required=True,
        help=(
            "CLIP-family checkpoint for zero-shot helmet classification: a "
            "locally-cached HuggingFace id or a local directory. Weight provenance "
            "is reviewed per artifact (ADR-001, U4 registry)."
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, or cuda:N")
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="detector confidence gate (default 0.5)"
    )
    parser.add_argument(
        "--label",
        action="append",
        metavar="NATIVE=CLASS",
        help=(
            "map a detector-native label to an ObjectClass (repeatable; default "
            "motorbike=motorcycle, motorcycle=motorcycle, person=person)"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit transformers to fetch the checkpoints (default: offline, local only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the offline no-helmet slice demo. Returns an exit code."""

    args = _build_parser().parse_args(argv)
    try:
        scene = _load_scene_config(args.scene)
        detector_config = DetectorConfig(
            label_map=_parse_label_map(args.label or DEFAULT_HELMET_LABEL_MAP),
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
        classifier = _build_zero_shot_classifier(
            checkpoint=args.helmet_checkpoint,
            device=args.device,
            local_files_only=not args.allow_download,
        )
        report = run_no_helmet_slice(
            clip=args.clip,
            scene=scene,
            detector=detector,
            # Truthful tracker provenance stamped onto every TrackState.tracker.
            tracker=IouTracker(tracker_config=TrackerConfig(tracker=_IOU_TRACKER_MODEL_REF)),
            classifier=classifier,
            detector_config=detector_config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            checkpoint=args.checkpoint,
            helmet_checkpoint=args.helmet_checkpoint,
            device=args.device,
        )
    except _HELMET_CLI_ERRORS as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    json.dump(report.to_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
