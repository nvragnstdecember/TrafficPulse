"""Offline illegal-stopping vertical-slice runner + demo command (P2-U6).

The **composition root** for the second vertical slice -- the structural twin of
the P1-U12 wrong-way runner (:mod:`trafficpulse.pipeline.runner`). It adds **no**
new reasoning: it opens a recorded clip with the P1-U5 ingestion, drives the P2-U5
``IllegalStoppingPipeline`` (real ``DetectionAdapter`` + injected ``Detector`` +
injected ``Tracker`` + the P2-U2/U3 derivations + P2-U4 illegal-stopping reasoner),
and persists the resulting ``ConfirmedEvent``s + minimal ``EvidenceManifest``s
through the P1-U11 ``EventStore``.

```
recorded clip
  -> open_video (P1-U5)                  -> FrameRecord (real PTS)
  -> IllegalStoppingPipeline (P2-U5)     -> ConfirmedEvent
  -> EventStore.persist (P1-U11)         -> events/ + manifests/ JSON under output_dir/run_id
```

Thin sibling, not a generic runner
-----------------------------------
Per the Phase 2 plan (E.8/E.9, and the P2-U5 decision) this is a **thin sibling**
of the wrong-way runner rather than a generalised ``--violation`` selector: two
violations do not justify a multi-rule runner, and a sibling leaves the shipped,
tested wrong-way composition root untouched. The stateless CLI helpers that do not
depend on wrong-way semantics (scene loading, label-map parsing, the real RT-DETR
build, the honest ``ModelRef`` constructors, the CLI error set) are **reused** from
:mod:`trafficpulse.pipeline.runner` so nothing is re-implemented.

Composition root, not a library seam
------------------------------------
Like the wrong-way runner, this module is the one place that legitimately names
concrete backends (``IouTracker``, and the real RT-DETR build via the reused
helper). It is therefore kept **out** of ``trafficpulse.pipeline``'s public
``__init__`` exports, so ``import trafficpulse.pipeline`` stays backend-free; the
runner is reached only through :func:`run_illegal_stopping_slice` or
``python -m trafficpulse.pipeline.illegal_stopping_runner``.

Detector honesty (real RT-DETR vs injected detections)
------------------------------------------------------
:func:`run_illegal_stopping_slice` takes an **injected** ``Detector`` so the caller
decides the perception level and the report records it (``detector_kind``): the CLI
builds the real RT-DETR backend; tests inject a scripted ``StubDetector`` to
exercise the real ingestion + real tracker + real rules + real persistence on a
synthetic clip whose vehicle a COCO RT-DETR does not fire on. Neither path lets the
runner fabricate a detection.

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

from ..contracts import SceneConfig, scene_config_hash
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import open_video
from ..persistence import EventStore
from ..rules.illegal_stopping import illegal_stopping_parameters
from ..tracking.config import TrackerConfig
from ..tracking.interface import Tracker
from ..tracking.iou_tracker import IouTracker
from .illegal_stopping import IllegalStoppingPipeline
from .runner import (
    _CLI_ERRORS,
    _IOU_TRACKER_MODEL_REF,
    _build_rtdetr_detector,
    _load_scene_config,
    _parse_label_map,
    _rtdetr_model_ref,
)


@dataclass(frozen=True)
class IllegalStoppingSliceRunReport:
    """An honest, JSON-serialisable summary of one illegal-stopping slice run.

    Records exactly what was executed -- the clip and its decoded properties, the
    injected detector/tracker identity (so ``detector_kind`` never lets a stub be
    mistaken for real RT-DETR), the resolved no-stopping zones and applied dwell
    threshold, the per-stage counts, and where the events were persisted. It makes
    **no** accuracy claim: counts describe this one run only.
    """

    clip_path: str
    scene_config_hash: str
    no_stopping_zone_ids: tuple[str, ...]
    stationary_duration_seconds: float
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


def run_illegal_stopping_slice(
    *,
    clip: Path | str,
    scene: SceneConfig,
    detector: Detector,
    tracker: Tracker,
    detector_config: DetectorConfig,
    output_dir: Path | str,
    run_id: str,
    camera_id: str | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
) -> IllegalStoppingSliceRunReport:
    """Run the full offline illegal-stopping slice on one clip and persist its events.

    Decodes ``clip`` through P1-U5 ingestion, drives the P2-U5
    ``IllegalStoppingPipeline`` with the **injected** ``detector`` + ``tracker``
    (the runner fabricates no detection and constructs no ``TrackState`` -- both
    come from the injected seams), and persists the confirmed events + minimal
    manifests via the P1-U11 ``EventStore`` under ``output_dir/run_id``. Returns an
    :class:`IllegalStoppingSliceRunReport`.

    ``checkpoint`` / ``device`` are recorded on the report as provenance only. A run
    that yields zero stopped-in-zone tracks is **not** an error -- it returns a
    report with zero event counts (no violation found), distinct from a clip that
    cannot be decoded (raises ``VideoIngestionError``).

    Raises:
        VideoIngestionError: the clip is missing, not a file, unreadable, or has no
            decodable frames.
        SceneConfigurationError: the scene declares no enabled no-stopping zone.
        ValueError: the scene declares no ``illegal_stopping`` block or an unset
            ``stationary_duration``.
        DetectorError: the injected detector fails.
        PersistenceError: a persisted record conflicts with an earlier differing run.
    """

    pipeline = IllegalStoppingPipeline(
        detector=detector,
        tracker=tracker,
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

    return IllegalStoppingSliceRunReport(
        clip_path=str(clip),
        scene_config_hash=scene_config_hash(scene),
        no_stopping_zone_ids=pipeline.no_stopping_zone_ids,
        stationary_duration_seconds=illegal_stopping_parameters(
            scene
        ).stationary_duration_seconds,
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
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trafficpulse.pipeline.illegal_stopping_runner",
        description=(
            "Offline illegal-stopping vertical-slice demo: decode one recorded clip, "
            "run real RT-DETR detection + IoU tracking + illegal-stopping reasoning, "
            "and persist any confirmed events with a minimal evidence manifest. Fully "
            "offline; no network access."
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
    """CLI entry point for the offline illegal-stopping slice demo. Returns an exit code."""

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
        report = run_illegal_stopping_slice(
            clip=args.clip,
            scene=scene,
            detector=detector,
            # Truthful tracker provenance stamped onto every TrackState.tracker.
            tracker=IouTracker(tracker_config=TrackerConfig(tracker=_IOU_TRACKER_MODEL_REF)),
            detector_config=detector_config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    except _CLI_ERRORS as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    json.dump(report.to_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
