"""The no-helmet composition root: clip -> events -> evidence -> EventStore (P4-U6).

Exercises the real ingestion, real tracker, real association, real crop extraction,
real reasoning, and the **unmodified** ``EventStore`` on a synthetic clip, with
scripted detection + classification (neither a COCO detector nor any classifier can
read a coloured rectangle). The integration claim under test is that helmet
violations need **no** special case anywhere below ``ConfirmedEvent``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _helmet_fixtures import (
    HELMET,
    NO_HELMET,
    TURBAN,
    helmet_detector_config,
    helmet_example_scene,
    scripted_helmet_classifier,
    scripted_rider_detector,
    write_no_helmet_clip,
)

from trafficpulse.classifier import RawHelmetPrediction, StubHelmetClassifier
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.ingestion.video import VideoIngestionError
from trafficpulse.persistence import EventStore
from trafficpulse.pipeline.no_helmet_runner import (
    NoHelmetSliceRunReport,
    _build_parser,
    run_no_helmet_slice,
)
from trafficpulse.tracking import IouTracker


def run(
    tmp_path: Path,
    *,
    prediction: RawHelmetPrediction = NO_HELMET,
    run_id: str = "run-1",
    classifier: StubHelmetClassifier | None = None,
) -> NoHelmetSliceRunReport:
    clip = write_no_helmet_clip(tmp_path / "clip.mp4")
    return run_no_helmet_slice(
        clip=clip,
        scene=helmet_example_scene(),
        detector=scripted_rider_detector(),
        tracker=IouTracker(),
        classifier=classifier or scripted_helmet_classifier(prediction),
        detector_config=helmet_detector_config(),
        output_dir=tmp_path / "out",
        run_id=run_id,
    )


# --- end to end --------------------------------------------------------------
def test_bare_headed_rider_confirms_and_persists(tmp_path: Path) -> None:
    report = run(tmp_path)

    assert report.event_count == 1
    assert report.manifest_count == 1


def test_every_stage_is_reported(tmp_path: Path) -> None:
    """A run that confirms nothing must be distinguishable from one that saw nothing."""

    report = run(tmp_path)

    assert report.frames_processed == 30
    assert report.unique_tracks == 2  # the bike and its rider
    assert report.riders_associated == 1
    assert report.helmet_observations > 0
    assert report.abstentions == 0
    assert report.exempt_riders == 0
    assert report.event_count == 1


def test_helmeted_rider_observed_but_not_confirmed(tmp_path: Path) -> None:
    report = run(tmp_path, prediction=HELMET)

    assert report.helmet_observations > 0  # perception worked
    assert report.event_count == 0  # reasoning declined


def test_turban_rider_is_exempt_and_reported(tmp_path: Path) -> None:
    report = run(tmp_path, prediction=TURBAN)

    assert report.exempt_riders == 1
    assert report.event_count == 0


def test_unscripted_classifier_abstains(tmp_path: Path) -> None:
    report = run(tmp_path, classifier=StubHelmetClassifier())

    assert report.helmet_observations > 0
    assert report.event_count == 0


# --- honesty of the report ---------------------------------------------------
def test_report_names_the_injected_perception_truthfully(tmp_path: Path) -> None:
    """A scripted stub must never be mistakable for a real model."""

    report = run(tmp_path)

    assert report.detector_kind == "StubDetector"
    assert report.classifier_kind == "StubHelmetClassifier"
    assert report.tracker_kind == "IouTracker"
    assert report.checkpoint is None
    assert report.helmet_checkpoint is None


def test_report_records_the_applied_thresholds(tmp_path: Path) -> None:
    report = run(tmp_path)

    assert report.min_persistence_seconds == 1.0
    assert report.max_observation_gap_seconds == 2.0


def test_report_is_json_serialisable(tmp_path: Path) -> None:
    payload = json.dumps(run(tmp_path).to_dict(), sort_keys=True)
    assert json.loads(payload)["event_count"] == 1


# --- evidence + EventStore (unmodified, no parallel persistence) -------------
def test_evidence_manifest_is_generated_by_the_existing_framework(tmp_path: Path) -> None:
    run(tmp_path)
    stored = EventStore(tmp_path / "out").load("run-1")

    assert len(stored) == 1
    manifest = stored[0].manifest
    assert manifest.event_id == stored[0].event.event_id
    assert manifest.evidence_package_id == f"evp-{stored[0].event.event_id}"


def test_manifest_carries_the_events_provenance(tmp_path: Path) -> None:
    run(tmp_path)
    stored = EventStore(tmp_path / "out").load("run-1")[0]

    assert stored.manifest.scene_config_hash == stored.event.scene_config_hash
    assert stored.manifest.created_at == stored.event.created_at  # never wall-clock


def test_persistence_layout_matches_every_other_violation(tmp_path: Path) -> None:
    """No parallel persistence: the same events/ + manifests/ layout, same keys."""

    run(tmp_path)
    run_dir = tmp_path / "out" / "run-1"
    event_files = sorted((run_dir / "events").glob("*.json"))
    manifest_files = sorted((run_dir / "manifests").glob("*.json"))

    assert [f.name for f in event_files] == [f.name for f in manifest_files]
    assert event_files[0].stem.startswith("evt-")


def test_stored_event_is_a_plain_confirmed_event(tmp_path: Path) -> None:
    """No parallel event type."""

    run(tmp_path)
    event = EventStore(tmp_path / "out").load("run-1")[0].event

    assert type(event).__name__ == "ConfirmedEvent"
    assert event.violation_type is ViolationType.NO_HELMET


# --- replay ------------------------------------------------------------------
def test_replay_yields_identical_events_and_identifiers(tmp_path: Path) -> None:
    first = run(tmp_path, run_id="run-a")
    second = run(tmp_path, run_id="run-b")

    a = EventStore(tmp_path / "out").load("run-a")[0].event
    b = EventStore(tmp_path / "out").load("run-b")[0].event

    assert a.event_id == b.event_id
    assert a.model_dump_json() == b.model_dump_json()
    assert first.scene_config_hash == second.scene_config_hash


def test_replay_into_the_same_run_is_an_idempotent_no_op(tmp_path: Path) -> None:
    """Duplicate suppression: write-once accepts a byte-identical replay."""

    run(tmp_path, run_id="run-1")
    run(tmp_path, run_id="run-1")  # must not raise EventConflictError

    assert len(EventStore(tmp_path / "out").load("run-1")) == 1


def test_replayed_manifests_are_byte_identical(tmp_path: Path) -> None:
    run(tmp_path, run_id="run-a")
    run(tmp_path, run_id="run-b")

    a = (tmp_path / "out" / "run-a" / "manifests")
    b = (tmp_path / "out" / "run-b" / "manifests")
    a_file = next(a.glob("*.json"))
    assert a_file.read_bytes() == (b / a_file.name).read_bytes()


def test_event_ordering_is_deterministic(tmp_path: Path) -> None:
    stored = EventStore(tmp_path / "out")
    run(tmp_path, run_id="run-1")

    ids = [s.event.event_id for s in stored.load("run-1")]
    assert ids == sorted(ids)


# --- fail-fast ---------------------------------------------------------------
def test_missing_clip_raises_a_typed_ingestion_error(tmp_path: Path) -> None:
    with pytest.raises(VideoIngestionError):
        run_no_helmet_slice(
            clip=tmp_path / "nope.mp4",
            scene=helmet_example_scene(),
            detector=scripted_rider_detector(),
            tracker=IouTracker(),
            classifier=scripted_helmet_classifier(),
            detector_config=helmet_detector_config(),
            output_dir=tmp_path / "out",
            run_id="run-1",
        )


def test_scene_without_a_no_helmet_block_fails_before_decoding(tmp_path: Path) -> None:
    scene = helmet_example_scene()
    stripped = scene.model_copy(
        update={
            "rule_parameters": tuple(
                b
                for b in scene.rule_parameters
                if b.violation_type is not ViolationType.NO_HELMET
            )
        }
    )
    with pytest.raises(ValueError, match="no no_helmet rule-parameter block"):
        run_no_helmet_slice(
            clip=write_no_helmet_clip(tmp_path / "clip.mp4"),
            scene=stripped,
            detector=scripted_rider_detector(),
            tracker=IouTracker(),
            classifier=scripted_helmet_classifier(),
            detector_config=helmet_detector_config(),
            output_dir=tmp_path / "out",
            run_id="run-1",
        )


# --- CLI ---------------------------------------------------------------------
def test_cli_requires_both_checkpoints() -> None:
    """The composition root builds real backends; both artifacts are explicit."""

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--clip", "c.mp4", "--scene", "s.yaml", "--output-dir", "out",
            "--run-id", "r", "--checkpoint", "det", "--helmet-checkpoint", "clip",
        ]
    )
    assert args.checkpoint == "det"
    assert args.helmet_checkpoint == "clip"


def test_cli_defaults_to_offline() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--clip", "c.mp4", "--scene", "s.yaml", "--output-dir", "out",
            "--run-id", "r", "--checkpoint", "det", "--helmet-checkpoint", "clip",
        ]
    )
    assert args.allow_download is False


def test_cli_rejects_a_missing_helmet_checkpoint() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--clip", "c.mp4", "--scene", "s.yaml", "--output-dir", "out",
             "--run-id", "r", "--checkpoint", "det"]
        )


# --- regression: the shipped slices are untouched ----------------------------
def test_no_helmet_runner_is_not_exported_from_the_pipeline_package() -> None:
    """Composition roots name backends, so importing the package stays backend-free."""

    import trafficpulse.pipeline as package

    assert not hasattr(package, "run_no_helmet_slice")


def test_importing_the_pipeline_package_pulls_in_no_ml_framework() -> None:
    import importlib
    import sys

    ml = [n for n in sys.modules if n.split(".")[0] in {"torch", "transformers"}]
    saved = {name: sys.modules.pop(name) for name in ml}
    try:
        sys.modules.pop("trafficpulse.pipeline", None)
        importlib.import_module("trafficpulse.pipeline")
        leaked = [n for n in sys.modules if n.split(".")[0] in {"torch", "transformers"}]
        assert leaked == []
    finally:
        sys.modules.update(saved)
        importlib.import_module("trafficpulse.pipeline")
