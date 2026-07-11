#!/usr/bin/env python3
"""Developer convenience demo for TrafficPulse (not part of the package).

Runs the two already-implemented, already-tested vertical slices
(``trafficpulse.pipeline.runner.run_wrong_way_slice`` and
``trafficpulse.pipeline.illegal_stopping_runner.run_illegal_stopping_slice``)
end to end on generated synthetic clips, then prints a concise summary of
whatever ``ConfirmedEvent``s came out.

This script adds **no** new detection, tracking, or rule logic. It is a thin
wrapper around the verified backend:

* clip + scripted-detection generation is reused as-is from the repository's
  own test fixtures (``tests/pipeline/_slice_fixtures.py`` for wrong-way,
  ``tests/pipeline/_stopping_fixtures.py`` for illegal-stopping) -- the same
  generators ``test_slice_runner.py`` / ``test_illegal_stopping_e2e.py``
  already exercise in CI;
* the wrong-way clip is reasoned over the committed
  ``configs/scenes/example-scene.yaml``; the illegal-stopping clip uses that
  same example scene with only its ``zone-no-stop`` polygon and
  ``stationary_duration`` patched into the clip's small pixel space (the
  fixture module's own ``illegal_stopping_test_scene()`` -- needed because a
  1920x1080 zone cannot be reached by a 320x240 demo clip);
* both slices run through the real ``IouTracker`` and the real rule engine.
  Detection is a scripted ``StubDetector`` (no RT-DETR checkpoint required),
  exactly as the CLI's own README already documents: "A COCO RT-DETR does not
  fire the vehicle class on synthetic pixels" -- this demo does not attempt
  real inference. Point ``--checkpoint``-based real RT-DETR runs at real
  clips via ``python -m trafficpulse.pipeline`` /
  ``python -m trafficpulse.pipeline.illegal_stopping_runner`` instead (see
  README "Vertical-slice demos (offline)");
* persistence is the unmodified ``EventStore`` -- events + evidence
  manifests land under ``--output-dir`` in the same layout the real CLI
  produces.

Usage
-----
    python -m pip install -e ".[dev]"   # av, numpy, pydantic, pyyaml
    python demo/run_demo.py
    python demo/run_demo.py --output-dir runs/demo --run-prefix my-demo

Fully offline; writes only under ``--output-dir`` (defaults to ``runs/demo``,
already covered by the repository's ``/runs/`` gitignore entry).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES_DIR = REPO_ROOT / "tests" / "pipeline"

# Reuse the repository's own repository-safe synthetic-clip fixtures instead of
# duplicating them. These live under tests/ (pytest "prepend" import mode, no
# package __init__.py), so a script run outside pytest needs this on sys.path.
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))

import yaml  # noqa: E402  (after sys.path setup)
from _slice_fixtures import (  # noqa: E402
    scripted_down_detector,
    write_wrong_way_clip,
)
from _stopping_fixtures import (  # noqa: E402
    illegal_stopping_test_scene,
    scripted_stopping_detector,
    stopping_detector_config,
    write_illegal_stopping_clip,
)

from trafficpulse.contracts import ConfirmedEvent, ObjectClass, SceneConfig  # noqa: E402
from trafficpulse.detector import DetectorConfig  # noqa: E402
from trafficpulse.persistence import EventStore  # noqa: E402
from trafficpulse.pipeline.illegal_stopping_runner import (  # noqa: E402
    IllegalStoppingSliceRunReport,
    run_illegal_stopping_slice,
)
from trafficpulse.pipeline.runner import (  # noqa: E402
    SliceRunReport,
    run_wrong_way_slice,
)
from trafficpulse.tracking import IouTracker  # noqa: E402

_EXAMPLE_SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
_WRONG_WAY_DIRECTION_ID = "dir-north"  # example scene's legal north; see README


@dataclass(frozen=True)
class _SliceResult:
    label: str
    clip_path: Path
    report: SliceRunReport | IllegalStoppingSliceRunReport
    events: tuple[ConfirmedEvent, ...]


def _load_example_scene() -> SceneConfig:
    raw = yaml.safe_load(_EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    return SceneConfig.model_validate(raw)


def _run_wrong_way(output_dir: Path, run_id: str) -> _SliceResult:
    clip = write_wrong_way_clip(output_dir / "clips" / "wrong_way.mp4")
    report = run_wrong_way_slice(
        clip=clip,
        scene=_load_example_scene(),
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        detector_config=DetectorConfig(label_map={"car": ObjectClass.CAR}),
        output_dir=output_dir,
        run_id=run_id,
        direction_id=_WRONG_WAY_DIRECTION_ID,
    )
    stored = EventStore(output_dir).load(run_id)
    return _SliceResult(
        label="wrong-way",
        clip_path=clip,
        report=report,
        events=tuple(s.event for s in stored),
    )


def _run_illegal_stopping(output_dir: Path, run_id: str) -> _SliceResult:
    clip = write_illegal_stopping_clip(output_dir / "clips" / "illegal_stopping.mp4")
    report = run_illegal_stopping_slice(
        clip=clip,
        scene=illegal_stopping_test_scene(),
        detector=scripted_stopping_detector(),
        tracker=IouTracker(),
        detector_config=stopping_detector_config(),
        output_dir=output_dir,
        run_id=run_id,
    )
    stored = EventStore(output_dir).load(run_id)
    return _SliceResult(
        label="illegal-stopping",
        clip_path=clip,
        report=report,
        events=tuple(s.event for s in stored),
    )


def _print_summary(results: list[_SliceResult], output_dir: Path) -> None:
    print()
    print("=" * 72)
    print("TrafficPulse demo summary (offline, scripted-detector slices)")
    print("=" * 72)
    for result in results:
        report = result.report
        print(f"\n[{result.label}]")
        print(f"  clip:              {result.clip_path}")
        print(f"  frames processed:  {report.frames_processed}")
        print(f"  unique tracks:     {report.unique_tracks}")
        print(f"  confirmed events:  {report.event_count}")
        print(f"  output:            {report.output_dir}")
        if not result.events:
            print("  -> no violation confirmed on this clip")
            continue
        for event in result.events:
            confidence = {
                field: value
                for field, value in event.confidence.model_dump().items()
                if value is not None
            }
            print(
                f"  -> event {event.event_id[:12]}... "
                f"type={event.violation_type.value} "
                f"rule={event.rule_id} "
                f"tracks={event.track_ids} "
                f"trigger_at={event.trigger_at.isoformat()} "
                f"confidence={confidence or '{}'}"
            )
    print(f"\nAll outputs written under: {output_dir}")
    print("(events/manifests JSON per run_id -- same layout the real CLI produces)")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic demo clips and run both TrafficPulse vertical "
            "slices (wrong-way, illegal-stopping) offline with a scripted "
            "detector, printing a summary of confirmed events. Convenience "
            "wrapper only -- see README 'Vertical-slice demos (offline)' for "
            "the real-checkpoint CLI path."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "demo",
        help="runtime output root (gitignored; default: runs/demo)",
    )
    parser.add_argument(
        "--run-prefix",
        default="demo",
        help="prefix for the two run ids (default: 'demo' -> demo-wrong-way, "
        "demo-illegal-stopping)",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    (output_dir / "clips").mkdir(parents=True, exist_ok=True)

    results = [
        _run_wrong_way(output_dir, f"{args.run_prefix}-wrong-way"),
        _run_illegal_stopping(output_dir, f"{args.run_prefix}-illegal-stopping"),
    ]
    _print_summary(results, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
