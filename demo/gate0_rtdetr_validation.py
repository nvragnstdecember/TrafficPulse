#!/usr/bin/env python3
"""P4-U1 / Gate 0 — RT-DETR class-coverage validation on real footage.

Why this exists
---------------
Every later Phase 4 unit (helmet classifier seam, rider association, helmet
observations) rests on two *unverified* assumptions:

1. RT-DETR actually detects ``motorcycle`` and ``person`` on real uploaded
   traffic footage -- the viewer's upload path has only ever mapped ``car``
   (``viewer/calibration.py``), so no motorcycle or rider has ever been detected
   by this repository on real pixels;
2. the resulting rider **head crops** are large enough for any downstream helmet
   classifier to be meaningful.

This tool measures both, on real video, and reports the numbers honestly. It is a
**diagnostic**, not a pipeline: it decides nothing, confirms nothing, persists no
event, and is not imported by the package. If Gate 0 fails, Phase 4 is re-scoped
rather than built on a false premise.

What it does (and does not) do
------------------------------
It composes only existing, already-tested seams -- exactly as the shipped
composition roots do:

```
open_video (P1-U5 ingestion)
  -> frame_record_to_frame (P3-U2 base; fixed media-time anchor)
  -> RTDetrDetector (P1-U7 backend, real inference)   -> RawDetection
  -> DetectionAdapter (P1-U6 seam)                    -> Detection
  -> IouTracker (P1-U9)                               -> TrackState
  -> measurement + annotation (this module)
```

It implements **no** detection, tracking, observation, rule, event, or
persistence logic. The rider gate and head-crop geometry below are **diagnostic
heuristics for measurement only** -- deliberately crude, and explicitly *not* the
production association/crop logic (that is P4-U4, behind the frozen
``Association`` contract). Nothing here is a contract, and nothing here should be
imported by ``src/``.

Honesty
-------
Every number printed comes from a genuine RT-DETR forward pass over the clip's
real pixels. No detection is authored, injected, or synthesised. Annotated frames
draw *only* boxes RT-DETR actually emitted. A clip containing pre-rendered
bounding boxes burnt into its pixels (some stock "AI demo" footage does) is **not**
a valid input here: the boxes drawn by this tool are real detections, but a
reviewer must not confuse them with the footage's own painted overlays.

Usage
-----
    ./.venv/Scripts/python.exe demo/gate0_rtdetr_validation.py --clip <path>
    ./.venv/Scripts/python.exe demo/gate0_rtdetr_validation.py \\
        --clip runs/viewer/_uploads/<clip>.webm --stride 6 --max-frames 120

Requires the optional real-detector extra (``pip install -e ".[rtdetr]"``) and a
locally cached checkpoint (offline by default; nothing is downloaded). Writes only
under ``--output-dir`` (default ``runs/gate0``, gitignored).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # standalone script convenience
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trafficpulse.contracts import ObjectClass  # noqa: E402
from trafficpulse.contracts.primitives import BoundingBox  # noqa: E402
from trafficpulse.detector import (  # noqa: E402
    DetectionAdapter,
    DetectorConfig,
    RTDetrConfig,
    RTDetrDetector,
)
from trafficpulse.ingestion.video import open_video  # noqa: E402
from trafficpulse.pipeline.base import frame_record_to_frame  # noqa: E402
from trafficpulse.tracking.iou_tracker import IouTracker  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trafficpulse.contracts import TrackState

# The Gate 0 label map: the native labels Phase 4 needs. ``car`` is retained so
# the existing wrong-way upload behaviour stays comparable.
#
# BOTH motorcycle spellings are mapped deliberately. Checkpoints disagree on the
# native COCO id2label vocabulary: ``PekingU/rtdetr_r50vd`` emits **"motorbike"**,
# while other ports emit "motorcycle". The adapter silently drops any native label
# absent from this map (P1-U6 behaviour), so mapping only "motorcycle" against this
# checkpoint yields **zero motorcycles** with no error -- a silent blind spot that
# would invalidate every downstream Phase 4 unit. Mapping both spellings is safe:
# a label the checkpoint never emits simply never matches.
GATE0_LABEL_MAP: dict[str, ObjectClass] = {
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "motorcycle": ObjectClass.MOTORCYCLE,
    "person": ObjectClass.PERSON,
}

DEFAULT_CHECKPOINT = "PekingU/rtdetr_r50vd"  # locally cached; nothing downloaded

# --- diagnostic heuristics (MEASUREMENT ONLY -- not production logic) ---------
# A person is counted as a *candidate rider* when its box overlaps a motorcycle
# box by at least this fraction of the smaller of the two areas. Crude on
# purpose: Gate 0 asks "are riders visible and how big are their heads", not
# "who is definitively riding what" (that is P4-U1 association in the plan of
# record, behind the frozen ``Association`` contract).
RIDER_OVERLAP_MIN = 0.30

# The head region approximated as the top fraction of a rider's bounding box.
# A placeholder for measuring crop scale only; P4-U4 owns real crop geometry.
HEAD_FRACTION = 0.30

# Below this head-crop height, a helmet classifier has essentially no signal.
# A reporting threshold for the Gate 0 verdict -- not a tuned production gate.
HEAD_CROP_MIN_USEFUL_PX = 24.0


@dataclass
class _TrackObservation:
    """One frame's worth of a track's geometry (diagnostic bookkeeping)."""

    frame_index: int
    bbox: BoundingBox
    tainted: bool


@dataclass
class _TrackRecord:
    """Accumulated per-track diagnostics."""

    track_id: str
    object_class: ObjectClass
    observations: list[_TrackObservation] = field(default_factory=list)

    @property
    def frame_span(self) -> int:
        return len(self.observations)

    @property
    def ever_tainted(self) -> bool:
        return any(o.tainted for o in self.observations)


def _area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _overlap_over_min_area(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection area over the smaller box's area (diagnostic rider gate)."""

    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(_area(a), _area(b))
    return intersection / smaller if smaller > 0.0 else 0.0


def _head_crop_height_px(person_bbox: BoundingBox) -> float:
    """Approximate head-crop height: the top fraction of the rider's box."""

    return (person_bbox.y2 - person_bbox.y1) * HEAD_FRACTION


def _candidate_riders(
    states: tuple[TrackState, ...],
) -> list[tuple[TrackState, TrackState]]:
    """Pair each candidate rider with its motorcycle (diagnostic heuristic)."""

    motorcycles = [s for s in states if s.object_class is ObjectClass.MOTORCYCLE]
    persons = [s for s in states if s.object_class is ObjectClass.PERSON]
    pairs: list[tuple[TrackState, TrackState]] = []
    for person in persons:
        best: tuple[float, TrackState] | None = None
        for moto in motorcycles:
            overlap = _overlap_over_min_area(person.bbox, moto.bbox)
            if overlap >= RIDER_OVERLAP_MIN and (best is None or overlap > best[0]):
                best = (overlap, moto)
        if best is not None:
            pairs.append((person, best[1]))
    return pairs


_CLASS_COLOURS: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.CAR: (60, 140, 255),
    ObjectClass.MOTORCYCLE: (255, 90, 40),
    ObjectClass.PERSON: (60, 220, 120),
}


def _annotate(
    image_array: object,
    states: tuple[TrackState, ...],
    rider_pairs: list[tuple[TrackState, TrackState]],
    out_path: Path,
) -> None:
    """Draw the REAL detections/tracks for one frame and save a PNG.

    Draws only what RT-DETR emitted and the tracker assigned: class, track id,
    and -- for candidate riders -- the approximated head-crop box with its pixel
    height, which is the number Gate 0 exists to measure.
    """

    from PIL import Image, ImageDraw  # local import: optional 'rtdetr' extra

    image = Image.fromarray(image_array)  # type: ignore[arg-type]
    draw = ImageDraw.Draw(image)
    rider_ids = {person.track_id for person, _ in rider_pairs}

    for state in states:
        colour = _CLASS_COLOURS.get(state.object_class, (200, 200, 200))
        box = state.bbox
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=colour, width=2)
        label = f"{state.object_class.value}#{state.track_id}"
        if state.tainted:
            label += " TAINTED"
        draw.text((box.x1 + 2, max(0.0, box.y1 - 10)), label, fill=colour)

    for person, moto in rider_pairs:
        box = person.bbox
        head_h = _head_crop_height_px(box)
        draw.rectangle(
            [box.x1, box.y1, box.x2, box.y1 + head_h], outline=(255, 230, 0), width=2
        )
        draw.text(
            (box.x1 + 2, box.y1 + head_h + 2),
            f"head~{head_h:.0f}px rider_of#{moto.track_id}",
            fill=(255, 230, 0),
        )

    if rider_ids:
        draw.text((4, 4), f"candidate riders: {len(rider_ids)}", fill=(255, 230, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _summarise(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": float(len(values)),
        "min": round(min(values), 1),
        "median": round(statistics.median(values), 1),
        "max": round(max(values), 1),
    }


def run_validation(
    *,
    clip: Path,
    output_dir: Path,
    checkpoint: str,
    device: str,
    score_threshold: float,
    stride: int,
    max_frames: int | None,
    annotate_count: int,
) -> dict[str, object]:
    """Run one real RT-DETR pass over ``clip`` and report Gate 0 measurements."""

    detector = RTDetrDetector(
        RTDetrConfig(
            checkpoint=checkpoint,
            device=device,
            local_files_only=True,  # offline: never download
            threshold=min(score_threshold, 0.5),
        )
    )
    adapter = DetectionAdapter(
        DetectorConfig(label_map=GATE0_LABEL_MAP, score_threshold=score_threshold)
    )
    tracker = IouTracker()

    camera_id = "cam-gate0"
    tracks: dict[str, _TrackRecord] = {}
    detections_by_class: dict[ObjectClass, int] = defaultdict(int)
    head_heights: list[float] = []
    rider_track_ids: set[str] = set()
    motorcycle_track_ids: set[str] = set()
    frames_processed = 0
    frames_with_motorcycle = 0
    annotated: list[str] = []
    # Prefer annotating frames that actually contain a candidate rider.
    annotate_dir = output_dir / "frames"

    with open_video(clip, camera_id=camera_id) as reader:
        metadata = reader.metadata
        for frame_record in reader:
            if frame_record.frame_index % stride != 0:
                continue
            if max_frames is not None and frames_processed >= max_frames:
                break
            frame = frame_record_to_frame(frame_record, camera_id=camera_id)
            detections = adapter.adapt_from(detector, frame)
            for detection in detections:
                detections_by_class[detection.object_class] += 1
            states = tracker.update(detections)
            frames_processed += 1

            for state in states:
                record = tracks.setdefault(
                    state.track_id, _TrackRecord(state.track_id, state.object_class)
                )
                record.observations.append(
                    _TrackObservation(
                        frame_index=state.frame_index or frame_record.frame_index,
                        bbox=state.bbox,
                        tainted=state.tainted,
                    )
                )
                if state.object_class is ObjectClass.MOTORCYCLE:
                    motorcycle_track_ids.add(state.track_id)

            rider_pairs = _candidate_riders(states)
            if any(s.object_class is ObjectClass.MOTORCYCLE for s in states):
                frames_with_motorcycle += 1
            for person, _moto in rider_pairs:
                rider_track_ids.add(person.track_id)
                head_heights.append(_head_crop_height_px(person.bbox))

            if rider_pairs and len(annotated) < annotate_count:
                out_path = annotate_dir / f"frame_{frame_record.frame_index:05d}.png"
                _annotate(frame_record.image, states, rider_pairs, out_path)
                annotated.append(str(out_path))

    spans = [float(r.frame_span) for r in tracks.values()]
    tainted_tracks = [r for r in tracks.values() if r.ever_tainted]
    tracks_by_class: dict[str, int] = defaultdict(int)
    for record in tracks.values():
        tracks_by_class[record.object_class.value] += 1

    head_summary = _summarise(head_heights)
    verdict = _verdict(
        motorcycle_detections=detections_by_class[ObjectClass.MOTORCYCLE],
        person_detections=detections_by_class[ObjectClass.PERSON],
        rider_count=len(rider_track_ids),
        head_summary=head_summary,
    )

    return {
        "clip": str(clip),
        "checkpoint": checkpoint,
        "device": device,
        "score_threshold": score_threshold,
        "stride": stride,
        "video": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "codec": metadata.codec,
            "frame_count": metadata.frame_count,
        },
        "frames_processed": frames_processed,
        "frames_with_motorcycle": frames_with_motorcycle,
        "detections_by_class": {k.value: v for k, v in sorted(detections_by_class.items())},
        "tracks_by_class": dict(sorted(tracks_by_class.items())),
        "motorcycle_track_count": len(motorcycle_track_ids),
        "candidate_rider_track_count": len(rider_track_ids),
        "track_stability": {
            "total_tracks": len(tracks),
            "median_frame_span": round(statistics.median(spans), 1) if spans else None,
            "max_frame_span": round(max(spans), 1) if spans else None,
            "single_frame_tracks": sum(1 for s in spans if s <= 1.0),
            "tainted_tracks": len(tainted_tracks),
        },
        "rider_head_crop_px": head_summary,
        "annotated_frames": annotated,
        "verdict": verdict,
    }


def _verdict(
    *,
    motorcycle_detections: int,
    person_detections: int,
    rider_count: int,
    head_summary: dict[str, float] | None,
) -> dict[str, object]:
    """State plainly whether Gate 0 passed, and why. No claim beyond the numbers."""

    reasons: list[str] = []
    if motorcycle_detections == 0:
        reasons.append("RT-DETR detected no motorcycle on this clip")
    if person_detections == 0:
        reasons.append("RT-DETR detected no person on this clip")
    if rider_count == 0:
        reasons.append("no person overlapped a motorcycle (no candidate rider)")
    if head_summary is not None and head_summary["median"] < HEAD_CROP_MIN_USEFUL_PX:
        reasons.append(
            f"median rider head crop {head_summary['median']:.0f}px is below the "
            f"{HEAD_CROP_MIN_USEFUL_PX:.0f}px reporting floor"
        )
    return {
        "gate0_passed": not reasons,
        "reasons": reasons,
        "note": (
            "Measurement only. Establishes detector class coverage and rider crop "
            "scale on this clip; claims no accuracy for any downstream classifier."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P4-U1 / Gate 0: validate RT-DETR motorcycle+person coverage on real footage."
    )
    parser.add_argument("--clip", required=True, type=Path, help="path to a real video file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gate0"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--stride", type=int, default=1, help="process every Nth frame (>=1)"
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--annotate", type=int, default=6, help="annotated frames to write")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stride < 1:
        print("error: --stride must be >= 1", file=sys.stderr)
        return 2
    if not args.clip.is_file():
        print(f"error: clip not found: {args.clip}", file=sys.stderr)
        return 2

    report = run_validation(
        clip=args.clip,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        stride=args.stride,
        max_frames=args.max_frames,
        annotate_count=args.annotate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "gate0_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    verdict = report["verdict"]
    assert isinstance(verdict, dict)
    print("=" * 70)
    print("  P4-U1 / Gate 0 — RT-DETR class-coverage validation")
    print("=" * 70)
    print(f"  clip:              {report['clip']}")
    print(f"  frames processed:  {report['frames_processed']} (stride {report['stride']})")
    print(f"  detections:        {report['detections_by_class']}")
    print(f"  tracks:            {report['tracks_by_class']}")
    print(f"  motorcycle tracks: {report['motorcycle_track_count']}")
    print(f"  candidate riders:  {report['candidate_rider_track_count']}")
    print(f"  track stability:   {report['track_stability']}")
    print(f"  rider head crop:   {report['rider_head_crop_px']}")
    print(f"  annotated frames:  {len(report['annotated_frames'])} under {args.output_dir}")
    print("-" * 70)
    print(f"  GATE 0 PASSED:     {verdict['gate0_passed']}")
    for reason in verdict["reasons"]:
        print(f"    - {reason}")
    print(f"  report:            {report_path}")
    print("=" * 70)
    return 0 if verdict["gate0_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
