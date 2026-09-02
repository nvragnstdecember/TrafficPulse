#!/usr/bin/env python
"""Run the T0 manifest's *calibrated* wrong-way variant on real footage.

    ./.venv/Scripts/python.exe demo/wrongway_calibrated_validation.py --stride 5

The manifest (``test-videos/evaluation/manifest.yaml``) records this as the one
expectation in the corpus whose sign is asserted and whose count is **not**:

    wrongway_001.ogv:
      calibrated_variant:
        expected: {wrong_way: ">0"}
        basis: expected-nonzero
        established: false

This script produces that result and asserts **only the sign**. It prints the count
it observed, but the count is a measurement of this run, not a target, and nothing
here compares it to a number.

What is being demonstrated, precisely
--------------------------------------
The clip is signed **lawful** contraflow through a roundabout. The reasoner is not
claiming those drivers broke a law; it is detecting *sustained opposition to the
scene's declared legal direction*, which is exactly what wrong-way reasoning claims
to do. That is the honest framing, and it is the manifest's own.

How the legal direction is obtained (and why not by hand)
----------------------------------------------------------
Two passes, mirroring what ``ProcessingService`` does for an uncalibrated upload:

1. **Measure.** Detect + track, then ``estimate_dominant_flow`` over the resulting
   tracks. The legal direction is the *observed dominant flow* -- the direction the
   majority of traffic actually travels.
2. **Reason.** Author a scene declaring that direction and run the real wrong-way
   slice against it.

The direction is therefore **derived from the footage**, not drawn by whoever ran
this. Hand-drawing a carriageway polygon "by eye" would be the author choosing the
answer, and on a clip whose whole point is the direction of travel that is precisely
the thing that must not be chosen.

The cost of that choice, stated plainly
-----------------------------------------
The authored scene is a **whole-frame** lane, because a measured flow vector does
not tell you where the carriageway edges are. So the lane-containment gate admits
every track in frame, and traffic on any opposing carriageway is eligible to be
reported. The manifest's variant describes an analyst drawing a lane over the main
carriageway only; this is the auto-calibrated approximation of it, and its result
must be read as such. It is a *weaker* scene than the manifest describes, not a
stronger one.

This result is auto-calibrated and must never be quoted as an uncalibrated result:
the uncalibrated run of this same clip is structurally zero (see
``test-videos/run_manifest.py``), and that zero and this non-zero are answers to
two different questions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from trafficpulse.contracts import TrackState, scene_config_hash  # noqa: E402
from trafficpulse.contracts.enums import ObjectClass  # noqa: E402
from trafficpulse.contracts.scene import ZoneType  # noqa: E402
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.rtdetr import RTDetrConfig, RTDetrDetector  # noqa: E402
from trafficpulse.ingestion.video import open_video  # noqa: E402
from trafficpulse.pipeline.base import frame_record_to_frame  # noqa: E402
from trafficpulse.pipeline.runner import run_wrong_way_slice  # noqa: E402
from trafficpulse.scenes import (  # noqa: E402
    CALIBRATION_SOURCE_AUTO,
    DirectionDraft,
    SceneDraft,
    ZoneDraft,
    build_scene,
    estimate_dominant_flow,
    full_frame_polygon,
)
from trafficpulse.tracking.iou_tracker import IouTracker  # noqa: E402

DEFAULT_CLIP = REPO / "test-videos" / "wrong-way" / "wrongway_001.ogv"
DEFAULT_OUT = REPO / "runs" / "wrongway-calibrated"

#: The same mapping the production launcher uses (``serve.LABEL_MAP``): this
#: checkpoint emits the VOC-style "motorbike" spelling.
LABEL_MAP: dict[str, ObjectClass] = {
    "person": ObjectClass.PERSON,
    "bicycle": ObjectClass.BICYCLE,
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
}


def _detector(checkpoint: str, device: str, threshold: float) -> RTDetrDetector:
    return RTDetrDetector(
        RTDetrConfig(
            checkpoint=checkpoint,
            device=device,
            local_files_only=True,  # offline: never download
            threshold=threshold,
        )
    )


def measure_flow(
    clip: Path,
    *,
    checkpoint: str,
    device: str,
    threshold: float,
    stride: int,
    max_frames: int | None,
) -> tuple[object, int, int]:
    """Pass 1: detect + track, and estimate the dominant flow from the tracks."""

    detector = _detector(checkpoint, device, threshold)
    adapter = DetectionAdapter(DetectorConfig(label_map=LABEL_MAP, score_threshold=threshold))
    tracker = IouTracker()

    states: list[TrackState] = []
    frames_used = 0
    detections = 0
    with open_video(clip, camera_id="cam-wrongway") as reader:
        for index, frame_record in enumerate(reader):
            if index % stride:
                continue
            if max_frames is not None and frames_used >= max_frames:
                break
            frames_used += 1
            frame = frame_record_to_frame(frame_record, camera_id="cam-wrongway")
            adapted = adapter.adapt_from(detector, frame)
            detections += len(adapted)
            states.extend(tracker.update(adapted))

    flow = estimate_dominant_flow(states)
    return flow, frames_used, detections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrated wrong-way validation (T0).")
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--checkpoint", default="PekingU/rtdetr_r50vd")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu | cuda | cuda:N (RTDetrConfig rejects 'auto')",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=5, help="decode every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.clip.is_file():
        print("clip not found: " + str(args.clip))
        return 2

    print("clip   : " + str(args.clip))
    print("stride : " + str(args.stride) + "  threshold: " + str(args.threshold))
    print("")
    print("--- pass 1: measuring dominant flow -------------------------------")
    flow, frames_used, detections = measure_flow(
        args.clip,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print("frames analysed : " + str(frames_used))
    print("detections      : " + str(detections))
    if flow is None:
        print("")
        print("ABSTAINED: no dominant flow could be measured, so no legal direction can be")
        print("declared and no calibrated scene can be authored. This is a real outcome, not")
        print("a failure -- but it means the calibrated variant cannot be run on this sample.")
        return 1
    print(
            "dominant flow   : dx="
            + str(round(flow.dx, 2))
            + " dy="
            + str(round(flow.dy, 2))
            + "  heading="
            + str(round(flow.heading_degrees, 1))
            + "deg  movers="
            + str(flow.mover_count)
        )

    draft = SceneDraft(
        scene_name="wrongway-001-autocalibrated",
        camera_id="cam-wrongway",
        frame_width=1920,
        frame_height=1080,
        zones=(
            ZoneDraft(
                zone_id="carriageway",
                zone_type=ZoneType.LANE,
                polygon=full_frame_polygon(1920, 1080),
            ),
        ),
        direction=DirectionDraft(
            direction_id="legal",
            zone_id="carriageway",
            dx=flow.dx,
            dy=flow.dy,
        ),
    )
    scene = build_scene(
        draft, scene_id="scene-wrongway-001", calibration_source=CALIBRATION_SOURCE_AUTO
    )
    print("scene hash      : " + scene_config_hash(scene))

    print("")
    print("--- pass 2: wrong-way reasoning against that direction ------------")
    report = run_wrong_way_slice(
        clip=args.clip,
        scene=scene,
        detector=_detector(args.checkpoint, args.device, args.threshold),
        tracker=IouTracker(),
        detector_config=DetectorConfig(label_map=LABEL_MAP, score_threshold=args.threshold),
        output_dir=args.out,
        run_id="wrongway-calibrated",
        direction_id="legal",
        camera_id="cam-wrongway",
        checkpoint=args.checkpoint,
        device=args.device,
    )

    count = report.event_count
    print("")
    print("=== RESULT (calibrated / auto-derived direction) ===")
    print("wrong_way events observed: " + str(count))
    print("")
    print("manifest expectation     : wrong_way > 0 (sign only; count NOT established)")
    if count > 0:
        print("VERDICT: PASS -- the sign matches. The count above is a measurement of this")
        print("run at this stride, not a target, and must not be quoted as a benchmark.")
        return 0
    print("VERDICT: NOT SATISFIED at this sampling -- zero events observed.")
    print("This is reported as-is. Do not tune thresholds to manufacture the expected sign.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
