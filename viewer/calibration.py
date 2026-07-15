"""Upload scene calibration for the TrafficPulse Viewer (demonstration layer).

Why this exists
---------------
The backend runners (``run_wrong_way_slice``) already accept **any** validated
``SceneConfig`` — but the viewer's upload path used to reason every uploaded clip
against the repository's *synthetic* example scene (a 1920x1080 frame whose legal
direction is straight "up"). Real CCTV footage has its own frame size and its own
road orientation, so reasoning against that unrelated scene structurally produced
zero confirmed violations. This module supplies the missing piece: a **per-clip
calibrated SceneConfig**, derived from the clip itself, fed into the *unchanged*
backend.

What it does (and does not) do
------------------------------
Given one uploaded clip, :func:`calibrate_and_capture` makes a **single real
RT-DETR inference pass** (genuine per-frame detections through the existing
``DetectionAdapter`` + ``IouTracker`` seams) and derives:

* the clip's real frame dimensions (from the P1-U5 ingestion metadata);
* the **observed dominant traffic-flow direction** — the vector sum of the net
  displacement of every substantial track (alive >= 1 s, net motion >= 40 px).

:func:`build_calibrated_scene` then constructs a validated ``SceneConfig`` in the
clip's own pixel space whose single legal direction **is** that observed flow.
Wrong-way semantics under this calibration are exactly the road-safety notion:
*a vehicle sustainedly opposing the dominant traffic stream*. On footage where
every vehicle travels with the flow the honest result remains zero events; a
genuine against-traffic vehicle contradicts the calibrated legal direction and
confirms through the unchanged reasoner.

This module implements **no** detection, tracking, observation, rule, event, or
persistence logic. It composes existing seams (the same ones the shipped CLI
composition roots use) and authors *declarative scene data*. The reasoning engine
is untouched.

Honesty of the two-pass design
------------------------------
The slice pass needs the scene up front (the pipeline is constructed with it), so
the flow must be measured *before* reasoning. Rather than paying a second
multi-minute CPU inference pass, the calibration pass **records** the real
RT-DETR output per frame, and :class:`RTDetrCapturedReplay` re-emits those exact
recorded detections to the unchanged ``run_wrong_way_slice``. Nothing is authored
by hand: every replayed ``RawDetection`` came out of the genuine RT-DETR forward
pass on the uploaded pixels, and ``Detection.source_model`` still stamps the real
checkpoint's ``ModelRef``. The report's ``detector_kind`` truthfully names the
replay class so a recorded-replay run can never be mistaken for a stub script.

Provisional status (stated, not hidden)
---------------------------------------
The produced scene is marked ``draft`` / ``provenance.origin=auto_calibration``:
the legal direction is *observed*, not operator-verified, and no metric (world)
calibration is claimed — ``calibration.type=none``. That is sufficient for the
wrong-way slice, which reasons purely on image-space headings.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # standalone import convenience
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trafficpulse.contracts import ModelRef, ObjectClass, SceneConfig  # noqa: E402
from trafficpulse.detector import RawDetection, StubDetector  # noqa: E402
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.interface import Detector  # noqa: E402
from trafficpulse.ingestion.video import open_video  # noqa: E402
from trafficpulse.pipeline.base import frame_record_to_frame  # noqa: E402
from trafficpulse.tracking.iou_tracker import IouTracker  # noqa: E402

# The single legal direction / lane the calibrated scene declares. The wrong-way
# runner is invoked with this direction id explicitly.
OBSERVED_DIRECTION_ID = "dir-observed"
OBSERVED_LANE_ZONE_ID = "zone-lane-observed"

# A track participates in the dominant-flow estimate only if it is a *substantial
# mover*: alive at least this long and displaced at least this far. This excludes
# detector jitter on parked/stationary objects from the flow estimate.
MIN_TRACK_LIFETIME_SECONDS = 1.0
MIN_NET_DISPLACEMENT_PX = 40.0

# Fixed timestamp stamped on generated scenes so the scene hash — and therefore
# the content-derived event ids — stay deterministic across repeated runs of the
# same clip (no wall-clock in the decision path, matching the backend's rule).
_SCENE_TIMESTAMP = "2026-07-14T00:00:00Z"


class UploadCalibrationError(Exception):
    """The uploaded clip does not support automatic scene calibration."""


@dataclass(frozen=True)
class CalibrationResult:
    """What one real-inference calibration pass observed about a clip.

    ``per_frame_raw`` holds the *recorded* RT-DETR output for every decoded frame
    (keyed by ``frame_index``) so the slice pass can replay genuine detections
    instead of paying a second inference pass.
    """

    camera_id: str
    width: int
    height: int
    flow_dx: float
    flow_dy: float
    flow_heading_degrees: float
    mover_count: int
    track_count: int
    frames_seen: int
    per_frame_raw: dict[int, tuple[RawDetection, ...]]


class RTDetrCapturedReplay(StubDetector):
    """Replays the calibration pass's **recorded real RT-DETR** output verbatim.

    A named subclass (rather than a bare ``StubDetector``) so the slice report's
    ``detector_kind`` states truthfully what ran: recorded RT-DETR inference
    replayed frame-by-frame — not a caller-authored script, and not a second
    inference pass.
    """


def upload_camera_id(clip_path: Path | str) -> str:
    """A stable, opaque per-clip camera id (deterministic across re-runs)."""

    digest = hashlib.sha256(Path(clip_path).name.encode("utf-8")).hexdigest()[:8]
    return f"cam-upload-{digest}"


def calibrate_and_capture(
    *,
    clip: Path | str,
    detector: Detector,
    detector_config: DetectorConfig,
    camera_id: str | None = None,
) -> CalibrationResult:
    """Run one real inference pass; derive the dominant flow and record detections.

    Uses only existing seams: P1-U5 ingestion, the injected real ``detector``, the
    shared ``DetectionAdapter`` (same ``detector_config`` the slice will use), and
    a throwaway ``IouTracker`` for flow estimation. The recorded raw detections
    are the detector's verbatim output (pre-adapter), so the slice pass replays
    exactly what RT-DETR emitted.

    Raises:
        VideoIngestionError: the clip cannot be decoded (propagated from ingestion).
        UploadCalibrationError: no substantial vehicle motion was observed, so no
            legal direction can be derived (surfaced honestly to the viewer).
    """

    cam = camera_id or upload_camera_id(clip)
    adapter = DetectionAdapter(detector_config)
    tracker = IouTracker()

    per_frame_raw: dict[int, tuple[RawDetection, ...]] = {}
    centers: dict[str, list[tuple[float, float, float]]] = {}  # track -> (t, cx, cy)
    frames_seen = 0
    with open_video(clip, camera_id=cam) as reader:
        for frame_record in reader:
            frame = frame_record_to_frame(frame_record, camera_id=frame_record.camera_id or cam)
            raws = tuple(detector.detect(frame))
            per_frame_raw[frame_record.frame_index] = raws
            frames_seen += 1
            for state in tracker.update(adapter.adapt(frame, raws)):
                box = state.bbox
                centers.setdefault(state.track_id, []).append(
                    (
                        frame_record.timestamp_seconds,
                        (box.x1 + box.x2) / 2.0,
                        (box.y1 + box.y2) / 2.0,
                    )
                )
        metadata = reader.metadata

    sum_dx = sum_dy = 0.0
    movers = 0
    for points in centers.values():
        if len(points) < 2:
            continue
        lifetime = points[-1][0] - points[0][0]
        dx = points[-1][1] - points[0][1]
        dy = points[-1][2] - points[0][2]
        if lifetime >= MIN_TRACK_LIFETIME_SECONDS and math.hypot(dx, dy) >= MIN_NET_DISPLACEMENT_PX:
            movers += 1
            sum_dx += dx
            sum_dy += dy

    magnitude = math.hypot(sum_dx, sum_dy)
    if movers == 0 or magnitude <= 0.0:
        raise UploadCalibrationError(
            "scene calibration failed: no sustained vehicle motion observed "
            f"(need a track alive >= {MIN_TRACK_LIFETIME_SECONDS:.1f}s moving >= "
            f"{MIN_NET_DISPLACEMENT_PX:.0f}px) — cannot derive a legal traffic "
            "direction for wrong-way reasoning on this clip"
        )

    # Round the unit vector so the scene content (and its hash) is stable against
    # float-noise across platforms while remaining accurate to ~0.06 degrees.
    flow_dx = round(sum_dx / magnitude, 4)
    flow_dy = round(sum_dy / magnitude, 4)
    heading = math.degrees(math.atan2(flow_dy, flow_dx)) % 360.0

    return CalibrationResult(
        camera_id=cam,
        width=metadata.width,
        height=metadata.height,
        flow_dx=flow_dx,
        flow_dy=flow_dy,
        flow_heading_degrees=heading,
        mover_count=movers,
        track_count=len(centers),
        frames_seen=frames_seen,
        per_frame_raw=per_frame_raw,
    )


def build_calibrated_scene(
    calibration: CalibrationResult, *, clip_label: str = "uploaded clip"
) -> SceneConfig:
    """Author a validated ``SceneConfig`` for the clip from its calibration.

    Declarative data only, in the clip's **own pixel space**: one full-frame lane
    zone, one legal direction equal to the observed dominant flow, and the
    project's provisional wrong-way rule parameters (the same values the example
    scene carries — 120 deg deviation, 1.0 s persistence). ``calibration.type`` is
    ``none`` because no metric (world) calibration is claimed; the wrong-way slice
    reasons purely on image-space headings.
    """

    width, height = calibration.width, calibration.height
    raw = {
        "scene": {
            "scene_id": f"scene-upload-{calibration.camera_id.removeprefix('cam-upload-')}",
            "scene_name": "Uploaded CCTV (auto-calibrated)",
            "config_version": "0.1.0-autocal",
            "schema_version": "1.0.0",
            "status": "draft",  # observed, not operator-validated
            "camera_id": calibration.camera_id,
            "site_id": "site-upload-01",
            "description": (
                f"Auto-calibrated scene for {clip_label}: frame {width}x{height}; "
                f"legal direction = observed dominant traffic flow "
                f"(heading {calibration.flow_heading_degrees:.1f} deg from "
                f"{calibration.mover_count} substantial tracks). Provisional; "
                "no operator verification and no metric calibration claimed."
            ),
            "created_at": _SCENE_TIMESTAMP,
            "updated_at": _SCENE_TIMESTAMP,
            "provenance": {
                "origin": "auto_calibration",
                "purpose": "uploaded_clip_wrong_way_analysis",
                "synthetic": False,
                "author_role": "viewer_auto_calibrator",
                "source_reference": "observed-dominant-flow",
                "notes": (
                    "Legal direction derived from the clip's own observed dominant "
                    "flow via one real RT-DETR + IoU-tracker pass; not an "
                    "operator-verified deployment calibration."
                ),
            },
        },
        "frame": {
            "reference_width": width,
            "reference_height": height,
            "coordinate_space": "pixel",
            "origin": "top_left",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
            "polygon_point_ordering": "ordered_ring",
        },
        "zones": [
            {
                "zone_id": OBSERVED_LANE_ZONE_ID,
                "zone_type": "lane",
                "enabled": True,
                "description": (
                    "Whole-frame monitored roadway (coarse). The wrong-way slice "
                    "derives headings from full tracks; no tighter lane geometry "
                    "is required or claimed."
                ),
                "polygon": [
                    [0.0, 0.0],
                    [float(width), 0.0],
                    [float(width), float(height)],
                    [0.0, float(height)],
                ],
                "legal_direction_id": OBSERVED_DIRECTION_ID,
                "signal_group_id": None,
                "applicable_violations": ["wrong_way"],
                "observation_consumers": ["heading_vs_lane", "in_zone"],
            }
        ],
        "legal_directions": [
            {
                "direction_id": OBSERVED_DIRECTION_ID,
                "description": "Legal travel = observed dominant traffic flow of this clip.",
                "vector": {"dx": calibration.flow_dx, "dy": calibration.flow_dy},
                "zone_ids": [OBSERVED_LANE_ZONE_ID],
                "tolerance_degrees": None,
                "tolerance_status": "unset",
            }
        ],
        "calibration": {
            "calibration_id": "cal-none-upload",
            "type": "none",  # no metric/world calibration is claimed
            "status": "absent",
            "verification_status": "unverified",
            "source": "auto_calibration",
            "created_at": _SCENE_TIMESTAMP,
            "world_unit": "meters",
            "quality_metrics": {"reprojection_rmse_px": None, "status": "unset"},
            "notes": (
                "No homography/world calibration; the wrong-way slice reasons on "
                "image-space headings only."
            ),
        },
        "rule_parameters": [
            {
                "violation_type": "wrong_way",
                "parameters": [
                    {
                        "id": "heading_deviation_max",
                        "value": 120.0,
                        "unit": "degrees",
                        "status": "provisional",
                        "note": (
                            "Provisional (architecture-review ~120 deg); "
                            "same as example scene."
                        ),
                    },
                    {
                        "id": "min_persistence",
                        "value": 1.0,
                        "unit": "seconds",
                        "status": "provisional",
                        "note": "Provisional (architecture-review ~1.0 s); same as example scene.",
                    },
                    {
                        "id": "min_speed",
                        "value": 1.5,
                        "unit": "m_per_s",
                        "status": "provisional",
                        "note": (
                            "Carried for provenance; not applied (no metric "
                            "calibration exists for this scene)."
                        ),
                    },
                ],
            }
        ],
    }
    return SceneConfig.model_validate(raw)


def default_upload_detector_config(checkpoint_model_ref: ModelRef) -> DetectorConfig:
    """The adapter config the upload path uses (car >= 0.5, real provenance stamp)."""

    return DetectorConfig(
        label_map={"car": ObjectClass.CAR},
        score_threshold=0.5,
        source_model=checkpoint_model_ref,
    )
