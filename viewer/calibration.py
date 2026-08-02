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

Where the algorithm lives now (H12)
------------------------------------
The dominant-flow estimator was **promoted into the runtime** as
:func:`trafficpulse.scenes.estimate_dominant_flow`, so the application's
calibration workflow and this demo layer share one implementation and one set of
thresholds. This module keeps what is genuinely viewer-specific: decoding the
clip, driving the detector, and recording the raw detections so the slice pass can
replay them instead of paying a second inference pass. It no longer owns the maths.

What it does (and does not) do
------------------------------
Given one uploaded clip, :func:`calibrate_and_capture` makes a **single real
RT-DETR inference pass** (genuine per-frame detections through the existing
``DetectionAdapter`` + ``IouTracker`` seams) and derives:

* the clip's real frame dimensions (from the P1-U5 ingestion metadata);
* the **observed dominant traffic-flow direction** — the vector sum of the net
  displacement of every substantial track (alive >= 1 s, net motion >= 40 px),
  computed by the promoted runtime estimator.

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
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # standalone import convenience
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trafficpulse.contracts import ModelRef, ObjectClass, SceneConfig, TrackState  # noqa: E402
from trafficpulse.detector import RawDetection, StubDetector  # noqa: E402
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.interface import Detector  # noqa: E402
from trafficpulse.ingestion.video import open_video  # noqa: E402
from trafficpulse.pipeline.base import frame_record_to_frame  # noqa: E402

# H12 promoted the dominant-flow estimator into the runtime
# (``trafficpulse.scenes.calibration``), where the application can use it too. The
# viewer now *consumes* it rather than owning a private copy: same algorithm, same
# thresholds, one implementation. The names are re-exported so this module's
# public surface -- and the demo scripts that read the thresholds -- are unchanged.
from trafficpulse.scenes import (  # noqa: E402
    FLOW_CLASSES as FLOW_CLASSES,  # re-exported: callers read it from this module
)
from trafficpulse.scenes import (  # noqa: E402
    MIN_NET_DISPLACEMENT_PX,
    MIN_TRACK_LIFETIME_SECONDS,
    estimate_dominant_flow,
)
from trafficpulse.tracking.iou_tracker import IouTracker  # noqa: E402

# The single legal direction / lane the calibrated scene declares. The wrong-way
# runner is invoked with this direction id explicitly.
OBSERVED_DIRECTION_ID = "dir-observed"
OBSERVED_LANE_ZONE_ID = "zone-lane-observed"

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
    observed: list[TrackState] = []
    frames_seen = 0
    with open_video(clip, camera_id=cam) as reader:
        for frame_record in reader:
            frame = frame_record_to_frame(frame_record, camera_id=frame_record.camera_id or cam)
            raws = tuple(detector.detect(frame))
            per_frame_raw[frame_record.frame_index] = raws
            frames_seen += 1
            observed.extend(tracker.update(adapter.adapt(frame, raws)))
        metadata = reader.metadata

    # The estimate itself is the runtime's (H12) -- this module contributes the I/O
    # and the recorded-detection replay, not the maths.
    flow = estimate_dominant_flow(observed)
    if flow is None:
        raise UploadCalibrationError(
            "scene calibration failed: no sustained vehicle motion observed "
            f"(need a track alive >= {MIN_TRACK_LIFETIME_SECONDS:.1f}s moving >= "
            f"{MIN_NET_DISPLACEMENT_PX:.0f}px) — cannot derive a legal traffic "
            "direction for wrong-way reasoning on this clip"
        )

    return CalibrationResult(
        camera_id=cam,
        width=metadata.width,
        height=metadata.height,
        flow_dx=flow.dx,
        flow_dy=flow.dy,
        flow_heading_degrees=flow.heading_degrees,
        mover_count=flow.mover_count,
        track_count=flow.track_count,
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


HELMET_ZONE_ID = "zone-roadway-observed"


def build_helmet_scene(
    *, width: int, height: int, camera_id: str, clip_label: str = "uploaded clip"
) -> SceneConfig:
    """Author a validated ``SceneConfig`` for running the no-helmet slice on a clip.

    Declarative data only, in the clip's **own pixel space**.

    Why this is not :func:`build_calibrated_scene`
    ----------------------------------------------
    Wrong-way reasoning is direction-gated, so its scene must be *calibrated*: an
    inference pass has to observe the dominant traffic flow before the scene can be
    authored, which is why the upload path pays two passes. No-helmet reasoning is
    **neither zone- nor direction-gated** -- it reasons over rider observations and
    time only. The single thing it needs from the clip is its frame size, which the
    P1-U5 ingestion metadata reports **without any inference**. So this scene is a
    pure function of the clip's dimensions, and the helmet upload path runs **one**
    inference pass rather than two.

    A full-frame zone is declared because ``SceneConfig`` structurally requires at
    least one; the no-helmet slice never reads it. No ``legal_direction`` is
    declared (nothing consumes one) and ``calibration.type`` is ``none`` because no
    metric calibration is claimed.

    The ``no_helmet`` rule parameters mirror the committed example scene's
    (provisional, untuned) values verbatim, so an uploaded clip is reasoned over on
    exactly the same policy the tests and the built-in demo use.
    """

    raw = {
        "scene": {
            "scene_id": f"scene-helmet-{camera_id.removeprefix('cam-upload-')}",
            "scene_name": "Uploaded CCTV (no-helmet slice)",
            "config_version": "0.1.0-autocal",
            "schema_version": "1.0.0",
            "status": "draft",  # derived, not operator-validated
            "camera_id": camera_id,
            "site_id": "site-upload-01",
            "description": (
                f"Scene for {clip_label}: frame {width}x{height}. No-helmet reasoning "
                "is neither zone- nor direction-gated, so only the frame size is "
                "derived from the clip; no traffic-flow calibration is performed or "
                "claimed."
            ),
            "created_at": _SCENE_TIMESTAMP,
            "updated_at": _SCENE_TIMESTAMP,
            "provenance": {
                "origin": "auto_calibration",
                "purpose": "uploaded_clip_no_helmet_analysis",
                "synthetic": False,
                "author_role": "viewer_auto_calibrator",
                "source_reference": "clip-frame-metadata",
                "notes": (
                    "Frame size read from the clip's ingestion metadata; no "
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
                "zone_id": HELMET_ZONE_ID,
                "zone_type": "roi",
                "enabled": True,
                "description": (
                    "Whole-frame monitored roadway. Declared because SceneConfig "
                    "requires at least one zone; the no-helmet slice does not read it "
                    "(helmet reasoning is not zone-gated)."
                ),
                "polygon": [
                    [0.0, 0.0],
                    [float(width), 0.0],
                    [float(width), float(height)],
                    [0.0, float(height)],
                ],
                "applicable_violations": ["no_helmet"],
                "observation_consumers": ["helmet_state"],
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
            "notes": "No homography/world calibration; the no-helmet slice needs none.",
        },
        "rule_parameters": [
            {
                "violation_type": "no_helmet",
                "parameters": [
                    {
                        "id": "min_persistence",
                        "value": 1.0,
                        "unit": "seconds",
                        "status": "provisional",
                        "note": "Mirrors the example scene's provisional value; not tuned.",
                    },
                    {
                        "id": "max_observation_gap",
                        "value": 2.0,
                        "unit": "seconds",
                        "status": "provisional",
                        "note": "Mirrors the example scene's provisional value; not tuned.",
                    },
                ],
            }
        ],
    }
    return SceneConfig.model_validate(raw)


def default_upload_detector_config(checkpoint_model_ref: ModelRef) -> DetectorConfig:
    """The adapter config the upload path uses (>= 0.5, real provenance stamp).

    Extended in P4-U1 (Gate 0) from ``car``-only to the classes Phase 4 needs.
    Validated by ``demo/gate0_rtdetr_validation.py`` on real Delhi traffic footage:
    RT-DETR detects all three classes reliably at score >= 0.5.

    Both motorcycle spellings are mapped deliberately. Checkpoints disagree on the
    native ``id2label`` vocabulary: ``PekingU/rtdetr_r50vd`` emits **"motorbike"**,
    other ports emit "motorcycle". An unmapped native label is silently dropped by
    the adapter (P1-U6 behaviour), so mapping only "motorcycle" against this
    checkpoint detects **zero** motorcycles with no error. Mapping both is safe: a
    spelling the checkpoint never emits simply never matches.
    """

    return DetectorConfig(
        label_map={
            "car": ObjectClass.CAR,
            "motorbike": ObjectClass.MOTORCYCLE,
            "motorcycle": ObjectClass.MOTORCYCLE,
            "person": ObjectClass.PERSON,
        },
        score_threshold=0.5,
        source_model=checkpoint_model_ref,
    )
