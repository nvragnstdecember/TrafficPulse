"""Typed per-camera scene configuration contract + deterministic hashing (U5).

This is the typed (pydantic) counterpart of the human-editable YAML scene
configuration under ``configs/scenes/``. It validates the *structure* of a
scene — types, enum membership, point-in-frame bounds, non-zero direction
vectors, a 3x3 homography shape, and referential integrity — and provides a
deterministic ``scene_config_hash`` that every ``ConfirmedEvent`` and
``EvidenceManifest`` embeds (their ``scene_config_hash`` field is a SHA-256 hex
string).

It implements **no behaviour**: no point-in-polygon, no line crossing, no
heading comparison, no homography maths, no speed estimation, no signal
classification, and no rule logic. Structural validation (bounds/reference
checks) is data validation, not scene processing.

Canonical serialization boundary for hashing
---------------------------------------------
The hash is computed over the model's JSON-mode dump serialized with
``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)``
and UTF-8 encoded. Because the pydantic model normalizes structure (field
order, enum values, int->float, datetimes -> ISO strings) and ``sort_keys``
normalizes mapping key order, two semantically identical configurations hash
identically regardless of YAML formatting or key ordering.
"""

import hashlib
import json
from enum import StrEnum
from typing import Self

from pydantic import (
    AwareDatetime,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from .enums import SignalState, SpeedUnit, ViolationType
from .observations import OBSERVATION_VARIANTS
from .primitives import ContractModel, NonEmptyStr, NonNegativeFloat

# --- structural type aliases -------------------------------------------------
Point = tuple[float, float]
Row3 = tuple[float, float, float]
Matrix3x3 = tuple[Row3, Row3, Row3]

# Live observation discriminators (reused, never duplicated).
_OBSERVATION_TYPES = frozenset(v.model_fields["obs_type"].default for v in OBSERVATION_VARIANTS)


# --- scene-specific closed vocabularies --------------------------------------
class SceneStatus(StrEnum):
    DRAFT = "draft"
    CALIBRATION_PENDING = "calibration_pending"
    VALIDATION_PENDING = "validation_pending"
    VALIDATED = "validated"
    ARCHIVED = "archived"


class CalibrationStatus(StrEnum):
    ABSENT = "absent"
    PROVISIONAL = "provisional"
    UNVERIFIED = "unverified"
    VALIDATED = "validated"
    REJECTED = "rejected"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    PROJECT_VERIFIED = "project_verified"
    EXTERNALLY_VERIFIED = "externally_verified"


class ParameterStatus(StrEnum):
    UNSET = "unset"
    PROVISIONAL = "provisional"
    TUNED = "tuned"
    VALIDATED = "validated"


class ZoneType(StrEnum):
    LANE = "lane"
    APPROACH = "approach"
    EXIT = "exit"
    INTERSECTION = "intersection"
    NO_STOPPING = "no_stopping"
    SPEED_MEASUREMENT = "speed_measurement"
    SIGNAL_CONTROLLED_REGION = "signal_controlled_region"
    ROI = "roi"


class SignalSourceMode(StrEnum):
    ROI_CLASSIFIER = "roi_classifier"
    SIMULATED_SCHEDULE = "simulated_schedule"
    MANUAL_ANNOTATION = "manual_annotation"


class RoiShape(StrEnum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


class CalibrationType(StrEnum):
    HOMOGRAPHY = "homography"
    NONE = "none"
    OTHER = "other"


class WorldUnit(StrEnum):
    METERS = "meters"


class CoordinateSpace(StrEnum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"


class OriginConvention(StrEnum):
    TOP_LEFT = "top_left"


class XAxisDirection(StrEnum):
    RIGHT = "right"


class YAxisDirection(StrEnum):
    DOWN = "down"


class PolygonOrdering(StrEnum):
    ORDERED_RING = "ordered_ring"


class ParameterUnit(StrEnum):
    FRAMES = "frames"
    SECONDS = "seconds"
    DEGREES = "degrees"
    M_PER_S = "m_per_s"
    KM_PER_H = "km_per_h"
    COUNT = "count"
    RATIO = "ratio"
    STATUS_REF = "status_ref"


# --- value objects -----------------------------------------------------------
class DirectionVector(ContractModel):
    """A 2D direction; validated non-zero, not normalized here."""

    dx: float
    dy: float

    @model_validator(mode="after")
    def _non_zero(self) -> Self:
        if self.dx == 0.0 and self.dy == 0.0:
            raise ValueError("direction vector must be non-zero")
        return self


class SceneProvenance(ContractModel):
    origin: NonEmptyStr
    purpose: NonEmptyStr
    synthetic: bool
    author_role: NonEmptyStr | None = None
    source_reference: NonEmptyStr | None = None
    notes: str | None = None


class SceneIdentity(ContractModel):
    scene_id: NonEmptyStr
    scene_name: NonEmptyStr
    config_version: NonEmptyStr
    schema_version: NonEmptyStr
    status: SceneStatus
    camera_id: NonEmptyStr
    site_id: NonEmptyStr
    description: str
    created_at: AwareDatetime
    updated_at: AwareDatetime
    provenance: SceneProvenance


class FrameSpec(ContractModel):
    reference_width: PositiveInt
    reference_height: PositiveInt
    coordinate_space: CoordinateSpace
    origin: OriginConvention
    x_axis_direction: XAxisDirection
    y_axis_direction: YAxisDirection
    polygon_point_ordering: PolygonOrdering


class Zone(ContractModel):
    zone_id: NonEmptyStr
    zone_type: ZoneType
    enabled: bool
    description: str | None = None
    polygon: tuple[Point, ...]
    legal_direction_id: NonEmptyStr | None = None
    signal_group_id: NonEmptyStr | None = None
    applicable_violations: tuple[ViolationType, ...] = ()
    observation_consumers: tuple[str, ...] = ()

    @field_validator("polygon")
    @classmethod
    def _min_points(cls, value: tuple[Point, ...]) -> tuple[Point, ...]:
        if len(value) < 3:
            raise ValueError("polygon requires at least 3 points")
        return value

    @field_validator("observation_consumers")
    @classmethod
    def _known_observations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        invalid = set(value) - _OBSERVATION_TYPES
        if invalid:
            raise ValueError(f"unknown observation types: {sorted(invalid)}")
        return value


class StopLineEndpoints(ContractModel):
    a: Point
    b: Point


class StopLine(ContractModel):
    stop_line_id: NonEmptyStr
    enabled: bool
    description: str | None = None
    endpoints: StopLineEndpoints
    crossing_direction: DirectionVector
    signal_group_id: NonEmptyStr
    zone_ids: tuple[NonEmptyStr, ...] = ()


class LegalDirection(ContractModel):
    direction_id: NonEmptyStr
    description: str
    vector: DirectionVector
    zone_ids: tuple[NonEmptyStr, ...] = ()
    tolerance_degrees: float | None = None
    tolerance_status: ParameterStatus | None = None


class Roi(ContractModel):
    shape: RoiShape
    x: NonNegativeFloat | None = None
    y: NonNegativeFloat | None = None
    width: PositiveFloat | None = None
    height: PositiveFloat | None = None
    polygon: tuple[Point, ...] | None = None

    @model_validator(mode="after")
    def _shape_fields(self) -> Self:
        if self.shape is RoiShape.RECTANGLE:
            if None in (self.x, self.y, self.width, self.height):
                raise ValueError("rectangle ROI requires x, y, width, height")
        elif self.polygon is None or len(self.polygon) < 3:
            raise ValueError("polygon ROI requires at least 3 points")
        return self


class SignalGroup(ContractModel):
    signal_group_id: NonEmptyStr
    enabled: bool
    description: str | None = None
    signal_source_mode: SignalSourceMode
    roi: Roi
    stop_line_ids: tuple[NonEmptyStr, ...] = ()
    zone_ids: tuple[NonEmptyStr, ...] = ()
    permitted_movements: tuple[str, ...] = ()
    expected_states: tuple[SignalState, ...] = ()


class SpeedLimit(ContractModel):
    speed_limit_id: NonEmptyStr
    value: NonNegativeFloat
    unit: SpeedUnit
    zone_ids: tuple[NonEmptyStr, ...] = ()
    source: NonEmptyStr
    verification_status: VerificationStatus


class Correspondence(ContractModel):
    image_xy: Point
    world_xy: Point
    description: str | None = None


class QualityMetrics(ContractModel):
    reprojection_rmse_px: float | None = None
    status: ParameterStatus


class Calibration(ContractModel):
    calibration_id: NonEmptyStr
    type: CalibrationType
    status: CalibrationStatus
    verification_status: VerificationStatus
    source: NonEmptyStr
    created_at: AwareDatetime
    world_unit: WorldUnit
    homography_matrix: Matrix3x3 | None = None
    correspondences: tuple[Correspondence, ...] = ()
    quality_metrics: QualityMetrics
    notes: str | None = None

    @model_validator(mode="after")
    def _matrix_present_for_homography(self) -> Self:
        if self.type is CalibrationType.HOMOGRAPHY and self.homography_matrix is None:
            raise ValueError("homography calibration requires a homography_matrix")
        return self


class RuleParameter(ContractModel):
    id: NonEmptyStr
    value: float | None = None
    unit: ParameterUnit
    status: ParameterStatus
    note: str | None = None


class RuleParameterBlock(ContractModel):
    violation_type: ViolationType
    parameters: tuple[RuleParameter, ...] = ()


def _unresolved(refs: tuple[str, ...], valid: set[str], label: str) -> list[str]:
    return [f"{label}:{ref}" for ref in refs if ref not in valid]


class SceneConfig(ContractModel):
    """A validated, hashable per-camera scene configuration (declarative data)."""

    scene: SceneIdentity
    frame: FrameSpec
    zones: tuple[Zone, ...]
    stop_lines: tuple[StopLine, ...] = ()
    legal_directions: tuple[LegalDirection, ...] = ()
    signal_groups: tuple[SignalGroup, ...] = ()
    speed_limits: tuple[SpeedLimit, ...] = ()
    calibration: Calibration
    rule_parameters: tuple[RuleParameterBlock, ...] = ()

    @model_validator(mode="after")
    def _structural_invariants(self) -> Self:
        self._check_unique_ids()
        self._check_points_in_bounds()
        self._check_references_resolve()
        return self

    def _check_unique_ids(self) -> None:
        if not self.zones:
            raise ValueError("scene requires at least one zone")
        categories: dict[str, list[str]] = {
            "zone": [z.zone_id for z in self.zones],
            "stop_line": [s.stop_line_id for s in self.stop_lines],
            "legal_direction": [d.direction_id for d in self.legal_directions],
            "signal_group": [g.signal_group_id for g in self.signal_groups],
            "speed_limit": [s.speed_limit_id for s in self.speed_limits],
            "rule_block": [str(b.violation_type) for b in self.rule_parameters],
        }
        for label, ids in categories.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} ids")

    def _image_points(self) -> list[Point]:
        points: list[Point] = []
        for zone in self.zones:
            points.extend(zone.polygon)
        for sl in self.stop_lines:
            points.append(sl.endpoints.a)
            points.append(sl.endpoints.b)
        for sg in self.signal_groups:
            roi = sg.roi
            if (
                roi.shape is RoiShape.RECTANGLE
                and roi.x is not None
                and roi.y is not None
                and roi.width is not None
                and roi.height is not None
            ):
                points.append((roi.x, roi.y))
                points.append((roi.x + roi.width, roi.y + roi.height))
            elif roi.polygon is not None:
                points.extend(roi.polygon)
        for corr in self.calibration.correspondences:
            points.append(corr.image_xy)
        return points

    def _check_points_in_bounds(self) -> None:
        width = self.frame.reference_width
        height = self.frame.reference_height
        out_of_bounds = [
            pt for pt in self._image_points() if not (0 <= pt[0] <= width and 0 <= pt[1] <= height)
        ]
        if out_of_bounds:
            raise ValueError(f"image points out of frame bounds: {out_of_bounds[:3]}")

    def _check_references_resolve(self) -> None:
        zone_ids = {z.zone_id for z in self.zones}
        sg_ids = {g.signal_group_id for g in self.signal_groups}
        sl_ids = {s.stop_line_id for s in self.stop_lines}
        dir_ids = {d.direction_id for d in self.legal_directions}
        missing: list[str] = []
        for sl in self.stop_lines:
            if sl.signal_group_id not in sg_ids:
                missing.append(f"stop_line->signal_group:{sl.signal_group_id}")
            missing += _unresolved(sl.zone_ids, zone_ids, "stop_line->zone")
        for sg in self.signal_groups:
            missing += _unresolved(sg.stop_line_ids, sl_ids, "signal_group->stop_line")
            missing += _unresolved(sg.zone_ids, zone_ids, "signal_group->zone")
        for direction in self.legal_directions:
            missing += _unresolved(direction.zone_ids, zone_ids, "legal_direction->zone")
        for sp in self.speed_limits:
            missing += _unresolved(sp.zone_ids, zone_ids, "speed_limit->zone")
        for zone in self.zones:
            if zone.legal_direction_id is not None and zone.legal_direction_id not in dir_ids:
                missing.append(f"zone->legal_direction:{zone.legal_direction_id}")
            if zone.signal_group_id is not None and zone.signal_group_id not in sg_ids:
                missing.append(f"zone->signal_group:{zone.signal_group_id}")
        if missing:
            raise ValueError(f"unresolved references: {missing[:5]}")


# --- deterministic hashing ---------------------------------------------------
def canonical_scene_bytes(scene: SceneConfig) -> bytes:
    """Canonical UTF-8 byte serialization used for hashing (see module docstring)."""

    payload = scene.model_dump(mode="json")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def scene_config_hash(scene: SceneConfig) -> str:
    """Return the deterministic SHA-256 hex digest of a validated scene config.

    The 64-character lowercase hex output is compatible with the ``Sha256Hex``
    ``scene_config_hash`` field on ``ConfirmedEvent`` and ``EvidenceManifest``.
    """

    return hashlib.sha256(canonical_scene_bytes(scene)).hexdigest()
