"""Authoring a validated ``SceneConfig`` from what an analyst drew (H12).

The bridge between a calibration surface and the frozen scene contract.
:class:`SceneDraft` is the **minimal analyst-authorable** description of a site --
a frame size, some polygons, a direction arrow, a couple of thresholds -- and
:func:`build_scene` expands it into a complete, validated
:class:`~trafficpulse.contracts.SceneConfig`.

Why a draft rather than authoring ``SceneConfig`` directly
-----------------------------------------------------------
``SceneConfig`` is a ~40-model structure carrying provenance, five status
vocabularies, schema versions, coordinate-space declarations, and a calibration
block. Almost none of that is a drawing decision: it is bookkeeping that must be
*correct*, not *chosen*. Asking a client to submit it whole would mean trusting a
browser to author its own provenance and status -- which is exactly how a scene
ends up claiming ``status: validated`` because a form defaulted that way.

So the draft is an **input vocabulary, not a parallel scene model**. Nothing
stores or returns a draft in place of a scene: the repository stores
``SceneConfig`` verbatim, the API returns ``SceneConfig`` verbatim, and this
module is the only thing that turns one into the other. It is the same
relationship ``ProcessRequest`` has to ``EngineConfig``.

Deterministic by construction
------------------------------
:func:`build_scene` reads **no wall-clock and no randomness**: the same draft
always produces byte-identical scene content, and therefore the same
``scene_config_hash``. That is what lets the scene store be content-addressed and
write-once (re-saving an unchanged drawing is a no-op rather than a new
revision), and it keeps scene identity out of the class of facts that drift.

``created_at`` / ``updated_at`` are stamped with the Unix epoch -- the "no
information" anchor this repository already uses for an untouched
:class:`~trafficpulse.contracts.ReviewCase`. A scene's *content* records no
creation instant because baking one in would make identical drawings hash
differently; when a scene was actually stored is recorded by the repository, not
by the hashed data.

Only what is drawn is claimed
------------------------------
Every scene this module authors declares ``calibration.type = none`` and
``status = draft``. No metric (world) calibration is performed, so none is
claimed; the analyst drew image-space geometry, which is exactly what wrong-way
and illegal-stopping reason over. Rule parameters are stamped
``ParameterStatus.PROVISIONAL`` because they are operator-chosen defaults, not
values validated against ground truth.

H13 readiness
-------------
Stop lines and signal groups are authorable here (:class:`StopLineDraft`,
:class:`SignalGroupDraft`), and junction geometry is just a zone with
``ZoneType.INTERSECTION`` or ``SIGNAL_CONTROLLED_REGION``. ``SceneConfig``
already models and cross-validates all three, so red-light jumping needs a
drawing tool and a reasoner -- not a scene redesign.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from ..contracts import SceneConfig
from ..contracts.enums import ViolationType
from ..contracts.scene import (
    Calibration,
    CalibrationStatus,
    CalibrationType,
    CoordinateSpace,
    DirectionVector,
    FrameSpec,
    LegalDirection,
    OriginConvention,
    ParameterStatus,
    ParameterUnit,
    PolygonOrdering,
    QualityMetrics,
    Roi,
    RoiShape,
    RuleParameter,
    RuleParameterBlock,
    SceneIdentity,
    SceneProvenance,
    SceneStatus,
    SignalGroup,
    SignalSourceMode,
    StopLine,
    StopLineEndpoints,
    VerificationStatus,
    WorldUnit,
    XAxisDirection,
    YAxisDirection,
    Zone,
    ZoneType,
)

Point = tuple[float, float]

#: The "no information" instant, matching the convention an untouched
#: ``ReviewCase`` already uses. See the module docstring for why scene content
#: carries no real creation time.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_SCHEMA_VERSION = "1.0.0"
_CONFIG_VERSION = "0.1.0-authored"

# Provisional defaults, mirroring the committed example scene verbatim so an
# authored scene reasons on exactly the policy the tests and demos use. Every one
# is stamped PROVISIONAL: they are architecture-review starting points, not values
# validated against ground truth.
DEFAULT_HEADING_DEVIATION_MAX = 120.0
DEFAULT_WRONG_WAY_PERSISTENCE = 1.0
DEFAULT_STATIONARY_DURATION = 5.0
# A debounce, not a dwell: red-light is committed at the stop line, so this only has
# to outlast boundary jitter. Kept short deliberately -- a long window would start
# excusing vehicles that clear the junction quickly, which is the opposite of intent.
DEFAULT_RED_LIGHT_PERSISTENCE = 0.4
DEFAULT_NO_HELMET_PERSISTENCE = 1.0
DEFAULT_NO_HELMET_MAX_GAP = 2.0
DEFAULT_TRIPLE_RIDING_PERSISTENCE = 1.0
DEFAULT_TRIPLE_RIDING_MAX_GAP = 2.0
DEFAULT_RIDER_COUNT_THRESHOLD = 3.0


class _DraftModel(BaseModel):
    """Frozen + strict base for the authoring vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ZoneDraft(_DraftModel):
    """One polygon an analyst drew, and what it means.

    ``zone_type`` is the full scene vocabulary rather than a reduced one, so a
    junction or signal-controlled region is authorable today even though nothing
    reasons over it yet (H13).
    """

    zone_id: str = Field(min_length=1, max_length=128)
    zone_type: ZoneType
    polygon: tuple[Point, ...]
    description: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _enough_points(self) -> Self:
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.zone_id!r} needs at least 3 points")
        return self


class DirectionDraft(_DraftModel):
    """The legal travel direction, and the zone it governs.

    ``dx``/``dy`` need not be normalised -- an analyst drags an arrow of arbitrary
    length and the contract only requires a non-zero vector. ``zone_id`` must name
    a zone in the same draft; wrong-way reasoning needs a lane to attach to.
    """

    direction_id: str = Field(default="dir-legal", min_length=1, max_length=128)
    dx: float
    dy: float
    zone_id: str = Field(min_length=1, max_length=128)
    description: str = Field(default="Legal travel direction", max_length=500)

    @model_validator(mode="after")
    def _non_zero(self) -> Self:
        if self.dx == 0.0 and self.dy == 0.0:
            raise ValueError("direction vector must be non-zero")
        return self


class StopLineDraft(_DraftModel):
    """A stop line and the direction crossing it counts as entering (H13-ready)."""

    stop_line_id: str = Field(min_length=1, max_length=128)
    a: Point
    b: Point
    crossing_dx: float
    crossing_dy: float
    signal_group_id: str = Field(min_length=1, max_length=128)
    zone_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _non_zero(self) -> Self:
        if self.crossing_dx == 0.0 and self.crossing_dy == 0.0:
            raise ValueError("crossing direction must be non-zero")
        return self


class SignalGroupDraft(_DraftModel):
    """A signal head's ROI and what it governs (H13-ready).

    ``signal_source_mode`` defaults to ``SIMULATED_SCHEDULE`` because that is the
    only mode with a shipped producer (``observations/signal.py``); nothing in the
    runtime classifies a signal from pixels yet.
    """

    signal_group_id: str = Field(min_length=1, max_length=128)
    roi_polygon: tuple[Point, ...]
    signal_source_mode: SignalSourceMode = SignalSourceMode.SIMULATED_SCHEDULE
    zone_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _enough_points(self) -> Self:
        if len(self.roi_polygon) < 3:
            raise ValueError(f"signal group {self.signal_group_id!r} ROI needs 3+ points")
        return self


class RuleTuning(_DraftModel):
    """The site-specific thresholds worth exposing to an analyst.

    Deliberately few. Most rule parameters are policy constants that should not
    vary per camera; these are the two that genuinely describe a *site* -- how far
    off the legal heading counts as opposing traffic, and how long a vehicle may
    dwell before stopping there is a violation. Omitted values take the
    provisional defaults.
    """

    heading_deviation_max_degrees: float | None = Field(default=None, gt=0.0, le=180.0)
    wrong_way_min_persistence_seconds: float | None = Field(default=None, gt=0.0, le=600.0)
    stationary_duration_seconds: float | None = Field(default=None, gt=0.0, le=3600.0)
    red_light_min_persistence_seconds: float | None = Field(default=None, gt=0.0, le=30.0)


class SceneDraft(_DraftModel):
    """Everything an analyst supplies to author one site's scene.

    The frame size anchors every coordinate: polygons are in the video's own pixel
    space, and ``SceneConfig`` rejects any point outside it, so a draft measured
    against a differently-sized video fails loudly instead of reasoning over
    geometry that lands off-frame.
    """

    scene_name: str = Field(min_length=1, max_length=200)
    camera_id: str = Field(min_length=1, max_length=128)
    site_id: str = Field(default="site-default", min_length=1, max_length=128)
    description: str = Field(default="", max_length=2000)
    frame_width: PositiveInt
    frame_height: PositiveInt
    zones: tuple[ZoneDraft, ...]
    direction: DirectionDraft | None = None
    stop_lines: tuple[StopLineDraft, ...] = ()
    signal_groups: tuple[SignalGroupDraft, ...] = ()
    tuning: RuleTuning = RuleTuning()

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if not self.zones:
            raise ValueError("a scene needs at least one zone")
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("duplicate zone ids in draft")
        if self.direction is not None and self.direction.zone_id not in set(zone_ids):
            raise ValueError(
                f"direction governs unknown zone {self.direction.zone_id!r}"
            )
        return self


def _provisional(
    identifier: str, value: float, unit: ParameterUnit, note: str
) -> RuleParameter:
    return RuleParameter(
        id=identifier,
        value=value,
        unit=unit,
        status=ParameterStatus.PROVISIONAL,
        note=note,
    )


def _rule_blocks(draft: SceneDraft) -> tuple[RuleParameterBlock, ...]:
    """The parameter blocks the drawn geometry can actually support.

    A block is emitted only when the scene has the geometry its rule needs -- a
    wrong-way block without a legal direction would advertise a rule that fails
    fast the moment it is configured. The two geometry-free motorcycle rules are
    always emitted: they reason over riders and time, not over the site.
    """

    tuning = draft.tuning
    blocks: list[RuleParameterBlock] = []

    if draft.direction is not None:
        blocks.append(
            RuleParameterBlock(
                violation_type=ViolationType.WRONG_WAY,
                parameters=(
                    _provisional(
                        "heading_deviation_max",
                        tuning.heading_deviation_max_degrees
                        or DEFAULT_HEADING_DEVIATION_MAX,
                        ParameterUnit.DEGREES,
                        "Operator-chosen; provisional (architecture-review ~120 deg).",
                    ),
                    _provisional(
                        "min_persistence",
                        tuning.wrong_way_min_persistence_seconds
                        or DEFAULT_WRONG_WAY_PERSISTENCE,
                        ParameterUnit.SECONDS,
                        "Operator-chosen; provisional (architecture-review ~1.0 s).",
                    ),
                ),
            )
        )

    if any(zone.zone_type is ZoneType.NO_STOPPING for zone in draft.zones):
        blocks.append(
            RuleParameterBlock(
                violation_type=ViolationType.ILLEGAL_STOPPING,
                parameters=(
                    _provisional(
                        "stationary_duration",
                        tuning.stationary_duration_seconds or DEFAULT_STATIONARY_DURATION,
                        ParameterUnit.SECONDS,
                        "Operator-chosen dwell threshold for this site; provisional.",
                    ),
                ),
            )
        )

    junction_types = (ZoneType.INTERSECTION, ZoneType.SIGNAL_CONTROLLED_REGION)
    if draft.stop_lines and any(zone.zone_type in junction_types for zone in draft.zones):
        blocks.append(
            RuleParameterBlock(
                violation_type=ViolationType.RED_LIGHT_JUMPING,
                parameters=(
                    _provisional(
                        "min_persistence",
                        tuning.red_light_min_persistence_seconds
                        or DEFAULT_RED_LIGHT_PERSISTENCE,
                        ParameterUnit.SECONDS,
                        "Debounce only -- the violation is complete at the stop line; "
                        "this window exists so a single frame of boundary jitter "
                        "cannot mint an event.",
                    ),
                ),
            )
        )

    blocks.append(
        RuleParameterBlock(
            violation_type=ViolationType.NO_HELMET,
            parameters=(
                _provisional(
                    "min_persistence",
                    DEFAULT_NO_HELMET_PERSISTENCE,
                    ParameterUnit.SECONDS,
                    "Mirrors the example scene's provisional value; not tuned.",
                ),
                _provisional(
                    "max_observation_gap",
                    DEFAULT_NO_HELMET_MAX_GAP,
                    ParameterUnit.SECONDS,
                    "Mirrors the example scene's provisional value; not tuned.",
                ),
            ),
        )
    )
    blocks.append(
        RuleParameterBlock(
            violation_type=ViolationType.TRIPLE_RIDING,
            parameters=(
                _provisional(
                    "min_persistence",
                    DEFAULT_TRIPLE_RIDING_PERSISTENCE,
                    ParameterUnit.SECONDS,
                    "Mirrors the example scene's provisional value; not tuned.",
                ),
                _provisional(
                    "max_observation_gap",
                    DEFAULT_TRIPLE_RIDING_MAX_GAP,
                    ParameterUnit.SECONDS,
                    "Mirrors the example scene's provisional value; not tuned.",
                ),
                _provisional(
                    "rider_count_threshold",
                    DEFAULT_RIDER_COUNT_THRESHOLD,
                    ParameterUnit.COUNT,
                    "Three or more riders on one motorcycle; statutory, not tuned.",
                ),
            ),
        )
    )
    return tuple(blocks)


def _observation_consumers(zone_type: ZoneType, *, governed: bool) -> tuple[str, ...]:
    """Which observation streams a zone feeds, from what it is and what governs it."""

    if zone_type is ZoneType.NO_STOPPING:
        return ("in_zone", "stationary")
    if governed:
        return ("in_zone", "heading_vs_lane")
    return ("in_zone",)


def _applicable(zone_type: ZoneType, *, governed: bool) -> tuple[ViolationType, ...]:
    if zone_type is ZoneType.NO_STOPPING:
        return (ViolationType.ILLEGAL_STOPPING,)
    if governed:
        return (ViolationType.WRONG_WAY,)
    return ()


def build_scene(draft: SceneDraft, *, scene_id: str) -> SceneConfig:
    """Expand a draft into a complete, validated ``SceneConfig``.

    ``scene_id`` is the scene's *logical* identity -- stable across edits, unlike
    the content hash that addresses one revision of it.

    Raises:
        pydantic.ValidationError: the drawn geometry cannot form a valid scene --
            a point outside the frame, an unresolved reference, a duplicate id.
            The contract's own structural validation is the authority here; this
            function adds no second rule set.
    """

    governed_zone = draft.direction.zone_id if draft.direction is not None else None
    zones = tuple(
        Zone(
            zone_id=zone.zone_id,
            zone_type=zone.zone_type,
            enabled=True,
            description=zone.description,
            polygon=zone.polygon,
            legal_direction_id=(
                draft.direction.direction_id
                if draft.direction is not None and zone.zone_id == governed_zone
                else None
            ),
            signal_group_id=next(
                (
                    group.signal_group_id
                    for group in draft.signal_groups
                    if zone.zone_id in group.zone_ids
                ),
                None,
            ),
            applicable_violations=_applicable(
                zone.zone_type, governed=zone.zone_id == governed_zone
            ),
            observation_consumers=_observation_consumers(
                zone.zone_type, governed=zone.zone_id == governed_zone
            ),
        )
        for zone in draft.zones
    )

    directions: tuple[LegalDirection, ...] = ()
    if draft.direction is not None:
        directions = (
            LegalDirection(
                direction_id=draft.direction.direction_id,
                description=draft.direction.description,
                vector=DirectionVector(dx=draft.direction.dx, dy=draft.direction.dy),
                zone_ids=(draft.direction.zone_id,),
                tolerance_degrees=None,
                tolerance_status=ParameterStatus.UNSET,
            ),
        )

    return SceneConfig(
        scene=SceneIdentity(
            scene_id=scene_id,
            scene_name=draft.scene_name,
            config_version=_CONFIG_VERSION,
            schema_version=_SCHEMA_VERSION,
            # Analyst-drawn geometry is not an operator-verified deployment
            # calibration; nothing here has been checked against ground truth.
            status=SceneStatus.DRAFT,
            camera_id=draft.camera_id,
            site_id=draft.site_id,
            description=draft.description,
            created_at=_EPOCH,
            updated_at=_EPOCH,
            provenance=SceneProvenance(
                origin="analyst_calibration",
                purpose="uploaded_clip_analysis",
                synthetic=False,
                author_role="analyst",
                source_reference=None,
                notes=(
                    "Image-space geometry drawn against the video's own frame. No "
                    "metric (world) calibration is performed or claimed."
                ),
            ),
        ),
        frame=FrameSpec(
            reference_width=draft.frame_width,
            reference_height=draft.frame_height,
            coordinate_space=CoordinateSpace.PIXEL,
            origin=OriginConvention.TOP_LEFT,
            x_axis_direction=XAxisDirection.RIGHT,
            y_axis_direction=YAxisDirection.DOWN,
            polygon_point_ordering=PolygonOrdering.ORDERED_RING,
        ),
        zones=zones,
        stop_lines=tuple(
            StopLine(
                stop_line_id=line.stop_line_id,
                enabled=True,
                endpoints=StopLineEndpoints(a=line.a, b=line.b),
                crossing_direction=DirectionVector(dx=line.crossing_dx, dy=line.crossing_dy),
                signal_group_id=line.signal_group_id,
                zone_ids=line.zone_ids,
            )
            for line in draft.stop_lines
        ),
        legal_directions=directions,
        signal_groups=tuple(
            SignalGroup(
                signal_group_id=group.signal_group_id,
                enabled=True,
                signal_source_mode=group.signal_source_mode,
                roi=Roi(shape=RoiShape.POLYGON, polygon=group.roi_polygon),
                stop_line_ids=tuple(
                    line.stop_line_id
                    for line in draft.stop_lines
                    if line.signal_group_id == group.signal_group_id
                ),
                zone_ids=group.zone_ids,
            )
            for group in draft.signal_groups
        ),
        calibration=Calibration(
            calibration_id="cal-image-space",
            # No homography was solved, so none is claimed. Wrong-way and
            # illegal-stopping reason purely in image space and need none.
            type=CalibrationType.NONE,
            status=CalibrationStatus.ABSENT,
            verification_status=VerificationStatus.UNVERIFIED,
            source="analyst_calibration",
            created_at=_EPOCH,
            world_unit=WorldUnit.METERS,
            quality_metrics=QualityMetrics(
                reprojection_rmse_px=None, status=ParameterStatus.UNSET
            ),
            notes="Image-space only; no world calibration solved for this scene.",
        ),
        rule_parameters=_rule_blocks(draft),
    )


def full_frame_polygon(width: int, height: int) -> tuple[Point, ...]:
    """The whole-frame ring, for a draft that monitors the entire view."""

    return (
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    )
