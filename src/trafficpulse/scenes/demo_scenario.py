"""The controlled demonstration scenario: one clip, four violation families.

A **hand-authored traffic situation** and the scene context that makes it
readable, in one place. It exists because the project's real-footage corpus
cannot demonstrate every violation at once -- no clip in it contains a red-light
runner, a wrong-way vehicle, an illegally stopped car and an overloaded
motorcycle -- and because the honest answer to that is a *declared* controlled
scenario, not a lowered threshold on real footage.

What this is, and is not
------------------------
It **is** the input side of a demonstration: where four vehicles are on each
frame, and the geometry, timing and thresholds an analyst would declare for that
site. Every one of those is context a camera genuinely cannot establish on its
own -- which way traffic legally travels, where stopping is prohibited, what the
signal was showing -- and supplying it is exactly what the calibration surface
is for.

It is **not** a result. Nothing here produces, names, or influences a
``ConfirmedEvent``. The clip is decoded, the detections are tracked, and the same
shipped reasoners decide what happened, with the same thresholds and the same
temporal guarantees they apply to real footage. If a reasoner declines to
confirm, the demonstration shows that; the refusal is the feature.

It is also **not** validation. The footage is synthetic rectangles, so a result
here establishes that the system *reasons correctly over declared context*, and
nothing whatever about detection accuracy on real pixels. Real-footage findings
live in ``docs/validation-matrix.md`` and are kept strictly separate.

The four scenarios, and why they cannot interfere
--------------------------------------------------
The whole point is that four *independent* reasoners each reach their own
conclusion from one pass, so the actors are placed to make cross-talk impossible
rather than merely unlikely:

* **Red-light jumping** -- a car descends the lane, crosses the stop line while
  the declared signal is ``red``, and enters the junction. The schedule turns
  ``green`` **after** it crossed, so the run also demonstrates the H13 latch: the
  signal is read once, at the crossing, and a later change cannot erase the act.
* **Wrong way** -- a car climbs the same lane, opposing the declared legal
  direction for 3.7 s against a 1.0 s threshold. Going *up*, its stop-line
  crossing is a **backward** one, which clears the crossing flag instead of
  setting it, so it can never register a junction entry and can never be
  confirmed for red-light.
* **Illegal stopping** -- a car pulls onto the right shoulder, inside the declared
  no-stopping zone, and holds. Its ground-contact point is outside the lane
  polygon, so no heading is derived for it and wrong-way cannot see it.
* **Triple riding** -- a motorcycle carrying three riders travels the left verge.
  Its ground-contact point stays outside the lane and outside the no-stopping
  zone throughout, so it is visible only to the geometry-free rider-count rule.

A vehicle waiting at the red signal is deliberately **not** part of the scenario
as an illegal-stopping actor: the no-stopping zone is on the shoulder, nowhere
near the junction, precisely so that stopping at a signal cannot be mistaken for
stopping where it is prohibited.

Pure data, with one I/O function
---------------------------------
Everything except :func:`render_demo_clip` is a pure function: the same call
returns the same boxes, the same draft, the same schedule, and therefore the same
``scene_config_hash``. ``av`` and ``numpy`` are imported *inside* the renderer so
importing this module stays cheap for the callers that only want the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..contracts.enums import ObjectClass, SignalState, ViolationType
from ..contracts.scene import ZoneType
from .builder import (
    DirectionDraft,
    RuleTuning,
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..contracts import SceneConfig

Box = tuple[float, float, float, float]
Point = tuple[float, float]

# --- clip geometry -----------------------------------------------------------------
#: A small 16:9 frame. Large enough to hold four spatially separated scenarios,
#: small enough that the clip encodes in well under a second and stays out of git.
DEMO_WIDTH = 480
DEMO_HEIGHT = 270

#: 10 fps gives PTS steps of 0.1 s, so a declared threshold in seconds maps to a
#: whole number of frames and the scenario timing is readable by eye.
DEMO_FPS = 10
DEMO_FRAME_COUNT = 60  # 6.0 s -- long enough that the slowest threshold
#: (the 3.0 s shoulder dwell, reached at 4.4 s) clears with margin, and short
#: enough that a full run stays quick to reproduce.

# --- declared scene geometry (image space; x right, y down) ------------------------
LANE_ZONE_ID = "zone-lane"
NO_STOPPING_ZONE_ID = "zone-no-stopping"
JUNCTION_ZONE_ID = "zone-junction"
DIRECTION_ID = "dir-legal"
STOP_LINE_ID = "sl-1"
SIGNAL_GROUP_ID = "sg-1"

#: The monitored carriageway: a vertical corridor down the middle of the frame.
LANE_POLYGON: tuple[Point, ...] = (
    (140.0, 0.0),
    (300.0, 0.0),
    (300.0, float(DEMO_HEIGHT)),
    (140.0, float(DEMO_HEIGHT)),
)

#: The junction the stop line guards, beyond the line and clear of it.
JUNCTION_POLYGON: tuple[Point, ...] = (
    (140.0, 140.0),
    (300.0, 140.0),
    (300.0, 200.0),
    (140.0, 200.0),
)

#: The right shoulder, declared off-limits for stopping. Deliberately far from the
#: junction: stopping at a red signal must never read as stopping where prohibited.
NO_STOPPING_POLYGON: tuple[Point, ...] = (
    (310.0, 150.0),
    (470.0, 150.0),
    (470.0, 250.0),
    (310.0, 250.0),
)

#: The signal head this approach obeys. Nothing classifies it -- the ROI exists so
#: the scene can name the group whose *declared* schedule governs the stop line.
SIGNAL_ROI_POLYGON: tuple[Point, ...] = (
    (20.0, 10.0),
    (60.0, 10.0),
    (60.0, 60.0),
    (20.0, 60.0),
)

STOP_LINE_A: Point = (140.0, 110.0)
STOP_LINE_B: Point = (300.0, 110.0)

#: Traffic legally travels *down* the frame; crossing the stop line downward is
#: what counts as entering the junction.
LEGAL_DX, LEGAL_DY = 0.0, 1.0
CROSSING_DX, CROSSING_DY = 0.0, 1.0

#: The one site threshold this scenario moves off its default: a shoulder dwell of
#: three seconds. Chosen so the clip can stay short, declared here rather than
#: hidden in a reasoner, and carried through the scene like any other operator
#: choice -- it is stamped ``PROVISIONAL``, not validated against anything.
DEMO_STATIONARY_DURATION_S = 3.0

#: The declared signal timing, in media seconds from the start of the clip. It
#: turns green at 2.0 s -- *after* the red-light actor crossed at 1.2 s -- so the
#: run exercises the latch rather than a signal that conveniently stays red.
DEMO_SIGNAL_SCHEDULE: tuple[tuple[float, SignalState], ...] = (
    (0.0, SignalState.RED),
    (2.0, SignalState.GREEN),
)

#: What this scenario was *built* to contain. An expectation, never a detection:
#: it is authored here, and the reasoners are never shown it.
DEMO_EXPECTED_VIOLATIONS: tuple[ViolationType, ...] = (
    ViolationType.WRONG_WAY,
    ViolationType.ILLEGAL_STOPPING,
    ViolationType.RED_LIGHT_JUMPING,
    ViolationType.TRIPLE_RIDING,
)

DEMO_SCENE_NAME = "Controlled demonstration intersection"

#: The declaration that travels with the scene, so a reviewer reading a stored
#: revision months later learns what it is without being told.
DEMO_SCENE_NOTES = (
    "Controlled demonstration scene, not a surveyed site. The carriageway, its "
    "legal travel direction (southbound, down the frame), the stop line, the "
    "junction beyond it and the shoulder no-stopping zone are all declared by the "
    "operator; the signal timing is declared per run. The footage is synthetic and "
    "establishes nothing about detection accuracy on real pixels -- it demonstrates "
    "that the shipped reasoners reach independent conclusions from declared context."
)


# --- actors ------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoActor:
    """One participant in the scenario, and where it is on every frame.

    ``boxes`` is indexed by frame and may be shorter than the clip: an actor that
    leaves the frame simply stops having a box, which is what a real detector
    would report and what lets the tracker retire the track normally.

    ``detector_label`` is the *native* label a COCO detector emits (``motorbike``,
    not ``motorcycle``), so a scripted replay exercises the same label map the real
    path uses. ``object_class`` is what that label maps to.
    """

    actor_id: str
    detector_label: str
    object_class: ObjectClass
    colour: tuple[int, int, int]
    boxes: tuple[Box, ...]
    scenario: str

    def box_at(self, frame_index: int) -> Box | None:
        """This actor's box on ``frame_index``, or ``None`` when it is not present."""

        if 0 <= frame_index < len(self.boxes):
            return self.boxes[frame_index]
        return None


_CAR_W, _CAR_H = 40.0, 30.0

# Red-light actor: descends the lane at 6 px/frame. Crosses y=110 on frame 12
# (t=1.2 s, declared red), reaches the junction on frame 17, and is still inside it
# on frame 26 -- comfortably past the 0.4 s crossing debounce.
_RL_X1, _RL_BOTTOM0, _RL_STEP = 200.0, 40.0, 6.0
_RL_FRAMES = 39  # leaves the frame at the bottom; every box stays in bounds

# Wrong-way actor: climbs the same lane at 6 px/frame for 3.7 s, against a legal
# direction of "down". Its stop-line crossing is backward, so it can never register
# a junction entry.
_WW_X1, _WW_BOTTOM0, _WW_STEP = 150.0, 260.0, -6.0
_WW_FRAMES = 38  # leaves the frame at the top; every box stays in bounds

# Illegal-stopping actor: pulls onto the shoulder over 10 frames, then holds for the
# rest of the clip. The stationarity window (5 samples) fills on frame 14 and the
# declared 3.0 s dwell therefore elapses at frame 44, well inside the clip.
_IS_X1, _IS_HOLD_BOTTOM, _IS_STEP = 360.0, 200.0, 7.0
_IS_ENTER_FRAMES = 10

# Triple-riding actor: a wide motorcycle on the left verge drifting 0.5 px/frame --
# slow enough that consecutive boxes overlap far above the tracker's IoU gate, and
# steady enough that the net-displacement window never calls it stationary.
_MC_X1_0, _MC_STEP = 10.0, 0.5
_MC_W, _MC_H = 110.0, 45.0
_MC_BOTTOM = 230.0
_RIDER_W = 30.0
_RIDER_TOP, _RIDER_BOTTOM = 140.0, 215.0
_RIDER_OFFSETS = (5.0, 40.0, 75.0)
_RIDER_COLOURS = ((230, 170, 60), (60, 210, 150), (170, 90, 220))

#: How many riders the motorcycle carries. Three is the statutory threshold the
#: scene declares; the rule counts what it associates and decides for itself.
DEMO_RIDER_COUNT = 3


def _rl_box(index: int) -> Box:
    bottom = _RL_BOTTOM0 + _RL_STEP * index
    return (_RL_X1, bottom - _CAR_H, _RL_X1 + _CAR_W, bottom)


def _ww_box(index: int) -> Box:
    bottom = _WW_BOTTOM0 + _WW_STEP * index
    return (_WW_X1, bottom - _CAR_H, _WW_X1 + _CAR_W, bottom)


def _is_box(index: int) -> Box:
    if index >= _IS_ENTER_FRAMES:
        bottom = _IS_HOLD_BOTTOM
    else:
        bottom = _IS_HOLD_BOTTOM - (_IS_ENTER_FRAMES - index) * _IS_STEP
    return (_IS_X1, bottom - _CAR_H, _IS_X1 + _CAR_W, bottom)


def _mc_box(index: int) -> Box:
    x1 = _MC_X1_0 + _MC_STEP * index
    return (x1, _MC_BOTTOM - _MC_H, x1 + _MC_W, _MC_BOTTOM)


def _rider_box(index: int, rider: int) -> Box:
    x1 = _MC_X1_0 + _MC_STEP * index + _RIDER_OFFSETS[rider]
    return (x1, _RIDER_TOP, x1 + _RIDER_W, _RIDER_BOTTOM)


def _scaled(boxes: tuple[Box, ...], scale: int) -> tuple[Box, ...]:
    if scale == 1:
        return boxes
    return tuple(tuple(v * scale for v in box) for box in boxes)  # type: ignore[misc]


def _scale_points(points: tuple[Point, ...], scale: int) -> tuple[Point, ...]:
    if scale == 1:
        return points
    return tuple((x * scale, y * scale) for x, y in points)


def demo_actors(
    frames: int = DEMO_FRAME_COUNT, *, scale: int = 1
) -> tuple[DemoActor, ...]:
    """Every actor in the scenario, in a fixed order (deterministic).

    Ordering is stable so a scripted replay produces detections in the same order
    on every run, which is what keeps the greedy associator's output -- and
    therefore every derived event id -- reproducible.

    ``scale`` multiplies every coordinate. It exists because the scenario has two
    renderings: coloured rectangles at 1x (deterministic, no dependencies, what the
    tests replay) and **real vehicle crops composited along these same
    trajectories**, which need a larger canvas for a detector to resolve them. One
    geometry, two renderings -- so a change to a trajectory cannot apply to only one
    of them. See :func:`scaled_demo_scene_draft`.
    """

    actors: list[DemoActor] = [
        DemoActor(
            actor_id="rl-runner",
            detector_label="car",
            object_class=ObjectClass.CAR,
            colour=(220, 50, 50),
            boxes=_scaled(tuple(_rl_box(i) for i in range(min(frames, _RL_FRAMES))), scale),
            scenario="Crosses the stop line while the declared signal is red.",
        ),
        DemoActor(
            actor_id="ww-driver",
            detector_label="car",
            object_class=ObjectClass.CAR,
            colour=(60, 110, 240),
            boxes=_scaled(tuple(_ww_box(i) for i in range(min(frames, _WW_FRAMES))), scale),
            scenario="Travels against the declared legal direction of the lane.",
        ),
        DemoActor(
            actor_id="is-stopper",
            detector_label="car",
            object_class=ObjectClass.CAR,
            colour=(240, 190, 40),
            boxes=_scaled(tuple(_is_box(i) for i in range(frames)), scale),
            scenario="Stops inside the declared no-stopping zone and holds.",
        ),
        DemoActor(
            actor_id="tr-motorcycle",
            detector_label="motorbike",
            object_class=ObjectClass.MOTORCYCLE,
            colour=(40, 60, 210),
            boxes=_scaled(tuple(_mc_box(i) for i in range(frames)), scale),
            scenario="Carries three riders for the whole clip.",
        ),
    ]
    actors.extend(
        DemoActor(
            actor_id=f"tr-rider-{rider}",
            detector_label="person",
            object_class=ObjectClass.PERSON,
            colour=_RIDER_COLOURS[rider],
            boxes=_scaled(tuple(_rider_box(i, rider) for i in range(frames)), scale),
            scenario="Rider astride the motorcycle.",
        )
        for rider in range(DEMO_RIDER_COUNT)
    )
    return tuple(actors)


#: The detector label map the scenario's scripted labels need. Named here rather
#: than restated by each caller, so the script and the tests map identically.
DEMO_LABEL_MAP: dict[str, ObjectClass] = {
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "motorcycle": ObjectClass.MOTORCYCLE,
    "person": ObjectClass.PERSON,
}


# --- the declared scene ------------------------------------------------------------
def demo_scene_draft(
    *,
    camera_id: str = "cam-controlled-demo",
    site_id: str = "site-controlled-demo",
    scale: int = 1,
) -> SceneDraft:
    """The calibration an analyst would draw for this scenario.

    Expressed as a :class:`~trafficpulse.scenes.builder.SceneDraft` -- the same
    vocabulary the calibration surface submits -- so the demonstration goes through
    the ordinary authoring path and produces an ordinary ``SceneConfig``. Nothing
    about it is privileged: it is a drawing, in the clip's own pixel space.
    """

    return SceneDraft(
        scene_name=DEMO_SCENE_NAME,
        camera_id=camera_id,
        site_id=site_id,
        description=DEMO_SCENE_NOTES,
        frame_width=DEMO_WIDTH * scale,
        frame_height=DEMO_HEIGHT * scale,
        zones=(
            ZoneDraft(
                zone_id=LANE_ZONE_ID,
                zone_type=ZoneType.LANE,
                polygon=_scale_points(LANE_POLYGON, scale),
                description="Monitored carriageway; traffic legally travels down the frame.",
            ),
            ZoneDraft(
                zone_id=JUNCTION_ZONE_ID,
                zone_type=ZoneType.INTERSECTION,
                polygon=_scale_points(JUNCTION_POLYGON, scale),
                description="Conflict area beyond the stop line.",
            ),
            ZoneDraft(
                zone_id=NO_STOPPING_ZONE_ID,
                zone_type=ZoneType.NO_STOPPING,
                polygon=_scale_points(NO_STOPPING_POLYGON, scale),
                description=(
                    "Shoulder declared off-limits for stopping. Deliberately clear of "
                    "the junction so waiting at the signal is not stopping here."
                ),
            ),
        ),
        direction=DirectionDraft(
            direction_id=DIRECTION_ID,
            dx=LEGAL_DX,
            dy=LEGAL_DY,
            zone_id=LANE_ZONE_ID,
            description="Legal travel direction: southbound (down the frame). Declared.",
        ),
        stop_lines=(
            StopLineDraft(
                stop_line_id=STOP_LINE_ID,
                a=_scale_points((STOP_LINE_A,), scale)[0],
                b=_scale_points((STOP_LINE_B,), scale)[0],
                crossing_dx=CROSSING_DX,
                crossing_dy=CROSSING_DY,
                signal_group_id=SIGNAL_GROUP_ID,
                zone_ids=(JUNCTION_ZONE_ID,),
            ),
        ),
        signal_groups=(
            SignalGroupDraft(
                signal_group_id=SIGNAL_GROUP_ID,
                roi_polygon=_scale_points(SIGNAL_ROI_POLYGON, scale),
                zone_ids=(JUNCTION_ZONE_ID,),
            ),
        ),
        tuning=RuleTuning(stationary_duration_seconds=DEMO_STATIONARY_DURATION_S),
    )


def demo_scene(scene_id: str = "scene-controlled-demo", *, scale: int = 1) -> SceneConfig:
    """The validated ``SceneConfig`` the draft expands into (deterministic)."""

    from .builder import build_scene

    return build_scene(demo_scene_draft(scale=scale), scene_id=scene_id)


# --- rendering (the only I/O here) -------------------------------------------------
def render_demo_clip(
    path: Path | str, *, frames: int = DEMO_FRAME_COUNT, scale: int = 1
) -> Path:
    """Encode the scenario as a real video file at ``path``; return the path.

    Deterministic pixels: each actor is a filled rectangle in its own colour, drawn
    in actor order. The codec is ``mpeg4`` in an mp4 container -- what PyAV's
    bundled FFmpeg encodes on every platform this project runs on, and what the
    ingestion tests already rely on.

    The clip is **not** committed. It is a few tens of kilobytes and is rebuilt in
    under a second from this specification, so regenerating it is cheaper and more
    honest than storing a binary whose provenance nobody can check.
    """

    import av
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    actors = demo_actors(frames, scale=scale)
    width, height = DEMO_WIDTH * scale, DEMO_HEIGHT * scale

    container = av.open(str(target), "w")
    try:
        stream = container.add_stream("mpeg4", rate=DEMO_FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(frames):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for actor in actors:
                box = actor.box_at(index)
                if box is None:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
                if x2 > x1 and y2 > y1:
                    image[y1:y2, x1:x2] = actor.colour
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return target
