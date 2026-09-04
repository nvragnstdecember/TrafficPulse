"""Fixtures for the controlled-demonstration tests: the clip and its scripted detector.

Thin by design. The scenario itself -- every actor's boxes, the declared geometry,
the signal timing, the expected families -- lives in
:mod:`trafficpulse.scenes.demo_scenario`, which is runtime code the operator script
also uses. This module only turns that specification into the two things a test
needs: a real decoded clip on disk, and a ``StubDetector`` scripted to match it.

Why a stub detector over real pixels
-------------------------------------
The clip is genuinely decoded (PyAV, real PTS, real frame order) and the tracker,
the observation derivations and all five reasoners are the real ones. Only the
*detector* is scripted, for the same reason every sibling fixture scripts it: a
COCO RT-DETR does not fire a vehicle class on synthetic rectangles, so a real
backend here would detect nothing and the test would prove only that. The script is
authored from the same boxes the renderer drew, so it replays what is on screen and
never invents a detection the pixels do not show.

Uniquely named (``_controlled_demo_fixtures``) for pytest's prepend import mode.
"""

from __future__ import annotations

from pathlib import Path

from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector
from trafficpulse.scenes import demo_scenario as scenario


def write_controlled_demo_clip(
    path: Path, *, frames: int = scenario.DEMO_FRAME_COUNT
) -> Path:
    """Render the controlled scenario to a real video file. Returns ``path``."""

    return scenario.render_demo_clip(path, frames=frames)


def scripted_controlled_demo_detector(
    frames: int = scenario.DEMO_FRAME_COUNT,
) -> StubDetector:
    """A ``StubDetector`` replaying exactly the boxes the renderer drew.

    Keyed by ``frame.frame_index`` so it aligns with decoded frame order, and
    emitting each actor's *native* detector label (``motorbike``, ``car``,
    ``person``) so the run exercises the same label map the real path uses.
    """

    actors = scenario.demo_actors(frames)
    per_frame = {
        index: tuple(
            RawDetection(label=actor.detector_label, score=0.9, box=box)
            for actor in actors
            if (box := actor.box_at(index)) is not None
        )
        for index in range(frames)
    }
    return StubDetector(per_frame=per_frame)


def controlled_demo_detector_config() -> DetectorConfig:
    """Detector config mapping the scenario's scripted labels to their classes."""

    return DetectorConfig(label_map=dict(scenario.DEMO_LABEL_MAP))


def controlled_demo_draft_payload() -> dict[str, object]:
    """The scene draft as the JSON body the calibration endpoint accepts.

    Round-tripped through the model's own JSON dump rather than hand-written, so
    the request body a test sends is provably the draft the runtime specifies.
    """

    payload: dict[str, object] = scenario.demo_scene_draft().model_dump(mode="json")
    return payload


def controlled_demo_red_light_rule() -> dict[str, object]:
    """The red-light rule declaration carrying this run's declared signal timing.

    Red-light is the one rule a scene cannot supply on its own -- the schedule names
    media-time instants, which belong to a clip rather than a camera -- so a client
    must send it, exactly as the calibration surface does.
    """

    return {
        "kind": "red_light_jumping",
        "schedule": [
            {"at_seconds": at, "state": state.value}
            for at, state in scenario.DEMO_SIGNAL_SCHEDULE
        ],
        "stop_line_id": scenario.STOP_LINE_ID,
        "zone_id": scenario.JUNCTION_ZONE_ID,
    }
