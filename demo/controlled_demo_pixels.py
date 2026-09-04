#!/usr/bin/env python
"""Render the controlled scenario with **real vehicle pixels**, for real inference.

    ./.venv/Scripts/python.exe demo/controlled_demo_pixels.py --verify

Why this exists
---------------
:func:`trafficpulse.scenes.demo_scenario.render_demo_clip` draws the scenario as
coloured rectangles. That is exactly right for the test suite -- deterministic, no
media dependency, and the tests pair it with a scripted detector -- but **a COCO
RT-DETR detects nothing in it**, verified: 0 detections across every sampled frame.
So the rectangle clip cannot be uploaded to a running TrafficPulse and produce
events, and a browser demonstration built on it would silently show zero.

This module renders the *same* scenario -- the same trajectories, the same declared
scene, the same expectations -- by compositing **real vehicle crops cut from the
project's own corpus** along those trajectories, on a plain road canvas. Real
RT-DETR then has real pixels to detect, and everything downstream is the ordinary
production path.

It authors the scenario, never the analysis
--------------------------------------------
The same discipline as ``demo/make_wrong_way_upload_clip.py``, which this follows:
the *input video* is constructed; nothing after it is. RT-DETR runs real inference
on the composited pixels and must actually find the vehicles, the real tracker
associates them, the real derivations and reasoners decide, and the unchanged store
persists. No detection, track, observation or event is fabricated.

What is genuinely constructed, and stated plainly:

* the **canvas** is a plain asphalt-coloured background with lane markings, not real
  road footage. It is deliberately empty so the only things in frame are the actors
  -- a real background would add its own vehicles, and their tracks would be
  uncontrolled additions to a controlled demonstration;
* the **crops** are real vehicles from real footage, but they do not change
  appearance as they move (no perspective, no lighting change, no wheel rotation);
* the **trajectories** are the authored scenario's.

Requires the corpus media
--------------------------
The crops are cut from ``test-videos/``, whose media is gitignored (see
``test-videos/README.md``). Without it this script fails with a clear message rather
than substituting anything. The rectangle clip and the whole test suite need none of
it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np

from trafficpulse.scenes import demo_scenario as scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "runs" / "controlled-demo" / "controlled-demo-pixels.mp4"

#: 4x the scenario's 480x270 geometry -> 1920x1080. Chosen from a measurement, not a
#: guess: a real car crop pasted at the scenario's car-box size is detected at 0.96
#: confidence at 2x, 3x and 4x, and 1080p is the resolution the rest of the corpus
#: uses, so the demo clip is not an outlier in the library.
SCALE = 4


class Crop(NamedTuple):
    """One real vehicle's pixels, and exactly where they came from.

    Provenance is part of the data, not a comment: a composited demonstration that
    cannot say which frame of which clip its pixels are from is not auditable.
    """

    source: Path
    frame_index: int
    box: tuple[int, int, int, int]
    note: str


#: A car from the 4K roundabout clip -- a fixed camera, clean daylight, and the
#: vehicle RT-DETR detects most confidently in frame 0 (score 0.93 in situ, 0.96
#: once composited).
CAR_CROP = Crop(
    source=REPO_ROOT / "test-videos" / "normal-traffic" / "clean_003.webm",
    frame_index=0,
    box=(2119, 110, 2461, 378),
    note="Car, Wikimedia roundabout clip (clean_003), frame 0.",
)

#: A motorcycle carrying three riders, from the Raxaul congestion clip. Frame 80 and
#: this margin specifically: of eight three-rider candidates sampled across the clip,
#: this is the only crop that still reads as a motorcycle *with three riders* after
#: being cut out and composited. The others lose the bike -- it is occluded by its own
#: riders, and RT-DETR only recovers it from the surrounding scene context.
MOTORCYCLE_CROP = Crop(
    source=REPO_ROOT / "test-videos" / "edge-cases" / "congestion" / "congestion_002.webm",
    frame_index=80,
    box=(608, 165, 1553, 1080),
    note="Motorcycle carrying three riders, Raxaul congestion clip, frame 80.",
)


def _read_frame(path: Path, index: int) -> np.ndarray:
    """Decode one frame of a source clip as RGB, or fail with a usable message."""

    if not path.is_file():
        raise SystemExit(
            f"missing corpus media: {path}\n"
            "The composited clip cuts its vehicles from test-videos/, whose media is "
            "gitignored. Fetch it (test-videos/fetch.py) or use the rectangle clip "
            "instead: demo/controlled_demo.py --render-only"
        )
    container = av.open(str(path))
    try:
        for position, frame in enumerate(container.decode(video=0)):
            if position == index:
                image: np.ndarray = frame.to_ndarray(format="rgb24")
                return image
    finally:
        container.close()
    raise SystemExit(f"{path} has no frame {index}")


def _crop_pixels(crop: Crop) -> np.ndarray:
    image = _read_frame(crop.source, crop.frame_index)
    x1, y1, x2, y2 = crop.box
    return image[y1:y2, x1:x2]


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize (numpy only; no new dependency)."""

    ys = (np.arange(height) * (image.shape[0] / height)).astype(int)
    xs = (np.arange(width) * (image.shape[1] / width)).astype(int)
    resized: np.ndarray = image[ys][:, xs]
    return resized


def road_canvas(width: int, height: int) -> np.ndarray:
    """A plain carriageway: asphalt, verges either side, a dashed centre line.

    Deliberately synthetic and deliberately empty. Using real road footage as the
    backdrop would put real vehicles in frame, and their tracks would be uncontrolled
    actors in what is supposed to be a controlled demonstration. RT-DETR finds
    nothing in this canvas, which is the property that makes the composited actors
    the only things in the run.
    """

    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (92, 92, 96)  # asphalt
    verge = (72, 80, 68)
    canvas[:, : int(width * 0.28)] = verge
    canvas[:, int(width * 0.64) :] = verge
    stripe = max(int(height / 40), 2)
    for y in range(0, height, int(height / 12)):
        canvas[y : y + stripe * 2, int(width * 0.455) : int(width * 0.465)] = (215, 215, 205)
    return canvas


def render(path: Path, *, frames: int = scenario.DEMO_FRAME_COUNT, scale: int = SCALE) -> Path:
    """Composite the scenario's actors as real vehicles. Returns ``path``.

    The three cars are pasted into the scenario's own car boxes. The motorcycle and
    its riders are pasted as **one block** covering the scenario's bike-plus-riders
    extent, because they are one photograph of one real motorcycle: cutting it into
    four boxes and pasting them separately would be assembling a scene rather than
    reusing one.
    """

    width, height = scenario.DEMO_WIDTH * scale, scenario.DEMO_HEIGHT * scale
    actors = {actor.actor_id: actor for actor in scenario.demo_actors(frames, scale=scale)}
    car = _crop_pixels(CAR_CROP)
    bike = _crop_pixels(MOTORCYCLE_CROP)
    background = road_canvas(width, height)

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    try:
        stream = container.add_stream("mpeg4", rate=scenario.DEMO_FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.bit_rate = 4_000_000  # crisp enough that the detector is not fighting the codec
        for index in range(frames):
            frame = background.copy()
            for actor_id in ("rl-runner", "ww-driver", "is-stopper"):
                box = actors[actor_id].box_at(index)
                if box is not None:
                    _paste(frame, car, box)
            block = _motorcycle_block(actors, index)
            if block is not None:
                _paste(frame, bike, _fit_aspect(bike, block))
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path


def _motorcycle_block(
    actors: dict[str, scenario.DemoActor], index: int
) -> tuple[float, float, float, float] | None:
    """The union of the bike box and its riders' boxes on one frame."""

    boxes = [
        box
        for actor_id in ("tr-motorcycle", "tr-rider-0", "tr-rider-1", "tr-rider-2")
        if (box := actors[actor_id].box_at(index)) is not None
    ]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _fit_aspect(
    patch: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Fit ``patch`` into ``box`` without distorting it, anchored at the bottom centre.

    The motorcycle photograph is nearly square; the scenario's bike-plus-riders block
    is wider than tall. Stretching one into the other squashes the machine out of
    recognition -- measurably: the stretched paste yields **no** motorcycle detection
    at all, while the aspect-preserved one is detected. The bottom edge is the anchor
    because it is the ground-contact point every zone derivation reads.
    """

    width = box[2] - box[0]
    height = box[3] - box[1]
    aspect = patch.shape[1] / patch.shape[0]
    # Scale up to the block's *diagonal* extent rather than fitting inside it: the
    # block describes where the machine sits, not how much of the frame it may use,
    # and a detector needs the vehicle at a plausible on-road size.
    fitted_height = max(height, width / aspect)
    fitted_width = fitted_height * aspect
    centre_x = (box[0] + box[2]) / 2
    return (
        centre_x - fitted_width / 2,
        box[3] - fitted_height,
        centre_x + fitted_width / 2,
        box[3],
    )


def _paste(canvas: np.ndarray, patch: np.ndarray, box: tuple[float, float, float, float]) -> None:
    """Paste ``patch`` into ``box``, clipped to the canvas."""

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, canvas.shape[1]), min(y2, canvas.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = _resize(patch, x2 - x1, y2 - y1)


def verify(clip: Path, *, frames: int) -> int:
    """Run the real detector, tracker and reasoners over the clip and report.

    The point of the whole exercise: the composited clip is only worth anything if
    **real inference** finds the actors. This reports what was actually confirmed --
    including families that were not, which is the outcome that must never be hidden.
    """

    sys.path.insert(0, str(REPO_ROOT))
    import serve  # the production composition: real RT-DETR, real label map

    from trafficpulse.engine import (
        EngineConfig,
        IllegalStoppingRuleConfig,
        InferenceEngine,
        RedLightRuleConfig,
        SignalPhaseSpec,
        TripleRidingRuleConfig,
        WrongWayRuleConfig,
    )
    from trafficpulse.engine.runner import build_detector, detector_adapter_config
    from trafficpulse.engine.sources import FileFrameSource
    from trafficpulse.tracking import IouTracker

    inference = serve.build_config().inference
    if inference is None:
        raise SystemExit("serve.build_config() declares no inference backend")

    print("\nRunning the real path (RT-DETR -> tracker -> reasoners). This takes a minute...")
    engine = InferenceEngine(
        scene=scenario.demo_scene(scale=SCALE),
        detector=build_detector(inference),
        tracker=IouTracker(),
        detector_config=detector_adapter_config(inference),
        config=EngineConfig(
            rules=(
                WrongWayRuleConfig(direction_id=scenario.DIRECTION_ID),
                IllegalStoppingRuleConfig(),
                TripleRidingRuleConfig(),
                RedLightRuleConfig(
                    schedule=tuple(
                        SignalPhaseSpec(at_seconds=at, state=state)
                        for at, state in scenario.DEMO_SIGNAL_SCHEDULE
                    ),
                    stop_line_id=scenario.STOP_LINE_ID,
                    zone_id=scenario.JUNCTION_ZONE_ID,
                ),
            ),
        ),
    )
    result = engine.run(FileFrameSource(clip))

    confirmed = {event.violation_type.value for event in result.events}
    expected = {v.value for v in scenario.DEMO_EXPECTED_VIOLATIONS}
    print(f"\ndetections: {result.metrics.detections}   tracks: {result.metrics.track_states}")
    print(f"events: {len(result.events)}")
    for event in result.events:
        print(f"  - {event.violation_type.value:<20} tracks={event.track_ids}")
    print("\nEXPECTED vs DETECTED (real inference, composited pixels)")
    for violation in sorted(expected | confirmed):
        mark = "matched" if violation in expected and violation in confirmed else (
            "MISSING" if violation in expected else "unexpected"
        )
        print(f"  {violation:<22} {mark}")
    missing = expected - confirmed
    if missing:
        print(f"\nNot confirmed: {sorted(missing)}. That is the honest result for this clip;")
        print("do not adjust the scenario until it goes away.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--frames", type=int, default=scenario.DEMO_FRAME_COUNT, help="frames to render"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the real RT-DETR path over the rendered clip and report what confirmed",
    )
    args = parser.parse_args(argv)

    print("Compositing the controlled scenario from real vehicle crops:")
    print(f"  {CAR_CROP.note}\n    {CAR_CROP.source.name} box={CAR_CROP.box}")
    print(f"  {MOTORCYCLE_CROP.note}\n    {MOTORCYCLE_CROP.source.name} box={MOTORCYCLE_CROP.box}")
    clip = render(args.out, frames=args.frames)
    print(f"\nClip written: {clip} ({clip.stat().st_size} bytes, "
          f"{scenario.DEMO_WIDTH * SCALE}x{scenario.DEMO_HEIGHT * SCALE})")
    print("The scenario, the scene to draw and the expectations are unchanged --")
    print("run demo/controlled_demo.py --render-only to print them.")

    if args.verify:
        return verify(clip, frames=args.frames)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
