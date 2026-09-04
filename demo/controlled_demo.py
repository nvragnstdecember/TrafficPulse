#!/usr/bin/env python
"""Run the **controlled demonstration**: one clip, four independently reasoned violations.

    # 1. render the clip (nothing is committed; it is rebuilt from the spec)
    ./.venv/Scripts/python.exe demo/controlled_demo.py --render-only

    # 2. with a TrafficPulse server running, drive the whole flow over HTTP
    ./.venv/Scripts/python.exe demo/controlled_demo.py --api http://127.0.0.1:8000

What this establishes, and what it does not
--------------------------------------------
It establishes that TrafficPulse, given **one video and the scene context a camera
cannot know by itself**, has four separate reasoners each reach their own
conclusion in a single pass -- and that the declared expectations play no part in
reaching them.

It establishes **nothing about real-world detection performance**. The scenario, the
canvas and the trajectories are authored; the vehicles in the ``--real-pixels`` clip
are real pixels on an empty synthetic road, which is not the same thing as real
traffic. Real-footage findings are a different category and live in
``docs/validation-matrix.md``; this script must never be quoted as validation, and
its output says so.

Why a controlled clip exists at all
------------------------------------
The project's real corpus contains no clip with a red-light runner, a wrong-way
vehicle, an illegally stopped car and an overloaded motorcycle in it -- and none
can honestly be manufactured by lowering a threshold until real footage produces
the wanted answer. Declaring a scenario, and saying that it is declared, is the
honest alternative.

Two clips, and which one to use
--------------------------------
``--render-only`` writes the **rectangle** clip: coloured boxes on black, fully
deterministic, no media dependency. It is what the test suite replays against a
*scripted* detector. **A COCO RT-DETR detects nothing in it** (measured: 0 detections
on every sampled frame), so uploading it to a running server produces no events.

``--real-pixels`` writes the **composited** clip instead: the same trajectories, the
same declared scene, the same expectations -- but each actor is a real vehicle cut
from the project's own corpus, on a plain road canvas. Real RT-DETR has real pixels
to find, so this is the clip to upload for a browser demonstration. It needs the
gitignored ``test-videos/`` media; see ``demo/controlled_demo_pixels.py``.

Use ``--real-pixels`` for anything a person watches, and the rectangle clip for
anything a machine checks.

With ``--api`` it does the same thing over HTTP against a running server: upload,
calibrate, declare the timing, declare the expectations, process, and print the
expected-vs-detected table. Nothing it sends is privileged; every request is one the
Videos workspace makes.

The scenario itself -- every actor, every polygon, the signal schedule and the
expected families -- lives in :mod:`trafficpulse.scenes.demo_scenario`, which the
test suite uses too, so this script and
``tests/app/test_app_controlled_demo.py`` can never drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from trafficpulse.scenes import demo_scenario as scenario

DEFAULT_CLIP = Path("runs/controlled-demo/controlled-demo.mp4")

#: Poll interval and ceiling for a processing job. The controlled clip is 6 s of
#: 480x270, so a run finishes in seconds; the ceiling exists so a wedged server
#: fails loudly instead of hanging a demo.
_POLL_SECONDS = 0.5
_POLL_TIMEOUT_SECONDS = 300.0

_BANNER = """
=============================================================================
 CONTROLLED DEMONSTRATION -- authored scenario, declared scene context
 This is NOT real-world validation. It shows that the shipped reasoners reach
 independent conclusions from context an operator declared. The scenario, the
 road and the trajectories are authored (see docs/validation-matrix.md 7b).
=============================================================================
"""


def _scaled(points: object, scale: int) -> object:
    """Scale a point or a ring of points for printing at the clip's real size."""

    if isinstance(points, tuple) and points and isinstance(points[0], tuple):
        return tuple((x * scale, y * scale) for x, y in points)  # type: ignore[misc]
    x, y = points  # type: ignore[misc]
    return (x * scale, y * scale)


def _print_scenario(scale: int = 1) -> None:
    print(_BANNER)
    print(f"Clip           : {scenario.DEMO_WIDTH * scale}x{scenario.DEMO_HEIGHT * scale}, "
          f"{scenario.DEMO_FRAME_COUNT} frames @ {scenario.DEMO_FPS} fps "
          f"({scenario.DEMO_FRAME_COUNT / scenario.DEMO_FPS:.1f} s)")
    print("\nActors (what the clip was built to contain):")
    for actor in scenario.demo_actors():
        if actor.detector_label == "person":
            continue  # riders are reported with the motorcycle they sit on
        print(f"  - {actor.actor_id:<16} {actor.scenario}")
    print("\nDeclared scene context (the camera cannot know any of this):")
    print(f"  legal direction    : dx={scenario.LEGAL_DX}, dy={scenario.LEGAL_DY} "
          "(southbound, down the frame)")
    print(f"  lane polygon       : {_scaled(scenario.LANE_POLYGON, scale)}")
    print(f"  stop line          : {_scaled(scenario.STOP_LINE_A, scale)} -> "
          f"{_scaled(scenario.STOP_LINE_B, scale)}")
    print(f"  junction polygon   : {_scaled(scenario.JUNCTION_POLYGON, scale)}")
    print(f"  no-stopping zone   : {_scaled(scenario.NO_STOPPING_POLYGON, scale)}")
    print(f"  dwell threshold    : {scenario.DEMO_STATIONARY_DURATION_S} s (operator-chosen)")
    print("  signal timing      : "
          + ", ".join(f"{at:.1f}s {state.value}" for at, state in scenario.DEMO_SIGNAL_SCHEDULE))
    print("\nDeclared expectations (ground truth for THIS demonstration only):")
    for violation in scenario.DEMO_EXPECTED_VIOLATIONS:
        print(f"  - {violation.value}")
    print("\nThe reasoners are never shown the expectations. They are compared with the")
    print("run's confirmed events afterwards, and a declared family with no event is")
    print("reported as missing rather than conjured.\n")


def _signal_schedule_payload() -> list[dict[str, Any]]:
    return [
        {"at_seconds": at, "state": state.value}
        for at, state in scenario.DEMO_SIGNAL_SCHEDULE
    ]


def _rules_payload(supported: list[str]) -> list[dict[str, Any]]:
    """The rule set the calibration surface would send for this scene.

    Red-light additionally carries the run's signal schedule, because timing names
    media-time instants and belongs to a clip rather than to a camera.
    """

    rules: list[dict[str, Any]] = []
    for violation in supported:
        if violation == "red_light_jumping":
            rules.append(
                {
                    "kind": "red_light_jumping",
                    "schedule": _signal_schedule_payload(),
                    "stop_line_id": scenario.STOP_LINE_ID,
                    "zone_id": scenario.JUNCTION_ZONE_ID,
                }
            )
        elif violation != "speeding":  # no shipped reasoner
            rules.append({"kind": violation})
    return rules


def _drive(api: str, clip: Path, scale: int = 1) -> int:
    """Upload, calibrate, declare, process and compare, over the real HTTP API."""

    import httpx

    with httpx.Client(base_url=api.rstrip("/"), timeout=60.0) as client:
        health = client.get("/api/health")
        if health.status_code != 200:
            print(f"FAILED: {api} is not serving the API ({health.status_code})")
            return 1
        print(f"Server         : {api}  {health.json()}\n")

        upload = client.post(
            "/api/video/upload",
            files={"file": (clip.name, clip.read_bytes(), "video/mp4")},
        )
        if upload.status_code == 409:
            # Uploads are content-addressed, so re-running is a no-op rather than a
            # duplicate -- reuse the existing id instead of failing the demo.
            video_id = upload.json()["error"]["video_id"]
            print(f"Video          : {video_id} (already uploaded; reused)")
        elif upload.status_code == 201:
            video_id = upload.json()["video_id"]
            print(f"Video          : {video_id} (uploaded)")
        else:
            print(f"FAILED to upload: {upload.status_code} {upload.text}")
            return 1

        scene_response = client.put(
            f"/api/videos/{video_id}/scene",
            json=scenario.demo_scene_draft(scale=scale).model_dump(mode="json"),
        )
        if scene_response.status_code != 200:
            print(f"FAILED to calibrate: {scene_response.status_code} {scene_response.text}")
            return 1
        scene = scene_response.json()
        print(f"Scene          : {scene['scene_hash'][:16]}...  "
              f"unlocks {', '.join(scene['supported_violations'])}")

        declaration = client.put(
            f"/api/videos/{video_id}/expectation",
            json={
                "expected_violations": [v.value for v in scenario.DEMO_EXPECTED_VIOLATIONS],
                "notes": scenario.DEMO_SCENE_NOTES,
                "declared_by": "controlled-demo-script",
            },
        )
        if declaration.status_code != 200:
            print(f"FAILED to declare: {declaration.status_code} {declaration.text}")
            return 1
        print(f"Expectation    : {len(scenario.DEMO_EXPECTED_VIOLATIONS)} families declared")

        job = client.post(
            "/api/process",
            json={"video_id": video_id, "rules": _rules_payload(scene["supported_violations"])},
        )
        if job.status_code != 202:
            print(f"FAILED to start processing: {job.status_code} {job.text}")
            return 1
        job_id = job.json()["job_id"]
        print(f"Job            : {job_id} ...", end="", flush=True)

        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = client.get(f"/api/process/{job_id}").json()
            if status["status"] in {"succeeded", "failed", "cancelled"}:
                break
            print(".", end="", flush=True)
            time.sleep(_POLL_SECONDS)
        print(f" {status.get('status', 'timed out')}")
        if status.get("status") != "succeeded":
            print(f"FAILED: {json.dumps(status, indent=2)}")
            return 1

        comparison = client.get(
            f"/api/videos/{video_id}/expectation/comparison", params={"job_id": job_id}
        ).json()
        _print_comparison(comparison)
        print(f"\nOpen the workspace at {api}/videos and select this clip to inspect")
        print("each event's evidence, measurements and rule trace.")
        return 0 if comparison["missing_count"] == 0 else 2


def _print_comparison(comparison: dict[str, Any]) -> None:
    print("\nEXPECTED vs DETECTED  (counts of real event ids; no accuracy is computed)")
    print(f"  {'family':<22} {'expected':<10} {'detected':<10} outcome")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 12}")
    for row in comparison["rows"]:
        print(
            f"  {row['violation_type']:<22} "
            f"{'yes' if row['expected'] else 'no':<10} "
            f"{row['detected_count']:<10} {row['outcome']}"
        )
    print(
        f"\n  expected {comparison['expected_count']} | "
        f"detected {comparison['detected_event_count']} event(s) | "
        f"matched {comparison['matched_count']} | "
        f"missing {comparison['missing_count']} | "
        f"unexpected {comparison['unexpected_count']}"
    )
    print("\n  No precision, recall or F1 is reported. Over one hand-authored clip those")
    print("  would be arithmetic against ground truth the same person wrote.")


def controlled_scale(real_pixels: bool) -> int:
    """The coordinate scale each rendering uses.

    The rectangle clip stays at the scenario's own 1x. The composited one is rendered
    at 4x (1920x1080) because a detector needs the vehicles at a plausible on-road
    size -- a measured choice, see ``controlled_demo_pixels.SCALE``. Both drive the
    *same* geometry, so the calibration to draw is the printed one either way.
    """

    if not real_pixels:
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import controlled_demo_pixels

    return int(controlled_demo_pixels.SCALE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render and run the controlled multi-violation demonstration."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CLIP,
        help=f"where to write the controlled clip (default: {DEFAULT_CLIP})",
    )
    parser.add_argument(
        "--api",
        default=None,
        help="base URL of a running TrafficPulse server to drive over HTTP "
        "(e.g. http://127.0.0.1:8000). Omit to only render the clip.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="write the clip and print the calibration, then stop.",
    )
    parser.add_argument(
        "--real-pixels",
        action="store_true",
        help="composite real vehicle crops instead of drawing rectangles. Required "
        "for any run against real inference: RT-DETR detects nothing in the "
        "rectangle clip. Needs the gitignored test-videos/ media.",
    )
    args = parser.parse_args(argv)

    scale = controlled_scale(args.real_pixels)
    _print_scenario(scale)
    if args.real_pixels:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import controlled_demo_pixels

        out = (
            controlled_demo_pixels.DEFAULT_OUT if args.out == DEFAULT_CLIP else args.out
        )
        clip = controlled_demo_pixels.render(out)
        print(f"Clip written   : {clip}  ({clip.stat().st_size} bytes, real vehicle crops)")
        print(f"  car          : {controlled_demo_pixels.CAR_CROP.note}")
        print(f"  motorcycle   : {controlled_demo_pixels.MOTORCYCLE_CROP.note}")
    else:
        clip = scenario.render_demo_clip(args.out)
        print(f"Clip written   : {clip}  ({clip.stat().st_size} bytes, rectangles)")

    if args.render_only or args.api is None:
        if not args.real_pixels:
            print("")
            print("NOTE: this is the rectangle clip, and RT-DETR detects nothing in it.")
            print("      Uploading it to a real server produces zero events. Re-run with")
            print("      --real-pixels for anything a person watches.")
        print("")
        print("Next: start TrafficPulse, open /videos, upload this clip, and use the")
        print("Controlled demo panel to draw the geometry above, declare the signal")
        print("timing, declare the expectations, and run the analysis.")
        print("Or re-run this script with --api http://127.0.0.1:8000 to do it over HTTP.")
        return 0

    return _drive(args.api, clip, scale)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
