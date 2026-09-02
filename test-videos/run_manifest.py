#!/usr/bin/env python
"""Execute the T0 regression manifest's *established* structural expectations.

    python test-videos/run_manifest.py            # report; exit 1 on any failure
    python test-videos/run_manifest.py --verbose  # also list what was skipped and why

What this runs, and what it deliberately does not
--------------------------------------------------
``evaluation/manifest.yaml`` labels every expectation with a ``basis`` and an
``established`` flag, and this runner honours both **literally**:

* ``established: true`` on a violation this repository can decide *by construction*
  -> **executed**. These are the manifest's own words: "guaranteed by the system's
  own design, independent of what is in the footage". An uncalibrated scene declares
  no legal direction and no no-stopping zone, so the capability layer refuses to
  build a wrong-way or illegal-stopping rule at all, and the count is zero before a
  single frame is decoded. That is checkable now, needs no labelling, and still
  catches the regressions that matter: a geometry rule leaking into an uncalibrated
  run, or a scene leaking between videos.

* ``established: false`` -> **never evaluated**. The calibrated wrong-way variant is
  the live example: its expectation is ``>0`` with the count explicitly not
  established. Silently "checking" it would manufacture a benchmark out of a
  placeholder, which is the exact failure ``manifest.yaml`` was written to prevent.

* ``unestablished:`` entries (``no_helmet: null``, ``triple_riding: null``) ->
  **never evaluated**, for the same reason. They are a labelling TODO, not a target.

* An expectation that depends on **what is in the footage** rather than on the
  system's design -- ``clean_002``'s "no motorcycles appear, so the motorcycle rules
  see nothing" -- is reported as **NOT VERIFIED**, never as a pass. It is a true
  claim about that clip, but only inference can confirm it, and this runner decodes
  no video. Counting it as a pass would be the runner asserting something it never
  checked.

So a green run means: "every expectation this repository can decide without footage
holds." It does not mean the corpus was analysed. That distinction is the point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MANIFEST = ROOT / "evaluation" / "manifest.yaml"

if str(REPO / "src") not in sys.path:  # runnable from a plain checkout
    sys.path.insert(0, str(REPO / "src"))

from trafficpulse.app.capabilities import supported_violations  # noqa: E402
from trafficpulse.contracts.enums import ViolationType  # noqa: E402
from trafficpulse.contracts.scene import ZoneType  # noqa: E402
from trafficpulse.scenes import (  # noqa: E402
    CALIBRATION_SOURCE_AUTO,
    SceneDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)

#: Violations whose absence an uncalibrated scene guarantees *structurally* -- the
#: geometry rules, which cannot be built without a declared direction, no-stopping
#: zone, or stop line + signal schedule.
STRUCTURAL: frozenset[ViolationType] = frozenset(
    {
        ViolationType.WRONG_WAY,
        ViolationType.ILLEGAL_STOPPING,
        ViolationType.RED_LIGHT_JUMPING,
    }
)

#: Violations that need pixels to decide. Present in a manifest entry, they are
#: reported as unverified rather than assumed.
FOOTAGE_DEPENDENT: frozenset[ViolationType] = frozenset(
    {ViolationType.NO_HELMET, ViolationType.TRIPLE_RIDING}
)


def uncalibrated_scene(name: str, *, width: int = 1920, height: int = 1080) -> Any:
    """The scene an uncalibrated clip actually gets: frame-correct, claiming nothing.

    Mirrors ``ProcessingService.provisional_scene``: one full-frame ROI zone, no
    direction, no stop line, no signal group. Built through the real ``build_scene``
    so this asks the production authoring path rather than a hand-made stub.

    The frame size is irrelevant to the structural question (no geometry is compared
    against it here) but must be *valid*, since ``SceneConfig`` rejects points outside
    the frame.
    """

    draft = SceneDraft(
        scene_name=name,
        camera_id="cam-" + name,
        frame_width=width,
        frame_height=height,
        zones=(
            ZoneDraft(
                zone_id="frame",
                zone_type=ZoneType.ROI,
                polygon=full_frame_polygon(width, height),
            ),
        ),
    )
    return build_scene(
        draft, scene_id="scene-" + name, calibration_source=CALIBRATION_SOURCE_AUTO
    )


def check_clip(filename: str, entry: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return ``(passes, failures, skips)`` for one manifest clip."""

    passes: list[str] = []
    failures: list[str] = []
    skips: list[str] = []

    if not entry.get("established", False):
        skips.append(filename + ": entry is established:false -- not evaluated")
        return passes, failures, skips

    scene = uncalibrated_scene(Path(filename).stem.replace(".", "-"))
    # The deployment half of the question is deliberately generous: assume the helmet
    # backend IS available, so a leak shows up rather than being masked by config.
    supported = set(supported_violations(scene, no_helmet_available=True))

    for name, expected in (entry.get("expected") or {}).items():
        try:
            violation = ViolationType(name)
        except ValueError:
            skips.append(filename + ": unknown violation " + repr(name) + " -- not evaluated")
            continue

        if violation in FOOTAGE_DEPENDENT:
            skips.append(
                filename
                + ": "
                + name
                + "="
                + repr(expected)
                + " depends on what is in the footage; NOT VERIFIED"
                " -- needs inference, not asserted here"
            )
            continue

        if violation not in STRUCTURAL:
            skips.append(filename + ": " + name + " has no structural guarantee -- skipped")
            continue

        if expected != 0:
            skips.append(
                filename
                + ": "
                + name
                + " expects "
                + repr(expected)
                + ", which is not a structural zero -- not evaluated"
            )
            continue

        if violation in supported:
            failures.append(
                filename
                + ": "
                + name
                + " expected 0 by construction, but an uncalibrated scene REPORTED IT"
                " AS SUPPORTED. A geometry rule is reachable without calibration --"
                " either a scene leaked between videos or the capability gate"
                " regressed. This is the exact failure the manifest exists to catch."
            )
        else:
            passes.append(
                filename + ": " + name + "=0 guaranteed (rule not buildable uncalibrated)"
            )

    for name in entry.get("unestablished") or {}:
        skips.append(
            filename + ": " + name + " is unestablished (null) -- a labelling TODO, not a target"
        )

    if "calibrated_variant" in entry:
        variant = entry["calibrated_variant"] or {}
        skips.append(
            filename
            + ": calibrated_variant expects "
            + repr(variant.get("expected"))
            + " with established="
            + repr(variant.get("established"))
            + " -- not evaluated here; run it explicitly"
        )

    return passes, failures, skips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the T0 manifest's structural checks.")
    parser.add_argument("--verbose", action="store_true", help="list skipped expectations too")
    args = parser.parse_args(argv)

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest.get("clips") or {}

    all_passes: list[str] = []
    all_failures: list[str] = []
    all_skips: list[str] = []
    for filename, entry in clips.items():
        passes, failures, skips = check_clip(filename, entry)
        all_passes += passes
        all_failures += failures
        all_skips += skips

    executed = len(all_passes) + len(all_failures)
    print("T0 manifest: " + str(len(clips)) + " clips")
    print("")
    print("EXECUTED (" + str(executed) + " established structural expectations)")
    for line in all_passes:
        print("  PASS  " + line)
    for line in all_failures:
        print("  FAIL  " + line)
    if executed == 0:
        print("  (none)")

    print("")
    print("NOT EVALUATED (" + str(len(all_skips)) + ") -- by design; see the module docstring")
    if args.verbose:
        for line in all_skips:
            print("  SKIP  " + line)
    else:
        print("  (re-run with --verbose to list them)")

    print("")
    print(
        "result: "
        + str(len(all_passes))
        + " passed, "
        + str(len(all_failures))
        + " failed, "
        + str(len(all_skips))
        + " not evaluated"
    )
    if all_failures:
        print("")
        print("A structural expectation failed. This is never a tuning question: it means")
        print("a rule became reachable without calibration, or scene state leaked.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
