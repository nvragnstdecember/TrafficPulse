"""Demo reliability check: drive the real demo path over real footage.

    .venv/Scripts/python demo/run_demo_smoke.py --frames 30

**This is not a measurement and must never be quoted as one.** It runs a handful of
short, contiguous segments cut from the clips P4-U10 already audited, through the
*actual* ``serve_demo`` composition -- real RT-DETR, real association, real head-crop
geometry, the real trained ResNet-50 -- and reports what came back. Its questions are
operational, not scientific:

* does the whole path run to completion on each clip, or does something crash?
* does a clip with no motorcycles produce no riders (rather than inventing some)?
* do multi-rider motorcycles come back marked unresolved?
* does the annotated video render?
* does anything claim a violation it should not?

Nothing here is tuned, and no threshold, label or model is changed on the basis of
anything it prints. The segments are short by design: this exercises the plumbing, and
a short segment exercises the plumbing exactly as well as a long one while costing
minutes instead of hours on CPU.

Why it trims first
------------------
The source clips are 13-60 seconds of 1080p. RT-DETR on CPU runs at roughly 2-3
seconds per frame, so a full clip is hours. Each segment is re-encoded to a small mp4
of ``--frames`` **contiguous** frames -- contiguous because the tracker needs real
frame-to-frame continuity, and sub-sampling would silently change the thing under test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import av

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Clip:
    """One source clip, and what P4-U10 already established about it."""

    name: str
    path: Path
    expectation: str
    """What a correct system should do here -- written down *before* the run, so the
    output is read against a stated expectation rather than rationalised after."""


CLIPS = (
    Clip(
        "gangtok-congestion",
        REPO_ROOT / "test-videos/edge-cases/congestion/congestion_001.webm",
        "India, handheld portrait. Sparse motorcycles; many frames contain none.",
    ),
    Clip(
        "raxaul-congestion",
        REPO_ROOT / "test-videos/edge-cases/congestion/congestion_002.webm",
        "India, dense level-crossing crowd. P4-U10 found 81% of crops multi-rider; "
        "riders here should come back UNRESOLVED, not attributed.",
    ),
    Clip(
        "chiangmai-intersection",
        REPO_ROOT / "test-videos/normal-traffic/clean_001.ogv",
        "Thailand, elevated static -- the CCTV-like view. Small heads (median 28px), "
        "which is the worst measured stratum for the classifier.",
    ),
    Clip(
        "contraflow-roundabout",
        REPO_ROOT / "test-videos/wrong-way/wrongway_001.ogv",
        "Australia. Contains NO motorcycles: the correct outcome is zero riders.",
    ),
)


def trim(source: Path, target: Path, *, frames: int, skip: int) -> Path | None:
    """Re-encode ``frames`` contiguous frames of ``source``, starting after ``skip``."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        rate = int(stream.average_rate or 25)
        # Re-encode through an ndarray round-trip rather than remuxing the decoded
        # frames: the sources are three different containers/codecs, and handing their
        # frames (with their own time bases and pixel formats) straight to one encoder
        # is what fails. Dimensions are forced even, which mpeg4 requires.
        width = stream.codec_context.width - (stream.codec_context.width % 2)
        height = stream.codec_context.height - (stream.codec_context.height % 2)
        out = av.open(str(target), "w")
        out_stream = out.add_stream("mpeg4", rate=rate)
        out_stream.width, out_stream.height, out_stream.pix_fmt = width, height, "yuv420p"
        written = 0
        for index, frame in enumerate(container.decode(video=0)):
            if index < skip:
                continue
            if written >= frames:
                break
            array = frame.to_ndarray(format="rgb24")[:height, :width]
            for packet in out_stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                out.mux(packet)
            written += 1
        for packet in out_stream.encode():
            out.mux(packet)
        out.close()
    return target if written else None


def _await_job(client: object, job_id: str, *, timeout_seconds: float) -> dict:
    """Poll a job to a terminal state, then wait for its overlay to settle.

    Exactly what the workspace does. The overlay is deliberately waited on: it is a
    second decode+encode pass that starts only after the run is durably persisted, and
    a smoke test that stopped at ``succeeded`` would report "no annotated video" for
    every clip simply by asking too early.
    """

    deadline = time.monotonic() + timeout_seconds
    status: dict = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/process/{job_id}").json()  # type: ignore[attr-defined]
        terminal = status["status"] in {"succeeded", "failed", "cancelled"}
        settled = status.get("overlay_status") in {"ready", "none", "failed"}
        if terminal and settled:
            return status
        time.sleep(1.0)
    status["timed_out"] = True
    return status


def run_clip(
    client: object,
    clip_path: Path,
    name: str,
    *,
    timeout_seconds: float,
    keep_dir: Path | None = None,
) -> dict:
    """Upload, process, and read back everything the workspace would read.

    Times the two phases a presenter actually waits through -- the run itself, and the
    annotated-video render that follows it -- because "will a live upload finish while
    the room watches" is the operational question this script exists to answer.
    """

    upload = client.post(  # type: ignore[attr-defined]
        "/api/video/upload",
        files={"file": (f"{name}.mp4", clip_path.read_bytes(), "video/mp4")},
    )
    if upload.status_code != 201:
        return {"stage": "upload", "status": upload.status_code, "body": upload.text[:400]}
    video_id = upload.json()["video_id"]

    started = time.monotonic()
    created = client.post("/api/process", json={"video_id": video_id})  # type: ignore[attr-defined]
    if created.status_code != 202:
        return {"stage": "submit", "status": created.status_code, "body": created.text[:400]}
    job_id = created.json()["job_id"]

    # Production runs jobs on a background thread, so this polls exactly as the
    # workspace does. The overlay render is a second decode+encode pass that finishes
    # after the job goes terminal, so the wait continues until it settles too.
    status = _await_job(client, job_id, timeout_seconds=timeout_seconds)
    elapsed = time.monotonic() - started
    analysis = client.get(f"/api/process/{job_id}/helmet-analysis")  # type: ignore[attr-defined]
    events = client.get("/api/events", params={"video_id": video_id, "job_id": job_id})  # type: ignore[attr-defined]
    overlay = client.get(f"/api/process/{job_id}/overlay")  # type: ignore[attr-defined]

    frames = status.get("frames_processed") or 0
    result: dict = {
        "stage": "complete",
        "job_status": status["status"],
        "error": status.get("error"),
        "frames_processed": frames,
        "seconds_total": round(elapsed, 1),
        "seconds_per_frame": round(elapsed / frames, 2) if frames else None,
        "overlay_status": status.get("overlay_status"),
        "overlay_http": overlay.status_code,
        "overlay_bytes": len(overlay.content) if overlay.status_code == 200 else 0,
        "events": [item["violation_type"] for item in events.json().get("items", [])],
        "analysis_http": analysis.status_code,
    }
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip_path, keep_dir / f"{name}.mp4")
        if overlay.status_code == 200:
            annotated = keep_dir / f"{name}.annotated.mp4"
            annotated.write_bytes(overlay.content)
            result["annotated_path"] = str(annotated)
    if analysis.status_code == 200:
        body = analysis.json()
        result["analysis"] = {
            key: body[key]
            for key in (
                "enforcement",
                "frames_observed",
                "riders_observed",
                "motorcycles_associated",
                "multi_rider_riders",
                "eligible_riders",
                "unresolved_riders",
                "abstained_riders",
                "unstable_riders",
                "gate_abstentions",
                "label_counts",
                "enforcement_counts",
            )
        }
        result["riders"] = [
            {
                "rider": rider["rider_track_id"],
                "bike": rider["motorcycle_track_id"],
                "n": rider["rider_count"],
                "label": rider["helmet_state"],
                "conf": rider["confidence"],
                "agree": round(rider["agreement"], 2),
                "raw_flips": rider["raw_label_flips"],
                "stab_flips": rider["stabilized_label_flips"],
                "head_px": rider["median_head_height_px"],
                "status": rider["enforcement"],
            }
            for rider in body["riders"]
        ]
    return result


def check_readiness(posture: object, clips: dict) -> list[str]:
    """The properties that must hold before this is shown to anyone.

    Every one of these is a *safety* property rather than a quality one, which is why
    they are asserted here instead of merely printed. A demo that quietly started
    emitting helmet violations, or that attributed a rider on a shared motorcycle,
    would look **better** on stage and be exactly the thing this project must not do --
    so the readiness check has to fail on it rather than leave it to be noticed.

    Returns the list of violated properties; empty means ready.
    """

    failures: list[str] = []
    states = {c.component_id: c.state.value for c in posture.components}  # type: ignore[attr-defined]
    if states.get("turban_exemption") != "unavailable":
        failures.append(
            f"turban exemption must remain unavailable on a binary backend, "
            f"got {states.get('turban_exemption')!r}"
        )
    if states.get("helmet_enforcement") != "disabled":
        failures.append(
            f"helmet enforcement must be disabled in the demo composition, "
            f"got {states.get('helmet_enforcement')!r}"
        )
    if getattr(posture, "turban_capable", True):
        failures.append("the demo backend must not claim turban capability")

    for name, outcome in clips.items():
        if outcome.get("stage") in {"missing"}:
            continue
        if outcome.get("stage") != "complete":
            failures.append(f"{name}: run did not complete ({outcome.get('stage')})")
            continue
        if outcome.get("job_status") != "succeeded":
            failures.append(f"{name}: job {outcome.get('job_status')} - {outcome.get('error')}")
        if "no_helmet" in outcome.get("events", []):
            failures.append(f"{name}: a helmet violation was emitted; enforcement is off")
        for rider in outcome.get("riders", []):
            if rider["n"] > 1 and rider["status"] != "multi_rider_unresolved":
                failures.append(
                    f"{name}: rider {rider['rider']} shares a motorcycle with "
                    f"{rider['n'] - 1} other(s) but is {rider['status']!r}, not unresolved"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30, help="contiguous frames per clip")
    parser.add_argument("--skip", type=int, default=60, help="source frames to skip first")
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds to wait per clip"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--only", default=None, help="run one clip by name")
    parser.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="keep the trimmed clip and its annotated output in this directory",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=None,
        help="seed this storage root instead of a throwaway one, so the processed "
        "videos remain in the library after the run (the demo fallback)",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="tp-demo-smoke-"))
    import os

    # A persistent root turns this from a check into a *seeding* run: the uploads,
    # the runs and the annotated videos survive, so a later server started on the
    # same root serves them from the library. Nothing is fabricated by doing this --
    # they are the outputs this run actually produced.
    storage = args.storage if args.storage is not None else workdir / "storage"
    os.environ["TRAFFICPULSE_APP_STORAGE"] = str(storage)

    import serve_demo
    from fastapi.testclient import TestClient

    from trafficpulse.app import create_app
    from trafficpulse.app.posture import describe

    config = serve_demo.build_config()
    posture = describe(config)
    print("=== deployment posture ===")
    for component in posture.components:
        print(f"  {component.label:<32} {component.state.value.upper()}")
    print(f"  backend: {posture.helmet_backend} emits {list(posture.helmet_backend_labels)}")
    print()

    report: dict = {
        "posture": {c.component_id: c.state.value for c in posture.components},
        "frames_per_clip": args.frames,
        "clips": {},
    }
    with TestClient(create_app(config)) as client:
        for clip in CLIPS:
            if args.only and clip.name != args.only:
                continue
            if not clip.path.is_file():
                report["clips"][clip.name] = {"stage": "missing", "path": str(clip.path)}
                print(f"[{clip.name}] SKIPPED - {clip.path} is not present")
                continue
            print(f"[{clip.name}] {clip.expectation}")
            trimmed = trim(
                clip.path, workdir / f"{clip.name}.mp4", frames=args.frames, skip=args.skip
            )
            if trimmed is None:
                report["clips"][clip.name] = {"stage": "trim", "error": "no frames decoded"}
                print("  could not decode any frames")
                continue
            try:
                outcome = run_clip(
                    client,
                    trimmed,
                    clip.name,
                    timeout_seconds=args.timeout,
                    keep_dir=args.keep,
                )
            except Exception as exc:  # noqa: BLE001 - the point is to see a crash, not raise
                outcome = {"stage": "exception", "error": repr(exc)}
            report["clips"][clip.name] = outcome
            print("  " + json.dumps({k: v for k, v in outcome.items() if k != "riders"}))
            for rider in outcome.get("riders", []):
                print("    " + json.dumps(rider))
            print()

    failures = check_readiness(posture, report["clips"])
    report["readiness"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    print("=== readiness ===")
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
    else:
        print("  PASS  enforcement disabled, turban unavailable, every shared "
              "motorcycle unresolved, no helmet violation emitted, no crash")
    print()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"report written to {args.out}")
    shutil.rmtree(workdir, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
