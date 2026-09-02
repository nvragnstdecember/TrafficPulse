"""Multi-violation overlay rendering, end to end through the app (R6).

The R6 claim: a run that confirms several violation types renders **all** of their
overlays into one annotated video and into the evidence stills -- none suppressed,
none replaced by another's.

Asserted on the composed scene rather than only on an HTTP 200: a rendered MP4 that
plays is not evidence that anything was drawn on it. The compositor the render
actually uses is built here from the finished engine, and its per-frame scene is
inspected element by element.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _app_helpers import StubEngineProvider, make_client, make_config
from _stopping_fixtures import (
    illegal_stopping_test_scene,
    stopping_detector_config,
    write_illegal_stopping_clip,
)
from fastapi.testclient import TestClient

from trafficpulse.app.overlay_video import OVERLAY_REGISTRY, build_job_compositor
from trafficpulse.detector import RawDetection, StubDetector
from trafficpulse.overlay.metadata import OverlayBanner, OverlayBox, OverlayLink
from trafficpulse.overlay.registry import OverlayCompositor, OverlayFrameRef, OverlayScene

# One vehicle: descends for 2.0 s (wrong-way against the scene's "north" legal
# direction), then holds inside the no-stopping zone. One track, two violations,
# two overlays that must coexist.
FRAMES = 70
_DESCENT_FRAMES = 20


def _box(frame_index: int) -> tuple[float, float, float, float]:
    bottom = 40.0 + frame_index * 8.0 if frame_index < _DESCENT_FRAMES else 200.0
    return (140.0, bottom - 30.0, 180.0, bottom)


def _detector() -> StubDetector:
    return StubDetector(
        per_frame={
            i: (RawDetection(label="car", score=0.9, box=_box(i)),) for i in range(FRAMES)
        }
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(illegal_stopping_test_scene().model_dump_json(), encoding="utf-8")
    config = make_config(tmp_path / "storage", scene_path=scene_path, default_rules=())
    return make_client(
        tmp_path / "storage",
        config=config,
        provider=StubEngineProvider(_detector, detector_config=stopping_detector_config()),
    )


def _run(client: TestClient, tmp_path: Path) -> tuple[str, str]:
    clip = write_illegal_stopping_clip(tmp_path / "multi.mp4", frames=FRAMES)
    video_id = client.post(
        "/api/video/upload",
        files={"file": ("multi.mp4", clip.read_bytes(), "video/mp4")},
    ).json()["video_id"]
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    assert client.get(f"/api/process/{job_id}").json()["status"] == "succeeded"
    return video_id, job_id


def test_the_registry_backs_the_application_driver() -> None:
    """The registry is the dispatch mechanism, not dead code.

    ``helmet_analysis`` is in this set but is **not** a violation: it is the
    perception-only overlay, registered against its own observer type so an analysis
    run can never be drawn through the no-helmet provider. It is asserted here for the
    same reason the five violations are -- the set is the registry's contract, and a
    kind appearing or vanishing from it silently is exactly what this test exists to
    catch.
    """

    assert OVERLAY_REGISTRY.known_kinds() == {
        "wrong_way",
        "illegal_stopping",
        "no_helmet",
        "triple_riding",
        "red_light_jumping",
        "helmet_analysis",
    }


def test_a_multi_violation_run_composes_every_confirmed_violations_overlay(
    client: TestClient, tmp_path: Path
) -> None:
    """Both violations reach the annotated video, drawn from their own metadata."""

    video_id, job_id = _run(client, tmp_path)
    events = client.get("/api/events", params={"video_id": video_id}).json()["items"]
    types = {item["violation_type"] for item in events}
    assert {"wrong_way", "illegal_stopping"} <= types, events

    # The compositor the render used, rebuilt from the same finished engine.
    job = _job_record(client, job_id)
    compositor = build_job_compositor(job.engine, _confirmed_events(client, video_id))
    assert compositor is not None
    kinds = [provider.violation_kind for provider in compositor.providers]
    assert kinds == sorted(kinds), "provider order must be deterministic"
    assert {"wrong_way", "illegal_stopping"} <= set(kinds)


def test_both_violations_draw_on_the_same_frame(client: TestClient, tmp_path: Path) -> None:
    """The load-bearing assertion: one frame, two violations, both represented.

    The clip drives wrong-way (frames 1-20) and then stops in the no-stopping zone,
    so frames 12-20 are the window where wrong-way has already been confirmed while
    illegal-stopping is still observing. Both providers must contribute there --
    neither suppressing, replacing, nor short-circuiting the other.
    """

    video_id, job_id = _run(client, tmp_path)
    job = _job_record(client, job_id)
    compositor = build_job_compositor(job.engine, _confirmed_events(client, video_id))
    assert compositor is not None

    scene = _scene_at(compositor, 19)

    # Two boxes for the one track: each violation drew the vehicle from its own
    # capture, with its own caption, rather than one overwriting the other.
    boxes = [e for e in scene.elements if isinstance(e, OverlayBox)]
    assert len(boxes) == 2
    assert {b.key for b in boxes} == {"iou-1"}
    captions = {b.caption.lines[0] for b in boxes if b.caption is not None}
    assert any("lane flow" in line for line in captions), captions  # wrong-way's
    assert any("no-stopping" in line or "Stopped" in line for line in captions), captions

    # Wrong-way has confirmed by here; its banner is present and is its own.
    banners = [e for e in scene.elements if isinstance(e, OverlayBanner)]
    assert [b.title for b in banners] == ["WRONG WAY"]

    # Both violations' *geometry* is drawn too: the wrong-way heading arrows and the
    # illegal-stopping zone ring share the frame.
    links = [e for e in scene.elements if isinstance(e, OverlayLink)]
    assert len(links) == 3  # 1 zone ring + 2 heading arrows (measured + legal)


def test_each_violation_owns_the_frames_its_rule_actually_observed(
    client: TestClient, tmp_path: Path
) -> None:
    """Overlays follow the evidence, and one violation never borrows another's frames.

    After the vehicle stops, wrong-way has nothing further to say -- its derivation
    produces no more observations -- so its overlay correctly disappears while
    illegal-stopping continues and confirms. An overlay that kept drawing wrong-way
    on a stationary vehicle would be claiming reasoning that never happened.
    """

    video_id, job_id = _run(client, tmp_path)
    job = _job_record(client, job_id)
    compositor = build_job_compositor(job.engine, _confirmed_events(client, video_id))
    assert compositor is not None

    late = _scene_at(compositor, FRAMES - 1)
    assert [
        b.title for b in late.elements if isinstance(b, OverlayBanner)
    ] == ["ILLEGAL STOPPING"]
    boxes = [e for e in late.elements if isinstance(e, OverlayBox)]
    assert len(boxes) == 1  # only the violation still being reasoned about
    assert boxes[0].caption is not None
    assert "Stopped" in boxes[0].caption.lines[0]


def test_the_annotated_video_is_produced_and_served(
    client: TestClient, tmp_path: Path
) -> None:
    """The artifact itself: rendered, non-trivial, and downloadable."""

    _, job_id = _run(client, tmp_path)
    status = client.get(f"/api/process/{job_id}").json()
    assert status["overlay_status"] == "ready"
    assert status["overlay_available"] is True

    response = client.get(f"/api/process/{job_id}/overlay")
    assert response.status_code == 200
    overlay = tmp_path / "storage" / "overlays" / f"{job_id}.mp4"
    assert overlay.is_file() and overlay.stat().st_size > 0


def test_every_event_keeps_its_own_evidence(client: TestClient, tmp_path: Path) -> None:
    """Rendering one violation's overlay must not cost another its evidence."""

    video_id, job_id = _run(client, tmp_path)
    assert client.get(f"/api/process/{job_id}").json()["evidence_status"] == "ready"

    for item in client.get("/api/events", params={"video_id": video_id}).json()["items"]:
        manifest = client.get(f"/api/evidence/{item['event_id']}")
        assert manifest.status_code == 200, manifest.text
        body = manifest.json()
        assert body["event_id"] == item["event_id"]
        assert body["trigger_frame"] is not None


# --- helpers -----------------------------------------------------------------------
def _scene_at(compositor: OverlayCompositor, frame_index: int) -> OverlayScene:
    """The fused scene one frame of the annotated video would be drawn from."""

    return compositor.scene_for(
        OverlayFrameRef(
            camera_id="cam-synthetic-01",
            frame_index=frame_index,
            media_seconds=frame_index / 10.0,
            width=320,
            height=240,
        )
    )



def _job_record(client: TestClient, job_id: str) -> object:
    """The live :class:`JobRecord`, for the engine that finished the run.

    Reaches into the wired context because the finished engine is deliberately not
    on the HTTP surface -- it is an in-process object, and the API exposes its
    *results*. A test that wants to inspect the composed overlay scene (rather than
    trust a rendered MP4) has to start from the same engine the render did.
    """

    context = client.app.state.context  # type: ignore[attr-defined]
    record = context.processing._jobs.get(job_id)
    assert record is not None and record.engine is not None
    return record


def _confirmed_events(client: TestClient, video_id: str) -> list[object]:
    from trafficpulse.contracts import ConfirmedEvent

    listed = client.get("/api/events", params={"video_id": video_id}).json()["items"]
    return [
        ConfirmedEvent.model_validate(client.get(f"/api/events/{item['event_id']}").json())
        for item in listed
    ]
