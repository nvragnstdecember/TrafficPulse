"""End-to-end no-helmet detection through the HTTP application (v1.1 U2).

Proves the last integration hop: a no-helmet video, uploaded and processed by the
*app* (not just the offline pipeline), surfaces a ``no_helmet`` ConfirmedEvent and
its evidence through the same ``/api/events`` and ``/api/evidence`` endpoints the
React workspace already consumes.

Everything under test is reused: the H6 engine's no-helmet rule wiring, the P4
observation/reasoner/classifier seams, persistence, and evidence. The only new
production wiring is the app injecting a ``HelmetClassifier`` into the engine
(``RealEngineProvider``); here the framework-free ``StubHelmetClassifier`` stands
in, exactly as the stub detector stands in for RT-DETR, so the whole path is
deterministic and GPU-free. Coverage of the smoothing/reasoner internals lives in
``tests/pipeline`` and is not repeated -- these tests assert the *app integration*.
"""

from __future__ import annotations

from pathlib import Path

from _app_helpers import StubEngineProvider, make_client, make_config
from _helmet_fixtures import (
    HELMET,
    NO_HELMET,
    helmet_detector_config,
    scripted_helmet_classifier,
    scripted_rider_detector,
    write_no_helmet_clip,
)

from trafficpulse.app.registry import JobStatus
from trafficpulse.classifier import RawHelmetPrediction
from trafficpulse.engine import NoHelmetRuleConfig


def _no_helmet_client(tmp_path: Path, *, prediction: RawHelmetPrediction, with_classifier: bool):
    """A client whose engine runs the no-helmet rule over the scripted fixture.

    The scripted rider detector + helmet classifier replay a caller-authored
    scenario (a rider astride a motorcycle, labelled ``prediction`` every frame);
    the example scene supplies the ``no_helmet`` block (min_persistence 1.0 s).
    """

    provider = StubEngineProvider(
        detector_factory=scripted_rider_detector,
        detector_config=helmet_detector_config(),
        classifier=scripted_helmet_classifier(prediction) if with_classifier else None,
    )
    config = make_config(tmp_path, default_rules=(NoHelmetRuleConfig(),))
    return make_client(tmp_path, provider=provider, config=config)


def _upload_clip(client: object, tmp_path: Path) -> str:
    clip = write_no_helmet_clip(tmp_path / "no_helmet.mp4")
    response = client.post(  # type: ignore[attr-defined]
        "/api/video/upload", files={"file": ("no_helmet.mp4", clip.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 201, response.text
    return response.json()["video_id"]


def _process(client: object, video_id: str) -> dict:
    created = client.post("/api/process", json={"video_id": video_id})  # type: ignore[attr-defined]
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    status = client.get(f"/api/process/{job_id}").json()  # type: ignore[attr-defined]
    return status


# --- positive: a no-helmet rider becomes a no_helmet event ---------------------
def test_no_helmet_video_produces_a_no_helmet_event(tmp_path: Path) -> None:
    client = _no_helmet_client(tmp_path, prediction=NO_HELMET, with_classifier=True)
    video_id = _upload_clip(client, tmp_path)

    status = _process(client, video_id)
    assert status["status"] == JobStatus.SUCCEEDED.value
    assert status["event_count"] == 1

    listing = client.get("/api/events", params={"video_id": video_id}).json()
    assert listing["total"] == 1
    summary = listing["items"][0]
    assert summary["violation_type"] == "no_helmet"
    assert summary["video_id"] == video_id
    # The event names both the rider track and the motorcycle track it rode.
    assert len(summary["track_ids"]) >= 2


def test_no_helmet_event_detail_and_evidence(tmp_path: Path) -> None:
    client = _no_helmet_client(tmp_path, prediction=NO_HELMET, with_classifier=True)
    video_id = _upload_clip(client, tmp_path)
    _process(client, video_id)

    event_id = client.get("/api/events", params={"video_id": video_id}).json()["items"][0][
        "event_id"
    ]

    detail = client.get(f"/api/events/{event_id}").json()
    assert detail["violation_type"] == "no_helmet"
    assert detail["trigger_at"]  # a media-time trigger instant
    assert detail["confidence"]  # a populated confidence breakdown
    assert detail["track_ids"]

    evidence = client.get(f"/api/evidence/{event_id}").json()
    assert evidence["event_id"] == event_id
    # The evidence pipeline is reused verbatim: a real trigger frame reference,
    # anchored to the event, travels with the manifest (model provenance is empty
    # here only because the scripted stubs stamp no ModelRef — a stub property,
    # not an integration gap).
    assert evidence["trigger_frame"] is not None
    assert evidence["evidence_package_id"] and evidence["created_at"]


# --- suppression: a helmeted rider produces nothing ----------------------------
def test_helmeted_rider_produces_no_event(tmp_path: Path) -> None:
    client = _no_helmet_client(tmp_path, prediction=HELMET, with_classifier=True)
    video_id = _upload_clip(client, tmp_path)

    status = _process(client, video_id)
    assert status["status"] == JobStatus.SUCCEEDED.value
    assert status["event_count"] == 0
    assert client.get("/api/events", params={"video_id": video_id}).json()["total"] == 0


def test_uncertain_rider_abstains(tmp_path: Path) -> None:
    # An all-"uncertain" run abstains: uncertainty is never a confirmation.
    uncertain = RawHelmetPrediction(label="uncertain", score=0.5)
    client = _no_helmet_client(tmp_path, prediction=uncertain, with_classifier=True)
    video_id = _upload_clip(client, tmp_path)
    assert _process(client, video_id)["event_count"] == 0


# --- fail-fast: a no_helmet rule needs a classifier ----------------------------
def test_no_helmet_rule_without_a_classifier_is_400(tmp_path: Path) -> None:
    client = _no_helmet_client(tmp_path, prediction=NO_HELMET, with_classifier=False)
    video_id = _upload_clip(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_configuration"


# --- config: helmet classifier is code-configured, off by default --------------
def test_helmet_classifier_config_defaults_off(tmp_path: Path) -> None:
    assert make_config(tmp_path).helmet_classifier is None
