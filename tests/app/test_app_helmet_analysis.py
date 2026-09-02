"""Helmet analysis and deployment posture through the HTTP application.

The integration hop for the rule/analysis split: a deployment configured to *classify*
rather than *enforce* runs a real job, reports per-rider helmet state through
``/api/process/{job}/helmet-analysis``, and produces **no** confirmed helmet event --
while ``/api/system/posture`` states, in words, why.

The turban-blindness of a binary backend is exercised end to end here rather than only
at the rule registry, because that is where it previously would have bitten: the derived
rule set is what an uncalibrated upload runs, so a guard that fires at engine-build time
would have failed every single job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _app_helpers import StubEngineProvider, make_client, make_config
from _helmet_fixtures import helmet_detector_config, scripted_helmet_classifier
from _triple_fixtures import scripted_rider_count_detector, write_triple_riding_clip

from trafficpulse.app.registry import JobStatus
from trafficpulse.classifier import (
    HelmetClassifier,
    RawHelmetPrediction,
    ResNetHelmetConfig,
    ZeroShotHelmetConfig,
)
from trafficpulse.engine import HelmetAnalysisConfig, NoHelmetRuleConfig


class _BinaryClassifier(HelmetClassifier):
    """Declares a binary vocabulary, exactly as the trained P4-U5 ResNet-50 does."""

    @property
    def supported_labels(self) -> frozenset[str]:
        return frozenset({"helmet", "no_helmet"})

    def classify(self, crops: Any) -> Any:
        return tuple(RawHelmetPrediction(label="no_helmet", score=0.93) for _ in crops)


class _ExplodingClassifier(HelmetClassifier):
    """A backend that fails at inference time, the way a bad checkpoint would."""

    def classify(self, crops: Any) -> Any:
        raise RuntimeError("boom: the helmet backend failed")


#: The declaration under test, as a module singleton so it is not constructed in a
#: default argument (and so every test that omits it shares one identical value).
ANALYSIS = HelmetAnalysisConfig()


def _client(
    tmp_path: Path,
    *,
    riders: int = 1,
    classifier: HelmetClassifier | None = None,
    analysis: HelmetAnalysisConfig | None = ANALYSIS,
    default_rules: tuple[Any, ...] = (),
):
    provider = StubEngineProvider(
        detector_factory=lambda: scripted_rider_count_detector(riders=riders),
        detector_config=helmet_detector_config(),
        classifier=classifier if classifier is not None else scripted_helmet_classifier(),
    )
    config = make_config(tmp_path, default_rules=default_rules).model_copy(
        update={"helmet_analysis": analysis}
    )
    return make_client(tmp_path, provider=provider, config=config)


def _run(client: Any, tmp_path: Path, *, riders: int = 1) -> str:
    clip = write_triple_riding_clip(tmp_path / "riders.mp4", riders=riders)
    video_id = client.post(
        "/api/video/upload", files={"file": ("riders.mp4", clip.read_bytes(), "video/mp4")}
    ).json()["video_id"]
    created = client.post("/api/process", json={"video_id": video_id})
    assert created.status_code == 202, created.text
    return created.json()["job_id"]


# --- the analysis endpoint ------------------------------------------------------------
def test_a_run_reports_per_rider_helmet_state_and_no_helmet_event(tmp_path: Path) -> None:
    """The headline: classification is served, and nothing is confirmed."""

    client = _client(tmp_path)
    job_id = _run(client, tmp_path)

    assert client.get(f"/api/process/{job_id}").json()["status"] == JobStatus.SUCCEEDED.value

    analysis = client.get(f"/api/process/{job_id}/helmet-analysis")
    assert analysis.status_code == 200, analysis.text
    body = analysis.json()
    assert body["job_id"] == job_id
    assert body["riders_observed"] >= 1
    assert body["motorcycles_associated"] >= 1
    assert len(body["riders"]) == body["riders_observed"]

    # No helmet violation reached the event store, which remains the only source.
    events = client.get("/api/events").json()["items"]
    assert all(event["violation_type"] != "no_helmet" for event in events)


def test_the_payload_states_the_enforcement_posture_it_was_read_under(
    tmp_path: Path,
) -> None:
    """A client cannot render the readings without also holding the disclaimer.

    Configured the way the demo launcher configures it -- a declared backend plus a
    declared analysis -- because the posture describes the *deployment*, and a test
    whose config declares no backend would be asserting a different situation.
    """

    provider = StubEngineProvider(
        detector_factory=lambda: scripted_rider_count_detector(riders=1),
        detector_config=helmet_detector_config(),
        classifier=scripted_helmet_classifier(),
    )
    config = make_config(tmp_path).model_copy(
        update={
            "helmet_classifier": ResNetHelmetConfig(
                checkpoint=tmp_path / "best.pt", abstain_below=0.8
            ),
            "helmet_analysis": HelmetAnalysisConfig(),
        }
    )
    client = make_client(tmp_path, provider=provider, config=config)
    job_id = _run(client, tmp_path)

    body = client.get(f"/api/process/{job_id}/helmet-analysis").json()

    assert body["enforcement"] == "disabled"
    assert body["enforcement"] == client.get("/api/system/posture").json()["helmet_enforcement"]


def test_a_shared_motorcycle_reports_every_rider_unresolved(tmp_path: Path) -> None:
    """No driver is invented, and the count is reported honestly."""

    client = _client(tmp_path, riders=3)
    job_id = _run(client, tmp_path, riders=3)

    body = client.get(f"/api/process/{job_id}/helmet-analysis").json()

    assert body["multi_rider_riders"] == body["riders_observed"] >= 2
    assert body["eligible_riders"] == 0
    assert {rider["enforcement"] for rider in body["riders"]} == {"multi_rider_unresolved"}
    assert all(rider["rider_count"] >= 2 for rider in body["riders"])


def test_a_run_without_a_configured_analysis_reports_none(tmp_path: Path) -> None:
    """Not an error: a deployment that configured none is working as intended."""

    client = _client(tmp_path, analysis=None)
    job_id = _run(client, tmp_path)

    response = client.get(f"/api/process/{job_id}/helmet-analysis")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "analysis_not_available"


def test_an_unknown_job_is_a_clean_404(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/process/job-does-not-exist/helmet-analysis")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "job_not_found"


def test_an_analysis_is_not_added_beside_the_no_helmet_rule(tmp_path: Path) -> None:
    """Declaring both would classify every rider twice per frame for one run."""

    client = _client(tmp_path, default_rules=(NoHelmetRuleConfig(),))
    job_id = _run(client, tmp_path)

    assert client.get(f"/api/process/{job_id}/helmet-analysis").status_code == 404


# --- resilience -------------------------------------------------------------------------
def test_a_failing_classifier_fails_the_job_cleanly_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """A backend failure is a job outcome, never an unhandled server error."""

    client = _client(tmp_path, classifier=_ExplodingClassifier())
    job_id = _run(client, tmp_path)

    status = client.get(f"/api/process/{job_id}").json()
    assert status["status"] == JobStatus.FAILED.value
    assert "boom" in (status["error"] or "")
    # The rest of the API keeps serving.
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/system/posture").status_code == 200


# --- the turban guard, end to end ---------------------------------------------------------
def test_a_binary_backend_never_gets_a_no_helmet_rule_derived_for_it(
    tmp_path: Path,
) -> None:
    """The regression this guards: the guard firing on every upload.

    A turban-blind backend cannot build the no-helmet rule. If the scene-derived rule
    set still offered it, every job would fail at engine build -- the safety guard
    turning into a crash. The derivation asks the guard, so the rule is never offered.
    """

    provider = StubEngineProvider(
        detector_factory=lambda: scripted_rider_count_detector(riders=1),
        detector_config=helmet_detector_config(),
        classifier=_BinaryClassifier(),
    )
    config = make_config(tmp_path).model_copy(
        update={
            "helmet_classifier": ResNetHelmetConfig(
                checkpoint=tmp_path / "best.pt", abstain_below=0.8
            ),
            "helmet_analysis": HelmetAnalysisConfig(),
        }
    )
    client = make_client(tmp_path, provider=provider, config=config)

    job_id = _run(client, tmp_path)

    status = client.get(f"/api/process/{job_id}").json()
    assert status["status"] == JobStatus.SUCCEEDED.value, status
    # Classification still happened -- refused enforcement, not lost perception.
    assert client.get(f"/api/process/{job_id}/helmet-analysis").json()["riders_observed"] >= 1


# --- posture --------------------------------------------------------------------------------
def test_posture_reports_a_binary_backend_as_turban_incapable(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(
        update={
            "helmet_classifier": ResNetHelmetConfig(
                checkpoint=tmp_path / "best.pt", abstain_below=0.8
            ),
            "helmet_analysis": HelmetAnalysisConfig(),
        }
    )
    client = make_client(tmp_path, provider=StubEngineProvider(), config=config)

    body = client.get("/api/system/posture").json()

    assert body["turban_capable"] is False
    assert body["helmet_backend"] == "ResNetHelmetConfig"
    assert "turban" not in body["helmet_backend_labels"]
    assert body["helmet_enforcement"] == "disabled"
    states = {item["component_id"]: item["state"] for item in body["components"]}
    assert states["turban_exemption"] == "unavailable"
    assert states["driver_attribution"] == "limited"
    assert states["helmet_classification"] == "active"


def test_posture_never_reports_helmet_enforcement_as_active(tmp_path: Path) -> None:
    """No configuration of this system currently earns that word.

    Even the turban-capable zero-shot backend, with no analysis declared and the rule
    therefore buildable, reports ``experimental`` -- the rule's temporal semantics have
    never been evaluated against real per-frame output.
    """

    config = make_config(tmp_path).model_copy(
        update={"helmet_classifier": ZeroShotHelmetConfig(checkpoint="some/clip")}
    )
    client = make_client(tmp_path, provider=StubEngineProvider(), config=config)

    body = client.get("/api/system/posture").json()

    assert body["turban_capable"] is True
    assert body["helmet_enforcement"] == "experimental"
    assert body["helmet_enforcement"] != "active"


def test_posture_reports_an_unconfigured_deployment_honestly(tmp_path: Path) -> None:
    client = make_client(
        tmp_path, provider=StubEngineProvider(), config=make_config(tmp_path)
    )

    body = client.get("/api/system/posture").json()

    assert body["helmet_backend"] is None
    assert body["helmet_backend_labels"] == []
    assert body["helmet_enforcement"] == "unavailable"


def test_every_posture_component_explains_itself(tmp_path: Path) -> None:
    """A bare status word invites a viewer to assume a bug where there is a guard."""

    client = make_client(
        tmp_path, provider=StubEngineProvider(), config=make_config(tmp_path)
    )

    for component in client.get("/api/system/posture").json()["components"]:
        assert component["detail"].strip().endswith("."), component
        assert len(component["detail"]) > 40, component
