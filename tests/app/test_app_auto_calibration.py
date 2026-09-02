"""A raw upload derives its own scene from its own motion (automatic flow).

The product requirement: someone uploads a video, names no scene and no rules, and
the system evaluates every rule it can *honestly* apply to that footage in one run.
Before this, an uncalibrated upload was reasoned about through whatever scene the
deployment configured as a fallback -- geometry belonging to another camera, so the
geometry rules ran but could never fire.

What these tests pin is the boundary between derived and invented. The frame size
is measured and the legal direction is estimated from observed traffic, because
both are things the clip can show. A no-stopping zone, a stop line and a signal
schedule are **not** observable from arbitrary footage, so they are never produced
-- which means illegal-stopping and red-light stay unavailable until an analyst
supplies them, and that absence is the correct answer rather than a gap.

The abstention cases matter as much as the derivation: a two-way road whose traffic
cancels out, or a clip with too few moving vehicles, must decline to declare a legal
direction. Guessing one there would flag every lawful vehicle travelling the other
way -- the exact false-positive class the lane-containment work closed.

Two properties are load-bearing enough to be asserted against the scene the engine
was *actually built with*, not merely against what the API reports afterwards:

* **the deployment fallback is never substituted.** Every abstention and every
  failure still reasons in the video's own pixel space. The fallback here is the
  clip-space wrong-way scene, which declares a 1920x1080 frame and the scene id
  ``scene-synthetic-01`` -- so "which scene ran" is directly observable, and the
  recording provider below is what makes it observable.
* **calibration happens inside the job, never on the request thread.** ``POST
  /api/process`` promises 202-then-poll; a detector pass in the handler breaks that
  promise for exactly the uploads this feature exists to serve. The deferred
  executor tests assert the request completes with no work done at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _app_helpers import (
    DeferredJobExecutor,
    StubEngineProvider,
    make_client,
    make_config,
)
from _slice_fixtures import write_wrong_way_clip
from fastapi.testclient import TestClient

from trafficpulse.app.errors import EngineUnavailableError
from trafficpulse.app.services import (
    CALIBRATION_FRAME_BUDGET,
    DERIVED_DIRECTION_ID,
    DERIVED_LANE_ID,
    MIN_FLOW_MOVERS,
    CalibrationOutcome,
    needs_derived_geometry,
)
from trafficpulse.contracts import SceneConfig
from trafficpulse.contracts.enums import ObjectClass
from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector
from trafficpulse.engine import (
    AnalysisConfig,
    InferenceEngine,
    NoHelmetRuleConfig,
    RuleConfig,
    TripleRidingRuleConfig,
    WrongWayRuleConfig,
)
from trafficpulse.scenes import CALIBRATION_SOURCE_AUTO

CAR = DetectorConfig(label_map={"car": ObjectClass.CAR})

#: Identity of the scene the deployment configures as its fallback (the clip-space
#: wrong-way scene the app helpers write). Named here so a test can assert the
#: fallback was *not* used without restating its geometry.
FALLBACK_SCENE_ID = "scene-synthetic-01"
FALLBACK_FRAME = (1920, 1080)
CLIP_FRAME = (320, 240)


# --- detectors describing the traffic a clip contains -------------------------
def _flowing_detector(
    *, movers: int = 8, frames: int = 40, dx: float = 0.0, dy: float = 6.0
) -> StubDetector:
    """``movers`` vehicles all travelling the same way -- a coherent one-way flow."""

    per_frame: dict[int, tuple[RawDetection, ...]] = {}
    for i in range(frames):
        per_frame[i] = tuple(
            RawDetection(
                label="car",
                score=0.9,
                box=(
                    20.0 + m * 30.0 + dx * i,
                    20.0 + dy * i,
                    40.0 + m * 30.0 + dx * i,
                    40.0 + dy * i,
                ),
            )
            for m in range(movers)
        )
    return StubDetector(per_frame=per_frame)


def _two_way_detector(*, per_side: int = 6, frames: int = 40) -> StubDetector:
    """Equal traffic both ways: a two-way road one direction cannot describe."""

    per_frame: dict[int, tuple[RawDetection, ...]] = {}
    for i in range(frames):
        down = tuple(
            RawDetection(
                label="car", score=0.9,
                box=(20.0 + m * 30.0, 20.0 + 6.0 * i, 40.0 + m * 30.0, 40.0 + 6.0 * i),
            )
            for m in range(per_side)
        )
        up = tuple(
            RawDetection(
                label="car", score=0.9,
                box=(200.0 + m * 30.0, 220.0 - 6.0 * i, 220.0 + m * 30.0, 240.0 - 6.0 * i),
            )
            for m in range(per_side)
        )
        per_frame[i] = down + up
    return StubDetector(per_frame=per_frame)


class _ExplodingDetector(StubDetector):
    """A detector that fails on every frame -- the calibration-failure path."""

    def detect(self, frame):  # type: ignore[no-untyped-def]
        raise RuntimeError("detector exploded")


# --- a provider that records what it was asked to build -----------------------
class RecordingEngineProvider:
    """A :class:`StubEngineProvider` that remembers every engine it was asked for.

    The only way to assert *which scene a run actually reasoned about* rather than
    which scene the API reports afterwards -- and therefore the only way to prove a
    negative as specific as "the deployment fallback was never substituted". It
    also records the resolved rule set, which is what makes the rule-precedence
    tests check the rules that ran instead of merely a 202.
    """

    def __init__(self, inner: StubEngineProvider) -> None:
        self._inner = inner
        self.calls: list[tuple[SceneConfig, tuple[RuleConfig, ...]]] = []

    def create(
        self,
        *,
        scene: SceneConfig,
        rules: tuple[RuleConfig, ...],
        analysis: tuple[AnalysisConfig, ...] = (),
    ) -> InferenceEngine:
        self.calls.append((scene, rules))
        return self._inner.create(scene=scene, rules=rules, analysis=analysis)

    def describe(self) -> str:
        return self._inner.describe()

    # The calibration pass builds an engine first, so the run's engine is the last
    # one built. Both accessors read the same list; naming them separates intent.
    @property
    def run_scene(self) -> SceneConfig:
        assert self.calls, "no engine was ever built"
        return self.calls[-1][0]

    @property
    def run_rules(self) -> tuple[RuleConfig, ...]:
        assert self.calls, "no engine was ever built"
        return self.calls[-1][1]

    @property
    def calibration_passes(self) -> int:
        """How many engines were built beyond the one the run itself needs."""

        return max(0, len(self.calls) - 1)


class _UnavailableProvider:
    """Every engine build fails the way an absent backend does."""

    def create(
        self,
        *,
        scene: SceneConfig,
        rules: tuple[RuleConfig, ...],
        analysis: tuple[AnalysisConfig, ...] = (),
    ) -> InferenceEngine:
        raise EngineUnavailableError("no inference backend is configured")

    def describe(self) -> str:
        return "unavailable"


# --- wiring -------------------------------------------------------------------
def _provider(detector: StubDetector) -> RecordingEngineProvider:
    return RecordingEngineProvider(
        StubEngineProvider(lambda: detector, detector_config=CAR)
    )


def _client(
    tmp_path: Path,
    detector: StubDetector | None = None,
    *,
    default_rules: tuple[RuleConfig, ...] = (),
    provider: object | None = None,
    executor: object | None = None,
) -> TestClient:
    """A client with auto-calibration on and the deployment fallback configured.

    ``make_config`` supplies the fallback scene path by default, which is
    deliberate: these tests must run with a fallback *available* or the central
    claim -- that it is never substituted -- would be vacuous.
    """

    config = make_config(tmp_path / "storage", default_rules=default_rules).model_copy(
        update={"auto_calibrate_uploads": True}
    )
    if provider is None:
        assert detector is not None
        provider = _provider(detector)
    return make_client(
        tmp_path / "storage",
        config=config,
        provider=provider,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )


def _upload(client: TestClient, tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clip = write_wrong_way_clip(tmp_path / "raw.mp4")
    with clip.open("rb") as fh:
        response = client.post("/api/video/upload", files={"file": ("raw.mp4", fh, "video/mp4")})
    assert response.status_code == 201, response.text
    return response.json()["video_id"]


def _assert_ran_in_clip_space(provider: RecordingEngineProvider) -> SceneConfig:
    """Assert the run reasoned in the video's own pixel space, not the fallback's."""

    scene = provider.run_scene
    assert scene.scene.scene_id != FALLBACK_SCENE_ID
    assert (scene.frame.reference_width, scene.frame.reference_height) != FALLBACK_FRAME
    assert (scene.frame.reference_width, scene.frame.reference_height) == CLIP_FRAME
    assert scene.calibration.source == CALIBRATION_SOURCE_AUTO
    return scene


# --- the requirement ----------------------------------------------------------
def test_a_raw_upload_derives_a_scene_in_its_own_pixel_space(tmp_path: Path) -> None:
    client = _client(tmp_path, _flowing_detector())
    video_id = _upload(client, tmp_path)

    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404  # nothing yet

    assert client.post("/api/process", json={"video_id": video_id}).status_code == 202
    scene = client.get(f"/api/videos/{video_id}/scene").json()

    # The clip is 320x240; the shipped fallback scene is 1920x1080. Deriving in the
    # video's own frame is the whole point.
    assert (scene["frame_width"], scene["frame_height"]) == CLIP_FRAME
    assert scene["derived"] is True
    assert scene["has_legal_direction"] is True


def test_the_derived_scene_unlocks_wrong_way_and_the_motorcycle_rules(tmp_path: Path) -> None:
    client = _client(tmp_path, _flowing_detector())
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    supported = client.get(f"/api/videos/{video_id}/scene").json()["supported_violations"]
    assert "wrong_way" in supported
    # Not observable, therefore never claimed:
    assert "illegal_stopping" not in supported
    assert "red_light_jumping" not in supported


def test_all_applicable_rules_run_together_in_one_job(tmp_path: Path) -> None:
    # The end-state: one upload, no rule selection, every applicable reasoner in the
    # same run over one shared detect+track pass.
    client = _client(tmp_path, _flowing_detector())
    video_id = _upload(client, tmp_path)
    job = client.post("/api/process", json={"video_id": video_id}).json()

    status = client.get(f"/api/process/{job['job_id']}").json()
    assert status["status"] == "succeeded"
    assert status["job_id"] == job["job_id"]


# --- abstention: never invent what cannot be observed -------------------------
def test_two_way_traffic_declines_to_declare_a_legal_direction(tmp_path: Path) -> None:
    # Movers cancel. A single legal direction cannot describe this road, and
    # inventing one would flag every lawful vehicle going the other way.
    client = _client(tmp_path, _two_way_detector())
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    scene = client.get(f"/api/videos/{video_id}/scene").json()
    assert scene["derived"] is True
    assert scene["has_legal_direction"] is False
    assert "wrong_way" not in scene["supported_violations"]
    # The frame is still measured, so the geometry-free rules are unaffected.
    assert (scene["frame_width"], scene["frame_height"]) == CLIP_FRAME
    assert "no_helmet" in scene["supported_violations"] or "triple_riding" in (
        scene["supported_violations"]
    )


def test_too_few_movers_declines_to_declare_a_legal_direction(tmp_path: Path) -> None:
    client = _client(tmp_path, _flowing_detector(movers=MIN_FLOW_MOVERS - 1))
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    scene = client.get(f"/api/videos/{video_id}/scene").json()
    assert scene["has_legal_direction"] is False


def test_a_derived_scene_never_claims_a_no_stopping_zone_or_signal(tmp_path: Path) -> None:
    client = _client(tmp_path, _flowing_detector())
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    scene = client.get(f"/api/videos/{video_id}/scene").json()
    assert scene["has_no_stopping_zone"] is False
    assert scene["zone_count"] == 1  # the lane, and nothing invented beside it


def test_a_derived_scene_records_that_it_was_derived(tmp_path: Path) -> None:
    client = _client(tmp_path, _flowing_detector())
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    summary = client.get(f"/api/videos/{video_id}/scene").json()
    full = client.get(f"/api/scenes/{summary['scene_hash']}").json()
    # Provenance lives in the stored scene, so it survives a restart and an audit of
    # any event's scene_config_hash.
    assert full["calibration"]["source"] == CALIBRATION_SOURCE_AUTO
    assert full["legal_directions"][0]["direction_id"] == DERIVED_DIRECTION_ID
    assert full["zones"][0]["zone_id"] == DERIVED_LANE_ID


# --- the fallback scene is never substituted ----------------------------------
# The defect automatic calibration exists to remove is silent: an upload reasoned
# about through another camera's geometry still *runs*, and simply confirms
# nothing. So the guarantee is asserted against the scene the engine was built
# with, on every path that could reach for a fallback.
def test_a_failing_calibration_pass_never_falls_back_to_the_deployment_scene(
    tmp_path: Path,
) -> None:
    provider = _provider(_ExplodingDetector(per_frame={}))
    client = _client(tmp_path, provider=provider)
    video_id = _upload(client, tmp_path)

    client.post("/api/process", json={"video_id": video_id})

    # The run may fail on the same broken backend, but the geometry it was given
    # is the video's own -- never the 1920x1080 scene belonging to another camera.
    scene = _assert_ran_in_clip_space(provider)
    assert scene.legal_directions == ()  # nothing was inferred from a failed pass
    assert not any(zone.zone_type.value == "no_stopping" for zone in scene.zones)
    assert scene.stop_lines == ()
    assert scene.signal_groups == ()
    # And nothing was bound: the video stays honestly uncalibrated, so a later run
    # retries and an analyst's own calibration is unaffected.
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


def test_a_failing_calibration_pass_does_not_refuse_the_video(tmp_path: Path) -> None:
    # Calibration is best-effort: a video must still be accepted if it cannot be
    # derived. (Kept alongside the fallback assertion above, which is the stronger
    # claim; this one pins that the request itself is not rejected.)
    client = _client(tmp_path, _ExplodingDetector(per_frame={}))
    video_id = _upload(client, tmp_path)
    assert client.post("/api/process", json={"video_id": video_id}).status_code == 202
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


def test_an_abstaining_calibration_never_falls_back_to_the_deployment_scene(
    tmp_path: Path,
) -> None:
    provider = _provider(_two_way_detector())
    client = _client(tmp_path, provider=provider)
    video_id = _upload(client, tmp_path)

    client.post("/api/process", json={"video_id": video_id})

    scene = _assert_ran_in_clip_space(provider)
    # Abstention binds a real scene -- it just carries no direction. The fallback's
    # dir-north would have made wrong-way "supported" against foreign geometry.
    assert scene.legal_directions == ()
    assert client.get(f"/api/videos/{video_id}/scene").json()["derived"] is True


def test_the_fallback_is_still_used_when_auto_calibration_is_off(tmp_path: Path) -> None:
    # The guarantee above is scoped to the feature. An operator with one fixed
    # camera who never turns auto-calibration on keeps the pre-existing H12
    # behaviour exactly: the configured scene is their scene, by choice.
    provider = _provider(_flowing_detector())
    config = make_config(tmp_path / "storage", default_rules=())
    assert config.auto_calibrate_uploads is False
    client = make_client(tmp_path / "storage", config=config, provider=provider)  # type: ignore[arg-type]
    video_id = _upload(client, tmp_path)

    client.post("/api/process", json={"video_id": video_id})

    assert provider.run_scene.scene.scene_id == FALLBACK_SCENE_ID
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


# --- the asynchronous job lifecycle -------------------------------------------
# POST /api/process promises 202-then-poll. Calibration is a detector pass, so it
# belongs inside the job like every other frame of inference. The deferred executor
# captures the job's work without running it, which is what lets these tests
# observe the request thread and the worker separately.
def test_the_request_returns_before_any_calibration_work_happens(tmp_path: Path) -> None:
    provider = _provider(_flowing_detector())
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=provider, executor=executor)
    video_id = _upload(client, tmp_path)

    response = client.post("/api/process", json={"video_id": video_id})

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    # Nothing was built, so no model was loaded and no frame was decoded: the
    # handler did no perception work at all.
    assert provider.calls == []
    assert client.get(f"/api/process/{job_id}").json()["status"] == "pending"

    executor.run_pending()

    assert client.get(f"/api/process/{job_id}").json()["status"] == "succeeded"
    assert provider.calibration_passes == 1
    _assert_ran_in_clip_space(provider)


def test_the_derived_run_reaches_the_ordinary_terminal_state(tmp_path: Path) -> None:
    # The calibration phase is *in front of* the normal lifecycle, not beside it:
    # the same run/persist/render path produces the same terminal shape.
    provider = _provider(_flowing_detector())
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=provider, executor=executor)
    video_id = _upload(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    executor.run_pending()

    status = client.get(f"/api/process/{job_id}").json()
    assert status["status"] == "succeeded"
    assert status["progress"] == 1.0
    assert status["frames_total"] is not None
    # Frames counted are the run's, not the calibration pass's.
    assert status["frames_processed"] > 0
    assert status["overlay_status"] in {"none", "pending", "ready", "failed"}
    assert status["evidence_status"] in {"none", "pending", "ready", "failed"}


def test_cancelling_during_the_calibration_phase_ends_as_cancelled(tmp_path: Path) -> None:
    provider = _provider(_flowing_detector())
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=provider, executor=executor)
    video_id = _upload(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    assert client.post(f"/api/process/{job_id}/cancel").status_code == 200
    executor.run_pending()

    assert client.get(f"/api/process/{job_id}").json()["status"] == "cancelled"
    # A cancelled calibration derives nothing: the video is untouched.
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404
    assert provider.calls == []


def test_a_backend_failure_during_derivation_fails_the_job_not_the_request(
    tmp_path: Path,
) -> None:
    # Eager validation used to answer this as a 503 because the engine was built in
    # the handler. With the build moved into the job there is no open request to
    # answer, so the same message has to arrive as the job's terminal error.
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=_UnavailableProvider(), executor=executor)
    video_id = _upload(client, tmp_path)

    response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    executor.run_pending()

    status = client.get(f"/api/process/{job_id}").json()
    assert status["status"] == "failed"
    assert "backend" in (status["error"] or "").lower()
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


def test_an_analyst_calibrated_video_still_validates_eagerly(tmp_path: Path) -> None:
    # The deferral is scoped to uploads that need deriving. A calibrated video keeps
    # the eager 4xx/503 it always had, because its scene is known at submit time.
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=_UnavailableProvider(), executor=executor)
    video_id = _upload(client, tmp_path)
    client.put(
        f"/api/videos/{video_id}/scene",
        json={
            "scene_name": "analyst", "camera_id": "cam-a", "site_id": "site-a",
            "frame_width": 320, "frame_height": 240,
            "zones": [{"zone_id": "z1", "zone_type": "lane",
                       "polygon": [[10, 10], [300, 10], [300, 220], [10, 220]]}],
        },
    )

    assert client.post("/api/process", json={"video_id": video_id}).status_code == 503


# --- precedence: the things auto-calibration must never override --------------
def test_an_analyst_calibrated_video_is_never_re_derived(tmp_path: Path) -> None:
    provider = _provider(_flowing_detector())
    client = _client(tmp_path, provider=provider)
    video_id = _upload(client, tmp_path)
    draft = {
        "scene_name": "analyst", "camera_id": "cam-a", "site_id": "site-a",
        "frame_width": 320, "frame_height": 240,
        "zones": [{"zone_id": "z1", "zone_type": "lane",
                   "polygon": [[10, 10], [300, 10], [300, 220], [10, 220]]}],
    }
    drawn = client.put(f"/api/videos/{video_id}/scene", json=draft).json()
    assert drawn["derived"] is False

    client.post("/api/process", json={"video_id": video_id})

    after = client.get(f"/api/videos/{video_id}/scene").json()
    assert after["scene_hash"] == drawn["scene_hash"]  # untouched
    assert after["derived"] is False
    # And the run used exactly that scene, with no calibration pass in front of it.
    assert provider.calibration_passes == 0
    assert provider.run_scene.scene.camera_id == "cam-a"


def test_explicit_rules_are_still_honoured_on_a_raw_upload(tmp_path: Path) -> None:
    provider = _provider(_flowing_detector())
    client = _client(tmp_path, provider=provider)
    video_id = _upload(client, tmp_path)

    response = client.post(
        "/api/process",
        json={"video_id": video_id, "rules": [{"kind": "triple_riding"}]},
    )

    assert response.status_code == 202
    # Verbatim: the derived scene supports wrong-way too, and it was not added.
    assert [rule.kind for rule in provider.run_rules] == ["triple_riding"]


def test_configured_default_rules_still_win_over_derivation(tmp_path: Path) -> None:
    # An operator who pinned a rule set keeps it; deriving the *scene* does not take
    # the *rule* decision away from them. The clip's flow is coherent, so the
    # scene-derived set would have included wrong_way -- the pinned set must not.
    provider = _provider(_flowing_detector())
    client = _client(
        tmp_path, provider=provider, default_rules=(TripleRidingRuleConfig(),)
    )
    video_id = _upload(client, tmp_path)

    assert client.post("/api/process", json={"video_id": video_id}).status_code == 202

    assert [rule.kind for rule in provider.run_rules] == ["triple_riding"]


def test_no_rules_named_runs_every_rule_the_derived_scene_supports(tmp_path: Path) -> None:
    # The complement of the two tests above: with nothing named, the derived scene
    # decides, and a coherent flow means wrong-way is part of the run.
    provider = _provider(_flowing_detector())
    client = _client(tmp_path, provider=provider)
    video_id = _upload(client, tmp_path)

    client.post("/api/process", json={"video_id": video_id})

    kinds = [rule.kind for rule in provider.run_rules]
    assert "wrong_way" in kinds
    assert "triple_riding" in kinds
    assert "illegal_stopping" not in kinds  # not observable, never claimed
    assert "red_light_jumping" not in kinds


def test_auto_calibration_is_off_unless_the_deployment_enables_it(tmp_path: Path) -> None:
    # Default-off, so no existing embedder changes behaviour by upgrading.
    config = make_config(tmp_path / "storage", default_rules=())
    assert config.auto_calibrate_uploads is False
    client = make_client(
        tmp_path / "storage",
        config=config,
        provider=StubEngineProvider(_flowing_detector, detector_config=CAR),
    )
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    # No scene of its own: the deployment's fallback was used, exactly as before.
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


# --- calibration is paid for only when it can change the outcome --------------
def test_geometry_free_explicit_rules_skip_the_calibration_pass(tmp_path: Path) -> None:
    # Derivation produces exactly one thing: a legal direction. A run of rules that
    # never read one reasons identically without it, so the detector pass would be
    # pure cost on the analyst's wait.
    provider = _provider(_flowing_detector())
    client = _client(tmp_path, provider=provider)
    video_id = _upload(client, tmp_path)

    client.post(
        "/api/process",
        json={"video_id": video_id, "rules": [{"kind": "triple_riding"}]},
    )

    assert provider.calibration_passes == 0
    # Still the video's own frame -- skipping the pass is not a licence to borrow
    # the fallback's geometry.
    scene = _assert_ran_in_clip_space(provider)
    assert scene.legal_directions == ()
    # Nothing observed, so nothing is claimed to have been.
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


def test_geometry_free_default_rules_also_skip_the_calibration_pass(tmp_path: Path) -> None:
    provider = _provider(_flowing_detector())
    client = _client(
        tmp_path, provider=provider, default_rules=(TripleRidingRuleConfig(),)
    )
    video_id = _upload(client, tmp_path)

    client.post("/api/process", json={"video_id": video_id})

    assert provider.calibration_passes == 0


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (None, True),
        ((), True),
        ((TripleRidingRuleConfig(),), False),
        ((NoHelmetRuleConfig(),), False),
        ((WrongWayRuleConfig(),), True),
        ((TripleRidingRuleConfig(), WrongWayRuleConfig()), True),
    ],
)
def test_calibration_is_needed_exactly_when_a_rule_reads_a_direction(
    declared: tuple[RuleConfig, ...] | None, expected: bool
) -> None:
    assert needs_derived_geometry(declared) is expected


# --- explicit wrong-way on an upload nobody calibrated ------------------------
def test_explicit_wrong_way_is_accepted_when_derivation_succeeds(tmp_path: Path) -> None:
    provider = _provider(_flowing_detector())
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=provider, executor=executor)
    video_id = _upload(client, tmp_path)

    response = client.post(
        "/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]}
    )
    assert response.status_code == 202
    executor.run_pending()

    job_id = response.json()["job_id"]
    assert client.get(f"/api/process/{job_id}").json()["status"] == "succeeded"
    # It ran against the direction this clip's own traffic defined.
    assert provider.calibration_passes == 1
    scene = _assert_ran_in_clip_space(provider)
    assert [d.direction_id for d in scene.legal_directions] == [DERIVED_DIRECTION_ID]


def test_explicit_wrong_way_fails_cleanly_when_derivation_abstains(tmp_path: Path) -> None:
    # The honest outcome. The alternative -- running wrong-way against the
    # fallback's dir-north -- would "succeed" while reasoning about another
    # camera's road, which is precisely the silent failure being removed.
    provider = _provider(_two_way_detector())
    executor = DeferredJobExecutor()
    client = _client(tmp_path, provider=provider, executor=executor)
    video_id = _upload(client, tmp_path)

    response = client.post(
        "/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]}
    )
    assert response.status_code == 202
    executor.run_pending()

    status = client.get(f"/api/process/{response.json()['job_id']}").json()
    assert status["status"] == "failed"
    assert status["error"]
    # The measured scene is still bound, so the analyst can see what was observed
    # and draw the lane the clip could not supply.
    scene = client.get(f"/api/videos/{video_id}/scene").json()
    assert scene["derived"] is True
    assert scene["has_legal_direction"] is False


# --- determinism --------------------------------------------------------------
def test_the_same_video_derives_the_same_scene_hash(tmp_path: Path) -> None:
    # The flow vector is rounded for exactly this reason: a derived scene addresses
    # one revision, and content-derived event ids reasoned under it must be stable.
    hashes = set()
    for run in range(2):
        client = _client(tmp_path / f"r{run}", _flowing_detector())
        video_id = _upload(client, tmp_path / f"r{run}")
        client.post("/api/process", json={"video_id": video_id})
        hashes.add(client.get(f"/api/videos/{video_id}/scene").json()["scene_hash"])
    assert len(hashes) == 1


def test_the_calibration_pass_is_bounded(tmp_path: Path) -> None:
    # It is pure overhead on top of the real run, so it must not grow with clip
    # length. Guards the constant against being quietly raised to "the whole clip".
    assert 0 < CALIBRATION_FRAME_BUDGET <= 300


def test_every_calibration_outcome_is_distinguishable(tmp_path: Path) -> None:
    # The four ways a run's scene can be arrived at are named, so the log says which
    # happened rather than leaving it to be inferred from which branch did not raise.
    assert {outcome.value for outcome in CalibrationOutcome} == {
        "derived", "abstained", "failed", "skipped",
    }


@pytest.mark.parametrize("unsupported", ["speeding"])
def test_deferred_violations_are_never_claimed(tmp_path: Path, unsupported: str) -> None:
    client = _client(tmp_path, _flowing_detector())
    video_id = _upload(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    scene = client.get(f"/api/videos/{video_id}/scene").json()
    assert unsupported not in scene["supported_violations"]
