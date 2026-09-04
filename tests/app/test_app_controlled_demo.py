"""The controlled demonstration, end to end through the HTTP API.

**One video, one analysis, four reasoners, four violation types** -- the claim the
whole controlled-demo feature exists to make, exercised the way a reviewer will:
upload a clip, draw its scene, declare the run's signal timing, declare what the
clip was built to contain, process it, and read the result back.

What is real here and what is not
----------------------------------
Real: the decode (PyAV, real PTS), the tracker, every observation derivation, all
four reasoners, the temporal thresholds, persistence, the evidence manifests, and
the HTTP surface. Scripted: only the detector, because a COCO RT-DETR does not fire
a vehicle class on synthetic rectangles (see ``_controlled_demo_fixtures``).

So these tests establish that the system **reasons correctly over declared scene
context**. They establish nothing about detection accuracy on real pixels, and the
docstrings say so rather than leaving it to be inferred.

The load-bearing negative
--------------------------
Half of this module is about what the demonstration *cannot* do: an expectation is
never a detection. Declaring four families on a clip that contains none must
produce four ``missing`` rows and zero events -- and it is tested, because "the demo
cannot manufacture events" is a property somebody has to be able to check rather
than trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _app_helpers import StubEngineProvider, make_client, make_config
from _controlled_demo_fixtures import (
    controlled_demo_detector_config,
    controlled_demo_draft_payload,
    controlled_demo_red_light_rule,
    scripted_controlled_demo_detector,
    write_controlled_demo_clip,
)
from fastapi.testclient import TestClient

from trafficpulse.scenes import demo_scenario as scenario

EXPECTED = tuple(v.value for v in scenario.DEMO_EXPECTED_VIOLATIONS)


def _app(storage: Path) -> TestClient:
    """An app with no configured scene: the demo must calibrate the clip itself."""

    return make_client(
        storage,
        config=make_config(storage, scene_path=None, default_rules=()),
        provider=StubEngineProvider(
            scripted_controlled_demo_detector,
            detector_config=controlled_demo_detector_config(),
        ),
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A fresh app per test, for the tests that mutate a declaration."""

    return _app(tmp_path / "storage")


@dataclass(frozen=True)
class DemoRun:
    """One completed controlled demonstration: the client, the video, the run."""

    client: TestClient
    storage: Path
    video_id: str
    job_id: str
    supported: list[str]


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> DemoRun:
    """Upload, calibrate, declare and process the controlled clip **once**.

    Module-scoped because a full run costs several seconds and most of what these
    tests assert is read-only: what was confirmed, what the comparison says, what
    the stored scene records. Every test that *mutates* a declaration takes the
    function-scoped ``client`` instead, so nothing here depends on test ordering.
    """

    storage = tmp_path_factory.mktemp("controlled-demo") / "storage"
    client = _app(storage)
    video_id = _upload(client, storage.parent)
    scene = _calibrate(client, video_id)
    _declare(client, video_id)
    job_id = _process(client, video_id, _rules_for(scene["supported_violations"]))
    return DemoRun(
        client=client,
        storage=storage,
        video_id=video_id,
        job_id=job_id,
        supported=scene["supported_violations"],
    )


def _upload(client: TestClient, tmp_path: Path) -> str:
    clip = write_controlled_demo_clip(tmp_path / "controlled-demo.mp4")
    response = client.post(
        "/api/video/upload",
        files={"file": ("controlled-demo.mp4", clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    video_id: str = response.json()["video_id"]
    return video_id


def _calibrate(client: TestClient, video_id: str) -> dict[str, Any]:
    response = client.put(
        f"/api/videos/{video_id}/scene", json=controlled_demo_draft_payload()
    )
    assert response.status_code == 200, response.text
    summary: dict[str, Any] = response.json()
    return summary


def _declare(client: TestClient, video_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "expected_violations": list(EXPECTED),
        "notes": scenario.DEMO_SCENE_NOTES,
        "declared_by": "analyst",
    }
    body.update(overrides)
    response = client.put(f"/api/videos/{video_id}/expectation", json=body)
    assert response.status_code == 200, response.text
    record: dict[str, Any] = response.json()
    return record


def _process(client: TestClient, video_id: str, rules: list[dict[str, Any]]) -> str:
    response = client.post("/api/process", json={"video_id": video_id, "rules": rules})
    assert response.status_code == 202, response.text
    job_id: str = response.json()["job_id"]
    status = client.get(f"/api/process/{job_id}").json()
    assert status["status"] == "succeeded", status
    return job_id


def _rules_for(supported: list[str]) -> list[dict[str, Any]]:
    """The rule declarations the calibration surface would send for this scene.

    Mirrors ``frontend/src/lib/calibration.ts``'s ``rulesForRun``: every supported
    family, with red-light additionally carrying the run's declared schedule because
    a scene cannot supply media-time timing.
    """

    rules: list[dict[str, Any]] = []
    for violation in supported:
        if violation == "red_light_jumping":
            rules.append(controlled_demo_red_light_rule())
        else:
            rules.append({"kind": violation})
    return rules


# --- the central claim -------------------------------------------------------------
def test_one_video_one_run_four_independently_reasoned_violation_types(
    demo: DemoRun,
) -> None:
    """The demonstration: four families confirmed from a single pass.

    Nothing about the four is coordinated. They are four separate reasoners reading
    four separate observation streams off one tracker's output, and each reaches its
    own conclusion under its own threshold.
    """

    client, video_id, job_id = demo.client, demo.video_id, demo.job_id
    assert set(demo.supported) >= set(EXPECTED)

    events = client.get("/api/events", params={"video_id": video_id, "job_id": job_id})
    assert events.status_code == 200, events.text
    items = events.json()["items"]
    assert {item["violation_type"] for item in items} == set(EXPECTED), items
    # Four distinct events, not one event counted four ways.
    assert len({item["event_id"] for item in items}) == len(items)
    # Every one is backed by its own evidence manifest, from the ordinary pipeline.
    for item in items:
        manifest = client.get(f"/api/evidence/{item['event_id']}")
        assert manifest.status_code == 200, manifest.text
        assert manifest.json()["event_id"] == item["event_id"]


def test_the_declared_geometry_is_what_unlocks_each_family(
    client: TestClient, tmp_path: Path
) -> None:
    """Without the drawing, the geometry-dependent families are simply unavailable.

    The honest version of "the camera cannot know this": an uncalibrated upload
    supports only the geometry-free rules, and drawing the lane, the zone and the
    junction is what adds the other three.
    """

    video_id = _upload(client, tmp_path)
    before = client.get(f"/api/videos/{video_id}").json()["supported_violations"]
    assert "wrong_way" not in before
    assert "illegal_stopping" not in before
    assert "red_light_jumping" not in before

    after = _calibrate(client, video_id)["supported_violations"]
    assert {"wrong_way", "illegal_stopping", "red_light_jumping"} <= set(after)


def test_the_operator_dwell_threshold_reaches_the_scene(
    client: TestClient, tmp_path: Path
) -> None:
    """The declared dwell is carried into the stored scene, not hidden in a reasoner."""

    video_id = _upload(client, tmp_path)
    scene_hash = _calibrate(client, video_id)["scene_hash"]

    stored = client.get(f"/api/scenes/{scene_hash}")
    assert stored.status_code == 200, stored.text
    blocks = {
        block["violation_type"]: {p["id"]: p for p in block["parameters"]}
        for block in stored.json()["rule_parameters"]
    }
    dwell = blocks["illegal_stopping"]["stationary_duration"]
    assert dwell["value"] == pytest.approx(scenario.DEMO_STATIONARY_DURATION_S)
    # Operator-chosen, not validated against ground truth -- and it says so.
    assert dwell["status"] == "provisional"


def test_the_scene_records_that_it_is_a_declared_demonstration(
    client: TestClient, tmp_path: Path
) -> None:
    """A reviewer reading the stored revision learns what it is without being told."""

    video_id = _upload(client, tmp_path)
    scene_hash = _calibrate(client, video_id)["scene_hash"]
    stored = client.get(f"/api/scenes/{scene_hash}").json()

    assert "Controlled demonstration" in stored["scene"]["description"]
    # Drawn, not derived from the clip's own motion.
    assert stored["calibration"]["source"] == "analyst_calibration"
    assert stored["scene"]["status"] == "draft"


def test_the_red_light_event_latched_red_although_the_signal_turned_green(
    demo: DemoRun,
) -> None:
    """The H13 latch, demonstrated rather than asserted in a unit test.

    The declared schedule goes green at 2.0 s; the vehicle crossed at 1.2 s and is
    confirmed afterwards. The recorded entry state must still be red -- a signal that
    changes after the act cannot erase it.
    """

    client = demo.client
    events = client.get(
        "/api/events", params={"video_id": demo.video_id, "job_id": demo.job_id}
    ).json()["items"]
    red_light = next(e for e in events if e["violation_type"] == "red_light_jumping")
    detail = client.get(f"/api/events/{red_light['event_id']}")
    assert detail.status_code == 200, detail.text
    measurements = {m["name"]: m["value"] for m in detail.json()["measurements"]}
    # The rule records the latched state as an ordinal; red is its highest.
    assert measurements["signal_state_at_entry"] == 4.0


# --- expected vs detected ----------------------------------------------------------
def test_expected_and_detected_are_compared_without_inventing_accuracy(
    demo: DemoRun,
) -> None:
    """Four declared, four confirmed, four matched, none unexpected."""

    response = demo.client.get(
        f"/api/videos/{demo.video_id}/expectation/comparison",
        params={"job_id": demo.job_id},
    )
    assert response.status_code == 200, response.text
    comparison = response.json()

    assert comparison["expected_count"] == 4
    assert comparison["matched_count"] == 4
    assert comparison["missing_count"] == 0
    assert comparison["unexpected_count"] == 0
    assert comparison["detected_event_count"] == 4
    assert {row["violation_type"] for row in comparison["rows"]} == set(EXPECTED)
    assert all(row["outcome"] == "matched" for row in comparison["rows"])
    # No accuracy is reported, and none may be added without this failing.
    for forbidden in ("precision", "recall", "f1", "accuracy"):
        assert forbidden not in comparison


def test_every_detected_count_is_backed_by_openable_event_ids(demo: DemoRun) -> None:
    """A count nobody can drill into is a claim, not evidence."""

    client = demo.client
    comparison = client.get(
        f"/api/videos/{demo.video_id}/expectation/comparison",
        params={"job_id": demo.job_id},
    ).json()
    for row in comparison["rows"]:
        assert len(row["event_ids"]) == row["detected_count"]
        for event_id in row["event_ids"]:
            detail = client.get(f"/api/events/{event_id}")
            assert detail.status_code == 200, detail.text
            assert detail.json()["violation_type"] == row["violation_type"]


# --- the load-bearing negatives ----------------------------------------------------
def test_declaring_an_expectation_cannot_manufacture_an_event(
    client: TestClient, tmp_path: Path
) -> None:
    """Declare everything, run nothing that can confirm it, get nothing.

    The property the whole separation exists for. The clip is processed with only
    the geometry-free triple-riding rule, so three of the four declared families
    structurally cannot be confirmed -- and the comparison reports them ``missing``
    rather than conjuring them.
    """

    video_id = _upload(client, tmp_path)
    _calibrate(client, video_id)
    _declare(client, video_id)

    job_id = _process(client, video_id, [{"kind": "triple_riding"}])
    events = client.get(
        "/api/events", params={"video_id": video_id, "job_id": job_id}
    ).json()["items"]
    assert {e["violation_type"] for e in events} == {"triple_riding"}

    comparison = client.get(
        f"/api/videos/{video_id}/expectation/comparison", params={"job_id": job_id}
    ).json()
    outcomes = {row["violation_type"]: row["outcome"] for row in comparison["rows"]}
    assert outcomes["triple_riding"] == "matched"
    assert outcomes["wrong_way"] == "missing"
    assert outcomes["illegal_stopping"] == "missing"
    assert outcomes["red_light_jumping"] == "missing"
    assert comparison["missing_count"] == 3


def test_an_expectation_never_appears_in_an_event_listing(demo: DemoRun) -> None:
    """A declared clip lists exactly its confirmed events -- four, all real.

    The declaration names four families and the run confirmed four events, so a
    surface that leaked expectations into the listing would show eight (or would
    show a family with no evidence behind it). It shows four, and the job's own
    ``event_count`` agrees.
    """

    client, video_id = demo.client, demo.video_id
    listing = client.get("/api/events", params={"video_id": video_id}).json()

    assert listing["total"] == 4
    assert client.get(f"/api/process/{demo.job_id}").json()["event_count"] == 4
    # Every listed row is a persisted event with its own evidence, not a claim.
    for item in listing["items"]:
        assert client.get(f"/api/evidence/{item['event_id']}").status_code == 200


def test_declaring_does_not_change_what_a_run_confirms(
    client: TestClient, tmp_path: Path
) -> None:
    """The same clip, run with and without a declaration, confirms the same events.

    Byte-for-byte the same event ids: the expectation is not merely ignored by the
    rules, it cannot reach anything that participates in deriving an event id.
    """

    video_id = _upload(client, tmp_path)
    scene = _calibrate(client, video_id)
    rules = _rules_for(scene["supported_violations"])

    undeclared_job = _process(client, video_id, rules)
    undeclared = client.get(
        "/api/events", params={"video_id": video_id, "job_id": undeclared_job}
    ).json()["items"]

    _declare(client, video_id)
    declared_job = _process(client, video_id, rules)
    declared = client.get(
        "/api/events", params={"video_id": video_id, "job_id": declared_job}
    ).json()["items"]

    assert {e["event_id"] for e in undeclared} == {e["event_id"] for e in declared}


def test_a_video_with_no_declaration_still_compares(
    client: TestClient, tmp_path: Path
) -> None:
    """"Nothing was claimed about this clip" is an answer, not an error."""

    video_id = _upload(client, tmp_path)
    scene = _calibrate(client, video_id)
    job_id = _process(client, video_id, _rules_for(scene["supported_violations"]))

    assert client.get(f"/api/videos/{video_id}/expectation").status_code == 404

    comparison = client.get(
        f"/api/videos/{video_id}/expectation/comparison", params={"job_id": job_id}
    )
    assert comparison.status_code == 200, comparison.text
    body = comparison.json()
    assert body["expectation"] is None
    assert body["expected_count"] == 0
    assert body["unexpected_count"] == 4
    assert all(row["outcome"] == "unexpected" for row in body["rows"])


def test_the_demonstration_survives_a_restart(demo: DemoRun) -> None:
    """A reviewer can restart the server and see the same declaration and comparison.

    Reproducibility (§18) is the point of persisting the declaration at all: a demo
    whose ground truth lives in one browser tab cannot be re-run by anyone else.
    """

    video_id, job_id = demo.video_id, demo.job_id
    before = demo.client.get(
        f"/api/videos/{video_id}/expectation/comparison", params={"job_id": job_id}
    ).json()

    # A second app over the same storage root: H10 recovery rebuilds the registries
    # from disk, and the declaration is read from its own store.
    restarted = _app(demo.storage)
    after = restarted.get(
        f"/api/videos/{video_id}/expectation/comparison", params={"job_id": job_id}
    )
    assert after.status_code == 200, after.text
    assert after.json() == before


def test_withdrawing_a_declaration_leaves_the_events_alone(
    client: TestClient, tmp_path: Path
) -> None:
    video_id = _upload(client, tmp_path)
    scene = _calibrate(client, video_id)
    _declare(client, video_id)
    job_id = _process(client, video_id, _rules_for(scene["supported_violations"]))

    assert client.delete(f"/api/videos/{video_id}/expectation").status_code == 204
    assert client.get(f"/api/videos/{video_id}/expectation").status_code == 404
    # Withdrawing is idempotent: the caller's intent is satisfied either way.
    assert client.delete(f"/api/videos/{video_id}/expectation").status_code == 204

    events = client.get(
        "/api/events", params={"video_id": video_id, "job_id": job_id}
    ).json()
    assert events["total"] == 4
