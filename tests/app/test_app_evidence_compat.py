"""Evidence rendering across restarts and pre-H14 repositories (H14 regression).

The compatibility contract this milestone had to keep: a repository written by
H11/H12/H13 -- which has manifests but no rendered artifacts -- must keep serving
evidence exactly as it did, and a repository written by H14 must keep serving its
artifacts after the process that rendered them is gone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from _app_helpers import make_client, make_config, upload_wrong_way_video
from fastapi.testclient import TestClient

pytest.importorskip("PIL", reason="evidence rendering needs Pillow (the 'overlay' extra)")


def _process(client: TestClient, tmp_path: Path) -> str:
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    listing = client.get("/api/events", params={"video_id": video_id}).json()
    event_id: str = listing["items"][0]["event_id"]
    return event_id


def _strip_rendered_artifacts(tmp_path: Path) -> None:
    """Reduce an H14 repository to the H11-H13 shape: manifests, no rendered anything."""

    config = make_config(tmp_path)
    shutil.rmtree(config.runs_dir / "rendered", ignore_errors=True)
    shutil.rmtree(config.artifacts_dir, ignore_errors=True)


def test_artifacts_survive_a_restart(tmp_path: Path) -> None:
    """Rendered evidence is durable: a new process serves it from disk unchanged."""

    first = make_client(tmp_path)
    event_id = _process(first, tmp_path)
    before = first.get(f"/api/evidence/{event_id}/artifacts/trigger_frame")
    assert before.status_code == 200

    # A fresh app over the same storage: nothing in memory, everything recovered.
    restarted = make_client(tmp_path)
    after = restarted.get(f"/api/evidence/{event_id}/artifacts/trigger_frame")
    assert after.status_code == 200
    assert after.content == before.content

    manifest = restarted.get(f"/api/evidence/{event_id}").json()
    assert manifest["trigger_frame"]["sha256"] is not None


def test_a_pre_h14_repository_still_serves_its_manifest(tmp_path: Path) -> None:
    """The H11-H13 shape: references without hashes, served exactly as persisted."""

    client = make_client(tmp_path)
    event_id = _process(client, tmp_path)
    _strip_rendered_artifacts(tmp_path)

    restarted = make_client(tmp_path)
    response = restarted.get(f"/api/evidence/{event_id}")
    assert response.status_code == 200
    manifest = response.json()
    # Unmerged: the persisted reference, with no hash, exactly as before H14.
    assert manifest["trigger_frame"]["locator"].startswith("frames/")
    assert manifest["trigger_frame"]["sha256"] is None
    assert manifest["rule_trace"]

    # And the persisted file is what is being served -- nothing was rewritten.
    persisted = json.loads(
        next(make_config(tmp_path).runs_dir.glob(f"*/manifests/{event_id}.json")).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["trigger_frame"] == manifest["trigger_frame"]


def test_a_pre_h14_repository_reports_artifacts_as_absent(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    event_id = _process(client, tmp_path)
    _strip_rendered_artifacts(tmp_path)

    restarted = make_client(tmp_path)
    response = restarted.get(f"/api/evidence/{event_id}/artifacts/trigger_frame")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "artifact_not_found"


def test_a_pre_h14_repository_still_packages_its_metadata(tmp_path: Path) -> None:
    """Download works everywhere; an unrendered event yields the metadata archive."""

    import io
    import zipfile

    client = make_client(tmp_path)
    event_id = _process(client, tmp_path)
    _strip_rendered_artifacts(tmp_path)

    restarted = make_client(tmp_path)
    response = restarted.get(f"/api/evidence/{event_id}/package")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == [
            f"{event_id}/event.json",
            f"{event_id}/manifest.json",
        ]


def test_reprocessing_reuses_the_stored_artifacts(tmp_path: Path) -> None:
    """A second run over the same video renders the same bytes and stores them once."""

    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    client.post("/api/process", json={"video_id": video_id})

    event_id = client.get("/api/events", params={"video_id": video_id}).json()["items"][0][
        "event_id"
    ]
    manifest = client.get(f"/api/evidence/{event_id}").json()
    assert manifest["trigger_frame"]["sha256"] is not None

    # Content addressing collapses the second run's identical frames onto the first's
    # files, so reprocessing costs no additional storage.
    stored = list(make_config(tmp_path).artifacts_dir.rglob("*.png"))
    digests = {path.stem for path in stored}
    assert len(stored) == len(digests)


def test_events_and_review_are_unaffected_by_rendering(tmp_path: Path) -> None:
    """H9/H11 surfaces keep working beside the new artifacts."""

    client = make_client(tmp_path)
    event_id = _process(client, tmp_path)

    assert client.get(f"/api/events/{event_id}").status_code == 200
    assert client.get(f"/api/events/{event_id}/review").status_code == 200
    assert client.get("/api/videos").json()["total"] == 1
