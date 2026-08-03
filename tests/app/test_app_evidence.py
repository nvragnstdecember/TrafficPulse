"""Evidence endpoints: manifest, rendered artifacts, and package (H7A + H14)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from _app_helpers import make_client, make_config, upload_wrong_way_video

from trafficpulse.evidence.artifacts import artifact_sha256

# Evidence rendering draws through the overlay renderer, so these end-to-end
# assertions need the optional 'overlay' extra -- the same posture tests/overlay
# takes. CI installs it, so the rendering path is genuinely exercised there.
pytest.importorskip("PIL", reason="evidence rendering needs Pillow (the 'overlay' extra)")


def _processed_event(client: object, tmp_path: Path) -> tuple[str, str]:
    """Upload + process the wrong-way clip; return ``(video_id, event_id)``."""

    assert hasattr(client, "post")
    video_id = upload_wrong_way_video(client, tmp_path)  # type: ignore[arg-type]
    client.post("/api/process", json={"video_id": video_id})  # type: ignore[attr-defined]
    listing = client.get(  # type: ignore[attr-defined]
        "/api/events", params={"video_id": video_id}
    ).json()
    return video_id, listing["items"][0]["event_id"]


def test_evidence_serves_rendered_frame_references(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)

    manifest = client.get(f"/api/evidence/{event_id}").json()
    assert manifest["event_id"] == event_id
    trigger = manifest["trigger_frame"]
    assert trigger is not None
    # H14: the served reference is content-addressed and integrity-checkable, unlike
    # the pre-render `frames/<camera>/<frame_id>` placeholder it replaces.
    assert trigger["locator"].startswith("artifacts/")
    assert trigger["sha256"] is not None
    assert trigger["media_type"] == "image/png"
    assert manifest["rule_trace"]  # a reviewable rule trace is still present


def test_rendering_never_alters_the_persisted_manifest(tmp_path: Path) -> None:
    """The load-bearing invariant: the write-once record is untouched by rendering."""

    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)

    runs = make_config(tmp_path).runs_dir
    persisted_files = list(runs.glob(f"*/manifests/{event_id}.json"))
    assert persisted_files, "the run persisted a manifest"
    persisted = json.loads(persisted_files[0].read_text(encoding="utf-8"))

    # On disk the manifest still names the frame by ingestion identity and carries
    # no hash -- exactly what the run wrote, before anything was rendered.
    assert persisted["trigger_frame"]["locator"].startswith("frames/")
    assert persisted["trigger_frame"]["sha256"] is None
    # The rendered references live in their own append-only journal instead.
    assert (runs / "rendered" / f"{event_id}.jsonl").is_file()


def test_artifact_endpoint_serves_verified_png_bytes(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)
    manifest = client.get(f"/api/evidence/{event_id}").json()

    response = client.get(f"/api/evidence/{event_id}/artifacts/trigger_frame")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    # The bytes served are exactly the ones the manifest's hash commits to.
    assert artifact_sha256(response.content) == manifest["trigger_frame"]["sha256"]


def test_artifact_endpoint_is_404_for_an_unrendered_kind(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)

    response = client.get(f"/api/evidence/{event_id}/artifacts/plate_crop")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "artifact_not_found"


def test_artifact_endpoint_rejects_a_tampered_artifact(tmp_path: Path) -> None:
    """A stored file that no longer matches its hash is withheld, not served."""

    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)
    manifest = client.get(f"/api/evidence/{event_id}").json()

    locator = manifest["trigger_frame"]["locator"]
    (make_config(tmp_path).artifacts_dir / locator).write_bytes(b"not the evidence")

    response = client.get(f"/api/evidence/{event_id}/artifacts/trigger_frame")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "artifact_not_found"


def test_package_bundles_event_manifest_and_frames(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)

    response = client.get(f"/api/evidence/{event_id}/package")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert f"evidence-{event_id}.zip" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = sorted(archive.namelist())
        assert f"{event_id}/event.json" in names
        assert f"{event_id}/manifest.json" in names
        frames = [name for name in names if name.startswith(f"{event_id}/frames/")]
        assert frames, "the package carries rendered frames"

        # Self-contained and verifiable: every reference in the packaged manifest
        # resolves to a file in the same archive whose bytes match its hash.
        manifest = json.loads(archive.read(f"{event_id}/manifest.json"))
        trigger = manifest["trigger_frame"]
        member = f"{event_id}/frames/{Path(trigger['locator']).name}"
        assert member in names
        assert artifact_sha256(archive.read(member)) == trigger["sha256"]


def test_package_is_byte_identical_across_downloads(tmp_path: Path) -> None:
    """Deterministic bytes: a package hash describes the evidence, not the download."""

    client = make_client(tmp_path)
    _, event_id = _processed_event(client, tmp_path)

    first = client.get(f"/api/evidence/{event_id}/package").content
    second = client.get(f"/api/evidence/{event_id}/package").content
    assert first == second


def test_evidence_unknown_event_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for path in (
        "/api/evidence/evt-nope",
        "/api/evidence/evt-nope/artifacts/trigger_frame",
        "/api/evidence/evt-nope/package",
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "event_not_found"
