"""Consistent error envelope, status codes, and no-traceback guarantee (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import (
    RaisingEngineProvider,
    UnavailableEngineProvider,
    make_client,
    upload_wrong_way_video,
)


def _assert_envelope(payload: dict, *, type_: str) -> None:
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"type", "message"}
    assert payload["error"]["type"] == type_
    assert isinstance(payload["error"]["message"], str) and payload["error"]["message"]


def test_404_envelope(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/process/job-nope")
    assert response.status_code == 404
    _assert_envelope(response.json(), type_="job_not_found")


def test_400_envelope(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/video/upload", files={"file": ("x.txt", b"hi", "text/plain")}
    )
    assert response.status_code == 400
    _assert_envelope(response.json(), type_="unsupported_media")


def test_422_envelope_for_body_validation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/process", json={})  # missing required video_id
    assert response.status_code == 422
    _assert_envelope(response.json(), type_="validation_error")


def test_500_hides_internals_and_exposes_no_traceback(tmp_path: Path) -> None:
    # A provider that raises an unexpected (non-AppError) error during submit.
    client = make_client(tmp_path, provider=RaisingEngineProvider())
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 500
    payload = response.json()
    _assert_envelope(payload, type_="internal_error")
    # The generic message must not leak the underlying exception text or a trace.
    assert "boom" not in payload["error"]["message"]
    assert "Traceback" not in response.text


def test_backend_unavailable_maps_to_503(tmp_path: Path) -> None:
    # A typed DetectorError from the engine provider (as the real RT-DETR backend
    # raises on a missing extra/checkpoint) is surfaced as a 503 -- exercised via
    # a stub that raises the same error, so no ML framework is imported.
    client = make_client(tmp_path, provider=UnavailableEngineProvider())
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 503
    _assert_envelope(response.json(), type_="engine_unavailable")
