"""Video upload: validation, storage, duplicate + size handling (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import make_client, make_config
from _slice_fixtures import write_wrong_way_clip


def _upload(client: object, name: str, data: bytes, content_type: str = "video/mp4") -> object:
    return client.post(  # type: ignore[attr-defined]
        "/api/video/upload", files={"file": (name, data, content_type)}
    )


def test_upload_stores_and_returns_metadata(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    data = write_wrong_way_clip(tmp_path / "clip.mp4").read_bytes()
    response = _upload(client, "clip.mp4", data)
    assert response.status_code == 201
    body = response.json()
    assert body["video_id"].startswith("vid-")
    assert body["filename"] == "clip.mp4"
    assert body["status"] == "stored"
    assert body["size_bytes"] == len(data)
    assert body["width"] == 320 and body["height"] == 240
    assert body["codec"]  # a real decoded codec name
    # The file is physically stored under the configured storage dir.
    stored = list((tmp_path / "videos").glob(f"{body['video_id']}*"))
    assert len(stored) == 1


def test_upload_is_content_addressed(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    data = write_wrong_way_clip(tmp_path / "clip.mp4").read_bytes()
    first = _upload(client, "a.mp4", data).json()["video_id"]
    # Same bytes under a different name -> duplicate (409), same id reported.
    duplicate = _upload(client, "b.mp4", data)
    assert duplicate.status_code == 409
    error = duplicate.json()["error"]
    assert error["type"] == "duplicate_video"
    assert first in error["message"]
    # The conflict carries the existing id so a client can open it directly; the
    # field appears only for this error (see ErrorDetail.video_id).
    assert error["video_id"] == first


def test_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = _upload(client, "notes.txt", b"hello", "text/plain")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_media"


def test_upload_rejects_empty_file(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = _upload(client, "empty.mp4", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["error"]["message"]


def test_upload_rejects_unreadable_video(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = _upload(client, "fake.mp4", b"this is not a video")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_media"
    # The unreadable file was removed, not left half-stored.
    assert list((tmp_path / "videos").glob("*")) == []


def test_upload_rejects_oversize(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, max_upload_bytes=16))
    response = _upload(client, "big.mp4", b"x" * 64)
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "payload_too_large"


def test_upload_without_a_file_is_422(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/video/upload")  # missing multipart file
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


def test_video_service_rejects_oversize_directly(tmp_path: Path) -> None:
    # The service enforces the size limit defensively even if a caller bypasses
    # the router's streaming cap.
    import pytest

    from trafficpulse.app.errors import PayloadTooLargeError
    from trafficpulse.app.registry import VideoStore
    from trafficpulse.app.services import VideoService

    service = VideoService(make_config(tmp_path, max_upload_bytes=8), VideoStore())
    with pytest.raises(PayloadTooLargeError):
        service.store_upload("big.mp4", b"x" * 64)
