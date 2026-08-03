"""Still rendering + whole-run evidence rendering over a real clip (H14)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import av
import numpy as np
import pytest

from trafficpulse.contracts import ConfirmedEvent, EvidenceManifest, MeasuredValue
from trafficpulse.contracts.enums import ArtifactKind, ViolationType
from trafficpulse.contracts.evidence import ArtifactReference, RuleTraceStep
from trafficpulse.evidence.artifacts import ArtifactStore
from trafficpulse.evidence.renderer import render_run_evidence
from trafficpulse.overlay import (
    OverlayAlert,
    OverlayBox,
    OverlayCompositor,
    OverlayEmphasis,
    OverlayFrameRef,
)
from trafficpulse.overlay.frame import encode_image, render_stills_at
from trafficpulse.persistence.evidence_stub import build_evidence_manifest
from trafficpulse.persistence.rendered_store import RenderedArtifactStore

pytest.importorskip("PIL", reason="evidence stills need Pillow (the optional 'overlay' extra)")

_FPS = 10
_FRAMES = 20
_AT = datetime(1970, 1, 1, tzinfo=UTC)


def _write_clip(path: Path, *, w: int = 96, h: int = 64) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=_FPS)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for index in range(_FRAMES):
        image = np.full((h, w, 3), 20, dtype=np.uint8)
        # A moving band, so different frames are visibly (and byte-wise) distinct.
        image[:, (index * 4) % w : (index * 4) % w + 6] = (220, 220, 220)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


class _Provider:
    """A minimal overlay provider: one confirmed box on every frame."""

    violation_kind = "wrong_way"

    def elements_for_frame(self, frame: OverlayFrameRef) -> tuple[OverlayBox, ...]:
        return (
            OverlayBox(
                bounds=(4.0, 4.0, 40.0, 40.0),
                emphasis=OverlayEmphasis.SUBJECT,
                alert=OverlayAlert.CONFIRMED,
                key=f"track-{frame.frame_index}",
            ),
        )


def _event(event_id: str, trigger_seconds: float) -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-1",
        start_at=_AT,
        trigger_at=_AT + timedelta(seconds=trigger_seconds),
        rule_id="rule-wrong-way",
        created_at=_AT,
    )


def _manifest(event: ConfirmedEvent, **times: float) -> EvidenceManifest:
    kinds = {
        "before": ArtifactKind.BEFORE_FRAME,
        "trigger": ArtifactKind.TRIGGER_FRAME,
        "after": ArtifactKind.AFTER_FRAME,
    }
    fields = {
        f"{name}_frame": ArtifactReference(
            kind=kinds[name], locator=f"frames/cam-1/vfrm-{name}"
        )
        for name in times
    }
    step = RuleTraceStep(
        index=0,
        label="evidence-frames",
        measurements=tuple(
            MeasuredValue(name=f"{kinds[name].value}_media_time", value=seconds, unit="s")
            for name, seconds in times.items()
        ),
    )
    return build_evidence_manifest(event).model_copy(
        update={**fields, "rule_trace": (step,)}
    )


# --- still renderer -----------------------------------------------------------------
def test_renders_one_still_per_requested_media_time(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    stills = render_stills_at(
        source_path=clip,
        media_times=[0.3, 0.9, 1.5],
        camera_id="cam-1",
        compositor=OverlayCompositor([_Provider()]),
    )

    assert set(stills) == {0.3, 0.9, 1.5}
    for target, still in stills.items():
        assert still.data.startswith(b"\x89PNG\r\n\x1a\n")
        assert still.media_type == "image/png"
        assert still.media_seconds == target
        # A 10 fps clip has a frame on every 0.1 s, so each request lands exactly.
        assert still.frame_media_seconds == pytest.approx(target, abs=1e-6)
        assert still.frame_index == round(target * _FPS)


def test_different_media_times_render_different_pixels(tmp_path: Path) -> None:
    """Guards against every slot silently rendering the same frame."""

    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    stills = render_stills_at(source_path=clip, media_times=[0.2, 1.2], camera_id="cam-1")
    assert stills[0.2].data != stills[1.2].data


def test_rendering_is_deterministic(tmp_path: Path) -> None:
    """The property content addressing depends on: same input, same bytes."""

    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    compositor = OverlayCompositor([_Provider()])
    first = render_stills_at(source_path=clip, media_times=[0.7], camera_id="cam-1",
                             compositor=compositor)
    second = render_stills_at(source_path=clip, media_times=[0.7], camera_id="cam-1",
                              compositor=compositor)
    assert first[0.7].data == second[0.7].data


def test_the_overlay_is_actually_drawn(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    plain = render_stills_at(source_path=clip, media_times=[0.7], camera_id="cam-1")
    drawn = render_stills_at(source_path=clip, media_times=[0.7], camera_id="cam-1",
                             compositor=OverlayCompositor([_Provider()]))
    assert plain[0.7].data != drawn[0.7].data


def test_no_compositor_still_yields_the_evidence_pixels(tmp_path: Path) -> None:
    """A run whose rules drew nothing still has real frames worth showing."""

    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    stills = render_stills_at(source_path=clip, media_times=[0.5], camera_id="cam-1")
    assert stills[0.5].data.startswith(b"\x89PNG\r\n\x1a\n")


def test_a_time_past_the_end_clamps_to_the_last_frame(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    stills = render_stills_at(source_path=clip, media_times=[99.0], camera_id="cam-1")
    assert stills[99.0].frame_index == _FRAMES - 1


def test_no_requested_times_decodes_nothing(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    assert render_stills_at(source_path=clip, media_times=[], camera_id="cam-1") == {}


def test_jpeg_is_encodable(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    stills = render_stills_at(source_path=clip, media_times=[0.4], camera_id="cam-1",
                              image_format="jpeg")
    assert stills[0.4].media_type == "image/jpeg"
    assert stills[0.4].data.startswith(b"\xff\xd8\xff")


def test_an_unsupported_still_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported still format"):
        encode_image(np.zeros((4, 4, 3), dtype=np.uint8), image_format="bmp")


# --- whole-run rendering ------------------------------------------------------------
def test_render_run_stores_artifacts_and_records_the_sidecar(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    rendered = RenderedArtifactStore(tmp_path / "runs")

    event = _event("evt-1", 1.0)
    report = render_run_evidence(
        pairs=[(event, _manifest(event, before=0.9, trigger=1.0, after=1.1))],
        source_path=clip,
        camera_id="cam-1",
        artifacts=artifacts,
        rendered=rendered,
        compositor=OverlayCompositor([_Provider()]),
    )

    assert report.events_rendered == 1
    assert report.artifacts_written == 3
    assert report.frames_decoded == 3

    stored = rendered.artifacts("evt-1")
    assert {reference.kind for reference in stored} == {
        ArtifactKind.BEFORE_FRAME,
        ArtifactKind.TRIGGER_FRAME,
        ArtifactKind.AFTER_FRAME,
    }
    for reference in stored:
        assert artifacts.verify(reference)


def test_re_rendering_a_run_writes_no_new_bytes(tmp_path: Path) -> None:
    """Repeated rendering is idempotent: same hashes, same files, same sidecar."""

    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    rendered = RenderedArtifactStore(tmp_path / "runs")
    event = _event("evt-1", 1.0)
    pairs = [(event, _manifest(event, before=0.9, trigger=1.0, after=1.1))]

    render_run_evidence(pairs=pairs, source_path=clip, camera_id="cam-1",
                        artifacts=artifacts, rendered=rendered)
    first = rendered.artifacts("evt-1")
    files = sorted(path.name for path in (tmp_path / "artifacts").rglob("*.png"))

    render_run_evidence(pairs=pairs, source_path=clip, camera_id="cam-1",
                        artifacts=artifacts, rendered=rendered)
    assert rendered.artifacts("evt-1") == first
    assert sorted(path.name for path in (tmp_path / "artifacts").rglob("*.png")) == files


def test_two_events_sharing_a_frame_share_one_stored_file(tmp_path: Path) -> None:
    """One decode for the run, and one file for a frame two events both need."""

    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    rendered = RenderedArtifactStore(tmp_path / "runs")

    first, second = _event("evt-1", 1.0), _event("evt-2", 1.0)
    report = render_run_evidence(
        pairs=[
            (first, _manifest(first, trigger=1.0)),
            (second, _manifest(second, trigger=1.0)),
        ],
        source_path=clip,
        camera_id="cam-1",
        artifacts=artifacts,
        rendered=rendered,
    )

    assert report.events_rendered == 2
    assert report.frames_decoded == 1  # the shared media time was rendered once
    assert (
        rendered.artifacts("evt-1")[0].locator == rendered.artifacts("evt-2")[0].locator
    )
    assert len(list((tmp_path / "artifacts").rglob("*.png"))) == 1


def test_a_manifest_with_no_recorded_times_renders_nothing(tmp_path: Path) -> None:
    """A pre-engine stub manifest is skipped, never rendered from a guessed position."""

    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    rendered = RenderedArtifactStore(tmp_path / "runs")

    event = _event("evt-1", 1.0)
    report = render_run_evidence(
        pairs=[(event, build_evidence_manifest(event))],
        source_path=clip,
        camera_id="cam-1",
        artifacts=artifacts,
        rendered=rendered,
    )
    assert report.is_empty
    assert rendered.artifacts("evt-1") == ()
    assert not list((tmp_path / "artifacts").rglob("*.png"))
