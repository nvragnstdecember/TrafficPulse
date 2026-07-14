#!/usr/bin/env python3
"""TrafficPulse Viewer v0.1 -- a lightweight, browser-based demonstration layer.

This is a **developer demonstration layer**, not production software and not
Phase 6. It adds **no** detection, tracking, reasoning, persistence, or contract
logic of its own. It is a thin presentation shell that *invokes the existing,
already-tested backend* and *displays whatever the backend produced*:

* built-in demo clips reuse the repository's own test fixtures
  (``tests/pipeline/_slice_fixtures.py`` and ``tests/pipeline/_stopping_fixtures.py``)
  and the already-shipped composition roots
  (``trafficpulse.pipeline.runner.run_wrong_way_slice`` and
  ``trafficpulse.pipeline.illegal_stopping_runner.run_illegal_stopping_slice``),
  exactly as ``demo/run_demo.py`` does -- so the offline scripted-detector slices
  confirm the same real ``ConfirmedEvent``s the CLI/tests already produce;
* uploaded clips are run through the **real RT-DETR** offline slice (genuine
  inference behind the P1-U6 seam), and the honest result is displayed -- which,
  on footage that does not match the demo scene's calibration, is commonly
  "no violations found". Nothing is fabricated: the stub replays a caller-authored
  script; RT-DETR runs real inference.

Confirmed events are read back from the unmodified ``EventStore`` output location,
so the reviewer never has to open a JSON file by hand.

Framework choice
----------------
Pure Python standard library (``http.server``). Streamlit is **not** already
installed in this repo's environment, so choosing it would *introduce* a heavy new
dependency -- contrary to the "no unnecessary dependencies" rule. The stdlib server
opens in a browser, needs nothing new installed, and keeps the demo layer trivially
auditable. All backend work still goes through the real ``trafficpulse`` package.

Launch
------
    ./.venv/Scripts/python.exe viewer/app.py           # then open http://127.0.0.1:8000
    ./.venv/Scripts/python.exe viewer/app.py --port 8080 --no-browser

Fully offline. Writes only under a scratch runtime dir (gitignored ``runs/viewer``)
and a temp uploads dir. Changes nothing in ``src/`` or ``tests/``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
_FIXTURES_DIR = REPO_ROOT / "tests" / "pipeline"
_VIEWER_DIR = Path(__file__).resolve().parent

# Make the real backend and the repository's own synthetic-clip fixtures importable
# without installing anything. The fixtures live under tests/ (pytest "prepend"
# import mode, no package __init__.py), so a plain script needs them on sys.path --
# the same shim demo/run_demo.py uses.
for _p in (str(_SRC), str(_FIXTURES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import av  # noqa: E402  (base dep; used only to render preview frames)
import yaml  # noqa: E402  (dev dep; scene loading, mirrors run_demo.py)

from trafficpulse.contracts import (  # noqa: E402
    ConfirmedEvent,
    ObjectClass,
    SceneConfig,
)
from trafficpulse.detector import DetectorConfig  # noqa: E402
from trafficpulse.persistence import EventStore  # noqa: E402
from trafficpulse.tracking import IouTracker  # noqa: E402
from trafficpulse.tracking.config import TrackerConfig  # noqa: E402

# --- reused, already-tested composition roots (NO reasoning duplicated here) ---
from trafficpulse.pipeline.runner import (  # noqa: E402
    SliceRunReport,
    run_wrong_way_slice,
)
from trafficpulse.pipeline.illegal_stopping_runner import (  # noqa: E402
    run_illegal_stopping_slice,
)

# Private composition helpers reused so the RT-DETR backend is built exactly the
# way the shipped CLI builds it (we do NOT re-implement detector construction).
from trafficpulse.pipeline.runner import (  # noqa: E402
    _IOU_TRACKER_MODEL_REF,
    _build_rtdetr_detector,
    _rtdetr_model_ref,
)

# Repository's own synthetic-clip + scripted-detector fixtures (built-in demos).
from _slice_fixtures import (  # noqa: E402
    scripted_down_detector,
    write_wrong_way_clip,
)
from _stopping_fixtures import (  # noqa: E402
    illegal_stopping_test_scene,
    scripted_stopping_detector,
    stopping_detector_config,
    write_illegal_stopping_clip,
)

_EXAMPLE_SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
_WRONG_WAY_DIRECTION_ID = "dir-north"  # example scene's legal north (see README)
_RUN_ROOT = REPO_ROOT / "runs" / "viewer"
_UPLOAD_DIR = REPO_ROOT / "runs" / "viewer" / "_uploads"
_DEFAULT_RTDETR_CHECKPOINT = "PekingU/rtdetr_r50vd"  # locally-cached HF id
_MEDIA_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)  # media-PTS anchor

# clip_id -> path on disk, so the browser can request the raw video / frames.
_CLIP_REGISTRY: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# Backend invocation (thin wrappers -- identical wiring to demo/run_demo.py)
# ---------------------------------------------------------------------------
def _load_example_scene() -> SceneConfig:
    raw = yaml.safe_load(_EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    return SceneConfig.model_validate(raw)


@dataclass
class _Analysis:
    report: SliceRunReport
    events: tuple[ConfirmedEvent, ...]
    clip_path: Path


def _register_clip(path: Path) -> str:
    clip_id = uuid.uuid4().hex[:16]
    _CLIP_REGISTRY[clip_id] = path
    return clip_id


def _analyze_builtin_wrong_way(run_id: str) -> _Analysis:
    """Run the offline wrong-way slice on the repo's synthetic clip (scripted detector)."""
    clip = write_wrong_way_clip(_RUN_ROOT / "clips" / f"{run_id}.mp4")
    report = run_wrong_way_slice(
        clip=clip,
        scene=_load_example_scene(),
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        detector_config=DetectorConfig(label_map={"car": ObjectClass.CAR}),
        output_dir=_RUN_ROOT,
        run_id=run_id,
        direction_id=_WRONG_WAY_DIRECTION_ID,
    )
    events = tuple(s.event for s in EventStore(_RUN_ROOT).load(run_id)) if report.event_count else ()
    return _Analysis(report=report, events=events, clip_path=clip)


def _analyze_builtin_illegal_stopping(run_id: str) -> _Analysis:
    """Run the offline illegal-stopping slice on the repo's synthetic clip."""
    clip = write_illegal_stopping_clip(_RUN_ROOT / "clips" / f"{run_id}.mp4")
    report = run_illegal_stopping_slice(
        clip=clip,
        scene=illegal_stopping_test_scene(),
        detector=scripted_stopping_detector(),
        tracker=IouTracker(),
        detector_config=stopping_detector_config(),
        output_dir=_RUN_ROOT,
        run_id=run_id,
    )
    events = tuple(s.event for s in EventStore(_RUN_ROOT).load(run_id)) if report.event_count else ()
    return _Analysis(report=report, events=events, clip_path=clip)


def _analyze_upload(run_id: str, clip_path: Path) -> _Analysis:
    """Run the REAL RT-DETR wrong-way slice on an uploaded clip (genuine inference).

    Uses the example scene and the shipped CLI's detector construction. This is the
    honest perception path: whatever RT-DETR + the reasoner produce is what shows --
    commonly zero events on footage that does not match the demo scene's geometry.
    """
    detector = _build_rtdetr_detector(
        checkpoint=_DEFAULT_RTDETR_CHECKPOINT,
        device="cpu",
        score_threshold=0.5,
        local_files_only=True,
    )
    report = run_wrong_way_slice(
        clip=clip_path,
        scene=_load_example_scene(),
        detector=detector,
        tracker=IouTracker(tracker_config=TrackerConfig(tracker=_IOU_TRACKER_MODEL_REF)),
        detector_config=DetectorConfig(
            label_map={"car": ObjectClass.CAR},
            score_threshold=0.5,
            source_model=_rtdetr_model_ref(_DEFAULT_RTDETR_CHECKPOINT),
        ),
        output_dir=_RUN_ROOT,
        run_id=run_id,
        direction_id=_WRONG_WAY_DIRECTION_ID,
        checkpoint=_DEFAULT_RTDETR_CHECKPOINT,
        device="cpu",
    )
    events = tuple(s.event for s in EventStore(_RUN_ROOT).load(run_id)) if report.event_count else ()
    return _Analysis(report=report, events=events, clip_path=clip_path)


# ---------------------------------------------------------------------------
# Presentation helpers (pure formatting of backend output -- no new logic)
# ---------------------------------------------------------------------------
def _format_media_time(dt: datetime) -> str:
    """Format an event's media timestamp as MM:SS.cc from the media epoch.

    Media PTS is anchored at a fixed UTC epoch (1970-01-01) by the ingestion layer,
    so the offset from that epoch is the clip-relative time. Matches the demo's
    ``00:01.10`` convention.
    """
    total = (dt - _MEDIA_EPOCH).total_seconds()
    if total < 0:
        total = 0.0
    minutes = int(total // 60)
    seconds = int(total % 60)
    hundredths = int(round((total - int(total)) * 100))
    if hundredths == 100:  # carry rounding
        seconds += 1
        hundredths = 0
    return f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"


def _humanize(value: str) -> str:
    return value.replace("_", " ").title()


def _event_to_card(event: ConfirmedEvent) -> dict[str, object]:
    """Project a ConfirmedEvent into the fields the violation card displays."""
    measurements = [
        {"name": m.name, "value": m.value, "unit": m.unit} for m in event.measurements
    ]
    thresholds = [
        {"name": t.name, "value": t.value, "unit": t.unit} for t in event.thresholds
    ]
    return {
        "violation_type": _humanize(event.violation_type.value),
        "vehicle_id": ", ".join(event.track_ids) or "-",
        "timestamp": _format_media_time(event.trigger_at),
        "status": "Confirmed",  # a ConfirmedEvent is, by construction, confirmed
        "event_id": event.event_id,
        "camera_id": event.camera_id,
        "rule_id": event.rule_id,
        "rule_version": event.rule_version,
        "measurements": measurements,
        "thresholds": thresholds,
        "scene_config_hash": event.scene_config_hash,
    }


def _extract_frames(path: Path, count: int = 6, max_decode: int = 900) -> list[str]:
    """Decode a few evenly-spaced frames to base64 JPEG data URIs (preview only).

    Uses PyAV (already a base dependency). This renders no overlays and adds no
    analysis -- it exists purely so the reviewer sees the clip even when the browser
    cannot decode the synthetic ``mpeg4`` codec directly.
    """
    frames = []
    try:
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_decode:
                    break
                frames.append(frame)
    except Exception:  # pragma: no cover - preview is best-effort
        return []
    if not frames:
        return []
    if len(frames) <= count:
        picked = frames
    else:
        step = (len(frames) - 1) / (count - 1)
        picked = [frames[int(round(k * step))] for k in range(count)]
    uris = []
    for frame in picked:
        try:
            image = frame.to_image()
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=80)
            uris.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        except Exception:  # pragma: no cover
            continue
    return uris


def _build_result_payload(analysis: _Analysis, source_label: str) -> dict[str, object]:
    report = analysis.report
    clip_id = _register_clip(analysis.clip_path)
    return {
        "type": "result",
        "summary": {
            "vehicles_processed": report.unique_tracks,
            "violations_found": report.event_count,
            "frames_processed": report.frames_processed,
            "track_states_emitted": report.track_states_emitted,
            "processing_status": "Complete",
            "detector": report.detector_kind,
            "tracker": report.tracker_kind,
            "source": source_label,
            "run_id": report.run_id,
            "output_dir": report.output_dir,
            "scene_config_hash": report.scene_config_hash,
        },
        "clip": {
            "clip_id": clip_id,
            "width": report.width,
            "height": report.height,
            "fps": report.fps,
            "codec": report.codec,
            # Browsers decode H.264/VP8/VP9/AV1 in <video>, but not MPEG-4 Part 2
            # (the synthetic demo clips' "mpeg4" codec). When not playable inline we
            # rely on the extracted preview frames + a download link instead.
            "playable": (report.codec or "").lower() in {"h264", "avc1", "vp8", "vp9", "av1"},
            "frames": _extract_frames(analysis.clip_path),
        },
        "violations": [_event_to_card(e) for e in analysis.events],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
_INDEX_HTML = (_VIEWER_DIR / "index.html").read_text(encoding="utf-8")

# The staged progress messages narrate the pipeline's real phases. Detection,
# tracking, reasoning and persistence execute as one deterministic backend pass
# (a single run_*_slice call); these labels are that pass's genuine stages.
_STAGES = [
    "Reading video...",
    "Loading detector...",
    "Running tracker...",
    "Applying reasoning...",
    "Generating evidence...",
]


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "TrafficPulseViewer/0.1"

    def log_message(self, fmt: str, *args: object) -> None:  # quieter console
        sys.stderr.write("  [viewer] " + (fmt % args) + "\n")

    # -- helpers ------------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_event(self, payload: dict[str, object]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- routing ------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/analyze":
            self._handle_analyze(parse_qs(parsed.query))
        elif route == "/api/clip":
            self._handle_clip(parse_qs(parsed.query))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._handle_upload()
        else:
            self._send(404, b"not found", "text/plain")

    # -- endpoints ----------------------------------------------------------
    def _handle_upload(self) -> None:
        """Accept a raw video body (X-Filename header). No multipart parsing."""
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._send(400, b'{"error":"empty upload"}', "application/json")
            return
        raw_name = self.headers.get("X-Filename", "upload.mp4")
        safe = "".join(c for c in Path(raw_name).name if c.isalnum() or c in "._-") or "upload.mp4"
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = _UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{safe}"
        data = self.rfile.read(length)
        dest.write_bytes(data)
        clip_id = _register_clip(dest)
        self._send(
            200,
            json.dumps({"clip_id": clip_id, "name": safe, "bytes": len(data)}).encode(),
            "application/json",
        )

    def _handle_clip(self, qs: dict[str, list[str]]) -> None:
        clip_id = (qs.get("id") or [""])[0]
        path = _CLIP_REGISTRY.get(clip_id)
        if not path or not path.is_file():
            self._send(404, b"clip not found", "text/plain")
            return
        data = path.read_bytes()
        # Minimal single-range support so browsers can seek small clips.
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                start_s, _, end_s = rng[len("bytes="):].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else len(data) - 1
                end = min(end, len(data) - 1)
                chunk = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            except (ValueError, OSError):
                pass
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_analyze(self, qs: dict[str, list[str]]) -> None:
        source = (qs.get("source") or ["builtin"])[0]
        scenario = (qs.get("scenario") or ["wrong_way"])[0]
        clip_id = (qs.get("clip") or [""])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        run_id = f"viewer-{uuid.uuid4().hex[:8]}"
        # Run the real backend on a worker thread; stream the genuine pipeline
        # stages while it runs, then stream the honest result.
        box: dict[str, object] = {}

        def worker() -> None:
            try:
                if source == "upload":
                    path = _CLIP_REGISTRY.get(clip_id)
                    if not path:
                        raise ValueError("uploaded clip not found; please re-upload")
                    box["result"] = _analyze_upload(run_id, path)
                    box["label"] = "Uploaded video (real RT-DETR inference)"
                elif scenario == "illegal_stopping":
                    box["result"] = _analyze_builtin_illegal_stopping(run_id)
                    box["label"] = "Built-in synthetic clip (illegal stopping)"
                else:
                    box["result"] = _analyze_builtin_wrong_way(run_id)
                    box["label"] = "Built-in synthetic clip (wrong way)"
            except BaseException as exc:  # noqa: BLE001 - surfaced to the UI honestly
                box["error"] = f"{type(exc).__name__}: {exc}"
                box["trace"] = traceback.format_exc()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            # Emit staged messages; pace them but never outrun the worker.
            for idx, stage in enumerate(_STAGES):
                self._sse_event({"type": "progress", "message": stage,
                                 "step": idx + 1, "total": len(_STAGES) + 1})
                waited = 0.0
                while thread.is_alive() and waited < 0.5:
                    time.sleep(0.1)
                    waited += 0.1
            thread.join()  # ensure the real work is complete before the result
            if "error" in box:
                self._sse_event({"type": "error", "message": str(box["error"])})
            else:
                analysis = box["result"]  # type: ignore[assignment]
                payload = _build_result_payload(analysis, str(box["label"]))
                self._sse_event({"type": "progress", "message": "Analysis complete.",
                                 "step": len(_STAGES) + 1, "total": len(_STAGES) + 1})
                self._sse_event(payload)
            self._sse_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrafficPulse Viewer v0.1 (offline demo UI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not auto-open a browser")
    args = parser.parse_args(argv)

    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (_RUN_ROOT / "clips").mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}/"
    print("=" * 68)
    print("  TrafficPulse Viewer v0.1  (offline developer demonstration)")
    print("=" * 68)
    print(f"  Serving at: {url}")
    print("  Backend:    reused verbatim from src/trafficpulse (unchanged)")
    print("  Press Ctrl+C to stop.")
    print("=" * 68)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
