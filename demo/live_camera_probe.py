"""Measure what live camera monitoring actually achieves on this machine.

Live mode's frame rate is a property of the hardware it runs on, not a number the
project can state once. This script measures it: it drives a **running server's**
live WebSocket exactly as the browser does -- one `start`, then JPEG frames with at
most two in flight -- and reports the rates, latencies and resident memory the
session really produced.

It is not a test double and it is not a simulation of the pipeline. Every frame it
sends goes through the same socket, the same session, the same detector, tracker,
associator, classifier and reasoners a browser's frames go through. The only thing
it replaces is the browser: it reads frames from a camera device, or -- on a machine
with no camera -- replays a video file as a producer at a chosen rate.

Usage
-----
Start the server first (see ``docs/live-camera.md``), then::

    # From a real camera (Windows DirectShow; use the name ffmpeg lists)
    python demo/live_camera_probe.py --camera "Integrated Camera" --seconds 60

    # From a file, replayed as if it were a camera (no camera on this machine)
    python demo/live_camera_probe.py --video runs/demo-clips/delhi_short.mp4 --seconds 60

    # List the camera devices this machine exposes
    python demo/live_camera_probe.py --list-cameras

Exit code is 0 when the session ran and produced measurements, 1 otherwise. The
measurements are printed as a table and, with ``--json PATH``, written as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import platform
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np

#: Frames the probe keeps unacknowledged, matching the browser client exactly. A
#: different number here would measure a different system.
MAX_IN_FLIGHT = 2

#: How often the producer offers a frame. Higher than any achievable inference
#: rate on purpose -- the point is to keep a slot filled the instant one frees.
PRODUCER_HZ = 20.0


# --- frame producers ------------------------------------------------------------
def _encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def list_cameras() -> None:
    """Print the video capture devices FFmpeg can see on this machine."""

    import av.logging

    system = platform.system()
    if system != "Windows":
        print(
            f"Device enumeration here is implemented for Windows (DirectShow); on "
            f"{system} pass the v4l2/avfoundation device path to --camera directly."
        )
        return
    av.logging.set_level(av.logging.INFO)
    # FFmpeg prints the device list to its own log and then exits non-zero, so the
    # listing *is* the error path: restore the default callback so the list reaches
    # stderr, and let the exit be swallowed.
    av.logging.restore_default_callback()
    with contextlib.suppress(Exception):
        av.open("dummy", format="dshow", options={"list_devices": "true"})


def camera_frames(device: str) -> Iterator[np.ndarray]:
    """RGB frames from a capture device, forever."""

    system = platform.system()
    fmt = {"Windows": "dshow", "Linux": "v4l2", "Darwin": "avfoundation"}.get(system)
    if fmt is None:  # pragma: no cover - exercised only on an unusual platform
        raise SystemExit(f"no capture backend known for {system}")
    url = f"video={device}" if fmt == "dshow" else device
    container = av.open(url, format=fmt)
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def video_frames(path: Path) -> Iterator[np.ndarray]:
    """RGB frames from a file, looping -- a stand-in producer, not a camera."""

    while True:
        container = av.open(str(path))
        try:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
        finally:
            container.close()


# --- measurement ------------------------------------------------------------------
@dataclass
class Measurements:
    """What the probe observed. Nothing here is derived from a configured target."""

    frames_produced: int = 0
    frames_sent: int = 0
    results: int = 0
    events: int = 0
    warnings: list[str] = field(default_factory=list)
    round_trip_ms: list[float] = field(default_factory=list)
    server_latency_ms: list[float] = field(default_factory=list)
    server_stats: dict[str, Any] = field(default_factory=dict)
    session: dict[str, Any] = field(default_factory=dict)
    rss_samples: list[tuple[float, int]] = field(default_factory=list)
    wall_seconds: float = 0.0
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        def stat(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean": None, "p50": None, "p95": None, "max": None}
            ordered = sorted(values)
            return {
                "mean": sum(ordered) / len(ordered),
                "p50": ordered[len(ordered) // 2],
                "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
                "max": ordered[-1],
            }

        wall = self.wall_seconds or 1e-9
        return {
            "wall_seconds": self.wall_seconds,
            "producer_fps": self.frames_produced / wall,
            "sent_fps": self.frames_sent / wall,
            "inference_fps_measured": self.results / wall,
            "inference_fps_server_reported": self.server_stats.get("inference_fps"),
            "server_processing_ms_mean": self.server_stats.get("processing_ms_mean"),
            "round_trip_ms": stat(self.round_trip_ms),
            "server_latency_ms": stat(self.server_latency_ms),
            "frames_produced": self.frames_produced,
            "frames_sent": self.frames_sent,
            "frames_processed": self.results,
            "frames_dropped_by_producer": self.frames_produced - self.frames_sent,
            "frames_dropped_by_server": self.server_stats.get("frames_dropped"),
            "events": self.events,
            "warnings": self.warnings[:5],
            "server_rss_growth_bytes": (
                self.rss_samples[-1][1] - self.rss_samples[0][1]
                if len(self.rss_samples) >= 2
                else None
            ),
            "session": self.session,
            "error": self.error,
        }


async def run_probe(
    *,
    url: str,
    frames: Iterator[np.ndarray],
    seconds: float,
    scene_hash: str | None,
    rss: ResidentMemory | None,
) -> Measurements:
    """Drive one live session for ``seconds`` and return what it measured.

    Two concurrent tasks, exactly as the browser client uses: a producer that offers
    frames and sends them while a slot is free, and a receiver that consumes results.
    A single loop that alternated between the two would serialise sending behind
    receiving and measure the probe rather than the server.
    """

    import websockets

    measured = Measurements()
    first = next(frames)
    height, width = int(first.shape[0]), int(first.shape[1])

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "start",
                    "width": width,
                    "height": height,
                    "scene_hash": scene_hash,
                }
            )
        )
        opened = json.loads(await socket.recv())
        if opened.get("type") != "session":
            measured.error = f"{opened.get('code')}: {opened.get('message')}"
            return measured
        measured.session = {
            "session_id": opened["session_id"],
            "size": f"{opened['width']}x{opened['height']}",
            "running_violations": opened["running_violations"],
            "unavailable_violations": [
                entry["violation_type"] for entry in opened["unavailable_violations"]
            ],
            "window_frames": opened["window_frames"],
        }

        started = time.monotonic()
        finished = asyncio.Event()
        state = {"in_flight": 0, "sequence": 0}
        sent_at: dict[int, float] = {}

        async def receive_loop() -> None:
            while not finished.is_set():
                raw = await socket.recv()
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "result":
                    state["in_flight"] = max(0, state["in_flight"] - 1)
                    measured.results += 1
                    measured.server_stats = message["stats"]
                    departed = sent_at.pop(message["sequence"], None)
                    if departed is not None:
                        measured.round_trip_ms.append(
                            (time.monotonic() - departed) * 1000.0
                        )
                    if message["stats"].get("latency_ms_last") is not None:
                        measured.server_latency_ms.append(
                            message["stats"]["latency_ms_last"]
                        )
                    if rss is not None:
                        measured.rss_samples.append(
                            (time.monotonic() - started, rss.bytes())
                        )
                elif kind == "events":
                    measured.events += len(message["events"])
                    for event in message["events"]:
                        print(
                            f"  EVENT {event['violation_type']:<18} "
                            f"tracks={','.join(event['track_ids'])}"
                        )
                elif kind == "warning":
                    state["in_flight"] = max(0, state["in_flight"] - 1)
                    measured.warnings.append(message["message"])
                elif kind == "error":
                    measured.error = f"{message['code']}: {message['message']}"
                    finished.set()
                    return

        async def produce_loop() -> None:
            while not finished.is_set() and time.monotonic() - started < seconds:
                image = next(frames)
                measured.frames_produced += 1
                if state["in_flight"] < MAX_IN_FLIGHT:
                    payload = base64.b64encode(_encode_jpeg(image)).decode("ascii")
                    sequence = state["sequence"]
                    state["sequence"] += 1
                    sent_at[sequence] = time.monotonic()
                    await socket.send(
                        json.dumps(
                            {
                                "type": "frame",
                                "sequence": sequence,
                                "capture_seconds": round(
                                    time.monotonic() - started, 6
                                ),
                                "data": payload,
                            }
                        )
                    )
                    state["in_flight"] += 1
                    measured.frames_sent += 1
                await asyncio.sleep(1.0 / PRODUCER_HZ)
            finished.set()

        receiver = asyncio.create_task(receive_loop())
        producer = asyncio.create_task(produce_loop())
        await producer
        measured.wall_seconds = time.monotonic() - started

        await socket.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(receiver, timeout=15.0)
        except TimeoutError:
            receiver.cancel()
        except Exception:  # noqa: BLE001 - the socket closing ends the receiver
            pass
        if not measured.server_stats:
            measured.error = measured.error or "the session produced no result"
    return measured


class ResidentMemory:
    """The server process's resident set size, sampled without a new dependency.

    Uses ``tasklist`` on Windows and ``/proc`` elsewhere. Absent either, sampling is
    skipped and the report says so rather than printing a fabricated figure.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def bytes(self) -> int:
        if platform.system() == "Windows":
            import subprocess

            output = subprocess.run(
                ["tasklist", "/FI", f"PID eq {self.pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            parts = output.strip().strip('"').split('","')
            if len(parts) < 5:
                return 0
            return int(parts[4].replace(",", "").replace(" K", "").replace("K", "")) * 1024
        try:
            status = Path(f"/proc/{self.pid}/status").read_text(encoding="utf-8")
        except OSError:
            return 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0


def _print_report(measured: Measurements) -> None:
    summary = measured.summary()
    print("\n=== live camera probe ===")
    if measured.error:
        print(f"  session error: {measured.error}")
    session = summary["session"]
    if session:
        print(f"  session          {session['session_id']} ({session['size']})")
        print(f"  running rules    {', '.join(session['running_violations']) or 'none'}")
        print(f"  not evaluated    {', '.join(session['unavailable_violations']) or 'none'}")
    print(f"  wall time        {summary['wall_seconds']:.1f} s")
    print(f"  producer         {summary['producer_fps']:.2f} fps offered")
    print(f"  sent             {summary['sent_fps']:.2f} fps")
    print(f"  INFERENCE        {summary['inference_fps_measured']:.2f} fps (measured)")
    reported = summary["inference_fps_server_reported"]
    print(
        "  server-reported  "
        + (f"{reported:.2f} fps" if reported is not None else "not reported")
    )
    per_frame = summary["server_processing_ms_mean"]
    print(
        "  per frame        "
        + (f"{per_frame:.0f} ms in the pipeline" if per_frame is not None else "not reported")
    )
    for name in ("round_trip_ms", "server_latency_ms"):
        stat = summary[name]
        if stat["mean"] is None:
            print(f"  {name:<16} no samples")
        else:
            print(
                f"  {name:<16} mean {stat['mean']:.0f} ms · p50 {stat['p50']:.0f} · "
                f"p95 {stat['p95']:.0f} · max {stat['max']:.0f}"
            )
    print(
        f"  frames           produced {summary['frames_produced']}, "
        f"sent {summary['frames_sent']}, processed {summary['frames_processed']}"
    )
    print(
        f"  dropped          {summary['frames_dropped_by_producer']} by the producer, "
        f"{summary['frames_dropped_by_server']} by the server"
    )
    print(f"  events           {summary['events']}")
    growth = summary["server_rss_growth_bytes"]
    print(
        "  server RSS       "
        + (
            f"{growth / 1e6:+.1f} MB over the run"
            if growth is not None
            else "not sampled (pass --server-pid)"
        )
    )
    for warning in summary["warnings"]:
        print(f"  warning          {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/api/live/ws")
    parser.add_argument("--camera", help="capture device name/path to read frames from")
    parser.add_argument("--video", type=Path, help="video file to replay as a producer")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--scene-hash", default=None)
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="sample this process's resident memory during the run",
    )
    parser.add_argument("--json", type=Path, help="also write the summary here")
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args(argv)

    if args.list_cameras:
        list_cameras()
        return 0
    if not args.camera and not args.video:
        parser.error("pass --camera or --video (or --list-cameras)")

    frames = camera_frames(args.camera) if args.camera else video_frames(args.video)
    rss = ResidentMemory(args.server_pid) if args.server_pid else None
    measured = asyncio.run(
        run_probe(
            url=args.url,
            frames=frames,
            seconds=args.seconds,
            scene_hash=args.scene_hash,
            rss=rss,
        )
    )
    _print_report(measured)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(measured.summary(), indent=2), encoding="utf-8")
        print(f"  wrote            {args.json}")
    return 1 if measured.error else 0


if __name__ == "__main__":
    sys.exit(main())
