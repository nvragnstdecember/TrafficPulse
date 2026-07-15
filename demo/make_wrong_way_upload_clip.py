#!/usr/bin/env python3
"""Generate a wrong-way VALIDATION clip for the upload pipeline (demo utility).

Purpose
-------
The upload path auto-calibrates each clip's legal direction from its observed
dominant traffic flow (``viewer/calibration.py``) and confirms a wrong-way event
only for a vehicle that *sustainedly opposes* that flow. Footage of normal
traffic therefore honestly yields **zero** events — so verifying the positive
path needs a clip that actually contains an against-traffic vehicle.

This script constructs one, in the same spirit as the repository's synthetic
test-clip fixtures (``tests/pipeline/_slice_fixtures.py``): it **authors the
scenario, never the analysis**. It takes real footage, crops a real vehicle's
pixels out of one frame of that footage, and composites that crop moving
smoothly *against* the road's dominant flow. Everything downstream stays fully
genuine: RT-DETR runs real inference on the composited pixels (the pasted
vehicle is a real car image the model must actually detect), the real IoU
tracker associates it, the real heading derivation + wrong-way reasoner decide,
and the unchanged EventStore persists. No detection, track, observation, or
event is fabricated; only the input video is a constructed validation scenario,
which this docstring and the output filename state plainly.

Defaults are tuned to the supplied Connaught-Place upload
(``898x506 @ ~30 fps``, dominant flow heading ~166 deg): the composited vehicle
travels bottom-left -> upper-right (heading ~339 deg, ~173 deg off the flow) for
~4.5 s — far beyond the 1.0 s wrong-way persistence threshold.

Usage
-----
    ./.venv/Scripts/python.exe demo/make_wrong_way_upload_clip.py \
        --source runs/viewer/_uploads/239de7f30a10_trafficpulsewrongway.mp4

Writes ``runs/demo/clips/wrong_way_upload_validation.mp4`` (H.264 when the
bundled FFmpeg provides libx264, else MPEG-4) — upload it in the Viewer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_SOURCE = REPO_ROOT / "runs" / "viewer" / "_uploads" / (
    "239de7f30a10_trafficpulsewrongway.mp4"
)
_DEFAULT_OUT = REPO_ROOT / "runs" / "demo" / "clips" / "wrong_way_upload_validation.mp4"


def _resize_nn(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize (numpy only; no new dependency)."""

    ys = (np.arange(height) * (image.shape[0] / height)).astype(int)
    xs = (np.arange(width) * (image.shape[1] / width)).astype(int)
    return image[ys][:, xs]


def _paste(frame: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> None:
    """Paste ``patch`` centred at ``(cx, cy)`` into ``frame`` (bounds-clamped)."""

    ph, pw = patch.shape[0], patch.shape[1]
    x1, y1 = cx - pw // 2, cy - ph // 2
    x2, y2 = x1 + pw, y1 + ph
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    frame[fy1:fy2, fx1:fx2] = patch[fy1 - y1 : fy2 - y1, fx1 - x1 : fx2 - x1]


def make_clip(
    *,
    source: Path,
    out: Path,
    crop_frame: int,
    crop_box: tuple[int, int, int, int],
    start_frame: int,
    end_frame: int,
    start_center: tuple[int, int],
    end_center: tuple[int, int],
    end_scale: float,
) -> Path:
    """Composite the cropped real vehicle onto the source clip and encode."""

    frames: list[np.ndarray] = []
    with av.open(str(source)) as container:
        rate = container.streams.video[0].average_rate or 30
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise SystemExit(f"error: no decodable frames in {source}")
    if crop_frame >= len(frames):
        raise SystemExit(f"error: --crop-frame {crop_frame} beyond clip ({len(frames)} frames)")

    x1, y1, x2, y2 = crop_box
    vehicle = frames[crop_frame][y1:y2, x1:x2].copy()
    height, width = frames[0].shape[0], frames[0].shape[1]

    end_frame = min(end_frame, len(frames) - 1)
    span = max(1, end_frame - start_frame)
    for index in range(start_frame, end_frame + 1):
        t = (index - start_frame) / span
        scale = 1.0 + (end_scale - 1.0) * t
        pw = max(8, int(vehicle.shape[1] * scale))
        ph = max(8, int(vehicle.shape[0] * scale))
        patch = _resize_nn(vehicle, pw, ph)
        cx = int(start_center[0] + (end_center[0] - start_center[0]) * t)
        cy = int(start_center[1] + (end_center[1] - start_center[1]) * t)
        _paste(frames[index], patch, cx, cy)

    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), "w")
    try:
        stream = container.add_stream("h264", rate=rate)
        codec_used = "h264"
    except av.FFmpegError:  # pragma: no cover - depends on bundled FFmpeg build
        stream = container.add_stream("mpeg4", rate=rate)
        codec_used = "mpeg4"
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for image in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"wrote {out}  ({len(frames)} frames, {width}x{height}, codec={codec_used})")
    print(
        "composited vehicle: real crop from frame "
        f"{crop_frame} {crop_box}, path {start_center}->{end_center} "
        f"frames [{start_frame}..{end_frame}] (against the dominant flow)"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a wrong-way validation clip: composite a real vehicle crop "
            "from the source footage moving against the dominant traffic flow. "
            "Constructed scenario, genuine analysis (see module docstring)."
        )
    )
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE,
                        help="source clip (default: the supplied uploaded CCTV clip)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="output path (default: runs/demo/clips/...validation.mp4)")
    parser.add_argument("--crop-frame", type=int, default=257,
                        help="frame to crop the real vehicle from (default 257)")
    parser.add_argument("--crop-box", type=int, nargs=4, default=(347, 215, 437, 290),
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="vehicle crop box in that frame (default: the maroon car)")
    parser.add_argument("--start-frame", type=int, default=160)
    parser.add_argument("--end-frame", type=int, default=295)
    parser.add_argument("--start-center", type=int, nargs=2, default=(200, 430), metavar=("X", "Y"))
    parser.add_argument("--end-center", type=int, nargs=2, default=(700, 240), metavar=("X", "Y"))
    parser.add_argument("--end-scale", type=float, default=0.6,
                        help="vehicle scale at path end (recedes toward the vanishing point)")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: source clip not found: {args.source}", file=sys.stderr)
        return 2
    make_clip(
        source=args.source,
        out=args.out,
        crop_frame=args.crop_frame,
        crop_box=tuple(args.crop_box),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        start_center=tuple(args.start_center),
        end_center=tuple(args.end_center),
        end_scale=args.end_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
