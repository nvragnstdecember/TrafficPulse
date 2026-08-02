#!/usr/bin/env python
"""Generate index.csv and the per-video metadata sidecars (T0).

Reads ``sources.yaml`` for the facts that come from the *provider* (title, source,
licence, URL, what the clip shows) and reads the media files themselves for the
facts that come from the *bytes* (duration, resolution, fps, codec, size). Nothing
is copied from one to the other and nothing is estimated: a clip that has not been
downloaded reports its technical fields as empty rather than as the provider's
advertised numbers.

    python test-videos/build_index.py           # write index.csv + *.meta.yaml
    python test-videos/build_index.py --check    # verify they are up to date (CI)

Probing reuses the project's own ingestion path (PyAV via
``trafficpulse.ingestion``) so the numbers recorded here are exactly the numbers
TrafficPulse will see when the clip is uploaded -- including, importantly, whether
the container can be decoded at all.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCES = ROOT / "sources.yaml"
INDEX = ROOT / "index.csv"

# The upload extensions TrafficPulse accepts (mirrors AppConfig's default). A clip
# outside this set is still catalogued -- it is real footage and worth keeping --
# but it must be transcoded before the application can ingest it.
ACCEPTED_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"})

COLUMNS = [
    "filename",
    "category",
    "source",
    "duration_seconds",
    "resolution",
    "fps",
    "expected_violations",
    "licence",
    "country",
    "camera_type",
    "difficulty",
    "status",
    "url",
]


def load_sources() -> dict[str, Any]:
    with SOURCES.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


def probe(path: Path) -> dict[str, Any]:
    """Measure a media file through the project's own ingestion path.

    Returns empty fields when the file is absent (not downloaded) and records the
    decode error when it is present but unreadable -- which is itself a finding
    worth having in the index.
    """

    if not path.is_file():
        return {"status": "not-downloaded"}

    sys.path.insert(0, str(REPO / "src"))
    try:
        from trafficpulse.ingestion.video import open_video
    except ImportError:  # pragma: no cover - the project is not installed
        return {"status": "present", "size_bytes": path.stat().st_size}

    try:
        with open_video(path) as reader:
            metadata = reader.metadata
    except Exception as exc:  # noqa: BLE001 - any decode failure is a real finding
        return {
            "status": "unreadable",
            "size_bytes": path.stat().st_size,
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    return {
        "status": "ok" if path.suffix.lower() in ACCEPTED_EXTENSIONS else "needs-transcode",
        "size_bytes": path.stat().st_size,
        "width": metadata.width,
        "height": metadata.height,
        "fps": round(metadata.fps, 3) if metadata.fps else None,
        "frame_count": metadata.frame_count,
        "duration_seconds": (
            round(metadata.duration_seconds, 2) if metadata.duration_seconds else None
        ),
        "codec": metadata.codec,
    }


def expected_for(entry: dict[str, Any]) -> str:
    """The index's expected-violations cell.

    ``none`` is a real, checkable expectation -- a clean clip must produce zero
    events. Anything else is written as the violation *type* only, never a count,
    because no count in this dataset has been established frame by frame. Counts
    live in evaluation/manifest.yaml and only where a provider shipped ground truth.
    """

    primary = entry.get("primary_violation") or "unknown"
    return "none" if primary == "none" else str(primary)


def row_for(entry: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    width, height = measured.get("width"), measured.get("height")
    return {
        "filename": entry["filename"],
        "category": entry["category"],
        "source": entry["source"],
        "duration_seconds": measured.get("duration_seconds") or "",
        "resolution": f"{width}x{height}" if width and height else "",
        "fps": measured.get("fps") or "",
        "expected_violations": expected_for(entry),
        "licence": entry["licence"],
        "country": entry.get("country", ""),
        "camera_type": entry.get("camera_type", ""),
        "difficulty": entry.get("difficulty", ""),
        "status": measured.get("status", ""),
        "url": entry.get("page") or entry.get("url", ""),
    }


def sidecar_for(entry: dict[str, Any], measured: dict[str, Any]) -> str:
    """The per-video metadata document, in the schema the task specifies."""

    document = {
        "name": entry["filename"],
        "source": entry["source"],
        "title": entry.get("title"),
        "license": entry["licence"],
        "attribution": entry.get("attribution"),
        "url": entry.get("page") or entry.get("url"),
        "duration": measured.get("duration_seconds"),
        "resolution": (
            f"{measured['width']}x{measured['height']}"
            if measured.get("width") and measured.get("height")
            else None
        ),
        "fps": measured.get("fps"),
        "codec": measured.get("codec"),
        "size_bytes": measured.get("size_bytes"),
        "camera_type": entry.get("camera_type"),
        "country": entry.get("country"),
        "primary_violation": entry.get("primary_violation"),
        "secondary_features": entry.get("secondary_features", []),
        "difficulty": entry.get("difficulty"),
        "expected_behavior": (entry.get("expected_behavior") or "").strip(),
        "notes": (entry.get("notes") or "").strip() or None,
        "media_status": measured.get("status"),
        "measured": measured.get("status") in {"ok", "needs-transcode"},
    }
    if measured.get("error"):
        document["decode_error"] = measured["error"]
    header = (
        "# Generated by build_index.py -- do not edit by hand.\n"
        "# Provider facts come from sources.yaml; technical facts are measured\n"
        "# from the file itself. `measured: false` means the clip is not present,\n"
        "# so its technical fields are unknown rather than assumed.\n"
    )
    return header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=88)


def build() -> tuple[str, dict[Path, str]]:
    data = load_sources()
    rows: list[dict[str, Any]] = []
    sidecars: dict[Path, str] = {}
    for entry in data.get("videos") or []:
        path = ROOT / str(entry["category"]) / str(entry["filename"])
        measured = probe(path)
        rows.append(row_for(entry, measured))
        sidecars[path.with_suffix(path.suffix + ".meta.yaml")] = sidecar_for(entry, measured)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r["category"], r["filename"])))
    return buffer.getvalue(), sidecars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail if the index is stale")
    args = parser.parse_args(argv)

    index_text, sidecars = build()

    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != index_text:
            print("index.csv is out of date; run: python test-videos/build_index.py")
            return 1
        print("index.csv is up to date")
        return 0

    INDEX.write_text(index_text, encoding="utf-8")
    for path, text in sidecars.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"wrote {INDEX.relative_to(REPO)} and {len(sidecars)} metadata sidecar(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
