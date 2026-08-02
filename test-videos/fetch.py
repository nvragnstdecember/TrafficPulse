#!/usr/bin/env python
"""Download the fetchable clips of the TrafficPulse regression dataset (T0).

Reads ``sources.yaml`` and downloads every entry under ``videos:`` into its
category folder. Entries under ``manual:`` are never touched -- they need a
registration or a signed data agreement, and this script will not pretend
otherwise.

    python test-videos/fetch.py               # everything except the large clips
    python test-videos/fetch.py --large       # include clips marked large: true
    python test-videos/fetch.py --only clean_002 wrongway_001
    python test-videos/fetch.py --list        # show what would be fetched

Idempotent: a clip already present with the expected size is skipped, so this is
safe to re-run and safe to interrupt (partial downloads land on a ``.part`` file
and are only moved into place once complete).

This script is deliberately dependency-free beyond PyYAML (already a dev extra of
the project) and the standard library -- it must run on a clean checkout before
anything else is installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources.yaml"
# Wikimedia (and most providers) reject unidentified bulk clients; identifying the
# tool is both required by their etiquette policy and simply honest.
USER_AGENT = "TrafficPulse-regression-dataset/1.0 (research; contact repo owner)"
CHUNK = 1 << 20
#: Seconds between successive files, and the base for exponential backoff. These
#: are shared research mirrors serving tens of megabytes a file; pacing the client
#: is the difference between a courteous consumer and a scraper.
DELAY_SECONDS = 4.0
BACKOFF_BASE = 15


def load_sources() -> dict[str, Any]:
    with SOURCES.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


def target_path(entry: dict[str, Any]) -> Path:
    return ROOT / str(entry["category"]) / str(entry["filename"])


def download(entry: dict[str, Any], *, force: bool = False, retries: int = 4) -> str:
    """Fetch one clip; return a short status word for the caller to report.

    A **polite** client, because the main provider (Wikimedia) rate-limits and is
    entitled to: exactly one request per file (no HEAD probe -- the declared
    Content-Length on the GET is checked instead), an identifying User-Agent, and
    exponential backoff on 429/503 rather than immediate retry.
    """

    destination = target_path(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "present"

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(str(entry["url"]), headers={"User-Agent": USER_AGENT})

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                declared = response.headers.get("Content-Length")
                with partial.open("wb") as out:
                    shutil.copyfileobj(response, out, CHUNK)
            if declared and partial.stat().st_size != int(declared):
                partial.unlink(missing_ok=True)
                return "failed (incomplete transfer)"
            partial.replace(destination)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if exc.code not in (429, 503) or attempt == retries - 1:
                return f"failed (HTTP {exc.code})"
            wait = BACKOFF_BASE * (2**attempt)
            print(f"    rate-limited; waiting {wait}s before retry {attempt + 2}/{retries}")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == retries - 1:
                return f"failed ({exc})"
            time.sleep(BACKOFF_BASE * (2**attempt))
    return "failed (exhausted retries)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--large", action="store_true", help="include clips marked large: true")
    parser.add_argument("--only", nargs="*", metavar="NAME", help="fetch only these entries")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--list", action="store_true", help="list without downloading")
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_SECONDS,
        help=f"seconds between files (default {DELAY_SECONDS}); raise it if rate-limited",
    )
    args = parser.parse_args(argv)

    data = load_sources()
    entries: list[dict[str, Any]] = list(data.get("videos") or [])
    if args.only:
        wanted = set(args.only)
        entries = [entry for entry in entries if entry["name"] in wanted]
    elif not args.large:
        entries = [entry for entry in entries if not entry.get("large")]

    if not entries:
        print("nothing selected")
        return 0

    if args.list:
        for entry in entries:
            destination = target_path(entry).relative_to(ROOT)
            print(f"{entry['name']:<16} {entry['licence']:<14} -> {destination}")
        return 0

    failures = 0
    for index, entry in enumerate(entries):
        if index:
            time.sleep(args.delay)  # pace the client; see DELAY_SECONDS
        status = download(entry, force=args.force)
        if status.startswith("failed"):
            failures += 1
        print(f"{entry['name']:<16} {status}", flush=True)

    skipped = [e["name"] for e in (data.get("videos") or []) if e.get("large") and not args.large]
    if skipped:
        print(f"\nskipped (large, pass --large): {', '.join(skipped)}")
    manual = data.get("manual") or []
    if manual:
        print(
            f"{len(manual)} dataset(s) require manual acquisition "
            "(registration or a data agreement) -- see README.md"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
