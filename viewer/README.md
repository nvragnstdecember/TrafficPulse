# TrafficPulse Viewer v0.1

A lightweight, **offline developer demonstration layer** that sits on top of the
existing TrafficPulse backend and lets you run an analysis and see the confirmed
violations **in a browser — without opening any JSON by hand**.

> This is **not** production software, **not** Phase 6, and **not** a new pipeline.
> It adds no detection, tracking, reasoning, persistence, or contract logic. It only
> *invokes the existing backend* and *displays what the backend produced*. Nothing
> under `src/`, `tests/`, `configs/`, or `docs/` is modified.

## What it does

1. You pick a **source** and click **Analyze**.
2. It shows staged progress (Reading video → Loading detector → Running tracker →
   Applying reasoning → Generating evidence → Analysis complete).
3. It displays a **summary panel** (Vehicles Processed, Violations Found, Frames
   Processed, Processing Status) and a **violation card** per confirmed event
   (Violation Type, Vehicle ID, Timestamp, Status + expandable evidence).

### Two honest analysis sources

| Source | What runs | Typical result |
|---|---|---|
| **Built-in demo clip** (Wrong Way / Illegal Stopping) | The repository's own offline scripted-detector slice — the exact runners `demo/run_demo.py` and the pipeline tests use (`run_wrong_way_slice` / `run_illegal_stopping_slice`), real `IouTracker`, real rule engine, real `EventStore`. | A real `ConfirmedEvent` (e.g. **Wrong Way · iou-1 · 00:01.10 · Confirmed**). |
| **Upload a video** | The **real RT-DETR** offline slice (genuine inference behind the P1-U6 seam) on your clip, against the example scene. | Honest — commonly *no violations* on footage that does not match the demo scene's calibration; requires the RT-DETR checkpoint cached locally. |

Neither path fabricates a detection: the built-in stub replays a caller-authored
script matched to a synthetic clip (a COCO RT-DETR does not fire the vehicle class
on synthetic pixels — see the CLI README), and the upload path runs real inference.

> **Note on inline video:** the built-in synthetic clips are encoded as MPEG-4
> Part 2 (`mpeg4`), which browsers cannot decode in `<video>`. For those clips the
> viewer shows **preview frames extracted from the analyzed clip** (via PyAV) plus a
> download link. Uploaded H.264 clips play inline.

## Launch

From the repo root, using the project virtualenv (which already has `av`, `numpy`,
`pydantic`, `pyyaml`, and — for the upload path — `torch`/`transformers`):

```bash
# Windows (this repo's venv)
./.venv/Scripts/python.exe viewer/app.py
```

Then open <http://127.0.0.1:8000/> (a browser opens automatically).

Options:

```bash
./.venv/Scripts/python.exe viewer/app.py --port 8080 --no-browser
```

Fully offline. No database, no auth, no networking, no new dependencies. Runtime
output is written under `runs/viewer/` (already covered by the repo's `/runs/`
gitignore entry); uploads go to `runs/viewer/_uploads/`.

## Files

- `viewer/app.py` — stdlib `http.server` app; invokes the existing backend and
  streams progress + results (Server-Sent Events). No reasoning of its own.
- `viewer/index.html` — the single-page UI (no external assets, no CDN).
- `viewer/README.md` — this file.

## Why the standard library (not Streamlit)

Streamlit is **not** installed in this environment, so choosing it would introduce a
heavy new dependency — contrary to the project's "no unnecessary dependencies" rule.
The Python standard library (`http.server`) opens in a browser, needs nothing
installed, and keeps the demo layer trivially auditable. All real work still goes
through the unmodified `trafficpulse` package.
