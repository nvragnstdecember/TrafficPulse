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
| **Upload a video** | The **real RT-DETR** offline slice (genuine inference behind the P1-U6 seam) against a **per-clip auto-calibrated scene** (`viewer/calibration.py`): one real inference pass records the detections and derives the clip's own frame geometry + legal direction (= the observed dominant traffic flow); the unchanged `run_wrong_way_slice` then reasons over those recorded detections against that scene. | Honest — *zero violations* when every vehicle travels with the flow; a genuine `ConfirmedEvent` for a vehicle that sustainedly opposes it. Requires the RT-DETR checkpoint cached locally. |

Neither path fabricates a detection: the built-in stub replays a caller-authored
script matched to a synthetic clip (a COCO RT-DETR does not fire the vehicle class
on synthetic pixels — see the CLI README), and the upload path runs real inference
(the slice pass replays the calibration pass's *recorded* real RT-DETR output
verbatim — `detector_kind: RTDetrCapturedReplay` — so the clip is inferred once,
not twice).

### Upload scene calibration (why it exists)

The upload path previously reasoned every clip against the repository's *synthetic*
example scene (1920×1080, legal direction "up"), which matches no real footage and
structurally produced zero events. `viewer/calibration.py` fixes this at the
demonstration layer — the backend is untouched: it derives the clip's real frame
size and its observed dominant traffic-flow direction from substantial tracks
(alive ≥ 1 s, net motion ≥ 40 px), authors a validated `SceneConfig` in the clip's
own pixel space (provenance `auto_calibration`, status `draft`, no metric
calibration claimed), and feeds it to the existing runner. Wrong-way under this
calibration means exactly: *a vehicle sustainedly opposing the dominant traffic
stream* (>120° deviation for ≥1.0 s — the same provisional thresholds as the
example scene).

To positively verify the wrong-way path end-to-end, generate a validation clip
(a real vehicle crop from your footage composited moving *against* the flow — a
constructed scenario, genuinely analyzed; see the script's docstring):

```bash
./.venv/Scripts/python.exe demo/make_wrong_way_upload_clip.py
# writes runs/demo/clips/wrong_way_upload_validation.mp4 — upload it in the Viewer
```

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
