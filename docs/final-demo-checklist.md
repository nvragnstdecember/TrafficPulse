# TrafficPulse — final demo checklist

**An operator checklist, not a report.** Work down it in order. Everything here has
been verified on this machine; where a claim is bounded, the bound is part of the
line and must be read out with it.

The authority on what may be claimed is
[`validation-matrix.md`](validation-matrix.md). If this checklist and the matrix ever
disagree, the matrix wins.

---

## A. Before the demo

### A1. Environment

```bash
cd /d/Projects/TrafficPulse
./.venv/Scripts/python.exe -c "import trafficpulse, torch, transformers; print(trafficpulse.__version__)"
# expect: 1.1.0
```

- [ ] Repo root is the working directory. **Every command below assumes it.**
- [ ] `.venv` exists. `ruff`, `mypy`, `pytest` live **only** in `.venv` — not on PATH.
- [ ] No network needed. RT-DETR and the helmet backend resolve **offline** from the
      local HuggingFace cache (`local_files_only=True`).

### A2. Frontend build

```bash
cd frontend && npm run build && cd ..
ls frontend/dist/index.html
```

- [ ] `frontend/dist/` exists. It is **already built** — rebuild only if you changed UI code.
- [ ] There is **no Vite dev server** in the demo. The API serves the SPA itself.

### A3. Model / checkpoint verification

```bash
ls runs/helmet_cnn_vit/final/resnet50_lr0.001_s0/checkpoints/best.pt
```

- [ ] Helmet checkpoint present (needed by `serve_demo.py` only, never downloaded).
- [ ] RT-DETR `PekingU/rtdetr_r50vd` cached locally.

### A4. Backend launch — pick ONE

**Helmet analysis demo** (the composition that classifies riders and refuses to enforce):

```bash
TRAFFICPULSE_APP_STORAGE=runs/demo-ready/storage \
TRAFFICPULSE_APP_STATIC_DIR=frontend/dist \
.venv/Scripts/python -m uvicorn serve_demo:app --host 127.0.0.1 --port 8000
```

**Canonical production app** (the documented production entrypoint):

```bash
TRAFFICPULSE_APP_STATIC_DIR=frontend/dist \
TRAFFICPULSE_APP_SCENE=configs/scenes/example-scene.yaml \
.venv/Scripts/python -m uvicorn serve:app --host 127.0.0.1 --port 8000
```

- [ ] **`TRAFFICPULSE_APP_STATIC_DIR=frontend/dist` is not optional.** Omit it and the
      process serves the JSON API only: `/` returns a bare `{"detail":"Not Found"}` and
      so does every SPA route. The API still answers, so the server looks healthy while
      the browser shows raw JSON. *(Verified 2026-09-03.)* This is the single most likely
      way to break the demo in the first thirty seconds.
- [ ] `TRAFFICPULSE_APP_STORAGE` defaults to `trafficpulse-data/`. Point it at
      `runs/demo-ready/storage` for the pre-seeded library; drop it to start empty.

### A5. Frontend availability

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/          # 200 (SPA)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/evidence  # 200 (deep link)
curl -s http://127.0.0.1:8000/api/health
# {"status":"ok","version":"1.1.0","engine":"ready","repository":"ready",
#  "inference_available":true,"scene_configured":true}
```

- [ ] `/` returns HTML, not JSON.
- [ ] A deep link (`/evidence`) returns 200 — SPA fallback works, so refreshing mid-demo is safe.
- [ ] `inference_available: true`. If false, the engine is not composed and processing will 503.

### A6. Validation scripts — **pre-run these; do not run them live**

| Script | Wall time (CPU) | Expected |
|---|---|---|
| `demo/wrongway_calibrated_validation.py --stride 5` | ~30 min | **16** wrong-way events |
| `demo/red_light_validation.py` | ~12 min | **0** events, **0** crossings |
| `demo/triple_riding_validation.py --frames 60` | ~3 min | **0** events |
| `test-videos/run_manifest.py` | seconds | 18 passed, 0 failed, 13 not evaluated |
| `demo/controlled_demo.py --real-pixels` | seconds | writes `runs/controlled-demo/controlled-demo-pixels.mp4` |

- [ ] All four run and their transcripts are saved/scrolled-back where you can show them.
- [ ] The controlled clip exists. It is **generated, not committed** — regenerate it on
      the demo machine. Use `--real-pixels`: **RT-DETR detects nothing in the plain
      rectangle clip**, so uploading that one shows zero events. `--real-pixels` needs
      the `test-videos/` media to be present (§A6 assumes it is).
- [ ] Keep the **command visible next to the number**. A count without its command is meaningless.

### A7. Evidence verification

```bash
curl -s "http://127.0.0.1:8000/api/events?limit=5" | head -c 200
cat runs/red-light-clean001/red-light-clean001/validation-report.json
```

- [ ] `/api/events` returns a non-zero `total` (the Evidence page needs it to be interesting).
- [ ] The red-light validation report exists and reads `"confirmed_events": 0`.
- [ ] Annotated demo videos present under `runs/demo-ready/clips/*.annotated.mp4`.

### A8. Browser check

- [ ] Open **http://127.0.0.1:8000** — dashboard renders, footer shows **`Backend ok · Engine ready · v1.1.0`**.
- [ ] Click every sidebar item once. No blank page, no raw error.
- [ ] Zoom to ~110–125% for a projector. Check the **Live** page's metric labels are not clipped.
- [ ] Pick light or dark **before** you start and leave it (Settings → Appearance).

---

## B. Demo order

1. **Dashboard** — the repository at a glance.
2. **Videos** — the workspace; open a processed clip; annotated video + event list.
3. **Evidence** — pick an event; measurements, thresholds, rule trace, artifacts.
4. **Controlled demonstration** — one clip, declared context, four reasoners (C5).
5. **Live camera** — capability strip (this is where the honesty lives).
6. **Settings** — Runtime card: version, engine, calibration state.
7. **Validation transcripts** — the four pre-run results (§A6).

Total ~15–18 minutes at a comfortable pace.

---

## C. The demo set — five demonstrations

Chosen so each shows something the others cannot. **Do not add a sixth to fill time.**

C1–C4 are **real-footage** demonstrations, bounded by
[`validation-matrix.md`](validation-matrix.md). C5 is a **controlled demonstration on
synthetic footage** and belongs to a different category — introduce it as such every
single time, before you show anything.

### C1. Wrong-way on real footage — the one real positive

- **Setup:** pre-run transcript + persisted events under `runs/wrongway-calibrated/`.
- **Clip:** `test-videos/wrong-way/wrongway_001.ogv` (Australia, roundabout).
- **Expected:** **16 confirmed wrong-way events**; 345 frames, 1708 detections,
  measured flow dx≈0.998 dy≈0.056, 24 movers, scene `781cface…`.
- **Say:** "Real RT-DETR, real tracker, real reasoner, real persisted events — against a
  legal direction *derived from the footage itself*. The rule detects sustained
  opposition to the scene's declared legal direction, which is exactly what it claims."
- **Do NOT claim:** that anyone drove illegally. **This is signed, lawful contraflow.**
  Also do not quote 16 without the `--stride 5`: the direction is measured, so the
  sampling changes the scene hash and can change the count.

### C2. Red-light on real footage — the correct zero

- **Setup:** pre-run transcript + `runs/red-light-clean001/.../validation-report.json`.
- **Clip:** `test-videos/normal-traffic/clean_001.ogv` (Chiang Mai, signalised).
- **Calibration required:** analyst-authored stop line `(900,940)→(1900,860)`, junction
  polygon, signal ROI, and a **signal schedule transcribed from the footage**
  (`MANUAL_ANNOTATION`). All of it lives in `demo/red_light_validation.py`.
- **Expected:** 434 frames, 107 vehicle tracks, **0 junction entries, 0 events**.
- **Say:** "The signal is transcribed, not invented — the head *and* a countdown are in
  frame, the countdown ticks 58 down to 44, and the red lamp is lit in every sampled
  frame. The approach is red, its traffic is stopped, nothing crosses the stop line, and
  the system confirms nothing. Drawn wrongly across the *cross* traffic, the same scene
  would have produced 4 crossings and confirmed all of them — falsely. The zero is a
  property of correct calibration."
- **Do NOT claim:** that red-light jumping is *detected*. **Nothing crossed, so the latch
  was never exercised.** This buys a real-footage run and a false-positive check —
  nothing about whether a genuine violator would be caught.

### C3. Triple riding, Raxaul — conservative abstention

- **Setup:** `demo/triple_riding_validation.py --frames 60` (~3 min, or pre-run).
- **Clip:** `test-videos/edge-cases/congestion/congestion_002.webm`.
- **Expected:** 79 rider-count observations, 6 motorcycles, best track `iou-7` with
  **75.5 % of its observations at ≥3 riders**, longest uninterrupted ≥3 run **0.50 s**
  against a **1.0 s** threshold → **0 events**.
- **Say:** "Three riders in three-quarters of one motorcycle's frames, and the system
  still confirms nothing, because the evidence never held for the required second. The
  gap is single-frame count dips, and the rule is right to end the run: a count of 1
  *asserts* one rider, and `RiderCountObservation` has no channel to say 'this frame is
  untrustworthy'. The fix belongs in the observation contract, not in the threshold."
- **Do NOT claim:** a detection rate. And do not describe the zero as a failure — it is
  the temporal guarantee working. Never suggest lowering the threshold.

### C4. Helmet perception — and the refusal to enforce

- **Setup:** `serve_demo.py` (§A4), live in the browser, ~55 s per clip.
- **Clip:** `runs/demo-ready/clips/raxaul-congestion.mp4` (lead with this one).
- **Expected:** every rider classified; `MULTI-RIDER — DRIVER UNRESOLVED`; **no**
  `ConfirmedEvent` at all.
- **Say:** "It classifies every rider and declines to name a violator it cannot
  attribute. Enforcement is off because of a guard we wrote deliberately: this model
  cannot emit `turban`, so the exemption could never fire."
- **Do NOT claim:** that helmet violations are detected (no helmet rule runs, nothing is
  confirmed); that multi-rider traffic is handled (it is explicitly unresolved); or
  **P4-U5's 0.929** — that was measured on whole-motorcycle crops, not the runtime's head
  crops, and is not a runtime number.

------------------------------------------------------------

### C5. The controlled demonstration — one video, four reasoners

**Category warning: this is synthetic footage. Say so before you click anything.**

- **Setup:** `demo/controlled_demo.py --real-pixels` (§A6), then upload the clip in
  the Videos workspace. Full walkthrough:
  [`controlled-demo.md`](controlled-demo.md) §5c.
- **Clip:** `runs/controlled-demo/controlled-demo-pixels.mp4` — 1920×1080, 6 s, four
  hand-authored situations rendered from **real vehicle crops** cut from this
  project's own corpus, on a plain synthetic road.
- **Calibration to draw** (the script prints these for whichever clip it wrote):
  lane `(560,0)(1200,0)(1200,1080)(560,1080)`; direction dragged **downward**;
  junction `(560,560)(1200,560)(1200,800)(560,800)`; stop line `(560,440)→(1200,440)`;
  no-stopping `(1240,600)(1880,600)(1880,1000)(1240,1000)`; dwell `3`;
  signal `0s Red`, `2s Green`.
- **Expected:** four confirmed events — wrong way, illegal stopping, red-light jumping,
  triple riding — and the comparison table reading **expected 4 · matched 4 · missing 0
  · unexpected 0**.
- **Say:** "I give it one video and declare the context the camera cannot know — which
  way traffic legally travels, where stopping is prohibited, where the stop line is,
  and what the signal was showing. The same pipeline then has four separate reasoners
  each reach their own conclusion. Note the signal turns green *after* the vehicle
  crossed, and the event still confirms: the state is latched at the crossing. And note
  what I declared is never shown to the reasoners — it lives in a store no rule can
  read, and a family they decline to confirm is reported as *missing*, not conjured."
- **Do NOT claim:** that this shows detection accuracy, or anything about real
  footage. RT-DETR is genuinely running here, but it is looking at **unchanging
  vehicle cut-outs sliding across an empty synthetic road** — no occlusion, no
  perspective change, no lighting variation, no other traffic. Say "four violations
  were *reasoned*", never "four violations were *detected*".
- **If asked "so does it work on real video?":** point at C1–C4 and the matrix. That is
  the honest answer, and it is a better one than a synthetic positive.

---

## D. Backup plan

- [ ] **Screenshots** of all six routes captured beforehand, in case the browser or the
      projector misbehaves.
- [ ] **Pre-generated evidence** already on disk: `runs/demo-ready/storage` (four
      processed videos), `runs/wrongway-calibrated/`, `runs/red-light-clean001/`.
- [ ] **Pre-annotated videos**: `runs/demo-ready/clips/*.annotated.mp4` — play these
      directly if live processing is risky, and *say* they were produced earlier by the
      same command.
- [ ] **A screen recording** of one full upload→process→review cycle is worth having if
      the room's machine is not this one.
- [ ] **Fallback launch:** if `serve_demo` fails to find the helmet checkpoint, launch
      `serve:app` instead (§A4) and drop demo C4 — the other four are unaffected.
- [ ] **If the controlled demo goes wrong live:** `demo/controlled_demo.py --api
      http://127.0.0.1:8000` does the same flow over HTTP and prints the comparison
      table. Have its output pre-captured as well.
- [ ] **If the port is busy:** change `--port`, and remember the browser URL changes with it.
- [ ] **Shutdown:** Ctrl-C in the uvicorn terminal. Nothing needs cleaning up.

---

## E. Five sentences to have ready

1. "A violation is not a model output — it is sustained, typed evidence over a track,
   checked against scene geometry, and reviewable."
2. "Nothing in our real-footage corpus is labelled, so we report *observations*, never
   accuracy. There is no precision or recall figure for any violation, and we do not
   quote one."
3. "The signal state is declared or transcribed, never perceived — nothing here
   classifies a traffic light from pixels."
4. "Where the system cannot attribute a violation to a responsible party, it abstains
   and says why, rather than guessing."
5. "The one thing we will not do is tune a threshold until a clip produces the answer we
   wanted for the demo."
