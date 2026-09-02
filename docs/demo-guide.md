# TrafficPulse demonstration guide

- **Entrypoint:** `serve_demo.py` (the production entrypoint remains `serve.py`)
- **Posture:** helmet **classification** is on; helmet **violation enforcement** is off
- **Evidence base:** [`helmet-runtime-evaluation.md`](helmet-runtime-evaluation.md) (P4-U8/U9/U10)

This document says what the demo shows, what may be claimed from it, and what may not.
It adopts no backend and changes no experimental result. Every number it cites lives in
the evaluation document; nothing is restated here that could drift from it.

---

## 1. Runbook

### 1.1 Launch (one terminal, one command)

```bash
cd /d/Projects/TrafficPulse
TRAFFICPULSE_APP_STORAGE=runs/demo-ready/storage \
TRAFFICPULSE_APP_STATIC_DIR=frontend/dist \
.venv/Scripts/python -m uvicorn serve_demo:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000**. The API serves the built SPA itself, so there is
no second terminal and no Vite dev server to go wrong. `frontend/dist` is already
built; rebuild with `cd frontend && npm run build` only if you change the UI.

`TRAFFICPULSE_APP_STORAGE=runs/demo-ready/storage` points at the library seeded on the
night of 2026-09-02, so four already-processed videos are present the moment the page
loads. That is the fallback (§1.6). Drop the variable to start empty.

**Expected startup** — roughly 15-20 s (model composition is lazy; the checkpoint loads
on the first job), then:

```
INFO  trafficpulse.app  logging configured at INFO
INFO  Uvicorn running on http://127.0.0.1:8000
```

Sanity check in another shell:

```bash
curl -s localhost:8000/api/health      # {"status":"ok", ... "inference_available":true}
curl -s localhost:8000/api/system/posture
```

`inference_available: true` and a posture whose `helmet_enforcement` is `disabled` mean
you are good. If `inference_available` is `false`, the RT-DETR checkpoint is not in the
local HuggingFace cache.

### 1.2 Prerequisites

```bash
pip install -e ".[dev,api,overlay,rtdetr]"
```

The demo needs one local artifact production does not: the P4-U5 ResNet-50 checkpoint
(`runs/helmet_cnn_vit/final/resnet50_lr0.001_s0/checkpoints/best.pt`, or
`TRAFFICPULSE_DEMO_HELMET_CHECKPOINT`). It is never downloaded. RT-DETR resolves offline
from the local HuggingFace cache exactly as in production.

### 1.3 Recommended clips

Pre-trimmed to **30 contiguous frames** and ready in `runs/demo-ready/clips/`:

| Clip | Why show it | Riders | What it demonstrates |
|---|---|---|---|
| **`raxaul-congestion.mp4`** — *lead with this* | Large, unmistakable head crops (median 143-196 px) | 3 on **one** motorcycle | `MULTI-RIDER — DRIVER UNRESOLVED` on every rider; helmet, no-helmet and uncertain readings side by side |
| **`gangtok-congestion.mp4`** | Both branches in one clip | 3 across 2 bikes | One lone rider `eligible` with `helmet` at 93%; two unresolved |
| `chiangmai-intersection.mp4` *(optional)* | The honest hard case | 11 across 8 bikes | Busiest frame; also the small-head regime — several crops under 30 px, two abstentions |
| `contraflow-roundabout.mp4` *(optional)* | The correct negative | 0 | No motorcycles: zero riders, zero events, no annotated video. Right answer, not a failure |

Each is ~1 second of source at full resolution. The originals in `test-videos/` are
untouched; these are copies.

### 1.4 Processing time (measured 2026-09-02, CPU)

| Clip | Frames | Wall | s/frame | Annotated output |
|---|---|---|---|---|
| gangtok-congestion | 30 | **58.3 s** | 1.94 | 1.37 MB |
| raxaul-congestion | 30 | **54.3 s** | 1.81 | 1.56 MB |
| chiangmai-intersection | 30 | **62.3 s** | 2.08 | 1.68 MB |
| contraflow-roundabout | 30 | **51.2 s** | 1.71 | none (nothing to draw) |

Wall time covers the whole request the presenter waits through: scene auto-calibration,
inference, persistence **and** the annotated-video render.

**Live processing is recommended.** Just under a minute per clip is a workable pause —
it is roughly the time it takes to explain the pipeline diagram. Do not upload anything
longer without re-measuring: cost is linear, so a 10-second clip is ~10 minutes.

### 1.5 UI sequence

1. Open **http://127.0.0.1:8000** → the dashboard loads.
2. **Videos** in the sidebar → the workspace.
3. Drop `runs/demo-ready/clips/raxaul-congestion.mp4` on the dropzone (or pick it from
   the library if you are using the fallback).
4. Press **Process**. The processing panel shows live phase and frame progress.
5. Wait ~55 s. Talk over it: the pipeline diagram in §2 is exactly what is running.
6. On completion the player switches by itself to the **annotated video** — boxes,
   association chains, head-crop boxes, per-rider captions, and a standing
   `HELMET ANALYSIS` banner.
7. Scroll to the **Helmet analysis** panel → summary tiles, then expand **Riders**.
8. Scroll to **System capabilities** → the posture strip.
9. Point at the event list: empty, and it says *why*.

### 1.6 Fallback if live processing is risky

Verified after a real server restart on the seeded storage:

- all four videos appear in the library;
- their jobs recover as `succeeded`;
- **the annotated videos still play** (1.37-1.68 MB, served over `/api/process/{id}/overlay`).

So: **Videos → pick a video from the library → the annotated result plays immediately.**
The boxes, captions and the `MULTI-RIDER / DRIVER UNRESOLVED` marking are drawn *into*
the frames, so the whole visual story survives.

**One thing does not survive a restart:** the Helmet analysis *panel*. The fold is
derived in-process and deliberately never persisted, so a recovered job returns 404 and
the panel does not render. If you need the panel, re-run the clip live — it is ~55 s.

Second-line fallback: the annotated MP4s in `runs/demo-ready/clips/*.annotated.mp4` play
in any video player, and `runs/demo-ready/readiness.json` holds the exact API responses.
Both are genuine TrafficPulse outputs generated on 2026-09-02; nothing is staged.

### 1.7 Readiness check and shutdown

```bash
# re-verify everything end to end (~4 min, four clips)
.venv/Scripts/python demo/run_demo_smoke.py --frames 30 --out runs/demo-ready/readiness.json
```

It exits non-zero if enforcement is not disabled, turban is not unavailable, a shared
motorcycle is attributed, a helmet violation is emitted, or anything crashes.

Shut down with **Ctrl-C** in the uvicorn terminal. Nothing needs cleaning up: uploads,
runs and annotated videos live under `runs/demo-ready/storage` and are safe to delete.

## 2. The flow

```text
Video / image upload            POST /api/video/upload
        ↓
Scene auto-calibration          derived from the clip's own motion (no geometry invented)
        ↓
RT-DETR detection               person + motorcycle, score ≥ 0.50
        ↓
IoU tracking                    per-object track identity
        ↓
Rider association               rider ↔ motorcycle (IoMin overlap)
        ↓
Head-crop extraction            top 30% of the rider box, full width
        ↓
Quality gate                    tiny / off-frame / no-pixel crops abstain before inference
        ↓
Helmet classifier               trained ResNet-50, T = 2.2298, abstain below 0.80
        ↓
Temporal stabilization          per-track plurality vote over a 5-sample window
        ↓
Analysis fold                   per rider: reading + enforcement status  (NO event minted)
        ↓
Workspace                       annotated video · helmet panel · capability strip · events
```

Everything from detection to the head crop is the **same code** a `no_helmet` rule run
would execute. The difference is what happens after: an analysis has no reasoner and no
finalize strategy, so it is structurally incapable of producing a `ConfirmedEvent`.

## 3. Why the demo classifies but does not enforce

Two facts, both established before this demo existed:

1. On runtime crops the trained ResNet-50 is far the strongest classifier available, and
   the shipped zero-shot backend is not doing the job at all (evaluation §3.1).
2. The trained model is **binary**. It cannot emit `turban`, and the no-helmet rule's
   exemption depends on `turban` observations. Building that rule on it would leave the
   exemption permanently dead and confirm turban-wearing riders as violators — a
   systematic false-positive class against a religious group (evaluation §4.1).

The classifier capability guard refuses that combination. The demo **does not bypass
it**: `acknowledge_turban_blind` is not set anywhere, and the rule derivation asks the
guard, so `no_helmet` is never even offered for this deployment. What the demo adds is
the missing third option — *classify without enforcing* — so that refusing the rule no
longer means losing helmet perception entirely.

### A second, independent blocker (found 2026-09-02)

Turban is **not** the only thing between here and enforcement, and it would be a costly
mistake to think so. The shipped no-helmet reasoner **never reads `rider_slot`**. It is
derived correctly and travels on every observation, but `rules/no_helmet.py` filters
only on exemption and a missing track id — so a **pillion passenger** with a sustained
bare-headed run is confirmed, and named on the event, exactly as a lone driver would be.

Measured directly against the shipped reasoner:

| `rider_slot` | confirmed events |
|---|---|
| `driver` | 1 |
| `unknown` (multi-rider) | **1** |
| `pillion` | **1** |

That is the opposite of what enforcement requires. Switching the rule on today — *even
on a turban-capable backend* — would attribute helmet violations to passengers across
the 42.4% of the frozen corpus and the 81% of a real congestion clip that are
multi-rider.

It is latent rather than live: the demo runs analysis, which mints no event, so nothing
is presently wrong in the running system. Closing it means the reasoner must **abstain**
on any rider whose slot is not `DRIVER` — a refusal, never a guess — and that change
needs its own review and re-validation, not a night before a demo.
`tests/rules/test_no_helmet_enforcement_preconditions.py` pins the gap so it cannot
close, widen, or be forgotten silently.

Every other violation family (wrong-way, illegal stopping, red-light jumping, triple
riding) is untouched and runs normally wherever the resolved scene supports it.

## 4. Multi-rider: the system declines, on purpose

When two or more riders are associated with one motorcycle, TrafficPulse reports
**`MULTI-RIDER — DRIVER UNRESOLVED`** and attributes nothing. It does not pick the
front-most, the largest, the lowest, or the first-tracked rider. Driver-versus-pillion
needs the motorcycle's travel direction; the shipped IoU tracker supplies no velocity,
so which end of the bike is the front is genuinely unknown.

This is worth demonstrating rather than hiding. It is 42.4% of the frozen evaluation
corpus and 81% of crops in a real congestion clip (evaluation §4.2) — the common case,
not an edge case — and a system that guessed here would attribute a violation to a
passenger.

## 5. Temporal stabilization: what it is and is not

P4-U10 §5.1(7) measured per-frame label flips on the majority of real tracks. Displayed
raw, that is unreadable. The demo therefore shows, per rider track, a **plurality vote
over the last five classified frames**, with the window's agreement beside it and an
unsettled track marked as such.

It is a **display aid**. It has not been evaluated on the frozen split or anywhere else,
its window length was chosen for legibility rather than tuned, and it makes no accuracy
claim. Critically, it does **not** feed the no-helmet rule: that rule's temporal-run
semantics are its own and remain unevaluated against real per-frame output, which is
still open blocker 3 in the evaluation. The panel reports the raw flip count beside the
smoothed label so the instability being managed stays visible.

## 6. What the demo may be said to show

- RT-DETR detection and IoU tracking on real traffic footage.
- Rider↔motorcycle association and the derived head-crop geometry.
- A trained classifier reading those runtime crops, with calibrated, pre-committed
  abstention.
- Per-track temporal smoothing of an unstable per-frame signal.
- The system **declining** to attribute a helmet state to a rider whose role it cannot
  determine, and declining to enforce a rule whose evidence it does not have.
- The end-to-end product: upload → calibrate → process → annotated video → review, and
  the non-helmet violation families end to end.

## 7. What the demo may **not** be said to show

- **That helmet violations are detected.** No helmet rule runs; no helmet event exists.
- **That multi-rider traffic is handled.** It is explicitly unresolved.
- **That the turban exemption works.** Its evidence is not demonstrated on *any*
  backend — including the zero-shot one, whose turban predictions land overwhelmingly on
  riders annotated as helmeted (evaluation §4.1).
- **That smoothing improves accuracy.** Unvalidated, presentation only.
- **P4-U5's 0.929.** That was measured on whole-motorcycle crops from the dataset's own
  annotation, not on the head crops the runtime produces. It is not a runtime number and
  must never be quoted as one.
- **Any accuracy figure as end-to-end performance.** The published classifier numbers are
  per-crop and conditional on a rider reaching the classifier at all. End to end, on the
  frozen test split, a helmet decision is produced for roughly 38% of annotated
  motorcycles (evaluation §3.4).
- **Anything about night or rain.** No such clip exists in `test-videos/`; that regime is
  untested, not passed.

## 7a. Five things to say, and five not to

**Say:**

1. "A violation is not a model output. It is sustained, typed evidence over a track,
   checked against scene geometry, and reviewable — that separation is the whole design."
2. "This backend is the strongest we measured on the crops the runtime actually
   produces: 0.857 native macro-F1, per-crop, on single-rider crops — with 74.4%
   recovery and a 42.4% multi-rider exclusion in the same breath."
3. "Two riders on one motorcycle: we do not know which is the driver, so we say so. The
   tracker gives no velocity, and guessing would accuse a passenger."
4. "Helmet enforcement is off, and it is off because of a guard we wrote deliberately:
   this model cannot say 'turban', so its exemption could never fire."
5. "Per-frame labels flip on real footage. We smooth them for display, we show the raw
   flip count next to the smoothed label, and we do not claim smoothing makes it more
   accurate."

**Do not say:**

1. "TrafficPulse detects helmet violations." — No helmet rule runs. Nothing is confirmed.
2. "92.9% accurate." — P4-U5 measured whole-motorcycle crops, not runtime head crops.
   That number says nothing about this pipeline.
3. "It handles multiple riders." — It explicitly declines to.
4. "The turban exemption works." — Not demonstrated on any backend, including the
   turban-capable one.
5. "It's production-ready" / "it's accurate on this footage." — Tonight's clips are a
   reliability check, not a measurement, and night and rain are untested.

## 8. The capability strip

`GET /api/system/posture` computes, from configuration alone, what the deployment may
claim. The workspace renders it beside the run. Under `serve_demo` it reads:

| Capability | State | Why |
|---|---|---|
| Detection | LIMITED | Runs at the validated threshold; recall is not complete. |
| Helmet classification | ACTIVE | Crops are classified and smoothed for display. |
| Driver attribution | LIMITED | Single-rider only; multi-rider is unresolved by design. |
| Turban exemption | UNAVAILABLE | The configured backend declares it cannot emit `turban`. |
| Helmet violation enforcement | DISABLED | This deployment classifies; it does not enforce. |

There is deliberately **no configuration in which helmet enforcement reports ACTIVE**.
With the turban-capable zero-shot backend and no analysis declared, the rule can build
and the strip reports EXPERIMENTAL — because the rule's confirmation semantics have
never been evaluated against real per-frame classifier output.

## 9. Failure behaviour worth knowing

| Situation | What happens |
|---|---|
| Unsupported / corrupt upload | 400 with a typed error; the workspace shows it and stays usable. |
| A clip with no motorcycles | Zero riders, zero events, no annotated video. This is correct, and the panel says so. |
| Crops too small or off-frame | The quality gate abstains **before** inference; counted and reported, never guessed. |
| Classifier failure at inference | The job goes `failed` with the message; every read endpoint keeps serving. |
| Overlay render failure | The job stays `succeeded`; events and evidence are already persisted and the original video still plays. |
| A run recovered after a restart | Reports no helmet analysis (the fold is derived in-process, not persisted) rather than a reconstructed one. |
