# Controlled demonstration mode

**One video, declared scene context, four independently reasoned violation types.**

This document describes what Controlled Demo Mode is, why scene calibration is
required at all, what each field means, and — most importantly — **what it does not
prove**.

The authority on what may be claimed anywhere in this project is
[`validation-matrix.md`](validation-matrix.md). If this document and the matrix ever
disagree, the matrix wins.

---

## 1. The problem it solves

TrafficPulse ships five violation reasoners. The project's real-footage corpus
(`test-videos/`) contains **no single clip** with a red-light runner, a wrong-way
vehicle, an illegally stopped car and an overloaded motorcycle in it — and it never
will, because such footage is not something you can obtain on demand, and
manufacturing the result by lowering a threshold until real footage produces the
wanted answer is exactly what this project refuses to do.

So the honest alternative is a **declared** scenario: a hand-authored clip, a scene
an operator draws, and a written statement of what was built into it. That is
Controlled Demo Mode.

## 2. Why calibration is required at all

A camera records pixels. It does not record:

| Fact | Why it is not in the pixels |
|---|---|
| Which way traffic **legally** travels | A vehicle going left is only a violation if left is the wrong way, which is a fact about the road, not about the vehicle. |
| Where stopping is **prohibited** | A no-stopping zone is a legal and operational designation. Traffic flowing over a stretch of road tells you nothing about whether stopping there is allowed. |
| What the **signal** was showing | Nothing in TrafficPulse classifies a signal head from pixels. The state is declared or transcribed, always. |
| How long a vehicle may **dwell** | A threshold is a policy choice about the site. |

TrafficPulse reasons over these; it does not infer them, and it does not verify
them. The calibration surface exists to let an operator supply exactly that missing
context — and the UI says so, in those words, above the drawing tools.

The one fact that *is* observable is the dominant traffic flow, and it is the one
thing the system derives on its own (`scenes/calibration.py`, opt-in per
deployment). It abstains rather than guessing when a clip's traffic does not define
a single direction.

## 3. What each calibration field means

Drawn on the frame, in the **video's own pixel space** (origin top-left, +x right,
+y down). The backend rejects a drawing whose frame size does not match the video's
decoded dimensions, so there is exactly one coordinate system.

| Field | What it declares | What it unlocks |
|---|---|---|
| **Lane** | The carriageway being monitored. Every scene needs one. | Nothing on its own — it is what the direction governs. |
| **Traffic direction** | The direction traffic legally travels, as an arrow dragged over the lane. Stored as a unit vector on a `LegalDirection`. | Wrong way |
| **No-stopping zone** | A polygon where stopping is prohibited. **Not observable** — a policy fact. | Illegal stopping |
| **Stop line** | The line whose crossing commits a vehicle to the junction. The *entry direction* is derived from the line's normal oriented toward the junction, so an analyst cannot point it backwards. | Red-light jumping |
| **Junction area** | The conflict area beyond the stop line. Keep it clear of the line, so a vehicle stopping just past the line is not counted as having entered. | Red-light jumping |
| **Signal head** | The ROI of the head this approach obeys. Optional — nothing classifies it; it exists so the scene can name the signal group whose declared schedule governs the stop line. | — |
| **Signal timing** | A step function of media time: "state X from t seconds". **Per run, not per scene** — a phase names an instant in one clip, and a scene is shared across many. | Red-light jumping (required; a schedule-less rule is refused at build time) |
| **Stopping dwell** | Seconds a vehicle may stand inside the no-stopping zone. Default 5 s, and the default is a *placeholder* — leaving the field blank sends nothing, so the scene never records a value the operator did not choose. | — |
| **Scene notes** | Free text stored **inside** the scene, so it travels with the geometry and is readable by anyone resolving an event's `scene_config_hash` months later. | — |

Everything an analyst sets is stamped `ParameterStatus.PROVISIONAL` in the stored
scene: operator-chosen, **not** tuned or validated against ground truth.

### The one thing the timeline exists for

A schedule whose first phase starts after 0 s leaves the head of the clip
**`unknown`** — and `unknown` never confirms. In a list of numbers that gap is
invisible; on the timeline bar it is the first thing you see, and the editor says so
explicitly.

## 4. Expected vs detected

A controlled clip may carry a **declaration**: the violation families it was built to
contain, plus a written statement of what the scenario is.

This is ground truth for a *demonstration*, and the separation from detections is
structural, not a promise:

- it is stored in `<storage>/expectations/`, a directory no rule, reasoner or engine
  ever opens;
- `ProcessRequest` has **no** field for it, and neither does `EngineConfig` — a test
  asserts both, so an expectation cannot reach the engine even by accident;
- it never appears in `/api/events`, in any count, or in any listing;
- `GET /api/videos/{id}/expectation/comparison` is the only place the two meet, and
  it consumes already-persisted events.

The comparison reports, per family:

| Outcome | Meaning |
|---|---|
| **Matched** | Declared, and independently confirmed. |
| **Not detected** | Declared, and nothing was confirmed. **Not a defect on its own** — either the evidence never met the rule's threshold, or the rule was not run. |
| **Unexpected** | Not declared, but confirmed anyway. Worth opening. |

Every `detected_count` is the length of a list of real `event_id`s, each of which
opens into the same evidence viewer as any other event.

**No accuracy is computed, anywhere.** Precision, recall or F1 over one hand-authored
clip would be arithmetic on a sample of four against ground truth the same person
wrote. The API omits them, the UI omits them, and tests assert their absence.

## 5. Running the controlled demonstration

### 5a. The shipped scenario

`trafficpulse.scenes.demo_scenario` is the single specification of a 480×270, 6-second
clip containing four deliberately separated situations:

| Actor | Scenario |
|---|---|
| `rl-runner` | Crosses the stop line while the declared signal is **red**, then enters the junction. The schedule turns **green at 2.0 s — after it crossed at 1.2 s** — so the run also demonstrates the H13 latch. |
| `ww-driver` | Climbs the same lane against the declared legal direction for 3.7 s (threshold 1.0 s). Going *up*, its stop-line crossing is **backward**, which clears the crossing flag rather than setting it, so it can never be confirmed for red-light. |
| `is-stopper` | Pulls onto the right shoulder, inside the declared no-stopping zone, and holds. Its ground-contact point is outside the lane, so no heading is derived and wrong-way cannot see it. |
| `tr-motorcycle` | Carries three riders the whole clip on the left verge, outside every governed polygon — visible only to the geometry-free rider-count rule. |

The separation is not incidental. `tests/scenes/test_demo_scenario.py` checks it
geometrically against the same polygons the scene declares, so an edit that nudges an
actor into the wrong zone fails there rather than silently weakening the demo.

### 5b. Two renderings of the same scenario — and which one to use

The scenario has **one** definition and **two** renderings. Both use the same
trajectories, the same declared scene and the same expectations; they differ only in
what the pixels are.

| | Rectangle clip | Composited clip |
|---|---|---|
| Command | `demo/controlled_demo.py --render-only` | `demo/controlled_demo.py --real-pixels` |
| Pixels | Coloured boxes on black | Real vehicles cut from `test-videos/`, on a plain road canvas |
| Size | 480×270, ~30 KB | 1920×270×4 = 1920×1080, ~640 KB |
| Detector | **Scripted** (replays the drawn boxes) | **Real RT-DETR** |
| Needs corpus media | No | **Yes** (`test-videos/`, gitignored) |
| Used by | the test suite, CI | the browser demonstration |

**RT-DETR detects nothing in the rectangle clip.** This is measured, not assumed: 0
detections on every sampled frame. Uploading it to a running server produces zero
events, which is why `--render-only` prints a warning saying so. Use it for anything
a machine checks; use `--real-pixels` for anything a person watches.

Neither clip is committed. Both are rebuilt from the specification in about a second,
which is cheaper and more honest than storing a binary whose provenance nobody can
check.

```bash
./.venv/Scripts/python.exe demo/controlled_demo.py --render-only     # rectangles
./.venv/Scripts/python.exe demo/controlled_demo.py --real-pixels     # real crops
./.venv/Scripts/python.exe demo/controlled_demo_pixels.py --verify   # + run real RT-DETR
```

#### What the composited clip authors, exactly

It follows `demo/make_wrong_way_upload_clip.py`'s discipline: **it authors the
scenario, never the analysis.** RT-DETR runs real inference on the composited pixels
and must actually find the vehicles; the real tracker associates them; the real
derivations and reasoners decide. No detection, track, observation or event is
fabricated. What *is* constructed, stated plainly:

- the **canvas** is a plain asphalt background with lane markings, not real road
  footage. It is deliberately empty: a real backdrop would contribute its own
  vehicles, whose tracks would be uncontrolled actors in a controlled demonstration;
- the **crops** are real vehicles (a car from `clean_003.webm` frame 0; a motorcycle
  carrying three riders from `congestion_002.webm` frame 80 — both recorded in the
  script with their source clip, frame index and box), but they do not change
  appearance as they move: no perspective, no lighting change, no wheel rotation;
- the **trajectories** are the authored scenario's.

One rendering detail is load-bearing rather than cosmetic: the motorcycle photograph
is nearly square and the scenario's bike-plus-riders block is wider than tall.
Stretching one into the other squashes the machine until **RT-DETR stops detecting a
motorcycle at all**, and triple riding then correctly reports *not detected*. The
compositor fits the crop without distorting it, anchored at the ground-contact edge.

### 5c. Drive it in the browser (the demo a viva wants to see)

1. Start TrafficPulse with the SPA (see [`deployment.md`](deployment.md)).
2. Open **Videos** and upload `runs/controlled-demo/controlled-demo-pixels.mp4`.
3. In **Scene calibration**, draw the geometry the script printed. For the
   composited (1920×1080) clip that is:
   - **Lane**: `(560,0) (1200,0) (1200,1080) (560,1080)`
   - **Traffic direction**: drag downward inside the lane
   - **Junction area**: `(560,560) (1200,560) (1200,800) (560,800)`
   - **Stop line**: `(560,440) → (1200,440)`
   - **No-stopping zone**: `(1240,600) (1880,600) (1880,1000) (1240,1000)`
   - **Stopping dwell**: `3`

   (The rectangle clip is a quarter of those coordinates; the script prints whichever
   applies to the clip it just wrote.)
4. In **Signal timing**, add phases `0 s → Red` and `2 s → Green`.
5. **Save scene.** The chip reports what it unlocked.
6. In **Controlled demo · expected vs detected**, select all four families, write the
   test context, and **Declare**.
7. **Re-run analysis with this scene.**
8. Read the comparison table, then open each event's evidence.

### 5d. Or drive it over HTTP

```bash
./.venv/Scripts/python.exe demo/controlled_demo.py --real-pixels --api http://127.0.0.1:8000
```

Every request it makes is one the Videos workspace makes. Nothing it sends is
privileged. Verified against `serve:app` with real RT-DETR on 2026-09-03:

```
Scene       : 305791950542fa64...  unlocks wrong_way, illegal_stopping,
                                   red_light_jumping, no_helmet, triple_riding
Expectation : 4 families declared

  family                 expected   detected   outcome
  triple_riding          yes        1          matched
  red_light_jumping      yes        1          matched
  wrong_way              yes        1          matched
  illegal_stopping       yes        1          matched

  expected 4 | detected 4 event(s) | matched 4 | missing 0 | unexpected 0
```

That run is **real inference** — 647 detections over 60 frames, the real IoU tracker,
and all four reasoners. It is still a controlled demonstration, for the reasons in §6.

### 5e. Reproducing it in the test suite

```bash
./.venv/Scripts/python.exe -m pytest tests/app/test_app_controlled_demo.py -q
./.venv/Scripts/python.exe -m pytest tests/scenes/test_demo_scenario.py -q
```

## 6. What this proves, and what it does not

**Proves:**

- one video → one analysis → four separate reasoners → four violation types, each
  confirmed under its own threshold, persisted, with its own evidence manifest;
- the geometry-dependent rules are genuinely unavailable until the geometry is
  declared, and become available when it is;
- an operator-chosen dwell threshold reaches the reasoner through the scene;
- the red-light latch survives the signal turning green after the crossing;
- a declared expectation cannot manufacture, influence or alter a single event.

**Does not prove — and must never be quoted as:**

- **any detection accuracy or real-world performance.** On the rectangle clip the
  detector is scripted outright. On the composited clip RT-DETR is real, but it is
  looking at unchanging vehicle cut-outs sliding across an empty synthetic road —
  no occlusion, no perspective change, no lighting variation, no clutter, and no
  other traffic. Detecting a pasted car there says nothing about detecting a car in
  Raxaul at dusk.
- **that four violations were "detected".** Four were *reasoned*, from context that
  was declared rather than perceived.
- **any real-world performance.** Real-footage findings are a separate category and
  live in [`validation-matrix.md`](validation-matrix.md), which states no precision or
  recall for any violation.
- **that the thresholds are right.** They are provisional and operator-chosen.

## 7. Three categories that must never be mixed

| Category | What it is | Where it lives |
|---|---|---|
| **Real-world validation** | Real footage, real RT-DETR, bounded observational claims, no ground truth | [`validation-matrix.md`](validation-matrix.md), `demo/*_validation.py` |
| **Controlled demonstration** | Hand-authored scenario + declared calibration + declared expectations | this document, `demo/controlled_demo.py` |
| **The frozen P4-U5 experiment** | The only benchmarked artifact in the repository, and it benchmarks a *classifier on crops*, not a violation | [`cnn-vs-vit-results.md`](cnn-vs-vit-results.md), [ADR-005](adr/ADR-005.md) |

Never report a number from one category as though it came from another.
