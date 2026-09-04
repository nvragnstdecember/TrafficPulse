# TrafficPulse validation matrix

**What is actually established, per violation — and what is not.**

This document exists because "the tests pass" and "it works on real footage" are
different claims, and a system that blurs them cannot be defended. Every row below
separates them explicitly, and every number is either reproducible by a command
printed here or is marked as not established.

It records *observations*, never accuracy. Nothing in the repository's real-footage
corpus is labelled, so no precision, recall or F1 exists for any violation, and none
is stated. Where a run produced a count, that count is a measurement of **that run at
that sampling** — not a detection rate, not a benchmark, and not something to tune
towards.

---

## 1. The five words, and why they are not interchangeable

| Word | Means | Does **not** mean |
|---|---|---|
| **tested** | Unit/integration tests exercise the logic against constructed inputs whose correct answer is known by construction. | That the logic sees what it needs on real pixels. |
| **observed** | The real path (real decode → real RT-DETR → real tracker → real reasoner) was run on real footage and produced a stated result. | That the result is correct — nothing was labelled. |
| **reproduced** | Re-running the *same* command on the *same* input yields the same result. | That a different sampling, clip or scene yields it. |
| **benchmarked** | Compared against ground truth by a pre-registered protocol. | Anything, anywhere in this table, for any violation. |
| **unvalidated** | Not run against real footage at all. | That it is broken — only that no evidence exists. |

Only the frozen P4-U5 CNN-vs-ViT helmet experiment is **benchmarked**, and it
benchmarks a *classifier on whole-motorcycle crops*, not a violation
(see [`cnn-vs-vit-results.md`](cnn-vs-vit-results.md), [ADR-005](adr/ADR-005.md)).

A sixth word is deliberately **not** in the table because it is not a validation
result at all: **demonstrated**. A controlled demonstration
([`controlled-demo.md`](controlled-demo.md)) runs the real reasoners over a
*hand-authored* scenario and *declared* scene context, on synthetic pixels. It
establishes that the system reasons correctly over declared context and **nothing
whatever** about detection accuracy or real-world performance. Every row in the
matrix below is about real footage; §7b states the boundary explicitly. Never quote a
controlled-demo result in this table's vocabulary, and never quote a matrix number as
though a controlled demo produced it.

---

## 2. The matrix

| Violation | Structural tests | Real footage | Calibration | Ground truth | Observed result | Reproducibility | Demo status | Principal limitation |
|---|---|---|---|---|---|---|---|---|
| **Wrong way** | ✅ full slice + engine + app + overlay | ✅ `wrongway_001.ogv` (real RT-DETR, real tracker) | ⚠️ auto-derived, whole-frame lane | ❌ none (`established: false`, sign only) | **16 confirmed events** ¹ | ⚠️ deterministic **given the sampling**; the auto-derived scene moves with it (§3.1) | **Demo ready**, with the wording in §3.1 | The lane is the whole frame, so the opposing carriageway is eligible; and the footage is *lawful* contraflow |
| **Triple riding** | ✅ rule + observation + pipeline + engine + app | ✅ 3 clips (Raxaul, Chiang Mai, Gangtok); Chiang Mai re-run over its **whole** clip | n/a — geometry-free | ❌ none (`triple_riding: null`) | **0 confirmed events** on all three; the real path runs end to end ² | ✅ deterministic per clip/sampling | **Conditionally demo ready** — demo the *abstention*, not a positive | Per-frame rider count has no uncertainty channel (§4) |
| **Illegal stopping** | ✅ rule + two-stream join + pipeline + engine + app | ⚠️ decoded, never confirmed | ❌ **no no-stopping zone exists for any real clip, and none can honestly be drawn on this corpus** (§7a) | ❌ none | 0 by construction (rule not buildable) | ✅ structural zero, T0-checked | **Conditionally demo ready** — needs footage the corpus does not contain | Every stationary vehicle in the corpus is stopped *lawfully*; gridlock is deliberately not a stop |
| **Red-light jumping** | ✅ rule + latch + crossing + signal + pipeline + engine + app | ✅ `clean_001.ogv` (real RT-DETR, real tracker, **transcribed** signal) | ✅ analyst-authored stop line + junction + signal ROI (§3.3) | ❌ none | **0 confirmed events**, and **0 stop-line crossings** — the governed approach is red and stopped ³ | ✅ deterministic (whole clip, no stride) | **Conditionally demo ready** — demo the *correct zero*, never a positive | The latch is **not exercised**: nothing crossed, so no signal was ever read at a crossing (§5) |
| **No helmet** | ✅ rule + attribution gate + exemption + pipeline + engine + app | ✅ classification observed (P4-U8/U9/U10) | n/a — geometry-free | ⚠️ classifier-level only, on a different crop population | see [`helmet-runtime-evaluation.md`](helmet-runtime-evaluation.md) | ✅ | **Experimental — do not claim** | Turban capability blocker + driver-only attribution (§6) |
| **Speeding** | ❌ no reasoner exists | ❌ | ❌ | ❌ | none | n/a | **Not demo ready** | Feasibility-gated; nothing is implemented |

¹ ² ³ See §3 for the exact commands, outputs and caveats.

---

## 3. What each real-footage run actually did

### 3.1 Wrong way — the one real positive

```bash
./.venv/Scripts/python.exe demo/wrongway_calibrated_validation.py --stride 5
```

Two passes over `test-videos/wrong-way/wrongway_001.ogv`, mirroring what
`ProcessingService` does for an uncalibrated upload:

1. **Measure.** Real RT-DETR + the IoU tracker over every 5th frame, then
   `estimate_dominant_flow` over the resulting tracks. Observed: **345 frames
   analysed, 1708 detections, dominant flow dx=1.0 dy=0.06 (heading 3.2°), 24
   movers.**
2. **Reason.** A scene declaring that direction on a whole-frame lane
   (`scene_config_hash = 781cface…`), then the real wrong-way slice against it.

**Observed: 16 confirmed wrong-way events.**

Three caveats travel with that number, and none of them is optional:

- **It is lawful contraflow.** The clip is *signed* contraflow through a
  roundabout. The reasoner detects **sustained opposition to the scene's declared
  legal direction**, which is exactly what it claims to do. It is *not* a finding
  that anyone drove illegally, and the demo must not be narrated as one.
- **The lane is the whole frame.** A measured flow vector says nothing about where
  the carriageway edges are, so the authored lane admits every track in frame. This
  is a **weaker** scene than the analyst-drawn one the T0 manifest describes, and the
  result must be read as the auto-calibrated approximation of it.
- **The scene hash — and the count — are functions of the sampling.** The legal
  direction is *measured from the footage*, so `--stride` changes the measured flow,
  which changes the scene, which changes the hash and can change the count. This is
  not hypothetical: an earlier session of this same script recorded scene
  `91d9745f…` and **13** events, and both runs' events are still on disk under
  `runs/wrongway-calibrated/`, cleanly separated by their scene hash (13 under
  `91d9745f…`, 16 under `781cface…`) because the event store is write-once and the
  hash is identity-bearing. Nothing was overwritten and nothing is ambiguous — but a
  count quoted without its command is meaningless.

  The variability is **not** in the detector or the reasoner, and this was measured
  rather than assumed. Three consecutive detector passes over the same 12 frames were
  byte-identical in this environment (torch 2.13.0+cpu, 10 threads), and pass 1 run
  twice at the same stride reproduces to the last digit:

  | `--stride` | frames | detections | dx | dy | movers | scene hash |
  |---|---|---|---|---|---|---|
  | 5 | 345 | 1708 | 0.9984 | 0.0560 | 24 | `781cface…` |
  | 5 *(repeat)* | 345 | 1708 | 0.9984 | 0.0560 | 24 | `781cface…` |
  | 3 | 575 | 2844 | 0.9986 | 0.0525 | 27 | `0441a73a…` |
  | 4 | 431 | 2133 | 0.9977 | 0.0676 | 29 | `0a4db317…` |
  | 6 | 288 | 1407 | 0.9965 | 0.0835 | 21 | `9820da7f…` |
  | 10 | 173 | 841 | 0.9300 | 0.3675 | 18 | `9773770d…` |

  (No sampling tried here reproduces the earlier session's `91d9745f…`; its arguments
  were not recorded, which is itself the lesson.)

  So the pipeline is deterministic — the same command reproduces to the last digit —
  and it is the *auto-calibration* that moves. Two things follow. Every sampling gives
  a different hash, because the direction is a measured float and the hash is over its
  exact value; and the direction itself is genuinely sensitive to sample density —
  halving it (stride 5 → 10) swung the derived heading by roughly 19°, which is a real
  change in what "wrong way" means for that run, not a rounding artefact.

  This is the honest cost of measuring a legal direction from the footage instead of
  drawing one, and it is the strongest argument in this document for an
  analyst-authored lane, which would remove the sensitivity entirely.

The corresponding **uncalibrated** run of this same clip is structurally zero
(`test-videos/run_manifest.py`). That zero and this sixteen answer two different
questions and must never be quoted as if they answered one.

### 3.2 Triple riding — the real path runs, and correctly confirms nothing

```bash
./.venv/Scripts/python.exe demo/triple_riding_validation.py --frames 60            # Raxaul
./.venv/Scripts/python.exe demo/triple_riding_validation.py --frames 60 \
    --clip test-videos/normal-traffic/clean_001.ogv --skip 30                      # Chiang Mai
./.venv/Scripts/python.exe demo/triple_riding_validation.py --frames 60 \
    --clip test-videos/edge-cases/congestion/congestion_001.webm --skip 30         # Gangtok
```

Every stage is production: decode → RT-DETR → IoU tracker → P4-U4 rider association
→ rider-count observations → `TripleRidingReasoner` → persisted `ConfirmedEvent`.
Nothing is stubbed or injected.

| Clip | Frames | Rider-count obs | Motorcycles | Rider-count distribution | Best track | Events |
|---|---|---|---|---|---|---|
| **Raxaul** level crossing (`congestion_002.webm`) | 60 | 79 | 6 | 0:9, 1:17, 2:12, **3:40**, 4:1 | `iou-7`: 53 obs, **75.5 % ≥3**, longest ≥3 run **0.5 s** (needs 1.0 s) | **0** |
| **Chiang Mai** intersection (`clean_001.ogv`) | 60 | 352 | 12 | 0:18, 1:316, 2:18 | no track ever reached 3 | **0** |
| **Chiang Mai**, *whole clip* (`--frames 434 --skip 0`) | 434 | 1112 | 29 | 0:143, 1:940, 2:29 | no track ever reached 3 | **0** |
| **Gangtok** congestion (`congestion_001.webm`) | 60 | 115 | 6 | 0:25, 1:90 | no track ever reached 2 | **0** |

Two of these zeros are simply correct: neither Chiang Mai nor Gangtok produced a
single ≥3 observation, so there was nothing to confirm and nothing was invented.
Raxaul's zero is the interesting one, and §4 is about it.

Chiang Mai was re-run over its **whole** clip rather than the 60-frame window the
first pass used, because a 60-frame sample of a 434-frame clip is thin evidence for
"nothing was there". The fuller sample more than doubles the motorcycles seen (12 →
**29**) and triples the observations (352 → **1112**), and the conclusion holds
harder than before: the **maximum rider count observed anywhere in the clip is 2**,
so not one of the 29 motorcycles ever presented the evidence the rule would need.
That zero is not the rule declining to confirm — it is there being nothing to
confirm.

`clean_002.webm` (Montreal) was checked and **cannot** contribute: a full detector
pass over it yields **zero motorcycle detections** in 3 780 track states, so it is
not triple-riding footage in any sense. `clean_003.webm` (7 s, top-down roundabout)
and `lowres_001.ogv` (176×144) are likewise unusable. Raxaul remains the only clip in
the corpus holding genuine ≥3-rider evidence, which is why §4 is about it and why the
conclusion below does not change.

`clean_004.webm` — the manifest's "richest motorcycle footage available" — is **not
present locally** (only its `.meta.yaml` is; the media is gitignored and was never
fetched), so it could not be included. It remains the best candidate for a first
labelling pass.

### 3.3 Red-light jumping — a transcribed signal, and a correct zero

```bash
./.venv/Scripts/python.exe demo/red_light_validation.py
```

The artifact §7a said was missing now exists, for
`test-videos/normal-traffic/clean_001.ogv` — a signalised intersection at Pak-Krok
Siwilai, Chiang Mai, filmed from a vehicle waiting on one approach.

**The signal state is transcribed, not invented.** The clip shows the head governing
the camera's own approach *and* a digital countdown beside it. Sampled once per
second, the countdown reads **58, 57, 56 … 44** — one decrement per second of media
time — and the head's **top (red) lamp is lit in every one of the 15 sampled
frames**. The approach is therefore RED for the clip's whole 14.5 s with 44 s still
to run, and the footage carries its own clock to corroborate the reading. The scene
declares `SignalSourceMode.MANUAL_ANNOTATION`, which is exactly what that mode is
for.

| | |
|---|---|
| Scene hash | `e2aa47c1…` (analyst-authored) |
| Stop line | `(900, 940) → (1900, 860)`, crossing direction `(0, −1)` |
| Junction zone | `(250, 560) … (1900, 900)` |
| Signal schedule | one phase: `t=0.0 s → RED` |
| Frames processed | **434** (the whole clip, no stride) |
| Vehicle tracks reasoned over | **107** |
| Validated junction entries | **0** |
| Steps with RED latched | **0** |
| Confirmed events | **0** |

**What this establishes.** The full real path — decode → RT-DETR → IoU tracker →
crossing derivation → declared-signal join → `RedLightReasoner` — runs on real
footage against a real, transcribed signal, and confirms nothing while a scene-wide
RED is in force and 107 vehicle tracks move through the junction polygon.

**What it does not.** It does **not exercise the latch.** No vehicle crossed the
governed stop line, so no signal was ever read at a crossing; every latched entry
state is `UNKNOWN`. A zero produced from zero input says nothing on its own about
whether the reasoner would confirm a genuine violation — the structural tests are
what cover that, and they remain the only thing that does.

**Why the zero is nonetheless a result.** It is a false-positive check, not an empty
scene. The traffic on the governed approach is stopped, which is what a red light is
supposed to produce; every vehicle that *moves* in this clip is cross traffic on the
conflicting phase, and the geometry deliberately does not govern it. Drawing the same
stop line across the cross-traffic path instead — the calibration mistake this scene
exists to avoid — was measured during this work and yields **4 forward crossings and
74 validated-inside observations**, every one of which the same declared RED would
have confirmed, and every one of which would have been false. The zero is a property
of correct calibration.

Two limitations travel with it. The camera is hand-held: global motion over the clip
spans **56 px horizontally and 48 px vertically**, so the authored geometry drifts
slightly against the world. That can only *create* spurious crossings, never suppress
them, so the zero is robust to it — but a positive result from this scene would not
have been. And a single-phase schedule cannot show the latch surviving a change to
green, because the light never changes within the footage available.

**Why not `clean_002.webm`,** which an earlier audit named as the obvious candidate:
it is a signalised intersection, but the camera **pans 440 px horizontally** (23 % of
frame width) between t≈2 s and t≈6 s, and `index.csv` describes it as
`fixed_elevated`, which is simply wrong. A `SceneConfig` polygon is static in pixel
space, so a stop line drawn at t=0 would sit 440 px from the real one seconds later
and would measure the camera's motion rather than any vehicle's. No defensible
geometry can be authored on it.

### 3.4 The T0 structural corpus

```bash
./.venv/Scripts/python.exe test-videos/run_manifest.py
# result: 18 passed, 0 failed, 13 not evaluated
```

The 18 are the expectations guaranteed *by construction*: with no calibrated scene,
the capability probe cannot build a wrong-way, illegal-stopping or red-light rule at
all, so the count is zero before a frame is decoded. The 13 not evaluated are the
`established: false` entries — deliberately never checked, because "checking" a
placeholder would manufacture a benchmark out of a guess.

---

## 4. The triple-riding limitation, stated exactly

Raxaul's best motorcycle track (`iou-7`) carried **three riders in 40 of its 53
observations (75.5 %)**. Its per-observation count sequence was:

```
33333333313333333341333333332323322223323323332233323
```

Read as support runs against the threshold of 3, that decomposes into:

- **≥3 runs (observations):** `9, 9, 8, 1, 2, 2, 2, 3, 3, 1`
- **sub-threshold stretches that break them:** `1, 1, 1, 1, 4, 1, 1, 2, 1` — **seven
  of nine are a single observation.**

So the two longest stretches of three-rider evidence — nine observations each — are
separated by **exactly one** observation that counted 1 rider.

The track's observations span **1.97 s** of media time, comfortably longer than the
1.0 s the rule needs. Measured against those real timestamps:

| Longest ≥3 support run on `iou-7` | |
|---|---|
| as the reasoner computes it | **0.50 s** — below the 1.0 s threshold, so no event |
| if a *single* sub-threshold observation between two supporting ones were treated as an abstention rather than a contradiction | **1.30 s** — above it |

The whole of the shortfall is therefore momentary count dips, not the riders being
absent. (That second figure is a **diagnostic**, produced by a throwaway script that
changed no rule and confirmed no event. It is stated to characterise the cause, and it
is emphatically *not* a proposed behaviour — see below.)

**The reasoner is not wrong to end the run there.** It is given
`RiderCountObservation.rider_count`, an integer that *asserts* how many riders are on
the motorcycle. A count of 1 asserts "one rider" — and if that is true, the run
genuinely ended. Bridging it anyway would mean the rule deciding, without evidence,
that the observation it was handed was false.

**The limitation is in the observation contract, not in the rule.** A per-frame count
drop has two indistinguishable causes:

1. a rider genuinely got off, or the count really is below threshold — the run
   *should* end;
2. the association momentarily failed under occlusion or a merged box — the count
   *underreports*, and the step should be withheld and bridged.

`RiderCountObservation` carries `rider_count: NonNegativeInt` and nothing else. It
has **no channel for "this frame's count is not trustworthy"**, so nothing downstream
can tell those two cases apart. Compare its siblings, which do:

- `HelmetStateObservation` has `uncertain` in its label set, and the no-helmet
  reasoner withholds those steps from the temporal base entirely, turning them into
  *bridged gaps* rather than contradictions — the exact mechanism triple riding
  needs and does not have;
- `SpeedObservation` carries `speed_sigma`, explicit uncertainty on the value.

**What must not be done about it.** Lowering `min_persistence` below 1.0 s, raising
`max_observation_gap` until dips are swallowed, or dropping the threshold to 2 would
all produce a positive demo, and all three would be tuning a threshold against an
observation until it yielded the answer the presenter wanted. The rule's temporal
guarantee — that a single flickering frame cannot mint a violation — is the whole
point of the rule, and it is worth more than a green box.

**What would legitimately fix it** is an uncertainty channel on the rider-count
observation (a confidence, or an `uncertain` sentinel) populated by the association
layer when it cannot resolve the riders on a motorcycle for a frame, plus the
withhold-and-bridge treatment no-helmet already implements. `RiderCountObservation`
is a **frozen contract**, so that is a deliberate, reviewed contract change and not
something to slip in during a validation pass.

Note the difference between that and simply bridging every dip. Bridging
unconditionally would have the *rule* overrule an observation it was handed — deciding,
with no evidence, that a count of 1 was wrong. A rider who genuinely dismounts must
still end the run, and under unconditional bridging they would not. Only the
observation layer knows which kind of dip it just produced, which is exactly why the
channel belongs there and why the 1.30 s figure above is a diagnostic and not a
patch.

---

## 4a. The ID-switch guard is correct and currently inert

Every reasoner refuses to accumulate support across a tainted step
(architecture-review §13: "tainted tracks may abstain but never confirm"), and the
mechanism is exercised throughout the suite. But the **shipped tracker never sets
the flag**: `tracking/iou_tracker.py` stamps `tainted=False` unconditionally, and says
why — a greedy matched-box associator "exposes no trustworthy ID-switch signal, so
fabricating taint would be dishonest."

So the honest position is two sentences, not one:

- The **contract and every rule** handle ID switches correctly, and a tracker that
  reports one gets the abstention it should. `StubTracker` proves this in tests.
- In a **real run today, ID switches go undetected** — the guard never fires. Every
  real-footage run in §3 reports `taint restarts: 0`, and that zero means "nothing was
  reported", not "nothing happened".

The consequence is asymmetric and worth naming: an ID switch that splits one vehicle
into two tracks *loses* support and under-reports; one that merges two vehicles into
one track could carry support across a vehicle boundary. Nothing in the current stack
detects either. Closing it means a tracker with motion state that can flag a
kinematic discontinuity — the audited-ByteTrack path recorded as future work in
`tracking/iou_tracker.py` — not a heuristic invented at this layer.

---

## 5. The red-light limitations, stated exactly

Red-light jumping **has now been run on real footage** against a signal transcribed
from the clip itself (§3.3), and the result is a correct zero. That closes the
artifact gap this section previously described, and it opens a narrower one that must
be stated just as plainly:

- **The latch has never been exercised on real footage.** The run confirms nothing
  because nothing crossed the governed stop line, not because a crossing was
  evaluated and rejected. Every latched entry state in that run is `UNKNOWN`. The
  reasoner's central mechanism — read the signal at the crossing, hold it through the
  junction — is covered by structural tests and by nothing else.
- **A phase change has never been observed.** The transcribed schedule has one phase,
  because the light does not change within the 14.5 s available. The property that
  motivates the latch in the first place (a light turning green after entry must not
  un-commit the act) is therefore untested outside the suite.

Closing either needs footage in which a vehicle actually crosses a stop line on red,
from a camera fixed enough to author geometry against — which the corpus does not
contain, and which must not be simulated by drawing a line somewhere convenient.

Two further properties a viva examiner will ask about, both deliberate:

- **The signal is declared, not perceived.** Nothing in TrafficPulse classifies a
  signal head from pixels; `SignalSourceMode.ROI_CLASSIFIER` is contract-defined and
  unimplemented. Every confirmed event records the latched state as an ordinal
  measurement so a reviewer audits what the system was *told*. `UNKNOWN`, `AMBER` and
  `OFF` never confirm.
- **A stop-line overrun is not distinguished from a passage.** The signal is read at
  the forward stop-line crossing and latched, which is what stops a light turning
  green after entry from un-committing the act. The cost is that a vehicle which
  crosses the line on red, stops short of the junction, and then enters lawfully on
  green still enters with `RED` latched. Separating the two needs evidence the join
  does not receive — whether the vehicle came to rest between the line and the
  polygon, which is the *stationary* stream's fact. It is recorded in
  `rules/red_light.py` rather than approximated with an uncalibrated staleness
  threshold.

A defect found during this audit and fixed: the latched crossing was not consumed by
the entry it enabled, so a later `False → True` junction transition with no crossing
of its own re-latched the stale state — minting a second event for one act, under a
signal that may since have gone green. Regression coverage:
`tests/rules/test_red_light.py::test_a_crossing_is_spent_by_the_entry_it_enabled`
and `::test_the_re_entry_after_a_spent_crossing_reports_unknown_not_a_stale_red`.

---

## 6. Why no-helmet stays experimental

Two independent blockers, either of which is sufficient:

- **Turban capability.** The exemption depends on a `turban` label. The trained
  P4-U5 ResNet-50 is binary and *cannot emit it*; the shipped zero-shot backend emits
  it unreliably (a near-tie softmax on real New Delhi footage misread bare heads as
  `turban`, which is why the exemption is now predominance-based rather than
  single-frame). The classifier capability guard therefore refuses to build the rule
  on a turban-blind backend, and `serve_demo.py` does not bypass it.
- **Driver-only attribution.** Helmet state is a fact about *a rider*; it becomes a
  violation only once the system can say *whose duty* it was. Only the `DRIVER` slot
  carries that claim, and that slot is assigned only when exactly one rider is
  associated with the motorcycle. Multi-rider motorcycles — 42.4 % of the frozen
  corpus, 81 % of a real congestion clip — are therefore **unattributable by design**.
  That is the honest outcome, not a gap to close by guessing.

Demonstrate helmet work through `serve_demo.py`, which classifies every rider, draws
the annotated video, reports the summary, and mints **no** `ConfirmedEvent` because an
analysis has no reasoner. See [`demo-guide.md`](demo-guide.md) for what may and may
not be claimed from it.

---

## 7. Live camera

Validated **backend-side** end to end against real RT-DETR and helmet backends over a
real WebSocket: ~1 fps of inference and ~1.6 s end-to-end delay on this machine's CPU,
persistent tracker state within a session, back-pressure holding exactly one frame
server-side, and an engine reset every 600 processed frames. Live events are **not
persisted** and do not enter the repository, the analytics or the review workflow.

Not validated: the **physical browser camera** (this machine exposes none, so
`getUserMedia` → canvas → socket has never run against real hardware), and live
wrong-way / illegal-stopping / red-light behaviour, which additionally need a
calibrated scene the live path is not given. See [`live-camera.md`](live-camera.md).

---

## 7a. The exact artifacts missing for the uncalibrated rules

**Red-light jumping's artifact has been authored** — see §3.3. What follows is what
remains missing for illegal stopping, and why the red-light one is only half a
resolution.

Neither rule is blocked by code. Both are blocked by an authored artifact, and both
artifacts are producible in the shipped calibration UI (H12/H13,
`components/workspace/scene-calibrator.tsx` and `signal-schedule-editor.tsx`) without
inventing anything.

The shipped `configs/scenes/example-scene.yaml` does **not** substitute for either. It
declares every zone type, but its polygons describe the synthetic scenario's road.
Binding it to real footage would be reasoning about another camera's geometry — the
precise error `auto_calibrate_uploads` exists to prevent, which is why auto-calibration
derives only *observable* facts (a dominant flow) and never invents a zone, a stop line
or a schedule.

**Illegal stopping needs** a per-video `SceneConfig` for a real clip with an enabled
`no_stopping` zone drawn over a region of *that* clip's road where a vehicle actually
comes to rest, plus a `stationary_duration`. The hard part is not the drawing — it is
finding footage where a stop is genuinely unlawful *and* saying so is defensible.

**No clip in this corpus can supply it, and the reason is not a shortage of stationary
vehicles — it is that every one of them is stopped lawfully.** Each candidate was
inspected during this pass:

| Clip | Stationary vehicle? | Why it cannot be used |
|---|---|---|
| `congestion_002.webm` (Raxaul) | yes, all of them | Gridlock at a level crossing. It is in the corpus *specifically* as the false-positive trap |
| `congestion_001.webm` (Gangtok) | yes | Congestion, same objection |
| `clean_001.ogv` (Chiang Mai) | yes | Stopped **at a red light** (§3.3). Lawful by definition |
| `clean_002.webm` (Montreal) | yes (a bus at a stop) | Lawful; and the camera pans 440 px, so no zone can be authored against it at all |
| `clean_003.webm` (roundabout) | **no** — every vehicle moves between t=0 and t=6 | The only fixed camera in the corpus (7 px of motion), but 7.14 s long and nothing ever stops |
| `wrongway_001.ogv`, `lowres_001.ogv` | — | Hand-held (576 px pan); 176×144 |

So the missing artifact is **footage**, not a drawing: a clip showing a vehicle at rest
where stopping is *restricted and the restriction is visible in frame*, from a camera
fixed enough to author a static polygon against, and long enough for the dwell to
exceed a threshold declared **before** looking at the clip. Choosing a
`stationary_duration` that a 7-second clip happens to satisfy would be tuning a
threshold to manufacture a positive, which is the one thing this rule must never do.

What must **not** be done in either case: authoring a zone or a schedule to make a
particular clip produce a particular answer. A schedule invented to make a vehicle look
like a violator is manufactured ground truth, and it would poison every number derived
from it.

---

## 7b. Controlled demonstration — a separate category, not a fix for §7a

§7a says the illegal-stopping artifact is missing because **no clip in this corpus can
honestly supply it**, and that remains true. Nothing in this section changes it, and
nothing here is evidence about real footage.

What exists now is a *declared* scenario -- one 6-second, 480x270 synthetic clip whose
four situations were hand-authored, whose scene an operator draws, and whose expected
contents are written down before it is run
([`controlled-demo.md`](controlled-demo.md), `trafficpulse.scenes.demo_scenario`,
`demo/controlled_demo.py`). Processing it confirms four violation types from one pass:
wrong way, illegal stopping, red-light jumping and triple riding.

**What that is worth.** It is the only place in the repository where the red-light
latch is actually exercised (a vehicle crosses on red, and the schedule turns green
afterwards -- §5's "the latch is not exercised" is about *real footage* and still
holds), and the only place illegal stopping confirms at all. It demonstrates that one
video, one analysis and four independent reasoners produce four independently
justified events, each with its own evidence manifest.

**Two renderings, and what each is worth.** The scenario has one definition and two
renderings. The **rectangle** clip pairs with a *scripted* detector -- RT-DETR detects
nothing in it, measured (0 detections on every sampled frame) -- and is what the test
suite replays. The **composited** clip pastes real vehicle crops cut from this very
corpus (a car from `clean_003.webm` frame 0; a three-rider motorcycle from
`congestion_002.webm` frame 80) along the same trajectories on a plain road canvas,
so real RT-DETR has real pixels to find. Driven over HTTP against `serve:app` on
2026-09-03 it produced **647 detections over 60 frames and all four families
confirmed** (expected 4, matched 4, missing 0, unexpected 0).

**What neither is worth.** Even the composited run is real inference on *unchanging
vehicle cut-outs sliding across an empty synthetic road* -- no occlusion, no
perspective change, no lighting variation, no clutter, no other traffic. Detecting a
pasted car there says nothing about detecting a car in Raxaul at dusk. It is **not** a
real-footage row, it does **not** move any cell in §2, and it must never be narrated
as "we detect illegal stopping".

**What must not be done.** The controlled scenario's thresholds are declared in
`demo_scenario.py` and were chosen so the clip could stay short -- not tuned until a
result appeared. Adjusting them, or the actors, to make a family confirm is the same
manufactured-ground-truth error §7a's last paragraph forbids.

---

## 8. Recommended demonstration set

Four examples, chosen so that each shows something the others cannot, and so that
**no** claim in the narration outruns §2. Prefer these four over a longer tour of
weaker material.

| # | What to show | Asset | How | Time | The claim, exactly |
|---|---|---|---|---|---|
| 1 | **Helmet perception, and the system refusing to enforce on it** | `runs/demo-ready/clips/raxaul-congestion.mp4` | `serve_demo.py` in the browser — see [`demo-guide.md`](demo-guide.md) | ~55 s live | "It classifies every rider and declines to name a violator it cannot attribute." Never: "it detects helmet violations." |
| 2 | **Wrong-way on real footage** | `test-videos/wrong-way/wrongway_001.ogv` + the persisted run under `runs/wrongway-calibrated/` | `demo/wrongway_calibrated_validation.py` — run **beforehand**, present the transcript and the persisted events | ~30 min, so pre-run it | "Real RT-DETR, real tracker, real reasoner, real events — against a direction derived from the footage itself, on lawful contraflow." |
| 3 | **Wrong-way live in the UI** | `runs/demo/clips/wrong_way_upload_validation.mp4` | upload it in the app under `serve.py` | ~1 min | "A **constructed validation clip**: real footage with a real vehicle crop composited against the flow. The analysis is genuine; the scenario is authored, and that is stated." |
| 4 | **The abstention that matters most** | `test-videos/edge-cases/congestion/congestion_002.webm` (Raxaul) | `demo/triple_riding_validation.py --frames 60` | ~3 min | "Three riders in 75 % of one motorcycle's frames, and the system still confirms nothing, because the evidence never held for the required second. That is the rule working." |

A fifth is available if there is time, and it is the only real-footage red-light
evidence that exists:

| # | What to show | Asset | How | Time | The claim, exactly |
|---|---|---|---|---|---|
| 5 | **Red-light reasoning on a real, transcribed signal** | `test-videos/normal-traffic/clean_001.ogv` | `demo/red_light_validation.py` — pre-run, present the transcript | ~12 min, so pre-run it | "The light is red — here is the head and the countdown in the frame — the approach is stopped, and the system confirms nothing. Drawn wrongly across the cross traffic the same scene would have confirmed four." Never: "it detects red-light jumping." |

A sixth is available, and it is the only demonstration in which several violation
families confirm at once. It belongs to a **different category** from every row above
and must be introduced as such:

| # | What to show | Asset | How | Time | The claim, exactly |
|---|---|---|---|---|---|
| 6 | **One video, four independently reasoned violation types** | `runs/controlled-demo/controlled-demo-pixels.mp4` (generated, not committed) | `demo/controlled_demo.py --real-pixels`, then calibrate and run it in the browser -- see [`controlled-demo.md`](controlled-demo.md) | ~2 min | "This is a **controlled demonstration on synthetic footage**, not validation. I declare the context the camera cannot know -- the legal direction, the no-stopping zone, the stop line, the signal timing -- and four separate reasoners each reach their own conclusion in one pass. The declared expectations never reach the engine; a family it declines to confirm is reported as missing, not conjured." Never: "this shows the system detects four violations." |

Two clips deliberately **not** in the demonstration set for triple riding:
`congestion_001.webm` (Gangtok) and `clean_001.ogv` (Chiang Mai) produced no ≥3 rider
observations at all, so their zeros demonstrate nothing beyond "nothing was there".
And **nothing demonstrates illegal stopping on real footage**, because §7a says no
such example exists in this corpus — show it on the controlled/calibrated path (§7b,
demonstration 6) or not at all.

**Never present a filename as a result.** `wrongway_001.ogv` lives under
`test-videos/wrong-way/` because of what it was *collected* for, not because anything
has confirmed what is in it.

---

## 9. Reproducing every number in this document

```bash
# structural corpus (fast, no inference)
./.venv/Scripts/python.exe test-videos/run_manifest.py

# wrong way on real footage (slow: two full RT-DETR passes on CPU)
./.venv/Scripts/python.exe demo/wrongway_calibrated_validation.py --stride 5

# triple riding on real footage (~3 min per clip on CPU)
./.venv/Scripts/python.exe demo/triple_riding_validation.py --frames 60
./.venv/Scripts/python.exe demo/triple_riding_validation.py \
    --clip test-videos/normal-traffic/clean_001.ogv --frames 434 --skip 0

# red light on real footage against a transcribed signal (~12 min on CPU)
./.venv/Scripts/python.exe demo/red_light_validation.py

# quality gates
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m mypy src
./.venv/Scripts/python.exe -m pytest -q
```

The media under `test-videos/` is gitignored; `test-videos/fetch.py` retrieves it from
the sources recorded in `sources.yaml`. `clean_004.webm` and `night_001.webm` are
currently absent from this checkout.
