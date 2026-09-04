# TrafficPulse — viva fact sheet

**A technical reference, not a presentation and not a script.** Every claim here is
one the repository can defend, and each is tagged with how strongly it is
established. Nothing is rounded up, and no accuracy figure appears that was not
already measured and recorded elsewhere in this repository.

## The five words (used consistently below)

| Tag | Means |
|---|---|
| **tested** | Unit/integration tests exercise the logic against inputs whose correct answer is known by construction. |
| **observed** | The real path (real decode → RT-DETR → tracker → reasoner) ran on real footage and produced a stated result. Nothing was labelled, so this is not correctness. |
| **reproduced** | Re-running the same command on the same input yields the same result. |
| **benchmarked** | Compared against ground truth under a pre-registered protocol. **Only P4-U5 qualifies.** |
| **unvalidated** | Never run against real footage. Not "broken" — only "no evidence". |

Canonical source of truth: [`validation-matrix.md`](validation-matrix.md).

---

## 1. Architecture

The central commitment is a hard separation between **perception** (what a model sees
in one frame) and **violation reasoning** (typed observations accumulated over a
track, combined with scene geometry and explicit rules).

```
video → ingestion (PTS-accurate) → detector → tracker
      → typed observations (heading / in-zone / stationary / crossing / signal / rider-count / helmet-state)
      → rule reasoners (temporal) → ConfirmedEvent → evidence manifest → persistence → analyst review
```

- Frozen, versioned **data contracts** (Pydantic) sit between every stage; JSON
  schemas are exported under `schemas/`. **tested**
- Reasoning is **pure and deterministic**: no wall-clock, no randomness, no I/O.
  Media time is PTS-anchored at a fixed UTC epoch. **tested**
- Events are **content-addressed** and the event store is **write-once**, so
  re-processing re-confirms byte-identical events rather than duplicating them. **tested**
- Five reasoning slices ship: wrong-way, illegal stopping, red-light jumping, triple
  riding, no-helmet. Speeding has **no reasoner at all**.
- 2966 backend tests, 599 frontend tests, `ruff` + strict `mypy` clean. **reproduced**

## 2. Why RT-DETR

- Real-time DETR (`PekingU/rtdetr_r50vd`), COCO-80, run at score threshold **0.50**.
- **The decisive reason was licensing, not accuracy.** [ADR-001](adr/ADR-001.md) chose a
  *permissive-only* stack: RT-DETR / D-FINE are **Apache-2.0**, so the whole detection
  path can be redistributed and defended. A YOLO-family detector under a copyleft
  licence was rejected on that ground, not on benchmark numbers. Say this if asked "why
  not YOLO" — the honest answer is provenance, not performance.
- Secondary properties recorded in the same ADR: it is **NMS-free** (a hybrid DETR doing
  set prediction), so there is no per-scene NMS hyper-parameter to tune, and it is
  available through HuggingFace Transformers.
- Loaded **offline** (`local_files_only=True`); no weights are vendored in the repo and
  the launcher never reaches the network.
- The checkpoint emits the VOC-style label `motorbike`; the label map is asserted by a
  test, because a wrong key there silently disables a whole violation class.
- **Recall is not complete.** On the frozen evaluation split, about **20 % of eligible
  riders had no overlapping detection**, and small/distant riders are the worst
  stratum. This is published in the deployment's own capability strip. **observed**
- The detector is behind an interface seam; it is injected, never constructed inside a
  rule. Three consecutive passes over the same 12 frames were **byte-identical** in
  this environment (torch 2.13.0+cpu). **reproduced**

## 3. Tracking and rider association

- Shipped tracker is a **permissive greedy IoU associator written in-repo**. ByteTrack
  was considered and rejected for this milestone because its provenance could not be
  verified offline; it is recorded as future audited work. **tested**
- **Rider ↔ motorcycle association** links person detections to motorcycle detections
  geometrically, producing `MotorcycleTrack` observations with a rider count and, when
  exactly one rider is associated, a `DRIVER` slot.
- Head-crop geometry for the helmet classifier is *derived* from the rider box, not
  detected separately.

## 4. Temporal reasoning

- All five reasoners delegate lifecycle mechanics to one shared
  `TemporalRunReasoner`: run tracking, gap breaking, taint reset, event minting.
- A violation is confirmed only when a support condition holds for at least
  `min_persistence` seconds **and** across **at least two observations** — a single
  flickering frame can never mint a violation. **tested**
- Every confirmed event records its **measurements** and the **thresholds** they were
  compared against, so a reviewer can audit the decision rather than trust it.
- Thresholds are marked `provisional` / `unset` in scene configuration and are
  **operator-chosen, not learned**. Defaults: heading deviation 120°, wrong-way
  persistence 1.0 s, stationary duration 5.0 s, triple-riding persistence 1.0 s.

## 5. Calibration

- A `SceneConfig` is **per-camera geometry** in the video's own pixel space: lanes,
  zones, stop lines, signal groups, legal directions.
- Scenes are **content-addressed**; a revision's hash is identity-bearing and is
  stamped onto every event it produced.
- Uploads with no scene get **auto-calibration**, which derives only *observable*
  facts — a dominant flow direction — and **never invents a zone, a stop line or a
  schedule**.
- A geometry-dependent rule that the resolved scene cannot support is **not run**,
  rather than run on a guess. **tested**
- The shipped `configs/scenes/example-scene.yaml` describes the *synthetic* scenario's
  road and is never bound to real footage.

### 5a. Why calibration is required at all — the one-sentence version

*A camera records pixels; it does not record which way traffic legally travels, where
stopping is prohibited, or what the signal was showing.* Those are facts about the
road and its rules. TrafficPulse reasons over them; it does not infer them and does
not verify them — and the calibration surface says exactly that above the drawing
tools. The one fact that **is** observable is the dominant traffic flow, which is why
that is the only thing auto-calibration derives. **tested**

### 5b. Operator thresholds

Four site thresholds are analyst-settable (dwell, heading deviation, wrong-way
persistence, red-light debounce). Leaving a field blank sends **nothing**, so the
scene never records a value nobody chose; everything set is stamped
`ParameterStatus.PROVISIONAL` — operator-chosen, not tuned against ground truth. A
saved calibration can be **reloaded** onto the drawing surface from the revision the
video is bound to, so a scene is auditable rather than write-only. **tested**

## 6. Wrong-way semantics

- Confirms **sustained opposition to the scene's declared legal direction**, measured
  as heading deviation beyond `heading_deviation_max` for `min_persistence`.
- **observed:** 16 confirmed events on `wrongway_001.ogv` at `--stride 5` (345 frames,
  1708 detections, measured flow dx≈0.998 dy≈0.056, 24 movers, scene `781cface…`).
- Three caveats that must travel with that number:
  1. The footage is **signed, lawful contraflow**. The rule is not claiming anyone
     broke a law.
  2. The lane is the **whole frame** — a measured flow vector says nothing about
     carriageway edges — so this is a weaker scene than an analyst would draw.
  3. The legal direction is *measured from the footage*, so the sampling changes the
     scene hash and can change the count. Stride 5→10 swung the derived heading by
     ~19°. A count quoted without its command is meaningless. **reproduced**

## 7. Red-light semantics

- Support is **latched at the stop-line crossing**, never `in_junction AND signal_red`.
  The naive predicate fails in the commonest real case — the light turning green a
  second after the violator entered — and it fails *selectively*, losing exactly the
  marginal cases enforcement exists to catch.
- The signal is read **at the forward crossing instant**, not at polygon entry, because
  a stop line and the junction it guards are generally not contiguous.
- One crossing is **spent by** the entry it enabled, so a later boundary wobble cannot
  re-latch a stale state and mint a second event for one act. (This was a real defect,
  found and fixed, with named regression tests.)
- `UNKNOWN`, `AMBER` and `OFF` **never confirm**. `AMBER` is excluded on purpose: it
  means "stop if safe to do so", a judgement no geometry here can make.
- **The signal state is declared or transcribed, never perceived.** Nothing classifies
  a signal head from pixels; `SignalSourceMode.ROI_CLASSIFIER` is contract-defined and
  unimplemented.
- **observed:** on `clean_001.ogv` with an analyst-authored stop line, junction and a
  schedule transcribed from an in-frame signal head *and countdown* (58→44, one tick
  per second of media time, red lamp lit in every sampled frame): 434 frames, 107
  vehicle tracks, **0 junction entries, 0 confirmed events**.
- **The latch is not exercised by that run.** Nothing crossed, so no signal was ever
  read at a crossing; every latched entry state is `UNKNOWN`. The zero is a
  false-positive check, not evidence a violator would be caught.
- The counterfactual was measured: the same line drawn across the *cross-traffic* path
  yields **4 forward crossings and 74 validated-inside observations**, all of which the
  same declared RED would have confirmed, all falsely.
- Known limitation: a stop-line **overrun** is not distinguished from a passage —
  separating them needs the stationary stream's fact, which this join does not receive.

## 8. Illegal-stopping semantics

- A two-stream join: **in-zone** membership of a `no_stopping` zone AND a **stationary**
  observation, sustained for `stationary_duration`. **tested**
- Deliberately excludes congested scenes, and does not re-associate a long-stationary
  vehicle across a tracker ID switch. Both are documented deferrals.
- **unvalidated on real footage**, and not for want of trying: **no clip in the corpus
  can honestly supply a zone.** Every stationary vehicle in it is stopped *lawfully* —
  gridlock (Raxaul, Gangtok), a red light (Chiang Mai), a bus stop (Montreal). The only
  fixed-camera clip (`clean_003`, 7 px of motion) is 7.14 s long and nothing ever stops.
- The missing artifact is **footage**, not a drawing: a vehicle at rest where stopping
  is restricted *and the restriction is visible in frame*, from a fixed camera, long
  enough to exceed a dwell threshold declared **before** looking at the clip.

## 9. Triple-riding semantics

- Confirms when a motorcycle's `rider_count` is **≥ 3** for `min_persistence` (1.0 s),
  across at least two observations. Pure geometry — no classifier. **tested**
- **observed, three clips, 0 events on all three.** The interesting one is Raxaul:
  best track `iou-7` carried three riders in **40 of 53 observations (75.5 %)**, but its
  longest uninterrupted ≥3 run was **0.50 s** against the 1.0 s threshold. Seven of the
  nine breaks were a **single** sub-threshold observation.
- Chiang Mai re-run over its **whole** clip (434 frames, 1112 observations, 29
  motorcycles): the **maximum rider count observed anywhere is 2**. Nothing to confirm.
  Montreal yields **zero motorcycle detections** entirely.
- **The rule is not wrong to end the run.** It is handed
  `RiderCountObservation.rider_count`, an integer that *asserts* how many riders there
  are. A count of 1 asserts one rider; bridging it anyway would mean the rule
  overruling an observation with no evidence, and a rider who genuinely dismounts must
  still end the run.
- **The limitation is in the observation contract, not the rule.**
  `RiderCountObservation` has **no uncertainty channel** — no way to say "this frame's
  count is untrustworthy". Its siblings do: `HelmetStateObservation` has `uncertain`,
  `SpeedObservation` has `speed_sigma`. The legitimate fix is that channel plus
  withhold-and-bridge, which is a reviewed contract change — **not** a lower threshold.

## 10. Helmet limitations

- **Experimental. No helmet rule runs in the demo; no helmet event is ever confirmed.**
- Two independent blockers, either sufficient:
  1. **Turban capability.** The exemption needs a `turban` label. The trained P4-U5
     ResNet-50 is binary and *cannot emit it*; the shipped zero-shot backend emits it
     unreliably. The capability guard therefore refuses to build the rule.
  2. **Driver-only attribution.** Helmet state is a fact about a rider; it becomes a
     violation only once the system can say *whose duty* it was. Only the `DRIVER` slot
     carries that, and it is assigned only when exactly one rider is associated.
- Consequently **42.4 % of the frozen test corpus (3,137 of 7,406 annotated
  motorcycles) is unattributable by design**, rising to **157 of 194 crops (81 %)** in a
  real Bihar congestion clip. **benchmarked** (classifier-level) / **observed** (clip).
- Best measured runtime backend: ResNet-50 at **0.857 native macro-F1**, *per-crop and
  conditional on a rider reaching the classifier at all*, with **74.4 % recovery**.
  End to end, a helmet decision is produced for roughly **38 %** of annotated
  motorcycles.
- Per-frame labels flip on real footage; they are smoothed per track **for display
  only**. Smoothing is **unvalidated** as an accuracy improvement and is never claimed
  as one.

## 11. ID-switch handling

- The contract and **every** reasoner handle ID switches correctly: a tainted step may
  abstain but can never contribute to a confirmation. `StubTracker` proves this. **tested**
- **The shipped tracker never sets the flag.** `iou_tracker.py` stamps `tainted=False`
  unconditionally and says why: greedy IoU "exposes no trustworthy ID-switch signal, so
  fabricating taint would be dishonest."
- Therefore, in a real run today **ID switches go undetected** — the guard never fires.
  Every real-footage run reports `taint restarts: 0`, and that zero means "nothing was
  reported", not "nothing happened".
- The consequence is asymmetric: a split loses support and under-reports; a merge could
  carry support across a vehicle boundary. Closing it needs a tracker with motion state
  that can flag a kinematic discontinuity — not a heuristic invented at the rule layer.

## 12. Abstention philosophy

The system prefers a defensible silence to a confident guess. Concretely:

- No `DRIVER` slot → **no helmet attribution**, rather than a guessed one.
- Turban-blind backend → the capability guard **refuses to build the rule**.
- No scene geometry → geometry rules are **not run**, rather than run on the example
  scene's road.
- No recorded stop-line crossing → the signal resolves `UNKNOWN` and **never latches**.
- Support that does not hold for the required second → **no event**, even at 75.5 %.
- Nothing labelled → **observations are reported, never accuracy**.

The line we do not cross: **no threshold is ever tuned until a clip produces the
answer we wanted.** Lowering triple riding's persistence, or its rider threshold, would
have produced a positive demo; all three were refused and the reasons are written down.

## 13. Evaluation methodology

- The evaluator is a **pure core**: the caller supplies model and data configuration;
  it computes nothing about acquisition. Metrics distinguish `None` ("not measured")
  from `0.0` ("measured as zero"). **tested**
- The real-footage corpus (`test-videos/`) is freely licensed, **gitignored**, and
  reproducible from `sources.yaml` via `fetch.py`. Media is never committed.
- `test-videos/evaluation/manifest.yaml` records expectations; entries marked
  `established: false` are **deliberately never checked**, because checking a
  placeholder would manufacture a benchmark out of a guess.
- T0 structural corpus: **18 passed, 0 failed, 13 not evaluated**. The 18 are
  guaranteed *by construction* — with no calibrated scene the rule cannot be built, so
  the count is zero before a frame is decoded. **reproduced**

## 14. CNN vs ViT (P4-U5) — the only benchmarked result

- Master-spec §4 requirement, run on the CC-BY-4.0 HELMET dataset under a
  **pre-registration frozen and git-tagged before the final runs**.
- Decision rule required sign-consistency across all three seeds **and** a CI excluding
  zero.
- **ResNet-50 wins accuracy:** mean test macro-F1 **0.92881** vs **0.91975** for
  DeiT-Small; difference −0.0091, 95 % CI **[−0.0138, −0.0043]**, sign-consistent across
  seeds 0/1/2.
- **DeiT-Small is the cheaper model** on the measured RTX 4060 Laptop benchmark:
  **−29.8 % median latency** (20.77 ms vs 29.57 ms) and **−34.2 % peak VRAM**
  (184.5 MiB vs 280.3 MiB) at batch 32.
- **Neither model is adopted into the runtime.** It trained on *whole-motorcycle*
  crops, not the runtime's derived *head* crops.
- **0.929 is not a runtime number and must never be quoted as one.** This is the single
  most likely number to be misused in a viva.
- The experiment is **frozen**: `experiments/`, `docs/cnn-vs-vit-results.md`,
  `docs/adr/`, `registry/`, `configs/`, `schemas/` are byte-untouched.

## 15. Runtime-equivalent evaluation

- P4-U8/U9/U10 asked a different question from P4-U5: not "which architecture is more
  accurate on the dataset's own crops" but "what happens when a backend is fed the
  crops **this runtime actually produces**".
- That is why the headline runtime figure (0.857 native macro-F1) is **lower** than
  P4-U5's 0.929 and is **not comparable** to it — different crop population, different
  conditioning.
- The runtime figure is **conditional**: per-crop, on single-rider crops, given the
  rider reached the classifier. The 74.4 % recovery and 42.4 % multi-rider exclusion
  must be stated in the same breath.
- **No backend was adopted.** The turban capability gap is the blocker.

## 16. Live camera

- A browser camera streamed into a persistent backend session running the **same**
  pipeline an uploaded video runs — same engine provider, detector threshold, tracker,
  association, classifier and reasoners. **There is no live-only rule anywhere.**
- One WebSocket carries the session. Back-pressure holds **exactly one frame** pending
  server-side; frames superseded while inference is busy are **dropped, never queued** —
  a live view must show the road now, not a backlog.
- **observed** on this machine's CPU: **~1 fps of inference and ~1.6 s end-to-end
  delay**. The preview stays smooth because it is deliberately not analysed
  frame-for-frame. The engine resets every 600 processed frames.
- Live events are **not persisted** and never enter the repository, analytics or review.
- **Not validated:** the physical browser camera — this machine exposes none, so
  `getUserMedia` → canvas → socket has never run against real hardware. Wrong-way,
  illegal-stopping and red-light behaviour is also unavailable live, because those need
  a calibrated scene the live path is not given.

## 17. The biggest limitations, ranked

1. **Nothing in the real-footage corpus is labelled.** No precision, recall or F1 exists
   for any violation. Every real-footage number is an *observation of one run*.
2. **ID switches are undetected in practice** — the guard is correct and inert (§11).
3. **Red-light's latch is unexercised on real pixels** (§7); illegal stopping has no
   real example at all (§8).
4. **Helmet enforcement is blocked** on turban capability and driver attribution (§10).
5. **Auto-calibrated geometry is sampling-dependent** — the wrong-way scene hash and
   count move with `--stride` (§6).
6. **Most of the corpus is hand-held.** Measured global motion: `clean_002` 468 px,
   `wrongway_001` 576 px, `congestion_002` 1897 px. Only `clean_003` (7 px) is a fixed
   camera, and it is 7 s long. Static `SceneConfig` polygons assume a fixed camera.
7. **Speeding, ANPR, penalty simulation and privacy/redaction are not implemented.**
8. **Single-process runtime**: in-memory job registry, one API worker, no shared state.
9. **Night and rain are untested** — no such clip exists. That regime is *untested*,
   not *passed*.

## 18. What is and is not demonstrated

| Capability | Status | The honest one-liner |
|---|---|---|
| Detection / tracking / association | **Demo ready** | Real RT-DETR + tracker on real footage, with recall limits published. |
| Wrong way | **Demo ready** | 16 real events — against a *derived* direction, on *lawful* contraflow. |
| Evidence + analyst review | **Demo ready** | Measurements, thresholds, rule trace, content-addressed artifacts, append-only decisions. |
| Analyst calibration surface | **Demo ready** | Scenes are drawn, content-addressed, and gate which rules can run. |
| Red-light jumping | **Conditional** | Ran on real footage against a transcribed signal; correctly confirmed **nothing**; latch unexercised. |
| Triple riding | **Conditional** | Full real path; **0 events**; demonstrate the *abstention*. |
| Illegal stopping | **Conditional** | Structurally complete; **no real example is possible** with this corpus. Confirms on the controlled clip (§18a). |
| Controlled multi-violation demo | **Demo ready — as a demonstration, not validation** | One synthetic clip + declared context → four families confirmed in one pass (§18a). |
| Live camera | **Conditional** | Backend validated end to end; **no physical camera** ever tested. |
| No-helmet enforcement | **Experimental — do not claim** | Classifies riders; mints **no** events; two independent blockers. |
| Speeding / ANPR / penalties / redaction | **Not demo ready** | Not implemented. Do not describe as future work "nearly done". |

### 18a. The controlled demonstration — a separate category

One 6-second synthetic clip, an operator-drawn scene, a declared signal schedule and a
written declaration of what the clip contains. Processing it confirms **four** families
in one pass: wrong way, illegal stopping, red-light jumping, triple riding
([`controlled-demo.md`](controlled-demo.md), validation matrix §7b).

- It is the **only** place the red-light latch is actually exercised — the vehicle
  crosses on red and the schedule turns green afterwards, and the event still confirms.
- It is the **only** place illegal stopping confirms at all.
- Declared expectations are stored where **no rule, reasoner or engine can read them**;
  `ProcessRequest` and `EngineConfig` have no field for them, and tests assert that. A
  declared family with no event is reported **missing**, never conjured.
- It computes **no accuracy**. Precision over one hand-authored clip would be
  arithmetic against ground truth the same person wrote.
- Two renderings: a **rectangle** clip the test suite replays against a scripted
  detector (RT-DETR detects nothing in it — measured), and a **composited** clip built
  from real vehicle crops cut from this project's own corpus, which real RT-DETR does
  detect. The composited run confirmed all four families over HTTP against `serve:app`
  (647 detections, 60 frames).
- **Even so, it says nothing about real-world performance.** RT-DETR is looking at
  unchanging vehicle cut-outs sliding across an empty synthetic road: no occlusion, no
  perspective change, no lighting variation, no other traffic. Never narrate it as "we
  detect four violations" — four were **reasoned**, from context that was declared.
  **demonstrated**, not *observed* and not *benchmarked*.

---

## Fast answers to likely questions

- *"What's your accuracy?"* — For a violation, none exists: nothing in the real-footage
  corpus is labelled. The only benchmarked number in the project is a **classifier**
  comparison on a public dataset (§14), and it is not a runtime number.
- *"Why didn't it detect anything in the red-light clip?"* — Because the approach was
  red and its traffic was stopped. Nothing crossed the line. Drawn wrongly it would have
  confirmed four vehicles, all falsely (§7).
- *"Three people are clearly on that bike — why no event?"* — They were, in 75.5 % of
  that track's frames. The evidence never held for one uninterrupted second, and the
  observation contract cannot distinguish a genuine count drop from a momentary
  association failure (§9).
- *"Is this production-ready?"* — No, and the repository says so in the README, the
  deployment guide and the validation matrix. It is a research foundation.
- *"Could you just lower the threshold?"* — Yes, and it would produce a positive demo.
  That is precisely why it was refused (§12).
- *"Isn't the controlled demo just faking it?"* — No, and the difference is checkable.
  The clip is real video that is really decoded; the tracker, every observation
  derivation and all four reasoners are the production ones with their production
  thresholds. What is authored is the **scenario and the scene context** — and every
  one of those context facts is something a camera genuinely cannot know. What is
  scripted is only the *detector*, because a COCO model does not fire on synthetic
  rectangles. The expectations I declare are never shown to the reasoners: they live in
  a separate store, `ProcessRequest` has no field for them, and if a rule declines to
  confirm, the table says *missing* (§18a).
- *"Why not just film the violations yourself?"* — Staging real traffic violations is
  unsafe and unlawful. A declared synthetic scenario, labelled as such, is the honest
  substitute; quietly presenting it as real footage would not be.
