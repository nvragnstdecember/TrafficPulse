# TrafficPulse — Phase 1 Implementation Plan

**Status:** Authoritative Phase 1 unit plan (current).
**Companion:** [`docs/phase-0-plan.md`](phase-0-plan.md) governs Phase 0-F only and is
**not** superseded or rewritten by this document.
**Canonical architecture reference:** [`docs/architecture-review.md`](architecture-review.md).

## 1. Status and authority

- **This document is the authoritative Phase 1 unit plan.** Before this document
  existed, Phase 1 had no unit-card plan: `docs/phase-0-plan.md` explicitly plans
  *Phase 0-F only* ("It does not plan Phase 1 in detail" … "Detailed Phase 1 unit
  cards are authored when Phase 1 begins, not in this document"). Phase 1 units
  P1-U1…P1-U7 were executed and committed to Git without a plan document; this file
  closes that gap.
- **Phase 0-F remains governed by [`docs/phase-0-plan.md`](phase-0-plan.md).** That
  document and its historical U1–U6 unit cards are intact and are not edited by this
  plan.
- **Two separate identifier namespaces.** Phase 0-F units are `U1`…`U6`
  (foundation: repo baseline, contracts, ontology, registry, scene schema, ADR
  pack). Phase 1 units are `P1-U1`, `P1-U2`, … These namespaces are **distinct**:
  Phase 0-F `U6` is the *ADR pack*; Phase 1 `P1-U6` is the *detector foundation*.
  Always write the `P1-` prefix for Phase 1 units.
- **Completed Phase 1 units are recorded retrospectively** from Git history
  (§3); this plan makes no capability claim beyond what those commits implemented.
- **Forward unit cards (§5) govern future Phase 1 work.** The next unit is
  unambiguous: **P1-U8 — Tracker integration foundation.**

## 2. Phase 1 objective

Deliver a **real, reproducible, reviewable traffic-violation vertical slice** —
**wrong-way driving, end to end, from recorded video** — and then extend the system
outward from that slice. Phase 1 is complete for its first milestone when a recorded
clip can be processed offline into a confirmed wrong-way event with a minimal,
provenance-bearing evidence output, using the frozen contracts and the permissive
detector/tracker posture, with no fabricated capability.

Non-negotiable properties preserved throughout Phase 1 (architecture-review §5, §13,
§14, §15, §19; ADR-001, ADR-002, ADR-003):

- **Offline-first** — recorded video in, events out (ADR-003); any "real-time"
  language stays confined to a labelled demo mode.
- **Contract seams** — each layer consumes and produces only the frozen U2 contracts
  (`Detection`, `TrackState`, `Observation`, `ViolationHypothesis`, `ConfirmedEvent`,
  `EvidenceManifest`); no framework/model/tracker-native type leaks past its seam.
- **Deterministic / reproducible** — reasoning is a pure function of typed inputs;
  identity is content-derived; no wall-clock or randomness in the decision path.
- **Uncertainty & abstention** — every non-confirmation is a logged, countable
  abstention; tainted (ID-switch) tracks may abstain but never confirm.
- **Evidence provenance** — every confirmed event is traceable to its inputs
  (scene-config hash, rule id/version, model/tracker refs).
- **Permissive-only perception** — detector and tracker code paths use Apache-2.0 /
  MIT only; **no AGPL** (no Ultralytics, no BoxMOT) (ADR-001).

## 3. Retrospective completed-unit matrix (P1-U1 … P1-U7)

Recorded from Git history at `HEAD = 8b6d51f`. Objectives describe what the commit
implemented; nothing here claims capability beyond the repository.

| Unit | Commit | Objective | Principal package / files | Role in the wrong-way vertical slice |
|---|---|---|---|---|
| **P1-U1** | `0dfc774` | Geometry primitives | `src/trafficpulse/geometry/` (`vectors.py`, `segments.py`, `polygons.py`) | Geometric kernel: `displacement`, `angle_between_degrees`, `is_zero_vector` — the math heading derivation is built on. |
| **P1-U2** | `0ff1bc0` | Synthetic trajectory generator | `src/trafficpulse/synth/` (`trajectories.py`, `scenarios.py`) | Reproducible source of golden **`TrackState`** sequences with known event labels — validates the rule path with no model and no video. |
| **P1-U3** | `7ad37c6` | Rule-engine core | `src/trafficpulse/rules/engine.py`, `rules/states.py` | Violation-agnostic hypothesis-lifecycle FSM + accumulation + abstention (the "≥2 observations to confirm" / "every non-confirmation abstains" guarantees). |
| **P1-U4** | `4651ffb` | Wrong-way reasoning | `src/trafficpulse/rules/wrong_way.py`, `observations/heading.py` | Heading-vs-lane observation **derivation from `TrackState`** + sustained-contradiction temporal reasoning → mints frozen `ConfirmedEvent`. |
| **P1-U5** | `dd70edd` | Video ingestion | `src/trafficpulse/ingestion/video.py` | PTS-accurate decode to `FrameRecord` (RGB uint8, media-relative PTS timestamps) — the slice's real input. |
| **P1-U6** | `07f8baa` | Detector integration foundation | `src/trafficpulse/detector/` (`interface.py`, `raw.py`, `frame.py`, `config.py`, `adapter.py`, `errors.py`, `stub.py`) | `Detector` seam + `DetectionAdapter` → frozen `Detection`; keeps the detector choice bounded (ADR-001). |
| **P1-U7** | `8b6d51f` | RT-DETR detector backend | `src/trafficpulse/detector/rtdetr.py` | First **real** detector behind the P1-U6 seam (Apache-2.0 HF Transformers RT-DETR port); produces real `Detection`s. |

**Load-bearing observation:** `observations/heading.py` already consumes
`Sequence[TrackState]`, and `synth/` already emits `TrackState`. So the path
`TrackState → heading → wrong-way → ConfirmedEvent` **already works today** on
synthetic tracks. The single missing bridge for a *real* slice is
**`Detection` → tracker → `TrackState`**, plus orchestration and evidence output.
This is what the forward sequence targets.

## 4. Forward unit sequence (bounded, dependency-ordered)

The canonical data flow (architecture-review §14) is:

```
Detection → TrackState → Association → Observation → ViolationHypothesis
          → ConfirmedEvent → EvidenceManifest → ReviewCase → SimulatedPenalty
```

Everything from `Observation` onward for wrong-way already exists (P1-U3/U4). The
critical path to the first real slice therefore fills exactly the `Detection →
TrackState` bridge and the wiring/output around it:

| Order | Unit | Title | Adds |
|---|---|---|---|
| 1 | **P1-U8** | Tracker integration foundation | `Tracker` seam + `StubTracker` + adapter/config/errors → frozen `TrackState` (no tracker dependency) |
| 2 | **P1-U9** | Real tracker backend | Concrete detection-based tracker (ByteTrack default / permissive fallback) behind the P1-U8 seam |
| 3 | **P1-U10** | Detection→tracking→observation orchestration | Thin pipeline wiring ingestion → detector → tracker → *existing* heading derivation → wrong-way reasoner |
| 4 | **P1-U11** | Minimal event persistence + evidence stub | Persist `ConfirmedEvent` + minimal `EvidenceManifest` with provenance |
| 5 | **P1-U12** | Real-video end-to-end verification + demo hardening | Run the full path on a repository-safe clip; one wrong-way event with evidence; determinism |

**Why these boundaries (chosen by scope minimisation, not the prompt's suggested
names):**

- **P1-U8 and P1-U9 are split** (not merged), mirroring the committed P1-U6/P1-U7
  detector pattern (contract-first foundation, then real backend). The split has
  concrete payoff: P1-U10 orchestration can be built and tested against
  `StubTracker` **immediately**, and the one unit carrying external-dependency and
  licence-audit risk (the concrete tracker) is isolated in P1-U9 — which also
  carries a permissive in-repo fallback so the demo can never be blocked by external
  tracker friction.
- **Orchestration (P1-U10) is a single thin unit and is *not* split from observation
  derivation** — because, resolved from the code, `observations/heading.py` already
  exists and already consumes `TrackState`. P1-U10 wires existing components; it does
  not re-implement derivation.
- **Persistence/evidence (P1-U11) is a separate unit** because it is a distinct
  ADR-002 storage concern; kept deliberately minimal (a stub manifest, no rendering).
- **Real-clip verification (P1-U12) is separated from P1-U10** because P1-U10 is
  developed and tested on synthetic tracks + `StubTracker`; the real-clip run is the
  integration/hardening milestone (exactly the "synthetic first, then one real clip"
  sequencing `phase-0-plan.md` recommends for the first slice).

**Critical-path length: 5 units (P1-U8 … P1-U12).** Everything else is post-slice
backlog (§7) and is deliberately kept off this path.

## 5. Unit cards (critical path)

Each card follows the master-spec §21 unit philosophy. Compute is CPU-first
throughout; no unit downloads a dataset or trains a model.

---

### P1-U8 — Tracker integration foundation

- **Objective:** a framework-neutral **tracker seam** that turns per-frame
  `Detection` streams into ordered, identity-bearing frozen U2 `TrackState`
  sequences — the tracking analogue of the P1-U6 detector foundation — so a concrete
  tracker (P1-U9) plugs in without an API change and downstream code depends only on
  `TrackState`.
- **Prerequisites:** P1-U6 (`Detection` seam), U2 `TrackState`/`TrackStatus`/
  `Velocity` contracts (frozen).
- **Exact scope:** a new `src/trafficpulse/tracking/` package containing:
  a `Tracker` ABC with an explicit single-stream stateful update contract
  (consume the `Detection`s of one frame in ascending `frame_index`, return the
  `TrackState`s active at that frame); a framework-neutral raw assignment type
  (`track_id`, source `detection`, `status`, `tainted`) if needed; a
  `TrackAdapter`/assembly step that stamps frozen `TrackState` (carrying `track_id`,
  `camera_id`, `frame_index`, `timestamp`, `object_class`, `bbox`, `status`,
  `tainted`, optional `velocity`, optional `tracker` `ModelRef`); a `TrackerConfig`
  (frozen pydantic) for framework-neutral settings (e.g. max-age/min-hits/IoU-match
  thresholds as generic knobs, and the `tracker` `ModelRef` provenance stamp); a
  deterministic `StubTracker` that replays scripted assignments per frame; and a
  tracker error taxonomy (`TrackerError` + subclasses) mirroring `detector/errors.py`.
- **Explicit out-of-scope:** any real/learned tracker or external tracker dependency
  (that is P1-U9); Kalman/motion models beyond what a stub needs; re-ID; multi-camera;
  association (rider/vehicle/plate); observation derivation; any pipeline wiring;
  any persistence; adding `tracking` to the governance allow-list is done **in this
  unit's implementation commit**, not before.
- **Deliverables:** the `tracking` package above; `tests/tracking/`.
- **Acceptance criteria:** ruff + `mypy --strict` + pytest green; `StubTracker`
  satisfies `Tracker`; importing `trafficpulse.tracking` pulls in **no** ML/tracker
  framework; adapter output is exactly the frozen `TrackState` contract (no
  tracker-native type leaks); identity and status assignment are deterministic;
  tainted flag is representable and set on scripted ID switches; the `tracking`
  package is added to the sanctioned-package allow-list in
  `tests/docs/test_adr_pack.py` (this unit is the first legitimate place to do so).
- **Required tests:** interface conformance; boundary test (no ML import; only
  `TrackState` escapes); adapter conversion + determinism; per-track ordering;
  status/tainted assignment; empty-input → empty output; malformed-assignment
  rejection.
- **Datasets/artifacts permitted:** none.
- **Compute:** negligible (CPU).
- **Verification:** `ruff check .` · `mypy src` · `pytest -q tests/tracking` ·
  `pytest -q`.
- **Stop conditions:** if the `TrackState` contract proves insufficient to express
  tracker output, **stop and record a contract-gap note** — do not modify the frozen
  contract without a separate decision.
- **Fallback:** if the stateful per-frame API proves awkward, fall back to a
  batch API (`Sequence[frame Detections] → dict[track_id, list[TrackState]]`) behind
  the same seam; keep the contract boundary identical.
- **Dependencies:** P1-U6, U2 contracts.

---

### P1-U9 — Real tracker backend

- **Objective:** the first **real** detection-based tracker behind the P1-U8 seam,
  producing genuine `TrackState` sequences from real `Detection`s — the tracking
  analogue of P1-U7.
- **Prerequisites:** P1-U8.
- **Exact scope:** one concrete tracker implementation behind `Tracker`.
  **Default direction: ByteTrack (MIT [V] per architecture-review §10).** Because no
  web research is available in this environment and no fresh external verification is
  claimed here, **this card requires an execution-time audit before integrating any
  concrete external tracker** (see "Tracker decision constraints" below): confirm the
  library identity, licence (**must be MIT/Apache**, **never AGPL** — BoxMOT is
  excluded, ADR-001), dependency-resolver plan (preserve Python ≥3.11 and NumPy
  `<2.2`; no mandatory base-dep additions — use an optional extra like the `rtdetr`
  extra), and provenance. Framework-native tracker types stay inside the backend.
- **Explicit out-of-scope:** pipeline wiring (P1-U10); persistence/evidence (P1-U11);
  re-ID; multi-camera; motion-model research beyond what the chosen tracker ships;
  committing any weights/artifacts.
- **Deliverables:** the tracker backend module in `tracking/`; an optional dependency
  extra if an external library is used; fake/stub-driven unit tests; an opt-in real
  smoke test (skipped by default), matching the P1-U7 pattern.
- **Acceptance criteria:** ruff + `mypy --strict` + pytest green; the backend
  satisfies `Tracker`; base install and CI stay tracker-dependency-free (lazy import
  / optional extra); no tracker-native object escapes the seam; deterministic given
  fixed input where the tracker permits; **execution-time licence/provenance/
  dependency audit recorded** and MIT/Apache confirmed; `pip check` clean if a
  dependency is added.
- **Required tests:** interface conformance on fake detections; boundary (no leak);
  determinism; identity continuity across frames; ID-switch → tainted; opt-in real
  smoke on synthetic in-memory detections (no network/GPU/weights in the default
  suite).
- **Datasets/artifacts permitted:** none required. Tracker-sanity clips (MOT17/20 —
  "tracker integration sanity only", architecture-review §7) are optional and **not**
  on the critical path; no dataset is downloaded without explicit approval.
- **Compute:** CPU-first; a real tracker is lightweight (architecture-review §10:
  "negligible").
- **Verification:** `python -m pip check` (if a dep was added) · `ruff check .` ·
  `mypy src` · `pytest -q tests/tracking` · `pytest -q`.
- **Stop conditions:** if the intended tracker's licence/provenance/dependency
  cannot be verified at execution time, **stop and report** rather than integrate.
- **Fallback:** a **minimal in-repo IoU/greedy associator** (permissive, no external
  dependency: greedy IoU matching + track age/min-hits lifecycle, no Kalman) behind
  the same seam. This keeps the critical path unblocked without any AGPL/licence risk
  and without an external dependency; ByteTrack then becomes a later enhancement.
- **Dependencies:** P1-U8.

---

### P1-U10 — Detection→tracking→observation vertical-slice orchestration

- **Objective:** a thin, deterministic **offline orchestration** that runs one
  recorded stream end to end through existing components: frames → detections →
  tracks → heading observations → wrong-way reasoning → `ConfirmedEvent`s.
- **Prerequisites:** P1-U8 (seam), P1-U5 (ingestion), P1-U6 (detector seam/adapter),
  P1-U4 (heading derivation + wrong-way reasoner), U5 `SceneConfig`. P1-U9 or the
  P1-U9 fallback provides real tracks; **P1-U10 can be built and tested against
  `StubTracker`** and does not block on P1-U9.
- **Exact scope:** a single orchestration module (e.g.
  `src/trafficpulse/pipeline/` or a `run_*` function in an existing package —
  decided at implementation time; **no new speculative package unless a module
  genuinely needs one**) that, for one `SceneConfig` and one video source:
  iterates `FrameRecord`s (P1-U5) → wraps each as a detector `Frame` → runs the
  injected `Detector` + `DetectionAdapter` → feeds per-frame `Detection`s to the
  injected `Tracker` → **groups the emitted `TrackState`s by `(camera_id, track_id)`
  in timestamp order** → calls the existing `derive_heading_observations_with_taint`
  per track with the scene's single lane legal-direction → feeds the observations
  (with taint restarts) to `WrongWayReasoner` → collects `ConfirmedEvent`s. Assumes a
  **single-lane scene** for the first slice.
- **Explicit out-of-scope:** re-implementing detection/tracking/derivation/reasoning
  (all injected/existing); zone-membership assignment / multi-lane routing
  (`in_zone`); association; persistence and evidence (P1-U11); real-clip validation
  and demo packaging (P1-U12); any change to frozen contracts or to P1-U4/U5/U6
  semantics.
- **Deliverables:** the orchestration function/module; `tests/pipeline/` (or
  equivalent) driving it with `StubDetector` + `StubTracker` + synthetic frames.
- **Acceptance criteria:** ruff + `mypy --strict` + pytest green; a scripted
  stub detector+tracker over synthetic frames yields the **same** `ConfirmedEvent`
  set as calling the existing derivation+reasoner directly on equivalent
  `TrackState`s (proving the wiring adds no behaviour); deterministic and
  order-independent per the P1-U3/U4 policy; no frozen contract or upstream semantic
  changed; per-track grouping and timestamp ordering verified.
- **Required tests:** stub-driven end-to-end producing a known wrong-way event;
  equivalence-to-direct-call test; multi-track grouping; taint-restart propagation
  through the pipeline; empty video → no events.
- **Datasets/artifacts permitted:** none (synthetic in-memory frames/detections).
- **Compute:** negligible (CPU; stubs).
- **Verification:** `ruff check .` · `mypy src` · `pytest -q`.
- **Stop conditions:** if single-lane routing proves insufficient even for the demo
  scene, record the gap; do **not** pull multi-lane `in_zone` assignment onto the
  critical path without a decision.
- **Fallback:** if full frame-by-frame streaming orchestration is heavy, fall back to
  a batch orchestration (decode all frames, then run the stages) — identical output
  contract.
- **Dependencies:** P1-U8, P1-U5, P1-U6, P1-U4, U5.

---

### P1-U11 — Minimal event persistence + evidence stub

- **Objective:** make a confirmed event **reviewable**: persist each
  `ConfirmedEvent` and a minimal, provenance-bearing `EvidenceManifest` deterministically.
- **Prerequisites:** P1-U10 (produces `ConfirmedEvent`s), U2 `ConfirmedEvent` /
  `EvidenceManifest` / `ArtifactReference` / `RuleTraceStep` contracts, ADR-002.
- **Exact scope:** a minimal writer that, given a run's `ConfirmedEvent`s, emits (a)
  the `ConfirmedEvent` records and (b) a minimal `EvidenceManifest` per event whose
  `trigger_frame`/`before_frame`/`after_frame` are `ArtifactReference`s pointing at
  **frame indices/relative locators only** (no rendered artifact), plus a short
  `rule_trace` and the provenance fields already present on the event
  (`scene_config_hash`, `rule_id`/`rule_version`, model/tracker `ModelRef`s). Output
  format is the smallest defensible choice consistent with ADR-002 (deterministic
  JSON files, or a minimal SQLite table) — decided at implementation time. All output
  paths are gitignored runtime locations.
- **Explicit out-of-scope:** the full evidence engine (clip/frame rendering, crops,
  overlays); ANPR/OCR; the review workflow; penalty simulation; content-addressed
  artifact hashing of real media; cross-run dedup (ADR-004 is Proposed — do not
  freeze it here).
- **Deliverables:** the writer module + tests; sample output shape documented.
- **Acceptance criteria:** ruff + `mypy --strict` + pytest green; persisted
  `ConfirmedEvent` round-trips back to an equal contract; the `EvidenceManifest`
  validates and references the correct event id + trigger frame; output is
  deterministic across repeated runs of identical input; no real media committed; no
  frozen contract changed.
- **Required tests:** event write→read round-trip equality; manifest validity +
  correct linkage; determinism (byte-identical or contract-equal across runs);
  gitignore coverage of the output location.
- **Datasets/artifacts permitted:** none; only synthetic/generated inputs.
- **Compute:** negligible (CPU).
- **Verification:** `ruff check .` · `mypy src` · `pytest -q`.
- **Stop conditions:** if ADR-002's storage substrate choice is contested, default to
  deterministic JSON files for the slice and record the choice as provisional.
- **Fallback:** JSON-file persistence if SQLite wiring is disproportionate for the
  first slice.
- **Dependencies:** P1-U10, U2 contracts, ADR-002.

---

### P1-U12 — Real-video end-to-end verification + demo hardening

- **Objective:** prove the first vertical slice on **real video**: a recorded clip
  produces at least one confirmed wrong-way event with a minimal evidence output,
  reproducibly, using the real detector (P1-U7) and a real/fallback tracker (P1-U9).
- **Prerequisites:** P1-U8–P1-U11, P1-U7 (real detector), P1-U5 ingestion, U5 scene.
- **Exact scope:** run the P1-U10 orchestration on a **repository-safe** clip
  (a locally generated synthetic video or an approved local clip with recorded
  provenance — **no dataset download, no private footage**) configured with a
  single-lane scene whose legal direction contradicts the moving object, producing
  ≥1 wrong-way `ConfirmedEvent` + its `EvidenceManifest` (P1-U11). Harden the path
  (clear typed errors on empty / no-detection / no-track runs; deterministic replay;
  documented run command). Record the real-run report (detector/tracker refs, frame
  count, event count) analogous to the P1-U7 smoke report.
- **Explicit out-of-scope:** accuracy/precision/recall claims (no evaluation dataset);
  additional violations; multi-lane/multi-camera; UI; analytics; ANPR; performance
  benchmarking beyond an informational note; committing any video/model artifact.
- **Deliverables:** an opt-in end-to-end verification (skipped by default unless deps
  + a local clip are provided, matching the P1-U7 smoke pattern); a short run
  report; a documented reproduction command.
- **Acceptance criteria:** on the repository-safe clip, the full path
  video→frames→detections→tracks→observations→reasoning→event→evidence completes and
  yields ≥1 wrong-way event with a valid `EvidenceManifest`; the run is deterministic
  on repeat; the default CI suite remains green **without** the clip, deps, or GPU; no
  frozen contract/semantic changed; no artifact committed.
- **Required tests:** opt-in real end-to-end test (guarded by env var + dependency
  availability, like `test_rtdetr_smoke.py`); the default suite must pass with it
  skipped.
- **Datasets/artifacts permitted:** a **generated** synthetic video or an approved
  local clip with provenance; nothing committed; no download without approval.
- **Compute:** CPU-first; CUDA only if available and compatible (optional).
- **Verification:** `ruff check .` · `mypy src` · `pytest -q` (slice skipped) ·
  the opt-in end-to-end run with a provided clip.
- **Stop conditions:** if no repository-safe clip can exercise a real detection→track
  path, report "implementation complete, real-clip verification pending an approved
  clip" (as P1-U7 did for the checkpoint) rather than fabricate a result.
- **Fallback:** a synthetic generated video (moving rectangle the detector fires on,
  or — if the real detector will not fire on synthetic pixels — the P1-U9 fallback
  tracker over injected detections) to exercise non-empty end-to-end output.
- **Dependencies:** P1-U8–P1-U11, P1-U7, P1-U5, U5.

## 6. First vertical-slice Definition of Done

The first Phase 1 milestone is **done** when this path runs offline and
deterministically:

```
recorded video
  → PTS-accurate frames (P1-U5 FrameRecord)
  → RT-DETR detections (P1-U7 behind the P1-U6 Detection seam)
  → tracker-produced TrackState sequences (P1-U9 behind the P1-U8 seam)
  → heading-vs-lane observations (existing P1-U4 derivation, per track)
  → wrong-way temporal reasoning (existing P1-U4 reasoner)
  → ConfirmedEvent
  → minimal EvidenceManifest + persisted event with provenance (P1-U11)
```

**"End-to-end" means:** a single command/entry point takes one recorded clip + one
`SceneConfig` and emits, offline and reproducibly, at least one wrong-way
`ConfirmedEvent` with a linked minimal `EvidenceManifest`, with every stage crossing
only its frozen contract seam, and with the run repeatable to an equal result.

**The Definition of Done explicitly does NOT require:** all six violation types; ANPR;
OCR; any UI; an analytics dashboard; production deployment; multi-camera operation;
model training; a full evidence-rendering engine; or any accuracy/precision/recall
claim. Those are backlog (§7), deliberately off the critical path.

## 7. Post-vertical-slice backlog (off the critical path)

Architecture-backed future work, **not** assigned P1-U numbers here (numbers are
assigned only when a unit is next and its dependencies are met). Grouped, not
ordered as a promise:

- **Perception depth:** OC-SORT alternative / ByteTrack A-B; multi-lane `in_zone`
  zone-membership routing; `Association` (rider↔vehicle, plate↔vehicle,
  architecture-review §5).
- **Additional violations:** no-helmet (hosts the mandatory CNN-vs-ViT helmet study,
  spec §12); triple riding; red-light jumping (signal-state ROI + line crossing);
  illegal stopping/parking; feasibility-gated speeding (calibrated zone only,
  excluded from penalty simulation until its gate passes).
- **Evidence & workflow:** the full evidence engine (best-frame selection, clip
  generation, overlays, content-addressed hashing); ANPR/OCR (PP-OCR via ONNX
  Runtime); review workflow / case states; simulated-penalty workflow; privacy /
  redaction.
- **Analytics & evaluation:** the evaluation harness for the seven protocols
  (architecture-review §23, `docs/evaluation-protocol.md`); analytics/technical
  dashboards; robustness slices (night).
- **Storage/runtime:** the observation writer (Parquet substrate, ADR-002); the event
  store runtime; cross-run event-identity/dedup semantics (ADR-004, currently
  Proposed).

None of the above is required for the first vertical-slice Definition of Done.

## 8. July 14, 2026 prioritisation note

The project owner wants a strong demonstrable state by **2026-07-14**. This is a
**prioritisation constraint, not a licence to weaken** architecture, tests, contracts,
licensing, provenance, or correctness, and **not** a completion promise for any
backlog item.

- **Immediate priority:** the shortest architecture-consistent path to the first
  **real wrong-way vertical slice** — the critical path **P1-U8 → P1-U9 → P1-U10 →
  P1-U11 → P1-U12** and nothing else.
- **A (minimum for the slice):** P1-U8, P1-U9, P1-U10, P1-U11, P1-U12.
- **B (valuable, safely *after* the slice):** OC-SORT alt, multi-lane routing,
  association, the observation writer / event-store runtime, the full evidence engine.
- **C (must NOT be pulled into the pre-demo critical path):** additional violation
  types, ANPR/OCR, review UI, analytics dashboards, penalty simulation, speeding,
  model training, multi-camera, production deployment.
- Sequencing is expressed by **dependency order and scope minimisation**, not by
  calendar-hour estimates or promised dates.

## 9. Tracker decision constraints (carry-forward for P1-U8 / P1-U9)

From the architecture and licensing evidence already in the repository
(architecture-review §5, §10; ADR-001) — **distinguished, not conflated**:

- **Architecture default:** **ByteTrack** (detection-based MOT, no re-ID needed for a
  fixed camera).
- **Permissive alternative:** **OC-SORT**.
- **Licence requirement:** the tracker code path is **MIT/Apache only**. **BoxMOT is
  excluded** (AGPL-3.0 licence-couples the stack — ADR-001 permissive-only posture).
- **Still required at execution time (P1-U9):** a fresh **licence + provenance +
  dependency-resolver audit** before integrating any concrete external tracker,
  verifying MIT/Apache, preserving Python ≥3.11 and NumPy `<2.2`, and using an
  optional dependency extra (never a mandatory base dependency). **No web research is
  available in this planning environment; no fresh external verification is claimed
  here** — the audit is a P1-U9 execution-time gate, with the permissive in-repo
  associator as the fallback if that gate cannot be cleanly passed.
