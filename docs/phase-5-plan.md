# TrafficPulse — Phase 5 plan

**Phase name:** Phase 5 — Metric Calibration, Feasibility-Gated Speeding, and Retro-Upgrade of Provisional Gates

- **Status:** Authoritative Phase 5 unit plan (planning document; no Phase 5 source
  implemented yet).
- **Date:** 2026-07-12
- **Authority:** This is the **authoritative Phase 5 unit plan**. It governs the
  `P5-U#` identifier namespace only. It does **not** supersede the Phase 0-F (`U#`),
  Phase 1 (`P1-U#`), Phase 2 (`P2-U#`), Phase 3 (`P3-U#`), or Phase 4 (`P4-U#`) plans;
  those remain governed by their own documents and are not rewritten here.
- **Canonical architecture reference:** [`docs/architecture-review.md`](architecture-review.md)
  remains THE canonical architecture reference; this plan interprets and sequences it
  for Phase 5 without modifying it or the master spec
  ([`TRAFFICPULSE_MASTER_SPEC.md`](../TRAFFICPULSE_MASTER_SPEC.md)).
- **Basis:** Implements the accepted architectural design review (2026-07-12):
  **calibration before speeding**; speeding remains **feasibility-gated** and excluded
  from any penalty simulation until a calibrated-scene evaluation justifies it;
  capability-first sequencing; composition over inheritance; frozen contracts
  preserved; no speculative architecture. See [`docs/architecture.md`](architecture.md)
  for the cross-phase roadmap index.

---

## 1. The reasoning chain this phase reinforces

```
Perception → Observations → Scene Semantics → Dynamic Context → Rule Reasoning → Evidence → Human Review
```

Phase 5 completes the **Scene Semantics → Observations** link for *metric* space:
lanes, stop lines, and zones become **world-referenced** through calibration, so a
speed observation is a physical quantity with an uncertainty, and the earlier
pixel-space gates (wrong-way motion, illegal-stopping stationarity) can be re-expressed
in metric terms **where a scene is calibrated**. Speeding is the sixth locked violation
and the one the master spec explicitly keeps **feasibility-gated**: a *conclusion*
derived from a calibrated trajectory with propagated uncertainty, never a raw
pixel-velocity threshold.

---

## 2. Relationship to Phases 0–4

Phases 0-F/1/2 froze the contracts and delivered wrong-way + illegal-stopping;
Phase 3 generalized the infrastructure, added dynamic context + red-light, and shipped
the observation-log substrate and event-level evaluation harness; Phase 4 implemented
association, quality-weighted confidence aggregation, triple riding, and no-helmet.
Phase 5 adds the **calibration/ground-plane** capability, then **speeding** (gated), and
**retro-upgrades** the two provisional pixel gates — closing the six-violation set.

Two facts from the current tree drive this phase:

- **The calibration schema already exists, unused.** `SceneConfig` carries a
  `CameraCalibration` (`homography` — an identity-matrix **placeholder**,
  `calibration_status`, `reprojection_error_px`, `scale_m_per_px`) and the scene hash
  covers it. No runtime projects pixels to world coordinates.
- **The provisional pixel gates are documented as provisional.** Illegal stopping's
  `motion_threshold` is "recorded for provenance but not applied (uncalibrated slice)";
  wrong-way uses a pixel-space heading test. Both are honest deferrals awaiting metric
  calibration — exactly what Phase 5 supplies. **These existing implementations remain
  valid**; Phase 5 *upgrades* them, it does not rewrite them.

---

## 3. Verified starting point (expected at Phase 5 entry)

Phase 5 begins only after Phase 4's Definition of Done holds. Expected state:

- Five violation slices run offline and deterministically on the generalized
  reasoner/pipeline bases; the observation log persists all observation variants; the
  evaluation harness scores all five rules.
- **`SpeedObservation` exists but has no derivation.** `contracts` defines
  `SpeedObservation` (`speed_kmph`, `speed_std_kmph`, `is_valid`, quality/plane fields
  per U2). `ViolationType.SPEEDING` and the scene `speeding` rule-parameter block
  (`speed_uncertainty_k`, `min_calibration_status`) exist.
- **No calibration runtime exists.** Homography application, pixel→world projection, and
  σ propagation are unimplemented; `calibration_status` is declared data only.
- **The evaluation protocol already gates speeding.** `docs/evaluation-protocol.md` §11
  (the speed feasibility gate) and §17 (provisional targets) define the conditions under
  which any speed claim is permissible; the architecture review keeps speeding
  feasibility-gated and out of penalty simulation until those conditions hold.
- **Sanctioned packages** are the Phase 4 set plus `evaluation`, `association`,
  `classifier`. Phase 5 adds **`calibration`** to the allow-list in
  `tests/docs/test_adr_pack.py`, in its introducing unit's commit.

---

## 4. Phase 5 objectives

- **A. Calibration / ground-plane runtime.** Implement homography application and
  pixel→world projection from the frozen `CameraCalibration` scene data, with an
  explicit **calibration-status gate** and per-point/per-track uncertainty propagation.
- **B. Speed observation derivation.** Derive `SpeedObservation` (`speed ± σ`) from
  world-referenced trajectories inside a measurement zone, honestly marking
  `is_valid=False` (abstain) when calibration status / geometry / track quality are
  insufficient.
- **C. Sixth violation: speeding (feasibility-gated).** A speeding reasoner using a
  robust per-track zone speed vs the posted limit with an uncertainty margin
  (`v − k·σ > limit`), confirming only above the gate and **excluded from any penalty
  simulation** until the §11 feasibility gate + §17 targets are met on a calibrated
  scene.
- **D. Retro-upgrade of provisional gates.** Where a scene is calibrated, re-express the
  illegal-stopping `motion_threshold` and wrong-way motion in **metric** terms, as an
  *upgrade path* that preserves the pixel-space behaviour for uncalibrated scenes and
  keeps all existing tests byte-identical for the uncalibrated case.
- **E. Speed-method validation (external gate).** A reproducible speed-accuracy
  validation against a metric reference (e.g. BrnoCompSpeed / a GNSS field session),
  separable and non-blocking to the offline reasoning path — the evidence that would
  lift the feasibility gate.
- **F. Offline + deterministic.** Calibration and speed derivation are deterministic
  pure functions of frozen inputs; the reasoning path replays bit-exactly from the
  observation log.

---

## 5. Architectural invariants (preserved)

1. **Perception ↔ reasoning separation.** Calibration/projection produce
   **observations** (`speed`), and metric-upgraded motion facts feed the existing
   observation derivations; rules consume only observations.
2. **Model-free deterministic replay.** Speed observations are derived by deterministic
   geometry and persisted to the observation log; replaying the log reproduces speeding
   decisions without a model.
3. **Frozen contracts.** `SpeedObservation`, `CameraCalibration`, `CalibrationStatus`,
   `SceneConfig`, `scene_config_hash`, and `ViolationType.SPEEDING` are unchanged. Phase
   5 expects **no** frozen-contract change (the calibration + speed fields already
   exist).
4. **≥ 2 observations to confirm; every non-confirmation is a countable abstention.**
   Preserved via the generalized base; insufficient calibration / geometry / track
   quality routes to abstention (`is_valid=False`), never a fabricated speed.
5. **Determinism.** No wall-clock; content-derived ids; deterministic ordering;
   projection + σ propagation are pure functions.
6. **Composition over deep inheritance.** The speeding reasoner is a configuration of the
   P3-U1 base; calibration is an **injected collaborator** of the speed derivation; the
   retro-upgrade injects a metric motion measure into the existing derivations rather
   than subclassing them.
7. **No speculative architecture.** A single `calibration` package appears in the unit
   that first needs it; the retro-upgrade adds no package.
8. **Honesty + feasibility-gating.** Speeding is excluded from penalty simulation until
   §11/§17 are satisfied on a calibrated scene; every speed carries σ; the scene hash
   binds calibration to each event; no accuracy claim before validation.

---

## 6. Migration notes (retro-upgrade unit P5-U4)

The retro-upgrade is an **upgrade path, not a rewrite**, and its safety rests on
preserving the uncalibrated behaviour exactly:

- **Existing implementations remain valid.** The wrong-way heading test and the
  illegal-stopping stationarity derivation keep their pixel-space behaviour as the
  default. The metric measure is applied **only when** `calibration_status` meets the
  scene's `min_calibration_status`; otherwise the pixel path is used unchanged.
- **Byte-identical for uncalibrated scenes.** All existing wrong-way / illegal-stopping
  tests (which use uncalibrated example scenes) must pass **unchanged**, with identical
  events and persisted files. Per the task constraints, no existing test is edited; a
  forced edit signals a behaviour change → **stop and report**.
- **`motion_threshold` becomes live where calibrated.** The illegal-stopping
  `motion_threshold` — currently "recorded but not applied" — becomes an applied
  **metric** threshold on calibrated scenes, closing a documented deferral without
  changing the uncalibrated slice.
- **Composition.** The metric measure is an injected strategy on the existing
  derivations, selected by calibration status — not a subclass of the reasoner.

---

## 7. Ordered unit cards

Dependency order: **P5-U1 → P5-U2 → P5-U3**, with **P5-U4** (retro-upgrade) after
P5-U1, and **P5-U5** (speed-method validation) as a separable external gate.

### P5-U1 — Calibration / ground-plane runtime

- **Objective.** Implement homography application and pixel→world projection from the
  frozen `CameraCalibration`, with a calibration-status gate and uncertainty
  propagation, as deterministic pure functions.
- **Why now.** Every metric capability (speed derivation, metric motion) depends on
  projection; it is the enabling primitive of the phase and reads only frozen scene
  data.
- **Inputs.** `CameraCalibration` (`homography`, `calibration_status`,
  `reprojection_error_px`, `scale_m_per_px`); `CalibrationStatus`; `geometry`
  primitives.
- **Exact scope.** A new `calibration/` package (e.g. `calibration/ground_plane.py`)
  providing pixel→world projection via the scene homography, a status gate
  (`is_metric_ready(scene, min_status)`), and per-point uncertainty propagation from
  `reprojection_error_px` / `scale_m_per_px`. Identity-placeholder homographies resolve
  to `calibration_status = uncalibrated` → not metric-ready (honest). Add `calibration`
  to the allow-list in this commit.
- **Outputs.** Deterministic projection + a calibration-readiness gate + σ propagation.
- **Acceptance criteria.** Projection is deterministic and matches hand-computed values
  on a known homography; the identity placeholder is reported not-metric-ready;
  uncertainty propagates monotonically with `reprojection_error_px`; no contract change.
- **Required tests.** Projection on a known homography; identity → not-ready; status
  gate; σ propagation; determinism. (`tests/calibration/test_ground_plane.py`.)
- **Explicit exclusions.** No homography *estimation/annotation tool* (calibration data
  is authored input); no speed derivation; no reasoning.
- **Stop condition.** Stop when projection + gate + σ propagation are complete and
  tested; do not start speed derivation.

### P5-U2 — Speed observation derivation

- **Objective.** Derive `SpeedObservation` (`speed ± σ`) from world-referenced
  trajectories inside a measurement zone, abstaining (`is_valid=False`) when calibration
  / geometry / track quality is insufficient.
- **Why now.** Projection (P5-U1) exists; speed is the observation the speeding reasoner
  consumes.
- **Inputs.** Ordered `TrackState`s; P5-U1 projection + gate; scene measurement-zone
  geometry; `SpeedObservation`; PTS media time (fixed-epoch anchor).
- **Exact scope.** A derivation in `observations/` (e.g. `observations/speed.py`)
  projecting a track's ground-plane path within the measurement zone, computing a robust
  speed with propagated σ (from projection uncertainty + temporal sampling), and
  emitting `SpeedObservation` with `is_valid` reflecting the calibration gate + minimum
  path-length / quality. Taint handling reused; sub-gate / short-path / uncalibrated
  cases emit `is_valid=False` (abstain), never a fabricated number.
- **Outputs.** Deterministic per-track speed observations with honest validity + σ.
- **Acceptance criteria.** A clean calibrated track yields `speed ± σ` matching a
  hand-computed value within σ; an uncalibrated scene / short path / tainted track yields
  `is_valid=False`; σ grows with reprojection error and shorter baselines; deterministic;
  no contract change.
- **Required tests.** Speed on a known calibrated trajectory; uncalibrated abstain;
  short-path abstain; σ behaviour; taint; determinism. (`tests/observations/test_speed.py`.)
- **Explicit exclusions.** No reasoning; no limit comparison; no penalty; no contract
  change.
- **Stop condition.** Stop when speed observations derive deterministically with honest
  validity; do not start the reasoner.

### P5-U3 — Speeding reasoner (feasibility-gated) + pipeline + e2e

- **Objective.** A speeding reasoner confirming `SPEEDING` when a robust per-track zone
  speed exceeds the posted limit by an uncertainty margin (`v − k·σ > limit`),
  **excluded from any penalty simulation** until the §11 feasibility gate + §17 targets
  are met, composed on the P3-U1 base + P3-U2 pipeline, persisted, and verified end to
  end on a recorded calibrated synthetic clip.
- **Why now.** Calibration (P5-U1) and speed observations (P5-U2) are the prerequisites;
  speeding completes the six-violation set.
- **Inputs.** P5-U2 speed observations; the scene `speeding` params (`speed_uncertainty_k`,
  `min_calibration_status`) + posted limit; the P3-U1 base; P3-U2 pipeline base;
  `EventStore`.
- **Exact scope.** `rules/speeding.py`: a loader requiring a calibrated measurement zone
  + posted limit (fail-fast otherwise); a robust per-track zone-speed aggregate (e.g.
  median of valid observations) with the uncertainty-margin rule `v − k·σ > limit`;
  confirmation via the P3-U1 base with the ≥2-valid-observation floor; content-derived
  `event_id`; run-level `models`; an explicit **`penalty_eligible=False` / simulation-
  excluded** marker on every speeding event until the gate lifts. A thin P3-U2 pipeline
  configuration; persistence; a recorded calibrated-clip e2e (injected detections +
  authored homography). Speeding events are visibly labelled feasibility-gated in the
  run report.
- **Outputs.** Confirmed, uncertainty-margined speeding events (persisted, penalty-
  excluded) on calibrated scenes; abstentions on uncalibrated / low-confidence cases.
- **Acceptance criteria.** `v − k·σ > limit` on a calibrated scene confirms one event
  marked simulation-excluded; a within-margin or uncalibrated case abstains (countable);
  ≥2-valid-observation floor; the scene hash binds calibration to the event; deterministic
  + byte-identical persistence; recorded-clip e2e passes; **no penalty simulation** is
  emitted; no contract change.
- **Required tests.** Confirm above margin (calibrated); abstain within margin; abstain
  uncalibrated; k·σ margin behaviour; penalty-excluded marker; ≥2-valid floor; taint;
  determinism; recorded-clip e2e. (`tests/rules/test_speeding.py`,
  `tests/pipeline/test_speeding_e2e.py`.)
- **Explicit exclusions.** No penalty simulation; no accuracy claim; no night/adverse
  operating claim; no contract change; no new package.
- **Stop condition.** Stop when speeding confirms/abstains deterministically end to end on
  a calibrated recorded clip, penalty-excluded; do not lift the feasibility gate here
  (that needs P5-U5 evidence).

### P5-U4 — Retro-upgrade of provisional pixel gates (calibrated scenes)

- **Objective.** Where a scene is calibrated, re-express the illegal-stopping
  `motion_threshold` and wrong-way motion in **metric** terms as an upgrade path,
  preserving pixel-space behaviour (byte-identical) for uncalibrated scenes.
- **Why now.** Calibration (P5-U1) now exists; the two documented "provisional / recorded
  but not applied" gates can be honestly upgraded without regressing the uncalibrated
  slices.
- **Inputs.** P5-U1 projection + gate; the existing wrong-way + illegal-stopping
  derivations; the scene `min_calibration_status` / `motion_threshold` params.
- **Exact scope.** Inject a **metric** motion/stationarity measure (selected by
  calibration status) into the existing derivations: on calibrated scenes the illegal-
  stopping `motion_threshold` becomes an applied metric threshold and wrong-way motion is
  measured in world units; on uncalibrated scenes the existing pixel behaviour is used
  unchanged. Composition (injected strategy), not a rewrite.
- **Outputs.** Metric-aware motion/stationarity on calibrated scenes; unchanged pixel
  behaviour otherwise.
- **Acceptance criteria.** On uncalibrated example scenes, all existing wrong-way /
  illegal-stopping tests pass **unchanged** (byte-identical events + files); on a
  calibrated scene the metric `motion_threshold` is applied and documented as live;
  deterministic; no contract change; no existing-test edits.
- **Required tests.** New tests for the calibrated metric path (applied `motion_threshold`,
  metric wrong-way motion); a regression assertion that the uncalibrated path is
  byte-identical. (`tests/observations/test_metric_motion_upgrade.py`.)
- **Explicit exclusions.** No change to uncalibrated behaviour; no contract change; no
  new package; no existing-test edits.
- **Stop condition.** Stop when the metric upgrade is live on calibrated scenes with the
  uncalibrated path provably unchanged.

### P5-U5 — Speed-method validation (external gate)

- **Objective.** A reproducible speed-accuracy validation against a metric reference
  (BrnoCompSpeed-style dataset or a GNSS-instrumented field session) — the evidence that
  would satisfy the §11 feasibility gate and permit lifting the speeding penalty
  exclusion.
- **Why now / positioning.** This is the external, dataset/field-gated unit; it is
  separable from and non-blocking to the offline speeding reasoning path (P5-U1…U4),
  mirroring the real-footage and dataset gates of earlier phases.
- **Inputs.** The §11 feasibility gate + §17 provisional targets; a registered metric-
  reference dataset (licence/access **resolved** first) or an approved GNSS field
  session; the P5-U2 speed derivation; the P3-U7 evaluation harness (extended with the
  §23 speed protocol).
- **Exact scope.** A reproducible speed-validation harness (config in, results JSON out,
  git-tagged) comparing derived `speed ± σ` against the metric reference on a **calibrated**
  scene, reporting error distribution + coverage of the σ intervals against the §17
  targets, and a recorded decision on whether the §11 gate is met. Extend `evaluation/`
  with the §23 speed protocol as needed.
- **Outputs.** A reproducible speed-validation result + an explicit, evidence-backed
  gate decision (met / not-met).
- **Acceptance criteria.** The harness reproduces from committed configs; error + σ-
  coverage are reported against §17 targets; **no dataset is downloaded before its
  registry gate resolves**; the penalty exclusion is lifted **only** if the recorded
  evidence meets §11 — otherwise speeding stays penalty-excluded with the reason recorded;
  default CI/tests pass without the dataset.
- **Required tests.** Validation-harness unit tests on synthetic reference fixtures
  (error + coverage computations, determinism); opt-in real-data run skipped by default.
  (`tests/evaluation/test_speed_validation.py`.)
- **Explicit exclusions.** No dataset download without a resolved gate; no penalty
  simulation even if the gate is met (penalty workflow is a separate backlog capability);
  no accuracy claim beyond the reported protocol.
- **Stop condition.** Stop when speed validation is reproducibly executed and the gate
  decision is recorded; if the reference is unavailable, record "speeding reasoning
  complete; feasibility gate pending metric-reference validation" — do not lift the gate
  without evidence.

---

## 8. Dependency graph

```
P5-U1 (calibration/ground-plane) ──┬─> P5-U2 (speed observation) ──> P5-U3 (speeding reasoner+e2e, penalty-excluded)
                                    └─> P5-U4 (retro-upgrade of pixel gates; calibrated scenes)

External / parallel gate (blocks only lifting the penalty exclusion, not P5-U1…U4):
  P5-U5 speed-method validation — metric-reference dataset / GNSS field session
  Real-footage validation        — carried forward from earlier phases
```

## 9. Implementation ordering

1. **P5-U1** — calibration/ground-plane first (enabling primitive).
2. **P5-U2** — speed observation from world-referenced trajectories.
3. **P5-U3** — speeding reasoner, feasibility-gated and penalty-excluded.
4. **P5-U4** — retro-upgrade of the provisional pixel gates (after P5-U1; parallelizable
   with P5-U2/U3).
5. **P5-U5** — external speed-method validation; the only path to lifting the gate.

## 10. Phase 5 Definition of Done

- P5-U1…U4 complete; each unit's acceptance criteria and required tests pass. P5-U5 is
  complete **or** honestly recorded as pending the metric-reference gate (reasoning
  complete; gate pending).
- Calibration projects pixels to world coordinates deterministically with a status gate
  and σ propagation; the identity placeholder is honestly not-metric-ready.
- Speed observations derive with honest validity + σ; speeding runs end to end offline on
  a recorded **calibrated** synthetic clip, deterministically and byte-identically on
  replay, with every speeding event marked **penalty-excluded / feasibility-gated**.
- The provisional pixel gates are metric-upgraded on calibrated scenes with the
  uncalibrated slices **byte-identical** (no existing test edited).
- The six locked violations are all implemented as offline, deterministic,
  observation-driven reasoning slices.
- Quality gates green: `ruff`, `mypy src`, full `pytest -q` (opt-in real-model/real-data
  tests skipped).
- **No frozen contract, schema, ADR, or master-spec change.** `calibration` added to the
  allow-list.

## 11. Claims allowed after Phase 5

- "TrafficPulse implements all six locked violations — wrong-way, illegal-stopping,
  red-light, triple-riding, no-helmet, and speeding — as offline, deterministic,
  observation-driven reasoning slices."
- "Speeding is derived from calibrated ground-plane trajectories with propagated
  uncertainty (`v − k·σ > limit`) and is **feasibility-gated and excluded from penalty
  simulation** until a metric-reference validation meets the evaluation-protocol gate."
- "On calibrated scenes, the illegal-stopping and wrong-way motion gates are expressed in
  metric terms, with pixel-space behaviour preserved unchanged on uncalibrated scenes."
- If P5-U5 meets the gate: "Speed accuracy was validated against a metric reference per
  evaluation-protocol §11/§17."

## 12. Claims still forbidden after Phase 5

- No real-world / event-level accuracy claim on real footage (external gate).
- No speeding penalty-eligibility or enforcement claim (penalty simulation is a separate
  backlog capability, and the gate lifts only on P5-U5 evidence).
- No calibrated-probability confidence claim (components only).
- No night / adverse-condition operating claim (robustness-analysis slice only).
- No production/enforcement-readiness claim; a confirmed event is not a legal
  determination.

## 13. Handoff criteria (to post-Phase-5 backlog)

Phase 5 closes the six-violation reasoning set. The remaining, out-of-scope capabilities
(explicitly **not** planned here) are:

- **Full evidence-engine runtime** — clip/frame rendering, crops, overlays,
  content-addressed media hashing, OCR/ANPR (the manifest is still a stub).
- **Durable storage** — SQLite runtime + Parquet event logs (ADR-002 defers; ADR-004
  stays Proposed until reprocessing/event-identity semantics are required).
- **Human-review UI/workflow and simulated-penalty workflow** — the last two links of the
  reasoning chain; the evidence and confidence breakdowns built in Phases 3–5 are their
  inputs.
- **Real-footage validation and privacy/redaction** — external, gated activities.

These are named for continuity, not scheduled; a future phase plan would govern them.

## 14. Stop conditions (phase-level)

- Stop at the end of P5-U4 (with P5-U5 done or honestly pending) and gates green.
- **Stop and report** if any unit discovers a genuine need for a frozen-contract, schema,
  ADR, or master-spec change (none is anticipated — calibration + speed fields already
  exist).
- Do not implement penalty simulation, the human-review UI, the evidence-media engine,
  ANPR, or the SQLite runtime in this phase.
- Do not lift the speeding penalty exclusion without recorded P5-U5 metric-reference
  evidence meeting evaluation-protocol §11.
- Do not download any dataset before its registry licence/access gate is resolved; do not
  let dataset/field validation gate completion of P5-U1…U4.
