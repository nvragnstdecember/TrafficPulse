# TrafficPulse — Phase 3 plan

**Phase name:** Phase 3 — Generalized Reasoning, Dynamic Traffic Context, and Red-Light Jumping

- **Status:** Authoritative Phase 3 unit plan (planning document; no Phase 3 source
  implemented yet).
- **Date:** 2026-07-12
- **Authority:** This is the **authoritative Phase 3 unit plan**. It governs the
  `P3-U#` identifier namespace only. It does **not** supersede
  [`docs/phase-0-plan.md`](phase-0-plan.md) (Phase 0-F, `U#`),
  [`docs/phase-1-plan.md`](phase-1-plan.md) (Phase 1, `P1-U#`), or
  [`docs/phase-2-plan.md`](phase-2-plan.md) (Phase 2, `P2-U#`); those remain
  governed by their own documents and their histories are not rewritten here. The
  four phases use **separate identifier namespaces** (`U#` vs `P1-U#` vs `P2-U#` vs
  `P3-U#`).
- **Canonical architecture reference:** [`docs/architecture-review.md`](architecture-review.md)
  remains THE canonical architecture reference; this plan interprets and sequences
  it for Phase 3 without modifying it or the master spec
  ([`TRAFFICPULSE_MASTER_SPEC.md`](../TRAFFICPULSE_MASTER_SPEC.md)).
- **Basis:** This plan implements the **accepted architectural design review**
  (2026-07-12): capability-first sequencing; generalized reasoning + pipeline
  infrastructure by **composition, not deep inheritance**; a **dynamic traffic
  context** stream; the **evaluation harness** and **observation-log substrate** as
  first-class deliverables; no speculative architecture and no monolithic
  "TrafficSemantics" engine. See [`docs/architecture.md`](architecture.md) for the
  cross-phase roadmap index.

---

## 1. The reasoning chain this phase reinforces

TrafficPulse is **not** an object detector. A violation is a *conclusion* derived
along an explicit chain, and every Phase 3 unit is placed on exactly one link of it:

```
Perception → Observations → Scene Semantics → Dynamic Context → Rule Reasoning → Evidence → Human Review
```

- **Perception** (frozen; Phase 1/2): detector + tracker behind the `Detection` /
  `TrackState` seams.
- **Observations** (extended here): typed per-frame facts; Phase 3 adds a
  scene-level **signal-state** observation and a **stop-line/junction-crossing**
  derivation, both from **existing** contracts.
- **Scene Semantics** (frozen; U5): lanes, legal directions, stop lines, junction /
  signal-controlled zones, signal groups with `permitted_movements` — all already
  declared in `SceneConfig`.
- **Dynamic Context** (new concept, introduced here): a **scene/region-indexed,
  time-varying** stream (signal state now; congestion later) that rules consume
  *alongside* per-track observations. This is the load-bearing new idea of Phase 3.
- **Rule Reasoning** (generalized here): the two shipped reasoners are refactored
  onto a shared, composed temporal-run base; red-light is the first rule to join a
  **context stream × a per-track stream**.
- **Evidence** (hardened here): the **observation-log substrate** finally persists
  the log that §15 deterministic replay depends on; the **evaluation harness**
  makes confirmed events measurable.
- **Human Review** (unchanged; still backlog): out of scope for Phase 3, but the
  evidence produced here is precisely what a later review workflow consumes.

---

## 2. Relationship to Phases 0–2

Phase 0-F (`U1…U6`) froze the contracts, ontology, registry/policy, evaluation
protocol, scene schema + `scene_config_hash`, and the ADR pack. Phase 1
(`P1-U1…P1-U12`) delivered the **wrong-way** slice end to end. Phase 2
(`P2-U1…P2-U7`) added **model-provenance propagation** and the **illegal-stopping**
slice. All of these are **frozen interfaces** and Phase 3 builds against them
unchanged.

Phase 3 is the **third violation** (red-light jumping) plus the **generalization
and evidence-substrate work** that the accepted design review identified as
prerequisites for every remaining violation. It adds **no new perception model**:
signal state is sourced from a declared/simulated signal log (offline-honest), not
from a learned classifier (that seam is a documented later concern).

---

## 3. Verified starting point

Verified at planning time against `main` (Phase 2 complete; `ruff`, `mypy src`
strict, and the full `pytest -q` suite green — **946 passed**, 4 opt-in real-model
tests skipped):

- **Two structurally identical violation slices exist.** `WrongWayReasoner`
  (`rules/wrong_way.py`) and `IllegalStoppingReasoner` (`rules/illegal_stopping.py`)
  are explicit **twins** (the latter's docstring: "structurally this reasoner is the
  wrong-way reasoner's twin"): per-`(camera, track)` run tracking, `_on_recovery`
  (close/abandon), threshold-elapsed confirmation, taint-restart reset,
  content-derived `event_id`, run-level `models` stamping, and a `_confirm` that
  mints a `ConfirmedEvent` with `measurements` / `thresholds`. `WrongWayPipeline`
  and `IllegalStoppingPipeline` are likewise twins sharing an identical
  detect → track → group → provenance-collect front half; only `finalize`'s
  derivation + reasoner back half differs.
- **The join pattern already exists.** `join_stopped_in_zone`
  (`rules/illegal_stopping.py`) joins two **per-track** observation streams on
  `(camera_id, track_id, timestamp)`. No reasoner yet joins a **scene-level**
  stream.
- **The red-light geometry already exists.** `geometry/segments.py` implements
  `stop_line_crossing`, `segments_intersect`, `side_of_line`, and `CrossingFact`
  (side-of-line + finite-segment intersection). No derivation consumes them yet.
- **The red-light contracts already exist.** `SignalStateObservation` (scene-level:
  `track_id` optional, carries `signal_state` + `roi_id`) and `InZoneObservation`
  are frozen U2 variants. `ViolationType.RED_LIGHT_JUMPING` exists.
- **The red-light scene semantics already exist.** `configs/scenes/example-scene.yaml`
  declares `zone-junction` (`intersection`), `zone-signal-ctrl`
  (`signal_controlled_region`, `observation_consumers: [in_zone, signal_state]`),
  `stopline-001` (`crossing_direction`, `signal_group_id`), and `sg-001`
  (`roi`, `permitted_movements: [free_left]`, `expected_states`). The
  `red_light_jumping` rule-parameter block already carries
  `amber_onset_grace: 0.3 (provisional)` and `crossing_dedup_window (unset)`.
- **No observation log is persisted.** ADR-002's Parquet observation-log substrate
  has not been built; replay today is per-run event JSON only. The §15
  "deterministic replay from the observation log" claim is architecturally intended
  but **not yet materially backed**.
- **No evaluation harness exists.** The seven §23 protocols are documented in
  `docs/evaluation-protocol.md` but unimplemented; no accuracy claim is permissible
  for any violation.
- **Sanctioned packages** under `src/trafficpulse/` are exactly `contracts`,
  `geometry`, `synth`, `rules`, `observations`, `ingestion`, `detector`,
  `tracking`, `pipeline`, `persistence` (enforced by
  `tests/docs/test_adr_pack.py`). Phase 3 adds **one** new package (`evaluation`),
  in the implementing unit's commit.

---

## 4. Phase 3 objectives

- **A. Generalized reasoning infrastructure (composition).** Extract the shared
  temporal-run machinery of the two twin reasoners into a **composed** base the
  violation reasoners *delegate to* — parameterized by an injected per-step
  predicate and a confirmation policy — with **byte-identical** behaviour for
  wrong-way and illegal-stopping (pinned by existing determinism/event-id tests).
- **B. Generalized pipeline infrastructure (composition).** Extract the shared
  detect → track → group → provenance front half into a composed base pipeline that
  takes an injected finalize strategy; wrong-way and illegal-stopping become thin
  configurations, behaviour unchanged.
- **C. Dynamic traffic context.** Introduce a scene-level, time-indexed context
  stream (signal state, from a declared/simulated log) and generalize the
  observation **join** to pair a per-track carrier with both per-track streams and
  a scene-level context stream — model-free, deterministic.
- **D. Third violation: red-light jumping.** Add an offline, deterministic
  red-light capability built from stop-line/junction-entry crossing (per-track) ×
  signal-state (context), with amber-onset grace, `permitted_movements` exclusion,
  crossing dedup, taint abstention, confirmed-event persistence, and
  evidence-manifest linkage — reusing the Phase 1/2 architecture, **frozen
  contracts**, and existing geometry, adding **no new perception model**.
- **E. Observation-log substrate (first-class).** Persist the derived observation
  streams (ADR-002 Parquet) so the §15 deterministic-replay defensibility claim is
  materially backed and the evaluation harness has a stable substrate.
- **F. Evaluation harness (first-class).** Implement the §23-E event-level
  evaluation protocol (event matching, per-rule P/R/F1, false-events/hour, duplicate
  rate, detection delay, evidence completeness) against the deterministic
  `EventStore`, so wrong-way, illegal-stopping, and red-light become *measurable*
  the moment ground-truth manifests exist.
- **G. Offline + deterministic.** All Phase 3 implementation is offline, uses no
  wall-clock in the decision path, and replays bit-exactly.

---

## 4a. Design decision — red-light preserves frozen contracts

Red-light jumping is modelled **without adding an observation contract**. The
crossing event is carried by the **existing** `InZoneObservation` for the junction /
signal-controlled zone (the `is_inside` False→True transition is the entry), with
`stop_line_crossing` geometry used *inside the derivation* to validate that the
entry is a forward crossing in the configured `crossing_direction` (reducing false
positives from reversing / boundary jitter). Signal state is carried by the
**existing** scene-level `SignalStateObservation`. No `line_crossing` observation
variant is introduced.

**If — and only if — the in-zone-transition model provably cannot express the
required crossing semantics for the first slice**, a dedicated crossing observation
would be the one justified frozen-contract addition; per §20 it must then be
**escalated ("stop and report") as an explicit contract decision**, never added
silently. The plan's expectation, from the code audit, is that no contract change is
required.

---

## 5. Architectural invariants (preserved)

1. **Perception ↔ reasoning separation.** Rules consume **only** `Observation`
   contracts; observation/derivation code consumes only frozen `TrackState` /
   geometry / scene data. No rule sees a detector/tracker framework type.
2. **Model-free deterministic replay** (architecture-review §15). The reasoning
   layer replays bit-exactly from observations with no model/GPU. The new
   context stream and join produce per-step facts from frozen observation data only;
   the observation-log substrate (Obj. E) makes this replay *materially* reproducible
   from persisted logs.
3. **Frozen contracts.** `Detection`, `TrackState`, all `Observation` variants,
   `ConfirmedEvent`, `EvidenceManifest`, `ModelRef`, `SignalState`, `ZoneKind`,
   `SceneConfig`, and `scene_config_hash` are unchanged. Every field/enum Phase 3
   uses already exists (§3).
4. **≥ 2 observations to confirm; every non-confirmation is a countable
   abstention** (architecture-review §13). The generalized reasoner base preserves
   this guarantee for all three violations.
5. **Determinism.** No wall-clock in decisions; content-derived ids; timestamp-
   ordered processing; deterministic output ordering; media time from PTS via the
   fixed-epoch anchor.
6. **Composition over deep inheritance.** The generalized reasoner/pipeline bases are
   **collaborators that violation logic delegates to**, parameterized by injected
   predicates/policies/strategies — not superclasses in a deep hierarchy. A new
   violation is a *configuration*, not a subclass override cascade.
7. **No speculative architecture.** Exactly one new package (`evaluation`) appears,
   in the unit that first implements it. Signal-state derivation lands in the
   existing `observations` package and the generalized join in the existing `rules`
   package; a dedicated `context` package appears **only if** a second dynamic-context
   type (e.g. congestion flow-state) later justifies it — not now.
8. **Honesty.** Signal state is a declared/simulated log input, not a learned
   classifier output; no accuracy claim; no claim that a real detector produced an
   event on synthetic pixels; congestion suppression remains a documented deferral.

---

## 6. Migration notes (generalization units P3-U1, P3-U2)

The two generalization units are **behaviour-preserving refactors**, and their
safety rests on the existing suite, not on new behaviour:

- **Existing implementations remain valid.** `WrongWayReasoner`,
  `IllegalStoppingReasoner`, `WrongWayPipeline`, and `IllegalStoppingPipeline` keep
  their public constructors, method names, and return types. Internally they
  delegate to the composed base; externally nothing changes.
- **Byte-identical output is the acceptance bar.** The already-green determinism and
  event-id tests (fresh-instance and reset-replay equality; direct-vs-pipeline
  equivalence) must pass **unchanged** — including identical `event_id`s and
  byte-identical persisted files. The `event_id` preimage scheme (scene hash,
  camera, violation, rule, track ids, start/trigger, hypothesis id) is preserved
  exactly; the run-level `models` stamping and its exclusion from `event_id` are
  preserved.
- **No test edits.** Per the task constraints these units add tests for the base but
  do **not** modify existing tests; if a refactor would force an existing-test edit,
  that is a signal the refactor changed behaviour — **stop and report** rather than
  edit the test.
- **Sequencing.** P3-U1/U2 land **before** the red-light reasoner/pipeline so that
  red-light is authored as a configuration of the generalized base from the start,
  not migrated afterward.

---

## 7. Ordered unit cards

Dependency order: **P3-U1 → P3-U2 → P3-U3 → P3-U4 → P3-U5**, with **P3-U6** and
**P3-U7** as first-class deliverables runnable in parallel once the reasoner/pipeline
shape is settled (they exercise the existing wrong-way / illegal-stopping paths and
do not depend on red-light).

### P3-U1 — Generalized temporal-run reasoner base (composition)

- **Objective.** Extract the shared per-`(camera, track)` temporal-run machinery of
  the two twin reasoners into a composed base that violation reasoners delegate to,
  parameterized by (a) an injected **per-step predicate** producing the run signal
  from a carrier observation and (b) a **confirmation policy** (sustained-duration
  *or* trigger-with-min-observations + dedup), leaving `WrongWayReasoner` /
  `IllegalStoppingReasoner` behaviour byte-identical.
- **Why now.** Two explicit twins exist; a third and fourth reasoner (red-light here,
  helmet/triple/speeding later) would multiply the copy-paste. Generalizing first,
  while behaviour is pinned by existing tests, is the low-risk moment.
- **Inputs.** `rules/engine.py` (`RuleEngine`), `rules/states.py`, the two existing
  reasoners, `ConfirmedEvent` / `MeasuredValue` / `ModelRef`.
- **Exact scope.** A new module in `rules/` (e.g. `rules/temporal.py`) providing a
  composed run-machine: per-track run lifecycle (open/attach/activate/confirm/close/
  abandon via `RuleEngine`), taint-restart reset, run-level `models` stamping, and
  content-derived `event_id` (identical preimage scheme). It accepts an injected
  predicate and a `ConfirmationPolicy`. Refactor both reasoners to delegate to it;
  their public APIs are unchanged. Two confirmation policies are provided:
  **sustained-duration** (wrong-way `min_persistence`, illegal-stopping
  `stationary_duration`) and **trigger-with-dedup** (for red-light in P3-U5).
- **Outputs.** A composed reasoner base; wrong-way and illegal-stopping reasoners as
  thin configurations of it.
- **Acceptance criteria.** All existing wrong-way / illegal-stopping tests pass
  **unchanged**; event ids, event sets, timings, and persisted files are
  byte-identical; the base carries no violation-specific knowledge (no `wrong_way` /
  `illegal_stopping` / `red_light` literals); composition, not a deep subclass
  hierarchy (the violation reasoners **hold** the base, they do not inherit a chain).
- **Required tests.** Base-level unit tests for the run lifecycle + both confirmation
  policies + taint reset + event-id determinism; an equivalence assertion that the
  refactored reasoners produce identical output to the pre-refactor golden fixtures.
  (Unique basename, e.g. `tests/rules/test_temporal_base.py`.)
- **Explicit exclusions.** No behaviour change; no contract change; no new package; no
  existing-test edits; no red-light logic yet.
- **Stop condition.** Stop when both reasoners delegate to the base with byte-identical
  output; do not start pipeline generalization here.

### P3-U2 — Generalized composition-pipeline base (composition)

- **Objective.** Extract the shared detect → track → group-by-`(camera, track)` →
  provenance-collect front half of the two twin pipelines into a composed base that
  takes an injected **finalize strategy** (per-track derivation + reasoner), leaving
  `WrongWayPipeline` / `IllegalStoppingPipeline` behaviour byte-identical.
- **Why now.** Same twin-duplication argument as P3-U1, at the orchestration layer;
  red-light's pipeline should be a configuration of the base from the start.
- **Inputs.** `pipeline/wrong_way.py`, `pipeline/illegal_stopping.py`,
  `pipeline/provenance.py`, the `Detector` / `Tracker` seams, `frame_record_to_frame`.
- **Exact scope.** A new module in `pipeline/` (e.g. `pipeline/base.py`) owning
  `process_frame` (detect + adapt + track + accumulate history + collect run-level
  `ModelRef`s), `reset`, `process`, and a `finalize` that delegates to an injected
  strategy computing events from the accumulated per-track history. Refactor both
  pipelines to inject their derivation+reasoner back half; public APIs unchanged. The
  library core still imports **no** backend (boundary test preserved).
- **Outputs.** A composed pipeline base; both pipelines as thin configurations.
- **Acceptance criteria.** All existing pipeline tests (equivalence, determinism,
  reset-replay, backend-free import boundary) pass **unchanged**; byte-identical
  events and persisted files; the base carries no violation-specific derivation.
- **Required tests.** Base-level tests for the front half + strategy delegation +
  determinism; equivalence to pre-refactor golden output.
  (`tests/pipeline/test_pipeline_base.py`.)
- **Explicit exclusions.** No behaviour change; no contract change; no new package; no
  existing-test edits; no generic multi-rule *runner* (a base the pipelines compose
  with, not a config-driven mega-runner).
- **Stop condition.** Stop when both pipelines compose with the base at byte-identical
  output; do not start dynamic-context work here.

### P3-U3 — Dynamic traffic context stream + generalized observation join

- **Objective.** Introduce a scene-level, time-indexed **signal-state context**
  derivation (from a declared/simulated signal log) and generalize the two-stream
  join into a deterministic join that pairs a per-track carrier stream with both
  additional per-track streams and a **scene-level context** stream keyed on
  `(camera_id, timestamp)` (`track_id = None`).
- **Why now.** Red-light is the first violation whose evidence combines a scene-level
  context (signal state) with a per-track fact (junction entry); this is the "dynamic
  traffic context" the design review requires, and the generalization is reused by
  any future context-consuming rule.
- **Inputs.** `SignalStateObservation`, `SignalGroup` / `SignalState` scene data,
  `SignalSourceMode` (`simulated_schedule` / `manual_annotation` already exist), the
  existing `join_stopped_in_zone`, `Producer` / `ProducerKind`.
- **Exact scope.** (a) A signal-state derivation in `observations/` (e.g.
  `observations/signal.py`) that emits scene-level `SignalStateObservation`s from a
  **declared signal schedule/log** for a `signal_group` — no learned classifier (the
  ROI-classifier seam is a documented later concern, mirroring the detector seam
  pattern). (b) A generalized join helper in `rules/` (e.g. `rules/joins.py`) that
  generalizes `join_stopped_in_zone`: a per-track carrier joined with (i) other
  per-track streams on `(camera, track, timestamp)` and (ii) a scene-level context
  stream on `(camera, timestamp)`, folding conservatively (a missing context/side
  never fabricates evidence), unioning taint restarts, deterministic ordering. The
  existing `join_stopped_in_zone` is re-expressed in terms of it (behaviour
  preserved) or left intact and the generalized helper added alongside — whichever
  keeps the illegal-stopping tests byte-identical.
- **Outputs.** A signal-state context derivation + a generalized, reusable join.
- **Acceptance criteria.** Signal-state observations are deterministic and scene-level
  (`track_id=None`); the generalized join reproduces `join_stopped_in_zone`'s results
  exactly on the illegal-stopping fixtures; a scene-level context correctly pairs to
  every per-track carrier at the same timestamp; missing context folds to a safe
  "no evidence" (never fabricated); fully deterministic.
- **Required tests.** Signal-log → observation derivation; generalized-join
  equivalence to the existing two-stream join; context×track pairing; conservative
  missing-side fold; taint-restart union; determinism.
  (`tests/observations/test_signal_state.py`, `tests/rules/test_joins.py`.)
- **Explicit exclusions.** No learned signal classifier; no red-light reasoning yet;
  no congestion context; no contract change; no new package.
- **Stop condition.** Stop when signal-state context + the generalized join are
  complete and tested; do not start crossing derivation.

### P3-U4 — Stop-line / junction-entry crossing derivation

- **Objective.** Deterministically derive the per-track **crossing** signal — a track
  crossing the configured stop line in the legal `crossing_direction` into the
  junction / signal-controlled zone — using existing geometry and the **existing**
  `InZoneObservation`, with **no new observation contract**.
- **Why now.** The crossing is the per-track half of red-light's evidence; it is pure
  geometry over frozen inputs and unblocks P3-U5.
- **Inputs.** Ordered `TrackState`s; `SceneConfig` `stop_lines` (`endpoints`,
  `crossing_direction`, `signal_group_id`, `zone_ids`) and junction /
  signal-controlled zones; `geometry.stop_line_crossing` / `CrossingFact`;
  `geometry.point_in_polygon`; `InZoneObservation`; the taint pattern from
  `observations/heading.py`.
- **Exact scope.** A derivation in `observations/` (e.g. `observations/crossing.py`)
  that, per usable step, emits an `InZoneObservation` for the configured junction /
  signal-controlled zone (bottom-center membership, reusing the P2-U2 pattern), while
  using `stop_line_crossing` to validate that entry coincides with a **forward**
  stop-line crossing (side change in `crossing_direction`, finite-segment
  intersection) so reversing / boundary jitter does not read as an entry. Reuse the
  heading/zone taint handling verbatim (skip tainted steps; mark taint restarts). The
  derivation makes **no** legality/signal/temporal decision (that is P3-U5).
- **Outputs.** A derivation result (in-zone/junction-entry observations + taint
  restarts) mirroring the existing derivations.
- **Acceptance criteria.** A forward crossing into the junction under the legal
  direction yields the entry observation; a reversing / non-crossing movement does
  not; boundary/edge policy matches the frozen `point_in_polygon` semantics; tainted
  steps skipped + next clean observation flagged restart; deterministic; **no contract
  added**.
- **Required tests.** Forward-crossing entry; reversing/no-cross rejection;
  direction-gating via `crossing_direction`; taint skip + restart; empty/short track;
  determinism. (`tests/observations/test_crossing.py`.)
- **Explicit exclusions.** No signal/temporal/legality logic; no new observation
  variant; no new geometry primitive; no scene-schema change.
- **Stop condition.** Stop when the crossing derivation is complete and tested; do not
  start the reasoner.

### P3-U5 — Red-light reasoner + pipeline + persistence + recorded-clip e2e

- **Objective.** A deterministic red-light reasoner that joins the crossing (per-track)
  and signal-state (context) streams, confirms a `RED_LIGHT_JUMPING` `ConfirmedEvent`
  when a track crosses into the junction while the signal is `RED` beyond amber-onset
  grace (excluding `permitted_movements`, deduplicated per `crossing_dedup_window`),
  composed onto the P3-U1 base and P3-U2 pipeline, persisted via the existing
  `EventStore`, and verified end to end on a recorded synthetic clip.
- **Why now.** With generalization (P3-U1/U2), the context stream + join (P3-U3), and
  the crossing derivation (P3-U4) in place, red-light is a **configuration** of the
  generalized base plus a small legality predicate — the cheapest remaining violation
  (no new model; geometry and scene semantics already exist).
- **Inputs.** The P3-U4 crossing derivation; the P3-U3 signal-state context +
  generalized join; the P3-U1 reasoner base (trigger-with-dedup confirmation policy);
  the P3-U2 pipeline base; `SignalGroup.permitted_movements`; the scene's
  `red_light_jumping` params (`amber_onset_grace`, `crossing_dedup_window`);
  `EventStore` + evidence stub.
- **Exact scope.** (a) `rules/red_light.py`: a loader
  `red_light_parameters(scene)` (requires the signal group + stop line + junction
  zone; loads `amber_onset_grace`; optional `crossing_dedup_window`); the legality
  predicate (crossing-into-junction AND `signal_state == RED` beyond amber grace AND
  movement not in `permitted_movements`); a `RedLightReasoner` composed on the P3-U1
  base with the trigger-with-dedup policy and the ≥2-observation floor; content-derived
  `event_id`; run-level `models`. (b) `pipeline/red_light.py`: a thin configuration of
  the P3-U2 base injecting the crossing + signal-state derivations and the reasoner;
  fail-fast `SceneConfigurationError` if the scene lacks a signal group / stop line /
  junction zone. (c) A recorded-synthetic-clip end-to-end test (a PyAV-generated clip
  of a rectangle crossing the stop line into the junction under a declared RED
  schedule; injected detections; real ingestion + real `IouTracker` + real reasoning +
  real persistence), mirroring the P1-U12 / P2-U6 honesty bar. (d) A demo entry point
  (a sibling runner or a `--violation red_light` selector, whichever is the smallest
  coherent change).
- **Outputs.** Confirmed red-light events persisted as deterministic JSON with minimal
  manifests; an offline demo path; an honest run report labelling the detector kind.
- **Acceptance criteria.** A track crossing into the junction while RED (beyond amber
  grace) confirms exactly one `RED_LIGHT_JUMPING` event with correct
  `start_at`/`trigger_at`, ≥2 supporting observations, `models` carried, scene hash,
  and rule trace; a crossing on GREEN/AMBER-within-grace, or a `permitted_movements`
  movement, confirms nothing (countable abstention); duplicate crossings within
  `crossing_dedup_window` do not double-confirm; taint prevents cross-switch
  confirmation; deterministic + byte-identical persisted files on replay; **no
  contract change**; no claim RT-DETR fired on synthetic pixels.
- **Required tests.** Confirm on RED crossing; no-confirm on GREEN / amber-grace /
  permitted-movement / reversing; dedup; taint no-confirm; ≥2-observation floor;
  event-id determinism; order-independence; recorded-clip e2e + manifest; backend-free
  import boundary. (`tests/rules/test_red_light.py`,
  `tests/pipeline/test_red_light_pipeline.py`,
  `tests/pipeline/test_red_light_e2e.py`.)
- **Explicit exclusions.** No learned signal classifier; no ANPR; no congestion
  interaction; no contract change; no new package.
- **Stop condition.** Stop when red-light confirms/abstains correctly and
  deterministically end to end on a recorded clip; do not start the observation-log or
  evaluation units here.

### P3-U6 — Observation-log substrate (first-class; ADR-002 Parquet)

- **Objective.** Persist the derived observation streams to a deterministic,
  replay-ready log (ADR-002 Parquet, one file per `(video, run)`) so the §15
  "deterministic replay from the observation log" claim is **materially backed** and
  the evaluation harness has a stable substrate.
- **Why now.** The design review flagged this as the biggest honesty gap: the
  strongest defensibility mechanism currently has no persisted substrate. It depends
  only on the observation contracts (present) and is independent of red-light.
- **Inputs.** The `Observation` discriminated union; `persistence/` (existing
  package); the derivations from Phases 1–3; ADR-002.
- **Exact scope.** An observation-log writer/reader in `persistence/` (e.g.
  `persistence/observation_log.py`) writing the per-run observation streams as
  Parquet (deterministic column/row ordering; content-stable), plus a reader that
  reconstructs the `Observation` objects for **model-free replay** into the reasoners.
  Parquet support is an **optional dependency** (e.g. a `pyarrow` extra), lazily
  imported; the base install and default CI stay dependency-free. Output paths are
  gitignored runtime locations. Demonstrate replay: reasoning over the persisted log
  yields byte-identical events to reasoning over the in-memory derivations.
- **Outputs.** A deterministic observation-log writer + reader; a replay-equivalence
  demonstration.
- **Acceptance criteria.** Written logs are byte-stable across identical runs;
  round-trip (derive → write → read → reason) yields byte-identical events to the
  in-memory path for all three violations; the base install imports without the
  Parquet extra (writer/reader raise a typed, actionable error if the extra is
  absent); no contract change.
- **Required tests.** Write determinism; read round-trip equality; replay-equivalence
  to in-memory reasoning; optional-extra absence handling; gitignore coverage.
  (`tests/persistence/test_observation_log.py`.)
- **Explicit exclusions.** No SQLite runtime (still deferred); no evidence-media
  rendering; no cross-run dedup (ADR-004 stays Proposed); no mandatory new base
  dependency.
- **Stop condition.** Stop when the observation log persists and replays
  byte-identically behind an optional extra; do not start the evaluation harness here.

### P3-U7 — Event-level evaluation harness (first-class; §23-E)

- **Objective.** Implement the §23-E event-level evaluation protocol so confirmed
  events become measurable: event matching against ground-truth manifests, per-rule
  precision/recall/F1, **false-events-per-hour** (headline), duplicate rate, median
  detection delay, and evidence completeness.
- **Why now.** The project's thesis is honest, layered evaluation; no accuracy claim is
  permissible for any violation without it. It consumes the deterministic `EventStore`
  (present) + a ground-truth format and is independent of red-light.
- **Inputs.** `ConfirmedEvent` / `EventStore`; `docs/evaluation-protocol.md` §23-E
  (matching: same rule AND temporal overlap AND median track-IoU / center-containment
  fallback; one-to-one greedy by descending confidence); a ground-truth event-manifest
  format.
- **Exact scope.** A new `evaluation/` package (e.g. `evaluation/events.py`) with a
  small, versioned ground-truth event-manifest schema (reusing contract primitives),
  the deterministic event-matching algorithm of §23-E, and the per-rule metric
  computations (P/R/F1, false-events/hour, duplicate rate, detection delay, evidence
  completeness). Add `evaluation` to the sanctioned-package allow-list in
  `tests/docs/test_adr_pack.py` **in this unit's commit** (this is the first
  legitimate place). Metrics are pure functions of typed inputs; no wall-clock, no
  randomness.
- **Outputs.** A deterministic event-level evaluation harness usable on any run's
  `EventStore` output against a ground-truth manifest.
- **Acceptance criteria.** Matching is deterministic and one-to-one; metrics match the
  §23-E definitions on hand-built fixtures (perfect match, misses, false positives,
  duplicates, near-boundary temporal overlap); no accuracy claim is made by the harness
  itself (it computes metrics on provided GT, it does not assert real-world
  performance); the `evaluation` package is added to the allow-list.
- **Required tests.** GT-manifest schema validation; matching on perfect/miss/FP/
  duplicate/boundary cases; each metric on a known fixture; determinism.
  (`tests/evaluation/test_event_matching.py`, `tests/evaluation/test_event_metrics.py`.)
- **Explicit exclusions.** Only the event-level protocol (§23-E) here; detection /
  tracking / helmet / ANPR / system / robustness protocols land with their capabilities
  in later phases; no real footage; no accuracy claim; no new dependency.
- **Stop condition.** Stop when event-level metrics compute deterministically against a
  GT manifest; Phase 3 coding is complete.

---

## 8. Dependency graph

```
P3-U1 (reasoner base) ──> P3-U2 (pipeline base) ─┐
                                                  │
P3-U3 (dynamic context + generalized join) ──────┤
                                                  ├─> P3-U5 (red-light reasoner+pipeline+e2e)
P3-U4 (crossing derivation) ─────────────────────┘

First-class, parallel (depend on the settled reasoner/pipeline shape from P3-U1/U2,
exercise the existing wrong-way / illegal-stopping paths; block nothing in the
red-light chain):
  P3-U6 (observation-log substrate)   — needs P3-U1/U2 settled
  P3-U7 (event-level evaluation)      — needs the EventStore (present)

External / parallel (block nothing in the chain):
  Real-footage validation gate — carried forward from Phase 2 (P2-V1 successor)
```

## 9. Implementation ordering

1. **P3-U1**, then **P3-U2** — generalize by composition first, byte-identical, before
   any third reasoner/pipeline exists.
2. **P3-U3** and **P3-U4** — the two evidence halves red-light needs (parallelizable
   with each other; both depend only on the settled base).
3. **P3-U5** — red-light as a configuration of the generalized base.
4. **P3-U6** and **P3-U7** — first-class evidence-substrate and evaluation deliverables,
   started in parallel once P3-U1/U2 are settled.

## 10. Phase 3 Definition of Done

- P3-U1…U7 complete; each unit's acceptance criteria and required tests pass.
- Wrong-way and illegal-stopping reasoners and pipelines delegate to composed bases
  with **byte-identical** events, event ids, and persisted files (no existing test
  edited).
- Red-light jumping runs end to end offline on a **recorded** synthetic clip through
  real ingestion + real `IouTracker` + dynamic signal-state context + crossing
  derivation + real reasoning + real persistence, deterministically and with
  byte-identical persisted files on replay, with an explicit injected detector.
- The observation-log substrate persists and **replays byte-identically** behind an
  optional Parquet extra; the base install/CI stay dependency-free.
- The event-level evaluation harness computes the §23-E metrics deterministically
  against a ground-truth manifest.
- Quality gates green: `ruff check .`, `mypy src`, full `pytest -q` (opt-in
  real-model tests still skipped by default).
- **No frozen contract, schema, ADR, or master-spec change.** Exactly one new package
  (`evaluation`) added to the allow-list.

## 11. Claims allowed after Phase 3

- "TrafficPulse implements three offline, deterministic violation-reasoning slices —
  wrong-way, illegal-stopping, and red-light jumping — end to end from recorded video,
  with red-light combining a **dynamic signal-state context** and a per-track stop-line
  crossing under scene semantics."
- "The two original reasoners/pipelines were generalized by composition with no change
  in behaviour (byte-identical events and persisted files)."
- "Derived observations are persisted to a deterministic observation log and the
  reasoning layer replays byte-identically from it without a model or GPU."
- "An event-level evaluation harness computes per-rule precision/recall/F1,
  false-events/hour, duplicate rate, and detection delay against ground-truth
  manifests."

## 12. Claims still forbidden after Phase 3

- No real-world / event-level accuracy, precision, recall, or false-events/hour claim
  on real footage (requires the external real-footage gate and ground-truth manifests
  from approved footage).
- No claim that a real detector (or a learned signal classifier) produced a confirmed
  event on synthetic pixels; signal state in this phase is a declared/simulated log.
- No congestion-robust illegal-stopping or red-light claim (congestion suppression
  remains deferred).
- No helmet / triple-riding / speeding / ANPR / review / penalty capability claim.
- No production/enforcement-readiness claim.

## 13. Handoff criteria

- **To Phase 4 (association + classifier violations).** The generalized reasoner base
  (P3-U1) accepts confidence-bearing predicates; the generalized pipeline base (P3-U2)
  accepts new finalize strategies; the observation log (P3-U6) can persist new
  observation variants; the evaluation harness (P3-U7) can score new rules. All are the
  seams Phase 4 builds on.
- **To real-footage validation.** The red-light pipeline runs offline and
  deterministically; a matching real `SceneConfig` (signal group + stop line + junction
  zone + a real signal log/source) is the only remaining external gate; no code change
  is needed to run validation.

## 14. Stop conditions (phase-level)

- Stop at the end of P3-U7 with gates green.
- **Stop and report** (do not silently change a contract or edit a test) if any unit
  discovers a genuine need for a frozen-contract, schema, ADR, or master-spec change —
  in particular if the in-zone-transition model for red-light (§4a) proves insufficient
  and a dedicated crossing observation is genuinely required.
- Do not begin association, helmet, triple-riding, speeding, review, or penalty work.
- Do not add a learned signal classifier, SQLite runtime, or evidence-media rendering in
  this phase.
- Do not let real-footage acquisition gate completion of P3-U1…U7.
