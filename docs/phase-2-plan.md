# TrafficPulse — Phase 2 plan

**Phase name:** Phase 2 — Evidence Integrity and Multi-Violation Generalization

- **Status:** Authoritative Phase 2 unit plan (planning document; no Phase 2 source
  implemented yet).
- **Date:** 2026-07-10
- **Authority:** This is the **authoritative Phase 2 unit plan**. It governs the
  `P2-U#` / `P2-V#` / `P2-R#` identifier namespace only. It does **not** supersede
  [`docs/phase-0-plan.md`](phase-0-plan.md) (Phase 0-F, `U#`) or
  [`docs/phase-1-plan.md`](phase-1-plan.md) (Phase 1, `P1-U#`); those remain
  governed by their own documents and their histories are not rewritten here. The
  three phases use **separate identifier namespaces** (`U#` vs `P1-U#` vs `P2-U#`).
- **Canonical architecture reference:** [`docs/architecture-review.md`](architecture-review.md)
  remains THE canonical architecture reference; this plan interprets and sequences
  it for Phase 2 without modifying it or the master spec
  ([`TRAFFICPULSE_MASTER_SPEC.md`](../TRAFFICPULSE_MASTER_SPEC.md)).

---

## 1. Relationship to Phase 0 and Phase 1

Phase 0-F (`U1…U6`) froze the contracts, ontology, dataset registry/policy,
evaluation protocol, scene-configuration schema + typed `SceneConfig` + stable
`scene_config_hash`, and the architecture/ADR pack. Those artifacts are **frozen
interfaces** and Phase 2 builds against them unchanged.

Phase 1 (`P1-U1…P1-U12`) delivered the first behavioral vertical slice — the
**wrong-way** capability — end to end and offline:

```
recorded clip
  → open_video (P1-U5, real PTS)                → FrameRecord
  → Detector + DetectionAdapter (P1-U6/U7)      → Detection
  → Tracker: real IouTracker (P1-U8/U9)         → TrackState
  → derive_heading_observations_with_taint (P1-U4) → HeadingVsLaneObservation
  → WrongWayReasoner (P1-U4)                     → ConfirmedEvent
  → EventStore.persist (P1-U11)                  → events/ + manifests/ JSON
  → EvidenceManifest stub (P1-U11)               → minimal reviewable manifest
  → run_wrong_way_slice / `python -m trafficpulse.pipeline` (P1-U12)
```

Phase 2 is the **second violation** plus the first **evidence-integrity**
hardening. It reuses every Phase 1 seam and adds no new perception model and no
dataset dependency.

---

## 2. Verified starting point

Verified at planning time (`main`, HEAD `89c0d1f`, clean working tree, synced with
`origin/main`; `pip check` clean; `ruff`/`mypy --strict` clean; **764 passed, 3
skipped**):

- **The wrong-way vertical slice is structurally complete.** The event-producing
  demo path is: recorded synthetic clip → real PTS ingestion → scripted
  `StubDetector` → `DetectionAdapter` → real `IouTracker` → heading observations →
  `WrongWayReasoner` → `ConfirmedEvent` → `EventStore` → `EvidenceManifest` stub.
- **Exact capability qualification.** Genuine RT-DETR inference is integrated and
  locally runnable offline (torch + transformers installed; `PekingU/rtdetr_r50vd`
  cached; opt-in real-model path exists), but a COCO RT-DETR does **not** fire the
  vehicle class on the synthetic rectangle clip, so the *real detector has not yet
  driven a confirmed event*. The deterministic event test therefore uses **injected
  (scripted-stub) detections** over the real ingestion/tracker/rules/persistence
  stack. This qualification must be preserved verbatim by Phase 2 and never
  softened into a claim that RT-DETR produced the event.
- **Known provenance gap.** `ConfirmedEvent.models` is `()` (empty) on every event
  the reasoner mints. `WrongWayReasoner._confirm` constructs the event without
  `models=`, and neither the P1-U10 pipeline nor the P1-U12 runner constructs
  truthful `ModelRef`s for the detector/tracker. `DetectorConfig.source_model` and
  `TrackerConfig.tracker` are both left unset in the current composition, so
  `Detection.source_model` and `TrackState.tracker` are also `None`.
  `EvidenceManifest.build_evidence_manifest` already copies `event.models`, so the
  manifest is honestly empty rather than fabricated — closing the gap at the event
  is sufficient to close it at the manifest.
- **Real-footage external gate.** No approved real-world footage exists
  (`registry/datasets/event-evaluation-footage.yaml`: `status:
  self_collection_required`, `access_confirmed_by_project: false`). Real-footage
  validation is an **external, gated activity** and is not part of the coding
  critical path.

---

## 3. Phase 2 objectives

- **A. Evidence-integrity: model provenance propagation.** Confirmed events (and
  their manifests) truthfully carry the detector/tracker `ModelRef`s already
  available at the pipeline composition boundary, without weakening model-free
  reasoning replay.
- **B. Second violation: illegal stopping/parking.** Add an offline, deterministic
  illegal-stopping capability built from zone membership + stationarity + temporal
  dwell + abstention/recovery, with confirmed-event persistence and
  evidence-manifest linkage — reusing the existing detector/tracker architecture
  and adding no new perception model.
- **C. Offline + deterministic.** All Phase 2 implementation is offline, uses no
  wall-clock in the decision path, and replays bit-exactly.
- **D. Real-footage readiness (not dependency).** Leave the project ready for an
  external real-footage validation run without making footage acquisition part of
  the implementation critical path.
- **E. Public-status correction.** After implementation, correct the stale README
  and public project-status documentation so they match the implemented repository.
- **F. Helmet research gate (parallel, non-blocking).** Advance helmet
  dataset/licence readiness as governance/research preparation only — no training,
  no CNN-vs-ViT work in this phase.

---

## 4. Non-goals (Phase 2 will NOT do)

- No red-light-jumping capability.
- No helmet detection/classification and **no** CNN-vs-ViT training of any kind.
- No ANPR / OCR.
- No speeding / speed estimation.
- No review UI, no penalty simulation.
- No SQLite/Parquet runtime (ADR-002 defers it; the JSON `EventStore` stays the
  storage posture; ADR-004 stays *Proposed*).
- No full evidence rendering engine (no clip/frame rendering, crops, overlays,
  content-addressed media hashing).
- No model-targeted synthetic image fixture engineered to trick RT-DETR into
  firing. Deterministic event tests use scripted-stub or injected detections, as in
  Phase 1.
- No change to any frozen contract, schema, ADR, or the master spec.
- No new `src/trafficpulse` package (all Phase 2 code lands in the already-sanctioned
  `observations`, `rules`, `pipeline`, and `persistence` packages).
- No stationary-object re-association / ID-churn memory (documented deferral, §10).
- No congestion-suppression behaviour (documented scoping, §10/§13).

---

## 5. Architectural invariants (preserved)

1. **Perception ↔ reasoning separation.** Rules consume **only** `Observation`
   contracts; observation derivation consumes only frozen `TrackState`/geometry/
   scene data. No rule sees a detector/tracker framework type.
2. **Model-free deterministic replay** (architecture-review §15). The reasoning
   layer replays bit-exactly from the observation log with no model/GPU. Phase 2
   must not route any model-dependent value into a reasoning *predicate*.
3. **Frozen contracts.** `Detection`, `TrackState`, all `Observation` variants,
   `ConfirmedEvent`, `EvidenceManifest`, `ModelRef`, `ZoneKind`, `ProducerKind`,
   `SceneConfig`, and `scene_config_hash` are unchanged. Every field/enum Phase 2
   uses already exists (verified in §6-audit and the self-audit, §16).
4. **≥ 2 observations to confirm; every non-confirmation is a countable
   abstention** (architecture-review §13). The generic `RuleEngine` already
   enforces the lifecycle; the new reasoner reuses it exactly as `WrongWayReasoner`
   does.
5. **Determinism.** No wall-clock in decisions; content-derived ids; timestamp-
   ordered processing; deterministic output ordering. Media time comes from PTS via
   the fixed-epoch anchor already used by the pipeline.
6. **Honesty.** No fabricated detection, no fabricated provenance, no accuracy claim
   from a single clip, no claim that the real detector produced an event when a stub
   did.
7. **Thin composition roots.** Orchestration wires existing seams; it re-implements
   nothing and imports no backend inside the library core (backends named only in
   the runner/composition root, as in P1-U12).

---

## 6. Provenance design decision for this phase (governs P2-U1)

**Decision: detector/tracker `ModelRef`s are collected at the pipeline composition
boundary as run-level provenance metadata and stamped onto the minted
`ConfirmedEvent` by the reasoner; they never enter any reasoning predicate.**

Rationale and mechanics, resolving Phase-4 design questions A.1–A.10:

1. **Where collected (A.1).** In the pipeline's per-frame step (`process_frame`),
   which already sees both `Detection.source_model` (from the `DetectionAdapter`)
   and `TrackState.tracker` (from the tracker seam). The pipeline accumulates the
   **union** of the distinct non-`None` `ModelRef`s observed across the run.
2. **How they reach `ConfirmedEvent.models` (A.2).** The pipeline passes the
   collected run-level tuple into the reasoner (constructor arg, defaulting to
   `()`); the reasoner sets `models=` when it constructs the `ConfirmedEvent` in its
   `_confirm` step. `EvidenceManifest` already inherits `event.models` verbatim via
   `build_evidence_manifest` — **no evidence-stub change is required**.
3. **Run-level, not observation-level (A.3).** Observations carry a `Producer`
   (`name`/`version`/`ProducerKind`), **not** a `ModelRef`, by frozen-contract
   design. Routing `ModelRef`s through observations is therefore both impossible
   without a contract change and undesirable (it would couple the model-free
   reasoning log to model identity). Provenance is attached to the *event* as
   inert metadata, alongside — never inside — the reasoning path.
4. **Deterministic ordering (A.4).** The collected tuple is sorted by
   `(name, version, weights_hash or "")` and de-duplicated, so identical runs
   produce byte-identical `models` tuples regardless of frame/emit order.
5. **Duplicate handling (A.5).** De-duplicated by the full `ModelRef` identity
   `(name, version, weights_hash)`; a detector emitting the same `source_model` on
   every frame contributes exactly one entry.
6. **Model-free replay preserved (A.6).** The `ModelRef`s are pure provenance data:
   no rule predicate, threshold, dwell timer, or transition ever reads them. Replaying
   the observation log with the same run-level `models` tuple yields byte-identical
   events; replaying with an empty tuple yields the same *decision* and differs only
   in the inert `models` field. The reasoning decision is independent of provenance.
7. **Does the runner construct enough truthful `ModelRef`s today? (A.7)** No — today
   `DetectorConfig.source_model` and `TrackerConfig.tracker` are unset. P2-U1 must
   also make the composition (P1-U12 runner and the new P2 runner) construct honest
   `ModelRef`s and pass them via `DetectorConfig.source_model` / `TrackerConfig.tracker`
   so `Detection.source_model` / `TrackState.tracker` are populated at their existing
   stamping points (`DetectionAdapter`, `TrackAdapter`). No new stamping site is
   introduced.
8. **Fields that can be populated honestly (A.8).** `ModelRef.name` and
   `ModelRef.version`: for RT-DETR, `name` = the checkpoint id/dir the runner already
   receives (`--checkpoint`), `version` = a provisional version string (e.g. the
   transformers/model-config identifier available at construction, or
   `"provisional"`); for the tracker, `name` = the in-repo tracker id (e.g.
   `"iou-tracker"`) and `version` = its provisional version string.
9. **What stays empty (A.9).** `ModelRef.weights_hash` stays `None` — nothing hashes
   weights in this phase; a hash would be fabricated. Stub-driven test paths pass no
   `source_model`/`tracker` (or explicit test refs) and therefore honestly carry the
   provenance they were given, never invented values.
10. **Manifest inheritance without fabrication (A.10).** `build_evidence_manifest`
    already copies `event.models`, `event.code_version`, `event.scene_config_hash`;
    once the event carries truthful `models`, the manifest carries the same, and
    nothing new is invented.

**Scope guard.** P2-U1 changes only reasoner/pipeline/runner **wiring** (adds a
`models` constructor arg to the reasoner and a collection step to the pipeline) and
constructs honest `ModelRef`s in the composition root. It changes **no** frozen
contract (the `ConfirmedEvent.models` and `EvidenceManifest.models` fields already
exist) and adds **no** dependency.

---

## 7. Illegal-stopping capability definition

Per architecture-review §5c and master-spec §2.5: a track that is **stationary**
inside a configured **no-stopping** zone for longer than a dwell threshold is an
illegal-stopping violation, subject to abstention on taint and (future) congestion.

- **Violation type.** `ViolationType.ILLEGAL_STOPPING` (frozen enum member, exists).
- **Eligible zones (first slice).** Scene zones with `zone_type == ZoneType.NO_STOPPING`
  whose `applicable_violations` include `illegal_stopping`. The example scene's
  `zone-no-stop` already declares `zone_type: no_stopping`, `applicable_violations:
  [illegal_stopping]`, `observation_consumers: [in_zone, stationary]`.
- **Parking zones deferred.** `ZoneKind.PARKING` exists for `InZoneObservation`, but
  the scene `ZoneType` closed set has **no** `parking` member, so parking-zone
  illegal-stopping would require a reviewed scene-schema change and is **out of scope
  for Phase 2** (documented, not implemented).
- **Learned components.** None beyond the detector — exactly as Phase 1.

---

## 8. Observation semantics

### 8a. In-zone observation (`InZoneObservation`; governs P2-U2)

Resolves Phase-4 design questions B.1–B.7.

- **Reference point (B.1/B.2).** **Bottom-center of the bbox** (`((x1+x2)/2, y2)`),
  the ground-contact point. Architecture-review §17 names "bbox bottom-center" as the
  ground-plane reference; for "is the vehicle stopped *in* this zone" the wheels'
  contact point is the defensible choice. This differs from the heading derivation's
  bbox-*center* — a deliberate, documented difference: heading needs a
  displacement-direction-invariant reference (center is exact for constant-size
  synthetic boxes), whereas zone membership needs ground contact. The choice is
  provisional and revisitable once calibrated ground-plane reasoning exists.
- **Polygon-edge semantics (B.3).** Reuse `geometry.point_in_polygon`, whose frozen
  policy is **boundary counts as inside** (on-edge/on-vertex → inside),
  deterministic. No new geometry code.
- **Eligible zone kinds (B.4).** First slice: `no_stopping` only (mapped scene
  `ZoneType.NO_STOPPING` → observation `ZoneKind.NO_STOPPING`; both wire value
  `"no_stopping"`). The derivation emits one `InZoneObservation` per (track, eligible
  zone) per usable frame, with `zone_id`, `zone_kind`, and `is_inside`.
- **Multiple zones (B.5).** A track may be inside more than one zone; one observation
  per (track, zone). The reasoner filters to eligible no-stopping zones.
- **Ambiguity (B.6).** None to represent for point-in-polygon: membership is a
  deterministic boolean (`is_inside`). Edge points resolve to inside per B.3.
- **Incomplete geometry (B.7).** If the scene declares no eligible no-stopping zone,
  the illegal-stopping pipeline raises `SceneConfigurationError` at construction
  (mirroring `_resolve_legal_direction`'s fail-fast). The derivation itself makes no
  violation decision.
- **Producer.** A `Producer(kind=ProducerKind.HEURISTIC)` (e.g.
  `name="in-zone", version="0.1.0-provisional"`), matching the heading derivation's
  provenance pattern. `ProducerKind.HEURISTIC` exists.
- **Non-goals.** No zone-priority logic, no temporal logic (zones carry none), no
  parking zones.

### 8b. Stationary observation (`StationaryObservation`; governs P2-U3)

Resolves Phase-4 design questions C.1–C.8.

- **Available contract fields (C.1).** `is_stationary: bool` (load-bearing),
  `speed_estimate: NonNegativeFloat | None`, `dwell_seconds: NonNegativeFloat | None`.
- **Method (C.2).** Sliding-window net displacement of the bbox bottom-center: a step
  is stationary iff the net displacement across a short trailing window is below a
  small **pixel-space** epsilon. A window (rather than a single pairwise step) is used
  because in-place jitter must still read as stationary (C.5). The window length is a
  provisional derivation parameter, labelled provisional.
- **Uncalibrated-slice honesty (C.7/C.8).** The scene's `motion_threshold` (0.5 m/s,
  `provisional`) is **loaded and recorded for provenance but not applied** in this
  uncalibrated synthetic slice — converting m/s to the pixel space of synthetic
  tracks needs a validated calibration that does not exist (`calibration.status:
  provisional`, `verification_status: unverified`). This mirrors exactly the accepted
  wrong-way pattern where `min_speed` is carried but not applied and the usable-motion
  gate is a geometric pixel-space test. `speed_estimate` is left `None` (no calibrated
  speed is claimed); `dwell_seconds` is **not** set on the per-frame observation (dwell
  is a reasoning-layer accumulation, C.6). The stationarity epsilon and window are
  provisional and explicitly labelled.
- **Time (C.3).** From `TrackState.timestamp` (PTS-anchored media time), never
  wall-clock.
- **Gaps (C.4).** Ordinary gaps (missing/immobile samples of one continuous track)
  are bridged by timestamp, exactly as heading derivation does. An explicit **taint**
  (ID-switch discontinuity) drops the step and marks the next clean observation as a
  **taint restart**, reusing the `HeadingDerivation.taint_restart_ids` mechanism
  verbatim so stationarity/dwell can never accumulate across an ID switch
  (architecture-review §13: tainted tracks may abstain but never confirm).
- **Hysteresis placement (C.6).** Hysteresis (enter/exit separation) and minimum
  dwell live in the **temporal reasoning** layer (architecture-review §13), not in
  observation derivation. The observation is a raw per-step stationarity fact; the
  reasoner accumulates dwell and applies confirmation.
- **Producer.** `Producer(kind=ProducerKind.HEURISTIC)` (e.g. `name="stationary"`),
  matching the pattern.

---

## 9. Temporal reasoning semantics (illegal-stopping reasoner; governs P2-U4)

Resolves Phase-4 design questions D.1–D.16. The reasoner is structurally the
wrong-way reasoner's twin: it drives the generic `RuleEngine` for lifecycle
mechanics and mints `ConfirmedEvent`s for sustained evidence.

- **Evidence predicate (D.1).** A per-(camera, track, timestamp) **stopped-in-zone**
  signal = `is_stationary` **AND** `is_inside` an eligible no-stopping zone. This
  combines the two observation streams by a deterministic join on
  `(camera_id, track_id, timestamp)` (a small pairing helper, analogous to heading's
  per-step `is_contradiction` flag). The reasoner consumes observations only — the
  join produces a per-step boolean from frozen observation facts, staying model-free.
- **Dwell semantics (D.2).** A contiguous run of stopped-in-zone steps is tracked per
  (camera, track). `dwell = current.timestamp − run_start.timestamp`. Confirm when
  `dwell ≥ stationary_duration`.
- **Minimum evidence (D.3).** ≥ 2 observations structurally required (a run needs a
  later observation than the one that opened it), satisfying architecture-review §13
  and matching the wrong-way structure.
- **Maximum observation gap (D.4).** First slice reuses the wrong-way gap handling:
  ordinary gaps bridge by timestamp; taint restarts reset the run. An explicit
  `max_observation_gap` tolerance (§5c "max-gap tolerance") is a **planned provisional
  scene rule-parameter** (added in P2-U4 as `{id: max_observation_gap, unit: seconds,
  status: provisional}` under the existing `illegal_stopping` block) that, when set,
  ends a run whose inter-observation gap exceeds it; when unset the run relies on
  timestamp bridging. This is additive config data, not a contract change.
- **Recovery (D.5).** A moving step, or a step outside every eligible zone, ends the
  run — `close` if already confirmed, else `abandon` — exactly `WrongWayReasoner._on_recovery`.
- **Missing evidence (D.6).** A track that never sustains a stopped-in-zone run
  produces no event; the open hypothesis abandons (a countable abstention).
- **Abstention conditions (D.7).** Taint restart (never confirm across an ID switch);
  recovery before `stationary_duration`; (future) congestion flag active. Every
  non-confirmation is a logged, countable abstention via the engine lifecycle.
- **Track-identity assumptions (D.8).** A single continuous track id. **No** stationary-
  object re-association across ID churn in this phase (see D.9).
- **Occlusion/gap limitations (D.9).** If the tracker churns the id of a long-stationary
  vehicle, dwell resets under the new id. The mitigation (a stationary-object memory
  re-associating by IoU, architecture-review §5c/§13) is an **explicit deferral** —
  documented, not implemented — so the first slice's dwell is honest about ID-churn
  limits.
- **Congestion false-positive risk (D.10).** Real: a queue at a signal is
  stationary-in-zone and would false-positive. **First slice explicitly excludes
  congestion-heavy scenes (D.11):** it targets single-vehicle / non-congested
  synthetic scenes; `congestion_suppression` stays `unset`; the reasoner implements no
  congestion behaviour and the plan states plainly that congested real footage
  requires scene-level flow-state suppression before any claim.
- **Event timing (D.12).** `start_at` = run-start timestamp (first stopped-in-zone
  observation); `trigger_at` = the observation at which `dwell ≥ stationary_duration`;
  `end_at` = `None` at confirmation (mirrors wrong-way; end disposition is future work).
- **Event id (D.13).** Content-derived SHA-256 over `(scene_config_hash, camera_id,
  violation_type=illegal_stopping, rule_id, track_ids, start_at, trigger_at,
  source_hypothesis_id)`, `"evt-" + digest[:16]` — the same deterministic, process-
  independent scheme as wrong-way. ADR-004 stays *Proposed*; this content-derived
  choice is the smallest deterministic option for the replay context and does not fix
  cross-run identity.
- **Rule id/version (D.14).** `rule_id = "illegal_stopping"`,
  `rule_version = "0.1.0-provisional"`.
- **Measurements & thresholds on the event (D.15).** `measurements`:
  `dwell_seconds` (measured). `thresholds`: `stationary_duration` (applied) and
  `motion_threshold` (recorded, **not applied** — flagged as provisional/uncalibrated).
- **Evidence-manifest rule trace (D.16).** Inherited unchanged from
  `build_evidence_manifest`: a rule step (`rule:illegal_stopping`, note = rule_version,
  thresholds) and a `confirmed` step (measurements). No evidence-stub change.

---

## 10. Evidence / persistence semantics

- Reuse the P1-U11 `EventStore` verbatim: per-run deterministic JSON under
  `output_dir/<run_id>/events/*.json` and `.../manifests/*.json`, keyed by
  `event_id`, write-once-per-`(run_id, event_id)`, idempotent identical replay,
  `EventConflictError` on differing rewrite.
- Reuse the P1-U11 evidence stub verbatim: `build_evidence_manifest` builds the
  minimal reviewable manifest (id linkage, trigger-frame *relative locator* with no
  hash, rule trace, carried provenance including the now-populated `models`).
- **No evidence rendering** (no frames/clips/crops/overlays/OCR/media hashing) — those
  remain future units.

---

## 11. Determinism requirements

- No wall-clock in any decision; `created_at` on events is the reasoner's data
  timestamp (`trigger_at`).
- Observation derivation, the stationary/in-zone join, and reasoning process
  observations in `(timestamp, observation_id)` order; events are emitted sorted by
  `(trigger_at, event_id)`; `models` tuples are sorted+de-duplicated.
- Identical clip + scene + injected components → equal event set and byte-identical
  persisted files, on fresh instances and on `reset`+replay.
- Media time from PTS via the fixed UTC-epoch anchor already used by the pipeline.

---

## 12. External real-footage validation gate (P2-V1; NOT on the coding critical path)

- **Nature.** An **external, gated validation activity**, separable from P2-U1…U7.
  Completion of P2-U1…U7 must not depend on, or be blocked by, footage acquisition.
- **Preconditions (all external).** Institutional recording permission / ethics
  clearance; a scouted site; a written data-handling plan; approved/owned/licensed
  vehicle footage; and a matching validated `SceneConfig` with a real no-stopping
  zone. Tracked in `registry/datasets/event-evaluation-footage.yaml`
  (`self_collection_required`, not yet secured).
- **Execution (uses only existing code).** approved footage + matching `SceneConfig`
  → existing real RT-DETR path → existing real `IouTracker` → the Phase 2
  observation/reasoner/persistence path. No new code is required to *run* validation;
  it exercises the shipped pipeline.
- **Honesty.** Until this gate runs, no real-world accuracy, event-level precision/
  recall, or "validated on real footage" claim may appear. The gate does not gate
  Phase 2 *completion*; it gates *claims*.

---

## 13. Parallel helmet dataset/licence research gate (P2-R1; non-blocking)

- **Nature.** Governance/research preparation only. It is **not** a coding unit, is
  **not** on the critical path, and starts **no** model training and **no** CNN-vs-ViT
  work.
- **Scope.** Confirm and record the HELMET (Myanmar) dataset licence and access terms
  (currently `licensing.status: unknown`, `access.status: unconfirmed` in
  `registry/datasets/helmet-myanmar.yaml`); record turban-label / Indian-domain gaps
  and the custom-annotation need; keep the entry `candidate` until licence + access
  gates resolve. **No dataset is downloaded**; no checksum is recorded until an
  authorized acquisition.
- **Exit into a later phase.** CNN-vs-ViT helmet training may begin only in a **later
  phase**, only after the data + licence gates are resolved and recorded.

---

## 14. Ordered unit cards

Dependency order: **P2-U1 → P2-U2 → P2-U3 → P2-U4 → P2-U5 → P2-U6 → P2-U7.**
P2-V1 and P2-R1 run outside this chain and block nothing in it.

### P2-U1 — Model provenance propagation

- **Objective.** Make confirmed events (and their manifests) carry the truthful
  detector/tracker `ModelRef`s available at composition, closing the known
  `ConfirmedEvent.models == ()` gap without touching a reasoning predicate.
- **Why now.** It is a pure evidence-integrity fix on the already-shipped wrong-way
  path, independent of the new violation, and it establishes the provenance-carrying
  reasoner/pipeline shape the illegal-stopping units reuse.
- **Inputs.** `Detection.source_model`, `TrackState.tracker`, `ConfirmedEvent.models`,
  `EvidenceManifest.models`, `DetectorConfig.source_model`, `TrackerConfig.tracker`,
  `WrongWayReasoner`, `WrongWayPipeline`, `run_wrong_way_slice` / CLI.
- **Exact scope.** (a) Add an optional run-level `models: tuple[ModelRef, ...] = ()`
  to the reasoner and set `models=` in its `_confirm`. (b) In the pipeline, collect the
  distinct non-`None` `ModelRef`s from `Detection.source_model` and `TrackState.tracker`
  seen during `process_frame`, sort+de-dup, and pass them to the reasoner in
  `finalize`. (c) In the composition root(s), construct honest `ModelRef`s
  (`name`/`version` populated; `weights_hash=None`) and pass them via
  `DetectorConfig.source_model` / `TrackerConfig.tracker`. Follows §6 exactly.
- **Outputs.** Wrong-way `ConfirmedEvent.models` and `EvidenceManifest.models` populated
  with truthful refs on real-composition runs; empty (or explicit test refs) on stub
  runs that supply none.
- **Acceptance criteria.** A composition run with detector+tracker refs yields events
  whose `models` equals the sorted-deduped union of the supplied refs; the manifest's
  `models` equals the event's; a stub run supplying no refs yields `models == ()`;
  the reasoning **decision** (which events, ids, timing) is byte-identical with and
  without provenance (proving predicates ignore `models`); `weights_hash` is `None`
  everywhere.
- **Required tests.** Unit: reasoner stamps and sorts/dedups a provided `models` tuple;
  reasoner decision/event-id unchanged when `models` varies. Pipeline: collection
  yields the expected sorted-deduped union; empty when none supplied. Determinism:
  fresh-instance and reset-replay equality of `models`. (Unique basenames, e.g.
  `tests/rules/test_provenance.py`, `tests/pipeline/test_provenance_propagation.py`.)
- **Explicit exclusions.** No contract change; no `weights_hash` computation; no
  provenance in observations or predicates; no new dependency; no new package.
- **Stop condition.** Stop once wrong-way events carry truthful `models` end to end and
  determinism/decision-independence tests pass. Do **not** start P2-U2 work here.

### P2-U2 — In-zone observation derivation

- **Objective.** Deterministically derive `InZoneObservation` facts (bottom-center vs
  eligible no-stopping polygons) from a `TrackState` sequence + `SceneConfig`.
- **Why now.** In-zone membership is one of the two evidence streams the illegal-
  stopping reasoner joins; it is pure geometry over frozen inputs and unblocks P2-U4.
- **Inputs.** Ordered `TrackState`s; `SceneConfig` zones; `geometry.point_in_polygon`;
  `InZoneObservation`, `ZoneKind`, `Producer`, `ProducerKind`.
- **Exact scope.** A new function in `observations/` (e.g. `observations/zones.py`,
  `derive_in_zone_observations`) that, per usable step, emits one `InZoneObservation`
  per eligible zone with `zone_id`, `zone_kind` (scene `ZoneType.NO_STOPPING` →
  `ZoneKind.NO_STOPPING`), `is_inside` (bottom-center point-in-polygon), timestamp,
  `track_id`, `camera_id`, `producer`. Reuse the heading derivation's taint handling
  (skip tainted steps; mark taint restarts) so downstream reasoning cannot bridge an
  ID switch. §8a semantics.
- **Outputs.** A derivation result (observations + taint-restart ids) mirroring
  `HeadingDerivation`.
- **Acceptance criteria.** Bottom-center inside the polygon → `is_inside=True`;
  boundary point → `True`; outside → `False`; a track inside two zones yields two
  observations; tainted steps are skipped and the next clean observation is a taint
  restart; no observation before two states; fully deterministic (no wall-clock/
  randomness).
- **Required tests.** Inside/outside/on-edge membership; multi-zone emission;
  eligible-zone filtering; taint skip + restart marker; empty/short track;
  determinism. (`tests/observations/test_in_zone.py`.)
- **Explicit exclusions.** No dwell/stationarity, no reasoning, no parking zones, no
  scene-schema change, no new geometry primitive.
- **Stop condition.** Stop when `InZoneObservation` derivation is complete and tested;
  do not start stationary derivation.

### P2-U3 — Stationary observation derivation

- **Objective.** Deterministically derive per-step `StationaryObservation` facts
  (sliding-window pixel-space stationarity of bottom-center) from a `TrackState`
  sequence.
- **Why now.** Stationarity is the second evidence stream the reasoner joins; it is
  pure geometry over frozen inputs and unblocks P2-U4.
- **Inputs.** Ordered `TrackState`s; the provisional stationarity epsilon + window;
  the scene `illegal_stopping` params (loaded for provenance); `StationaryObservation`,
  `Producer`, `ProducerKind`.
- **Exact scope.** A new function in `observations/` (e.g.
  `derive_stationary_observations`) that, per usable step, emits one
  `StationaryObservation` with `is_stationary` (net window displacement below the
  pixel epsilon), `speed_estimate=None`, `dwell_seconds=None`, timestamp, ids,
  producer. Load `motion_threshold` for provenance but **do not apply** it (uncalibrated
  slice, §8b); reuse the heading taint handling. Threshold/window labelled provisional.
- **Outputs.** A derivation result (observations + taint-restart ids) mirroring
  `HeadingDerivation`.
- **Acceptance criteria.** A never-moving track → all `is_stationary=True`; steady
  motion above epsilon → `False`; bounded in-place jitter within the window → `True`
  (jitter-robust); tainted steps skipped + next clean observation flagged restart;
  `motion_threshold` not applied (varying it does not change `is_stationary`);
  deterministic.
- **Required tests.** Stationary vs moving vs jitter; window behaviour; taint skip +
  restart; `motion_threshold`-independence; empty/short track; determinism.
  (`tests/observations/test_stationary.py`.)
- **Explicit exclusions.** No dwell accumulation, no hysteresis, no zone logic, no m/s
  application, no calibrated speed claim, no scene-schema change.
- **Stop condition.** Stop when `StationaryObservation` derivation is complete and
  tested; do not start the reasoner.

### P2-U4 — Illegal-stopping temporal reasoner

- **Objective.** A deterministic reasoner that joins stationary + in-zone facts,
  accumulates dwell, and mints `ILLEGAL_STOPPING` `ConfirmedEvent`s, reusing the
  generic `RuleEngine`.
- **Why now.** With both evidence streams available (P2-U2/U3) and the provenance-
  carrying reasoner shape established (P2-U1), the temporal rule is the core new
  reasoning pattern of Phase 2.
- **Inputs.** `InZoneObservation` + `StationaryObservation` derivations; `RuleEngine`;
  the scene `illegal_stopping` params (`stationary_duration`, `motion_threshold`,
  optional new provisional `max_observation_gap`); `ConfirmedEvent`, `MeasuredValue`,
  `ModelRef`; run-level `models` (P2-U1 shape).
- **Exact scope.** A new module `rules/illegal_stopping.py` with: (a) a loader
  `illegal_stopping_parameters(scene)` (requires `stationary_duration`; loads
  `motion_threshold` for provenance; optional `max_observation_gap`); (b) a
  deterministic join of the two observation streams into a per-step stopped-in-zone
  signal keyed by `(camera, track, timestamp)`; (c) an `IllegalStoppingReasoner`
  mirroring `WrongWayReasoner` (per-track run, timestamp-driven, taint-restart reset,
  recovery close/abandon, `_confirm` minting the event with `models`, content-derived
  `event_id`). §9 semantics. If `max_observation_gap` is introduced, add it as a
  provisional entry under the existing `illegal_stopping` rule-parameter block in
  `configs/scenes/example-scene.yaml` (additive config data).
- **Outputs.** `ConfirmedEvent`s for sustained illegal stopping, with `dwell_seconds`
  measurement, `stationary_duration`/`motion_threshold` thresholds, populated `models`,
  scene hash, and `source_hypothesis_id`.
- **Acceptance criteria.** A track stationary-in-zone for `≥ stationary_duration`
  confirms exactly one event with correct `start_at`/`trigger_at`, `end_at=None`,
  `violation_type=ILLEGAL_STOPPING`, `dwell_seconds ≥ threshold`; a shorter dwell
  confirms nothing (countable abstention); leaving the zone or moving resets the run;
  taint prevents cross-switch confirmation; `motion_threshold` recorded not applied;
  ≥ 2 observations required; deterministic and order-independent; `models` carried
  from the run-level tuple.
- **Required tests.** Confirm on sustained dwell; no-confirm on short dwell; recovery
  (move / exit zone) reset; taint no-confirm; ≥2-observation floor; measurement/
  threshold contents; event-id determinism; order-independence; `motion_threshold`-
  independence of the decision. (`tests/rules/test_illegal_stopping.py`.)
- **Explicit exclusions.** No congestion suppression; no stationary-object re-
  association; no parking zones; no m/s application; no contract/ADR/master-spec
  change; no new package.
- **Stop condition.** Stop when the reasoner confirms/abstains correctly and
  deterministically on in-memory `TrackState` sequences; do not start orchestration.

### P2-U5 — Illegal-stopping pipeline orchestration + persistence integration

- **Objective.** A thin offline pipeline composing detector+tracker+in-zone+stationary
  derivations+reasoner into confirmed illegal-stopping events, persisted via the
  existing `EventStore`.
- **Why now.** The reasoner needs a composition to run over `FrameRecord`s and to reuse
  the P1-U11 persistence, exactly as `WrongWayPipeline` does for wrong-way.
- **Inputs.** `Detector`/`Tracker` abstractions; `DetectionAdapter`; `SceneConfig`;
  the P2-U2/U3 derivations; the P2-U4 reasoner; `EventStore`; the provenance-collection
  step (P2-U1).
- **Exact scope.** A new `pipeline/illegal_stopping.py` (`IllegalStoppingPipeline`)
  mirroring `WrongWayPipeline`: `process_frame` (detect+adapt+track, accumulate
  per-track history and run-level `ModelRef`s), `finalize` (derive in-zone+stationary
  per track, run a fresh `IllegalStoppingReasoner`, return events sorted by
  `(trigger_at, event_id)`), `reset`, `process`. Resolve eligible no-stopping zones at
  construction (fail-fast `SceneConfigurationError` if none). **Decision (E.8):** a
  **thin second pipeline**, not a generalized multi-rule runner — two violations do not
  justify premature generic orchestration (E.9). The library core imports **no** backend
  (boundary test analogue). Export it from `pipeline/__init__`.
- **Outputs.** Confirmed illegal-stopping events persisted as deterministic JSON with
  minimal manifests, identical in shape to the wrong-way slice.
- **Acceptance criteria.** Given stub/injected detections placing a track stationary in
  an eligible no-stopping zone long enough, the pipeline produces exactly one persisted
  `ILLEGAL_STOPPING` event + manifest; the pipeline adds no behaviour over calling the
  derivations+reasoner directly (equivalence test, like wrong-way's); deterministic on
  fresh instances and reset-replay; empty/legal input → no event; the library core
  imports no backend; `models` propagated.
- **Required tests.** End-to-end stub-driven confirmation; direct-vs-pipeline
  equivalence; determinism; empty/legal cases; backend-free import boundary; real
  `IouTracker` in-memory integration. (`tests/pipeline/test_illegal_stopping_pipeline.py`.)
- **Explicit exclusions.** No generic multi-rule runner; no SQLite/Parquet; no evidence
  rendering; no CLI changes yet (P2-U6); no new dependency.
- **Stop condition.** Stop when the illegal-stopping pipeline confirms + persists
  deterministically; do not start the recorded-clip/CLI unit.

### P2-U6 — Recorded-synthetic-video end-to-end verification + demo integration

- **Objective.** Prove the illegal-stopping slice end to end on a **recorded** synthetic
  clip through real ingestion + real tracker + real persistence, with an explicit
  scripted/injected detector; integrate a demo entry point if architecturally minimal.
- **Why now.** Parity with the wrong-way slice's honesty bar (P1-U12): a real decoded
  clip, real PTS, real `IouTracker`, real persistence — event driven by injected
  detections, never by falsely claiming RT-DETR fired.
- **Inputs.** A test-time PyAV-generated clip (no committed binary, no download); a
  purpose-built **test** `SceneConfig` whose no-stopping zone matches the small clip's
  pixel coordinates (test fixture only — the example scene's `zone-no-stop` is the
  analogue for the real/demo path); a scripted `StubDetector`; real `open_video`, real
  `IouTracker`, `EventStore`.
- **Exact scope.** A fixtures module (unique basename, e.g.
  `tests/pipeline/_stopping_fixtures.py`) that writes a clip of a rectangle that **moves
  into then holds position inside** the no-stopping zone (the `enter-then-stop` motion
  shape) and a matching scripted detector; an end-to-end test running the P2-U5 pipeline
  + persistence and asserting exactly one confirmed `ILLEGAL_STOPPING` event with a
  minimal manifest. Demo integration: extend the P1-U12 CLI/runner **only if** it is the
  smallest coherent change (e.g. a `--violation illegal_stopping` selector on the
  existing runner, or a sibling `run_illegal_stopping_slice`); if generalization would
  bloat the runner, add a thin sibling runner instead. Detector honesty preserved:
  the report records `detector_kind` (stub vs rtdetr); no claim RT-DETR produced the
  event.
- **Outputs.** A passing recorded-clip end-to-end test; a runnable offline demo path for
  illegal stopping; an honest run report.
- **Acceptance criteria.** Real decode (PTS) + real `IouTracker` + real reasoner + real
  persistence produce exactly one confirmed illegal-stopping event on the synthetic clip
  via injected detections; deterministic + byte-identical persisted files on replay; the
  report truthfully labels the detector; no real-accuracy claim; the real RT-DETR path
  remains only opt-in and is never claimed to fire on synthetic pixels.
- **Required tests.** Recorded-clip end-to-end confirmation + manifest; determinism/
  byte-identical persistence; (optional) opt-in real-RT-DETR integration test skipped by
  default, mirroring `test_slice_e2e_rtdetr.py`. (`tests/pipeline/test_illegal_stopping_e2e.py`.)
- **Explicit exclusions.** No model-tricking fixture; no real footage; no accuracy claim;
  no evidence rendering; no new dependency.
- **Stop condition.** Stop when the recorded-clip slice passes deterministically and the
  demo entry point runs offline; do not start README work.

### P2-U7 — README and public project-status synchronization

- **Objective.** Correct the stale public README and status docs so they match the
  implemented repository after P2-U1…U6.
- **Why now.** Only after the capability lands can the public status be corrected
  truthfully; the README is currently materially stale.
- **Inputs.** The current README and `docs/architecture.md`; the shipped Phase 1 + Phase
  2 capabilities.
- **Exact scope.** Update the README: test count (currently "485"), project-status
  narrative ("detector/tracker integration not started" is false), the capability table
  and roadmap (detector, tracker, persistence, pipeline, wrong-way **and** illegal-
  stopping are implemented), the mermaid implemented/planned coloring, the repository-
  structure tree (add `detector/`, `tracking/`, `pipeline/`, `persistence/`), and the
  provenance/evidence status (events now carry truthful `models`). Preserve every honest
  limitation: synthetic-only validation, no real footage, RT-DETR does not fire on
  synthetic pixels, no accuracy claim, congestion/ID-churn deferrals. Update
  `docs/architecture.md` pointers to reference this Phase 2 plan if needed.
- **Outputs.** A README and status docs consistent with the repository and with this
  plan.
- **Acceptance criteria.** No statement contradicts the implemented tree; the test count
  matches the actual suite; both violations and all shipped packages are represented; no
  new or softened accuracy claim is introduced; the RT-DETR-on-synthetic-pixels
  qualification and the external real-footage gate remain stated.
- **Required tests.** Governance tests remain green. A README/status governance
  assertion is added **only if strictly required** to keep the doc from silently
  drifting; otherwise no test change (documentation-only).
- **Explicit exclusions.** No runtime/source/contract/ADR/master-spec change; no roadmap
  claim that Phase 2 did not deliver.
- **Stop condition.** Stop when the public status is truthful and gates are green; Phase
  2 coding is complete.

### P2-V1 — Approved real-footage validation run (external gate; blocks nothing)

- **Objective.** Validate the shipped pipelines on approved/owned/licensed real footage.
- **Why now / positioning.** Runs **in parallel and outside** the coding chain; does not
  block P2-U1…U7. It converts "implemented + synthetic-verified" into "validated on real
  footage" **only** when the external preconditions are met.
- **Inputs.** Approved footage + a validated real `SceneConfig` (no-stopping zone) +
  institutional permission/ethics + data-handling plan; the already-shipped real RT-DETR
  + `IouTracker` + Phase 2 pipelines.
- **Exact scope.** Run the existing pipelines on approved footage; record honest results.
  No new code is required to run it.
- **Acceptance criteria.** A documented run on approved footage with results recorded;
  any accuracy language appears **only** here and only after this gate runs.
- **Required tests.** None in the coding suite (external activity).
- **Explicit exclusions.** No dependency of P2-U1…U7 on this gate; no footage acquisition
  on the coding critical path; no claim before the gate runs.
- **Stop condition.** N/A to the coding chain; it is a separate, gated activity.

### P2-R1 — Helmet dataset and licence readiness review (parallel research gate)

- **Objective.** Advance helmet dataset/licence readiness as governance/research only.
- **Why now / positioning.** Runs **in parallel**, blocks nothing, and starts **no**
  training and **no** CNN-vs-ViT work.
- **Inputs.** `registry/datasets/helmet-myanmar.yaml`; `docs/dataset-policy.md`; the
  ontology (turban/uncertain labels).
- **Exact scope.** Confirm and record HELMET licence + access terms; record turban/
  Indian-domain gaps and custom-annotation need; keep the entry `candidate` until gates
  resolve. No download; no checksum until authorized acquisition.
- **Acceptance criteria.** The registry entry reflects verified licence/access status
  (or an honest "unknown" with the verification attempt recorded); a short readiness note
  states what must resolve before any later training phase.
- **Required tests.** Registry governance tests remain green.
- **Explicit exclusions.** No dataset download; no training; no CNN-vs-ViT; no model
  selection; no Phase 2 critical-path dependency.
- **Stop condition.** Stop when licence/access status is recorded and the readiness note
  exists; training belongs to a later phase.

---

## 15. Dependency graph

```
P2-U1 (provenance)
  └─> P2-U2 (in-zone obs) ─┐
                            ├─> P2-U4 (illegal-stopping reasoner)
      P2-U3 (stationary) ──┘         └─> P2-U5 (pipeline + persistence)
                                             └─> P2-U6 (recorded-clip e2e + demo)
                                                    └─> P2-U7 (README/status)

P2-U2 and P2-U3 are independent of each other (both depend only on P2-U1's shape
being settled for the shared reasoner interface; they touch no provenance code and
may proceed in parallel once P2-U1 lands).

External / parallel (block nothing in the chain):
  P2-V1 (real-footage validation)  — gated on external permissions/footage
  P2-R1 (helmet licence readiness) — governance/research only
```

---

## 16. Phase 2 Definition of Done

- P2-U1…U7 complete; each unit's acceptance criteria and required tests pass.
- Wrong-way **and** illegal-stopping confirmed events carry truthful, sorted, de-
  duplicated `ModelRef`s (or honestly empty on stub runs), inherited by manifests;
  `weights_hash` is `None` everywhere (nothing fabricated).
- The illegal-stopping slice runs end to end offline on a **recorded** synthetic clip
  through real ingestion + real `IouTracker` + real reasoning + real persistence, with an
  explicit scripted/injected detector, deterministically and with byte-identical
  persisted files on replay.
- Quality gates green: `pip check`, `ruff check .`, `mypy src`, full `pytest -q`
  (all new tests passing; opt-in real-model tests still skipped by default).
- Public README/status docs match the implemented repository, preserving every honest
  limitation.
- No frozen contract, schema, ADR, or master-spec change; no new dependency; no new
  `src/trafficpulse` package (the sanctioned-package guard is unchanged and green).
- P2-V1 and P2-R1 remain outside the chain and did not block completion; no real-footage
  or training work occurred on the coding critical path.

---

## 17. Claims allowed after Phase 2

- "TrafficPulse implements two offline, deterministic violation reasoning slices —
  wrong-way and illegal-stopping — end to end from recorded video through real
  ingestion, real IoU tracking, typed observations, temporal reasoning, confirmed-event
  minting, and minimal-manifest persistence."
- "Confirmed events and their evidence manifests carry truthful detector/tracker model
  references (name + version; weights hashes not computed)."
- "The illegal-stopping reasoner is validated on synthetic tracks and a recorded
  synthetic clip; its dwell/zone/stationarity logic is deterministic and replayable
  from the observation log without a model."
- "The full illegal-stopping path (real decode, real tracker, real rules, real
  persistence) is verified on a generated clip with injected detections; genuine RT-DETR
  inference integrates end to end via the opt-in path."

## 18. Claims still forbidden after Phase 2

- No real-world / event-level accuracy, precision, recall, or false-events-per-hour
  claim (requires P2-V1 on approved footage).
- No claim that RT-DETR (or any real detector) produced a confirmed event on synthetic
  pixels.
- No claim of congestion-robust or ID-churn-robust illegal-stopping (both explicitly
  deferred).
- No calibrated-speed / m/s-thresholded claim (uncalibrated slice; `motion_threshold`
  recorded not applied).
- No parking-zone illegal-stopping claim (scene `ZoneType` has no parking member).
- No helmet / CNN-vs-ViT / ANPR / speeding / red-light / review / penalty capability
  claim.
- No production/enforcement-readiness claim.

---

## 19. Handoff criteria

- **To real-footage validation (P2-V1).** The Phase 2 pipelines run offline and
  deterministically; a matching real `SceneConfig` shape is defined; institutional
  permission/ethics + footage + data-handling plan are the only remaining (external)
  gates. No code change is needed to run validation.
- **To evaluation-harness work.** Confirmed events carry the identity, rule id/version,
  thresholds, measurements, scene hash, and `models` needed for event-level matching;
  the deterministic `EventStore` provides a stable substrate for an event-level
  evaluation harness (a later phase), and ADR-004 run-identity accounting is respected.
- **To CNN-vs-ViT helmet research.** P2-R1 records the HELMET licence/access status and
  the turban/Indian-domain gaps; training may begin only in a later phase once those
  gates resolve — never on the Phase 2 critical path.

---

## 20. Stop conditions (phase-level)

- Stop at the end of P2-U7 with gates green and public status corrected.
- **Stop and report** (do not silently change a contract) if any unit discovers a
  genuine contradiction requiring a frozen-contract, schema, ADR, or master-spec change.
- Do not begin red-light, helmet training, ANPR, speeding, review, or penalty work.
- Do not add SQLite/Parquet or an evidence rendering engine in this phase.
- Do not let footage acquisition, institutional permission, dataset licences, or
  CNN-vs-ViT results gate completion of P2-U1…U7.
