# TrafficPulse — Phase 4 plan

**Phase name:** Phase 4 — Association, Confidence Aggregation, and Classifier-Driven Violations (Triple Riding + No-Helmet)

- **Status:** Authoritative Phase 4 unit plan (planning document; no Phase 4 source
  implemented yet).
- **Date:** 2026-07-12
- **Authority:** This is the **authoritative Phase 4 unit plan**. It governs the
  `P4-U#` identifier namespace only. It does **not** supersede the Phase 0-F (`U#`),
  Phase 1 (`P1-U#`), Phase 2 (`P2-U#`), or Phase 3 (`P3-U#`) plans; those remain
  governed by their own documents and are not rewritten here.
- **Canonical architecture reference:** [`docs/architecture-review.md`](architecture-review.md)
  remains THE canonical architecture reference; this plan interprets and sequences it
  for Phase 4 without modifying it or the master spec
  ([`TRAFFICPULSE_MASTER_SPEC.md`](../TRAFFICPULSE_MASTER_SPEC.md)).
- **Basis:** Implements the accepted architectural design review (2026-07-12):
  **association before helmet/triple riding**; quality-weighted confidence aggregation
  before any classifier-driven violation; capability-first sequencing; composition over
  inheritance; frozen contracts preserved; no speculative architecture. See
  [`docs/architecture.md`](architecture.md) for the cross-phase roadmap index.

---

## 1. The reasoning chain this phase reinforces

```
Perception → Observations → Scene Semantics → Dynamic Context → Rule Reasoning → Evidence → Human Review
```

Phase 4 is where the **Association** link of the flow (`Detection → TrackState →
Association → Observation`, architecture-review §14) is implemented for the first
time, and where **Rule Reasoning** gains **quality-weighted confidence aggregation**
for noisy per-frame classifier evidence (architecture-review §13). Triple riding and
no-helmet are both **conclusions** derived from associated riders + temporally
aggregated classifier confidence under scene semantics — not single-frame detections.

---

## 2. Relationship to Phases 0–3

Phases 0-F/1/2 froze the contracts and delivered wrong-way + illegal-stopping. Phase 3
generalized the reasoner/pipeline infrastructure (by composition), added the dynamic
traffic-context stream and red-light jumping, and delivered the observation-log
substrate and the event-level evaluation harness. Phase 4 builds **directly** on the
Phase 3 generalized bases:

- The **generalized reasoner base** (P3-U1) already accepts injected predicates and a
  confirmation policy — Phase 4 adds a **quality-weighted accumulation** policy for
  classifier evidence without touching the base's lifecycle guarantees.
- The **generalized pipeline base** (P3-U2) accepts new finalize strategies — the
  association + classifier derivations plug in as one.
- The **observation log** (P3-U6) persists the new observation variants; the
  **evaluation harness** (P3-U7) scores the two new rules.

Phase 4 introduces the project's **first learned components beyond the detector** (a
rider-occupancy / helmet-state classifier), behind a seam mirroring the P1-U6/U7
`Detector` pattern. Their **training** is dataset-gated (§12); their **integration and
reasoning** are offline-testable against a stub classifier without any dataset.

---

## 3. Verified starting point (expected at Phase 4 entry)

Phase 4 begins only after Phase 3's Definition of Done holds. Expected state:

- Three violation slices (wrong-way, illegal-stopping, red-light) run offline and
  deterministically on the generalized reasoner/pipeline bases.
- **The `Association` contract exists but has no derivation.** `contracts/association.py`
  defines `Association` (`rider_of_motorcycle`, `head_of_rider`, `plate_of_vehicle`,
  with confidence + interval); `AssociationType` is a frozen enum. This is the one box
  in the §14 data flow never implemented.
- **The classifier-violation contracts exist.** `RiderCountObservation`
  (`rider_count`, optional `motorcycle_track_id`) and `HelmetStateObservation`
  (`helmet_state` over the frozen 4-label `{helmet, no_helmet, turban, uncertain}`,
  optional `rider_slot`, `crop_height_px`) are frozen U2 variants. `RiderSlot`
  (`driver`/`pillion`/`third`/`unknown`) and `ViolationType.TRIPLE_RIDING` /
  `NO_HELMET` exist.
- **The scene rule-parameter blocks exist.** `configs/scenes/example-scene.yaml`
  already carries `triple_riding` (`rider_count_threshold: 3`,
  `persistence_window_frames: unset`) and `no_helmet` (`min_confirming_observations`,
  `candidate_persistence_frames: unset`, `abstain_on_uncertain: unset`).
- **The ontology + rule-layer mapping are frozen.** `docs/ontology.md` /
  `configs/ontology.yaml` fix the helmet 4-label scheme and the mapping
  `turban → exempt (no event)`, `uncertain → abstain` — a **rule-layer** decision, not
  encoded in contracts.
- **`ConfirmedEvent.confidence` is still unset** by the shipped geometry-only
  reasoners; a component confidence breakdown (architecture-review §13) has never been
  produced.
- **Sanctioned packages** are the Phase 3 set plus `evaluation`. Phase 4 adds
  **`association`** and **`classifier`** to the allow-list in
  `tests/docs/test_adr_pack.py`, each in its introducing unit's commit.

---

## 4. Phase 4 objectives

- **A. Association derivation.** Implement rider↔motorcycle (and plate↔vehicle)
  association behind the frozen `Association` contract — geometric association first
  (containment/overlap of person tracks within a motorcycle bbox), with an occupancy-
  classifier alternative left as a documented A/B option.
- **B. Quality-weighted confidence aggregation.** Extend the P3-U1 reasoner base with a
  **weighted temporal accumulation** confirmation policy (log-odds or EMA, per
  architecture-review §13) and populate the `ConfirmedEvent.confidence` **component
  breakdown** — the mechanism classifier-driven violations require.
- **C. Fifth violation: triple riding.** Rider-count observation derivation (from
  associations) + a sustained `rider_count ≥ threshold` (K-of-N good-visibility)
  reasoner, with occlusion/oscillation abstention.
- **D. Sixth violation: no-helmet riding.** A helmet-state classifier behind a seam +
  per-rider-slot quality-weighted temporal voting, with `turban → exempt`,
  `uncertain → abstain` at the rule layer.
- **E. The mandatory CNN-vs-ViT helmet experiment** (architecture-review §12,
  master-spec §4) — the project's required genuine ViT contribution — as a
  reproducible experiment, **dataset-gated** and separable from the reasoning
  integration.
- **F. Offline + deterministic.** All reasoning is offline, model-free in the decision
  predicate path where a threshold is applied, and replays bit-exactly from the
  observation log; classifier confidence enters as **observation data**, never as a
  hidden model call inside a rule.

---

## 5. Architectural invariants (preserved)

1. **Perception ↔ reasoning separation.** The helmet/occupancy classifier sits behind a
   seam and emits **observations** (`helmet_state`, `rider_count`); rules consume only
   those observations. No rule calls a model.
2. **Model-free deterministic replay.** Classifier confidence is captured in the
   observation and persisted to the observation log; replaying the log reproduces the
   decision byte-for-byte without re-running the model.
3. **Frozen contracts.** `Association`, `RiderCountObservation`,
   `HelmetStateObservation`, `HelmetState`, `RiderSlot`, `ConfirmedEvent.confidence`
   (the field already exists) are unchanged. Phase 4 expects **no** frozen-contract
   change.
4. **≥ 2 observations to confirm; every non-confirmation is a countable abstention.**
   Preserved via the generalized base; `uncertain`/occlusion/oscillation all route to
   logged abstentions.
5. **Determinism.** No wall-clock; content-derived ids; deterministic ordering; the
   weighted accumulator is a pure, order-independent function of the observation set.
6. **Composition over deep inheritance.** The weighted-accumulation confirmation policy
   is an **injected** policy on the P3-U1 base; the classifier is an **injected** seam
   implementation; association is a **collaborator** — no deep subclass hierarchy.
7. **No speculative architecture.** `association` and `classifier` packages appear in
   the units that first implement them; the classifier seam is generic enough to host
   the later ROI signal classifier and rider-occupancy variant, but only helmet is
   implemented now.
8. **Honesty + ViT integrity.** The CNN-vs-ViT experiment is pre-registered, fair, and
   reports negatives/ties as such (§12); no dataset is downloaded before its registry
   licence/access gate resolves; no accuracy claim before evaluation on approved data.

---

## 6. Design decisions

- **D1 — Association method (geometric first).** The first association derivation is
  **geometric**: a person track is a rider of a motorcycle track when its bottom-center
  / bbox sits within (or sufficiently overlaps) the motorcycle bbox over a sustained
  window, with confidence from overlap ratio and stability. The occupancy-classifier
  alternative (a 1/2/3+ crop classifier, architecture-review §5d) is a **documented A/B
  option** behind the same `Association`/observation seam, deferred unless the geometric
  path proves inadequate — matching the spec's "capability, not fixed mechanism"
  treatment.
- **D2 — Confidence aggregation lives in the reasoner base, as a policy.** Weighted
  accumulation (log-odds/EMA, chosen per rule and unit-tested against synthetic
  observation sequences) is a confirmation **policy** injected into the P3-U1 base,
  not a new engine. It populates the `ConfirmedEvent.confidence` component breakdown
  (detector, classifier, association, temporal consistency, crop quality) — stored as
  components, **never** relabelled a calibrated probability unless calibration is
  demonstrated (architecture-review §13).
- **D3 — Classifier seam (offline-testable).** A `HelmetClassifier` ABC + a scripted
  `StubHelmetClassifier` (mirroring `StubDetector`/`StubTracker`) make the no-helmet
  reasoning path fully testable offline with **no** dataset and **no** weights; a real
  backend and the CNN-vs-ViT experiment are separate, dataset-gated units.
- **D4 — Turban/uncertain at the rule layer.** `turban → exempt (no event)` and
  `uncertain → abstain` are applied by the no-helmet reasoner per the frozen ontology,
  not by the classifier or the contracts.
- **D5 — Crop-quality gating and abstention.** Sub-minimum head-crop height, motion
  blur, slot instability, oscillating rider count, and overlapping bikes all route to
  countable abstentions, never confirmations (architecture-review §5e/§5d).

---

## 7. Ordered unit cards

Dependency order: **P4-U1 → P4-U2 → {P4-U3, P4-U4} → P4-U5**. P4-U3 (triple) and P4-U4
(no-helmet reasoning) both depend on association + confidence aggregation and may
proceed in parallel; P4-U5 (the CNN-vs-ViT experiment) is dataset-gated and separable.

### P4-U1 — Association derivation (rider↔motorcycle; plate↔vehicle)

- **Objective.** Deterministically derive `Association` links (rider↔motorcycle first;
  plate↔vehicle as a small extension) from `TrackState` sequences + scene data, behind
  the frozen `Association` contract, with confidence + interval.
- **Why now.** Association is the unimplemented §14 layer that both triple riding and
  no-helmet require; it is pure geometry over frozen inputs.
- **Inputs.** Ordered `TrackState`s (person + motorcycle + plate classes);
  `geometry` (containment/overlap); `Association`, `AssociationType`, `Confidence`,
  `TimeInterval`.
- **Exact scope.** A new `association/` package (e.g. `association/riders.py`) deriving
  rider↔motorcycle associations by sustained geometric containment/overlap, with
  confidence from overlap + stability and an `interval` over the associated window;
  taint handling reused from the observation derivations. Add `association` to the
  allow-list in this commit. Plate↔vehicle association is a thin analogous extension
  (deferred to a follow-up sub-unit if it would bloat the card).
- **Outputs.** Deterministic `Association` records per `(motorcycle, rider)` link.
- **Acceptance criteria.** A rider consistently within a motorcycle bbox associates with
  calibrated-by-overlap confidence; transient/ambiguous overlaps abstain (no link or a
  low-confidence link flagged for the rule layer); deterministic + order-independent;
  no contract change.
- **Required tests.** Single/multi-rider association; overlap-threshold behaviour;
  ambiguous-overlap abstention; taint handling; determinism.
  (`tests/association/test_riders.py`.)
- **Explicit exclusions.** No occupancy classifier (D1); no reasoning; no rider-count
  observation yet; no contract change.
- **Stop condition.** Stop when rider↔motorcycle associations derive deterministically;
  do not start confidence aggregation.

### P4-U2 — Quality-weighted confidence aggregation (reasoner-base policy)

- **Objective.** Add a weighted temporal-accumulation confirmation policy to the P3-U1
  reasoner base and populate the `ConfirmedEvent.confidence` component breakdown, so
  classifier-driven violations confirm on quality-weighted evidence rather than a raw
  boolean predicate.
- **Why now.** Both remaining violations consume noisy per-frame classifier confidence;
  the binary predicate + timestamp-elapsed model is insufficient (architecture-review
  §13). It is a base capability both P4-U3 and P4-U4 reuse.
- **Inputs.** The P3-U1 reasoner base; `ConfirmedEvent.confidence` (frozen field);
  `Confidence` primitive; synthetic observation sequences (`synth/`).
- **Exact scope.** A weighted-accumulation `ConfirmationPolicy` (log-odds or EMA,
  documented per rule) injected into the P3-U1 base, plus a confidence-breakdown
  assembler that fills the `ConfirmedEvent.confidence` components from the accumulated
  evidence. **Composition only** — no change to the base lifecycle or to the existing
  geometry-only policies; wrong-way/illegal-stopping/red-light behaviour is untouched.
- **Outputs.** A reusable weighted-accumulation policy + confidence-breakdown assembler.
- **Acceptance criteria.** Weighted accumulation is deterministic and order-independent;
  ≥2-observation floor and abstention guarantees preserved; the confidence breakdown is
  stored as components and never labelled a calibrated probability; the three existing
  reasoners are byte-identical (they do not use the new policy); no contract change.
- **Required tests.** Weighted confirm/abstain on synthetic confidence sequences;
  order-independence; component-breakdown contents; regression that existing reasoners
  are unchanged. (`tests/rules/test_confidence_aggregation.py`.)
- **Explicit exclusions.** No calibration claim; no probability relabelling; no
  contract change; no classifier here.
- **Stop condition.** Stop when the weighted policy + breakdown are complete and the
  existing reasoners are provably unchanged; do not start a violation reasoner.

### P4-U3 — Rider-count observation + triple-riding reasoner

- **Objective.** Derive `RiderCountObservation` per motorcycle (from P4-U1
  associations) and confirm `TRIPLE_RIDING` on a sustained `rider_count ≥ threshold`
  (K-of-N good-visibility), composed on the P3-U1 base with the P4-U2 policy.
- **Why now.** Association (P4-U1) and confidence aggregation (P4-U2) are the two
  prerequisites; triple riding is a direct application.
- **Inputs.** P4-U1 associations; `RiderCountObservation`; the scene `triple_riding`
  params (`rider_count_threshold`, `persistence_window_frames`); the P3-U1 base +
  P4-U2 policy; P3-U2 pipeline base; `EventStore`.
- **Exact scope.** A rider-count derivation in `observations/` (count of confident
  rider associations per motorcycle per usable frame, with `motorcycle_track_id`), and a
  `rules/triple_riding.py` reasoner (sustained count ≥ threshold over K-of-N frames,
  abstaining on oscillation/overlap/partial visibility), a thin pipeline configuration
  of the P3-U2 base, persistence, and a recorded-clip e2e (injected detections +
  scripted associations), mirroring prior slices.
- **Outputs.** Confirmed triple-riding events with rider-count measurements + confidence
  breakdown, persisted deterministically.
- **Acceptance criteria.** Sustained count ≥ threshold confirms one event; oscillating /
  partially-visible counts abstain (countable); ≥2-observation floor; taint prevents
  cross-switch confirmation; deterministic + byte-identical persistence; recorded-clip
  e2e passes; no contract change.
- **Required tests.** Confirm on sustained ≥3; abstain on oscillation/overlap; K-of-N
  window; taint; determinism; recorded-clip e2e. (`tests/rules/test_triple_riding.py`,
  `tests/pipeline/test_triple_riding_e2e.py`.)
- **Explicit exclusions.** No occupancy classifier (geometric associations only in this
  slice); no contract change; no new package.
- **Stop condition.** Stop when triple riding confirms/abstains deterministically end to
  end; do not start no-helmet.

### P4-U4 — Helmet-state classifier seam + no-helmet reasoner

- **Objective.** Add a `HelmetClassifier` seam (+ scripted stub) emitting
  `HelmetStateObservation` per rider slot, and a no-helmet reasoner performing per-slot
  quality-weighted temporal voting with `turban → exempt`, `uncertain → abstain`,
  composed on the P3-U1 base + P4-U2 policy — all offline-testable without a dataset.
- **Why now.** With association (rider slots) and confidence aggregation available, the
  no-helmet **reasoning** path can be built and fully tested against a stub classifier,
  independent of the dataset-gated experiment (P4-U5).
- **Inputs.** P4-U1 associations (rider slots); `HelmetStateObservation`, `HelmetState`,
  `RiderSlot`; the frozen ontology mapping; the scene `no_helmet` params; the P3-U1 base
  + P4-U2 policy; P3-U2 pipeline base; `EventStore`.
- **Exact scope.** A new `classifier/` package with a `HelmetClassifier` ABC + a
  scripted `StubHelmetClassifier` (no weights, no dataset), a head-crop/rider-slot
  derivation emitting `HelmetStateObservation` (with `crop_height_px` for quality
  weighting), and a `rules/no_helmet.py` reasoner (per-slot quality-weighted vote;
  `turban → exempt`; `uncertain`/sub-min-crop/blur → abstain), a thin pipeline
  configuration, persistence, and a recorded-clip e2e with a scripted classifier. Add
  `classifier` to the allow-list in this commit. The classifier seam keeps all
  framework-native types inside the backend (boundary preserved).
- **Outputs.** Confirmed no-helmet events (per slot) with quality-weighted confidence
  breakdown, persisted deterministically; turban/uncertain correctly exempt/abstain.
- **Acceptance criteria.** A sustained no-helmet slot confirms; a turban slot never
  confirms (exempt); uncertain/low-quality slots abstain (countable); per-slot voting is
  deterministic + order-independent; ≥2-observation floor; boundary test (no framework
  type escapes the seam); recorded-clip e2e passes; no contract change.
- **Required tests.** Confirm on sustained no-helmet; turban-exempt; uncertain-abstain;
  crop-quality gating; per-slot independence; seam boundary; determinism; recorded-clip
  e2e. (`tests/rules/test_no_helmet.py`, `tests/classifier/test_helmet_seam.py`,
  `tests/pipeline/test_no_helmet_e2e.py`.)
- **Explicit exclusions.** No real classifier weights; no dataset; no CNN-vs-ViT
  training (that is P4-U5); no ANPR; no contract change.
- **Stop condition.** Stop when no-helmet reasoning confirms/exempts/abstains
  deterministically against the stub classifier; the real model + experiment are P4-U5.

### P4-U5 — Mandatory CNN-vs-ViT helmet experiment (dataset-gated)

- **Objective.** Execute the pre-registered CNN-vs-ViT helmet-state classification
  experiment (architecture-review §12) — the project's required genuine ViT
  contribution — producing a fair, reproducible comparison, and (if it passes) a real
  `HelmetClassifier` backend behind the P4-U4 seam.
- **Why now / positioning.** This is the one Phase 4 unit that **requires datasets**; it
  is **gated** on the helmet dataset/licence readiness (the P2-R1 successor) and runs
  separably from P4-U1…U4, mirroring how P1-U7 (real RT-DETR backend) followed the
  P1-U6 seam. No reasoning path depends on it (the stub classifier covers reasoning).
- **Inputs.** The pre-registered §12 design (ResNet-50 vs DeiT-Small, 3 seeds, video-
  level split, macro-F1 + calibration + latency/VRAM, McNemar + bootstrap CI); the
  helmet dataset registry entries (licence/access must be **resolved** first); the
  `experiments/` area; the P4-U4 `HelmetClassifier` seam.
- **Exact scope.** A reproducible experiment under `experiments/helmet_cnn_vit/`
  (config in, results JSON out, git-tagged — ADR-002 file-based experiment tracking),
  faithfully implementing the §12 protocol (leakage-safe splits, fair augmentation,
  pre-committed statistics/interpretation), plus — only if a model clears the protocol —
  a real `HelmetClassifier` backend behind the P4-U4 seam as an **optional extra**
  (lazily imported; base install/CI unchanged; no weights committed).
- **Outputs.** A pre-registered, reproducible experiment with honestly reported results
  (including ties/negatives); optionally a real classifier backend behind the seam.
- **Acceptance criteria.** The experiment is reproducible from committed configs +
  seeds; splits are video-level leakage-safe; results report macro-F1 / calibration /
  latency / VRAM with the pre-committed statistics; a difference is claimed only under
  the §12 sign-consistency + bootstrap-CI rule; **no dataset is downloaded before its
  registry licence/access gate is resolved**; the default CI/test suite passes without
  the datasets or weights.
- **Required tests.** Experiment-harness unit tests on tiny synthetic fixtures
  (leakage-guard, metric computations, determinism); opt-in real-training run skipped by
  default. (`tests/experiments/test_helmet_experiment_harness.py`.)
- **Explicit exclusions.** No dataset download without a resolved registry gate; no
  weights committed; no accuracy claim outside the experiment's own reported protocol;
  no dependency of P4-U1…U4 on this unit.
- **Stop condition.** Stop when the experiment is reproducibly executed and reported; if
  the dataset gate is unresolved, record "reasoning integration complete (stub); real
  helmet model pending dataset/licence resolution" — do not fabricate a result.

---

## 8. Dependency graph

```
P4-U1 (association) ──> P4-U2 (confidence aggregation) ──┬─> P4-U3 (triple riding)
                                                         └─> P4-U4 (no-helmet reasoning, stub classifier)
                                                                     └─> P4-U5 (CNN-vs-ViT experiment; dataset-gated)

External / parallel gates (block only claims / the experiment, not P4-U1…U4):
  Helmet dataset + licence readiness — the P2-R1 successor (gates P4-U5)
  Real-footage validation             — carried forward from Phase 2
```

## 9. Implementation ordering

1. **P4-U1** — association first (the unimplemented §14 layer both violations need).
2. **P4-U2** — confidence aggregation as a base policy (both violations need it).
3. **P4-U3** and **P4-U4** — triple riding and no-helmet *reasoning* in parallel, both
   offline-testable (stub classifier for no-helmet).
4. **P4-U5** — the dataset-gated CNN-vs-ViT experiment, separable and non-blocking to
   the reasoning path.

## 10. Phase 4 Definition of Done

- P4-U1…U4 complete; each unit's acceptance criteria and required tests pass. P4-U5 is
  complete **or** honestly recorded as pending the dataset/licence gate (reasoning
  integration complete via the stub).
- Association is derived deterministically behind the frozen `Association` contract.
- Confidence aggregation is available as a base policy and populates the
  `ConfirmedEvent.confidence` component breakdown; the three prior reasoners remain
  byte-identical.
- Triple riding and no-helmet run end to end offline on recorded synthetic clips through
  real ingestion + tracking + association + classifier seam (stub) + weighted reasoning
  + persistence, deterministically and with byte-identical persisted files on replay.
- Turban is exempt and uncertain abstains at the rule layer, per the frozen ontology.
- Quality gates green: `ruff`, `mypy src`, full `pytest -q` (opt-in real-model tests
  skipped).
- **No frozen contract, schema, ADR, or master-spec change.** `association` and
  `classifier` added to the allow-list.

## 11. Claims allowed after Phase 4

- "TrafficPulse implements five offline, deterministic violation slices — wrong-way,
  illegal-stopping, red-light, triple-riding, and no-helmet — with triple-riding and
  no-helmet derived from **associated riders** and **quality-weighted temporal
  classifier evidence**, not single-frame detections."
- "Confirmed events carry a component confidence breakdown (detector, classifier,
  association, temporal consistency, crop quality)."
- "No-helmet reasoning applies the frozen ontology's `turban → exempt`,
  `uncertain → abstain` at the rule layer and is validated offline against a scripted
  classifier."
- If P4-U5 passes: "A pre-registered, leakage-safe CNN-vs-ViT helmet experiment was
  executed and reported per architecture-review §12."

## 12. Claims still forbidden after Phase 4

- No real-world / event-level accuracy claim on real footage (external gate).
- No claim of a real helmet-classifier result until P4-U5 runs on licence-cleared data;
  until then the no-helmet path is validated only against the stub classifier.
- No calibrated-probability confidence claim (components only, unless calibration is
  demonstrated).
- No speeding capability claim (Phase 5); no ANPR / review / penalty claim.
- No production/enforcement-readiness claim.

## 13. Handoff criteria

- **To Phase 5 (calibration + speeding).** Five violations run on the generalized bases;
  the observation log persists all their observation variants; the evaluation harness
  scores all five rules. Calibration is the remaining capability, needed by speeding and
  to retro-upgrade the provisional pixel gates.
- **To real-footage / dataset validation.** The classifier seam accepts a real backend
  with no reasoning change; the CNN-vs-ViT experiment is reproducible once datasets clear.

## 14. Stop conditions (phase-level)

- Stop at the end of P4-U4 (with P4-U5 done or honestly pending) and gates green.
- **Stop and report** if any unit discovers a genuine need for a frozen-contract,
  schema, ADR, or master-spec change (none is anticipated).
- Do not begin calibration/ground-plane or speeding work (Phase 5).
- Do not download any dataset before its registry licence/access gate is resolved.
- Do not let dataset/licence resolution or real-footage acquisition gate completion of
  P4-U1…U4.
