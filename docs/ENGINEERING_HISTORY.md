# TrafficPulse — Canonical Engineering History

**Document status:** Canonical engineering history and architectural handoff brief.
**Covers:** 2026-07-07 (first commit) → 2026-07-30.
**Repository state at time of writing:** 71 commits, branch `feature/h9-case-management`, 2016 backend tests + 354 frontend tests passing.

---

## 0. How to read this document, and how far to trust it

This document is assembled from **primary artifacts in the repository** — the full
git log, `TRAFFICPULSE_MASTER_SPEC.md`, `docs/architecture-review.md`, the four
ADRs, the six phase plans, `README.md`, the dataset registry, the CI workflow, and
the source tree itself — plus direct participation in the most recent milestones.

Confidence is not uniform, and pretending otherwise would defeat the purpose:

| Period | Basis | Confidence |
|---|---|---|
| Phase 0-R / 0-F, Phases 1–3, Phase 4, H1–H8, v1.1 U1–U3 | Reconstructed from commits, plans, ADRs, code, and engineering notes | High on *what* and *why-as-recorded*; lower on undocumented conversation |
| Overlay framework, overlay production integration, overlay polish, analyst review workspace (Phase 2 UI), H9 | Direct participation | Very high |

Where something exists **only as discussion** and never reached the repository, it
is called out explicitly in §11. Where the repository's own documentation is now
**wrong**, that is called out in §9.4 rather than quietly repeated.

Status vocabulary used throughout:

- **Implemented** — code in the tree, covered by tests.
- **Planned** — a written plan exists; no code.
- **Rejected** — considered and deliberately declined, with a recorded reason.
- **Deferred** — accepted as desirable, explicitly postponed.
- **Superseded** — was true, has been replaced; the earlier state is preserved as history.

---

## 1. What TrafficPulse is

TrafficPulse is an **evidence-first, multi-violation traffic-detection and
simulated-penalty system** for fixed-camera roadside video in Indian urban traffic.
It is an academic capstone engineering project, not a production enforcement
system, and it makes **no validated real-world accuracy claim**.

### 1.1 The central commitment

> **A single model output or single-frame detection is never a violation.**

This one sentence generates most of the architecture. Entities are tracked over
time; typed observations are derived; scene geometry and explicit rules are applied;
evidence accumulates; confirmed events become reviewable cases; only a
human-approved case yields a *simulated* penalty.

### 1.2 The canonical pipeline

```
Detection → TrackState → Association → Observation → TemporalState
  → ViolationHypothesis → ConfirmedEvent → EvidencePackage → ReviewCase → SimulatedPenalty
```

This is `architecture-review.md` §14 and it has survived unchanged from Phase 0 to
today. Every layer is a frozen pydantic contract in `src/trafficpulse/contracts/`.

### 1.3 The six locked violations

1. No-helmet riding — **Implemented**
2. Triple riding — **Implemented**
3. Red-light jumping — **Planned** (Phase 3, partially built)
4. Wrong-way driving — **Implemented**
5. Illegal stopping / parking — **Implemented**
6. Speeding — **Planned**, feasibility-gated (Phase 5)

Scope has never been silently expanded or reduced. Accident / stalled-vehicle
detection is **stretch-only** and remains untouched.

---

## 2. Design philosophy — the ten principles and what they actually caused

From `TRAFFICPULSE_MASTER_SPEC.md` §3. These are not decoration; each one has left
a visible mark on the codebase.

1. **Perception and violation reasoning are separate layers.** → `rules/` imports
   nothing from `detector/` or `classifier/`. Enforced by boundary tests.
2. **A model detection alone does not equal a violation.** → every reasoner
   requires sustained temporal support.
3. **Temporal evidence, track history, scene context, explicit rules.** →
   `TemporalRunReasoner` and the `RuleEngine` FSM.
4. **Model outputs, observations, hypotheses, events, cases, penalties stay
   conceptually distinct.** → ten separate contracts, no shortcuts between them.
5. **Evidence-first: every confirmed event explainable from stored evidence.** →
   `EvidenceManifest`, `measurements`, `thresholds`, `models` provenance.
6. **Human review mandatory before simulated penalty.** → H9's review lifecycle;
   `ReviewStatus` gates everything downstream.
7. **Model-specific implementations behind stable interfaces.** → `Detector`,
   `Tracker`, `HelmetClassifier` protocols; backends are optional extras.
8. **Scene geometry and thresholds are versioned configuration, not constants.** →
   `SceneConfig` + `scene_config_hash` stamped onto every event.
9. **Dataset provenance, licensing, splits, leakage, reproducibility are
   first-class.** → `registry/` with per-entry licence status; **no dataset has
   ever been downloaded**.
10. **Privacy and redaction designed into the evidence workflow.** → **Deferred**;
    contracts anticipate it, no runtime exists.

### 2.1 Derived philosophy that emerged during implementation

These were not in the spec but became load-bearing conventions:

- **Never fabricate a value.** An unmeasured quantity is `null`, never `0`. This
  appears in `ConfidenceBreakdown` (`aggregate` is deliberately `None`), in
  `JobStatusResponse` (`fps` is null before two frames), in `StatGrid` (renders
  `—`), and in the overlay (a component the reasoner did not measure is omitted
  from the banner rather than drawn as `0%`).
- **Preserve uncertainty deliberately.** Estimates are not promoted to facts.
  `architecture-review.md` §1 states this as a document-level commitment and the
  registry encodes it as vocabulary (`verified` / `unknown` / `unconfirmed`).
- **Prove before concluding.** Established after a specific correction (§8.1): a
  claim about what footage contains must be demonstrated on the frames, not
  inferred from aggregate statistics.
- **No speculative structure.** A package appears only when a unit implements code
  for it. Enforced mechanically (§10.1).
- **Composition, not inheritance.** Accepted at the 2026-07-12 design review and
  applied in P3-U1/P3-U2.

---

## 3. Chronological history

### Phase 0-R — Research and architecture review (2026-07-07)

**Commits:** `ae72b00`, `84e45e5`

The project began with `TRAFFICPULSE_MASTER_SPEC.md` (680 lines) and a Phase 0
plan. The spec was then subjected to a **critical architecture review**
(`docs/architecture-review.md`, 393 lines), which is the single most important
document in the repository. It reached a **GO WITH CHANGES** recommendation.

The review did something unusual and valuable: it **criticised the spec that
commissioned it**, in writing, in sections that survive today.

**Strong decisions it preserved unchanged:** the ten principles; the pre-gated
speeding scope; the ViT anti-branding clause; the research-first anti-scaffold
workflow; dataset-registry duties and "no download before licence review"; the
leakage section; hardware realism; the honest 10–14-day framing.

**Contradictions it resolved:**
- The external **"real-time" tagline overstated the spec** → align the tagline to
  the spec, not vice versa. This became ADR-003.
- Spec §2.2 **hard-coded a mechanism** for triple riding → the review reframed it
  as a *capability*, not a fixed mechanism.

**Things it identified as harder than they appear** (all of which proved true):
rider–vehicle association in dense traffic; track persistence for minutes-long
stationary vehicles; congestion suppression; night signal-state classification;
two-line motorcycle plates; event-level ground-truth annotation cost.

**Things it declared over-engineered** — **Rejected**:
- Two separate dashboards → merge into one analytics/evaluation view.
- "Experiment tracking support" → keep file-based, no tracking service.

**The largest omission it found:** there was **no evaluation-footage acquisition
plan**. This was elevated to a first-class dependency (§9, risk #1).

**Scope it deferred:** night as a *supported condition* (robustness analysis only);
any detector-family CNN-vs-ViT comparison (optional, gated); accident/stalled
detection (stretch).

### Phase 0-F — Foundations and interface freezing (2026-07-07 → 2026-07-08)

**Commits:** `4b1b847` (U1) → `c48d214` (U6)

**Principle:** *interface-freezing only* — types, schemas, policies, decisions.
Nothing with runtime behaviour, no ML dependencies.

| Unit | Deliverable | Status |
|---|---|---|
| U1 | Repository baseline, pyproject, CI, tooling | Implemented |
| U2 | Domain contracts + exported JSON schemas | Implemented |
| U3 | Label ontology (`configs/ontology.yaml`) | Implemented |
| U4 | Dataset registry + evaluation protocol | Implemented |
| U5 | Scene-configuration contract + stable hashing | Implemented |
| U6 | Architecture decisions (ADR-001…004) | Implemented |

**U3's most consequential decision:** the helmet ontology is **four labels**, not
two — `{helmet, no_helmet, turban, uncertain}`. The `turban` label exists because
religious headwear is legally exempt (MV Act §129), and the review insisted this be
an *ontology* requirement rather than a rule-layer afterthought. The mapping
(`turban → exempt`, `uncertain → abstain`) is deliberately **not** encoded in the
contracts — it is a rule-layer policy decision. This distinction later mattered
enormously (§8.3).

**U4's posture:** the registry records **candidacy, provenance, access, licensing,
task fit, split/leakage metadata, privacy, integrity** — and downloads nothing.
Eight entries exist (`aicity-track5`, `helmet-myanmar`, `idd`, `ccpd`,
`brnocompspeed`, `anpr-synthetic-indian`, `traffic-signal-rois`,
`event-evaluation-footage`). **No dataset has ever been downloaded.** This remains
true today.

### The ADR pack (2026-07-07 → 2026-07-08)

**ADR-001 — Detector / licence posture. Status: Accepted.**

The most carefully argued document in the project. It separates **five licence
concerns that must not be treated as interchangeable**: framework licence, weights
licence, dataset licence, derived-artifact licence, deployment implications.

- **Option A — Ultralytics (AGPL-3.0)** for velocity. **Rejected.** Verification on
  2026-07-08 confirmed from the official licensing page that *trained
  models/weights fall under AGPL-3.0 by default*, and any commercial, closed-source,
  SaaS, embedded, or privately-deployed use — including custom-trained weights —
  requires a paid Enterprise licence. For a project whose stated driver is future
  reuse and distribution, this was disqualifying.
- **Option B — permissive-only stack.** **Chosen.** RT-DETR / D-FINE (Apache-2.0),
  timm (Apache-2.0), ByteTrack / OC-SORT (MIT), PaddleOCR (Apache-2.0).
- **Option C — defer.** **Rejected** — the facts were now verified and the U2
  `Detection` seam made the choice bounded and reversible, so no genuine blocker
  justified continued deferral.

It also recorded that **RT-DETR is redistributed inside the AGPL `ultralytics`
package**, so it must be integrated only from its Apache-2.0 source or the HF
Transformers port — an easy trap to fall into.

A separate, later decision (same day) adopted **Apache-2.0 for TrafficPulse's own
source**, declared via PEP 639 in `pyproject.toml`.

**ADR-002 — Storage. Status: Accepted.** SQLite (SQLAlchemy) for events/cases/review
state; filesystem content-addressed artifacts; Parquet observation logs; YAML
registries; file-based experiment tracking. **Rejected:** PostgreSQL (unnecessary
operational weight), document/NoSQL/object store (no distributed requirement).
Critically: *"SQLAlchemy/PyArrow become Phase 1 runtime dependencies when the
storage runtime is built — none are added now."* **This runtime was never built**
(§9.3).

**ADR-003 — Offline-first + labeled near-real-time demo. Status: Accepted.** On an
RTX 4060 Laptop (8 GB VRAM), the full concurrent stack is not expected to run in
real time. **Rejected:** claiming a real-time integrated pipeline ("unsupported on
the target hardware and dishonest before measurement"); offline-only with no demo
(weakens the demonstration).

**ADR-004 — Reprocessing and event identity. Status: Proposed — and still Proposed
today.** Proposes per-run immutable ids plus lineage linkage; **rejects** mutable
overwritten events (violates append-only auditability); **defers** the cross-run
deduplication rule. Deliberately left Proposed because no event-store runtime
existed yet. It nonetheless produced a live invariant: *manifests are append-only;
no run silently overwrites another* — which is exactly what `EventStore._write_once`
enforces, and what forced H9's design nine phases later (§3.9).

### Phase 1 — First vertical slice: wrong-way driving (2026-07-08 → 2026-07-10)

**Commits:** `0dfc774` (P1-U1) → `89c0d1f` (P1-U12)

A process anomaly worth recording: **P1-U1…P1-U7 were executed and committed
without a plan document.** `docs/phase-1-plan.md` (commit `8bd8e83`, 2026-07-09)
was authored *retroactively* to govern the remaining units and to document what had
already shipped. The document says so explicitly rather than pretending otherwise.

| Unit | Deliverable |
|---|---|
| P1-U1 | Geometry primitives (vectors, segments, polygons) |
| P1-U2 | Synthetic trajectory generator — golden trajectories with known labels, no model, no video |
| P1-U3 | Generic rule-engine core — violation-agnostic hypothesis FSM |
| P1-U4 | Wrong-way reasoning + heading-vs-lane observation |
| P1-U5 | PTS-accurate video ingestion (PyAV) |
| P1-U6 | Detector integration foundation (`Detector` protocol, `DetectionAdapter`, `StubDetector`) |
| P1-U7 | RT-DETR backend |
| P1-U8 | Tracker integration foundation |
| P1-U9 | IoU tracker backend |
| P1-U10 | `WrongWayPipeline` orchestration |
| P1-U11 | Minimal event persistence + evidence stub |
| P1-U12 | End-to-end slice runner + demo hardening |

**P1-U2 is a strategic choice worth naming.** Building a synthetic trajectory
generator *before* touching a detector meant the entire reasoning layer could be
developed and tested with **known ground truth and no ML dependency at all**. This
is why `rules/` has always been testable without a GPU, and it is the practical
expression of principle 1.

**P1-U5's discipline:** media-relative timestamps from **PTS only** — no fabricated
FPS fallback. The architecture review had elevated PTS/VFR discipline from a
speed-local concern to an ingestion-wide requirement.

**P1-U7 — RT-DETR posture** (from engineering notes):
- Integrates the **Apache-2.0 HuggingFace `transformers` port**, not Ultralytics.
- Optional extra `trafficpulse[rtdetr]`; **all imports lazy inside methods** so base
  import, dev, and CI stay ML-free.
- `torchvision` and `pillow` are required at runtime by `AutoImageProcessor` —
  discovered via the real smoke test, not from documentation.
- The detector **clips predicted boxes to the frame rectangle and drops off-frame
  boxes**, because RT-DETR predicts fractionally outside the image and the frozen
  `BoundingBox` contract rejects out-of-range coordinates — one bad box would
  otherwise fail the entire adapter batch.
- **No checkpoint is blessed, defaulted, or committed.** `local_files_only=True` by
  default. There is **no model registry** in the repo — only a dataset registry.
- Unit tests are **fake-engine driven**; the real smoke test is opt-in and
  auto-skips.

**P1-U9 — a documented deviation from plan. Superseded/Rejected.** The plan's
default tracker was **ByteTrack**. At execution time it was **blocked**: no package
installed, no local licence/provenance evidence, web verification unavailable, and
canonical ByteTrack needs `lap`/`scipy` + Kalman — all on the P1-U8 boundary test's
forbidden-import list. Per the card's stop condition, *an unverifiable external
tracker must not be integrated*. A dependency-free **greedy-IoU associator** shipped
instead: pure stdlib, class-constrained IoU matching, deterministic greedy
assignment, age/min-hits lifecycle, **no Kalman**. Documented limitations:
`velocity=None`, `tainted=False`. ByteTrack remains a future separately-audited
enhancement behind the same seam.

**P1-U11 — the persistence decision that shaped everything after it.** ADR-002 names
SQLite, but the card's stop-condition defaulted to deterministic JSON files for this
slice, and **no dependency was added**. `EventStore` writes
`<root>/<run_id>/events/<event_id>.json` and `manifests/<event_id>.json`, with
**write-once semantics per `(run_id, event_id)`**: identical replay is an idempotent
no-op; a differing write raises `EventConflictError`. This honours ADR-004's
append-only direction while freezing no cross-run identity rule.

### Phase 2 — Evidence integrity + illegal stopping (2026-07-10 → 2026-07-11)

**Commits:** `0e17de3` (plan) → `6eb7de6` (docs sync)

This phase introduced a **planning discipline that persisted**: the plan document
was written *first* (`0e17de3`), and it declared separate identifier namespaces
(`P2-U#` / `P2-V#` / `P2-R#` — units, verifications, research) that explicitly do
**not** supersede Phase 0-F's `U#` or Phase 1's `P1-U#`. Each phase plan since then
opens by declaring what it does not rewrite.

| Unit | Deliverable |
|---|---|
| P2-U1 | Model-provenance propagation |
| P2-U2 | In-zone observation derivation |
| P2-U3 | Stationary observation derivation |
| P2-U4 | Illegal-stopping reasoner |
| P2-U5 | `IllegalStoppingPipeline` |
| P2-U6 | Recorded-clip end-to-end verification |

**P2-U1** made confirmed events carry truthful, sorted, de-duplicated detector and
tracker `ModelRef`s. A subtlety recorded at the time: the reasoner stamps `models`
verbatim and **cannot** normalise them (circular import), and `models` is
deliberately **excluded from `event_id`** — so the *decision* is byte-identical with
or without provenance.

**P2-U5's shape was a deliberate architectural choice:** `IllegalStoppingPipeline`
was built as a **thin sibling** of `WrongWayPipeline`, not a generalisation. The
Phase 2 decision (E.8/E.9) was explicit that generalisation would come later, when
there were enough instances to generalise *from*. That restraint paid off in P3-U1/U2.

**Documented deferrals — Deferred:** congestion suppression, and re-association of a
long-stationary vehicle across a tracker ID switch. Both are named in the README as
explicit limitations rather than hidden.

### The design review and roadmap re-sequencing (2026-07-12)

`architecture-review.md` §28 records an **accepted design review** of the remaining
roadmap. It is *additive* — it changes sequencing, not scope, and explicitly leaves
the earlier text intact.

**Accepted principles:** capability-first sequencing; generalized reasoning +
pipeline infrastructure **by composition, not deep inheritance**; a dynamic traffic
context stream; **association before** helmet/triple-riding; **calibration before**
speeding; the evaluation harness and observation-log substrate as first-class
deliverables.

**Explicitly rejected here:** speculative architecture, and a monolithic
"TrafficSemantics" engine.

**Two factual reconciliations recorded as supersessions:**
1. §6 had listed red-light jumping under an earlier phase. Sequencing is now:
   wrong-way = P1, illegal-stopping = P2, **red-light = P3**, triple-riding +
   no-helmet = P4, speeding = P5.
2. Earlier text referenced **ByteTrack** as the planned tracker; Phase 1 shipped the
   greedy-IoU associator instead.

### Phase 3 — Generalized reasoning + dynamic context (2026-07-12 → 2026-07-13)

**Commits:** `bb7f569` (P3-U1) → `5f76461` (P3-U4)

| Unit | Deliverable | Status |
|---|---|---|
| P3-U1 | Generalized temporal-run reasoner base (composition) | **Implemented** |
| P3-U2 | Generalized composition-pipeline base | **Implemented** |
| P3-U3 | Dynamic traffic context stream + generalized observation join | **Implemented** |
| P3-U4 | Stop-line / junction-entry crossing derivation | **Implemented** |
| P3-U5 | Red-light reasoner + pipeline + persistence + e2e | **Planned — never built** |
| P3-U6 | Observation-log substrate (ADR-002 Parquet) | **Planned — never built** |
| P3-U7 | Event-level evaluation harness (§23-E) | **Planned — partially superseded by H5** |

**This is the most significant incomplete phase in the project.** P3-U1…U4 built
*all the infrastructure for red-light jumping* — the shared reasoner, the shared
pipeline base, the signal-state context stream, and the stop-line crossing
derivation — and then **the red-light reasoner itself (P3-U5) was never written**.
The project pivoted to the helmet track instead.

`TemporalRunReasoner` (P3-U1) is the quiet hero of the codebase: extracted from the
byte-for-byte duplication between the wrong-way and illegal-stopping reasoners, it
now drives **every** violation, including the two that came later. It carries no
violation-specific knowledge — violation type, threshold, gap tolerance, and the
measurement payload are all injected.

### The Viewer detour (2026-07-14 → 2026-07-15)

**Commits:** `121f15d`, `6d81857`, `6e40fdf`, `ef21b53`

A standalone `viewer/` application (Flask-ish `app.py` + `index.html`, launched by
`launch.py` in a pywebview window) was built for a demonstration. It is **not** the
current frontend and is now effectively superseded by the React SPA — but it is
still in the tree and still runs.

This detour produced the project's **first real debugging investigation** (§8.1).

### Phase 4 — Executed differently from planned (2026-07-17)

**Commits:** `91ed268` (P4-U1) → `db7db61`

Here the **plan and the execution diverge**, and the divergence was never
reconciled in the plan document. This matters for anyone reading `phase-4-plan.md`.

| Planned (`phase-4-plan.md`) | Executed (git) |
|---|---|
| P4-U1 Association derivation | P4-U1 **Validate RT-DETR motorcycle/person coverage on real footage** |
| P4-U2 Quality-weighted confidence aggregation | P4-U2 **Helmet-classifier seam + `FrameObserver` hook** |
| P4-U3 Rider-count observation + triple-riding reasoner | P4-U3 **Zero-shot helmet classifier backend** |
| P4-U4 Helmet-state classifier seam + no-helmet reasoner | P4-U4 **Helmet observation pipeline** |
| P4-U5 **Mandatory CNN-vs-ViT helmet experiment** | P4-U5 **No-helmet reasoning + violation derivation** |
| — | P4-U6 **Integrate helmet violations end to end** |

**What actually happened:** Phase 4 became a **no-helmet-only vertical slice**.
Triple riding was pushed out (it shipped later as v1.1-U3). The **mandatory
CNN-vs-ViT experiment was not executed** and has not been executed since (§9.2).

**P4-U2 introduced the `FrameObserver` hook** — the first place in the architecture
where a rule needs *pixels*. This is significant: `finalize` never sees pixels, so
the observer is the seam through which a classifier participates. It later became
the mechanism the entire overlay framework depends on.

**P4-U3 shipped a zero-shot CLIP classifier**, not a trained one — a pragmatic
choice given that no helmet dataset had been (or has been) downloaded.

### H1–H5 — The dataset and training track (2026-07-18 → 2026-07-21)

**Commits:** `08ce173` (H1) → `0e0122f` (H5)

A new identifier namespace (`H#`) appears here, without a phase plan document. This
track lives in `experiments/helmet_rtdetr/` and is **infrastructure for training a
helmet detector**, not the classifier comparison the spec mandated.

| Milestone | Deliverable |
|---|---|
| H1 | Dataset registry + ingestion infrastructure |
| H2 | Unified annotation conversion pipeline |
| H3 | Deterministic dataset splitting pipeline |
| H4A | Model-agnostic training infrastructure |
| H4B | RT-DETR training pipeline integration |
| H5 | RT-DETR evaluation framework |

**H5's design principle** (recorded in engineering notes): the evaluator is
**pure-core** — the caller supplies model and data configs. It also established a
`None`-vs-`0.0` metric convention, consistent with the never-fabricate rule.

**No training run has been executed** and no weights are committed. This
infrastructure is ready and unused.

### H6 — Real-time inference engine (2026-07-21)

**Commit:** `a46cfcb`

`src/trafficpulse/engine/` composes the shipped seams into a deterministic engine:
ingestion → detection → tracking → reasoning → confirmation → evidence →
persistence, with scheduling, batching, metrics, and structured logs.

Design notes: the engine composes `CompositionPipeline` via **public strategy
factories**; the checkpoint mechanism defers open after-window events; scheduling is
**PTS-only**. This is the labeled near-real-time demo mode ADR-003 authorised.

### H7A–H7E — The web application (2026-07-21 → 2026-07-22)

**Commits:** `56f004e` (H7A) → `f7400e9` (H7E), including the first three pull
requests in the project's history (`#1`, `#2`, `#3`) — a shift to branch-based
development.

| Milestone | Deliverable |
|---|---|
| H7A | FastAPI application layer over H6 |
| H7B | Frontend foundation (Vite + React + TypeScript SPA) |
| H7C | Video workspace (upload → process → review) |
| H7D | Live processing integration |
| H7E | Analyst review workflow |

**H7A's key seam:** an injectable `EngineProvider`, so the same app runs with the
real RT-DETR backend in production and with stub-injected engines in tests,
unchanged. FastAPI is an **optional extra** (`trafficpulse[api]`) — the core
library's import graph stays free of a web framework.

A recorded trap: **keep RT-DETR out of in-process tests** — there is a `sys.modules`
boundary trap if it leaks in.

**H7B's layering:** `pages → hooks → services → client`, with centralised typed
query keys and endpoint registry. No component performs network calls.

**H7D** added cooperative cancellation, live polling, richer lifecycle phases
(`initializing` / `running` / `finalizing` derived from *truthful* frame counters,
never fabricated), and recovery.

**H7E** added severity, multi-select, an evidence viewer, analyst notes
(localStorage), and JSON/CSV export — **frontend-only, no backend changes**. The
notes store was later deleted in H9 (§3.9).

### H8 — Release hardening (2026-07-22 → 2026-07-23)

**Commits:** `f02ef06`, plus version → 1.0.0

Opt-in CORS, single-process static SPA serving with an SPA fallback, deployment
documentation, a demo intro. A recorded nuance: **real inference is code-configured,
not environment-configured** — `serve.py` is the operator composition root that
constructs a typed `AppConfig` with the real RT-DETR + CLIP backends.

`serve.py` deliberately enables only `triple_riding` and `no_helmet`: those are
motorcycle-perception rules that need **no per-camera geometry**, so they work on
arbitrary uploaded footage. `wrong_way` and `illegal_stopping` are **excluded**
because they require a `SceneConfig` calibrated to the uploaded video's camera, and
running them against the synthetic example scene would produce meaningless geometry.

### v1.1 U1–U3 — Motorcycle perception and triple riding (2026-07-22)

**Commits:** `abe8796` (U2), `fa900c4` (U3)

| Unit | Deliverable | Note |
|---|---|---|
| v1.1-U1 | `perception/` package aggregating detector/tracker/association into Motorcycle/Rider/MotorcycleTrack observations | Association already existed |
| v1.1-U2 | Helmet app integration | The no-helmet pipeline already existed from P4; U2 injected the classifier through the app provider |
| v1.1-U3 | Triple riding | New reasoning slice: `rider_count` observation + `TripleRidingReasoner` on the shared `TemporalRunReasoner`; **pure geometry, no classifier** |

This closes the loop on Phase 4's planned scope, one phase late and under a
different identifier namespace.

### The turban-exemption investigation (2026-07-23)

**Commit:** `329ac61` — see §8.3.

### The overlay framework (2026-07-24)

**Commit:** `7720413`

`src/trafficpulse/overlay/` — a **generic, violation-agnostic visualization layer**:
`metadata` (frozen scene/element models, two semantic axes `OverlayEmphasis` +
`OverlayAlert`, `OverlayLayer` z-order) + `theme` (tokens → RGB) + `layout` (pure
label-collision solver), all Pillow-free; `registry` (`OverlayProvider` protocol,
`OverlayCompositor`) above them; `renderer` (`PillowOverlayRenderer`) as the **only**
pixel code, importing Pillow lazily.

**The "no recompute" rule:** overlays are drawn from metadata the pipeline *already
produced*. `HelmetFrameObserver` gained an opt-in `capture_overlay=True` that records
rider/motorcycle boxes and the **exact** head-crop box the classifier saw, as a
byte-identical side effect of the classification pass it already runs.

### Phase 1 (overlay) — Production integration (2026-07-29)

**Commit:** `e693475` — see §8.4 for the root-cause investigation.

### Overlay polish pass (2026-07-29/30)

Presentation-only. Measured on the reference clip: box shake 3.42 → 1.69 px/frame
(−51%), 27% of frames perfectly still, displayed-score churn 6.07 → 1.54 pts/frame
(−75%), captions 1650 → 771 (−53%). See §8.5.

### Analyst Review Workspace, Phase 2 (2026-07-29)

**Commit:** `49acfc4`

Built on H7C/H7E rather than replacing them. Added: event narrative timeline,
review statistics, processing summary, workflow stepper, client-side thumbnails,
review playback windows, timestamp search, time/confidence-range filters. Fixed the
always-null confidence bug (§8.6). One additive backend change: `EventSummary`
gained `start_at` and `confidence`.

### H9 — Analyst decision and case management (2026-07-29/30)

**Branch:** `feature/h9-case-management` (uncommitted at time of writing)

The current work. Extended `ReviewCase` + `ReviewStatus` — **contracts that had
existed since Phase 0-F U2 and had never been wired to anything**. Added an
append-only review journal (`persistence/review_store.py`), `ReviewEntry`, a
transition table, `ReviewService`, two endpoints, and a `ReviewPanel` in the
existing detail card. Deleted the localStorage notes store as now-duplicated state.

### H10 — Persistent repository recovery (2026-08-01)

**Commit:** `9b979ad`

Closed the gap §9.5 described: events were on disk after a restart but unreachable,
because nothing rehydrated the in-memory `VideoStore`/`JobStore` from `runs/`.

`app/recovery.py` writes two small per-entity sidecars — `videos/<video_id>.json`
(`VideoSnapshot`) and `runs/<job_id>/run.json` (`RunSnapshot`) — and rebuilds both
registries from them at startup. The persistence model was **not** touched:
`EventStore` stays write-once, `ReviewStore` stays an append-only journal.

Three decisions worth keeping:

- **What must be stored is exactly what is unrecoverable.** A `ConfirmedEvent`
  names a camera, not an upload, so the **job→video linkage exists nowhere else**;
  neither does the client-supplied filename. Everything else is derived — an event
  id *is* a filename, so the event index is rebuilt from a directory listing with
  **no `ConfirmedEvent` deserialised**; overlay availability is a file-existence
  check; review state needs no recovery at all.
- **Per-entity sidecars, not one `index.json`.** A single index would be a mutable
  hot file every job rewrites: write amplification, a concurrent-writer hazard, and
  one corruption losing the whole repository.
- **A job recorded as running is settled as failed**, with a message saying a
  restart interrupted it — restoring it as `running` would leave a client polling
  a job nothing can advance. Same reasoning retires a `pending` overlay.

### H11 — Historical video library (2026-08-02)

The frontend completion of H10. Recovery made persisted work *reachable by id*;
nothing made it **discoverable**. Three gaps, all read-side:

1. `VideoStore` was addressable but not enumerable (`get`/`contains`, no listing),
   so no endpoint could have listed videos even if one had existed.
2. The SPA's notion of "the video" was literally the one `VideoUploadResponse` in
   `localStorage`; `VideosPage` branched on it. Historical videos were not hidden,
   they were **unrepresentable**.
3. **The source video was not servable.** Playback used
   `URL.createObjectURL(file)` — a session-only blob of the picked file. Only the
   *overlay* had an endpoint, so a recovered video whose run produced no overlay had
   no playable source at all.

Added: `VideoStore.videos()`, `JobStore.for_video()`,
`ReviewStore.reviewed_event_ids()`, a `VideoLibraryService`, and
`GET /api/videos`, `GET /api/videos/{id}`, `GET /api/videos/{id}/media`. Frontend:
a `VideoLibrary` card under the dropzone, `openVideo` on the processing controller,
and `lib/video-source.ts` resolving overlay → local object URL → stored upload.

Decisions worth keeping:

- **A listing deserialises nothing.** Event counts come from the in-memory index
  (filenames, per H10), overlay availability from a path already on the job record,
  and review progress from **one directory listing** of `runs/reviews/` — journal
  *names* answer "has anybody touched this event" without opening a file. So the
  library reports review progress as events *acted on*, deliberately not *decided*;
  only a per-event fold can tell those apart, and that is what the event list does.
- **Recovered and live videos are the same shape by construction**, because both
  are read through the registries H10 rebuilds. A regression test asserts the two
  responses are byte-equal across a restart, so "recovered" can never become a
  state the UI has to explain.
- **`uploaded_at` is the one genuinely new fact** (nothing recorded an upload
  instant). It is optional, and a pre-H11 snapshot recovers it from the stored
  file's mtime — which *is* when the upload was accepted, since the file is written
  once. A video with neither reports `null` and sorts last in **both** directions
  rather than being given a substitute date.
- **A deleted file does not delete the row.** `video_media_not_found` is distinct
  from `video_not_found`: the events and review history are still valid, so the
  library keeps the video and reports that playback is unavailable.

---

## 4. Architectural decisions not captured in ADRs

These are as load-bearing as the ADRs but were decided in-flight:

1. **The detector/tracker/classifier seams are protocols, and every backend is an
   optional extra.** Base install has no ML dependency. Enforced by boundary tests
   with forbidden-import lists.
2. **`event_id` is a content-derived SHA-256** over identity-bearing fields (scene
   hash, camera, violation, rule, track ids, start/trigger, hypothesis id) —
   deliberately excluding `models`, so provenance never changes the decision.
3. **Media time is anchored at a fixed UTC epoch.** `FrameRecord → Frame` conversion
   anchors PTS at the Unix epoch, so an event's `trigger_at` maps directly onto the
   video's own 0..duration timeline. The frontend relies on this
   (`eventMediaSeconds`).
4. **Status is derived, never stored twice** (H9). The review journal is the record;
   `ReviewCase` is a fold over it.
5. **Wall-clock is forbidden except where it timestamps a human act.** Everything
   else uses media time. `ReviewEntry.at` is the documented exception.
6. **Review state may never live in the write-once event store** (H9) — see §8.7.
7. **Presentation may smooth; data may not.** The overlay's EMA + deadband smoother
   writes nothing back.

---

## 5. Refactors

| Refactor | Commit | Motivation |
|---|---|---|
| Extract `TemporalRunReasoner` | `bb7f569` (P3-U1) | Wrong-way and illegal-stopping reasoners were byte-for-byte duplicating the run machine. Composition, not inheritance. |
| Extract `CompositionPipeline` | `be5e092` (P3-U2) | Same duplication in the two pipelines' front halves. |
| Generalized observation join | `9a38a2f` (P3-U3) | Needed for multi-stream rules (signal + crossing). |
| `perception/` aggregation package | v1.1-U1 | Motorcycle/rider observations were scattered across detector/tracker/association. |
| Banner layout split from painting | Overlay polish | Captions needed banner rects *before* being placed, though banners paint last. |
| Notes moved backend + local store deleted | H9 | Server persistence made the localStorage store duplicated state. |

---

## 6. Testing and verification milestones

| Point | Backend tests | Note |
|---|---|---|
| Phase 2 complete | ~933 | Baseline before P2-U6 |
| P2-U6 | 946 | |
| Viewer detour | 1022 | |
| Turban fix | 1891 | |
| README's stated figure | 1840 | **Now stale** |
| Overlay polish | 1971 | |
| Analyst review Phase 2 | 1971 | +frontend 329 |
| **H9 (current)** | **2016** | +frontend **354** |

**Conventions that emerged:**
- Test module basenames must be **globally unique** across `tests/` subdirectories —
  there are no `__init__.py` files, so pytest's prepend import mode collides
  otherwise.
- Opt-in real-model tests auto-skip unless both the extra is installed *and* an env
  var points at a local checkpoint. Nine such tests exist.
- Boundary tests assert forbidden imports, not just behaviour.
- `tests/docs/test_adr_pack.py` mechanically enforces the sanctioned-package
  allow-list (§10.1).

**CI:** single Linux runner, Python 3.12, `ruff check .` → `mypy src` (strict) →
`pytest -q`. A separate **native-Windows verification checklist**
(`docs/windows-verification.md`) exists because development is Windows-native and CI
is Linux.

---

## 7. Deployment evolution

1. **P1-U12 / P2-U6** — CLI slice runners (`python -m trafficpulse.pipeline`).
2. **Viewer (2026-07-14)** — `launch.py` + pywebview desktop window on port 8000.
   **Superseded.**
3. **H7A** — ASGI app (`trafficpulse.app.asgi:app`), no server declared; host/port
   travel in `AppConfig`.
4. **H8** — `serve.py` operator composition root with real backends; opt-in CORS;
   single-process SPA serving with SPA fallback; version 1.0.0.

Dev runs two processes (API + Vite proxying `/api`); production can serve the built
SPA from the API process.

---

## 8. Debugging investigations

### 8.1 "Uploads always produce 0 violations" (2026-07-14)

**Symptom:** every uploaded clip produced zero wrong-way violations.

**Root cause:** `viewer/app.py::_analyze_upload` reasoned every upload against the
synthetic `example-scene.yaml` (1920×1080, legal direction "up"). Real clips have
their own frame size and road orientation, so the reasoner **correctly** never
confirmed. Notably it was **not** a lane-polygon gate — the wrong-way slice derives
headings from full tracks and ignores zone membership.

**Fix (viewer layer only, backend untouched):** one real RT-DETR pass records
per-frame detections and derives dominant flow; `build_calibrated_scene()` authors a
validated `SceneConfig` in the clip's own pixel space with legal direction = observed
flow; `RTDetrCapturedReplay` replays the recorded detections into the unchanged
slice, so the clip is inferred once, not twice, and `detector_kind` names the replay
truthfully.

**The honest finding:** the supplied Connaught Place clip has dominant flow 166.3°,
and all 23 substantial movers are within 22° of it — it contains **genuinely zero
wrong-way vehicles**. "Toward the camera" *is* the legal direction there. A positive
demo clip was constructed by compositing a real car crop moving against the flow —
a constructed scenario with genuine analysis, documented as such in its docstring.

**A structural insight recorded at the time:** auto-calibration can never flag the
majority direction; only a minority opposing vehicle can confirm. Whole-clip time
reversal therefore also yields zero, because the flow reverses too.

### 8.2 "Prove it" — the evidence standard (2026-07-14)

When the conclusion "this clip contains no wrong-way vehicle" was first presented
from aggregate track statistics, it was **rejected**. The demand was: identify the
specific vehicle in the extracted frames, state its track ID, show its computed
heading vector.

The conclusion was accepted only after annotated frames with boxes, IDs and heading
vectors drawn on the footage, a trajectory-arrow overlay, and a per-vehicle table
tying visible cars to track IDs.

**This became a standing convention** (§10.4) and is arguably the single most
influential process decision in the project — it is why the overlay framework exists
at all.

### 8.3 The turban exemption erasing a real violation (2026-07-23)

**Symptom:** a real New Delhi clip with an obviously bare-headed rider produced **0
NO_HELMET events**.

**Investigation:** full per-stage instrumentation, not guesswork. Perception was
fine — the rider tracked as `iou-1` for the whole 10s clip, associated to his
scooter, and CLIP classified him `no_helmet` in **244 of 264 frames**, including 204
consecutive from t=0.

**Root cause:** `rules/no_helmet.py::exempt_riders` used a **latching single-frame
turban exemption** — *any one* `turban` observation exempted that rider for the
entire clip. The untuned zero-shot CLIP misreads a bare head as `turban` on a
minority of frames (~6% here, a near-tie softmax), and those stray frames
retroactively cancelled a 6.8-second no-helmet episode.

**Fix:** predominance-based exemption — exempt only when `turban` is the rider's
predominant reading (`turban >= no_helmet`; ties exempt). Deliberately minimal: no
threshold lowering, no classifier bypass. The bias still favours the exempt rider.
Result on the same video: **0 → 3 events**.

**The honest framing recorded in the code:** the residual issue is *classifier
quality* (loose full-width head crop + untuned prompts), not rule logic. The rule is
now robust to that noise; better crops and prompts remain future work, gated on
having held-out helmet data.

**A trap recorded during this work:** `serve.py`'s `LABEL_MAP` uses `motorbike`, not
`motorcycle`, because `PekingU/rtdetr_r50vd`'s `id2label` uses the VOC-style
spelling. **Do not "fix" this.**

### 8.4 The overlay that never reached the workspace (2026-07-29)

**Symptom:** the annotated video never appeared in the Video Workspace.

**Investigation:** a full stage-by-stage trace found **every stage implemented and
working** — capture, provider, compositor, renderer, MP4 encoding, persistence, the
endpoint, the frontend fetch, and the workspace wiring.

**Root cause — ordering, not a missing stage.** `ProcessingService._run` called
`mark_succeeded` *before* `_render_overlay_video`. The frontend's `refetchInterval`
returned `false` on any terminal status. So the client's first poll after the run
saw `succeeded` + `overlay_available: false`, **stopped polling permanently**, and
fell back to the raw upload — while the MP4 was rendered correctly moments later and
served correctly, with nothing left to ask for it.

**Why tests never caught it:** every app test used `SynchronousJobExecutor`, where
`_run` completes before the HTTP call returns, making the race window exactly zero.
It only manifested under the production `ThreadJobExecutor`.

**Fix:** an `OverlayStatus` axis (`none`/`pending`/`ready`/`failed`) published on the
job status, with `mark_overlay_pending` called **before** `mark_succeeded`, and a
client that polls until *both* axes settle. A regression test asserts the mark
ordering directly (the window is microseconds and not observable externally), and a
second test drives the real `ThreadJobExecutor` with a held-open render.

### 8.5 Overlay defects found only by looking (2026-07-29/30)

Six defects were found by rendering frames and inspecting them, not by testing:

1. Banner metric rendered red-on-deep-red — the accent colour is tuned to sit
   *against* the banner background, making the headline figure the least legible
   text on the most important line.
2. The chip said 98% while the banner said 97% — live per-frame score vs episode
   mean. Both truthful; together they read as an inconsistency.
3. The evidence meter rendered white, not the specified amber — the dataclass
   default had been changed but not the call site.
4. The banner's "Bike" line churned frame to frame under a fixed event id, because
   it read the live association rather than the event's own `track_ids`.
5. A second confirmed rider was silenced by the small-box caption rule, leaving two
   identical red boxes and no way to map them to their banners.
6. Caption lines stacked by glyph *height* (which excludes the font's top bearing)
   crept upward and collided with the meter.

### 8.6 Confidence was always null (2026-07-29)

`lib/workspace.ts::extractConfidence` searched a `ConfidenceBreakdown` for keys
`overall | combined | score | value`. The contract publishes `classifier`,
`temporal_consistency`, `association`, and `aggregate` — and **`aggregate` is
deliberately `None`**. So confidence was **always null everywhere in the UI**: cards
showed "Conf —", the meter was permanently empty, the confidence sort was inert, and
raising the min-confidence filter emptied the list. Replaced with a priority-ordered
`headlineConfidence()` that also reports *which* component it used.

### 8.7 Where review state could not live (H9, 2026-07-30)

Not a bug — a design collision found by reading before writing. `EventStore` is
write-once; review state is mutable. Storing decisions in `events/` would force
either relaxing write-once (losing the guarantee that a persisted event is exactly
what the reasoner concluded) or refusing every decision after the first. Hence the
sibling journal.

---

## 9. Current state (exact, as of 2026-07-30)

### 9.1 Implemented

**Backend** — 16 sanctioned packages under `src/trafficpulse/`: `contracts`,
`geometry`, `synth`, `rules`, `observations`, `ingestion`, `detector`, `tracking`,
`pipeline`, `persistence`, `classifier`, `association`, `engine`, `perception`,
`app`, `overlay`.

Four of six violations reason end to end: **wrong-way, illegal-stopping, no-helmet,
triple-riding**. RT-DETR detector + IoU tracker + zero-shot CLIP helmet classifier,
all behind protocols, all optional extras. Deterministic JSON event store with
write-once replay. FastAPI app with upload / **video library** / processing /
events / evidence / metrics / overlay / **review** endpoints. Generic overlay
framework producing annotated H.264 MP4s. Append-only analyst review journal with a
validated state machine. Startup recovery rebuilding the runtime indices from disk
(H10), so a restarted process serves — and lists — its whole repository (H11).

**Frontend** — Vite + React + TypeScript SPA. Video workspace (upload → live
processing → frame-accurate review), the historical video library (H11), analyst
review workspace (cards, narrative timeline, statistics, processing summary,
workflow stepper, thumbnails, review playback windows, search, filters, export),
and the H9 decision surface (status badge, decision panel, notes editor, metadata,
audit history).

**Verification** (as of H11, 2026-08-02) — ruff clean, mypy strict clean (131
files), 2061 backend tests passing (9 opt-in skipped), 381 frontend tests passing,
0 lint errors.

### 9.2 The largest gap against the original specification

**The mandatory CNN-vs-ViT experiment has not been executed.**

Master spec §4 requires "genuine, measurable use of Vision Transformer technology"
via a controlled CNN-versus-ViT comparison, with helmet-state recognition as the
leading candidate, reporting accuracy/precision/recall/F1/confusion
matrix/latency/throughput/VRAM/model size, robustness slices, class-imbalance
handling, and honest interpretation of negative results. `architecture-review.md`
§12 reaffirms it. `phase-4-plan.md` P4-U5 schedules it.

**What exists instead:** `experiments/helmet_rtdetr/` — dataset ingestion,
annotation conversion, deterministic splitting, model-agnostic training, RT-DETR
training, and an evaluation framework. All of it *infrastructure for training a
detector*, none of it the classifier comparison.

**The one "ViT" in the runtime is `openai/clip-vit-base-patch32`** — a zero-shot
CLIP classifier whose backbone happens to be a ViT. **This is not the mandated
experiment** and must not be presented as one.

**Blocking dependency:** no helmet dataset has been downloaded, because U4 policy
forbids download before licence review and that review has not concluded.

### 9.3 Deferred, with the reason

| Item | Reason |
|---|---|
| SQLite + Parquet storage runtime (ADR-002) | JSON `EventStore` sufficed; no dependency added |
| ADR-004 cross-run dedup rule | Deliberately Proposed until a reprocessing feature needs it |
| Observation-log substrate (P3-U6) | Never reached |
| Red-light reasoner (P3-U5) | Infrastructure built, reasoner never written |
| Speeding (Phase 5) | Feasibility-gated; needs calibration first |
| ANPR, privacy/redaction, simulated penalties | Contracts exist; no runtime |
| Full evidence-rendering engine | Manifest is a reference stub; no media hashing/OCR |
| Congestion suppression; stationary re-association across ID switch | Explicit Phase 2 deferrals |
| ByteTrack | Provenance-unverifiable offline; greedy-IoU shipped instead |
| Real-footage validation | External gate: permissions/ethics + approved footage + validated `SceneConfig` |

### 9.4 Known documentation drift

`README.md` is **stale**. It reports 1840 backend / 279 frontend tests (actual:
2016 / 354), describes the project as complete only through H8, and lists
"human-review UI" and "simulated-penalty workflow" as unimplemented — the review UI
now exists. `phase-4-plan.md` describes a unit breakdown that does not match what
was executed. Neither has been reconciled.

### 9.5 A structural limitation worth knowing before you touch the API

**Resolved by H10/H11 (2026-08-01/02); kept because the shape still governs.**

`EventService` resolves events through the **in-memory `JobStore`**
(`succeeded_for_video`, `job_for_event`), and `VideoLibraryService` reads the same
registries. Those indices are no longer built only by live traffic — `app/recovery.py`
rebuilds them from per-entity sidecars at startup — but they are still **in-memory
indices over an on-disk truth**, and everything the API can address is exactly what
recovery managed to rebuild.

Two consequences remain live:

- A run persisted **before H10** has events on disk and no `run.json`, so nothing
  records which upload they belong to. Recovery reports it as skipped rather than
  inventing the linkage; those events stay unaddressable, by design.
- Recovery never raises. An unreadable sidecar costs that one video or run, and the
  repository degrades to a smaller one. The library therefore shows what survived,
  not necessarily everything that was ever written.

---

## 10. Unwritten conventions

These are not in any document. Violating them will produce work that gets rejected.

### 10.1 Mechanically enforced

- **New top-level packages under `src/trafficpulse/` must be added to the allow-list
  in `tests/docs/test_adr_pack.py`.** A module inside an already-sanctioned package
  needs no change. This is the mechanical expression of "no speculative structure."
- **Test module basenames must be globally unique across `tests/`.**
- **`ruff`, `mypy`, `pytest` live only in `.venv`** — not on PATH. Invoke as
  `.venv/Scripts/python.exe -m <tool>`.

### 10.2 Documentation style

Docstrings explain **why**, not what. Every non-obvious decision carries its
rationale *and its rejected alternative* inline. Module docstrings routinely run
20–40 lines and describe boundaries, determinism guarantees, and what the module
deliberately does not do. Prose uses `--` rather than em-dashes in Python source.

### 10.3 Historical honesty in documents

Superseded text is **never deleted**. `architecture-review.md` §28 explicitly leaves
the earlier §6 phasing intact and records the reconciliation additively. ADR-001
preserves its own prior "Proposed — documented-open" state as history. Phase plans
open by declaring which earlier plans they do *not* supersede.

### 10.4 Evidence standard

Any claim about what footage contains must be demonstrated **on the frames** —
boxes, track IDs, vectors — naming the specific objects a reader might be thinking
of, and explaining why plausible counter-hypotheses fail. Aggregate statistics alone
are not acceptable evidence.

### 10.5 Epistemic markers

`[V]` verified from an official source, `[K]` known-unknown requiring confirmation,
`[U]` unresolved, `[E]` estimate, `[R]` recommendation. Used throughout the docs and
carried into the registry's status vocabularies.

### 10.6 Performance notes

RT-DETR r50 on CPU is ≈3–4 minutes per 300-frame clip. Run heavy passes in the
background. The reference clip (`vid-a62fcda4015bd549.mp4`, 301 frames, 898×506,
30fps, New Delhi) takes ~350s for full real inference.

---

## 11. Things that exist only in discussion

- **The CNN-vs-ViT experiment design** — specified in three documents, never run.
- **ByteTrack integration** — planned, blocked, replaced, still nominally "a future
  separately-audited enhancement."
- **The review-UI-framework ADR** — `architecture-review.md` §27 says "Phase 2 will
  add a review-UI-framework ADR." It never was. React + Vite was chosen in H7B
  without an ADR, and Streamlit/HTMX (the documented candidates) were silently
  dropped.
- **Hash-chained JSONL audit log** — §21 specifies *hash-chained*. H9 shipped
  append-only JSONL **without** the hash chain.
- **Watermarking "SIMULATION — NOT A LEGAL NOTICE"** — specified; no rendering
  engine exists to watermark.
- **GNSS speed field day** — listed as unresolved [U] in §27; never attempted.
- **AI City Challenge Track 5 access** — registry entry exists, access unconfirmed.
- **Site acquisition for evaluation footage** — the review's #1 risk; no site
  secured, no real evaluation footage acquired. The clips used are stock/CCTV
  footage, not purpose-recorded.
- **Better helmet crops and prompt tuning** — identified as the dominant error
  source in §8.3; gated on dataset access.

---

## 12. Future roadmap

**Immediately actionable (no external dependency):**

1. **Rehydrate the job index from `runs/` on startup** — the single highest-value
   fix; makes the whole API survive a restart (§9.5).
2. **Reconcile `README.md` and `phase-4-plan.md` with reality** (§9.4).
3. **P3-U5 red-light reasoner** — all four infrastructure units are already built.
4. **Hash-chain the review journal** to match §21.
5. **Reviewer identity** — the API records whoever the client claims.
6. **Review-state statistics** — the false-positive count is now a genuine
   detector-quality metric and nothing surfaces it.

**Gated on dataset access:**

7. **The mandatory CNN-vs-ViT experiment** (§9.2) — the largest spec gap.
8. **Trained helmet classifier** to replace zero-shot CLIP.

**Gated on external activity:**

9. **Real-footage validation** — permissions/ethics + approved footage + validated
   `SceneConfig`.
10. **Phase 5 speeding** — requires calibration (P5-U1) first, and is
    feasibility-gated by design.

**Larger deferred items:** observation-log substrate (P3-U6), event-level evaluation
harness (P3-U7), SQLite/Parquet storage runtime, full evidence-rendering engine,
ANPR, privacy/redaction, simulated-penalty lifecycle.

---

## 13. Briefing: what you need to know to be an architectural partner here

*Written for a collaborator with no memory of prior sessions.*

### 13.1 Read these first, in this order

1. `docs/architecture-review.md` — the canonical reference. §13, §14, §15, §19,
   §21, §25, §27, §28 are cited constantly in code comments.
2. `docs/adr/ADR-001.md` — the licence posture that constrains every dependency.
3. `TRAFFICPULSE_MASTER_SPEC.md` §3 (ten principles) and §4 (ViT requirement).
4. The phase plan for whatever you are touching.

### 13.2 The five things that will get your work rejected

1. **Fabricating a value.** If it was not measured, it is `null`. Never `0`, never
   "now", never an interpolation across a gap.
2. **Creating a package speculatively.** A directory appears when a unit implements
   code for it — and must be added to the allow-list.
3. **Adding a base dependency.** ML, web, and drawing dependencies are optional
   extras, lazily imported. The base import graph must stay clean.
4. **Writing mutable state into the event store.** It is write-once by design.
5. **Claiming something about footage without showing it on the frames.**

### 13.3 The invariants you must not break

- Perception never imports reasoning; reasoning never imports perception.
- Rules consume **only** Observations — this is what makes reasoning replayable
  without a GPU (§15, described as "the project's strongest defensibility
  mechanism").
- Confirmed events are immutable; `event_id` is content-derived and excludes
  `models`.
- Tainted tracks may abstain but never confirm.
- No rule emits CONFIRMED from fewer than two observations; every non-confirmation
  produces a logged, countable abstention.
- Confidence is a **component breakdown**, never a single number, and never called a
  probability unless calibration is demonstrated.
- The overlay renderer contains no violation logic. Providers own semantics.

### 13.4 How work is actually sequenced here

Plan document first (declaring what it does not supersede) → units with explicit
stop conditions → implementation → verification gates → an honest report naming what
was *not* done. Stop conditions are real: P1-U9 stopped and shipped a fallback
rather than integrate an unverifiable dependency.

### 13.5 The project's actual character

This is a codebase that **optimises for defensibility over velocity**. It will
happily ship a simpler mechanism (greedy IoU, zero-shot CLIP, JSON storage) rather
than an unverifiable better one, and then document exactly what was traded away. The
documentation is unusually honest about its own gaps — the architecture review
criticises the spec, ADR-001 records what it cannot claim, and the README lists
deferrals rather than hiding them.

The most useful thing you can do is **hold that standard**: when you find something
broken, trace it to root cause and prove the diagnosis before fixing; when you
cannot do something, say so plainly and say why; and when you deviate from a plan,
record the deviation rather than quietly reconciling it.

---

## Appendix A — Commit index

`ae72b00` spec + phase-0 plan · `4b1b847` U1 baseline · `f8c70a0` U2 contracts ·
`e5a47a3` U3 ontology · `777c594` U4 registry · `dd43681` U5 scene config ·
`c48d214` U6 ADRs · `0dfc774` P1-U1 · `0ff1bc0` P1-U2 · `7ad37c6` P1-U3 ·
`4651ffb` P1-U4 · `dd70edd` P1-U5 · `8d32027` ADR-001 resolution ·
`30e9cae` public-release prep · `07f8baa` P1-U6 · `8b6d51f` P1-U7 ·
`8bd8e83` phase-1 plan (retroactive) · `8fe84f6` P1-U8 · `261889d` P1-U9 ·
`efe5120` P1-U10 · `50edf2c` P1-U11 · `89c0d1f` P1-U12 · `0e17de3` phase-2 plan ·
`bb990e9` P2-U1 · `ed7653c` P2-U2 · `f6b2907` P2-U3 · `3c771d9` P2-U4 ·
`c147bce` P2-U5 · `d6b406d` P2-U6 · `6eb7de6` docs sync · `5af9ce0` demo runner ·
`bb7f569` P3-U1 · `be5e092` P3-U2 · `9a38a2f` P3-U3 · `5f76461` P3-U4 ·
`121f15d`/`6d81857` Viewer v0.1 · `6e40fdf` upload auto-calibration ·
`91ed268` P4-U1 · `df9aadd` P4-U2 · `b2e8df2` P4-U3 · `db15090` P4-U4 ·
`0c9e191` P4-U5 · `4b8ff7e` P4-U6 · `db7db61` real helmet inference ·
`08ce173` H1 · `33d42de` H2 · `8f1b297` H3 · `2627b00` H4A · `b989df9` H4B ·
`0e0122f` H5 · `a46cfcb` H6 · `56f004e` H7A · `b8f2674` H7C · `53c294a` H7D ·
`f7400e9` H7E · `f02ef06` real-inference composition root · `abe8796` v1.1-U2 ·
`fa900c4` v1.1-U3 · `329ac61` turban fix · `7720413` overlay framework ·
`e693475` overlay production integration · `49acfc4` analyst review workspace

## Appendix B — Identifier namespaces

`U#` Phase 0-F · `P1-U#` Phase 1 · `P2-U#`/`P2-V#`/`P2-R#` Phase 2 ·
`P3-U#` Phase 3 · `P4-U#` Phase 4 (planned ≠ executed) · `P5-U#` Phase 5 (unbuilt) ·
`H#`/`H7x` post-phase milestones (no plan documents) · `v1.1-U#` motorcycle
perception series

Namespaces are independent and never renumbered.
