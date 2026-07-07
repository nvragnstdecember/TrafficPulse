# TrafficPulse — Phase 0-F Foundation Plan

**Status:** Approved plan, pending explicit go-ahead to begin U1
**Companion:** `docs/architecture-review.md` (canonical architecture reference)
**Scope of this document:** Phase 0-F only (repository foundations + interface-freezing). It does not plan Phase 1 in detail.

## Naming

- **Phase 0-R** — research, feasibility, architecture review. **Substantially complete** (captured in `docs/architecture-review.md`).
- **Phase 0-F** — repository foundations and interface-freezing work that must occur **before** any Phase 1 behavioral implementation. **This document.**

## Phase 0-F principles

1. Interface-freezing only. Phase 0-F contains **types, schemas, policies, and decisions** — nothing with runtime behavior and no ML dependencies.
2. Modules appear only when a unit needs them. **No speculative empty packages.** The Python package root is created minimally in U1 for import + version smoke testing only; the `contracts` package first appears in U2 when domain contracts are actually implemented; a `common` package appears only when genuinely shared code exists.
3. Every deliverable is independently verifiable with a stated command or inspection.
4. Uncertainty is preserved: unconfirmed dataset access, provisional thresholds, and unresolved ADRs are recorded as such, never silently promoted.
5. The phase is bounded by its own Definition of Done, which contains nothing that can expand — this is what structurally prevents an endless planning phase.
6. Nothing is committed unless explicitly asked; no dataset is downloaded; no model is trained.

## Dependency graph & allowed parallelism

```
U1
├─▶ U2  ┐
├─▶ U3  ┘  (U2 ∥ U3)
       ├─▶ U4   (needs U2 + U3)
       └─▶ U5   (needs U2)        (U4 ∥ U5)
                 └─▶ U6           (needs U1–U5 informative)
```

Canonical sequence: **U1 → {U2 ∥ U3} → {U4 ∥ U5} → U6.** Target: 2–3 focused days. Allowed parallelism: U2 with U3; U4 with U5. U4 must not start before both U2 and U3 are complete; U5 must not start before U2.

## Exact Phase 0-F boundary

**In Phase 0-F:** repository quality baseline; domain contracts; label ontology; dataset registry + split/leakage policy + evaluation-protocol document; scene-configuration schema; architecture document + ADR pack.

**Deferred to Phase 1 (behavioral — must NOT appear in Phase 0-F):** geometry algorithms; synthetic track generator; rule-engine implementation; wrong-way rule implementation; evaluation implementation code; ingestion implementation; detector integration; tracker integration; observation-writer runtime; event-store runtime; ML training; dataset downloads.

The boundary test: if a deliverable has runtime behavior or an ML dependency, it belongs in Phase 1.

---

## U1 — Repository baseline

- **Objective:** a minimal, quality-gated repository skeleton with an importable package root.
- **Why it exists:** every later unit needs a place to live with enforced quality checks; establishing this once prevents ad-hoc structure later.
- **Prerequisites:** explicit go-ahead (currently pending).
- **Exact scope:** `pyproject.toml`; the **minimal Python package root only** — `src/trafficpulse/__init__.py` exposing a version, sufficient for package import and version smoke testing; `tests/`; ruff + mypy + pytest configuration; single-environment **Linux** CI; `docs/windows-verification.md` checklist.
- **Non-goals:** **do not create a `common/` package**; **do not create a `contracts/` package** (it first appears in U2); no module beyond the package root; no ML dependency; multi-OS CI matrix; any behavioral code. No empty speculative package directories of any kind.
- **Deliverables:** the files above; green CI; a Windows verification checklist document.
- **Files/dirs affected:** repo root, `.github/workflows/`, `src/trafficpulse/__init__.py`, `tests/`, `docs/`.
- **Acceptance criteria:** CI green on Linux; ruff, mypy, pytest all pass locally on Windows per the checklist; `import trafficpulse` succeeds and exposes a version; the repository contains no empty `common/` or `contracts/` package directories.
- **Required tests:** import + version smoke test (`tests/test_import.py`).
- **Verification:** `ruff check .` · `mypy src` · `pytest -q` · inspection that `src/trafficpulse/` contains only `__init__.py` (no `common/`, no `contracts/`).
- **Required datasets:** none.
- **Expected compute:** negligible (CPU).
- **Stop conditions:** CI infrastructure fighting > 0.5 day → collapse to a single job.
- **Fallback:** documented local-only checks if CI setup is blocked.
- **Dependencies:** none.
- **Completion evidence before dependents begin:** CI run link/log showing green; local command output for all three checks; the Windows checklist file present; directory listing confirming the minimal package root.

## U2 — Domain contracts

- **Objective:** the typed layer boundary of the data-flow (`docs/architecture-review.md` §14) as pydantic models. **This unit introduces the `contracts` package.**
- **Why it exists:** freezing these types lets U4/U5 and all of Phase 1 build against stable interfaces in parallel; the Observation type is the load-bearing perception↔reasoning contract.
- **Prerequisites:** U1.
- **Exact scope:** create `src/trafficpulse/contracts/` and implement models for Detection, TrackState, Association, Observation variants (`in_zone`, `signal_state`, `heading_vs_lane`, `stationary`, `rider_count`, `helmet_state`, `speed±σ`), ViolationHypothesis, ConfirmedEvent, EvidenceManifest, ReviewCase, SimulatedPenalty; shared enums; a JSON-schema export script.
- **Non-goals:** any logic that produces these objects (no ingestion, no rules, no writers); no persistence runtime. Do not create a `common/` package unless a second unit demonstrably shares code with U2 (it does not yet).
- **Deliverables:** contract modules under `src/trafficpulse/contracts/`; deterministic schema-export script.
- **Files/dirs affected:** `src/trafficpulse/contracts/` (new), `tests/contracts/`, a `schemas/` output location.
- **Acceptance criteria:** serialization round-trip property tests pass; mypy clean; schema export is byte-stable across runs; the `contracts` package now exists and contains real implemented models (not a placeholder).
- **Required tests:** round-trip (model → JSON → model) property tests; schema-export determinism test.
- **Verification:** `pytest tests/contracts -q`; run the export twice and diff — no differences.
- **Required datasets:** none.
- **Expected compute:** negligible.
- **Stop conditions:** contract churn > 1 day → freeze a v0 with open fields explicitly marked.
- **Fallback:** mark unstable fields optional in v0 and record them as open in U6.
- **Dependencies:** U1. **Parallel with U3.**
- **Completion evidence:** passing test output; two identical schema-export runs; directory listing showing `contracts/` now populated.

## U3 — Label ontology

- **Objective:** freeze annotation semantics **before** any labeling occurs.
- **Why it exists:** annotation is costly and irreversible-in-practice; the helmet 4-label scheme and its rule-layer mapping must be fixed before crops are labeled.
- **Prerequisites:** U1.
- **Exact scope:** `docs/ontology.md` + machine-readable `configs/ontology.yaml` — detection classes; helmet 4-label scheme {helmet, no_helmet, turban, uncertain} with rule-layer mapping (`turban → exempt`, `uncertain → abstain`); rider-count labels; plate-transcription rules; event-annotation-guide skeleton.
- **Non-goals:** any annotation work; any classifier; any dataset download; any new package.
- **Deliverables:** the two files above.
- **Files/dirs affected:** `docs/`, `configs/`, `tests/ontology/`.
- **Acceptance criteria:** `configs/ontology.yaml` schema-validates; the document is explicitly reviewed and approved.
- **Required tests:** ontology YAML schema-validation test.
- **Verification:** `pytest tests/ontology -q`; human review sign-off recorded.
- **Required datasets:** none.
- **Expected compute:** negligible.
- **Stop conditions:** unresolved label debate > 0.5 day → record the open question and freeze a v0 that abstains on the ambiguous class.
- **Fallback:** fold contested classes into `uncertain` for v0.
- **Dependencies:** U1. **Parallel with U2.**
- **Completion evidence:** validation test output; recorded approval.

## U4 — Dataset registry, split/leakage policy, evaluation protocol

- **Objective:** freeze data governance **before** any training.
- **Why it exists:** leakage prevention and licence discipline are only meaningful if fixed before data is used; the evaluation protocol must exist before results are produced.
- **Prerequisites:** U2 and U3.
- **Exact scope:** `registry/datasets.yaml` schema + entries for every candidate in `docs/architecture-review.md` §7, each with licence + access status + an owner; split-manifest **format**; `docs/leakage-policy.md`; `docs/evaluation-protocol.md` (the seven protocols of §23, and the **candidate provisional** speed-gate targets from §17 recorded and justified — or revised — here, explicitly as provisional).
- **Non-goals:** any dataset download; any split *generation runtime* (format only); any evaluation *implementation code*.
- **Deliverables:** the registry schema + populated entries; both policy documents; the evaluation-protocol document.
- **Files/dirs affected:** `registry/`, `docs/`, `tests/registry/`.
- **Acceptance criteria:** registry loads and schema-validates; every entry carries an explicit access/licence status (including "pending — owner, date" where unverified) — **no entry is silently marked usable**; both policy docs reviewed; the evaluation protocol records the speed-gate numbers as **candidate provisional targets pending justification**, and keeps AI City access **unconfirmed**.
- **Required tests:** registry schema-validation test; a test asserting every entry has a non-empty access-status field.
- **Verification:** `pytest tests/registry -q`; inspection that unconfirmed items read "pending", not "ok", and that speed targets read "provisional".
- **Required datasets:** none downloaded; registry references only.
- **Expected compute:** negligible.
- **Stop conditions:** licence verification blocked → entry marked pending with owner + date, never omitted or guessed.
- **Fallback:** proceed with pending entries; block their *use* (not their registration) until resolved.
- **Dependencies:** U2, U3. **Parallel with U5.**
- **Completion evidence:** validation test output; a rendered registry showing explicit statuses; both docs present and reviewed; the evaluation protocol showing speed targets marked provisional.

## U5 — Scene configuration schema

- **Objective:** versioned scene configuration as validated data, with stable hashing.
- **Why it exists:** every event embeds a scene-config hash; the schema and hash must be fixed before rules or evidence consume them in Phase 1.
- **Prerequisites:** U2.
- **Exact scope:** a pydantic scene schema covering `docs/architecture-review.md` §16 fields; one example config; a config-hash function; validation tests. The schema lives in the existing `contracts` package (or an adjacent schema module) — it does **not** justify a new speculative package.
- **Non-goals:** the annotation tool (Phase 2); any homography computation; any calibration runtime.
- **Deliverables:** scene schema; example config; hash utility.
- **Files/dirs affected:** `src/trafficpulse/contracts/` (scene schema), `scenes/` (example only), `tests/scene/`.
- **Acceptance criteria:** the example config validates; the hash is stable across a serialization round-trip.
- **Required tests:** schema-validation test; hash-stability test.
- **Verification:** `pytest tests/scene -q`.
- **Required datasets:** none.
- **Expected compute:** negligible.
- **Stop conditions:** schema churn > 0.5 day → freeze v0 with optional fields for uncertain elements.
- **Fallback:** mark speculative geometry fields optional in v0.
- **Dependencies:** U2. **Parallel with U4.**
- **Completion evidence:** passing validation + hash-stability output.

## U6 — Architecture document & ADR pack finalization

- **Objective:** lock architecture decisions with recorded rationale.
- **Why it exists:** the implementation agent must read a single settled architecture reference and a clear ADR status list before Phase 1.
- **Prerequisites:** U1–U5 (informative).
- **Exact scope:** ensure `docs/architecture-review.md` is present and current; author the ADR pack — ADR-001 (detector/licence posture), ADR-002 (storage), ADR-003 (offline-first + demo mode), ADR-004 (reprocessing/event identity).
- **Non-goals:** any behavioral code; any implementation of the decisions; any new package.
- **Deliverables:** `docs/architecture.md` (or confirmation that `docs/architecture-review.md` is the canonical reference) + `docs/adr/ADR-001..004.md`.
- **Files/dirs affected:** `docs/`, `docs/adr/`.
- **Acceptance criteria:**
  - ADR-002 and ADR-003 are **accepted**.
  - **ADR-004 may remain proposed** if event-identity semantics are not yet required.
  - **ADR-001 may remain unresolved** at the end of Phase 0-F **only if** the ADR file explicitly documents: (a) its unresolved status, (b) a named **decision owner**, (c) a **decision deadline** set at "before the first detector-integration unit of Phase 1," and (d) the **consequences** (detector integration is blocked until resolved; detector-independent Phase 1 work is not blocked). If any of (a)–(d) is missing, ADR-001 does not satisfy this unit and either the documentation is completed or ADR-001 is resolved.
  - Each ADR states context, decision (or documented-open status), and consequences.
- **Required tests:** none (documentation unit); inspection only.
- **Verification:** inspection that ADR-002/003 carry accepted status, ADR-004 carries at least proposed, and ADR-001 carries either a resolved decision or a complete owner+deadline+consequences record.
- **Required datasets:** none.
- **Expected compute:** negligible.
- **Stop conditions:** none specific; if ADR-001 cannot be resolved, it is documented-open per the acceptance criteria rather than forced.
- **Fallback:** if ADR-001 is genuinely undecided, record it as documented-open (owner, deadline, consequences) — this is an accepted end state for Phase 0-F and explicitly does not block detector-independent Phase 1 work.
- **Dependencies:** U1–U5.
- **Completion evidence:** ADR files present with the required statuses; ADR-001 either resolved or documented-open with all four elements.

---

## Minimal initial repository tree (state immediately after U1)

```
trafficpulse/
├── pyproject.toml
├── .github/workflows/ci.yml
├── docs/windows-verification.md
├── src/trafficpulse/
│   └── __init__.py          # version + package root ONLY
└── tests/test_import.py
```

No `common/` package. No `contracts/` package. Those appear later — `contracts/` in U2, `common/` only if and when genuinely shared code exists.

## Expected repository state after Phase 0-F (all units complete)

```
trafficpulse/
├── pyproject.toml
├── .github/workflows/ci.yml
├── configs/ontology.yaml
├── registry/datasets.yaml
├── scenes/<example>.yaml
├── schemas/                     # exported JSON schemas
├── docs/
│   ├── architecture-review.md · architecture.md (or pointer)
│   ├── ontology.md · leakage-policy.md · evaluation-protocol.md
│   ├── windows-verification.md
│   └── adr/ADR-001..004.md
├── src/trafficpulse/
│   ├── __init__.py
│   └── contracts/               # created in U2; scene schema added in U5
└── tests/  (test_import, contracts/, ontology/, registry/, scene/)
```

No `ingestion/`, `perception/`, `tracking/`, `rules/`, `events/`, `evidence/`, etc. — those appear in Phase 1 when a unit needs them. No `common/` package unless genuinely shared code has appeared.

## Definition of Done (Phase 0-F)

All must hold:
- repository quality baseline operational, with **only the minimal package root created in U1** (no speculative `common/` or `contracts/` packages at U1; `contracts/` created in U2);
- required checks (ruff, mypy, pytest) passing;
- Windows verification documented **and actually executed** where applicable, dated;
- domain contracts implemented and round-trip tested (in the `contracts` package introduced by U2);
- deterministic schema export (two identical runs);
- ontology documented and machine-readable, reviewed;
- dataset registry schema + initial candidate entries with **explicit access/licence status** (pending entries labeled, not omitted);
- leakage and split policy **frozen before training**;
- evaluation protocol documented, with speed-gate numbers recorded as **candidate provisional targets pending justification** and AI City access **unconfirmed**;
- scene schema validated with **stable hashing**;
- architecture document present;
- **ADR-002 and ADR-003 accepted**;
- **ADR-001 resolved, OR documented-open with a named owner, a deadline set before the first detector-integration unit of Phase 1, and stated consequences**;
- **ADR-004 allowed to remain proposed** if event-identity semantics are not yet required;
- **no datasets downloaded**;
- **no models trained**;
- **no speculative empty module or package tree.**

## Verification philosophy

Every unit is closed only by observed evidence: a command's output, a passing test, or a recorded inspection/approval. No unit is marked done on assertion alone. Nothing is claimed verified unless it was actually run or inspected. Unconfirmed external facts (dataset access, licence text) are recorded as pending with an owner and date — never asserted.

## Windows-specific verification expectations

CI runs single-environment Linux; Windows is verified via `docs/windows-verification.md`, a checklist the team **executes and dates** at least once during Phase 0-F: run `ruff check .`, `mypy src`, `pytest -q` on Windows and record results. Known Windows guidance carried from the architecture review: avoid mmcv; plan to run PP-OCR via ONNX Runtime in Phase 1; use PyAV for PTS-accurate decode; WSL2 is acceptable for training only, with the demo native on Windows.

## Documentation expectations

Documentation stays synchronized with the repository: `docs/architecture-review.md` is the canonical reference; U3–U4 produce the ontology, leakage, split, and evaluation documents; U6 produces the ADR pack. Any deviation from an accepted ADR is recorded as a new ADR, not an edit that erases history.

## ADR list & status expectations

| ADR | Topic | Expected status at end of Phase 0-F |
|---|---|---|
| ADR-001 | Detector / licence posture (AGPL Ultralytics vs permissive-only) | **Resolved, OR documented-open** with named owner + deadline (before the first detector-integration unit of Phase 1) + consequences. **Currently UNRESOLVED.** Does **not** block detector-independent Phase 1 work; **does** block detector integration until resolved. |
| ADR-002 | Storage (SQLite + filesystem artifacts + Parquet logs) | Accepted |
| ADR-003 | Offline-first + labeled near-real-time demo mode | Accepted |
| ADR-004 | Reprocessing / event-identity semantics | **Proposed** allowed |

## Transition criteria: Phase 0-F → Phase 1

- Phase 1 may begin when the Definition of Done holds. Contracts, ontology, split/leakage policy, evaluation protocol, and scene schema must be frozen so Phase 1 builds against stable interfaces.
- **ADR-001 gating is scoped, not global:** an unresolved-but-documented-open ADR-001 **does not block** detector-independent Phase 1 work (geometry utilities, synthetic track generation, rule-engine foundations, and other work that does not select or integrate a detector ecosystem). ADR-001 **must be resolved before the first detector-integration unit** begins.
- No behavioral Phase 1 work may begin before the go-ahead for Phase 1, independent of ADR-001 status.

## Recommended first Phase 1 vertical slice

**Wrong-way driving, end to end**, is the recommended first Phase 1 violation (`docs/architecture-review.md` §5a, §6): it needs only the detector plus geometry and hysteresis, no additional learned component, and it exercises the full ingestion → detection → tracking → observation → rule → event → evidence-stub path. The slice is validated first against the synthetic track generator (golden trajectories, known event labels — no model, no video), then against one real clip producing one event with an evidence stub.

Note on sequencing relative to ADR-001: the detector-independent portions of this slice (geometry utilities, synthetic track generator, rule-engine foundations, wrong-way rule logic against synthetic tracks) may proceed while ADR-001 is documented-open; the detector-integration portion (and therefore the real-clip validation) waits until ADR-001 is resolved.

## Higher-level Phase 1 direction (not a detailed plan)

Phase 1 builds the first behavioral vertical slice and the first violation: PTS-accurate ingestion; detector adapter behind the perception contract (gated on ADR-001); ByteTrack integration; observation writer (Parquet); geometry utilities; synthetic track generator; rule-engine core (FSM base, accumulator, abstention — tested on synthetic tracks); the wrong-way rule end to end; a minimal event store. Milestones 2–3 of the progressive demonstration strategy land here. Detailed Phase 1 unit cards are authored when Phase 1 begins, not in this document.
