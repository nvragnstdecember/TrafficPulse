# TrafficPulse Architecture (entry point)

- **Status:** current (Phase 0-F complete)
- **Date:** 2026-07-07

## Canonical architecture reference

**[`docs/architecture-review.md`](architecture-review.md) is THE canonical
architecture reference for TrafficPulse.** It is accepted, current, and not
superseded. This file is a thin entry point: it confirms that pointer, indexes
the Architecture Decision Records, and records the Phase 0-F foundation status so
the Phase 1 agent has one place to start.

`TRAFFICPULSE_MASTER_SPEC.md` remains the product/research specification;
`docs/architecture-review.md` interprets and constrains it without modifying it.

## ADR index

| ADR | Topic | Status |
|---|---|---|
| [ADR-001](adr/ADR-001.md) | Detector / licence posture (AGPL Ultralytics vs permissive-only) | **Accepted** (2026-07-08) — permissive-only posture; RT-DETR primary direction (D-FINE alternative); detector behind the U2 `Detection` contract |
| [ADR-002](adr/ADR-002.md) | Storage (SQLite + filesystem artifacts + Parquet logs) | **Accepted** |
| [ADR-003](adr/ADR-003.md) | Offline-first + labeled near-real-time demo mode | **Accepted** |
| [ADR-004](adr/ADR-004.md) | Reprocessing / event-identity semantics | **Proposed** |

ADR-001 is now **Accepted** (2026-07-08): TrafficPulse adopts a permissive-only
detector posture — RT-DETR as the primary integration direction (D-FINE an
alternative), with the detector kept behind the U2 `Detection` contract. This
**lifts the Phase 1 detector-integration gate**; the first detector-integration
unit may now proceed. (It was documented-open through Phase 0-F and resolved at
its deadline, before that unit.) Any deviation from an accepted ADR is recorded
as a new ADR, not an edit that erases history.

## Phase 0-F unit completion matrix

| Unit | Deliverable | Location |
|---|---|---|
| U1 | Quality-gated repo baseline + Linux CI + Windows checklist | `pyproject.toml`, `.github/workflows/ci.yml`, `docs/windows-verification.md` |
| U2 | Typed domain contracts + deterministic JSON-schema export | `src/trafficpulse/contracts/`, `schemas/` |
| U3 | Label ontology (machine-readable + guide) | `configs/ontology.yaml`, `docs/ontology.md` |
| U4 | Dataset registry + split/leakage policy + evaluation protocol | `registry/`, `docs/dataset-policy.md`, `docs/evaluation-protocol.md` |
| U5 | Scene-config YAML schema + typed contract + stable hashing | `configs/scenes/`, `src/trafficpulse/contracts/scene.py` |
| U6 | Architecture entry point + ADR pack | this file, `docs/adr/` |

## What is frozen vs provisional (carry-forward)

**Frozen as interface/schema:** U2 contracts and exported schemas; U3 ontology
closed sets (aligned to U2 enums); U4 registry schema; U5 scene YAML schema, typed
`SceneConfig`, and the deterministic `scene_config_hash` (compatible with the U2
`ConfirmedEvent`/`EvidenceManifest` `scene_config_hash` fields).

**Explicitly provisional (must not be promoted without evidence):** speed
feasibility targets (MAE ≈ ≤ 3 km/h, P95 ≈ ≤ 6 km/h — candidate targets pending
U4 justification); event-matching temporal-IoU ≥ 0.3 (candidate default);
scene rule parameters (wrong-way ~120°/~1.0 s/~1.5 m/s, illegal-stop ~10 s/~0.5
m/s, red-light grace ~0.3 s) — all marked `provisional`/`unset` in
`configs/scenes/example-scene.yaml`, requiring tuning on held-out data.

**Unresolved external dependencies (carry-forward):** own event-evaluation footage
acquisition (permissions/ethics — the schedule long pole); AI City T5 access
(unconfirmed); several dataset licences (IDD, HELMET, BrnoCompSpeed, AI City)
recorded `unknown`/`unclear`, not promoted. (ADR-001, the detector/licence
decision, is now **resolved** — permissive-only, RT-DETR primary — and no longer a
carry-forward item.)

## What Phase 0-F does NOT prove

Phase 0-F delivers coherent contracts, governance, evaluation *definitions*, and
configuration boundaries. It does **not** prove — and this repository makes no
claim about — model accuracy, dataset availability, real-time performance, speed
accuracy, event-level precision/recall, legal compliance, production readiness, or
deployment readiness. No behavioral TrafficPulse system exists yet.

## Phase 1 entry conditions

- The Definition of Done in `docs/phase-0-plan.md` holds (contracts, ontology,
  split/leakage policy, evaluation protocol, and scene schema frozen).
- ADR-001, ADR-002, and ADR-003 are accepted; ADR-004 is proposed.
- **ADR-001 is resolved (Accepted, permissive-only), so the detector-integration
  unit is unblocked**; the detector-independent Phase 1 work (geometry, synthetic
  tracks, rule-engine foundations, ingestion) that began under the plan's
  recommended first slice (wrong-way) has already landed (P1-U1…P1-U5).
