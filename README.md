# TrafficPulse

**Evidence-first, offline-first traffic-violation reasoning for fixed-camera roadside video.**

TrafficPulse is a research-oriented engineering project that treats a traffic
violation as something to be *reasoned about over time and backed by evidence* —
never as a single model output or a single-frame detection. Its central design
commitment is a hard separation between **perception** (what a model sees in a
frame) and **violation reasoning** (typed observations accumulated over a track,
combined with scene geometry and explicit rules, confirmed only with temporal
evidence and reviewable by a human). This repository is the **foundation layer**
of that system: frozen data contracts, governance, and the first
detector-independent reasoning components — not a finished detector pipeline.

---

## Project status

**Early-stage. Foundations + first detector-independent reasoning slices.** No
behavioral end-to-end violation system exists yet. Concretely:

- **Phase 0-F (foundations): complete** — typed domain contracts, exported JSON
  schemas, label ontology, dataset registry + policies, scene-configuration
  schema, and the architecture/ADR pack are all in place and tested.
- **Phase 1 (behavioral, detector-independent slices): in progress** — geometry
  primitives, a synthetic trajectory generator, a generic rule-engine core,
  wrong-way temporal reasoning, and PTS-accurate video ingestion have landed and
  are unit-tested.
- **Not started:** detector/tracker integration and any real-video end-to-end
  path (see [Roadmap](#planned-capabilities--roadmap)).

Quality gates are green: `ruff`, `mypy --strict`, and **485 passing tests** on
the current tree, with single-environment Linux CI and a native-Windows
verification checklist.

> This is **not** a production enforcement system and makes **no** validated
> real-world accuracy claim. See [Limitations](#research--deployment-limitations).

---

## What TrafficPulse is trying to solve

Naïve "AI traffic enforcement" tends to equate one detector firing with a
punishable offence. That is neither technically defensible nor fair. TrafficPulse
is an academic capstone that instead optimises for **technical defensibility,
reproducibility, honest evaluation, and clear separation of implemented vs
planned capability** (see [`TRAFFICPULSE_MASTER_SPEC.md`](TRAFFICPULSE_MASTER_SPEC.md)).

The intended long-term scope is six locked violation types — no-helmet riding,
triple riding, red-light jumping, wrong-way driving, illegal stopping/parking,
and (feasibility-gated) speeding. Of these, only the **wrong-way** reasoning path
exists today, and only against *synthetic* trajectories.

## Architecture overview

The durable idea is a typed, one-directional data flow where **rules consume only
typed observations**, which makes the reasoning layer deterministically
replayable from a log without a GPU or a model
(`docs/architecture-review.md` §14–§15).

```mermaid
flowchart LR
    V[Video ingestion<br/>PTS-accurate] --> D[Detection]
    D --> T[Tracking]
    T --> A[Association]
    A --> O[Observations<br/>typed per-frame facts]
    O --> R[Rule engine<br/>FSM + accumulation]
    R --> E[Confirmed event]
    E --> EV[Evidence package]
    EV --> RC[Human review case]
    RC --> P[Simulated penalty]

    classDef done fill:#1f7a4d,stroke:#0d3,color:#fff;
    classDef todo fill:#555,stroke:#999,color:#fff,stroke-dasharray:4 3;
    class V,O,R,E done;
    class D,T,A,EV,RC,P todo;
```

*Green = implemented (contract + detector-independent logic). Grey/dashed =
contract defined, behaviour planned.* Ingestion, the observation contract, the
rule engine, and confirmed-event minting exist today; detection, tracking,
association, evidence rendering, review, and penalty simulation are contracts
awaiting implementation.

Design posture, supported by the project docs:

- **Offline-first** — recorded video in, events out; "real-time" language is
  confined to a labelled near-real-time demo mode (ADR-003,
  `docs/architecture-review.md` §22).
- **Evidence-oriented** — every confirmed event is meant to be explainable from
  stored evidence and rule context (`docs/architecture-review.md` §19).
- **Deterministic contracts + explicit uncertainty/abstention** — every
  non-confirmation is a logged, countable abstention; confidence is a component
  breakdown, never presented as a calibrated probability without demonstrated
  calibration (`docs/architecture-review.md` §13).

## Current implemented capabilities

All of the following are implemented **and** covered by tests in this repository:

| Capability | Where | Notes |
|---|---|---|
| Frozen typed domain contracts | `src/trafficpulse/contracts/` | pydantic models for Detection, TrackState, Association, all Observation variants, ViolationHypothesis, ConfirmedEvent, EvidenceManifest, ReviewCase, SimulatedPenalty, shared enums/primitives; round-trip + validation tested |
| Deterministic JSON-schema export | `schemas/*.schema.json` | byte-stable export of the contracts |
| Label ontology | `configs/ontology.yaml`, `docs/ontology.md` | detection classes; helmet 4-label scheme `{helmet, no_helmet, turban, uncertain}` with rule-layer mapping (`turban → exempt`, `uncertain → abstain`); schema-validated |
| Dataset registry + policy | `registry/`, `docs/dataset-policy.md`, `docs/evaluation-protocol.md` | governance/metadata only, each entry with explicit access/licence status; **no dataset is downloaded** |
| Scene configuration + stable hashing | `configs/scenes/`, `src/trafficpulse/contracts/scene.py` | typed `SceneConfig` + deterministic `scene_config_hash` (SHA-256) |
| Geometry primitives | `src/trafficpulse/geometry/` | vectors, segments, polygons (P1-U1) |
| Synthetic trajectory generator | `src/trafficpulse/synth/` | golden trajectories with known event labels — no model, no video (P1-U2) |
| Generic rule-engine core | `src/trafficpulse/rules/engine.py`, `states.py` | violation-agnostic hypothesis lifecycle FSM + accumulation + abstention (P1-U3) |
| Wrong-way reasoning | `src/trafficpulse/rules/wrong_way.py`, `observations/heading.py` | sustained-contradiction → `ConfirmedEvent`, validated on synthetic tracks only (P1-U4) |
| PTS-accurate video ingestion | `src/trafficpulse/ingestion/video.py` | PyAV backend; media-relative timestamps from PTS only (no fabricated FPS fallback); deterministic frame identity (P1-U5) |

## Planned capabilities / roadmap

Defined by contract or design, **not yet implemented**:

- **Detector integration** — permissive-only posture, RT-DETR as the primary
  direction, behind the U2 `Detection` contract (ADR-001). The integration gate
  is lifted, but no detector code exists yet.
- **Tracker integration** — ByteTrack / OC-SORT behind the tracking contract.
- **Real-video end-to-end** — the wrong-way slice against a real clip (currently
  synthetic-only); no real footage has been processed.
- **Remaining five violations** — no-helmet (hosts the mandatory CNN-vs-ViT
  experiment), triple riding, red-light jumping, illegal stopping, and
  feasibility-gated speeding.
- **ANPR, evidence-engine runtime, observation writer (Parquet), event store,
  privacy/redaction, human-review UI, simulated-penalty workflow, analytics /
  evaluation code.**

No model weights, datasets, or training pipelines are included.

## Repository structure

```text
TrafficPulse/
├── LICENSE                     # Apache-2.0 (project source code)
├── README.md
├── SECURITY.md
├── pyproject.toml              # packaging, ruff/mypy/pytest config
├── TRAFFICPULSE_MASTER_SPEC.md # product/research specification
├── .github/workflows/ci.yml    # single-env Linux quality gate
├── configs/
│   ├── ontology.yaml
│   └── scenes/                 # scene schema + synthetic example
├── registry/                   # dataset registry schema + candidate entries
│   ├── schema.yaml
│   └── datasets/*.yaml
├── schemas/                    # exported JSON schemas
├── docs/
│   ├── architecture-review.md  # canonical architecture reference
│   ├── architecture.md · phase-0-plan.md
│   ├── ontology.md · dataset-policy.md · evaluation-protocol.md
│   ├── scene-configuration.md · windows-verification.md
│   └── adr/ADR-001..004.md
├── src/trafficpulse/
│   ├── contracts/ · geometry/ · synth/
│   ├── rules/ · observations/ · ingestion/
└── tests/                      # contracts, geometry, synth, rules,
                                # observations, ingestion, ontology,
                                # registry, scenes, docs
```

Packages appear only when a unit needs them — there is no speculative scaffold.

## Quick start

Requires **Python ≥ 3.11**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2. Install the package (editable) plus the dev tooling
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 3. Run the quality gates
ruff check .
mypy src
pytest -q

# 4. Import / version smoke check
python -c "import trafficpulse; print(trafficpulse.__version__)"
```

Native-Windows verification steps are recorded in
[`docs/windows-verification.md`](docs/windows-verification.md).

## Quality gates / testing

- **Lint/format:** `ruff check .`
- **Types:** `mypy src` (strict mode).
- **Tests:** `pytest -q` (currently 485 passing).
- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same
  three checks on Linux for every push to `main` and every pull request.

Every source package is exercised by at least one test; the ADR/architecture
invariants themselves are checked by `tests/docs/test_adr_pack.py`.

## Dataset and model policy

- **No dataset is downloaded or vendored** by this repository. `registry/` holds
  *governance metadata only* — provenance, access status, licence status, task
  fit, split/leakage metadata — so decisions are reviewable **before** any
  acquisition (`docs/dataset-policy.md`, `docs/evaluation-protocol.md`).
- **No model weights** are included. Detector/tracker/OCR selection follows a
  permissive-only posture (ADR-001); weight and pretraining-data terms are
  reviewed **per artifact** before any use or distribution.
- Leakage prevention (group-based splits, whole-site holdout, validation-only
  threshold tuning) is frozen as policy before any training occurs.

## Licensing distinctions

TrafficPulse deliberately keeps four licence questions **separate** — the project
licence does not extend to any of the others:

| Scope | Governed by |
|---|---|
| **TrafficPulse source code** | **Apache-2.0** (this repository — see [`LICENSE`](LICENSE)) |
| Detector / framework components | Independently; permissive-only posture under **ADR-001** (not yet integrated) |
| Datasets | Independently, by `registry/` metadata + `docs/dataset-policy.md` and each dataset's own terms |
| Model weights / artifacts | Independently, by their own upstream terms (reviewed per artifact) |

Adopting Apache-2.0 for this repository does **not** relicense datasets, model
weights, or third-party code, and ADR-001 remains specifically about the
detector-stack licence posture — not the project licence.

## Research & deployment limitations

- **This is not a production enforcement system.** It is a research/academic
  capstone foundation.
- **A confirmed event is not a legal determination of guilt.** Human review is a
  mandatory design step before any *simulated* penalty, and all penalty artefacts
  are simulation-only.
- **No validated real-world accuracy.** No detector or tracker is integrated, no
  real footage has been processed, and no accuracy, throughput, or event-level
  precision/recall number is claimed. Real-world accuracy requires dataset-backed
  evaluation that has not been performed.
- **Wrong-way reasoning is validated on synthetic trajectories only.**
- Deployment assumptions stay bounded by the current project scope; speeding in
  particular is feasibility-gated and excluded from any penalty simulation until
  a calibrated-scene evaluation justifies it (`docs/evaluation-protocol.md` §11).

## Documentation

- [`TRAFFICPULSE_MASTER_SPEC.md`](TRAFFICPULSE_MASTER_SPEC.md) — product/research specification
- [`docs/architecture-review.md`](docs/architecture-review.md) — **canonical** architecture & feasibility reference
- [`docs/architecture.md`](docs/architecture.md) — entry point + ADR index
- [`docs/phase-0-plan.md`](docs/phase-0-plan.md) — Phase 0-F foundation plan
- [`docs/phase-1-plan.md`](docs/phase-1-plan.md) — authoritative Phase 1 unit plan (completed P1-U1…P1-U7; forward critical path from P1-U8)
- [`docs/ontology.md`](docs/ontology.md) · [`docs/dataset-policy.md`](docs/dataset-policy.md) · [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) · [`docs/scene-configuration.md`](docs/scene-configuration.md)
- [`docs/adr/`](docs/adr/) — architecture decision records (ADR-001..004)
- [`docs/windows-verification.md`](docs/windows-verification.md) — native-Windows check record

## Contributing

This is currently a research-stage capstone repository without a formal
contribution process. Issues and discussion are welcome; please read the
architecture reference and ADRs first so proposals fit the frozen contracts and
the evidence-first design. Security reports follow [`SECURITY.md`](SECURITY.md).

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE).
