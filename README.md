# TrafficPulse

**Evidence-first, offline-first traffic-violation reasoning for fixed-camera roadside video.**

TrafficPulse is a research-oriented engineering project that treats a traffic
violation as something to be *reasoned about over time and backed by evidence* —
never as a single model output or a single-frame detection. Its central design
commitment is a hard separation between **perception** (what a model sees in a
frame) and **violation reasoning** (typed observations accumulated over a track,
combined with scene geometry and explicit rules, confirmed only with temporal
evidence and reviewable by a human). This repository contains the frozen data
contracts and governance **plus two offline, deterministic violation-reasoning
slices — wrong-way and illegal-stopping — wired end to end from recorded video
through a real detector, a real tracker, typed observations, temporal reasoning,
confirmed-event minting, and minimal-manifest persistence.** It is a research
foundation, **not** a production enforcement system.

---

## Project status

**Research foundation + two end-to-end offline violation slices.** The perception
seam (real detector + tracker) and the reasoning/persistence path are implemented
and tested; validation is on synthetic trajectories and generated synthetic clips
only. Concretely:

- **Phase 0-F (foundations): complete** — typed domain contracts, exported JSON
  schemas, label ontology, dataset registry + policies, scene-configuration
  schema, and the architecture/ADR pack are all in place and tested.
- **Phase 1 (first vertical slice): complete** — geometry primitives, a synthetic
  trajectory generator, a generic rule-engine core, wrong-way temporal reasoning,
  PTS-accurate video ingestion, RT-DETR detector integration, IoU tracking,
  minimal event persistence + evidence manifests, the wrong-way pipeline, and a
  recorded-clip end-to-end slice runner have all landed and are unit-tested.
- **Phase 2 (evidence integrity + second violation): complete** — truthful
  detector/tracker model-provenance propagation onto confirmed events, in-zone and
  stationary observation derivation, an illegal-stopping temporal reasoner, an
  illegal-stopping pipeline, and a recorded-clip end-to-end illegal-stopping slice
  runner.
- **Not started:** the remaining four locked violations, real-footage validation,
  the full evidence-rendering engine, human review, and simulated penalties (see
  [Roadmap](#planned-capabilities--roadmap)).

Quality gates are green: `ruff`, `mypy src` (strict), and **946 passing tests** (4
opt-in real-model tests skipped by default) on the current tree, with
single-environment Linux CI and a native-Windows verification checklist.

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
and (feasibility-gated) speeding. Of these, **two** reasoning paths exist today —
**wrong-way** and **illegal-stopping** — each validated on *synthetic* trajectories
and on *generated synthetic clips* (through real ingestion, real tracking, real
rules, and real persistence), with detections injected by a scripted stub. No real
footage has been processed and no real-world accuracy is claimed.

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
    classDef partial fill:#8a6d1f,stroke:#d9a637,color:#fff;
    classDef todo fill:#555,stroke:#999,color:#fff,stroke-dasharray:4 3;
    class V,D,T,A,O,R,E done;
    class EV partial;
    class RC,P todo;
```

*Green = implemented and covered by tests (real PTS ingestion, RT-DETR detection,
IoU tracking + association, typed observations, the rule engine, and confirmed-event
minting). Amber = minimal implementation (a reviewable `EvidenceManifest` stub and a
deterministic JSON event store — **no** clip/frame rendering, crops, overlays, or
media hashing yet). Grey/dashed = contract defined, behaviour planned (human review,
simulated penalty).*

Design posture, supported by the project docs:

- **Offline-first** — recorded video in, events out; "real-time" language is
  confined to a labelled near-real-time demo mode (ADR-003,
  `docs/architecture-review.md` §22).
- **Evidence-oriented** — every confirmed event is meant to be explainable from
  stored evidence and rule context; confirmed events carry truthful detector/tracker
  model references (`docs/architecture-review.md` §19).
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
| Detector integration (RT-DETR) | `src/trafficpulse/detector/` | permissive-only RT-DETR backend behind the `Detection` contract, a `DetectionAdapter` seam, and a scripted `StubDetector`; torch/transformers are an optional extra, lazily imported (P1-U6/U7) |
| Tracker integration (IoU) | `src/trafficpulse/tracking/` | in-repo greedy-IoU associator + `TrackState` adapter and a scripted `StubTracker`, behind the tracking contract (P1-U8/U9) |
| PTS-accurate video ingestion | `src/trafficpulse/ingestion/video.py` | PyAV backend; media-relative timestamps from PTS only (no fabricated FPS fallback); deterministic frame identity (P1-U5) |
| Heading / in-zone / stationary observations | `src/trafficpulse/observations/` | typed per-frame facts: heading-vs-lane (P1-U4), in-zone membership (P2-U2), and pixel-space stationarity (P2-U3), all with taint handling |
| Wrong-way reasoning | `src/trafficpulse/rules/wrong_way.py`, `observations/heading.py` | sustained-contradiction → `ConfirmedEvent` (P1-U4) |
| Illegal-stopping reasoning | `src/trafficpulse/rules/illegal_stopping.py` | stationary-in-zone dwell → `ConfirmedEvent`; joins the in-zone + stationary streams; taint/recovery abstention (P2-U4) |
| Vertical-slice pipelines | `src/trafficpulse/pipeline/` | thin, deterministic offline orchestration: `WrongWayPipeline` (P1-U10) and `IllegalStoppingPipeline` (P2-U5), each detector/tracker-backend-agnostic |
| Model-provenance propagation | `src/trafficpulse/pipeline/provenance.py` | confirmed events + manifests carry truthful, sorted/de-duplicated detector/tracker `ModelRef`s (name + version; `weights_hash` not computed) (P2-U1) |
| Minimal event persistence + evidence manifest | `src/trafficpulse/persistence/` | deterministic per-run JSON `EventStore` (write-once, idempotent replay) + a minimal reviewable `EvidenceManifest` stub — no rendering/hashing (P1-U11) |
| Recorded-clip slice runners / demos | `src/trafficpulse/pipeline/runner.py`, `pipeline/illegal_stopping_runner.py` | offline composition roots that decode a real clip and persist confirmed events; real RT-DETR built in the CLI, scripted stub injected in tests (P1-U12, P2-U6) |

Both violation slices run end to end offline on a **recorded synthetic clip**
through real ingestion, the real IoU tracker, real reasoning, and real persistence —
deterministically and with byte-identical persisted files on replay — with
detections supplied by an injected scripted stub.

## Planned capabilities / roadmap

Defined by contract or design, **not yet implemented**:

- **Remaining four violations** — no-helmet (hosts the mandatory CNN-vs-ViT
  experiment), triple riding, red-light jumping, and feasibility-gated speeding.
- **Real-footage validation** — an external, gated activity (permissions/ethics +
  approved footage + a matching validated `SceneConfig`); no real footage has been
  processed, and the shipped pipelines run it with **no** new code once footage is
  approved.
- **Congestion-robust / ID-churn-robust illegal stopping** — the first slice
  excludes congested scenes and does not re-associate a long-stationary vehicle
  across a tracker ID switch (both explicit, documented deferrals).
- **Full evidence-engine runtime** — clip/frame rendering, crops, overlays,
  content-addressed media hashing, and OCR (the current manifest is a minimal
  reference stub only).
- **Durable storage** — SQLite + Parquet observation/event logs (ADR-002 defers
  this; the JSON `EventStore` is the current storage posture, ADR-004 stays
  *Proposed*).
- **ANPR, privacy/redaction, human-review UI, simulated-penalty workflow, and
  analytics / evaluation-harness code.**

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
│   ├── architecture.md · phase-0-plan.md · phase-1-plan.md · phase-2-plan.md
│   ├── ontology.md · dataset-policy.md · evaluation-protocol.md
│   ├── scene-configuration.md · windows-verification.md
│   └── adr/ADR-001..004.md
├── src/trafficpulse/
│   ├── contracts/ · geometry/ · synth/ · ingestion/
│   ├── detector/ · tracking/ · observations/
│   ├── rules/ · pipeline/ · persistence/
└── tests/                      # contracts, geometry, synth, rules, observations,
                                # ingestion, detector, tracking, pipeline,
                                # persistence, ontology, registry, scenes, docs
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

## Vertical-slice demos (offline)

Two offline, deterministic commands run the violation slices end to end — one
recorded clip + one `SceneConfig` → PTS-accurate frames → real RT-DETR detection →
IoU tracking → temporal reasoning → a persisted `ConfirmedEvent` with a minimal
`EvidenceManifest`. Each needs the optional detector extra and a locally-available
checkpoint (nothing is downloaded by default):

```bash
python -m pip install -e ".[dev,rtdetr]"      # optional torch/transformers extra

# Wrong-way slice
python -m trafficpulse.pipeline \
  --clip path/to/clip.mp4 \
  --scene configs/scenes/example-scene.yaml \
  --output-dir runs --run-id demo-ww \
  --checkpoint <locally-cached-rtdetr-id-or-dir> --device cpu \
  --direction-id dir-north

# Illegal-stopping slice (no --direction-id; the no-stopping zone is resolved
# from the scene, which must declare a no_stopping zone matching the footage)
python -m trafficpulse.pipeline.illegal_stopping_runner \
  --clip path/to/clip.mp4 \
  --scene path/to/scene-with-no-stopping-zone.yaml \
  --output-dir runs --run-id demo-stop \
  --checkpoint <locally-cached-rtdetr-id-or-dir> --device cpu
```

Both are **fully offline** (pass `--allow-download` only to let `transformers` fetch
a checkpoint), fail fast with a typed message on a missing clip / invalid scene /
missing checkpoint, and write only under `--output-dir/<run-id>` (gitignored). Each
run prints a JSON report (frame/track/event counts, detector/tracker refs, scene
hash) and makes **no** accuracy claim from a single clip.

A COCO RT-DETR does not fire the vehicle class on synthetic pixels, so a genuine
event requires an **approved real clip**; each whole path (real ingestion, real
tracker, real rules, real persistence) is otherwise verified deterministically on a
generated clip in `tests/pipeline/` with injected detections
(`test_slice_runner.py`, `test_illegal_stopping_e2e.py`). Real RT-DETR inference is
proven end to end by the opt-in `tests/pipeline/test_slice_e2e_rtdetr.py` (and the
illegal-stopping opt-in in `test_illegal_stopping_e2e.py`) — set
`TRAFFICPULSE_E2E_MODEL` to a locally-cached checkpoint; skipped by default.

## Quality gates / testing

- **Lint/format:** `ruff check .`
- **Types:** `mypy src` (strict mode).
- **Tests:** `pytest -q` (currently 946 passing; 4 opt-in real-model tests skipped
  by default).
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
  reviewed **per artifact** before any use or distribution. The RT-DETR backend is
  an optional extra and loads a **locally-available** checkpoint only (offline by
  default); no checkpoint is committed.
- Leakage prevention (group-based splits, whole-site holdout, validation-only
  threshold tuning) is frozen as policy before any training occurs.

## Licensing distinctions

TrafficPulse deliberately keeps four licence questions **separate** — the project
licence does not extend to any of the others:

| Scope | Governed by |
|---|---|
| **TrafficPulse source code** | **Apache-2.0** (this repository — see [`LICENSE`](LICENSE)) |
| Detector / framework components | Independently; permissive-only posture under **ADR-001** |
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
  are simulation-only (and not yet implemented).
- **No validated real-world accuracy.** No real footage has been processed, and no
  accuracy, throughput, or event-level precision/recall number is claimed.
  Real-world accuracy requires dataset-backed evaluation on approved footage that
  has not been performed.
- **Wrong-way and illegal-stopping reasoning are validated on synthetic
  trajectories and generated synthetic clips only**, with detections supplied by an
  injected scripted stub. A COCO RT-DETR does **not** fire the vehicle class on
  those synthetic pixels, so no confirmed event has been produced by a real detector
  on real pixels; genuine RT-DETR inference is exercised only through the opt-in
  end-to-end tests.
- **Illegal stopping is not congestion-robust or ID-churn-robust.** The first slice
  targets non-congested, single-vehicle synthetic scenes and does not re-associate a
  long-stationary vehicle across a tracker ID switch (both explicit deferrals); its
  `motion_threshold` is recorded for provenance but not applied (uncalibrated slice).
- Deployment assumptions stay bounded by the current project scope; speeding in
  particular is feasibility-gated and excluded from any penalty simulation until
  a calibrated-scene evaluation justifies it (`docs/evaluation-protocol.md` §11).

## Documentation

- [`TRAFFICPULSE_MASTER_SPEC.md`](TRAFFICPULSE_MASTER_SPEC.md) — product/research specification
- [`docs/architecture-review.md`](docs/architecture-review.md) — **canonical** architecture & feasibility reference
- [`docs/architecture.md`](docs/architecture.md) — entry point + ADR index
- [`docs/phase-0-plan.md`](docs/phase-0-plan.md) — Phase 0-F foundation plan
- [`docs/phase-1-plan.md`](docs/phase-1-plan.md) — authoritative Phase 1 unit plan (completed P1-U1…P1-U12)
- [`docs/phase-2-plan.md`](docs/phase-2-plan.md) — authoritative Phase 2 unit plan (evidence integrity + illegal stopping; completed P2-U1…P2-U7)
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
