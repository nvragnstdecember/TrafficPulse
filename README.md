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
- **Evaluation + real-time engine (H5–H6): complete** — a pure evaluation
  framework and a deterministic real-time inference engine that composes the
  shipped seams (ingestion → detection → tracking → reasoning → confirmation →
  evidence → persistence) with scheduling, batching, metrics, and structured logs.
- **Web application (H7A–H7E): complete** — a FastAPI HTTP API over the engine
  (upload, processing jobs, events, evidence, metrics) and a React single-page app:
  a video workspace (upload → live processing → frame-accurate review), a resilient
  live lifecycle (polling, cancellation, recovery, graceful failures), and an
  analyst review layer (severity, multi-select, an evidence viewer, notes, and
  JSON/CSV export). See [`frontend/README.md`](frontend/README.md) and the
  [Deployment guide](docs/deployment.md).
- **Release hardening (H8): complete** — deployment configuration (opt-in CORS +
  single-process SPA serving), production documentation, and a demo-ready first-run
  experience.
- **Review, recovery, and library (H9–H12): complete** — an append-only analyst
  decision journal, startup recovery that rebuilds the runtime indices from disk,
  a browsable historical video library, and per-video scene calibration.
- **Red-light jumping (H13): complete** — the fifth locked violation, with support
  latched at the stop-line crossing.
- **Evidence rendering (H14): complete** — content-addressed, hash-verified
  evidence frames drawn by the shared overlay renderer, merged into the manifest at
  read time, and downloadable as a deterministic ZIP package.
- **Dashboard & analytics (H15): complete** — a single server-side aggregation
  layer behind `GET /api/analytics/summary`, with the dashboard and analytics pages
  built on it.
- **Platform hardening (H16): complete** — production logging with request
  correlation, an env-driven production launcher (`serve.py`), containerisation
  (Dockerfile + compose), readiness reporting, and an evidence render lifecycle
  with a repair path.
- **Mandatory CNN-vs-ViT experiment (P4-U5): complete** — the master-spec §4
  requirement, executed on the CC-BY-4.0 HELMET dataset under a pre-registration
  frozen and git-tagged before the final runs. Under the frozen decision rule
  **ResNet-50 wins the accuracy comparison** (mean test macro-F1 0.92881 vs 0.91975
  for DeiT-Small; pooled 95% CI [-0.01380, -0.00434], sign-consistent across three
  seeds), while **DeiT-Small is the cheaper model** on the measured RTX 4060 Laptop
  benchmark (-29.8% latency, -34.2% peak VRAM at batch 32). Neither model is adopted
  into the runtime: it trained on whole-motorcycle crops, not the runtime's derived
  head crops. See [`docs/cnn-vs-vit-results.md`](docs/cnn-vs-vit-results.md) and
  [ADR-005](docs/adr/ADR-005.md).
- **Live camera monitoring: complete** — a browser camera streamed into a
  persistent backend session that runs the **same** pipeline an uploaded video runs
  (same engine provider, same detector threshold, same tracker, association, helmet
  classifier and violation reasoners; no live-only rule anywhere). One WebSocket
  carries the session; back-pressure keeps exactly one frame pending server-side and
  two in flight client-side, so latency and memory cannot accumulate. Measured on
  this machine's CPU: **~1 fps of AI inference and ~1.6 s end-to-end delay** — the
  camera preview stays smooth because it is deliberately not analysed
  frame-for-frame. No camera frame or live event is written to disk. See
  [`docs/live-camera.md`](docs/live-camera.md).
- **Not started:** speeding (the sixth locked violation), real-footage validation,
  ANPR, and simulated penalties (see [Roadmap](#planned-capabilities--roadmap)).

Quality gates are green: `ruff`, `mypy src` (strict), **2957 passing backend tests**
(10 opt-in real-model/GPU tests skipped by default) plus **587 passing frontend
tests**, on the current tree, with single-environment Linux CI and a
native-Windows verification checklist.

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

Defined by contract or design, **not yet implemented** (sequenced across Phases 3–5
per the accepted design review — see the phase plans in [Documentation](#documentation)):

- **Remaining violations** — feasibility-gated speeding (Phase 5). Red-light jumping,
  triple riding, and no-helmet have since shipped, and the mandatory CNN-vs-ViT
  experiment that no-helmet hosted is **complete** (see
  [`docs/cnn-vs-vit-results.md`](docs/cnn-vs-vit-results.md)); a **trained** helmet
  classifier remains outstanding, because that experiment measured whole-motorcycle
  crops rather than the runtime's head crops and [ADR-005](docs/adr/ADR-005.md)
  therefore adopts neither candidate. Phases 3–5 also deliver generalized
  reasoning/pipeline infrastructure (by composition), a dynamic traffic-context stream,
  the observation-log substrate, the event-level evaluation harness, and metric
  calibration.
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
python -m pip install -e ".[dev,api]"   # 'api' enables the FastAPI web layer

# 3. Run the quality gates
ruff check .
mypy src
pytest -q

# 4. Import / version smoke check
python -c "import trafficpulse; print(trafficpulse.__version__)"
```

Native-Windows verification steps are recorded in
[`docs/windows-verification.md`](docs/windows-verification.md).

### Run the web application

`frontend/` is the single user-facing TrafficPulse UI. There are two launch paths
and no others: **development** (Vite dev server in front of the API) and
**built** (the API serves the compiled SPA from one process).

```bash
# Development — terminal 1: API
uvicorn serve:app --reload --port 8000

# Development — terminal 2: SPA (proxies /api to the API)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

```bash
# Built / demo — one process, one port
cd frontend && npm run build && cd ..          # → frontend/dist
TRAFFICPULSE_APP_STATIC_DIR=frontend/dist uvicorn serve:app --port 8000
```

`serve:app` is the canonical entrypoint for both: it is `AppConfig.from_env()` plus
the code-level model composition, and it falls back to the shipped example scene, so
uploads process and the live camera works without any environment variable.
`trafficpulse.app.asgi:app` is the environment-only composition with **no inference
backend** — it serves every read endpoint and the whole UI, but processing returns
`503 engine_unavailable`. `frontend/dist` is strictly `npm run build` output, never a
second UI.

The full setup, environment variables, production topologies (reverse-proxy or
single-process SPA serving), real-inference configuration, demo workflow, and
troubleshooting are in the **[Deployment & Operations guide](docs/deployment.md)**.
The frontend architecture and scripts are in
[`frontend/README.md`](frontend/README.md).

### Monitor a live camera

With both processes running, open **Live camera** in the app, press **Start
camera**, grant access, then press **Start monitoring**. Turning the camera on never
starts analysis by itself; monitoring is always an explicit second act. Camera
access needs a secure context, so use `http://localhost` or HTTPS. Architecture, the
WebSocket protocol, measured throughput on this hardware, and the limitations are in
**[`docs/live-camera.md`](docs/live-camera.md)**.

### Run the demonstration composition

`serve_demo.py` is a second, explicitly-labelled composition for demonstrating helmet
perception: it selects the trained P4-U5 ResNet-50 (by far the strongest classifier on
runtime crops) and declares helmet **analysis** rather than the helmet violation rule.

```bash
uvicorn serve_demo:app --host 127.0.0.1 --port 8000
```

It is **not** a production adoption and does not change the default: `serve.py` still
composes the zero-shot backend, and ADR-005 still adopts none. The trained model is
binary and cannot emit `turban`, so the classifier capability guard refuses to build a
no-helmet rule around it — and this launcher does not bypass that guard. It classifies
and reports; it confirms no helmet violation. What may and may not be claimed from it is
written down in the **[Demonstration guide](docs/demo-guide.md)**; the evidence behind
those limits is in
[`docs/helmet-runtime-evaluation.md`](docs/helmet-runtime-evaluation.md).

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

**Backend**

- **Lint/format:** `ruff check .`
- **Types:** `mypy src` (strict mode).
- **Tests:** `pytest -q` (currently 2957 passing; 10 opt-in real-model/GPU tests
  skipped by default). Install the `api` and `overlay` extras so the web layer and
  the overlay renderer are both type-checked and tested — this is exactly what CI
  installs: `pip install -e ".[dev,api,overlay]"`. Without `overlay`, `mypy src`
  cannot resolve `PIL` and the renderer tests skip.

**Frontend** (from `frontend/`)

- **Types:** `npm run typecheck` · **Lint:** `npm run lint` · **Build:**
  `npm run build` · **Tests:** `npm run test` (587 passing, jsdom, mocked API — no
  backend). Coverage via `npm run coverage`.

- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the backend
  checks on Linux for every push to `main` and every pull request.

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
- **Live camera monitoring runs at ~1 fps of AI inference on this machine's CPU**,
  with ~1.6 s between the road and the screen (measured; see
  [`docs/live-camera.md`](docs/live-camera.md) §6). It is not real-time. Its
  analysis window resets tracker state every 600 processed frames to bound memory,
  so track ids restart there and a violation straddling that boundary is not
  confirmed. Live events are not persisted and do not enter the repository, the
  analytics or the review workflow.
- **The browser camera path has not been exercised against real hardware.** The
  live session was validated end to end against the real RT-DETR and helmet
  backends over a real WebSocket, and the browser capture/teardown path is covered
  by tests against recording doubles — but the development machine exposes no
  physical camera, so `getUserMedia` → canvas → socket has never run against one.
- Deployment assumptions stay bounded by the current project scope; speeding in
  particular is feasibility-gated and excluded from any penalty simulation until
  a calibrated-scene evaluation justifies it (`docs/evaluation-protocol.md` §11).

## Documentation

- [`TRAFFICPULSE_MASTER_SPEC.md`](TRAFFICPULSE_MASTER_SPEC.md) — product/research specification
- [`docs/deployment.md`](docs/deployment.md) — **deployment & operations guide** (setup, run, env, CORS/static, health, demo, troubleshooting)
- [`docs/demo-guide.md`](docs/demo-guide.md) — **demonstration guide**: what `serve_demo.py` shows, and exactly what may and may not be claimed from it
- [`docs/live-camera.md`](docs/live-camera.md) — **live camera monitoring**: architecture, session lifecycle, frame transport, back-pressure, measured performance, and limitations
- [`frontend/README.md`](frontend/README.md) — frontend architecture, workspace, live processing, review workflow, keyboard shortcuts
- [`docs/architecture-review.md`](docs/architecture-review.md) — **canonical** architecture & feasibility reference
- [`docs/architecture.md`](docs/architecture.md) — entry point + ADR index
- [`docs/phase-0-plan.md`](docs/phase-0-plan.md) — Phase 0-F foundation plan
- [`docs/phase-1-plan.md`](docs/phase-1-plan.md) — authoritative Phase 1 unit plan (completed P1-U1…P1-U12)
- [`docs/phase-2-plan.md`](docs/phase-2-plan.md) — authoritative Phase 2 unit plan (evidence integrity + illegal stopping; completed P2-U1…P2-U7)
- [`docs/phase-3-plan.md`](docs/phase-3-plan.md) — authoritative Phase 3 plan (generalized reasoning/pipeline infrastructure, dynamic traffic context, red-light jumping, observation-log substrate, event-level evaluation harness; planned)
- [`docs/phase-4-plan.md`](docs/phase-4-plan.md) — authoritative Phase 4 plan (association, quality-weighted confidence aggregation, triple riding, no-helmet + CNN-vs-ViT experiment; the experiment is complete — see docs/cnn-vs-vit-results.md)
- [`docs/phase-5-plan.md`](docs/phase-5-plan.md) — authoritative Phase 5 plan (metric calibration, feasibility-gated speeding, retro-upgrade of provisional pixel gates; planned)
- [`docs/ontology.md`](docs/ontology.md) · [`docs/dataset-policy.md`](docs/dataset-policy.md) · [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) · [`docs/scene-configuration.md`](docs/scene-configuration.md)
- [`docs/cnn-vs-vit-results.md`](docs/cnn-vs-vit-results.md) — **CNN-vs-ViT helmet experiment results** (P4-U5): pre-registered outcome, statistics, calibration, robustness, cost benchmark, and limitations
- [`docs/adr/`](docs/adr/) — architecture decision records (ADR-001..005)
- [`docs/windows-verification.md`](docs/windows-verification.md) — native-Windows check record

## Contributing

This is currently a research-stage capstone repository without a formal
contribution process. Issues and discussion are welcome; please read the
architecture reference and ADRs first so proposals fit the frozen contracts and
the evidence-first design. Security reports follow [`SECURITY.md`](SECURITY.md).

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE).
