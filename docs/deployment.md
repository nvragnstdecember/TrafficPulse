# TrafficPulse — Deployment & Operations Guide (H8)

Production setup, run, and operations for the TrafficPulse **web application**: the
H7A FastAPI service over the H6 inference engine, and the H7B–H7E React single-page
app (SPA) that reviews confirmed violations against the uploaded video.

> This is a research foundation, **not** a production traffic-enforcement system,
> and it makes no validated real-world accuracy claim. See
> [Known limitations](#known-limitations).

---

## 1. Architecture at a glance

```
Browser (React SPA)
  → TanStack Query → typed API client (JSON only, no direct fetch)
      → FastAPI (trafficpulse.app)  ── /api/health /api/video/upload /api/process /api/events /api/evidence /api/metrics
          → Application services (validate, drive jobs, read events/evidence/metrics)
              → Inference engine (H6): decode → detect → track → reason → confirm → evidence
                  → Persistence (write-once JSON event + manifest store)
```

The SPA depends only on the JSON contract; it never imports backend code. The
engine, detector, tracker, and rules stay server-side.

**Two supported topologies**

- **Two processes (recommended for dev, common in prod):** a static host / reverse
  proxy serves the SPA and proxies `/api` to the FastAPI process.
- **Single process (simplest prod):** FastAPI serves the built SPA itself
  (`TRAFFICPULSE_APP_STATIC_DIR`), so one process serves both the app and the API.

---

## 2. Prerequisites

- **Python** ≥ 3.11 (3.12 used in CI).
- **Node.js** ≥ 18 (for the frontend build; Node 22 used locally).
- No system FFmpeg required — PyAV ships its own; no GPU required unless you enable
  the real RT-DETR backend.

---

## 3. Developer setup

### Backend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"     # 'api' pulls in FastAPI + multipart
```

The `api` extra is required to run or type-check the application layer (the base
install is intentionally web-framework-free). Add `rtdetr` as well to enable the
real detector (see [§9](#9-real-inference-rt-detr)).

### Frontend

```bash
cd frontend
npm install
```

---

## 4. Running in development

**Terminal 1 — API** (a scene is required before processing; the example scene works):

```bash
export TRAFFICPULSE_APP_SCENE=configs/scenes/example-scene.yaml   # Windows: set / $env:
uvicorn trafficpulse.app.asgi:app --reload --port 8000
```

**Terminal 2 — SPA** (Vite dev server proxies `/api` to `127.0.0.1:8000`):

```bash
cd frontend && npm run dev        # http://localhost:5173
```

Open the dev URL; the workspace is under **Videos**. Health is at
`http://127.0.0.1:8000/api/health`; interactive API docs at `/docs`.

---

## 5. Environment variables

All backend configuration is read once by `AppConfig.from_env()` (see
`src/trafficpulse/app/config.py`). Every path is relative or operator-supplied —
no absolute path is ever assumed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRAFFICPULSE_APP_STORAGE` | `trafficpulse-data` | Root for uploads (`/videos`) and run outputs (`/runs`). |
| `TRAFFICPULSE_APP_SCENE` | _(none)_ | Path to the governing `SceneConfig` (JSON/YAML). Required before a job runs; its absence surfaces as a clean HTTP error, not a crash. |
| `TRAFFICPULSE_APP_HOST` | `127.0.0.1` | Advisory bind host (travels with the config; pass to your ASGI server). |
| `TRAFFICPULSE_APP_PORT` | `8000` | Advisory bind port. |
| `TRAFFICPULSE_APP_MAX_UPLOAD_BYTES` | `536870912` (512 MiB) | Hard upload size cap (enforced while streaming). |
| `TRAFFICPULSE_APP_CORS_ORIGINS` | _(none)_ | Comma-separated browser origins allowed to call the API cross-origin. Empty adds **no** CORS middleware. |
| `TRAFFICPULSE_APP_STATIC_DIR` | _(none)_ | Directory of a built SPA (`frontend/dist`) to serve from the API at `/`. Empty serves the JSON API only. |

The frontend's build/runtime knobs (`VITE_API_BASE_URL`, `VITE_API_TIMEOUT_MS`,
`VITE_API_PROXY_TARGET`, `VITE_MAX_UPLOAD_BYTES`, `VITE_ACCEPTED_VIDEO_FORMATS`) are
documented in [`frontend/README.md`](../frontend/README.md#configuration).

---

## 6. Production deployment

### Build the SPA

```bash
cd frontend && npm run build        # → frontend/dist (type-checked, code-split)
```

### Topology A — reverse proxy (SPA static + `/api` proxy)

Serve `frontend/dist` from any static host (nginx, Caddy, S3+CDN) and proxy `/api`
to the FastAPI process. Configure the static host to **fall back to `index.html`**
for unknown paths so client-side routes (e.g. `/videos`) survive a refresh. Run the
API with a production ASGI server:

```bash
uvicorn trafficpulse.app.asgi:app --host 0.0.0.0 --port 8000
# or: gunicorn -k uvicorn.workers.UvicornWorker trafficpulse.app.asgi:app
```

Because the SPA and API share an origin, no CORS is needed. If the SPA is served
from a **different** origin, set `TRAFFICPULSE_APP_CORS_ORIGINS` to that origin and
build the SPA with `VITE_API_BASE_URL` pointing at the API.

### Topology B — single process (FastAPI serves the SPA)

```bash
export TRAFFICPULSE_APP_STATIC_DIR=frontend/dist
export TRAFFICPULSE_APP_SCENE=configs/scenes/example-scene.yaml
uvicorn trafficpulse.app.asgi:app --host 0.0.0.0 --port 8000
```

FastAPI then serves the SPA at `/` (hashed assets under `/assets`) with an
`index.html` fallback for client-side routes, while `/api/*` always takes
precedence. This needs no reverse proxy and no CORS. Note: with the SPA mounted,
an unknown `/api/...` path returns the app shell rather than a JSON 404 (the mount
is the catch-all); real API routes are unaffected.

---

## 7. CORS

CORS is **opt-in**. With `TRAFFICPULSE_APP_CORS_ORIGINS` unset there is no CORS
middleware and no cross-origin surface — correct for same-origin and dev-proxy
deployments. Set it to a comma-separated allow-list only when the browser origin
differs from the API origin.

---

## 8. Health & readiness

`GET /api/health` returns `{"status","version","engine"}`:

- `status` is `ok` when the service is serving.
- `engine` reports backend readiness: `ready` when a real inference backend is
  available, else `unconfigured` (the service still serves every read endpoint and
  stub-injected jobs).

Use it as a liveness/readiness probe. `GET /api/metrics` exposes aggregate job
counts (`jobs_total/pending/running/succeeded/failed/cancelled`, `events_total`)
plus the latest run's engine metrics.

---

## 9. Real inference (RT-DETR)

The default env-configured server has **no** real detector (`engine: unconfigured`):
it serves every read endpoint and the full UI, but a processing request returns a
clean `503 engine_unavailable`, which the workspace surfaces gracefully.

To run real detection:

1. `pip install -e ".[dev,api,rtdetr]"` (Apache-2.0 Transformers RT-DETR; no weights
   are vendored or downloaded).
2. Acquire a permissive RT-DETR checkpoint locally (operator-driven; see
   `docs/adr/ADR-001.md` for the licence posture).
3. Construct `AppConfig(..., inference=InferenceConfig(...))` in a small launcher and
   pass it to `create_app` (the RT-DETR backend is built lazily, per job). The
   inference backend is intentionally **not** wired from environment variables in
   v1.0 — it is a code-level composition decision (checkpoint provenance is a
   per-artifact review, not a string).

Everything else (upload, the processing lifecycle, cancellation, evidence review,
export) is fully exercisable without a real backend using the deterministic stub
provider in the test suite.

---

## 10. Demo workflow

1. Start the API (with a scene) and the SPA (§4), or the single-process build (§6B).
2. Open the app → **Videos**. The first-run stage explains the flow
   (upload → detect → review).
3. Drop a fixed-camera clip. Upload progress is real and cancellable.
4. Watch the live lifecycle — `queued → initializing → running → finalizing →
   completed` (or `failed` / `cancelled`) — with progress, throughput, ETA, and an
   activity log.
5. Review: confirmed violations appear as timeline markers and in a filterable,
   severity-ranked list. Select one to inspect its measurements-vs-thresholds,
   evidence manifest, and open the **evidence viewer** (zoom / pan / fullscreen /
   frame navigation over the local video).
6. Add analyst notes, copy ids, and **export** selected events (JSON / CSV) or a
   single event's evidence manifest.
7. A browser refresh mid-job reconnects and restores selection + playback position.

Without a real RT-DETR backend, steps 1–4 and the error handling are fully
demonstrable; confirmed events in step 5 require the real backend (§9) or the
test-suite stub.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `mypy` / `pytest` can't import `fastapi` | Install the `api` extra: `pip install -e ".[dev,api]"`. |
| Processing returns `503 engine_unavailable` | No scene configured, or no real inference backend — set `TRAFFICPULSE_APP_SCENE` and/or configure RT-DETR (§9). The UI shows this as a recoverable error. |
| Upload rejected `400 unsupported_media` | Extension not allowed or the file isn't a readable video; accepted containers are `.mp4/.avi/.mkv/.mov/.webm/.m4v`. |
| Upload `409 duplicate_video` | The exact bytes are already stored; the UI recognizes this and opens the existing video. |
| Upload `413 payload_too_large` | Exceeds `TRAFFICPULSE_APP_MAX_UPLOAD_BYTES` (default 512 MiB). |
| SPA loads but API calls fail in the browser | Origin mismatch — use the dev proxy, serve same-origin, or set `TRAFFICPULSE_APP_CORS_ORIGINS` + `VITE_API_BASE_URL`. |
| Deep-link refresh (e.g. `/videos`) 404s | Configure the static host to fall back to `index.html` (Topology A), or use the single-process static mount (Topology B), which does this for you. |
| "Playback isn't available for this session" in the workspace | The uploaded file's object URL is session-only; re-select the local file to preview (the backend serves no media by design). |

---

## 12. Known limitations

- **No validated real-world accuracy.** Validation is on synthetic trajectories and
  generated clips; two reasoning slices ship (wrong-way, illegal-stopping).
- **Real inference is code-configured, not env-configured** (§9).
- **In-memory job/video registries.** Jobs and the upload index live in process
  memory; a restart forgets in-flight jobs (persisted confirmed events on disk
  survive). No multi-worker shared state — run a single API worker, or add a shared
  store before scaling out.
- **No authentication/authorization** on the API in v1.0 (the client is auth-ready:
  a bearer token provider can be registered).
- **Analyst notes are local only** (browser `localStorage`); they are never sent to
  the backend.
- **The legacy `viewer/` + `launch.py`** desktop demo is a separate, older
  exploration and is not part of the H7 web-application deployment described here.
