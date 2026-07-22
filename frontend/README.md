# TrafficPulse Frontend (H7B · H7C · H7D)

The TrafficPulse web client: the reusable application architecture (shell,
routing, theming, design system, typed API infrastructure, state, and testing)
from **H7B**, the **H7C video workspace** (upload → process → frame-accurate
review), and the **H7D live processing integration** — a resilient end-to-end
lifecycle over the FastAPI backend with live polling, cancellation, recovery, and
graceful failure handling. It talks to the H7A HTTP API over JSON only — it never
imports backend code, and no component calls `fetch` directly (pages → hooks →
services → typed client).

## Stack

React 18 · TypeScript (strict) · Vite · Tailwind CSS · shadcn-style components
(Radix UI) · React Router · TanStack Query · React Hook Form · Zod · Zustand ·
Lucide · Vitest + Testing Library.

## Scripts

| Command             | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `npm run dev`       | Dev server (proxies `/api` to the backend) |
| `npm run build`     | Type-check + production build (code-split) |
| `npm run typecheck` | `tsc -b` type-check only                   |
| `npm run lint`      | ESLint                                     |
| `npm run format`    | Prettier write                             |
| `npm run test`      | Vitest (jsdom, mocked API — no backend)    |
| `npm run coverage`  | Vitest with V8 coverage                    |

## Configuration

Runtime config is read once in `src/lib/env.ts`:

- `VITE_API_BASE_URL` — API origin (default: same-origin; dev proxies `/api`).
- `VITE_API_TIMEOUT_MS` — per-request timeout (default 30s).
- `VITE_API_PROXY_TARGET` — dev proxy target (default `http://127.0.0.1:8000`).
- `VITE_MAX_UPLOAD_BYTES` — upload size limit (default 512 MiB, mirroring H7A).
- `VITE_ACCEPTED_VIDEO_FORMATS` — comma-separated container extensions.

## Architecture

```
src/
  app/          Root <App/> (providers + router)
  api/          Typed client (timeout, cancellation, progress, errors) + types + query config
  assets/       Brand mark
  components/
    ui/         shadcn-style primitives (Button, Card, Dialog, Tabs, Toast, Form, …)
    common/     Composed pieces (PageHeader, EmptyState, StatusChip, ProgressBar, VirtualList, …)
    layout/     App shell (Sidebar, TopNav, StatusFooter, MobileNav)
    workspace/  H7C feature UI (dropzone, player, timeline, event list/detail, processing)
  hooks/        Query hooks + workspace controllers (processing, player, shortcuts, events)
  lib/          cn(), formatters, env, upload constraints, job lifecycle, workspace domain logic
  pages/        Route pages (lazy-loaded)
  providers/    Theme, Query, and the composed AppProviders
  routes/       Route tree + paths + route error boundary
  services/     Endpoint wrappers (the only layer that names endpoints)
  store/        Zustand stores (ui, settings, selection, upload, processing, notifications)
  styles/       Design tokens + global CSS
  test/         Vitest setup, render helpers, and wire-shaped fixtures
```

**Layering:** pages → hooks → services → API client. Design tokens live as CSS
variables (light/dark) mapped through Tailwind, so no component hardcodes a
value. The theme has no flash on load (an inline bootstrap in `index.html`
applies the persisted theme before first paint).

## The video workspace (H7C)

`/videos` is a single stage-based page: until a video exists it shows the upload
dropzone (so no event query is issued for a video that isn't there); once one
does, the workspace mounts.

**Upload → process → review**

1. **Upload** — drag-and-drop or browse, validated client-side against the
   configured formats and size limit (`lib/upload.ts`) before any request. The
   upload runs over XHR so progress is real, and it is cancellable.
2. **Process** — `useProcessing` starts the job and polls it through the Query
   layer, mapping the backend's four job statuses onto the workspace lifecycle
   (`idle → uploading → queued → running → completed | failed`), with progress,
   elapsed time, ETA, and an activity log. Cancel, retry, replace, and remove are
   all actions on that one controller; the video/job ids are persisted, so a page
   refresh reconnects to an in-flight job.
3. **Review** — the uploaded file plays locally from an object URL (the backend
   renders no media). Events appear as markers on a zoomable timeline, in a
   virtualized filterable list, and in a detail panel with the rule's
   measurements against its thresholds plus the evidence manifest (artifact
   references, rule trace, model provenance).

**Design notes**

- **Pure domain logic.** Media-time mapping, marker clustering, filtering,
  sorting, and clock formatting live in `lib/workspace.ts` as pure functions, so
  the interactive behaviour is unit-tested without React or the network.
- **One playback controller.** `useVideoController` owns all playback state
  outside the UI; `PlayerProvider` shares it with the player, its controls, the
  timeline, and the detail panel, so selection and seeking stay in sync.
- **Keyboard:** `Space`/`K` play-pause · `←`/`→` seek 5s · `,`/`.` step a frame ·
  `J`/`L` previous/next event · `F` fullscreen — suppressed while typing.
- **Event time.** The backend anchors PTS at the Unix epoch, so an event's
  `trigger_at` maps directly onto the uploaded video's own 0..duration timeline.

## Live processing integration (H7D)

H7D makes the workspace a fully-integrated live client of the backend.

**Processing lifecycle.** The backend's job statuses (`pending → running →
succeeded | failed | cancelled`) are surfaced as a richer UI lifecycle. Two
sub-phases of `running` are **derived from the job's own truthful frame
counters** (never fabricated) in `lib/job.ts`:

```
idle → uploading → queued → initializing → running → finalizing → completed
                                                                ↘ failed
                                                                ↘ cancelled
```

`initializing` = running with no frame processed yet; `finalizing` = every known
frame processed but not yet flipped to succeeded (evidence + persistence). Every
transition is reflected immediately.

**Polling.** All polling is TanStack Query — no custom loops. `useJob` polls
while the job is active and stops on any terminal state; the event list polls
while active so markers appear as the run confirms them; requests dedupe; transient
failures fall back to the client's exponential retry backoff; and polling resumes
automatically on reconnect.

**Live timeline & synchronization.** `mergeWorkspaceEvents` merges each poll into
the prior set, **preserving object references** for unchanged events (and the
whole array when nothing changed) — so appends update only what moved, existing
rows/markers don't rerender, and the current selection survives background
refreshes. Events, detail, and evidence all update without a manual refresh.

**Cancellation.** A cancellable job exposes a **Cancel** action → `POST
/api/process/{job_id}/cancel`. The backend flags the run, the engine stops
cooperatively at the next frame (`InferenceEngine.run(..., should_cancel=…)`), and
the job transitions to `cancelled` with nothing persisted. An in-flight _upload_
is instead aborted locally (`AbortController`).

**Recovery.** The persisted store (`processing-store`) keeps the job id, video id,
selected event, and playback position, so a browser refresh into an active job
restores the upload, processing state, timeline, selection, and playback position.

**Errors.** Backend-unavailable/timeout/network failures during polling raise a
reconnect banner with a retry; a failed job shows its error with a retry; delayed
or failed evidence/detail shows an inline retry; a duplicate upload (409) is
recognized and **opens the already-stored video** instead of dead-ending, using
the `video_id` the conflict carries.

**Backend touchpoints (additive, backwards-compatible).** A cancel endpoint +
`JobStatus.CANCELLED`; an optional `should_cancel` predicate on the engine run;
`video_id` on the `duplicate_video` error envelope (omitted otherwise); and a
`jobs_cancelled` metric. No existing endpoint contract changed.
