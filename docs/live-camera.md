# Live camera monitoring

TrafficPulse's second input mode: a browser camera streamed into a **persistent
backend session** that runs the same pipeline an uploaded video runs.

The whole design rests on one decision — *live mode adds no semantics*. There is no
live detector, no live tracker, no live rider heuristic, no live violation rule. A
live session holds one `InferenceEngine`, built by the same `EngineProvider` that
builds one for a processing job, and drives it through the `submit` / `drain` /
`finalize` surface that engine has documented for live producers since H6. Anything
this document says about detection thresholds, tracking, association, helmet
classification or violation confirmation is therefore a statement about the
existing system, not about a live variant of it.

---

## 1. How to run it

```bash
# 1. Backend, with the real RT-DETR + helmet backends (the documented entrypoint)
uvicorn serve:app --host 127.0.0.1 --port 8000

# 2. Frontend (development)
cd frontend && npm run dev        # http://localhost:5173, /api proxied to :8000
```

Then, in the browser:

1. open **Live camera** in the left navigation;
2. press **Start camera** and grant camera access when the browser asks;
3. press **Start monitoring**;
4. point the camera at traffic;
5. press **Stop monitoring**, then **Stop camera**.

A production deployment serving the built SPA from the API (`TRAFFICPULSE_APP_STATIC_DIR`)
needs no extra configuration: the WebSocket is same-origin on the same `/api` prefix.

**Browsers require a secure context for camera access.** `http://localhost` counts as
one; `http://<lan-ip>` does not, so a phone pointed at another machine's dev server
will be refused camera permission by the browser itself. Serve over HTTPS to demo
from another device.

### Pre-flight

`GET /api/live/status` answers whether live monitoring can start *before* the page
asks anyone for camera access, and the UI shows its `detail` sentence when it
cannot. It is false when no inference backend is configured, when no drawing
backend (Pillow, the `overlay` extra) is installed, or when every session slot is
taken.

---

## 2. Architecture

```
browser                          server (one process)
────────────────────────────────────────────────────────────────────────────────
<video> preview  ─── smooth, camera-rate, never analysed
     │
  canvas capture (only when a send slot is free)
     │  base64 JPEG
     ▼
  WebSocket  ────────────────►  /api/live/ws
                                     │
                                receiver task ──► LiveSession: ONE pending frame
                                                        │  (a newer frame replaces
                                                        │   the waiting one)
                                worker task  ◄──────────┘
                                     │  asyncio.to_thread
                                     ▼
                            InferenceEngine  (from EngineProvider — the same one
                                     │        a processing job gets)
                            submit → drain
                                     │
              RT-DETR ─► IoU tracker ─► rider association ─► head crop
                                     ─► helmet classifier ─► temporal state
                                     │
                            finalize (cadenced) ─► shipped violation reasoners
                                     │
                     ConfirmedEvent ─┴─► overlay providers ─► Pillow renderer
                                                                    │
     annotated JPEG + tracks + riders + events + measured stats ◄────┘
```

Code layout (`src/trafficpulse/app/live/`):

| module | responsibility |
|---|---|
| `config.py` | pacing and bounding knobs. No value here changes what anything concludes. |
| `errors.py` | typed live failures, each with a stable slug the socket reports. |
| `imaging.py` | JPEG ⇄ RGB array, with the byte and pixel guards on untrusted input. |
| `scene.py` | which scene a camera is reasoned about, and what that scene can support. |
| `session.py` | the persistent session: state, back-pressure, window, annotation. |
| `manager.py` | creation, isolation, the session cap, disposal, readiness. |
| `protocol.py` | every WebSocket message, typed. |
| `../routers/live.py` | the two HTTP endpoints and the socket handler. |

It is a **subpackage of `app`**, not a new top-level package, because it is an
application-layer composition exactly like `ProcessingService`: it owns no
reasoning, no perception and no storage.

---

## 3. Session lifecycle

**The session is the connection.** Opening the socket opens the session; closing it
closes the session. There is no session id to keep in sync between two channels and
no way to leave an engine running by navigating away.

```
connect ──► start {width,height,scene_hash?} ──► session {...}
                                                   │
                          ┌────────────────────────┴───────────────────────┐
                          │  frame {sequence, capture_seconds, data}       │
                          │       ──► result {...}  (+ events {...})       │
                          │       ──► warning {...} (one bad frame)        │
                          └────────────────────────┬───────────────────────┘
                                                   │
   stop ──► stopped {stats} ──► close 1000         │
   client disconnect ─────────────────────────────►│  session closed, engine reset
   inference failure ──► error {...} ──► close 4400┘
```

Every exit path runs the same cleanup: the session is closed and removed from the
manager, its engine reset, its pending frame dropped.

### What the session keeps between frames

Session identity, the tracker's tracks, every rider association and helmet reading
derived so far, the reasoners' accumulated history, the frame counter that gives
frames their identity, the media-time cursor, the events already announced, and the
measured counters. This is what makes a motorcycle crossing the view *one*
motorcycle with one track id, and a violation something sustained over seconds
rather than asserted from a single picture.

### Isolation

Each session gets its **own** engine from the provider, and an engine owns its own
tracker, observers and history. Two live cameras cannot share a track id or
contaminate each other — not by convention, but because there is no shared object
between them. `max_sessions` defaults to **2**; a third connection is refused with
`live_capacity_error` and close code 4400, because inference is not parallel and a
third session would only make the first two slower.

---

## 4. Frame transport

One WebSocket carries control and frames, as JSON text. Frames are base64 JPEG
inside a `frame` message. That costs ~33% over a binary frame and buys one framing
convention instead of two; at a handful of frames per second and tens of kilobytes
each, the overhead is irrelevant and the simplicity is not.

Guards on the frame payload: `max_frame_bytes` (2 MiB, bounding the *compressed*
payload) and `max_frame_pixels` (1920×1080, bounding what that payload may decode
to — a small compressed image can expand to an enormous one).

**Media time is the producer's.** The client sends `capture_seconds` from its own
monotonic clock; the server stamps no timestamp of its own, the same honesty rule
ingestion applies to a file's PTS. If a client's clock steps backwards (a suspended
tab, a re-attached device), the frame is **dropped with a warning and counted**, not
restamped: the tracker requires strictly increasing media time, and a fabricated
interval would corrupt tracking and every duration threshold built on it.

The session's frame size is fixed when it opens, because the scene's geometry is
measured in that frame. A frame of a different size is refused per-frame with a
warning; the session survives and keeps processing correctly-sized frames.

---

## 5. Back-pressure

Two limits, one on each side, and neither of them is a queue:

* **Client**: at most **2** frames unacknowledged. It captures and encodes a frame
  only when a slot is free, so no CPU is spent encoding frames that would be
  discarded, and the achieved capture rate settles at whatever inference absorbs.
* **Server**: exactly **one** pending frame. A frame arriving while one already
  waits **replaces** it, and the displaced frame is counted as dropped.

So latency cannot accumulate, memory cannot grow with a backlog, and what reaches
the detector is always the most recent view of the road. The dropped count is
published; it is the design working, not a fault.

### The analysis window

The engine accumulates per-track history for the whole stream and reasons over all
of it on every `finalize` — right for a clip of known length, unbounded for a camera
that runs all day. After `window_frames` (default **600**) processed frames the
session runs a final reasoning pass and then **resets the engine**, which bounds
both memory and per-finalize cost.

**This costs something real, and the UI says so:** track ids restart at the
boundary, and a violation whose support straddles it is not confirmed. At the
measured ~1 fps below, 600 frames is roughly ten minutes.

---

## 6. Measured performance

Measured on this development machine — **CPU only, no CUDA** — with
`demo/live_camera_probe.py`, which drives the real server's socket exactly as the
browser does (one `start`, then frames with at most two in flight). Real RT-DETR
(`PekingU/rtdetr_r50vd`, score threshold 0.50) and the real CLIP zero-shot helmet
classifier; source frames 898×506 from a real Delhi traffic clip; two rules running
(`no_helmet`, `triple_riding`).

| | 60 s run | 300 s run |
|---|---|---|
| Frames offered by the producer | 12.6 fps | 11.4 fps |
| Frames sent | 1.08 fps | 1.06 fps |
| **AI inference throughput** | **1.05 fps** | **1.05 fps** |
| Per frame, inside the pipeline | — | **1047 ms** |
| End-to-end delay (mean) | 1647 ms | 1599 ms |
| End-to-end delay (p95 / max) | 1884 / 2126 ms | 1882 / 1981 ms |
| Frames processed | 63 | 315 |
| Frames dropped by the producer | 692 | 3112 |
| Frames dropped by the server | 0 | 0 |
| Violations confirmed | 6 | 28 |
| Server RSS growth | +0.0 MB | **+0.1 MB** |

**So: live camera monitoring with ~1 frame per second of AI inference on this
machine's CPU, and about 1.6 seconds between the road and the screen.** It is not
real-time and the product does not claim it is. The camera preview stays smooth at
the camera's own rate because it is never analysed frame-for-frame.

Two numbers that must not be confused, and that the UI keeps apart:

* **throughput** (`inference_fps`) — frames per second the server completes,
  measured as a rate over processed frames;
* **delay** (`latency_ms_last`) — one frame from arrival to annotated result,
  *including* the wait behind a frame already being processed.

With two frames in flight the delay is about **two** processing times, so
`1 / delay` would report roughly half the true throughput. Publishing that as "the
frame rate" would understate the pipeline while claiming to measure it, which is why
`inference_fps` is a throughput measurement and `processing_ms_mean` is published
beside it.

A CUDA deployment will be substantially faster; nothing in this document is a target
and nothing in the UI is a nominal figure. Re-measure on your own hardware:

```bash
python demo/live_camera_probe.py --list-cameras
python demo/live_camera_probe.py --camera "<device name>" --seconds 60 --server-pid <pid>
# or, with no camera on the machine, replay a clip as a producer:
python demo/live_camera_probe.py --video runs/demo-clips/delhi_short.mp4 --seconds 60
```

---

## 7. What live mode evaluates, and what it does not

Live mode runs **exactly** the rules the resolved scene and the deployment can
support — decided by the same `capabilities.probe_scene` / `rules_for` the upload
path uses. It can never run a rule file mode would refuse, or refuse one file mode
would run.

A camera nobody has calibrated gets a **provisional scene**: its measured frame
size, one full-frame lane, and nothing else. Nothing is derived from a camera view.
Automatic calibration estimates a clip's dominant flow from a bounded prefix of a
*recorded* file; at the moment live monitoring starts there is no traffic history to
measure, and a direction guessed from the first seconds of an arbitrary view would
be a guess presented as a measurement.

| violation | uncalibrated camera | calibrated camera |
|---|---|---|
| Triple riding / overloading | ✅ runs | ✅ runs |
| No helmet | ✅ runs when the deployment's classifier passes the turban capability guard | same |
| Wrong way | ❌ no legal travel direction declared | ✅ runs |
| Illegal stopping | ❌ no no-stopping zone declared | ✅ runs |
| Red-light jumping | ❌ needs a stop line, its junction, **and** the period's signal timing | ❌ — a live session is given no schedule |

The session's opening message carries the unavailable list **with the server's
reason for each**, and the UI prints them beside the event feed. This is deliberate:
an empty violation list means either "nothing happened" or "that violation is not
being evaluated here", and a viewer cannot tell them apart from the list alone.

### Reasoning through a calibrated camera

For a fixed camera an analyst has calibrated, pass the scene revision's hash in the
opening message:

```json
{"type": "start", "width": 1280, "height": 720, "scene_hash": "<sha256>"}
```

The revision must have been authored against that frame size; a mismatch is refused
rather than applied, because a polygon measured on another frame lands somewhere
else in this one and every rule scoped to it would then silently confirm nothing.
**There is no UI for this yet** — the protocol and the backend support it, and it is
covered by tests, but the Live camera page always opens an uncalibrated session.

### Helmet analysis is not helmet enforcement

The page shows the same capability strip the video workspace shows. Nothing about
helmet reasoning changes in live mode: the classifier's binary vocabulary is
unchanged, `turban` is never mapped to `no_helmet`, and the capability guard still
refuses to build a `no_helmet` rule on a turban-blind backend. Where the rule *does*
build, its confirmations remain `EXPERIMENTAL` for the reasons
`docs/helmet-runtime-evaluation.md` §6 sets out, and the strip says so on the same
screen as the output.

One consequence is visible in the measured runs above: 28 helmet events in five
minutes, many for the same rider. That is the existing rule's episode behaviour on
per-frame classifier output that flips between frames — the very instability that
makes helmet enforcement experimental — not a live-mode artefact. The same footage
processed as a file behaves the same way.

### Driver attribution

Unchanged, and unchanged deliberately. A motorcycle carrying more than one rider
reports `driver_resolved: false`, and the UI prints **"N riders — DRIVER
UNRESOLVED"**. The tracker supplies no velocity, so which end of the motorcycle is
the front is unknown; live mode invents no heuristic to fill that in.

---

## 8. The WebSocket protocol

`ws://<host>/api/live/ws`. All messages are JSON text.

**Client → server**

| type | fields |
|---|---|
| `start` | `width`, `height`, `scene_hash?` — first message, exactly once |
| `frame` | `sequence`, `capture_seconds`, `data` (base64 JPEG) |
| `stop` | — |

**Server → client**

| type | fields |
|---|---|
| `session` | `session_id`, `camera_id`, `width`, `height`, `scene_hash`, `scene_calibrated`, `running_violations`, `unavailable_violations[{violation_type, reason}]`, `window_frames` |
| `result` | `frame_index`, `sequence`, `capture_seconds`, `tracks[]`, `motorcycles[]`, `riders[]`, `annotated` (base64 JPEG or null), `window_rolled_over`, `stats` |
| `events` | `events[]` — the verbatim `ConfirmedEvent` contract, not a reduced live shape |
| `warning` | `code`, `message` — one frame was refused; the session continues |
| `error` | `code`, `message` — the session is over; the socket closes with 4400 |
| `stopped` | `session_id`, `stats` |

An unknown message type from a newer server is ignored by the client rather than
treated as a failure.

**HTTP**

* `GET /api/live/status` — readiness pre-flight.
* `GET /api/live/sessions` — the sessions this process is running (operator view).

---

## 9. Error handling

| failure | behaviour |
|---|---|
| Camera permission denied | Typed message naming the recovery ("allow camera access for this site…"). No session is opened. |
| No camera / camera busy | Their own messages; the page stays usable and the camera can be retried. |
| No camera API (insecure origin, old browser) | Reported before any request. |
| Backend has no inference or drawing backend | `GET /api/live/status` says so before camera access is requested; **Start monitoring** stays disabled. |
| Session cap reached | `live_capacity_error`, close 4400. |
| Unknown / mismatched `scene_hash` | `scene_not_found` / `live_protocol_error`, close 4400. |
| Undecodable, oversized or wrong-size frame | `warning`; that frame is dropped, the session continues. |
| Out-of-order capture clock | `warning`; that frame is dropped, never restamped. |
| Inference raises | `live_inference_error`, session closed. Terminal on purpose: after a failed frame the tracker's state rests on a gap it does not know about, so continuing would make track identity mean less than it claims. |
| WebSocket dropped / server restarted | Client reports "the connection to the analysis server was lost" and **leaves the camera running**; monitoring can be restarted without re-granting permission. |
| Component unmount / navigation | Capture loop stopped, socket closed, every `MediaStream` track stopped — the browser's camera indicator goes out. |

Nothing fails silently, and no traceback ever reaches a client.

---

## 10. Privacy

* **No camera frame is written to disk.** Frames live in memory for the length of
  one inference and are then released.
* **No live event is persisted.** A live event has no source file to re-render
  evidence frames from, so persisting one would put a record in the write-once event
  store whose evidence manifest could never be resolved. Live events are delivered
  to the connected client and are gone when the session ends.
* The repository (`/api/videos`, `/api/events`, `/api/evidence`) continues to hold
  exactly what was processed from stored video, and is unaffected by live mode.
* Sessions are in-memory and do not survive a restart.

A backend test asserts the storage root is byte-for-byte empty after a live session
that confirmed real events.

---

## 11. Configuration

`LiveConfig` (`src/trafficpulse/app/live/config.py`), passed to `create_app` as
`live_config=`. Every field is a pacing or bounding decision; none changes what any
model or reasoner concludes.

| field | default | meaning |
|---|---|---|
| `max_sessions` | 2 | concurrent live sessions this process will run |
| `window_frames` | 600 | processed frames before the engine is reset |
| `finalize_interval_seconds` | 1.0 | minimum wall time between reasoning passes |
| `max_frame_bytes` | 2 MiB | largest accepted encoded frame |
| `max_frame_pixels` | 1920×1080 | largest accepted decoded frame |
| `jpeg_quality` | 70 | quality of the annotated frame sent back |
| `latency_samples` | 30 | recent frames the latency average covers |

---

## 12. Troubleshooting

**"Live monitoring is unavailable"** — read the sentence under it. No inference
backend means `AppConfig.inference` is unset: run `serve:app`, not
`trafficpulse.app.asgi:app`. No drawing backend means Pillow is missing:
`pip install 'trafficpulse[overlay]'`.

**The camera never appears** — check the browser's camera permission for the site,
and that the page is on `https` or `localhost`. A camera in use by another
application reports "another application is probably using it".

**Monitoring connects and then immediately errors** — read the `error` message; it
carries the server's own reason. `live_inference_error` usually means the checkpoint
could not be loaded; the same problem makes `POST /api/process` return 503.

**The annotated frame never appears** — the run may have published nothing to draw
(no motorcycle with associated riders yet). The camera preview keeps running
underneath; the "Sent / processed" counter shows whether frames are being analysed
at all.

**Everything works but no violation appears** — check the "not evaluated" list on
the page. On an uncalibrated camera only triple riding and (where configured)
helmet reasoning run at all.

**The frame rate is much lower than expected** — it is a CPU/GPU question, not a
code one. Measure it with `demo/live_camera_probe.py` and compare against §6.

**The WebSocket 404s or 500s behind a proxy** — the proxy must forward the upgrade.
The Vite dev proxy does (`ws: true` in `vite.config.ts`); nginx needs
`proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`.

---

## 13. Known limitations

1. **~1 fps AI inference on CPU** (§6). Not real-time; measured, not estimated.
2. **The analysis window resets state every `window_frames` frames.** Track ids
   restart and a violation straddling the boundary is not confirmed. This is the
   price of bounding memory for an all-day session, and it is stated in the UI.
3. **No scene-selection UI.** A calibrated camera can only be used by passing
   `scene_hash` on the socket; the page always opens an uncalibrated session, so in
   the browser today live mode evaluates triple riding and helmet reasoning only.
4. **Red-light jumping can never run live**, even on a calibrated camera: its rule
   config carries the period's signal schedule and no default can supply one.
5. **Live events are not persisted** and do not appear in the repository, analytics
   or the review workflow. They exist for the duration of the session.
6. **Helmet enforcement remains `EXPERIMENTAL`**, with the turban capability and
   per-frame instability blockers unchanged (§7).
7. **Driver attribution remains unresolved for multi-rider motorcycles** (§7).
8. **Real-camera browser validation is outstanding.** The full session pipeline has
   been validated end-to-end against the real models over a real WebSocket (§6), and
   the browser capture path is covered by tests against recording doubles — but no
   physical camera exists on the development machine, so the `getUserMedia` →
   canvas → socket path has never been exercised against real hardware. See the
   entry in `README.md`.
9. **One process, no horizontal scaling.** Sessions are in-memory and capped at 2.
