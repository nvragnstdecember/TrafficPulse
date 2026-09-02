"""One persistent live camera session: the state a camera stream needs to exist.

What makes this a *session* rather than a loop
-----------------------------------------------
A camera is not a sequence of independent images, and the difference is entirely
state. Between two frames this object keeps the tracker's tracks, every rider
association and helmet reading derived so far, the reasoners' accumulated history,
the frame counter that gives frames their identity, and the events already sent --
so a motorcycle crossing the view is *one* motorcycle with one id, and a violation
can be supported over seconds rather than asserted from a single picture. Treating
each frame as its own run is the failure mode this class exists to prevent, and it
is why nothing here is rebuilt per frame.

Reuse, not a parallel pipeline
------------------------------
There is no live detector, live tracker, live associator, live classifier or live
rule anywhere in this module. The session holds one
:class:`~trafficpulse.engine.InferenceEngine` -- built by the same
:class:`~trafficpulse.app.engine_provider.EngineProvider` that builds one for an
uploaded video -- and drives it through the ``submit`` / ``drain`` / ``finalize``
surface the engine already documents for live producers. Detection thresholds,
tracking, rider association, head crops, helmet classification, temporal state and
every violation reasoner are therefore *identical* to file mode by construction:
this file could not diverge from them without going through the provider, and it
does not.

Its own three jobs are the ones a file does not have:

**Frame identity.** A file's frames arrive with PTS; a camera's do not. Each
accepted frame is stamped by :func:`~trafficpulse.engine.frame_record_from_array`
-- the identity scheme ingestion uses, so live frames are indistinguishable
downstream -- with a strictly ascending index and the producer's own capture time.

**Back-pressure.** Inference is far slower than capture (see ``docs/live-camera.md``
for measured numbers), so the session holds exactly **one** pending frame. A frame
arriving while another is already waiting replaces it and the displaced one is
counted as dropped. That is the whole queue: latency cannot accumulate, memory
cannot grow with the backlog, and what reaches the detector is always the most
recent view of the road.

**A bounded window.** The engine accumulates history for the whole stream and
reasons over all of it on each ``finalize``. After
:attr:`~trafficpulse.app.live.config.LiveConfig.window_frames` processed frames the
session resets the engine, which bounds both memory and reasoning cost. The cost is
real and is reported rather than hidden: track ids restart at the boundary, and a
violation whose support straddles it is not confirmed.

Nothing is persisted
--------------------
No camera frame is written to disk, and neither is a live event. A live event has
no source file to re-render evidence frames from, so persisting one would put a
record in the write-once event store whose evidence manifest could never be
resolved. Live events are delivered to the connected client and are gone when the
session ends; the repository continues to hold exactly what was processed from a
stored video.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ...contracts import ConfirmedEvent, SceneConfig, TrackState
from ...engine import InferenceEngine, frame_record_from_array
from ...overlay import OverlayCompositor, OverlayFrameRef, PillowOverlayRenderer
from ...overlay.registry import OverlayProvider
from ...pipeline.helmet_observer import HelmetOverlayFrame
from ...pipeline.triple_riding import TripleRidingOverlayFrame
from ..overlay_video import OVERLAY_REGISTRY, overlay_sources
from .config import LiveConfig
from .errors import LiveFrameError, LiveInferenceError
from .imaging import decode_frame, encode_jpeg
from .scene import LiveRulePlan

#: Cap on the remembered event ids, so an all-day session's de-duplication set
#: cannot grow without bound either. Far above any realistic live event count;
#: exceeding it forgets the oldest half, which at worst re-announces an old event.
_MAX_REMEMBERED_EVENTS = 4096


@dataclass(frozen=True)
class PendingFrame:
    """One encoded camera frame waiting for inference."""

    data: bytes

    capture_seconds: float
    """The producer's own capture time, relative to the start of capture. Supplied
    by the client and never fabricated here -- the same media-time honesty rule
    ingestion applies to PTS."""

    client_sequence: int

    received_at: float
    """Monotonic arrival instant, used to measure queue + inference latency."""


@dataclass(frozen=True)
class LiveTrackView:
    """One tracked object on the frame just processed, as the client displays it."""

    track_id: str
    object_class: str
    status: str
    bbox: tuple[float, float, float, float]
    confidence: float | None


@dataclass(frozen=True)
class LiveRiderView:
    """One associated rider on the frame just processed.

    ``driver_resolved`` is ``False`` whenever the rider shares its motorcycle, and
    that is a report of the shipped behaviour rather than a live-mode policy: the
    tracker supplies no velocity, so which end of the motorcycle is the front is
    unknown, and the observation layer returns an ``UNKNOWN`` rider slot instead of
    guessing. No heuristic is applied here to fill it in.
    """

    rider_track_id: str
    motorcycle_track_id: str
    rider_count: int
    driver_resolved: bool
    helmet_label: str | None
    helmet_confidence: float | None
    helmet_gated: bool


@dataclass(frozen=True)
class LiveMotorcycleView:
    """One motorcycle's occupancy on the frame just processed.

    Published beside :class:`LiveRiderView` because occupancy and helmet state come
    from different observers and are available in different deployments: a run
    without a classifier still knows how many riders the associator attached to a
    motorcycle, and that -- not the helmet reading -- is what decides whether a
    driver can be attributed at all.
    """

    motorcycle_track_id: str
    rider_count: int
    driver_resolved: bool


@dataclass(frozen=True)
class LiveStats:
    """What the session has actually measured. Nothing here is estimated."""

    frames_received: int
    frames_dropped: int
    frames_processed: int
    frames_rejected: int
    frames_out_of_order: int
    active_tracks: int
    events_emitted: int
    windows_completed: int
    window_frames_processed: int
    uptime_seconds: float

    inference_fps: float | None
    """**Throughput**: processed frames per wall second, over the span from the first
    processed frame to the most recent.

    Measured as a rate rather than as ``1 / latency`` on purpose. Two frames may be
    in flight at once, so a frame's end-to-end latency includes the time it waited
    behind the previous one -- inverting it would report roughly *half* the frames
    per second the system genuinely completes. ``None`` until two frames have been
    processed, because one frame measures no rate at all."""

    processing_ms_mean: float | None
    """How long one frame spends **in** the pipeline, averaged over recent frames:
    decode, detect, track, associate, classify, reason, draw. This is the cost of a
    frame; the reciprocal is the rate above."""

    latency_ms_mean: float | None
    latency_ms_last: float | None
    """End-to-end: a frame's arrival at the server to its annotated result -- so it
    includes any wait behind a frame already being processed. Larger than
    ``processing_ms_mean`` by exactly that wait, which is what a viewer perceives as
    the delay between the road and the screen."""


@dataclass(frozen=True)
class LiveFrameResult:
    """The outcome of processing exactly one live frame."""

    frame_index: int
    client_sequence: int
    capture_seconds: float
    tracks: tuple[LiveTrackView, ...]
    motorcycles: tuple[LiveMotorcycleView, ...]
    riders: tuple[LiveRiderView, ...]
    annotated_jpeg: bytes | None
    new_events: tuple[ConfirmedEvent, ...]
    window_rolled_over: bool
    stats: LiveStats


@dataclass
class _Counters:
    """Mutable session tallies, all guarded by the session lock."""

    received: int = 0
    dropped: int = 0
    processed: int = 0
    rejected: int = 0
    out_of_order: int = 0
    windows: int = 0
    window_processed: int = 0
    events_emitted: int = 0
    active_tracks: int = 0
    latencies: deque[float] = field(default_factory=deque)
    processing: deque[float] = field(default_factory=deque)
    first_processed_at: float | None = None
    last_processed_at: float | None = None


class LiveSession:
    """A persistent camera-monitoring session over one injected engine.

    Thread-affinity, deliberately explicit: :meth:`offer`, :meth:`stats` and
    :meth:`close` are safe to call from the socket's event loop, while
    :meth:`process` is blocking work that belongs on a worker thread and is
    serialised by its own lock -- so the engine, which is not thread-safe, is only
    ever touched by one thread at a time.
    """

    def __init__(
        self,
        *,
        session_id: str,
        engine: InferenceEngine,
        scene: SceneConfig,
        plan: LiveRulePlan,
        config: LiveConfig,
        width: int,
        height: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        self.scene = scene
        self.plan = plan
        self.width = width
        self.height = height
        self.camera_id = scene.scene.camera_id
        self._engine = engine
        self._config = config
        self._clock = clock
        self._started_at = clock()

        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._pending: PendingFrame | None = None
        self._closed = False
        self._counters = _Counters(
            latencies=deque(maxlen=config.latency_samples),
            processing=deque(maxlen=config.latency_samples),
        )

        # Frame identity + media time. Both continue across a window rollover: the
        # reset clears what the engine *reasons* over, while re-using an index or
        # stepping back in time would make two different frames indistinguishable.
        self._frame_index = 0
        self._last_capture_seconds: float | None = None

        self._emitted_event_ids: set[str] = set()
        self._window_events: list[ConfirmedEvent] = []
        self._last_finalize_at = clock()
        self._renderer: PillowOverlayRenderer | None = None

    # --- read-only surface ---------------------------------------------------
    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    # --- producer side (safe from the event loop) ----------------------------
    def offer(self, frame: PendingFrame) -> bool:
        """Hold ``frame`` as the one pending frame; report whether one was dropped.

        The single-slot queue described in the module docstring. Returning the drop
        rather than hiding it is what lets the client show a truthful "the camera is
        producing faster than inference consumes" reading instead of a fabricated
        frame rate.
        """

        with self._lock:
            if self._closed:
                return False
            self._counters.received += 1
            displaced = self._pending is not None
            if displaced:
                self._counters.dropped += 1
            self._pending = frame
            return displaced

    def take(self) -> PendingFrame | None:
        """Claim the pending frame, if there is one."""

        with self._lock:
            if self._closed:
                return None
            pending, self._pending = self._pending, None
            return pending

    def stats(self) -> LiveStats:
        """A snapshot of what has actually been measured (cheap; lock-guarded)."""

        with self._lock:
            counters = self._counters
            latencies = list(counters.latencies)
            processing = list(counters.processing)
            mean = sum(latencies) / len(latencies) if latencies else None
            processing_mean = sum(processing) / len(processing) if processing else None
            return LiveStats(
                frames_received=counters.received,
                frames_dropped=counters.dropped,
                frames_processed=counters.processed,
                frames_rejected=counters.rejected,
                frames_out_of_order=counters.out_of_order,
                active_tracks=counters.active_tracks,
                events_emitted=counters.events_emitted,
                windows_completed=counters.windows,
                window_frames_processed=max(counters.window_processed, 0),
                uptime_seconds=self._clock() - self._started_at,
                inference_fps=self._throughput(counters),
                processing_ms_mean=(
                    processing_mean * 1000.0 if processing_mean is not None else None
                ),
                latency_ms_mean=(mean * 1000.0) if mean is not None else None,
                latency_ms_last=(latencies[-1] * 1000.0) if latencies else None,
            )

    @staticmethod
    def _throughput(counters: _Counters) -> float | None:
        """Frames actually completed per wall second, or ``None`` if unmeasurable.

        The span runs from the *first* processed frame, not from the session's start:
        a session spends its first moments waiting for a camera and a first frame,
        and charging that idle time against the pipeline would understate what the
        hardware does.
        """

        first, last = counters.first_processed_at, counters.last_processed_at
        if first is None or last is None or counters.processed < 2:
            return None
        span = last - first
        if span <= 0:
            return None
        # ``processed - 1`` intervals between ``processed`` frames.
        return (counters.processed - 1) / span

    def close(self) -> None:
        """Release the session's state. Idempotent."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
        # Outside the session lock: reset() touches the engine, and a worker may
        # still be finishing a frame -- the process lock is what orders the two.
        with self._process_lock:
            self._engine.reset()
            self._window_events.clear()
            self._emitted_event_ids.clear()

    # --- consumer side (worker thread) ---------------------------------------
    def process(self, frame: PendingFrame) -> LiveFrameResult:
        """Run one frame through the whole pipeline; return what to display.

        Raises:
            LiveFrameError: the payload is not a usable frame for this session --
                undecodable, oversized, or a different resolution than the scene
                this session was built for. The session survives: the caller warns
                the client and waits for the next frame.
            LiveInferenceError: the engine raised. Terminal for the session (see
                that error's docstring for why continuing would be dishonest).
        """

        with self._process_lock:
            return self._process_locked(frame)

    def _process_locked(self, frame: PendingFrame) -> LiveFrameResult:
        processing_started = self._clock()
        image = decode_frame(
            frame.data,
            max_bytes=self._config.max_frame_bytes,
            max_pixels=self._config.max_frame_pixels,
        )
        height, width = int(image.shape[0]), int(image.shape[1])
        if (width, height) != (self.width, self.height):
            with self._lock:
                self._counters.rejected += 1
            raise LiveFrameError(
                f"camera frame is {width}x{height} but this session's scene is "
                f"{self.width}x{self.height}; stop and restart monitoring to rebuild "
                "the session for the new resolution"
            )

        capture_seconds = self._next_capture_seconds(frame.capture_seconds)
        # Read then advance, so the first live frame is index 0 -- the same 0-based
        # numbering ingestion gives a file's first frame.
        frame_index = self._frame_index
        self._frame_index += 1
        record = frame_record_from_array(
            image,
            source_id=self.session_id,
            frame_index=frame_index,
            timestamp_seconds=capture_seconds,
            camera_id=self.camera_id,
        )
        try:
            self._engine.submit(record)
            self._engine.drain()
        except Exception as exc:  # noqa: BLE001 - typed and terminated, never swallowed
            raise LiveInferenceError(f"inference failed on a live frame: {exc}") from exc

        tracks = self._tracks_for_frame(record.frame_index)
        motorcycles, riders = self._associations_for_frame(record.frame_index)
        new_events = self._reason(final=False)
        annotated = self._annotate(image, record.frame_index, capture_seconds)
        rolled_over = self._maybe_roll_window()

        completed = self._clock()
        with self._lock:
            self._counters.processed += 1
            self._counters.window_processed += 1
            self._counters.latencies.append(completed - frame.received_at)
            self._counters.processing.append(completed - processing_started)
            if self._counters.first_processed_at is None:
                self._counters.first_processed_at = completed
            self._counters.last_processed_at = completed
            self._counters.events_emitted += len(new_events)
            self._counters.active_tracks = len(tracks)
        return LiveFrameResult(
            frame_index=record.frame_index,
            client_sequence=frame.client_sequence,
            capture_seconds=capture_seconds,
            tracks=tracks,
            motorcycles=motorcycles,
            riders=riders,
            annotated_jpeg=annotated,
            new_events=new_events,
            window_rolled_over=rolled_over,
            stats=self.stats(),
        )

    def track_states(self) -> Sequence[TrackState]:
        """The window's accumulated states (read-only; for tests and diagnostics)."""

        return self._engine.track_states()

    # --- internals -----------------------------------------------------------
    def _next_capture_seconds(self, offered: float) -> float:
        """Validate this frame's media time, or refuse the frame.

        The tracker seam requires **strictly increasing** media time and enforces it
        itself. A browser's capture clock is monotonic in principle and occasionally
        is not in practice (a suspended tab, a re-attached device), so this refuses
        the offending frame rather than nudging its timestamp forward: a fabricated
        interval would make the tracker's motion, and every duration threshold built
        on it, quietly wrong. One dropped frame costs a fraction of a second of
        coverage; a fabricated one corrupts the reasoning.

        Raises:
            LiveFrameError: the offered time is not after the previous frame's. The
                worker warns the client and the session continues.
        """

        if self._last_capture_seconds is not None and offered <= self._last_capture_seconds:
            with self._lock:
                self._counters.out_of_order += 1
            raise LiveFrameError(
                f"camera frame timestamp {offered:.3f}s is not after the previous "
                f"frame's {self._last_capture_seconds:.3f}s; the frame is dropped "
                "rather than restamped, because a fabricated interval would corrupt "
                "tracking and every duration threshold built on it"
            )
        self._last_capture_seconds = offered
        return offered

    def _tracks_for_frame(self, frame_index: int) -> tuple[LiveTrackView, ...]:
        """This frame's tracked objects, read off the engine's accumulated states."""

        return tuple(
            LiveTrackView(
                track_id=state.track_id,
                object_class=str(state.object_class),
                status=str(state.status),
                bbox=(state.bbox.x1, state.bbox.y1, state.bbox.x2, state.bbox.y2),
                confidence=state.confidence,
            )
            for state in self._engine.track_states()
            if state.frame_index == frame_index
        )

    def _associations_for_frame(
        self, frame_index: int
    ) -> tuple[tuple[LiveMotorcycleView, ...], tuple[LiveRiderView, ...]]:
        """This frame's motorcycle occupancy and rider readings, as observed.

        Read entirely off the captures the rules' own observers produced while
        processing this frame -- no association is recomputed, no crop re-extracted
        and no classifier re-run. Which of the two lists is populated depends on
        which observers the deployment's rule set actually ran: occupancy comes from
        the rider-count observer, helmet readings from the helmet observer, and a
        deployment running only one of them gets only that half rather than a
        fabricated version of the other.

        ``rider_count`` is the number of riders the associator attached to that
        motorcycle **on this frame**, which is exactly the quantity the shipped
        overlay providers use to decide multi-rider.
        """

        helmet_readings: dict[str, tuple[str, float | None, bool]] = {}
        counts: dict[str, int] = {}
        pairs: list[tuple[str, str]] = []
        for source in overlay_sources(self._engine):
            frames = getattr(source, "overlay_frames", None)
            if frames is None:
                continue
            for captured in frames():
                if getattr(captured, "frame_index", None) != frame_index:
                    continue
                if isinstance(captured, HelmetOverlayFrame):
                    for rider in captured.riders:
                        helmet_readings[rider.rider_track_id] = (
                            rider.helmet_label,
                            rider.confidence,
                            rider.gated,
                        )
                        pairs.append((rider.rider_track_id, rider.motorcycle_track_id))
                elif isinstance(captured, TripleRidingOverlayFrame):
                    counts[captured.motorcycle_track_id] = captured.rider_count

        # A deployment with a classifier but no rider-count rule still gets truthful
        # occupancy: count the riders this frame's own associations attached.
        for _, motorcycle_id in pairs:
            counts.setdefault(motorcycle_id, sum(1 for _, m in pairs if m == motorcycle_id))

        motorcycles = tuple(
            LiveMotorcycleView(
                motorcycle_track_id=motorcycle_id,
                rider_count=rider_count,
                # The one place driver attribution is decided, and it is decided by
                # occupancy alone -- see LiveRiderView's docstring.
                driver_resolved=rider_count <= 1,
            )
            for motorcycle_id, rider_count in sorted(counts.items())
        )

        riders: list[LiveRiderView] = []
        seen: set[tuple[str, str]] = set()
        for rider_id, motorcycle_id in pairs:
            if (rider_id, motorcycle_id) in seen:
                continue
            seen.add((rider_id, motorcycle_id))
            label, confidence, gated = helmet_readings.get(rider_id, (None, None, False))
            rider_count = counts.get(motorcycle_id, 1)
            riders.append(
                LiveRiderView(
                    rider_track_id=rider_id,
                    motorcycle_track_id=motorcycle_id,
                    rider_count=rider_count,
                    driver_resolved=rider_count <= 1,
                    helmet_label=label,
                    helmet_confidence=confidence,
                    helmet_gated=gated,
                )
            )
        riders.sort(key=lambda view: (view.motorcycle_track_id, view.rider_track_id))
        return motorcycles, tuple(riders)

    def _reason(self, *, final: bool) -> tuple[ConfirmedEvent, ...]:
        """Finalize at the configured cadence; return only the newly confirmed events.

        ``finalize`` is the engine's own idempotent whole-history pass, so repeating
        it is safe and re-derives the same events; the de-duplication here is about
        not *re-announcing* one, never about deciding it. Nothing about an event --
        whether it confirms, when it triggered, what evidence it rests on -- is
        computed in this method.
        """

        now = self._clock()
        if not final and now - self._last_finalize_at < self._config.finalize_interval_seconds:
            return ()
        self._last_finalize_at = now
        events = self._engine.finalize()
        self._window_events = list(events)
        fresh = tuple(
            event for event in events if event.event_id not in self._emitted_event_ids
        )
        for event in fresh:
            self._emitted_event_ids.add(event.event_id)
        if len(self._emitted_event_ids) > _MAX_REMEMBERED_EVENTS:
            self._emitted_event_ids = set(
                sorted(self._emitted_event_ids)[_MAX_REMEMBERED_EVENTS // 2 :]
            )
        return fresh

    def _annotate(
        self, image: NDArray[np.uint8], frame_index: int, media_seconds: float
    ) -> bytes | None:
        """Draw this frame's inference through the existing overlay framework.

        The same registry dispatch the annotated-video renderer uses, over the same
        captures, drawn by the same renderer: live annotation is the shipped overlay
        pipeline applied to one in-memory frame instead of a re-decoded file. So a
        violation that gains an overlay provider gains a live overlay for free, and
        no drawing rule is stated twice.

        Returns ``None`` when there is nothing to draw, so the client keeps showing
        its own camera preview rather than a blank frame.
        """

        providers: list[OverlayProvider] = []
        for source in overlay_sources(self._engine):
            provider = OVERLAY_REGISTRY.create_for(source, self._window_events)
            if provider is not None and provider.has_content():
                providers.append(provider)
        if not providers:
            return None
        providers.sort(key=lambda provider: provider.violation_kind)
        scene = OverlayCompositor(providers).scene_for(
            OverlayFrameRef(
                camera_id=self.camera_id,
                frame_index=frame_index,
                media_seconds=media_seconds,
                width=self.width,
                height=self.height,
            )
        )
        if not scene.elements:
            return None
        if self._renderer is None:
            self._renderer = PillowOverlayRenderer()
        return encode_jpeg(
            self._renderer.render(image, scene), quality=self._config.jpeg_quality
        )

    def _maybe_roll_window(self) -> bool:
        """Reset the engine once the window is full; report whether it happened.

        The window's last reasoning pass runs *before* the reset, so an event the
        window can still support is confirmed rather than lost to the boundary. What
        the boundary does cost -- track identity, and any violation whose support
        straddles it -- is stated in the module docstring and surfaced to the client,
        which is the honest treatment of a bound that genuinely discards state.
        """

        with self._lock:
            full = self._counters.window_processed + 1 >= self._config.window_frames
        if not full:
            return False
        self._reason(final=True)
        self._engine.reset()
        self._window_events.clear()
        with self._lock:
            self._counters.windows += 1
            # The caller's ``+= 1`` for this frame lands on 0: the frame that filled
            # the window belongs to the window that just closed, not the new one.
            self._counters.window_processed = -1
        return True
