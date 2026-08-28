"""Application services -- the business logic between routers and H6 (H7A).

A small set of services holds everything the thin routers delegate to. They
**compose** the existing layers and add no reasoning, detection, tracking, or
persistence logic of their own:

* :class:`VideoService` -- validates and stores uploads (readability via P1-U5
  ingestion, addressed by content hash), and serves the stored file back.
* :class:`VideoLibraryService` -- joins the video, job, and review indices into
  browsable per-video metadata (H11), loading no event, evidence, or overlay.
* :class:`ProcessingService` -- creates jobs and drives the H6 engine
  (``engine.run`` + ``engine.persist``) through the injected executor + provider.
* :class:`EventService` / :class:`EvidenceService` -- read persisted events and
  manifests back from the H6 ``EventStore`` (the durable source of truth).
* :class:`MetricsService` -- aggregates job counts and surfaces the latest job's
  H6 ``EngineMetrics`` verbatim.

Every lower-layer failure is translated to a typed :class:`AppError`, so the HTTP
contract never leaks an internal exception or a traceback.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import islice
from pathlib import Path

from pydantic import ValidationError

from ..contracts import (
    ConfirmedEvent,
    EvidenceManifest,
    ReviewEntry,
    SceneConfig,
    TrackState,
    can_transition,
    next_status,
)
from ..contracts.enums import ArtifactKind, ReviewAction, ReviewStatus, ViolationType
from ..contracts.scene import ZoneType, scene_config_hash
from ..detector.errors import DetectorError
from ..engine import (
    EngineRunResult,
    FileFrameSource,
    InferenceEngine,
    RuleConfig,
    TripleRidingRuleConfig,
    WrongWayRuleConfig,
)
from ..engine.errors import (
    EngineConfigurationError,
    RunCancelledError,
    UnsupportedRuleError,
)
from ..evidence import (
    ArtifactStore,
    build_evidence_package,
    evidence_package_filename,
    merge_rendered_artifacts,
    render_run_evidence,
    rendered_artifact_for,
)
from ..ingestion.video import VideoIngestionError
from ..persistence import (
    CorruptRecordError,
    EventStore,
    RenderedArtifactStore,
    ReviewStore,
    RunNotFoundError,
    SceneStore,
    StoredEvent,
)
from ..pipeline.errors import SceneConfigurationError
from ..scenes import (
    CALIBRATION_SOURCE_AUTO,
    DirectionDraft,
    FlowEstimate,
    SceneDraft,
    ZoneDraft,
    build_scene,
    estimate_dominant_flow,
)
from .capabilities import rules_for, supported_violations
from .config import AppConfig
from .engine_provider import EngineProvider
from .errors import (
    AppError,
    ArtifactNotFoundError,
    DuplicateVideoError,
    EngineUnavailableError,
    EventNotFoundError,
    InvalidConfigurationError,
    InvalidTransitionError,
    JobNotFoundError,
    OverlayNotFoundError,
    PayloadTooLargeError,
    SceneNotFoundError,
    UnsupportedMediaError,
    VideoMediaNotFoundError,
    VideoNotFoundError,
)
from .models import (
    EventListResponse,
    EventSort,
    EventSummary,
    EvidenceRepairResponse,
    JobStatusResponse,
    MetricsResponse,
    ReviewDecisionRequest,
    ReviewResponse,
    SceneSummary,
    SceneValidationResponse,
    VideoListResponse,
    VideoSort,
    VideoSummary,
)
from .overlay_video import build_job_compositor, render_job_overlay
from .registry import (
    EvidenceStatus,
    JobExecutor,
    JobRecord,
    JobStatus,
    JobStore,
    OverlayStatus,
    VideoRecord,
    VideoStore,
)

_logger = logging.getLogger("trafficpulse.app")

#: Identifiers used by every auto-derived scene, so a reviewer can recognise one
#: at a glance and the UI can name the lane the direction governs.
DERIVED_LANE_ID = "lane-auto"
DERIVED_DIRECTION_ID = "dir-auto"

#: The fewest substantial movers a legal direction may be declared from. Below
#: this the vector is dominated by one or two vehicles, and a wrong-way rule built
#: on it would be reasoning about an accident of who happened to drive past.
#: Abstaining costs a rule; guessing costs every lawful vehicle going the other way.
MIN_FLOW_MOVERS = 5

#: Frames the calibration pass may submit. Bounded because it is pure overhead on
#: top of the real run: enough motion to estimate flow (several seconds at any
#: normal frame rate), never the whole clip.
CALIBRATION_FRAME_BUDGET = 90


@dataclass(frozen=True)
class DerivedScene:
    """A scene authored from one clip's observed motion, and what it rests on.

    ``flow`` is ``None`` when derivation **abstained** -- the clip was measured but
    its traffic does not define a single legal direction. That is a successful
    outcome carrying an honest absence, not a failure, and the scene is bound
    either way.
    """

    scene: SceneConfig
    scene_hash: str
    flow: FlowEstimate | None


class CalibrationOutcome(StrEnum):
    """How the scene a run is about to use was arrived at.

    Recorded so the reason a run reasoned about the geometry it did is legible in
    the log and in the tests, rather than being inferable only from which branch
    happened not to raise.
    """

    #: The clip's motion supported a legal direction; the derived scene is bound.
    DERIVED = "derived"
    #: The clip was measured but its traffic defines no single legal direction.
    #: A frame-correct scene with no direction is bound; wrong-way stays off.
    ABSTAINED = "abstained"
    #: The calibration pass itself failed. Nothing is bound and the run proceeds
    #: on an unbound, frame-correct, direction-free scene.
    FAILED = "failed"
    #: No rule in the resolved set needs a derived legal direction, so no
    #: detector pass was run at all. Same unbound frame-correct scene.
    SKIPPED = "skipped"


JobIdFactory = Callable[[], str]


def needs_derived_geometry(declared: tuple[RuleConfig, ...] | None) -> bool:
    """Whether running the calibration pass could change what a job does.

    The only thing derivation produces is a **legal direction**, so it is worth
    paying for only when a rule will read one. A caller that named its rules -- or
    a deployment that pinned them -- has already decided what runs, so the question
    is answerable exactly: does that set contain wrong-way? A set of geometry-free
    rules (no-helmet, triple-riding) reasons identically on a direction-free scene,
    so the detector pass would be pure cost on the analyst's wait.

    With nothing declared the set is derived *from the scene*, so the answer is yes
    by construction: without calibration the scene could never offer wrong-way, and
    the upload would silently lose a violation its footage actually supports.
    """

    if not declared:
        return True
    return any(isinstance(rule, WrongWayRuleConfig) for rule in declared)


def _default_job_id() -> str:
    return "job-" + uuid.uuid4().hex[:16]


# --- videos --------------------------------------------------------------------
class VideoService:
    """Validates and stores uploaded videos, addressed by content hash."""

    def __init__(self, config: AppConfig, store: VideoStore) -> None:
        self._config = config
        self._store = store

    def assert_supported_extension(self, filename: str) -> None:
        """Fast-fail before reading a body: reject an unsupported extension."""

        if not self._config.is_supported_extension(Path(filename).suffix):
            allowed = ", ".join(sorted(self._config.allowed_extensions))
            raise UnsupportedMediaError(
                f"unsupported file extension {Path(filename).suffix!r}; "
                f"supported: {allowed}"
            )

    def store_upload(self, filename: str, data: bytes) -> VideoRecord:
        """Validate + persist one upload; return its record.

        Validates extension, non-emptiness, size, content-uniqueness, and
        readability (by actually opening the file through P1-U5 ingestion). A file
        that fails the readability check is removed, so no half-valid upload
        lingers on disk.
        """

        self.assert_supported_extension(filename)
        if not data:
            raise UnsupportedMediaError("uploaded file is empty")
        if len(data) > self._config.max_upload_bytes:
            raise PayloadTooLargeError(
                f"upload is {len(data)} bytes; the limit is "
                f"{self._config.max_upload_bytes} bytes"
            )

        video_id = "vid-" + hashlib.sha256(data).hexdigest()[:16]
        if self._store.contains(video_id):
            raise DuplicateVideoError(
                f"an identical video already exists as {video_id}", video_id=video_id
            )

        self._config.videos_dir.mkdir(parents=True, exist_ok=True)
        path = self._config.videos_dir / f"{video_id}{Path(filename).suffix.lower()}"
        path.write_bytes(data)

        try:
            source = FileFrameSource(path)
            metadata = source.metadata
        except VideoIngestionError as exc:
            path.unlink(missing_ok=True)
            raise UnsupportedMediaError(
                f"uploaded file is not a readable video: {exc}"
            ) from exc

        record = VideoRecord(
            video_id=video_id,
            filename=filename,
            path=path,
            size_bytes=len(data),
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            duration_seconds=metadata.duration_seconds,
            codec=metadata.codec,
            uploaded_at=datetime.now(UTC),
        )
        self._store.add(record)
        return record

    def require(self, video_id: str) -> VideoRecord:
        """Return the record for ``video_id`` or raise :class:`VideoNotFoundError`."""

        record = self._store.get(video_id)
        if record is None:
            raise VideoNotFoundError(f"no uploaded video with id {video_id!r}")
        return record

    def media_path(self, video_id: str) -> Path:
        """The stored source file for a video, for playback (H11).

        The read side of ``store_upload``. Uploads are addressed by content id and
        stored under it, so nothing but this service can turn an id back into a
        path -- which is why a video that was not uploaded in the current browser
        session had no playable source before H11.
        """

        record = self.require(video_id)
        if not record.path.is_file():
            raise VideoMediaNotFoundError(
                f"the stored file for video {video_id!r} is no longer on disk"
            )
        return record.path


class SceneService:
    """Authors, stores, and binds per-video scenes (H12).

    The service that ends the application-wide scene. It owns three things and
    delegates the rest:

    * **authoring** -- delegated wholly to
      :func:`~trafficpulse.scenes.builder.build_scene`, which is deterministic and
      pure, so this service invents no geometry and stamps no provenance;
    * **storage** -- delegated to the content-addressed
      :class:`~trafficpulse.persistence.SceneStore`, so an edit is a new revision
      rather than an overwrite and every historical event's ``scene_config_hash``
      stays resolvable;
    * **binding** -- one call to ``VideoStore.bind_scene``, which persists through
      the same recovery observer every other video change uses.

    Scene *validity* is never decided here: the ``SceneConfig`` contract validates
    its own structure and references, and the shipped rules decide what they can
    reason over (see :mod:`trafficpulse.app.capabilities`). This service only
    reports what they said.
    """

    def __init__(
        self,
        store: SceneStore,
        videos: VideoService,
        video_store: VideoStore,
        *,
        classifier_available: bool = False,
        fallback: SceneConfig | None = None,
    ) -> None:
        self._store = store
        self._videos = videos
        self._video_store = video_store
        self._classifier_available = classifier_available
        # The server's configured scene, used only for videos nobody has
        # calibrated. Retained so an operator with one fixed camera keeps working
        # exactly as before H12; it is a default, no longer the only answer.
        self._fallback = fallback

    @property
    def has_fallback(self) -> bool:
        """Whether the server has a configured fallback scene (H16 readiness).

        An uncalibrated video can only be processed when one exists, so this is a
        genuine readiness signal rather than a configuration detail.
        """

        return self._fallback is not None

    # --- reading -------------------------------------------------------------
    def get(self, scene_hash: str) -> SceneConfig:
        """One stored scene revision, verbatim (404 when this repository has none).

        Addressable by any ``scene_config_hash``, including one read off a
        historical ``ConfirmedEvent`` -- which is what makes an event's declared
        provenance something a client can actually fetch and inspect.
        """

        try:
            scene = self._store.get(scene_hash)
        except CorruptRecordError as exc:
            raise SceneNotFoundError(f"scene {scene_hash!r} is stored but unreadable") from exc
        if scene is None:
            raise SceneNotFoundError(f"no stored scene with hash {scene_hash!r}")
        return scene

    def for_video(self, video_id: str) -> SceneConfig | None:
        """The scene processing should use for a video: its own, else the fallback.

        Resolution order is the whole point of H12 -- the video's binding wins, and
        the server's configured scene is consulted only when there is no binding.
        Returns ``None`` when neither exists, which the caller reports as a clean
        "this video has no scene" rather than processing against unrelated geometry.
        """

        record = self._video_store.get(video_id)
        if record is not None and record.scene_hash is not None:
            stored = self._store.get(record.scene_hash)
            if stored is not None:
                return stored
            # The binding survived but its revision did not. Falling through to the
            # server default silently would reason over another camera's geometry.
            _logger.warning(
                "video %s is bound to missing scene %s; treating as uncalibrated",
                video_id,
                record.scene_hash,
            )
        return self._fallback

    def summary_for_video(self, video_id: str) -> SceneSummary:
        """The bound scene's summary, or 404 when the video is not calibrated."""

        record = self._videos.require(video_id)
        if record.scene_hash is None:
            raise SceneNotFoundError(f"video {video_id!r} has no calibrated scene")
        return self.summarise(self.get(record.scene_hash))

    def summarise(self, scene: SceneConfig) -> SceneSummary:
        """Present a scene for browsing (see :class:`SceneSummary`)."""

        return SceneSummary(
            scene_hash=scene_config_hash(scene),
            scene_id=scene.scene.scene_id,
            scene_name=scene.scene.scene_name,
            camera_id=scene.scene.camera_id,
            site_id=scene.scene.site_id,
            status=scene.scene.status,
            calibration_status=scene.calibration.status,
            frame_width=scene.frame.reference_width,
            frame_height=scene.frame.reference_height,
            zone_count=len(scene.zones),
            has_legal_direction=bool(scene.legal_directions),
            has_no_stopping_zone=any(
                zone.enabled and zone.zone_type is ZoneType.NO_STOPPING
                for zone in scene.zones
            ),
            # Read off the scene's own recorded provenance rather than tracked
            # beside it, so a stored scene still reports truthfully after a restart.
            derived=scene.calibration.source == CALIBRATION_SOURCE_AUTO,
            supported_violations=self.supported_violations(scene),
        )

    def supported_violations(self, scene: SceneConfig) -> tuple[ViolationType, ...]:
        return supported_violations(scene, classifier_available=self._classifier_available)

    def violations_for_video(self, video_id: str) -> tuple[ViolationType, ...]:
        """What can be run for a video, given whichever scene resolves for it."""

        scene = self.for_video(video_id)
        return () if scene is None else self.supported_violations(scene)

    # --- automatic calibration (observable facts only) -------------------------
    def is_calibrated(self, video_id: str) -> bool:
        """Whether this video has a scene of **its own**, as opposed to the fallback.

        The question auto-calibration turns on, and deliberately not "did
        :meth:`for_video` return something" -- that is true for every video once a
        deployment configures a fallback. An analyst's binding is authoritative and
        must never be re-derived over.
        """

        record = self._video_store.get(video_id)
        return record is not None and record.scene_hash is not None

    def _draft_for(
        self, video_id: str, *, flow: FlowEstimate | None, name: str
    ) -> SceneDraft:
        """A draft carrying the video's real frame and only what was observed.

        One zone spanning the frame, because the roadway's true extent is not
        observable and a smaller invented polygon would silently exclude traffic.
        A legal direction is attached **only** when ``flow`` is given; no
        no-stopping zone, stop line or signal group is ever produced here, because
        none of them can be read off arbitrary footage.
        """

        video = self._videos.require(video_id)
        inset = 1.0  # keep every vertex strictly inside the declared frame
        width, height = float(video.width), float(video.height)
        polygon = (
            (inset, inset),
            (width - inset, inset),
            (width - inset, height - inset),
            (inset, height - inset),
        )
        direction = (
            DirectionDraft(
                direction_id=DERIVED_DIRECTION_ID,
                dx=flow.dx,
                dy=flow.dy,
                zone_id=DERIVED_LANE_ID,
                description=(
                    "Legal travel = this clip's observed dominant traffic flow "
                    f"({flow.heading_degrees:.1f} deg, {flow.mover_count} moving "
                    "vehicles). Derived, not surveyed."
                ),
            )
            if flow is not None
            else None
        )
        return SceneDraft(
            scene_name=name,
            camera_id=f"cam-{video_id}",
            site_id="site-auto",
            description=(
                "Automatically derived from the clip's own motion. Frame size is "
                "measured; legal direction is estimated from observed traffic. No "
                "no-stopping zone, stop line or signal timing is claimed -- none of "
                "those is observable from footage."
            ),
            frame_width=video.width,
            frame_height=video.height,
            zones=(
                ZoneDraft(
                    zone_id=DERIVED_LANE_ID,
                    zone_type=ZoneType.LANE,
                    polygon=polygon,
                    description="Full frame; the roadway's true extent is not observable.",
                ),
            ),
            direction=direction,
        )

    def provisional_scene(self, video_id: str) -> SceneConfig:
        """A frame-correct scene claiming no direction, for the calibration pass.

        Exists so the perception pass runs against the video's own frame rather
        than an unrelated one. It supports only the geometry-free rules, and is
        never bound -- it is scaffolding for the pass that produces the real one.
        """

        return build_scene(
            self._draft_for(video_id, flow=None, name="auto-provisional"),
            scene_id=f"scene-{video_id}",
            calibration_source=CALIBRATION_SOURCE_AUTO,
        )

    def derive_from_motion(
        self, video_id: str, states: Sequence[TrackState]
    ) -> DerivedScene:
        """Author, store and bind a scene from observed motion. Never invents.

        Returns the bound scene, its hash and the flow it rests on (``flow`` is
        ``None`` when no legal direction could be justified). Abstention is a real
        outcome, not a failure: :func:`estimate_dominant_flow` returns ``None`` for
        a two-way road whose movers cancel, and too few movers is too little
        evidence to declare a direction from. In both cases the video still gets a
        frame-correct scene of **its own** -- the geometry-free rules are
        unaffected -- and wrong-way simply stays unavailable until an analyst draws
        the lane. The abstaining scene is still bound, because "this clip was
        measured and its traffic does not define one direction" is a finding worth
        recording, and it is what keeps the caller off the deployment fallback.

        The scene is returned as well as bound so the caller can run against
        exactly the revision it stored, without a re-read that could observe a
        concurrent rebinding.
        """

        flow = estimate_dominant_flow(states)
        if flow is not None and flow.mover_count < MIN_FLOW_MOVERS:
            _logger.info(
                "video %s: flow estimate rests on %d mover(s) (< %d); declining to "
                "declare a legal direction",
                video_id,
                flow.mover_count,
                MIN_FLOW_MOVERS,
            )
            flow = None
        draft = self._draft_for(
            video_id,
            flow=flow,
            name="auto-derived" if flow is not None else "auto-derived-no-direction",
        )
        scene = build_scene(
            draft,
            scene_id=f"scene-{video_id}",
            calibration_source=CALIBRATION_SOURCE_AUTO,
        )
        scene_hash = self._store.put(scene)
        self._video_store.bind_scene(video_id, scene_hash)
        return DerivedScene(scene=scene, scene_hash=scene_hash, flow=flow)

    def default_rules_for(self, scene: SceneConfig) -> tuple[RuleConfig, ...]:
        """The rule set to run over ``scene`` when the client named none.

        The same capability probe :meth:`supported_violations` reports from, turned
        into runnable declarations -- so what a client is *offered* and what the
        server *runs by default* can never disagree. This service is the natural
        owner because deciding it needs both the scene and the deployment's
        classifier availability, which is exactly what this service already holds.
        """

        return rules_for(scene, classifier_available=self._classifier_available)

    # --- writing -------------------------------------------------------------
    def calibrate(self, video_id: str, draft: SceneDraft) -> SceneSummary:
        """Author this video's scene from a draft, store it, and bind it.

        One call because the three steps have no independent meaning: an authored
        scene nobody bound is orphaned data, and a binding to a scene nobody stored
        is dangling. Idempotent by construction -- an unchanged drawing rebuilds
        the same content, hashes to the same address, and rewrites the same bytes.

        The draft's frame size must match the video's decoded dimensions: geometry
        drawn against a different frame would land in the wrong place, and the
        contract's in-bounds check would only catch the cases that overflow.
        """

        video = self._videos.require(video_id)
        if (draft.frame_width, draft.frame_height) != (video.width, video.height):
            raise InvalidConfigurationError(
                f"the drawing is against a {draft.frame_width}x{draft.frame_height} "
                f"frame but video {video_id!r} is {video.width}x{video.height}; "
                "geometry must be drawn in the video's own pixel space"
            )

        scene = self._build(draft, video_id=video_id)
        scene_hash = self._store.put(scene)
        self._video_store.bind_scene(video_id, scene_hash)
        return self.summarise(scene)

    def validate(self, video_id: str, draft: SceneDraft) -> SceneValidationResponse:
        """Check a draft without storing it, reporting what it would unlock.

        The calibration surface's live feedback. A draft that cannot form a valid
        scene comes back with the contract's own messages rather than a 422, so a
        half-finished drawing is a *state* the UI can render, not a request error.
        """

        video = self._videos.require(video_id)
        if (draft.frame_width, draft.frame_height) != (video.width, video.height):
            return SceneValidationResponse(
                valid=False,
                errors=(
                    f"drawing frame {draft.frame_width}x{draft.frame_height} does not "
                    f"match the video's {video.width}x{video.height}",
                ),
            )
        try:
            scene = self._build(draft, video_id=video_id)
        except ValidationError as exc:
            return SceneValidationResponse(valid=False, errors=_validation_messages(exc))
        return SceneValidationResponse(
            valid=True,
            supported_violations=self.supported_violations(scene),
            scene_hash=scene_config_hash(scene),
        )

    def _build(self, draft: SceneDraft, *, video_id: str) -> SceneConfig:
        """Expand a draft, giving the scene a logical id derived from the video.

        The logical ``scene_id`` is stable across edits of the same video's scene
        (the *hash* is what changes), so successive revisions are recognisably the
        same site rather than looking like unrelated scenes.
        """

        return build_scene(draft, scene_id=f"scene-{video_id}")


def _validation_messages(error: ValidationError) -> tuple[str, ...]:
    """Flatten a pydantic error into client-safe lines naming the offending field."""

    lines: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "scene"
        lines.append(f"{location}: {detail['msg']}")
    return tuple(lines)


class VideoLibraryService:
    """Browsable metadata for every stored video -- the historical library (H11).

    A **join over the existing registries**, holding no state and adding no
    persistence of its own: :class:`~trafficpulse.app.registry.VideoStore` supplies
    the uploads (rebuilt from disk by H10 recovery), ``JobStore`` supplies each
    video's runs and its event index, and ``ReviewStore`` supplies review progress.
    A recovered repository therefore lists exactly as a live one does, because both
    are read through the same indices.

    What it deliberately does not do
    --------------------------------
    It never opens a ``ConfirmedEvent``, an ``EvidenceManifest``, an overlay, or a
    review journal. Event counts come from the in-memory event index (filenames,
    per H10), overlay availability from a path already held on the job record, and
    review progress from a single directory listing. Listing a repository of any
    size therefore costs no deserialisation -- the detail endpoints, which already
    exist, are what load a video's actual analysis once one is selected.
    """

    def __init__(
        self,
        videos: VideoStore,
        jobs: JobStore,
        reviews: ReviewStore | None = None,
        scenes: SceneService | None = None,
    ) -> None:
        self._videos = videos
        self._jobs = jobs
        # Optional for the same reason EventService's is: a library can be built
        # without the review layer, and then every video honestly reports that no
        # event has been reviewed.
        self._reviews = reviews
        # Optional likewise: without the scene layer a row reports no calibration
        # and no supported violations, which is the truthful answer for a library
        # built without it rather than a guess.
        self._scenes = scenes

    def list(self, *, limit: int, offset: int, sort: VideoSort) -> VideoListResponse:
        """Return a deterministic page of video summaries."""

        reviewed = self._reviewed_ids()
        summaries = [
            self._summarise(record, reviewed) for record in self._videos.videos()
        ]
        ordered = _sorted_videos(summaries, sort)
        return VideoListResponse(
            items=tuple(ordered[offset : offset + limit]),
            total=len(ordered),
            limit=limit,
            offset=offset,
        )

    def get(self, video_id: str) -> VideoSummary:
        """One video's library summary, or raise :class:`VideoNotFoundError`."""

        record = self._videos.get(video_id)
        if record is None:
            raise VideoNotFoundError(f"no uploaded video with id {video_id!r}")
        return self._summarise(record, self._reviewed_ids())

    def _reviewed_ids(self) -> frozenset[str]:
        return self._reviews.reviewed_event_ids() if self._reviews is not None else frozenset()

    def _summarise(self, record: VideoRecord, reviewed: frozenset[str]) -> VideoSummary:
        jobs = self._jobs.for_video(record.video_id)
        opening = _opening_job(jobs)
        # Deduplicated across runs, matching what GET /api/events?video_id= returns:
        # reprocessing a video produces the same content-derived event ids, so
        # summing per-run counts would report the same violation several times.
        event_ids = {
            event_id
            for job in jobs
            if job.status is JobStatus.SUCCEEDED
            for event_id in job.event_ids
        }
        return VideoSummary(
            video_id=record.video_id,
            filename=record.filename,
            uploaded_at=record.uploaded_at,
            size_bytes=record.size_bytes,
            width=record.width,
            height=record.height,
            fps=record.fps,
            duration_seconds=record.duration_seconds,
            codec=record.codec,
            job_id=opening.job_id if opening is not None else None,
            status=opening.status if opening is not None else None,
            job_count=len(jobs),
            event_count=len(event_ids),
            events_reviewed=len(event_ids & reviewed),
            overlay_available=(
                opening is not None
                and opening.overlay_video is not None
                and opening.overlay_video.exists()
            ),
            media_available=record.path.is_file(),
            scene_hash=record.scene_hash,
            supported_violations=(
                self._scenes.violations_for_video(record.video_id)
                if self._scenes is not None
                else ()
            ),
        )


# --- processing ----------------------------------------------------------------
class ProcessingService:
    """Creates and drives processing jobs over the injected H6 engine."""

    def __init__(
        self,
        *,
        config: AppConfig,
        scenes: SceneService,
        provider: EngineProvider,
        store: EventStore,
        job_store: JobStore,
        executor: JobExecutor,
        videos: VideoService,
        job_id_factory: JobIdFactory = _default_job_id,
        artifacts: ArtifactStore | None = None,
        rendered: RenderedArtifactStore | None = None,
    ) -> None:
        self._config = config
        # The H14 rendering stores. Optional so a service can be built without the
        # rendering layer at all (older tests, a deployment with no drawing backend);
        # absent them a run persists exactly what it always did and renders nothing.
        self._artifacts = artifacts
        self._rendered = rendered
        # H12: the scene is resolved *per video* at submit time, not held here.
        # A service that owns one scene is exactly the singleton this milestone
        # removed -- it made every upload reason against one camera's geometry.
        self._scenes = scenes
        self._provider = provider
        self._store = store
        self._jobs = job_store
        self._executor = executor
        self._videos = videos
        self._job_id_factory = job_id_factory

    def submit(
        self, *, video_id: str, rules: tuple[RuleConfig, ...] | None
    ) -> JobRecord:
        """Validate the request, create a job, and schedule its execution.

        The scene comes from the **video**, so two videos from different cameras are
        reasoned over their own geometry in the same process. Where it comes from
        depends on the video and the deployment:

        * an analyst-calibrated video uses its bound scene;
        * an uncalibrated one uses a scene derived from its own motion when
          ``auto_calibrate_uploads`` is on -- and then the configured scene is
          **not** a fallback for it, because another camera's geometry is not a
          safe substitute (see :meth:`_scene_for_derived_run`);
        * otherwise the server's configured scene, the pre-H12 behaviour an
          operator with one fixed camera still relies on.

        Rule resolution, in order:

        1. the request's own ``rules`` -- honoured verbatim, including rules this
           scene cannot satisfy, which still fail fast as a clean 400. A client that
           asked for something specific is told it cannot have it, never quietly
           given something else;
        2. the server's configured ``default_rules`` -- a deliberate deployment
           override, so an operator who pinned a rule set keeps it;
        3. otherwise **derived from the resolved scene**: every shipped rule this
           scene can legitimately support (see
           :meth:`SceneService.default_rules_for`). This is what makes an ordinary
           upload a multi-violation run instead of a fixed two-rule one, while
           keeping the system scene-aware -- it never runs a rule the scene cannot
           support, and never reaches for a violation with no shipped reasoner.

        Validation is eager **when the scene is already known**: the engine is built
        here, so an invalid scene/rule combination or an unavailable backend fails
        as a clean HTTP error before the job is scheduled.

        A video whose scene must be derived cannot be validated eagerly -- the rules
        are chosen from a scene that does not exist yet, and producing it costs a
        detector pass. Doing that here would block a request this endpoint documents
        as 202-then-poll, so the scene decision, the rule resolution and the engine
        build all move into the job (:meth:`_run_deriving_scene`), and a problem that
        would have been a 400/503 arrives instead as that job's ``failed`` status
        carrying the same message. Only that path is deferred; everything else
        validates exactly as it always did.
        """

        video = self._videos.require(video_id)
        # H12+: an upload nobody calibrated derives a scene from its own motion, so
        # the geometry rules reason about *this* camera. That needs a detector pass,
        # which must not happen on the request thread -- this endpoint's contract is
        # 202-then-poll -- so the whole scene decision moves inside the job. An
        # analyst-calibrated video never takes this path (see ``is_calibrated``) and
        # keeps the eager validation below unchanged.
        if self._config.auto_calibrate_uploads and not self._scenes.is_calibrated(video_id):
            job = JobRecord(job_id=self._job_id_factory(), video_id=video_id)
            self._jobs.add(job)
            self._executor.submit(lambda: self._run_deriving_scene(job, video, rules))
            return job

        scene = self._scenes.for_video(video_id)
        if scene is None:
            raise InvalidConfigurationError(
                f"video {video_id!r} has no calibrated scene and the server has no "
                "default scene configured; calibrate the video before processing it"
            )
        resolved = self._resolve_rules(video_id, scene, rules)

        engine = self._build_engine(scene, resolved)
        job = JobRecord(job_id=self._job_id_factory(), video_id=video_id, engine=engine)
        self._jobs.add(job)
        self._executor.submit(lambda: self._run(job, video, scene))
        return job

    def _resolve_rules(
        self, video_id: str, scene: SceneConfig, rules: tuple[RuleConfig, ...] | None
    ) -> tuple[RuleConfig, ...]:
        """Apply the documented rule-resolution order (request, deployment, scene)."""

        resolved = rules or self._config.default_rules or self._scenes.default_rules_for(scene)
        if not resolved:
            raise InvalidConfigurationError(
                "no rules were specified, the server has no default rule set, and "
                f"the scene resolved for video {video_id!r} supports no shipped "
                "rule; calibrate the video before processing it"
            )
        return resolved

    # --- the calibration phase of a job (never on the request thread) ---------
    def _run_deriving_scene(
        self, job: JobRecord, video: VideoRecord, rules: tuple[RuleConfig, ...] | None
    ) -> None:
        """Run one job whose scene has to be derived before inference can start.

        The lifecycle is the ordinary one with a bounded phase in front of it:
        create job -> derive the scene -> build the engine -> :meth:`_run` (which
        marks running, infers, persists, renders). Everything after derivation is
        the *same code path* an analyst-calibrated video takes; nothing about
        inference, persistence, evidence, overlays or run scoping is duplicated
        here.

        The job stays ``PENDING`` across derivation and only goes ``RUNNING`` when
        real inference starts, so progress never counts calibration frames as
        analysis frames. Cancellation is honoured throughout -- the derivation loop
        polls the same flag the engine does -- and any configuration or backend
        problem that would have been an eager 4xx/503 becomes this job's ``FAILED``
        status carrying the identical message, because there is no longer a request
        open to answer with it.
        """

        try:
            if self._jobs.is_cancel_requested(job.job_id):
                raise RunCancelledError("cancelled before calibration started")
            scene, outcome = self._scene_for_derived_run(job, video, rules)
            resolved = self._resolve_rules(video.video_id, scene, rules)
            engine = self._build_engine(scene, resolved)
        except RunCancelledError:
            _logger.info("processing job %s cancelled during calibration", job.job_id)
            self._jobs.mark_cancelled(job.job_id)
            return
        except Exception as exc:  # noqa: BLE001 - a job thread must never crash silently
            _logger.exception("processing job %s failed before inference", job.job_id)
            self._jobs.mark_failed(job.job_id, str(exc))
            return
        _logger.info(
            "job %s: scene calibration %s; running %d rule(s) against a %dx%d scene",
            job.job_id,
            outcome.value,
            len(resolved),
            scene.frame.reference_width,
            scene.frame.reference_height,
        )
        self._jobs.set_engine(job.job_id, engine)
        self._run(job, video, scene)

    def _needs_derived_geometry(self, rules: tuple[RuleConfig, ...] | None) -> bool:
        """Whether calibration could change this job, given the rules it declares."""

        return needs_derived_geometry(rules or self._config.default_rules)

    def _scene_for_derived_run(
        self, job: JobRecord, video: VideoRecord, rules: tuple[RuleConfig, ...] | None
    ) -> tuple[SceneConfig, CalibrationOutcome]:
        """The scene an uncalibrated upload is reasoned about, and how it was reached.

        **Never returns the deployment fallback.** That is the whole point: the
        fallback is another camera's geometry, and running an upload's geometry
        rules against it is the defect automatic calibration exists to remove --
        silently, since those rules then run and can only ever confirm nothing.
        Every branch here yields a scene in *this video's* pixel space:

        * derivation succeeded -- the bound scene, carrying the observed direction;
        * derivation abstained -- the bound scene, carrying no direction, so
          wrong-way is simply unavailable;
        * derivation failed, or was not needed -- the unbound provisional scene:
          the video's real frame, one full-frame lane, and nothing inferred.

        A failed or skipped derivation binds nothing, so the video stays honestly
        uncalibrated: ``GET /api/videos/{id}/scene`` still reports 404, a later run
        retries, and an analyst's own calibration is unaffected. In no branch is a
        legal direction, a no-stopping zone or a signal schedule invented.
        """

        if not self._needs_derived_geometry(rules):
            _logger.info(
                "job %s: no rule needs a derived legal direction; skipping the "
                "calibration pass and running against video %s's own frame",
                job.job_id,
                video.video_id,
            )
            return self._scenes.provisional_scene(video.video_id), CalibrationOutcome.SKIPPED

        # Built before the try: a video whose own frame cannot produce a scene has
        # nothing safe to fall back *to*, so it must fail the job rather than borrow
        # geometry from somewhere else.
        provisional = self._scenes.provisional_scene(video.video_id)
        try:
            derived = self._calibrate(job, video, provisional)
        except RunCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never fail a run over calibration
            _logger.warning(
                "job %s: calibration failed for video %s (%s); running against the "
                "video's own frame with no legal direction, and binding no scene",
                job.job_id,
                video.video_id,
                exc,
            )
            return provisional, CalibrationOutcome.FAILED

        if derived.flow is None:
            _logger.info(
                "job %s: video %s measured, but its traffic defines no single legal "
                "direction; bound a direction-free scene (wrong-way unavailable)",
                job.job_id,
                video.video_id,
            )
            return derived.scene, CalibrationOutcome.ABSTAINED
        _logger.info(
            "job %s: video %s auto-calibrated as scene %s (legal direction %.1f deg "
            "from %d mover(s))",
            job.job_id,
            video.video_id,
            derived.scene_hash[:12],
            derived.flow.heading_degrees,
            derived.flow.mover_count,
        )
        return derived.scene, CalibrationOutcome.DERIVED

    def _calibrate(
        self, job: JobRecord, video: VideoRecord, provisional: SceneConfig
    ) -> DerivedScene:
        """The bounded perception pass, and the scene derived from what it saw.

        Runs the detector and tracker over a bounded prefix of the clip with **no
        reasoning at all**: this loop submits and drains but deliberately never
        calls ``finalize``, so no observation is derived, no event can be minted and
        nothing is persisted. The tracks it accumulates are the only output.

        Bounded on purpose -- it is overhead on top of the real run, and a few
        seconds of traffic is all a flow estimate needs -- and cancellable, so a
        client that changes its mind during calibration is not made to wait out the
        whole budget.
        """

        # The engine contract requires at least one rule, so the pass carries the
        # one rule any scene supports without geometry or a classifier.
        engine = self._provider.create(scene=provisional, rules=(TripleRidingRuleConfig(),))
        source = FileFrameSource(video.path, camera_id=provisional.scene.camera_id)
        engine.reset()
        for record in islice(source.frames(), CALIBRATION_FRAME_BUDGET):
            if self._jobs.is_cancel_requested(job.job_id):
                raise RunCancelledError("cancelled during scene calibration")
            engine.submit(record)
            engine.drain()
        return self._scenes.derive_from_motion(video.video_id, engine.track_states())

    def _build_engine(
        self, scene: SceneConfig, rules: tuple[RuleConfig, ...]
    ) -> InferenceEngine:
        try:
            return self._provider.create(scene=scene, rules=rules)
        except (
            SceneConfigurationError,
            EngineConfigurationError,
            UnsupportedRuleError,
            ValueError,
        ) as exc:
            raise InvalidConfigurationError(str(exc)) from exc
        except EngineUnavailableError:
            raise
        except DetectorError as exc:
            raise EngineUnavailableError(
                f"the inference backend is unavailable: {exc}"
            ) from exc

    def _run(self, job: JobRecord, video: VideoRecord, scene: SceneConfig) -> None:
        """Execute one job: run the H6 engine and persist its output.

        The run is cooperatively cancellable: the engine checks the job's cancel
        flag between frames, and a :class:`RunCancelledError` is recorded as a
        clean ``CANCELLED`` outcome (not a failure) with nothing persisted.
        """

        assert job.engine is not None
        try:
            source = FileFrameSource(video.path, camera_id=scene.scene.camera_id)
            self._jobs.mark_running(job.job_id, frames_total=source.metadata.frame_count)
            result = job.engine.run(
                source, should_cancel=lambda: self._jobs.is_cancel_requested(job.job_id)
            )
            stored = job.engine.persist(result, store=self._store, run_id=job.job_id)
            # Declare the overlay pending BEFORE the job goes terminal. A client
            # stops polling once it sees a terminal job status, so the very first
            # succeeded status it reads must already announce that an annotated
            # video is on its way -- otherwise the render (a second decode+encode
            # pass, often longer than inference) lands after the client has stopped
            # listening and the workspace plays the raw upload forever.
            self._jobs.mark_overlay_pending(job.job_id)
            self._jobs.mark_succeeded(job.job_id, result)
            # Evidence stills first: they are what the review surface shows, they
            # decode only up to the last evidence frame, and they must not wait
            # behind a full-clip re-encode.
            self._render_evidence_artifacts(job, video, scene, stored)
            self._render_overlay_video(job, video, scene, result)
        except RunCancelledError:
            _logger.info("processing job %s cancelled", job.job_id)
            self._jobs.mark_cancelled(job.job_id)
        except Exception as exc:  # noqa: BLE001 - a job thread must never crash silently
            _logger.exception("processing job %s failed", job.job_id)
            self._jobs.mark_failed(job.job_id, str(exc))

    def _render_evidence_artifacts(
        self,
        job: JobRecord,
        video: VideoRecord,
        scene: SceneConfig,
        stored: Sequence[StoredEvent],
    ) -> None:
        """Render + store this run's evidence frames (best-effort, never fatal).

        Runs *after* the events and their manifests are durably persisted, and writes
        only to the content-addressed artifact store and the append-only sidecar --
        so a failure here costs pictures, never records. The manifests it reads are
        the ones just written, which is what makes the rendered frames provably the
        frames the engine picked.

        The lifecycle (H16) is recorded around the work: ``PENDING`` is persisted
        *before* rendering starts, so a restart mid-render leaves a state recovery
        can recognise as interrupted rather than one that passes for complete.
        Whatever happens, the status is left terminal.
        """

        if self._artifacts is None or self._rendered is None or not stored:
            self._jobs.set_evidence_status(job.job_id, EvidenceStatus.NONE)
            return
        assert job.engine is not None
        self._jobs.set_evidence_status(job.job_id, EvidenceStatus.PENDING)
        try:
            report = render_run_evidence(
                pairs=[(pair.event, pair.manifest) for pair in stored],
                source_path=video.path,
                camera_id=scene.scene.camera_id,
                artifacts=self._artifacts,
                rendered=self._rendered,
                # The same metadata the annotated video is drawn from, so a still and
                # the video can never disagree about what was concluded.
                compositor=build_job_compositor(job.engine, [pair.event for pair in stored]),
            )
            _logger.info(
                "job %s rendered evidence for %d event(s): %d artifact(s) from %d frame(s)",
                job.job_id,
                report.events_rendered,
                report.artifacts_written,
                report.frames_decoded,
            )
            self._jobs.set_evidence_status(
                job.job_id,
                EvidenceStatus.READY if report.events_rendered else EvidenceStatus.NONE,
            )
        except Exception:  # noqa: BLE001 - evidence rendering is never fatal to a run
            self._jobs.set_evidence_status(job.job_id, EvidenceStatus.FAILED)
            _logger.exception(
                "evidence render failed for job %s; events and manifests are persisted",
                job.job_id,
            )

    def _render_overlay_video(
        self, job: JobRecord, video: VideoRecord, scene: SceneConfig, result: EngineRunResult
    ) -> None:
        """Render the annotated overlay video for a finished job (best-effort).

        Presentation-only: the events + evidence are already persisted before this
        runs, so any render failure (e.g. Pillow absent, an encode error) is logged
        and swallowed -- the job stays ``succeeded``, the original video still plays,
        and every read endpoint still works. The original upload is never modified;
        the annotated video is a separate artifact under ``overlays_dir``.

        Whatever happens, the job's :class:`~trafficpulse.app.registry.OverlayStatus`
        is left **terminal** (``READY`` / ``NONE`` / ``FAILED``): a client polling for
        a pending overlay must always be released, including on the failure paths.
        """

        assert job.engine is not None
        try:
            output = self._config.overlays_dir / f"{job.job_id}.mp4"
            rendered = render_job_overlay(
                engine=job.engine,
                source_path=video.path,
                output_path=output,
                events=result.events,
                camera_id=scene.scene.camera_id,
            )
            if rendered is not None:
                self._jobs.set_overlay_video(job.job_id, rendered.output_path)
            else:
                # The run produced no overlay metadata: nothing to draw, not a fault.
                self._jobs.resolve_overlay(job.job_id, OverlayStatus.NONE)
        except Exception:  # noqa: BLE001 - overlay is a presentation aid, never fatal
            self._jobs.resolve_overlay(job.job_id, OverlayStatus.FAILED)
            _logger.exception(
                "overlay render failed for job %s; original video still plays", job.job_id
            )

    def repair_evidence(self, job_id: str) -> EvidenceRepairResponse:
        """Re-render evidence frames for events of ``job_id`` that have none (H16).

        The repair path for a render interrupted by a restart. It **does not
        reprocess the video**: no detector, tracker, or reasoner runs, no event is
        created or altered, and the write-once manifests are only read. It decodes
        the source at the media times those manifests already record and stores the
        frames that are missing.

        Only events with *no* rendered artifact are touched -- an event that already
        has frames is left exactly as it was, so repair can never replace evidence
        that was rendered correctly.

        **Repaired frames carry no overlay annotation.** The per-frame reasoning
        metadata lives in the engine that produced the run and does not survive the
        process, so a repaired still shows the real evidence pixels without the
        boxes and banners a freshly-rendered one has. That is a visible, reported
        limitation rather than a silent difference: the response says how many
        events were repaired, and the alternative -- leaving the analyst with no
        picture at all -- is worse.
        """

        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")
        if self._artifacts is None or self._rendered is None:
            raise InvalidConfigurationError(
                "this deployment has no evidence rendering layer configured"
            )
        if job.status is not JobStatus.SUCCEEDED:
            raise InvalidConfigurationError(
                f"job {job_id!r} is {job.status.value!r}; only a succeeded run has "
                "evidence to repair"
            )

        video = self._videos.require(job.video_id)
        try:
            stored = self._store.load(job_id)
        except RunNotFoundError as exc:
            raise JobNotFoundError(f"run {job_id!r} has no persisted records") from exc

        missing = [
            pair for pair in stored if not self._rendered.artifacts(pair.event.event_id)
        ]
        if not missing:
            self._jobs.set_evidence_status(job_id, EvidenceStatus.READY)
            return EvidenceRepairResponse(
                job_id=job_id, events_repaired=0, artifacts_written=0,
                evidence_status=EvidenceStatus.READY,
            )

        self._jobs.set_evidence_status(job_id, EvidenceStatus.PENDING)
        try:
            report = render_run_evidence(
                pairs=[(pair.event, pair.manifest) for pair in missing],
                source_path=video.path,
                # The camera the events were reasoned under; `missing` is non-empty
                # here, and every event of a run shares one camera.
                camera_id=missing[0].event.camera_id,
                artifacts=self._artifacts,
                rendered=self._rendered,
                compositor=None,  # see the docstring: no metadata survives a restart
            )
        except Exception as exc:  # noqa: BLE001 - repair is best-effort, like the render
            self._jobs.set_evidence_status(job_id, EvidenceStatus.FAILED)
            _logger.exception("evidence repair failed for job %s", job_id)
            raise InvalidConfigurationError(f"evidence repair failed: {exc}") from exc

        still_missing = any(
            not self._rendered.artifacts(pair.event.event_id) for pair in stored
        )
        status = EvidenceStatus.FAILED if still_missing else EvidenceStatus.READY
        self._jobs.set_evidence_status(job_id, status)
        _logger.info(
            "job %s evidence repair: %d event(s), %d artifact(s), now %s",
            job_id,
            report.events_rendered,
            report.artifacts_written,
            status.value,
        )
        return EvidenceRepairResponse(
            job_id=job_id,
            events_repaired=report.events_rendered,
            artifacts_written=report.artifacts_written,
            evidence_status=status,
        )

    def overlay_video_path(self, job_id: str) -> Path:
        """The rendered overlay video for a job, or a 404 when none is available."""

        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")
        if job.overlay_video is None or not job.overlay_video.exists():
            raise OverlayNotFoundError(f"no overlay video is available for job {job_id!r}")
        return job.overlay_video

    def cancel(self, job_id: str) -> JobStatusResponse:
        """Request cancellation of a job and return its current status.

        Cancellation is cooperative and asynchronous: for a running job this
        flags the background run, which stops at the next frame and transitions
        to ``cancelled`` (the client keeps polling until it observes that).
        Cancelling an already-finished job is a safe no-op that returns its
        existing status. An unknown job id is a 404.
        """

        if self._jobs.get(job_id) is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")
        self._jobs.request_cancel(job_id)
        return self.status(job_id)

    def status(self, job_id: str) -> JobStatusResponse:
        """Return one job's live status (unavailable values are null, not faked)."""

        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")

        metrics = job.metrics()
        processed = metrics.frames_processed if metrics is not None else 0
        total = job.frames_total
        fps = metrics.media_fps if metrics is not None else None

        progress: float | None
        if job.status is JobStatus.SUCCEEDED:
            progress = 1.0
        elif total and metrics is not None:
            progress = min(1.0, processed / total)
        else:
            progress = None

        remaining: float | None = None
        if (
            job.status is JobStatus.RUNNING
            and metrics is not None
            and metrics.wall_fps
            and total
        ):
            remaining = max(0.0, (total - processed) / metrics.wall_fps)

        return JobStatusResponse(
            job_id=job.job_id,
            video_id=job.video_id,
            status=job.status,
            progress=progress,
            frames_processed=processed,
            frames_total=total,
            fps=fps,
            estimated_remaining_seconds=remaining,
            event_count=len(job.event_ids),
            error=job.error,
            overlay_available=job.overlay_video is not None and job.overlay_video.exists(),
            overlay_status=job.overlay_status,
            evidence_status=job.evidence_status,
        )


# --- events + evidence ---------------------------------------------------------
class EventService:
    """Reads persisted confirmed events back from the H6 ``EventStore``."""

    def __init__(
        self,
        store: EventStore,
        job_store: JobStore,
        reviews: ReviewStore | None = None,
    ) -> None:
        self._store = store
        self._jobs = job_store
        # Optional so an EventService can still be built without the review layer
        # (older tests, and any caller that only reads events). Absent it, every
        # summary reports the honest default of PENDING.
        self._reviews = reviews

    def list(
        self,
        *,
        video_id: str | None,
        job_id: str | None = None,
        limit: int,
        offset: int,
        sort: EventSort,
    ) -> EventListResponse:
        """Return a deterministic page of event summaries.

        Two scopes, both legitimate (R7):

        * ``job_id`` given -- **one run's** events. This is what a review surface
          needs: after a video is reprocessed, the analyst is looking at a
          particular run and must not be shown a superseded run's conclusions.
        * ``job_id`` omitted -- every succeeded run of the video, deduplicated by
          ``event_id``. Reprocessing re-confirms byte-identical events, so this is
          "what has this video ever been found to contain", which the library's
          ``event_count`` and the review journal are both built around.

        Scoping happens **here**, at the data-access boundary: the selected runs are
        the only ones read from the store, so a caller can never be handed events it
        then has to filter. An unknown, unfinished, or mismatched ``job_id`` selects
        no run and yields an empty page -- the same shape an unknown ``video_id``
        has always produced, rather than a new error class for the same situation.
        """

        summaries: list[EventSummary] = []
        seen: set[str] = set()
        # Newest run first, so a duplicated event is attributed to the most recent
        # run that produced it. That matches ``JobStore.job_for_event`` (which
        # ``locate`` uses to serve the event and its evidence), ``_opening_job``, and
        # the analytics breakdown -- previously this loop ran oldest-first and
        # labelled such an event with a run its evidence was *not* served from.
        for job in reversed(self._scoped_jobs(video_id=video_id, job_id=job_id)):
            if not job.event_ids:
                continue  # a succeeded job that confirmed nothing persisted no run
            for pair in self._store.load(job.job_id):
                event = pair.event
                if event.event_id in seen:
                    continue
                seen.add(event.event_id)
                summaries.append(
                    EventSummary(
                        event_id=event.event_id,
                        video_id=job.video_id,
                        job_id=job.job_id,
                        violation_type=event.violation_type,
                        camera_id=event.camera_id,
                        track_ids=event.track_ids,
                        start_at=event.start_at,
                        trigger_at=event.trigger_at,
                        rule_id=event.rule_id,
                        # Verbatim: the components are the contract's, un-merged.
                        confidence=event.confidence,
                    )
                )

        ordered = _sorted_summaries(summaries, sort)
        page = ordered[offset : offset + limit]
        # Fold review status only for the page actually returned: badging is a
        # per-event journal read, so doing it for the whole result set would make
        # paging cost more the deeper you go for data nobody asked for.
        if self._reviews is not None and page:
            statuses = self._reviews.statuses(summary.event_id for summary in page)
            page = [
                summary.model_copy(
                    update={
                        "review_status": statuses.get(
                            summary.event_id, ReviewStatus.PENDING
                        )
                    }
                )
                for summary in page
            ]
        return EventListResponse(
            items=tuple(page), total=len(ordered), limit=limit, offset=offset
        )

    def _scoped_jobs(
        self, *, video_id: str | None, job_id: str | None
    ) -> tuple[JobRecord, ...]:
        """The runs a listing should read, in insertion (oldest-first) order.

        Naming a run narrows to exactly that run, and a ``video_id`` given alongside
        it is applied as a **check** rather than ignored: a pair that does not agree
        describes no run this repository holds, so it selects none instead of
        quietly answering the question the client did not ask.

        Only succeeded runs are ever selected -- a pending, running, or failed job
        has no persisted events to read, so scoping to one is honestly empty.
        """

        if job_id is None:
            return self._jobs.succeeded_for_video(video_id)
        job = self._jobs.get(job_id)
        if (
            job is None
            or job.status is not JobStatus.SUCCEEDED
            or (video_id is not None and job.video_id != video_id)
        ):
            return ()
        return (job,)

    def get(self, event_id: str) -> ConfirmedEvent:
        """Return the full contract for one event or raise :class:`EventNotFoundError`."""

        return self.locate(event_id)[0]

    def locate(self, event_id: str) -> tuple[ConfirmedEvent, EvidenceManifest]:
        """Find one event + its manifest across succeeded runs (raises if unknown)."""

        job_id = self._jobs.job_for_event(event_id)
        if job_id is None:
            raise EventNotFoundError(f"no confirmed event with id {event_id!r}")
        try:
            stored = self._store.load(job_id)
        except RunNotFoundError as exc:  # pragma: no cover - index implies persistence
            raise EventNotFoundError(
                f"event {event_id!r} is indexed but its run is missing"
            ) from exc
        for pair in stored:
            if pair.event.event_id == event_id:
                return pair.event, pair.manifest
        raise EventNotFoundError(  # pragma: no cover - index guarantees membership
            f"event {event_id!r} is indexed to a run that does not contain it"
        )


class ReviewService:
    """Records analyst decisions and serves the review case + audit history (H9).

    The application-layer guard on the review lifecycle. It owns exactly three
    responsibilities and delegates everything else:

    * the event must exist (delegated to :class:`EventService`, so a decision can
      never be recorded against an id the system cannot produce evidence for);
    * the transition must be legal (delegated to the contract's pure state
      machine, so the API and any future caller enforce one rule set);
    * the action is **appended**, never written over (delegated to
      :class:`~trafficpulse.persistence.review_store.ReviewStore`).

    It holds no state. The current status is always re-folded from the journal, so
    two concurrent reviewers cannot leave the service caching a decision that disk
    disagrees with.
    """

    def __init__(self, events: EventService, store: ReviewStore) -> None:
        self._events = events
        self._store = store

    def get(self, event_id: str) -> ReviewResponse:
        """The event's current case and its full history (raises if unknown)."""

        _, manifest = self._events.locate(event_id)
        return ReviewResponse(
            case=self._store.case(
                event_id, evidence_package_id=manifest.evidence_package_id
            ),
            history=self._store.history(event_id),
        )

    def decide(self, event_id: str, request: ReviewDecisionRequest) -> ReviewResponse:
        """Record one analyst action and return the resulting case + history.

        Raises :class:`~trafficpulse.app.errors.EventNotFoundError` for an unknown
        event and :class:`~trafficpulse.app.errors.InvalidTransitionError` when the
        action is not legal from the current status -- the latter is a 409, not a
        422, because the request is well-formed and it is the *state* that refuses
        it. Re-sending a decision after somebody else has decided therefore fails
        loudly instead of silently overwriting their call.
        """

        _, manifest = self._events.locate(event_id)
        current = self._store.status(event_id)
        action = request.action
        if not can_transition(current, action):
            raise InvalidTransitionError(
                f"cannot {action.value!r} an event that is {current.value!r}"
            )

        resulting = next_status(current, action)
        at = datetime.now(UTC)
        note = request.note.strip() if request.note else None
        reason = request.reason.strip() if request.reason else None
        entry = ReviewEntry(
            entry_id=self._entry_id(event_id, action, at),
            event_id=event_id,
            action=action,
            status_before=current,
            status_after=resulting,
            reviewer=request.reviewer,
            at=at,
            note=note or None,
            reason=reason or None,
        )
        self._store.append(entry)
        return ReviewResponse(
            case=self._store.case(
                event_id, evidence_package_id=manifest.evidence_package_id
            ),
            history=self._store.history(event_id),
        )

    def statuses_for(self, event_ids: Iterable[str]) -> dict[str, ReviewStatus]:
        """Review statuses for many events, for badging a list."""

        return self._store.statuses(event_ids)

    @staticmethod
    def _entry_id(event_id: str, action: ReviewAction, at: datetime) -> str:
        """A content-derived id for one journal entry.

        Follows the project's existing identity convention (a SHA-256 over the
        identity-bearing fields, as ``event_id`` uses) rather than a random uuid, so
        an entry's id is reproducible from the entry itself.
        """

        material = f"{event_id}|{action.value}|{at.isoformat()}"
        return "rev-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class EvidenceService:
    """Serves an event's evidence: the manifest, its artifacts, and its package.

    The read side of the H14 rendering engine, and the place the two halves of the
    evidence record are joined. The manifest comes from the **write-once**
    ``EventStore`` exactly as the run persisted it; rendered artifacts come from the
    **append-only** :class:`~trafficpulse.persistence.RenderedArtifactStore`; the
    served manifest is composed from both at read time. Nothing here writes to, or
    could write to, a persisted event or manifest.

    Both rendering stores are optional. Without them the service behaves exactly as
    it did before H14 -- manifests are served verbatim and artifact requests report a
    clean absence -- which is what keeps every pre-H14 repository working.
    """

    def __init__(
        self,
        event_service: EventService,
        rendered: RenderedArtifactStore | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._events = event_service
        self._rendered = rendered
        self._artifacts = artifacts

    def get(self, event_id: str) -> EvidenceManifest:
        """The served manifest: as persisted, with any rendered artifacts merged in.

        An event with nothing rendered returns the persisted manifest unchanged.
        """

        manifest = self._events.locate(event_id)[1]
        if self._rendered is None:
            return manifest
        return merge_rendered_artifacts(manifest, self._rendered.artifacts(event_id))

    def artifact(self, event_id: str, kind: ArtifactKind) -> tuple[bytes, str]:
        """One rendered artifact's bytes + media type, or raise a clean 404.

        The integrity check is not decoration: an artifact whose stored bytes no
        longer hash to the reference the manifest serves is *not* the evidence the
        manifest describes, so it is reported missing rather than served under a
        claim the system can no longer make.
        """

        manifest = self.get(event_id)
        reference = rendered_artifact_for(manifest, kind)
        if reference is None or self._artifacts is None:
            raise ArtifactNotFoundError(
                f"no rendered {kind.value!r} artifact for event {event_id!r}"
            )
        data = self._artifacts.read(reference.locator)
        if data is None or not self._artifacts.verify(reference):
            raise ArtifactNotFoundError(
                f"the stored {kind.value!r} artifact for event {event_id!r} is "
                "missing or does not match its recorded hash"
            )
        return data, reference.media_type or "application/octet-stream"

    def package(self, event_id: str) -> tuple[bytes, str]:
        """One event's downloadable evidence package: ZIP bytes + filename.

        Built on demand and never stored: it is a deterministic function of the
        event, the served manifest, and artifacts that are already content-addressed,
        so caching the archive would duplicate bytes the store already holds. A
        package is always produced -- an event with nothing rendered yields the
        metadata-only archive, which is still the complete record of what the system
        concluded.
        """

        event, _ = self._events.locate(event_id)
        manifest = self.get(event_id)
        artifacts = self._artifacts if self._artifacts is not None else ArtifactStore(Path())
        data = build_evidence_package(event=event, manifest=manifest, artifacts=artifacts)
        return data, evidence_package_filename(event_id)


# --- metrics -------------------------------------------------------------------
class MetricsService:
    """Aggregates job counts and surfaces the latest H6 ``EngineMetrics``."""

    def __init__(self, job_store: JobStore) -> None:
        self._jobs = job_store

    def snapshot(self) -> MetricsResponse:
        jobs = self._jobs.jobs()
        by_status = {status: 0 for status in JobStatus}
        events_total = 0
        for job in jobs:
            by_status[job.status] += 1
            events_total += len(job.event_ids)

        latest = None
        for job in reversed(jobs):  # most recent job carrying metrics
            metrics = job.metrics()
            if metrics is not None:
                latest = metrics
                break

        return MetricsResponse(
            jobs_total=len(jobs),
            jobs_pending=by_status[JobStatus.PENDING],
            jobs_running=by_status[JobStatus.RUNNING],
            jobs_succeeded=by_status[JobStatus.SUCCEEDED],
            jobs_failed=by_status[JobStatus.FAILED],
            jobs_cancelled=by_status[JobStatus.CANCELLED],
            events_total=events_total,
            latest=latest,
        )


def _opening_job(jobs: Sequence[JobRecord]) -> JobRecord | None:
    """The run a client should open for a video: its analysis, not its last attempt.

    The most recent **succeeded** run wins, because that is the one that has events,
    evidence, and possibly an overlay -- the thing "reopen this video" means. Only
    when no run ever succeeded does the most recent run of any status stand in, so a
    video whose processing failed still reports why instead of looking unprocessed.
    Jobs arrive in insertion order, so "most recent" is the last match.
    """

    succeeded = [job for job in jobs if job.status is JobStatus.SUCCEEDED]
    if succeeded:
        return succeeded[-1]
    return jobs[-1] if jobs else None


def _sorted_videos(summaries: list[VideoSummary], sort: VideoSort) -> list[VideoSummary]:
    """Deterministically order video summaries (video_id is the final tie-break).

    A video with no recorded upload instant cannot be placed on the time axis, so
    it is sorted to the end of *either* direction rather than being given a
    substitute date -- the ordering says "unknown", which is what is true.
    """

    if sort in (VideoSort.FILENAME_ASC, VideoSort.FILENAME_DESC):
        return sorted(
            summaries,
            key=lambda s: (s.filename.lower(), s.video_id),
            reverse=sort is VideoSort.FILENAME_DESC,
        )
    descending = sort is VideoSort.UPLOADED_AT_DESC
    return sorted(
        summaries,
        key=lambda s: (
            s.uploaded_at is None,
            _sort_instant(s.uploaded_at, descending=descending),
            s.video_id,
        ),
    )


def _sort_instant(value: datetime | None, *, descending: bool) -> float:
    """A comparable key for an optional instant, negated for descending order.

    Negating the timestamp (rather than reversing the whole sort) keeps the
    "unknown last" flag and the ``video_id`` tie-break ascending in both
    directions, so the ordering stays stable and unknowns never lead the page.
    """

    if value is None:
        return 0.0
    seconds = value.timestamp()
    return -seconds if descending else seconds


def _sorted_summaries(
    summaries: list[EventSummary], sort: EventSort
) -> list[EventSummary]:
    """Deterministically order summaries (event_id is always the final tie-break)."""

    if sort in (EventSort.EVENT_ID_ASC, EventSort.EVENT_ID_DESC):
        return sorted(
            summaries,
            key=lambda s: s.event_id,
            reverse=sort is EventSort.EVENT_ID_DESC,
        )
    return sorted(
        summaries,
        key=lambda s: (s.trigger_at, s.event_id),
        reverse=sort is EventSort.TRIGGER_AT_DESC,
    )


# The public AppError base is imported for handlers/tests; re-export keeps the
# service module the single import surface for the application error taxonomy.
__all__ = [
    "VideoService",
    "VideoLibraryService",
    "SceneService",
    "ProcessingService",
    "EventService",
    "EvidenceService",
    "MetricsService",
    "AppError",
]
