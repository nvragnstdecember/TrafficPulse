"""Creation, isolation and disposal of live camera sessions.

The application-layer owner of live mode, and the counterpart of
:class:`~trafficpulse.app.services.ProcessingService` for a stream that has no
file: it resolves the scene, decides the rule set through the existing capability
probe, asks the injected :class:`~trafficpulse.app.engine_provider.EngineProvider`
for an engine, and hands back a :class:`~trafficpulse.app.live.session.LiveSession`.

Isolation is structural, not a convention
------------------------------------------
Each session gets its **own** engine from the provider, and an engine owns its own
tracker, observers and accumulated history. Two live cameras therefore cannot share
a track id, contaminate each other's associations, or see each other's events --
not because this class is careful, but because there is no object between them to
contaminate. The manager holds only a name-to-session dictionary.

Why the session cap is low
--------------------------
Inference is the process's scarce resource and it is not parallel: a second
concurrent session halves the frame rate of the first and doubles the resident
model memory. The cap makes that a stated limit with a clear refusal rather than a
mystery slowdown.

Readiness is answered before a camera is opened
-----------------------------------------------
:meth:`readiness` reports whether live monitoring could start at all -- is an
inference backend configured, is a drawing backend installed -- so the UI can say
"this deployment cannot monitor live" before asking a person for camera permission,
instead of after.
"""

from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass

from ...contracts import SceneConfig
from ...detector.errors import DetectorError
from ...engine import AnalysisConfig, InferenceEngine
from ...engine.errors import EngineConfigurationError, UnsupportedRuleError
from ...pipeline.errors import SceneConfigurationError
from ...scenes import CALIBRATION_SOURCE_ANALYST
from ..config import AppConfig
from ..engine_provider import EngineProvider
from ..errors import EngineUnavailableError
from ..posture import no_helmet_rule_available
from ..services import SceneService
from .config import LiveConfig
from .errors import LiveCapacityError, LiveProtocolError, LiveUnavailableError
from .scene import LiveRulePlan, live_rule_plan, provisional_live_scene
from .session import LiveSession

_logger = logging.getLogger("trafficpulse.live")


@dataclass(frozen=True)
class LiveReadiness:
    """Whether this deployment can monitor a live camera, and why not if it cannot."""

    ready: bool
    detail: str
    active_sessions: int
    max_sessions: int
    inference_configured: bool
    drawing_backend_available: bool
    helmet_classifier_configured: bool


@dataclass(frozen=True)
class LiveSessionSummary:
    """A live session as an operator sees it from outside the socket."""

    session_id: str
    camera_id: str
    width: int
    height: int
    scene_calibrated: bool
    frames_processed: int
    uptime_seconds: float


def _drawing_backend_available() -> bool:
    """Whether Pillow is importable (live mode decodes and draws with it)."""

    from importlib.util import find_spec

    return find_spec("PIL") is not None


class LiveSessionManager:
    """Owns every live session in this process (see the module docstring)."""

    def __init__(
        self,
        *,
        config: AppConfig,
        live_config: LiveConfig,
        provider: EngineProvider,
        scenes: SceneService,
    ) -> None:
        self._config = config
        self._live_config = live_config
        self._provider = provider
        self._scenes = scenes
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    @property
    def live_config(self) -> LiveConfig:
        return self._live_config

    # --- readiness -----------------------------------------------------------
    def readiness(self) -> LiveReadiness:
        """Can this deployment monitor a live camera right now, and if not, why not?"""

        inference = self._config.inference is not None
        drawing = _drawing_backend_available()
        with self._lock:
            active = len(self._sessions)
        if not inference:
            detail = (
                "No inference backend is configured, so there is nothing to run on a "
                "camera stream. Live monitoring is unavailable; uploaded-video "
                "processing reports the same 503."
            )
        elif not drawing:
            detail = (
                "No drawing backend is installed, so camera frames cannot be decoded "
                "or annotated. Install the optional 'overlay' extra "
                "(pip install 'trafficpulse[overlay]')."
            )
        elif active >= self._live_config.max_sessions:
            detail = (
                f"All {self._live_config.max_sessions} live session slot(s) are in "
                "use. Stop an existing session before starting another."
            )
        else:
            detail = (
                "Live monitoring can start. Inference runs as fast as this hardware "
                "allows and is far slower than camera capture; the session reports "
                "its measured rate rather than the camera's."
            )
        return LiveReadiness(
            ready=inference and drawing and active < self._live_config.max_sessions,
            detail=detail,
            active_sessions=active,
            max_sessions=self._live_config.max_sessions,
            inference_configured=inference,
            drawing_backend_available=drawing,
            helmet_classifier_configured=self._config.helmet_classifier is not None,
        )

    # --- lifecycle -----------------------------------------------------------
    def create(
        self, *, width: int, height: int, scene_hash: str | None = None
    ) -> LiveSession:
        """Open a session for a camera producing ``width`` x ``height`` frames.

        ``scene_hash`` names a stored scene revision to reason through -- the way a
        fixed, analyst-calibrated camera enables the violations that need geometry.
        It must be a revision authored against this camera's frame size; a mismatch
        is refused rather than silently reasoned about, because a polygon measured
        on a different frame lands somewhere else in this one, and every rule scoped
        to it would then quietly confirm nothing.

        Without one the session gets the provisional scene: the frame, and nothing
        claimed. See :mod:`trafficpulse.app.live.scene`.

        Raises:
            LiveProtocolError: the declared frame size is not usable.
            LiveCapacityError: the session cap is reached.
            LiveUnavailableError: no inference or drawing backend.
            SceneNotFoundError: ``scene_hash`` names no stored revision.
        """

        if width <= 0 or height <= 0:
            raise LiveProtocolError(
                f"camera frame size must be positive, got {width}x{height}"
            )
        if width * height > self._live_config.max_frame_pixels:
            raise LiveProtocolError(
                f"camera frame {width}x{height} exceeds this deployment's "
                f"{self._live_config.max_frame_pixels}-pixel live limit"
            )
        readiness = self.readiness()
        if not readiness.inference_configured or not readiness.drawing_backend_available:
            raise LiveUnavailableError(readiness.detail)

        session_id = f"live-{secrets.token_hex(8)}"
        scene = self._resolve_scene(
            scene_hash, width=width, height=height, session_id=session_id
        )
        plan = live_rule_plan(
            scene, no_helmet_available=no_helmet_rule_available(self._config)
        )
        if not plan.rules:
            raise LiveUnavailableError(
                "no shipped rule can run against this camera's scene, so a live "
                "session would observe without being able to reason. Calibrate the "
                "camera, or configure a helmet classifier, to enable one."
            )
        engine = self._build_engine(scene, plan)
        session = LiveSession(
            session_id=session_id,
            engine=engine,
            scene=scene,
            plan=plan,
            config=self._live_config,
            width=width,
            height=height,
        )
        with self._lock:
            # Re-checked under the lock: readiness() above is advisory, and two
            # sockets can pass it concurrently. This is the authority.
            if len(self._sessions) >= self._live_config.max_sessions:
                raise LiveCapacityError(readiness.detail)
            self._sessions[session_id] = session
        _logger.info(
            "live session %s opened: %dx%d, %d rule(s) [%s]",
            session_id,
            width,
            height,
            len(plan.rules),
            ", ".join(str(kind) for kind in plan.supported) or "none",
        )
        return session

    def get(self, session_id: str) -> LiveSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> bool:
        """Close and forget one session. Returns whether it existed."""

        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        _logger.info(
            "live session %s closed after %d processed frame(s)",
            session_id,
            session.stats().frames_processed,
        )
        return True

    def close_all(self) -> int:
        """Close every session (process shutdown). Returns how many were closed."""

        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
        return len(sessions)

    def summaries(self) -> tuple[LiveSessionSummary, ...]:
        """Every open session, for the operator-facing listing."""

        with self._lock:
            sessions = list(self._sessions.values())
        return tuple(
            LiveSessionSummary(
                session_id=session.session_id,
                camera_id=session.camera_id,
                width=session.width,
                height=session.height,
                # A provisional scene claims no direction and no zone but the frame;
                # anything reached by hash was authored by somebody.
                scene_calibrated=session.scene.calibration.source
                == CALIBRATION_SOURCE_ANALYST,
                frames_processed=session.stats().frames_processed,
                uptime_seconds=session.stats().uptime_seconds,
            )
            for session in sorted(sessions, key=lambda s: s.session_id)
        )

    # --- internals -----------------------------------------------------------
    def _resolve_scene(
        self, scene_hash: str | None, *, width: int, height: int, session_id: str
    ) -> SceneConfig:
        if scene_hash is None:
            return provisional_live_scene(
                width=width,
                height=height,
                camera_id=f"cam-{session_id}",
                scene_id=f"scene-{session_id}",
            )
        scene = self._scenes.get(scene_hash)  # raises SceneNotFoundError
        frame = scene.frame
        if (frame.reference_width, frame.reference_height) != (width, height):
            raise LiveProtocolError(
                f"scene {scene_hash[:12]} was authored for "
                f"{frame.reference_width}x{frame.reference_height} but this camera "
                f"produces {width}x{height}; its geometry would land in the wrong "
                "place, so it is refused rather than applied"
            )
        return scene

    def _build_engine(self, scene: SceneConfig, plan: LiveRulePlan) -> InferenceEngine:
        """Build this session's engine, translating the composed layers' failures.

        The same translation :class:`~trafficpulse.app.services.ProcessingService`
        performs for a job, for the same reason: the live channel must report a
        missing checkpoint as "the backend is unavailable", never as an internal
        error, and never as a session that opens and then silently sees nothing.
        """

        try:
            return self._provider.create(
                scene=scene,
                rules=plan.rules,
                # Analysis mirrors the job path exactly: helmet classification runs as
                # perception when the deployment declared it and no no-helmet rule is
                # already carrying its own helmet observer.
                analysis=self._analysis_for(plan),
            )
        except EngineUnavailableError as exc:
            raise LiveUnavailableError(exc.message) from exc
        except DetectorError as exc:
            raise LiveUnavailableError(
                f"the inference backend is unavailable: {exc}"
            ) from exc
        except (
            SceneConfigurationError,
            EngineConfigurationError,
            UnsupportedRuleError,
            ValueError,
        ) as exc:
            raise LiveUnavailableError(
                f"a live engine cannot be built for this camera: {exc}"
            ) from exc

    def _analysis_for(self, plan: LiveRulePlan) -> tuple[AnalysisConfig, ...]:
        from ...engine import NoHelmetRuleConfig

        declared = self._config.helmet_analysis
        if declared is None:
            return ()
        if any(isinstance(rule, NoHelmetRuleConfig) for rule in plan.rules):
            return ()
        return (declared,)


__all__ = ["LiveReadiness", "LiveSessionManager", "LiveSessionSummary"]
