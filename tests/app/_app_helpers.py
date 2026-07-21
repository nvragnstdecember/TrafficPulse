"""Shared builders for the H7A application-API tests.

Provides a stub :class:`EngineProvider` (a real H6 ``InferenceEngine`` over a
scripted stub detector -- no torch, no RT-DETR, no GPU) and a factory that wires
a ``TestClient`` over a fully-configured app with the synchronous executor, so
the whole request/job lifecycle is deterministic. Reuses the pipeline fixture
helpers (path-shimmed by this directory's conftest). Uniquely named
(``_app_helpers``) for pytest's prepend import mode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from _pipeline_helpers import DETECTOR_CONFIG, NORTH_DIRECTION_ID, SCENE
from _slice_fixtures import scripted_down_detector, write_wrong_way_clip
from fastapi.testclient import TestClient

from trafficpulse.app import AppConfig, SynchronousJobExecutor, create_app
from trafficpulse.app.registry import JobExecutor
from trafficpulse.contracts import SceneConfig
from trafficpulse.detector import DetectorConfig
from trafficpulse.detector.interface import Detector
from trafficpulse.engine import (
    EngineConfig,
    EngineMetrics,
    InferenceEngine,
    RuleConfig,
    WrongWayRuleConfig,
)
from trafficpulse.tracking import IouTracker

# A scene path the app can load: reuse the committed example scene.
EXAMPLE_SCENE_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example-scene.yaml"
)
DEFAULT_RULES: tuple[RuleConfig, ...] = (WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),)


class StubEngineProvider:
    """An :class:`EngineProvider` yielding a real engine over a stub detector.

    ``detector_factory`` builds a fresh scripted detector per job (so replays are
    independent). ``create`` runs the genuine H6 engine construction -- including
    rule validation against the scene -- so an invalid scene/rule combination
    raises exactly as the real backend would.
    """

    def __init__(
        self,
        detector_factory: Callable[[], Detector] = scripted_down_detector,
        *,
        detector_config: DetectorConfig = DETECTOR_CONFIG,
        readiness: str = "ready",
    ) -> None:
        self._detector_factory = detector_factory
        self._detector_config = detector_config
        self._readiness = readiness

    def create(
        self, *, scene: SceneConfig, rules: tuple[RuleConfig, ...]
    ) -> InferenceEngine:
        return InferenceEngine(
            scene=scene,
            detector=self._detector_factory(),
            tracker=IouTracker(),
            detector_config=self._detector_config,
            config=EngineConfig(rules=rules),
        )

    def describe(self) -> str:
        return self._readiness


def make_config(
    storage: Path,
    *,
    scene_path: Path | None = EXAMPLE_SCENE_PATH,
    default_rules: tuple[RuleConfig, ...] = DEFAULT_RULES,
    max_upload_bytes: int = 512 * 1024 * 1024,
) -> AppConfig:
    return AppConfig(
        storage_dir=storage,
        scene_path=scene_path,
        default_rules=default_rules,
        max_upload_bytes=max_upload_bytes,
    )


def make_client(
    storage: Path,
    *,
    provider: StubEngineProvider | None = None,
    executor: JobExecutor | None = None,
    config: AppConfig | None = None,
) -> TestClient:
    """A ``TestClient`` over a fully-wired app (stub provider, sync executor)."""

    app = create_app(
        config if config is not None else make_config(storage),
        engine_provider=provider if provider is not None else StubEngineProvider(),
        executor=executor if executor is not None else SynchronousJobExecutor(),
    )
    return TestClient(app, raise_server_exceptions=False)


def upload_wrong_way_video(client: TestClient, tmp_path: Path, *, name: str = "clip.mp4") -> str:
    """Write the wrong-way clip, upload it, and return its ``video_id``."""

    clip = write_wrong_way_clip(tmp_path / name)
    data = clip.read_bytes()
    response = client.post(
        "/api/video/upload", files={"file": (name, data, "video/mp4")}
    )
    assert response.status_code == 201, response.text
    video_id: str = response.json()["video_id"]
    return video_id


def make_metrics(**overrides: object) -> EngineMetrics:
    """A zeroed :class:`EngineMetrics` with chosen fields overridden (for unit tests)."""

    base: dict[str, object] = {
        "frames_read": 0,
        "frames_skipped_stride": 0,
        "frames_skipped_fps": 0,
        "frames_dropped_backpressure": 0,
        "frames_admitted": 0,
        "frames_processed": 0,
        "batches_processed": 0,
        "detections": 0,
        "track_states": 0,
        "events_confirmed": 0,
        "queue_peak": 0,
        "media_fps": None,
        "wall_fps": None,
        "latencies": {},
        "memory_bytes_current": None,
        "memory_bytes_peak": None,
        "gpu_memory_bytes_current": None,
        "gpu_memory_bytes_peak": None,
    }
    base.update(overrides)
    return EngineMetrics(**base)  # type: ignore[arg-type]


class FakeEngine:
    """A minimal object exposing a ``metrics`` snapshot for status unit tests."""

    def __init__(self, metrics: EngineMetrics) -> None:
        self._metrics = metrics

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics


class RaisingDetector(Detector):
    """A detector that raises on inference -- to exercise the job-failure path."""

    def detect(self, frame: object) -> object:
        raise RuntimeError("boom: scripted detector failure")


class RaisingEngineProvider:
    """A provider whose ``create`` raises an unexpected error (500 path)."""

    def create(self, *, scene: object, rules: object) -> InferenceEngine:
        raise RuntimeError("boom: unexpected provider failure")

    def describe(self) -> str:
        return "ready"


class UnavailableEngineProvider:
    """A provider whose ``create`` raises a typed ``DetectorError`` (503 path).

    Exercises the service's ``DetectorError -> EngineUnavailableError`` mapping
    -- the same translation the real RT-DETR provider triggers on a missing
    extra/checkpoint -- without importing any ML framework, so the detector
    package's import-isolation invariant is preserved.
    """

    def create(self, *, scene: object, rules: object) -> InferenceEngine:
        from trafficpulse.detector.errors import DetectorError

        raise DetectorError("the inference backend is down")

    def describe(self) -> str:
        return "unconfigured"


__all__ = [
    "StubEngineProvider",
    "RaisingEngineProvider",
    "UnavailableEngineProvider",
    "RaisingDetector",
    "FakeEngine",
    "make_metrics",
    "make_config",
    "make_client",
    "upload_wrong_way_video",
    "EXAMPLE_SCENE_PATH",
    "DEFAULT_RULES",
    "SCENE",
    "NORTH_DIRECTION_ID",
]
