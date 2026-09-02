"""Shared builders for the live-camera tests.

Drives the **real** live stack -- the real ``LiveSessionManager``, the real
``LiveSession``, a real H6 ``InferenceEngine``, the real IoU tracker, the real rider
associator and the real overlay renderer -- over a scripted stub detector and a
scripted stub helmet classifier. Only the two model backends are stubs, exactly as
the H7A application tests already do it, so a live test that passes is evidence
about the live pipeline and not about a mock of it.

Frames are encoded as real JPEGs of the declared size, because the session decodes
what it is sent: an encoder change or a size guard regression has to be able to fail
one of these tests.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from _helmet_fixtures import (
    HEIGHT,
    NO_HELMET,
    WIDTH,
    helmet_detector_config,
    scripted_helmet_classifier,
    scripted_rider_detector,
)
from _triple_fixtures import scripted_rider_count_detector
from fastapi.testclient import TestClient

from trafficpulse.app import AppConfig, SynchronousJobExecutor, create_app
from trafficpulse.app.live import LiveConfig
from trafficpulse.classifier import ZeroShotHelmetConfig
from trafficpulse.classifier.interface import HelmetClassifier
from trafficpulse.contracts import SceneConfig
from trafficpulse.contracts.enums import ObjectClass
from trafficpulse.detector import DetectorConfig
from trafficpulse.detector.interface import Detector
from trafficpulse.engine import (
    AnalysisConfig,
    EngineConfig,
    InferenceConfig,
    InferenceEngine,
    RuleConfig,
)
from trafficpulse.tracking import IouTracker

#: A detector backend declaration that is *present but never built*: the live
#: manager's readiness check asks whether one is configured, and the stub provider
#: is what actually creates engines -- so this only has to exist, and constructing
#: it imports no ML framework.
STUB_INFERENCE = InferenceConfig(
    checkpoint="stub-checkpoint",
    label_map={"motorbike": ObjectClass.MOTORCYCLE, "person": ObjectClass.PERSON},
    local_files_only=True,
)


#: A turban-capable classifier declaration, never built (see ``live_config``).
STUB_HELMET = ZeroShotHelmetConfig(checkpoint="stub-helmet-checkpoint", local_files_only=True)


def jpeg_frame(index: int, *, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """A real JPEG of the given size with enough structure to survive a crop gate.

    The pixels are deterministic but not uniform: the scripted detector ignores
    them, while the head-crop quality gate does not -- a flat frame would be
    rejected as blurred and no helmet reading would ever be produced, which would
    make a passing helmet test meaningless.
    """

    from PIL import Image

    ys, xs = np.mgrid[0:height, 0:width]
    pattern = ((xs * 7 + ys * 13 + index * 31) % 256).astype(np.int64)
    rgb = np.stack([pattern, (pattern * 3) % 256, 255 - pattern], axis=-1).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def frame_message(index: int, *, width: int = WIDTH, height: int = HEIGHT) -> dict[str, object]:
    """A ``frame`` message carrying frame ``index`` at 10 fps of capture time."""

    return {
        "type": "frame",
        "sequence": index,
        "capture_seconds": round(index * 0.1, 6),
        "data": base64.b64encode(jpeg_frame(index, width=width, height=height)).decode(),
    }


class LiveStubProvider:
    """An ``EngineProvider`` yielding a real engine over scripted stub backends.

    A **fresh** detector, tracker and classifier per ``create`` call, which is what
    makes the session-isolation test meaningful: if two sessions shared any of them,
    the test that asserts they do not would still pass against a shared instance.
    """

    def __init__(
        self,
        *,
        detector_factory: object = None,
        detector_config: DetectorConfig | None = None,
        classifier: HelmetClassifier | None = None,
        fail_on_create: Exception | None = None,
    ) -> None:
        self._detector_factory = detector_factory or (lambda: scripted_rider_detector(4000))
        self._detector_config = (
            detector_config if detector_config is not None else helmet_detector_config()
        )
        # Defaulted rather than left None: the deployment config these tests build
        # declares a helmet classifier, so the no_helmet rule is in every live rule
        # set and an engine built without one is refused -- correctly, and unhelpfully
        # for a test whose subject is something else entirely.
        self._classifier = (
            classifier if classifier is not None else scripted_helmet_classifier(NO_HELMET)
        )
        self._fail_on_create = fail_on_create
        self.created = 0

    def create(
        self,
        *,
        scene: SceneConfig,
        rules: tuple[RuleConfig, ...],
        analysis: tuple[AnalysisConfig, ...] = (),
    ) -> InferenceEngine:
        if self._fail_on_create is not None:
            raise self._fail_on_create
        self.created += 1
        detector: Detector = self._detector_factory()  # type: ignore[operator]
        return InferenceEngine(
            scene=scene,
            detector=detector,
            tracker=IouTracker(),
            detector_config=self._detector_config,
            config=EngineConfig(rules=rules, analysis=analysis),
            classifier=self._classifier,
            capture_overlay=True,
        )

    def describe(self) -> str:
        return "ready"


def multi_rider_provider(riders: int = 3) -> LiveStubProvider:
    """A provider whose detector puts ``riders`` people on one motorcycle."""

    return LiveStubProvider(
        detector_factory=lambda: scripted_rider_count_detector(riders, 4000),
        classifier=scripted_helmet_classifier(NO_HELMET),
    )


def live_config(storage: Path, **overrides: object) -> AppConfig:
    """An app config mirroring ``serve.py``: a detector and a turban-capable classifier.

    Neither backend is ever built -- the stub provider creates every engine -- but
    both have to be *declared*, because the application decides what live mode may
    run from the configuration rather than from what a provider happens to inject.
    A turban-capable classifier config is what lets ``no_helmet`` past the capability
    guard, which is the posture ``serve.py`` deploys, so these tests exercise the
    rule set a real live session actually gets.
    """

    fields: dict[str, object] = {
        "storage_dir": storage,
        "scene_path": None,
        "inference": STUB_INFERENCE,
        "helmet_classifier": STUB_HELMET,
    }
    fields.update(overrides)
    return AppConfig(**fields)


def live_client(
    storage: Path,
    *,
    provider: LiveStubProvider | None = None,
    config: AppConfig | None = None,
    live: LiveConfig | None = None,
) -> TestClient:
    """A ``TestClient`` over an app whose live mode is fully wired."""

    app = create_app(
        config if config is not None else live_config(storage),
        engine_provider=provider if provider is not None else LiveStubProvider(),  # type: ignore[arg-type]
        executor=SynchronousJobExecutor(),
        live_config=live if live is not None else LiveConfig(),
    )
    return TestClient(app, raise_server_exceptions=False)


__all__ = [
    "HEIGHT",
    "WIDTH",
    "LiveStubProvider",
    "STUB_HELMET",
    "STUB_INFERENCE",
    "frame_message",
    "jpeg_frame",
    "live_client",
    "live_config",
    "multi_rider_provider",
    "scripted_helmet_classifier",
    "scripted_rider_detector",
]
