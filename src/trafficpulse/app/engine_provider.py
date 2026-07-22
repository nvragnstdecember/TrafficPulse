"""The engine-provider seam: how the app obtains an H6 engine (H7A).

:class:`EngineProvider` is the single injection point between the application
services and the H6 inference engine. A service asks a provider to ``create`` an
:class:`~trafficpulse.engine.InferenceEngine` for a scene + rule set; it never
constructs one itself and never names a detector/tracker backend. This is what
lets the whole API be tested with a stub-detector engine while production builds
the real RT-DETR backend -- the services are identical on both paths.

:class:`RealEngineProvider` is the production composition root: it delegates to
the H6 :func:`~trafficpulse.engine.build_engine` (which lazily loads RT-DETR),
injecting ``time.perf_counter`` so live jobs report real wall-clock FPS. Building
it imports no ML framework; only ``create`` -- called when a job is actually
submitted -- loads the model, and its typed failures (missing extra/checkpoint)
propagate for the service to translate into a clean HTTP error.

Tests supply their own provider (a stub-detector engine); it lives in the test
suite, not here, so the shipped package carries no test scaffolding.
"""

from __future__ import annotations

import time
from typing import Protocol

from ..contracts import SceneConfig
from ..engine import EngineConfig, InferenceEngine, RuleConfig, build_engine
from .config import AppConfig
from .errors import EngineUnavailableError


class EngineProvider(Protocol):
    """Creates a configured H6 engine for one job; the injectable backend seam."""

    def create(self, *, scene: SceneConfig, rules: tuple[RuleConfig, ...]) -> InferenceEngine:
        """Build an engine for ``scene`` running ``rules``.

        May raise the composed layers' typed errors (scene/rule validation, or a
        backend/checkpoint failure); the caller translates them to HTTP errors.
        """
        ...

    def describe(self) -> str:
        """A short readiness token for the health endpoint (e.g. ``"ready"``)."""
        ...


class RealEngineProvider:
    """Builds the real RT-DETR engine via the H6 composition root."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def create(self, *, scene: SceneConfig, rules: tuple[RuleConfig, ...]) -> InferenceEngine:
        if self._config.inference is None:
            raise EngineUnavailableError(
                "no inference backend is configured; set AppConfig.inference to "
                "process video with the real RT-DETR engine"
            )
        return self._build_real_engine(scene, rules)  # pragma: no cover - needs RT-DETR

    def _build_real_engine(  # pragma: no cover - requires the optional RT-DETR backend
        self, scene: SceneConfig, rules: tuple[RuleConfig, ...]
    ) -> InferenceEngine:
        """Build the real RT-DETR engine via H6 (loads torch; excluded from the
        framework-free test suite, whose provider is a stub -- see the module
        docstring). Its typed ``DetectorError`` on a missing extra/checkpoint is
        translated to a 503 by the processing service, which *is* tested with a
        stub that raises the same error without importing any ML framework.

        When a ``no_helmet`` rule is present the engine also needs a
        ``HelmetClassifier``; it is built here from ``AppConfig.helmet_classifier``
        (lazily loading transformers) and injected. If that config is absent the
        classifier stays ``None`` and the engine's rule registry fails the
        ``no_helmet`` rule fast -- a clean configuration error, never a silent
        miss."""

        from ..classifier import ZeroShotHelmetClassifier

        engine_config = EngineConfig(rules=rules, inference=self._config.inference)
        classifier = (
            ZeroShotHelmetClassifier(self._config.helmet_classifier)
            if self._config.helmet_classifier is not None
            else None
        )
        # build_engine also mints an EventStore; the processing service owns
        # persistence (one store rooted at runs_dir), so only the engine is kept.
        engine, _store = build_engine(
            scene=scene,
            config=engine_config,
            classifier=classifier,
            output_root=self._config.runs_dir,
            perf=time.perf_counter,  # real wall-clock FPS for live job metrics
        )
        return engine

    def describe(self) -> str:
        return "ready" if self._config.inference is not None else "unconfigured"
