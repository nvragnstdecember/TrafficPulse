"""Typed application configuration for the HTTP API layer (H7A).

``AppConfig`` is the single, frozen, strict source of every deployment knob the
application layer needs: where uploads and run outputs live, which scene governs
reasoning, the bind host/port, the upload guard-rails, and the default rule set
and (optional) real inference backend the composition root builds. It mirrors
the frozen+strict posture of ``DetectorConfig`` / ``EngineConfig`` and the domain
contracts -- nothing here is a mutable global.

No hardcoded absolute paths
---------------------------
Every path is relative or operator-supplied. :meth:`AppConfig.from_env` reads a
small set of ``TRAFFICPULSE_APP_*`` environment variables with portable relative
defaults, so the same code runs on any OS and in CI without editing. The scene is
optional at construction (a server can start before a scene is chosen); it is
required only when a processing job is actually submitted, and its absence then
fails as a clean HTTP error rather than at import.

Reuse, not duplication
----------------------
``default_rules`` and ``inference`` are the H6 :data:`~trafficpulse.engine.RuleConfig`
and :class:`~trafficpulse.engine.InferenceConfig` verbatim -- the application does
not invent a parallel configuration vocabulary for rules or the detector backend.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..engine import InferenceConfig, RuleConfig

# Container formats PyAV's bundled FFmpeg decodes on every platform; the upload
# validator additionally *opens* each file, so this is a fast-fail pre-filter,
# not the authority on readability.
DEFAULT_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
)
DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB
_DEFAULT_STORAGE = "trafficpulse-data"


class AppConfig(BaseModel):
    """Frozen, strict configuration for one running application instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_dir: Path
    """Root of application storage. Uploaded videos live under ``videos_dir`` and
    persisted run outputs under ``runs_dir`` (both derived), so one directory is
    the whole on-disk footprint."""

    scene_path: Path | None = None
    """Path to the governing ``SceneConfig`` (JSON or YAML). Optional at startup;
    required when a job is submitted (absence then surfaces as a clean 400)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    max_upload_bytes: int = Field(default=DEFAULT_MAX_UPLOAD_BYTES, ge=1)
    allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_EXTENSIONS

    default_rules: tuple[RuleConfig, ...] = ()
    """The rule set applied when a processing request supplies none (H6 configs)."""

    inference: InferenceConfig | None = None
    """The real RT-DETR backend the production engine provider builds. ``None``
    leaves the server able to serve every read endpoint and to run stub-injected
    jobs, while real processing reports the backend as unconfigured."""

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _normalise_extensions(cls, value: object) -> object:
        """Lower-case every extension and guarantee a leading dot."""

        if isinstance(value, str) or not isinstance(value, frozenset | set | list | tuple):
            return value  # let pydantic raise the type error
        return frozenset(
            ("." + ext.lstrip(".")).lower() for ext in value if ext and str(ext).strip()
        )

    @property
    def videos_dir(self) -> Path:
        """Where uploaded source videos are stored."""

        return self.storage_dir / "videos"

    @property
    def runs_dir(self) -> Path:
        """The ``EventStore`` root for persisted per-job events + manifests."""

        return self.storage_dir / "runs"

    def is_supported_extension(self, suffix: str) -> bool:
        """Whether ``suffix`` (e.g. ``".mp4"``) is an accepted container."""

        return ("." + suffix.lstrip(".")).lower() in self.allowed_extensions

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Build a config from ``TRAFFICPULSE_APP_*`` variables (portable defaults).

        Recognised: ``TRAFFICPULSE_APP_STORAGE`` (default ``trafficpulse-data``),
        ``TRAFFICPULSE_APP_SCENE``, ``TRAFFICPULSE_APP_HOST``,
        ``TRAFFICPULSE_APP_PORT``, ``TRAFFICPULSE_APP_MAX_UPLOAD_BYTES``. Unknown
        variables are ignored; malformed numeric values raise ``ValueError`` via
        pydantic. No absolute path is ever assumed.
        """

        env = os.environ if environ is None else environ
        scene = env.get("TRAFFICPULSE_APP_SCENE")
        fields: dict[str, object] = {
            "storage_dir": Path(env.get("TRAFFICPULSE_APP_STORAGE", _DEFAULT_STORAGE)),
            "scene_path": Path(scene) if scene else None,
        }
        if "TRAFFICPULSE_APP_HOST" in env:
            fields["host"] = env["TRAFFICPULSE_APP_HOST"]
        if "TRAFFICPULSE_APP_PORT" in env:
            fields["port"] = int(env["TRAFFICPULSE_APP_PORT"])
        if "TRAFFICPULSE_APP_MAX_UPLOAD_BYTES" in env:
            fields["max_upload_bytes"] = int(env["TRAFFICPULSE_APP_MAX_UPLOAD_BYTES"])
        return cls(**fields)


def load_scene(path: Path) -> object:
    """Load a ``SceneConfig`` from a JSON or YAML file (thin, reused parser).

    Kept deliberately tiny -- it is presentation/wiring, not logic: JSON via
    pydantic (a base dependency), YAML via the lazily-imported dev extra. Returns
    the validated ``SceneConfig``; typed as ``object`` here only to avoid importing
    the contract at module scope (the services annotate it precisely).
    """

    from ..contracts import SceneConfig

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return SceneConfig.model_validate(yaml.safe_load(text))
    return SceneConfig.model_validate_json(text)
