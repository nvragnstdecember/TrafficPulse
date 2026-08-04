"""Health/readiness endpoint (H7A; readiness detail added in H16)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ... import __version__
from ..dependencies import ConfigDep, ProviderDep, SceneServiceDep
from ..models import HealthResponse

router = APIRouter(tags=["health"])


def repository_status(storage_dir: Path) -> str:
    """Whether the storage root is present and writable.

    Reported, never fatal: a repository that cannot be written still serves every
    read endpoint, so turning this into a failed health check would pull a
    degraded-but-useful service out of rotation. The probe writes and removes a
    zero-byte file, because a directory can exist and still be read-only.
    """

    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        probe = storage_dir / ".healthcheck"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        return "unavailable"
    return "ready"


@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Service health",
    description="Liveness + readiness. `status` is 'ok' whenever the process is "
    "serving. The readiness fields distinguish that from being able to work: "
    "`repository` reports whether the storage root is writable, "
    "`inference_available` whether a processing job can actually run, and "
    "`scene_configured` whether a fallback scene exists for uncalibrated videos. "
    "The original three fields are unchanged for existing clients.",
)
def health(
    provider: ProviderDep, config: ConfigDep, scenes: SceneServiceDep
) -> HealthResponse:
    engine = provider.describe()
    return HealthResponse(
        status="ok",
        version=__version__,
        engine=engine,
        repository=repository_status(config.storage_dir),
        inference_available=engine == "ready",
        scene_configured=scenes.has_fallback,
    )
