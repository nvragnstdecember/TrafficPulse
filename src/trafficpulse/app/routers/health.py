"""Health/readiness endpoint (H7A)."""

from __future__ import annotations

from fastapi import APIRouter

from ... import __version__
from ..dependencies import ProviderDep
from ..models import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Service health",
    description="Liveness probe returning the service status, package version, "
    "and whether an inference backend is available.",
)
def health(provider: ProviderDep) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, engine=provider.describe())
