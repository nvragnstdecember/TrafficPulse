"""Metrics endpoint: aggregate job counts + the latest H6 engine metrics (H7A)."""

from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import MetricsServiceDep
from ..models import MetricsResponse

router = APIRouter(tags=["metrics"])


@router.get(
    "/api/metrics",
    response_model=MetricsResponse,
    summary="Engine + job metrics",
    description="Return aggregate processing-job counts and the most recent "
    "job's H6 EngineMetrics (frame counters, latencies, throughput) verbatim.",
)
def get_metrics(metrics: MetricsServiceDep) -> MetricsResponse:
    return metrics.snapshot()
