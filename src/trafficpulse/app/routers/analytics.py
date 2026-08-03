"""Analytics endpoint: the whole dashboard in one response (H15).

A single aggregation endpoint rather than a family of widget endpoints. Every
section is derived from one read of the registries, so splitting them would both
multiply round-trips and let two sections of the same dashboard describe different
instants.

The aggregation is server-side by design: the client renders the summary and
derives no repository figures of its own.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import AnalyticsServiceDep
from ..models import AnalyticsSummary

router = APIRouter(tags=["analytics"])


@router.get(
    "/api/analytics/summary",
    response_model=AnalyticsSummary,
    summary="Repository analytics summary",
    description="Return the complete dashboard payload: repository overview, "
    "processing statistics, violation breakdown, evidence coverage, review "
    "progress, repository health, recent activity, and the latest run's engine "
    "metrics. Aggregated from the in-memory registries and directory listings -- no "
    "event or manifest file is opened, so the cost is O(videos + jobs) regardless "
    "of how many violations the repository holds. Every dated figure comes from a "
    "wall-clock instant the system recorded (upload, run lifecycle, review action); "
    "media timestamps are never presented as calendar dates.",
)
def get_analytics_summary(analytics: AnalyticsServiceDep) -> AnalyticsSummary:
    return analytics.summary()
