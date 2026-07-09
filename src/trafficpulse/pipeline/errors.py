"""Error taxonomy for the vertical-slice orchestration boundary (P1-U10).

Orchestration owns only a *stable, useful* error boundary: the one failure that
is genuinely the pipeline's own -- resolving which single lane / legal direction
governs a run from a :class:`~trafficpulse.contracts.SceneConfig`. Everything else
is deliberately **not** wrapped: detector errors
(:class:`~trafficpulse.detector.errors.InvalidFrameError`,
:class:`~trafficpulse.detector.errors.MalformedDetectorOutputError`), tracker
errors (:class:`~trafficpulse.tracking.errors.NonMonotonicFrameError`, ...), and
the ``ValueError`` :func:`~trafficpulse.rules.wrong_way.wrong_way_parameters`
raises for a missing ``wrong_way`` block are stable lower-level TrafficPulse
errors and propagate unchanged, so the orchestrator adds no generic catch-all that
would erase a useful diagnostic.
"""


class PipelineError(Exception):
    """Base class for all orchestration-level errors."""


class SceneConfigurationError(PipelineError):
    """The scene cannot be resolved to a single governing lane / legal direction.

    Raised when the single-lane vertical slice (P1-U10) cannot pick exactly one
    legal direction: the scene declares no legal direction, declares more than one
    with no ``direction_id`` selecting between them, names a ``direction_id`` that
    does not exist, or the chosen legal direction carries no zone/lane id.
    """
