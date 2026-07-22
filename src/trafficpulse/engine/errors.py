"""Error taxonomy for the real-time inference engine (H6).

Every engine failure raises a subclass of :class:`EngineError`, so callers
depend on this module's stable types and never on an implementation detail.
Failures from composed layers keep their own taxonomies (``VideoIngestionError``,
``DetectorError``, ``TrackerError``, ``PersistenceError``, pipeline
``SceneConfigurationError``): the engine wires those layers, it does not
re-wrap their errors.
"""


class EngineError(Exception):
    """Base class for every real-time inference-engine error."""


class EngineConfigurationError(EngineError):
    """An engine configuration is semantically invalid (cross-field rule).

    Field-level bounds surface as pydantic ``ValidationError`` at construction,
    matching the rest of the runtime configuration layer; this typed error is
    for rules a single field cannot express (e.g. a configured no-helmet rule
    with no injected classifier).
    """


class UnsupportedRuleError(EngineError):
    """A configured rule names a violation with no shipped reasoning slice.

    The engine composes the *existing* rule implementations (wrong-way,
    illegal-stopping, no-helmet). Violation types that have contracts but no
    reasoner yet (red-light jumping, triple riding, speeding) fail loudly here
    rather than silently confirming nothing; they slot in through the same
    registry once their reasoners ship.
    """


class FrameSourceError(EngineError):
    """A frame source misbehaved at the engine boundary.

    Covers non-monotonic timestamps/frame indices from a live adapter source --
    the engine refuses to fabricate order for a stream that has none. File
    sources keep the ingestion taxonomy (``VideoIngestionError``)."""
