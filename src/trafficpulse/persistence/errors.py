"""Error taxonomy for the event-persistence boundary (P1-U11).

Every persistence failure raises a subclass of :class:`PersistenceError`, so
callers depend only on this package's error types -- never on a filesystem
``OSError`` detail or a ``pydantic.ValidationError`` leaking out of a reload.
This mirrors ``detector/errors.py`` and ``pipeline/errors.py``: a small, stable,
useful boundary and nothing speculative.

The taxonomy is deliberately minimal for the first slice:

* :class:`RunNotFoundError` -- a load addressed a run directory that does not
  exist (a caller/addressing problem, distinct from corrupt content).
* :class:`CorruptRecordError` -- a persisted record on disk cannot be parsed back
  into its frozen U2 contract (missing sibling manifest, malformed JSON, or a
  ``ValidationError`` from a tampered/truncated file). Any originating
  ``pydantic.ValidationError`` / ``OSError`` is chained as ``__cause__``.
* :class:`EventConflictError` -- a write would silently overwrite an existing
  record with *differing* content under the same ``(run_id, event_id)``. This
  enforces ADR-004's proposed "manifests are append-only; no run silently
  overwrites another" without freezing any cross-run identity/dedup rule.
"""


class PersistenceError(Exception):
    """Base class for all event-persistence errors."""


class RunNotFoundError(PersistenceError):
    """A load addressed a run that has not been persisted.

    Raised by :meth:`~trafficpulse.persistence.store.EventStore.load` when the
    run directory for the requested ``run_id`` does not exist -- an addressing
    problem, kept distinct from a corrupt persisted record.
    """


class CorruptRecordError(PersistenceError):
    """A persisted record cannot be reloaded into its frozen U2 contract.

    Raised when a stored event/manifest file is missing its sibling, contains
    malformed JSON, or fails ``ConfirmedEvent`` / ``EvidenceManifest`` validation
    (e.g. a truncated or tampered file). The originating
    ``pydantic.ValidationError`` or ``OSError`` is chained as ``__cause__``.
    """


class EventConflictError(PersistenceError):
    """A write would silently overwrite differing content for the same event.

    Raised when persisting an ``event_id`` into a run that already holds a record
    with the *same* id but *different* bytes. Re-persisting byte-identical content
    (deterministic replay) is an idempotent no-op and never raises; only a genuine
    content conflict does -- honouring ADR-004's proposed append-only posture
    (no run silently overwrites another) without deciding cross-run dedup.
    """
