"""Error taxonomy for the classifier-integration boundary (P4-U2).

Every classifier-integration failure raises a subclass of
:class:`HelmetClassifierError`, so callers depend only on this package's error
types -- never on an ML framework's exceptions leaking through the seam. This
mirrors the detector boundary's :class:`~trafficpulse.detector.errors.DetectorError`
contract and is part of what makes a backend swap (P4-U3) bounded: a caller's
``except HelmetClassifierError`` stays correct across every backend.

Only the base class lives here for now. Concrete subclasses land in the unit that
first *raises* them -- adapter/validation errors with the P4-U4 adapter, backend
dependency/artifact errors with the P4-U3 real backend -- rather than being
declared speculatively ahead of any raiser (architecture-review: no speculative
architecture).
"""


class HelmetClassifierError(Exception):
    """Base class for all helmet-classifier integration errors."""
