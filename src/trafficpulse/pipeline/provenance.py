"""Deterministic run-level model-provenance collection (P2-U1).

The pipeline composition boundary is the one place that sees both the detector's
``Detection.source_model`` and the tracker's ``TrackState.tracker`` -- the two
truthful ``ModelRef``s already stamped by their adapters (``DetectorConfig``
``source_model`` / ``TrackerConfig`` ``tracker``). This module's single job is to
turn the stream of per-frame provenance observed during a run into the stable,
run-level ``models`` tuple stamped onto every confirmed event.

Truthful, not fabricated
------------------------
Nothing here invents a ``ModelRef``: it only collects the refs the composition
root actually supplied and the adapters actually stamped. A stub run that supplies
no ``source_model`` / ``tracker`` yields an **empty** tuple, honestly, rather than
a placeholder ref.

Deterministic ordering + de-duplication
----------------------------------------
:func:`normalize_model_refs` de-duplicates by full ``ModelRef`` identity
(``name``, ``version``, ``weights_hash``) and sorts by
``(name, version, weights_hash or "")``. The result is therefore a pure function
of the *set* of distinct refs seen, independent of frame order, emission order, or
any set/hash iteration order -- identical runs produce byte-identical ``models``
tuples across instances and processes.

Provenance is pure metadata: it never enters a reasoning predicate (that stays the
reasoner's concern), so this collection cannot influence which events confirm.
"""

from collections.abc import Iterable

from ..contracts import ModelRef


def normalize_model_refs(refs: Iterable[ModelRef]) -> tuple[ModelRef, ...]:
    """De-duplicate and deterministically order a stream of ``ModelRef``s.

    De-duplication is by the full identity ``(name, version, weights_hash)`` (so a
    detector emitting the same ``source_model`` on every frame contributes exactly
    one entry); ordering is by ``(name, version, weights_hash or "")``. The output
    is independent of input order, making it stable across frames, instances, and
    processes.
    """

    unique: dict[tuple[str, str, str | None], ModelRef] = {
        (ref.name, ref.version, ref.weights_hash): ref for ref in refs
    }
    return tuple(
        sorted(unique.values(), key=lambda r: (r.name, r.version, r.weights_hash or ""))
    )
