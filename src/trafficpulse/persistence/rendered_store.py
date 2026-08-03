"""Append-only journal of rendered evidence artifacts (H14).

The sidecar that lets the evidence pipeline gain *rendered* artifacts without any
persisted ``EvidenceManifest`` ever changing. Direct sibling of
:class:`~trafficpulse.persistence.review_store.ReviewStore`, same posture and for
the same reason.

Why this is *not* stored in the manifest
-----------------------------------------
``EventStore`` is **write-once**: re-persisting differing content under an existing
id raises :class:`~trafficpulse.persistence.errors.EventConflictError`, which is
what makes a replay byte-identical and a persisted manifest trustworthy. Rendering
happens *after* the manifest is written and can happen again later (a re-render, a
new output format), so folding rendered references into the manifest would force
one of two bad outcomes: relax write-once, or refuse to ever render. Instead the
references live beside the manifest and are merged **at read time** by
``EvidenceService``. **Nothing in this module can alter an inference result.**

Layout
------
```
<root>/rendered/<event_id>.jsonl    # one ArtifactReference per line, append-only
```
Keyed by ``event_id`` alone, deliberately **not** nested under a run -- the same
choice the review journal makes, and for the same reason: a rendered frame is about
an *event*, so it stays addressable even when run indexing is unavailable, and it
can never collide with the write-once run directory.

JSON Lines rather than a growing JSON array: appending is a single ``open(...,
"a")`` with no read-modify-write, so recording a new artifact cannot corrupt or
lose earlier ones.

Duplicates are collapsed on read, not refused on write
------------------------------------------------------
Artifact locators are content addresses (see
:mod:`trafficpulse.evidence.artifacts`), so re-recording an unchanged artifact is
both expected and harmless: :meth:`RenderedArtifactStore.artifacts` de-duplicates
by ``(kind, locator)``, keeping first-seen order. That makes repeated rendering
idempotent from the reader's point of view without the writer having to read the
file first.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference
from .errors import CorruptRecordError
from .store import DEFAULT_RUN_ROOT

_RENDERED_DIR = "rendered"


class RenderedArtifactStore:
    """Append-only per-event journal of rendered-artifact references.

    Constructed with the same runtime root as the event store; writes only under
    its own ``rendered/`` subtree, so it can never touch a write-once event record.
    """

    def __init__(self, root: Path | str = DEFAULT_RUN_ROOT) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The runtime root this journal writes under."""

        return self._root

    def journal_path(self, event_id: str) -> Path:
        """The append-only journal file for one event's rendered artifacts."""

        return self._root / _RENDERED_DIR / f"{event_id}.jsonl"

    def record(
        self, event_id: str, references: Iterable[ArtifactReference]
    ) -> tuple[ArtifactReference, ...]:
        """Append rendered-artifact references for one event; return the full set.

        Appends every supplied reference and returns the event's de-duplicated
        artifacts (including any recorded earlier). Recording nothing is a no-op
        that still reports what is already known, so a caller need not special-case
        an event whose render produced no frames.
        """

        pending = tuple(references)
        if pending:
            path = self.journal_path(event_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(
                reference.model_dump_json() + "\n" for reference in pending
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
        return self.artifacts(event_id)

    def artifacts(self, event_id: str) -> tuple[ArtifactReference, ...]:
        """Every rendered artifact recorded for an event, de-duplicated, in order.

        An event with no journal has no rendered artifacts -- the honest answer for
        a repository written before H14, and the reason those repositories keep
        serving evidence unchanged.
        """

        path = self.journal_path(event_id)
        if not path.is_file():
            return ()
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - unreadable file is environmental
            raise CorruptRecordError(
                f"cannot read rendered-artifact journal {path}"
            ) from exc

        seen: set[tuple[str, str]] = set()
        artifacts: list[ArtifactReference] = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                reference = ArtifactReference.model_validate_json(line)
            except ValidationError as exc:
                raise CorruptRecordError(
                    f"line {number} of {path} is not a valid ArtifactReference"
                ) from exc
            key = (reference.kind.value, reference.locator)
            if key in seen:
                continue
            seen.add(key)
            artifacts.append(reference)
        return tuple(artifacts)

    def artifact(
        self, event_id: str, kind: ArtifactKind
    ) -> ArtifactReference | None:
        """The first rendered artifact of ``kind`` for an event, or ``None``."""

        for reference in self.artifacts(event_id):
            if reference.kind is kind:
                return reference
        return None

    def rendered_event_ids(self) -> frozenset[str]:
        """Every event id that has rendered artifacts (one directory listing)."""

        directory = self._root / _RENDERED_DIR
        if not directory.is_dir():
            return frozenset()
        return frozenset(path.stem for path in directory.glob("*.jsonl"))
