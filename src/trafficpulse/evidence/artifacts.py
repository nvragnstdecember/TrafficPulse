"""Content-addressed storage for rendered evidence artifacts (H14).

Where a rendered artifact's bytes live, and the one place that turns bytes into an
:class:`~trafficpulse.contracts.evidence.ArtifactReference`. The storage layout
*is* the integrity hash (architecture-review §25): an artifact is written under its
own SHA-256, so the locator, the file path, and the ``sha256`` field are three
views of the same fact and cannot drift apart.

Why content addressing (and not a path keyed by event)
------------------------------------------------------
Rendering is deterministic -- the same frame drawn with the same scene by the same
code produces the same bytes -- so identical content naturally collapses onto one
file. Two events whose evidence windows overlap on a frame share storage instead of
duplicating it, and a re-render of an unchanged event is a hash computation and a
no-op write rather than a second copy. It also makes the store immune to relocation
in a way the ingestion frame ids are not: nothing in a locator depends on where the
source video sits on disk.

Write-once, like everything else durable here
---------------------------------------------
A locator addresses its own content, so a differing write to an existing locator is
a hash collision, not a legitimate update. Writes are therefore idempotent: identical
bytes are a no-op. The store never deletes and never overwrites, which is what lets a
manifest reference an artifact and know the bytes behind it cannot have changed.

Boundary
--------
Bytes in, reference out. This module renders nothing, decodes nothing, and imports
no drawing or media dependency -- it is a filesystem adapter over the frozen
``ArtifactReference`` contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference

#: Media types this store knows how to name on disk. A media type outside this map
#: is still storable -- it simply gets no extension, which keeps the store honest
#: rather than guessing a wrong one.
_EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "application/json": ".json",
    "application/zip": ".zip",
    "video/mp4": ".mp4",
}

#: Width of the directory shard prefix. Two hex characters give 256 buckets, which
#: keeps any single directory listable for the repository sizes this system targets.
_SHARD = 2


def artifact_sha256(data: bytes) -> str:
    """The SHA-256 hex digest of ``data`` (the artifact's content address)."""

    return hashlib.sha256(data).hexdigest()


def artifact_locator(digest: str, media_type: str) -> str:
    """The deterministic relative locator for a digest + media type.

    ``artifacts/<first two hex chars>/<digest><ext>`` -- relative, POSIX-separated,
    and stable across platforms and storage roots, so a locator persisted on one
    machine resolves on another.
    """

    extension = _EXTENSIONS.get(media_type, "")
    return f"artifacts/{digest[:_SHARD]}/{digest}{extension}"


class ArtifactStore:
    """Content-addressed filesystem store for rendered artifacts.

    Constructed with the storage ``root`` the relative locators are resolved
    against. Holds no state beyond that path.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        # Cached (file count, total bytes). Computed lazily on first ask and then
        # maintained incrementally by `put`, because this store is the only writer
        # of artifact bytes. See `usage`.
        self._usage: tuple[int, int] | None = None

    @property
    def root(self) -> Path:
        """The directory relative locators are resolved against."""

        return self._root

    def path_for(self, locator: str) -> Path:
        """The absolute path a relative locator resolves to.

        Rejects any locator that escapes the root. Locators reach this method from
        persisted records and, through the API, from client-supplied ids, so the
        check is deliberately belt-and-braces:

        * the locator is parsed with **POSIX** semantics regardless of host OS,
          because ``Path("/etc/passwd").is_absolute()`` is ``False`` on Windows and a
          repository written on one platform is readable on the other;
        * a Windows drive/UNC anchor is rejected explicitly, since POSIX parsing
          treats ``C:/x`` as relative;
        * the resolved result must still sit under the root, which catches anything
          the syntactic checks miss (symlinks included).
        """

        posix = PurePosixPath(locator.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise ValueError(f"unsafe artifact locator {locator!r}")
        if PureWindowsPath(locator).anchor:
            raise ValueError(f"unsafe artifact locator {locator!r}")

        root = self._root.resolve()
        candidate = (root / Path(*posix.parts)).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"unsafe artifact locator {locator!r}")
        return candidate

    def contains(self, locator: str) -> bool:
        """Whether the artifact behind ``locator`` is present on disk."""

        try:
            return self.path_for(locator).is_file()
        except ValueError:
            return False

    def read(self, locator: str) -> bytes | None:
        """The artifact's bytes, or ``None`` when it is not stored."""

        try:
            path = self.path_for(locator)
        except ValueError:
            return None
        if not path.is_file():
            return None
        return path.read_bytes()

    def put(
        self, data: bytes, *, kind: ArtifactKind, media_type: str
    ) -> ArtifactReference:
        """Store ``data`` under its content address; return its typed reference.

        Idempotent: re-storing identical bytes rewrites nothing and returns an equal
        reference, which is what makes a re-render of an unchanged event free. The
        returned reference always carries the hash -- unlike the pre-H14 frame
        references, an artifact this store produced is *always* integrity-checkable.
        """

        digest = artifact_sha256(data)
        locator = artifact_locator(digest, media_type)
        path = self.path_for(locator)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a sibling temp file and replace, so a crash mid-write can
            # never leave a truncated file sitting at a valid content address.
            temp = path.with_name(f"{path.name}.partial")
            temp.write_bytes(data)
            temp.replace(path)
            if self._usage is not None:
                count, total = self._usage
                self._usage = (count + 1, total + len(data))
        return ArtifactReference(
            kind=kind, locator=locator, sha256=digest, media_type=media_type
        )

    def usage(self) -> tuple[int, int]:
        """``(file count, total bytes)`` held by this store.

        Walking the store on every request is what the H16 investigation found the
        analytics dashboard doing -- an O(artifacts) filesystem scan every 30
        seconds, growing with every rendered frame. Since this class is the **only**
        writer of artifact bytes, the figure can be maintained instead of
        rediscovered: it is computed once per process (a single cold scan) and then
        updated in place by :meth:`put`.

        The cache is therefore per-process and rebuilt after a restart, which is
        exactly right -- a fresh process should measure what is actually on disk
        rather than trust a number it did not compute.

        A file that vanishes mid-scan is skipped rather than raising: a dashboard
        must not fail over a concurrently-removed artifact.
        """

        if self._usage is None:
            count = total = 0
            if self._root.is_dir():
                for path in self._root.rglob("*"):
                    try:
                        if path.is_file():
                            count += 1
                            total += path.stat().st_size
                    except OSError:  # pragma: no cover - race with removal
                        continue
            self._usage = (count, total)
        return self._usage

    def verify(self, reference: ArtifactReference) -> bool:
        """Whether the stored bytes match the reference's declared hash.

        A reference with no ``sha256`` cannot be verified and is reported as
        unverified (``False``) rather than trusted.
        """

        if reference.sha256 is None:
            return False
        data = self.read(reference.locator)
        if data is None:
            return False
        return artifact_sha256(data) == reference.sha256
