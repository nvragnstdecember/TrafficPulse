"""Content-addressed scene repository (H12).

Stores validated :class:`~trafficpulse.contracts.SceneConfig`\\ s, addressed by
their own deterministic ``scene_config_hash``:

```
<root>/scenes/<scene_config_hash>.json     # the SceneConfig, verbatim
```

Why content-addressed, and why that makes it write-once
--------------------------------------------------------
Scenes are *edited* -- an analyst redraws a zone, raises a dwell threshold -- so
the obvious design is a mutable record per scene. That design is wrong here, and
the reason is already stamped into every event this system has ever produced:
:attr:`~trafficpulse.contracts.ConfirmedEvent.scene_config_hash` records the
scene a violation was reasoned under. Mutating a scene in place would leave every
event that referenced it pointing at content that no longer exists -- the record
would still *claim* provenance while the thing it named had silently changed.

Addressing a scene by its hash makes an edit a **new revision** instead of an
overwrite. The old revision stays exactly as it was, so an event's
``scene_config_hash`` resolves -- for the first time -- to the precise geometry
and thresholds that produced it. Provenance stops being a claim and becomes
something you can fetch.

This is the same posture as
:class:`~trafficpulse.persistence.store.EventStore` (write-once, content-derived
identity) rather than a third storage philosophy, and it is the same trick used
for ``video_id`` and ``event_id``: identity *is* content.

Idempotence falls out
---------------------
Because :func:`~trafficpulse.scenes.builder.build_scene` is deterministic, saving
an unchanged drawing produces the same hash and therefore the same path with
byte-identical content -- a no-op, not a new revision and not a conflict. Only a
real change to the geometry or thresholds mints a new address.

Garbage is not collected
------------------------
Superseded revisions are never deleted. A scene that no video is bound to may
still be the scene some historical event was reasoned under, so "unreferenced" is
not the same as "unneeded". Scenes are small JSON documents; keeping them is the
cheap side of the trade.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..contracts import SceneConfig
from ..contracts.scene import scene_config_hash
from .errors import CorruptRecordError
from .store import DEFAULT_RUN_ROOT

_SCENES_DIR = "scenes"


class SceneStore:
    """Write-once, content-addressed store of validated scene configurations.

    Holds no mutable state -- a thin filesystem adapter, like its ``EventStore``
    and ``ReviewStore`` siblings.
    """

    def __init__(self, root: Path | str = DEFAULT_RUN_ROOT) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, scene_hash: str) -> Path:
        """The file holding one scene revision (may not exist)."""

        return self._root / _SCENES_DIR / f"{scene_hash}.json"

    # --- writing ------------------------------------------------------------
    def put(self, scene: SceneConfig) -> str:
        """Store a scene revision; return the hash that addresses it.

        Idempotent: re-storing identical content rewrites the same bytes to the
        same path. There is deliberately no conflict check -- unlike an event id,
        which is a *digest of selected identity fields* and so could in principle
        be minted for differing content, a scene's address is a digest of its
        **entire** content. Equal address means equal bytes, by construction.
        """

        digest = scene_config_hash(scene)
        path = self.path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(scene.model_dump_json(), encoding="utf-8")
        return digest

    # --- reading ------------------------------------------------------------
    def get(self, scene_hash: str) -> SceneConfig | None:
        """Load one scene revision, or ``None`` when this repository has no such scene.

        Absence is not an error: an event may carry a ``scene_config_hash`` from a
        run whose scene was never stored (anything processed before H12, or under
        a file-configured scene), and the caller reports that honestly rather than
        failing.

        Raises:
            CorruptRecordError: the file exists but is unreadable or is not a
                valid ``SceneConfig`` -- a real fault, distinct from absence.
        """

        path = self.path(scene_hash)
        if not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - unreadable file is environmental
            raise CorruptRecordError(f"cannot read scene {path}") from exc
        try:
            return SceneConfig.model_validate_json(raw)
        except ValidationError as exc:
            raise CorruptRecordError(f"scene {path} is not a valid SceneConfig") from exc

    def hashes(self) -> tuple[str, ...]:
        """Every stored revision's address, sorted -- from a directory listing.

        Cheap by construction: a scene's *filename* is its hash, so enumerating
        the repository opens no file and deserialises nothing. The same trick
        H10 uses for the event index.
        """

        directory = self._root / _SCENES_DIR
        if not directory.is_dir():
            return ()
        return tuple(sorted(path.stem for path in directory.glob("*.json")))

    def contains(self, scene_hash: str) -> bool:
        return self.path(scene_hash).is_file()
