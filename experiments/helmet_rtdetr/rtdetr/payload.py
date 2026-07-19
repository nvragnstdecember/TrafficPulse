"""Checkpoint weight payloads alongside the H4A metadata (H4B).

H4A's :class:`CheckpointManager` persists metadata-only JSON checkpoints and was
designed with a ``payload_path`` slot for exactly this unit. Rather than modify
H4A, payloads live **next to** the metadata by convention — ``ckpt-<id>.pt``
beside ``ckpt-<id>.json`` — and retention mirrors the manager's own:
:meth:`prune` deletes every payload whose id the manager no longer retains, so
weight files can never outlive (or orphan from) their metadata.

The payload is one torch-saved dict: model / optimizer / scheduler / scaler
state_dicts. Loading uses ``weights_only=False`` **only because the file is
self-written by this same pipeline in the local run directory** — it is never a
downloaded or third-party artifact; scheduler/scaler state_dicts contain plain
Python objects the safe loader rejects.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..errors import PayloadNotFoundError
from .model import require_torch

PAYLOAD_KEYS = ("model", "optimizer", "scheduler", "scaler")


class PayloadStore:
    """Saves/loads/prunes ``ckpt-<id>.pt`` weight payloads in a checkpoint dir."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def path_for(self, checkpoint_id: str) -> Path:
        return self._dir / f"ckpt-{checkpoint_id}.pt"

    def save(self, checkpoint_id: str, payload: dict[str, Any]) -> Path:
        """Persist one payload dict; return its path."""

        torch = require_torch()
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(checkpoint_id)
        torch.save(payload, path)
        return path

    def load(self, checkpoint_id: str) -> dict[str, Any]:
        """Load one payload dict, or raise :class:`PayloadNotFoundError`."""

        torch = require_torch()
        path = self.path_for(checkpoint_id)
        if not path.is_file():
            raise PayloadNotFoundError(
                f"checkpoint {checkpoint_id!r} has metadata but no weight payload at {path}; "
                "it may predate H4B or have been externally deleted"
            )
        # weights_only=False: self-written local file (see module docstring).
        loaded: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        return loaded

    def prune(self, retained_ids: Iterable[str]) -> tuple[str, ...]:
        """Delete payloads not in ``retained_ids``; return the deleted ids (sorted)."""

        retained = set(retained_ids)
        deleted: list[str] = []
        if not self._dir.is_dir():
            return ()
        for path in sorted(self._dir.glob("ckpt-*.pt")):
            checkpoint_id = path.stem.removeprefix("ckpt-")
            if checkpoint_id not in retained:
                path.unlink()
                deleted.append(checkpoint_id)
        return tuple(deleted)
