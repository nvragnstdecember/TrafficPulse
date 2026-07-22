"""Deterministic on-disk layout for the helmet training data (H1).

Defines the canonical directory structure under a gitignored data root and can
create it idempotently. It **downloads nothing** -- it only prepares and describes
where data will live, and resolves the paths ingestion checks for.

```
<root>/                     (default: <repo>/data, gitignored via /data/)
  raw/<dataset_id>/         extracted source datasets (populated by a later unit)
  interim/                  unified intermediate format (H2)
  processed/                training-ready splits/tensors (H2/H3)
  splits/                   train/val/test assignment files (H3)
```

The layout is a pure function of its ``root``; construction touches no disk, so
tests point it at a temporary directory and nothing is ever written under the
tracked tree.
"""

from __future__ import annotations

from pathlib import Path

from .models import DatasetEntry

# Repository-relative default; ``/data/`` is gitignored so nothing here is tracked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = _REPO_ROOT / "data"

_SUBDIRS = ("raw", "interim", "processed", "splits")


class DatasetLayout:
    """The canonical training-data directory layout rooted at ``root``."""

    def __init__(self, root: Path | str = DEFAULT_DATA_ROOT) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def raw(self) -> Path:
        return self._root / "raw"

    @property
    def interim(self) -> Path:
        return self._root / "interim"

    @property
    def processed(self) -> Path:
        return self._root / "processed"

    @property
    def splits(self) -> Path:
        return self._root / "splits"

    def raw_dir(self, dataset_id: str) -> Path:
        """The directory a given dataset's extracted source files live under."""

        return self.raw / dataset_id

    def required_targets(self, entry: DatasetEntry) -> tuple[Path, ...]:
        """Absolute paths ingestion expects to exist for ``entry`` (relative -> raw_dir)."""

        base = self.raw_dir(entry.dataset_id)
        return tuple(base / rel for rel in entry.required_paths)

    def create(self) -> None:
        """Create the four top-level subdirectories (idempotent). Downloads nothing."""

        for name in _SUBDIRS:
            (self._root / name).mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        """Whether every top-level subdirectory is present."""

        return all((self._root / name).is_dir() for name in _SUBDIRS)
