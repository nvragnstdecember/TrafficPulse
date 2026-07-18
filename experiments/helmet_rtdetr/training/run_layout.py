"""Deterministic experiment-run directory layout (H4A).

```
<root>/<experiment-name>/
    config.json       the full ExperimentConfig as written at first begin()
    seed_plan.json    the derived SeedPlan
    state.json        the final TrainingState (written by end())
    checkpoints/      CheckpointManager's metadata + index
    logs/             structured event log (events.jsonl)
    metrics/          metrics.json (updated every epoch, so resume recovers it)
    artifacts/        reserved for H4B outputs (plots, exports)
```

Default root
------------
:data:`DEFAULT_RUNS_ROOT` is ``<repo>/runs/helmet_rtdetr`` — deliberately **not**
``experiments/helmet_rtdetr/runs/``: the repo's ``.gitignore`` anchors ``/runs/``
at the top level only, so a runs directory nested inside ``experiments/`` would be
committable and one careless ``git add`` away from checking training artifacts
into history. The top-level ``runs/`` tree is already ignored; run outputs belong
there. Tests always pass an explicit temporary root.

Construction touches no disk; :meth:`create` is idempotent.
"""

from __future__ import annotations

from pathlib import Path

# See the module docstring for why this lives under the gitignored top-level runs/.
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs" / "helmet_rtdetr"

_SUBDIRS = ("checkpoints", "logs", "metrics", "artifacts")


class RunLayout:
    """The canonical run directory for one named experiment under one root."""

    def __init__(self, root: Path, name: str) -> None:
        self._root = root
        self._name = name

    @property
    def run_dir(self) -> Path:
        return self._root / self._name

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def artifacts(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def seed_plan_path(self) -> Path:
        return self.run_dir / "seed_plan.json"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / "metrics.json"

    def is_initialized(self) -> bool:
        """Whether an experiment has already begun here (its config was written)."""

        return self.config_path.is_file()

    def create(self) -> None:
        """Create the run directory tree (idempotent; writes no files)."""

        for name in _SUBDIRS:
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
