"""Generic metrics storage (H4A).

A model-agnostic store for scalar training metrics: per-name history of
``(epoch, step, value)`` points, epoch summaries, and a deterministic JSON dump
(series sorted by name) so two runs that recorded the same values serialise
byte-identically regardless of recording order across names.

Values are validated at the door: a NaN/inf is **rejected**, never silently
stored — a poisoned metric would otherwise propagate into best-checkpoint
selection and quietly corrupt it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from ..errors import InvalidMetricNameError, InvalidMetricValueError, MetricNotFoundError
from ..models import NonEmptyStr, _Model

# Lowercase segments separated by '.', '_', '/' or '-' (e.g. "val/no_helmet_ap").
METRIC_NAME_RE = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*$")


def validate_metric_name(name: str) -> str:
    """Return ``name`` if sanctioned, else raise :class:`InvalidMetricNameError`."""

    if not METRIC_NAME_RE.match(name):
        raise InvalidMetricNameError(
            f"metric name {name!r} does not match {METRIC_NAME_RE.pattern!r}"
        )
    return name


class MetricPoint(_Model):
    """One recorded scalar: which epoch/step it belongs to and its finite value."""

    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    value: float

    @model_validator(mode="after")
    def _finite(self) -> Self:
        if not math.isfinite(self.value):
            raise InvalidMetricValueError(
                f"metric value must be finite, got {self.value!r}"
            )
        return self


class MetricSeries(_Model):
    """One metric's full history, in recording order."""

    name: NonEmptyStr
    points: tuple[MetricPoint, ...]


class MetricsDump(_Model):
    """The serialisable form of a whole store (series sorted by name)."""

    series: tuple[MetricSeries, ...]


class MetricsStore:
    """Mutable in-memory metrics accumulator with deterministic serialisation."""

    def __init__(self) -> None:
        self._series: dict[str, list[MetricPoint]] = {}

    def record(self, name: str, value: float, *, epoch: int, step: int) -> MetricPoint:
        """Validate and append one scalar; return the stored point."""

        validate_metric_name(name)
        point = MetricPoint(epoch=epoch, step=step, value=value)
        self._series.setdefault(name, []).append(point)
        return point

    def names(self) -> tuple[str, ...]:
        """All recorded metric names, sorted."""

        return tuple(sorted(self._series))

    def history(self, name: str) -> tuple[MetricPoint, ...]:
        """The full history of one metric, in recording order."""

        if name not in self._series:
            raise MetricNotFoundError(f"no metric {name!r} recorded (known: {list(self.names())})")
        return tuple(self._series[name])

    def latest(self, name: str) -> MetricPoint:
        """The most recently recorded point of one metric."""

        return self.history(name)[-1]

    def epoch_summary(self, epoch: int) -> dict[str, float]:
        """The last recorded value per metric within ``epoch`` (sorted by name)."""

        summary: dict[str, float] = {}
        for name in self.names():
            in_epoch = [p for p in self._series[name] if p.epoch == epoch]
            if in_epoch:
                summary[name] = in_epoch[-1].value
        return summary

    def to_dump(self) -> MetricsDump:
        """The deterministic serialisable form (series sorted by name)."""

        return MetricsDump(
            series=tuple(
                MetricSeries(name=name, points=tuple(self._series[name]))
                for name in self.names()
            )
        )

    def to_json(self) -> str:
        return self.to_dump().model_dump_json(indent=2)

    def save(self, path: Path) -> Path:
        """Write the dump as JSON to ``path`` (parents created); return ``path``."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dump(cls, dump: MetricsDump) -> MetricsStore:
        store = cls()
        for series in dump.series:
            store._series[series.name] = list(series.points)
        return store

    @classmethod
    def load_or_new(cls, path: Path) -> MetricsStore:
        """Rebuild a store from ``path``, or return an empty one if it is absent."""

        if not path.is_file():
            return cls()
        return cls.from_dump(MetricsDump.model_validate_json(path.read_text(encoding="utf-8")))
