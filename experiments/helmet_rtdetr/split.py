"""Split configuration types (H3): split names and validated ratios.

Kept dependency-free of the manifest/builder so both can import them without a
cycle.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Self

from pydantic import model_validator

from .errors import InvalidRatioError
from .models import _Model

_RATIO_SUM_TOLERANCE = 1e-9


class SplitName(StrEnum):
    """The three splits, in canonical order."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# Canonical iteration order (train, val, test) used everywhere splits are listed.
SPLIT_ORDER: tuple[SplitName, ...] = (SplitName.TRAIN, SplitName.VAL, SplitName.TEST)


class SplitRatios(_Model):
    """Train/val/test fractions; each in ``[0, 1]`` and summing to 1.0."""

    train: float
    val: float
    test: float

    @model_validator(mode="after")
    def _valid(self) -> Self:
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise InvalidRatioError(f"ratio {name}={value!r} must be in [0, 1]")
        if not math.isclose(self.train + self.val + self.test, 1.0, abs_tol=_RATIO_SUM_TOLERANCE):
            raise InvalidRatioError(
                f"ratios must sum to 1.0, got {self.train + self.val + self.test!r}"
            )
        if self.train <= 0.0:
            raise InvalidRatioError("train ratio must be positive")
        return self

    def as_dict(self) -> dict[str, float]:
        """Ratios keyed by split name, in canonical order."""

        return {SplitName.TRAIN.value: self.train, SplitName.VAL.value: self.val,
                SplitName.TEST.value: self.test}

    def requested(self, split: SplitName) -> float:
        return {SplitName.TRAIN: self.train, SplitName.VAL: self.val, SplitName.TEST: self.test}[
            split
        ]
