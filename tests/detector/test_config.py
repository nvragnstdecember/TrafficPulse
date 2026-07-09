"""Configuration-validation tests for ``DetectorConfig`` (P1-U6).

Covers construction, defaults, the non-empty label-map rule, closed-set label
values, the ``[0, 1]`` score threshold, immutability (frozen), and strictness
(``extra='forbid'``). Configuration validation is intentionally surfaced as
pydantic's ``ValidationError`` (not a detector-package error), matching the U5
scene contract.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from trafficpulse.contracts import ModelRef, ObjectClass
from trafficpulse.detector import DetectorConfig


def test_minimal_config_constructs_with_defaults() -> None:
    cfg = DetectorConfig(label_map={"motorcycle": ObjectClass.MOTORCYCLE})
    assert cfg.label_map["motorcycle"] is ObjectClass.MOTORCYCLE
    assert cfg.score_threshold == 0.0
    assert cfg.source_model is None


def test_full_config_constructs() -> None:
    cfg = DetectorConfig(
        label_map={"car": ObjectClass.CAR, "person": ObjectClass.PERSON},
        score_threshold=0.5,
        source_model=ModelRef(name="rt-detr", version="r50-apache"),
    )
    assert cfg.score_threshold == 0.5
    assert cfg.source_model is not None
    assert cfg.source_model.name == "rt-detr"
    assert cfg.label_map["person"] is ObjectClass.PERSON


def test_empty_label_map_rejected() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(label_map={})


def test_empty_label_key_rejected() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(label_map={"": ObjectClass.CAR})


def test_label_map_value_must_be_object_class() -> None:
    bad: dict[str, Any] = {"car": "not-an-object-class"}
    with pytest.raises(ValidationError):
        DetectorConfig(label_map=bad)


def test_score_threshold_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(label_map={"car": ObjectClass.CAR}, score_threshold=1.5)


def test_score_threshold_below_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(label_map={"car": ObjectClass.CAR}, score_threshold=-0.1)


def test_score_threshold_boundaries_accepted() -> None:
    label_map = {"car": ObjectClass.CAR}
    assert DetectorConfig(label_map=label_map, score_threshold=0.0).score_threshold == 0.0
    assert DetectorConfig(label_map=label_map, score_threshold=1.0).score_threshold == 1.0


def test_config_is_frozen() -> None:
    cfg = DetectorConfig(label_map={"car": ObjectClass.CAR})
    with pytest.raises(ValidationError):
        cfg.score_threshold = 0.9  # type: ignore[misc]


def test_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DetectorConfig(label_map={"car": ObjectClass.CAR}, weights_path="model.pt")  # type: ignore[call-arg]
