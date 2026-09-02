"""P4-U9 analysis logic: the accounting that decides what the comparison means.

The scoring itself is P4-U6-V's ``score.py``, unchanged and already covered. What is new in
P4-U9 is the accounting *around* it -- the turban report, the per-backend decision
assembly, and the crop selection that feeds both -- so that is what is tested here, on
fixtures rather than on models.

The turban tests matter most: they are the executable form of the rule that ``turban`` is
never folded into ``no_helmet``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.p4u9 import (  # noqa: E402
    decisions,
    turban_report,
)
from helmet_runtime_validation.rederive import recovered_rows  # noqa: E402


def posterior(**values: float) -> dict[str, float]:
    return dict(values)


# --- turban accounting -------------------------------------------------------
def test_turban_predictions_are_counted_not_converted() -> None:
    probabilities = {
        "a": posterior(helmet=0.1, no_helmet=0.2, turban=0.6, uncertain=0.1),
        "b": posterior(helmet=0.7, no_helmet=0.2, turban=0.05, uncertain=0.05),
    }
    report = turban_report(probabilities, ["a", "b"], ["helmet", "helmet"])
    assert report["turban_predictions"] == 1
    assert report["turban_share"] == pytest.approx(0.5)
    assert report["native_argmax_counts"] == {"helmet": 1, "turban": 1}
    assert "no_helmet" not in report["native_argmax_counts"], (
        "a turban prediction must never be counted as a no_helmet call"
    )


def test_the_annotated_labels_of_turban_predictions_are_reported_not_scored() -> None:
    """HELMET has no turban class, so the cross-tab is descriptive, never a score."""

    probabilities = {
        "a": posterior(helmet=0.1, no_helmet=0.1, turban=0.7, uncertain=0.1),
        "b": posterior(helmet=0.1, no_helmet=0.1, turban=0.7, uncertain=0.1),
        "c": posterior(helmet=0.1, no_helmet=0.1, turban=0.7, uncertain=0.1),
    }
    report = turban_report(probabilities, ["a", "b", "c"], ["helmet", "helmet", "no_helmet"])
    assert report["annotated_label_of_turban_predictions"] == {"helmet": 2, "no_helmet": 1}
    assert "correct" not in json.dumps(report), "no accuracy is claimed for turban"
    assert "never mapped to no_helmet" in report["note"]


def test_a_backend_with_no_turban_prompt_reports_zero_turban() -> None:
    probabilities = {"a": posterior(helmet=0.9, no_helmet=0.1)}
    report = turban_report(probabilities, ["a"], ["helmet"])
    assert report["turban_predictions"] == 0
    assert report["abstention_share_from_non_binary_vocabulary"] == 0.0


def test_turban_report_on_no_crops_reports_none_not_zero() -> None:
    report = turban_report({}, [], [])
    assert report["turban_share"] is None
    assert report["crops"] == 0


# --- decision assembly -------------------------------------------------------
def test_zeroshot_turban_abstains_natively_but_still_makes_a_forced_binary_call() -> None:
    """The two operating points differ exactly here, and both are reported."""

    probabilities = {"a": posterior(helmet=0.30, no_helmet=0.10, turban=0.55, uncertain=0.05)}
    made = decisions("zeroshot", probabilities, ["a"], None)
    assert made["native_labels"] == ["__abstain__"]
    assert made["forced_labels"] == ["helmet"], "renormalised over the binary prompts only"
    assert made["forced_scores"][0] == pytest.approx(0.75)


def test_a_binary_backend_abstains_below_its_frozen_threshold() -> None:
    probabilities = {"a": posterior(helmet=0.70, no_helmet=0.30)}
    assert decisions("resnet", probabilities, ["a"], 0.80)["native_labels"] == ["__abstain__"]
    assert decisions("resnet", probabilities, ["a"], 0.60)["native_labels"] == ["helmet"]
    assert decisions("resnet", probabilities, ["a"], None)["native_labels"] == ["helmet"]


def test_the_forced_view_never_abstains_for_any_backend() -> None:
    probabilities = {"a": posterior(helmet=0.51, no_helmet=0.49)}
    for backend in ("resnet", "deit", "zeroshot"):
        made = decisions(backend, probabilities, ["a"], 0.95)
        assert "__abstain__" not in made["forced_labels"]


def test_decisions_follow_the_requested_crop_order() -> None:
    probabilities = {
        "a": posterior(helmet=0.9, no_helmet=0.1),
        "b": posterior(helmet=0.1, no_helmet=0.9),
    }
    assert decisions("resnet", probabilities, ["b", "a"], None)["forced_labels"] == [
        "no_helmet",
        "helmet",
    ]


# --- crop selection ----------------------------------------------------------
def test_only_recovered_outcomes_become_crops(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"crop_id": "crop-b", "reason": "recovered"},
                {"crop_id": "crop-a", "reason": "recovered"},
                {"crop_id": "crop-c", "reason": "motorcycle_matched_but_no_rider_associated"},
                {"crop_id": "crop-d", "reason": "no_motorcycle_detected_at_iou_0.50"},
            )
        ),
        encoding="utf-8",
    )
    rows = recovered_rows(path)
    assert [row["crop_id"] for row in rows] == ["crop-a", "crop-b"], "sorted, recovered only"
