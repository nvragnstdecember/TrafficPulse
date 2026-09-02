"""P4-U9: score the three backends on P4-U8's corrected runtime-crop population.

Every decision rule here is P4-U6-V's, **imported** rather than restated: the two operating
points, the abstention accounting, the threshold-selection rule, the log-space McNemar and
the frozen P4-U5 metric and bootstrap primitives. This module adds only what P4-U9's
protocol asks for on top of them, and writes to its own directory so nothing P4-U6-V
produced is touched.

What it adds
------------
* **A pre-declared sensitivity check** (PROTOCOL_P4U9 section 5.3): the same test pass also
  applies P4-U6-V's frozen thresholds, so a reader can see whether the conclusion depends
  on re-selecting them.
* **A shared-subset decomposition** (section 6.2). The 3,508 crops recovered under both
  conventions are byte-identical between the two runs, so the two experiments' predictions
  on them are the same by construction. The headline difference is therefore attributable
  entirely to the crops the correction added, and this module measures the added crops
  separately to show whether they are harder.
* **Turban accounting** (section 9): how often the only turban-capable backend uses that
  label, what it costs in coverage, and what the binary models put in its place.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_cnn_vit.stats import bootstrap_delta_macro_f1  # noqa: E402

from helmet_runtime_validation.analyse import (  # noqa: E402
    ABSTAIN,
    BACKENDS,
    evaluate,
    exact_mcnemar,
    forced_binary,
    load,
    manifest,
    native,
    select_threshold,
)

#: P4-U6-V's frozen operating points, quoted for the section 5.3 sensitivity check. They are
#: read from its results file rather than hard-coded, so they cannot drift out of sync.
P4U6V_DIR = REPO_ROOT / "runs" / "helmet_runtime_validation"


def inherited_thresholds() -> dict[str, float | None]:
    payload = json.loads(
        (P4U6V_DIR / "frozen_operating_points.json").read_text(encoding="utf-8")
    )
    return {name: entry["selected"] for name, entry in payload["selections"].items()}


def p4u6v_recovered_ids() -> set[str]:
    return {
        json.loads(line)["crop_id"]
        for line in (P4U6V_DIR / "crops.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def decisions(
    backend: str,
    probabilities: dict[str, dict[str, float]],
    crop_ids: Sequence[str],
    threshold: float | None,
) -> dict[str, list[Any]]:
    """Both operating points' decisions for one backend, over one crop list."""

    forced_labels, forced_scores, native_labels, native_scores = [], [], [], []
    for crop_id in crop_ids:
        row = probabilities[crop_id]
        label, score = forced_binary(row)
        forced_labels.append(label)
        forced_scores.append(score)
        label, score = native(backend, row, threshold)
        native_labels.append(label)
        native_scores.append(score)
    return {
        "forced_labels": forced_labels,
        "forced_scores": forced_scores,
        "native_labels": native_labels,
        "native_scores": native_scores,
    }


def _counts(labels: Sequence[str]) -> dict[str, int]:
    from collections import Counter

    return dict(sorted(Counter(labels).items()))


def turban_report(
    probabilities: dict[str, dict[str, float]],
    crop_ids: Sequence[str],
    truth: Sequence[str],
) -> dict[str, Any]:
    """Quantify the vocabulary asymmetry, without ever mapping ``turban`` anywhere.

    HELMET has no turban label, so a turban prediction has no ground truth of its own. What
    *can* be measured is how often the only turban-capable backend reaches for the label,
    what binary label those crops carry in the annotation, and what the abstention costs in
    coverage. None of that licenses treating turban as a violation.
    """

    from collections import Counter

    arg_max = [max(probabilities[c], key=lambda k: probabilities[c][k]) for c in crop_ids]
    by_label: dict[str, Counter[str]] = {}
    for prediction, actual in zip(arg_max, truth, strict=True):
        by_label.setdefault(prediction, Counter())[actual] += 1
    total = len(crop_ids)
    turban = sum(1 for p in arg_max if p == "turban")
    uncertain = sum(1 for p in arg_max if p == "uncertain")
    return {
        "crops": total,
        "native_argmax_counts": _counts(arg_max),
        "turban_predictions": turban,
        "turban_share": turban / total if total else None,
        "uncertain_predictions": uncertain,
        "uncertain_share": uncertain / total if total else None,
        "abstention_share_from_non_binary_vocabulary": (
            (turban + uncertain) / total if total else None
        ),
        "annotated_label_of_turban_predictions": dict(
            sorted(by_label.get("turban", Counter()).items())
        ),
        "note": (
            "HELMET carries no turban annotation. These crops' annotated labels are the "
            "binary driver states only; they are NOT evidence that a turban prediction is "
            "right or wrong, and turban is never mapped to no_helmet."
        ),
    }


def evaluate_backends(
    out_dir: Path,
    split: str,
    thresholds: dict[str, float | None],
    crop_ids: Sequence[str],
    truth: Sequence[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    results: dict[str, Any] = {}
    forced: dict[str, list[str]] = {}
    for backend in BACKENDS:
        probabilities = load(out_dir, backend, split)
        made = decisions(backend, probabilities, crop_ids, thresholds[backend])
        forced[backend] = made["forced_labels"]
        results[backend] = {
            "abstain_below": thresholds[backend],
            "forced_binary": evaluate(truth, made["forced_labels"], made["forced_scores"]),
            "native": evaluate(truth, made["native_labels"], made["native_scores"]),
            "native_label_counts": _counts(made["native_labels"]),
            "native_abstentions": sum(1 for label in made["native_labels"] if label == ABSTAIN),
        }
        if backend == "zeroshot":
            results[backend]["turban"] = turban_report(probabilities, crop_ids, truth)
    return results, forced


def pairwise(truth: Sequence[str], forced: dict[str, list[str]]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for index, first in enumerate(BACKENDS):
        for second in BACKENDS[index + 1 :]:
            a, b = forced[first], forced[second]
            interval = bootstrap_delta_macro_f1(truth, a, b, resamples=10000, seed=0)
            pairs[f"{first}_vs_{second}"] = {
                "mcnemar": exact_mcnemar(
                    [p == t for p, t in zip(a, truth, strict=True)],
                    [p == t for p, t in zip(b, truth, strict=True)],
                ),
                "bootstrap_delta_macro_f1": interval.model_dump(),
            }
    return pairs


def subset_report(
    out_dir: Path,
    split: str,
    thresholds: dict[str, float | None],
    rows: Sequence[dict[str, Any]],
    keep: Any,
    name: str,
) -> dict[str, Any]:
    """Re-evaluate one stratum of the corrected population (section 6.2)."""

    selected = [row for row in rows if keep(row)]
    if not selected:
        return {"stratum": name, "crops": 0, "backends": None}
    crop_ids = [str(row["crop_id"]) for row in selected]
    truth = [str(row["label"]) for row in selected]
    results, _ = evaluate_backends(out_dir, split, thresholds, crop_ids, truth)
    return {
        "stratum": name,
        "crops": len(selected),
        "class_distribution": _counts(truth),
        "backends": {
            backend: {
                "forced_macro_f1": value["forced_binary"]["metrics"]["macro_f1"],
                "forced_accuracy": value["forced_binary"]["metrics"]["accuracy"],
                "native_macro_f1": (
                    value["native"]["metrics"]["macro_f1"] if value["native"]["metrics"] else None
                ),
                "native_coverage": value["native"]["coverage"],
            }
            for backend, value in results.items()
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation_p4u9")
    )
    parser.add_argument("--stage", required=True, choices=["select", "test"])
    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.stage == "select":
        rows = manifest(out_dir, "val")
        crop_ids = [str(row["crop_id"]) for row in rows]
        truth = [str(row["label"]) for row in rows]
        selections = {}
        for backend in BACKENDS:
            probabilities = load(out_dir, backend, "val")
            selections[backend] = select_threshold(truth, probabilities, crop_ids, backend)
            print(f"{backend}: selected abstain_below={selections[backend]['selected']}")
        (out_dir / "frozen_operating_points.json").write_text(
            json.dumps(
                {
                    "protocol": "PROTOCOL_P4U9.md section 5.2",
                    "split_used_for_selection": "val",
                    "population": "P4-U8 rider-inclusive corrected recovery population",
                    "val_crops": len(rows),
                    "selections": selections,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out_dir / 'frozen_operating_points.json'}")
        return 0

    frozen = json.loads(
        (out_dir / "frozen_operating_points.json").read_text(encoding="utf-8")
    )
    thresholds = {name: entry["selected"] for name, entry in frozen["selections"].items()}
    inherited = inherited_thresholds()

    rows = manifest(out_dir, "test")
    crop_ids = [str(row["crop_id"]) for row in rows]
    truth = [str(row["label"]) for row in rows]

    results, forced = evaluate_backends(out_dir, "test", thresholds, crop_ids, truth)
    for backend in BACKENDS:
        macro = results[backend]["forced_binary"]["metrics"]["macro_f1"]
        print(f"{backend}: forced macro-F1 = {macro:.4f}")

    sensitivity, _ = evaluate_backends(out_dir, "test", inherited, crop_ids, truth)
    pairs = pairwise(truth, forced)
    for name, entry in pairs.items():
        interval = entry["bootstrap_delta_macro_f1"]
        print(
            f"{name}: delta={interval['observed']:+.4f} "
            f"CI[{interval['lower']:+.4f},{interval['upper']:+.4f}] "
            f"McNemar p={entry['mcnemar']['p_value']:.4g}"
        )

    shared_ids = p4u6v_recovered_ids()
    strata = [
        subset_report(
            out_dir,
            "test",
            thresholds,
            rows,
            lambda row: str(row["crop_id"]) in shared_ids,
            "shared_with_p4u6v",
        ),
        subset_report(
            out_dir,
            "test",
            thresholds,
            rows,
            lambda row: str(row["crop_id"]) not in shared_ids,
            "added_by_the_correction",
        ),
        subset_report(
            out_dir,
            "test",
            thresholds,
            rows,
            lambda row: float(row["head_height_px"]) < 30.0,
            "head_region_under_30px",
        ),
        subset_report(
            out_dir,
            "test",
            thresholds,
            rows,
            lambda row: float(row["head_height_px"]) >= 30.0,
            "head_region_30px_or_more",
        ),
    ]

    payload = {
        "protocol": "experiments/helmet_runtime_validation/PROTOCOL_P4U9.md",
        "population": (
            "P4-U8 rider-inclusive corrected recovery population; single-rider only; "
            "P4-U5 frozen split"
        ),
        "note": (
            "Separate experiment from P4-U5 and from P4-U6-V. One checkpoint per family "
            "(seed 0), so the P4-U5 seed-consistency decision rule is deliberately not "
            "reused and NO adoption claim is made here."
        ),
        "test_crops": len(rows),
        "frozen_operating_points": frozen["selections"],
        "results": results,
        "sensitivity_inherited_p4u6v_thresholds": {
            "thresholds": inherited,
            "results": sensitivity,
        },
        "pairwise": pairs,
        "strata": strata,
    }
    (out_dir / "results_p4u9.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'results_p4u9.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
