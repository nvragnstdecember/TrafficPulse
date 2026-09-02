"""Analyse the runtime-crop scores: operating points, metrics, paired comparisons.

Implements PROTOCOL §5.1 exactly. Metric and statistics primitives are **imported from the
frozen P4-U5 package** rather than reimplemented -- they are already unit-tested, and a
second implementation of macro-F1 would be a silent divergence risk. Importing frozen code
is read-only; nothing in ``helmet_cnn_vit`` is modified.

The P4-U5 *decision rule* is deliberately not imported (PROTOCOL §6.1): it requires a seed
dimension this experiment does not have.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_cnn_vit.metrics import compute_metrics  # noqa: E402
from helmet_cnn_vit.stats import bootstrap_delta_macro_f1  # noqa: E402


def exact_mcnemar(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> dict:
    """Exact binomial McNemar, computed in log space.

    Identical in definition to ``helmet_cnn_vit.stats.mcnemar`` -- and asserted equal to it
    in the tests wherever that one can run -- but it does not overflow.

    The frozen implementation evaluates ``sum(comb(n, i)) / 2.0**n`` in float, which raises
    ``OverflowError`` once the discordant count ``n`` exceeds about 1023. P4-U5 never hit
    that (its discordant counts were in the hundreds); this experiment does, because
    zero-shot and the trained models disagree on well over a thousand crops. The frozen
    helper is **not** patched -- it is part of a tagged, published experiment -- so the
    log-space form lives here instead, and the limitation is recorded in the report.
    """

    both = only_first = only_second = neither = 0
    for a, b in zip(first_correct, second_correct, strict=True):
        if a and b:
            both += 1
        elif a:
            only_first += 1
        elif b:
            only_second += 1
        else:
            neither += 1

    n = only_first + only_second
    if n == 0:
        p_value = 1.0
    else:
        k = min(only_first, only_second)
        # log C(n, i) - n log 2, summed with a max-shifted log-sum-exp.
        log_terms = [
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - n * math.log(2.0)
            for i in range(k + 1)
        ]
        peak = max(log_terms)
        log_tail = peak + math.log(sum(math.exp(t - peak) for t in log_terms))
        p_value = min(1.0, 2.0 * math.exp(log_tail))
    return {
        "only_first_correct": only_first,
        "only_second_correct": only_second,
        "both_correct": both,
        "both_wrong": neither,
        "p_value": p_value,
        "discordant": n,
    }

BACKENDS = ("zeroshot", "resnet", "deit")
BINARY = ("helmet", "no_helmet")
ABSTAIN = "__abstain__"

#: PROTOCOL §5.1, fixed before any val number was observed.
THRESHOLD_GRID: tuple[float | None, ...] = (
    None, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)
COVERAGE_FLOOR = 0.90


def load(out_dir: Path, backend: str, split: str) -> dict[str, dict[str, float]]:
    payload = json.loads(
        (out_dir / f"scores_{backend}_{split}.json").read_text(encoding="utf-8")
    )
    return {p["crop_id"]: p["probabilities"] for p in payload["predictions"]}


def manifest(out_dir: Path, split: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (out_dir / "crops.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["split"] == split]


def forced_binary(probabilities: dict[str, float]) -> tuple[str, float]:
    """Arg-max restricted to the binary vocabulary, renormalised (PROTOCOL §5.1 A)."""

    helmet = float(probabilities.get("helmet", 0.0))
    no_helmet = float(probabilities.get("no_helmet", 0.0))
    total = helmet + no_helmet
    if total <= 0.0:  # pragma: no cover - a degenerate posterior
        return "helmet", 0.5
    helmet, no_helmet = helmet / total, no_helmet / total
    return ("no_helmet", no_helmet) if no_helmet > helmet else ("helmet", helmet)


def native(
    backend: str, probabilities: dict[str, float], threshold: float | None
) -> tuple[str, float]:
    """Each backend's own behaviour (PROTOCOL §5.1 B)."""

    label = max(probabilities, key=lambda k: probabilities[k])
    score = float(probabilities[label])
    if backend == "zeroshot":
        # turban and uncertain are abstentions: never silently a binary call.
        return (label, score) if label in BINARY else (ABSTAIN, score)
    if threshold is not None and score < threshold:
        return ABSTAIN, score
    return label, score


def evaluate(truth: Sequence[str], decisions: Sequence[str], scores: Sequence[float]) -> dict:
    """Metrics on the covered subset, plus explicit coverage."""

    covered = [i for i, d in enumerate(decisions) if d != ABSTAIN]
    coverage = len(covered) / len(decisions) if decisions else 0.0
    if not covered:
        return {"coverage": coverage, "covered": 0, "total": len(decisions), "metrics": None}
    sub_truth = [truth[i] for i in covered]
    sub_pred = [decisions[i] for i in covered]
    sub_conf = [scores[i] for i in covered]
    no_helmet_scores = [
        s if p == "no_helmet" else 1.0 - s for p, s in zip(sub_pred, sub_conf, strict=True)
    ]
    metrics = compute_metrics(
        sub_truth, sub_pred, no_helmet_scores=no_helmet_scores, confidences=sub_conf
    )
    return {
        "coverage": coverage,
        "covered": len(covered),
        "total": len(decisions),
        "abstained": len(decisions) - len(covered),
        "metrics": metrics.model_dump(),
    }


def select_threshold(truth, probabilities_by_crop, crop_ids, backend) -> dict:
    """PROTOCOL §5.1 B selection rule, on val only."""

    trace = []
    best = None
    for threshold in THRESHOLD_GRID:
        decisions, scores = [], []
        for crop_id in crop_ids:
            label, score = native(backend, probabilities_by_crop[crop_id], threshold)
            decisions.append(label)
            scores.append(score)
        result = evaluate(truth, decisions, scores)
        macro = result["metrics"]["macro_f1"] if result["metrics"] else None
        trace.append(
            {"threshold": threshold, "coverage": result["coverage"], "val_macro_f1": macro}
        )
        if macro is None or result["coverage"] < COVERAGE_FLOOR:
            continue
        # Strict '>' keeps the earlier (lower) threshold on a tie: prefer coverage.
        if best is None or macro > best[1]:
            best = (threshold, macro)
    return {
        "selected": best[0] if best else None,
        "val_macro_f1": best[1] if best else None,
        "grid": trace,
        "coverage_floor": COVERAGE_FLOOR,
        "rule": "max val macro-F1 subject to coverage >= 0.90; ties to lower threshold",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation"))
    parser.add_argument("--stage", required=True, choices=["select", "test"])
    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.stage == "select":
        rows = manifest(out_dir, "val")
        crop_ids = [r["crop_id"] for r in rows]
        truth = [r["label"] for r in rows]
        selections = {}
        for backend in BACKENDS:
            probabilities = load(out_dir, backend, "val")
            selections[backend] = select_threshold(truth, probabilities, crop_ids, backend)
            print(f"{backend}: selected abstain_below={selections[backend]['selected']}")
        (out_dir / "frozen_operating_points.json").write_text(
            json.dumps(
                {
                    "protocol": "PROTOCOL.md §5.1",
                    "split_used_for_selection": "val",
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

    # --- test: the frozen configuration, applied once -------------------------
    frozen = json.loads((out_dir / "frozen_operating_points.json").read_text(encoding="utf-8"))
    rows = manifest(out_dir, "test")
    crop_ids = [r["crop_id"] for r in rows]
    truth = [r["label"] for r in rows]

    results: dict[str, dict] = {}
    forced_decisions: dict[str, list[str]] = {}
    for backend in BACKENDS:
        probabilities = load(out_dir, backend, "test")
        f_labels, f_scores = [], []
        n_labels, n_scores = [], []
        threshold = frozen["selections"][backend]["selected"]
        for crop_id in crop_ids:
            row = probabilities[crop_id]
            label, score = forced_binary(row)
            f_labels.append(label)
            f_scores.append(score)
            label, score = native(backend, row, threshold)
            n_labels.append(label)
            n_scores.append(score)
        forced_decisions[backend] = f_labels
        results[backend] = {
            "frozen_abstain_below": threshold,
            "forced_binary": evaluate(truth, f_labels, f_scores),
            "native": evaluate(truth, n_labels, n_scores),
            "native_label_counts": _counts(n_labels),
        }
        macro = results[backend]["forced_binary"]["metrics"]["macro_f1"]
        print(f"{backend}: forced macro-F1 = {macro:.4f}")

    # Paired comparisons on the forced-binary view (100% coverage for all three).
    pairs = {}
    for i, first in enumerate(BACKENDS):
        for second in BACKENDS[i + 1 :]:
            a, b = forced_decisions[first], forced_decisions[second]
            correct_a = [p == t for p, t in zip(a, truth, strict=True)]
            correct_b = [p == t for p, t in zip(b, truth, strict=True)]
            interval = bootstrap_delta_macro_f1(truth, a, b, resamples=10000, seed=0)
            pairs[f"{first}_vs_{second}"] = {
                "mcnemar": exact_mcnemar(correct_a, correct_b),
                "bootstrap_delta_macro_f1": interval.model_dump(),
            }
            print(
                f"{first} vs {second}: delta={interval.observed:+.4f} "
                f"CI[{interval.lower:+.4f},{interval.upper:+.4f}]"
            )

    payload = {
        "protocol": "experiments/helmet_runtime_validation/PROTOCOL.md",
        "note": (
            "P4-U6-V. Separate from P4-U5; makes no adoption claim and does not reuse the "
            "P4-U5 seed-consistency decision rule (PROTOCOL §6.1)."
        ),
        "test_crops": len(rows),
        "frozen_operating_points": frozen["selections"],
        "results": results,
        "pairwise": pairs,
    }
    (out_dir / "results_runtime_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'results_runtime_validation.json'}")
    return 0


def _counts(labels: Sequence[str]) -> dict[str, int]:
    from collections import Counter

    return dict(sorted(Counter(labels).items()))


if __name__ == "__main__":
    sys.exit(main())
