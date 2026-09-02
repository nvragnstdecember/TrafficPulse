"""P4-U7 offline analysis: failure taxonomy, threshold sweep, detector operating point.

Reads the single low-threshold ``detection_dump.jsonl`` and answers every question in
PROTOCOL_P4U7 by *filtering* it. The detector is never re-run, so no threshold can be
reached by searching until a number improves.

The taxonomy (§3) is kept strictly separate throughout: A (nothing detected), B (detected
but below the IoU floor), C (matched but association failed), D (recovered). Collapsing
them would hide which of the four plausible causes is actually operating.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.derive import GT_MATCH_IOU  # noqa: E402

#: PROTOCOL_P4U7 §5, fixed before the dump was produced.
THRESHOLD_GRID: tuple[float, ...] = (0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
BASELINE_THRESHOLD = 0.50

#: The production association policy, restated as plain geometry so the sweep needs no
#: TrackState construction. Verified against RiderAssociationConfig by the tests.
from trafficpulse.association.riders import DEFAULT_MIN_OVERLAP  # noqa: E402


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def overlap_over_min_area(a: Sequence[float], b: Sequence[float]) -> float:
    """IoMin, the production association measure (``association.riders``)."""

    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def classify_frame(record: dict, threshold: float) -> list[dict]:
    """Bucket every eligible rider in one frame at one threshold (§3 taxonomy)."""

    motorcycles = [m for m in record["motorcycles"] if m["score"] >= threshold]
    persons = [p for p in record["persons"] if p["score"] >= threshold]
    # Every motorcycle detection, regardless of threshold, for the "did a sub-threshold
    # detection exist?" question that separates operating point from capability.
    all_motorcycles = record["motorcycles"]

    out: list[dict] = []
    used: set[int] = set()
    for item in sorted(record["eligible"], key=lambda e: e["crop_id"]):
        gt = item["box"]
        best_index, best_iou = None, 0.0
        for index, moto in enumerate(motorcycles):
            if index in used:
                continue
            value = iou(gt, moto["box"])
            if value > best_iou:
                best_index, best_iou = index, value

        # Best over *all* detections, ignoring the threshold entirely.
        sub_best_iou, sub_best_score = 0.0, 0.0
        for moto in all_motorcycles:
            value = iou(gt, moto["box"])
            if value > sub_best_iou:
                sub_best_iou, sub_best_score = value, moto["score"]

        entry = {
            "crop_id": item["crop_id"],
            "split": item["split"],
            "label": item["label"],
            "site_id": item["site_id"],
            "gt_area": area(gt),
            "gt_h": gt[3] - gt[1],
            "gt_w": gt[2] - gt[0],
            "best_iou": best_iou,
            "best_iou_any_score": sub_best_iou,
            "score_of_best_any": sub_best_score,
            "n_gt_in_frame": len(record["gt_all"]),
        }

        if best_index is None or best_iou < GT_MATCH_IOU:
            # A vs B: did *any* detection at this threshold touch the GT box at all?
            entry["bucket"] = "A_no_detection" if best_iou == 0.0 else "B_detected_unmatched"
            out.append(entry)
            continue

        used.add(best_index)
        moto_box = motorcycles[best_index]["box"]
        riders = [
            p for p in persons if overlap_over_min_area(p["box"], moto_box) >= DEFAULT_MIN_OVERLAP
        ]
        if len(riders) != 1:
            entry["bucket"] = "C_association_failed"
            entry["n_riders"] = len(riders)
            out.append(entry)
            continue

        entry["bucket"] = "D_recovered"
        entry["rider_box"] = riders[0]["box"]
        entry["head_h"] = (riders[0]["box"][3] - riders[0]["box"][1]) * 0.30
        out.append(entry)
    return out


def detection_prf(record: dict, threshold: float) -> tuple[int, int, int]:
    """(tp, fp, fn) for motorcycle detection against the **full** frame annotation."""

    detections = sorted(
        (m for m in record["motorcycles"] if m["score"] >= threshold),
        key=lambda m: -m["score"],
    )
    truths = [g["box"] for g in record["gt_all"]]
    matched: set[int] = set()
    tp = 0
    for det in detections:
        best_index, best_iou = None, GT_MATCH_IOU
        for index, truth in enumerate(truths):
            if index in matched:
                continue
            value = iou(det["box"], truth)
            if value >= best_iou:
                best_index, best_iou = index, value
        if best_index is not None:
            matched.add(best_index)
            tp += 1
    return tp, len(detections) - tp, len(truths) - tp


def sweep(records: Sequence[dict], split: str) -> list[dict]:
    """Every grid point, on one split, from the same fixed evidence."""

    rows = []
    for threshold in THRESHOLD_GRID:
        buckets: Counter[str] = Counter()
        labels: Counter[str] = Counter()
        tp = fp = fn = 0
        for record in records:
            # The official split is video-level, so a frame belongs wholly to one split.
            # Detection metrics are scoped the same way: a val selection that counted test
            # frames would be tuning on test through the back door.
            if not any(e["split"] == split for e in record["eligible"]):
                continue
            scoped = dict(record)
            scoped["eligible"] = [e for e in record["eligible"] if e["split"] == split]
            for entry in classify_frame(scoped, threshold):
                buckets[entry["bucket"]] += 1
                if entry["bucket"] == "D_recovered":
                    labels[entry["label"]] += 1
            a, b, c = detection_prf(record, threshold)
            tp += a
            fp += b
            fn += c
        eligible = sum(buckets.values())
        recovered = buckets["D_recovered"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "eligible": eligible,
                "recovered": recovered,
                "coverage": recovered / eligible if eligible else 0.0,
                "buckets": dict(buckets),
                "recovered_labels": dict(labels),
                "recovered_no_helmet_share": (
                    labels["no_helmet"] / recovered if recovered else 0.0
                ),
                "detection_tp": tp,
                "detection_fp": fp,
                "detection_fn": fn,
                "detection_precision": precision,
                "detection_recall": recall,
                "detection_f1": f1,
            }
        )
    return rows


def select(rows: Sequence[dict]) -> dict:
    """PROTOCOL_P4U7 §5: max val detection F1; ties to the higher threshold."""

    best = None
    for row in rows:
        # Grid is descending, so a strict '>' keeps the first (higher) threshold on a tie.
        if best is None or row["detection_f1"] > best["detection_f1"]:
            best = row
    return {
        "selected_threshold": best["threshold"],
        "val_detection_f1": best["detection_f1"],
        "rule": "max val motorcycle-detection F1 (IoU>=0.50); ties to the higher threshold",
        "grid": list(rows),
    }


def load_dump(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "helmet_detection_recovery"))
    parser.add_argument("--stage", required=True, choices=["diagnose", "select", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    records = load_dump(out_dir / "detection_dump.jsonl")

    if args.stage == "diagnose":
        # Baseline taxonomy only: what the P4-U6-V operating point actually did.
        entries: list[dict] = []
        for record in records:
            entries.extend(classify_frame(record, BASELINE_THRESHOLD))
        payload = {
            "threshold": BASELINE_THRESHOLD,
            "total": len(entries),
            "buckets": dict(Counter(e["bucket"] for e in entries)),
            "entries": entries,
        }
        (out_dir / "baseline_taxonomy.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in payload.items() if k != "entries"}, indent=2))
        return 0

    if args.stage == "select":
        rows = sweep(records, "val")
        selection = select(rows)
        (out_dir / "frozen_detector_operating_point.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for row in rows:
            print(
                f"  thr={row['threshold']:.2f} cov={row['coverage']:.3f} "
                f"recovered={row['recovered']:5d} detP={row['detection_precision']:.3f} "
                f"detR={row['detection_recall']:.3f} detF1={row['detection_f1']:.3f}"
            )
        print(f"SELECTED {selection['selected_threshold']}")
        return 0

    frozen = json.loads(
        (out_dir / "frozen_detector_operating_point.json").read_text(encoding="utf-8")
    )
    threshold = frozen["selected_threshold"]
    rows = sweep(records, "test")
    chosen = next(r for r in rows if r["threshold"] == threshold)
    baseline = next(r for r in rows if r["threshold"] == BASELINE_THRESHOLD)
    payload = {
        "frozen_threshold": threshold,
        "baseline_threshold": BASELINE_THRESHOLD,
        "test_at_frozen": chosen,
        "test_at_baseline": baseline,
        "full_test_grid": rows,
    }
    (out_dir / "test_recovery.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"frozen": chosen, "baseline": baseline}, indent=2))
    return 0


def site_breakdown(entries: Sequence[dict]) -> dict[str, dict[str, int]]:
    """Per-site bucket counts, for the 'does failure cluster by site?' question."""

    by_site: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        by_site[entry["site_id"]][entry["bucket"]] += 1
    return {site: dict(counts) for site, counts in sorted(by_site.items())}


if __name__ == "__main__":
    sys.exit(main())
