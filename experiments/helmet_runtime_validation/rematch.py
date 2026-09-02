"""P4-U8: re-measure runtime recovery under a rider-inclusive matching convention.

The evaluation scaffold P4-U6-V used compared a HELMET annotation box (motorcycle **plus**
riders) against an RT-DETR ``motorbike`` box (vehicle only). P4-U7 diagnosed that mismatch.
This module implements the corrected correspondence declared in ``PROTOCOL_P4U8.md`` section
3: a runtime motorcycle's **evaluation proxy** is the union of its own box with the boxes of
the riders the *production* ``associate_riders`` linked to it, and the annotation is matched
against that proxy at the unchanged IoU floor of 0.50.

Nothing here is a production change and nothing here is a detector improvement. The
detector, its threshold, the tracker, the association policy and the head-crop geometry are
all exactly P4-U6-V's; only the scaffold's notion of "does this runtime object correspond to
that annotation" changes.

Why this runs offline
---------------------
P4-U7's ``detection_dump.jsonl`` already holds every ``motorbike``/``person`` detection at a
0.01 floor over exactly the frames P4-U6-V processed. Filtering it at 0.50 reproduces the
production detection set, because the score threshold is a post-processing filter over a
fixed set of RT-DETR query outputs. That is not assumed: :func:`analyse_split` is run in
``motorcycle_only`` mode first and must reproduce P4-U6-V's recovered crop-id set exactly
(``--stage verify``), and only then is the corrected convention reported.

Production components are **called**, never reimplemented: ``associate_riders`` decides
which riders belong to which motorcycle, and ``head_region_box`` /
``DEFAULT_MIN_CROP_HEIGHT_PX`` decide the head-crop gates.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.derive import GT_MATCH_IOU, SCORE_THRESHOLD  # noqa: E402
from trafficpulse.association.riders import (  # noqa: E402
    RiderAssociationConfig,
    associate_riders,
)
from trafficpulse.contracts import TrackState  # noqa: E402
from trafficpulse.contracts.enums import ObjectClass, TrackStatus  # noqa: E402
from trafficpulse.contracts.primitives import BoundingBox  # noqa: E402
from trafficpulse.observations.helmet import (  # noqa: E402
    DEFAULT_HEAD_FRACTION,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    head_region_box,
)

Box = tuple[float, float, float, float]
Convention = Literal["motorcycle_only", "rider_inclusive"]

#: The two conventions this module can run. ``motorcycle_only`` exists solely to reproduce
#: P4-U6-V and prove the offline reconstruction is faithful; ``rider_inclusive`` is the
#: corrected rule of PROTOCOL_P4U8 section 3.
CONVENTIONS: tuple[Convention, ...] = ("motorcycle_only", "rider_inclusive")

#: PROTOCOL_P4U8 section 3.4: a match whose motorcycle-only IoU is below this did
#: essentially all its matching through the rider union, and is counted as suspicious.
SUSPICIOUS_MOTORCYCLE_IOU = 0.10

BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)

#: The reason codes, spelled exactly as P4-U6-V's ``derivation_summary.json`` spells them so
#: the two runs can be compared key-for-key.
REASON_NO_MATCH = f"no_motorcycle_detected_at_iou_{GT_MATCH_IOU:.2f}"
REASON_NO_RIDER = "motorcycle_matched_but_no_rider_associated"
REASON_MANY_RIDERS = "multiple_riders_associated_to_single_rider_gt"
REASON_OFF_FRAME = "head_region_off_frame"
REASON_TOO_SHORT = "head_crop_below_min_height_gate"
REASON_RECOVERED = "recovered"

#: P4-U7's four-way taxonomy, kept distinct (PROTOCOL_P4U7 section 3).
BUCKET_OF_REASON = {
    REASON_NO_MATCH: "B_detected_unmatched",  # refined to A when best IoU is exactly 0
    REASON_NO_RIDER: "C_association_failed",
    REASON_MANY_RIDERS: "C_association_failed",
    REASON_OFF_FRAME: "D_gated_off_frame",
    REASON_TOO_SHORT: "D_gated_too_short",
    REASON_RECOVERED: "D_recovered",
}


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes; ``0.0`` when they do not overlap."""

    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: Sequence[Sequence[float]]) -> Box:
    """The smallest axis-aligned box containing every input box.

    The evaluation proxy of PROTOCOL_P4U8 section 3.1 step 3. Raises on an empty
    sequence rather than inventing a degenerate box.
    """

    if not boxes:
        raise ValueError("union_box needs at least one box")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def track_states(
    video_id: str,
    frame_index: int,
    motorcycles: Sequence[Box],
    persons: Sequence[Box],
) -> tuple[TrackState, ...]:
    """Frozen ``TrackState``s for one frame's detections, so production code can run.

    A fresh ``IouTracker`` on a single frame assigns identity and nothing else: every
    detection spawns a track carrying the detection's own box, ``tainted`` is always
    ``False``, and no geometry is altered. Rebuilding that state directly is therefore
    exact, and it lets ``associate_riders`` -- the real policy -- be *called* rather than
    reimplemented.

    Ids are zero-padded and class-prefixed so the ordering used by ``associate_riders``'
    tie-break is deterministic and follows detection order. A live run's ``trk-N`` ids
    differ in name; see PROTOCOL_P4U8 section 6 for why that cannot change an outcome in
    practice.
    """

    states: list[TrackState] = []
    for index, box in enumerate(motorcycles):
        states.append(
            _state(video_id, frame_index, f"mot-{index:05d}", ObjectClass.MOTORCYCLE, box)
        )
    for index, box in enumerate(persons):
        states.append(_state(video_id, frame_index, f"per-{index:05d}", ObjectClass.PERSON, box))
    return tuple(states)


def _state(
    video_id: str, frame_index: int, track_id: str, object_class: ObjectClass, box: Sequence[float]
) -> TrackState:
    return TrackState(
        track_id=track_id,
        camera_id=video_id,
        timestamp=BASE_TS,
        frame_index=frame_index,
        object_class=object_class,
        bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        status=TrackStatus.ACTIVE,
        tainted=False,
    )


def riders_by_motorcycle(states: Sequence[TrackState]) -> dict[str, list[TrackState]]:
    """``motorcycle track id -> associated rider states``, via the production policy."""

    by_id = {state.track_id: state for state in states}
    out: dict[str, list[TrackState]] = defaultdict(list)
    for association in associate_riders(states, config=RiderAssociationConfig()):
        out[association.object_track_id].append(by_id[association.subject_track_id])
    return dict(out)


@dataclass(frozen=True)
class Proxy:
    """One runtime motorcycle as the evaluation scaffold sees it."""

    track_id: str
    motorcycle_box: Box
    rider_boxes: tuple[Box, ...]
    proxy_box: Box

    @property
    def rider_count(self) -> int:
        return len(self.rider_boxes)


def build_proxies(states: Sequence[TrackState], *, convention: Convention) -> list[Proxy]:
    """Every detected motorcycle with its matching box under ``convention``.

    Under ``motorcycle_only`` the proxy box is the vehicle box (P4-U6-V). Under
    ``rider_inclusive`` it is the union of the vehicle box with its associated riders'
    boxes (PROTOCOL_P4U8 section 3.1). The rider list itself is identical either way --
    only the box used for matching differs -- so the association-failure counts are
    directly comparable between the two runs.
    """

    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}")
    riders = riders_by_motorcycle(states)
    proxies: list[Proxy] = []
    for state in states:
        if state.object_class is not ObjectClass.MOTORCYCLE or state.tainted:
            continue
        motorcycle_box = _box(state)
        rider_boxes = tuple(_box(r) for r in riders.get(state.track_id, ()))
        proxy_box = (
            union_box((motorcycle_box, *rider_boxes))
            if convention == "rider_inclusive" and rider_boxes
            else motorcycle_box
        )
        proxies.append(
            Proxy(
                track_id=state.track_id,
                motorcycle_box=motorcycle_box,
                rider_boxes=rider_boxes,
                proxy_box=proxy_box,
            )
        )
    return proxies


def _box(state: TrackState) -> Box:
    return (state.bbox.x1, state.bbox.y1, state.bbox.x2, state.bbox.y2)


def head_crop_outcome(
    rider_box: Box, frame_width: int, frame_height: int
) -> tuple[str | None, float]:
    """Replay the production head-crop gates without the pixels.

    Returns ``(reason_or_None, head_height_px)``. The geometry is production's:
    :func:`head_region_box` cuts the top ``DEFAULT_HEAD_FRACTION`` of the rider box at full
    width, ``extract_head_region`` then clips it to the frame with floor/ceil and reports no
    crop when nothing survives, and ``height_px`` is the *unclipped* region height, which is
    what the ``DEFAULT_MIN_CROP_HEIGHT_PX`` gate compares. A test asserts this agrees with
    ``extract_head_region`` on real arrays.
    """

    box = head_region_box(
        BoundingBox(x1=rider_box[0], y1=rider_box[1], x2=rider_box[2], y2=rider_box[3]),
        head_fraction=DEFAULT_HEAD_FRACTION,
    )
    if box is None:  # pragma: no cover - BoundingBox forbids zero-height rider boxes
        return REASON_OFF_FRAME, 0.0
    height_px = box.y2 - box.y1
    x1, y1 = max(0, math.floor(box.x1)), max(0, math.floor(box.y1))
    x2, y2 = min(frame_width, math.ceil(box.x2)), min(frame_height, math.ceil(box.y2))
    if x2 <= x1 or y2 <= y1:
        return REASON_OFF_FRAME, height_px
    if height_px < DEFAULT_MIN_CROP_HEIGHT_PX:
        return REASON_TOO_SHORT, height_px
    return None, height_px


@dataclass
class Outcome:
    """One eligible annotation's fate, with everything the report stratifies on."""

    crop_id: str
    split: str
    video_id: str
    frame_index: int
    track_id: str
    site_id: str
    label: str
    reason: str
    bucket: str
    gt_area: float
    gt_height: float
    best_iou: float
    motorcycle_only_iou: float
    head_height_px: float | None = None
    rider_box: Box | None = None
    matched_proxy: str | None = None
    suspicious_match: bool = False


@dataclass
class FrameResult:
    outcomes: list[Outcome] = field(default_factory=list)
    contested_proxies: int = 0


def match_frame(record: dict[str, Any], *, convention: Convention, split: str) -> FrameResult:
    """Bucket every eligible annotation in one frame under ``convention``.

    The assignment procedure is P4-U6-V's, verbatim: annotations are considered in
    ``crop_id`` order and each takes the highest-IoU proxy not already used in this frame,
    subject to the unchanged 0.50 floor. Keeping it identical is what makes the box
    convention the single variable that changes.
    """

    motorcycles = [
        tuple(m["box"]) for m in record["motorcycles"] if m["score"] >= SCORE_THRESHOLD
    ]
    persons = [tuple(p["box"]) for p in record["persons"] if p["score"] >= SCORE_THRESHOLD]
    states = track_states(record["video_id"], record["frame_index"], motorcycles, persons)
    proxies = build_proxies(states, convention=convention)

    eligible = sorted(
        (e for e in record["eligible"] if e["split"] == split), key=lambda e: e["crop_id"]
    )
    result = FrameResult()
    used: set[str] = set()
    clears_floor: Counter[str] = Counter()

    for item in eligible:
        gt: Box = tuple(item["box"])  # type: ignore[assignment]
        best: Proxy | None = None
        best_iou = 0.0
        best_any = 0.0
        best_motorcycle_only = 0.0
        for proxy in proxies:
            value = iou(gt, proxy.proxy_box)
            if value >= GT_MATCH_IOU:
                clears_floor[proxy.track_id] += 1
            best_any = max(best_any, value)
            if proxy.track_id in used:
                continue
            if value > best_iou:
                best, best_iou = proxy, value
                best_motorcycle_only = iou(gt, proxy.motorcycle_box)

        outcome = Outcome(
            crop_id=item["crop_id"],
            split=item["split"],
            video_id=record["video_id"],
            frame_index=record["frame_index"],
            track_id=item["track_id"],
            site_id=item["site_id"],
            label=item["label"],
            reason=REASON_RECOVERED,
            bucket="D_recovered",
            gt_area=(gt[2] - gt[0]) * (gt[3] - gt[1]),
            gt_height=gt[3] - gt[1],
            best_iou=best_iou,
            motorcycle_only_iou=best_motorcycle_only,
        )

        if best is None or best_iou < GT_MATCH_IOU:
            outcome.reason = REASON_NO_MATCH
            # A vs B (PROTOCOL_P4U7 section 3): did anything overlap the annotation at all?
            outcome.bucket = "A_no_detection" if best_any == 0.0 else "B_detected_unmatched"
            result.outcomes.append(outcome)
            continue

        used.add(best.track_id)
        outcome.matched_proxy = best.track_id
        outcome.suspicious_match = best_motorcycle_only < SUSPICIOUS_MOTORCYCLE_IOU

        if best.rider_count == 0:
            outcome.reason, outcome.bucket = REASON_NO_RIDER, "C_association_failed"
            result.outcomes.append(outcome)
            continue
        if best.rider_count > 1:
            outcome.reason, outcome.bucket = REASON_MANY_RIDERS, "C_association_failed"
            result.outcomes.append(outcome)
            continue

        rider_box = best.rider_boxes[0]
        reason, height = head_crop_outcome(
            rider_box, int(record["frame_w"]), int(record["frame_h"])
        )
        outcome.head_height_px = height
        outcome.rider_box = rider_box
        if reason is not None:
            outcome.reason, outcome.bucket = reason, BUCKET_OF_REASON[reason]
        result.outcomes.append(outcome)

    result.contested_proxies = sum(1 for count in clears_floor.values() if count > 1)
    return result


# --- reporting ---------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    """Min / Q1 / median / Q3 / max, computed by index so it needs no dependency."""

    if not values:
        return None
    ordered = sorted(values)
    last = len(ordered) - 1

    def at(fraction: float) -> float:
        return round(ordered[int(round(fraction * last))], 3)

    return {
        "min": round(ordered[0], 3),
        "q1": at(0.25),
        "median": at(0.50),
        "q3": at(0.75),
        "max": round(ordered[-1], 3),
    }


def _rate(recovered: int, total: int) -> float | None:
    """Recovery rate, or ``None`` when the stratum is empty (never a fabricated 0.0)."""

    return recovered / total if total else None


def _stratify(outcomes: Sequence[Outcome], key: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[str(key(outcome))].append(outcome)
    return {
        name: {
            "eligible": len(items),
            "recovered": sum(1 for o in items if o.reason == REASON_RECOVERED),
            "recovery_rate": _rate(
                sum(1 for o in items if o.reason == REASON_RECOVERED), len(items)
            ),
            "buckets": dict(sorted(Counter(o.bucket for o in items).items())),
        }
        for name, items in sorted(groups.items())
    }


def _size_quartiles(outcomes: Sequence[Outcome]) -> dict[str, dict[str, Any]]:
    """Recovery by annotation-area quartile.

    The cut points come from the **eligible** population of this split, so the strata are a
    property of the population and cannot shift with the result (PROTOCOL_P4U8 section 5).
    """

    if not outcomes:
        return {}
    areas = sorted(o.gt_area for o in outcomes)
    last = len(areas) - 1
    cuts = [areas[int(round(f * last))] for f in (0.25, 0.50, 0.75)]

    def quartile(outcome: Outcome) -> str:
        for index, cut in enumerate(cuts):
            if outcome.gt_area <= cut:
                return f"Q{index + 1}"
        return "Q4"

    table = _stratify(outcomes, quartile)
    for index, name in enumerate(("Q1", "Q2", "Q3", "Q4")):
        if name in table:
            table[name]["area_upper_bound"] = round(cuts[index], 1) if index < 3 else None
    return table


def summarise(outcomes: Sequence[Outcome], *, contested: int) -> dict[str, Any]:
    """Everything PROTOCOL_P4U8 section 5 requires, for one split under one convention."""

    total = len(outcomes)
    recovered = [o for o in outcomes if o.reason == REASON_RECOVERED]
    matched = [o for o in outcomes if o.matched_proxy is not None]
    association_failures = [o for o in outcomes if o.bucket == "C_association_failed"]
    detection_failures = [
        o for o in outcomes if o.bucket in ("A_no_detection", "B_detected_unmatched")
    ]
    return {
        "eligible_single_rider": total,
        "recovered": len(recovered),
        "recovery_rate": _rate(len(recovered), total),
        "reasons": dict(sorted(Counter(o.reason for o in outcomes).items())),
        "buckets": dict(sorted(Counter(o.bucket for o in outcomes).items())),
        "detection_failure_rate_of_eligible": _rate(len(detection_failures), total),
        "association_failure_rate_of_eligible": _rate(len(association_failures), total),
        "association_failure_rate_of_matched": _rate(len(association_failures), len(matched)),
        "matched_motorcycles": len(matched),
        "class_distribution_eligible": dict(sorted(Counter(o.label for o in outcomes).items())),
        "class_distribution_recovered": dict(
            sorted(Counter(o.label for o in recovered).items())
        ),
        "recovery_by_class": _stratify(outcomes, lambda o: o.label),
        "recovery_by_site": _stratify(outcomes, lambda o: o.site_id),
        "recovery_by_size_quartile": _size_quartiles(outcomes),
        "head_height_px_recovered": _quantiles([
            o.head_height_px for o in recovered if o.head_height_px is not None
        ]),
        "gt_height_px_eligible": _quantiles([o.gt_height for o in outcomes]),
        "gt_height_px_recovered": _quantiles([o.gt_height for o in recovered]),
        "match_iou_recovered": _quantiles([o.best_iou for o in recovered]),
        "motorcycle_only_iou_recovered": _quantiles([o.motorcycle_only_iou for o in recovered]),
        "suspicious_matches": sum(1 for o in matched if o.suspicious_match),
        "suspicious_match_rate_of_matched": _rate(
            sum(1 for o in matched if o.suspicious_match), len(matched)
        ),
        "contested_proxies": contested,
    }


def analyse_split(
    records: Iterable[dict[str, Any]], *, split: str, convention: Convention
) -> tuple[dict[str, Any], list[Outcome]]:
    """Run the matcher over every frame of one split and summarise it."""

    outcomes: list[Outcome] = []
    contested = 0
    for record in records:
        if not any(e["split"] == split for e in record["eligible"]):
            continue
        result = match_frame(record, convention=convention, split=split)
        outcomes.extend(result.outcomes)
        contested += result.contested_proxies
    return summarise(outcomes, contested=contested), outcomes


# --- corpus-level context (rider counts, eligibility shares) -------------------------


def population_shape(crop_dir: Path, split: str) -> dict[str, Any]:
    """Frozen-corpus context: how much of the split the single-rider premise excludes."""

    from helmet_cnn_vit.datasets import load_rows

    rows = load_rows(crop_dir, split)
    counts = Counter(row.rider_count for row in rows)
    single = counts.get(1, 0)
    return {
        "total_frozen_crops": len(rows),
        "single_rider_eligible": single,
        "eligible_share": _rate(single, len(rows)),
        "multi_rider_excluded": len(rows) - single,
        "multi_rider_share": _rate(len(rows) - single, len(rows)),
        "rider_count_distribution": {str(k): v for k, v in sorted(counts.items())},
        "class_distribution_all": dict(sorted(Counter(row.label for row in rows).items())),
        "class_distribution_single_rider": dict(
            sorted(Counter(row.label for row in rows if row.rider_count == 1).items())
        ),
    }


def recovery_by_rider_count(
    crop_dir: Path, split: str, outcomes: Sequence[Outcome]
) -> dict[str, dict[str, Any]]:
    """Recovery per rider count over the **whole** frozen split, not just the eligible part.

    Multi-rider annotations are never matched by this unit -- the single-rider premise is
    architectural (PROTOCOL_P4U8 section 4) -- so their recovery is reported as an explicit
    ``not_evaluated`` row rather than as a zero, which would read as a measured failure.
    """

    from helmet_cnn_vit.datasets import load_rows

    rows = load_rows(crop_dir, split)
    recovered = {o.crop_id for o in outcomes if o.reason == REASON_RECOVERED}
    table: dict[str, dict[str, Any]] = {}
    for count, group in sorted(_group_by(rows, lambda r: r.rider_count).items()):
        ids = [row.crop_id for row in group]
        table[str(count)] = {
            "frozen_crops": len(ids),
            "evaluated": count == 1,
            "recovered": sum(1 for i in ids if i in recovered) if count == 1 else None,
            "recovery_rate": (
                _rate(sum(1 for i in ids if i in recovered), len(ids)) if count == 1 else None
            ),
            "note": None if count == 1 else "not_evaluated: rider_slot cannot name the driver",
        }
    return table


def _group_by(items: Iterable[Any], key: Any) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        out[key(item)].append(item)
    return dict(out)


# --- fidelity check -----------------------------------------------------------------


def verify_against_p4u6v(
    records: Sequence[dict[str, Any]], baseline_dir: Path
) -> dict[str, Any]:
    """Reproduce P4-U6-V exactly in ``motorcycle_only`` mode (PROTOCOL_P4U8 section 6).

    Compares the recovered crop-id **set** and the per-reason counts against P4-U6-V's own
    ``crops.jsonl`` / ``derivation_summary.json``. A mismatch invalidates the offline
    reconstruction, so the caller must refuse to publish corrected numbers when this fails.
    """

    baseline_ids = {
        json.loads(line)["crop_id"]
        for line in (baseline_dir / "crops.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    baseline_summary = json.loads(
        (baseline_dir / "derivation_summary.json").read_text(encoding="utf-8")
    )

    reproduced: set[str] = set()
    reasons: Counter[str] = Counter()
    for split in ("val", "test"):
        _, outcomes = analyse_split(records, split=split, convention="motorcycle_only")
        reasons.update(o.reason for o in outcomes)
        reproduced.update(o.crop_id for o in outcomes if o.reason == REASON_RECOVERED)

    missing = sorted(baseline_ids - reproduced)
    extra = sorted(reproduced - baseline_ids)
    return {
        "baseline_recovered": len(baseline_ids),
        "reproduced_recovered": len(reproduced),
        "missing_from_reproduction": len(missing),
        "extra_in_reproduction": len(extra),
        "missing_examples": missing[:10],
        "extra_examples": extra[:10],
        "baseline_outcomes": baseline_summary["outcomes"],
        "reproduced_outcomes": dict(sorted(reasons.items())),
        "exact": not missing and not extra,
    }


def load_dump(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stream_dump(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _outcome_rows(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    return [
        {
            "crop_id": o.crop_id,
            "split": o.split,
            "video_id": o.video_id,
            "frame_index": o.frame_index,
            "track_id": o.track_id,
            "site_id": o.site_id,
            "label": o.label,
            "reason": o.reason,
            "bucket": o.bucket,
            "best_iou": round(o.best_iou, 5),
            "motorcycle_only_iou": round(o.motorcycle_only_iou, 5),
            "gt_area": round(o.gt_area, 2),
            "gt_height": round(o.gt_height, 2),
            "head_height_px": None if o.head_height_px is None else round(o.head_height_px, 3),
            # NOT rounded: the crop slice floors/ceils these, so a 3-decimal round can
            # move a crop edge by a pixel. P4-U9 cuts its crops from exactly these values.
            "rider_box": None if o.rider_box is None else list(o.rider_box),
            "matched_proxy": o.matched_proxy,
            "suspicious_match": o.suspicious_match,
        }
        for o in outcomes
    ]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        default=str(REPO_ROOT / "runs" / "helmet_detection_recovery" / "detection_dump.jsonl"),
    )
    parser.add_argument(
        "--baseline", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation")
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation_p4u8")
    )
    parser.add_argument(
        "--crops", default=str(REPO_ROOT / "data" / "processed" / "helmet-cnnvit")
    )
    parser.add_argument("--stage", required=True, choices=["verify", "val", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_dump(Path(args.dump))
    crop_dir = Path(args.crops)

    if args.stage == "verify":
        report = verify_against_p4u6v(records, Path(args.baseline))
        (out_dir / "reconstruction_fidelity.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 0 if report["exact"] else 1

    fidelity_path = out_dir / "reconstruction_fidelity.json"
    if not fidelity_path.is_file():
        raise SystemExit("run --stage verify first: an unproven reconstruction publishes nothing")
    if not json.loads(fidelity_path.read_text(encoding="utf-8"))["exact"]:
        raise SystemExit("reconstruction is not exact; corrected numbers must not be published")

    split = args.stage
    payload: dict[str, Any] = {
        "protocol": "experiments/helmet_runtime_validation/PROTOCOL_P4U8.md",
        "label": "evaluation-scaffold correction, NOT a detector improvement",
        "split": split,
        "score_threshold": SCORE_THRESHOLD,
        "gt_match_iou": GT_MATCH_IOU,
        "population": population_shape(crop_dir, split),
        "conventions": {},
    }
    for convention in CONVENTIONS:
        summary, outcomes = analyse_split(records, split=split, convention=convention)
        summary["recovery_by_rider_count"] = recovery_by_rider_count(crop_dir, split, outcomes)
        payload["conventions"][convention] = summary
        (out_dir / f"outcomes_{convention}_{split}.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in _outcome_rows(outcomes)
            ),
            encoding="utf-8",
        )
        print(
            f"{split}/{convention}: recovered {summary['recovered']}/"
            f"{summary['eligible_single_rider']} "
            f"({(summary['recovery_rate'] or 0.0) * 100:.1f}%)"
        )

    (out_dir / f"recovery_{split}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / f'recovery_{split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
