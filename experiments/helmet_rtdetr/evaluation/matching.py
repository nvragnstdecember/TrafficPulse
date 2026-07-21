"""Deterministic COCO-compatible greedy IoU matching (H5).

Assumptions, stated explicitly
------------------------------
* **Geometry**: boxes are pixel COCO ``(x, y, w, h)`` (the H2 :class:`BBox`);
  IoU is plain intersection-over-union on axis-aligned boxes. Zero-overlap and
  touching boxes score 0.0.
* **Greedy, score-ordered**: predictions are visited in descending score order
  and each claims the highest-IoU *still unmatched* ground-truth box at or
  above the threshold — the standard COCO evaluator behaviour for non-crowd
  ground truth (this package has no crowd annotations; H2 exports none).
* **Class-aware by default**: a prediction may only match ground truth of its
  own class. The confusion matrix intentionally matches class-agnostically
  (``class_aware=False``) to expose inter-class confusion.
* **One-to-one**: each ground-truth box is claimed at most once; later (lower
  scored) predictions that overlap an already-claimed box become false
  positives, as in COCO.
* **Total determinism**: every ordering is a pure function of content. The
  prediction order breaks score ties by ``(image_id, class, quantised box)``;
  the ground-truth order is ``(image_path, class id, quantised box)``; IoU ties
  between candidate ground-truth boxes resolve to the lowest index in that
  order. No randomness, no dict-iteration dependence, no id()/hash order.

Inputs to :func:`match_greedy` MUST already be in these canonical orders
(:func:`prediction_order` / :func:`ground_truth_order`); the returned indices
refer to those sequences. The evaluator sorts once and reuses the ordering
across the whole IoU ladder.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..models import _Model
from ..rtdetr.data import LABEL_IDS
from ..unified import BBox, UnifiedObject
from .models import Prediction


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two pixel ``(x, y, w, h)`` boxes."""

    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    return intersection / (a.area + b.area - intersection)


def prediction_order(predictions: Iterable[Prediction]) -> tuple[Prediction, ...]:
    """The canonical prediction order: score descending, content tiebreak.

    The tiebreak ``(image_id, class, quantised box)`` makes the order a total,
    content-derived one — two runs that produce the same predictions in any
    order sort identically.
    """

    return tuple(
        sorted(
            predictions,
            key=lambda p: (-p.score, p.image_id, p.label.value, p.bbox.quantised_key()),
        )
    )


def ground_truth_order(objects: Iterable[UnifiedObject]) -> tuple[UnifiedObject, ...]:
    """The canonical ground-truth order: ``(image_path, class id, quantised box)``."""

    return tuple(
        sorted(
            objects,
            key=lambda o: (o.image_path, LABEL_IDS[o.label], o.bbox.quantised_key()),
        )
    )


def cap_per_image(
    predictions: Sequence[Prediction], max_detections: int
) -> tuple[Prediction, ...]:
    """Keep at most ``max_detections`` predictions per image (COCO's cap).

    Expects canonical order; because that order is globally score-descending,
    keeping each image's first ``max_detections`` entries keeps its highest
    scored ones, and the overall order is preserved.
    """

    kept: list[Prediction] = []
    seen: dict[str, int] = {}
    for prediction in predictions:
        count = seen.get(prediction.image_id, 0)
        if count < max_detections:
            kept.append(prediction)
            seen[prediction.image_id] = count + 1
    return tuple(kept)


class MatchResult(_Model):
    """The outcome of one greedy matching pass at one IoU threshold.

    Indices refer to the canonically ordered sequences given to
    :func:`match_greedy`. ``matches`` pairs ``(prediction_index,
    ground_truth_index)`` in prediction order.
    """

    iou_threshold: float
    class_aware: bool
    matches: tuple[tuple[int, int], ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_ground_truth: tuple[int, ...]


def match_greedy(
    predictions: Sequence[Prediction],
    ground_truth: Sequence[UnifiedObject],
    *,
    iou_threshold: float,
    class_aware: bool = True,
) -> MatchResult:
    """Greedy score-ordered one-to-one matching (see module docstring).

    ``predictions`` / ``ground_truth`` must already be in canonical order.
    """

    # Candidate ground truth indexed by image (and class when class-aware).
    by_key: dict[tuple[str, str], list[int]] = {}
    for index, obj in enumerate(ground_truth):
        key = (obj.image_path, obj.label.value if class_aware else "")
        by_key.setdefault(key, []).append(index)

    claimed: set[int] = set()
    matches: list[tuple[int, int]] = []
    unmatched_predictions: list[int] = []
    for p_index, prediction in enumerate(predictions):
        key = (prediction.image_id, prediction.label.value if class_aware else "")
        best_index = -1
        best_iou = 0.0
        for g_index in by_key.get(key, ()):
            if g_index in claimed:
                continue
            overlap = iou(prediction.bbox, ground_truth[g_index].bbox)
            # Strictly-greater keeps IoU ties on the lowest canonical index.
            if overlap >= iou_threshold and overlap > best_iou:
                best_index = g_index
                best_iou = overlap
        if best_index >= 0:
            claimed.add(best_index)
            matches.append((p_index, best_index))
        else:
            unmatched_predictions.append(p_index)

    return MatchResult(
        iou_threshold=iou_threshold,
        class_aware=class_aware,
        matches=tuple(matches),
        unmatched_predictions=tuple(unmatched_predictions),
        unmatched_ground_truth=tuple(
            index for index in range(len(ground_truth)) if index not in claimed
        ),
    )
