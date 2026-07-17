"""Helmet-state observation derivation (P4-U4).

Turns a rider's ``TrackState`` + the frame's pixels + a
``RawHelmetPrediction`` into a frozen U2 ``HelmetStateObservation``. This is
*observation derivation*, not reasoning: it produces a per-frame fact about what
was seen and makes no violation, persistence, exemption, or confirmation
decision. ``turban -> exempt`` and ``uncertain -> abstain`` are rule-layer
mappings (P4-U5), deliberately not applied here.

This module is **pure and pixel-local**: it holds no model, calls no classifier,
and imports no ML framework. It computes crop geometry, measures crop quality,
and stamps contracts. The classifier is invoked by the pipeline observer, which
wires this module to the P4-U2 seam.

Head-region geometry: containment over tightness (evidence-based)
-----------------------------------------------------------------
The head region is the **top ``head_fraction`` of the rider's bounding box, at
full box width**, clipped to the frame. Full width is a deliberate choice, not
laziness:

P4-U1's real-footage evidence (``demo/gate0_rtdetr_validation.py``, New Delhi
clip) showed a rider whose person box was ~440x483px with the head ~65x75px sitting
**left of the box's horizontal centre** -- the rider was leaning toward the
handlebars. A narrow, centre-anchored crop would have *missed that head entirely*.
Because the detector gives a person box and nothing finer, and no head detector
exists in the project (``ObjectClass`` has no ``head`` member, by design), the only
honest options are a wide region that reliably **contains** the head, or a tight
region that sometimes **misses** it. A miss is unrecoverable and silently wrong; a
loose crop is merely noisy, and the noise is visible to the classifier's own
confidence. We choose containment.

The cost is stated plainly: the region contains substantial torso and background,
so a classifier sees the head as a minority of the crop. This is a **known
weakness of this geometry**, expected to be the dominant error source for the
zero-shot backend, and the first thing to revisit when either a head detector or
pose keypoints exist. It is isolated here, behind named configuration, precisely
so it can be replaced without touching anything else.

Quality gating: how uncertainty is expressed
--------------------------------------------
The frozen four-label ontology already defines ``uncertain`` as covering *"blur,
occlusion, tiny crop, truncation, poor illumination, or ambiguous headgear"*
(``configs/ontology.yaml``), so a failed quality gate **is** ``HelmetState.UNCERTAIN``
-- no contract change, no new field, no parallel status channel. Quality travels
with the observation as the triple (``helmet_state``, ``confidence``,
``crop_height_px``).

Confidence is never fabricated. The gates split by *when* they fire:

* **Pre-classification gates** (missing pixels, degenerate/off-frame region, crop
  too small, too blurred): the classifier is never run, so there is no score to
  report and ``confidence`` is ``None``. ``None`` honestly means "not measured" --
  it is never coerced to ``0.0``, which would be a fabricated measurement.
* **Post-classification gate** (the classifier ran but scored below
  ``min_confidence``): the observation reports ``UNCERTAIN`` with the classifier's
  **real** score. The score is not discarded and not rounded up.

An uncertain observation is never silently upgraded: a label only survives when
its crop cleared every gate *and* its score cleared the floor.

Rider slots: honest limits
--------------------------
``rider_slot`` is ``DRIVER`` when exactly one rider is associated with a
motorcycle (there is no pillion without a driver), and ``UNKNOWN`` otherwise.
Distinguishing driver from pillion requires knowing which end of the bike is the
front, which requires the bike's travel direction -- and the shipped ``IouTracker``
is explicit that it supplies **no velocity** ("a greedy matched-box associator has
no Kalman/motion state and thus no interpretable velocity"). Rather than guess an
ordering from image-space position (which inverts with travel direction), this
derivation reports ``UNKNOWN``, the value the frozen ``RiderSlot`` enum provides
for exactly this case. A motion-capable tracker (ByteTrack/OC-SORT) or temporal
heading derivation would lift this.

Determinism
-----------
Output is a pure function of the inputs. No wall-clock, no randomness; ids are
content-derived; the image is read but never mutated or copied beyond the crop
slice. Blur is measured with a fixed integer-kernel Laplacian over a deterministic
grayscale reduction.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..classifier.raw import RawHelmetPrediction
from ..contracts import HelmetStateObservation, Producer, TrackState
from ..contracts.enums import HelmetState, ProducerKind, RiderSlot
from ..contracts.primitives import BoundingBox

DEFAULT_HELMET_PRODUCER = Producer(
    name="helmet-state", version="0.1.0-provisional", kind=ProducerKind.MODEL
)

# The head region as a fraction of the rider box's height. Provisional: see the
# module docstring's containment-over-tightness rationale.
DEFAULT_HEAD_FRACTION = 0.30

# Below this head-region height a crop carries essentially no helmet signal.
# Provisional: P4-U1 measured a median rider head region of ~30px and a minimum of
# ~10px on real footage, so this floor is *informed* by evidence but is not tuned
# on held-out data.
DEFAULT_MIN_CROP_HEIGHT_PX = 12.0

# Native classifier label -> frozen HelmetState. Mirrors DetectorConfig.label_map:
# each backend has its own vocabulary and an unmapped label must never be guessed.
# The stub's and the zero-shot backend's default vocabularies both use these
# spellings; another backend maps its own here.
DEFAULT_HELMET_LABEL_MAP: dict[str, HelmetState] = {
    "helmet": HelmetState.HELMET,
    "no_helmet": HelmetState.NO_HELMET,
    "turban": HelmetState.TURBAN,
    "uncertain": HelmetState.UNCERTAIN,
}


class HeadCropConfig(BaseModel):
    """Head-region geometry and crop-quality policy (frozen + strict)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    head_fraction: float = Field(default=DEFAULT_HEAD_FRACTION, gt=0.0, le=1.0)
    """Fraction of the rider box's height taken as the head region, from the top.
    **Provisional**; see the module docstring."""

    min_crop_height_px: float = Field(default=DEFAULT_MIN_CROP_HEIGHT_PX, ge=0.0)
    """Head regions shorter than this abstain (``UNCERTAIN``, confidence ``None``)
    without running the classifier. **Provisional**."""

    min_blur_variance: float | None = None
    """Variance-of-Laplacian floor; a crop below it abstains as too blurred.
    ``None`` (the default) **disables** the gate.

    Deliberately unset: a blur threshold is resolution- and scene-dependent and
    the project has no held-out data to calibrate it against. Shipping a guessed
    value would silently discard good crops -- a false-negative source invisible in
    the output. This mirrors the repository's ``status: unset`` discipline for
    rule parameters (``configs/scenes/example-scene.yaml``): the mechanism exists,
    the value waits for evidence."""


class HelmetObservationConfig(BaseModel):
    """Policy for turning classifier output into helmet observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_map: dict[str, HelmetState] = Field(
        default_factory=lambda: dict(DEFAULT_HELMET_LABEL_MAP)
    )
    """Native classifier label -> frozen ``HelmetState``. A label absent from this
    map abstains (``UNCERTAIN``) rather than being guessed -- the P4-U1 ``motorbike``
    lesson: an unmapped vocabulary must fail loudly in the data, never silently."""

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    """Classifier scores below this floor abstain (``UNCERTAIN``), keeping the
    **real** score on the observation. Default ``0.0`` gates nothing (pure
    pass-through). **Provisional** when set: no held-out data exists to tune it."""

    head_crop: HeadCropConfig = HeadCropConfig()
    """Head-region geometry + crop-quality policy."""

    producer: Producer = DEFAULT_HELMET_PRODUCER
    """Observation provenance stamped on every emitted observation."""


@dataclass(frozen=True)
class HeadRegion:
    """A resolved head region: the pixel box, its height, and its crop.

    ``image`` is ``None`` when the region could not be cut (no frame pixels, or a
    degenerate/off-frame box). ``height_px`` is the region's height in the frame's
    pixel space and is reported even when the crop itself is unusable -- it is the
    quality datum the frozen contract carries.
    """

    box: BoundingBox | None
    height_px: float
    image: NDArray[np.uint8] | None


def head_region_box(rider_bbox: BoundingBox, *, head_fraction: float) -> BoundingBox | None:
    """The head region of a rider box: its top ``head_fraction``, at full width.

    Returns ``None`` if the resulting box would be degenerate (zero height), which
    the frozen ``BoundingBox`` contract rejects. See the module docstring for why
    the region spans the full box width.
    """

    height = (rider_bbox.y2 - rider_bbox.y1) * head_fraction
    y2 = rider_bbox.y1 + height
    if y2 <= rider_bbox.y1:
        return None
    return BoundingBox(x1=rider_bbox.x1, y1=rider_bbox.y1, x2=rider_bbox.x2, y2=y2)


def extract_head_region(
    rider_bbox: BoundingBox,
    image: NDArray[np.uint8] | None,
    *,
    head_fraction: float,
) -> HeadRegion:
    """Cut the head region from ``image`` (clipped to the frame).

    The box is clipped to the image rectangle before slicing, because a detector
    box may extend past the frame edge. A region with no in-frame area yields a
    ``None`` crop. The slice is a NumPy **view**: the frame is never mutated or
    copied.
    """

    box = head_region_box(rider_bbox, head_fraction=head_fraction)
    if box is None:
        return HeadRegion(box=None, height_px=0.0, image=None)
    height_px = box.y2 - box.y1
    if image is None:
        return HeadRegion(box=box, height_px=height_px, image=None)

    frame_height, frame_width = int(image.shape[0]), int(image.shape[1])
    x1 = max(0, int(np.floor(box.x1)))
    y1 = max(0, int(np.floor(box.y1)))
    x2 = min(frame_width, int(np.ceil(box.x2)))
    y2 = min(frame_height, int(np.ceil(box.y2)))
    if x2 <= x1 or y2 <= y1:  # entirely off-frame
        return HeadRegion(box=box, height_px=height_px, image=None)
    return HeadRegion(box=box, height_px=height_px, image=image[y1:y2, x1:x2])


def laplacian_variance(crop: NDArray[np.uint8]) -> float:
    """Focus measure: the variance of a 4-neighbour Laplacian over grayscale.

    A standard, dependency-free sharpness proxy: a blurred image has little
    high-frequency content, so its Laplacian response has low variance. Grayscale
    is an unweighted channel mean (deterministic, and the weighting is irrelevant
    to a variance-based focus measure). Returns ``0.0`` for a crop too small to
    convolve (which the size gate has normally already rejected).
    """

    gray = crop.astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def rider_slot(rider_count: int) -> RiderSlot:
    """The rider's slot, or ``UNKNOWN`` when it cannot be honestly determined.

    A lone rider on a motorcycle is its driver. With two or more riders, telling
    driver from pillion needs the bike's travel direction, which the shipped
    tracker does not provide (no velocity). See the module docstring.
    """

    return RiderSlot.DRIVER if rider_count == 1 else RiderSlot.UNKNOWN


def observation_id(camera_id: str, rider_track_id: str, iso_timestamp: str) -> str:
    """Deterministic, content-derived observation id (no wall-clock, no counter)."""

    preimage = "\x1f".join((camera_id, rider_track_id, iso_timestamp))
    return "hlm-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def gate_crop(region: HeadRegion, *, config: HeadCropConfig) -> str | None:
    """Pre-classification quality gate; return a reason to abstain, else ``None``.

    Fires **before** the classifier runs, so a rejected crop costs no inference and
    yields no score to report (the caller records ``confidence=None``). The returned
    reason is diagnostic only: the frozen contract has no field for it, and the
    observation expresses the outcome as ``UNCERTAIN`` per the U3 ontology.
    """

    if region.box is None:
        return "degenerate head region (zero height)"
    if region.image is None:
        return "no pixels for the head region (missing image or entirely off-frame)"
    if region.height_px < config.min_crop_height_px:
        return (
            f"head crop {region.height_px:.1f}px is below the "
            f"{config.min_crop_height_px:.1f}px floor"
        )
    if config.min_blur_variance is not None:
        variance = laplacian_variance(region.image)
        if variance < config.min_blur_variance:
            return f"head crop is too blurred (laplacian variance {variance:.1f})"
    return None


@dataclass(frozen=True)
class HelmetDerivation:
    """Emitted observations plus the ids of those resuming after taint.

    ``taint_restart_ids`` mirrors the P1-U4/P2-U2 derivation contract so a later
    reasoner can never bridge an ID-switch discontinuity. ``abstentions`` counts
    gated crops by reason -- **diagnostics only**, never persisted and never read by
    reasoning; the frozen contract carries no quality-reason field, so this exists
    so an operator can see *why* a clip produced no usable evidence.
    """

    observations: tuple[HelmetStateObservation, ...]
    taint_restart_ids: frozenset[str]
    abstentions: tuple[str, ...] = ()


def build_observation(
    rider: TrackState,
    *,
    region: HeadRegion,
    prediction: RawHelmetPrediction | None,
    gate_reason: str | None,
    rider_count: int,
    config: HelmetObservationConfig,
) -> HelmetStateObservation:
    """Stamp one frozen ``HelmetStateObservation`` for one rider at one frame.

    ``prediction`` is ``None`` exactly when ``gate_reason`` is set (the classifier
    was never run), in which case the observation is ``UNCERTAIN`` with
    ``confidence=None`` -- "not measured", never a fabricated ``0.0``. Otherwise the
    prediction's native label is mapped through ``config.label_map`` and the real
    score is carried; an unmapped label or a sub-floor score abstains while keeping
    that real score.
    """

    if gate_reason is not None or prediction is None:
        state = HelmetState.UNCERTAIN
        confidence: float | None = None  # never measured -> never fabricated
    else:
        score = max(0.0, min(1.0, prediction.score))
        mapped = config.label_map.get(prediction.label)
        # Abstain on an unmapped vocabulary or a sub-floor score -- but keep the
        # real score either way: uncertainty never discards the measurement that
        # produced it.
        state = (
            mapped
            if mapped is not None and score >= config.min_confidence
            else HelmetState.UNCERTAIN
        )
        confidence = score

    return HelmetStateObservation(
        observation_id=observation_id(
            rider.camera_id, rider.track_id, rider.timestamp.isoformat()
        ),
        camera_id=rider.camera_id,
        track_id=rider.track_id,  # the RIDER; the motorcycle link lives in Association
        timestamp=rider.timestamp,
        confidence=confidence,
        producer=config.producer,
        helmet_state=state,
        rider_slot=rider_slot(rider_count),
        crop_height_px=region.height_px,
    )


def head_regions_for(
    riders: Sequence[TrackState],
    image: NDArray[np.uint8] | None,
    *,
    config: HeadCropConfig,
) -> tuple[HeadRegion, ...]:
    """Resolve each rider's head region, in input order (a convenience for callers)."""

    return tuple(
        extract_head_region(rider.bbox, image, head_fraction=config.head_fraction)
        for rider in riders
    )
