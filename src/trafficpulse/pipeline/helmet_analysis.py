"""Helmet **analysis**: perception without enforcement, and the fold that reports it.

Two things live here, and the pairing is the point.

:class:`HelmetAnalysisObserver` is the observer an
:class:`~trafficpulse.engine.config.HelmetAnalysisConfig` declaration builds. It runs the
whole P4-U4 perception chain -- association, head-crop extraction, the quality gate, the
batched classifier call -- by **delegating to the shipped**
:class:`~trafficpulse.pipeline.helmet_observer.HelmetFrameObserver`. It adds no
inference, changes no geometry, and re-implements nothing.

:func:`analyse` folds what that observer accumulated into a
:class:`HelmetAnalysisReport`: per rider track, what was seen, how stable the reading
was, and -- separately -- whether the rider is even *eligible* for a violation decision.

Why a distinct class rather than the observer itself
-----------------------------------------------------
Composition, not inheritance, and for two concrete reasons.

1. **It is a different claim.** A ``no_helmet`` rule's observer feeds a reasoner that
   mints ``ConfirmedEvent``s. This one feeds nothing. Giving the two the same type would
   make "is this run enforcing?" a question about the caller's intent rather than about
   the object, and intent is exactly what gets lost between layers.
2. **The overlay registry dispatches on capture type.** ``OverlayProviderRegistry``
   binds one source type to one violation kind and resolves by ``isinstance``, so a
   *subclass* of ``HelmetFrameObserver`` would silently resolve to the no-helmet
   provider and an analysis run would be drawn as if a violation rule had run. A
   separate type is what lets the analysis overlay say something different -- and say it
   without touching the shipped provider that rule runs depend on.

Separating *what was classified* from *what may be enforced*
-------------------------------------------------------------
``docs/helmet-runtime-evaluation.md`` establishes that a helmet label is not a violation
decision, for reasons that are independent of how good the classifier is:

* driver-versus-pillion attribution does not exist -- ``rider_slot`` is ``UNKNOWN`` for
  every multi-rider motorcycle by design, covering 42.4% of the frozen test corpus and
  81% of crops in a real congestion clip;
* the turban exemption's evidence is not demonstrably real on any shipped backend;
* per-frame labels are unstable on the majority of real tracks.

So :class:`RiderEnforcementStatus` is reported **beside** the classification rather than
folded into it. A rider carries a helmet reading *and* a plain statement of why that
reading is, or is not, something a violation rule could act on. Nothing here decides a
violation; the field exists so a surface can refuse to imply one.

Determinism
-----------
The fold is pure. It reads only values the observation pass already produced, orders
rider tracks by id and frames by timestamp, and holds no clock. Frame *indices* are the
rank of each distinct observation timestamp, which is a faithful ordering for a
single-camera run and is all the temporal stabilizer needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..association.riders import RiderAssociationConfig
from ..classifier.interface import HelmetClassifier
from ..contracts import Association, HelmetStateObservation, TrackState
from ..contracts.enums import HelmetState, RiderSlot
from ..detector.frame import Frame
from ..observations.helmet import HelmetDerivation, HelmetObservationConfig
from ..observations.helmet_stability import (
    HelmetSample,
    HelmetStabilizationConfig,
    StabilizedSample,
    stabilize,
)
from .helmet_observer import HelmetFrameObserver, HelmetOverlayFrame


class RiderEnforcementStatus(StrEnum):
    """Whether a rider's helmet reading is something a violation rule could act on.

    Deliberately **not** a violation outcome and never persisted as one. Every value
    other than :attr:`ELIGIBLE` names a specific, documented reason the project cannot
    presently turn this reading into an enforcement decision -- so a surface can show
    the classification honestly while refusing to imply a violation.

    :attr:`ELIGIBLE` means only "no known blocker applies to *this rider*". It is not a
    statement that the deployment is enforcing anything: whether a violation rule runs
    at all is a separate, run-level fact the application owns.
    """

    ELIGIBLE = "eligible"
    """Single rider on its motorcycle, and a settled, non-abstained reading."""

    MULTI_RIDER_UNRESOLVED = "multi_rider_unresolved"
    """Two or more riders share the motorcycle, so driver attribution is unavailable.
    No heuristic is applied: not front-most, not largest, not lowest, not first-tracked.
    The shipped tracker supplies no velocity, so which end of the bike is the front is
    genuinely unknown, and guessing would attribute a violation to a possibly innocent
    passenger."""

    CLASSIFICATION_ABSTAINED = "classification_abstained"
    """The stabilized reading is ``uncertain`` -- a gated crop (too small, off-frame, no
    pixels) or a sub-floor score. An abstention is an outcome, not a missing value."""

    UNSTABLE = "unstable"
    """Too few samples for the stabilizer to call the track settled. The reading is
    still reported; it is simply not yet supported by enough frames to be leaned on."""


@dataclass(frozen=True)
class RiderAnalysis:
    """One rider track, folded across a run: what was seen, and what may be claimed.

    The classification fields and :attr:`enforcement` are independent by construction.
    A rider can carry a confident ``no_helmet`` reading *and*
    ``MULTI_RIDER_UNRESOLVED``; that combination is the normal case in dense traffic and
    is precisely what must not collapse into a violation.
    """

    rider_track_id: str
    motorcycle_track_id: str | None
    """The motorcycle this rider was last associated with; ``None`` only if the rider
    somehow emitted an observation with no association (not reachable through the
    shipped chain, which abstains rather than emitting an unassociated rider)."""

    rider_count: int
    """The largest number of riders seen sharing this rider's motorcycle on any one
    frame. ``1`` means it was never seen carrying anyone else."""

    multi_rider: bool
    samples: int
    first_frame: int
    last_frame: int

    label: str
    """The **stabilized** label, in the frozen ontology's spelling."""

    confidence: float | None
    agreement: float
    settled: bool
    raw_flips: int
    """How many times the *unsmoothed* label changed between consecutive samples -- the
    P4-U10 instability measurement, taken on this run rather than quoted."""

    stabilized_flips: int
    label_counts: tuple[tuple[str, int], ...]
    median_head_height_px: float | None
    """Median head-crop height for this rider. Below ~30px both trained backends degrade
    sharply (P4-U9 §3.3), so this is the disclosure that says which regime a reading came
    from. ``None`` when no crop height was recorded."""

    enforcement: RiderEnforcementStatus


@dataclass(frozen=True)
class HelmetAnalysisReport:
    """A whole run's helmet perception, aggregated. Emits and implies no violation.

    Counts are of **rider tracks**, not of frames or of crops, because a track is what a
    viewer sees as "a person" -- and mixing the two is how a demo ends up claiming a
    hundred violations from thirty frames of one rider.
    """

    frames_observed: int
    """Frames on which at least one rider was associated and observed."""

    riders_observed: int
    motorcycles_associated: int
    """Distinct motorcycle tracks that carried at least one associated rider. This is
    *not* every motorcycle the detector saw -- a bike whose rider never associated is
    not counted, which is the honest reading of what this stage produced."""

    riders: tuple[RiderAnalysis, ...]
    label_counts: tuple[tuple[str, int], ...]
    """Rider tracks by their final stabilized label, sorted by label."""

    enforcement_counts: tuple[tuple[str, int], ...]
    """Rider tracks by :class:`RiderEnforcementStatus`, sorted by status."""

    multi_rider_riders: int
    multi_rider_motorcycles: int
    gate_abstentions: int
    """Crops the quality gate rejected before any inference ran (too small, off-frame,
    no pixels). Counted per observation, not per track."""

    @property
    def eligible_riders(self) -> int:
        return self._enforcement(RiderEnforcementStatus.ELIGIBLE)

    @property
    def unresolved_riders(self) -> int:
        return self._enforcement(RiderEnforcementStatus.MULTI_RIDER_UNRESOLVED)

    @property
    def abstained_riders(self) -> int:
        return self._enforcement(RiderEnforcementStatus.CLASSIFICATION_ABSTAINED)

    @property
    def unstable_riders(self) -> int:
        return self._enforcement(RiderEnforcementStatus.UNSTABLE)

    def _enforcement(self, status: RiderEnforcementStatus) -> int:
        return next((count for name, count in self.enforcement_counts if name == status), 0)


class HelmetAnalysisObserver:
    """Perception-only helmet observer: classifies, and can never confirm.

    Satisfies the P4-U2 ``FrameObserver`` protocol structurally by delegating to a
    :class:`~trafficpulse.pipeline.helmet_observer.HelmetFrameObserver`, so the
    association, crop geometry, quality gate and batching are byte-for-byte the ones the
    rule path uses -- an analysis run and a rule run see the same pixels and get the same
    labels. What differs is only what may be concluded, and that difference is carried by
    the type rather than by a comment.
    """

    def __init__(
        self,
        *,
        classifier: HelmetClassifier,
        config: HelmetObservationConfig | None = None,
        association_config: RiderAssociationConfig | None = None,
        capture_overlay: bool = False,
        stabilization: HelmetStabilizationConfig | None = None,
    ) -> None:
        self._inner = HelmetFrameObserver(
            classifier=classifier,
            config=config,
            association_config=association_config,
            capture_overlay=capture_overlay,
        )
        self._stabilization = (
            stabilization if stabilization is not None else HelmetStabilizationConfig()
        )

    # --- FrameObserver protocol ---------------------------------------------
    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        self._inner.observe(frame, states)

    def reset(self) -> None:
        self._inner.reset()

    # --- accumulated output --------------------------------------------------
    def derivation(self) -> HelmetDerivation:
        return self._inner.derivation()

    def associations(self) -> tuple[Association, ...]:
        return self._inner.associations()

    def overlay_frames(self) -> tuple[HelmetOverlayFrame, ...]:
        return self._inner.overlay_frames()

    @property
    def stabilization(self) -> HelmetStabilizationConfig:
        """The display-smoothing policy this run was configured with (never tuned)."""

        return self._stabilization

    def report(self) -> HelmetAnalysisReport:
        """Fold this run's observations into a :class:`HelmetAnalysisReport`."""

        return analyse(
            self.derivation(),
            self.associations(),
            stabilization=self._stabilization,
        )


def _frame_ranks(observations: Sequence[HelmetStateObservation]) -> dict[str, int]:
    """Map each distinct observation timestamp to its rank (a stand-in frame index).

    Observations carry a media timestamp, not a frame index. For a single-camera run the
    rank of the distinct timestamps is the frame order exactly, which is all the
    temporal stabilizer needs -- it votes over *consecutive samples of one track*, and
    never over an absolute index.
    """

    stamps = sorted({observation.timestamp.isoformat() for observation in observations})
    return {stamp: rank for rank, stamp in enumerate(stamps)}


def analyse(
    derivation: HelmetDerivation,
    associations: Sequence[Association] = (),
    *,
    stabilization: HelmetStabilizationConfig | None = None,
) -> HelmetAnalysisReport:
    """Fold an accumulated helmet stream into a report. Pure; mints no event.

    ``associations`` supplies the rider -> motorcycle link, which is where the rider
    count comes from. Called without them the fold still works and still reports
    multi-rider status, because ``rider_slot`` already encodes it (``DRIVER`` only ever
    means "exactly one rider on this bike") -- the associations add the bike identity and
    the exact count on top.
    """

    observations = derivation.observations
    ranks = _frame_ranks(observations)

    # Riders sharing one motorcycle at one instant -- the honest rider count, taken from
    # the association stream the observer already produced.
    riders_per_bike_instant: dict[tuple[str, str], set[str]] = {}
    bike_of_rider: dict[str, str] = {}
    for association in associations:
        key = (association.timestamp.isoformat(), association.object_track_id)
        riders_per_bike_instant.setdefault(key, set()).add(association.subject_track_id)
        # Last association wins: sorted by (timestamp, id), so this is the rider's most
        # recent bike rather than an arbitrary one.
        bike_of_rider[association.subject_track_id] = association.object_track_id

    samples: list[HelmetSample] = []
    per_rider_counts: dict[str, int] = {}
    per_rider_heights: dict[str, list[float]] = {}
    for observation in observations:
        track_id = observation.track_id
        if track_id is None:
            # The observation contract allows a scene-level observation with no track,
            # and a per-rider report has nowhere to put one. Skipped rather than given
            # a synthetic id: the helmet observer never emits one (every observation is
            # stamped from a rider's TrackState), so this is a contract-shape guard.
            continue
        stamp = observation.timestamp.isoformat()
        samples.append(
            HelmetSample(
                frame_index=ranks[stamp],
                track_id=track_id,
                label=observation.helmet_state.value,
                confidence=observation.confidence,
            )
        )
        bike = bike_of_rider.get(track_id)
        exact = len(riders_per_bike_instant.get((stamp, bike or ""), ()))
        # Without an association to count against, ``rider_slot`` is still authoritative
        # about single-vs-multi: DRIVER is emitted only for a lone rider.
        count = exact if exact else (1 if observation.rider_slot is RiderSlot.DRIVER else 2)
        per_rider_counts[track_id] = max(per_rider_counts.get(track_id, 0), count)
        if observation.crop_height_px is not None:
            per_rider_heights.setdefault(track_id, []).append(observation.crop_height_px)

    by_track: dict[str, list[StabilizedSample]] = {}
    for entry in stabilize(samples, config=stabilization):
        by_track.setdefault(entry.track_id, []).append(entry)

    riders: list[RiderAnalysis] = []
    for track_id in sorted(by_track):
        entries = by_track[track_id]
        raw_labels = [entry.raw_label for entry in entries]
        counts: dict[str, int] = {}
        for raw_label in raw_labels:
            counts[raw_label] = counts.get(raw_label, 0) + 1
        last = entries[-1]
        rider_count = per_rider_counts.get(track_id, 1)
        multi_rider = rider_count > 1
        heights = sorted(per_rider_heights.get(track_id, ()))
        riders.append(
            RiderAnalysis(
                rider_track_id=track_id,
                motorcycle_track_id=bike_of_rider.get(track_id),
                rider_count=rider_count,
                multi_rider=multi_rider,
                samples=len(entries),
                first_frame=entries[0].frame_index,
                last_frame=last.frame_index,
                label=last.label,
                confidence=last.confidence,
                agreement=last.agreement,
                settled=last.settled,
                raw_flips=sum(
                    1
                    for previous, current in zip(raw_labels, raw_labels[1:], strict=False)
                    if previous != current
                ),
                stabilized_flips=sum(
                    1
                    for previous, current in zip(entries, entries[1:], strict=False)
                    if previous.label != current.label
                ),
                label_counts=tuple(sorted(counts.items())),
                median_head_height_px=heights[len(heights) // 2] if heights else None,
                enforcement=_enforcement_for(
                    multi_rider=multi_rider, label=last.label, settled=last.settled
                ),
            )
        )

    label_counts: dict[str, int] = {}
    enforcement_counts: dict[str, int] = {}
    multi_bikes: set[str] = set()
    for rider in riders:
        label_counts[rider.label] = label_counts.get(rider.label, 0) + 1
        enforcement_counts[rider.enforcement.value] = (
            enforcement_counts.get(rider.enforcement.value, 0) + 1
        )
        if rider.multi_rider and rider.motorcycle_track_id is not None:
            multi_bikes.add(rider.motorcycle_track_id)

    return HelmetAnalysisReport(
        frames_observed=len(ranks),
        riders_observed=len(riders),
        motorcycles_associated=len(set(bike_of_rider.values())),
        riders=tuple(riders),
        label_counts=tuple(sorted(label_counts.items())),
        enforcement_counts=tuple(sorted(enforcement_counts.items())),
        multi_rider_riders=sum(1 for rider in riders if rider.multi_rider),
        multi_rider_motorcycles=len(multi_bikes),
        gate_abstentions=len(derivation.abstentions),
    )


def _enforcement_for(
    *, multi_rider: bool, label: str, settled: bool
) -> RiderEnforcementStatus:
    """Classify one rider's eligibility. Order matters and is deliberate.

    Multi-rider is checked **first** because it is the strongest and most permanent of
    the blockers: no amount of classifier confidence or temporal support makes an
    unattributable rider attributable, so reporting such a rider as merely "unstable"
    would understate why the reading cannot be enforced.
    """

    if multi_rider:
        return RiderEnforcementStatus.MULTI_RIDER_UNRESOLVED
    if label == HelmetState.UNCERTAIN.value:
        return RiderEnforcementStatus.CLASSIFICATION_ABSTAINED
    if not settled:
        return RiderEnforcementStatus.UNSTABLE
    return RiderEnforcementStatus.ELIGIBLE
