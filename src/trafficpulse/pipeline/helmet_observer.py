"""The helmet-observation ``FrameObserver`` (P4-U4).

The first implementation of the P4-U2 :class:`~trafficpulse.pipeline.base.FrameObserver`
hook, and the stage that makes helmet perception a native TrafficPulse capability:

```
Tracker states + frame pixels (per frame)
  -> associate_riders (P4-U4)                       -> Association
  -> extract_head_region (P4-U4)                    -> HeadRegion
  -> quality gate (pre-classification)              -> abstain, or
  -> HelmetClassifier.classify (P4-U2 seam, batched) -> RawHelmetPrediction
  -> build_observation (P4-U4)                      -> HelmetStateObservation
```

It **observes**; it never decides. No violation, no event, no persistence, no
evidence: it accumulates an observation stream that a later unit (P4-U5) reasons
over. Rules remain the sole authority on violations (architecture-review §14/§15).

This module *wires*; it computes nothing itself. Association, crop geometry, the
quality gate, and contract stamping all live in the derivation modules
(``association/riders.py``, ``observations/helmet.py``); classification happens
strictly behind the P4-U2 seam. Consequently this module holds no geometry
constant, no threshold, and **no ML import** -- it depends on the
``HelmetClassifier`` abstraction only, never on a backend, so the stub and the
zero-shot backend are interchangeable here.

One classifier call per frame
-----------------------------
Every rider's crop for a frame is gathered and classified in a **single batched
call** -- the reason the P4-U2 seam takes a sequence. Gated crops are excluded from
the batch entirely, so an unusable crop costs no inference. With zero usable crops
the classifier is never called at all.

Determinism and replay
----------------------
Riders are processed in ``(camera_id, track_id)`` order within each frame, so the
batch composition -- and therefore the output -- never depends on tracker emission
order. Ids are content-derived; there is no wall-clock and no randomness.
:meth:`reset` clears the accumulated stream and taint bookkeeping, so replaying a
stream reproduces an identical result.

Taint handling
--------------
A tainted rider or motorcycle never associates (the association derivation
abstains), so a tainted rider emits no observation at all. The first clean
observation for a rider that had been tainted is flagged as a **taint restart**,
reusing the P1-U4/P2-U2 mechanism verbatim, so the P4-U5 reasoner cannot bridge an
ID-switch discontinuity (§13: tainted tracks may abstain but never confirm).
"""

from collections.abc import Sequence

from ..association.riders import RiderAssociationConfig, associate_riders
from ..classifier.crop import Crop
from ..classifier.interface import HelmetClassifier
from ..classifier.raw import RawHelmetPrediction
from ..contracts import Association, HelmetStateObservation, TrackState
from ..detector.frame import Frame
from ..observations.helmet import (
    HeadRegion,
    HelmetDerivation,
    HelmetObservationConfig,
    build_observation,
    extract_head_region,
    gate_crop,
)


class HelmetFrameObserver:
    """Accumulates ``HelmetStateObservation``s from frames + tracked states.

    Satisfies the P4-U2 ``FrameObserver`` protocol structurally. Construct with an
    injected :class:`~trafficpulse.classifier.interface.HelmetClassifier` (the stub
    in tests, a real backend in production) -- this class never names a backend.

    The accumulated stream is exposed via :meth:`derivation`; nothing is persisted,
    and no event is produced.
    """

    def __init__(
        self,
        *,
        classifier: HelmetClassifier,
        config: HelmetObservationConfig | None = None,
        association_config: RiderAssociationConfig | None = None,
    ) -> None:
        self._classifier = classifier
        self._config = config if config is not None else HelmetObservationConfig()
        self._association_config = association_config
        self._observations: list[HelmetStateObservation] = []
        self._associations: list[Association] = []
        self._abstentions: list[str] = []
        # Riders seen tainted since their last emitted observation; the next clean
        # observation for such a rider is a taint restart.
        self._tainted_since_emit: set[str] = set()
        self._restart_ids: set[str] = set()

    # --- FrameObserver protocol ---------------------------------------------
    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        """Derive this frame's helmet observations; accumulate them internally."""

        for state in states:
            if state.tainted:
                # Mark the identity discontinuity; the association stage abstains.
                self._tainted_since_emit.add(state.track_id)

        associations = associate_riders(states, config=self._association_config)
        if not associations:
            return
        self._associations.extend(associations)

        riders_by_id = {s.track_id: s for s in states}
        # Deterministic batch composition, independent of tracker emission order.
        ordered = sorted(associations, key=lambda a: (a.camera_id, a.subject_track_id))
        # How many riders share each motorcycle this frame (drives the rider slot).
        riders_per_motorcycle: dict[str, int] = {}
        for association in ordered:
            riders_per_motorcycle[association.object_track_id] = (
                riders_per_motorcycle.get(association.object_track_id, 0) + 1
            )

        pending: list[tuple[TrackState, HeadRegion, int]] = []
        gated: list[tuple[TrackState, HeadRegion, int, str]] = []
        for association in ordered:
            rider = riders_by_id.get(association.subject_track_id)
            if rider is None:  # unreachable in practice; never fabricate a rider
                continue
            region = extract_head_region(
                rider.bbox, frame.image, head_fraction=self._config.head_crop.head_fraction
            )
            rider_count = riders_per_motorcycle[association.object_track_id]
            reason = gate_crop(region, config=self._config.head_crop)
            if reason is None:
                pending.append((rider, region, rider_count))
            else:
                gated.append((rider, region, rider_count, reason))
                self._abstentions.append(reason)

        # One batched call per frame; gated crops never reach the classifier.
        predictions: Sequence[RawHelmetPrediction] = ()
        if pending:
            crops = [
                Crop(
                    camera_id=rider.camera_id,
                    frame_index=frame.frame_index,
                    timestamp=rider.timestamp,
                    track_id=rider.track_id,
                    image=region.image,
                )
                for rider, region, _ in pending
            ]
            predictions = self._classifier.classify(crops)

        for (rider, region, rider_count), prediction in zip(pending, predictions, strict=True):
            self._emit(
                build_observation(
                    rider,
                    region=region,
                    prediction=prediction,
                    gate_reason=None,
                    rider_count=rider_count,
                    config=self._config,
                )
            )
        for rider, region, rider_count, reason in gated:
            self._emit(
                build_observation(
                    rider,
                    region=region,
                    prediction=None,
                    gate_reason=reason,
                    rider_count=rider_count,
                    config=self._config,
                )
            )

    def reset(self) -> None:
        """Return the observer to its initial (pre-stream) state for replay."""

        self._observations = []
        self._associations = []
        self._abstentions = []
        self._tainted_since_emit = set()
        self._restart_ids = set()

    # --- accumulated output --------------------------------------------------
    def _emit(self, observation: HelmetStateObservation) -> None:
        if observation.track_id in self._tainted_since_emit:
            self._restart_ids.add(observation.observation_id)
            self._tainted_since_emit.discard(observation.track_id)
        self._observations.append(observation)

    def derivation(self) -> HelmetDerivation:
        """The accumulated observation stream, in the derivation contract's shape.

        Observations are sorted by ``(timestamp, observation_id)`` so the stream is
        a pure function of the frames seen, not of emission order within a frame.
        """

        return HelmetDerivation(
            observations=tuple(
                sorted(self._observations, key=lambda o: (o.timestamp, o.observation_id))
            ),
            taint_restart_ids=frozenset(self._restart_ids),
            abstentions=tuple(self._abstentions),
        )

    def associations(self) -> tuple[Association, ...]:
        """The accumulated rider<->motorcycle links (the rider's bike identity).

        ``HelmetStateObservation.track_id`` names the rider; this is where the
        motorcycle it belongs to lives. P4-U5 joins the two to attribute anything
        to a vehicle. Sorted deterministically.
        """

        return tuple(
            sorted(self._associations, key=lambda a: (a.timestamp, a.association_id))
        )
