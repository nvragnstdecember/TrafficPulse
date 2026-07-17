"""Entity-to-entity association derivation for TrafficPulse (Phase 4, P4-U4).

The ``Association`` box of the architecture-review §14 data flow
(``Detection -> TrackState -> Association -> Observation``) -- the one layer of that
flow that had a frozen contract but no implementation until now.

This layer turns frozen ``TrackState`` geometry into frozen U2 ``Association``
contracts. It performs no detection, tracking, classification, observation, or
reasoning, reads no pixels, and carries no ML dependency.

First (and only) derivation: rider <-> motorcycle, needed because helmet state is
meaningful only for a rider. ``head_of_rider`` and ``plate_of_vehicle`` are
members of the frozen ``AssociationType`` enum but are deliberately **not**
implemented here -- they appear when a unit needs them, not before.
"""

from .riders import (
    DEFAULT_MIN_OVERLAP,
    RiderAssociationConfig,
    associate_riders,
    overlap_over_min_area,
)

__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "RiderAssociationConfig",
    "associate_riders",
    "overlap_over_min_area",
]
