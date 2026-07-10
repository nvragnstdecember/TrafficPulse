"""Observation derivation for TrafficPulse (perception->reasoning boundary).

First appears in P1-U4 with the wrong-way heading-vs-lane derivation. This layer
turns geometry + configured scene data into the frozen U2 ``Observation``
contracts that the reasoning layer consumes; it performs no detection, tracking,
or rule reasoning.
"""

from .heading import (
    DEFAULT_PRODUCER,
    HeadingDerivation,
    derive_heading_observations,
    derive_heading_observations_with_taint,
)
from .stationary import (
    DEFAULT_STATIONARY_PRODUCER,
    STATIONARY_EPSILON_PX,
    STATIONARY_WINDOW,
    StationaryDerivation,
    derive_stationary_observations,
    derive_stationary_observations_with_taint,
)
from .zones import (
    DEFAULT_IN_ZONE_PRODUCER,
    InZoneDerivation,
    derive_in_zone_observations,
    derive_in_zone_observations_with_taint,
)

__all__ = [
    "DEFAULT_PRODUCER",
    "HeadingDerivation",
    "derive_heading_observations",
    "derive_heading_observations_with_taint",
    "DEFAULT_IN_ZONE_PRODUCER",
    "InZoneDerivation",
    "derive_in_zone_observations",
    "derive_in_zone_observations_with_taint",
    "DEFAULT_STATIONARY_PRODUCER",
    "STATIONARY_EPSILON_PX",
    "STATIONARY_WINDOW",
    "StationaryDerivation",
    "derive_stationary_observations",
    "derive_stationary_observations_with_taint",
]
