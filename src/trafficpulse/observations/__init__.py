"""Observation derivation for TrafficPulse (perception->reasoning boundary).

First appears in P1-U4 with the wrong-way heading-vs-lane derivation. This layer
turns geometry + configured scene data into the frozen U2 ``Observation``
contracts that the reasoning layer consumes; it performs no detection, tracking,
or rule reasoning.
"""

from .crossing import (
    DEFAULT_CROSSING_PRODUCER,
    CrossingDerivation,
    derive_crossing_observations,
    derive_crossing_observations_with_taint,
)
from .heading import (
    DEFAULT_PRODUCER,
    HeadingDerivation,
    derive_heading_observations,
    derive_heading_observations_with_taint,
)
from .helmet import (
    DEFAULT_HEAD_FRACTION,
    DEFAULT_HELMET_LABEL_MAP,
    DEFAULT_HELMET_PRODUCER,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    HeadCropConfig,
    HeadRegion,
    HelmetDerivation,
    HelmetObservationConfig,
    build_observation,
    extract_head_region,
    gate_crop,
    head_region_box,
    head_regions_for,
    laplacian_variance,
    observation_id,
    rider_slot,
)
from .rider_count import (
    DEFAULT_RIDER_COUNT_PRODUCER,
    RiderCountDerivation,
    derive_rider_count_observations,
)
from .signal import (
    DEFAULT_SIGNAL_PRODUCER,
    SignalPhase,
    derive_signal_state_observations,
    signal_state_at,
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
    "DEFAULT_CROSSING_PRODUCER",
    "CrossingDerivation",
    "derive_crossing_observations",
    "derive_crossing_observations_with_taint",
    "DEFAULT_PRODUCER",
    "HeadingDerivation",
    "derive_heading_observations",
    "derive_heading_observations_with_taint",
    "DEFAULT_HEAD_FRACTION",
    "DEFAULT_HELMET_LABEL_MAP",
    "DEFAULT_HELMET_PRODUCER",
    "DEFAULT_MIN_CROP_HEIGHT_PX",
    "HeadCropConfig",
    "HeadRegion",
    "HelmetDerivation",
    "HelmetObservationConfig",
    "build_observation",
    "extract_head_region",
    "gate_crop",
    "head_region_box",
    "head_regions_for",
    "laplacian_variance",
    "observation_id",
    "rider_slot",
    "DEFAULT_RIDER_COUNT_PRODUCER",
    "RiderCountDerivation",
    "derive_rider_count_observations",
    "DEFAULT_SIGNAL_PRODUCER",
    "SignalPhase",
    "derive_signal_state_observations",
    "signal_state_at",
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
