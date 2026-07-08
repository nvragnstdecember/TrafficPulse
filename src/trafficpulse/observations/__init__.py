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

__all__ = [
    "DEFAULT_PRODUCER",
    "HeadingDerivation",
    "derive_heading_observations",
    "derive_heading_observations_with_taint",
]
