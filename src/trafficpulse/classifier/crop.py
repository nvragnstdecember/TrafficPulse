"""The per-crop boundary object consumed by the classifier seam (P4-U2).

``Crop`` is the single image region handed across the classification seam. It
mirrors the detector's :class:`~trafficpulse.detector.frame.Frame` exactly one
level down: it bundles the *identity* a future adapter stamps onto every
``HelmetStateObservation`` (camera id, frame index, timestamp, and the track the
region was cut from) with an **opaque** pixel payload a real classifier runs
inference over.

Why the image is optional and opaque
------------------------------------
This unit is the classification *foundation*: it fixes the seam and performs
**no** inference, model loading, cropping, or preprocessing. The stub classifier
never reads pixels, so tests construct crops without them. Carrying the image
slot now -- typed but unread by this unit's code -- is exactly what lets a real
backend (P4-U3) consume pixels through the **same** ``classify(crops)`` signature
without an API change, mirroring how ``Frame.image`` let RT-DETR slot in behind
the unchanged P1-U6 ``Detector`` seam.

The payload mirrors the ingestion RGB ``uint8`` array shape ``(height, width,
3)``, but this layer treats it as opaque and never inspects, resizes, normalizes,
or copies it.

What is deliberately NOT here
-----------------------------
``Crop`` carries **no** crop metadata (source bounding box, crop height, blur or
quality scores) and **no** rider slot. Head-crop geometry, rider-slot attribution,
and quality gating are P4-U4 concerns that consume this seam; putting them here
would bind the seam to one caller's policy. ``track_id`` is present because it is
*identity* -- the region's provenance -- not policy.

``image`` is excluded from equality and repr (like the ingestion ``FrameRecord``
and the detector ``Frame``) so crops compare by stable identity rather than pixel
content, and so a present array never triggers ambiguous NumPy ``==`` semantics.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Crop:
    """One image region at the classification boundary: identity plus opaque pixels."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    track_id: str
    image: NDArray[np.uint8] | None = field(default=None, compare=False, repr=False)
