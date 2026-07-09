"""The per-frame boundary object consumed by the detector integration (P1-U6).

``Frame`` is the single frame handed across the detection seam. It bundles the
*frame identity* the adapter stamps onto every ``Detection`` (camera id, frame
index, timestamp) with an **optional, opaque** decoded-image payload that a real
detector will run inference over.

Why the image is optional and opaque
------------------------------------
This unit is the detector-integration *foundation*: it fixes the output-
adaptation seam and performs **no** inference, model loading, or image
preprocessing. The stub detector and the adapter never read pixels, so tests
construct frames without them. Carrying the image slot now -- typed but unused by
this unit's code -- is exactly what lets a future RT-DETR detector consume pixels
through the **same** ``Detector.detect(frame)`` signature without an API change
(ADR-001: "configuration should support future RT-DETR integration without
requiring API changes"). The payload mirrors the ingestion RGB ``uint8`` array
shape ``(height, width, 3)``, but this layer treats it as opaque and never
inspects, resizes, normalizes, or copies it.

``image`` is excluded from equality and repr (like the ingestion ``FrameRecord``)
so frames compare by their stable identity rather than by pixel content, and so a
present array never triggers ambiguous NumPy ``==`` semantics.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Frame:
    """One frame at the detection boundary: identity plus an optional opaque image."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    image: NDArray[np.uint8] | None = field(default=None, compare=False, repr=False)
