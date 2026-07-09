"""The detector-native output type at the integration boundary (P1-U6).

``RawDetection`` is the *framework-neutral* per-object output a detector emits. A
real detector (a future RT-DETR unit) converts its own tensors / framework
objects into ``RawDetection`` values **inside** its ``detect()`` implementation;
those tensors never escape the detector. The adapter then converts
``RawDetection`` values into the frozen U2 ``Detection`` contract. This is the
seam ADR-001 requires: "no detector-specific type may leak past that seam into
tracking, observations, rules, events, or evidence."

The fields are deliberately loose so the *adapter* -- not the caller -- owns
validation:

* ``label`` is the detector's own class string, mapped to the frozen
  ``ObjectClass`` via the configured label map (an unmapped label is an unmodeled
  class and is dropped);
* ``score`` may be any float (the adapter rejects non-finite or out-of-``[0, 1]``
  values as malformed);
* ``box`` is a native ``(x1, y1, x2, y2)`` pixel tuple in the U2/U5 image-space
  convention (top-left origin, ``+x`` right, ``+y`` down); the adapter rejects
  boxes the ``BoundingBox`` contract would reject.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDetection:
    """One detector-native object detection (pre-adaptation, unvalidated)."""

    label: str
    score: float
    box: tuple[float, float, float, float]
