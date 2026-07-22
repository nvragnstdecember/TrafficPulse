"""A deterministic stub helmet classifier for tests and wiring (P4-U2).

``StubHelmetClassifier`` implements the ``HelmetClassifier`` interface without any
inference, model, weights, dataset, or framework dependency: it *replays* a
caller-supplied script of ``RawHelmetPrediction`` values keyed by crop identity.
It exists so the classification seam, the ``FrameObserver`` hook, and every future
helmet-consuming layer (observations P4-U4, reasoning P4-U5) can be exercised
deterministically -- and fully offline -- before any real model or licence-cleared
dataset exists. It ignores ``crop.image`` entirely.

This is the ``StubDetector`` / ``StubTracker`` pattern applied to classification,
and it is what lets the entire no-helmet reasoning path be built and tested while
the real model remains gated on dataset/licence resolution.

Script resolution (most specific wins)
--------------------------------------
For each crop, in order:

1. ``per_crop[(frame_index, track_id)]`` -- a specific rider at a specific frame,
   for temporal scripts (e.g. a rider whose state changes mid-run);
2. ``per_track[track_id]`` -- a constant label for one rider across the whole clip,
   the common case (a rider is bare-headed for the entire pass);
3. ``default`` -- everything else.

``default`` defaults to ``uncertain`` at full confidence, so an unscripted crop
**abstains** rather than silently asserting a helmet state. That mirrors the U3
ontology principle -- *"prefer an abstention/unknown label over guessing when
evidence is weak"* -- and means a test that forgets to script a rider fails by
abstaining, never by fabricating a violation.

Native vocabulary
-----------------
The stub's ``label`` strings happen to match the U3 ontology ids (``"helmet"``,
``"no_helmet"``, ``"turban"``, ``"uncertain"``) because that is the most legible
choice for a scripted double. This is a **convenience of the stub, not a property
of the seam**: ``RawHelmetPrediction.label`` is a backend-native string, and every
backend's vocabulary is mapped explicitly by the P4-U4 adapter. Do not read the
stub's spelling as a required vocabulary (P4-U1's ``"motorbike"`` finding is the
standing reminder that vocabularies differ).

Determinism: ``classify`` is a pure function of the script and crop identity -- no
wall-clock, no randomness, no per-call mutable state. The returned sequence is an
immutable tuple, so a caller cannot mutate the stub's script through the result.
"""

from collections.abc import Mapping, Sequence

from .crop import Crop
from .interface import HelmetClassifier
from .raw import RawHelmetPrediction

# An unscripted crop abstains rather than guessing (see module docstring).
UNCERTAIN = RawHelmetPrediction(label="uncertain", score=1.0)


class StubHelmetClassifier(HelmetClassifier):
    """A ``HelmetClassifier`` that replays scripted predictions per crop identity."""

    def __init__(
        self,
        default: RawHelmetPrediction = UNCERTAIN,
        *,
        per_track: Mapping[str, RawHelmetPrediction] | None = None,
        per_crop: Mapping[tuple[int, str], RawHelmetPrediction] | None = None,
    ) -> None:
        self._default = default
        self._per_track: dict[str, RawHelmetPrediction] = dict(per_track or {})
        self._per_crop: dict[tuple[int, str], RawHelmetPrediction] = dict(per_crop or {})

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        """Return one scripted prediction per crop, in input order."""

        return tuple(self._for(crop) for crop in crops)

    def _for(self, crop: Crop) -> RawHelmetPrediction:
        specific = self._per_crop.get((crop.frame_index, crop.track_id))
        if specific is not None:
            return specific
        return self._per_track.get(crop.track_id, self._default)
