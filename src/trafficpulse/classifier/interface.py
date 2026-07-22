"""The abstract helmet-classifier interface -- the DI seam (P4-U2).

``HelmetClassifier`` is the abstraction every helmet-classification implementation
satisfies and that callers depend on instead of a concrete model. Injecting a
``HelmetClassifier`` (the deterministic ``StubHelmetClassifier`` in tests, a real
backend in P4-U3) is what keeps the model choice a bounded, localized change:
downstream code depends on this interface and on the frozen
``HelmetStateObservation`` contract, never on an ML framework.

This is the detector seam (P1-U6) one level down, and it exists for the same
reason ADR-001 gives: *"no detector-specific type may leak past that seam into
tracking, observations, rules, events, or evidence."* The same holds here. A
backend may be a CNN, a ViT, a zero-shot vision-language model, or a detector
cropped to a head region -- the rest of TrafficPulse cannot tell, because only
:class:`~trafficpulse.classifier.crop.Crop` goes in and
:class:`~trafficpulse.classifier.raw.RawHelmetPrediction` comes out.

Licence note: this seam is model-agnostic, but the *repository* is not
licence-agnostic. ADR-001 (Accepted) mandates a permissive-only integrated code
path and excludes AGPL detector/model code (notably ``ultralytics``, which covers
YOLO11/YOLO12 and their trained weights). A backend of that lineage is barred by
the ADR, not by this interface; swapping it in would require a new ADR, not an
edit here. Permissive backends (Apache-2.0 / MIT / BSD) drop in freely.

Batch in, batch out (a deliberate departure from ``Detector.detect``)
--------------------------------------------------------------------
``classify`` takes a **sequence** of crops and returns one prediction per crop,
in the same order. ``Detector.detect`` is single-frame because detection *is*
inherently per-frame; classification is not. A single frame can contain many
riders, so a per-crop call signature would force every real backend to run one
forward pass per rider and would make batching impossible to add later without
changing every implementation and caller.

P4-U1 measured RT-DETR at roughly 1.5 s/frame on CPU for this footage; compounding
that with an un-batchable per-rider call is the difference between a demo that
runs and one that does not. Fixing the shape now costs nothing -- a backend that
cannot batch simply loops internally -- and the seam is the one thing in this
design that is expensive to change later.

The classification contract (stateless, pure, order-preserving)
--------------------------------------------------------------
* **One prediction per crop, in input order.** Implementations must return exactly
  ``len(crops)`` predictions; the caller pairs them positionally. Returning a
  different arity is a programming error, not an abstention -- a backend with no
  opinion about a crop must say so *in the prediction* (its own "uncertain"
  vocabulary), never by omitting it. Silent omission would misalign every
  subsequent pairing.
* **Empty in, empty out.** ``classify(())`` returns ``()`` and must not load a
  model or raise.
* **Stateless across calls.** Unlike ``Tracker``, a classifier holds no temporal
  state and needs no ``reset``: a crop's prediction depends only on that crop.
  Temporal aggregation is the reasoning layer's job (P4-U5), never the model's --
  which is what keeps replay from the observation log model-free.
* **Deterministic.** The same crops must yield the same predictions.
* **Framework-neutral.** Only ``RawHelmetPrediction`` values cross this boundary --
  never tensors, model handles, or framework result objects. Failures raise
  :class:`~trafficpulse.classifier.errors.HelmetClassifierError` subclasses, never
  a framework exception.

This foundation intentionally specifies **no** model, preprocessing, label
vocabulary, quality gate, or crop geometry. The stub replays scripted labels; a
real backend computes them; both satisfy this same seam.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .crop import Crop
from .raw import RawHelmetPrediction


class HelmetClassifier(ABC):
    """Abstract, framework-neutral, stateless helmet-state classifier."""

    @abstractmethod
    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        """Classify each crop; return one prediction per crop, in input order.

        Implementations must be deterministic for a given construction and input,
        must return exactly ``len(crops)`` predictions, must return ``()`` for an
        empty input without loading a model, and must not let framework objects
        escape: only ``RawHelmetPrediction`` values cross this boundary. A real
        backend runs inference over ``crop.image``; the stub ignores pixels and is
        a pure function of crop identity.

        Raises:
            HelmetClassifierError: on any classifier-integration failure. A
                framework-native exception must never escape this seam.
        """
        raise NotImplementedError
