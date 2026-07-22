"""The classifier-native output type at the integration boundary (P4-U2).

``RawHelmetPrediction`` is the *framework-neutral* per-crop output a helmet
classifier emits. A real backend (P4-U3) converts its own tensors / framework
objects into ``RawHelmetPrediction`` values **inside** its ``classify()``
implementation; those tensors never escape the backend. A future adapter (P4-U4)
converts them into the frozen U2 ``HelmetStateObservation`` contract. This is the
same seam discipline ADR-001 requires of the detector: no classifier-specific
type may leak past this boundary into observations, rules, events, or evidence.

Why ``label`` is a plain string, not a ``HelmetState``
-----------------------------------------------------
It mirrors ``RawDetection.label`` deliberately. Each backend has its **own** class
vocabulary, and mapping that vocabulary onto the frozen four-label ontology is a
*configuration* decision, not a backend decision -- so the backend stays free of
TrafficPulse semantics and a new model is a config change, not a code change.

P4-U1 proved this is not hypothetical: ``PekingU/rtdetr_r50vd`` emits the native
label ``"motorbike"`` where COCO says ``"motorcycle"``, and the adapter silently
drops unmapped labels. Vocabulary mismatch between a model and this project is a
*normal* condition to be mapped explicitly at a seam, never assumed away. A
backend that emitted ``HelmetState`` directly would bury that mapping -- and its
failure mode -- inside model code.

The fields are deliberately loose so the *adapter* -- not the backend, and not the
caller -- owns validation:

* ``label`` is the classifier's own class string, mapped to the frozen
  ``HelmetState`` via a configured label map (P4-U4);
* ``score`` may be any float (the adapter rejects non-finite or out-of-``[0, 1]``
  values as malformed).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RawHelmetPrediction:
    """One classifier-native helmet prediction (pre-adaptation, unvalidated)."""

    label: str
    score: float
