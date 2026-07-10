"""Vertical-slice orchestration for TrafficPulse (Phase 1, unit P1-U10).

A thin, deterministic **offline** orchestration layer that composes the existing,
independently-tested components into the first wrong-way vertical slice -- it wires
them, it does not re-implement any of them:

```
FrameRecord (P1-U5) -> Detector + DetectionAdapter (P1-U6) -> Detection
  -> Tracker (P1-U8) -> TrackState -> heading derivation (P1-U4)
  -> WrongWayReasoner (P1-U4) -> ConfirmedEvent
```

``WrongWayPipeline`` depends only on the ``Detector`` / ``Tracker`` abstractions,
the frozen U2 contracts, and the existing observation/rule APIs -- never on a
detector or tracker *backend*. Any implementation of the two seams (``StubDetector``
+ ``StubTracker`` in tests, ``RTDetrDetector`` + ``IouTracker`` for a real run)
drops in through the constructor unchanged. Persistence and evidence output belong
to P1-U11 and are intentionally not part of this layer.
"""

from .errors import PipelineError, SceneConfigurationError
from .provenance import normalize_model_refs
from .wrong_way import WrongWayPipeline, frame_record_to_frame

__all__ = [
    "WrongWayPipeline",
    "frame_record_to_frame",
    "normalize_model_refs",
    "PipelineError",
    "SceneConfigurationError",
]
