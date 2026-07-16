"""Guards for the viewer's upload detector/calibration configuration (P4-U1).

The viewer is a demonstration layer and is otherwise untested by design (it adds
no reasoning of its own). These tests exist because P4-U1 (Gate 0) put two pieces
of *load-bearing configuration* there, each of which fails **silently** when wrong:

* the upload **label map** decides which native detector labels ever become
  ``ObjectClass`` values. An unmapped label is dropped by the adapter without an
  error (P1-U6 behaviour), so a wrong spelling detects **zero** objects of that
  class and every downstream Phase 4 unit reasons over nothing. Gate 0 found
  exactly this: ``PekingU/rtdetr_r50vd`` emits the native label ``"motorbike"``,
  not ``"motorcycle"``.
* ``FLOW_CLASSES`` keeps pedestrians out of the dominant-flow estimate that every
  wrong-way upload calibrates against.

These are cheap structural assertions over declarative configuration -- no ML, no
network, no checkpoint. Whether the *checkpoint* really emits ``"motorbike"`` is a
fact about the artifact, verifiable only by real inference
(``demo/gate0_rtdetr_validation.py``), not by this test.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "viewer") not in sys.path:  # viewer/ is a script dir, not a package
    sys.path.insert(0, str(_REPO_ROOT / "viewer"))

from calibration import (  # noqa: E402
    FLOW_CLASSES,
    default_upload_detector_config,
)

from trafficpulse.contracts import ModelRef, ObjectClass  # noqa: E402

_MODEL_REF = ModelRef(name="rtdetr", version="test")


def test_upload_label_map_covers_the_phase4_classes() -> None:
    label_map = default_upload_detector_config(_MODEL_REF).label_map
    assert label_map["car"] is ObjectClass.CAR
    assert label_map["person"] is ObjectClass.PERSON


def test_upload_label_map_maps_both_motorcycle_spellings() -> None:
    """Regression guard for the Gate 0 finding (silent zero-detection blind spot)."""

    label_map = default_upload_detector_config(_MODEL_REF).label_map
    assert label_map["motorbike"] is ObjectClass.MOTORCYCLE
    assert label_map["motorcycle"] is ObjectClass.MOTORCYCLE


def test_upload_flow_estimate_excludes_pedestrians() -> None:
    """Persons are detected (helmet reasoning needs riders) but never define flow."""

    assert ObjectClass.PERSON not in FLOW_CLASSES
    assert ObjectClass.CAR in FLOW_CLASSES
    assert ObjectClass.MOTORCYCLE in FLOW_CLASSES


def test_upload_detector_config_stamps_provenance() -> None:
    config = default_upload_detector_config(_MODEL_REF)
    assert config.source_model == _MODEL_REF
    assert config.score_threshold == 0.5
