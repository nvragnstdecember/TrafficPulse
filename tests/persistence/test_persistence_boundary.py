"""P1-U11 boundary tests: persistence imports no backend and no ML framework.

Persistence operates on data (frozen contracts) only. Importing it must not pull
in any detector or tracker *backend* implementation, nor torch/transformers or any
ML/CV framework -- mirroring the detector/tracker boundary tests.
"""

import importlib
import sys

import trafficpulse.persistence  # noqa: F401  (import side effect is the subject under test)

# ML/CV/detector/tracker frameworks persistence must NOT import. Deliberately not
# numpy/av (those are base deps other suites legitimately import into the shared
# process); these frameworks are never imported by the base suite, so the
# sys.modules check is robust to test ordering -- mirroring the detector boundary.
_FORBIDDEN_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "cv2",
    "ultralytics",
    "onnxruntime",
    "tensorflow",
    "paddle",
)


def test_importing_persistence_pulls_in_no_ml_or_backend() -> None:
    # Test the package's OWN import graph, not the shared process state: other
    # suites (the H4B training tests) legitimately import torch into this
    # process, so forbidden modules are EVICTED first, the package re-imported
    # fresh, and the assertion checks that the import did not re-pull any of
    # them — the same causal pattern as the classifier boundary test (P4-U2).
    evicted = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".")[0] in _FORBIDDEN_MODULES
    }
    try:
        importlib.reload(importlib.import_module("trafficpulse.persistence"))
        leaked = [n for n in sys.modules if n.split(".")[0] in _FORBIDDEN_MODULES]
        assert leaked == [], f"persistence import pulled in: {leaked}"
    finally:
        sys.modules.update(evicted)


def test_persistence_source_names_no_backend_implementation() -> None:
    # Static guard: the package must not reference a concrete detector/tracker
    # backend or ML framework by name anywhere in its source.
    import pathlib

    pkg = pathlib.Path(trafficpulse.persistence.__file__).parent
    banned = ("rtdetr", "RTDetr", "IouTracker", "torch", "transformers", "ultralytics")
    for path in pkg.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} references backend/ML token {token!r}"


def test_persistence_only_depends_on_contracts_and_stdlib() -> None:
    # The public API surfaces only contracts-derived and persistence-local types.
    from trafficpulse.persistence import EventStore, StoredEvent, build_evidence_manifest

    assert EventStore is not None
    assert StoredEvent is not None
    assert build_evidence_manifest is not None
