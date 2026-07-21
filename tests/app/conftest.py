"""Make the ``tests/pipeline`` fixture helpers importable to the app tests.

The application API tests drive real (stub-detector) H6 engines over the same
synthetic clips the pipeline/engine tests use, so their uniquely-named fixture
modules (``_pipeline_helpers``, ``_slice_fixtures``, ``_stopping_fixtures``,
``_helmet_fixtures``) are reused rather than duplicated. This conftest puts that
directory on ``sys.path`` for the tests in this directory only -- the same shim
``tests/engine`` and ``tests/experiments`` use.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PIPELINE_TESTS = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_TESTS) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_TESTS))
