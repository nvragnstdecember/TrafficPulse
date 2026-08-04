"""Import shims for the app tests: pipeline fixtures, and the root launcher.

The application API tests drive real (stub-detector) H6 engines over the same
synthetic clips the pipeline/engine tests use, so their uniquely-named fixture
modules (``_pipeline_helpers``, ``_slice_fixtures``, ``_stopping_fixtures``,
``_helmet_fixtures``) are reused rather than duplicated. This conftest puts that
directory on ``sys.path`` for the tests in this directory only -- the same shim
``tests/engine`` and ``tests/experiments`` use.

The repository **root** is added for the same reason and in the same way (H16).
``serve.py`` is a top-level module by design: it is the documented production
entrypoint, run as ``uvicorn serve:app`` from the repository root (and from
``/app`` in the container), so it deliberately does not live inside the installed
``trafficpulse`` package. That makes it importable only when the root is on the
path -- which ``python -m pytest`` provides for free by injecting the working
directory, but the bare ``pytest`` console script CI runs does **not**. The
launcher composition tests therefore passed locally and failed in CI. Adding the
root here fixes that at the layer the repository already uses for exactly this
problem, without moving the launcher or widening the path for unrelated tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TESTS_ROOT.parent

for _path in (_TESTS_ROOT / "pipeline", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
