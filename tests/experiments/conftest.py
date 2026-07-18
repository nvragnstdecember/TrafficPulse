"""Make the ``experiments/`` training packages importable to these tests.

The helmet training pipeline lives under ``experiments/`` (dev infrastructure, not
the shipped ``trafficpulse`` runtime), so it is not installed. This conftest puts
``experiments/`` on ``sys.path`` for the tests in this directory only -- the same
shim ``tests/viewer`` uses for the viewer scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))
