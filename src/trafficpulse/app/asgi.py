"""ASGI entry point for the application API (H7A).

Exposes ``app`` -- a FastAPI application built from environment configuration --
so any ASGI server can serve it, e.g.::

    uvicorn trafficpulse.app.asgi:app --host 0.0.0.0 --port 8000

Building the app imports no ML framework and loads no model: the real RT-DETR
backend is constructed lazily, only when a processing job is actually submitted.
Host and port live in :class:`~trafficpulse.app.config.AppConfig` (read from the
``TRAFFICPULSE_APP_*`` variables) for the operator's launch command.
"""

from __future__ import annotations

from .api import create_app
from .config import AppConfig

app = create_app(AppConfig.from_env())
