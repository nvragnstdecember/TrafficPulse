# TrafficPulse production image (H16).
#
# Multi-stage: the SPA is built with Node and copied into a Python runtime that
# serves both it and the API from one process, which is the deployment shape
# docs/deployment.md §6B describes (TRAFFICPULSE_APP_STATIC_DIR).
#
# What this image deliberately does NOT contain
# ---------------------------------------------
# * **No model weights.** ADR-001 makes checkpoint acquisition an operator
#   decision reviewed per artifact; the launcher loads from a local HuggingFace
#   cache with local_files_only=True. Mount that cache at /models (see
#   compose.yaml) to enable real inference.
# * **No CUDA.** This is a CPU-only image, which keeps it small and portable.
#   RT-DETR on CPU is roughly 2-3 s/frame -- correct for review workflows and
#   demos, not for realtime. A GPU deployment starts from a CUDA base image and
#   changes only this file.
# * **No ML extra by default.** `rtdetr` (torch + transformers) is installed via
#   the INSTALL_EXTRAS build argument, because it multiplies the image size and
#   many deployments only need the read/review surface.
#
# Build:  docker build -t trafficpulse .
# Run:    docker run -p 8000:8000 -v trafficpulse-data:/data trafficpulse

# --- stage 1: build the SPA ----------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Copy the manifests first so `npm ci` is cached until dependencies actually change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Same-origin deployment: the API serves the SPA, so no VITE_API_BASE_URL is set
# and every request goes to the origin the app was loaded from.
RUN npm run build


# --- stage 2: python runtime ---------------------------------------------------
FROM python:3.12-slim AS runtime

# `api` is required (FastAPI + multipart); `overlay` brings Pillow, without which
# evidence stills and annotated video cannot be rendered. Add `rtdetr` at build
# time for real inference: --build-arg INSTALL_EXTRAS=api,overlay,rtdetr
ARG INSTALL_EXTRAS=api,overlay

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Storage lives on a volume, not in the image layer.
    TRAFFICPULSE_APP_STORAGE=/data \
    TRAFFICPULSE_APP_STATIC_DIR=/app/frontend/dist \
    TRAFFICPULSE_APP_HOST=0.0.0.0 \
    TRAFFICPULSE_APP_PORT=8000 \
    TRAFFICPULSE_APP_LOG_LEVEL=INFO \
    # Keep model resolution offline and inside the mounted cache.
    HF_HOME=/models \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# PyAV ships manylinux wheels with FFmpeg bundled, so no system codec packages are
# needed. curl is installed only for the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer: copy only what the install needs so source edits do not
# invalidate the (slow) dependency install.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --upgrade pip \
    && pip install ".[${INSTALL_EXTRAS}]" "uvicorn[standard]>=0.30"

# Application code: the launcher, the example scene, and the built SPA.
COPY serve.py ./
COPY configs/ ./configs/
COPY --from=frontend /build/dist ./frontend/dist

# Run unprivileged; /data is a mount point owned by the runtime user.
RUN useradd --create-home --uid 10001 trafficpulse \
    && mkdir -p /data /models \
    && chown -R trafficpulse:trafficpulse /data /models
USER trafficpulse

VOLUME ["/data"]
EXPOSE 8000

# Liveness only. `status: ok` means the process is serving; the readiness detail
# (repository writable, inference available) is in the same payload for an
# orchestrator that wants to gate traffic on more than liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# serve.py is the documented production entrypoint: AppConfig.from_env() plus the
# code-level model composition.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
