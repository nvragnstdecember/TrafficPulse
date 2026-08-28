"""The production launcher: real RT-DETR + helmet backend over env configuration.

**This is the documented production entrypoint.** Run it with any ASGI server::

    uvicorn serve:app --host 0.0.0.0 --port 8000

Why this module exists at all
-----------------------------
``trafficpulse.app.asgi`` builds an application purely from the environment, which
means it has **no inference backend**: every read endpoint works and the full UI
loads, but a processing request returns a clean ``503 engine_unavailable``. That
is the right default for a library entrypoint -- and the wrong thing to deploy.

The split is deliberate (see ``docs/adr/ADR-001.md``): a model checkpoint is a
licence-and-provenance decision reviewed per artifact, so it is composed in code
rather than read from an arbitrary environment string. Everything that is a
*deployment* decision -- where storage lives, which port, log level, CORS, upload
limits, whether to serve the SPA -- stays environmental.

So this module does exactly one thing on top of ``AppConfig.from_env()``: it
supplies the model composition.

Configured here (code-level, licence-reviewed)
    ``inference``, ``helmet_classifier``, ``label_map``

Configured by environment (deployment-level)
    ``TRAFFICPULSE_APP_STORAGE``, ``_SCENE``, ``_HOST``, ``_PORT``,
    ``_MAX_UPLOAD_BYTES``, ``_CORS_ORIGINS``, ``_STATIC_DIR``, ``_LOG_LEVEL``

Rule choice: **none is pinned here.** ``default_rules`` is deliberately left empty
so the application derives a job's rules from the scene that actually resolves for
the video being processed -- every shipped rule that scene can legitimately
support, run together. Pinning a fixed pair here (as this launcher previously did
with ``triple_riding`` + ``no_helmet``) meant a calibrated video still ran only the
two geometry-free motorcycle rules until an analyst manually reprocessed it, which
is precisely the multi-violation gap. The scene remains the authority, and with
``auto_calibrate_uploads`` on (below) an uncalibrated upload gets a scene derived
from *its own* motion rather than the shipped example's -- so a rule whose geometry
the resolved scene cannot satisfy is never selected, and no run ever reasons about
another camera's road. Setting ``default_rules`` here again would re-pin the set
and is supported for an operator who wants that.

Checkpoints load offline from the local HuggingFace cache (``local_files_only``),
so this launcher never reaches the network and never vendors weights.
"""

from __future__ import annotations

import os
from pathlib import Path

from trafficpulse.app import AppConfig, create_app
from trafficpulse.classifier import ZeroShotHelmetConfig
from trafficpulse.contracts.enums import ObjectClass
from trafficpulse.engine import InferenceConfig

# The RT-DETR (COCO-80) checkpoint's native labels -> TrafficPulse classes. This
# checkpoint uses the VOC-style "motorbike" spelling (verified against the cached
# model's id2label), which is what association/perception need for motorcycles.
# A wrong key here silently disables a whole violation class, which is why
# tests/app/test_serve_composition.py asserts this mapping.
LABEL_MAP: dict[str, ObjectClass] = {
    "person": ObjectClass.PERSON,
    "bicycle": ObjectClass.BICYCLE,
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
}

#: The shipped example scene, used only when the operator sets no ``_SCENE``.
DEFAULT_SCENE_PATH = Path("configs/scenes/example-scene.yaml")

#: Detector checkpoint. Overridable by an operator who has reviewed a different
#: artifact's licence; the default is the one this project validated.
DETECTOR_CHECKPOINT = os.environ.get(
    "TRAFFICPULSE_APP_DETECTOR_CHECKPOINT", "PekingU/rtdetr_r50vd"
)
#: Zero-shot helmet classifier checkpoint, same posture as the detector.
HELMET_CHECKPOINT = os.environ.get(
    "TRAFFICPULSE_APP_HELMET_CHECKPOINT", "openai/clip-vit-base-patch32"
)
#: Torch device for detection: ``auto`` uses CUDA when present, else CPU
#: (CPU inference is roughly 2-3 s/frame).
DETECTOR_DEVICE = os.environ.get("TRAFFICPULSE_APP_DEVICE", "auto")


def build_config() -> AppConfig:
    """The production configuration: environment first, model composition on top.

    A function rather than only a module constant, so tests can build and assert
    the composition without importing an ASGI application or loading a model --
    constructing these configs imports no ML framework; the backends are built
    lazily, per job.
    """

    env = AppConfig.from_env()
    scene_path = env.scene_path
    if scene_path is None and DEFAULT_SCENE_PATH.is_file():
        # Keep the operator's choice when they made one; otherwise fall back to the
        # shipped example so a fresh deployment is immediately demonstrable. When
        # neither exists (e.g. an image built without ``configs/``) the server still
        # serves every read endpoint and processes videos carrying their own scene.
        scene_path = DEFAULT_SCENE_PATH

    return env.model_copy(
        update={
            "scene_path": scene_path,
            # An upload nobody calibrated is derived from its own motion rather than
            # reasoned about through the shipped example scene's unrelated geometry.
            # This is what makes a raw upload an automatic multi-violation run: the
            # geometry rules get *this* camera's frame and observed traffic
            # direction. With this on, the scene above is no longer a fallback for
            # uncalibrated uploads at all -- an upload that cannot be derived runs
            # against its own frame with nothing inferred, never against this one.
            # Only observable facts are derived -- no no-stopping zone, no stop
            # line, no signal schedule -- so illegal-stopping and red-light still
            # wait for an analyst, and a calibrated video is never touched.
            "auto_calibrate_uploads": True,
            "inference": InferenceConfig(
                checkpoint=DETECTOR_CHECKPOINT,
                label_map=LABEL_MAP,
                device=DETECTOR_DEVICE,
                score_threshold=0.5,
                local_files_only=True,
            ),
            "helmet_classifier": ZeroShotHelmetConfig(
                checkpoint=HELMET_CHECKPOINT,
                device="cpu",
                local_files_only=True,
            ),
            # Deliberately absent: ``default_rules``. Leaving it empty is what makes
            # the processing service derive a job's rules from the resolved scene --
            # see this module's docstring.
        }
    )


config = build_config()
app = create_app(config)
