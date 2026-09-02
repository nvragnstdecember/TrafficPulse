"""The demonstration launcher: the trained ResNet-50 backend, classifying but not enforcing.

    uvicorn serve_demo:app --host 127.0.0.1 --port 8000

**This is a demo configuration, not a production adoption.** It changes nothing about
what the project has decided: ADR-005 still adopts no backend, ``serve.py`` is still the
production entrypoint, and the zero-shot classifier is still what a production deployment
gets. This module exists because the two questions -- *which backend reads a crop best*
and *which backend may enforce a violation* -- have different answers, and until now the
system could only be configured as if they had one.

What this composes, and why each piece
---------------------------------------
It reuses ``serve.py`` wholesale for everything that is not a helmet decision -- the
RT-DETR checkpoint, the licence-reviewed ``LABEL_MAP``, the device, the scene fallback,
auto-calibration, the empty ``default_rules`` that keeps rule selection scene-derived --
and overrides exactly two fields.

**1. The helmet backend: the trained P4-U5 ResNet-50.** On the corrected runtime
population (P4-U9) it scores 0.857 native macro-F1 against the shipped zero-shot
backend's 0.297, on the *same* crops, cut by the *same* production geometry. The
zero-shot backend calls ``no_helmet`` on 93% of everything at 25.7% precision; a demo
built on it would not be showing helmet classification, it would be showing a
near-constant predictor. Selecting ResNet here is a demonstration choice backed by the
measurement, and it is confined to this file -- ``AppConfig.helmet_classifier`` chooses a
backend by which config object it is given, so nothing downstream knows or cares.

``temperature`` and ``abstain_below`` are the **pre-committed** P4-U5/P4-U9 values (seed
0's validation-fitted T = 2.2298; the 0.80 floor selected on validation and re-derived
identically on the corrected population). Neither was chosen here, and neither is tuned
against anything observed at demo time.

**2. Helmet analysis instead of the helmet rule.** This is the part that matters.
ResNet-50 is **binary**: it cannot emit ``turban`` under any configuration, and it
declares that. The no-helmet violation rule's exemption depends on ``turban``
observations, so building that rule on this backend would leave the exemption
permanently dead and confirm turban-wearing riders as violators -- a systematic false
accusation against a religious group. The classifier capability guard refuses exactly
that combination, and **this launcher does not bypass it**:
``acknowledge_turban_blind`` is not set anywhere here, and
:func:`trafficpulse.app.posture.no_helmet_rule_available` reports ``False``, so the
scene-derived rule set never even offers ``no_helmet``.

Declaring ``helmet_analysis`` is what makes that a usable posture rather than a dead end.
The classifier runs, every rider is classified, the annotated video is drawn, the summary
is reported -- and no ``ConfirmedEvent`` is minted, because an analysis has no reasoner
and structurally cannot produce one. ``GET /api/system/posture`` states this in words on
every page that renders it.

What is still enforced
----------------------
Everything else. Wrong-way, illegal stopping, red-light jumping and triple riding are
untouched by this file and run exactly as they do under ``serve.py``, on whatever the
resolved scene supports. Only the *helmet* violation is off, and only because its
evidence is not there.

What this demo may and may not be said to show
-----------------------------------------------
May: RT-DETR detection; rider association; the head-crop geometry; a trained classifier
reading those crops; per-track temporal smoothing of an unstable per-frame signal; the
system declining to attribute a helmet state to a rider whose role it cannot determine;
and the other violation families end to end.

May not: that helmet violations are detected (no rule runs); that multi-rider traffic is
handled (42.4% of the frozen corpus is unattributable by design); that the turban
exemption works (its evidence is not demonstrated on any backend); that smoothing
improves accuracy (it is unvalidated and is presentation only); or P4-U5's 0.929, which
was measured on whole-motorcycle crops and says nothing about runtime accuracy.

Requires the checkpoint
-----------------------
Unlike ``serve.py``, whose checkpoints resolve from the local HuggingFace cache, this one
needs a **local file**: the P4-U5 ``best.pt``. It is never downloaded (ADR-001: an
artifact's provenance is a per-artifact review). Point ``TRAFFICPULSE_DEMO_HELMET_CHECKPOINT``
at one, or leave the default, which is where the P4-U5 run wrote seed 0.
"""

from __future__ import annotations

import os
from pathlib import Path

import serve

from trafficpulse.app import AppConfig, create_app
from trafficpulse.classifier import ResNetHelmetConfig
from trafficpulse.engine import HelmetAnalysisConfig

#: The trained P4-U5 ResNet-50. **Seed 0 specifically**, because the temperature below
#: was fitted on seed 0's validation split -- pairing a fitted temperature with a
#: different checkpoint would apply one model's calibration to another's logits.
DEMO_HELMET_CHECKPOINT = Path(
    os.environ.get(
        "TRAFFICPULSE_DEMO_HELMET_CHECKPOINT",
        "runs/helmet_cnn_vit/final/resnet50_lr0.001_s0/checkpoints/best.pt",
    )
)

#: Logit temperature fitted on **validation** by P4-U5 (seed 0: ECE 0.0355 -> 0.0065).
#: A property of that checkpoint, quoted, not chosen here.
DEMO_HELMET_TEMPERATURE = 2.2298

#: Winning-probability floor below which the backend emits ``uncertain`` rather than
#: forcing a binary call. **Pre-committed**: selected on validation in P4-U6-V and
#: returned unchanged by P4-U9's re-selection on the corrected population. It is not
#: re-tuned here, and must not be -- tuning an operating point on observed outputs is
#: how a test split leaks into a deployment.
DEMO_HELMET_ABSTAIN_BELOW = 0.80


def build_config() -> AppConfig:
    """The demo configuration: the production composition, with two fields overridden.

    Delegates to ``serve.build_config()`` rather than restating it, so the detector
    checkpoint, the ``LABEL_MAP`` whose keys silently gate whole violation classes, the
    device, the scene fallback and the auto-calibration posture have exactly one
    definition. Constructing this loads no ML framework and reads no checkpoint; the
    backend is built lazily, per job.
    """

    return serve.build_config().model_copy(
        update={
            "helmet_classifier": ResNetHelmetConfig(
                checkpoint=DEMO_HELMET_CHECKPOINT,
                device="cpu",
                temperature=DEMO_HELMET_TEMPERATURE,
                abstain_below=DEMO_HELMET_ABSTAIN_BELOW,
            ),
            # Classify, do not enforce. The capability guard independently refuses the
            # no-helmet rule on this backend; this is what makes that refusal a working
            # configuration instead of simply losing helmet perception altogether.
            "helmet_analysis": HelmetAnalysisConfig(),
        }
    )


def __getattr__(name: str) -> object:
    """Build ``config`` / ``app`` on first access (see ``serve.__getattr__``)."""

    if name == "config":
        return build_config()
    if name == "app":
        return create_app(build_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
