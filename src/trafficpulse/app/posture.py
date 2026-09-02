"""What this deployment can honestly claim about helmet reasoning (the status strip).

A **deployment** question, not a scene one -- which is why it lives here and not beside
:mod:`trafficpulse.app.capabilities`. That module answers "what can this *scene* run";
this one answers "what does this *configuration* entitle the operator to say", and the
two vary independently: a perfectly calibrated junction still cannot enforce a helmet
violation on a backend that cannot express the turban exemption.

Why the application has to state this at all
--------------------------------------------
Every limitation reported here is already enforced somewhere in the stack -- the
capability guard refuses a turban-blind rule, ``rider_slot`` returns ``UNKNOWN`` for
multi-rider motorcycles, an analysis mints no event. But enforcement is invisible: the
system that refuses to confirm a violation and the system that had nothing to confirm
look identical from outside. A viewer shown boxes and helmet labels will read them as
enforcement output unless told otherwise, and that reading is the one
``docs/helmet-runtime-evaluation.md`` §6 says is not safe to make.

So this module turns each guard into a sentence. It **decides nothing**: it reports what
the configuration has already determined, by asking the same authorities the runtime
asks -- the classifier config's own ``declared_labels``, and whether an analysis was
declared. If a guard changes, this reports the change; it holds no second copy of the
policy.

Costs nothing to compute
-------------------------
No model is constructed and no checkpoint is read. ``declared_labels`` is a property of
the *config* precisely so a question like this one can be answered without loading torch
(see :attr:`~trafficpulse.classifier.resnet.ResNetHelmetConfig.declared_labels`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..classifier.capabilities import TURBAN_LABEL
from .config import AppConfig


class PostureState(StrEnum):
    """How far a capability can be relied on. Ordered from strongest to weakest.

    ``LIMITED`` and ``EXPERIMENTAL`` are deliberately distinct. ``LIMITED`` means the
    capability works within a stated boundary and is useful inside it (detection runs,
    but misses ~20% of eligible riders). ``EXPERIMENTAL`` means it *runs* but its output
    is not something the evidence supports acting on. Collapsing the two would let a
    surface present an unvalidated enforcement decision as a merely-caveated one.
    """

    ACTIVE = "active"
    LIMITED = "limited"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PostureComponent:
    """One line of the status strip: what it is, how far it goes, and why."""

    component_id: str
    label: str
    state: PostureState
    detail: str
    """A complete sentence a non-specialist can act on. Never a bare metric, and never
    a reassurance -- where a capability is weak this says so plainly."""


@dataclass(frozen=True)
class SystemPosture:
    """The whole strip, plus the two facts a client may want to branch on.

    ``helmet_enforcement`` is repeated as a field because it is the one state a UI must
    be able to check without string-matching a component id: it decides whether a helmet
    reading may be presented as anything approaching a violation.
    """

    components: tuple[PostureComponent, ...]
    helmet_backend: str | None
    """The configured backend's config class name, or ``None``. A class name rather than
    a friendly label because it is the thing an operator can grep for in ``serve.py``."""

    helmet_backend_labels: tuple[str, ...]
    """Exactly what the configured backend can emit, as it declares itself. Empty when
    no backend is configured, or when a backend declines to declare (which is a real and
    different state -- undeclared is not the same as incapable)."""

    turban_capable: bool
    helmet_enforcement: PostureState


def _detection(config: AppConfig) -> PostureComponent:
    if config.inference is None:
        return PostureComponent(
            component_id="detection",
            label="Detection",
            state=PostureState.UNAVAILABLE,
            detail=(
                "No inference backend is configured. Every read endpoint works and the "
                "workspace loads, but processing a video returns 503."
            ),
        )
    return PostureComponent(
        component_id="detection",
        label="Detection",
        state=PostureState.LIMITED,
        detail=(
            "RT-DETR is running at the validated 0.50 score threshold. Recall is not "
            "complete: on the frozen evaluation split about 20% of eligible riders had "
            "no overlapping detection, and small/distant riders are the worst stratum."
        ),
    )


def _classification(config: AppConfig, *, backend: str | None) -> PostureComponent:
    if backend is None:
        return PostureComponent(
            component_id="helmet_classification",
            label="Helmet classification",
            state=PostureState.UNAVAILABLE,
            detail="No helmet classifier is configured, so no helmet state is produced.",
        )
    return PostureComponent(
        component_id="helmet_classification",
        label="Helmet classification",
        state=PostureState.ACTIVE,
        detail=(
            "Head crops are classified per frame and smoothed per rider track for "
            "display. Accuracy figures published for this backend are per-crop and "
            "conditional on a rider reaching the classifier at all; they are not "
            "end-to-end system accuracy."
        ),
    )


def _driver_attribution() -> PostureComponent:
    """Always ``LIMITED``: there is no configuration in which this is solved.

    Stated unconditionally and on purpose. Driver-versus-pillion needs the motorcycle's
    travel direction, the shipped tracker supplies no velocity, and no backend choice
    changes that -- so making this component conditional would imply some configuration
    exists that lifts it. None does.
    """

    return PostureComponent(
        component_id="driver_attribution",
        label="Driver attribution",
        state=PostureState.LIMITED,
        detail=(
            "Single-rider motorcycles only. When two or more riders are associated with "
            "one motorcycle the driver cannot be identified -- the tracker supplies no "
            "velocity, so which end of the bike is the front is unknown -- and those "
            "riders are reported as unresolved rather than guessed at. This is 42.4% of "
            "the frozen evaluation corpus and 81% of crops in a real congestion clip."
        ),
    )


def _turban(*, backend: str | None, declared: tuple[str, ...], capable: bool) -> PostureComponent:
    if backend is None:
        return PostureComponent(
            component_id="turban_exemption",
            label="Turban exemption",
            state=PostureState.UNAVAILABLE,
            detail="No helmet classifier is configured, so no exemption can be evaluated.",
        )
    if not declared:
        return PostureComponent(
            component_id="turban_exemption",
            label="Turban exemption",
            state=PostureState.UNAVAILABLE,
            detail=(
                "The configured backend declares no vocabulary, so what it can emit is "
                "unknown. Undeclared is not the same as incapable, but it is not "
                "evidence of capability either."
            ),
        )
    if not capable:
        return PostureComponent(
            component_id="turban_exemption",
            label="Turban exemption",
            state=PostureState.UNAVAILABLE,
            detail=(
                "The configured backend is binary and declares it cannot emit 'turban'. "
                "The capability guard therefore refuses to build a no-helmet violation "
                "rule on it, because the exemption would silently never fire and "
                "turban-wearing riders would be confirmed as violations."
            ),
        )
    return PostureComponent(
        component_id="turban_exemption",
        label="Turban exemption",
        state=PostureState.EXPERIMENTAL,
        detail=(
            "The configured backend can emit 'turban', but that output is not "
            "demonstrated to be a working turban detector: on the frozen evaluation "
            "split 96.8% of its turban predictions land on riders annotated as "
            "helmeted. The exemption is therefore available but unvalidated."
        ),
    )


def _enforcement(config: AppConfig, *, backend: str | None, capable: bool) -> PostureComponent:
    """The one component a surface must never get wrong.

    Order of the branches is the order of the blockers' severity: no backend at all, then
    an explicit analysis-only deployment, then the capability guard, and only then the
    residual case where the rule *can* build -- which is still not "validated", because
    the temporal-run semantics have never been evaluated against real per-frame output.
    There is no branch that reports helmet enforcement as ``ACTIVE``, deliberately: no
    configuration of this system currently earns that word.
    """

    if backend is None:
        return PostureComponent(
            component_id="helmet_enforcement",
            label="Helmet violation enforcement",
            state=PostureState.UNAVAILABLE,
            detail="No helmet classifier is configured, so no helmet rule can run.",
        )
    if config.helmet_analysis is not None:
        return PostureComponent(
            component_id="helmet_enforcement",
            label="Helmet violation enforcement",
            state=PostureState.DISABLED,
            detail=(
                "This deployment runs helmet classification as analysis only. No helmet "
                "violation is confirmed, no helmet event is recorded, and nothing shown "
                "here is an enforcement decision. Other violation families (wrong-way, "
                "illegal stopping, red-light, triple riding) are unaffected and run "
                "normally where the scene supports them."
            ),
        )
    if not capable:
        return PostureComponent(
            component_id="helmet_enforcement",
            label="Helmet violation enforcement",
            state=PostureState.UNAVAILABLE,
            detail=(
                "Refused by the classifier capability guard: the configured backend "
                "cannot emit 'turban', so the rule's exemption could never fire."
            ),
        )
    return PostureComponent(
        component_id="helmet_enforcement",
        label="Helmet violation enforcement",
        state=PostureState.EXPERIMENTAL,
        detail=(
            "The rule can build on this backend, but its confirmation semantics have "
            "never been evaluated against real per-frame classifier output, which flips "
            "between consecutive frames on the majority of tracked riders. Treat any "
            "helmet violation it confirms as unvalidated."
        ),
    )


def no_helmet_rule_available(config: AppConfig) -> bool:
    """Whether this deployment can actually **build** a ``no_helmet`` rule.

    Two things can make it false, and both are configuration rather than scene facts:
    no classifier is configured at all, or one is configured that has declared it
    cannot emit ``turban``, in which case
    :func:`~trafficpulse.classifier.capabilities.require_turban_capability` refuses the
    rule. The second is why this function exists: the rule derivation used to ask only
    "is a classifier configured", which on a binary backend produced a rule set every
    job would then be refused for -- the guard firing correctly, at the worst possible
    moment, on every upload.

    Answered from ``declared_labels`` on the *config*, so nothing is constructed and no
    checkpoint is read. An **undeclared** backend passes, exactly as the guard treats it:
    unknown is not incapable.
    """

    helmet_config = config.helmet_classifier
    if helmet_config is None:
        return False
    return TURBAN_LABEL in helmet_config.declared_labels


def describe(config: AppConfig) -> SystemPosture:
    """Compute the deployment's posture. Pure; loads no model and reads no checkpoint."""

    helmet_config = config.helmet_classifier
    backend = type(helmet_config).__name__ if helmet_config is not None else None
    declared = (
        tuple(sorted(helmet_config.declared_labels)) if helmet_config is not None else ()
    )
    capable = TURBAN_LABEL in declared

    enforcement = _enforcement(config, backend=backend, capable=capable)
    return SystemPosture(
        components=(
            _detection(config),
            _classification(config, backend=backend),
            _driver_attribution(),
            _turban(backend=backend, declared=declared, capable=capable),
            enforcement,
        ),
        helmet_backend=backend,
        helmet_backend_labels=declared,
        turban_capable=capable,
        helmet_enforcement=enforcement.state,
    )
