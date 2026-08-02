"""Which rules a given scene can actually run (H12).

Answers the question the calibration workflow turns on: *given this video's
scene, what can the engine be asked to do?* Before H12 the answer was a constant
-- the server's one scene, the operator's one rule list -- and the geometry rules
were simply switched off because that scene could not describe an arbitrary
upload. Now it varies per video, so it has to be computed.

Asks the real authorities; encodes no second rule set
------------------------------------------------------
Every rule already fails fast on a scene it cannot satisfy: wrong-way resolves a
governing legal direction and its parameter block, illegal-stopping resolves an
enabled no-stopping zone and a dwell threshold. Re-stating those requirements
here would create a second definition of "supported" that drifts from the first
the moment a reasoner's needs change -- and it would drift *silently*, offering a
rule that then fails at submit.

So this module **calls the shipped factories** and treats a raised
``SceneConfigurationError`` / ``ValueError`` as "not supported". The factories are
pure scene resolution -- they build no detector, decode nothing, and load no model
-- so probing costs a dictionary lookup and some tuple scans.

Not purely a scene question
----------------------------
No-helmet additionally needs a configured :class:`HelmetClassifier`, which is a
deployment fact rather than a scene fact (the engine's rule registry refuses a
``no_helmet`` rule without one). That is why this lives in the application layer
and takes ``classifier_available`` alongside the scene, instead of sitting in
:mod:`trafficpulse.scenes` where it would have to guess.
"""

from __future__ import annotations

from ..contracts import SceneConfig
from ..contracts.enums import ViolationType
from ..engine import (
    IllegalStoppingRuleConfig,
    NoHelmetRuleConfig,
    RuleConfig,
    TripleRidingRuleConfig,
    WrongWayRuleConfig,
)
from ..pipeline.errors import SceneConfigurationError
from ..pipeline.illegal_stopping import illegal_stopping_finalize_strategy
from ..pipeline.red_light import resolve_red_light_geometry
from ..pipeline.wrong_way import wrong_way_finalize_strategy
from ..rules.no_helmet import no_helmet_parameters
from ..rules.red_light import red_light_parameters
from ..rules.triple_riding import triple_riding_parameters


def _supports_wrong_way(scene: SceneConfig) -> bool:
    try:
        wrong_way_finalize_strategy(scene)
    except (SceneConfigurationError, ValueError):
        return False
    return True


def _supports_illegal_stopping(scene: SceneConfig) -> bool:
    try:
        illegal_stopping_finalize_strategy(scene)
    except (SceneConfigurationError, ValueError):
        return False
    return True


def _supports_no_helmet(scene: SceneConfig) -> bool:
    # The parameter loader rather than the strategy factory: building the no-helmet
    # strategy requires a live HelmetClassifier, and probing must not need one.
    # Availability of the classifier is asked separately, by the caller.
    try:
        no_helmet_parameters(scene)
    except (SceneConfigurationError, ValueError):
        return False
    return True


def _supports_triple_riding(scene: SceneConfig) -> bool:
    try:
        triple_riding_parameters(scene)
    except (SceneConfigurationError, ValueError):
        return False
    return True


def _supports_red_light(scene: SceneConfig) -> bool:
    # Scene readiness only: geometry (a stop line and the junction it guards) plus
    # the parameter block. The signal *schedule* is per-run and lives on the rule
    # config, so it is deliberately not part of what a scene can support -- a
    # calibrated junction supports red-light reasoning whether or not a particular
    # video's timing has been entered yet.
    try:
        red_light_parameters(scene)
        resolve_red_light_geometry(scene)
    except (SceneConfigurationError, ValueError):
        return False
    return True


def supported_violations(
    scene: SceneConfig, *, classifier_available: bool
) -> tuple[ViolationType, ...]:
    """The violations this scene (and deployment) can reason about, in a fixed order.

    Deterministic ordering so a client's rule list -- and any UI built on it --
    does not reshuffle between requests.
    """

    supported: list[ViolationType] = []
    if _supports_wrong_way(scene):
        supported.append(ViolationType.WRONG_WAY)
    if _supports_illegal_stopping(scene):
        supported.append(ViolationType.ILLEGAL_STOPPING)
    if _supports_red_light(scene):
        supported.append(ViolationType.RED_LIGHT_JUMPING)
    if classifier_available and _supports_no_helmet(scene):
        supported.append(ViolationType.NO_HELMET)
    if _supports_triple_riding(scene):
        supported.append(ViolationType.TRIPLE_RIDING)
    return tuple(supported)


def rules_for(violations: tuple[ViolationType, ...]) -> tuple[RuleConfig, ...]:
    """The default rule declarations for a set of supported violations.

    Each rule's own defaults apply: the per-rule knobs that are not scene facts
    (stationarity window, direction selection when a scene declares several) stay
    at the values the engine config declares, so this never invents a policy.
    """

    by_violation: dict[ViolationType, RuleConfig] = {
        ViolationType.WRONG_WAY: WrongWayRuleConfig(),
        ViolationType.ILLEGAL_STOPPING: IllegalStoppingRuleConfig(),
        ViolationType.NO_HELMET: NoHelmetRuleConfig(),
        ViolationType.TRIPLE_RIDING: TripleRidingRuleConfig(),
        # Deliberately absent: RED_LIGHT_JUMPING. Its rule config carries the run's
        # signal schedule, which no default can supply, and a schedule-less config
        # is refused by the registry rather than silently confirming nothing. A
        # client must send the timing explicitly.
    }
    return tuple(by_violation[violation] for violation in violations if violation in by_violation)
