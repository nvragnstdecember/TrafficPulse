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

Probing needs the selector, not just the scene
-----------------------------------------------
Two rules govern *one* approach of a site that may declare several: wrong-way
picks a legal direction, red-light picks a stop line and the junction it guards.
Their factories deliberately refuse to guess when a scene declares more than one
-- silently reasoning about the wrong approach would be worse than failing. A
probe that called them with **no** selector therefore learned only whether the
scene was unambiguous, and reported every multi-approach site as unsupported even
though the rule runs perfectly once an id is supplied.

:func:`probe_scene` instead enumerates the scene's own declared directions and
junctions and asks the factory about **each** of them, keeping the first that
resolves (declaration order, so the answer is deterministic). That is strictly
more informative and no weaker: a violation is reported supported only when a
real, named selector actually constructs its rule, and that resolved selector is
carried into the rule config the processing path runs -- so "supported" and
"runnable" can never diverge.

Not purely a scene question
----------------------------
No-helmet additionally needs a deployment that can actually build the rule, which is
not a scene fact at all. That is why this lives in the application layer and takes
``no_helmet_available`` alongside the scene, instead of sitting in
:mod:`trafficpulse.scenes` where it would have to guess.

Two independent things can make that false, and the caller resolves both (see
:func:`trafficpulse.app.posture.no_helmet_rule_available`): no ``HelmetClassifier`` is
configured, or one is configured that has **declared it cannot emit** ``turban``, in
which case the classifier capability guard refuses the rule. The second case matters
here for a concrete reason: the derived rule set is what an uncalibrated upload runs,
so offering ``no_helmet`` on a turban-blind backend would put a rule in every job that
the engine then refuses at build time -- turning a deliberate safety guard into a
crash on every upload. Asking the guard is the same discipline as asking the scene
factories: one authority, consulted, never restated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import SceneConfig, ZoneType
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

# The zone kinds red-light reasoning accepts as "the junction the stop line
# guards". Mirrors the factory's own list, and is used only to enumerate the
# candidates to *ask* it about -- never to decide support.
_JUNCTION_TYPES = (ZoneType.INTERSECTION, ZoneType.SIGNAL_CONTROLLED_REGION)


@dataclass(frozen=True)
class SceneCapabilities:
    """What one scene can support, and the selector each governed rule needs.

    The single probe result both :func:`supported_violations` and :func:`rules_for`
    read, so the list a client is offered and the rules the server actually runs
    come from one computation and cannot disagree.

    ``wrong_way_direction_id`` / ``red_light_selectors`` are ``None`` when the rule
    is unsupported, and otherwise name the **first declared** selector that
    resolved -- deterministic, and exactly what the rule config must carry for the
    run to build.
    """

    wrong_way_direction_id: str | None = None
    illegal_stopping: bool = False
    red_light_selectors: tuple[str, str] | None = None
    no_helmet: bool = False
    triple_riding: bool = False

    @property
    def wrong_way(self) -> bool:
        """Whether some declared legal direction can govern a wrong-way run."""

        return self.wrong_way_direction_id is not None

    @property
    def red_light(self) -> bool:
        """Whether some declared stop line + junction can govern a red-light run."""

        return self.red_light_selectors is not None


def _resolve_wrong_way_direction(scene: SceneConfig) -> str | None:
    """The first declared legal direction the wrong-way factory accepts, else ``None``.

    Enumerating the scene's own directions is what lets a multi-lane site report
    wrong-way support: the factory refuses an *ambiguous* request, not a specific
    one. A scene that declares no direction, or whose ``wrong_way`` parameter block
    is missing, resolves nothing -- every attempt raises and the loop falls
    through -- so validation is unchanged.
    """

    for direction in scene.legal_directions:
        try:
            wrong_way_finalize_strategy(scene, direction_id=direction.direction_id)
        except (SceneConfigurationError, ValueError):
            continue
        return direction.direction_id
    return None


def _resolve_red_light_selectors(scene: SceneConfig) -> tuple[str, str] | None:
    """The first ``(stop_line_id, zone_id)`` pair red-light geometry accepts.

    The parameter block is checked once up front (it cannot vary by approach), then
    each enabled stop line is offered each enabled junction zone. The factory still
    enforces the ``stop_line.zone_ids`` guard, so a pair naming a junction that stop
    line does not guard is rejected here exactly as it would be at submit.
    """

    try:
        red_light_parameters(scene)
    except (SceneConfigurationError, ValueError):
        return None
    junctions = tuple(
        zone for zone in scene.zones if zone.enabled and zone.zone_type in _JUNCTION_TYPES
    )
    for line in scene.stop_lines:
        if not line.enabled:
            continue
        for zone in junctions:
            try:
                resolve_red_light_geometry(
                    scene, stop_line_id=line.stop_line_id, zone_id=zone.zone_id
                )
            except (SceneConfigurationError, ValueError):
                continue
            return line.stop_line_id, zone.zone_id
    return None


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


def probe_scene(scene: SceneConfig) -> SceneCapabilities:
    """Ask every shipped rule what it can do with ``scene`` (pure; no I/O, no model).

    Scene readiness only. The red-light *schedule* is per-run and lives on the rule
    config, so it is deliberately not part of what a scene can support -- a
    calibrated junction supports red-light reasoning whether or not a particular
    video's timing has been entered yet.
    """

    return SceneCapabilities(
        wrong_way_direction_id=_resolve_wrong_way_direction(scene),
        illegal_stopping=_supports_illegal_stopping(scene),
        red_light_selectors=_resolve_red_light_selectors(scene),
        no_helmet=_supports_no_helmet(scene),
        triple_riding=_supports_triple_riding(scene),
    )


def supported_violations(
    scene: SceneConfig, *, no_helmet_available: bool
) -> tuple[ViolationType, ...]:
    """The violations this scene (and deployment) can reason about, in a fixed order.

    Deterministic ordering so a client's rule list -- and any UI built on it --
    does not reshuffle between requests.
    """

    capabilities = probe_scene(scene)
    supported: list[ViolationType] = []
    if capabilities.wrong_way:
        supported.append(ViolationType.WRONG_WAY)
    if capabilities.illegal_stopping:
        supported.append(ViolationType.ILLEGAL_STOPPING)
    if capabilities.red_light:
        supported.append(ViolationType.RED_LIGHT_JUMPING)
    if no_helmet_available and capabilities.no_helmet:
        supported.append(ViolationType.NO_HELMET)
    if capabilities.triple_riding:
        supported.append(ViolationType.TRIPLE_RIDING)
    return tuple(supported)


def rules_for(scene: SceneConfig, *, no_helmet_available: bool) -> tuple[RuleConfig, ...]:
    """The default rule declarations for a scene: every rule it can legitimately run.

    This is the server's answer to "process this video" when the client named no
    rules. It is deliberately *not* "run all six violation types": it is the
    intersection of what is **shipped**, what this **scene** supports, and what this
    **deployment** has configured -- each decided by the rule's own factory in
    :func:`probe_scene`, never by a second rule set stated here. ``speeding`` has no
    shipped reasoner and so can never appear.

    Each rule's own defaults apply for the knobs that are not scene facts
    (stationarity window, no-helmet persistence), so this invents no policy. The
    governed rule carries the **resolved** selector, which is what makes a
    multi-approach scene's derived rules actually build.

    Deliberately absent: ``red_light_jumping``. Its rule config carries the run's
    signal schedule, which no default can supply, and a schedule-less config is
    refused by the engine's rule registry rather than silently confirming nothing.
    A client must send the timing explicitly (which the calibration surface does).

    Ordering matches :func:`supported_violations`, so the derived set is
    deterministic for a given scene.
    """

    capabilities = probe_scene(scene)
    rules: list[RuleConfig] = []
    if capabilities.wrong_way_direction_id is not None:
        rules.append(WrongWayRuleConfig(direction_id=capabilities.wrong_way_direction_id))
    if capabilities.illegal_stopping:
        rules.append(IllegalStoppingRuleConfig())
    if no_helmet_available and capabilities.no_helmet:
        rules.append(NoHelmetRuleConfig())
    if capabilities.triple_riding:
        rules.append(TripleRidingRuleConfig())
    return tuple(rules)
