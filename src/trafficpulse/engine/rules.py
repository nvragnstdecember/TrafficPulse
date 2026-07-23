"""Rule registry: existing reasoning slices composed onto one shared front half (H6).

The three shipped violation slices (wrong-way P1-U4/U10, illegal-stopping
P2-U4/U5, no-helmet P4-U5) each contribute a ``FinalizeStrategy`` -- built by the
**public factories their own pipeline modules expose** -- so the engine runs any
subset of them over **one** detect+track front half instead of re-running
detection per rule. This module adds no rule semantics: every threshold,
persistence window, and confirmation decision stays inside the existing
reasoners; the engine only routes.

* :func:`build_rules` maps validated :class:`RuleConfig` declarations to
  :class:`BuiltRule`\\ s, failing loudly (:class:`UnsupportedRuleError`) for
  violation types that have contracts but no shipped reasoner (red-light
  jumping, triple riding, speeding) -- they plug in here, additively, when
  their reasoners land. A configured no-helmet rule without an injected
  ``HelmetClassifier`` is an :class:`EngineConfigurationError` at build time,
  never a silent no-op.
* :class:`MultiRuleFinalize` fans ``CompositionPipeline.finalize`` out to every
  built strategy: one reasoner per rule per finalize (fresh, exactly like the
  single-rule pipelines), events unioned per track and deterministically
  sorted by the shared base. Hypothesis generation/confirmation therefore runs
  through the same P1-U3 ``RuleEngine`` lifecycle each slice already uses.
* :class:`CompositeFrameObserver` fans the pixel side-channel out to every
  rule's observer (currently only no-helmet has one), preserving the P4-U2
  contract: observe in stream order, reset for replay, decide nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..classifier.interface import HelmetClassifier
from ..contracts import ConfirmedEvent, ModelRef, SceneConfig, TrackState
from ..contracts.enums import ViolationType
from ..detector.frame import Frame
from ..pipeline.base import FinalizeStrategy, FrameObserver
from ..pipeline.illegal_stopping import illegal_stopping_finalize_strategy
from ..pipeline.no_helmet import no_helmet_finalize_strategy
from ..pipeline.triple_riding import triple_riding_finalize_strategy
from ..pipeline.wrong_way import wrong_way_finalize_strategy
from .config import (
    IllegalStoppingRuleConfig,
    NoHelmetRuleConfig,
    RuleConfig,
    TripleRidingRuleConfig,
    WrongWayRuleConfig,
)
from .errors import EngineConfigurationError, UnsupportedRuleError

# Violations with frozen contracts but no shipped reasoner yet: named here so the
# registry's refusal message states exactly what exists and what does not.
_UNSHIPPED = (
    ViolationType.RED_LIGHT_JUMPING,
    ViolationType.SPEEDING,
)


@dataclass(frozen=True)
class BuiltRule:
    """One configured rule, realised: its strategy and (optional) pixel observer."""

    violation: ViolationType
    strategy: FinalizeStrategy[Any]
    observer: FrameObserver | None = None


def build_rules(
    configs: Sequence[RuleConfig],
    *,
    scene: SceneConfig,
    classifier: HelmetClassifier | None = None,
    capture_overlay: bool = False,
) -> tuple[BuiltRule, ...]:
    """Realise the configured rules against one scene (fail-fast; see module doc).

    ``capture_overlay`` (default off) enables overlay-metadata capture on any rule
    observer that supports it (currently no-helmet), so the engine can redraw
    inference for the visualization framework without re-running a model.

    Raises:
        EngineConfigurationError: a no-helmet rule is configured but no
            ``HelmetClassifier`` was injected.
        UnsupportedRuleError: never for a validated :class:`RuleConfig` today --
            the config union is closed over shipped rules -- but raised by
            :func:`require_shipped` for callers probing by violation type.
        SceneConfigurationError / ValueError: the scene cannot satisfy a rule's
            parameters (propagated from the pipeline factories, unchanged).
    """

    built: list[BuiltRule] = []
    for config in configs:
        if isinstance(config, WrongWayRuleConfig):
            built.append(
                BuiltRule(
                    violation=ViolationType.WRONG_WAY,
                    strategy=wrong_way_finalize_strategy(
                        scene, direction_id=config.direction_id
                    ),
                )
            )
        elif isinstance(config, IllegalStoppingRuleConfig):
            built.append(
                BuiltRule(
                    violation=ViolationType.ILLEGAL_STOPPING,
                    strategy=illegal_stopping_finalize_strategy(
                        scene,
                        stationary_window=config.stationary_window,
                        stationary_epsilon_px=config.stationary_epsilon_px,
                    ),
                )
            )
        elif isinstance(config, NoHelmetRuleConfig):
            if classifier is None:
                raise EngineConfigurationError(
                    "a no_helmet rule is configured but no HelmetClassifier was "
                    "injected; pass classifier= when building the engine"
                )
            strategy, observer = no_helmet_finalize_strategy(
                scene, classifier=classifier, capture_overlay=capture_overlay
            )
            built.append(
                BuiltRule(
                    violation=ViolationType.NO_HELMET,
                    strategy=strategy,
                    observer=observer,
                )
            )
        else:
            assert isinstance(config, TripleRidingRuleConfig)  # closed discriminated union
            # Pure geometry over the perception + association seams: no classifier.
            tr_strategy, tr_observer = triple_riding_finalize_strategy(scene)
            built.append(
                BuiltRule(
                    violation=ViolationType.TRIPLE_RIDING,
                    strategy=tr_strategy,
                    observer=tr_observer,
                )
            )
    return tuple(built)


def require_shipped(violation: ViolationType) -> None:
    """Fail loudly for violation types that have no shipped reasoning slice.

    The typed refusal the H6 spec's rule surface needs: contracts exist for all
    six violation types, reasoners for three. Probing an unshipped one raises
    :class:`UnsupportedRuleError` naming both sets, so the gap is explicit and
    additively closeable.
    """

    if violation in _UNSHIPPED:
        shipped = (
            ViolationType.WRONG_WAY,
            ViolationType.ILLEGAL_STOPPING,
            ViolationType.NO_HELMET,
            ViolationType.TRIPLE_RIDING,
        )
        raise UnsupportedRuleError(
            f"violation {violation.value!r} has a frozen contract but no shipped "
            f"reasoning slice yet; shipped: {[v.value for v in shipped]}"
        )


class CompositeFrameObserver:
    """Fans the P4-U2 pixel side-channel out to several rule observers.

    Children are called in registry (configuration) order on every frame --
    each sees the true stream, exactly as if it were the sole observer -- and
    all reset together for replay. Mutates nothing, decides nothing.
    """

    def __init__(self, observers: Sequence[FrameObserver]) -> None:
        self._observers = tuple(observers)

    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        for observer in self._observers:
            observer.observe(frame, states)

    def reset(self) -> None:
        for observer in self._observers:
            observer.reset()


@dataclass(frozen=True)
class MultiRuleFinalize:
    """One ``FinalizeStrategy`` running every built rule's back half.

    The reasoner handed back by :meth:`build_reasoner` is the tuple of per-rule
    reasoners (fresh each finalize, matching the single-rule pipelines);
    :meth:`events_for_track` chains each rule's events for the track and the
    shared base sorts the union by ``(trigger_at, event_id)`` -- so a one-rule
    engine is event-identical to the corresponding standalone pipeline (the
    tests assert this equivalence).
    """

    rules: tuple[BuiltRule, ...]

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> tuple[Any, ...]:
        return tuple(
            rule.strategy.build_reasoner(scene_config_hash=scene_config_hash, models=models)
            for rule in self.rules
        )

    def events_for_track(
        self, reasoner: tuple[Any, ...], track: list[TrackState]
    ) -> Iterable[ConfirmedEvent]:
        events: list[ConfirmedEvent] = []
        for rule_reasoner, rule in zip(reasoner, self.rules, strict=True):
            events.extend(rule.strategy.events_for_track(rule_reasoner, track))
        return events
