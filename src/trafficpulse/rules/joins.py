"""Generalized deterministic observation join (P3-U3, dynamic traffic context).

Generalizes the two-stream ``(camera, track, timestamp)`` join that
``rules.illegal_stopping.join_stopped_in_zone`` performs into a reusable join that
also pairs a **scene-level context** stream. It is the second half of the P3-U3
dynamic-context infrastructure (the first being the signal-state derivation), and
the mechanism any future context-consuming rule -- red-light first -- reuses.

Shape
-----
Given one per-track **carrier** stream, :func:`join_streams` pairs each carrier
observation with:

* **other per-track streams** on ``(camera_id, track_id, timestamp)`` -- their
  observations at the carrier's key are collected into ``track_facts`` (in stream
  order, then input order); and
* a **scene-level context** stream on ``(camera_id, timestamp)`` (``track_id`` is
  ``None`` for context) -- the single context observation at the carrier's
  ``(camera, timestamp)`` is attached as ``context`` (first declaration wins on a
  duplicate key).

The join computes **no violation semantics**: it hands back the raw paired facts
and lets the caller fold them into its own per-step signal. This is what keeps it
reusable and free of any rule's vocabulary.

Conservative folding (never fabricate evidence)
-----------------------------------------------
A carrier with no matching per-track fact gets an **empty** ``track_facts`` tuple;
a carrier with no matching context gets ``context = None``. A downstream rule folds
those to "no evidence" -- an absent side can never manufacture a positive signal
(Phase-3 "never fabricate evidence"). Symmetrically, a per-track/context fact with
no carrier at its key produces no step at all (there is nothing to reason over).

Taint restarts (unioned onto the carrier)
-----------------------------------------
A carrier id is a taint restart iff it is one on the carrier stream **or** its
``(camera, track, timestamp)`` key carries a restart on any joined per-track
stream. Context carries no taint (a scene-level declared log has no track identity
to switch), so it contributes none. The reasoner resets its persistence run at a
restart regardless of which per-track stream flagged it.

Determinism
-----------
Steps preserve carrier input order; per-track facts preserve (stream, input)
order; context is resolved by a first-wins index. No wall-clock, no randomness, no
set/hash iteration in the emit path.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Protocol, TypeVar

from ..contracts.observations import ObservationBase

# Covariant: a stream of a specific observation variant is usable wherever a stream
# of the base is expected (an ``InZoneDerivation`` is a ``TaintedStream`` of the
# base for the purpose of collecting its facts).
_ObsT_co = TypeVar("_ObsT_co", bound=ObservationBase, covariant=True)
CarrierT = TypeVar("CarrierT", bound=ObservationBase)


class TaintedStream(Protocol[_ObsT_co]):
    """The shared shape of every derivation result: observations + taint restarts.

    ``HeadingDerivation``, ``InZoneDerivation``, and ``StationaryDerivation`` all
    structurally satisfy this (each exposes an ``observations`` tuple and a
    ``taint_restart_ids`` frozenset), so the join consumes any of them without a
    new adapter.
    """

    @property
    def observations(self) -> tuple[_ObsT_co, ...]: ...

    @property
    def taint_restart_ids(self) -> frozenset[str]: ...


@dataclass(frozen=True)
class JoinedStep(Generic[CarrierT]):
    """One carrier observation paired with its per-track facts and scene context.

    ``track_facts`` are the joined per-track observations at the carrier's
    ``(camera, track, timestamp)`` key (empty when none matched); ``context`` is
    the scene-level observation at the carrier's ``(camera, timestamp)`` key
    (``None`` when none matched). The caller folds these into its own signal.
    """

    carrier: CarrierT
    track_facts: tuple[ObservationBase, ...]
    context: ObservationBase | None


@dataclass(frozen=True)
class JoinResult(Generic[CarrierT]):
    """The joined steps (in carrier order) plus the unioned taint-restart ids."""

    steps: tuple[JoinedStep[CarrierT], ...]
    taint_restart_ids: frozenset[str]


_TrackKey = tuple[str, str | None, datetime]
_ContextKey = tuple[str, datetime]


def join_streams(
    carrier: TaintedStream[CarrierT],
    *,
    track_streams: Sequence[TaintedStream[ObservationBase]] = (),
    context: Sequence[ObservationBase] = (),
) -> JoinResult[CarrierT]:
    """Join a per-track ``carrier`` with other per-track streams and a scene context.

    See the module docstring for the full contract. Returns a
    :class:`JoinResult`: one :class:`JoinedStep` per carrier observation (in carrier
    input order), each carrying the per-track ``track_facts`` and the scene-level
    ``context`` matched at the carrier's key, plus the union of taint restarts from
    the carrier and the joined per-track streams.
    """

    # Index the other per-track streams by (camera, track, timestamp).
    facts_by_key: dict[_TrackKey, list[ObservationBase]] = {}
    restart_keys: set[_TrackKey] = set()
    for stream in track_streams:
        for obs in stream.observations:
            key: _TrackKey = (obs.camera_id, obs.track_id, obs.timestamp)
            facts_by_key.setdefault(key, []).append(obs)
            if obs.observation_id in stream.taint_restart_ids:
                restart_keys.add(key)

    # Index the scene-level context by (camera, timestamp); first declaration wins.
    context_by_key: dict[_ContextKey, ObservationBase] = {}
    for ctx in context:
        context_by_key.setdefault((ctx.camera_id, ctx.timestamp), ctx)

    steps: list[JoinedStep[CarrierT]] = []
    restart_ids: set[str] = set()
    for carrier_obs in carrier.observations:
        track_key: _TrackKey = (carrier_obs.camera_id, carrier_obs.track_id, carrier_obs.timestamp)
        context_key: _ContextKey = (carrier_obs.camera_id, carrier_obs.timestamp)
        steps.append(
            JoinedStep(
                carrier=carrier_obs,
                track_facts=tuple(facts_by_key.get(track_key, ())),
                context=context_by_key.get(context_key),
            )
        )
        if carrier_obs.observation_id in carrier.taint_restart_ids or track_key in restart_keys:
            restart_ids.add(carrier_obs.observation_id)
    return JoinResult(tuple(steps), frozenset(restart_ids))
