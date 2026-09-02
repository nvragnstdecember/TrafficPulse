"""The preconditions helmet **enforcement** would have to satisfy, and where it stands.

These tests exist because of a specific, plausible accident. The project's recorded
blocker for enabling the no-helmet violation rule is the *turban capability* -- so an
engineer reading only that could reasonably conclude that a turban-capable backend is
the one thing standing between here and enforcement, set
``acknowledge_turban_blind=True`` or swap the backend, and ship.

That conclusion would be wrong, and these tests are how it stays visible.

**Characterisation, not endorsement.** The rider-slot test below pins behaviour this
project considers a *defect* for enforcement purposes. It is asserted so that the gap
cannot be closed, widened, or forgotten silently: if someone adds a driver gate, this
test fails and must be updated deliberately; until someone does, it is a standing,
executable statement that the gap is still open.

Nothing here changes any rule. The demo runs helmet **analysis**, which mints no event
at all, so none of this is live -- it is what would matter the moment it were.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from trafficpulse.contracts import (
    Association,
    HelmetStateObservation,
    Producer,
    SceneConfig,
)
from trafficpulse.contracts.enums import (
    AssociationType,
    HelmetState,
    ProducerKind,
    RiderSlot,
)
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.no_helmet import NoHelmetReasoner, no_helmet_parameters

_SCENE_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example-scene.yaml"
_BASE = datetime(1970, 1, 1, tzinfo=UTC)
_PRODUCER = Producer(name="test-helmet", version="0", kind=ProducerKind.MODEL)
_RIDER_TRACK_ID = "rider-1"


@pytest.fixture(scope="module")
def scene() -> SceneConfig:
    return SceneConfig.model_validate(
        yaml.safe_load(_SCENE_PATH.read_text(encoding="utf-8"))
    )


def _observations(
    scene: SceneConfig, *, slot: RiderSlot | None, state: HelmetState, count: int = 30
) -> list[HelmetStateObservation]:
    """A sustained, unambiguous track: ``count`` frames at 10fps, well past 1.0s.

    ``slot`` may be ``None``: the contract makes ``rider_slot`` optional, so an
    underived slot is a real input the gate has to handle.
    """

    label = slot.value if slot is not None else "noslot"
    return [
        HelmetStateObservation(
            observation_id=f"hlm-{label}-{state.value}-{index:03d}",
            camera_id=scene.scene.camera_id,
            track_id=_RIDER_TRACK_ID,
            timestamp=_BASE + timedelta(seconds=index * 0.1),
            confidence=0.95,
            producer=_PRODUCER,
            helmet_state=state,
            rider_slot=slot,
            crop_height_px=60.0,
        )
        for index in range(count)
    ]


def _associations(scene: SceneConfig, *, count: int = 30) -> list[Association]:
    return [
        Association(
            association_id=f"asc-{index:03d}",
            camera_id=scene.scene.camera_id,
            timestamp=_BASE + timedelta(seconds=index * 0.1),
            association_type=AssociationType.RIDER_OF_MOTORCYCLE,
            subject_track_id="rider-1",
            object_track_id="bike-1",
            confidence=0.9,
        )
        for index in range(count)
    ]


def _reasoner(scene: SceneConfig) -> NoHelmetReasoner:
    return NoHelmetReasoner(RuleEngine(), no_helmet_parameters(scene))


def _confirm(scene: SceneConfig, observations: list[HelmetStateObservation]) -> int:
    reasoner = _reasoner(scene)
    return len(
        reasoner.run(observations, associations=_associations(scene, count=len(observations)))
    )


# --- what already works -----------------------------------------------------------
def test_a_sustained_bare_headed_driver_confirms(scene: SceneConfig) -> None:
    """The rule itself is complete: given the right evidence it does confirm.

    Worth pinning, because the reason enforcement is off is *not* that the reasoner is
    unfinished. It works. What is missing is evidence it can be trusted with.
    """

    observations = _observations(
        scene, slot=RiderSlot.DRIVER, state=HelmetState.NO_HELMET
    )
    assert _confirm(scene, observations) == 1


def test_a_helmeted_rider_confirms_nothing(scene: SceneConfig) -> None:
    observations = _observations(
        scene, slot=RiderSlot.DRIVER, state=HelmetState.HELMET
    )
    assert _confirm(scene, observations) == 0


def test_an_uncertain_track_abstains_rather_than_confirming(scene: SceneConfig) -> None:
    """Abstention is an outcome: uncertainty never accumulates into a violation."""

    observations = _observations(
        scene, slot=RiderSlot.DRIVER, state=HelmetState.UNCERTAIN
    )
    assert _confirm(scene, observations) == 0


# --- precondition 1: driver attribution (CLOSED) ------------------------------------
@pytest.mark.parametrize("slot", [RiderSlot.UNKNOWN, RiderSlot.PILLION, RiderSlot.THIRD])
def test_a_rider_who_is_not_the_driver_abstains_rather_than_confirming(
    scene: SceneConfig, slot: RiderSlot
) -> None:
    """**Closed gap** (was: a non-``DRIVER`` rider confirmed like a lone driver).

    ``rider_slot`` is derived (``observations.helmet.rider_slot``) and travels on every
    observation. The reasoner now reads it and confirms only for ``DRIVER`` -- the sole
    slot that says *which* rider the bare head belongs to, assigned only when exactly
    one rider is associated with the motorcycle.

    Before the gate, a **pillion passenger** -- or any rider on a shared motorcycle
    whose role is ``UNKNOWN`` -- was confirmed and named on the event exactly as a lone
    driver would be, on the 42.4% of the frozen corpus and the 81% of a real congestion
    clip that are multi-rider. Persistence never fixed that: it measures how long the
    evidence held, never whose it was.

    This assertion was inverted deliberately (see the module docstring). It is now the
    executable statement of an enforced safety contract: if someone removes the driver
    gate, this fails.
    """

    confirmed = _confirm(scene, _observations(scene, slot=slot, state=HelmetState.NO_HELMET))

    assert confirmed == 0, (
        "a non-DRIVER rider was confirmed: the driver-attribution gate has been "
        "removed or weakened. Helmet evidence that cannot be attributed to the driver "
        "must abstain, never guess."
    )


def test_an_underived_rider_slot_also_abstains(scene: SceneConfig) -> None:
    """``None`` is not a slot. An observation carrying no slot cannot be attributed.

    ``rider_slot`` is optional on the contract, so a derivation that never set it must
    not fall through the gate as though it had said ``DRIVER``.
    """

    confirmed = _confirm(scene, _observations(scene, slot=None, state=HelmetState.NO_HELMET))

    assert confirmed == 0


def test_the_withheld_rider_is_recorded_rather_than_silently_dropped(
    scene: SceneConfig,
) -> None:
    """Abstention must be reportable: "no event" and "we declined" are different.

    A bare head the system saw but could not attribute is a limitation an operator has
    to be able to surface. It is recorded on the reasoner rather than dropped.
    """

    reasoner = _reasoner(scene)
    observations = _observations(scene, slot=RiderSlot.PILLION, state=HelmetState.NO_HELMET)
    events = reasoner.run(
        observations, associations=_associations(scene, count=len(observations))
    )

    assert events == ()
    assert reasoner.attribution_abstained_track_ids == frozenset({_RIDER_TRACK_ID})


# --- precondition 2: turban exemption (OPEN) -----------------------------------------
def test_the_exemption_needs_turban_observations_that_a_binary_backend_cannot_produce(
    scene: SceneConfig,
) -> None:
    """The exemption is real and it works -- **given** turban observations.

    A predominantly-turban track never confirms. But a binary backend cannot emit
    ``turban`` at all, so on such a backend those same frames arrive as ``no_helmet``
    and the same rider confirms. The exemption does not fail loudly; it simply has
    nothing to act on, which is precisely why the capability guard refuses the
    combination at composition time instead of leaving it to be discovered here.
    """

    turban = _observations(scene, slot=RiderSlot.DRIVER, state=HelmetState.TURBAN)
    assert _confirm(scene, turban) == 0

    # The identical rider, as a turban-blind backend would report them.
    as_binary_backend_sees_them = _observations(
        scene, slot=RiderSlot.DRIVER, state=HelmetState.NO_HELMET
    )
    assert _confirm(scene, as_binary_backend_sees_them) == 1
