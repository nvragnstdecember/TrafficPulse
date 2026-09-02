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


@pytest.fixture(scope="module")
def scene() -> SceneConfig:
    return SceneConfig.model_validate(
        yaml.safe_load(_SCENE_PATH.read_text(encoding="utf-8"))
    )


def _observations(
    scene: SceneConfig, *, slot: RiderSlot, state: HelmetState, count: int = 30
) -> list[HelmetStateObservation]:
    """A sustained, unambiguous track: ``count`` frames at 10fps, well past 1.0s."""

    return [
        HelmetStateObservation(
            observation_id=f"hlm-{slot.value}-{state.value}-{index:03d}",
            camera_id=scene.scene.camera_id,
            track_id="rider-1",
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


def _confirm(scene: SceneConfig, observations: list[HelmetStateObservation]) -> int:
    reasoner = NoHelmetReasoner(RuleEngine(), no_helmet_parameters(scene))
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


# --- precondition 1: driver attribution (OPEN) --------------------------------------
@pytest.mark.parametrize("slot", [RiderSlot.UNKNOWN, RiderSlot.PILLION])
def test_the_rule_does_not_yet_require_the_rider_to_be_the_driver(
    scene: SceneConfig, slot: RiderSlot
) -> None:
    """**Known gap.** A rider whose role is not ``DRIVER`` still confirms.

    ``rider_slot`` is derived (``observations.helmet.rider_slot``) and travels on every
    observation, but the reasoner never reads it -- so a **pillion passenger**, or a
    rider on a shared motorcycle whose role is ``UNKNOWN``, is confirmed and named on
    the event exactly as a lone driver would be.

    That is a second, independent blocker to enabling enforcement, and it is *not* the
    one the project's records emphasise. Turning the rule on today -- even on a
    turban-capable backend -- would attribute a helmet violation to passengers, on the
    42.4% of the frozen corpus and the 81% of a real congestion clip that are
    multi-rider.

    This test asserts the gap rather than the fix. Closing it means the reasoner must
    refuse a non-``DRIVER`` slot (abstain, never guess), at which point this test
    should be inverted deliberately and with its own review.
    """

    confirmed = _confirm(scene, _observations(scene, slot=slot, state=HelmetState.NO_HELMET))

    assert confirmed == 1, (
        "the driver-attribution gap has changed; if a driver gate was added this "
        "assertion must be inverted deliberately, not adjusted to pass"
    )


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
