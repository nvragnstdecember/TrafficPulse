"""The helmet-analysis overlay: what it draws, and what it can never draw.

The provider exists because reusing the no-helmet provider for an analysis run would
put the visual language of enforcement -- confirmation styling, violation banners --
on a run that confirmed nothing. These tests pin the difference.
"""

from __future__ import annotations

import pytest

from trafficpulse.observations.helmet_stability import HelmetStabilizationConfig
from trafficpulse.overlay.metadata import (
    OverlayAlert,
    OverlayBanner,
    OverlayBox,
    OverlayEmphasis,
)
from trafficpulse.overlay.providers.helmet_analysis import HelmetAnalysisOverlayProvider
from trafficpulse.overlay.registry import OverlayFrameRef
from trafficpulse.pipeline.helmet_observer import HelmetOverlayFrame, HelmetOverlayRider


def _rider(
    track: str,
    *,
    bike: str = "m1",
    label: str = "helmet",
    confidence: float | None = 0.9,
    gated: bool = False,
) -> HelmetOverlayRider:
    return HelmetOverlayRider(
        rider_track_id=track,
        rider_bbox=(60.0, 50.0, 140.0, 280.0),
        motorcycle_track_id=bike,
        motorcycle_bbox=(50.0, 150.0, 150.0, 300.0),
        head_bbox=(60.0, 50.0, 140.0, 119.0),
        helmet_label=label,
        confidence=confidence,
        gated=gated,
    )


def _frames(
    labels: list[str], *, riders_per_frame: int = 1
) -> list[HelmetOverlayFrame]:
    return [
        HelmetOverlayFrame(
            frame_index=index,
            media_seconds=index * 0.1,
            riders=tuple(
                _rider(f"p{n}", label=label) for n in range(1, riders_per_frame + 1)
            ),
        )
        for index, label in enumerate(labels)
    ]


def _ref(index: int) -> OverlayFrameRef:
    return OverlayFrameRef(
        camera_id="cam-1",
        frame_index=index,
        media_seconds=index * 0.1,
        width=320,
        height=240,
    )


def _boxes(elements: object) -> list[OverlayBox]:
    return [element for element in elements if isinstance(element, OverlayBox)]  # type: ignore[union-attr]


def _captions(elements: object) -> list[str]:
    lines: list[str] = []
    for box in _boxes(elements):
        if box.caption is not None:
            lines.extend(box.caption.lines)
    return lines


# --- the thing it must never do -------------------------------------------------------
def test_no_element_is_ever_drawn_as_a_confirmed_violation() -> None:
    """``CONFIRMED`` is the visual language of enforcement, and there is none here."""

    provider = HelmetAnalysisOverlayProvider(
        _frames(["no_helmet"] * 8), stabilization=HelmetStabilizationConfig(min_samples=1)
    )

    for index in range(8):
        elements = provider.elements_for_frame(_ref(index))
        assert all(
            getattr(element, "alert", OverlayAlert.NONE) is not OverlayAlert.CONFIRMED
            for element in elements
        )


def test_every_drawn_frame_carries_the_analysis_mode_banner() -> None:
    """A still pulled from the video cannot be separated from its disclaimer."""

    provider = HelmetAnalysisOverlayProvider(_frames(["helmet"] * 3))

    for index in range(3):
        banners = [
            element
            for element in provider.elements_for_frame(_ref(index))
            if isinstance(element, OverlayBanner)
        ]
        assert len(banners) == 1
        assert banners[0].title == "HELMET ANALYSIS"
        assert banners[0].alert is OverlayAlert.NONE
        assert any("No violation decision" in detail for detail in banners[0].details)


# --- what it draws --------------------------------------------------------------------
def test_it_draws_the_motorcycle_the_rider_and_the_exact_head_crop() -> None:
    provider = HelmetAnalysisOverlayProvider(_frames(["helmet"]))

    emphases = {box.emphasis for box in _boxes(provider.elements_for_frame(_ref(0)))}

    assert emphases == {
        OverlayEmphasis.OBJECT,
        OverlayEmphasis.SUBJECT,
        OverlayEmphasis.REGION,
    }


def test_it_draws_the_stabilized_label_not_the_frames_raw_one() -> None:
    """The whole reason the provider owns a stabilizer: one flip must not show."""

    provider = HelmetAnalysisOverlayProvider(
        _frames(["helmet", "helmet", "no_helmet", "helmet"])
    )

    captions = _captions(provider.elements_for_frame(_ref(2)))

    assert "Helmet" in captions
    assert "No Helmet" not in captions


def test_a_less_than_unanimous_window_says_so() -> None:
    """A 2-of-3 call must not read like a 3-of-3 one."""

    provider = HelmetAnalysisOverlayProvider(
        _frames(["helmet", "helmet", "no_helmet"]),
        stabilization=HelmetStabilizationConfig(window=3, min_samples=1),
    )

    captions = _captions(provider.elements_for_frame(_ref(2)))

    assert any("of 3 frames" in line for line in captions)


def test_an_unscored_crop_prints_no_confidence_chip() -> None:
    """A label with no figure costs a caption and conveys nothing."""

    frames = [
        HelmetOverlayFrame(
            frame_index=0,
            media_seconds=0.0,
            riders=(_rider("p1", label="uncertain", confidence=None, gated=True),),
        )
    ]
    provider = HelmetAnalysisOverlayProvider(frames)

    head = next(
        box
        for box in _boxes(provider.elements_for_frame(_ref(0)))
        if box.emphasis is OverlayEmphasis.REGION
    )
    assert head.caption is None


# --- multi-rider ------------------------------------------------------------------------
def test_a_shared_motorcycle_is_captioned_unresolved_and_names_no_driver() -> None:
    provider = HelmetAnalysisOverlayProvider(
        _frames(["no_helmet"] * 6, riders_per_frame=2),
        stabilization=HelmetStabilizationConfig(min_samples=1),
    )

    captions = _captions(provider.elements_for_frame(_ref(5)))

    assert "MULTI-RIDER" in captions
    assert "DRIVER UNRESOLVED" in captions
    assert captions.count("MULTI-RIDER") == 1, "one statement per vehicle, not per rider"
    assert "Driver unresolved" in captions
    # No rider is singled out, and no helmet call is attached to a rider caption.
    assert "No Helmet" not in [
        line for line in captions if line.startswith("Track")
    ]


def test_a_multi_rider_is_never_drawn_in_the_observing_register() -> None:
    """Amber says "the system is arguing something"; about an unresolved rider it is not."""

    provider = HelmetAnalysisOverlayProvider(
        _frames(["no_helmet"] * 6, riders_per_frame=2),
        stabilization=HelmetStabilizationConfig(min_samples=1),
    )

    alerts = {box.alert for box in _boxes(provider.elements_for_frame(_ref(5)))}

    assert alerts == {OverlayAlert.NONE}


def test_a_lone_rider_reading_no_helmet_is_drawn_as_observing() -> None:
    """Visible without being a confirmation -- the honest middle register."""

    provider = HelmetAnalysisOverlayProvider(
        _frames(["no_helmet"] * 6), stabilization=HelmetStabilizationConfig(min_samples=1)
    )

    alerts = {box.alert for box in _boxes(provider.elements_for_frame(_ref(5)))}

    assert OverlayAlert.OBSERVING in alerts
    assert OverlayAlert.CONFIRMED not in alerts


# --- provider protocol --------------------------------------------------------------------
def test_a_run_that_captured_nothing_has_no_content_to_draw() -> None:
    provider = HelmetAnalysisOverlayProvider(())

    assert provider.has_content() is False
    assert provider.elements_for_frame(_ref(0)) == ()


def test_a_frame_with_no_capture_contributes_nothing() -> None:
    provider = HelmetAnalysisOverlayProvider(_frames(["helmet"]))

    assert provider.elements_for_frame(_ref(99)) == ()


def test_the_provider_is_a_pure_lookup_and_replays_identically() -> None:
    provider = HelmetAnalysisOverlayProvider(_frames(["helmet", "no_helmet", "helmet"]))

    assert provider.elements_for_frame(_ref(1)) == provider.elements_for_frame(_ref(1))


def test_it_ignores_the_events_the_driver_passes_every_provider() -> None:
    """An analysis draws what was classified; another rule's events are not its business."""

    from trafficpulse.classifier import StubHelmetClassifier
    from trafficpulse.overlay.providers import register_defaults
    from trafficpulse.overlay.registry import OverlayProviderRegistry
    from trafficpulse.pipeline.helmet_analysis import HelmetAnalysisObserver

    registry = OverlayProviderRegistry()
    register_defaults(registry)
    observer = HelmetAnalysisObserver(classifier=StubHelmetClassifier(), capture_overlay=True)

    provider = registry.create_for(observer, ())

    assert isinstance(provider, HelmetAnalysisOverlayProvider)
    assert provider.violation_kind == "helmet_analysis"


@pytest.mark.parametrize("kind", ["no_helmet", "wrong_way", "triple_riding"])
def test_its_registry_kind_is_not_a_violation(kind: str) -> None:
    assert HelmetAnalysisOverlayProvider.violation_kind != kind
