"""Presentation-only temporal smoothing (EMA + deadband over keyed streams)."""

from __future__ import annotations

import pytest

from trafficpulse.overlay.smoothing import SmoothingConfig, StreamSmoother


def test_the_first_observation_passes_through_untouched() -> None:
    # Nothing to average against yet; inventing a starting point would draw the box
    # somewhere it was never observed.
    smoother = StreamSmoother()
    assert smoother.box("a", 0, (10.0, 20.0, 30.0, 40.0)) == (10.0, 20.0, 30.0, 40.0)


def test_jitter_is_damped_towards_the_running_average() -> None:
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.5, box_deadband_px=0.0))
    smoother.box("a", 0, (0.0, 0.0, 100.0, 100.0))
    out = smoother.box("a", 1, (10.0, 0.0, 100.0, 100.0))
    assert out[0] == pytest.approx(5.0)  # half way, not the full jump


def test_sustained_motion_is_followed_not_suppressed() -> None:
    # Smoothing must lag genuine movement, never refuse it: a vehicle crossing the
    # frame has to end up where it actually is.
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.5, box_deadband_px=0.0))
    for i in range(40):
        out = smoother.box("a", i, (float(i), 0.0, float(i) + 50.0, 50.0))
    assert out[0] == pytest.approx(39.0, abs=1.5)


def test_the_deadband_holds_a_barely_moving_box_perfectly_still() -> None:
    # The anti-flicker mechanism: sub-threshold shake republishes nothing at all,
    # so the drawn box is bit-identical frame to frame.
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.5, box_deadband_px=2.0))
    first = smoother.box("a", 0, (100.0, 100.0, 200.0, 200.0))
    for i in range(1, 12):
        wobble = 0.6 if i % 2 else -0.6
        assert smoother.box("a", i, (100.0 + wobble, 100.0, 200.0, 200.0)) == first


def test_the_deadband_yields_once_the_drift_is_real() -> None:
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.6, box_deadband_px=2.0))
    first = smoother.box("a", 0, (100.0, 100.0, 200.0, 200.0))
    moved = smoother.box("a", 1, (140.0, 100.0, 240.0, 200.0))
    assert moved != first


def test_streams_are_independent_per_key() -> None:
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.5, box_deadband_px=0.0))
    smoother.box("a", 0, (0.0, 0.0, 10.0, 10.0))
    # A different key must not inherit "a"'s history.
    assert smoother.box("b", 0, (500.0, 500.0, 510.0, 510.0))[0] == 500.0


def test_a_track_returning_after_a_gap_restarts_rather_than_gliding() -> None:
    # Interpolating across the gap would draw the object travelling through
    # positions nothing was ever observed at.
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.5, max_gap_frames=3))
    smoother.box("a", 0, (0.0, 0.0, 10.0, 10.0))
    resumed = smoother.box("a", 50, (900.0, 0.0, 910.0, 10.0))
    assert resumed == (900.0, 0.0, 910.0, 10.0)


def test_a_brief_dropout_within_tolerance_keeps_its_history() -> None:
    smoother = StreamSmoother(SmoothingConfig(box_alpha=0.5, box_deadband_px=0.0, max_gap_frames=3))
    smoother.box("a", 0, (0.0, 0.0, 10.0, 10.0))
    out = smoother.box("a", 2, (100.0, 0.0, 110.0, 10.0))
    assert out[0] == pytest.approx(50.0)  # averaged, not restarted


def test_a_displayed_score_stops_churning() -> None:
    # The reason a percentage readout is smoothed at all: a score oscillating within
    # the deadband must render one stable figure, not tick every frame.
    smoother = StreamSmoother(SmoothingConfig(value_alpha=0.3, value_deadband=0.02))
    first = smoother.value("s", 0, 0.90)
    seen = {smoother.value("s", i, 0.90 + (0.005 if i % 2 else -0.005)) for i in range(1, 15)}
    assert seen == {first}


def test_a_real_score_change_is_reported() -> None:
    smoother = StreamSmoother(SmoothingConfig(value_alpha=0.5, value_deadband=0.02))
    smoother.value("s", 0, 0.20)
    assert smoother.value("s", 1, 0.90) == pytest.approx(0.55)


def test_an_absent_measurement_is_never_smoothed_into_a_number() -> None:
    # None means "not measured" (a gated crop), not zero. It must pass through and
    # leave the running state intact so the stream resumes cleanly.
    smoother = StreamSmoother(SmoothingConfig(value_alpha=0.5, value_deadband=0.0))
    smoother.value("s", 0, 0.80)
    assert smoother.value("s", 1, None) is None
    assert smoother.value("s", 2, 0.80) == pytest.approx(0.80)


def test_smoothing_is_deterministic() -> None:
    def run() -> list[tuple[float, float, float, float]]:
        smoother = StreamSmoother()
        return [smoother.box("a", i, (float(i), 1.0, float(i) + 9.0, 11.0)) for i in range(20)]

    assert run() == run()
