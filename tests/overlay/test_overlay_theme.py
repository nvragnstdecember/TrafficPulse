"""The theme: semantic tokens -> concrete styling."""

from __future__ import annotations

from trafficpulse.overlay import DEFAULT_THEME, OverlayAlert, OverlayEmphasis

T = DEFAULT_THEME


def test_base_emphasis_colours_match_the_spec() -> None:
    assert T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.NONE).stroke == (30, 174, 83)  # green
    assert T.box_style(OverlayEmphasis.OBJECT, OverlayAlert.NONE).stroke == (59, 130, 246)  # blue
    assert T.box_style(OverlayEmphasis.REGION, OverlayAlert.NONE).stroke == (250, 204, 21)  # yellow


def test_observing_warms_the_subject_and_reddens_the_region() -> None:
    # The alert axis escalates in three steps, so the model's current reading is
    # visible before any confirmation exists.
    base = T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.NONE)
    observing = T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.OBSERVING)
    assert observing.stroke == (217, 119, 6)  # amber: something is being argued
    assert observing.label_bg != base.label_bg  # amber chip signals the state
    # the sub-region carrying the adverse reading (a bare-head crop) goes red now
    assert T.box_style(OverlayEmphasis.REGION, OverlayAlert.OBSERVING).stroke == (239, 68, 68)


def test_observing_leaves_the_related_object_alone() -> None:
    # The vehicle is context until the violation is confirmed; colouring it while
    # evidence is still accumulating would overstate the case.
    assert T.box_style(OverlayEmphasis.OBJECT, OverlayAlert.OBSERVING).stroke == (59, 130, 246)


def test_observing_is_not_yet_a_confirmation() -> None:
    observing = T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.OBSERVING)
    confirmed = T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.CONFIRMED)
    assert observing.fill is None  # the fill wash is reserved for confirmation
    assert observing.stroke_width < confirmed.stroke_width


def test_confirmation_turns_everything_red_and_head_brightest() -> None:
    subject = T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.CONFIRMED)
    obj = T.box_style(OverlayEmphasis.OBJECT, OverlayAlert.CONFIRMED)
    region = T.box_style(OverlayEmphasis.REGION, OverlayAlert.CONFIRMED)
    assert subject.stroke == (239, 68, 68)
    assert obj.stroke == (239, 68, 68)
    assert region.stroke == (255, 71, 71)  # brightest red for the head
    # confirmed boxes get a faint fill wash and a thicker stroke
    assert subject.fill is not None
    base_width = T.box_style(OverlayEmphasis.SUBJECT, OverlayAlert.NONE).stroke_width
    assert subject.stroke_width > base_width


def test_links_and_banners_follow_alert() -> None:
    assert T.link_style(OverlayEmphasis.SUBJECT, OverlayAlert.CONFIRMED).stroke == (239, 68, 68)
    assert T.link_style(OverlayEmphasis.SUBJECT, OverlayAlert.NONE).stroke == (30, 174, 83)
    assert T.banner_style(OverlayAlert.CONFIRMED).accent == (255, 71, 71)
