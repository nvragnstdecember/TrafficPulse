"""The overlay theme: the one place semantic tokens become concrete pixels.

A provider speaks only in :class:`~trafficpulse.overlay.metadata.OverlayEmphasis`
and :class:`~trafficpulse.overlay.metadata.OverlayAlert`; the theme resolves each
``(emphasis, alert)`` pair to a concrete :class:`BoxStyle` (stroke colour + width,
fill, label chip, metric colour) and likewise for links and banners. Centralising
colour here is what keeps overlays *consistent* across every violation and lets the
whole look change in one edit -- providers never name a colour.

Palette provenance (matches the shipped app)
--------------------------------------------
The default palette is the frontend's own design tokens
(``frontend/src/styles/globals.css``) converted to RGB, so a rendered frame and the
web UI read as one product:

* rider / subject   -> ``--success``     green   (142 71% 40%)  -> (30, 174, 83)
* head / region     -> amber-400         yellow                  -> (250, 204, 21)
* observing accent  -> ``--warning``     amber   (32 95% 44%)   -> (217, 119, 6)
* confirmed / alert -> ``--destructive`` red     (0 84% 60%)    -> (239, 68, 68)
* context (muted)   -> slate-400                                  -> (148, 163, 184)

Motorcycle/object uses a blue (217 91% 60% -> (59, 130, 246)) that harmonises with
the token set (the app has no dedicated object hue; this is the one added colour).

Escalation along the alert axis
-------------------------------
Stroke colour escalates in **three** steps, not two, so a viewer can read the
reasoning state without waiting for a confirmation:

* ``NONE``      -- the resting palette above (rider green, bike blue, region yellow);
* ``OBSERVING`` -- the *subject* warms to amber and the *region* (the sub-area whose
  reading is the adverse evidence -- a head crop the classifier read as "no helmet")
  turns red, while the related *object* stays blue: the model's current opinion is
  visible, but the vehicle is not yet implicated;
* ``CONFIRMED`` -- everything goes red, one pixel wider, with a faint fill wash and a
  red caption chip.

Consequently the region box is red exactly when the classifier is asserting the
adverse reading for that frame -- and yellow when it is not -- rather than being
permanently red, which would spend the strongest colour convention in the palette
on a box that is present for compliant riders too.

No pixels, no Pillow
--------------------
The theme is pure data (RGB(A) tuples + sizes). It imports nothing from the
renderer and never touches an image, so it is usable -- and testable -- in the
Pillow-free base install.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metadata import OverlayAlert, OverlayEmphasis

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

# --- palette (frontend design tokens -> RGB; see module docstring) ------------
_GREEN: RGB = (30, 174, 83)
_BLUE: RGB = (59, 130, 246)
_YELLOW: RGB = (250, 204, 21)
_AMBER: RGB = (217, 119, 6)
_SLATE: RGB = (148, 163, 184)
_RED: RGB = (239, 68, 68)
_RED_BRIGHT: RGB = (255, 71, 71)
_WHITE: RGB = (255, 255, 255)

# label-chip backgrounds by alert state (translucent so context shows through)
_CHIP_NEUTRAL: RGBA = (15, 23, 42, 214)  # slate-900
_CHIP_OBSERVING: RGBA = (180, 83, 9, 224)  # amber-700
_CHIP_CONFIRMED: RGBA = (185, 28, 28, 232)  # red-700
# A caption's progress meter: white on a translucent dark track. Deliberately NOT
# the box's own colour -- the caption chip is already tinted by the alert state, so
# an amber bar on an amber chip (or red on red) is invisible exactly when the meter
# matters most. The chip carries the semantics; the bar only has to be readable.
_METER_TRACK: RGBA = (8, 12, 24, 170)


@dataclass(frozen=True)
class BoxStyle:
    """Resolved styling for one rectangle + its caption."""

    stroke: RGB
    stroke_width: int
    fill: RGBA | None
    label_text: RGB
    label_bg: RGBA
    metric_text: RGB
    # The caption's optional progress meter: an unfilled track plus a filled bar.
    meter_track: RGBA = (8, 12, 24, 170)
    meter_fill: RGB = _WHITE


@dataclass(frozen=True)
class LinkStyle:
    """Resolved styling for an association polyline."""

    stroke: RGB
    stroke_width: int
    node_radius: int


@dataclass(frozen=True)
class BannerStyle:
    """Resolved styling for a pinned banner."""

    background: RGBA
    title_text: RGB
    body_text: RGB
    accent: RGB


@dataclass(frozen=True)
class Typography:
    """Font sizes (px) and spacing the renderer uses; scaled to the frame."""

    caption_line: int = 13
    caption_metric: int = 20
    caption_title: int = 15
    banner_title: int = 22
    banner_line: int = 14
    pad: int = 6
    line_gap: int = 2


# base (nothing observed yet) stroke colour per emphasis
_BASE_STROKE: dict[OverlayEmphasis, RGB] = {
    OverlayEmphasis.SUBJECT: _GREEN,
    OverlayEmphasis.OBJECT: _BLUE,
    OverlayEmphasis.REGION: _YELLOW,
    OverlayEmphasis.CONTEXT: _SLATE,
}
# evidence-accumulating stroke colour per emphasis. The alert axis escalates in
# three visible steps, not two: the *subject* warms to amber (something is being
# argued about it) while the *region* -- the sub-area whose reading is the adverse
# evidence, e.g. a head crop classified "no helmet" -- goes red immediately, so a
# reader can see what the model currently thinks before any confirmation exists.
# The related object (the vehicle) deliberately does not change: it is context
# until the violation is confirmed, and colouring it early would overstate the case.
_OBSERVING_STROKE: dict[OverlayEmphasis, RGB] = {
    OverlayEmphasis.SUBJECT: _AMBER,
    OverlayEmphasis.OBJECT: _BLUE,
    OverlayEmphasis.REGION: _RED,
    OverlayEmphasis.CONTEXT: _SLATE,
}
# confirmed stroke colour per emphasis (region gets the brightest red)
_ALERT_STROKE: dict[OverlayEmphasis, RGB] = {
    OverlayEmphasis.SUBJECT: _RED,
    OverlayEmphasis.OBJECT: _RED,
    OverlayEmphasis.REGION: _RED_BRIGHT,
    OverlayEmphasis.CONTEXT: _RED,
}
_STROKE_BY_ALERT: dict[OverlayAlert, dict[OverlayEmphasis, RGB]] = {
    OverlayAlert.NONE: _BASE_STROKE,
    OverlayAlert.OBSERVING: _OBSERVING_STROKE,
    OverlayAlert.CONFIRMED: _ALERT_STROKE,
}
_BASE_WIDTH: dict[OverlayEmphasis, int] = {
    OverlayEmphasis.SUBJECT: 3,
    OverlayEmphasis.OBJECT: 3,
    OverlayEmphasis.REGION: 2,
    OverlayEmphasis.CONTEXT: 1,
}
_CHIP: dict[OverlayAlert, RGBA] = {
    OverlayAlert.NONE: _CHIP_NEUTRAL,
    OverlayAlert.OBSERVING: _CHIP_OBSERVING,
    OverlayAlert.CONFIRMED: _CHIP_CONFIRMED,
}


@dataclass(frozen=True)
class OverlayTheme:
    """Resolves generic tokens to concrete styling (see module docstring)."""

    typography: Typography = field(default_factory=Typography)
    confirmed_fill_alpha: int = 46  # faint red wash inside a confirmed box

    def box_style(self, emphasis: OverlayEmphasis, alert: OverlayAlert) -> BoxStyle:
        confirmed = alert is OverlayAlert.CONFIRMED
        stroke = _STROKE_BY_ALERT[alert][emphasis]
        width = _BASE_WIDTH[emphasis] + (1 if confirmed else 0)
        fill: RGBA | None = (*stroke, self.confirmed_fill_alpha) if confirmed else None
        return BoxStyle(
            stroke=stroke,
            stroke_width=width,
            fill=fill,
            label_text=_WHITE,
            label_bg=_CHIP[alert],
            metric_text=_WHITE,
            meter_track=_METER_TRACK,
            meter_fill=_WHITE,
        )

    def link_style(self, emphasis: OverlayEmphasis, alert: OverlayAlert) -> LinkStyle:
        confirmed = alert is OverlayAlert.CONFIRMED
        stroke = _RED if confirmed else _BASE_STROKE[emphasis]
        return LinkStyle(stroke=stroke, stroke_width=2, node_radius=3)

    def banner_style(self, alert: OverlayAlert) -> BannerStyle:
        return BannerStyle(
            background=_CHIP[alert],
            title_text=_WHITE,
            body_text=_WHITE,
            accent=_ALERT_STROKE[OverlayEmphasis.REGION]
            if alert is OverlayAlert.CONFIRMED
            else _AMBER,
        )


DEFAULT_THEME = OverlayTheme()
