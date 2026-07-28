"""The overlay renderer: draw an ``OverlayScene`` onto a frame, nothing more.

The renderer is the *only* part of the framework that touches pixels, and it
contains **no violation logic** -- it draws boxes, links, captions (their lines, an
emphasised metric, and a generic 0..1 progress meter), and banners exactly as the
scene describes, styled by the theme. It never asks *what* a meter is measuring or
*why* a box is red. Add a new violation and this file never changes.

Backend seam (base install stays Pillow-free)
---------------------------------------------
:class:`OverlayRenderer` is a small protocol. :class:`PillowOverlayRenderer` is the
shipped implementation; it imports Pillow **lazily** inside ``__init__`` (Pillow is
the optional ``rtdetr`` extra -- it is already required by the real detector/
classifier backends), so importing this module pulls in no drawing dependency and
the metadata/theme/providers stay usable in the base install. A missing Pillow
raises the typed :class:`OverlayBackendUnavailableError`, never a bare ``ImportError``.

Draw order & performance
------------------------
Elements are drawn in ascending :class:`~trafficpulse.overlay.metadata.OverlayLayer`
(objects, subjects, links, regions, labels, banner), so overlays are always legible
regardless of emission order. Captions are laid out by the pure
:mod:`~trafficpulse.overlay.layout` solver (which the renderer feeds measured text
sizes) so labels never stack unreadably. Cost is O(number of elements) with no
model inference and no per-pixel scene scan -- rendering a frame is linear in the
handful of tracked objects on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .layout import LabelRequest, place_labels
from .metadata import (
    Corner,
    OverlayBanner,
    OverlayBox,
    OverlayCaption,
    OverlayLink,
    OverlayScene,
)
from .theme import DEFAULT_THEME, OverlayTheme


class OverlayError(Exception):
    """Base class for overlay-rendering errors."""


class OverlayBackendUnavailableError(OverlayError):
    """The drawing backend (Pillow) is not installed.

    Pillow ships with the optional ``rtdetr`` extra (it already backs the real
    detector/classifier). Install with ``pip install 'trafficpulse[rtdetr]'``.
    """


class FrameSizeMismatchError(OverlayError):
    """The image passed to :meth:`render` does not match the scene's dimensions."""


class OverlayRenderer(Protocol):
    """Draws a scene onto an RGB ``uint8`` image, returning a new image."""

    def render(self, image: NDArray[np.uint8], scene: OverlayScene) -> NDArray[np.uint8]:
        """Return a copy of ``image`` with ``scene`` drawn on it."""
        ...


class PillowOverlayRenderer:
    """The shipped renderer (lazy-Pillow; see module docstring)."""

    def __init__(self, theme: OverlayTheme = DEFAULT_THEME, *, scale: float | None = None) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:  # pragma: no cover - exercised only without Pillow
            raise OverlayBackendUnavailableError(
                "the overlay renderer needs Pillow (the optional 'rtdetr' extra); "
                "install with: pip install 'trafficpulse[rtdetr]'"
            ) from exc
        self._Image = Image
        self._ImageDraw = ImageDraw
        self._ImageFont = ImageFont
        self._theme = theme
        self._scale = scale
        self._font_cache: dict[int, Any] = {}

    # A cross-platform ladder of scalable sans-serif faces; the first that resolves
    # wins. Falls back to Pillow's own scalable default (>=10) so text is never
    # rendered with the tiny non-scalable bitmap face.
    _FONT_CANDIDATES = (
        "DejaVuSans.ttf",  # Linux/CI (and Pillow's bundled face on many builds)
        "Arial.ttf",
        "arial.ttf",  # Windows
        "Helvetica.ttc",  # macOS
    )

    def _font(self, size: int) -> Any:
        if size not in self._font_cache:
            self._font_cache[size] = self._resolve_font(size)
        return self._font_cache[size]

    def _resolve_font(self, size: int) -> Any:
        for name in self._FONT_CANDIDATES:
            try:
                return self._ImageFont.truetype(name, size)
            except OSError:
                continue
        try:  # Pillow >= 10 returns a *scalable* default when given a size
            return self._ImageFont.load_default(size=size)
        except TypeError:  # pragma: no cover - very old Pillow
            return self._ImageFont.load_default()

    def _text_size(self, draw: Any, text: str, font: Any) -> tuple[float, float]:
        x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=font)
        return x2 - x1, y2 - y1

    def _line_advance(self, draw: Any, text: str, font: Any) -> float:
        """Distance from a line's draw origin to the next one's.

        The *bottom* of the glyph box measured from the origin -- not its height.
        Height excludes the font's top bearing, so stacking by height creeps upward
        a little per line and eventually collides with whatever follows the text.
        Measurement and painting both advance by this, so a chip is always exactly
        as tall as what is drawn inside it.
        """

        return float(draw.textbbox((0, 0), text, font=font)[3])

    # --- public API ---------------------------------------------------------
    def render(self, image: NDArray[np.uint8], scene: OverlayScene) -> NDArray[np.uint8]:
        h, w = int(image.shape[0]), int(image.shape[1])
        if (w, h) != (scene.width, scene.height):
            raise FrameSizeMismatchError(
                f"image is {w}x{h} but scene declares {scene.width}x{scene.height}"
            )
        base = self._Image.fromarray(np.ascontiguousarray(image)).convert("RGBA")
        draw_layer = self._Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = self._ImageDraw.Draw(draw_layer)
        typ = self._theme.typography
        s = self._scale if self._scale is not None else max(1.0, min(w, h) / 540.0)

        # Ascending layer order -> legible overlays independent of emission order.
        ordered = sorted(scene.elements, key=lambda e: int(e.layer))
        boxes = [e for e in ordered if isinstance(e, OverlayBox)]
        banners = [e for e in ordered if isinstance(e, OverlayBanner)]

        # 1..4: geometry (boxes below, links between, region boxes above) in layer order.
        for element in ordered:
            if isinstance(element, OverlayBox):
                self._draw_box(draw, element)
            elif isinstance(element, OverlayLink):
                self._draw_link(draw, element)

        # 5: banners are positioned first (pinned, painted last) so the caption
        # solver can route captions out from under them.
        banner_layout = self._layout_banners(draw, banners, w, h, s, typ)

        # 6: captions, deconflicted against each other and the banners.
        self._draw_captions(draw, boxes, w, h, s, typ, [rect for _, rect in banner_layout])

        # 7: banners, on top.
        self._draw_banners(draw, banner_layout, s, typ)

        out = self._Image.alpha_composite(base, draw_layer).convert("RGB")
        return np.asarray(out, dtype=np.uint8)

    # --- element painters ---------------------------------------------------
    def _draw_box(self, draw: Any, box: OverlayBox) -> None:
        style = self._theme.box_style(box.emphasis, box.alert)
        x1, y1, x2, y2 = box.bounds
        if style.fill is not None:
            draw.rectangle((x1, y1, x2, y2), fill=style.fill)
        draw.rectangle((x1, y1, x2, y2), outline=(*style.stroke, 255), width=style.stroke_width)

    def _draw_link(self, draw: Any, link: OverlayLink) -> None:
        style = self._theme.link_style(link.emphasis, link.alert)
        pts = [(p.x, p.y) for p in link.points]
        draw.line(pts, fill=(*style.stroke, 235), width=style.stroke_width, joint="curve")
        r = style.node_radius
        for x, y in pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(*style.stroke, 255))

    def _draw_captions(
        self,
        draw: Any,
        boxes: list[OverlayBox],
        w: int,
        h: int,
        s: float,
        typ: Any,
        blocked: Sequence[tuple[float, float, float, float]] = (),
    ) -> None:
        line_font = self._font(int(typ.caption_line * s))
        title_font = self._font(int(typ.caption_title * s))
        metric_font = self._font(int(typ.caption_metric * s))
        pad = typ.pad * s
        gap = typ.line_gap * s

        captioned: list[tuple[OverlayBox, OverlayCaption]] = [
            (b, b.caption) for b in boxes if b.caption is not None
        ]
        measured = [
            self._measure_caption(draw, cap, line_font, title_font, metric_font, gap)
            for _, cap in captioned
        ]
        requests = [
            LabelRequest(box=b.bounds, width=mw, height=mh, prefer=cap.prefer, pad=pad)
            for (b, cap), (mw, mh) in zip(captioned, measured, strict=True)
        ]
        positions = place_labels(requests, float(w), float(h), blocked=blocked)
        for (box, cap), rect in zip(captioned, positions, strict=True):
            self._paint_caption(draw, box, cap, rect, line_font, title_font, metric_font, pad, gap)

    def _measure_caption(
        self, draw: Any, caption: OverlayCaption, line_font: Any, title_font: Any,
        metric_font: Any, gap: float,
    ) -> tuple[float, float]:
        widths: list[float] = []
        height = 0.0
        for i, line in enumerate(caption.lines):
            font = title_font if i == 0 else line_font
            widths.append(self._text_size(draw, line, font)[0])
            height += self._line_advance(draw, line, font) + gap
        if caption.metric is not None:
            widths.append(self._text_size(draw, caption.metric, metric_font)[0])
            height += self._line_advance(draw, caption.metric, metric_font) + gap
        if caption.progress is not None:
            widths.append(self._meter_width(gap))
            # two gaps: one above the bar, one keeping it off the chip's bottom edge
            height += self._meter_height(gap) + gap * 2
        pad = gap * 2
        return (max(widths, default=0.0) + pad * 2, height + pad)

    # A meter is sized from the caption's own spacing unit, so it scales with the
    # frame exactly as the text does and needs no separate theme knob.
    def _meter_width(self, gap: float) -> float:
        return max(48.0, gap * 32.0)

    def _meter_height(self, gap: float) -> float:
        return max(3.0, gap * 2.0)

    def _paint_caption(
        self, draw: Any, box: OverlayBox, caption: OverlayCaption,
        rect: tuple[float, float, float, float], line_font: Any, title_font: Any,
        metric_font: Any, pad: float, gap: float,
    ) -> None:
        style = self._theme.box_style(box.emphasis, box.alert)
        x1, y1, x2, y2 = rect
        radius = max(3.0, pad)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=style.label_bg)
        # accent bar keyed to the box stroke, so caption <-> box association is obvious
        draw.rounded_rectangle((x1, y1, x1 + max(3.0, pad * 0.5), y2), radius=radius,
                               fill=(*style.stroke, 255))
        tx, ty = x1 + pad * 1.6, y1 + pad * 0.6
        for i, line in enumerate(caption.lines):
            font = title_font if i == 0 else line_font
            draw.text((tx, ty), line, font=font, fill=(*style.label_text, 255))
            ty += self._line_advance(draw, line, font) + gap
        if caption.metric is not None:
            draw.text((tx, ty), caption.metric, font=metric_font, fill=(*style.metric_text, 255))
            ty += self._line_advance(draw, caption.metric, metric_font) + gap
        if caption.progress is not None:
            # +gap of leading so the bar reads as its own row, not a strikethrough
            # under the line above it (the measure pass reserves the same space).
            self._paint_meter(draw, caption.progress, tx, ty + gap, gap, style)

    def _paint_meter(
        self, draw: Any, progress: float, x: float, y: float, gap: float, style: Any
    ) -> None:
        """Draw a caption's generic 0..1 observation-state meter (track + fill).

        Knows nothing about what is progressing -- it paints a fraction. Zero
        progress still paints the track, so "nothing yet" is visibly distinct from
        "no meter at all".
        """

        w, h = self._meter_width(gap), self._meter_height(gap)
        radius = h / 2.0
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=style.meter_track)
        filled = w * max(0.0, min(1.0, progress))
        if filled > 0.0:
            draw.rounded_rectangle(
                (x, y, x + max(filled, h), y + h), radius=radius, fill=(*style.meter_fill, 255)
            )

    def _layout_banners(
        self, draw: Any, banners: list[OverlayBanner], w: int, h: int, s: float, typ: Any
    ) -> list[tuple[OverlayBanner, tuple[float, float, float, float]]]:
        """Measure + position every banner, without drawing.

        Split from painting so the caption solver can treat the resulting rects as
        occupied space: banners are pinned and painted last, so a caption placed
        under one would be hidden -- typically the caption of the very track the
        banner is about.
        """

        title_font = self._font(int(typ.banner_title * s))
        line_font = self._font(int(typ.banner_line * s))
        pad = typ.pad * s * 1.5
        gap = typ.line_gap * s
        # stack banners per corner so several confirmed violations never overlap
        offsets: dict[Corner, float] = {}
        placed: list[tuple[OverlayBanner, tuple[float, float, float, float]]] = []
        for banner in banners:
            tw, th = self._text_size(draw, banner.title, title_font)
            icon_w = (th * 1.3 + gap) if banner.icon else 0.0
            widths = [tw + icon_w]
            total_h = self._line_advance(draw, banner.title, title_font) + gap
            for line in banner.lines:
                widths.append(self._text_size(draw, line, line_font)[0])
                total_h += self._line_advance(draw, line, line_font) + gap
            bw = max(widths) + pad * 2
            bh = total_h + pad
            corner = banner.corner
            oy = offsets.get(corner, 0.0)
            x1 = pad if corner in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT) else w - bw - pad
            y1 = (pad + oy) if corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT) else h - bh - pad - oy
            placed.append((banner, (x1, y1, x1 + bw, y1 + bh)))
            offsets[corner] = oy + bh + pad
        return placed

    def _draw_banners(
        self,
        draw: Any,
        layout: list[tuple[OverlayBanner, tuple[float, float, float, float]]],
        s: float,
        typ: Any,
    ) -> None:
        title_font = self._font(int(typ.banner_title * s))
        line_font = self._font(int(typ.banner_line * s))
        pad = typ.pad * s * 1.5
        gap = typ.line_gap * s
        for banner, (x1, y1, x2, y2) in layout:
            style = self._theme.banner_style(banner.alert)
            th = self._text_size(draw, banner.title, title_font)[1]
            icon_w = (th * 1.3 + gap) if banner.icon else 0.0
            draw.rounded_rectangle((x1, y1, x2, y2), radius=pad, fill=style.background)
            draw.rounded_rectangle((x1, y1, x1 + max(4.0, pad * 0.4), y2), radius=pad,
                                   fill=(*style.accent, 255))
            tx, ty = x1 + pad * 1.4, y1 + pad * 0.5
            if banner.icon:
                self._draw_warning_icon(draw, tx, ty, th, style.accent)
                tx += icon_w
            draw.text((tx, ty), banner.title, font=title_font, fill=(*style.title_text, 255))
            ty += self._line_advance(draw, banner.title, title_font) + gap
            for line in banner.lines:
                draw.text((x1 + pad * 1.4, ty), line, font=line_font, fill=(*style.body_text, 255))
                ty += self._line_advance(draw, line, line_font) + gap

    def _draw_warning_icon(self, draw: Any, x: float, y: float, size: float, accent: Any) -> None:
        """A font-independent warning triangle with an exclamation (generic alert mark)."""

        apex = (x + size / 2, y)
        left = (x, y + size)
        right = (x + size, y + size)
        draw.polygon([apex, left, right], fill=(*accent, 255), outline=(255, 255, 255, 255))
        cx = x + size / 2
        draw.line([(cx, y + size * 0.32), (cx, y + size * 0.66)], fill=(255, 255, 255, 255),
                  width=max(1, int(size * 0.10)))
        r = max(1.0, size * 0.06)
        draw.ellipse((cx - r, y + size * 0.74 - r, cx + r, y + size * 0.74 + r),
                     fill=(255, 255, 255, 255))
