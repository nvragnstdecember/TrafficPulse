"""The generic overlay metadata model: what to draw, never how or from what.

``OverlayScene`` is the **frozen, framework-neutral, pixel-free** contract between
the inference side of TrafficPulse and its visualization side. An inference
component (via a violation-specific *overlay provider*) emits a scene; a *renderer*
consumes it. Neither the scene nor the renderer knows anything about helmets,
wrong-way, tracking, or any detector -- the scene describes only geometry, text,
and two orthogonal semantic axes, and the renderer draws exactly what it is told.

Why a metadata model at all (and not draw calls)
------------------------------------------------
Separating *what to draw* (this module) from *how it looks* (``theme``) and *how
it is rasterised* (``renderer``) is what lets one renderer serve every present and
future violation without change, keeps the always-installed base package free of
any drawing dependency (Pillow is an optional extra -- see ``renderer``), and makes
overlays testable as data: a provider's output is asserted structurally, with no
image and no GPU.

The two semantic axes (the whole vocabulary a provider speaks)
--------------------------------------------------------------
Every drawable carries two orthogonal tokens the ``theme`` resolves to concrete
pixels, so colour lives in one place and providers stay declarative:

* :class:`OverlayEmphasis` -- *what kind of thing this is* in the scene's visual
  hierarchy (the primary subject, a related object, a sub-region, background
  context). Not a colour; the theme assigns one.
* :class:`OverlayAlert` -- *the temporal reasoning state* (nothing yet, evidence
  accumulating, or a confirmed violation). This is the axis that drives the
  "observing -> confirmed" transition every temporal violation shares.

Keeping these two axes generic -- rather than baking in "no-helmet red" -- is what
makes the model reusable: a speeding or wrong-way provider expresses itself in the
same four emphases and three alert states, and inherits the same look for free.

Z-order
-------
Each element declares an :class:`OverlayLayer`; the renderer draws in ascending
layer so the ordering is a property of the *data*, not of provider or renderer
code. The default layers encode the house order (objects under subjects under
links under regions under labels under banners) but a provider may place an
element on any layer.

Determinism / purity
--------------------
Every type here is a frozen, strict pydantic model of plain scalars: no pixels, no
wall-clock, no numpy, no Pillow. A scene is fully serialisable, so it can be
logged, diffed, cached, or shipped to a different renderer unchanged.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class OverlayEmphasis(StrEnum):
    """*What kind of thing* an element is in the visual hierarchy (theme-resolved).

    A generic vocabulary shared by every violation; the theme maps each to a
    concrete colour/weight. ``SUBJECT`` is the entity a violation is about (a
    rider), ``OBJECT`` a related entity (the motorcycle), ``REGION`` a sub-area of
    interest (the classified head crop), ``CONTEXT`` muted background detail.
    """

    SUBJECT = "subject"
    OBJECT = "object"
    REGION = "region"
    CONTEXT = "context"


class OverlayAlert(StrEnum):
    """The temporal-reasoning state an element reflects (theme-resolved).

    ``NONE`` -- detected, no evidence yet; ``OBSERVING`` -- the temporal reasoner is
    accumulating supporting evidence but has not confirmed; ``CONFIRMED`` -- a
    violation has been confirmed for this element. The theme escalates styling
    across these, which is how the shared "observing -> confirmed" transition is
    expressed once for all violations.
    """

    NONE = "none"
    OBSERVING = "observing"
    CONFIRMED = "confirmed"


class OverlayLayer(IntEnum):
    """Ascending paint order (the renderer draws low to high).

    The default house order; providers assign these but may choose any value.
    Objects sit beneath subjects, association links above both, regions above the
    links, text above all geometry, and the banner on top -- guaranteeing legible
    overlays regardless of emission order.
    """

    BACKDROP = 0
    OBJECT = 10
    SUBJECT = 20
    LINK = 30
    REGION = 40
    LABEL = 50
    BANNER = 60


class Corner(StrEnum):
    """Which corner of a box a caption prefers before collision resolution."""

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OverlayPoint(_Frozen):
    """A point in **image pixel space** (origin top-left, +x right, +y down)."""

    x: float
    y: float


class OverlayCaption(_Frozen):
    """The text attached to a box: one or more lines plus an optional metric.

    ``lines`` are drawn stacked (e.g. ``("Rider", "Track: iou-1")``). ``metric`` is
    an optional emphasised value rendered larger/bolder than the lines (e.g. a
    ``"97%"`` confidence) so the reader's eye lands on the number. ``prefer`` is the
    box corner the caption starts at; the layout pass may move it to avoid overlap.
    """

    lines: tuple[str, ...] = ()
    metric: str | None = None
    prefer: Corner = Corner.TOP_LEFT


class OverlayBox(_Frozen):
    """A rectangle with a semantic role and an optional caption.

    ``bounds`` is ``(x1, y1, x2, y2)`` in image pixels. ``emphasis`` and ``alert``
    are the two theme-resolved axes. ``key`` is an optional stable identifier
    (e.g. a track id) carried only for diagnostics/anchoring; the renderer never
    interprets it.
    """

    kind: Literal["box"] = "box"
    bounds: tuple[float, float, float, float]
    emphasis: OverlayEmphasis
    alert: OverlayAlert = OverlayAlert.NONE
    layer: OverlayLayer = OverlayLayer.SUBJECT
    caption: OverlayCaption | None = None
    key: str | None = None


class OverlayLink(_Frozen):
    """A polyline connecting anchor points -- e.g. the head->rider->motorcycle chain.

    ``points`` are visited in order (>= 2). The link expresses an inference
    *relationship*; its emphasis/alert let the theme tie it visually to the
    entities it joins (and turn it red on confirmation).
    """

    kind: Literal["link"] = "link"
    points: tuple[OverlayPoint, ...] = Field(min_length=2)
    emphasis: OverlayEmphasis = OverlayEmphasis.SUBJECT
    alert: OverlayAlert = OverlayAlert.NONE
    layer: OverlayLayer = OverlayLayer.LINK


class OverlayBanner(_Frozen):
    """A pinned headline (e.g. a confirmed-violation banner).

    ``title`` is the headline (``"NO HELMET"``); ``lines`` are supporting detail
    (track, timestamp, violation id). ``icon`` is an optional leading glyph
    (``"!"``). Positioned by the renderer at ``corner`` of the frame, stacking when
    several banners are present, so it never depends on any box's geometry.
    """

    kind: Literal["banner"] = "banner"
    title: str
    lines: tuple[str, ...] = ()
    icon: str | None = None
    alert: OverlayAlert = OverlayAlert.CONFIRMED
    layer: OverlayLayer = OverlayLayer.BANNER
    corner: Corner = Corner.TOP_LEFT


OverlayElement: TypeAlias = Annotated[
    OverlayBox | OverlayLink | OverlayBanner, Field(discriminator="kind")
]


class OverlayScene(_Frozen):
    """One frame's complete, renderer-ready description.

    ``width`` / ``height`` are the target image dimensions the coordinates are in
    (a scene is meaningful independent of any particular image, and the renderer
    validates the image matches). ``elements`` is the flat, order-independent set of
    drawables; the renderer sorts them by :class:`OverlayLayer`. ``frame_index`` and
    ``media_seconds`` are optional provenance for logging/replay -- never drawn
    unless a provider also put them in a banner.
    """

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    elements: tuple[OverlayElement, ...] = ()
    frame_index: int | None = None
    media_seconds: float | None = None

    def merged(self, *others: OverlayScene) -> OverlayScene:
        """Return a scene combining this scene's elements with ``others``'.

        The compositor uses this to fuse the contributions of several overlay
        providers for one frame into a single scene. Dimensions/identity are taken
        from ``self``; only elements accumulate.
        """

        elements = self.elements + tuple(e for other in others for e in other.elements)
        return self.model_copy(update={"elements": elements})
