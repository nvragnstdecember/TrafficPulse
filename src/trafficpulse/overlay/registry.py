"""Overlay providers, the compositor, and the provider registry.

This module is the seam that makes the overlay framework extensible without ever
touching the renderer. It defines three things and no violation logic:

* :class:`OverlayFrameRef` -- the generic per-frame identity the compositor hands
  every provider (camera, frame index, media time, image size). It carries no
  violation data.
* :class:`OverlayProvider` -- the protocol each violation implements: given a frame
  ref, return the :class:`~trafficpulse.overlay.metadata.OverlayElement`\\ s that
  violation wants drawn on that frame, sourced entirely from inference metadata the
  provider was constructed with (no model runs here).
* :class:`OverlayCompositor` -- fuses the elements of every registered provider into
  one :class:`~trafficpulse.overlay.metadata.OverlayScene` per frame, which the
  renderer then draws. The compositor knows nothing about helmets, speed, or lanes;
  it only concatenates and orders.

Extensibility (how a future violation plugs in)
-----------------------------------------------
A new violation contributes exactly two things and changes no shared code: a
per-frame *metadata* shape it already has from its own reasoning, and an
``OverlayProvider`` that maps that metadata to generic elements. It registers a
factory under its violation kind via :class:`OverlayProviderRegistry`; a driver
(e.g. the engine) builds a provider per active rule and hands them to one
compositor. The renderer, the metadata model, and the theme are untouched. See
``providers/no_helmet.py`` for the reference implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .metadata import OverlayElement, OverlayScene


class OverlayFrameRef(BaseModel):
    """Generic per-frame identity + geometry passed to every provider.

    Deliberately violation-neutral: it says *which* frame is being drawn and how
    big it is, never *what* is on it. Providers look up their own metadata by
    ``frame_index`` (or ``media_seconds``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str
    frame_index: int = Field(ge=0)
    media_seconds: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)


@runtime_checkable
class OverlayProvider(Protocol):
    """A violation's contribution to the overlay: metadata in, elements out.

    Implementations are constructed with their own inference outputs (already
    produced -- a provider runs no detection/tracking/association/classification and
    recomputes nothing expensive) and return, per frame, the generic elements that
    violation wants drawn. ``violation_kind`` is the registry key.
    """

    @property
    def violation_kind(self) -> str:
        """The registry key for this provider (e.g. ``"no_helmet"``)."""
        ...

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        """Return the elements to draw for ``frame`` (possibly empty)."""
        ...

    def has_content(self) -> bool:
        """Whether this provider has anything to contribute to the run at all.

        Asked **once**, by the driver, to decide whether an annotated video is worth
        producing -- not per frame. ``False`` means the rule ran but published no
        overlay metadata (its capture was disabled, or it observed nothing), which
        is why a run can legitimately have events and still have no overlay.

        The answer is the provider's own, because "nothing to draw" is
        violation-specific: a provider carrying scene geometry has something worth
        showing on every frame even when no track was ever seen -- a mis-drawn zone
        should be visible immediately rather than deduced from an absence -- while a
        provider whose only content is per-frame capture has nothing without it.
        """
        ...


class OverlayCompositor:
    """Fuses every provider's per-frame elements into one scene.

    Holds an ordered list of providers; :meth:`scene_for` calls each and merges
    their elements. Ordering across providers is stable (registration order); the
    renderer's :class:`~trafficpulse.overlay.metadata.OverlayLayer` sort is what
    ultimately governs paint order, so two providers' elements interleave correctly
    by layer regardless of which provider emitted them.
    """

    def __init__(self, providers: Sequence[OverlayProvider] = ()) -> None:
        self._providers: list[OverlayProvider] = list(providers)

    def add(self, provider: OverlayProvider) -> None:
        self._providers.append(provider)

    @property
    def providers(self) -> tuple[OverlayProvider, ...]:
        return tuple(self._providers)

    def scene_for(self, frame: OverlayFrameRef) -> OverlayScene:
        """Build the merged scene for one frame from all providers."""

        elements: list[OverlayElement] = []
        for provider in self._providers:
            elements.extend(provider.elements_for_frame(frame))
        return OverlayScene(
            width=frame.width,
            height=frame.height,
            elements=tuple(elements),
            frame_index=frame.frame_index,
            media_seconds=frame.media_seconds,
        )


# A factory builds a provider from that violation's run context. The context type
# is provider-specific (each violation knows its own inference outputs); the
# registry stays generic over it.
OverlayProviderFactory = Callable[..., OverlayProvider]


class OverlayProviderRegistry:
    """Maps a violation kind to a provider factory (the plug-in point).

    A violation registers its factory once (module import time); a driver that
    knows which rules ran asks the registry to build a provider per kind and hands
    them to an :class:`OverlayCompositor`. Nothing here -- and nothing in the
    renderer -- is violation-specific.

    Dispatching by capture type (R6)
    ---------------------------------
    A driver holding a finished run has the rules' **capture objects** -- a helmet
    observer, a red-light capture -- and not their violation kinds; asking it to
    recover the kind means an ``isinstance`` chain in the driver, which is exactly
    the hardcoding the registry exists to remove. So a registration may also declare
    the ``source_type`` its factory consumes, and :meth:`kind_for` /
    :meth:`create_for` turn a capture straight into the right provider. The kind is
    still the registry key; the source type is an additional index onto it.
    """

    def __init__(self) -> None:
        self._factories: dict[str, OverlayProviderFactory] = {}
        self._by_source: dict[type, str] = {}

    def register(
        self,
        kind: str,
        factory: OverlayProviderFactory,
        *,
        source_type: type | None = None,
    ) -> None:
        """Bind ``factory`` to ``kind``, optionally indexed by its capture type.

        Raises:
            ValueError: ``kind`` is already registered, or ``source_type`` is already
                claimed by another kind -- an ambiguous capture would make
                :meth:`create_for` non-deterministic, so it is refused at
                registration rather than resolved by arbitrary order.
        """

        if kind in self._factories:
            raise ValueError(f"an overlay provider is already registered for {kind!r}")
        if source_type is not None and source_type in self._by_source:
            raise ValueError(
                f"{source_type.__name__} is already the overlay source for "
                f"{self._by_source[source_type]!r}; one capture type maps to one kind"
            )
        self._factories[kind] = factory
        if source_type is not None:
            self._by_source[source_type] = kind

    def create(self, kind: str, *args: object, **kwargs: object) -> OverlayProvider:
        try:
            factory = self._factories[kind]
        except KeyError:
            raise KeyError(f"no overlay provider registered for violation kind {kind!r}") from None
        return factory(*args, **kwargs)

    def kind_for(self, source: object) -> str | None:
        """The violation kind ``source`` is the capture for, or ``None`` if unknown.

        ``None`` is the answer for any object no violation claimed -- an observer a
        rule uses for its own reasoning but publishes no overlay from, for instance.
        A driver skips those; it never has to know what they are.
        """

        for source_type, kind in self._by_source.items():
            if isinstance(source, source_type):
                return kind
        return None

    def create_for(
        self, source: object, *args: object, **kwargs: object
    ) -> OverlayProvider | None:
        """Build the provider for ``source``'s kind, or ``None`` when unregistered."""

        kind = self.kind_for(source)
        if kind is None:
            return None
        return self.create(kind, source, *args, **kwargs)

    def known_kinds(self) -> frozenset[str]:
        return frozenset(self._factories)


# The process-wide registry future violations register into.
OVERLAY_PROVIDERS = OverlayProviderRegistry()
