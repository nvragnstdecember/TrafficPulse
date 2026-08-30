"""HELMET positional-encoding label grammar (P4-U5).

HELMET annotates one bounding box per *tracked motorcycle*, labelled with a
single string that encodes every rider position on that motorcycle and whether
each one wears a helmet -- for example ``DHelmet``, ``DNoHelmetP1NoHelmet``, or
``DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3Helmet``.

Verified against the real corpus
--------------------------------
Unlike the H2 converters in :mod:`helmet_rtdetr.convert.helmet`, whose docstring
correctly flags their layout as assumed, everything here was confirmed against
the downloaded dataset on 2026-08-29 (``annotation.zip``, sha256
``6b7f2245dc09e6da3bbe1699c0ce87e4777f32080570b967f4f93c894af9e55c``): 910
per-video CSVs with the header ``track_id,frame_id,x,y,w,h,label``, 283,377 rows,
10,006 tracks, **exactly 36 distinct label strings**, all of which parse under the
grammar below, and **zero** tracks whose label changes mid-track.

Grammar
-------
``label := (position helmet_state)+`` where ``position`` is one of ``D`` (driver),
``P0``, ``P1``, ``P2``, ``P3`` (passenger slots) and ``helmet_state`` is
``Helmet`` or ``NoHelmet``. A valid label names the driver exactly once, names it
first, and never repeats a position.

Why the parser is strict
------------------------
The label carries the entire supervision signal. :func:`parse_label` therefore
*fully* matches -- a trailing or interior unrecognised fragment raises
:class:`~helmet_cnn_vit.errors.MalformedRiderLabelError` rather than yielding a
partial configuration. Dropping such a row would silently bias the class balance;
partially matching it would fabricate a label. This is the P4-U1 ``motorbike``
lesson applied to a richer vocabulary.

Note the ordering subtlety: ``NoHelmet`` must be tried before ``Helmet``, since
``Helmet`` is a suffix of ``NoHelmet``. The regex alternation below is ordered
accordingly, and :func:`parse_label` re-serialises what it parsed and compares it
to the input, so an ordering mistake could not pass silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .errors import MalformedRiderLabelError


class RiderPosition(StrEnum):
    """A seating position on the motorcycle, in canonical front-to-back order."""

    DRIVER = "D"
    P0 = "P0"  # a child/front position, ahead of the driver
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class HelmetState(StrEnum):
    """The binary helmet state of one rider.

    Deliberately *not* imported from ``trafficpulse.contracts``: HELMET has no
    ``uncertain`` state, and borrowing the runtime's three-valued enum here would
    imply a label this dataset does not carry.
    """

    HELMET = "helmet"
    NO_HELMET = "no_helmet"


#: Canonical position order, used for the parsed tuple and for re-serialisation.
POSITION_ORDER: tuple[RiderPosition, ...] = (
    RiderPosition.DRIVER,
    RiderPosition.P0,
    RiderPosition.P1,
    RiderPosition.P2,
    RiderPosition.P3,
)

_STATE_TOKENS: dict[str, HelmetState] = {
    "NoHelmet": HelmetState.NO_HELMET,
    "Helmet": HelmetState.HELMET,
}
_STATE_SPELLING: dict[HelmetState, str] = {v: k for k, v in _STATE_TOKENS.items()}

# NoHelmet first: "Helmet" is a suffix of "NoHelmet" and would otherwise shadow it.
_TOKEN = re.compile(r"(D|P0|P1|P2|P3)(NoHelmet|Helmet)")


@dataclass(frozen=True, slots=True)
class RiderConfiguration:
    """The parsed rider configuration of one annotated motorcycle.

    A plain frozen dataclass rather than the package's pydantic ``_Model``: this is
    constructed once per annotation row (283,377 of them) and carries no untrusted
    field values of its own -- :func:`parse_label` has already validated the only
    input there is.
    """

    riders: tuple[tuple[RiderPosition, HelmetState], ...]

    @property
    def driver_state(self) -> HelmetState:
        """The driver's helmet state -- the experiment's classification target."""

        for position, state in self.riders:
            if position is RiderPosition.DRIVER:
                return state
        raise MalformedRiderLabelError(  # pragma: no cover - parse_label guarantees D
            "configuration has no driver"
        )

    @property
    def rider_count(self) -> int:
        """Total riders on the motorcycle, driver included."""

        return len(self.riders)

    @property
    def any_no_helmet(self) -> bool:
        """Whether *any* rider is unhelmeted -- the runtime rule's violation condition."""

        return any(state is HelmetState.NO_HELMET for _, state in self.riders)

    def to_label(self) -> str:
        """Re-serialise to the source vocabulary (round-trips :func:`parse_label`)."""

        return "".join(
            f"{position.value}{_STATE_SPELLING[state]}" for position, state in self.riders
        )


def parse_label(raw: str) -> RiderConfiguration:
    """Parse a HELMET label string into a :class:`RiderConfiguration`.

    Raises :class:`~helmet_cnn_vit.errors.MalformedRiderLabelError` unless the
    whole string is grammatical, names the driver exactly once and first, and
    repeats no position.
    """

    if not raw:
        raise MalformedRiderLabelError("empty label")

    riders = tuple(
        (RiderPosition(position), _STATE_TOKENS[state]) for position, state in _TOKEN.findall(raw)
    )
    config = RiderConfiguration(riders=riders)

    # Full-consumption check: re-serialising must reproduce the input exactly. This
    # catches trailing junk, interior junk, and any tokeniser-ordering mistake.
    if config.to_label() != raw:
        raise MalformedRiderLabelError(
            f"label {raw!r} is not valid positional-encoding grammar "
            f"(parsed as {config.to_label()!r})"
        )

    positions = [position for position, _ in riders]
    if len(set(positions)) != len(positions):
        raise MalformedRiderLabelError(f"label {raw!r} repeats a rider position")
    if positions[0] is not RiderPosition.DRIVER:
        raise MalformedRiderLabelError(f"label {raw!r} does not name the driver first")
    return config


#: The 36 label strings observed across all 910 clips on 2026-08-29, with their
#: annotation-row counts. Recorded so a corpus build can report any label outside
#: the verified vocabulary instead of quietly absorbing it, and so the class
#: balance quoted in the pre-registration is reproducible from the repository
#: alone. Sums to the 283,377 rows in ``annotation.zip``.
VERIFIED_LABEL_COUNTS: dict[str, int] = {
    "DHelmet": 113620,
    "DHelmetP0Helmet": 274,
    "DHelmetP0HelmetP1Helmet": 563,
    "DHelmetP0HelmetP1HelmetP2Helmet": 18,
    "DHelmetP0HelmetP1NoHelmetP2Helmet": 100,
    "DHelmetP0HelmetP1NoHelmetP2NoHelmet": 22,
    "DHelmetP0NoHelmet": 868,
    "DHelmetP0NoHelmetP1Helmet": 1047,
    "DHelmetP0NoHelmetP1HelmetP2Helmet": 46,
    "DHelmetP0NoHelmetP1NoHelmet": 637,
    "DHelmetP0NoHelmetP1NoHelmetP2Helmet": 468,
    "DHelmetP0NoHelmetP1NoHelmetP2NoHelmet": 180,
    "DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3Helmet": 49,
    "DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 29,
    "DHelmetP1Helmet": 57142,
    "DHelmetP1HelmetP2Helmet": 516,
    "DHelmetP1HelmetP2NoHelmet": 120,
    "DHelmetP1NoHelmet": 12202,
    "DHelmetP1NoHelmetP2Helmet": 2862,
    "DHelmetP1NoHelmetP2NoHelmet": 2170,
    "DHelmetP1NoHelmetP2NoHelmetP3Helmet": 59,
    "DHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 43,
    "DNoHelmet": 49815,
    "DNoHelmetP0HelmetP1NoHelmet": 28,
    "DNoHelmetP0NoHelmet": 837,
    "DNoHelmetP0NoHelmetP1Helmet": 23,
    "DNoHelmetP0NoHelmetP1NoHelmet": 1535,
    "DNoHelmetP0NoHelmetP1NoHelmetP2Helmet": 14,
    "DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet": 865,
    "DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 10,
    "DNoHelmetP1Helmet": 3174,
    "DNoHelmetP1HelmetP2Helmet": 77,
    "DNoHelmetP1NoHelmet": 27409,
    "DNoHelmetP1NoHelmetP2Helmet": 196,
    "DNoHelmetP1NoHelmetP2NoHelmet": 6080,
    "DNoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 279,
}
