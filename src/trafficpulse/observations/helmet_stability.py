"""Per-track temporal stabilization of helmet labels -- **runtime presentation only**.

What this is for
----------------
P4-U10 §5.1(7) measured, on real video, that a classifier's arg-max label flips
between *consecutive frames* on the majority of tracked riders: 7 of 11 tracks in the
Chiang Mai clip, 2 of 3 in Gangtok and 4 of 16 in Raxaul for ResNet-50, and more for
DeiT-Small. On one Gangtok rider the sequence reads ``helmet 0.99 / no_helmet 0.78 /
helmet 0.99``. Displayed frame by frame that is unreadable, and it *looks* like a broken
system even where the underlying per-crop accuracy is exactly what the evaluation
measured.

This module smooths that **for display and for reporting**. It is a plurality vote over
a short trailing window of a single rider track's own samples. Nothing more.

What this is emphatically **not**
---------------------------------
It is **not** a validated accuracy improvement, and no claim of one is made anywhere in
this project. It has not been evaluated on the frozen HELMET split, on the P4-U8
corrected population, or on anything else; its window length and support floor were
*chosen for legibility* and are not tuned against any test set -- doing so would leak a
held-out split into an operating point (the same reason ``ResNetHelmetConfig`` insists
``abstain_below`` be pre-committed). P4-U9's numbers describe per-crop, forced-choice
classification and continue to do so; a smoothed label is a different quantity and is
never substituted for one.

It is also **not** a violation rule and grants no exemption. The no-helmet reasoner has
its own temporal-run semantics (``rules.no_helmet``), which this does not feed, replace,
or approximate. Blocker 3 of ``docs/helmet-runtime-evaluation.md`` -- "the no-helmet
rule's run semantics have never been evaluated against per-frame instability" -- is
**untouched** by this module. Smoothing what is displayed does not evaluate what is
enforced.

Why a plurality vote and not something cleverer
-----------------------------------------------
A short vote is explainable in one sentence to a reviewer, is exactly reproducible, and
has one obvious failure mode -- it lags a genuine change by up to ``window`` samples --
which is disclosed through :attr:`StabilizedSample.settled` and
:attr:`StabilizedSample.agreement` rather than hidden. A learned temporal model would be
one more unvalidated component in a system whose open blockers are already about
unvalidated components.

Determinism
-----------
Pure. Output is a function of the input sample sequence and the config: samples are
processed in ``(frame_index, track_id)`` order, ties break by summed confidence and then
by label string, and nothing reads a clock or a random source. An unscored sample
(``confidence is None``) votes for its own label like any other -- an abstention is a
real observation about a frame, and dropping it would let one confident frame speak for
a rider the classifier mostly could not read.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

#: Label used when a window carries no vote at all. Deliberately the frozen ontology's
#: own spelling (``HelmetState.UNCERTAIN``) so a stabilized label maps through the same
#: adapter as a raw one and no new vocabulary is introduced anywhere.
UNCERTAIN_LABEL = "uncertain"


class HelmetStabilizationConfig(BaseModel):
    """Window policy for :func:`stabilize`. Frozen and strict, like the contracts.

    Both values are **presentation choices, not tuned parameters**; see the module
    docstring. They are configuration rather than constants so a deployment that wants
    raw per-frame output can ask for it (``enabled=False``) and get behaviour identical
    to never having called this module -- there is no second code path downstream.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    """``False`` passes every raw label through untouched (reporting ``settled=True``
    and ``agreement=1.0``), so smoothing can be disabled without branching anywhere."""

    window: int = Field(default=5, ge=1)
    """How many of a track's most recent samples vote. Five is roughly a sixth of a
    second at 30fps: long enough to outvote a single-frame flip, short enough that a
    rider whose state genuinely changes is re-labelled within a fifth of a second."""

    min_samples: int = Field(default=3, ge=1)
    """Below this many samples for a track the vote is reported as **unsettled**. The
    label is still emitted -- it is the best available reading -- but a caller that
    treats an unsettled reading as a firm one does so knowingly. One frame is not
    evidence of stability, and this is what stops a track's first frame from being
    presented as a settled call."""


@dataclass(frozen=True)
class HelmetSample:
    """One rider track's raw reading on one frame.

    Deliberately primitive -- a frame index, a track id, a native label string and an
    optional score -- so both producers can feed it: the overlay capture
    (``HelmetOverlayRider.helmet_label``) and the observation stream
    (``HelmetStateObservation.helmet_state``, whose ``StrEnum`` values are the *same*
    spellings). Knowing about neither is what keeps this module pure and testable.

    ``confidence`` is ``None`` for a crop the classifier never scored (a gated crop):
    "not measured", never a fabricated ``0.0``.
    """

    frame_index: int
    track_id: str
    label: str
    confidence: float | None = None


@dataclass(frozen=True)
class StabilizedSample:
    """The stabilized reading for one track on one frame.

    ``raw_label`` is preserved beside ``label`` on purpose: the smoothing must stay
    inspectable against what the model actually said on that frame, and a demo that
    could not show the difference would be hiding the instability it exists to manage.
    """

    frame_index: int
    track_id: str
    raw_label: str
    raw_confidence: float | None
    label: str
    confidence: float | None
    """Mean of the *measured* scores in the window that voted for ``label``; ``None``
    when none of those samples was scored."""

    agreement: float
    """Fraction of the window agreeing with ``label`` (1.0 when unanimous) -- the honest
    read of "how stable is this call right now"."""

    samples: int
    """How many samples the window actually held (at most ``config.window``)."""

    settled: bool
    """Whether the window held at least ``config.min_samples`` samples."""


@dataclass(frozen=True)
class HelmetTrackSummary:
    """One rider track, folded across a whole run.

    ``raw_flips`` is the P4-U10 instability measurement, reproduced here on live output
    rather than quoted from the document: how many times the *unsmoothed* label changed
    between consecutive samples of this track. ``stabilized_flips`` is the same count
    after smoothing. Both are reported side by side; neither is an accuracy claim, and
    a drop from one to the other is a legibility fact, not a correctness one.
    """

    track_id: str
    samples: int
    raw_flips: int
    stabilized_flips: int
    label: str
    confidence: float | None
    agreement: float
    settled: bool
    label_counts: tuple[tuple[str, int], ...]
    """Every raw label this track produced, with its count, sorted by label -- sorted
    rather than ranked so the shape is stable and two runs can be diffed."""

    first_frame: int
    last_frame: int


def _ordered(samples: Iterable[HelmetSample]) -> list[HelmetSample]:
    """Samples in deterministic processing order, independent of emission order."""

    return sorted(samples, key=lambda s: (s.frame_index, s.track_id))


def _vote(window: Sequence[HelmetSample]) -> tuple[str, float | None, float]:
    """The window's plurality label, its mean measured score, and its agreement.

    Ties break by **summed confidence**, then by label string. Both are deterministic;
    the second only fires when two labels tie on count *and* score, where every choice
    is arbitrary and the only thing that matters is that it is stable across runs.
    """

    if not window:  # unreachable via stabilize(); defensive, and honest if reached
        return UNCERTAIN_LABEL, None, 0.0
    counts: dict[str, int] = {}
    scores: dict[str, list[float]] = {}
    for sample in window:
        counts[sample.label] = counts.get(sample.label, 0) + 1
        if sample.confidence is not None:
            scores.setdefault(sample.label, []).append(sample.confidence)
    label = min(counts, key=lambda name: (-counts[name], -sum(scores.get(name, ())), name))
    measured = scores.get(label, ())
    confidence = sum(measured) / len(measured) if measured else None
    return label, confidence, counts[label] / len(window)


def stabilize(
    samples: Iterable[HelmetSample], *, config: HelmetStabilizationConfig | None = None
) -> tuple[StabilizedSample, ...]:
    """Smooth each track's label over a trailing window (see the module docstring).

    Returns one output per input sample, in ``(frame_index, track_id)`` order. Each
    track's window is independent: one rider's readings never influence another's, and a
    track that disappears and returns simply resumes accumulating. There is no
    cross-track state and no attempt to bridge identities -- that is the tracker's job
    and not something this module is entitled to guess at.
    """

    policy = config if config is not None else HelmetStabilizationConfig()
    ordered = _ordered(samples)
    if not policy.enabled:
        return tuple(
            StabilizedSample(
                frame_index=sample.frame_index,
                track_id=sample.track_id,
                raw_label=sample.label,
                raw_confidence=sample.confidence,
                label=sample.label,
                confidence=sample.confidence,
                agreement=1.0,
                samples=1,
                settled=True,
            )
            for sample in ordered
        )

    windows: dict[str, deque[HelmetSample]] = {}
    out: list[StabilizedSample] = []
    for sample in ordered:
        window = windows.setdefault(sample.track_id, deque(maxlen=policy.window))
        window.append(sample)
        label, confidence, agreement = _vote(tuple(window))
        out.append(
            StabilizedSample(
                frame_index=sample.frame_index,
                track_id=sample.track_id,
                raw_label=sample.label,
                raw_confidence=sample.confidence,
                label=label,
                confidence=confidence,
                agreement=agreement,
                samples=len(window),
                settled=len(window) >= policy.min_samples,
            )
        )
    return tuple(out)


def stabilized_index(
    samples: Iterable[HelmetSample], *, config: HelmetStabilizationConfig | None = None
) -> dict[tuple[int, str], StabilizedSample]:
    """:func:`stabilize`, keyed by ``(frame_index, track_id)`` for per-frame lookup.

    The shape an overlay provider wants: it draws one frame at a time and must stay a
    pure lookup rather than carrying filter state between calls.
    """

    return {
        (entry.frame_index, entry.track_id): entry
        for entry in stabilize(samples, config=config)
    }


def summarise_tracks(
    samples: Iterable[HelmetSample], *, config: HelmetStabilizationConfig | None = None
) -> tuple[HelmetTrackSummary, ...]:
    """Fold a run's samples into one summary per rider track, ordered by track id.

    The reported ``label`` is the track's **last** stabilized reading rather than a
    whole-run vote, because that is the reading the run actually ended on and the one a
    viewer sees on the last frame the rider appears in. A whole-run vote is a different
    (arguably better) statistic, but it would disagree with the video, and a summary
    that contradicts its own footage is worse than one that lags.
    """

    stabilized = stabilize(samples, config=config)
    by_track: dict[str, list[StabilizedSample]] = {}
    for entry in stabilized:
        by_track.setdefault(entry.track_id, []).append(entry)

    summaries: list[HelmetTrackSummary] = []
    for track_id in sorted(by_track):
        entries = by_track[track_id]
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.raw_label] = counts.get(entry.raw_label, 0) + 1
        raw_flips = sum(
            1
            for previous, current in zip(entries, entries[1:], strict=False)
            if previous.raw_label != current.raw_label
        )
        stabilized_flips = sum(
            1
            for previous, current in zip(entries, entries[1:], strict=False)
            if previous.label != current.label
        )
        last = entries[-1]
        summaries.append(
            HelmetTrackSummary(
                track_id=track_id,
                samples=len(entries),
                raw_flips=raw_flips,
                stabilized_flips=stabilized_flips,
                label=last.label,
                confidence=last.confidence,
                agreement=last.agreement,
                settled=last.settled,
                label_counts=tuple(sorted(counts.items())),
                first_frame=entries[0].frame_index,
                last_frame=last.frame_index,
            )
        )
    return tuple(summaries)
