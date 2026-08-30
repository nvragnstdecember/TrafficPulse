"""Error taxonomy for the CNN-vs-ViT helmet experiment (P4-U5).

Rooted at :class:`~helmet_rtdetr.errors.HelmetDataError` so the whole
``experiments/`` area keeps a single catchable base, while every failure mode
this unit introduces gets its own type. Nothing here is part of the
``trafficpulse`` runtime package.
"""

from __future__ import annotations

from helmet_rtdetr.errors import HelmetDataError


class CnnVitError(HelmetDataError):
    """Base for every CNN-vs-ViT experiment failure."""


# --- label parsing -------------------------------------------------------------
class MalformedRiderLabelError(CnnVitError):
    """A HELMET label string is not valid positional-encoding grammar.

    Raised rather than guessed. The label carries the entire supervision signal,
    so a string this parser does not fully understand must stop the run: silently
    dropping it would bias the class balance, and partially matching it would
    fabricate a label. See :mod:`helmet_cnn_vit.labels`.
    """


# --- corpus construction -------------------------------------------------------
class CorpusBuildError(CnnVitError):
    """A crop corpus could not be built from the annotation files."""


class MissingAnnotationError(CorpusBuildError):
    """An expected per-video annotation CSV is absent."""


class InconsistentTrackLabelError(CorpusBuildError):
    """One ``track_id`` carries more than one label within a video.

    HELMET annotates a rider configuration per tracked motorcycle, so the label is
    a property of the track and is constant across its frames (verified across all
    910 clips at acquisition time). A track that violates this is a data fault, not
    something to average over.
    """


# --- splitting -----------------------------------------------------------------
class SplitAssignmentError(CnnVitError):
    """The official split assignment could not be applied."""


class UnassignedVideoError(SplitAssignmentError):
    """A video in the corpus has no row in the authors' ``data_split.csv``.

    The official split is preserved verbatim (dataset-policy: official-split
    preservation), so a video it does not mention cannot be placed by guesswork.
    """


class UnknownSplitNameError(SplitAssignmentError):
    """``data_split.csv`` names a split outside {training, validation, test}."""


# --- metrics / statistics --------------------------------------------------------
class MetricsError(CnnVitError):
    """Base for metric-computation failures."""


class MismatchedPredictionsError(MetricsError):
    """Predictions and labels differ in length, or a paired test got unpaired inputs."""


class EmptyEvaluationError(MetricsError):
    """A metric was requested over an empty set of predictions."""
