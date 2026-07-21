"""Error taxonomy for the helmet training-data registry + ingestion (H1).

Every failure raises a subclass of :class:`HelmetDataError`, so callers depend
only on this module's error types. The taxonomy deliberately separates the four
failure kinds the H1 spec calls for:

* **malformed metadata** -> :class:`MalformedRegistryError` (wraps a pydantic
  ``ValidationError`` raised while parsing untrusted JSON/dicts into the typed
  models);
* **duplicate ids** -> :class:`DuplicateDatasetIdError`;
* **invalid / unsubstantiated licence** -> :class:`InvalidLicenseError`;
* **unsupported registry version** -> :class:`UnsupportedRegistryVersionError`.

Lookup and ingestion add :class:`DatasetNotFoundError` and :class:`IngestionError`.

These are *training-pipeline* errors. They live in ``experiments/`` and are not
part of the ``trafficpulse`` runtime package.
"""

from __future__ import annotations


class HelmetDataError(Exception):
    """Base class for every helmet training-data error."""


class RegistryValidationError(HelmetDataError):
    """Base for registry-construction failures (semantic, not field-level)."""


class DuplicateDatasetIdError(RegistryValidationError):
    """Two registry entries share a ``dataset_id`` (ids must be unique keys)."""


class InvalidLicenseError(RegistryValidationError):
    """A licence claim is not substantiated.

    Raised when a licence is marked *verified* without recording *how* it was
    verified, or when an attribution-required licence (e.g. CC-BY-4.0) is verified
    without an attribution string. This encodes the project's hard-won rule: a
    licence may not be asserted usable on recollection -- the verification source
    must be recorded (ADR-001 per-artifact review).
    """


class UnsupportedRegistryVersionError(RegistryValidationError):
    """The registry declares a ``schema_version`` this code does not support."""


class MalformedRegistryError(RegistryValidationError):
    """Untrusted registry data failed to parse into the typed models.

    Wraps the originating ``pydantic.ValidationError`` as ``__cause__`` so the
    field-level detail is preserved while callers catch one stable type.
    """


class DatasetNotFoundError(HelmetDataError):
    """A requested ``dataset_id`` is not present in the registry."""


class IngestionError(HelmetDataError):
    """A dataset could not be ingested for a reason other than a clean report.

    Missing directories/files are reported as data (an ``IngestionReport``), not
    raised; this is reserved for genuine ingestion faults.
    """


# --- annotation conversion (H2) ----------------------------------------------
class ConversionError(HelmetDataError):
    """Base for annotation-conversion failures."""


class MalformedAnnotationError(ConversionError):
    """A source annotation file is structurally invalid (missing keys, bad rows)."""


class UnsupportedLabelError(ConversionError):
    """A source label is not in the adapter's label map.

    Raised rather than guessed -- the P4-U1 ``motorbike`` lesson: an unmapped
    vocabulary must fail loudly, never be silently dropped or mis-assigned.
    """


class UnknownHelmetLayoutError(ConversionError):
    """No registered HELMET layout adapter recognises the given directory.

    The HELMET dataset ships in more than one annotation layout across mirrors and
    reimplementations; the pipeline sniffs for a matching adapter and refuses to
    parse an unrecognised layout rather than mis-parse it.
    """


class DuplicateAnnotationError(ConversionError):
    """Two annotations share a content-derived ``object_id`` (same image+box+label)."""


class FrameNumberingError(ConversionError):
    """A video's objects have inconsistent frame numbering."""


class MissingImageError(ConversionError):
    """A unified object references an image file that does not exist on disk."""


# --- dataset splitting (H3) ---------------------------------------------------
class SplitError(HelmetDataError):
    """Base for dataset-splitting failures."""


class InvalidRatioError(SplitError):
    """Split ratios are out of range or do not sum to 1.0."""


class LeakageError(SplitError):
    """The same group (or image) appears in more than one split.

    The highest-priority failure the splitter guards against: frames from one
    video, or a single image, must never straddle train/val/test.
    """


class EmptySplitError(SplitError):
    """A split with a positive requested ratio ended up empty (or the corpus is)."""


class InconsistentProvenanceError(SplitError):
    """One image path is associated with more than one dataset id."""


class InvalidManifestError(SplitError):
    """A split manifest document is malformed."""


# --- training infrastructure (H4A) --------------------------------------------
class TrainingError(HelmetDataError):
    """Base for training-infrastructure failures."""


class InvalidTrainingConfigError(TrainingError):
    """A training configuration is semantically invalid (cross-field rule).

    Field-level bounds (a negative epoch count, an out-of-range fraction) surface
    as pydantic ``ValidationError`` at construction, as everywhere else in this
    package; this typed error is for rules a single field cannot express.
    """


class DuplicateExperimentError(TrainingError):
    """An experiment run directory already exists and resume is not enabled."""


class TrainerStateError(TrainingError):
    """A lifecycle method was called out of order (e.g. ``end_epoch`` before
    ``begin_epoch``, or ``begin`` twice)."""


class ResumeError(TrainingError):
    """A resume was requested but cannot proceed safely.

    Covers a stored config that does not match the current one, an unreadable
    stored config, and resuming an experiment that already finished.
    """


class CheckpointError(TrainingError):
    """A checkpoint could not be written, read, or interpreted."""


class CheckpointNotFoundError(CheckpointError):
    """The requested checkpoint (id / latest / best) does not exist."""


class InvalidMetricNameError(TrainingError):
    """A metric name does not match the sanctioned pattern."""


class InvalidMetricValueError(TrainingError):
    """A metric value is not a finite number (NaN/inf are never recorded)."""


class MetricNotFoundError(TrainingError):
    """A metric was requested that has never been recorded."""


# --- RT-DETR integration (H4B) -------------------------------------------------
class BackendUnavailableError(TrainingError):
    """torch / transformers (the optional training backend) are not installed."""


class ModelIOError(TrainingError):
    """A model checkpoint could not be loaded or saved."""


class DatasetIOError(TrainingError):
    """A training dataset could not be constructed (missing split/images, bad rows)."""


class PayloadNotFoundError(TrainingError):
    """A checkpoint's weight payload (.pt) is absent although its metadata exists."""


# --- evaluation framework (H5) --------------------------------------------------
class EvaluationError(HelmetDataError):
    """Base for evaluation-framework failures."""


class InvalidEvaluationConfigError(EvaluationError):
    """An evaluation configuration is semantically invalid (cross-field rule).

    Field-level bounds surface as pydantic ``ValidationError`` at construction,
    as everywhere else in this package; this typed error is for rules a single
    field cannot express (e.g. an unsorted IoU-threshold ladder).
    """


class InvalidPredictionError(EvaluationError):
    """A prediction is not evaluable.

    Raised for a prediction whose class is outside the detector's binary label
    space (e.g. ``motorcycle``, which is a context class, never a detection
    target), for a decoded label id the label map does not contain, and for a
    prediction that references an image the evaluation universe does not know —
    each a symptom of mismatched inputs that must fail loudly, never be
    silently dropped.
    """


class EvaluationDataError(EvaluationError):
    """Evaluation inputs could not be loaded or are inconsistent.

    Covers a missing split manifest, a malformed manifest line, and duplicate
    ground-truth objects (same content-derived ``object_id``).
    """
