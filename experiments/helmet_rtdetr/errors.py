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
