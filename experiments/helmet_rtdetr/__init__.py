"""Helmet RT-DETR training pipeline — dev infrastructure (H1: registry + ingestion).

This package is **training infrastructure**, not part of the ``trafficpulse``
runtime: it lives under ``experiments/``, never ships in the wheel, adds no ML
dependency, and downloads nothing. H1 provides the typed dataset registry,
provenance models, on-disk layout, and ingestion inspection. Annotation
conversion, training, evaluation, and the inference backend are later units.
"""

from __future__ import annotations

from .convert import (
    AnnotationAdapter,
    CocoAdapter,
    HelmetFlatCsvAdapter,
    HelmetLayoutAdapter,
    HelmetTrackCsvAdapter,
    map_label,
    sniff_helmet_layout,
)
from .corpus import (
    CorpusBuilder,
    DuplicatePolicy,
    UnifiedCorpus,
    export_corpus,
    require_image_references,
    validate_image_references,
)
from .errors import (
    ConversionError,
    DatasetNotFoundError,
    DuplicateAnnotationError,
    DuplicateDatasetIdError,
    FrameNumberingError,
    HelmetDataError,
    IngestionError,
    InvalidLicenseError,
    MalformedAnnotationError,
    MalformedRegistryError,
    MissingImageError,
    RegistryValidationError,
    UnknownHelmetLayoutError,
    UnsupportedLabelError,
    UnsupportedRegistryVersionError,
)
from .ingestion import (
    IngestionReport,
    IngestionStatus,
    discover,
    inspect_dataset,
    ready_datasets,
)
from .layout import DEFAULT_DATA_ROOT, DatasetLayout
from .models import (
    ATTRIBUTION_REQUIRED,
    CURRENT_SCHEMA_VERSION,
    PERMISSIVE_LICENSES,
    SUPPORTED_SCHEMA_VERSIONS,
    ArchiveMetadata,
    CorpusMember,
    CorpusVersion,
    DatasetEntry,
    DatasetRegistry,
    DatasetSource,
    LicenseId,
    LicenseInfo,
    VerificationStatus,
)
from .registry import default_helmet_registry, load_registry
from .unified import BBox, ObjectProvenance, UnifiedClass, UnifiedObject

__all__ = [
    # models
    "ArchiveMetadata",
    "CorpusMember",
    "CorpusVersion",
    "DatasetEntry",
    "DatasetRegistry",
    "DatasetSource",
    "LicenseId",
    "LicenseInfo",
    "VerificationStatus",
    "PERMISSIVE_LICENSES",
    "ATTRIBUTION_REQUIRED",
    "CURRENT_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    # layout
    "DatasetLayout",
    "DEFAULT_DATA_ROOT",
    # registry
    "default_helmet_registry",
    "load_registry",
    # ingestion
    "IngestionReport",
    "IngestionStatus",
    "discover",
    "inspect_dataset",
    "ready_datasets",
    # unified schema (H2)
    "UnifiedObject",
    "UnifiedClass",
    "BBox",
    "ObjectProvenance",
    # converters (H2)
    "AnnotationAdapter",
    "map_label",
    "CocoAdapter",
    "HelmetLayoutAdapter",
    "HelmetTrackCsvAdapter",
    "HelmetFlatCsvAdapter",
    "sniff_helmet_layout",
    # corpus (H2)
    "CorpusBuilder",
    "UnifiedCorpus",
    "DuplicatePolicy",
    "export_corpus",
    "validate_image_references",
    "require_image_references",
    # errors
    "HelmetDataError",
    "RegistryValidationError",
    "DuplicateDatasetIdError",
    "InvalidLicenseError",
    "UnsupportedRegistryVersionError",
    "MalformedRegistryError",
    "DatasetNotFoundError",
    "IngestionError",
    "ConversionError",
    "MalformedAnnotationError",
    "UnsupportedLabelError",
    "UnknownHelmetLayoutError",
    "DuplicateAnnotationError",
    "FrameNumberingError",
    "MissingImageError",
]
