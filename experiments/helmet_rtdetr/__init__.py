"""Helmet RT-DETR training pipeline — dev infrastructure (H1: registry + ingestion).

This package is **training infrastructure**, not part of the ``trafficpulse``
runtime: it lives under ``experiments/``, never ships in the wheel, adds no ML
dependency, and downloads nothing. H1 provides the typed dataset registry,
provenance models, on-disk layout, and ingestion inspection. Annotation
conversion, training, evaluation, and the inference backend are later units.
"""

from __future__ import annotations

from .errors import (
    DatasetNotFoundError,
    DuplicateDatasetIdError,
    HelmetDataError,
    IngestionError,
    InvalidLicenseError,
    MalformedRegistryError,
    RegistryValidationError,
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
    # errors
    "HelmetDataError",
    "RegistryValidationError",
    "DuplicateDatasetIdError",
    "InvalidLicenseError",
    "UnsupportedRegistryVersionError",
    "MalformedRegistryError",
    "DatasetNotFoundError",
    "IngestionError",
]
