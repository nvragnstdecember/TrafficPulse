"""Dataset ingestion inspection for the helmet training pipeline (H1).

Discovers which registered datasets are actually present on disk and reports the
result **as data**, never by raising -- a missing directory or file is a normal
state (nothing has been downloaded yet), not a fault. It downloads nothing.

The licence gate comes first, on purpose: a dataset whose licence is not
verified-permissive is reported ``LICENSE_NOT_USABLE`` and its files are not even
inspected. The pipeline never ingests data it is not cleared to train on.
"""

from __future__ import annotations

from enum import StrEnum

from .layout import DatasetLayout
from .models import DatasetEntry, DatasetRegistry, _Model


class IngestionStatus(StrEnum):
    """The outcome of inspecting one dataset on disk."""

    READY = "ready"  # licence usable, directory + all required files present
    LICENSE_NOT_USABLE = "license_not_usable"  # licence unverified / non-permissive
    MISSING_DIRECTORY = "missing_directory"  # raw dir absent (nothing downloaded)
    MISSING_FILES = "missing_files"  # dir present but required files absent


class IngestionReport(_Model):
    """The result of inspecting one dataset (a value, not an exception)."""

    dataset_id: str
    status: IngestionStatus
    missing_paths: tuple[str, ...] = ()
    message: str


def inspect_dataset(entry: DatasetEntry, layout: DatasetLayout) -> IngestionReport:
    """Inspect one dataset's on-disk presence and return a report.

    Order: licence gate -> directory -> required files. The first failing check
    determines the status; downstream checks are not run once one fails.
    """

    if not entry.license.is_usable:
        return IngestionReport(
            dataset_id=entry.dataset_id,
            status=IngestionStatus.LICENSE_NOT_USABLE,
            message=(
                f"licence {entry.license.license_id.value!r} "
                f"({entry.license.verification_status.value}) is not verified-permissive; "
                "dataset will not be ingested"
            ),
        )

    raw_dir = layout.raw_dir(entry.dataset_id)
    if not raw_dir.is_dir():
        return IngestionReport(
            dataset_id=entry.dataset_id,
            status=IngestionStatus.MISSING_DIRECTORY,
            message=f"raw directory not found: {raw_dir} (dataset not yet downloaded)",
        )

    missing = tuple(
        rel
        for rel, target in zip(
            entry.required_paths, layout.required_targets(entry), strict=True
        )
        if not target.exists()
    )
    if missing:
        return IngestionReport(
            dataset_id=entry.dataset_id,
            status=IngestionStatus.MISSING_FILES,
            missing_paths=missing,
            message=f"{len(missing)} required path(s) missing under {raw_dir}",
        )

    return IngestionReport(
        dataset_id=entry.dataset_id,
        status=IngestionStatus.READY,
        message=f"ready: {raw_dir} present with all required paths",
    )


def discover(registry: DatasetRegistry, layout: DatasetLayout) -> tuple[IngestionReport, ...]:
    """Inspect every registered dataset; return reports sorted by ``dataset_id``."""

    return tuple(
        inspect_dataset(entry, layout)
        for entry in sorted(registry.entries, key=lambda e: e.dataset_id)
    )


def ready_datasets(registry: DatasetRegistry, layout: DatasetLayout) -> tuple[str, ...]:
    """The ids of datasets that are fully present and licence-cleared, sorted."""

    return tuple(
        report.dataset_id
        for report in discover(registry, layout)
        if report.status is IngestionStatus.READY
    )
