"""Ingestion inspection for helmet training data (H1).

No downloads: these tests build a temporary raw tree by hand and assert the
ingestion reports describe it faithfully, and that the licence gate refuses data
that is not verified-permissive.
"""

from __future__ import annotations

from pathlib import Path

from helmet_rtdetr.ingestion import (
    IngestionStatus,
    discover,
    inspect_dataset,
    ready_datasets,
)
from helmet_rtdetr.layout import DatasetLayout
from helmet_rtdetr.models import (
    DatasetEntry,
    DatasetRegistry,
    DatasetSource,
    LicenseId,
    LicenseInfo,
    VerificationStatus,
)

_USABLE = LicenseInfo(
    license_id=LicenseId.CC_BY_4_0,
    verification_status=VerificationStatus.VERIFIED,
    verification_source="https://example.test/cc-by",
    attribution="Test authors, CC-BY 4.0",
)


def entry(
    dataset_id: str,
    required: tuple[str, ...] = ("a.json",),
    license_info: LicenseInfo = _USABLE,
) -> DatasetEntry:
    return DatasetEntry(
        dataset_id=dataset_id,
        name=dataset_id,
        version="1",
        modality="images",
        domain="test",
        source=DatasetSource(url="https://example.test", retrieval_method="manual"),
        license=license_info,
        required_paths=required,
    )


def _place(layout: DatasetLayout, dataset_id: str, files: tuple[str, ...]) -> None:
    raw = layout.raw_dir(dataset_id)
    raw.mkdir(parents=True, exist_ok=True)
    for rel in files:
        target = raw / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")


# --- the four statuses -------------------------------------------------------
def test_fully_present_dataset_is_ready(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    _place(layout, "ds", ("a.json",))

    report = inspect_dataset(entry("ds"), layout)
    assert report.status is IngestionStatus.READY
    assert report.missing_paths == ()


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    report = inspect_dataset(entry("ds"), DatasetLayout(tmp_path))
    assert report.status is IngestionStatus.MISSING_DIRECTORY
    assert "not yet downloaded" in report.message


def test_missing_files_are_listed(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    _place(layout, "ds", ("a.json",))  # present
    e = entry("ds", required=("a.json", "b.json", "sub/c.json"))

    report = inspect_dataset(e, layout)
    assert report.status is IngestionStatus.MISSING_FILES
    assert report.missing_paths == ("b.json", "sub/c.json")


def test_unusable_licence_blocks_ingestion_before_files(tmp_path: Path) -> None:
    """A non-usable licence is reported even if the files happen to be present."""

    layout = DatasetLayout(tmp_path)
    _place(layout, "ds", ("a.json",))  # files exist...
    e = entry("ds", license_info=LicenseInfo(license_id=LicenseId.UNKNOWN))

    report = inspect_dataset(e, layout)
    assert report.status is IngestionStatus.LICENSE_NOT_USABLE  # ...but licence gate wins


def test_licence_gate_precedes_directory_check(tmp_path: Path) -> None:
    """Even with no directory, an unusable licence is the reported reason."""

    e = entry("ds", license_info=LicenseInfo(license_id=LicenseId.CC_BY_NC_4_0))
    report = inspect_dataset(e, DatasetLayout(tmp_path))
    assert report.status is IngestionStatus.LICENSE_NOT_USABLE


# --- discover + ready_datasets -----------------------------------------------
def test_discover_reports_every_entry_sorted(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    reg = DatasetRegistry(entries=(entry("gamma"), entry("alpha"), entry("beta")))

    reports = discover(reg, layout)
    assert tuple(r.dataset_id for r in reports) == ("alpha", "beta", "gamma")


def test_ready_datasets_returns_only_ready_ids(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    _place(layout, "present", ("a.json",))
    reg = DatasetRegistry(
        entries=(
            entry("present"),  # dir + files -> ready
            entry("absent"),  # no dir -> missing
            entry("blocked", license_info=LicenseInfo(license_id=LicenseId.UNKNOWN)),
        )
    )
    assert ready_datasets(reg, layout) == ("present",)


def test_reports_are_serialisable_values(tmp_path: Path) -> None:
    """Reports are data, not exceptions -- they round-trip through JSON."""

    report = inspect_dataset(entry("ds"), DatasetLayout(tmp_path))
    assert '"status":"missing_directory"' in report.model_dump_json()


def test_default_registry_on_empty_disk_yields_no_ready_datasets(tmp_path: Path) -> None:
    """A fresh checkout (no downloads) reports cleanly: nothing ready, no crash."""

    from helmet_rtdetr.registry import default_helmet_registry

    reports = discover(default_helmet_registry(), DatasetLayout(tmp_path))
    statuses = {r.dataset_id: r.status for r in reports}
    # helmet-myanmar is licence-usable but not downloaded -> missing directory
    assert statuses["helmet-myanmar"] is IngestionStatus.MISSING_DIRECTORY
    # the other two are blocked at the licence gate
    assert statuses["roboflow-moto-helmet"] is IngestionStatus.LICENSE_NOT_USABLE
    assert statuses["aicity-track5"] is IngestionStatus.LICENSE_NOT_USABLE
    assert ready_datasets(default_helmet_registry(), DatasetLayout(tmp_path)) == ()
