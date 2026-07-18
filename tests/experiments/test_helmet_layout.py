"""On-disk layout for helmet training data (H1)."""

from __future__ import annotations

from pathlib import Path

from helmet_rtdetr.layout import DEFAULT_DATA_ROOT, DatasetLayout
from helmet_rtdetr.models import (
    DatasetEntry,
    DatasetSource,
    LicenseId,
    LicenseInfo,
)


def _entry(required: tuple[str, ...]) -> DatasetEntry:
    return DatasetEntry(
        dataset_id="ds",
        name="DS",
        version="1",
        modality="images",
        domain="test",
        source=DatasetSource(url="https://example.test", retrieval_method="manual"),
        license=LicenseInfo(license_id=LicenseId.MIT),
        required_paths=required,
    )


def test_construction_touches_no_disk(tmp_path: Path) -> None:
    root = tmp_path / "data"
    DatasetLayout(root)  # must not create anything
    assert not root.exists()


def test_paths_are_deterministic(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    assert layout.raw == tmp_path / "raw"
    assert layout.interim == tmp_path / "interim"
    assert layout.processed == tmp_path / "processed"
    assert layout.splits == tmp_path / "splits"
    assert layout.raw_dir("helmet-myanmar") == tmp_path / "raw" / "helmet-myanmar"


def test_create_is_idempotent(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    layout.create()
    layout.create()  # must not raise
    assert layout.exists()
    for name in ("raw", "interim", "processed", "splits"):
        assert (tmp_path / name).is_dir()


def test_exists_is_false_before_create(tmp_path: Path) -> None:
    assert DatasetLayout(tmp_path).exists() is False


def test_required_targets_resolve_under_raw_dir(tmp_path: Path) -> None:
    layout = DatasetLayout(tmp_path)
    targets = layout.required_targets(_entry(("annotation", "video/clip.mp4")))
    assert targets == (
        tmp_path / "raw" / "ds" / "annotation",
        tmp_path / "raw" / "ds" / "video/clip.mp4",
    )


def test_no_required_paths_yields_no_targets(tmp_path: Path) -> None:
    assert DatasetLayout(tmp_path).required_targets(_entry(())) == ()


def test_default_root_is_the_gitignored_data_dir() -> None:
    assert DEFAULT_DATA_ROOT.name == "data"
    assert DatasetLayout().root == DEFAULT_DATA_ROOT
