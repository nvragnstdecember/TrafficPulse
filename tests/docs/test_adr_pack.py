"""Machine-verifiable invariants for the U6 ADR pack and architecture entry point.

Validates the actual artifacts required by the Phase 0-F U6 acceptance criteria:
the ADR files exist with required metadata/sections and correct statuses, the
architecture document confirms the canonical reference and links every ADR, and
no Phase 1 runtime package has appeared under ``src/trafficpulse``.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "adr"
ARCHITECTURE = REPO_ROOT / "docs" / "architecture.md"
ARCHITECTURE_REVIEW = REPO_ROOT / "docs" / "architecture-review.md"

ADR_IDS = ("ADR-001", "ADR-002", "ADR-003", "ADR-004")
ACCEPTED_ADRS = ("ADR-002", "ADR-003")
REQUIRED_SECTIONS = ("## Context", "## Decision", "## Consequences")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _status(text: str) -> str:
    match = re.search(r"^- \*\*Status:\*\*\s*(.+)$", text, re.MULTILINE)
    assert match is not None, "missing '- **Status:**' metadata line"
    return match.group(1).strip()


def test_all_adrs_exist() -> None:
    for adr_id in ADR_IDS:
        assert (ADR_DIR / f"{adr_id}.md").is_file(), f"missing {adr_id}.md"


def test_each_adr_has_required_metadata_and_sections() -> None:
    for adr_id in ADR_IDS:
        text = _read(ADR_DIR / f"{adr_id}.md")
        assert "- **Status:**" in text, f"{adr_id} missing Status"
        assert "- **Date:**" in text, f"{adr_id} missing Date"
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{adr_id} missing '{section}'"


def test_storage_and_offline_adrs_accepted() -> None:
    for adr_id in ACCEPTED_ADRS:
        assert _status(_read(ADR_DIR / f"{adr_id}.md")) == "Accepted", f"{adr_id} not Accepted"


def test_adr004_at_least_proposed() -> None:
    assert _status(_read(ADR_DIR / "ADR-004.md")) in {"Proposed", "Accepted"}


def test_adr001_documented_open_with_required_elements() -> None:
    text = _read(ADR_DIR / "ADR-001.md")
    status = _status(text).lower()
    assert "proposed" in status or "documented-open" in status or "unresolved" in status
    # (a) unresolved status is explicit
    assert "documented-open" in text.lower() or "unresolved" in text.lower()
    # (b) named decision owner
    assert "Decision owner" in text
    # (c) decision deadline set at the required point
    assert "before the first detector-integration unit of Phase 1" in text
    # (d) consequences recorded
    assert "## Consequences" in text
    assert "blocked" in text.lower()


def test_architecture_confirms_canonical_reference_and_links_adrs() -> None:
    text = _read(ARCHITECTURE)
    assert ARCHITECTURE_REVIEW.is_file()
    assert "architecture-review.md" in text
    assert "canonical" in text.lower()
    for adr_id in ADR_IDS:
        link = f"adr/{adr_id}.md"
        assert link in text, f"architecture.md does not link {link}"
        assert (REPO_ROOT / "docs" / "adr" / f"{adr_id}.md").is_file()


def test_no_phase1_runtime_packages() -> None:
    package = REPO_ROOT / "src" / "trafficpulse"
    subdirs = {p.name for p in package.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert subdirs == {"contracts"}, f"unexpected package dirs: {subdirs}"
