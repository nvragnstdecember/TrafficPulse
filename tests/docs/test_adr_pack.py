"""Machine-verifiable invariants for the U6 ADR pack and architecture entry point.

Validates the actual artifacts required by the Phase 0-F U6 acceptance criteria:
the ADR files exist with required metadata/sections and correct statuses, the
architecture document confirms the canonical reference and links every ADR, and
only sanctioned runtime packages have appeared under ``src/trafficpulse`` (the
U2 ``contracts`` layer, the detector-independent Phase 1 layers P1-U1..U5, and the
detector-integration foundation P1-U6 that ADR-001 unblocked; no unsanctioned
Phase 1 package such as tracking, events, or evidence).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "adr"
ARCHITECTURE = REPO_ROOT / "docs" / "architecture.md"
ARCHITECTURE_REVIEW = REPO_ROOT / "docs" / "architecture-review.md"
PHASE0_PLAN = REPO_ROOT / "docs" / "phase-0-plan.md"
PHASE1_PLAN = REPO_ROOT / "docs" / "phase-1-plan.md"

ADR_IDS = ("ADR-001", "ADR-002", "ADR-003", "ADR-004")
ACCEPTED_ADRS = ("ADR-002", "ADR-003")
REQUIRED_SECTIONS = ("## Context", "## Decision", "## Consequences")

# Retrospective Phase 1 completed-unit -> commit-hash mapping (Git history through
# HEAD 8b6d51f). The Phase 1 plan must record each unit against its commit; this
# guards the mapping from silently drifting away from Git history.
PHASE1_COMPLETED_UNITS = {
    "P1-U1": "0dfc774",
    "P1-U2": "0ff1bc0",
    "P1-U3": "7ad37c6",
    "P1-U4": "4651ffb",
    "P1-U5": "dd70edd",
    "P1-U6": "07f8baa",
    "P1-U7": "8b6d51f",
}


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


def test_adr001_accepted_permissive_posture() -> None:
    # ADR-001 was resolved at its deadline (before the first detector-integration
    # unit of Phase 1): Accepted, permissive-only, RT-DETR primary, detector behind
    # the frozen U2 ``Detection`` contract. This test enforces the *resolved*
    # current status; the prior open-state contract lived here before resolution.
    text = _read(ADR_DIR / "ADR-001.md")
    lower = text.lower()
    # Status is now Accepted (no lingering open/unresolved current status).
    assert _status(text) == "Accepted", f"ADR-001 not Accepted: {_status(text)!r}"
    # Permissive-only posture with RT-DETR as the primary integration direction.
    assert "permissive-only" in lower
    assert "rt-detr" in lower
    # Detector stays behind the frozen U2 Detection contract (bounded seam).
    assert "u2" in lower and "detection" in lower and "contract" in lower
    # Required ADR record elements preserved.
    assert "Decision owner" in text
    assert "## Consequences" in text
    # The open->resolved history and its deadline are preserved, not erased.
    assert "before the first detector-integration unit of Phase 1" in text
    # The consequence that matters for Phase 1: the integration gate is lifted.
    assert "lifted" in lower


def test_architecture_confirms_canonical_reference_and_links_adrs() -> None:
    text = _read(ARCHITECTURE)
    assert ARCHITECTURE_REVIEW.is_file()
    assert "architecture-review.md" in text
    assert "canonical" in text.lower()
    for adr_id in ADR_IDS:
        link = f"adr/{adr_id}.md"
        assert link in text, f"architecture.md does not link {link}"
        assert (REPO_ROOT / "docs" / "adr" / f"{adr_id}.md").is_file()


def test_only_sanctioned_runtime_packages() -> None:
    # Permitted so far: the U2 ``contracts`` layer plus the detector-independent
    # Phase 1 layers ``geometry`` (P1-U1), ``synth`` (P1-U2), ``rules`` (P1-U3),
    # ``observations`` (P1-U4), and ``ingestion`` (P1-U5), which ADR-001/003 and
    # architecture-review §25 sanction as non-blocked Phase 1 work; the
    # ``detector`` integration foundation (P1-U6), unblocked by ADR-001's Accepted
    # permissive-only posture and kept behind the frozen U2 ``Detection`` contract;
    # and the ``tracking`` integration foundation (P1-U8), the tracking analogue,
    # kept behind the frozen U2 ``TrackState`` contract and carrying no tracker
    # dependency. Any other package (events, evidence, ...) would be premature scope.
    allowed = {
        "contracts",
        "geometry",
        "synth",
        "rules",
        "observations",
        "ingestion",
        "detector",
        "tracking",
    }
    package = REPO_ROOT / "src" / "trafficpulse"
    subdirs = {p.name for p in package.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert subdirs <= allowed, f"unexpected package dirs: {subdirs - allowed}"


# --- Phase 1 plan governance -------------------------------------------------
# The authoritative Phase 1 unit plan must exist, must declare its authority and
# the Phase 0/Phase 1 namespace separation, must record the completed P1-U1..P1-U7
# units against Git history, and must name the next unit — so the plan cannot
# silently disappear or drift.
def test_phase1_plan_exists() -> None:
    assert PHASE1_PLAN.is_file(), "missing docs/phase-1-plan.md (authoritative Phase 1 plan)"


def test_phase1_plan_declares_authority_and_namespaces() -> None:
    text = _read(PHASE1_PLAN)
    lower = text.lower()
    # It is the authoritative Phase 1 plan.
    assert "authoritative phase 1 unit plan" in lower
    # Phase 0-F stays governed by its own document (not superseded).
    assert "Phase 0-F remains governed by" in text
    # The two identifier namespaces (U# vs P1-U#) are kept distinct.
    assert "separate identifier namespaces" in lower


def test_phase1_plan_records_completed_units_with_commits() -> None:
    text = _read(PHASE1_PLAN)
    for unit, commit in PHASE1_COMPLETED_UNITS.items():
        assert unit in text, f"Phase 1 plan does not record {unit}"
        assert commit in text, f"Phase 1 plan does not cite {unit} commit {commit}"


def test_phase1_plan_defines_next_unit() -> None:
    text = _read(PHASE1_PLAN)
    # The next unit is unambiguous: P1-U8 tracker integration foundation.
    assert "P1-U8" in text
    assert "Tracker integration foundation" in text


def test_phase0_plan_remains_phase0_scoped() -> None:
    # Phase 0-F plan is left historically intact and still disclaims Phase 1 detail.
    assert PHASE0_PLAN.is_file(), "docs/phase-0-plan.md must remain present"
    assert "It does not plan Phase 1 in detail" in _read(PHASE0_PLAN)


def test_architecture_links_phase1_plan() -> None:
    # The entry-point doc must point to the Phase 1 plan for discoverability.
    assert "phase-1-plan.md" in _read(ARCHITECTURE)
