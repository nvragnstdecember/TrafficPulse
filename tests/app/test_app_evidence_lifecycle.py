"""Evidence render lifecycle, recovery, and repair (H16).

Before H16 evidence rendering had no status: a render interrupted by a restart
left a partial artifact set that was indistinguishable from a complete one. These
tests pin the property that closes that -- **interrupted rendering never appears
complete** -- and the repair path that fixes it without reprocessing the video.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _app_helpers import make_client, make_config, upload_wrong_way_video
from fastapi.testclient import TestClient

from trafficpulse.app.recovery import RunSnapshot
from trafficpulse.app.registry import EvidenceStatus

pytest.importorskip("PIL", reason="evidence rendering needs Pillow (the 'overlay' extra)")


def _process(client: TestClient, tmp_path: Path) -> str:
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})
    job_id: str = response.json()["job_id"]
    return job_id


def _snapshot_path(tmp_path: Path, job_id: str) -> Path:
    return make_config(tmp_path).runs_dir / job_id / "run.json"


def _rewrite_snapshot(tmp_path: Path, job_id: str, **updates: object) -> None:
    path = _snapshot_path(tmp_path, job_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")


# --- the happy path ------------------------------------------------------------
def test_a_completed_render_reports_ready(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)

    status = client.get(f"/api/process/{job_id}").json()
    assert status["evidence_status"] == EvidenceStatus.READY.value


def test_the_status_survives_a_restart(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)

    restarted = make_client(tmp_path)
    status = restarted.get(f"/api/process/{job_id}").json()
    assert status["evidence_status"] == EvidenceStatus.READY.value


# --- the interruption this milestone exists for --------------------------------
def test_an_interrupted_render_never_looks_complete(tmp_path: Path) -> None:
    """A restart mid-render settles to failed, never to ready or none."""

    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)
    # Exactly what a process killed mid-render leaves behind: PENDING on disk.
    _rewrite_snapshot(tmp_path, job_id, evidence_status="pending")

    restarted = make_client(tmp_path)
    status = restarted.get(f"/api/process/{job_id}").json()
    assert status["evidence_status"] == EvidenceStatus.FAILED.value


def test_pending_never_survives_recovery(tmp_path: Path) -> None:
    """Nothing resumes a pending render, so a client must never be left polling."""

    snapshot = RunSnapshot(
        job_id="job-x",
        video_id="vid-x",
        status="succeeded",
        evidence_status=EvidenceStatus.PENDING,
    )
    from trafficpulse.app.recovery import RepositoryRecovery

    assert RepositoryRecovery._evidence_status(snapshot) is EvidenceStatus.FAILED


def test_a_pre_h16_snapshot_reports_none(tmp_path: Path) -> None:
    """A run whose render was never tracked says so, rather than claiming ready."""

    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)
    path = _snapshot_path(tmp_path, job_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("evidence_status", None)
    path.write_text(json.dumps(data), encoding="utf-8")

    restarted = make_client(tmp_path)
    status = restarted.get(f"/api/process/{job_id}").json()
    assert status["evidence_status"] == EvidenceStatus.NONE.value


# --- repair ---------------------------------------------------------------------
def test_repair_re_renders_only_the_missing_events(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)

    # Simulate the interruption: drop the rendered-artifact journals, leaving the
    # write-once events and manifests untouched.
    rendered_dir = make_config(tmp_path).runs_dir / "rendered"
    journals = sorted(rendered_dir.glob("*.jsonl"))
    assert journals, "the run rendered evidence"
    for journal in journals:
        journal.unlink()
    _rewrite_snapshot(tmp_path, job_id, evidence_status="pending")

    restarted = make_client(tmp_path)
    assert (
        restarted.get(f"/api/process/{job_id}").json()["evidence_status"]
        == EvidenceStatus.FAILED.value
    )

    response = restarted.post(f"/api/process/{job_id}/evidence/repair")
    assert response.status_code == 200
    body = response.json()
    assert body["events_repaired"] == len(journals)
    assert body["artifacts_written"] > 0
    assert body["evidence_status"] == EvidenceStatus.READY.value

    # And the evidence is fetchable again.
    event_id = journals[0].stem
    assert (
        restarted.get(f"/api/evidence/{event_id}/artifacts/trigger_frame").status_code == 200
    )


def test_repair_reprocesses_nothing(tmp_path: Path) -> None:
    """No detector, no reasoner, and no new events: the run is not re-run."""

    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)
    before = client.get("/api/events").json()

    for journal in (make_config(tmp_path).runs_dir / "rendered").glob("*.jsonl"):
        journal.unlink()

    restarted = make_client(tmp_path)
    restarted.post(f"/api/process/{job_id}/evidence/repair")

    after = restarted.get("/api/events").json()
    assert after["total"] == before["total"]
    assert [item["event_id"] for item in after["items"]] == [
        item["event_id"] for item in before["items"]
    ]
    # Exactly one run still exists -- repair created no job.
    assert restarted.get("/api/metrics").json()["jobs_total"] == 1


def test_repairing_a_complete_run_is_a_no_op(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)

    body = client.post(f"/api/process/{job_id}/evidence/repair").json()
    assert body["events_repaired"] == 0
    assert body["artifacts_written"] == 0
    assert body["evidence_status"] == EvidenceStatus.READY.value


def test_repair_leaves_already_rendered_evidence_untouched(tmp_path: Path) -> None:
    """Repair can never replace evidence that was rendered correctly."""

    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)
    event_id = next(
        (make_config(tmp_path).runs_dir / "rendered").glob("*.jsonl")
    ).stem
    before = client.get(f"/api/evidence/{event_id}").json()["trigger_frame"]

    client.post(f"/api/process/{job_id}/evidence/repair")

    after = client.get(f"/api/evidence/{event_id}").json()["trigger_frame"]
    assert after == before


def test_repairing_an_unknown_job_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/process/job-nope/evidence/repair")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "job_not_found"


def test_repairing_an_unsucceeded_run_is_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    job_id = _process(client, tmp_path)
    _rewrite_snapshot(tmp_path, job_id, status="failed")

    restarted = make_client(tmp_path)
    response = restarted.post(f"/api/process/{job_id}/evidence/repair")
    assert response.status_code == 400
