"""Content-addressed, write-once scene storage (H12).

The store's whole reason to exist is that an event's ``scene_config_hash`` must
stay resolvable after the scene is edited. These tests pin that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trafficpulse.contracts.scene import ZoneType, scene_config_hash
from trafficpulse.persistence import CorruptRecordError, SceneStore
from trafficpulse.scenes import (
    DirectionDraft,
    SceneDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)

WIDTH, HEIGHT = 640, 480


def _scene(*, dy: float = -1.0, name: str = "Junction") -> object:
    draft = SceneDraft(
        scene_name=name,
        camera_id="cam-1",
        frame_width=WIDTH,
        frame_height=HEIGHT,
        zones=(
            ZoneDraft(
                zone_id="zone-lane",
                zone_type=ZoneType.LANE,
                polygon=full_frame_polygon(WIDTH, HEIGHT),
            ),
        ),
        direction=DirectionDraft(dx=0.0, dy=dy, zone_id="zone-lane"),
    )
    return build_scene(draft, scene_id="scene-1")


def test_a_stored_scene_is_addressed_by_its_own_hash(tmp_path: Path) -> None:
    store = SceneStore(tmp_path)
    scene = _scene()

    digest = store.put(scene)  # type: ignore[arg-type]

    assert digest == scene_config_hash(scene)  # type: ignore[arg-type]
    assert store.path(digest).is_file()
    assert store.get(digest) == scene


def test_storing_the_same_scene_twice_is_a_no_op(tmp_path: Path) -> None:
    # An analyst who reopens calibration and saves without changing anything must
    # not mint a second revision.
    store = SceneStore(tmp_path)
    scene = _scene()

    first = store.put(scene)  # type: ignore[arg-type]
    second = store.put(scene)  # type: ignore[arg-type]

    assert first == second
    assert store.hashes() == (first,)


def test_editing_a_scene_adds_a_revision_and_preserves_the_old_one(tmp_path: Path) -> None:
    # The property the whole design exists for: an event confirmed under the first
    # revision can still fetch the exact geometry it was reasoned against, months
    # after the analyst redrew the scene.
    store = SceneStore(tmp_path)
    original = _scene(dy=-1.0)
    edited = _scene(dy=1.0)

    old_hash = store.put(original)  # type: ignore[arg-type]
    new_hash = store.put(edited)  # type: ignore[arg-type]

    assert old_hash != new_hash
    assert store.get(old_hash) == original
    assert store.get(new_hash) == edited
    assert set(store.hashes()) == {old_hash, new_hash}


def test_an_unknown_hash_is_absent_not_an_error(tmp_path: Path) -> None:
    # An event may carry a scene hash from a run predating this store, or from a
    # file-configured scene. That is a fact to report, not a fault.
    store = SceneStore(tmp_path)

    assert store.get("0" * 64) is None
    assert store.contains("0" * 64) is False
    assert store.hashes() == ()


def test_a_corrupt_scene_file_is_a_fault_not_an_absence(tmp_path: Path) -> None:
    store = SceneStore(tmp_path)
    digest = store.put(_scene())  # type: ignore[arg-type]
    store.path(digest).write_text("{ not a scene", encoding="utf-8")

    with pytest.raises(CorruptRecordError):
        store.get(digest)


def test_enumerating_revisions_opens_no_file(tmp_path: Path) -> None:
    # A scene's filename is its hash, so listing the repository deserialises
    # nothing -- corrupting every file must not affect enumeration.
    store = SceneStore(tmp_path)
    digest = store.put(_scene())  # type: ignore[arg-type]
    store.path(digest).write_text("{ not a scene", encoding="utf-8")

    assert store.hashes() == (digest,)


def test_scenes_never_touch_the_run_tree(tmp_path: Path) -> None:
    # Scenes are per-camera and outlive any run; nesting them under runs/ would
    # misstate their lifetime and risk colliding with write-once event records.
    store = SceneStore(tmp_path)
    store.put(_scene())  # type: ignore[arg-type]

    written = {
        path.relative_to(tmp_path).parts[0] for path in tmp_path.rglob("*") if path.is_file()
    }
    assert written == {"scenes"}
