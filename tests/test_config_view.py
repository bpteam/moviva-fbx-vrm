import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from character_vrm.core import events
from moviva_fbx_to_vrm.config_view import ConfigView, StepList, parse_preflight

DOC = {"model": {"file": {"name": "dora.glb"}, "geometry": {"ok": True, "confidence": 0.6}, "armatures": 0,
                 "skinnedMeshes": 0, "shapeKeys": 0, "textures": [{}], "problems": ["no armature"]},
       "facts": {"humanoid_geometry": True, "armature": False, "skinned": False, "required_bones_mapped": 0,
                 "face_geometry": True, "expressions_missing": True, "hair_bones": False, "spring_bones_possible": False}}


def test_parse_preflight_finds_the_line_among_log_noise():
    assert parse_preflight("INFO: x\n@preflight {\"a\": 1}\nmore") == {"a": 1}
    assert parse_preflight("nothing here") is None
    assert parse_preflight("@preflight {broken") is None


def test_config_view_replans_and_keeps_disabled_steps_off():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    view = ConfigView()
    view.load(DOC)
    on = {s.spec.id: s.enabled for s in view.steps}
    assert on["auto_rig"] and on["face_expressions"] and not on["ai_rigging"]
    view._changed("auto_rig", False)
    on = {s.spec.id: s.enabled for s in view.steps}
    assert not on["auto_rig"] and not on["face_expressions"]  # blocked, not silently kept
    assert [s for s in view.steps if s.spec.id == "face_expressions"][0].blocked_by == ["auto_rig"]


def test_step_list_follows_events():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    lst = StepList()
    lst.reset([])
    lst.feed(events.encode({"type": "step_started", "step": "import"}))
    assert "▶" in lst.rows["import"][0].text()
    lst.feed(events.encode({"type": "step_completed", "step": "import", "status": "SUCCESS", "seconds": 1.5}))
    assert "✓" in lst.rows["import"][0].text() and "1.5s" in lst.rows["import"][0].text()
    lst.feed("garbage line")  # ignored
