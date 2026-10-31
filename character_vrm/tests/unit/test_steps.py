from character_vrm.core import events
from character_vrm.core.config import ConversionConfig
from character_vrm.core.steps import (EXPERIMENTAL, NOT_NEEDED, RECOMMENDED, UNAVAILABLE, Facts, apply_selection,
                                      cli_flags, plan)


def by_id(steps):
    return {s.spec.id: s for s in steps}


def test_unrigged_humanoid_defaults():
    s = by_id(plan(Facts(humanoid_geometry=True, face_geometry=True, expressions_missing=True)))
    assert s["auto_rig"].status == RECOMMENDED and s["auto_rig"].enabled
    assert s["face_expressions"].enabled and s["finger_rig"].status == EXPERIMENTAL and not s["finger_rig"].enabled
    assert s["ai_rigging"].status == UNAVAILABLE and not s["ai_rigging"].enabled


def test_rigged_model_does_not_reprocess():
    s = by_id(plan(Facts(humanoid_geometry=True, armature=True, skinned=True, face_geometry=True)))
    assert s["auto_rig"].status == NOT_NEEDED and not s["auto_rig"].enabled
    assert s["face_expressions"].status == NOT_NEEDED  # expressions already present


def test_not_humanoid_offers_no_rig():
    s = by_id(plan(Facts()))
    assert s["auto_rig"].status == UNAVAILABLE and not s["auto_rig"].enabled


def test_disabling_the_rig_blocks_dependents_with_a_reason():
    s = by_id(plan(Facts(humanoid_geometry=True, face_geometry=True, expressions_missing=True), {"auto_rig": False}))
    assert not s["face_expressions"].enabled and s["face_expressions"].blocked_by == ["auto_rig"]
    assert "switched off" in s["face_expressions"].why


def test_user_choice_is_never_overridden():
    steps = plan(Facts(humanoid_geometry=True, face_geometry=True, expressions_missing=True), {"face_expressions": False})
    assert not by_id(steps)["face_expressions"].enabled and by_id(steps)["auto_rig"].enabled


def test_selection_becomes_config_and_flags():
    steps = plan(Facts(humanoid_geometry=True, face_geometry=True, expressions_missing=True), {"spring_bones": False})
    cfg = apply_selection(ConversionConfig(), steps)
    assert cfg.auto_rig and cfg.auto_author.generate_face and not cfg.spring.enabled
    assert cli_flags(steps) == ["--auto-rig", "--generate-face", "--no-springs"]


def test_events_round_trip_and_failures_are_reported():
    seen = []
    assert events.decode(events.encode({"type": "step_started", "step": "x"})) == {"type": "step_started", "step": "x"}
    assert events.decode("@event {nope") is None and events.decode("hello") is None
    try:
        with events.step(seen.append, "x", "doing x"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert [e["type"] for e in seen] == ["step_started", "step_completed"] and seen[1]["status"] == "FAILED"
