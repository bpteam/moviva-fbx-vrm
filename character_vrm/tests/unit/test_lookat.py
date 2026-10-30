from character_vrm.core.config import LookAtConfig
from character_vrm.core.expressions import resolve_expressions
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.lookat import plan_lookat
from character_vrm.tests.fixtures.rigs import build_armature, generic_namer, mixamo_namer
from character_vrm.tests.fixtures.scenes import ARKIT_KEYS, make_scene


def plan(armature, keys, cfg=None):
    scene = make_scene(armature=armature, face_keys=keys)
    return plan_lookat(resolve_humanoid(armature), resolve_expressions(scene), cfg)


def test_bone_lookat_when_both_eyes_exist():
    p, warnings = plan(build_armature("A", generic_namer()), [])
    assert p.type == "bone" and not warnings


def test_expression_lookat_when_only_morphs_exist():
    p, warnings = plan(build_armature("A", mixamo_namer()), ARKIT_KEYS)  # mixamo fixture has no eye bones
    assert p.type == "expression" and not warnings


def test_bone_lookat_wins_when_both_are_available():
    p, _ = plan(build_armature("A", generic_namer()), ARKIT_KEYS)
    assert p.type == "bone"


def test_missing_lookat_is_a_warning_not_an_error():
    p, warnings = plan(build_armature("A", mixamo_namer()), [])
    assert p.type is None
    assert [w.code for w in warnings] == ["lookat-missing"]


def test_config_can_force_or_disable():
    arm = build_armature("A", generic_namer())
    assert plan(arm, ARKIT_KEYS, LookAtConfig(type="expression"))[0].type == "expression"
    assert plan(arm, ARKIT_KEYS, LookAtConfig(type="none"))[0].type is None
    p, warnings = plan(build_armature("A", mixamo_namer()), [], LookAtConfig(type="bone"))
    assert p.type is None and warnings


def test_offset_is_only_ever_user_supplied():
    arm = build_armature("A", generic_namer())
    assert plan(arm, [])[0].offset_from_head_bone is None
    assert plan(arm, [], LookAtConfig(offset_from_head_bone=(0.0, 0.1, 0.0)))[0].offset_from_head_bone == (0.0, 0.1, 0.0)
