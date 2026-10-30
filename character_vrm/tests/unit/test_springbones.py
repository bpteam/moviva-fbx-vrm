from dataclasses import replace

from character_vrm.core.config import DEFAULT_SPRING_PRESET, SpringConfig
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.springbones import plan_spring_bones
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer


def build(extra, weighted_all=True, **cfg):
    arm = build_armature("A", mixamo_namer(), extra=extra)
    weighted = {b.name for b in arm.bones} if weighted_all else set()
    return arm, plan_spring_bones(arm, resolve_humanoid(arm), weighted, SpringConfig(**cfg))


def chain(prefix, n, parent="mixamorig:Head", z=1.75, x=0.0):
    out, previous = [], parent
    for i in range(1, n + 1):
        name = f"{prefix}{i:02d}"
        out.append((name, previous, (x, 0.1, z - 0.05 * i)))
        previous = name
    return out


def test_hair_chain_becomes_one_spring():
    _, plan = build(chain("HairBack", 3))
    assert [c.bones for c in plan.chains] == [["HairBack01", "HairBack02", "HairBack03"]]
    assert plan.chains[0].confidence >= 0.9 and plan.chains[0].reason.startswith("hair tokens")


def test_several_chains_are_separate_springs():
    _, plan = build(chain("HairBack", 3, x=-0.05) + chain("HairFront", 2, x=0.05) + chain("Bangs", 2, x=0.0))
    assert len(plan.chains) == 3


def test_ambiguous_names_are_never_physics():
    _, plan = build(chain("Lock", 3))  # "lock" is ambiguous: confidence 0.6
    assert plan.chains == []
    assert plan.skipped and "confidence" in plan.skipped[0]["reason"]


def test_non_hair_bones_are_ignored():
    _, plan = build(chain("Tail", 3) + chain("Cloth", 3))
    assert plan.chains == [] and plan.skipped == []


def test_single_bone_is_not_a_chain():
    _, plan = build(chain("Hair", 1))
    assert plan.chains == []
    assert "shorter" in plan.skipped[0]["reason"]


def test_control_bones_without_weights_are_skipped():
    _, plan = build(chain("HairCtrl", 3) + chain("HairIK", 3), weighted_all=False)
    assert plan.chains == []
    plan2 = build(chain("HairBack", 3), weighted_all=False)[1]
    assert plan2.chains == [] and "no skin weights" in plan2.skipped[0]["reason"]


def test_humanoid_bones_are_never_spring_bones():
    arm = build_armature("A", mixamo_namer(), extra=chain("HairBack", 3))
    hum = resolve_humanoid(arm)
    plan = plan_spring_bones(arm, hum, {b.name for b in arm.bones})
    mapped = {m.source_bone for m in hum.mappings.values()}
    assert not [b for c in plan.chains for b in c.bones if b in mapped]


def test_branching_chain_is_not_automatic_but_can_be_included():
    extra = chain("HairBack", 2) + [("HairBack03", "HairBack02", (0.05, 0.1, 1.6)), ("HairBack04", "HairBack02", (-0.05, 0.1, 1.6))]
    _, auto = build(extra)
    assert auto.chains == [] and "branching" in auto.skipped[0]["reason"]
    _, forced = build(extra, include=["HairBack01"])
    assert len(forced.chains) == 1 and forced.chains[0].reason == "config include"


def test_include_can_add_unnamed_chain():
    _, auto = build(chain("Wisp", 3))
    assert auto.chains == []  # nothing says "hair"
    _, plan = build(chain("Wisp", 3), include=["Wisp01"])
    assert plan.chains and plan.chains[0].bones == ["Wisp01", "Wisp02", "Wisp03"]


def test_ignore_removes_a_chain():
    _, plan = build(chain("HairBack", 3) + chain("HairFront", 3, x=0.05), ignore=["HairFront01"])
    assert [c.bones[0] for c in plan.chains] == ["HairBack01"]


def test_disabled():
    _, plan = build(chain("HairBack", 3), enabled=False)
    assert plan.chains == [] and plan.colliders == []


def test_preset_defaults_come_from_config_and_can_be_overridden():
    _, default = build(chain("HairBack", 3))
    preset = default.chains[0].preset
    assert (preset.stiffness, preset.drag_force, preset.gravity_power) == (
        DEFAULT_SPRING_PRESET.stiffness, DEFAULT_SPRING_PRESET.drag_force, DEFAULT_SPRING_PRESET.gravity_power)
    custom = replace(DEFAULT_SPRING_PRESET, stiffness=2.5, gravity_power=0.0)
    _, plan = build(chain("HairBack", 3), preset=custom)
    assert plan.chains[0].preset.stiffness == 2.5 and plan.chains[0].preset.gravity_power == 0.0


def test_hit_radius_scales_with_skeleton_height():
    arm = build_armature("A", mixamo_namer(), extra=chain("HairBack", 3))
    tall = build_armature("A", mixamo_namer(), extra=chain("HairBack", 3))
    from character_vrm.core.model import BoneInfo, ArmatureInfo
    big = ArmatureInfo("A", [BoneInfo(b.name, b.parent, tuple(v * 100 for v in b.head), tuple(v * 100 for v in b.tail)) for b in tall.bones])
    p1 = plan_spring_bones(arm, resolve_humanoid(arm), {b.name for b in arm.bones})
    p2 = plan_spring_bones(big, resolve_humanoid(big), {b.name for b in big.bones})
    assert p2.chains[0].preset.hit_radius > 50 * p1.chains[0].preset.hit_radius


def test_colliders_only_with_chains_and_are_conservative():
    arm, plan = build(chain("HairBack", 3))
    assert {c.name for c in plan.colliders} == {"head_collider", "chest_collider"}
    for c in plan.colliders:
        assert 0.0 < c.radius < 0.25
        assert c.offset[0] == 0.0 and c.offset[2] == 0.0
    _, none = build([])
    assert none.colliders == []
    _, off = build(chain("HairBack", 3), colliders=False)
    assert off.colliders == []
