import pytest

from character_vrm.core.humanoid import HumanoidError, resolve_humanoid
from character_vrm.core.model import ArmatureInfo, BoneInfo, REQUIRED_HUMAN_BONES
from character_vrm.tests.fixtures.rigs import (
    build_armature, cc_namer, generic_namer, mixamo_namer, opaque_namer, rigify_namer,
)


def _assert_complete(result, expected: dict[str, str]):
    assert result.ok, (result.missing_required, result.errors)
    assert result.mapped_required == len(REQUIRED_HUMAN_BONES)
    for vrm, src in expected.items():
        assert result.source_of(vrm) == src, (vrm, result.mappings.get(vrm))


def test_mixamo_profile():
    arm = build_armature("Armature", mixamo_namer())
    result = resolve_humanoid(arm)
    assert result.profile == "mixamo"
    _assert_complete(result, {
        "hips": "mixamorig:Hips", "spine": "mixamorig:Spine", "chest": "mixamorig:Spine1",
        "upperChest": "mixamorig:Spine2", "neck": "mixamorig:Neck", "head": "mixamorig:Head",
        "leftUpperArm": "mixamorig:LeftArm", "leftLowerArm": "mixamorig:LeftForeArm",
        "rightUpperLeg": "mixamorig:RightUpLeg", "leftIndexProximal": "mixamorig:LeftHandIndex1",
        "leftThumbMetacarpal": "mixamorig:LeftHandThumb1", "rightLittleDistal": "mixamorig:RightHandPinky3",
    })
    assert result.mappings["hips"].confidence == 0.99
    assert result.mappings["hips"].reason == "profile:mixamo"


def test_mixamo_variant_prefix():
    result = resolve_humanoid(build_armature("A", mixamo_namer("mixamorig1:")))
    assert result.profile == "mixamo" and result.ok


def test_character_creator_profile_and_twist_bones_untouched():
    extra = [("CC_Base_L_UpperarmTwist01", "CC_Base_L_Upperarm", (0.3, 0, 1.45)),
             ("CC_Base_L_ForearmTwist01", "CC_Base_L_Forearm", (0.55, 0, 1.45)),
             ("CC_Base_L_BreastBase", "CC_Base_Spine02", (0.08, -0.1, 1.35))]
    arm = build_armature("A", cc_namer(), extra=extra)
    result = resolve_humanoid(arm)
    assert result.profile == "character-creator"
    _assert_complete(result, {"hips": "CC_Base_Hip", "spine": "CC_Base_Waist", "chest": "CC_Base_Spine01",
                              "head": "CC_Base_Head", "leftUpperArm": "CC_Base_L_Upperarm",
                              "leftHand": "CC_Base_L_Hand", "leftLowerLeg": "CC_Base_L_Calf"})
    mapped = set(result.as_map().values())
    assert not mapped & {"CC_Base_L_UpperarmTwist01", "CC_Base_L_ForearmTwist01", "CC_Base_L_BreastBase"}


@pytest.mark.parametrize("prefix", ["DEF-", ""])
def test_rigify_profile(prefix):
    result = resolve_humanoid(build_armature("A", rigify_namer(prefix)))
    assert result.profile == "rigify"
    _assert_complete(result, {"hips": f"{prefix}spine", "spine": f"{prefix}spine.001",
                              "head": f"{prefix}spine.006", "leftUpperArm": f"{prefix}upper_arm.L",
                              "rightLowerLeg": f"{prefix}shin.R", "leftIndexIntermediate": f"{prefix}f_index.02.L"})


def test_generic_names_without_a_profile():
    result = resolve_humanoid(build_armature("A", generic_namer()))
    assert result.profile is None
    _assert_complete(result, {"hips": "Pelvis", "spine": "Spine", "chest": "Chest", "upperChest": "UpperChest",
                              "neck": "Neck", "head": "Head", "leftUpperArm": "Left_UpperArm",
                              "rightFoot": "Right_Foot", "leftLittleDistal": "Left_Little_3",
                              "leftEye": "Left_Eye", "leftToes": "Left_Toes", "rightShoulder": "Right_Shoulder"})
    assert result.mappings["leftUpperArm"].reason == "name"
    assert result.mappings["spine"].reason == "hierarchy"


def test_cc_like_rig_resolves_spine_by_hierarchy_without_profile():
    # Same topology as CC but renamed so the profile does not fire.
    rename = {"CC_Base_": "X_"}
    arm = build_armature("A", cc_namer())
    bones = [BoneInfo(b.name.replace("CC_Base_", "X_"), b.parent.replace("CC_Base_", "X_") if b.parent else None,
                      b.head, b.tail) for b in arm.bones]
    result = resolve_humanoid(ArmatureInfo("A", bones))
    assert result.profile is None
    _assert_complete(result, {"hips": "X_Hip", "spine": "X_Waist", "chest": "X_Spine01",
                              "upperChest": "X_Spine02", "neck": "X_NeckTwist01", "head": "X_Head"})
    assert result.mappings["spine"].confidence == 0.95  # capped by the head anchor (0.95)
    assert result.mappings["spine"].reason == "hierarchy"


def test_opaque_names_are_not_silently_guessed():
    arm = build_armature("A", opaque_namer())
    result = resolve_humanoid(arm)  # default threshold
    assert not result.ok
    assert result.missing_required
    assert result.below_threshold  # the guesses are reported, not applied
    assert all(m.confidence < 0.75 for m in result.below_threshold)
    assert not result.mappings


def test_opaque_names_can_be_accepted_by_lowering_the_threshold():
    arm = build_armature("A", opaque_namer())
    result = resolve_humanoid(arm, min_confidence=0.5)
    assert result.ok
    assert all(m.confidence <= 0.6 for m in result.mappings.values())


def test_explicit_config_beats_profile_and_heuristics():
    arm = build_armature("A", mixamo_namer())
    result = resolve_humanoid(arm, {"chest": "mixamorig:Spine2", "leftHand": "mixamorig:LeftForeArm"})
    assert result.mappings["chest"].reason == "config" and result.mappings["chest"].confidence == 1.0
    assert result.source_of("chest") == "mixamorig:Spine2"
    # the source bone taken by config is not reused by a lower priority
    assert result.source_of("upperChest") != "mixamorig:Spine2"


def test_config_referencing_a_missing_bone_fails():
    arm = build_armature("A", mixamo_namer())
    with pytest.raises(HumanoidError, match="does not exist"):
        resolve_humanoid(arm, {"hips": "Nope"})


def test_config_can_rescue_an_unusual_rig():
    arm = build_armature("A", opaque_namer())
    names = {b.name: b for b in arm.bones}
    # Bone.001 == hips ... derive the explicit map from a generic build with identical topology
    reference = resolve_humanoid(build_armature("R", generic_namer())).as_map()
    ordered_ref = [b.name for b in build_armature("R", generic_namer()).bones]
    ordered_opq = [b.name for b in arm.bones]
    translate = dict(zip(ordered_ref, ordered_opq))
    explicit = {vrm: translate[src] for vrm, src in reference.items()}
    result = resolve_humanoid(arm, explicit)
    assert result.ok
    assert all(m.reason == "config" for m in result.mappings.values())


def test_missing_required_bones_are_reported():
    arm = build_armature("A", generic_namer(), drop={"upleg.R", "lowleg.R", "foot.R", "toes.R"})
    result = resolve_humanoid(arm)
    assert not result.ok
    assert {"rightUpperLeg", "rightLowerLeg", "rightFoot"} <= set(result.missing_required)


def test_missing_fingers_are_not_fatal():
    result = resolve_humanoid(build_armature("A", mixamo_namer(), fingers=False))
    assert result.ok
    assert result.mapped_optional > 0
    assert "leftIndexProximal" not in result.mappings


def test_missing_head_fails():
    result = resolve_humanoid(build_armature("A", generic_namer(), drop={"head", "headtop", "eye.L", "eye.R"}))
    assert "head" in result.missing_required and not result.ok


def test_invalid_hierarchy_is_an_error():
    arm = build_armature("A", generic_namer())
    # hand parented directly under the shoulder: nearest mapped ancestor is not the lower arm
    bones = [BoneInfo(b.name, "Left_Shoulder" if b.name == "Left_Hand" else b.parent, b.head, b.tail)
             for b in arm.bones]
    result = resolve_humanoid(ArmatureInfo("A", bones))
    assert not result.ok
    assert any("leftHand" in e for e in result.errors)


def test_mirrored_rig_warns():
    arm = build_armature("A", generic_namer())
    flipped = [BoneInfo(b.name, b.parent, (-b.head[0], b.head[1], b.head[2]), (-b.tail[0], b.tail[1], b.tail[2]))
               for b in arm.bones]
    result = resolve_humanoid(ArmatureInfo("A", flipped))
    assert any(w.code == "humanoid-mirrored" for w in result.warnings)


def test_no_source_bone_is_used_twice():
    result = resolve_humanoid(build_armature("A", generic_namer()))
    sources = [m.source_bone for m in result.mappings.values()]
    assert len(sources) == len(set(sources))


def test_confidence_and_reason_present_on_every_mapping():
    result = resolve_humanoid(build_armature("A", generic_namer()))
    for m in result.mappings.values():
        assert 0.0 < m.confidence <= 1.0 and m.reason
    assert 0.9 < result.mean_confidence <= 1.0


def test_hierarchy_fills_missing_lower_arm_name():
    # LowerArm renamed to something unknowable; upper arm and hand are named.
    arm = build_armature("A", generic_namer())
    bones = [BoneInfo("Elbow_Thing" if b.name == "Left_LowerArm" else b.name,
                      "Elbow_Thing" if b.parent == "Left_LowerArm" else b.parent, b.head, b.tail)
             for b in arm.bones]
    result = resolve_humanoid(ArmatureInfo("A", bones))
    assert result.source_of("leftLowerArm") == "Elbow_Thing"
    assert result.mappings["leftLowerArm"].reason == "hierarchy"


def test_position_based_fingers_need_confirmation():
    arm = build_armature("A", generic_namer())
    # scramble finger names, keep topology
    bones = []
    for b in arm.bones:
        name = b.name
        if "_Little_" in name or "_Ring_" in name or "_Middle_" in name or "_Index_" in name or "_Thumb_" in name:
            name = f"Fg.{len(bones):03d}"
        bones.append((name, b))
    ren = {b.name: n for n, b in bones}
    scrambled = [BoneInfo(n, ren.get(b.parent, b.parent) if b.parent else None, b.head, b.tail) for n, b in bones]
    result = resolve_humanoid(ArmatureInfo("A", scrambled), min_confidence=0.5)
    fingers = [m for k, m in result.mappings.items() if "Index" in k or "Thumb" in k]
    assert fingers and all(m.reason == "anatomical" for m in fingers)
    strict = resolve_humanoid(ArmatureInfo("A", scrambled))
    assert not [k for k in strict.mappings if "Index" in k]
    assert any(w.code == "humanoid-low-confidence" for w in strict.warnings)
