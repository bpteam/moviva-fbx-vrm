import pytest

from character_vrm.core.names import parse_bone_name


@pytest.mark.parametrize("name", [
    "mixamorig:LeftArm", "CC_Base_L_Upperarm", "upper_arm.L", "LeftUpperArm", "L_UpperArm",
    "J_Bip_L_UpperArm", "upperarm_l", "Bip01 L UpperArm", "mixamorig1:LeftArm", "DEF-upper_arm.L",
])
def test_left_upper_arm_variants(name):
    parsed = parse_bone_name(name)
    assert (parsed.side, parsed.part) == ("left", "upperarm"), name
    assert parsed.usable


@pytest.mark.parametrize("name,side,part", [
    ("mixamorig:RightForeArm", "right", "lowerarm"),
    ("CC_Base_R_Forearm", "right", "lowerarm"),
    ("thigh.R", "right", "upperleg"),
    ("mixamorig:LeftUpLeg", "left", "upperleg"),
    ("CC_Base_L_Calf", "left", "lowerleg"),
    ("shin.L", "left", "lowerleg"),
    ("CC_Base_L_ToeBase", "left", "toes"),
    ("eye.L", "left", "eye"),
    ("Bip001 Head", None, "head"),
    ("Hips.001", None, "hips"),
])
def test_side_and_part(name, side, part):
    parsed = parse_bone_name(name)
    assert (parsed.side, parsed.part) == (side, part)


def test_fingers_win_over_the_hand_they_belong_to():
    thumb = parse_bone_name("mixamorig:LeftHandThumb1")
    assert (thumb.side, thumb.part, thumb.index) == ("left", "thumb", 1)
    mid = parse_bone_name("CC_Base_L_Mid2")
    assert (mid.part, mid.index) == ("middle", 2)
    rig = parse_bone_name("DEF-f_index.01.L")
    assert (rig.side, rig.part, rig.index) == ("left", "index", 1)
    assert parse_bone_name("pinkie_3_R").part == "little"


def test_namespace_digits_are_not_bone_indices():
    assert parse_bone_name("mixamorig1:LeftLeg").index is None
    assert parse_bone_name("Bip01 L Thigh").index is None


@pytest.mark.parametrize("name,flag", [
    ("CC_Base_L_UpperarmTwist01", "twist"),
    ("forearm_roll.L", "twist"),
    ("HeadTop_End", "end"),
    ("LeftHandIndex4_end", "end"),
    ("HeadBand", "helper"),
    ("RightHandProp", "helper"),
    ("upperarm_ik.L", "helper"),
])
def test_twist_end_helper_bones_are_not_main_chain(name, flag):
    parsed = parse_bone_name(name)
    assert getattr(parsed, flag)
    assert not parsed.usable


def test_non_anatomical_names_have_no_part():
    for name in ("HairBack01", "LeftEyelash", "Bone.014", "cloth_skirt_02"):
        assert parse_bone_name(name).part is None
