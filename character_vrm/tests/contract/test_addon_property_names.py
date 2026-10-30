"""The snake_case property names our camelCase VRM names map to, checked against the add-on's own attribute
lists. The two tuples below were extracted from the v4.7.2 sources (`Vrm1HumanBonesPropertyGroup`,
`Vrm1ExpressionsPresetPropertyGroup` in editor/vrm1/property_group.py); they change only together
with an add-on upgrade."""

from character_vrm.blender.compat_v4_7_2 import snake
from character_vrm.core.model import ALL_HUMAN_BONES, VRM_EXPRESSIONS

ADDON_4_7_2_HUMAN_BONES = (
    "hips", "spine", "chest", "upper_chest", "neck", "head", "left_eye", "right_eye", "jaw",
    "left_upper_leg", "left_lower_leg", "left_foot", "left_toes", "right_upper_leg", "right_lower_leg",
    "right_foot", "right_toes", "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand", "left_thumb_metacarpal",
    "left_thumb_proximal", "left_thumb_distal", "left_index_proximal", "left_index_intermediate",
    "left_index_distal", "left_middle_proximal", "left_middle_intermediate", "left_middle_distal",
    "left_ring_proximal", "left_ring_intermediate", "left_ring_distal", "left_little_proximal",
    "left_little_intermediate", "left_little_distal", "right_thumb_metacarpal", "right_thumb_proximal",
    "right_thumb_distal", "right_index_proximal", "right_index_intermediate", "right_index_distal",
    "right_middle_proximal", "right_middle_intermediate", "right_middle_distal", "right_ring_proximal",
    "right_ring_intermediate", "right_ring_distal", "right_little_proximal", "right_little_intermediate",
    "right_little_distal",
)

ADDON_4_7_2_EXPRESSION_PRESETS = (
    "happy", "angry", "sad", "relaxed", "surprised", "neutral", "aa", "ih", "ou", "ee", "oh", "blink",
    "blink_left", "blink_right", "look_up", "look_down", "look_left", "look_right",
)


def test_every_vrm_humanoid_bone_has_the_add_on_property():
    assert {snake(b) for b in ALL_HUMAN_BONES} == set(ADDON_4_7_2_HUMAN_BONES)
    assert len(ALL_HUMAN_BONES) == len(ADDON_4_7_2_HUMAN_BONES) == 55


def test_every_expression_we_produce_has_the_add_on_property():
    assert {snake(e) for e in VRM_EXPRESSIONS} <= set(ADDON_4_7_2_EXPRESSION_PRESETS)
