"""Phase 6: hand analysis, finger classification without names, missing-finger capability and the experimental rig plan."""

import pytest

from character_vrm.core.anatomy.confidence import Thresholds
from character_vrm.core.anatomy.hands import (
    BONES_PRESENT, FUSED, GEOMETRY_SEPARABLE, classify_finger_chains, segment_fingers,
)
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.authoring.fingers import plan_all_hands, plan_finger_rig
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import ArmatureInfo, BoneInfo, FINGER_BONES
from character_vrm.tests.fixtures.hands import hand_mesh, with_finger_bones
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer
from character_vrm.tests.fixtures.scenes import make_scene


def rig_with_fingers(naming="Bone"):
    arm, names = with_finger_bones(build_armature("Armature", mixamo_namer(), fingers=False), naming=naming)
    return arm, names


def test_finger_chains_map_without_standard_names():
    arm, names = rig_with_fingers("Bone")
    scene = make_scene(armature=arm)
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    left = a.left_hand
    assert left.state == BONES_PRESENT and list(left.fingers) == ["Thumb", "Index", "Middle", "Ring", "Little"]
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
        assert left.fingers[finger].bones[0] == names[("L", finger, 1)], finger
        assert a.right_hand.fingers[finger].bones[0] == names[("R", finger, 1)], finger
    assert left.fingers["Index"].vrm_bones["leftIndexProximal"] == names[("L", "Index", 1)]
    assert left.fingers["Thumb"].vrm_bones["leftThumbMetacarpal"] == names[("L", "Thumb", 1)]
    assert left.fingers["Little"].vrm_bones["leftLittleDistal"] == names[("L", "Little", 3)]
    # all 30 finger bones end up in the humanoid map although no name said "finger"
    assert sum(1 for k in a.humanoid if k in FINGER_BONES) == 30
    assert set(FINGER_BONES) <= set(a.inferred_bones) | set(a.humanoid)


def test_both_hands_are_mirrored_and_symmetry_is_evidence():
    arm, _ = rig_with_fingers("Bone")
    scene = make_scene(armature=arm)
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    assert a.left_hand.fingers["Index"].evidence["symmetry"] == 1.0
    assert a.right_hand.fingers["Index"].evidence["symmetry"] == 1.0
    assert a.left_hand.confidence > 0.8


def test_hand_without_finger_bones_is_a_separate_capability_condition():
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    scene = make_scene(armature=arm)
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    assert a.left_hand.state != BONES_PRESENT and not a.left_hand.independent_articulation
    assert not a.left_hand.fingers and not any(k in a.humanoid for k in FINGER_BONES)


def test_separated_finger_geometry_is_detected():
    m = hand_mesh(bone="b")
    seg = segment_fingers(m.vertex_positions, (0.7, 0, 1.45), (0, 0, 1))
    assert not seg.fused and set(seg.clusters) == {"Thumb", "Index", "Middle", "Ring", "Little"}
    assert seg.confidence > 0.9


def test_mitten_geometry_is_reported_as_fused():
    m = hand_mesh(bone="b", fused=True)
    seg = segment_fingers(m.vertex_positions, (0.7, 0, 1.45), (0, 0, 1))
    assert seg.fused and not seg.clusters and "mitten" in seg.reason


def test_hand_without_a_separable_thumb_is_never_rigged():
    m = hand_mesh(bone="b", thumb=False)
    seg = segment_fingers(m.vertex_positions, (0.7, 0, 1.45), (0, 0, 1))
    assert seg.confidence <= 0.55 and "Thumb" not in seg.clusters


def scene_with_hands(**kw):
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    scene = make_scene(armature=arm)
    scene.meshes = [hand_mesh(bone="mixamorig:LeftHand", **kw),
                    hand_mesh("HandsR", wrist=(-0.7, 0, 1.45), sign=-1, bone="mixamorig:RightHand", **kw)]
    return scene, arm


def test_missing_finger_bones_with_separable_geometry_are_detected():
    scene, arm = scene_with_hands()
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    assert a.left_hand.state == GEOMETRY_SEPARABLE and a.right_hand.state == GEOMETRY_SEPARABLE


def test_finger_rig_plan_is_gated_by_flag_and_confidence():
    scene, arm = scene_with_hands()
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    off = plan_finger_rig(a.left_hand, a.joints["leftHand"], enabled=False)
    assert not off.bones and "disabled" in off.skipped.reason
    strict = plan_finger_rig(a.left_hand, a.joints["leftHand"], Thresholds(generated_fingers=0.999), True)
    assert not strict.bones and "below" in strict.skipped.reason
    ok = plan_finger_rig(a.left_hand, a.joints["leftHand"], Thresholds(generated_fingers=0.9), True)
    assert len(ok.bones) == 15 and ok.skipped is None and ok.validate() == []


def test_generated_finger_structure_and_weights():
    scene, arm = scene_with_hands()
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    plan = plan_finger_rig(a.left_hand, a.joints["leftHand"], Thresholds(generated_fingers=0.9), True)
    names = [b.name for b in plan.bones]
    assert names[:3] == ["Prosopon_left_Thumb_1", "Prosopon_left_Thumb_2", "Prosopon_left_Thumb_3"]
    by_name = {b.name: b for b in plan.bones}
    assert by_name["Prosopon_left_Index_1"].parent == "mixamorig:LeftHand"
    assert by_name["Prosopon_left_Index_2"].parent == "Prosopon_left_Index_1"
    assert plan.humanoid["leftIndexProximal"] == "Prosopon_left_Index_1"
    assert plan.humanoid["leftThumbMetacarpal"] == "Prosopon_left_Thumb_1"
    assert all(b.name.startswith("Prosopon_") for b in plan.bones)
    # weights only ever move part of the existing hand weight
    for verts in plan.transfers.values():
        for ws in verts.values():
            assert 0 <= sum(ws.values()) <= 1.0 + 1e-9
    # the fingertip end is (almost) fully weighted to the phalanx chain
    assert any(sum(ws.values()) > 0.95 for ws in plan.transfers["Hands"].values())


def test_fused_hand_is_reported_unavailable_and_not_rigged():
    scene, arm = scene_with_hands(fused=True)
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    assert a.left_hand.state == FUSED
    assert any(w.code == "fingers-fused" and "unavailable" in w.message for w in a.warnings)
    plans = plan_all_hands(a, True)
    assert all(not p.bones for p in plans)
    assert all("independent finger articulation unavailable" in p.skipped.reason for p in plans)


def test_no_finger_rig_is_planned_when_bones_exist():
    arm = build_armature("Armature", mixamo_namer())
    scene = make_scene(armature=arm)
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    assert plan_all_hands(a, True) == []
