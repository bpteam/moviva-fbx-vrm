"""Phases 5, 14, 15: rest-pose calibration plan, anatomy-derived colliders, and the assembled authoring plan."""

import math

import pytest

from character_vrm.core.anatomy.confidence import Thresholds
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.authoring.colliders import plan_colliders
from character_vrm.core.authoring.plan import AuthoringResult, apply_authoring_result, build_authoring_plan, thresholds_from
from character_vrm.core.authoring.rest_pose import RELAXED_ARM_ABDUCTION_DEG, analyze_rest_pose, plan_rest_pose
from character_vrm.core.config import AutoAuthorConfig
from character_vrm.core.expressions import resolve_expressions
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import ArmatureInfo, BoneInfo, SpringPlan
from character_vrm.core.pipeline import analyze as full_analyze
from character_vrm.core.config import ConversionConfig
from character_vrm.tests.fixtures.heads import head_mesh
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer
from character_vrm.tests.fixtures.scenes import make_scene
from character_vrm.tests.fixtures.transforms import rotation, transformed


def analysis_of(scene):
    arm = scene.armatures[0]
    return analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75)), arm


def posed(arm_drop: float = 0.0, head_lean: float = 0.0):
    """The fixture rig with the arms lowered by ``arm_drop`` metres at the elbow / wrist and an optional head lean."""
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    moved = {}
    for b in arm.bones:
        head, tail = list(b.head), list(b.tail)
        if arm_drop and ("Arm" in b.name or "Hand" in b.name) and "Shoulder" not in b.name:
            side = 1 if head[0] > 0 else -1
            depth = abs(head[0]) - 0.2
            head[2] -= arm_drop * depth / 0.5
            head[0] = side * (0.2 + (abs(head[0]) - 0.2) * math.sqrt(max(0.0, 1 - (arm_drop / 0.5) ** 2 * 0.0)))
        moved[b.name] = [head, tail]
    bones = []
    for b in arm.bones:
        head, tail = moved[b.name]
        if head_lean and b.name in ("mixamorig:Head", "mixamorig:HeadTop_End"):
            head = list(head)
            if b.name.endswith("HeadTop_End"):
                head[0] += head_lean
        bones.append(BoneInfo(b.name, b.parent, tuple(head), tuple(tail), b.deform))
    # recompute tails as the first child's head, like the fixture does
    kids = {}
    for b in bones:
        if b.parent:
            kids.setdefault(b.parent, []).append(b)
    fixed = [BoneInfo(b.name, b.parent, b.head, kids[b.name][0].head if b.name in kids else (b.head[0], b.head[1], b.head[2] + 0.05), b.deform)
             for b in bones]
    return ArmatureInfo("Armature", fixed)


def test_t_pose_is_detected_and_corrected_toward_the_relaxed_target():
    a, _ = analysis_of(make_scene())
    cal = analyze_rest_pose(a)
    assert cal.pose_type == "T-pose"
    assert cal.arm_abduction_deg["left"] == pytest.approx(90, abs=1) and cal.arm_abduction_deg["right"] == pytest.approx(90, abs=1)
    rot = cal.left_upper_arm
    assert rot is not None and rot.angle_deg == pytest.approx(90 - RELAXED_ARM_ABDUCTION_DEG, abs=1.5)
    assert cal.right_upper_arm.angle_deg == pytest.approx(rot.angle_deg, abs=0.5)
    # rotating about the forward axis: quaternion z-component only, mirrored between the sides
    assert abs(rot.quaternion[2]) > 0.5 and abs(rot.quaternion[0]) < 0.05
    assert rot.quaternion[2] == pytest.approx(-cal.right_upper_arm.quaternion[2], abs=1e-6)


def test_relaxed_target_matches_the_runtime_rig_calibration_numbers():
    # avatar/src/avatar/RigCalibration.ts: outward 0.38, down 0.925
    assert RELAXED_ARM_ABDUCTION_DEG == pytest.approx(math.degrees(math.atan2(0.38, 0.925)))


def test_a_pose_and_relaxed_pose_are_classified():
    for drop, expected in ((0.0, "T-pose"),):
        a, _ = analysis_of(make_scene(armature=posed(drop)))
        assert analyze_rest_pose(a).pose_type == expected
    # arms hanging along the body: elbow and wrist below the shoulder
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    bones = []
    for b in arm.bones:
        h, t = list(b.head), list(b.tail)
        if any(k in b.name for k in ("LeftForeArm", "LeftHand", "RightForeArm", "RightHand")):
            s = 1 if h[0] > 0 else -1
            depth = abs(h[0]) - 0.2
            h = [s * (0.2 + depth * 0.5), 0.0, 1.45 - depth * 0.9]
        bones.append(BoneInfo(b.name, b.parent, tuple(h), tuple(t), b.deform))
    scene = make_scene(armature=ArmatureInfo("Armature", bones))
    a, _ = analysis_of(scene)
    cal = analyze_rest_pose(a)
    assert cal.pose_type in ("A-pose", "relaxed") and cal.arm_abduction_deg["left"] < 60
    assert cal.left_upper_arm is not None or cal.arm_abduction_deg["left"] < RELAXED_ARM_ABDUCTION_DEG + 1


def test_head_tilt_is_measured_and_corrected():
    a, _ = analysis_of(make_scene(armature=posed(head_lean=0.08)))
    cal = analyze_rest_pose(a)
    assert cal.head_tilt_deg > 5 and cal.head is not None and cal.head.angle_deg > 5
    straight, _ = analysis_of(make_scene())
    assert analyze_rest_pose(straight).head is None  # a level head needs no correction


def test_rest_pose_is_invariant_to_the_source_orientation():
    scene = make_scene()
    ref = analyze_rest_pose(analysis_of(scene)[0])
    moved = analyze_rest_pose(analysis_of(transformed(scene, rotation(30, 60, -20), 3.0, (1, 2, 3)))[0])
    assert moved.pose_type == ref.pose_type
    assert moved.arm_abduction_deg["left"] == pytest.approx(ref.arm_abduction_deg["left"], abs=0.5)
    assert moved.left_upper_arm.angle_deg == pytest.approx(ref.left_upper_arm.angle_deg, abs=0.5)


def test_rest_pose_plan_is_metadata_only_and_serialisable():
    a, _ = analysis_of(make_scene())
    plan = plan_rest_pose(a)
    assert plan.mutates_scene is False and plan.validate() == []
    d = plan.to_dict()
    assert d["poseType"] == "T-pose" and d["space"].startswith("anatomical-frame")
    assert len(d["bones"]["leftUpperArm"]["rest"]) == 4 and 0 < d["confidence"] <= 1


# ---- colliders -------------------------------------------------------------------------------


def test_colliders_are_derived_from_anatomy_and_scale_with_it():
    scene = make_scene()
    scene.meshes = [head_mesh()]
    a, arm = analysis_of(scene)
    colliders, skipped = plan_colliders(a, arm)
    by = {c.name: c for c in colliders}
    assert {"head_collider", "neck_collider", "upper_chest_collider", "left_shoulder_collider",
            "right_shoulder_collider"} <= set(by)
    assert by["head_collider"].radius == pytest.approx(0.5 * a.measurements.get("head_width"), rel=0.01)
    assert by["head_collider"].radius > by["neck_collider"].radius > 0
    assert all(c.offset[0] == 0 and c.offset[2] == 0 for c in colliders)  # bone-axis offsets only
    big = transformed(scene, None, 2.0)
    a2, arm2 = analysis_of(big)
    by2 = {c.name: c for c in plan_colliders(a2, arm2)[0]}
    for name, c in by.items():
        assert by2[name].radius == pytest.approx(2 * c.radius, rel=0.03), name


def test_collider_whose_centre_is_off_the_bone_axis_is_skipped_with_a_reason():
    scene = make_scene()
    scene.meshes = [head_mesh(center=(0.0, 0.0, 1.70))]
    a, arm = analysis_of(scene)
    a.landmarks["head_center"].position = (0.2, 0.0, 1.7)  # far off the head bone axis
    colliders, skipped = plan_colliders(a, arm)
    assert "head_collider" not in {c.name for c in colliders}
    assert any("off the bone axis" in s.reason for s in skipped)


# ---- assembled plan --------------------------------------------------------------------------


def make_plan(config, scene=None):
    scene = scene or make_scene()
    scene.meshes = scene.meshes if scene is not None and scene.meshes and scene.meshes[0].vertex_positions else [head_mesh()]
    a, arm = analysis_of(scene)
    ex = resolve_expressions(scene, {}, 0.65, {m.name for m in scene.meshes})
    return a, build_authoring_plan(a, scene, arm, ex, SpringPlan(), config), scene


def test_plan_bundles_every_stage_and_stays_inspectable():
    a, plan, _ = make_plan(AutoAuthorConfig(enabled=True))
    assert plan.rest_pose is not None and plan.face is not None and plan.hair_rig is not None
    assert {k.name for k in plan.shape_keys} >= {"Prosopon_BlinkLeft", "Prosopon_aa"}
    assert plan.needs_mutation and plan.issues == []
    assert plan.armature == "Armature"


def test_conservative_defaults_generate_no_rigs_and_no_basic_expressions():
    a, plan, _ = make_plan(AutoAuthorConfig(enabled=True))
    assert not any(k.vrm_expression in ("happy", "sad", "surprised") or k.name.endswith(("browUp", "squint")) for k in plan.shape_keys)
    assert all(not f.bones for f in plan.fingers) and not plan.hair_rig.chains


def test_generate_face_adds_the_basic_expressions():
    _, plan, _ = make_plan(AutoAuthorConfig(generate_face=True))
    assert AutoAuthorConfig(generate_face=True).active
    assert {"Prosopon_happy", "Prosopon_squint"} <= {k.name for k in plan.shape_keys}


def test_confidence_thresholds_are_central_and_overridable():
    assert thresholds_from(AutoAuthorConfig()).generated_face == 0.85
    assert thresholds_from(AutoAuthorConfig(confidence={"generated_face": 0.5})).generated_face == 0.5
    _, plan, _ = make_plan(AutoAuthorConfig(enabled=True, confidence={"generated_face": 1.01}))
    assert plan.shape_keys == [] and not plan.needs_mutation


def test_result_folding_binds_only_keys_that_exist_and_never_replaces_source_mappings():
    a, plan, scene = make_plan(AutoAuthorConfig(enabled=True))

    class Vrm:
        expressions: dict = {}
        humanoid: dict = {}
        springs = SpringPlan()
    vrm = Vrm()
    vrm.expressions = {}
    applied = {k.name: k.meshes for k in plan.shape_keys if k.name != "Prosopon_ih"}
    notes = apply_authoring_result(vrm, plan, AuthoringResult(applied_shape_keys=applied))
    assert "ih" not in vrm.expressions and any("ih" in n for n in notes)
    assert {b.shape_key for b in vrm.expressions["blink"].binds} == {"Prosopon_BlinkLeft", "Prosopon_BlinkRight"}
    src = vrm.expressions["aa"]
    vrm2 = Vrm()
    vrm2.expressions = {"aa": src}
    apply_authoring_result(vrm2, plan, AuthoringResult(applied_shape_keys=applied))
    assert vrm2.expressions["aa"] is src  # existing wins


def test_disabled_springs_produce_no_spring_plan():
    scene = make_scene(hair=True)
    scene.meshes = [head_mesh()]
    a, arm = analysis_of(scene)
    ex = resolve_expressions(scene, {}, 0.65, set())
    plan = build_authoring_plan(a, scene, arm, ex, SpringPlan(), AutoAuthorConfig(enabled=True), spring_enabled=False)
    assert plan.springs is None
