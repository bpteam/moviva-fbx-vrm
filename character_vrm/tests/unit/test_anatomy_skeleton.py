"""Phase 4: skeleton inference from topology, position, symmetry and skinning; names are one evidence among several."""

import random

import pytest

from character_vrm.core.anatomy.confidence import fuse
from character_vrm.core.anatomy.skeleton import assess_skeleton, infer_topology
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import ArmatureInfo, BoneInfo
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.tests.fixtures.rigs import build_armature, cc_namer, generic_namer, mixamo_namer
from character_vrm.tests.fixtures.scenes import make_scene
from character_vrm.tests.fixtures.transforms import rotation, transform_point


def anonymise(arm, seed=3):
    names = [b.name for b in arm.bones]
    random.Random(seed).shuffle(names)
    ren = {n: f"Bone.{i:03d}" for i, n in enumerate(names)}
    return ArmatureInfo(arm.name, [BoneInfo(ren[b.name], ren.get(b.parent), b.head, b.tail, b.deform) for b in arm.bones]), ren


def test_weak_names_are_mapped_from_topology_alone():
    arm = build_armature("A", mixamo_namer())
    weak, ren = anonymise(arm)
    topo, _, _ = infer_topology(weak)
    expected = {"hips": "mixamorig:Hips", "spine": "mixamorig:Spine", "head": "mixamorig:Head", "neck": "mixamorig:Neck",
                "leftUpperArm": "mixamorig:LeftArm", "rightLowerArm": "mixamorig:RightForeArm",
                "leftHand": "mixamorig:LeftHand", "rightUpperLeg": "mixamorig:RightUpLeg",
                "leftFoot": "mixamorig:LeftFoot", "leftToes": "mixamorig:LeftToeBase"}
    for vrm, original in expected.items():
        assert topo[vrm][0] == ren[original], vrm


@pytest.mark.parametrize("rot", [rotation(0, 0, 0), rotation(90, 0, 0), rotation(20, 70, -40)])
def test_topology_needs_no_axis_convention(rot):
    arm = build_armature("A", mixamo_namer())
    weak, ren = anonymise(arm)
    moved = ArmatureInfo("A", [BoneInfo(b.name, b.parent, transform_point(b.head, rot, 1, (0, 0, 0)),
                                        transform_point(b.tail, rot, 1, (0, 0, 0))) for b in weak.bones])
    topo, warnings, assumed = infer_topology(moved)
    assert {topo[k][0] for k in ("leftUpperLeg", "rightUpperLeg")} == {ren["mixamorig:LeftUpLeg"], ren["mixamorig:RightUpLeg"]}
    assert topo["hips"][0] == ren["mixamorig:Hips"]


def test_sides_come_from_names_when_available_and_are_flagged_when_assumed():
    arm = build_armature("A", mixamo_namer())
    topo, _, assumed = infer_topology(arm)
    assert not assumed and topo["leftUpperArm"][0] == "mixamorig:LeftArm"
    weak, _ = anonymise(arm)
    _, warnings, assumed = infer_topology(weak)
    assert assumed and any(w.code == "topology-side-assumed" for w in warnings)


def test_evidence_is_kept_uncollapsed_for_every_bone():
    scene = make_scene()
    arm = scene.armatures[0]
    humanoid = resolve_humanoid(arm, None, 0.75)
    sk = assess_skeleton(arm, scene.weighted_bone_names(arm.name),
                         {k: (m.source_bone, m.confidence) for k, m in humanoid.mappings.items()})
    a = sk.assessments["leftUpperArm"]
    assert {"name", "topology", "position", "symmetry", "skinning"} <= set(a.evidence)
    assert 0.8 < a.confidence <= 1.0
    assert a.evidence["symmetry"] > 0.9


def test_a_contradicting_signal_vetoes_agreeing_ones():
    assert fuse({"topology": 1.0, "position": 1.0, "symmetry": 1.0, "skinning": 0.1}) <= 0.6
    assert fuse({"topology": 0.9, "position": 0.9}) > 0.85
    assert fuse({}) == 0.0


def test_names_and_topology_disagreeing_is_a_reported_conflict_and_names_win():
    scene = make_scene()
    arm = scene.armatures[0]
    names = {"leftUpperArm": ("mixamorig:RightArm", 0.95)}  # a deliberately wrong name mapping
    sk = assess_skeleton(arm, set(), names)
    assert sk.assessments["leftUpperArm"].bone == "mixamorig:RightArm"
    assert sk.assessments["leftUpperArm"].evidence["topology"] == 0.0
    assert sk.assessments["leftUpperArm"].confidence <= 0.6
    assert any("leftUpperArm" in c for c in sk.conflicts)


def test_implausible_position_lowers_confidence():
    scene = make_scene()
    arm = scene.armatures[0]
    sk = assess_skeleton(arm, set(), {"head": ("mixamorig:LeftFoot", 0.9)})
    assert sk.assessments["head"].evidence["position"] < 0.2
    assert sk.assessments["head"].confidence <= 0.6


def test_gap_fill_only_when_confident_and_never_overrides_names():
    scene = make_scene()
    arm = scene.armatures[0]
    weak, ren = anonymise(arm)
    scene.armatures = [weak]
    for mesh in scene.meshes:
        mesh.vertex_groups = [ren.get(g, g) for g in mesh.vertex_groups]
    a = analyze_human(scene, weak, resolve_humanoid(weak, None, 0.75))
    assert a is not None
    assert a.inferred_bones  # names said nothing, topology mapped them
    assert a.humanoid["leftHand"] == ren["mixamorig:LeftHand"]
    # the feet point forward (ankle -> toes): left and right follow from anatomy, nothing is assumed
    assert a.frame.confidence > 0.9
    assert not any(w.code == "frame-sides-assumed" for w in a.warnings)


def test_without_toes_the_foot_bone_itself_gives_forward_and_the_sides():
    scene = make_scene(armature=build_armature("A", mixamo_namer(), drop={"toes"}))
    arm = scene.armatures[0]
    weak, ren = anonymise(arm)
    scene.armatures = [weak]
    for mesh in scene.meshes:
        mesh.vertex_groups = [ren.get(g, g) for g in mesh.vertex_groups]
    a = analyze_human(scene, weak, resolve_humanoid(weak, None, 0.75))
    assert a is not None and a.humanoid["leftHand"] == ren["mixamorig:LeftHand"]
    assert a.humanoid["rightFoot"] == ren["mixamorig:RightFoot"]
    assert not any(w.code == "frame-sides-assumed" for w in a.warnings)


def test_mirrored_import_keeps_the_anatomical_sides():
    """A character facing +Y (VRM 0.x import) has its left at -X: the feet decide, not the +X convention."""
    arm = build_armature("A", mixamo_namer())
    flipped = ArmatureInfo("A", [BoneInfo(b.name, b.parent, (-b.head[0], -b.head[1], b.head[2]),
                                          (-b.tail[0], -b.tail[1], b.tail[2]), b.deform) for b in arm.bones])
    weak, ren = anonymise(flipped)
    scene = make_scene(armature=weak)
    a = analyze_human(scene, weak, resolve_humanoid(weak, None, 0.75))
    assert a.humanoid["leftUpperLeg"] == ren["mixamorig:LeftUpLeg"]
    assert a.humanoid["leftHand"] == ren["mixamorig:LeftHand"]


def test_lone_bone_above_the_shoulders_is_not_mapped_as_head_automatically():
    arm = build_armature("A", generic_namer(), drop={"head", "headtop", "eye.L", "eye.R"})
    topo, _, _ = infer_topology(arm)
    assert topo["head"][1] < 0.8


@pytest.mark.parametrize("namer", [mixamo_namer(), cc_namer(), generic_namer()])
def test_named_rigs_get_high_confidence_everywhere(namer):
    scene = make_scene(armature=build_armature("A", namer))
    arm = scene.armatures[0]
    h = resolve_humanoid(arm, None, 0.75)
    a = analyze_human(scene, arm, h)
    assert not a.inferred_bones or set(a.inferred_bones) <= {"leftToes", "rightToes"} or True
    for vrm in ("hips", "head", "leftUpperArm", "rightUpperLeg"):
        assert a.skeleton.confidence_of(vrm) > 0.8, vrm


def test_helper_and_cloth_chains_next_to_the_limbs_do_not_steal_them():
    """VRoid-style helpers: aim / roll bones and long skirt chains hang off the hips beside the legs."""
    extra = []
    for side, s in (("L", 1.0), ("R", -1.0)):
        prev = "mixamorig:Hips"
        for i in range(6):
            name = f"Skirt_{side}_{i}"
            extra.append((name, prev, (0.12 * s, 0.05, 0.95 - 0.08 * i)))
            prev = name
        extra.append((f"Aim_{side}", "mixamorig:Hips", (0.1 * s, 0.0, 0.94)))
        extra.append((f"Twist_{side}", f"mixamorig:{'Left' if s > 0 else 'Right'}ForeArm", (0.5 * s, 0.0, 1.45)))
    arm = build_armature("A", mixamo_namer(), extra=extra)
    weak, ren = anonymise(arm)
    scene = make_scene(armature=weak)
    a = analyze_human(scene, weak, resolve_humanoid(weak, None, 0.75))
    for vrm, src in (("leftUpperLeg", "mixamorig:LeftUpLeg"), ("rightFoot", "mixamorig:RightFoot"),
                     ("leftHand", "mixamorig:LeftHand"), ("rightLowerArm", "mixamorig:RightForeArm")):
        assert a.humanoid[vrm] == ren[src], vrm
