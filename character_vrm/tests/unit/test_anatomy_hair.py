"""Phases 13-14: hair detection from many signals, chains with poor names, generated rig plan and spring presets."""

import pytest

from character_vrm.core.anatomy.confidence import Thresholds
from character_vrm.core.anatomy.hair import (
    ALPHA_CARDS, HAIR_CONFIDENT, HAIR_POSSIBLE, NOT_HAIR, PONYTAIL, SOLID_MASS, classify_hair_type,
)
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.authoring.hair_rig import plan_hair_rig
from character_vrm.core.authoring.springbones import SPRING_PRESETS, plan_hair_springs, preset_for
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import ArmatureInfo, BoneInfo, MaterialInfo, MeshInfo
from character_vrm.tests.fixtures.hands import strip_mesh
from character_vrm.tests.fixtures.heads import head_mesh
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer
from character_vrm.tests.fixtures.scenes import make_scene

HEAD = "mixamorig:Head"


def hair_chain_bones(prefix, n=4, x=0.0, y=0.11, z0=1.75, parent=HEAD, step=0.06):
    """A chain hanging behind the head (+Y is backwards); returns the ``extra`` tuples and the names."""
    names = [f"{prefix}.{i:03d}" for i in range(n)]
    extra, prev = [], parent
    for i, nm in enumerate(names):
        extra.append((nm, prev, (x, y, z0 - step * i)))
        prev = nm
    return extra, names


def scene_with(meshes, extra=(), materials=None):
    arm = build_armature("Armature", mixamo_namer(), extra=list(extra))
    scene = make_scene(armature=arm, materials=materials)
    scene.meshes = meshes
    return scene, arm


def analyse(scene, arm, thresholds=None):
    kw = {"thresholds": thresholds} if thresholds else {}
    return analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75), **kw)


def ponytail_mesh(name="Hair_Tail", bone=None, material="Hair"):
    return strip_mesh(name, (0.0, 0.10, 1.76), (0.0, 0.16, 1.12), 0.05, bone or HEAD, 14, material)


def test_hair_mesh_is_detected_from_its_name_and_material():
    scene, arm = scene_with([head_mesh(), ponytail_mesh()], materials=[MaterialInfo("Skin"), MaterialInfo("Hair")])
    a = analyse(scene, arm)
    by = {m.mesh: m for m in a.hair.meshes}
    assert by["Hair_Tail"].classification == HAIR_CONFIDENT and by["Hair_Tail"].confidence > 0.9
    assert {"object_name", "material_name"} <= set(by["Hair_Tail"].evidence)
    assert by["Face"].classification == NOT_HAIR  # a face on the head is not hair


def test_unnamed_mesh_on_the_scalp_is_only_possible_never_confident():
    m = ponytail_mesh("Object.014", material="Mat.003")
    scene, arm = scene_with([head_mesh(), m], materials=[MaterialInfo("Skin"), MaterialInfo("Mat.003")])
    a = analyse(scene, arm)
    by = {x.mesh: x for x in a.hair.meshes}
    assert by["Object.014"].classification == HAIR_POSSIBLE
    assert set(by["Object.014"].evidence) >= {"head_proximity", "scalp"} and "object_name" not in by["Object.014"].evidence


def test_a_body_mesh_far_from_the_head_is_not_hair():
    scene, arm = scene_with([head_mesh(), strip_mesh("Object.9", (0.0, 0.0, 1.0), (0.0, 0.0, 0.2), 0.3, "mixamorig:Hips", 12)])
    a = analyse(scene, arm)
    assert next(x for x in a.hair.meshes if x.mesh == "Object.9").classification == NOT_HAIR


def test_hair_chains_are_found_with_poor_bone_names_from_hierarchy_and_skinning():
    extra, names = hair_chain_bones("Bone", n=4)
    tail = ponytail_mesh(bone=names[0])
    tail.vertex_weights = {i: {names[min(3, i // 8)]: 1.0} for i in range(len(tail.vertex_positions))}
    tail.vertex_groups = names
    scene, arm = scene_with([head_mesh(), tail], extra, [MaterialInfo("Skin"), MaterialInfo("Hair")])
    a = analyse(scene, arm)
    assert len(a.hair.chains) == 1
    chain = a.hair.chains[0]
    assert chain.bones == names and not chain.named and chain.confidence > 0.75
    assert chain.evidence["skinning"] == 1.0 and chain.evidence["hierarchy"] == 1.0


def test_face_bones_are_not_hair_chains():
    jaw = [("Jaw.001", HEAD, (0.0, -0.05, 1.62)), ("Jaw.002", "Jaw.001", (0.0, -0.07, 1.58))]
    front = [("Bone.300", HEAD, (0.0, -0.10, 1.70)), ("Bone.301", "Bone.300", (0.0, -0.11, 1.66))]
    scene, arm = scene_with([head_mesh()], jaw + front)
    a = analyse(scene, arm)
    assert [c for c in a.hair.chains if c.confidence >= 0.4] == []


def test_alpha_card_masses_are_classified_and_never_rigged():
    cards = []
    verts, polys, weights = [], [], {}
    for k in range(20):
        x = -0.08 + 0.008 * k
        base = len(verts)
        verts += [(x, 0.05, 1.78), (x + 0.02, 0.05, 1.78), (x + 0.02, 0.09, 1.55), (x, 0.09, 1.55)]
        polys.append((base, base + 1, base + 2, base + 3))
    weights = {i: {HEAD: 1.0} for i in range(len(verts))}
    hair = MeshInfo("Hair_Cards", len(verts), 40, "Armature", [HEAD], ["HairCards"], vertex_positions=verts,
                    polygons=polys, vertex_weights=weights)
    scene, arm = scene_with([head_mesh(), hair], materials=[MaterialInfo("Skin"), MaterialInfo("HairCards", alpha_mode="BLEND")])
    a = analyse(scene, arm)
    assessed = next(m for m in a.hair.meshes if m.mesh == "Hair_Cards")
    assert assessed.hair_type == ALPHA_CARDS and assessed.classification == HAIR_CONFIDENT
    plan = plan_hair_rig(a, scene, "Armature", enabled=True, thresholds=Thresholds(generated_hair=0.5))
    assert plan.chains == [] and any("alpha-cards" in s.reason for s in plan.skipped)


def test_hair_type_classification():
    scene, arm = scene_with([head_mesh(), ponytail_mesh()])
    a = analyse(scene, arm)
    assert classify_hair_type(ponytail_mesh(), a.frame, a.head_height, alpha=False) == (PONYTAIL, 1)
    blob = head_mesh("Hair_Cap", center=(0.0, 0.0, 1.76), radii=(0.09, 0.10, 0.06), grid=24)
    assert classify_hair_type(blob, a.frame, a.head_height, alpha=False)[0] == SOLID_MASS


def test_generated_hair_rig_is_off_by_default_and_reports_why():
    scene, arm = scene_with([head_mesh(), ponytail_mesh()], materials=[MaterialInfo("Skin"), MaterialInfo("Hair")])
    a = analyse(scene, arm)
    plan = plan_hair_rig(a, scene, "Armature", enabled=False)
    assert plan.chains == [] and "feature disabled" in plan.skipped[0].reason


def test_generated_ponytail_rig_when_enabled_and_confident():
    scene, arm = scene_with([head_mesh(), ponytail_mesh()], materials=[MaterialInfo("Skin"), MaterialInfo("Hair")])
    a = analyse(scene, arm)
    plan = plan_hair_rig(a, scene, "Armature", enabled=True)
    assert len(plan.chains) == 1 and plan.validate() == []
    chain = plan.chains[0]
    assert 3 <= len(chain.bones) <= 8 and chain.category == "ponytail"
    assert chain.bones[0].parent == HEAD and chain.bones[1].parent == chain.bones[0].name
    assert all(b.name.startswith("Prosopon_") for b in chain.bones)
    # root is near the scalp, tip near the end of the ribbon; the chain descends
    assert chain.bones[0].head[2] > chain.bones[-1].tail[2] and chain.bones[0].head[2] > 1.6
    weights = plan.weights["Hair_Tail"]
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in weights.values())
    root_vertex = min(weights, key=lambda v: -scene.meshes[1].vertex_positions[v][2])
    assert weights[root_vertex].get(chain.bones[0].name, 0) > 0.9


def test_existing_hair_bones_mean_no_generated_rig():
    extra, names = hair_chain_bones("HairBack", n=4)
    scene, arm = scene_with([head_mesh(), ponytail_mesh()], extra, [MaterialInfo("Skin"), MaterialInfo("Hair")])
    a = analyse(scene, arm)
    plan = plan_hair_rig(a, scene, "Armature", enabled=True)
    assert plan.chains == [] and "already exist" in plan.skipped[0].reason


def test_low_confidence_hair_is_not_rigged():
    scene, arm = scene_with([head_mesh(), ponytail_mesh()], materials=[MaterialInfo("Skin"), MaterialInfo("Hair")])
    a = analyse(scene, arm)
    plan = plan_hair_rig(a, scene, "Armature", enabled=True, thresholds=Thresholds(generated_hair=1.01))
    assert plan.chains == [] and any("below" in s.reason for s in plan.skipped)


# ---- spring bones -----------------------------------------------------------------------------


def named_chain_scene(n_chains, n_bones=4, step=0.06, prefix="HairBack"):
    extra, all_names = [], []
    for c in range(n_chains):
        e, names = hair_chain_bones(f"{prefix}{c}", n=n_bones, x=-0.05 + 0.03 * c, step=step)
        extra += e
        all_names += names
    verts_meshes = [head_mesh(), strip_mesh("Hair_Main", (0.0, 0.10, 1.76), (0.0, 0.14, 1.2), 0.1, all_names[0])]
    hair = verts_meshes[1]
    hair.vertex_groups = all_names
    hair.vertex_weights = {i: {all_names[i % len(all_names)]: 1.0} for i in range(len(hair.vertex_positions))}
    return scene_with(verts_meshes, extra, [MaterialInfo("Skin"), MaterialInfo("Hair")])


def test_spring_presets_are_central_and_selected_by_category():
    scene, arm = named_chain_scene(1, n_bones=5, step=0.06)
    a = analyse(scene, arm)
    chain = a.hair.chains[0]
    assert chain.category == "ponytail"  # one long chain at the back
    scene2, arm2 = named_chain_scene(5, n_bones=4, step=0.03)
    a2 = analyse(scene2, arm2)
    assert {c.category for c in a2.hair.chains} == {"short-hair"}
    assert set(SPRING_PRESETS) == {"short-hair", "long-hair", "ponytail", "braid", "soft-strand"}


def test_spring_parameters_follow_length_and_anatomy_not_one_constant():
    short = preset_for("soft-strand", 0.5, 1.7, 0.17)
    long_ = preset_for("soft-strand", 1.6, 1.7, 0.17)
    assert long_.stiffness < short.stiffness                         # long chains are softer
    tall = preset_for("ponytail", 1.0, 3.4, 0.34)
    assert tall.hit_radius > preset_for("ponytail", 1.0, 1.7, 0.17).hit_radius
    tiny_head = preset_for("ponytail", 1.0, 1.7, 0.02)
    assert tiny_head.hit_radius <= 0.1 * 0.02 + 1e-9                 # never larger than the head allows


def test_hair_springs_from_inferred_chains_and_static_fallback_when_unsure():
    scene, arm = named_chain_scene(2, n_bones=4)
    a = analyse(scene, arm)
    plan, skipped = plan_hair_springs(a, None)
    assert len(plan.chains) == len(a.hair.chains) == 2 and skipped == []
    assert all(c.preset.stiffness > 0 and "inferred hair chain" in c.reason for c in plan.chains)
    plan, skipped = plan_hair_springs(a, None, Thresholds(hair_confident=1.01))
    assert plan.chains == [] and all("hair stays static" in s.reason for s in skipped)
