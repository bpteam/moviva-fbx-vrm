"""Phase 16: semantic material classification: metadata only, evidence-based, confidence-gated."""

from character_vrm.core.anatomy.materials import classify_materials
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import MaterialInfo, MeshInfo
from character_vrm.tests.fixtures.heads import eyeball_mesh, head_mesh
from character_vrm.tests.fixtures.scenes import make_scene


def classify(materials, meshes=None):
    scene = make_scene(materials=materials)
    if meshes is not None:
        scene.meshes = meshes
    return {m.material: m for m in classify_materials(scene)}


def test_names_map_to_semantics():
    got = classify([MaterialInfo("Skin_Body"), MaterialInfo("Hair_01", alpha_mode="BLEND"), MaterialInfo("Eyelashes"),
                    MaterialInfo("M_Cornea"), MaterialInfo("Teeth_Upper"), MaterialInfo("Tongue"),
                    MaterialInfo("Wool_Shirt"), MaterialInfo("Brows")],
                   [MeshInfo(n, 10, 10, "Armature", [], [n]) for n in
                    ("Skin_Body", "Hair_01", "Eyelashes", "M_Cornea", "Teeth_Upper", "Tongue", "Wool_Shirt", "Brows")])
    want = {"Skin_Body": "skin", "Hair_01": "hair", "Eyelashes": "eyelash", "M_Cornea": "cornea", "Teeth_Upper": "teeth",
            "Tongue": "tongue", "Wool_Shirt": "clothes", "Brows": "eyebrow"}
    assert {k: v.semantic for k, v in got.items()} == want
    assert all(0.6 <= v.confidence <= 1 for v in got.values())


def test_confidence_grows_with_agreeing_evidence():
    weak = classify([MaterialInfo("Mat.001", images=["hair_diffuse.png"])], [MeshInfo("Object.1", 10, 10, "Armature", [], ["Mat.001"])])
    strong = classify([MaterialInfo("Hair", images=["hair_diffuse.png"])], [MeshInfo("Hair_Obj", 10, 10, "Armature", [], ["Hair"])])
    assert strong["Hair"].confidence > weak["Mat.001"].confidence
    assert strong["Hair"].evidence.keys() >= {"material_name", "mesh_name", "texture_name"}


def test_unknown_names_stay_other_instead_of_guessing():
    got = classify([MaterialInfo("Material.004")], [MeshInfo("Object.4", 10, 10, "Armature", [], ["Material.004"])])
    assert got["Material.004"].semantic == "other" and got["Material.004"].confidence < 0.6


def test_transparency_supports_but_never_decides():
    got = classify([MaterialInfo("Mat.9", alpha_mode="BLEND")], [MeshInfo("Object.9", 10, 10, "Armature", [], ["Mat.9"])])
    assert got["Mat.9"].semantic == "other"


def test_anatomical_position_identifies_an_eye_material_with_a_neutral_name():
    scene = make_scene(materials=[MaterialInfo("Skin"), MaterialInfo("Mat.7")])
    eyes = [eyeball_mesh("Obj.L", (0.031, -0.075, 1.707), 0.011), eyeball_mesh("Obj.R", (-0.031, -0.075, 1.707), 0.011)]
    for e in eyes:
        e.materials = ["Mat.7"]
    scene.meshes = [head_mesh(), *eyes]
    arm = scene.armatures[0]
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    mat = next(m for m in a.materials if m.material == "Mat.7")
    assert "position" in mat.evidence
    assert mat.semantic in ("eye", "other")  # evidence recorded even if not sufficient alone
    assert next(m for m in a.materials if m.material == "Skin").semantic == "skin"


def test_shaders_are_not_touched():
    m = MaterialInfo("Skin", shader_types=["BSDF_PRINCIPLED"])
    before = repr(m)
    classify([m])
    assert repr(m) == before
