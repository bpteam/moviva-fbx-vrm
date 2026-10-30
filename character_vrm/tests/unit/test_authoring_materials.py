"""Real-time material look: only what is measurably wrong is changed; maps beside the file are found per material."""

from character_vrm.core.anatomy.materials import MaterialSemantic, classify_materials
from character_vrm.core.authoring.materials import (
    DEAD_SPECULAR, LOOKS, index_textures, material_role, pick_texture, plan_materials,
)
from character_vrm.core.model import CharacterInspection, MaterialInfo, MeshInfo


def _inspection(*materials: MaterialInfo) -> CharacterInspection:
    return CharacterInspection("model.fbx", "fbx", materials=list(materials))


def _sem(name: str, semantic: str, meshes: list[str] | None = None) -> MaterialSemantic:
    return MaterialSemantic(name, semantic, 0.95, meshes=meshes or [])


def test_fbx_dead_specular_and_implausible_roughness_are_fixed():
    ins = _inspection(MaterialInfo("Std_Cornea_R", roughness=0.86, specular=0.0),
                      MaterialInfo("Scalp", roughness=0.0, specular=0.0))
    plan = plan_materials([_sem("Std_Cornea_R", "cornea"), _sem("Scalp", "hair")], ins)
    cornea, scalp = plan.edits
    assert (cornea.role, cornea.roughness, cornea.specular) == ("cornea", LOOKS["cornea"].roughness, 0.5)
    assert (scalp.role, scalp.roughness) == ("scalp", LOOKS["scalp"].roughness)


def test_plausible_or_unknown_values_are_left_alone():
    ins = _inspection(MaterialInfo("Skin", roughness=0.55, specular=0.5), MaterialInfo("Face"))
    assert plan_materials([_sem("Skin", "skin"), _sem("Face", "skin")], ins).edits == []
    assert DEAD_SPECULAR < 0.5


def test_blended_hair_is_clipped_and_eye_overlays_are_not_touched():
    ins = _inspection(MaterialInfo("Hair_Transparency", alpha_mode="BLEND", roughness=0.55, specular=0.2),
                      MaterialInfo("Std_Eye_Occlusion_R", alpha_mode="BLEND", roughness=0.9, specular=0.0))
    plan = plan_materials([_sem("Hair_Transparency", "hair"), _sem("Std_Eye_Occlusion_R", "eye")], ins)
    assert [(e.material, e.changes) for e in plan.edits] == [("Hair_Transparency", ["alpha clip 0.35"])]
    assert material_role("Std_Tearline_L", "cornea") is None


def test_maps_beside_the_source_are_found_per_material_and_mesh(tmp_path):
    for mesh in ("Side_bangs", "Side_part_mid"):
        d = tmp_path / "textures" / "Model" / mesh / "Hair_Transparency"
        d.mkdir(parents=True)
        (d / "Hair_Transparency_roughness.png").write_bytes(b"x")
    skin = tmp_path / "textures" / "Model" / "CC_Base_Body" / "Std_Skin_Head"
    skin.mkdir(parents=True)
    (skin / "Std_Skin_Head_roughness.png").write_bytes(b"x")
    (skin / "Std_Skin_Head_ao.png").write_bytes(b"x")
    (skin / "Std_Skin_Head_Diffuse.jpg").write_bytes(b"x")
    index = index_textures(tmp_path)
    assert set(index) == {("hair_transparency", "roughness"), ("std_skin_head", "roughness"), ("std_skin_head", "ao")}

    ins = _inspection(MaterialInfo("Std_Skin_Head", roughness=0.55, specular=0.27),
                      MaterialInfo("Hair_Transparency.001", roughness=0.86, specular=0.0),
                      MaterialInfo("Hair_Other", roughness=0.86))
    plan = plan_materials([_sem("Std_Skin_Head", "skin"), _sem("Hair_Transparency.001", "hair", ["Side_part_mid"])],
                          ins, index)
    head, hair = plan.edits
    assert set(head.textures) == {"roughness", "ao"} and head.roughness is None and head.specular is None
    assert "Side_part_mid" in hair.textures["roughness"] and plan.validate() == []
    cands = index[("hair_transparency", "roughness")]
    assert pick_texture(cands, ["Unrelated"]) is None


def test_ambiguous_maps_are_skipped_not_guessed(tmp_path):
    for mesh in ("A", "B"):
        (tmp_path / mesh).mkdir()
        (tmp_path / mesh / "Hair_roughness.png").write_bytes(b"x")
    ins = _inspection(MaterialInfo("Hair", roughness=0.6, specular=0.5))
    plan = plan_materials([_sem("Hair", "hair", ["C"])], ins, index_textures(tmp_path))
    assert plan.edits == [] and "none names its mesh" in plan.skipped[0].reason


def test_a_missing_map_is_a_plan_issue(tmp_path):
    ins = _inspection(MaterialInfo("Skin", roughness=0.55, specular=0.5))
    plan = plan_materials([_sem("Skin", "skin")], ins, {("skin", "roughness"): [str(tmp_path / "gone.png")]})
    assert plan.validate() and plan.valid_edits() == []


def test_a_cornea_material_on_the_eye_mesh_reads_as_cornea():
    ins = CharacterInspection("m.fbx", "fbx", meshes=[MeshInfo("CC_Base_Eye", 10, 10, materials=["Std_Cornea_R"])],
                              materials=[MaterialInfo("Std_Cornea_R", images=["Std_Cornea_R_Diffuse"])])
    (sem,) = classify_materials(ins)
    assert sem.semantic == "cornea"


def test_maps_of_the_model_beside_other_models_are_its_own(tmp_path):
    # blend/Amber.Fbx with blend/textures/Amber/... next to blend/Megane/textures/Mégane/...
    for folder in (("textures", "Amber"), ("Megane", "textures", "M\u00e9gane")):
        d = tmp_path.joinpath(*folder, "CC_Base_Body", "Std_Skin_Head")
        d.mkdir(parents=True)
        (d / "Std_Skin_Head_roughness.png").write_bytes(b"x")
    ins = CharacterInspection(str(tmp_path / "Amber.Fbx"), "fbx", materials=[MaterialInfo("Std_Skin_Head")])
    plan = plan_materials([_sem("Std_Skin_Head", "skin", ["CC_Base_Body"])], ins, index_textures(tmp_path))
    assert "Amber" in plan.edits[0].textures["roughness"]
    nfd = "Me\u0301gane"  # the same name, decomposed, as some file systems store it
    assert "gane" in pick_texture(index_textures(tmp_path)[("std_skin_head", "roughness")], [], nfd)
