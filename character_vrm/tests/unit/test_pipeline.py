import json
from pathlib import Path

import pytest

from character_vrm.core.config import ConversionConfig, SpringConfig
from character_vrm.core.errors import ConversionError
from character_vrm.core.model import ArmatureInfo, BoneInfo, ImageInfo, MaterialInfo
from character_vrm.core.pipeline import ConverterCore, analyze, source_format
from character_vrm.core.report import format_inspection, suggest_map
from character_vrm.core.config import parse_config
from character_vrm.tests.fakes import FakeBlenderBridge
from character_vrm.tests.fixtures.rigs import (
    build_armature, cc_namer, generic_namer, mixamo_namer, opaque_namer,
)
from character_vrm.tests.fixtures.scenes import ARKIT_KEYS, VROID_KEYS, make_scene


def touch(tmp_path, name="model.glb") -> Path:
    path = tmp_path / name
    path.write_bytes(b"x")
    return path


def convert(tmp_path, scene, config=None, name="model.glb", **bridge_kw):
    bridge = FakeBlenderBridge(scene, **bridge_kw)
    out = tmp_path / "out.vrm"
    outcome = ConverterCore(bridge).convert(touch(tmp_path, name), out, config, dry_run=False)
    return bridge, out, outcome


def codes(outcome):
    return [w["code"] for w in outcome.report["warnings"]]


def test_full_conversion_flow_and_report(tmp_path):
    bridge, out, outcome = convert(tmp_path, make_scene(face_keys=ARKIT_KEYS, hair=True))
    assert bridge.calls == ["reset_scene", "import_character", "inspect", "isolate_armature", "configure_vrm", "export_vrm"]
    assert bridge.imported == (str(tmp_path / "model.glb"), "glb")
    assert out.exists()
    r = outcome.report
    assert r["status"] == "success"
    assert r["toolchain"] == {"blender": "5.2.2", "vrmAddon": "4.7.2", "target": "VRM 1.0"}
    assert r["humanoid"]["requiredMapped"] == 15 and r["humanoid"]["profile"] == "mixamo"
    assert r["expressions"]["capability"] == "full" and r["expressions"]["arkit"] == "likely"
    assert r["expressions"]["mapped"]["aa"]["status"] == "generated"
    assert r["lookAt"]["type"] == "expression"
    assert [c["bones"] for c in r["springBones"]["chains"]] == [["HairBack01", "HairBack02", "HairBack03"]]
    assert r["materials"]["mtoonConverted"] == 0 and r["materials"]["pbrPreserved"] == 1
    assert r["output"]["specVersion"] == "1.0" and r["output"]["humanBones"] >= 15
    for section in ("source", "geometry", "humanoid", "expressions", "lookAt", "springBones", "materials", "warnings"):
        assert section in r
    json.dumps(r)  # must be serialisable


def test_plan_reaches_the_adapter(tmp_path):
    bridge, _, _ = convert(tmp_path, make_scene(face_keys=VROID_KEYS))
    plan = bridge.plan
    assert plan.armature == "Armature"
    assert plan.humanoid["hips"] == "mixamorig:Hips"
    assert plan.meta == {"vrm_name": "out"}  # nothing licence-related is invented
    assert set(plan.expressions) >= {"aa", "blink", "happy"}


def test_meta_comes_only_from_config(tmp_path):
    cfg = parse_config({"meta": {"name": "Ada", "version": "1.0", "authors": ["Me"], "avatarPermission": "everyone"}})
    bridge, _, _ = convert(tmp_path, make_scene(), cfg)
    assert bridge.plan.meta == {"vrm_name": "Ada", "version": "1.0", "authors": ["Me"], "avatar_permission": "everyone"}
    assert "allow_redistribution" not in bridge.plan.meta and "commercial_usage" not in bridge.plan.meta


def test_dry_run_exports_nothing(tmp_path):
    bridge = FakeBlenderBridge(make_scene(face_keys=ARKIT_KEYS))
    out = tmp_path / "out.vrm"
    outcome = ConverterCore(bridge).convert(touch(tmp_path), out, None, dry_run=True)
    assert outcome.report["status"] == "dry-run" and not out.exists()
    assert "configure_vrm" not in bridge.calls and "export_vrm" not in bridge.calls


def test_fbx_without_shape_keys_warns_and_still_converts(tmp_path):
    _, _, outcome = convert(tmp_path, make_scene(fmt="fbx"), name="model.fbx")
    assert outcome.report["status"] == "success"
    warning = next(w for w in outcome.report["warnings"] if w["code"] == "fbx-no-shape-keys")
    assert "no facial shape keys" in warning["message"] and "Prefer a GLB/glTF source" in warning["message"]
    assert outcome.report["expressions"]["capability"] == "none"
    assert outcome.report["expressions"]["mapped"] == {}  # nothing invented


def test_glb_without_shape_keys_uses_the_generic_warning(tmp_path):
    _, _, outcome = convert(tmp_path, make_scene())
    assert "no-shape-keys" in codes(outcome) and "fbx-no-shape-keys" not in codes(outcome)


def test_fbx_with_shape_keys_does_not_warn(tmp_path):
    _, _, outcome = convert(tmp_path, make_scene(fmt="fbx", face_keys=VROID_KEYS), name="model.fbx")
    assert "fbx-no-shape-keys" not in codes(outcome)
    assert outcome.report["expressions"]["capability"] == "full"


def test_fbx_import_is_reported_with_fbx_format(tmp_path):
    bridge, _, _ = convert(tmp_path, make_scene(fmt="fbx"), name="Model.FBX")
    assert bridge.imported[1] == "fbx"


def test_unsupported_format(tmp_path):
    with pytest.raises(ConversionError) as exc:
        source_format("model.obj")
    assert exc.value.code == "unsupported-format"


def test_missing_source(tmp_path):
    with pytest.raises(ConversionError) as exc:
        ConverterCore(FakeBlenderBridge(make_scene())).convert(tmp_path / "nope.glb", tmp_path / "o.vrm")
    assert exc.value.code == "source-missing"


def test_no_armature_fails_hard(tmp_path):
    scene = make_scene()
    scene.armatures.clear()
    with pytest.raises(ConversionError) as exc:
        convert(tmp_path, scene)
    assert exc.value.code == "no-armature"


def test_armature_without_skinning_fails(tmp_path):
    scene = make_scene()
    for mesh in scene.meshes:
        mesh.vertex_groups = []
    with pytest.raises(ConversionError) as exc:
        convert(tmp_path, scene)
    assert exc.value.code == "no-skinned-mesh"


@pytest.mark.parametrize("drop,expected", [
    ({"hips"}, "hips"), ({"head", "headtop", "eye.L", "eye.R"}, "head"),
    ({"upperarm.L", "lowerarm.L", "hand.L"}, "leftUpperArm"),
    ({"upleg.R", "lowleg.R", "foot.R", "toes.R"}, "rightUpperLeg"),
])
def test_missing_required_bones_fail_hard_and_nothing_is_exported(tmp_path, drop, expected):
    arm = build_armature("Armature", generic_namer(), drop=drop)
    bridge = FakeBlenderBridge(make_scene(armature=arm))
    with pytest.raises(ConversionError) as exc:
        ConverterCore(bridge).convert(touch(tmp_path), tmp_path / "o.vrm")
    assert exc.value.code in ("humanoid-incomplete", "humanoid-invalid")
    assert any(expected in d for d in exc.value.details)
    assert "export_vrm" not in bridge.calls and "configure_vrm" not in bridge.calls
    assert exc.value.report["status"] == "failed"  # a failed report is still produced


def test_opaque_names_are_mapped_by_anatomy_analysis_and_convert_without_a_map(tmp_path):
    """Bone.000, Bone.001, ...: names say nothing, but topology, position and symmetry identify the humanoid."""
    arm = build_armature("Armature", opaque_namer())
    bridge, _, outcome = convert(tmp_path, make_scene(armature=arm))
    assert outcome.report["status"] == "success"
    mappings = outcome.report["humanoid"]["mappings"]
    assert {m["reason"].split(" ")[0] for m in mappings} == {"anatomy"}
    assert bridge.plan.humanoid["hips"] and bridge.plan.humanoid["leftUpperArm"] != bridge.plan.humanoid["rightUpperArm"]


def test_unusual_rig_without_humanoid_structure_needs_a_map_and_then_converts(tmp_path):
    """A skeleton with no mirrored limb pairs cannot be inferred: it still needs a reviewed map."""
    arm = build_armature("Armature", opaque_namer())
    chain = ArmatureInfo("Armature", [
        BoneInfo(f"Bone.{i:03d}", f"Bone.{i - 1:03d}" if i else None, (0.0, 0.0, 0.1 * i), (0.0, 0.0, 0.1 * i + 0.1))
        for i in range(12)])
    with pytest.raises(ConversionError):
        convert(tmp_path, make_scene(armature=chain))
    core_bridge = FakeBlenderBridge(make_scene(armature=arm))
    analysis = ConverterCore(core_bridge).inspect(touch(tmp_path), ConversionConfig(min_confidence=0.5))
    suggestion = suggest_map(analysis, 0.5)
    assert suggestion["humanoid"]
    # the reviewed suggestion is accepted as an explicit config and converts
    cfg = parse_config({"humanoid": suggestion["humanoid"]})
    _, _, outcome = convert(tmp_path, make_scene(armature=arm), cfg)
    assert outcome.report["humanoid"]["profile"] is None
    assert {m["reason"] for m in outcome.report["humanoid"]["mappings"]} == {"config"}


def test_config_naming_a_missing_bone_is_a_config_error(tmp_path):
    with pytest.raises(ConversionError) as exc:
        convert(tmp_path, make_scene(), parse_config({"humanoid": {"hips": "Nope"}}))
    assert exc.value.code == "config-invalid"


def test_ambiguous_primary_armature_fails_without_override(tmp_path):
    second = build_armature("Armature.001", mixamo_namer())
    scene = make_scene(extra_armature=second, extra_triangles=400_000)
    with pytest.raises(ConversionError) as exc:
        convert(tmp_path, scene)
    assert exc.value.code == "ambiguous-armature"
    assert any("Armature.001" in d for d in exc.value.details)


def test_override_resolves_ambiguity(tmp_path):
    second = build_armature("Armature.001", mixamo_namer())
    scene = make_scene(extra_armature=second, extra_triangles=400_000)
    bridge, _, outcome = convert(tmp_path, scene, ConversionConfig(armature="Armature"))
    assert outcome.report["status"] == "success"
    assert bridge.hidden == ["Prop"]  # the other armature's mesh is excluded from the export
    assert "meshes-excluded" in codes(outcome)


def test_secondary_prop_armature_is_not_ambiguous(tmp_path):
    prop = build_armature("Prop", generic_namer(), drop={"upleg.L", "lowleg.L", "foot.L", "toes.L"})
    scene = make_scene(extra_armature=prop)
    bridge, _, outcome = convert(tmp_path, scene)
    assert outcome.report["geometry"]["armatures"][0]["selected"] is True
    assert "armature-secondary" in codes(outcome)


def test_export_failure_is_fatal(tmp_path):
    with pytest.raises(ConversionError) as exc:
        convert(tmp_path, make_scene(), fail_export=True)
    assert exc.value.code == "export-failed"


@pytest.mark.parametrize("mode,needle", [
    ("none", "was not written"), ("garbage", "not a GLB"), ("no-skins", "skinning was lost"),
    ("spec-0", "specVersion"), ("missing-bone", "leftHand"),
])
def test_invalid_output_is_fatal(tmp_path, mode, needle):
    with pytest.raises(ConversionError) as exc:
        convert(tmp_path, make_scene(), output=mode)
    assert exc.value.code == "invalid-output"
    assert needle in str(exc.value)


def test_missing_fingers_eyes_and_hair_are_warnings_only(tmp_path):
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    _, _, outcome = convert(tmp_path, make_scene(armature=arm))
    assert {"fingers-missing", "eyes-missing", "springs-none", "lookat-missing"} <= set(codes(outcome))
    assert outcome.report["status"] == "success"


def test_heavy_assets_only_warn(tmp_path):
    mats = [MaterialInfo(f"M{i}", True, ["BSDF_PRINCIPLED"], False, [], ["baseColor"], []) for i in range(70)]
    proc = MaterialInfo("Marble", True, ["BSDF_PRINCIPLED"], True, ["TEX_NOISE"], [], [])
    images = [ImageInfo("huge", 16384, 16384)]
    _, _, outcome = convert(tmp_path, make_scene(materials=[*mats, proc], images=images, triangles=2_000_000))
    assert {"materials-many", "materials-procedural", "textures-large", "geometry-heavy"} <= set(codes(outcome))
    assert outcome.report["status"] == "success"


def test_no_springs_flag_and_config(tmp_path):
    cfg = ConversionConfig(spring=SpringConfig(enabled=False))
    bridge, _, outcome = convert(tmp_path, make_scene(hair=True), cfg)
    assert bridge.plan.springs.chains == [] and outcome.report["springBones"]["enabled"] is False


def test_cc_rig_with_twist_bones_keeps_them_unmapped(tmp_path):
    extra = [("CC_Base_L_UpperarmTwist01", "CC_Base_L_Upperarm", (0.3, 0, 1.45))]
    arm = build_armature("Armature", cc_namer(), extra=extra)
    bridge, _, outcome = convert(tmp_path, make_scene(armature=arm))
    assert outcome.report["humanoid"]["profile"] == "character-creator"
    assert "CC_Base_L_UpperarmTwist01" not in bridge.plan.humanoid.values()
    assert outcome.report["humanoid"]["preservedUnmappedBones"] >= 1


def test_inspect_text_matches_the_documented_shape():
    scene = make_scene(face_keys=ARKIT_KEYS, hair=True)
    text = format_inspection(analyze(scene, ConversionConfig()))
    for needle in ("Character inspection", "Geometry", "skinned meshes", "triangles", "Armature", "Humanoid",
                   "mapped required: 15/15", "Face", "ARKit profile:   likely", "Expressions",
                   "blink             generated", "aa                generated", "Hair", "candidate chains: 1",
                   "Result", "convertible"):
        assert needle in text, needle
