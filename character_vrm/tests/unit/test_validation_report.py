import json
import struct

import pytest

from character_vrm.core.errors import ConversionError
from character_vrm.core.materials import analyze_materials
from character_vrm.core.model import MaterialInfo, ImageInfo
from character_vrm.core.pipeline import analyze
from character_vrm.core.config import ConversionConfig
from character_vrm.core.validation import check_vrm_output, read_glb_json
from character_vrm.core.pipeline import build_plan
from character_vrm.tests.fixtures.glb import glb_bytes, vrm_gltf
from character_vrm.tests.fixtures.scenes import ARKIT_KEYS, make_scene


def setup(tmp_path, **kw):
    scene = make_scene(face_keys=ARKIT_KEYS, hair=True)
    plan = build_plan(analyze(scene, ConversionConfig()), tmp_path / "o.vrm")
    path = tmp_path / "o.vrm"
    path.write_bytes(glb_bytes(vrm_gltf(plan, scene, **kw)))
    return scene, plan, path


def test_valid_output_is_summarised(tmp_path):
    scene, plan, path = setup(tmp_path)
    check = check_vrm_output(path, plan, scene)
    assert check.spec_version == "1.0" and check.humanoid_bones >= 15 and check.springs == 1
    assert "aa" in check.expressions and check.skins == 2 and not check.warnings


@pytest.mark.parametrize("mutate,needle", [
    (lambda p: p.write_bytes(b""), "empty"),
    (lambda p: p.write_bytes(b"XXXX" + b"\0" * 40), "not a GLB"),
    (lambda p: p.write_bytes(struct.pack("<4sII", b"glTF", 1, 20) + b"\0" * 8), "unsupported GLB version"),
    (lambda p: p.write_bytes(p.read_bytes()[:-3]), "header length"),
])
def test_corrupt_containers(tmp_path, mutate, needle):
    scene, plan, path = setup(tmp_path)
    mutate(path)
    with pytest.raises(ConversionError) as exc:
        check_vrm_output(path, plan, scene)
    assert exc.value.code == "invalid-output" and needle in str(exc.value)


def test_expression_bind_out_of_range(tmp_path):
    scene, plan, path = setup(tmp_path)
    gltf = read_glb_json(path)
    gltf["extensions"]["VRMC_vrm"]["expressions"]["preset"]["aa"]["morphTargetBinds"][0]["index"] = 999
    path.write_bytes(glb_bytes(gltf))
    with pytest.raises(ConversionError, match="out of range"):
        check_vrm_output(path, plan, scene)


def test_spring_joint_with_invalid_node(tmp_path):
    scene, plan, path = setup(tmp_path)
    gltf = read_glb_json(path)
    gltf["extensions"]["VRMC_springBone"]["springs"][0]["joints"][0]["node"] = 9999
    path.write_bytes(glb_bytes(gltf))
    with pytest.raises(ConversionError, match="invalid node"):
        check_vrm_output(path, plan, scene)


def test_missing_springs_in_output_is_a_warning(tmp_path):
    scene, plan, path = setup(tmp_path)
    gltf = read_glb_json(path)
    del gltf["extensions"]["VRMC_springBone"]
    path.write_bytes(glb_bytes(gltf))
    assert [w.code for w in check_vrm_output(path, plan, scene).warnings] == ["output-springs-missing"]


def test_missing_materials_in_output(tmp_path):
    scene, plan, path = setup(tmp_path, materials=0)
    with pytest.raises(ConversionError, match="materials"):
        check_vrm_output(path, plan, scene)


def test_materials_are_preserved_not_converted():
    scene = make_scene(materials=[
        MaterialInfo("A", True, ["BSDF_PRINCIPLED"], False, [], ["baseColor", "normal", "roughness", "occlusion", "emissive"], ["a"]),
        MaterialInfo("B", True, ["BSDF_DIFFUSE"], False, [], [], []),
        MaterialInfo("C", True, ["BSDF_PRINCIPLED"], True, ["TEX_NOISE"], [], []),
    ], images=[ImageInfo("a", 2048, 4096)])
    result = analyze_materials(scene)
    assert (len(result.pbr_preserved), result.non_principled, result.procedural) == (1, ["B"], ["C"])
    assert result.channels["occlusion"] == 1 and result.max_texture == (2048, 4096)
    assert {w.code for w in result.warnings} == {"materials-procedural", "materials-non-principled"}
