"""Real Blender: a mannequin mesh with no armature becomes a skinned humanoid that the converter exports as VRM."""

import json

import numpy as np
import pytest

pytest.importorskip("bpy")
from moviva_fbx_to_vrm.runtime import load_blender

try:
    bpy = load_blender()  # bundled Blender + the VRM Add-on (packaging/fetch_addon.py)
except RuntimeError as exc:
    pytest.skip(str(exc), allow_module_level=True)

from character_vrm.blender import importer
from character_vrm.blender.bridge import RealBlenderBridge
from character_vrm.core.config import ConversionConfig
from character_vrm.core.pipeline import ConverterCore
from character_vrm.tests.fixtures.mannequin import mannequin_points


@pytest.fixture
def unrigged_glb(tmp_path):
    importer.reset_scene()
    pts = mannequin_points(300)
    n = len(pts) // 3 * 3  # arbitrary triangles over the point cloud: only the vertices matter here
    me = bpy.data.meshes.new("Mannequin")
    me.from_pydata([tuple(p) for p in pts[:n]], [], [(i, i + 1, i + 2) for i in range(0, n, 3)])
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("Mannequin", me))
    path = tmp_path / "mannequin.glb"
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
    return path


def test_unrigged_mannequin_is_rigged_and_exported(unrigged_glb, tmp_path):
    out = tmp_path / "m.vrm"
    outcome = ConverterCore(RealBlenderBridge()).convert(unrigged_glb, out, ConversionConfig(auto_rig=True))
    rig = outcome.report["autoRig"]
    assert rig["status"] in ("PASS", "CHECK") and rig["method"]
    assert out.stat().st_size > 1000
    assert json.loads(json.dumps(rig))["bones"][0] == "Hips"
