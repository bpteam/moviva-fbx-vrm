"""core.surface_labels and compress.add_surface_labels on a hand-made skinned GLB."""

import numpy as np
import pytest

from character_vrm.core import compress
from character_vrm.core import surface_labels as SL


@pytest.mark.parametrize("name,cls", [
    ("F00_002_01_Tops_01_CLOTH (Instance)", "cloth"), ("N00_000_00_Body_00_SKIN", "skin"),
    ("F00_000_Hair_00_HAIR_01", "hair"), ("F00_000_00_EyeIris_00_EYE", "face_detail"),
    ("M00_001_01_AccessoryNeck_01_CLOTH", "cloth"), ("Body_Teeth", "face_detail"), ("Body_Hands", "skin"),
    ("Dress", "cloth"), ("Gemstone_1", "accessory"), ("Hair_SideWisps", "hair"), ("mesh_4_9.nr", None),
])
def test_class_from_names(name, cls):
    assert SL.class_from_name(name) == cls


def test_limb_needs_most_of_the_weight_on_the_arm():
    assert SL.label("skin", 0.9) == SL.ID["skin_limb"]
    assert SL.label("skin", 0.45) == SL.ID["skin_body"]  # the shoulder joins the torso
    assert SL.label("cloth", 0.7) == SL.ID["cloth_limb"]
    assert SL.label("accessory", 0.8) == SL.ID["cloth_limb"]  # a bracelet moves as a sleeve
    assert SL.label("hair", 1.0) == SL.ID["hair"]
    assert SL.decide("mesh_4_9.nr", "Object_2") == ("cloth", "default")  # never silently skin


def skinned_glb(tmp_path):
    """Two primitives over one vertex buffer: a skin material (vertices 0-3) and a sleeve (4-7). Joint 0 is the
    hips, joint 1 the left upper arm; vertices 2, 3, 6, 7 are on the arm."""
    pos = np.arange(24, dtype=np.float32).reshape(8, 3)
    joints = np.zeros((8, 4), np.uint8)
    joints[[2, 3, 6, 7], 0] = 1
    weights = np.zeros((8, 4), np.float32)
    weights[:, 0] = 1
    tri_a, tri_b = np.array([0, 1, 2, 1, 2, 3], np.uint16), np.array([4, 5, 6, 5, 6, 7], np.uint16)
    parts = [pos.tobytes(), joints.tobytes(), weights.tobytes(), tri_a.tobytes(), tri_b.tobytes()]
    blob, views, off = b"", [], 0
    for p in parts:
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(p)})
        blob += p
        off += len(p)
    acc = [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
           {"bufferView": 1, "componentType": 5121, "count": 8, "type": "VEC4"},
           {"bufferView": 2, "componentType": 5126, "count": 8, "type": "VEC4"},
           {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
           {"bufferView": 4, "componentType": 5123, "count": 6, "type": "SCALAR"}]
    attrs = {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(blob)}], "bufferViews": views,
           "accessors": acc, "nodes": [{"name": "hips"}, {"name": "arm"}, {"name": "body", "mesh": 0, "skin": 0}],
           "skins": [{"joints": [0, 1]}], "materials": [{"name": "X_Body_00_SKIN"}, {"name": "X_Tops_01_CLOTH"}],
           "meshes": [{"name": "Body", "primitives": [{"attributes": attrs, "indices": 3, "material": 0},
                                                     {"attributes": attrs, "indices": 4, "material": 1}]}],
           "extensionsUsed": ["VRMC_vrm"],
           "extensions": {"VRMC_vrm": {"humanoid": {"humanBones": {"hips": {"node": 0}, "leftUpperArm": {"node": 1}}}}}}
    path = tmp_path / "s.vrm"
    compress.write_glb(path, doc, blob)
    return path


def test_labels_written_as_attribute_and_extension(tmp_path):
    path = skinned_glb(tmp_path)
    report = compress.add_surface_labels(path)
    doc, blob = compress.read_glb(path)
    a = doc["meshes"][0]["primitives"][0]["attributes"]["_MOVIVA_SURFACE"]
    assert doc["meshes"][0]["primitives"][1]["attributes"]["_MOVIVA_SURFACE"] == a  # one per vertex buffer
    labels = compress._read_accessor(doc, blob, a)[:, 0].astype(int).tolist()
    s, c = SL.ID["skin_body"], SL.ID["cloth_body"]
    assert labels == [s, s, SL.ID["skin_limb"], SL.ID["skin_limb"], c, c, SL.ID["cloth_limb"], SL.ID["cloth_limb"]]
    assert doc["bufferViews"][doc["accessors"][a]["bufferView"]]["byteStride"] == 4
    assert doc["extensions"]["MOVIVA_surface_labels"]["classes"] == list(SL.CLASSES)
    assert "MOVIVA_surface_labels" in doc["extensionsUsed"]
    assert "MOVIVA_surface_labels" not in doc.get("extensionsRequired", [])
    assert doc["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]["leftUpperArm"] == {"node": 1}
    assert report["Body"]["materials"]["X_Tops_01_CLOTH"] == {"class": "cloth", "signal": "material-name"}
    # the labels survive meshopt (a 4-byte stride is a plain attribute to it)
    pytest.importorskip("meshoptimizer")
    compress.compress_geometry(path)
    doc, _ = compress.read_glb(path)
    assert "_MOVIVA_SURFACE" in doc["meshes"][0]["primitives"][0]["attributes"]
