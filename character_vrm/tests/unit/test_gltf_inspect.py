"""The pure-Python GLB / VRM reader: Blender's coordinate conversion, bind pose, skin weights, material slots and
sparse morph deltas, without Blender."""

import struct
from pathlib import Path

import pytest

from character_vrm.core.gltf_inspect import GltfError, inspect_glb, to_blender
from character_vrm.tests.fixtures.glb import binary_glb

AVATAR = Path(__file__).resolve().parents[4] / "avatar" / "public" / "models" / "mia.vrm"


def _tiny(tmp_path):
    """Two joints (root, child 1 m up) and one skinned quad with a morph target that moves vertex 2 up by 0.1."""
    pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    idx = [0, 1, 2, 0, 2, 3]
    joints = [(0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]
    weights = [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]
    morph = [(0, 0, 0), (0, 0, 0), (0, 0.1, 0), (0, 0, 0)]
    ibm = [(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1)]
    blob = b"".join(struct.pack("<3f", *p) for p in pos)
    blob += b"".join(struct.pack("<H", i) for i in idx)
    blob += b"".join(struct.pack("<4B", *j) for j in joints)
    blob += b"".join(struct.pack("<4f", *w) for w in weights)
    blob += b"".join(struct.pack("<3f", *m) for m in morph)
    blob += b"".join(struct.pack("<16f", *m) for m in ibm)
    views = [(0, 48), (48, 12), (60, 16), (76, 64), (140, 48), (188, 128)]
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": o, "byteLength": n} for o, n in views],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": 4, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 5, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
        "materials": [{"name": "Skin"}],
        "meshes": [{"name": "Face", "primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                                                    "indices": 1, "material": 0, "targets": [{"POSITION": 4}]}],
                    "extras": {"targetNames": ["Face.A"]}}],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": 5}],
        "nodes": [{"name": "Face", "mesh": 0, "skin": 0}, {"name": "Root", "children": [2]},
                  {"name": "Child", "translation": [0, 1, 0]}],
    }
    path = tmp_path / "tiny.glb"
    path.write_bytes(binary_glb(gltf, blob))
    return path


def test_reader_converts_like_the_blender_importer(tmp_path):
    gi = inspect_glb(_tiny(tmp_path))
    ins = gi.inspection
    arm = ins.armatures[0]
    assert [b.name for b in arm.bones] == ["Root", "Child"] and arm.bone("Child").parent == "Root"
    assert arm.bone("Child").head == pytest.approx(to_blender((0, 1, 0)))  # glTF +Y up -> Blender +Z up
    mesh = ins.meshes[0]
    assert mesh.name == "Face" and mesh.armature == arm.name and mesh.triangles == 2
    assert mesh.polygon_materials == [0, 0] and mesh.materials == ["Skin"]
    assert mesh.vertex_weights[2] == {"Child": 1.0}
    assert mesh.vertex_positions[2] == pytest.approx(to_blender((1, 1, 0)))  # bind pose = identity here
    key = ins.shape_keys[0]
    assert (key.mesh, key.name) == ("Face", "Face.A")
    assert set(key.deltas) == {2} and key.deltas[2] == pytest.approx(to_blender((0, 0.1, 0)))


def test_non_glb_input_is_refused(tmp_path):
    p = tmp_path / "x.glb"
    p.write_bytes(b"nope")
    with pytest.raises(GltfError):
        inspect_glb(p)


@pytest.mark.skipif(not AVATAR.is_file(), reason="bundled sample avatar not present")
def test_bundled_vrm_reads_with_morph_deltas_and_material_slots():
    ins = inspect_glb(AVATAR).inspection
    face = next(m for m in ins.meshes if m.name == "Face")
    assert len(face.materials) > 3 and len(face.polygon_materials) == len(face.polygons)
    blink = next(k for k in ins.shape_keys if k.name.endswith("Fcl_EYE_Close_L"))
    assert len(blink.deltas) > 50
