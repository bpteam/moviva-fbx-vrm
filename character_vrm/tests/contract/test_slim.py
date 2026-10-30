from types import SimpleNamespace as NS

from character_vrm.blender import slim
from character_vrm.core.model import MorphBind


def mesh(name, *keys):
    blocks = [NS(name=k) for k in ("Basis", *keys)]
    obj = NS(name=name, type="MESH", data=NS(shape_keys=NS(key_blocks=blocks, reference_key=blocks[0])))
    obj.shape_key_remove = blocks.remove
    return obj


def test_only_expression_keys_basis_and_moviva_vroid_morphs_survive():
    face, body = mesh("Face", "V_Open", "Mouth_Smile_L", "Fcl_BRW_Joy"), mesh("Body", "V_Open")
    plan = NS(expressions={"aa": NS(binds=[MorphBind("Face", "V_Open")])})  # only the expressions matter
    removed = slim.drop_unbound_shape_keys(plan, [face, body, NS(name="Armature", type="ARMATURE", data=None)])
    assert removed == 2
    assert [b.name for b in face.data.shape_keys.key_blocks] == ["Basis", "V_Open", "Fcl_BRW_Joy"]
    assert [b.name for b in body.data.shape_keys.key_blocks] == ["Basis"]  # same key name, unbound on this mesh


def test_textures_shrink_to_1024_on_the_longest_side_keeping_the_aspect():
    calls = []
    images = [NS(size=(2048, 1024), scale=lambda w, h: calls.append((w, h))),
              NS(size=(1024, 1024), scale=lambda w, h: calls.append("small one scaled")),
              NS(size=(0, 0), scale=lambda w, h: calls.append("empty one scaled"))]
    assert slim.cap_textures(images) == 1 and calls == [(1024, 512)]
