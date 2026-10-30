"""Contract tests for the Blender-side authors, against tiny hand-written object graphs (not a fake Blender).

They prove the *calls* and the safety rules (namespace, never overwrite, rollback, mode restore). They do not prove that
real Blender accepts them: that is what tests/integration/README.md is for.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from character_vrm.blender import anatomy_adapter, hair_author, rig_author, shape_key_author
from character_vrm.blender.bridge import RealBlenderBridge
from character_vrm.core.authoring.fingers import FingerRigPlan, NewBone
from character_vrm.core.authoring.hair_rig import HairRigChain, HairRigPlan
from character_vrm.core.authoring.plan import AuthoringResult, AutoAuthorPlan
from character_vrm.core.authoring.face_plan import FaceAuthoringPlan
from character_vrm.core.authoring.plans import ShapeKeyPlan
from character_vrm.core.config import AutoAuthorConfig


class V(tuple):
    def __new__(cls, *xyz):
        if len(xyz) == 1:
            xyz = tuple(xyz[0])
        return super().__new__(cls, tuple(float(x) for x in xyz))

    def __add__(self, o):
        return V(*(a + b for a, b in zip(self, o)))

    def __sub__(self, o):
        return V(*(a - b for a, b in zip(self, o)))

    @property
    def length(self):
        return sum(a * a for a in self) ** 0.5


class Identity:
    def to_3x3(self):
        return self

    def inverted(self):
        return self

    def __matmul__(self, v):
        return V(*v)


class Point:
    def __init__(self, co):
        self.co = V(*co)


class Block:
    def __init__(self, name, points):
        self.name, self.data, self.value, self.slider_min, self.slider_max = name, [Point(p.co) for p in points], 1.0, -1, 1


class KeyBlocks(dict):
    pass


class Keys:
    def __init__(self, base_points):
        self.key_blocks = KeyBlocks()
        self.reference_key = Block("Basis", base_points)
        self.key_blocks["Basis"] = self.reference_key


class Mesh:
    def __init__(self, n=6):
        self.vertices = [Point((i, 0, 0)) for i in range(n)]
        self.shape_keys = None


class MeshObject:
    type = "MESH"

    def __init__(self, name, n=6, with_keys=False):
        self.name, self.data, self.matrix_world = name, Mesh(n), Identity()
        self.removed = []
        if with_keys:
            self.data.shape_keys = Keys(self.data.vertices)

    def shape_key_add(self, name, from_mix=False):
        if self.data.shape_keys is None:
            self.data.shape_keys = Keys(self.data.vertices)
            block = self.data.shape_keys.reference_key
            block.name = name
            self.data.shape_keys.key_blocks = KeyBlocks({name: block})
            return block
        block = Block(name, self.data.shape_keys.reference_key.data)
        self.data.shape_keys.key_blocks[name] = block
        return block

    def shape_key_remove(self, block):
        self.removed.append(block.name)
        del self.data.shape_keys.key_blocks[block.name]


@pytest.fixture(autouse=True)
def mathutils_stub(monkeypatch):
    mod = types.ModuleType("mathutils")
    mod.Vector = V
    monkeypatch.setitem(sys.modules, "mathutils", mod)


def bpy_with(*objects):
    bpy = MagicMock()
    bpy.data.objects.get.side_effect = lambda n: next((o for o in objects if o.name == n), None)
    return bpy


def plan(name="Prosopon_aa", offsets=None):
    return ShapeKeyPlan(name, offsets or {"Face": {1: (0.0, -0.01, 0.0), 3: (0.0, 0.0, 0.02)}}, 0.9)


# ---- shape keys ------------------------------------------------------------------------------------


def test_creates_a_basis_and_the_prosopon_key_with_planned_deltas():
    face = MeshObject("Face")
    applied, failed = shape_key_author.apply_shape_keys([plan()], bpy_with(face))
    assert applied == {"Prosopon_aa": ["Face"]} and failed == {}
    keys = face.data.shape_keys
    assert "Basis" in keys.key_blocks and "Prosopon_aa" in keys.key_blocks
    block = keys.key_blocks["Prosopon_aa"]
    assert block.value == 0.0 and block.slider_min == 0.0 and block.slider_max == 1.0
    assert block.data[1].co == V(1, -0.01, 0) and block.data[3].co == V(3, 0, 0.02) and block.data[2].co == V(2, 0, 0)


def test_source_shape_keys_are_never_modified_or_overwritten():
    face = MeshObject("Face", with_keys=True)
    source = face.shape_key_add("mouthSmile")
    before = [tuple(p.co) for p in source.data]
    ref_before = [tuple(p.co) for p in face.data.shape_keys.reference_key.data]
    applied, failed = shape_key_author.apply_shape_keys([plan()], bpy_with(face))
    assert applied and [tuple(p.co) for p in source.data] == before
    assert [tuple(p.co) for p in face.data.shape_keys.reference_key.data] == ref_before
    # a name that already exists is refused, not overwritten
    applied, failed = shape_key_author.apply_shape_keys([plan()], bpy_with(face))
    assert applied == {} and "already exists" in failed["Prosopon_aa"]


def test_names_outside_the_namespace_are_refused():
    applied, failed = shape_key_author.apply_shape_keys([plan("aa")], bpy_with(MeshObject("Face")))
    assert applied == {} and "namespace" in failed["aa"]


def test_partial_failure_rolls_the_key_back_everywhere():
    face = MeshObject("Face")
    both = plan(offsets={"Face": {1: (0, 0, 0.01)}, "Missing": {0: (0, 0, 0.01)}})
    applied, failed = shape_key_author.apply_shape_keys([both], bpy_with(face))
    assert applied == {} and "not found" in failed["Prosopon_aa"]
    assert "Prosopon_aa" not in face.data.shape_keys.key_blocks and face.removed == ["Prosopon_aa"]


def test_validation_accepts_the_planned_key_and_rejects_deviations():
    face = MeshObject("Face")
    p = plan()
    bpy = bpy_with(face)
    applied, _ = shape_key_author.apply_shape_keys([p], bpy)
    assert shape_key_author.validate_shape_keys([p], applied, bpy) == {}
    face.data.shape_keys.key_blocks["Prosopon_aa"].data[5].co = V(5, 1, 0)      # an unplanned vertex moved
    bad = shape_key_author.validate_shape_keys([p], applied, bpy)
    assert "Prosopon_aa" in bad and "deviates" in bad["Prosopon_aa"]
    face.data.shape_keys.key_blocks["Prosopon_aa"].data[5].co = V(5, 0, 0)
    face.data.shape_keys.key_blocks["Prosopon_aa"].data[3].co = V(3, 0, 0)      # a planned vertex did not move
    assert "moved" in shape_key_author.validate_shape_keys([p], applied, bpy)["Prosopon_aa"]
    face.data.shape_keys.key_blocks["Prosopon_aa"].data[3].co = V(3, 0, 0.02)
    face.data.shape_keys.key_blocks["Prosopon_aa"].data[1].co = V(1, -0.5, 0)   # a planned vertex is wrong
    assert "deviates" in shape_key_author.validate_shape_keys([p], applied, bpy)["Prosopon_aa"]


def test_validation_ignores_planned_offsets_below_the_tolerance():
    # a vertex with a trace of jaw weight gets a planned offset too small to register as a move
    face = MeshObject("Face")
    p = plan(offsets={"Face": {1: (0.0, -0.01, 0.0), 2: (0.0, 0.0, shape_key_author.TOLERANCE / 10)}})
    bpy = bpy_with(face)
    applied, _ = shape_key_author.apply_shape_keys([p], bpy)
    assert shape_key_author.validate_shape_keys([p], applied, bpy) == {}


# ---- bones and weights ------------------------------------------------------------------------------


class EditBone:
    def __init__(self, name):
        self.name, self.head, self.tail, self.parent, self.use_deform, self.use_connect = name, None, None, None, False, True


class EditBones(dict):
    def new(self, name):
        self[name] = EditBone(name)
        return self[name]


def armature_bpy(existing=("Hand",)):
    arm = MagicMock()
    arm.matrix_world.inverted.return_value = Identity()
    edit = EditBones({n: EditBone(n) for n in existing})
    arm.data.edit_bones = edit
    bpy = MagicMock()
    bpy.data.objects.__getitem__.side_effect = lambda n: arm
    return bpy, arm, edit


def test_bones_are_created_in_edit_mode_and_the_mode_is_always_restored():
    bpy, arm, edit = armature_bpy()
    created = rig_author.add_bones("Armature", [NewBone("Prosopon_a_1", "Hand", (0, 0, 1), (0, 0, 2)),
                                                NewBone("Prosopon_a_2", "Prosopon_a_1", (0, 0, 2), (0, 0, 3))], bpy)
    assert created == {"Prosopon_a_1", "Prosopon_a_2"}
    b = edit["Prosopon_a_2"]
    assert b.parent is edit["Prosopon_a_1"] and b.use_deform is True and b.use_connect is False and b.tail == V(0, 0, 3)
    modes = [c.kwargs["mode"] for c in bpy.ops.object.mode_set.call_args_list]
    assert modes == ["EDIT", "OBJECT"]


@pytest.mark.parametrize("bone,message", [(NewBone("Hand", "Hand", (0, 0, 0), (0, 0, 1)), "already exists"),
                                          (NewBone("Prosopon_x", "Nope", (0, 0, 0), (0, 0, 1)), "not found")])
def test_bone_creation_never_overwrites_and_restores_object_mode_on_error(bone, message):
    bpy, arm, edit = armature_bpy()
    with pytest.raises((ValueError, LookupError), match=message):
        rig_author.add_bones("Armature", [bone], bpy)
    assert [c.kwargs["mode"] for c in bpy.ops.object.mode_set.call_args_list][-1] == "OBJECT"


class Group:
    def __init__(self, name, index):
        self.name, self.index, self.added, self.removed = name, index, [], []

    def add(self, verts, w, mode):
        self.added.append((tuple(verts), round(w, 6), mode))

    def remove(self, verts):
        self.removed.append(tuple(verts))


class Groups(dict):
    def new(self, name):
        self[name] = Group(name, len(self))
        return self[name]


class WeightedObject:
    def __init__(self, name, groups, weights):
        self.name = name
        self.vertex_groups = Groups({g.name: g for g in groups})
        self.data = types.SimpleNamespace(vertices=[types.SimpleNamespace(groups=[types.SimpleNamespace(group=g, weight=w)
                                                                                  for g, w in ws.items()]) for ws in weights])


def test_weight_transfer_moves_only_a_fraction_of_the_hand_weight():
    hand = Group("Hand", 0)
    obj = WeightedObject("Hands", [hand], [{0: 0.8}, {0: 1.0}])
    bpy = bpy_with(obj)
    rig_author.transfer_weights({"Hands": {0: {"Prosopon_f_1": 0.5, "Prosopon_f_2": 0.25}}}, "Hand", bpy)
    assert obj.vertex_groups["Prosopon_f_1"].added == [((0,), 0.4, "REPLACE")]
    assert obj.vertex_groups["Prosopon_f_2"].added == [((0,), 0.2, "REPLACE")]
    assert hand.added == [((0,), 0.2, "REPLACE")]  # 0.8 * (1 - 0.75) stays with the hand bone


def test_finger_rig_applies_bones_then_weights():
    bpy, arm, edit = armature_bpy(("Hand",))
    hand = Group("Hand", 0)
    obj = WeightedObject("Hands", [hand], [{0: 1.0}])
    objects = {"Armature": arm, "Hands": obj}
    bpy.data.objects.__getitem__.side_effect = lambda n: objects[n]
    bpy.data.objects.get.side_effect = lambda n: objects.get(n)
    p = FingerRigPlan("left", "Hand", [NewBone("Prosopon_left_Index_1", "Hand", (0, 0, 0), (0, 0, 1), "leftIndexProximal")],
                      {"Hands": {0: {"Prosopon_left_Index_1": 1.0}}}, 0.95)
    assert rig_author.apply_finger_rig(p, "Armature", bpy) == {"Prosopon_left_Index_1"}
    assert obj.vertex_groups["Prosopon_left_Index_1"].added


def test_hair_rig_replaces_the_head_share_only_on_planned_vertices():
    bpy, arm, edit = armature_bpy(("Head",))
    head = Group("Head", 0)
    obj = WeightedObject("Hair", [head], [{0: 1.0}, {0: 1.0}])
    objects = {"Armature": arm, "Hair": obj}
    bpy.data.objects.__getitem__.side_effect = lambda n: objects[n]
    bpy.data.objects.get.side_effect = lambda n: objects.get(n)
    chain = HairRigChain("Prosopon_Hair_x", "Hair", [NewBone("Prosopon_Hair_x_1", "Head", (0, 0, 1), (0, 0, 0)),
                                                     NewBone("Prosopon_Hair_x_2", "Prosopon_Hair_x_1", (0, 0, 0), (0, 0, -1)),
                                                     NewBone("Prosopon_Hair_x_3", "Prosopon_Hair_x_2", (0, 0, -1), (0, 0, -2))],
                         "ponytail", 0.95)
    hp = HairRigPlan([chain], {"Hair": {1: {"Prosopon_Hair_x_1": 0.5, "Prosopon_Hair_x_2": 0.5}}}, parent_bone="Head")
    created = hair_author.apply_hair_rig(hp, "Armature", bpy)
    assert len(created) == 3 and head.removed == [(1,)]
    assert obj.vertex_groups["Prosopon_Hair_x_1"].added == [((1,), 0.5, "REPLACE")]


# ---- the router ------------------------------------------------------------------------------------


def authoring_plan(keys=(), fingers=(), hair=None):
    face = FaceAuthoringPlan(shape_keys=list(keys))
    return AutoAuthorPlan(AutoAuthorConfig(enabled=True), "Armature", face=face, fingers=list(fingers), hair_rig=hair)


def test_router_creates_validates_and_reports():
    face = MeshObject("Face")
    result = anatomy_adapter.apply_authoring(authoring_plan([plan()]), bpy_with(face))
    assert result.applied_shape_keys == {"Prosopon_aa": ["Face"]} and result.failed == {} and result.validation == []


def test_router_removes_keys_that_fail_result_validation(monkeypatch):
    face = MeshObject("Face")
    monkeypatch.setattr(shape_key_author, "validate_shape_keys", lambda plans, applied, bpy=None: {"Prosopon_aa": "vertex 1 deviates"})
    result = anatomy_adapter.apply_authoring(authoring_plan([plan()]), bpy_with(face))
    assert result.applied_shape_keys == {} and "deviates" in result.failed["Prosopon_aa"]
    assert "Prosopon_aa" not in face.data.shape_keys.key_blocks
    assert result.validation == ["Prosopon_aa: vertex 1 deviates"]


def test_router_records_a_failed_rig_and_carries_on(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("edit mode refused")
    monkeypatch.setattr(rig_author, "apply_finger_rig", boom)
    face = MeshObject("Face")
    finger = FingerRigPlan("left", "Hand", [NewBone("Prosopon_left_Index_1", "Hand", (0, 0, 0), (0, 0, 1), "leftIndexProximal")], {}, 0.95)
    result = anatomy_adapter.apply_authoring(authoring_plan([plan()], [finger]), bpy_with(face))
    assert "edit mode refused" in result.failed["left finger rig"] and "Prosopon_aa" in result.applied_shape_keys
    assert result.created_bones == set()


def test_real_bridge_delegates_to_the_adapter(monkeypatch):
    seen = []
    monkeypatch.setattr(anatomy_adapter, "apply_authoring", lambda plan: seen.append(plan) or AuthoringResult())
    p = authoring_plan()
    assert isinstance(RealBlenderBridge().apply_authoring(p), AuthoringResult) and seen == [p]
