"""Adapter contract tests: with small MagicMock object graphs, verify the calls we issue against the
VRM Add-on 4.7.2 property tree. They do NOT test Blender or the add-on itself (that is the local
integration run, tests/integration/README.md)."""

from unittest.mock import MagicMock, call

import pytest

from character_vrm.blender import compat_v4_7_2 as addon
from character_vrm.core.errors import ConversionError
from character_vrm.core.model import (
    ColliderPlan, ExpressionMapping, LookAtPlan, MorphBind, SpringChain, SpringPlan,
)
from character_vrm.core.config import DEFAULT_SPRING_PRESET


class Node:
    """Emulates a ``...node`` group: remembers assignments like the add-on's ``bone_name`` does."""

    def __init__(self, accept=True):
        self.bone_name = ""
        self._accept = accept

    def __setattr__(self, key, value):
        if key == "bone_name" and not getattr(self, "_accept", True):
            return
        object.__setattr__(self, key, value)


def armature(accept=True):
    arm = MagicMock(name="armature")
    ext = arm.data.vrm_addon_extension
    ext.spec_version = "1.0"
    return arm, ext


def test_snake_case_matches_the_add_on_property_names():
    cases = {"hips": "hips", "upperChest": "upper_chest", "leftUpperArm": "left_upper_arm",
             "leftThumbMetacarpal": "left_thumb_metacarpal", "rightLittleIntermediate": "right_little_intermediate",
             "blinkLeft": "blink_left", "lookUp": "look_up", "aa": "aa", "leftToes": "left_toes"}
    for camel, expected in cases.items():
        assert addon.snake(camel) == expected


def test_spec_version_is_set_to_1_0():
    arm, ext = armature()
    ext.spec_version = "0.0"

    class Ext:
        spec_version = "0.0"
    arm.data.vrm_addon_extension = Ext()
    addon.set_spec_version(arm)
    assert arm.data.vrm_addon_extension.spec_version == "1.0"


def test_spec_version_failure_is_reported():
    arm = MagicMock()

    class Stubborn:
        @property
        def spec_version(self):
            return "0.0"

        @spec_version.setter
        def spec_version(self, value):
            pass
    arm.data.vrm_addon_extension = Stubborn()
    with pytest.raises(ConversionError):
        addon.set_spec_version(arm)


def test_humanoid_assignments_are_issued():
    arm, ext = armature()
    human_bones = ext.vrm1.humanoid.human_bones
    nodes = {}

    def make(name):
        node = Node()
        nodes[name] = node
        getattr(human_bones, name).node = node
    for name in ("hips", "spine", "left_upper_arm", "upper_chest", "left_thumb_metacarpal"):
        make(name)

    addon.apply_humanoid(arm, {"hips": "Hips", "spine": "Spine", "leftUpperArm": "LUA", "upperChest": "UC",
                               "leftThumbMetacarpal": "T1"})
    assert {k: v.bone_name for k, v in nodes.items()} == {
        "hips": "Hips", "spine": "Spine", "left_upper_arm": "LUA", "upper_chest": "UC", "left_thumb_metacarpal": "T1"}
    assert human_bones.initial_automatic_bone_assignment is False


def test_humanoid_rejection_by_the_add_on_is_detected():
    arm, ext = armature()
    ext.vrm1.humanoid.human_bones.hips.node = Node(accept=False)
    with pytest.raises(addon.AddonApplyError, match="rejected humanoid"):
        addon.apply_humanoid(arm, {"hips": "Hips"})


def test_expression_assignments_are_issued():
    arm, ext = armature()
    preset = ext.vrm1.expressions.preset
    binds_by_expr = {}
    for name in ("aa", "blink_left"):
        collection = MagicMock()
        added = []

        def add(added=added):
            entry = MagicMock()
            added.append(entry)
            return entry
        collection.add.side_effect = add
        getattr(preset, name).morph_target_binds = collection
        binds_by_expr[name] = (collection, added)

    addon.apply_expressions(arm, {
        "aa": ExpressionMapping("aa", "compound", [MorphBind("Face", "jawOpen", 0.8), MorphBind("Teeth", "jawOpen", 0.8)], 0.8, "x"),
        "blinkLeft": ExpressionMapping("blinkLeft", "direct", [MorphBind("Face", "Blink_L", 1.0)], 0.9, "x"),
    })
    collection, added = binds_by_expr["aa"]
    collection.clear.assert_called_once()
    assert [(e.node.mesh_object_name, e.index, e.weight) for e in added] == [("Face", "jawOpen", 0.8), ("Teeth", "jawOpen", 0.8)]
    collection, added = binds_by_expr["blink_left"]
    assert [(e.node.mesh_object_name, e.index, e.weight) for e in added] == [("Face", "Blink_L", 1.0)]
    assert ext.vrm1.expressions.initial_automatic_expression_assignment is False


def test_look_at_assignments():
    arm, ext = armature()
    addon.apply_look_at(arm, LookAtPlan("bone", "eyes"))
    assert ext.vrm1.look_at.type == "bone"
    addon.apply_look_at(arm, LookAtPlan("expression", "morphs", (0.0, 0.1, 0.0)))
    assert ext.vrm1.look_at.type == "expression" and ext.vrm1.look_at.offset_from_head_bone == (0.0, 0.1, 0.0)
    arm2, ext2 = armature()
    ext2.vrm1.look_at.type = "untouched"
    addon.apply_look_at(arm2, LookAtPlan(None, "none"))
    assert ext2.vrm1.look_at.type == "untouched"


def test_spring_bone_assignments_are_issued():
    arm, ext = armature()
    sb = ext.spring_bone1
    context = MagicMock(name="context")
    colliders, groups, springs = [], [], []

    def add_collider(ctx, armature_obj):
        assert ctx is context and armature_obj is arm
        c = MagicMock(uuid=f"c{len(colliders)}")
        colliders.append(c)
        return c

    def add_group():
        g = MagicMock(uuid=f"g{len(groups)}")
        groups.append(g)
        return g

    def add_spring():
        s = MagicMock()
        s.joints_added, s.groups_added = [], []
        s.add_joint.side_effect = lambda: s.joints_added.append(MagicMock()) or s.joints_added[-1]
        s.add_collider_group.side_effect = lambda: s.groups_added.append(MagicMock()) or s.groups_added[-1]
        springs.append(s)
        return s
    sb.add_collider.side_effect = add_collider
    sb.add_collider_group.side_effect = add_group
    sb.add_spring.side_effect = add_spring

    plan = SpringPlan(
        chains=[SpringChain("Hair_A", ["A1", "A2", "A3"], 0.95, "x", DEFAULT_SPRING_PRESET)],
        colliders=[ColliderPlan("head_collider", "Head", 0.1, (0, 0.05, 0))])
    addon.apply_spring_bones(arm, plan, context)

    collider = colliders[0]
    assert collider.node.bone_name == "Head" and collider.shape_type == "Sphere"
    assert collider.shape.sphere.radius == 0.1 and collider.shape.sphere.offset == (0, 0.05, 0)
    assert groups[0].vrm_name == "head_collider"
    assert groups[0].add_collider.return_value.collider_uuid == "c0"
    spring = springs[0]
    assert spring.vrm_name == "Hair_A"
    assert [j.node.bone_name for j in spring.joints_added] == ["A1", "A2", "A3"]
    for joint in spring.joints_added:
        assert (joint.stiffness, joint.drag_force, joint.gravity_power, joint.hit_radius) == (
            DEFAULT_SPRING_PRESET.stiffness, DEFAULT_SPRING_PRESET.drag_force,
            DEFAULT_SPRING_PRESET.gravity_power, DEFAULT_SPRING_PRESET.hit_radius)
    assert spring.groups_added[0].collider_group_uuid == "g0"


def test_no_chains_means_no_spring_calls():
    arm, ext = armature()
    addon.apply_spring_bones(arm, SpringPlan(colliders=[ColliderPlan("x", "b", 0.1, (0, 0, 0))]), MagicMock())
    assert ext.spring_bone1.mock_calls == []


def test_meta_only_sets_what_is_given():
    arm, ext = armature()
    meta = ext.vrm1.meta
    meta.authors.add.side_effect = lambda: MagicMock()
    addon.apply_meta(arm, {"vrm_name": "Ada", "version": "1.0", "authors": ["A", "B"], "commercial_usage": "personalProfit"})
    assert meta.vrm_name == "Ada" and meta.version == "1.0" and meta.commercial_usage == "personalProfit"
    meta.authors.clear.assert_called_once()
    assert meta.authors.add.call_count == 2


def test_meta_leaves_licence_defaults_alone_when_not_configured():
    class Meta:
        """Records every attribute assignment."""

        def __init__(self):
            object.__setattr__(self, "assigned", [])
            object.__setattr__(self, "authors", MagicMock())

        def __setattr__(self, key, value):
            self.assigned.append(key)

    arm = MagicMock()
    meta = Meta()
    arm.data.vrm_addon_extension.vrm1.meta = meta
    addon.apply_meta(arm, {"vrm_name": "Ada"})
    assert meta.assigned == ["vrm_name"]


def test_export_operator_is_called_with_the_armature_pinned():
    ops = MagicMock()
    ops.export_scene.vrm.return_value = {"FINISHED"}
    addon.export_vrm(ops, "/tmp/out.vrm", "Armature")
    ops.export_scene.vrm.assert_called_once_with(filepath="/tmp/out.vrm", armature_object_name="Armature",
                                                 enable_advanced_preferences=True, export_try_sparse_sk=True,
                                                 export_all_influences=False)


def test_all_bone_influences_is_an_explicit_opt_in():
    ops = MagicMock()
    ops.export_scene.vrm.return_value = {"FINISHED"}
    addon.export_vrm(ops, "/tmp/out.vrm", "Armature")
    assert ops.export_scene.vrm.call_args.kwargs["export_all_influences"] is False
    addon.export_vrm(ops, "/tmp/out.vrm", "Armature", all_influences=True)
    assert ops.export_scene.vrm.call_args.kwargs["export_all_influences"] is True
    assert ops.export_scene.vrm.call_args.kwargs["enable_advanced_preferences"] is True


@pytest.mark.parametrize("result", [{"CANCELLED"}, set(), {"RUNNING_MODAL"}])
def test_export_that_does_not_finish_raises(result):
    ops = MagicMock()
    ops.export_scene.vrm.return_value = result
    with pytest.raises(ConversionError) as exc:
        addon.export_vrm(ops, "/tmp/out.vrm", "Armature")
    assert exc.value.code == "export-failed"


# --- discovery -----------------------------------------------------------------------


def test_addon_module_names():
    assert addon.find_addon_module_name(["x", "bl_ext.user_default.vrm"]) == "bl_ext.user_default.vrm"
    assert addon.find_addon_module_name(["io_scene_vrm"]) == "io_scene_vrm"
    assert addon.find_addon_module_name(["bl_ext.user_default.vrma_other", "io_scene_fbx"]) is None


def test_version_detection(monkeypatch):
    import sys
    import types
    bpy = MagicMock()
    bpy.context.preferences.addons.keys.return_value = ["bl_ext.user_default.vrm"]
    module = types.ModuleType("bl_ext.user_default.vrm")
    module.bl_info = {"version": (4, 7, 2)}
    monkeypatch.setitem(sys.modules, "bl_ext.user_default.vrm", module)
    assert addon.detect_addon_version(bpy) == ("4.7.2", None)
    module.bl_info = {"version": (4, 10, 1)}
    assert addon.detect_addon_version(bpy)[0] == "4.10.1"

    bpy.context.preferences.addons.keys.return_value = ["bl_ext.user_default.other_vrm_tool", "io_scene_gltf2"]
    version, problem = addon.detect_addon_version(bpy)
    assert version is None and "not installed or not enabled" in problem


def test_extension_version_detection_reads_manifest(monkeypatch, tmp_path):
    import sys
    import types

    package = tmp_path / "vrm"
    package.mkdir()
    (package / "__init__.py").write_text("# extension package\n", encoding="utf-8")
    (package / "blender_manifest.toml").write_text('id = "vrm"\nversion = "4.7.2"\n', encoding="utf-8")

    bpy = MagicMock()
    bpy.context.preferences.addons.keys.return_value = ["bl_ext.blender_org.vrm"]
    module = types.ModuleType("bl_ext.blender_org.vrm")
    module.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "bl_ext.blender_org.vrm", module)

    assert addon.detect_addon_version(bpy) == ("4.7.2", None)


def test_exporter_availability():
    bpy = MagicMock(spec=["ops"])
    bpy.ops = MagicMock(spec=["export_scene"])
    bpy.ops.export_scene = MagicMock(spec=["vrm"])
    assert addon.exporter_available(bpy)
    bpy.ops.export_scene = MagicMock(spec=["gltf"])
    assert not addon.exporter_available(bpy)
