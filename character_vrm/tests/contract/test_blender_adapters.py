"""Contract tests for the other Blender-side modules, again against tiny MagicMock graphs."""

import ast
import pathlib
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

import character_vrm
from character_vrm.blender import importer, scene_inspector, vrm_addon
from character_vrm.blender.bridge import RealBlenderBridge
from character_vrm.core.errors import ConversionError
from character_vrm.core.model import LookAtPlan, SpringPlan, VrmPlan

PACKAGE = pathlib.Path(character_vrm.__file__).parent


def test_package_imports_without_bpy():
    """Fresh interpreter with ``bpy`` made un-importable: every module must still import."""
    code = (
        "import sys, importlib, pkgutil\n"
        "sys.modules['bpy'] = None\n"
        "import character_vrm\n"
        "for m in pkgutil.walk_packages(character_vrm.__path__, 'character_vrm.'):\n"
        "    if '.tests' in m.name or m.name.endswith(('blender_entry', '__main__')): continue\n"
        "    importlib.import_module(m.name)\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PACKAGE.parent)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def _python_files():
    return [p for p in PACKAGE.rglob("*.py") if "tests" not in p.parts]


def test_bpy_is_only_imported_inside_blender_adapters():
    allowed = {"importer.py", "scene_inspector.py", "vrm_addon.py", "compat_v4_7_2.py", "bridge.py",
               "anatomy_adapter.py", "shape_key_author.py", "rig_author.py", "hair_author.py", "face_geometry.py",
               "material_author.py", "slim.py", "autorig.py"}
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "bpy" or n.startswith("bpy.") for n in names):
                if not (path.parent.name == "blender" and path.name in allowed):
                    offenders.append(str(path.relative_to(PACKAGE)))
    assert offenders == []


def test_bpy_is_never_imported_at_module_level():
    for path in (PACKAGE / "blender").glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert not any("bpy" == getattr(a, "name", "").split(".")[0] for a in getattr(node, "names", [])), path.name


def test_vrm_addon_internals_only_live_in_compat_v4_7_2():
    offenders = [str(p.relative_to(PACKAGE)) for p in _python_files()
                 if "vrm_addon_extension" in p.read_text() and p.name != "compat_v4_7_2.py"]
    assert offenders == []


def test_no_custom_glb_or_vrm_serializer():
    for path in _python_files():
        text = path.read_text()
        assert "struct.pack" not in text, path.name  # reading a GLB header is allowed, writing one is not
        assert "write_bytes" not in text, path.name


# --- importer ------------------------------------------------------------------------


def test_gltf_uses_the_gltf_importer():
    bpy = MagicMock()
    bpy.ops.import_scene.gltf.return_value = {"FINISHED"}
    importer.import_character("/m/a.glb", "glb", bpy)
    bpy.ops.import_scene.gltf.assert_called_once_with(filepath="/m/a.glb")
    importer.import_character("/m/a.gltf", "gltf", bpy)
    assert bpy.ops.import_scene.gltf.call_count == 2
    bpy.ops.wm.fbx_import.assert_not_called()


def test_fbx_uses_the_current_fbx_importer_not_fbx2gltf_or_the_legacy_operator():
    bpy = MagicMock()
    bpy.ops.wm.fbx_import.return_value = {"FINISHED"}
    importer.import_character("/m/a.fbx", "fbx", bpy)
    bpy.ops.wm.fbx_import.assert_called_once_with(filepath="/m/a.fbx")
    bpy.ops.import_scene.fbx.assert_not_called()
    assert "subprocess" not in (PACKAGE / "blender" / "importer.py").read_text()  # no external FBX converter


def test_import_failure_and_missing_operator():
    bpy = MagicMock()
    bpy.ops.import_scene.gltf.return_value = {"CANCELLED"}
    with pytest.raises(ConversionError) as exc:
        importer.import_character("/m/a.glb", "glb", bpy)
    assert exc.value.code == "import-failed"
    bpy = MagicMock()
    bpy.ops.wm = MagicMock(spec=[])
    with pytest.raises(ConversionError) as exc:
        importer.import_character("/m/a.fbx", "fbx", bpy)
    assert exc.value.code == "importer-missing"


def test_reset_scene_removes_everything_without_ui_context():
    bpy = MagicMock()
    objects = [MagicMock(), MagicMock()]
    store = list(objects)
    bpy.data.objects.__iter__ = lambda self: iter(list(store))
    bpy.data.objects.remove.side_effect = lambda obj, do_unlink=False: store.remove(obj)
    bpy.context.object = None
    importer.reset_scene(bpy)
    assert bpy.data.objects.remove.call_count == 2 and store == []
    bpy.ops.object.mode_set.assert_not_called()  # no active object: no mode switch, no UI dependence


def test_reset_scene_fails_loudly_if_objects_survive():
    bpy = MagicMock()
    stubborn = MagicMock()
    stubborn.name = "Cube"
    bpy.data.objects.__iter__ = lambda self: iter([stubborn])
    bpy.context.object = None
    with pytest.raises(ConversionError) as exc:
        importer.reset_scene(bpy)
    assert exc.value.code == "scene-not-clean"


# --- scene inspection --------------------------------------------------------------------


def _bone(name, parent, head, tail, deform=True):
    b = MagicMock()
    b.name, b.parent, b.use_deform = name, parent, deform
    b.head_local, b.tail_local = head, tail
    return b


class Identity:
    def __matmul__(self, other):
        return other


def test_scene_inspection_builds_a_bpy_free_model():
    hips = _bone("Hips", None, (0, 0, 1), (0, 0, 1.1))
    spine = _bone("Spine", hips, (0, 0, 1.1), (0, 0, 1.2))
    armature = MagicMock(type="ARMATURE")
    armature.name = "Armature"
    armature.matrix_world = Identity()
    armature.data.bones = [hips, spine]

    modifier = MagicMock(type="ARMATURE", object=armature)
    face = MagicMock(type="MESH", modifiers=[modifier], parent=armature)
    face.name = "Face"
    face.data.vertices = range(100)
    face.data.polygons = [MagicMock(vertices=[0, 1, 2, 3]), MagicMock(vertices=[0, 1, 2])]
    group = MagicMock()
    group.name = "Hips"
    face.vertex_groups = [group]
    mat = MagicMock()
    mat.name = "Skin"
    slot = MagicMock(material=mat)
    face.material_slots = [slot]
    reference, blink, jaw = MagicMock(), MagicMock(slider_min=0.0, slider_max=1.0), MagicMock(slider_min=0.0, slider_max=1.0)
    blink.name, jaw.name = "eyeBlinkLeft", "jawOpen"
    face.data.shape_keys.reference_key = reference
    face.data.shape_keys.key_blocks = [reference, blink, jaw]

    tex = MagicMock(type="TEX_IMAGE")
    tex.image.name = "skin.png"
    principled = MagicMock(type="BSDF_PRINCIPLED")
    base = MagicMock(is_linked=True)
    principled.inputs.get.side_effect = lambda n: base if n == "Base Color" else None
    mat.node_tree.nodes = [tex, principled]
    image = MagicMock(size=(2048, 1024), filepath="", packed_file=object())
    image.name = "skin.png"

    bpy = MagicMock()
    bpy.data.objects = [armature, face]
    bpy.data.materials = [mat]
    bpy.data.images = [image]

    result = scene_inspector.inspect_scene("/m/a.glb", "glb", bpy)
    assert [a.name for a in result.armatures] == ["Armature"]
    assert [b.name for b in result.armatures[0].bones] == ["Hips", "Spine"]
    assert result.armatures[0].bone("Spine").parent == "Hips"
    mesh = result.meshes[0]
    assert (mesh.name, mesh.vertices, mesh.triangles, mesh.armature, mesh.vertex_groups, mesh.materials) == (
        "Face", 100, 3, "Armature", ["Hips"], ["Skin"])
    assert mesh.skinned
    assert [(k.mesh, k.name) for k in result.shape_keys] == [("Face", "eyeBlinkLeft"), ("Face", "jawOpen")]  # Basis skipped
    assert result.materials[0].channels == ["baseColor"] and result.materials[0].images == ["skin.png"]
    assert (result.images[0].width, result.images[0].height, result.images[0].packed) == (2048, 1024, True)


def test_isolate_hides_other_armatures_and_their_meshes():
    def obj(name, kind, bound=None):
        o = MagicMock(type=kind)
        o.name = name
        o.modifiers = [MagicMock(type="ARMATURE", object=bound)] if bound else []
        o.parent = None
        return o
    main, other = obj("Main", "ARMATURE"), obj("Other", "ARMATURE")
    body, prop, loose = obj("Body", "MESH", main), obj("Prop", "MESH", other), obj("Loose", "MESH")
    bpy = MagicMock()
    bpy.data.objects = [main, other, body, prop, loose]
    hidden = scene_inspector.isolate_armature("Main", bpy)
    assert hidden == ["Other", "Prop"]
    other.hide_set.assert_called_once_with(True)
    main.hide_set.assert_not_called() and body.hide_set.assert_not_called() and loose.hide_set.assert_not_called()


# --- vrm_addon glue / bridge ----------------------------------------------------------------


def test_environment_facts_collect_operator_availability():
    bpy = MagicMock()
    bpy.app.version = (5, 2, 2)
    bpy.context.preferences.addons.keys.return_value = []
    facts = vrm_addon.environment_facts(bpy)
    assert facts.blender_version == "5.2.2" and facts.addon_version is None and facts.addon_error


def test_configure_applies_in_order_and_activates_the_armature(monkeypatch):
    calls = []
    for name in ("set_spec_version", "apply_humanoid", "apply_meta", "apply_expressions", "apply_look_at", "apply_spring_bones"):
        monkeypatch.setattr(vrm_addon.addon, name, lambda *a, _n=name, **k: calls.append(_n))
    armature = MagicMock(type="ARMATURE")
    bpy = MagicMock()
    bpy.data.objects.get.return_value = armature
    plan = VrmPlan("Arm", {"hips": "Hips"}, {}, LookAtPlan(None, ""), SpringPlan(), {})
    vrm_addon.configure(plan, bpy)
    assert calls == ["set_spec_version", "apply_humanoid", "apply_meta", "apply_expressions", "apply_look_at", "apply_spring_bones"]
    assert bpy.context.view_layer.objects.active is armature


def test_export_calls_the_official_operator():
    bpy = MagicMock()
    bpy.ops.export_scene.vrm.return_value = {"FINISHED"}
    vrm_addon.export("Arm", "/tmp/o.vrm", bpy=bpy)
    bpy.ops.export_scene.vrm.assert_called_once_with(filepath="/tmp/o.vrm", armature_object_name="Arm",
                                                     enable_advanced_preferences=True, export_try_sparse_sk=True,
                                                     export_all_influences=False)


def test_real_bridge_is_lazy():
    assert RealBlenderBridge() is not None  # constructing it never touches bpy
