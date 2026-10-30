"""Glue between the pure plan and the add-on adapter (``compat_v4_7_2``). Imports ``bpy`` lazily."""

from __future__ import annotations

from typing import Any

from ..core.errors import ConversionError
from ..core.model import VrmPlan
from ..toolchain import EnvironmentFacts
from . import compat_v4_7_2 as addon


def environment_facts(bpy: Any = None) -> EnvironmentFacts:
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    facts = EnvironmentFacts(blender_version=".".join(str(p) for p in bpy.app.version))
    facts.addon_version, facts.addon_error = addon.detect_addon_version(bpy)
    facts.exporter_operator = addon.exporter_available(bpy)
    facts.gltf_importer = hasattr(bpy.ops.import_scene, "gltf")
    facts.gltf_exporter = hasattr(bpy.ops.export_scene, "gltf")
    facts.fbx_importer = hasattr(bpy.ops.wm, "fbx_import")
    return facts


def configure(plan: VrmPlan, bpy: Any = None) -> None:
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    armature = bpy.data.objects.get(plan.armature)
    if armature is None or armature.type != "ARMATURE":
        raise ConversionError("armature-vanished", f"Armature object '{plan.armature}' is not in the scene.")
    bpy.context.view_layer.objects.active = armature

    addon.set_spec_version(armature)
    addon.apply_humanoid(armature, plan.humanoid)
    addon.apply_meta(armature, plan.meta)
    addon.apply_expressions(armature, plan.expressions)
    addon.apply_look_at(armature, plan.look_at)
    addon.apply_spring_bones(armature, plan.springs, bpy.context)


def export(armature: str, output_path: str, all_influences: bool = False, bpy: Any = None) -> None:
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    addon.export_vrm(bpy.ops, output_path, armature, all_influences)
