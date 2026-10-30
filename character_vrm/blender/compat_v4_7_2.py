"""The ONLY module that touches the VRM Add-on for Blender's property tree.

Pinned to VRM Add-on **4.7.2** (``toolchain.json``); every property path below was read from the
``v4.7.2`` sources (``editor/**``), none is guessed:

* ``armature.data.vrm_addon_extension.spec_version``                       (editor/extension.py)
* ``...vrm1.humanoid.human_bones.<snake_case>.node.bone_name``               (editor/vrm1/property_group.py)
* ``...vrm1.expressions.preset.<name>.morph_target_binds[].node.mesh_object_name / .index / .weight``
* ``...vrm1.look_at.type / .offset_from_head_bone``
* ``...spring_bone1.colliders / collider_groups / springs`` and their ``add_*`` helpers
* ``...vrm1.meta.*``
* ``bpy.ops.export_scene.vrm(filepath=..., armature_object_name=...)``     (exporter/export_scene.py)

Upgrading the add-on means re-reading those sources and changing this file (plus ``toolchain.json``).
Nothing here imports ``bpy`` at module level; callers pass the objects in, which is also what the
contract tests rely on.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable

from ..core.errors import ConversionError
from ..core.model import ExpressionMapping, LookAtPlan, SpringPlan

SUPPORTED_ADDON_VERSION = "4.7.2"  # asserted equal to toolchain.json by a unit test
SPEC_VERSION = "1.0"

_SNAKE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class AddonApplyError(ConversionError):
    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__("addon-apply-failed", message, details)


def snake(name: str) -> str:
    """``leftUpperArm`` -> ``left_upper_arm``, ``blinkLeft`` -> ``blink_left``."""
    return _SNAKE.sub("_", name).lower()


def extension(armature: Any) -> Any:
    return armature.data.vrm_addon_extension


# ---------------------------------------------------------------- discovery / doctor


def find_addon_module_name(addon_names: Iterable[str]) -> str | None:
    """Legacy add-on module ``io_scene_vrm`` or the Blender extension ``bl_ext.<repo>.vrm``."""
    for name in addon_names:
        if name == "io_scene_vrm" or (name.startswith("bl_ext.") and name.endswith(".vrm")):
            return name
    return None


def _version_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (tuple, list)) and value:
        return ".".join(str(part) for part in value)
    return None


def _manifest_version(module: Any) -> str | None:
    """Read the version from a Blender Extension manifest next to the loaded package."""
    roots: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        roots.append(Path(module_file).parent)
    for module_path in getattr(module, "__path__", ()) or ():
        roots.append(Path(module_path))

    seen: set[Path] = set()
    for root in roots:
        manifest = root / "blender_manifest.toml"
        if manifest in seen or not manifest.is_file():
            continue
        seen.add(manifest)
        try:
            data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        version = _version_string(data.get("version"))
        if version:
            return version
    return None


def detect_addon_version(bpy: Any) -> tuple[str | None, str | None]:
    """Return ``(version, problem)`` for a legacy add-on or Blender Extension package."""
    names = list(bpy.context.preferences.addons.keys())
    module_name = find_addon_module_name(names)
    if module_name is None:
        hints = [n for n in names if "vrm" in n.lower()]
        return None, "VRM Add-on is not installed or not enabled" + (f" (similar: {hints})" if hints else "")
    module = sys.modules.get(module_name)
    version = _version_string(getattr(module, "bl_info", {}).get("version")) if module is not None else None
    if version is None and module is not None and module_name.startswith("bl_ext."):
        version = _manifest_version(module)
    if not version:
        return None, f"{module_name} is enabled but exposes neither bl_info nor blender_manifest.toml version"
    return version, None


def exporter_available(bpy: Any) -> bool:
    return hasattr(bpy.ops.export_scene, "vrm")


# ---------------------------------------------------------------- configuration


def set_spec_version(armature: Any) -> None:
    extension(armature).spec_version = SPEC_VERSION
    if extension(armature).spec_version != SPEC_VERSION:
        raise AddonApplyError(f"could not set spec_version {SPEC_VERSION}")


def apply_humanoid(armature: Any, mapping: dict[str, str]) -> None:
    human_bones = extension(armature).vrm1.humanoid.human_bones
    human_bones.initial_automatic_bone_assignment = False  # ours is authoritative, never the add-on's guess
    failures = []
    for vrm_bone, source in mapping.items():
        node = getattr(human_bones, snake(vrm_bone)).node
        node.bone_name = source
        if node.bone_name != source:  # the add-on ignores unknown / duplicate assignments silently
            failures.append(f"{vrm_bone}: '{source}' was not accepted (got '{node.bone_name}')")
    if failures:
        raise AddonApplyError("VRM Add-on rejected humanoid bone assignments", failures)


def apply_expressions(armature: Any, expressions: dict[str, ExpressionMapping]) -> None:
    group = extension(armature).vrm1.expressions
    group.initial_automatic_expression_assignment = False
    failures = []
    for name, mapping in expressions.items():
        expression = getattr(group.preset, snake(name))
        expression.morph_target_binds.clear()
        for bind in mapping.binds:
            entry = expression.morph_target_binds.add()
            entry.node.mesh_object_name = bind.mesh
            entry.index = bind.shape_key
            entry.weight = bind.weight
            if entry.node.mesh_object_name != bind.mesh:
                failures.append(f"{name}: mesh object '{bind.mesh}' not accepted")
    if failures:
        raise AddonApplyError("VRM Add-on rejected expression binds", failures)


def apply_look_at(armature: Any, plan: LookAtPlan) -> None:
    look_at = extension(armature).vrm1.look_at
    if plan.type is not None:
        look_at.type = plan.type  # "bone" | "expression"
    if plan.offset_from_head_bone is not None:
        look_at.offset_from_head_bone = plan.offset_from_head_bone


def apply_spring_bones(armature: Any, plan: SpringPlan, context: Any) -> None:
    if not plan.chains:
        return
    spring_bone = extension(armature).spring_bone1

    groups = []
    for collider_plan in plan.colliders:
        collider = spring_bone.add_collider(context, armature)
        collider.node.bone_name = collider_plan.bone
        collider.shape_type = "Sphere"
        collider.shape.sphere.radius = collider_plan.radius
        collider.shape.sphere.offset = collider_plan.offset
        group = spring_bone.add_collider_group()
        group.vrm_name = collider_plan.name
        group.add_collider().collider_uuid = collider.uuid
        groups.append(group)

    for chain in plan.chains:
        spring = spring_bone.add_spring()
        spring.vrm_name = chain.name
        for bone in chain.bones:
            joint = spring.add_joint()
            joint.node.bone_name = bone
            joint.hit_radius = chain.preset.hit_radius
            joint.stiffness = chain.preset.stiffness
            joint.gravity_power = chain.preset.gravity_power
            joint.drag_force = chain.preset.drag_force
        for group in groups:
            spring.add_collider_group().collider_group_uuid = group.uuid


_META_SCALARS = (
    "vrm_name", "version", "copyright_information", "contact_information", "third_party_licenses",
    "avatar_permission", "commercial_usage", "credit_notation", "modification", "allow_redistribution",
    "allow_excessively_violent_usage", "allow_excessively_sexual_usage", "allow_political_or_religious_usage",
    "allow_antisocial_or_hate_usage", "other_license_url",
)


def apply_meta(armature: Any, meta: dict[str, object]) -> None:
    """Only what the caller provides. Licence fields left out keep the add-on's restrictive defaults."""
    target = extension(armature).vrm1.meta
    for key in _META_SCALARS:
        if key in meta:
            setattr(target, key, meta[key])
    for key in ("authors", "references"):
        if key in meta:
            collection = getattr(target, key)
            collection.clear()
            for value in meta[key]:  # type: ignore[union-attr]
                collection.add().value = value


# ---------------------------------------------------------------- export


def export_vrm(ops: Any, filepath: str, armature_object_name: str, all_influences: bool = False) -> None:
    """``bpy.ops.export_scene.vrm``. ``armature_object_name`` pins the exported armature, which matters
    as soon as the scene holds more than one. ``all_influences`` is an explicit opt-in: the add-on's
    default (at most four bone influences per vertex) is what every consumer supports."""
    kwargs: dict[str, Any] = {"filepath": filepath, "armature_object_name": armature_object_name}
    if all_influences:
        kwargs.update(enable_advanced_preferences=True, export_all_influences=True)
    result = ops.export_scene.vrm(**kwargs)
    if result != {"FINISHED"}:
        raise ConversionError("export-failed", f"bpy.ops.export_scene.vrm did not finish: {sorted(result)}")
