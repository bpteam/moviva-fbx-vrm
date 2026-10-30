"""MUTATE for generated bones: finger rigs (and the shared bone / weight helpers used by ``hair_author``).

Every generated bone carries the ``Prosopon_`` prefix; existing bones and weights are only ever *reduced* on the
vertices a plan explicitly transfers. Not verified against a real Blender by the authors.
"""

from __future__ import annotations

from typing import Any

from ..core.authoring.fingers import FingerRigPlan, NewBone


def add_bones(armature_name: str, bones: list[NewBone], bpy: Any = None) -> set[str]:
    """Create ``bones`` (world-space head/tail, parents by name) in edit mode. Returns the created bone names."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    from mathutils import Vector  # type: ignore[import-not-found]

    obj = bpy.data.objects[armature_name]
    inverse = obj.matrix_world.inverted()
    created: set[str] = set()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit = obj.data.edit_bones
        for bone in bones:
            if bone.name in edit:
                raise ValueError(f"bone '{bone.name}' already exists; nothing is overwritten")
            eb = edit.new(bone.name)
            eb.head = inverse @ Vector(bone.head)
            eb.tail = inverse @ Vector(bone.tail)
            eb.use_deform = True
            parent = edit.get(bone.parent)
            if parent is None:
                raise LookupError(f"parent bone '{bone.parent}' of '{bone.name}' not found")
            eb.parent = parent
            eb.use_connect = False
            created.add(bone.name)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return created


def _weight(vertex: Any, group_index: int) -> float:
    return next((g.weight for g in vertex.groups if g.group == group_index), 0.0)


def transfer_weights(transfers: dict[str, dict[int, dict[str, float]]], source_group: str, bpy: Any = None) -> None:
    """Move a fraction of ``source_group`` weight on each listed vertex to new groups (fractions are of that weight)."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    for mesh_name, verts in transfers.items():
        obj = bpy.data.objects.get(mesh_name)
        if obj is None:
            raise LookupError(f"mesh object '{mesh_name}' not found")
        source = obj.vertex_groups.get(source_group)
        if source is None:
            raise LookupError(f"'{mesh_name}' has no vertex group '{source_group}'")
        for vertex_index, fractions in verts.items():
            vertex = obj.data.vertices[vertex_index]
            total = _weight(vertex, source.index)
            if total <= 0.0:
                continue
            moved = 0.0
            for group_name, fraction in fractions.items():
                group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
                group.add([vertex_index], total * fraction, "REPLACE")
                moved += fraction
            source.add([vertex_index], total * max(0.0, 1.0 - moved), "REPLACE")


def apply_finger_rig(plan: FingerRigPlan, armature_name: str, bpy: Any = None) -> set[str]:
    created = add_bones(armature_name, plan.bones, bpy)
    transfer_weights(plan.transfers, plan.hand_bone, bpy)
    return created
