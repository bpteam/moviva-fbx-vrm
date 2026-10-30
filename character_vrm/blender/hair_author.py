"""MUTATE for generated hair bones (``--generate-hair-rig``). Not verified against a real Blender by the authors."""

from __future__ import annotations

from typing import Any

from ..core.authoring.hair_rig import HairRigPlan
from .rig_author import add_bones


def apply_hair_rig(plan: HairRigPlan, armature_name: str, bpy: Any = None) -> set[str]:
    """Create the chains' bones and give each hair vertex its planned weights (the head-bone share is replaced only on
    the planned vertices)."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    created = add_bones(armature_name, [b for chain in plan.chains for b in chain.bones], bpy)
    for mesh_name, verts in plan.weights.items():
        obj = bpy.data.objects.get(mesh_name)
        if obj is None:
            raise LookupError(f"mesh object '{mesh_name}' not found")
        head = obj.vertex_groups.get(plan.parent_bone)
        for vertex_index, weights in verts.items():
            if head is not None:
                head.remove([vertex_index])
            for group_name, weight in weights.items():
                group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
                group.add([vertex_index], weight, "REPLACE")
    return created
