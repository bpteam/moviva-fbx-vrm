"""Blender-side geometry evidence for generated face keys (BVH queries, not semantic detection).

The semantic work (landmarks, regions, deformation fields) is pure Python in ``core/anatomy`` and
``core/authoring``. This module only answers one question the pure core cannot: after a generated key is applied at
weight 1, does any moved skin vertex end up *inside* an eyeball? Not verified against real Blender by the authors.
"""

from __future__ import annotations

from typing import Any

from ..core.authoring.plans import ShapeKeyPlan


def eyelids_penetrate_eyeball(plan: ShapeKeyPlan, eye_objects: list[str], bpy: Any = None) -> list[str]:
    """Names of ``mesh#vertex`` whose deformed position lies inside one of the ``eye_objects`` (ray-parity test)."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    from mathutils import Vector  # type: ignore[import-not-found]
    from mathutils.bvhtree import BVHTree  # type: ignore[import-not-found]

    trees = []
    for name in eye_objects:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            trees.append(BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get()))
    inside: list[str] = []
    if not trees:
        return inside
    for mesh_name, offsets in plan.offsets.items():
        obj = bpy.data.objects.get(mesh_name)
        if obj is None:
            continue
        world = obj.matrix_world
        for index, delta in offsets.items():
            point = world @ obj.data.vertices[index].co + Vector(delta)
            for tree in trees:
                hits, direction = 0, Vector((0.0, 0.0, 1.0))
                origin = point.copy()
                for _ in range(16):
                    location, _n, _i, _d = tree.ray_cast(origin, direction)
                    if location is None:
                        break
                    hits += 1
                    origin = location + direction * 1e-6
                if hits % 2 == 1:
                    inside.append(f"{mesh_name}#{index}")
                    break
    return inside
