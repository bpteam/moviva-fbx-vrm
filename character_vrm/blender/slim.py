"""MUTATE before export: drops what a VRM app never uses and caps texture size (file size and video memory).

A Character Creator model carries ~270 shape keys on every face mesh; VRM apps reach shape keys only through the
expressions, which bind a few dozen. The rest were ~70% of a converted file.
"""

from __future__ import annotations

from typing import Any

from ..core.model import VrmPlan

MAX_TEXTURE = 1024  # px, longest side: 2048 -> 1024 is a quarter of the video memory, still sharp at call size
# Moviva drives these VRoid face morphs by name when no expression binds them (avatar/src/avatar/AvatarLoader.ts).
KEEP_SUFFIXES = ("Fcl_BRW_Joy", "Fcl_BRW_Angry", "Fcl_BRW_Sorrow", "Fcl_BRW_Surprised", "Fcl_EYE_Joy",
                 "Fcl_EYE_Angry", "Fcl_EYE_Sorrow", "Fcl_EYE_Surprised", "Fcl_MTH_Fun")


def slim(plan: VrmPlan, bpy: Any = None) -> tuple[int, int]:
    """Returns (shape keys removed, textures scaled down)."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    return drop_unbound_shape_keys(plan, bpy.data.objects), cap_textures(bpy.data.images)


def drop_unbound_shape_keys(plan: VrmPlan, objects: Any) -> int:
    bound = {(b.mesh, b.shape_key) for mapping in plan.expressions.values() for b in mapping.binds}
    removed = 0
    for obj in objects:
        keys = obj.data.shape_keys if obj.type == "MESH" else None
        if keys is None:
            continue
        for block in list(keys.key_blocks):
            if block == keys.reference_key or (obj.name, block.name) in bound or block.name.endswith(KEEP_SUFFIXES):
                continue
            obj.shape_key_remove(block)
            removed += 1
    return removed


def cap_textures(images: Any, limit: int = MAX_TEXTURE) -> int:
    scaled = 0
    for image in images:
        width, height = image.size
        if max(width, height) <= limit:
            continue
        factor = limit / max(width, height)
        image.scale(max(1, round(width * factor)), max(1, round(height * factor)))
        scaled += 1
    return scaled
