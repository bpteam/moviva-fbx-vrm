"""Stage 1 (clean scene) and stage 2 (import). ``bpy`` is imported inside the functions."""

from __future__ import annotations

from typing import Any

from ..core.errors import ConversionError

# Every datablock collection that can hold character data. Removed programmatically, no UI context needed.
_DATA_COLLECTIONS = (
    "objects", "meshes", "armatures", "materials", "images", "actions", "textures", "cameras", "lights",
    "collections", "node_groups", "shape_keys", "curves", "grease_pencils", "worlds",
)


def reset_scene(bpy: Any = None) -> None:
    """Leave no startup cube, selection or previous import behind."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except (RuntimeError, AttributeError):
        pass  # no active object / no context in background mode: nothing to leave

    data = bpy.data
    for obj in list(data.objects):
        data.objects.remove(obj, do_unlink=True)
    for name in _DATA_COLLECTIONS:
        collection = getattr(data, name, None)
        if collection is None:
            continue
        for block in list(collection):
            try:
                collection.remove(block)
            except (RuntimeError, TypeError, ReferenceError):
                pass  # in use by the scene itself (e.g. the master collection, a world): harmless
    for block in list(data.objects):
        raise ConversionError("scene-not-clean", f"Could not clear object '{block.name}' from the scene.")


def import_character(path: str, fmt: str, bpy: Any = None) -> None:
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    if fmt in ("glb", "gltf"):
        operator = getattr(bpy.ops.import_scene, "gltf", None)
        if operator is None:
            raise ConversionError("importer-missing", "bpy.ops.import_scene.gltf is unavailable (glTF add-on disabled?).")
        result = operator(filepath=path)
    elif fmt == "fbx":
        # Blender 5.x's current FBX importer. Facial shape keys are verified after import, not assumed.
        operator = getattr(bpy.ops.wm, "fbx_import", None)
        if operator is None:
            raise ConversionError("importer-missing", "bpy.ops.wm.fbx_import is unavailable in this Blender build.")
        result = operator(filepath=path)
    else:  # pragma: no cover - source_format() rejects this earlier
        raise ConversionError("unsupported-format", f"Unsupported format {fmt!r}")

    if result != {"FINISHED"}:
        raise ConversionError("import-failed", f"Blender import of {path} returned {sorted(result)}.")
