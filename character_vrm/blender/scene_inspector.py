"""Stage 3: turn the live Blender scene into a ``CharacterInspection`` (no ``bpy`` types leak out)."""

from __future__ import annotations

from typing import Any

from ..core.model import (
    ArmatureInfo, BoneInfo, CharacterInspection, GeometryOptimizationPlan, ImageInfo, MaterialInfo, MeshInfo,
    ShapeKeyInfo,
)

_PROCEDURAL = frozenset({
    "TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_MAGIC", "TEX_GRADIENT", "TEX_BRICK", "TEX_CHECKER",
    "TEX_MUSGRAVE", "TEX_WHITE_NOISE", "TEX_SKY", "TEX_GABOR",
})
_CHANNELS = {  # Principled BSDF input -> channel label
    "Base Color": "baseColor", "Metallic": "metallic", "Roughness": "roughness",
    "Emission Color": "emissive", "Alpha": "alpha", "Normal": "normal",
}


def bound_armature(obj: Any) -> str | None:
    """Name of the armature object that deforms ``obj`` (armature modifier, else armature parent)."""
    for modifier in getattr(obj, "modifiers", []):
        if modifier.type == "ARMATURE" and modifier.object is not None:
            return str(modifier.object.name)
    parent = getattr(obj, "parent", None)
    if parent is not None and parent.type == "ARMATURE" and getattr(obj, "parent_type", "") in ("ARMATURE", "OBJECT", "BONE"):
        return str(parent.name)
    return None


def _analyze_material(mat: Any) -> MaterialInfo:
    info = MaterialInfo(name=mat.name)
    tree = getattr(mat, "node_tree", None)
    info.uses_nodes = bool(tree)
    if not tree:
        return info
    types = {n.type for n in tree.nodes}
    info.shader_types = sorted(t for t in types if t.startswith("BSDF") or t in ("EMISSION", "SUBSURFACE_SCATTERING"))
    info.procedural_nodes = sorted(types & _PROCEDURAL)
    info.procedural = bool(info.procedural_nodes)
    info.images = sorted({n.image.name for n in tree.nodes if n.type == "TEX_IMAGE" and getattr(n, "image", None)})

    principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is not None:
        for socket_name, label in _CHANNELS.items():
            socket = principled.inputs.get(socket_name)
            if socket is not None and socket.is_linked:
                info.channels.append(label)
        rough = principled.inputs.get("Roughness")
        if rough is not None and not rough.is_linked:
            info.roughness = float(rough.default_value)
        level, tint = principled.inputs.get("Specular IOR Level"), principled.inputs.get("Specular Tint")
        if level is not None and not level.is_linked:
            brightest = 1.0 if tint is None or tint.is_linked else max(tint.default_value[:3])
            info.specular = float(level.default_value) * brightest
        alpha = principled.inputs.get("Alpha")
        if alpha is not None and (alpha.is_linked or getattr(alpha, "default_value", 1.0) < 1.0):
            info.alpha_mode = "BLEND"
    # glTF importer keeps occlusion in a custom group node.
    for node in tree.nodes:
        group = getattr(node, "node_tree", None) if node.type == "GROUP" else None
        if group is not None and "glTF" in group.name:
            occlusion = node.inputs.get("Occlusion")
            if occlusion is not None and occlusion.is_linked:
                info.channels.append("occlusion")
    return info


def _block_coords(block: Any, count: int) -> list[float]:
    """Flat xyz coordinates of a shape key block (``foreach_get`` when available: it is fast)."""
    flat = [0.0] * (3 * count)
    data = getattr(block, "data", None)
    if data is None:
        return flat
    try:
        data.foreach_get("co", flat)
    except (AttributeError, TypeError, RuntimeError):
        for i, point in enumerate(data):
            co = point.co
            flat[3 * i:3 * i + 3] = [float(co[0]), float(co[1]), float(co[2])]
    return flat


def _block_deltas(block: Any, basis: list[float], world: Any, epsilon: float = 1e-7) -> dict[int, tuple]:
    """Sparse world-space offsets of a shape key from the basis (what the anatomy analysis reads)."""
    coords = _block_coords(block, len(basis) // 3)
    local: dict[int, tuple] = {}
    for i in range(len(basis) // 3):
        d = (coords[3 * i] - basis[3 * i], coords[3 * i + 1] - basis[3 * i + 1], coords[3 * i + 2] - basis[3 * i + 2])
        if abs(d[0]) + abs(d[1]) + abs(d[2]) > epsilon:
            local[i] = d
    if not local or world is None:
        return local
    try:
        linear = world.to_3x3()
        out = {}
        for i, d in local.items():
            w = linear @ _vector(d)
            out[i] = (float(w[0]), float(w[1]), float(w[2]))
        return out
    except (AttributeError, TypeError, ValueError, ImportError):
        return local


def _vector(d: tuple) -> Any:
    from mathutils import Vector  # Blender only
    return Vector(d)


def inspect_scene(source_path: str, fmt: str, bpy: Any = None) -> CharacterInspection:
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    result = CharacterInspection(source_path=source_path, source_format=fmt)
    data = bpy.data

    for obj in data.objects:
        if obj.type != "ARMATURE":
            continue
        world = obj.matrix_world
        bones = [
            BoneInfo(
                name=b.name, parent=b.parent.name if b.parent else None,
                head=tuple(world @ b.head_local), tail=tuple(world @ b.tail_local),  # type: ignore[arg-type]
                deform=bool(b.use_deform),
            )
            for b in obj.data.bones
        ]
        result.armatures.append(ArmatureInfo(obj.name, bones))

    used_images: set[str] = set()
    for obj in data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        group_names = [g.name for g in obj.vertex_groups]
        positions: list[tuple[float, float, float]] = []
        world = getattr(obj, "matrix_world", None)
        for vertex in mesh.vertices:
            co = getattr(vertex, "co", None)
            try:
                transformed = world @ co if world is not None and co is not None else co
                value = tuple(float(c) for c in transformed)
            except (TypeError, ValueError):
                value = ()
            positions.append(value if len(value) == 3 else (0.0, 0.0, 0.0))
        polygons = [tuple(int(i) for i in p.vertices) for p in mesh.polygons]
        polygon_materials = [int(getattr(p, "material_index", 0)) for p in mesh.polygons]
        weights: dict[int, dict[str, float]] = {}
        for vertex_index, vertex in enumerate(mesh.vertices):
            influences: dict[str, float] = {}
            for group in getattr(vertex, "groups", []):
                if 0 <= int(group.group) < len(group_names):
                    influences[group_names[int(group.group)]] = float(group.weight)
            if influences:
                weights[int(getattr(vertex, "index", vertex_index))] = influences
        parent_bone = str(getattr(obj, "parent_bone", "")) if getattr(obj, "parent_type", "") == "BONE" else None
        attached = [parent_bone] if parent_bone else []
        for constraint in getattr(obj, "constraints", []):
            target = getattr(constraint, "subtarget", None)
            if target:
                attached.append(str(target))
        result.meshes.append(MeshInfo(
            name=obj.name, vertices=len(mesh.vertices),
            triangles=sum(len(p.vertices) - 2 for p in mesh.polygons),
            armature=bound_armature(obj),
            vertex_groups=group_names,
            materials=[s.material.name for s in obj.material_slots if s.material],
            vertex_positions=positions, polygons=polygons, polygon_materials=polygon_materials,
            vertex_weights=weights,
            parent_bone=parent_bone, attached_bones=attached,
        ))
        keys = getattr(mesh, "shape_keys", None)
        if keys is not None:
            reference = keys.reference_key
            basis = _block_coords(reference, len(mesh.vertices))
            for block in keys.key_blocks:
                if block == reference:
                    continue
                result.shape_keys.append(ShapeKeyInfo(obj.name, block.name, block.slider_min, block.slider_max,
                                                      deltas=_block_deltas(block, basis, world)))

    referenced = {s.material.name for o in data.objects if o.type == "MESH" for s in o.material_slots if s.material}
    for mat in data.materials:
        if mat.name in referenced:
            info = _analyze_material(mat)
            used_images.update(info.images)
            result.materials.append(info)

    for image in data.images:
        if image.name in used_images:
            width, height = image.size
            result.images.append(ImageInfo(image.name, int(width), int(height), bool(image.packed_file), image.filepath))
    try:
        from . import face_detect
        result.face_points = face_detect.detect(bpy)
    except Exception:  # optional evidence: the geometry analysis stands on its own
        result.face_points = {}
    return result


def isolate_armature(armature: str, bpy: Any = None) -> list[str]:
    """Hide (and thereby exclude from the VRM Add-on's export) every other armature and the meshes it
    deforms. Meshes bound to the primary armature, and unskinned meshes, stay untouched."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]

    hidden: list[str] = []
    for obj in bpy.data.objects:
        other = (obj.type == "ARMATURE" and obj.name != armature) or (
            obj.type == "MESH" and bound_armature(obj) not in (None, armature))
        if other:
            obj.hide_set(True)
            obj.hide_viewport = True
            hidden.append(obj.name)
    return hidden


def apply_geometry_plan(plan: GeometryOptimizationPlan, bpy: Any = None) -> None:
    """Apply polygon deletions while leaving vertex/shape-key topology aligned.

    Only faces are removed; vertices are intentionally retained.  This keeps
    Basis and every shape key on the same vertex topology, while unreferenced
    vertices are harmless to the glTF export.  All dependency cleanup is
    deferred until after this pass and is conservative by design.
    """
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    import bmesh

    for name in plan.delete_objects:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    for name, polygon_indices in plan.remove_polygons.items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            continue
        mesh = obj.data
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            faces = list(bm.faces)
            doomed = [faces[index] for index in polygon_indices if 0 <= index < len(faces)]
            if doomed:
                bmesh.ops.delete(bm, geom=doomed, context="FACES_ONLY")
            bm.to_mesh(mesh)
            mesh.update()
        finally:
            bm.free()

    # Never remove materials or images here.  A shared skin atlas can be used
    # by a surviving hand polygon even when all leg polygons disappeared.
