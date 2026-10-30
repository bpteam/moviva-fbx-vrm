"""MUTATE + VALIDATE RESULT for the real-time material look (``core/authoring/materials.py``).

Every edit is expressed the way Blender's glTF exporter reads a Principled BSDF, so the exported file carries it:

- roughness map: a Non-Color image into ``Roughness`` (packed into ``metallicRoughnessTexture``), at most
  ``MAX_MAP_SIZE`` pixels on a side (both maps);
- ambient occlusion: a Non-Color image into the ``Occlusion`` input of the ``glTF Material Output`` node group;
- specular: ``Specular IOR Level`` and a white ``Specular Tint`` (``KHR_materials_specular``, default F0 at 0.5);
- alpha clip: ``Alpha -> [Math: X < cutoff] -> [Math: 1 - X] -> Alpha``, the exporter's ``MASK`` pattern.

A socket that is already linked is never re-wired (the source's own map wins) except the alpha, which is only
wrapped. Failures are recorded per material, never swallowed. Checked against Blender 5.2.2's bundled exporter; see ``tests/integration/README.md``.
"""

from __future__ import annotations

from typing import Any

from ..core.authoring.materials import MAX_MAP_SIZE, MaterialEdit

GLTF_GROUP = "glTF Material Output"  # the exporter's settings node group (io_scene_gltf2 material_helpers)
_TAG = "prosopon"                     # label of every node this module adds


def apply_materials(edits: list[MaterialEdit], bpy: Any = None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Returns ``(done: material -> changes made, failed: material -> reason)``."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    done: dict[str, list[str]] = {}
    failed: dict[str, str] = {}
    for edit in edits:
        try:
            changes = _apply(edit, bpy)
            problem = _validate(edit, changes, bpy)
            if problem:
                failed[f"material {edit.material}"] = problem
            elif changes:
                done[edit.material] = changes
        except Exception as exc:  # noqa: BLE001 - one bad material must not abort the conversion
            failed[f"material {edit.material}"] = str(exc)
    return done, failed


def _principled(mat: Any) -> Any:
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _image_node(tree: Any, path: str, bpy: Any, x: float, y: float) -> Any:
    image = bpy.data.images.load(path, check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    w, h = image.size
    if max(w, h) > MAX_MAP_SIZE:
        k = MAX_MAP_SIZE / max(w, h)
        image.scale(max(1, round(w * k)), max(1, round(h * k)))
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image, node.label, node.location = image, _TAG, (x, y)
    return node


def _gltf_group(bpy: Any) -> Any:
    group = bpy.data.node_groups.get(GLTF_GROUP)
    if group is None:
        group = bpy.data.node_groups.new(GLTF_GROUP, "ShaderNodeTree")
        group.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        group.nodes.new("NodeGroupInput")
        group.nodes.new("NodeGroupOutput")
    return group


def _apply(edit: MaterialEdit, bpy: Any) -> list[str]:
    mat = bpy.data.materials.get(edit.material)
    bsdf = _principled(mat)
    if bsdf is None:
        raise LookupError("no Principled BSDF")
    tree, links = mat.node_tree, mat.node_tree.links
    x, y = bsdf.location.x - 600, bsdf.location.y
    changes: list[str] = []

    rough = bsdf.inputs["Roughness"]
    if not rough.is_linked:
        if "roughness" in edit.textures:
            links.new(_image_node(tree, edit.textures["roughness"], bpy, x, y - 300).outputs["Color"], rough)
            changes.append("roughness map")
        elif edit.roughness is not None:
            rough.default_value = edit.roughness
            changes.append(f"roughness {edit.roughness}")

    if edit.specular is not None:
        bsdf.inputs["Specular IOR Level"].default_value = edit.specular
        tint = bsdf.inputs["Specular Tint"]
        if not tint.is_linked:
            tint.default_value = (1.0, 1.0, 1.0, 1.0)
        changes.append(f"specular {edit.specular}")

    if "ao" in edit.textures and not any(n.type == "GROUP" and n.node_tree is not None
                                         and n.node_tree.name == GLTF_GROUP for n in tree.nodes):
        group = tree.nodes.new("ShaderNodeGroup")
        group.node_tree, group.label, group.location = _gltf_group(bpy), _TAG, (bsdf.location.x + 300, y - 400)
        links.new(_image_node(tree, edit.textures["ao"], bpy, x, y - 600).outputs["Color"], group.inputs["Occlusion"])
        changes.append("ao map")

    alpha = bsdf.inputs["Alpha"]
    if edit.alpha_cutoff is not None and alpha.is_linked:
        source = alpha.links[0].from_socket
        links.remove(alpha.links[0])
        less = tree.nodes.new("ShaderNodeMath")
        less.operation, less.label, less.location = "LESS_THAN", _TAG, (x + 300, y - 150)
        less.inputs[1].default_value = edit.alpha_cutoff
        one_minus = tree.nodes.new("ShaderNodeMath")
        one_minus.operation, one_minus.label, one_minus.location = "SUBTRACT", _TAG, (x + 450, y - 150)
        one_minus.inputs[0].default_value = 1.0
        links.new(source, less.inputs[0])
        links.new(less.outputs[0], one_minus.inputs[1])
        links.new(one_minus.outputs[0], alpha)
        changes.append(f"alpha clip {edit.alpha_cutoff}")
    return changes


def _validate(edit: MaterialEdit, changes: list[str], bpy: Any) -> str | None:
    """VALIDATE RESULT: every reported change is present in the material."""
    bsdf = _principled(bpy.data.materials.get(edit.material))
    if bsdf is None:
        return "Principled BSDF disappeared"
    if "roughness map" in changes and not bsdf.inputs["Roughness"].is_linked:
        return "roughness map not linked"
    if any(c.startswith("alpha clip") for c in changes):
        node = bsdf.inputs["Alpha"].links[0].from_node if bsdf.inputs["Alpha"].is_linked else None
        if node is None or node.type != "MATH" or node.operation != "SUBTRACT":
            return "alpha clip not in place"
    if edit.specular is not None and abs(bsdf.inputs["Specular IOR Level"].default_value - edit.specular) > 1e-6:
        return "specular not set"
    return None
