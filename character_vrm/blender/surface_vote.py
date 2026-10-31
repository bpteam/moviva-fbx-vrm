"""MediaPipe multi-class selfie segmentation (Apache-2.0) voted over each material's pixels on rendered views.

For materials whose name says nothing (``mesh_4_7.nr``): the character stands with its arms down (the segmenter
was trained on people, not T-poses), is rendered from the front, three-quarters and the back (the side view
confuses hair and face), and every sampled pixel casts a ray to learn which material it shows; the material gets
the class most of its pixels were segmented as. The pose is put back afterwards.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

SIZE = 384
STEP = 3  # one ray every few pixels


def model_path() -> Path | None:
    from .face_detect import model_path as face_model
    face = face_model()
    path = face.with_name("selfie_multiclass.tflite") if face else None
    return path if path and path.is_file() else None


def vote(bpy: Any, armature_name: str, materials: set[str]) -> dict:
    """{"materials": {material: class}, "points": [[x, y, z, class], ...]} for ``materials``; {} without MediaPipe.

    A material whose pixels agree (80 % or more) gets one class. One that mixes several (a whole character on one
    atlas texture) gets none; its labelled points (rest-pose glTF positions of the vertices the rays hit) decide
    vertex by vertex."""
    from .face_detect import _can_render
    path = model_path()
    if not materials or path is None or not _can_render():
        return {}
    for name in ("cv2", "matplotlib", "matplotlib.pyplot"):
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from mathutils import Euler, Quaternion, Vector
    except ImportError:
        return {}
    from ..core.surface_labels import class_from_votes
    from . import compat_v4_7_2 as addon

    arm = bpy.data.objects.get(armature_name)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.visible_get()]
    if arm is None or not meshes:
        return {}
    saved_pose = {pb.name: (pb.rotation_mode, pb.rotation_quaternion.copy()) for pb in arm.pose.bones}
    for side, sign in (("left", 1), ("right", -1)):  # arms down along the body
        bone = addon.humanoid_bone(arm, f"{side}UpperArm")
        pb = arm.pose.bones.get(bone) if bone else None
        if pb is None:
            continue
        pb.rotation_mode = "QUATERNION"
        axis = ((arm.matrix_world @ pb.bone.matrix_local).to_3x3().inverted() @ Vector((0, 1, 0))).normalized()
        pb.rotation_quaternion = Quaternion(axis, math.radians(sign * 68)) @ pb.rotation_quaternion
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    saved = (scene.camera, scene.render.engine, scene.render.resolution_x, scene.render.resolution_y,
             scene.render.filepath)
    cam = bpy.data.objects.new("moviva_surface_cam", bpy.data.cameras.new("moviva_surface_cam"))
    scene.collection.objects.link(cam)
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    height = hi.z - lo.z
    span = 1.05 * max(height, hi.x - lo.x)
    cam.data.type, cam.data.ortho_scale = "ORTHO", span
    scene.camera = cam
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light, scene.display.shading.color_type = "STUDIO", "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    counts: dict[str, dict[int, int]] = {m: {} for m in materials}
    hits: dict[tuple[str, int], dict[int, int]] = {}  # (object, vertex) -> category counts
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(path)), output_category_mask=True))
    try:
        centre = (lo + hi) / 2
        graph = bpy.context.evaluated_depsgraph_get()
        with tempfile.TemporaryDirectory(prefix="moviva-surface-") as tmp:
            for yaw in (0.0, -math.pi / 4, math.pi):  # front, three-quarters, back
                d = 3 * height
                cam.location = centre + Vector((d * math.sin(yaw), -d * math.cos(yaw), 0))
                cam.rotation_euler = Euler((math.pi / 2, 0, yaw))
                bpy.context.view_layer.update()
                scene.render.filepath = os.path.join(tmp, "view.png")
                bpy.ops.render.render(write_still=True)
                img = bpy.data.images.load(scene.render.filepath)
                px = np.array(img.pixels[:], np.float32).reshape(SIZE, SIZE, 4)[::-1, :, :3]
                bpy.data.images.remove(img)
                rgb = np.ascontiguousarray((px.clip(0, 1) * 255).astype(np.uint8))
                mask = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).category_mask.numpy_view()
                mask = mask.reshape(SIZE, SIZE)
                rot = cam.matrix_world.to_3x3()
                right, up, ahead = rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0)), rot @ Vector((0, 0, -1))
                for y in range(0, SIZE, STEP):
                    for x in range(0, SIZE, STEP):
                        origin = cam.location + right * ((x + 0.5) / SIZE - 0.5) * span + up * (
                            0.5 - (y + 0.5) / SIZE) * span
                        hit, _, _, face, obj, _ = scene.ray_cast(graph, origin, ahead)
                        if not hit or obj.type != "MESH":
                            continue
                        slot = obj.data.polygons[face].material_index if face < len(obj.data.polygons) else -1
                        mat = obj.material_slots[slot].material if 0 <= slot < len(obj.material_slots) else None
                        if mat is not None and mat.name in counts:
                            k = int(mask[y, x])
                            counts[mat.name][k] = counts[mat.name].get(k, 0) + 1
                            for v in obj.data.polygons[face].vertices:
                                c = hits.setdefault((obj.name, v), {})
                                c[k] = c.get(k, 0) + 1
    finally:
        seg.close()
        scene.camera, scene.render.engine, scene.render.resolution_x, scene.render.resolution_y, \
            scene.render.filepath = saved
        data = cam.data
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.data.cameras.remove(data)
        for pb in arm.pose.bones:
            mode, quat = saved_pose[pb.name]
            pb.rotation_mode, pb.rotation_quaternion = mode, quat
        bpy.context.view_layer.update()
    agreed, mixed = {}, set()
    for m, v in counts.items():
        cls = class_from_votes(v)
        total = sum(n for k, n in v.items() if k)
        if cls and total and max(n for k, n in v.items() if k) >= 0.8 * total:
            agreed[m] = cls
        elif cls:
            mixed.add(m)
    points = []
    for (obj_name, v), c in hits.items():
        obj = bpy.data.objects[obj_name]
        cls = class_from_votes(c, min_pixels=1)
        if cls is None:
            continue
        p = obj.matrix_world @ obj.data.vertices[v].co
        points.append([p.x, p.z, -p.y, cls])  # glTF axes
    return {"materials": agreed, "points": points if mixed else []}
