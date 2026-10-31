"""MediaPipe Face Landmarker (Apache-2.0) on a rendered view of the head, its points projected onto the mesh.

The head is in the top third of the scene, rendered orthographically from the front (-Y, then +Y when no face is
found: a glTF character faces -Y in Blender); each landmark's pixel is cast as a ray onto the meshes. The model file
is fetched at build time (``packaging/fetch_mediapipe.py``). Everything is optional: no MediaPipe, no model, no face
found, and the anatomy analyzer works from the geometry alone.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

# MediaPipe index -> face landmark name (the subject's left is the image's right)
POINTS = {
    "left_upper_eyelid": 386, "left_lower_eyelid": 374, "right_upper_eyelid": 159, "right_lower_eyelid": 145,
    "left_eye": 473, "right_eye": 468, "left_eyebrow": 334, "right_eyebrow": 105,
    "mouth_left": 291, "mouth_right": 61, "upper_lip": 13, "lower_lip": 14, "nose_tip": 1, "nose_bridge": 6,
    "chin": 152, "left_nostril": 327, "right_nostril": 98, "left_cheek": 425, "right_cheek": 205,
    "left_jaw": 397, "right_jaw": 172,
}
SIZE = 512


def model_path() -> Path | None:
    candidates = [os.environ.get("MOVIVA_FACE_MODEL", "")]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(getattr(sys, "_MEIPASS", ""), "vendor", "mediapipe", "face_landmarker.task"))
    candidates.append(str(Path(__file__).resolve().parents[2] / "vendor" / "mediapipe" / "face_landmarker.task"))
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


def _landmarker(path: Path):
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)), num_faces=1, min_face_detection_confidence=0.3))


def detect(bpy: Any) -> dict[str, tuple[float, float, float]]:
    """{landmark name: world position} for the face found, or {}."""
    if os.environ.get("MOVIVA_NO_FACE_RENDER"):  # a machine without OpenGL (CI's Windows runners) crashes rendering
        return {}
    path = model_path()
    if path is None:
        return {}
    for name in ("cv2", "matplotlib", "matplotlib.pyplot"):  # only its drawing helpers import them
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        import numpy as np
        import mediapipe as mp
        from mathutils import Euler, Vector
    except ImportError:
        return {}
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.visible_get()]
    if not meshes:
        return {}
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    height = hi.z - lo.z
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, hi.z - 0.15 * height))
    span = 0.3 * height  # the top of the figure: an adult's head is an eighth of it, a chibi's a third

    scene = bpy.context.scene
    saved = (scene.camera, scene.render.engine, scene.render.resolution_x, scene.render.resolution_y,
             scene.render.filepath)
    cam = bpy.data.objects.new("moviva_face_cam", bpy.data.cameras.new("moviva_face_cam"))
    scene.collection.objects.link(cam)
    cam.data.type, cam.data.ortho_scale = "ORTHO", span
    scene.camera = cam
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light, scene.display.shading.color_type = "STUDIO", "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    found: dict[str, tuple[float, float, float]] = {}
    lm = _landmarker(path)
    try:
        with tempfile.TemporaryDirectory(prefix="moviva-face-") as tmp:
            for yaw in (0.0, 3.14159265):  # from -Y, then from +Y
                d = 2 * height
                cam.location = centre + Vector((d * np.sin(yaw), -d * np.cos(yaw), 0))
                cam.rotation_euler = Euler((np.pi / 2, 0, yaw))
                bpy.context.view_layer.update()
                scene.render.filepath = os.path.join(tmp, f"face{int(yaw)}.png")
                bpy.ops.render.render(write_still=True)
                img = bpy.data.images.load(scene.render.filepath)
                px = np.array(img.pixels[:], np.float32).reshape(SIZE, SIZE, 4)[::-1, :, :3]
                bpy.data.images.remove(img)
                rgb = np.ascontiguousarray((px.clip(0, 1) * 255).astype(np.uint8))
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not res.face_landmarks:
                    continue
                pts = res.face_landmarks[0]
                right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
                up = cam.matrix_world.to_3x3() @ Vector((0, 1, 0))
                ahead = cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
                graph = bpy.context.evaluated_depsgraph_get()
                for name, i in POINTS.items():
                    p = pts[i]
                    origin = cam.location + right * ((p.x - 0.5) * span) + up * ((0.5 - p.y) * span)
                    hit, loc, *_ = scene.ray_cast(graph, origin, ahead)
                    if hit:
                        found[name] = tuple(loc)
                break
    finally:
        lm.close()
        scene.camera, scene.render.engine, scene.render.resolution_x, scene.render.resolution_y, \
            scene.render.filepath = saved
        data = cam.data
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.data.cameras.remove(data)
    return found
