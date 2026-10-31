"""Autorig Workbench (GPL-3.0, vendored at build time by ``packaging/fetch_autorig.py``) as the body auto-rigger.

The workbench fits Mesh2Motion's human skeleton (with fingers) to a mesh and skins it. It works on its own welded
copy of the mesh in a fresh Blender scene, so it runs as a child of this same program (``aw-rig``; the same
bundled ``bpy``, no second Blender), and only its skeleton and weights come back: the armature is moved into this
scene's frame and the weights are carried onto the untouched source meshes by nearest surface point, so the
materials, UVs and shape keys stay the source's own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Mesh2Motion bone -> VRM humanoid name; the converter's resolver then maps them by name.
RENAME = {"pelvis": "Hips", "spine_01": "Spine", "spine_02": "Chest", "spine_03": "UpperChest", "neck_01": "Neck",
          "head": "Head"}
for _s, _S in (("l", "Left"), ("r", "Right")):
    RENAME.update({f"clavicle_{_s}": f"{_S}Shoulder", f"upperarm_{_s}": f"{_S}UpperArm",
                   f"lowerarm_{_s}": f"{_S}LowerArm", f"hand_{_s}": f"{_S}Hand", f"thigh_{_s}": f"{_S}UpperLeg",
                   f"calf_{_s}": f"{_S}LowerLeg", f"foot_{_s}": f"{_S}Foot", f"ball_{_s}": f"{_S}Toes",
                   f"thumb_01_{_s}": f"{_S}ThumbMetacarpal", f"thumb_02_{_s}": f"{_S}ThumbProximal",
                   f"thumb_03_{_s}": f"{_S}ThumbDistal"})
    for _f, _F in (("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("pinky", "Little")):
        RENAME.update({f"{_f}_01_{_s}": f"{_S}{_F}Proximal", f"{_f}_02_{_s}": f"{_S}{_F}Intermediate",
                       f"{_f}_03_{_s}": f"{_S}{_F}Distal"})

TIMEOUT = 900
# m2m_rig turns the mesh so its head points down -Y: the yaw it applied for each facing it reports.
YAW = {"+x": -np.pi / 2, "-x": np.pi / 2, "+y": np.pi, "-y": 0.0}


def root() -> Path | None:
    """The vendored workbench: $MOVIVA_AUTORIG, the frozen app's vendor/, or the checkout's vendor/."""
    candidates = [os.environ.get("MOVIVA_AUTORIG", "")]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(getattr(sys, "_MEIPASS", ""), "vendor", "autorig"))
    candidates.append(str(Path(__file__).resolve().parents[2] / "vendor" / "autorig"))
    for c in candidates:
        if c and (Path(c) / "autorig" / "steps" / "m2m_rig.py").is_file():
            return Path(c)
    return None


def command(args: list[str]) -> list[str]:
    """This program as the workbench's runner: ``<app> aw-rig …`` frozen; inside a Blender executable (the
    ``character-vrm`` CLI) that same Blender; else this module under the Python that has ``bpy``."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "aw-rig", *args]
    import bpy
    if bpy.app.binary_path:
        return [bpy.app.binary_path, "-b", "--factory-startup", "--python", __file__, "--", *args]
    return [sys.executable, "-m", "character_vrm.blender.workbench", *args]


def main(argv: list[str]) -> int:
    """Run m2m_rig.py in this process (it needs ``bpy`` importable and exits through ``os._exit``), its arm
    measurement replaced by ours where ours finds both arms."""
    import runpy
    base = root()
    if base is None:
        print('M2M_RIG {"error": "Autorig Workbench is not installed"}', flush=True)
        return 2
    import bpy  # noqa: F401  (load before the script so its own import finds the module)
    steps = base / "autorig" / "steps"
    sys.path[:0] = [str(steps), str(base / "autorig" / "core")]
    import rerig_humanoid
    rerig_humanoid.measure = _measure_with_arms(rerig_humanoid.measure)
    script = steps / "m2m_rig.py"
    sys.argv = [str(script), "--", *argv]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _measure_with_arms(measure):
    """rerig_humanoid.measure, with the shoulders, elbows, wrists and knuckles from ``core.limbs.find_arms``: the
    workbench takes the widest part of the body for the arms, which is the hair on a bob or arms hanging at the
    sides. Its fingers follow the wrist-to-knuckle line, so they move with ours."""
    def measured(co, lo, size, h, main=None):
        J = measure(co, lo, size, h, main)
        try:
            import bpy
            from mathutils import Vector
            from character_vrm.core.limbs import find_arms, surface_samples
            mesh = max((o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("Mannequin")),
                       key=lambda o: len(o.data.vertices))
            me = mesh.data
            me.calc_loop_triangles()
            verts = np.array([v.co[:] for v in me.vertices])
            tris = np.array([t.vertices[:] for t in me.loop_triangles])
            edges = np.array([e.vertices[:] for e in me.edges])
            arms = find_arms(surface_samples(verts[tris]), verts, edges)
        except Exception as exc:  # keep the workbench's own arms
            print("MOVIVA_ARMS " + json.dumps({"error": str(exc)[:200]}), flush=True)
            return J
        if arms is None:
            print('MOVIVA_ARMS {"found": false}', flush=True)
            return J
        for side, a in arms.items():
            for k in ("shoulder", "elbow", "wrist", "knuckle"):
                J[side + k] = Vector(a[k].tolist())
        print("MOVIVA_ARMS " + json.dumps({"found": True, **{s: {k: np.round(v, 4).tolist() for k, v in a.items()
                                                                if k != "length"} for s, a in arms.items()}}), flush=True)
        return J
    return measured


def _export_meshes(bpy: Any, meshes: list[Any], path: str) -> None:
    for o in bpy.data.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_animations=False,
                              export_skins=False, export_morph=False, export_materials="NONE", export_yup=True)


def _rigid(points: np.ndarray, facing: str) -> np.ndarray:
    """The 4x4 m2m_rig applied: the yaw for ``facing``, then centred over the origin with the feet on z = 0."""
    a = YAW[facing]
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    p = points @ R.T
    lo, hi = p.min(axis=0), p.max(axis=0)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
    return M


def fit(bpy: Any, meshes: list[Any], points: np.ndarray, log: Any = print) -> tuple[Any, list[str], np.ndarray, dict]:
    """Rig ``meshes`` (already baked to world space) with the workbench. Returns the armature object in this scene,
    its deform bone names, per-vertex weights (rows in ``points`` order) and the workbench's own summary."""
    if root() is None:
        raise RuntimeError("Autorig Workbench is not installed")
    with tempfile.TemporaryDirectory(prefix="moviva-aw-") as tmp:
        src = os.path.join(tmp, "body.glb")
        _export_meshes(bpy, meshes, src)
        out = os.path.join(tmp, "out")
        args = ["--in", src, "--name", "body", "--out", out, "--template", "human", "--format", "glb",
                "--clips", "Idle", "--triangles", "20000", "--no-kind-height", "--no-audit", "--no-preview",
                "--m2m", str(root() / "mesh2motion")]
        proc = subprocess.run(command(args), capture_output=True, text=True, errors="replace",
                              stdin=subprocess.DEVNULL, timeout=TIMEOUT)
        summary: dict = {}
        arms = None
        for line in proc.stdout.splitlines():
            if line.startswith("M2M_RIG "):
                summary = json.loads(line[8:])
            elif line.startswith("MOVIVA_ARMS "):
                arms = json.loads(line[12:])
        summary["arms"] = arms
        if summary.get("error") or not os.path.isfile(os.path.join(out, "body.glb")):
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
            raise RuntimeError(summary.get("error") or f"workbench exit {proc.returncode}: {' | '.join(tail)}")
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(out, "body.glb"))
    new = [o for o in bpy.data.objects if o not in before]
    try:
        return _adopt(bpy, new, points, summary, log)
    except Exception:
        for o in new:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except ReferenceError:
                pass
        raise


def _adopt(bpy: Any, new: list[Any], points: np.ndarray, summary: dict, log: Any):
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.interpolate import poly_3d_calc
    arm = next(o for o in new if o.type == "ARMATURE")
    rig_mesh = next(o for o in new if o.type == "MESH")
    for o in new:
        o.animation_data_clear()
    for act in [a for a in bpy.data.actions if a.users == 0]:
        bpy.data.actions.remove(act)

    back = Matrix(np.linalg.inv(_rigid(points, summary.get("facing", "-y"))).tolist())
    for o in new:
        if o.parent is None:
            o.matrix_world = back @ o.matrix_world
    bpy.context.view_layer.update()

    # the workbench mesh's weights, in world space, interpolated onto every source vertex
    for b in arm.data.bones:
        if b.name in RENAME:
            b.name = RENAME[b.name]  # vertex groups follow a bone rename
    for b in arm.data.bones:  # Mesh2Motion's end markers and root carry no skin
        if b.name == "root" or "leaf" in b.name:
            b.use_deform = False
    names = [b.name for b in arm.data.bones if b.use_deform]
    col = {n: i for i, n in enumerate(names)}
    mw = rig_mesh.matrix_world
    verts = [mw @ v.co for v in rig_mesh.data.vertices]
    polys = [tuple(p.vertices) for p in rig_mesh.data.polygons]
    src_W = np.zeros((len(verts), len(names)))
    gi = {g.index: col.get(g.name) for g in rig_mesh.vertex_groups}
    for v in rig_mesh.data.vertices:
        for g in v.groups:
            j = gi.get(g.group)
            if j is not None:
                src_W[v.index, j] = g.weight
    tree = BVHTree.FromPolygons(verts, polys)
    W = np.zeros((len(points), len(names)))
    for i, p in enumerate(points):
        loc, _, face, _ = tree.find_nearest(Vector(p))
        if face is None:
            continue
        idx = polys[face]
        bary = poly_3d_calc([verts[k] for k in idx], loc)
        W[i] = np.asarray(bary) @ src_W[list(idx)]
    for o in new:
        if o is not arm:
            bpy.data.objects.remove(o, do_unlink=True)

    bpy.context.view_layer.objects.active = arm
    for o in bpy.data.objects:
        o.select_set(o == arm)
    arm.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    log(f"workbench: {len(names)} bones, facing {summary.get('facing')}, {summary.get('seconds')} s")
    return arm, names, W, summary


if __name__ == "__main__":
    if "--" in sys.argv:  # inside a Blender executable: blender -b --python workbench.py -- <args>
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:]))
    raise SystemExit(main(sys.argv[1:]))
