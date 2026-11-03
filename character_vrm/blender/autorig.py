"""Procedural auto-rig inside Blender: bake meshes to world space, fit a skeleton, skin, keep the best result.

Attempt 1 is Blender's bone-heat weighting (``ARMATURE_AUTO``), attempt 2 the deterministic distance weights from
``core.autorig``. Each result is read back from the real vertex groups and pose-tested numerically; the better
one is kept. Only runs on a scene with no armature, so a source rig is never replaced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.autorig import (HIERARCHY, STATUS_ORDER, AutoRigResult, better, deformation_tests, distance_weights, fit_skeleton)
from ..core.errors import ConversionError

MAX_INFLUENCES = 4


def _meshes(bpy: Any) -> list[Any]:
    return [o for o in bpy.data.objects if o.type == "MESH" and o.data.vertices]


def _bake_world(bpy: Any, meshes: list[Any]) -> None:
    """Move each mesh's world transform into its vertices and unparent it (empties carry the importer's rotation)."""
    from mathutils import Matrix
    for obj in meshes:
        me = obj.data
        if me.users > 1:
            obj.data = me = me.copy()
        me.transform(obj.matrix_world, shape_keys=True)
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
    for o in [o for o in bpy.data.objects if o.type == "EMPTY" and not o.children]:
        bpy.data.objects.remove(o, do_unlink=True)


def _points(meshes: list[Any]) -> tuple[np.ndarray, list[int]]:
    parts, sizes = [], []
    for obj in meshes:
        a = np.empty(len(obj.data.vertices) * 3, np.float64)
        obj.data.vertices.foreach_get("co", a)
        parts.append(a.reshape(-1, 3))
        sizes.append(len(parts[-1]))
    return np.vstack(parts), sizes


def _build_armature(bpy: Any, fit: Any) -> Any:
    from mathutils import Vector
    data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        eb = data.edit_bones
        up = Vector(tuple(fit.frame.up))
        for name, (head, tail) in fit.bones.items():
            b = eb.new(name)
            b.head, b.tail = Vector(tuple(head)), Vector(tuple(tail))
            if b.length < 1e-4:
                b.tail = b.head + up * 1e-2
        for name, parent in HIERARCHY.items():
            if parent:
                eb[name].parent = eb[parent]
                eb[name].use_connect = bool((eb[name].head - eb[parent].tail).length < 1e-6)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def _read_weights(meshes: list[Any], names: list[str]) -> np.ndarray:
    col = {n: i for i, n in enumerate(names)}
    total = sum(len(o.data.vertices) for o in meshes)
    W = np.zeros((total, len(names)))
    base = 0
    for obj in meshes:
        gi = {g.index: col[g.name] for g in obj.vertex_groups if g.name in col}
        for v in obj.data.vertices:
            for g in v.groups:
                j = gi.get(g.group)
                if j is not None:
                    W[base + v.index, j] = g.weight
        base += len(obj.data.vertices)
    return W


def _write_weights(meshes: list[Any], names: list[str], W: np.ndarray) -> None:
    base = 0
    for obj in meshes:
        obj.vertex_groups.clear()
        groups = [obj.vertex_groups.new(name=n) for n in names]
        n = len(obj.data.vertices)
        for j, g in enumerate(groups):
            for i in np.nonzero(W[base:base + n, j] > 1e-4)[0]:
                g.add([int(i)], float(W[base + i, j]), "REPLACE")
        base += n


def _bind(bpy: Any, meshes: list[Any], arm: Any) -> None:
    for obj in meshes:
        obj.parent = arm
        if not any(m.type == "ARMATURE" for m in obj.modifiers):
            mod = obj.modifiers.new("Armature", "ARMATURE")
            mod.object = arm


def _normalize_limit(W: np.ndarray) -> np.ndarray:
    """At most MAX_INFLUENCES per vertex, sum 1."""
    if W.shape[1] > MAX_INFLUENCES:
        cut = np.sort(W, axis=1)[:, -MAX_INFLUENCES][:, None]
        W = np.where(W >= cut, W, 0.0)
    s = W.sum(axis=1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def _bone_heat(bpy: Any, meshes: list[Any], arm: Any) -> None:
    for o in bpy.data.objects:
        o.select_set(False)
    for obj in meshes:
        obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def _workbench(bpy: Any, meshes: list[Any], pts: np.ndarray, result: AutoRigResult):
    """(test, W, method, armature, names) from Autorig Workbench, or None when it is missing or fails."""
    from . import workbench
    if workbench.root() is None:
        result.attempts.append({"method": "autorig-workbench", "status": "SKIPPED", "error": "not installed"})
        return None
    try:
        arm, names, W, summary = workbench.fit(bpy, meshes, pts, log=lambda *_: None)
    except Exception as exc:  # the workbench is alpha: any failure falls back to the procedural fit
        result.attempts.append({"method": "autorig-workbench", "status": "FAIL", "error": str(exc).strip()[:300]})
        return None
    from ..core.limbs import gate_by_distance, island_cleanup, islands, smooth_weights
    edges = _edges(meshes)
    label = islands(len(pts), edges)
    mw = arm.matrix_world
    heads = {b.name: np.array(mw @ b.head_local) for b in arm.data.bones}
    segments = {b.name: (np.array(mw @ b.head_local), np.array(mw @ b.tail_local)) for b in arm.data.bones}
    wfit = _fit_from_armature(arm, pts)

    def judge(W: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        W, gated = gate_by_distance(_normalize_limit(W), names, pts, segments)
        W, cleanup = island_cleanup(smooth_weights(W, edges), names, label, pts, heads)
        cleanup["far_bones_removed"] = gated
        W = _normalize_limit(W)
        test = deformation_tests(pts, names, W, wfit) if wfit.ok else {"status": "CHECK", "score": 1.0,
                                                                       "tests": {}, "weights": {}}
        return test, W, cleanup

    # its skeleton with two skinnings: the workbench's own (on its welded voxel proxy), and Blender's bone heat on
    # the source meshes themselves (separate pieces, such as an arm modelled apart, stay apart)
    tries = []
    test, W1, cleanup = judge(W)
    tries.append((test, W1, "autorig-workbench"))
    result.attempts.append({"method": "autorig-workbench", "facing": summary.get("facing"), "arms": summary.get("arms"),
                            "cleanup": cleanup, "seconds": summary.get("seconds"), **test})
    try:
        Wh = _welded_bone_heat(bpy, meshes, arm, names)
        empty = Wh.sum(axis=1) < 1e-6  # bone heat leaves interior or unreachable vertices out
        Wh[empty] = W[empty]
        test, W2, cleanup = judge(Wh)
        tries.append((test, W2, "autorig-workbench+bone-heat"))
        result.attempts.append({"method": "autorig-workbench+bone-heat", "filled_unweighted": int(empty.sum()),
                                "cleanup": cleanup, **test})
    except RuntimeError as exc:
        result.attempts.append({"method": "autorig-workbench+bone-heat", "status": "FAIL",
                                "error": str(exc).strip()[:200]})
    test, W, method = min(tries, key=lambda t: (STATUS_ORDER[t[0]["status"]], t[0]["score"]))
    if wfit.ok:
        result.height = wfit.height
    return test, W, method, arm, names


def _welded_bone_heat(bpy: Any, meshes: list[Any], arm: Any, names: list[str]) -> np.ndarray:
    """Bone heat on welded copies of the meshes (it finds no solution on meshes split at UV seams or flat-shaded
    edges), carried back to the source vertices by position. Rows in ``_points`` order."""
    import bmesh
    from mathutils.kdtree import KDTree
    copies = []
    try:
        for obj in meshes:
            me = obj.data.copy()
            if me.shape_keys:
                c = bpy.data.objects.new(obj.name + "_heat", me)
                c.shape_key_clear()
            else:
                c = bpy.data.objects.new(obj.name + "_heat", me)
            bpy.context.scene.collection.objects.link(c)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 * max(c.dimensions) + 1e-9)
            bm.to_mesh(me)
            bm.free()
            c.vertex_groups.clear()
            copies.append(c)
        _bone_heat(bpy, copies, arm)
        out = []
        for obj, c in zip(meshes, copies):
            Wc = _read_weights([c], names)
            tree = KDTree(len(c.data.vertices))
            for v in c.data.vertices:
                tree.insert(v.co, v.index)
            tree.balance()
            out.append(Wc[[tree.find(v.co)[1] for v in obj.data.vertices]])
        return np.vstack(out)
    finally:
        for c in copies:
            me = c.data
            bpy.data.objects.remove(c, do_unlink=True)
            bpy.data.meshes.remove(me)


def _edges(meshes: list[Any]) -> np.ndarray:
    """Edges over the stacked vertices of ``meshes`` (``_points`` order), with vertices at one spot (split normals
    or UV seams) joined, so a piece is one island."""
    parts, base, keys = [], 0, []
    for obj in meshes:
        e = np.empty(len(obj.data.edges) * 2, np.int64)
        obj.data.edges.foreach_get("vertices", e)
        parts.append(e.reshape(-1, 2) + base)
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        keys.append(co.reshape(-1, 3))
        base += len(obj.data.vertices)
    co = np.vstack(keys)
    _, first = np.unique(np.round(co / (np.ptp(co, axis=0).max() * 1e-5 + 1e-12)), axis=0, return_inverse=True)
    first = first.ravel()
    same = np.full(first.max() + 1, -1)
    links = []
    for i, k in enumerate(first):  # each vertex linked to the first one at its spot
        if same[k] < 0:
            same[k] = i
        elif same[k] != i:
            links.append((same[k], i))
    return np.vstack(parts + ([np.array(links)] if links else []))


def _fit_from_armature(arm: Any, pts: np.ndarray):
    """A RigFit over the armature's VRM-named bones, for the same deformation tests the procedural fit gets."""
    from ..core.autorig import Frame, RigFit
    mw = arm.matrix_world
    bones = {b.name: (np.array(mw @ b.head_local), np.array(mw @ b.tail_local)) for b in arm.data.bones}
    if any(n not in bones for n in HIERARCHY):
        return RigFit(ok=False)
    left = bones["LeftUpperArm"][0] - bones["RightUpperArm"][0]
    left[2] = 0
    left /= np.linalg.norm(left)
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.cross(left, up)
    lo = pts.min(axis=0)
    origin = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2, lo[2]])
    frame = Frame(origin=origin, left=left, up=up, forward=fwd, confidence=1.0)
    return RigFit(ok=True, bones={n: bones[n] for n in HIERARCHY}, frame=frame, height=float(np.ptp(pts[:, 2])),
                  confidence=1.0)


def _finish(bpy: Any, meshes: list[Any], result: AutoRigResult, test: dict, W: np.ndarray, method: str, arm: Any,
            names: list[str]) -> AutoRigResult:
    for obj in meshes:  # a losing bone-heat attempt left its modifier behind
        for m in [m for m in obj.modifiers if m.type == "ARMATURE"]:
            obj.modifiers.remove(m)
    _bind(bpy, meshes, arm)
    _write_weights(meshes, names, W)
    result.method, result.bones = method, names
    result.status = test["status"]
    if method.startswith("autorig-workbench"):
        result.confidence = max(result.confidence, 0.8 if test["status"] == "PASS" else 0.6)
    if test["status"] == "FAIL":
        result.warnings.append("no automatic skinning passed the deformation tests; the weights are kept for "
                               "inspection but the result is not trustworthy")
    return result


def auto_rig(bpy: Any = None) -> AutoRigResult:
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    if any(o.type == "ARMATURE" for o in bpy.data.objects):
        raise ConversionError("autorig-has-armature", "The scene already has an armature; auto-rig never replaces one.")
    meshes = _meshes(bpy)
    if not meshes:
        raise ConversionError("autorig-no-mesh", "No mesh to rig.")
    if any(m.type == "ARMATURE" for o in meshes for m in o.modifiers):
        raise ConversionError("autorig-has-armature", "A mesh already carries an armature modifier.")

    _bake_world(bpy, meshes)
    pts, _ = _points(meshes)
    fit = fit_skeleton(pts)
    result = AutoRigResult("FAIL", confidence=fit.confidence, measured=fit.measured, height=fit.height,
                           meshes=[o.name for o in meshes], warnings=list(fit.warnings))

    # Attempt 1: Autorig Workbench (Mesh2Motion's human skeleton with fingers, its own skinning).
    wb = _workbench(bpy, meshes, pts, result)
    if wb is not None and wb[0]["status"] == "PASS":
        return _finish(bpy, meshes, result, *wb)
    if not fit.ok:
        if wb is not None:
            return _finish(bpy, meshes, result, *wb)
        return result

    arm = _build_armature(bpy, fit)
    names = list(fit.bones)
    best: tuple[dict, np.ndarray, str] | None = None

    _, W_fallback = distance_weights(pts, fit, MAX_INFLUENCES)

    def consider(method: str, W: np.ndarray | None) -> None:
        nonlocal best
        if W is None:
            result.attempts.append({"method": method, "status": "FAIL", "error": "no weights produced"})
            return
        empty = W.sum(axis=1) < 1e-6  # bone heat leaves detached or interior vertices unweighted
        W = np.where(empty[:, None], W_fallback, W)
        test = deformation_tests(pts, names, W, fit)
        result.attempts.append({"method": method, "filled_unweighted": int(empty.sum()), **test})
        if best is None or better(test, best[0]):
            best = (test, W, method)

    # Attempt 2: Blender bone heat.
    try:
        _bone_heat(bpy, meshes, arm)
        W1 = _read_weights(meshes, names)
        W1 = _normalize_limit(W1) if W1.any() else None
    except RuntimeError as exc:
        result.attempts.append({"method": "blender-bone-heat", "status": "FAIL", "error": str(exc).strip()[:200]})
        W1 = None
    else:
        consider("blender-bone-heat", W1)
    # Attempt 3 (a genuinely different method), only when attempt 2 was not clean.
    if best is None or best[0]["status"] != "PASS":
        consider("distance-weights", W_fallback)

    assert best is not None
    test, W, method = best
    if wb is not None and not better(test, wb[0]):
        bpy.data.objects.remove(arm, do_unlink=True)
        return _finish(bpy, meshes, result, *wb)
    if wb is not None:
        bpy.data.objects.remove(wb[3], do_unlink=True)
    _bind(bpy, meshes, arm)
    _write_weights(meshes, names, W)
    result.method, result.bones = method, names
    result.status = test["status"]
    if test["status"] == "FAIL":
        result.warnings.append("no automatic skinning passed the deformation tests; the weights are kept for "
                               "inspection but the result is not trustworthy")
    return result
