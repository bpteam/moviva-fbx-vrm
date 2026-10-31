"""Procedural auto-rig inside Blender: bake meshes to world space, fit a skeleton, skin, keep the best result.

Attempt 1 is Blender's bone-heat weighting (``ARMATURE_AUTO``), attempt 2 the deterministic distance weights from
``core.autorig``. Each result is read back from the real vertex groups and pose-tested numerically; the better
one is kept. Only runs on a scene with no armature, so a source rig is never replaced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.autorig import (HIERARCHY, AutoRigResult, better, deformation_tests, distance_weights, fit_skeleton)
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
    if not fit.ok:
        return AutoRigResult("FAIL", warnings=fit.warnings)
    result = AutoRigResult("FAIL", confidence=fit.confidence, measured=fit.measured, height=fit.height,
                           meshes=[o.name for o in meshes], warnings=list(fit.warnings))

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

    # Attempt 1: Blender bone heat.
    try:
        _bone_heat(bpy, meshes, arm)
        W1 = _read_weights(meshes, names)
        W1 = _normalize_limit(W1) if W1.any() else None
    except RuntimeError as exc:
        result.attempts.append({"method": "blender-bone-heat", "status": "FAIL", "error": str(exc).strip()[:200]})
        W1 = None
    else:
        consider("blender-bone-heat", W1)
    # Attempt 2 (a genuinely different method), only when attempt 1 was not clean.
    if best is None or best[0]["status"] != "PASS":
        consider("distance-weights", W_fallback)

    assert best is not None
    test, W, method = best
    _bind(bpy, meshes, arm)
    _write_weights(meshes, names, W)
    result.method, result.bones = method, names
    result.status = test["status"]
    if test["status"] == "FAIL":
        result.warnings.append("no automatic skinning passed the deformation tests; the weights are kept for "
                               "inspection but the result is not trustworthy")
    return result
