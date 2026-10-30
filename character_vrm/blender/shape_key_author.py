"""MUTATE + VALIDATE RESULT for generated ``Prosopon_*`` shape keys.

Inputs are ``ShapeKeyPlan`` objects that the pure core already validated. Source shape keys are never touched: a key
whose name already exists is refused, not overwritten. Not verified against a real Blender by the authors; see
``tests/integration/README.md``.
"""

from __future__ import annotations

from typing import Any

from ..core.authoring.plans import PROSOPON_PREFIX, ShapeKeyPlan

TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-6  # of the coordinate's magnitude: a few float32 ulps


def apply_shape_keys(plans: list[ShapeKeyPlan], bpy: Any = None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Create every planned key on every affected mesh. Returns ``(applied: key -> meshes, failed: key -> reason)``."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    applied: dict[str, list[str]] = {}
    failed: dict[str, str] = {}
    for plan in plans:
        if not plan.name.startswith(PROSOPON_PREFIX):
            failed[plan.name] = "outside the Prosopon_ namespace"
            continue
        created: list[str] = []
        try:
            for mesh_name, offsets in plan.offsets.items():
                obj = bpy.data.objects.get(mesh_name)
                if obj is None or obj.type != "MESH":
                    raise LookupError(f"mesh object '{mesh_name}' not found")
                _add_key(obj, plan.name, offsets)
                created.append(mesh_name)
        except Exception as exc:  # noqa: BLE001 - one bad key must not abort the conversion
            failed[plan.name] = str(exc)
            for mesh_name in created:  # a half-created key is worse than none
                _remove_key(bpy.data.objects.get(mesh_name), plan.name)
            continue
        applied[plan.name] = created
    return applied, failed


def _add_key(obj: Any, name: str, offsets: dict[int, tuple[float, float, float]]) -> None:
    from mathutils import Vector  # type: ignore[import-not-found]

    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    keys = obj.data.shape_keys
    if name in keys.key_blocks:
        raise ValueError(f"shape key '{name}' already exists on '{obj.name}'; source keys are never overwritten")
    reference = keys.reference_key
    block = obj.shape_key_add(name=name, from_mix=False)
    block.value = 0.0
    block.slider_min, block.slider_max = 0.0, 1.0
    inverse = obj.matrix_world.to_3x3().inverted()  # plan offsets are world-space deltas
    for index, delta in offsets.items():
        local = inverse @ Vector(delta)
        block.data[index].co = reference.data[index].co + local


def _remove_key(obj: Any, name: str) -> None:
    if obj is not None and obj.data.shape_keys is not None and name in obj.data.shape_keys.key_blocks:
        obj.shape_key_remove(obj.data.shape_keys.key_blocks[name])


def validate_shape_keys(plans: list[ShapeKeyPlan], applied: dict[str, list[str]], bpy: Any = None) -> dict[str, str]:
    """VALIDATE RESULT: each created key moves exactly the planned vertices by the planned amount (and nothing else)."""
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    from mathutils import Vector  # type: ignore[import-not-found]

    bad: dict[str, str] = {}
    for plan in plans:
        if plan.name not in applied:
            continue
        for mesh_name in applied[plan.name]:
            obj = bpy.data.objects.get(mesh_name)
            keys = obj.data.shape_keys if obj is not None else None
            if keys is None or plan.name not in keys.key_blocks:
                bad[plan.name] = f"key missing on '{mesh_name}' after creation"
                break
            block, reference = keys.key_blocks[plan.name], keys.reference_key
            inverse = obj.matrix_world.to_3x3().inverted()
            offsets = plan.offsets[mesh_name]
            moved = planned = missing = 0
            for index in range(len(block.data)):
                # shape keys store float32 coordinates: what a vertex can register as a move grows with its distance
                # from the origin (a centimetre-scale Character Creator mesh loses offsets below ~1e-5)
                tol = max(TOLERANCE, RELATIVE_TOLERANCE * reference.data[index].co.length)
                want = inverse @ Vector(offsets.get(index, (0.0, 0.0, 0.0)))
                delta = block.data[index].co - reference.data[index].co
                # a planned offset near the tolerance (a vertex with a trace of jaw weight) may or may not register
                planned += want.length > 2 * tol
                missing += want.length > 2 * tol and delta.length <= tol
                if delta.length > tol:
                    moved += 1
                    if (delta - want).length > 1e-4 * max(1.0, want.length) + tol:
                        bad[plan.name] = f"vertex {index} on '{mesh_name}' deviates from the plan"
                        break
            else:
                if missing:
                    bad[plan.name] = f"{moved} vertices moved on '{mesh_name}', {planned} planned ({missing} did not move)"
                continue
            break
    return bad
