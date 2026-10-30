"""Routes an ``AutoAuthorPlan`` to the Blender authors and reports back what exists (VALIDATE RESULT).

The pure core has already analysed, planned and validated the plan; this is the only place it touches the scene.
Failures are recorded per item, never swallowed and never partially reported as success. Not verified against a real
Blender by the authors; run ``tests/integration/README.md``.
"""

from __future__ import annotations

from typing import Any

from ..core.authoring.plan import AuthoringResult, AutoAuthorPlan
from . import hair_author, material_author, rig_author, shape_key_author


def apply_authoring(plan: AutoAuthorPlan, bpy: Any = None) -> AuthoringResult:
    if bpy is None:
        import bpy  # type: ignore[no-redef]
    result = AuthoringResult()

    for finger in plan.fingers:
        if not finger.bones or finger.validate():
            continue
        try:
            result.created_bones |= rig_author.apply_finger_rig(finger, plan.armature, bpy)
        except Exception as exc:  # noqa: BLE001
            result.failed[f"{finger.side} finger rig"] = str(exc)

    if plan.hair_rig is not None and plan.hair_rig.chains and not plan.hair_rig.validate():
        try:
            result.created_bones |= hair_author.apply_hair_rig(plan.hair_rig, plan.armature, bpy)
        except Exception as exc:  # noqa: BLE001
            result.failed["hair rig"] = str(exc)

    if plan.materials is not None:
        done, failed = material_author.apply_materials(plan.materials.valid_edits(), bpy)
        result.materials = done
        result.failed.update(failed)

    keys = plan.shape_keys
    if keys:
        applied, failed = shape_key_author.apply_shape_keys(keys, bpy)
        bad = shape_key_author.validate_shape_keys(keys, applied, bpy)
        for name, reason in bad.items():  # VALIDATE RESULT: a key that does not match its plan is removed
            failed[name] = reason
            for mesh in applied.pop(name, []):
                shape_key_author._remove_key(bpy.data.objects.get(mesh), name)
        result.applied_shape_keys = applied
        result.failed.update(failed)
        result.validation = [f"{name}: {reason}" for name, reason in bad.items()]
    return result
