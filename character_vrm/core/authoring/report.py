"""Report vocabulary for the Auto-Author: SOURCE / MAPPED / INFERRED / GENERATED / MISSING."""

from __future__ import annotations

from typing import Any

from ..anatomy.confidence import STATUS_GENERATED, STATUS_MISSING
from .plan import AuthoringResult, AutoAuthorPlan
from .plans import Skipped


def _skip(s: Skipped) -> dict[str, Any]:
    return {"what": s.what, "reason": s.reason, "confidence": s.confidence}


def authoring_to_dict(plan: AutoAuthorPlan | None, result: AuthoringResult | None) -> dict[str, Any]:
    if plan is None:
        return {"enabled": False}
    face = plan.face
    applied = set(result.applied_shape_keys) if result else None
    out: dict[str, Any] = {
        "enabled": True,
        "flags": {"autoAuthor": plan.config.enabled, "generateFace": plan.config.generate_face,
                  "generateFingerRig": plan.config.generate_finger_rig, "generateHairRig": plan.config.generate_hair_rig},
        "applied": result is not None,
        "restPose": plan.rest_pose.to_dict() if plan.rest_pose else None,
        "expressions": {}, "shapeKeys": [],
        "fingers": [], "hairRig": None, "springs": None, "colliders": [], "skipped": [], "planIssues": [
            {"plan": i.plan, "message": i.message} for i in plan.issues],
        "result": None,
    }
    if face is not None:
        for name, d in face.decisions.items():
            entry: dict[str, Any] = {"status": d.status, "confidence": round(d.confidence, 3), "detail": d.detail}
            if d.status == STATUS_MISSING:
                sk = next((s for s in face.skipped if s.what == name), None)
                if sk:
                    entry["reason"] = sk.reason
            if d.status == STATUS_GENERATED and applied is not None and not any(
                    k in applied for k in d.detail.split(" + ")):
                entry["status"], entry["reason"] = STATUS_MISSING, "generated key was not created"
            out["expressions"][name] = entry
        for sk in face.skipped:
            if sk.what in ("browUp", "browDown", "squint") or sk.what not in out["expressions"]:
                out["expressions"].setdefault(sk.what, {"status": STATUS_MISSING, "reason": sk.reason,
                                                        "confidence": sk.confidence or 0.0, "detail": ""})
        out["shapeKeys"] = [{"name": k.name, "meshes": k.meshes, "vertices": k.vertex_count,
                             "maxOffset": round(k.max_offset, 5), "confidence": round(k.confidence, 3),
                             "boundTo": k.vrm_expression, "description": k.description,
                             "valid": k.name in face._valid_names()} for k in face.shape_keys]
        out["augmented"] = {expr: [{"key": k, "weight": w} for k, w in adds] for expr, adds in face.augment.items()}
    for f in plan.fingers:
        out["fingers"].append({"side": f.side, "bones": len(f.bones), "confidence": round(f.confidence, 3),
                               "skipped": _skip(f.skipped) if f.skipped else None})
    if plan.hair_rig is not None:
        out["hairRig"] = {"chains": [{"name": c.name, "bones": len(c.bones), "category": c.category,
                                      "confidence": round(c.confidence, 3)} for c in plan.hair_rig.chains],
                          "skipped": [_skip(s) for s in plan.hair_rig.skipped]}
    if plan.springs is not None:
        out["springs"] = {"chains": [{"name": c.name, "bones": c.bones, "confidence": c.confidence, "reason": c.reason,
                                      "stiffness": c.preset.stiffness, "drag": c.preset.drag_force,
                                      "gravity": c.preset.gravity_power, "hitRadius": c.preset.hit_radius}
                                     for c in plan.springs.chains],
                          "skipped": [_skip(s) for s in plan.hair_spring_skipped]}
        out["colliders"] = [{"name": c.name, "bone": c.bone, "radius": c.radius, "offset": list(c.offset)}
                            for c in plan.springs.colliders]
    if plan.materials is not None:
        out["materials"] = [{"material": e.material, "role": e.role, "confidence": round(e.confidence, 3),
                             "changes": e.changes, "textures": e.textures} for e in plan.materials.edits]
    out["skipped"] = [_skip(s) for s in [*plan.skipped, *plan.collider_skipped]]
    if result is not None:
        out["result"] = {"shapeKeys": {k: v for k, v in result.applied_shape_keys.items()},
                         "materials": result.materials,
                         "createdBones": sorted(result.created_bones), "failed": result.failed,
                         "validation": result.validation}
    return out


def format_authoring(plan: AutoAuthorPlan | None, result: AuthoringResult | None = None) -> str:
    if plan is None:
        return "Auto-Author\n  off (use --auto-author)"
    d = authoring_to_dict(plan, result)
    lines = ["Auto-Author"]
    rp = d["restPose"] or {}
    if rp:
        lines.append(f"  rest pose      {rp['poseType']} (arm abduction {rp['armAbductionDeg']}, target "
                     f"{rp['targetAbductionDeg']} deg), metadata only")
    lines += ["", "Face"]
    for name, e in d["expressions"].items():
        extra = e.get("detail") or e.get("reason") or ""
        conf = f" confidence {e['confidence']:.2f}" if e.get("confidence") else ""
        lines.append(f"  {name:<14} {e['status']}{conf}" + (f"  ({extra})" if extra else ""))
    lines += ["", "Hair"]
    hr = d["hairRig"]
    if hr is None or (not hr["chains"] and not hr["skipped"]):
        lines.append("  generated rig  none needed or possible")
    for s in (hr["skipped"] if hr else []):
        lines.append(f"  generated rig  skipped: {s['reason']}")
    for c in (hr["chains"] if hr else []):
        lines.append(f"  generated rig  {c['name']} ({c['category']}, {c['bones']} bones) GENERATED")
    sp = d["springs"]
    lines.append(f"  spring bones   {len(sp['chains']) if sp else 0} chain(s), {len(d['colliders'])} collider(s)")
    for s in (sp["skipped"] if sp else []):
        lines.append(f"  spring skipped {s['what']}: {s['reason']}")
    lines += ["", "Materials"]
    if not d.get("materials"):
        lines.append("  nothing to change")
    for m in d.get("materials", []):
        lines.append(f"  {m['material']:<24} {m['role']:<8} {', '.join(m['changes'])}")
    lines += ["", "Fingers"]
    if not d["fingers"]:
        lines.append("  no generation needed")
    for f in d["fingers"]:
        lines.append(f"  {f['side']}: " + (f"skipped: {f['skipped']['reason']}" if f["skipped"] else
                                          f"{f['bones']} bones GENERATED"))
    for i in d["planIssues"]:
        lines.append(f"  PLAN ISSUE {i['plan']}: {i['message']}")
    return "\n".join(lines)
