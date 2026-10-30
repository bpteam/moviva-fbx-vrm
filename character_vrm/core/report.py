"""Conversion report, ``inspect`` text and the suggested ``*.vrm-map.json``."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .anatomy.report import analysis_to_dict, format_analysis
from .authoring.report import authoring_to_dict, format_authoring
from .authoring.rest_pose import plan_rest_pose
from .expressions import BLINKS, LOOKS, VISEMES
from .model import ALL_HUMAN_BONES, REQUIRED_HUMAN_BONES, ExpressionMapping

if TYPE_CHECKING:  # pragma: no cover
    from ..toolchain import Toolchain
    from .errors import ConversionError
    from .pipeline import Analysis
    from .validation import OutputCheck


def _sha256(path: str | Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _expr(mapping: ExpressionMapping) -> dict[str, Any]:
    return {
        "status": "available" if mapping.kind in ("direct", "config") else "generated",
        "kind": mapping.kind, "confidence": mapping.confidence, "reason": mapping.reason,
        "binds": [{"mesh": b.mesh, "shapeKey": b.shape_key, "weight": b.weight} for b in mapping.binds],
    }


def build_report(
    toolchain: "Toolchain", analysis: "Analysis | None", status: str, output: "OutputCheck | None",
    source: str | Path, error: "ConversionError | None" = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": status,
        "toolchain": {"blender": toolchain.blender, "vrmAddon": toolchain.vrm_addon, "target": toolchain.target},
        "source": {"path": str(source), "format": Path(source).suffix.lower().lstrip("."), "sha256": _sha256(source)},
        "geometry": {}, "humanoid": {}, "expressions": {}, "lookAt": {}, "springBones": {}, "materials": {},
        "output": None, "warnings": [], "errors": [],
    }
    if error:
        report["errors"].append({"code": error.code, "message": error.message, "details": error.details})
    if analysis is None:
        return report

    ins, hum, ex = analysis.inspection, analysis.humanoid, analysis.expressions
    mat, springs = analysis.materials, analysis.springs
    arm_reason_values = {"arm-weight", "arm-parent", "shoulder-transition"}
    protected_arm_vertices = sum(
        1 for reasons in analysis.geometry.protected_reasons.values()
        for vertex_reasons in reasons.values()
        if {str(reason.value) for reason in vertex_reasons} & arm_reason_values
    )
    report["geometry"] = {
        "meshes": len(ins.meshes), "skinnedMeshes": len(ins.skinned_meshes), "triangles": ins.triangle_count,
        "materials": len(ins.materials), "textures": len(ins.images), "shapeKeys": ex.shape_key_count,
        "maxTexture": list(mat.max_texture), "skeletonHeightM": round(analysis.armature.height, 3),
        "excludedMeshes": analysis.excluded_meshes,
        "armatures": [{"name": s.name, "score": s.score, "bones": s.bones, "skinnedMeshes": s.skinned_meshes,
                       "requiredMapped": s.required_mapped, "selected": s.name == analysis.selection.name}
                      for s in analysis.selection.scores],
        "profile": analysis.geometry.profile,
        "semanticFrame": analysis.geometry.semantic_frame,
        "preserveFullArms": analysis.geometry.profile in ("video-call", "face"),
        "cutLandmark": analysis.geometry.cut_landmark,
        "cutSource": analysis.geometry.cut_source,
        "landmarkM": analysis.geometry.landmark,
        "safetyMarginM": analysis.geometry.safety_margin,
        "cutPlaneM": analysis.geometry.cut_plane,
        "protectedObjects": sorted(analysis.geometry.protected_objects),
        "protectedArmVertices": protected_arm_vertices,
        "protectedVertices": {k: len(v) for k, v in analysis.geometry.protected_vertices.items()},
        "deleteObjects": list(analysis.geometry.delete_objects),
        "cropObjects": list(analysis.geometry.crop_objects),
        "preserveObjects": list(analysis.geometry.preserve_objects),
        "removedPolygons": {k: len(v) for k, v in analysis.geometry.remove_polygons.items()},
    }
    report["humanoid"] = {
        "armature": hum.armature, "profile": hum.profile,
        "requiredMapped": hum.mapped_required, "requiredTotal": len(REQUIRED_HUMAN_BONES),
        "optionalMapped": hum.mapped_optional, "meanConfidence": round(hum.mean_confidence, 3),
        "mappings": [{"vrmBone": m.vrm_bone, "sourceBone": m.source_bone, "confidence": m.confidence,
                      "reason": m.reason} for m in hum.mappings.values()],
        "missingRequired": hum.missing_required,
        "belowThreshold": [asdict(m) for m in hum.below_threshold],
        "preservedUnmappedBones": len(analysis.armature.bones) - len(hum.mappings),
    }
    report["expressions"] = {
        "capability": ex.capability, "arkit": ex.arkit, "arkitCoverage": round(ex.arkit_coverage, 2),
        "mapped": {n: _expr(m) for n, m in ex.mappings.items()},
        "missing": ex.missing,
        "belowThreshold": [{"name": m.name, "reason": m.reason, "confidence": m.confidence} for m in ex.below_threshold],
    }
    report["lookAt"] = {"type": analysis.look_at.type, "reason": analysis.look_at.reason,
                        "offsetFromHeadBone": list(analysis.look_at.offset_from_head_bone)
                        if analysis.look_at.offset_from_head_bone else None}
    if analysis.authoring is not None and analysis.authoring.springs is not None:
        springs = analysis.authoring.springs  # the anatomy-derived plan replaces the name-based one
    report["springBones"] = {
        "enabled": analysis.config.spring.enabled,
        "chains": [{"name": c.name, "bones": c.bones, "confidence": c.confidence, "reason": c.reason,
                    "preset": asdict(c.preset)} for c in springs.chains],
        "colliders": [asdict(c) for c in springs.colliders],
        "skipped": springs.skipped,
    }
    report["materials"] = {
        "total": mat.total, "mtoonConverted": 0, "pbrPreserved": len(mat.pbr_preserved),
        "procedural": mat.procedural, "nonPrincipled": mat.non_principled, "channels": mat.channels,
        "textures": mat.textures,
    }
    if output:
        report["output"] = {"path": output.path, "sizeBytes": output.size_bytes, "specVersion": output.spec_version,
                            "nodes": output.nodes, "meshes": output.meshes, "skins": output.skins,
                            "materials": output.materials, "images": output.images,
                            "humanBones": output.humanoid_bones, "expressions": output.expressions,
                            "springs": output.springs, "lookAt": output.look_at_type}
    report["anatomy"] = anatomy_section(analysis)
    report["authoring"] = authoring_to_dict(analysis.authoring, analysis.authoring_result)
    report["warnings"] = [{"code": w.code, "message": w.message} for w in analysis.warnings]
    return report


def anatomy_section(analysis: "Analysis") -> dict[str, Any]:
    """The per-avatar calibration artifact: the *same* dict is what ``analyze --json`` writes."""
    if analysis.anatomy is None:
        return analysis_to_dict(None)
    pose = analysis.authoring.rest_pose.to_dict() if analysis.authoring and analysis.authoring.rest_pose \
        else plan_rest_pose(analysis.anatomy).to_dict()
    generated = sorted(analysis.authoring_result.applied_shape_keys) if analysis.authoring_result else []
    hair_extra = None
    if analysis.authoring and analysis.authoring.hair_rig:
        hr = analysis.authoring.hair_rig
        hair_extra = {"generatedRig": [c.name for c in hr.chains], "generatedRigSkipped": [s.reason for s in hr.skipped]}
    return analysis_to_dict(analysis.anatomy, pose, generated, hair_extra)


# ----------------------------------------------------------------- inspect / map


def _status(analysis: "Analysis", name: str) -> str:
    return analysis.expressions.status(name)


def format_inspection(analysis: "Analysis") -> str:
    base = _format_inspection(analysis)
    parts = [base, "", format_analysis(analysis.anatomy)]
    if analysis.authoring is not None:
        parts += ["", format_authoring(analysis.authoring, analysis.authoring_result)]
    return "\n".join(parts)


def _format_inspection(analysis: "Analysis") -> str:
    ins, hum, ex = analysis.inspection, analysis.humanoid, analysis.expressions
    ok, reasons = analysis.convertible
    arm_reason_values = {"arm-weight", "arm-parent", "shoulder-transition"}
    protected_arm_vertices = sum(
        1 for mesh_reasons in analysis.geometry.protected_reasons.values()
        for vertex_reasons in mesh_reasons.values()
        if {str(reason.value) for reason in vertex_reasons} & arm_reason_values
    )
    lines = [
        "Character inspection", "",
        "Geometry",
        f"  meshes:        {len(ins.meshes):>8}",
        f"  skinned meshes:{len(ins.skinned_meshes):>8}",
        f"  triangles:     {ins.triangle_count:>8,}",
        f"  materials:     {len(ins.materials):>8}",
        f"  textures:      {len(ins.images):>8}", "",
        "Geometry profile",
        f"  {analysis.geometry.profile} ("
        f"{'WAIST-UP' if analysis.geometry.profile == 'video-call' else 'BUST + FULL ARMS' if analysis.geometry.profile == 'face' else 'FULL BODY'})",
        f"  cut source:    {analysis.geometry.cut_source or 'none'}",
        f"  cut plane:     {analysis.geometry.cut_plane if analysis.geometry.cut_plane is not None else 'none'}",
        f"  protected arm vertices: {protected_arm_vertices:,}",
        "  skeleton: unchanged", "",
        "Armature",
        f"  bones:         {len(analysis.armature.bones):>8}",
        f"  candidate: {analysis.selection.name}"
        + (f" (of {len(analysis.selection.scores)})" if len(analysis.selection.scores) > 1 else ""), "",
        "Humanoid",
        f"  mapped required: {hum.mapped_required}/{len(REQUIRED_HUMAN_BONES)}",
        f"  optional:        {hum.mapped_optional}",
        f"  confidence:     {round(hum.mean_confidence * 100)}%"
        + (f"  (profile: {hum.profile})" if hum.profile else ""), "",
        "Face",
        f"  shape keys:    {ex.shape_key_count:>8}",
        f"  ARKit profile:   {'likely' if ex.arkit == 'likely' else ex.arkit}", "",
        "Expressions",
    ]
    for name in (*BLINKS, *VISEMES):
        lines.append(f"  {name:<17} {_status(analysis, name)}")
    lines += ["", "LookAt", f"  {analysis.look_at.type or 'not configured'} ({analysis.look_at.reason})", "",
              "Hair", f"  candidate chains: {len(analysis.springs.chains) + len(analysis.springs.skipped)}"
              f" ({len(analysis.springs.chains)} enabled)", ""]
    if analysis.warnings:
        lines.append("Warnings")
        lines += [f"  - [{w.code}] {w.message}" for w in analysis.warnings]
        lines.append("")
    lines += ["Result", "  convertible" if ok else "  NOT convertible without a map file"]
    lines += [f"    - {r}" for r in reasons]
    return "\n".join(lines)


def suggest_map(analysis: "Analysis", min_confidence: float) -> dict[str, Any]:
    """A map file the user can review. Only mappings above the threshold go into ``humanoid``;
    lower-confidence guesses are listed under ``_needsReview`` so they are never accepted implicitly."""
    hum, ex = analysis.humanoid, analysis.expressions
    suggestion: dict[str, Any] = {
        "_comment": "Generated by `character-vrm map`. Explicit entries override every heuristic. "
                    "Review _needsReview and move entries into `humanoid` only if they are right.",
        "armature": analysis.selection.name,
        "humanoid": {b: hum.mappings[b].source_bone for b in ALL_HUMAN_BONES if b in hum.mappings},
        "expressions": {
            name: {b.shape_key: b.weight for b in m.binds if b.mesh == m.binds[0].mesh}
            for name, m in ex.mappings.items() if m.binds
        },
        "lookAt": {"type": analysis.look_at.type or "none"},
        "springBones": {
            "enabled": analysis.config.spring.enabled,
            "include": [c.bones[0] for c in analysis.springs.chains],
            "ignore": [],
        },
        "_confidence": {b: m.confidence for b, m in hum.mappings.items()},
        "_needsReview": {
            m.vrm_bone: {"source": m.source_bone, "confidence": m.confidence, "reason": m.reason}
            for m in hum.below_threshold
        },
        "_missingRequired": hum.missing_required,
    }
    suggestion["_thresholds"] = {"humanoid": min_confidence}
    if analysis.springs.skipped:
        suggestion["_skippedSpringCandidates"] = [
            {"bone": s.get("bone"), "reason": s["reason"]} for s in analysis.springs.skipped]
    return suggestion
