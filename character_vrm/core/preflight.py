"""Geometry-first preflight: what the model is, before any conversion. Works with or without an armature."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .autorig import fit_skeleton
from .config import ConversionConfig
from .errors import ConversionError
from .model import CharacterInspection
from .pipeline import analyze
from .steps import Facts

GEOMETRY_MIN_CONFIDENCE = 0.35  # below this the silhouette is not recognizably a standing humanoid


def hardware() -> dict:
    import os
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            gpu = out[0] if out else None
        except (OSError, subprocess.SubprocessError):
            gpu = None
    return {"cpuCores": os.cpu_count(), "nvidiaGpu": gpu}


def run_preflight(inspection: CharacterInspection, source: str | Path, config: ConversionConfig) -> tuple[dict, Facts]:
    """-> (JSON-able description, planner facts)."""
    pts = np.array([p for m in inspection.meshes for p in m.vertex_positions], dtype=float).reshape(-1, 3)
    fit = fit_skeleton(pts) if len(pts) else None
    geometry = {"ok": bool(fit and fit.ok), "confidence": fit.confidence if fit else 0.0,
                "measured": fit.measured if fit else {}, "warnings": fit.warnings if fit else ["no mesh vertices"]}
    facts = Facts(humanoid_geometry=bool(fit and fit.ok and fit.confidence >= GEOMETRY_MIN_CONFIDENCE),
                  armature=bool(inspection.armatures), skinned=bool(inspection.skinned_meshes))

    info: dict = {
        "file": {"name": Path(source).name, "format": inspection.source_format,
                 "bytes": Path(source).stat().st_size if Path(source).is_file() else 0},
        "meshes": len(inspection.meshes), "triangles": inspection.triangle_count,
        "textures": [{"name": i.name, "width": i.width, "height": i.height} for i in inspection.images],
        "armatures": len(inspection.armatures), "skinnedMeshes": len(inspection.skinned_meshes),
        "shapeKeys": len(inspection.shape_keys), "geometry": geometry,
        "complexity": "heavy" if inspection.triangle_count > 400_000 else
                      "standard" if inspection.triangle_count > 60_000 else "fast",
        "hardware": hardware(), "problems": [],
    }
    # Face geometry: a head with enough vertices above the neck, from the geometry alone.
    if fit and fit.ok and fit.frame is not None:
        top = fit.frame.to_local(pts)
        head_u = fit.frame.to_local(fit.bones["Head"][0])[1]
        facts.face_geometry = int((top[:, 1] > head_u).sum()) >= 1500

    if inspection.armatures:
        try:
            a = analyze(inspection, config)
            info["rig"] = {"requiredMapped": sum(1 for _ in a.humanoid.mappings), "missing": a.humanoid.missing_required,
                           "convertible": a.convertible[0]}
            facts.required_bones_mapped = len(a.humanoid.mappings)
            facts.expressions_missing = bool(a.expressions.missing)
            facts.spring_bones_possible = bool(a.springs.chains)
            info["expressions"] = {"available": sorted(a.expressions.mappings), "missing": a.expressions.missing}
        except ConversionError as exc:
            info["problems"].append(exc.message)
    else:
        facts.expressions_missing = True  # nothing to preserve: expressions follow the rig step
        info["problems"].append("The source has no armature and no skin weights.")
    if not facts.humanoid_geometry and not inspection.armatures:
        info["problems"].append("The geometry does not look like a standing humanoid.")
    return info, facts
