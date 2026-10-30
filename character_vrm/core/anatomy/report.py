"""Serialisation and text rendering of ``HumanModelAnalysis`` (the one calibration artifact, no second source of truth)."""

from __future__ import annotations

from typing import Any

from .confidence import STATUS_INFERRED, STATUS_MAPPED, STATUS_MISSING, STATUS_SOURCE
from .hands import BONES_PRESENT, HandAnalysis
from .model import HumanModelAnalysis

REPORT_VERSION = 1


def _r(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(x, n)


def _vec(v) -> list[float]:
    return [round(c, 5) for c in v]


def _hand(h: HandAnalysis | None) -> dict[str, Any] | None:
    if h is None:
        return None
    out: dict[str, Any] = {"state": h.state, "confidence": _r(h.confidence, 3),
                           "independentArticulation": h.independent_articulation}
    if h.fingers:
        out["fingers"] = {f: {"bones": c.bones, "vrm": c.vrm_bones, "confidence": _r(c.confidence, 3),
                              "evidence": {k: round(v, 3) for k, v in c.evidence.items()}}
                          for f, c in h.fingers.items()}
    if h.segmentation is not None:
        s = h.segmentation
        out["geometry"] = {"separable": bool(s.clusters) and not s.fused, "fused": s.fused, "regions": s.runs_found,
                           "confidence": _r(s.confidence, 3), "reason": s.reason}
    if h.state == "fused-fingers":
        out["note"] = "independent finger articulation unavailable"
    return out


def analysis_to_dict(a: HumanModelAnalysis | None, pose: dict | None = None, face_generated: list[str] | None = None,
                     hair_extra: dict | None = None) -> dict[str, Any]:
    if a is None:
        return {"version": REPORT_VERSION, "available": False}
    m = a.measurements
    norm = m.normalized()
    out: dict[str, Any] = {
        "version": REPORT_VERSION, "available": True,
        "anatomy": {
            "height": _r(m.get("total_height") or m.total_height_units, 4),
            "shoulderWidth": _r(m.get("shoulder_width")), "headWidth": _r(m.get("head_width")),
            "measurements": {k: {"value": _r(v.value), "normalized": _r(norm.get(k)), "confidence": _r(v.confidence, 3),
                                 "source": v.source.value, "method": v.method} for k, v in m.values.items()},
            "frame": {"origin": _vec(a.frame.origin), "up": _vec(a.frame.up), "right": _vec(a.frame.right),
                      "forward": _vec(a.frame.forward), "scale": _r(a.frame.scale), "confidence": _r(a.frame.confidence, 3),
                      "evidence": {k: round(v, 3) for k, v in (a.frame.evidence or {}).items()}},
        },
        "landmarks": {k: {"position": _vec(v.position), "confidence": _r(v.confidence, 3), "source": v.source.value,
                          "method": v.method} for k, v in a.landmarks.items()},
        "skeleton": {
            "bones": {vrm: {"bone": s.bone, "confidence": _r(s.confidence, 3), "source": s.source.value,
                            "evidence": s.evidence, "method": s.method} for vrm, s in a.skeleton.assessments.items()},
            "inferred": dict(a.inferred_bones), "conflicts": a.skeleton.conflicts, "sidesAssumed": a.skeleton.side_assumed,
        },
        "pose": pose or {},
        "hands": {"left": _hand(a.left_hand), "right": _hand(a.right_hand)},
        "face": None,
        "hair": None,
        "materials": {mat.material: {"semantic": mat.semantic, "confidence": _r(mat.confidence, 3),
                                     "source": mat.source.value, "evidence": mat.evidence} for mat in a.materials},
        "warnings": [{"code": w.code, "message": w.message, "area": w.area} for w in a.warnings],
    }
    if a.face is not None:
        f = a.face
        out["face"] = {
            "quality": _r(f.quality, 3), "generated": face_generated or [], "headWidth": _r(f.head_width),
            "headHeight": _r(f.head_height), "mouthWidth": _r(f.mouth_width), "eyeBones": f.eye_bones, "jawBone": f.jaw_bone,
            "landmarks": sorted(f.landmarks),
            "landmarkMethods": {n: {"method": lm.method, "confidence": _r(lm.confidence, 3)}
                                for n, lm in sorted(f.landmarks.items())},
            "eyeSource": f.eye_source, "mouthSource": f.mouth_source, "jawWeightedLips": f.jaw_weighted_lips,
            "aaOpening": _r(f.aa_opening) if f.aa_opening is not None else None,
            "mouthCandidates": [{"source": c.source, "seam": _r(c.seam), "halfWidth": _r(c.half_width)
                                 if c.half_width is not None else None, "confidence": _r(c.confidence, 3),
                                 "detail": c.detail} for c in f.mouth_candidates],
        }
        if a.mouth is not None:
            out["face"]["mouth"] = {
                "approaches": a.mouth.approaches, "morphs": a.mouth.mouth_morphs, "innerMouthMeshes": a.mouth.inner_mouth_meshes,
                "geometry": {"vertices": a.mouth.geometry.vertices, "usable": a.mouth.geometry.usable,
                             "score": a.mouth.geometry.score, "reason": a.mouth.geometry.reason} if a.mouth.geometry else None}
    if a.hair is not None:
        h = a.hair
        out["hair"] = {
            "classification": h.classification, "springChains": len(h.chains),
            "meshes": [{"mesh": x.mesh, "material": x.material, "classification": x.classification,
                        "confidence": _r(x.confidence, 3),
                        "type": x.hair_type, "components": x.components, "evidence": x.evidence} for x in h.meshes],
            "chains": [{"root": c.root, "bones": c.bones, "category": c.category, "confidence": _r(c.confidence, 3),
                        "lengthRatio": _r(c.length_ratio, 3), "evidence": c.evidence} for c in h.chains],
            **(hair_extra or {}),
        }
    return out


def format_analysis(a: HumanModelAnalysis | None) -> str:
    if a is None:
        return "Anatomy\n  unavailable: no anatomical frame could be built (hips / head / a left-right pair missing)"
    m = a.measurements
    fr = a.frame
    lines = ["Anatomy", f"  frame confidence  {fr.confidence:.2f}   up {tuple(round(c, 2) for c in fr.up)}  "
             f"forward {tuple(round(c, 2) for c in fr.forward)}",
             f"  height            {m.get('total_height') or m.total_height_units:.3f}",
             ]
    for key in ("head_height", "head_width", "shoulder_width", "chest_width", "waist_width", "hip_width", "torso_length"):
        v = m.get(key)
        if v:
            lines.append(f"  {key:<17} {v:.3f}   ({m.values[key].source.value}, {m.values[key].confidence:.2f})")
    inferred = a.inferred_bones
    lines += ["", "Skeleton", f"  {len(a.skeleton.assessments)} bones assessed, {len(inferred)} mapped by analysis "
              "(names said nothing)"]
    for c in a.skeleton.conflicts:
        lines.append(f"  CONFLICT {c}")
    lines += ["", "Hands"]
    for side in ("left", "right"):
        h = a.hand(side)
        if h is None:
            lines.append(f"  {side}: no hand")
        elif h.state == BONES_PRESENT:
            lines.append(f"  {side}: {len(h.fingers)} finger chains mapped, confidence {h.confidence:.2f}")
        else:
            lines.append(f"  {side}: {h.state}")
    lines += ["", "Face"]
    if a.face is None:
        lines.append("  no face analysis (not enough facial geometry)")
    else:
        lines.append(f"  quality {a.face.quality:.2f}, {len(a.face.landmarks)} landmarks, "
                     f"mouth via {', '.join(a.mouth.approaches) if a.mouth and a.mouth.approaches else 'nothing'}")
        lines.append(f"  eyes from {a.face.eye_source}; lip seam from {a.face.mouth_source} "
                     f"(candidates: {', '.join(f'{c.source} {c.confidence:.2f}' for c in a.face.mouth_candidates)})")
    lines += ["", "Hair"]
    if a.hair is None or (not a.hair.meshes and not a.hair.chains):
        lines.append("  none detected")
    else:
        for x in a.hair.meshes:
            if x.classification != "NOT_HAIR":
                where = f"{x.mesh}/{x.material}" if x.material else x.mesh
                lines.append(f"  surface {where}: {x.classification} {x.confidence:.2f} ({x.hair_type})")
        lines.append(f"  bone chains: {len(a.hair.chains)}" + (f" ({a.hair.classification})" if a.hair.classification else ""))
    lines += ["", "Materials"]
    for mat in a.materials:
        lines.append(f"  {mat.material}: {mat.semantic} {mat.confidence:.2f}")
    if a.warnings:
        lines += ["", "Warnings"] + [f"  [{w.code}] {w.message}" for w in a.warnings]
    return "\n".join(lines)
