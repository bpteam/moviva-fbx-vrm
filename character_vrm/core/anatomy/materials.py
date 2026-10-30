"""Phase 16: semantic material classification. Metadata only: shaders are never modified."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import CharacterInspection, MaterialInfo, MeshInfo
from .confidence import DEFAULT_THRESHOLDS, AnalysisSource
from .frame import AnatomicalFrame
from .landmarks import Landmark
from .vec import Vec3, mean

SEMANTICS = ("skin", "hair", "eye", "cornea", "eyelash", "eyebrow", "teeth", "tongue", "mouth_interior", "clothes", "other")

_TOKENS: dict[str, tuple[str, ...]] = {
    "skin": ("skin", "body", "flesh", "epiderm", "face", "head", "torso", "nude"),
    "hair": ("hair", "bang", "fringe", "ponytail", "braid", "scalp", "strand"),
    "eye": ("eye", "iris", "sclera", "eyeball", "pupil"),
    "cornea": ("cornea", "tearline", "eyewet", "lens", "eyegloss", "eyeshell"),
    "eyelash": ("lash",),
    "eyebrow": ("brow",),
    "teeth": ("teeth", "tooth", "gum"),
    "tongue": ("tongue",),
    "mouth_interior": ("mouth", "oral", "cavity", "innermouth"),
    "clothes": ("cloth", "shirt", "dress", "jacket", "pants", "skirt", "shoe", "sock", "glove", "coat", "suit",
                "hoodie", "jean", "sleeve", "boot", "sweater", "top", "trouser", "uniform"),
}
_RELIABILITY = {"material_name": 0.85, "mesh_name": 0.6, "texture_name": 0.5, "position": 0.55, "transparency": 0.35}


@dataclass
class MaterialSemantic:
    material: str
    semantic: str
    confidence: float
    source: AnalysisSource = AnalysisSource.MATERIAL
    evidence: dict[str, float] = field(default_factory=dict)
    meshes: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else " " for c in s).replace(" ", "")


def _hits(name: str, tokens: tuple[str, ...]) -> bool:
    n = _norm(name)
    return any(t in n for t in tokens)


_SPECIFIC = ("eyelash", "eyebrow", "cornea", "teeth", "tongue", "hair")
_GENERIC = ("skin", "eye", "mouth_interior", "clothes")


def _readings(name: str) -> set[str]:
    """Semantics a name suggests; a specific reading ("Eyelashes") suppresses the generic one it contains ("eye")."""
    found = {sem for sem, tokens in _TOKENS.items() if _hits(name, tokens)}
    if found & set(_SPECIFIC):
        found -= set(_GENERIC)
    return found


def _position_scores(meshes: list[MeshInfo], frame: AnatomicalFrame | None, landmarks: dict[str, Landmark],
                     head_height: float | None) -> dict[str, float]:
    """Anatomical-position evidence: where the material's vertices sit relative to the face landmarks."""
    out: dict[str, float] = {}
    if frame is None or not head_height:
        return out
    pts: list[Vec3] = [frame.to_local(p) for m in meshes for p in m.vertex_positions]
    if len(pts) < 4:
        return out
    c = mean(pts)
    assert c is not None
    extent_y = max(p[1] for p in pts) - min(p[1] for p in pts)
    head_top = landmarks.get("head_top")
    top_y = frame.to_local(head_top.position)[1] if head_top else None
    if extent_y > 2.2 * head_height:
        out["skin"] = 0.5
        out["clothes"] = 0.5
    def near(name: str, radius: float) -> bool:
        lm = landmarks.get(name)
        if lm is None:
            return False
        l = frame.to_local(lm.position)
        return abs(c[0] - l[0]) < radius and abs(c[1] - l[1]) < radius and abs(c[2] - l[2]) < 1.5 * radius
    small = extent_y < 0.6 * head_height
    if small and (near("left_eye", 0.25 * head_height) or near("right_eye", 0.25 * head_height)):
        out["eye"] = 0.8
        out["eyelash"] = 0.4
    if small and near("mouth_center", 0.22 * head_height):
        out["teeth"] = 0.6
        out["mouth_interior"] = 0.5
        out["tongue"] = 0.4
    if top_y is not None and c[1] > top_y - 0.45 * head_height and extent_y < 2.0 * head_height:
        out["hair"] = max(out.get("hair", 0.0), 0.55)
    return out


def classify_materials(inspection: CharacterInspection, frame: AnatomicalFrame | None = None,
                       landmarks: dict[str, Landmark] | None = None, head_height: float | None = None,
                       threshold: float = DEFAULT_THRESHOLDS.material_semantic) -> list[MaterialSemantic]:
    landmarks = landmarks or {}
    users: dict[str, list[MeshInfo]] = {}
    for mesh in inspection.meshes:
        for mat in mesh.materials:
            users.setdefault(mat, []).append(mesh)
    out: list[MaterialSemantic] = []
    for mat in inspection.materials:
        meshes = users.get(mat.name, [])
        evidence: dict[str, dict[str, float]] = {k: {} for k in SEMANTICS}
        for sem in _readings(mat.name):
            evidence[sem]["material_name"] = 1.0
        for m in meshes:
            for sem in _readings(m.name):
                evidence[sem]["mesh_name"] = 1.0
        for image in mat.images:
            for sem in _readings(image):
                evidence[sem]["texture_name"] = 1.0
        for sem, score in _position_scores(meshes, frame, landmarks, head_height).items():
            evidence[sem]["position"] = score
        if mat.alpha_mode in ("BLEND", "MASK", "CLIP"):
            for sem in ("hair", "eyelash", "cornea", "eyebrow"):
                if evidence[sem]:
                    evidence[sem]["transparency"] = 1.0
            if not any(evidence[s] for s in ("hair", "eyelash", "cornea", "eyebrow")):
                pass  # transparency alone proves nothing
        scored: dict[str, float] = {}
        for sem, ev in evidence.items():
            if not ev:
                continue
            miss = 1.0
            for channel, v in ev.items():
                miss *= 1.0 - _RELIABILITY[channel] * v
            scored[sem] = 1.0 - miss
        if not scored:
            out.append(MaterialSemantic(mat.name, "other", 0.3, evidence={}, meshes=[m.name for m in meshes]))
            continue
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        best, conf = ranked[0]
        # a competing reading lowers certainty, unless the material's own name settles it and the rival only comes
        # from where it sits (a cornea material on the eye mesh, a scalp material on the hair mesh)
        if len(ranked) > 1 and ranked[1][1] >= 0.5 and not (
                "material_name" in evidence[best] and "material_name" not in evidence[ranked[1][0]]):
            conf *= 1.0 - 0.5 * ranked[1][1]
        sem = best if conf >= threshold else "other"
        sources = AnalysisSource.MATERIAL if any(k in evidence[best] for k in ("material_name", "texture_name")) \
            else AnalysisSource.GEOMETRY
        out.append(MaterialSemantic(mat.name, sem, round(conf, 4), sources,
                                    {k: round(v, 3) for k, v in evidence[best].items()}, [m.name for m in meshes]))
    return out
