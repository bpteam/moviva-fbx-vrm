"""Eye and jaw bones from what they carry (pure).

Names aside, an eye bone is recognisable by its skin: a small, compact cluster of vertices in front of the head,
mirrored by another at the same height on the other side. A jaw bone carries the lower face: a large region in
front, below the eyes and around the midline, reaching down to the chin. Only bones below the head are considered,
only when the name resolver left the slot empty, and only at confidence that the mapping gate accepts when the
evidence is clean.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ArmatureInfo, CharacterInspection
from .frame import AnatomicalFrame
from .vec import Vec3


@dataclass
class _Cluster:
    bone: str
    count: int
    centre: Vec3          # local (left, up, forward)
    extent: float         # largest local extent
    top: float
    bottom: float


def _clusters(inspection: CharacterInspection, armature: ArmatureInfo, frame: AnatomicalFrame, bones: set[str],
              skip: dict[str, set[int]]) -> dict[str, _Cluster]:
    pts: dict[str, list[Vec3]] = {}
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature.name) or not mesh.vertex_weights:
            continue
        hair = skip.get(mesh.name, set())
        for i, ws in mesh.vertex_weights.items():
            if i in hair or i >= len(mesh.vertex_positions):
                continue
            b, w = max(ws.items(), key=lambda kv: kv[1])
            if w >= 0.5 and b in bones:
                pts.setdefault(b, []).append(frame.to_local(mesh.vertex_positions[i]))
    out: dict[str, _Cluster] = {}
    for b, ps in pts.items():
        n = len(ps)
        c = (sum(p[0] for p in ps) / n, sum(p[1] for p in ps) / n, sum(p[2] for p in ps) / n)
        ext = max(max(p[k] for p in ps) - min(p[k] for p in ps) for k in range(3))
        out[b] = _Cluster(b, n, c, ext, max(p[1] for p in ps), min(p[1] for p in ps))
    return out


def infer_face_bones(inspection: CharacterInspection, armature: ArmatureInfo, frame: AnatomicalFrame,
                     humanoid: dict[str, str], hair_vertices: dict[str, set[int]] | None = None,
                     ) -> dict[str, tuple[str, float, dict[str, float]]]:
    """``{vrm_bone: (bone, confidence, evidence)}`` for ``leftEye`` / ``rightEye`` / ``jaw`` when missing."""
    head = humanoid.get("head")
    if head is None or armature.bone(head) is None:
        return {}
    used = set(humanoid.values())
    below = {b for b in armature.descendants(head) if b not in used}
    if not below:
        return {}
    clusters = _clusters(inspection, armature, frame, below | {head}, hair_vertices or {})
    base = frame.to_local(armature.bone(head).head)
    scale = max(frame.scale, 1e-9)
    out: dict[str, tuple[str, float, dict[str, float]]] = {}

    # --- eyes: a mirrored pair of compact clusters in front of the head joint
    if "leftEye" not in humanoid and "rightEye" not in humanoid:
        cands = [c for c in clusters.values() if c.bone != head and c.count >= 12 and c.extent < 0.05 * scale
                 and c.centre[2] > base[2] and c.centre[1] > base[1]]
        best = None
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a.centre[0] * b.centre[0] >= 0:
                    continue
                sep = abs(a.centre[0] - b.centre[0])
                height = 1.0 - min(1.0, abs(a.centre[1] - b.centre[1]) / (0.1 * sep + 1e-9))
                mirror = 1.0 - min(1.0, abs(abs(a.centre[0]) - abs(b.centre[0])) / (0.3 * sep + 1e-9))
                depth = 1.0 - min(1.0, abs(a.centre[2] - b.centre[2]) / (0.2 * sep + 1e-9))
                size = min(a.count, b.count) / max(a.count, b.count)
                score = 0.35 * height + 0.3 * mirror + 0.2 * depth + 0.15 * size
                if sep > 0.12 * scale:
                    continue
                if best is None or score > best[0]:
                    best = (score, a, b, {"height": height, "mirror": mirror, "depth": depth, "size": size})
        if best is not None and best[0] >= 0.85:
            _, a, b, ev = best
            left, right = (a, b) if a.centre[0] > 0 else (b, a)
            conf = round(min(0.9, 0.8 + 0.1 * (best[0] - 0.85) / 0.15), 4)
            out["leftEye"] = (left.bone, conf, ev)
            out["rightEye"] = (right.bone, conf, ev)

    # --- jaw: the bone that carries the lower face (front, below the eyes, around the midline, down to the chin)
    if "jaw" not in humanoid:
        eye_h = None
        eyes = [clusters[out[k][0]] for k in ("leftEye", "rightEye") if k in out] or \
            [clusters[humanoid[k]] for k in ("leftEye", "rightEye") if humanoid.get(k) in clusters]
        if eyes:
            eye_h = sum(e.centre[1] for e in eyes) / len(eyes)
        qualifying: list[_Cluster] = []
        for c in clusters.values():
            if c.bone == head or any(c.bone == v[0] for v in out.values()) or c.count < 60:
                continue
            if abs(c.centre[0] - base[0]) > 0.02 * scale or c.centre[2] <= base[2]:
                continue
            if eye_h is not None and c.top > eye_h:
                continue  # it reaches above the eyes: a facial root, not the jaw
            if eye_h is not None and c.bottom > base[1] - 0.2 * (eye_h - base[1]):
                continue  # the jaw reaches down to the chin, below the head joint (upper teeth do not)
            qualifying.append(c)
        # the jaw is the root of that part of the rig: teeth / tongue bones below it carry lower-face vertices too
        names = {c.bone for c in qualifying}
        roots = [c for c in qualifying if not any(a in names for a in armature.ancestors(c.bone))]
        best = max(roots, key=lambda c: c.count, default=None)
        if best is not None:
            out["jaw"] = (best.bone, 0.82, {"front": 1.0, "below_eyes": 1.0 if eye_h is not None else 0.5,
                                            "reaches_chin": 1.0})
    return out
