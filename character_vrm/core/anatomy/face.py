"""Phase 7: face analysis from evidence, with the canonical template as the last resort.

    SURFACE ROLES -> EYES -> CHIN / NOSE (midline profile) -> MOUTH (fused evidence) -> EYELIDS -> TEMPLATE for the rest

The template lives in a unit face box (``u`` in [-1, 1] across half the head width, ``v`` in [0, 1] from chin to head
top). Only features the geometry does not reveal (brows, cheeks, jaw corners, nostrils) come from it, aligned by a
2D similarity to the anchors that *were* measured (chin, eyes, mouth). Every measured landmark records its source and
the competing candidates, so a report shows *why* the mouth is where it is.

Evidence, strongest first:

- **eyes**: the eye geometry the eye bones carry (or the eye-material surfaces), never the eye joints alone: VRoid
  places ``J_Adj_*_FaceEye`` well inside the head;
- **mouth**: (1) an existing mouth-opening morph (the lip seam is where it stops pulling vertices down), (2) the
  skin's own opening at the lips (a small closed boundary loop on the midline), (3) the jaw bone's skinning (lower
  lip on the jaw, upper lip not), (4) the mouth interior / teeth surfaces, (5) the concavity between the lips on the
  midline profile, (6) the template;
- **eyelids**: an existing blink morph, the eye opening loop in the skin, the visible eye surface.

A landmark that lands behind the front half of the head is capped at low confidence: a mouth on the back of the
skull must never pass a generation gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import ArmatureInfo, CharacterInspection, MeshInfo
from .confidence import AnalysisSource, AnalysisWarning, fuse
from .eyes import eye_extents, eyelids, find_eyes
from .facegeo import CANONICAL_FACE, FACE_LANDMARKS, FaceGeometry, FaceMorphs, FacePoint, pct
from .frame import AnatomicalFrame
from .landmarks import Landmark
from .mouth import MOUTH_WIDTH_PER_IOD, MouthEstimate, fuse_mouth, morph_opening, mouth_candidates, step_split
from .surfaces import BROW, EYE, LASH, SKIN, TEETH, vertex_roles
from .vec import Vec3

__all__ = ["CANONICAL_FACE", "FACE_LANDMARKS", "FaceAnalysis", "FaceGeometry", "FaceMorphs", "FacePoint",
           "MouthEstimate", "analyze_face", "apply_similarity", "collect_face_points", "fit_similarity_2d",
           "mesh_role", "step_split"]

_MIN_FACE_POINTS = 150
_MIN_DETECTED_FACE_POINTS = 30  # with MediaPipe's landmarks the geometry only has to carry the deformation
DETECTED_CONFIDENCE = 0.8


@dataclass
class FaceAnalysis:
    landmarks: dict[str, Landmark] = field(default_factory=dict)  # world space
    quality: float = 0.0
    head_width: float = 0.0
    head_height: float = 0.0
    mouth_width: float = 0.0
    fit: dict[str, float] = field(default_factory=dict)
    geometry: FaceGeometry = field(default_factory=FaceGeometry)
    eye_bones: bool = False
    jaw_bone: bool = False
    evidence: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[AnalysisWarning] = field(default_factory=list)
    mouth_source: str = "template"
    mouth_candidates: list[MouthEstimate] = field(default_factory=list)
    eye_source: str = "template"
    lip_seam: float | None = None       # local height
    jaw_weighted_lips: bool = False     # the lower lip rides on the jaw bone (use its weights as the lower-lip mask)
    jaw_pivot: Vec3 | None = None       # local position of the jaw joint (set by ``analyze_human``)
    aa_opening: float | None = None     # how far an existing ``aa`` pulls the lower lip down at the seam
    eye_extent: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)  # local x/y box per eye
    brow_bottom: dict[str, float] = field(default_factory=dict)  # local height of the brow geometry's lower edge

    def local(self, frame: AnatomicalFrame, name: str) -> Vec3 | None:
        lm = self.landmarks.get(name)
        return frame.to_local(lm.position) if lm else None


def mesh_role(mesh: MeshInfo) -> str:
    """Role of a whole mesh (kept for callers that have no per-polygon data)."""
    from .surfaces import slot_role
    return slot_role(mesh, None)


def fit_similarity_2d(src: list[tuple[float, float]], dst: list[tuple[float, float]],
                      allow_rotation: bool = True) -> tuple[float, float, tuple[float, float]] | None:
    """Least-squares similarity ``dst ~ s * R(theta) * src + t`` (Umeyama, 2D). Returns ``(s, theta, t)``."""
    n = len(src)
    if n < 2 or n != len(dst):
        return None
    cs = (sum(p[0] for p in src) / n, sum(p[1] for p in src) / n)
    cd = (sum(p[0] for p in dst) / n, sum(p[1] for p in dst) / n)
    a = b = var = 0.0
    for (sx, sy), (dx, dy) in zip(src, dst):
        sx, sy, dx, dy = sx - cs[0], sy - cs[1], dx - cd[0], dy - cd[1]
        a += sx * dx + sy * dy
        b += sx * dy - sy * dx
        var += sx * sx + sy * sy
    if var < 1e-12:
        return None
    theta = math.atan2(b, a) if allow_rotation else 0.0
    if allow_rotation:
        s = math.hypot(a, b) / var
    else:
        s = a / var
    c, si = math.cos(theta), math.sin(theta)
    t = (cd[0] - s * (c * cs[0] - si * cs[1]), cd[1] - s * (si * cs[0] + c * cs[1]))
    return s, theta, t


def apply_similarity(fit: tuple[float, float, tuple[float, float]], p: tuple[float, float]) -> tuple[float, float]:
    s, theta, t = fit
    c, si = math.cos(theta), math.sin(theta)
    return (s * (c * p[0] - si * p[1]) + t[0], s * (si * p[0] + c * p[1]) + t[1])


def collect_face_points(inspection: CharacterInspection, armature: ArmatureInfo, frame: AnatomicalFrame,
                        face_bones: set[str], head_base: Vec3, neck_level_margin: float = 0.0, *,
                        hair_slots: dict[str, set[int | None]] | None = None, jaw_bones: set[str] | None = None,
                        eye_bones: dict[str, str] | None = None) -> FaceGeometry:
    """Face search region: vertices weighted to the head *and every bone below it* (facial, jaw, eye, teeth ...
    bones: CC skins the face to ``CC_Base_FacialBone`` / ``UpperJaw``, not to the head), at or above the head base.

    Meshes without skin weights are included only when their name says head/face (never blind). Roles are per
    polygon (material slot); ``hair_slots`` marks the surfaces the hair analysis found."""
    geo = FaceGeometry()
    base_y = frame.to_local(head_base)[1] - neck_level_margin
    jaw_bones = jaw_bones or set()
    eye_bones = eye_bones or {}
    hair_slots = hair_slots or {}
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature.name) or not mesh.vertex_positions:
            continue
        named = any(t in mesh.name.lower() for t in ("head", "face", "eye", "teeth", "tongue", "mouth", "lash", "brow"))
        weighted = bool(mesh.vertex_weights)
        if not weighted and not named:
            continue
        roles = vertex_roles(mesh, hair_slots.get(mesh.name))
        used = False
        for i, pos in enumerate(mesh.vertex_positions):
            jaw = 0.0
            side = None
            if weighted:
                ws = mesh.vertex_weights.get(i, {})
                w = sum(v for b, v in ws.items() if b in face_bones)
                if w < 0.5:
                    continue
                jaw = sum(v for b, v in ws.items() if b in jaw_bones)
                for s, bone in eye_bones.items():
                    if ws.get(bone, 0.0) >= 0.5:
                        side = s
            local = frame.to_local(pos)
            if local[1] < base_y:
                continue
            role = roles.get(i, SKIN)
            if side is not None and role == SKIN:
                role = EYE  # an eyeball the materials did not name
            geo.points.append(FacePoint(mesh.name, i, local, role, jaw, side))
            used = True
        if used:
            geo.meshes[mesh.name] = mesh
    return geo


def _profile_chin(skin: list[FacePoint], head_width: float, y_lo: float, y_hi: float, x_mid: float = 0.0) -> float:
    """Chin height from the midline front profile.

    Walk down the midline from the mouth band: the chin is the lowest level whose front surface still stands on the
    face (within 0.35 head widths of the front at mouth height). Empty levels and isolated inner points (the mouth
    interior shows through a thin strip) are skipped; three failing levels in a row mean the profile has turned back
    under the chin."""
    strip = [p.local for p in skin if abs(p.local[0] - x_mid) < 0.06 * head_width and y_lo <= p.local[1] <= y_hi]
    if not strip:
        return y_lo
    bins = 48
    span = max(y_hi - y_lo, 1e-9)
    front: list[float | None] = [None] * bins
    for x, y, z in strip:
        k = min(bins - 1, int((y - y_lo) / span * bins))
        if front[k] is None or z > front[k]:
            front[k] = z
    ref_band = [f for f in front[int(0.25 * bins):int(0.5 * bins)] if f is not None]
    if not ref_band:
        return min(p[1] for p in strip)
    ref = sorted(ref_band)[len(ref_band) // 2]
    threshold = ref - 0.35 * head_width
    chin_k = int(0.35 * bins)
    misses = 0
    for k in range(int(0.35 * bins), -1, -1):
        f = front[k]
        if f is None:
            continue
        if f >= threshold:
            chin_k, misses = k, 0
        else:
            misses += 1
            if misses >= 3:
                break
    return y_lo + chin_k / bins * span


class _HeightField:
    def __init__(self, points: list[FacePoint], cell: float) -> None:
        self.cell = cell
        self.z: dict[tuple[int, int], float] = {}
        for p in points:
            x, y, z = p.local
            k = (round(x / cell), round(y / cell))
            if k not in self.z or z > self.z[k]:
                self.z[k] = z

    def sample(self, x: float, y: float, radius: float) -> tuple[float | None, float]:
        """(front-most z within ``radius``, fill fraction of the disc)."""
        r = max(1, math.ceil(radius / self.cell))
        cx, cy = round(x / self.cell), round(y / self.cell)
        best, filled, total = None, 0, 0
        for ix in range(cx - r, cx + r + 1):
            for iy in range(cy - r, cy + r + 1):
                if (ix - cx) ** 2 + (iy - cy) ** 2 > r * r:
                    continue
                total += 1
                z = self.z.get((ix, iy))
                if z is not None:
                    filled += 1
                    best = z if best is None or z > best else best
        return best, (filled / total if total else 0.0)


# ---------------------------------------------------------------- analysis

def analyze_face(frame: AnatomicalFrame, geometry: FaceGeometry, head_base: Vec3, head_top: Vec3 | None,
                 eye_positions: dict[str, Vec3] | None = None, jaw_bone: bool = False,
                 morphs: FaceMorphs | None = None,
                 detected: dict[str, Vec3] | None = None) -> FaceAnalysis | None:
    """``detected``: landmarks found on a rendered view (MediaPipe), world space. They replace only what the
    geometry gave weakly (a template guess, low confidence), and let a sparse low-poly face be analysed at all."""
    skin = geometry.skin()
    result = FaceAnalysis(geometry=geometry, jaw_bone=jaw_bone, eye_bones=bool(eye_positions))
    if len(skin) < (_MIN_DETECTED_FACE_POINTS if detected else _MIN_FACE_POINTS):
        return None
    morphs = morphs or FaceMorphs()
    xs = [p.local[0] for p in skin]
    ys = [p.local[1] for p in skin]
    zs = [p.local[2] for p in skin]
    head_width = pct(xs, 0.99) - pct(xs, 0.01)
    y_top = max(ys)
    y_bottom = min(ys)
    if head_width <= 1e-9 or y_top <= y_bottom:
        return None
    z_min, z_max = min(zs), max(zs)
    depth = max(z_max - z_min, 1e-9)
    by_key = {(p.mesh, p.index): p for p in geometry.points}

    # ---- eyes: eye geometry first, eye joints second
    eyes, eye_source = find_eyes(geometry, eye_positions, frame, head_width)
    result.eye_source = eye_source
    x_mid = 0.5 * (eyes["left"][0] + eyes["right"][0]) if len(eyes) == 2 else 0.5 * (pct(xs, 0.99) + pct(xs, 0.01))
    eye_y = sum(e[1] for e in eyes.values()) / len(eyes) if eyes else None
    iod = abs(eyes["left"][0] - eyes["right"][0]) if len(eyes) == 2 else None

    # ---- chin (front profile below the eyes) and the face scale
    chin_hi = eye_y if eye_y is not None else y_top
    chin_y = _profile_chin(skin, head_width, y_bottom, chin_hi, x_mid)
    if eye_y is not None and eye_y - chin_y < 0.15 * head_width:
        chin_y = y_bottom
    face_h = y_top - chin_y
    if eye_y is not None:
        face_h = max(face_h, (eye_y - chin_y) / CANONICAL_FACE["left_eye"][1])
    if face_h <= 0.3 * head_width:
        result.warnings.append(AnalysisWarning("face-degenerate", "face box is degenerate", "face"))
        return None
    span = (eye_y - chin_y) if eye_y is not None else CANONICAL_FACE["left_eye"][1] * face_h
    eye_line = chin_y + span
    # mouth-interior surfaces up at the eyes are the atlas' eyeliner, lashes and brows (older VRoid puts them all in
    # its "FaceMouth" material): lashes up to the top of the eye, brows above it
    result.eye_extent = eye_extents(geometry, eyes, x_mid)
    eye_top = max((b[3] for b in result.eye_extent.values()), default=eye_line + 0.25 * span)
    for p in geometry.points:
        if p.role == TEETH and p.local[1] > chin_y + 0.8 * span:
            p.role = BROW if p.local[1] > eye_top + 0.05 * span else LASH
    result.head_width, result.head_height = head_width, face_h
    result.fit = {"chinY": chin_y, "topY": y_top, "eyeY": eye_line}

    front_pts = [p for p in skin if p.local[2] >= z_min + 0.4 * depth]
    spacing = math.sqrt(head_width * face_h / max(len(front_pts), 1))
    cell = max(face_h / 60.0, 1.6 * spacing, 1e-6)
    field_ = _HeightField(skin, cell)
    radius = 0.09 * face_h

    def surface_z(x: float, y: float) -> tuple[float | None, float]:
        z, fill = field_.sample(x, y, 0.35 * radius)
        if z is None:
            z, fill = field_.sample(x, y, radius)
        else:
            fill = field_.sample(x, y, radius)[1]
        return z, fill

    # ---- nose tip: front-most midline point between the mouth band and the eyes
    nose = None
    cand = [p.local for p in skin if abs(p.local[0] - x_mid) < 0.08 * head_width
            and chin_y + 0.42 * span <= p.local[1] <= chin_y + 0.95 * span]
    if cand:
        best = max(cand, key=lambda q: q[2])
        nose = (x_mid, best[1], best[2])

    # ---- mouth: every source that has something to say, then fused
    candidates = mouth_candidates(skin, geometry, by_key, frame, morphs, x_mid, chin_y, span, head_width, depth,
                                   z_max, nose, iod, jaw_bone)
    mouth = fuse_mouth(candidates, chin_y, span)
    result.mouth_candidates = candidates
    result.mouth_source = mouth.source
    result.lip_seam = mouth.seam
    # the jaw rotation needs the pivot and the weights, not the seam: a morph that places the seam elsewhere (a
    # Character Creator "aa" carrying the chin-shaping Jaw_Open corrective) does not make the skinning wrong
    result.jaw_weighted_lips = any(c.source == "jaw-skinning" for c in candidates)
    half = mouth.half_width
    if half is None:
        half = 0.5 * MOUTH_WIDTH_PER_IOD * iod if iod else CANONICAL_FACE["mouth_left"][0] * head_width / 2
    result.mouth_width = 2 * half
    if morphs.get("aa"):
        result.aa_opening = morph_opening(skin, frame, morphs.get("aa"), mouth.center_x, mouth.seam, half, span,
                                          z_max, depth)

    # ---- template alignment for the features no evidence names (brows, cheeks, jaw corners, nostrils)
    scale_x = head_width / 2 / face_h
    src = [(0.0, 0.0)]
    dst = [(x_mid, chin_y)]
    for s in ("left", "right"):
        if s in eyes:
            u, v = CANONICAL_FACE[f"{s}_eye"]
            src.append((u * scale_x, v))
            dst.append((eyes[s][0], eyes[s][1]))
    src.append((0.0, CANONICAL_FACE["mouth_center"][1]))
    dst.append((mouth.center_x, mouth.seam))
    if len(eyes) < 2:
        src.append((0.0, 1.0))
        dst.append((x_mid, chin_y + face_h))
    fit = fit_similarity_2d(src, dst, allow_rotation=len(eyes) == 2)
    if fit is None:
        fit = (face_h, 0.0, (x_mid, chin_y))
    result.evidence["fit"] = {"eye_bones": 1.0 if eye_source == "eye-bone" else 0.0,
                              "eye_geometry": 1.0 if eye_source == "eye-geometry" else 0.0}

    def template_point(name: str) -> tuple[float, float]:
        u, v = CANONICAL_FACE[name]
        return apply_similarity(fit, (u * scale_x, v))

    placed: dict[str, tuple[float, float, str, float]] = {}  # name -> (x, y, method, evidence confidence)
    for name in FACE_LANDMARKS:
        x, y = template_point(name)
        placed[name] = (x, y, "template-fit", 0.5)

    # measured features override the template
    placed["chin"] = (x_mid, chin_y, "profile", 0.8)
    if nose is not None:
        placed["nose_tip"] = (nose[0], nose[1], "template-fit+profile-refined", 0.85)
    mconf = mouth.confidence
    mtag = f"mouth:{mouth.source}"
    placed["mouth_center"] = (mouth.center_x, mouth.seam, mtag, mconf)
    placed["mouth_left"] = (mouth.center_x + half, mouth.seam, mtag, mconf)
    placed["mouth_right"] = (mouth.center_x - half, mouth.seam, mtag, mconf)
    lip_h = max(0.06 * span, 0.35 * half)
    placed["upper_lip"] = (mouth.center_x, mouth.seam + 0.5 * lip_h, mtag, mconf)
    placed["lower_lip"] = (mouth.center_x, mouth.seam - 0.5 * lip_h, mtag, mconf)
    lids = eyelids(geometry, by_key, frame, morphs, eyes, head_width, span, x_mid)
    lid_z: dict[str, float] = {}
    for s, (cx, up, low, method, conf, uz) in lids.items():
        if uz is not None:
            lid_z[f"{s}_upper_eyelid"] = uz
        placed[f"{s}_upper_eyelid"] = (cx, up, method, conf)
        placed[f"{s}_lower_eyelid"] = (cx, low, method, conf)
        # brows: the brow surfaces when there are any, else the template offset above the measured lid
        sign = 1.0 if s == "left" else -1.0
        brow = sorted(p.local[1] for p in geometry.points if p.role == BROW and (p.local[0] - x_mid) * sign > 0
                      and abs(p.local[0] - cx) < 0.2 * head_width and p.local[1] > up)
        if len(brow) >= 4:
            placed[f"{s}_eyebrow"] = (cx, brow[len(brow) // 2], "brow-geometry", 0.85)
            result.brow_bottom[s] = brow[0]
        else:
            bx, by, _, _ = placed[f"{s}_eyebrow"]
            placed[f"{s}_eyebrow"] = (cx, max(by, up + 0.06 * span), "template-fit", 0.5)

    landmarks: dict[str, Landmark] = {}
    per: dict[str, dict[str, float]] = {}
    front_plane = z_min + 0.5 * depth
    for name, (x, y, method, econf) in placed.items():
        if name in ("left_eye", "right_eye") and name[:-4] in eyes:
            ex, ey, ez = eyes[name[:-4]]
            landmarks[name] = Landmark(name, (ex, ey, ez), 0.0,
                                       AnalysisSource.SKELETON if eye_source == "eye-bone" else AnalysisSource.GEOMETRY,
                                       eye_source)
            per[name] = {"surface": 1.0, "evidence": 0.95 if eye_source == "eye-geometry" else 0.85, "front": 1.0}
            continue
        z, fill = surface_z(x, y)
        if z is None:
            continue
        z = lid_z.get(name, z)
        front = 1.0 if z >= front_plane else 0.0
        per[name] = {"surface": min(1.0, fill * 1.4), "evidence": econf, "front": front}
        landmarks[name] = Landmark(name, (x, y, z), 0.0, AnalysisSource.GEOMETRY, method)
    if "nose_tip" in per:
        per["nose_tip"]["profile"] = min(1.0, max(0.0, (landmarks["nose_tip"].position[2] - z_min) / depth))

    # symmetry between mirrored landmarks (height and depth must agree)
    for name in list(landmarks):
        if name.startswith("left_") and "right_" + name[5:] in landmarks:
            a, b = landmarks[name].position, landmarks["right_" + name[5:]].position
            ok = 1.0 - min(1.0, (abs(a[1] - b[1]) + abs(a[2] - b[2])) / (0.08 * face_h))
            per[name]["symmetry"] = per["right_" + name[5:]]["symmetry"] = ok
    # vertical order of the midline (chin < mouth < nose < eyes); a violation is strong counter-evidence
    order = [n for n in ("chin", "mouth_center", "nose_tip") if n in landmarks]
    heights = [landmarks[n].position[1] for n in order] + ([eye_line] if eyes else [])
    ordered = all(a < b for a, b in zip(heights, heights[1:]))
    weights = {"surface": 0.3, "evidence": 0.35, "symmetry": 0.15, "profile": 0.1, "front": 0.1}
    core_conf: list[float] = []
    for name, lm in landmarks.items():
        ev = per[name]
        c = fuse(ev, weights, veto_below=0.15, veto_cap=0.2)
        if ev.get("front", 1.0) < 0.5:
            c = min(c, 0.2)  # a face landmark on the back half of the head is wrong, whatever else agrees
        if not ordered and name in order:
            c = min(c, 0.4)
        lm.confidence = round(c, 4)
        result.evidence[name] = {k: round(v, 3) for k, v in ev.items()}
        if name in ("chin", "mouth_center", "nose_tip", "left_eye", "right_eye", "left_upper_eyelid",
                    "right_upper_eyelid"):
            core_conf.append(c)
    result.evidence["mouth"] = {c.source: round(c.confidence, 3) for c in candidates}
    for name, lm in landmarks.items():
        result.landmarks[name] = Landmark(name, frame.to_world(lm.position), lm.confidence, lm.source, lm.method)
    for name, pos in (detected or {}).items():
        old = result.landmarks.get(name)
        if old is None or "template" in old.method or old.confidence < DETECTED_CONFIDENCE:
            result.landmarks[name] = Landmark(name, tuple(pos), DETECTED_CONFIDENCE, AnalysisSource.GEOMETRY,
                                              "mediapipe")
    if detected and {"mouth_left", "mouth_right"} <= set(detected):
        ml, mr = frame.to_local(detected["mouth_left"]), frame.to_local(detected["mouth_right"])
        result.mouth_width = max(result.mouth_width, abs(ml[0] - mr[0]))
        if "mouth_center" not in detected and {"upper_lip", "lower_lip"} <= set(detected):
            u, b = detected["upper_lip"], detected["lower_lip"]
            result.landmarks["mouth_center"] = Landmark("mouth_center", tuple((u[i] + b[i]) / 2 for i in range(3)),
                                                        DETECTED_CONFIDENCE, AnalysisSource.GEOMETRY, "mediapipe")
    if detected:
        core_conf = [result.landmarks[n].confidence for n in ("chin", "mouth_center", "nose_tip", "left_eye", "right_eye",
                     "left_upper_eyelid", "right_upper_eyelid") if n in result.landmarks]
    result.quality = round(sum(core_conf) / len(core_conf), 4) if core_conf else 0.0
    if not ordered:
        result.warnings.append(AnalysisWarning("face-order", "chin / mouth / nose / eyes are not in anatomical order",
                                               "face"))
    if mouth.source == "template":
        result.warnings.append(AnalysisWarning("mouth-template", "no geometric evidence for the mouth: the mouth "
                                               "position is the template's guess", "face"))
    if len(landmarks) < 0.7 * len(FACE_LANDMARKS):
        result.warnings.append(AnalysisWarning("face-partial", "only part of the face template found surface support",
                                               "face"))
    return result
