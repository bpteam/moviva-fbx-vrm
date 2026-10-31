"""``HumanModelAnalysis``: the single canonical semantic description every later stage consumes.

Pure data: no ``bpy`` objects, no scene handles. Built by ``analyze_human`` from a ``CharacterInspection`` and the
mapped humanoid; nothing here mutates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..model import ArmatureInfo, CharacterInspection, MeshInfo
from .confidence import DEFAULT_THRESHOLDS, AnalysisSource, AnalysisWarning, Thresholds
from ..expressions import canon
from .face import FaceAnalysis, FaceGeometry, FaceMorphs, analyze_face, collect_face_points
from .frame import AnatomicalFrame, build_frame
from .hair import HairAnalysis, analyze_hair
from .hands import (
    BONES_PRESENT, FUSED, GEOMETRY_SEPARABLE, NO_EVIDENCE, HandAnalysis, classify_finger_chains, compare_hands,
    segment_fingers,
)
from .landmarks import Landmark, build_body_landmarks
from .materials import MaterialSemantic, classify_materials
from .measurements import HumanMeasurements, Measurement, measure_body
from .regions import MouthAnalysis, analyze_mouth
from .facebones import infer_face_bones
from .skeleton import BoneAssessment, SkeletonAnalysis, assess_skeleton
from .vec import Vec3, dist, dot, normalize, sub

__all__ = ["HumanModelAnalysis", "analyze_human"]


@dataclass
class HumanModelAnalysis:
    frame: AnatomicalFrame
    measurements: HumanMeasurements
    landmarks: dict[str, Landmark]
    skeleton: SkeletonAnalysis
    left_hand: HandAnalysis | None
    right_hand: HandAnalysis | None
    face: FaceAnalysis | None
    hair: HairAnalysis | None
    materials: list[MaterialSemantic]
    warnings: list[AnalysisWarning] = field(default_factory=list)
    # Supporting semantic data (world-space positions of mapped joints, the final humanoid map).
    joints: dict[str, Vec3] = field(default_factory=dict)
    humanoid: dict[str, str] = field(default_factory=dict)
    inferred_bones: dict[str, str] = field(default_factory=dict)  # mapped by analysis, not by the name resolver
    mouth: MouthAnalysis | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS

    def hand(self, side: str) -> HandAnalysis | None:
        return self.left_hand if side == "left" else self.right_hand

    @property
    def head_height(self) -> float | None:
        return self.measurements.get("head_height")

    @property
    def head_width(self) -> float | None:
        return self.measurements.get("head_width")

    def landmark_level(self, name: str) -> float | None:
        lm = self.landmarks.get(name)
        return self.frame.height_of(lm.position) if lm else None


def _joints(armature: ArmatureInfo, humanoid: dict[str, str]) -> dict[str, Vec3]:
    joints: dict[str, Vec3] = {}
    for vrm, src in humanoid.items():
        bone = armature.bone(src)
        if bone is not None:
            joints[vrm] = bone.head
    for side in ("left", "right"):
        hand = armature.bone(humanoid.get(f"{side}Hand", ""))
        if hand is not None:
            far = [hand.tail] + [armature.bone(d).tail for d in armature.descendants(hand.name) if armature.bone(d)]
            joints[f"{side}HandTip"] = max(far, key=lambda p: dist(p, hand.head))
        toes = armature.bone(humanoid.get(f"{side}Toes", ""))
        if toes is not None:
            joints[f"{side}ToesTip"] = toes.tail
    return joints


def _weight_of(mesh: MeshInfo, vertex: int, bones: set[str]) -> float:
    return sum(w for b, w in mesh.vertex_weights.get(vertex, {}).items() if b in bones)


MORPH_EXPRESSIONS = ("aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight")


def face_morphs(inspection: CharacterInspection, binds: dict[str, list[tuple[str, str, float]]]) -> FaceMorphs:
    """World-space deltas of the bound morphs (weighted sum over the binds), plus a ``jawOpen`` key if present."""
    keys = {(k.mesh, k.name): k for k in inspection.shape_keys if k.deltas}
    out = FaceMorphs()
    for expr in MORPH_EXPRESSIONS:
        acc: dict[tuple[str, int], Vec3] = {}
        for mesh, key, weight in binds.get(expr, []):
            info = keys.get((mesh, key))
            if info is None:
                continue
            for i, d in info.deltas.items():
                o = acc.get((mesh, i), (0.0, 0.0, 0.0))
                acc[(mesh, i)] = (o[0] + weight * d[0], o[1] + weight * d[1], o[2] + weight * d[2])
        if acc:
            out.deltas[expr] = acc
    for (mesh, name), info in keys.items():
        if canon(name) == "jawopen":
            jaw = out.deltas.setdefault("jawOpen", {})
            for i, d in info.deltas.items():
                jaw[(mesh, i)] = d
    return out


def _head_size_estimate(inspection: CharacterInspection, armature: ArmatureInfo, frame: AnatomicalFrame,
                        head_bones: set[str], head_base: Vec3) -> tuple[float, float] | None:
    """(height, width) of everything the head carries, hair included: only used to scale the hair analysis."""
    base = frame.to_local(head_base)[1]
    xs: list[float] = []
    ys: list[float] = []
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature.name) or not mesh.vertex_weights:
            continue
        for i, ws in mesh.vertex_weights.items():
            if sum(w for b, w in ws.items() if b in head_bones) >= 0.5 and i < len(mesh.vertex_positions):
                loc = frame.to_local(mesh.vertex_positions[i])
                if loc[1] >= base - 0.06 * frame.scale:
                    xs.append(loc[0])
                    ys.append(loc[1])
    if len(xs) < 50:
        return None
    xs.sort()
    ys.sort()
    width = xs[int(0.98 * (len(xs) - 1))] - xs[int(0.02 * (len(xs) - 1))]
    height = ys[int(0.99 * (len(ys) - 1))] - (base - 0.06 * frame.scale)
    return (max(height, 1e-6), max(width, 1e-6))


def analyze_human(
    inspection: CharacterInspection, armature: ArmatureInfo, humanoid: Any,
    thresholds: Thresholds = DEFAULT_THRESHOLDS, waist_fallback_t: float = 0.72,
    existing_visemes: list[str] | None = None,
    expression_binds: dict[str, list[tuple[str, str, float]]] | None = None,
) -> HumanModelAnalysis | None:
    """``humanoid`` is a ``HumanoidResult`` (anything with ``mappings: {vrm: BoneMapping}``).

    ``expression_binds`` (VRM expression -> ``[(mesh, shape key, weight)]``) are the existing morphs the expression
    resolver found; their deltas locate the lip seam and the eyelids exactly.

    Returns ``None`` when no anatomical frame can be built (hips / head / a left-right pair missing)."""
    warnings: list[AnalysisWarning] = []
    weighted = inspection.weighted_bone_names(armature.name)
    names = {vrm: (m.source_bone, m.confidence) for vrm, m in humanoid.mappings.items()}
    skeleton = assess_skeleton(armature, weighted, names)
    warnings.extend(skeleton.warnings)
    for conflict in skeleton.conflicts:
        warnings.append(AnalysisWarning("skeleton-conflict", conflict, "skeleton"))

    # Existing mappings win; analysis fills gaps only when confident and the source bone is still free.
    final: dict[str, str] = {vrm: src for vrm, (src, _) in names.items()}
    inferred: dict[str, str] = {}
    used = set(final.values())
    for vrm, a in skeleton.assessments.items():
        if vrm in final or a.bone in used or a.confidence < thresholds.automatic_mapping:
            continue
        final[vrm] = a.bone
        inferred[vrm] = a.bone
        used.add(a.bone)

    joints = _joints(armature, final)
    points = dict(joints)
    head_bone = armature.bone(final.get("head", ""))
    if head_bone is not None:
        points["top"] = head_bone.tail
    lows = [p for k, p in joints.items() if k in ("leftToesTip", "rightToesTip")] + \
           [armature.bone(final[k]).tail for k in ("leftToes", "rightToes") if k in final and armature.bone(final[k])]
    frame, fw = build_frame(points, fallback_left_hint=(1.0, 0.0, 0.0))
    warnings.extend(fw)
    if frame is None:
        return None
    if lows:
        ground = min(dot(p, frame.up) for p in [*lows, *(joints[k] for k in ("leftFoot", "rightFoot") if k in joints)])
        frame.scale = max(dot(points["top"], frame.up) - ground, 1e-6) if "top" in points else frame.scale
    if skeleton.side_assumed:
        frame.confidence = min(frame.confidence, 0.55)
        warnings.append(AnalysisWarning("frame-sides-assumed", "left/right assumed from the +X convention: "
                                        "frame confidence lowered", "frame"))

    # ---- eye / jaw bones the names did not give, from what they carry
    for vrm, (bone, conf, ev) in infer_face_bones(inspection, armature, frame, final).items():
        if conf >= thresholds.automatic_mapping and bone not in used:
            final[vrm] = inferred[vrm] = bone
            used.add(bone)
            joints[vrm] = armature.bone(bone).head
            skeleton.assessments[vrm] = BoneAssessment(vrm, bone, conf, AnalysisSource.SKELETON, ev, "skinning")

    # ---- head region: the head bone and everything below it (facial, jaw, eye, teeth ... bones)
    face_bones: set[str] = set()
    jaw_bones: set[str] = set()
    eye_bones: dict[str, str] = {}
    if head_bone is not None:
        face_bones = {head_bone.name, *armature.descendants(head_bone.name)}
    if "jaw" in final:
        jaw_bones = {final["jaw"], *armature.descendants(final["jaw"])}
    for side in ("left", "right"):
        if f"{side}Eye" in final:
            eye_bones[side] = final[f"{side}Eye"]

    # ---- hair first: its surfaces must not be read as face skin, nor inflate the head measurements
    hair: HairAnalysis | None = None
    if head_bone is not None:
        est = _head_size_estimate(inspection, armature, frame, face_bones, head_bone.head)
        if est is not None:
            hair = analyze_hair(inspection, armature, final, frame, head_bone.head, est[0], est[1], thresholds)
            warnings.extend(hair.warnings)
    hair_slots = hair.hair_slots(confident_only=True) if hair else {}

    face_geo = FaceGeometry()
    if head_bone is not None:
        face_geo = collect_face_points(inspection, armature, frame, face_bones, head_bone.head,
                                       neck_level_margin=0.1 * frame.scale, hair_slots=hair_slots,
                                       jaw_bones=jaw_bones, eye_bones=eye_bones)
    skin = face_geo.skin()
    head_top: Vec3 | None = None
    if skin and head_bone is not None:
        # Keep the head bone's lateral / depth position, replace only the height with the skull top.
        base = frame.to_local(head_bone.head)
        head_top = frame.to_world((base[0], max(p.local[1] for p in skin), base[2]))
    elif head_bone is not None:
        head_top = head_bone.tail
    head_world_points = [frame.to_world(p.local) for p in skin]

    # ---- measurements
    arm_bones: set[str] = set()
    for vrm in ("leftUpperArm", "rightUpperArm"):
        if vrm in final:
            arm_bones.add(final[vrm])
            arm_bones.update(armature.descendants(final[vrm]))
    hair_vertices = hair.hair_vertex_sets(inspection, confident_only=True) if hair else {}
    body_points: list[Vec3] = []
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature.name) or not mesh.vertex_positions:
            continue
        skip = hair_vertices.get(mesh.name, set())
        for i, pos in enumerate(mesh.vertex_positions):
            if i in skip:
                continue
            if mesh.vertex_weights and _weight_of(mesh, i, arm_bones) >= 0.3:
                continue
            body_points.append(pos)
    floor = None
    if lows:
        floor = min(dot(p, frame.up) for p in [*lows, *(joints[k] for k in ("leftFoot", "rightFoot") if k in joints)])
    measurements = measure_body(frame, joints, head_top, floor, body_points or None, None, head_world_points or None)

    landmarks = build_body_landmarks(frame, joints, head_top, waist_fallback_t)

    # ---- face
    face: FaceAnalysis | None = None
    if head_bone is not None:
        eyes = {f"{s}_eye": joints[f"{s}Eye"] for s in ("left", "right") if f"{s}Eye" in joints}
        morphs = face_morphs(inspection, expression_binds or {})
        face = analyze_face(frame, face_geo, head_bone.head, head_top, eyes or None, "jaw" in final, morphs,
                            detected=getattr(inspection, "face_points", None) or None)
        if face is not None and "jaw" in joints:
            face.jaw_pivot = frame.to_local(joints["jaw"])
        if face is None:
            warnings.append(AnalysisWarning("face-unavailable", "not enough skinned facial geometry for face "
                                            "landmark fitting", "face"))
        else:
            warnings.extend(face.warnings)
            for name, lm in face.landmarks.items():
                landmarks[{"mouth_center": "mouth", "nose_tip": "nose"}.get(name, name)] = lm
            if face.head_width:
                # the face analysis measures the head without its hair
                measurements.values["head_width"] = Measurement(
                    face.head_width, max(face.quality, 0.5), AnalysisSource.GEOMETRY, "face-skin-extent")
            if face.head_height:
                measurements.values["head_height"] = Measurement(
                    face.head_height, max(face.quality, 0.5), AnalysisSource.GEOMETRY, "chin-to-skull-top")

    # ---- hands
    hands: dict[str, HandAnalysis | None] = {}
    for side in ("left", "right"):
        hand_src = final.get(f"{side}Hand")
        if hand_src is None:
            hands[side] = None
            continue
        ha = classify_finger_chains(armature, hand_src, side, weighted, frame)
        if ha.state != BONES_PRESENT:
            hand = armature.bone(hand_src)
            pts: list[Vec3] = []
            srcs: list[tuple[str, int]] = []
            for mesh in inspection.meshes:
                if mesh.armature not in (None, armature.name) or not mesh.vertex_positions:
                    continue
                for i, p in enumerate(mesh.vertex_positions):
                    if _weight_of(mesh, i, {hand_src}) >= 0.5:
                        pts.append(p)
                        srcs.append((mesh.name, i))
            along = None
            if hand is not None:
                along = normalize(sub(hand.tail, hand.head))
            seg = segment_fingers(pts, hand.head, along) if hand is not None else None
            ha.segmentation = seg
            if seg is not None:
                seg.points, seg.sources = pts, srcs
            if seg is None or not pts:
                ha.state = NO_EVIDENCE
            elif seg.fused:
                ha.state = FUSED
                ha.warnings.append(AnalysisWarning(
                    "fingers-fused", f"{side} hand: independent finger articulation unavailable (fused fingers)", "hands"))
            elif seg.confidence >= 0.5 and seg.clusters:
                ha.state = GEOMETRY_SEPARABLE
                ha.confidence = seg.confidence
            else:
                ha.state = NO_EVIDENCE
        hands[side] = ha
    lh, rh = hands["left"], hands["right"]
    if lh and rh and lh.state == BONES_PRESENT and rh.state == BONES_PRESENT:
        compare_hands(lh, rh, frame, armature)
    for ha in (lh, rh):
        if ha is None:
            continue
        for vrm, src in ha.vrm_map().items():
            if vrm not in final and src not in used and ha.confidence >= thresholds.automatic_mapping:
                final[vrm] = src
                inferred[vrm] = src
                used.add(src)
        warnings.extend(ha.warnings)

    hh, hw = measurements.get("head_height"), measurements.get("head_width")
    if hair is not None and head_bone is not None and hh and hw:
        # re-run with the skin-only head size (the estimate included the hair)
        hair = analyze_hair(inspection, armature, final, frame, head_bone.head, hh, hw, thresholds)

    # ---- materials
    materials = classify_materials(inspection, frame, landmarks, hh, thresholds.material_semantic)

    mouth = analyze_mouth(face, frame, existing_visemes or []) if face is not None else None
    return HumanModelAnalysis(
        frame=frame, measurements=measurements, landmarks=landmarks, skeleton=skeleton, left_hand=lh, right_hand=rh,
        face=face, hair=hair, materials=materials, warnings=warnings, joints=joints, humanoid=final,
        inferred_bones=inferred, mouth=mouth, thresholds=thresholds)
