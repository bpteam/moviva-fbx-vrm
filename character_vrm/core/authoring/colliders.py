"""Phase 15: conservative spring colliders from canonical anatomy (head, neck, upper chest, shoulders)."""

from __future__ import annotations

from ..anatomy.model import HumanModelAnalysis
from ..anatomy.vec import dot, normalize, sub, dist
from ..model import ArmatureInfo, ColliderPlan
from .plans import Skipped

# radius = factor * measured size (never a world-space constant)
HEAD_FACTOR = 0.50      # x head width
NECK_FACTOR = 0.27      # x head width
CHEST_FACTOR = 0.42     # x chest width
SHOULDER_FACTOR = 0.12  # x shoulder width
MAX_OFF_AXIS = 0.6      # collider centres farther than this x radius from the bone axis are skipped (offset is bone-Y only)


def _collider(name: str, armature: ArmatureInfo, analysis: HumanModelAnalysis, vrm_bone: str, centre, radius: float,
              skipped: list[Skipped]) -> ColliderPlan | None:
    src = analysis.humanoid.get(vrm_bone)
    bone = armature.bone(src) if src else None
    if bone is None or centre is None or radius <= 0:
        return None
    axis = normalize(sub(bone.tail, bone.head))
    if axis is None:
        return None
    along = dot(sub(centre, bone.head), axis)
    off_axis = dist(sub(centre, bone.head), tuple(a * along for a in axis))  # type: ignore[arg-type]
    if off_axis > MAX_OFF_AXIS * radius:
        skipped.append(Skipped(name, f"anatomical centre is {off_axis:.3f} off the bone axis "
                                     "(only bone-axis offsets are representable)"))
        return None
    return ColliderPlan(name=name, bone=bone.name, radius=round(radius, 4), offset=(0.0, round(along, 4), 0.0))


def plan_colliders(analysis: HumanModelAnalysis, armature: ArmatureInfo) -> tuple[list[ColliderPlan], list[Skipped]]:
    skipped: list[Skipped] = []
    out: list[ColliderPlan | None] = []
    m = analysis.measurements
    lm = analysis.landmarks
    head_w, chest_w, shoulder_w = m.get("head_width"), m.get("chest_width"), m.get("shoulder_width")
    if head_w and "head_center" in lm:
        out.append(_collider("head_collider", armature, analysis, "head", lm["head_center"].position,
                             HEAD_FACTOR * head_w, skipped))
    if head_w and "neck" in analysis.joints:
        out.append(_collider("neck_collider", armature, analysis, "neck", analysis.joints["neck"],
                             NECK_FACTOR * head_w, skipped))
    upper = next((v for v in ("upperChest", "chest", "spine") if v in analysis.joints), None)
    if chest_w and upper:
        out.append(_collider("upper_chest_collider", armature, analysis, upper, analysis.joints[upper],
                             CHEST_FACTOR * chest_w, skipped))
    if shoulder_w:
        for side in ("left", "right"):
            key = f"{side}UpperArm"
            if key in analysis.joints:
                out.append(_collider(f"{side}_shoulder_collider", armature, analysis, key, analysis.joints[key],
                                     SHOULDER_FACTOR * shoulder_w, skipped))
    return [c for c in out if c is not None], skipped
