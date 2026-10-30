"""Phase 5: automatic rest-pose analysis and calibration plan.

The bind pose (T, A, relaxed, anything else, plus head tilt and torso lean) is measured in the anatomical frame and
compared with Moviva's canonical relaxed pose. The result is *metadata*: the source skeleton is not modified.
The relaxed-arm target is the one the runtime's ``calibrateRig`` aims at (0.38 outward, -0.925 down), so the two agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..anatomy.model import HumanModelAnalysis
from ..anatomy.vec import Vec3, angle_between, dot, normalize, quat_angle_deg, quat_from_unit_vectors, sub
from .plans import PlanIssue

# Same numbers as avatar/src/avatar/RigCalibration.ts (outward 0.38, down 0.925).
RELAXED_ARM_OUTWARD = 0.38
RELAXED_ARM_DOWN = -0.925
RELAXED_ARM_ABDUCTION_DEG = math.degrees(math.atan2(RELAXED_ARM_OUTWARD, -RELAXED_ARM_DOWN))
TILT_TOLERANCE_DEG = 1.5


@dataclass
class BoneRotation:
    """Rotation, in the anatomical frame, taking the bind direction to the canonical one."""

    quaternion: tuple[float, float, float, float]
    angle_deg: float
    current: Vec3 | None = None  # the measured bind direction (frame-local)
    target: Vec3 | None = None
    confidence: float = 0.0


@dataclass
class RestPoseCalibration:
    spine: BoneRotation | None = None
    chest: BoneRotation | None = None
    neck: BoneRotation | None = None
    head: BoneRotation | None = None
    left_shoulder: BoneRotation | None = None
    left_upper_arm: BoneRotation | None = None
    left_lower_arm: BoneRotation | None = None
    right_shoulder: BoneRotation | None = None
    right_upper_arm: BoneRotation | None = None
    right_lower_arm: BoneRotation | None = None
    pose_type: str = "unknown"
    arm_abduction_deg: dict[str, float] = field(default_factory=dict)
    torso_lean_deg: float = 0.0
    head_tilt_deg: float = 0.0
    confidence: float = 0.0

    def bones(self) -> dict[str, BoneRotation]:
        names = {"spine": "spine", "chest": "chest", "neck": "neck", "head": "head",
                 "left_shoulder": "leftShoulder", "left_upper_arm": "leftUpperArm", "left_lower_arm": "leftLowerArm",
                 "right_shoulder": "rightShoulder", "right_upper_arm": "rightUpperArm",
                 "right_lower_arm": "rightLowerArm"}
        return {vrm: getattr(self, attr) for attr, vrm in names.items() if getattr(self, attr) is not None}


@dataclass
class RestPosePlan:
    calibration: RestPoseCalibration
    mutates_scene: bool = False  # the source skeleton is left alone: this is metadata for the runtime
    notes: list[str] = field(default_factory=list)

    def validate(self) -> list[PlanIssue]:
        issues = []
        for name, rot in self.calibration.bones().items():
            if rot.angle_deg > 120:
                issues.append(PlanIssue("rest-pose", f"{name}: correction of {rot.angle_deg:.0f} deg is implausible"))
        return issues

    def to_dict(self) -> dict:
        cal = self.calibration
        return {
            "poseType": cal.pose_type, "armAbductionDeg": {k: round(v, 2) for k, v in cal.arm_abduction_deg.items()},
            "targetAbductionDeg": round(RELAXED_ARM_ABDUCTION_DEG, 2),
            "torsoLeanDeg": round(cal.torso_lean_deg, 2), "headTiltDeg": round(cal.head_tilt_deg, 2),
            "confidence": round(cal.confidence, 3), "space": "anatomical-frame (left, up, forward)",
            "bones": {vrm: {"rest": [round(x, 6) for x in r.quaternion], "angleDeg": round(r.angle_deg, 2),
                            "confidence": round(r.confidence, 3)} for vrm, r in cal.bones().items()},
            "mutatesScene": self.mutates_scene, "notes": self.notes,
        }


def _local_dir(analysis: HumanModelAnalysis, a: str, b: str) -> Vec3 | None:
    pa, pb = analysis.joints.get(a), analysis.joints.get(b)
    return normalize(analysis.frame.vector_to_local(sub(pb, pa))) if pa and pb else None


def _rotation(current: Vec3, target: Vec3, conf: float, tol: float = 0.0) -> BoneRotation | None:
    q = quat_from_unit_vectors(current, target)
    angle = quat_angle_deg(q)
    if angle <= tol:
        return None
    return BoneRotation(q, angle, current, target, conf)


def analyze_rest_pose(analysis: HumanModelAnalysis) -> RestPoseCalibration:
    cal = RestPoseCalibration()
    frame = analysis.frame
    conf_of = lambda v: analysis.skeleton.confidence_of(v) or 0.7  # noqa: E731
    up: Vec3 = (0.0, 1.0, 0.0)
    abductions: list[float] = []
    swings: list[float] = []

    for side, sign in (("left", 1.0), ("right", -1.0)):
        d = _local_dir(analysis, f"{side}UpperArm", f"{side}LowerArm")
        if d is None:
            continue
        abd = math.degrees(math.atan2(sign * d[0], -d[1]))
        swing = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))  # out of the coronal plane, forward positive
        cal.arm_abduction_deg[side] = abd
        abductions.append(abd)
        swings.append(swing)
        target = normalize((sign * RELAXED_ARM_OUTWARD, RELAXED_ARM_DOWN, 0.0))
        rot = _rotation(d, target, min(conf_of(f"{side}UpperArm"), conf_of(f"{side}LowerArm")), 1.0)  # type: ignore[arg-type]
        setattr(cal, f"{side}_upper_arm", rot)
        # The elbow bend is character, not pose error: the forearm is left alone.
        sh_d = _local_dir(analysis, f"{side}Shoulder", f"{side}UpperArm")
        if sh_d is not None:
            elevation = math.degrees(math.atan2(sh_d[1], abs(sh_d[0]) or 1e-9))
            if abs(elevation) > 25:  # raised / dropped clavicle
                setattr(cal, f"{side}_shoulder", _rotation(sh_d, normalize((sign, 0.0, 0.0)), conf_of(f"{side}Shoulder")))  # type: ignore[arg-type]

    if abductions:
        mean_abd = sum(abductions) / len(abductions)
        asym = max(abductions) - min(abductions)
        if asym > 15 or (swings and max(abs(s) for s in swings) > 25):
            cal.pose_type = "non-standard"
        elif mean_abd >= 75:
            cal.pose_type = "T-pose"
        elif mean_abd >= 35:
            cal.pose_type = "A-pose"
        else:
            cal.pose_type = "relaxed"

    # Gravity reference from the legs (independent of the trunk); the frame's up is hips -> head.
    ref_up = up
    legs = [analysis.joints.get(k) for k in ("leftFoot", "rightFoot")]
    hips = analysis.joints.get("hips")
    if hips and all(legs):
        feet = tuple((legs[0][i] + legs[1][i]) / 2 for i in range(3))  # type: ignore[index]
        ref = normalize(frame.vector_to_local(sub(hips, feet)))  # type: ignore[arg-type]
        if ref:
            ref_up = ref
    trunk = [("spine", "spine", "chest" if "chest" in analysis.joints else "neck"),
             ("chest", "chest", "upperChest" if "upperChest" in analysis.joints else "neck"),
             ("neck", "neck", "head")]
    for attr, a, b in trunk:
        if a not in analysis.joints or b not in analysis.joints or a == b:
            continue
        d = _local_dir(analysis, a, b)
        if d is None:
            continue
        setattr(cal, attr, _rotation(d, ref_up, min(conf_of(a), conf_of(b)), TILT_TOLERANCE_DEG))
    hd = None
    if "head" in analysis.joints and "head_top" in analysis.landmarks:
        hd = normalize(frame.vector_to_local(sub(analysis.landmarks["head_top"].position, analysis.joints["head"])))
    if hd is not None:
        cal.head = _rotation(hd, ref_up, conf_of("head"), TILT_TOLERANCE_DEG)
        cal.head_tilt_deg = angle_between(hd, ref_up)
    torso = _local_dir(analysis, "hips", "neck" if "neck" in analysis.joints else "head")
    if torso is not None:
        cal.torso_lean_deg = angle_between(torso, ref_up)
    cal.confidence = round(min(frame.confidence, min((conf_of(v) for v in ("leftUpperArm", "rightUpperArm", "head")
                                                     if v in analysis.joints), default=0.0)), 4)
    return cal


def plan_rest_pose(analysis: HumanModelAnalysis) -> RestPosePlan:
    cal = analyze_rest_pose(analysis)
    notes = []
    if cal.pose_type == "non-standard":
        notes.append("arm angles differ between the sides or swing forward: calibration is per side")
    return RestPosePlan(cal, False, notes)
