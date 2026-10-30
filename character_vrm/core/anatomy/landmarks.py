"""Phase 3: semantic landmarks. Geometry Profiles consume these; nothing else computes waist/shoulder levels."""

from __future__ import annotations

from dataclasses import dataclass

from .confidence import AnalysisSource
from .frame import AnatomicalFrame
from .vec import Vec3, lerp, mean

LANDMARK_NAMES = (
    "head_center", "head_top", "left_eye", "right_eye", "nose", "mouth", "chin", "neck",
    "left_shoulder", "right_shoulder", "chest", "waist", "hips",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand_center", "right_hand_center",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "upper_chest",
)


@dataclass
class Landmark:
    name: str
    position: Vec3  # world space of the source scene
    confidence: float
    source: AnalysisSource
    method: str = ""  # how it was found, e.g. "detected" | "fallback-normalized"


def _lm(name: str, pos: Vec3 | None, conf: float, src: AnalysisSource, method: str) -> Landmark | None:
    return Landmark(name, pos, conf, src, method) if pos is not None else None


def waist_landmark(frame: AnatomicalFrame, joints: dict[str, Vec3], fallback_t: float = 0.72) -> Landmark | None:
    """The anatomical waist. Detected from the lower torso interval; otherwise a named normalised fallback
    (``shoulders = 0, hips = 1``); the ``method`` says which, so reports never hide it."""
    spine, chest, upper = joints.get("spine"), joints.get("chest"), joints.get("upperChest")
    h = frame.height_of
    if spine and chest and h(chest) > h(spine):
        return Landmark("waist", lerp(spine, chest, 0.5), 0.9, AnalysisSource.SKELETON, "detected")
    if spine and upper and h(upper) > h(spine):
        return Landmark("waist", lerp(spine, upper, 0.35), 0.85, AnalysisSource.SKELETON, "detected")
    hips = joints.get("hips")
    shoulder = shoulder_level_point(joints)
    if hips and shoulder:
        return Landmark("waist", lerp(shoulder, hips, fallback_t), 0.5, AnalysisSource.SKELETON, "fallback-normalized")
    return None


def shoulder_level_point(joints: dict[str, Vec3]) -> Vec3 | None:
    pts = [joints[n] for n in ("leftShoulder", "rightShoulder", "leftUpperArm", "rightUpperArm") if n in joints]
    if pts:
        return mean(pts)
    for name in ("upperChest", "chest", "spine"):
        if name in joints:
            return joints[name]
    return None


def upper_chest_landmark(joints: dict[str, Vec3]) -> Landmark | None:
    for name in ("upperChest", "chest", "spine"):
        if name in joints:
            return Landmark("upper_chest", joints[name], 0.9, AnalysisSource.SKELETON, "detected")
    p = shoulder_level_point(joints)
    return Landmark("upper_chest", p, 0.5, AnalysisSource.SKELETON, "fallback-shoulder") if p else None


def build_body_landmarks(frame: AnatomicalFrame, joints: dict[str, Vec3], head_top: Vec3 | None,
                         fallback_t: float = 0.72) -> dict[str, Landmark]:
    S = AnalysisSource
    out: dict[str, Landmark | None] = {}
    head = joints.get("head")
    if head is not None and head_top is not None:
        out["head_center"] = _lm("head_center", lerp(head, head_top, 0.5), 0.75, S.SKELETON, "head-bone-top-midpoint")
    out["head_top"] = _lm("head_top", head_top, 0.8, S.SKELETON, "top-of-head")
    out["neck"] = _lm("neck", joints.get("neck") or head, 0.9 if "neck" in joints else 0.5, S.SKELETON, "neck-bone")
    out["hips"] = _lm("hips", joints.get("hips"), 0.95, S.SKELETON, "hips-bone")
    out["waist"] = waist_landmark(frame, joints, fallback_t)
    out["chest"] = _lm("chest", joints.get("chest") or joints.get("upperChest") or joints.get("spine"), 0.85,
                       S.SKELETON, "chest-bone")
    out["upper_chest"] = upper_chest_landmark(joints)
    for side in ("left", "right"):
        cap = side
        out[f"{side}_shoulder"] = _lm(f"{side}_shoulder", joints.get(f"{cap}Shoulder") or joints.get(f"{cap}UpperArm"),
                                      0.9, S.SKELETON, "shoulder-or-upper-arm")
        out[f"{side}_elbow"] = _lm(f"{side}_elbow", joints.get(f"{cap}LowerArm"), 0.9, S.SKELETON, "lower-arm-joint")
        out[f"{side}_wrist"] = _lm(f"{side}_wrist", joints.get(f"{cap}Hand"), 0.9, S.SKELETON, "hand-joint")
        hand_tip = joints.get(f"{cap}HandTip")
        wrist = joints.get(f"{cap}Hand")
        out[f"{side}_hand_center"] = _lm(f"{side}_hand_center", lerp(wrist, hand_tip, 0.5) if wrist and hand_tip else wrist,
                                         0.8 if hand_tip else 0.6, S.SKELETON, "wrist-fingertips-midpoint")
        out[f"{side}_knee"] = _lm(f"{side}_knee", joints.get(f"{cap}LowerLeg"), 0.9, S.SKELETON, "lower-leg-joint")
        out[f"{side}_ankle"] = _lm(f"{side}_ankle", joints.get(f"{cap}Foot"), 0.9, S.SKELETON, "foot-joint")
        eye = joints.get(f"{cap}Eye")
        if eye is not None:
            out[f"{side}_eye"] = _lm(f"{side}_eye", eye, 0.9, S.SKELETON, "eye-bone")
    return {k: v for k, v in out.items() if v is not None}
