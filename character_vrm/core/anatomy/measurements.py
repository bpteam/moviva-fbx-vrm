"""Phase 2: normalised body measurements from the mapped skeleton and (where present) mesh geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

from .confidence import AnalysisSource
from .frame import AnatomicalFrame
from .vec import Vec3, dist

MEASUREMENT_NAMES = (
    "total_height", "head_height", "head_width", "shoulder_width", "chest_width", "waist_width", "hip_width",
    "torso_length", "upper_arm_length", "forearm_length", "hand_length",
    "thigh_length", "lower_leg_length", "foot_length",
)


@dataclass
class Measurement:
    value: float  # source units
    confidence: float
    source: AnalysisSource
    method: str = ""


@dataclass
class HumanMeasurements:
    values: dict[str, Measurement] = field(default_factory=dict)
    total_height_units: float = 0.0

    def get(self, name: str) -> float | None:
        m = self.values.get(name)
        return m.value if m else None

    def normalized(self) -> dict[str, float]:
        """Every measurement divided by total height: unit-free, invariant to model scale."""
        h = self.total_height_units or self.get("total_height") or 0.0
        return {k: m.value / h for k, m in self.values.items() if h > 0}

    def __getattr__(self, name: str) -> float | None:  # ``measurements.shoulder_width``
        if name in MEASUREMENT_NAMES:
            return self.get(name)
        raise AttributeError(name)


def _seg(points: dict[str, Vec3], a: str, b: str) -> float | None:
    return dist(points[a], points[b]) if a in points and b in points else None


def slice_width(points: list[Vec3], frame: AnatomicalFrame, height: float, band: float, max_half: float | None = None,
                skip: set[int] | None = None) -> tuple[float, int] | None:
    """Lateral extent of the mesh around a level along ``up`` (arms excluded through ``max_half`` / ``skip``)."""
    lats: list[float] = []
    for i, p in enumerate(points):
        if skip and i in skip:
            continue
        if abs(frame.height_of(p) - height) <= band:
            x = frame.lateral_of(p)
            if max_half is None or abs(x) <= max_half:
                lats.append(x)
    if len(lats) < 8:
        return None
    return max(lats) - min(lats), len(lats)


def measure_body(frame: AnatomicalFrame, joints: dict[str, Vec3], top: Vec3 | None, floor: float | None,
                 body_points: list[Vec3] | None = None, body_skip: set[int] | None = None,
                 head_points: list[Vec3] | None = None) -> HumanMeasurements:
    """``joints`` are world positions of mapped bones (heads) plus ``leftFootTip``/``rightFootTip``/... extras."""
    out = HumanMeasurements()
    S = AnalysisSource
    up_top = frame.height_of(top) if top is not None else frame.height_of(joints["head"]) if "head" in joints else None
    ground = floor if floor is not None else min((frame.height_of(p) for p in joints.values()), default=None)
    if up_top is not None and ground is not None and up_top > ground:
        h = up_top - ground
        out.values["total_height"] = Measurement(h, 0.9 if top is not None else 0.7, S.SKELETON, "skeleton-extent")
        out.total_height_units = h
    else:
        out.total_height_units = frame.scale

    def put(name: str, value: float | None, confidence: float, source: AnalysisSource, method: str) -> None:
        if value is not None and value > 0:
            out.values[name] = Measurement(value, confidence, source, method)

    put("torso_length", _seg(joints, "hips", "neck") or _seg(joints, "hips", "head"), 0.85, S.SKELETON, "hips-neck")
    put("upper_arm_length", _avg(_seg(joints, "leftUpperArm", "leftLowerArm"), _seg(joints, "rightUpperArm", "rightLowerArm")),
        0.9, S.SKELETON, "bone-lengths")
    put("forearm_length", _avg(_seg(joints, "leftLowerArm", "leftHand"), _seg(joints, "rightLowerArm", "rightHand")),
        0.9, S.SKELETON, "bone-lengths")
    put("hand_length", _avg(_seg(joints, "leftHand", "leftHandTip"), _seg(joints, "rightHand", "rightHandTip")),
        0.8, S.SKELETON, "hand-to-fingertip")
    put("thigh_length", _avg(_seg(joints, "leftUpperLeg", "leftLowerLeg"), _seg(joints, "rightUpperLeg", "rightLowerLeg")),
        0.9, S.SKELETON, "bone-lengths")
    put("lower_leg_length", _avg(_seg(joints, "leftLowerLeg", "leftFoot"), _seg(joints, "rightLowerLeg", "rightFoot")),
        0.9, S.SKELETON, "bone-lengths")
    put("foot_length", _avg(_seg(joints, "leftFoot", "leftToesTip"), _seg(joints, "rightFoot", "rightToesTip")),
        0.7, S.SKELETON, "foot-to-toe")

    lu, ru = joints.get("leftUpperArm"), joints.get("rightUpperArm")
    shoulder = abs(frame.lateral_of(lu) - frame.lateral_of(ru)) if lu and ru else None
    put("shoulder_width", shoulder, 0.8, S.SKELETON, "upper-arm-joints")
    lh, rh = joints.get("leftUpperLeg"), joints.get("rightUpperLeg")
    hip_joints = abs(frame.lateral_of(lh) - frame.lateral_of(rh)) if lh and rh else None

    # Head: from the head bone base to the top of the character (or head geometry).
    head = joints.get("head")
    if head is not None and up_top is not None:
        head_h = up_top - frame.height_of(head)
        if head_h > 0:
            put("head_height", head_h, 0.7, S.SKELETON, "head-bone-to-top")
    if head_points and len(head_points) >= 20:
        lats = [frame.lateral_of(p) for p in head_points]
        put("head_width", max(lats) - min(lats), 0.85, S.GEOMETRY, "head-mesh-extent")
        ups = [frame.height_of(p) for p in head_points]
        put("head_height", max(ups) - min(ups), 0.85, S.GEOMETRY, "head-mesh-extent")
    elif "head_height" in out.values:
        put("head_width", out.values["head_height"].value * 0.72, 0.45, S.SKELETON, "head-height-ratio (inferred)")

    # Torso levels from geometry, exclusively excluding arms; fall back to skeleton proportions.
    if body_points:
        half = (shoulder or out.total_height_units * 0.25) / 2 * 1.05
        band = 0.02 * out.total_height_units
        for name, joint in (("chest_width", "chest"), ("waist_width", "spine"), ("hip_width", "hips")):
            j = joints.get(joint) or joints.get("upperChest")
            if j is None:
                continue
            got = slice_width(body_points, frame, frame.height_of(j), band, half if name != "hip_width" else None, body_skip)
            if got:
                put(name, got[0], 0.8, S.GEOMETRY, f"mesh-slice@{joint}")
    if "chest_width" not in out.values and shoulder:
        put("chest_width", shoulder * 0.85, 0.4, S.SKELETON, "shoulder-ratio")
    if "hip_width" not in out.values and hip_joints:
        put("hip_width", hip_joints * 1.6, 0.4, S.SKELETON, "hip-joint-ratio")
    if "waist_width" not in out.values and "chest_width" in out.values and "hip_width" in out.values:
        put("waist_width", (out.values["chest_width"].value + out.values["hip_width"].value) / 2 * 0.9, 0.35,
            S.SKELETON, "chest-hip-blend")
    return out


def _avg(*vals: float | None) -> float | None:
    real = [v for v in vals if v]
    return sum(real) / len(real) if real else None
