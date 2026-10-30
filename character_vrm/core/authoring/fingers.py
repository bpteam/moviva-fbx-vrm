"""Phase 6: experimental generated finger rig (``--generate-finger-rig``), confidence-gated.

    detect palm -> separate finger regions -> estimate axes -> phalanx joints -> bone chains -> weights

Only planned when the hand geometry separates into five clear regions; a mitten is reported, never rigged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..anatomy.confidence import DEFAULT_THRESHOLDS, Thresholds
from ..anatomy.hands import FINGER_ORDER, GEOMETRY_SEPARABLE, FUSED, HandAnalysis
from ..anatomy.model import HumanModelAnalysis
from ..anatomy.vec import Vec3, add, dot, lerp, mean, principal_axis, scale, smoothstep, sub
from .plans import PROSOPON_PREFIX, PlanIssue, Skipped

JOINT_FRACTIONS = (0.0, 0.36, 0.68, 1.0)  # knuckle, PIP, DIP, tip along each finger


@dataclass
class NewBone:
    name: str
    parent: str
    head: Vec3
    tail: Vec3
    vrm_bone: str | None = None


@dataclass
class FingerRigPlan:
    side: str
    hand_bone: str
    bones: list[NewBone] = field(default_factory=list)
    # mesh -> vertex -> {new bone: fraction of the hand bone's weight that moves to it}
    transfers: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    confidence: float = 0.0
    skipped: Skipped | None = None

    @property
    def humanoid(self) -> dict[str, str]:
        return {b.vrm_bone: b.name for b in self.bones if b.vrm_bone}

    def validate(self) -> list[PlanIssue]:
        issues: list[PlanIssue] = []
        names = [b.name for b in self.bones]
        if len(set(names)) != len(names):
            issues.append(PlanIssue("fingers", "duplicate generated bone names"))
        for b in self.bones:
            if not b.name.startswith(PROSOPON_PREFIX):
                issues.append(PlanIssue("fingers", f"{b.name} outside the Prosopon_ namespace"))
            if sum(abs(x - y) for x, y in zip(b.head, b.tail)) < 1e-9:
                issues.append(PlanIssue("fingers", f"{b.name} has zero length"))
        for mesh, verts in self.transfers.items():
            for v, ws in verts.items():
                if sum(ws.values()) > 1.0 + 1e-6:
                    issues.append(PlanIssue("fingers", f"weights of {mesh}[{v}] exceed the hand weight"))
        return issues


def plan_finger_rig(hand: HandAnalysis, hand_head: Vec3, thresholds: Thresholds = DEFAULT_THRESHOLDS,
                    enabled: bool = True) -> FingerRigPlan:
    plan = FingerRigPlan(hand.side, hand.hand_bone or "")
    seg = hand.segmentation
    if hand.state == FUSED:
        plan.skipped = Skipped(f"{hand.side} fingers", "independent finger articulation unavailable (fused fingers)")
        return plan
    if hand.state != GEOMETRY_SEPARABLE or seg is None:
        plan.skipped = Skipped(f"{hand.side} fingers", "no finger geometry to rig" if hand.state != "bones-present"
                               else "finger bones already exist")
        return plan
    if not enabled:
        plan.skipped = Skipped(f"{hand.side} fingers", "feature disabled (--generate-finger-rig)", seg.confidence)
        return plan
    if seg.confidence < thresholds.generated_fingers or len(seg.clusters) < 5:
        plan.skipped = Skipped(f"{hand.side} fingers", f"confidence {seg.confidence:.2f} below "
                               f"{thresholds.generated_fingers:.2f}" if len(seg.clusters) >= 5 else
                               "thumb not separable", seg.confidence)
        return plan
    along = seg.along
    assert along is not None
    plan.confidence = seg.confidence
    for finger in FINGER_ORDER:
        idxs = seg.clusters.get(finger, [])
        pts = [seg.points[i] for i in idxs]
        if len(pts) < 12:
            plan = FingerRigPlan(hand.side, hand.hand_bone or "")
            plan.skipped = Skipped(f"{hand.side} {finger}", "finger cluster too small")
            return plan
        pa = principal_axis(pts)
        axis = pa[0] if pa else along
        if dot(axis, along) < 0:
            axis = scale(axis, -1)
        ts = [dot(sub(p, hand_head), axis) for p in pts]
        t0, t1 = min(ts), max(ts)
        # Centreline: centroid of a thin slice at each joint fraction.
        joints: list[Vec3] = []
        for f in JOINT_FRACTIONS:
            t = t0 + (t1 - t0) * f
            band = 0.08 * (t1 - t0)
            slice_pts = [p for p, tt in zip(pts, ts) if abs(tt - t) <= band] or pts
            joints.append(mean(slice_pts))  # type: ignore[arg-type]
        vrm_parts = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else ("Proximal", "Intermediate", "Distal")
        for k, part in enumerate(vrm_parts):
            plan.bones.append(NewBone(
                f"{PROSOPON_PREFIX}{hand.side}_{finger}_{k + 1}",
                hand.hand_bone if k == 0 else f"{PROSOPON_PREFIX}{hand.side}_{finger}_{k}",
                joints[k], joints[k + 1], f"{hand.side}{finger}{part}"))
        # Weights: fraction of the hand weight moved to each phalanx, cross-faded across the joints.
        names = [f"{PROSOPON_PREFIX}{hand.side}_{finger}_{k + 1}" for k in range(3)]
        for i, p, t in zip(idxs, pts, ts):
            u = (t - t0) / max(t1 - t0, 1e-9)
            ws: dict[str, float] = {}
            for k in range(3):
                lo, hi = JOINT_FRACTIONS[k], JOINT_FRACTIONS[k + 1]
                if k == 0:
                    inside = 1.0 - smoothstep((u - hi + 0.08) / 0.16)
                elif k == 2:
                    inside = smoothstep((u - lo + 0.08) / 0.16)
                else:
                    inside = smoothstep((u - lo + 0.08) / 0.16) * (1.0 - smoothstep((u - hi + 0.08) / 0.16))
                if inside > 1e-3:
                    ws[names[k]] = inside
            total = sum(ws.values()) or 1.0
            share = smoothstep(u / 0.25)  # the knuckle end stays with the hand bone
            ws = {n: w / total * share for n, w in ws.items()}
            mesh, vertex = seg.sources[i]
            plan.transfers.setdefault(mesh, {})[vertex] = ws
    return plan


def plan_all_hands(analysis: HumanModelAnalysis, enabled: bool, thresholds: Thresholds = DEFAULT_THRESHOLDS,
                   ) -> list[FingerRigPlan]:
    plans = []
    for side in ("left", "right"):
        hand = analysis.hand(side)
        if hand is None or hand.state in ("bones-present", "no-finger-evidence"):
            continue
        head = analysis.joints.get(f"{side}Hand")
        if head is None:
            continue
        plans.append(plan_finger_rig(hand, head, thresholds, enabled))
    return plans
