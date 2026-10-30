"""Phase 1: canonical anatomical coordinate frame, independent of world axes, scale, origin and pose."""

from __future__ import annotations

from dataclasses import dataclass

from .confidence import AnalysisWarning
from .vec import Vec3, add, cross, dot, length, mean, normalize, orthogonalize, scale, sub


@dataclass
class AnatomicalFrame:
    """``origin`` is the hips; ``up``/``right``/``forward`` are orthonormal world vectors, ``scale`` the body height.

    Local coordinates are ``(left, up, forward)`` in source units; ``normalized`` divides by ``scale``.
    ``right`` points to the character's right (it is ``-left``); the convention is ``left = up x forward``.
    """

    origin: Vec3
    up: Vec3
    right: Vec3
    forward: Vec3
    scale: float
    confidence: float
    evidence: dict[str, float] | None = None

    @property
    def left(self) -> Vec3:
        return (-self.right[0], -self.right[1], -self.right[2])

    def to_local(self, p: Vec3) -> Vec3:
        d = sub(p, self.origin)
        return (dot(d, self.left), dot(d, self.up), dot(d, self.forward))

    def to_world(self, local: Vec3) -> Vec3:
        return add(self.origin, add(add(scale(self.left, local[0]), scale(self.up, local[1])),
                                    scale(self.forward, local[2])))

    def vector_to_local(self, v: Vec3) -> Vec3:
        return (dot(v, self.left), dot(v, self.up), dot(v, self.forward))

    def vector_to_world(self, v: Vec3) -> Vec3:
        return add(add(scale(self.left, v[0]), scale(self.up, v[1])), scale(self.forward, v[2]))

    def to_normalized(self, p: Vec3) -> Vec3:
        loc = self.to_local(p)
        s = self.scale or 1.0
        return (loc[0] / s, loc[1] / s, loc[2] / s)

    def height_of(self, p: Vec3) -> float:
        """Coordinate along ``up`` (absolute, not relative to the origin): equals ``z`` for a Z-up scene."""
        return dot(p, self.up)

    def lateral_of(self, p: Vec3) -> float:
        return dot(sub(p, self.origin), self.left)


def _pos(points: dict[str, Vec3], name: str) -> Vec3 | None:
    return points.get(name)


def build_frame(points: dict[str, Vec3], foot_forward_hint: Vec3 | None = None,
                fallback_left_hint: Vec3 | None = None) -> tuple[AnatomicalFrame | None, list[AnalysisWarning]]:
    """Infer the frame from mapped-bone world positions.

    ``points`` keys: ``hips``, ``head`` (head-base or head centre), optional ``leftShoulder`` / ``rightShoulder`` /
    ``leftUpperArm`` / ``rightUpperArm`` / ``leftUpperLeg`` / ``rightUpperLeg`` (any subset of the lateral pairs),
    ``leftFoot`` / ``leftToes`` / ``rightFoot`` / ``rightToes``, ``top`` (highest point of the character).
    ``fallback_left_hint`` is used only when the left/right pair is unavailable (then confidence drops).
    """
    warnings: list[AnalysisWarning] = []
    hips, head = _pos(points, "hips"), _pos(points, "head")
    if hips is None or head is None:
        return None, [AnalysisWarning("frame-unavailable", "hips and head are required to infer the anatomical frame",
                                      "frame")]
    up = normalize(sub(head, hips))
    if up is None:
        return None, [AnalysisWarning("frame-degenerate", "hips and head coincide", "frame")]

    lateral_votes: list[Vec3] = []
    for left, right in (("leftUpperArm", "rightUpperArm"), ("leftShoulder", "rightShoulder"),
                        ("leftUpperLeg", "rightUpperLeg"), ("leftFoot", "rightFoot")):
        l, r = _pos(points, left), _pos(points, right)
        if l is not None and r is not None:
            v = orthogonalize(sub(l, r), up)
            if v is not None:
                lateral_votes.append(v)
    evidence: dict[str, float] = {}
    if lateral_votes:
        # Votes must agree; a flipped pair (mislabelled sides) lowers confidence instead of cancelling silently.
        ref = lateral_votes[0]
        agree = [dot(ref, v) for v in lateral_votes]
        summed = normalize(tuple(sum(c) for c in zip(*lateral_votes)))  # type: ignore[arg-type]
        left = orthogonalize(summed or ref, up)
        evidence["lateral"] = max(0.0, min(1.0, sum(1 for a in agree if a > 0.8) / len(agree)))
        if len(lateral_votes) == 1:
            evidence["lateral"] = min(evidence["lateral"], 0.85)
    elif fallback_left_hint is not None:
        left = orthogonalize(fallback_left_hint, up)
        evidence["lateral"] = 0.4
        warnings.append(AnalysisWarning(
            "frame-left-assumed", "no left/right bone pair available: left assumed from the import convention", "frame"))
    else:
        return None, warnings + [AnalysisWarning("frame-no-lateral", "no left/right evidence", "frame")]
    if left is None:
        return None, warnings + [AnalysisWarning("frame-degenerate", "left axis is parallel to up", "frame")]

    forward = cross(left, up)  # left x up == forward (left = up x forward)
    forward_n = normalize(forward)
    if forward_n is None:
        return None, warnings
    forward = forward_n
    # Independent forward evidence: feet point forward.
    f_votes: list[float] = []
    for side in ("left", "right"):
        foot, toes = _pos(points, f"{side}Foot"), _pos(points, f"{side}Toes")
        if foot is not None and toes is not None:
            d = orthogonalize(sub(toes, foot), up)
            if d is not None:
                f_votes.append(dot(d, forward))
    if foot_forward_hint is not None:
        d = orthogonalize(foot_forward_hint, up)
        if d is not None:
            f_votes.append(dot(d, forward))
    if f_votes:
        agreement = sum(f_votes) / len(f_votes)
        evidence["forward"] = max(0.0, min(1.0, 0.5 + agreement / 2))
        if agreement < -0.3:
            evidence["forward"] = 0.05
            warnings.append(AnalysisWarning(
                "frame-mirrored-rig",
                "toes point opposite to the left-right derived forward: the rig looks mirrored or its sides are "
                "swapped; orientation confidence is low", "frame"))
    else:
        evidence["forward"] = 0.7

    top = _pos(points, "top") or head
    body_height = max(dot(sub(top, hips), up) + _height_below_hips(points, hips, up), 1e-6)
    confidence = min(evidence.values()) if evidence else 0.0
    confidence = round(0.5 * confidence + 0.5 * (sum(evidence.values()) / len(evidence)), 4)
    right = scale(left, -1.0)
    return AnatomicalFrame(origin=hips, up=up, right=right, forward=forward, scale=body_height,
                           confidence=confidence, evidence=evidence), warnings


def _height_below_hips(points: dict[str, Vec3], hips: Vec3, up: Vec3) -> float:
    lows = [dot(sub(hips, p), up) for k, p in points.items() if k in ("leftFoot", "rightFoot", "leftToes", "rightToes", "floor")]
    return max(lows) if lows else dot(sub(hips, hips), up) + 0.0


def frame_from_skeleton_points(points: dict[str, Vec3]) -> tuple[AnatomicalFrame | None, list[AnalysisWarning]]:
    mid = mean([p for k, p in points.items() if k in ("leftFoot", "rightFoot")])
    if mid is not None and "floor" not in points:
        points = {**points, "floor": mid}
    return build_frame(points)
