"""Anatomy-scaled deformation fields shared by blink, visemes and basic expressions.

A field is built in the frame's local coordinates ``(left, up, forward)`` and converted to world vectors last; every
amplitude is a fraction of a *measured* size (mouth width, lid opening, head height), never a Blender-space constant.
"""

from __future__ import annotations

from ..anatomy.face import FaceAnalysis, FacePoint
from ..anatomy.frame import AnatomicalFrame
from ..anatomy.regions import Ellipse
from ..anatomy.vec import Vec3, clamp01, cosine_falloff, smoothstep
from .plans import PROSOPON_PREFIX, ShapeKeyPlan

MOVE_ROLES = ("skin", "lash", "teeth")  # eyeballs and brows stay put unless a field asks for brows explicitly


class FieldBuilder:
    def __init__(self, face: FaceAnalysis, frame: AnatomicalFrame) -> None:
        self.face = face
        self.frame = frame
        self.local: dict[tuple[str, int], Vec3] = {}

    def add(self, region: Ellipse, fn, roles: tuple[str, ...] = MOVE_ROLES) -> None:
        """``fn(dx, dy, r, w) -> (ddx, ddy, ddz)`` for every point of ``roles`` inside ``region``."""
        for p in self.face.geometry.points:
            if p.role not in roles or not region.contains(p.local):
                continue
            dy = p.local[1] - region.cy
            ry = region.ry_up if dy >= 0 else region.ry_down
            r = (((p.local[0] - region.cx) / region.rx) ** 2 + (dy / ry) ** 2) ** 0.5
            w = cosine_falloff(r)
            if w <= 0:
                continue
            d = fn(p.local[0] - region.cx, dy, r, w)
            key = (p.mesh, p.index)
            old = self.local.get(key, (0.0, 0.0, 0.0))
            self.local[key] = (old[0] + d[0], old[1] + d[1], old[2] + d[2])

    def build(self, name: str, confidence: float, description: str, vrm_expression: str | None = None,
              min_move: float = 1e-9) -> ShapeKeyPlan:
        plan = ShapeKeyPlan(PROSOPON_PREFIX + name, confidence=confidence, description=description,
                            vrm_expression=vrm_expression)
        biggest = 0.0
        for (mesh, idx), d in self.local.items():
            if abs(d[0]) + abs(d[1]) + abs(d[2]) <= min_move:
                continue
            w = self.frame.vector_to_world(d)
            plan.offsets.setdefault(mesh, {})[idx] = w
            biggest = max(biggest, (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5)
        plan.max_offset = biggest
        return plan


def sgn(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def mirror_x(face: FaceAnalysis, frame: AnatomicalFrame, side: str, name: str) -> float | None:
    loc = face.local(frame, f"{side}_{name}")
    return loc[0] if loc else None
