"""Shared face-analysis data: the canonical template, face points, morph deltas and skin openings (pure)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import MeshInfo
from .frame import AnatomicalFrame
from .surfaces import SKIN, boundary_loops
from .vec import Vec3

# (u, v): u across half the head width (left = +), v from the chin (0) to the top of the head (1).
CANONICAL_FACE: dict[str, tuple[float, float]] = {
    "chin": (0.0, 0.0),
    "lower_lip": (0.0, 0.135), "mouth_center": (0.0, 0.19), "upper_lip": (0.0, 0.245),
    "mouth_left": (0.27, 0.19), "mouth_right": (-0.27, 0.19),
    "nose_tip": (0.0, 0.365), "nose_bridge": (0.0, 0.50),
    "left_nostril": (0.11, 0.335), "right_nostril": (-0.11, 0.335),
    "left_eye": (0.36, 0.53), "right_eye": (-0.36, 0.53),
    "left_upper_eyelid": (0.36, 0.565), "right_upper_eyelid": (-0.36, 0.565),
    "left_lower_eyelid": (0.36, 0.495), "right_lower_eyelid": (-0.36, 0.495),
    "left_eyebrow": (0.36, 0.63), "right_eyebrow": (-0.36, 0.63),
    "left_cheek": (0.62, 0.33), "right_cheek": (-0.62, 0.33),
    "left_jaw": (0.70, 0.12), "right_jaw": (-0.70, 0.12),
}


FACE_LANDMARKS = tuple(CANONICAL_FACE)


@dataclass
class FacePoint:
    mesh: str
    index: int
    local: Vec3   # anatomical-frame local coordinates (left, up, forward), source units
    role: str     # skin | eye | teeth | hair | lash | brow
    jaw: float = 0.0          # skin weight on the jaw bone and its children
    eye_side: str | None = None  # "left" / "right" when the vertex rides on that eye bone


@dataclass
class FaceGeometry:
    points: list[FacePoint] = field(default_factory=list)
    meshes: dict[str, MeshInfo] = field(default_factory=dict)  # meshes the points come from (for topology)

    def skin(self) -> list[FacePoint]:
        return [p for p in self.points if p.role == SKIN]


# ---------------------------------------------------------------- morph evidence

@dataclass
class FaceMorphs:
    """World-space deltas of existing expression morphs, per VRM expression: ``{expr: {(mesh, vertex): delta}}``."""

    deltas: dict[str, dict[tuple[str, int], Vec3]] = field(default_factory=dict)

    def get(self, expr: str) -> dict[tuple[str, int], Vec3]:
        return self.deltas.get(expr, {})


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


def local_deltas(frame: AnatomicalFrame, d: dict[tuple[str, int], Vec3]) -> dict[tuple[str, int], Vec3]:
    return {k: frame.vector_to_local(v) for k, v in d.items()}


def moved_vertices(d: dict[tuple[str, int], Vec3], fraction: float = 0.15) -> set[tuple[str, int]]:
    if not d:
        return set()
    big = max(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in d.values())
    return {k for k, v in d.items() if math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) >= fraction * big}


def skin_loops(geometry: FaceGeometry, by_key, head_width):
    """(mesh name, loop points, the mesh's skin points) for every boundary loop of the face skin."""
    for name, mesh in geometry.meshes.items():
        if not mesh.polygons:
            continue
        polys = [poly for poly in mesh.polygons
                 if all((name, v) in by_key and by_key[(name, v)].role == SKIN for v in poly)]
        if len(polys) < 20:
            continue
        own = [p.local for p in by_key.values() if p.mesh == name and p.role == SKIN]
        for loop in boundary_loops(mesh, polys, 1e-4 * head_width):
            pts = [by_key[(name, v)].local for v in loop.vertices if (name, v) in by_key]
            if len(pts) >= 6:
                yield name, pts, own


def surrounded(own: list[Vec3], cx: float, cy: float, w: float, z_front: float) -> bool:
    """Skin of the same mesh lies right above and right below the loop's centre: an opening *in* the face (lips,
    eyes), not the edge of a mesh that stops there (a body mesh's hole for the head)."""
    r = max(0.35 * w, 1e-9)
    above = any(abs(p[0] - cx) < r and cy < p[1] < cy + 2.5 * r and p[2] >= z_front for p in own)
    below = any(abs(p[0] - cx) < r and cy - 2.5 * r < p[1] < cy and p[2] >= z_front for p in own)
    return above and below
