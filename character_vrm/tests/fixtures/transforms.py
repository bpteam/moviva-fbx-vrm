"""Rigidly move / rotate / scale a whole ``CharacterInspection`` (for invariance tests)."""

from __future__ import annotations

import copy
import math

from character_vrm.core.model import ArmatureInfo, BoneInfo, CharacterInspection

Vec3 = tuple[float, float, float]


def rotation(rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> list[list[float]]:
    """Degrees, applied X then Y then Z."""
    ax, ay, az = (math.radians(a) for a in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    mx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    my = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    mz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]

    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mul(mz, mul(my, mx))


def transform_point(p: Vec3, r, s: float, t: Vec3) -> Vec3:
    return tuple(s * sum(r[i][j] * p[j] for j in range(3)) + t[i] for i in range(3))  # type: ignore[return-value]


def transformed(scene: CharacterInspection, r=None, s: float = 1.0, t: Vec3 = (0.0, 0.0, 0.0)) -> CharacterInspection:
    r = r or rotation()
    out = copy.deepcopy(scene)
    out.armatures = [ArmatureInfo(a.name, [BoneInfo(b.name, b.parent, transform_point(b.head, r, s, t),
                                                    transform_point(b.tail, r, s, t), b.deform) for b in a.bones])
                     for a in scene.armatures]
    for mesh in out.meshes:
        mesh.vertex_positions = [transform_point(p, r, s, t) for p in mesh.vertex_positions]
    return out
