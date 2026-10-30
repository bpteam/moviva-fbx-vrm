"""Synthetic head geometry for pure-Python face / hair / hand tests (not a fake ``bpy``: plain ``MeshInfo`` data)."""

from __future__ import annotations

import math

from character_vrm.core.model import MeshInfo

Vec3 = tuple[float, float, float]


def _gauss(x: float, y: float, cx: float, cy: float, sx: float, sy: float) -> float:
    return math.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def head_mesh(name: str = "Face", *, center: Vec3 = (0.0, 0.0, 1.70), radii: Vec3 = (0.085, 0.10, 0.115),
              weight_bone: str = "mixamorig:Head", grid: int = 120, lips: bool = True, nose: bool = True,
              eye_sockets: bool = False, flip: bool = False) -> MeshInfo:
    """An ellipsoid head facing -Y (Blender import convention) with a nose, lips and optional eye recesses.

    Face features are placed with the canonical template proportions (chin at the ellipsoid bottom).
    """
    cx, cy, cz = center
    rx, ry, rz = radii
    chin_z, top_z = cz - rz, cz + rz
    face_h = top_z - chin_z
    positions: list[Vec3] = []
    weights: dict[int, dict[str, float]] = {}
    polys: list[tuple[int, ...]] = []
    n_u, n_v = grid, grid
    for iv in range(n_v + 1):
        t = -1.0 + 2.0 * iv / n_v                       # uniform in height: even vertex density on the face
        ring = max(0.0, 1.0 - abs(t) ** 2.6) ** 0.5     # blunt top and chin, like a skull
        for iu in range(n_u):
            theta = 2 * math.pi * iu / n_u
            x = rx * ring * math.cos(theta)
            y = -ry * ring * math.sin(theta)  # theta = 90 deg -> forward (-Y)
            z = rz * t
            depth = 0.0
            fx, fz = x, (cz + z - chin_z) / face_h  # v in 0..1 of this surface point
            if y < 0:  # front half only
                if nose:
                    depth += 0.022 * _gauss(fx, fz, 0.0, 0.365, 0.016, 0.05)
                if lips:
                    depth += 0.008 * _gauss(fx, fz, 0.0, 0.245, 0.030, 0.018)
                    depth += 0.010 * _gauss(fx, fz, 0.0, 0.135, 0.030, 0.018)
                if eye_sockets:
                    for s in (-1, 1):
                        depth -= 0.008 * _gauss(fx, fz, s * 0.031, 0.53, 0.014, 0.03)
            # Push outward along the surface normal direction in y only (forward).
            positions.append((cx + x, cy + y - depth if y < 0 else cy + y, cz + z))
    for iv in range(n_v):
        for iu in range(n_u):
            a = iv * n_u + iu
            b = iv * n_u + (iu + 1) % n_u
            c = (iv + 1) * n_u + (iu + 1) % n_u
            d = (iv + 1) * n_u + iu
            polys.append((a, b, c, d))
    for i in range(len(positions)):
        weights[i] = {weight_bone: 1.0}
    return MeshInfo(name, len(positions), len(polys) * 2, "Armature", [weight_bone], ["Skin"],
                    vertex_positions=positions, polygons=polys, vertex_weights=weights)


def eyeball_mesh(name: str, center: Vec3, radius: float, bone: str = "mixamorig:Head", n: int = 10) -> MeshInfo:
    positions: list[Vec3] = []
    polys = []
    for iv in range(n + 1):
        phi = -math.pi / 2 + math.pi * iv / n
        for iu in range(n):
            theta = 2 * math.pi * iu / n
            positions.append((center[0] + radius * math.cos(phi) * math.cos(theta),
                              center[1] + radius * math.cos(phi) * math.sin(theta),
                              center[2] + radius * math.sin(phi)))
    for iv in range(n):
        for iu in range(n):
            polys.append((iv * n + iu, iv * n + (iu + 1) % n, (iv + 1) * n + (iu + 1) % n, (iv + 1) * n + iu))
    return MeshInfo(name, len(positions), len(polys) * 2, "Armature", [bone], ["Eye"],
                    vertex_positions=positions, polygons=polys,
                    vertex_weights={i: {bone: 1.0} for i in range(len(positions))})
