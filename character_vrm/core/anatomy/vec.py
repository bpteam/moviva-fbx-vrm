"""Tiny stdlib vector maths (the converter has no numpy dependency: it also runs in Blender's Python)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
EPS = 1e-9


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3 | None:
    n = length(a)
    return None if n < EPS else (a[0] / n, a[1] / n, a[2] / n)


def dist(a: Vec3, b: Vec3) -> float:
    return length(sub(a, b))


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def mean(points: Iterable[Vec3]) -> Vec3 | None:
    pts = list(points)
    if not pts:
        return None
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def orthogonalize(v: Vec3, against: Vec3) -> Vec3 | None:
    """``v`` with its component along the unit vector ``against`` removed, normalised."""
    return normalize(sub(v, scale(against, dot(v, against))))


def angle_between(a: Vec3, b: Vec3) -> float:
    """Degrees, 0..180."""
    na, nb = normalize(a), normalize(b)
    if na is None or nb is None:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(na, nb)))))


def principal_axis(points: Sequence[Vec3], iterations: int = 40) -> tuple[Vec3, float, Vec3] | None:
    """Dominant direction of a point cloud: ``(axis, elongation, centroid)`` (power iteration on the covariance).

    ``elongation`` is ``1 - lambda2 / lambda1`` in [0, 1]: 1 for a line, 0 for an isotropic blob.
    """
    c = mean(points)
    if c is None or len(points) < 3:
        return None
    cov = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = sub(p, c)
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]

    def apply(m: list[list[float]], v: Vec3) -> Vec3:
        return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]

    v: Vec3 = normalize((1.0, 0.7, 0.4)) or (1.0, 0.0, 0.0)
    lam1 = 0.0
    for _ in range(iterations):
        w = apply(cov, v)
        lam1 = length(w)
        nv = normalize(w)
        if nv is None:
            return None
        v = nv
    # Second eigenvalue by deflation.
    deflated = [[cov[i][j] - lam1 * v[i] * v[j] for j in range(3)] for i in range(3)]
    u: Vec3 = normalize(cross(v, (0.3, 0.9, 0.1))) or (0.0, 1.0, 0.0)
    lam2 = 0.0
    for _ in range(iterations):
        w = apply(deflated, u)
        lam2 = length(w)
        nu = normalize(w)
        if nu is None:
            lam2 = 0.0
            break
        u = nu
    elong = 0.0 if lam1 < EPS else max(0.0, min(1.0, 1.0 - lam2 / lam1))
    return v, elong, c


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def cosine_falloff(r: float) -> float:
    """1 at r=0, 0 at r>=1, smooth."""
    return 0.0 if r >= 1.0 else 0.5 * (1.0 + math.cos(math.pi * max(0.0, r)))


def quat_from_unit_vectors(a: Vec3, b: Vec3) -> tuple[float, float, float, float]:
    """Shortest-arc rotation taking direction ``a`` to ``b`` as ``(x, y, z, w)``."""
    na, nb = normalize(a), normalize(b)
    if na is None or nb is None:
        return (0.0, 0.0, 0.0, 1.0)
    d = dot(na, nb)
    if d < -0.999999:  # opposite: any perpendicular axis
        axis = normalize(cross(na, (1.0, 0.0, 0.0))) or normalize(cross(na, (0.0, 1.0, 0.0))) or (0.0, 0.0, 1.0)
        return (axis[0], axis[1], axis[2], 0.0)
    c = cross(na, nb)
    w = 1.0 + d
    n = math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2 + w * w)
    return (c[0] / n, c[1] / n, c[2] / n, w / n)


def quat_angle_deg(q: tuple[float, float, float, float]) -> float:
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(q[3])))))
