"""A synthetic T-pose humanoid (capsules) for the auto-rig tests. Z up, faces -Y, character's left = +X, 1.8 m."""

from __future__ import annotations

import numpy as np

# (start, end, radius): metres, (x, y, z)
_PARTS = [
    ((0, 0, 0.95), (0, 0, 1.30), 0.15),    # pelvis + belly
    ((0, 0, 1.30), (0, 0, 1.50), 0.16),    # chest
    ((0, 0, 1.50), (0, 0, 1.62), 0.05),    # neck
    ((0, 0, 1.62), (0, 0, 1.76), 0.11),    # head
    ((0.10, 0, 1.45), (0.78, 0, 1.45), 0.04),   # left arm (T pose)
    ((-0.10, 0, 1.45), (-0.78, 0, 1.45), 0.04),
    ((0.10, 0, 0.92), (0.10, 0, 0.08), 0.07),   # left leg
    ((-0.10, 0, 0.92), (-0.10, 0, 0.08), 0.07),
    ((0.10, 0, 0.05), (0.10, -0.15, 0.05), 0.05),  # left foot (toes toward -Y)
    ((-0.10, 0, 0.05), (-0.10, -0.15, 0.05), 0.05),
]


def mannequin_points(density: int = 2500, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for a, b, r in _PARTS:
        a, b = np.array(a, float), np.array(b, float)
        n = int(density * (np.linalg.norm(b - a) + r) / 0.3)
        t = rng.random(n)
        ang = rng.random(n) * 2 * np.pi
        axis = (b - a) / np.linalg.norm(b - a)
        e1 = np.cross(axis, [0.3, 0.5, 0.8]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        out.append(a + t[:, None] * (b - a) + r * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2))
    return np.vstack(out)
