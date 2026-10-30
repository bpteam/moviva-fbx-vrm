"""Synthetic hand and hair geometry (plain ``MeshInfo`` data)."""

from __future__ import annotations

import math


from character_vrm.core.model import MeshInfo

Vec3 = tuple[float, float, float]


def _grid(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, n: tuple[int, int, int]) -> list[Vec3]:
    out = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                out.append((x0 + (x1 - x0) * i / max(n[0] - 1, 1), y0 + (y1 - y0) * j / max(n[1] - 1, 1),
                            z0 + (z1 - z0) * k / max(n[2] - 1, 1)))
    return out


def hand_mesh(name: str = "Hands", *, wrist: Vec3 = (0.7, 0.0, 1.45), sign: float = 1.0, bone: str = "hand.L",
              fused: bool = False, thumb: bool = True) -> MeshInfo:
    """A hand pointing along ``sign * +X``: a palm slab, four finger boxes with clear gaps and (optionally) a thumb.

    ``fused`` makes the four fingers one solid mitten block.
    """
    pts: list[Vec3] = []
    palm_len, width = 0.09, 0.085
    # palm
    for p in _grid(0.0, palm_len, -width / 2, width / 2, -0.012, 0.012, (10, 16, 4)):
        pts.append(p)
    finger_y = [0.030, 0.010, -0.010, -0.030]   # index, middle, ring, little
    finger_len = [0.066, 0.072, 0.068, 0.056]
    if fused:
        for p in _grid(palm_len, palm_len + 0.07, -0.040, 0.040, -0.010, 0.010, (12, 24, 3)):
            pts.append(p)
    else:
        for y, ln in zip(finger_y, finger_len):
            for p in _grid(palm_len, palm_len + ln, y - 0.0065, y + 0.0065, -0.008, 0.008, (12, 6, 3)):
                pts.append(p)
    if thumb:
        for p in _grid(0.02, 0.075, 0.052, 0.064, -0.009, 0.009, (10, 3, 3)):
            pts.append(p)
    world = [(wrist[0] + sign * x, wrist[1] + y, wrist[2] + z) for x, y, z in pts]
    return MeshInfo(name, len(world), len(world), "Armature", [bone], ["Skin"], vertex_positions=world,
                    polygons=[], vertex_weights={i: {bone: 1.0} for i in range(len(world))})


def strip_mesh(name: str, root: Vec3, tip: Vec3, width: float, bone: str = "mixamorig:Head", n_len: int = 12,
               material: str = "Hair", quads: bool = False) -> MeshInfo:
    """A long thin ribbon from ``root`` to ``tip`` (one connected component), weighted to ``bone``."""
    verts: list[Vec3] = []
    polys = []
    dx, dy, dz = (tip[0] - root[0], tip[1] - root[1], tip[2] - root[2])
    for i in range(n_len + 1):
        t = i / n_len
        c = (root[0] + dx * t, root[1] + dy * t, root[2] + dz * t)
        verts.append((c[0] - width / 2, c[1], c[2]))
        verts.append((c[0] + width / 2, c[1], c[2]))
    for i in range(n_len):
        a = 2 * i
        polys.append((a, a + 1, a + 3, a + 2))
    return MeshInfo(name, len(verts), len(polys) * 2, "Armature", [bone], [material], vertex_positions=verts,
                    polygons=polys, vertex_weights={i: {bone: 1.0} for i in range(len(verts))})


def with_finger_bones(arm, *, prefix: str = "mixamorig:", naming: str = "Bone", seed: int = 0):
    """Return ``(armature, names)``: ``arm`` plus anatomically laid-out finger chains under both hands.

    Hand ``L`` at +X pointing +X, hand ``R`` at -X pointing -X. The thumb starts near the wrist and diverges by 35 degrees;
    the other fingers start at the knuckles and run parallel. ``naming="Bone"`` gives meaningless names.
    """
    import random
    from character_vrm.core.model import ArmatureInfo, BoneInfo

    bones = list(arm.bones)
    names: dict[tuple[str, str, int], str] = {}
    counter = [0]

    def name_of(side: str, finger: str, i: int) -> str:
        if naming == "Bone":
            counter[0] += 1
            return f"Bone.{counter[0]:03d}"
        return f"{prefix}{'Left' if side == 'L' else 'Right'}Hand{finger}{i}"

    specs = {"Thumb": (0.02, 0.058, 35.0, (0.030, 0.024, 0.02)), "Index": (0.09, 0.030, 0.0, (0.035, 0.024, 0.02)),
             "Middle": (0.09, 0.010, 0.0, (0.038, 0.026, 0.02)), "Ring": (0.09, -0.010, 0.0, (0.034, 0.024, 0.019)),
             "Little": (0.09, -0.030, 0.0, (0.028, 0.02, 0.016))}
    order = list(specs)
    random.Random(seed).shuffle(order)
    for side, sign, hand in (("L", 1.0, f"{prefix}LeftHand"), ("R", -1.0, f"{prefix}RightHand")):
        wrist = (sign * 0.7, 0.0, 1.45)
        for finger in order:
            x0, y0, ang, lens = specs[finger]
            dx, dy = math.cos(math.radians(ang)), -math.sin(math.radians(ang)) * -1
            parent = hand
            x, y = x0, y0
            for i, ln in enumerate(lens, 1):
                nm = name_of(side, finger, i)
                names[(side, finger, i)] = nm
                head = (wrist[0] + sign * x, wrist[1] + y, wrist[2])
                x, y = x + dx * ln, y + dy * ln
                tail = (wrist[0] + sign * x, wrist[1] + y, wrist[2])
                bones.append(BoneInfo(nm, parent, head, tail))
                parent = nm
    return ArmatureInfo(arm.name, bones), names
