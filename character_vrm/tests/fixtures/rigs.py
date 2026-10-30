"""Synthetic T-pose humanoid armatures with configurable naming schemes.

Positions are in metres, +Z up, the character's left is +X (Blender import
convention). Nothing here touches ``bpy``.
"""

from __future__ import annotations

from typing import Callable

from character_vrm.core.model import ArmatureInfo, BoneInfo

FINGER_ROLES = ("thumb", "index", "middle", "ring", "little")

Namer = Callable[[str], "str | None"]  # role -> bone name (None = bone absent)


def _spec(namer: Namer, fingers: bool = True) -> list[tuple[str, str | None, tuple[float, float, float]]]:
    """Return (role, parent_role, head) triples."""
    out: list[tuple[str, str | None, tuple[float, float, float]]] = [
        ("root", None, (0.0, 0.0, 0.0)),
        ("hips", "root", (0.0, 0.0, 1.0)),
        ("spine", "hips", (0.0, 0.0, 1.1)),
        ("chest", "spine", (0.0, 0.0, 1.25)),
        ("upperchest", "chest", (0.0, 0.0, 1.4)),
        ("neck", "upperchest", (0.0, 0.0, 1.5)),
        ("head", "neck", (0.0, 0.0, 1.6)),
        ("headtop", "head", (0.0, 0.0, 1.8)),
    ]
    for side, sign in (("L", 1.0), ("R", -1.0)):
        s = sign
        out += [
            (f"shoulder.{side}", "upperchest", (0.05 * s, 0.0, 1.45)),
            (f"upperarm.{side}", f"shoulder.{side}", (0.2 * s, 0.0, 1.45)),
            (f"lowerarm.{side}", f"upperarm.{side}", (0.45 * s, 0.0, 1.45)),
            (f"hand.{side}", f"lowerarm.{side}", (0.7 * s, 0.0, 1.45)),
            (f"upleg.{side}", "hips", (0.1 * s, 0.0, 0.95)),
            (f"lowleg.{side}", f"upleg.{side}", (0.1 * s, 0.0, 0.5)),
            (f"foot.{side}", f"lowleg.{side}", (0.1 * s, 0.0, 0.1)),
            (f"toes.{side}", f"foot.{side}", (0.1 * s, -0.12, 0.03)),
            (f"eye.{side}", "head", (0.03 * s, -0.08, 1.68)),
        ]
        if fingers:
            spread = {"thumb": -0.03, "index": 0.02, "middle": 0.0, "ring": -0.02, "little": -0.04}
            for k, finger in enumerate(FINGER_ROLES):
                parent = f"hand.{side}"
                y = 0.03 * (2 - k) if finger != "thumb" else -0.04
                for i in (1, 2, 3):
                    role = f"{finger}{i}.{side}"
                    x = (0.72 + 0.03 * i) * s
                    out.append((role, parent, (x, y + spread[finger] * 0, 1.45 - (0.02 if finger == "thumb" else 0))))
                    parent = role
    return out


def build_armature(name: str, namer: Namer, *, fingers: bool = True,
                   extra: list[tuple[str, str, tuple[float, float, float]]] | None = None,
                   drop: set[str] | None = None) -> ArmatureInfo:
    """``extra`` entries are (bone name, parent *bone name*, head)."""
    drop = drop or set()
    entries: list[tuple[str, str | None, tuple[float, float, float]]] = []  # bone names
    role_to_name: dict[str, str] = {}
    for role, parent_role, head in _spec(namer, fingers):
        bone_name = namer(role)
        if bone_name is None or role in drop:
            role_to_name[role] = role_to_name.get(parent_role or "", "") if parent_role else ""
            continue
        role_to_name[role] = bone_name
        parent = role_to_name.get(parent_role) if parent_role else None
        entries.append((bone_name, parent or None, head))
    for bone_name, parent, head in extra or []:
        entries.append((bone_name, parent, head))
    children: dict[str, list[tuple[float, float, float]]] = {}
    for bone_name, parent, head in entries:
        if parent:
            children.setdefault(parent, []).append(head)
    bones = []
    for bone_name, parent, head in entries:
        kids = children.get(bone_name)
        tail = kids[0] if kids else (head[0], head[1], head[2] + 0.05)
        bones.append(BoneInfo(bone_name, parent, head, tail))
    return ArmatureInfo(name, bones)


def mixamo_namer(prefix: str = "mixamorig:") -> Namer:
    base = {"hips": "Hips", "spine": "Spine", "chest": "Spine1", "upperchest": "Spine2", "neck": "Neck",
            "head": "Head", "headtop": "HeadTop_End"}
    limb = {"shoulder": "Shoulder", "upperarm": "Arm", "lowerarm": "ForeArm", "hand": "Hand",
            "upleg": "UpLeg", "lowleg": "Leg", "foot": "Foot", "toes": "ToeBase", "eye": "Eye"}
    finger = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "little": "Pinky"}

    def namer(role: str) -> str | None:
        if role == "root":
            return None
        if role in base:
            return prefix + base[role]
        r, _, side = role.partition(".")
        side_name = "Left" if side == "L" else "Right"
        if r in limb:
            return None if r == "eye" else prefix + side_name + limb[r]
        for key, val in finger.items():
            if r.startswith(key):
                return f"{prefix}{side_name}Hand{val}{r[len(key):]}"
        return None
    return namer


def cc_namer() -> Namer:
    base = {"root": "CC_Base_BoneRoot", "hips": "CC_Base_Hip", "spine": "CC_Base_Waist", "chest": "CC_Base_Spine01",
            "upperchest": "CC_Base_Spine02", "neck": "CC_Base_NeckTwist01", "head": "CC_Base_Head",
            "headtop": "CC_Base_HeadTop"}
    limb = {"shoulder": "Clavicle", "upperarm": "Upperarm", "lowerarm": "Forearm", "hand": "Hand",
            "upleg": "Thigh", "lowleg": "Calf", "foot": "Foot", "toes": "ToeBase", "eye": "Eye"}
    finger = {"thumb": "Thumb", "index": "Index", "middle": "Mid", "ring": "Ring", "little": "Pinky"}

    def namer(role: str) -> str | None:
        if role in base:
            return base[role]
        r, _, side = role.partition(".")
        if r in limb:
            return f"CC_Base_{side}_{limb[r]}"
        for key, val in finger.items():
            if r.startswith(key):
                return f"CC_Base_{side}_{val}{r[len(key):]}"
        return None
    return namer


def rigify_namer(prefix: str = "DEF-") -> Namer:
    base = {"hips": "spine", "spine": "spine.001", "chest": "spine.002", "upperchest": "spine.003",
            "neck": "spine.004", "head": "spine.006"}
    limb = {"shoulder": "shoulder", "upperarm": "upper_arm", "lowerarm": "forearm", "hand": "hand",
            "upleg": "thigh", "lowleg": "shin", "foot": "foot", "toes": "toe", "eye": "eye"}

    def namer(role: str) -> str | None:
        if role in base:
            return prefix + base[role]
        r, _, side = role.partition(".")
        if r in limb:
            return f"{prefix}{limb[r]}.{side}"
        for key in ("thumb", "index", "middle", "ring", "little"):
            if r.startswith(key):
                i = int(r[len(key):])
                if key == "thumb":
                    return f"{prefix}thumb.{i:02d}.{side}"
                return f"{prefix}f_{'pinky' if key == 'little' else key}.{i:02d}.{side}"
        return None
    return namer


def opaque_namer() -> Namer:
    """Names that carry no anatomy at all: only topology and position can resolve them."""
    counter: dict[str, str] = {}

    def namer(role: str) -> str | None:
        if role not in counter:
            counter[role] = f"Bone.{len(counter):03d}"
        return counter[role]
    return namer


def generic_namer() -> Namer:
    """Left_/Right_ + upper-camel names, no profile match: exercises the name resolver."""
    base = {"root": "Root", "hips": "Pelvis", "spine": "Spine", "chest": "Chest", "upperchest": "UpperChest",
            "neck": "Neck", "head": "Head", "headtop": "HeadEnd"}
    limb = {"shoulder": "Shoulder", "upperarm": "UpperArm", "lowerarm": "LowerArm", "hand": "Hand",
            "upleg": "UpperLeg", "lowleg": "LowerLeg", "foot": "Foot", "toes": "Toes", "eye": "Eye"}
    finger = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "little": "Little"}

    def namer(role: str) -> str | None:
        if role in base:
            return base[role]
        r, _, side = role.partition(".")
        pre = "Left" if side == "L" else "Right"
        if r in limb:
            return f"{pre}_{limb[r]}"
        for key, val in finger.items():
            if r.startswith(key):
                return f"{pre}_{val}_{r[len(key):]}"
        return None
    return namer
