"""Blender Rigify: prefers the ``DEF-`` deform bones, falls back to metarig names.

Rigify's spine naming: ``spine`` is the pelvis, ``spine.001`` .. ``.003`` the
torso, ``spine.004``/``.005`` the neck and ``spine.006`` the head.
"""

from __future__ import annotations

from .base import BoneProfile, finish, limb_table


def _table(prefix: str) -> dict[str, str]:
    table = {
        "hips": f"{prefix}spine", "spine": f"{prefix}spine.001", "chest": f"{prefix}spine.002",
        "upperChest": f"{prefix}spine.003", "neck": f"{prefix}spine.004",
        "head": f"{prefix}spine.006", "jaw": f"{prefix}jaw",
    }
    table.update(limb_table(
        lambda tag, base: f"{prefix}{base}.{tag}",
        shoulder="shoulder", upper_arm="upper_arm", lower_arm="forearm", hand="hand",
        upper_leg="thigh", lower_leg="shin", foot="foot", toes="toe",
        fingers=lambda tag, finger, i: (
            f"{prefix}thumb.{i:02d}.{tag}" if finger == "Thumb"
            else f"{prefix}f_{ {'Little': 'pinky'}.get(finger, finger.lower()) }.{i:02d}.{tag}"
        ),
        eye="eye",
    ))
    return table


def build(names: frozenset[str]):
    candidates = []
    for prefix in ("DEF-", ""):
        result = finish(_table(prefix), names)
        if result:
            candidates.append(result)
    return max(candidates, key=lambda r: (r[1], len(r[0]))) if candidates else None


PROFILE = BoneProfile("rigify", build)
