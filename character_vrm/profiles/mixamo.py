"""Mixamo rigs: ``mixamorig:Hips``, ``mixamorig:LeftUpLeg`` ..."""

from __future__ import annotations

import re

from .base import BoneProfile, finish, limb_table

_PREFIX = re.compile(r"^(mixamorig\d*[:_])Hips$")
_FINGER = {"Thumb": "Thumb", "Index": "Index", "Middle": "Middle", "Ring": "Ring", "Little": "Pinky"}


def build(names: frozenset[str]):
    prefix = None
    for name in names:
        if (m := _PREFIX.match(name)):
            prefix = m.group(1)
            break
    if prefix is None:
        # Renamed export without the namespace: require the Mixamo-specific ``UpLeg``.
        if "Hips" in names and "LeftUpLeg" in names and "RightUpLeg" in names:
            prefix = ""
        else:
            return None
    side = {"L": "Left", "R": "Right"}
    table = {
        "hips": f"{prefix}Hips", "spine": f"{prefix}Spine", "chest": f"{prefix}Spine1",
        "upperChest": f"{prefix}Spine2", "neck": f"{prefix}Neck", "head": f"{prefix}Head",
    }
    table.update(limb_table(
        lambda tag, base: f"{prefix}{side[tag]}{base}",
        shoulder="Shoulder", upper_arm="Arm", lower_arm="ForeArm", hand="Hand",
        upper_leg="UpLeg", lower_leg="Leg", foot="Foot", toes="ToeBase",
        fingers=lambda tag, finger, i: f"{prefix}{side[tag]}Hand{_FINGER[finger]}{i}",
        eye="Eye",
    ))
    return finish(table, names)


PROFILE = BoneProfile("mixamo", build)
