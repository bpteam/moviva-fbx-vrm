"""Reallusion Character Creator / iClone: ``CC_Base_*``.

``CC_Base_NeckTwist01`` is the actual neck joint in CC rigs (there is no plain
``Neck``), so it is mapped to VRM ``neck`` here; the other twist / helper bones
stay untouched.
"""

from __future__ import annotations

from .base import BoneProfile, finish, limb_table

from .arkit import Recipe

# Unambiguous CC3+/CC4 identifiers (normalised).
EXPRESSION_ALIASES: dict[str, dict[str, float]] = {
    "blinkLeft": {"eyeblinkl": 0.9},
    "blinkRight": {"eyeblinkr": 0.9},
}

# CC's viseme and facial vocabulary composed into VRM presets (names are matched through ``expressions.canon``).
# CC visemes (``V_Open``, ``V_Wide``, ``V_Tight_O``) shape the lips only: iClone opens the jaw with the jaw *bone*,
# which a VRM expression cannot drive. Measured on a CC4 export (converter benchmark, Mégane): ``V_Open`` moves at
# most 8 mm, ``Jaw_Open`` 23 mm. So every viseme carries its share of the ``Jaw_Open`` morph.
RECIPES: dict[str, Recipe] = {
    "aa": Recipe({"V_Open": 0.7}, {"Jaw_Open": 0.5}, 0.85),
    "ih": Recipe({"V_Wide": 0.7}, {"Jaw_Open": 0.15}, 0.85),
    "ee": Recipe({"V_Wide": 1.0}, {"Jaw_Open": 0.1}, 0.85),
    "ou": Recipe({"V_Tight_O": 1.0}, {"Jaw_Open": 0.1}, 0.85),
    "oh": Recipe({"V_Tight_O": 0.7}, {"Jaw_Open": 0.3, "V_Open": 0.3}, 0.85),
    "blink": Recipe({"Eye_Blink_L": 1.0, "Eye_Blink_R": 1.0}, {}, 0.9),
    "happy": Recipe({"Mouth_Smile_L": 0.8, "Mouth_Smile_R": 0.8}, {"Cheek_Raise_L": 0.4, "Cheek_Raise_R": 0.4}, 0.75),
    "angry": Recipe({"Brow_Drop_L": 0.9, "Brow_Drop_R": 0.9},
                    {"Nose_Sneer_L": 0.3, "Nose_Sneer_R": 0.3, "Mouth_Frown_L": 0.4, "Mouth_Frown_R": 0.4}, 0.75),
    "sad": Recipe({"Brow_Raise_Inner_L": 0.8, "Brow_Raise_Inner_R": 0.8},
                  {"Mouth_Frown_L": 0.6, "Mouth_Frown_R": 0.6}, 0.75),
    "surprised": Recipe({"Brow_Raise_Inner_L": 0.6, "Brow_Raise_Inner_R": 0.6, "Eye_Wide_L": 0.8, "Eye_Wide_R": 0.8},
                        {"Brow_Raise_Outer_L": 0.7, "Brow_Raise_Outer_R": 0.7, "V_Open": 0.3}, 0.75),
    "lookUp": Recipe({"Eye_L_Look_Up": 1.0, "Eye_R_Look_Up": 1.0}, {}, 0.85),
    "lookDown": Recipe({"Eye_L_Look_Down": 1.0, "Eye_R_Look_Down": 1.0}, {}, 0.85),
    "lookLeft": Recipe({"Eye_L_Look_L": 1.0, "Eye_R_Look_L": 1.0}, {}, 0.85),
    "lookRight": Recipe({"Eye_L_Look_R": 1.0, "Eye_R_Look_R": 1.0}, {}, 0.85),
}

_PREFIX = "CC_Base_"


def build(names: frozenset[str]):
    if f"{_PREFIX}Hip" not in names and f"{_PREFIX}Waist" not in names:
        return None
    table = {
        "hips": f"{_PREFIX}Hip", "spine": f"{_PREFIX}Waist", "chest": f"{_PREFIX}Spine01",
        "upperChest": f"{_PREFIX}Spine02", "neck": f"{_PREFIX}NeckTwist01",
        "head": f"{_PREFIX}Head", "jaw": f"{_PREFIX}JawRoot",
    }
    table.update(limb_table(
        lambda tag, base: f"{_PREFIX}{tag}_{base}",
        shoulder="Clavicle", upper_arm="Upperarm", lower_arm="Forearm", hand="Hand",
        upper_leg="Thigh", lower_leg="Calf", foot="Foot", toes="ToeBase",
        fingers=lambda tag, finger, i: f"{_PREFIX}{tag}_{ {'Little': 'Pinky', 'Middle': 'Mid'}.get(finger, finger) }{i}",
        eye="Eye",
    ))
    return finish(table, names)


PROFILE = BoneProfile("character-creator", build)
