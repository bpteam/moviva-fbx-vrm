"""VRoid Studio / three-vrm sample rigs (``J_Bip_*``) and VRoid morph names (``Fcl_*``)."""

from __future__ import annotations

from .base import BoneProfile, finish, limb_table

# normalised (lowercase alphanumerics) shape-key name -> confidence
EXPRESSION_ALIASES: dict[str, dict[str, float]] = {
    "happy": {"fclalljoy": 0.98},
    "angry": {"fclallangry": 0.98},
    "sad": {"fclallsorrow": 0.98},
    "relaxed": {"fclallfun": 0.98},  # VRM 0.x "fun" became VRM 1.0 "relaxed"
    "surprised": {"fclallsurprised": 0.98},
    "blink": {"fcleyeclose": 0.98},
    "blinkLeft": {"fcleyeclosel": 0.98},
    "blinkRight": {"fcleyecloser": 0.98},
    "aa": {"fclmtha": 0.98},
    "ih": {"fclmthi": 0.98},
    "ou": {"fclmthu": 0.98},
    "ee": {"fclmthe": 0.98},
    "oh": {"fclmtho": 0.98},
}

_P = "J_Bip_"


def build(names: frozenset[str]):
    if f"{_P}C_Hips" not in names:
        return None
    table = {
        "hips": f"{_P}C_Hips", "spine": f"{_P}C_Spine", "chest": f"{_P}C_Chest",
        "upperChest": f"{_P}C_UpperChest", "neck": f"{_P}C_Neck", "head": f"{_P}C_Head",
    }
    table.update(limb_table(
        lambda tag, base: f"{_P}{tag}_{base}",
        shoulder="Shoulder", upper_arm="UpperArm", lower_arm="LowerArm", hand="Hand",
        upper_leg="UpperLeg", lower_leg="LowerLeg", foot="Foot", toes="ToeBase",
        fingers=lambda tag, finger, i: f"{_P}{tag}_{finger}{i}",
    ))
    table["leftEye"] = "J_Adj_L_FaceEye"
    table["rightEye"] = "J_Adj_R_FaceEye"
    return finish(table, names)


PROFILE = BoneProfile("vroid", build)
