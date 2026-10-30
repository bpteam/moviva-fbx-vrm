"""ARKit 52 blendshapes and compound VRM expression recipes built from them.

The recipe weights are heuristics, tunable here (and overridable per model via
``expressions`` in the map file). A recipe is only applied when its *required*
shape keys exist; nothing is ever fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


ARKIT_52: tuple[str, ...] = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
)
ARKIT_NORMALISED: frozenset[str] = frozenset(_norm(n) for n in ARKIT_52)  # == canon() of each name


@dataclass(frozen=True)
class Recipe:
    required: dict[str, float]  # ARKit name -> weight; all must exist
    optional: dict[str, float]  # bound when present
    confidence: float = 0.8


RECIPES: dict[str, Recipe] = {
    # eyes
    "blinkLeft": Recipe({"eyeBlinkLeft": 1.0}, {}, 0.9),
    "blinkRight": Recipe({"eyeBlinkRight": 1.0}, {}, 0.9),
    "blink": Recipe({"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}, {}, 0.9),
    # visemes
    "aa": Recipe({"jawOpen": 0.8}, {}),
    "ih": Recipe({"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5}, {"jawOpen": 0.25}),
    "ou": Recipe({"mouthPucker": 0.9}, {"jawOpen": 0.15}),
    "ee": Recipe({"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}, {"jawOpen": 0.3}),
    "oh": Recipe({"mouthFunnel": 0.8}, {"jawOpen": 0.4}),
    # emotions (coarse; better authored ones should come from the source asset)
    "happy": Recipe({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8},
                    {"cheekSquintLeft": 0.4, "cheekSquintRight": 0.4}, 0.7),
    "angry": Recipe({"browDownLeft": 0.9, "browDownRight": 0.9},
                    {"noseSneerLeft": 0.3, "noseSneerRight": 0.3,
                     "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4}, 0.7),
    "sad": Recipe({"browInnerUp": 0.8}, {"mouthFrownLeft": 0.6, "mouthFrownRight": 0.6}, 0.7),
    "surprised": Recipe({"browInnerUp": 0.6, "eyeWideLeft": 0.8, "eyeWideRight": 0.8},
                        {"browOuterUpLeft": 0.7, "browOuterUpRight": 0.7, "jawOpen": 0.3}, 0.7),
    # eye direction (used by expression-based lookAt). "Left"/"Right" = the character's own side.
    "lookUp": Recipe({"eyeLookUpLeft": 0.5, "eyeLookUpRight": 0.5}, {}, 0.85),
    "lookDown": Recipe({"eyeLookDownLeft": 0.5, "eyeLookDownRight": 0.5}, {}, 0.85),
    "lookLeft": Recipe({"eyeLookOutLeft": 0.5, "eyeLookInRight": 0.5}, {}, 0.85),
    "lookRight": Recipe({"eyeLookOutRight": 0.5, "eyeLookInLeft": 0.5}, {}, 0.85),
}
