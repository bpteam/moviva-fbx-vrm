"""Per-vertex surface labels for avatar motion (pure): skin or clothing, limb or body, hair, accessory, face detail.

The avatar runtime measures how deep one part goes into another; with labels it can let a sleeve compress against a
cardigan but never let a hand sink into it. The label of a vertex is its material's class (what the surface *is*) and
its skinning (what it moves with):

- class, strongest signal first: the material name (VRoid ends every name with ``_SKIN`` / ``_CLOTH`` / ``_HAIR`` /
  ``_FACE`` / ``_EYE``; other rigs say ``Body_Hands``, ``Dress``, ``Hair_Bun``), then the mesh name, then MediaPipe's
  multi-class segmentation voted over the material's pixels on rendered views, else clothing (never silently skin);
- limb: arm, hand and finger bones carry at least 0.6 of the vertex' weight; less (the shoulder, the armpit) is body.

Written into the VRM as the ``_MOVIVA_SURFACE`` vertex attribute (ids below) and the ``MOVIVA_surface_labels`` root
extension, by ``core.compress.add_surface_labels``.
"""

from __future__ import annotations

import re

CLASSES = ("skin_body", "cloth_body", "skin_limb", "cloth_limb", "hair", "accessory", "face_detail")
ID = {name: i for i, name in enumerate(CLASSES)}
LIMB_SHARE = 0.6
ARM_BONES = tuple(f"{s}{b}" for s in ("left", "right") for b in (
    "UpperArm", "LowerArm", "Hand", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "IndexProximal",
    "IndexIntermediate", "IndexDistal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "RingProximal",
    "RingIntermediate", "RingDistal", "LittleProximal", "LittleIntermediate", "LittleDistal"))

_VROID = {"SKIN": "skin", "CLOTH": "cloth", "HAIR": "hair", "FACE": "face_detail", "EYE": "face_detail"}
# class -> tokens; checked in this order, first hit wins (``Body_Teeth`` is a face detail, ``Body_Hips`` skin)
_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("face_detail", ("eye", "iris", "pupil", "cornea", "sclera", "lash", "brow", "teeth", "tooth", "tongue", "gum",
                     "mouth", "tear", "lacrimal", "eyeline", "highlight", "nostril", "innermouth", "reflection")),
    ("hair", ("hair", "bang", "fringe", "ponytail", "braid", "strand", "scalp", "skullcap", "wisp", "bun")),
    ("accessory", ("accessory", "gem", "gemstone", "metal", "earring", "necklace", "pendant", "bead", "jewel", "ring",
                   "hat", "cap", "bag", "glasses", "ribbon", "chain", "leaf", "crown", "horn", "wing", "tail")),
    ("cloth", ("cloth", "clothes", "shirt", "jacket", "coat", "cardigan", "sweater", "hoodie", "dress", "skirt",
               "pants", "trousers", "shorts", "sleeve", "cuff", "glove", "sock", "stocking", "tights", "shoe", "boot",
               "collar", "tie", "button", "uniform", "outfit", "tops", "bottoms", "vest", "bra", "armor", "armour",
               "cape", "scarf", "belt", "apron", "robe", "kimono")),
    ("skin", ("skin", "body", "face", "head", "hand", "hands", "arm", "arms", "forearm", "forearms", "leg", "legs",
              "neck", "torso", "shoulder", "shoulders", "hips", "finger", "fingernails", "nail", "ear", "ears",
              "nipple", "nipples", "lips", "foot", "feet")),
)


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name or "")
    return [t for t in re.split(r"[^A-Za-z]+", spaced.lower()) if t]


def class_from_name(name: str) -> str | None:
    """skin / cloth / hair / face_detail / accessory from a material or mesh name, or None."""
    vroid = [_VROID[p] for p in re.split(r"[^A-Za-z]+", name or "") if p in _VROID]
    if vroid:  # VRoid: "F00_002_01_Tops_01_CLOTH (Instance)"
        return vroid[-1]
    toks = _tokens(name)
    for cls, words in _TOKENS:
        if any(t in words for t in toks):
            return cls
    return None


# MediaPipe multi-class selfie segmentation categories -> class
SEGMENTER = {1: "hair", 2: "skin", 3: "skin", 4: "cloth", 5: "accessory"}


def class_from_votes(votes: dict[int, int], min_pixels: int = 50) -> str | None:
    """The class most of a material's rendered pixels were segmented as (background ignored)."""
    counted = {SEGMENTER[k]: 0 for k in SEGMENTER}
    for k, n in votes.items():
        if k in SEGMENTER:
            counted[SEGMENTER[k]] += n
    best = max(counted, key=counted.get)
    return best if counted[best] >= min_pixels else None


def decide(material: str, mesh: str, votes: dict | None = None) -> tuple[str, str]:
    """(class, signal) for one material slot: name, MediaPipe (one class, or "per-vertex" when its pixels disagree),
    mesh name, else clothing."""
    cls = class_from_name(material)
    if cls:
        return cls, "material-name"
    votes = votes or {}
    if votes.get("materials", {}).get(material):
        return votes["materials"][material], "mediapipe"
    if votes.get("points"):
        return "per-vertex", "mediapipe-per-vertex"
    cls = class_from_name(mesh)
    if cls:
        return cls, "mesh-name"
    return "cloth", "default"


def label(cls: str, arm_weight: float) -> int:
    """The label id of a vertex of class ``cls`` with ``arm_weight`` on arm, hand and finger bones."""
    limb = arm_weight >= LIMB_SHARE
    if cls == "skin":
        return ID["skin_limb" if limb else "skin_body"]
    if cls == "cloth":
        return ID["cloth_limb" if limb else "cloth_body"]
    if cls == "accessory":
        return ID["cloth_limb"] if limb else ID["accessory"]  # a bracelet or a watch moves as a sleeve does
    return ID[cls]
