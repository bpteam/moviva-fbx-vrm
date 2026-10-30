"""Bone-name normalisation: ``mixamorig:LeftArm``, ``CC_Base_L_Upperarm``,
``upper_arm.L``, ``LeftUpperArm``, ``L_UpperArm`` all parse to side=left, part=upperarm."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..profiles import generic as vocab

_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SEPARATORS = re.compile(r"[\s_.\-:|/]+")
_MIXAMO_NS = re.compile(r"^mixamorig\d*$")


@dataclass
class ParsedBone:
    raw: str
    side: str | None = None  # "left" | "right" | None
    part: str | None = None
    index: int | None = None
    twist: bool = False
    helper: bool = False
    end: bool = False
    tokens: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """A main-chain candidate: not a twist / helper / end-site bone."""
        return not (self.twist or self.helper or self.end)


def _tokenise(name: str) -> list[str]:
    tokens: list[str] = []
    for chunk in _SEPARATORS.split(name):
        tokens.extend(t.lower() for t in _CAMEL.findall(chunk))
    return tokens


def parse_bone_name(name: str) -> ParsedBone:
    tokens = [t for t in _tokenise(name)]
    parsed = ParsedBone(raw=name, tokens=list(tokens))
    words: list[str] = []
    previous = ""
    for token in tokens:
        was_previous, previous = previous, token
        if token.isdigit():
            # "mixamorig1:", "Bip01": the digits belong to the namespace, not to the bone.
            if was_previous not in ("mixamorig", "bip") and parsed.index is None:
                parsed.index = int(token)
            continue
        if _MIXAMO_NS.match(token) or token in vocab.IGNORED_TOKENS:
            continue
        if token in vocab.LEFT_TOKENS:
            parsed.side = parsed.side or "left"
            continue
        if token in vocab.RIGHT_TOKENS:
            parsed.side = parsed.side or "right"
            continue
        if token in vocab.TWIST_TOKENS:
            parsed.twist = True
            continue
        if token in vocab.HELPER_TOKENS:
            parsed.helper = True
            continue
        if token in vocab.END_TOKENS:
            parsed.end = True
            continue
        words.append(token)

    merged: list[str] = []
    i = 0
    while i < len(words):
        pair = tuple(words[i:i + 2])
        if len(pair) == 2 and pair in vocab.MERGES:
            merged.append(vocab.MERGES[pair])  # type: ignore[index]
            i += 2
        else:
            merged.append(words[i])
            i += 1

    parts = [vocab.PARTS[w] for w in merged if w in vocab.PARTS]
    # "HandThumb1": the finger wins over the hand it belongs to.
    finger = next((p for p in parts if p in vocab.FINGER_PARTS), None)
    if finger:
        parsed.part = finger
    elif parts:
        parsed.part = parts[0]
    # Unknown words are normally a harmless model/rig prefix ("Knight_LeftArm"); only words
    # that name a different object make the bone a non-humanoid helper ("HeadBand").
    if parsed.part and any(w in _DISQUALIFYING for w in merged):
        parsed.helper = True
    return parsed


_DISQUALIFYING = frozenset({
    "band", "prop", "lash", "lashes", "lid", "brow", "tongue", "teeth", "tooth", "accessory", "weapon",
    "glove", "sleeve", "shoe", "sock", "hair", "earring", "necklace", "ribbon",
})
