"""Rig-profile plumbing shared by the profile modules (pure Python)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from ..core.model import REQUIRED_HUMAN_BONES

MIN_PROFILE_SCORE = 0.6  # fraction of required bones a profile must find to be selected


@dataclass(frozen=True)
class BoneProfile:
    name: str
    #: bone names -> (vrm_bone -> source bone, score) or None when the rig is not this profile
    build: Callable[[frozenset[str]], "tuple[dict[str, str], float] | None"]


def finish(table: dict[str, str], names: Iterable[str]) -> "tuple[dict[str, str], float] | None":
    """Keep entries whose source bone exists; score = share of required bones present."""
    present = frozenset(names)
    mapping = {vrm: src for vrm, src in table.items() if src in present}
    score = sum(1 for b in REQUIRED_HUMAN_BONES if b in mapping) / len(REQUIRED_HUMAN_BONES)
    return (mapping, score) if score >= MIN_PROFILE_SCORE else None


def limb_table(
    fmt: Callable[[str, str], str],
    sides: tuple[tuple[str, str], tuple[str, str]] = (("left", "L"), ("right", "R")),
    *,
    shoulder: str, upper_arm: str, lower_arm: str, hand: str,
    upper_leg: str, lower_leg: str, foot: str, toes: str,
    fingers: Callable[[str, str, int], str] | None = None,
    eye: str | None = None,
) -> dict[str, str]:
    """Build the per-side half of a profile. ``fmt(side_tag, base)`` renders a bone name."""
    table: dict[str, str] = {}
    for vrm_side, tag in sides:
        table[f"{vrm_side}Shoulder"] = fmt(tag, shoulder)
        table[f"{vrm_side}UpperArm"] = fmt(tag, upper_arm)
        table[f"{vrm_side}LowerArm"] = fmt(tag, lower_arm)
        table[f"{vrm_side}Hand"] = fmt(tag, hand)
        table[f"{vrm_side}UpperLeg"] = fmt(tag, upper_leg)
        table[f"{vrm_side}LowerLeg"] = fmt(tag, lower_leg)
        table[f"{vrm_side}Foot"] = fmt(tag, foot)
        table[f"{vrm_side}Toes"] = fmt(tag, toes)
        if eye:
            table[f"{vrm_side}Eye"] = fmt(tag, eye)
        if fingers:
            for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
                parts = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else (
                    "Proximal", "Intermediate", "Distal")
                for i, part in enumerate(parts, start=1):
                    table[f"{vrm_side}{finger}{part}"] = fingers(tag, finger, i)
    return table
