"""Known rig / shape-key profiles. Pure data + small builders; no ``bpy``."""

from __future__ import annotations

from . import character_creator, mixamo, rigify, vroid
from .base import BoneProfile

#: order matters only for ties
BONE_PROFILES: tuple[BoneProfile, ...] = (
    mixamo.PROFILE, character_creator.PROFILE, rigify.PROFILE, vroid.PROFILE,
)

__all__ = ["BONE_PROFILES", "BoneProfile"]
