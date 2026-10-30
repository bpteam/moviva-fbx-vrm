"""Conversion config: tunable defaults and the ``*.vrm-map.json`` schema.

Every default that a user may want to change lives here, not in the resolvers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .model import ALL_HUMAN_BONES, VRM_EXPRESSIONS, SpringPreset

# --- thresholds ------------------------------------------------------------
# A required humanoid bone below this confidence fails the conversion; an
# optional one is dropped with a warning. Anatomical-position guesses are
# deliberately below it: they must be confirmed via a map file.
DEFAULT_MIN_CONFIDENCE = 0.75
DEFAULT_EXPRESSION_MIN_CONFIDENCE = 0.65
DEFAULT_SPRING_MIN_CONFIDENCE = 0.8

# --- geometry profiles ------------------------------------------------------
# Margins are metres in the imported Blender coordinate system.  They are
# deliberately conservative: the cut is below the semantic landmark.
ARM_PROTECTION_WEIGHT = 0.05
VIDEO_CALL_WAIST_FALLBACK_T = 0.72
VIDEO_CALL_BODY_SAFETY_MARGIN = 0.12
FACE_BODY_SAFETY_MARGIN = 0.08


@dataclass(frozen=True)
class GeometryProfileSpec:
    enabled: bool
    semantic_frame: str
    cut_landmark: str | None
    body_safety_margin: float
    preserve_full_arms: bool
    preserve_full_hair: bool
    preserve_head: bool
    fallback_t: float | None = None

    def __getitem__(self, key: str):
        """Allow diagnostics/tests to read the central config like a dict."""
        return getattr(self, key)


GEOMETRY_PROFILES: dict[str, GeometryProfileSpec] = {
    "full": GeometryProfileSpec(False, "full-body", None, 0.0, True, True, True),
    "video-call": GeometryProfileSpec(
        True, "waist-up", "waist", VIDEO_CALL_BODY_SAFETY_MARGIN, True, True, True,
        VIDEO_CALL_WAIST_FALLBACK_T),
    "face": GeometryProfileSpec(True, "bust", "upper-chest", FACE_BODY_SAFETY_MARGIN, True, True, True),
}

# --- spring bones ----------------------------------------------------------
# Conservative on purpose: stiff, damped, nearly weightless. Bad or absent hair
# physics is preferable to exploding hair.
DEFAULT_SPRING_PRESET = SpringPreset(stiffness=1.0, drag_force=0.5, gravity_power=0.1, hit_radius=0.01)
REFERENCE_HEIGHT_M = 1.7  # hit radius / collider sizes scale with skeleton height / this

# token -> confidence that a bone carrying it is hair
HAIR_TOKENS: dict[str, float] = {
    "hair": 0.95,
    "bang": 0.95,
    "bangs": 0.95,
    "fringe": 0.95,
    "ponytail": 0.95,
    "braid": 0.9,
    "strand": 0.8,
    "lock": 0.6,  # ambiguous ("lock" of a door): never automatic
}
MIN_SPRING_CHAIN_BONES = 2
MAX_SPRING_CHAIN_BONES = 16

# --- collider proportions (fractions of skeleton height) -------------------
HEAD_COLLIDER_RADIUS = (0.045, 0.075)  # clamp range
CHEST_COLLIDER_RADIUS = (0.06, 0.09)

# --- report warnings (never limits) ----------------------------------------
WARN_TRIANGLES = 1_000_000
WARN_MATERIALS = 64
WARN_TEXTURE_SIZE = 8192
HEIGHT_RANGE_M = (0.5, 3.0)


class ConfigError(ValueError):
    pass


@dataclass
class SpringConfig:
    enabled: bool = True
    include: list[str] = field(default_factory=list)  # chain root bones (chain follows the single child line)
    ignore: list[str] = field(default_factory=list)
    preset: SpringPreset = DEFAULT_SPRING_PRESET
    colliders: bool = True


@dataclass
class ExpressionOverride:
    disabled: bool = False
    binds: dict[str, float] = field(default_factory=dict)  # shape key name -> weight


@dataclass
class MetaConfig:
    """Only what the user states. Licence fields the user does not set keep the add-on's
    restrictive defaults; nothing is inferred from the source asset."""

    name: str | None = None
    version: str | None = None
    authors: list[str] = field(default_factory=list)
    copyright_information: str | None = None
    contact_information: str | None = None
    references: list[str] = field(default_factory=list)
    third_party_licenses: str | None = None
    avatar_permission: str | None = None
    commercial_usage: str | None = None
    credit_notation: str | None = None
    modification: str | None = None
    allow_redistribution: bool | None = None
    allow_excessively_violent_usage: bool | None = None
    allow_excessively_sexual_usage: bool | None = None
    allow_political_or_religious_usage: bool | None = None
    allow_antisocial_or_hate_usage: bool | None = None
    other_license_url: str | None = None


@dataclass
class LookAtConfig:
    type: str | None = None  # "bone" | "expression" | "none"
    offset_from_head_bone: tuple[float, float, float] | None = None


@dataclass
class GeometryConfig:
    profile: str = "full"
    keep_objects: list[str] = field(default_factory=list)
    drop_objects: list[str] = field(default_factory=list)


@dataclass
class AutoAuthorConfig:
    """Avatar Auto-Author switches. Conservative by default: nothing is generated unless asked for.

    ``enabled`` (``--auto-author``): rest-pose calibration, hair -> spring bones from the anatomy analysis, anatomical
    colliders, and generated blink / visemes *when confidence is high*. The three ``generate_*`` flags are the
    experimental extras and stay off until they have real-model coverage.
    """

    enabled: bool = False
    generate_face: bool = False         # + basic generated expressions (smile / sad / surprise / brows / squint)
    generate_finger_rig: bool = False   # experimental: bones for fingers that exist only as geometry
    generate_hair_rig: bool = False     # experimental: bones for a hair mesh that has none
    confidence: dict[str, float] = field(default_factory=dict)  # overrides of ``anatomy.confidence.CONFIDENCE``

    @property
    def active(self) -> bool:
        return self.enabled or self.generate_face or self.generate_finger_rig or self.generate_hair_rig


@dataclass
class ConversionConfig:
    armature: str | None = None
    humanoid: dict[str, str] = field(default_factory=dict)
    expressions: dict[str, ExpressionOverride] = field(default_factory=dict)
    look_at: LookAtConfig = field(default_factory=LookAtConfig)
    spring: SpringConfig = field(default_factory=SpringConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    expression_min_confidence: float = DEFAULT_EXPRESSION_MIN_CONFIDENCE
    spring_min_confidence: float = DEFAULT_SPRING_MIN_CONFIDENCE
    all_bone_influences: bool = False  # reserved; the add-on default (<= 4 influences) is used
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    auto_author: AutoAuthorConfig = field(default_factory=AutoAuthorConfig)


_TOP_KEYS = {"armature", "humanoid", "expressions", "lookAt", "springBones", "meta", "geometry", "autoAuthor"}
_META_KEYS = {
    "name": "name", "version": "version", "authors": "authors",
    "copyrightInformation": "copyright_information", "contactInformation": "contact_information",
    "references": "references", "thirdPartyLicenses": "third_party_licenses",
    "avatarPermission": "avatar_permission", "commercialUsage": "commercial_usage",
    "creditNotation": "credit_notation", "modification": "modification",
    "allowRedistribution": "allow_redistribution",
    "allowExcessivelyViolentUsage": "allow_excessively_violent_usage",
    "allowExcessivelySexualUsage": "allow_excessively_sexual_usage",
    "allowPoliticalOrReligiousUsage": "allow_political_or_religious_usage",
    "allowAntisocialOrHateUsage": "allow_antisocial_or_hate_usage",
    "otherLicenseUrl": "other_license_url",
}
# Values the add-on's enum properties accept (editor/vrm1/property_group.py, Vrm1MetaPropertyGroup).
_META_ENUMS = {
    "avatarPermission": {"onlyAuthor", "onlySeparatelyLicensedPerson", "everyone"},
    "commercialUsage": {"personalNonProfit", "personalProfit", "corporation"},
    "creditNotation": {"required", "unnecessary"},
    "modification": {"prohibited", "allowModification", "allowModificationRedistribution"},
}
_SPRING_PRESET_KEYS = {
    "stiffness": "stiffness", "dragForce": "drag_force",
    "gravityPower": "gravity_power", "hitRadius": "hit_radius",
}


def _reject_unknown(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _expression_override(name: str, raw: Any) -> ExpressionOverride:
    if name not in VRM_EXPRESSIONS:
        raise ConfigError(f"expressions.{name}: not a VRM expression; allowed: {list(VRM_EXPRESSIONS)}")
    if raw is None or raw is False:
        return ExpressionOverride(disabled=True)
    if isinstance(raw, str):
        return ExpressionOverride(binds={raw: 1.0})
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return ExpressionOverride(binds={x: 1.0 for x in raw})
    if isinstance(raw, dict):
        binds: dict[str, float] = {}
        for key, weight in raw.items():
            if not isinstance(weight, (int, float)) or not 0.0 <= float(weight) <= 1.0:
                raise ConfigError(f"expressions.{name}.{key}: weight must be a number in [0, 1]")
            binds[key] = float(weight)
        if not binds:
            raise ConfigError(f"expressions.{name}: empty binding")
        return ExpressionOverride(binds=binds)
    raise ConfigError(f"expressions.{name}: expected string, list of strings, object or null")


def parse_config(data: dict[str, Any]) -> ConversionConfig:
    if not isinstance(data, dict):
        raise ConfigError("config root must be an object")
    _reject_unknown("config", {k: v for k, v in data.items() if not k.startswith("_")}, _TOP_KEYS)
    cfg = ConversionConfig()

    geometry = data.get("geometry") or {}
    _reject_unknown("geometry", geometry, {"profile", "keepObjects", "dropObjects"})
    profile = geometry.get("profile", "full")
    if profile not in GEOMETRY_PROFILES:
        raise ConfigError(f"geometry.profile: expected one of {sorted(GEOMETRY_PROFILES)}")
    cfg.geometry.profile = profile
    for key, attr in (("keepObjects", "keep_objects"), ("dropObjects", "drop_objects")):
        value = geometry.get(key, [])
        if not (isinstance(value, list) and all(isinstance(x, str) and x for x in value)):
            raise ConfigError(f"geometry.{key}: expected a list of object names")
        setattr(cfg.geometry, attr, list(value))

    author = data.get("autoAuthor") or {}
    _reject_unknown("autoAuthor", author, {"enabled", "generateFace", "generateFingerRig", "generateHairRig", "confidence"})
    for key, attr in (("enabled", "enabled"), ("generateFace", "generate_face"),
                      ("generateFingerRig", "generate_finger_rig"), ("generateHairRig", "generate_hair_rig")):
        if key in author:
            if not isinstance(author[key], bool):
                raise ConfigError(f"autoAuthor.{key}: expected true or false")
            setattr(cfg.auto_author, attr, author[key])
    from .anatomy.confidence import CONFIDENCE
    for key, value in (author.get("confidence") or {}).items():
        if key not in CONFIDENCE:
            raise ConfigError(f"autoAuthor.confidence.{key}: unknown threshold; allowed: {sorted(CONFIDENCE)}")
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"autoAuthor.confidence.{key}: expected a number in [0, 1]")
        cfg.auto_author.confidence[key] = float(value)

    if (armature := data.get("armature")) is not None:
        if not isinstance(armature, str):
            raise ConfigError("armature: expected string")
        cfg.armature = armature

    for vrm_bone, source in (data.get("humanoid") or {}).items():
        if vrm_bone.startswith("_"):
            continue
        if vrm_bone not in ALL_HUMAN_BONES:
            raise ConfigError(f"humanoid.{vrm_bone}: not a VRM 1.0 humanoid bone")
        if not isinstance(source, str) or not source:
            raise ConfigError(f"humanoid.{vrm_bone}: expected a bone name")
        cfg.humanoid[vrm_bone] = source

    for name, raw in (data.get("expressions") or {}).items():
        if name.startswith("_"):
            continue
        cfg.expressions[name] = _expression_override(name, raw)

    look_at = data.get("lookAt") or {}
    _reject_unknown("lookAt", look_at, {"type", "offsetFromHeadBone"})
    if (look_type := look_at.get("type")) is not None:
        if look_type not in ("bone", "expression", "none"):
            raise ConfigError("lookAt.type: expected 'bone', 'expression' or 'none'")
        cfg.look_at.type = look_type
    if (offset := look_at.get("offsetFromHeadBone")) is not None:
        if not (isinstance(offset, list) and len(offset) == 3 and all(isinstance(x, (int, float)) for x in offset)):
            raise ConfigError("lookAt.offsetFromHeadBone: expected [x, y, z]")
        cfg.look_at.offset_from_head_bone = (float(offset[0]), float(offset[1]), float(offset[2]))

    spring = data.get("springBones") or {}
    _reject_unknown("springBones", spring, {"enabled", "include", "ignore", "preset", "colliders"})
    for key in ("include", "ignore"):
        value = spring.get(key, [])
        if not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise ConfigError(f"springBones.{key}: expected a list of bone names")
        setattr(cfg.spring, key, list(value))
    if "enabled" in spring:
        cfg.spring.enabled = bool(spring["enabled"])
    if "colliders" in spring:
        cfg.spring.colliders = bool(spring["colliders"])
    if preset := spring.get("preset"):
        _reject_unknown("springBones.preset", preset, set(_SPRING_PRESET_KEYS))
        overrides = {}
        for key, attr in _SPRING_PRESET_KEYS.items():
            if key in preset:
                if not isinstance(preset[key], (int, float)) or preset[key] < 0:
                    raise ConfigError(f"springBones.preset.{key}: expected a non-negative number")
                overrides[attr] = float(preset[key])
        cfg.spring.preset = replace(DEFAULT_SPRING_PRESET, **overrides)

    meta = data.get("meta") or {}
    _reject_unknown("meta", meta, set(_META_KEYS))
    for key, value in meta.items():
        if key in _META_ENUMS and value not in _META_ENUMS[key]:
            raise ConfigError(f"meta.{key}: expected one of {sorted(_META_ENUMS[key])}")
        setattr(cfg.meta, _META_KEYS[key], value)
    return cfg


def load_config(path: str | Path) -> ConversionConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    return parse_config(raw)
