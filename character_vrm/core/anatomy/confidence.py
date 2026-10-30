"""Provenance, confidence fusion and the central confidence thresholds.

Every inferred or generated fact carries ``confidence`` and an ``AnalysisSource``. Never present a guess as
source-authored data (``Provenance.KNOWN``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AnalysisSource(StrEnum):
    EXPLICIT = "explicit"        # the map file / the user said so
    RIG_PROFILE = "rig-profile"  # a recognised rig convention (Mixamo, VRoid, CC, ...)
    SKELETON = "skeleton"        # names / topology / bone positions
    GEOMETRY = "geometry"        # mesh surface analysis
    MATERIAL = "material"        # material / texture evidence
    GENERATED = "generated"      # synthesised by the Auto-Author, never in the source


class Provenance(StrEnum):
    KNOWN = "KNOWN"          # source-authored
    INFERRED = "INFERRED"    # deduced from evidence, not authored
    GENERATED = "GENERATED"  # created by the converter
    MISSING = "MISSING"


# The report vocabulary (SOURCE / MAPPED / INFERRED / GENERATED / MISSING).
STATUS_SOURCE, STATUS_MAPPED, STATUS_INFERRED, STATUS_GENERATED, STATUS_MISSING = (
    "SOURCE", "MAPPED", "INFERRED", "GENERATED", "MISSING")

# Single place for every gate; modules take these from here (or from a ``Thresholds`` instance), never inline.
CONFIDENCE: dict[str, float] = {
    "automatic_mapping": 0.80,
    "generated_face": 0.85,
    "generated_fingers": 0.90,
    "generated_hair": 0.90,
    "hair_confident": 0.75,
    "hair_possible": 0.40,
    "material_semantic": 0.60,
    "landmark_usable": 0.50,
    "frame_usable": 0.60,
}


@dataclass(frozen=True)
class Thresholds:
    automatic_mapping: float = CONFIDENCE["automatic_mapping"]
    generated_face: float = CONFIDENCE["generated_face"]
    generated_fingers: float = CONFIDENCE["generated_fingers"]
    generated_hair: float = CONFIDENCE["generated_hair"]
    hair_confident: float = CONFIDENCE["hair_confident"]
    hair_possible: float = CONFIDENCE["hair_possible"]
    material_semantic: float = CONFIDENCE["material_semantic"]
    landmark_usable: float = CONFIDENCE["landmark_usable"]
    frame_usable: float = CONFIDENCE["frame_usable"]


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class Evidence:
    """Named evidence scores in [0, 1]. Kept un-collapsed so reports can show the diagnostics."""

    scores: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, score: float) -> "Evidence":
        self.scores[name] = max(0.0, min(1.0, score))
        return self

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.scores.items()}


DEFAULT_EVIDENCE_WEIGHTS = {"topology": 0.30, "position": 0.25, "symmetry": 0.20, "skinning": 0.15, "name": 0.30}


def fuse(evidence: dict[str, float], weights: dict[str, float] | None = None, veto_below: float = 0.30,
         veto_cap: float = 0.60) -> float:
    """Weighted mean over the evidence that is present, with a veto.

    A single strongly contradicting signal (< ``veto_below``) caps the result at ``veto_cap`` no matter how
    good the rest is: two agreeing signals must not outvote a plainly wrong one.
    """
    weights = weights or DEFAULT_EVIDENCE_WEIGHTS
    used = {k: v for k, v in evidence.items() if k in weights}
    if not used:
        return 0.0
    total = sum(weights[k] for k in used)
    value = sum(weights[k] * v for k, v in used.items()) / total
    if any(v < veto_below for v in used.values()):
        value = min(value, veto_cap)
    return max(0.0, min(1.0, value))


@dataclass
class AnalysisWarning:
    code: str
    message: str
    area: str = "analysis"
