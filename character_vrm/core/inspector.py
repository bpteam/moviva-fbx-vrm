"""Armature selection and inspection summaries over the ``bpy``-free scene model."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT_MIN_CONFIDENCE
from .errors import ConversionError
from .humanoid import HumanoidError, HumanoidResult, resolve_humanoid
from .model import REQUIRED_HUMAN_BONES, CharacterInspection, ConversionWarning

AMBIGUITY_MARGIN = 0.1  # two complete candidates closer than this are "ambiguous"


@dataclass
class ArmatureScore:
    name: str
    score: float
    required_mapped: int
    skinned_meshes: int
    bones: int
    humanoid: HumanoidResult


@dataclass
class ArmatureSelection:
    name: str
    scores: list[ArmatureScore]
    warnings: list[ConversionWarning] = field(default_factory=list)

    @property
    def chosen(self) -> ArmatureScore:
        return next(s for s in self.scores if s.name == self.name)


def score_armatures(
    inspection: CharacterInspection,
    explicit: dict[str, str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[ArmatureScore]:
    total_skinned_tris = sum(m.triangles for m in inspection.skinned_meshes) or 1
    max_bones = max((len(a.bones) for a in inspection.armatures), default=1) or 1
    scores = []
    for arm in inspection.armatures:
        try:
            humanoid = resolve_humanoid(arm, explicit, min_confidence)
        except HumanoidError:
            # a map file written for another armature: it simply does not fit this one
            humanoid = resolve_humanoid(arm, None, min_confidence)
        meshes = [m for m in inspection.meshes_for(arm.name) if m.skinned]
        skinned_share = sum(m.triangles for m in meshes) / total_skinned_tris
        score = (0.55 * humanoid.mapped_required / len(REQUIRED_HUMAN_BONES)
                 + 0.35 * skinned_share + 0.10 * len(arm.bones) / max_bones)
        scores.append(ArmatureScore(arm.name, round(score, 4), humanoid.mapped_required, len(meshes),
                                    len(arm.bones), humanoid))
    return sorted(scores, key=lambda s: (-s.score, s.name))


def select_armature(
    inspection: CharacterInspection,
    requested: str | None = None,
    explicit: dict[str, str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ArmatureSelection:
    if not inspection.armatures:
        raise ConversionError(
            "no-armature", "The source contains no armature. A rigged, skinned humanoid is required; "
            "this converter never creates or auto-weights a skeleton.")
    scores = score_armatures(inspection, explicit, min_confidence)
    selection: ArmatureSelection

    if requested:
        if requested not in {s.name for s in scores}:
            raise ConversionError(
                "armature-not-found", f"Requested armature '{requested}' does not exist.",
                [f"available: {[s.name for s in scores]}"])
        selection = ArmatureSelection(requested, scores)
    else:
        best = scores[0]
        rivals = [s for s in scores[1:] if s.required_mapped == len(REQUIRED_HUMAN_BONES)
                  and best.required_mapped == len(REQUIRED_HUMAN_BONES) and s.skinned_meshes > 0
                  and best.score - s.score < AMBIGUITY_MARGIN]
        if rivals:
            raise ConversionError(
                "ambiguous-armature",
                "More than one armature looks like the primary character. Choose one with --armature "
                "or the 'armature' key of the map file.",
                [f"{s.name}: score {s.score:.2f}, {s.skinned_meshes} skinned mesh(es), {s.bones} bones"
                 for s in [best, *rivals]])
        selection = ArmatureSelection(best.name, scores)
        if len(scores) > 1:
            others = ", ".join(s.name for s in scores if s.name != best.name)
            selection.warnings.append(ConversionWarning(
                "armature-secondary",
                f"Primary armature is '{best.name}'; other armature(s) ({others}) and the meshes they deform are "
                "excluded from the export."))

    chosen = selection.chosen
    if chosen.skinned_meshes == 0:
        raise ConversionError(
            "no-skinned-mesh", f"Armature '{chosen.name}' does not deform any mesh (no armature modifier with "
            "vertex groups). There is no skinning to preserve.")
    return selection
