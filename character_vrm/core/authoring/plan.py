"""``AutoAuthorPlan``: the whole authoring stage as one inspectable object.

    ANALYZE (HumanModelAnalysis) -> PLAN (this) -> VALIDATE PLAN -> MUTATE (blender/*_author.py) -> VALIDATE RESULT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dataclasses import replace as dc_replace

from ..anatomy.confidence import CONFIDENCE, DEFAULT_THRESHOLDS, Thresholds
from ..anatomy.model import HumanModelAnalysis
from ..config import AutoAuthorConfig
from ..expressions import ExpressionResult
from ..model import ArmatureInfo, CharacterInspection, ExpressionMapping, MorphBind, SpringChain, SpringPlan
from .colliders import plan_colliders
from .expressions import plan_face_authoring
from .face_plan import FaceAuthoringPlan
from .fingers import FingerRigPlan, plan_all_hands
from .hair_rig import HairRigPlan, plan_hair_rig
from .materials import MaterialPlan, index_textures, plan_materials
from .plans import PlanIssue, ShapeKeyPlan, Skipped
from .rest_pose import RestPosePlan, plan_rest_pose
from .springbones import plan_hair_springs, preset_for


def thresholds_from(config: AutoAuthorConfig) -> Thresholds:
    return dc_replace(DEFAULT_THRESHOLDS, **{k: v for k, v in config.confidence.items() if k in CONFIDENCE})


@dataclass
class AutoAuthorPlan:
    config: AutoAuthorConfig
    armature: str = ""
    rest_pose: RestPosePlan | None = None
    face: FaceAuthoringPlan | None = None
    fingers: list[FingerRigPlan] = field(default_factory=list)
    hair_rig: HairRigPlan | None = None
    springs: SpringPlan | None = None
    materials: MaterialPlan | None = None
    skipped: list[Skipped] = field(default_factory=list)
    issues: list[PlanIssue] = field(default_factory=list)
    hair_spring_skipped: list[Skipped] = field(default_factory=list)
    collider_skipped: list[Skipped] = field(default_factory=list)

    @property
    def shape_keys(self) -> list[ShapeKeyPlan]:
        return [k for k in (self.face.shape_keys if self.face else []) if k.name in self.face._valid_names()]

    @property
    def needs_mutation(self) -> bool:
        return bool(self.shape_keys or any(f.bones for f in self.fingers if not f.validate())
                    or (self.materials and self.materials.valid_edits())
                    or (self.hair_rig and self.hair_rig.chains and not self.hair_rig.validate()))

    def expression_mappings(self) -> dict[str, ExpressionMapping]:
        return self.face.expression_mappings() if self.face else {}


def build_authoring_plan(
    analysis: HumanModelAnalysis, inspection: CharacterInspection, armature: ArmatureInfo,
    expressions: ExpressionResult, springs: SpringPlan, config: AutoAuthorConfig, spring_enabled: bool = True,
) -> AutoAuthorPlan:
    """PLAN + VALIDATE PLAN. Never touches a scene."""
    th = thresholds_from(config)
    plan = AutoAuthorPlan(config, armature.name)
    plan.rest_pose = plan_rest_pose(analysis)
    plan.issues += plan.rest_pose.validate()

    plan.face = plan_face_authoring(analysis, expressions, generate_visemes_blink=config.enabled or config.generate_face,
                                    generate_basic=config.generate_face, thresholds=th)
    plan.issues += plan.face.validate(analysis)

    plan.fingers = plan_all_hands(analysis, config.generate_finger_rig, th)
    for f in plan.fingers:
        plan.issues += f.validate()
        if f.skipped:
            plan.skipped.append(f.skipped)

    # the maps a Character Creator style export ships beside the model file (none when there is no such file)
    textures = index_textures(Path(inspection.source_path).parent) if os.path.isfile(inspection.source_path) else {}
    plan.materials = plan_materials(analysis.materials, inspection, textures)
    plan.issues += plan.materials.validate()
    plan.skipped += plan.materials.skipped

    plan.hair_rig = plan_hair_rig(analysis, inspection, armature.name, config.generate_hair_rig, th)
    plan.issues += plan.hair_rig.validate()
    plan.skipped += plan.hair_rig.skipped

    if not spring_enabled:
        plan.springs = None
        return plan
    merged, hair_skipped = plan_hair_springs(analysis, springs, th)
    plan.hair_spring_skipped = hair_skipped
    colliders, collider_skipped = plan_colliders(analysis, armature)
    plan.collider_skipped = collider_skipped
    # Generated hair bones (experimental, off by default) get spring chains from their category preset.
    if plan.hair_rig and plan.hair_rig.chains and not plan.hair_rig.validate():
        height = analysis.measurements.total_height_units or 1.7
        for chain in plan.hair_rig.chains:
            merged.chains.append(SpringChain(
                chain.name, [b.name for b in chain.bones], chain.confidence,
                f"generated hair rig ({chain.category})",
                preset_for(chain.category, 1.0, height, analysis.head_width)))
    # Anatomy-derived colliders replace the name-based ones whenever there is anything to collide with.
    if merged.chains and colliders:
        merged.colliders = colliders
    plan.springs = merged
    return plan


@dataclass
class AuthoringResult:
    """What the Blender mutation stage reports back (VALIDATE RESULT)."""

    applied_shape_keys: dict[str, list[str]] = field(default_factory=dict)  # key name -> meshes it exists on
    created_bones: set[str] = field(default_factory=set)
    materials: dict[str, list[str]] = field(default_factory=dict)           # material -> changes made
    failed: dict[str, str] = field(default_factory=dict)                    # what -> reason
    validation: list[str] = field(default_factory=list)


def apply_authoring_result(vrm_plan, authoring: AutoAuthorPlan, result: AuthoringResult) -> list[str]:
    """Fold a *validated* mutation result into the VRM plan. Existing mappings are never replaced."""
    notes: list[str] = []
    if authoring.face is not None:
        for expr, mapping in authoring.expression_mappings().items():
            binds = [b for b in mapping.binds if b.mesh in result.applied_shape_keys.get(b.shape_key, [])]
            if not binds or expr in vrm_plan.expressions:
                if not binds:
                    notes.append(f"{expr}: generated key was not created; not bound")
                continue
            vrm_plan.expressions[expr] = ExpressionMapping(expr, mapping.kind, binds, mapping.confidence, mapping.reason)
        for expr, adds in authoring.face.augment.items():
            current = vrm_plan.expressions.get(expr)
            if current is None:
                continue
            extra = [MorphBind(mesh, key, weight) for key, weight in adds
                     for mesh in result.applied_shape_keys.get(key, [])]
            if not extra:
                notes.append(f"{expr}: generated jaw key was not created; the existing mapping stays as it was")
                continue
            vrm_plan.expressions[expr] = ExpressionMapping(
                expr, current.kind, [*current.binds, *extra], current.confidence,
                f"{current.reason} + {', '.join(k for k, _ in adds)}")
    for f in authoring.fingers:
        names = {b.name for b in f.bones}
        if f.bones and names <= result.created_bones:
            vrm_plan.humanoid.update(f.humanoid)
        elif f.bones:
            notes.append(f"{f.side} finger rig was not created; fingers stay unmapped")
    if authoring.springs is not None:
        valid_generated = result.created_bones
        generated = {b.name for c in (authoring.hair_rig.chains if authoring.hair_rig else []) for b in c.bones}
        keep = [c for c in authoring.springs.chains if not (set(c.bones) & generated) or set(c.bones) <= valid_generated]
        vrm_plan.springs = SpringPlan(keep, authoring.springs.colliders if keep else [], authoring.springs.skipped)
    return notes
