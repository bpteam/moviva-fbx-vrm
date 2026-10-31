"""ConverterCore: the whole conversion, expressed against the BlenderBridge protocol only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..toolchain import Toolchain, load_toolchain
from .config import HEIGHT_RANGE_M, ConversionConfig
from .errors import ConversionError
from .expressions import BLINKS, VISEMES, ExpressionConfigError, ExpressionResult, resolve_expressions
from . import events
from .autorig import AutoRigResult
from .anatomy.model import HumanModelAnalysis, analyze_human
from .anatomy.morph_signatures import classify_morphs
from .authoring.plan import AuthoringResult, AutoAuthorPlan, apply_authoring_result, build_authoring_plan, thresholds_from
from .authoring.visemes import VISEMES as _VISEMES
from .geometry import plan_geometry
from .humanoid import HumanoidError, HumanoidResult, resolve_humanoid
from .inspector import ArmatureSelection, select_armature
from .lookat import plan_lookat
from .materials import MaterialsResult, analyze_materials
from .model import (
    FINGER_BONES, ArmatureInfo, CharacterInspection, ConversionWarning, GeometryOptimizationPlan,
    BoneMapping, ExpressionMapping, LookAtPlan, MorphBind, SpringPlan, VrmPlan,
)
from .report import build_report
from .springbones import plan_spring_bones
from .validation import OutputCheck, check_vrm_output, require_humanoid

if TYPE_CHECKING:  # the core depends on the protocol's shape only, never on Blender-facing code
    from ..blender.bridge import BlenderBridge

SUPPORTED_FORMATS = {".glb": "glb", ".gltf": "gltf", ".fbx": "fbx"}

FBX_NO_SHAPE_KEYS = (
    "Source FBX imported successfully, but no facial shape keys were available after Blender import. "
    "Humanoid conversion can continue. Blink / visemes / facial expressions cannot be generated from "
    "nonexistent morph targets. Prefer a GLB/glTF source containing morph targets when available."
)
NO_SHAPE_KEYS = (
    "The source contains no facial shape keys. Humanoid conversion can continue, but blink / visemes / "
    "facial expressions cannot be generated from nonexistent morph targets."
)


def source_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ConversionError("unsupported-format", f"Unsupported source format '{suffix}'. "
                              f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}.")
    return SUPPORTED_FORMATS[suffix]


@dataclass
class Analysis:
    inspection: CharacterInspection
    selection: ArmatureSelection
    humanoid: HumanoidResult
    expressions: ExpressionResult
    look_at: LookAtPlan
    springs: SpringPlan
    materials: MaterialsResult
    geometry: GeometryOptimizationPlan
    config: ConversionConfig
    excluded_meshes: list[str] = field(default_factory=list)
    warnings: list[ConversionWarning] = field(default_factory=list)
    anatomy: HumanModelAnalysis | None = None
    authoring: AutoAuthorPlan | None = None
    authoring_result: AuthoringResult | None = None

    @property
    def armature(self) -> ArmatureInfo:
        return self.inspection.armature(self.selection.name)  # type: ignore[return-value]

    @property
    def convertible(self) -> tuple[bool, list[str]]:
        reasons = [f"missing required bone {b}" for b in self.humanoid.missing_required]
        reasons += self.humanoid.errors
        return not reasons, reasons


def analyze(inspection: CharacterInspection, config: ConversionConfig) -> Analysis:
    selection = select_armature(inspection, config.armature, config.humanoid, config.min_confidence)
    armature = inspection.armature(selection.name)
    assert armature is not None
    try:
        humanoid = resolve_humanoid(armature, config.humanoid, config.min_confidence)
    except HumanoidError as exc:
        raise ConversionError("config-invalid", str(exc)) from exc

    excluded = [m.name for m in inspection.meshes if m.armature not in (None, selection.name)]
    included = {m.name for m in inspection.meshes} - set(excluded)
    try:
        expressions = resolve_expressions(inspection, config.expressions, config.expression_min_confidence, included)
    except ExpressionConfigError as exc:
        raise ConversionError("config-invalid", str(exc)) from exc

    # One canonical anatomy analysis feeds everything below (look-at, springs, geometry, authoring, report).
    thresholds = thresholds_from(config.auto_author)
    anatomy = analyze_human(inspection, armature, humanoid, thresholds,
                            existing_visemes=[v for v in _VISEMES if v in expressions.mappings],
                            expression_binds={e: [(b.mesh, b.shape_key, b.weight) for b in m.binds]
                                              for e, m in expressions.mappings.items()})
    if anatomy is not None:
        _apply_inferred_bones(humanoid, anatomy, config.min_confidence)
        _infer_expressions_by_motion(inspection, expressions, anatomy, included, config)
    look_at, look_warnings = plan_lookat(humanoid, expressions, config.look_at)
    springs = plan_spring_bones(armature, humanoid, inspection.weighted_bone_names(selection.name),
                                config.spring, config.spring_min_confidence)
    materials = analyze_materials(inspection)
    geometry = plan_geometry(inspection, armature, humanoid, config.geometry, anatomy)

    analysis = Analysis(inspection, selection, humanoid, expressions, look_at, springs, materials, geometry,
                        config, excluded)
    analysis.anatomy = anatomy
    analysis.warnings = _collect_warnings(analysis, look_warnings)
    if config.auto_author.active:
        if anatomy is None:
            analysis.warnings.append(ConversionWarning(
                "auto-author-unavailable", "Anatomy analysis could not build a frame: nothing is authored."))
        else:
            analysis.authoring = build_authoring_plan(anatomy, inspection, armature, expressions, springs,
                                                      config.auto_author, config.spring.enabled)
            for issue in analysis.authoring.issues:
                analysis.warnings.append(ConversionWarning("authoring-plan-issue", f"{issue.plan}: {issue.message}"))
    return analysis


def _infer_expressions_by_motion(inspection: CharacterInspection, expressions: ExpressionResult,
                                 anatomy: HumanModelAnalysis, included: set[str], config: ConversionConfig) -> None:
    """Visemes / blinks the names did not give, recognised from what the existing keys do to the face."""
    if anatomy.face is None:
        return
    disabled = {n for n, o in config.expressions.items() if o.disabled}
    wanted = {e for e in (*VISEMES, *BLINKS) if e not in expressions.mappings and e not in disabled}
    if not wanted:
        return
    taken = {b.shape_key for m in expressions.mappings.values() for b in m.binds}
    keys = [k for k in inspection.shape_keys if k.mesh in included]
    for p in classify_morphs(inspection, anatomy.face, anatomy.frame, wanted, taken):
        if p.confidence < config.expression_min_confidence:
            continue
        binds = [MorphBind(k.mesh, k.name, 1.0) for k in keys if k.name == p.key]
        if not binds:
            continue
        expressions.mappings[p.expression] = ExpressionMapping(p.expression, "motion", binds, p.confidence,
                                                               f"morph motion: {p.detail}")
        if p.expression in expressions.missing:
            expressions.missing.remove(p.expression)
    if "blink" in wanted and "blink" not in expressions.mappings and \
            "blinkLeft" in expressions.mappings and "blinkRight" in expressions.mappings:
        left, right = expressions.mappings["blinkLeft"], expressions.mappings["blinkRight"]
        expressions.mappings["blink"] = ExpressionMapping(
            "blink", "compound", [*left.binds, *right.binds], min(left.confidence, right.confidence),
            "blinkLeft+blinkRight")
        if "blink" in expressions.missing:
            expressions.missing.remove("blink")


def _apply_inferred_bones(humanoid: HumanoidResult, anatomy: HumanModelAnalysis, min_confidence: float) -> None:
    """Bones the anatomy analysis mapped from topology / position where names said nothing (confident only)."""
    for vrm, source in anatomy.inferred_bones.items():
        if vrm in humanoid.mappings:
            continue
        assessment = anatomy.skeleton.assessments.get(vrm)
        confidence = assessment.confidence if assessment else 0.8
        evidence = assessment.evidence if assessment else {}
        humanoid.mappings[vrm] = BoneMapping(vrm, source, confidence, f"anatomy analysis (evidence {evidence})")
    humanoid.missing_required = [b for b in humanoid.missing_required if b not in humanoid.mappings]
    # A weak name-based guess that the anatomy analysis confirmed at higher confidence is no longer "below threshold".
    humanoid.below_threshold = [m for m in humanoid.below_threshold if m.vrm_bone not in humanoid.mappings]


def _collect_warnings(a: Analysis, look_warnings: list[ConversionWarning]) -> list[ConversionWarning]:
    out: list[ConversionWarning] = list(a.selection.warnings) + list(a.humanoid.warnings)
    if a.excluded_meshes:
        out.append(ConversionWarning(
            "meshes-excluded", f"{len(a.excluded_meshes)} mesh(es) deformed by other armatures are excluded: "
            f"{', '.join(a.excluded_meshes[:8])}."))
    height = a.armature.height
    if not HEIGHT_RANGE_M[0] <= height <= HEIGHT_RANGE_M[1]:
        out.append(ConversionWarning(
            "scale-suspicious", f"Skeleton height is {height:.2f} m (expected {HEIGHT_RANGE_M[0]}-{HEIGHT_RANGE_M[1]} m). "
            "The source may use centimetres / a wrong import scale; the converter does not rescale the rig."))

    ex = a.expressions
    if ex.shape_key_count == 0:
        code, message = ("fbx-no-shape-keys", FBX_NO_SHAPE_KEYS) if a.inspection.source_format == "fbx" \
            else ("no-shape-keys", NO_SHAPE_KEYS)
        out.append(ConversionWarning(code, message))
    else:
        missing = [n for n in (*BLINKS, *VISEMES) if n not in ex.mappings]
        if missing:
            out.append(ConversionWarning("expressions-missing", f"No usable morph data for: {', '.join(missing)}."))
        if ex.arkit == "none" and ex.shape_key_count:
            out.append(ConversionWarning("arkit-missing", "No ARKit-style shape keys detected; only direct names are mapped."))
    out.extend(ex.warnings)
    out.extend(look_warnings)

    fingers = sum(1 for b in FINGER_BONES if b in a.humanoid.mappings)
    if fingers < len(FINGER_BONES):
        out.append(ConversionWarning("fingers-missing", f"{fingers} of {len(FINGER_BONES)} finger bones mapped."))
    if "leftEye" not in a.humanoid.mappings or "rightEye" not in a.humanoid.mappings:
        out.append(ConversionWarning("eyes-missing", "Eye bones are not mapped."))
    if a.config.spring.enabled and not a.springs.chains:
        out.append(ConversionWarning(
            "springs-none", "No identifiable hair bones: no spring bones generated."
            + (f" {len(a.springs.skipped)} candidate(s) skipped (see report)." if a.springs.skipped else "")))
    out.extend(a.materials.warnings)
    return out


def build_plan(analysis: Analysis, output_path: str | Path) -> VrmPlan:
    cfg = analysis.config.meta
    meta: dict[str, object] = {"vrm_name": cfg.name or Path(output_path).stem}
    for attr in ("version", "copyright_information", "contact_information", "third_party_licenses",
                 "avatar_permission", "commercial_usage", "credit_notation", "modification",
                 "allow_redistribution", "allow_excessively_violent_usage", "allow_excessively_sexual_usage",
                 "allow_political_or_religious_usage", "allow_antisocial_or_hate_usage", "other_license_url"):
        value = getattr(cfg, attr)
        if value is not None:
            meta[attr] = value
    if cfg.authors:
        meta["authors"] = list(cfg.authors)
    if cfg.references:
        meta["references"] = list(cfg.references)
    return VrmPlan(
        armature=analysis.selection.name, humanoid=analysis.humanoid.as_map(),
        expressions=dict(analysis.expressions.mappings), look_at=analysis.look_at,
        springs=analysis.springs, meta=meta, geometry=analysis.geometry)


@dataclass
class ConversionOutcome:
    report: dict
    analysis: Analysis
    output: OutputCheck | None


class ConverterCore:
    def __init__(self, bridge: "BlenderBridge", toolchain: Toolchain | None = None, events=None) -> None:
        self.bridge = bridge
        self.events = events or (lambda event: None)  # structured progress: see core/events.py
        self.toolchain = toolchain or load_toolchain()

    def load(self, source: str | Path) -> CharacterInspection:
        """Stage 1-3: clean scene, import, inspect."""
        fmt = source_format(source)
        if not Path(source).is_file():
            raise ConversionError("source-missing", f"{source}: no such file")
        self.bridge.reset_scene()
        self.bridge.import_character(str(source), fmt)
        return self.bridge.inspect(str(source), fmt)

    def auto_rig(self, inspection: CharacterInspection, source: str | Path) -> tuple[CharacterInspection, AutoRigResult]:
        """Skeleton + weights for a source without an armature, then a fresh inspection of the rigged scene."""
        result = self.bridge.auto_rig()
        if result.status == "FAIL":
            raise ConversionError("autorig-failed", "Automatic rigging did not produce a trustworthy skeleton and "
                                  "skin weights; nothing was exported.", [*result.warnings, *(
                                      f"{a['method']}: {a['status']}" for a in result.attempts)])
        return self.bridge.inspect(str(source), inspection.source_format), result

    def inspect(self, source: str | Path, config: ConversionConfig | None = None) -> Analysis:
        return analyze(self.load(source), config or ConversionConfig())

    def _author(self, analysis: Analysis, plan: VrmPlan, authoring: AutoAuthorPlan) -> None:
        if authoring.needs_mutation:
            # ANALYZE -> PLAN -> VALIDATE PLAN happened in analyze(); now MUTATE and VALIDATE RESULT.
            analysis.authoring_result = self.bridge.apply_authoring(authoring)
        else:
            analysis.authoring_result = AuthoringResult()
        for note in apply_authoring_result(plan, authoring, analysis.authoring_result):
            analysis.warnings.append(ConversionWarning("authoring-not-applied", note))
        for what, reason in analysis.authoring_result.failed.items():
            analysis.warnings.append(ConversionWarning("authoring-failed", f"{what}: {reason}"))

    def convert(
        self, source: str | Path, output: str | Path, config: ConversionConfig | None = None, dry_run: bool = False,
    ) -> ConversionOutcome:
        config = config or ConversionConfig()
        analysis: Analysis | None = None
        ev = self.events
        try:
            with events.step(ev, "import", "Importing the model"):
                inspection = self.load(source)
            rig: AutoRigResult | None = None
            if config.auto_rig and not inspection.armatures and not dry_run:
                with events.step(ev, "auto_rig", "Generating skeleton and skin weights") as done:
                    inspection, rig = self.auto_rig(inspection, source)
                    done["status"] = "SUCCESS" if rig.status == "PASS" else "WARNING"
                    done["detail"] = f"{rig.method}, deformation tests {rig.status}"
            else:
                events.skipped(ev, "auto_rig", "The model already has a skeleton" if inspection.armatures
                               else "Not requested")
            with events.step(ev, "map", "Analyzing the skeleton and face"):
                analysis = analyze(inspection, config)
            if rig is not None:
                analysis.warnings.append(ConversionWarning(
                    "autorig", f"Skeleton and skin weights were generated procedurally ({rig.method}, "
                    f"deformation tests {rig.status}, confidence {rig.confidence:.2f}); inspect the pose in a viewer."))
            require_humanoid(analysis.humanoid, config.min_confidence)
            plan = build_plan(analysis, output)
            if dry_run:
                return ConversionOutcome(build_report(self.toolchain, analysis, "dry-run", None, source), analysis, None)

            self.bridge.isolate_armature(plan.armature)
            authoring = analysis.authoring
            if authoring is None:
                events.skipped(ev, "authoring", "Not requested")
            else:
                with events.step(ev, "authoring", "Authoring face, hair and materials"):
                    self._author(analysis, plan, authoring)
            if plan.geometry is not None and plan.geometry.requires_mutation:
                self.bridge.apply_geometry(plan.geometry)
            with events.step(ev, "export", "Writing the VRM"):
                self.bridge.configure_vrm(plan)
                self.bridge.export_vrm(plan.armature, str(output), config.all_bone_influences)
            with events.step(ev, "validate", "Validating the VRM"):
                check = check_vrm_output(output, plan, inspection)
            analysis.warnings.extend(check.warnings)
            report = build_report(self.toolchain, analysis, "success", check, source)
            if rig is not None:
                report["autoRig"] = rig.to_dict()
            return ConversionOutcome(report, analysis, check)
        except ConversionError as exc:
            exc.report = build_report(self.toolchain, analysis, "failed", None, source, error=exc)  # type: ignore[attr-defined]
            raise
