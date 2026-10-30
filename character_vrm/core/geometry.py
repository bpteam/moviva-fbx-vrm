"""Conservative geometry-profile planning.

This module is intentionally Blender-free.  It turns a scene inspection and a
mapped humanoid into a mutation plan; the Blender adapter is the only place
that applies that plan to mesh data.  In particular, a rest-pose position is
never enough to remove arm geometry.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .anatomy.landmarks import shoulder_level_point, upper_chest_landmark, waist_landmark
from .anatomy.model import analyze_human
from .config import (
    ARM_PROTECTION_WEIGHT, GEOMETRY_PROFILES, VIDEO_CALL_WAIST_FALLBACK_T, GeometryConfig, GeometryProfileSpec,
)
from .errors import ConversionError
from .model import (
    ArmatureInfo, CharacterInspection, GeometryOptimizationPlan, MeshInfo, ProtectedReason,
)

if TYPE_CHECKING:
    from .anatomy.model import HumanModelAnalysis

_HEAD_TOKENS = ("head", "face", "eye", "eyelash", "brow", "teeth", "tongue", "mouth", "ear")
_HAIR_TOKENS = ("hair", "bang", "fringe", "ponytail", "braid", "strand")


@dataclass
class ArmProtectionMask:
    """Skinning-first mask shared by the planner and diagnostics."""

    protected_bones: set[str]
    protected_vertices: dict[str, set[int]] = field(default_factory=dict)
    reasons: dict[str, dict[int, set[ProtectedReason]]] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return sum(len(vertices) for vertices in self.protected_vertices.values())


def _tokens(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else " " for c in name)


def _contains(name: str, tokens: tuple[str, ...]) -> bool:
    value = _tokens(name)
    return any(token in value for token in tokens)


def _mapped_source(humanoid: object, vrm_name: str) -> str | None:
    mappings = getattr(humanoid, "mappings", {})
    mapping = mappings.get(vrm_name)
    return getattr(mapping, "source_bone", None)


def protected_arm_bones(armature: ArmatureInfo, humanoid: object) -> set[str]:
    """Return arm chains plus their mapped/unmapped helper descendants."""
    roots: list[str] = []
    for side in ("left", "right"):
        for part in ("Shoulder", "UpperArm", "LowerArm", "Hand"):
            source = _mapped_source(humanoid, f"{side}{part}")
            if source:
                roots.append(source)
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            for segment in ("Metacarpal", "Proximal", "Intermediate", "Distal"):
                source = _mapped_source(humanoid, f"{side}{finger}{segment}")
                if source:
                    roots.append(source)
    protected: set[str] = set()
    for root in roots:
        if not armature.has(root):
            continue
        protected.add(root)
        protected.update(armature.descendants(root))
    return protected


def _arm_weight(mesh: MeshInfo, vertex: int, arm_bones: set[str]) -> float:
    return sum(weight for bone, weight in mesh.vertex_weights.get(vertex, {}).items() if bone in arm_bones)


def _is_attached_to_arm(mesh: MeshInfo, arm_bones: set[str]) -> bool:
    return bool(set(mesh.attached_bones) & arm_bones) or mesh.parent_bone in arm_bones


def _has_hair_evidence(mesh: MeshInfo) -> bool:
    return any(_contains(group, _HAIR_TOKENS) for group in (*mesh.vertex_groups, *mesh.attached_bones))


def build_arm_protection_mask(
    inspection: CharacterInspection, armature: ArmatureInfo, humanoid: object,
    threshold: float = ARM_PROTECTION_WEIGHT,
) -> ArmProtectionMask:
    """Build the mandatory arm mask from vertex influences, not object names."""
    bones = protected_arm_bones(armature, humanoid)
    mask = ArmProtectionMask(bones)
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature.name):
            continue
        protected: set[int] = set()
        reasons: dict[int, set[ProtectedReason]] = {}
        if _is_attached_to_arm(mesh, bones):
            protected.update(range(len(mesh.vertex_positions)))
            for vertex in protected:
                reasons[vertex] = {ProtectedReason.ARM_PARENT}
        for vertex in range(len(mesh.vertex_positions)):
            if _arm_weight(mesh, vertex, bones) >= threshold:
                protected.add(vertex)
                reasons.setdefault(vertex, set()).add(ProtectedReason.ARM_WEIGHT)
        if protected:
            mask.protected_vertices[mesh.name] = protected
            mask.reasons[mesh.name] = reasons
    return mask


def _shoulder_level(anatomy: "HumanModelAnalysis") -> float | None:
    point = shoulder_level_point(anatomy.joints)
    return anatomy.frame.height_of(point) if point else None


def _waist_level(anatomy: "HumanModelAnalysis", spec: GeometryProfileSpec) -> tuple[float | None, str]:
    """The waist is a landmark of the canonical anatomy; only the profile's fallback ratio is geometry policy."""
    t = spec.fallback_t if spec.fallback_t is not None else VIDEO_CALL_WAIST_FALLBACK_T
    waist = waist_landmark(anatomy.frame, anatomy.joints, t)
    return (anatomy.frame.height_of(waist.position), waist.method) if waist else (None, "unavailable")


def _upper_chest_level(anatomy: "HumanModelAnalysis") -> tuple[float | None, str]:
    lm = upper_chest_landmark(anatomy.joints)
    return (anatomy.frame.height_of(lm.position), lm.method) if lm else (None, "unavailable")


def _polygon_is_below(mesh: MeshInfo, polygon: tuple[int, ...], cut: float, level: Callable) -> bool:
    if not mesh.vertex_positions or not polygon:
        return False  # uncertain geometry is kept
    points = [mesh.vertex_positions[i] for i in polygon if 0 <= i < len(mesh.vertex_positions)]
    return bool(points) and all(level(point) < cut for point in points)


def plan_geometry(
    inspection: CharacterInspection,
    armature: ArmatureInfo,
    humanoid: object,
    config: GeometryConfig,
    anatomy: "HumanModelAnalysis | None" = None,
) -> GeometryOptimizationPlan:
    """Build a conservative crop plan for ``full``, ``video-call`` or ``face``.

    Every anatomical level (waist, shoulders, upper chest) comes from the canonical ``HumanModelAnalysis``; when the
    caller has none it is computed here, once. No world axis is assumed: heights are measured along the anatomical up.
    """
    if config.profile not in GEOMETRY_PROFILES:
        raise ConversionError("config-invalid", f"Unknown geometry profile '{config.profile}'.")
    conflicting = sorted(set(config.keep_objects) & set(config.drop_objects))
    if conflicting:
        raise ConversionError("config-invalid", f'Object "{conflicting[0]}" is listed in both keepObjects and dropObjects.')
    spec = GEOMETRY_PROFILES[config.profile]
    plan = GeometryOptimizationPlan(
        profile=config.profile, cut_plane=None, semantic_frame=spec.semantic_frame,
        cut_landmark=spec.cut_landmark, safety_margin=spec.body_safety_margin)

    chosen_meshes = [mesh for mesh in inspection.meshes if mesh.armature in (None, armature.name)]
    arm_mask = build_arm_protection_mask(inspection, armature, humanoid)
    arm_bones = arm_mask.protected_bones
    head_source = _mapped_source(humanoid, "head")
    head_bones = ({head_source} | set(armature.descendants(head_source))) if head_source else set()
    if not spec.enabled:
        unsafe = sorted(
            set(config.drop_objects)
            & {
                mesh.name for mesh in chosen_meshes
                if _is_attached_to_arm(mesh, arm_bones)
                or _contains(mesh.name, _HEAD_TOKENS)
                or _contains(mesh.name, _HAIR_TOKENS)
                or _has_hair_evidence(mesh)
                or bool(set(mesh.vertex_groups) & head_bones)
                or bool(set(mesh.vertex_groups) & arm_bones)
            }
        )
        if unsafe:
            raise ConversionError(
                "unsafe-geometry-drop", f'Cannot drop object "{unsafe[0]}".',
                [f'object contains protected arm/head/hair geometry required by geometry profile "{config.profile}"'])
        plan.delete_objects = list(config.drop_objects)
        plan.preserve_objects = [m.name for m in chosen_meshes if m.name not in plan.delete_objects]
        return plan

    if anatomy is None:
        anatomy = analyze_human(inspection, armature, humanoid)
    if anatomy is None:
        landmark, source = None, "unavailable"
    elif config.profile == "video-call":
        landmark, source = _waist_level(anatomy, spec)
    else:
        landmark, source = _upper_chest_level(anatomy)
    level = anatomy.frame.height_of if anatomy is not None else (lambda p: p[2])
    plan.landmark, plan.cut_source = landmark, source
    if landmark is not None:
        plan.cut_plane = landmark - spec.body_safety_margin

    for mesh in chosen_meshes:
        reasons: dict[int, set[ProtectedReason]] = {}
        protected: set[int] = set()
        whole_reason: ProtectedReason | None = None
        if config.keep_objects and mesh.name in config.keep_objects:
            whole_reason = ProtectedReason.EXPLICIT_KEEP
        elif spec.preserve_head and _contains(mesh.name, _HEAD_TOKENS):
            whole_reason = ProtectedReason.HEAD
        elif spec.preserve_full_hair and _contains(mesh.name, _HAIR_TOKENS):
            whole_reason = ProtectedReason.HAIR
        elif spec.preserve_full_hair and _has_hair_evidence(mesh):
            whole_reason = ProtectedReason.HAIR
        elif spec.preserve_full_arms and _is_attached_to_arm(mesh, arm_bones):
            whole_reason = ProtectedReason.ARM_PARENT

        if whole_reason is not None:
            plan.protected_objects.add(mesh.name)
            plan.preserve_objects.append(mesh.name)
            if mesh.vertex_positions:
                protected.update(range(len(mesh.vertex_positions)))
                for vertex in protected:
                    reasons[vertex] = {whole_reason}
        else:
            if spec.preserve_full_arms:
                protected.update(arm_mask.protected_vertices.get(mesh.name, set()))
                reasons.update(arm_mask.reasons.get(mesh.name, {}))
            for vertex in range(len(mesh.vertex_positions)):
                head_weight = sum(weight for bone, weight in mesh.vertex_weights.get(vertex, {}).items()
                                  if bone in head_bones)
                if spec.preserve_head and head_weight >= ARM_PROTECTION_WEIGHT:
                    protected.add(vertex)
                    reasons.setdefault(vertex, set()).add(ProtectedReason.HEAD)

            # Keep a conservative shoulder/armpit transition.  This catches
            # mixed torso/arm polygons even when their arm weights are diluted.
            shoulder = _shoulder_level(anatomy) if anatomy is not None else None
            if mesh.vertex_positions and shoulder is not None:
                height = max(armature.height, 1.0)
                for vertex, position in enumerate(mesh.vertex_positions):
                    if abs(level(position) - shoulder) <= 0.10 * height and \
                            abs(anatomy.frame.lateral_of(position)) >= 0.03 * height:
                        protected.add(vertex)
                        reasons.setdefault(vertex, set()).add(ProtectedReason.SHOULDER_TRANSITION)

        if protected:
            plan.protected_vertices[mesh.name] = protected
            plan.protected_reasons[mesh.name] = reasons

        if plan.cut_plane is None or mesh.name in plan.protected_objects:
            if mesh.name not in plan.preserve_objects:
                plan.preserve_objects.append(mesh.name)
            continue

        removed: set[int] = set()
        kept: set[int] = set()
        for index, polygon in enumerate(mesh.polygons):
            if any(vertex in protected for vertex in polygon):
                kept.add(index)
            elif _polygon_is_below(mesh, polygon, plan.cut_plane, level):
                removed.add(index)
            else:
                kept.add(index)
        if removed:
            plan.remove_polygons[mesh.name] = removed
            plan.keep_polygons[mesh.name] = kept
            plan.crop_objects.append(mesh.name)
        else:
            plan.preserve_objects.append(mesh.name)

        # If an ordinary object is entirely below the body cut, removing it
        # is safe and avoids retaining an empty lower-body mesh.  Objects with
        # unknown topology stay in the conservative crop list.
        if mesh.vertex_positions and plan.cut_plane is not None and all(
            level(point) < plan.cut_plane for point in mesh.vertex_positions
        ) and not protected:
            plan.delete_objects.append(mesh.name)
            if mesh.name in plan.crop_objects:
                plan.crop_objects.remove(mesh.name)

    protected_names = set(plan.protected_objects) | set(config.keep_objects) | set(plan.protected_vertices)
    protected_names |= {
        mesh.name for mesh in chosen_meshes
        if bool(set(mesh.vertex_groups) & arm_bones)
        or bool(set(mesh.vertex_groups) & head_bones)
        or _has_hair_evidence(mesh)
    }
    unsafe = sorted(set(config.drop_objects) & protected_names)
    if unsafe:
        name = unsafe[0]
        raise ConversionError(
            "unsafe-geometry-drop",
            f'Cannot drop object "{name}".',
            [f'object contains protected geometry required by geometry profile "{config.profile}"'])
    plan.delete_objects.extend(name for name in config.drop_objects if name not in plan.delete_objects)
    # These are intentionally only candidates.  The Blender adapter performs
    # topology mutation first; shared material/atlas data is never cropped.
    plan.material_cleanup_candidates = list({m for mesh in chosen_meshes for m in mesh.materials})
    plan.image_cleanup_candidates = list({image.name for image in inspection.images})
    return plan
