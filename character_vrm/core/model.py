"""Normalised, ``bpy``-independent scene description and conversion plans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

Vec3 = tuple[float, float, float]

REQUIRED_HUMAN_BONES: tuple[str, ...] = (
    "hips", "spine", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
)

_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")


def _finger_names() -> tuple[str, ...]:
    out = []
    for side in ("left", "right"):
        for finger in _FINGERS:
            parts = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else (
                "Proximal", "Intermediate", "Distal")
            out.extend(f"{side}{finger}{part}" for part in parts)
    return tuple(out)


FINGER_BONES: tuple[str, ...] = _finger_names()

OPTIONAL_HUMAN_BONES: tuple[str, ...] = (
    "chest", "upperChest", "neck",
    "leftShoulder", "rightShoulder",
    "leftToes", "rightToes",
    "leftEye", "rightEye", "jaw",
) + FINGER_BONES

ALL_HUMAN_BONES: tuple[str, ...] = REQUIRED_HUMAN_BONES + OPTIONAL_HUMAN_BONES

VRM_EXPRESSIONS: tuple[str, ...] = (
    "blink", "blinkLeft", "blinkRight",
    "aa", "ih", "ou", "ee", "oh",
    "happy", "angry", "sad", "relaxed", "surprised",
    "lookUp", "lookDown", "lookLeft", "lookRight",
)


@dataclass
class BoneInfo:
    name: str
    parent: str | None
    head: Vec3  # world space, metres as Blender reports them
    tail: Vec3
    deform: bool = True

    @property
    def length(self) -> float:
        return math.dist(self.head, self.tail)


@dataclass
class ArmatureInfo:
    name: str
    bones: list[BoneInfo]
    _by_name: dict[str, BoneInfo] = field(init=False, repr=False, compare=False, default_factory=dict)
    _children: dict[str, list[str]] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._by_name = {b.name: b for b in self.bones}
        self._children = {b.name: [] for b in self.bones}
        for bone in self.bones:
            if bone.parent in self._children:
                self._children[bone.parent].append(bone.name)

    def bone(self, name: str) -> BoneInfo | None:
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def children(self, name: str) -> list[str]:
        return list(self._children.get(name, []))

    def parent(self, name: str) -> str | None:
        bone = self._by_name.get(name)
        return bone.parent if bone and bone.parent in self._by_name else None

    def ancestors(self, name: str) -> list[str]:
        """Nearest first, excluding ``name``."""
        out: list[str] = []
        seen = {name}
        current = self.parent(name)
        while current and current not in seen:
            out.append(current)
            seen.add(current)
            current = self.parent(current)
        return out

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor in self.ancestors(descendant)

    def depth(self, name: str) -> int:
        return len(self.ancestors(name))

    def descendants(self, name: str) -> list[str]:
        out: list[str] = []
        stack = list(reversed(self._children.get(name, [])))
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(self._children.get(current, [])))
        return out

    def subtree_size(self, name: str) -> int:
        return len(self.descendants(name))

    def lowest_common_ancestor(self, names: Iterable[str]) -> str | None:
        chains = [[n, *self.ancestors(n)] for n in names if n in self._by_name]
        if not chains:
            return None
        common = set(chains[0])
        for chain in chains[1:]:
            common &= set(chain)
        # Strictly above every input, deepest first.
        for candidate in chains[0][1:]:
            if candidate in common:
                return candidate
        return None

    @property
    def height(self) -> float:
        """Vertical extent of the skeleton (Blender +Z is up)."""
        if not self.bones:
            return 0.0
        zs = [p[2] for b in self.bones for p in (b.head, b.tail)]
        return max(zs) - min(zs)


@dataclass
class ShapeKeyInfo:
    mesh: str
    name: str
    slider_min: float = 0.0
    slider_max: float = 1.0
    # Optional sparse offsets from the basis (vertex index -> world-space delta). Filled by the inspectors so the
    # anatomy analysis can read *where* an existing key moves the face; empty when unknown.
    deltas: dict[int, Vec3] = field(default_factory=dict)


@dataclass
class MeshInfo:
    name: str
    vertices: int
    triangles: int
    armature: str | None = None  # armature object that deforms this mesh
    vertex_groups: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    # Optional topology data.  The importer fills these for geometry-profile
    # planning; keeping them optional preserves the small inspection fixture
    # API used by older callers.
    vertex_positions: list[Vec3] = field(default_factory=list)
    polygons: list[tuple[int, ...]] = field(default_factory=list)
    polygon_materials: list[int] = field(default_factory=list)  # material slot per polygon (index into materials)
    vertex_weights: dict[int, dict[str, float]] = field(default_factory=dict)
    parent_bone: str | None = None
    attached_bones: list[str] = field(default_factory=list)

    @property
    def skinned(self) -> bool:
        return self.armature is not None and bool(self.vertex_groups)


@dataclass
class MaterialInfo:
    name: str
    uses_nodes: bool = True
    shader_types: list[str] = field(default_factory=list)
    procedural: bool = False
    procedural_nodes: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)  # baseColor, normal, ...
    images: list[str] = field(default_factory=list)
    alpha_mode: str = "OPAQUE"
    roughness: float | None = None  # the constant roughness when no texture drives it (None: textured or unknown)
    specular: float | None = None   # effective specular: IOR level x brightest tint channel (None: unknown)


@dataclass
class ImageInfo:
    name: str
    width: int
    height: int
    packed: bool = False
    filepath: str = ""


@dataclass
class CharacterInspection:
    source_path: str
    source_format: str  # "glb" | "gltf" | "fbx"
    armatures: list[ArmatureInfo] = field(default_factory=list)
    meshes: list[MeshInfo] = field(default_factory=list)
    materials: list[MaterialInfo] = field(default_factory=list)
    shape_keys: list[ShapeKeyInfo] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    face_points: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # MediaPipe, world space

    @property
    def triangle_count(self) -> int:
        return sum(m.triangles for m in self.meshes)

    @property
    def skinned_meshes(self) -> list[MeshInfo]:
        return [m for m in self.meshes if m.skinned]

    def armature(self, name: str) -> ArmatureInfo | None:
        return next((a for a in self.armatures if a.name == name), None)

    def meshes_for(self, armature: str) -> list[MeshInfo]:
        return [m for m in self.meshes if m.armature == armature]

    def weighted_bone_names(self, armature: str) -> set[str]:
        return {g for m in self.meshes_for(armature) if m.skinned for g in m.vertex_groups}

    def shape_keys_by_name(self) -> dict[str, list[ShapeKeyInfo]]:
        out: dict[str, list[ShapeKeyInfo]] = {}
        for key in self.shape_keys:
            out.setdefault(key.name, []).append(key)
        return out


# ---------------------------------------------------------------- plans


@dataclass
class BoneMapping:
    vrm_bone: str
    source_bone: str
    confidence: float
    reason: str


@dataclass
class MorphBind:
    mesh: str
    shape_key: str
    weight: float = 1.0


@dataclass
class ExpressionMapping:
    name: str
    kind: str  # "direct" | "compound" | "config"
    binds: list[MorphBind]
    confidence: float
    reason: str


@dataclass
class LookAtPlan:
    type: str | None  # "bone" | "expression" | None
    reason: str
    offset_from_head_bone: Vec3 | None = None  # only ever user-provided, see lookat.py


class ProtectedReason(StrEnum):
    ARM_WEIGHT = "arm-weight"
    ARM_PARENT = "arm-parent"
    SHOULDER_TRANSITION = "shoulder-transition"
    HAIR = "hair"
    HEAD = "head"
    EXPLICIT_KEEP = "explicit-keep"


@dataclass
class GeometryOptimizationPlan:
    """A conservative, topology-level geometry decision made without Blender.

    ``protected_vertices`` and ``remove_polygons`` are keyed by mesh name.
    A polygon is removed only when every one of its vertices is outside the
    protected mask and below the body cut.  The armature is deliberately not
    part of this plan: geometry optimization never removes or rewrites bones.
    """

    profile: str
    cut_plane: float | None
    protected_objects: set[str] = field(default_factory=set)
    protected_vertices: dict[str, set[int]] = field(default_factory=dict)
    protected_reasons: dict[str, dict[int, set[ProtectedReason]]] = field(default_factory=dict)
    delete_objects: list[str] = field(default_factory=list)
    crop_objects: list[str] = field(default_factory=list)
    preserve_objects: list[str] = field(default_factory=list)
    remove_polygons: dict[str, set[int]] = field(default_factory=dict)
    keep_polygons: dict[str, set[int]] = field(default_factory=dict)
    material_cleanup_candidates: list[str] = field(default_factory=list)
    image_cleanup_candidates: list[str] = field(default_factory=list)
    semantic_frame: str = "full-body"
    cut_landmark: str | None = None
    cut_source: str | None = None
    landmark: float | None = None
    safety_margin: float = 0.0

    @property
    def requires_mutation(self) -> bool:
        # Cropped profiles still need to pass through the Blender mutation
        # stage when an imported inspection has no topology detail (for
        # example an older bridge fixture): the adapter can conservatively
        # no-op, but the profile must not silently bypass that stage.
        return self.profile != "full" or bool(self.delete_objects or any(self.remove_polygons.values()))


@dataclass(frozen=True)
class SpringPreset:
    stiffness: float
    drag_force: float
    gravity_power: float
    hit_radius: float  # metres for a 1.7 m character; scaled by skeleton height


@dataclass
class SpringChain:
    name: str
    bones: list[str]
    confidence: float
    reason: str
    preset: SpringPreset


@dataclass
class ColliderPlan:
    name: str
    bone: str
    radius: float
    offset: Vec3  # bone-local, Y runs along the bone (VRM add-on convention)


@dataclass
class SpringPlan:
    chains: list[SpringChain] = field(default_factory=list)
    colliders: list[ColliderPlan] = field(default_factory=list)
    skipped: list[dict[str, object]] = field(default_factory=list)


@dataclass
class ConversionWarning:
    code: str
    message: str


@dataclass
class VrmPlan:
    """Everything the VRM Add-on adapter has to apply, decided by the pure core."""

    armature: str
    humanoid: dict[str, str]  # vrm bone -> source bone
    expressions: dict[str, ExpressionMapping]
    look_at: LookAtPlan
    springs: SpringPlan
    meta: dict[str, object]  # add-on meta property name -> value (only what the user stated + name)
    geometry: GeometryOptimizationPlan | None = None
