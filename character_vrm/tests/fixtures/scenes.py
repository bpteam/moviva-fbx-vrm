"""Synthetic ``CharacterInspection`` scenes (what ``FakeArmature/FakeMesh/FakeShapeKey/FakeMaterial`` describe)."""

from __future__ import annotations

from character_vrm.core.model import (
    ArmatureInfo, CharacterInspection, ImageInfo, MaterialInfo, MeshInfo, ShapeKeyInfo,
)
from character_vrm.profiles.arkit import ARKIT_52
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer

FaceKeys = list[str]


def hair_bones(prefix: str = "mixamorig:Head", count: int = 3, tag: str = "HairBack01"):
    """A three-bone hair chain hanging off the head."""
    names = [f"{tag[:-2]}{i + 1:02d}" for i in range(count)]
    extra = []
    parent = f"{prefix}"
    for i, name in enumerate(names):
        extra.append((name, parent, (0.0, 0.1, 1.75 - 0.05 * i)))
        parent = name
    return extra, names


def make_scene(
    *,
    fmt: str = "glb",
    armature: ArmatureInfo | None = None,
    face_keys: FaceKeys | None = None,
    hair: bool = False,
    materials: list[MaterialInfo] | None = None,
    images: list[ImageInfo] | None = None,
    extra_armature: ArmatureInfo | None = None,
    triangles: int = 480_000,
    extra_triangles: int = 2000,
) -> CharacterInspection:
    hair_names: list[str] = []
    if armature is None:
        extra: list = []
        if hair:
            extra, hair_names = hair_bones()
        armature = build_armature("Armature", mixamo_namer(), extra=extra)
    weighted = [b.name for b in armature.bones if b.name != armature.bones[0].name]
    meshes = [
        MeshInfo("Body", 120_000, triangles, armature.name, weighted, ["Skin"]),
        MeshInfo("Face", 20_000, 40_000, armature.name, weighted[:10], ["Skin"]),
    ]
    scene = CharacterInspection(
        source_path=f"/tmp/model.{fmt}", source_format=fmt, armatures=[armature], meshes=meshes,
        materials=materials if materials is not None else [
            MaterialInfo("Skin", True, ["BSDF_PRINCIPLED"], False, [], ["baseColor", "normal", "roughness"], ["skin.png"])],
        images=images if images is not None else [ImageInfo("skin.png", 4096, 4096, True)],
    )
    for key in face_keys or []:
        scene.shape_keys.append(ShapeKeyInfo("Face", key))
    if extra_armature is not None:
        scene.armatures.append(extra_armature)
        scene.meshes.append(MeshInfo("Prop", 1000, extra_triangles, extra_armature.name, [b.name for b in extra_armature.bones], []))
    return scene


ARKIT_KEYS: FaceKeys = list(ARKIT_52)
VROID_KEYS: FaceKeys = ["Fcl_ALL_Neutral", "Fcl_ALL_Angry", "Fcl_ALL_Fun", "Fcl_ALL_Joy", "Fcl_ALL_Sorrow",
                        "Fcl_ALL_Surprised", "Fcl_EYE_Close", "Fcl_EYE_Close_L", "Fcl_EYE_Close_R",
                        "Fcl_MTH_A", "Fcl_MTH_I", "Fcl_MTH_U", "Fcl_MTH_E", "Fcl_MTH_O"]
