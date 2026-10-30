"""Fail-hard checks: humanoid completeness before export, and the written VRM afterwards.

The output check *reads* the file the add-on wrote (GLB container + JSON chunk); it never writes one.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConversionError
from .humanoid import HumanoidResult
from .model import REQUIRED_HUMAN_BONES, CharacterInspection, ConversionWarning, VrmPlan

_GROUPS = (
    ("hips", ("hips",)), ("spine", ("spine",)), ("head", ("head",)),
    ("arms", ("leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand")),
    ("legs", ("leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot")),
)


def require_humanoid(result: HumanoidResult, min_confidence: float) -> None:
    problems: list[str] = []
    for label, bones in _GROUPS:
        missing = [b for b in bones if b in result.missing_required]
        if missing:
            problems.append(f"missing {label}: {missing}")
    low = {m.vrm_bone: m for m in result.below_threshold if m.vrm_bone in REQUIRED_HUMAN_BONES}
    for vrm, m in low.items():
        problems.append(f"{vrm} -> {m.source_bone} only reached confidence {m.confidence:.2f} "
                        f"({m.reason}), below {min_confidence:.2f}")
    problems.extend(result.errors)
    if problems:
        raise ConversionError(
            "humanoid-incomplete" if not result.errors else "humanoid-invalid",
            "The humanoid could not be mapped safely. Run `character-vrm map <model>` and review the "
            "suggested .vrm-map.json (or lower --min-confidence if you accept the guesses).", problems)


# ------------------------------------------------------------------- output


@dataclass
class OutputCheck:
    path: str
    size_bytes: int
    nodes: int = 0
    meshes: int = 0
    skins: int = 0
    materials: int = 0
    images: int = 0
    humanoid_bones: int = 0
    expressions: list[str] = field(default_factory=list)
    springs: int = 0
    look_at_type: str | None = None
    spec_version: str | None = None
    warnings: list[ConversionWarning] = field(default_factory=list)


def read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ConversionError("invalid-output", f"{path}: too small to be a GLB ({len(data)} bytes)")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ConversionError("invalid-output", f"{path}: not a GLB container (magic {magic!r})")
    if version != 2:
        raise ConversionError("invalid-output", f"{path}: unsupported GLB version {version}")
    if length != len(data):
        raise ConversionError("invalid-output", f"{path}: header length {length} != file size {len(data)}")
    chunk_len, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON":
        raise ConversionError("invalid-output", f"{path}: first chunk is not JSON")
    try:
        return json.loads(data[20:20 + chunk_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError("invalid-output", f"{path}: unreadable JSON chunk: {exc}") from exc


def check_vrm_output(path: str | Path, plan: VrmPlan, source: CharacterInspection) -> OutputCheck:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ConversionError("invalid-output", f"{path}: was not written or is empty")
    gltf = read_glb_json(path)
    check = OutputCheck(str(path), path.stat().st_size)
    errors: list[str] = []

    nodes = gltf.get("nodes", [])
    check.nodes, check.meshes = len(nodes), len(gltf.get("meshes", []))
    check.skins, check.materials, check.images = (len(gltf.get(k, [])) for k in ("skins", "materials", "images"))

    ext = (gltf.get("extensions") or {}).get("VRMC_vrm")
    if "VRMC_vrm" not in gltf.get("extensionsUsed", []) or not ext:
        errors.append("VRMC_vrm extension is missing")
    else:
        check.spec_version = ext.get("specVersion")
        if check.spec_version != "1.0":
            errors.append(f"VRMC_vrm.specVersion is {check.spec_version!r}, expected '1.0'")
        bones = (ext.get("humanoid") or {}).get("humanBones") or {}
        check.humanoid_bones = len(bones)
        for required in REQUIRED_HUMAN_BONES:
            node = (bones.get(required) or {}).get("node")
            if not isinstance(node, int) or not 0 <= node < len(nodes):
                errors.append(f"humanoid bone {required} is missing or references an invalid node ({node!r})")
        check.expressions = sorted((ext.get("expressions") or {}).get("preset", {}))
        check.look_at_type = (ext.get("lookAt") or {}).get("type")
        _check_expressions(ext, gltf, errors)

    if check.meshes == 0:
        errors.append("no meshes were exported")
    if source.skinned_meshes and check.skins == 0:
        errors.append("the source had skinned meshes but the output has no skins: skinning was lost")
    if source.materials and check.materials == 0:
        errors.append("the source had materials but the output has none")
    if source.images and check.images == 0:
        check.warnings.append(ConversionWarning("output-no-textures", "The source had textures but the output has no images."))
    for i, image in enumerate(gltf.get("images", [])):
        if "bufferView" not in image and "uri" not in image:
            errors.append(f"image {i} has neither bufferView nor uri")

    spring = (gltf.get("extensions") or {}).get("VRMC_springBone")
    if spring:
        check.springs = len(spring.get("springs", []))
        for s, sp in enumerate(spring.get("springs", [])):
            for j in sp.get("joints", []):
                if not 0 <= j.get("node", -1) < len(nodes):
                    errors.append(f"spring {s} references invalid node {j.get('node')}")
    if plan.springs.chains and not check.springs:
        check.warnings.append(ConversionWarning(
            "output-springs-missing", "Spring chains were configured but the exported file contains no springs."))

    if errors:
        raise ConversionError("invalid-output", f"{path} is not a valid VRM 1.0 output.", errors)
    return check


def _check_expressions(ext: dict[str, Any], gltf: dict[str, Any], errors: list[str]) -> None:
    nodes, meshes = gltf.get("nodes", []), gltf.get("meshes", [])
    presets = (ext.get("expressions") or {}).get("preset", {})
    for name, expr in presets.items():
        for bind in expr.get("morphTargetBinds", []):
            node = bind.get("node")
            if not isinstance(node, int) or not 0 <= node < len(nodes):
                errors.append(f"expression {name}: invalid node {node!r}")
                continue
            mesh_index = nodes[node].get("mesh")
            if mesh_index is None or not 0 <= mesh_index < len(meshes):
                errors.append(f"expression {name}: node {node} has no mesh")
                continue
            targets = len(meshes[mesh_index]["primitives"][0].get("targets", [])) if meshes[mesh_index].get("primitives") else 0
            if not 0 <= bind.get("index", -1) < targets:
                errors.append(f"expression {name}: morph target index {bind.get('index')} out of range ({targets})")
