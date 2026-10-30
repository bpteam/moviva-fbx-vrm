"""Material audit. Materials are preserved as glTF PBR; MToon conversion is never applied."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import WARN_MATERIALS, WARN_TEXTURE_SIZE, WARN_TRIANGLES
from .model import CharacterInspection, ConversionWarning

_PREVIEW = 8


@dataclass
class MaterialsResult:
    total: int = 0
    pbr_preserved: list[str] = field(default_factory=list)
    procedural: list[str] = field(default_factory=list)
    non_principled: list[str] = field(default_factory=list)
    channels: dict[str, int] = field(default_factory=dict)
    textures: int = 0
    max_texture: tuple[int, int] = (0, 0)
    warnings: list[ConversionWarning] = field(default_factory=list)


def _names(items: list[str]) -> str:
    shown = ", ".join(items[:_PREVIEW])
    return shown + (f" and {len(items) - _PREVIEW} more" if len(items) > _PREVIEW else "")


def analyze_materials(inspection: CharacterInspection) -> MaterialsResult:
    result = MaterialsResult(total=len(inspection.materials), textures=len(inspection.images))
    for mat in inspection.materials:
        if mat.procedural:
            result.procedural.append(mat.name)
        elif mat.uses_nodes and "BSDF_PRINCIPLED" not in mat.shader_types:
            result.non_principled.append(mat.name)
        else:
            result.pbr_preserved.append(mat.name)
        for channel in mat.channels:
            result.channels[channel] = result.channels.get(channel, 0) + 1
    if inspection.images:
        result.max_texture = max(((i.width, i.height) for i in inspection.images), key=lambda s: s[0] * s[1])

    if result.procedural:
        result.warnings.append(ConversionWarning(
            "materials-procedural",
            f"{len(result.procedural)} material(s) use procedural nodes that glTF cannot represent and will be "
            f"exported without them: {_names(result.procedural)}."))
    if result.non_principled:
        result.warnings.append(ConversionWarning(
            "materials-non-principled",
            f"{len(result.non_principled)} material(s) do not use a Principled BSDF; only their basic colour "
            f"survives export: {_names(result.non_principled)}."))
    if result.total > WARN_MATERIALS:
        result.warnings.append(ConversionWarning(
            "materials-many", f"{result.total} materials (more than {WARN_MATERIALS}); this costs draw calls at runtime."))
    if max(result.max_texture) > WARN_TEXTURE_SIZE:
        result.warnings.append(ConversionWarning(
            "textures-large", f"Largest texture is {result.max_texture[0]}x{result.max_texture[1]}."))
    if inspection.triangle_count > WARN_TRIANGLES:
        result.warnings.append(ConversionWarning(
            "geometry-heavy", f"{inspection.triangle_count:,} triangles. No limit is enforced; runtime cost is on you."))
    return result
