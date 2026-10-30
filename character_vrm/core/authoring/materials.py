"""Real-time material look: what an imported material needs so a glTF PBR renderer (three.js / three-vrm) shows it
as the artist meant, planned from the material semantics (``anatomy/materials.py``) and the texture set beside the
source file. Pure: the Blender side is ``blender/material_author.py``.

FBX import (Character Creator, ActorCore, most game exports) leaves the real-time look behind:

- the specular of eyes, cornea, teeth, tongue and hair arrives as zero (``Specular Tint`` black or ``Specular IOR
  Level`` 0), so the eyes get no highlight and look dead;
- the roughness and ambient-occlusion maps the export ships next to the file are not connected (one constant
  roughness for the whole skin: the "plastic doll" look);
- hair cards come as alpha *blend*. A real-time renderer draws blended surfaces without depth writes and sorts them
  per object, so layered cards of one mesh show through each other ("transparent hair"). Alpha *clip* (glTF
  ``MASK``) keeps them solid; the scalp cap stays blended so the hairline fades into the skin.

Only the look is planned; the material's textures, UVs and node layout otherwise stay as imported.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ..anatomy.materials import MaterialSemantic
from ..model import CharacterInspection
from .plans import PlanIssue, Skipped

# Principled BSDF values per role, applied only where the imported value is measurably wrong for the surface.
# ``specular`` is Blender's "Specular IOR Level" (0.5 = the glTF default F0 0.04 at IOR 1.5; skin is a little lower,
# ~0.028) with a white tint; it replaces an effective specular under ``DEAD_SPECULAR``. ``roughness`` replaces a
# constant roughness outside ``plausible`` (a cornea must be glossy, a scalp must not be a mirror) and never a map.
# Eye overlays (occlusion shadow, tear line) are authored as they must be and are left alone.
DEAD_SPECULAR = 0.05


@dataclass(frozen=True)
class MaterialLook:
    roughness: float | None = None
    plausible: tuple[float, float] = (0.0, 1.0)
    specular: float | None = None
    alpha_cutoff: float | None = None  # alpha clip (glTF MASK) instead of blend, for materials that blend
    maps: tuple[str, ...] = ("roughness", "ao")


LOOKS: dict[str, MaterialLook] = {
    "skin": MaterialLook(0.5, (0.3, 0.75), specular=0.35),
    "eye": MaterialLook(0.3, (0.05, 0.5), specular=0.5, maps=()),
    "cornea": MaterialLook(0.05, (0.0, 0.2), specular=0.5, maps=()),  # the wet outer layer: the highlight lives here
    "eyelash": MaterialLook(0.6, (0.3, 0.9), specular=0.25, maps=()),
    "teeth": MaterialLook(0.35, (0.1, 0.6), specular=0.5),
    "tongue": MaterialLook(0.45, (0.2, 0.7), specular=0.5),
    "hair": MaterialLook(0.55, (0.3, 0.9), specular=0.5, alpha_cutoff=0.35),
    "scalp": MaterialLook(0.8, (0.5, 1.0), specular=0.2, maps=("ao",)),
    "clothes": MaterialLook(),
    "other": MaterialLook(),
}
TEXTURE_SUFFIXES: dict[str, tuple[str, ...]] = {"roughness": ("roughness", "rough"), "ao": ("ao", "occlusion")}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".webp")
MAX_MAP_SIZE = 2048        # added roughness / AO maps are scaled down to this: 4K data maps doubled the file for
                           # detail a video-call framing never shows
MAX_SCANNED_FILES = 20000  # the texture walk stops here (a model dropped into a big folder)
_OVERLAY_TOKENS = ("occlusion", "tearline", "tear_line", "eyewet")
_BLENDER_DUPLICATE = re.compile(r"\.\d{3}$")


@dataclass
class MaterialEdit:
    material: str
    role: str
    confidence: float
    roughness: float | None = None
    specular: float | None = None
    alpha_cutoff: float | None = None
    textures: dict[str, str] = field(default_factory=dict)  # channel ("roughness", "ao") -> image file

    @property
    def changes(self) -> list[str]:
        out = [f"{c} map" for c in sorted(self.textures)]
        if self.roughness is not None and "roughness" not in self.textures:
            out.append(f"roughness {self.roughness}")
        if self.specular is not None:
            out.append(f"specular {self.specular}")
        if self.alpha_cutoff is not None:
            out.append(f"alpha clip {self.alpha_cutoff}")
        return out


@dataclass
class MaterialPlan:
    edits: list[MaterialEdit] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)

    def validate(self) -> list[PlanIssue]:
        issues: list[PlanIssue] = []
        for e in self.edits:
            for label, v in (("roughness", e.roughness), ("specular", e.specular), ("alpha cutoff", e.alpha_cutoff)):
                if v is not None and not 0.0 <= v <= 1.0:
                    issues.append(PlanIssue(e.material, f"{label} {v} outside 0..1"))
            for channel, path in e.textures.items():
                if not os.path.isfile(path):
                    issues.append(PlanIssue(e.material, f"{channel} map {path} does not exist"))
        return issues

    def valid_edits(self) -> list[MaterialEdit]:
        bad = {i.plan for i in self.validate()}
        return [e for e in self.edits if e.material not in bad and e.changes]


def material_role(name: str, semantic: str) -> str | None:
    """The look to apply, from the semantic class; eye overlays (the occlusion shadow and the tear line, which read
    as eye or cornea) are recognised by name and left alone (None)."""
    n = name.lower()
    if semantic in ("eye", "cornea") and any(t in n for t in _OVERLAY_TOKENS):
        return None
    if semantic == "hair" and "scalp" in n:
        return "scalp"
    return semantic if semantic in LOOKS else None


def base_name(material: str) -> str:
    """``Hair_Transparency.001`` -> ``hair_transparency`` (Blender's duplicate suffix is not part of the file names)."""
    return _BLENDER_DUPLICATE.sub("", material).lower()


def index_textures(root: str | Path, limit: int = MAX_SCANNED_FILES) -> dict[tuple[str, str], list[str]]:
    """``(material base name, channel) -> image files`` for files named ``<Material>_<suffix>.<ext>`` under ``root``
    (the Character Creator / ActorCore / Substance export layout)."""
    out: dict[tuple[str, str], list[str]] = {}
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for f in sorted(filenames):
            seen += 1
            if seen > limit:
                return out
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            low = stem.lower()
            for channel, suffixes in TEXTURE_SUFFIXES.items():
                for s in suffixes:
                    if low.endswith("_" + s):
                        out.setdefault((low[: -len(s) - 1], channel), []).append(os.path.join(dirpath, f))
    return out


def _parts(path: str) -> list[str]:
    return [unicodedata.normalize("NFC", p).lower() for p in Path(path).parts]


def pick_texture(candidates: list[str], meshes: list[str], model: str | None = None) -> str | None:
    """One file for a material. Several models beside each other (``textures/<Model>/...`` per export): the maps
    under a folder named after this model (``model``, the source file stem) win. Several folders of one model holding
    the same material name (two hair meshes both using ``Hair_Transparency``): the one whose path names a mesh of this
    material wins. Still ambiguous -> None."""
    if model:
        key = unicodedata.normalize("NFC", model).lower()
        own = [c for c in candidates if key in _parts(c)]
        candidates = own or candidates
    if len(candidates) == 1:
        return candidates[0]
    owned = [c for c in candidates if any(m.lower() in _parts(c) for m in meshes)]
    return owned[0] if len(owned) == 1 else None


def plan_materials(semantics: list[MaterialSemantic], inspection: CharacterInspection,
                   textures: dict[tuple[str, str], list[str]] | None = None) -> MaterialPlan:
    """PLAN. ``textures`` is ``index_textures`` of the source folder (None: look for none)."""
    textures = textures or {}
    model = Path(inspection.source_path).stem if inspection.source_path else None
    infos = {m.name: m for m in inspection.materials}
    plan = MaterialPlan()
    for sem in semantics:
        info = infos.get(sem.material)
        role = material_role(sem.material, sem.semantic)
        if info is None or role is None:
            continue
        look = LOOKS[role]
        edit = MaterialEdit(sem.material, role, sem.confidence)
        lo, hi = look.plausible
        if look.roughness is not None and info.roughness is not None and not lo <= info.roughness <= hi:
            edit.roughness = look.roughness
        if look.specular is not None and info.specular is not None and info.specular < DEAD_SPECULAR:
            edit.specular = look.specular
        if look.alpha_cutoff is not None and info.alpha_mode == "BLEND":
            edit.alpha_cutoff = look.alpha_cutoff
        for channel in look.maps:
            cands = textures.get((base_name(sem.material), channel), [])
            path = pick_texture(cands, sem.meshes, model) if cands else None
            if path is not None:
                edit.textures[channel] = path
            elif len(cands) > 1:
                plan.skipped.append(Skipped(f"{sem.material} {channel} map",
                                            f"{len(cands)} files match and none names its mesh"))
        if edit.changes:
            plan.edits.append(edit)
    return plan
