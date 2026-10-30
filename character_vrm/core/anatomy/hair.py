"""Phase 13: hair analysis. Multiple weak signals, one classification; chains found even with poor bone names."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import ArmatureInfo, CharacterInspection, MaterialInfo, MeshInfo
from ..names import parse_bone_name
from .confidence import DEFAULT_THRESHOLDS, AnalysisSource, AnalysisWarning, Thresholds, fuse
from .frame import AnatomicalFrame
from .vec import Vec3, dot, mean, principal_axis, sub

HAIR_CONFIDENT, HAIR_POSSIBLE, NOT_HAIR = "HAIR_CONFIDENT", "HAIR_POSSIBLE", "NOT_HAIR"
_NAME_TOKENS = ("hair", "bang", "fringe", "ponytail", "braid", "plait", "strand", "pigtail", "twintail", "scalp")
_NOT_HAIR_TOKENS = ("face", "head", "body", "skin", "eye", "teeth", "tongue", "torso", "cloth", "shirt", "dress", "pants")
_FACE_BONE_TOKENS = frozenset({"jaw", "eye", "brow", "lid", "tongue", "teeth", "lip", "cheek", "nose", "ear", "mouth"})
_RELIABILITY = {"object_name": 0.85, "material_name": 0.7, "texture_name": 0.55, "head_proximity": 0.45,
                "outer_shell": 0.7,
                "scalp": 0.35, "hair_bones": 0.9, "alpha": 0.2}

# Hair types.
SOLID_MASS, CLUMP, PONYTAIL, BRAID, ALPHA_CARDS, UNKNOWN = "solid-mass", "clump", "ponytail", "braid", "alpha-cards", "unknown"
# Spring categories.
SHORT_HAIR, LONG_HAIR, SOFT_STRAND = "short-hair", "long-hair", "soft-strand"


def _has(name: str, tokens: tuple[str, ...]) -> bool:
    n = "".join(c.lower() for c in name if c.isalnum())
    return any(t in n for t in tokens)


@dataclass
class HairMeshAssessment:
    """One surface: a whole mesh (``slot`` None) or one material slot of a multi-material mesh."""

    mesh: str
    classification: str
    confidence: float
    evidence: dict[str, float]
    hair_type: str = UNKNOWN
    components: int = 0
    slot: int | None = None
    material: str | None = None


@dataclass
class HairChain:
    root: str
    bones: list[str]
    confidence: float
    evidence: dict[str, float]
    category: str = SOFT_STRAND
    length_ratio: float = 0.0   # chain length / head height
    at_back: bool = False
    source: AnalysisSource = AnalysisSource.SKELETON
    named: bool = False


@dataclass
class HairAnalysis:
    meshes: list[HairMeshAssessment] = field(default_factory=list)
    chains: list[HairChain] = field(default_factory=list)
    classification: str | None = None
    warnings: list[AnalysisWarning] = field(default_factory=list)

    @property
    def confident_meshes(self) -> list[HairMeshAssessment]:
        return [m for m in self.meshes if m.classification == HAIR_CONFIDENT]

    @property
    def has_authored_bones(self) -> bool:
        return bool(self.chains)

    def hair_slots(self, confident_only: bool = False) -> dict[str, set[int | None]]:
        """mesh -> hair surfaces (slot indices; ``None`` = the whole mesh)."""
        out: dict[str, set[int | None]] = {}
        for a in self.meshes:
            if a.classification == HAIR_CONFIDENT or (not confident_only and a.classification == HAIR_POSSIBLE):
                out.setdefault(a.mesh, set()).add(a.slot)
        return out

    def hair_vertex_sets(self, inspection: CharacterInspection, confident_only: bool = False) -> dict[str, set[int]]:
        """mesh -> vertices on hair surfaces."""
        out: dict[str, set[int]] = {}
        meshes = {m.name: m for m in inspection.meshes}
        for name, slots in self.hair_slots(confident_only).items():
            mesh = meshes.get(name)
            if mesh is None:
                continue
            if None in slots:
                out[name] = set(range(len(mesh.vertex_positions)))
                continue
            verts: set[int] = set()
            for poly, slot in zip(mesh.polygons, mesh.polygon_materials):
                if slot in slots:
                    verts.update(poly)
            out[name] = verts
        return out


def connected_components(mesh: MeshInfo) -> list[list[int]]:
    parent = list(range(len(mesh.vertex_positions)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used: set[int] = set()
    for poly in mesh.polygons:
        used.update(poly)
        for a, b in zip(poly, poly[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for v in used:
        groups.setdefault(find(v), []).append(v)
    return sorted(groups.values(), key=len, reverse=True)


def classify_hair_type(mesh: MeshInfo, frame: AnatomicalFrame, head_height: float, alpha: bool,
                       named_braid: bool = False) -> tuple[str, int]:
    comps = connected_components(mesh) if mesh.polygons else []
    if not comps:
        return UNKNOWN, 0
    quads = sum(1 for p in mesh.polygons if len(p) == 4) / max(len(mesh.polygons), 1)
    stats = []
    for comp in comps[:60]:
        pts = [frame.to_local(mesh.vertex_positions[v]) for v in comp]
        pa = principal_axis(pts)
        if pa is None:
            continue
        axis, elong, _ = pa
        ext = max(dot(sub(p, pa[2]), axis) for p in pts) - min(dot(sub(p, pa[2]), axis) for p in pts)
        stats.append((len(comp), elong, ext, pa[2]))
    if not stats:
        return UNKNOWN, len(comps)
    if alpha and len(comps) >= 12 and quads > 0.5:
        return ALPHA_CARDS, len(comps)
    big = [s for s in stats if s[0] >= 0.15 * len(mesh.vertex_positions)]
    if named_braid:
        return BRAID, len(comps)
    if big and len(big) <= 3 and all(s[1] >= 0.8 and s[2] >= 0.9 * head_height for s in big):
        back = all(s[3][2] < 0 for s in big)
        return (PONYTAIL if back else CLUMP), len(comps)
    if len(comps) >= 4 and sum(1 for s in stats if s[1] >= 0.75 and s[2] >= 0.5 * head_height) >= max(3, len(stats) // 2):
        return CLUMP, len(comps)
    if len(comps) <= 3 and stats[0][1] < 0.65:
        return SOLID_MASS, len(comps)
    return (ALPHA_CARDS if alpha else UNKNOWN), len(comps)


def _surfaces(inspection: CharacterInspection, armature: str):
    """(mesh, slot, vertex indices) per surface; ``slot`` / ``indices`` are None for a single-surface mesh."""
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature):
            continue
        used = sorted(set(mesh.polygon_materials)) if len(mesh.polygon_materials) == len(mesh.polygons) else []
        if len(mesh.materials) <= 1 or len(used) <= 1:
            yield mesh, None, None
            continue
        per: dict[int, set[int]] = {}
        for poly, slot in zip(mesh.polygons, mesh.polygon_materials):
            per.setdefault(slot, set()).update(poly)
        for slot in used:
            if slot < len(mesh.materials):
                yield mesh, slot, sorted(per[slot])


def _sub_mesh(mesh: MeshInfo, slot: int | None) -> MeshInfo:
    if slot is None:
        return mesh
    polys = [p for p, s in zip(mesh.polygons, mesh.polygon_materials) if s == slot]
    return MeshInfo(mesh.name, mesh.vertices, len(polys), mesh.armature, mesh.vertex_groups, [mesh.materials[slot]],
                    vertex_positions=mesh.vertex_positions, polygons=polys)


def _bone_mass(inspection: CharacterInspection, armature: ArmatureInfo) -> dict[str, dict[str, float]]:
    """bone -> {mesh: total skin weight}."""
    mass: dict[str, dict[str, float]] = {}
    for mesh in inspection.meshes:
        if mesh.armature not in (None, armature.name):
            continue
        for weights in mesh.vertex_weights.values():
            for bone, w in weights.items():
                mass.setdefault(bone, {})
                mass[bone][mesh.name] = mass[bone].get(mesh.name, 0.0) + w
    return mass


def _candidate_chains(armature: ArmatureInfo, head_source: str, mapped: set[str]) -> list[list[str]]:
    """Unmapped deform bone chains hanging off the head (or a mapped head descendant)."""
    roots = []
    for name in armature.descendants(head_source):
        if name in mapped:
            continue
        info = armature.bone(name)
        if info is None or not info.deform:
            continue
        parent = armature.parent(name)
        if parent == head_source or parent in mapped or (parent is not None and parent not in mapped
                                                         and not armature.is_ancestor(head_source, parent)):
            roots.append(name)
    chains = []
    for root in roots:
        path = [root]
        while True:
            kids = [c for c in armature.children(path[-1]) if c not in mapped]
            if not kids:
                break
            path.append(max(kids, key=lambda k: (armature.subtree_size(k), armature.bone(k).length)))
        chains.append(path)
    return chains


def analyze_hair(inspection: CharacterInspection, armature: ArmatureInfo, humanoid_map: dict[str, str],
                 frame: AnatomicalFrame, head_base: Vec3, head_height: float, head_width: float,
                 thresholds: Thresholds = DEFAULT_THRESHOLDS) -> HairAnalysis:
    result = HairAnalysis()
    mapped = set(humanoid_map.values())
    head_source = humanoid_map.get("head")
    if not head_height or head_height <= 0:
        result.warnings.append(AnalysisWarning("hair-no-head-size", "head size unknown: hair analysis skipped", "hair"))
        return result
    head_local = frame.to_local(head_base)
    mass = _bone_mass(inspection, armature)
    materials: dict[str, MaterialInfo] = {m.name: m for m in inspection.materials}

    # --- chains first: authored hair bones are strong evidence for the meshes they deform.
    chains = _candidate_chains(armature, head_source, mapped) if head_source else []
    hair_bones: set[str] = set()
    for path in chains:
        first = armature.bone(path[0])
        tokens = {t for b in path for t in parse_bone_name(b).tokens}
        if tokens & _FACE_BONE_TOKENS and not any(_has(b, _NAME_TOKENS) for b in path):
            continue
        if len(path) < 2:
            continue
        start = frame.to_local(first.head)
        end = frame.to_local(armature.bone(path[-1]).tail)
        length = sum(armature.bone(b).length for b in path)
        direction = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
        down = max(0.0, min(1.0, -direction[1] / max(length, 1e-9)))
        named = any(_has(b, _NAME_TOKENS) for b in path)
        hair_mass = sum(m for b in path for mesh, m in mass.get(b, {}).items()
                        if any(_has(mesh, _NAME_TOKENS) or _has(x, _NAME_TOKENS)
                               for x in [mesh, *(mat for me in inspection.meshes if me.name == mesh for mat in me.materials)]))
        total = sum(m for b in path for m in mass.get(b, {}).values())
        skinning = (hair_mass / total) if total > 0 else (0.35 if mass else 0.5)
        ev = {"name": 1.0 if named else 0.0,
              "hierarchy": 1.0 if armature.parent(path[0]) == head_source else 0.7,
              "direction": down, "skinning": skinning}
        weights = {"name": 0.3, "hierarchy": 0.2, "direction": 0.2, "skinning": 0.3}
        if not named:
            weights["name"] = 0.0
            ev.pop("name")
        conf = fuse(ev, weights, veto_below=0.15, veto_cap=0.5)
        # Front-of-face placement is a face rig, not hair.
        if start[2] - head_local[2] > 0.35 * head_width and abs(start[0]) < 0.4 * head_width and not named:
            conf = min(conf, 0.3)
        chain = HairChain(path[0], path, round(conf, 4), {k: round(v, 3) for k, v in ev.items()},
                          length_ratio=length / head_height, at_back=start[2] < head_local[2], named=named)
        result.chains.append(chain)
        if conf >= thresholds.hair_possible:
            hair_bones.update(path)

    # --- surfaces: a whole mesh, or each material slot of a multi-material mesh (one mesh often carries skin *and*
    # hair: VRoid's body with the back hair, single-mesh store characters)
    surfaces = list(_surfaces(inspection, armature.name))
    local_pts = {}
    for mesh, slot, vidx in surfaces:
        positions = [mesh.vertex_positions[i] for i in vidx] if vidx is not None else mesh.vertex_positions
        local_pts[(mesh.name, slot)] = [frame.to_local(p) for p in positions]
    # the top of everything the head carries: hair is the outer shell that reaches it
    near_head = [max(p[1] for p in pts) for pts in local_pts.values() if pts
                 and sum(1 for p in pts if p[1] >= head_local[1] - 0.1 * head_height) > 0.5 * len(pts)]
    envelope_top = max(near_head, default=None)
    for mesh, slot, vidx in surfaces:
        ev: dict[str, float] = {}
        if _has(mesh.name, _NAME_TOKENS):
            ev["object_name"] = 1.0
        slot_mats = [mesh.materials[slot]] if slot is not None else list(mesh.materials)
        mats = [materials[m] for m in slot_mats if m in materials]
        if any(_has(m.name, _NAME_TOKENS) for m in mats):
            ev["material_name"] = 1.0
        if any(_has(i, _NAME_TOKENS) for m in mats for i in m.images):
            ev["texture_name"] = 1.0
        if slot is not None and ev.get("object_name") and not ev.get("material_name") \
                and any(_has(m, _NOT_HAIR_TOKENS) for m in slot_mats):
            ev.pop("object_name")  # the skin slot of a mesh called "Hair..." is not hair
        positions = [mesh.vertex_positions[i] for i in vidx] if vidx is not None else mesh.vertex_positions
        pts = local_pts[(mesh.name, slot)]
        if positions:
            above = sum(1 for p in pts if p[1] >= head_local[1] - 0.1 * head_height and abs(p[0]) <= 1.3 * head_width * 0.5 + 0.2 * head_width
                        and abs(p[2] - head_local[2]) <= 1.5 * head_width)
            frac = above / len(pts)
            body_extent = max(p[1] for p in pts) - min(p[1] for p in pts)
            if body_extent < 3.0 * head_height:
                ev["head_proximity"] = frac
            top = sorted(pts, key=lambda p: -p[1])[: max(3, len(pts) // 10)]
            ev["scalp"] = 1.0 if all(abs(p[0]) < 0.6 * head_width and abs(p[2] - head_local[2]) < 0.8 * head_width for p in top) \
                and body_extent < 3.0 * head_height else 0.0
        wsrc = [mesh.vertex_weights.get(i, {}) for i in vidx] if vidx is not None else list(mesh.vertex_weights.values())
        bone_w = sum(w for weights in wsrc for b, w in weights.items() if b in hair_bones)
        all_w = sum(w for weights in wsrc for w in weights.values())
        if all_w > 0 and hair_bones:
            ev["hair_bones"] = bone_w / all_w
        alpha = any(m.alpha_mode in ("BLEND", "MASK", "CLIP") for m in mats)
        if alpha and any(k in ev for k in ("object_name", "material_name", "head_proximity")):
            ev["alpha"] = 1.0
        if pts and envelope_top is not None and ev.get("head_proximity", 0.0) > 0.4:
            # geometry alone: the outer shell of the head (reaches the very top, wraps round behind the skull) made
            # of many strands or cut-out cards. Eyes, teeth and face skin are none of these.
            reaches_top = max(p[1] for p in pts) >= envelope_top - 0.04 * head_height
            behind = sum(1 for p in pts if p[2] < head_local[2] - 0.3 * head_width) / len(pts)
            if reaches_top and behind >= 0.15:
                comps = len(connected_components(_sub_mesh(mesh, slot))) if mesh.polygons else 0
                if comps >= 3 or alpha:
                    ev["outer_shell"] = 1.0
        miss = 1.0
        for k, v in ev.items():
            miss *= 1.0 - _RELIABILITY[k] * v
        conf = 1.0 - miss
        # Position alone is a weak reading: it may not exceed POSSIBLE without a naming / skinning signal.
        if not any(k in ev and ev[k] > 0.5 for k in ("object_name", "material_name", "texture_name", "hair_bones",
                                                     "outer_shell")):
            conf = min(conf, thresholds.hair_confident - 0.05)
            # ... and a mesh that names itself face / skin / body is not hair just because it sits on the head.
            if _has(mesh.name, _NOT_HAIR_TOKENS) or any(_has(m, _NOT_HAIR_TOKENS) for m in slot_mats):
                conf = min(conf, 0.3)
        cls = HAIR_CONFIDENT if conf >= thresholds.hair_confident else HAIR_POSSIBLE if conf >= thresholds.hair_possible else NOT_HAIR
        htype, comps = (UNKNOWN, 0)
        if cls != NOT_HAIR and positions:
            htype, comps = classify_hair_type(_sub_mesh(mesh, slot), frame, head_height, alpha,
                                              _has(mesh.name, ("braid", "plait")))
        result.meshes.append(HairMeshAssessment(mesh.name, cls, round(conf, 4), {k: round(v, 3) for k, v in ev.items()},
                                                htype, comps, slot, mesh.materials[slot] if slot is not None else None))

    # --- categories
    usable = [c for c in result.chains if c.confidence >= thresholds.hair_possible]
    for chain in usable:
        chain.category = _chain_category(chain, len(usable))
    result.classification = _overall(usable, result.meshes)
    if not usable and result.confident_meshes:
        result.warnings.append(AnalysisWarning(
            "hair-no-bones", "hair mesh detected but no usable hair bone chains", "hair"))
    return result


def _chain_category(chain: HairChain, n_chains: int) -> str:
    if any(_has(b, ("braid", "plait")) for b in chain.bones):
        return "braid"
    if n_chains <= 3 and chain.length_ratio >= 0.9 and chain.at_back:
        return "ponytail"
    if chain.length_ratio >= 1.0 and n_chains >= 4:
        return LONG_HAIR
    if chain.length_ratio < 0.7 and n_chains >= 4:
        return SHORT_HAIR
    return SOFT_STRAND


def _overall(chains: list[HairChain], meshes: list[HairMeshAssessment]) -> str | None:
    if chains:
        counts: dict[str, int] = {}
        for c in chains:
            counts[c.category] = counts.get(c.category, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    types = [m.hair_type for m in meshes if m.classification != NOT_HAIR]
    if not types:
        return None
    if SOLID_MASS in types:
        return SHORT_HAIR
    return None
