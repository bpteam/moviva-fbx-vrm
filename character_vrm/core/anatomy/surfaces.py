"""Per-vertex surface roles and mesh-boundary loops (pure).

A *surface* is one material slot of one mesh: a single mesh often carries skin, eyes, lashes, the mouth interior and
even hair (VRoid's ``Face`` mesh, a CGTrader single-mesh character), so roles are assigned per polygon from the slot's
material and the mesh name, never from "any material of the mesh".

Boundary loops of the skin are the openings of the face: the eyes and the mouth (a lip seam or a hole into the mouth
interior) are small closed loops, the outer edge of the face / neck cut is a large one. Vertices are welded by position
first, because glTF and FBX split vertices along UV seams.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import MeshInfo
from .vec import Vec3

SKIN, EYE, TEETH, HAIR, LASH, BROW = "skin", "eye", "teeth", "hair", "lash", "brow"

# token -> role; checked on the material name first, then the mesh name. Order matters (first hit wins).
_ROLE_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (LASH, ("lash", "eyeline", "eyeliner", "occlusion", "tearline", "tear")),
    (BROW, ("brow", "eyebrow")),
    (TEETH, ("teeth", "tooth", "tongue", "mouth", "gum", "oral")),
    (HAIR, ("hair", "bang", "fringe", "ponytail", "braid", "strand", "scalp")),
    (EYE, ("eye", "iris", "pupil", "cornea", "sclera", "eyeball", "highlight")),
    (SKIN, ("skin", "face", "head", "body")),
)


def _tokens(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else " " for c in name)


def name_role(name: str) -> str | None:
    s = _tokens(name)
    for role, tokens in _ROLE_TOKENS:
        if any(t in s for t in tokens):
            return role
    return None


def slot_role(mesh: MeshInfo, slot: int | None) -> str:
    """Role of one material slot: the material decides, else the mesh name, else skin."""
    if slot is not None and 0 <= slot < len(mesh.materials):
        r = name_role(mesh.materials[slot])
        if r is not None:
            return r
    if slot is None and len(mesh.materials) == 1:
        r = name_role(mesh.materials[0])
        if r is not None:
            return r
    return name_role(mesh.name) or SKIN


def vertex_slots(mesh: MeshInfo) -> dict[int, int]:
    """vertex -> material slot of the polygons that use it (the most frequent one)."""
    if not mesh.polygon_materials or len(mesh.polygon_materials) != len(mesh.polygons):
        return {}
    counts: dict[int, dict[int, int]] = {}
    for poly, slot in zip(mesh.polygons, mesh.polygon_materials):
        for v in poly:
            c = counts.setdefault(v, {})
            c[slot] = c.get(slot, 0) + 1
    return {v: max(c.items(), key=lambda kv: kv[1])[0] for v, c in counts.items()}


def vertex_roles(mesh: MeshInfo, hair_slots: set[int | None] | None = None) -> dict[int, str]:
    """vertex -> role. ``hair_slots`` are slots (``None`` = the whole mesh) the hair analysis judged to be hair."""
    hair_slots = hair_slots or set()
    slots = vertex_slots(mesh)
    whole = slot_role(mesh, None)
    roles: dict[int, str] = {}
    by_slot: dict[int, str] = {}
    for i in range(len(mesh.vertex_positions)):
        s = slots.get(i)
        if None in hair_slots or (s is not None and s in hair_slots):
            roles[i] = HAIR
            continue
        if s is None:
            roles[i] = whole if len(mesh.materials) <= 1 else (name_role(mesh.name) or SKIN)
            continue
        if s not in by_slot:
            by_slot[s] = slot_role(mesh, s)
        roles[i] = by_slot[s]
    return roles


# ---------------------------------------------------------------- welding and boundaries

def weld_map(positions: list[Vec3], tolerance: float) -> list[int]:
    """vertex -> representative vertex with (almost) the same position."""
    cell = max(tolerance, 1e-12)
    rep: dict[tuple[int, int, int], int] = {}
    out = []
    for i, p in enumerate(positions):
        k = (round(p[0] / cell), round(p[1] / cell), round(p[2] / cell))
        out.append(rep.setdefault(k, i))
    return out


@dataclass
class BoundaryLoop:
    mesh: str
    vertices: list[int] = field(default_factory=list)  # welded representatives

    def __len__(self) -> int:
        return len(self.vertices)


def boundary_loops(mesh: MeshInfo, polygons: list[tuple[int, ...]], tolerance: float) -> list[BoundaryLoop]:
    """Connected components of the boundary edges of ``polygons`` (a subset of the mesh), after welding."""
    weld = weld_map(mesh.vertex_positions, tolerance)
    count: dict[tuple[int, int], int] = {}
    for poly in polygons:
        w = [weld[v] for v in poly]
        for a, b in zip(w, w[1:] + w[:1]):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            count[e] = count.get(e, 0) + 1
    edges = [e for e, c in count.items() if c == 1]
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for v in parent:
        groups.setdefault(find(v), []).append(v)
    return [BoundaryLoop(mesh.name, sorted(g)) for g in sorted(groups.values(), key=len, reverse=True)]
