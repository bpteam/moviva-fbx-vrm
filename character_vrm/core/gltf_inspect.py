"""A ``bpy``-free GLB / VRM reader that builds the same ``CharacterInspection`` the Blender inspector produces.

It exists for analysis and benchmarking without Blender (``character_vrm.benchmark``): the anatomy analyzer, the
expression resolver and the Auto-Author planner run on its output exactly as they do inside Blender. Coordinates are
converted the way Blender's glTF importer does (glTF +Y up -> Blender +Z up: ``(x, y, z) -> (x, -z, y)``); skinned
vertices are placed in their bind pose; morph targets become sparse ``ShapeKeyInfo.deltas``.

Standard library only. It reads what analysis needs (positions, triangles, material slots, skin weights, morph
targets, bones); it is not a general glTF loader.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .model import ArmatureInfo, BoneInfo, CharacterInspection, MaterialInfo, MeshInfo, ShapeKeyInfo, Vec3

_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_NORM_DIV = {"b": 127.0, "B": 255.0, "h": 32767.0, "H": 65535.0}
DELTA_EPSILON = 1e-7

Mat4 = tuple[float, ...]  # column-major, glTF convention


class GltfError(ValueError):
    pass


@dataclass
class GltfDocument:
    json: dict
    bin: bytes
    path: str = ""

    def accessor(self, index: int) -> list[tuple[float, ...]]:
        acc = self.json["accessors"][index]
        fmt, size = _COMPONENT[acc["componentType"]]
        n = _COUNT[acc["type"]]
        count = acc["count"]
        normalized = acc.get("normalized", False)
        out: list[tuple[float, ...]]
        if "bufferView" in acc:
            view = self.json["bufferViews"][acc["bufferView"]]
            base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = view.get("byteStride") or size * n
            if stride == size * n:
                flat = struct.unpack_from(f"<{count * n}{fmt}", self.bin, base)
                out = [flat[i * n:(i + 1) * n] for i in range(count)]
            else:
                rec = struct.Struct(f"<{n}{fmt}")
                out = [rec.unpack_from(self.bin, base + i * stride) for i in range(count)]
        else:
            out = [(0.0,) * n for _ in range(count)]
        sparse = acc.get("sparse")
        if sparse:
            ifmt, isize = _COMPONENT[sparse["indices"]["componentType"]]
            iview = self.json["bufferViews"][sparse["indices"]["bufferView"]]
            ibase = iview.get("byteOffset", 0) + sparse["indices"].get("byteOffset", 0)
            idx = struct.unpack_from(f"<{sparse['count']}{ifmt}", self.bin, ibase)
            vview = self.json["bufferViews"][sparse["values"]["bufferView"]]
            vbase = vview.get("byteOffset", 0) + sparse["values"].get("byteOffset", 0)
            vals = struct.unpack_from(f"<{sparse['count'] * n}{fmt}", self.bin, vbase)
            out = list(out)
            for k, i in enumerate(idx):
                out[i] = vals[k * n:(k + 1) * n]
        if normalized and fmt in _NORM_DIV:
            d = _NORM_DIV[fmt]
            out = [tuple(max(-1.0, c / d) for c in v) for v in out]
        return out


def read_glb(path: str | Path) -> GltfDocument:
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        if str(path).lower().endswith(".gltf"):
            raise GltfError("only binary .glb / .vrm files are supported by the pure-Python reader")
        raise GltfError(f"{path}: not a GLB container")
    offset, gltf, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8:offset + 8 + length]
        if kind == b"JSON":
            gltf = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        offset += 8 + length
    if gltf is None:
        raise GltfError(f"{path}: no JSON chunk")
    return GltfDocument(gltf, binary, str(path))


# ---------------------------------------------------------------- matrices (column-major 4x4)

def _mul(a: Mat4, b: Mat4) -> Mat4:
    return tuple(sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4)) for c in range(4) for r in range(4))


def _trs(node: dict) -> Mat4:
    if "matrix" in node:
        return tuple(float(x) for x in node["matrix"])
    tx, ty, tz = node.get("translation", (0.0, 0.0, 0.0))
    qx, qy, qz, qw = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    r = (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw),
         2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw),
         2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy))
    return (r[0] * sx, r[1] * sx, r[2] * sx, 0.0, r[3] * sy, r[4] * sy, r[5] * sy, 0.0,
            r[6] * sz, r[7] * sz, r[8] * sz, 0.0, tx, ty, tz, 1.0)


def _point(m: Mat4, p) -> Vec3:
    x, y, z = p[0], p[1], p[2]
    return (m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14])


def _vector(m: Mat4, v) -> Vec3:
    x, y, z = v[0], v[1], v[2]
    return (m[0] * x + m[4] * y + m[8] * z, m[1] * x + m[5] * y + m[9] * z, m[2] * x + m[6] * y + m[10] * z)


def to_blender(p) -> Vec3:
    """glTF (+Y up, +Z forward) -> Blender (+Z up), as the glTF importer converts it."""
    return (float(p[0]), -float(p[2]), float(p[1]))


def _globals(doc: GltfDocument) -> tuple[list[Mat4], list[int | None]]:
    nodes = doc.json.get("nodes", [])
    parent: list[int | None] = [None] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    world: list[Mat4 | None] = [None] * len(nodes)

    def get(i: int) -> Mat4:
        if world[i] is None:
            local = _trs(nodes[i])
            world[i] = local if parent[i] is None else _mul(get(parent[i]), local)  # type: ignore[arg-type]
        return world[i]  # type: ignore[return-value]

    return [get(i) for i in range(len(nodes))], parent


# ---------------------------------------------------------------- inspection

@dataclass
class GltfInspection:
    """The inspection plus the glTF indices the benchmark needs to read ground truth back."""

    inspection: CharacterInspection
    document: GltfDocument
    node_names: list[str] = field(default_factory=list)
    mesh_object: dict[int, str] = field(default_factory=dict)       # glTF mesh index -> MeshInfo name
    mesh_targets: dict[int, list[str]] = field(default_factory=dict)  # glTF mesh index -> shape key names


def _node_name(doc: GltfDocument, i: int) -> str:
    return doc.json["nodes"][i].get("name") or f"node_{i}"


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]:03d}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def inspect_glb(path: str | Path, armature_name: str = "Armature") -> GltfInspection:
    doc = read_glb(path)
    g = doc.json
    nodes = g.get("nodes", [])
    world, parent = _globals(doc)
    names = _unique([_node_name(doc, i) for i in range(len(nodes))])
    fmt = "glb"
    result = CharacterInspection(source_path=str(path), source_format=fmt)

    # --- armature: every joint of every skin, parented to its nearest joint ancestor (Blender's importer does the
    # same: one armature, bones = joints).
    joints: list[int] = []
    for skin in g.get("skins", []):
        for j in skin.get("joints", []):
            if j not in joints:
                joints.append(j)
    joint_set = set(joints)
    heads = {j: to_blender(_point(world[j], (0.0, 0.0, 0.0))) for j in joints}
    bones: list[BoneInfo] = []
    joint_parent: dict[int, int | None] = {}
    for j in joints:
        p = parent[j]
        while p is not None and p not in joint_set:
            p = parent[p]
        joint_parent[j] = p
    kids: dict[int, list[int]] = {j: [] for j in joints}
    for j, p in joint_parent.items():
        if p is not None:
            kids[p].append(j)
    for j in joints:
        head = heads[j]
        tail = None
        ch = kids[j]
        if ch:
            # Point at the child that continues the chain (the one with the largest subtree, then the farthest).
            def subtree(k: int) -> int:
                return 1 + sum(subtree(c) for c in kids[k])
            best = max(ch, key=lambda k: (subtree(k), math.dist(heads[k], head)))
            if math.dist(heads[best], head) > 1e-6:
                tail = heads[best]
        if tail is None:
            p = joint_parent[j]
            if p is not None and math.dist(heads[p], head) > 1e-6:
                d = tuple(head[i] - heads[p][i] for i in range(3))
                tail = (head[0] + 0.5 * d[0], head[1] + 0.5 * d[1], head[2] + 0.5 * d[2])
            else:
                tail = (head[0], head[1], head[2] + 0.05)
        bones.append(BoneInfo(names[j], names[joint_parent[j]] if joint_parent[j] is not None else None, head, tail))
    if bones:
        result.armatures.append(ArmatureInfo(armature_name, bones))

    # --- materials
    materials = g.get("materials", [])
    mat_names = _unique([m.get("name") or f"material_{i}" for i, m in enumerate(materials)])
    for i, m in enumerate(materials):
        alpha = m.get("alphaMode", "OPAQUE")
        info = MaterialInfo(mat_names[i], channels=["baseColor"], alpha_mode="BLEND" if alpha == "BLEND" else
                            "CLIP" if alpha == "MASK" else "OPAQUE")
        tex = (m.get("pbrMetallicRoughness") or {}).get("baseColorTexture")
        if tex is not None:
            try:
                src = g["textures"][tex["index"]].get("source")
                img = g["images"][src]
                info.images = [img.get("name") or img.get("uri") or f"image_{src}"]
            except (KeyError, IndexError, TypeError):
                pass
        result.materials.append(info)

    # --- meshes: one object per mesh node
    out = GltfInspection(result, doc, names)
    ibm_cache: dict[int, list[Mat4]] = {}
    for ni, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mi = node["mesh"]
        mesh = g["meshes"][mi]
        skin = g["skins"][node["skin"]] if "skin" in node else None
        skin_joints: list[int] = skin.get("joints", []) if skin else []
        bind: list[Mat4] = []
        if skin is not None:
            if node["skin"] not in ibm_cache:
                if "inverseBindMatrices" in skin:
                    ibm_cache[node["skin"]] = [tuple(m) for m in doc.accessor(skin["inverseBindMatrices"])]
                else:
                    ibm_cache[node["skin"]] = [_trs({})] * len(skin_joints)
            ibms = ibm_cache[node["skin"]]
            bind = [_mul(world[j], ibms[k]) for k, j in enumerate(skin_joints)]
        target_names = list((mesh.get("extras") or {}).get("targetNames") or [])
        n_targets = max((len(p.get("targets", [])) for p in mesh["primitives"]), default=0)
        if len(target_names) < n_targets:
            target_names += [f"Key {i}" for i in range(len(target_names), n_targets)]
        target_names = _unique(target_names[:n_targets])

        positions: list[Vec3] = []
        polygons: list[tuple[int, ...]] = []
        poly_mats: list[int] = []
        weights: dict[int, dict[str, float]] = {}
        deltas: list[dict[int, Vec3]] = [{} for _ in range(n_targets)]
        slots: list[str] = []
        base_of: dict[int, int] = {}  # POSITION accessor -> vertex offset (primitives may share vertices)
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            pos_acc = attrs["POSITION"]
            if "material" in prim:
                mname = mat_names[prim["material"]]
                if mname not in slots:
                    slots.append(mname)
                slot = slots.index(mname)
            else:
                slot = 0
            if pos_acc in base_of:
                base = base_of[pos_acc]
            else:
                base = len(positions)
                base_of[pos_acc] = base
                local = doc.accessor(pos_acc)
                mats: list[Mat4] = []
                if skin is not None and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                    js = doc.accessor(attrs["JOINTS_0"])
                    ws = doc.accessor(attrs["WEIGHTS_0"])
                    for vi, (jv, wv) in enumerate(zip(js, ws)):
                        infl: dict[str, float] = {}
                        for jj, ww in zip(jv, wv):
                            if ww > 0 and int(jj) < len(skin_joints):
                                name = names[skin_joints[int(jj)]]
                                infl[name] = infl.get(name, 0.0) + float(ww)
                        if infl:
                            weights[base + vi] = infl
                        k = max(range(len(wv)), key=lambda i: wv[i])
                        mats.append(bind[int(jv[k])] if int(jv[k]) < len(bind) else world[ni])
                else:
                    mats = [world[ni]] * len(local)
                for vi, p in enumerate(local):
                    positions.append(to_blender(_point(mats[vi], p)))
                for ti, target in enumerate(prim.get("targets", [])):
                    if "POSITION" not in target or ti >= n_targets:
                        continue
                    for vi, d in enumerate(doc.accessor(target["POSITION"])):
                        if abs(d[0]) + abs(d[1]) + abs(d[2]) > DELTA_EPSILON:
                            deltas[ti][base + vi] = to_blender(_vector(mats[vi], d))
            idx = [int(i[0]) for i in doc.accessor(prim["indices"])] if "indices" in prim else \
                list(range(len(doc.accessor(pos_acc))))
            for t in range(0, len(idx) - 2, 3):
                polygons.append((base + idx[t], base + idx[t + 1], base + idx[t + 2]))
                poly_mats.append(slot)
        obj_name = names[ni]
        groups = sorted({b for w in weights.values() for b in w})
        info = MeshInfo(
            name=obj_name, vertices=len(positions), triangles=len(polygons),
            armature=armature_name if skin is not None and bones else None, vertex_groups=groups,
            materials=slots, vertex_positions=positions, polygons=polygons, polygon_materials=poly_mats,
            vertex_weights=weights)
        result.meshes.append(info)
        out.mesh_object[mi] = obj_name
        out.mesh_targets[mi] = target_names
        for ti, tname in enumerate(target_names):
            result.shape_keys.append(ShapeKeyInfo(obj_name, tname, deltas=deltas[ti]))
    return out
