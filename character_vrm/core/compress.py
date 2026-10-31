"""Opt-in compression of a finished VRM (GLB): KTX2 textures and meshopt geometry.

Both rewrite the GLB in place at the glTF level, so every VRM extension (humanoid, expressions, MToon, springs)
passes through untouched. Only loaders with the decoders open the result: three.js with ``KTX2Loader`` /
``MeshoptDecoder`` (Moviva) does; many VRM apps do not, so the converter leaves both off unless asked.

- KTX2: each image goes through Khronos' ``ktx create`` (KTX-Software, Apache-2.0) as UASTC + zstd with mipmaps; the
  texture then names it under ``KHR_texture_basisu``. Normal maps stay linear, every other image is sRGB.
- meshopt: ``EXT_meshopt_compression`` on vertex attributes and indices, encoded by ``meshoptimizer`` (MIT).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


def read_glb(path: Path) -> tuple[dict, bytearray]:
    data = path.read_bytes()
    magic, _, _ = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"{path} is not a GLB/VRM file")
    jlen, jtype = struct.unpack_from("<II", data, 12)
    doc = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blob = bytearray()
    if off < len(data):
        blen, _ = struct.unpack_from("<II", data, off)
        blob = bytearray(data[off + 8:off + 8 + blen])
    return doc, blob


def write_glb(path: Path, doc: dict, blob: bytes) -> None:
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob = bytes(blob) + b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(js), JSON_CHUNK) + js)
        f.write(struct.pack("<II", len(blob), BIN_CHUNK) + blob)
    os.replace(tmp, path)


class _Packer:
    """Rebuilds the binary chunk: kept buffer views copied, replaced ones appended."""

    def __init__(self, doc: dict, blob: bytes) -> None:
        self.doc, self.old, self.new = doc, blob, bytearray()

    def add(self, data: bytes) -> tuple[int, int]:
        self.new += b"\0" * (-len(self.new) % 4)
        start = len(self.new)
        self.new += data
        return start, len(data)

    def view_bytes(self, i: int) -> bytes:
        bv = self.doc["bufferViews"][i]
        o = bv.get("byteOffset", 0)
        return bytes(self.old[o:o + bv["byteLength"]])


def ktx_tool() -> str | None:
    """``ktx`` from the vendored KTX-Software, else PATH."""
    names = ["ktx.exe" if sys.platform == "win32" else "ktx"]
    roots = [os.environ.get("MOVIVA_KTX", "")]
    if getattr(sys, "frozen", False):
        roots.append(os.path.join(getattr(sys, "_MEIPASS", ""), "vendor", "ktx", "bin"))
    roots.append(str(Path(__file__).resolve().parents[2] / "vendor" / "ktx" / "bin"))
    for r in roots:
        for n in names:
            if r and Path(r, n).is_file():
                return str(Path(r, n))
    return shutil.which("ktx")


def _linear_images(doc: dict) -> set[int]:
    """Images used only as normal maps (glTF ``normalTexture``, MToon's too): encoded as linear, not sRGB."""
    normal, other = set(), set()

    def walk(o, key=""):
        if isinstance(o, dict):
            if "index" in o and isinstance(o["index"], int) and key.lower().endswith("texture"):
                tex = doc["textures"][o["index"]]
                src = tex.get("source", tex.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))
                if src is not None:
                    (normal if "normal" in key.lower() else other).add(src)
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)

    walk(doc.get("materials", []))
    walk(doc.get("extensions", {}))
    return normal - other


def compress_textures(path: Path, log=print) -> dict:
    """Every PNG/JPEG image as KTX2 (UASTC, zstd, mipmaps). Returns {"images", "before", "after"} in bytes."""
    tool = ktx_tool()
    if tool is None:
        raise RuntimeError("KTX-Software's 'ktx' tool is not available")
    doc, blob = read_glb(path)
    images = doc.get("images", [])
    linear = _linear_images(doc)
    pack = _Packer(doc, blob)
    replaced: dict[int, bytes] = {}
    before = after = 0
    with tempfile.TemporaryDirectory(prefix="moviva-ktx-") as tmp:
        for i, im in enumerate(images):
            if "bufferView" not in im or im.get("mimeType") not in ("image/png", "image/jpeg"):
                continue
            raw = pack.view_bytes(im["bufferView"])
            src = Path(tmp, f"{i}.{'png' if im['mimeType'] == 'image/png' else 'jpg'}")
            src.write_bytes(raw)
            out = Path(tmp, f"{i}.ktx2")
            fmt = "R8G8B8A8_UNORM" if i in linear else "R8G8B8A8_SRGB"
            cmd = [tool, "create", "--format", fmt, "--encode", "uastc", "--uastc-quality", "2", "--zstd", "18",
                   "--generate-mipmap", "--assign-tf", "linear" if i in linear else "srgb", str(src), str(out)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out.is_file():
                log(f"KTX2: image {i} kept as is ({proc.stderr.strip()[:120]})")
                continue
            replaced[i] = out.read_bytes()
            before += len(raw)
            after += len(replaced[i])
    if not replaced:
        return {"images": 0, "before": 0, "after": 0}
    # rebuild the binary chunk: every buffer view keeps its bytes but a replaced image's, which gets the KTX2 file
    image_view = {im["bufferView"]: i for i, im in enumerate(images) if i in replaced}
    for vi, bv in enumerate(doc["bufferViews"]):
        data = replaced[image_view[vi]] if vi in image_view else pack.view_bytes(vi)
        bv["byteOffset"], bv["byteLength"] = pack.add(data)
    doc["buffers"][0]["byteLength"] = len(pack.new)
    for i in replaced:
        images[i]["mimeType"] = "image/ktx2"
    for tex in doc.get("textures", []):
        if tex.get("source") in replaced:
            tex.setdefault("extensions", {})["KHR_texture_basisu"] = {"source": tex.pop("source")}
    for key in ("extensionsUsed", "extensionsRequired"):
        doc[key] = sorted(set(doc.get(key, [])) | {"KHR_texture_basisu"})
    write_glb(path, doc, pack.new)
    return {"images": len(replaced), "before": before, "after": after}


_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def compress_geometry(path: Path) -> dict:
    """Vertex attributes, morph targets and indices as ``EXT_meshopt_compression`` (meshoptimizer's codecs; the
    data three.js' ``MeshoptDecoder`` reads). A view shared in ways the codec cannot express stays as it is.
    Returns {"views", "before", "after"} in bytes."""
    import numpy as np
    import meshoptimizer as mo
    mo.encode_vertex_version(0)  # the version every MeshoptDecoder reads
    doc, blob = read_glb(path)
    acc = doc["accessors"]
    mode: dict[int, tuple[str, int, int]] = {}  # view -> (mode, stride, count)
    bad: set[int] = set()

    def claim(view: int, m: str, stride: int, count: int) -> None:
        if view in mode and mode[view] != (m, stride, count):
            bad.add(view)
        mode[view] = (m, stride, count)

    for a in acc:  # sparse views and anything not a plain mesh attribute stay raw
        for part in ("indices", "values"):
            if "sparse" in a:
                bad.add(a["sparse"][part]["bufferView"])
    for mesh in doc["meshes"]:
        for p in mesh["primitives"]:
            attrs = list(p["attributes"].values()) + [i for t in p.get("targets", []) for i in t.values()]
            for i in attrs:
                a = acc[i]
                if "bufferView" not in a:
                    continue
                stride = _SIZE[a["componentType"]] * _WIDTH[a["type"]]
                v = a["bufferView"]
                if a.get("byteOffset", 0) or doc["bufferViews"][v].get("byteStride", stride) != stride or stride % 4:
                    bad.add(v)
                claim(v, "ATTRIBUTES", stride, a["count"])
            if "indices" in p:
                a = acc[p["indices"]]
                size = _SIZE[a["componentType"]]
                if a.get("byteOffset", 0) or size == 1:
                    bad.add(a["bufferView"])
                claim(a["bufferView"], "TRIANGLES" if p.get("mode", 4) == 4 else "INDICES", size, a["count"])
    pack = _Packer(doc, blob)
    fallback = 0
    before = after = 0
    views = 0
    for vi, bv in enumerate(doc["bufferViews"]):
        raw = pack.view_bytes(vi)
        if vi in mode and vi not in bad and len(raw) == mode[vi][1] * mode[vi][2]:
            m, stride, count = mode[vi]
            if m == "ATTRIBUTES":
                enc = mo.encode_vertex_buffer(np.frombuffer(raw, np.uint8).reshape(count, stride), count, stride)
            else:
                idx = np.frombuffer(raw, np.uint16 if stride == 2 else np.uint32).astype(np.uint32)
                enc = (mo.encode_index_buffer if m == "TRIANGLES" else mo.encode_index_sequence)(idx, count, int(idx.max()) + 1)
            off, length = pack.add(enc)
            bv["extensions"] = {"EXT_meshopt_compression": {"buffer": 0, "byteOffset": off, "byteLength": length,
                                                             "byteStride": stride, "count": count, "mode": m}}
            bv["buffer"], bv["byteOffset"] = 1, fallback
            fallback += len(raw) + (-len(raw) % 4)
            before, after, views = before + len(raw), after + length, views + 1
        else:
            bv["byteOffset"], bv["byteLength"] = pack.add(raw)
    if not views:
        return {"views": 0, "before": 0, "after": 0}
    doc["buffers"] = [{"byteLength": len(pack.new)},
                      {"byteLength": fallback, "extensions": {"EXT_meshopt_compression": {"fallback": True}}}]
    for key in ("extensionsUsed", "extensionsRequired"):
        doc[key] = sorted(set(doc.get(key, [])) | {"EXT_meshopt_compression"})
    write_glb(path, doc, pack.new)
    return {"views": views, "before": before, "after": after}


def _read_accessor(doc: dict, blob: bytes, i: int):
    """An accessor as a (count, width) float array (normalised integers scaled to 0..1). Dense views only."""
    import numpy as np
    a = doc["accessors"][i]
    dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
    width = _WIDTH[a["type"]]
    bv = doc["bufferViews"][a["bufferView"]]
    item = np.dtype(dtype).itemsize * width
    stride = bv.get("byteStride", item)
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    raw = np.frombuffer(bytes(blob[start:start + stride * (a["count"] - 1) + item]), np.uint8)
    rows = np.lib.stride_tricks.as_strided(raw, (a["count"], item), (stride, 1)).copy()
    out = rows.view(dtype).reshape(a["count"], width).astype(np.float64)
    if a.get("normalized") and dtype != np.float32:
        out /= np.iinfo(dtype).max
    return out


def add_surface_labels(path: Path, votes: dict | None = None) -> dict:
    """The ``_MOVIVA_SURFACE`` vertex attribute on every skinned primitive and the ``MOVIVA_surface_labels`` root
    extension (``core.surface_labels``). Returns the report section: per mesh, vertices per label and the signal
    that decided each material."""
    import numpy as np
    from . import surface_labels as SL
    doc, blob = read_glb(path)
    human = doc.get("extensions", {}).get("VRMC_vrm", {}).get("humanoid", {}).get("humanBones", {})
    arm_nodes = {b["node"] for name, b in human.items() if name in SL.ARM_BONES}
    pack = _Packer(doc, blob)
    labels_for: dict[int, np.ndarray] = {}  # POSITION accessor -> labels (primitives may share vertex buffers)
    report: dict[str, dict] = {}
    for mesh in doc.get("meshes", []):
        mesh_name = mesh.get("name", "")
        entry = report.setdefault(mesh_name or f"mesh{len(report)}", {"labels": {}, "materials": {}})
        for p in mesh["primitives"]:
            attrs = p["attributes"]
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs or "indices" not in p:
                continue
            mat = doc["materials"][p["material"]].get("name", "") if "material" in p else ""
            cls, signal = SL.decide(mat, mesh_name, votes)
            entry["materials"][mat] = {"class": cls, "signal": signal}
            pos = attrs["POSITION"]
            count = doc["accessors"][pos]["count"]
            if pos not in labels_for:
                labels_for[pos] = np.full(count, SL.ID["cloth_body"], np.uint8)
            skin = next((s for s in doc.get("skins", []) if s), None)
            joints = _read_accessor(doc, blob, attrs["JOINTS_0"]).astype(int)
            weights = _read_accessor(doc, blob, attrs["WEIGHTS_0"])
            skin_index = next((n.get("skin") for n in doc["nodes"] if n.get("mesh") == doc["meshes"].index(mesh)), 0)
            nodes = np.array(doc["skins"][skin_index or 0]["joints"]) if skin else np.zeros(1, int)
            arm = (np.isin(nodes[joints.clip(0, len(nodes) - 1)], list(arm_nodes)) * weights).sum(axis=1)
            used = np.unique(_read_accessor(doc, blob, p["indices"]).astype(int).ravel())
            if cls == "per-vertex":  # the nearest point MediaPipe labelled, vertex by vertex
                pts = np.array([q[:3] for q in votes["points"]], float)
                kinds = [q[3] for q in votes["points"]]
                xyz = _read_accessor(doc, blob, pos)[used]
                near = [int(np.argmin(((pts - x) ** 2).sum(axis=1))) for x in xyz]
                ids = np.array([SL.label(kinds[n], w) for n, w in zip(near, arm[used])], np.uint8)
            else:
                ids = np.array([SL.label(cls, w) for w in arm[used]], np.uint8)
            labels_for[pos][used] = ids
            for k, n in zip(*np.unique(ids, return_counts=True)):
                name = SL.CLASSES[k]
                entry["labels"][name] = entry["labels"].get(name, 0) + int(n)
    if not labels_for:
        return {}
    # keep every view's bytes, append one 4-byte-strided view per labelled vertex buffer (glTF aligns attributes)
    for vi in range(len(doc["bufferViews"])):
        bv = doc["bufferViews"][vi]
        bv["byteOffset"], bv["byteLength"] = pack.add(pack.view_bytes(vi))
    accessor_of: dict[int, int] = {}
    for pos, lab in labels_for.items():
        padded = np.zeros((len(lab), 4), np.uint8)
        padded[:, 0] = lab
        off, length = pack.add(padded.tobytes())
        doc["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": length, "byteStride": 4,
                                   "target": 34962})
        doc["accessors"].append({"bufferView": len(doc["bufferViews"]) - 1, "componentType": 5121,
                                 "count": len(lab), "type": "SCALAR"})
        accessor_of[pos] = len(doc["accessors"]) - 1
    for mesh in doc["meshes"]:
        for p in mesh["primitives"]:
            if p["attributes"].get("POSITION") in accessor_of and "JOINTS_0" in p["attributes"]:
                p["attributes"]["_MOVIVA_SURFACE"] = accessor_of[p["attributes"]["POSITION"]]
    doc["buffers"][0]["byteLength"] = len(pack.new)
    doc.setdefault("extensions", {})["MOVIVA_surface_labels"] = {"version": 1, "classes": list(SL.CLASSES),
                                                                  "source": "character_vrm"}
    doc["extensionsUsed"] = sorted(set(doc.get("extensionsUsed", [])) | {"MOVIVA_surface_labels"})
    write_glb(path, doc, pack.new)
    return report


def simplify_hair(path: Path, ratio: float = 0.5, error: float = 0.01) -> dict:
    """Hair meshes with about ``ratio`` of their triangles (meshoptimizer's simplifier, borders locked so strand tips
    and UV seams stay put), then their unused vertices dropped from every
    attribute. Hair primitives with morph targets are left alone. Returns {"triangles": [before, after],
    "vertices": [before, after]}."""
    import numpy as np
    import meshoptimizer as mo
    from . import surface_labels as SL
    doc, blob = read_glb(path)
    hair: dict[int, list[dict]] = {}  # POSITION accessor -> its hair primitives (when all users are hair)
    users: dict[int, list[dict]] = {}
    for mesh in doc.get("meshes", []):
        for p in mesh["primitives"]:
            users.setdefault(p["attributes"]["POSITION"], []).append(p)
            mat = doc["materials"][p["material"]].get("name", "") if "material" in p else ""
            if SL.decide(mat, mesh.get("name", ""))[0] == "hair" and not p.get("targets") and "indices" in p:
                hair.setdefault(p["attributes"]["POSITION"], []).append(p)
    hair = {pos: ps for pos, ps in hair.items() if len(ps) == len(users[pos])}
    if not hair:
        return {}
    tri_before = tri_after = v_before = v_after = 0
    new_data: dict[int, np.ndarray] = {}  # accessor -> rewritten rows
    for pos, prims in hair.items():
        xyz = _read_accessor(doc, blob, pos).astype(np.float32)
        kept: list[np.ndarray] = []
        for p in prims:
            idx = _read_accessor(doc, blob, p["indices"]).astype(np.uint32).ravel()
            out = np.zeros(len(idx), np.uint32)
            target = int(len(idx) * ratio) // 3 * 3
            # borders locked: strand tips and UV seams (split vertices are borders) stay where they are
            n = mo.simplify(out, idx, xyz, target_index_count=target, target_error=error,
                            options=mo.SIMPLIFY_LOCK_BORDER)
            kept.append(out[:n])
            tri_before += len(idx) // 3
            tri_after += n // 3
        used = np.unique(np.concatenate(kept))
        remap = np.full(len(xyz), -1, np.int64)
        remap[used] = np.arange(len(used))
        v_before += len(xyz)
        v_after += len(used)
        for p, k in zip(prims, kept):
            new_data[p["indices"]] = remap[k].astype(np.uint32).reshape(-1, 1)
        for name, a in prims[0]["attributes"].items():  # every vertex attribute of this buffer, labels included
            new_data[a] = _read_accessor_raw(doc, blob, a)[used]
    # rebuild: changed accessors get fresh views (attributes padded to 4-byte strides), the rest keep their bytes
    index_ids = {p["indices"] for ps in hair.values() for p in ps}
    pack = _Packer(doc, blob)
    for vi in range(len(doc["bufferViews"])):
        bv = doc["bufferViews"][vi]
        bv["byteOffset"], bv["byteLength"] = pack.add(pack.view_bytes(vi))
    for ai, rows in new_data.items():
        a = doc["accessors"][ai]
        view: dict = {"buffer": 0}
        if ai in index_ids:
            a["componentType"] = 5125
            data = np.ascontiguousarray(rows, np.uint32).tobytes()
        else:
            raw = np.ascontiguousarray(rows).view(np.uint8).reshape(len(rows), -1)
            stride = raw.shape[1] + (-raw.shape[1] % 4)
            padded = np.zeros((len(rows), stride), np.uint8)
            padded[:, :raw.shape[1]] = raw
            data, view["byteStride"] = padded.tobytes(), stride
            if "min" in a:
                a["min"], a["max"] = rows.min(axis=0).tolist(), rows.max(axis=0).tolist()
        view["byteOffset"], view["byteLength"] = pack.add(data)
        doc["bufferViews"].append(view)
        a["bufferView"], a["count"] = len(doc["bufferViews"]) - 1, len(rows)
        a.pop("byteOffset", None)
    doc["buffers"][0]["byteLength"] = len(pack.new)
    write_glb(path, doc, pack.new)
    return {"triangles": [tri_before, tri_after], "vertices": [v_before, v_after]}


def _read_accessor_raw(doc: dict, blob: bytes, i: int):
    """An accessor's elements as stored, shape (count, components), its own dtype (no normalisation)."""
    import numpy as np
    a = doc["accessors"][i]
    dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
    width = _WIDTH[a["type"]]
    bv = doc["bufferViews"][a["bufferView"]]
    item = np.dtype(dtype).itemsize * width
    stride = bv.get("byteStride", item)
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    raw = np.frombuffer(bytes(blob[start:start + stride * (a["count"] - 1) + item]), np.uint8)
    rows = np.lib.stride_tricks.as_strided(raw, (a["count"], item), (stride, 1)).copy()
    return rows.view(dtype).reshape(a["count"], width)
