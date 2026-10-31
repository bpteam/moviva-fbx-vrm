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
