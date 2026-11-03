"""core.compress on a hand-made GLB: VRM extensions pass through, the compressed parts are well-formed."""

import struct
import zlib

import numpy as np
import pytest

from character_vrm.core import compress


def png(w=8, h=8) -> bytes:
    raw = b"".join(b"\0" + bytes(sum(([x * 30 % 256, y * 30 % 256, 128, 255] for x in range(w)), [])) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def glb(tmp_path):
    pos = np.random.default_rng(0).random((60, 3), np.float32)
    idx = np.arange(60, dtype=np.uint16)
    image = png()
    blob = pos.tobytes() + idx.tobytes()
    blob += b"\0" * (-len(blob) % 4)
    img_off = len(blob)
    blob += image
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes},
                        {"buffer": 0, "byteOffset": pos.nbytes, "byteLength": idx.nbytes},
                        {"buffer": 0, "byteOffset": img_off, "byteLength": len(image)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 60, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5123, "count": 60, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "extensionsUsed": ["VRMC_vrm"],
        "extensions": {"VRMC_vrm": {"specVersion": "1.0", "expressions": {"preset": {"aa": {"morphTargetBinds": [
            {"node": 0, "index": 3, "weight": 1.0}]}}}}},
    }
    path = tmp_path / "a.vrm"
    compress.write_glb(path, doc, blob)
    return path, doc


def test_meshopt_keeps_vrm_and_writes_the_extension(tmp_path):
    pytest.importorskip("meshoptimizer")
    path, before = glb(tmp_path)
    stats = compress.compress_geometry(path)
    doc, blob = compress.read_glb(path)
    assert stats["views"] == 2
    assert doc["extensions"] == before["extensions"]  # VRM data untouched (an index inside it is not a texture)
    assert "EXT_meshopt_compression" in doc["extensionsRequired"]
    assert doc["buffers"][1]["extensions"]["EXT_meshopt_compression"]["fallback"] is True
    modes = {doc["bufferViews"][i]["extensions"]["EXT_meshopt_compression"]["mode"] for i in (0, 1)}
    assert modes == {"ATTRIBUTES", "TRIANGLES"}
    ext = doc["bufferViews"][0]["extensions"]["EXT_meshopt_compression"]
    assert ext["byteOffset"] + ext["byteLength"] <= len(blob) and ext["count"] == 60 and ext["byteStride"] == 12
    image_view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    start = image_view["byteOffset"]
    assert bytes(blob[start:start + 8]) == b"\x89PNG\r\n\x1a\n"  # an uncompressed view moved with its bytes


def test_ktx2_marks_textures_and_keeps_vrm(tmp_path):
    if compress.ktx_tool() is None:
        pytest.skip("KTX-Software not fetched (packaging/fetch_ktx.py)")
    path, before = glb(tmp_path)
    stats = compress.compress_textures(path, log=lambda *_: None)
    doc, blob = compress.read_glb(path)
    assert stats["images"] == 1 and doc["images"][0]["mimeType"] == "image/ktx2"
    assert doc["textures"][0] == {"extensions": {"KHR_texture_basisu": {"source": 0}}}
    assert doc["extensions"] == before["extensions"]
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    assert bytes(blob[view["byteOffset"]:view["byteOffset"] + 12]) == b"\xabKTX 20\xbb\r\n\x1a\n"


def test_simplify_hair_drops_vertices_and_keeps_every_attribute(tmp_path):
    pytest.importorskip("meshoptimizer")
    n = 20  # a dense, nearly flat 20x20 sheet named as hair
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    pos = np.stack([xs.ravel(), ys.ravel(), 0.001 * np.sin(xs.ravel() * 3)], 1).astype(np.float32)
    lab = np.zeros((n * n, 4), np.uint8)
    lab[:, 0] = 4
    quads = [(r * n + c, r * n + c + 1, (r + 1) * n + c) for r in range(n - 1) for c in range(n - 1)]
    quads += [(r * n + c + 1, (r + 1) * n + c + 1, (r + 1) * n + c) for r in range(n - 1) for c in range(n - 1)]
    idx = np.array(quads, np.uint32).ravel()
    blob = pos.tobytes() + lab.tobytes() + idx.tobytes()
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(blob)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes},
                           {"buffer": 0, "byteOffset": pos.nbytes, "byteLength": lab.nbytes, "byteStride": 4},
                           {"buffer": 0, "byteOffset": pos.nbytes + lab.nbytes, "byteLength": idx.nbytes}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": n * n, "type": "VEC3",
                          "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
                         {"bufferView": 1, "componentType": 5121, "count": n * n, "type": "SCALAR"},
                         {"bufferView": 2, "componentType": 5125, "count": len(idx), "type": "SCALAR"}],
           "materials": [{"name": "F00_000_Hair_00_HAIR_01"}],
           "meshes": [{"name": "Hair001", "primitives": [{"attributes": {"POSITION": 0, "_MOVIVA_SURFACE": 1},
                                                          "indices": 2, "material": 0}]}]}
    path = tmp_path / "h.vrm"
    compress.write_glb(path, doc, blob)
    stats = compress.simplify_hair(path)
    doc, blob = compress.read_glb(path)
    assert stats["vertices"][1] < stats["vertices"][0] and stats["triangles"][1] < stats["triangles"][0]
    p = doc["meshes"][0]["primitives"][0]
    count = doc["accessors"][p["attributes"]["POSITION"]]["count"]
    assert count == stats["vertices"][1] == doc["accessors"][p["attributes"]["_MOVIVA_SURFACE"]]["count"]
    assert set(compress._read_accessor(doc, blob, p["attributes"]["_MOVIVA_SURFACE"])[:, 0]) == {4}
    assert compress._read_accessor(doc, blob, p["indices"]).max() < count
