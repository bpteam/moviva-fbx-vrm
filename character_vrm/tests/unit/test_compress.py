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
