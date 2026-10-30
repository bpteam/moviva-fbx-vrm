"""Builds a *test* GLB that looks like what the VRM Add-on writes, so the output validation can be
exercised without Blender. This is test scaffolding only, not a serializer used by the converter."""

from __future__ import annotations

import json
import struct

from character_vrm.core.model import CharacterInspection, VrmPlan


def glb_bytes(gltf: dict) -> bytes:
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    total = 12 + 8 + len(payload)
    return struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(payload), b"JSON") + payload


def vrm_gltf(plan: VrmPlan, source: CharacterInspection, *, skins: int | None = None,
             drop_bone: str | None = None, spec_version: str = "1.0", materials: int | None = None) -> dict:
    nodes: list[dict] = []
    bone_nodes: dict[str, int] = {}
    for vrm_bone in plan.humanoid:
        if vrm_bone == drop_bone:
            continue
        bone_nodes[vrm_bone] = len(nodes)
        nodes.append({"name": plan.humanoid[vrm_bone]})
    mesh_nodes: dict[str, int] = {}
    meshes = []
    for mesh in source.meshes:
        keys = [k.name for k in source.shape_keys if k.mesh == mesh.name]
        mesh_nodes[mesh.name] = len(nodes)
        nodes.append({"name": mesh.name, "mesh": len(meshes)})
        meshes.append({"primitives": [{"targets": [{} for _ in keys], "attributes": {}}],
                       "extras": {"targetNames": keys}})
    presets = {}
    for name, mapping in plan.expressions.items():
        binds = []
        for bind in mapping.binds:
            names = meshes[list(mesh_nodes).index(bind.mesh)]["extras"]["targetNames"]
            binds.append({"node": mesh_nodes[bind.mesh], "index": names.index(bind.shape_key), "weight": bind.weight})
        presets[name] = {"morphTargetBinds": binds}
    gltf: dict = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["VRMC_vrm"],
        "nodes": nodes, "meshes": meshes,
        "skins": [{"joints": list(bone_nodes.values())}] * (len(source.skinned_meshes) if skins is None else skins),
        "materials": [{}] * (len(source.materials) if materials is None else materials),
        "images": [{"bufferView": 0}] * len(source.images),
        "extensions": {"VRMC_vrm": {
            "specVersion": spec_version,
            "humanoid": {"humanBones": {b: {"node": n} for b, n in bone_nodes.items()}},
            "expressions": {"preset": presets},
            "lookAt": {"type": plan.look_at.type or "bone"},
        }},
    }
    if plan.springs.chains:
        gltf["extensionsUsed"].append("VRMC_springBone")
        gltf["extensions"]["VRMC_springBone"] = {"springs": [
            {"joints": [{"node": bone_nodes.get("head", 0)} for _ in chain.bones]} for chain in plan.springs.chains]}
    return gltf


def binary_glb(gltf: dict, binary: bytes) -> bytes:
    """A GLB with a BIN chunk (for the pure-Python reader)."""
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(payload) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(payload), b"JSON") + payload
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
