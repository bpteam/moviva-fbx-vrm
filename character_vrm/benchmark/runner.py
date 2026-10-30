"""Scores the anatomy analyzer, expression resolver and Auto-Author against VRM ground truth.

Three passes per model:

- ``named``: the asset as authored (names, materials, morph names intact);
- ``blind``: every bone, mesh, material, image and shape-key name replaced by an opaque id, so only geometry,
  topology, skinning and morph *motion* are left (the ``Bone.001`` / ``Key 17`` case);
- ``strip``: the truth viseme / blink morphs removed, Auto-Author on: what the generator makes from the face
  geometry is compared with what the artist authored.

Errors are divided by the interocular distance (IOD), the usual face-landmark normalisation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core.config import AutoAuthorConfig, ConversionConfig
from ..core.gltf_inspect import GltfInspection, inspect_glb
from ..core.model import CharacterInspection
from ..core.pipeline import Analysis, analyze
from .truth import FACE_KEYS, MOVED_FRACTION, VISEMES, Truth, read_truth


@dataclass
class Renames:
    bones: dict[str, str] = field(default_factory=dict)
    meshes: dict[str, str] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)


def obfuscate(ins: CharacterInspection) -> tuple[CharacterInspection, Renames]:
    """A copy with every name replaced by an opaque id (geometry, skinning and morph deltas unchanged)."""
    out = copy.deepcopy(ins)
    r = Renames()
    for arm in out.armatures:
        for b in arm.bones:
            r.bones.setdefault(b.name, f"Bone.{len(r.bones):03d}")
    mats: dict[str, str] = {}
    imgs: dict[str, str] = {}
    for arm in out.armatures:
        arm.bones = [type(b)(r.bones[b.name], r.bones.get(b.parent) if b.parent else None, b.head, b.tail, b.deform)
                     for b in arm.bones]
        arm.__post_init__()
    for i, m in enumerate(out.meshes):
        r.meshes[m.name] = f"Mesh.{i:03d}"
        m.name = r.meshes[m.name]
        m.vertex_groups = [r.bones.get(g, g) for g in m.vertex_groups]
        m.vertex_weights = {v: {r.bones.get(b, b): w for b, w in ws.items()} for v, ws in m.vertex_weights.items()}
        m.materials = [mats.setdefault(n, f"Material.{len(mats):03d}") for n in m.materials]
        m.parent_bone = r.bones.get(m.parent_bone, m.parent_bone) if m.parent_bone else None
        m.attached_bones = [r.bones.get(b, b) for b in m.attached_bones]
    for mat in out.materials:
        mat.name = mats.setdefault(mat.name, f"Material.{len(mats):03d}")
        mat.images = [imgs.setdefault(n, f"Image.{len(imgs):03d}") for n in mat.images]
    for img in out.images:
        img.name = imgs.setdefault(img.name, f"Image.{len(imgs):03d}")
    for k in out.shape_keys:
        r.keys.setdefault(k.name, f"Key {len(r.keys)}")
        k.mesh = r.meshes[k.mesh]
        k.name = r.keys[k.name]
    return out, r


def rename_truth(truth: Truth, r: Renames) -> Truth:
    t = copy.deepcopy(truth)
    t.humanoid = {k: r.bones.get(v, v) for k, v in truth.humanoid.items()}
    t.expressions = {e: [(r.meshes[m], r.keys[k]) for m, k in b] for e, b in truth.expressions.items()}
    t.hair_vertices = {r.meshes[m]: v for m, v in truth.hair_vertices.items()}
    t.moved = {e: {r.meshes[m]: v for m, v in mv.items()} for e, mv in truth.moved.items()}
    return t


def strip_face_keys(ins: CharacterInspection, truth: Truth) -> CharacterInspection:
    out = copy.copy(ins)
    doomed = {k for e in FACE_KEYS for _, k in truth.expressions.get(e, [])}
    out.shape_keys = [k for k in ins.shape_keys if k.name not in doomed]
    return out


def _dist(a, b) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _landmark_errors(a: Analysis, t: Truth) -> dict[str, float | None]:
    lm = a.anatomy.landmarks if a.anatomy else {}
    face = a.anatomy.face if a.anatomy else None
    iod = t.interocular or 1.0
    out: dict[str, float | None] = {}

    def err(pred, true) -> float | None:
        return round(_dist(pred, true) / iod, 3) if pred is not None and true is not None else None

    fl = face.landmarks if face else {}
    out["mouth"] = err(lm["mouth"].position if "mouth" in lm else None, t.landmarks.get("mouth"))
    for s in ("left", "right"):
        out[f"mouth_{s}"] = err(fl[f"mouth_{s}"].position if f"mouth_{s}" in fl else None, t.landmarks.get(f"mouth_{s}"))
        up = fl.get(f"{s}_upper_eyelid")
        out[f"{s}_eyelid"] = err(up.position if up else None, t.landmarks.get(f"{s}_eyelid"))
        out[f"{s}_eye"] = err(lm[f"{s}_eye"].position if f"{s}_eye" in lm else None, t.landmarks.get(f"{s}_eye"))
    if face and face.mouth_width and t.mouth_width:
        out["mouth_width_ratio"] = round(face.mouth_width / t.mouth_width, 3)
    return out


def _humanoid_score(a: Analysis, t: Truth) -> dict:
    got = a.humanoid.as_map()
    ok = [b for b, s in t.humanoid.items() if got.get(b) == s]
    wrong = {b: (got[b], s) for b, s in t.humanoid.items() if b in got and got[b] != s}
    missing = [b for b in t.humanoid if b not in got]
    return {"correct": len(ok), "total": len(t.humanoid), "wrong": wrong, "missing": missing,
            "accuracy": round(len(ok) / max(len(t.humanoid), 1), 3)}


def _expression_score(a: Analysis, t: Truth) -> dict:
    out = {}
    for e in FACE_KEYS:
        want = {k for _, k in t.expressions.get(e, [])}
        if not want:
            continue
        m = a.expressions.mappings.get(e)
        got = {b.shape_key for b in m.binds} if m else set()
        out[e] = "ok" if got == want else ("missing" if not got else f"wrong {sorted(got)[:4]} != {sorted(want)[:4]}")
    return out


def _hair_score(a: Analysis, t: Truth, ins: CharacterInspection) -> dict | None:
    if not t.hair_vertices or a.anatomy is None:
        return None
    predicted = a.anatomy.hair.hair_vertex_sets(ins) if a.anatomy.hair else {}
    tp = sum(len(v & predicted.get(m, set())) for m, v in t.hair_vertices.items())
    pos = sum(len(v) for v in predicted.values())
    true = sum(len(v) for v in t.hair_vertices.values())
    face_hair = 0
    if a.anatomy.face:
        face_hair = sum(1 for p in a.anatomy.face.geometry.points
                        if p.role != "hair" and p.index in t.hair_vertices.get(p.mesh, ()))
    return {"precision": round(tp / pos, 3) if pos else 0.0, "recall": round(tp / true, 3) if true else 0.0,
            "hairCountedAsFace": face_hair}


def _generation_score(a: Analysis, t: Truth) -> dict:
    out: dict = {}
    plan = a.authoring
    if plan is None or plan.face is None:
        return {"error": "no authoring plan"}
    keys = {k.vrm_expression: k for k in plan.face.shape_keys if k.vrm_expression}
    issues = {i.plan: i.message for i in plan.face.issues}
    positions = {m.name: m.vertex_positions for m in a.inspection.meshes}
    iod = t.interocular or 1.0
    eye_region = {(m, i) for e in ("blinkLeft", "blinkRight") for m, vs in t.moved.get(e, {}).items() for i in vs}
    mouth_region = {(m, i) for e in VISEMES for m, vs in t.moved.get(e, {}).items() for i in vs}
    for e in (*VISEMES, "blinkLeft", "blinkRight"):
        truth_moved = {(m, i) for m, vs in t.moved.get(e, {}).items() for i in vs}
        if not truth_moved:
            continue
        k = keys.get(e)
        if k is None:
            skipped = [s.reason for s in plan.face.skipped if s.what == e]
            out[e] = {"generated": False, "reason": skipped[0] if skipped else "not planned"}
            continue
        mags = {(m, i): sum(c * c for c in d) ** 0.5 for m, vs in k.offsets.items() for i, d in vs.items()}
        biggest = max(mags.values(), default=0.0)
        moved = {x for x, v in mags.items() if v >= MOVED_FRACTION * biggest}
        inter = len(moved & truth_moved)
        union = len(moved | truth_moved)
        total = sum(mags.values()) or 1.0
        hair = sum(v for (m, i), v in mags.items() if i in t.hair_vertices.get(m, ()))
        wrong_region = eye_region if e in VISEMES else mouth_region
        leak = sum(v for x, v in mags.items() if x in wrong_region and x not in truth_moved)

        def centroid(s):
            pts = [positions[m][i] for m, i in s]
            return tuple(sum(p[j] for p in pts) / len(pts) for j in range(3)) if pts else None

        c1, c2 = centroid(moved), centroid(truth_moved)
        out[e] = {"generated": True, "valid": k.name not in issues, "issue": issues.get(k.name),
                  "iou": round(inter / union, 3) if union else 0.0,
                  "centroidError": round(_dist(c1, c2) / iod, 3) if c1 and c2 else None,
                  "hairLeak": round(hair / total, 3), "otherRegionLeak": round(leak / total, 3)}
    return out


def run_pass(ins: CharacterInspection, t: Truth, *, auto_author: bool = False) -> dict:
    cfg = ConversionConfig()
    if auto_author:
        cfg.auto_author = AutoAuthorConfig(enabled=True)
    a = analyze(ins, cfg)
    face = a.anatomy.face if a.anatomy else None
    res = {"landmarks": _landmark_errors(a, t), "humanoid": _humanoid_score(a, t),
           "expressions": _expression_score(a, t), "hair": _hair_score(a, t, ins),
           "face": {"quality": face.quality if face else None,
                    "points": len(face.geometry.points) if face else 0} if a.anatomy else None,
           "warnings": sorted({w.code for w in a.warnings})}
    if auto_author:
        res["generation"] = _generation_score(a, t)
    res["_analysis"] = a
    return res


def run_model(path: str, passes: tuple[str, ...] = ("named", "blind", "strip")) -> dict:
    gi: GltfInspection = inspect_glb(path)
    truth = read_truth(gi)
    out: dict = {"model": path, "truth": {"landmarks": sorted(truth.landmarks), "iod": round(truth.interocular, 4),
                                           "mouthWidth": round(truth.mouth_width, 4),
                                           "expressions": sorted(truth.expressions), "notes": truth.notes}}
    ins = gi.inspection
    stale = {k.name for k in ins.shape_keys if k.name.startswith("Prosopon_")}
    if stale:
        # keys generated by an earlier conversion of this file are not source evidence
        ins.shape_keys = [k for k in ins.shape_keys if k.name not in stale]
        out["truth"]["notes"].append(f"ignored {len(stale)} previously generated Prosopon_ keys")
    if "named" in passes:
        out["named"] = run_pass(ins, truth)
    if "blind" in passes:
        blind, r = obfuscate(ins)
        out["blind"] = run_pass(blind, rename_truth(truth, r))
    if "strip" in passes:
        out["strip"] = run_pass(strip_face_keys(ins, truth), truth, auto_author=True)
    return out
