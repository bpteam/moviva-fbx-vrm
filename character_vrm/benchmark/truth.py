"""Ground truth read back from an authored VRM (0.x or 1.0).

A VRM already says what the analyzer has to find: the humanoid map, the eye bones, the morph targets that *are* the
visemes and the blink, and (by material) which surface is hair. Face landmarks are derived from what those morphs
move, so they are measured on the asset, not assumed:

- lip seam: along the mouth, the height where the downward displacement of ``aa`` drops sharply going up (the
  lower lip moves, the upper lip does not) — ``face.step_split``;
- mouth centre: the front-most surface at the seam, laterally at the centre of the viseme region;
- mouth corners: lateral extent of the lips (strongly moved viseme vertices) at the seam height;
- upper eyelids: the displacement-weighted centroid of what each half-blink pulls down (world Z is up, as in
  Blender).

A sidecar ``<model>.truth.json`` may name the morphs for models whose VRM presets are not authored truth (for
example a converter output): ``{"expressions": {"aa": ["V_Open"], ...}, "hairMaterials": ["Hair"]}``, and
``"mouthTruth": "jaw"`` takes the lip seam from the jaw skinning for rigs whose mouth morphs are correctives on top of
a jaw bone (Character Creator).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..core.gltf_inspect import GltfInspection
from ..core.model import Vec3

VISEMES = ("aa", "ih", "ou", "ee", "oh")
FACE_KEYS = (*VISEMES, "blink", "blinkLeft", "blinkRight")
_VRM0_PRESETS = {"a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh", "blink": "blink", "blink_l": "blinkLeft",
                 "blink_r": "blinkRight", "joy": "happy", "angry": "angry", "sorrow": "sad", "fun": "relaxed",
                 "surprised": "surprised"}
_VRM0_BONES = {"leftThumbProximal": "leftThumbMetacarpal", "leftThumbIntermediate": "leftThumbProximal",
               "rightThumbProximal": "rightThumbMetacarpal", "rightThumbIntermediate": "rightThumbProximal"}
HAIR_TOKENS = ("hair", "scalp")
MOVED_FRACTION = 0.15  # a vertex "moves" in a key when it moves at least this share of the key's largest offset


@dataclass
class Truth:
    humanoid: dict[str, str] = field(default_factory=dict)            # vrm bone -> source bone
    expressions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # vrm expr -> [(mesh, key)]
    landmarks: dict[str, Vec3] = field(default_factory=dict)          # world (Blender) space
    mouth_width: float = 0.0
    interocular: float = 0.0
    hair_vertices: dict[str, set[int]] = field(default_factory=dict)  # mesh -> vertex indices
    moved: dict[str, dict[str, set[int]]] = field(default_factory=dict)  # expr -> mesh -> moved vertices
    notes: list[str] = field(default_factory=list)
    mouth_source: str = "morph"


def _ext(gltf: dict) -> tuple[str, dict]:
    ext = gltf.get("extensions", {})
    if "VRMC_vrm" in ext:
        return "1.0", ext["VRMC_vrm"]
    if "VRM" in ext:
        return "0.x", ext["VRM"]
    return "", {}


def _sidecar(path: str) -> dict:
    p = Path(path + ".truth.json") if not path.endswith(".truth.json") else Path(path)
    alt = Path(path).with_suffix(".truth.json")
    for c in (p, alt):
        if c.is_file():
            return json.loads(c.read_text())
    return {}


def read_truth(gi: GltfInspection) -> Truth:
    doc, ins = gi.document, gi.inspection
    g = doc.json
    version, vrm = _ext(g)
    truth = Truth()
    names = gi.node_names
    side = _sidecar(doc.path)
    truth.mouth_source = side.get("mouthTruth", "morph")

    # --- humanoid
    if version == "1.0":
        for bone, spec in vrm.get("humanoid", {}).get("humanBones", {}).items():
            truth.humanoid[bone] = names[spec["node"]]
    elif version == "0.x":
        for spec in vrm.get("humanoid", {}).get("humanBones", []):
            bone = _VRM0_BONES.get(spec["bone"], spec["bone"])
            truth.humanoid[bone] = names[spec["node"]]
    # VRM 0.x thumbs: proximal/intermediate/distal -> metacarpal/proximal/distal (VRM 1.0 naming)

    # --- expressions
    keys_by_mesh: dict[str, list[str]] = {gi.mesh_object[mi]: gi.mesh_targets[mi] for mi in gi.mesh_object}
    if "expressions" in side:
        for expr, key_names in side["expressions"].items():
            truth.expressions[expr] = [(m, k) for m, ks in keys_by_mesh.items() for k in ks if k in key_names]
    elif version == "1.0":
        node_mesh = {i: n.get("mesh") for i, n in enumerate(g.get("nodes", []))}
        for expr, spec in vrm.get("expressions", {}).get("preset", {}).items():
            binds = []
            for b in spec.get("morphTargetBinds", []):
                mi = node_mesh.get(b["node"])
                if mi in gi.mesh_object and b.get("weight", 1.0) > 0:
                    binds.append((gi.mesh_object[mi], gi.mesh_targets[mi][b["index"]]))
            if binds:
                truth.expressions[expr] = binds
    elif version == "0.x":
        for group in vrm.get("blendShapeMaster", {}).get("blendShapeGroups", []):
            expr = _VRM0_PRESETS.get(group.get("presetName", ""))
            if not expr:
                continue
            binds = [(gi.mesh_object[b["mesh"]], gi.mesh_targets[b["mesh"]][b["index"]])
                     for b in group.get("binds", []) if b["mesh"] in gi.mesh_object and b.get("weight", 100) > 0]
            if binds:
                truth.expressions[expr] = binds

    # --- hair by material
    hair_tokens = tuple(t.lower() for t in side.get("hairMaterials", HAIR_TOKENS))
    for mesh in ins.meshes:
        hair_slots = {i for i, m in enumerate(mesh.materials) if any(t in m.lower() for t in hair_tokens)}
        if not hair_slots:
            continue
        verts: set[int] = set()
        for poly, slot in zip(mesh.polygons, mesh.polygon_materials):
            if slot in hair_slots:
                verts.update(poly)
        if verts:
            truth.hair_vertices[mesh.name] = verts

    # --- morph-derived landmarks
    delta = {(k.mesh, k.name): k.deltas for k in ins.shape_keys}
    positions = {m.name: m.vertex_positions for m in ins.meshes}
    for expr in FACE_KEYS:
        moved: dict[str, set[int]] = {}
        for mesh, key in truth.expressions.get(expr, []):
            d = delta.get((mesh, key), {})
            if not d:
                continue
            biggest = max(_len(v) for v in d.values())
            moved.setdefault(mesh, set()).update(i for i, v in d.items() if _len(v) >= MOVED_FRACTION * biggest)
        if moved:
            truth.moved[expr] = moved

    joints = {b.name: b.head for a in ins.armatures for b in a.bones}
    for s in ("left", "right"):
        bone = truth.humanoid.get(f"{s}Eye")
        # The eye centre is the geometry the eye bone carries (VRoid places the bone well behind and inside the
        # eyeball, so the joint itself is not the eye); the joint is only the fallback.
        eye_pts = [m.vertex_positions[i] for m in ins.meshes for i, w in m.vertex_weights.items()
                   if bone and w.get(bone, 0.0) >= 0.5 and not _hair(truth, m.name, i)] if bone else []
        if len(eye_pts) >= 8:
            truth.landmarks[f"{s}_eye"] = _mean(eye_pts)
        elif bone in joints:
            truth.landmarks[f"{s}_eye"] = joints[bone]
        key = "blinkLeft" if s == "left" else "blinkRight"
        # the upper lid: what the half-blink pulls down (weighted by how far)
        pts, ws = [], []
        for mesh, key_name in truth.expressions.get(key, []):
            for i, d in delta.get((mesh, key_name), {}).items():
                if i in truth.moved.get(key, {}).get(mesh, ()) and not _hair(truth, mesh, i) and d[2] < 0:
                    pts.append(positions[mesh][i])
                    ws.append(-d[2])
        if pts:
            tw = sum(ws)
            truth.landmarks[f"{s}_eyelid"] = tuple(sum(p[j] * w for p, w in zip(pts, ws)) / tw for j in range(3))
    if "left_eyelid" in truth.landmarks and "right_eyelid" in truth.landmarks:
        truth.interocular = _dist(truth.landmarks["left_eyelid"], truth.landmarks["right_eyelid"])
    elif "left_eye" in truth.landmarks and "right_eye" in truth.landmarks:
        truth.interocular = _dist(truth.landmarks["left_eye"], truth.landmarks["right_eye"])
    _mouth_truth(truth, ins, positions, delta, joints)
    corner_keys = side.get("cornerKeys")
    if corner_keys and "mouth" in truth.landmarks:
        # a one-sided smile pulls the corner of its side most: its most displaced vertices are the corner
        for side_name, key_name in zip(("mouth_left", "mouth_right"), corner_keys):
            for (mesh, name), d in delta.items():
                if name != key_name or not d:
                    continue
                top = sorted(d.items(), key=lambda kv: -_len(kv[1]))[:8]
                truth.landmarks[side_name] = _mean([positions[mesh][i] for i, _ in top])
                break
        if "mouth_left" in truth.landmarks and "mouth_right" in truth.landmarks:
            truth.mouth_width = _dist(truth.landmarks["mouth_left"], truth.landmarks["mouth_right"])
    return truth


def _len(v: Vec3) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def _dist(a: Vec3, b: Vec3) -> float:
    return _len((a[0] - b[0], a[1] - b[1], a[2] - b[2]))


def _mean(pts: list[Vec3]) -> Vec3:
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def _hair(truth: Truth, mesh: str, i: int) -> bool:
    return i in truth.hair_vertices.get(mesh, ())


def _jaw_seam_truth(truth: Truth, ins, joints) -> None:
    """For jaw-rigged models whose morphs are correctives on top of the jaw bone (CC): the lip seam is where the
    jaw's skin weight ends on the midline (lower lip on the jaw, upper lip not)."""
    from ..core.anatomy.face import step_split
    jaw = truth.humanoid.get("jaw")
    arm = ins.armatures[0]
    if not jaw:
        truth.notes.append("mouthTruth=jaw but no jaw bone")
        return
    jaw_bones = {jaw, *arm.descendants(jaw)}
    head = truth.humanoid.get("head")
    head_bones = {head, *arm.descendants(head)}
    hips, hj = joints.get(truth.humanoid.get("hips", "")), joints.get(head)
    le, re_ = truth.landmarks.get("left_eyelid"), truth.landmarks.get("right_eyelid")
    up = _norm(_sub(hj, hips))
    left = _norm(_ortho(_sub(le, re_), up))
    fwd = _norm(_cross(left, up))
    eye_h = _dot(le, up)
    pts = []
    for m in ins.meshes:
        slots = {i for i, mat in enumerate(m.materials) if "skin" in mat.lower() or "head" in mat.lower()}
        verts = {v for poly, s in zip(m.polygons, m.polygon_materials) if s in slots for v in poly}
        for i in verts:
            ws = m.vertex_weights.get(i, {})
            if sum(w for b, w in ws.items() if b in head_bones) < 0.5:
                continue
            p = m.vertex_positions[i]
            if eye_h - 0.12 < _dot(p, up) < eye_h - 0.02:
                pts.append((p, sum(w for b, w in ws.items() if b in jaw_bones)))
    zf = max(_dot(p, fwd) for p, _ in pts)
    front = [(p, w) for p, w in pts if _dot(p, fwd) > zf - 0.03]
    mid = [(_dot(p, up), w) for p, w in front if abs(_dot(p, left)) < 0.004]
    split = step_split(mid)
    if split is None:
        truth.notes.append("no jaw seam")
        return
    seam = split[0]
    row = sorted(abs(_dot(p, left)) for p, w in front if abs(_dot(p, up) - seam) < 0.006 and 0.15 < w < 0.85)
    corner = row[int(0.9 * (len(row) - 1))] if row else 0.0
    z = max(_dot(p, fwd) for p, _ in front if abs(_dot(p, up) - seam) < 0.004 and abs(_dot(p, left)) < 0.004)
    centre = _add(_scale(up, seam), _scale(fwd, z))
    truth.mouth_width = 2 * corner
    truth.landmarks["mouth"] = centre
    truth.landmarks["mouth_left"] = _add(centre, _scale(left, corner))
    truth.landmarks["mouth_right"] = _add(centre, _scale(left, -corner))
    truth.notes.append("mouth truth from the jaw skinning (sidecar mouthTruth=jaw)")


def _mouth_truth(truth: Truth, ins, positions, delta, joints) -> None:
    if truth.mouth_source == "jaw":
        _jaw_seam_truth(truth, ins, joints)
        return
    aa = truth.expressions.get("aa")
    if not aa or "aa" not in truth.moved:
        truth.notes.append("no aa morph: no mouth truth")
        return
    # Face axes from the humanoid: up = hips -> head, left from the eyes / upper arms, forward = left x up.
    hips, head = joints.get(truth.humanoid.get("hips", "")), joints.get(truth.humanoid.get("head", ""))
    le, re_ = truth.landmarks.get("left_eyelid"), truth.landmarks.get("right_eyelid")
    if le is None or re_ is None:
        la, ra = joints.get(truth.humanoid.get("leftUpperArm", "")), joints.get(truth.humanoid.get("rightUpperArm", ""))
        le, re_ = la, ra
    if hips is None or head is None or le is None or re_ is None:
        truth.notes.append("no body axes: no mouth truth")
        return
    up = _norm(_sub(head, hips))
    left = _norm(_ortho(_sub(le, re_), up))
    fwd = _norm(_cross(left, up))
    union: dict[tuple[str, int], float] = {}
    for expr in VISEMES:
        for mesh, vs in truth.moved.get(expr, {}).items():
            for i in vs:
                if not _hair(truth, mesh, i):
                    union[(mesh, i)] = union.get((mesh, i), 0.0) + 1.0
    if not union:
        return
    pts = {k: positions[k[0]][k[1]] for k in union}
    cx = sum(_dot(p, left) for p in pts.values()) / len(pts)
    # the lip region: front-most part of the viseme region (the inner mouth and teeth sit behind it)
    zs = sorted(_dot(p, fwd) for p in pts.values())
    z_front = zs[int(0.9 * (len(zs) - 1))]
    depth = zs[-1] - zs[0]
    aa_d: dict[tuple[str, int], Vec3] = {}
    for mesh, key in aa:
        for i, d in delta.get((mesh, key), {}).items():
            o = aa_d.get((mesh, i), (0.0, 0.0, 0.0))
            aa_d[(mesh, i)] = (o[0] + d[0], o[1] + d[1], o[2] + d[2])
    lips = [(k, p) for k, p in pts.items() if _dot(p, fwd) >= z_front - 0.45 * depth]
    widths = sorted(abs(_dot(p, left) - cx) for _, p in lips)
    half = widths[int(0.95 * (len(widths) - 1))] if widths else 0.0
    band = [(_dot(p, up), -_dot(aa_d.get(k, (0.0, 0.0, 0.0)), up)) for k, p in lips if abs(_dot(p, left) - cx) < 0.35 * half]
    if len(band) < 6:
        truth.notes.append("too few lip vertices on the midline")
        return
    from ..core.anatomy.face import step_split
    split = step_split(band)
    if split is None:
        truth.notes.append("aa does not open the lips")
        return
    seam = split[0]
    near = [p for _, p in lips if abs(_dot(p, left) - cx) < 0.35 * half and abs(_dot(p, up) - seam) < 0.3 * half]
    z = max((_dot(p, fwd) for p in near), default=z_front)
    at_seam = [abs(_dot(p, left) - cx) for _, p in lips if abs(_dot(p, up) - seam) < 0.25 * half]
    at_seam.sort()
    corner = at_seam[int(0.97 * (len(at_seam) - 1))] if at_seam else half
    truth.mouth_width = 2 * corner
    centre = _add(_add(_scale(left, cx), _scale(up, seam)), _scale(fwd, z))
    truth.landmarks["mouth"] = centre
    truth.landmarks["mouth_left"] = _add(centre, _scale(left, corner))
    truth.landmarks["mouth_right"] = _add(centre, _scale(left, -corner))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = _len(a) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _ortho(v, axis):
    return _sub(v, _scale(axis, _dot(v, axis)))
