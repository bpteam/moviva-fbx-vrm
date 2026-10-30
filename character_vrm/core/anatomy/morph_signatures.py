"""Existing shape keys recognised by what they do to the face, not by their names (pure).

For keys whose names say nothing (``Key 17``, ``Shape.004``), every key's deltas are measured against the
analysed face: how much of its motion lands on each eye and on the mouth, how far it closes an eyelid, opens the lips,
spreads or purses the corners, pushes the lips forward or lifts the corners. Blinks are the keys that close one eye
(or both) and leave the mouth alone; visemes are the symmetric mouth-only keys, assigned to ``aa / ih / ee / ou / oh``
by the closest prototype. A key is only proposed when it is clearly that kind of motion; everything else stays
unmapped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations

from ..model import CharacterInspection
from .face import FaceAnalysis
from .frame import AnatomicalFrame
from .mouth import step_split
from .surfaces import LASH, SKIN

# (open, spread, protrude, corner lift), each a fraction of the mouth width. Measured on the VRoid A/I/U/E/O keys of
# the benchmark models (``character_vrm.benchmark``); a set is only assigned when some key really opens the mouth
# (lip correctives that expect a jaw bone, such as Character Creator's, open nothing and stay unassigned).
VISEME_PROTOTYPES: dict[str, tuple[float, float, float, float]] = {
    "aa": (0.34, 0.04, -0.06, 0.0),
    "ih": (0.025, 0.07, -0.04, 0.01),
    "ee": (0.17, 0.09, -0.06, 0.03),
    "ou": (0.13, -0.17, 0.03, -0.05),
    "oh": (0.41, -0.015, -0.05, -0.07),
}
MIN_OPENING = 0.15  # an ``aa`` candidate must open the lips by this share of the mouth width
MIN_CONFIDENCE = 0.7
PROSOPON_KEY_PREFIX = "Prosopon_"  # keys a previous conversion generated are never source evidence


@dataclass
class KeySignature:
    key: str
    energy: dict[str, float] = field(default_factory=dict)  # region -> share of the key's motion
    closure: dict[str, float] = field(default_factory=dict)  # side -> upper-lid travel down / eye height
    mouth: tuple[float, float, float, float] | None = None
    symmetry: float = 0.0


@dataclass
class MorphProposal:
    expression: str
    key: str
    confidence: float
    detail: str


_GENERIC_TOKENS = frozenset({"key", "keys", "shape", "shapes", "morph", "morphs", "blend", "blendshape", "target",
                             "targets", "basis", "sk", "bs", "mesh", "object", "obj", "new"})


def opaque_name(name: str) -> bool:
    """True for names that say nothing (``Key 17``, ``Shape.004``, ``blendShape1.target_12``). A key with a
    meaningful name (``Fcl_MTH_Joy``, ``Smile``) already means something: motion must never turn an emotion into a
    viseme just because it opens the mouth too."""
    from ..expressions import _tokens, strip_namespace
    words = [t for t in _tokens(strip_namespace(name)) if t.isalpha() and len(t) >= 2 and t not in _GENERIC_TOKENS]
    return not words


def front_surface(face: FaceAnalysis, cell: float) -> dict[tuple[int, int], float]:
    """Front-most depth per (x, y) cell of the face skin: teeth, tongue and the mouth bag lie behind it."""
    out: dict[tuple[int, int], float] = {}
    for p in face.geometry.points:
        if p.role not in (SKIN, LASH):
            continue
        k = (round(p.local[0] / cell), round(p.local[1] / cell))
        if k not in out or p.local[2] > out[k]:
            out[k] = p.local[2]
    return out


def _on_front(p, front: dict[tuple[int, int], float], cell: float, tol: float) -> bool:
    k = (round(p.local[0] / cell), round(p.local[1] / cell))
    z = max((front.get((k[0] + i, k[1] + j), -1e9) for i in (-1, 0, 1) for j in (-1, 0, 1)), default=-1e9)
    return p.local[2] >= z - tol


def _signature(name: str, deltas: dict[tuple[str, int], tuple], by_key, face: FaceAnalysis,
               frame: AnatomicalFrame, front: dict[tuple[int, int], float] | None = None,
               cell: float = 0.0) -> KeySignature | None:
    fl = {n: face.local(frame, n) for n in ("mouth_center", "left_eye", "right_eye")}
    m = fl["mouth_center"]
    if m is None:
        return None
    half = max(face.mouth_width / 2, 1e-9)
    sig = KeySignature(name)
    total = peak = 0.0
    region = {"mouth": 0.0, "left_eye": 0.0, "right_eye": 0.0, "other": 0.0}
    central: list[tuple[float, float]] = []
    corners, lifts, fronts = [], [], []
    lid: dict[str, list[float]] = {"left": [], "right": []}
    left_e = right_e = 0.0
    for k, d in deltas.items():
        p = by_key.get(k)
        if p is None or p.role not in (SKIN, LASH):
            continue
        if front is not None and not _on_front(p, front, cell, 0.12 * half):
            continue  # inside the mouth (teeth, tongue) when the materials did not say so
        dl = frame.vector_to_local(d)
        mag = math.sqrt(dl[0] ** 2 + dl[1] ** 2 + dl[2] ** 2)
        if mag <= 0:
            continue
        total += mag
        peak = max(peak, mag)
        x, y, z = p.local
        near_eye = None
        for s in ("left", "right"):
            e = fl[f"{s}_eye"]
            box = face.eye_extent.get(s)
            rx = max(0.5 * (box[1] - box[0]) if box else 0.0, 0.08 * face.head_width) * 1.4
            if e is not None and abs(x - e[0]) < rx and -0.6 * rx < y - e[1] < 1.2 * rx:
                near_eye = s
        if near_eye:
            region[f"{near_eye}_eye"] += mag
            if y > fl[f"{near_eye}_eye"][1]:
                lid[near_eye].append(-dl[1])
        elif abs(x - m[0]) < 2.2 * half and -1.6 * half < y - m[1] < 1.2 * half:
            region["mouth"] += mag
            if x > m[0]:
                left_e += mag
            else:
                right_e += mag
            if abs(x - m[0]) < 0.4 * half:
                central.append((y, -dl[1]))
                fronts.append(dl[2])
            elif 0.7 * half < abs(x - m[0]) < 1.4 * half and abs(y - m[1]) < 0.5 * half:
                corners.append(dl[0] * (1.0 if x > m[0] else -1.0))
                lifts.append(dl[1])
        else:
            region["other"] += mag
    if total <= 0 or peak < 0.05 * 2 * half:
        return None  # a key that barely moves the face (a "neutral" / corrective) says nothing about its purpose
    sig.energy = {k: v / total for k, v in region.items()}
    for s in ("left", "right"):
        box = face.eye_extent.get(s)
        eh = (box[3] - box[2]) if box else 0.05 * face.head_height
        vals = lid[s]
        sig.closure[s] = (sum(sorted(vals)[-max(1, len(vals) // 4):]) / max(1, len(vals) // 4)) / max(eh, 1e-9) \
            if vals else 0.0
    if central or corners:
        mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
        w = 2 * half
        # the opening is the key's own sharpest drop of downward travel along the midline (lower lip vs upper lip),
        # independent of where the analysis put the seam
        split = step_split(central) if len(central) >= 8 else None
        opening = split[1] if split is not None else 0.0
        sig.mouth = (opening / w, mean(corners) / w, mean(fronts) / w, mean(lifts) / w)
    sig.symmetry = min(left_e, right_e) / max(left_e, right_e) if max(left_e, right_e) > 0 else 0.0
    return sig


def classify_morphs(inspection: CharacterInspection, face: FaceAnalysis, frame: AnatomicalFrame,
                    wanted: set[str], taken: set[str]) -> list[MorphProposal]:
    """Proposals for the ``wanted`` expressions among keys not in ``taken`` (names are never read)."""
    by_key = {(p.mesh, p.index): p for p in face.geometry.points}
    per_key: dict[str, dict[tuple[str, int], tuple]] = {}
    for k in inspection.shape_keys:
        if k.name in taken or not k.deltas or k.name.startswith(PROSOPON_KEY_PREFIX) or not opaque_name(k.name):
            continue
        acc = per_key.setdefault(k.name, {})
        for i, d in k.deltas.items():
            acc[(k.mesh, i)] = d
    cell = max(0.08 * face.mouth_width, 1e-6)
    front = front_surface(face, cell)
    sigs = [s for s in (_signature(n, d, by_key, face, frame, front, cell) for n, d in per_key.items()) if s is not None]
    out: list[MorphProposal] = []

    # --- blinks: one eye closes, the rest of the face stays
    for expr, side, other in (("blinkLeft", "left", "right"), ("blinkRight", "right", "left")):
        if expr not in wanted:
            continue
        cands = [(s.closure[side] * s.energy[f"{side}_eye"], s) for s in sigs
                 if s.energy[f"{side}_eye"] >= 0.45 and s.energy[f"{other}_eye"] < 0.1 and s.energy["mouth"] < 0.1
                 and s.closure[side] > 0.25]
        if cands:
            cands.sort(key=lambda c: -c[0])
            best = cands[0][1]
            margin = 1.0 if len(cands) == 1 else 1.0 - cands[1][0] / max(cands[0][0], 1e-9)
            conf = round(min(0.9, 0.7 + 0.2 * min(1.0, best.closure[side]) + 0.1 * margin), 3)
            out.append(MorphProposal(expr, best.key, conf,
                                     f"closes the {side} lid ({best.closure[side]:.2f} of the eye height)"))
    if "blink" in wanted:
        cands = [(min(s.closure["left"], s.closure["right"]), s) for s in sigs
                 if s.energy["left_eye"] >= 0.25 and s.energy["right_eye"] >= 0.25 and s.energy["mouth"] < 0.1
                 and min(s.closure["left"], s.closure["right"]) > 0.25]
        if cands:
            cands.sort(key=lambda c: -c[0])
            best = cands[0][1]
            conf = round(min(0.9, 0.7 + 0.2 * min(1.0, cands[0][0])), 3)
            out.append(MorphProposal("blink", best.key, conf, "closes both lids"))

    # --- visemes: symmetric, mouth-only keys, assigned jointly to the nearest prototypes
    vis = [v for v in VISEME_PROTOTYPES if v in wanted]
    mouth_keys = [s for s in sigs if s.mouth is not None and s.energy["mouth"] >= 0.75 and s.symmetry >= 0.6
                  and s.mouth[0] > -0.02]
    order = {name: i for i, name in enumerate(dict.fromkeys(k.name for k in inspection.shape_keys))}
    if vis and mouth_keys and max(s.mouth[0] for s in mouth_keys) >= MIN_OPENING:  # type: ignore[index]
        def dist(s: KeySignature, v: str) -> float:
            p = VISEME_PROTOTYPES[v]
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(s.mouth, p)))  # type: ignore[arg-type]
        # keep the few nearest keys per viseme, then solve the small assignment exactly
        pool = sorted({s.key: s for v in vis for s in sorted(mouth_keys, key=lambda s: dist(s, v))[:3]}.values(),
                      key=lambda s: s.key)
        best_total, best_assign = None, None
        n = min(len(vis), len(pool))
        for keys in permutations(pool, n):
            # viseme sets are authored together: keys far apart in the key list are less likely one set
            idx = [order.get(k.key, 0) for k in keys]
            total = sum(dist(k, v) for k, v in zip(keys, vis[:n])) + 0.03 * min(10, max(idx) - min(idx) - (n - 1))
            if best_total is None or total < best_total:
                best_total, best_assign = total, keys
        for v, s in zip(vis[:n], best_assign or ()):
            d = dist(s, v)
            conf = round(max(0.0, 1.0 - d / 0.5), 3)
            if conf >= MIN_CONFIDENCE:
                o, sp, pr, li = s.mouth  # type: ignore[misc]
                out.append(MorphProposal(v, s.key, conf,
                                         f"mouth motion open {o:.2f}, spread {sp:.2f}, forward {pr:.2f}, lift {li:.2f}"))
    return out
