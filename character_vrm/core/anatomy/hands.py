"""Phase 6: hand analysis.

Existing finger bones are classified by chain geometry (never by name): five chains under the hand bone, the thumb is
the chain that deviates most from the direction the other four share, the remaining chains are ordered from the thumb
side. Fingers missing as bones are detected as a *capability condition*; whether the hand geometry can support a
generated rig is decided by ``segment_fingers`` and reported honestly (``fused`` means a mitten).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import ArmatureInfo
from .confidence import AnalysisSource, AnalysisWarning, fuse
from .frame import AnatomicalFrame
from .vec import (
    Vec3, angle_between, cross, dist, dot, mean, normalize, orthogonalize, principal_axis, scale, sub,
)

FINGER_ORDER = ("Thumb", "Index", "Middle", "Ring", "Little")

# Capability states.
BONES_PRESENT = "bones-present"
GEOMETRY_SEPARABLE = "finger-bones-missing-geometry-separable"
FUSED = "fused-fingers"
NO_EVIDENCE = "no-finger-evidence"


@dataclass
class FingerChain:
    finger: str            # Thumb | Index | Middle | Ring | Little
    bones: list[str]       # source bones, root first (tip bone already dropped)
    vrm_bones: dict[str, str]  # vrm bone name -> source bone
    confidence: float
    evidence: dict[str, float]
    source: AnalysisSource = AnalysisSource.SKELETON


@dataclass
class FingerSegmentation:
    """Result of separating finger regions in hand geometry (no bones needed)."""

    clusters: dict[str, list[int]] = field(default_factory=dict)  # finger -> vertex indices (of the analysed set)
    confidence: float = 0.0
    fused: bool = False
    runs_found: int = 0
    reason: str = ""
    along: Vec3 | None = None
    across: Vec3 | None = None
    origin: Vec3 | None = None
    hand_length: float = 0.0
    points: list[Vec3] = field(default_factory=list)
    sources: list[tuple[str, int]] = field(default_factory=list)  # (mesh, vertex) per analysed point


@dataclass
class HandAnalysis:
    side: str
    hand_bone: str | None
    palm_center: Vec3 | None
    state: str = NO_EVIDENCE
    fingers: dict[str, FingerChain] = field(default_factory=dict)
    segmentation: FingerSegmentation | None = None
    confidence: float = 0.0
    warnings: list[AnalysisWarning] = field(default_factory=list)

    @property
    def independent_articulation(self) -> bool:
        return self.state == BONES_PRESENT

    def vrm_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for chain in self.fingers.values():
            out.update(chain.vrm_bones)
        return out


def _chain(armature: ArmatureInfo, root: str) -> list[str]:
    path = [root]
    while True:
        kids = [c for c in armature.children(path[-1]) if armature.bone(c) and armature.bone(c).deform]
        if not kids:
            break
        path.append(max(kids, key=lambda k: (armature.subtree_size(k), armature.bone(k).length)))
    if len(path) >= 3:
        last, prev = armature.bone(path[-1]), armature.bone(path[-2])
        if last and prev and last.length < 0.7 * prev.length and len(armature.children(path[-1])) == 0 and len(path) >= 4:
            path = path[:-1]  # tip / end-site bone
    return path


def _direction(armature: ArmatureInfo, chain: list[str]) -> Vec3 | None:
    a = armature.bone(chain[0])
    b = armature.bone(chain[-1])
    return normalize(sub(b.tail, a.head)) if a and b else None


def classify_finger_chains(armature: ArmatureInfo, hand_bone: str, side: str, weighted: set[str],
                           frame: AnatomicalFrame | None = None) -> HandAnalysis:
    hand = armature.bone(hand_bone)
    analysis = HandAnalysis(side=side, hand_bone=hand_bone, palm_center=None)
    if hand is None:
        return analysis
    roots = [c for c in armature.children(hand_bone) if armature.bone(c) and armature.bone(c).deform]
    chains = [_chain(armature, r) for r in roots]
    chains = [c for c in chains if len(c) >= 2 or len(armature.children(c[0])) == 0 and len(c) >= 1]
    chains = [c for c in chains if len(c) >= 2]
    if len(chains) < 2:
        return analysis
    dirs = [_direction(armature, c) for c in chains]
    if any(d is None for d in dirs):
        return analysis
    roots_pos = [armature.bone(c[0]).head for c in chains]
    tips = [armature.bone(c[-1]).tail for c in chains]
    analysis.palm_center = mean([hand.head, *roots_pos])

    # Thumb: largest deviation from the mean direction of all the others.
    def others_mean(i: int) -> Vec3:
        m = mean([d for j, d in enumerate(dirs) if j != i and d is not None])
        return normalize(m) or dirs[i]  # type: ignore[return-value]

    deviation = [angle_between(dirs[i], others_mean(i)) for i in range(len(chains))]  # type: ignore[arg-type]
    thumb_i = max(range(len(chains)), key=lambda i: (deviation[i], -armature.bone(chains[i][0]).length))
    ordered_rest = [i for i in range(len(chains)) if i != thumb_i]
    axis_pts = [roots_pos[i] for i in ordered_rest]
    pa = principal_axis(axis_pts) if len(axis_pts) >= 3 else None
    across = pa[0] if pa else normalize(sub(axis_pts[-1], axis_pts[0]))
    if across is None:
        return analysis
    ordered_rest.sort(key=lambda i: dot(roots_pos[i], across))
    if dist(roots_pos[ordered_rest[0]], roots_pos[thumb_i]) > dist(roots_pos[ordered_rest[-1]], roots_pos[thumb_i]):
        ordered_rest.reverse()  # index is the chain nearest the thumb

    count_score = {5: 0.95, 4: 0.7}.get(len(chains), 0.45)
    others_dev = sorted(deviation[i] for i in ordered_rest)
    gap = deviation[thumb_i] - (others_dev[-1] if others_dev else 0.0)
    thumb_clarity = max(0.0, min(1.0, gap / 25.0 + 0.4))
    names = ["Thumb", *FINGER_ORDER[1:1 + len(ordered_rest)]]
    if len(chains) == 5:
        idxs = [thumb_i, *ordered_rest]
    else:
        idxs = [thumb_i, *ordered_rest]
        names = ["Thumb", *FINGER_ORDER[1:][:len(ordered_rest)]]
    for finger, i in zip(names, idxs):
        chain = chains[i]
        vrm = _vrm_names(side, finger, chain)
        skin = 1.0 if weighted and all(b in weighted for b in chain[:3]) else (0.4 if weighted else 0.6)
        ev = {"topology": count_score, "direction": thumb_clarity if finger == "Thumb" else 0.85, "skinning": skin}
        analysis.fingers[finger] = FingerChain(finger, chain, vrm, fuse(ev, {"topology": .4, "direction": .35, "skinning": .25}), ev)
    analysis.state = BONES_PRESENT
    analysis.confidence = sum(c.confidence for c in analysis.fingers.values()) / len(analysis.fingers)
    if len(chains) < 5:
        analysis.warnings.append(AnalysisWarning(
            "fingers-partial", f"{side} hand has {len(chains)} finger chains instead of 5", "hands"))
    return analysis


def _vrm_names(side: str, finger: str, chain: list[str]) -> dict[str, str]:
    parts = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else ("Proximal", "Intermediate", "Distal")
    if finger != "Thumb" and len(chain) >= 4:
        chain = chain[-3:]  # a leading metacarpal has no VRM slot for the four fingers
    if finger == "Thumb" and len(chain) == 2:
        parts = ("Proximal", "Distal")
    elif len(chain) == 2 and finger != "Thumb":
        parts = ("Proximal", "Distal")
    return {f"{side}{finger}{p}": b for p, b in zip(parts, chain[:3])}


def compare_hands(left: HandAnalysis, right: HandAnalysis, frame: AnatomicalFrame, armature: ArmatureInfo) -> float:
    """Symmetry between the hands: the thumb must sit on the mirrored side. Returns a score and adjusts confidences."""
    def thumb_side(h: HandAnalysis) -> float | None:
        thumb, little = h.fingers.get("Thumb"), h.fingers.get("Little")
        if not thumb or not little:
            return None
        t = frame.to_local(armature.bone(thumb.bones[0]).head)
        l = frame.to_local(armature.bone(little.bones[0]).head)
        return t[2] - l[2]  # forward difference; a mirrored hand gives the same sign
    a, b = thumb_side(left), thumb_side(right)
    if a is None or b is None:
        return 0.5
    score = 1.0 if (a >= 0) == (b >= 0) or min(abs(a), abs(b)) < 1e-4 else 0.0
    for h in (left, right):
        for chain in h.fingers.values():
            chain.evidence["symmetry"] = score
            chain.confidence = fuse(chain.evidence, {"topology": .3, "direction": .3, "skinning": .2, "symmetry": .2})
        if h.fingers:
            h.confidence = sum(c.confidence for c in h.fingers.values()) / len(h.fingers)
    if score < 0.5:
        for h in (left, right):
            h.warnings.append(AnalysisWarning(
                "fingers-asymmetric", "thumbs are not on mirrored sides of the two hands", "hands"))
    return score


def _rotate_about(v: Vec3, axis: Vec3, degrees: float) -> Vec3:
    from .vec import cross
    r = math.radians(degrees)
    c, s_ = math.cos(r), math.sin(r)
    k = cross(axis, v)
    d = dot(axis, v)
    return (v[0] * c + k[0] * s_ + axis[0] * d * (1 - c), v[1] * c + k[1] * s_ + axis[1] * d * (1 - c),
            v[2] * c + k[2] * s_ + axis[2] * d * (1 - c))


@dataclass
class _Layout:
    along: Vec3
    across: Vec3
    ts: list[float]
    ss: list[float]
    smin: float
    width: float
    runs: list[list[int]]
    hist: list[int]
    gap_bins: int


def _layout(points: list[Vec3], wrist: Vec3, along: Vec3, bins: int) -> _Layout | None:
    ts = [dot(sub(p, wrist), along) for p in points]
    proj = [sub(p, scale(along, dot(p, along))) for p in points]
    pw = principal_axis(proj)
    if pw is None:
        return None
    across = pw[0]
    ss = [dot(p, across) for p in points]
    smin, smax = min(ss), max(ss)
    width = max(smax - smin, 1e-9)
    max_t = max(ts)
    slab = [i for i, t in enumerate(ts) if 0.66 * max_t <= t <= 0.94 * max_t]
    if len(slab) < 30:
        return None
    hist = [0] * bins
    for i in slab:
        hist[min(bins - 1, int((ss[i] - smin) / width * bins))] += 1
    threshold = max(1, int(0.006 * len(slab)))
    occupied = [hist[b] > threshold for b in range(bins)]
    # Close one-bin holes caused by sparse sampling (a real gap between fingers is wider than that).
    closed = [occupied[b] or (0 < b < bins - 1 and occupied[b - 1] and occupied[b + 1]) for b in range(bins)]
    runs: list[list[int]] = []
    cur = None
    for b, c in enumerate(closed):
        if c:
            cur = [b, b] if cur is None else [cur[0], b]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    runs = [r for r in runs if sum(hist[r[0]:r[1] + 1]) >= 0.03 * len(slab)]
    gap_bins = sum(runs[k + 1][0] - runs[k][1] - 1 for k in range(len(runs) - 1))
    return _Layout(along, across, ts, ss, smin, width, runs, hist, gap_bins)


def segment_fingers(points: list[Vec3], wrist: Vec3, along_hint: Vec3 | None, bins: int = 48) -> FingerSegmentation:
    """Separate finger regions in hand geometry. ``points`` are hand-weighted vertices in world space.

    The finger direction is refined by a small angular search (the whole-hand axis is skewed by palm and thumb): the
    direction that yields the clearest separation between fingers wins."""
    seg = FingerSegmentation(origin=wrist)
    if len(points) < 60:
        seg.reason = "too little hand geometry"
        return seg
    pa = principal_axis(points)
    along = along_hint
    if pa is not None:
        axis = pa[0]
        if along is None or (angle_between(along, axis) > 35 and angle_between(along, scale(axis, -1)) > 35):
            along = axis
    along = normalize(along) if along else None
    if along is None:
        seg.reason = "no hand direction"
        return seg
    if dot(sub(mean(points), wrist), along) < 0:
        along = scale(along, -1)  # point away from the wrist
    base = _layout(points, wrist, along, bins)
    if base is None:
        seg.reason = "no distal slab"
        return seg
    normal = normalize(cross(base.along, base.across))
    best = base
    if normal is not None:
        for deg in (-16, -12, -8, -4, 4, 8, 12, 16):
            cand = _layout(points, wrist, _rotate_about(base.along, normal, deg), bins)
            if cand is not None and (len(cand.runs), cand.gap_bins) > (len(best.runs), best.gap_bins) \
                    and len(cand.runs) <= 5:
                best = cand
    along, across, ts, ss = best.along, best.across, best.ts, best.ss
    smin, width, runs = best.smin, best.width, best.runs
    seg.along, seg.across = along, across
    seg.hand_length = max(ts)
    seg.runs_found = len(runs)
    if len(runs) < 4:
        seg.fused = True
        seg.reason = f"only {len(runs)} separated finger region(s): fused fingers / mitten"
        return seg
    if len(runs) > 5:
        seg.reason = f"{len(runs)} separated regions: cannot tell fingers from accessories"
        return seg
    bins_ = bins
    max_t = max(ts)
    centers = [((r[0] + r[1] + 1) / 2 / bins_ * width + smin) for r in runs]
    palm_t = 0.45 * max_t
    clusters: dict[int, list[int]] = {k: [] for k in range(len(runs))}
    for i, t in enumerate(ts):
        if t < palm_t:
            continue
        k = min(range(len(runs)), key=lambda k: abs(ss[i] - centers[k]))
        if abs(ss[i] - centers[k]) <= (runs[k][1] - runs[k][0] + 3) / bins_ * width:
            clusters[k].append(i)
    widths = [runs[k][1] - runs[k][0] + 1 for k in range(len(runs))]
    gaps = [runs[k + 1][0] - runs[k][1] - 1 for k in range(len(runs) - 1)]
    clarity = min(1.0, min(gaps) / 2.0)
    if len(runs) == 5:
        starts = {k: min((ts[i] for i in v), default=1e9) for k, v in clusters.items()}
        thumb = min((0, 4), key=lambda k: starts[k])
        rest = [k for k in range(5) if k != thumb]
        if thumb == 4:
            rest.reverse()
        order = [thumb, *rest]
        seg.clusters = {finger: clusters[k] for finger, k in zip(FINGER_ORDER, order)}
        balance = min(widths[k] for k in rest) / max(widths[k] for k in rest)
        seg.confidence = round(0.55 + 0.25 * clarity + 0.2 * balance, 3)
        return seg
    # Four finger regions in the distal slab: the thumb is shorter and usually ends before it. Look for a separate
    # cluster that sticks out sideways past the fingers' span, nearer the wrist.
    span_lo = smin + runs[0][0] / bins_ * width
    span_hi = smin + (runs[-1][1] + 1) / bins_ * width
    tol = 0.08 * width
    cand_lo = [i for i, t in enumerate(ts) if 0.12 * max_t <= t <= 0.8 * max_t and ss[i] < span_lo - tol]
    cand_hi = [i for i, t in enumerate(ts) if 0.12 * max_t <= t <= 0.8 * max_t and ss[i] > span_hi + tol]
    thumb_pts = max((cand_lo, cand_hi), key=len)
    thumb_on_lo = thumb_pts is cand_lo
    if len(thumb_pts) < 0.02 * len(points):
        seg.reason = "four finger regions and no separable thumb"
        seg.confidence = 0.55
        order = list(range(4))
        seg.clusters = {finger: clusters[k] for finger, k in zip(FINGER_ORDER[1:], order)}
        return seg
    rest = list(range(4))
    if not thumb_on_lo:
        rest.reverse()  # index is the finger nearest the thumb
    seg.clusters = {"Thumb": thumb_pts, **{finger: clusters[k] for finger, k in zip(FINGER_ORDER[1:], rest)}}
    balance = min(widths) / max(widths)
    seg.confidence = round(0.55 + 0.25 * clarity + 0.2 * balance, 3)
    return seg
