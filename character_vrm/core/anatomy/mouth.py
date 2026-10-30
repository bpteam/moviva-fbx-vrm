"""Mouth localisation: every source of evidence proposes a lip seam, the consensus wins (pure).

Sources, strongest first: an existing mouth-opening morph (the seam is where it stops pulling the surface down), the
skin's own opening at the lips (a small closed boundary loop, surrounded by skin), the jaw bone's skinning (lower lip
on the jaw, upper lip not), the mouth interior / teeth surfaces, the concavity between the lips on the midline profile,
and the template.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .facegeo import (
    CANONICAL_FACE, FaceGeometry, FaceMorphs, FacePoint, local_deltas, moved_vertices, skin_loops, surrounded,
)
from .frame import AnatomicalFrame
from .surfaces import SKIN, TEETH
from .vec import Vec3

# Mouth height as a fraction of chin -> eye line; the template's own value and the plausible band around it
# (realistic faces sit near 0.36, stylised / anime faces lower and higher).
MOUTH_T_TEMPLATE = CANONICAL_FACE["mouth_center"][1] / CANONICAL_FACE["left_eye"][1]
MOUTH_T_BAND = (0.08, 0.72)
# Mouth width as a fraction of the distance between the eye centres when nothing measures it.
MOUTH_WIDTH_PER_IOD = 0.62
# Jaw skinning is read in a strip this wide (fraction of head width) either side of the midline, and states the seam
# only when this share of the strip's samples lies on the right side of it.
JAW_MIDLINE_HALF = 0.04
JAW_SEAM_AGREEMENT = 0.9
SOURCE_CONFIDENCE = {"morph": 0.95, "lip-opening": 0.88, "jaw-skinning": 0.8, "mouth-interior": 0.72,
                     "profile": 0.62, "template": 0.35}


@dataclass
class MouthEstimate:
    source: str
    seam: float           # local height of the lip seam
    center_x: float
    half_width: float | None
    confidence: float
    detail: str = ""


def step_split(samples: list[tuple[float, float]], window: int | None = None) -> tuple[float, float] | None:
    """``samples`` = (height, value). The lip seam: the *highest* sharp drop in value going upwards.

    For every split the mean of the ``window`` samples just below minus the mean of those just above is the drop;
    among drops of at least half the largest, the highest one wins: above the seam the upper lip does not move (or
    carries no jaw weight), while below it the motion may grow further towards the chin (a jaw turning, a chin that
    moves more than the lip) — that step must not pull the seam down. Returns ``(split height, drop)``."""
    band = sorted(samples)
    n = len(band)
    k = window or max(3, n // 12)
    if n < 2 * k + 1:
        return None
    prefix = [0.0]
    for _, v in band:
        prefix.append(prefix[-1] + v)
    drops: list[tuple[int, float]] = []
    for s in range(k, n - k + 1):
        drop = (prefix[s] - prefix[s - k]) / k - (prefix[s + k] - prefix[s]) / k
        drops.append((s, drop))
    best = max((d for _, d in drops), default=0.0)
    if best <= 0.0:
        return None
    # local maxima that are at least half the best step; the highest of them is the seam
    peaks = [(s, d) for i, (s, d) in enumerate(drops) if d >= 0.5 * best
             and (i == 0 or d >= drops[i - 1][1]) and (i == len(drops) - 1 or d >= drops[i + 1][1])]
    s, d = max(peaks, key=lambda sd: sd[0]) if peaks else max(drops, key=lambda sd: sd[1])
    split = 0.5 * (band[s - 1][0] + band[s][0]) if s < n else band[-1][0]
    return split, d


def jaw_seam(samples: list[tuple[float, float]], min_side: int = 4) -> tuple[float, float] | None:
    """``samples`` = (height, jaw weight) on the midline. The height that best separates the jaw-carried lower lip
    (weight >= 0.5, below) from the rest (above), and the share of samples on the right side of it.

    A classifier, not a mean over a window: the inner roll of the lips (upper-lip skin hanging below the lower lip's
    top) and a smooth weight fall-off at the corners do not dilute a seam the skinning states clearly. None without
    ``min_side`` samples on each side."""
    band = sorted(samples)
    n = len(band)
    jaw = [w >= 0.5 for _, w in band]
    total_on = sum(jaw)
    if total_on < min_side or n - total_on < min_side:
        return None
    best_s, best_err = 0, n + 1
    on_below = 0
    for s in range(1, n):
        on_below += jaw[s - 1]
        err = (s - on_below) + (total_on - on_below)  # not-jaw below the split + jaw above it
        if err <= best_err and band[s - 1][0] < band[s][0]:  # ties: the highest split, as in ``step_split``
            best_s, best_err = s, err
    if best_s == 0:
        return None
    return 0.5 * (band[best_s - 1][0] + band[best_s][0]), 1.0 - best_err / n


def morph_opening(skin: list[FacePoint], frame: AnatomicalFrame, deltas: dict[tuple[str, int], Vec3],
                  cx: float, seam: float, half: float, span: float, z_max: float, depth: float) -> float:
    """How far a morph opens the lips: downward travel of the lower lip minus that of the upper lip, measured just
    below and above the seam over the middle of the mouth (source units)."""
    lower: list[float] = []
    upper: list[float] = []
    for p in skin:
        x, y, z = p.local
        if abs(x - cx) > 0.5 * half or z < z_max - 0.35 * depth:
            continue
        d = deltas.get((p.mesh, p.index))
        dy = frame.vector_to_local(d)[1] if d else 0.0
        if seam - 0.12 * span <= y < seam:
            lower.append(-dy)
        elif seam <= y < seam + 0.08 * span:
            upper.append(-dy)
    if not lower:
        return 0.0
    return sum(lower) / len(lower) - (sum(upper) / len(upper) if upper else 0.0)


def _front_band(skin: list[FacePoint], x_mid: float, half: float, lo: float, hi: float, z_front: float,
                depth: float) -> list[FacePoint]:
    return [p for p in skin if abs(p.local[0] - x_mid) <= half and lo <= p.local[1] <= hi
            and p.local[2] >= z_front - 0.35 * depth]


def mouth_candidates(skin, geometry: FaceGeometry, by_key, frame, morphs: FaceMorphs, x_mid, chin_y, span,
                      head_width, depth, z_max, nose, iod, jaw_bone) -> list[MouthEstimate]:
    out: list[MouthEstimate] = []
    lo, hi = chin_y + MOUTH_T_BAND[0] * span, chin_y + MOUTH_T_BAND[1] * span
    if nose is not None:
        hi = min(hi, nose[1])

    # (1) morph: where an existing mouth-opening morph stops pulling the surface down
    open_keys = [e for e in ("aa", "oh", "jawOpen") if morphs.get(e)]
    if open_keys:
        d = local_deltas(frame, morphs.get(open_keys[0]))
        union: set[tuple[str, int]] = set()
        for e in ("aa", "ih", "ou", "ee", "oh", "jawOpen"):
            union |= moved_vertices(morphs.get(e))
        lips = [by_key[k] for k in union if k in by_key and by_key[k].role == SKIN]
        if len(lips) >= 12:
            lz = sorted(p.local[2] for p in lips)
            z_front = lz[int(0.9 * (len(lz) - 1))]
            ldepth = lz[-1] - lz[0]
            lips = [p for p in lips if p.local[2] >= z_front - 0.45 * ldepth]
            cx = sum(p.local[0] for p in lips) / len(lips)
            widths = sorted(abs(p.local[0] - cx) for p in lips)
            half = widths[int(0.95 * (len(widths) - 1))]
            band = [(p.local[1], -d.get((p.mesh, p.index), (0.0, 0.0, 0.0))[1]) for p in lips
                    if abs(p.local[0] - cx) < 0.35 * half]
            split = step_split(band)
            biggest = max((abs(v) for _, v in band), default=0.0)
            # the drop must be a real share of the motion: a lip-shaping key that barely moves the lips says nothing
            if split is not None and split[1] > 0.25 * biggest and biggest > 0.01 * span:
                seam = split[0]
                at = sorted(abs(p.local[0] - cx) for p in lips if abs(p.local[1] - seam) < 0.25 * half)
                corner = at[int(0.97 * (len(at) - 1))] if at else half
                out.append(MouthEstimate("morph", seam, cx, corner, SOURCE_CONFIDENCE["morph"],
                                         f"{open_keys[0]} morph, {len(lips)} lip vertices"))

    # (2) the skin's opening at the lips: a small closed boundary loop on the midline
    loop = _mouth_loop(geometry, by_key, x_mid, lo, hi, head_width, z_max, depth)
    if loop is not None:
        out.append(loop)

    # (3) jaw skinning: the lower lip rides on the jaw, the upper lip does not. Read on the midline only: towards the
    # corners the lip line curves and the weights fade, which mixes heights
    if jaw_bone:
        band_pts = _front_band(skin, x_mid, JAW_MIDLINE_HALF * head_width, lo, hi, z_max, depth)
        split = jaw_seam([(p.local[1], p.jaw) for p in band_pts]) if len(band_pts) >= 12 else None
        if split is not None and split[1] >= JAW_SEAM_AGREEMENT:
            seam = split[0]
            # the corners: where the jaw's share falls off sideways at the seam height
            row = [p for p in skin if abs(p.local[1] - seam) < 0.04 * span and p.local[2] >= z_max - 0.5 * depth]
            moving = sorted(abs(p.local[0] - x_mid) for p in row if 0.15 < p.jaw < 0.85)
            half = moving[int(0.9 * (len(moving) - 1))] if len(moving) >= 4 else None
            out.append(MouthEstimate("jaw-skinning", seam, x_mid, half,
                                     SOURCE_CONFIDENCE["jaw-skinning"] * split[1],
                                     f"jaw weight separation {split[1]:.2f}"))

    # (4) mouth interior / teeth surfaces
    inner = [p for p in geometry.points if p.role == TEETH and lo <= p.local[1] <= hi + 0.1 * span]
    if len(inner) >= 12:
        iz = sorted(p.local[2] for p in inner)
        front = [p for p in inner if p.local[2] >= iz[int(0.6 * (len(iz) - 1))]]
        ys = sorted(p.local[1] for p in front)
        xs = sorted(p.local[0] for p in inner)
        cx = 0.5 * (xs[int(0.03 * (len(xs) - 1))] + xs[int(0.97 * (len(xs) - 1))])
        half = 0.5 * (xs[int(0.97 * (len(xs) - 1))] - xs[int(0.03 * (len(xs) - 1))])
        out.append(MouthEstimate("mouth-interior", ys[len(ys) // 2], cx, half, SOURCE_CONFIDENCE["mouth-interior"],
                                 f"{len(inner)} interior vertices"))

    # (5) the concavity between the lips on the midline profile
    prof = _profile_seam(skin, x_mid, lo, hi, head_width, depth, span, chin_y, nose)
    if prof is not None:
        out.append(prof)
    out.append(MouthEstimate("template", chin_y + MOUTH_T_TEMPLATE * span, x_mid, None, SOURCE_CONFIDENCE["template"]))
    return out


def _mouth_loop(geometry: FaceGeometry, by_key, x_mid, lo, hi, head_width, z_max, depth) -> MouthEstimate | None:
    best: tuple[float, MouthEstimate] | None = None
    for name, pts, own in skin_loops(geometry, by_key, head_width):
        if True:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = max(p[2] for p in pts)
            w = max(p[0] for p in pts) - min(p[0] for p in pts)
            h = max(p[1] for p in pts) - min(p[1] for p in pts)
            if abs(cx - x_mid) > 0.06 * head_width or not lo <= cy <= hi:
                continue
            if not 0.08 * head_width <= w <= 0.5 * head_width or h > 0.8 * w:
                continue
            if cz < z_max - 0.5 * depth:
                continue  # a mouth-bag or throat opening deep inside, not the lips
            if not surrounded(own, cx, cy, w, z_max - 0.5 * depth):
                continue
            score = -abs(cx - x_mid) - h
            if best is None or score > best[0]:
                best = (score, MouthEstimate("lip-opening", cy, cx, 0.5 * w, SOURCE_CONFIDENCE["lip-opening"],
                                             f"boundary loop of {len(pts)} vertices on {name}"))
    return best[1] if best else None


# Where the lip seam sits between the chin (0) and the nose tip (1) on the midline: subnasal ~0.8, seam ~0.5,
# mentolabial sulcus ~0.25 (realistic proportions; stylised faces stay near the middle too).
SEAM_NOSE_FRACTION = (0.52, 0.16)


def _profile_seam(skin, x_mid, lo, hi, head_width, depth, span, chin_y, nose) -> MouthEstimate | None:
    """The concavity between the lips on the midline profile. The profile has several dips (under the nose, between
    the lips, above the chin); each is scored by its depth and by how close it sits to the seam's usual place
    between chin and nose."""
    strip = [p.local for p in skin if abs(p.local[0] - x_mid) < 0.05 * head_width and lo <= p.local[1] <= hi]
    if len(strip) < 12:
        return None
    bins = 40
    step = (hi - lo) / bins
    front: list[float | None] = [None] * bins
    for x, y, z in strip:
        k = min(bins - 1, int((y - lo) / step))
        if front[k] is None or z > front[k]:
            front[k] = z
    for k in range(1, bins - 1):
        if front[k] is None and front[k - 1] is not None and front[k + 1] is not None:
            front[k] = 0.5 * (front[k - 1] + front[k + 1])
    top = nose[1] if nose is not None else chin_y + span * CANONICAL_FACE["nose_tip"][1] / CANONICAL_FACE["left_eye"][1]
    ref_len = max(top - chin_y, 1e-9)
    best: tuple[float, float, int] | None = None  # (score, dip, bin)
    for k in range(2, bins - 2):
        if front[k] is None:
            continue
        left = [f for f in front[max(0, k - 8):k] if f is not None]
        right = [f for f in front[k + 1:k + 9] if f is not None]
        if not left or not right:
            continue
        dip = min(max(left), max(right)) - front[k]
        if dip <= 0:
            continue
        f = (lo + (k + 0.5) * step - chin_y) / ref_len
        prior = math.exp(-0.5 * ((f - SEAM_NOSE_FRACTION[0]) / SEAM_NOSE_FRACTION[1]) ** 2)
        score = dip * prior
        if best is None or score > best[0]:
            best = (score, dip, k)
    if best is None or best[1] < 0.004 * depth:
        return None
    conf = SOURCE_CONFIDENCE["profile"] * min(1.0, best[1] / (0.02 * depth))
    return MouthEstimate("profile", lo + (best[2] + 0.5) * step, x_mid, None, conf, f"lip concavity {best[1]:.4f}")


def fuse_mouth(candidates: list[MouthEstimate], chin_y: float, span: float) -> MouthEstimate:
    """Consensus of the plausible estimates: they are grouped by seam height, the group with the largest summed
    confidence wins (two agreeing independent sources beat one stronger outlier), its most trusted member gives the
    position, the widest measured member the width. The template only votes when nothing else does."""
    plausible = [c for c in candidates
                 if MOUTH_T_BAND[0] <= (c.seam - chin_y) / max(span, 1e-9) <= MOUTH_T_BAND[1]]
    measured = [c for c in plausible if c.source != "template"]
    if not measured:
        return plausible[0] if plausible else candidates[-1]
    tol = 0.06 * span
    groups = [[c for c in measured if abs(c.seam - ref.seam) < tol] for ref in measured]
    group = max(groups, key=lambda g: (sum(c.confidence for c in g), max(c.confidence for c in g)))
    best = max(group, key=lambda c: c.confidence)
    others = [c for c in measured if c not in group]
    conf = best.confidence + 0.05 * (len(group) - 1) - 0.1 * sum(1 for c in others if c.confidence > 0.5)
    widths = [c.half_width for c in group if c.half_width]
    half = max(widths) if widths else None
    return MouthEstimate(best.source, best.seam, best.center_x, half, max(0.1, min(0.99, conf)),
                         best.detail + (f"; agrees: {', '.join(c.source for c in group if c is not best)}"
                                        if len(group) > 1 else ""))
