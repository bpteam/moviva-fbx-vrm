"""Procedural humanoid auto-rig from static geometry: skeleton fit, deterministic skin weights, deformation tests.

Pure numpy, no ``bpy``. Input is the world-space vertex cloud of the character; output is a minimal deform
skeleton (VRM humanoid bones only) plus weights. Every landmark records whether it was *measured* from the
geometry or *assumed* from body proportions, so the report never claims more than the geometry showed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# bone -> parent. Names are the VRM humanoid names in PascalCase; the converter's resolver maps them by name.
HIERARCHY: dict[str, str | None] = {
    "Hips": None, "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    **{f"{s}{b}": (p if p in ("Chest", "Hips") else f"{s}{p}") for s in ("Left", "Right") for b, p in (
        ("Shoulder", "Chest"), ("UpperArm", "Shoulder"), ("LowerArm", "UpperArm"), ("Hand", "LowerArm"),
        ("UpperLeg", "Hips"), ("LowerLeg", "UpperLeg"), ("Foot", "LowerLeg"))},
}
MIN_POINTS = 200  # fewer vertices cannot carry a body silhouette
SLICES = 100


@dataclass
class Frame:
    origin: np.ndarray
    left: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    confidence: float
    notes: list[str] = field(default_factory=list)

    @property
    def basis(self) -> np.ndarray:  # rows: left, up, forward
        return np.stack([self.left, self.up, self.forward])

    def to_local(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p) - self.origin) @ self.basis.T

    def to_world(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p) @ self.basis + self.origin


@dataclass
class RigFit:
    ok: bool
    bones: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # world head, tail
    frame: Frame | None = None
    height: float = 0.0
    measured: dict[str, bool] = field(default_factory=dict)  # landmark -> measured from geometry
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


def detect_frame(points: np.ndarray) -> Frame:
    """Up = Z for Blender scenes (glTF/FBX importers convert to Z-up); a clearly taller other axis wins."""
    ext = np.ptp(points, axis=0)
    notes: list[str] = []
    up_axis = 2
    if ext[2] < 0.6 * ext.max():
        up_axis = int(np.argmax(ext))
        notes.append(f"tallest axis is {'XYZ'[up_axis]}, not Z")
    horiz = [a for a in range(3) if a != up_axis]
    lat_axis = max(horiz, key=lambda a: ext[a])
    depth_axis = next(a for a in horiz if a != lat_axis)
    ratio = float(ext[lat_axis] / max(ext[depth_axis], 1e-9))
    up = np.zeros(3); up[up_axis] = 1.0
    fwd = np.zeros(3); fwd[depth_axis] = -1.0  # Blender: characters face -Y
    low = points[:, up_axis] < points[:, up_axis].min() + 0.06 * ext[up_axis]
    leg = (points[:, up_axis] > points[:, up_axis].min() + 0.15 * ext[up_axis]) & \
          (points[:, up_axis] < points[:, up_axis].min() + 0.30 * ext[up_axis])
    sign_conf = 0.0
    if low.any() and leg.any():
        c = float(np.median(points[leg, depth_axis]))
        front, back = float(points[low, depth_axis].max() - c), float(c - points[low, depth_axis].min())
        if max(front, back) > 1.15 * max(min(front, back), 1e-9):  # toes overhang the shin on the facing side
            fwd[depth_axis] = 1.0 if front > back else -1.0
            sign_conf = 1.0
        else:
            notes.append("facing direction assumed (-Y); feet gave no clear toe direction")
    left = np.cross(up, fwd)
    conf = min(1.0, (ratio - 1.0) * 2.0 + 0.5) * (0.6 + 0.4 * sign_conf)
    mid = np.zeros(3)
    mid[lat_axis] = float((points[:, lat_axis].min() + points[:, lat_axis].max()) / 2)
    mid[depth_axis] = float(np.median(points[:, depth_axis]))
    mid[up_axis] = float(points[:, up_axis].min())
    return Frame(mid, left, up, fwd, max(0.0, conf), notes)


def _central_halfwidth(l: np.ndarray, gap: float) -> float:
    """Half-width of the lateral cluster that contains l = 0 (clusters split at gaps wider than ``gap``,
    widened to four times the mean vertex spacing so a sparse low-poly slice is not cut apart)."""
    s = np.sort(l)
    gap = max(gap, 4.0 * float(np.ptp(s)) / max(len(s), 1))
    cuts = np.where(np.diff(s) > gap)[0]
    edges = np.concatenate([[0], cuts + 1, [len(s)]])
    for a, b in zip(edges[:-1], edges[1:]):
        if s[a] - gap <= 0 <= s[b - 1] + gap:
            return float(max(abs(s[a]), abs(s[b - 1])))
    return float(np.abs(s).max())


def fit_skeleton(points: np.ndarray) -> RigFit:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < MIN_POINTS:
        return RigFit(False, warnings=[f"only {len(pts)} vertices: too few to find a body"])
    frame = detect_frame(pts)
    loc = frame.to_local(pts)
    l, u, f = loc[:, 0], loc[:, 1], loc[:, 2]
    height = float(u.max() - u.min())
    if height <= 1e-6:
        return RigFit(False, warnings=["model has no height"])
    fit = RigFit(True, frame=frame, height=height, warnings=list(frame.notes))
    H = height
    meas = fit.measured

    # Per-slice central width and population.
    edges = np.linspace(0, H, SLICES + 1)
    idx = np.clip((u / H * SLICES).astype(int), 0, SLICES - 1)
    width = np.full(SLICES, np.nan)
    extent = np.full(SLICES, np.nan)  # outermost |lateral| per slice: robust on sparse, hollow low-poly slices
    count = np.zeros(SLICES, int)
    split = np.zeros(SLICES, bool)
    gap = 0.015 * H
    for i in range(SLICES):
        m = idx == i
        count[i] = m.sum()
        if count[i] < max(6, 0.0005 * len(pts)):
            continue
        width[i] = _central_halfwidth(l[m], gap)
        extent[i] = float(np.abs(l[m]).max())
        lm = l[m]
        g = max(gap, 4.0 * float(np.median(np.diff(np.sort(lm)))))  # typical spacing, not range: two legs are far apart
        split[i] = (lm > g).sum() >= 3 and (lm < -g).sum() >= 3 and \
            not ((lm > -g) & (lm < g)).any()
    t = (edges[:-1] + edges[1:]) / 2 / H

    # Crotch: highest slice (below 60 %) of an unbroken run of left/right-separated legs starting at the feet.
    crotch = None
    run_top = None
    for i in range(int(0.6 * SLICES)):
        if split[i]:
            run_top = i
        elif run_top is not None and i - run_top > 4 and count[i] >= 4:
            break
    if run_top is not None and run_top >= 0.15 * SLICES:
        crotch = (run_top + 1) / SLICES * H
    meas["crotch"] = crotch is not None
    if crotch is None:
        crotch = 0.47 * H
        fit.warnings.append("legs do not separate (skirt, robe or one-piece?): hips placed at 47 % of height")

    # Neck: the narrowest outline between the legs and the top of a (possibly huge) head, highest among near-ties.
    lo, hi = int(0.40 * SLICES), int(0.88 * SLICES)  # above that: a hair tip, not a neck
    # running max over 3 slices: a slice that happens to hold few vertices must not read as a narrow waist
    pad = np.where(np.isnan(extent), 0.0, extent)
    smooth = np.maximum(np.maximum(np.roll(pad, 1), pad), np.roll(pad, -1))
    w = np.where(np.isnan(extent[lo:hi]), np.inf, smooth[lo:hi])
    if np.isfinite(w).any():
        near = np.where(w <= w.min() * 1.08)[0]
        neck_u = float(edges[lo + near[-1]] + H / SLICES / 2)
        meas["neck"] = True
    else:
        neck_u, meas["neck"] = 0.84 * H, False
    neck_u = float(np.clip(neck_u, crotch + 0.2 * H, 0.97 * H))

    sh_u = neck_u - 0.05 * H
    # Waist: the narrowest torso-only slice between the hips and the chest (a skirt flares below, arms above).
    wb = [i for i in range(SLICES) if crotch + 0.2 * (sh_u - crotch) <= (edges[i] + edges[i + 1]) / 2
          <= sh_u - 0.1 * (sh_u - crotch) and not np.isnan(width[i])]
    waist_half = float(min((width[i] for i in wb), default=0.12 * H))
    # A hollow low-poly slice has no vertices near the middle, so the central cluster reads far too thin;
    # half the outline (arms included) is a floor for the torso half-width.
    ext_w = [extent[i] for i in wb if not np.isnan(extent[i])]
    if ext_w and waist_half < 0.05 * H:  # only when the cluster is implausibly thin
        waist_half = max(waist_half, 0.5 * float(np.median(ext_w)))
    waist_u = (crotch + sh_u) / 2
    # Shoulder width: the widest torso-only slice between mid-torso and the neck (arms may be out, so cap it).
    band = [i for i in range(SLICES) if (waist_u + sh_u) / 2 <= (edges[i] + edges[i + 1]) / 2 <= sh_u
            and not np.isnan(width[i])]
    sh_half = float(np.clip(max((width[i] for i in band), default=waist_half), 0.8 * waist_half, 1.3 * waist_half))
    ext_s = [extent[i] for i in band if not np.isnan(extent[i])]
    if ext_s and sh_half < 0.05 * H:
        sh_half = max(sh_half, 0.5 * float(min(ext_s)))

    # Arms: vertices clearly outside the torso between the knees and the neck.
    # A skirt or wide hips must not pass for an arm: the outside test uses the widest of waist and hip band.
    hb = (u > crotch - 0.05 * H) & (u < crotch + 0.15 * H)
    hip_half = _central_halfwidth(l[hb], gap) if hb.sum() >= 4 else waist_half
    ref_half = max(waist_half, hip_half)
    arm_pts: dict[str, np.ndarray] = {}
    for side, sgn in (("Left", 1.0), ("Right", -1.0)):
        m = (sgn * l > 1.25 * ref_half) & (u > crotch - 0.1 * H) & (u < neck_u)
        arm_pts[side] = loc[m] if m.sum() >= max(8, 0.002 * len(pts)) else np.empty((0, 3))
    fc = float(np.median(f[(u > crotch) & (u < neck_u)])) if ((u > crotch) & (u < neck_u)).any() else 0.0

    B: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    def bone(name: str, head, tail) -> None:
        B[name] = (np.asarray(head, float), np.asarray(tail, float))

    hips_u = crotch + 0.04 * H
    sh_base = neck_u - 0.03 * H
    spine_end = hips_u + 0.45 * (sh_base - hips_u)
    bone("Hips", (0, hips_u, fc), (0, hips_u + 0.5 * (spine_end - hips_u), fc))
    bone("Spine", (0, hips_u + 0.5 * (spine_end - hips_u), fc), (0, spine_end, fc))
    bone("Chest", (0, spine_end, fc), (0, sh_base, fc))
    bone("Neck", (0, sh_base, fc), (0, neck_u + 0.02 * H, fc))
    bone("Head", (0, neck_u + 0.02 * H, fc), (0, H, fc))

    ankle_u = 0.045 * H
    low = (u < crotch - 0.05 * H)
    foot = u < 0.05 * H
    for side, sgn in (("Left", 1.0), ("Right", -1.0)):
        # arm
        sh = np.array([sgn * sh_half, sh_base, fc])
        pts_a = arm_pts[side]
        if len(pts_a):
            tip = pts_a[np.argmax(np.linalg.norm(pts_a - sh, axis=1))]
            wrist = sh + (tip - sh) * 0.88
            meas[f"{side}Arm"] = True
        else:
            wrist = np.array([sgn * (sh_half + 0.02 * H), crotch + 0.02 * H, fc])
            tip = wrist + (wrist - sh) * 0.12
            meas[f"{side}Arm"] = False
            fit.warnings.append(f"{side} arm not separable from the torso: placed hanging at the side")
        elbow = sh + (wrist - sh) * 0.5
        clav = np.array([sgn * 0.3 * sh_half, sh_base, fc])
        bone(f"{side}Shoulder", clav, sh)
        bone(f"{side}UpperArm", sh, elbow)
        bone(f"{side}LowerArm", elbow, wrist)
        bone(f"{side}Hand", wrist, tip)
        # leg
        lm = low & (sgn * l > 0)
        leg_l = float(np.median(np.abs(l[lm]))) if lm.sum() >= 8 else 0.45 * waist_half
        meas[f"{side}Leg"] = bool(lm.sum() >= 8)
        hip = np.array([sgn * leg_l, hips_u, fc])
        ankle = np.array([sgn * leg_l, ankle_u, float(np.median(f[lm])) if lm.sum() >= 8 else fc])
        knee = hip + (ankle - hip) * 0.5
        fm = foot & (sgn * l > 0)
        toe_f = float((f[fm].max() if frame.forward is not None and fm.any() else ankle[2] + 0.05 * H))
        toe_f = max(toe_f, ankle[2] + 0.03 * H)
        bone(f"{side}UpperLeg", hip, knee)
        bone(f"{side}LowerLeg", knee, ankle)
        bone(f"{side}Foot", ankle, (sgn * leg_l, 0.01 * H, toe_f))

    fit.bones = {n: (frame.to_world(h), frame.to_world(tl)) for n, (h, tl) in B.items()}
    n_meas = sum(meas.values())
    fit.confidence = round(frame.confidence * (0.4 + 0.6 * n_meas / max(len(meas), 1)), 3)
    return fit


# ------------------------------------------------------------------ weights


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def distance_weights(points: np.ndarray, fit: RigFit, max_influences: int = 4) -> tuple[list[str], np.ndarray]:
    """Deterministic weights: inverse-distance to bone segments, limbs restricted to their own side."""
    names = list(fit.bones)
    assert fit.frame is not None
    loc = fit.frame.to_local(points)
    lat = loc[:, 0]
    eps = 0.01 * fit.height
    D = np.empty((len(points), len(names)))
    for j, n in enumerate(names):
        a, b = fit.bones[n]
        D[:, j] = _segment_distance(points, a, b)
        if n.startswith(("Left", "Right")):
            sgn = 1.0 if n.startswith("Left") else -1.0
            D[sgn * lat < -0.03 * fit.height, j] = np.inf  # never reach across the body
    W = 1.0 / (D + eps) ** 5  # steep falloff: a torso vertex beside an arm must not follow it
    W[~np.isfinite(D)] = 0.0
    W = np.where(W < 0.05 * W.max(axis=1, keepdims=True), 0.0, W)  # drop negligible influences
    if max_influences < len(names):
        cut = np.sort(W, axis=1)[:, -max_influences][:, None]
        W = np.where(W >= cut, W, 0.0)
    s = W.sum(axis=1, keepdims=True)
    W = np.divide(W, s, out=np.zeros_like(W), where=s > 0)
    return names, W


# ------------------------------------------------------------------ deformation tests


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _subtree(name: str) -> list[str]:
    out = [name]
    for n, p in HIERARCHY.items():
        if p == name:
            out += _subtree(n)
    return out


def weight_stats(W: np.ndarray, max_influences: int = 4) -> dict:
    s = W.sum(axis=1)
    return {"unweighted": int((s < 1e-6).sum()), "not_normalized": int((np.abs(s - 1) > 1e-3).sum() - (s < 1e-6).sum()),
            "over_influence": int(((W > 1e-6).sum(axis=1) > max_influences).sum())}


def deformation_tests(points: np.ndarray, names: list[str], W: np.ndarray, fit: RigFit) -> dict:
    """Pose the skeleton numerically (linear blend skinning) and check that the right part moves and nothing else.

    Each test: ``moved`` = vertices displaced by more than 1 % of height; ``leaked`` = moved vertices outside the
    region the bone is expected to affect. PASS needs a real movement and <= 5 % leakage; up to 25 % is CHECK (a skirt legitimately drags the other leg).
    """
    assert fit.frame is not None
    fr, H = fit.frame, fit.height
    loc = fr.to_local(points)
    col = {n: i for i, n in enumerate(names)}
    lbasis = fr.basis  # rows left, up, forward
    neck_u = fit.bones["Neck"][0] @ lbasis[1] - fr.origin @ lbasis[1]
    crotch_u = fit.bones["Hips"][0] @ lbasis[1] - fr.origin @ lbasis[1]

    def chain_distance(bones: list[str]) -> np.ndarray:
        return np.min([_segment_distance(points, *fit.bones[b]) for b in bones], axis=0)

    def pose(bone: str, axis_row: int, deg: float, leak_mask) -> dict:
        head = fit.bones[bone][0]
        R = _rot(lbasis[axis_row], deg)
        sub = [col[n] for n in _subtree(bone) if n in col]
        wsum = W[:, sub].sum(axis=1)
        moved_pts = (points - head) @ R.T + head
        disp = np.linalg.norm((moved_pts - points) * wsum[:, None], axis=1)
        moved = disp > 0.01 * H
        leaked = moved & leak_mask
        n_moved = int(moved.sum())
        frac = float(leaked.sum() / n_moved) if n_moved else 1.0
        status = "FAIL" if n_moved == 0 or frac > 0.25 else ("CHECK" if frac > 0.05 else "PASS")
        return {"status": status, "moved": n_moved, "leaked": int(leaked.sum()), "leak_fraction": round(frac, 4),
                "max_displacement_ratio": round(float(disp.max() / H), 4)}

    lat, up = loc[:, 0], loc[:, 1]
    # A moved vertex leaks when it is on the wrong side of the body or farther than 10 % of the height from the
    # limb it should belong to; the head may only move vertices from the neck up.
    arm_l = chain_distance(["LeftUpperArm", "LeftLowerArm", "LeftHand"]) > 0.10 * H
    arm_r = chain_distance(["RightUpperArm", "RightLowerArm", "RightHand"]) > 0.10 * H
    leg_l = chain_distance(["LeftUpperLeg", "LeftLowerLeg", "LeftFoot"]) > 0.10 * H
    leg_r = chain_distance(["RightUpperLeg", "RightLowerLeg", "RightFoot"]) > 0.10 * H
    tests = {
        "arm_raise_left": pose("LeftUpperArm", 2, -70, (lat < -0.03 * H) | arm_l),
        "arm_raise_right": pose("RightUpperArm", 2, 70, (lat > 0.03 * H) | arm_r),
        "elbow_bend_left": pose("LeftLowerArm", 1, 60, (lat < -0.03 * H) | arm_l),
        "head_rotation": pose("Head", 1, 30, up < neck_u - 0.06 * H),
        "leg_raise_left": pose("LeftUpperLeg", 0, -35, (lat < -0.03 * H) | leg_l),
        "leg_raise_right": pose("RightUpperLeg", 0, -35, (lat > 0.03 * H) | leg_r),
    }
    stats = weight_stats(W)
    order = {"PASS": 0, "CHECK": 1, "FAIL": 2}
    worst = max((t["status"] for t in tests.values()), key=order.get)
    if stats["unweighted"] > 0.001 * len(points) or stats["not_normalized"]:
        worst = "FAIL"
    return {"status": worst, "tests": tests, "weights": stats,
            "score": round(sum(t["leak_fraction"] for t in tests.values()) / len(tests), 4)}


# ------------------------------------------------------------------ result


@dataclass
class AutoRigResult:
    """What the adapter did, in report form. ``status``: PASS / CHECK (usable, inspect it) / FAIL (not applied)."""

    status: str
    method: str = ""
    attempts: list[dict] = field(default_factory=list)
    bones: list[str] = field(default_factory=list)
    confidence: float = 0.0
    measured: dict[str, bool] = field(default_factory=dict)
    height: float = 0.0
    meshes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("status", "method", "attempts", "bones", "confidence", "measured",
                                              "height", "meshes", "warnings")}


STATUS_ORDER = {"PASS": 0, "CHECK": 1, "FAIL": 2}


def better(a: dict, b: dict) -> bool:
    """Is deformation result ``a`` better than ``b``? Status first, then mean leakage."""
    return (STATUS_ORDER[a["status"]], a["score"]) < (STATUS_ORDER[b["status"]], b["score"])
