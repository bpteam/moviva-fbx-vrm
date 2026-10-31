"""Arms found by geodesic distance through the body's volume, whatever pose they hang in.

Slice-based fitters (the procedural fit here, Autorig Workbench's ``auto_landmarks``) look for the arms as the
widest part of the body or as runs beside the torso; a wide bob of hair, a cape or arms hanging against the hips
defeat them. Through the voxelised volume, the hands are the farthest points from the top of the head after the
feet, so the farthest-point order finds them: hair and capes hang off the head and stay geodesically close to it. Each arm's centre line is then traced along the mesh surface from the fingertip (an arm resting
against the body touches it in the volume, never on the surface): centroids of geodesic bands, up to where a band
widens into the torso (the armpit) or the arm's own piece ends.

The skin-weight cleanup that follows a volume-based skinning lives here too: per mesh piece, by distance to the
bones, and smoothed along the surface.

Pure numpy. Input: points in a frame with Z up, the character facing -Y (its left is +X).
"""

from __future__ import annotations

import numpy as np

UPPER_ARM, FOREARM, KNUCKLE = 0.40, 0.33, 0.55  # shares of shoulder-to-fingertip (Autorig Workbench's AUTO)
_NEIGHBOURS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _shift(a: np.ndarray, d: tuple[int, int, int], fill) -> np.ndarray:
    out = np.full_like(a, fill)
    src = tuple(slice(max(0, -k), a.shape[i] - max(0, k)) for i, k in enumerate(d))
    dst = tuple(slice(max(0, k), a.shape[i] - max(0, -k)) for i, k in enumerate(d))
    out[dst] = a[src]
    return out


def _dilate(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for d in _NEIGHBOURS:
        out |= _shift(a, d, False)
    return out


def voxelize(samples: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Solid occupancy (shell closed by one dilation, interior filled) and the grid origin."""
    lo = samples.min(axis=0) - 2 * voxel
    idx = np.floor((samples - lo) / voxel).astype(int)
    shape = idx.max(axis=0) + 3
    shell = np.zeros(shape, bool)
    shell[tuple(idx.T)] = True
    shell = _dilate(shell)
    outside = np.zeros(shape, bool)  # flood the empty space from the grid's border
    outside[0, :, :] = outside[-1, :, :] = outside[:, 0, :] = outside[:, -1, :] = True
    outside &= ~shell
    while True:
        grown = _dilate(outside) & ~shell
        if grown.sum() == outside.sum():
            break
        outside = grown
    return ~outside, lo


def geodesic(solid: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Steps (6-connected) from the seed voxels through the solid; -1 where unreachable."""
    dist = np.full(solid.shape, -1, np.int32)
    front = seeds & solid
    dist[front] = 0
    step = 0
    while front.any():
        step += 1
        nxt = _dilate(front) & solid & (dist < 0)
        dist[nxt] = step
        front = nxt
    return dist


def find_arms(samples: np.ndarray, verts: np.ndarray, edges: np.ndarray, height: float | None = None,
              voxels: int = 160) -> dict | None:
    """{"Left": {...}, "Right": {...}} with shoulder, elbow, wrist, knuckle, tip (world points), or None when the
    volume does not show two hands and two feet in a humanoid arrangement. ``samples``: points on the surface
    (dense enough for the voxel size); ``verts``/``edges``: the welded mesh the arms are traced along."""
    samples = np.asarray(samples, float)
    H = float(height or np.ptp(samples[:, 2]))
    v = H / voxels
    solid, lo = voxelize(samples, v)
    centre = lambda ijk: lo + (np.asarray(ijk) + 0.5) * v
    # the body: the solid component holding the most voxels, reached from its topmost voxel
    occ = np.argwhere(solid)
    top = occ[np.argmax(occ[:, 2])]
    seed = np.zeros_like(solid)
    seed[tuple(top)] = True
    d_top = geodesic(solid, seed)
    if (d_top >= 0).sum() < 0.5 * solid.sum():
        return None
    # farthest-point order from the head: feet and hands come first (hair and capes stay close to the head); a few
    # more points cover a foot the voxels left unreachable or a nose that sticks out.
    # shortcut: a hand resting on the hip is reached through the hip, so its tip is not an end and no arms are
    # found (the workbench's own arms are kept); local maxima with a prominence test would cover that pose
    found = []
    d_set = d_top.copy()
    for _ in range(6):
        far = np.unravel_index(np.argmax(np.where(d_set >= 0, d_set, -1)), d_set.shape)
        found.append((np.array(far), int(d_top[far])))
        s = np.zeros_like(solid)
        s[far] = True
        d_new = geodesic(solid, s)
        d_set = np.where((d_set >= 0) & (d_new >= 0), np.minimum(d_set, d_new), d_set)
    z0 = samples[:, 2].min()
    low = samples[samples[:, 2] < z0 + 0.1 * H]
    cx = float(np.median(low[:, 0])) if len(low) else float(np.median(samples[:, 0]))
    # the hands: the pair of ends on opposite sides at much the same height, between the feet and the head,
    # farthest from the head
    cand = [(e, d) for e, d in found if 0.15 * H < centre(e)[2] - z0 < 0.8 * H and abs(centre(e)[0] - cx) > 0.05 * H]
    pairs = [(min(d1, d2), e1, e2) for i, (e1, d1) in enumerate(cand) for e2, d2 in cand[i + 1:]
             if (centre(e1)[0] - cx) * (centre(e2)[0] - cx) < 0
             and abs(abs(centre(e1)[0] - cx) - abs(centre(e2)[0] - cx)) < 0.15 * H
             and abs(centre(e1)[2] - centre(e2)[2]) < 0.15 * H]
    if not pairs:
        return None
    _, h1, h2 = max(pairs, key=lambda p: p[0])
    hands = [h1, h2]
    out = {}
    for hand in hands:
        side = "Left" if centre(hand)[0] > cx else "Right"
        arm = _arm_chain(np.asarray(verts, float), np.asarray(edges, int), centre(hand), H)
        if arm is None:
            return None
        out[side] = arm
    return out


def surface_samples(triangles: np.ndarray, n: int = 300_000, seed: int = 0) -> np.ndarray:
    """``n`` points spread evenly (by area) over triangles shaped (m, 3, 3)."""
    T = np.asarray(triangles, float)
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(T), n, p=area / area.sum())
    r = rng.random((n, 2))
    flip = r.sum(axis=1) > 1
    r[flip] = 1 - r[flip]
    return T[i, 0] + r[:, :1] * (T[i, 1] - T[i, 0]) + r[:, 1:] * (T[i, 2] - T[i, 0])


def surface_distance(verts: np.ndarray, edges: np.ndarray, start: int) -> np.ndarray:
    """Dijkstra along mesh edges from ``start``; inf off its island."""
    import heapq
    n = len(verts)
    length = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    order = np.argsort(np.concatenate([edges[:, 0], edges[:, 1]]), kind="stable")
    nbr = np.concatenate([edges[:, 1], edges[:, 0]])[order]
    w = np.concatenate([length, length])[order]
    first = np.searchsorted(np.concatenate([edges[:, 0], edges[:, 1]])[order], np.arange(n + 1))
    dist = np.full(n, np.inf)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, a = heapq.heappop(heap)
        if d > dist[a]:
            continue
        for k in range(first[a], first[a + 1]):
            b, nd = nbr[k], d + w[k]
            if nd < dist[b]:
                dist[b] = nd
                heapq.heappush(heap, (nd, b))
    return dist


def _arm_chain(verts: np.ndarray, edges: np.ndarray, tip_point: np.ndarray, H: float) -> dict | None:
    """The arm's centre line along the surface from the fingertip: centroids of geodesic bands, up to where a band
    spreads far wider than the arm (the armpit) or the arm's own piece of mesh ends (a separately modelled arm)."""
    tip = int(np.argmin(np.linalg.norm(verts - tip_point, axis=1)))
    if np.linalg.norm(verts[tip] - tip_point) > 0.05 * H:
        return None
    d = surface_distance(verts, edges, tip)
    reach = d[np.isfinite(d)].max()
    band = 0.02 * H
    line, spread = [], []
    for t in np.arange(0.0, reach, band):
        ring = verts[(d >= t) & (d < t + band)]
        if len(ring) < 3:
            continue
        c = ring.mean(axis=0)
        line.append(c)
        spread.append(float(np.sqrt(((ring - c) ** 2).sum(axis=1).mean())))
    if len(line) < 5:
        return None
    # the armpit: the first band far wider than the arm traced so far (the hand's first bands, wider than the
    # forearm, are not the reference); a separately modelled arm just ends
    join = len(line) - 1
    for i in range(6, len(spread)):
        if spread[i] > 2.2 * float(np.median(spread[3:i])):
            join = i
            break
    line = np.array(line[: join + 1])
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    L = arc[-1]
    if L < 0.12 * H:
        return None
    at = lambda s: np.array([np.interp(s, arc, line[:, k]) for k in range(3)])
    wrist_s = L * (1 - UPPER_ARM - FOREARM)
    return {"tip": line[0], "knuckle": at(wrist_s * (1 - KNUCKLE)), "wrist": at(wrist_s),
            "elbow": at(L * (1 - UPPER_ARM)), "shoulder": line[-1], "length": float(L)}


def islands(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component label per vertex (union-find over edges)."""
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.asarray(edges, int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


LIMB_PARTS = {"Arm": ("LowerArm", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"),
              "Leg": ("LowerLeg", "Foot", "Toes")}
END_JOINT = {"Arm": "Hand", "Leg": "Foot"}  # the head of this bone (wrist, ankle) lies inside the limb's own piece


def _cols(names: list[str], side: str, kind: str, upper: bool = False) -> list[int]:
    parts = LIMB_PARTS[kind] + ((f"Upper{kind}",) if upper else ())
    return [i for i, n in enumerate(names) if n.startswith(side) and any(k in n for k in parts)]


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def gate_by_distance(W: np.ndarray, names: list[str], points: np.ndarray, bones: dict[str, tuple],
                     ratio: float = 2.5, slack: float = 0.03) -> tuple[np.ndarray, int]:
    """Zero the weight of a bone much farther from the vertex than its nearest bone (a spike: one vertex of the
    shorts on the upper arm). Kept: within ``ratio`` x the nearest bone's distance plus ``slack`` x the height."""
    H = float(np.ptp(points[:, 2])) or 1.0
    D = np.full(W.shape, np.inf)
    for j, n in enumerate(names):
        if n in bones:
            D[:, j] = _segment_distance(points, *bones[n])
    limit = D.min(axis=1, keepdims=True) * ratio + slack * H
    bad = (W > 0) & (D > limit)
    keep = np.where(bad, 0.0, W)
    s = keep.sum(axis=1, keepdims=True)
    keep = np.where(s > 1e-9, keep / np.maximum(s, 1e-12), W)  # never leave a vertex with nothing
    return keep, int(bad.any(axis=1).sum())


def island_cleanup(W: np.ndarray, names: list[str], label: np.ndarray, points: np.ndarray,
                   heads: dict[str, np.ndarray], rigid_share: float = 0.01) -> tuple[np.ndarray, dict]:
    """Volume-based skinning bleeds across pieces that only touch: shorts follow the hand hanging beside them, an
    arm modelled apart follows the thigh it hangs against. Per mesh island (``label``):
      * a small piece (under ``rigid_share`` of the vertices and a tenth of the height across) moves rigidly with
        its island's mean weights, so an accessory keeps its shape;
      * the pieces nearest a wrist (ankle) own that limb: an arm's piece never follows a leg and a leg's never an
        arm; any other piece loses the bones below that elbow (knee) unless they carry most of its weight;
      * a vertex left with nothing takes its island's mean of what remains.
    ``heads``: bone name -> head position. Returns the new weights and counts of what changed."""
    W = W.copy()
    H = float(np.ptp(points[:, 2])) or 1.0
    stats = {"limb_bleed_removed": 0, "rigid_pieces": 0}
    labs = np.unique(label)
    pieces = {lab: np.nonzero(label == lab)[0] for lab in labs}
    limbs = [(side, kind) for side in ("Left", "Right") for kind in ("Arm", "Leg") if f"{side}{END_JOINT[kind]}" in heads]
    owners: dict[tuple[str, str], set] = {}
    for side, kind in limbs:
        j = heads[f"{side}{END_JOINT[kind]}"]
        dmin = {lab: float(np.linalg.norm(points[idx] - j, axis=1).min()) for lab, idx in pieces.items()}
        best = min(dmin.values())
        owners[(side, kind)] = {lab for lab, d in dmin.items() if d <= 1.5 * best + 0.005 * H}

    def drop(idx: np.ndarray, cols: list[int]) -> None:
        if cols and W[np.ix_(idx, cols)].any():
            stats["limb_bleed_removed"] += int((W[np.ix_(idx, cols)] > 0).any(axis=1).sum())
            W[np.ix_(idx, cols)] = 0.0

    small = {lab for lab, idx in pieces.items()
             if len(idx) < rigid_share * len(W) and np.ptp(points[idx], axis=0).max() < 0.1 * H}
    for lab, idx in pieces.items():
        if lab in small:
            continue
        before = W[idx].copy()
        owned = [k for k in limbs if lab in owners[k]]
        kinds = {kind for _, kind in owned}
        for side, kind in limbs:
            if (side, kind) in owned:
                continue
            if kinds and kind not in kinds:  # an arm's own piece and a leg: the whole other limb goes
                drop(idx, _cols(names, side, kind, upper=True))
            else:
                cols = _cols(names, side, kind)
                mass = W[idx].sum(axis=0)
                if cols and mass[cols].sum() < 0.5 * mass.sum():
                    drop(idx, cols)
        if not W[idx].any():  # nothing left to follow: keep what the skinning gave
            W[idx] = before
        s = W[idx].sum(axis=1, keepdims=True)
        mean = W[idx].sum(axis=0)
        mean = mean / mean.sum() if mean.sum() > 0 else mean
        W[idx] = np.where(s > 1e-9, W[idx] / np.maximum(s, 1e-12), mean)
    # an accessory (a bracelet, a button, a sock's rim) moves rigidly with the skin it sits on: the weights of the
    # nearest vertex of a larger piece to its middle
    big = np.concatenate([idx for lab, idx in pieces.items() if lab not in small]) if len(small) < len(pieces) else None
    for lab in small:
        idx = pieces[lab]
        if big is None:
            continue
        host = big[np.argmin(np.linalg.norm(points[big] - points[idx].mean(axis=0), axis=1))]
        W[idx] = W[host]
        stats["rigid_pieces"] += 1
    return W, stats


def smooth_weights(W: np.ndarray, edges: np.ndarray, iterations: int = 6, alpha: float = 0.5) -> np.ndarray:
    """Laplacian smoothing of the weights along the mesh: a hard step between two bones (a thigh's skin cut off
    from the hip) becomes a blend over a few edge rings. Pieces stay apart: only edges carry weight."""
    e = np.asarray(edges, int)
    n = len(W)
    deg = np.bincount(e.ravel(), minlength=n).astype(float)[:, None]
    has = deg[:, 0] > 0
    for _ in range(iterations):
        acc = np.zeros_like(W)
        np.add.at(acc, e[:, 0], W[e[:, 1]])
        np.add.at(acc, e[:, 1], W[e[:, 0]])
        avg = np.divide(acc, deg, out=W.copy(), where=deg > 0)
        W = np.where(has[:, None], (1 - alpha) * W + alpha * avg, W)
    return W
