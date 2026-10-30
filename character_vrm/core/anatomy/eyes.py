"""Eye centres and eyelids (pure): eye geometry over eye joints; blink morph, skin opening, visible eye surface."""

from __future__ import annotations

from .facegeo import FaceGeometry, FaceMorphs, local_deltas, moved_vertices, skin_loops
from .frame import AnatomicalFrame
from .surfaces import EYE, LASH
from .vec import Vec3

def find_eyes(geometry: FaceGeometry, eye_positions: dict[str, Vec3] | None, frame: AnatomicalFrame,
          head_width: float) -> tuple[dict[str, Vec3], str]:
    """Eye centres in local space: the eye geometry (per eye bone, else eye surfaces split by side)."""
    eye_pts = [p for p in geometry.points if p.role == EYE]
    joints = {s[:-4]: frame.to_local(pos) for s, pos in (eye_positions or {}).items()}
    out: dict[str, Vec3] = {}
    if eye_pts:
        mid = 0.5 * (joints["left"][0] + joints["right"][0]) if len(joints) == 2 else \
            sum(p.local[0] for p in eye_pts) / len(eye_pts)
        for s, sign in (("left", 1.0), ("right", -1.0)):
            pts = [p.local for p in eye_pts if p.eye_side == s]
            if len(pts) < 8:
                pts = [p.local for p in eye_pts if (p.local[0] - mid) * sign > 0.02 * head_width]
            if len(pts) >= 8:
                out[s] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts),
                          sum(p[2] for p in pts) / len(pts))
        if len(out) == 2:
            return out, "eye-geometry"
    if len(joints) == 2:
        return joints, "eye-bone"
    return out, "eye-geometry" if out else "template"


def eye_extents(geometry: FaceGeometry, eyes: dict[str, Vec3], x_mid: float) -> dict[str, tuple[float, float, float, float]]:
    """side -> (x min, x max, y min, y max) of that eye's surfaces (3rd-97th percentile): the size of the eye, which
    stylised faces make far larger than a head-width fraction."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for s, sign in (("left", 1.0), ("right", -1.0)):
        if s not in eyes:
            continue
        pts = [p.local for p in geometry.points if p.role == EYE and (p.eye_side == s or
               (p.eye_side is None and (p.local[0] - x_mid) * sign > 0))]
        if len(pts) < 8:
            continue
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        lo, hi = int(0.03 * (len(xs) - 1)), int(0.97 * (len(xs) - 1))
        out[s] = (xs[lo], xs[hi], ys[lo], ys[hi])
    return out


def eyelids(geometry: FaceGeometry, by_key, frame, morphs: FaceMorphs, eyes: dict[str, Vec3], head_width, span,
             x_mid: float):
    """side -> (x, upper lid y, lower lid y, method, confidence), from a blink morph, the skin's eye opening, or the
    visible eye surface."""
    out: dict[str, tuple] = {}
    for s, sign in (("left", 1.0), ("right", -1.0)):
        if s not in eyes:
            continue
        ex, ey, ez = eyes[s]
        # (1) blink morph: the upper lid is what moves down
        d = morphs.get("blinkLeft" if s == "left" else "blinkRight") or \
            {k: v for k, v in morphs.get("blink").items() if k in by_key and (by_key[k].local[0] - x_mid) * sign > 0}
        if d:
            dl = local_deltas(frame, d)
            moved = [by_key[k] for k in moved_vertices(d, 0.3) if k in by_key and by_key[k].role != EYE
                     and abs(by_key[k].local[0] - ex) < 0.3 * head_width]
            down = [p for p in moved if dl[(p.mesh, p.index)][1] < 0]
            if len(down) >= 6:
                ux = sum(p.local[0] for p in down) / len(down)
                uy = sum(p.local[1] for p in down) / len(down)
                ups = [p for p in moved if dl[(p.mesh, p.index)][1] > 0]
                ly = sum(p.local[1] for p in ups) / len(ups) if len(ups) >= 4 else ey - (uy - ey)
                out[s] = (ux, uy, min(ly, uy - 0.02 * span), "blink-morph", 0.95,
                          sum(p.local[2] for p in down) / len(down))
                continue
        # (2) the eye opening in the skin
        # stylised eyes draw the upper lid as a lash / eyeliner strip above the skin's opening: it is the lid
        lash = sorted(p.local[1] for p in geometry.points if p.role == LASH and p.local[1] >= ey
                      and abs(p.local[0] - ex) < 0.15 * head_width and (p.local[0] - x_mid) * sign > 0)
        lash_y = lash[len(lash) // 2] if len(lash) >= 6 else None
        loop = _eye_loop(geometry, by_key, ex, ey, head_width)
        if loop is not None:
            lx, up, low = loop
            if lash_y is not None and lash_y > up:
                out[s] = (lx, lash_y, low, "eye-opening+lash", 0.85, None)
            else:
                out[s] = (lx, up, low, "eye-opening", 0.88, None)
            continue
        # (3) the visible eye surface (front-most eye vertices)
        pts = [p.local for p in geometry.points if p.role == EYE and abs(p.local[0] - ex) < 0.2 * head_width
               and (p.local[0] - x_mid) * sign > 0]
        if len(pts) >= 8:
            zs = sorted(p[2] for p in pts)
            vis = [p for p in pts if p[2] >= zs[int(0.5 * (len(zs) - 1))]]
            ys = sorted(p[1] for p in vis)
            out[s] = (ex, ys[int(0.9 * (len(ys) - 1))], ys[int(0.1 * (len(ys) - 1))], "eye-surface", 0.7, None)
    return out


def _eye_loop(geometry: FaceGeometry, by_key, ex, ey, head_width) -> tuple[float, float, float] | None:
    """The skin opening around one eye: a loop that encloses the eye centre, the tightest one."""
    best = None
    for name, pts, own in skin_loops(geometry, by_key, head_width):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if not (min(xs) < ex < max(xs) and min(ys) - 0.1 * h < ey < max(ys) + 0.1 * h) or w > 0.45 * head_width:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        up = [p[1] for p in pts if p[1] >= cy]
        low = [p[1] for p in pts if p[1] < cy]
        if not up or not low:
            continue
        area = w * h
        if best is None or area < best[0]:
            best = (area, (cx, sum(up) / len(up), sum(low) / len(low)))
    return best[1] if best else None
