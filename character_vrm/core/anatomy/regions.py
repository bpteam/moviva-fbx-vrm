"""Phase 9/10: eyelid and mouth region extraction and geometry-usability checks (pure, on ``FaceGeometry``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .face import FaceAnalysis, FacePoint
from .frame import AnatomicalFrame
from .vec import Vec3

MIN_EYE_VERTICES = 40
MIN_EYE_ROWS = 5
MIN_MOUTH_VERTICES = 80
MIN_MOUTH_ROWS = 6


@dataclass
class Ellipse:
    """Elliptical front region in the frame's (left, up) plane."""

    cx: float
    cy: float
    rx: float
    ry_up: float
    ry_down: float
    z_min: float

    def radius(self, p: Vec3) -> float:
        dy = p[1] - self.cy
        ry = self.ry_up if dy >= 0 else self.ry_down
        return ((p[0] - self.cx) / self.rx) ** 2 + (dy / ry) ** 2 if self.rx > 0 and ry > 0 else 9.0

    def contains(self, p: Vec3) -> bool:
        return self.radius(p) <= 1.0 and p[2] >= self.z_min


def eye_ellipse(face: FaceAnalysis, frame: AnatomicalFrame, side: str) -> Ellipse | None:
    up, low = face.local(frame, f"{side}_upper_eyelid"), face.local(frame, f"{side}_lower_eyelid")
    if up is None or low is None:
        return None
    cx, cy = (up[0] + low[0]) / 2, (up[1] + low[1]) / 2
    z = min(up[2], low[2])
    rx, ry_up, ry_down = 0.13 * face.head_width, 0.055 * face.head_height, 0.055 * face.head_height
    box = face.eye_extent.get(side)
    if box is not None:
        # the measured eye: stylised eyes are far larger than a head-width fraction
        rx = max(rx, 0.6 * (box[1] - box[0]))
        ry_up = max(ry_up, 1.25 * (max(box[3], up[1]) - cy))
        ry_down = max(ry_down, 1.25 * (cy - min(box[2], low[1])))
    brow = face.brow_bottom.get(side)
    if brow is None:
        b = face.local(frame, f"{side}_eyebrow")
        brow = b[1] - 0.02 * face.head_height if b else None
    if brow is not None and brow > cy:
        ry_up = min(ry_up, 0.9 * (brow - cy))  # the lid closes; the brow stays
    return Ellipse(cx, cy, rx, ry_up, ry_down, z - 0.30 * face.head_width)


def mouth_ellipse(face: FaceAnalysis, frame: AnatomicalFrame) -> Ellipse | None:
    m = face.local(frame, "mouth_center")
    chin = face.local(frame, "chin")
    if m is None or chin is None or face.mouth_width <= 0:
        return None
    down = max((m[1] - chin[1]) * 0.95, 0.05 * face.head_height)
    return Ellipse(m[0], m[1], 0.75 * face.mouth_width, 0.16 * face.head_height, down, m[2] - 0.30 * face.head_width)


def points_in(face: FaceAnalysis, region: Ellipse, roles: tuple[str, ...] = ("skin",)) -> list[FacePoint]:
    return [p for p in face.geometry.points if p.role in roles and region.contains(p.local)]


def rows(points: list[FacePoint], step: float) -> int:
    return len({round(p.local[1] / step) for p in points}) if step > 0 else 0


@dataclass
class RegionUsability:
    vertices: int
    rows: int
    usable: bool
    score: float
    reason: str = ""


def eye_usability(face: FaceAnalysis, frame: AnatomicalFrame, side: str) -> RegionUsability:
    ell = eye_ellipse(face, frame, side)
    if ell is None:
        return RegionUsability(0, 0, False, 0.0, "no eyelid landmarks")
    pts = points_in(face, ell)
    r = rows(pts, 0.012 * face.head_height)
    score = min(1.0, len(pts) / (2.0 * MIN_EYE_VERTICES)) * 0.5 + min(1.0, r / (2.0 * MIN_EYE_ROWS)) * 0.5
    ok = len(pts) >= MIN_EYE_VERTICES and r >= MIN_EYE_ROWS
    return RegionUsability(len(pts), r, ok, round(score, 3), "" if ok else "eyelid region has too little geometry")


def mouth_usability(face: FaceAnalysis, frame: AnatomicalFrame) -> RegionUsability:
    ell = mouth_ellipse(face, frame)
    if ell is None:
        return RegionUsability(0, 0, False, 0.0, "no mouth landmarks")
    pts = points_in(face, ell)
    r = rows(pts, 0.008 * face.head_height)
    score = min(1.0, len(pts) / (2.0 * MIN_MOUTH_VERTICES)) * 0.5 + min(1.0, r / (2.0 * MIN_MOUTH_ROWS)) * 0.5
    ok = len(pts) >= MIN_MOUTH_VERTICES and r >= MIN_MOUTH_ROWS
    return RegionUsability(len(pts), r, ok, round(score, 3),
                           "" if ok else "mouth region has too little geometry (flat textured mouth?)")


@dataclass
class MouthAnalysis:
    """What can drive mouth animation for this model."""

    jaw_bone: bool
    mouth_morphs: list[str] = field(default_factory=list)
    geometry: RegionUsability | None = None
    inner_mouth_meshes: list[str] = field(default_factory=list)
    approaches: list[str] = field(default_factory=list)


def analyze_mouth(face: FaceAnalysis, frame: AnatomicalFrame, existing_visemes: list[str]) -> MouthAnalysis:
    geo = mouth_usability(face, frame)
    out = MouthAnalysis(jaw_bone=face.jaw_bone, mouth_morphs=list(existing_visemes), geometry=geo,
                        inner_mouth_meshes=sorted({p.mesh for p in face.geometry.points if p.role == "teeth"}))
    if existing_visemes:
        out.approaches.append("mouth-morphs")
    if face.jaw_bone:
        out.approaches.append("jaw-bone")
    if geo.usable:
        out.approaches.append("geometry-deformation")
    return out
