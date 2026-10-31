"""Phase 9: generated blink from detected eyelid regions (``Prosopon_BlinkLeft`` / ``Prosopon_BlinkRight``)."""

from __future__ import annotations

from ..anatomy.confidence import DEFAULT_THRESHOLDS, Thresholds
from ..anatomy.model import HumanModelAnalysis
from ..anatomy.regions import eye_ellipse, eye_usability
from ..anatomy.vec import smoothstep
from .facefields import FieldBuilder
from .plans import ShapeKeyPlan, Skipped

UPPER_SHARE = 0.75   # the upper lid does most of the closing
LOWER_SHARE = 0.25


def blink_fields(analysis: HumanModelAnalysis, side: str, closure: float = 1.0, name: str | None = None) -> FieldBuilder | None:
    """The blink deformation of one eye as a field (``closure`` 1 = fully closed). Reused by ``squint``."""
    face, frame = analysis.face, analysis.frame
    if face is None:
        return None
    ell = eye_ellipse(face, frame, side)
    up, low = face.local(frame, f"{side}_upper_eyelid"), face.local(frame, f"{side}_lower_eyelid")
    if ell is None or up is None or low is None:
        return None
    radius = 0.11 * face.head_width  # eyeball radius estimate from head width
    box = face.eye_extent.get(side)
    half = 0.5 * (box[1] - box[0]) if box is not None else 0.6 * ell.rx  # the eye opening's half width
    top, bottom = up[1] - ell.cy, low[1] - ell.cy  # the lid margins over the eye's middle
    fb = FieldBuilder(face, frame)

    def field(dx: float, dy: float, r: float, w: float):
        # Each lid margin travels the whole way to where the lids meet (a quarter of the gap up from the lower
        # margin), following the eye's almond: the gap narrows to nothing at the corners. Skin past a margin
        # follows less the farther it is, to nothing halfway to the region's edge (the brow stays, the cheek too).
        t = max(0.0, 1.0 - (dx / half) ** 2) ** 0.5 if half > 0 else 0.0
        yu, yl = top * t, bottom * t
        meet = yl + LOWER_SHARE * (yu - yl)
        if dy >= 0:
            reach = (meet - yu) * closure
            fade = 1.0 - smoothstep(max(0.0, dy - yu) / max(0.5 * (ell.ry_up - yu), 1e-9))
        else:
            reach = (meet - yl) * closure
            fade = 1.0 - smoothstep(max(0.0, yl - dy) / max(0.5 * (ell.ry_down + yl), 1e-9))
        move = reach * fade * (1.0 - smoothstep(max(0.0, abs(dx) - half) / max(ell.rx - half, 1e-9)))
        y0, y1 = dy, dy + move
        dz = max(0.0, (y0 * y0 - y1 * y1) / (2 * radius))  # lids ride over the eyeball's curvature
        return (0.0, move, dz)

    fb.add(ell, field)
    return fb


def plan_blink(analysis: HumanModelAnalysis, thresholds: Thresholds = DEFAULT_THRESHOLDS,
               ) -> tuple[list[ShapeKeyPlan], list[Skipped]]:
    face, frame = analysis.face, analysis.frame
    if face is None:
        return [], [Skipped("blink", "no face analysis (not enough facial geometry)")]
    plans: list[ShapeKeyPlan] = []
    skipped: list[Skipped] = []
    usab = {s: eye_usability(face, frame, s) for s in ("left", "right")}
    for side, key, expr in (("left", "BlinkLeft", "blinkLeft"), ("right", "BlinkRight", "blinkRight")):
        u = usab[side]
        confidence = round(0.6 * face.quality + 0.4 * u.score, 4)
        if not u.usable:
            skipped.append(Skipped(expr, u.reason, confidence))
            continue
        if confidence < thresholds.generated_face:
            skipped.append(Skipped(expr, f"confidence {confidence:.2f} below {thresholds.generated_face:.2f}", confidence))
            continue
        fb = blink_fields(analysis, side)
        if fb is None:
            skipped.append(Skipped(expr, "eyelid landmarks missing", confidence))
            continue
        plans.append(fb.build(key, confidence, f"{side} upper/lower eyelids close over the eyeball", expr))
    return plans, skipped
