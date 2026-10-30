"""Phase 11: generated visemes ``Prosopon_aa/ih/ou/ee/oh`` from mouth geometry, scaled by mouth width and head size.

The jaw part of a viseme is a *rotation* about the jaw joint wherever the rig has a jaw bone that carries the lower
lip (``FaceAnalysis.jaw_weighted_lips``): every face vertex turns by its own jaw weight, exactly as linear-blend
skinning would move it, so the lower lip, chin and lower teeth open together and the upper lip stays. Without a jaw
bone the lower face is separated from the upper at the measured lip seam and translated. The lip shaping (spread,
pucker, protrusion, upper-lip lift) is a local field around the mouth in both cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..anatomy.confidence import DEFAULT_THRESHOLDS, Thresholds
from ..anatomy.model import HumanModelAnalysis
from ..anatomy.regions import mouth_ellipse, mouth_usability
from ..anatomy.vec import smoothstep
from .facefields import FieldBuilder, sgn
from .plans import ShapeKeyPlan, Skipped

VISEMES = ("aa", "ih", "ou", "ee", "oh")
MAX_JAW_HEAD_FRACTION = 0.16  # the jaw never opens more than this fraction of the head height
JAW_OPEN_KEY = "JawOpen"      # the generated jaw rotation for jaw-rigged models whose own visemes do not open it


@dataclass(frozen=True)
class VisemeShape:
    """Every amplitude is a fraction of the measured mouth width."""

    jaw_open: float
    upper_lift: float = 0.0
    spread: float = 0.0       # corners move outwards
    corner_lift: float = 0.0
    pucker: float = 0.0       # corners move inwards
    protrude: float = 0.0     # lips move forward


SHAPES: dict[str, VisemeShape] = {
    "aa": VisemeShape(jaw_open=0.38, upper_lift=0.03, spread=0.03),                       # jaw open, tall opening
    "ih": VisemeShape(jaw_open=0.12, upper_lift=0.02, spread=0.14, corner_lift=0.02),     # wider corners, moderate
    "ou": VisemeShape(jaw_open=0.10, pucker=0.24, protrude=0.16),                         # rounded, pursed, forward
    "ee": VisemeShape(jaw_open=0.06, spread=0.20, corner_lift=0.03),                      # wide, small vertical
    "oh": VisemeShape(jaw_open=0.26, pucker=0.14, protrude=0.09),                         # round, more jaw than ou
}
# ``Prosopon_JawOpen`` turns the jaw by an anatomical angle, not by a share of the mouth width: jaw rotation is what
# speech physiology measures, it does not depend on how the mouth was measured, and it scales with any head. Speech
# opens the incisors about 10-20 mm on open vowels (6-12 degrees about the jaw joint), the maximal opening is 40-50 mm
# (about 30 degrees). Lip sync drives "aa" to about 0.8 at its loudest and 0.3-0.5 in running speech, so the key at
# weight 1 is a wide spoken /a/ and running speech lands in that 6-12 degree band.
JAW_OPEN_DEG = 20.0
# Share of that rotation each existing viseme is augmented with: open vowels open the jaw, rounded ones less, spread
# ones least (see ``plan_jaw_open``).
JAW_SHARE = {"aa": 1.0, "oh": 0.75, "ou": 0.35, "ih": 0.35, "ee": 0.25}


def _jaw_drop(analysis: HumanModelAnalysis, amount: float) -> float:
    face = analysis.face
    return min(amount * face.mouth_width, MAX_JAW_HEAD_FRACTION * face.head_height)


def jaw_lever(analysis: HumanModelAnalysis) -> float | None:
    """Distance from the jaw joint to the lip seam in the side view (None without a jaw joint or a mouth)."""
    face = analysis.face
    if face is None or face.jaw_pivot is None:
        return None
    lip = face.local(analysis.frame, "mouth_center")
    if lip is None:
        return None
    lever = math.hypot(lip[1] - face.jaw_pivot[1], lip[2] - face.jaw_pivot[2])
    return lever if lever > 1e-6 else None


def jaw_rotation(fb: FieldBuilder, analysis: HumanModelAnalysis, drop: float | None = None,
                 angle: float | None = None) -> bool:
    """Add the jaw opening as a rotation about the jaw joint, weighted per vertex by its jaw skin weight. Either
    ``angle`` (radians) or ``drop``, the wanted downward travel of the lower lip at the seam. False when the rig gives
    nothing to rotate."""
    face, frame = analysis.face, analysis.frame
    if face is None or face.jaw_pivot is None or face.lip_seam is None or not face.jaw_weighted_lips:
        return False
    px, py, pz = face.jaw_pivot
    lever = jaw_lever(analysis)
    if lever is None:
        return False
    if angle is None:
        # the lip seam point turns on a circle about the pivot: the angle that brings it down by ``drop``
        angle = math.asin(min(0.95, (drop or 0.0) / lever))
    angle = min(angle, math.radians(25.0))
    c, s = math.cos(angle), math.sin(angle)
    moved = False
    for p in face.geometry.points:
        if p.jaw <= 0.0 or p.role not in ("skin", "teeth", "lash"):
            continue
        y, z = p.local[1] - py, p.local[2] - pz
        ny, nz = y * c - z * s, y * s + z * c  # about the lateral axis: the front of the jaw goes down
        d = (0.0, p.jaw * (ny - y), p.jaw * (nz - z))
        key = (p.mesh, p.index)
        old = fb.local.get(key, (0.0, 0.0, 0.0))
        fb.local[key] = (old[0] + d[0], old[1] + d[1], old[2] + d[2])
        moved = True
    return moved


def viseme_field(analysis: HumanModelAnalysis, shape: VisemeShape, scale: float = 1.0) -> FieldBuilder | None:
    face, frame = analysis.face, analysis.frame
    if face is None:
        return None
    ell = mouth_ellipse(face, frame)
    if ell is None:
        return None
    mw = face.mouth_width
    jaw = _jaw_drop(analysis, shape.jaw_open) * scale
    band = 0.02 * face.head_height  # soft transition across the mouth line: no tearing between the lips
    fb = FieldBuilder(face, frame)
    rotated = jaw > 0 and jaw_rotation(fb, analysis, jaw)

    def field(dx: float, dy: float, r: float, w: float):
        lower = smoothstep((band - dy) / (2 * band))        # 0 above the mouth line, 1 below
        g_out = smoothstep(abs(dx) / (0.6 * mw))            # 0 at the centre, 1 at/after the corners
        drop = 0.0 if rotated else -jaw * lower * w
        ddy = drop + shape.upper_lift * mw * scale * (1 - lower) * w + shape.corner_lift * mw * scale * g_out * w
        ddx = sgn(dx) * (shape.spread - shape.pucker) * mw * scale * g_out * w
        ddz = shape.protrude * mw * scale * (1 - 0.4 * g_out) * w
        return (ddx, ddy, ddz)

    fb.add(ell, field)
    return fb


def plan_visemes(analysis: HumanModelAnalysis, missing: set[str], thresholds: Thresholds = DEFAULT_THRESHOLDS,
                 ) -> tuple[list[ShapeKeyPlan], list[Skipped]]:
    wanted = [v for v in VISEMES if v in missing]
    if not wanted:
        return [], []
    face, frame = analysis.face, analysis.frame
    if face is None:
        return [], [Skipped(v, "no face analysis (not enough facial geometry)") for v in wanted]
    use = mouth_usability(face, frame)
    confidence = round(0.5 * face.quality + 0.5 * use.score, 4)
    if not use.usable:
        return [], [Skipped(v, use.reason, confidence) for v in wanted]
    if confidence < thresholds.generated_face:
        return [], [Skipped(v, f"confidence {confidence:.2f} below {thresholds.generated_face:.2f}", confidence)
                    for v in wanted]
    plans: list[ShapeKeyPlan] = []
    how = "jaw rotation + lip field" if face.jaw_weighted_lips and face.jaw_pivot is not None else "lip field"
    for v in wanted:
        fb = viseme_field(analysis, SHAPES[v])
        if fb is None:
            continue
        plans.append(fb.build(v, confidence, f"mouth geometry deformation for '{v}' ({how})", v))
    return plans, []


def plan_jaw_open(analysis: HumanModelAnalysis, existing_opening: float | None,
                  thresholds: Thresholds = DEFAULT_THRESHOLDS) -> tuple[ShapeKeyPlan | None, Skipped | None]:
    """``Prosopon_JawOpen`` for a jaw-rigged model whose *existing* visemes barely open the mouth (Character Creator:
    its visemes are lip correctives that expect the jaw bone to turn, which a VRM expression cannot do).

    ``existing_opening`` is how far the existing ``aa`` pulls the lower lip down at the seam (source units; None when
    there is no ``aa``). The key is planned only when that is under a quarter of the opening a generated ``aa`` gives."""
    face = analysis.face
    lever = jaw_lever(analysis)
    if face is None or not face.jaw_weighted_lips or lever is None:
        return None, None
    angle = math.radians(JAW_OPEN_DEG)
    want = lever * math.sin(angle)
    if existing_opening is None or existing_opening >= 0.25 * want:
        return None, None
    confidence = round(0.5 * face.quality + 0.5 * mouth_usability(face, analysis.frame).score, 4)
    if confidence < thresholds.generated_face:
        return None, Skipped(JAW_OPEN_KEY, f"confidence {confidence:.2f} below {thresholds.generated_face:.2f}",
                             confidence)
    fb = FieldBuilder(face, analysis.frame)
    if not jaw_rotation(fb, analysis, angle=angle):
        return None, None
    return fb.build(JAW_OPEN_KEY, confidence, f"jaw rotation {JAW_OPEN_DEG:g} deg about the jaw joint (existing aa opens "
                                              f"{existing_opening:.4f} of {want:.4f})", None), None
