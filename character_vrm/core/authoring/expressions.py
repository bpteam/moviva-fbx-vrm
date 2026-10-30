"""Phases 8 and 12: expression sourcing (existing wins) and basic generated expressions.

Priority per expression: explicit mapping > recognised source morph > compound from existing morphs >
generated geometry shape key > missing. Generated keys never replace an existing mapping.
"""

from __future__ import annotations

from ..anatomy.confidence import (
    DEFAULT_THRESHOLDS, STATUS_GENERATED, STATUS_MAPPED, STATUS_MISSING, STATUS_SOURCE, Thresholds,
)
from ..anatomy.model import HumanModelAnalysis
from ..anatomy.regions import Ellipse, eye_ellipse, mouth_ellipse, mouth_usability
from ..anatomy.vec import smoothstep
from ..expressions import ExpressionResult
from ..model import ExpressionMapping, MorphBind
from .blink import blink_fields, plan_blink
from .facefields import FieldBuilder, sgn
from .plans import ExpressionDecision, ShapeKeyPlan, Skipped
from .visemes import JAW_SHARE, SHAPES, VISEMES, plan_jaw_open, plan_visemes, viseme_field

TRACKED = ("blink", "blinkLeft", "blinkRight", *VISEMES, "happy", "sad", "surprised")
BASIC = ("happy", "sad", "surprised", "browUp", "browDown", "squint")  # browUp/browDown/squint have no VRM preset


def decide_existing(expressions: ExpressionResult) -> dict[str, ExpressionDecision]:
    """Where each tracked expression comes from *before* any generation."""
    out: dict[str, ExpressionDecision] = {}
    for name in TRACKED:
        m = expressions.mappings.get(name)
        if m is None:
            out[name] = ExpressionDecision(name, STATUS_MISSING)
        elif m.kind in ("direct", "config"):
            detail = ", ".join(b.shape_key for b in m.binds)
            out[name] = ExpressionDecision(name, STATUS_SOURCE, m.confidence, detail)
        else:
            out[name] = ExpressionDecision(name, STATUS_MAPPED, m.confidence,
                                           " + ".join(b.shape_key for b in m.binds) or m.reason)
    return out


def _brow_ellipse(analysis: HumanModelAnalysis, side: str) -> Ellipse | None:
    face, frame = analysis.face, analysis.frame
    b = face.local(frame, f"{side}_eyebrow") if face else None
    if b is None:
        return None
    return Ellipse(b[0], b[1], 0.17 * face.head_width, 0.05 * face.head_height, 0.05 * face.head_height,
                   b[2] - 0.30 * face.head_width)


def _cheek_ellipse(analysis: HumanModelAnalysis, side: str) -> Ellipse | None:
    face, frame = analysis.face, analysis.frame
    c = face.local(frame, f"{side}_cheek") if face else None
    if c is None:
        return None
    return Ellipse(c[0], c[1], 0.16 * face.head_width, 0.10 * face.head_height, 0.10 * face.head_height,
                   c[2] - 0.30 * face.head_width)


def _brow_roles() -> tuple[str, ...]:
    return ("skin", "brow")


def _brows(fb: FieldBuilder, analysis: HumanModelAnalysis, up: float, inner_extra: float = 0.0) -> None:
    fh = analysis.face.head_height
    for side in ("left", "right"):
        ell = _brow_ellipse(analysis, side)
        if ell is None:
            continue

        def field(dx: float, dy: float, r: float, w: float, _sign=sgn(ell.cx), _rx=ell.rx):
            # 1 at the end of the brow nearest the midline, 0 at its outer end.
            inner_w = smoothstep(0.5 - 0.5 * (dx * _sign) / _rx)
            return (0.0, (up * fh + inner_extra * fh * inner_w) * w, 0.0)

        fb.add(ell, field, _brow_roles())


def _corners(fb: FieldBuilder, analysis: HumanModelAnalysis, lift: float, spread: float) -> None:
    face = analysis.face
    ell = mouth_ellipse(face, analysis.frame)
    if ell is None:
        return
    mw = face.mouth_width

    def field(dx: float, dy: float, r: float, w: float):
        g = smoothstep(abs(dx) / (0.6 * mw))
        return (sgn(dx) * spread * mw * g * w, lift * mw * g * w, 0.0)

    fb.add(ell, field)


def _cheeks(fb: FieldBuilder, analysis: HumanModelAnalysis, raise_frac: float) -> None:
    fh = analysis.face.head_height
    for side in ("left", "right"):
        ell = _cheek_ellipse(analysis, side)
        if ell is not None:
            fb.add(ell, lambda dx, dy, r, w: (0.0, raise_frac * fh * w, 0.0))


def _jaw(fb: FieldBuilder, analysis: HumanModelAnalysis, amount: float) -> None:
    sub = viseme_field(analysis, SHAPES["aa"], scale=amount / SHAPES["aa"].jaw_open)
    if sub is not None:
        for k, d in sub.local.items():
            old = fb.local.get(k, (0.0, 0.0, 0.0))
            fb.local[k] = (old[0] + d[0], old[1] + d[1], old[2] + d[2])


def plan_basic_expressions(analysis: HumanModelAnalysis, wanted: set[str],
                           thresholds: Thresholds = DEFAULT_THRESHOLDS) -> tuple[list[ShapeKeyPlan], list[Skipped]]:
    face, frame = analysis.face, analysis.frame
    wanted = wanted & set(BASIC)
    if not wanted:
        return [], []
    if face is None:
        return [], [Skipped(w, "no face analysis") for w in sorted(wanted)]
    use = mouth_usability(face, frame)
    conf = round(0.6 * face.quality + 0.4 * use.score, 4)
    if conf < thresholds.generated_face or not use.usable:
        return [], [Skipped(w, use.reason or f"confidence {conf:.2f} below {thresholds.generated_face:.2f}", conf)
                    for w in sorted(wanted)]
    plans: list[ShapeKeyPlan] = []

    def make(name: str, expr: str | None, description: str, fb: FieldBuilder) -> None:
        if name in wanted:
            plans.append(fb.build(name, conf, description, expr))

    fb = FieldBuilder(face, frame)
    _corners(fb, analysis, lift=0.14, spread=0.08)
    _cheeks(fb, analysis, 0.03)
    make("happy", "happy", "mouth corners up and out, cheeks raised", fb)

    fb = FieldBuilder(face, frame)
    _corners(fb, analysis, lift=-0.08, spread=-0.02)
    _brows(fb, analysis, up=0.0, inner_extra=0.035)
    make("sad", "sad", "mouth corners down, inner brows up", fb)

    fb = FieldBuilder(face, frame)
    _brows(fb, analysis, up=0.05)
    _jaw(fb, analysis, 0.16)
    make("surprised", "surprised", "brows raised, jaw slightly open", fb)

    fb = FieldBuilder(face, frame)
    _brows(fb, analysis, up=0.05)
    make("browUp", None, "brows raised", fb)

    fb = FieldBuilder(face, frame)
    _brows(fb, analysis, up=-0.035)
    make("browDown", None, "brows lowered", fb)

    fb = FieldBuilder(face, frame)
    for side in ("left", "right"):
        sub = blink_fields(analysis, side, closure=0.4)
        if sub is not None:
            for k, d in sub.local.items():
                old = fb.local.get(k, (0.0, 0.0, 0.0))
                fb.local[k] = (old[0] + d[0], old[1] + d[1], old[2] + d[2])
    _cheeks(fb, analysis, 0.02)
    make("squint", None, "eyelids narrowed, cheeks raised", fb)
    return plans, []


def bind_generated(plans: list[ShapeKeyPlan]) -> dict[str, ExpressionMapping]:
    """VRM expression -> mapping onto generated keys (only presets: browUp/browDown/squint stay unbound)."""
    by_expr: dict[str, list[ShapeKeyPlan]] = {}
    for p in plans:
        if p.vrm_expression:
            by_expr.setdefault(p.vrm_expression, []).append(p)
    out: dict[str, ExpressionMapping] = {}
    for expr, group in by_expr.items():
        binds = [MorphBind(mesh, p.name, 1.0) for p in group for mesh in p.meshes]
        confidence = min(p.confidence for p in group)
        out[expr] = ExpressionMapping(expr, "generated-geometry", binds, confidence,
                                      f"generated from face geometry ({', '.join(p.name for p in group)})")
    return out


def plan_face_authoring(analysis: HumanModelAnalysis, expressions: ExpressionResult, *, generate_visemes_blink: bool,
                        generate_basic: bool, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> "FaceAuthoringPlan":
    from .face_plan import FaceAuthoringPlan
    decisions = decide_existing(expressions)
    plan = FaceAuthoringPlan(decisions=decisions)
    need_blink = (
        "blink" not in expressions.mappings and not ("blinkLeft" in expressions.mappings and "blinkRight" in expressions.mappings))
    if generate_visemes_blink:
        if need_blink:
            keys, skipped = plan_blink(analysis, thresholds)
            plan.shape_keys.extend(keys)
            plan.skipped.extend(skipped)
        missing_vis = {v for v in VISEMES if v not in expressions.mappings}
        keys, skipped = plan_visemes(analysis, missing_vis, thresholds)
        plan.shape_keys.extend(keys)
        plan.skipped.extend(skipped)
        if "aa" in expressions.mappings and analysis.face is not None:
            # existing visemes that shape the lips but leave the jaw shut (the rig's jaw bone was meant to open it)
            jaw, jaw_skip = plan_jaw_open(analysis, analysis.face.aa_opening, thresholds)
            if jaw_skip is not None:
                plan.skipped.append(jaw_skip)
            if jaw is not None:
                plan.shape_keys.append(jaw)
                for expr, share in JAW_SHARE.items():
                    if expr in expressions.mappings:
                        plan.augment.setdefault(expr, []).append((jaw.name, share))
                        d = decisions[expr]
                        decisions[expr] = ExpressionDecision(expr, d.status, d.confidence,
                                                             f"{d.detail} + {jaw.name} x{share}")
    else:
        for name in ("blinkLeft", "blinkRight", *VISEMES):
            if name not in expressions.mappings:
                plan.skipped.append(Skipped(name, "generation disabled"))
    if generate_basic:
        wanted = {n for n in ("happy", "sad", "surprised") if n not in expressions.mappings} | {"browUp", "browDown", "squint"}
        keys, skipped = plan_basic_expressions(analysis, wanted, thresholds)
        plan.shape_keys.extend(keys)
        plan.skipped.extend(skipped)
    for key in plan.shape_keys:
        if key.vrm_expression and key.vrm_expression in decisions and decisions[key.vrm_expression].status == STATUS_MISSING:
            decisions[key.vrm_expression] = ExpressionDecision(
                key.vrm_expression, STATUS_GENERATED, key.confidence, key.name)
    # A generated blink also drives the combined ``blink`` preset when nothing else does.
    if any(k.vrm_expression in ("blinkLeft", "blinkRight") for k in plan.shape_keys) and "blink" not in expressions.mappings:
        lr = [k for k in plan.shape_keys if k.vrm_expression in ("blinkLeft", "blinkRight")]
        if len(lr) == 2:
            for k in lr:
                plan.extra_binds.setdefault("blink", []).append(k)
            decisions["blink"] = ExpressionDecision("blink", STATUS_GENERATED, min(k.confidence for k in lr),
                                                    " + ".join(k.name for k in lr))
    return plan
