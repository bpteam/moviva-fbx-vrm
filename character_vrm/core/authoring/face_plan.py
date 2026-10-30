"""``FaceAuthoringPlan``: what the face stage would do, inspectable before anything is mutated."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..anatomy.model import HumanModelAnalysis
from ..anatomy.vec import dist
from ..model import ExpressionMapping, MorphBind
from .expressions import bind_generated
from .plans import PROSOPON_PREFIX, ExpressionDecision, PlanIssue, ShapeKeyPlan, Skipped

# Vertices near these landmarks must not move (per key family), as a fraction of head height.
_PROTECT = {
    "blink": ("left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_cheek", "right_cheek", "mouth_center"),
    "viseme": ("left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose_bridge"),
}
PROTECT_RADIUS = 0.04
PROTECT_TOLERANCE = 0.01
MAX_OFFSET_HEAD_FRACTION = 0.25


@dataclass
class FaceAuthoringPlan:
    shape_keys: list[ShapeKeyPlan] = field(default_factory=list)
    decisions: dict[str, ExpressionDecision] = field(default_factory=dict)
    skipped: list[Skipped] = field(default_factory=list)
    extra_binds: dict[str, list[ShapeKeyPlan]] = field(default_factory=dict)  # e.g. blink -> [BlinkLeft, BlinkRight]
    # generated keys added *on top of* an existing mapping: expr -> [(key name, weight)] (the jaw of a CC viseme)
    augment: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    issues: list[PlanIssue] = field(default_factory=list)

    def expression_mappings(self) -> dict[str, ExpressionMapping]:
        maps = bind_generated([k for k in self.shape_keys if k.name in self._valid_names()])
        for expr, keys in self.extra_binds.items():
            keys = [k for k in keys if k.name in self._valid_names()]
            if keys:
                maps[expr] = ExpressionMapping(
                    expr, "generated-geometry", [MorphBind(m, k.name, 1.0) for k in keys for m in k.meshes],
                    min(k.confidence for k in keys), f"generated from face geometry ({', '.join(k.name for k in keys)})")
        return maps

    def _valid_names(self) -> set[str]:
        bad = {i.plan for i in self.issues}
        return {k.name for k in self.shape_keys if k.name not in bad}

    def validate(self, analysis: HumanModelAnalysis) -> list[PlanIssue]:
        """Reject keys that would tear the face: too large, wrong namespace, or touching protected anatomy."""
        self.issues = []
        face, frame = analysis.face, analysis.frame
        if face is None:
            return self.issues
        local = {(p.mesh, p.index): p for p in face.geometry.points}
        for key in self.shape_keys:
            if not key.name.startswith(PROSOPON_PREFIX):
                self.issues.append(PlanIssue(key.name, "generated key outside the Prosopon_ namespace"))
                continue
            if not key.offsets:
                self.issues.append(PlanIssue(key.name, "empty deformation"))
                continue
            if key.max_offset > MAX_OFFSET_HEAD_FRACTION * face.head_height:
                self.issues.append(PlanIssue(key.name, f"offset {key.max_offset:.4f} exceeds "
                                             f"{MAX_OFFSET_HEAD_FRACTION:.2f} x head height"))
            if any(local[(m, i)].role in ("eye", "hair") for m, vs in key.offsets.items() for i in vs if (m, i) in local):
                self.issues.append(PlanIssue(key.name, "moves eyeball or hair geometry"))
            family = "blink" if key.name.lower().startswith(PROSOPON_PREFIX.lower() + "blink") else \
                "viseme" if key.vrm_expression in ("aa", "ih", "ou", "ee", "oh") or key.name.endswith("JawOpen") \
                else None
            if family:
                for lm_name in _PROTECT[family]:
                    lm = face.landmarks.get(lm_name)
                    if lm is None:
                        continue
                    c = frame.to_local(lm.position)
                    for m, vs in key.offsets.items():
                        for i, d in vs.items():
                            p = local.get((m, i))
                            if p is None:
                                continue
                            if dist((p.local[0], p.local[1], 0.0), (c[0], c[1], 0.0)) < PROTECT_RADIUS * face.head_height:
                                if dist(d, (0.0, 0.0, 0.0)) > PROTECT_TOLERANCE * face.head_height:
                                    self.issues.append(PlanIssue(key.name, f"deforms {lm_name} area"))
                                    break
                        else:
                            continue
                        break
        return self.issues
