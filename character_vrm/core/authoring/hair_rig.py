"""Phase 13: generated hair rig (``--generate-hair-rig``), confidence-gated and deliberately narrow.

Targets clear long clumps / ponytails / strands. Ambiguous alpha-card masses stay static: unstable physics is worse
than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..anatomy.confidence import DEFAULT_THRESHOLDS, Thresholds
from ..anatomy.hair import ALPHA_CARDS, CLUMP, HAIR_CONFIDENT, PONYTAIL, BRAID, connected_components
from ..anatomy.model import HumanModelAnalysis
from ..anatomy.vec import Vec3, dot, mean, principal_axis, scale, smoothstep, sub
from ..model import CharacterInspection, MeshInfo
from .fingers import NewBone
from .plans import PROSOPON_PREFIX, PlanIssue, Skipped

MIN_BONES, MAX_BONES = 3, 8
TARGET_TYPES = (CLUMP, PONYTAIL, BRAID)


@dataclass
class HairRigChain:
    name: str
    mesh: str
    bones: list[NewBone]
    category: str
    confidence: float


@dataclass
class HairRigPlan:
    chains: list[HairRigChain] = field(default_factory=list)
    weights: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)  # mesh -> vertex -> bone -> weight
    skipped: list[Skipped] = field(default_factory=list)
    parent_bone: str = ""

    def validate(self) -> list[PlanIssue]:
        issues: list[PlanIssue] = []
        for c in self.chains:
            if not (MIN_BONES <= len(c.bones) <= MAX_BONES):
                issues.append(PlanIssue("hair", f"{c.name}: {len(c.bones)} bones outside {MIN_BONES}-{MAX_BONES}"))
            if any(not b.name.startswith(PROSOPON_PREFIX) for b in c.bones):
                issues.append(PlanIssue("hair", f"{c.name}: bone outside the Prosopon_ namespace"))
        for mesh, verts in self.weights.items():
            for v, ws in verts.items():
                if abs(sum(ws.values()) - 1.0) > 1e-6 and sum(ws.values()) > 1.0 + 1e-6:
                    issues.append(PlanIssue("hair", f"{mesh}[{v}] weights exceed 1"))
        return issues


def plan_hair_rig(analysis: HumanModelAnalysis, inspection: CharacterInspection, armature_name: str,
                  enabled: bool, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> HairRigPlan:
    plan = HairRigPlan(parent_bone=analysis.humanoid.get("head", ""))
    hair = analysis.hair
    if hair is None:
        return plan
    if hair.chains:
        plan.skipped.append(Skipped("hair rig", "usable hair bones already exist"))
        return plan
    targets = hair.confident_meshes
    if not targets:
        return plan
    if not enabled:
        plan.skipped.append(Skipped("hair rig", "feature disabled (--generate-hair-rig)", max(m.confidence for m in targets)))
        return plan
    head_local = analysis.frame.to_local(analysis.joints["head"]) if "head" in analysis.joints else None
    head_h = analysis.head_height or 0.0
    for assessment in targets:
        mesh = next((m for m in inspection.meshes if m.name == assessment.mesh), None)
        if mesh is None or not mesh.vertex_positions:
            continue
        if assessment.hair_type not in TARGET_TYPES:
            plan.skipped.append(Skipped(f"hair {mesh.name}", f"hair type '{assessment.hair_type}' is not rigged "
                                        "(only clear clumps / ponytails / braids)", assessment.confidence))
            continue
        if assessment.confidence < thresholds.generated_hair:
            plan.skipped.append(Skipped(f"hair {mesh.name}", f"confidence {assessment.confidence:.2f} below "
                                        f"{thresholds.generated_hair:.2f}", assessment.confidence))
            continue
        for k, comp in enumerate(connected_components(mesh)[:12]):
            if len(comp) < 0.08 * len(mesh.vertex_positions) and k > 0:
                continue
            pts = [mesh.vertex_positions[v] for v in comp]
            pa = principal_axis(pts)
            if pa is None or pa[1] < 0.75:
                plan.skipped.append(Skipped(f"hair {mesh.name}#{k}", "component is not elongated"))
                continue
            axis, _, _ = pa
            # Root -> tip: the root end is the one nearer the head, then the higher one.
            ts = [dot(p, axis) for p in pts]
            t_lo, t_hi = min(ts), max(ts)
            lo_pts = [p for p, t in zip(pts, ts) if t <= t_lo + 0.1 * (t_hi - t_lo)]
            hi_pts = [p for p, t in zip(pts, ts) if t >= t_hi - 0.1 * (t_hi - t_lo)]
            head_w = analysis.joints["head"]
            root_is_hi = _closer(hi_pts, head_w) < _closer(lo_pts, head_w)
            if root_is_hi:
                axis = scale(axis, -1)
                ts = [-t for t in ts]
                t_lo, t_hi = min(ts), max(ts)
            length = t_hi - t_lo
            if head_h and length < 0.6 * head_h:
                plan.skipped.append(Skipped(f"hair {mesh.name}#{k}", "too short to simulate"))
                continue
            n_bones = int(max(MIN_BONES, min(MAX_BONES, round(length / (0.35 * (head_h or length))))))
            joints: list[Vec3] = []
            for j in range(n_bones + 1):
                t = t_lo + length * j / n_bones
                band = 0.5 * length / n_bones
                sl = [p for p, tt in zip(pts, ts) if abs(tt - t) <= band] or pts
                joints.append(mean(sl))  # type: ignore[arg-type]
            name = f"{PROSOPON_PREFIX}Hair_{mesh.name}_{k}"
            bones = [NewBone(f"{name}_{j + 1}", plan.parent_bone if j == 0 else f"{name}_{j}", joints[j], joints[j + 1])
                     for j in range(n_bones)]
            category = "ponytail" if assessment.hair_type == PONYTAIL else "braid" if assessment.hair_type == BRAID else "soft-strand"
            plan.chains.append(HairRigChain(name, mesh.name, bones, category, assessment.confidence))
            # Weights: smooth blend between the two nearest joints along the axis; the root stays on the head.
            wmesh = plan.weights.setdefault(mesh.name, {})
            for v, t in zip(comp, ts):
                u = (t - t_lo) / length * n_bones
                j = min(n_bones - 1, max(0, int(u)))
                f = smoothstep(u - j)
                ws = {bones[j].name: 1.0 - f}
                if j + 1 < n_bones:
                    ws[bones[j + 1].name] = f
                else:
                    ws[bones[j].name] = 1.0
                wmesh[v] = ws
    return plan


def _closer(points: list[Vec3], ref: Vec3) -> float:
    return sum(((p[0] - ref[0]) ** 2 + (p[1] - ref[1]) ** 2 + (p[2] - ref[2]) ** 2) ** 0.5 for p in points) / max(len(points), 1)
