"""Phase 14: spring-bone presets by hair category, scaled by chain length and anatomy (central table)."""

from __future__ import annotations

from dataclasses import replace

from ..anatomy.confidence import DEFAULT_THRESHOLDS, Thresholds
from ..anatomy.hair import LONG_HAIR, SHORT_HAIR, SOFT_STRAND, HairChain
from ..anatomy.model import HumanModelAnalysis
from ..config import REFERENCE_HEIGHT_M
from ..model import SpringChain, SpringPlan, SpringPreset
from .plans import Skipped

# One table, one place: (stiffness, drag, gravity, hit radius in metres for a 1.7 m character).
SPRING_PRESETS: dict[str, SpringPreset] = {
    SHORT_HAIR: SpringPreset(stiffness=1.6, drag_force=0.60, gravity_power=0.05, hit_radius=0.010),
    LONG_HAIR: SpringPreset(stiffness=0.9, drag_force=0.45, gravity_power=0.12, hit_radius=0.012),
    "ponytail": SpringPreset(stiffness=0.7, drag_force=0.40, gravity_power=0.15, hit_radius=0.014),
    "braid": SpringPreset(stiffness=1.1, drag_force=0.50, gravity_power=0.10, hit_radius=0.012),
    SOFT_STRAND: SpringPreset(stiffness=1.0, drag_force=0.50, gravity_power=0.10, hit_radius=0.010),
}
MIN_BONES, MAX_BONES = 2, 16


def preset_for(category: str, length_ratio: float, height: float, head_width: float | None) -> SpringPreset:
    """Category preset adapted to the chain: long chains get softer springs, the hit radius follows the head width."""
    base = SPRING_PRESETS.get(category, SPRING_PRESETS[SOFT_STRAND])
    softness = max(0.6, min(1.5, 1.0 / max(length_ratio, 0.4)))  # chain length / head height
    hit = base.hit_radius * height / REFERENCE_HEIGHT_M
    if head_width:
        hit = min(hit, 0.1 * head_width)
    return replace(base, stiffness=round(base.stiffness * softness, 3), hit_radius=round(hit, 4))


def plan_hair_springs(analysis: HumanModelAnalysis, existing: SpringPlan | None,
                      thresholds: Thresholds = DEFAULT_THRESHOLDS) -> tuple[SpringPlan, list[Skipped]]:
    """Chains from the analysis (any bone names) merged with the name-based plan; every chain re-preset by category."""
    hair = analysis.hair
    skipped: list[Skipped] = []
    plan = SpringPlan(chains=[], colliders=list(existing.colliders) if existing else [],
                      skipped=list(existing.skipped) if existing else [])
    height = analysis.measurements.total_height_units or REFERENCE_HEIGHT_M
    covered: set[str] = set()
    if existing:
        for chain in existing.chains:
            covered.update(chain.bones)
            cat = _category_of(chain.bones, hair)
            ratio = _ratio(chain.bones, analysis)
            plan.chains.append(SpringChain(chain.name, chain.bones, chain.confidence, chain.reason + f"; preset {cat}",
                                           preset_for(cat, ratio, height, analysis.head_width)))
    if hair is None:
        return plan, skipped
    for chain in hair.chains:
        if set(chain.bones) & covered:
            continue
        if chain.confidence < thresholds.hair_confident:
            skipped.append(Skipped(f"spring {chain.root}", f"hair confidence {chain.confidence:.2f} below "
                                   f"{thresholds.hair_confident:.2f}: hair stays static", chain.confidence))
            continue
        if not (MIN_BONES <= len(chain.bones) <= MAX_BONES):
            skipped.append(Skipped(f"spring {chain.root}", f"{len(chain.bones)} bones outside {MIN_BONES}-{MAX_BONES}",
                                   chain.confidence))
            continue
        plan.chains.append(SpringChain(
            f"Hair_{chain.root}", chain.bones, chain.confidence,
            f"inferred hair chain ({chain.category}; evidence {chain.evidence})",
            preset_for(chain.category, chain.length_ratio, height, analysis.head_width)))
        covered.update(chain.bones)
    return plan, skipped


def _category_of(bones: list[str], hair) -> str:
    if hair is not None:
        for c in hair.chains:
            if set(bones) & set(c.bones):
                return c.category
    return SOFT_STRAND


def _ratio(bones: list[str], analysis: HumanModelAnalysis) -> float:
    head_h = analysis.head_height
    if not head_h or analysis.hair is None:
        return 1.0
    for c in analysis.hair.chains:
        if set(bones) & set(c.bones):
            return c.length_ratio
    return 1.0
