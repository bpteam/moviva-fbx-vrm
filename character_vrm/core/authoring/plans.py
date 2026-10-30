"""Shared plan types for the Auto-Author."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..anatomy.confidence import AnalysisSource, Provenance
from ..anatomy.vec import Vec3

PROSOPON_PREFIX = "Prosopon_"


@dataclass
class ShapeKeyPlan:
    """A generated shape key. ``offsets`` are world-space deltas per mesh and vertex (the adapter converts them into
    mesh-local space). Never modifies an existing shape key: the name always carries the ``Prosopon_`` prefix."""

    name: str
    offsets: dict[str, dict[int, Vec3]] = field(default_factory=dict)
    confidence: float = 0.0
    source: AnalysisSource = AnalysisSource.GENERATED
    description: str = ""
    max_offset: float = 0.0
    vrm_expression: str | None = None  # the VRM preset this key is bound to (None: authored only)

    @property
    def vertex_count(self) -> int:
        return sum(len(v) for v in self.offsets.values())

    @property
    def meshes(self) -> list[str]:
        return sorted(self.offsets)


@dataclass
class ExpressionDecision:
    """One row of the report: where an expression comes from."""

    name: str
    status: str  # SOURCE | MAPPED | INFERRED | GENERATED | MISSING
    confidence: float = 0.0
    detail: str = ""  # e.g. "mouthSmileLeft + mouthSmileRight", "skipped: confidence 0.61 below 0.85"


@dataclass
class PlanIssue:
    plan: str
    message: str


@dataclass
class Skipped:
    """Something the plan declined to do, with the reason (never silent)."""

    what: str
    reason: str
    confidence: float | None = None


__all__ = ["ExpressionDecision", "PlanIssue", "PROSOPON_PREFIX", "Provenance", "ShapeKeyPlan", "Skipped"]
