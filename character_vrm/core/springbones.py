"""Best-effort spring-bone detection for hair. Conservative by design: only chains whose
names say "hair" with high confidence are enabled; ambiguous bones are reported, never simulated."""

from __future__ import annotations

import re
from dataclasses import replace

from .config import (
    CHEST_COLLIDER_RADIUS, DEFAULT_SPRING_MIN_CONFIDENCE, HAIR_TOKENS, HEAD_COLLIDER_RADIUS,
    MAX_SPRING_CHAIN_BONES, MIN_SPRING_CHAIN_BONES, REFERENCE_HEIGHT_M, SpringConfig,
)
from .humanoid import HumanoidResult
from .model import ArmatureInfo, ColliderPlan, SpringChain, SpringPlan
from .names import parse_bone_name

_NOT_HAIR = frozenset({"twist", "roll", "ik", "ctrl", "control", "mch", "pole", "target"})
_STEM_STRIP = re.compile(r"\d+|\b(l|r|left|right)\b", re.IGNORECASE)


def _token_confidence(name: str) -> float:
    tokens = parse_bone_name(name).tokens
    if any(t in _NOT_HAIR for t in tokens):
        return 0.0
    return max((HAIR_TOKENS.get(t, 0.0) for t in tokens), default=0.0)


def _stem(name: str) -> str:
    tokens = [t for t in parse_bone_name(name).tokens if not t.isdigit() and t not in ("end",)]
    return "_".join(tokens)


def _clamp(value: float, bounds: tuple[float, float], height: float) -> float:
    return max(bounds[0] * height, min(bounds[1] * height, value))


def plan_spring_bones(
    armature: ArmatureInfo,
    humanoid: HumanoidResult,
    weighted_bones: set[str],
    config: SpringConfig | None = None,
    min_confidence: float = DEFAULT_SPRING_MIN_CONFIDENCE,
) -> SpringPlan:
    config = config or SpringConfig()
    plan = SpringPlan()
    if not config.enabled:
        plan.skipped.append({"bone": None, "reason": "spring bones disabled"})
        return plan

    mapped = {m.source_bone for m in humanoid.mappings.values()}
    ignored = set(config.ignore)
    for name in config.ignore:
        ignored.update(armature.descendants(name))
    height = armature.height or REFERENCE_HEIGHT_M
    scale = height / REFERENCE_HEIGHT_M
    preset = replace(config.preset, hit_radius=config.preset.hit_radius * scale)

    confidence = {b.name: _token_confidence(b.name) for b in armature.bones}
    is_candidate = lambda n: confidence.get(n, 0.0) > 0 and n not in mapped and n not in ignored  # noqa: E731

    def cand_children(name: str) -> list[str]:
        return [c for c in armature.children(name) if is_candidate(c)]

    roots = [b.name for b in armature.bones
             if is_candidate(b.name) and not (armature.parent(b.name) and is_candidate(armature.parent(b.name)))]

    def longest_path(start: str) -> list[str]:
        path, current = [start], start
        while (kids := cand_children(current)):
            current = max(kids, key=lambda c: (armature.subtree_size(c), c))
            path.append(current)
        return path

    def branching(start: str) -> bool:
        return any(len(cand_children(n)) > 1 for n in [start, *armature.descendants(start)] if is_candidate(n))

    chains: dict[str, SpringChain] = {}

    def consider(root: str, forced: bool) -> None:
        path = longest_path(root) if not forced else _forced_path(root)
        reason_conf = 1.0 if forced else min(confidence[b] for b in path)
        reasons = ["config include" if forced else "hair tokens"]
        if not forced:
            if branching(root):
                reason_conf = min(reason_conf, 0.7)
                reasons.append("branching chain")
            if len({_stem(b) for b in path}) > 1:
                reason_conf -= 0.1
                reasons.append("inconsistent naming")
        skip: str | None = None
        if len(path) < MIN_SPRING_CHAIN_BONES:
            skip = f"chain shorter than {MIN_SPRING_CHAIN_BONES} bones"
        elif len(path) > MAX_SPRING_CHAIN_BONES:
            skip = f"chain longer than {MAX_SPRING_CHAIN_BONES} bones"
        elif not forced and any(b not in weighted_bones for b in path[:-1]):
            skip = "bones carry no skin weights (control bones, not hair geometry)"
        elif reason_conf < min_confidence:
            skip = f"confidence {reason_conf:.2f} below {min_confidence:.2f}: " + ", ".join(reasons)
        if skip:
            plan.skipped.append({"bone": root, "bones": path, "confidence": round(reason_conf, 2), "reason": skip})
            return
        chains[root] = SpringChain(f"Hair_{root}", path, round(reason_conf, 2), ", ".join(reasons), preset)

    def _forced_path(root: str) -> list[str]:
        path, current = [root], root
        while True:
            kids = [c for c in armature.children(current) if c not in mapped and c not in ignored]
            if not kids:
                return path
            current = max(kids, key=lambda c: (armature.subtree_size(c), c))
            path.append(current)

    for root in roots:
        consider(root, forced=False)
    for name in config.include:
        if not armature.has(name) or name in mapped or name in ignored:
            plan.skipped.append({"bone": name, "reason": "springBones.include names an unknown, humanoid or ignored bone"})
            continue
        for existing in [r for r, c in chains.items() if name in c.bones]:
            del chains[existing]
        plan.skipped = [s for s in plan.skipped if s.get("bone") != name]
        consider(name, forced=True)

    plan.chains = list(chains.values())
    if plan.chains and config.colliders:
        plan.colliders = _colliders(armature, humanoid, height)
    return plan


def _colliders(armature: ArmatureInfo, humanoid: HumanoidResult, height: float) -> list[ColliderPlan]:
    out: list[ColliderPlan] = []
    for label, sources, bounds in (
        ("head", ("head",), HEAD_COLLIDER_RADIUS),
        ("chest", ("upperChest", "chest", "spine"), CHEST_COLLIDER_RADIUS),
    ):
        source = next((humanoid.source_of(s) for s in sources if humanoid.source_of(s)), None)
        bone = armature.bone(source) if source else None
        if not bone:
            continue
        out.append(ColliderPlan(
            name=f"{label}_collider", bone=bone.name,
            radius=round(_clamp(0.5 * bone.length, bounds, height), 4),
            offset=(0.0, round(0.5 * bone.length, 4), 0.0),  # bone-local: +Y runs head -> tail
        ))
    return out
