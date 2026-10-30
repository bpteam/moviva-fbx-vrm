"""Phase 4: skeleton semantic analysis.

Names are only one evidence channel. The topology inference below maps a humanoid from hierarchy alone
(``Bone.001``, ``Bone.002``, ...): it finds the mirrored limb pairs, the torso between them and the head branch,
then scores every assignment with independent evidence (topology, position, symmetry, skinning, name) that is
**kept un-collapsed** for the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import ArmatureInfo, BoneInfo
from ..names import parse_bone_name
from .confidence import AnalysisSource, AnalysisWarning, Evidence, fuse
from .limbs import infer_limbs
from .vec import Vec3, cross, dist, dot, normalize, orthogonalize, sub


@dataclass
class BoneAssessment:
    vrm_bone: str
    bone: str
    confidence: float
    source: AnalysisSource
    evidence: dict[str, float]
    method: str = ""


@dataclass
class SkeletonAnalysis:
    assessments: dict[str, BoneAssessment] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[AnalysisWarning] = field(default_factory=list)
    topology_only: dict[str, str] = field(default_factory=dict)  # what pure topology found (diagnostic)
    side_assumed: bool = False

    def confidence_of(self, vrm_bone: str) -> float:
        a = self.assessments.get(vrm_bone)
        return a.confidence if a else 0.0

    def source_bone(self, vrm_bone: str) -> str | None:
        a = self.assessments.get(vrm_bone)
        return a.bone if a else None


# Expected height of each joint as a fraction of the skeleton's vertical extent (floor = 0). Generous on purpose:
# it discriminates "obviously wrong" (a head at the knees) from plausible, never one plausible bone from another.
EXPECTED_HEIGHT: dict[str, tuple[float, float]] = {
    "hips": (0.42, 0.62), "spine": (0.48, 0.70), "chest": (0.56, 0.78), "upperChest": (0.62, 0.82),
    "neck": (0.74, 0.92), "head": (0.80, 0.96),
    "shoulder": (0.68, 0.88), "upperArm": (0.68, 0.88), "lowerArm": (0.45, 0.88), "hand": (0.30, 0.88),
    "upperLeg": (0.40, 0.60), "lowerLeg": (0.18, 0.40), "foot": (0.0, 0.14), "toes": (0.0, 0.08),
    "eye": (0.85, 0.99),
}


def _role(vrm_bone: str) -> str:
    for side in ("left", "right"):
        if vrm_bone.startswith(side):
            rest = vrm_bone[len(side):]
            return rest[0].lower() + rest[1:]
    return vrm_bone


class _Tree:
    """Deform-bone hierarchy (helpers such as IK targets and poles are skipped; their children re-attach)."""

    def __init__(self, armature: ArmatureInfo) -> None:
        deform = [b for b in armature.bones if b.deform]
        keep = {b.name for b in (deform if len(deform) >= 12 else armature.bones)}
        self.armature = armature
        self.parent: dict[str, str | None] = {}
        self.children: dict[str, list[str]] = {n: [] for n in keep}
        for b in armature.bones:
            if b.name not in keep:
                continue
            p = armature.parent(b.name)
            while p is not None and p not in keep:
                p = armature.parent(p)
            self.parent[b.name] = p
            if p is not None:
                self.children[p].append(b.name)
        self.roots = [n for n, p in self.parent.items() if p is None]
        self._size: dict[str, int] = {}
        self._sig: dict[str, tuple[int, ...]] = {}

    def bone(self, name: str) -> BoneInfo:
        return self.armature.bone(name)  # type: ignore[return-value]

    def size(self, name: str) -> int:
        if name not in self._size:
            self._size[name] = 1 + sum(self.size(c) for c in self.children[name])
        return self._size[name]

    def signature(self, name: str) -> tuple[int, ...]:
        """Number of nodes per depth below ``name``: equal for mirrored limbs."""
        if name not in self._sig:
            levels = [name]
            sig: list[int] = []
            while levels:
                sig.append(len(levels))
                levels = [c for n in levels for c in self.children[n]]
            self._sig[name] = tuple(sig)
        return self._sig[name]

    def main_path(self, start: str) -> list[str]:
        """Follow the largest subtree (ties: longest bone)."""
        path = [start]
        while self.children[path[-1]]:
            kids = self.children[path[-1]]
            path.append(max(kids, key=lambda k: (self.size(k), self.bone(k).length)))
        return path

    def path_length(self, path: list[str]) -> float:
        return sum(self.bone(n).length for n in path)

    def ancestors(self, name: str) -> list[str]:
        out: list[str] = []
        p = self.parent.get(name)
        while p is not None:
            out.append(p)
            p = self.parent.get(p)
        return out

    def is_ancestor(self, a: str, b: str) -> bool:
        return a in self.ancestors(b)


def _mirror_pairs(tree: _Tree, node: str) -> list[tuple[str, str]]:
    kids = tree.children[node]
    out = []
    for i in range(len(kids)):
        for j in range(i + 1, len(kids)):
            a, b = kids[i], kids[j]
            if tree.signature(a) != tree.signature(b) or tree.size(a) < 3:
                continue
            la, lb = tree.bone(a).length, tree.bone(b).length
            if la < 1e-9 or lb < 1e-9 or max(la, lb) / min(la, lb) > 1.25:
                continue
            if dist(tree.bone(a).head, tree.bone(b).head) < 1e-6:
                continue
            out.append((a, b))
    return out


@dataclass
class _Hub:
    node: str
    pair: tuple[str, str]
    reach: float  # chain length of the longer main path of the pair


def _find_hubs(tree: _Tree) -> list[_Hub]:
    hubs: list[_Hub] = []
    for node in tree.parent:
        for a, b in _mirror_pairs(tree, node):
            reach = max(tree.path_length(tree.main_path(a)), tree.path_length(tree.main_path(b)))
            hubs.append(_Hub(node, (a, b), reach))
    return hubs


def _side_votes(tree: _Tree, a: str, b: str) -> int:
    """+1 when the *names* say ``a`` is left / ``b`` right, -1 for the reverse, 0 when they say nothing."""
    def side(root: str) -> str | None:
        for n in [root, *tree.ancestors(root), *tree.children[root]]:
            s = parse_bone_name(n).side
            if s:
                return s
        return None
    sa, sb = side(a), side(b)
    if sa == "left" and sb != "left":
        return 1
    if sa == "right" and sb != "right":
        return -1
    if sb == "left" and sa != "left":
        return -1
    if sb == "right" and sa != "right":
        return 1
    return 0


def infer_topology(armature: ArmatureInfo, left_hint: Vec3 | None = None) -> tuple[
        dict[str, tuple[str, float, str]], list[AnalysisWarning], bool]:
    """Map a humanoid from hierarchy alone. Returns ``({vrm_bone: (source_bone, topology_score, method)}, warnings,
    side_assumed)``. No axis is assumed except for the final left/right decision (names first, then
    ``left_hint``, then the Blender import convention +X, flagged as assumed)."""
    warnings: list[AnalysisWarning] = []
    tree = _Tree(armature)
    hubs = _find_hubs(tree)
    if len(hubs) < 2:
        return {}, [AnalysisWarning("topology-no-limbs", "fewer than two mirrored limb pairs in the hierarchy",
                                    "skeleton")], False

    # Legs are the shallowest hub; the arms are the next hub below it that is limb-sized (pigtails / ears are not).
    hubs.sort(key=lambda h: (len(tree.ancestors(h.node)), -h.reach))
    leg_hub = hubs[0]
    arm_hub = next((h for h in hubs[1:] if tree.is_ancestor(leg_hub.node, h.node)
                    and h.reach >= 0.45 * leg_hub.reach), None)
    out: dict[str, tuple[str, float, str]] = {}
    if arm_hub is None:
        return {}, [AnalysisWarning("topology-no-arms", "no arm pair found below the leg pair", "skeleton")], False

    hips = leg_hub.node
    hub = arm_hub.node
    out["hips"] = (hips, 0.95, "legs-hub")

    # Torso: hips -> hub.
    chain = [n for n in reversed([hub, *tree.ancestors(hub)]) if n != hips and tree.is_ancestor(hips, n)]
    if chain:
        topo = 0.95 if len(chain) <= 3 else 0.75
        out["spine"] = (chain[0], topo, "torso-chain")
        if len(chain) >= 2:
            out["chest"] = (chain[1], topo, "torso-chain")
        if len(chain) >= 3:
            out["upperChest"] = (chain[-1], topo, "torso-chain")

    # Side labels.
    votes = 0
    for a, b in (leg_hub.pair, arm_hub.pair):
        votes += _side_votes(tree, a, b)
    up = normalize(sub(tree.bone(hub).head, tree.bone(hips).head))
    side_assumed = False
    swap = False
    if votes:
        swap = votes < 0  # pair[0] is the right side
    else:
        side_assumed = True
        hint = left_hint or (1.0, 0.0, 0.0)
        a_head, b_head = tree.bone(leg_hub.pair[0]).head, tree.bone(leg_hub.pair[1]).head
        lateral = sub(a_head, b_head)
        if up is not None:
            lateral = orthogonalize(lateral, up) or lateral  # type: ignore[assignment]
        swap = dot(lateral, hint) < 0
        warnings.append(AnalysisWarning(
            "topology-side-assumed", "bone names carry no left/right information: left is assumed to be +X "
            "(Blender import convention)", "skeleton"))

    def sides(pair: tuple[str, str]) -> dict[str, str]:
        return {"left": pair[1], "right": pair[0]} if swap else {"left": pair[0], "right": pair[1]}

    for side, root in sides(leg_hub.pair).items():
        path = tree.main_path(root)
        for role, idx in (("UpperLeg", 0), ("LowerLeg", 1), ("Foot", 2), ("Toes", 3)):
            if idx < len(path):
                out[f"{side}{role}"] = (path[idx], 0.93 if idx < 3 else 0.8, "leg-chain")
    for side, root in sides(arm_hub.pair).items():
        path = tree.main_path(root)
        hand_idx = next((i for i, n in enumerate(path) if i >= 1 and len(tree.children[n]) >= 3), None)
        if hand_idx is None:
            hand_idx = len(path) - 1
            if hand_idx >= 2 and tree.bone(path[-1]).length < 0.5 * tree.bone(path[-2]).length:
                hand_idx -= 1  # trailing tip bone
        before = path[:hand_idx]
        topo = 0.93 if 2 <= len(before) <= 3 else 0.7
        if len(before) >= 3:
            out[f"{side}Shoulder"] = (before[0], topo, "arm-chain")
            out[f"{side}UpperArm"] = (before[1], topo, "arm-chain")
            out[f"{side}LowerArm"] = (before[-1], topo, "arm-chain")
        elif len(before) == 2:
            out[f"{side}UpperArm"] = (before[0], topo, "arm-chain")
            out[f"{side}LowerArm"] = (before[1], topo, "arm-chain")
        else:
            warnings.append(AnalysisWarning("topology-short-arm", f"{side} arm chain too short", "skeleton"))
            continue
        out[f"{side}Hand"] = (path[hand_idx], topo, "arm-chain")

    # Neck / head: the largest non-pair branch off the hub.
    branches = [c for c in tree.children[hub] if c not in arm_hub.pair and
                c not in [tree.parent.get(arm_hub.pair[0]), tree.parent.get(arm_hub.pair[1])]]
    # Clavicles hang below the pair; the hub's own pair children are excluded above.
    branches = [c for c in tree.children[hub] if c not in arm_hub.pair]
    if branches:
        head_path = tree.main_path(max(branches, key=tree.size))
        branch_i = next((i for i, n in enumerate(head_path) if len(tree.children[n]) >= 2), None)
        if branch_i is None:
            last = head_path[-1]
            if len(head_path) >= 2 and tree.bone(last).length < 0.6 * tree.bone(head_path[-2]).length:
                branch_i = len(head_path) - 2
            else:
                branch_i = len(head_path) - 1
        # A lone bone above the arm pair is as likely a neck as a head: not enough for automatic mapping.
        out["head"] = (head_path[branch_i], 0.9 if len(head_path) >= 2 else 0.5, "head-branch")
        if branch_i >= 1:
            out["neck"] = (head_path[0], 0.9 if branch_i == 1 else 0.75, "head-branch")
    else:
        warnings.append(AnalysisWarning("topology-no-head", "no branch above the arm pair", "skeleton"))
    return out, warnings, side_assumed


def _height_fraction(armature: ArmatureInfo, up: Vec3, point: Vec3) -> float:
    ys = [dot(p, up) for b in armature.bones for p in (b.head, b.tail)]
    lo, hi = min(ys), max(ys)
    return (dot(point, up) - lo) / max(hi - lo, 1e-9)


def _score_range(value: float, rng: tuple[float, float], slack: float = 0.15) -> float:
    lo, hi = rng
    if lo <= value <= hi:
        return 1.0
    d = lo - value if value < lo else value - hi
    return max(0.0, 1.0 - d / slack)


def assess_skeleton(armature: ArmatureInfo, weighted_bones: set[str], names: dict[str, tuple[str, float]],
                    left_hint: Vec3 | None = None) -> SkeletonAnalysis:
    """Fuse topology, position, symmetry, skinning and name evidence for every humanoid bone.

    ``names`` is the existing (name / rig-profile driven) mapping: ``{vrm_bone: (source_bone, confidence)}``.
    Existing mappings are never overridden here; disagreements are reported in ``conflicts``.
    """
    analysis = SkeletonAnalysis()
    # Extremities first (feet lowest, hands widest, forward from the toes): immune to helper / cloth bones next to
    # the limbs. The hierarchy-signature inference stays as the fallback.
    limbs, _notes = infer_limbs(armature)
    if all(k in limbs for k in ("hips", "leftUpperLeg", "rightUpperLeg", "leftHand", "rightHand")):
        topo, warnings, assumed = limbs, [], False
    else:
        topo, warnings, assumed = infer_topology(armature, left_hint)
    analysis.warnings.extend(warnings)
    analysis.side_assumed = assumed
    analysis.topology_only = {k: v[0] for k, v in topo.items()}

    candidates: dict[str, tuple[str, dict[str, float], AnalysisSource, str]] = {}
    for vrm, (bone, conf) in names.items():
        candidates[vrm] = (bone, {"name": conf}, AnalysisSource.RIG_PROFILE if conf >= 0.95 else AnalysisSource.SKELETON, "names")
    for vrm, (bone, score, method) in topo.items():
        if vrm in candidates:
            existing, evidence, source, _ = candidates[vrm]
            if existing == bone:
                evidence["topology"] = score
            else:
                evidence["topology"] = 0.0
                analysis.conflicts.append(f"{vrm}: names say '{existing}', topology says '{bone}'")
        else:
            candidates[vrm] = (bone, {"topology": score}, AnalysisSource.SKELETON, method)

    # A global up axis from the mapped hips/head only to score positions (the real frame is built later).
    # The reference up comes from the topology (hips -> the highest torso/head bone it found), never from the very
    # bones being judged: a wrong "head" must not define which way is up.
    up = None
    hips_bone = topo.get("hips") or (candidates.get("hips") and (candidates["hips"][0],))
    if hips_bone:
        for probe in ("head", "neck", "upperChest", "chest"):
            target = topo.get(probe)
            if target:
                up = normalize(sub(armature.bone(target[0]).head, armature.bone(hips_bone[0]).head))  # type: ignore[union-attr]
                break
    if up is None:
        hips, head = candidates.get("hips"), candidates.get("head")
        if hips and head:
            up = normalize(sub(armature.bone(head[0]).head, armature.bone(hips[0]).head))  # type: ignore[union-attr]
    for vrm, (bone, evidence, source, method) in candidates.items():
        info = armature.bone(bone)
        if info is None:
            continue
        if up is not None:
            rng = EXPECTED_HEIGHT.get(_role(vrm))
            if rng:
                evidence["position"] = _score_range(_height_fraction(armature, up, info.head), rng)
        if weighted_bones:
            evidence["skinning"] = 1.0 if bone in weighted_bones else (0.35 if _role(vrm) not in ("eye",) else 0.6)
    for vrm, (bone, evidence, _, _) in list(candidates.items()):
        if not vrm.startswith("left"):
            continue
        other = candidates.get("right" + vrm[4:])
        a, b = armature.bone(bone), armature.bone(other[0]) if other else None
        if a and b and up is not None:
            lens = sorted((a.length, b.length))
            ratio = lens[0] / lens[1] if lens[1] > 1e-9 else 1.0
            ha = dot(a.head, up)
            hb = dot(b.head, up)
            span = max(abs(dot(p, up)) for p in (a.head, b.head)) or 1.0
            height_match = 1.0 - min(1.0, abs(ha - hb) / max(0.05 * span, 1e-6))
            sym = 0.6 * ratio + 0.4 * height_match
            candidates[vrm][1]["symmetry"] = sym
            candidates["right" + vrm[4:]][1]["symmetry"] = sym

    for vrm, (bone, evidence, source, method) in candidates.items():
        conf = fuse(evidence)
        analysis.assessments[vrm] = BoneAssessment(vrm, bone, round(conf, 4), source, Evidence(evidence).as_dict(), method)
    return analysis
