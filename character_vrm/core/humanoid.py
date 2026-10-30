"""Humanoid auto-mapping.

Priority (a higher one is never overwritten by a lower one):
1. explicit config   2. known rig profile   3. normalised names
4. hierarchy / topology   5. anatomical position

Every match carries a confidence and the reason it was made. The source rig is
never modified here: the result is only a set of ``vrm_bone -> source_bone`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..profiles import BONE_PROFILES, BoneProfile
from ..profiles import generic as vocab
from .config import DEFAULT_MIN_CONFIDENCE
from .model import (
    ALL_HUMAN_BONES, FINGER_BONES, REQUIRED_HUMAN_BONES, ArmatureInfo, BoneMapping, ConversionWarning,
)
from .names import ParsedBone, parse_bone_name

LEFT, RIGHT = "left", "right"
SIDE_SIGN = {LEFT: 1.0, RIGHT: -1.0}  # Blender import convention: the character's left is +X
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")

# VRM parent rules: the nearest *mapped* ancestor must be one of these.
_TORSO_TOP = ["upperChest", "chest", "spine"]
_PARENT_RULES: dict[str, list[str]] = {
    "spine": ["hips"], "chest": ["spine"], "upperChest": ["chest", "spine"],
    "neck": _TORSO_TOP, "head": ["neck", *_TORSO_TOP],
    "jaw": ["head"], "leftEye": ["head"], "rightEye": ["head"],
}
for _s in (LEFT, RIGHT):
    _PARENT_RULES.update({
        f"{_s}Shoulder": _TORSO_TOP,
        f"{_s}UpperArm": [f"{_s}Shoulder", "neck", *_TORSO_TOP],
        f"{_s}LowerArm": [f"{_s}UpperArm"], f"{_s}Hand": [f"{_s}LowerArm"],
        f"{_s}UpperLeg": ["hips"], f"{_s}LowerLeg": [f"{_s}UpperLeg"],
        f"{_s}Foot": [f"{_s}LowerLeg"], f"{_s}Toes": [f"{_s}Foot"],
    })
    for _f in FINGERS:
        _h = f"{_s}Hand"
        if _f == "Thumb":
            _PARENT_RULES.update({
                f"{_s}ThumbMetacarpal": [_h],
                f"{_s}ThumbProximal": [f"{_s}ThumbMetacarpal", _h],
                f"{_s}ThumbDistal": [f"{_s}ThumbProximal", f"{_s}ThumbMetacarpal"],
            })
        else:
            _PARENT_RULES.update({
                f"{_s}{_f}Proximal": [_h],
                f"{_s}{_f}Intermediate": [f"{_s}{_f}Proximal"],
                f"{_s}{_f}Distal": [f"{_s}{_f}Intermediate", f"{_s}{_f}Proximal"],
            })


class HumanoidError(Exception):
    """Unusable input (e.g. the map file names a bone that does not exist)."""


@dataclass
class HumanoidResult:
    armature: str
    mappings: dict[str, BoneMapping] = field(default_factory=dict)
    profile: str | None = None
    missing_required: list[str] = field(default_factory=list)
    below_threshold: list[BoneMapping] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[ConversionWarning] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.errors

    @property
    def mapped_required(self) -> int:
        return sum(1 for b in REQUIRED_HUMAN_BONES if b in self.mappings)

    @property
    def mapped_optional(self) -> int:
        return sum(1 for b in self.mappings if b not in REQUIRED_HUMAN_BONES)

    @property
    def mean_confidence(self) -> float:
        if not self.mappings:
            return 0.0
        return sum(m.confidence for m in self.mappings.values()) / len(self.mappings)

    def source_of(self, vrm_bone: str) -> str | None:
        m = self.mappings.get(vrm_bone)
        return m.source_bone if m else None

    def as_map(self) -> dict[str, str]:
        return {k: v.source_bone for k, v in self.mappings.items()}


class _Resolver:
    def __init__(self, arm: ArmatureInfo, min_confidence: float) -> None:
        self.arm = arm
        self.min_confidence = min_confidence
        self.parsed: dict[str, ParsedBone] = {b.name: parse_bone_name(b.name) for b in arm.bones}
        self.mappings: dict[str, BoneMapping] = {}
        self.used: dict[str, str] = {}
        self.profile: str | None = None
        self.warnings: list[ConversionWarning] = []

    # -- helpers ------------------------------------------------------------
    def assign(self, vrm: str, src: str | None, confidence: float, reason: str,
               anchors: tuple[str, ...] = ()) -> bool:
        """``anchors``: VRM bones this match was derived from; it can never be more trusted than they are."""
        if not src or vrm in self.mappings or src in self.used or not self.arm.has(src):
            return False
        confidence = min([confidence, *(self.mappings[a].confidence for a in anchors if a in self.mappings)])
        self.mappings[vrm] = BoneMapping(vrm, src, round(confidence, 2), reason)
        self.used[src] = vrm
        return True

    def src(self, vrm: str) -> str | None:
        m = self.mappings.get(vrm)
        return m.source_bone if m else None

    def usable(self, name: str) -> bool:
        return self.parsed[name].usable

    def main_child(self, name: str) -> str | None:
        kids = [c for c in self.arm.children(name) if self.usable(c) and c not in self.used]
        if not kids:
            return None
        kids.sort(key=lambda c: (-self.arm.subtree_size(c), -self.arm.bone(c).length, c))  # type: ignore[union-attr]
        return kids[0]

    # -- 1. config ------------------------------------------------------------
    def from_config(self, explicit: dict[str, str]) -> None:
        for vrm, src in explicit.items():
            if not self.arm.has(src):
                raise HumanoidError(f"humanoid.{vrm}: bone '{src}' does not exist in armature '{self.arm.name}'")
            if src in self.used:
                raise HumanoidError(f"humanoid.{vrm}: bone '{src}' is already assigned to '{self.used[src]}'")
            self.assign(vrm, src, 1.0, "config")

    # -- 2. profile -------------------------------------------------------------
    def from_profile(self, profiles: tuple[BoneProfile, ...]) -> None:
        names = frozenset(b.name for b in self.arm.bones)
        best: tuple[float, BoneProfile, dict[str, str]] | None = None
        for profile in profiles:
            built = profile.build(names)
            if built and (best is None or built[1] > best[0]):
                best = (built[1], profile, built[0])
        if best:
            self.profile = best[1].name
            for vrm, src in best[2].items():
                self.assign(vrm, src, 0.99, f"profile:{best[1].name}")

    # -- 3. names ---------------------------------------------------------------
    def _name_targets(self, pb: ParsedBone) -> list[tuple[str, float]]:
        part, side = pb.part, pb.side
        indexed_penalty = 0.1 if pb.index is not None and part not in vocab.FINGER_PARTS and part != "neck" else 0.0
        if part == "hips":
            return [("hips", 1.0)]
        if part == "pelvis":
            return [("hips", 0.8)]
        if part in ("neck", "head", "jaw"):
            return [(part, 0.95 - indexed_penalty - (0.01 * (pb.index or 0) if part == "neck" else 0))]
        if part in ("chest", "upperchest"):
            return [("chest" if part == "chest" else "upperChest", 0.9 - indexed_penalty)]
        if side is None:
            return []
        table = {
            "eye": ("Eye", 0.95), "shoulder": ("Shoulder", 0.95), "upperarm": ("UpperArm", 0.97),
            "lowerarm": ("LowerArm", 0.97), "hand": ("Hand", 0.97), "upperleg": ("UpperLeg", 0.97),
            "lowerleg": ("LowerLeg", 0.97), "leg": ("LowerLeg", 0.8), "foot": ("Foot", 0.97),
            "toes": ("Toes", 0.9),
        }
        if part in table:
            suffix, base = table[part]
            return [(f"{side}{suffix}", base - indexed_penalty)]
        if part in vocab.FINGER_PARTS and pb.index in (1, 2, 3):
            finger = part.capitalize()
            names = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else (
                "Proximal", "Intermediate", "Distal")
            return [(f"{side}{finger}{names[pb.index - 1]}", 0.95)]
        return []

    def from_names(self) -> None:
        candidates: dict[str, list[tuple[float, int, str]]] = {}
        for bone in self.arm.bones:
            pb = self.parsed[bone.name]
            if not pb.part or not pb.usable:
                continue
            for vrm, score in self._name_targets(pb):
                candidates.setdefault(vrm, []).append((score, self.arm.depth(bone.name), bone.name))
        for vrm, cands in candidates.items():
            cands.sort(key=lambda c: (-c[0], c[1], c[2]))
            score = cands[0][0]
            reason = "name"
            rivals = [c for c in cands[1:] if abs(c[0] - score) < 0.02]
            if rivals:
                score -= 0.1
                reason = f"name (ambiguous: {len(rivals) + 1} candidates)"
            for _, _, name in cands:  # first free candidate wins
                if self.assign(vrm, name, score, reason):
                    break

    # -- 4. hierarchy -----------------------------------------------------------
    def from_hierarchy(self) -> None:
        self._hips()
        self._torso()
        self._shoulders()
        for _ in range(2):  # gaps can cascade (upper arm known -> lower arm -> hand)
            for side in (LEFT, RIGHT):
                self._limb(side, "UpperArm", "LowerArm", "Hand")
                self._limb(side, "UpperLeg", "LowerLeg", "Foot")
                self._toes(side)
        self._finger_chains()

    def _hips(self) -> None:
        if "hips" in self.mappings:
            return
        anchors = [self.src(v) for v in ("head", "leftUpperLeg", "rightUpperLeg")]
        legs = [a for a in anchors[1:] if a]
        if anchors[0] and len(legs) == 2:
            lca, conf = self.arm.lowest_common_ancestor([a for a in anchors if a]), 0.88
            used = ("head", "leftUpperLeg", "rightUpperLeg")
        elif len(legs) == 2:
            lca, conf = self.arm.lowest_common_ancestor(legs), 0.72
            used = ("leftUpperLeg", "rightUpperLeg")
        else:
            return
        if lca and self.usable(lca) and self._plausible_hips_height(lca, legs):
            self.assign("hips", lca, conf, "hierarchy (common ancestor of head and legs)", used)

    def _plausible_hips_height(self, hips: str, legs: list[str]) -> bool:
        """A ground-level root bone is a common ancestor too, but it is not the pelvis."""
        floor = min(self.arm.bone(leg).head[2] for leg in legs)  # type: ignore[union-attr]
        return self.arm.bone(hips).head[2] >= floor - 0.1 * (self.arm.height or 1.0)  # type: ignore[union-attr]

    def _torso(self) -> None:
        hips, head = self.src("hips"), self.src("head")
        if not hips or not head or not self.arm.is_ancestor(hips, head):
            return
        chain = self.arm.ancestors(head)
        path = list(reversed(chain[:chain.index(hips)]))  # hips-side first, excludes hips and head
        neck_at = next((i for i, b in enumerate(path) if self.parsed[b].part == "neck"), None)
        named_neck = self.src("neck")
        if named_neck in path:
            neck_at = path.index(named_neck)
        if neck_at is not None:
            neck_bone = path[neck_at]
            self.assign("neck", neck_bone, 0.95 if self.usable(neck_bone) else 0.9,
                        "hierarchy" if self.usable(neck_bone) else "hierarchy (twist-named neck joint)",
                        ("hips", "head"))
            torso_path = path[:neck_at]
        elif len(path) >= 4 and self.parsed[path[-1]].part in (None, "spine") and self.usable(path[-1]):
            # nothing is named "neck": with a torso this long the last bone below the head is the neck
            self.assign("neck", path[-1], 0.7, "hierarchy (unnamed neck)", ("hips", "head"))
            torso_path = path[:-1]
        else:
            torso_path = path
        torso = [b for b in torso_path if self.usable(b) and self.parsed[b].part in (None, "spine", "chest", "upperchest")]
        if not torso:
            return
        named = all(self.parsed[b].part is not None for b in torso)
        conf = (0.96 if len(torso) <= 3 else 0.9) if named else 0.76
        anchors = ("hips", "head")
        self.assign("spine", torso[0], conf, "hierarchy", anchors)
        if len(torso) >= 2:
            self.assign("chest", torso[1], conf, "hierarchy", anchors)
        if len(torso) >= 3:
            self.assign("upperChest", torso[-1], conf, "hierarchy", anchors)

    def _shoulders(self) -> None:
        torso = {self.src(v) for v in ("hips", "spine", "chest", "upperChest", "neck")} - {None}
        for side in (LEFT, RIGHT):
            upper = self.src(f"{side}UpperArm")
            parent = self.arm.parent(upper) if upper else None
            if (parent and parent not in self.used and self.usable(parent)
                    and self.arm.parent(parent) in torso):
                self.assign(f"{side}Shoulder", parent, 0.8, "hierarchy", (f"{side}UpperArm",))

    def _limb(self, side: str, top: str, mid: str, end: str) -> None:
        t, m, e = (f"{side}{x}" for x in (top, mid, end))
        ts, ms, es = self.src(t), self.src(m), self.src(e)
        if ts and es and not ms:
            between = [b for b in self.arm.ancestors(es)[:self.arm.ancestors(es).index(ts)]
                       if self.usable(b) and b not in self.used] if ts in self.arm.ancestors(es) else []
            if between:
                self.assign(m, between[-1], 0.85, "hierarchy", (t, e))
        ms = self.src(m)
        if ts and not ms:
            self.assign(m, self.main_child(ts), 0.8, "hierarchy", (t,))
            ms = self.src(m)
        if ms and not es:
            self.assign(e, self.main_child(ms), 0.8, "hierarchy", (m,))
        if ms and not ts:
            parent = self.arm.parent(ms)
            if parent and self.usable(parent) and parent not in self.used:
                self.assign(t, parent, 0.8, "hierarchy", (m,))

    def _toes(self, side: str) -> None:
        foot = self.src(f"{side}Foot")
        if not foot or f"{side}Toes" in self.mappings:
            return
        kids = [c for c in self.arm.children(foot) if self.usable(c) and c not in self.used]
        if len(kids) == 1 and self.parsed[kids[0]].part in (None, "toes"):
            self.assign(f"{side}Toes", kids[0], 0.76, "hierarchy", (f"{side}Foot",))

    def _finger_chains(self) -> None:
        for side in (LEFT, RIGHT):
            for finger in FINGERS:
                parts = ["Metacarpal", "Proximal", "Distal"] if finger == "Thumb" else [
                    "Proximal", "Intermediate", "Distal"]
                names = [f"{side}{finger}{p}" for p in parts]
                for a, b in zip(names, names[1:]):
                    if self.src(a) and b not in self.mappings:
                        self.assign(b, self.main_child(self.src(a)), 0.8, "hierarchy", (a,))  # type: ignore[arg-type]

    # -- 5. anatomical position ---------------------------------------------------
    def core_by_position(self) -> None:
        """Hips and head from the shape of the skeleton alone. Always low confidence."""
        arm = self.arm
        height = arm.height or 1.0
        if "hips" not in self.mappings:
            for bone in arm.bones:
                if bone.name in self.used or not self.usable(bone.name):
                    continue
                kids = [arm.bone(c) for c in arm.children(bone.name) if self.usable(c)]
                up = [k for k in kids if k.head[2] > bone.head[2] + 0.05 * height]  # type: ignore[union-attr]
                down = [k for k in kids if k.head[2] < bone.head[2] and abs(k.head[0] - bone.head[0]) > 0.01 * height]  # type: ignore[union-attr]
                if up and any(k.head[0] > bone.head[0] for k in down) and any(k.head[0] < bone.head[0] for k in down):
                    self.assign("hips", bone.name, 0.6, "anatomical")
                    break
        hips = self.src("hips")
        if hips and "head" not in self.mappings:
            chain, current = [], hips
            while True:
                kids = [c for c in arm.children(current) if self.usable(c) and c not in self.used]
                up = [c for c in kids if arm.bone(c).head[2] > arm.bone(current).head[2]]  # type: ignore[union-attr]
                if not up:
                    break
                current = max(up, key=lambda c: arm.bone(c).head[2])  # type: ignore[union-attr]
                chain.append(current)
            while chain and not arm.children(chain[-1]):
                chain.pop()  # drop end-site leaves
            if len(chain) >= 3:
                self.assign("head", chain[-1], 0.6, "anatomical", ("hips",))

    def from_position(self) -> None:
        for side in (LEFT, RIGHT):
            self._arm_by_position(side)
            self._leg_by_position(side)
            self._fingers_by_position(side)

    def _walk(self, start: str) -> list[str]:
        chain, current = [start], start
        while (nxt := self.main_child(current)) is not None:
            chain.append(nxt)
            current = nxt
        return chain

    def _arm_by_position(self, side: str) -> None:
        if all(f"{side}{p}" in self.mappings for p in ("UpperArm", "LowerArm", "Hand")):
            return
        top = self.src("upperChest") or self.src("chest") or self.src("spine")
        hips, head = self.src("hips"), self.src("head")
        if not top or not hips:
            return
        sign, arm = SIDE_SIGN[side], self.arm
        hips_bone = arm.bone(hips)
        height = arm.height or 1.0
        head_line = set(arm.ancestors(head)) | {head} if head else set()
        roots = [c for c in arm.children(top) if c not in head_line and c not in self.used
                 and self.usable(c)
                 and (arm.bone(c).head[0] - hips_bone.head[0]) * sign > 0.02 * height]  # type: ignore[union-attr]
        if not roots:
            return
        reach = lambda r: max((arm.bone(d).head[0] - hips_bone.head[0]) * sign for d in [r, *arm.descendants(r)])  # noqa: E731
        chain = self._walk(max(roots, key=lambda r: (reach(r), arm.subtree_size(r))))
        hand_at = next((i for i, b in enumerate(chain)
                        if len([c for c in arm.children(b) if self.usable(c)]) >= 3), len(chain) - 1)
        if hand_at < 2:
            return
        for vrm, idx in ((f"{side}Hand", hand_at), (f"{side}LowerArm", hand_at - 1), (f"{side}UpperArm", hand_at - 2)):
            self.assign(vrm, chain[idx], 0.6, "anatomical")
        if hand_at >= 3:
            self.assign(f"{side}Shoulder", chain[hand_at - 3], 0.55, "anatomical")

    def _leg_by_position(self, side: str) -> None:
        if all(f"{side}{p}" in self.mappings for p in ("UpperLeg", "LowerLeg", "Foot")):
            return
        hips = self.src("hips")
        if not hips:
            return
        arm, sign = self.arm, SIDE_SIGN[side]
        hips_bone = arm.bone(hips)
        head_line = set(arm.ancestors(self.src("head") or "")) if self.src("head") else set()
        low = [c for c in arm.descendants(hips) if c not in head_line and c not in self.used and self.usable(c)
               and arm.bone(c).head[2] <= hips_bone.head[2]  # type: ignore[union-attr]
               and (arm.bone(c).head[0] - hips_bone.head[0]) * sign > 0.01 * (arm.height or 1.0)]  # type: ignore[union-attr]
        if not low:
            return
        root = max(low, key=lambda c: (arm.bone(c).head[2], -arm.depth(c)))  # type: ignore[union-attr]
        chain = self._walk(root)
        if len(chain) >= 3:
            for vrm, bone in zip((f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot"), chain):
                self.assign(vrm, bone, 0.6, "anatomical")

    def _fingers_by_position(self, side: str) -> None:
        hand = self.src(f"{side}Hand")
        if not hand or any(f"{side}{f}Proximal" in self.mappings for f in FINGERS):
            return
        roots = [c for c in self.arm.children(hand) if self.usable(c) and c not in self.used
                 and len(self._walk(c)) >= 3]
        if len(roots) != 5:
            return
        heads = {r: self.arm.bone(r).head for r in roots}  # type: ignore[union-attr]

        def dist(a: str, b: str) -> float:
            return sum((x - y) ** 2 for x, y in zip(heads[a], heads[b])) ** 0.5

        thumb = max(roots, key=lambda r: sum(dist(r, o) for o in roots if o != r))
        rest = sorted((r for r in roots if r != thumb), key=lambda r: dist(thumb, r))
        for finger, root in zip(FINGERS, [thumb, *rest]):
            parts = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else (
                "Proximal", "Intermediate", "Distal")
            for part, bone in zip(parts, self._walk(root)):
                self.assign(f"{side}{finger}{part}", bone, 0.6, "anatomical")

    # -- finish -----------------------------------------------------------------
    def finish(self) -> HumanoidResult:
        result = HumanoidResult(armature=self.arm.name, profile=self.profile)
        # confidence gate
        for vrm in list(self.mappings):
            m = self.mappings[vrm]
            if m.confidence < self.min_confidence:
                result.below_threshold.append(m)
                del self.mappings[vrm]
                del self.used[m.source_bone]
                if vrm not in REQUIRED_HUMAN_BONES:
                    result.warnings.append(ConversionWarning(
                        "humanoid-low-confidence",
                        f"Optional bone {vrm} -> {m.source_bone} dropped: confidence {m.confidence:.2f} "
                        f"({m.reason}) is below {self.min_confidence:.2f}."))
        self._drop_invalid_hierarchy(result)
        result.mappings = {k: self.mappings[k] for k in ALL_HUMAN_BONES if k in self.mappings}
        result.missing_required = [b for b in REQUIRED_HUMAN_BONES if b not in result.mappings]
        result.warnings.extend(self.warnings)
        self._side_sanity(result)
        return result

    def _nearest_mapped_ancestor(self, src: str) -> str | None:
        for anc in self.arm.ancestors(src):
            if anc in self.used:
                return self.used[anc]
        return None

    def _drop_invalid_hierarchy(self, result: HumanoidResult) -> None:
        # repeat: dropping an optional bone can change another bone's nearest mapped ancestor
        for _ in range(3):
            changed = False
            for vrm, m in list(self.mappings.items()):
                rule = _PARENT_RULES.get(vrm)
                if not rule:
                    continue
                parent = self._nearest_mapped_ancestor(m.source_bone)
                if parent in rule or (vrm in ("hips",)):
                    continue
                message = (f"{vrm} ({m.source_bone}): nearest mapped ancestor is {parent or 'none'}, "
                           f"expected one of {rule}")
                if vrm in REQUIRED_HUMAN_BONES:
                    if message not in result.errors:
                        result.errors.append("invalid humanoid hierarchy: " + message)
                else:
                    del self.mappings[vrm]
                    del self.used[m.source_bone]
                    result.warnings.append(ConversionWarning("humanoid-hierarchy", f"Optional bone dropped, {message}."))
                    changed = True
            if not changed:
                break
        hips, head = self.src("hips"), self.src("head")
        if hips and head and not self.arm.is_ancestor(hips, head):
            result.errors.append(f"invalid humanoid hierarchy: hips ({hips}) is not an ancestor of head ({head})")

    def _side_sanity(self, result: HumanoidResult) -> None:
        left, right = self.src("leftHand"), self.src("rightHand")
        if left and right:
            if self.arm.bone(left).head[0] < self.arm.bone(right).head[0]:  # type: ignore[union-attr]
                result.warnings.append(ConversionWarning(
                    "humanoid-mirrored",
                    "leftHand lies at lower X than rightHand. Either the rig faces +Y / is mirrored or the "
                    "left/right assignment is swapped; verify the result before using it."))


def resolve_humanoid(
    armature: ArmatureInfo,
    explicit: dict[str, str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    profiles: tuple[BoneProfile, ...] = BONE_PROFILES,
) -> HumanoidResult:
    resolver = _Resolver(armature, min_confidence)
    resolver.from_config(explicit or {})
    resolver.from_profile(profiles)
    resolver.from_names()
    resolver.from_hierarchy()
    resolver.core_by_position()
    resolver.from_hierarchy()  # hips / head found by position unlock the torso, gaps and chains
    resolver.from_position()
    resolver.from_hierarchy()
    return resolver.finish()


__all__ = ["HumanoidError", "HumanoidResult", "resolve_humanoid", "FINGER_BONES"]
