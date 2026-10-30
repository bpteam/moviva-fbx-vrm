"""Humanoid limbs from geometry and ancestry, not from subtree sizes (pure).

Helper bones (twist / roll / aim bones, cloth and skirt chains, breast and hood bones) sit *next to* the limb chains
and often have larger subtrees, so "follow the biggest child" picks them. The extremities do not lie: the feet are
the lowest bones, the hands (with their fingers) the ones farthest out to the sides. A limb is the ancestor path from
the torso to its extremity; helper siblings are never on that path.

- up: Blender's +Z (the inspectors hand everything over Z-up);
- lateral: the dominant horizontal spread of the upper body (arms, shoulders);
- forward: foot -> toes; left = up x forward (the frame convention). No name is read.
"""

from __future__ import annotations

from ..model import ArmatureInfo
from .vec import Vec3, cross, dot, normalize, principal_axis, sub

UP: Vec3 = (0.0, 0.0, 1.0)


def _path(arm: ArmatureInfo, top: str, bottom: str) -> list[str]:
    """Bones from just below ``top`` down to ``bottom`` (``top`` excluded)."""
    chain = [bottom]
    for a in arm.ancestors(bottom):
        if a == top:
            return list(reversed(chain))
        chain.append(a)
    return []


def _lca(arm: ArmatureInfo, names: list[str]) -> str | None:
    return arm.lowest_common_ancestor(names)


def _near(a: Vec3, b: Vec3, tol: float) -> bool:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) <= tol * tol


def _deepest(arm: ArmatureInfo, name: str) -> str:
    """Follow the child with the most descendants (then the longest) to a leaf."""
    cur = name
    while True:
        kids = [c for c in arm.children(cur) if arm.bone(c).deform]
        if not kids:
            return cur
        cur = max(kids, key=lambda c: (arm.subtree_size(c), arm.bone(c).length))


def _two_longest(arm: ArmatureInfo, bones: list[str]) -> list[int]:
    idx = sorted(range(len(bones)), key=lambda i: -arm.bone(bones[i]).length)[:2]
    return sorted(idx)


def infer_limbs(arm: ArmatureInfo) -> tuple[dict[str, tuple[str, float, str]], list[str]]:
    """``({vrm_bone: (bone, topology score, method)}, notes)``; empty when the extremities cannot be found."""
    notes: list[str] = []
    bones = [b for b in arm.bones if b.deform] or list(arm.bones)
    if len(bones) < 12:
        return {}, ["too few bones"]
    zs = [dot(b.head, UP) for b in bones]
    floor, top = min(zs), max(dot(b.tail, UP) for b in bones)
    height = max(top - floor, 1e-9)
    # lateral axis: horizontal spread of the upper body
    upper = [b.head for b in bones if dot(b.head, UP) > floor + 0.5 * height]
    flat = [(p[0], p[1], 0.0) for p in upper]
    pa = principal_axis(flat) if len(flat) >= 3 else None
    lateral = normalize((pa[0][0], pa[0][1], 0.0)) if pa else (1.0, 0.0, 0.0)
    if lateral is None:
        lateral = (1.0, 0.0, 0.0)
    cx = sum(dot(b.head, lateral) for b in bones) / len(bones)

    def side_of(p: Vec3) -> float:
        return dot(p, lateral) - cx

    # feet: the lowest bone on each side of the midline
    low = sorted(bones, key=lambda b: min(dot(b.head, UP), dot(b.tail, UP)))
    feet = {}
    for b in low:
        s = 1 if side_of(b.head) > 0 else -1
        if s not in feet and abs(side_of(b.head)) > 0.02 * height and dot(b.head, UP) < floor + 0.2 * height:
            feet[s] = b.name
        if len(feet) == 2:
            break
    # hands: the bone farthest out on each side, above the hips
    wide = sorted((b for b in bones if dot(b.head, UP) > floor + 0.35 * height), key=lambda b: -abs(side_of(b.head)))
    hands = {}
    for b in wide:
        s = 1 if side_of(b.head) > 0 else -1
        if s not in hands:
            hands[s] = b.name
        if len(hands) == 2:
            break
    if len(feet) < 2 or len(hands) < 2:
        return {}, ["extremities not found"]
    # in a relaxed pose the elbow may stick out further than the hand: continue to the end of that limb
    hands = {s: _deepest(arm, n) for s, n in hands.items()}
    hips = _lca(arm, [feet[1], feet[-1], hands[1], hands[-1]])
    leg_root = _lca(arm, [feet[1], feet[-1]])
    hub = _lca(arm, [hands[1], hands[-1]])
    if hips is None or leg_root is None or hub is None:
        return {}, ["no common ancestor of the limbs"]
    out: dict[str, tuple[str, float, str]] = {"hips": (hips, 0.92, "limb-ancestry")}

    # forward from the feet: ankle -> toes
    legs: dict[int, list[str]] = {}
    for s, foot_tip in feet.items():
        path = _path(arm, leg_root, foot_tip)
        # drop a trailing end/tip bone: much shorter than its parent and childless
        while len(path) > 4 and not arm.children(path[-1]) and arm.bone(path[-1]).length < 0.5 * arm.bone(path[-2]).length:
            path.pop()
        legs[s] = path
    forward = None
    for s, path in legs.items():
        if len(path) < 3:
            return {}, ["leg chain too short"]
        i, j = _two_longest(arm, path[:max(2, len(path) - 1)])
        foot_i = j + 1
        if foot_i < len(path):
            f = arm.bone(path[foot_i])
            toe = arm.bone(path[foot_i + 1]).tail if foot_i + 1 < len(path) else f.tail
            d = sub(toe, f.head)
            d = (d[0], d[1], 0.0)
            n = normalize(d)
            if n is not None:
                forward = n if forward is None else normalize((forward[0] + n[0], forward[1] + n[1], 0.0))
    if forward is None:
        return {}, ["no foot direction"]
    left_axis = normalize(cross(UP, forward)) or lateral
    left_sign = 1 if dot(lateral, left_axis) > 0 else -1  # which side of ``lateral`` is the character's left
    side_name = {left_sign: "left", -left_sign: "right"}

    for s, path in legs.items():
        side = side_name[s]
        i, j = _two_longest(arm, path[:max(2, len(path) - 1)])
        out[f"{side}UpperLeg"] = (path[i], 0.92, "limb-ancestry")
        out[f"{side}LowerLeg"] = (path[j], 0.92, "limb-ancestry")
        if j + 1 < len(path):
            out[f"{side}Foot"] = (path[j + 1], 0.9, "limb-ancestry")
        if j + 1 < len(path):
            # toes: the foot's child that carries toes (a childless "share" helper at the same spot is not it)
            kids = [c for c in arm.children(path[j + 1]) if arm.bone(c).deform]
            if kids:
                toes = max(kids, key=lambda c: (len(arm.children(c)), arm.subtree_size(c), arm.bone(c).length))
                out[f"{side}Toes"] = (toes, 0.85, "limb-ancestry")

    for s, tip in hands.items():
        side = side_name[s]
        path = _path(arm, hub, tip)
        # the hand: the first bone with three or more *chains* below it (fingers); twist / share helpers are leaves
        hand_i = next((k for k, n in enumerate(path) if k >= 2 and
                       sum(1 for c in arm.children(n) if arm.bone(c).deform and arm.children(c)) >= 3), None)
        if hand_i is None:
            hand_i = len(path) - 1
            if hand_i >= 3 and not arm.children(path[-1]) and arm.bone(path[-1]).length < 0.5 * arm.bone(path[-2]).length:
                hand_i -= 1
        before = path[:hand_i]
        if len(before) < 2:
            notes.append(f"{side} arm chain too short")
            continue
        i, j = _two_longest(arm, before)
        out[f"{side}UpperArm"] = (before[i], 0.9, "limb-ancestry")
        out[f"{side}LowerArm"] = (before[j], 0.9, "limb-ancestry")
        out[f"{side}Hand"] = (path[hand_i], 0.9, "limb-ancestry")
        if i >= 1:
            out[f"{side}Shoulder"] = (before[i - 1], 0.85, "limb-ancestry")

    # torso: hips -> hub (the hub itself is the last torso bone)
    torso = _path(arm, hips, hub)
    if torso:
        out["spine"] = (torso[0], 0.9, "limb-ancestry")
        if len(torso) >= 3:
            out["chest"] = (torso[1] if len(torso) == 3 else torso[len(torso) // 2], 0.85, "limb-ancestry")
            out["upperChest"] = (torso[-1], 0.85, "limb-ancestry")
        elif len(torso) == 2:
            out["chest"] = (torso[1], 0.85, "limb-ancestry")

    # neck / head: from the hub towards the highest bone that is not on an arm path
    arm_bones = {n for s, tip in hands.items() for n in _path(arm, hub, tip)}
    above = [b for b in bones if b.name not in arm_bones and arm.is_ancestor(hub, b.name)]
    if above:
        # the head is where the skull is: the bone whose own head is highest among long-enough, non-leaf chains
        highest = max(above, key=lambda b: dot(b.head, UP))
        path = _path(arm, hub, highest.name)
        # the head joint: the first bone on that path above shoulder height that branches, else the last long bone
        shoulder_h = max(dot(arm.bone(out[f"{s}UpperArm"][0]).head, UP) for s in ("left", "right")
                         if f"{s}UpperArm" in out) if any(f"{s}UpperArm" in out for s in ("left", "right")) else None
        head_i = None
        for k, n in enumerate(path):
            if k >= 1 and len(arm.children(n)) >= 2 and (shoulder_h is None or dot(arm.bone(n).head, UP) > shoulder_h):
                head_i = k
                break
        if head_i is None:
            head_i = min(1, len(path) - 1)
        # a facial root at the very same joint (CC's FacialBone) belongs to the head: step up to its parent
        while head_i >= 2 and arm.bone(path[head_i]).head == arm.bone(path[head_i - 1]).head or (
                head_i >= 2 and _near(arm.bone(path[head_i]).head, arm.bone(path[head_i - 1]).head, 0.005 * height)):
            head_i -= 1
        # a lone bone above the arms is as likely a neck as a head: not enough for automatic mapping
        out["head"] = (path[head_i], 0.85 if len(path) >= 2 else 0.5, "limb-ancestry")
        if head_i >= 1:
            out["neck"] = (path[0], 0.85, "limb-ancestry")
    return out, notes
