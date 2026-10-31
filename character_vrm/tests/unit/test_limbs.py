"""core.limbs on a figure the slice fitters get wrong: arms hanging against the body under a bob wider than them."""

import numpy as np

from character_vrm.core.limbs import (find_arms, gate_by_distance, island_cleanup, islands, smooth_weights,
                                      surface_samples)


def tube(a, b, r, rings=24, sides=10):
    """A closed tube from a to b (Z-up figure): vertices, edges, triangles."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = (b - a) / np.linalg.norm(b - a)
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = [a + t * (b - a) + r * (np.cos(q) * u + np.sin(q) * w) for t in np.linspace(0, 1, rings) for q in ang]
    V += [a, b]
    E, T = [], []
    for i in range(rings):
        for j in range(sides):
            k, k2 = i * sides + j, i * sides + (j + 1) % sides
            E.append((k, k2))
            if i + 1 < rings:
                E.append((k, k + sides))
                T += [(k, k2, k + sides), (k2, k2 + sides, k + sides)]
    ca, cb = rings * sides, rings * sides + 1
    for j in range(sides):
        E += [(ca, j), (cb, (rings - 1) * sides + j)]
        T += [(ca, j, (j + 1) % sides), (cb, (rings - 1) * sides + (j + 1) % sides, (rings - 1) * sides + j)]
    return np.array(V), np.array(E), np.array(T)


def figure():
    """Height 1.8: legs, torso, a big head with a bob wider than the shoulders, arms in a slight A-pose with the upper
    arms against the torso.
    Each part is its own piece of mesh, as on a low-poly game model."""
    parts = {
        "leg_l": tube((0.09, 0, 0.0), (0.09, 0, 0.8), 0.06), "leg_r": tube((-0.09, 0, 0.0), (-0.09, 0, 0.8), 0.06),
        "torso": tube((0, 0, 0.75), (0, 0, 1.35), 0.2), "neck": tube((0, 0, 1.33), (0, 0, 1.42), 0.05),
        "head": tube((0, 0, 1.4), (0, 0, 1.8), 0.13), "hair": tube((0, 0.02, 1.3), (0, 0.02, 1.75), 0.3),
        "arm_l": tube((0.245, 0, 1.3), (0.36, 0, 0.75), 0.045), "arm_r": tube((-0.245, 0, 1.3), (-0.36, 0, 0.75), 0.045),
    }
    V, E, T, owner, base = [], [], [], [], 0
    for name, (v, e, t) in parts.items():
        V.append(v)
        E.append(e + base)
        T.append(t + base)
        owner += [name] * len(v)
        base += len(v)
    return np.vstack(V), np.vstack(E), np.vstack(T), np.array(owner)


def test_hanging_arms_are_found_under_a_wide_bob():
    V, E, T, _ = figure()
    arms = find_arms(surface_samples(V[T], 200_000), V, E)
    assert arms is not None and set(arms) == {"Left", "Right"}
    for side, sx in (("Left", 1), ("Right", -1)):
        a = arms[side]
        assert a["wrist"][0] * sx > 0.15  # on its own side, out at the arm, not in the hair
        assert a["tip"][2] < a["wrist"][2] < a["elbow"][2] < a["shoulder"][2]
        assert 0.7 < a["tip"][2] < 0.85 and 1.15 < a["shoulder"][2] < 1.35


def test_cleanup_keeps_an_arms_piece_off_the_thigh_and_spikes_off_the_arm():
    V, E, _, owner = figure()
    names = ["Hips", "LeftUpperLeg", "LeftLowerArm", "LeftHand", "LeftFoot"]
    bones = {"Hips": (np.array([0, 0, 0.8]), np.array([0, 0, 1.0])),
             "LeftUpperLeg": (np.array([0.1, 0, 0.8]), np.array([0.1, 0, 0.4])),
             "LeftLowerArm": (np.array([0.29, 0, 1.08]), np.array([0.33, 0, 0.88])),
             "LeftHand": (np.array([0.33, 0, 0.88]), np.array([0.36, 0, 0.75])),
             "LeftFoot": (np.array([0.09, 0, 0.05]), np.array([0.09, -0.1, 0.0]))}
    W = np.zeros((len(V), 5))
    W[:, 0] = 1.0
    arm = owner == "arm_l"
    W[arm] = [0, 0.5, 0.3, 0.2, 0]  # half the arm's skin given to the thigh it hangs against
    leg = np.nonzero(owner == "leg_l")[0]
    spike = leg[np.argmin(np.abs(V[leg, 2] - 0.4))]
    W[spike] = [0, 0.6, 0.4, 0, 0]  # a vertex mid-thigh partly on the forearm
    heads = {n: b[0] for n, b in bones.items()}
    label = islands(len(V), E)
    W, gated = gate_by_distance(W, names, V, bones)
    W, stats = island_cleanup(smooth_weights(W, E), names, label, V, heads)
    assert gated >= 1 and W[spike, 2] == 0
    assert W[arm, 1].max() == 0 and stats["limb_bleed_removed"] > 0
    assert np.allclose(W.sum(axis=1), 1)
