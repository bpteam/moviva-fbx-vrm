import numpy as np

from character_vrm.core.autorig import HIERARCHY, deformation_tests, detect_frame, distance_weights, fit_skeleton
from character_vrm.tests.fixtures.mannequin import mannequin_points


def test_frame_of_mannequin():
    f = detect_frame(mannequin_points())
    assert np.allclose(f.up, (0, 0, 1)) and np.allclose(f.forward, (0, -1, 0)) and np.allclose(f.left, (1, 0, 0))


def test_too_few_points_is_refused():
    assert not fit_skeleton(np.random.rand(50, 3)).ok


def test_skeleton_landmarks_land_on_the_body():
    fit = fit_skeleton(mannequin_points())
    assert fit.ok and set(fit.bones) == set(HIERARCHY)
    assert abs(fit.bones["Hips"][0][2] - 0.95) < 0.12
    assert 1.4 < fit.bones["Neck"][0][2] < 1.6
    assert fit.bones["LeftHand"][0][0] > 0.5 > 0 > fit.bones["RightHand"][0][0] * -1 + -1  # left = +X
    assert fit.bones["RightHand"][0][0] < -0.5
    assert 0.0 < fit.bones["LeftFoot"][0][2] < 0.12 and abs(fit.bones["LeftUpperLeg"][0][0] - 0.10) < 0.04
    assert fit.measured["crotch"] and fit.measured["LeftArm"] and fit.measured["RightArm"]


def test_distance_weights_deform_the_right_parts():
    pts = mannequin_points()
    fit = fit_skeleton(pts)
    names, W = distance_weights(pts, fit)
    assert np.allclose(W.sum(axis=1), 1.0) and ((W > 0).sum(axis=1) <= 4).all()
    result = deformation_tests(pts, names, W, fit)
    assert result["status"] in ("PASS", "CHECK"), result
    assert result["tests"]["head_rotation"]["moved"] > 0
    assert result["tests"]["arm_raise_left"]["status"] != "FAIL"


def test_bad_weights_fail_the_deformation_test():
    pts = mannequin_points()
    fit = fit_skeleton(pts)
    names, _ = distance_weights(pts, fit)
    W = np.zeros((len(pts), len(names)))
    W[:, names.index("LeftUpperArm")] = 1.0  # everything follows the arm
    assert deformation_tests(pts, names, W, fit)["status"] == "FAIL"


def test_every_parent_exists():
    assert all(p is None or p in HIERARCHY for p in HIERARCHY.values())
    assert HIERARCHY["LeftLowerArm"] == "LeftUpperArm" and HIERARCHY["RightShoulder"] == "Chest"
