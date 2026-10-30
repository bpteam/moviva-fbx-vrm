"""Phases 1-3: anatomical frame, measurements and landmarks are independent of axes, scale and origin."""

import math

import pytest

from character_vrm.core.anatomy.frame import build_frame
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.tests.fixtures.heads import head_mesh
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer
from character_vrm.tests.fixtures.scenes import make_scene
from character_vrm.tests.fixtures.transforms import rotation, transformed


def analysis_of(scene):
    arm = scene.armatures[0]
    return analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))


@pytest.fixture(scope="module")
def base():
    scene = make_scene()
    scene.meshes = [head_mesh()]
    return scene, analysis_of(scene)


def test_frame_of_a_z_up_blender_character(base):
    _, a = base
    assert a.frame.up == pytest.approx((0, 0, 1), abs=1e-6)
    assert a.frame.forward == pytest.approx((0, -1, 0), abs=1e-6)  # Blender import convention: facing -Y
    assert a.frame.left == pytest.approx((1, 0, 0), abs=1e-6)  # left = up x forward
    assert a.frame.confidence > 0.9
    assert a.frame.scale == pytest.approx(1.735, abs=0.02)


@pytest.mark.parametrize("r,s,t", [
    (rotation(0, 0, 0), 1.0, (0, 0, 0)),
    (rotation(90, 0, 0), 1.0, (0, 0, 0)),               # FBX-style Y-up
    (rotation(-90, 0, 180), 1.0, (0, 0, 0)),
    (rotation(33, -17, 71), 1.0, (0, 0, 0)),            # arbitrary object rotation
    (rotation(0, 0, 0), 100.0, (0, 0, 0)),              # centimetre-scale rig
    (rotation(12, 40, -5), 0.37, (5.0, -3.0, 12.0)),    # scale + origin + rotation together
])
def test_frame_measurements_and_landmarks_are_invariant(base, r, s, t):
    scene, ref = base
    a = analysis_of(transformed(scene, r, s, t))
    assert a is not None and a.frame.confidence > 0.9
    # normalised measurements do not change
    for name, value in ref.measurements.normalized().items():
        assert a.measurements.normalized()[name] == pytest.approx(value, rel=0.02, abs=2e-3), name
    # landmarks expressed in the frame (normalised) do not change either
    for name, lm in ref.landmarks.items():
        want = ref.frame.to_normalized(lm.position)
        got = a.frame.to_normalized(a.landmarks[name].position)
        assert got == pytest.approx(want, abs=3e-3), name


def test_frame_up_follows_the_body_not_the_world(base):
    scene, _ = base
    a = analysis_of(transformed(scene, rotation(90, 0, 0)))
    # +Z up rotated 90 deg about X becomes -Y up
    assert a.frame.up == pytest.approx((0, -1, 0), abs=1e-6) or a.frame.up == pytest.approx((0, 1, 0), abs=1e-6)
    hips_to_head = [b for b in a.joints]  # sanity: the joint set is complete
    assert "head" in hips_to_head and "hips" in hips_to_head


def test_local_coordinates_round_trip(base):
    _, a = base
    p = (0.3, -0.2, 1.1)
    assert a.frame.to_world(a.frame.to_local(p)) == pytest.approx(p, abs=1e-9)


def test_mirrored_rig_is_flagged_and_low_confidence():
    """Toes pointing backwards relative to the left/right-derived forward: sides swapped or rig mirrored."""
    frame, warnings = build_frame({
        "hips": (0, 0, 1), "head": (0, 0, 1.6), "top": (0, 0, 1.8),
        "leftUpperArm": (0.2, 0, 1.45), "rightUpperArm": (-0.2, 0, 1.45),
        "leftFoot": (0.1, 0, 0.1), "rightFoot": (-0.1, 0, 0.1),
        "leftToes": (0.1, 0.12, 0.03), "rightToes": (-0.1, 0.12, 0.03),  # +Y = backwards for this frame
        "floor": (0, 0, 0)})
    assert frame is not None and frame.confidence < 0.6
    assert any(w.code == "frame-mirrored-rig" for w in warnings)


def test_frame_is_unavailable_without_hips_and_head():
    frame, warnings = build_frame({"leftUpperArm": (1, 0, 0), "rightUpperArm": (-1, 0, 0)})
    assert frame is None and warnings[0].code == "frame-unavailable"


def test_measurements_are_normalised_never_absolute(base):
    _, a = base
    n = a.measurements.normalized()
    assert n["total_height"] == pytest.approx(1.0)
    assert 0.2 < n["shoulder_width"] < 0.3 and 0.1 < n["head_height"] < 0.14
    m = a.measurements.values["head_width"]
    assert m.source.value == "geometry" and 0 < m.confidence <= 1


def test_measurement_provenance_marks_estimates_as_not_geometry():
    scene = make_scene()  # no mesh vertices at all
    a = analysis_of(scene)
    assert a.measurements.values["chest_width"].source.value == "skeleton"
    assert a.measurements.values["head_width"].confidence < 0.5  # ratio estimate, honest about it


def test_landmarks_carry_confidence_and_source(base):
    _, a = base
    for name in ("head_top", "neck", "hips", "waist", "left_shoulder", "right_wrist", "left_knee", "left_ankle",
                 "mouth", "nose", "chin", "left_eye"):
        lm = a.landmarks[name]
        assert 0 < lm.confidence <= 1 and lm.source.value in ("skeleton", "geometry", "generated")
    assert a.landmarks["waist"].method == "detected"
    assert a.landmarks["left_shoulder"].position[0] > 0 > a.landmarks["right_shoulder"].position[0]
