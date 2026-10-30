"""Phase 7 and 10: canonical face template fitting on geometry (no topology assumptions), search region, mouth analysis."""

import math

import pytest

from character_vrm.core.anatomy.face import CANONICAL_FACE, apply_similarity, fit_similarity_2d
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.anatomy.regions import eye_usability, mouth_usability
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.tests.fixtures.heads import eyeball_mesh, head_mesh
from character_vrm.tests.fixtures.scenes import make_scene
from character_vrm.tests.fixtures.transforms import rotation, transformed


def analyse(meshes, **kw):
    scene = make_scene()
    scene.meshes = meshes
    arm = scene.armatures[0]
    return scene, analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75), **kw)


@pytest.fixture(scope="module")
def fitted():
    return analyse([head_mesh()])


def test_similarity_fit_recovers_scale_rotation_translation():
    src = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (1.0, 1.5)]
    theta, s, t = math.radians(20), 2.5, (0.3, -1.2)
    dst = [(s * (math.cos(theta) * x - math.sin(theta) * y) + t[0], s * (math.sin(theta) * x + math.cos(theta) * y) + t[1])
           for x, y in src]
    fit = fit_similarity_2d(src, dst)
    assert fit[0] == pytest.approx(s) and fit[1] == pytest.approx(theta) and fit[2] == pytest.approx(t)
    assert apply_similarity(fit, (1.0, 1.0)) == pytest.approx(
        (s * (math.cos(theta) - math.sin(theta)) + t[0], s * (math.sin(theta) + math.cos(theta)) + t[1]))
    assert fit_similarity_2d([(0, 0)], [(1, 1)]) is None


def test_template_is_symmetric_and_ordered_like_a_face():
    for name, (u, v) in CANONICAL_FACE.items():
        if name.startswith("left_"):
            assert CANONICAL_FACE["right_" + name[5:]] == (-u, v), name
    v = {k: p[1] for k, p in CANONICAL_FACE.items()}
    assert v["chin"] < v["lower_lip"] < v["mouth_center"] < v["upper_lip"] < v["nose_tip"] < v["left_eye"] < v["left_eyebrow"]


def test_landmarks_land_where_the_face_is(fitted):
    _, a = fitted
    f, fr = a.face, a.frame
    chin_z, top_z = 1.585, 1.815  # the synthetic ellipsoid, world Z
    face_h = top_z - chin_z
    world = {n: lm.position for n, lm in f.landmarks.items()}
    assert world["mouth_center"][2] == pytest.approx(chin_z + 0.19 * face_h, abs=0.06 * face_h)
    assert world["left_eye"][2] == pytest.approx(chin_z + 0.53 * face_h, abs=0.06 * face_h)
    assert world["nose_tip"][2] == pytest.approx(chin_z + 0.365 * face_h, abs=0.06 * face_h)
    # the face looks along -Y: surface landmarks sit in front of the head centre
    assert world["mouth_center"][1] < -0.05 and world["nose_tip"][1] < world["mouth_center"][1]
    assert world["left_eye"][0] > 0 > world["right_eye"][0]
    assert f.quality > 0.8 and all(0 < lm.confidence <= 1 for lm in f.landmarks.values())
    assert {"left_upper_eyelid", "right_lower_eyelid", "mouth_left", "upper_lip", "chin", "left_cheek"} <= set(f.landmarks)


def test_nose_tip_is_refined_from_the_midline_profile(fitted):
    _, a = fitted
    assert a.face.landmarks["nose_tip"].method == "template-fit+profile-refined"
    assert a.face.evidence["nose_tip"]["profile"] > 0.9


def test_fit_is_invariant_to_rotation_scale_and_origin():
    scene, ref = analyse([head_mesh()])
    moved = transformed(scene, rotation(40, -25, 65), 0.5, (3, 1, -2))
    arm = moved.armatures[0]
    b = analyze_human(moved, arm, resolve_humanoid(arm, None, 0.75))
    for name in ("mouth_center", "nose_tip", "left_eye", "chin", "right_cheek"):
        want = ref.frame.to_normalized(ref.face.landmarks[name].position)
        got = b.frame.to_normalized(b.face.landmarks[name].position)
        assert got == pytest.approx(want, abs=0.01), name


def test_eyeball_meshes_are_not_part_of_the_skin_surface_but_stay_in_the_search_region():
    eyes = [eyeball_mesh("EyeL", (0.031, -0.075, 1.707), 0.011), eyeball_mesh("EyeR", (-0.031, -0.075, 1.707), 0.011)]
    _, a = analyse([head_mesh(), *eyes])
    roles = {p.role for p in a.face.geometry.points}
    assert {"skin", "eye"} <= roles
    assert a.face.quality > 0.8


def test_face_search_region_uses_head_weights_not_the_whole_mesh():
    body_only = head_mesh("Body", weight_bone="mixamorig:Spine")  # same shape but skinned to the spine
    _, a = analyse([body_only])
    assert a.face is None and any(w.code == "face-unavailable" for w in a.warnings)


def test_too_little_facial_geometry_gives_no_face():
    _, a = analyse([head_mesh(grid=8)])
    assert a.face is None or a.face.quality < 0.5 or not mouth_usability(a.face, a.frame).usable


def test_flat_mouth_geometry_is_reported_as_unusable():
    _, a = analyse([head_mesh(grid=40)])
    assert a.face is not None
    use = mouth_usability(a.face, a.frame)
    assert not use.usable and "too little geometry" in use.reason
    assert "geometry-deformation" not in a.mouth.approaches


def test_dense_mouth_and_eyes_are_usable(fitted):
    _, a = fitted
    assert mouth_usability(a.face, a.frame).usable
    assert eye_usability(a.face, a.frame, "left").usable and eye_usability(a.face, a.frame, "right").usable
    assert "geometry-deformation" in a.mouth.approaches


def test_existing_visemes_are_listed_as_a_mouth_approach():
    _, a = analyse([head_mesh()], existing_visemes=["aa", "ih"])
    assert a.mouth.approaches[0] == "mouth-morphs" and a.mouth.mouth_morphs == ["aa", "ih"]
