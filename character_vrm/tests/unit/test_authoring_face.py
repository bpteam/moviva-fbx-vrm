"""Phases 8, 9, 11, 12: existing expressions win; generated blink / visemes / basic expressions are local, symmetric,
anatomy-scaled, namespaced and gated."""

import copy

import pytest

from character_vrm.core.anatomy.confidence import Thresholds
from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.anatomy.regions import eye_ellipse, mouth_ellipse
from character_vrm.core.authoring.blink import plan_blink
from character_vrm.core.authoring.expressions import decide_existing, plan_face_authoring
from character_vrm.core.authoring.face_plan import FaceAuthoringPlan
from character_vrm.core.authoring.visemes import plan_visemes
from character_vrm.core.expressions import resolve_expressions
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.tests.fixtures.heads import eyeball_mesh, head_mesh
from character_vrm.tests.fixtures.scenes import ARKIT_KEYS, VROID_KEYS, make_scene
from character_vrm.tests.fixtures.transforms import transformed


def build(face_keys=(), meshes=None, scene_transform=None):
    scene = make_scene(face_keys=list(face_keys))
    scene.meshes = meshes or [head_mesh()]
    for key in face_keys:
        scene.shape_keys[:] = [k for k in scene.shape_keys]
        scene.shape_keys[-1].mesh = scene.meshes[0].name
    if scene_transform:
        scene = transformed(scene, *scene_transform)
    arm = scene.armatures[0]
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    ex = resolve_expressions(scene, {}, 0.65, {m.name for m in scene.meshes})
    return scene, a, ex


@pytest.fixture(scope="module")
def bare():
    return build()


def generated(plan):
    return {k.name: k for k in plan.shape_keys}


def test_everything_missing_is_generated_with_the_prosopon_namespace(bare):
    _, a, ex = bare
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    keys = generated(plan)
    assert set(keys) == {"Prosopon_BlinkLeft", "Prosopon_BlinkRight", "Prosopon_aa", "Prosopon_ih", "Prosopon_ou",
                         "Prosopon_ee", "Prosopon_oh"}
    assert all(k.name.startswith("Prosopon_") for k in keys.values()) and plan.validate(a) == []
    assert plan.decisions["blinkLeft"].status == "GENERATED" and plan.decisions["aa"].status == "GENERATED"
    assert plan.decisions["happy"].status == "MISSING"  # basic expressions are opt-in
    maps = plan.expression_mappings()
    assert set(maps) >= {"blink", "blinkLeft", "blinkRight", "aa", "ih", "ou", "ee", "oh"}
    assert {b.shape_key for b in maps["blink"].binds} == {"Prosopon_BlinkLeft", "Prosopon_BlinkRight"}
    assert all(m.kind == "generated-geometry" and 0 < m.confidence <= 1 for m in maps.values())


def test_existing_vroid_and_arkit_morphs_always_take_priority():
    _, a, ex = build(VROID_KEYS)
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=True)
    assert plan.decisions["aa"].status in ("SOURCE", "MAPPED") and plan.decisions["blinkLeft"].status in ("SOURCE", "MAPPED")
    assert not [k for k in plan.shape_keys if k.vrm_expression in ("aa", "ih", "ou", "ee", "oh", "blinkLeft", "blinkRight")]
    assert "blinkLeft" not in plan.expression_mappings()  # nothing may replace the source mapping
    _, a, ex = build(ARKIT_KEYS)
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=True)
    assert plan.decisions["blinkLeft"].status in ("SOURCE", "MAPPED")
    assert plan.decisions["happy"].status == "MAPPED" and "mouthSmile" in plan.decisions["happy"].detail
    assert not [k for k in plan.shape_keys if k.vrm_expression in ("happy", "blinkLeft", "aa")]


def test_only_the_missing_visemes_are_generated():
    _, a, ex = build(["A", "I"])  # generic A/I morphs exist for aa/ih
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    names = set(generated(plan))
    assert "Prosopon_aa" not in names and {"Prosopon_ou", "Prosopon_ee", "Prosopon_oh"} <= names


def test_blink_moves_only_the_eyelid_regions(bare):
    _, a, _ = bare
    keys, skipped = plan_blink(a)
    left = next(k for k in keys if k.name == "Prosopon_BlinkLeft")
    assert not skipped and left.vertex_count > 40
    by_pos = {(p.mesh, p.index): p.local for p in a.face.geometry.points}
    ell_l, ell_r = eye_ellipse(a.face, a.frame, "left"), eye_ellipse(a.face, a.frame, "right")
    for mesh, verts in left.offsets.items():
        for idx in verts:
            assert ell_l.contains(by_pos[(mesh, idx)])           # inside the *left* eye region only
            assert not ell_r.contains(by_pos[(mesh, idx)])
    # upper lid moves down, lower lid moves up, by amounts of the measured lid gap
    ups = [a.frame.vector_to_local(d)[1] for m, v in left.offsets.items() for i, d in v.items() if by_pos[(m, i)][1] > ell_l.cy]
    downs = [a.frame.vector_to_local(d)[1] for m, v in left.offsets.items() for i, d in v.items() if by_pos[(m, i)][1] < ell_l.cy]
    assert max(ups) < 0 < min(downs) and abs(min(ups)) > abs(max(downs))


def test_blink_does_not_touch_eyeballs_brows_or_nose():
    eyes = [eyeball_mesh("EyeL", (0.031, -0.075, 1.707), 0.011), eyeball_mesh("EyeR", (-0.031, -0.075, 1.707), 0.011)]
    _, a, ex = build(meshes=[head_mesh(), *eyes])
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    for key in plan.shape_keys:
        assert "EyeL" not in key.offsets and "EyeR" not in key.offsets, key.name  # the eye mesh is preserved
    assert plan.validate(a) == []


def test_generated_keys_are_left_right_symmetric(bare):
    _, a, ex = bare
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    keys = generated(plan)
    l, r = keys["Prosopon_BlinkLeft"], keys["Prosopon_BlinkRight"]
    assert l.vertex_count == pytest.approx(r.vertex_count, rel=0.1) and l.max_offset == pytest.approx(r.max_offset, rel=0.05)
    by_pos = {(p.mesh, p.index): p.local for p in a.face.geometry.points}
    for name in ("Prosopon_aa", "Prosopon_ee", "Prosopon_ou"):
        left = sum(a.frame.vector_to_local(d)[0] for m, v in keys[name].offsets.items() for i, d in v.items()
                   if by_pos[(m, i)][0] > 0)
        right = sum(a.frame.vector_to_local(d)[0] for m, v in keys[name].offsets.items() for i, d in v.items()
                    if by_pos[(m, i)][0] < 0)
        assert left == pytest.approx(-right, abs=1e-6 + 0.02 * (abs(left) + abs(right))), name  # mirrored x motion


def test_visemes_are_visibly_different_and_plausible(bare):
    _, a, ex = bare
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    keys = generated(plan)
    by_pos = {(p.mesh, p.index): p.local for p in a.face.geometry.points}

    def stats(key):
        dy = [a.frame.vector_to_local(d) for m, v in key.offsets.items() for i, d in v.items()]
        return {"open": -min(d[1] for d in dy), "z": max(d[2] for d in dy),
                "spread": max(d[0] * (1 if by_pos[(m, i)][0] > 0 else -1) for m, v in key.offsets.items()
                              for i, d in ((i, a.frame.vector_to_local(x)) for i, x in v.items()))}
    aa, ih, ou, ee, oh = (stats(keys[f"Prosopon_{n}"]) for n in ("aa", "ih", "ou", "ee", "oh"))
    assert aa["open"] > oh["open"] > ou["open"] and aa["open"] > 2 * ee["open"]    # jaw: aa > oh > ou, ee small
    assert ou["z"] > 2 * aa["z"] and ou["spread"] < ee["spread"]                        # ou pursed / forward, ee wide
    assert ee["spread"] > ih["spread"] > 0 > ou["spread"] * 0 - 1e-12
    face_h = a.face.head_height
    assert max(k.max_offset for k in keys.values()) < 0.25 * face_h                    # no face tearing


def test_offsets_scale_with_anatomy_not_world_constants(bare):
    scene, a, ex = bare
    _, big_a, big_ex = build(scene_transform=(None, 2.0, (0, 0, 0)))
    small = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    big = plan_face_authoring(big_a, big_ex, generate_visemes_blink=True, generate_basic=False)
    for name in ("Prosopon_aa", "Prosopon_BlinkLeft", "Prosopon_ou"):
        assert generated(big)[name].max_offset == pytest.approx(2.0 * generated(small)[name].max_offset, rel=0.03), name
        assert generated(big)[name].vertex_count == generated(small)[name].vertex_count


def test_low_confidence_or_flat_mouth_skips_with_a_reason_and_generates_nothing():
    _, a, ex = build(meshes=[head_mesh(grid=40)])
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=True)
    assert plan.shape_keys == []
    reasons = {s.what: s.reason for s in plan.skipped}
    assert "too little geometry" in reasons["aa"] and "eyelid" in reasons["blinkLeft"]
    assert plan.decisions["aa"].status == "MISSING"
    _, a, ex = build()
    strict = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False,
                                 thresholds=Thresholds(generated_face=1.01))
    assert strict.shape_keys == [] and all("below" in s.reason for s in strict.skipped)


def test_no_face_no_generation():
    scene = make_scene()
    arm = scene.armatures[0]
    a = analyze_human(scene, arm, resolve_humanoid(arm, None, 0.75))
    ex = resolve_expressions(scene, {}, 0.65, set())
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=True)
    assert plan.shape_keys == [] and any("no face analysis" in s.reason for s in plan.skipped)


def test_generation_can_be_switched_off():
    _, a, ex = build()
    plan = plan_face_authoring(a, ex, generate_visemes_blink=False, generate_basic=False)
    assert plan.shape_keys == [] and any(s.reason == "generation disabled" for s in plan.skipped)


def test_basic_expressions_only_with_the_explicit_flag(bare):
    _, a, ex = bare
    off = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    assert not {"Prosopon_happy", "Prosopon_browUp"} & set(generated(off))
    on = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=True)
    keys = generated(on)
    assert {"Prosopon_happy", "Prosopon_sad", "Prosopon_surprised", "Prosopon_browUp", "Prosopon_browDown",
            "Prosopon_squint"} <= set(keys)
    maps = on.expression_mappings()
    assert {"happy", "sad", "surprised"} <= set(maps)
    assert keys["Prosopon_browUp"].vrm_expression is None  # no VRM preset: authored, not bound
    assert on.validate(a) == []


def test_happy_lifts_corners_and_browup_lifts_brows(bare):
    _, a, ex = bare
    on = plan_face_authoring(a, ex, generate_visemes_blink=False, generate_basic=True)
    keys = generated(on)
    by_pos = {(p.mesh, p.index): p.local for p in a.face.geometry.points}
    mouth = mouth_ellipse(a.face, a.frame)
    corner_dy = [a.frame.vector_to_local(d)[1] for m, v in keys["Prosopon_happy"].offsets.items() for i, d in v.items()
                 if abs(by_pos[(m, i)][0] - mouth.cx) > 0.4 * a.face.mouth_width and abs(by_pos[(m, i)][1] - mouth.cy) < 0.1 * a.face.head_height]
    assert corner_dy and sum(corner_dy) / len(corner_dy) > 0
    brow = [a.frame.vector_to_local(d)[1] for v in keys["Prosopon_browUp"].offsets.values() for d in v.values()]
    assert min(brow) >= 0 and max(brow) > 0.02 * a.face.head_height


def test_plan_validation_rejects_tearing_wrong_namespace_and_protected_areas(bare):
    _, a, ex = bare
    plan = plan_face_authoring(a, ex, generate_visemes_blink=True, generate_basic=False)
    assert plan.validate(a) == []
    bad = copy.deepcopy(plan)
    key = next(k for k in bad.shape_keys if k.name == "Prosopon_aa")
    first_mesh = next(iter(key.offsets))
    idx = next(iter(key.offsets[first_mesh]))
    key.offsets[first_mesh][idx] = (0.0, 0.0, 5.0)     # a huge offset
    key.max_offset = 5.0
    key.name = "aa"                                    # and a name outside the namespace
    issues = bad.validate(a)
    assert any("namespace" in i.message for i in issues)
    key.name = "Prosopon_aa"
    assert any("exceeds" in i.message for i in bad.validate(a))
    assert "Prosopon_aa" not in bad._valid_names() and "aa" not in bad.expression_mappings()  # invalid keys are never bound
