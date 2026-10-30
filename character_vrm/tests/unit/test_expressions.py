import pytest

from character_vrm.core.config import ExpressionOverride
from character_vrm.core.expressions import ExpressionConfigError, norm, resolve_expressions
from character_vrm.tests.fixtures.scenes import ARKIT_KEYS, VROID_KEYS, make_scene


def resolve(keys, **kw):
    return resolve_expressions(make_scene(face_keys=keys), **kw)


def test_no_shape_keys_means_no_expressions_and_nothing_is_invented():
    result = resolve([])
    assert result.mappings == {} and result.capability == "none"
    assert set(result.missing) >= {"blink", "aa", "happy", "lookUp"}


def test_vroid_names_map_directly():
    result = resolve(VROID_KEYS)
    assert result.status("aa") == "available"
    assert result.mappings["aa"].binds[0].shape_key == "Fcl_MTH_A"
    assert result.mappings["aa"].reason == "profile:vroid"
    assert result.mappings["relaxed"].binds[0].shape_key == "Fcl_ALL_Fun"
    assert result.mappings["happy"].binds[0].shape_key == "Fcl_ALL_Joy"
    assert result.mappings["blinkLeft"].binds[0].shape_key == "Fcl_EYE_Close_L"
    assert result.capability == "full"


def test_aiueo_direct():
    result = resolve(["A", "I", "U", "E", "O", "Blink"])
    for vrm, key in (("aa", "A"), ("ih", "I"), ("ou", "U"), ("ee", "E"), ("oh", "O"), ("blink", "Blink")):
        assert [b.shape_key for b in result.mappings[vrm].binds] == [key]
        assert result.mappings[vrm].kind == "direct"


def test_oculus_visemes():
    result = resolve(["viseme_aa", "viseme_I", "viseme_U", "viseme_E", "viseme_O", "viseme_PP"])
    assert set(result.mappings) == {"aa", "ih", "ou", "ee", "oh"}


def test_arkit_detected_and_compounds_generated():
    result = resolve(ARKIT_KEYS)
    assert result.arkit == "likely" and result.arkit_coverage == 1.0
    assert result.capability == "full"
    aa = result.mappings["aa"]
    assert aa.kind == "compound" and result.status("aa") == "generated"
    assert {b.shape_key for b in aa.binds} == {"jawOpen"}
    oh = {b.shape_key: b.weight for b in result.mappings["oh"].binds}
    assert set(oh) == {"mouthFunnel", "jawOpen"} and oh["mouthFunnel"] > oh["jawOpen"]
    ou = {b.shape_key for b in result.mappings["ou"].binds}
    assert ou == {"mouthPucker", "jawOpen"}
    ih = {b.shape_key for b in result.mappings["ih"].binds}
    assert {"mouthSmileLeft", "mouthSmileRight"} <= ih
    assert {b.shape_key for b in result.mappings["blink"].binds} == {"eyeBlinkLeft", "eyeBlinkRight"}
    assert [b.shape_key for b in result.mappings["blinkLeft"].binds] == ["eyeBlinkLeft"]


def test_arkit_eye_direction_compounds():
    result = resolve(ARKIT_KEYS)
    left = {b.shape_key for b in result.mappings["lookLeft"].binds}
    assert left == {"eyeLookOutLeft", "eyeLookInRight"}
    assert all(name in result.mappings for name in ("lookUp", "lookDown", "lookLeft", "lookRight"))


def test_compound_requires_its_principal_shape_keys():
    result = resolve(["mouthSmileLeft", "eyeBlinkLeft"])  # no jawOpen / funnel / pucker / right side
    assert "aa" not in result.mappings and "oh" not in result.mappings and "ou" not in result.mappings
    assert "ih" not in result.mappings  # needs both smile sides
    assert "blinkLeft" in result.mappings and "blink" not in result.mappings
    assert result.arkit == "none"


def test_direct_beats_compound():
    result = resolve(ARKIT_KEYS + ["Blink", "A"])
    assert result.mappings["blink"].kind == "direct"
    assert result.mappings["aa"].kind == "direct"


def test_blink_is_composed_from_direct_left_right():
    result = resolve(["Blink_L", "Blink_R"])
    assert result.mappings["blink"].reason == "blinkLeft+blinkRight"
    assert {b.shape_key for b in result.mappings["blink"].binds} == {"Blink_L", "Blink_R"}


def test_binds_cover_every_mesh_that_carries_the_key():
    scene = make_scene(face_keys=["Blink"])
    from character_vrm.core.model import ShapeKeyInfo
    scene.shape_keys.append(ShapeKeyInfo("Body", "Blink"))
    binds = resolve_expressions(scene).mappings["blink"].binds
    assert {b.mesh for b in binds} == {"Face", "Body"}


def test_meshes_of_excluded_armatures_are_ignored():
    scene = make_scene(face_keys=["Blink"])
    assert resolve_expressions(scene, included_meshes={"Body"}).mappings == {}


def test_config_override_replaces_and_disables():
    scene = make_scene(face_keys=ARKIT_KEYS + ["Custom_A", "Custom_B"])
    result = resolve_expressions(scene, {
        "aa": ExpressionOverride(binds={"Custom_A": 0.7, "Custom_B": 0.3}),
        "blink": ExpressionOverride(disabled=True),
    })
    assert result.mappings["aa"].kind == "config" and result.mappings["aa"].confidence == 1.0
    assert {b.shape_key: b.weight for b in result.mappings["aa"].binds} == {"Custom_A": 0.7, "Custom_B": 0.3}
    assert "blink" not in result.mappings and "blink" in result.missing


def test_config_naming_a_missing_shape_key_fails():
    with pytest.raises(ExpressionConfigError, match="does not exist"):
        resolve_expressions(make_scene(face_keys=["X"]), {"aa": ExpressionOverride(binds={"Nope": 1.0})})


def test_low_confidence_is_reported_not_applied():
    result = resolve(["Smile"], min_confidence=0.85)  # generic 'smile' alias is 0.8
    assert "happy" not in result.mappings
    assert result.below_threshold and any(w.code == "expression-low-confidence" for w in result.warnings)


def test_source_shape_keys_are_untouched():
    scene = make_scene(face_keys=ARKIT_KEYS)
    before = [(k.mesh, k.name) for k in scene.shape_keys]
    resolve_expressions(scene)
    assert before == [(k.mesh, k.name) for k in scene.shape_keys]


def test_norm():
    assert norm("Eye_Blink.L") == "eyeblinkl" == norm("eyeBlinkL")


def test_namespaces_and_side_suffixes_are_normalised():
    from character_vrm.core.expressions import canon, strip_namespace
    assert strip_namespace("Face_Blendshape.Fcl_MTH_A") == "Fcl_MTH_A"
    assert strip_namespace("vrc.v_aa") == "v_aa" and strip_namespace("blendShape1.A") == "A"
    assert strip_namespace("Eye_Blink.L") == "Eye_Blink.L"  # a side marker is not a namespace
    assert canon("Mouth_Smile_L") == canon("mouthSmileLeft") == canon("Body.mouth_smile_left") == "mouthsmileleft"


def test_prefixed_vroid_keys_map_like_bare_ones():
    result = resolve(["Face_Blendshape." + k for k in VROID_KEYS])
    assert result.mappings["aa"].binds[0].shape_key == "Face_Blendshape.Fcl_MTH_A"
    assert result.capability == "full"


def test_vrchat_mmd_and_daz_visemes():
    for keys in (["vrc.v_aa", "vrc.v_ih", "vrc.v_ou", "vrc.v_e", "vrc.v_oh"], ["あ", "い", "う", "え", "お"],
                 ["Genesis8Female__eCTRLvAA", "Genesis8Female__eCTRLvIH", "Genesis8Female__eCTRLvUW",
                  "Genesis8Female__eCTRLvEE", "Genesis8Female__eCTRLvOW"]):
        assert set(resolve(keys).mappings) >= {"aa", "ih", "ou", "ee", "oh"}, keys


def test_character_creator_visemes_carry_the_jaw_morph():
    keys = ["V_Open", "V_Wide", "V_Tight_O", "Jaw_Open", "Eye_Blink_L", "Eye_Blink_R", "Mouth_Smile_L",
            "Mouth_Smile_R", "Brow_Drop_L", "Brow_Drop_R"]
    result = resolve(keys)
    aa = {b.shape_key: b.weight for b in result.mappings["aa"].binds}
    assert set(aa) == {"V_Open", "Jaw_Open"} and result.mappings["aa"].reason == "profile:character-creator"
    assert {b.shape_key for b in result.mappings["ou"].binds} >= {"V_Tight_O"}
    assert {b.shape_key for b in result.mappings["blink"].binds} == {"Eye_Blink_L", "Eye_Blink_R"}
    assert {b.shape_key for b in result.mappings["happy"].binds} >= {"Mouth_Smile_L", "Mouth_Smile_R"}
    assert result.capability == "full"


def test_only_names_that_say_nothing_are_recognised_by_motion():
    from character_vrm.core.anatomy.morph_signatures import opaque_name
    assert opaque_name("Key 17") and opaque_name("Shape.004") and opaque_name("blendShape1.target_12")
    assert not opaque_name("Fcl_MTH_Joy") and not opaque_name("Smile") and not opaque_name("Face_Blendshape.Fcl_EYE_Joy_L")
