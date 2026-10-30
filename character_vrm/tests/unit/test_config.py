import json

import pytest

from character_vrm.core.config import ConfigError, load_config, parse_config


def test_empty_config_is_defaults():
    cfg = parse_config({})
    assert cfg.humanoid == {} and cfg.spring.enabled and cfg.min_confidence == 0.75


def test_spec_example_parses():
    cfg = parse_config({
        "humanoid": {"hips": "CC_Base_Hip", "spine": "CC_Base_Waist", "chest": "CC_Base_Spine01", "head": "CC_Base_Head"},
        "expressions": {}, "lookAt": {}, "springBones": {"include": [], "ignore": []},
        "meta": {"name": "Character", "version": "1.0", "authors": ["A"]},
    })
    assert cfg.humanoid["spine"] == "CC_Base_Waist"
    assert (cfg.meta.name, cfg.meta.version, cfg.meta.authors) == ("Character", "1.0", ["A"])


@pytest.mark.parametrize("data,match", [
    ({"humanoid": {"tail": "x"}}, "not a VRM 1.0 humanoid bone"),
    ({"humanoid": {"hips": ""}}, "expected a bone name"),
    ({"bogus": 1}, "unknown key"),
    ({"expressions": {"grin": "x"}}, "not a VRM expression"),
    ({"expressions": {"aa": {"K": 2}}}, "weight"),
    ({"lookAt": {"type": "sideways"}}, "lookAt.type"),
    ({"lookAt": {"offsetFromHeadBone": [1, 2]}}, "offsetFromHeadBone"),
    ({"springBones": {"include": "x"}}, "list of bone names"),
    ({"springBones": {"preset": {"stiff": 1}}}, "unknown key"),
    ({"springBones": {"preset": {"stiffness": -1}}}, "non-negative"),
    ({"meta": {"license": "MIT"}}, "unknown key"),
    ({"meta": {"commercialUsage": "yes"}}, "meta.commercialUsage"),
    ({"meta": {"avatarPermission": "anyone"}}, "meta.avatarPermission"),
])
def test_invalid_configs(data, match):
    with pytest.raises(ConfigError, match=match):
        parse_config(data)


def test_expression_forms():
    cfg = parse_config({"expressions": {"aa": "Mouth_A", "ih": ["I1", "I2"], "ou": {"U": 0.5}, "oh": None}})
    assert cfg.expressions["aa"].binds == {"Mouth_A": 1.0}
    assert cfg.expressions["ih"].binds == {"I1": 1.0, "I2": 1.0}
    assert cfg.expressions["ou"].binds == {"U": 0.5}
    assert cfg.expressions["oh"].disabled


def test_underscore_keys_are_annotations():
    cfg = parse_config({"_comment": "hi", "humanoid": {"_note": "x", "hips": "H"}, "_confidence": {}})
    assert cfg.humanoid == {"hips": "H"}


def test_spring_preset_override():
    cfg = parse_config({"springBones": {"preset": {"stiffness": 2, "hitRadius": 0.05}}})
    assert cfg.spring.preset.stiffness == 2 and cfg.spring.preset.hit_radius == 0.05
    assert cfg.spring.preset.drag_force == 0.5  # untouched default


def test_load_config_reports_bad_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{nope")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)
    path.write_text(json.dumps({"armature": "A"}))
    assert load_config(path).armature == "A"
