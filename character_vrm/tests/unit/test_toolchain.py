import json

import pytest

from character_vrm.blender import compat_v4_7_2
from character_vrm.toolchain import TOOLCHAIN_PATH, EnvironmentFacts, evaluate_environment, load_toolchain

GOOD = EnvironmentFacts("5.2.2", "4.7.2", True, True, True, True)


def test_toolchain_json_is_pinned():
    assert json.loads(TOOLCHAIN_PATH.read_text()) == {"blender": "5.2.2", "vrmAddon": "4.7.2", "vrm": "1.0"}
    tc = load_toolchain()
    assert (tc.blender, tc.vrm_addon, tc.target) == ("5.2.2", "4.7.2", "VRM 1.0")


def test_adapter_supports_exactly_the_pinned_addon():
    assert compat_v4_7_2.SUPPORTED_ADDON_VERSION == load_toolchain().vrm_addon
    assert compat_v4_7_2.SPEC_VERSION == load_toolchain().vrm


def test_doctor_ok_and_format():
    report = evaluate_environment(GOOD)
    assert report.ok
    text = report.format()
    for line in ("Moviva Character Converter", "Blender", "  expected: 5.2.2", "  detected: 5.2.2",
                 "VRM Add-on", "  expected: 4.7.2", "VRM exporter", "  bpy.ops.export_scene.vrm",
                 "Target", "  VRM 1.0", "Environment OK"):
        assert line in text


@pytest.mark.parametrize("blender", ["5.2.1", "5.2.3", "4.5.0", "5.3.0", None])
def test_any_other_blender_fails(blender):
    facts = EnvironmentFacts(blender, "4.7.2", True, True, True, True)
    report = evaluate_environment(facts)
    assert not report.ok
    assert "Environment NOT OK" in report.format()


@pytest.mark.parametrize("addon", ["4.7.1", "4.8.0", "4.9.0", None])
def test_any_other_addon_fails(addon):
    facts = EnvironmentFacts("5.2.2", addon, True, True, True, True)
    assert not evaluate_environment(facts).ok


def test_missing_export_operator_fails():
    facts = EnvironmentFacts("5.2.2", "4.7.2", False, True, True, True)
    report = evaluate_environment(facts)
    assert not report.ok
    assert next(c for c in report.checks if c.title == "VRM exporter").ok is False


def test_missing_importer_operators_fail():
    for field in ("gltf_importer", "gltf_exporter", "fbx_importer"):
        facts = EnvironmentFacts("5.2.2", "4.7.2", True, True, True, True)
        setattr(facts, field, False)
        assert not evaluate_environment(facts).ok, field


def test_addon_error_is_shown():
    facts = EnvironmentFacts("5.2.2", None, True, True, True, True, addon_error="VRM Add-on is not installed or not enabled")
    assert "not installed or not enabled" in evaluate_environment(facts).format()
