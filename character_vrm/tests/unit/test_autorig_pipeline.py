"""ConverterCore + auto-rig, against the fake bridge: when it runs, when it must not, how failure stops."""

import copy

import pytest

from character_vrm.core.autorig import AutoRigResult
from character_vrm.core.config import ConversionConfig
from character_vrm.core.errors import ConversionError
from character_vrm.core.pipeline import ConverterCore
from character_vrm.tests.fakes import FakeBlenderBridge
from character_vrm.tests.fixtures.scenes import make_scene


def run(tmp_path, bare, rigged, cfg, result=None):
    src = tmp_path / "m.glb"
    src.write_bytes(b"x")
    bridge = FakeBlenderBridge(bare)
    bridge.rigged, bridge.rig_result = rigged, result
    return bridge, ConverterCore(bridge).convert(src, tmp_path / "m.vrm", cfg)


def bare_of(scene):
    bare = copy.deepcopy(scene)
    bare.armatures = []
    return bare


def test_unrigged_source_is_rigged_then_converted(tmp_path):
    rigged = make_scene()
    bridge, outcome = run(tmp_path, bare_of(rigged), rigged, ConversionConfig(auto_rig=True),
                          AutoRigResult("CHECK", method="blender-bone-heat", confidence=0.8))
    assert "auto_rig" in bridge.calls and bridge.calls.index("auto_rig") < bridge.calls.index("export_vrm")
    assert outcome.report["autoRig"]["status"] == "CHECK"
    assert any(w["code"] == "autorig" for w in outcome.report["warnings"])


def test_auto_rig_off_keeps_the_old_refusal(tmp_path):
    with pytest.raises(ConversionError) as exc:
        run(tmp_path, bare_of(make_scene()), None, ConversionConfig())
    assert exc.value.code == "no-armature"


def test_existing_armature_is_never_replaced(tmp_path):
    bridge, _ = run(tmp_path, make_scene(), None, ConversionConfig(auto_rig=True))
    assert "auto_rig" not in bridge.calls


def test_failed_rig_stops_before_export(tmp_path):
    with pytest.raises(ConversionError) as exc:
        run(tmp_path, bare_of(make_scene()), make_scene(), ConversionConfig(auto_rig=True), AutoRigResult("FAIL"))
    assert exc.value.code == "autorig-failed"
