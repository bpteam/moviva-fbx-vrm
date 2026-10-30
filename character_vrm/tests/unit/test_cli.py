import io
import json
import os
import stat
from pathlib import Path

import pytest

from character_vrm import cli, launcher
from character_vrm.commands import build_parser, run
from character_vrm.tests.fakes import FakeBlenderBridge
from character_vrm.tests.fixtures.scenes import ARKIT_KEYS, make_scene
from character_vrm.toolchain import EnvironmentFacts


def fake_exe(tmp_path, name="blender") -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# --- argument parsing --------------------------------------------------------------


def test_parser_commands():
    p = build_parser()
    assert p.parse_args(["doctor"]).command == "doctor"
    a = p.parse_args(["convert", "a.fbx", "b.vrm", "--config", "m.json", "--dry-run", "--no-springs",
                      "--armature", "Arm", "--min-confidence", "0.6", "--blender", "/x/blender"])
    assert (a.source, a.output, a.config, a.dry_run, a.no_springs, a.armature, a.min_confidence, a.blender) == (
        "a.fbx", "b.vrm", "m.json", True, True, "Arm", 0.6, "/x/blender")
    assert p.parse_args(["inspect", "a.glb"]).command == "inspect"
    assert p.parse_args(["map", "a.glb", "--output", "x.json"]).output == "x.json"


def test_blender_option_accepted_before_and_after_the_subcommand():
    p = build_parser()
    assert p.parse_args(["--blender", "/b", "doctor"]).blender == "/b"
    assert p.parse_args(["doctor", "--blender", "/b"]).blender == "/b"
    assert p.parse_args(["--blender", "/b", "inspect", "a.glb"]).blender == "/b"


def test_parser_rejects_bad_usage():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["convert", "only-one.glb"])
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_blender_option_is_not_forwarded():
    assert cli.strip_blender_option(["--blender", "/b", "convert", "a", "b"]) == ["convert", "a", "b"]
    assert cli.strip_blender_option(["convert", "a", "b", "--blender=/b", "--dry-run"]) == ["convert", "a", "b", "--dry-run"]
    assert build_parser(with_blender=False).parse_args(cli.strip_blender_option(["--blender", "/b", "doctor"])).command == "doctor"


# --- blender resolution: --blender > BLENDER_BIN > PATH -------------------------------


def test_resolution_priority(tmp_path):
    flag, env_bin, path_bin = (fake_exe(tmp_path, n) for n in ("flag", "env", "path"))
    which = lambda name: path_bin  # noqa: E731
    assert launcher.resolve_blender(flag, {"BLENDER_BIN": env_bin}, which) == flag
    assert launcher.resolve_blender(None, {"BLENDER_BIN": env_bin}, which) == env_bin
    assert launcher.resolve_blender(None, {}, which) == path_bin


def test_resolution_errors(tmp_path):
    with pytest.raises(launcher.LauncherError, match="not found"):
        launcher.resolve_blender(None, {}, lambda n: None)
    with pytest.raises(launcher.LauncherError, match="--blender"):
        launcher.resolve_blender(str(tmp_path / "missing"), {}, lambda n: None)
    with pytest.raises(launcher.LauncherError, match="BLENDER_BIN"):
        launcher.resolve_blender(None, {"BLENDER_BIN": str(tmp_path / "missing")}, lambda n: None)
    not_exec = tmp_path / "plain"
    not_exec.write_text("x")
    with pytest.raises(launcher.LauncherError):
        launcher.resolve_blender(str(not_exec), {}, lambda n: None)


def test_explicit_flag_never_falls_back_to_env(tmp_path):
    env_bin = fake_exe(tmp_path, "env")
    with pytest.raises(launcher.LauncherError):
        launcher.resolve_blender(str(tmp_path / "bad"), {"BLENDER_BIN": env_bin}, lambda n: env_bin)


def test_spawn_command_shape():
    cmd = launcher.build_command("/opt/blender", ["convert", "a.glb", "b.vrm"])
    assert cmd[:6] == ["/opt/blender", "--background", "--python-exit-code", "1", "--python", str(launcher.ENTRY_SCRIPT)]
    assert cmd[6:] == ["--", "convert", "a.glb", "b.vrm"]
    assert launcher.ENTRY_SCRIPT.name == "blender_entry.py" and launcher.ENTRY_SCRIPT.is_file()


def test_main_spawns_blender_and_forwards_exit_code(tmp_path, monkeypatch):
    blender = fake_exe(tmp_path)
    source = tmp_path / "c.glb"
    source.write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(cli, "run_blender", lambda b, argv: seen.update(b=b, argv=argv) or 7)
    code = cli.main(["--blender", blender, "convert", str(source), str(tmp_path / "o.vrm"), "--dry-run"])
    assert code == 7
    assert seen["b"] == blender and seen["argv"] == ["convert", str(source), str(tmp_path / "o.vrm"), "--dry-run"]


def test_main_fails_fast_before_starting_blender(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_blender", lambda *a: pytest.fail("Blender must not start"))
    assert cli.main(["convert", str(tmp_path / "missing.glb"), "o.vrm"]) == 1
    (tmp_path / "m.obj").write_bytes(b"x")
    assert cli.main(["convert", str(tmp_path / "m.obj"), "o.vrm"]) == 2
    (tmp_path / "m.glb").write_bytes(b"x")
    assert cli.main(["convert", str(tmp_path / "m.glb"), "o.vrm", "--config", str(tmp_path / "nope.json")]) == 1


# --- command layer against the fake bridge -------------------------------------------


def run_cmd(argv, bridge):
    out = io.StringIO()
    code = run(argv, lambda: bridge, out)
    return code, out.getvalue()


def test_doctor_command():
    code, text = run_cmd(["doctor"], FakeBlenderBridge())
    assert code == 0 and "Environment OK" in text
    bad = FakeBlenderBridge(facts=EnvironmentFacts("5.1.0", "4.7.2", True, True, True, True))
    code, text = run_cmd(["doctor"], bad)
    assert code == 1 and "Environment NOT OK" in text


def test_map_command_writes_suggestion_beside_the_model(tmp_path):
    model = tmp_path / "character.glb"
    model.write_bytes(b"x")
    code, text = run_cmd(["map", str(model)], FakeBlenderBridge(make_scene(face_keys=ARKIT_KEYS)))
    assert code == 0
    written = tmp_path / "character.vrm-map.json"
    data = json.loads(written.read_text())
    assert data["humanoid"]["hips"] == "mixamorig:Hips" and data["armature"] == "Armature"
    assert data["expressions"]["aa"] == {"jawOpen": 0.8}
    # the suggestion is itself a valid map file
    from character_vrm.core.config import load_config
    assert load_config(written).humanoid["hips"] == "mixamorig:Hips"


def test_map_does_not_convert():
    bridge = FakeBlenderBridge(make_scene())
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "m.glb"
        model.write_bytes(b"x")
        run_cmd(["map", str(model)], bridge)
    assert "export_vrm" not in bridge.calls and "configure_vrm" not in bridge.calls


def test_inspect_command_exit_code_reflects_convertibility(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    code, text = run_cmd(["inspect", str(model)], FakeBlenderBridge(make_scene()))
    assert code == 0 and "Character inspection" in text
    from character_vrm.tests.fixtures.rigs import build_armature, generic_namer
    broken = make_scene(armature=build_armature("Armature", generic_namer(), drop={"head", "headtop", "eye.L", "eye.R"}))
    code, text = run_cmd(["inspect", str(model)], FakeBlenderBridge(broken))
    assert code == 1 and "NOT convertible" in text


def test_convert_command_writes_output_and_report_side_by_side(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    out = tmp_path / "sub" / "m.vrm"
    out.parent.mkdir()
    code, text = run_cmd(["convert", str(model), str(out)], FakeBlenderBridge(make_scene()))
    assert code == 0 and out.exists()
    report = json.loads((tmp_path / "sub" / "m.vrm.report.json").read_text())
    assert report["status"] == "success" and report["toolchain"]["vrmAddon"] == "4.7.2"


def test_convert_dry_run_writes_no_files(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    out = tmp_path / "m.vrm"
    code, text = run_cmd(["convert", str(model), str(out), "--dry-run"], FakeBlenderBridge(make_scene()))
    assert code == 0 and not out.exists() and not (tmp_path / "m.vrm.report.json").exists()
    assert "Dry run" in text


def test_convert_failure_exit_code_and_failed_report(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    out = tmp_path / "m.vrm"
    code, text = run_cmd(["convert", str(model), str(out)], FakeBlenderBridge(make_scene(), fail_export=True))
    assert code == 1 and "ERROR" in text and "export-failed" in text
    report = json.loads((tmp_path / "m.vrm.report.json").read_text())
    assert report["status"] == "failed" and report["errors"][0]["code"] == "export-failed"


def test_fbx_warning_is_printed_prominently(tmp_path):
    model = tmp_path / "m.fbx"
    model.write_bytes(b"x")
    code, text = run_cmd(["convert", str(model), str(tmp_path / "m.vrm")], FakeBlenderBridge(make_scene(fmt="fbx")))
    assert code == 0
    assert "WARNING" in text and "no facial shape keys" in text and "Prefer a GLB/glTF source containing morph targets" in text


def test_cli_flags_override_the_map_file(tmp_path):
    model = tmp_path / "m.glb"
    model.write_bytes(b"x")
    cfg = tmp_path / "m.vrm-map.json"
    cfg.write_text(json.dumps({"armature": "Nope"}))
    code, text = run_cmd(["convert", str(model), str(tmp_path / "o.vrm"), "--config", str(cfg), "--armature", "Armature"],
                         FakeBlenderBridge(make_scene()))
    assert code == 0
    code, text = run_cmd(["convert", str(model), str(tmp_path / "o.vrm"), "--min-confidence", "3"], FakeBlenderBridge(make_scene()))
    assert code == 1 and "min-confidence" in text
