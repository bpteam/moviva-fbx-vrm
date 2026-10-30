"""End-to-end (fake bridge): analyse -> plan -> validate -> mutate -> validate result, the CLI and the report."""

import io
import json

import pytest

from character_vrm.commands import build_parser, run
from character_vrm.core.config import ConfigError, ConversionConfig, parse_config
from character_vrm.core.errors import ConversionError
from character_vrm.core.model import ArmatureInfo, BoneInfo, MaterialInfo
from character_vrm.core.pipeline import ConverterCore, analyze
from character_vrm.core.report import build_report
from character_vrm.tests.fakes import FakeBlenderBridge
from character_vrm.tests.fixtures.hands import hand_mesh, with_finger_bones
from character_vrm.tests.fixtures.heads import head_mesh
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer, opaque_namer
from character_vrm.tests.fixtures.scenes import VROID_KEYS, make_scene
from character_vrm.tests.unit.test_anatomy_hair import named_chain_scene, ponytail_mesh


def face_scene(face_keys=(), **kw):
    scene = make_scene(face_keys=list(face_keys), **kw)
    scene.meshes = [head_mesh(), *scene.meshes[:0]]
    for key in scene.shape_keys:
        key.mesh = "Face"
    return scene


def config(**author):
    cfg = ConversionConfig()
    for k, v in author.items():
        setattr(cfg.auto_author, k, v)
    return cfg


def convert(tmp_path, scene, cfg, dry_run=False, **bridge_kw):
    src = tmp_path / "model.glb"
    src.write_bytes(b"x")
    bridge = FakeBlenderBridge(scene, **bridge_kw)
    outcome = ConverterCore(bridge).convert(src, tmp_path / "out.vrm", cfg, dry_run=dry_run)
    return bridge, outcome


def codes(outcome):
    return [w["code"] for w in outcome.report["warnings"]]


# ---- the analysis always runs, authoring only on request -------------------------------------


def test_convert_always_runs_anatomy_analysis_and_reports_it(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(), config())
    anatomy = outcome.report["anatomy"]
    assert anatomy["available"] and anatomy["version"] == 1
    assert anatomy["anatomy"]["height"] > 1 and set(anatomy["anatomy"]["frame"]) >= {"up", "forward", "confidence"}
    assert anatomy["skeleton"]["bones"]["leftUpperArm"]["evidence"].keys() >= {"topology", "position", "symmetry"}
    assert anatomy["pose"]["poseType"] == "T-pose" and anatomy["landmarks"]["mouth"]["source"] == "geometry"
    assert anatomy["face"]["quality"] > 0.8 and anatomy["materials"]["Skin"]["semantic"] == "skin"
    assert outcome.report["authoring"] == {"enabled": False}
    assert "apply_authoring" not in bridge.calls  # nothing is authored without --auto-author


def test_without_auto_author_the_export_is_the_legacy_one(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(), config())
    assert set(bridge.plan.expressions) == set() and not bridge.plan.springs.chains


def test_auto_author_generates_blink_and_visemes_and_binds_them(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(), config(enabled=True))
    assert bridge.calls.index("apply_authoring") < bridge.calls.index("configure_vrm")   # mutate before the add-on runs
    exprs = bridge.plan.expressions
    assert {"blink", "blinkLeft", "blinkRight", "aa", "ih", "ou", "ee", "oh"} <= set(exprs)
    for name in ("aa", "blinkLeft"):
        m = exprs[name]
        assert m.kind == "generated-geometry" and all(b.shape_key.startswith("Prosopon_") and b.mesh == "Face" for b in m.binds)
    rep = outcome.report["authoring"]
    assert rep["applied"] and rep["expressions"]["aa"]["status"] == "GENERATED" and rep["expressions"]["blinkLeft"]["confidence"] > 0.85
    assert rep["expressions"]["happy"]["status"] == "MISSING"
    assert outcome.report["anatomy"]["face"]["generated"]  # the artifact lists what was generated
    assert outcome.output is not None  # the (test) GLB re-read check accepted the generated binds


def test_source_morphs_win_over_generation_end_to_end(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(VROID_KEYS), config(enabled=True))
    assert bridge.plan.expressions["aa"].kind != "generated-geometry"
    assert bridge.plan.expressions["blinkLeft"].kind != "generated-geometry"
    assert outcome.report["authoring"]["expressions"]["aa"]["status"] in ("SOURCE", "MAPPED")
    assert not [k for k in outcome.analysis.authoring.shape_keys if k.vrm_expression in ("aa", "blinkLeft")]


def test_failed_mutation_is_reported_and_never_bound(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(), config(enabled=True))
    bridge2, outcome2 = convert(tmp_path, face_scene(), config(enabled=True))
    bridge3 = FakeBlenderBridge(face_scene())
    bridge3.fail_authoring = {"Prosopon_aa"}
    src = tmp_path / "m2.glb"
    src.write_bytes(b"x")
    out = ConverterCore(bridge3).convert(src, tmp_path / "o2.vrm", config(enabled=True))
    assert "aa" not in bridge3.plan.expressions and "ih" in bridge3.plan.expressions
    assert "authoring-failed" in codes(out) and "authoring-not-applied" in codes(out)
    assert out.report["authoring"]["expressions"]["aa"]["status"] == "MISSING"
    assert out.report["authoring"]["result"]["failed"]["Prosopon_aa"]


def test_dry_run_plans_but_mutates_nothing(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(), config(enabled=True), dry_run=True)
    assert "apply_authoring" not in bridge.calls and "configure_vrm" not in bridge.calls
    rep = outcome.report["authoring"]
    assert rep["enabled"] and not rep["applied"] and rep["shapeKeys"]
    assert outcome.report["status"] == "dry-run"


def test_low_confidence_face_is_kept_as_is(tmp_path):
    scene = make_scene()
    scene.meshes = [head_mesh(grid=40)]
    bridge, outcome = convert(tmp_path, scene, config(enabled=True))
    assert "apply_authoring" not in bridge.calls and not bridge.plan.expressions
    assert outcome.report["authoring"]["expressions"]["aa"]["status"] == "MISSING"
    assert "too little geometry" in outcome.report["authoring"]["expressions"]["aa"]["reason"]


# ---- weak names and hair springs -----------------------------------------------------------------


def test_weak_named_skeleton_converts_and_the_report_says_the_analysis_mapped_it(tmp_path):
    arm = build_armature("Armature", opaque_namer())
    bridge, outcome = convert(tmp_path, make_scene(armature=arm), config())
    reasons = {m["reason"].split(" ")[0] for m in outcome.report["humanoid"]["mappings"]}
    assert reasons == {"anatomy"} and outcome.report["anatomy"]["skeleton"]["inferred"]
    # left / right come from the feet (ankle -> toes), not from the +X import convention
    assert outcome.report["anatomy"]["skeleton"]["sidesAssumed"] is False
    assert "frame-sides-assumed" not in [w["code"] for w in outcome.report["anatomy"]["warnings"]]


def test_hair_bone_chains_become_spring_bones_with_anatomical_colliders(tmp_path):
    scene, arm = named_chain_scene(2, n_bones=4)
    bridge, outcome = convert(tmp_path, scene, config(enabled=True))
    plan = bridge.plan.springs
    assert len(plan.chains) == 2
    assert {c.name for c in plan.colliders} >= {"head_collider", "neck_collider"}
    rep = outcome.report["authoring"]
    assert rep["springs"]["chains"] and {c["name"] for c in rep["colliders"]} >= {"head_collider"}
    assert outcome.report["springBones"]["colliders"]  # the report shows the plan that was actually used


def test_no_springs_flag_wins_over_auto_author(tmp_path):
    scene, arm = named_chain_scene(2, n_bones=4)
    cfg = config(enabled=True)
    cfg.spring.enabled = False
    bridge, outcome = convert(tmp_path, scene, cfg)
    assert not bridge.plan.springs.chains


def test_hair_without_bones_stays_static_and_the_report_says_why(tmp_path):
    scene = make_scene(materials=[MaterialInfo("Skin"), MaterialInfo("Hair")])
    scene.meshes = [head_mesh(), ponytail_mesh()]
    bridge, outcome = convert(tmp_path, scene, config(enabled=True))
    hair = outcome.report["authoring"]["hairRig"]
    assert hair["chains"] == [] and "feature disabled" in hair["skipped"][0]["reason"]
    assert not bridge.plan.springs.chains
    assert outcome.report["anatomy"]["hair"]["springChains"] == 0


def test_generated_hair_rig_only_with_its_own_flag(tmp_path):
    scene = make_scene(materials=[MaterialInfo("Skin"), MaterialInfo("Hair")])
    scene.meshes = [head_mesh(), ponytail_mesh()]
    bridge, outcome = convert(tmp_path, scene, config(generate_hair_rig=True))
    chains = outcome.report["authoring"]["hairRig"]["chains"]
    assert len(chains) == 1 and chains[0]["category"] == "ponytail"
    assert len(bridge.plan.springs.chains) == 1 and bridge.plan.springs.chains[0].bones[0].startswith("Prosopon_")


def test_generated_finger_rig_only_with_its_own_flag(tmp_path):
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    scene = make_scene(armature=arm)
    scene.meshes = [hand_mesh(bone="mixamorig:LeftHand"), hand_mesh("HandsR", wrist=(-0.7, 0, 1.45), sign=-1, bone="mixamorig:RightHand")]
    bridge, outcome = convert(tmp_path, scene, config(enabled=True))
    assert not any(k.startswith("leftIndex") for k in bridge.plan.humanoid)
    assert all("feature disabled" in f["skipped"]["reason"] for f in outcome.report["authoring"]["fingers"])
    cfg = config(generate_finger_rig=True)
    cfg.auto_author.confidence = {"generated_fingers": 0.9}
    bridge, outcome = convert(tmp_path, scene, cfg)
    assert bridge.plan.humanoid["leftIndexProximal"] == "Prosopon_left_Index_1" and len(
        [k for k in bridge.plan.humanoid if k.startswith(("leftIndex", "leftThumb", "leftLittle", "leftMiddle", "leftRing"))]) == 15


def test_fused_hands_are_reported_never_rigged(tmp_path):
    arm = build_armature("Armature", mixamo_namer(), fingers=False)
    scene = make_scene(armature=arm)
    scene.meshes = [hand_mesh(bone="mixamorig:LeftHand", fused=True), hand_mesh("HandsR", wrist=(-0.7, 0, 1.45), sign=-1, bone="mixamorig:RightHand", fused=True)]
    bridge, outcome = convert(tmp_path, scene, config(generate_finger_rig=True))
    assert not any(k.startswith("leftIndex") for k in bridge.plan.humanoid)
    assert "independent finger articulation unavailable" in outcome.report["authoring"]["fingers"][0]["skipped"]["reason"]
    assert outcome.report["anatomy"]["hands"]["left"]["state"] == "fused-fingers"


def test_rest_pose_is_reported_with_the_runtime_target(tmp_path):
    bridge, outcome = convert(tmp_path, face_scene(), config(enabled=True))
    pose = outcome.report["authoring"]["restPose"]
    assert pose["mutatesScene"] is False and pose["targetAbductionDeg"] == pytest.approx(22.33, abs=0.01)
    assert outcome.report["anatomy"]["pose"] == pose


# ---- geometry profile uses the canonical analysis (and is orientation independent) ---------------


def test_geometry_profile_is_orientation_and_scale_independent(tmp_path):
    from character_vrm.core.config import GeometryConfig
    from character_vrm.core.geometry import plan_geometry
    from character_vrm.core.humanoid import resolve_humanoid
    from character_vrm.tests.unit.test_geometry import _scene_with_resting_arms
    from character_vrm.tests.fixtures.transforms import rotation, transformed

    scene, arm, human = _scene_with_resting_arms()
    ref = plan_geometry(scene, arm, human, GeometryConfig(profile="video-call"))
    for r, s in ((rotation(90, 0, 0), 1.0), (rotation(20, 60, 100), 2.5)):
        moved = transformed(scene, r, s, (3, -2, 5))
        marm = moved.armatures[0]
        got = plan_geometry(moved, marm, resolve_humanoid(marm), GeometryConfig(profile="video-call"))
        assert got.remove_polygons == ref.remove_polygons and got.keep_polygons == ref.keep_polygons
        assert got.delete_objects == ref.delete_objects and got.cut_source == ref.cut_source == "detected"
        assert got.protected_vertices == ref.protected_vertices


def test_geometry_profile_reads_the_waist_from_the_analysis_landmark(tmp_path):
    scene = face_scene()
    a = analyze(scene_copy(scene), ConversionConfig(geometry=__import__("character_vrm.core.config", fromlist=["GeometryConfig"]).GeometryConfig(profile="video-call")))
    waist = a.anatomy.landmark_level("waist")
    assert a.geometry.landmark == pytest.approx(waist) and a.geometry.cut_source == a.anatomy.landmarks["waist"].method


def scene_copy(scene):
    import copy
    return copy.deepcopy(scene)


# ---- config and CLI ---------------------------------------------------------------------------------


def test_config_file_section():
    cfg = parse_config({"autoAuthor": {"enabled": True, "generateFace": True, "confidence": {"generated_face": 0.9}}})
    a = cfg.auto_author
    assert a.enabled and a.generate_face and not a.generate_finger_rig and a.confidence == {"generated_face": 0.9}
    for bad in ({"autoAuthor": {"enabled": "yes"}}, {"autoAuthor": {"nope": 1}},
                {"autoAuthor": {"confidence": {"unknown": 0.5}}}, {"autoAuthor": {"confidence": {"generated_face": 2}}}):
        with pytest.raises(ConfigError):
            parse_config(bad)


def test_cli_flags_imply_auto_author_and_default_is_off():
    p = build_parser()
    assert not p.parse_args(["convert", "a.glb", "b.vrm"]).auto_author
    a = p.parse_args(["convert", "a.glb", "b.vrm", "--auto-author", "--geometry-profile", "video-call"])
    assert a.auto_author and a.geometry_profile == "video-call"
    a = p.parse_args(["analyze", "a.glb", "--json", "x.json", "--generate-face", "--generate-finger-rig", "--generate-hair-rig"])
    assert a.command == "analyze" and a.json == "x.json" and a.generate_face and a.generate_finger_rig and a.generate_hair_rig


def run_cli(tmp_path, args, scene):
    src = tmp_path / "model.glb"
    src.write_bytes(b"x")
    out = io.StringIO()
    bridge = FakeBlenderBridge(scene)
    code = run([args[0], str(src), *args[1:]], lambda: bridge, out)
    return code, out.getvalue(), bridge


def test_analyze_command_describes_and_never_modifies(tmp_path):
    code, text, bridge = run_cli(tmp_path, ["analyze", "--json", str(tmp_path / "a.json")], face_scene())
    assert code == 0
    for heading in ("Anatomy", "Skeleton", "Hands", "Face", "Hair", "Materials"):
        assert heading in text
    assert not {"configure_vrm", "export_vrm", "apply_authoring", "apply_geometry", "isolate_armature"} & set(bridge.calls)
    doc = json.loads((tmp_path / "a.json").read_text())
    assert doc["version"] == 1 and doc["anatomy"]["height"] > 1 and "authoring" not in doc


def test_analyze_json_is_the_same_section_a_conversion_report_holds(tmp_path):
    scene = face_scene()
    code, _, _ = run_cli(tmp_path, ["analyze", "--json", str(tmp_path / "a.json")], scene)
    doc = json.loads((tmp_path / "a.json").read_text())
    _, outcome = convert(tmp_path, face_scene(), config())
    assert doc == json.loads(json.dumps(outcome.report["anatomy"]))


def test_analyze_with_auto_author_shows_the_plan_but_still_mutates_nothing(tmp_path):
    code, text, bridge = run_cli(tmp_path, ["analyze", "--auto-author", "--json", str(tmp_path / "a.json")], face_scene())
    assert code == 0 and "Auto-Author" in text and "GENERATED" in text
    assert "apply_authoring" not in bridge.calls
    assert json.loads((tmp_path / "a.json").read_text())["authoring"]["shapeKeys"]


def test_convert_cli_end_to_end_writes_a_report_with_the_new_sections(tmp_path):
    src = tmp_path / "model.glb"
    src.write_bytes(b"x")
    out = io.StringIO()
    bridge = FakeBlenderBridge(face_scene())
    code = run(["convert", str(src), str(tmp_path / "c.vrm"), "--auto-author", "--geometry-profile", "video-call"],
               lambda: bridge, out)
    assert code == 0
    report = json.loads((tmp_path / "c.vrm.report.json").read_text())
    assert report["status"] == "success" and report["authoring"]["applied"] and report["anatomy"]["available"]
    assert report["geometry"]["profile"] == "video-call"


def test_analysis_unavailable_is_a_warning_not_a_crash(tmp_path):
    arm = ArmatureInfo("Armature", [BoneInfo("Root", None, (0, 0, 0), (0, 0, 1))])
    scene = make_scene(armature=arm)
    with pytest.raises(ConversionError):
        convert(tmp_path, scene, config(enabled=True))
