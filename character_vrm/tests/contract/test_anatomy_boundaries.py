"""Architecture boundaries of the anatomy / authoring layers (AST checks, no Blender needed)."""

import ast
import dataclasses
import pathlib

import character_vrm
from character_vrm.core.anatomy.model import HumanModelAnalysis

CORE = pathlib.Path(character_vrm.__file__).parent / "core"


def imports(path: pathlib.Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(("." * node.level) + (node.module or ""))
    return out


def test_anatomy_and_authoring_never_import_the_blender_package_or_bpy():
    """(``core/pipeline.py`` names the bridge protocol under TYPE_CHECKING only; that is the existing boundary.)"""
    for path in [*(CORE / "anatomy").glob("*.py"), *(CORE / "authoring").glob("*.py")]:
        for name in imports(path):
            assert "blender" not in name.split("."), (path.name, name)
            assert not name.startswith(("bpy", "mathutils", "bmesh")), (path.name, name)


def test_anatomy_analysis_does_not_depend_on_authoring():
    """The analyzer establishes facts; the authoring layer decides what to do with them (never the reverse)."""
    for path in (CORE / "anatomy").glob("*.py"):
        assert not any("authoring" in n for n in imports(path)), path.name


def test_authoring_plans_do_not_depend_on_blender_adapters():
    for path in (CORE / "authoring").glob("*.py"):
        assert not any(n.endswith("blender") or ".blender" in n for n in imports(path)), path.name


def test_module_layout_matches_the_documented_structure():
    anatomy = {p.stem for p in (CORE / "anatomy").glob("*.py")}
    assert {"model", "frame", "skeleton", "landmarks", "measurements", "hands", "face", "hair", "materials",
            "confidence"} <= anatomy
    authoring = {p.stem for p in (CORE / "authoring").glob("*.py")}
    assert {"rest_pose", "fingers", "expressions", "visemes", "blink", "hair_rig", "springbones", "colliders"} <= authoring
    blender = {p.stem for p in (CORE.parent / "blender").glob("*.py")}
    assert {"anatomy_adapter", "face_geometry", "shape_key_author", "rig_author", "hair_author"} <= blender


def test_no_module_is_a_giant_analyzer():
    for path in list((CORE / "anatomy").glob("*.py")) + list((CORE / "authoring").glob("*.py")):
        assert len(path.read_text().splitlines()) < 500, path.name


def test_human_model_analysis_is_pure_semantic_data():
    names = {f.name for f in dataclasses.fields(HumanModelAnalysis)}
    assert {"frame", "measurements", "landmarks", "skeleton", "left_hand", "right_hand", "face", "hair", "materials",
            "warnings"} <= names
    for f in dataclasses.fields(HumanModelAnalysis):
        assert "bpy" not in str(f.type)


def test_confidence_thresholds_live_in_one_place():
    """Modules take thresholds from ``anatomy.confidence`` (or a ``Thresholds`` value), never as literals."""
    from character_vrm.core.anatomy.confidence import CONFIDENCE
    assert CONFIDENCE["generated_face"] == 0.85 and CONFIDENCE["generated_fingers"] == 0.90 == CONFIDENCE["generated_hair"]
    assert CONFIDENCE["automatic_mapping"] == 0.80
    for name in ("visemes.py", "blink.py", "fingers.py", "hair_rig.py"):
        text = (CORE / "authoring" / name).read_text()
        for literal in ("0.85", "0.90", "0.9 "):
            assert f"< {literal}" not in text and f">= {literal}" not in text, (name, literal)
