from character_vrm.core.config import GeometryConfig
from character_vrm.core.geometry import plan_geometry
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import MeshInfo
from character_vrm.core.errors import ConversionError
from character_vrm.tests.fixtures.scenes import make_scene


def _scene_with_resting_arms():
    scene = make_scene()
    armature = scene.armatures[0]
    humanoid = resolve_humanoid(armature)
    left_arm = humanoid.mappings["leftLowerArm"].source_bone
    right_hand = humanoid.mappings["rightHand"].source_bone

    body = scene.meshes[0]
    body.vertex_positions = [
        (0.00, 0.0, 0.70), (0.02, 0.0, 0.75), (0.00, 0.0, 0.80),  # torso below cut
        (0.30, 0.0, 0.65), (0.40, 0.0, 0.60), (0.50, 0.0, 0.55),  # resting forearm/hand
        (0.08, 0.0, 1.34), (0.18, 0.0, 1.33), (0.25, 0.0, 1.32),  # shoulder transition
    ]
    body.polygons = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    body.vertex_weights = {
        3: {left_arm: 1.0}, 4: {left_arm: 1.0}, 5: {left_arm: 1.0},
        6: {left_arm: 0.06}, 7: {left_arm: 0.06}, 8: {left_arm: 0.06},
    }
    scene.meshes.append(MeshInfo(
        "Legs", 3, 1, armature.name, [humanoid.mappings["leftUpperLeg"].source_bone], ["Skin"],
        vertex_positions=[(0.0, 0.0, 0.2), (0.1, 0.0, 0.1), (0.0, 0.0, 0.0)],
        polygons=[(0, 1, 2)],
        vertex_weights={0: {humanoid.mappings["leftUpperLeg"].source_bone: 1.0}},
    ))
    scene.meshes.append(MeshInfo("Watch", 0, 0, None, [], ["Metal"], parent_bone=right_hand))
    return scene, armature, humanoid


def test_video_call_keeps_resting_complete_arms_and_removes_lower_body():
    scene, armature, humanoid = _scene_with_resting_arms()
    plan = plan_geometry(scene, armature, humanoid, GeometryConfig(profile="video-call"))
    assert plan.semantic_frame == "waist-up"
    assert plan.cut_source == "detected"
    assert 1 in plan.keep_polygons["Body"]  # forearm/hand is below the plane but protected by weights
    assert 2 in plan.keep_polygons["Body"]  # shoulder transition is conservative
    assert "Legs" in plan.delete_objects
    assert "Watch" in plan.protected_objects
    assert set(plan.protected_vertices["Body"]) >= {3, 4, 5, 6, 7, 8}


def test_face_is_bust_but_keeps_complete_arms():
    scene, armature, humanoid = _scene_with_resting_arms()
    plan = plan_geometry(scene, armature, humanoid, GeometryConfig(profile="face"))
    assert plan.semantic_frame == "bust"
    assert 0 in plan.remove_polygons["Body"]
    assert 1 in plan.keep_polygons["Body"]
    assert "Legs" in plan.delete_objects


def test_explicit_drop_cannot_delete_mixed_arm_geometry():
    scene, armature, humanoid = _scene_with_resting_arms()
    try:
        plan_geometry(scene, armature, humanoid, GeometryConfig(profile="face", drop_objects=["Body"]))
    except ConversionError as exc:
        assert exc.code == "unsafe-geometry-drop"
    else:  # pragma: no cover - assertion keeps the invariant obvious
        raise AssertionError("protected arm geometry must not be explicitly droppable")
