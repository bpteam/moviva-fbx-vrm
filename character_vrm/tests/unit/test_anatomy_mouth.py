"""Mouth evidence: the lip seam from a step in motion / skinning, consensus fusion, surface roles per material slot,
skin openings as boundary loops, and the jaw rotation the viseme generator builds from the jaw skinning."""

import math

import pytest

from character_vrm.core.anatomy.model import analyze_human
from character_vrm.core.anatomy.mouth import MouthEstimate, fuse_mouth, jaw_seam, step_split
from character_vrm.core.anatomy.surfaces import boundary_loops, vertex_roles
from character_vrm.core.authoring.visemes import JAW_OPEN_DEG, SHAPES, plan_jaw_open, viseme_field
from character_vrm.core.humanoid import resolve_humanoid
from character_vrm.core.model import MeshInfo
from character_vrm.tests.fixtures.heads import head_mesh
from character_vrm.tests.fixtures.rigs import build_armature, mixamo_namer
from character_vrm.tests.fixtures.scenes import make_scene


def test_step_split_finds_the_seam_not_the_chin():
    # lower lip moves 1, chin moves 3, upper lip 0: the seam is the step between lip and lip
    samples = [(y / 100, 3.0) for y in range(0, 10)] + [(y / 100, 1.0) for y in range(10, 20)] + \
              [(y / 100, 0.0) for y in range(20, 40)]
    seam, drop = step_split(samples, window=5)
    assert seam == pytest.approx(0.195, abs=0.011) and drop == pytest.approx(1.0)


def test_consensus_beats_a_single_stronger_outlier():
    c = [MouthEstimate("lip-opening", 0.50, 0.0, 0.05, 0.88), MouthEstimate("jaw-skinning", 0.57, 0.0, 0.028, 0.6),
         MouthEstimate("mouth-interior", 0.572, 0.0, 0.026, 0.72), MouthEstimate("template", 0.56, 0.0, None, 0.35)]
    m = fuse_mouth(c, chin_y=0.53, span=0.11)
    assert m.source == "mouth-interior" and m.seam == pytest.approx(0.572) and m.half_width == pytest.approx(0.028)


def test_roles_come_from_each_material_slot_not_from_any_material_of_the_mesh():
    mesh = MeshInfo("Face", 6, 2, materials=["Face_SKIN", "EyeIris_EYE"], vertex_positions=[(0, 0, 0)] * 6,
                    polygons=[(0, 1, 2), (3, 4, 5)], polygon_materials=[0, 1])
    roles = vertex_roles(mesh)
    assert [roles[i] for i in range(6)] == ["skin"] * 3 + ["eye"] * 3
    assert all(r == "hair" for r in vertex_roles(mesh, {1}).values() if r != "skin")


def test_a_hole_in_a_grid_is_one_small_boundary_loop():
    n = 8
    pos = [(x, y, 0.0) for y in range(n) for x in range(n)]
    polys = [(y * n + x, y * n + x + 1, (y + 1) * n + x + 1, (y + 1) * n + x)
             for y in range(n - 1) for x in range(n - 1) if not (x == 3 and y == 3)]
    loops = boundary_loops(MeshInfo("G", len(pos), 0, vertex_positions=pos), polys, 1e-6)
    assert len(loops) == 2 and sorted(len(l) for l in loops) == [4, 28]


def test_jaw_seam_survives_the_inner_lip_roll():
    # lower lip on the jaw below 0.20, upper lip above; the inner roll puts upper-lip skin just below the seam and a
    # stray jaw vertex just above it, as in Character Creator heads
    samples = [(y / 100, 1.0) for y in range(0, 20)] + [(y / 100, 0.0) for y in range(20, 40)]
    samples += [(0.185, 0.0), (0.19, 0.01), (0.205, 0.97)]
    seam, agreement = jaw_seam(samples)
    assert 0.18 <= seam <= 0.205 and agreement > 0.9  # anywhere inside the roll: the lips meet there


def test_jaw_seam_needs_both_sides():
    assert jaw_seam([(y / 100, 1.0) for y in range(40)]) is None
    assert jaw_seam([(y / 100, 0.0) for y in range(40)] + [(0.0, 1.0)]) is None


def _jaw_scene(curved: bool = False):
    extra = [("Jaw", "mixamorig:Head", (0.0, -0.02, 1.64))]
    arm = build_armature("Armature", mixamo_namer(), extra=extra)
    head = head_mesh()
    # the lower face rides on the jaw below the lip seam (template mouth v 0.19 of the 1.585..1.815 ellipsoid)
    seam = 1.585 + 0.19 * 0.23
    for i, p in enumerate(head.vertex_positions):
        if curved:  # the lip line rises towards the corners and the jaw's share fades there (a smooth rig)
            w = max(0.0, 1.0 - abs(p[0]) / 0.03) if p[2] < seam + 0.4 * abs(p[0]) else 0.0
        else:
            w = 1.0 if p[2] < seam else 0.0
        head.vertex_weights[i] = {"Jaw": w, "mixamorig:Head": 1.0 - w} if w else {"mixamorig:Head": 1.0}
    head.vertex_groups = ["mixamorig:Head", "Jaw"]
    scene = make_scene(armature=arm)
    scene.meshes = [head]
    humanoid = resolve_humanoid(arm, {"jaw": "Jaw"}, 0.75)
    return scene, analyze_human(scene, arm, humanoid), seam


def test_jaw_skinning_places_the_seam_and_the_jaw_opens_by_rotation():
    scene, a, seam = _jaw_scene()
    face = a.face
    assert face.jaw_weighted_lips and face.jaw_pivot is not None
    assert "jaw-skinning" in {c.source for c in face.mouth_candidates}
    assert a.frame.height_of(a.landmarks["mouth"].position) == pytest.approx(seam, abs=0.012)
    fb = viseme_field(a, SHAPES["aa"])
    pos = scene.meshes[0].vertex_positions
    moved = {i: d for (m, i), d in fb.local.items() if math.dist(d, (0, 0, 0)) > 1e-5}
    # below the seam the chin travels down with the jaw; above it only the lip shaping moves the upper lip a little
    chin = min((i for i in moved if pos[i][2] < seam - 0.01), key=lambda i: pos[i][1])  # front of the chin (-Y)
    assert fb.local[("Face", chin)][1] < -0.005
    above = [math.dist(d, (0, 0, 0)) for i, d in moved.items() if pos[i][2] > seam + 0.01]
    assert max(above, default=0.0) < 0.3 * math.dist(fb.local[("Face", chin)], (0, 0, 0))


def test_jaw_skinning_is_read_on_the_midline_when_the_lip_line_curves():
    _, a, seam = _jaw_scene(curved=True)
    assert a.face.jaw_weighted_lips
    assert "jaw-skinning" in {c.source for c in a.face.mouth_candidates}
    assert a.frame.height_of(a.landmarks["mouth"].position) == pytest.approx(seam, abs=0.012)


def test_jaw_open_is_added_only_when_the_existing_aa_opens_nothing():
    _, a, _ = _jaw_scene()
    key, skipped = plan_jaw_open(a, existing_opening=0.0)
    assert key is not None and key.name == "Prosopon_JawOpen" and key.vrm_expression is None
    assert plan_jaw_open(a, existing_opening=0.05) == (None, None)


def test_the_jaw_key_turns_the_jaw_by_the_anatomical_angle():
    scene, a, seam = _jaw_scene()
    key, _ = plan_jaw_open(a, existing_opening=0.0)
    pos = scene.meshes[0].vertex_positions
    pivot = (0.0, -0.02, 1.64)
    chin = min((i for i in key.offsets["Face"] if pos[i][2] < seam - 0.02), key=lambda i: pos[i][1])
    p, d = pos[chin], key.offsets["Face"][chin]
    before = math.atan2(p[2] - pivot[2], p[1] - pivot[1])
    after = math.atan2(p[2] + d[2] - pivot[2], p[1] + d[1] - pivot[1])
    assert abs(math.degrees(after - before)) == pytest.approx(JAW_OPEN_DEG, abs=1.0)

