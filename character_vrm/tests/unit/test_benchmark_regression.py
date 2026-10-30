"""Ground-truth regression: the anatomy analyzer, expression resolver and Auto-Author scored against authored VRMs
(``character_vrm.benchmark``). The bundled sample avatar always runs; other models run when present locally
(``PROSOPON_BENCHMARK_MODELS``: paths separated by ``:``). Errors are in interocular distances (IOD)."""

import os
from pathlib import Path

import pytest

from character_vrm.benchmark import run_model

AVATAR = Path(__file__).resolve().parents[4] / "avatar" / "public" / "models" / "mia.vrm"
EXTRA = [p for p in os.environ.get("PROSOPON_BENCHMARK_MODELS", "").split(":") if p and Path(p).is_file()]
MODELS = ([str(AVATAR)] if AVATAR.is_file() else []) + EXTRA


@pytest.fixture(scope="module", params=MODELS or [None])
def result(request):
    if request.param is None:
        pytest.skip("no benchmark model available")
    return run_model(request.param)


def test_face_landmarks_are_on_the_authored_face(result):
    for name in ("named", "strip"):
        lm = result[name]["landmarks"]
        assert lm["mouth"] is not None and lm["mouth"] <= 0.15, (name, lm)
        assert max(lm["mouth_left"], lm["mouth_right"]) <= 0.2, (name, lm)
        assert max(lm["left_eyelid"], lm["right_eyelid"]) <= 0.15, (name, lm)
        assert 0.8 <= lm["mouth_width_ratio"] <= 1.25, (name, lm)
    # without names the geometry still finds the face
    lm = result["blind"]["landmarks"]
    assert lm["mouth"] <= 0.25 and max(lm["left_eyelid"], lm["right_eyelid"]) <= 0.25, lm


def test_skeleton_is_mapped_with_and_without_names(result):
    for name in ("named", "blind"):
        h = result[name]["humanoid"]
        assert not h["wrong"], (name, h["wrong"])
        assert h["accuracy"] >= 0.98, (name, h)


def test_expressions_resolve_by_name_and_blinks_by_motion(result):
    assert all(v == "ok" for v in result["named"]["expressions"].values()), result["named"]["expressions"]
    blind = result["blind"]["expressions"]
    assert all(v == "ok" or v == "missing" or k == "blink" for k, v in blind.items()), blind  # never a wrong key
    assert blind.get("blinkLeft") == "ok" and blind.get("blinkRight") == "ok", blind


def test_hair_is_never_read_as_face_skin_when_named(result):
    hair = result["named"]["hair"]
    if hair is not None:
        assert hair["hairCountedAsFace"] == 0 and hair["recall"] >= 0.95, hair


def test_generated_face_keys_land_where_the_artist_put_them(result):
    gen = result["strip"]["generation"]
    for expr, g in gen.items():
        if not g.get("generated"):
            continue
        assert g["valid"], (expr, g)
        assert g["hairLeak"] == 0.0, (expr, g)
        assert g["otherRegionLeak"] <= 0.05, (expr, g)
    if "mia.vrm" in result["model"]:
        # named keys that are something else (an eye smile, an open-mouth joy) must not stand in for the removed
        # visemes / blinks: the generator has to make them
        for expr in ("aa", "ih", "ou", "ee", "oh", "blinkLeft", "blinkRight"):
            assert gen[expr]["generated"], (expr, gen[expr])
        for expr in ("ih", "ou", "ee", "oh"):
            assert gen[expr]["iou"] >= 0.5, (expr, gen[expr])
