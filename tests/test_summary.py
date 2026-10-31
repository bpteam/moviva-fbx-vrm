import copy
import json
from pathlib import Path

from moviva_fbx_to_vrm.summary import BAD, FAILED, LIMITED, OK, READY, WARN, summarize

AMBER = json.loads((Path(__file__).parent / "fixtures" / "amber.report.json").read_text(encoding="utf-8"))


def by_title(summary):
    return {c.title: c.status for c in summary.checks}


def test_real_character_creator_model_is_ready_with_notes():
    s = summarize(AMBER)
    assert s.verdict == READY
    checks = by_title(s)
    assert checks["Body"] == checks["Fingers"] == checks["Lip sync"] == checks["Blinking"] == OK
    assert checks["Weight"] == WARN  # 1.6 M triangles


def test_no_mouth_shapes_still_converts_but_says_lips_stay_still():
    report = copy.deepcopy(AMBER)
    for name in ("aa", "ih", "ou", "ee", "oh"):
        report["expressions"]["mapped"].pop(name)
    s = summarize(report)
    assert s.verdict == LIMITED
    assert by_title(s)["Lip sync"] == BAD


def test_missing_skeleton_bones_fail_with_human_words():
    report = copy.deepcopy(AMBER)
    report["status"] = "failed"
    report["errors"] = [{"code": "humanoid-incomplete", "message": "…",
                         "details": ["missing required: ['leftUpperLeg', 'rightUpperLeg']"]}]
    s = summarize(report)
    assert s.verdict == FAILED
    assert "not recognisable as a human" in s.headline
    assert s.problems == ["missing bones: ['leftUpperLeg', 'rightUpperLeg']"]


def test_crash_without_report_shows_the_last_log_lines():
    s = summarize(None, "a\nb\nSegmentation fault\n")
    assert s.verdict == FAILED
    assert s.problems[-1] == "Segmentation fault"


def test_unknown_error_code_falls_back_to_generic_advice():
    s = summarize({"status": "failed", "errors": [{"code": "something-new", "message": "boom"}]})
    assert s.verdict == FAILED and s.problems == ["boom"] and s.advice


def test_every_language_has_every_string_and_placeholder():
    import json
    import re
    from moviva_fbx_to_vrm.i18n import LANGUAGES, LOCALES
    english = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))
    for lang in LANGUAGES:
        strings = json.loads((LOCALES / f"{lang}.json").read_text(encoding="utf-8"))
        assert strings.keys() == english.keys(), lang
        for key, text in english.items():
            assert set(re.findall(r"{\w+}", strings[key])) == set(re.findall(r"{\w+}", text)), (lang, key)


def test_russian_summary_and_language_detection():
    from moviva_fbx_to_vrm.i18n import language_for, set_language
    assert [language_for(n) for n in ("ru_RU", "zh_TW", "pt_BR", "uk-UA")] == ["ru", "zh", "en", "uk"]
    set_language("ru")
    try:
        assert summarize(AMBER).headline == "Аватар готов."
    finally:
        set_language("en")


def test_generated_rig_is_reported_first():
    report = {**AMBER, "autoRig": {"status": "CHECK", "method": "blender-bone-heat"}}
    s = summarize(report)
    assert s.checks[0].status == WARN and "blender-bone-heat" in s.checks[0].detail
