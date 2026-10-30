"""Turns a ``character_vrm`` conversion report (``<output>.report.json``) into plain words for the window.

The report is the converter's own contract; this module only reads it. Every line answers "what will my avatar do"
and, when something is missing, "what can I do about it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .i18n import t

OK, WARN, BAD = "ok", "warn", "bad"
READY, LIMITED, FAILED = "ready", "limited", "failed"

VISEMES = ("aa", "ih", "ou", "ee", "oh")
MOODS = ("happy", "angry", "sad", "relaxed", "surprised")
HEAVY_TRIANGLES = 150_000
HEAVY_BYTES = 150 * 1024 * 1024

KNOWN_ERRORS = ("no-armature", "no-skinned-mesh", "humanoid-incomplete", "humanoid-invalid", "ambiguous-armature",
                "armature-not-found", "unsupported-format", "source-missing", "import-failed", "importer-missing",
                "export-failed", "invalid-output", "unsafe-geometry-drop")


@dataclass
class Check:
    status: str
    title: str
    detail: str


@dataclass
class Summary:
    verdict: str
    headline: str
    advice: str = ""
    checks: list[Check] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def summarize(report: dict[str, Any] | None, log_tail: str = "") -> Summary:
    if not report:
        return Summary(FAILED, t("crash.headline"), t("crash.advice", fix=t("fix")),
                       problems=[line for line in log_tail.splitlines()[-6:] if line.strip()])
    if report.get("status") == "failed":
        return _failed(report)
    checks = _checks(report)
    if not any(c.status == BAD for c in checks):
        return Summary(READY, t("verdict.ready"), checks=checks)
    return Summary(LIMITED, t("verdict.limited"), t("advice.limited"), checks=checks)


def _failed(report: dict[str, Any]) -> Summary:
    error = (report.get("errors") or [{}])[0]
    code = error.get("code", "")
    if code in KNOWN_ERRORS:
        headline, advice = t(f"error.{code}.title"), t(f"error.{code}.advice", fix=t("fix"), report=t("report"))
    else:
        headline, advice = t("failed.generic"), t("fix")
    problems = [_bone_words(d) for d in error.get("details", [])] or [error.get("message", "")]
    checks = _checks(report) if report.get("humanoid") else []
    return Summary(FAILED, headline, advice, checks, [p for p in problems if p])


def _bone_words(detail: str) -> str:
    return (detail.replace("missing required: ", t("problem.missing_bones"))
            .replace("missing required bone ", t("problem.missing_bone")))


def _checks(report: dict[str, Any]) -> list[Check]:
    checks = [_body(report), _hands(report), _blink(report), _lips(report), _moods(report), _eyes(report),
              _hair(report), _weight(report)]
    return [c for c in checks if c is not None]


def _join(items) -> str:
    return t("list.sep").join(items)


def _body(r: dict[str, Any]) -> Check:
    h = r["humanoid"]
    missing = h.get("missingRequired") or []
    if missing:
        return Check(BAD, t("check.body"), t("body.missing", bones=", ".join(missing)))
    return Check(OK, t("check.body"), t("body.ok", count=h["requiredTotal"]))


def _hands(r: dict[str, Any]) -> Check | None:
    hands = (r.get("anatomy") or {}).get("hands") or {}
    states = {side: (hands.get(side) or {}).get("state") for side in ("left", "right")}
    if not any(states.values()):
        return None
    if all(s == "bones-present" for s in states.values()):
        return Check(OK, t("check.fingers"), t("fingers.ok"))
    if any(s == "bones-present" for s in states.values()):
        return Check(WARN, t("check.fingers"), t("fingers.one"))
    return Check(WARN, t("check.fingers"), t("fingers.none"))


def _status(r: dict[str, Any], name: str) -> str | None:
    entry = r["expressions"].get("mapped", {}).get(name)
    return entry["status"] if entry else None


def _blink(r: dict[str, Any]) -> Check:
    if _status(r, "blink") or (_status(r, "blinkLeft") and _status(r, "blinkRight")):
        return Check(OK, t("check.blinking"), t("blink.ok", generated=_made(r, ("blink",))))
    return Check(WARN, t("check.blinking"), t("blink.none"))


def _lips(r: dict[str, Any]) -> Check:
    found = [v for v in VISEMES if _status(r, v)]
    if len(found) == len(VISEMES):
        return Check(OK, t("check.lips"), t("lips.ok", generated=_made(r, VISEMES)))
    if found:
        return Check(WARN, t("check.lips"), t("lips.some", found=", ".join(found)))
    return Check(BAD, t("check.lips"), t("lips.none"))


def _moods(r: dict[str, Any]) -> Check:
    found = [m for m in MOODS if _status(r, m)]
    names = _join(t(f"mood.{m}") for m in found)
    if len(found) >= 3:
        return Check(OK, t("check.expressions"), t("moods.ok", found=names, generated=_made(r, found)))
    if found:
        return Check(WARN, t("check.expressions"), t("moods.some", found=names))
    return Check(WARN, t("check.expressions"), t("moods.none"))


def _made(r: dict[str, Any], names) -> str:
    return t("generated") if any(_status(r, n) == "generated" for n in names) else ""


def _eyes(r: dict[str, Any]) -> Check:
    kind = (r.get("lookAt") or {}).get("type")
    if kind in ("bone", "expression"):
        return Check(OK, t("check.eyes"), t("eyes.ok"))
    return Check(WARN, t("check.eyes"), t("eyes.none"))


def _hair(r: dict[str, Any]) -> Check | None:
    springs = r.get("springBones") or {}
    if springs.get("chains"):
        return Check(OK, t("check.hair"), t("hair.ok"))
    hair = ((r.get("anatomy") or {}).get("hair") or {}).get("classification")
    if not hair:
        return None
    return Check(WARN, t("check.hair"), t("hair.static"))


def _weight(r: dict[str, Any]) -> Check:
    triangles = (r.get("geometry") or {}).get("triangles") or 0
    size = (r.get("output") or {}).get("sizeBytes") or 0
    count = f"{triangles:,}".replace(",", "\u202f")  # thin space: reads right in every language
    if triangles > HEAVY_TRIANGLES or size > HEAVY_BYTES:
        return Check(WARN, t("check.weight"), t("weight.heavy", triangles=count, size=_mb(size)))
    if size:
        return Check(OK, t("check.weight"), t("weight.ok_size", triangles=count, size=_mb(size)))
    return Check(OK, t("check.weight"), t("weight.ok", triangles=count))


def _mb(size: int) -> str:
    return t("size.mb", mb=f"{size / 1024 / 1024:.0f}") if size else t("size.unknown")
