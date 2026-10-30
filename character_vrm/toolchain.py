"""Pinned toolchain (``toolchain.json``) and the pure doctor logic.

Nothing here imports ``bpy``: the Blender side only *collects* the facts
(``EnvironmentFacts``) and this module decides whether they are acceptable.
Upgrading Blender or the VRM Add-on is an explicit change of ``toolchain.json``
plus a review of ``blender/compat_v4_7_2.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TOOLCHAIN_PATH = Path(__file__).with_name("toolchain.json")


@dataclass(frozen=True)
class Toolchain:
    blender: str
    vrm_addon: str
    vrm: str

    @property
    def target(self) -> str:
        return f"VRM {self.vrm}"


def load_toolchain(path: Path = TOOLCHAIN_PATH) -> Toolchain:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return Toolchain(blender=data["blender"], vrm_addon=data["vrmAddon"], vrm=data["vrm"])
    except KeyError as exc:
        raise ValueError(f"{path}: missing key {exc}") from exc


@dataclass
class EnvironmentFacts:
    """What the Blender process found. ``None`` means "not found"."""

    blender_version: str | None = None
    addon_version: str | None = None
    exporter_operator: bool = False
    gltf_importer: bool = False
    gltf_exporter: bool = False  # the VRM exporter drives Blender's glTF exporter
    fbx_importer: bool = False
    addon_error: str | None = None


@dataclass
class DoctorCheck:
    title: str
    lines: list[str]
    ok: bool


@dataclass
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)
    target: str = ""

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def format(self) -> str:
        out = ["Moviva Character Converter", ""]
        for check in self.checks:
            out.append(check.title)
            out.extend(f"  {line}" for line in check.lines)
            out.append("  OK" if check.ok else "  FAIL")
            out.append("")
        out.extend(["Target", f"  {self.target}", ""])
        out.append("Environment OK" if self.ok else "Environment NOT OK")
        return "\n".join(out)


def _version_check(title: str, expected: str, detected: str | None, extra: str | None) -> DoctorCheck:
    lines = [f"expected: {expected}", f"detected: {detected if detected else 'not found'}"]
    if extra:
        lines.append(extra)
    return DoctorCheck(title, lines, ok=detected == expected)


def evaluate_environment(facts: EnvironmentFacts, toolchain: Toolchain | None = None) -> DoctorReport:
    """Exact-match policy: any other Blender / add-on version fails."""
    toolchain = toolchain or load_toolchain()
    report = DoctorReport(target=toolchain.target)
    report.checks.append(_version_check("Blender", toolchain.blender, facts.blender_version, None))
    report.checks.append(
        _version_check("VRM Add-on", toolchain.vrm_addon, facts.addon_version, facts.addon_error)
    )
    report.checks.append(
        DoctorCheck("VRM exporter", ["bpy.ops.export_scene.vrm"], ok=facts.exporter_operator)
    )
    report.checks.append(
        DoctorCheck(
            "Import / export operators",
            [
                f"bpy.ops.import_scene.gltf: {'present' if facts.gltf_importer else 'MISSING'}",
                f"bpy.ops.wm.fbx_import: {'present' if facts.fbx_importer else 'MISSING'}",
                f"bpy.ops.export_scene.gltf: {'present' if facts.gltf_exporter else 'MISSING'}",
            ],
            ok=facts.gltf_importer and facts.fbx_importer and facts.gltf_exporter,
        )
    )
    return report
