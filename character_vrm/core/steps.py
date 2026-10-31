"""The processing-step registry and the dependency-aware planner. The window renders ``plan()``; it decides nothing.

A step is *optional* when the user may switch it off. Mandatory stages (import, preflight, humanoid mapping, VRM
export, final validation) always run and are listed in ``MANDATORY`` for the progress view only. ``plan()`` answers,
for one model: is the step applicable, is it available, is it on by default, and what blocks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ConversionConfig

RECOMMENDED, OPTIONAL, NOT_NEEDED, UNAVAILABLE, EXPERIMENTAL = (
    "RECOMMENDED", "OPTIONAL", "NOT_NEEDED", "UNAVAILABLE", "EXPERIMENTAL")

MANDATORY = [  # (id, name) in execution order, the optional steps run between "map" and "export"
    ("import", "Import model"), ("preflight", "Analyze geometry"), ("map", "Map humanoid skeleton"),
    ("export", "Export VRM"), ("validate", "Validate result"),
]


@dataclass(frozen=True)
class StepSpec:
    id: str
    name: str
    engine: str
    does: str
    hardware: str
    software: str
    limits: str
    cost: str  # relative: "fast" | "standard" | "heavy"
    requires: tuple[str, ...] = ()  # capability names the model must have (see Facts.capabilities)
    after: tuple[str, ...] = ()  # step ids that must be enabled and applicable for this one to run
    experimental: bool = False
    available: bool = True
    unavailable_reason: str = ""


STEPS: tuple[StepSpec, ...] = (
    StepSpec("auto_rig", "Generate humanoid skeleton and skin weights",
             "Autorig Workbench + Mesh2Motion skeleton, Moviva arm tracer",
             "Fits Mesh2Motion's human skeleton with fingers to the body (arms traced along the mesh) and skins it "
             "two ways, the workbench's own and Blender bone heat; each is pose-tested and the better kept. "
             "Falls back to the procedural fit when the workbench finds no body.",
             "CPU", "bundled Blender", "Needs recognizable humanoid geometry, preferably a T- or A-pose. "
             "Stylized or covered limbs reduce quality. Skeleton and skin weights are one step.", "standard",
             requires=("humanoid_geometry", "no_armature")),
    StepSpec("face_expressions", "Generate facial expressions", "MediaPipe Face Landmarker + Moviva Face Auto-Author",
             "Finds eyes, lips and brows (MediaPipe on a rendered view, the face geometry otherwise) and generates "
             "missing blink, visemes and basic emotions as shape keys on the face mesh.",
             "CPU", "bundled", "Painted eyes cannot blink; a face MediaPipe does not recognise needs dense geometry. "
             "What cannot be made safely is skipped and reported.",
             "standard", requires=("armature_or_rig", "expressions_missing"), after=("auto_rig",)),
    StepSpec("spring_bones", "Configure secondary animation", "Moviva hair / spring-bone authoring",
             "Turns existing hair bones into VRM spring bones with colliders.", "CPU", "bundled",
             "Only existing hair bones are used unless the experimental hair rig is on.", "fast",
             requires=("armature_or_rig",), after=("auto_rig",)),
    StepSpec("finger_rig", "Generate finger bones", "Moviva experimental finger rig",
             "Bones for fingers that exist only as geometry.", "CPU", "bundled",
             "Experimental; mitten hands are reported as unavailable.", "standard", requires=("armature_or_rig",),
             experimental=True, after=("auto_rig",)),
    StepSpec("hair_rig", "Generate hair bones", "Moviva experimental hair rig",
             "Bones for a hair mesh that has none.", "CPU", "bundled", "Experimental.", "standard",
             requires=("armature_or_rig",), experimental=True, after=("auto_rig",)),
    StepSpec("ai_rigging", "AI auto-rigging (replaces the CPU auto-rig)", "SkinTokens / TokenRig",
             "AI skeleton (fingers included) and skin weights on the GPU; textures and scale are kept.",
             "NVIDIA GPU, 14 GB+ VRAM", "Python 3.11 + PyTorch CUDA + weights, downloaded once on demand (~12 GB)",
             "Linux x64 only for now. About a minute per model. Results vary from run to run.", "heavy",
             requires=("humanoid_geometry", "no_armature"), experimental=True),
)
BY_ID = {s.id: s for s in STEPS}
ALTERNATIVES = {"auto_rig": ("ai_rigging",)}  # a prerequisite another enabled step provides as well


@dataclass
class Facts:
    """What the preflight found. ``capabilities`` is derived; see ``capabilities_of``."""

    humanoid_geometry: bool = False
    armature: bool = False
    skinned: bool = False
    required_bones_mapped: int = 0
    face_geometry: bool = False
    expressions_missing: bool = False
    hair_bones: bool = False
    spring_bones_possible: bool = False
    ai_rig: str = "UNAVAILABLE"  # SkinTokens: READY / NOT_INSTALLED / UNAVAILABLE (set by the window, not the preflight)
    ai_rig_reason: str = ""

    def capabilities(self) -> set[str]:
        caps = set()
        if self.humanoid_geometry: caps.add("humanoid_geometry")
        if not self.armature: caps.add("no_armature")
        if self.armature or self.humanoid_geometry: caps.add("armature_or_rig")  # or one will be generated
        if self.face_geometry: caps.add("face_geometry")
        if self.expressions_missing: caps.add("expressions_missing")
        return caps


@dataclass
class PlannedStep:
    spec: StepSpec
    status: str
    enabled: bool
    why: str
    blocked_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        s = self.spec
        return {"id": s.id, "name": s.name, "engine": s.engine, "does": s.does, "hardware": s.hardware,
                "software": s.software, "limits": s.limits, "cost": s.cost, "status": self.status,
                "enabled": self.enabled, "why": self.why, "blockedBy": self.blocked_by, "experimental": s.experimental}


def plan(facts: Facts, selected: dict[str, bool] | None = None) -> list[PlannedStep]:
    """Default selection when ``selected`` is None; otherwise the user's choice, checked against prerequisites.

    A user-disabled step is never re-enabled. A step whose prerequisite step is disabled is *blocked* (and does not
    run), with the reason, unless the model already provides what the prerequisite would have produced.
    """
    caps = facts.capabilities()
    out: dict[str, PlannedStep] = {}
    for spec in STEPS:
        missing = [r for r in spec.requires if r not in caps]
        if not spec.available:
            out[spec.id] = PlannedStep(spec, UNAVAILABLE, False, spec.unavailable_reason)
            continue
        if spec.id == "ai_rigging" and facts.ai_rig == UNAVAILABLE and not missing:
            out[spec.id] = PlannedStep(spec, UNAVAILABLE, False, facts.ai_rig_reason or "No suitable NVIDIA GPU.")
            continue
        if "no_armature" in missing or (spec.id == "auto_rig" and facts.armature):
            out[spec.id] = PlannedStep(spec, NOT_NEEDED, False, "The model already has a skeleton; it is preserved.")
            continue
        if spec.id == "face_expressions" and "expressions_missing" in missing and "face_geometry" not in missing:
            out[spec.id] = PlannedStep(spec, NOT_NEEDED, False, "The model already has blink and viseme expressions.")
            continue
        if "humanoid_geometry" in missing:
            out[spec.id] = PlannedStep(spec, UNAVAILABLE, False, "No recognizable humanoid body was found in the geometry.")
            continue
        if "face_geometry" in missing:
            out[spec.id] = PlannedStep(spec, UNAVAILABLE, False, "No face geometry dense enough to author expressions.")
            continue
        status = EXPERIMENTAL if spec.experimental else RECOMMENDED
        default_on = not spec.experimental
        if spec.id == "spring_bones" and not facts.spring_bones_possible:
            status, default_on = OPTIONAL, False
        why = {"auto_rig": "The model has a humanoid body but no skeleton.",
               "face_expressions": "Blink and visemes are missing. Whatever the face cannot carry safely is skipped and reported.",
               "spring_bones": "Hair bones were found.",
               "ai_rigging": "Installed: runs instead of the CPU auto-rig." if facts.ai_rig == "READY" else
               "Not installed: switching it on offers the one-time download."}.get(spec.id, "Optional enhancement.")
        if spec.id == "spring_bones" and not facts.spring_bones_possible:
            why = "No hair bones were found, so there is nothing to configure."
        enabled = default_on if selected is None else bool(selected.get(spec.id, default_on))
        out[spec.id] = PlannedStep(spec, status, enabled, why)
    # Dependencies: a step after a disabled prerequisite cannot run unless the model has a rig already.
    for step in out.values():
        if not step.enabled:
            continue
        for dep in step.spec.after:
            d = out[dep]
            if d.status == NOT_NEEDED:
                continue  # the model already has what the prerequisite makes
            if any(out[a].enabled for a in ALTERNATIVES.get(dep, ()) if a in out):
                continue
            if not d.enabled:
                step.enabled = False
                step.blocked_by.append(dep)
                step.why = f"Needs '{d.spec.name}', which is switched off: the model has no skeleton to work on."
    return list(out.values())


def apply_selection(config: ConversionConfig, steps: list[PlannedStep]) -> ConversionConfig:
    """Write the plan into the converter configuration. Disabled steps stay off."""
    on = {s.spec.id: s.enabled for s in steps}
    config.auto_rig = on.get("auto_rig", False) and not on.get("ai_rigging", False)
    config.spring.enabled = on.get("spring_bones", False)
    a = config.auto_author
    if on.get("face_expressions"):
        a.enabled = a.generate_face = True
    if on.get("finger_rig"):
        a.enabled = a.generate_finger_rig = True
    if on.get("hair_rig"):
        a.enabled = a.generate_hair_rig = True
    return config


def cli_flags(steps: list[PlannedStep]) -> list[str]:
    """The ``convert`` options that realise a plan (the window runs the converter as a subprocess)."""
    on = {s.spec.id: s.enabled for s in steps}
    flags = []
    if on.get("ai_rigging"): flags.append("--ai-rig")  # the model then has a skeleton: no CPU auto-rig
    elif on.get("auto_rig"): flags.append("--auto-rig")
    if on.get("face_expressions"): flags.append("--generate-face")
    if on.get("finger_rig"): flags.append("--generate-finger-rig")
    if on.get("hair_rig"): flags.append("--generate-hair-rig")
    if not on.get("spring_bones"): flags.append("--no-springs")
    return flags
