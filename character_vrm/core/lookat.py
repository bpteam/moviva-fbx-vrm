"""VRM LookAt: bone-based by default, expression-based when the model has eye-direction morphs."""

from __future__ import annotations

from .config import LookAtConfig
from .expressions import LOOKS, ExpressionResult
from .humanoid import HumanoidResult
from .model import ConversionWarning, LookAtPlan


def plan_lookat(
    humanoid: HumanoidResult,
    expressions: ExpressionResult,
    config: LookAtConfig | None = None,
) -> tuple[LookAtPlan, list[ConversionWarning]]:
    """A missing LookAt is a warning, never an error.

    ``offsetFromHeadBone`` is deliberately not derived from the skeleton: the VRM Add-on stores it
    in an axis convention that depends on the head bone's own axes, so it is passed through only
    when the user supplies it in the map file. Unset, the add-on default applies.
    """
    config = config or LookAtConfig()
    warnings: list[ConversionWarning] = []
    has_eyes = "leftEye" in humanoid.mappings and "rightEye" in humanoid.mappings
    has_look_morphs = all(name in expressions.mappings for name in LOOKS)

    kind = config.type
    if kind == "none":
        return LookAtPlan(None, "disabled by config"), warnings
    if kind == "bone" and not has_eyes:
        warnings.append(ConversionWarning("lookat-missing", "lookAt.type=bone requested but leftEye/rightEye are not mapped."))
        return LookAtPlan(None, "eye bones missing"), warnings
    if kind == "expression" and not has_look_morphs:
        warnings.append(ConversionWarning(
            "lookat-missing", "lookAt.type=expression requested but lookUp/lookDown/lookLeft/lookRight are not all available."))
        return LookAtPlan(None, "look expressions missing"), warnings

    if kind == "bone" or (kind is None and has_eyes):
        return LookAtPlan("bone", "leftEye and rightEye are mapped", config.offset_from_head_bone), warnings
    if kind == "expression" or (kind is None and has_look_morphs):
        return LookAtPlan("expression", "all four eye-direction expressions are available",
                          config.offset_from_head_bone), warnings
    warnings.append(ConversionWarning(
        "lookat-missing", "No eye bones and no eye-direction morphs: VRM LookAt is not configured."))
    return LookAtPlan(None, "no eye bones and no eye-direction morphs"), warnings
