"""VRM expression mapping from *existing* shape keys. Nothing is fabricated.

Precedence per expression: explicit config  >  direct shape key (VRoid / A-I-U-E-O /
generic names)  >  compound of known shape keys (ARKit recipes)  >  missing.
Shape keys are only read; the source is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..profiles import arkit, character_creator, generic, vroid
from .config import DEFAULT_EXPRESSION_MIN_CONFIDENCE, ExpressionOverride
from .model import VRM_EXPRESSIONS, CharacterInspection, ConversionWarning, ExpressionMapping, MorphBind

VISEMES = ("aa", "ih", "ou", "ee", "oh")
BLINKS = ("blink", "blinkLeft", "blinkRight")
LOOKS = ("lookUp", "lookDown", "lookLeft", "lookRight")


class ExpressionConfigError(ValueError):
    pass


def norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


_NAMESPACE_SEPARATORS = (".", ":", "|", "__")
_SIDE = {"l": "left", "r": "right", "lt": "left", "rt": "right"}


def strip_namespace(name: str) -> str:
    """``Face_Blendshape.Fcl_MTH_A`` -> ``Fcl_MTH_A``; ``vrc.v_aa`` -> ``v_aa``; ``Genesis8Female__eCTRLvAA`` ->
    ``eCTRLvAA``. A trailing side marker (``Eye_Blink.L``) is not a namespace and is kept."""
    out = name
    for sep in _NAMESPACE_SEPARATORS:
        if sep in out:
            head, _, tail = out.rpartition(sep)
            if norm(tail) and norm(tail) not in ("l", "r", "left", "right") and head:
                out = tail
    return out


def _tokens(name: str) -> list[str]:
    tokens: list[str] = []
    cur = ""
    for i, c in enumerate(name):
        if not c.isalnum():
            if cur:
                tokens.append(cur)
            cur = ""
            continue
        if cur and c.isupper() and (cur[-1].islower() or (i + 1 < len(name) and name[i + 1].islower() and cur[-1].isupper())):
            tokens.append(cur)
            cur = ""
        cur += c
    if cur:
        tokens.append(cur)
    return [t.lower() for t in tokens]


def canon(name: str) -> str:
    """Namespace-free, side-normalised key: ``Mouth_Smile_L`` == ``mouthSmileLeft`` == ``Body.mouth_smile_left``."""
    return "".join(_SIDE.get(t, t) for t in _tokens(strip_namespace(name)))


def name_keys(name: str) -> set[str]:
    """Every normalised spelling a shape-key name is indexed under (the alias tables are matched against all)."""
    bare = strip_namespace(name)
    return {norm(name), norm(bare), canon(name)}


@dataclass
class ExpressionResult:
    mappings: dict[str, ExpressionMapping] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    below_threshold: list[ExpressionMapping] = field(default_factory=list)
    arkit_coverage: float = 0.0
    shape_key_count: int = 0
    warnings: list[ConversionWarning] = field(default_factory=list)

    @property
    def arkit(self) -> str:
        return "likely" if self.arkit_coverage >= 0.8 else "partial" if self.arkit_coverage >= 0.25 else "none"

    def status(self, name: str) -> str:
        m = self.mappings.get(name)
        if m is None:
            return "missing"
        if m.kind == "motion":
            return "inferred"
        return "available" if m.kind in ("direct", "config") else "generated"

    @property
    def capability(self) -> str:
        has_blink = "blink" in self.mappings
        visemes = [v for v in VISEMES if v in self.mappings]
        if has_blink and len(visemes) == len(VISEMES):
            return "full"
        return "partial" if (has_blink or visemes or self.mappings) else "none"


def _alias_tables() -> list[tuple[str, dict[str, dict[str, float]]]]:
    # more specific profiles first so their reason wins on equal confidence
    return [("profile:vroid", vroid.EXPRESSION_ALIASES),
            ("profile:character-creator", character_creator.EXPRESSION_ALIASES),
            ("name", generic.EXPRESSION_ALIASES)]


def _binds_for(keys: dict[str, list], names: set[str], weight: float = 1.0) -> list[MorphBind]:
    return [MorphBind(k.mesh, k.name, weight) for n in sorted(names) for k in keys.get(n, [])]


def resolve_expressions(
    inspection: CharacterInspection,
    overrides: dict[str, ExpressionOverride] | None = None,
    min_confidence: float = DEFAULT_EXPRESSION_MIN_CONFIDENCE,
    included_meshes: set[str] | None = None,
) -> ExpressionResult:
    overrides = overrides or {}
    result = ExpressionResult()
    shape_keys = [k for k in inspection.shape_keys if included_meshes is None or k.mesh in included_meshes]
    result.shape_key_count = len({k.name for k in shape_keys})
    by_norm: dict[str, list] = {}
    by_exact: dict[str, list] = {}
    for key in shape_keys:
        for n in name_keys(key.name):
            by_norm.setdefault(n, []).append(key)
        by_exact.setdefault(key.name, []).append(key)
    result.arkit_coverage = len(arkit.ARKIT_NORMALISED & set(by_norm)) / len(arkit.ARKIT_NORMALISED)

    candidates: dict[str, ExpressionMapping] = {}

    # 1. explicit config
    for name, override in overrides.items():
        if override.disabled:
            result.missing.append(name)
            continue
        binds = []
        for key_name, weight in override.binds.items():
            if key_name not in by_exact:
                raise ExpressionConfigError(
                    f"expressions.{name}: shape key '{key_name}' does not exist on the exported meshes")
            binds.extend(MorphBind(k.mesh, k.name, weight) for k in by_exact[key_name])
        candidates[name] = ExpressionMapping(name, "config", binds, 1.0, "config")

    disabled = {n for n, o in overrides.items() if o.disabled}

    # 2. direct
    for name in VRM_EXPRESSIONS:
        if name in candidates or name in disabled:
            continue
        best: tuple[float, str, set[str]] | None = None
        for reason, table in _alias_tables():
            for alias, conf in table.get(name, {}).items():
                if alias in by_norm and (best is None or conf > best[0]):
                    best = (conf, reason, {k.name for k in by_norm[alias]})
                elif alias in by_norm and best and conf == best[0]:
                    best[2].update(k.name for k in by_norm[alias])
        if best:
            candidates[name] = ExpressionMapping(name, "direct", _binds_for(by_exact, best[2]), best[0], best[1])

    # 3. compounds: rig-specific recipes first (they know the rig's own viseme set), then ARKit
    for reason, recipes in (("profile:character-creator", character_creator.RECIPES), ("arkit-compound", arkit.RECIPES)):
        for name, recipe in recipes.items():
            if name in candidates or name in disabled:
                continue
            if not all(canon(k) in by_norm for k in recipe.required):
                continue
            binds: list[MorphBind] = []
            seen: set[tuple[str, str]] = set()
            for table in (recipe.required, recipe.optional):
                for key_name, weight in table.items():
                    for k in by_norm.get(canon(key_name), []):
                        if (k.mesh, k.name) not in seen:
                            seen.add((k.mesh, k.name))
                            binds.append(MorphBind(k.mesh, k.name, weight))
            candidates[name] = ExpressionMapping(name, "compound", binds, recipe.confidence, reason)

    # blink from its two halves
    if "blink" not in candidates and "blink" not in disabled and "blinkLeft" in candidates and "blinkRight" in candidates:
        left, right = candidates["blinkLeft"], candidates["blinkRight"]
        candidates["blink"] = ExpressionMapping(
            "blink", "compound", [*left.binds, *right.binds], min(left.confidence, right.confidence),
            "blinkLeft+blinkRight")

    for name in VRM_EXPRESSIONS:
        mapping = candidates.get(name)
        if mapping is None:
            if name not in result.missing:
                result.missing.append(name)
        elif mapping.confidence < min_confidence:
            result.below_threshold.append(mapping)
            result.missing.append(name)
            result.warnings.append(ConversionWarning(
                "expression-low-confidence",
                f"Expression {name} <- {[b.shape_key for b in mapping.binds]} not applied: confidence "
                f"{mapping.confidence:.2f} < {min_confidence:.2f}."))
        else:
            result.mappings[name] = mapping
    return result
