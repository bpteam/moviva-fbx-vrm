"""``FakeBlenderBridge``: the test double for the BlenderBridge protocol (not a fake ``bpy``)."""

from __future__ import annotations

import copy
from pathlib import Path

from character_vrm.core.authoring.plan import AuthoringResult, AutoAuthorPlan
from character_vrm.core.errors import ConversionError
from character_vrm.core.model import CharacterInspection, GeometryOptimizationPlan, ShapeKeyInfo, VrmPlan
from character_vrm.tests.fixtures.glb import glb_bytes, vrm_gltf
from character_vrm.toolchain import EnvironmentFacts


class FakeBlenderBridge:
    def __init__(
        self,
        inspection: CharacterInspection | None = None,
        facts: EnvironmentFacts | None = None,
        *,
        fail_export: bool = False,
        output: str = "valid",  # valid | none | garbage | no-skins | spec-0 | missing-bone
    ) -> None:
        self.scene = inspection
        self.facts = facts or EnvironmentFacts("5.2.2", "4.7.2", True, True, True, True)
        self.fail_export = fail_export
        self.output = output
        self.calls: list[str] = []
        self.imported: tuple[str, str] | None = None
        self.plan: VrmPlan | None = None
        self.hidden: list[str] = []
        self.export_args: tuple | None = None
        self.geometry_plan: GeometryOptimizationPlan | None = None
        self.authoring_plan: AutoAuthorPlan | None = None
        self.rigged: CharacterInspection | None = None  # what the scene looks like after auto_rig()
        self.rig_result = None
        self.fail_authoring: set[str] = set()  # shape-key names the "mutation" refuses to create

    def environment(self) -> EnvironmentFacts:
        self.calls.append("environment")
        return self.facts

    def reset_scene(self) -> None:
        self.calls.append("reset_scene")

    def import_character(self, path: str, fmt: str) -> None:
        self.calls.append("import_character")
        self.imported = (path, fmt)

    def inspect(self, source_path: str, fmt: str) -> CharacterInspection:
        self.calls.append("inspect")
        assert self.scene is not None
        scene = copy.deepcopy(self.scene)
        scene.source_path, scene.source_format = source_path, fmt
        return scene

    def auto_rig(self):
        self.calls.append("auto_rig")
        from character_vrm.core.autorig import AutoRigResult
        assert self.rigged is not None, "test did not provide the post-rig scene"
        self.scene = self.rigged
        return self.rig_result or AutoRigResult("PASS", method="fake", confidence=0.9)

    def isolate_armature(self, armature: str) -> list[str]:
        self.calls.append("isolate_armature")
        assert self.scene is not None
        self.hidden = [m.name for m in self.scene.meshes if m.armature not in (None, armature)]
        return self.hidden

    def apply_authoring(self, plan: AutoAuthorPlan) -> AuthoringResult:
        self.calls.append("apply_authoring")
        self.authoring_plan = plan
        result = AuthoringResult()
        for key in plan.shape_keys:
            if key.name in self.fail_authoring:
                result.failed[key.name] = "refused by the test bridge"
            else:
                result.applied_shape_keys[key.name] = list(key.meshes)
                assert self.scene is not None
                for mesh in key.meshes:  # the "mutated scene" now carries the generated key
                    self.scene.shape_keys.append(ShapeKeyInfo(mesh, key.name))
        for f in plan.fingers:
            result.created_bones.update(b.name for b in f.bones)
        if plan.hair_rig:
            result.created_bones.update(b.name for c in plan.hair_rig.chains for b in c.bones)
        if plan.materials:
            result.materials = {e.material: e.changes for e in plan.materials.valid_edits()}
        return result

    def apply_geometry(self, plan: GeometryOptimizationPlan) -> None:
        self.calls.append("apply_geometry")
        self.geometry_plan = plan

    def configure_vrm(self, plan: VrmPlan) -> None:
        self.calls.append("configure_vrm")
        self.plan = plan

    def export_vrm(self, armature: str, output_path: str, all_influences: bool = False) -> None:
        self.calls.append("export_vrm")
        self.export_args = (armature, output_path, all_influences)
        if self.fail_export:
            raise ConversionError("export-failed", "bpy.ops.export_scene.vrm did not finish: ['CANCELLED']")
        assert self.plan is not None and self.scene is not None
        path = Path(output_path)
        if self.output == "none":
            return
        if self.output == "garbage":
            path.write_bytes(b"not a glb at all, but long enough to pass a size check")
            return
        kwargs = {"no-skins": {"skins": 0}, "spec-0": {"spec_version": "0.0"},
                  "missing-bone": {"drop_bone": "leftHand"}}.get(self.output, {})
        path.write_bytes(glb_bytes(vrm_gltf(self.plan, self.scene, **kwargs)))
