"""The adapter boundary between the pure converter core and Blender.

Production: ``ConverterCore -> BlenderBridge -> RealBlenderBridge -> bpy``.
Tests:      ``ConverterCore -> FakeBlenderBridge`` (tests/fakes.py).

This module imports no ``bpy`` at import time, so ``import character_vrm`` works anywhere.
"""

from __future__ import annotations

from typing import Protocol

from ..core.authoring.plan import AuthoringResult, AutoAuthorPlan
from ..core.model import CharacterInspection, GeometryOptimizationPlan, VrmPlan
from ..toolchain import EnvironmentFacts


class BlenderBridge(Protocol):
    def environment(self) -> EnvironmentFacts:
        """Facts for ``doctor``: versions and operator availability."""

    def reset_scene(self) -> None:
        """Stage 1: known-empty scene, independent of startup file, selection, UI context."""

    def import_character(self, path: str, fmt: str) -> None:
        """Stage 2: import ``glb`` / ``gltf`` / ``fbx`` through Blender's own importers."""

    def inspect(self, source_path: str, fmt: str) -> CharacterInspection:
        """Stage 3: what actually survived the import, as a ``bpy``-free description."""

    def isolate_armature(self, armature: str) -> list[str]:
        """Exclude every other armature and the meshes it deforms from the export."""

    def apply_authoring(self, plan: AutoAuthorPlan) -> AuthoringResult:
        """MUTATE + VALIDATE RESULT: create the planned ``Prosopon_`` shape keys / bones and report what exists."""

    def apply_geometry(self, plan: GeometryOptimizationPlan) -> None:
        """Apply a previously planned crop after masks and dependencies are known."""

    def configure_vrm(self, plan: VrmPlan) -> None:
        """Stages 6-9, 11: write the plan into the VRM Add-on's properties."""

    def export_vrm(self, armature: str, output_path: str, all_influences: bool = False) -> None:
        """Stage 12: ``bpy.ops.export_scene.vrm``. Raises ``ConversionError`` if it does not finish."""


class RealBlenderBridge:
    """Runs inside Blender's Python. All ``bpy`` use is delegated to the adapter modules."""

    def environment(self) -> EnvironmentFacts:
        from . import vrm_addon
        return vrm_addon.environment_facts()

    def reset_scene(self) -> None:
        from . import importer
        importer.reset_scene()

    def import_character(self, path: str, fmt: str) -> None:
        from . import importer
        importer.import_character(path, fmt)

    def inspect(self, source_path: str, fmt: str) -> CharacterInspection:
        from . import scene_inspector
        return scene_inspector.inspect_scene(source_path, fmt)

    def isolate_armature(self, armature: str) -> list[str]:
        from . import scene_inspector
        return scene_inspector.isolate_armature(armature)

    def apply_authoring(self, plan: AutoAuthorPlan) -> AuthoringResult:
        from . import anatomy_adapter
        return anatomy_adapter.apply_authoring(plan)

    def apply_geometry(self, plan: GeometryOptimizationPlan) -> None:
        from . import scene_inspector
        scene_inspector.apply_geometry_plan(plan)

    def configure_vrm(self, plan: VrmPlan) -> None:
        from . import vrm_addon
        vrm_addon.configure(plan)

    def export_vrm(self, armature: str, output_path: str, all_influences: bool = False) -> None:
        from . import vrm_addon
        vrm_addon.export(armature, output_path, all_influences)
