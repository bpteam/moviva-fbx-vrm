"""Loads Blender (the ``bpy`` module) and the bundled VRM Add-on into this process.

The user never installs Blender: ``bpy`` comes from PyPI at build time and the VRM Add-on from
``packaging/fetch_addon.py``. Blender's user resources point to a private directory, so a Blender installed on the same
machine keeps its own preferences and add-ons untouched, and they never leak into a conversion.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ADDON_MODULE = "io_scene_vrm"


def vendor_dir() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen) / "vendor" if frozen else Path(__file__).resolve().parents[1] / "vendor"


def load_blender():
    addons = vendor_dir() / "addons"
    if not (addons / ADDON_MODULE / "__init__.py").is_file():
        raise RuntimeError(f"VRM Add-on not found in {addons}. Run: python packaging/fetch_addon.py")
    os.environ["BLENDER_USER_RESOURCES"] = str(Path(tempfile.gettempdir()) / "moviva-fbx-to-vrm-blender")
    sys.path.insert(0, str(addons))
    import bpy
    import addon_utils  # importable only once bpy is loaded

    if addon_utils.enable(ADDON_MODULE, default_set=True) is None:
        raise RuntimeError("Blender could not enable the VRM Add-on.")
    return bpy
