# PyInstaller spec: one folder (onedir) app, plus a macOS .app bundle.
# Run from the project root:  pyinstaller packaging/moviva-fbx-to-vrm.spec --noconfirm
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
NAME = "Moviva FBX to VRM"
BPY = Path(importlib.util.find_spec("bpy").origin).parent

# bpy is a whole Blender: copied as-is, never analysed (its .so, datafiles and Python scripts load each other by path).
# Blender's own scripts and the VRM Add-on import standard-library modules PyInstaller cannot see, so ship them all.
skip = {"tkinter", "turtle", "turtledemo", "idlelib", "test", "lib2to3", "ensurepip", "pydoc_data", "__phello__"}
stdlib = sorted(m for m in sys.stdlib_module_names if m not in skip and not m.startswith("_"))
hidden = [sub for name in stdlib + ["numpy", "zstandard", "requests", "cattrs"]
          for sub in collect_submodules(name, filter=lambda m: ".tests" not in m and ".test." not in m)]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    datas=[
        (str(BPY), "bpy"),
        (str(ROOT / "vendor"), "vendor"),
        (str(ROOT / "character_vrm" / "toolchain.json"), "character_vrm"),
        (str(ROOT / "moviva_fbx_to_vrm" / "assets"), "moviva_fbx_to_vrm/assets"),
    ],
    hiddenimports=hidden + collect_submodules("character_vrm", filter=lambda m: ".tests" not in m),
    excludes=["bpy", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=NAME,
    console=False,  # the window; `convert …` still prints when started from a terminal on Linux / macOS
    icon=str(ROOT / "packaging" / "icon.png"),  # PyInstaller converts it (Pillow) to .ico / .icns
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
if sys.platform == "darwin":
    app = BUNDLE(coll, name=f"{NAME}.app", icon=str(ROOT / "packaging" / "icon.png"),
                 bundle_identifier="app.moviva.fbx-to-vrm",
                 info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "14.0"})
