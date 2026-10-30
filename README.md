<h1 align="center"><a href="https://moviva.bpt22.tech/?utm_source=github">Moviva FBX → VRM</a></h1>

<p align="center">
  <a href="https://moviva.bpt22.tech/?utm_source=github"><img src="docs/moviva.jpg" alt="Moviva: make ChatGPT Voice feel alive" width="100%"></a>
</p>

<p align="center">
  <b>Sponsored by <a href="https://moviva.bpt22.tech/?utm_source=github">Moviva</a></b>, the AI avatar that turns ChatGPT Voice into a character who speaks,
  listens and reacts to you in real time.
</p>

Drop a rigged 3D character (FBX, GLB or glTF), get a **VRM 1.0** avatar for [Moviva](https://moviva.bpt22.tech/?utm_source=github) and any other VRM app.
Everything runs on your computer: nothing is uploaded, nothing to install, Blender is built in.

The window speaks the six languages of the Moviva extension (English, Russian, Spanish, Ukrainian, Japanese,
Simplified Chinese) and follows the system language; strings live in `moviva_fbx_to_vrm/assets/locales/`.

After a conversion the window says, in plain words, what the avatar can do (body, fingers, blinking, lip sync,
facial expressions, eye contact, hair, weight) and, when the model cannot be converted or something is missing,
why and what to fix.

## Download

[Releases](../../releases): one archive per system. Unpack it: inside is a `moviva-fbx-to-vrm` folder (an app on macOS); double-click
`moviva-fbx-to-vrm` (`moviva-fbx-to-vrm.exe` on Windows) to open the window.

| System | File | First start |
|---|---|---|
| Windows 10/11 x64 | `moviva-fbx-to-vrm-windows-x64.zip` | SmartScreen: *More info → Run anyway* (the app is not signed) |
| macOS 14+ Apple Silicon | `moviva-fbx-to-vrm-macos-arm64.zip` | *System Settings → Privacy & Security → Open Anyway*, or `xattr -dr com.apple.quarantine moviva-fbx-to-vrm.app` |
| Linux x64 (glibc 2.35+: Ubuntu 22.04, Debian 12, Fedora 36 or newer) | `moviva-fbx-to-vrm-linux-x64.tar.gz` | needs an X11/Wayland desktop (`libxcb-cursor0` on Debian/Ubuntu) |

Intel Macs and ARM Linux are not supported: Blender publishes no `bpy` build for them.

## Try it with Moviva

1. Install [Moviva](https://moviva.bpt22.tech/?utm_source=github) for Chrome (free).
2. Convert your character here.
3. Load the `.vrm` in Moviva as your own avatar (Moviva Plus) and talk to it in ChatGPT Voice: it lip-syncs,
   blinks, gestures and shows emotions as the conversation goes.

## What makes a good source model

- **A skeleton** (armature) that moves the mesh. Unrigged models cannot be converted; Mixamo's free auto-rigger fixes that.
- **One character** per file.
- **Facial blend shapes** (ARKit, VRoid, Character Creator, or visemes) for lip sync, blinking and expressions. Without
  them the converter generates what the face geometry allows; otherwise the avatar moves but its lips stay still.
- **Finger bones** for hand gestures, **hair bones** for swaying hair.
- A reasonable size: above ~150k triangles or ~150 MB the avatar loads and animates slowly on weaker machines.

## Command line

The same program converts without the window:

```bash
# Linux
tar -xzf moviva-fbx-to-vrm-linux-x64.tar.gz
./moviva-fbx-to-vrm/moviva-fbx-to-vrm                # the window
./moviva-fbx-to-vrm/moviva-fbx-to-vrm doctor         # checks the built-in Blender and VRM Add-on
./moviva-fbx-to-vrm/moviva-fbx-to-vrm convert model.fbx model.vrm --auto-author --generate-face --geometry-profile video-call

# macOS
moviva-fbx-to-vrm.app/Contents/MacOS/moviva-fbx-to-vrm doctor
```

All `character-vrm` commands and options work; see [character_vrm/README.md](character_vrm/README.md). A conversion
writes `model.vrm` and `model.vrm.report.json`. On Windows the app is a windowed program: run it as
`start /wait "" moviva-fbx-to-vrm\moviva-fbx-to-vrm.exe convert …` in `cmd` (or `Start-Process -Wait` in PowerShell) so the prompt
waits for it.

## How it is built

| Part | Where it comes from |
|---|---|
| `character_vrm/` | the converter: humanoid mapping, anatomy analysis, face / hair authoring, VRM plan (pure Python) |
| `moviva_fbx_to_vrm/` | the window (PySide6), the plain-words summary, the in-process Blender loader |
| Blender 5.2.2 | the [`bpy`](https://pypi.org/project/bpy/) wheel from PyPI, installed at build time |
| VRM Add-on for Blender 4.7.2 | [GitHub release](https://github.com/saturday06/VRM-Addon-for-Blender/releases/tag/v4.7.2), downloaded and checksum-verified at build time by `packaging/fetch_addon.py` |

Neither Blender nor the add-on is in this repository. The window runs each conversion as a second copy of itself, so
a Blender crash on a broken model never takes the window down. Blender's user resources point to a private temporary
directory: a Blender installed on the same machine keeps its own settings and add-ons.

### Develop

Python **3.13** exactly (the `bpy` wheel is built for one Python version).

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python packaging/fetch_addon.py
python -m pytest -q
python -m moviva_fbx_to_vrm            # the window
python -m moviva_fbx_to_vrm doctor     # the command line
pyinstaller packaging/moviva-fbx-to-vrm.spec --noconfirm --workpath .pyi-work --distpath dist
```

`.github/workflows/build.yml` tests, builds and smoke-tests Linux, Windows and macOS (Apple Silicon) on every pull
request (the binaries are downloadable from the run for 14 days). A merge to `master` does the same and then publishes
the three archives as GitHub release `v<version>.<run number>` (version from `pyproject.toml`); if any system fails,
nothing is released. PyInstaller cannot cross-compile, so each system is
built on its own runner.

### Intro picture

`moviva_fbx_to_vrm/assets/intro.png`, **1088 × 400 px** (shown at 544 × 200, sharp on HiDPI screens). Without it the
window shows a text placeholder. The window icon is `assets/logo.png`; the app icon is `packaging/icon.png`.

### Upgrading Blender or the add-on

Change `character_vrm/toolchain.json`, the `bpy` pin in `pyproject.toml` and `VERSION` / `SHA256` in
`packaging/fetch_addon.py` together, then review `character_vrm/blender/compat_v4_7_2.py`. `doctor` refuses any
mismatch.

## License

GPL-3.0-or-later (it runs Blender's GPL Python API). The VRM Add-on for Blender is MIT / GPL-3.0-or-later.
