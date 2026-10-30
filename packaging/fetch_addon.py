"""Downloads the pinned VRM Add-on for Blender into ``vendor/addons/io_scene_vrm`` (build time only)."""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "4.7.2"
URL = ("https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/"
       f"v{VERSION}/VRM_Addon_for_Blender-{VERSION.replace('.', '_')}.zip")
SHA256 = "f3dc4f3c1e84d501b9d52e0efee45867dc2127271aa763f1e36f2d4483732dbf"
TARGET = Path(__file__).resolve().parents[1] / "vendor" / "addons" / "io_scene_vrm"


def main() -> int:
    data = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"ERROR {URL}: sha256 {digest}, expected {SHA256}", file=sys.stderr)
        return 1
    shutil.rmtree(TARGET, ignore_errors=True)
    TARGET.mkdir(parents=True)
    prefix = "VRM_Addon_for_Blender-release/"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            out = TARGET / info.filename[len(prefix):]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(info))
    print(f"VRM Add-on {VERSION} -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
