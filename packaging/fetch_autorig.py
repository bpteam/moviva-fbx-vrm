"""Downloads the pinned Autorig Workbench and Mesh2Motion's human skeleton into ``vendor/autorig`` (build time only).

Autorig Workbench (GPL-3.0, https://github.com/sambena/autorig-workbench) fits and skins a body rig; only its
``autorig/core`` and ``autorig/steps`` are kept. Mesh2Motion (rigs and clips CC0, https://github.com/Mesh2Motion)
supplies the human skeleton it fits.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

AW_COMMIT = "80a127582aa55f4c8378d60dcff57725dd24ef8f"
AW_URL = f"https://codeload.github.com/sambena/autorig-workbench/zip/{AW_COMMIT}"
AW_SHA256 = "05b8f62e0e6ba6660b890edaa2d06eabb332abea7e350cb141ba2f9a912868ac"
M2M_COMMIT = "66631497cb4581d957d08a5bff9e1775c4446d89"
M2M_FILES = {
    "rigs/rig-human.glb": "3f7ad51c9471458a920dbe373755c3a7223e3672bb572406fa7d2e65deb8f593",
    "animations/human-base-animations.glb": "406eb0a8dc4ab366e623b79b6e3005a4951392e1bda78ae39c1099d31147733c",
}
TARGET = Path(__file__).resolve().parents[1] / "vendor" / "autorig"
KEEP = ("autorig/core/", "autorig/steps/", "LICENSE")


def _get(url: str, sha256: str) -> bytes:
    data = urllib.request.urlopen(url, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{url}: sha256 {digest}, expected {sha256}")
    return data


def main() -> int:
    try:
        aw = _get(AW_URL, AW_SHA256)
        m2m = {f: _get(f"https://raw.githubusercontent.com/sambena/mesh2motion-app/{M2M_COMMIT}/static/{f}", h)
               for f, h in M2M_FILES.items()}
    except RuntimeError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    shutil.rmtree(TARGET, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(aw)) as archive:
        for info in archive.infolist():
            rel = info.filename.split("/", 1)[1]
            if info.is_dir() or not rel.startswith(KEEP) or "__pycache__" in rel:
                continue
            out = TARGET / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(info))
    for f, data in m2m.items():
        out = TARGET / "mesh2motion" / "static" / f
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    print(f"Autorig Workbench {AW_COMMIT[:12]} + Mesh2Motion human -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
