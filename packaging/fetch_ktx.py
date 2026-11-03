"""Downloads Khronos' KTX-Software ``ktx`` tool into ``vendor/ktx`` (build time only; Linux x64 for now).

It encodes the opt-in KTX2 textures (``character_vrm/core/compress.py``). Apache-2.0. Only ``bin/ktx`` and the
``libktx`` it loads from ``../lib`` (its rpath) are kept. On other systems nothing is fetched and the KTX2 option is
shown as unavailable.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "4.4.2"
ASSETS = {
    "linux": (f"KTX-Software-{VERSION}-Linux-x86_64.tar.bz2",
              "a8781bad05f9624edbf910b7f258cd0a4ba7d3e63b49ecc0a0ab440bf6a0a245"),
}
TARGET = Path(__file__).resolve().parents[1] / "vendor" / "ktx"


def main() -> int:
    asset = ASSETS.get("linux" if sys.platform.startswith("linux") else sys.platform)
    if asset is None:
        print(f"KTX-Software: no build for {sys.platform} yet; KTX2 stays unavailable there")
        return 0
    name, sha = asset
    url = f"https://github.com/KhronosGroup/KTX-Software/releases/download/v{VERSION}/{name}"
    data = urllib.request.urlopen(url, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha:
        print(f"ERROR {url}: sha256 {digest}, expected {sha}", file=sys.stderr)
        return 1
    shutil.rmtree(TARGET, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
        for m in tar.getmembers():
            rel = m.name.split("/", 1)[-1]
            if rel == "bin/ktx" or (rel.startswith("lib/libktx.so") and "jni" not in rel):
                out = TARGET / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                if m.issym():
                    out.symlink_to(m.linkname)
                elif m.isfile():
                    out.write_bytes(tar.extractfile(m).read())
                    out.chmod(0o755)
    print(f"KTX-Software {VERSION} -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
