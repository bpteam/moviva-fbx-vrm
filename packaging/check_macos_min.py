"""Fails when any Mach-O file in the built app needs a newer macOS than the app promises (``LSMinimumSystemVersion``).

Run on macOS after the build:  python packaging/check_macos_min.py "dist/moviva-fbx-to-vrm.app" 12.0
"""

import re
import subprocess
import sys
from pathlib import Path

MACHO = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe"}  # 64-bit, 32-bit, universal
# LC_BUILD_VERSION prints "minos 12.0"; the older LC_VERSION_MIN_MACOSX prints "version 10.13" after its cmd line
MIN = re.compile(r"LC_BUILD_VERSION.*?minos (\d+(?:\.\d+)*)|LC_VERSION_MIN_MACOSX.*?version (\d+(?:\.\d+)*)", re.S)


def version(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split("."))


def main(app: str, limit: str) -> int:
    too_new = []
    for path in Path(app).rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as f:
            if f.read(4) not in MACHO:
                continue
        out = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True).stdout
        needs = max((version(a or b) for a, b in MIN.findall(out)), default=(0,))
        if needs > version(limit):
            too_new.append(f"{'.'.join(map(str, needs))}  {path}")
    for line in too_new:
        print(f"ERROR needs macOS {line}", file=sys.stderr)
    print(f"{len(too_new)} Mach-O files need a macOS newer than {limit}")
    return 1 if too_new else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
