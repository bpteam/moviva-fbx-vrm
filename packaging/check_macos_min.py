"""Fails when any Mach-O file in the built app needs a newer macOS than the app promises (``LSMinimumSystemVersion``).

Run on macOS after the build:  python packaging/check_macos_min.py "dist/moviva-fbx-to-vrm.app" 14.0
"""

import re
import subprocess
import sys
from pathlib import Path

# thin 64/32-bit (little-endian as on Apple Silicon), universal and 64-bit universal (big-endian on disk, either order)
MACHO = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
         b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"}
# LC_BUILD_VERSION prints "minos 12.0"; the older LC_VERSION_MIN_MACOSX prints "version 10.13" after its cmd line
MIN = re.compile(r"LC_BUILD_VERSION.*?minos (\d+(?:\.\d+)*)|LC_VERSION_MIN_MACOSX.*?version (\d+(?:\.\d+)*)", re.S)


def version(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split("."))


def is_macho(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) in MACHO


def minimum(path: Path) -> tuple[int, ...] | None:
    """The newest macOS any architecture of ``path`` needs; None when otool cannot tell (an unverified file)."""
    run = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True)
    found = [version(a or b) for a, b in MIN.findall(run.stdout)]
    return max(found) if run.returncode == 0 and found else None


def main(app: str, limit: str) -> int:
    too_new = []
    for path in Path(app).rglob("*"):
        if path.is_symlink() or not path.is_file() or not is_macho(path):
            continue
        needs = minimum(path)
        if needs is None or needs > version(limit):
            too_new.append(f"{'.'.join(map(str, needs)) if needs else 'unknown'}  {path}")
    for line in too_new:
        print(f"ERROR needs macOS {line}", file=sys.stderr)
    print(f"{len(too_new)} Mach-O files need a macOS newer than {limit} or could not be read")
    return 1 if too_new else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
