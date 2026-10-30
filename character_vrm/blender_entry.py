"""Runs INSIDE Blender:  blender --background --python-exit-code 1 --python blender_entry.py -- <command> ...

Everything after ``--`` is the character-vrm command line (without ``--blender``).
"""

import os
import sys
from pathlib import Path

# Make ``character_vrm`` importable from Blender's Python (its parent directory is ``converter/``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _forwarded_args() -> list:
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def main() -> None:
    from character_vrm.blender.bridge import RealBlenderBridge
    from character_vrm.commands import run

    code = run(_forwarded_args(), RealBlenderBridge)
    sys.stdout.flush()
    sys.stderr.flush()
    if code:
        # Explicit exit code regardless of how this Blender build treats SystemExit in --python scripts.
        os._exit(code)


if __name__ == "__main__":
    main()
