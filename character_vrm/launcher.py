"""Spawns Blender. This is the only thing the outer (plain Python) process does."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Sequence

BLENDER_ENV = "BLENDER_BIN"
ENTRY_SCRIPT = Path(__file__).with_name("blender_entry.py")


class LauncherError(RuntimeError):
    pass


def resolve_blender(
    explicit: str | None,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Priority: ``--blender``  >  ``$BLENDER_BIN``  >  ``blender`` on PATH."""
    env = os.environ if env is None else env
    if explicit:
        source, candidate = "--blender", explicit
    elif env.get(BLENDER_ENV):
        source, candidate = f"${BLENDER_ENV}", env[BLENDER_ENV]
    else:
        found = which("blender")
        if not found:
            raise LauncherError(
                f"Blender not found. Pass --blender PATH, set {BLENDER_ENV}, or put blender on PATH.")
        return found
    path = Path(candidate).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise LauncherError(f"{source} points to '{candidate}', which is not an executable file.")
    return str(path)


def build_command(blender: str, forwarded: Sequence[str]) -> list[str]:
    return [blender, "--background", "--python-exit-code", "1", "--python", str(ENTRY_SCRIPT), "--", *forwarded]


def run_blender(blender: str, forwarded: Sequence[str]) -> int:
    try:
        return subprocess.call(build_command(blender, forwarded))
    except OSError as exc:
        raise LauncherError(f"Could not start {blender}: {exc}") from exc
