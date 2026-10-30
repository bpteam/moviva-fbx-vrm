"""``python3 -m character_vrm.cli`` / ``character-vrm``: parse arguments, spawn Blender, forward its exit code.

The conversion itself runs inside Blender (``blender_entry.py``); this process never imports ``bpy``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from .commands import EXIT_FAILED, EXIT_USAGE, build_parser
from .core.pipeline import source_format
from .core.errors import ConversionError
from .launcher import LauncherError, resolve_blender, run_blender


def strip_blender_option(argv: Sequence[str]) -> list[str]:
    """Everything except ``--blender PATH`` / ``--blender=PATH`` (Blender's entry point does not know it)."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--blender":
            skip = True
        elif not arg.startswith("--blender="):
            out.append(arg)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)  # argparse exits with 2 on bad usage

    # Cheap failures before starting a multi-second Blender.
    try:
        if args.command != "doctor":
            if not Path(args.source).is_file():
                print(f"ERROR source not found: {args.source}", file=sys.stderr)
                return EXIT_FAILED
            source_format(args.source)
        if getattr(args, "config", None) and not Path(args.config).is_file():
            print(f"ERROR config not found: {args.config}", file=sys.stderr)
            return EXIT_FAILED
    except ConversionError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        blender = resolve_blender(getattr(args, "blender", None))
        return run_blender(blender, strip_blender_option(argv))
    except LauncherError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
