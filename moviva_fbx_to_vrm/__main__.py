"""No arguments: the window. Arguments: the ``character-vrm`` command line, run in this process with bundled Blender.

The window runs every conversion as a second copy of this program (``<app> convert …``), so a Blender crash on a
broken model ends that copy, not the window.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from .gui import run_gui
        return run_gui()

    if sys.stdout is None or sys.stderr is None:  # a windowed Windows build started without pipes
        _attach_console()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # the window reads UTF-8 progress line by line through a pipe
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    from .runtime import load_blender
    try:
        load_blender()
    except RuntimeError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    from character_vrm.blender.bridge import RealBlenderBridge
    from character_vrm.commands import run

    class ReportingBridge(RealBlenderBridge):
        """Prints a ``@stage`` line before each slow step; the window shows it as progress."""

        def import_character(self, path, fmt):
            print("@stage reading")
            return super().import_character(path, fmt)

        def inspect(self, source_path, fmt):
            print("@stage analysing")
            return super().inspect(source_path, fmt)

        def apply_authoring(self, plan):
            print("@stage authoring")
            return super().apply_authoring(plan)

        def export_vrm(self, armature, output_path, all_influences=False):
            print("@stage writing")
            return super().export_vrm(armature, output_path, all_influences)

    code = run(argv, ReportingBridge)
    sys.stdout.flush()
    return code


def _attach_console() -> None:
    """Print to the terminal the windowed exe was started from (``cmd``, PowerShell), or nowhere."""
    import os
    target = os.devnull
    if sys.platform == "win32":
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            target = "CONOUT$"
    stream = open(target, "w", encoding="utf-8", errors="replace")
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


if __name__ == "__main__":
    raise SystemExit(main())
