"""PyInstaller entry point (it needs a script, not ``-m``)."""

from moviva_fbx_to_vrm.__main__ import main

raise SystemExit(main())
