"""Renders the README screenshots into ``docs/``: the start page and a result with lip sync missing.

Run from the project root:  QT_QPA_PLATFORM=offscreen QT_SCALE_FACTOR=2 python packaging/screenshots.py
The result comes from ``tests/fixtures/amber.report.json`` (a real conversion) with its mouth shapes removed.
"""

import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moviva_fbx_to_vrm.gui import STYLE, Window  # noqa: E402
from moviva_fbx_to_vrm.i18n import set_language  # noqa: E402
from moviva_fbx_to_vrm.summary import VISEMES, summarize  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    set_language("en")
    window = Window()
    window.resize(600, 900)
    window.show()
    app.processEvents()
    window.grab().save(str(ROOT / "docs" / "screenshot-start.png"))

    report = json.loads((ROOT / "tests" / "fixtures" / "amber.report.json").read_text(encoding="utf-8"))
    for name in VISEMES:
        report["expressions"]["mapped"].pop(name, None)
    window.output = Path("~/Models/amber.vrm")  # a neutral path, not the machine that renders
    window._show(summarize(report))
    app.processEvents()
    window.grab().save(str(ROOT / "docs" / "screenshot-result.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
