"""The window: drop a character, wait, get a VRM and a plain-words account of what the avatar can do."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from . import __version__
from .i18n import language_for, set_language, t
from .summary import BAD, FAILED, OK, READY, WARN, Summary, summarize

SOURCES = (".fbx", ".glb", ".gltf")
ASSETS = Path(__file__).with_name("assets")
INTRO_SIZE = (544, 200)  # shown size; assets/intro.png should be twice that (1088 x 400) for sharp HiDPI screens
STAGES = ("reading", "analysing", "authoring", "writing")  # ``@stage <name>`` lines from the converter
COLORS = {OK: "#57d797", WARN: "#ffc053", BAD: "#ff6574", READY: "#57d797", FAILED: "#ff6574"}
MARKS = {OK: "✓", WARN: "!", BAD: "✕"}

STYLE = """
QWidget { background: #0b1319; color: #eef3f7; font-size: 14px; }
QLabel#title { font-size: 20px; font-weight: 700; }
QLabel#muted, QCheckBox#muted { color: #a9bac8; }
QFrame#drop { border: 2px dashed #405461; border-radius: 14px; background: #0c141a; }
QFrame#drop[hover="true"] { border-color: #ff5d2b; background: #21191a; }
QFrame#card { background: #101b22; border-radius: 12px; }
QLabel#intro { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #15212b, stop:1 #21191a);
               border-radius: 12px; color: #a9bac8; }
QLabel#step { background: #ff5d2b; color: white; border-radius: 11px; font-weight: 700; }
QPushButton { background: #15212b; border: 1px solid #24343f; border-radius: 8px; padding: 9px 16px; }
QPushButton:hover { background: #1b2a36; }
QPushButton#primary { background: #ff5d2b; border: none; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #ff7546; }
QProgressBar { background: #15212b; border: none; border-radius: 4px; max-height: 8px; }
QProgressBar::chunk { background: #ff5d2b; border-radius: 4px; }
QScrollArea { border: none; }
"""


def converter_command(args: list[str]) -> tuple[str, list[str]]:
    """This same program, in command-line mode (frozen app or ``python -m``)."""
    if getattr(sys, "frozen", False):
        return sys.executable, args
    return sys.executable, ["-m", "moviva_fbx_to_vrm", *args]


def free_output_path(source: Path) -> Path:
    candidate = source.with_suffix(".vrm")
    n = 2
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}-{n}.vrm")
        n += 1
    return candidate


class DropZone(QFrame):
    def __init__(self, on_file) -> None:
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(150)
        self.on_file = on_file
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("⬇")
        icon.setStyleSheet("font-size: 40px;")
        big = QLabel(t("drop.title"))
        big.setStyleSheet("font-size: 17px; font-weight: 600;")
        small = QLabel(t("drop.hint"))
        small.setObjectName("muted")
        for w in (icon, big, small):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setWordWrap(True)
            w.setStyleSheet(w.styleSheet() + "background: transparent;")
            layout.addWidget(w)

    def _hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    @staticmethod
    def _source(event) -> Path | None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SOURCES:
                return path
        return None

    def dragEnterEvent(self, event) -> None:
        if self._source(event):
            event.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, event) -> None:
        self._hover(False)

    def dropEvent(self, event) -> None:
        self._hover(False)
        path = self._source(event)
        if path:
            self.on_file(path)

    def mousePressEvent(self, event) -> None:
        name, _ = QFileDialog.getOpenFileName(self, t("drop.dialog"), "", f"{t('drop.filter')} (*.fbx *.glb *.gltf)")
        if name:
            self.on_file(Path(name))


class Window(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(t("window.title"))
        self.setWindowIcon(QIcon(str(ASSETS / "logo.png")))
        self.resize(600, 780)
        self.process: QProcess | None = None
        self.log: list[str] = []
        self.output: Path | None = None
        self.started = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        header = QHBoxLayout()
        header.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(QPixmap(str(ASSETS / "logo.png")).scaled(
            44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        names = QVBoxLayout()
        names.setSpacing(0)
        title = QLabel("Moviva")
        title.setObjectName("title")
        subtitle = QLabel(t("header.subtitle"))
        subtitle.setObjectName("muted")
        names.addWidget(title)
        names.addWidget(subtitle)
        header.addWidget(logo)
        header.addLayout(names, 1)
        root.addLayout(header)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._start_page())
        self.pages.addWidget(self._progress_page())
        self.pages.addWidget(self._result_page())
        root.addWidget(self.pages, 1)

        footer = QLabel(t("footer", version=__version__))
        footer.setObjectName("muted")
        footer.setStyleSheet("font-size: 11px;")
        root.addWidget(footer)

        self.ticker = QTimer(self, interval=1000, timeout=self._tick)

    # ------------------------------------------------------------------ pages

    def _start_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._intro(), 0, Qt.AlignmentFlag.AlignHCenter)
        for n in (1, 2, 3):
            row = QHBoxLayout()
            row.setSpacing(10)
            badge = QLabel(str(n))
            badge.setObjectName("step")
            badge.setFixedSize(22, 22)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(badge)
            step = QLabel(t(f"step.{n}"))
            step.setWordWrap(True)
            row.addWidget(step, 1)
            layout.addLayout(row)
        layout.addSpacing(4)
        layout.addWidget(DropZone(self.convert), 1)
        self.auto_face = QCheckBox(t("autoface"))
        self.auto_face.setChecked(True)
        layout.addWidget(self.auto_face)
        return page

    @staticmethod
    def _intro() -> QLabel:
        """``assets/intro.png`` when the build has one; a plain placeholder otherwise."""
        intro = QLabel()
        intro.setObjectName("intro")
        intro.setFixedSize(*INTRO_SIZE)
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        picture = QPixmap(str(ASSETS / "intro.png"))
        if picture.isNull():
            intro.setText(t("intro.placeholder"))
        else:
            picture.setDevicePixelRatio(picture.width() / INTRO_SIZE[0])
            intro.setPixmap(picture)
        return intro

    def _progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.setSpacing(12)
        self.progress_name = QLabel()
        self.progress_name.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.progress_stage = QLabel()
        self.progress_time = QLabel()
        self.progress_time.setObjectName("muted")
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        hint = QLabel(t("progress.hint"))
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        cancel = QPushButton(t("button.cancel"))
        cancel.clicked.connect(self.cancel)
        for w in (self.progress_name, self.progress_stage, bar, self.progress_time, hint):
            layout.addWidget(w)
        layout.addSpacing(8)
        layout.addWidget(cancel, 0, Qt.AlignmentFlag.AlignLeft)
        return page

    def _result_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        layout.addWidget(self.result_scroll, 1)
        buttons = QHBoxLayout()
        self.reveal = QPushButton(t("button.reveal"))
        self.reveal.setObjectName("primary")
        self.reveal.clicked.connect(lambda: self._open(self.output.parent))
        again = QPushButton(t("button.again"))
        again.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        buttons.addWidget(self.reveal)
        buttons.addStretch(1)
        buttons.addWidget(again)
        layout.addLayout(buttons)
        return page

    # ------------------------------------------------------------- conversion

    def convert(self, source: Path) -> None:
        self.output = free_output_path(source)
        self.output.with_name(self.output.name + ".report.json").unlink(missing_ok=True)  # never read a stale one
        args = ["convert", str(source), str(self.output), "--geometry-profile", "video-call"]  # waist up, as in calls
        if self.auto_face.isChecked():
            args += ["--auto-author", "--generate-face"]
        self.log = []
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._crashed)
        self.progress_name.setText(source.name)
        self.progress_stage.setText(t("stage.starting"))
        self.progress_time.setText("0:00")
        self.pages.setCurrentIndex(1)
        self.started = time.monotonic()
        self.ticker.start()
        program, argv = converter_command(args)
        self.process.start(program, argv)

    def cancel(self) -> None:
        if self.process:
            self.process.finished.disconnect(self._finished)
            self.process.kill()
            self.process.waitForFinished(3000)
        self.ticker.stop()
        self.pages.setCurrentIndex(0)

    def _tick(self) -> None:
        seconds = int(time.monotonic() - self.started)
        self.progress_time.setText(f"{seconds // 60}:{seconds % 60:02d}")

    def _read(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            self.log.append(line)
            stage = line.strip().removeprefix("@stage ")
            if stage in STAGES and line.startswith("@stage "):
                self.progress_stage.setText(t(f"stage.{stage}"))
        del self.log[:-400]

    def _crashed(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log.append(t("error.start", error=self.process.errorString()))
            self._finished()

    def _finished(self, *_) -> None:
        self.ticker.stop()
        report_path = self.output.with_name(self.output.name + ".report.json")
        report = None
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self._show(summarize(report, "\n".join(self.log)))

    # ----------------------------------------------------------------- result

    def _show(self, summary: Summary) -> None:
        holder = QWidget()
        self.result_box = QVBoxLayout(holder)
        self.result_box.setContentsMargins(0, 0, 8, 0)
        color = COLORS.get(summary.verdict, COLORS[WARN])
        head = QLabel(summary.headline)
        head.setWordWrap(True)
        head.setStyleSheet(f"font-size: 19px; font-weight: 600; color: {color};")
        self.result_box.addWidget(head)
        if summary.verdict != FAILED:
            where = QLabel(t("result.saved", name=self.output.name))
            where.setObjectName("muted")
            self.result_box.addWidget(where)
        if summary.advice:
            advice = QLabel(summary.advice)
            advice.setWordWrap(True)
            self.result_box.addWidget(advice)
        for problem in summary.problems:
            self.result_box.addWidget(self._row(BAD, "", problem))
        for check in summary.checks:
            self.result_box.addWidget(self._row(check.status, check.title, check.detail))
        self.result_box.addStretch(1)
        self.result_scroll.setWidget(holder)  # replaces (and deletes) the previous result
        self.reveal.setVisible(summary.verdict != FAILED)
        self.pages.setCurrentIndex(2)

    @staticmethod
    def _row(status: str, title: str, detail: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        row = QHBoxLayout(card)
        row.setContentsMargins(14, 10, 14, 10)
        mark = QLabel(MARKS[status])
        mark.setFixedWidth(22)
        mark.setStyleSheet(f"color: {COLORS[status]}; font-size: 16px; font-weight: 700; background: transparent;")
        text = QLabel(f"<b>{title}</b><br>{detail}" if title else detail)
        text.setWordWrap(True)
        text.setStyleSheet("background: transparent;")
        row.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(text, 1)
        return card

    @staticmethod
    def _open(path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Moviva FBX to VRM")
    app.setStyleSheet(STYLE)
    set_language(language_for(QLocale.system().name()))
    window = Window()
    window.show()
    return app.exec()
