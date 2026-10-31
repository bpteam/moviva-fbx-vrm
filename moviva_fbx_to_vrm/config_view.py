"""The configure screen (one card per processing step) and the live step list. Plans come from ``character_vrm.core.steps``."""

from __future__ import annotations

import json
from html import escape

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressDialog,
                               QToolButton, QVBoxLayout, QWidget)

from character_vrm.core import events
from character_vrm.core.steps import MANDATORY, Facts, PlannedStep, plan

from .i18n import t

STATUS_COLOR = {"RECOMMENDED": "#57d797", "OPTIONAL": "#a9bac8", "NOT_NEEDED": "#a9bac8",
                "UNAVAILABLE": "#ff6574", "EXPERIMENTAL": "#ffc053"}
RUN_MARK = {"PENDING": ("○", "#a9bac8"), "RUNNING": ("▶", "#ff7546"), "SUCCESS": ("✓", "#57d797"),
            "WARNING": ("!", "#ffc053"), "SKIPPED": ("–", "#a9bac8"), "FAILED": ("✕", "#ff6574")}


def parse_preflight(output: str) -> dict | None:
    """The ``@preflight {json}`` line of the converter's output, or None."""
    for line in output.splitlines():
        if line.startswith("@preflight "):
            try:
                return json.loads(line[len("@preflight "):])
            except ValueError:
                return None
    return None


def model_lines(model: dict) -> list[tuple[bool, str]]:
    """(ok, text) rows summarising the preflight, in the order the user cares about."""
    g = model["geometry"]
    rows = [(g["ok"] and g["confidence"] >= 0.35, t("analysis.geometry", confidence=f"{g['confidence']:.0%}")),
            (model["armatures"] > 0, t("analysis.skeleton")),
            (model["skinnedMeshes"] > 0, t("analysis.skin")),
            (model["shapeKeys"] > 0, t("analysis.morphs", n=model["shapeKeys"])),
            (bool(model["textures"]), t("analysis.textures", n=len(model["textures"])))]
    return rows


class StepCard(QFrame):
    toggled = Signal()

    def __init__(self, step: PlannedStep) -> None:
        super().__init__()
        self.setObjectName("card")
        s = step.spec
        self.box = QCheckBox(s.name)
        self.box.setChecked(step.enabled)
        self.box.setEnabled(step.status not in ("UNAVAILABLE", "NOT_NEEDED") and not step.blocked_by)
        self.box.toggled.connect(lambda _: self.toggled.emit())
        badge = QLabel(f'<span style="color:{STATUS_COLOR[step.status]}">{step.status.replace("_", " ")}</span>')
        self.more = QLabel(
            f"{escape(s.does)}<br><b>{t('card.hardware')}</b> {escape(s.hardware)} · <b>{t('card.software')}</b> "
            f"{escape(s.software)} · <b>{t('card.cost')}</b> {escape(s.cost)}<br><b>{t('card.limits')}</b> {escape(s.limits)}")
        self.more.setWordWrap(True)
        self.more.setVisible(False)
        self.more.setObjectName("muted")
        toggle = QToolButton()
        toggle.setText("…")
        toggle.clicked.connect(lambda: self.more.setVisible(not self.more.isVisible()))
        top = QHBoxLayout()
        top.addWidget(self.box, 1)
        top.addWidget(badge)
        top.addWidget(toggle)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        engine = QLabel(f"{escape(s.engine)} — {escape(step.why)}")
        engine.setObjectName("muted")
        engine.setWordWrap(True)
        lay.addWidget(engine)
        lay.addWidget(self.more)


class ConfigView(QWidget):
    """Re-plans on every toggle: a step switched off stays off, and what depends on it is blocked with a reason."""

    def __init__(self) -> None:
        super().__init__()
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.facts = Facts()
        self.selected: dict[str, bool] = {}
        self.steps: list[PlannedStep] = []
        self.profile = "video-call"  # what Moviva shows; kept across models in one session

    def load(self, document: dict) -> None:
        from character_vrm.ai import skintokens
        st = skintokens.status()
        self.facts = Facts(**{**document["facts"], "ai_rig": st["state"], "ai_rig_reason": st.get("reason", "")})
        self.model = document["model"]
        self.selected = {}
        self._render()

    def _render(self) -> None:
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.steps = plan(self.facts, self.selected or None)
        head = QLabel(f"<b>{escape(self.model['file']['name'])}</b>")
        self.layout_.addWidget(head)
        for ok, text in model_lines(self.model):
            row = QLabel(f'<span style="color:{"#57d797" if ok else "#ff6574"}">{"✓" if ok else "✕"}</span> {escape(text)}')
            self.layout_.addWidget(row)
        for problem in self.model["problems"]:
            note = QLabel(escape(problem))
            note.setObjectName("muted")
            note.setWordWrap(True)
            self.layout_.addWidget(note)
        self.layout_.addWidget(self._profile_row())
        for step in self.steps:
            card = StepCard(step)
            card.toggled.connect(lambda c=card, sid=step.spec.id: self._changed(sid, c.box.isChecked()))
            self.layout_.addWidget(card)
        self.layout_.addStretch(1)

    def _profile_row(self) -> QFrame:
        """Visible part: the converter's --geometry-profile, with what it removes in the tooltip."""
        row = QFrame()
        row.setObjectName("card")
        lay = QHBoxLayout(row)
        lay.addWidget(QLabel(t("geometry.title")))
        info = QLabel("ⓘ")
        info.setToolTip(f"<p>{escape(t('geometry.help'))}</p>")
        lay.addWidget(info)
        lay.addStretch(1)
        combo = QComboBox()
        for key in ("video-call", "face", "full"):
            combo.addItem(t(f"geometry.{key}"), key)
        combo.setCurrentIndex(combo.findData(self.profile))
        combo.currentIndexChanged.connect(lambda _: setattr(self, "profile", combo.currentData()))
        lay.addWidget(combo)
        return row

    def _changed(self, step_id: str, on: bool) -> None:
        self.selected = {s.spec.id: s.enabled for s in self.steps}
        if step_id == "ai_rigging" and on and self.facts.ai_rig != "READY":
            on = self._install_ai()  # stays off unless the download was approved and finished
        self.selected[step_id] = on
        self._render()

    def _install_ai(self) -> bool:
        """Ask, then run ``<app> ai install`` with its output as progress. Nothing is downloaded without a yes."""
        from character_vrm.ai import skintokens
        gpu = skintokens.status().get("gpu", {})
        ok = QMessageBox.question(self, "Install AI rigging components?", (
            "Engine: SkinTokens / TokenRig (MIT)\n"
            f"GPU: {gpu.get('name', '?')} ({gpu.get('mib', 0) // 1024} GB)\n\n"
            "Downloads once: its source, Python 3.11, PyTorch with CUDA libraries, flash-attn and the model weights "
            "(about 12 GB on disk, into " f"{skintokens.home()}).\n\n"
            "All processing stays on this computer. The components are kept for later conversions."))
        if ok != QMessageBox.StandardButton.Yes:
            return False
        from .gui import converter_command
        program, args = converter_command(["ai", "install"])
        dialog = QProgressDialog("Installing SkinTokens…", "Cancel", 0, 0, self)
        dialog.setWindowTitle("Installing SkinTokens")
        dialog.setMinimumWidth(460)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        def show() -> None:  # the installer's last line ("Downloading model weights", pip's progress)
            lines = [x for x in bytes(proc.readAllStandardOutput()).decode("utf-8", "replace").splitlines() if x.strip()]
            if lines:
                dialog.setLabelText(lines[-1][-120:])
        proc.readyReadStandardOutput.connect(show)
        proc.finished.connect(dialog.close)
        dialog.canceled.connect(proc.kill)  # a rerun resumes: finished steps are skipped
        proc.start(program, args)
        dialog.exec()
        proc.waitForFinished(-1)
        st = skintokens.status()
        self.facts.ai_rig, self.facts.ai_rig_reason = st["state"], st.get("reason", "")
        if st["state"] != "READY":
            QMessageBox.warning(self, "SkinTokens", "The installation did not finish. Switch the step on again to "
                                "resume it.")
        return st["state"] == "READY"


class StepList(QWidget):
    """The live per-step list; rows are the mandatory stages plus the steps the plan enabled."""

    def __init__(self) -> None:
        super().__init__()
        self.lay = QVBoxLayout(self)
        self.rows: dict[str, tuple[QLabel, str]] = {}

    def reset(self, plan_steps: list[PlannedStep]) -> None:
        while self.lay.count():
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        names = dict(MANDATORY)
        names["auto_rig"] = "Generate skeleton and skin weights"
        names["authoring"] = "Generate expressions, hair and materials"
        order = ["import", "auto_rig", "map", "authoring", "export", "validate"]
        self.rows = {}
        for sid in order:
            self.rows[sid] = (QLabel(), names[sid])
            self.lay.addWidget(self.rows[sid][0])
            self._set(sid, "PENDING", "")

    def _set(self, sid: str, status: str, detail: str) -> None:
        label, name = self.rows[sid]
        mark, color = RUN_MARK[status]
        extra = f' <span style="color:#a9bac8">{escape(detail)}</span>' if detail else ""
        label.setText(f'<span style="color:{color}">{mark}</span> {escape(name)}{extra}')

    def feed(self, line: str) -> None:
        event = events.decode(line.strip())
        if not event or event.get("step") not in self.rows:
            return
        sid = event["step"]
        if event["type"] == "step_started":
            self._set(sid, "RUNNING", "")
        elif event["type"] == "step_completed":
            secs = event.get("seconds", 0)
            detail = " · ".join(x for x in (event.get("detail", ""), f"{secs:.1f}s" if secs else "") if x)
            self._set(sid, event.get("status", "SUCCESS"), detail)
