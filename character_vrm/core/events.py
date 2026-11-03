"""Structured progress events (JSON lines, prefix ``@event ``) next to the legacy ``@stage`` lines.

Event types: ``step_started``, ``step_completed`` (status SUCCESS / WARNING / SKIPPED / FAILED), ``step_progress``
(only when the step knows its real current / total). Percentages are never invented.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Callable, Iterator

PREFIX = "@event "
Emit = Callable[[dict], None]


def encode(event: dict) -> str:
    return PREFIX + json.dumps(event, separators=(",", ":"))


def decode(line: str) -> dict | None:
    """The event in an output line, or None for any other line (including a malformed event)."""
    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX):])
    except ValueError:
        return None
    return event if isinstance(event, dict) and "type" in event else None


@contextmanager
def step(emit: Emit, step_id: str, message: str) -> Iterator[dict]:
    """``with step(emit, "auto_rig", "Generating skeleton") as done: ... done["status"] = "WARNING"``"""
    result = {"status": "SUCCESS", "detail": ""}
    started = time.monotonic()
    emit({"type": "step_started", "step": step_id, "message": message})
    try:
        yield result
    except BaseException as exc:
        emit({"type": "step_completed", "step": step_id, "status": "FAILED", "detail": str(exc)[:500],
              "seconds": round(time.monotonic() - started, 2)})
        raise
    emit({"type": "step_completed", "step": step_id, **result, "seconds": round(time.monotonic() - started, 2)})


def skipped(emit: Emit, step_id: str, reason: str) -> None:
    emit({"type": "step_completed", "step": step_id, "status": "SKIPPED", "detail": reason, "seconds": 0})
