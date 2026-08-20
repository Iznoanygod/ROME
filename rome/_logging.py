"""ROME-A logging — one line per lifecycle event, styled to match IMPRESS.

ROME-A schedules training out of sight of the host workflow, so a campaign
operator needs to *see* what it is doing: when a design arrives, when a round is
submitted, when a new model is published. This wires a single stdout handler on
the ``rome`` logger namespace whose output mirrors IMPRESS's
``impress.utils.logger.ImpressLogger`` — same ``HH:MM:SS.mmm [LEVEL]
[COMPONENT]`` shape and the same per-level / per-component colours — so ROME-A's
lines sit alongside IMPRESS's ``[PIPELINE-P1]`` lines in one run::

    12:34:56.789 [INFO] [ROME-DATA] received design 8oep (pLDDT=95.0) — corpus 8
    12:34:57.001 [INFO] [ROME-TRAINER] submitting training round 1 (8 designs)
    12:34:58.512 [INFO] [ROME-MODEL] published v1 -> .../v_48_020.pt

It matches IMPRESS's *format* without importing IMPRESS, so ROME-A keeps working
in workflows that have nothing to do with it (the LLM/GRPO trainer, the dummy
loop). The handler is attached once, lazily, and only if the ``rome`` logger has
none, so an application that configures its own logging keeps control.

Environment:

* ``ROME_LOG_LEVEL`` — ``INFO`` (default), ``DEBUG`` for per-record detail, or
  ``WARNING`` to quiet the lifecycle lines.
* ``ROME_LOG_COLOR`` — ``0`` disables ANSI colour (also disabled when ``NO_COLOR``
  is set). Default on, like IMPRESS, since Dragon captures a non-tty stdout.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

_ROOT = "rome"
_configured = False


# IMPRESS's palette (impress/utils/logger.py), so the two logs look like one.
class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


_LEVEL_COLOR = {
    "DEBUG": _C.BRIGHT_BLACK,
    "INFO": _C.BRIGHT_CYAN,
    "WARNING": _C.BRIGHT_YELLOW,
    "ERROR": _C.BRIGHT_RED,
    "CRITICAL": _C.RED + _C.BOLD,
}

#: Component (the part after ``rome.``) -> colour. Falls back to bright white.
_COMPONENT_COLOR = {
    "DATA": _C.BRIGHT_YELLOW,        # matches IMPRESS "data"
    "TRAINER": _C.BRIGHT_MAGENTA,
    "MODEL": _C.BRIGHT_GREEN,        # matches IMPRESS "checkpoint"
    "STREAM": _C.BRIGHT_CYAN,
    "MANAGER": _C.BRIGHT_WHITE,
}


class _ImpressStyleFormatter(logging.Formatter):
    """Format a record as ``HH:MM:SS.mmm [LEVEL] [ROME-<COMPONENT>] message``."""

    def __init__(self, use_colors: bool):
        super().__init__()
        self.use_colors = use_colors

    def _paint(self, text: str, color: str) -> str:
        return f"{color}{text}{_C.RESET}" if self.use_colors else text

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        # component = the segment after "rome." (rome.trainer -> TRAINER)
        tail = record.name.split(".", 1)[1] if "." in record.name else "core"
        component = f"ROME-{tail.upper()}"
        comp_color = _COMPONENT_COLOR.get(tail.upper(), _C.BRIGHT_WHITE)
        return (
            f"{self._paint(ts, _C.DIM)} "
            f"{self._paint(f'[{record.levelname}]', _LEVEL_COLOR.get(record.levelname, ''))} "
            f"{self._paint(f'[{component}]', comp_color)} "
            f"{record.getMessage()}"
        )


def _use_colors() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return os.environ.get("ROME_LOG_COLOR", "1") != "0"


def _configure() -> None:
    global _configured
    logger = logging.getLogger(_ROOT)
    level = os.environ.get("ROME_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ImpressStyleFormatter(_use_colors()))
        logger.addHandler(handler)
        logger.propagate = False        # own the output; do not double-print
    _configured = True


def get_logger(name: str = _ROOT) -> logging.Logger:
    """A logger under the ``rome`` namespace, with the shared handler attached.

    ``name`` is usually ``__name__`` (``rome.trainer`` -> the ``[ROME-TRAINER]``
    tag). A bare label is placed under ``rome.`` too, so one level and one
    handler govern all of ROME-A's logging.
    """
    if not _configured:
        _configure()
    if name == _ROOT or name.startswith(_ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")
