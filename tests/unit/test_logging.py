"""ROME-A's IMPRESS-style logging: format shape and lifecycle events.

The logger exists so a campaign operator can *see* ROME-A working alongside
IMPRESS. Two things must hold: the line shape matches IMPRESS's
``HH:MM:SS.mmm [LEVEL] [COMPONENT]``, and the events the user asked for — a
design received, a round submitted, a model published — actually reach the log.
"""

import logging
import re

import pytest

from rome._logging import _ImpressStyleFormatter, get_logger


def _record(name, level, msg):
    return logging.LogRecord(name, level, __file__, 1, msg, (), None)


def test_format_matches_impress_shape_without_color():
    fmt = _ImpressStyleFormatter(use_colors=False)
    line = fmt.format(_record("rome.data", logging.INFO, "received design abcd"))
    # HH:MM:SS.mmm [INFO] [ROME-DATA] received design abcd
    assert re.match(
        r"^\d{2}:\d{2}:\d{2}\.\d{3} \[INFO\] \[ROME-DATA\] received design abcd$",
        line,
    ), line


def test_component_tag_is_derived_from_logger_name():
    fmt = _ImpressStyleFormatter(use_colors=False)
    assert "[ROME-TRAINER]" in fmt.format(
        _record("rome.trainer", logging.INFO, "x"))
    assert "[ROME-MODEL]" in fmt.format(
        _record("rome.model", logging.INFO, "x"))


def test_colors_wrap_the_message_when_enabled():
    fmt = _ImpressStyleFormatter(use_colors=True)
    line = fmt.format(_record("rome.model", logging.INFO, "published v1"))
    assert "\033[" in line and line.endswith("published v1")


def test_get_logger_places_bare_names_under_rome():
    assert get_logger("stream").name == "rome.stream"
    assert get_logger("rome.stream").name == "rome.stream"
    assert get_logger().name == "rome"


def test_data_manager_logs_received_and_rejected(namespace, caplog):
    from rome.data import DataConfig, DataManager

    mgr = DataManager(namespace, DataConfig(min_score=50.0, score_key="score"))

    # The rome logger has propagate=False (it owns its handler), so caplog's
    # root handler never sees it — attach caplog's handler to rome directly.
    rome_logger = logging.getLogger("rome")
    rome_logger.addHandler(caplog.handler)
    old_level = rome_logger.level
    rome_logger.setLevel(logging.DEBUG)
    try:
        assert mgr.add(score=95.0, uid="keep0001") is not None
        assert mgr.add(score=10.0, uid="drop0001") is None   # below min_score
    finally:
        rome_logger.removeHandler(caplog.handler)
        rome_logger.setLevel(old_level)

    text = caplog.text
    assert "received design keep0001" in text
    assert "rejected design drop0001" in text
