from __future__ import annotations

import io
import logging
import sys
from collections.abc import Iterator

import pytest

from pdbai.console_logging import configure_console_logging


@pytest.fixture
def package_logger() -> Iterator[logging.Logger]:
    # The CLI tests install the handler for good, so each test here starts
    # from a bare "pdbai" logger and gets the earlier state back afterwards.
    logger = logging.getLogger("pdbai")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    for handler in saved_handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    yield logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    for handler in saved_handlers:
        logger.addHandler(handler)
    logger.setLevel(saved_level)


def test_info_messages_print_to_stderr_with_the_prefix(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_console_logging()

    package_logger.info("searching %d entries", 3)

    captured = capsys.readouterr()
    assert captured.err == "[pdbai] searching 3 entries\n"
    assert captured.out == ""


def test_messages_of_child_loggers_print_too(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_console_logging()

    logging.getLogger("pdbai.network").info("cache hit")

    assert capsys.readouterr().err == "[pdbai] cache hit\n"


def test_debug_messages_do_not_print(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_console_logging()

    package_logger.debug("detail")

    assert package_logger.level == logging.INFO
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("level", "name"),
    [(logging.WARNING, "WARNING"), (logging.ERROR, "ERROR")],
)
def test_warnings_and_above_name_the_level(
    package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
    level: int,
    name: str,
) -> None:
    configure_console_logging()

    package_logger.log(level, "entry skipped")

    assert capsys.readouterr().err == f"[pdbai] {name}: entry skipped\n"


def test_a_level_set_by_the_host_is_kept(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    package_logger.setLevel(logging.DEBUG)

    configure_console_logging()
    package_logger.debug("detail")

    assert package_logger.level == logging.DEBUG
    assert capsys.readouterr().err == "[pdbai] detail\n"


def test_third_party_and_root_loggers_are_left_alone(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    third_party = logging.getLogger("urllib3")
    root_handlers, root_level = root.handlers[:], root.level
    third_party_handlers = third_party.handlers[:]
    third_party_level = third_party.level

    configure_console_logging()
    third_party.warning("connection reset")

    assert root.handlers == root_handlers
    assert root.level == root_level
    assert third_party.handlers == third_party_handlers
    assert third_party.level == third_party_level
    assert "[pdbai]" not in capsys.readouterr().err


def test_messages_still_propagate_to_the_root(
    package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_console_logging()

    package_logger.info("searching")

    assert caplog.messages == ["searching"]
    assert capsys.readouterr().err == "[pdbai] searching\n"


def test_output_follows_sys_stderr_swapped_after_install(
    package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_console_logging()
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stderr", replacement)

    package_logger.info("searching")

    assert replacement.getvalue() == "[pdbai] searching\n"
    monkeypatch.undo()
    assert capsys.readouterr().err == ""


def test_installing_twice_prints_each_message_once(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_console_logging()
    configure_console_logging()

    package_logger.info("searching")

    assert len(package_logger.handlers) == 1
    assert capsys.readouterr().err == "[pdbai] searching\n"


def test_a_logged_traceback_prints_with_the_prefix_on_every_line(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_console_logging()

    try:
        raise ValueError("bad entry")
    except ValueError:
        package_logger.exception("entry failed")

    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "[pdbai] ERROR: entry failed"
    assert lines[1] == "[pdbai] Traceback (most recent call last):"
    assert lines[-1] == "[pdbai] ValueError: bad entry"
    assert all(line.startswith("[pdbai] ") for line in lines)


def test_an_empty_message_prints_one_prefixed_line(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_console_logging()

    package_logger.info("")

    assert capsys.readouterr().err == "[pdbai] \n"
