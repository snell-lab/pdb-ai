"""Printing pdbai's log messages to standard error, for the CLI and web app.

The handler goes on the "pdbai" logger. It therefore prints the messages of
every logger beneath it, such as the network progress logger "pdbai.network"
and the web app's logger. Third-party loggers and the root logger are left
alone. Nothing is added to them, and their levels stay as they are. pdbai's
messages still propagate to the root, so a host that logs through the root
sees them too.
"""

from __future__ import annotations

import logging
import sys


class _StderrHandler(logging.Handler):
    """A handler that prints each message to the current sys.stderr.

    A StreamHandler keeps the sys.stderr it was created with. pytest swaps
    sys.stderr during each test and closes the stand-in afterwards. A
    StreamHandler created in one test would then fail in every later one.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


class _ConsoleFormatter(logging.Formatter):
    """A formatter that starts every line with "[pdbai]" and labels warnings.

    From WARNING up, the first line also names the level. Progress messages
    are the normal flow and stay unlabelled, so a warning or an error stands
    out. Every line carries the prefix, including each line of a traceback.
    pdbai's output then stays recognisable when other text on standard error
    interleaves with it, such as the web server's request log.
    """

    def format(self, record: logging.LogRecord) -> str:
        # An empty message still prints one prefixed line rather than none.
        first, *rest = super().format(record).splitlines() or [""]
        if record.levelno >= logging.WARNING:
            first = f"{record.levelname}: {first}"
        return "\n".join(f"[pdbai] {line}" for line in (first, *rest))


def configure_console_logging() -> None:
    """Print pdbai's log messages to standard error from now on.

    The "pdbai" logger is set to INFO unless a level was already set on it.
    A second call does nothing, so each entry point can call this on start
    without checking.
    """
    package_logger = logging.getLogger("pdbai")
    if any(isinstance(handler, _StderrHandler) for handler in package_logger.handlers):
        return
    # A level set earlier is the host's choice, such as DEBUG for a closer
    # look. Overriding it would hide messages the host asked for, or flood it
    # with ones it did not.
    if package_logger.level == logging.NOTSET:
        package_logger.setLevel(logging.INFO)
    handler = _StderrHandler()
    handler.setFormatter(_ConsoleFormatter())
    package_logger.addHandler(handler)
