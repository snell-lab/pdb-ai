"""Writing a file so that no reader ever sees it half-written.

Every file a run reads back whole is written through this module. That
includes the run's manifest and partial CSVs, the cached downloads, the
assembled and published datasets, and the reports. The run's entry and event
logs are the exception. They are appended to in place, and their readers skip
a line that is not yet complete.

Each write goes to a scratch file beside its target, and a rename then puts
it in place. The scratch file sits beside the target so that the rename stays
within one file system, where it is atomic. The scratch file is flushed to
disk before the rename. So a crash just after the rename cannot leave the
target empty or truncated.

Windows also refuses a rename onto a file that another rename is still
replacing. Threads of one process that write the same file therefore take
turns at the rename. Each path maps to one of a fixed set of locks, so the
locks take the same memory however many files a run writes.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

# Windows refuses a replacement while another process has either file open.
# Virus scanners, search indexers, and concurrent readers do that briefly. So
# a replacement refused with PermissionError is retried after each of these
# delays, and the error of the last attempt is raised.
_REPLACE_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)

_SCRATCH_NAME = ".{name}.{unique}.tmp"
# How many bytes longer a scratch file's name is than its target's. A caller
# that limits the length of a name must leave this much free. The unique part
# is the 32 hexadecimal digits of a UUID.
SCRATCH_NAME_EXTRA_BYTES = len(_SCRATCH_NAME.format(name="", unique="")) + 32


# Two paths that share a lock only wait for each other's renames, which take
# milliseconds, so a modest number of locks is enough.
_REPLACE_LOCKS = tuple(threading.Lock() for _ in range(64))


def _replace_lock(target: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(target))
    return _REPLACE_LOCKS[hash(key) % len(_REPLACE_LOCKS)]


def _replace_retrying(source: Path, target: Path) -> None:
    with _replace_lock(target):
        _replace_with_retries(source, target)


def _replace_with_retries(source: Path, target: Path) -> None:
    for delay in _REPLACE_RETRY_DELAYS_SECONDS:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(source, target)


@contextmanager
def write_atomically(path: Path) -> Generator[Path]:
    """Yield an empty scratch file that replaces ``path`` when the block ends.

    The caller writes the new content to the scratch file and closes it within
    the block. Missing parent directories are created. If the block raises, the
    scratch file is removed and ``path`` is left as it was. Readers of ``path``
    see the previous file or the complete new one, even after a crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(
        _SCRATCH_NAME.format(name=path.name, unique=uuid.uuid4().hex)
    )
    # The scratch file becomes the new file. So it is created with the usual
    # permissions, not the owner-only ones that tempfile.mkstemp gives.
    scratch.touch(exist_ok=False)
    try:
        yield scratch
        # Windows flushes a file to disk only through a handle open for
        # writing.
        with scratch.open("rb+") as stream:
            os.fsync(stream.fileno())
        _replace_retrying(scratch, path)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise


def write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, as write_atomically does."""
    with write_atomically(path) as scratch:
        scratch.write_text(text, encoding="utf-8")


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, as write_atomically does."""
    with write_atomically(path) as scratch:
        scratch.write_bytes(data)


__all__ = [
    "SCRATCH_NAME_EXTRA_BYTES",
    "write_atomically",
    "write_bytes_atomically",
    "write_text_atomically",
]
