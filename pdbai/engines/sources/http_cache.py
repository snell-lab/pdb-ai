"""The on-disk cache every source engine reads its responses through.

A JSON response is stored in an envelope. The envelope is a JSON object that
holds the response and the time it was cached. A downloaded file, such as an
mmCIF file, is stored as downloaded, and its modification time dates it.

Both are reused for CACHE_TTL_SECONDS and then fetched again, because
entries are revised after deposition. Citations, refinement statistics, and
remediated or re-refined coordinates all change.

A cache file that cannot be read counts as missing, so a damaged cache costs
a download rather than a run. Writes are atomic, so a reader never meets a
half-written file. A downloaded file must pass the check of its FileFormat
before it is written. An mmCIF file must open a data block, and an MTZ or
gzip file must start with its signature. An error page served with status
200 therefore never replaces a good file, and a cached file is judged by the
same check before it is reused. When a source answers that it no longer has
a file, the expired copy is removed, so no receipt names it as the file the
run used.

The caller chooses each cache file's path, and the path is the cache key.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import requests

from ...atomic import write_bytes_atomically, write_text_atomically
from ...json_values import optional_object_dict
from ...network import network_logger

CACHE_TTL_SECONDS = 30 * 24 * 60 * 60


def _is_fresh(timestamp: float) -> bool:
    # A time in the future means the clock or the file is wrong, so it does
    # not count as fresh.
    return 0.0 <= time.time() - timestamp <= CACHE_TTL_SECONDS


def read_envelope(path: Path) -> dict[str, Any] | None:
    """Return a cached JSON envelope, or None when it must be fetched again.

    None stands for a file that is missing, unreadable, not a JSON object,
    without a time it was cached, or older than CACHE_TTL_SECONDS.
    """
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        # ValueError covers text that is not UTF-8 or not JSON, and
        # RecursionError a document nested too deeply to parse.
        return None
    envelope = optional_object_dict(value)
    if envelope is None:
        return None
    cached_at = envelope.get("cached_at")
    if isinstance(cached_at, bool) or not isinstance(cached_at, (int, float)):
        return None
    if not _is_fresh(float(cached_at)):
        return None
    return envelope


def write_envelope(path: Path, payload: Mapping[str, object]) -> None:
    """Write a payload to the cache with the time it was cached.

    The payload's keys sit beside cached_at. A value JSON cannot hold is
    written as its text. A failed write does not raise, since the caller
    already has the payload. It is logged as a warning instead. A cache that
    never writes makes every run fetch again, and the warning keeps that
    from going unnoticed.
    """
    envelope: dict[str, object] = {"cached_at": time.time(), **payload}
    try:
        write_text_atomically(path, json.dumps(envelope, default=str, sort_keys=True))
    except Exception as exception:
        network_logger.warning("Could not write cache file %s: %s", path, exception)


def _stream_lines(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Yield the lines of each chunk, split at CR and CRLF as well as LF."""
    for chunk in chunks:
        yield from chunk.splitlines()


def _opens_mmcif_block(stream: BinaryIO) -> bool:
    """Return whether the first line with content opens an mmCIF data block.

    Blank lines and comment lines before it are skipped. Lines are read
    lazily up to the first one with content, since a structure can be
    hundreds of megabytes.
    """
    for line in _stream_lines(stream):
        stripped = line.strip()
        if not stripped or stripped.startswith(b"#"):
            continue
        return stripped.lower().startswith(b"data_")
    return False


def _starts_with(signature: bytes) -> Callable[[BinaryIO], bool]:
    """Return a check that a stream starts with ``signature``."""

    def check(stream: BinaryIO) -> bool:
        return stream.read(len(signature)) == signature

    return check


@dataclass(frozen=True)
class FileFormat:
    """A kind of file a source serves, and the check its content must pass.

    ``name`` names the file in log lines and in the message that a source
    has none. ``mismatch`` ends the message about a file that fails the
    check, as in "response is not an MTZ reflection file". ``check`` reads a
    file from its start and says whether it is of this kind.
    """

    name: str
    mismatch: str
    check: Callable[[BinaryIO], bool]

    def matches(self, path: Path) -> bool:
        """Return whether the file at ``path`` passes the check.

        A file that cannot be read does not.
        """
        try:
            with path.open("rb") as handle:
                return self.check(handle)
        except OSError:
            return False


MMCIF_FILE = FileFormat(
    "mmCIF", "does not contain an mmCIF data block", _opens_mmcif_block
)
# Every MTZ reflection file starts with the four bytes "MTZ ", and every gzip
# file with the two bytes 1f 8b.
MTZ_FILE = FileFormat("MTZ", "is not an MTZ reflection file", _starts_with(b"MTZ "))
GZIP_FILE = FileFormat("gzip file", "is not a gzip file", _starts_with(b"\x1f\x8b"))


def _is_fresh_file(path: Path, file_format: FileFormat) -> bool:
    """Return whether a cached file is recent and passes its format's check.

    A file that cannot be read is not fresh.
    """
    try:
        return (
            path.is_file()
            and _is_fresh(path.stat().st_mtime)
            and file_format.matches(path)
        )
    except OSError:
        return False


def _reports_missing_file(response: requests.Response) -> bool:
    """Return whether a response says the source has no file at the URL.

    A 404 or 410 says so. PDB-Redo instead answers status 500 with the JSON
    body {"error": "Not Found"}, which an outage never sends. The retry
    session has already retried that 500, since urllib3 decides on the
    status alone.
    """
    if response.status_code in {404, 410}:
        return True
    if response.status_code != 500:
        return False
    try:
        body: object = json.loads(response.content)
    except (ValueError, RecursionError):
        return False
    error = optional_object_dict(body)
    return error is not None and error.get("error") == "Not Found"


class NotAvailableError(RuntimeError):
    """The error a source answers with when it has no file for an entry.

    It stands for an expected outcome, such as an entry PDB-Redo has not
    re-refined, rather than a failure.
    """


def fetch_file(
    url: str,
    path: Path,
    file_format: FileFormat,
    *,
    session: requests.Session,
    timeout: float,
    label: str,
    missing_statuses: frozenset[int] = frozenset(),
) -> bool:
    """Ensure a current file is at ``path``; return whether it was cached.

    The file at ``path`` is reused while it is recent and passes the check
    of ``file_format``, and is otherwise downloaded from ``url`` and written
    there. ``timeout`` is in seconds. An answer that the source has no such
    file, a 404, a 410, or PDB-Redo's 500 with a "Not Found" body, raises
    NotAvailableError, and any file left at ``path`` is removed first, since
    the source no longer stands behind it. A status in ``missing_statuses``
    counts as that answer too, for an archive that answers 403 where it has
    no file. Any other failed request or write raises RuntimeError. So does
    a response that fails the check, and the cached file then stays as it
    was. ``label`` names the source and entry in log lines and error
    messages.
    """
    if _is_fresh_file(path, file_format):
        network_logger.info("Using cached %s %s at %s", label, file_format.name, path)
        return True
    network_logger.info("Requesting %s %s from %s", label, file_format.name, url)
    try:
        response = session.get(url, timeout=timeout)
    except Exception as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    if response.status_code in missing_statuses or _reports_missing_file(response):
        try:
            path.unlink(missing_ok=True)
        except OSError as exception:
            network_logger.warning(
                "Could not remove cache file %s: %s", path, exception
            )
        raise NotAvailableError(f"{label} has no {file_format.name} at {url}")
    try:
        response.raise_for_status()
    except Exception as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    if not file_format.check(io.BytesIO(response.content)):
        raise RuntimeError(f"{label} download failed: response {file_format.mismatch}")
    try:
        write_bytes_atomically(path, response.content)
    except OSError as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    return False


__all__ = [
    "CACHE_TTL_SECONDS",
    "GZIP_FILE",
    "MMCIF_FILE",
    "MTZ_FILE",
    "FileFormat",
    "NotAvailableError",
    "fetch_file",
    "read_envelope",
    "write_envelope",
]
