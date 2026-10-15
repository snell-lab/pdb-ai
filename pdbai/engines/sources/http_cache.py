"""The on-disk cache every source engine reads its responses through.

A JSON response is stored in an envelope. The envelope is a JSON object that
holds the response and the time it was cached. An mmCIF file is stored as
downloaded, and its modification time dates it.

Both are reused for CACHE_TTL_SECONDS and then fetched again, because
entries are revised after deposition. Citations, refinement statistics, and
remediated or re-refined coordinates all change.

A cache file that cannot be read counts as missing, so a damaged cache costs
a download rather than a run. Writes are atomic, so a reader never meets a
half-written file. A downloaded mmCIF file must open a data block before it
is written, and a downloaded MTZ or gzip file must start with its signature, so
an error page served with status 200 never replaces a good file. When a
source answers that it no longer has a file, the expired copy is removed, so
no receipt names it as the file the run used.

The caller chooses each cache file's path, and the path is the cache key.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

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

    The payload's keys sit beside cached_at. A payload key named cached_at
    raises ValueError, since it would replace the envelope's own value. A value
    JSON cannot hold is written as its text. A failed write does not raise,
    since the caller already has the payload. It is logged as a warning
    instead. A cache that never writes makes every run fetch again, and the
    warning keeps that from going unnoticed.
    """
    if "cached_at" in payload:
        raise ValueError("Cache payload key 'cached_at' is reserved for the envelope.")
    envelope: dict[str, object] = {"cached_at": time.time(), **payload}
    try:
        write_text_atomically(path, json.dumps(envelope, default=str, sort_keys=True))
    except Exception as exception:
        network_logger.warning("Could not write cache file %s: %s", path, exception)


def _starts_mmcif_block(lines: Iterable[bytes]) -> bool:
    """Return whether the first line with content opens an mmCIF data block.

    Blank lines and comment lines before it are skipped.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(b"#"):
            continue
        return stripped.lower().startswith(b"data_")
    return False


def _stream_lines(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Yield the lines of each chunk, split at CR and CRLF as well as LF."""
    for chunk in chunks:
        yield from chunk.splitlines()


def is_fresh_mmcif(path: Path) -> bool:
    """Return whether a cached mmCIF file is recent and opens a data block.

    A file that cannot be read is not fresh.
    """
    try:
        if not path.is_file() or not _is_fresh(path.stat().st_mtime):
            return False
        # Lines are read lazily up to the first one with content, since a
        # cached structure can be hundreds of megabytes.
        with path.open("rb") as handle:
            return _starts_mmcif_block(_stream_lines(handle))
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


def fetch_mmcif(
    url: str,
    path: Path,
    *,
    session: requests.Session,
    timeout: float,
    label: str,
) -> bool:
    """Ensure a current mmCIF file is at ``path``; return whether it was cached.

    The file at ``path`` is reused while is_fresh_mmcif accepts it, and is
    otherwise downloaded from ``url`` and written there. ``timeout`` is in
    seconds. An answer that the source has no such file, a 404, a 410, or
    PDB-Redo's 500 with a "Not Found" body, raises NotAvailableError, and
    any file left at ``path`` is removed first, since the source no longer
    stands behind it. Any other failed request or write raises RuntimeError. So does a response that does
    not open an mmCIF data block, and the cached file then stays as it was.
    ``label`` names the source and entry in log lines and error messages.
    """
    if is_fresh_mmcif(path):
        network_logger.info("Using cached %s mmCIF at %s", label, path)
        return True
    network_logger.info("Requesting %s mmCIF from %s", label, url)
    try:
        response = session.get(url, timeout=timeout)
    except Exception as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    if _reports_missing_file(response):
        try:
            path.unlink(missing_ok=True)
        except OSError as exception:
            network_logger.warning(
                "Could not remove cache file %s: %s", path, exception
            )
        raise NotAvailableError(f"{label} has no mmCIF at {url}")
    try:
        response.raise_for_status()
    except Exception as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    # The response is read line by line, so a large file is not split whole
    # just to find its first line.
    if not _starts_mmcif_block(_stream_lines(io.BytesIO(response.content))):
        raise RuntimeError(
            f"{label} download failed: response does not contain an mmCIF data block"
        )
    try:
        write_bytes_atomically(path, response.content)
    except OSError as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    return False


# The four bytes every MTZ reflection file starts with, and the two every
# gzip file starts with.
MTZ_SIGNATURE = b"MTZ "
GZIP_SIGNATURE = b"\x1f\x8b"


def _starts_with(path: Path, signature: bytes) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(signature)) == signature
    except OSError:
        return False


def is_mtz_file(path: Path) -> bool:
    """Return whether the file at ``path`` starts with the MTZ signature.

    A file that cannot be read does not.
    """
    return _starts_with(path, MTZ_SIGNATURE)


def is_gzip_file(path: Path) -> bool:
    """Return whether the file at ``path`` starts with the gzip signature.

    A file that cannot be read does not.
    """
    return _starts_with(path, GZIP_SIGNATURE)


def _is_fresh_signed(path: Path, signature: bytes) -> bool:
    try:
        return (
            path.is_file()
            and _is_fresh(path.stat().st_mtime)
            and _starts_with(path, signature)
        )
    except OSError:
        return False


def is_fresh_mtz(path: Path) -> bool:
    """Return whether a cached MTZ file is recent and starts as one.

    A file that cannot be read is not fresh.
    """
    return _is_fresh_signed(path, MTZ_SIGNATURE)


def is_fresh_gzip(path: Path) -> bool:
    """Return whether a cached gzip file is recent and starts as one.

    A file that cannot be read is not fresh.
    """
    return _is_fresh_signed(path, GZIP_SIGNATURE)


def _fetch_signed(
    url: str,
    path: Path,
    *,
    session: requests.Session,
    timeout: float,
    label: str,
    signature: bytes,
    kind: str,
    description: str,
    missing_statuses: frozenset[int] = frozenset(),
) -> bool:
    """Ensure a current file that starts with ``signature`` is at ``path``.

    ``kind`` names the file in log lines and in the message that the source
    has none, and ``description`` completes "response is not". A status in
    ``missing_statuses`` counts as the source having no such file, beside
    the answers _reports_missing_file knows.
    """
    if _is_fresh_signed(path, signature):
        network_logger.info("Using cached %s %s at %s", label, kind, path)
        return True
    network_logger.info("Requesting %s %s from %s", label, kind, url)
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
        raise NotAvailableError(f"{label} has no {kind} at {url}")
    try:
        response.raise_for_status()
    except Exception as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    if not response.content.startswith(signature):
        raise RuntimeError(f"{label} download failed: response is not {description}")
    try:
        write_bytes_atomically(path, response.content)
    except OSError as exception:
        raise RuntimeError(f"{label} download failed: {exception}") from exception
    return False


def fetch_mtz(
    url: str,
    path: Path,
    *,
    session: requests.Session,
    timeout: float,
    label: str,
) -> bool:
    """Ensure a current MTZ file is at ``path``; return whether it was cached.

    The rules are those of fetch_mmcif, with the MTZ signature in place of
    the data block. The file at ``path`` is reused while is_fresh_mtz accepts
    it, and is otherwise downloaded from ``url``. An answer that the source
    has no such file raises NotAvailableError, and any file left at ``path``
    is removed first. Any other failed request or write raises RuntimeError.
    So does a response that does not start with the MTZ signature, and the
    cached file then stays as it was.
    """
    return _fetch_signed(
        url,
        path,
        session=session,
        timeout=timeout,
        label=label,
        signature=MTZ_SIGNATURE,
        kind="MTZ",
        description="an MTZ reflection file",
    )


def fetch_gzip(
    url: str,
    path: Path,
    *,
    session: requests.Session,
    timeout: float,
    label: str,
    missing_statuses: frozenset[int] = frozenset(),
) -> bool:
    """Ensure a current gzip file is at ``path``; return whether it was cached.

    The rules are those of fetch_mtz, with the gzip signature. A status in
    ``missing_statuses`` also counts as the source having no such file, for
    an archive that answers 403 where it has none.
    """
    return _fetch_signed(
        url,
        path,
        session=session,
        timeout=timeout,
        label=label,
        signature=GZIP_SIGNATURE,
        kind="gzip file",
        description="a gzip file",
        missing_statuses=missing_statuses,
    )


__all__ = [
    "CACHE_TTL_SECONDS",
    "GZIP_SIGNATURE",
    "MTZ_SIGNATURE",
    "NotAvailableError",
    "fetch_gzip",
    "fetch_mmcif",
    "fetch_mtz",
    "is_fresh_gzip",
    "is_fresh_mmcif",
    "is_fresh_mtz",
    "is_gzip_file",
    "is_mtz_file",
    "read_envelope",
    "write_envelope",
]
