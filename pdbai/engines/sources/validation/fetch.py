"""Fetching an entry's wwPDB validation report and caching the parsed values.

The report is a gzip-compressed XML file in the RCSB archive. It is fetched
and parsed in memory. The parsed values, not the XML, are cached as JSON in
an envelope that expires with the other source caches. Nothing is
cached after a failure, so a failed fetch is tried again on the next run.

A missing or broken report must not fail the entry, since the
validation columns are optional annotations. No network or parse failure
raises here. Each outcome is a ValidationReportResult with a status. The
status is available for a report fetched and parsed now, and cached for one
read from the cache. It is not_available when the archive has no report for
the entry, which the archive signals with HTTP 403, 404, or 410. It is
fetch_failed for any other network or HTTP error. It is parse_failed when
the payload is not gzip or not a readable report. A failed result carries
an empty report, so the validation columns come out empty.
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import requests

from ....identifiers import normalize_pdb_id
from ....json_values import optional_object_dict
from ....network import network_logger
from ..http_cache import read_envelope, write_envelope
from .parse import parse_validation_xml

VALIDATION_STATUS_AVAILABLE = "available"
VALIDATION_STATUS_CACHED = "cached"
VALIDATION_STATUS_NOT_AVAILABLE = "not_available"
VALIDATION_STATUS_FETCH_FAILED = "fetch_failed"
VALIDATION_STATUS_PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class ValidationReportResult:
    """A parsed validation report, or an empty one, and how it was obtained.

    ``report`` has the shape parse_validation_xml returns. ``status`` is one
    of the VALIDATION_STATUS values. ``message`` says why the report is
    missing, and is None for a real report.
    """

    report: dict[str, object]
    status: str
    message: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether ``report`` holds a real report, fetched or cached."""
        return self.status in {
            VALIDATION_STATUS_AVAILABLE,
            VALIDATION_STATUS_CACHED,
        }


def _read_cached_report(path: Path) -> dict[str, object] | None:
    """Return the report cached at ``path``, or None when it must be fetched.

    A report without a mapping of residues counts as missing, as a damaged
    file does, so it costs a download rather than empty columns.
    """
    envelope = read_envelope(path)
    report = optional_object_dict(envelope.get("report")) if envelope else None
    if report is None or not isinstance(report.get("residues"), Mapping):
        return None
    return report


def _failure_result(status: str, message: str) -> ValidationReportResult:
    """Return a result with an empty report of the shape a parsed one has."""
    empty_report: dict[str, object] = {
        "residues": {},
        "clashscore": None,
        "percent_rama_outliers": None,
        "percent_rotamer_outliers": None,
    }
    return ValidationReportResult(report=empty_report, status=status, message=message)


def validation_report_url(pdb_id: str) -> str:
    """Return the archive URL of an entry's wwPDB validation XML report.

    The archive files each report under the second and third characters of
    the entry's lower-case ID. Raise ValueError when ``pdb_id`` is not a PDB
    ID.
    """
    normalized_id = normalize_pdb_id(pdb_id).lower()
    middle = normalized_id[1:3]
    return (
        "https://files.rcsb.org/pub/pdb/validation_reports"
        f"/{middle}/{normalized_id}/{normalized_id}_validation.xml.gz"
    )


def fetch_validation_report(
    pdb_id: str,
    *,
    session: requests.Session,
    timeout: float,
    cache_path: Path,
) -> ValidationReportResult:
    """Return an entry's validation report from the cache or the archive.

    A fresh report at ``cache_path`` is returned without a request.
    Otherwise the report is fetched and, if it parses, written to
    ``cache_path``. ``timeout`` is in seconds. Raise ValueError when
    ``pdb_id`` is not a PDB ID. Every other failure is returned as a result
    whose message names the entry.
    """
    entry_id = normalize_pdb_id(pdb_id)
    cached_report = _read_cached_report(cache_path)
    if cached_report is not None:
        network_logger.info("Using cached validation report for %s", entry_id)
        return ValidationReportResult(
            report=cached_report, status=VALIDATION_STATUS_CACHED
        )

    url = validation_report_url(entry_id)
    try:
        response = session.get(url, timeout=timeout)
        # files.rcsb.org answers 403, not 404, for a file it does not hold,
        # as a store that keeps its listing private does.
        if response.status_code in {403, 404, 410}:
            message = (
                f"wwPDB validation report is not available for {entry_id} "
                f"(HTTP {response.status_code})."
            )
            network_logger.info(message)
            return _failure_result(VALIDATION_STATUS_NOT_AVAILABLE, message)
        response.raise_for_status()
    except Exception as exception:
        message = f"wwPDB validation report fetch failed for {entry_id}: {exception}"
        network_logger.warning(message)
        return _failure_result(VALIDATION_STATUS_FETCH_FAILED, message)

    try:
        xml_bytes = gzip.decompress(response.content)
    except Exception as exception:
        message = (
            f"wwPDB validation report parse failed for {entry_id}: "
            f"invalid gzip payload ({exception})"
        )
        network_logger.warning(message)
        return _failure_result(VALIDATION_STATUS_PARSE_FAILED, message)

    try:
        parsed_report = parse_validation_xml(xml_bytes)
    except Exception as exception:
        message = f"wwPDB validation report parse failed for {entry_id}: {exception}"
        network_logger.warning(message)
        return _failure_result(VALIDATION_STATUS_PARSE_FAILED, message)
    write_envelope(cache_path, {"report": parsed_report})
    network_logger.info(
        "Parsed validation XML for %s: %d residue and conformer entries, clashscore=%s",
        entry_id,
        len(cast(dict[str, object], parsed_report["residues"])),
        parsed_report["clashscore"],
    )
    return ValidationReportResult(
        report=parsed_report, status=VALIDATION_STATUS_AVAILABLE
    )


__all__ = [
    "VALIDATION_STATUS_AVAILABLE",
    "VALIDATION_STATUS_CACHED",
    "VALIDATION_STATUS_FETCH_FAILED",
    "VALIDATION_STATUS_NOT_AVAILABLE",
    "VALIDATION_STATUS_PARSE_FAILED",
    "ValidationReportResult",
    "fetch_validation_report",
    "validation_report_url",
]
