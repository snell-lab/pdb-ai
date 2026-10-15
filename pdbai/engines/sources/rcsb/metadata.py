"""An entry's Core Entry record, fetched from RCSB through the cache.

fetch_rcsb_metadata returns the curated fields that extraction.py reads from
the record. It also returns the record itself, for engines that need more of
it. rcsb_entry_url gives the record's address, for the run manifest and for
the AlphaFold engine.

The cache holds the record as RCSB sent it, not the fields read from it. So
a fix to the extraction applies at once to every entry already cached.

A failed fetch does not raise. It returns a result marked as not ok, and the
run goes on without the entry's metadata. A response that is not a non-empty
JSON object counts as a failed fetch, since RCSB always returns an object
for an entry it holds. Reading it as an entry would report every field as
missing and cache that for the whole cache lifetime. For the same reason an
empty cached record is fetched again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import requests

from ....identifiers import normalize_pdb_id
from ....json_values import optional_object_dict
from ....network import network_logger
from ....results import MetadataFetchResult
from ..http_cache import read_envelope, write_envelope
from .extraction import (
    extract_beamline_and_wavelength,
    extract_citation_fields,
    extract_experiment_fields,
)


def rcsb_entry_url(pdb_id: str) -> str:
    """Return the RCSB Data API URL of an entry's Core Entry record."""
    return (
        f"https://data.rcsb.org/rest/v1/core/entry/{normalize_pdb_id(pdb_id).lower()}"
    )


def _read_cached_entry(path: Path) -> dict[str, Any] | None:
    """Return the record cached at ``path``, or None when it must be fetched."""
    envelope = read_envelope(path)
    entry = optional_object_dict(envelope.get("entry")) if envelope else None
    return entry or None


def _require_entry_record(payload: object) -> dict[str, Any]:
    """Return the payload as a dict, or raise ValueError.

    Anything but a non-empty JSON object raises.
    """
    record = optional_object_dict(payload)
    if not record:
        raise ValueError(
            "the response must be a non-empty JSON object; "
            f"got {type(payload).__name__} {payload!r:.80}"
        )
    return record


def fetch_rcsb_metadata(
    pdb_id: str,
    *,
    session: requests.Session,
    timeout: float,
    metadata_cache_path: Path,
) -> MetadataFetchResult:
    """Return an entry's curated fields and its record, from the cache or RCSB.

    A fresh record at ``metadata_cache_path`` is used without a request.
    Otherwise the record is fetched and written there. ``timeout`` is in
    seconds. A failed fetch returns empty metadata, with ok False and the
    error as the message, and caches nothing. A response that is not a
    non-empty JSON object is a failed fetch. A malformed PDB ID raises
    ValueError.
    """
    normalized_id = normalize_pdb_id(pdb_id)
    entry_json = _read_cached_entry(metadata_cache_path)
    cached = entry_json is not None
    if entry_json is not None:
        network_logger.info("Using cached RCSB metadata for %s", normalized_id)
    else:
        url = rcsb_entry_url(normalized_id)
        network_logger.info(
            "Requesting RCSB metadata for %s from %s", normalized_id, url
        )
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            entry_json = _require_entry_record(cast(object, response.json()))
        except Exception as exception:
            return MetadataFetchResult(
                metadata={},
                ok=False,
                message=f"Metadata fetch failed for {normalized_id}: {exception}",
                level="error",
            )
        write_envelope(metadata_cache_path, {"entry": entry_json})

    return MetadataFetchResult(
        metadata={
            **extract_beamline_and_wavelength(entry_json),
            **extract_experiment_fields(entry_json),
            **extract_citation_fields(entry_json),
        },
        ok=True,
        message=None,
        level="info",
        entry_record=entry_json,
        cached=cached,
    )


__all__ = ["fetch_rcsb_metadata", "rcsb_entry_url"]
