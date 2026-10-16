"""Archive-wide entry selection through the RCSB Search API.

PDB-AI treats the PDB as the hub. Every other source, such as PDB-Redo,
AlphaFold, wwPDB validation, and UniProt, is looked up by PDB identifier. A
cohort is therefore selected once, against the archive itself. This module
turns the run's filters into one query and sends it to the RCSB Search API.

The RCSB searches its own index of the whole archive and answers with only
the identifiers of matching entries. PDB-AI downloads and scans nothing to
find them, and keeps no local copy of the archive. The manifest keeps the
query, the service, and the time instead. That is enough to reproduce the
cohort, or to explain why it differs from a later one.

The technique and the X-ray source choose the cohort. Range filters, which
pdbai.selection_filters declares, narrow it by numbers the archive records,
such as the data-collection temperature.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import requests

from .engines.core import (
    ALL_XRAY_SOURCES,
    DEFAULT_TECHNIQUE,
    INTEGRATIVE_METHOD,
    EngineOptions,
    get_technique,
    get_xray_source,
)
from .identifiers import normalize_pdb_id
from .network import DEFAULT_TIMEOUT_SECONDS, create_retry_session, network_logger
from .positive_numbers import positive_int
from .selection_filters import (
    SEARCH_FILTERS,
    UNIPROT_ACCESSIONS,
    check_bounds,
    describe_bounds,
    range_filter_nodes,
)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_SEARCH_SERVICE = "rcsb_search"


def _attribute_node(attribute: str, value: str) -> dict[str, Any]:
    """Return an RCSB Search API node matching one entry attribute exactly."""
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": attribute,
            "operator": "exact_match",
            "value": value,
        },
    }


@dataclass(frozen=True)
class Selection:
    """The filters one run asks the archive for.

    ``technique`` and ``xray_source`` mirror the run's own options, so what
    the search returns for them is what intake admits. ``bounds`` holds the
    range filters, keyed by the identifiers pdbai.selection_filters
    declares, such as ``{"min_temperature_K": 273}``. ``uniprot_accessions``
    keeps only the entries mapped to one of those accessions, as
    UNIPROT_ACCESSIONS describes, and is empty when the filter is not set.
    These narrow the search only, and intake does not check them. ``limit``
    caps how many identifiers are kept. The first ones in sorted order are
    kept, so a cohort is sampled the same way every time.
    """

    technique: str = DEFAULT_TECHNIQUE
    xray_source: str = ALL_XRAY_SOURCES
    limit: int | None = None
    bounds: Mapping[str, float] = field(default_factory=dict[str, float], hash=False)
    uniprot_accessions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check every filter, raising ValueError, and copy the bounds.

        The technique and source follow the same rules as a run's own options,
        so a source is allowed only for X-ray diffraction. The limit must be
        a positive integer or None. The bounds must pass check_bounds, and
        the accessions the check of UNIPROT_ACCESSIONS.
        """
        EngineOptions(technique=self.technique, xray_source=self.xray_source)
        # SelectionResult.from_dict passes a stored limit and stored bounds
        # through unchecked, of any type.
        if self.limit is not None:
            positive_int(self.limit, "limit")
        # A copy in declaration order, so that a mapping the caller changes
        # later cannot change a selection that is meant to be frozen, and so
        # that equal selections describe themselves the same way.
        object.__setattr__(self, "bounds", check_bounds(self.bounds))
        object.__setattr__(
            self,
            "uniprot_accessions",
            UNIPROT_ACCESSIONS.check(self.uniprot_accessions),
        )

    def describe(self) -> str:
        """Return the technique, any X-ray source, and any filters as one phrase."""
        labels = [get_technique(self.technique).display_label]
        if self.xray_source != ALL_XRAY_SOURCES:
            labels.append(get_xray_source(self.xray_source).display_label)
        labels.extend(describe_bounds(self.bounds))
        if self.uniprot_accessions:
            labels.append(UNIPROT_ACCESSIONS.describe(self.uniprot_accessions))
        return ", ".join(labels)

    def to_dict(self) -> dict[str, Any]:
        """Return the selection as a JSON-safe dictionary.

        Every filter has a key, holding None when the selection does not set
        it, so the record has the same keys whatever was asked.
        """
        return {
            "technique": self.technique,
            "xray_source": self.xray_source,
            "limit": self.limit,
            **{
                search_filter.id: self.bounds.get(search_filter.id)
                for search_filter in SEARCH_FILTERS
            },
            UNIPROT_ACCESSIONS.id: list(self.uniprot_accessions) or None,
        }


def _retrieval_time(value: object) -> float:
    """Return a stored retrieval time, or raise ValueError if it cannot be one.

    The time must not be negative, and time.gmtime must be able to convert
    it. A record that passes here can then always be written out again in
    its ISO 8601 form.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("selection.retrieved_at must be a number.")
    try:
        timestamp = float(value)
        time.gmtime(timestamp)
    except (OverflowError, OSError, ValueError) as exception:
        raise ValueError(
            "selection.retrieved_at is not a time the platform can represent."
        ) from exception
    if timestamp < 0:
        raise ValueError("selection.retrieved_at must not be before 1970.")
    return timestamp


def _string(
    record: Mapping[str, Any], key: str, default: str, *, path: str = "selection"
) -> str:
    """Return a string field, taking the default only when it is missing.

    A field that is present but not a string, null included, raises
    ValueError naming it as ``path``.``key``.
    """
    value = record.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key} must be a string.")
    return value


@dataclass(frozen=True)
class SelectionResult:
    """The identifiers the archive returned for ``filters``, with how and when.

    This record is written into the run manifest. A cohort built from a
    search can then be reproduced, audited, or compared with a later
    archive. ``truncated`` says whether the filters' limit dropped
    identifiers the search found. It is stored rather than worked out from
    ``total_count``, since hits skipped as invalid also leave fewer
    identifiers than matches.
    """

    filters: Selection
    pdb_ids: tuple[str, ...]
    total_count: int
    service: str
    service_url: str
    query: dict[str, Any]
    retrieved_at: float
    truncated: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def retrieved_at_iso(self) -> str:
        """Return the retrieval time as an ISO 8601 UTC timestamp."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.retrieved_at))

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-safe dictionary."""
        return {
            "filters": self.filters.to_dict(),
            "description": self.filters.describe(),
            "pdb_ids": list(self.pdb_ids),
            "total_count": self.total_count,
            "returned_count": len(self.pdb_ids),
            "truncated": self.truncated,
            "service": self.service,
            "service_url": self.service_url,
            "query": self.query,
            "retrieved_at": self.retrieved_at,
            "retrieved_at_iso": self.retrieved_at_iso,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: object) -> SelectionResult:
        """Rebuild a result from its dictionary form, checking every field.

        The record must also hold together the way one from search_archive
        does. Its identifiers must be sorted and distinct. There may be no
        more of them than the search matched or the limit allows.
        ``truncated`` may be set only when the limit was reached. A record
        that breaks any of these rules raises ValueError naming the field. A
        record the browser sends back is provenance, so it is held to the same
        rules as one the server made.

        A filter the record leaves out, or holds as None, was not set. A
        record written before a filter existed therefore reads as a search
        without it.
        """
        if not isinstance(value, Mapping):
            raise ValueError("selection must be a JSON object.")
        record = cast(Mapping[str, Any], value)
        raw_filters = record.get("filters")
        if not isinstance(raw_filters, Mapping):
            raise ValueError("selection.filters must be a JSON object.")
        stored = cast(Mapping[str, Any], raw_filters)
        filters = Selection(
            technique=_string(
                stored, "technique", DEFAULT_TECHNIQUE, path="selection.filters"
            ),
            xray_source=_string(
                stored, "xray_source", ALL_XRAY_SOURCES, path="selection.filters"
            ),
            limit=stored.get("limit"),
            bounds={
                search_filter.id: stored[search_filter.id]
                for search_filter in SEARCH_FILTERS
                if stored.get(search_filter.id) is not None
            },
            uniprot_accessions=_stored_accessions(stored.get(UNIPROT_ACCESSIONS.id)),
        )
        raw_ids = record.get("pdb_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) for item in cast(list[object], raw_ids)
        ):
            raise ValueError("selection.pdb_ids must be a list of strings.")
        pdb_ids = tuple(normalize_pdb_id(item) for item in cast(list[str], raw_ids))
        if list(pdb_ids) != sorted(set(pdb_ids)):
            raise ValueError("selection.pdb_ids must be sorted and distinct.")
        total = record.get("total_count")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("selection.total_count must be a non-negative integer.")
        if total < len(pdb_ids):
            raise ValueError("selection.total_count is below the identifiers kept.")
        limit = filters.limit
        if limit is not None and len(pdb_ids) > limit:
            raise ValueError("selection.pdb_ids holds more identifiers than the limit.")
        truncated = record.get("truncated", False)
        if not isinstance(truncated, bool):
            raise ValueError("selection.truncated must be true or false.")
        if truncated and (len(pdb_ids) != limit or total == len(pdb_ids)):
            raise ValueError(
                "selection.truncated is set, but the limit did not drop identifiers."
            )
        retrieved_at = _retrieval_time(record.get("retrieved_at"))
        query = record.get("query")
        if not isinstance(query, Mapping):
            raise ValueError("selection.query must be a JSON object.")
        raw_warnings: object = record.get("warnings", [])
        if not isinstance(raw_warnings, list) or not all(
            isinstance(item, str) for item in cast(list[object], raw_warnings)
        ):
            raise ValueError("selection.warnings must be a list of strings.")
        return cls(
            filters=filters,
            pdb_ids=pdb_ids,
            total_count=total,
            service=_string(record, "service", ""),
            service_url=_string(record, "service_url", ""),
            query=dict(cast(Mapping[str, Any], query)),
            retrieved_at=retrieved_at,
            truncated=truncated,
            warnings=tuple(cast(list[str], raw_warnings)),
        )


def _stored_accessions(value: object) -> tuple[str, ...]:
    # A stored list is checked by the Selection it goes into. Anything else
    # but None is refused here with the filter's own message.
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(cast(list[str], value))
    raise ValueError(
        f"selection.filters.{UNIPROT_ACCESSIONS.id} must be a list of UniProt "
        f"accessions; got {value!r}."
    )


class ArchiveSearchError(RuntimeError):
    """An archive search that failed or answered with no result set."""


def build_search_request(selection: Selection) -> dict[str, Any]:
    """Return the RCSB Search API request body that expresses a selection."""
    technique = get_technique(selection.technique)
    # Technique intake classifies an integrative entry by its structure
    # determination methodology, not by exptl.method, so the search does too.
    nodes = [
        _attribute_node(
            "rcsb_entry_info.structure_determination_methodology", "integrative"
        )
        if technique.method == INTEGRATIVE_METHOD
        else _attribute_node("exptl.method", technique.method)
    ]
    if selection.xray_source != ALL_XRAY_SOURCES:
        source = get_xray_source(selection.xray_source)
        nodes.append(_attribute_node("diffrn_source.source", source.source))
    nodes.extend(range_filter_nodes(selection.bounds))
    if selection.uniprot_accessions:
        nodes.extend(UNIPROT_ACCESSIONS.nodes(selection.uniprot_accessions))
    query: dict[str, Any] = (
        nodes[0]
        if len(nodes) == 1
        else {"type": "group", "logical_operator": "and", "nodes": nodes}
    )
    return {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            # Computed structure models are left out. They are not PDB entries,
            # and they have no PDB-Redo, validation, or intake record.
            "results_content_type": ["experimental"],
        },
    }


def _search_session() -> requests.Session:
    # The query only reads, so retrying the POST on a transient failure is
    # safe.
    return create_retry_session(allowed_methods=("GET", "POST"))


def _post_query(
    request_body: dict[str, Any], session: requests.Session | None, timeout: float
) -> requests.Response:
    """Send the query and return the answer, its body already read.

    A request that fails after any retries raises ArchiveSearchError.
    """
    try:
        if session is not None:
            return session.post(RCSB_SEARCH_URL, json=request_body, timeout=timeout)
        # A session made here is closed here, not at garbage collection.
        with _search_session() as owned:
            return owned.post(RCSB_SEARCH_URL, json=request_body, timeout=timeout)
    except requests.RequestException as exception:
        raise ArchiveSearchError(f"PDB search failed: {exception}") from exception


def _read_hits(response: requests.Response) -> tuple[list[str], int, list[str]]:
    """Return the identifiers, match count, and warnings an answer holds.

    The identifiers are normalized but neither sorted nor deduplicated. A hit
    that is not a valid PDB identifier is skipped with one warning for all
    of them. A reported match count that differs from the number of hits is
    warned about, and the larger of the two is returned. An answer other
    than 200 with a result set, or 204 for an empty one, raises
    ArchiveSearchError.
    """
    if response.status_code == 204:
        return [], 0, []
    if response.status_code != 200:
        raise ArchiveSearchError(
            f"PDB search failed: {RCSB_SEARCH_URL} answered HTTP "
            f"{response.status_code}."
        )
    try:
        payload = cast(object, response.json())
    except ValueError as exception:
        raise ArchiveSearchError("PDB search returned malformed JSON.") from exception
    if not isinstance(payload, dict):
        raise ArchiveSearchError("PDB search returned an unexpected payload.")
    body = cast(dict[str, Any], payload)
    raw_hits = body.get("result_set")
    if not isinstance(raw_hits, list):
        raise ArchiveSearchError("PDB search returned no result set.")
    hits = cast(list[object], raw_hits)

    warnings: list[str] = []
    identifiers: list[str] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        identifier = cast(dict[str, Any], hit).get("identifier")
        if isinstance(identifier, str):
            with contextlib.suppress(ValueError):
                identifiers.append(normalize_pdb_id(identifier))
    skipped = len(hits) - len(identifiers)
    if skipped:
        warnings.append(
            f"{skipped} search hit(s) were not valid PDB identifiers and were ignored."
        )
    # The count is checked against the hits returned, not the identifiers
    # kept from them, so hits skipped above are not reported a second time.
    raw_total = body.get("total_count")
    total_count = (
        raw_total
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else len(hits)
    )
    if total_count != len(hits):
        warnings.append(
            f"The search reported {total_count} matching entries but returned "
            f"{len(hits)} hits."
        )
        total_count = max(total_count, len(hits))
    return identifiers, total_count, warnings


def search_archive(
    selection: Selection,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SelectionResult:
    """Return every current PDB entry matching a selection.

    The identifiers come back sorted, distinct, and capped by the selection's
    limit. The result also records the query and the retrieval time.
    ``timeout`` is in seconds. It bounds each connection attempt and each
    wait for the next part of the answer, not the whole search. A
    ``session`` that is given is used as it is and left open. Without one, a
    session made here retries the POST as create_retry_session sets out, and
    is closed afterwards. A failure that persists raises ArchiveSearchError,
    as does an answer that is not a result set. A hit that is not a valid
    PDB identifier is skipped with a warning.
    """
    request_body = build_search_request(selection)
    network_logger.info(
        "Searching the PDB for %s via %s", selection.describe(), RCSB_SEARCH_URL
    )
    response = _post_query(request_body, session, timeout)
    retrieved_at = time.time()

    identifiers, total_count, warnings = _read_hits(response)
    unique_ids = sorted(set(identifiers))
    kept = unique_ids[: selection.limit] if selection.limit is not None else unique_ids
    network_logger.info(
        "PDB search matched %d entries for %s; keeping %d",
        total_count,
        selection.describe(),
        len(kept),
    )
    return SelectionResult(
        filters=selection,
        pdb_ids=tuple(kept),
        total_count=total_count,
        service=RCSB_SEARCH_SERVICE,
        service_url=RCSB_SEARCH_URL,
        query=request_body,
        retrieved_at=retrieved_at,
        truncated=len(kept) < len(unique_ids),
        warnings=tuple(warnings),
    )


__all__ = [
    "RCSB_SEARCH_SERVICE",
    "RCSB_SEARCH_URL",
    "ArchiveSearchError",
    "Selection",
    "SelectionResult",
    "build_search_request",
    "search_archive",
]
