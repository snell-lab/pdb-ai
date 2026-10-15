"""The route that selects PDB identifiers by searching the archive.

POST /api/select asks the RCSB Search API which current entries match a
technique and X-ray source, narrowed by any range filter the request sets.
It answers with the matching identifiers inside the provenance record that
/api/build later stores in the run manifest. The browser fills the
identifier box with them, so a build from a search takes exactly the path a
typed list takes.

The search runs on the request thread, because the page can do nothing useful
until it has the answer. A timeout on every wait keeps a stalled archive from
holding the request open for long. A failed search is logged with its reason.
"""

from __future__ import annotations

from flask import Flask, abort, jsonify
from flask.typing import ResponseReturnValue

from ..selection import ArchiveSearchError, Selection, search_archive
from .jobs import JobService
from .options import (
    parse_limit,
    parse_search_accessions,
    parse_search_bounds,
    parse_technique,
    parse_xray_source,
)
from .route_helpers import bad_request, camel_keys, json_body

# Each connection attempt, and each wait for the next part of the answer, gives
# up after this long. The RCSB answers even an archive-wide search within a few
# seconds, so a longer silence means the service is stalled. The page then
# shows the failure instead of a button that stays disabled.
_SEARCH_TIMEOUT_SECONDS = 20.0

# The page shows these fields of the selection record. The answer repeats them
# beside the record, under camelCase keys.
_SUMMARY_FIELDS = (
    "description",
    "total_count",
    "returned_count",
    "truncated",
    "retrieved_at_iso",
    "warnings",
)


def register_select_routes(app: Flask, service: JobService) -> None:
    """Register POST /api/select on ``app``, under the caps of ``service``."""

    @app.route("/api/select", methods=["POST"])
    def select() -> ResponseReturnValue:
        """Return the archive entries that match the requested filters.

        The JSON body may hold ``technique`` and xraySource, each as text that
        names one. It may also hold ``limit``, a positive whole number or its
        text. It may hold a range filter under each key that
        parse_search_bounds reads, such as minTemperatureK, as a positive
        number or its text, and uniprotAccessions as text or a list of text,
        as parse_search_accessions reads it. Each takes its default when it
        is missing or blank, and a filter left out is not applied. The answer
        holds the search record as ``selection``, with its identifiers in
        selection.pdb_ids. Beside the record it holds ``description``,
        totalCount, returnedCount, ``truncated``, retrievedAtIso,
        ``warnings``, and maxPdbIds, which is null when there is no cap. Bad
        input aborts with 400. A search the archive does not answer aborts
        with 502.
        """
        payload = json_body()
        max_pdb_ids = service.max_pdb_ids
        try:
            requested_limit = parse_limit(payload.get("limit"))
            # A deployment cap on build requests, MAX_PDB_IDS, caps the search
            # the same way. The page then never loads more identifiers than
            # /api/build accepts. The cap is recorded as the selection's
            # limit, so a web run's limit may be the server's cap rather than
            # one the user asked for.
            caps = [cap for cap in (requested_limit, max_pdb_ids) if cap is not None]
            criteria = Selection(
                technique=parse_technique(payload.get("technique")),
                xray_source=parse_xray_source(payload.get("xraySource")),
                limit=min(caps, default=None),
                bounds=parse_search_bounds(payload),
                uniprot_accessions=parse_search_accessions(payload),
            )
        except ValueError as exception:
            bad_request(exception)

        try:
            result = search_archive(criteria, timeout=_SEARCH_TIMEOUT_SECONDS)
        except ArchiveSearchError as exception:
            app.logger.warning(
                "PDB search for %s failed: %s", criteria.describe(), exception
            )
            abort(502, str(exception))

        record = result.to_dict()
        # The record keeps its snake_case keys on purpose. It is the manifest's
        # provenance, and the browser sends it back unchanged. It is also the
        # only copy of the identifiers in the answer. /api/build needs them in
        # the record, and a search can match some 200,000 entries, which are
        # too many to send twice.
        summary = {name: record[name] for name in _SUMMARY_FIELDS}
        return jsonify(
            camel_keys({**summary, "max_pdb_ids": max_pdb_ids, "selection": record})
        )


__all__ = ["register_select_routes"]
