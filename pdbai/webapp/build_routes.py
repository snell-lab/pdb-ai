"""The route that starts a dataset build from PDB identifiers.

POST /api/build checks everything a request says before it starts anything.
A refused request therefore leaves no run directory and no job state behind.
A selection record sent with the identifiers must list those identifiers. It
must also name the technique and X-ray source the request builds with. The
run keeps the record as the provenance of its cohort, so the record must be
true of the run.

A request that names a run must resume it when it exists, and may start it
only when it does not. A finished run is therefore never built over by
accident. A run that a job of this server or another process is building is
refused with 409. It is not accepted only to fail in the background.
"""

from __future__ import annotations

from flask import Flask, jsonify
from flask.typing import ResponseReturnValue

from ..selection import SelectionResult
from .jobs import JobService, RunAlreadyActiveError
from .options import parse_flag, parse_pdb_ids, settings_from_request
from .route_helpers import bad_request, json_body, refuse_run_in_use, run_id_to_build


def register_build_routes(app: Flask, service: JobService) -> None:
    """Register POST /api/build on ``app``, building through ``service``."""

    # The function's name is the endpoint name factory.py passes to url_for.
    @app.route("/api/build", methods=["POST"])
    def build() -> ResponseReturnValue:
        """Start a build from the JSON body's PDB IDs, and return its run ID.

        The body holds pdbIds, as text or a list of text, and the run
        settings that settings_from_request reads. It may also hold runId,
        ``resume``, and ``selection``. ``selection`` is the record that
        /api/select returned, sent back unchanged, so its keys stay
        snake_case. A build that starts answers 200 with the run's ID as
        runId.

        The route answers 415 for a body that is not JSON. It answers 400 for
        a body that is not an object, for no identifiers or an invalid one,
        and for more distinct identifiers than the server's cap. It answers
        400 for invalid settings, and for a malformed ``selection`` or one
        that describes other identifiers or settings. It also answers 400 for
        a runId that is not a valid run ID, and for a ``resume`` that is not a
        flag or comes without a runId. Resuming a run that does not exist is a
        404. A runId that names an existing run without ``resume`` is a 409,
        and so is a run that is being built or deleted.
        """
        payload = json_body()
        try:
            pdb_ids = parse_pdb_ids(payload.get("pdbIds"))
            settings = settings_from_request(payload, source="download")
            resume = parse_flag(payload.get("resume"), "resume")
        except ValueError as exception:
            bad_request(exception)
        if not pdb_ids:
            bad_request("At least one valid PDB ID is required.")
        # The build keeps each identifier once, so the cap counts each once.
        distinct_ids = list(dict.fromkeys(pdb_ids))
        max_pdb_ids = service.max_pdb_ids
        if max_pdb_ids is not None and len(distinct_ids) > max_pdb_ids:
            bad_request(
                f"A request may name at most {max_pdb_ids} distinct PDB IDs; "
                f"got {len(distinct_ids)}."
            )

        search_result: SelectionResult | None = None
        raw_selection = payload.get("selection")
        if raw_selection is not None:
            try:
                search_result = SelectionResult.from_dict(raw_selection)
            except ValueError as exception:
                bad_request(f"Invalid selection record: {exception}")
            if list(search_result.pdb_ids) != sorted(distinct_ids):
                bad_request(
                    "The selection record must list the same PDB IDs as pdbIds; "
                    f"it lists {len(search_result.pdb_ids)} and pdbIds names "
                    f"{len(distinct_ids)} distinct."
                )
            searched = search_result.filters
            options = settings.options
            if (searched.technique, searched.xray_source) != (
                options.technique,
                options.xray_source,
            ):
                bad_request(
                    "The selection record must describe the requested technique "
                    f"and X-ray source; it has {searched.technique!r} and "
                    f"{searched.xray_source!r}, the request {options.technique!r} "
                    f"and {options.xray_source!r}."
                )

        run_id = run_id_to_build(
            service.working_directory,
            payload.get("runId"),
            resume=resume,
            is_held=service.is_held,
        )

        try:
            service.spawn_download_build(
                run_id,
                pdb_ids=pdb_ids,
                engine_options=settings.options,
                column_filter_mode=settings.column_filter_mode,
                column_filter_list=settings.column_filter_list,
                resume=resume,
                search_result=search_result,
            )
        except RunAlreadyActiveError:
            refuse_run_in_use(run_id)
        app.logger.info("Run %s started building %d PDB IDs.", run_id, len(pdb_ids))
        return jsonify({"runId": run_id})


__all__ = ["register_build_routes"]
