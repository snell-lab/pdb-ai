"""Request checks, response keys, and housekeeping shared by the routes.

Every check here either returns a value that passed it or aborts the request
through ``abort``. A route therefore never goes on with an unchecked run ID or
request body. The application's error handlers turn each abort into the same
JSON error response. A bad run ID is a 400, and a run that cannot be found is
a 404. When a refusal would name a local path, the path is logged and the
client gets a plain message.

The API's camelCase keys are made here, by the same camel_case the engine
options use, so a setting has one JSON name everywhere. Nothing else removes
scratch archives and filtered tables, so they are deleted here once they are
old.
"""

from __future__ import annotations

import stat
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import NoReturn

from flask import abort, current_app, request
from werkzeug.exceptions import BadRequest

from ..engines.core import camel_case
from ..json_values import optional_object_dict
from ..runs import RunState, new_run_id, open_run, validate_run_id
from ..runs.storage import run_id_refusal

# A download link the page has just shown must still work when the user clicks
# it, so a file is kept for an hour after it was last written. An hour is far
# longer than any fetch takes. It is also short enough that the workspace does
# not fill up across sessions.
_STALE_FILE_AGE_SECONDS = 60 * 60

# The 404 message for a run ID that names no run, whatever the route.
RUN_NOT_FOUND = "Run not found."


def camel_keys(fields: Mapping[str, object]) -> dict[str, object]:
    """Return the fields in a new dict, under the API's camelCase keys.

    Only the top-level keys are converted. Nested values are returned as they
    are. Their keys are data, such as column and table names, not field names.
    """
    return {camel_case(name): value for name, value in fields.items()}


def bad_request(error: ValueError | str) -> NoReturn:
    """Abort with a 400 response whose error is ``error``'s message.

    Only a ValueError or a plain message is accepted, because the message
    reaches the client. The validators write their ValueError messages for
    the client to read.
    """
    abort(400, str(error))


def json_body() -> dict[str, object]:
    """Return the request's JSON object body as a new dict.

    A request whose Content-Type is not JSON aborts with 415. A body that is
    not valid JSON, or is valid JSON but not an object, aborts with 400.
    """
    # A browser sends a cross-origin JSON write only after a CORS preflight,
    # and this app never grants one. Requiring JSON therefore backs up the
    # Origin check.
    if not request.is_json:
        abort(415, "Content-Type must be application/json.")
    try:
        body: object = request.get_json()
    except BadRequest:
        bad_request("Request body must be valid JSON.")
    payload = optional_object_dict(body)
    if payload is None:
        bad_request("Request body must be a JSON object.")
    return payload


def validated_run_id(run_id: object) -> str:
    """Return ``run_id`` unchanged when it is a string that can name a run.

    A run_id that is not a string, or that breaks the rule validate_run_id
    enforces, aborts with 400.
    """
    if not isinstance(run_id, str):
        bad_request(f"Run ID must be a string, not {run_id!r}.")
    try:
        return validate_run_id(run_id)
    except ValueError as exception:
        bad_request(exception)


def existing_run(working_directory: Path, run_id: str) -> RunState:
    """Return the handle on the run that run_id names in working_directory.

    An invalid run_id aborts with 400. A run directory that is missing, or is
    not a directory, aborts with 404. So does a run or runs directory that is
    a symbolic link.
    """
    validated = validated_run_id(run_id)
    try:
        run_state = open_run(working_directory, validated)
    except ValueError as exception:
        # The ID is valid, so only a symbolic link can be refused here. The
        # refusal names a local path, which belongs in the log, not the
        # response.
        current_app.logger.warning("Run %r refused: %s", validated, exception)
        abort(404, RUN_NOT_FOUND)
    if not run_state.directory.root.is_dir():
        abort(404, RUN_NOT_FOUND)
    return run_state


def refuse_run_in_use(run_id: str) -> NoReturn:
    """Abort with 409 because a build or a deletion holds the run."""
    # One message serves a run held by a build in any process, and a run held
    # by a deletion here.
    abort(409, f"Run {run_id!r} is already being built or deleted; try again later.")


def run_id_to_build(
    working_directory: Path,
    raw_run_id: object,
    *,
    resume: bool,
    is_held: Callable[[str], bool],
) -> str:
    """Return the ID of the run a build request builds, once it may start.

    raw_run_id is the request's runId. None or an empty string names no run,
    so a new ID is returned, and with ``resume`` the request aborts with 400.
    Any other value must be a valid run ID string, or the request aborts with
    400, so 0, false, or [] is refused rather than replaced with a new ID.
    The rest is the rule run_id_refusal holds. Resuming a run that does not
    exist aborts with 404, as does a run or runs directory that is a symbolic
    link, and resuming a run that is_held reports as held aborts with 409. A
    runId that names an existing run without ``resume`` aborts with 409, so a
    finished run is never built over by accident.
    """
    if raw_run_id is None or raw_run_id == "":
        if resume:
            bad_request("resume requires a runId naming the run to resume.")
        return new_run_id()
    run_id = validated_run_id(raw_run_id)
    try:
        refusal = run_id_refusal(working_directory, run_id, resume=resume)
    except ValueError as exception:
        # The ID is valid, so only a symbolic link can be refused here, as in
        # existing_run.
        current_app.logger.warning("Run %r refused: %s", run_id, exception)
        abort(404, RUN_NOT_FOUND)
    if refusal == "missing":
        abort(404, RUN_NOT_FOUND)
    if refusal == "exists":
        abort(
            409,
            f"Run {run_id!r} already exists; resume it or choose another run ID.",
        )
    if resume and is_held(run_id):
        refuse_run_in_use(run_id)
    return run_id


def remove_stale_files(target_directory: Path, pattern: str) -> None:
    """Delete the old files in target_directory that match the glob ``pattern``.

    A regular file or symbolic link last modified more than an hour ago is
    deleted. A symbolic link is judged by its own age, and the link is deleted,
    never its target. A directory or anything else is left alone. A file that
    cannot be examined or deleted is skipped until a later call.
    """
    cutoff = time.time() - _STALE_FILE_AGE_SECONDS
    for candidate in target_directory.glob(pattern):
        # A file can vanish, or be held open on Windows, between the listing
        # and the deletion. A later call retries it.
        with suppress(OSError):
            status = candidate.lstat()
            removable = stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode)
            if removable and status.st_mtime < cutoff:
                candidate.unlink()


__all__ = [
    "RUN_NOT_FOUND",
    "bad_request",
    "camel_keys",
    "existing_run",
    "json_body",
    "refuse_run_in_use",
    "remove_stale_files",
    "run_id_to_build",
    "validated_run_id",
]
