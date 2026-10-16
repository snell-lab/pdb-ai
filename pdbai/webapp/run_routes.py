"""Routes that report, list, serve, pause, resume, and delete stored runs.

Every route that names a run checks the run ID before it touches the disk,
so no route reaches a path outside the runs directory. The file route serves
only a run's tables, report, and summary, and never a symbolic link. The
archive adds a snapshot of the manifest. No route serves the lock, the
partials, or job.json as a file.

The listing and the status route both take a run's status from its manifest,
so the two agree. A run that a job of this server or another process holds
is refused for resume and deletion with 409. A deletion holds the run, here
and against other processes, while it renames the directory away from the
run ID. No build can then start on a half-deleted run. Windows is the
exception, as delete_run explains.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, send_file
from flask.typing import ResponseReturnValue

from ..column_filtering import parse_column_filter_mode
from ..json_values import object_dict, str_list
from ..runs import RunDirectory, RunLockedError, list_run_summaries, run_table_names
from ..runs.state import manifest_text, read_json_object
from ..runs.storage import deletion_name
from .jobs import JobService, RunAlreadyActiveError, effective_status, run_file_url
from .options import engine_options_from_settings
from .route_helpers import (
    RUN_NOT_FOUND,
    camel_keys,
    existing_run,
    refuse_run_in_use,
    remove_stale_files,
    validated_run_id,
)

# Archives can be several gigabytes. They are assembled on the disk that holds
# the runs, because the system's temporary directory may be small. The sweep
# of this folder removes any archive that a killed server left behind.
_SCRATCH_DIRECTORY_NAME = "tmp"
_ARCHIVE_PREFIX = "download-all-"

_FILE_NOT_FOUND = "File not found."


def _is_plain_file(path: Path) -> bool:
    # A symbolic link could lead out of the run directory, so it is never
    # served.
    return path.is_file() and not path.is_symlink()


def _served_filenames(directory: RunDirectory, manifest: dict[str, object]) -> set[str]:
    """Return the names of a run's files that may be downloaded.

    They are the tables that ``manifest`` names, the run report, and the run
    summary.
    """
    tables = {
        directory.dataset_path_for(table).name for table in run_table_names(manifest)
    }
    return tables | {directory.report_path.name, directory.summary_path.name}


def register_run_routes(app: Flask, service: JobService) -> None:
    """Register the routes for stored runs on ``app``.

    Every route that names a run answers a bad run ID with 400. Every one but
    pause answers 404 "Run not found." for a run ID that names no run
    directory.
    """

    @app.route("/api/runs/<run_id>/status")
    def run_status(run_id: str) -> ResponseReturnValue:
        """Return the run's status, latest progress, result summary, and error.

        A job this process still keeps is answered from memory. Any other run
        is answered from its files, so it stays visible after a restart. A
        run whose files record no status is a 404.
        """
        job_status = service.in_memory_status(run_id)
        if job_status is None:
            existing_run(service.working_directory, run_id)
            job_status = service.persisted_status(run_id)
        if job_status is None:
            abort(404, RUN_NOT_FOUND)
        return jsonify(camel_keys(job_status))

    @app.route("/api/runs", methods=["GET"])
    def list_runs() -> ResponseReturnValue:
        """Return a summary of every run, the most recently changed first.

        Each summary is under camelCase keys. Its ``live`` key says whether a
        job of this server is building the run. A runs directory that is a
        symbolic link is a 500.
        """
        # The live runs are taken before the summaries are read. A run that
        # settles meanwhile is then counted live, not shown as stale.
        live_ids = service.live_run_ids()
        try:
            summaries = list_run_summaries(service.working_directory)
        except ValueError:
            app.logger.exception("The runs directory was refused.")
            abort(500, "The runs directory cannot be read.")
        runs: list[dict[str, object]] = []
        for summary in summaries:
            live = summary["run_id"] in live_ids
            held = live or (
                summary["status"] == "running" and service.is_held(summary["run_id"])
            )
            summary["live"] = live
            summary["status"] = effective_status(summary["status"], held=held)
            runs.append(camel_keys(summary))
        return jsonify({"runs": runs})

    @app.route("/api/runs/<run_id>/files/<filename>")
    def run_file(run_id: str, filename: str) -> ResponseReturnValue:
        """Send one of the run's tables, its run report, or its run summary.

        The file is saved as "<run ID>_<filename>". Any other name, a missing
        file, or a symbolic link is a 404.
        """
        directory = existing_run(service.working_directory, run_id).directory
        manifest = read_json_object(directory.manifest_path) or {}
        path = directory.root / filename
        served = _served_filenames(directory, manifest)
        if filename not in served or not _is_plain_file(path):
            abort(404, _FILE_NOT_FOUND)
        # On Windows an open download blocks the atomic replace of this file.
        # A resume that settles the run meanwhile may then end in error.
        return send_file(path, as_attachment=True, download_name=f"{run_id}_{filename}")

    @app.route("/api/runs/<run_id>/download-all")
    def download_run_archive(run_id: str) -> ResponseReturnValue:
        """Send a zip of the run's tables, report, summary, and manifest.

        A run with no table yet is a 404. A run whose files change while the
        zip is built is a 409.
        """
        working_directory = service.working_directory
        run_state = existing_run(working_directory, run_id)
        directory = run_state.directory
        manifest = run_state.manifest()
        tables = [
            path
            for path in map(directory.dataset_path_for, run_table_names(manifest))
            if _is_plain_file(path)
        ]
        if not tables:
            abort(404, "No assembled dataset is available for this run yet.")
        reports = [
            path
            for path in (directory.report_path, directory.summary_path)
            if _is_plain_file(path)
        ]

        scratch_directory = working_directory / _SCRATCH_DIRECTORY_NAME
        scratch_directory.mkdir(parents=True, exist_ok=True)
        remove_stale_files(scratch_directory, f"{_ARCHIVE_PREFIX}*.zip")
        archive = tempfile.NamedTemporaryFile(
            suffix=".zip", prefix=_ARCHIVE_PREFIX, dir=scratch_directory, delete=False
        )
        archive_path = Path(archive.name)
        try:
            with (
                archive,
                zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file,
            ):
                for path in (*tables, *reports):
                    zip_file.write(path, arcname=path.name)
                # The manifest is written from this snapshot, not copied from
                # disk. For a run in progress, the snapshot includes entry
                # updates not yet folded into the manifest file.
                zip_file.writestr(directory.manifest_path.name, manifest_text(manifest))
        except OSError:
            # A file can vanish or be replaced after it was listed, as when
            # the run is deleted or resumed meanwhile.
            archive_path.unlink(missing_ok=True)
            app.logger.exception("The archive of run %r could not be built.", run_id)
            abort(409, "Run changed while its archive was built.")
        except BaseException:
            archive_path.unlink(missing_ok=True)
            raise

        response = send_file(
            archive_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"pdb_dataset_{run_id}.zip",
        )
        # A file passed straight through to the server closes only itself, and
        # the response's close hooks would never run. Sent through the
        # response's own iterator instead, the archive is removed once the
        # WSGI server closes the response. The server does that even when the
        # client disconnects. The sweep above removes the archive of a killed
        # server.
        response.direct_passthrough = False
        response.call_on_close(lambda: archive_path.unlink(missing_ok=True))
        return response

    @app.route("/api/runs/<run_id>/pause", methods=["POST"])
    def pause_run(run_id: str) -> ResponseReturnValue:
        """Ask the job building the run to pause, and answer "pausing".

        The build stops after the entry in hand. A run that no job of this
        server is building is a 409.
        """
        validated_run_id(run_id)
        if not service.request_pause(run_id):
            abort(409, "Run is not being built by this server.")
        return jsonify({"status": "pausing", "runId": run_id})

    @app.route("/api/runs/<run_id>/resume", methods=["POST"])
    def resume_run(run_id: str) -> ResponseReturnValue:
        """Start a job that resumes the run from its manifest.

        The answer's status is "resumed". A run held here or by another
        process is a 409. So is a run of uploaded files, a run with no PDB
        IDs, or a run whose stored settings cannot be used. A run without a
        readable manifest is a 404.
        """
        run_state = existing_run(service.working_directory, run_id)
        # This early check refuses a run that another process holds, before
        # any job starts. Within this process, the claim the job takes is the
        # check that decides.
        if service.is_held(run_id):
            refuse_run_in_use(run_id)
        manifest = run_state.manifest()
        if not manifest:
            abort(404, RUN_NOT_FOUND)
        settings = object_dict(manifest.get("settings"))
        if settings.get("source") == "upload":
            abort(
                409,
                "Runs built from uploaded mmCIF files cannot be resumed from the "
                "web app. Upload the files again to start a new run.",
            )
        pdb_ids = str_list(manifest.get("pdb_ids"))
        if not pdb_ids:
            abort(409, "No PDB IDs are recorded for this run.")
        try:
            engine_options = engine_options_from_settings(settings)
            column_filter_mode = parse_column_filter_mode(
                settings.get("column_filter_mode")
            )
        except ValueError as exception:
            app.logger.warning("Run %r cannot be resumed: %s", run_id, exception)
            abort(409, f"Run settings are unusable. {exception}")

        try:
            service.spawn_download_build(
                run_id,
                pdb_ids=pdb_ids,
                engine_options=engine_options,
                column_filter_mode=column_filter_mode,
                column_filter_list=(
                    str_list(settings.get("column_filter_list")) or None
                ),
                resume=True,
                file_url=run_file_url(run_id),
            )
        except RunAlreadyActiveError:
            refuse_run_in_use(run_id)
        return jsonify({"status": "resumed", "runId": run_id})

    @app.route("/api/runs/<run_id>", methods=["DELETE"])
    def delete_run(run_id: str) -> ResponseReturnValue:
        """Delete the run's directory, and answer "deleted".

        A run held here or by another process is a 409. A directory that
        cannot be removed is a 500.
        """
        run_state = existing_run(service.working_directory, run_id)
        run_root = run_state.directory.root
        # The run leaves its ID by a rename while this process holds it. Once
        # the removal begins, no process can claim the run or write to it, so
        # the removal itself needs no hold. Windows cannot rename a directory
        # while the lock file inside it is open. There the run is released
        # just before the rename.
        tombstone = run_root.with_name(deletion_name(run_id))
        try:
            with service.claimed(run_id), run_state.claimed():
                service.forget(run_id)
                if sys.platform != "win32":
                    run_root.rename(tombstone)
            if sys.platform == "win32":
                run_root.rename(tombstone)
            shutil.rmtree(tombstone)
        except (RunAlreadyActiveError, RunLockedError):
            refuse_run_in_use(run_id)
        except OSError:
            app.logger.exception("Run %r could not be deleted.", run_id)
            abort(500, "The run could not be deleted.")
        return jsonify({"status": "deleted", "runId": run_id})


__all__ = ["register_run_routes"]
