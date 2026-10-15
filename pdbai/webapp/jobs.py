"""Background dataset builds for the web app, and the state each one leaves.

A job is one build of one run on a background thread. JobService claims the
run in this process before the thread starts, so two jobs never build one
run here at once. The build itself claims the run against other processes
through the run's own lock. While a job is in this process, its record here
is what /api/runs/<run_id>/status reports. Once the job's final state is
written to the run's job.json, the record is dropped. From then on the file
serves the job, after a restart too.

job.json is written only once the build has shown that it holds the run. It
shows this by reporting progress, returning a result, or pausing. A job may
be refused because another process holds the run, or because the run exists
with other settings. Such a job never overwrites that run's job state, and a
run that never started leaves no directory behind. Its record is the only
place its error is kept, so status returns it once and then drops it. Later
reads then defer to the run's files, which the process that holds the run
may still be writing. A status read back from disk defers to the run's
manifest, so the status route and the run listing agree. A run recorded as
running that no worker holds, here or in another process, is reported as
stale.

A job's web payload carries its status, its latest progress event, its
error, and the result summary the page renders. The per-entry results and
the progress history stay in the run's own files, run_report.csv and
events.jsonl. The page reads neither, and sending them would make every poll
of a large run heavy.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, TypeGuard, get_args

from ..column_filtering import ColumnFilterMode
from ..engines.core import EngineOptions
from ..json_values import optional_object_dict
from ..orchestration import (
    build_atom_site_dataset,
    build_atom_site_dataset_from_mmcif_files,
)
from ..results import DatasetBuildResult, RunPausedError
from ..runs import ProgressCallback, ProgressStage, open_run
from ..runs.state import RunStatus, read_json_object
from ..selection import SelectionResult
from .route_helpers import camel_keys

_logger = logging.getLogger(__name__)

# Progress may arrive once per structure, and job.json needs no write for each.
_PERSIST_MIN_INTERVAL_SECONDS = 1.0

_RUN_STATUSES: frozenset[str] = frozenset(get_args(RunStatus.__value__))

_PROGRESS_STAGES: frozenset[str] = frozenset(get_args(ProgressStage.__value__))

# A client is shown a run's own status or one of two more. "stale" marks a run
# recorded as running that no worker holds. "unknown" marks a run with no
# status.
type ClientStatus = Literal[
    "running", "paused", "completed", "error", "stale", "unknown"
]


class ProgressEvent(TypedDict):
    """One progress report, under the keys events.jsonl and the page use.

    subject names what the stage is working on, as ProgressCallback
    describes.
    """

    stage: ProgressStage
    current: int
    total: int
    subject: str


class JobStatus(TypedDict):
    """A job's status as its status route reports it, before camel_keys runs."""

    run_id: str
    status: ClientStatus
    progress: ProgressEvent | None
    # The summary of the build's result, already under camelCase keys.
    result: dict[str, object] | None
    error: str | None


class _PersistedJob(TypedDict):
    # This is the shape of a run's job.json.
    status: RunStatus
    progress: ProgressEvent | None
    result: dict[str, object] | None
    error: str | None


class RunAlreadyActiveError(RuntimeError):
    """The refusal raised when a job or a deletion already holds a run here."""


@dataclass
class _Job:
    """One job's record, changed only by its worker once the job starts."""

    run_id: str
    # The token under which the job holds the run in JobService._claims.
    claim_token: object
    cancel_event: threading.Event = field(default_factory=threading.Event)
    status: RunStatus = "running"
    progress: ProgressEvent | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    # This is set once the build has shown that it holds the run. Only then
    # may job.json be written.
    holds_run: bool = False
    # This is when job.json was last written, by time.monotonic(), or None
    # before the first write.
    persisted_at: float | None = None


def _is_run_status(value: object) -> TypeGuard[RunStatus]:
    return isinstance(value, str) and value in _RUN_STATUSES


def _is_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_progress_event(value: object) -> TypeGuard[ProgressEvent]:
    event = optional_object_dict(value)
    if event is None:
        return False
    stage = event.get("stage")
    return (
        isinstance(stage, str)
        and stage in _PROGRESS_STAGES
        and _is_count(event.get("current"))
        and _is_count(event.get("total"))
        and isinstance(event.get("subject"), str)
    )


def _persisted_job(value: object) -> _PersistedJob | None:
    # job.json may be damaged or come from elsewhere, so a record with any
    # value of the wrong type is refused whole. Any other key is ignored.
    record = optional_object_dict(value)
    if record is None:
        return None
    status = record.get("status")
    progress = record.get("progress")
    raw_result = record.get("result")
    result = optional_object_dict(raw_result)
    error = record.get("error")
    if not _is_run_status(status):
        return None
    if progress is not None and not _is_progress_event(progress):
        return None
    if raw_result is not None and result is None:
        return None
    if error is not None and not isinstance(error, str):
        return None
    return _PersistedJob(status=status, progress=progress, result=result, error=error)


def effective_status(status: object, *, held: bool) -> ClientStatus:
    """Return a run's recorded status as a client should see it.

    A run recorded as running that no worker holds was stopped with its
    process, as by a restart. It is reported as stale. Any value that is not
    a status a run records is reported as unknown, including a value that is
    not a string.
    """
    if not _is_run_status(status):
        return "unknown"
    if status == "running" and not held:
        return "stale"
    return status


def _result_payload(run_id: str, result: DatasetBuildResult) -> dict[str, object]:
    # Web builds are not published, so each file is linked where settlement
    # wrote it in the run directory. Its name alone identifies it there.
    def link(path: Path) -> str:
        return f"/api/runs/{run_id}/files/{path.name}"

    summary = {
        # These are the entries the dataset holds, which the page shows as
        # processed.
        "pdb_count": sum(1 for entry in result.pdb_results if entry.succeeded()),
        "requested_count": len(result.pdb_ids),
        "atom_count": result.atom_count,
        "column_count": result.column_count,
    }
    # Table names are data, so the keys of the table links stay as they are.
    return camel_keys(
        {
            "csv_url": link(result.csv_path),
            "additional_csv_urls": {
                table: link(path) for table, path in result.additional_csv_paths.items()
            },
            "run_report_url": link(result.run_report_path),
            "run_summary_url": link(result.run_summary_path),
            "summary": camel_keys(summary),
            "warnings": list(result.warnings),
            "preview": result.preview_rows,
            "column_filter": (
                None
                if result.column_filter is None
                else camel_keys(result.column_filter.to_dict())
            ),
        }
    )


def _without_local_paths(
    result: DatasetBuildResult, uploads: Sequence[Path], working_directory: Path
) -> DatasetBuildResult:
    """Return the result with no local path left in its warnings.

    A warning can quote a file, as a parse error quotes the upload it could
    not read. An upload is named by its file name, as the user sent it, and
    any other path in the working directory by its path relative to it.
    """
    replacements = [(str(path), path.name) for path in uploads]
    for root in dict.fromkeys(
        [str(working_directory), str(working_directory.resolve())]
    ):
        replacements.extend([(root + os.sep, ""), (root, ".")])
    # A longer path goes first, so a file's own path wins over the directory
    # that contains it.
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    def scrubbed(text: str) -> str:
        for local, shown in replacements:
            text = text.replace(local, shown)
        return text

    return dataclasses.replace(
        result, warnings=[scrubbed(warning) for warning in result.warnings]
    )


class JobService:
    """The web app's background builds, and the state each job keeps.

    One service serves the whole app. Every method may be called from any
    request thread.
    """

    def __init__(
        self,
        *,
        working_directory: Path,
        dataset_directory: Path,
        max_pdb_ids: int | None,
        max_upload_files: int | None,
    ) -> None:
        """Keep the service's settings, without creating either directory."""
        self._working_directory = working_directory
        self._dataset_directory = dataset_directory
        self._max_pdb_ids = max_pdb_ids
        self._max_upload_files = max_upload_files
        # _lock guards _claims, _jobs, and every field of each job record.
        # _claims maps a run to the token of whatever holds it in this
        # process, either a job's worker or a claimed block. _jobs keeps each
        # job's record until its final state is on disk. Once a job starts,
        # only its own worker thread changes its record or writes its
        # job.json, so those writes land in order. Every read copies what it
        # needs under the lock and builds its answer after releasing it. A
        # result summary is never changed once set, so a copy may share it.
        # Files are written without _lock held. A method whose name ends in
        # _locked expects its caller to hold _lock.
        self._lock = threading.Lock()
        self._claims: dict[str, object] = {}
        self._jobs: dict[str, _Job] = {}

    @property
    def working_directory(self) -> Path:
        """Return the directory that holds the runs."""
        return self._working_directory

    @property
    def dataset_directory(self) -> Path:
        """Return the directory of the column-filtered tables."""
        return self._dataset_directory

    @property
    def max_pdb_ids(self) -> int | None:
        """Return the cap on distinct PDB IDs per request, or None if none."""
        return self._max_pdb_ids

    @property
    def max_upload_files(self) -> int | None:
        """Return the most files one upload may hold, or None for no cap."""
        return self._max_upload_files

    def _claim_locked(self, run_id: str) -> object:
        if run_id in self._claims:
            raise RunAlreadyActiveError(
                f"Run {run_id!r} is already being built or deleted by this server."
            )
        token = object()
        self._claims[run_id] = token
        return token

    def _release_locked(self, run_id: str, token: object) -> None:
        # A later claim of the run is not this holder's to release.
        if self._claims.get(run_id) is token:
            del self._claims[run_id]

    def _is_live_locked(self, run_id: str) -> bool:
        job = self._jobs.get(run_id)
        return job is not None and self._claims.get(run_id) is job.claim_token

    def live_run_ids(self) -> frozenset[str]:
        """Return the runs a job of this process is building."""
        with self._lock:
            return frozenset(
                run_id for run_id in self._jobs if self._is_live_locked(run_id)
            )

    def is_held(self, run_id: str) -> bool:
        """Return whether a worker holds the run, here or in another process.

        A run is held when a job or a ``claimed`` block holds it in this
        process, or when another process holds its lock. An invalid run_id
        raises ValueError, as does a run or runs directory that is a symbolic
        link. A lock file that cannot be opened raises OSError.
        """
        with self._lock:
            if run_id in self._claims:
                return True
        return open_run(self._working_directory, run_id).is_held()

    def in_memory_status(self, run_id: str) -> JobStatus | None:
        """Return the status of a job this process still keeps, or None.

        A job is kept while it runs. After it ends, it is kept until its final
        state is written to disk, where persisted_status reads it. A job that
        ended before it held the run never reaches disk. Its status is
        returned by one call, and the next call returns None.
        """
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            # Such a job was refused, as by another process's lock. Keeping
            # its error would hide that process's live status for good.
            if not job.holds_run and not self._is_live_locked(run_id):
                del self._jobs[run_id]
            return JobStatus(
                run_id=run_id,
                status=job.status,
                progress=job.progress,
                result=job.result,
                error=job.error,
            )

    def persisted_status(self, run_id: str) -> JobStatus | None:
        """Return a run's status as its files record it, or None.

        The status is the manifest's when it records one, and otherwise
        job.json's. Progress and result come from job.json. A run in error
        takes its error from the manifest when it records one, and otherwise
        from job.json. A job.json that is not a job record, or that holds a
        value of the wrong type, is ignored. None is returned when neither
        file gives a status. An invalid run_id raises ValueError, as does a
        run or runs directory that is a symbolic link.
        """
        run_state = open_run(self._working_directory, run_id)
        job = _persisted_job(run_state.read_job_state())
        manifest = read_json_object(run_state.directory.manifest_path) or {}
        recorded = manifest.get("run_status")
        if not _is_run_status(recorded):
            if job is None:
                return None
            recorded = job["status"]
        held = recorded == "running" and self.is_held(run_id)
        error: str | None = None
        if recorded == "error":
            manifest_error = manifest.get("error")
            if isinstance(manifest_error, str):
                error = manifest_error
            elif job is not None:
                error = job["error"]
        return JobStatus(
            run_id=run_id,
            status=effective_status(recorded, held=held),
            progress=None if job is None else job["progress"],
            result=None if job is None else job["result"],
            error=error,
        )

    def request_pause(self, run_id: str) -> bool:
        """Ask the job building the run to pause, and return whether one is.

        The build stops once the entry in hand is recorded, and it settles the
        run as paused. False means no job of this process is building the run.
        """
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None or job.status != "running":
                return False
            if not self._is_live_locked(run_id):
                return False
            job.cancel_event.set()
            return True

    def forget(self, run_id: str) -> None:
        """Drop the record of a job that has ended, as when its run is deleted.

        A job still building the run keeps its record.
        """
        with self._lock:
            if run_id in self._jobs and not self._is_live_locked(run_id):
                del self._jobs[run_id]

    @contextmanager
    def claimed(self, run_id: str) -> Generator[None]:
        """Hold the run in this process for the block, as a job would.

        No job can start on the run during the block, which is what a
        deletion needs. RunAlreadyActiveError is raised when a job or another
        block already holds the run.
        """
        with self._lock:
            token = self._claim_locked(run_id)
        try:
            yield
        finally:
            with self._lock:
                self._release_locked(run_id, token)

    def _persist(self, job: _Job, *, final: bool) -> bool:
        # Return whether job.json now holds the job's state. Progress is
        # written at most once a second, and a final state always.
        now = time.monotonic()
        with self._lock:
            if not job.holds_run:
                return False
            if (
                not final
                and job.persisted_at is not None
                and now - job.persisted_at < _PERSIST_MIN_INTERVAL_SECONDS
            ):
                return False
            job.persisted_at = now
            record = _PersistedJob(
                status=job.status,
                progress=job.progress,
                result=job.result,
                error=job.error,
            )
        try:
            open_run(self._working_directory, job.run_id).write_job_state(record)
        except (OSError, ValueError):
            # The job's state stays in memory, so a failed write loses
            # nothing until the server stops.
            _logger.warning(
                "Could not write the job state of run %r.",
                job.run_id,
                exc_info=True,
            )
            return False
        return True

    def _progress_reporter(self, job: _Job) -> ProgressCallback:
        def report(
            stage: ProgressStage, current: int, total: int, subject: str
        ) -> None:
            with self._lock:
                job.progress = ProgressEvent(
                    stage=stage, current=current, total=total, subject=subject
                )
                # The build reports progress only while it holds the run.
                job.holds_run = True
            self._persist(job, final=False)

        return report

    def _finish(
        self,
        job: _Job,
        status: RunStatus,
        *,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> bool:
        # Record the job's final state and return whether it is on disk.
        with self._lock:
            job.status = status
            job.result = result
            job.error = error
            # A build that returns a result or pauses has held the run.
            if status != "error":
                job.holds_run = True
        return self._persist(job, final=True)

    def _run(
        self,
        job: _Job,
        build: Callable[[_Job], DatasetBuildResult],
        cleanup: Callable[[], None] | None,
    ) -> None:
        # This is the worker thread's body. Every way out of it leaves the job
        # settled and the run released.
        persisted = False
        try:
            result = build(job)
        except RunPausedError:
            persisted = self._finish(job, "paused")
        except BaseException as exception:
            # Nothing above a worker thread could handle what it raises. Even
            # SystemExit therefore ends here, once the job is settled.
            _logger.exception("The build of run %r failed.", job.run_id)
            # This is the message settle_run records. It falls back to the
            # type's name for an exception that carries no text.
            message = str(exception) or type(exception).__name__
            persisted = self._finish(job, "error", error=message)
        else:
            persisted = self._finish(
                job,
                "paused" if result.paused else "completed",
                result=_result_payload(job.run_id, result),
            )
        finally:
            # The run stays claimed until its uploads are gone, so a job that
            # is no longer live has left nothing behind.
            if cleanup is not None:
                cleanup()
            with self._lock:
                self._release_locked(job.run_id, job.claim_token)
                # Once on disk, the job is served from there.
                if persisted and self._jobs.get(job.run_id) is job:
                    del self._jobs[job.run_id]

    def _start(
        self,
        run_id: str,
        build: Callable[[_Job], DatasetBuildResult],
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        # Claim the run and start its worker, or raise RunAlreadyActiveError.
        with self._lock:
            token = self._claim_locked(run_id)
            job = _Job(run_id=run_id, claim_token=token)
            self._jobs[run_id] = job
        # A build can take hours, and a daemon thread does not keep the server
        # from stopping meanwhile. A build stopped with the server leaves its
        # run stale. A resume continues it from its last recorded entry.
        thread = threading.Thread(
            target=self._run,
            args=(job, build, cleanup),
            name=f"pdbai-build-{run_id}",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            if cleanup is not None:
                cleanup()
            with self._lock:
                self._release_locked(run_id, token)
                if self._jobs.get(run_id) is job:
                    del self._jobs[run_id]
            raise

    def spawn_download_build(
        self,
        run_id: str,
        *,
        pdb_ids: Sequence[str],
        engine_options: EngineOptions,
        column_filter_mode: ColumnFilterMode | None,
        column_filter_list: Iterable[str] | None,
        resume: bool,
        search_result: SelectionResult | None = None,
    ) -> None:
        """Start a background build of run_id from PDB IDs.

        The arguments are those of build_atom_site_dataset. The tables stay in
        the run directory. When the run is held in this process, nothing
        starts and RunAlreadyActiveError is raised. Anything the build raises
        is recorded as the job's error instead. That includes RunLockedError
        for a run another process holds.
        """
        requested = list(pdb_ids)

        def build(job: _Job) -> DatasetBuildResult:
            result = build_atom_site_dataset(
                pdb_ids=requested,
                working_directory=self._working_directory,
                search_result=search_result,
                engine_options=engine_options,
                column_filter_mode=column_filter_mode,
                column_filter_list=column_filter_list,
                publish=False,
                run_id=run_id,
                resume=resume,
                progress_callback=self._progress_reporter(job),
                cancel_token=job.cancel_event,
            )
            return _without_local_paths(result, (), self._working_directory)

        self._start(run_id, build)

    def spawn_upload_build(
        self,
        run_id: str,
        *,
        mmcif_files: Sequence[Path],
        engine_options: EngineOptions,
        column_filter_mode: ColumnFilterMode | None,
        column_filter_list: Iterable[str] | None,
        resume: bool,
        upload_directory: Path,
    ) -> None:
        """Start a background build of run_id from uploaded mmCIF files.

        The arguments are those of build_atom_site_dataset_from_mmcif_files.
        upload_directory holds the files. Only the build needs them, and they
        may be unpublished data. The directory is therefore removed once the
        build ends, or at once when the thread cannot start. When the run is
        held in this process, nothing starts, nothing is removed, and
        RunAlreadyActiveError is raised. Anything the build raises is
        recorded as the job's error.
        """
        files = list(mmcif_files)

        def build(job: _Job) -> DatasetBuildResult:
            result = build_atom_site_dataset_from_mmcif_files(
                mmcif_files=files,
                working_directory=self._working_directory,
                engine_options=engine_options,
                column_filter_mode=column_filter_mode,
                column_filter_list=column_filter_list,
                publish=False,
                run_id=run_id,
                resume=resume,
                progress_callback=self._progress_reporter(job),
                cancel_token=job.cancel_event,
            )
            return _without_local_paths(result, files, self._working_directory)

        def cleanup() -> None:
            shutil.rmtree(upload_directory, ignore_errors=True)

        self._start(run_id, build, cleanup)


__all__ = [
    "ClientStatus",
    "JobService",
    "JobStatus",
    "ProgressEvent",
    "RunAlreadyActiveError",
    "effective_status",
]
