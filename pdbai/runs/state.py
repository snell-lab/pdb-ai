"""A run's persisted record, and the lock that claims the run.

The record is the run's manifest, the log of entry updates not yet folded
into the manifest, its progress events, and the web app's job state. One
process at a time claims a run and writes its record through a RunState. Any
process may read the record meanwhile, without the lock.

Each entry a build finishes changes the record, and a run may hold many
thousands of entries. Rewriting the whole manifest for each would make a
build's cost grow with the square of its size, so an entry update is
appended to entries.jsonl instead. The manifest holds everything else and is
rewritten only when the rest of the record changes. Each rewrite folds the
logged updates into the manifest and empties the log. A reader of the
entries applies the log to the manifest it read.

The manifest and the job state are replaced whole through an atomic write,
so a reader sees the old file or the new one. The entry log and the event
log are appended to in place, and their readers skip a line that is not yet
complete, whether it is still being written or was cut off by a crash. A
reader therefore sees each update in full or not at all.

The claim is an operating-system lock on run.lock, not a file whose presence
marks the run as taken. The lock ends with the process that holds it, even
one that crashed, so no stale claim has to be cleared by hand. A run still
recorded as running with no holder is therefore known to be stale.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Literal, NamedTuple

from ..atomic import write_text_atomically
from ..engines.core import (
    DATASET_SHAPES,
    DEFAULT_DATASET_SHAPE,
    is_engine_table_name,
    parse_dataset_shape,
)
from ..json_values import object_dict, object_list, optional_object_dict, str_list
from ..provenance import capture_execution_session
from .storage import RunDirectory, new_run_id, run_directory_path, settings_hash

# The stages a run reports progress in. The first three are reported for each
# entry during a build, and the last two for the whole dataset while it is
# assembled and published.
type ProgressStage = Literal["download", "intake", "parse", "assemble", "finalize"]

# Receives each progress report, giving its stage, the position reached, the
# total, and its subject: the ID of the entry in hand while building, a
# partial's file name or "dataset" while assembling, and "dataset" or the
# published file's name while publishing.
type ProgressCallback = Callable[[ProgressStage, int, int, str], None]

type RunStatus = Literal["running", "paused", "completed", "error"]

# A file's inode, modification time, and size. Each atomic write of a file
# gives it a new inode, so the stamp changes even when the other two do not.
type FileStamp = tuple[int, int, int]


class RunLockedError(RuntimeError):
    """The error raised on claiming a run that another worker holds."""


class RunNotResumableError(ValueError):
    """The error raised on loading a stored run that cannot be built again.

    Its manifest is unreadable or incomplete, or the run was created with
    other settings, or its done entries were written under another column
    schema.
    """


class RunRecord(NamedTuple):
    """A run's manifest and entries as read from its directory."""

    # The manifest without its entries.
    manifest: dict[str, Any]
    # Every entry with the logged updates applied, or None when the manifest
    # holds no entries at all.
    entries: dict[str, dict[str, Any]] | None


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Return the JSON object a file holds, or None when it holds none.

    A file that is missing, unreadable, not UTF-8, not JSON, or JSON other
    than an object gives None.
    """
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return optional_object_dict(loaded)


def manifest_text(manifest: Mapping[str, object]) -> str:
    """Return the text of a manifest file that holds ``manifest``."""
    return json.dumps(manifest, default=str, indent=2, sort_keys=True) + "\n"


def file_stamp(path: Path) -> FileStamp | None:
    """Return a file's stamp, or None when the file cannot be reached."""
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_ino, status.st_mtime_ns, status.st_size)


def apply_entry_updates(log_bytes: bytes, entries: dict[str, dict[str, Any]]) -> int:
    """Apply logged entry updates to ``entries`` and return the bytes read.

    ``log_bytes`` is a stretch of the entry log that begins at the start of a
    line. ``entries`` is changed in place. The count returned runs to the end
    of the last complete line.

    Only complete lines are applied, so a line still being appended, or cut
    off by a crash, is left for a later read. A line that is not an update is
    skipped. Applying an update a second time changes nothing.
    """
    end = log_bytes.rfind(b"\n") + 1
    for raw_line in log_bytes[:end].splitlines():
        try:
            record: object = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        update = optional_object_dict(record)
        if update is None:
            continue
        pdb_id = update.get("pdb_id")
        if isinstance(pdb_id, str):
            entries.setdefault(pdb_id, {}).update(object_dict(update.get("fields")))
    return end


def entries_of(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the per-structure entries a manifest holds."""
    return {
        pdb_id: object_dict(entry)
        for pdb_id, entry in object_dict(manifest.get("entries")).items()
    }


def read_run_record(directory: RunDirectory) -> RunRecord | None:
    """Return a run's record as it stands, or None without a readable manifest.

    The run may be written while it is read. A writer replaces the manifest
    and then empties the log, so a manifest read before that and a log read
    after it would lack the updates in between. The manifest is therefore read
    again whenever it changed while the log was read. A log that cannot be
    read is taken to hold no updates.
    """
    while True:
        stamp = file_stamp(directory.manifest_path)
        manifest = read_json_object(directory.manifest_path)
        if manifest is None:
            return None
        if not isinstance(manifest.get("entries"), Mapping):
            manifest.pop("entries", None)
            return RunRecord(manifest, None)
        entries = entries_of(manifest)
        try:
            log = directory.entries_path.read_bytes()
        except OSError:
            log = b""
        apply_entry_updates(log, entries)
        if file_stamp(directory.manifest_path) == stamp:
            del manifest["entries"]
            return RunRecord(manifest, entries)


def error_message(exception: BaseException) -> str:
    """Return the text a run records for the error that stopped it.

    Some exceptions, such as a bare AssertionError, carry no text. The text
    is then the name of the exception's type.
    """
    return str(exception) or type(exception).__name__


def run_table_names(manifest: Mapping[str, Any]) -> list[str]:
    """Return the tables a run writes, primary first, from its manifest.

    A manifest may come from elsewhere, and table names become file paths.
    An engine table is therefore listed only under a name an engine could
    declare.
    """
    settings = object_dict(manifest.get("settings"))
    try:
        shape = parse_dataset_shape(settings.get("dataset_shape"))
    except ValueError:
        shape = DEFAULT_DATASET_SHAPE
    names = list(DATASET_SHAPES[shape].tables)
    for record in object_list(manifest.get("engine_tables")):
        name = object_dict(record).get("name")
        if isinstance(name, str) and is_engine_table_name(name) and name not in names:
            names.append(name)
    return names


# Windows has no flock, so the lock there is a byte-range lock through
# msvcrt.
if sys.platform == "win32":
    import msvcrt

    def _try_lock(stream: IO[str]) -> bool:
        # The lock covers one byte at the stream's position, which must be
        # the byte _unlock releases.
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(stream: IO[str]) -> None:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(stream: IO[str]) -> bool:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(stream: IO[str]) -> None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _append_execution_session(
    manifest: dict[str, Any],
    selection: Mapping[str, object] | None,
    reference_data: Mapping[str, object] | None,
) -> None:
    # Every execution of a run, the first and each resume, is a session of
    # its own, so the provenance names every environment that wrote the run.
    provenance = object_dict(manifest.get("provenance"))
    sessions = object_list(provenance.get("execution_sessions"))
    session = capture_execution_session(reference_data=reference_data)
    if selection is not None:
        session["selection"] = dict(selection)
    sessions.append(session)
    provenance["execution_sessions"] = sessions
    manifest["provenance"] = provenance


class RunState:
    """One handle on a run's record, for one thread at a time to use.

    The handle keeps the manifest and the entries in memory once it has read
    them. Every change is made there and persisted before the call returns.
    Only the process that claimed the run with ``claimed`` may
    change it. A method that changes the record raises RuntimeError on a
    handle that has not yet read or initialized it.
    """

    def __init__(self, run_directory: RunDirectory) -> None:
        """Make a handle on a run that has not read its record yet."""
        self.directory = run_directory
        # The manifest without its entries, and the entries, which change far
        # more often and are persisted by appending to a log.
        self._manifest: dict[str, Any] | None = None
        self._entries: dict[str, dict[str, Any]] = {}

    def _loaded_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            raise RuntimeError(
                f"Run {self.directory.root.name!r} is not loaded; call "
                "load_or_initialize first."
            )
        return self._manifest

    def _snapshot(self) -> dict[str, Any]:
        if self._manifest is None:
            return {}
        return {
            **self._manifest,
            "entries": {pdb_id: dict(entry) for pdb_id, entry in self._entries.items()},
        }

    def _write_manifest(self) -> None:
        # Writing the manifest folds every logged entry update into it and then
        # empties the log. After a crash between the two, the log still holds
        # updates the manifest already has, and applying them again gives the
        # same entries.
        write_text_atomically(
            self.directory.manifest_path, manifest_text(self._snapshot())
        )
        if self.directory.entries_path.exists():
            write_text_atomically(self.directory.entries_path, "")

    def _resumed_record(
        self,
        pdb_ids: list[str],
        requested_settings_hash: str,
        schema_fingerprint: str,
    ) -> RunRecord:
        # The stored record with pdb_ids added, ready to run again. Nothing is
        # written, and a record that cannot be resumed raises
        # RunNotResumableError.
        name = self.directory.root.name
        record = read_run_record(self.directory)
        if record is None:
            raise RunNotResumableError(f"Run {name!r} has an unreadable manifest.")
        manifest, entries = record
        stored_settings_hash = manifest.get("settings_hash")
        if not isinstance(stored_settings_hash, str) or not stored_settings_hash:
            raise RunNotResumableError(f"Run {name!r} has an invalid settings hash.")
        if optional_object_dict(manifest.get("settings")) is None or entries is None:
            raise RunNotResumableError(f"Run {name!r} has an incomplete manifest.")
        if stored_settings_hash != requested_settings_hash:
            raise RunNotResumableError(
                f"Run {name!r} was created with different settings; "
                "use a new run_id instead of mixing incompatible partial datasets."
            )
        stored_fingerprint = manifest.get("schema_fingerprint")
        if stored_fingerprint != schema_fingerprint:
            if any(entry.get("state") == "done" for entry in entries.values()):
                raise RunNotResumableError(
                    f"Run {name!r} was written under a different column schema; "
                    "start a new run instead of mixing its partials with new ones."
                )
            # An entry that stopped before it was done may have recorded the
            # columns it offered, which belong to the schema the run leaves.
            manifest.pop("filter_available_columns", None)
            manifest.pop("engine_tables", None)
            manifest["schema_fingerprint"] = schema_fingerprint

        ordered = str_list(manifest.get("pdb_ids"))
        known = set(ordered)
        for pdb_id in pdb_ids:
            entries.setdefault(pdb_id, {"state": "pending"})
            if pdb_id not in known:
                ordered.append(pdb_id)
                known.add(pdb_id)
        manifest["pdb_ids"] = ordered
        # A resumed run is running again, so an error from an earlier attempt
        # no longer describes it.
        manifest["run_status"] = "running"
        manifest["status_updated_at"] = time.time()
        manifest.pop("error", None)
        return RunRecord(manifest, entries)

    def _new_record(
        self,
        pdb_ids: list[str],
        settings: Mapping[str, object],
        requested_settings_hash: str,
        selection: Mapping[str, object] | None,
        schema_fingerprint: str,
    ) -> RunRecord:
        manifest: dict[str, Any] = {
            "run_id": self.directory.root.name,
            "created_at": time.time(),
            "settings_hash": requested_settings_hash,
            "settings": dict(settings),
            "pdb_ids": pdb_ids,
            "run_status": "running",
            "schema_fingerprint": schema_fingerprint,
        }
        if selection is not None:
            manifest["selection"] = dict(selection)
        return RunRecord(manifest, {pdb_id: {"state": "pending"} for pdb_id in pdb_ids})

    def load_or_initialize(
        self,
        pdb_ids: Iterable[str],
        settings: Mapping[str, object],
        selection: Mapping[str, object] | None = None,
        *,
        schema_fingerprint: str,
        reference_data: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Load the run's manifest, or initialize it when the run is new.

        Either way the run is left running, with this execution added to its
        provenance, and the manifest is written. A stored run gains the
        identifiers in ``pdb_ids`` that it did not have, as pending entries,
        and loses the error of an earlier attempt. A snapshot of the manifest
        is returned, entries included.

        A stored manifest that is unreadable or incomplete raises
        RunNotResumableError, as does one created with other settings. A
        setting whose value JSON cannot hold raises TypeError.

        ``selection`` records how the identifiers were chosen from the
        archive. It says where the list came from and is not a setting, so it
        takes no part in the settings hash. A stored run keeps the selection
        it was created with, and the new one is recorded with this execution.
        ``reference_data`` holds receipts for the data files the run's
        engines ship with, and is recorded with this execution too.

        ``schema_fingerprint`` identifies the columns and tables the code
        writes. A run with done entries whose manifest records another
        fingerprint raises RunNotResumableError, because its partials would
        not assemble with new ones. A run without done entries takes the new
        fingerprint instead, and forgets the columns and engine tables its
        entries recorded.

        Nothing is written when any of these errors is raised.
        """
        pdb_list = list(pdb_ids)
        requested_settings_hash = settings_hash(settings)
        if self.directory.manifest_path.exists():
            record = self._resumed_record(
                pdb_list, requested_settings_hash, schema_fingerprint
            )
        else:
            record = self._new_record(
                pdb_list,
                settings,
                requested_settings_hash,
                selection,
                schema_fingerprint,
            )
        _append_execution_session(record.manifest, selection, reference_data)
        self.directory.ensure()
        self._manifest = record.manifest
        self._entries = record.entries or {}
        self._write_manifest()
        return self._snapshot()

    @contextmanager
    def claimed(self) -> Generator[None]:
        """Hold the run for this process until the block ends.

        Raise RunLockedError when another process, or another handle in this
        one, already holds the run, as when the command line and the web app
        resume the same run at once. The run directory is created if needed.
        """
        self.directory.ensure()
        with self.directory.lock_path.open("a+") as stream:
            if not _try_lock(stream):
                raise RunLockedError(
                    f"Run {self.directory.root.name!r} is already being processed "
                    "by another worker."
                )
            try:
                yield
            finally:
                _unlock(stream)

    def is_held(self) -> bool:
        """Return whether some worker holds the run at this moment.

        The lock is tried without waiting and released at once. The run
        directory is never created, so a run without one is not held. An flock
        belongs to one open file, and a Windows lock to one handle. A worker
        of this same process that holds the run through another handle is
        therefore reported as holding it. A worker that tries to claim the run
        in the instant the probe holds the lock is refused, as if another
        worker held it. A lock file that exists but cannot be opened raises
        OSError.
        """
        try:
            stream = self.directory.lock_path.open("r+")
        except (FileNotFoundError, NotADirectoryError):
            return False
        with stream:
            if not _try_lock(stream):
                return True
            _unlock(stream)
            return False

    def manifest(self) -> dict[str, Any]:
        """Return a snapshot of the run manifest, entries included.

        A handle that has not read the record yet reads it from the run
        directory. The snapshot is {} when the run has no readable manifest.
        """
        if self._manifest is None:
            record = read_run_record(self.directory)
            if record is None:
                return {}
            self._manifest = record.manifest
            self._entries = record.entries or {}
        return self._snapshot()

    def set_run_status(self, status: RunStatus, *, error: str | None = None) -> None:
        """Set and persist the run status, and the error that stopped the run.

        Without an error, the one an earlier status recorded is cleared.
        """
        manifest = self._loaded_manifest()
        manifest["run_status"] = status
        manifest["status_updated_at"] = time.time()
        if error is None:
            manifest.pop("error", None)
        else:
            manifest["error"] = error
        self._write_manifest()

    def update_entry(self, pdb_id: str, **fields: object) -> None:
        """Update and persist one structure entry.

        The update is appended to the entry log and flushed to disk before the
        call returns, so it survives a crash from then on.
        """
        self._loaded_manifest()
        update = {**fields, "updated_at": time.time()}
        self._entries.setdefault(pdb_id, {}).update(update)
        line = json.dumps({"pdb_id": pdb_id, "fields": update}, default=str)
        with self.directory.entries_path.open("ab+") as stream:
            # An earlier write that failed partway may have left a line
            # without its end, which this line must not be joined to.
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write(line.encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def set_output_receipts(self, receipts: list[dict[str, object]]) -> None:
        """Store the run's output receipts, replacing those stored before."""
        manifest = self._loaded_manifest()
        provenance = object_dict(manifest.get("provenance"))
        provenance["outputs"] = receipts
        manifest["provenance"] = provenance
        self._write_manifest()

    def set_published_filename(self, filename: str) -> None:
        """Store the filename the run's primary table was published under."""
        manifest = self._loaded_manifest()
        if manifest.get("published_filename") == filename:
            return
        manifest["published_filename"] = filename
        self._write_manifest()

    def published_filename(self) -> str | None:
        """Return the filename the run was last published under, if it was."""
        filename = self.manifest().get("published_filename")
        return filename if isinstance(filename, str) else None

    def set_engine_tables(self, records: list[dict[str, object]]) -> None:
        """Store the description of every engine table the run writes."""
        manifest = self._loaded_manifest()
        manifest["engine_tables"] = records
        self._write_manifest()

    def add_available_columns(self, columns: Iterable[str]) -> None:
        """Record columns a structure offered to the column filter.

        The columns not yet recorded follow those that were, in the order
        given. The manifest is written only when the record changes.
        """
        manifest = self._loaded_manifest()
        existing = manifest.get("filter_available_columns")
        merged = list(dict.fromkeys([*str_list(existing), *columns]))
        if merged == existing:
            return
        manifest["filter_available_columns"] = merged
        self._write_manifest()

    def available_columns(self) -> list[str]:
        """Return every column any structure offered to the column filter."""
        return str_list(self.manifest().get("filter_available_columns"))

    def done_pdb_ids(self, required_tables: Iterable[str] = ()) -> set[str]:
        """Return the identifiers of the entries that need no more work.

        An entry counts when it is recorded done and every partial it records
        is a file in the run's partials directory. It must also record a
        partial for each of ``required_tables``.
        """
        required = list(required_tables)
        completed: set[str] = set()
        for pdb_id, entry in entries_of(self.manifest()).items():
            if entry.get("state") != "done":
                continue
            partial_paths = object_dict(entry.get("partial_paths"))
            if not partial_paths:
                continue
            if any(table not in partial_paths for table in required):
                continue
            if all(
                self.directory.contained_partial_path(relative_path) is not None
                for relative_path in partial_paths.values()
            ):
                completed.add(pdb_id)
        return completed

    def report_progress(
        self,
        callback: ProgressCallback | None,
        stage: ProgressStage,
        current: int,
        total: int,
        subject: str,
    ) -> None:
        """Log a progress event about ``subject`` and pass it to ``callback``.

        ``subject`` names what the stage is working on, as ProgressCallback
        describes.
        """
        event = {
            "ts": time.time(),
            "stage": stage,
            "current": current,
            "total": total,
            "subject": subject,
        }
        with self.directory.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        if callback:
            callback(stage, current, total, subject)

    def write_job_state(self, state: Mapping[str, object]) -> None:
        """Persist the current web-job state in place of the stored one.

        Writes through different handles are not put in order, so the last to
        finish wins.
        """
        write_text_atomically(
            self.directory.job_state_path, json.dumps(state, default=str)
        )

    def read_job_state(self) -> dict[str, Any] | None:
        """Return the stored web-job state, or None when none can be read."""
        return read_json_object(self.directory.job_state_path)


def open_run(working_directory: Path, run_id: str | None = None) -> RunState:
    """Return a handle on a run, which need not exist yet.

    A ``run_id`` of None gives a new run under a new identifier. An invalid
    run ID raises ValueError, as does a run directory or runs directory that
    is a symbolic link. Nothing is read or written. The run
    directory is created when the run is claimed or initialized.
    """
    effective_run_id = new_run_id() if run_id is None else run_id
    return RunState(
        RunDirectory(run_directory_path(working_directory, effective_run_id))
    )


__all__ = [
    "FileStamp",
    "ProgressCallback",
    "ProgressStage",
    "RunLockedError",
    "RunNotResumableError",
    "RunRecord",
    "RunState",
    "RunStatus",
    "apply_entry_updates",
    "entries_of",
    "error_message",
    "file_stamp",
    "manifest_text",
    "open_run",
    "read_json_object",
    "read_run_record",
    "run_table_names",
]
