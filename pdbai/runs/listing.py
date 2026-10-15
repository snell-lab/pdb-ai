"""Summaries of the persisted runs in a working directory, for the Runs panel.

The panel polls every few seconds while builds write, so a poll reads as
little as it can. Each process caches what it read of every run's manifest
and entry states, and a poll reads only the entry updates logged since the
poll before, until the run's manifest changes.

A poll takes no lock on a run, so builds go on writing while it reads. The
manifest is replaced whole, and a poll reads either the old one or the new.
The entry and event logs are appended to in place, and a poll skips a line
that is not yet complete. A changed manifest means the log was folded into
it and emptied, so the cached run is then read again from the start.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

from ..json_values import object_dict, optional_object_dict, str_list
from .state import (
    FileStamp,
    apply_entry_updates,
    entries_of,
    file_stamp,
    read_json_object,
    run_table_names,
)
from .storage import RunDirectory, runs_root, validate_run_id

# The last progress event is read from the end of the event log, which keeps a
# poll's cost independent of how long the run has been going. An event is far
# shorter than this.
_EVENT_TAIL_BYTES = 8192


class RunCounts(TypedDict):
    """The number of a run's entries in each state, and in all."""

    done: int
    failed: int
    refused: int
    # Every entry not done, failed, or refused, whatever state it records.
    pending: int
    total: int


class RunSummary(TypedDict):
    """What the Runs panel shows of one run."""

    run_id: str
    # The status the manifest records. The run's process may have stopped
    # without recording another, so "running" says nothing of a live worker.
    status: str
    created_at: float | None
    # When the status last changed, or else when the run was created.
    updated_at: float | None
    settings: dict[str, object]
    pdb_count: int
    # How the identifiers were chosen, or None when the run recorded nothing.
    selection: dict[str, object] | None
    counts: RunCounts
    # The last complete progress event, or None when the log holds none.
    last_progress: dict[str, object] | None
    # Whether the primary table has been assembled.
    has_dataset: bool
    error: str | None
    # Whether a job of this process is building the run. The web app adds
    # it.
    live: NotRequired[bool]


@dataclass
class _CachedRun:
    """What a poll read of one run, kept for the next poll.

    ``states`` holds every entry's state. It starts from the manifest whose
    stamp is ``manifest_stamp``. The updates in the log that ``log_identity``
    names are then applied to it, up to byte ``log_offset``.
    """

    manifest_stamp: FileStamp
    log_identity: tuple[int, int] | None
    log_offset: int
    states: dict[str, str]
    run_id: str
    status: str
    created_at: float | None
    updated_at: float | None
    settings: dict[str, object]
    pdb_count: int
    selection: dict[str, object] | None
    primary_table: str
    error: str | None


# What the last poll read of each run, by run directory. Only a thread that
# holds _CACHE_LOCK reads or changes it.
_CACHE: dict[Path, _CachedRun] = {}
_CACHE_LOCK = threading.Lock()


def _timestamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _is_run_name(name: str) -> bool:
    try:
        validate_run_id(name)
    except ValueError:
        return False
    return True


def _read_last_event(events_path: Path) -> dict[str, object] | None:
    """Return the last complete event in the end of the event log.

    A last line still being written is passed over for the one before it. The
    result is None when the log is missing or empty, or its end holds no
    complete event.
    """
    try:
        with events_path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            offset = max(0, size - _EVENT_TAIL_BYTES)
            stream.seek(offset)
            tail = stream.read()
    except OSError:
        return None

    lines = tail.split(b"\n")
    if offset > 0:
        # Unless the read began at the start of the log, the first line may
        # be the end of a longer one.
        lines = lines[1:]
    for raw_line in reversed(lines):
        try:
            value: object = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        event = optional_object_dict(value)
        if event is not None:
            return event
    return None


def _read_run_from_manifest(
    run_directory: RunDirectory, log_identity: tuple[int, int] | None
) -> _CachedRun | None:
    # The run as its manifest alone gives it, before any logged update, or
    # None without a readable manifest.
    stamp = file_stamp(run_directory.manifest_path)
    manifest = read_json_object(run_directory.manifest_path)
    if stamp is None or manifest is None:
        return None
    created_at = _timestamp(manifest.get("created_at"))
    error = manifest.get("error")
    return _CachedRun(
        manifest_stamp=stamp,
        log_identity=log_identity,
        log_offset=0,
        states={
            pdb_id: str(entry.get("state") or "pending")
            for pdb_id, entry in entries_of(manifest).items()
        },
        run_id=str(manifest.get("run_id") or run_directory.root.name),
        status=str(manifest.get("run_status") or "unknown"),
        created_at=created_at,
        updated_at=_timestamp(manifest.get("status_updated_at")) or created_at,
        settings=object_dict(manifest.get("settings")),
        pdb_count=len(str_list(manifest.get("pdb_ids"))),
        selection=optional_object_dict(manifest.get("selection")),
        primary_table=run_table_names(manifest)[0],
        error=None if error is None else str(error),
    )


def _refreshed_run_locked(run_directory: RunDirectory) -> _CachedRun | None:
    """Return a run brought up to date, or None without a readable manifest.

    The log is opened before the manifest is looked at. A new log starts
    only after the manifest has absorbed the log before it. A manifest read
    after the log is opened therefore holds every update from earlier logs.
    The open log's own updates are applied from its start. An update the
    manifest already holds changes nothing when it is applied again.
    """
    try:
        log = run_directory.entries_path.open("rb")
    except OSError:
        log = None
    try:
        log_identity: tuple[int, int] | None = None
        log_size = 0
        if log is not None:
            status = os.fstat(log.fileno())
            log_identity = (status.st_dev, status.st_ino)
            log_size = status.st_size

        run = _CACHE.get(run_directory.root)
        if (
            run is None
            or run.manifest_stamp != file_stamp(run_directory.manifest_path)
            or run.log_identity != log_identity
            or run.log_offset > log_size
        ):
            run = _read_run_from_manifest(run_directory, log_identity)
            if run is None:
                _CACHE.pop(run_directory.root, None)
                return None
            _CACHE[run_directory.root] = run

        if log is not None:
            log.seek(run.log_offset)
            updates: dict[str, dict[str, object]] = {}
            run.log_offset += apply_entry_updates(log.read(), updates)
            for pdb_id, fields in updates.items():
                if "state" in fields:
                    run.states[pdb_id] = str(fields["state"] or "pending")
                else:
                    run.states.setdefault(pdb_id, "pending")
        return run
    finally:
        if log is not None:
            log.close()


def _counts(states: Mapping[str, str]) -> RunCounts:
    counted = Counter(states.values())
    settled = counted["done"] + counted["failed"] + counted["refused"]
    return RunCounts(
        done=counted["done"],
        failed=counted["failed"],
        refused=counted["refused"],
        pending=len(states) - settled,
        total=len(states),
    )


def list_run_summaries(working_directory: Path) -> list[RunSummary]:
    """Return a summary of every persisted run, the most recently active first.

    A run is placed by when its status last changed, so a resumed run comes
    before newer runs that have ended. A directory without a readable
    manifest is not listed. Neither is a symbolic link, nor a directory whose
    name is not a run ID, such as one being deleted. A runs directory that is
    itself a symbolic link raises ValueError.

    Each summary is the caller's to change, except for the values inside
    ``settings`` and ``selection``, which the cache shares.
    """
    root = runs_root(working_directory)
    run_roots = (
        [
            path
            for path in root.iterdir()
            if _is_run_name(path.name) and path.is_dir() and not path.is_symlink()
        ]
        if root.exists()
        else []
    )

    summaries: list[RunSummary] = []
    for run_root in run_roots:
        run_directory = RunDirectory(run_root)
        # Flask serves polls from several threads, and the cached run is
        # changed as it is refreshed, so one poll at a time reads a run.
        with _CACHE_LOCK:
            run = _refreshed_run_locked(run_directory)
            if run is None:
                continue
            summary = RunSummary(
                run_id=run.run_id,
                status=run.status,
                created_at=run.created_at,
                updated_at=run.updated_at,
                settings=dict(run.settings),
                pdb_count=run.pdb_count,
                selection=None if run.selection is None else dict(run.selection),
                counts=_counts(run.states),
                last_progress=_read_last_event(run_directory.events_path),
                has_dataset=run_directory.dataset_path_for(run.primary_table).exists(),
                error=run.error,
            )
        summaries.append(summary)

    with _CACHE_LOCK:
        # A run deleted since the last poll is forgotten.
        for cached_root in [path for path in _CACHE if path.parent == root]:
            if cached_root not in run_roots:
                del _CACHE[cached_root]

    summaries.sort(key=lambda summary: summary["updated_at"] or 0.0, reverse=True)
    return summaries


__all__ = ["RunCounts", "RunSummary", "list_run_summaries"]
