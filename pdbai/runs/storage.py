"""Run directory layout, run IDs and output filenames, and settings hashes.

Run IDs and output filenames arrive from the command line and the web app and
become file names, so each must be one name that every operating system
accepts, and the path it names must lie directly inside its directory.

Every rule of every operating system is checked wherever the code runs, so a
name accepted on one system is not refused when the working directory is
opened on another. A symbolic link is refused rather than followed, since
following it could lead a write out of the directory the name was checked
against. Names are limited in UTF-8 bytes rather than characters, because
file systems count bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..atomic import SCRATCH_NAME_EXTRA_BYTES

_RUNS_DIRECTORY_NAME = "runs"
# The device names Windows reserves, as ntpath.isreserved lists them. A file
# of such a name, with or without an extension, opens the device.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{device}{digit}" for device in ("COM", "LPT") for digit in "123456789¹²³"}
)
_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"/\\|?*')
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
# File systems allow a name 255 bytes, and a file written atomically needs a
# scratch file beside it whose name is longer than its own.
_MAX_FILENAME_BYTES = 255 - SCRATCH_NAME_EXTRA_BYTES
_OUTPUT_FILENAME_SUFFIX = ".csv"


def new_run_id() -> str:
    """Generate a unique run identifier."""
    return uuid.uuid4().hex


def settings_hash(settings: Mapping[str, object]) -> str:
    """Return a short hash of a run's settings, the same in every process.

    The order of keys does not change the hash. A value JSON cannot hold
    raises TypeError, since the text of such a value may differ from one
    process to the next.

    A run stores the hash and resumes only under settings that give the same
    one. Changing how settings are serialized here would therefore block
    resuming every stored run.
    """
    blob = json.dumps(dict(settings), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def is_windows_reserved_name(name: str) -> bool:
    """Return whether Windows takes a file name for a device.

    The extension does not matter, so NUL.csv names the NUL device too.
    """
    # Windows ignores spaces before the extension, so "NUL .csv" is the NUL
    # device as well.
    return name.partition(".")[0].rstrip(" ").upper() in _WINDOWS_RESERVED_NAMES


def _validate_portable_filename(value: str, *, label: str) -> str:
    """Return ``value`` when every operating system accepts it as a file name.

    A name that breaks a rule raises ValueError, in a message that starts with
    ``label`` and names the rule.
    """
    if not value:
        raise ValueError(f"{label} must not be empty.")
    if value in {".", ".."}:
        raise ValueError(f"{label} must be a single name, not {value!r}.")
    # Windows drops spaces and periods from the end of a name, so two names
    # that differ only there would be one file.
    if value != value.strip():
        raise ValueError(f"{label} must not start or end with whitespace.")
    if value.endswith("."):
        raise ValueError(f"{label} must not end with a period.")
    if any(character in _WINDOWS_INVALID_CHARACTERS for character in value):
        raise ValueError(
            f"{label} must not contain path separators or reserved filename characters."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} must not contain control characters.")
    if is_windows_reserved_name(value):
        raise ValueError(f"{label} uses a reserved filename.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{label} must be valid Unicode text.") from None
    if len(encoded) > _MAX_FILENAME_BYTES:
        raise ValueError(f"{label} must be at most {_MAX_FILENAME_BYTES} UTF-8 bytes.")
    return value


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` unchanged when it can name a run directory.

    A run ID is 1 to 128 ASCII letters, digits, periods, underscores, and
    hyphens, and starts with a letter or digit. Anything else raises
    ValueError, as does a name that is not a portable file name.
    """
    validated = _validate_portable_filename(run_id, label="Run ID")
    if _RUN_ID_PATTERN.fullmatch(validated) is None:
        raise ValueError(
            "Run ID must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, periods, underscores, or hyphens (maximum "
            "128 characters)."
        )
    return validated


def deletion_name(run_id: str) -> str:
    """Return a new name a run directory takes while it is deleted.

    The name starts with a period, which no run ID does, so it never names a
    run and no process can claim the run under it. An invalid ``run_id``
    raises ValueError.
    """
    return f".deleting-{validate_run_id(run_id)}-{uuid.uuid4().hex}"


def validate_output_filename(output_filename: str) -> str:
    """Return ``output_filename`` unchanged when it can name a published table.

    A name that is not a portable file name raises ValueError, as does one
    that does not end in .csv after at least one other character, or that
    starts with a period.
    """
    validated = _validate_portable_filename(output_filename, label="Output filename")
    # A leading period hides the file from a directory listing, which is where
    # a user looks for the dataset. The scratch files of atomic writes are
    # the hidden names here.
    if validated.startswith("."):
        raise ValueError("Output filename must not start with a period.")
    if (
        not validated.endswith(_OUTPUT_FILENAME_SUFFIX)
        or validated == _OUTPUT_FILENAME_SUFFIX
    ):
        raise ValueError(
            f"Output filename must be a name that ends in {_OUTPUT_FILENAME_SUFFIX}."
        )
    return validated


def _resolved_child_path(root: Path, name: str, *, label: str) -> Path:
    """Return the resolved path of ``name`` directly inside ``root``.

    ``root`` may itself be reached through symbolic links. A ``name`` that is
    a symbolic link raises ValueError. So does one that resolves to anywhere
    but directly inside ``root``, as a Windows junction can.
    """
    resolved_root = root.expanduser().resolve()
    unresolved_path = resolved_root / name
    if unresolved_path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link.")
    resolved_path = unresolved_path.resolve()
    if resolved_path.parent != resolved_root:
        raise ValueError(f"{label} must stay directly inside {resolved_root}.")
    return resolved_path


def runs_root(working_directory: Path) -> Path:
    """Return the directory that holds a working directory's runs.

    A runs directory that is a symbolic link raises ValueError.
    """
    return _resolved_child_path(
        working_directory,
        _RUNS_DIRECTORY_NAME,
        label="Runs directory",
    )


def run_directory_path(working_directory: Path, run_id: str) -> Path:
    """Return the directory of a run, which may not exist yet.

    A ``run_id`` that is not a valid run ID raises ValueError, as does a run
    directory or runs directory that is a symbolic link.
    """
    validated = validate_run_id(run_id)
    return _resolved_child_path(runs_root(working_directory), validated, label="Run ID")


def run_id_refusal(
    working_directory: Path, run_id: str, *, resume: bool
) -> Literal["missing", "exists"] | None:
    """Return why a build may not take ``run_id``, or None when it may.

    A resume needs the run, so it is refused as "missing" when no run
    directory is there. A build without resume must not name an existing run,
    so it is refused as "exists" when anything is at the run's path. A
    mistyped ID then neither starts a new run nor builds over a finished one.
    The command line and the web app both apply this rule, each in its own
    words. An invalid ``run_id`` raises ValueError, as does a run directory
    or runs directory that is a symbolic link.
    """
    path = run_directory_path(working_directory, run_id)
    if resume:
        return None if path.is_dir() else "missing"
    return "exists" if path.exists() else None


def output_file_path(working_directory: Path, output_filename: str) -> Path:
    """Return the path of a published table, directly in the working directory.

    An ``output_filename`` that is not a valid output filename raises
    ValueError. So does one that names a symbolic link, or anything else that
    is not a file, such as the runs directory.
    """
    validated = validate_output_filename(output_filename)
    path = _resolved_child_path(
        working_directory,
        validated,
        label="Output filename",
    )
    if path.exists() and not path.is_file():
        raise ValueError(f"Output filename {validated!r} names something not a file.")
    return path


@dataclass(frozen=True)
class RunDirectory:
    """The names of the files and folders inside one run directory."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        """Return the run manifest path."""
        return self.root / "manifest.json"

    @property
    def entries_path(self) -> Path:
        """Return the log of entry updates not yet folded into the manifest."""
        return self.root / "entries.jsonl"

    @property
    def events_path(self) -> Path:
        """Return the log of progress events, appended to as the run goes."""
        return self.root / "events.jsonl"

    @property
    def partials_directory(self) -> Path:
        """Return the directory of the per-structure partial CSV files."""
        return self.root / "partials"

    def partial_path_for(self, pdb_id: str, table: str) -> Path:
        """Return the path of one structure's partial CSV file for one table.

        Each table has a folder of its own, so a file name holds only the
        structure's identifier. Two partials of one table share a path only
        when their structures share an identifier.

        A ``pdb_id`` or ``table`` that is not a portable file name raises
        ValueError, so neither can name a path outside the partials directory.
        """
        _validate_portable_filename(table, label="Table name")
        _validate_portable_filename(pdb_id, label="Structure identifier")
        return self.partials_directory / table / f"{pdb_id}.csv"

    def dataset_path_for(self, table: str) -> Path:
        """Return the path of one table's assembled dataset.

        The flat shape's one table is the whole dataset, so its file is
        dataset.csv. Any other table's file is dataset_<table>.csv.
        """
        name = "dataset.csv" if table == "flat" else f"dataset_{table}.csv"
        return self.root / name

    @property
    def report_path(self) -> Path:
        """Return the run-report path."""
        return self.root / "run_report.csv"

    @property
    def summary_path(self) -> Path:
        """Return the Markdown run-summary path."""
        return self.root / "run_summary.md"

    @property
    def lock_path(self) -> Path:
        """Return the path of the file that marks the run as claimed."""
        return self.root / "run.lock"

    @property
    def job_state_path(self) -> Path:
        """Return the persisted web-job state path."""
        return self.root / "job.json"

    def ensure(self) -> None:
        """Create the run directory and its partials directory if needed."""
        self.partials_directory.mkdir(parents=True, exist_ok=True)

    def contained_partial_path(self, relative_path: object) -> Path | None:
        """Return the partial a manifest names, or None if it is not the run's.

        ``relative_path`` is the path a manifest entry records for a partial,
        relative to the run directory. A manifest may come from elsewhere, so
        the path is followed only to a file inside this run's partials
        directory. The result is None when the value is not text, the path
        leaves the partials directory, or there is no such file.
        """
        if not isinstance(relative_path, str) or not relative_path:
            return None
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.partials_directory.resolve()):
            return None
        return path if path.is_file() else None


__all__ = [
    "RunDirectory",
    "deletion_name",
    "is_windows_reserved_name",
    "new_run_id",
    "output_file_path",
    "run_directory_path",
    "run_id_refusal",
    "runs_root",
    "settings_hash",
    "validate_output_filename",
    "validate_run_id",
]
