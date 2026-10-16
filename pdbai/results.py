"""Entry and run outcomes, shared by the layers of a dataset build.

Each entry of a run is in one of the states ProcessState lists. A build
takes each entry it reaches out of pending once. A later build of the same
run leaves a done entry alone. It processes a failed or refused entry again
from the start. A refusal is the run's filter at work, not an error. A
refused entry does not change the command line's exit status, while a
failed or pending one does.

These types live in the package root because every layer of a build uses
them. Orchestration imports runs, and runs imports reporting. The types sit
below all three, so none of them imports a layer above it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeGuard, get_args

from .column_filtering import ColumnFilterMode
from .engines.core import ALL_DATASET_SOURCES, DatasetSource, ReportValue
from .json_values import object_dict, str_list


@dataclass
class ColumnFilterSummary:
    """What a column filter asked for and what it matched.

    ``requested_columns`` holds the names and globs the filter gave, and
    ``missing_columns`` those of them that matched no column.
    ``available_columns`` holds the columns the filter was applied to.
    """

    mode: ColumnFilterMode
    matched_columns: list[str]
    missing_columns: list[str]
    requested_columns: list[str]
    available_columns: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return the summary as a dict of JSON values."""
        return {
            "mode": self.mode,
            "matched_columns": list(self.matched_columns),
            "missing_columns": list(self.missing_columns),
            "requested_columns": list(self.requested_columns),
            "available_columns": list(self.available_columns),
        }


@dataclass
class MetadataFetchResult:
    """The outcome of fetching an entry's RCSB record."""

    metadata: dict[str, object]
    ok: bool
    message: str | None = None
    # The raw entry record the metadata was extracted from, for engines that
    # need more of it than the normalized fields.
    entry_record: dict[str, object] | None = None
    # Whether the entry record came from a fresh cache rather than the network.
    cached: bool = False


# Where an entry stands in the run. It is "pending" until a build records it,
# and "done" once its partial CSVs are written. It is "failed" when its
# download, parse, or annotation went wrong. It is "refused" when technique
# intake turned it away. Intake reads the RCSB record before anything is
# downloaded, and the parsed file before any annotation engine runs. The
# outcome flags on PdbProcessResult say how the entry reached its state.
type ProcessState = Literal["pending", "done", "failed", "refused"]

_PROCESS_STATES: frozenset[str] = frozenset(get_args(ProcessState.__value__))

# The outcome of technique intake. It is "matched" when the entry's reported
# methods include the run's technique. It is "refused" when they do not, or
# when its radiation source fails the run's X-ray source filter. It is
# "unverified" when the entry reports no method at all and is admitted
# anyway.
type TechniqueStatus = Literal["matched", "refused", "unverified"]

_TECHNIQUE_STATUSES: frozenset[str] = frozenset(get_args(TechniqueStatus.__value__))


# Helpers for PdbProcessResult.from_manifest.
def _is_process_state(value: object) -> TypeGuard[ProcessState]:
    return isinstance(value, str) and value in _PROCESS_STATES


def _is_technique_status(value: object) -> TypeGuard[TechniqueStatus]:
    return isinstance(value, str) and value in _TECHNIQUE_STATUSES


def _is_dataset_source(value: object) -> TypeGuard[DatasetSource]:
    return isinstance(value, str) and value in ALL_DATASET_SOURCES


def _report_value(value: object) -> ReportValue:
    return (
        value if value is None or isinstance(value, (bool, int, float, str)) else None
    )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


@dataclass
class PdbProcessResult:
    """What a build did with one entry, and where the entry stands.

    The fields here record the host's own steps, which are intake, download,
    parse, and output. Each engine's outcome is in ``engine_report``, keyed
    by the report fields the engines declare, so a new engine needs no field
    here. A step method answers None for a step the entry never reached or
    whose outcome was never recorded. That way a step that may not have been
    tried is never reported as failed.
    """

    pdb_id: str
    source: DatasetSource = "download"
    state: ProcessState = "pending"
    technique_status: TechniqueStatus | None = None
    technique_message: str | None = None
    experimental_methods: list[str] = field(default_factory=list[str])
    radiation_sources: list[str] = field(default_factory=list[str])
    download_ok: bool = False
    download_error: str | None = None
    parse_ok: bool = False
    parse_error: str | None = None
    output_error: str | None = None
    # None until a download's RCSB record is sought, and always None for an
    # upload, which never seeks one. An acquisition that raises leaves it
    # None, since the error may have come before the fetch.
    metadata_ok: bool | None = None
    metadata_message: str | None = None
    engine_table_row_counts: dict[str, int] = field(default_factory=dict[str, int])
    engine_report: dict[str, ReportValue] = field(
        default_factory=dict[str, ReportValue]
    )

    def metadata_succeeded(self) -> bool | None:
        """Return whether the RCSB record was fetched, or None if not known."""
        if self.source == "upload" or self.state == "pending":
            return None
        return self.metadata_ok

    def download_succeeded(self) -> bool | None:
        """Return whether the structure file arrived, or None if not tried."""
        if self.download_ok:
            return True
        if self.download_error is not None or self.state == "failed":
            return False
        return None

    def parse_succeeded(self) -> bool | None:
        """Return whether the structure file parsed, or None without a file."""
        return self.parse_ok if self.download_ok else None

    def refused(self) -> bool:
        """Return whether the entry was refused at technique intake."""
        return self.state == "refused"

    def succeeded(self) -> bool:
        """Return whether the entry is done, which is False while pending."""
        return self.state == "done"

    def to_manifest(self) -> dict[str, object]:
        """Return the fields a build records for this entry in the manifest."""
        return {
            "state": self.state,
            "source": self.source,
            "technique_status": self.technique_status,
            "technique_message": self.technique_message,
            "experimental_methods": list(self.experimental_methods),
            "radiation_sources": list(self.radiation_sources),
            "download_ok": self.download_ok,
            "download_error": self.download_error,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "output_error": self.output_error,
            "metadata_ok": self.metadata_ok,
            "metadata_message": self.metadata_message,
            "engine_report": dict(self.engine_report),
            "engine_table_row_counts": dict(self.engine_table_row_counts),
        }

    @classmethod
    def from_manifest(
        cls,
        pdb_id: str,
        entry: Mapping[str, object],
        *,
        default_source: DatasetSource,
    ) -> PdbProcessResult:
        """Rebuild the result of an entry from its manifest entry.

        The run report uses this for an entry the current build did not
        record. The entry is JSON from disk, so each value is checked and the
        read never raises. An unknown state reads as pending, and an unknown
        source as ``default_source``. An unknown technique status reads as
        None. A download or parse flag that is missing or not True reads as
        False, which is what a pending entry needs, since its manifest entry
        holds only its state. A metadata flag that is missing or not a
        Boolean reads as None, meaning unknown. A malformed engine report
        value becomes None, and a malformed table row count is dropped. A key
        this result has no field for is ignored. A resumed run does not use
        this result to decide what to skip. It trusts a stored done state
        only when a partial exists for every table the run writes.
        """
        raw_state = entry.get("state")
        state = raw_state if _is_process_state(raw_state) else "pending"
        raw_source = entry.get("source")
        # Every entry of a run comes from the one source its build reads.
        source = raw_source if _is_dataset_source(raw_source) else default_source
        raw_technique_status = entry.get("technique_status")
        technique_status = (
            raw_technique_status if _is_technique_status(raw_technique_status) else None
        )
        return cls(
            pdb_id=pdb_id,
            source=source,
            state=state,
            technique_status=technique_status,
            technique_message=_optional_string(entry.get("technique_message")),
            experimental_methods=str_list(entry.get("experimental_methods")),
            radiation_sources=str_list(entry.get("radiation_sources")),
            download_ok=entry.get("download_ok") is True,
            download_error=_optional_string(entry.get("download_error")),
            parse_ok=entry.get("parse_ok") is True,
            parse_error=_optional_string(entry.get("parse_error")),
            output_error=_optional_string(entry.get("output_error")),
            metadata_ok=_optional_bool(entry.get("metadata_ok")),
            metadata_message=_optional_string(entry.get("metadata_message")),
            engine_table_row_counts={
                table: count
                for table, count in object_dict(
                    entry.get("engine_table_row_counts")
                ).items()
                if isinstance(count, int) and not isinstance(count, bool)
            },
            engine_report={
                name: _report_value(value)
                for name, value in object_dict(entry.get("engine_report")).items()
            },
        )


@dataclass
class DatasetBuildResult:
    """The paths, counts, warnings, and previews a dataset build returns."""

    csv_path: Path
    run_report_path: Path
    run_summary_path: Path
    atom_count: int
    column_count: int
    pdb_ids: Sequence[str]
    preview_rows: list[dict[str, object]]
    warnings: list[str]
    pdb_results: list[PdbProcessResult] = field(default_factory=list[PdbProcessResult])
    column_filter: ColumnFilterSummary | None = None
    additional_csv_paths: dict[str, Path] = field(default_factory=dict[str, Path])
    # True when the build was cancelled before it recorded every entry. The
    # run is then paused, with a dataset of the entries done so far.
    paused: bool = False


class RunPausedError(RuntimeError):
    """The error raised when a run pauses before any entry is done.

    No dataset was assembled, so the caller has no result to report.
    """


class NoDatasetError(RuntimeError):
    """The error raised when a run finishes with no entry done.

    Every entry was refused or failed, so no dataset was assembled. The run
    report and summary are still written.
    """
