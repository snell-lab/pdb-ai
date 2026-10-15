"""The public functions that build a dataset, from PDB IDs or from files.

Every front end starts a build here, the command line and the web app alike.
Each function checks all its arguments first. It then claims the run, so
that no other process can change it, and records the run's settings. Last,
it hands the entries to the download or upload code. It returns the finished
dataset's paths, counts, and warnings.

The arguments are checked before the claim, so a request that cannot run is
refused before it touches any run. A run stays claimed until it is settled.
A build that fails after the claim leaves the run in error, with the reason
recorded. Otherwise the run would be left running with no worker. The run
list would then show it as stale, with no reason given.

A run's settings are its column filter, its engine options, and for an
upload build the identity of every file. A run resumes only under the
settings it was created with.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from ..column_filtering import ColumnFilterMode, parse_column_filter
from ..dataset_shapes import output_table_names
from ..engines.catalog import get_engine_registry
from ..engines.compatibility import require_compatible_options
from ..engines.core import DatasetSource, EngineOptions
from ..identifiers import normalize_pdb_id
from ..integers import positive_int
from ..json_values import str_list
from ..results import DatasetBuildResult
from ..runs import (
    ProgressCallback,
    RunState,
    open_run,
    published_table_paths,
    validate_output_filename,
)
from ..schema import schema_fingerprint
from ..selection import SelectionResult
from .build import DatasetBuild
from .download import process_downloads
from .upload import identify_uploads, process_uploads


@dataclass(frozen=True)
class _BuildRequest:
    """The run arguments both builders share, checked and resolved."""

    source: DatasetSource
    working_directory: Path
    options: EngineOptions
    column_filter_mode: ColumnFilterMode | None
    column_filter_list: list[str]
    output_filename: str | None
    publish: bool
    run_id: str | None
    resume: bool
    progress_callback: ProgressCallback | None
    cancel_token: threading.Event | None


def _checked_request(
    source: DatasetSource,
    *,
    working_directory: Path,
    engine_options: EngineOptions | None,
    column_filter_mode: ColumnFilterMode | None,
    column_filter_list: Iterable[str] | None,
    output_filename: str | None,
    publish: bool,
    run_id: str | None,
    resume: bool,
    progress_callback: ProgressCallback | None,
    cancel_token: threading.Event | None,
) -> _BuildRequest:
    """Check a build's run arguments and resolve its engine options.

    Raises ValueError for an invalid argument, or for an engine the technique
    or the source rules out. An enabled engine that cannot run here raises
    its own preflight error.
    """
    if resume and run_id is None:
        raise ValueError("run_id must be provided when resume=True.")
    mode, values = parse_column_filter(column_filter_mode, column_filter_list)

    registry = get_engine_registry()
    options = registry.with_defaults(engine_options or EngineOptions())
    require_compatible_options(options, source=source)
    registry.preflight(options, source)

    working_directory = Path(working_directory).expanduser().resolve()
    if output_filename is not None:
        validate_output_filename(output_filename)
        if publish:
            # Every table's filename is checked before the build, not once
            # it has run and the tables are published.
            published_table_paths(
                working_directory,
                output_filename,
                output_table_names(options, source),
            )

    return _BuildRequest(
        source=source,
        working_directory=working_directory,
        options=options,
        column_filter_mode=mode,
        column_filter_list=values,
        output_filename=output_filename,
        publish=publish,
        run_id=run_id,
        resume=resume,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )


def _record_run_failure(run_state: RunState, exception: Exception) -> None:
    # Some exceptions, such as a bare AssertionError, carry no text. The
    # message then falls back to the type's name, as settle_run's does. The
    # failure being recorded is the one the caller must see, so a record that
    # cannot be written, as on a full disk, does not take its place.
    with suppress(OSError):
        run_state.set_run_status(
            "error", error=str(exception) or type(exception).__name__
        )


@contextmanager
def _claimed_run(
    request: _BuildRequest,
    pdb_ids: list[str],
    *,
    search_result: SelectionResult | None = None,
    upload_inputs: Sequence[Mapping[str, object]] | None = None,
    warnings: Sequence[str] = (),
) -> Generator[DatasetBuild]:
    """Claim the run, record its settings, and yield the build.

    The run stays claimed for the whole build. No other process can add
    partials to it or settle it meanwhile. RunLockedError is raised when
    another worker holds the run. ValueError is raised when the run exists
    with other settings or was written under another column schema. Either
    error leaves the run as it was.

    Settlement records the run's status itself. An exception that stops the
    build before that is raised as it is, and leaves the run in error with
    its message. An interruption such as KeyboardInterrupt is not an
    exception of that kind. It leaves the run running, so the run list shows
    it as stale.
    """
    settings: dict[str, object] = {
        "column_filter_mode": request.column_filter_mode,
        # The order is kept. An include filter lays out the first partial's
        # columns in the order the names were given, and a resumed run must
        # lay them out the same way.
        "column_filter_list": list(dict.fromkeys(request.column_filter_list)),
        **request.options.as_settings_dict(),
    }
    if upload_inputs is not None:
        # Readers take a run without a source to be a download run.
        settings["source"] = "upload"
        settings["upload_inputs"] = list(upload_inputs)

    run_state = open_run(request.working_directory, request.run_id)
    with run_state.claimed():
        registry = get_engine_registry()
        run_state.load_or_initialize(
            pdb_ids=pdb_ids,
            settings=settings,
            selection=search_result.to_dict() if search_result is not None else None,
            schema_fingerprint=schema_fingerprint(
                request.options, source=request.source
            ),
            reference_data=registry.reference_data(request.options, request.source),
        )
        try:
            yield DatasetBuild(
                run_state=run_state,
                working_directory=request.working_directory,
                # A resumed run keeps the identifiers its earlier builds named,
                # with any new ones after them, so the build and its settlement
                # cover the whole run and not only the identifiers given now.
                pdb_ids=str_list(run_state.manifest().get("pdb_ids")),
                options=request.options,
                output_filename=request.output_filename,
                progress_callback=request.progress_callback,
                resume=request.resume,
                cancel_token=request.cancel_token,
                publish=request.publish,
                source=request.source,
                column_filter_mode=request.column_filter_mode,
                column_filter_list=request.column_filter_list,
                warnings=list(warnings),
            )
        except Exception as exception:
            # settle_run sets the status whenever it stops, paused included,
            # so a run still running failed before settlement.
            if run_state.manifest().get("run_status") == "running":
                _record_run_failure(run_state, exception)
            raise


def _unique_pdb_ids(pdb_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    counts = Counter(normalize_pdb_id(pdb_id) for pdb_id in pdb_ids if pdb_id.strip())
    return list(counts), [pdb_id for pdb_id, count in counts.items() if count > 1]


def build_atom_site_dataset(
    pdb_ids: Iterable[str],
    working_directory: Path,
    *,
    search_result: SelectionResult | None = None,
    engine_options: EngineOptions | None = None,
    column_filter_mode: ColumnFilterMode | None = None,
    column_filter_list: Iterable[str] | None = None,
    output_filename: str | None = None,
    publish: bool = True,
    run_id: str | None = None,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_token: threading.Event | None = None,
    max_workers: int | None = None,
) -> DatasetBuildResult:
    """Build an atom-level dataset from PDB IDs, downloading each entry.

    Blank identifiers are skipped and the rest normalized. A repeated one is
    ignored with a warning. The build runs in the run named run_id, or in a
    new run when run_id is None. Each entry is recorded before the next is
    processed, so a stopped run can be continued. With ``resume``, which
    needs a run_id, the entries an earlier build left done are not processed
    again. Setting cancel_token stops the build once the entry in hand is
    recorded, and leaves the run paused.

    ``search_result`` is the archive search that produced pdb_ids. The
    manifest records it under ``selection`` as provenance of the list. max_workers bounds the parallel
    downloads, and None means DEFAULT_MAX_WORKERS. progress_callback is
    called for each stage with the stage, the position, the total, and the
    entry or file concerned.

    With ``publish``, the assembled tables are copied into the working
    directory, the primary table as output_filename. When output_filename is
    None, the run keeps the name it was last published under, or takes the
    first free default name. Without ``publish``, the result points at the
    tables in the run directory and output_filename is not used.

    ValueError is raised for an invalid argument before anything is written.
    It is also raised when the run exists with other settings, or was written
    under another column schema. An enabled engine that cannot run here
    raises its own preflight error. RunLockedError is raised when another
    worker holds the run. These errors leave every run as it was.

    Once the run is claimed, RunPausedError is raised when the build is
    cancelled before the run has a done entry. The run is then left paused.
    NoDatasetError is raised when the build finishes with no done entry. Any
    other error that stops the build, such as a table that cannot be
    published, is raised as it is. After NoDatasetError or any such error,
    the run is left in error with its message.
    """
    pdb_ids, repeated_ids = _unique_pdb_ids(pdb_ids)
    if not pdb_ids:
        raise ValueError("At least one valid PDB ID is required.")
    if max_workers is not None:
        max_workers = positive_int(max_workers, "max_workers")
    request = _checked_request(
        "download",
        working_directory=working_directory,
        engine_options=engine_options,
        column_filter_mode=column_filter_mode,
        column_filter_list=column_filter_list,
        output_filename=output_filename,
        publish=publish,
        run_id=run_id,
        resume=resume,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )
    warnings: list[str] = []
    if repeated_ids:
        warnings.append(
            "Duplicate PDB IDs were ignored after normalization: "
            f"{', '.join(repeated_ids)}."
        )

    with _claimed_run(
        request, pdb_ids, search_result=search_result, warnings=warnings
    ) as build:
        process_downloads(build, max_workers=max_workers)
        return build.finish()


def build_atom_site_dataset_from_mmcif_files(
    mmcif_files: Iterable[Path],
    working_directory: Path,
    *,
    engine_options: EngineOptions | None = None,
    column_filter_mode: ColumnFilterMode | None = None,
    column_filter_list: Iterable[str] | None = None,
    output_filename: str | None = None,
    publish: bool = True,
    run_id: str | None = None,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_token: threading.Event | None = None,
) -> DatasetBuildResult:
    """Build an atom-level dataset from mmCIF files, gzipped or not.

    Each file becomes an entry whose id is taken from its name, with a
    numeric suffix when another file's name gives the same id. The run's
    settings record every file's name, size, and SHA-256, so resuming the run
    requires the same files. A file that cannot be read raises OSError before
    the run is claimed. The other arguments, and the other errors raised, are
    as for build_atom_site_dataset.
    """
    mmcif_files = list(mmcif_files)
    if not mmcif_files:
        raise ValueError("At least one mmCIF file is required.")
    request = _checked_request(
        "upload",
        working_directory=working_directory,
        engine_options=engine_options,
        column_filter_mode=column_filter_mode,
        column_filter_list=column_filter_list,
        output_filename=output_filename,
        publish=publish,
        run_id=run_id,
        resume=resume,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )
    uploads = identify_uploads(mmcif_files)

    with _claimed_run(
        request,
        [upload.pdb_id for upload in uploads],
        upload_inputs=[upload.to_settings() for upload in uploads],
    ) as build:
        process_uploads(build, uploads)
        return build.finish()


__all__ = ["build_atom_site_dataset", "build_atom_site_dataset_from_mmcif_files"]
