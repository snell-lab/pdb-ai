"""The routes that build a run from mmCIF uploads and filter a CSV's columns.

POST /api/build-upload checks the settings, the run, and every file's name
before it writes a byte, so a refused request leaves nothing behind. The run
is checked by the rule POST /api/build applies. Each file is then saved in a
numbered folder of its own, expanded when gzipped. Two files of one name
therefore stay apart, and the build numbers their entries by its own rule.
The bytes saved, after expansion, count against the request size cap, so a
small gzip file cannot fill the disk. The files may be unpublished data, so
only the server's user may read them. They are removed when the build ends
or the request fails. Upload folders that a stopped server left behind are
removed when the next one starts.

POST /api/filter-upload reads a CSV in one streaming pass. It writes the
chosen columns to a new file under a generated name, through
write_atomically, so the download link never serves a half-written table. A
CSV whose header or rows cannot be read as a table is refused, not guessed
at.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
import shutil
import uuid
import zlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import IO

from flask import Flask, abort, jsonify, request, url_for
from flask.typing import ResponseReturnValue
from werkzeug.datastructures import FileStorage

from ..atomic import write_atomically
from ..column_filtering import (
    ColumnFilterMode,
    kept_columns,
    missing_columns_warning,
    parse_column_filter,
    resolve_requested_columns,
)
from ..orchestration.upload import portable_upload_stem
from ..results import ColumnFilterSummary
from ..runs import PREVIEW_ROW_COUNT, validate_output_filename
from .jobs import JobService, RunAlreadyActiveError
from .options import parse_flag, settings_from_request
from .route_helpers import (
    bad_request,
    camel_keys,
    refuse_run_in_use,
    remove_stale_files,
    run_id_to_build,
)

_MMCIF_EXTENSIONS = (".cif", ".mmcif")
# The page's file picker accepts these suffixes, so it offers only files that
# this route will save.
ACCEPTED_UPLOAD_SUFFIXES: tuple[str, ...] = (
    *_MMCIF_EXTENSIONS,
    *(extension + ".gz" for extension in _MMCIF_EXTENSIONS),
)

_COPY_CHUNK_BYTES = 1024 * 1024

_UPLOADS_DIRECTORY_NAME = "uploads"

# A filtered table's stem keeps only ASCII letters, digits, underscores, and
# hyphens, which every file system and URL accepts as they are. The stem is
# cut short so that the whole name stays a valid output file name.
_UNSAFE_STEM_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_FILTERED_STEM_LENGTH = 100
# The filter route names each table <stem>_filtered_<32 hex digits>.csv. Only
# names of that form are served or removed as stale, so no other file in the
# directory is ever touched.
_FILTERED_TABLE_GLOB = "*_filtered_" + "[0-9a-f]" * 32 + ".csv"
_FILTERED_TABLE_NAME = re.compile(
    rf"[A-Za-z0-9_-]{{1,{_MAX_FILTERED_STEM_LENGTH}}}_filtered_[0-9a-f]{{32}}\.csv"
)


def is_filtered_table_name(filename: str) -> bool:
    """Return whether ``filename`` is a name the filter route gives a table."""
    return _FILTERED_TABLE_NAME.fullmatch(filename) is not None


def _saved_name(filename: str) -> tuple[str, bool]:
    """Return an upload's saved name, and whether the upload is gzipped.

    The name is the file's base name without any .gz suffix, and its stem is
    made portable by portable_upload_stem. A name that does not end in an
    mmCIF extension aborts with 400. So does a name whose stem has no ASCII
    letter or digit.
    """
    # A browser sends the bare name, but some clients send a whole path.
    # PureWindowsPath splits a path on both "/" and "\".
    name = PureWindowsPath(filename).name
    compressed = name.lower().endswith(".gz")
    if compressed:
        name = name[:-3]
    lowered = name.lower()
    matched = next(
        (known for known in _MMCIF_EXTENSIONS if lowered.endswith(known)), None
    )
    if matched is None:
        bad_request(
            f"Upload {filename!r} is not an mmCIF file; its name must end in "
            ".cif or .mmcif, optionally followed by .gz."
        )
    # The extension keeps the case the client gave it.
    extension = name[-len(matched) :]
    try:
        stem = portable_upload_stem(name[: -len(matched)])
    except ValueError:
        bad_request(
            f"Upload {filename!r} cannot be saved, because its name has no "
            "ASCII letter or digit before the extension."
        )
    return stem + extension, compressed


def _copy_within_budget(
    source: IO[bytes] | io.BufferedIOBase,
    destination: IO[bytes],
    budget: int | None,
) -> int | None:
    """Copy ``source`` to ``destination`` and return the ``budget`` left.

    ``budget`` is a number of bytes, and None allows any size. Copying more
    than ``budget`` bytes aborts with 413.
    """
    while chunk := source.read(_COPY_CHUNK_BYTES):
        if budget is not None:
            budget -= len(chunk)
            if budget < 0:
                abort(
                    413,
                    "The uploaded files expand to more than the "
                    f"{request.max_content_length} bytes a request may hold.",
                )
        destination.write(chunk)
    return budget


def _save_upload(
    upload: FileStorage, target: Path, *, compressed: bool, budget: int | None
) -> int | None:
    """Save ``upload`` at ``target``, expanded when ``compressed``.

    It returns what is left of ``budget``, and going past ``budget`` aborts
    with 413. A compressed file that is not valid gzip aborts with 400. An
    OSError from writing ``target`` is raised as it is. The client then sees a
    server error without the local path.
    """
    with target.open("wb") as destination:
        if not compressed:
            return _copy_within_budget(upload.stream, destination, budget)
        try:
            with gzip.GzipFile(fileobj=upload.stream, mode="rb") as source:
                return _copy_within_budget(source, destination, budget)
        except (gzip.BadGzipFile, EOFError, zlib.error):
            bad_request(f"Upload {upload.filename!r} is not a readable gzip file.")


def _filtered_columns(
    header: list[str], mode: ColumnFilterMode | None, requested: list[str]
) -> tuple[list[str], ColumnFilterSummary | None, list[str]]:
    """Return the columns the filter keeps, its summary, and its warnings.

    The columns follow kept_columns, as a build's do, with one exception. An
    include filter that matches nothing keeps every column, not only pdb_id,
    and adds a warning. The filtered copy is this tool's only output, so an
    empty one would be of no use. Without a filter, every column is kept and
    there is no summary.
    """
    if mode is None:
        return list(header), None, []
    warnings: list[str] = []
    matched, missing = resolve_requested_columns(header, requested)
    if matched:
        columns = kept_columns(header, mode, matched)
    else:
        columns = list(header)
        warnings.append(
            "No columns matched the requested column filter; dataset left unchanged."
        )
    if missing:
        warnings.append(missing_columns_warning(missing, header))
    summary = ColumnFilterSummary(
        mode=mode,
        matched_columns=matched,
        missing_columns=missing,
        requested_columns=requested,
        available_columns=list(header),
    )
    return columns, summary, warnings


def _checked_header(reader: Iterator[list[str]]) -> list[str]:
    """Return a CSV's header row, read from ``reader``.

    A CSV with no header aborts with 400. So does a header that names a column
    twice, since a filter could not tell which of the two columns it means.
    """
    header = next(reader, None)
    if not header:
        bad_request("The CSV file is empty; its first row must name the columns.")
    repeated = sorted(name for name, count in Counter(header).items() if count > 1)
    if repeated:
        bad_request(f"The CSV header names these columns more than once: {repeated!r}.")
    return header


@dataclass(frozen=True)
class _FilteredTable:
    """What writing a filtered copy of a CSV kept and found."""

    columns: list[str]
    summary: ColumnFilterSummary | None
    warnings: list[str]
    row_count: int
    preview: list[dict[str, str]]


def _write_filtered(
    source: io.TextIOWrapper,
    target: Path,
    mode: ColumnFilterMode | None,
    requested: list[str],
) -> _FilteredTable:
    """Write the filtered copy of the CSV ``source`` to ``target``.

    The result holds the kept columns, the filter's summary and warnings, the
    number of data rows, and the first rows as a preview. Blank lines are
    skipped. A row with more or fewer fields than the header aborts with 400,
    and the message names its line.
    """
    reader = csv.reader(source)
    header = _checked_header(reader)
    columns, summary, warnings = _filtered_columns(header, mode, requested)
    positions = [header.index(column) for column in columns]
    preview: list[dict[str, str]] = []
    row_count = 0
    with target.open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.writer(output_stream)
        writer.writerow(columns)
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                bad_request(
                    f"Line {reader.line_num} of the CSV file has {len(row)} fields; "
                    f"the header has {len(header)}."
                )
            selected = [row[position] for position in positions]
            writer.writerow(selected)
            if row_count < PREVIEW_ROW_COUNT:
                preview.append(dict(zip(columns, selected, strict=True)))
            row_count += 1
    return _FilteredTable(columns, summary, warnings, row_count, preview)


def _filtered_filename(filename: str) -> str:
    """Return a new, unique file name for the filtered copy of ``filename``."""
    # PureWindowsPath drops a folder written with either separator.
    stem = PureWindowsPath(filename).stem
    safe_stem = (
        _UNSAFE_STEM_CHARACTERS.sub("_", stem)[:_MAX_FILTERED_STEM_LENGTH].strip("_")
        or "dataset"
    )
    return validate_output_filename(f"{safe_stem}_filtered_{uuid.uuid4().hex}.csv")


def register_upload_routes(app: Flask, service: JobService) -> None:
    """Register the build-upload and filter-upload routes on ``app``.

    Registering also deletes the upload folders a stopped server left behind
    in the working directory of ``service``.
    """
    uploads_root = service.working_directory / _UPLOADS_DIRECTORY_NAME
    # Each build removes its own files. A folder here at startup was left by a
    # server stopped mid-build, and no build of this server can need it.
    shutil.rmtree(uploads_root, ignore_errors=True)

    @app.route("/api/build-upload", methods=["POST"])
    def build_upload() -> ResponseReturnValue:
        """Start a build from the form's mmCIF files, and return its run ID.

        The form's mmcifFiles parts are .cif or .mmcif files, each optionally
        gzipped. They come with the run settings that settings_from_request
        reads, and optionally runId and ``resume``. A part with no file name
        is ignored, since a browser sends one for an empty file field. A build
        that starts answers 200 with the run's ID as runId. With ``resume``
        true, runId names an upload run to continue. The same files must then
        be sent again, because the run keeps each file's name and fingerprint
        and refuses any other.

        The route answers 400 for no files, for more files than the server's
        cap, and for invalid settings. It answers 400 for a runId that is not
        a valid run ID, and for a ``resume`` that is not a flag or comes
        without a runId. It also answers 400 for a file name that is not an
        mmCIF name or has no ASCII letter or digit, and for a gzip file that
        cannot be read. It answers 413 when the request, or its files once
        expanded, pass the request size cap. Resuming a run that does not
        exist is a 404. A runId that names an existing run without ``resume``
        is a 409, and so is a run that is being built or deleted.
        """
        uploads = [
            (part, filename)
            for part in request.files.getlist("mmcifFiles")
            if (filename := part.filename)
        ]
        if not uploads:
            bad_request("At least one mmCIF file is required.")
        max_files = service.max_upload_files
        if max_files is not None and len(uploads) > max_files:
            bad_request(f"At most {max_files} files per upload; got {len(uploads)}.")
        try:
            settings = settings_from_request(request.form, source="upload")
            resume = parse_flag(request.form.get("resume"), "resume")
        except ValueError as exception:
            bad_request(exception)
        run_id = run_id_to_build(
            service.working_directory,
            request.form.get("runId"),
            resume=resume,
            is_held=service.is_held,
        )
        # Every name is checked before any file is written, so a bad name
        # leaves nothing to clean up.
        saved_names = [_saved_name(filename) for _part, filename in uploads]

        upload_directory = uploads_root / uuid.uuid4().hex
        try:
            # The files may be unpublished structures, so only the server's
            # own user may read them.
            upload_directory.mkdir(mode=0o700, parents=True)
            budget = request.max_content_length
            mmcif_files: list[Path] = []
            for index, ((upload, _filename), (name, compressed)) in enumerate(
                zip(uploads, saved_names, strict=True), start=1
            ):
                target = upload_directory / str(index) / name
                target.parent.mkdir()
                budget = _save_upload(
                    upload, target, compressed=compressed, budget=budget
                )
                mmcif_files.append(target)
            try:
                service.spawn_upload_build(
                    run_id,
                    mmcif_files=mmcif_files,
                    engine_options=settings.options,
                    column_filter_mode=settings.column_filter_mode,
                    column_filter_list=settings.column_filter_list,
                    resume=resume,
                    upload_directory=upload_directory,
                )
            except RunAlreadyActiveError:
                refuse_run_in_use(run_id)
        except BaseException:
            # Once the build has started, it owns the folder. Nothing after the
            # spawn can raise, so this never removes a folder the build owns.
            shutil.rmtree(upload_directory, ignore_errors=True)
            raise
        app.logger.info("Run %s started building %d uploads.", run_id, len(uploads))
        return jsonify({"runId": run_id})

    @app.route("/api/filter-upload", methods=["POST"])
    def filter_upload() -> ResponseReturnValue:
        """Write the column-filtered copy of the CSV in the form, and link it.

        The form holds a UTF-8 CSV as ``file``. It may also hold
        columnFilterMode, include or exclude, with columnFilterList, the names
        or globs to match. The answer holds the copy's link as csvUrl, and
        its ``summary`` of rowCount and columnCount. It also holds
        ``warnings``, the first rows as ``preview``, and the filter's summary
        as columnFilter, which is null without a filter. A filtered copy can
        be downloaded for at least an hour.

        The route answers 400 for no file, for an unknown mode, and for a mode
        or list given without the other. It also answers 400 for a CSV that is
        not UTF-8, cannot be parsed, has no header, repeats a column name, or
        has a row of the wrong length.
        """
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            bad_request("A CSV file is required.")
        try:
            mode, requested = parse_column_filter(
                request.form.get("columnFilterMode"),
                request.form.get("columnFilterList"),
                mode_key="columnFilterMode",
                list_key="columnFilterList",
            )
        except ValueError as exception:
            bad_request(exception)

        dataset_directory = service.dataset_directory
        remove_stale_files(dataset_directory, _FILTERED_TABLE_GLOB)
        output_path = dataset_directory / _filtered_filename(uploaded.filename)
        # The upload is read where Werkzeug spooled it. The table is therefore
        # never held in memory whole, or copied beside the served files.
        source = io.TextIOWrapper(uploaded.stream, encoding="utf-8", newline="")
        try:
            with write_atomically(output_path) as scratch:
                table = _write_filtered(source, scratch, mode, requested)
        except UnicodeDecodeError:
            bad_request("The CSV file must be UTF-8 text.")
        except csv.Error as exception:
            bad_request(f"The CSV file could not be parsed: {exception}.")

        response: dict[str, object] = {
            "csvUrl": url_for("download_filtered_table", filename=output_path.name),
            "summary": {"rowCount": table.row_count, "columnCount": len(table.columns)},
            "warnings": table.warnings,
            "preview": table.preview,
            "columnFilter": (
                None if table.summary is None else camel_keys(table.summary.to_dict())
            ),
        }
        return jsonify(response)


__all__ = [
    "ACCEPTED_UPLOAD_SUFFIXES",
    "is_filtered_table_name",
    "register_upload_routes",
]
