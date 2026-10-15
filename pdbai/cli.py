"""The command line that builds a dataset from PDB identifiers or a search.

The command line is one of three front ends over the same build. The others
are the web app and the Python API. It turns its arguments into engine options
and a column filter with resolve_run_settings, which the web app uses too. It
then hands them to build_atom_site_dataset and builds nothing itself. With
--select, the identifiers come from one archive search. The search result goes
to the build with them, so the manifest records where the cohort came from.

The engine flags are not written by hand. Each setting the engine registry
declares gets a generated flag with the engine's own help text. So an engine
that code outside PDB-AI registers appears on the command line with no
change here.

Progress and the run summary go to standard output. Log messages go to
standard error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from .column_filtering import (
    COLUMN_FILTER_MODES,
    parse_column_filter,
    parse_column_filter_mode,
)
from .console_logging import configure_console_logging
from .engines.catalog import get_engine_registry
from .engines.core import (
    ALL_XRAY_SOURCES,
    DATASET_SHAPES,
    DEFAULT_DATASET_SHAPE,
    DEFAULT_TECHNIQUE,
    TECHNIQUE_IDS,
    TECHNIQUES,
    XRAY_SOURCE_IDS,
    XRAY_SOURCES,
    get_technique,
    get_xray_source,
)
from .identifiers import normalize_pdb_id
from .integers import parse_positive_int
from .orchestration import DEFAULT_MAX_WORKERS, build_atom_site_dataset
from .profiles import PROFILES, RunSettings, resolve_run_settings
from .results import DatasetBuildResult, NoDatasetError, PdbProcessResult
from .runs import (
    RunLockedError,
    RunNotResumableError,
    open_run,
    validate_output_filename,
    validate_run_id,
)
from .selection import ArchiveSearchError, Selection, SelectionResult, search_archive
from .selection_filters import SEARCH_FILTERS, UNIPROT_ACCESSIONS


def _argparse_type[T](parse: Callable[[str], T]) -> Callable[[str], T]:
    """Return an argparse type that reports the error message ``parse`` gives.

    A ValueError from ``parse`` becomes a usage error carrying its message.
    """

    def parse_argument(value: str) -> T:
        try:
            return parse(value)
        except ValueError as exception:
            raise argparse.ArgumentTypeError(str(exception)) from exception

    return parse_argument


def _positive_int_argument(flag: str) -> Callable[[str], int]:
    """Return a positive-integer argparse type whose errors name ``flag``."""
    return _argparse_type(partial(parse_positive_int, name=flag))


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the PDB identifiers and the archive search that can replace them."""
    parser.add_argument(
        "pdb_ids",
        nargs="*",
        # The build checks identifiers too. Checking them here makes a stray
        # word such as NONE a usage error with status 2, not a traceback.
        type=_argparse_type(normalize_pdb_id),
        help=(
            "One or more PDB identifiers (e.g. 5MCC 4HHB). Omit them and pass "
            "--select to take the identifiers from an archive search instead."
        ),
    )
    selection = parser.add_argument_group(
        "archive selection",
        "Instead of listing identifiers, ask the PDB for every current entry "
        "matching --technique and --xray-source, narrowed by any range filter "
        "below. One query to the RCSB Search API returns the cohort; the "
        "query, service, and time are recorded in the run manifest so the "
        "cohort can be reproduced. Every downstream source (PDB-Redo, "
        "AlphaFold, validation) is then looked up by PDB ID as usual, and "
        "intake still checks each entry's technique and source. A range "
        "filter narrows the search only.",
    )
    selection.add_argument(
        "--select",
        action="store_true",
        help=(
            "Select the PDB identifiers by searching the archive with the "
            "technique, X-ray source, and range filters, instead of listing "
            "them."
        ),
    )
    selection.add_argument(
        "--list-only",
        action="store_true",
        dest="list_only",
        help=(
            "With --select: print the matching PDB identifiers, one per line, to "
            "standard output and exit without building anything. The summary "
            "goes to standard error so the output can be piped or saved."
        ),
    )
    selection.add_argument(
        "--limit",
        type=_positive_int_argument("--limit"),
        default=None,
        help=(
            "With --select: keep at most this many matching identifiers, taken "
            "in sorted order so the sample is deterministic. The full match "
            "count is still reported and recorded."
        ),
    )
    # Each range filter is declared once in pdbai.selection_filters, and its
    # flag is generated here, so a new filter needs no change to this file.
    for search_filter in SEARCH_FILTERS:
        described = search_filter.help
        selection.add_argument(
            search_filter.cli_flag,
            type=_argparse_type(
                partial(search_filter.parse, name=search_filter.cli_flag)
            ),
            default=None,
            dest=search_filter.id,
            metavar=search_filter.quantity.unit_name.upper(),
            help=f"With --select: {described[:1].lower()}{described[1:]}",
        )
    described = UNIPROT_ACCESSIONS.help
    selection.add_argument(
        UNIPROT_ACCESSIONS.cli_flag,
        type=_argparse_type(_read_accessions),
        default=None,
        dest=UNIPROT_ACCESSIONS.id,
        metavar="FILE",
        help=(
            f"With --select: {described[:1].lower()}{described[1:]} FILE lists "
            "the accessions, one per line or separated by commas or spaces, "
            "such as the distinct uniprot_id values of another run's dataset."
        ),
    )


def _read_accessions(path_text: str) -> tuple[str, ...]:
    """Return the accessions listed in a file, or raise ValueError naming the flag."""
    flag = UNIPROT_ACCESSIONS.cli_flag
    try:
        text = Path(path_text).read_text(encoding="utf-8")
    except OSError as exception:
        raise ValueError(f"{flag} cannot read {path_text!r}: {exception}") from None
    return UNIPROT_ACCESSIONS.parse(text, f"{flag} ({path_text})")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags for what the run processes, where, and under which run."""
    parser.add_argument(
        "--technique",
        choices=TECHNIQUE_IDS,
        default=DEFAULT_TECHNIQUE,
        metavar="{" + ",".join(TECHNIQUE_IDS) + "}",
        help=(
            "Experimental technique this run processes; one per run. Entries "
            "the PDB records under any other method are refused at intake, "
            "before download, and listed as refused in the run report. Engines "
            "that do not apply to the technique cannot be enabled. Techniques: "
            + "; ".join(
                f"'{technique.id}' for {technique.display_label}"
                for technique in TECHNIQUES
            )
            + f". Default: {DEFAULT_TECHNIQUE}."
        ),
    )
    parser.add_argument(
        "--xray-source",
        choices=XRAY_SOURCE_IDS,
        default=ALL_XRAY_SOURCES,
        dest="xray_source",
        metavar="{" + ",".join(XRAY_SOURCE_IDS) + "}",
        help=(
            "Second-level filter for --technique x_ray_diffraction: the "
            "radiation source. 'all' (default) admits every X-ray entry. A "
            "specific source refuses entries reporting other sources, and "
            "entries recording no source at all, saying which. Sources: "
            + "; ".join(f"'{s.id}' for {s.display_label}" for s in XRAY_SOURCES)
            + ". Only valid with the X-ray diffraction technique."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd() / "workspace",
        help=(
            "Directory holding the downloads, the run directories with their "
            "per-structure partials, and the published output CSVs."
        ),
    )
    parser.add_argument(
        "--output",
        type=_argparse_type(validate_output_filename),
        default=None,
        help="Optional filename (not a path) for the generated CSV, ending in .csv.",
    )
    parser.add_argument(
        "--dataset-shape",
        choices=DATASET_SHAPES,
        default=DEFAULT_DATASET_SHAPE,
        dest="dataset_shape",
        help=(
            " ".join(
                f"'{shape.name}': {shape.description}"
                for shape in DATASET_SHAPES.values()
            )
            + " Star-schema tables are keyed for joining, so per-atom data isn't "
            "duplicated with every residue/chain/structure-wide value. "
            f"Default: {DEFAULT_DATASET_SHAPE}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int_argument("--workers"),
        default=None,
        help=(
            "Number of workers downloading entries' data in parallel, ahead of "
            f"processing. Defaults to min({DEFAULT_MAX_WORKERS}, number of PDB "
            "IDs). With 1, one entry downloads at a time."
        ),
    )
    profile_ids = [profile.id for profile in PROFILES]
    parser.add_argument(
        "--profile",
        choices=profile_ids,
        default=None,
        metavar="{" + ",".join(profile_ids) + "}",
        help=(
            "Apply a pre-defined column preset to the output dataset. Profiles: "
            + " ".join(f"'{profile.id}': {profile.description}" for profile in PROFILES)
            + " Switches on every engine whose columns the profile keeps. "
            "--column-filter-mode with --column-filter-list replaces the "
            "profile's column filter, but its engines are still switched on."
        ),
    )
    parser.add_argument(
        "--column-filter-mode",
        # argparse choices would refuse Include. The rule that the web app
        # and the library use ignores case, so it parses the mode instead.
        type=_argparse_type(parse_column_filter_mode),
        default=None,
        dest="column_filter_mode",
        metavar="{" + ",".join(COLUMN_FILTER_MODES) + "}",
        help=(
            "Manually include only specific columns ('include') or strip specific "
            "columns ('exclude') from the output; case is ignored. Use with "
            "--column-filter-list. Replaces the column filter of --profile, "
            "whose engines are still switched on."
        ),
    )
    parser.add_argument(
        "--column-filter-list",
        nargs="+",
        default=None,
        dest="column_filter_list",
        metavar="COLUMN",
        help=(
            "One or more column names to apply to --column-filter-mode. Each "
            "value may also hold several names separated by commas, "
            "semicolons, or line breaks. A name matches the column spelled the "
            "same, else one spelled the same ignoring case, else one with the "
            "same letters and digits, so '_cell_length_a' finds "
            "'_cell.length_a'. A '*' matches any run of characters, ignoring "
            "case: '*_factor' matches any column ending with '_factor'; "
            "'_cell.*' matches any column starting with '_cell.'. A name that "
            "matches nothing is ignored with a warning."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=_argparse_type(validate_run_id),
        default=None,
        dest="run_id",
        help=(
            "Optional identifier for this run. Used to name "
            "workspace/runs/<run-id>/, which holds the manifest, per-structure "
            "partials, and progress log. Without --resume it must name no "
            "existing run. Pair with --resume to continue an interrupted run. "
            "Must be one portable directory name, not a path. Defaults to a "
            "freshly generated UUID."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the existing run --run-id <ID> names, built with the same "
            "settings. PDB IDs it lacks are added to it. Each of its PDB IDs "
            "the manifest records as done, whose partial file for every table "
            "still exists in the run directory, is skipped; any other is "
            "processed again. The dataset covers the whole run."
        ),
    )


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one flag per setting the engines declare."""
    engines = parser.add_argument_group(
        "optional engines",
        "RCSB's standard mmCIF is always the base structure source. Everything "
        "below is an independently toggleable engine.",
    )
    # Each flag defaults to None, which means not given. The engine's declared
    # default then applies, not one from argparse. So the command line, the
    # web app, and the library start from the same settings.
    for option in get_engine_registry().option_specs():
        kind_arguments: dict[str, Any]
        match option.kind:
            case "flag":
                action = "store_false" if option.cli_negated else "store_true"
                kind_arguments = {"action": action}
            case "choice":
                kind_arguments = {"choices": option.choices}
            case "count":
                kind_arguments = {"type": _argparse_type(option.parse)}
        engines.add_argument(
            option.resolved_cli_flag,
            dest=option.name,
            default=None,
            help=option.help,
            **kind_arguments,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build analysis-ready structural biology datasets from PDB IDs."
    )
    _add_input_arguments(parser)
    _add_run_arguments(parser)
    _add_engine_arguments(parser)
    return parser


def _run_settings(args: argparse.Namespace) -> RunSettings:
    """Return the run's settings as resolve_run_settings settles them.

    A column filter given without its mode or without its names raises
    ValueError. So does any setting that resolve_run_settings refuses.
    """
    column_filter_mode, column_filter_list = parse_column_filter(
        args.column_filter_mode,
        args.column_filter_list,
        mode_key="--column-filter-mode",
        list_key="--column-filter-list",
    )
    return resolve_run_settings(
        {
            option.name: getattr(args, option.name)
            for option in get_engine_registry().option_specs()
        },
        technique=args.technique,
        xray_source=args.xray_source,
        dataset_shape=args.dataset_shape,
        profile_id=args.profile,
        column_filter_mode=column_filter_mode,
        column_filter_list=column_filter_list or None,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, RunSettings]:
    """Parse the command line and resolve the run's settings.

    A malformed PDB identifier is a usage error. So are flags that conflict,
    a setting that refines an engine the run leaves off, and settings that
    the technique or the profile rules out. Each one exits through argparse
    with status 2, before anything is searched or downloaded.
    """
    return _parse_args(_build_parser(), argv)


def _parse_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> tuple[argparse.Namespace, RunSettings]:
    """Parse argv with ``parser``, as parse_args describes."""
    args = parser.parse_args(argv)
    if args.select and args.pdb_ids:
        parser.error("give either PDB identifiers or --select, not both")
    if not args.select and not args.pdb_ids:
        parser.error("PDB identifiers are required unless --select is given")
    if args.list_only and not args.select:
        parser.error("--list-only requires --select")
    if args.limit is not None and not args.select:
        parser.error("--limit requires --select")
    for search_filter in SEARCH_FILTERS:
        if getattr(args, search_filter.id) is not None and not args.select:
            parser.error(f"{search_filter.cli_flag} requires --select")
    if getattr(args, UNIPROT_ACCESSIONS.id) is not None and not args.select:
        parser.error(f"{UNIPROT_ACCESSIONS.cli_flag} requires --select")
    if args.resume and args.run_id is None:
        parser.error("--resume requires --run-id")
    try:
        settings = _run_settings(args)
    except ValueError as exception:
        parser.error(str(exception))
    _refuse_refinements_of_flags_left_off(parser, args, settings)
    if args.select:
        # Filters that contradict each other, such as a minimum above its
        # maximum, are a usage error here, before the archive is asked.
        try:
            _selection(args)
        except ValueError as exception:
            parser.error(str(exception))
    return args, settings


def _refuse_refinements_of_flags_left_off(
    parser: argparse.ArgumentParser, args: argparse.Namespace, settings: RunSettings
) -> None:
    """Exit with a usage error when a setting refines a flag the run leaves off.

    Such a setting, like --sasa-points without --include-sasa, would change
    nothing, and a run that ignored it would look as if it had applied. A
    flag a profile turns on counts as on.
    """
    registry = get_engine_registry()
    for option in registry.option_specs():
        if option.refines is None or getattr(args, option.name) is None:
            continue
        if settings.options.flag(option.refines):
            continue
        refined = registry.option_spec(option.refines)
        flag = refined.resolved_cli_flag if refined is not None else option.refines
        parser.error(f"{option.resolved_cli_flag} requires {flag}")


def _run_id_problem(args: argparse.Namespace) -> str | None:
    """Return why --run-id cannot be built as asked, or None when it can.

    --resume needs the run to exist, and a run ID without --resume must not
    name one, so a mistyped ID neither starts a new run nor builds over a
    finished one. The web app refuses the same requests.
    """
    if args.run_id is None:
        return None
    try:
        exists = open_run(args.workdir, args.run_id).directory.root.exists()
    except ValueError as exception:
        return str(exception)
    if args.resume and not exists:
        return (
            f"run {args.run_id!r} does not exist in {args.workdir}; nothing to resume"
        )
    if not args.resume and exists:
        return (
            f"run {args.run_id!r} already exists in {args.workdir}; "
            "pass --resume to continue it or choose another --run-id"
        )
    return None


def _print_run_header(args: argparse.Namespace, status_stream: TextIO) -> None:
    technique = get_technique(args.technique)
    print(f"Technique: {technique.label} ('{technique.id}')", file=status_stream)
    if args.xray_source != ALL_XRAY_SOURCES:
        source = get_xray_source(args.xray_source)
        print(f"X-ray source: {source.label} ('{source.id}')", file=status_stream)


def _selection(args: argparse.Namespace) -> Selection:
    """Return the selection the parsed flags ask for, or raise ValueError."""
    given = {
        search_filter.id: getattr(args, search_filter.id)
        for search_filter in SEARCH_FILTERS
    }
    return Selection(
        technique=args.technique,
        xray_source=args.xray_source,
        limit=args.limit,
        bounds={name: value for name, value in given.items() if value is not None},
        uniprot_accessions=getattr(args, UNIPROT_ACCESSIONS.id) or (),
    )


def _search(args: argparse.Namespace, status_stream: TextIO) -> SelectionResult:
    """Search the archive for the run's cohort and report what it matched."""
    selection = _selection(args)
    print(f"Searching the PDB for: {selection.describe()}", file=status_stream)
    search_result = search_archive(selection)
    summary = (
        f"PDB search matched {search_result.total_count} entries "
        f"({search_result.service} at {search_result.retrieved_at_iso})"
    )
    if search_result.truncated:
        summary += (
            f"; keeping the first {len(search_result.pdb_ids)} (--limit {args.limit})"
        )
    print(summary + ".", file=status_stream)
    for warning in search_result.warnings:
        print(f"  - {warning}", file=status_stream)
    return search_result


def _option_label(name: str) -> str:
    declared = get_engine_registry().option_spec(name)
    return declared.label if declared is not None else name


def _print_profile_notes(settings: RunSettings) -> None:
    profile = settings.profile
    if profile is None:
        return
    technique = get_technique(settings.options.technique)
    if settings.uses_profile_filter:
        print(
            f"Applying column profile '{profile.name}': "
            f"{len(settings.column_filter_list or ())} columns "
            f"({profile.filter_mode})"
        )
    if settings.toggles_turned_on:
        print(
            f"Profile '{profile.name}' requires: "
            + ", ".join(_option_label(toggle) for toggle in settings.toggles_turned_on)
            + ". Enabling them automatically."
        )
    if settings.toggles_ruled_out:
        print(
            f"Profile '{profile.name}' also lists: "
            + ", ".join(_option_label(toggle) for toggle in settings.toggles_ruled_out)
            + f". Not available for {technique.label}, so left off."
        )


_REFUSED_AT_INTAKE = "Refused at technique intake"

_STAGE_LABELS = {
    "download": "Downloading data for",
    "intake": f"{_REFUSED_AT_INTAKE}:",
    "parse": "Parsing atom-site for",
}


def _report_progress(stage: str, current: int, total: int, subject: str) -> None:
    label = _STAGE_LABELS.get(stage, stage.title())
    print(f"[{current}/{total}] {label} {subject}")


def _intake_detail(item: PdbProcessResult) -> str:
    methods = "; ".join(item.experimental_methods) or "no method reported"
    sources = "; ".join(item.radiation_sources)
    return f"{methods} ({sources})" if sources else methods


@dataclass(frozen=True)
class _StatusGroup:
    """One kind of outcome that an entry can end a run with."""

    # Printed after the number in the summary line, as in "3 parse failed".
    count_label: str
    # Whether an entry belongs to this group.
    selects: Callable[[PdbProcessResult], bool]
    # Whether any entry in this group makes the command exit with status 2.
    fails_run: bool
    # The heading for this group's entries, such as "Parse failures", and the
    # message for each entry. None means the entries are counted but not
    # listed.
    listing: tuple[str, Callable[[PdbProcessResult], str]] | None


# Every outcome the run summary reports, in the order it prints them. An
# entry refused at intake does not fail the run, because refusing entries is
# what the technique filter is for. An entry whose metadata failed does not
# fail the run either.
_STATUS_GROUPS: tuple[_StatusGroup, ...] = (
    _StatusGroup(
        "refused at technique intake",
        PdbProcessResult.refused,
        fails_run=False,
        listing=(_REFUSED_AT_INTAKE, _intake_detail),
    ),
    _StatusGroup(
        "download failed",
        lambda item: item.download_succeeded() is False,
        fails_run=True,
        listing=("Download failures", lambda item: f"{item.download_error}"),
    ),
    _StatusGroup(
        "parse failed",
        lambda item: item.parse_succeeded() is False,
        fails_run=True,
        listing=("Parse failures", lambda item: f"{item.parse_error}"),
    ),
    # An entry that failed after its file was parsed failed at output. Intake
    # can also refuse an entry on what its parsed file reports. Such an entry
    # did not fail, so it counts only as refused.
    _StatusGroup(
        "output failed",
        lambda item: item.parse_succeeded() is True and item.state == "failed",
        fails_run=True,
        listing=("Output failures", lambda item: f"{item.output_error}"),
    ),
    _StatusGroup(
        "pending",
        lambda item: item.state == "pending",
        fails_run=True,
        listing=None,
    ),
    _StatusGroup(
        "metadata failed",
        lambda item: item.metadata_succeeded() is False and bool(item.metadata_message),
        fails_run=False,
        listing=("Metadata failures", lambda item: f"{item.metadata_message}"),
    ),
)


def _print_run_summary(result: DatasetBuildResult) -> int:
    """Print what the run produced and return its exit status, 0 or 2."""
    print(f"Generated dataset: {result.csv_path}")
    for table, path in result.additional_csv_paths.items():
        print(f"Generated {table} table: {path}")
    print(f"Run report: {result.run_report_path}")
    print(f"Run summary (toggles + data dictionary): {result.run_summary_path}")
    # The same count the web app shows: entries in the dataset, out of every
    # entry the run names, refused and failed ones included.
    in_dataset = sum(1 for item in result.pdb_results if item.succeeded())
    print(f"PDB entries in the dataset: {in_dataset} of {len(result.pdb_ids)}")
    print(f"Atom-site rows: {result.atom_count}")
    failed = False
    if result.pdb_results:
        groups = [
            (group, [item for item in result.pdb_results if group.selects(item)])
            for group in _STATUS_GROUPS
        ]
        print(
            "PDB status summary: "
            + ", ".join(f"{len(items)} {group.count_label}" for group, items in groups)
            + "."
        )
        for group, items in groups:
            if group.listing is None or not items:
                continue
            heading, detail = group.listing
            print(f"{heading}:")
            for item in items:
                print(f"  - {item.pdb_id}: {detail(item)}")
        failed = any(group.fails_run and items for group, items in groups)
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 2 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line on argv and return its exit status.

    The status is 2 when the archive search fails or matches nothing. It is
    also 2 when any entry failed to download, parse, or write its output, or
    was left pending, and when no entry was done, so no dataset was built.
    Otherwise it is 0. A usage error exits through argparse
    with status 2 before anything else happens. So does a --run-id that names
    no run to resume, or names an existing run without --resume. A run that
    another process holds, or that cannot be resumed with these settings,
    ends the command with status 2 and the reason. Other errors that the
    build raises are not caught, so they end the command with a traceback.
    """
    parser = _build_parser()
    args, settings = _parse_args(parser, argv)
    configure_console_logging()

    def fail(message: str) -> int:
        # This is the shape argparse gives a usage error. A script reading
        # standard error then meets one form, whichever check stopped it.
        print(f"{parser.prog}: error: {message}", file=sys.stderr)
        return 2

    # With --list-only, the identifiers on standard output are the result.
    # Status lines go to standard error, so redirecting standard output saves
    # only the identifiers.
    status_stream = sys.stderr if args.list_only else sys.stdout
    if not args.list_only:
        problem = _run_id_problem(args)
        if problem is not None:
            return fail(problem)
    _print_run_header(args, status_stream)

    search_result: SelectionResult | None = None
    if args.select:
        try:
            search_result = _search(args, status_stream)
        except ArchiveSearchError as exception:
            return fail(str(exception))
        if args.list_only:
            for pdb_id in search_result.pdb_ids:
                print(pdb_id)
            return 0
        if not search_result.pdb_ids:
            return fail("no PDB entries matched the selection")
        pdb_ids = list(search_result.pdb_ids)
    else:
        pdb_ids = list(args.pdb_ids)

    _print_profile_notes(settings)
    try:
        result = build_atom_site_dataset(
            pdb_ids=pdb_ids,
            working_directory=args.workdir,
            output_filename=args.output,
            max_workers=args.workers,
            column_filter_mode=settings.column_filter_mode,
            column_filter_list=settings.column_filter_list,
            progress_callback=_report_progress,
            run_id=args.run_id,
            resume=args.resume,
            engine_options=settings.options,
            search_result=search_result,
        )
    except (RunLockedError, RunNotResumableError) as exception:
        # Both are raised as the run is claimed, before anything is
        # downloaded, and leave the run as it was.
        return fail(str(exception))
    except NoDatasetError as exception:
        # Every entry was refused or failed. The run report is still written.
        return fail(str(exception))
    return _print_run_summary(result)


if __name__ == "__main__":
    raise SystemExit(main())
