"""Processing, which takes one acquired entry to its recorded outcome.

Download and upload builds acquire an entry in different ways. Both hand it
over as an AcquiredEntry, so every step from here on is the same for both.
The entry's structure file is parsed and its technique checked, unless its
RCSB record settled that already. The entry is then annotated and recorded.

The stages that can fail are download, parse, and output. A failed stage
gives the entry an error for that stage, and the build moves on to the next
entry. The run report and the exit status read these errors, so a failed
stage is recorded rather than raised.

The technique check on the file comes before annotation, so no annotation
engine spends time on a structure the run would refuse.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..engines.core import (
    STRUCTURE_CIF_PATH,
    STRUCTURE_METADATA,
    EngineContext,
    EnginePhaseResult,
)
from ..results import PdbProcessResult
from ..structure import parse_mmcif_atom_rows
from ..structure.mmcif_parser import AtomSiteParseResult
from .annotation import AnnotatedStructure, annotate_structure
from .build import DatasetBuild
from .technique_intake import ReportedExperiment, decide_technique

type _FailureStage = Literal["download", "parse", "output"]


@dataclass(frozen=True)
class AcquiredEntry:
    """One entry with everything acquired that processing needs.

    A download arrives after its RCSB record was fetched and its acquisition
    engines ran. An upload arrives as its file. An entry refused or failed
    during acquisition arrives too, so its outcome is recorded the same way.
    """

    result: PdbProcessResult
    # The engine context that annotation goes on to use. It holds what the
    # host and the acquisition engines published, such as the file's path.
    context: EngineContext
    # Receipts of the files the entry was built from, for its manifest entry.
    input_receipts: tuple[dict[str, object], ...] = ()
    # Empty for an upload, which runs no acquisition engine.
    acquisition_result: EnginePhaseResult = field(default_factory=EnginePhaseResult)


class _StageError(Exception):
    def __init__(self, stage: _FailureStage, message: str) -> None:
        super().__init__(message)
        self.stage: _FailureStage = stage


def _structure_file(acquired: AcquiredEntry) -> Path:
    result = acquired.result
    if result.download_error:
        raise _StageError("download", result.download_error)
    cif_path = acquired.context.artifacts.get(STRUCTURE_CIF_PATH)
    if cif_path is None:
        raise _StageError(
            "download", f"No structure file was acquired for {result.pdb_id}."
        )
    return cif_path


def _parse(acquired: AcquiredEntry, cif_path: Path) -> AtomSiteParseResult:
    result = acquired.result
    try:
        parsed = parse_mmcif_atom_rows(
            cif_path,
            acquired.context.artifacts.get(STRUCTURE_METADATA) or {},
            structure_id=result.pdb_id,
        )
    except RuntimeError as exception:
        raise _StageError("parse", str(exception)) from exception
    result.parse_ok = True
    return parsed


def _annotate(
    acquired: AcquiredEntry, parsed: AtomSiteParseResult
) -> AnnotatedStructure:
    pdb_id = acquired.result.pdb_id
    if parsed.dataframe.empty:
        raise _StageError(
            "output", f"Parsed atom-site DataFrame was empty for {pdb_id}."
        )
    try:
        return annotate_structure(parsed.dataframe, acquired.context)
    except Exception as exception:
        # Whatever annotation raises, such as a required engine's failure,
        # fails this entry's output and not the build.
        raise _StageError(
            "output", f"Annotation failed for {pdb_id}: {exception}"
        ) from exception


def _record_refusal(build: DatasetBuild, acquired: AcquiredEntry) -> None:
    result = acquired.result
    build.report_progress("intake", result.pdb_id)
    build.record_failed_or_refused(result, input_receipts=acquired.input_receipts)


def _record_failure(
    build: DatasetBuild, acquired: AcquiredEntry, stage: _FailureStage, message: str
) -> None:
    result = acquired.result
    result.state = "failed"
    match stage:
        case "download":
            result.download_error = message
        case "parse":
            result.parse_error = message
        case "output":
            result.output_error = message
    build.warnings.append(message)
    build.record_failed_or_refused(result, input_receipts=acquired.input_receipts)


def process_entry(build: DatasetBuild, acquired: AcquiredEntry) -> None:
    """Take one acquired entry through every remaining stage and record it.

    A refusal is recorded against the entry, and so is a failure to acquire,
    parse, annotate, or split it into tables. Neither is raised, so one entry
    cannot end the build. An OSError writing the entry's partials or its
    manifest entry is raised, since it would fail every later entry too.
    """
    result = acquired.result
    if result.refused():
        _record_refusal(build, acquired)
        return
    try:
        cif_path = _structure_file(acquired)
        build.report_progress("parse", result.pdb_id)
        # Only a download whose RCSB record could not be fetched carries a
        # message. The file in hand still lets the entry be processed.
        if result.metadata_message:
            build.warnings.append(result.metadata_message)
        # The previous entry is unreachable by now, and this one is about to
        # hold a whole structure. Collecting here stops cyclic garbage that
        # holds DataFrames from piling up over a long build.
        gc.collect()
        parsed = _parse(acquired, cif_path)
        build.warnings.extend(parsed.warnings)
        if result.technique_status in (None, "unverified"):
            # There was no archive record, or it named no method, so the file
            # decides.
            decide_technique(
                result.pdb_id,
                ReportedExperiment.from_parsed_file(parsed),
                build.options,
            ).apply_to(result)
            if result.refused():
                _record_refusal(build, acquired)
                return
        annotated = _annotate(acquired, parsed)
    except _StageError as error:
        _record_failure(build, acquired, error.stage, str(error))
        return

    phase_results = acquired.acquisition_result.merged(annotated.phase_result)
    result.engine_report = dict(phase_results.report)
    build.warnings.extend(
        f"Engine warning for {result.pdb_id}: {message}"
        for message in phase_results.warnings
    )
    try:
        build.record_done(result, annotated, input_receipts=acquired.input_receipts)
    except (ValueError, KeyError) as exception:
        # These come from this entry's rows, such as a star-schema split
        # that finds no residue author identity column or a table with no
        # columns to write. An OSError is the run's and is not caught. A
        # KeyError's text is the repr of its message, so the message itself
        # is used.
        detail = (
            exception.args[0]
            if isinstance(exception, KeyError) and exception.args
            else exception
        )
        _record_failure(
            build,
            acquired,
            "output",
            f"Writing the tables failed for {result.pdb_id}: {detail}",
        )


__all__ = ["AcquiredEntry", "process_entry"]
