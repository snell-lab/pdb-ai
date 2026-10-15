"""The two AlphaFold Database engines and the settings that turn them on.

The work is split between two phases. The acquisition engine runs once the
RCSB structure is downloaded. It looks the entry up in RCSB and in the
AlphaFold Database, and publishes the result. The annotation engine runs
once the atom table exists. It maps that result onto the residues and
reports the entry's status.

Two switches turn the engines on. include_alphafold asks for everything,
models and pLDDT included. include_uniprot alone asks only for the UniProt
mapping, so no model is fetched. The uniprot_id column is written whenever
the engines run, because a model status means nothing without the accession
it belongs to.

An uploaded structure has no RCSB entry, so the acquisition engine does not
run for it. The annotation engine still runs and gives the upload's protein
atoms the status not_available_for_upload. When the acquisition engine fails
for an entry, the annotation engine gives its protein atoms lookup_failed.
A lookup that never ran or that failed then reads neither as a missing
model nor as atoms the engine left alone.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

from ...core import (
    ENTRY_RECORD,
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    PROFILE_STRUCTURE_QUALITY,
    ColumnGranularity,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    ReportValue,
    SourceFile,
    column_specs,
)
from ..http_cache import read_envelope
from ..rcsb.metadata import rcsb_entry_url
from .annotation import annotate_atom_dataframe
from .artifacts import AFDB_ENTRY
from .fetch import fetch_afdb_entry, unavailable_upload_result
from .models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_LOOKUP_FAILED,
    SEQUENCE_IDENTITY_LEVELS,
    AFDBEntryResult,
    discovery_cache_path,
    model_cache_path,
    sequence_cluster_column,
)
from .references import RCSB_POLYMER_ENTITY_URL

_ANNOTATION_ENGINE_NAME = "alphafold_database"


def _uses_afdb(options: EngineOptions) -> bool:
    return options.flag("include_alphafold") or options.flag("include_uniprot")


def _fetches_models(options: EngineOptions) -> bool:
    # The UniProt flag alone asks only for the mapping. Turning the engine on
    # by name asks for everything it writes, as its own flag does.
    return (
        options.flag("include_alphafold")
        or _ANNOTATION_ENGINE_NAME in options.enabled_engines
    )


def _whenever_the_engine_runs(_options: EngineOptions) -> bool:
    # Every run of the engine maps UniProt accessions, so the column's toggle
    # only says which flag it is attributed to.
    return True


class AlphaFoldAcquisitionEngine:
    """The engine that looks an entry up in RCSB and the AlphaFold Database."""

    spec = EngineSpec(
        name="alphafold_database_source",
        display_name="AlphaFold Database source",
        phase=EnginePhase.ACQUISITION,
        toggle="include_alphafold",
        enabled_if=_uses_afdb,
        dependencies=("rcsb_source",),
        provides=(AFDB_ENTRY,),
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Look the entry up and publish the result under AFDB_ENTRY.

        The source files are the cache files that hold what the lookup used.
        """
        services = context.require_services()
        cache_directory = services.cache_root / "afdb"
        started_at = time.time()
        result = fetch_afdb_entry(
            context.pdb_id,
            session=services.session,
            timeout=services.timeout,
            cache_directory=cache_directory,
            fetch_models=_fetches_models(context.options),
            entry_json=context.artifacts.get(ENTRY_RECORD),
        )
        return EngineResult(
            artifacts={AFDB_ENTRY: result},
            source_files=_cached_source_files(result, cache_directory, started_at),
        )


def _cached_before(path: Path, started_at: float) -> bool | None:
    # A cache file the lookup wrote is dated after it started, and one it
    # read was dated before. A file without a readable date is not known.
    envelope = read_envelope(path)
    cached_at = envelope.get("cached_at") if envelope is not None else None
    if not isinstance(cached_at, (int, float)):
        return None
    return cached_at < started_at


def _cached_source_files(
    result: AFDBEntryResult, cache_directory: Path, started_at: float
) -> tuple[SourceFile, ...]:
    """Return the cache files that hold the parts of the result they cache.

    A file is listed when the file exists and its part of the result is one
    the lookup caches, by the rules references.py and model_fetch.py give.
    A file an earlier run left beside a partial or failed result is
    therefore not listed, since this lookup neither read nor wrote it. Each
    file says whether it was cached before ``started_at``, the time the
    lookup began, and so was read rather than written.
    """
    source_files: list[SourceFile] = []
    discovery_path = discovery_cache_path(cache_directory, result.pdb_id)
    discovery_cached = (
        result.discovery_complete
        and bool(result.references)
        and not result.unmapped_protein_entity_ids
    )
    if discovery_cached and discovery_path.is_file():
        # The mapping comes from the entry record and the record of each
        # protein entity.
        pdb_lower = result.pdb_id.lower()
        source_urls = [
            rcsb_entry_url(result.pdb_id),
            *(
                f"{RCSB_POLYMER_ENTITY_URL}{pdb_lower}/{quote(entity_id, safe='')}"
                for entity_id in result.protein_entity_ids
            ),
        ]
        source_files.append(
            SourceFile(
                "rcsb_polymer_entity_mapping_cache",
                discovery_path,
                "; ".join(source_urls),
                _cached_before(discovery_path, started_at),
            )
        )
    for accession, model in sorted(result.models.items()):
        model_path = model_cache_path(cache_directory, accession)
        model_cached = model.status == AFDB_STATUS_AVAILABLE and model.has_scores
        if not model_cached or not model_path.is_file():
            continue
        source_files.append(
            SourceFile(
                "alphafold_model_cache",
                model_path,
                "; ".join(model.source_urls) or None,
                _cached_before(model_path, started_at),
            )
        )
    return tuple(source_files)


class AlphaFoldAnnotationEngine:
    """The engine that maps an entry's AlphaFold lookup onto its residues."""

    spec = EngineSpec(
        name=_ANNOTATION_ENGINE_NAME,
        display_name="AlphaFold Database",
        phase=EnginePhase.ANNOTATION,
        toggle="include_alphafold",
        enabled_if=_uses_afdb,
        dependencies=("alphafold_database_source",),
        options=(
            OptionSpec(
                "include_alphafold",
                label="AlphaFold Database",
                caption=(
                    "Model availability, provenance, pLDDT coverage, and "
                    "per-residue pLDDT, using entity-aware UniProt alignments."
                ),
                help=(
                    "Fetch AlphaFold Database model availability, provenance, "
                    "pLDDT coverage, and per-residue pLDDT. Off by default. "
                    "Entity-aware UniProt mapping and uniprot_id provenance are "
                    "included regardless of --include-uniprot."
                ),
                enabled_text=(
                    "AlphaFold Database model availability, model/UniProt "
                    "provenance, per-entity pLDDT coverage, and per-residue "
                    "pLDDT were fetched."
                ),
            ),
            OptionSpec(
                "include_uniprot",
                label="UniProt accession lookup",
                caption=(
                    "Entity-aware UniProt accessions and PDB-to-UniProt alignments "
                    "(uniprot_id) from RCSB."
                ),
                help=(
                    "Fetch entity-aware UniProt accessions and PDB-to-UniProt "
                    "alignments (uniprot_id column) from RCSB. Off by default."
                ),
                enabled_text=(
                    "Entity-aware UniProt accessions and PDB-to-UniProt alignments "
                    "were fetched."
                ),
                summary_override=lambda options: (
                    "**include_uniprot** (`False`): standalone UniProt output was "
                    "not requested, but `uniprot_id` is retained as required AFDB "
                    "mapping/provenance context."
                    if options.flag("include_alphafold")
                    and not options.flag("include_uniprot")
                    else None
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "afdb_status",
                "AlphaFold Database lookup status for the entry.",
                on_failure=AFDB_STATUS_LOOKUP_FAILED,
                in_summary=True,
                label="AFDB",
            ),
            ReportField(
                "afdb_message",
                "Why the AlphaFold Database lookup fell short, when it did.",
                carries_message=True,
            ),
        ),
        columns=(
            *column_specs(
                ("uniprot_id",),
                group="PDB Fields",
                toggle="include_uniprot",
                enabled_if=_whenever_the_engine_runs,
                granularity=ColumnGranularity.RESIDUE,
                dtype_by_name={"uniprot_id": "text"},
                descriptions={
                    "uniprot_id": (
                        "UniProt accession aligned to the residue's PDB polymer "
                        "entity using RCSB entity-reference mappings."
                    )
                },
                profiles=frozenset(
                    {
                        PROFILE_RESEARCH_READY,
                        PROFILE_RADIATION_SENSITIVITY,
                        PROFILE_STRUCTURE_QUALITY,
                    }
                ),
            ),
            *column_specs(
                tuple(
                    sequence_cluster_column(level) for level in SEQUENCE_IDENTITY_LEVELS
                ),
                group="PDB Fields",
                toggle="include_uniprot",
                enabled_if=_whenever_the_engine_runs,
                granularity=ColumnGranularity.CHAIN,
                descriptions={
                    sequence_cluster_column(level): (
                        f"Identifier of the RCSB sequence cluster at {level} % "
                        "identity that the chain's polymer entity belongs to, from "
                        "the entity's rcsb_polymer_entity_group_membership. Chains "
                        "with the same identifier have sequences at least that "
                        "similar. Missing for an entity RCSB places in no cluster, "
                        "and always for an uploaded structure."
                    )
                    for level in SEQUENCE_IDENTITY_LEVELS
                },
                profiles=frozenset(
                    {
                        PROFILE_RESEARCH_READY,
                        PROFILE_RADIATION_SENSITIVITY,
                        PROFILE_STRUCTURE_QUALITY,
                    }
                ),
            ),
            *column_specs(
                (
                    "afdb_status",
                    "afdb_model_id",
                    "afdb_plddt_coverage",
                    "afdb_plddt",
                    "afdb_residue_has_plddt",
                ),
                group="AlphaFold Database Fields",
                toggle="include_alphafold",
                enabled_if=_fetches_models,
                granularity=ColumnGranularity.RESIDUE,
                dtype_by_name={
                    "afdb_status": "text",
                    "afdb_model_id": "text",
                    "afdb_plddt_coverage": "numeric",
                    "afdb_plddt": "numeric",
                    "afdb_residue_has_plddt": "boolean",
                },
                descriptions={
                    "afdb_status": (
                        "AlphaFold Database lookup and model-availability status "
                        "for the residue's mapped protein entity."
                    ),
                    "afdb_model_id": (
                        "Exact AlphaFold Database ModelCIF or fragment identifier "
                        "that supplied the residue's confidence score."
                    ),
                    "afdb_plddt_coverage": (
                        "Fraction of unique observed protein residues in the "
                        "entity that received an AlphaFold confidence score."
                    ),
                    "afdb_plddt": (
                        "AlphaFold predicted local distance difference test "
                        "confidence score on a scale from 0 to 100."
                    ),
                    "afdb_residue_has_plddt": (
                        "True when the observed residue received an unambiguous "
                        "AlphaFold confidence score, false when an eligible "
                        "protein residue received none, and missing where not "
                        "applicable."
                    ),
                },
                profiles=frozenset({PROFILE_RESEARCH_READY}),
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the AlphaFold columns and the entry's status for the report.

        An upload, or an entry whose acquisition failed, is given the status
        the module docstring names, and afdb_message in the report says why.
        """
        entry = context.artifacts.get(AFDB_ENTRY)
        if entry is None and context.source == "upload":
            entry = unavailable_upload_result(context.pdb_id)
        if entry is None:
            entry = AFDBEntryResult(
                pdb_id=context.pdb_id,
                status=AFDB_STATUS_LOOKUP_FAILED,
                failure_reason=context.failures.get(
                    "alphafold_database_source", "AFDB lookup produced no result."
                ),
                discovery_complete=False,
            )
        report: dict[str, ReportValue] = {
            "afdb_status": entry.status,
            "afdb_message": entry.failure_reason,
        }
        return EngineResult(
            columns=annotate_atom_dataframe(context.atom_dataframe, entry),
            report=report,
        )


ALPHAFOLD_ACQUISITION_ENGINE = AlphaFoldAcquisitionEngine()
ALPHAFOLD_ANNOTATION_ENGINE = AlphaFoldAnnotationEngine()

__all__ = ["ALPHAFOLD_ACQUISITION_ENGINE", "ALPHAFOLD_ANNOTATION_ENGINE"]
