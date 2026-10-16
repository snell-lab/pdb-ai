"""The four engines that bring PDB-Redo's model of an entry into a run.

The source engine downloads PDB-Redo's mmCIF file into the cache. The
structure engine parses the file into atom rows. The comparison engine adds
the columns that compare_with_rcsb_atoms returns, and reports how well the atoms
paired. The map source engine downloads PDB-Redo's reflection file, which
holds the map coefficients of the re-refined model.

The first three answer to the include_pdb_redo flag, which the comparison
engine declares. RABDAM also needs the first two when it scores PDB-Redo's
model, so the file is downloaded even with the flag off. The map source
engine answers to include_density, which the density engine declares, and
runs when that engine reads PDB-Redo's maps, which also needs the other
three.

Both source engines read a local copy of PDB-Redo's databank before they
make a request (see mirror.py).

PDB-Redo has no entry for some X-ray structures, and that is not a failure.
The source engine then returns no file, and the structure engine warns and
reports pdb_redo_structure_parsed as False. Any other failed download raises,
and the structure engine does not warn about it a second time.

The comparison counts as a success only when at least one atom paired and
the alignment was not rejected. Otherwise its value and difference columns
are empty, and a warning or the source engine's failure says why.
"""

from __future__ import annotations

import pandas as pd

from ....identifiers import normalize_pdb_id
from ....structure import parse_mmcif_atom_rows
from ...core import (
    STRUCTURE_METADATA,
    ColumnGranularity,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    SourceFile,
    column_specs,
)
from ..http_cache import NotAvailableError, fetch_mmcif, fetch_mtz, is_mtz_file
from ..map_sources import DENSITY_SOURCE_PDB_REDO, DENSITY_TOGGLE, density_reads
from .artifacts import PDB_REDO_CIF_PATH, PDB_REDO_DATAFRAME, PDB_REDO_MTZ_PATH
from .comparison import OUTPUT_COLUMNS, compare_with_rcsb_atoms
from .mirror import mirror_file

# PDB-Redo re-refines X-ray entries against their deposited reflection data.
_PDB_REDO_TECHNIQUES = frozenset({"x_ray_diffraction"})


class PdbRedoAcquisitionEngine:
    """The engine that downloads PDB-Redo's mmCIF file for an entry."""

    spec = EngineSpec(
        name="pdb_redo_source",
        display_name="PDB-Redo source",
        phase=EnginePhase.ACQUISITION,
        toggle="include_pdb_redo",
        dependencies=("rcsb_source",),
        provides=(PDB_REDO_CIF_PATH,),
        techniques=_PDB_REDO_TECHNIQUES,
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the path of the entry's file, unless PDB-Redo lacks one.

        Any other failed download raises RuntimeError, and so does a
        ``context`` without network services.
        """
        services = context.require_services()
        pdb_id = normalize_pdb_id(context.pdb_id).lower()
        mirrored = mirror_file(services.cache_root, pdb_id, f"{pdb_id}_final.cif")
        if mirrored is not None:
            return EngineResult(
                artifacts={PDB_REDO_CIF_PATH: mirrored},
                source_files=(
                    SourceFile("pdb_redo_mmcif", mirrored, mirrored.as_uri(), True),
                ),
            )
        url = f"https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.cif"
        path = services.cache_root / "pdb_redo" / f"{pdb_id}.cif"
        try:
            cached = fetch_mmcif(
                url,
                path,
                session=services.session,
                timeout=services.timeout,
                label=f"PDB-Redo {pdb_id.upper()}",
            )
        except NotAvailableError:
            # Not every X-ray entry has a PDB-Redo re-refinement. A missing
            # one is an expected outcome, which the structure engine reports.
            return EngineResult(
                source_files=(SourceFile("pdb_redo_mmcif", path, url, False),),
            )
        return EngineResult(
            artifacts={PDB_REDO_CIF_PATH: path},
            source_files=(SourceFile("pdb_redo_mmcif", path, url, cached),),
        )


class PdbRedoMapAcquisitionEngine:
    """The engine that downloads PDB-Redo's reflection file for an entry."""

    spec = EngineSpec(
        name="pdb_redo_map_source",
        display_name="PDB-Redo map coefficients",
        phase=EnginePhase.ACQUISITION,
        toggle=DENSITY_TOGGLE,
        # The file is fetched only when the density engine reads PDB-Redo's
        # maps, not when it reads the wwPDB validation maps alone.
        enabled_if=lambda options: density_reads(options, DENSITY_SOURCE_PDB_REDO),
        dependencies=("rcsb_source",),
        provides=(PDB_REDO_MTZ_PATH,),
        techniques=_PDB_REDO_TECHNIQUES,
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the path of the entry's reflection file, unless PDB-Redo lacks one.

        A file of the local copy that does not start as an MTZ file raises
        RuntimeError, and so does any failed download but the answer that
        PDB-Redo has no such file.
        """
        services = context.require_services()
        pdb_id = normalize_pdb_id(context.pdb_id).lower()
        mirrored = mirror_file(services.cache_root, pdb_id, f"{pdb_id}_final.mtz")
        if mirrored is not None:
            if not is_mtz_file(mirrored):
                raise RuntimeError(
                    f"PDB-Redo {pdb_id.upper()} reflection file {mirrored} of the "
                    "local copy is not an MTZ file"
                )
            return EngineResult(
                artifacts={PDB_REDO_MTZ_PATH: mirrored},
                source_files=(
                    SourceFile("pdb_redo_mtz", mirrored, mirrored.as_uri(), True),
                ),
            )
        url = f"https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.mtz"
        path = services.cache_root / "pdb_redo" / f"{pdb_id}_final.mtz"
        try:
            cached = fetch_mtz(
                url,
                path,
                session=services.session,
                timeout=services.timeout,
                label=f"PDB-Redo {pdb_id.upper()}",
            )
        except NotAvailableError:
            # As with the model, an entry PDB-Redo has not re-refined is an
            # expected outcome, which the density engine reports.
            return EngineResult(
                source_files=(SourceFile("pdb_redo_mtz", path, url, False),),
            )
        return EngineResult(
            artifacts={PDB_REDO_MTZ_PATH: path},
            source_files=(SourceFile("pdb_redo_mtz", path, url, cached),),
        )


class PdbRedoStructureEngine:
    """The engine that parses PDB-Redo's mmCIF file into atom rows."""

    spec = EngineSpec(
        name="pdb_redo_structure",
        display_name="PDB-Redo structure",
        phase=EnginePhase.ANNOTATION,
        toggle="include_pdb_redo",
        dependencies=("pdb_redo_source",),
        provides=(PDB_REDO_DATAFRAME,),
        report_fields=(
            ReportField(
                "pdb_redo_structure_parsed",
                "True when PDB-Redo's re-refined mmCIF was downloaded and parsed; "
                "false when PDB-Redo has no entry or it could not be read.",
                kind="flag",
                on_failure=False,
                in_summary=True,
                label="PDB-Redo structure",
            ),
        ),
        techniques=_PDB_REDO_TECHNIQUES,
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the file's atom rows, and report whether there were any.

        An entry PDB-Redo lacks, a file that cannot be parsed, and a file
        without atoms each give a warning rather than raising. A failed
        download gives none, since it is the source engine's failure.
        """
        path = context.artifacts.get(PDB_REDO_CIF_PATH)
        consequence = (
            "PDB-Redo comparison columns and PDB-Redo-scored RABDAM columns will "
            "be empty for this structure; RCSB-sourced columns are unaffected."
        )
        if path is None:
            warnings = (
                ()
                if "pdb_redo_source" in context.failures
                else (f"PDB-Redo has no entry for {context.pdb_id}. {consequence}",)
            )
            return EngineResult(
                warnings=warnings, report={"pdb_redo_structure_parsed": False}
            )
        metadata = context.artifacts.get(STRUCTURE_METADATA) or {}
        try:
            # Atoms pair only within one entry ID. Naming the entry, rather
            # than letting the parser read it from the cache file's name,
            # gives PDB-Redo's atoms the ID the RCSB atoms were parsed under.
            parsed = parse_mmcif_atom_rows(path, metadata, structure_id=context.pdb_id)
            if parsed.dataframe.empty:
                raise RuntimeError("Parsed PDB-Redo atom-site DataFrame was empty")
        except RuntimeError as exception:
            return EngineResult(
                warnings=(
                    f"PDB-Redo parse failed for {context.pdb_id}: {exception}. {consequence}",
                ),
                report={"pdb_redo_structure_parsed": False},
            )
        return EngineResult(
            artifacts={PDB_REDO_DATAFRAME: parsed.dataframe},
            report={"pdb_redo_structure_parsed": True},
        )


class PdbRedoComparisonEngine:
    """The engine that sets PDB-Redo's atom values beside the RCSB atoms."""

    spec = EngineSpec(
        name="pdb_redo_comparison",
        display_name="PDB-Redo comparison",
        phase=EnginePhase.ANNOTATION,
        toggle="include_pdb_redo",
        dependencies=("pdb_redo_structure",),
        sources=frozenset({"download"}),
        options=(
            OptionSpec(
                "include_pdb_redo",
                label="PDB-Redo comparison",
                caption=(
                    "Also fetch PDB-Redo's re-refined mmCIF and report curated "
                    "numeric deltas (coordinates / B-factor / occupancy / ANISOU) "
                    "versus the RCSB baseline."
                ),
                help=(
                    "Also fetch PDB-Redo's re-refined mmCIF and report curated "
                    "numeric deltas (coordinates/B-factor/occupancy/ANISOU) versus "
                    "the RCSB baseline. Off by default."
                ),
                enabled_text=(
                    "PDB-Redo curated numeric deltas (coordinates/B-factor/"
                    "occupancy/ANISOU) were computed against the RCSB baseline."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "pdb_redo_comparison_success",
                "True when PDB-Redo atoms were compared with the RCSB atoms; "
                "false when PDB-Redo has no model, no atom paired, or the "
                "alignment was rejected.",
                kind="flag",
                on_failure=False,
                in_summary=True,
                label="PDB-Redo comparison",
            ),
            ReportField(
                "pdb_redo_alignment_match_rate",
                "Share of protein heavy RCSB atoms matched one-to-one to PDB-Redo.",
                kind="fraction",
                in_summary=True,
                label="PDB-Redo match rate",
            ),
            ReportField(
                "pdb_redo_alignment_rcsb_ambiguous_atom_count",
                "RCSB atoms left out of the alignment because their identity repeats.",
                kind="number",
            ),
            ReportField(
                "pdb_redo_alignment_pdb_redo_ambiguous_atom_count",
                "PDB-Redo atoms left out of the alignment because their identity "
                "repeats.",
                kind="number",
            ),
        ),
        # The coordinator refuses a column the spec does not declare, so the
        # spec declares exactly the columns compare_with_rcsb_atoms returns.
        columns=column_specs(
            OUTPUT_COLUMNS,
            group="PDB-Redo Fields",
            dtype_by_name={
                name: "boolean" if name == "pdb_redo_revised" else "numeric"
                for name in OUTPUT_COLUMNS
            },
            granularity_by_name={
                name: ColumnGranularity.STRUCTURE
                for name in OUTPUT_COLUMNS
                if name.startswith("pdb_redo_alignment_")
            },
            descriptions={
                "pdb_redo_x": "PDB-Redo re-refined Cartesian x coordinate in Å.",
                "pdb_redo_y": "PDB-Redo re-refined Cartesian y coordinate in Å.",
                "pdb_redo_z": "PDB-Redo re-refined Cartesian z coordinate in Å.",
                "pdb_redo_x_delta": (
                    "PDB-Redo x coordinate minus the RCSB x coordinate, in Å."
                ),
                "pdb_redo_y_delta": (
                    "PDB-Redo y coordinate minus the RCSB y coordinate, in Å."
                ),
                "pdb_redo_z_delta": (
                    "PDB-Redo z coordinate minus the RCSB z coordinate, in Å."
                ),
                "pdb_redo_bfactor_delta": (
                    "PDB-Redo isotropic B factor minus the RCSB isotropic B factor, "
                    "in Å²."
                ),
                "pdb_redo_occupancy_delta": (
                    "PDB-Redo atom occupancy minus the RCSB atom occupancy."
                ),
                "pdb_redo_u_equivalent_delta": (
                    "PDB-Redo equivalent isotropic U minus the RCSB equivalent "
                    "isotropic U, in Å²."
                ),
                "pdb_redo_revised": (
                    "True when any compared coordinate, B-factor, occupancy, or "
                    "equivalent-U delta has absolute magnitude greater than 0.001; "
                    "false when compared values are within tolerance, and missing "
                    "when no numeric atom counterpart was available."
                ),
                "pdb_redo_alignment_match_rate": (
                    "Fraction of protein heavy RCSB atoms matched one-to-one to "
                    "PDB-Redo by author residue and atom identity. An entry "
                    "without protein heavy atoms counts all its heavy atoms, and "
                    "one without heavy atoms counts every atom. 0 when the "
                    "alignment was rejected."
                ),
                "pdb_redo_alignment_rcsb_ambiguous_atom_count": (
                    "Number of RCSB atoms excluded from PDB-Redo alignment because "
                    "their model-aware residue and atom identity was duplicated."
                ),
                "pdb_redo_alignment_pdb_redo_ambiguous_atom_count": (
                    "Number of PDB-Redo atoms excluded from alignment because their "
                    "model-aware residue and atom identity was duplicated."
                ),
            },
        ),
        techniques=_PDB_REDO_TECHNIQUES,
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the comparison columns and the alignment's results.

        Without PDB-Redo's atom rows the columns are empty. A rejected
        alignment, one that paired no atom, or one that paired atoms on RCSB
        label chain IDs adds a warning. The report counts the comparison a
        success only when an alignment was made, was not rejected, and
        paired at least one atom.
        """
        pdb_redo_dataframe = context.artifacts.get(PDB_REDO_DATAFRAME)
        comparison = compare_with_rcsb_atoms(
            context.atom_dataframe,
            pdb_redo_dataframe if pdb_redo_dataframe is not None else pd.DataFrame(),
            pdb_id=context.pdb_id,
        )
        alignment = comparison.alignment
        warnings: tuple[str, ...] = ()
        if alignment is not None:
            if alignment.rejected_reason:
                warnings = (
                    f"PDB-Redo comparison for {context.pdb_id} was rejected because "
                    f"{alignment.rejected_reason}; coordinate, B-factor, "
                    "occupancy, and U deltas are empty for this structure.",
                )
            elif alignment.matched_atom_count == 0:
                warnings = (
                    f"PDB-Redo comparison for {context.pdb_id} paired no PDB-Redo "
                    "atom with an RCSB atom by chain, residue, and atom name; "
                    "coordinate, B-factor, occupancy, and U deltas are empty for "
                    "this structure.",
                )
            elif alignment.chain_id_source != "author":
                warnings = (
                    f"PDB-Redo file for {context.pdb_id} carries the RCSB label "
                    "chain IDs in place of the deposited author chain IDs; atoms "
                    "were aligned on label chain IDs instead.",
                )
        return EngineResult(
            columns=comparison.columns,
            warnings=warnings,
            report={
                "pdb_redo_comparison_success": (
                    alignment is not None
                    and alignment.rejected_reason is None
                    and alignment.matched_atom_count > 0
                ),
                "pdb_redo_alignment_match_rate": (
                    alignment.match_rate if alignment is not None else None
                ),
                "pdb_redo_alignment_rcsb_ambiguous_atom_count": (
                    alignment.rcsb_ambiguous_atom_count
                    if alignment is not None
                    else None
                ),
                "pdb_redo_alignment_pdb_redo_ambiguous_atom_count": (
                    alignment.pdb_redo_ambiguous_atom_count
                    if alignment is not None
                    else None
                ),
            },
        )


PDB_REDO_ACQUISITION_ENGINE = PdbRedoAcquisitionEngine()
PDB_REDO_MAP_ACQUISITION_ENGINE = PdbRedoMapAcquisitionEngine()
PDB_REDO_STRUCTURE_ENGINE = PdbRedoStructureEngine()
PDB_REDO_COMPARISON_ENGINE = PdbRedoComparisonEngine()

__all__ = [
    "PDB_REDO_ACQUISITION_ENGINE",
    "PDB_REDO_COMPARISON_ENGINE",
    "PDB_REDO_MAP_ACQUISITION_ENGINE",
    "PDB_REDO_STRUCTURE_ENGINE",
]
