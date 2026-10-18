"""The wwPDB validation engines, which add density fit and model geometry.

Two engines share the include_validation setting. The source engine runs in
the acquisition phase. It fetches the entry's report, or reads it from the
cache, and publishes it under VALIDATION_REPORT. The annotation engine runs
in the annotation phase and turns the report into columns. For each residue
it adds eds_rscc, eds_rsr, rama_class, and rotamer_class. For the structure
it adds the clashscore and the Ramachandran and rotamer outlier
percentages. A report exists only for an archive entry, so neither engine
runs on an upload.

Each atom takes its residue's values through its model, author chain,
author residue number, and insertion code. These are the identifiers the
report itself uses. A structure whose residue_numbering is label has label
values in chain_id and res_num, which the report does not use. Its atoms
are then looked up by their mmCIF author items alone, and an atom without
them finds no residue. A residue modelled in alternate conformers takes the
values of the conformer the structure parser kept, as the package docstring
explains. An atom whose residue the report does not list gets empty values.
The numeric columns are floats, and the class columns are text or missing.
A missing or broken report never fails the entry. Its columns are
empty, and validation_status and validation_message say why.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pandas as pd

from ....structure.residue_identity import (
    author_chain_and_number,
    clean_string_series,
    first_nonempty_string_series,
)
from ...core import (
    MINIMAL_PROFILE,
    ColumnGranularity,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    ReportValue,
    SourceFile,
    column_specs,
)
from ..scalars import to_float, to_text
from .artifacts import VALIDATION_REPORT
from .fetch import (
    VALIDATION_STATUS_CACHED,
    VALIDATION_STATUS_FETCH_FAILED,
    fetch_validation_report,
    validation_report_url,
)

# wwPDB issues validation reports for these methods. NMR entries have no
# map, so their density-fit values stay empty. The geometry and clash values
# apply to every report.
_VALIDATION_TECHNIQUES = frozenset(
    {
        "x_ray_diffraction",
        "electron_microscopy",
        "neutron_diffraction",
        "electron_crystallography",
        "solution_nmr",
        "solid_state_nmr",
    }
)


class ValidationAcquisitionEngine:
    """The source engine that fetches or reads the validation report."""

    spec = EngineSpec(
        name="validation_source",
        display_name="wwPDB validation source",
        phase=EnginePhase.ACQUISITION,
        toggle="include_validation",
        dependencies=("rcsb_source",),
        provides=(VALIDATION_REPORT,),
        techniques=_VALIDATION_TECHNIQUES,
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Publish the entry's report and record its cache file as a source."""
        services = context.require_services()
        cache_path = services.cache_root / f"{context.pdb_id.lower()}_validation.json"
        validation = fetch_validation_report(
            context.pdb_id,
            session=services.session,
            timeout=services.timeout,
            cache_path=cache_path,
        )
        return EngineResult(
            artifacts={VALIDATION_REPORT: validation},
            source_files=(
                SourceFile(
                    "wwpdb_validation_report",
                    cache_path,
                    validation_report_url(context.pdb_id),
                    validation.status == VALIDATION_STATUS_CACHED,
                ),
            ),
        )


class ValidationAnnotationEngine:
    """The engine that turns the validation report into columns."""

    spec = EngineSpec(
        name="validation",
        display_name="wwPDB validation",
        phase=EnginePhase.ANNOTATION,
        toggle="include_validation",
        dependencies=("validation_source",),
        sources=frozenset({"download"}),
        options=(
            OptionSpec(
                "include_validation",
                label="wwPDB validation",
                caption=(
                    "Per-residue electron-density fit (RSCC / RSR) and Ramachandran "
                    "/ rotamer outlier flags, plus structure-level clashscore."
                ),
                help=(
                    "Fetch and parse the wwPDB validation XML report for "
                    "per-residue RSCC/RSR/Ramachandran/rotamer outlier flags and "
                    "structure-level clashscore (eds_rscc, eds_rsr, rama_class, "
                    "rotamer_class, clashscore, percent_rama_outliers, "
                    "percent_rotamer_outliers). Off by default."
                ),
                enabled_text=(
                    "The wwPDB validation XML report (RSCC/RSR/Ramachandran/"
                    "rotamer outliers, clashscore) was fetched and parsed."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "validation_status",
                "How the wwPDB validation report was obtained, or why it was not.",
                on_failure=VALIDATION_STATUS_FETCH_FAILED,
                in_summary=True,
                label="Validation",
            ),
            ReportField(
                "validation_message",
                "Why the wwPDB validation report is missing, when it is.",
                carries_message=True,
            ),
        ),
        columns=(
            *column_specs(
                ("eds_rscc", "eds_rsr", "rama_class", "rotamer_class"),
                group="PDB Fields",
                granularity=ColumnGranularity.RESIDUE,
                dtype_by_name={
                    "eds_rscc": "numeric",
                    "eds_rsr": "numeric",
                    "rama_class": "text",
                    "rotamer_class": "text",
                },
                descriptions={
                    "eds_rscc": (
                        "Residue real-space correlation coefficient between "
                        "calculated and experimental electron density; values "
                        "closer to 1 indicate better agreement."
                    ),
                    "eds_rsr": (
                        "Residue real-space R factor comparing calculated and "
                        "experimental electron density; lower values indicate "
                        "better agreement."
                    ),
                    "rama_class": (
                        "wwPDB validation class of the residue's backbone "
                        "conformation in Ramachandran space."
                    ),
                    "rotamer_class": (
                        "wwPDB validation rotamer assignment for the residue's "
                        "side chain: a MolProbity rotamer name (mt, t80, m-80, ...) "
                        "or OUTLIER."
                    ),
                },
                # The per-residue density fit and Ramachandran class are in the
                # minimal profile.
                profiles_by_name={
                    "eds_rscc": MINIMAL_PROFILE,
                    "eds_rsr": MINIMAL_PROFILE,
                    "rama_class": MINIMAL_PROFILE,
                },
            ),
            *column_specs(
                ("clashscore", "percent_rama_outliers", "percent_rotamer_outliers"),
                group="PDB Fields",
                granularity=ColumnGranularity.STRUCTURE,
                dtype_by_name={
                    "clashscore": "numeric",
                    "percent_rama_outliers": "numeric",
                    "percent_rotamer_outliers": "numeric",
                },
                descriptions={
                    "clashscore": (
                        "Structure-level number of serious all-atom steric "
                        "clashes per 1,000 atoms."
                    ),
                    "percent_rama_outliers": (
                        "Percentage of evaluated residues classified as "
                        "Ramachandran outliers."
                    ),
                    "percent_rotamer_outliers": (
                        "Percentage of evaluated residues classified as "
                        "side-chain rotamer outliers."
                    ),
                },
            ),
        ),
        techniques=_VALIDATION_TECHNIQUES,
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the report's values on every atom, and the report's status.

        The numeric columns are floats and the class columns text, and a
        value the report leaves out, or one of the wrong type, is missing.
        Without a report the columns are empty and the status is
        fetch_failed. The message is then the source engine's recorded
        failure, when it has one. A report the source engine could not
        obtain also adds its message to the run's warnings.
        """
        validation = context.artifacts.get(VALIDATION_REPORT)
        if validation is None:
            message = context.failures.get(
                "validation_source", "wwPDB validation acquisition produced no result."
            )
            report: dict[str, ReportValue] = {
                "validation_status": VALIDATION_STATUS_FETCH_FAILED,
                "validation_message": message,
            }
            parsed_report: Mapping[str, object] = {}
            # The source engine's failure is already in the run's warnings.
            warnings: tuple[str, ...] = ()
        else:
            report = {
                "validation_status": validation.status,
                "validation_message": validation.message,
            }
            parsed_report = validation.report
            warnings = (
                (validation.message,)
                if not validation.ok and validation.message is not None
                else ()
            )
        residues = _residue_values(parsed_report.get("residues"))

        atoms = context.atom_dataframe
        model = first_nonempty_string_series(
            atoms, ("model_num", "_atom_site.pdbx_PDB_model_num"), "1"
        )
        chain, residue_number = author_chain_and_number(atoms)
        insertion_code = clean_string_series(atoms, "_atom_site.pdbx_PDB_ins_code")
        # This is the key parse_validation_xml builds. An atom table without
        # model numbers reads as model 1, as a report without them does.
        keys = model + ":" + chain + ":" + residue_number + insertion_code
        conformers = _kept_conformers(atoms, keys)

        empty: Mapping[str, object] = {}
        # Each atom's residue is looked up once for all four residue columns.
        # The kept conformer's own values come first when the report has them.
        residue_values_by_atom = [
            (residues.get(f"{key}:{conformer}") if conformer else None)
            or residues.get(key, empty)
            for key, conformer in zip(
                cast(list[str], keys.tolist()),
                cast(list[str], conformers.tolist()),
                strict=True,
            )
        ]
        output = pd.DataFrame(index=atoms.index)
        for source_name, output_name in (("rscc", "eds_rscc"), ("rsr", "eds_rsr")):
            output[output_name] = pd.Series(
                [
                    to_float(values.get(source_name))
                    for values in residue_values_by_atom
                ],
                index=output.index,
                dtype="float64",
            )
        for name in ("rama_class", "rotamer_class"):
            output[name] = pd.Series(
                [to_text(values.get(name)) for values in residue_values_by_atom],
                index=output.index,
                dtype=object,
            )
        for name in ("clashscore", "percent_rama_outliers", "percent_rotamer_outliers"):
            value = to_float(parsed_report.get(name))
            output[name] = pd.Series(
                float("nan") if value is None else value,
                index=output.index,
                dtype="float64",
            )
        return EngineResult(columns=output, report=report, warnings=warnings)


def _residue_values(raw_residues: object) -> dict[str, Mapping[str, object]]:
    """Return the report's residue values by key, without malformed ones.

    A cached report is read back from JSON, so a damaged file could hold
    anything. A residue whose values are not a mapping is dropped. When
    ``raw_residues`` is not a mapping, no residue is returned.
    """
    residues: dict[str, Mapping[str, object]] = {}
    if isinstance(raw_residues, Mapping):
        for key, value in cast(Mapping[object, object], raw_residues).items():
            if isinstance(value, Mapping):
                residues[str(key)] = cast(Mapping[str, object], value)
    return residues


def _kept_conformers(atoms: pd.DataFrame, keys: pd.Series) -> pd.Series:
    """Return the altcode of each atom's residue, or "" when it has none.

    ``keys`` holds each atom's residue key, on the index of ``atoms``. The
    result is on that index too.

    A residue's altcode is the first alternate location identifier among
    its atoms, so atoms without one share it. The structure parser leaves
    one conformer per residue, so a residue has at most one altcode unless
    its rows were not parsed that way.
    """
    alt_ids = clean_string_series(atoms, "_atom_site.label_alt_id")
    labelled = alt_ids.ne("")
    if not labelled.any():
        return pd.Series("", index=atoms.index, dtype=str)
    first_alt_ids = alt_ids[labelled].groupby(keys[labelled], sort=False).first()
    return keys.map(first_alt_ids).fillna("").astype(str)


VALIDATION_ACQUISITION_ENGINE = ValidationAcquisitionEngine()
VALIDATION_ANNOTATION_ENGINE = ValidationAnnotationEngine()

__all__ = ["VALIDATION_ACQUISITION_ENGINE", "VALIDATION_ANNOTATION_ENGINE"]
