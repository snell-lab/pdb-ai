"""The registered selenium and halogen engine, with its option, table, and report field.

The engine runs in the annotation phase, once per structure, and only when
include_selenium_halogens is set. It hands the atom table to
compute_structure_selenium_halogens, and returns the table, the atom link column,
and the report fields. An empty table is left out of the result, and the
host writes its header alone.

selenium_halogens_status is no_sites for a structure without a selenium or
halogen atom, ok otherwise, and failed when the engine raised. A
file in which two such atoms share a site identifier fails in this way,
since the table could not tell them apart.
"""

from __future__ import annotations

import pandas as pd

from ...core import (
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
)
from .columns import (
    SELENIUM_HALOGEN_LINK_COLUMN,
    SELENIUM_HALOGEN_LINK_COLUMN_SPEC,
    SELENIUM_HALOGEN_SITES_TABLE,
    SELENIUM_HALOGEN_SITES_TABLE_NAME,
)
from .compute import compute_structure_selenium_halogens
from .constants import (
    BOND_TOLERANCE_A,
    SELENIUM_HALOGENS_STATUS_FAILED,
    SELENIUM_HALOGENS_STATUS_NONE,
    SELENIUM_HALOGENS_STATUS_OK,
)


class SeleniumHalogenEngine:
    """The engine that writes the selenium_halogen_sites table of each structure."""

    spec = EngineSpec(
        name="selenium_halogens",
        display_name="Selenium and halogens",
        phase=EnginePhase.ANNOTATION,
        toggle="include_selenium_halogens",
        # Distances, occupancies, and B-factors are defined for any
        # coordinate model. What a value means for one technique is applied
        # at analysis time from the structures table.
        techniques=None,
        columns=(SELENIUM_HALOGEN_LINK_COLUMN_SPEC,),
        tables=(SELENIUM_HALOGEN_SITES_TABLE,),
        options=(
            OptionSpec(
                "include_selenium_halogens",
                label="Selenium and halogens",
                caption=(
                    "Selenium and halogen atoms, fluorine to iodine, in "
                    "selenomethionine, halogenated ligands and nucleotides, "
                    "and halide ions, with bond distances, occupancy, and "
                    "B-factors against the atoms around them, in their own "
                    "table."
                ),
                help=(
                    "Write one row per selenium, fluorine, chlorine, bromine, "
                    "or iodine atom to the selenium_halogen_sites table, with the atoms it is bonded "
                    "to, and its occupancy and B-factor beside those of the "
                    "bonded atoms and of the rest of its component. Heavy "
                    "metals are in the metal_sites table. Nothing is judged. "
                    "Off by default."
                ),
                # The tolerance reaches the run summary and the manifest.
                enabled_text=(
                    "Each selenium, fluorine, chlorine, bromine, and iodine "
                    "atom with coordinates was read in the conformer the atom table keeps, and the "
                    "atoms bonded to it were found among the atoms of the "
                    "file within the sum of the two covalent radii plus "
                    f"{BOND_TOLERANCE_A:g} Å."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "selenium_halogens_status",
                f"Selenium and halogen outcome: {SELENIUM_HALOGENS_STATUS_NONE} for a structure "
                "without a selenium or halogen atom, "
                f"{SELENIUM_HALOGENS_STATUS_OK} otherwise, whatever gaps its rows "
                f"report, and {SELENIUM_HALOGENS_STATUS_FAILED} when the engine "
                "raised.",
                on_failure=SELENIUM_HALOGENS_STATUS_FAILED,
                in_summary=True,
                label="Selenium and halogens",
            ),
            ReportField(
                "selenium_halogens_message",
                "The engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the selenium_halogen_sites table, link column, and report of one structure."""
        computation = compute_structure_selenium_halogens(context.atom_dataframe)
        table = computation.table
        return EngineResult(
            columns=pd.DataFrame({SELENIUM_HALOGEN_LINK_COLUMN: computation.link}),
            # A table left out is written with its header alone.
            tables={SELENIUM_HALOGEN_SITES_TABLE_NAME: table}
            if not table.empty
            else {},
            report={"selenium_halogens_status": computation.status},
        )


SELENIUM_HALOGEN_ENGINE = SeleniumHalogenEngine()

__all__ = ["SELENIUM_HALOGEN_ENGINE", "SeleniumHalogenEngine"]
