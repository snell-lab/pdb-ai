"""The registered site-environment engine, with its option, table, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_site_environment is set. It hands the atom table to
compute_structure_site_environment, and returns the table, the depth column,
and the report fields. An empty table is left out of the result, and the
host writes its header alone.

The engine finds its groups with the residue-site engine's function, and
does not need that engine to be enabled. Turned on together, the two tables
share their site_id.

site_environment_status is no_sites for a structure without a group of any
residue-site class, ok otherwise, and failed when the engine raised. A row
whose measures are missing says why on the row.
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
    DEPTH_COLUMN,
    DEPTH_COLUMN_SPEC,
    SITE_ENVIRONMENT_TABLE,
    SITE_ENVIRONMENT_TABLE_NAME,
)
from .compute import compute_structure_site_environment
from .constants import (
    HBOND_DISTANCE_A,
    PACKING_INNER_RADIUS_A,
    PACKING_OUTER_RADIUS_A,
    SITE_ENVIRONMENT_STATUS_FAILED,
    SITE_ENVIRONMENT_STATUS_NO_SITES,
    SITE_ENVIRONMENT_STATUS_OK,
    SYMMETRY_SEARCH_RADIUS_A,
)


class SiteEnvironmentEngine:
    """The engine that writes the site_environment table of each structure."""

    spec = EngineSpec(
        name="site_environment",
        display_name="Site environment",
        phase=EnginePhase.ANNOTATION,
        toggle="include_site_environment",
        # Distances and counts are defined for any coordinate model. Only the
        # crystal contacts need a unit cell, and a row says when there is none.
        techniques=None,
        columns=(DEPTH_COLUMN_SPEC,),
        tables=(SITE_ENVIRONMENT_TABLE,),
        options=(
            OptionSpec(
                "include_site_environment",
                label="Site environment",
                caption=(
                    "The surroundings of each radiation-sensitive group and "
                    "control of the residue-site engine: hydrogen bonds, salt "
                    "bridges, the nearest disulfide, metal, aromatic ring, and "
                    "ligand, packing, solvent exposure, depth, and crystal "
                    "contacts, in their own table, and every protein atom's "
                    "depth."
                ),
                help=(
                    "Write one row per group of the residue_sites table to the "
                    "site_environment table, with the same site_id: its "
                    "possible hydrogen bonds to protein and water, its salt "
                    "bridges if a carboxylate, the distance to the nearest "
                    "disulfide sulfur, metal, aromatic ring, cation, and "
                    "ligand, the protein atoms within 6 and 10 Å, its "
                    "solvent-accessible area and depth, and the nearest atom "
                    "of a symmetry mate. Every protein atom also gets its "
                    "depth below the surface, as atom_depth_A. Nothing is "
                    "judged. Off by default."
                ),
                # The distances reach the run summary and the manifest.
                enabled_text=(
                    "The groups of the residue-site engine were read in the "
                    "conformer the atom table keeps, each in its own model. "
                    f"Hydrogen bonds were counted within {HBOND_DISTANCE_A:.1f} "
                    "Å by distance alone, protein atoms within "
                    f"{PACKING_INNER_RADIUS_A:.0f} and "
                    f"{PACKING_OUTER_RADIUS_A:.0f} Å of each group, and "
                    "symmetry mates within "
                    f"{SYMMETRY_SEARCH_RADIUS_A:.0f} Å where the entry has a "
                    "unit cell."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "site_environment_status",
                f"Site-environment outcome: {SITE_ENVIRONMENT_STATUS_NO_SITES} "
                "for a structure without a group of any residue-site class, "
                f"{SITE_ENVIRONMENT_STATUS_OK} otherwise, whatever gaps its "
                f"rows report, and {SITE_ENVIRONMENT_STATUS_FAILED} when the "
                "engine raised.",
                on_failure=SITE_ENVIRONMENT_STATUS_FAILED,
                in_summary=True,
                label="Site environment",
            ),
            ReportField(
                "site_environment_message",
                "The engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the site_environment table, depth column, and report of one structure."""
        computation = compute_structure_site_environment(context.atom_dataframe)
        table = computation.table
        return EngineResult(
            columns=pd.DataFrame({DEPTH_COLUMN: computation.depth}),
            # A table left out is written with its header alone.
            tables={SITE_ENVIRONMENT_TABLE_NAME: table} if not table.empty else {},
            report={"site_environment_status": computation.status},
        )


SITE_ENVIRONMENT_ENGINE = SiteEnvironmentEngine()

__all__ = ["SITE_ENVIRONMENT_ENGINE", "SiteEnvironmentEngine"]
