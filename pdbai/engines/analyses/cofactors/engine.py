"""The registered cofactor engine, with its option, table, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_cofactors is set. It hands the atom table to
compute_structure_cofactors, and returns the table, the atom link column, and
the report fields. An empty table is left out of the result, and the host
writes its header alone.

cofactors_status is no_cofactors for a structure without a recognized
cofactor, ok otherwise, and failed when the engine raised. A file in which
two cofactors share a site identifier fails in this way, since the table
could not tell them apart.
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
    COFACTOR_LINK_COLUMN,
    COFACTOR_LINK_COLUMN_SPEC,
    COFACTOR_SITES_TABLE,
    COFACTOR_SITES_TABLE_NAME,
)
from .compute import compute_structure_cofactors
from .constants import (
    COFACTORS_STATUS_FAILED,
    COFACTORS_STATUS_NONE,
    COFACTORS_STATUS_OK,
    LINK_SEARCH_RADIUS_A,
)


class CofactorEngine:
    """The engine that writes the cofactor_sites table of each structure."""

    spec = EngineSpec(
        name="cofactors",
        display_name="Cofactors",
        phase=EnginePhase.ANNOTATION,
        toggle="include_cofactors",
        # Angles, distances, occupancies, and B-factors are defined for any
        # coordinate model. What a value means for one technique is applied
        # at analysis time from the structures table.
        techniques=None,
        columns=(COFACTOR_LINK_COLUMN_SPEC,),
        tables=(COFACTOR_SITES_TABLE,),
        options=(
            OptionSpec(
                "include_cofactors",
                label="Cofactors",
                caption=(
                    "Flavins, hemes and chlorophylls, fluorescent-protein "
                    "chromophores, retinal, and p-coumaric acid, with the "
                    "geometry, occupancy, and B-factors of each one's reactive "
                    "group, in their own table."
                ),
                help=(
                    "Write one row per recognized cofactor to the "
                    "cofactor_sites table, with the flavin butterfly angle, "
                    "the central metal's distance from the plane of a "
                    "tetrapyrrole, the torsions of a chromophore, a retinal, "
                    "or a p-coumaric acid, and the group's occupancy and "
                    "B-factors beside the rest of the component. Nothing is "
                    "judged. Off by default."
                ),
                # The search radius reaches the run summary and the manifest.
                enabled_text=(
                    "Flavins, tetrapyrroles, and fluorescent-protein "
                    "chromophores were recognized by the atom names of their "
                    "reactive groups, and retinal and p-coumaric acid by their "
                    "components RET and HC4, in the conformer the atom table "
                    "keeps. The protein atom a retinal or a p-coumaric acid "
                    "links to was looked for among the atoms of the file "
                    f"within {LINK_SEARCH_RADIUS_A:g} Å."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "cofactors_status",
                f"Cofactor outcome: {COFACTORS_STATUS_NONE} for a structure "
                f"without a recognized cofactor, {COFACTORS_STATUS_OK} "
                "otherwise, whatever gaps its rows report, and "
                f"{COFACTORS_STATUS_FAILED} when the engine raised.",
                on_failure=COFACTORS_STATUS_FAILED,
                in_summary=True,
                label="Cofactors",
            ),
            ReportField(
                "cofactors_message",
                "The engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the cofactor_sites table, link column, and report of one structure."""
        computation = compute_structure_cofactors(context.atom_dataframe)
        table = computation.table
        return EngineResult(
            columns=pd.DataFrame({COFACTOR_LINK_COLUMN: computation.link}),
            # A table left out is written with its header alone.
            tables={COFACTOR_SITES_TABLE_NAME: table} if not table.empty else {},
            report={"cofactors_status": computation.status},
        )


COFACTOR_ENGINE = CofactorEngine()

__all__ = ["COFACTOR_ENGINE", "CofactorEngine"]
