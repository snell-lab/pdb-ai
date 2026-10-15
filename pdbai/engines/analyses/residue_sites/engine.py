"""The registered residue-site engine, with its option, table, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_residue_sites is set. It hands the atom table to
compute_structure_residue_sites, and returns the table, the two atom columns,
and the report fields. An empty table is left out of the result, and the
host writes its header alone.

residue_sites_status is no_sites for a structure without a residue of any
class, ok otherwise, and failed when the engine raised. A row's gaps are
reported on the row, in site_status, so filtering structures on an ok status
does not drop those with ordinary modelling gaps.
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
    B_FACTOR_RATIO_COLUMN,
    B_FACTOR_RATIO_COLUMN_SPEC,
    RESIDUE_SITE_LINK_COLUMN,
    RESIDUE_SITE_LINK_COLUMN_SPEC,
    RESIDUE_SITES_TABLE,
    RESIDUE_SITES_TABLE_NAME,
)
from .compute import compute_structure_residue_sites
from .constants import (
    METAL_CONTACT_DISTANCE_A,
    RESIDUE_SITES_STATUS_FAILED,
    RESIDUE_SITES_STATUS_NO_SITES,
    RESIDUE_SITES_STATUS_OK,
)
from .reference import reference_receipts

_WARNING_PREFIX = "Residue sites: "


class ResidueSiteEngine:
    """The engine that writes the residue_sites table of each structure."""

    spec = EngineSpec(
        name="residue_sites",
        display_name="Residue sites",
        phase=EnginePhase.ANNOTATION,
        toggle="include_residue_sites",
        # Counts, occupancies, and B-factors are defined for any coordinate
        # model, and the radiation-damage study compares X-ray and neutron
        # structures with the same code. What a value means for one
        # technique is applied at analysis time from the structures table.
        techniques=None,
        columns=(RESIDUE_SITE_LINK_COLUMN_SPEC, B_FACTOR_RATIO_COLUMN_SPEC),
        tables=(RESIDUE_SITES_TABLE,),
        # The table of radiolysis rate constants ships with the engine, and
        # each run records its receipt.
        reference_data=reference_receipts,
        options=(
            OptionSpec(
                "include_residue_sites",
                label="Residue sites",
                caption=(
                    "The radiation-sensitive groups of amino acids, the "
                    "carboxylates of Asp, Glu, and the C-terminus, the sulfurs "
                    "of Met and Cys, the hydroxyl of Tyr, and the rings of Trp "
                    "and His, beside control groups of the same shape, and the "
                    "oxygens of residues deposited oxidized, with modelled "
                    "atoms, occupancy, "
                    "and B-factor ratios, in their own table, and every "
                    "atom's B-factor against the atom it hangs from."
                ),
                help=(
                    "Write one row per sensitive group of each residue, and "
                    "per control group of Asn, Gln, Phe, Leu, Ile, Thr, and "
                    "Ser, to the "
                    "residue_sites table, with how many of its atoms the "
                    "model holds, their occupancy, their B-factors against "
                    "the atom the group is attached to, and whether a metal "
                    "binds it. Residues deposited oxidized, cysteines, "
                    "methionines, tryptophans, histidines, tyrosines, and "
                    "phenylalanines, are rows for the oxygens they gained. "
                    "Every atom of a standard amino acid also gets its "
                    "B-factor over that of the atom it hangs from, as "
                    "b_factor_ratio_to_parent. Nothing is judged. Off by "
                    "default."
                ),
                # The contact distance reaches the run summary and the manifest.
                enabled_text=(
                    "The carboxylates of Asp, Glu, and each C-terminus with a "
                    "modelled OXT, the sulfurs of Met and Cys, the hydroxyl "
                    "of Tyr, the indole of Trp, and the imidazole of His were "
                    "read beside the amides of Asn and Gln, the ring carbons "
                    "of Phe, the delta carbons of Leu and Ile, and the "
                    "hydroxyls of Thr and Ser, with the oxygens of residues "
                    "deposited oxidized, in the conformer "
                    "the atom table keeps. A group with a site "
                    f"atom within {METAL_CONTACT_DISTANCE_A:.1f} Å of a metal "
                    "atom of the file was flagged, not removed."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "residue_sites_status",
                f"Residue-site outcome: {RESIDUE_SITES_STATUS_NO_SITES} for a "
                "structure without a residue of any site class, "
                f"{RESIDUE_SITES_STATUS_OK} otherwise, whatever gaps its rows "
                f"report, and {RESIDUE_SITES_STATUS_FAILED} when the engine "
                "raised.",
                on_failure=RESIDUE_SITES_STATUS_FAILED,
                in_summary=True,
                label="Residue sites",
            ),
            ReportField(
                "residue_site_without_key_atom_count",
                "Residues of a site class that lack the class's key atom. No "
                "row holds them, so this is how many groups the residue_sites "
                "table cannot see. It is not a count of damaged groups, since "
                "a side chain is cut back for many reasons.",
                kind="number",
            ),
            ReportField(
                "residue_sites_message",
                "Each residue-site warning for the structure, joined, or the "
                "engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the residue_sites table, link column, and report of one structure."""
        computation = compute_structure_residue_sites(context.atom_dataframe)
        table = computation.table
        return EngineResult(
            columns=pd.DataFrame(
                {
                    RESIDUE_SITE_LINK_COLUMN: computation.link,
                    B_FACTOR_RATIO_COLUMN: computation.b_factor_ratio,
                }
            ),
            # A table left out is written with its header alone.
            tables={RESIDUE_SITES_TABLE_NAME: table} if not table.empty else {},
            warnings=tuple(
                f"{_WARNING_PREFIX}{context.pdb_id}: {warning}"
                for warning in computation.warnings
            ),
            report={
                "residue_sites_status": computation.status,
                "residue_site_without_key_atom_count": (
                    computation.without_key_atom_count
                ),
                "residue_sites_message": " | ".join(computation.warnings) or None,
            },
        )


RESIDUE_SITE_ENGINE = ResidueSiteEngine()

__all__ = ["RESIDUE_SITE_ENGINE", "ResidueSiteEngine"]
