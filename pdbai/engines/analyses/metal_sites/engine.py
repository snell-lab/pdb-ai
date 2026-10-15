"""The registered metal-site engine, with its option, tables, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_metal_sites is set. It hands the atom table and the flat dictionary
of the structure file to compute_structure_metal_sites, and returns the three
tables, the atom link column, and the report fields. An empty table is left
out of the result, and the host writes its header alone.

metal_sites_status is no_metals for a structure without a metal atom, ok when
every deposited metalc record that names a metal atom found its donor,
degraded when one did not, and failed when the engine raised. A file in which
two metal atoms share a site identifier fails in this way, since the tables
could not tell them apart.
"""

from __future__ import annotations

import math

import pandas as pd

from ....structure import read_mmcif_as_flat_dict
from ...core import (
    STRUCTURE_CIF_PATH,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
)
from .columns import (
    METAL_BONDS_TABLE,
    METAL_BONDS_TABLE_NAME,
    METAL_LINK_COLUMN,
    METAL_LINK_COLUMN_SPEC,
    METAL_PAIRS_TABLE,
    METAL_PAIRS_TABLE_NAME,
    METAL_SITES_TABLE,
    METAL_SITES_TABLE_NAME,
)
from .compute import compute_structure_metal_sites
from .constants import (
    CANDIDATE_SEARCH_RADIUS_A,
    FIRST_SPHERE_TOLERANCE_A,
    METAL_STATUS_DEGRADED,
    METAL_STATUS_FAILED,
    METAL_STATUS_NO_METALS,
    METAL_STATUS_OK,
    NEARBY_METAL_RADIUS_A,
)
from .reference import reference_receipts

_WARNING_PREFIX = "Metal sites: "


class MetalSiteEngine:
    """The engine that writes the metal-site tables of each structure."""

    spec = EngineSpec(
        name="metal_sites",
        display_name="Metal sites",
        phase=EnginePhase.ANNOTATION,
        toggle="include_metal_sites",
        # Distances and counts are defined for any coordinate model, and the
        # radiation-damage study compares X-ray and neutron structures with
        # the same code. What a value means for one technique is applied at
        # analysis time from the structures table.
        techniques=None,
        consumes=(STRUCTURE_CIF_PATH,),
        columns=(METAL_LINK_COLUMN_SPEC,),
        tables=(METAL_SITES_TABLE, METAL_BONDS_TABLE, METAL_PAIRS_TABLE),
        reference_data=reference_receipts,
        options=(
            OptionSpec(
                "include_metal_sites",
                label="Metal sites",
                caption=(
                    "Metal atoms and their donor atoms, with bond distances, "
                    "coordination counts, and each bond's deviation from a "
                    "literature distance, and the distances between metal "
                    "atoms that lie near each other, in three tables of their "
                    "own."
                ),
                help=(
                    "Find each metal atom's donor atoms from deposited "
                    "_struct_conn metalc records and from distances within the "
                    "first coordination sphere, symmetry mates included, and "
                    "write one row per metal atom to the metal_sites table, "
                    "one per bond to the metal_ligand_bonds table, and one per "
                    "pair of nearby metal atoms to the metal_metal_distances "
                    "table. Distances are compared with literature values and "
                    "not judged. Off by default."
                ),
                # Both radii reach the run summary and the manifest.
                enabled_text=(
                    "Donor atoms of each metal atom were read from _struct_conn "
                    "metalc records and inferred where a nitrogen, oxygen, or "
                    "sulfur atom of water or a standard amino acid lies within "
                    "the literature distance for the pair plus "
                    f"{FIRST_SPHERE_TOLERANCE_A:g} Å, searched to "
                    f"{CANDIDATE_SEARCH_RADIUS_A:g} Å among the atoms of the "
                    "file and of its symmetry mates, in the conformer the atom "
                    "table keeps. Reference distances are those of Harding "
                    "(2006) and, for nickel, Zheng et al. (2008). Pairs of "
                    "metal atoms of the file within "
                    f"{NEARBY_METAL_RADIUS_A:g} Å of each other were measured."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "metal_sites_status",
                f"Metal-site outcome: {METAL_STATUS_NO_METALS} for a structure "
                f"without a metal atom, {METAL_STATUS_OK} when every deposited "
                "metalc record that names a metal atom found its donor atom, "
                f"{METAL_STATUS_DEGRADED} when one did not, and "
                f"{METAL_STATUS_FAILED} when the engine raised.",
                on_failure=METAL_STATUS_FAILED,
                in_summary=True,
                label="Metal sites",
            ),
            ReportField(
                "metal_unresolved_record_count",
                "Deposited metalc records that named a metal atom of the atom "
                "table and no usable donor atom of it.",
                kind="number",
            ),
            ReportField(
                "metal_dpi_A",
                "The entry's diffraction-component precision index in Å, or "
                "empty when a term of it is missing or the structure has no "
                "metal atom.",
                kind="number",
            ),
            ReportField(
                "metal_sites_message",
                "Each metal-site warning for the structure, joined, or the "
                "engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the metal-site tables, link column, and report of one structure.

        Raises RuntimeError when no structure file was published, and passes
        on what reading the file or computing its sites raises.
        """
        path = context.artifacts.get(STRUCTURE_CIF_PATH)
        if path is None:
            raise RuntimeError(
                f"No structure file was available for {context.pdb_id}, so its "
                "metal sites could not be read."
            )
        computation = compute_structure_metal_sites(
            context.atom_dataframe, read_mmcif_as_flat_dict(path)
        )
        tables: dict[str, pd.DataFrame] = {}
        if not computation.sites.empty:
            tables[METAL_SITES_TABLE_NAME] = computation.sites
        if not computation.bonds.empty:
            tables[METAL_BONDS_TABLE_NAME] = computation.bonds
        if not computation.pairs.empty:
            tables[METAL_PAIRS_TABLE_NAME] = computation.pairs
        return EngineResult(
            columns=pd.DataFrame({METAL_LINK_COLUMN: computation.link}),
            tables=tables,
            warnings=tuple(
                f"{_WARNING_PREFIX}{context.pdb_id}: {warning}"
                for warning in computation.warnings
            ),
            report={
                "metal_sites_status": computation.status,
                "metal_unresolved_record_count": computation.unresolved_record_count,
                "metal_dpi_A": (
                    computation.dpi if math.isfinite(computation.dpi) else None
                ),
                "metal_sites_message": " | ".join(computation.warnings) or None,
            },
        )


METAL_SITE_ENGINE = MetalSiteEngine()

__all__ = ["METAL_SITE_ENGINE", "MetalSiteEngine"]
