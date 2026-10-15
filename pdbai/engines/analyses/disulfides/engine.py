"""The registered disulfide engine, with its option, table, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_disulfides is set. It reads the structure file the atom table was
parsed from, the download or the upload, since the atom table holds only one
conformer of each residue. It hands that file's flat dictionary and the atom
table to compute_structure_disulfides, and returns the table, the atom link
column, and the report fields. An empty table is left out of the result, and
the host writes its header alone.

disulfide_status is ok when every deposited disulfide within one molecule
found a pair, even when the table is empty or a row names a gap in its
pair_status. A row's gaps are reported on the row, so filtering structures on
an ok status does not drop those with ordinary modelling gaps, such as a
truncated side chain or a placeholder B-factor. The status is degraded when a
deposited disulfide within one molecule found no pair in any model, and
failed when the engine raised. A file in which two cysteine atoms share a
site identifier fails in this way, since the table could not tell them
apart. disulfide_cysteine_without_sg_count counts the cysteines
with a CB but no SG, which no row holds. It says how many cysteines the table
cannot see and is not a count of damaged bonds, since a side chain is cut
back for many reasons. Each warning is also one of the run's warnings, and
the run report keeps each distinct warning once, joined, in
disulfide_message.
"""

from __future__ import annotations

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
    DISULFIDE_LINK_COLUMN,
    DISULFIDE_LINK_COLUMN_SPEC,
    DISULFIDES_TABLE,
    DISULFIDES_TABLE_NAME,
)
from .compute import compute_structure_disulfides
from .constants import (
    DISULFIDE_SEARCH_CUTOFF_A,
    DISULFIDE_STATUS_DEGRADED,
    DISULFIDE_STATUS_FAILED,
    DISULFIDE_STATUS_OK,
    METAL_CONTACT_DISTANCE_A,
)

_WARNING_PREFIX = "Disulfides: "


class DisulfideEngine:
    """The engine that writes the disulfides table of each structure."""

    spec = EngineSpec(
        name="disulfides",
        display_name="Disulfides",
        phase=EnginePhase.ANNOTATION,
        toggle="include_disulfides",
        # Geometry, occupancy, and B-factors are defined for any coordinate
        # model, and the radiation-damage study needs at least X-ray and
        # neutron structures. What a value means for one technique, such as a
        # neutron B-factor or the placeholder B of an NMR or EM model, is
        # applied at analysis time from the structures table.
        techniques=None,
        consumes=(STRUCTURE_CIF_PATH,),
        columns=(DISULFIDE_LINK_COLUMN_SPEC,),
        tables=(DISULFIDES_TABLE,),
        options=(
            OptionSpec(
                "include_disulfides",
                label="Disulfides",
                caption=(
                    "Candidate disulfides, cysteine pairs found from deposited "
                    "records and SG–SG distances, with S–S distances, SG "
                    "occupancy, alternate conformers, SG/CB B-factor ratios, "
                    "and a metal-site flag, in their own table."
                ),
                help=(
                    "Find cysteine pairs from deposited _struct_conn disulf "
                    "records and from SG–SG distances of at most "
                    f"{DISULFIDE_SEARCH_CUTOFF_A:.1f} Å, with no lower limit, "
                    "and write one row per pair and model to the disulfides "
                    "table. A row is a candidate, not a proven bond, and "
                    "metal_bridged flags the pairs that share a metal atom. Off "
                    "by default."
                ),
                # Both distances reach the run summary and the manifest.
                enabled_text=(
                    "Cysteine pairs were read from _struct_conn disulf records "
                    "within one molecule and found where coexisting SG "
                    f"conformers lie at most {DISULFIDE_SEARCH_CUTOFF_A:.1f} Å "
                    "apart, with no lower limit, model by model, over every "
                    "alternate conformer in the structure file. Pairs whose SG "
                    f"atoms both lie within {METAL_CONTACT_DISTANCE_A:.1f} Å of "
                    "one metal atom were flagged, not removed."
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "disulfide_status",
                f"Disulfide outcome: {DISULFIDE_STATUS_OK} when every deposited "
                "disulfide within one molecule matched a pair, whatever gaps its "
                "rows report, "
                f"{DISULFIDE_STATUS_DEGRADED} when a deposited disulfide within "
                f"one molecule matched no cysteine pair, and "
                f"{DISULFIDE_STATUS_FAILED} when the engine raised.",
                on_failure=DISULFIDE_STATUS_FAILED,
                in_summary=True,
                label="Disulfides",
            ),
            ReportField(
                "disulfide_unmatched_record_count",
                "Deposited cysteine disulfide records within one molecule that "
                "matched no pair of distinct cysteines with SG atoms in any model.",
                kind="number",
            ),
            ReportField(
                "disulfide_symmetry_record_count",
                "Deposited cysteine disulfide records that join the molecule to a "
                "symmetry mate, which the table leaves out.",
                kind="number",
            ),
            ReportField(
                "disulfide_cysteine_without_sg_count",
                "Cysteines, counted once per model, with a CB but no SG in any "
                "conformer. No row holds them, so this is how many cysteines the "
                "disulfides table cannot see. It is not a count of damaged "
                "bonds, since a side chain is cut back for many reasons.",
                kind="number",
            ),
            ReportField(
                "disulfide_message",
                "Each distinct disulfide warning for the structure once, joined, "
                "or the engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the disulfides table, link column, and report of one structure.

        Raises RuntimeError when no structure file was published, and passes
        on what reading the file or computing its disulfides raises.
        """
        path = context.artifacts.get(STRUCTURE_CIF_PATH)
        if path is None:
            raise RuntimeError(
                f"No structure file was available for {context.pdb_id}, so its "
                "disulfides could not be read."
            )
        computation = compute_structure_disulfides(
            context.atom_dataframe,
            read_mmcif_as_flat_dict(path),
            pdb_id=context.pdb_id,
        )
        table = computation.table
        return EngineResult(
            columns=pd.DataFrame({DISULFIDE_LINK_COLUMN: computation.link}),
            # A table left out is written with its header alone.
            tables={DISULFIDES_TABLE_NAME: table} if not table.empty else {},
            warnings=tuple(
                f"{_WARNING_PREFIX}{warning}" for warning in computation.warnings
            ),
            report={
                "disulfide_status": computation.status,
                "disulfide_unmatched_record_count": (
                    computation.unmatched_record_count
                ),
                "disulfide_symmetry_record_count": computation.symmetry_record_count,
                "disulfide_cysteine_without_sg_count": (
                    computation.cysteine_without_sg_count
                ),
                "disulfide_message": " | ".join(computation.warnings) or None,
            },
        )


DISULFIDE_ENGINE = DisulfideEngine()

__all__ = ["DISULFIDE_ENGINE", "DisulfideEngine"]
