"""The names, kinds, and descriptions of the SASA output columns.

The columns are listed by the level they describe. An atom column holds one
value per atom. A residue column repeats one value on every atom of a protein
residue. A structure column repeats one value on every row of a structure.
engine.py declares each column at its level, with its kind and the
description written here.
"""

from __future__ import annotations

from .constants import RASA_BURIED_BELOW, RASA_EXPOSED_FROM

ATOM_OUTPUT_COLUMNS: tuple[str, ...] = (
    "atom_sasa_protein",
    "atom_sasa_bound",
    "sasa_component_category",
)

RESIDUE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "residue_sasa_protein",
    "rasa_protein",
    "is_buried_protein",
    "is_exposed_protein",
    "residue_sasa_bound",
    "rasa_bound",
    "is_buried_bound",
    "is_exposed_bound",
    "nonprotein_occluded_sasa",
    "nonprotein_occluded_fraction",
    "has_bound_nonprotein",
    "residue_sasa_complete",
    "rasa_eligible",
    "rasa_ineligible_reason",
    "is_terminal_residue",
    "is_occupancy_ambiguous",
)

STRUCTURE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "has_experimental_additive",
    "has_unknown_nonpolymer",
    "bound_context_complete",
    "bound_context_ambiguous",
    "sasa_status",
    "sasa_ok",
    "sasa_warning_codes",
    "sasa_warning_messages",
    "sasa_normalization",
    "sasa_protein_radius_scheme",
    "sasa_nonprotein_radius_scheme",
    "sasa_probe_radius",
    "sasa_n_sphere_points",
    "sasa_accuracy_mode",
    "sasa_structure_context",
    "sasa_include_experimental_additives",
    "sasa_include_unknown_nonpolymers",
)

ALL_OUTPUT_COLUMNS = (
    ATOM_OUTPUT_COLUMNS + RESIDUE_OUTPUT_COLUMNS + STRUCTURE_OUTPUT_COLUMNS
)

# Raw areas and free-text messages are left out of the compact ML profile.
ML_PROFILE_COLUMNS: frozenset[str] = frozenset(ALL_OUTPUT_COLUMNS) - {
    "atom_sasa_protein",
    "atom_sasa_bound",
    "residue_sasa_protein",
    "residue_sasa_bound",
    "nonprotein_occluded_sasa",
    "sasa_warning_messages",
}

SASA_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "atom_sasa_protein": (
        "Per-protein-polymer-heavy-atom SASA in Å² with only protein-polymer "
        "atoms acting as occluders, calculated independently for each model."
    ),
    "atom_sasa_bound": (
        "Per-protein-polymer-heavy-atom SASA in Å² with non-protein polymers, "
        "ligands, cofactors, and metals also occluding, together with unknown "
        "non-polymers and experimental additives when the run includes them."
    ),
    "sasa_component_category": (
        "Biological/mmCIF-derived atom category: protein_polymer, "
        "nonprotein_polymer, water, ligand_or_cofactor, metal, "
        "experimental_additive, or unknown_nonpolymer."
    ),
    "residue_sasa_protein": (
        "Sum of calculable atom_sasa_protein values for a canonically identified "
        "protein residue. May be a partial sum; check residue_sasa_complete."
    ),
    "rasa_protein": (
        "Eligible residue_sasa_protein divided by the Tien2013_ALLOWED_DSSP "
        "maximum for that amino-acid type."
    ),
    "is_buried_protein": (
        f"Nullable boolean; True when rasa_protein < {RASA_BURIED_BELOW:.2f}."
    ),
    "is_exposed_protein": (
        f"Nullable boolean; True when rasa_protein >= {RASA_EXPOSED_FROM:.2f}."
    ),
    "residue_sasa_bound": (
        "Sum of calculable atom_sasa_bound values for a canonically identified "
        "protein residue. May be a partial sum; check residue_sasa_complete."
    ),
    "rasa_bound": (
        "Eligible residue_sasa_bound divided by the same "
        "Tien2013_ALLOWED_DSSP amino-acid maximum."
    ),
    "is_buried_bound": (
        f"Nullable boolean; True when rasa_bound < {RASA_BURIED_BELOW:.2f}."
    ),
    "is_exposed_bound": (
        f"Nullable boolean; True when rasa_bound >= {RASA_EXPOSED_FROM:.2f}."
    ),
    "nonprotein_occluded_sasa": (
        "residue_sasa_protein minus residue_sasa_bound in Å²: the area the "
        "residue loses to the non-protein occluders of the bound context."
    ),
    "nonprotein_occluded_fraction": (
        "nonprotein_occluded_sasa divided by residue_sasa_protein; missing "
        "when the denominator is zero or unavailable."
    ),
    "has_bound_nonprotein": (
        "Nullable boolean; True when nonprotein_occluded_sasa is above zero."
    ),
    "residue_sasa_complete": (
        "Nullable boolean indicating that every expected heavy atom of the "
        "residue was present and calculable, so residue_sasa_protein is a "
        "complete sum. False for a residue type with no expected atom list."
    ),
    "rasa_eligible": (
        "Nullable boolean indicating eligibility for Tien-normalized RASA and "
        "burial/exposure flags."
    ),
    "rasa_ineligible_reason": (
        "Semicolon-separated structured reasons why RASA was not emitted."
    ),
    "is_terminal_residue": (
        "Nullable boolean indicating a residue at either end of its modeled "
        "chain or beside a sequence or peptide-bond break, which is excluded "
        "from Tien normalization."
    ),
    "is_occupancy_ambiguous": (
        "Nullable boolean indicating partial or ambiguous occupancy that makes "
        "the residue ineligible for normalized RASA."
    ),
    "has_experimental_additive": (
        "Whether the input contains a recognized crystallization additive or "
        "cryoprotectant."
    ),
    "has_unknown_nonpolymer": (
        "Whether the input contains an unclassified non-polymer component."
    ),
    "bound_context_complete": (
        "Structure-wide worst-case flag: whether the bound context of every "
        "model with protein atoms was computed, and every requested occluder "
        "in it had a usable occupancy, coordinates, and a supported radius."
    ),
    "bound_context_ambiguous": (
        "Structure-wide worst-case flag: whether included unknown or "
        "partial-occupancy partners make the bound context ambiguous."
    ),
    "sasa_status": (
        "Structure-level SASA status: ok, degraded, or failed. Missing when the "
        "SASA engine itself failed on the structure."
    ),
    "sasa_ok": (
        "Nullable structure-level boolean; True only when every requested SASA "
        "context completed without degradation."
    ),
    "sasa_warning_codes": (
        "Semicolon-separated stable codes of every SASA warning, including the "
        "advisory ones that do not lower sasa_status."
    ),
    "sasa_warning_messages": (
        "Human-readable SASA diagnostic messages corresponding to warning codes."
    ),
    "sasa_normalization": "Named RASA scale: Tien2013_ALLOWED_DSSP.",
    "sasa_protein_radius_scheme": (
        "Protein atom-radius convention used for SASA: DSSP, radii by atom role."
    ),
    "sasa_nonprotein_radius_scheme": (
        "Element-radius convention used for non-protein occluders: Alvarez2013."
    ),
    "sasa_probe_radius": "Solvent-probe radius in Å: 1.4, that of water.",
    "sasa_n_sphere_points": (
        "Number of Shrake-Rupley surface sample points used per target atom."
    ),
    "sasa_accuracy_mode": "Named sphere-grid mode: fast, default, high, or custom.",
    "sasa_structure_context": (
        "Coordinate context used without implicit transforms: asymmetric_unit "
        "or as_provided."
    ),
    "sasa_include_experimental_additives": (
        "Recorded policy flag: whether recognized experimental additives were "
        "included as bound-context occluders."
    ),
    "sasa_include_unknown_nonpolymers": (
        "Recorded policy flag: whether radius-supported unknown non-polymers "
        "were included as bound-context occluders."
    ),
}

# Boolean columns are nullable, so a row the calculation did not reach holds a
# missing value rather than False. A column in neither tuple below is text.
BOOLEAN_OUTPUT_COLUMNS: tuple[str, ...] = (
    "is_buried_protein",
    "is_exposed_protein",
    "is_buried_bound",
    "is_exposed_bound",
    "has_bound_nonprotein",
    "residue_sasa_complete",
    "rasa_eligible",
    "is_terminal_residue",
    "is_occupancy_ambiguous",
    "has_experimental_additive",
    "has_unknown_nonpolymer",
    "bound_context_complete",
    "bound_context_ambiguous",
    "sasa_ok",
    "sasa_include_experimental_additives",
    "sasa_include_unknown_nonpolymers",
)

NUMERIC_OUTPUT_COLUMNS: tuple[str, ...] = (
    "atom_sasa_protein",
    "atom_sasa_bound",
    "residue_sasa_protein",
    "rasa_protein",
    "residue_sasa_bound",
    "rasa_bound",
    "nonprotein_occluded_sasa",
    "nonprotein_occluded_fraction",
    "sasa_probe_radius",
    "sasa_n_sphere_points",
)

__all__ = [
    "ALL_OUTPUT_COLUMNS",
    "ATOM_OUTPUT_COLUMNS",
    "BOOLEAN_OUTPUT_COLUMNS",
    "ML_PROFILE_COLUMNS",
    "NUMERIC_OUTPUT_COLUMNS",
    "RESIDUE_OUTPUT_COLUMNS",
    "SASA_COLUMN_DESCRIPTIONS",
    "STRUCTURE_OUTPUT_COLUMNS",
]
