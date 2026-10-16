"""The SASA of one structure, written as columns of its atom table.

compute_structure_sasa turns the atom table of one structure into the SASA output
columns. The targets are the protein heavy atoms with positive occupancy and
finite coordinates. Each model is computed on its own, so the models of an
ensemble neither bury one another nor share residues.

Each model is computed twice. In the protein context, only the targets
occlude. In the bound context, the heavy atoms of ligands, cofactors, metals,
and non-protein polymers occlude too. Unknown non-polymers also occlude
unless the user excludes them. Experimental additives occlude only if the
user includes them. Water never occludes. The difference between the two
contexts is the area a residue loses to its partners.

A requested occluder is left out of the bound context when its occupancy,
its coordinates, or its element's radius is missing or unusable. No default
stands in for the missing value. Instead, the bound context is marked
incomplete. The protein context is unaffected.

A residue's area is the sum of its atoms' areas. It becomes a relative area
only when the residue has a Tien maximum and all its expected heavy atoms,
each at full occupancy. The residue must also not be at a chain end or beside
a break, and its area must have been computed. Otherwise the relative area is
missing, rasa_ineligible_reason says why, and the raw area remains.

The status is failed when no model gave protein-context areas. It is
degraded when a model failed, when the bound context is incomplete, or when
any warning outside ADVISORY_WARNING_CODES was raised. Otherwise it is ok.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ....structure.elements import HYDROGEN_ELEMENTS
from ....structure.residue_identity import (
    ResidueIdentities,
    build_residue_identities,
    first_numeric_series,
)
from .classification import (
    component_categories,
    normalized_atom_names,
    normalized_elements,
    protein_radii,
    rasa_and_burial,
    terminal_residues,
)
from .columns import (
    ALL_OUTPUT_COLUMNS,
    BOOLEAN_OUTPUT_COLUMNS,
    NUMERIC_OUTPUT_COLUMNS,
    RESIDUE_OUTPUT_COLUMNS,
)
from .constants import (
    ALVAREZ_VDW_RADII,
    EXPECTED_HEAVY_ATOMS,
    SASA_NONPROTEIN_RADIUS_SCHEME,
    SASA_NORMALIZATION_NAME,
    SASA_PROBE_RADIUS,
    SASA_PROTEIN_RADIUS_SCHEME,
)
from .sphere import compute_sasa, resolve_sphere_point_count

# These warnings describe the surroundings a calculation ran in, not a
# shortfall in it. A partial-occupancy occluder or an unknown one makes the
# biological reading of the bound context uncertain. Every area is still
# exact, and bound_context_ambiguous already reports the doubt, so neither
# warning lowers the status.
ADVISORY_WARNING_CODES: frozenset[str] = frozenset(
    {
        "PARTIAL_OCCUPANCY_NONPROTEIN",
        "UNKNOWN_NONPOLYMER_INCLUDED",
    }
)


@dataclass(frozen=True)
class SASAWarning:
    """A warning raised while computing the SASA of a structure.

    ``code`` is a stable identifier, and ``message`` is the text for people.
    ``context`` is "protein" or "bound", and ``model`` is the model number.
    Either is None when the warning concerns the whole structure.
    """

    code: str
    message: str
    context: str | None = None
    model: str | None = None

    @property
    def degrades(self) -> bool:
        """Return whether this warning lowers the status to degraded."""
        return self.code not in ADVISORY_WARNING_CODES


@dataclass(frozen=True)
class SASAComputationResult:
    """The SASA columns of one structure, with its status and warnings.

    ``columns`` holds every output column under the index of the input table.
    ``status`` is "ok", "degraded", or "failed". ``warnings`` holds every
    warning in the order raised, advisory ones included.
    """

    columns: pd.DataFrame
    status: str
    warnings: tuple[SASAWarning, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether every context ran with no degrading warning."""
        return self.status == "ok"


def _typed_column(column: str, values: list[object], index: pd.Index) -> pd.Series:
    """Return ``values`` as a Series of the dtype ``column`` is declared with.

    None becomes NaN in a numeric column and a missing value in the others.
    """
    if column in NUMERIC_OUTPUT_COLUMNS:
        return pd.Series(values, index=index, dtype=float)
    if column in BOOLEAN_OUTPUT_COLUMNS:
        return pd.Series(values, index=index, dtype="boolean")
    return pd.Series(
        [pd.NA if value is None else value for value in values],
        index=index,
        dtype=object,
    )


def _all_missing_columns(index: pd.Index) -> pd.DataFrame:
    """Return every SASA output column under ``index``, all values missing."""
    missing: list[object] = [None] * len(index)
    return pd.DataFrame(
        {column: _typed_column(column, missing, index) for column in ALL_OUTPUT_COLUMNS}
    )


def _set_structure_columns(
    columns: pd.DataFrame, values: Mapping[str, str | float | bool | None]
) -> None:
    """Write each structure-level value onto every row of ``columns``.

    None becomes a missing value.
    """
    for column, value in values.items():
        if column in BOOLEAN_OUTPUT_COLUMNS:
            columns[column] = pd.Series(value, index=columns.index, dtype="boolean")
        else:
            columns[column] = pd.NA if value is None else value


def _area_sum(areas: np.ndarray) -> float:
    """Return the sum of the finite ``areas``, or NaN when none is finite."""
    return float(np.nansum(areas)) if np.isfinite(areas).any() else np.nan


def _residue_columns(
    identities: ResidueIdentities,
    atom_names: pd.Series,
    coordinates: np.ndarray,
    *,
    full_occupancy: np.ndarray,
    target_atoms: np.ndarray,
    valid_atoms: np.ndarray,
    atom_sasa_protein: np.ndarray,
    atom_sasa_bound: np.ndarray,
) -> pd.DataFrame:
    """Return the residue-level SASA columns under the atom table's index.

    Each protein residue's values repeat on all of its atoms, and every other
    atom holds missing values. ``atom_sasa_protein`` and ``atom_sasa_bound``
    hold each atom's areas in Å², NaN where none was computed.
    ``full_occupancy`` marks the atoms at an occupancy of 1.
    ``target_atoms`` marks the targets. ``valid_atoms`` marks
    every atom with finite coordinates and positive occupancy, and the search
    for chain breaks uses it.
    """
    # Protein residues are numbered by first appearance, and every other atom
    # gets the code -1.
    codes, _ = pd.factorize(
        identities.residue_keys.where(identities.protein_polymer.to_numpy(dtype=bool))
    )
    positions = np.flatnonzero(codes >= 0)
    grouped = positions[np.argsort(codes[positions], kind="stable")]
    residue_positions = (
        np.split(grouped, np.flatnonzero(np.diff(codes[grouped])) + 1)
        if len(grouped)
        else []
    )
    terminal = terminal_residues(
        identities, residue_positions, valid_atoms, atom_names, coordinates
    )
    names: list[str] = atom_names.tolist()
    residue_names: list[str] = identities.res_name.tolist()

    rows: list[dict[str, Any]] = []
    for residue, residue_atoms in enumerate(residue_positions):
        row: dict[str, Any] = dict.fromkeys(RESIDUE_OUTPUT_COLUMNS)
        rows.append(row)
        residue_name = residue_names[int(residue_atoms[0])]
        protein_sum = _area_sum(atom_sasa_protein[residue_atoms])
        bound_sum = _area_sum(atom_sasa_bound[residue_atoms])

        expected_atoms = EXPECTED_HEAVY_ATOMS.get(residue_name)
        observed_valid_atoms: set[str] = {
            names[position] for position in residue_atoms if target_atoms[position]
        }
        missing_atoms = (
            sorted(expected_atoms - observed_valid_atoms)
            if expected_atoms is not None
            else []
        )
        # Only the expected heavy atoms count, so an atom such as OXT cannot
        # make a residue ambiguous. An expected atom with zero occupancy makes
        # it both incomplete and ambiguous.
        is_occupancy_ambiguous = expected_atoms is not None and any(
            names[position] in expected_atoms and not full_occupancy[position]
            for position in residue_atoms
        )
        is_terminal = residue in terminal

        reasons: list[str] = []
        if expected_atoms is None:
            reasons.append("unsupported_residue")
        if missing_atoms:
            reasons.append("missing_atoms:" + ",".join(missing_atoms))
        if is_occupancy_ambiguous:
            reasons.append("partial_or_ambiguous_occupancy")
        if is_terminal:
            reasons.append("terminal_residue")
        if not np.isfinite(protein_sum):
            reasons.append("no_calculable_sasa")
        eligible = not reasons

        row["residue_sasa_protein"] = protein_sum
        row["residue_sasa_bound"] = bound_sum
        # A model's areas are computed together or not at all. A residue in a
        # model whose protein context failed has its atoms but no sum, so it
        # is not complete.
        row["residue_sasa_complete"] = (
            expected_atoms is not None
            and not missing_atoms
            and bool(np.isfinite(protein_sum))
        )
        row["rasa_eligible"] = eligible
        row["rasa_ineligible_reason"] = None if eligible else ";".join(reasons)
        row["is_terminal_residue"] = is_terminal
        row["is_occupancy_ambiguous"] = is_occupancy_ambiguous

        if np.isfinite(protein_sum) and np.isfinite(bound_sum):
            occluded = max(0.0, protein_sum - bound_sum)
            row["nonprotein_occluded_sasa"] = occluded
            row["nonprotein_occluded_fraction"] = (
                occluded / protein_sum if protein_sum > 0 else np.nan
            )
            row["has_bound_nonprotein"] = occluded > 0.0

        normalized = rasa_and_burial(residue_name, protein_sum) if eligible else None
        if normalized is None:
            continue
        (
            row["rasa_protein"],
            row["is_buried_protein"],
            row["is_exposed_protein"],
        ) = normalized
        bound_normalized = (
            rasa_and_burial(residue_name, bound_sum) if np.isfinite(bound_sum) else None
        )
        if bound_normalized is not None:
            (
                row["rasa_bound"],
                row["is_buried_bound"],
                row["is_exposed_bound"],
            ) = bound_normalized

    # One extra all-missing row serves every atom outside a protein residue.
    rows.append(dict.fromkeys(RESIDUE_OUTPUT_COLUMNS))
    table_index = pd.RangeIndex(len(rows))
    table = pd.DataFrame(
        {
            column: _typed_column(column, [row[column] for row in rows], table_index)
            for column in RESIDUE_OUTPUT_COLUMNS
        }
    )
    return table.iloc[np.where(codes >= 0, codes, len(rows) - 1)].set_axis(
        identities.residue_keys.index
    )


@dataclass
class _Outcome:
    """The status, warnings, and structure-wide flags of one structure."""

    status: str
    warnings: list[SASAWarning] = field(default_factory=list[SASAWarning])
    has_experimental_additive: bool = False
    has_unknown_nonpolymer: bool = False
    bound_context_complete: bool = False
    bound_context_ambiguous: bool = False


def compute_structure_sasa(
    atom_dataframe: pd.DataFrame,
    *,
    accuracy_mode: str = "default",
    n_sphere_points: int | None = None,
    include_experimental_additives: bool = False,
    include_unknown_nonpolymers: bool = True,
    structure_context: str = "as_provided",
) -> SASAComputationResult:
    """Return the SASA columns, status, and warnings of one structure's table.

    ``accuracy_mode`` and ``n_sphere_points`` choose the sphere grid, as in
    resolve_sphere_point_count. The two include flags decide whether
    experimental additives and unknown non-polymers occlude the bound
    context. ``structure_context`` is recorded but not applied. An empty
    table, or one with no protein atom to compute, gives a failed result
    rather than an error.

    Raises ValueError for an unknown accuracy mode or a point count that is
    not a positive integer. It also raises ValueError when the table holds a
    coordinate, an occupancy, or a flag that cannot be read.
    """
    mode, point_count = resolve_sphere_point_count(accuracy_mode, n_sphere_points)
    columns = _all_missing_columns(atom_dataframe.index)
    outcome = _fill_atom_and_residue_columns(
        columns,
        atom_dataframe,
        point_count=point_count,
        include_experimental_additives=include_experimental_additives,
        include_unknown_nonpolymers=include_unknown_nonpolymers,
    )
    warnings = outcome.warnings
    _set_structure_columns(
        columns,
        {
            "has_experimental_additive": outcome.has_experimental_additive,
            "has_unknown_nonpolymer": outcome.has_unknown_nonpolymer,
            "bound_context_complete": outcome.bound_context_complete,
            "bound_context_ambiguous": outcome.bound_context_ambiguous,
            "sasa_status": outcome.status,
            "sasa_ok": outcome.status == "ok",
            # Each code once, in the order it was first raised.
            "sasa_warning_codes": (
                ";".join(dict.fromkeys(warning.code for warning in warnings))
                if warnings
                else None
            ),
            "sasa_warning_messages": (
                " | ".join(warning.message for warning in warnings)
                if warnings
                else None
            ),
            "sasa_normalization": SASA_NORMALIZATION_NAME,
            "sasa_protein_radius_scheme": SASA_PROTEIN_RADIUS_SCHEME,
            "sasa_nonprotein_radius_scheme": SASA_NONPROTEIN_RADIUS_SCHEME,
            "sasa_probe_radius": SASA_PROBE_RADIUS,
            "sasa_n_sphere_points": point_count,
            "sasa_accuracy_mode": mode,
            "sasa_structure_context": structure_context,
            "sasa_include_experimental_additives": include_experimental_additives,
            "sasa_include_unknown_nonpolymers": include_unknown_nonpolymers,
        },
    )
    return SASAComputationResult(columns, outcome.status, tuple(warnings))


def _fill_atom_and_residue_columns(
    columns: pd.DataFrame,
    atom_dataframe: pd.DataFrame,
    *,
    point_count: int,
    include_experimental_additives: bool,
    include_unknown_nonpolymers: bool,
) -> _Outcome:
    """Compute every model and write the atom and residue columns of ``columns``.

    Returns the status, the warnings in the order raised, and the
    structure-wide flags that compute_structure_sasa records.
    """
    if len(atom_dataframe) == 0:
        return _Outcome(
            "failed",
            [
                SASAWarning(
                    "EMPTY_INPUT",
                    "No coordinate rows were supplied to the SASA engine.",
                )
            ],
        )

    coordinates = np.column_stack(
        [
            first_numeric_series(
                atom_dataframe, (axis, f"_atom_site.Cartn_{axis}")
            ).to_numpy(dtype=float)
            for axis in "xyz"
        ]
    )
    valid_coordinates = np.isfinite(coordinates).all(axis=1)

    identities = build_residue_identities(atom_dataframe)
    elements = normalized_elements(atom_dataframe)
    atom_names = normalized_atom_names(atom_dataframe)
    categories = component_categories(identities, elements)
    columns["sasa_component_category"] = categories

    # Alternate conformers need no handling here, because the parser keeps one
    # conformer of each residue. A table with no occupancy column is read as
    # fully occupied.
    occupancy = first_numeric_series(
        atom_dataframe,
        ("occupancy", "_atom_site.occupancy"),
        default_if_absent=1.0,
    ).to_numpy(dtype=float)
    positive_occupancy = np.isfinite(occupancy) & (occupancy > 0.0)
    zero_occupancy = np.isfinite(occupancy) & ~positive_occupancy
    full_occupancy = np.isclose(occupancy, 1.0, atol=1.0e-6)
    # Hydrogens are neither targets nor occluders, because the DSSP radii and
    # the Tien maxima are for heavy atoms. An atom with no element is judged by
    # its name.
    is_hydrogen = elements.isin(HYDROGEN_ELEMENTS) | (
        elements.eq("") & atom_names.str.startswith(("H", "D"))
    )
    heavy = ~is_hydrogen.to_numpy(dtype=bool)
    protein_polymer = identities.protein_polymer.to_numpy(dtype=bool)
    target_atoms = valid_coordinates & positive_occupancy & heavy & protein_polymer

    category_array = categories.to_numpy(dtype=object)
    outcome = _Outcome(
        "failed",
        has_experimental_additive=bool(
            (category_array == "experimental_additive").any()
        ),
        has_unknown_nonpolymer=bool((category_array == "unknown_nonpolymer").any()),
    )
    warnings = outcome.warnings

    if not target_atoms.any():
        # No model has a target, so no bound context was built and no partner
        # was included in one. The main path likewise skips a model without
        # targets before it looks at that model's partners, so nothing here
        # makes the bound context ambiguous.
        warnings.append(
            SASAWarning(
                "NO_VALID_PROTEIN_ATOMS",
                "No positive-occupancy protein heavy atoms with finite coordinates were available.",
                context="protein",
            )
        )
        return outcome

    invalid_target_identity = identities.residue_keys.isna().to_numpy() & target_atoms
    if invalid_target_identity.any():
        warnings.append(
            SASAWarning(
                "INVALID_PROTEIN_RESIDUE_IDENTITY",
                f"{int(invalid_target_identity.sum())} calculable protein atom "
                "row(s) lack a canonical residue identity; atom SASA is retained "
                "but residue SASA/RASA is unavailable for those rows.",
                context="protein",
            )
        )

    occluder_categories = ["ligand_or_cofactor", "metal", "nonprotein_polymer"]
    if include_unknown_nonpolymers:
        occluder_categories.append("unknown_nonpolymer")
    if include_experimental_additives:
        occluder_categories.append("experimental_additive")
    requested_occluders = heavy & np.isin(category_array, occluder_categories)
    unknown = category_array == "unknown_nonpolymer"
    # NaN for an element with no Alvarez radius, the blank element included.
    occluder_radii = np.asarray(
        [ALVAREZ_VDW_RADII.get(element, np.nan) for element in elements],
        dtype=float,
    )
    target_radii = protein_radii(atom_names)

    atom_sasa_protein = np.full(len(atom_dataframe), np.nan, dtype=float)
    atom_sasa_bound = np.full(len(atom_dataframe), np.nan, dtype=float)
    any_protein_context_computed = False
    bound_context_complete = True
    bound_context_ambiguous = False

    pdb_values = identities.pdb_id.to_numpy(dtype=object)
    model_values = identities.model.to_numpy(dtype=object)
    for pdb_value, model in dict.fromkeys(zip(pdb_values, model_values, strict=True)):
        model_mask = (pdb_values == pdb_value) & (model_values == model)
        target_positions = np.flatnonzero(model_mask & target_atoms)
        if len(target_positions) == 0:
            # A model with no protein at all has nothing to compute, which is
            # not a failure.
            if (model_mask & protein_polymer).any():
                bound_context_complete = False
                structure_label = f" in structure {pdb_value}" if pdb_value else ""
                warnings.append(
                    SASAWarning(
                        "NO_VALID_PROTEIN_ATOMS_IN_MODEL",
                        f"Model {model}{structure_label} contains protein rows "
                        "but no selected positive-occupancy heavy atoms with "
                        "finite coordinates.",
                        context="protein",
                        model=model,
                    )
                )
            continue
        try:
            protein_areas = compute_sasa(
                coordinates[target_positions],
                target_radii[target_positions],
                n_sphere_points=point_count,
            )
        except Exception as exception:
            # One failed model is reported, and the other models still run.
            bound_context_complete = False
            warnings.append(
                SASAWarning(
                    "PROTEIN_MODEL_FAILED",
                    f"Protein-context SASA failed for model {model}: {exception}",
                    context="protein",
                    model=model,
                )
            )
            continue
        atom_sasa_protein[target_positions] = protein_areas
        any_protein_context_computed = True

        requested = model_mask & requested_occluders
        # An occluder with zero occupancy is absent by the model's own account,
        # so it is left out without a warning. A missing occupancy says nothing
        # either way, so leaving that atom out is reported.
        unusable_occupancy = requested & ~np.isfinite(occupancy)
        unusable_coordinates = requested & positive_occupancy & ~valid_coordinates
        usable = requested & positive_occupancy & valid_coordinates
        unsupported = usable & np.isnan(occluder_radii)
        included = usable & ~unsupported
        if unusable_occupancy.any():
            bound_context_complete = False
            warnings.append(
                SASAWarning(
                    "UNUSABLE_NONPROTEIN_OCCUPANCY",
                    f"Bound context excluded {int(unusable_occupancy.sum())} "
                    f"requested non-protein atom(s) in model {model} because "
                    "occupancy was missing or non-finite.",
                    context="bound",
                    model=model,
                )
            )
        if unusable_coordinates.any():
            bound_context_complete = False
            warnings.append(
                SASAWarning(
                    "UNUSABLE_NONPROTEIN_COORDINATES",
                    f"Bound context excluded {int(unusable_coordinates.sum())} "
                    f"requested non-protein atom(s) in model {model} because "
                    "coordinates were missing or non-finite.",
                    context="bound",
                    model=model,
                )
            )
        if unsupported.any():
            bound_context_complete = False
            unsupported_elements = {
                elements.iloc[position] or "<blank>"
                for position in np.flatnonzero(unsupported)
            }
            warnings.append(
                SASAWarning(
                    "UNSUPPORTED_NONPROTEIN_ELEMENT",
                    "Bound context excluded unsupported non-protein element(s): "
                    + ", ".join(sorted(unsupported_elements)),
                    context="bound",
                    model=model,
                )
            )
        if (included & ~full_occupancy).any():
            bound_context_ambiguous = True
            warnings.append(
                SASAWarning(
                    "PARTIAL_OCCUPANCY_NONPROTEIN",
                    f"Model {model} contains a selected partial-occupancy non-protein occluder.",
                    context="bound",
                    model=model,
                )
            )
        # A requested unknown partner makes the bound context ambiguous unless
        # its zero occupancy declares it absent, even when it is left out for
        # another reason.
        if (requested & unknown & ~zero_occupancy).any():
            bound_context_ambiguous = True
            warnings.append(
                SASAWarning(
                    "UNKNOWN_NONPOLYMER_INCLUDED",
                    f"Model {model} includes unclassified non-polymer occluders.",
                    context="bound",
                    model=model,
                )
            )

        # With no occluder beyond the protein, the bound context is the protein
        # context.
        if not included.any():
            atom_sasa_bound[target_positions] = protein_areas
            continue
        # The targets come first, so each keeps its place in target_positions
        # as its index among the occluders.
        occluder_positions = np.concatenate(
            (target_positions, np.flatnonzero(included))
        )
        try:
            atom_sasa_bound[target_positions] = compute_sasa(
                coordinates[occluder_positions],
                np.concatenate(
                    (target_radii[target_positions], occluder_radii[included])
                ),
                n_sphere_points=point_count,
                target_indices=np.arange(len(target_positions)),
            )
        except Exception as exception:
            bound_context_complete = False
            warnings.append(
                SASAWarning(
                    "BOUND_MODEL_FAILED",
                    f"Bound-context SASA failed for model {model}: {exception}",
                    context="bound",
                    model=model,
                )
            )

    columns["atom_sasa_protein"] = atom_sasa_protein
    columns["atom_sasa_bound"] = atom_sasa_bound
    residues = _residue_columns(
        identities,
        atom_names,
        coordinates,
        full_occupancy=full_occupancy,
        target_atoms=target_atoms,
        valid_atoms=valid_coordinates & positive_occupancy,
        atom_sasa_protein=atom_sasa_protein,
        atom_sasa_bound=atom_sasa_bound,
    )
    for column in RESIDUE_OUTPUT_COLUMNS:
        columns[column] = residues[column]

    # Every failed model and every gap in the bound context raised a
    # degrading warning, so the warnings alone decide between ok and degraded.
    if not any_protein_context_computed:
        outcome.status = "failed"
    elif any(warning.degrades for warning in warnings):
        outcome.status = "degraded"
    else:
        outcome.status = "ok"
    outcome.bound_context_complete = bound_context_complete
    outcome.bound_context_ambiguous = bound_context_ambiguous
    return outcome


__all__ = [
    "ADVISORY_WARNING_CODES",
    "SASAComputationResult",
    "SASAWarning",
    "compute_structure_sasa",
]
